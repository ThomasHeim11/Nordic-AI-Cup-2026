"""(mu + lambda) evolution strategy over hivemind_policy.PARAMS.

    python evolve.py --gens 8 --pop 12 --seeds-per-gen 4 --workers 8

Fitness is the mean simulator score over a handful of seeds; the seed set is
re-drawn every generation so the genome cannot overfit a particular map.  The
best-so-far genome is written to src/utils/controllers/hivemind_params.json
after every generation, and hivemind_policy loads that file on import.
"""
import argparse, json, os, random, statistics, sys, time
from multiprocessing import Pool

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "src", "utils", "controllers", "hivemind_params.json")
LOG = os.path.join(HERE, "evolve_log.jsonl")

# (low, high) search bounds.  Anything not listed keeps its PARAMS default.
BOUNDS = {
    "danger_dist":         (80.0, 140.0),
    "aware_dist":          (150.0, 400.0),
    "retreat_speed_frac":  (0.4, 1.0),
    "fruit_seek":          (100.0, 400.0),
    "fruit_full_frac":     (0.6, 1.0),
    "tree_seek":           (150.0, 400.0),
    "tree_camp":           (30.0, 120.0),
    "scan_turn":           (0.0, 0.5),
    "crowd_radius":        (30.0, 150.0),
    "crowd_limit":         (1.0, 8.0),
    "separation":          (10.0, 120.0),
    "separation_gain":     (0.0, 2.0),
    "wall_margin":         (15.0, 80.0),
    "move_deadzone":       (0.05, 0.6),
    "wander_jitter":       (0.05, 0.8),
    "turn_gain":           (0.1, 1.0),
    "max_turn":            (0.2, 1.5),
    "spawn_energy":        (120.0, 450.0),
    "spawn_energy_low":    (105.0, 250.0),
    "spawn_min_pop":       (2.0, 12.0),
    "spawn_max_pop":       (6.0, 40.0),
    "spawn_elite_frac":    (0.1, 1.0),
    "spawn_predator_clear":(50.0, 400.0),
    "elder_age":           (40.0, 120.0),
    "elder_spawn_energy":  (102.0, 250.0),
    "elder_pop_slack":     (0.8, 1.6),
    "food_gate":           (0.0, 4.0),
    "food_gate_tree_w":    (0.0, 1.5),
    "memory_ttl":          (30.0, 200.0),
    "memory_visit_radius": (20.0, 90.0),
    "memory_revisit_after":(5.0, 80.0),
    "hungry_frac":         (0.2, 0.9),
    "late_pop":            (2.0, 14.0),
    "late_t0":             (200.0, 900.0),
    "late_t1":             (600.0, 2000.0),
}
INT_KEYS = {"crowd_limit", "spawn_min_pop", "spawn_max_pop"}


def _worker(args):
    params, seed = args
    import bench
    from src.utils.controllers import hivemind_policy as hp
    bench.patch_fast_biome_render()
    bench.patch_death_counter()
    hp.PARAMS.update(params)
    r = bench.run_one(hp, seed)
    return r["score"]


def clamp(genome):
    out = {}
    for k, (lo, hi) in BOUNDS.items():
        v = min(hi, max(lo, genome[k]))
        out[k] = float(round(v)) if k in INT_KEYS else v
    return out


def mutate(genome, rng, sigma):
    child = dict(genome)
    for k, (lo, hi) in BOUNDS.items():
        if rng.random() < 0.35:
            child[k] += rng.gauss(0.0, sigma * (hi - lo))
    return clamp(child)


def crossover(a, b, rng):
    return clamp({k: (a[k] if rng.random() < 0.5 else b[k]) for k in BOUNDS})


def evaluate(pool, genomes, seeds):
    jobs = [(g, s) for g in genomes for s in seeds]
    scores = pool.map(_worker, jobs, chunksize=1)
    per = []
    n = len(seeds)
    for i in range(len(genomes)):
        chunk = scores[i * n:(i + 1) * n]
        per.append((statistics.mean(chunk), min(chunk)))
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=8)
    ap.add_argument("--pop", type=int, default=12)
    ap.add_argument("--elite", type=int, default=4)
    ap.add_argument("--seeds-per-gen", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--rng-seed", type=int, default=42)
    ap.add_argument("--out", default=OUT, help="where the best-so-far genome is written (default: the live params file)")
    a = ap.parse_args()
    global OUT
    OUT = a.out

    from src.utils.controllers import hivemind_policy as hp
    base = clamp({k: hp.PARAMS[k] for k in BOUNDS})
    rng = random.Random(a.rng_seed)

    population = [base] + [mutate(base, rng, a.sigma) for _ in range(a.pop - 1)]
    best = (float("-inf"), base)
    t0 = time.time()
    with Pool(a.workers) as pool:
        for gen in range(a.gens):
            seeds = [rng.randint(1, 10**6) for _ in range(a.seeds_per_gen)]
            fitness = evaluate(pool, population, seeds)
            ranked = sorted(zip(fitness, population), key=lambda x: x[0][0], reverse=True)
            (top_mean, top_min), top_genome = ranked[0]
            if top_mean > best[0]:
                best = (top_mean, top_genome)
                with open(OUT, "w") as f:
                    json.dump({"score": top_mean, "gen": gen, "seeds": seeds, "params": top_genome}, f, indent=2)
            with open(LOG, "a") as f:
                f.write(json.dumps({"gen": gen, "seeds": seeds, "elapsed": time.time() - t0,
                                    "ranked": [(m, mn) for (m, mn), _ in ranked],
                                    "best_gen": top_genome}) + "\n")
            print(f"gen {gen:2d} | best {top_mean:8.1f} (min {top_min:7.1f}) | "
                  f"gen mean {statistics.mean(m for (m, _), _ in ranked):8.1f} | "
                  f"all-time {best[0]:8.1f} | {time.time() - t0:6.0f}s", flush=True)

            elites = [g for _, g in ranked[:a.elite]]
            children = []
            while len(children) < a.pop - a.elite:
                pa, pb = rng.sample(elites, 2)
                children.append(mutate(crossover(pa, pb, rng), rng, a.sigma))
            population = elites + children

    print("best genome:", json.dumps(best[1], indent=2))


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    main()
