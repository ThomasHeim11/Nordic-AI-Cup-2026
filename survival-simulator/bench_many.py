"""Parallel multi-seed benchmark for A/B testing policy variants.

    python bench_many.py --seeds 201-224 --workers 4 --set disperse_seconds=0 --tag base

Prints mean/stdev/min plus starvation & predation totals; appends a JSON line
to bench_many.jsonl so runs can be compared later.
"""
import argparse, json, os, statistics, sys, time
from multiprocessing import Pool
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
HERE = os.path.dirname(os.path.abspath(__file__))


def _worker(args):
    overrides, seed, policy_name = args
    sys.path.insert(0, HERE)
    import importlib
    import bench
    hp = importlib.import_module(policy_name)
    if not getattr(bench, "_patched", False):      # once per worker process
        bench.patch_fast_biome_render()
        bench.patch_death_counter()
        bench._patched = True
    if hasattr(hp, "PARAMS"):
        hp.PARAMS.update(overrides)
    r = bench.run_one(hp, seed)
    return r


def parse_seeds(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="201-224")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--tag", default="")
    ap.add_argument("--policy", default="src.utils.controllers.hivemind_policy",
                    help="policy module, e.g. rl.policy_nn (weights via SURVIVAL_NN=...)")
    a = ap.parse_args()
    overrides = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        overrides[k] = float(v)
    seeds = parse_seeds(a.seeds)
    t0 = time.time()
    with Pool(a.workers) as pool:
        results = pool.map(_worker, [(overrides, s, a.policy) for s in seeds], chunksize=1)
    scores = [r["score"] for r in results]
    summary = {
        "tag": a.tag, "policy": a.policy, "weights": os.environ.get("SURVIVAL_NN", ""),
        "overrides": overrides, "n": len(scores),
        "mean": statistics.mean(scores), "stdev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        "min": min(scores), "max": max(scores),
        "starved": sum(r["starved"] for r in results), "eaten": sum(r["eaten"] for r in results),
        "peak_pop": statistics.mean(r["peak_pop"] for r in results),
        "seeds": a.seeds, "elapsed": time.time() - t0,
    }
    with open(os.path.join(HERE, "bench_many.jsonl"), "a") as f:
        f.write(json.dumps(summary) + "\n")
    print(f"[{a.tag}] n={summary['n']} mean={summary['mean']:.1f} sd={summary['stdev']:.1f} "
          f"min={summary['min']:.0f} max={summary['max']:.0f} starved={summary['starved']} "
          f"eaten={summary['eaten']} peak={summary['peak_pop']:.1f} overrides={overrides} ({summary['elapsed']:.0f}s)")


if __name__ == "__main__":
    main()
