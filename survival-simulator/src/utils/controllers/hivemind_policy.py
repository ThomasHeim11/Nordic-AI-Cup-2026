"""Hivemind controller for the survival simulator.

Design notes (all verified against the simulator source, not the README):

* ``move_direction`` is RELATIVE to the agent's heading -- ``environment.py``
  does ``direction = entity.direction + direction``.  Heading and travel
  direction are therefore independent, and turning is only ever about where the
  vision cone points.
* Energy cost of movement is linear in distance (0.05/unit walking) with no
  fixed component, while living costs 0.1 per tick regardless.  Ground covered
  per unit of cost is therefore maximised by always walking at full ``speed``;
  standing still is strictly wasteful unless fruit is about to spawn next to us.
* A predator only charges when the agent is NOT looking at it
  (``abs(agent_looking_dir) > pi/2``) or when it is already within
  ``hearing_radius * 1.5 == 90``.  Keeping our heading pointed at the nearest
  predator downgrades it to a slow pivot that burns its (small, 200) energy
  pool, and we can retreat at cheap walking speed while doing so.
* Sprinting away costs 5.5/tick and only nets 5 units/tick on a predator, so
  escaping a committed charge costs ~180 energy.  Never being inside 90 units is
  an order of magnitude cheaper than escaping.
* Spawning converts 100 parent energy into a 75-energy body: a 75% efficient
  way to turn energy into population, and population is what keeps the run
  alive.  Traits mutate +-50% with 10% probability per trait on each spawn, so
  choosing *which* agents breed is directed evolution within a single run.
"""

import json
import math
import os
import random
from typing import Dict, List, Sequence, Tuple

from src.utils.DTOs import ActionRequest

TAU = 2.0 * math.pi

# Trait ceilings enforced by Environment.spawn_agent, used to normalise traits.
MAX_SPEED = 20.0
MAX_SPRINT = 40.0
MAX_MAX_ENERGY = 1000.0
MAX_HEARING = 100.0          # chunk_size / 4
MAX_VISION = 400.0           # chunk_size
MAX_CONE = math.pi / 2

# Biomes we would rather not be standing in: you pay energy for the distance you
# request but only travel distance * move_penalty.
BIOME_MOVE_PENALTY = {
    "forest": 1.0,
    "grassland": 1.0,
    "swamp": 0.5,
    "desert": 0.8,
    "river": 0.3,
}

PARAMS: Dict[str, float] = {
    # --- predators ---
    "danger_dist": 105.0,        # inside this a charge is coming: sprint out
    "aware_dist": 280.0,         # react to predators at all up to here
    "retreat_speed_frac": 1.0,   # walking-speed fraction used while retreating
    # --- foraging ---
    "fruit_seek": 260.0,         # chase a fruit up to this distance
    "fruit_full_frac": 0.92,     # stop foraging above this fraction of max energy
    "tree_seek": 420.0,
    "tree_camp": 60.0,           # within this of a tree we stop moving and wait
    "scan_turn": 0.22,           # idle scan rate (rad/tick); buys predator warning
    "crowd_radius": 70.0,        # too many neighbours here -> go find another tree
    "crowd_limit": 3,
    # --- flocking / exploration ---
    "separation": 55.0,
    "separation_gain": 0.9,
    "wall_margin": 40.0,
    "move_deadzone": 0.25,
    "wander_jitter": 0.30,
    "turn_gain": 0.55,
    "max_turn": 0.75,
    # --- breeding ---
    "spawn_energy": 260.0,
    "spawn_energy_low": 140.0,   # threshold used while the population is critical
    "spawn_min_pop": 6,          # at or below this, anyone who can afford it breeds
    "spawn_max_pop": 14,         # above this, only elders breed
    "spawn_elite_frac": 0.45,    # otherwise only the top fraction breeds
    "spawn_predator_clear": 220.0,
    "elder_age": 58.0,           # ageing drain (0.01*age/tick) starts at 60-120 s
    "elder_spawn_energy": 108.0, # elders turn every spare 100 energy into a child
    "elder_pop_slack": 1.0,      # elders may breed while pop < max_pop * this
    # --- food-gated breeding (anti boom-bust) ---
    "food_gate": 1.2,            # min (fruit obs + tree_w * tree obs) per agent to allow breeding
    "food_gate_tree_w": 0.5,
    # --- tree memory (dead reckoning in the agent's own frame) ---
    "memory_ttl": 110.0,         # seconds a remembered tree stays valid (trees die at 50-100 s)
    "memory_visit_radius": 45.0, # within this of a remembered tree counts as visited
    "memory_revisit_after": 25.0,# seconds before a visited tree is worth another look
    # --- dispersal / barren patch ---
    "disperse_seconds": 0.0,     # newborn dispersal walk; measured -60 on 24 seeds -> off
    "barren_seconds": 1e6,       # leave a fruitless tree after this long; measured harmful -> off
}

# Evolved overrides: evolve.py writes the best genome here and the server picks
# it up on import.  Bounds for the evolver live in evolve.py, not here.
PARAMS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hivemind_params.json")


def load_params(path: str = PARAMS_FILE) -> bool:
    """Overlay PARAMS with the JSON file if it exists.  Returns True if loaded."""
    if not os.path.exists(path):
        return False
    with open(path) as f:
        data = json.load(f)
    PARAMS.update({k: float(v) for k, v in data.get("params", data).items() if k in PARAMS})
    return True


load_params()

# Per-agent scratch memory (wander heading).  Keyed by agent_id.
_MEMORY: Dict[int, Dict[str, float]] = {}
_RNG = random.Random(0xC0FFEE)


def reset() -> None:
    """Drop all per-run state.  Call between simulations."""
    _MEMORY.clear()


def _wrap(angle: float) -> float:
    return (angle + math.pi) % TAU - math.pi


def trait_score(state: dict) -> float:
    """Heritable fitness used to decide who is allowed to breed.

    Sensing is weighted hardest: hearing is omnidirectional and a hearing radius
    at the 100 ceiling exceeds the predator's 90-unit charge trigger, meaning
    such an agent always gets warning before a charge can start.
    """
    return (
        2.0 * min(state["hearing_radius"] / MAX_HEARING, 1.0)
        + 1.0 * min(state["vision_range"] / MAX_VISION, 1.0)
        + 1.0 * min(state["vision_angle"] / MAX_CONE, 1.0)
        + 1.5 * min(state["max_energy"] / MAX_MAX_ENERGY, 1.0)
        + 0.5 * min(state["speed"] / MAX_SPEED, 1.0)
        + 0.5 * min(state["sprint_speed"] / MAX_SPRINT, 1.0)
    )


def _closest_point_on_edge(coords) -> Tuple[float, float]:
    """Closest point of a local-frame edge segment to the agent (the origin)."""
    (x1, y1), (x2, y2) = coords
    dx, dy = x2 - x1, y2 - y1
    denom = dx * dx + dy * dy
    if denom <= 1e-9:
        return x1, y1
    t = -(x1 * dx + y1 * dy) / denom
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return x1 + t * dx, y1 + t * dy


def _split(observations: Sequence[dict]):
    fruits, trees, mates, predators, edges = [], [], [], [], []
    for o in observations:
        kind = o.get("type")
        if kind == "Fruit":
            fruits.append(o)
        elif kind == "Tree":
            trees.append(o)
        elif kind == "Agent":
            mates.append(o)
        elif kind == "Predator":
            predators.append(o)
        elif kind == "Edge":
            edges.append(o)
    return fruits, trees, mates, predators, edges


def _slide_along_walls(vx: float, vy: float, edges: Sequence[dict], margin: float) -> Tuple[float, float]:
    """Remove the part of a desired velocity that points into a nearby wall.

    Pure repulsion makes agents jitter in place next to obstacles (paying the
    full walking cost for zero progress); sliding keeps the tangential component
    so they flow around corners instead.
    """
    for e in edges:
        cx, cy = _closest_point_on_edge(e["coords"])
        d = math.hypot(cx, cy)
        if d >= margin or d < 1e-6:
            continue
        nx, ny = cx / d, cy / d               # unit vector from agent towards wall
        into = vx * nx + vy * ny
        if into > 0.0:
            k = 1.0 - d / margin              # 0 at margin, 1 when touching
            vx -= into * nx * k
            vy -= into * ny * k
            if d < margin * 0.35:             # nearly touching: back off a little
                vx -= nx * 0.4 * k
                vy -= ny * 0.4 * k
    return vx, vy


def _separation(mates: Sequence[dict], radius: float, gain: float) -> Tuple[float, float]:
    px = py = 0.0
    for m in mates:
        d = m["distance"]
        if d < radius and d > 1e-6:
            w = gain * (1.0 - d / radius)
            a = m["angle"]
            px -= math.cos(a) * w
            py -= math.sin(a) * w
    return px, py


def _remember_trees(mem: dict, trees: Sequence[dict], p: Dict[str, float]) -> None:
    """Add visible trees to the agent's own-frame map (deduped, with a timestamp)."""
    now = mem["t"]
    mem["trees"] = [t for t in mem["trees"] if now - t["seen"] < p["memory_ttl"]]
    for o in trees:
        ang = mem["heading"] + o["angle"]
        tx = mem["x"] + o["distance"] * math.cos(ang)
        ty = mem["y"] + o["distance"] * math.sin(ang)
        for t in mem["trees"]:
            if math.hypot(t["x"] - tx, t["y"] - ty) < 30.0:
                t["x"], t["y"], t["seen"] = tx, ty, now
                break
        else:
            mem["trees"].append({"x": tx, "y": ty, "seen": now, "visited": -1e9})
    # mark trees we are standing next to as visited
    for t in mem["trees"]:
        if math.hypot(t["x"] - mem["x"], t["y"] - mem["y"]) < p["memory_visit_radius"]:
            t["visited"] = now


def _remembered_target(mem: dict, p: Dict[str, float]):
    """Relative angle to the nearest remembered tree worth a visit, or None."""
    now = mem["t"]
    best, best_d = None, 1e9
    for t in mem["trees"]:
        if now - t["visited"] < p["memory_revisit_after"]:
            continue
        d = math.hypot(t["x"] - mem["x"], t["y"] - mem["y"])
        if d < best_d:
            best, best_d = t, d
    if best is None or best_d < 5.0:
        return None
    return _wrap(math.atan2(best["y"] - mem["y"], best["x"] - mem["x"]) - mem["heading"])


def _advance_pose(mem: dict, biome: str, move_distance: float, move_direction: float, turn: float) -> None:
    """Dead-reckon our own pose from the command we are about to send.

    The simulator moves first (in the old heading + move_direction), then turns.
    Movement is scaled by the biome's move penalty, which we know from the
    biome name; obstacle deflections are not observable and are ignored.
    """
    dist = move_distance * BIOME_MOVE_PENALTY.get(biome, 1.0)
    ang = mem["heading"] + move_direction
    mem["x"] += dist * math.cos(ang)
    mem["y"] += dist * math.sin(ang)
    mem["heading"] = _wrap(mem["heading"] + turn)
    mem["t"] += 0.1


def decide(state: dict, may_spawn: bool, rng: random.Random, spawn_energy: float = None,
           elder_may_spawn: bool = True) -> ActionRequest:
    """Choose one agent's action for this tick.

    ``may_spawn`` / ``spawn_energy`` govern normal breeding.  ``elder_may_spawn``
    lets an agent past ``elder_age`` breed on the cheaper elder threshold: its
    energy is about to be burnt by the ageing drain anyway, so a 75-energy child
    is the best thing it can buy with it.
    """
    agent_id = state["agent_id"]
    age = state["age"]
    if spawn_energy is None:
        spawn_energy = PARAMS["spawn_energy"]
    energy = state["energy"]
    max_energy = state["max_energy"]
    speed = state["speed"]
    sprint_speed = state["sprint_speed"]

    fruits, trees, mates, predators, edges = _split(state["observations"])
    mem = _MEMORY.get(agent_id)
    if mem is None:
        mem = _MEMORY[agent_id] = {"wander": rng.uniform(-math.pi, math.pi),
                                   "x": 0.0, "y": 0.0, "heading": 0.0, "t": 0.0, "trees": [],
                                   "disperse_until": -1.0, "last_fruit_t": 0.0, "leave_until": -1.0}
        if state.get("age", 99.0) < 0.5:
            # Newborn: pick a direction away from the nearest mate (the parent)
            # and commit to it, so the lineage spreads instead of piling up.
            nearest = min(mates, key=lambda o: o["distance"]) if mates else None
            away = _wrap(nearest["angle"] + math.pi) if nearest else rng.uniform(-math.pi, math.pi)
            mem["wander"] = away
            mem["disperse_until"] = PARAMS["disperse_seconds"]
    p = PARAMS
    _remember_trees(mem, trees, p)
    if fruits:
        mem["last_fruit_t"] = mem["t"]
    dispersing = mem["t"] < mem["disperse_until"]
    leaving = mem["t"] < mem["leave_until"]

    # ------------------------------------------------------------------ threat
    nearest_pred = min(predators, key=lambda o: o["distance"]) if predators else None
    spawn = False
    face_angle = None

    if nearest_pred is not None and nearest_pred["distance"] <= p["aware_dist"]:
        # Retreat vector: sum of unit vectors pointing away from every threat,
        # weighted by proximity.
        rx = ry = 0.0
        for o in predators:
            d = max(o["distance"], 1e-6)
            if d > p["aware_dist"]:
                continue
            w = p["aware_dist"] / d
            rx -= math.cos(o["angle"]) * w
            ry -= math.sin(o["angle"]) * w
        if abs(rx) < 1e-9 and abs(ry) < 1e-9:
            rx, ry = -math.cos(nearest_pred["angle"]), -math.sin(nearest_pred["angle"])
        rx, ry = _slide_along_walls(rx, ry, edges, p["wall_margin"])
        if abs(rx) < 1e-9 and abs(ry) < 1e-9:
            rx, ry = -math.cos(nearest_pred["angle"]), -math.sin(nearest_pred["angle"])
        move_direction = math.atan2(ry, rx)

        if nearest_pred["distance"] < p["danger_dist"]:
            # A charge is either underway or one tick away.  Max sprint is the
            # cheapest escape per unit of ground gained.  Below 20% energy the
            # simulator silently caps us at walking speed anyway.
            move_distance = sprint_speed if energy > max_energy / 5 else speed
        else:
            move_distance = speed * p["retreat_speed_frac"]

        # Point the heading at the nearest predator: while it sees us looking at
        # it and we are outside 90 units it pivots instead of charging.
        face_angle = nearest_pred["angle"]
        turn_override = None
        mem["wander"] = _wrap(move_direction)
    else:
        # ------------------------------------------------------------- foraging
        #
        # The economics: standing still costs 0.1/tick (1/s), walking at full
        # speed costs 0.5/tick (5/s) on top of it, and a tree emits a fruit worth
        # 20 (fresh) to 60 (ripe after 20 s) roughly every 10 s inside a 20-60 px
        # ring.  So camping a tree is a ~2-6 energy/s income against a 1/s cost,
        # while roaming blind early -- when the map holds ~30 fruits -- is barely
        # break-even.  Move only towards something we can actually see.
        tx = ty = 0.0
        target = None
        idle = False

        if fruits and energy < max_energy * p["fruit_full_frac"]:
            f = min(fruits, key=lambda o: o["distance"])
            if f["distance"] <= p["fruit_seek"]:
                target = f["angle"]
                tx, ty = math.cos(target) * 2.0, math.sin(target) * 2.0

        crowded = sum(1 for m in mates if m["distance"] < p["crowd_radius"]) >= p["crowd_limit"]

        if target is None and trees and not dispersing and not leaving:
            t = min(trees, key=lambda o: o["distance"])
            if t["distance"] <= p["tree_seek"]:
                if t["distance"] > p["tree_camp"] or crowded:
                    # Walk to the fruit factory (or away from an over-subscribed
                    # one, via the separation term below).
                    target = t["angle"]
                    tx, ty = math.cos(target) * 1.5, math.sin(target) * 1.5
                elif mem["t"] - mem["last_fruit_t"] > p["barren_seconds"]:
                    # This patch is not feeding us: commit to leaving it.
                    mem["leave_until"] = mem["t"] + p["disperse_seconds"]
                    mem["last_fruit_t"] = mem["t"]
                    mem["wander"] = _wrap(t["angle"] + math.pi)
                    leaving = True
                else:
                    # Sit still and let fruit accumulate and ripen next to us.
                    # Fruit gains 2 energy/s up to 60 and only rots at 50 s, so
                    # waiting is worth up to 3x per fruit and costs one fifth of
                    # what walking would.
                    idle = True

        if target is None and not idle:
            remembered = None if (dispersing or leaving) else _remembered_target(mem, p)
            if dispersing or leaving:
                tx, ty = math.cos(mem["wander"]), math.sin(mem["wander"])   # straight line, no jitter
            elif remembered is not None:
                target = remembered
                tx, ty = math.cos(remembered) * 1.3, math.sin(remembered) * 1.3
            else:
                mem["wander"] = _wrap(mem["wander"] + rng.uniform(-p["wander_jitter"], p["wander_jitter"]))
                tx, ty = math.cos(mem["wander"]), math.sin(mem["wander"])

        sx, sy = _separation(mates, p["separation"], p["separation_gain"])
        vx, vy = tx + sx, ty + sy
        if idle and abs(vx) < 1e-9 and abs(vy) < 1e-9:
            vx = vy = 0.0
        vx, vy = _slide_along_walls(vx, vy, edges, p["wall_margin"])
        mag = math.hypot(vx, vy)

        if mag < p["move_deadzone"]:
            # Nothing worth walking for: park.  Sweep the vision cone while
            # parked -- hearing only reaches 50 units and a predator commits to
            # a charge at 90, so the cone is our only early warning.
            move_direction = 0.0
            move_distance = 0.0
            face_angle = None
            turn_override = p["scan_turn"]
        else:
            move_direction = math.atan2(vy, vx)
            # Scale the step down when the steer is weak (e.g. squeezed against
            # a wall) so we do not pay full price for a shuffle.
            move_distance = speed * min(1.0, mag)
            face_angle = move_direction
            turn_override = None
            if target is None:
                mem["wander"] = _wrap(move_direction)

        # Breed only when it is safe and there is food in sight: a child is
        # born with 75 energy next to its parent, so it inherits the parent's
        # larder as well as its traits.
        safe = nearest_pred is None or nearest_pred["distance"] > p["spawn_predator_clear"]
        if safe and may_spawn and energy > spawn_energy and (fruits or trees):
            spawn = True
        elif safe and elder_may_spawn and age >= p["elder_age"] and energy > p["elder_spawn_energy"]:
            spawn = True

    # Movement costs are charged on the requested distance but the biome scales
    # the distance actually travelled, so there is nothing to gain from asking
    # for more than we can use.
    move_distance = max(0.0, min(move_distance, sprint_speed))

    turn = 0.0
    if turn_override is not None:
        turn = turn_override
    elif face_angle is not None:
        turn = _wrap(face_angle) * p["turn_gain"]
        turn = max(-p["max_turn"], min(p["max_turn"], turn))

    _advance_pose(mem, state.get("biome", ""), move_distance, _wrap(move_direction), turn)
    return ActionRequest(
        agent_id=agent_id,
        move_distance=float(move_distance),
        move_direction=float(_wrap(move_direction)),
        turn_angle=float(turn),
        spawn_agent=bool(spawn),
    )


def relay_threats(states: Sequence[dict]) -> Dict[int, List[dict]]:
    """Share predator sightings between agents that can see each other.

    For an observer A that sees teammate B at (d, theta_AB) with ``rel_dir``
    (the bearing of A as seen from B, which encodes B's heading), the rotation
    from A's frame to B's frame is ``rot = theta_AB + pi - rel_dir``.  Every
    predator A sees is translated to B's origin and rotated into B's frame, so
    B can react to a threat that is outside its own cone and hearing range.
    """
    seen: Dict[int, List[Tuple[float, float]]] = {}
    for st in states:
        seen[st["agent_id"]] = [
            (o["distance"], o["angle"]) for o in st["observations"] if o.get("type") == "Predator"
        ]

    extra: Dict[int, List[dict]] = {}
    for st in states:
        mine = seen.get(st["agent_id"])
        if not mine:
            continue
        for m in st["observations"]:
            if m.get("type") != "Agent" or "id" not in m or "rel_dir" not in m:
                continue
            d, theta_ab, rel_dir = m["distance"], m["angle"], m["rel_dir"]
            rot = theta_ab + math.pi - rel_dir
            c, sn = math.cos(-rot), math.sin(-rot)
            bx, by = d * math.cos(theta_ab), d * math.sin(theta_ab)
            bucket = extra.setdefault(m["id"], [])
            for dp, tp in mine:
                rx = dp * math.cos(tp) - bx
                ry = dp * math.sin(tp) - by
                qx, qy = rx * c - ry * sn, rx * sn + ry * c
                bucket.append({
                    "type": "Predator",
                    "distance": math.hypot(qx, qy),
                    "angle": math.atan2(qy, qx),
                    "rel_dir": 0.0,
                    "relayed": True,
                })
    return extra


def _merge_relayed(state: dict, relayed: Sequence[dict], dedupe: float = 35.0) -> dict:
    """Return a copy of ``state`` with relayed predators that are not already seen."""
    if not relayed:
        return state
    own = [o for o in state["observations"] if o.get("type") == "Predator"]
    merged = list(state["observations"])
    for r in relayed:
        rx, ry = r["distance"] * math.cos(r["angle"]), r["distance"] * math.sin(r["angle"])
        dup = False
        for o in own:
            ox, oy = o["distance"] * math.cos(o["angle"]), o["distance"] * math.sin(o["angle"])
            if math.hypot(rx - ox, ry - oy) < dedupe:
                dup = True
                break
        if not dup:
            merged.append(r)
    out = dict(state)
    out["observations"] = merged
    return out


def decide_all(agent_states: Sequence[dict], rng: random.Random = None) -> List[ActionRequest]:
    """Decide for the whole species at once.

    Population-level decisions (who is allowed to breed) need the full roster,
    which is why this, rather than the per-agent entry point, is the real API.
    """
    rng = rng or _RNG
    states = [s for s in agent_states if s is not None]
    pop = len(states)

    live_ids = {s["agent_id"] for s in states}
    for stale in [k for k in _MEMORY if k not in live_ids]:
        del _MEMORY[stale]

    p = PARAMS
    if pop == 0:
        return []
    # Food per capita, from what the species can currently see.  Breeding into
    # a stripped map is what produces the boom-bust crashes.
    n_fruit = sum(1 for s in states for o in s["observations"] if o.get("type") == "Fruit")
    n_tree = sum(1 for s in states for o in s["observations"] if o.get("type") == "Tree")
    food_index = (n_fruit + p["food_gate_tree_w"] * n_tree) / pop
    threshold = p["spawn_energy"]
    if pop <= p["spawn_min_pop"]:
        breeders = live_ids                      # rebuild numbers first
        threshold = p["spawn_energy_low"]
    elif pop >= p["spawn_max_pop"] or food_index < p["food_gate"]:
        breeders = set()                         # the range is saturated / stripped
    else:
        scored = sorted(states, key=trait_score, reverse=True)
        keep = max(1, int(len(scored) * p["spawn_elite_frac"]))
        breeders = {s["agent_id"] for s in scored[:keep]}
    elders_ok = pop < p["spawn_max_pop"] * p["elder_pop_slack"]

    relayed = relay_threats(states)
    return [
        decide(_merge_relayed(s, relayed.get(s["agent_id"], ())), s["agent_id"] in breeders, rng,
               spawn_energy=threshold, elder_may_spawn=elders_ok)
        for s in states
    ]


def action_decision(observation_response: dict, rng: random.Random) -> ActionRequest:
    """Single-agent entry point, kept for compatibility with the dummy interface."""
    return decide(observation_response, True, rng)
