"""Survival agent endpoint, tuned for per-tick latency.

The evaluator charges every millisecond of our response time against a 600 s
accumulated-wait budget, so this is a bare ASGI app instead of FastAPI:
  * body parsed with orjson, no pydantic validation on the hot path
    (the evaluator always sends the full StepResponse; missing keys fall back
    to defaults instead of a 422),
  * response serialised with orjson,
  * uvicorn on uvloop + httptools.
Behaviour is identical to the FastAPI version: same policy, same reset rule
(sim_time going backwards = new simulation), never raises.
"""
import json
import logging
import os
import random
import threading

try:
    import orjson

    def _loads(b):
        return orjson.loads(b)

    def _dumps(o):
        return orjson.dumps(o)
except ImportError:  # pragma: no cover - orjson is in requirements-server.txt
    def _loads(b):
        return json.loads(b)

    def _dumps(o):
        return json.dumps(o).encode()

from src.utils.controllers import hivemind_policy

HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", "9052"))

log = logging.getLogger("agent_server")

# The evaluation runs three simulations back to back against one server, so
# per-run memory has to be dropped when a new run starts.  sim_time is
# monotonic within a run and restarts near zero for the next one.
_state = {"last_sim_time": -1.0, "rng": random.Random(1)}
_lock = threading.Lock()

_STATE_DEFAULTS = {
    "agent_id": 0, "energy": 0.0, "biome": "", "age": 0.0, "speed": 0.0, "sprint_speed": 0.0,
    "hearing_radius": 0.0, "vision_angle": 0.0, "vision_range": 0.0, "max_energy": 0.0,
    "observations": [],
}
_IDLE = {"move_distance": 0.0, "move_direction": 0.0, "turn_angle": 0.0, "spawn_agent": False}
_HEADERS = [(b"content-type", b"application/json")]


def _clean_states(raw):
    """Fill in any missing fields so the policy never KeyErrors on a partial body."""
    out = []
    for s in raw or ():
        if not isinstance(s, dict):
            continue
        if len(s) < len(_STATE_DEFAULTS):
            s = {**_STATE_DEFAULTS, **s}
        out.append(s)
    return out


def handle_predict(body: bytes) -> bytes:
    states = []
    try:
        step = _loads(body) if body else {}
        sim_time = float(step.get("sim_time") or 0.0)
        states = _clean_states(step.get("agent_status"))
        with _lock:
            if sim_time < _state["last_sim_time"]:
                hivemind_policy.reset()
                _state["rng"] = random.Random(1)
                log.info("new simulation detected (sim_time %.1f -> %.1f)", _state["last_sim_time"], sim_time)
            _state["last_sim_time"] = sim_time
            actions = [a.model_dump() for a in hivemind_policy.decide_all(states, _state["rng"])]
    except Exception:  # never let the run die on our account
        log.exception("policy failed; returning idle actions")
        actions = [{"agent_id": s.get("agent_id", 0), **_IDLE} for s in states]
    return _dumps({"actions": actions})


async def _send_json(send, status: int, body: bytes):
    await send({"type": "http.response.start", "status": status,
                "headers": _HEADERS + [(b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})


async def app(scope, receive, send):
    if scope["type"] == "lifespan":
        while True:
            msg = await receive()
            if msg["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif msg["type"] == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"})
                return
    if scope["type"] != "http":
        return
    method, path = scope["method"], scope["path"]
    if method == "POST" and path.rstrip("/") == "/predict":
        chunks = []
        while True:
            msg = await receive()
            chunks.append(msg.get("body", b""))
            if not msg.get("more_body"):
                break
        body = chunks[0] if len(chunks) == 1 else b"".join(chunks)
        await _send_json(send, 200, handle_predict(body))
    elif method in ("GET", "HEAD"):
        await _send_json(send, 200, b'{"message":"Agent endpoint running!"}')
    else:
        await _send_json(send, 404, b'{"detail":"Not Found"}')


# Warm up at import: the first call pays for lazy imports / param loading.
handle_predict(b'{"game_status":"ok","score":0,"sim_time":0,"n_agents":0,"agent_status":[]}')
hivemind_policy.reset()
_state["last_sim_time"] = -1.0

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    try:
        import uvloop, httptools  # noqa: F401
        loop, http = "uvloop", "httptools"
    except ImportError:
        loop, http = "auto", "auto"
    uvicorn.run(app, host=HOST, port=PORT, loop=loop, http=http, access_log=False,
                log_level=os.environ.get("AGENT_LOG_LEVEL", "warning"))
