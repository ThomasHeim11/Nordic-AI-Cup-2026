import logging
import os
import random
import threading

from fastapi import FastAPI, Body

from src.utils.DTOs import StepResponse
from src.utils.controllers import hivemind_policy

HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", "9052"))

log = logging.getLogger("agent_server")
app = FastAPI(title="Survival Simulator Agent Endpoint")

# The evaluation runs three simulations back to back against one server, so
# per-run memory has to be dropped when a new run starts.  sim_time is
# monotonic within a run and restarts near zero for the next one.
_state = {"last_sim_time": -1.0, "rng": random.Random(1)}
_lock = threading.Lock()


@app.post("/predict")
def predict(step: StepResponse = Body(...)):
    """Receives the current simulation state and returns actions for all agents."""
    with _lock:
        if step.sim_time < _state["last_sim_time"]:
            hivemind_policy.reset()
            _state["rng"] = random.Random(1)
            log.info("new simulation detected (sim_time %.1f -> %.1f)", _state["last_sim_time"], step.sim_time)
        _state["last_sim_time"] = step.sim_time

        try:
            states = [agent.model_dump() for agent in step.agent_status]
            actions = [a.model_dump() for a in hivemind_policy.decide_all(states, _state["rng"])]
        except Exception:  # never let the run die on our account
            log.exception("policy failed; returning idle actions")
            actions = [
                {"agent_id": a.agent_id, "move_distance": 0.0, "move_direction": 0.0,
                 "turn_angle": 0.0, "spawn_agent": False}
                for a in step.agent_status
            ]
    return {"actions": actions}


@app.get("/")
def index():
    return {"message": "Agent endpoint running!"}


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=HOST, port=PORT, log_level=os.environ.get("AGENT_LOG_LEVEL", "warning"))
