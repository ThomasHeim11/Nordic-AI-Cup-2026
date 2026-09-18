# shared by the survival jobs: venv + deps (idempotent)
cd ~/survival-simulator
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv && ./.venv/bin/python -m pip install -q --upgrade pip
  ./.venv/bin/python -m pip install -q fastapi numpy pydantic pygame requests scipy shapely uvicorn
fi
