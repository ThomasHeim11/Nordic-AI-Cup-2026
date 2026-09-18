#!/usr/bin/env bash
# Medical pipeline on a CUDA node (run inside the salloc on n009).
#   bash cluster_medical.sh eval            # install, then offline score on the 39 training conversations
#   bash cluster_medical.sh serve           # start api.py + cloudflared tunnel (prints URL)
#   LLM_HF_REPO=Qwen/Qwen2.5-32B-Instruct-AWQ bash cluster_medical.sh eval   # try another LLM
set -e
cd "$(dirname "$0")"
export MEDICAL_BACKEND=torch
export ASR_HF_REPO=${ASR_HF_REPO:-openai/whisper-large-v3-turbo}
export LLM_HF_REPO=${LLM_HF_REPO:-Qwen/Qwen2.5-14B-Instruct}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv && ./.venv/bin/python -m pip install -q --upgrade pip
  ./.venv/bin/python -m pip install -q torch --index-url https://download.pytorch.org/whl/cu126 || ./.venv/bin/python -m pip install -q torch
  ./.venv/bin/python -m pip install -q transformers accelerate datasets sentence-transformers fastapi uvicorn "pydantic>=2.7,<3" requests numpy av faster-whisper
fi
./.venv/bin/python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
case "$1" in
  eval)
    rm -rf cache/transcripts    # transcripts must come from this backend's ASR
    ./.venv/bin/python qa_eval.py --transcribe-only 2>&1 | grep -E "transcribed|Error|error" | tail -3
    echo "== quote mode (Mac 7B reference: 0.706) =="
    ./.venv/bin/python qa_eval.py --mode quote --rerank 1 --seg-sub-min 0.5 --snap 1 --snap-start-tol 0.4 --snap-end-tol 0.2 2>&1 | grep -E "^(  |Accuracy|SCORE|LLM time)"
    echo "== units mode (Mac 7B reference: 0.685) =="
    ./.venv/bin/python qa_eval.py --mode units --rerank 1 --snap 1 --snap-start-tol 0.4 --snap-end-tol 0.2 2>&1 | grep -E "^(  |Accuracy|SCORE|LLM time)"
    ;;
  serve)
    if [ ! -x ./cloudflared ]; then curl -sL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64 -o cloudflared && chmod +x cloudflared; fi
    pkill -f "python api.py" || true
    nohup ./.venv/bin/python api.py > api_9054.log 2>&1 &
    for i in $(seq 1 120); do sleep 5; curl -s --max-time 2 http://127.0.0.1:9054/ >/dev/null 2>&1 && break; done
    curl -s http://127.0.0.1:9054/; echo
    pkill -f "cloudflared tunnel --url http://localhost:9054" || true
    nohup ./cloudflared tunnel --url http://localhost:9054 --no-autoupdate > tunnel_9054.log 2>&1 &
    sleep 8; echo "TUNNEL URL: $(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' tunnel_9054.log | head -1)"
    ;;
  *) echo "usage: bash cluster_medical.sh eval|serve";;
esac
