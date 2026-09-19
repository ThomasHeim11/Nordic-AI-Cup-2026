# Deployment — how to get the three endpoints online

## Survival: direct from the Mac via router port-forward (best latency we have)
- Home line (Get/Telia) has a real public IP: `178.232.205.38`. Router: ASUS RT-N12E at 192.168.1.1
  (admin password unknown; **UPnP is on**, which is how the forward was created).
- Create/refresh the forward (survives until the router reboots):
  `upnpc -a 192.168.1.22 9052 9052 TCP`   (brew install miniupnpc). Check: `upnpc -l`.
- Submit URL: `http://178.232.205.38:9052` (plain http). Measured 17–37 ms/request from n009.
- **VPN must be OFF on the Mac** (otherwise replies leave through the VPN and the port is dead).
- Keep the Mac awake: `caffeinate -dims &`. Nothing heavy on the Mac during a run.
- Cluster tunnel is the fallback (60 ms/tick → cut at ~1000 s).


All three servers run on this Mac. Each one needs a **public URL**; the simplest is a
Cloudflare quick tunnel (no account needed, installed via brew).

> Claude Code's auto-mode classifier refuses to open ingress tunnels itself, so the
> `cloudflared` commands below must be run by a human (type `! <command>` in the prompt
> or run in a separate terminal).

## 1. Start the servers (one terminal tab each, or use run_all.sh)
```bash
cd survival-simulator   && ./.venv/bin/python agent_server.py   # :9052
cd drone-flyby          && ./.venv/bin/python api.py            # :9053
cd medical-appointment  && ./.venv/bin/python api.py            # :9054
```
Health checks: `curl localhost:9052/`, `curl localhost:9053/api`, `curl localhost:9054/`.
Medical takes ~1–2 min to start (loads whisper + Qwen and warms both up).

## 2. Open one tunnel per server (each prints a https://….trycloudflare.com URL)
```bash
cloudflared tunnel --url http://localhost:9052 --no-autoupdate
cloudflared tunnel --url http://localhost:9053 --no-autoupdate
cloudflared tunnel --url http://localhost:9054 --no-autoupdate
```
Quick-tunnel URLs change every time cloudflared restarts → re-submit on the form after a restart.
Cloudflare limits: 100 MB request body (medical bodies are ≤ 5 MB), 100 s idle timeout
(medical budget is 60 s, fine).

## 3. Submit on https://cases.nordicaicup.com/
API key: in `.env` at the project root (single line, no `KEY=` prefix).
URL to paste = tunnel URL + `/predict`, e.g. `https://xxxx.trycloudflare.com/predict`.

Order of buttons: **Verify** → **Queue validation attempt** (unlimited) → **Evaluate** (ONE per challenge).
Survival evaluation runs 3 simulations back to back → keep the server up for ~15 min.

## venvs after the folder move
The three `.venv`s were created under the old `Nordic-AI-Cup-2026/` path, so their `pip`
launcher scripts have a dead absolute path. `./.venv/bin/python` still works; install with
`./.venv/bin/python -m pip install …` (never `./.venv/bin/pip`).

## Memory — the Mac has 16 GB and that is the binding constraint
- Medical server ≈ 6 GB (Qwen-7B-4bit + whisper-turbo), drone ≈ 1.5 GB, survival ≈ 0.2 GB.
  All three servers together fit **only if nothing else runs** (no training, no evolver, no evals).
- Stop servers with `pkill -f "api.py"` / `pkill -f agent_server.py` — the process name is
  capital-P `Python`, so `pkill -f "python api.py"` silently matches nothing and you end up with
  two copies of Qwen in memory (this happened on 2026-09-18 and pushed the machine into 13 GB of swap).
- Check with `ps -eo pid,rss,command | grep -E "api.py|agent_server"` and `sysctl vm.swapusage`.

## Checklist before pressing Evaluate
- [ ] Server has been restarted with the final code/weights and warmed up.
- [ ] Nothing else heavy running on the Mac (evolver / training stopped) — latency matters for
      drone (333 ms/frame) and medical (60 s/conversation).
- [ ] A validation attempt with the *same* build scored as expected.
- [ ] Mac won't sleep: `caffeinate -dims &`.

## Survival on AWS (Sat 19 Sep) — the latency fix
- EC2 t3.micro, eu-north-1 (Stockholm), Ubuntu 26.04, public IP 16.170.155.200, security group opens TCP 22 + 9052.
- Access: `ssh -i ~/Downloads/nordic.pem ubuntu@16.170.155.200`. Repo made public so the VM can `git clone`.
- Server: root `Dockerfile` (survival agent only). Container `survival` runs with `--restart unless-stopped`, port 9052.
- Update after a code/genome change:  ssh in, then
  `cd ~/Nordic-AI-Cup-2026 && git pull -q && sudo docker build -q -t survival . && sudo docker rm -f survival && sudo docker run -d --restart unless-stopped -p 9052:9052 -e PORT=9052 --name survival survival && curl -s localhost:9052/`
- Measured from Oslo: /predict 35 ms median (Mac at home: ~100 ms per tick from Helsinki). Free plan credits ($100); terminate the instance Sunday evening (EC2 → Instance state → Terminate).
- Render (https://survival-3f1o.onrender.com) is a cold spare only (569).
