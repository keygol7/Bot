#!/usr/bin/env bash
# Bootstrap the arb bot on a fresh Ubuntu host (e.g. AWS us-east-1) for a latency-colocated
# move. IDEMPOTENT and READ-ONLY: it installs deps, builds the venv, and VALIDATES the link
# to the venues + LLM. It NEVER starts live trading and never places an order — do the live
# cutover deliberately (see deploy/README.md). Re-runnable; safe to run twice.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
echo "==> Repo: $REPO_DIR"

echo "==> System deps (Python 3.12 from Ubuntu is fine; bot needs >=3.11)"
sudo apt-get update -y
sudo apt-get install -y python3 python3-venv python3-pip git sqlite3 build-essential tmux curl

echo "==> Python venv + package"
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[venues,dev]"

echo "==> .env"
if [ ! -f .env ]; then
  cat <<'MSG'
!! .env is MISSING. Copy it from your current box (it holds venue creds, LLM_BASE_URL,
   and the RISK_* caps), e.g. from your workstation:
       scp <user>@<unraid-ip>:~/Bot/.env  /tmp/.env   # then move it onto this box
   IMPORTANT: if the LLM (Ollama :11434) runs on a LAN host, set LLM_BASE_URL to a
   Tailscale/VPN IP reachable from here. Then re-run this script.
MSG
  exit 1
fi

echo "==> Core tests (deterministic, no network)"
.venv/bin/python -m pytest -q

echo "==> Read-only validation — NO orders are placed"
echo "--- LLM link (--check-llm) ---"
.venv/bin/python -m bot.dryrun --check-llm || \
  echo "!! LLM unreachable. Fix LLM_BASE_URL (Tailscale to the Ollama host?) before going live."
echo "--- venue creds + balances + FLAT (--check-flat) — confirms keys work from THIS IP ---"
.venv/bin/python -m bot.dryrun --check-flat || \
  echo "!! Venue check failed. Re-auth or allowlist THIS box's IP at the venue before live."

echo "==> Latency from this host"
bash deploy/latency_probe.sh

cat <<'NEXT'

==================== bootstrap complete (READ-ONLY, nothing live) ====================
Next, DELIBERATELY:
  1. Review the latency above — Kalshi 'connect' should be ~0.001-0.003s here. Confirm the
     Polymarket 'ttfb' is also low (that's the hedge leg).
  2. Install the live unit:
       sudo cp deploy/arbbot-stream.service /etc/systemd/system/
       sudo $EDITOR /etc/systemd/system/arbbot-stream.service   # fix User= and the paths
       sudo systemctl daemon-reload && sudo systemctl enable arbbot-stream
  3. CUTOVER (never two live instances at once):
       - confirm BOTH venues FLAT here:  .venv/bin/python -m bot.dryrun --check-flat
       - STOP the old box's live unit:    (on Unraid)  sudo systemctl stop arbbot-stream
       - START here:                      sudo systemctl start arbbot-stream
       - verify the startup guard passes: journalctl -u arbbot-stream -f
=====================================================================================
NEXT
