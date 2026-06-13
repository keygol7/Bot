# Network Setup — bot on Ubuntu VM, LLM on a separate Windows PC

This is the recommended split for the self-hosted deployment:

- **Ubuntu VM** runs the bot (pure Python). It needs **outbound internet** to reach
  the exchanges (Kalshi / QCEX) **and LAN access** to the Windows PC for the LLM.
- **Windows PC** runs the local LLM (Ollama) and serves it over the LAN. The RTX
  5070 (12 GB) comfortably runs a 7B–14B instruct model at 4-bit — ideal for the
  matching / resolution-reasoning jobs.

```
  ┌────────────────────┐        LAN  :11434         ┌────────────────────────┐
  │  Ubuntu VM (bot)   │ ─────────────────────────▶ │ Windows PC (Ollama LLM)│
  │  python -m bot.*   │                            │ RTX 5070, qwen2.5      │
  └─────────┬──────────┘                            └────────────────────────┘
            │ outbound HTTPS
            ▼
   Kalshi / Polymarket US (QCEX)
```

---

## Part A — Windows PC (the LLM server)

1. **Install Ollama** from https://ollama.com/download, then pull the models:
   ```powershell
   ollama pull qwen2.5:14b-instruct
   ollama pull nomic-embed-text
   ```
   (Use `qwen2.5:7b-instruct` if you want more VRAM headroom.)

2. **Expose Ollama on the LAN** — it binds to `127.0.0.1` by default. In an
   **Administrator PowerShell**:
   ```powershell
   setx OLLAMA_HOST "0.0.0.0:11434" /M
   New-NetFirewallRule -DisplayName "Ollama LAN" -Direction Inbound `
       -LocalPort 11434 -Protocol TCP -Action Allow
   ```
   Then **quit Ollama from the system tray and relaunch it** so it rebinds.

3. **Find the Windows LAN IP**: run `ipconfig`, note the `IPv4 Address`
   (e.g. `192.168.1.50`). Give this PC a **static IP or DHCP reservation** so the
   address doesn't change. (Tailscale is a great alternative — use the Tailscale IP
   and skip the firewall rule.)

4. **Verify locally**: open `http://localhost:11434/v1/models` in a browser; you
   should see your models as JSON.

---

## Part B — Ubuntu VM (the bot)

```bash
sudo apt update && sudo apt install -y python3 python3-venv git
git clone <your-repo-url> && cd Bot
python3 -m venv .venv && source .venv/bin/activate

pip install -e ".[venues,dev]"   # httpx, websockets, cryptography, pytest
python -m pytest                 # expect all tests passing
python -m bot                    # offline demo sanity check
```

> **Python version:** Ubuntu 24.04 ships 3.12 (fine). On 22.04 (3.10) install
> Python 3.11 via the deadsnakes PPA, or ask for the floor to be relaxed to 3.10.

---

## Part C — Point the bot at the Windows LLM

Copy `.env.example` to `.env` and set the LAN address from Part A.3:

```
LLM_BASE_URL=http://192.168.1.50:11434/v1
LLM_REASONING_MODEL=qwen2.5:14b-instruct
LLM_EMBEDDING_MODEL=nomic-embed-text
```

**Confirm the bot can reach the model** (built-in probe — fails fast with guidance):

```bash
python -m bot.dryrun --check-llm
```

Expected on success:
```
OK: LLM reachable at http://192.168.1.50:11434/v1
  available models: ['qwen2.5:14b-instruct', 'nomic-embed-text']
  sample reply: 'OK'
```
If it fails, the message tells you what to check (server running, `LLM_BASE_URL`,
firewall). You can also test the raw endpoint directly:
```bash
curl http://192.168.1.50:11434/v1/models
```

---

## Part D — Run the dry run

```bash
python -m bot.dryrun --once --limit 25        # smoke test (no LLM; bundle arbs only)
python -m bot.dryrun --interval 15 --llm      # full soak, cross-venue matching via the Windows LLM
sqlite3 data/bot.db "SELECT * FROM opportunities;"   # review findings
```

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `--check-llm` → "cannot reach LLM" | Ollama not listening on `0.0.0.0` (redo A.2 + relaunch), wrong IP in `LLM_BASE_URL`, or Windows firewall blocking 11434 |
| `curl` to `:11434` hangs | Firewall rule missing, or the two machines aren't on the same subnet |
| Kalshi `403 / 401` on dry run | Read endpoint needs auth or your IP is blocked — set `KALSHI_API_KEY_ID` + the RSA key in `.env`; the bot signs requests automatically (stays read-only) |
| Cross-venue pairs never appear | QCEX not configured (set `QCEX_API_KEY_ID`) or `--llm` not passed (cross-venue matches require LLM confirmation) |
| Model name warning in `--check-llm` | `LLM_REASONING_MODEL` doesn't match a pulled Ollama tag — `ollama list` to see exact names |
