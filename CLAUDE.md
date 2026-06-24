# CLAUDE.md — Prediction-Market Arbitrage Bot (Kalshi + Polymarket US/QCEX)

Operating notes for Claude Code sessions. The repo docs are the source of truth for
*design*; this file is for *operating* the live system and the few things not obvious
from the code.

## What this is

A self-hosted bot that does risk-free cross-venue arbitrage (buy YES on one venue, NO
on the other when combined cost < $1 after fees) on **Kalshi** (CFTC) and **Polymarket
US / QCEX** (CFTC-licensed). A local LLM confirms two markets are the *same event*
before any trade; it never sits on the trade path (verdicts are precomputed + cached).

Read these for design/runbook context:
- `README.md` — architecture, safety model, all run modes
- `deploy/README.md` — systemd runbook (install/manage/safety for the live unit)
- `NETWORK_SETUP.md` — LAN setup when the LLM runs on a separate box
- `git log` — every design decision is narrated in the commit messages

## ⚠️ This is LIVE, real money

- `BOT_RUN_MODE=LIVE` and `EXEC_MAKER_MODE=true` in `.env`. `--stream` places **real
  orders on both venues**. Polymarket US has **no demo** — its leg is always real.
- **Never restart, stop, or deploy to the live service without explicit confirmation
  from the user.** After ANY crash, open positions must be verified manually before
  re-enabling — the bot does not reconcile pre-existing positions on startup (a
  startup guard refuses to trade unless every venue is funded and flat).
- Risk backstops in `.env`: `RISK_MAX_DAILY_LOSS` (sticky kill switch), `RISK_MIN_EDGE`,
  caps sized from balance. The kill switch is in-memory — a restart resets it.

## Running the live service

`arbbot-stream.service` (systemd) is the production unit. Currently:
`ExecStart=.../python -m bot.dryrun --stream --limit 5000 --interval 120`

```bash
systemctl status arbbot-stream            # is it running?
journalctl -u arbbot-stream -f            # live logs (readable without sudo)
journalctl -u arbbot-stream --since "1 hour ago"
sudo systemctl restart arbbot-stream      # ONLY after git pull / .env change + user OK
```

`arbbot.service` is the read-only monitor variant (no orders) — safe to run anytime.

## Observing results

```bash
sqlite3 data/bot.db "SELECT * FROM opportunities ORDER BY rowid DESC LIMIT 20;"
sqlite3 data/bot.db "SELECT * FROM fills; SELECT * FROM pnl;"
```

Each scan cycle logs a summary line: `cycle: markets=… cross_candidates=… llm_confirms=…
deep_fetches=… bundle_arbs=… cross_detected=… cross_actionable=… executed=…`. Watch
`executed` and `cross_actionable` for real activity; `HALT` / `STARTUP GUARD FAILED`
means the kill switch tripped — investigate before re-enabling.

## Dev / test

```bash
.venv/bin/python -m pytest                       # deterministic core, no network/creds
.venv/bin/python -m bot                           # offline DRY_RUN demo
.venv/bin/python -m bot.dryrun --check-llm        # verify local LLM link
.venv/bin/python -m bot.dryrun --check-flat       # per-venue balance/positions/FLAT verdict
.venv/bin/python -m bot.dryrun --once --limit 500 # one read-only scan cycle (no orders)
```

The deterministic core (`bot/` minus venue adapters) imports only the stdlib; venue
adapters lazily import `httpx`/`websockets`/`cryptography`. Use the venv at `.venv`.

## Conventions

- Logs display in **America/Denver** time (set via `TZ` in the unit) to match the venue
  UIs; all *internal* time logic is explicit UTC (`datetime.now(timezone.utc)`).
- Secrets live in `.env` (never committed). `git remote`: github.com/keygol7/Bot.
- Stray/stale artifacts seen in the working tree: `soak.pid`/`soak.log` (old run),
  `j.env` (root-owned, provenance unconfirmed) — don't assume these are live.
