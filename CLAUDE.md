# CLAUDE.md — Prediction-Market Arbitrage Bot (Kalshi + two Polymarket venues)

Operating notes for Claude Code sessions. The repo docs are the source of truth for
*design*; this file is for *operating* the live system and the few things not obvious
from the code.

## What this is

A self-hosted bot that does cross-venue arbitrage (buy YES on one venue, NO on the
other when combined cost < $1 after fees) on **Kalshi**, **Polymarket US / QCEX**, and
an explicitly opt-in **Polymarket.com** CLOB V2 adapter covering its open binary board.
A local LLM confirms two markets are the *same event*
before any trade; it never sits on the trade path (verdicts are precomputed + cached).

Read these for design/runbook context:
- `README.md` — architecture, safety model, all run modes
- `deploy/README.md` — systemd runbook (install/manage/safety for the live unit)
- `NETWORK_SETUP.md` — LAN setup when the LLM runs on a separate box
- `git log` — every design decision is narrated in the commit messages

## Architecture (big picture)

One async entrypoint, `bot/dryrun.py`, hosts every run mode (read-only scan,
`--live` polling executor, `--stream` WS executor) plus a suite of diagnostic
subcommands (`--check-flat`, `--show-book`, `--find`, `--watchlist`, `--test-order`,
`--probe-ws`…). It wires together four layers that are otherwise decoupled:

- **Deterministic core** (`engine.py`, `strategies/arbitrage.py`, `fees.py`,
  `data/book.py`, `execution/risk.py`, `models.py`, `modes.py`) — stdlib-only, pure,
  unit-tested without network. `Engine.evaluate_pairs` takes *already-confirmed*
  same-event pairs, detects price edges (`detect_cross_venue` / `detect_bundle`),
  applies risk caps, persists, and in a live mode hands each `ArbOpportunity` to the
  executor. Arb math: a binary market pays $1; if YES+NO cost < $1 after fees, the
  spread is risk-free.
- **Venue adapters** (`venues/base.py` Protocol + `kalshi.py`, `polymarket_us.py`,
  `polymarket_com.py`) —
  the *only* code that branches per-exchange. Everything else is written once against
  `Venue`. They lazily import `httpx`/`websockets`/`cryptography`, normalize to
  `MarketQuote`, and expose `scan_quotes` (cheap wide list), `fetch_quote` (sized depth
  for one market), `stream_order_book` (market WS), `stream_private`/`stream_lifecycle`,
  and `place_order`. `ratelimit.py` is a per-venue token bucket.
- **Matching pipeline** (`matching/`) — decides two markets are the *same event*. Flow:
  `embed.py` shortlists by lexical or semantic (`embed_client.py`, `nomic-embed-text`)
  similarity → deterministic guards (`scope.py` period/scope, resolve-date gap,
  `fingerprint.py` structured complement check) → `llm_match.confirm_match` via local
  LLM (`llm_client.py`). Verdicts are cached in SQLite (`match_verdicts`) and **never
  on the trade path**. `MATCH_USE_FINGERPRINT=true` makes the fingerprint sweep
  authoritative and lets discovery skip the embedding/LLM calls entirely.
- **Execution** (`execution/executor.py`, `streaming/`) — turns an opportunity into two
  market-neutral fill-or-kill legs. The discipline is in `executor.py`'s docstring:
  leg-2 fails cleanly → auto-unwind leg 1; any *ambiguous* state (network error /
  partial) → HALT + trip the sticky kill switch (never guess). Maker mode rests the
  Kalshi leg and takes the deep (Polymarket) hedge on fill.

**Two-phase scan** (one cycle, `run_cycle`): phase 1 makes a single `scan_quotes` list
call per venue to price the whole board top-of-book, shortlists candidates by price edge
+ match; phase 2 fetches sized depth (`fetch_quote`) only for that handful. So `--limit
5000` is ~2 calls, not 5000.

**Streaming model** (`--stream`, the production mode): a **slow loop** (`refresh_specs`,
every `--interval`s) re-runs scan→match→confirm to maintain the watchlist of live
same-event pairs; a **fast loop** (`streaming/engine.py`) subscribes both venues' market
WebSockets to just those markets and, on every book tick, re-checks the live edge and
fires the executor in milliseconds. Private-fill streams (`streaming/fills.py`) confirm
fills in real time; `stream_lifecycle` blocks non-OPEN markets and prunes settled ones.
`startup_guard.py` refuses to start unless every venue is funded and flat.

**Persistence** (`data/store.py`) — SQLite at `data/bot.db`: `opportunities`, `fills`,
`pnl`, `markets`, and the `match_verdicts` cache. Off the hot path.

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
