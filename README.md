# Prediction-Market Arbitrage Bot — Kalshi + Polymarket US (QCEX)

A self-hosted bot that monitors and trades prediction markets autonomously. The
core strategy is **risk-free cross-platform arbitrage** (buy YES on one venue and
NO on the other when their combined cost is below \$1 after fees); a local-LLM
analyst layer confirms that two markets are truly the same event before any trade,
and an LLM-assisted **directional** strategy is planned as a separate, capped book.

> **Status: live read-only DRY_RUN.** Read-only venue adapters (with optional Kalshi
> request signing), the arbitrage detector, risk gating, persistence, the matching
> pipeline, and a **live dry-run runner that paper-trades on real prices** are
> implemented and unit-tested (64 tests). **No live order placement yet** — that lands
> in the latency-optimized execution phase.

## Design at a glance

Algorithms own the fast, deterministic work (detection, execution, risk). Local
LLMs run **offline** for the two judgment tasks they're actually good at: deciding
whether two markets are the *same event with the same resolution criteria*, and
(phase 2) parsing news/resolution rules for directional edge. Nothing leaves the
box — the LLM is an on-prem OpenAI-compatible endpoint (vLLM/Ollama). The LLM never
sits on the trade path; match verdicts are precomputed and cached.

| Concern | Owner |
|---|---|
| Price ingest, book math, arb detection, order placement, risk limits | **Algorithm** (deterministic, latency-critical) |
| "Same event + same resolution?" / news + directional edge | **Local LLM** (offline, cached) |

## Layout

```
bot/
  config/settings.py   # env-driven settings (.env), risk limits
  modes.py             # DRY_RUN | LIVE_SMALL | LIVE
  models.py            # venue-agnostic MarketQuote / PriceLevel / Side
  fees.py              # ZeroFeeModel (Polymarket), KalshiFeeModel (price-dependent)
  venues/              # base protocol + Kalshi and Polymarket US (QCEX) adapters
  data/book.py         # in-memory order books (hot path)
  data/store.py        # SQLite persistence + match-verdict cache (off hot path)
  matching/            # lexical pre-filter + local-LLM same-event confirmation
  strategies/arbitrage.py  # cross-venue + single-venue arb detection
  execution/risk.py    # position/exposure caps, daily-loss kill switch
  engine.py            # matched quotes -> detect -> risk -> record (DRY_RUN: no orders)
  demo.py              # offline end-to-end DRY_RUN demonstration
tests/                 # standard-library-only deterministic core, 52 tests
```

The deterministic core imports only the standard library, so the test suite runs
with nothing but `pytest`. Venue adapters import `httpx` / `websockets` /
`cryptography` lazily.

## Quick start

```bash
python -m pytest          # run the test suite (no network/credentials needed)
python -m bot             # offline DRY_RUN demo of the full pipeline
cp .env.example .env      # then fill in venue credentials for live phases
```

The demo builds synthetic books for the same event on both venues, shortlists the
pair, confirms it (stand-in for the local LLM), and prints the arbitrage the engine
would act on — without placing any order.

## Live dry run (paper trading on real prices)

`python -m bot.dryrun` polls **live** markets, runs the same matching → detection →
risk pipeline on real prices, and persists every opportunity to SQLite — placing
**no orders**.

```bash
python -m bot.dryrun --once --limit 25     # one cycle, smoke test
python -m bot.dryrun --interval 15         # continuous soak (Ctrl-C to stop)
python -m bot.dryrun --interval 15 --llm   # also confirm cross-venue matches via local LLM
```

> Running the bot on an Ubuntu VM with the LLM on a separate Windows PC? See
> **[NETWORK_SETUP.md](NETWORK_SETUP.md)** for the full LAN runbook. Verify the link
> first with `python -m bot.dryrun --check-llm`.

- **Kalshi runs immediately.** If its read endpoints require auth (or your IP is
  blocked), set `KALSHI_API_KEY_ID` + the RSA key in `.env` — the bot signs requests
  automatically when credentials are present. It stays read-only regardless.
- **Polymarket US reads need no key** — market data is on the public
  `gateway.polymarket.us`, so cross-venue detection works in the dry run without
  credentials. (API keys are only needed to *place orders* later.) Single-venue
  **bundle arbs** (YES+NO < \$1) also produce signal on each venue alone.
- **Cross-venue arbs need `--llm`** (and a local model at `LLM_BASE_URL`): without it
  the runner lists candidate pairs but never marks one tradeable. Verdicts are cached.
- Review findings with `sqlite3 data/bot.db "SELECT * FROM opportunities;"`.

This is the soak stage from the plan — run it for a sustained period and check the
opportunity log for real edges and matcher false positives before enabling live
trading.

## Safety model

- **DRY_RUN by default** — detects and logs, never trades. Live order placement is
  gated behind `BOT_RUN_MODE` and is not implemented yet.
- **Risk limits from day one** — per-market and total exposure caps, a daily-loss
  **kill switch** (sticky once tripped), and an append-only audit log.
- **Fees are modeled, not ignored** — Kalshi's price-dependent fee and Polymarket's
  ~zero fee are subtracted before any edge is reported.
- **Settlement-mismatch guard** — the matcher fails *closed*: an unparseable or
  low-confidence LLM verdict never green-lights a trade.
- **Secrets** live in `.env` / a secrets manager and are never committed.

## Roadmap

1. ✅ Scaffold + read-only venue adapters + normalization + persistence
2. ✅ Matching pipeline (lexical pre-filter + local-LLM confirmation + cache)
3. ✅ Arbitrage detector + DRY_RUN engine
4. ✅ Live read-only DRY_RUN runner (`python -m bot.dryrun`) — paper trading on real prices
5. ⏳ Live WebSocket ingest + latency-optimized two-leg execution (LIVE_SMALL)
6. ⏳ Dashboard + LLM-driven directional book (separate capped strategy)

## Legal note

Targets **Kalshi** (CFTC-regulated) and **Polymarket US / QCEX** (CFTC-licensed).
It does **not** touch the geoblocked international Polymarket. State-level rules for
these venues are in flux — respect each platform's ToS and your local eligibility.
This software is for authorized use by its operator; it is not financial advice.
