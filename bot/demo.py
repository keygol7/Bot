"""Offline DRY_RUN demonstration of the full decision pipeline.

Runs with NO credentials and NO network: it builds synthetic order books for the
same event on both venues, runs the lexical pre-filter, confirms the match with a
stand-in for the local LLM, then runs the engine (DRY_RUN) and prints what it would
do. This is a smoke test of the wiring, not a live run.

    python -m bot
"""

from __future__ import annotations

import logging

from bot.data.book import BookStore
from bot.data.store import Store
from bot.engine import Engine
from bot.execution.risk import RiskLimits, RiskManager
from bot.fees import KalshiFeeModel, ZeroFeeModel
from bot.matching.embed import candidate_pairs
from bot.matching.llm_match import confirm_match
from bot.models import PriceLevel
from bot.modes import RunMode


def _fake_local_llm(prompt: str) -> str:
    """Stand-in for the on-prem reasoning model. The real one is an OpenAI-compatible
    localhost endpoint (vLLM/Ollama). Here we hard-confirm so the demo flows end to
    end; live runs replace this with an actual model call."""
    return '{"same_event": true, "confidence": 0.95, "rationale": "same event/resolution"}'


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # 1. Synthetic live books for the SAME event on the two venues.
    books = BookStore()
    books.update(
        "kalshi", "FED-MAR26", title="Fed cuts rates in March 2026",
        yes_bids=[PriceLevel(0.40, 100)],   # -> NO ask 0.60
        no_asks=[],                          # Kalshi NO derived from YES bid
        yes_asks=[PriceLevel(0.41, 100)],
    )
    books.update(
        "polymarket_us", "0xfedmar26", title="Fed cuts interest rates March 2026",
        yes_asks=[PriceLevel(0.62, 100)],
        no_asks=[PriceLevel(0.55, 60)],      # buy NO here for 0.55
    )

    kalshi_quotes = [b.to_quote() for b in books._books.values() if b.venue == "kalshi"]
    poly_quotes = [b.to_quote() for b in books._books.values() if b.venue == "polymarket_us"]

    # 2. Cheap lexical pre-filter -> shortlist candidate cross-venue pairs.
    candidates = candidate_pairs(kalshi_quotes, poly_quotes, threshold=0.3)
    print(f"\nShortlisted {len(candidates)} candidate pair(s) by lexical similarity:")
    for c in candidates:
        print(f"  score={c.score:.2f}  {c.a.title!r}  <->  {c.b.title!r}")

    # 3. Local-LLM confirmation (same event AND same resolution) — cached in prod.
    confirmed = []
    for c in candidates:
        verdict = confirm_match(c.a, c.b, _fake_local_llm)
        print(f"  verdict: same_event={verdict.same_event} conf={verdict.confidence} "
              f"-> {'TRADEABLE' if verdict.tradeable() else 'skip'}")
        if verdict.tradeable():
            # Stamp a shared event_key so the engine treats them as one event.
            c.a.event_key = c.b.event_key = "FED-MAR-2026"
            confirmed.append((c.a, c.b))

    # 4. Engine: detect arb, gate on risk, record — DRY_RUN places no orders.
    store = Store(":memory:")
    engine = Engine(
        fee_models={"kalshi": KalshiFeeModel(), "polymarket_us": ZeroFeeModel()},
        risk=RiskManager(RiskLimits()),
        store=store,
        mode=RunMode.DRY_RUN,
        min_edge=0.01,
    )
    result = engine.evaluate_pairs(confirmed)

    print(f"\nDetected {len(result.detected)} opportunity(ies); "
          f"{len(result.actionable)} actionable in DRY_RUN "
          f"(best simulated profit ${result.best_profit:.2f}).")
    print("No real orders were placed (DRY_RUN).")
    store.close()


if __name__ == "__main__":
    main()
