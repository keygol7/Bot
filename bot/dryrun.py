"""Live read-only DRY_RUN runner.

Polls real markets from the enabled venues, runs the existing matching + arbitrage
+ risk pipeline on live prices, and persists every opportunity it finds — while
placing NO orders. This is the "paper trading on live data" stage: run it on the
Denver box, let it soak, and review the ``opportunities`` table for real edges and
matcher false positives before any capital is committed.

Safety: only read-only venue calls are made; the run mode is forced to DRY_RUN and
the executor is never invoked. A cross-venue pair is only ever evaluated for trading
after the matcher confirms it (which requires the local LLM); without ``--llm`` the
runner reports candidate pairs but never marks them tradeable.

CLI:
    python -m bot.dryrun --once --limit 25
    python -m bot.dryrun --interval 15 --llm
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass, field

from bot.config.settings import Settings, load_settings
from bot.data.store import Store
from bot.engine import Engine, ScanResult
from bot.execution.risk import RiskManager
from bot.fees import FeeModel, ZeroFeeModel
from bot.matching.embed import candidate_pairs
from bot.matching.llm_match import MatchVerdict, confirm_match
from bot.models import MarketQuote
from bot.modes import RunMode
from bot.strategies.arbitrage import ArbOpportunity, detect_bundle

log = logging.getLogger("bot.dryrun")


@dataclass
class CycleResult:
    markets_seen: int = 0
    quotes: int = 0
    bundle_opps: list[ArbOpportunity] = field(default_factory=list)
    candidate_pairs: int = 0
    cross: ScanResult = field(default_factory=ScanResult)

    def summary(self) -> str:
        return (
            f"markets={self.markets_seen} quotes={self.quotes} "
            f"bundle_arbs={len(self.bundle_opps)} cross_candidates={self.candidate_pairs} "
            f"cross_detected={len(self.cross.detected)} "
            f"cross_actionable={len(self.cross.actionable)}"
        )


async def _verdict_for(
    a: MarketQuote, b: MarketQuote, store: Store | None, complete_fn
) -> MatchVerdict | None:
    """Cached, else LLM-confirmed, same-event verdict. ``None`` when unconfirmable."""
    if store is not None:
        row = store.get_verdict(a.venue, a.market_id, b.venue, b.market_id)
        if row is not None:
            return MatchVerdict(
                same_event=bool(row["same_event"]),
                confidence=float(row["confidence"] or 0.0),
                rationale=row["rationale"] or "",
            )
    if complete_fn is None:
        return None  # no LLM -> cannot confirm -> never tradeable
    verdict = await asyncio.to_thread(confirm_match, a, b, complete_fn)
    if store is not None:
        store.cache_verdict(
            a.venue, a.market_id, b.venue, b.market_id,
            same_event=verdict.same_event, confidence=verdict.confidence,
            rationale=verdict.rationale,
        )
    return verdict


async def run_cycle(
    venues: list,
    *,
    store: Store | None,
    risk: RiskManager,
    fee_models: dict[str, FeeModel],
    min_edge: float,
    match_threshold: float,
    complete_fn,
    limit: int,
) -> CycleResult:
    result = CycleResult()
    quotes_by_venue: dict[str, list[MarketQuote]] = {}

    # 1. Pull markets + top-of-book from each venue (read-only). A venue being
    #    down (network error, 4xx/5xx) must not kill the cycle — log and move on,
    #    so the soak keeps running and retries on the next cycle.
    for v in venues:
        qs: list[MarketQuote] = []
        try:
            markets = await v.list_markets(limit)
        except Exception as exc:
            log.warning("list_markets failed for %s: %s", v.name, exc)
            quotes_by_venue[v.name] = qs
            continue
        result.markets_seen += len(markets)
        for m in markets:
            if store is not None:
                store.upsert_market(v.name, m.market_id, m.title)
            try:
                q = await v.fetch_quote(m)
            except Exception as exc:  # one bad market shouldn't kill the cycle
                log.warning("fetch_quote failed %s:%s: %s", v.name, m.market_id, exc)
                continue
            if q is not None:
                qs.append(q)
        quotes_by_venue[v.name] = qs
        result.quotes += len(qs)

    # 2. Single-venue bundle arbs (works with one venue — immediate signal).
    for vname, qs in quotes_by_venue.items():
        fee = fee_models.get(vname, ZeroFeeModel())
        for q in qs:
            opp = detect_bundle(q, fee=fee, min_edge=min_edge)
            if opp is None:
                continue
            decision = risk.check(f"{q.venue}:{q.market_id}", opp.notional)
            if store is not None:
                store.record_opportunity(opp, acted=False)
            result.bundle_opps.append(opp)
            log.info("BUNDLE %s | risk: %s", opp, decision.reason or "ok")

    # 3. Cross-venue arbs across confirmed same-event pairs.
    confirmed: list[tuple[MarketQuote, MarketQuote]] = []
    names = list(quotes_by_venue)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            cands = candidate_pairs(
                quotes_by_venue[names[i]], quotes_by_venue[names[j]],
                threshold=match_threshold,
            )
            result.candidate_pairs += len(cands)
            for c in cands:
                verdict = await _verdict_for(c.a, c.b, store, complete_fn)
                if verdict is not None and verdict.tradeable():
                    event_key = f"{c.a.label}|{c.b.label}"
                    c.a.event_key = c.b.event_key = event_key
                    confirmed.append((c.a, c.b))

    engine = Engine(
        fee_models=fee_models, risk=risk, store=store,
        mode=RunMode.DRY_RUN, min_edge=min_edge,
    )
    result.cross = engine.evaluate_pairs(confirmed)
    return result


def _build_venues(settings: Settings) -> list:
    """Kalshi is always enabled; QCEX joins once credentials are configured."""
    from bot.venues.kalshi import KalshiVenue

    venues = [KalshiVenue(settings.kalshi)]
    if settings.qcex.is_configured:
        from bot.venues.polymarket_us import PolymarketUSVenue

        venues.append(PolymarketUSVenue(settings.qcex))
        log.info("QCEX enabled (credentials configured)")
    else:
        log.info("QCEX disabled (no credentials) — Kalshi-only; bundle arbs active")
    return venues


async def run(
    *,
    settings: Settings | None = None,
    venues: list | None = None,
    once: bool = False,
    interval: float = 15.0,
    limit: int = 50,
    use_llm: bool = False,
    match_threshold: float = 0.5,
    min_edge: float | None = None,
    store: Store | None = None,
) -> CycleResult | None:
    settings = settings or load_settings()
    own_store = store is None
    store = store if store is not None else Store(settings.db_path)
    risk = RiskManager(settings.risk)
    if min_edge is None:
        min_edge = settings.risk.min_edge
    if venues is None:
        venues = _build_venues(settings)
    fee_models = {v.name: v.fee_model for v in venues}

    complete_fn = None
    if use_llm:
        from bot.matching.llm_client import make_complete_fn

        complete_fn = make_complete_fn(settings.llm)

    last: CycleResult | None = None
    try:
        while True:
            last = await run_cycle(
                venues, store=store, risk=risk, fee_models=fee_models,
                min_edge=min_edge, match_threshold=match_threshold,
                complete_fn=complete_fn, limit=limit,
            )
            log.info("cycle: %s", last.summary())
            if once:
                break
            await asyncio.sleep(interval)
    finally:
        if own_store:
            store.close()
        for v in venues:
            aclose = getattr(v, "aclose", None)
            if aclose is not None:
                await aclose()
    return last


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Live read-only DRY_RUN arbitrage monitor")
    p.add_argument("--once", action="store_true", help="run a single cycle and exit")
    p.add_argument("--interval", type=float, default=15.0, help="seconds between cycles")
    p.add_argument("--limit", type=int, default=50, help="markets to pull per venue")
    p.add_argument("--match-threshold", type=float, default=0.5,
                   help="lexical similarity to shortlist a cross-venue pair")
    p.add_argument("--llm", action="store_true",
                   help="confirm cross-venue matches with the local LLM (default off)")
    p.add_argument("--min-edge", type=float, default=None,
                   help="min per-contract edge to record (default from settings)")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(run(
        once=args.once, interval=args.interval, limit=args.limit,
        use_llm=args.llm, match_threshold=args.match_threshold, min_edge=args.min_edge,
    ))


if __name__ == "__main__":
    main()
