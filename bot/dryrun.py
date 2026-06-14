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
from bot.matching.embed import candidate_pairs, semantic_candidate_pairs
from bot.matching.llm_match import MatchVerdict, confirm_match
from bot.models import MarketQuote
from bot.modes import RunMode
from bot.strategies.arbitrage import (
    ArbOpportunity,
    bundle_price_edge,
    cross_price_edge,
    detect_bundle,
)
from bot.venues.base import RawMarket

log = logging.getLogger("bot.dryrun")


@dataclass
class CycleResult:
    markets_seen: int = 0
    quotes: int = 0
    deep_fetches: int = 0
    bundle_opps: list[ArbOpportunity] = field(default_factory=list)
    candidate_pairs: int = 0
    cross: ScanResult = field(default_factory=ScanResult)

    def summary(self) -> str:
        return (
            f"markets={self.markets_seen} quotes={self.quotes} "
            f"deep_fetches={self.deep_fetches} bundle_arbs={len(self.bundle_opps)} "
            f"cross_candidates={self.candidate_pairs} "
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
    embed_fn=None,
) -> CycleResult:
    result = CycleResult()

    # ----- Phase 1: cheap wide price scan (one list call per venue) -----
    # Price-only quotes (no depth) for every market, so we can match/shortlist
    # across the whole board without an order-book call per market. A venue being
    # down must not kill the cycle.
    quotes_by_venue: dict[str, list[MarketQuote]] = {}
    for v in venues:
        try:
            qs = await v.scan_quotes(limit)
        except Exception as exc:
            log.warning("scan_quotes failed for %s: %s", v.name, exc)
            quotes_by_venue[v.name] = []
            continue
        quotes_by_venue[v.name] = qs
        result.markets_seen += len(qs)
        result.quotes += len(qs)
        if store is not None:
            store.upsert_markets((v.name, q.market_id, q.title, None) for q in qs)

    price_lookup = {
        (q.venue, q.market_id): q for qs in quotes_by_venue.values() for q in qs
    }
    deep_needed: set[tuple[str, str]] = set()

    # Bundle candidates by price (single venue): shortlist for a sized fetch.
    bundle_candidates: list[tuple[str, MarketQuote]] = []
    for vname, qs in quotes_by_venue.items():
        fee = fee_models.get(vname, ZeroFeeModel())
        for q in qs:
            edge = bundle_price_edge(q, fee)
            if edge is not None and edge > min_edge:
                bundle_candidates.append((vname, q))
                deep_needed.add((vname, q.market_id))

    # Cross candidates: lexical similarity -> price edge -> LLM confirmation.
    def shortlist(ga, gb):
        if embed_fn is not None:
            return semantic_candidate_pairs(ga, gb, embed_fn, threshold=match_threshold)
        return candidate_pairs(ga, gb, threshold=match_threshold)

    confirmed_lite: list[tuple[MarketQuote, MarketQuote]] = []
    names = list(quotes_by_venue)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            for c in shortlist(quotes_by_venue[names[i]], quotes_by_venue[names[j]]):
                edge = cross_price_edge(
                    c.a, c.b, fee_models.get(c.a.venue), fee_models.get(c.b.venue)
                )
                # Skip only when both legs are priced in the scan AND show no edge.
                # If a leg has no list price (edge == -inf), let it through — the real
                # price/edge is resolved from the depth fetch after confirmation.
                if edge != float("-inf") and edge <= min_edge:
                    continue
                result.candidate_pairs += 1
                verdict = await _verdict_for(c.a, c.b, store, complete_fn)
                if verdict is not None and verdict.tradeable():
                    confirmed_lite.append((c.a, c.b))
                    deep_needed.add((c.a.venue, c.a.market_id))
                    deep_needed.add((c.b.venue, c.b.market_id))

    # ----- Phase 2: deep (sized) fetch, only for shortlisted markets -----
    venue_by_name = {v.name: v for v in venues}
    deep: dict[tuple[str, str], MarketQuote] = {}
    for key in deep_needed:
        v = venue_by_name.get(key[0])
        base = price_lookup.get(key)
        if v is None or base is None:
            continue
        try:
            dq = await v.fetch_quote(RawMarket(market_id=key[1], title=base.title, raw={}))
        except Exception as exc:
            log.warning("deep fetch failed %s:%s: %s", key[0], key[1], exc)
            continue
        if dq is not None:
            deep[key] = dq
    result.deep_fetches = len(deep_needed)

    # Finalize bundle arbs with real sizes.
    for vname, q in bundle_candidates:
        dq = deep.get((vname, q.market_id))
        if dq is None:
            continue
        opp = detect_bundle(dq, fee=fee_models.get(vname, ZeroFeeModel()), min_edge=min_edge)
        if opp is None:
            continue
        decision = risk.check(f"{vname}:{q.market_id}", opp.notional)
        if store is not None:
            store.record_opportunity(opp, acted=False)
        result.bundle_opps.append(opp)
        log.info("BUNDLE %s | risk: %s", opp, decision.reason or "ok")

    # Finalize cross-venue arbs with real sizes.
    confirmed: list[tuple[MarketQuote, MarketQuote]] = []
    for a, b in confirmed_lite:
        da = deep.get((a.venue, a.market_id))
        db = deep.get((b.venue, b.market_id))
        if da is None or db is None:
            continue
        da.event_key = db.event_key = f"{a.label}|{b.label}"
        confirmed.append((da, db))

    engine = Engine(
        fee_models=fee_models, risk=risk, store=store,
        mode=RunMode.DRY_RUN, min_edge=min_edge,
    )
    result.cross = engine.evaluate_pairs(confirmed)
    return result


def _build_venues(settings: Settings) -> list:
    """Both venues read live. Polymarket US market data is on a PUBLIC gateway, so
    no credentials are needed for the dry run — they're only required to place
    orders later."""
    from bot.venues.kalshi import KalshiVenue
    from bot.venues.polymarket_us import PolymarketUSVenue

    venues = [KalshiVenue(settings.kalshi), PolymarketUSVenue(settings.qcex)]
    if settings.qcex.is_trading_configured:
        log.info("Polymarket US: public-gateway reads + trading creds present")
    else:
        log.info("Polymarket US: public-gateway reads (no trading creds — read-only)")
    return venues


async def run(
    *,
    settings: Settings | None = None,
    venues: list | None = None,
    once: bool = False,
    interval: float = 15.0,
    limit: int = 50,
    use_llm: bool = False,
    use_embed: bool = False,
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

    embed_fn = None
    if use_embed:
        from bot.matching.embed_client import make_embed_fn

        embed_fn = make_embed_fn(settings.llm)

    last: CycleResult | None = None
    try:
        while True:
            last = await run_cycle(
                venues, store=store, risk=risk, fee_models=fee_models,
                min_edge=min_edge, match_threshold=match_threshold,
                complete_fn=complete_fn, limit=limit, embed_fn=embed_fn,
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


def check_llm(settings: Settings) -> int:
    """Probe the configured local LLM and print a diagnosis. Returns an exit code."""
    from bot.matching.llm_client import LocalLLMClient

    client = LocalLLMClient(settings.llm.base_url, settings.llm.reasoning_model)
    ok, message = client.check()
    client.close()
    print(("OK: " if ok else "FAIL: ") + message)
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Live read-only DRY_RUN arbitrage monitor")
    p.add_argument("--check-llm", action="store_true",
                   help="probe the local LLM (LLM_BASE_URL) and exit")
    p.add_argument("--once", action="store_true", help="run a single cycle and exit")
    p.add_argument("--interval", type=float, default=15.0, help="seconds between cycles")
    p.add_argument("--limit", type=int, default=50, help="markets to pull per venue")
    p.add_argument("--match-threshold", type=float, default=None,
                   help="title-match cutoff to shortlist a cross-venue pair "
                        "(default 0.5 lexical, 0.80 with --embed)")
    p.add_argument("--llm", action="store_true",
                   help="confirm cross-venue matches with the local LLM (default off)")
    p.add_argument("--embed", action="store_true",
                   help="match titles semantically via the local embedding model "
                        "(recommended; far better than lexical for reworded events)")
    p.add_argument("--min-edge", type=float, default=None,
                   help="min per-contract edge to record (default from settings)")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.check_llm:
        raise SystemExit(check_llm(load_settings()))

    threshold = args.match_threshold
    if threshold is None:
        threshold = 0.80 if args.embed else 0.5

    asyncio.run(run(
        once=args.once, interval=args.interval, limit=args.limit,
        use_llm=args.llm, use_embed=args.embed, match_threshold=threshold,
        min_edge=args.min_edge,
    ))


if __name__ == "__main__":
    main()
