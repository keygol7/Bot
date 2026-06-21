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
import time
from dataclasses import dataclass, field

from bot.config.settings import Settings, load_settings
from bot.data.store import Store
from bot.engine import Engine, ScanResult
from bot.execution.risk import RiskManager
from bot.fees import FeeModel, ZeroFeeModel
from bot.matching.embed import candidate_pairs, semantic_candidate_pairs
from bot.matching.llm_match import MatchVerdict, confirm_match
from bot.matching.scope import scope_mismatch
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
    llm_confirms: int = 0
    bundle_opps: list[ArbOpportunity] = field(default_factory=list)
    candidate_pairs: int = 0
    cross: ScanResult = field(default_factory=ScanResult)
    executions: list = field(default_factory=list)  # ExecutionReport (live mode only)
    # Confirmed same-event pairs: (venue_a, market_a, venue_b, market_b, event_key).
    confirmed_pairs: list = field(default_factory=list)
    # (venue, market_id) seen in this cycle's scan — the currently-live market set.
    scanned: set = field(default_factory=set)

    def summary(self) -> str:
        executed = sum(1 for e in self.executions if e.status.value in ("SUCCESS", "UNWOUND"))
        return (
            f"markets={self.markets_seen} quotes={self.quotes} "
            f"cross_candidates={self.candidate_pairs} llm_confirms={self.llm_confirms} "
            f"deep_fetches={self.deep_fetches} bundle_arbs={len(self.bundle_opps)} "
            f"cross_detected={len(self.cross.detected)} "
            f"cross_actionable={len(self.cross.actionable)} executed={executed}"
        )


def _cached_verdict(store: Store | None, a: MarketQuote, b: MarketQuote) -> MatchVerdict | None:
    """Return a previously-cached verdict for this pair, or ``None``."""
    if store is None:
        return None
    row = store.get_verdict(a.venue, a.market_id, b.venue, b.market_id)
    if row is None:
        return None
    return MatchVerdict(
        same_event=bool(row["same_event"]),
        confidence=float(row["confidence"] or 0.0),
        rationale=row["rationale"] or "",
    )


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
    max_confirms: int = 50,
    max_resolve_gap_days: float = 3.0,
    executor=None,
    use_fingerprint: bool = False,
    fingerprint_metrics=None,
    close_within_days: float = 0.0,
    match_cross_venue: bool = True,
) -> CycleResult:
    result = CycleResult()

    # Targeted scan window: only pull markets closing within ``close_within_days`` (the
    # live/imminent set) so today's games are always covered without a huge --limit.
    # 0 = disabled (scan the whole board). Passed to each venue's scan_quotes.
    max_close_ts: int | None = (
        int(time.time() + close_within_days * 86400) if close_within_days > 0 else None
    )
    if max_close_ts is not None:
        log.info("targeted scan: markets closing within %.2g days (max_close_ts=%d)",
                 close_within_days, max_close_ts)

    # ----- Phase 1: cheap wide price scan (one list call per venue) -----
    # Price-only quotes (no depth) for every market, so we can match/shortlist
    # across the whole board without an order-book call per market. A venue being
    # down must not kill the cycle.
    quotes_by_venue: dict[str, list[MarketQuote]] = {}
    for v in venues:
        try:
            qs = await v.scan_quotes(limit, max_close_ts=max_close_ts)
        except Exception as exc:
            log.warning("scan_quotes failed for %s: %s", v.name, exc)
            quotes_by_venue[v.name] = []
            continue
        quotes_by_venue[v.name] = qs
        result.markets_seen += len(qs)
        result.quotes += len(qs)
        result.scanned.update((v.name, q.market_id) for q in qs)
        if store is not None:
            store.upsert_markets((v.name, q.market_id, q.title, None) for q in qs)
        log.info("scan: %s returned %d markets", v.name, len(qs))

    # Milestone: the full market pulldown is complete; matching begins next.
    log.info("scan complete: %d markets across %d venues — matching...",
             result.markets_seen, len(quotes_by_venue))

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
        # Matching (embeddings/LLM) must not crash the cycle — isolate failures.
        try:
            if embed_fn is not None:
                return semantic_candidate_pairs(ga, gb, embed_fn, threshold=match_threshold)
            return candidate_pairs(ga, gb, threshold=match_threshold)
        except Exception as exc:
            log.warning("matching failed this cycle (%s); skipping cross-venue", exc)
            return []

    confirmed_lite: list[tuple[MarketQuote, MarketQuote]] = []
    # Skip the cross-venue match loop (embedding shortlist + LLM confirm) when discovery is
    # off: the fingerprint sweep builds the watchlist straight from the scanned markets, so
    # this pass would only refresh the unused match_verdicts cache. Empty names = no loop.
    names = list(quotes_by_venue) if match_cross_venue else []
    if not match_cross_venue:
        log.info("cross-venue discovery skipped (fingerprint sweep is authoritative)")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            for c in shortlist(quotes_by_venue[names[i]], quotes_by_venue[names[j]]):
                # Deterministic settlement guard FIRST: markets that resolve far apart
                # in time cannot be the same event (single game vs season championship).
                if (
                    c.a.close_time is not None and c.b.close_time is not None
                    and abs(c.a.close_time - c.b.close_time) > max_resolve_gap_days * 86400
                ):
                    continue
                # Scope/period guard: same teams/date but different resolution scope
                # (e.g. "win 2nd half" vs "win the match") is not the same market.
                if scope_mismatch(c.a.title, c.b.title):
                    continue
                # Structured complement gate (when enabled): skip non-complementary
                # pairs BEFORE the LLM — stops wasting confirmations on junk (e.g.
                # KXWCMENTION novelty cross-products) and keeps discovery aligned with
                # what the watchlist will actually trade.
                if use_fingerprint:
                    from bot.matching.fingerprint import (
                        are_complementary, from_kalshi, from_polymarket,
                    )

                    def _fpq(q):
                        return (from_kalshi(q.market_id, q.title) if q.venue == "kalshi"
                                else from_polymarket(q.market_id, q.title))

                    fa, fb = _fpq(c.a), _fpq(c.b)
                    if not are_complementary(fa, fb):
                        continue
                    if fingerprint_metrics and fa.metric not in fingerprint_metrics:
                        continue
                result.candidate_pairs += 1

                verdict = _cached_verdict(store, c.a, c.b)
                if verdict is None:
                    # Not cached: confirm with the LLM, bounded per cycle. Highest-
                    # similarity first; the cache fills in the rest over cycles.
                    if complete_fn is None or result.llm_confirms >= max_confirms:
                        continue
                    verdict = await asyncio.to_thread(confirm_match, c.a, c.b, complete_fn)
                    result.llm_confirms += 1
                    # Heartbeat for long seed runs: a count every 25 confirmations.
                    if result.llm_confirms % 25 == 0:
                        log.info("matching progress: %d confirmations done "
                                 "(%d candidates seen, %d confirmed so far)",
                                 result.llm_confirms, result.candidate_pairs,
                                 len(result.confirmed_pairs))
                    if store is not None:
                        store.cache_verdict(
                            c.a.venue, c.a.market_id, c.b.venue, c.b.market_id,
                            same_event=verdict.same_event, confidence=verdict.confidence,
                            rationale=verdict.rationale,
                        )

                if not verdict.tradeable():
                    continue

                # Confirmed same-event pair -> always part of the streaming watchlist,
                # regardless of whether there's an edge *right now*. The live price edge
                # is checked per-tick by the streaming engine (or below for polling).
                result.confirmed_pairs.append(
                    (c.a.venue, c.a.market_id, c.b.venue, c.b.market_id,
                     f"{c.a.label}|{c.b.label}")
                )

                # Act this cycle only if there's a current price edge (or no list price,
                # in which case resolve it from the depth fetch).
                edge = cross_price_edge(
                    c.a, c.b, fee_models.get(c.a.venue), fee_models.get(c.b.venue)
                )
                if edge == float("-inf") or edge > min_edge:
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

    # ----- Live execution (only when an executor is wired) -----
    if executor is not None:
        for opp in [*result.bundle_opps, *result.cross.actionable]:
            report = await executor.execute(opp)
            result.executions.append(report)
            log.info("EXEC %s | %s", opp.event_key, report)
            if executor.risk.is_killed:
                log.critical("kill switch tripped — halting execution this cycle")
                break
    return result


def apply_balance_caps(risk, snapshots) -> float:
    """Set the per-market and total exposure caps from the live balance check.

    Both are set to the sum of funded venue balances so the funded cash (tracked
    per-venue in the executor) is the real limit — no hardcoded dollar caps. Returns
    the total used (0.0 if no balances were readable, leaving the caps unchanged).
    """
    total = sum(s.balance for s in snapshots if getattr(s, "balance", None) is not None)
    if total <= 0:
        return 0.0
    risk.limits.max_total_exposure = total
    risk.limits.max_position_per_market = total
    log.info("risk caps set from balances: per-market=$%.2f total=$%.2f", total, total)
    return total


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
    max_confirms: int = 50,
    max_resolve_gap_days: float = 3.0,
    live: bool = False,
    store: Store | None = None,
    close_within_days: float | None = None,
) -> CycleResult | None:
    settings = settings or load_settings()
    own_store = store is None
    store = store if store is not None else Store(settings.db_path)
    risk = RiskManager(settings.risk)
    if min_edge is None:
        min_edge = settings.risk.min_edge
    if close_within_days is None:
        close_within_days = settings.scan_close_within_days
    if venues is None:
        venues = _build_venues(settings)
    fee_models = {v.name: v.fee_model for v in venues}

    executor = None
    if live:
        not_ready = [
            v.name for v in venues
            if not (getattr(v, "is_trading_configured", False) or getattr(v, "authenticated", False))
        ]
        if not_ready:
            log.error(
                "LIVE requested but trading credentials missing for %s — staying READ-ONLY. "
                "Set the trading API keys in .env to enable execution.", not_ready,
            )
            live = False
    if live:
        from bot.execution.executor import Executor

        executor = Executor(
            {v.name: v for v in venues}, risk,
            fee_models=fee_models, store=store,
            max_order_contracts=settings.risk.max_order_contracts,
            min_leg_depth=settings.exec_min_leg_depth,
            depth_safety=settings.exec_depth_fraction,
        )
        log.warning(
            "LIVE EXECUTION ENABLED (mode=%s) — placing REAL orders, max %s contracts/order, "
            "caps: per-market $%.0f, total $%.0f, daily-loss $%.0f",
            settings.run_mode.value, settings.risk.max_order_contracts,
            settings.risk.max_position_per_market, settings.risk.max_total_exposure,
            settings.risk.max_daily_loss,
        )

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
                max_confirms=max_confirms, max_resolve_gap_days=max_resolve_gap_days,
                executor=executor, close_within_days=close_within_days,
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


async def build_watchlist(cached, scanned, venues, store=None):
    """Confirmed pairs whose BOTH legs are live now -> the streaming watchlist.

    Durable across embedding/LLM variance: discovery only *adds* to the cache. A
    confirmed pair must NOT drop off the watchlist merely because a leg falls outside
    the discovery scan's --limit page (e.g. Kalshi returns thousands of markets and
    the UFC tickers sit past the first 1000). For any cached leg the wide scan didn't
    surface, do a targeted per-market liveness probe — bounded, since it only touches
    cached markets (a handful), not the whole board.

    ``cached`` is ``[(venue_a, market_a, venue_b, market_b, event_key), ...]``;
    ``scanned`` is the set of ``(venue, market_id)`` the wide scan returned this cycle.
    Returns ``list[ConfirmedPair]``.
    """
    from bot.streaming.engine import ConfirmedPair

    live = set(scanned)
    venue_by_name = {v.name: v for v in venues}
    settled: list[tuple[str, str]] = []   # confirmed closed -> prune from the cache

    to_probe = {
        (vn, mid)
        for (va, ma, vb, mb, _ek) in cached
        for (vn, mid) in ((va, ma), (vb, mb))
        if (vn, mid) not in live
    }
    for (vn, mid) in to_probe:
        v = venue_by_name.get(vn)
        if v is None:
            continue
        is_open = getattr(v, "is_open", None)
        if is_open is not None:
            # Liveness by market STATUS, not book depth: keep open-but-illiquid
            # pre-match markets (empty book now, two-sided near game time) and drop
            # only CONFIRMED closed/settled ones. Unknown status -> keep (don't repeat
            # the regression of dropping live markets on uncertainty).
            try:
                ok = await is_open(mid)
            except Exception as exc:
                log.info("watchlist status probe %s:%s failed (%s)", vn, mid, exc)
                ok = None
            if ok is False:
                log.info("watchlist drop %s:%s — market closed/settled", vn, mid)
                settled.append((vn, mid))
                continue
            live.add((vn, mid))
        else:
            # Fallback (venues without is_open): any answered quote counts as live.
            try:
                q = await v.fetch_quote(RawMarket(market_id=mid, title="", raw={}))
            except Exception as exc:
                log.info("watchlist probe %s:%s not live (%s)", vn, mid, exc)
                continue
            if q is not None:
                live.add((vn, mid))

    # Settled markets are terminal — prune them (and their verdicts) so the cache
    # stays focused on live events and we don't re-probe dead markets next cycle.
    if store is not None and settled:
        verdicts = sum(store.prune_market(vn, mid) for (vn, mid) in settled)
        log.info("pruned %d settled markets (%d cached verdicts) from the cache",
                 len(settled), verdicts)

    out = []
    missing_a = missing_b = 0  # legs of cached pairs still not live after probing
    for (va, ma, vb, mb, ek) in cached:
        a_live = (va, ma) in live
        b_live = (vb, mb) in live
        if a_live and b_live:
            out.append(ConfirmedPair(
                event_key=ek or f"{va}:{ma}|{vb}:{mb}",
                venue_a=va, market_a=ma, venue_b=vb, market_b=mb,
            ))
        else:
            if not a_live:
                missing_a += 1
            if not b_live:
                missing_b += 1
    log.info("watchlist: %d confirmed pairs live (of %d cached)", len(out), len(cached))
    if cached and not out:
        # Both probe and scan failed for every pair — show one concrete cached leg
        # per venue with its live flag so a closed/renamed market is obvious.
        log.warning("watchlist EMPTY after probing %d legs: side-A missing=%d "
                    "side-B missing=%d", len(to_probe), missing_a, missing_b)
        seen: set[str] = set()
        for (va, ma, vb, mb, _ek) in cached:
            for (vn, mid) in ((va, ma), (vb, mb)):
                if vn not in seen:
                    seen.add(vn)
                    log.warning("  cached %s market %r live=%s",
                                vn, mid, (vn, mid) in live)
    return out


async def stream(
    *,
    settings: Settings | None = None,
    refresh_interval: float = 300.0,
    limit: int = 1000,
    use_llm: bool = True,
    use_embed: bool = True,
    match_threshold: float | None = None,
    min_edge: float | None = None,
    max_confirms: int = 50,
    max_resolve_gap_days: float = 3.0,
    close_within_days: float | None = None,
) -> None:
    """Streaming LIVE execution: slow match loop refreshes confirmed pairs; fast WS
    loop re-checks edge on every book update and fires the executor instantly.
    Requires trading credentials (places REAL orders). Validate in demo first."""
    from bot.execution.executor import Executor
    from bot.matching.embed_client import make_embed_fn
    from bot.matching.llm_client import make_complete_fn
    from bot.streaming.engine import StreamingEngine
    from bot.streaming.fills import FillTracker

    settings = settings or load_settings()
    if close_within_days is not None:
        settings.scan_close_within_days = close_within_days
    if min_edge is None:
        min_edge = settings.risk.min_edge
    if match_threshold is None:
        match_threshold = 0.65 if use_embed else 0.5

    venues = _build_venues(settings)
    not_ready = [
        v.name for v in venues
        if not (getattr(v, "is_trading_configured", False) or getattr(v, "authenticated", False))
    ]
    if not_ready:
        log.error("streaming requires trading credentials for %s — aborting", not_ready)
        return

    store = Store(settings.db_path)
    risk = RiskManager(settings.risk)
    fee_models = {v.name: v.fee_model for v in venues}
    tracker = FillTracker()
    executor = Executor(
        {v.name: v for v in venues}, risk, fee_models=fee_models, store=store,
        max_order_contracts=settings.risk.max_order_contracts, fill_confirmer=tracker,
        min_leg_depth=settings.exec_min_leg_depth,
        depth_safety=settings.exec_depth_fraction,
        hedge_buffer=settings.exec_hedge_buffer,
        maker_timeout=settings.exec_maker_timeout,
        maker_improvement=settings.exec_maker_improvement,
        maker_arm_cushion=settings.exec_maker_arm_cushion,
        maker_poll=settings.exec_maker_poll,
    )

    venue_by_name = {v.name: v for v in venues}

    async def depth_fetch(venue_name: str, market_id: str):
        # Sized order-book quote for one market — used to confirm real depth + fresh
        # price the instant a WS price edge appears (WS ticker carries no size).
        v = venue_by_name.get(venue_name)
        if v is None:
            return None
        return await v.fetch_quote(RawMarket(market_id=market_id, title="", raw={}))

    # Maker mode captures the spread (no slippage), so thin edges need no hedge buffer —
    # fire at just min_edge. Taker mode requires the buffer (fire at min_edge + buffer)
    # so the hedge fills through movement instead of unwinding.
    fire_threshold = (min_edge + settings.exec_maker_arm_cushion if settings.exec_maker_mode
                      else min_edge + settings.exec_hedge_buffer)
    engine = StreamingEngine(
        executor=executor, fee_models=fee_models, min_edge=fire_threshold, depth_fetch=depth_fetch,
        max_ws_quote_age=settings.stream_max_ws_quote_age,
        min_leg_price=settings.stream_min_leg_price, store=store,
        maker_mode=settings.exec_maker_mode,
        edge_snapshot_top=settings.stream_edge_snapshot_top,
        edge_persist_secs=settings.stream_edge_persist_secs,
    )

    # Discovery (embedding shortlist + LLM confirm) only feeds match_verdicts, which the
    # fingerprint sweep ignores — so skip the clients entirely when it's off.
    discover = settings.stream_discovery
    complete_fn = make_complete_fn(settings.llm) if (use_llm and discover) else None
    embed_fn = make_embed_fn(settings.llm) if (use_embed and discover) else None

    async def refresh_balances():
        # Re-read available cash per venue so each arb is sized against what's actually
        # there (covers settlements, deposits, and any drift from the running estimate).
        snaps = []
        for v in venues:
            fn = getattr(v, "account_snapshot", None)
            if fn is None:
                continue
            try:
                snaps.append(await fn())
            except Exception as exc:
                log.warning("balance refresh failed for %s: %s", v.name, exc)
        if snaps:
            executor.set_balances(snaps)
            if settings.risk_caps_from_balance:
                apply_balance_caps(risk, snaps)

    async def refresh_specs():
        # Discovery cycle: scans markets + confirms/caches new pairs (embeddings/LLM).
        res = await run_cycle(
            venues, store=store, risk=RiskManager(settings.risk), fee_models=fee_models,
            min_edge=min_edge, match_threshold=match_threshold, complete_fn=complete_fn,
            limit=limit, embed_fn=embed_fn, max_confirms=max_confirms,
            max_resolve_gap_days=max_resolve_gap_days, executor=None,
            use_fingerprint=settings.match_use_fingerprint,
            fingerprint_metrics=settings.match_fingerprint_metrics or None,
            close_within_days=settings.scan_close_within_days,
            match_cross_venue=discover,
        )
        await refresh_balances()
        cached = store.confirmed_pairs(
            use_fingerprint=settings.match_use_fingerprint,
            fingerprint_metrics=settings.match_fingerprint_metrics or None,
            sweep_max_past_s=(settings.match_sweep_past_days * 86400) or None,
        )
        return await build_watchlist(cached, res.scanned, venues, store=store)

    async def feed_private(v):
        try:
            async for ev in v.stream_private():
                await tracker.apply(ev)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("private stream %s ended: %s", v.name, exc)

    async def feed_lifecycle(v):
        # Kalshi market_lifecycle_v2: feed the engine's state guard (block non-OPEN
        # legs) and prune settled/determined markets from the cache instantly, instead
        # of waiting for the 5-min REST is_open probe. Filtered to the watchlist.
        stream = getattr(v, "stream_lifecycle", None)
        if stream is None:
            return
        try:
            async for ev in stream():
                if not engine.watches(v.name, ev.market_ticker):
                    continue
                if ev.state is not None:
                    engine.set_market_state(v.name, ev.market_ticker, ev.state)
                if ev.terminal:
                    store.prune_market(v.name, ev.market_ticker)
                    log.info("lifecycle: %s %s — blocked + pruned", v.name, ev.market_ticker)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("lifecycle stream %s ended: %s", v.name, exc)

    # Startup reconciliation: every trading venue must be funded and flat before a
    # single order can be placed. A leftover leg from a prior crash/abort would turn
    # a market-neutral arb into naked risk; this fails closed (kill switch) if so.
    from bot.execution.startup_guard import reconcile_startup

    guard = await reconcile_startup(
        venues, risk,
        min_balance=settings.startup_min_balance,
        allow_existing_positions=settings.startup_allow_positions,
    )
    if not guard.ok:
        log.critical("aborting stream — startup guard failed: %s", "; ".join(guard.reasons))
        for v in venues:
            aclose = getattr(v, "aclose", None)
            if aclose is not None:
                await aclose()
        store.close()
        return

    # Seed sizing balances from the guard's snapshots (refreshed each cycle thereafter).
    executor.set_balances(guard.snapshots)
    if settings.risk_caps_from_balance:
        apply_balance_caps(risk, guard.snapshots)

    private_tasks = [asyncio.create_task(feed_private(v)) for v in venues]
    private_tasks += [asyncio.create_task(feed_lifecycle(v)) for v in venues]
    log.warning("matching mode: fingerprint=%s metrics=%s (MATCH_USE_FINGERPRINT)",
                settings.match_use_fingerprint,
                sorted(settings.match_fingerprint_metrics) or "all")
    log.warning("discovery: %s (STREAM_DISCOVERY)",
                "embedding+LLM pass each cycle -> match_verdicts" if settings.stream_discovery
                else "OFF — scan + fingerprint sweep only (no embedding/LLM calls)")
    log.warning("scan window: %s (SCAN_CLOSE_WITHIN_DAYS)",
                f"markets closing within {settings.scan_close_within_days:g} days"
                if settings.scan_close_within_days > 0 else "all open markets (no window)")
    log.warning("liquidity guard: %s (EXEC_MIN_LEG_DEPTH)",
                f"both legs need >= {settings.exec_min_leg_depth:g} contracts of depth"
                if settings.exec_min_leg_depth > 0 else "off (will trade any depth)")
    log.warning("WS depth trust: %s (STREAM_MAX_WS_QUOTE_AGE)",
                f"fire off live book when both legs sized & < {settings.stream_max_ws_quote_age:g}s old"
                if settings.stream_max_ws_quote_age > 0 else "off (always REST re-fetch)")
    log.warning("FOK protection: trade %g%% of shown depth, place %s leg first "
                "(EXEC_DEPTH_FRACTION)", settings.exec_depth_fraction * 100, "kalshi")
    log.warning("settling guard: skip fires when a leg is <= $%.2f or >= $%.2f "
                "(STREAM_MIN_LEG_PRICE)", settings.stream_min_leg_price,
                1.0 - settings.stream_min_leg_price)
    if settings.exec_maker_mode:
        guard = (f"cancel-on-drift every {settings.exec_maker_poll:g}s"
                 if settings.exec_maker_poll > 0 else "NO drift guard")
        log.warning("execution: MAKER mode — rest the %s leg as a maker (fire edges >= lock "
                    "$%.2f + drift cushion $%.2f = $%.2f, %gs timeout, %s), take the deep leg "
                    "on fill (EXEC_MAKER_MODE/EXEC_MAKER_ARM_CUSHION/EXEC_MAKER_POLL)",
                    "kalshi", min_edge, settings.exec_maker_arm_cushion, fire_threshold,
                    settings.exec_maker_timeout, guard)
    else:
        log.warning("execution: TAKER mode — only fire edges >= lock $%.2f + hedge $%.2f "
                    "= $%.2f; hedge leg gets $%.2f of fill room (EXEC_HEDGE_BUFFER)",
                    min_edge, settings.exec_hedge_buffer, fire_threshold,
                    settings.exec_hedge_buffer)
    ceiling = (f"{settings.risk.max_order_contracts:g} ct/order"
               if settings.risk.max_order_contracts and settings.risk.max_order_contracts > 0
               else "no per-order ceiling — sized to balances/depth")
    log.warning("STREAMING LIVE — real orders on confirmed pairs (%s, "
                "caps $%.0f/$%.0f/$%.0f)", ceiling,
                settings.risk.max_position_per_market, settings.risk.max_total_exposure,
                settings.risk.max_daily_loss)
    try:
        await engine.run(venues, refresh_specs, refresh_interval=refresh_interval)
    finally:
        for t in private_tasks:
            t.cancel()
        await asyncio.gather(*private_tasks, return_exceptions=True)
        store.close()
        for v in venues:
            aclose = getattr(v, "aclose", None)
            if aclose is not None:
                await aclose()


async def _probe_stream(venue, market_ids, *, n: int = 3, timeout: float = 20.0):
    """Open a venue's market WS, collect up to ``n`` quotes within ``timeout``.

    Returns (ok, message, samples). The stream reconnects internally on error, so a
    credential/URL problem shows up as a timeout with 0 messages (check logs for why).
    """
    got = []
    agen = venue.stream_order_book(market_ids)

    async def _collect():
        async for q in agen:
            got.append(q)
            if len(got) >= n:
                break

    try:
        await asyncio.wait_for(_collect(), timeout)
        msg = "ok" if got else f"no messages in {timeout:.0f}s (auth/URL? see warnings)"
        return (len(got) > 0), msg, got
    except asyncio.TimeoutError:
        return (len(got) > 0), f"timeout after {timeout:.0f}s, {len(got)} msgs", got
    except Exception as exc:
        return False, f"error: {exc}", got
    finally:
        aclose = getattr(agen, "aclose", None)
        if aclose is not None:
            await aclose()


def check_ws(settings: Settings) -> int:
    """Probe each venue's market WebSocket and print a diagnosis. Returns an exit code."""
    async def _run() -> int:
        venues = _build_venues(settings)
        rc = 0
        try:
            for v in venues:
                try:
                    qs = await v.scan_quotes(50)
                    mids = [q.market_id for q in qs][:30]
                except Exception as exc:
                    print(f"{v.name} WS: FAIL (couldn't list markets to subscribe: {exc})")
                    rc = 1
                    continue
                ok, msg, samples = await _probe_stream(v, mids)
                tag = "OK" if ok else "FAIL"
                extra = ""
                if samples:
                    s = samples[0]
                    extra = f" | sample {s.market_id} yes_ask={s.yes_ask} no_ask={s.no_ask}"
                print(f"{v.name} market WS: {tag} ({msg}){extra}")
                if not ok:
                    rc = 1
        finally:
            for v in venues:
                aclose = getattr(v, "aclose", None)
                if aclose is not None:
                    await aclose()
        print("(private fill streams can only be verified by placing a test order.)")
        return rc

    return asyncio.run(_run())


def probe_ws_book(settings: Settings, slug: str, n: int = 6) -> int:
    """Diagnose the Polymarket WS-vs-REST top-of-book gap for one slug.

    Fetches the REST ``/book`` top, then captures the first ``n`` raw MARKET_DATA WS
    messages and, for each, prints the raw level counts + the parsed top. If the WS
    messages carry only a few levels and their parsed top diverges from REST, the channel
    is delta-based (and our snapshot assumption is the bug). If they carry the full book
    and still differ, it's a side/parse issue or genuine fast movement.
    """
    from bot.venues.polymarket_us import parse_market_data

    async def _run() -> int:
        venues = {v.name: v for v in _build_venues(settings)}
        v = venues.get("polymarket_us")
        if v is None:
            print("polymarket_us venue not configured (need QCEX creds)")
            return 2
        try:
            rest = await v.fetch_quote(RawMarket(market_id=slug, title="", raw={}))
            if rest is None:
                print(f"REST /book: no quote for {slug} (404?)")
            else:
                print(f"REST /book : yes_ask={rest.yes_ask}@{rest.yes_ask_size:g}  "
                      f"no_ask={rest.no_ask}@{rest.no_ask_size:g}  state={getattr(rest,'state',None)}")
            print(f"\ncapturing {n} raw WS MARKET_DATA messages for {slug} ...\n")
            try:
                msgs = await v.capture_market_data([slug], limit=n)
            except Exception as exc:
                print(f"WS capture failed: {exc}")
                return 1
            for i, m in enumerate(msgs):
                md = m.get("marketData") or {}
                bids = md.get("bids") or []
                offers = md.get("offers") or []
                top_bids = [(b.get("px", {}).get("value", b.get("px")), b.get("qty")) for b in bids[:3]]
                top_offers = [(o.get("px", {}).get("value", o.get("px")), o.get("qty")) for o in offers[:3]]
                q = parse_market_data(m)
                keys = sorted(k for k in md.keys() if k not in ("bids", "offers"))
                print(f"[msg {i}] slug={md.get('marketSlug')} other_keys={keys}")
                print(f"         bids({len(bids)}) top={top_bids}")
                print(f"         offers({len(offers)}) top={top_offers}")
                if q is not None:
                    print(f"         -> parsed yes_ask={q.yes_ask}@{q.yes_ask_size:g}  "
                          f"no_ask={q.no_ask}@{q.no_ask_size:g}")
            print("\nread: if bids/offers counts are small and the parsed top swings between "
                  "messages / disagrees with REST, the channel is DELTA-based — each message "
                  "is a partial book, so parsing it as a full snapshot picks a wrong top.")
        finally:
            aclose = getattr(v, "aclose", None)
            if aclose is not None:
                await aclose()
        return 0

    return asyncio.run(_run())


def show_book(settings: Settings, spec: str) -> int:
    """Fetch and print one market's real order book (sized quote) from its venue.
    ``spec`` is ``venue:market_id``. Read-only — diagnoses whether a market actually
    has depth (vs an empty book behind an indicative ticker price)."""
    venue_name, _, market = spec.partition(":")
    if not market:
        print("usage: --show-book VENUE:MARKET_ID")
        return 2

    async def _run() -> int:
        venues = {v.name: v for v in _build_venues(settings)}
        v = venues.get(venue_name)
        if v is None:
            print(f"unknown venue {venue_name!r}; known: {sorted(venues)}")
            return 2
        try:
            q = await v.fetch_quote(RawMarket(market_id=market, title="", raw={}))
        except Exception as exc:
            print(f"{venue_name}:{market} -> fetch failed: {exc}")
            return 1
        finally:
            aclose = getattr(v, "aclose", None)
            if aclose is not None:
                await aclose()
        if q is None:
            print(f"{venue_name}:{market} -> no quote (market not found / 404)")
            return 0
        ya = f"{q.yes_ask}@{q.yes_ask_size:g}" if q.yes_ask is not None else "None@0"
        na = f"{q.no_ask}@{q.no_ask_size:g}" if q.no_ask is not None else "None@0"
        print(f"{venue_name}:{market}  yes_ask={ya}  no_ask={na}")
        if q.yes_ask is None and q.no_ask is None:
            print("  -> EMPTY book (no resting orders to buy against) — not tradeable")
        return 0

    return asyncio.run(_run())


def find_markets(settings: Settings, term: str, *, limit: int = 8000) -> int:
    """Search BOTH venues' scans for a term (team/player) and show whether the bot can
    see and pair it. Answers "there's a live match — why isn't the bot trading it?":
    it scans exactly like discovery (so a market missing here is outside the scan
    window), lists the matches per venue, and runs the fingerprint matcher across the
    cross-venue candidates so you can see which pair (if any) is complementary.
    """
    from bot.matching.fingerprint import complement_reason, from_kalshi, from_polymarket

    t = term.lower()

    async def _run() -> int:
        venues = _build_venues(settings)
        by_venue: dict[str, list] = {}
        try:
            for v in venues:
                try:
                    qs = await v.scan_quotes(limit)
                except Exception as exc:
                    print(f"{v.name}: scan failed ({exc})")
                    continue
                hits = [q for q in qs
                        if t in (q.title or "").lower() or t in q.market_id.lower()]
                by_venue[v.name] = hits
                print(f"\n== {v.name}: {len(hits)} of {len(qs)} scanned markets match '{term}' ==")
                for q in hits[:40]:
                    print(f"  {q.market_id}  | {q.title}")
        finally:
            for v in venues:
                aclose = getattr(v, "aclose", None)
                if aclose is not None:
                    await aclose()

        k = by_venue.get("kalshi", [])
        p = by_venue.get("polymarket_us", [])
        if not k or not p:
            print(f"\n>>> no cross-venue pair: kalshi={len(k)} poly={len(p)} markets "
                  f"matching '{term}' in the scan window. If it's live on a venue but 0 "
                  f"here, it's beyond the scan --limit (coverage gap).")
            return 0
        print(f"\n== fingerprint pairing ({len(k)}x{len(p)} combos; showing complementary "
              f"+ near-misses) ==")
        shown = 0
        for a in k:
            fa = from_kalshi(a.market_id, a.title)
            for b in p:
                fb = from_polymarket(b.market_id, b.title)
                reason = complement_reason(fa, fb)
                if reason == "ok" or shown < 20:
                    tag = "MATCH" if reason == "ok" else f"no ({reason})"
                    print(f"  [{tag}] {a.market_id}  <>  {b.market_id}")
                    shown += 1
        return 0

    return asyncio.run(_run())


def count_markets(settings: Settings, *, limit: int = 50000) -> int:
    """Pull every open market from each venue (no matching) and print the counts.

    Answers "how many markets will a seed run pull?" cheaply and repeatably — it runs
    the same paginated scan_quotes the discovery cycle uses, then exits."""
    async def _run() -> int:
        venues = _build_venues(settings)
        total = 0
        try:
            for v in venues:
                try:
                    qs = await v.scan_quotes(limit)
                except Exception as exc:
                    print(f"{v.name}: scan failed ({exc})")
                    continue
                total += len(qs)
                print(f"{v.name}: {len(qs)} open markets")
            print(f"total across {len(venues)} venues: {total} markets "
                  f"(a seed run pulls these, then matches across venues)")
        finally:
            for v in venues:
                aclose = getattr(v, "aclose", None)
                if aclose is not None:
                    await aclose()
        return 0

    return asyncio.run(_run())


def show_watchlist(settings: Settings, *, limit: int = 500) -> int:
    """Print the LIVE streaming watchlist: cached tradeable pairs whose both legs are
    currently quotable. Probes each leg the way the streamer does (drops settled/empty
    books), so this is exactly what would be traded right now — not just the cache."""
    store = Store(settings.db_path)
    venues = _build_venues(settings)

    async def _run() -> int:
        try:
            cached = store.confirmed_pairs(
                use_fingerprint=settings.match_use_fingerprint,
                fingerprint_metrics=settings.match_fingerprint_metrics or None,
                sweep_max_past_s=(settings.match_sweep_past_days * 86400) or None,
            )
            # No wide scan needed: build_watchlist probes each cached leg directly.
            live = await build_watchlist(cached, set(), venues)

            def title_of(v: str, mid: str) -> str:
                r = store.conn.execute(
                    "SELECT title FROM markets WHERE venue=? AND market_id=?", (v, mid)
                ).fetchone()
                return (r["title"] if r and r["title"] else mid)

            print(f"{len(live)} live watchlist pairs (of {len(cached)} cached tradeable):\n")
            for p in live[:limit]:
                print(f"  • [{p.venue_a}] {p.market_a}  {title_of(p.venue_a, p.market_a)}")
                print(f"    [{p.venue_b}] {p.market_b}  {title_of(p.venue_b, p.market_b)}")
            if len(live) > limit:
                print(f"\n  … {len(live) - limit} more (raise --limit)")
            return 0
        finally:
            for v in venues:
                aclose = getattr(v, "aclose", None)
                if aclose is not None:
                    await aclose()
            store.close()

    return asyncio.run(_run())


def test_order(settings: Settings, spec: str, *, do_fill: bool = False) -> int:
    """Place ONE test order to verify the live buying path (auth, signing, submission,
    response parsing) on a real venue. ``spec`` is ``venue:market_id``.

    Default = a CANARY: a 1-contract fill-or-kill BUY YES at $0.01 that cannot match,
    so it is KILLED immediately — proving the order path works while spending $0 and
    taking NO position. ``--fill`` instead buys 1 contract at the current ask (REAL
    money, REAL naked position you must close yourself)."""
    from bot.models import Side

    venue_name, _, market = spec.partition(":")
    if not market:
        print("usage: --test-order VENUE:MARKET_ID  (e.g. kalshi:KXSOMETICKER)")
        return 2

    async def _run() -> int:
        venues = {v.name: v for v in _build_venues(settings)}
        v = venues.get(venue_name)
        if v is None:
            print(f"unknown venue {venue_name!r}; known: {sorted(venues)}")
            return 2
        if not (getattr(v, "is_trading_configured", False) or getattr(v, "authenticated", False)):
            print(f"{venue_name}: no trading credentials configured — cannot place orders")
            return 1
        try:
            if do_fill:
                q = await v.fetch_quote(RawMarket(market_id=market, title="", raw={}))
                if q is None or q.yes_ask is None:
                    print(f"{venue_name}:{market}: no ask available to fill against")
                    return 1
                price, label = q.yes_ask, "REAL FILL"
                log.warning("placing a REAL 1-contract BUY YES @ %.2f on %s:%s — this is "
                            "a live naked position you must close manually", price, venue_name, market)
            else:
                price, label = 0.01, "canary (no-fill)"
            res = await v.place_order(market, Side.YES, "buy", price, 1)
            print(f"{label}: {venue_name}:{market} -> {res}")
            if do_fill:
                if res.left_a_position:
                    print("  ✅ buying works — order FILLED. Close this position manually.")
                else:
                    print("  order accepted but not filled (ask moved?); path works, no position.")
            else:
                if res.status.value == "KILLED" or not res.left_a_position:
                    print("  ✅ order path works (accepted + processed); no fill, $0 spent.")
                else:
                    print("  ⚠️ canary unexpectedly filled (ask was $0.01?); you hold 1 contract.")
            return 0
        except Exception as exc:
            print(f"  ❌ order path FAILED: {exc}")
            return 1
        finally:
            aclose = getattr(v, "aclose", None)
            if aclose is not None:
                await aclose()

    return asyncio.run(_run())


def recheck_matches(settings: Settings, *, limit: int = 100) -> int:
    """Re-judge the CURRENT tradeable set with the live prompt + model, WITHOUT writing
    to the cache. A cheap way to test whether the prompt rewrite + your current model
    are good enough before committing to a full re-seed.

    Only the post-guard tradeable pairs are rechecked (a few hundred at most), so the
    model is tested on exactly the subject/contest cases the deterministic guards can't
    resolve. Prints which pairs the model would now reject (good if they were false
    positives)."""
    from bot.matching.llm_client import make_complete_fn

    store = Store(settings.db_path)
    complete_fn = make_complete_fn(settings.llm)
    try:
        def title_of(v: str, mid: str) -> str:
            r = store.conn.execute(
                "SELECT title FROM markets WHERE venue=? AND market_id=?", (v, mid)
            ).fetchone()
            return (r["title"] if r and r["title"] else mid)

        pairs = store.confirmed_pairs()
        sample = pairs[:limit]
        print(f"rechecking {len(sample)} of {len(pairs)} tradeable pairs with model "
              f"{settings.llm.reasoning_model!r} (no cache writes)...\n")
        flips = 0
        for (va, ma, vb, mb, _ek) in sample:
            a = MarketQuote(venue=va, market_id=ma, title=title_of(va, ma))
            b = MarketQuote(venue=vb, market_id=mb, title=title_of(vb, mb))
            v = confirm_match(a, b, complete_fn)
            if not (v.same_event and v.tradeable()):
                flips += 1
                print(f"  WOULD DROP (conf {v.confidence:.2f}): {a.title}")
                print(f"                          || {b.title}")
                if v.rationale:
                    print(f"                          -> {v.rationale}")
        kept = len(sample) - flips
        print(f"\nresult: model KEEPS {kept}, would REJECT {flips} of {len(sample)}. "
              f"Rejections are good if they were false positives; spot-check the KEEPS "
              f"for any remaining wrong-subject/wrong-contest pairs.")
        return 0
    finally:
        store.close()


def diagnose_event(settings: Settings, needle: str, limit: int = 80) -> int:
    """Read-only root-cause for a MISSING match: why isn't event <needle> tradeable?

    The two venues spell teams differently (Kalshi 'BELIRI'/'Belgium', Polymarket
    'bel-irn'/'BEL'), so a single substring can't catch both sides. Instead this ANCHORS
    on every scanned market whose title/id contains <needle>, then scans the ENTIRE
    opposite-venue table for a fingerprint complement — encoding-agnostic. For each
    matchable anchor it prints the best complements (and same-metric near-misses) with the
    deterministic reason AND whether the pair is cached, so the failure mode is clear:

      * no complement at all        -> the other venue doesn't list it (real scan/coverage gap)
      * reason 'ok' but 'NOT cached' -> the embedding/lexical shortlist never proposed the
        pair, so it never entered ``match_verdicts``; the authoritative fingerprint, gated
        behind that shortlist, never sees it. This is the recall bottleneck.
      * reason != 'ok'              -> the fingerprint itself rejects it (the reason says why)
    """
    from bot.matching.fingerprint import (
        complement_reason, from_kalshi, from_polymarket,
    )

    store = Store(settings.db_path)
    try:
        like = f"%{needle.lower()}%"
        anchors = store.conn.execute(
            """SELECT venue, market_id, title FROM markets
               WHERE lower(title) LIKE ? OR lower(market_id) LIKE ?
               ORDER BY venue, market_id LIMIT ?""",
            (like, like, limit),
        ).fetchall()
        if not anchors:
            print(f"no scanned markets match '{needle}'. If you expected one, it isn't in "
                  f"the DB — not scanned (out of the imminent window, or the venue doesn't "
                  f"list it). Widen --close-within-days / re-scan.")
            return 0
        all_markets = store.conn.execute(
            "SELECT venue, market_id, title FROM markets"
        ).fetchall()

        def fp(venue, mid, title):
            return (from_kalshi(mid, title) if venue == "kalshi"
                    else from_polymarket(mid, title))

        # Pre-compute the opposite-venue fingerprints once.
        others = [(m, fp(m["venue"], m["market_id"], m["title"] or "")) for m in all_markets]

        matchable_anchors = 0
        for a in anchors:
            fa = fp(a["venue"], a["market_id"], a["title"] or "")
            if not fa.matchable:
                continue                          # unmatchable by design — skip the noise
            matchable_anchors += 1
            print(f"\n[{a['venue']}] {a['market_id']}  {a['title']}")
            print(f"    metric={fa.metric} league={fa.league} thr={fa.threshold} "
                  f"subj={sorted(fa.subject)} matchup={sorted(fa.matchup)}")
            hits, near = [], []
            for m, fb in others:
                if m["venue"] == a["venue"] or not fb.matchable:
                    continue
                reason = complement_reason(fa, fb)
                if reason == "ok":
                    hits.append((reason, m))
                elif fa.metric == fb.metric:      # same metric, just failed a later gate
                    near.append((reason, m))
            if not hits and not near:
                print("    -> NO complement or same-metric market on the other venue "
                      "(coverage gap: the counterpart isn't scanned / doesn't exist)")
            for _reason, m in hits[:6]:
                v = store.get_verdict(a["venue"], a["market_id"], m["venue"], m["market_id"])
                cached = (f"cached same_event={v['same_event']}" if v else "NOT cached")
                print(f"    ✓ ok                  [{cached}]  [{m['venue']}] "
                      f"{m['market_id']}  {m['title']}")
            for reason, m in near[:4]:
                print(f"    ✗ {reason:<18}  [{m['venue']}] {m['market_id']}  {m['title']}")
        print(f"\n  {matchable_anchors} matchable anchor(s) examined. legend: "
              "'✓ ok + NOT cached' = recall bottleneck (shortlist never proposed it); "
              "'✗ <reason>' = fingerprint rejected; 'NO complement' = coverage gap.")
        return 0
    finally:
        store.close()


def check_llm(settings: Settings) -> int:
    """Probe the configured local LLM and print a diagnosis. Returns an exit code."""
    from bot.matching.llm_client import LocalLLMClient

    client = LocalLLMClient(settings.llm.base_url, settings.llm.reasoning_model)
    ok, message = client.check()
    client.close()
    print(("OK: " if ok else "FAIL: ") + message)
    return 0 if ok else 1


def inspect_matches(
    settings: Settings, *, show_rejected: bool = False, tradeable_only: bool = False,
    limit: int = 50,
) -> int:
    """Print the cached match verdicts with both market titles so a human can audit
    whether the matcher pairs the *same* event/resolution. Read-only.

    Confirmed pairs (same_event=1) are the ones that can be traded, so a false
    positive here is the dangerous case — eyeball that the two titles really are the
    same event with the same resolution. ``--show-rejected`` also lists rejected pairs
    to catch false negatives (missed matches). ``--tradeable-only`` lists EXACTLY the
    post-filter set the streamer will trade (confidence floor + fan-out guard)."""
    store = Store(settings.db_path)
    try:
        def title_of(venue: str, mid: str) -> str:
            row = store.conn.execute(
                "SELECT title FROM markets WHERE venue=? AND market_id=?", (venue, mid)
            ).fetchone()
            return (row["title"] if row and row["title"] else mid)

        def rows(same_event: int):
            return store.conn.execute(
                """SELECT v.confidence AS conf, v.rationale AS why,
                          v.venue_a, v.market_a, v.venue_b, v.market_b,
                          ma.title AS title_a, mb.title AS title_b
                   FROM match_verdicts v
                   LEFT JOIN markets ma ON ma.venue=v.venue_a AND ma.market_id=v.market_a
                   LEFT JOIN markets mb ON mb.venue=v.venue_b AND mb.market_id=v.market_b
                   WHERE v.same_event=?
                   ORDER BY v.confidence DESC
                   LIMIT ?""",
                (same_event, limit),
            ).fetchall()

        tradeable_pairs = store.confirmed_pairs(
            use_fingerprint=settings.match_use_fingerprint,
            fingerprint_metrics=settings.match_fingerprint_metrics or None,
            sweep_max_past_s=(settings.match_sweep_past_days * 86400) or None,
        )
        if tradeable_only:
            # Exactly what the streamer will trade — audit this before going live.
            print(f"{len(tradeable_pairs)} TRADEABLE pairs (matches the live filter):\n")
            for (va, ma, vb, mb, _ek) in tradeable_pairs[:limit]:
                # Show the market id alongside the title so it can be copied straight
                # into --test-order VENUE:MARKET.
                print(f"  ✓ [{va}] {ma}  {title_of(va, ma)}")
                print(f"      [{vb}] {mb}  {title_of(vb, mb)}")
            if len(tradeable_pairs) > limit:
                print(f"\n  … {len(tradeable_pairs) - limit} more (raise --limit to see all)")
            return 0

        total = store.conn.execute("SELECT COUNT(*) c FROM match_verdicts").fetchone()["c"]
        raw_confirmed = store.conn.execute(
            "SELECT COUNT(*) c FROM match_verdicts WHERE same_event=1"
        ).fetchone()["c"]
        # What the streamer will ACTUALLY trade: confidence floor + fan-out guard.
        tradeable = len(tradeable_pairs)
        confirmed = rows(1)
        print(f"match cache: {total} verdicts total, {raw_confirmed} marked same-event, "
              f"{tradeable} TRADEABLE after confidence+fan-out filters "
              f"(showing up to {limit} same-event below; use --tradeable-only to audit "
              f"the exact trade set)\n")
        if not confirmed:
            print("  (no confirmed pairs yet — discovery hasn't matched anything; "
                  "check --check-llm and that --embed/--llm are on)")
        for r in confirmed:
            print(f"  ✓ {r['conf']:.2f}  [{r['venue_a']}] {r['title_a'] or r['market_a']}")
            print(f"          [{r['venue_b']}] {r['title_b'] or r['market_b']}")
            if r["why"]:
                print(f"          → {r['why']}")
        if show_rejected:
            rej = rows(0)
            print(f"\nrejected (not same event), showing up to {limit}:")
            for r in rej:
                print(f"  ✗ {r['conf']:.2f}  [{r['venue_a']}] {r['title_a'] or r['market_a']}"
                      f"  vs  [{r['venue_b']}] {r['title_b'] or r['market_b']}")
        return 0
    finally:
        store.close()


def probe_account(settings: Settings) -> int:
    """Discover the Polymarket US account/portfolio REST endpoints empirically.

    Signs and GETs a ranked list of candidate paths (balance is under ``account``,
    positions under ``portfolio`` per the official SDK) and prints the status + a
    JSON snippet for any that return 200, so the real paths/shapes can be locked in
    without guessing. Read-only — never places an order.
    """
    from bot.venues.polymarket_us import PolymarketUSVenue

    if not settings.qcex.is_trading_configured:
        print("Polymarket US trading creds not configured "
              "(set QCEX_API_KEY_ID + QCEX_SECRET_KEY or the PEM path)")
        return 1

    v = PolymarketUSVenue(settings.qcex)
    candidates = {
        "BALANCE": [
            "/v1/account/balances", "/v1/account/balance", "/v1/account",
            "/v1/balances", "/v1/balance", "/v1/portfolio/balances",
        ],
        "POSITIONS": [
            "/v1/portfolio/positions", "/v1/positions", "/v1/account/positions",
        ],
    }

    async def _probe():
        client = v._api()
        for label, paths in candidates.items():
            print(f"\n== {label} ==")
            for path in paths:
                await v._limiter.wait()
                try:
                    resp = await client.get(path, headers=v._auth_headers("GET", path))
                except Exception as exc:
                    print(f"  {path}  ERROR {exc}")
                    continue
                marker = "  <-- 200 OK" if resp.status_code == 200 else ""
                print(f"  {resp.status_code:>3}  {path}{marker}")
                if resp.status_code == 200:
                    print(f"       body: {resp.text[:600]}")
        await v.aclose()

    asyncio.run(_probe())
    print("\nPaste the 200-OK path(s) + body above and I'll lock the parser.")
    return 0


def edge_report(settings: Settings, *, hours: float = 24.0) -> int:
    """Summarize logged edge observations (what the streamer saw + did) over a window.

    Turns the soak into a distribution: how many actionable edges appeared, how big,
    and what happened to them (executed / settling-phantom / edge-gone / not-open).
    """
    store = Store(settings.db_path)
    try:
        since = time.time() - hours * 3600
        rows = store.conn.execute(
            """SELECT outcome, COUNT(*) n, AVG(edge) avg_edge, MAX(edge) max_edge,
                      AVG(size) avg_size
               FROM edge_observations WHERE ts >= ? GROUP BY outcome ORDER BY n DESC""",
            (since,),
        ).fetchall()
        total = sum(r["n"] for r in rows)
        print(f"edge observations in the last {hours:g}h: {total}\n")
        if not total:
            print("  (none yet — let it soak, ideally across live games)")
            return 0
        print(f"  {'outcome':<26}{'count':>6}{'avg_edge':>10}{'max_edge':>10}{'avg_size':>10}")
        for r in rows:
            print(f"  {r['outcome']:<26}{r['n']:>6}{r['avg_edge']:>10.3f}"
                  f"{r['max_edge']:>10.3f}{r['avg_size']:>10.1f}")
        locked = store.conn.execute(
            "SELECT COUNT(*) n FROM edge_observations WHERE ts>=? AND outcome='SUCCESS'",
            (since,),
        ).fetchone()["n"]
        settling = sum(r["n"] for r in rows if r["outcome"] == "skip_settling")
        print(f"\n  locked (SUCCESS): {locked}   settling-phantom: {settling}   "
              f"other: {total - locked - settling}")
        print("  -> 'skip_settling' + 'edge_gone_after_depth' are noise; SUCCESS/UNWOUND "
              "are real fires. A healthy venue pair shows few genuine edges.")
        return 0
    finally:
        store.close()


def compare_filters(settings: Settings, limit: int = 40) -> int:
    """SHADOW report: the structured fingerprint matcher vs the live filter.

    Over every LLM-confirmed verdict (same_event=1), compare which pairs the current
    live filter (store.confirmed_pairs) keeps against which the fingerprint matcher
    (are_complementary) keeps. Prints the ADDS (fingerprint keeps, live drops — the
    volume the rework would unlock) and REMOVES (live keeps, fingerprint drops — a
    precision change to eyeball). Trades nothing; changes nothing.
    """
    from collections import Counter

    from bot.matching.fingerprint import (
        complement_reason, from_kalshi, from_polymarket,
    )

    store = Store(settings.db_path)
    try:
        rows = store.conn.execute(
            """SELECT v.venue_a, v.market_a, v.venue_b, v.market_b, v.confidence,
                      ma.title AS title_a, mb.title AS title_b
               FROM match_verdicts v
               LEFT JOIN markets ma ON ma.venue=v.venue_a AND ma.market_id=v.market_a
               LEFT JOIN markets mb ON mb.venue=v.venue_b AND mb.market_id=v.market_b
               WHERE v.same_event=1""",
        ).fetchall()
        live_keys = {(r[0], r[1], r[2], r[3]) for r in store.confirmed_pairs()}

        def fp(venue, mid, title):
            return (from_kalshi(mid, title) if venue == "kalshi"
                    else from_polymarket(mid, title))

        adds, removes, both, reasons = [], [], 0, Counter()
        new_kept = 0
        for r in rows:
            key = (r["venue_a"], r["market_a"], r["venue_b"], r["market_b"])
            reason = complement_reason(
                fp(r["venue_a"], r["market_a"], r["title_a"] or ""),
                fp(r["venue_b"], r["market_b"], r["title_b"] or ""),
            )
            keep_new = reason == "ok"
            keep_old = key in live_keys
            new_kept += keep_new
            reasons[reason if not keep_new else "ok"] += 1
            if keep_new and keep_old:
                both += 1
            elif keep_new and not keep_old:
                adds.append(r)
            elif keep_old and not keep_new:
                removes.append((r, reason))

        print(f"confirmed verdicts (same_event=1): {len(rows)}")
        print(f"  live filter keeps:        {len(live_keys)}")
        print(f"  fingerprint keeps:        {new_kept}")
        print(f"  both agree (intersection): {both}")
        print(f"  ADDS (fingerprint only):   {len(adds)}   <- volume the rework unlocks")
        print(f"  REMOVES (live only):       {len(removes)}   <- eyeball for precision\n")

        print("reject reasons across all confirmed verdicts:")
        for reason, n in reasons.most_common(15):
            print(f"  {n:>5}  {reason}")

        def _show(label, items, get_reason=None):
            print(f"\n== {label} (showing {min(limit, len(items))} of {len(items)}) ==")
            for it in items[:limit]:
                r = it[0] if get_reason else it
                extra = f"   [{it[1]}]" if get_reason else ""
                print(f"  [{r['venue_a']}] {r['market_a']}  {r['title_a']}")
                print(f"  [{r['venue_b']}] {r['market_b']}  {r['title_b']}{extra}\n")

        _show("ADDS — fingerprint keeps, live drops", adds)
        _show("REMOVES — live keeps, fingerprint drops", removes, get_reason=True)
        return 0
    finally:
        store.close()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Live read-only DRY_RUN arbitrage monitor")
    p.add_argument("--check-llm", action="store_true",
                   help="probe the local LLM (LLM_BASE_URL) and exit")
    p.add_argument("--check-ws", action="store_true",
                   help="probe each venue's market WebSocket and exit")
    p.add_argument("--inspect-matches", action="store_true",
                   help="print cached match verdicts (with titles) to audit the "
                        "matcher, then exit")
    p.add_argument("--count-markets", action="store_true",
                   help="pull every open market from each venue, print the counts, "
                        "and exit (no matching)")
    p.add_argument("--find", metavar="TERM", default=None,
                   help="search both venues' scans for a team/player and show whether "
                        "the bot sees + can pair it (diagnose a live match not trading)")
    p.add_argument("--probe-account", action="store_true",
                   help="probe Polymarket US account/portfolio endpoints (read-only) "
                        "to discover the real balance/positions paths")
    p.add_argument("--compare-filters", action="store_true",
                   help="SHADOW: compare the structured fingerprint matcher vs the live "
                        "filter over cached verdicts (adds/removes); trades nothing")
    p.add_argument("--edge-report", action="store_true",
                   help="summarize logged edge observations (frequency/size/outcome) "
                        "over the last --hours and exit")
    p.add_argument("--hours", type=float, default=24.0,
                   help="time window for --edge-report (default 24h)")
    p.add_argument("--show-book", metavar="VENUE:MARKET", default=None,
                   help="fetch and print one market's real order book (sized quote)")
    p.add_argument("--probe-ws-book", metavar="SLUG", default=None,
                   help="diagnose the Polymarket WS-vs-REST top-of-book gap: print the REST "
                        "/book top then the first raw WS MARKET_DATA messages for SLUG")
    p.add_argument("--test-order", metavar="VENUE:MARKET", default=None,
                   help="place ONE test order to verify the live buying path; default "
                        "is a no-fill canary ($0 spent). Add --fill for a real buy.")
    p.add_argument("--fill", action="store_true",
                   help="with --test-order, actually buy 1 contract at the ask "
                        "(REAL money, REAL position you must close manually)")
    p.add_argument("--diagnose-event", metavar="SUBSTR", default=None,
                   help="root-cause a MISSING match: for every scanned market whose "
                        "title/id contains SUBSTR, print its fingerprint and every "
                        "cross-venue complement reason + whether it's cached (read-only)")
    p.add_argument("--show-rejected", action="store_true",
                   help="with --inspect-matches, also list rejected (non-match) pairs")
    p.add_argument("--tradeable-only", action="store_true",
                   help="with --inspect-matches, list ONLY the post-filter set the "
                        "streamer will trade (confidence floor + fan-out guard)")
    p.add_argument("--recheck-matches", action="store_true",
                   help="re-judge the current tradeable set with the live prompt+model "
                        "(no cache writes); test the matcher before a full re-seed")
    p.add_argument("--watchlist", action="store_true",
                   help="print the LIVE streaming watchlist (cached pairs whose both "
                        "legs are quotable right now) and exit")
    p.add_argument("--once", action="store_true", help="run a single cycle and exit")
    p.add_argument("--interval", type=float, default=15.0, help="seconds between cycles")
    p.add_argument("--limit", type=int, default=50,
                   help="max markets to scan per venue (TOTAL across paginated pages; "
                        "Kalshi pages at 1000, so >1000 follows the cursor)")
    p.add_argument("--match-threshold", type=float, default=None,
                   help="title-match cutoff to shortlist a cross-venue pair "
                        "(default 0.5 lexical, 0.65 with --embed)")
    p.add_argument("--llm", action="store_true",
                   help="confirm cross-venue matches with the local LLM (default off)")
    p.add_argument("--embed", action="store_true",
                   help="match titles semantically via the local embedding model "
                        "(recommended; far better than lexical for reworded events)")
    p.add_argument("--min-edge", type=float, default=None,
                   help="min per-contract edge to record (default from settings)")
    p.add_argument("--max-confirms", type=int, default=50,
                   help="max NEW LLM match-confirmations per cycle (cached pairs are "
                        "free; the rest are confirmed over later cycles)")
    p.add_argument("--max-resolve-gap-days", type=float, default=3.0,
                   help="reject cross-venue pairs whose resolution dates differ by "
                        "more than this many days (settlement-mismatch guard)")
    p.add_argument("--close-within-days", type=float, default=None,
                   help="TARGETED scan: only scan markets closing within this many days "
                        "(the live/imminent set). Default from SCAN_CLOSE_WITHIN_DAYS "
                        "(0 = scan all). Keeps the scan small + complete for today's games.")
    p.add_argument("--live", action="store_true",
                   help="PLACE REAL ORDERS on confirmed arbs (needs trading creds; "
                        "point base URLs at the sandbox/demo first). Default: monitor only.")
    p.add_argument("--stream", action="store_true",
                   help="STREAMING LIVE: slow match loop + fast WebSocket execution on "
                        "confirmed pairs (real orders; needs trading creds). Implies --embed/--llm.")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.check_llm:
        raise SystemExit(check_llm(load_settings()))

    if args.check_ws:
        raise SystemExit(check_ws(load_settings()))

    if args.inspect_matches:
        raise SystemExit(inspect_matches(
            load_settings(), show_rejected=args.show_rejected,
            tradeable_only=args.tradeable_only, limit=args.limit,
        ))

    if args.diagnose_event:
        raise SystemExit(diagnose_event(load_settings(), args.diagnose_event, limit=args.limit))

    if args.count_markets:
        raise SystemExit(count_markets(load_settings()))

    if args.find:
        raise SystemExit(find_markets(load_settings(), args.find))

    if args.probe_account:
        raise SystemExit(probe_account(load_settings()))

    if args.compare_filters:
        raise SystemExit(compare_filters(load_settings(), limit=args.limit))

    if args.edge_report:
        raise SystemExit(edge_report(load_settings(), hours=args.hours))

    if args.show_book:
        raise SystemExit(show_book(load_settings(), args.show_book))

    if args.probe_ws_book:
        raise SystemExit(probe_ws_book(load_settings(), args.probe_ws_book))

    if args.recheck_matches:
        raise SystemExit(recheck_matches(load_settings(), limit=args.limit))

    if args.watchlist:
        raise SystemExit(show_watchlist(load_settings()))

    if args.test_order:
        raise SystemExit(test_order(load_settings(), args.test_order, do_fill=args.fill))

    threshold = args.match_threshold
    if threshold is None:
        threshold = 0.65 if (args.embed or args.stream) else 0.5

    if args.stream:
        interval = args.interval if args.interval != 15.0 else 300.0  # streaming default 5m
        asyncio.run(stream(
            refresh_interval=interval, limit=args.limit, use_llm=True, use_embed=True,
            match_threshold=threshold, min_edge=args.min_edge,
            max_confirms=args.max_confirms, max_resolve_gap_days=args.max_resolve_gap_days,
            close_within_days=args.close_within_days,
        ))
        return

    asyncio.run(run(
        once=args.once, interval=args.interval, limit=args.limit,
        use_llm=args.llm, use_embed=args.embed, match_threshold=threshold,
        min_edge=args.min_edge, max_confirms=args.max_confirms,
        max_resolve_gap_days=args.max_resolve_gap_days, live=args.live,
        close_within_days=args.close_within_days,
    ))


if __name__ == "__main__":
    main()
