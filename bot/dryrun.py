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
    names = list(quotes_by_venue)
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
                executor=executor,
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


async def build_watchlist(cached, scanned, venues):
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
        try:
            q = await v.fetch_quote(RawMarket(market_id=mid, title="", raw={}))
        except Exception as exc:
            log.info("watchlist probe %s:%s not live (%s)", vn, mid, exc)
            continue
        if q is not None:
            live.add((vn, mid))

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
    )
    engine = StreamingEngine(executor=executor, fee_models=fee_models, min_edge=min_edge)

    complete_fn = make_complete_fn(settings.llm) if use_llm else None
    embed_fn = make_embed_fn(settings.llm) if use_embed else None

    async def refresh_specs():
        # Discovery cycle: scans markets + confirms/caches new pairs (embeddings/LLM).
        res = await run_cycle(
            venues, store=store, risk=RiskManager(settings.risk), fee_models=fee_models,
            min_edge=min_edge, match_threshold=match_threshold, complete_fn=complete_fn,
            limit=limit, embed_fn=embed_fn, max_confirms=max_confirms,
            max_resolve_gap_days=max_resolve_gap_days, executor=None,
        )
        return await build_watchlist(store.confirmed_pairs(), res.scanned, venues)

    async def feed_private(v):
        try:
            async for ev in v.stream_private():
                await tracker.apply(ev)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("private stream %s ended: %s", v.name, exc)

    private_tasks = [asyncio.create_task(feed_private(v)) for v in venues]
    log.warning("STREAMING LIVE — real orders on confirmed pairs (max %s ct/order, "
                "caps $%.0f/$%.0f/$%.0f)", settings.risk.max_order_contracts,
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

        tradeable_pairs = store.confirmed_pairs()
        if tradeable_only:
            # Exactly what the streamer will trade — audit this before going live.
            print(f"{len(tradeable_pairs)} TRADEABLE pairs (confidence+fan-out filtered):\n")
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
    p.add_argument("--test-order", metavar="VENUE:MARKET", default=None,
                   help="place ONE test order to verify the live buying path; default "
                        "is a no-fill canary ($0 spent). Add --fill for a real buy.")
    p.add_argument("--fill", action="store_true",
                   help="with --test-order, actually buy 1 contract at the ask "
                        "(REAL money, REAL position you must close manually)")
    p.add_argument("--show-rejected", action="store_true",
                   help="with --inspect-matches, also list rejected (non-match) pairs")
    p.add_argument("--tradeable-only", action="store_true",
                   help="with --inspect-matches, list ONLY the post-filter set the "
                        "streamer will trade (confidence floor + fan-out guard)")
    p.add_argument("--recheck-matches", action="store_true",
                   help="re-judge the current tradeable set with the live prompt+model "
                        "(no cache writes); test the matcher before a full re-seed")
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

    if args.count_markets:
        raise SystemExit(count_markets(load_settings()))

    if args.recheck_matches:
        raise SystemExit(recheck_matches(load_settings(), limit=args.limit))

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
        ))
        return

    asyncio.run(run(
        once=args.once, interval=args.interval, limit=args.limit,
        use_llm=args.llm, use_embed=args.embed, match_threshold=threshold,
        min_edge=args.min_edge, max_confirms=args.max_confirms,
        max_resolve_gap_days=args.max_resolve_gap_days, live=args.live,
    ))


if __name__ == "__main__":
    main()
