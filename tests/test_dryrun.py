"""Tests for the two-phase live DRY_RUN runner using injected stub venues (no net)."""

import asyncio
import dataclasses

from bot.config.settings import Settings
from bot.data.store import Store
from bot.dryrun import run, run_cycle
from bot.execution.risk import RiskLimits, RiskManager
from bot.fees import ZeroFeeModel
from bot.models import MarketQuote


class StubVenue:
    """Duck-typed venue: phase-1 ``scan_quotes`` (price-only) + phase-2 ``fetch_quote``
    (sized). It stores sized quotes and zeroes the sizes for the cheap scan, so tests
    exercise the real two-phase path."""

    def __init__(self, name, quotes: dict[str, MarketQuote]):
        self.name = name
        self.fee_model = ZeroFeeModel()
        self._quotes = quotes

    async def scan_quotes(self, limit=500, *, max_close_ts=None):
        return [
            dataclasses.replace(q, yes_ask_size=0.0, no_ask_size=0.0)
            for q in self._quotes.values()
        ]

    async def fetch_quote(self, market):
        return self._quotes.get(market.market_id)


def mq(venue, mid, title, **kw):
    return MarketQuote(venue=venue, market_id=mid, title=title, **kw)


def generous_risk():
    return RiskManager(RiskLimits(max_position_per_market=1e9, max_total_exposure=1e12))


def run_cycle_kw(venues, complete_fn=None, store=None, min_edge=0.01, threshold=0.3,
                 embed_fn=None, max_confirms=50, use_fingerprint=False):
    return asyncio.run(run_cycle(
        venues, store=store, risk=generous_risk(),
        fee_models={v.name: ZeroFeeModel() for v in venues},
        min_edge=min_edge, match_threshold=threshold, complete_fn=complete_fn,
        limit=50, embed_fn=embed_fn, max_confirms=max_confirms,
        use_fingerprint=use_fingerprint,
    ))


def test_close_within_days_passes_max_close_ts_to_scan():
    # close_within_days > 0 -> run_cycle computes a max_close_ts (now + days) and hands
    # it to each venue's scan_quotes; 0 -> None (scan all).
    import time as _time

    captured = []

    class RecordingVenue(StubVenue):
        async def scan_quotes(self, limit=500, *, max_close_ts=None):
            captured.append(max_close_ts)
            return []

    before = _time.time()
    asyncio.run(run_cycle(
        [RecordingVenue("kalshi", {})], store=None, risk=generous_risk(),
        fee_models={"kalshi": ZeroFeeModel()}, min_edge=0.01, match_threshold=0.3,
        complete_fn=None, limit=50, close_within_days=2.0,
    ))
    after = _time.time()
    assert len(captured) == 1 and captured[0] is not None
    assert int(before + 2 * 86400) <= captured[0] <= int(after + 2 * 86400)

    captured.clear()
    asyncio.run(run_cycle(
        [RecordingVenue("kalshi", {})], store=None, risk=generous_risk(),
        fee_models={"kalshi": ZeroFeeModel()}, min_edge=0.01, match_threshold=0.3,
        complete_fn=None, limit=50, close_within_days=0.0,
    ))
    assert captured == [None]                              # disabled -> no window


def test_kalshi_close_within_days_windows_kalshi_only():
    # The KALSHI-ONLY window scopes Kalshi to imminent markets with an UNBOUNDED limit (its
    # per-game lines sit past the --limit cap), while Polymarket keeps the shared limit and
    # NO window (its per-game endDates are far-future, so a window would drop them).
    import time as _time

    seen = {}

    class RecordingVenue(StubVenue):
        async def scan_quotes(self, limit=500, *, max_close_ts=None):
            seen[self.name] = (limit, max_close_ts)
            return []

    before = _time.time()
    asyncio.run(run_cycle(
        [RecordingVenue("kalshi", {}), RecordingVenue("polymarket_us", {})],
        store=None, risk=generous_risk(),
        fee_models={"kalshi": ZeroFeeModel(), "polymarket_us": ZeroFeeModel()},
        min_edge=0.01, match_threshold=0.3, complete_fn=None, limit=5000,
        close_within_days=0.0, kalshi_close_within_days=30.0,
    ))
    after = _time.time()
    k_limit, k_window = seen["kalshi"]
    p_limit, p_window = seen["polymarket_us"]
    assert k_limit == 0 and k_window is not None                 # kalshi: unbounded + windowed
    assert int(before + 30 * 86400) <= k_window <= int(after + 30 * 86400)
    assert p_limit == 5000 and p_window is None                  # poly: shared limit, NO window


def test_max_confirms_caps_llm_calls_per_cycle():
    # Many candidate pairs, all with a price edge; cap LLM confirmations at 2.
    kalshi = {f"K{i}": mq("kalshi", f"K{i}", f"Team{i} game", yes_ask=0.40, yes_ask_size=100, no_ask=0.65, no_ask_size=100) for i in range(5)}
    poly = {f"P{i}": mq("polymarket_us", f"P{i}", f"Team{i} game", yes_ask=0.62, yes_ask_size=100, no_ask=0.55, no_ask_size=60) for i in range(5)}
    calls = []
    fake = lambda p: (calls.append(1), '{"same_event": false, "confidence": 0.1}')[1]

    result = run_cycle_kw(
        [StubVenue("kalshi", kalshi), StubVenue("polymarket_us", poly)],
        complete_fn=fake, store=Store(":memory:"), max_confirms=2,
    )
    assert result.llm_confirms == 2          # capped
    assert len(calls) == 2                    # the model was called exactly twice


def test_discovery_off_skips_matching_but_still_scans():
    # match_cross_venue=False: no embedding/LLM discovery (the fingerprint sweep is
    # authoritative), but the scan still upserts markets so the sweep has data.
    kalshi = {f"K{i}": mq("kalshi", f"K{i}", f"Team{i} game", yes_ask=0.40, yes_ask_size=100, no_ask=0.65, no_ask_size=100) for i in range(3)}
    poly = {f"P{i}": mq("polymarket_us", f"P{i}", f"Team{i} game", yes_ask=0.62, yes_ask_size=100, no_ask=0.55, no_ask_size=60) for i in range(3)}
    calls = []
    fake = lambda p: (calls.append(1), '{"same_event": true, "confidence": 0.9}')[1]
    store = Store(":memory:")
    result = asyncio.run(run_cycle(
        [StubVenue("kalshi", kalshi), StubVenue("polymarket_us", poly)],
        store=store, risk=generous_risk(),
        fee_models={"kalshi": ZeroFeeModel(), "polymarket_us": ZeroFeeModel()},
        min_edge=0.01, match_threshold=0.3, complete_fn=fake, limit=50,
        match_cross_venue=False,
    ))
    assert calls == []                                # the LLM was never called
    assert result.candidate_pairs == 0                # no cross-venue matching ran
    assert store.conn.execute(                        # markets still scanned + upserted
        "SELECT COUNT(*) c FROM markets").fetchone()["c"] == 6


def test_embedding_matcher_pairs_reworded_titles():
    # Lexically dissimilar titles, but the embedder maps them to the same vector.
    ka = mq("kalshi", "K1", "Fed lowers its benchmark rate by March", yes_ask=0.40, yes_ask_size=100, no_ask=0.65, no_ask_size=100)
    pa = mq("polymarket_us", "P1", "March FOMC rate cut", yes_ask=0.62, yes_ask_size=100, no_ask=0.55, no_ask_size=60)
    vecs = {ka.title: [1.0, 0.0], pa.title: [1.0, 0.0]}
    embed_fn = lambda texts: [vecs[t] for t in texts]
    fake = lambda p: '{"same_event": true, "confidence": 0.95}'

    result = run_cycle_kw(
        [StubVenue("kalshi", {"K1": ka}), StubVenue("polymarket_us", {"P1": pa})],
        complete_fn=fake, embed_fn=embed_fn, threshold=0.8,
    )
    assert result.candidate_pairs >= 1
    assert len(result.cross.actionable) >= 1


def test_bundle_arb_detected_via_two_phase():
    # yes_ask + no_ask = 0.95 < 1 -> bundle. Sizing comes from the phase-2 fetch.
    kb = mq("kalshi", "B1", "Bundle market", yes_ask=0.40, yes_ask_size=50, no_ask=0.55, no_ask_size=50)
    store = Store(":memory:")
    result = asyncio.run(run(
        once=True, venues=[StubVenue("kalshi", {"B1": kb})],
        store=store, settings=Settings(), min_edge=0.01,
    ))
    assert len(result.bundle_opps) == 1
    assert round(result.bundle_opps[0].edge_per_contract, 4) == 0.05
    assert result.bundle_opps[0].max_contracts == 50          # real size from phase 2
    assert result.deep_fetches == 1                            # only the candidate fetched
    assert store.conn.execute("SELECT COUNT(*) c FROM opportunities").fetchone()["c"] == 1
    store.close()


def test_no_deep_fetch_without_candidates():
    # A market with no edge (sum > 1) is never deep-fetched.
    flat = mq("kalshi", "F1", "Flat", yes_ask=0.55, yes_ask_size=10, no_ask=0.55, no_ask_size=10)
    result = run_cycle_kw([StubVenue("kalshi", {"F1": flat})])
    assert result.bundle_opps == []
    assert result.deep_fetches == 0


def test_cross_venue_confirmed_with_llm():
    ka = mq("kalshi", "K1", "Fed cuts rates March 2026", yes_ask=0.40, yes_ask_size=100, no_ask=0.65, no_ask_size=100)
    pa = mq("polymarket_us", "P1", "Fed cuts interest rates March 2026", yes_ask=0.62, yes_ask_size=100, no_ask=0.55, no_ask_size=60)
    venues = [StubVenue("kalshi", {"K1": ka}), StubVenue("polymarket_us", {"P1": pa})]
    store = Store(":memory:")
    fake = lambda prompt: '{"same_event": true, "confidence": 0.95, "rationale": "same"}'

    result = run_cycle_kw(venues, complete_fn=fake, store=store)

    assert result.candidate_pairs >= 1
    assert len(result.cross.detected) >= 1
    assert len(result.cross.actionable) >= 1
    assert result.deep_fetches == 2                            # both legs fetched
    assert store.get_verdict("polymarket_us", "P1", "kalshi", "K1") is not None
    store.close()


def test_cross_venue_skipped_without_llm():
    ka = mq("kalshi", "K1", "Fed cuts rates March 2026", yes_ask=0.40, yes_ask_size=100, no_ask=0.65, no_ask_size=100)
    pa = mq("polymarket_us", "P1", "Fed cuts interest rates March 2026", yes_ask=0.62, yes_ask_size=100, no_ask=0.55, no_ask_size=60)
    venues = [StubVenue("kalshi", {"K1": ka}), StubVenue("polymarket_us", {"P1": pa})]

    result = run_cycle_kw(venues, complete_fn=None)
    # Price edge surfaces the candidate, but nothing is tradeable without confirmation.
    assert result.candidate_pairs >= 1
    assert result.cross.detected == []


def test_confirmed_no_edge_pair_is_watched_but_not_acted():
    # Same event, no current price edge (0.55+0.55 > 1). It must still be confirmed
    # and added to the streaming watchlist (confirmed_pairs), but NOT traded.
    ka = mq("kalshi", "K1", "Fed cut March", yes_ask=0.55, yes_ask_size=100, no_ask=0.55, no_ask_size=100)
    pa = mq("polymarket_us", "P1", "Fed cut March", yes_ask=0.55, yes_ask_size=100, no_ask=0.55, no_ask_size=100)
    called = []
    fake = lambda p: called.append(1) or '{"same_event": true, "confidence": 0.95}'
    result = run_cycle_kw([StubVenue("kalshi", {"K1": ka}), StubVenue("polymarket_us", {"P1": pa})], complete_fn=fake)
    assert result.candidate_pairs >= 1
    assert len(called) >= 1                      # confirmation runs regardless of edge
    assert len(result.confirmed_pairs) >= 1      # on the watchlist for streaming
    assert result.cross.detected == []           # but no edge -> not acted
    assert result.deep_fetches == 0


def test_date_gate_still_skips_before_llm():
    # Far-apart resolution dates are rejected before any LLM call (unchanged).
    JUN, SEP = 1781725200.0, 1790683200.0
    ka = mq("kalshi", "K1", "X", yes_ask=0.40, yes_ask_size=100, no_ask=0.65, no_ask_size=100, close_time=JUN)
    pa = mq("polymarket_us", "P1", "X", yes_ask=0.62, yes_ask_size=100, no_ask=0.55, no_ask_size=60, close_time=SEP)
    called = []
    fake = lambda p: called.append(1) or '{"same_event": true, "confidence": 0.95}'
    result = run_cycle_kw([StubVenue("kalshi", {"K1": ka}), StubVenue("polymarket_us", {"P1": pa})], complete_fn=fake)
    assert result.candidate_pairs == 0 and called == [] and result.confirmed_pairs == []


def test_cross_pair_without_list_prices_still_confirmed():
    # Polymarket leg has no list price (yes_ask/no_ask None) — the cheap price gate
    # must NOT drop it; price/edge is resolved from the phase-2 fetch.
    ka_scan = mq("kalshi", "K1", "Fed cut March", yes_ask=0.40, yes_ask_size=0, no_ask=0.65, no_ask_size=0)
    pa_scan = mq("polymarket_us", "P1", "Fed cut March", yes_ask=None, no_ask=None)  # no list price
    # Sized (phase-2) quotes carry the real prices/sizes.
    ka = mq("kalshi", "K1", "Fed cut March", yes_ask=0.40, yes_ask_size=100, no_ask=0.65, no_ask_size=100)
    pa = mq("polymarket_us", "P1", "Fed cut March", yes_ask=0.62, yes_ask_size=100, no_ask=0.55, no_ask_size=60)

    class TwoPhaseStub(StubVenue):
        def __init__(self, name, scan_q, deep_q):
            super().__init__(name, {deep_q.market_id: deep_q})
            self._scan = scan_q
        async def scan_quotes(self, limit=500, *, max_close_ts=None):
            return [self._scan]

    venues = [TwoPhaseStub("kalshi", ka_scan, ka), TwoPhaseStub("polymarket_us", pa_scan, pa)]
    fake = lambda p: '{"same_event": true, "confidence": 0.95}'
    result = run_cycle_kw(venues, complete_fn=fake)
    assert result.candidate_pairs >= 1            # reached the LLM despite missing list price
    assert len(result.cross.actionable) >= 1      # priced + detected from phase-2 fetch


def test_resolution_date_gate_rejects_game_vs_championship():
    # The exact false positive seen live: a single game (Jun 16) vs a World Series
    # futures market (Sep 27). Same teams/words, but resolution dates ~100 days apart.
    JUN = 1781725200.0  # 2026-06-16
    SEP = 1790683200.0  # 2026-09-27 (~103 days later)
    ka = mq("kalshi", "K1", "LA Angels vs Arizona Winner? - LA Angels",
            yes_ask=0.40, yes_ask_size=100, no_ask=0.65, no_ask_size=100, close_time=JUN)
    pa = mq("polymarket_us", "P1", "MLB Champion - LA Angels",
            yes_ask=0.62, yes_ask_size=100, no_ask=0.55, no_ask_size=60, close_time=SEP)
    called = []
    fake = lambda p: called.append(1) or '{"same_event": true, "confidence": 0.95}'

    result = run_cycle_kw(
        [StubVenue("kalshi", {"K1": ka}), StubVenue("polymarket_us", {"P1": pa})],
        complete_fn=fake,
    )
    assert result.candidate_pairs == 0     # gated out before the LLM
    assert called == []                     # LLM never consulted for the mismatch
    assert result.cross.detected == []


def test_resolution_date_gate_allows_same_date():
    SEP = 1790683200.0
    ka = mq("kalshi", "K1", "World Series Champion - LA Dodgers",
            yes_ask=0.40, yes_ask_size=100, no_ask=0.65, no_ask_size=100, close_time=SEP)
    pa = mq("polymarket_us", "P1", "MLB Champion - LA Dodgers",
            yes_ask=0.62, yes_ask_size=100, no_ask=0.55, no_ask_size=60, close_time=SEP + 3600)
    fake = lambda p: '{"same_event": true, "confidence": 0.95}'
    result = run_cycle_kw(
        [StubVenue("kalshi", {"K1": ka}), StubVenue("polymarket_us", {"P1": pa})],
        complete_fn=fake,
    )
    assert result.candidate_pairs >= 1
    assert len(result.cross.actionable) >= 1


def test_scan_failure_isolated_per_venue():
    class DeadVenue(StubVenue):
        async def scan_quotes(self, limit=500, *, max_close_ts=None):
            raise RuntimeError("venue down")

    kb = mq("kalshi", "B1", "Bundle", yes_ask=0.40, yes_ask_size=50, no_ask=0.55, no_ask_size=50)
    venues = [DeadVenue("polymarket_us", {}), StubVenue("kalshi", {"B1": kb})]
    result = run_cycle_kw(venues)
    assert len(result.bundle_opps) == 1                        # healthy venue still works


def test_deep_fetch_failure_does_not_crash():
    class FlakyDeep(StubVenue):
        async def fetch_quote(self, market):
            raise RuntimeError("transient")

    kb = mq("kalshi", "B1", "Bundle", yes_ask=0.40, yes_ask_size=50, no_ask=0.55, no_ask_size=50)
    result = run_cycle_kw([FlakyDeep("kalshi", {"B1": kb})])
    assert result.deep_fetches == 1       # attempted
    assert result.bundle_opps == []       # but no sized quote -> no recorded arb


def test_scanned_set_holds_every_live_market():
    # ``scanned`` is the (venue, market_id) universe the streaming watchlist
    # intersects cached pairs against; a cached pair drops off the watchlist iff
    # one of its legs is absent here. This is the root-cause check for "0 of N live".
    ka = mq("kalshi", "K1", "A", yes_ask=0.55, yes_ask_size=10, no_ask=0.55, no_ask_size=10)
    pa = mq("polymarket_us", "P1", "B", yes_ask=0.55, yes_ask_size=10, no_ask=0.55, no_ask_size=10)
    result = run_cycle_kw([StubVenue("kalshi", {"K1": ka}), StubVenue("polymarket_us", {"P1": pa})])
    assert result.scanned == {("kalshi", "K1"), ("polymarket_us", "P1")}


def test_build_watchlist_keeps_pair_when_both_legs_scanned():
    from bot.dryrun import build_watchlist

    cached = [("kalshi", "K1", "polymarket_us", "P1", "ufc")]
    scanned = {("kalshi", "K1"), ("polymarket_us", "P1")}
    venues = [StubVenue("kalshi", {}), StubVenue("polymarket_us", {})]
    out = asyncio.run(build_watchlist(cached, scanned, venues))
    assert len(out) == 1 and out[0].event_key == "ufc"


def test_build_watchlist_drops_thin_polymarket_leg():
    # min_poly_depth filter: a pair whose Polymarket leg lacks takeable depth is dropped;
    # a deep one is kept. Points the watchlist at markets the bot can actually hedge.
    from bot.dryrun import build_watchlist

    thin = mq("polymarket_us", "P1", "x", yes_ask=0.5, yes_ask_size=2, no_ask=0.5, no_ask_size=2)
    deep = mq("polymarket_us", "P2", "y", yes_ask=0.5, yes_ask_size=50, no_ask=0.5, no_ask_size=50)
    cached = [("kalshi", "K1", "polymarket_us", "P1", "thin"),
              ("kalshi", "K2", "polymarket_us", "P2", "deep")]
    scanned = {("kalshi", "K1"), ("kalshi", "K2"),
               ("polymarket_us", "P1"), ("polymarket_us", "P2")}
    venues = [StubVenue("kalshi", {}), StubVenue("polymarket_us", {"P1": thin, "P2": deep})]
    out = asyncio.run(build_watchlist(cached, scanned, venues, min_poly_depth=10))
    assert {p.event_key for p in out} == {"deep"}


def test_build_watchlist_probes_leg_outside_scan_window():
    # The exact live bug: the Kalshi leg sits past the discovery --limit, so it's
    # absent from ``scanned``. A targeted fetch_quote must rescue it onto the watchlist.
    from bot.dryrun import build_watchlist

    ka = mq("kalshi", "K1", "UFC fight", yes_ask=0.4, yes_ask_size=10, no_ask=0.6, no_ask_size=10)
    cached = [("kalshi", "K1", "polymarket_us", "P1", "ufc")]
    scanned = {("polymarket_us", "P1")}                 # Kalshi leg NOT in scan
    venues = [StubVenue("kalshi", {"K1": ka}), StubVenue("polymarket_us", {})]
    out = asyncio.run(build_watchlist(cached, scanned, venues))
    assert len(out) == 1                                # probe rescued the missing leg


def test_build_watchlist_drops_pair_when_probe_finds_nothing():
    # Leg absent from scan AND the probe returns no quote (closed/renamed) -> dropped.
    from bot.dryrun import build_watchlist

    cached = [("kalshi", "K1", "polymarket_us", "P1", "ufc")]
    scanned = {("polymarket_us", "P1")}
    venues = [StubVenue("kalshi", {}), StubVenue("polymarket_us", {})]  # K1 not fetchable
    out = asyncio.run(build_watchlist(cached, scanned, venues))
    assert out == []


def test_build_watchlist_uses_status_keeps_open_drops_closed():
    # With is_open() available: an OPEN market with an empty book is kept; a CLOSED
    # market is dropped — independent of book depth.
    from bot.dryrun import build_watchlist

    class StatusVenue(StubVenue):
        def __init__(self, name, open_map):
            super().__init__(name, {})
            self._open = open_map

        async def is_open(self, mid):
            return self._open.get(mid)

    cached = [
        ("kalshi", "Kopen", "polymarket_us", "P1", "e1"),   # open both -> kept
        ("kalshi", "Kclosed", "polymarket_us", "P2", "e2"),  # kalshi closed -> dropped
    ]
    venues = [
        StatusVenue("kalshi", {"Kopen": True, "Kclosed": False}),
        StatusVenue("polymarket_us", {"P1": True, "P2": True}),
    ]
    out = asyncio.run(build_watchlist(cached, set(), venues))
    assert {p.market_a for p in out} == {"Kopen"}


def test_build_watchlist_keeps_market_when_status_unknown():
    # is_open() -> None (can't tell) must KEEP the market (no regression to dropping
    # live markets on uncertainty).
    from bot.dryrun import build_watchlist

    class UnknownVenue(StubVenue):
        async def is_open(self, mid):
            return None

    cached = [("kalshi", "K1", "polymarket_us", "P1", "e1")]
    venues = [UnknownVenue("kalshi", {}), UnknownVenue("polymarket_us", {})]
    out = asyncio.run(build_watchlist(cached, set(), venues))
    assert len(out) == 1


def test_build_watchlist_keeps_open_but_illiquid_market():
    # An open-but-illiquid market answers a quote with no asks yet (empty book). It
    # must STAY on the watchlist — it goes two-sided closer to game time, and execution
    # declines an empty book safely. (Conflating this with "settled" nuked the list.)
    from bot.dryrun import build_watchlist

    empty = mq("kalshi", "K1", "pre-match market", yes_ask=None, no_ask=None)
    cached = [("kalshi", "K1", "polymarket_us", "P1", "ufc")]
    scanned = {("polymarket_us", "P1")}
    venues = [StubVenue("kalshi", {"K1": empty}), StubVenue("polymarket_us", {})]
    out = asyncio.run(build_watchlist(cached, scanned, venues))
    assert len(out) == 1


def test_build_watchlist_survives_probe_error():
    from bot.dryrun import build_watchlist

    class FetchExplodes(StubVenue):
        async def fetch_quote(self, market):
            raise RuntimeError("404 not found")

    cached = [("kalshi", "K1", "polymarket_us", "P1", "ufc")]
    scanned = {("polymarket_us", "P1")}
    venues = [FetchExplodes("kalshi", {}), StubVenue("polymarket_us", {})]
    out = asyncio.run(build_watchlist(cached, scanned, venues))
    assert out == []                                    # error treated as not-live, no crash


def test_inspect_matches_lists_confirmed_with_titles(capsys):
    from bot.dryrun import inspect_matches

    s = Store(":memory:")
    s.upsert_market("kalshi", "K1", "Lima vs Borshchev")
    s.upsert_market("polymarket_us", "P1", "UFC: Lima vs Borshchev")
    s.cache_verdict("kalshi", "K1", "polymarket_us", "P1",
                    same_event=True, confidence=0.95, rationale="same fight")
    s.cache_verdict("kalshi", "K2", "polymarket_us", "P2",
                    same_event=False, confidence=0.2, rationale="different fights")

    # Point inspect_matches at our in-memory store by monkeypatching Store construction.
    import bot.dryrun as dr
    orig = dr.Store
    dr.Store = lambda path: s          # type: ignore[assignment]
    try:
        rc = inspect_matches(Settings(), show_rejected=True)
    finally:
        dr.Store = orig
    out = capsys.readouterr().out
    assert rc == 0
    assert "1 marked same-event" in out
    assert "TRADEABLE" in out
    assert "Lima vs Borshchev" in out
    assert "same fight" in out
    assert "rejected (not same event)" in out     # rejected section shown
    s.close()


def test_inspect_matches_tradeable_only_lists_exact_trade_set(capsys):
    from bot.dryrun import inspect_matches

    s = Store(":memory:")
    s.upsert_market("kalshi", "KXUFCFIGHT-26JUN20GANE-GANE", "Gane by KO")
    s.upsert_market("polymarket_us", "P1", "Gane by KO")
    s.cache_verdict("kalshi", "KXUFCFIGHT-26JUN20GANE-GANE", "polymarket_us", "P1",
                    same_event=True, confidence=1.0)
    # A fan-out market that must NOT appear in the tradeable-only view.
    for p in ("PA", "PB"):
        s.cache_verdict("kalshi", "KXUFCFIGHT-26JUN20FIELD-X", "polymarket_us", p,
                        same_event=True, confidence=1.0)

    import bot.dryrun as dr
    orig = dr.Store
    dr.Store = lambda path: s          # type: ignore[assignment]
    try:
        rc = inspect_matches(Settings(), tradeable_only=True)
    finally:
        dr.Store = orig
    out = capsys.readouterr().out
    assert rc == 0
    assert "1 TRADEABLE pairs" in out      # only the clean 1:1, fan-out excluded
    assert "Gane by KO" in out
    assert "Kfield" not in out
    s.close()


def test_show_book_prints_quote(capsys, monkeypatch):
    import bot.dryrun as dr
    from bot.dryrun import show_book

    ka = mq("kalshi", "K1", "m", yes_ask=0.4, yes_ask_size=30, no_ask=0.62, no_ask_size=20)
    venues = [StubVenue("kalshi", {"K1": ka})]
    monkeypatch.setattr(dr, "_build_venues", lambda s: venues)
    rc = show_book(Settings(), "kalshi:K1")
    out = capsys.readouterr().out
    assert rc == 0
    assert "yes_ask=0.4@30" in out and "no_ask=0.62@20" in out


def test_show_book_flags_empty(capsys, monkeypatch):
    import bot.dryrun as dr
    from bot.dryrun import show_book

    empty = mq("kalshi", "K1", "m", yes_ask=None, no_ask=None)
    monkeypatch.setattr(dr, "_build_venues", lambda s: [StubVenue("kalshi", {"K1": empty})])
    rc = show_book(Settings(), "kalshi:K1")
    out = capsys.readouterr().out
    assert rc == 0 and "EMPTY book" in out


def test_count_markets_reports_per_venue_and_total(capsys, monkeypatch):
    import bot.dryrun as dr
    from bot.dryrun import count_markets

    venues = [
        StubVenue("kalshi", {f"K{i}": mq("kalshi", f"K{i}", f"m{i}") for i in range(3)}),
        StubVenue("polymarket_us", {"P1": mq("polymarket_us", "P1", "m")}),
    ]
    monkeypatch.setattr(dr, "_build_venues", lambda settings: venues)
    rc = count_markets(Settings())
    out = capsys.readouterr().out
    assert rc == 0
    assert "kalshi: 3 open markets" in out
    assert "polymarket_us: 1 open markets" in out
    assert "total across 2 venues: 4 markets" in out


def test_test_order_canary_places_nofill(capsys, monkeypatch):
    import bot.dryrun as dr
    from bot.dryrun import test_order
    from bot.execution.orders import OrderResult, OrderStatus
    from bot.models import Side

    captured = {}

    class TradeVenue:
        name = "kalshi"
        is_trading_configured = True

        async def place_order(self, market, side, action, price, contracts):
            captured.update(market=market, side=side, action=action, price=price, contracts=contracts)
            return OrderResult(venue="kalshi", market_id=market, side=side, action=action,
                               requested=contracts, filled=0.0, status=OrderStatus.KILLED)

    monkeypatch.setattr(dr, "_build_venues", lambda s: [TradeVenue()])
    rc = test_order(Settings(), "kalshi:KXTICK")
    out = capsys.readouterr().out
    assert rc == 0
    assert captured == {"market": "KXTICK", "side": Side.YES, "action": "buy",
                        "price": 0.01, "contracts": 1}
    assert "order path works" in out


def test_test_order_unknown_venue(capsys, monkeypatch):
    import bot.dryrun as dr
    from bot.dryrun import test_order

    monkeypatch.setattr(dr, "_build_venues", lambda s: [])
    rc = test_order(Settings(), "nope:X")
    assert rc == 2
    assert "unknown venue" in capsys.readouterr().out


def test_recheck_matches_reports_model_rejections(capsys, monkeypatch):
    import bot.dryrun as dr
    import bot.matching.llm_client as llm_client
    from bot.dryrun import recheck_matches

    s = Store(":memory:")
    # Safe winner-type pair (passes the whitelist) that the model now rejects on the
    # subject (opposite teams) — the kind of catch recheck is meant to surface.
    s.upsert_market("kalshi", "KXWNBAGAME-26JUN17BRAHTI-HAI", "Will Haiti win against Brazil? - Haiti")
    s.upsert_market("polymarket_us", "P1", "Will Brazil win against Haiti? - Brazil")
    s.cache_verdict("kalshi", "KXWNBAGAME-26JUN17BRAHTI-HAI", "polymarket_us", "P1",
                    same_event=True, confidence=1.0)

    monkeypatch.setattr(dr, "Store", lambda path: s)
    # Model now correctly rejects (YES pays for different teams).
    monkeypatch.setattr(
        llm_client, "make_complete_fn",
        lambda cfg: (lambda prompt: '{"same_event": false, "confidence": 0.97, '
                     '"rationale": "different teams"}'),
    )
    rc = recheck_matches(Settings(), limit=50)
    out = capsys.readouterr().out
    assert rc == 0
    assert "WOULD DROP" in out
    assert "would REJECT 1 of 1" in out
    s.close()


def test_show_watchlist_prints_live_pairs(capsys, monkeypatch):
    import bot.dryrun as dr
    from bot.dryrun import show_watchlist

    s = Store(":memory:")
    K = "KXUFCFIGHT-26JUN20GANE-GANE"
    s.upsert_market("kalshi", K, "Gane by KO")
    s.upsert_market("polymarket_us", "P1", "Gane by KO")
    s.cache_verdict("kalshi", K, "polymarket_us", "P1", same_event=True, confidence=1.0)

    ka = mq("kalshi", K, "Gane by KO", yes_ask=0.4, no_ask=0.6)
    pa = mq("polymarket_us", "P1", "Gane by KO", yes_ask=0.62, no_ask=0.55)
    venues = [StubVenue("kalshi", {K: ka}), StubVenue("polymarket_us", {"P1": pa})]
    monkeypatch.setattr(dr, "_build_venues", lambda s: venues)
    monkeypatch.setattr(dr, "Store", lambda path: s)

    rc = show_watchlist(Settings())
    out = capsys.readouterr().out
    assert rc == 0
    assert "1 live watchlist pairs" in out
    assert "Gane by KO" in out
    s.close()


def test_check_ws_probe_collects_quotes():
    from bot.dryrun import _probe_stream

    class WSVenue:
        name = "kalshi"

        async def stream_order_book(self, market_ids):
            for mid in market_ids:
                yield mq("kalshi", mid, "t", yes_ask=0.4, no_ask=0.6)

    ok, msg, samples = asyncio.run(_probe_stream(WSVenue(), ["K1", "K2", "K3"], n=2, timeout=5.0))
    assert ok and len(samples) == 2 and msg == "ok"


def test_check_ws_probe_times_out_when_silent():
    from bot.dryrun import _probe_stream

    class SilentVenue:
        name = "polymarket_us"

        async def stream_order_book(self, market_ids):
            await asyncio.sleep(10)
            yield  # never reached within the timeout

    ok, msg, samples = asyncio.run(_probe_stream(SilentVenue(), ["P1"], n=1, timeout=0.1))
    assert not ok and samples == [] and "timeout" in msg


def test_apply_balance_caps_sets_from_balances():
    from types import SimpleNamespace

    from bot.dryrun import apply_balance_caps

    risk = RiskManager(RiskLimits(max_position_per_market=20, max_total_exposure=50))
    snaps = [SimpleNamespace(venue="kalshi", balance=115.52),
             SimpleNamespace(venue="polymarket_us", balance=142.53)]
    total = apply_balance_caps(risk, snaps)
    assert round(total, 2) == 258.05
    assert risk.limits.max_total_exposure == total
    assert risk.limits.max_position_per_market == total


def test_apply_balance_caps_noop_when_no_balance():
    from types import SimpleNamespace

    from bot.dryrun import apply_balance_caps

    risk = RiskManager(RiskLimits(max_position_per_market=20, max_total_exposure=50))
    total = apply_balance_caps(risk, [SimpleNamespace(venue="x", balance=None)])
    assert total == 0.0
    assert risk.limits.max_total_exposure == 50      # caps left unchanged
    assert risk.limits.max_position_per_market == 20


def test_compare_filters_reports_adds(capsys, tmp_path):
    from bot.config.settings import Settings
    from bot.dryrun import compare_filters

    path = str(tmp_path / "cmp.db")
    store = Store(path)
    seed = [
        # A WINNER whose Kalshi series (WCWINNER) isn't on the live allowlist and whose
        # title lacks the word "win" -> live filter drops it; fingerprint sees
        # winner==winner, same team/date -> keeps it. A real ADD.
        ("kalshi", "KXWCWINNER-26JUN22NORSEN-NOR", "Norway vs Senegal - Norway"),
        ("polymarket_us", "aec-fwc-nor-sen-2026-06-22-nor", "Norway vs Senegal - Norway"),
        # A MENTION novelty the LLM rubber-stamped: live drops (allowlist) AND
        # fingerprint drops (unmatchable). Neither keeps -> not an add.
        ("kalshi", "KXWCMENTION-26JUN22NORSEN-SHUT", "Norway vs Senegal - Shutout"),
        ("polymarket_us", "atc-fwc-nor-sen-2026-06-22-sen", "Norway to advance - Senegal"),
    ]
    for v, m, t in seed:
        store.upsert_market(v, m, t)
    store.cache_verdict("kalshi", "KXWCWINNER-26JUN22NORSEN-NOR",
                        "polymarket_us", "aec-fwc-nor-sen-2026-06-22-nor",
                        same_event=True, confidence=0.95)
    store.cache_verdict("kalshi", "KXWCMENTION-26JUN22NORSEN-SHUT",
                        "polymarket_us", "atc-fwc-nor-sen-2026-06-22-sen",
                        same_event=True, confidence=0.95)
    store.close()

    settings = Settings()
    object.__setattr__(settings, "db_path", path)
    assert compare_filters(settings) == 0
    out = capsys.readouterr().out
    assert "ADDS (fingerprint only):   1" in out


def test_run_cycle_fingerprint_gate_skips_novelty(capsys):
    # With use_fingerprint, the discovery cycle must NOT confirm a KXWCMENTION novelty
    # cross-product (unmatchable metric), but MUST still confirm a real winner pair.

    mention_k = mq("kalshi", "KXWCMENTION-26JUN22NORSEN-SHUT",
                   "Norway vs Senegal - Shutout", yes_ask=0.21, yes_ask_size=100,
                   no_ask=0.79, no_ask_size=100)
    mention_p = mq("polymarket_us", "atc-fwc-nor-sen-2026-06-22-sen",
                   "Norway vs Senegal - Senegal", yes_ask=0.31, yes_ask_size=100,
                   no_ask=0.69, no_ask_size=100)
    win_k = mq("kalshi", "KXATPMATCH-26JUN22DESHA-DE",
               "Will Alex de Minaur win the de Minaur vs Shapovalov match? - Alex de Minaur",
               yes_ask=0.40, yes_ask_size=100, no_ask=0.62, no_ask_size=100)
    win_p = mq("polymarket_us", "aec-atp-alemin-densha-2026-06-22",
               "Alex de Minaur vs. Denis Shapovalov - Alex de Minaur",
               yes_ask=0.62, yes_ask_size=100, no_ask=0.40, no_ask_size=60)
    venues = [
        StubVenue("kalshi", {mention_k.market_id: mention_k, win_k.market_id: win_k}),
        StubVenue("polymarket_us", {mention_p.market_id: mention_p, win_p.market_id: win_p}),
    ]
    fake = lambda prompt: '{"same_event": true, "confidence": 0.95, "rationale": "x"}'
    res = run_cycle_kw(venues, complete_fn=fake, use_fingerprint=True)
    confirmed = {(a, b) for (_, a, _, b, _) in res.confirmed_pairs}
    assert ("KXATPMATCH-26JUN22DESHA-DE", "aec-atp-alemin-densha-2026-06-22") in confirmed
    assert all("MENTION" not in a for (a, b) in confirmed)


def test_build_watchlist_prunes_settled(tmp_path):
    import asyncio as _a
    from bot.dryrun import build_watchlist

    path = str(tmp_path / "w.db")
    store = Store(path)
    # One pair whose Kalshi leg has settled.
    store.upsert_market("kalshi", "KSET", "x")
    store.upsert_market("polymarket_us", "PLIVE", "x")
    store.cache_verdict("kalshi", "KSET", "polymarket_us", "PLIVE",
                        same_event=True, confidence=1.0)
    cached = [("kalshi", "KSET", "polymarket_us", "PLIVE", "KSET|PLIVE")]

    class V:
        def __init__(self, name, open_map):
            self.name = name
            self._open = open_map
        async def is_open(self, mid):
            return self._open.get(mid)

    venues = [V("kalshi", {"KSET": False}), V("polymarket_us", {"PLIVE": True})]
    out = _a.run(build_watchlist(cached, set(), venues, store=store))
    assert out == []                                  # settled pair dropped
    # The settled market + its verdict were pruned, so next cycle won't re-probe it.
    assert store.confirmed_pairs(use_fingerprint=False, safe_types_only=False) == []
    assert store.conn.execute(
        "SELECT COUNT(*) c FROM match_verdicts").fetchone()["c"] == 0
    store.close()


def test_prune_market_removes_verdicts():
    s = Store(":memory:")
    s.upsert_market("kalshi", "K", "t")
    s.upsert_market("polymarket_us", "P", "t")
    s.cache_verdict("kalshi", "K", "polymarket_us", "P", same_event=True, confidence=1.0)
    assert s.prune_market("kalshi", "K") == 1
    assert s.conn.execute("SELECT COUNT(*) c FROM match_verdicts").fetchone()["c"] == 0
    assert s.conn.execute(
        "SELECT COUNT(*) c FROM markets WHERE market_id='K'").fetchone()["c"] == 0
    s.close()
