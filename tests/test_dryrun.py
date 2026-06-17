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

    async def scan_quotes(self, limit=500):
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
                 embed_fn=None, max_confirms=50):
    return asyncio.run(run_cycle(
        venues, store=store, risk=generous_risk(),
        fee_models={v.name: ZeroFeeModel() for v in venues},
        min_edge=min_edge, match_threshold=threshold, complete_fn=complete_fn,
        limit=50, embed_fn=embed_fn, max_confirms=max_confirms,
    ))


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
        async def scan_quotes(self, limit=500):
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
        async def scan_quotes(self, limit=500):
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
    s.upsert_market("kalshi", "K1", "Gane by KO")
    s.upsert_market("polymarket_us", "P1", "Gane by KO")
    s.cache_verdict("kalshi", "K1", "polymarket_us", "P1", same_event=True, confidence=1.0)
    # A fan-out market that must NOT appear in the tradeable-only view.
    for p in ("PA", "PB"):
        s.cache_verdict("kalshi", "Kfield", "polymarket_us", p, same_event=True, confidence=1.0)

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
    s.upsert_market("kalshi", "K1", "Will Haiti win against Brazil? - Haiti")
    s.upsert_market("polymarket_us", "P1", "Will Brazil win against Haiti? - Brazil")
    s.cache_verdict("kalshi", "K1", "polymarket_us", "P1", same_event=True, confidence=1.0)

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
    s.upsert_market("kalshi", "K1", "Gane by KO")
    s.upsert_market("polymarket_us", "P1", "Gane by KO")
    s.cache_verdict("kalshi", "K1", "polymarket_us", "P1", same_event=True, confidence=1.0)

    ka = mq("kalshi", "K1", "Gane by KO", yes_ask=0.4, no_ask=0.6)
    pa = mq("polymarket_us", "P1", "Gane by KO", yes_ask=0.62, no_ask=0.55)
    venues = [StubVenue("kalshi", {"K1": ka}), StubVenue("polymarket_us", {"P1": pa})]
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
