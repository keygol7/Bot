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


def run_cycle_kw(venues, complete_fn=None, store=None, min_edge=0.01, threshold=0.3):
    return asyncio.run(run_cycle(
        venues, store=store, risk=generous_risk(),
        fee_models={v.name: ZeroFeeModel() for v in venues},
        min_edge=min_edge, match_threshold=threshold, complete_fn=complete_fn, limit=50,
    ))


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


def test_no_price_edge_pair_does_not_reach_llm():
    # Same event, but no combined price edge (0.55 + 0.55 > 1) -> never LLM-confirmed.
    ka = mq("kalshi", "K1", "Rain in Seattle tomorrow", yes_ask=0.55, yes_ask_size=100, no_ask=0.55, no_ask_size=100)
    pa = mq("polymarket_us", "P1", "Rain in Seattle tomorrow", yes_ask=0.55, yes_ask_size=100, no_ask=0.55, no_ask_size=100)
    called = []
    fake = lambda p: called.append(1) or '{"same_event": true, "confidence": 0.95}'
    result = run_cycle_kw([StubVenue("kalshi", {"K1": ka}), StubVenue("polymarket_us", {"P1": pa})], complete_fn=fake)
    assert result.candidate_pairs == 0
    assert called == []                                         # LLM never invoked


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
