"""Tests for the live DRY_RUN runner using injected stub venues — no network."""

import asyncio

from bot.config.settings import Settings
from bot.data.store import Store
from bot.dryrun import run, run_cycle
from bot.execution.risk import RiskLimits, RiskManager
from bot.fees import ZeroFeeModel
from bot.models import MarketQuote
from bot.venues.base import RawMarket


class StubVenue:
    """Implements the duck-typed venue interface the runner needs (read-only)."""

    def __init__(self, name, quotes: dict[str, MarketQuote]):
        self.name = name
        self.fee_model = ZeroFeeModel()
        self._quotes = quotes

    async def list_markets(self, limit=200):
        return [RawMarket(market_id=mid, title=q.title, raw={}) for mid, q in self._quotes.items()]

    async def fetch_quote(self, market):
        return self._quotes.get(market.market_id)


def mq(venue, mid, title, **kw):
    return MarketQuote(venue=venue, market_id=mid, title=title, **kw)


def generous_risk():
    return RiskManager(RiskLimits(max_position_per_market=1e9, max_total_exposure=1e12))


def test_bundle_arb_detected_kalshi_only():
    # yes_ask + no_ask = 0.95 < 1 -> single-venue bundle arb.
    kb = mq("kalshi", "B1", "Bundle market", yes_ask=0.40, yes_ask_size=50, no_ask=0.55, no_ask_size=50)
    store = Store(":memory:")
    result = asyncio.run(run(
        once=True, venues=[StubVenue("kalshi", {"B1": kb})],
        store=store, settings=Settings(), min_edge=0.01,
    ))
    assert len(result.bundle_opps) == 1
    assert round(result.bundle_opps[0].edge_per_contract, 4) == 0.05
    # Persisted, and no cross-venue activity with a single venue.
    assert store.conn.execute("SELECT COUNT(*) c FROM opportunities").fetchone()["c"] == 1
    assert result.cross.detected == []
    store.close()


def test_cross_venue_confirmed_with_llm():
    ka = mq("kalshi", "K1", "Fed cuts rates March 2026", yes_ask=0.40, yes_ask_size=100, no_ask=0.65, no_ask_size=100)
    pa = mq("polymarket_us", "P1", "Fed cuts interest rates March 2026", yes_ask=0.62, yes_ask_size=100, no_ask=0.55, no_ask_size=60)
    venues = [StubVenue("kalshi", {"K1": ka}), StubVenue("polymarket_us", {"P1": pa})]
    store = Store(":memory:")
    fake_complete = lambda prompt: '{"same_event": true, "confidence": 0.95, "rationale": "same"}'

    result = asyncio.run(run_cycle(
        venues, store=store, risk=generous_risk(),
        fee_models={"kalshi": ZeroFeeModel(), "polymarket_us": ZeroFeeModel()},
        min_edge=0.01, match_threshold=0.3, complete_fn=fake_complete, limit=50,
    ))

    assert result.candidate_pairs >= 1
    assert len(result.cross.detected) >= 1
    assert len(result.cross.actionable) >= 1   # viable in DRY_RUN
    # Verdict cached for reuse (order-independent lookup).
    assert store.get_verdict("polymarket_us", "P1", "kalshi", "K1") is not None
    store.close()


def test_cross_venue_skipped_without_llm():
    ka = mq("kalshi", "K1", "Fed cuts rates March 2026", yes_ask=0.40, yes_ask_size=100, no_ask=0.65, no_ask_size=100)
    pa = mq("polymarket_us", "P1", "Fed cuts interest rates March 2026", yes_ask=0.62, yes_ask_size=100, no_ask=0.55, no_ask_size=60)
    venues = [StubVenue("kalshi", {"K1": ka}), StubVenue("polymarket_us", {"P1": pa})]

    result = asyncio.run(run_cycle(
        venues, store=Store(":memory:"), risk=generous_risk(),
        fee_models={"kalshi": ZeroFeeModel(), "polymarket_us": ZeroFeeModel()},
        min_edge=0.01, match_threshold=0.3, complete_fn=None, limit=50,
    ))
    # Candidates surface, but nothing is tradeable without LLM confirmation.
    assert result.candidate_pairs >= 1
    assert result.cross.detected == []


def test_list_markets_failure_isolated_per_venue():
    class DeadVenue(StubVenue):
        async def list_markets(self, limit=200):
            raise RuntimeError("venue down")

    kb = mq("kalshi", "B1", "Bundle", yes_ask=0.40, yes_ask_size=50, no_ask=0.55, no_ask_size=50)
    venues = [DeadVenue("polymarket_us", {}), StubVenue("kalshi", {"B1": kb})]
    result = asyncio.run(run_cycle(
        venues, store=None, risk=generous_risk(),
        fee_models={"kalshi": ZeroFeeModel(), "polymarket_us": ZeroFeeModel()},
        min_edge=0.01, match_threshold=0.5, complete_fn=None, limit=10,
    ))
    # The healthy venue still produces its bundle arb despite the dead one.
    assert len(result.bundle_opps) == 1


def test_fetch_quote_failure_does_not_crash_cycle():
    class FlakyVenue(StubVenue):
        async def fetch_quote(self, market):
            raise RuntimeError("transient")

    result = asyncio.run(run_cycle(
        [FlakyVenue("kalshi", {"X": mq("kalshi", "X", "x")})],
        store=None, risk=generous_risk(), fee_models={"kalshi": ZeroFeeModel()},
        min_edge=0.01, match_threshold=0.5, complete_fn=None, limit=10,
    ))
    assert result.markets_seen == 1
    assert result.quotes == 0
    assert result.bundle_opps == []
