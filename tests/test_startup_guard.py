"""Startup reconciliation guard: funded + flat, else fail closed (kill switch)."""

import asyncio

import pytest

from bot.execution.account import AccountSnapshot, VenuePosition
from bot.execution.risk import RiskManager
from bot.execution.startup_guard import reconcile_startup
from bot.venues.polymarket_us import _account_balance, _parse_positions


class FakeVenue:
    def __init__(self, name, snapshot=None, *, trading=True, raises=False):
        self.name = name
        self.is_trading_configured = trading
        self._snapshot = snapshot
        self._raises = raises

    async def account_snapshot(self):
        if self._raises:
            raise RuntimeError("boom")
        return self._snapshot


class NoMethodVenue:
    name = "nomethod"
    is_trading_configured = True


def run(coro):
    return asyncio.run(coro)


def test_passes_when_funded_and_flat():
    risk = RiskManager()
    v = FakeVenue("kalshi", AccountSnapshot("kalshi", balance=300.0, positions=[]))
    res = run(reconcile_startup([v], risk, min_balance=100.0))
    assert res.ok and not risk.is_killed
    assert res.snapshots[0].balance == 300.0


def test_fails_and_trips_kill_switch_on_low_balance():
    risk = RiskManager()
    v = FakeVenue("kalshi", AccountSnapshot("kalshi", balance=50.0, positions=[]))
    res = run(reconcile_startup([v], risk, min_balance=100.0))
    assert not res.ok and risk.is_killed
    assert "balance" in res.reasons[0]


def test_fails_on_unknown_balance():
    risk = RiskManager()
    v = FakeVenue("kalshi", AccountSnapshot("kalshi", balance=None, positions=[]))
    res = run(reconcile_startup([v], risk, min_balance=0.0))
    assert not res.ok and risk.is_killed


def test_fails_on_pre_existing_position():
    risk = RiskManager()
    snap = AccountSnapshot("kalshi", balance=300.0,
                           positions=[VenuePosition("KXTEST", quantity=5, resting_orders=0)])
    v = FakeVenue("kalshi", snap)
    res = run(reconcile_startup([v], risk, min_balance=0.0))
    assert not res.ok and risk.is_killed
    assert "pre-existing" in res.reasons[0]


def test_fails_on_resting_order():
    risk = RiskManager()
    snap = AccountSnapshot("kalshi", balance=300.0,
                           positions=[VenuePosition("KXTEST", quantity=0, resting_orders=2)])
    res = run(reconcile_startup([FakeVenue("kalshi", snap)], risk))
    assert not res.ok and risk.is_killed


def test_allow_existing_positions_overrides():
    risk = RiskManager()
    snap = AccountSnapshot("kalshi", balance=300.0,
                           positions=[VenuePosition("KXTEST", quantity=5)])
    res = run(reconcile_startup([FakeVenue("kalshi", snap)], risk,
                                allow_existing_positions=True))
    assert res.ok and not risk.is_killed


def test_snapshot_error_fails_closed():
    risk = RiskManager()
    res = run(reconcile_startup([FakeVenue("poly", raises=True)], risk))
    assert not res.ok and risk.is_killed


def test_missing_method_fails_closed():
    risk = RiskManager()
    res = run(reconcile_startup([NoMethodVenue()], risk))
    assert not res.ok and risk.is_killed


def test_non_trading_venue_skipped():
    risk = RiskManager()
    # A non-trading venue is ignored; with no trading venues at all, that's a failure.
    res = run(reconcile_startup([FakeVenue("x", trading=False)], risk))
    assert not res.ok
    assert "no trading-configured venues" in res.reasons[0]


def test_multi_venue_one_bad_fails_all():
    risk = RiskManager()
    good = FakeVenue("kalshi", AccountSnapshot("kalshi", balance=300.0))
    bad = FakeVenue("poly", AccountSnapshot("poly", balance=10.0))
    res = run(reconcile_startup([good, bad], risk, min_balance=100.0))
    assert not res.ok and risk.is_killed
    assert any("poly" in r for r in res.reasons)


# ---- Polymarket payload parsing ----

def test_poly_balance_parses_various_shapes():
    assert _account_balance({"availableBalance": "123.45"}) == 123.45
    assert _account_balance({"balance": {"value": "50", "currency": "USD"}}) == 50.0
    assert _account_balance({"cashBalance": 7}) == 7.0
    assert _account_balance({"unknown": 1}) is None


def test_poly_positions_parse_and_filter_flat():
    body = {"positions": [
        {"marketSlug": "a", "quantity": "3"},
        {"slug": "b", "netQuantity": "0", "openOrders": 1},
        {"slug": "c", "size": "0", "openOrders": 0},     # flat -> dropped
    ]}
    pos = _parse_positions(body)
    ids = {p.market_id for p in pos}
    assert ids == {"a", "b"}


def test_poly_positions_unknown_shape_raises():
    # Fail closed: an unrecognized payload must NOT be read as a flat account.
    with pytest.raises(ValueError):
        _parse_positions({"weird": 123})


def test_poly_positions_empty_list_ok():
    assert _parse_positions({"positions": []}) == []
    assert _parse_positions([]) == []
