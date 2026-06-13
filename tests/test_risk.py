import pytest

from bot.execution.risk import RiskLimits, RiskManager


def make(**kw):
    return RiskManager(RiskLimits(**kw))


def test_allows_within_limits():
    r = make(max_position_per_market=200, max_total_exposure=5000, max_daily_loss=500)
    assert r.check("kalshi:A", 100)
    r.record_fill("kalshi:A", 100)
    assert r.position("kalshi:A") == 100
    assert r.total_exposure == 100


def test_per_market_cap():
    r = make(max_position_per_market=200, max_total_exposure=5000, max_daily_loss=500)
    r.record_fill("kalshi:A", 150)
    d = r.check("kalshi:A", 100)  # would be 250 > 200
    assert not d.allowed and "per-market" in d.reason


def test_total_exposure_cap():
    r = make(max_position_per_market=5000, max_total_exposure=300, max_daily_loss=500)
    r.record_fill("kalshi:A", 200)
    d = r.check("poly:B", 200)  # total would be 400 > 300
    assert not d.allowed and "total exposure" in d.reason


def test_daily_loss_trips_kill_switch():
    r = make(max_position_per_market=200, max_total_exposure=5000, max_daily_loss=100)
    r.record_pnl(-60)
    assert not r.is_killed
    r.record_pnl(-50)  # cumulative -110 <= -100
    assert r.is_killed
    assert not r.check("kalshi:A", 10)  # all opens rejected once killed


def test_manual_kill_and_reset():
    r = make()
    r.trip_kill_switch("operator stop")
    assert r.is_killed and "operator stop" in r.kill_reason
    assert not r.check("kalshi:A", 10)
    r.reset_kill_switch()
    assert not r.is_killed
    assert r.check("kalshi:A", 10)


def test_release_capital_with_negative_fill():
    r = make()
    r.record_fill("kalshi:A", 100)
    r.record_fill("kalshi:A", -100)
    assert r.position("kalshi:A") == 0
    assert r.total_exposure == 0


def test_rejects_nonpositive_notional():
    r = make()
    assert not r.check("kalshi:A", 0)
    assert not r.check("kalshi:A", -5)


def test_invalid_limits():
    with pytest.raises(ValueError):
        RiskLimits(max_total_exposure=0)
    with pytest.raises(ValueError):
        RiskLimits(min_edge=-1)
