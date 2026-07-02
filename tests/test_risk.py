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


def test_retain_markets_releases_settled_exposure():
    # record_fill only ever ADDS, so settled markets would pin exposure forever and
    # tighten the caps monotonically. retain_markets releases anything not in the
    # venue's still-open set (fed by the balance poll).
    r = make(max_position_per_market=200, max_total_exposure=5000, max_daily_loss=500)
    r.record_fill("kalshi:A", 100)
    r.record_fill("kalshi:B", 50)
    r.record_fill("poly:C", 25)
    released = r.retain_markets({"kalshi:B"})          # A and C settled
    assert released == 125
    assert r.total_exposure == 50
    assert r.position("kalshi:A") == 0 and r.position("poly:C") == 0
    assert r.position("kalshi:B") == 50                # still-open exposure kept


def test_daily_pnl_rolls_at_utc_midnight():
    # "daily" must mean a real UTC day, not process lifetime: losses from yesterday
    # roll off the counter (the kill switch itself stays sticky and is NOT reset).
    r = make(max_position_per_market=200, max_total_exposure=5000, max_daily_loss=500)
    r.record_pnl(-100)
    assert r.daily_pnl == -100
    r._pnl_day = "2000-01-01"                          # simulate a prior-day counter
    r.record_pnl(-10)
    assert r.daily_pnl == -10                          # yesterday's -100 rolled off
    assert not r.is_killed
