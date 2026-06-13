from bot.data.store import Store
from bot.engine import Engine
from bot.execution.risk import RiskLimits, RiskManager
from bot.fees import KalshiFeeModel, ZeroFeeModel
from bot.modes import RunMode
from bot.models import MarketQuote

FEES = {"kalshi": KalshiFeeModel(), "polymarket_us": ZeroFeeModel()}


def arb_pair():
    a = MarketQuote("kalshi", "A", "A", "E1", yes_ask=0.40, yes_ask_size=100, no_ask=0.65, no_ask_size=100)
    b = MarketQuote("polymarket_us", "B", "B", "E1", yes_ask=0.62, yes_ask_size=100, no_ask=0.55, no_ask_size=60)
    return a, b


def test_dry_run_detects_but_does_not_act():
    store = Store(":memory:")
    eng = Engine(
        fee_models=FEES,
        risk=RiskManager(RiskLimits(max_position_per_market=5000, max_total_exposure=100000)),
        store=store,
        mode=RunMode.DRY_RUN,
        min_edge=0.01,
    )
    res = eng.evaluate_pairs([arb_pair()])
    assert len(res.detected) == 1
    assert len(res.actionable) == 1          # viable in DRY_RUN
    assert res.skipped == []
    # Recorded to the store but not marked acted.
    row = store.conn.execute("SELECT acted FROM opportunities").fetchone()
    assert row["acted"] == 0
    store.close()


def test_risk_limit_skips_opportunity():
    eng = Engine(
        fee_models=FEES,
        risk=RiskManager(RiskLimits(max_position_per_market=5, max_total_exposure=5)),
        mode=RunMode.DRY_RUN,
    )
    res = eng.evaluate_pairs([arb_pair()])
    assert len(res.detected) == 1
    assert res.actionable == []
    assert len(res.skipped) == 1
    assert "cap" in res.skipped[0][1]


def test_live_mode_blocked_until_executor_exists():
    eng = Engine(
        fee_models=FEES,
        risk=RiskManager(RiskLimits(max_position_per_market=5000, max_total_exposure=100000)),
        mode=RunMode.LIVE_SMALL,
    )
    res = eng.evaluate_pairs([arb_pair()])
    assert res.actionable == []
    assert "executor not enabled" in res.skipped[0][1]


def test_no_pairs_no_opportunities():
    eng = Engine(fee_models=FEES, risk=RiskManager(), mode=RunMode.DRY_RUN)
    res = eng.evaluate_pairs([])
    assert res.detected == [] and res.best_profit == 0.0
