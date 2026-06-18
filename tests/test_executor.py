"""Executor orchestration tests — the money-critical logic, with fake venues."""

import asyncio

from bot.data.store import Store
from bot.execution.executor import ExecStatus, Executor
from bot.execution.orders import OrderResult, OrderStatus
from bot.execution.risk import RiskLimits, RiskManager
from bot.fees import ZeroFeeModel
from bot.models import Side
from bot.strategies.arbitrage import ArbOpportunity


def opp(max_contracts=10, yes_price=0.40, no_price=0.55, yv="kalshi", nv="poly"):
    gross = yes_price + no_price
    return ArbOpportunity(
        event_key="E", buy_yes_venue=yv, buy_yes_market="K1", buy_no_venue=nv,
        buy_no_market="P1", yes_price=yes_price, no_price=no_price, gross_cost=gross,
        fee_per_pair=0.0, edge_per_contract=1 - gross, max_contracts=max_contracts,
        total_fees=0.0, total_profit=(1 - gross) * max_contracts, notional=gross * max_contracts,
    )


def res(venue, side, status, filled, avg, action="buy", requested=2):
    return OrderResult(venue=venue, market_id="M", side=side, action=action,
                       requested=requested, filled=filled, avg_price=avg,
                       order_id="o", status=status, raw={})


class FakeVenue:
    """Returns programmed OrderResults in sequence; records calls."""

    def __init__(self, name, responses):
        self.name = name
        self._responses = list(responses)
        self.calls = []

    async def place_order(self, market_id, side, action, price, contracts, *, tif="fill_or_kill"):
        self.calls.append((market_id, side.value, action, price, contracts, tif))
        return self._responses.pop(0)


def make_exec(venues, max_order_contracts=2, limits=None, store=None):
    risk = RiskManager(limits or RiskLimits(max_position_per_market=1e9, max_total_exposure=1e12))
    return Executor(
        {v.name: v for v in venues}, risk,
        fee_models={v.name: ZeroFeeModel() for v in venues},
        store=store, max_order_contracts=max_order_contracts,
    ), risk


def test_both_legs_fill_locks_arb():
    yes = FakeVenue("kalshi", [res("kalshi", Side.YES, OrderStatus.FILLED, 2, 0.40)])
    no = FakeVenue("poly", [res("poly", Side.NO, OrderStatus.FILLED, 2, 0.55)])
    store = Store(":memory:")
    ex, risk = make_exec([yes, no], store=store)
    report = asyncio.run(ex.execute(opp()))
    assert report.status is ExecStatus.SUCCESS
    assert round(report.realized_pnl, 4) == round(2 * (1 - 0.40 - 0.55), 4) == 0.10
    assert round(risk.daily_pnl, 2) == 0.10
    assert store.conn.execute("SELECT COUNT(*) c FROM fills").fetchone()["c"] == 2


def test_leg2_killed_unwinds_leg1():
    # leg1 fills; leg2 killed -> sell leg1 back (2nd response on the yes venue).
    yes = FakeVenue("kalshi", [
        res("kalshi", Side.YES, OrderStatus.FILLED, 2, 0.40),
        res("kalshi", Side.YES, OrderStatus.FILLED, 2, 0.38, action="sell"),
    ])
    no = FakeVenue("poly", [res("poly", Side.NO, OrderStatus.KILLED, 0, None)])
    ex, risk = make_exec([yes, no])
    report = asyncio.run(ex.execute(opp()))
    assert report.status is ExecStatus.UNWOUND
    assert round(report.realized_pnl, 4) == round(2 * (0.38 - 0.40), 4)  # small loss
    assert not risk.is_killed                       # unwind succeeded -> keep trading
    assert yes.calls[1][2] == "sell"                 # second yes call was the unwind


def test_leg1_killed_skips_no_position():
    yes = FakeVenue("kalshi", [res("kalshi", Side.YES, OrderStatus.KILLED, 0, None)])
    no = FakeVenue("poly", [])  # must never be called
    ex, risk = make_exec([yes, no])
    report = asyncio.run(ex.execute(opp()))
    assert report.status is ExecStatus.SKIPPED
    assert no.calls == []
    assert not risk.is_killed


def test_leg2_error_halts_and_trips_kill_switch():
    yes = FakeVenue("kalshi", [res("kalshi", Side.YES, OrderStatus.FILLED, 2, 0.40)])
    no = FakeVenue("poly", [res("poly", Side.NO, OrderStatus.ERROR, 0, None)])
    ex, risk = make_exec([yes, no])
    report = asyncio.run(ex.execute(opp()))
    assert report.status is ExecStatus.HALTED
    assert risk.is_killed                            # ambiguous hedge state -> stop everything


def test_failed_unwind_halts():
    yes = FakeVenue("kalshi", [
        res("kalshi", Side.YES, OrderStatus.FILLED, 2, 0.40),
        res("kalshi", Side.YES, OrderStatus.KILLED, 0, None, action="sell"),  # unwind fails
    ])
    no = FakeVenue("poly", [res("poly", Side.NO, OrderStatus.KILLED, 0, None)])
    ex, risk = make_exec([yes, no])
    report = asyncio.run(ex.execute(opp()))
    assert report.status is ExecStatus.HALTED
    assert risk.is_killed


def test_skips_when_kill_switch_active():
    yes = FakeVenue("kalshi", [])
    no = FakeVenue("poly", [])
    ex, risk = make_exec([yes, no])
    risk.trip_kill_switch("manual")
    report = asyncio.run(ex.execute(opp()))
    assert report.status is ExecStatus.SKIPPED
    assert yes.calls == []


def test_skips_when_size_below_one():
    yes = FakeVenue("kalshi", [])
    no = FakeVenue("poly", [])
    ex, _ = make_exec([yes, no])
    report = asyncio.run(ex.execute(opp(max_contracts=0)))
    assert report.status is ExecStatus.SKIPPED


def test_size_capped_by_max_order_contracts():
    yes = FakeVenue("kalshi", [res("kalshi", Side.YES, OrderStatus.FILLED, 2, 0.40)])
    no = FakeVenue("poly", [res("poly", Side.NO, OrderStatus.FILLED, 2, 0.55)])
    ex, _ = make_exec([yes, no], max_order_contracts=2)
    asyncio.run(ex.execute(opp(max_contracts=100)))
    assert yes.calls[0][4] == 2                       # requested contracts capped at 2


def test_max_size_binds_on_depth():
    ex, _ = make_exec([FakeVenue("kalshi", []), FakeVenue("poly", [])], max_order_contracts=0)
    size, caps = ex._max_size(opp(max_contracts=7))   # no balances/order-cap -> depth wins
    assert size == 7 and min(caps, key=caps.get) == "depth"


def test_max_size_binds_on_balance():
    ex, _ = make_exec([FakeVenue("kalshi", []), FakeVenue("poly", [])], max_order_contracts=0)
    # YES leg cash: $4 * 0.99 / $0.40 = 9.9 -> 9 contracts (the binding limit).
    ex.set_balances([type("S", (), {"venue": "kalshi", "balance": 4.0})(),
                     type("S", (), {"venue": "poly", "balance": 100.0})()])
    size, caps = ex._max_size(opp(max_contracts=100))
    assert size == 9 and min(caps, key=caps.get) == "cash_yes"


def test_max_size_binds_on_per_market_cap():
    ex, _ = make_exec(
        [FakeVenue("kalshi", []), FakeVenue("poly", [])], max_order_contracts=0,
        limits=RiskLimits(max_position_per_market=20, max_total_exposure=1e9),
    )
    # gross 0.95 -> 20 / 0.95 = 21.05 -> 21 contracts.
    size, caps = ex._max_size(opp(max_contracts=100))
    assert size == 21 and min(caps, key=caps.get) == "per_market"


def test_max_size_no_order_ceiling_when_zero():
    ex, _ = make_exec([FakeVenue("kalshi", []), FakeVenue("poly", [])], max_order_contracts=0)
    _, caps = ex._max_size(opp(max_contracts=100))
    assert "order_cap" not in caps                     # <=0 disables the per-order ceiling


def test_balances_decremented_after_success():
    yes = FakeVenue("kalshi", [res("kalshi", Side.YES, OrderStatus.FILLED, 5, 0.40, requested=5)])
    no = FakeVenue("poly", [res("poly", Side.NO, OrderStatus.FILLED, 5, 0.55, requested=5)])
    ex, _ = make_exec([yes, no], max_order_contracts=0)
    ex.set_balances([type("S", (), {"venue": "kalshi", "balance": 100.0})(),
                     type("S", (), {"venue": "poly", "balance": 100.0})()])
    report = asyncio.run(ex.execute(opp(max_contracts=5)))   # depth binds at 5
    assert report.status is ExecStatus.SUCCESS
    assert yes.calls[0][4] == 5                              # sized up to depth, not 2
    assert round(ex._balances["kalshi"], 2) == 98.0          # 100 - 5*0.40
    assert round(ex._balances["poly"], 2) == 97.25           # 100 - 5*0.55


def test_fill_confirmer_overrides_rest_result():
    # REST says KILLED, but the private fill stream confirms a FILL -> executor uses WS.
    class Confirmer:
        async def confirm(self, venue, order_id, requested, timeout):
            return OrderStatus.FILLED, requested, 0.40

    yes = FakeVenue("kalshi", [
        res("kalshi", Side.YES, OrderStatus.KILLED, 0, None)  # REST under-reports
    ])
    no = FakeVenue("poly", [res("poly", Side.NO, OrderStatus.FILLED, 2, 0.55)])
    ex, _ = make_exec([yes, no])
    ex.fill_confirmer = Confirmer()
    # leg1 REST=KILLED but confirmer flips it to FILLED, so the arb proceeds.
    report = asyncio.run(ex.execute(opp()))
    assert report.status is ExecStatus.SUCCESS


def test_fill_confirmer_needs_order_id():
    # No order_id -> confirmer is skipped, REST result stands.
    class Confirmer:
        async def confirm(self, *a, **k):
            raise AssertionError("should not be called without order_id")

    leg = res("kalshi", Side.YES, OrderStatus.KILLED, 0, None)
    leg.order_id = None
    yes = FakeVenue("kalshi", [leg])
    no = FakeVenue("poly", [])
    ex, _ = make_exec([yes, no])
    ex.fill_confirmer = Confirmer()
    report = asyncio.run(ex.execute(opp()))
    assert report.status is ExecStatus.SKIPPED   # leg1 killed, no order id -> skip


def test_risk_cap_skips():
    yes = FakeVenue("kalshi", [])
    no = FakeVenue("poly", [])
    ex, _ = make_exec([yes, no], limits=RiskLimits(max_position_per_market=0.5, max_total_exposure=0.5))
    report = asyncio.run(ex.execute(opp()))
    # A risk cap that leaves room for <1 contract is now caught at sizing, naming the
    # binding constraint (the per-market cap) rather than a generic risk rejection.
    assert report.status is ExecStatus.SKIPPED and "per_market" in report.reason
