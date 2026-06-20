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


def test_leg1_partial_unwinds_not_halts():
    # FoK partial-filled leg1 (observed on Polymarket, e.g. 0.62/6). The known filled
    # amount is unwound and the bot keeps running — no halt, no naked position.
    yes = FakeVenue("kalshi", [
        res("kalshi", Side.YES, OrderStatus.PARTIAL, 0.62, 0.40, requested=6),
        res("kalshi", Side.YES, OrderStatus.FILLED, 0.62, 0.38, action="sell", requested=0.62),
    ])
    no = FakeVenue("poly", [])                       # leg2 never attempted on a partial leg1
    ex, risk = make_exec([yes, no])
    report = asyncio.run(ex.execute(opp(max_contracts=6)))
    assert report.status is ExecStatus.UNWOUND
    assert not risk.is_killed                         # bot keeps trading
    assert no.calls == []                             # no hedge leg placed
    assert yes.calls[1][2] == "sell"                  # the partial was sold back


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


def test_leg1_rejected_aborts_without_halting():
    # A definitive 4xx rejection (e.g. Kalshi 409) -> no position, abort the single
    # trade and KEEP trading. Must NOT trip the sticky kill switch.
    yes = FakeVenue("kalshi", [res("kalshi", Side.YES, OrderStatus.REJECTED, 0, None)])
    no = FakeVenue("poly", [])                       # leg2 never attempted
    ex, risk = make_exec([yes, no])
    report = asyncio.run(ex.execute(opp()))
    assert report.status is ExecStatus.SKIPPED
    assert no.calls == [] and not risk.is_killed     # one bad market doesn't freeze the bot


def test_leg2_rejected_unwinds_not_halts():
    # leg1 fills; leg2 cleanly rejected (4xx) -> definitively no leg-2 position -> unwind.
    yes = FakeVenue("kalshi", [
        res("kalshi", Side.YES, OrderStatus.FILLED, 2, 0.40),
        res("kalshi", Side.YES, OrderStatus.FILLED, 2, 0.38, action="sell"),
    ])
    no = FakeVenue("poly", [res("poly", Side.NO, OrderStatus.REJECTED, 0, None)])
    ex, risk = make_exec([yes, no])
    report = asyncio.run(ex.execute(opp()))
    assert report.status is ExecStatus.UNWOUND and not risk.is_killed


def test_leg2_reject_reason_in_unwind_report():
    # A rejected leg2 carries its HTTP status + venue body into the UNWOUND report so
    # the reason is diagnosable (not an opaque [REJECTED]).
    rej = OrderResult(venue="poly", market_id="P1", side=Side.NO, action="buy",
                      requested=2, filled=0, avg_price=None, order_id=None,
                      status=OrderStatus.REJECTED,
                      raw={"http_status": 409, "body": "in-play trading halted"})
    yes = FakeVenue("kalshi", [
        res("kalshi", Side.YES, OrderStatus.FILLED, 2, 0.40),
        res("kalshi", Side.YES, OrderStatus.FILLED, 2, 0.38, action="sell"),
    ])
    no = FakeVenue("poly", [rej])
    ex, risk = make_exec([yes, no])
    report = asyncio.run(ex.execute(opp()))
    assert report.status is ExecStatus.UNWOUND
    assert "409" in report.reason and "in-play trading halted" in report.reason


def test_leg2_ambiguous_halt_includes_reason():
    err = OrderResult(venue="poly", market_id="P1", side=Side.NO, action="buy",
                      requested=2, filled=0, avg_price=None, order_id=None,
                      status=OrderStatus.ERROR, raw={"error": "connection reset"})
    yes = FakeVenue("kalshi", [res("kalshi", Side.YES, OrderStatus.FILLED, 2, 0.40)])
    no = FakeVenue("poly", [err])
    ex, risk = make_exec([yes, no])
    report = asyncio.run(ex.execute(opp()))
    assert report.status is ExecStatus.HALTED and risk.is_killed
    assert "connection reset" in report.reason


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


def test_liquidity_guard_skips_thin_book():
    # Depth below the min-leg-depth floor -> skip before placing anything (so we never
    # end up half-filled on a book too thin to hedge/unwind).
    yes = FakeVenue("kalshi", [])
    no = FakeVenue("poly", [])
    risk = RiskManager(RiskLimits(max_position_per_market=1e9, max_total_exposure=1e12))
    ex = Executor({v.name: v for v in [yes, no]}, risk,
                  fee_models={v.name: ZeroFeeModel() for v in [yes, no]},
                  max_order_contracts=0, min_leg_depth=25)
    report = asyncio.run(ex.execute(opp(max_contracts=10)))   # 10 < 25 floor
    assert report.status is ExecStatus.SKIPPED
    assert "thin book" in report.reason
    assert yes.calls == [] and no.calls == [] and not risk.is_killed


def test_liquidity_guard_allows_deep_book():
    yes = FakeVenue("kalshi", [res("kalshi", Side.YES, OrderStatus.FILLED, 30, 0.40)])
    no = FakeVenue("poly", [res("poly", Side.NO, OrderStatus.FILLED, 30, 0.55)])
    risk = RiskManager(RiskLimits(max_position_per_market=1e9, max_total_exposure=1e12))
    ex = Executor({v.name: v for v in [yes, no]}, risk,
                  fee_models={v.name: ZeroFeeModel() for v in [yes, no]},
                  max_order_contracts=0, min_leg_depth=25)
    report = asyncio.run(ex.execute(opp(max_contracts=30)))   # 30 >= 25 floor
    assert report.status is ExecStatus.SUCCESS


class QuotingVenue(FakeVenue):
    """FakeVenue that also answers fetch_quote (for the unwind's bid lookup)."""

    def __init__(self, name, responses, quote):
        super().__init__(name, responses)
        self._quote = quote

    async def fetch_quote(self, market):
        return self._quote


def test_unwind_crosses_real_bid():
    # On a thin book the unwind must sell at the REAL best bid (yes_bid = 1 - no_ask),
    # not a fixed haircut off the entry price (which could sit above the bid and never
    # fill -> stuck naked + halt).
    from bot.models import MarketQuote

    # yes_bid = 1 - 0.95 = 0.05, well below the 0.35 haircut (0.40 - 0.05).
    thin = MarketQuote(venue="kalshi", market_id="K1", title="", no_ask=0.95)
    yes = QuotingVenue("kalshi", [
        res("kalshi", Side.YES, OrderStatus.FILLED, 2, 0.40),
        res("kalshi", Side.YES, OrderStatus.FILLED, 2, 0.05, action="sell"),
    ], thin)
    no = FakeVenue("poly", [res("poly", Side.NO, OrderStatus.KILLED, 0, None)])
    ex, risk = make_exec([yes, no])
    report = asyncio.run(ex.execute(opp()))
    assert report.status is ExecStatus.UNWOUND and not risk.is_killed
    assert yes.calls[1][2] == "sell"
    assert yes.calls[1][3] == 0.05            # crossed the real bid, not the 0.35 haircut


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


def test_confirmer_partial_does_not_downgrade_rest_filled():
    # The 0.62/6 race: the synchronous REST result is a full FILL, but the WS confirmer
    # times out mid-stream and returns a non-terminal PARTIAL. REST must win (no halt).
    class Confirmer:
        async def confirm(self, venue, order_id, requested, timeout):
            return OrderStatus.PARTIAL, 0.62, 0.88

    yes = FakeVenue("kalshi", [res("kalshi", Side.YES, OrderStatus.FILLED, 6, 0.40, requested=6)])
    no = FakeVenue("poly", [res("poly", Side.NO, OrderStatus.FILLED, 6, 0.55, requested=6)])
    ex, risk = make_exec([yes, no], max_order_contracts=0)
    ex.fill_confirmer = Confirmer()
    report = asyncio.run(ex.execute(opp(max_contracts=6)))
    assert report.status is ExecStatus.SUCCESS and not risk.is_killed


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


def test_aggressive_limits_preserve_edge_floor():
    # Fat edge (0.05) with a 0.01 floor -> 0.04 surplus split toward the NO leg.
    ex, _ = make_exec([FakeVenue("kalshi", []), FakeVenue("poly", [])],
                      limits=RiskLimits(min_edge=0.01))
    o = opp(yes_price=0.40, no_price=0.55)            # gross 0.95, edge 0.05
    yes_limit, no_limit = ex._aggressive_limits(o)
    assert yes_limit > 0.40 and no_limit > 0.55       # both reach past the quoted ask
    assert no_limit - 0.55 > yes_limit - 0.40         # NO (completing leg) gets more room
    # Worst case (both fill at the limit) still locks >= the floor.
    assert round(1 - (yes_limit + no_limit), 4) >= 0.01


def test_aggressive_limits_thin_edge_stays_conservative():
    # Edge at the floor -> no surplus -> no slippage room (don't chase a thin edge).
    ex, _ = make_exec([FakeVenue("kalshi", []), FakeVenue("poly", [])],
                      limits=RiskLimits(min_edge=0.05))
    o = opp(yes_price=0.45, no_price=0.50)            # gross 0.95, edge 0.05 == floor
    yes_limit, no_limit = ex._aggressive_limits(o)
    assert yes_limit == 0.45 and no_limit == 0.50


def test_aggressive_limits_used_in_execution():
    # The legs are actually placed at the widened limits, not the bare quoted ask.
    yes = FakeVenue("kalshi", [res("kalshi", Side.YES, OrderStatus.FILLED, 2, 0.40)])
    no = FakeVenue("poly", [res("poly", Side.NO, OrderStatus.FILLED, 2, 0.55)])
    ex, _ = make_exec([yes, no], limits=RiskLimits(min_edge=0.01, max_position_per_market=1e9,
                                                   max_total_exposure=1e12))
    asyncio.run(ex.execute(opp(yes_price=0.40, no_price=0.55)))
    assert yes.calls[0][3] > 0.40                     # leg1 limit widened past the ask
    assert no.calls[0][3] > 0.55                      # leg2 limit widened past the ask


def test_order_error_result_classification():
    from types import SimpleNamespace

    from bot.execution.orders import OrderStatus as OS
    from bot.execution.orders import order_error_result

    # 4xx (server rejected, no fill) -> REJECTED, body captured.
    http409 = Exception("Client error '409 Conflict'")
    http409.response = SimpleNamespace(status_code=409, text="market not accepting orders")
    r = order_error_result("kalshi", "M", Side.YES, "buy", 1, http409)
    assert r.status is OS.REJECTED and r.raw["http_status"] == 409
    assert "not accepting" in r.raw["body"]

    # 5xx (ambiguous — may have processed) -> ERROR.
    http500 = Exception("Server error")
    http500.response = SimpleNamespace(status_code=500, text="oops")
    assert order_error_result("kalshi", "M", Side.YES, "buy", 1, http500).status is OS.ERROR

    # Network error (no response) -> ERROR.
    assert order_error_result("kalshi", "M", Side.YES, "buy", 1,
                              TimeoutError("timed out")).status is OS.ERROR


def test_risk_cap_skips():
    yes = FakeVenue("kalshi", [])
    no = FakeVenue("poly", [])
    ex, _ = make_exec([yes, no], limits=RiskLimits(max_position_per_market=0.5, max_total_exposure=0.5))
    report = asyncio.run(ex.execute(opp()))
    # A risk cap that leaves room for <1 contract is now caught at sizing, naming the
    # binding constraint (the per-market cap) rather than a generic risk rejection.
    assert report.status is ExecStatus.SKIPPED and "per_market" in report.reason
