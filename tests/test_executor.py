"""Executor orchestration tests — the money-critical logic, with fake venues."""

import asyncio

from bot.data.store import Store
from bot.execution.executor import ExecStatus, Executor
from bot.execution.orders import OrderResult, OrderStatus
from bot.execution.risk import RiskLimits, RiskManager
from bot.fees import ZeroFeeModel
from bot.models import MarketQuote, Side
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
    """Returns programmed OrderResults in sequence; records calls.

    ``hedge_depth`` is the top-of-book size its ``fetch_quote`` reports — the live-book
    read the executor's hedge-fillability check uses. Default huge = ample (won't cap);
    set small/0 to exercise the size-down / skip paths. Ask is quoted low so it's always
    at/through the hedge limit (the check gates on price, then returns this depth)."""

    def __init__(self, name, responses, hedge_depth=1e9):
        self.name = name
        self._responses = list(responses)
        self.calls = []
        self.hedge_depth = hedge_depth

    async def place_order(self, market_id, side, action, price, contracts, *,
                          tif="fill_or_kill", post_only=False, expiration_ts=None):
        self.calls.append((market_id, side.value, action, price, contracts, tif, post_only))
        return self._responses.pop(0)

    async def fetch_quote(self, market):
        return MarketQuote(
            venue=self.name, market_id=getattr(market, "market_id", ""), title="",
            yes_ask=0.01, yes_ask_size=self.hedge_depth,
            no_ask=0.01, no_ask_size=self.hedge_depth,
        )


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


def test_unwind_failure_quarantines_and_keeps_trading():
    # leg1 fills, leg2 rejects, AND the unwind also fails -> stuck naked leg. The market is
    # AUTO-BLACKLISTED and the trade QUARANTINED (recorded) — but the global kill switch is
    # NOT tripped, so the bot keeps trading the rest of the book. The recurring thin-prop fix.
    store = Store(":memory:")
    yes = FakeVenue("kalshi", [
        res("kalshi", Side.YES, OrderStatus.FILLED, 2, 0.40),                 # leg1 fills
        res("kalshi", Side.YES, OrderStatus.KILLED, 0, None, action="sell"),  # unwind fails
    ])
    no = FakeVenue("poly", [res("poly", Side.NO, OrderStatus.KILLED, 0, None)])  # leg2 rejects
    ex, risk = make_exec([yes, no], store=store)
    report = asyncio.run(ex.execute(opp()))
    assert report.status is ExecStatus.QUARANTINED
    assert not risk.is_killed                              # bot keeps trading other pairs
    assert store.blacklisted_keys() == {store._pair_key("kalshi", "K1", "poly", "P1")}


def test_unwind_failure_without_store_hard_halts():
    # No store -> can't blacklist -> fall back to a HARD halt (fail closed) so a stranded leg
    # that can't be quarantined still stops the bot.
    yes = FakeVenue("kalshi", [
        res("kalshi", Side.YES, OrderStatus.FILLED, 2, 0.40),
        res("kalshi", Side.YES, OrderStatus.KILLED, 0, None, action="sell"),
    ])
    no = FakeVenue("poly", [res("poly", Side.NO, OrderStatus.KILLED, 0, None)])
    ex, risk = make_exec([yes, no])                        # store=None
    report = asyncio.run(ex.execute(opp()))
    assert report.status is ExecStatus.HALTED and risk.is_killed


def test_scarcity_gate_reserves_capital_for_fat_edges():
    # When a venue is nearly drained, a THIN edge is skipped (reserve the last cash for a
    # fat edge); a FAT edge still fires; and with ample balance, the thin edge fires.
    def fresh():
        return (FakeVenue("kalshi", [res("kalshi", Side.YES, OrderStatus.FILLED, 2, 0.40)]),
                FakeVenue("poly", [res("poly", Side.NO, OrderStatus.FILLED, 2, 0.57)]))

    # scarce kalshi ($10 < $20 floor) + thin 1c edge -> reserved (skipped, nothing fired)
    y, n = fresh()
    ex, _ = make_exec([y, n])
    ex.scarcity_balance, ex.scarcity_min_edge = 20.0, 0.02
    ex._balances = {"kalshi": 10.0, "poly": 100.0}
    rep = asyncio.run(ex.execute(opp(yes_price=0.40, no_price=0.59)))   # edge 0.01
    assert rep.status is ExecStatus.SKIPPED and "scarce" in rep.reason and y.calls == []

    # same scarce balance but a FAT 3c edge -> fires
    y, n = fresh()
    ex, _ = make_exec([y, n])
    ex.scarcity_balance, ex.scarcity_min_edge = 20.0, 0.02
    ex._balances = {"kalshi": 10.0, "poly": 100.0}
    rep = asyncio.run(ex.execute(opp(yes_price=0.40, no_price=0.57)))   # edge 0.03
    assert rep.status is ExecStatus.SUCCESS

    # ample balance + thin 1c edge -> NOT gated (capital isn't scarce)
    y, n = fresh()
    ex, _ = make_exec([y, n])
    ex.scarcity_balance, ex.scarcity_min_edge = 20.0, 0.02
    ex._balances = {"kalshi": 100.0, "poly": 100.0}
    rep = asyncio.run(ex.execute(opp(yes_price=0.40, no_price=0.59)))   # edge 0.01
    assert rep.status is ExecStatus.SUCCESS


def test_rebalance_gate_skips_expensive_leg_on_drained_venue():
    # When a venue is below the rebalance floor, an arb whose leg on it is the EXPENSIVE side
    # is skipped (reserve its cash for cheap-on-it arbs); a cheap-on-it arb fires.
    def fresh():
        return (FakeVenue("kalshi", [res("kalshi", Side.YES, OrderStatus.FILLED, 2, 0.40)]),
                FakeVenue("poly", [res("poly", Side.NO, OrderStatus.FILLED, 2, 0.57)]))

    # kalshi drained ($15 < $30 floor). This arb's kalshi (YES) leg is the EXPENSIVE side
    # (0.80) -> skip so kalshi's cash goes to arbs where kalshi is the cheap half.
    y, n = fresh()
    ex, _ = make_exec([y, n])
    ex.rebalance_floor = 30.0
    ex._balances = {"kalshi": 15.0, "poly": 100.0}
    rep = asyncio.run(ex.execute(opp(yv="kalshi", nv="poly", yes_price=0.80, no_price=0.17)))
    assert rep.status is ExecStatus.SKIPPED and "rebalance" in rep.reason and y.calls == []

    # same drained kalshi, but here the kalshi (YES) leg is the CHEAP side (0.17) -> fires
    # (spends little on the scarce venue, shifts cost to the funded poly side)
    y, n = fresh()
    ex, _ = make_exec([y, n])
    ex.rebalance_floor = 30.0
    ex._balances = {"kalshi": 15.0, "poly": 100.0}
    rep = asyncio.run(ex.execute(opp(yv="kalshi", nv="poly", yes_price=0.17, no_price=0.80)))
    assert rep.status is ExecStatus.SUCCESS

    # both venues funded -> no rebalance gating even on a lopsided arb
    y, n = fresh()
    ex, _ = make_exec([y, n])
    ex.rebalance_floor = 30.0
    ex._balances = {"kalshi": 100.0, "poly": 100.0}
    rep = asyncio.run(ex.execute(opp(yv="kalshi", nv="poly", yes_price=0.80, no_price=0.17)))
    assert rep.status is ExecStatus.SUCCESS


def test_churn_guard_blocks_reverse_direction():
    # Already holding a hedge (net-long NO on kalshi K1, net-long YES on poly P1). An opp to
    # buy YES@kalshi + NO@poly is the REVERSE direction -> churn (forfeits Kalshi premium)
    # -> SKIP, nothing fired.
    yes = FakeVenue("kalshi", [res("kalshi", Side.YES, OrderStatus.FILLED, 2, 0.40)])
    no = FakeVenue("poly", [res("poly", Side.NO, OrderStatus.FILLED, 2, 0.55)])
    ex, _ = make_exec([yes, no])
    ex._positions = {("kalshi", "K1"): -3.0, ("poly", "P1"): 3.0}
    rep = asyncio.run(ex.execute(opp()))
    assert rep.status is ExecStatus.SKIPPED and "churn" in rep.reason
    assert yes.calls == [] and no.calls == []


def test_churn_guard_allows_same_direction_add():
    # Same hedge direction already held (net-long YES@kalshi, NO@poly) -> adding deepens the
    # hedge (not churn) -> fires, and the cache reflects the larger position afterward.
    yes = FakeVenue("kalshi", [res("kalshi", Side.YES, OrderStatus.FILLED, 2, 0.40)])
    no = FakeVenue("poly", [res("poly", Side.NO, OrderStatus.FILLED, 2, 0.55)])
    ex, _ = make_exec([yes, no])
    ex._positions = {("kalshi", "K1"): 3.0, ("poly", "P1"): -3.0}
    rep = asyncio.run(ex.execute(opp()))
    assert rep.status is ExecStatus.SUCCESS
    assert ex._positions[("kalshi", "K1")] == 5.0 and ex._positions[("poly", "P1")] == -5.0


def test_churn_guard_allows_flat_open_and_tracks_fill():
    # Flat on the pair -> opens normally; the settled fill is then tracked so the REVERSE
    # opp would be blocked next time.
    yes = FakeVenue("kalshi", [res("kalshi", Side.YES, OrderStatus.FILLED, 2, 0.40)])
    no = FakeVenue("poly", [res("poly", Side.NO, OrderStatus.FILLED, 2, 0.55)])
    ex, _ = make_exec([yes, no])                       # _positions empty -> flat
    rep = asyncio.run(ex.execute(opp()))
    assert rep.status is ExecStatus.SUCCESS
    assert ex._positions[("kalshi", "K1")] == 2.0 and ex._positions[("poly", "P1")] == -2.0
    # the true reverse buys YES where we're long NO (poly P1) and NO where we're long YES
    # (kalshi K1) -> now blocked as churn
    from dataclasses import replace
    rev = replace(opp(), buy_yes_venue="poly", buy_yes_market="P1",
                  buy_no_venue="kalshi", buy_no_market="K1")
    assert ex._churn_skip(rev) is not None


def test_set_balances_caches_and_refreshes_positions():
    from bot.execution.account import AccountSnapshot, VenuePosition
    ex, _ = make_exec([FakeVenue("kalshi", []), FakeVenue("poly", [])])
    ex.set_balances([
        AccountSnapshot("kalshi", 100.0, [VenuePosition("K1", -3, 0)]),
        AccountSnapshot("poly", 100.0, [VenuePosition("P1", 3, 0)]),
    ])
    assert ex._positions[("kalshi", "K1")] == -3 and ex._positions[("poly", "P1")] == 3
    assert ex._churn_skip(opp()) is not None                # reverse hedge -> blocked
    # a later snapshot with the kalshi market settled (flat) drops it from the cache
    ex.set_balances([AccountSnapshot("kalshi", 100.0, [])])
    assert ("kalshi", "K1") not in ex._positions
    assert ex._churn_skip(opp()) is None                    # no longer a reverse hedge


def test_reservation_drains_cache_before_legs_fire():
    # The fire-time reservation must reduce the cached balance for BOTH legs BEFORE any
    # order is placed — so a concurrent fast-loop execution sees the drain and won't
    # over-commit a draining venue (the insufficient_balance-reject -> unwind bug).
    holder, seen = {}, {}

    class CheckVenue(FakeVenue):
        async def place_order(self, market_id, side, action, price, contracts, **k):
            seen.setdefault(self.name, holder["ex"]._balance(self.name))  # cache AT fire
            return await super().place_order(market_id, side, action, price, contracts, **k)

    kalshi = CheckVenue("kalshi", [res("kalshi", Side.YES, OrderStatus.FILLED, 2, 0.40)])
    poly = CheckVenue("poly", [res("poly", Side.NO, OrderStatus.FILLED, 2, 0.55)])
    ex, risk = make_exec([kalshi, poly])
    holder["ex"] = ex
    ex._balances = {"kalshi": 100.0, "poly": 100.0}
    rep = asyncio.run(ex.execute(opp(yv="kalshi", nv="poly", yes_price=0.40, no_price=0.55)))
    assert rep.status is ExecStatus.SUCCESS
    assert seen["kalshi"] < 100.0 and seen["poly"] < 100.0   # reserved before firing


def test_reservation_releases_leaving_only_actual_spend():
    # After the trade, the reservation is released and only the ACTUAL fill cost remains
    # debited (kalshi 2*0.40, poly 2*0.55) — the reserve nets out, no double-counting.
    kalshi = FakeVenue("kalshi", [res("kalshi", Side.YES, OrderStatus.FILLED, 2, 0.40)])
    poly = FakeVenue("poly", [res("poly", Side.NO, OrderStatus.FILLED, 2, 0.55)])
    ex, risk = make_exec([kalshi, poly])
    ex._balances = {"kalshi": 100.0, "poly": 100.0}
    asyncio.run(ex.execute(opp(yv="kalshi", nv="poly", yes_price=0.40, no_price=0.55)))
    assert round(ex._balances["kalshi"], 2) == round(100.0 - 2 * 0.40, 2)
    assert round(ex._balances["poly"], 2) == round(100.0 - 2 * 0.55, 2)


def test_reservation_released_on_skip_leaves_balance_intact():
    # Leg 1 doesn't fill -> clean skip. The reservation must be released so the cache is
    # unchanged (no phantom drain that would wrongly throttle the next trade).
    kalshi = FakeVenue("kalshi", [res("kalshi", Side.YES, OrderStatus.KILLED, 0, None)])
    poly = FakeVenue("poly", [])
    risk = RiskManager(RiskLimits(max_position_per_market=1e9, max_total_exposure=1e12))
    ex = Executor({v.name: v for v in [kalshi, poly]}, risk,
                  fee_models={v.name: ZeroFeeModel() for v in [kalshi, poly]},
                  max_order_contracts=2, take_first_venue="kalshi")
    ex._balances = {"kalshi": 100.0, "poly": 100.0}
    asyncio.run(ex.execute(opp(yv="kalshi", nv="poly", yes_price=0.40, no_price=0.55)))
    assert ex._balances["kalshi"] == 100.0 and ex._balances["poly"] == 100.0


def test_min_venue_balance_skips_drained_hedge_leg():
    # A venue too drained to fund its leg must NOT trade — else we'd fire the first leg and
    # the second rejects for insufficient_balance, leaving a naked position (Kalshi at $0.38).
    yes = FakeVenue("kalshi", [])
    no = FakeVenue("poly", [])
    risk = RiskManager(RiskLimits(max_position_per_market=1e9, max_total_exposure=1e12))
    ex = Executor(
        {v.name: v for v in [yes, no]}, risk,
        fee_models={v.name: ZeroFeeModel() for v in [yes, no]},
        max_order_contracts=0, min_venue_balance=5.0)
    ex._balances = {"kalshi": 100.0, "poly": 0.38}   # poly (the NO/hedge leg) is drained
    report = asyncio.run(ex.execute(opp(yv="kalshi", nv="poly")))
    assert report.status is ExecStatus.SKIPPED and "cash_no" in report.reason
    assert yes.calls == [] and no.calls == []         # nothing fired -> no naked leg


def test_take_first_venue_fires_rejection_prone_leg_first():
    # With take_first_venue = the rejection-prone venue (poly), it fires FIRST. A poly KILL/
    # 500 is then a CLEAN SKIP (no kalshi leg placed, no unwind) instead of a kalshi unwind —
    # the structural fix for "Polymarket 500s the hedge -> kalshi unwind" seen in live data.
    poly = FakeVenue("poly", [res("poly", Side.NO, OrderStatus.KILLED, 0, None)])  # leg1 fails
    kalshi = FakeVenue("kalshi", [])                  # the hedge leg — must never be placed
    risk = RiskManager(RiskLimits(max_position_per_market=1e9, max_total_exposure=1e12))
    ex = Executor(
        {v.name: v for v in [poly, kalshi]}, risk,
        fee_models={v.name: ZeroFeeModel() for v in [poly, kalshi]},
        max_order_contracts=2, take_first_venue="poly",
    )
    report = asyncio.run(ex.execute(opp(yv="kalshi", nv="poly")))   # poly is the NO leg
    assert len(poly.calls) == 1                        # poly fired FIRST
    assert kalshi.calls == []                          # its KILL -> clean skip, no hedge leg
    assert report.status is ExecStatus.SKIPPED and not risk.is_killed


def _rel_exec(venues, store=None):
    """Executor with the empirical reliability gate armed (probe 2 / proven 3 / max-fails 2)."""
    risk = RiskManager(RiskLimits(max_position_per_market=1e9, max_total_exposure=1e12))
    return Executor(
        {v.name: v for v in venues}, risk,
        fee_models={v.name: ZeroFeeModel() for v in venues},
        store=store, max_order_contracts=100,
        probe_contracts=2, market_proven_fills=3, market_max_fails=2,
    )


def test_reliability_caps_unproven_market_to_probe_size():
    # An untested market trades only at the tiny probe size (2) regardless of depth/balance,
    # so a phantom-depth book can leave at most a 2-contract naked remainder, not 100.
    yes = FakeVenue("kalshi", [res("kalshi", Side.YES, OrderStatus.FILLED, 2, 0.40)])
    no = FakeVenue("poly", [res("poly", Side.NO, OrderStatus.FILLED, 2, 0.55)])
    ex = _rel_exec([yes, no])
    size, caps = ex._max_size(opp(max_contracts=100))
    assert caps["reliability"] == 2 and size == 2


def test_reliability_scales_proven_market_on_demonstrated_fill_size():
    # A proven market scales on the LARGEST size it actually FILLED (geometric), not a slow
    # per-fill count: cap = max(probe, ramp_factor x max_fill). Reaches full size in a few
    # fills (no lost edge); a market only proven small can't jump past ramp_factor x what it
    # proved (so a phantom-at-size book can't strand a big naked leg). ramp_factor default 3.
    ex = _rel_exec([FakeVenue("kalshi", []), FakeVenue("poly", [])])   # probe 2, ramp 3
    # proven, largest fill so far = 2 (the probes) -> cap = max(2, 3*2) = 6
    ex._market_rel[("kalshi", "K1")] = (3, 0, 0, 2.0)
    ex._market_rel[("poly", "P1")] = (3, 0, 0, 2.0)
    _, caps = ex._max_size(opp(max_contracts=100))
    assert caps["reliability"] == 6
    # after a 6-contract fill, max_fill=6 -> cap = 18 (the old linear ramp would still be ~8)
    ex._market_rel[("kalshi", "K1")] = (4, 0, 0, 6.0)
    ex._market_rel[("poly", "P1")] = (4, 0, 0, 6.0)
    _, caps2 = ex._max_size(opp(max_contracts=100))
    assert caps2["reliability"] == 18


def test_reliability_excludes_repeatedly_failing_market():
    # A market that has KILL/REJECTed max_fails times without ever proving is excluded (cap 0).
    yes = FakeVenue("kalshi", [])
    no = FakeVenue("poly", [])
    ex = _rel_exec([yes, no])
    ex._market_rel[("poly", "P1")] = (0, 2, 2, 0.0)    # poly leg proven phantom
    size, caps = ex._max_size(opp(max_contracts=100))
    assert caps["reliability"] == 0 and size == 0


def test_reliability_excludes_proven_market_on_consecutive_fail_streak():
    # The Valorant case: a market proved real depth (5 fills) then its liquidity drained
    # mid-game (consecutive fails). Despite being "proven", a streak >= max_fails must
    # EXCLUDE it (cap 0) so it stops firing full size into vanished volume -> unwinds.
    yes = FakeVenue("kalshi", [])
    no = FakeVenue("poly", [])
    ex = _rel_exec([yes, no])
    ex._market_rel[("kalshi", "K1")] = (5, 3, 2, 6.0)  # proven, but 2 consecutive fails now
    ex._market_rel[("poly", "P1")] = (8, 0, 0, 6.0)
    size, caps = ex._max_size(opp(max_contracts=100))
    assert caps["reliability"] == 0 and size == 0
    # ...and a single fresh fill resets the streak -> trades again (scaled on max_fill, not 0).
    ex._market_rel[("kalshi", "K1")] = (6, 3, 0, 6.0)
    _, caps2 = ex._max_size(opp(max_contracts=100))
    assert caps2["reliability"] == 3 * 6.0             # ramp_factor x max_fill, no longer excluded


def test_reliability_records_fok_outcomes_and_persists():
    # A FILLED buy is recorded as a fill, and persists to the store for cross-restart memory.
    store = Store(":memory:")
    yes = FakeVenue("kalshi", [res("kalshi", Side.YES, OrderStatus.FILLED, 2, 0.40)])
    no = FakeVenue("poly", [res("poly", Side.NO, OrderStatus.KILLED, 0, None)])
    ex = _rel_exec([yes, no], store=store)
    asyncio.run(ex.execute(opp()))
    # kalshi YES filled 2 -> (1 fill, 0 fails, 0 streak, max_fill 2); poly NO killed -> (0,1,1,0)
    assert ex._market_rel[("kalshi", "K1")] == (1, 0, 0, 2.0)
    assert ex._market_rel[("poly", "P1")] == (0, 1, 1, 0.0)
    assert store.market_reliability()[("poly", "P1")] == (0, 1, 1, 0.0)


def test_reliability_streak_resets_on_fill_in_store():
    # fail, fail -> streak 2; then a fill (size 5) resets streak to 0 and sets max_fill=5.
    store = Store(":memory:")
    store.record_market_outcome("kalshi", "K1", ok=False)
    store.record_market_outcome("kalshi", "K1", ok=False)
    assert store.market_reliability()[("kalshi", "K1")] == (0, 2, 2, 0.0)
    store.record_market_outcome("kalshi", "K1", ok=True, fill_size=5.0)
    assert store.market_reliability()[("kalshi", "K1")] == (1, 2, 0, 5.0)


def test_reliability_loads_history_from_store_on_init():
    store = Store(":memory:")
    store.record_market_outcome("poly", "P1", ok=False)
    store.record_market_outcome("poly", "P1", ok=False)
    ex = _rel_exec([FakeVenue("kalshi", []), FakeVenue("poly", [])], store=store)
    assert ex._market_rel[("poly", "P1")] == (0, 2, 2, 0.0)  # excluded from the first tick after restart


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


class SnapshotVenue(FakeVenue):
    """FakeVenue that also answers account_snapshot (for leg-1 ERROR reconciliation)."""

    def __init__(self, name, responses, positions):
        super().__init__(name, responses)
        self._positions = positions

    async def account_snapshot(self):
        from bot.execution.account import AccountSnapshot
        return AccountSnapshot(self.name, 1000.0, self._positions)


def test_leg1_error_reconciles_flat_and_skips():
    # A transient leg-1 ERROR with the account actually flat -> clean skip, no halt.
    kalshi = SnapshotVenue("kalshi", [res("kalshi", Side.YES, OrderStatus.ERROR, 0, None)], [])
    poly = FakeVenue("poly", [])
    ex, risk = make_exec([kalshi, poly])
    report = asyncio.run(ex.execute(opp()))
    assert report.status is ExecStatus.SKIPPED
    assert not risk.is_killed and poly.calls == []     # not killed -> keeps trading


def test_leg1_error_completes_hedge_and_locks():
    # Leg-1 ERROR but the order ACTUALLY FILLED (a real position exists) — the executed-but-
    # unacked case. Instead of halting naked, the executor completes the hedge for the unhedged
    # imbalance and LOCKS the arb (the recovery a human did by hand for the Valorant DKGENA halt).
    from bot.execution.account import VenuePosition
    kalshi = SnapshotVenue(
        "kalshi", [res("kalshi", Side.YES, OrderStatus.ERROR, 0, None)],
        [VenuePosition("K1", 2, 0)],                    # leg1 errored-but-filled: 2 naked
    )
    poly = SnapshotVenue(
        "poly", [res("poly", Side.NO, OrderStatus.FILLED, 2, 0.55)],  # hedge completes
        [],                                              # poly flat -> unhedged = 2 - 0
    )
    ex, risk = make_exec([kalshi, poly])
    report = asyncio.run(ex.execute(opp()))
    assert report.status is ExecStatus.SUCCESS
    assert not risk.is_killed                            # locked, not halted
    assert poly.calls and poly.calls[0][4] == 2          # hedge fired for the 2 naked contracts


def test_leg1_error_hedge_completion_fails_halts():
    # Leg-1 ERROR left a position, but the recovery hedge can't fill -> we now KNOW it's naked
    # -> halt for manual reconcile (no silent drift).
    from bot.execution.account import VenuePosition
    kalshi = SnapshotVenue(
        "kalshi", [res("kalshi", Side.YES, OrderStatus.ERROR, 0, None)],
        [VenuePosition("K1", 2, 0)],
    )
    poly = SnapshotVenue("poly", [res("poly", Side.NO, OrderStatus.KILLED, 0, None)], [])
    ex, risk = make_exec([kalshi, poly])
    report = asyncio.run(ex.execute(opp()))
    assert report.status is ExecStatus.HALTED and risk.is_killed


def test_leg1_error_excess_imbalance_halts():
    # The naked position EXCEEDS this trade's size (e.g. prior accumulation) -> don't auto-fire
    # a large unexplained order; halt for manual reconcile and DON'T touch the book.
    from bot.execution.account import VenuePosition
    kalshi = SnapshotVenue(
        "kalshi", [res("kalshi", Side.YES, OrderStatus.ERROR, 0, None)],
        [VenuePosition("K1", 50, 0)],                    # 50 naked vs trade size 2
    )
    poly = SnapshotVenue("poly", [], [])                 # readable, flat
    ex, risk = make_exec([kalshi, poly])
    report = asyncio.run(ex.execute(opp()))
    assert report.status is ExecStatus.HALTED and risk.is_killed
    assert poly.calls == []                              # excess guard fired before any hedge


def test_leg1_error_already_balanced_skips():
    # Leg-1 ERROR but BOTH legs already hold the position (covered by prior hedged fills) — no
    # naked remainder -> skip and keep trading, don't halt and don't double-hedge.
    from bot.execution.account import VenuePosition
    kalshi = SnapshotVenue("kalshi", [res("kalshi", Side.YES, OrderStatus.ERROR, 0, None)],
                           [VenuePosition("K1", 3, 0)])
    poly = SnapshotVenue("poly", [], [VenuePosition("P1", 3, 0)])
    ex, risk = make_exec([kalshi, poly])
    report = asyncio.run(ex.execute(opp()))
    assert report.status is ExecStatus.SKIPPED and not risk.is_killed
    assert poly.calls == []


def test_leg1_error_unreadable_hedge_side_halts():
    # Leg-1 ERROR with a position, but the hedge venue can't be read (no snapshot) -> can't
    # compute the imbalance -> fail closed (halt) rather than guess.
    from bot.execution.account import VenuePosition
    kalshi = SnapshotVenue(
        "kalshi", [res("kalshi", Side.YES, OrderStatus.ERROR, 0, None)],
        [VenuePosition("K1", 2, 0)],
    )
    poly = FakeVenue("poly", [])                         # no account_snapshot -> unreadable
    ex, risk = make_exec([kalshi, poly])
    report = asyncio.run(ex.execute(opp()))
    assert report.status is ExecStatus.HALTED and risk.is_killed


def test_leg1_error_unreconcilable_halts():
    # Venue can't be reconciled (no account_snapshot) -> fail closed (halt).
    kalshi = FakeVenue("kalshi", [res("kalshi", Side.YES, OrderStatus.ERROR, 0, None)])
    poly = FakeVenue("poly", [])
    ex, risk = make_exec([kalshi, poly])
    report = asyncio.run(ex.execute(opp()))
    assert report.status is ExecStatus.HALTED and risk.is_killed


class FakeConfirmer:
    def __init__(self, results):
        self._results = results        # {venue: (OrderStatus, filled, avg)}

    async def confirm(self, venue, order_id, requested, timeout):
        # Unknown order -> non-terminal "no info" so _place keeps its REST result.
        return self._results.get(venue, (OrderStatus.ERROR, 0.0, None))


def make_maker_exec(venues, confirmer, timeout=0.01):
    risk = RiskManager(RiskLimits(max_position_per_market=1e9, max_total_exposure=1e12))
    ex = Executor({v.name: v for v in venues}, risk,
                  fee_models={v.name: ZeroFeeModel() for v in venues},
                  max_order_contracts=0, fill_confirmer=confirmer, maker_timeout=timeout)
    return ex, risk


def test_execute_maker_fills_then_hedges_locks_arb():
    # Kalshi NO rests as a maker, fills (via the confirmer), then the Poly YES taker
    # fills -> locked arb. No unwind, thin-edge friendly.
    kalshi = FakeVenue("kalshi", [res("kalshi", Side.NO, OrderStatus.RESTING, 0, None)])
    poly = FakeVenue("poly", [res("poly", Side.YES, OrderStatus.FILLED, 5, 0.40)])
    ex, risk = make_maker_exec([kalshi, poly], FakeConfirmer({"kalshi": (OrderStatus.FILLED, 5, 0.55)}))
    report = asyncio.run(ex.execute_maker(opp(yv="poly", nv="kalshi", max_contracts=5,
                                              yes_price=0.40, no_price=0.55)))
    assert report.status is ExecStatus.SUCCESS
    assert round(report.realized_pnl, 4) == round(5 * (1 - 0.40 - 0.55), 4)
    assert kalshi.calls[0][6] is True              # the kalshi leg was a maker (post_only)
    assert kalshi.calls[0][3] == 0.54              # posted one tick INSIDE the 0.55 ask
    assert poly.calls[0][2] == "buy"               # poly taken as the hedge


def test_execute_maker_hedge_reprices_off_live_book():
    # Poly moved while the maker rested: the hedge must cross the LIVE ask (0.50), not
    # the stale opp price (0.40) — otherwise it kills and forces an unwind.
    from bot.models import MarketQuote
    moved = MarketQuote(venue="poly", market_id="K1", title="", yes_ask=0.50, no_ask=0.50)
    kalshi = FakeVenue("kalshi", [res("kalshi", Side.NO, OrderStatus.RESTING, 0, None)])
    poly = QuotingVenue("poly", [res("poly", Side.YES, OrderStatus.FILLED, 5, 0.50)], moved)
    ex, risk = make_maker_exec([kalshi, poly], FakeConfirmer({"kalshi": (OrderStatus.FILLED, 5, 0.55)}))
    report = asyncio.run(ex.execute_maker(opp(yv="poly", nv="kalshi", max_contracts=5,
                                              yes_price=0.40, no_price=0.55)))
    assert report.status is ExecStatus.SUCCESS
    assert poly.calls[0][3] == 0.50                # crossed the live ask, not the stale 0.40


class OrderFillVenue(FakeVenue):
    """FakeVenue that also answers order_filled_qty — the authoritative per-order fill the
    executor reconciles against. fill_qty=None simulates an unreadable order (read failed)."""
    def __init__(self, name, responses, fill_qty=None, hedge_depth=1e9):
        super().__init__(name, responses, hedge_depth)
        self._fill_qty = fill_qty

    async def order_filled_qty(self, order_id, requested, side):
        return self._fill_qty


def test_maker_reconciles_underreported_confirmer_fill():
    # The confirmer reports the maker as UNFILLED (0), but the venue order shows 17 actually
    # filled (a Polymarket maker partial the private stream missed). The bot must trust the
    # venue and HEDGE the true 17 — not walk away leaving them naked (the RECONCILE HALT bug).
    kalshi = OrderFillVenue("kalshi", [res("kalshi", Side.NO, OrderStatus.RESTING, 0, None)],
                            fill_qty=17)
    poly = FakeVenue("poly", [res("poly", Side.YES, OrderStatus.FILLED, 17, 0.40)])
    ex, risk = make_maker_exec([kalshi, poly],
                               FakeConfirmer({"kalshi": (OrderStatus.KILLED, 0, None)}))
    report = asyncio.run(ex.execute_maker(opp(yv="poly", nv="kalshi", max_contracts=20,
                                              yes_price=0.40, no_price=0.55)))
    assert report.status is ExecStatus.SUCCESS
    assert poly.calls[0][2] == "buy" and poly.calls[0][4] == 17   # hedged the TRUE fill


def test_maker_halts_when_confirmer_zero_and_venue_unreadable():
    # Confirmer says 0 AND the venue order can't be read -> ambiguous. Fail closed (HALT),
    # never guess 0 and leak a possibly-naked fill, never hedge a fill that may not exist.
    kalshi = OrderFillVenue("kalshi", [res("kalshi", Side.NO, OrderStatus.RESTING, 0, None)],
                            fill_qty=None)
    poly = FakeVenue("poly", [])
    ex, risk = make_maker_exec([kalshi, poly],
                               FakeConfirmer({"kalshi": (OrderStatus.KILLED, 0, None)}))
    report = asyncio.run(ex.execute_maker(opp(yv="poly", nv="kalshi", max_contracts=20)))
    assert report.status is ExecStatus.HALTED and risk.is_killed
    assert poly.calls == []                                       # never hedged a guessed fill


def test_maker_venue_confirms_truly_unfilled_is_clean_skip():
    # Venue authoritatively confirms 0 filled -> a clean no-trade skip, NOT a halt.
    kalshi = OrderFillVenue("kalshi", [res("kalshi", Side.NO, OrderStatus.RESTING, 0, None)],
                            fill_qty=0.0)
    poly = FakeVenue("poly", [])
    ex, risk = make_maker_exec([kalshi, poly],
                               FakeConfirmer({"kalshi": (OrderStatus.KILLED, 0, None)}))
    report = asyncio.run(ex.execute_maker(opp(yv="poly", nv="kalshi", max_contracts=20)))
    assert report.status is ExecStatus.SKIPPED and not risk.is_killed
    assert poly.calls == []


class SlowConfirmer:
    """Confirmer that returns a fixed terminal result after a delay — lets the drift
    guard poll at least once before the maker resolves."""

    def __init__(self, result, delay=0.0):
        self._result = result
        self._delay = delay

    async def confirm(self, venue, order_id, requested, timeout):
        await asyncio.sleep(self._delay)
        return self._result


class CancelVenue(FakeVenue):
    """FakeVenue that records cancel_order calls (the maker venue in drift tests)."""

    def __init__(self, name, responses):
        super().__init__(name, responses)
        self.cancelled = []

    async def cancel_order(self, order_id):
        self.cancelled.append(order_id)
        return {"ok": True}


def test_execute_maker_cancels_on_adverse_drift():
    # While the maker rests, the taker (Poly YES) drifts up to 0.50 so the would-be hedge
    # (NO 0.54 + YES 0.50 = 1.04) can no longer lock the floor -> cancel the maker before it
    # fills. No hedge is ever taken, no loss locked.
    from bot.models import MarketQuote
    kalshi = CancelVenue("kalshi", [res("kalshi", Side.NO, OrderStatus.RESTING, 0, None)])
    drifted = MarketQuote(venue="poly", market_id="P1", title="", yes_ask=0.50, no_ask=0.50)
    poly = QuotingVenue("poly", [], drifted)        # hedge place_order must never happen
    risk = RiskManager(RiskLimits(min_edge=0.01, max_position_per_market=1e9, max_total_exposure=1e12))
    ex = Executor({v.name: v for v in [kalshi, poly]}, risk,
                  fee_models={v.name: ZeroFeeModel() for v in [kalshi, poly]},
                  max_order_contracts=0,
                  fill_confirmer=SlowConfirmer((OrderStatus.KILLED, 0, None), delay=0.1),
                  maker_timeout=0.5, maker_arm_cushion=0.0, maker_poll=0.01)
    report = asyncio.run(ex.execute_maker(opp(yv="poly", nv="kalshi", max_contracts=5,
                                              yes_price=0.40, no_price=0.55)))
    assert report.status is ExecStatus.SKIPPED and "unfilled" in report.reason
    assert kalshi.cancelled == ["o"]                # maker cancelled on drift
    assert poly.calls == []                          # never hedged -> no loss


def test_execute_maker_no_drift_still_fills():
    # Taker stays put while resting -> guard never cancels, maker fills, hedge locks.
    from bot.models import MarketQuote
    steady = MarketQuote(venue="poly", market_id="P1", title="", yes_ask=0.40, no_ask=0.40)
    kalshi = CancelVenue("kalshi", [res("kalshi", Side.NO, OrderStatus.RESTING, 0, None)])
    poly = QuotingVenue("poly", [res("poly", Side.YES, OrderStatus.FILLED, 5, 0.40)], steady)
    risk = RiskManager(RiskLimits(min_edge=0.01, max_position_per_market=1e9, max_total_exposure=1e12))
    ex = Executor({v.name: v for v in [kalshi, poly]}, risk,
                  fee_models={v.name: ZeroFeeModel() for v in [kalshi, poly]},
                  max_order_contracts=0,
                  fill_confirmer=SlowConfirmer((OrderStatus.FILLED, 5, 0.54), delay=0.0),
                  maker_timeout=0.5, maker_arm_cushion=0.0, maker_poll=0.05)
    report = asyncio.run(ex.execute_maker(opp(yv="poly", nv="kalshi", max_contracts=5,
                                              yes_price=0.40, no_price=0.55)))
    assert report.status is ExecStatus.SUCCESS and kalshi.cancelled == []


def test_maker_depth_guards_on_hedge_leg_not_min():
    # Maker (kalshi NO) leg is thin (5) but the HEDGE (poly YES) leg is deep (200). The
    # resting maker adds liquidity, so its own thin book is irrelevant — the guard must
    # check the hedge leg (200 >= 10) and proceed, sizing against the hedge depth.
    kalshi = FakeVenue("kalshi", [res("kalshi", Side.NO, OrderStatus.RESTING, 0, None)])
    poly = FakeVenue("poly", [res("poly", Side.YES, OrderStatus.FILLED, 5, 0.40)])
    risk = RiskManager(RiskLimits(min_edge=0.01, max_position_per_market=1e9, max_total_exposure=1e12))
    ex = Executor({v.name: v for v in [kalshi, poly]}, risk,
                  fee_models={v.name: ZeroFeeModel() for v in [kalshi, poly]},
                  max_order_contracts=0, min_leg_depth=10,
                  fill_confirmer=FakeConfirmer({"kalshi": (OrderStatus.FILLED, 5, 0.54)}),
                  maker_timeout=0.01)
    o = opp(yv="poly", nv="kalshi", max_contracts=5, yes_price=0.40, no_price=0.55)
    o.yes_size, o.no_size = 200, 5            # hedge (poly YES) deep, maker (kalshi NO) thin
    report = asyncio.run(ex.execute_maker(o))
    assert report.status is ExecStatus.SUCCESS            # NOT skipped for thin book
    assert kalshi.calls[0][4] == 200                      # sized against the hedge depth, not 5


def test_maker_depth_falls_back_to_min_when_sizes_unknown():
    # Without per-leg sizes (both 0), the guard falls back to max_contracts — old behavior.
    kalshi = FakeVenue("kalshi", [res("kalshi", Side.NO, OrderStatus.RESTING, 0, None)])
    poly = FakeVenue("poly", [])
    risk = RiskManager(RiskLimits(min_edge=0.01, max_position_per_market=1e9, max_total_exposure=1e12))
    ex = Executor({v.name: v for v in [kalshi, poly]}, risk,
                  fee_models={v.name: ZeroFeeModel() for v in [kalshi, poly]},
                  max_order_contracts=0, min_leg_depth=10,
                  fill_confirmer=FakeConfirmer({}), maker_timeout=0.01)
    o = opp(yv="poly", nv="kalshi", max_contracts=5, yes_price=0.40, no_price=0.55)  # sizes default 0
    report = asyncio.run(ex.execute_maker(o))
    assert report.status is ExecStatus.SKIPPED and "thin hedge book" in report.reason
    assert kalshi.calls == []


def test_execute_maker_thin_edge_below_cushion_skips():
    # A sub-cushion edge must NOT arm a maker: while it rests the taker can drift against
    # it, and the post-fill hedge is forced — a thin edge that drifts locks a guaranteed
    # loss. The maker is never even placed.
    kalshi = FakeVenue("kalshi", [res("kalshi", Side.NO, OrderStatus.RESTING, 0, None)])
    poly = FakeVenue("poly", [])
    risk = RiskManager(RiskLimits(min_edge=0.01, max_position_per_market=1e9, max_total_exposure=1e12))
    ex = Executor({v.name: v for v in [kalshi, poly]}, risk,
                  fee_models={v.name: ZeroFeeModel() for v in [kalshi, poly]},
                  max_order_contracts=0, fill_confirmer=FakeConfirmer({}),
                  maker_timeout=0.01, maker_arm_cushion=0.05)
    # edge 0.02 (0.45 + 0.53) < floor 0.01 + cushion 0.05 = 0.06 -> skip, no order.
    report = asyncio.run(ex.execute_maker(opp(yv="poly", nv="kalshi", max_contracts=5,
                                              yes_price=0.45, no_price=0.53)))
    assert report.status is ExecStatus.SKIPPED and "cushion" in report.reason
    assert kalshi.calls == []                       # never rested a maker


def test_execute_maker_fat_edge_above_cushion_arms():
    # The cushion only blocks thin edges: an edge clearing floor+cushion still arms.
    kalshi = FakeVenue("kalshi", [res("kalshi", Side.NO, OrderStatus.RESTING, 0, None)])
    poly = FakeVenue("poly", [res("poly", Side.YES, OrderStatus.FILLED, 5, 0.40)])
    risk = RiskManager(RiskLimits(min_edge=0.01, max_position_per_market=1e9, max_total_exposure=1e12))
    ex = Executor({v.name: v for v in [kalshi, poly]}, risk,
                  fee_models={v.name: ZeroFeeModel() for v in [kalshi, poly]},
                  max_order_contracts=0, fill_confirmer=FakeConfirmer({"kalshi": (OrderStatus.FILLED, 5, 0.50)}),
                  maker_timeout=0.01, maker_arm_cushion=0.05)
    # edge 0.10 (0.40 + 0.50) >= floor 0.01 + cushion 0.05 -> arms and locks.
    report = asyncio.run(ex.execute_maker(opp(yv="poly", nv="kalshi", max_contracts=5,
                                              yes_price=0.40, no_price=0.50)))
    assert report.status is ExecStatus.SUCCESS and kalshi.calls[0][6] is True


def test_execute_maker_unfilled_is_no_trade():
    kalshi = FakeVenue("kalshi", [res("kalshi", Side.NO, OrderStatus.RESTING, 0, None)])
    poly = FakeVenue("poly", [])                    # hedge must never be attempted
    ex, risk = make_maker_exec([kalshi, poly], FakeConfirmer({"kalshi": (OrderStatus.KILLED, 0, None)}))
    report = asyncio.run(ex.execute_maker(opp(yv="poly", nv="kalshi", max_contracts=5)))
    assert report.status is ExecStatus.SKIPPED and "unfilled" in report.reason
    assert poly.calls == [] and not risk.is_killed


def test_execute_maker_hedge_fails_unwinds_maker():
    # Maker fills; the Poly hedge KILLs (no fill) -> re-cross once (hedge_retries=1), still
    # nothing -> unwind the (rare) filled maker leg. A clean KILLED is flat, so re-crossing
    # can't double up; exhausting the retries falls through to the unwind.
    kalshi = FakeVenue("kalshi", [
        res("kalshi", Side.NO, OrderStatus.RESTING, 0, None),
        res("kalshi", Side.NO, OrderStatus.FILLED, 5, 0.54, action="sell"),
    ])
    poly = FakeVenue("poly", [
        res("poly", Side.YES, OrderStatus.KILLED, 0, None),     # initial hedge: no fill
        res("poly", Side.YES, OrderStatus.KILLED, 0, None),     # re-cross: still no fill
    ])
    ex, risk = make_maker_exec([kalshi, poly], FakeConfirmer({"kalshi": (OrderStatus.FILLED, 5, 0.55)}))
    report = asyncio.run(ex.execute_maker(opp(yv="poly", nv="kalshi", max_contracts=5,
                                              yes_price=0.40, no_price=0.55)))
    assert report.status is ExecStatus.UNWOUND and not risk.is_killed
    assert kalshi.calls[1][2] == "sell"            # the maker NO leg was sold back
    assert len(poly.calls) == 2                    # hedge was re-crossed before unwinding


def test_execute_maker_partial_hedge_recrosses_then_settles():
    # Maker (kalshi NO) fills 5; the Poly YES hedge only PARTIAL-fills 3, so the remaining 2
    # is re-crossed at the live ask and fills -> fully hedged -> locked arb, no naked leg.
    from bot.models import MarketQuote
    quote = MarketQuote(venue="poly", market_id="P1", title="", yes_ask=0.40, no_ask=0.40)
    kalshi = FakeVenue("kalshi", [res("kalshi", Side.NO, OrderStatus.RESTING, 0, None)])
    poly = QuotingVenue("poly", [
        res("poly", Side.YES, OrderStatus.PARTIAL, 3, 0.40),   # initial hedge: 3 of 5
        res("poly", Side.YES, OrderStatus.FILLED, 2, 0.40),    # re-cross: the last 2
    ], quote)
    ex, risk = make_maker_exec(
        [kalshi, poly],
        FakeConfirmer({"kalshi": (OrderStatus.FILLED, 5, 0.55), "poly": (OrderStatus.PARTIAL, 3, 0.40)}))
    report = asyncio.run(ex.execute_maker(opp(yv="poly", nv="kalshi", max_contracts=5,
                                              yes_price=0.40, no_price=0.55)))
    assert report.status is ExecStatus.SUCCESS and not risk.is_killed
    assert round(report.realized_pnl, 4) == round(5 * (1 - 0.40 - 0.55), 4)   # full 5 locked
    assert len(poly.calls) == 2                    # hedged in two crossings
    assert poly.calls[1][4] == 2                   # re-cross was for the unhedged remainder


def test_execute_maker_partial_hedge_settles_matched_unwinds_excess():
    # Maker fills 5; Poly hedges 3 then can't fill the rest (re-cross KILLs). End flat-or-
    # locked: settle the matched 3 as an arb and UNWIND the unhedged 2 of the maker leg.
    from bot.models import MarketQuote
    quote = MarketQuote(venue="kalshi", market_id="K1", title="", yes_ask=0.46, no_ask=0.46)
    kalshi = QuotingVenue("kalshi", [
        res("kalshi", Side.NO, OrderStatus.RESTING, 0, None),
        res("kalshi", Side.NO, OrderStatus.FILLED, 2, 0.53, action="sell"),   # unwind the excess 2
    ], quote)
    poly = QuotingVenue("poly", [
        res("poly", Side.YES, OrderStatus.PARTIAL, 3, 0.40),   # initial hedge: 3 of 5
        res("poly", Side.YES, OrderStatus.KILLED, 0, None),    # re-cross: nothing more available
    ], MarketQuote(venue="poly", market_id="P1", title="", yes_ask=0.40, no_ask=0.40))
    ex, risk = make_maker_exec(
        [kalshi, poly],
        FakeConfirmer({"kalshi": (OrderStatus.FILLED, 5, 0.55), "poly": (OrderStatus.PARTIAL, 3, 0.40)}))
    report = asyncio.run(ex.execute_maker(opp(yv="poly", nv="kalshi", max_contracts=5,
                                              yes_price=0.40, no_price=0.55)))
    # Locked 3 (+0.15) and unwound the naked 2 (~-0.04) -> never halts, never naked.
    assert report.status is ExecStatus.UNWOUND and not risk.is_killed
    assert kalshi.calls[-1][2] == "sell" and kalshi.calls[-1][4] == 2   # sold back the excess 2
    matched = 3 * (1 - 0.40 - 0.55)
    unwind = 2 * (0.53 - 0.55)
    assert round(report.realized_pnl, 4) == round(matched + unwind, 4)


def test_execute_maker_gtc_cancels_unfilled_on_timeout():
    # A GOOD_TILL_CANCEL maker does not self-expire: if it never fills, the executor must
    # explicitly CANCEL it at the timeout (else it would rest unhedged). maker_poll=0 path.
    kalshi = CancelVenue("kalshi", [res("kalshi", Side.NO, OrderStatus.RESTING, 0, None)])
    poly = FakeVenue("poly", [])                    # hedge must never be placed
    risk = RiskManager(RiskLimits(min_edge=0.01, max_position_per_market=1e9, max_total_exposure=1e12))
    ex = Executor({v.name: v for v in [kalshi, poly]}, risk,
                  fee_models={v.name: ZeroFeeModel() for v in [kalshi, poly]},
                  max_order_contracts=0,
                  fill_confirmer=SlowConfirmer((OrderStatus.KILLED, 0, None), delay=0.1),
                  maker_timeout=0.02, maker_arm_cushion=0.0, maker_poll=0.0)
    report = asyncio.run(ex.execute_maker(opp(yv="poly", nv="kalshi", max_contracts=5,
                                              yes_price=0.40, no_price=0.55)))
    assert report.status is ExecStatus.SKIPPED and "unfilled" in report.reason
    assert kalshi.cancelled == ["o"]                # GTC maker explicitly cancelled at timeout
    assert poly.calls == []                          # never hedged


def test_hedge_fillable_reads_live_book_not_preview():
    # _hedge_fillable must source from the live order book (the preview API can't simulate
    # fills): returns the buy-side top-of-book depth, 0 when there's no offer, 0 on a fetch
    # error (never proceed blind).
    yes = FakeVenue("kalshi", [])
    ex, _ = make_exec([yes, FakeVenue("poly", [])])

    class BookVenue:
        def __init__(self, q): self._q = q; self.name = "poly"
        async def fetch_quote(self, m): return self._q
    class BoomVenue:
        name = "poly"
        async def fetch_quote(self, m): raise RuntimeError("network")

    leg = ("poly", "P1", Side.NO, 0.55)
    # NO buy: depth comes from no_ask_size when a no_ask exists
    deep = BookVenue(MarketQuote(venue="poly", market_id="P1", title="",
                                 no_ask=0.46, no_ask_size=37.0))
    assert asyncio.run(ex._hedge_fillable(deep, leg, 5)) == 37.0
    # no resting offer on the side -> 0
    empty = BookVenue(MarketQuote(venue="poly", market_id="P1", title="", no_ask=None))
    assert asyncio.run(ex._hedge_fillable(empty, leg, 5)) == 0.0
    # fetch failure -> 0 (don't arm a leg we can't confirm)
    assert asyncio.run(ex._hedge_fillable(BoomVenue(), leg, 5)) == 0.0


def test_hedge_fillable_deep_cushion_fraction():
    # The deep-cushion: _hedge_fillable commits only `hedge_depth_fraction` of the shown
    # hedge depth, so a partial vanish before the FOK can't reject it. Deep books still let
    # the trade fire full size (the caller caps at min(size, this)); thin hedges size down.
    risk = RiskManager(RiskLimits(max_position_per_market=1e9, max_total_exposure=1e12))
    ex = Executor({"kalshi": FakeVenue("kalshi", []), "poly": FakeVenue("poly", [])}, risk,
                  fee_models={"kalshi": ZeroFeeModel(), "poly": ZeroFeeModel()},
                  hedge_depth_fraction=0.5)

    class BookVenue:
        def __init__(self, q): self._q = q; self.name = "poly"
        async def fetch_quote(self, m): return self._q
    leg = ("poly", "P1", Side.NO, 0.55)
    # shown hedge depth 40 -> commit only 20 (50% cushion). A deep book (40 >> a size of 5)
    # still fills the full 5 downstream; a thin book would cap the trade to 20.
    deep = BookVenue(MarketQuote(venue="poly", market_id="P1", title="",
                                 no_ask=0.46, no_ask_size=40.0))
    assert asyncio.run(ex._hedge_fillable(deep, leg, 5)) == 20.0


def test_execute_maker_skips_when_hedge_preview_fills_nothing():
    # The hedge leg's live book shows NO depth -> can't hedge. Don't arm a maker we can't
    # hedge (else the post-fill hedge KILLs and forces an unwind).
    kalshi = FakeVenue("kalshi", [])                # maker must never be placed
    poly = PreviewVenue("poly", [], preview_filled=0)
    ex, risk = make_maker_exec([kalshi, poly], FakeConfirmer({}))
    report = asyncio.run(ex.execute_maker(opp(yv="poly", nv="kalshi", max_contracts=5,
                                              yes_price=0.40, no_price=0.55)))
    assert report.status is ExecStatus.SKIPPED and "hedge unfillable" in report.reason
    assert kalshi.calls == []                        # never armed the maker


def test_execute_maker_sizes_down_to_hedge_fillable():
    # The hedge would only fill 3 of 5 -> arm the maker for 3, not 5 (so the whole fill
    # hedges) instead of arming 5 and unwinding the unhedgeable 2.
    kalshi = FakeVenue("kalshi", [res("kalshi", Side.NO, OrderStatus.RESTING, 0, None)])
    poly = PreviewVenue("poly", [res("poly", Side.YES, OrderStatus.FILLED, 3, 0.40)], preview_filled=3)
    ex, risk = make_maker_exec([kalshi, poly], FakeConfirmer({"kalshi": (OrderStatus.FILLED, 3, 0.55)}))
    report = asyncio.run(ex.execute_maker(opp(yv="poly", nv="kalshi", max_contracts=5,
                                              yes_price=0.40, no_price=0.55)))
    assert report.status is ExecStatus.SUCCESS
    assert kalshi.calls[0][4] == 3                   # maker armed for the fillable 3, not 5
    assert round(report.realized_pnl, 4) == round(3 * (1 - 0.40 - 0.55), 4)


def test_maker_cancel_http_4xx_is_handled_gracefully():
    # A 400/404 on the GTC cancel means the order is already not cancellable (filled/expired)
    # -> swallow it (log with body), never crash the loop. The maker still resolves as a
    # clean no-trade; a genuinely stranded order is caught by the periodic reconciler.
    class _Resp:
        status_code = 400
        text = '{"message":"order not in a cancellable state"}'

    class _Err(Exception):
        response = _Resp()

    class Cancel400Venue(FakeVenue):
        def __init__(self, name, responses):
            super().__init__(name, responses)
            self.cancel_attempts = 0

        async def cancel_order(self, order_id):
            self.cancel_attempts += 1
            raise _Err()

    kalshi = Cancel400Venue("kalshi", [res("kalshi", Side.NO, OrderStatus.RESTING, 0, None)])
    poly = FakeVenue("poly", [])
    risk = RiskManager(RiskLimits(min_edge=0.01, max_position_per_market=1e9, max_total_exposure=1e12))
    ex = Executor({v.name: v for v in [kalshi, poly]}, risk,
                  fee_models={v.name: ZeroFeeModel() for v in [kalshi, poly]},
                  max_order_contracts=0,
                  fill_confirmer=SlowConfirmer((OrderStatus.KILLED, 0, None), delay=0.1),
                  maker_timeout=0.02, maker_arm_cushion=0.0, maker_poll=0.0)
    report = asyncio.run(ex.execute_maker(opp(yv="poly", nv="kalshi", max_contracts=5,
                                              yes_price=0.40, no_price=0.55)))
    assert report.status is ExecStatus.SKIPPED and "unfilled" in report.reason
    assert kalshi.cancel_attempts == 1               # cancel attempted; 4xx swallowed, no crash
    assert poly.calls == []


def test_execute_maker_hedge_error_flat_unwinds():
    # Maker fills; the Poly hedge ERRORS (500/timeout) and reconciles to FLAT. After the
    # retries are exhausted it UNWINDS the naked maker fill instead of halting and holding
    # it into settlement (the Ruzic loss).
    kalshi = FakeVenue("kalshi", [
        res("kalshi", Side.NO, OrderStatus.RESTING, 0, None),
        res("kalshi", Side.NO, OrderStatus.FILLED, 5, 0.54, action="sell"),
    ])
    poly = SnapshotVenue("poly", [
        res("poly", Side.YES, OrderStatus.ERROR, 0, None),
        res("poly", Side.YES, OrderStatus.ERROR, 0, None),    # retry also errors -> unwind
    ], [])
    ex, risk = make_maker_exec([kalshi, poly], FakeConfirmer({"kalshi": (OrderStatus.FILLED, 5, 0.55)}))
    report = asyncio.run(ex.execute_maker(opp(yv="poly", nv="kalshi", max_contracts=5,
                                              yes_price=0.40, no_price=0.55)))
    assert report.status is ExecStatus.UNWOUND and not risk.is_killed
    assert kalshi.calls[1][2] == "sell"            # the naked maker leg was flattened


def test_execute_maker_hedge_error_retries_then_settles():
    # Maker fills; the first hedge ERRORS but reconciles to FLAT (nothing landed) -> retry;
    # the retry FILLS -> locked arb. A transient venue error doesn't cost a round trip.
    kalshi = FakeVenue("kalshi", [res("kalshi", Side.NO, OrderStatus.RESTING, 0, None)])
    poly = SnapshotVenue("poly", [
        res("poly", Side.YES, OrderStatus.ERROR, 0, None),     # transient hedge error
        res("poly", Side.YES, OrderStatus.FILLED, 5, 0.40),    # retry fills
    ], [])
    ex, risk = make_maker_exec([kalshi, poly], FakeConfirmer({"kalshi": (OrderStatus.FILLED, 5, 0.55)}))
    report = asyncio.run(ex.execute_maker(opp(yv="poly", nv="kalshi", max_contracts=5,
                                              yes_price=0.40, no_price=0.55)))
    assert report.status is ExecStatus.SUCCESS and not risk.is_killed
    assert len(poly.calls) == 2                    # retried the hedge once, then filled


def test_execute_maker_hedge_error_present_settles():
    # Maker fills; the Poly hedge ERRORS but the venue actually holds the hedge -> settle
    # the locked arb rather than freezing.
    from bot.execution.account import VenuePosition
    kalshi = FakeVenue("kalshi", [res("kalshi", Side.NO, OrderStatus.RESTING, 0, None)])
    poly = SnapshotVenue("poly", [res("poly", Side.YES, OrderStatus.ERROR, 0, None)],
                         [VenuePosition("K1", 5, 0)])
    ex, risk = make_maker_exec([kalshi, poly], FakeConfirmer({"kalshi": (OrderStatus.FILLED, 5, 0.55)}))
    report = asyncio.run(ex.execute_maker(opp(yv="poly", nv="kalshi", max_contracts=5,
                                              yes_price=0.40, no_price=0.55)))
    assert report.status is ExecStatus.SUCCESS and not risk.is_killed


def test_maker_dynamic_rests_on_thin_leg():
    # yes leg (poly) is thin (size 5), no leg (kalshi) is deep (200) -> rest the maker on
    # the THIN poly leg (post_only), TAKE the deep kalshi leg.
    poly = FakeVenue("poly", [res("poly", Side.YES, OrderStatus.RESTING, 0, None)])
    kalshi = FakeVenue("kalshi", [res("kalshi", Side.NO, OrderStatus.FILLED, 5, 0.55)])
    risk = RiskManager(RiskLimits(max_position_per_market=1e9, max_total_exposure=1e12))
    ex = Executor({"poly": poly, "kalshi": kalshi}, risk,
                  fee_models={"poly": ZeroFeeModel(), "kalshi": ZeroFeeModel()},
                  max_order_contracts=5, fill_confirmer=FakeConfirmer({"poly": (OrderStatus.FILLED, 5, 0.40)}),
                  maker_timeout=0.01, maker_dynamic=True)
    o = ArbOpportunity(
        event_key="E", buy_yes_venue="poly", buy_yes_market="P1",
        buy_no_venue="kalshi", buy_no_market="K1", yes_price=0.40, no_price=0.55,
        gross_cost=0.95, fee_per_pair=0.0, edge_per_contract=0.05, max_contracts=5,
        total_fees=0.0, total_profit=0.25, notional=4.75, yes_size=5, no_size=200)
    report = asyncio.run(ex.execute_maker(o))
    assert poly.calls[0][1] == "YES" and poly.calls[0][6] is True   # poly maker, post_only
    assert kalshi.calls[0][1] == "NO"                                # kalshi taker hedge
    assert report.status is ExecStatus.SUCCESS


def test_execute_maker_rejected_when_would_cross():
    kalshi = FakeVenue("kalshi", [res("kalshi", Side.NO, OrderStatus.REJECTED, 0, None)])
    poly = FakeVenue("poly", [])
    ex, risk = make_maker_exec([kalshi, poly], FakeConfirmer({"kalshi": (OrderStatus.FILLED, 5, 0.55)}))
    report = asyncio.run(ex.execute_maker(opp(yv="poly", nv="kalshi", max_contracts=5)))
    assert report.status is ExecStatus.SKIPPED and "cross" in report.reason
    assert poly.calls == []


def test_leg2_error_halts_and_trips_kill_switch():
    yes = FakeVenue("kalshi", [res("kalshi", Side.YES, OrderStatus.FILLED, 2, 0.40)])
    no = FakeVenue("poly", [res("poly", Side.NO, OrderStatus.ERROR, 0, None)])
    ex, risk = make_exec([yes, no])
    report = asyncio.run(ex.execute(opp()))
    assert report.status is ExecStatus.HALTED
    assert risk.is_killed                            # ambiguous hedge state -> stop everything


class PreviewVenue(FakeVenue):
    """FakeVenue whose hedge leg's live book shows ``preview_filled`` contracts of depth —
    the quantity the executor's hedge-fillability check will read and size to / skip on."""

    def __init__(self, name, responses, preview_filled):
        super().__init__(name, responses, hedge_depth=preview_filled)


def test_hedge_preview_skips_when_would_fill_nothing():
    # The hedge preview says leg 2 would fill ZERO (phantom depth) -> skip before placing
    # leg 1. No naked leg, no synchronous FOK into empty liquidity.
    yes = FakeVenue("kalshi", [res("kalshi", Side.YES, OrderStatus.FILLED, 2, 0.40)])
    no = PreviewVenue("poly", [res("poly", Side.NO, OrderStatus.FILLED, 2, 0.55)],
                      preview_filled=0)
    ex, risk = make_exec([yes, no])
    report = asyncio.run(ex.execute(opp()))
    assert report.status is ExecStatus.SKIPPED
    assert yes.calls == [] and no.calls == []         # nothing placed
    assert not risk.is_killed


def test_hedge_preview_sizes_down_to_fillable():
    # The hedge would fill only 1 of the 2 we'd take -> size the WHOLE arb down to 1 and
    # lock it (capture the liquidity that's there instead of skipping).
    yes = FakeVenue("kalshi", [res("kalshi", Side.YES, OrderStatus.FILLED, 1, 0.40)])
    no = PreviewVenue("poly", [res("poly", Side.NO, OrderStatus.FILLED, 1, 0.55)],
                      preview_filled=1)
    ex, risk = make_exec([yes, no])                   # max_order_contracts=2 -> size 2
    report = asyncio.run(ex.execute(opp()))
    assert report.status is ExecStatus.SUCCESS
    assert yes.calls[0][4] == 1 and no.calls[0][4] == 1   # both legs sized to fillable 1


def test_hedge_preview_proceeds_when_fills():
    # The hedge preview confirms a full fill -> proceed and lock the arb as normal.
    yes = FakeVenue("kalshi", [res("kalshi", Side.YES, OrderStatus.FILLED, 2, 0.40)])
    no = PreviewVenue("poly", [res("poly", Side.NO, OrderStatus.FILLED, 2, 0.55)],
                      preview_filled=2)
    ex, risk = make_exec([yes, no])
    report = asyncio.run(ex.execute(opp()))
    assert report.status is ExecStatus.SUCCESS
    assert len(yes.calls) == 1 and len(no.calls) == 1


def test_leg2_error_reconciles_hedge_present_settles():
    # A leg-2 ERROR (e.g. a Polymarket 500 on POST) where the venue actually HOLDS the
    # hedge -> the order landed and the arb is locked; settle it instead of freezing the
    # whole bot on a transient server error.
    from bot.execution.account import VenuePosition
    yes = FakeVenue("kalshi", [res("kalshi", Side.YES, OrderStatus.FILLED, 2, 0.40)])
    no = SnapshotVenue("poly", [res("poly", Side.NO, OrderStatus.ERROR, 0, None)],
                       [VenuePosition("P1", 2, 0)])
    ex, risk = make_exec([yes, no])
    report = asyncio.run(ex.execute(opp()))
    assert report.status is ExecStatus.SUCCESS
    assert not risk.is_killed                         # 500-but-filled doesn't freeze the bot


def test_leg2_error_reconciles_flat_unwinds():
    # A leg-2 ERROR where the venue is FLAT -> the synchronous FOK didn't fill (no hedge),
    # so unwind leg 1 and KEEP trading (one bad market doesn't freeze the whole bot).
    yes = FakeVenue("kalshi", [
        res("kalshi", Side.YES, OrderStatus.FILLED, 2, 0.40),
        res("kalshi", Side.YES, OrderStatus.FILLED, 2, 0.38, action="sell"),
    ])
    no = SnapshotVenue("poly", [res("poly", Side.NO, OrderStatus.ERROR, 0, None)], [])
    ex, risk = make_exec([yes, no])
    report = asyncio.run(ex.execute(opp()))
    assert report.status is ExecStatus.UNWOUND and not risk.is_killed


def test_leg2_error_partial_hedge_halts():
    # A leg-2 ERROR leaving a PARTIAL hedge (1 of 2) is a known-but-mismatched naked
    # remainder we can't auto-resolve -> halt for manual reconciliation.
    from bot.execution.account import VenuePosition
    yes = FakeVenue("kalshi", [res("kalshi", Side.YES, OrderStatus.FILLED, 2, 0.40)])
    no = SnapshotVenue("poly", [res("poly", Side.NO, OrderStatus.ERROR, 0, None)],
                       [VenuePosition("P1", 1, 0)])
    ex, risk = make_exec([yes, no])
    report = asyncio.run(ex.execute(opp()))
    assert report.status is ExecStatus.HALTED and risk.is_killed


def test_halt_records_held_legs_and_provisional_pnl():
    # A halt must not vanish from the books: record the held (filled) legs to `fills` and a
    # provisional HALT pnl row for the cash that moved, so a halt is reconcilable instead of
    # silently understating losses in the trade log.
    from bot.execution.account import VenuePosition
    store = Store(":memory:")
    yes = FakeVenue("kalshi", [res("kalshi", Side.YES, OrderStatus.FILLED, 2, 0.40)])
    no = SnapshotVenue("poly", [res("poly", Side.NO, OrderStatus.ERROR, 0, None)],
                       [VenuePosition("P1", 1, 0)])
    ex, risk = make_exec([yes, no], store=store)
    report = asyncio.run(ex.execute(opp()))
    assert report.status is ExecStatus.HALTED and risk.is_killed
    # the filled YES leg is now recorded (previously a halt recorded NOTHING but an audit row)
    fills = store.conn.execute("SELECT venue, side, contracts FROM fills").fetchall()
    assert any(f["venue"] == "kalshi" and f["side"] == "YES" and f["contracts"] == 2 for f in fills)
    # a single provisional HALT pnl row captures the cash out for the held leg
    pnl = store.conn.execute("SELECT amount, note FROM pnl").fetchall()
    assert len(pnl) == 1 and "HALT provisional" in pnl[0]["note"]
    assert round(pnl[0]["amount"], 4) == round(-2 * 0.40, 4)   # bought 2 YES @0.40 -> cash out
    assert round(report.realized_pnl, 4) == round(-2 * 0.40, 4)


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
    """FakeVenue that also answers fetch_quote (for the unwind's bid lookup and the
    hedge-fillability pre-check). Injects ``hedge_depth`` where the quote names a price
    but no size, so the hedge check sees real depth (these tests assert on price, not size)."""

    def __init__(self, name, responses, quote):
        super().__init__(name, responses)
        self._quote = quote

    async def fetch_quote(self, market):
        from dataclasses import replace
        q = self._quote
        return replace(
            q,
            yes_ask_size=q.yes_ask_size or (self.hedge_depth if q.yes_ask is not None else 0.0),
            no_ask_size=q.no_ask_size or (self.hedge_depth if q.no_ask is not None else 0.0),
        )


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


def test_places_kalshi_leg_first_when_kalshi_is_no():
    # Buy YES on poly + NO on kalshi: kalshi (the rejection-prone venue) is the NO leg,
    # so it's placed FIRST. If it rejects, poly is never touched -> clean skip, no unwind.
    kalshi = FakeVenue("kalshi", [res("kalshi", Side.NO, OrderStatus.REJECTED, 0, None)])
    poly = FakeVenue("poly", [])                       # must never be called
    ex, risk = make_exec([kalshi, poly])
    report = asyncio.run(ex.execute(opp(yv="poly", nv="kalshi")))
    assert report.status is ExecStatus.SKIPPED
    assert poly.calls == [] and kalshi.calls != []     # kalshi first, poly skipped
    assert kalshi.calls[0][1] == "NO"                  # the NO leg went first
    assert not risk.is_killed                          # no position -> no halt, no unwind


def test_unwinds_first_leg_when_kalshi_no_filled_then_poly_fails():
    # Kalshi NO fills first; poly YES then fails -> unwind the NO leg (sell NO back).
    kalshi = FakeVenue("kalshi", [
        res("kalshi", Side.NO, OrderStatus.FILLED, 5, 0.55),
        res("kalshi", Side.NO, OrderStatus.FILLED, 5, 0.53, action="sell"),
    ])
    poly = FakeVenue("poly", [res("poly", Side.YES, OrderStatus.KILLED, 0, None)])
    ex, risk = make_exec([kalshi, poly])
    report = asyncio.run(ex.execute(opp(yv="poly", nv="kalshi", max_contracts=5)))
    assert report.status is ExecStatus.UNWOUND and not risk.is_killed
    assert kalshi.calls[0][2] == "buy" and kalshi.calls[1][2] == "sell"
    assert kalshi.calls[1][1] == "NO"                  # unwound the NO side it bought


def test_depth_safety_trims_size():
    # depth_safety < 1 trades only a fraction of shown depth (FOK headroom).
    yes = FakeVenue("kalshi", [res("kalshi", Side.YES, OrderStatus.FILLED, 8, 0.40)])
    no = FakeVenue("poly", [res("poly", Side.NO, OrderStatus.FILLED, 8, 0.55)])
    risk = RiskManager(RiskLimits(max_position_per_market=1e9, max_total_exposure=1e12))
    ex = Executor({v.name: v for v in [yes, no]}, risk,
                  fee_models={v.name: ZeroFeeModel() for v in [yes, no]},
                  max_order_contracts=0, depth_safety=0.8)
    asyncio.run(ex.execute(opp(max_contracts=10)))     # 10 * 0.8 = 8
    assert yes.calls[0][4] == 8                         # contracts requested = trimmed depth


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


def test_leg_limits_buffer_goes_to_hedge_leg():
    # Buffer goes to the SECOND (hedge) leg up to hedge_buffer; leftover widens the
    # first. Here NO is the hedge (second) leg.
    ex = Executor({"kalshi": FakeVenue("kalshi", []), "poly": FakeVenue("poly", [])},
                  RiskManager(RiskLimits(min_edge=0.01)),
                  fee_models={"kalshi": ZeroFeeModel(), "poly": ZeroFeeModel()},
                  hedge_buffer=0.03)
    o = opp(yes_price=0.40, no_price=0.55)             # edge 0.05, floor 0.01, surplus 0.04
    first_limit, second_limit = ex._leg_limits(o, Side.YES, Side.NO)
    assert round(second_limit - 0.55, 4) == 0.03       # hedge (NO) gets the full buffer
    assert round(first_limit - 0.40, 4) == 0.01        # leftover 0.04-0.03 widens first
    assert round(1 - (first_limit + second_limit), 4) >= 0.01   # still locks the floor


def test_leg_limits_thin_edge_no_room():
    ex = Executor({"kalshi": FakeVenue("kalshi", []), "poly": FakeVenue("poly", [])},
                  RiskManager(RiskLimits(min_edge=0.05)),
                  fee_models={"kalshi": ZeroFeeModel(), "poly": ZeroFeeModel()},
                  hedge_buffer=0.03)
    o = opp(yes_price=0.45, no_price=0.50)             # edge 0.05 == floor -> no surplus
    first_limit, second_limit = ex._leg_limits(o, Side.YES, Side.NO)
    assert first_limit == 0.45 and second_limit == 0.50


def test_hedge_buffer_blocks_thin_edge():
    # An edge below lock + hedge_buffer must NOT fire (it would just unwind).
    yes = FakeVenue("kalshi", [])
    no = FakeVenue("poly", [])
    ex = Executor({"kalshi": yes, "poly": no},
                  RiskManager(RiskLimits(min_edge=0.01, max_position_per_market=1e9,
                                         max_total_exposure=1e12)),
                  fee_models={"kalshi": ZeroFeeModel(), "poly": ZeroFeeModel()},
                  max_order_contracts=0, hedge_buffer=0.03)
    report = asyncio.run(ex.execute(opp(yes_price=0.47, no_price=0.51)))  # edge 0.02 < 0.04
    assert report.status is ExecStatus.SKIPPED and "unwind" in report.reason
    assert yes.calls == [] and no.calls == []


def test_hedge_buffer_on_kalshi_first_buffers_the_poly_hedge():
    # The exact live case: Kalshi is the NO leg (placed first); the Poly YES hedge is
    # second and must get the buffer (not the first leg).
    kalshi = FakeVenue("kalshi", [res("kalshi", Side.NO, OrderStatus.FILLED, 2, 0.55)])
    poly = FakeVenue("poly", [res("poly", Side.YES, OrderStatus.FILLED, 2, 0.40)])
    ex = Executor({"kalshi": kalshi, "poly": poly},
                  RiskManager(RiskLimits(min_edge=0.01, max_position_per_market=1e9,
                                         max_total_exposure=1e12)),
                  fee_models={"kalshi": ZeroFeeModel(), "poly": ZeroFeeModel()},
                  max_order_contracts=0, hedge_buffer=0.03)
    asyncio.run(ex.execute(opp(yv="poly", nv="kalshi", yes_price=0.40, no_price=0.55)))
    assert kalshi.calls[0][1] == "NO"                  # kalshi placed first
    assert round(poly.calls[0][3] - 0.40, 4) == 0.03   # the Poly YES hedge got the buffer


def test_scaled_hedge_buffer_interpolates_by_depth():
    from bot.execution.executor import scaled_hedge_buffer
    # Off (deep_depth=0) -> always the full buffer.
    assert scaled_hedge_buffer(0.03, 99999, 10, 0) == 0.03
    # Thin book (<= thin_depth) -> full buffer; deep book (>= deep_depth) -> one-tick floor.
    assert scaled_hedge_buffer(0.03, 10, 10, 1000) == 0.03
    assert scaled_hedge_buffer(0.03, 1000, 10, 1000) == 0.01
    # Midway interpolates between full and floor.
    mid = scaled_hedge_buffer(0.03, 505, 10, 1000)
    assert 0.01 < mid < 0.03


def test_deep_book_fires_thinner_edge_than_thin_book():
    # With depth-scaling on, a 0.03 edge that a thin book SKIPS (full 0.03 buffer -> bar
    # 0.04) is TAKEN on a deep book (buffer scales to 0.01 -> bar 0.02). floor = 0.01.
    def run(depth):
        yes = FakeVenue("kalshi", [res("kalshi", Side.YES, OrderStatus.FILLED, 2, 0.40)])
        no = FakeVenue("poly", [res("poly", Side.NO, OrderStatus.FILLED, 2, 0.57)])
        risk = RiskManager(RiskLimits(max_position_per_market=1e9, max_total_exposure=1e12))
        ex = Executor({"kalshi": yes, "poly": no}, risk,
                      fee_models={"kalshi": ZeroFeeModel(), "poly": ZeroFeeModel()},
                      max_order_contracts=0, hedge_buffer=0.03, buffer_deep_depth=1000)
        o = opp(max_contracts=depth, yes_price=0.40, no_price=0.57)   # edge 0.03
        return asyncio.run(ex.execute(o))

    assert run(2).status is ExecStatus.SKIPPED        # thin -> full 0.03 buffer -> bar 0.04
    assert run(2000).status is ExecStatus.SUCCESS     # deep -> 0.01 buffer -> bar 0.02


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

    # 5xx (ambiguous — may have processed) -> ERROR, with the HTTP status + body captured
    # (a 500 often carries a diagnostic message saying WHY the order was refused).
    http500 = Exception("Server error")
    http500.response = SimpleNamespace(status_code=500, text="price off tick grid")
    r500 = order_error_result("kalshi", "M", Side.YES, "buy", 1, http500)
    assert r500.status is OS.ERROR and r500.raw["http_status"] == 500
    assert "off tick grid" in r500.raw["body"]

    # Network error (no response) -> ERROR, no body to capture.
    rnet = order_error_result("kalshi", "M", Side.YES, "buy", 1, TimeoutError("timed out"))
    assert rnet.status is OS.ERROR and rnet.raw["body"] is None


def test_risk_cap_skips():
    yes = FakeVenue("kalshi", [])
    no = FakeVenue("poly", [])
    ex, _ = make_exec([yes, no], limits=RiskLimits(max_position_per_market=0.5, max_total_exposure=0.5))
    report = asyncio.run(ex.execute(opp()))
    # A risk cap that leaves room for <1 contract is now caught at sizing, naming the
    # binding constraint (the per-market cap) rather than a generic risk rejection.
    assert report.status is ExecStatus.SKIPPED and "per_market" in report.reason
