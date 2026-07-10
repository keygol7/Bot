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
    no = FakeVenue("poly", [res("poly", Side.NO, OrderStatus.KILLED, 0, None),
                            res("poly", Side.NO, OrderStatus.KILLED, 0, None)])  # + recross
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
    no = FakeVenue("poly", [res("poly", Side.NO, OrderStatus.KILLED, 0, None),
                            res("poly", Side.NO, OrderStatus.KILLED, 0, None)])  # leg2+recross reject
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
    no = FakeVenue("poly", [res("poly", Side.NO, OrderStatus.KILLED, 0, None),
                            res("poly", Side.NO, OrderStatus.KILLED, 0, None)])
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


def test_rebalance_gate_keeps_favorites_on_drained_venue():
    # Settlement cash flows to the venue holding the WINNER (price ~= win probability), so
    # a drained venue must hold the FAVORITE leg — settlements then replenish it. The old
    # cheap-legs-only gate measurably death-spiraled Kalshi (-$198 settlement flow / 48h).
    def fresh():
        return (FakeVenue("kalshi", [res("kalshi", Side.YES, OrderStatus.FILLED, 2, 0.40)]),
                FakeVenue("poly", [res("poly", Side.NO, OrderStatus.FILLED, 2, 0.57)]))

    # kalshi drained ($15 < $30 floor) and this arb's kalshi (YES) leg is the LONGSHOT
    # (0.17 vs 0.80) -> skip: a longshot mostly loses at settlement and bleeds kalshi.
    y, n = fresh()
    ex, _ = make_exec([y, n])
    ex.rebalance_floor = 30.0
    ex._balances = {"kalshi": 15.0, "poly": 100.0}
    rep = asyncio.run(ex.execute(opp(yv="kalshi", nv="poly", yes_price=0.17, no_price=0.80)))
    assert rep.status is ExecStatus.SKIPPED and "LONGSHOT" in rep.reason and y.calls == []

    # same drained kalshi, but kalshi holds the FAVORITE (0.80) -> fires (the settlement
    # pays kalshi $1/contract with ~80% probability -> median replenishment)
    y, n = fresh()
    ex, _ = make_exec([y, n])
    ex.rebalance_floor = 30.0
    ex._balances = {"kalshi": 15.0, "poly": 100.0}
    rep = asyncio.run(ex.execute(opp(yv="kalshi", nv="poly", yes_price=0.80, no_price=0.17)))
    assert rep.status is ExecStatus.SUCCESS

    # both venues funded -> no rebalance gating even on a lopsided arb
    y, n = fresh()
    ex, _ = make_exec([y, n])
    ex.rebalance_floor = 30.0
    ex._balances = {"kalshi": 100.0, "poly": 100.0}
    rep = asyncio.run(ex.execute(opp(yv="kalshi", nv="poly", yes_price=0.80, no_price=0.17)))
    assert rep.status is ExecStatus.SUCCESS

    # BOTH venues below the floor -> NO funded venue to steer toward, so the gate must NOT
    # reserve (that would deadlock the bot into idle). It fires instead. Regression for the
    # both-drained deadlock.
    y, n = fresh()
    ex, _ = make_exec([y, n])
    ex.rebalance_floor = 40.0
    ex._balances = {"kalshi": 30.0, "poly": 39.0}        # both < $40 floor
    rep = asyncio.run(ex.execute(opp(yv="kalshi", nv="poly", yes_price=0.80, no_price=0.17)))
    assert rep.status is ExecStatus.SUCCESS and y.calls and n.calls


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
    # A market that KILL/REJECTed at probe size max_fails times is excluded for a COOLDOWN
    # (cap 0 while it lasts), then re-probes at probe size — never a permanent ratchet
    # (the old cap-0-forever meant the streak could never reset: 41 markets dead-listed).
    import time as _time
    yes = FakeVenue("kalshi", [])
    no = FakeVenue("poly", [])
    ex = _rel_exec([yes, no])
    ex._market_rel[("poly", "P1")] = (0, 2, 2, 0.0)    # poly leg proven phantom
    ex._excluded_until[("poly", "P1")] = _time.time() + 300   # cooling down
    size, caps = ex._max_size(opp(max_contracts=100))
    assert caps["reliability"] == 0 and size == 0
    # cooldown expired -> re-probes at probe size, NOT dead forever
    ex._excluded_until[("poly", "P1")] = _time.time() - 1
    _, caps2 = ex._max_size(opp(max_contracts=100))
    assert caps2["reliability"] == ex.probe_contracts


def test_reliability_excludes_proven_market_on_consecutive_fail_streak():
    # The Valorant case: a market proved real depth (5 fills) then its liquidity drained
    # mid-game (consecutive fails). Despite being "proven", a streak >= max_fails must
    # EXCLUDE it (cap 0) so it stops firing full size into vanished volume -> unwinds.
    yes = FakeVenue("kalshi", [])
    no = FakeVenue("poly", [])
    ex = _rel_exec([yes, no])
    import time as _time
    ex._market_rel[("kalshi", "K1")] = (5, 3, 2, 6.0)  # proven, but 2 consecutive fails now
    ex._market_rel[("poly", "P1")] = (8, 0, 0, 6.0)
    ex._excluded_until[("kalshi", "K1")] = _time.time() + 300
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


def test_leg1_partial_hedges_filled_portion():
    # FoK partial-filled leg1 (e.g. 0.62/6): we HOLD 0.62 — a good position. The old
    # behavior sold it straight back (the worst unwind category, -$32 all-time); now the
    # trade shrinks to the filled amount and hedges it -> a locked (smaller) arb.
    yes = FakeVenue("kalshi", [
        res("kalshi", Side.YES, OrderStatus.PARTIAL, 0.62, 0.40, requested=6),
    ])
    no = FakeVenue("poly", [
        res("poly", Side.NO, OrderStatus.FILLED, 0.62, 0.55, requested=0.62),
    ])
    ex, risk = make_exec([yes, no])
    report = asyncio.run(ex.execute(opp(max_contracts=6)))
    assert report.status is ExecStatus.SUCCESS        # hedged, not unwound
    assert not risk.is_killed
    assert no.calls[0][4] == 0.62                     # hedge sized to the PARTIAL amount


def test_leg1_partial_unwinds_only_when_hedge_and_recross_fail():
    # Partial leg1 whose hedge AND breakeven recross both fail -> unwind just the
    # filled portion (never more), bot keeps running.
    yes = FakeVenue("kalshi", [
        res("kalshi", Side.YES, OrderStatus.PARTIAL, 0.62, 0.40, requested=6),
        res("kalshi", Side.YES, OrderStatus.FILLED, 0.62, 0.38, action="sell", requested=0.62),
    ])
    no = FakeVenue("poly", [
        res("poly", Side.NO, OrderStatus.KILLED, 0, None),   # hedge FOK fails
        res("poly", Side.NO, OrderStatus.KILLED, 0, None),   # breakeven recross fails too
    ])
    ex, risk = make_exec([yes, no])
    report = asyncio.run(ex.execute(opp(max_contracts=6)))
    assert report.status is ExecStatus.UNWOUND
    assert not risk.is_killed
    assert yes.calls[1][2] == "sell" and yes.calls[1][4] == 0.62


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
    ex, risk = make_maker_exec([kalshi, poly], FakeConfirmer({"kalshi": (OrderStatus.FILLED, 5, 0.45)}))
    report = asyncio.run(ex.execute_maker(opp(yv="poly", nv="kalshi", max_contracts=5,
                                              yes_price=0.40, no_price=0.55)))
    assert report.status is ExecStatus.SUCCESS
    assert round(report.realized_pnl, 4) == round(5 * (1 - 0.40 - 0.55), 4)
    assert kalshi.calls[0][6] is True              # the kalshi leg was a maker (post_only)
    assert kalshi.calls[0][3] == 0.54              # posted one tick INSIDE the 0.55 ask
    assert poly.calls[0][2] == "buy"               # poly taken as the hedge


def test_maker_fee_credit_arms_thin_arb_that_taker_fee_would_skip():
    # A 2c raw spread on a mid-priced pair: at the Kalshi TAKER fee (0.07) the maker leg's
    # edge is ~0.25c < the 0.5c floor (skip), but at the real MAKER fee (0.0175) it's ~1.5c,
    # so it should arm and lock. Proves the rested leg is priced at the maker fee.
    from bot.fees import KalshiFeeModel

    def build(maker_model):
        kalshi = FakeVenue("kalshi", [res("kalshi", Side.NO, OrderStatus.RESTING, 0, None)])
        if maker_model is not None:
            kalshi.maker_fee_model = maker_model
        poly = FakeVenue("poly", [res("poly", Side.YES, OrderStatus.FILLED, 40, 0.49)])
        risk = RiskManager(RiskLimits(max_position_per_market=1e9, max_total_exposure=1e12))
        ex = Executor({"kalshi": kalshi, "poly": poly}, risk,
                      fee_models={"kalshi": KalshiFeeModel(0.07), "poly": ZeroFeeModel()},
                      max_order_contracts=0, min_lock_edge=0.005,
                      fill_confirmer=FakeConfirmer({"kalshi": (OrderStatus.FILLED, 40, 0.51)}),
                      maker_timeout=0.01)
        return ex
    o = opp(yv="poly", nv="kalshi", max_contracts=40, yes_price=0.49, no_price=0.49)
    # maker fee modeled -> arms one tick inside the ask and locks
    ex_maker = build(KalshiFeeModel(0.0175))
    assert asyncio.run(ex_maker.execute_maker(o)).status is ExecStatus.SUCCESS
    px_maker_fee = [c[3] for c in ex_maker.venues["kalshi"].calls if c[2] == "buy"][0]
    # taker fee on the rested leg (no maker model): previously SKIPPED (edge at the
    # ASK failed the arm bar); the maker now rests at a fee-adjusted cap that locks
    # >= floor + cushion at the actual resting price — it must arm, never above the
    # cap (1 - 0.49 - taker_fee_ct - arm ~= 0.4825)
    ex_taker = build(None)
    rep = asyncio.run(ex_taker.execute_maker(o))
    buys = [c[3] for c in ex_taker.venues["kalshi"].calls if c[2] == "buy"]
    assert buys, f"should rest, got {rep.status}: {rep.reason}"
    assert buys[0] <= 0.4825 + 1e-9


def test_execute_maker_hedge_reprices_off_live_book():
    # Poly moved while the maker rested: the hedge must cross the LIVE ask (0.50), not
    # the stale opp price (0.40) — otherwise it kills and forces an unwind.
    from bot.models import MarketQuote
    moved = MarketQuote(venue="poly", market_id="K1", title="", yes_ask=0.50, no_ask=0.50)
    kalshi = FakeVenue("kalshi", [res("kalshi", Side.NO, OrderStatus.RESTING, 0, None)])
    poly = QuotingVenue("poly", [res("poly", Side.YES, OrderStatus.FILLED, 5, 0.50)], moved)
    ex, risk = make_maker_exec([kalshi, poly], FakeConfirmer({"kalshi": (OrderStatus.FILLED, 5, 0.45)}))
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
    # Confirmer says 0 AND the venue order can't be read while the venue is HEALTHY
    # -> genuinely ambiguous. Fail closed (HALT + kill), never guess 0, never hedge
    # a fill that may not exist.
    kalshi = OrderFillVenue("kalshi", [res("kalshi", Side.NO, OrderStatus.RESTING, 0, None)],
                            fill_qty=None)
    async def _healthy(limit=1):
        return [object()]
    kalshi.list_markets = _healthy                 # venue-wide probe says HEALTHY
    poly = FakeVenue("poly", [])
    ex, risk = make_maker_exec([kalshi, poly],
                               FakeConfirmer({"kalshi": (OrderStatus.KILLED, 0, None)}))
    report = asyncio.run(ex.execute_maker(opp(yv="poly", nv="kalshi", max_contracts=20)))
    assert report.status is ExecStatus.HALTED and risk.is_killed
    assert poly.calls == []                                       # never hedged a guessed fill


def test_maker_fill_unreadable_during_outage_parks_no_kill():
    # Same ambiguity but the venue is DOWN venue-wide (portfolio API flapping,
    # 2026-07-09): park a recovery task instead of killing the whole bot.
    kalshi = OrderFillVenue("kalshi", [res("kalshi", Side.NO, OrderStatus.RESTING, 0, None)],
                            fill_qty=None)   # no list_markets -> probe says DOWN
    poly = FakeVenue("poly", [])
    ex, risk = make_maker_exec([kalshi, poly],
                               FakeConfirmer({"kalshi": (OrderStatus.KILLED, 0, None)}))
    report = asyncio.run(ex.execute_maker(opp(yv="poly", nv="kalshi", max_contracts=20)))
    assert report.status is ExecStatus.HALTED
    assert not risk.is_killed                     # recovery parked, bot keeps running
    assert "recovery parked" in report.reason
    assert "kalshi" in ex.venue_down
    for t in ex._recovery_tasks:
        t.cancel()


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
                  fill_confirmer=FakeConfirmer({"kalshi": (OrderStatus.FILLED, 5, 0.46)}),
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


def test_execute_maker_thin_edge_rests_deeper_at_safe_price():
    # A sub-cushion edge must never arm an ADVERSE-FILL maker — but a maker chooses
    # its price: it now rests DEEPER in the spread at the price that manufactures
    # floor + cushion against the current hedge ask (1 - 0.45 - 0.06 = 0.49), so a
    # fill can only lock >= the cushion, never a loss.
    kalshi = FakeVenue("kalshi", [res("kalshi", Side.NO, OrderStatus.RESTING, 0, None)])
    poly = FakeVenue("poly", [])
    risk = RiskManager(RiskLimits(min_edge=0.01, max_position_per_market=1e9, max_total_exposure=1e12))
    ex = Executor({v.name: v for v in [kalshi, poly]}, risk,
                  fee_models={v.name: ZeroFeeModel() for v in [kalshi, poly]},
                  max_order_contracts=0, fill_confirmer=FakeConfirmer({}),
                  maker_timeout=0.01, maker_arm_cushion=0.05)
    report = asyncio.run(ex.execute_maker(opp(yv="poly", nv="kalshi", max_contracts=5,
                                              yes_price=0.45, no_price=0.53)))
    assert kalshi.calls, "the maker should rest (deeper), not skip"
    assert kalshi.calls[0][3] <= 0.49 + 1e-9        # at/below the manufactured price
    # no adverse-fill exposure: a fill at 0.49 + hedge at 0.45 locks the 0.06 arm

def test_execute_maker_no_room_to_rest_skips():
    # When even a 1c rest cannot clear floor + cushion (hedge ask too high), skip.
    kalshi = FakeVenue("kalshi", [])
    poly = FakeVenue("poly", [])
    risk = RiskManager(RiskLimits(min_edge=0.01, max_position_per_market=1e9, max_total_exposure=1e12))
    ex = Executor({v.name: v for v in [kalshi, poly]}, risk,
                  fee_models={v.name: ZeroFeeModel() for v in [kalshi, poly]},
                  max_order_contracts=0, fill_confirmer=FakeConfirmer({}),
                  maker_timeout=0.01, maker_arm_cushion=0.05)
    report = asyncio.run(ex.execute_maker(opp(yv="poly", nv="kalshi", max_contracts=5,
                                              yes_price=0.94, no_price=0.05)))
    assert report.status is ExecStatus.SKIPPED and "no room" in report.reason
    assert kalshi.calls == []


def test_execute_maker_fat_edge_above_cushion_arms():
    # The cushion only blocks thin edges: an edge clearing floor+cushion still arms.
    kalshi = FakeVenue("kalshi", [res("kalshi", Side.NO, OrderStatus.RESTING, 0, None)])
    poly = FakeVenue("poly", [res("poly", Side.YES, OrderStatus.FILLED, 5, 0.40)])
    risk = RiskManager(RiskLimits(min_edge=0.01, max_position_per_market=1e9, max_total_exposure=1e12))
    ex = Executor({v.name: v for v in [kalshi, poly]}, risk,
                  fee_models={v.name: ZeroFeeModel() for v in [kalshi, poly]},
                  max_order_contracts=0, fill_confirmer=FakeConfirmer({"kalshi": (OrderStatus.FILLED, 5, 0.5)}),
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
    ex, risk = make_maker_exec([kalshi, poly], FakeConfirmer({"kalshi": (OrderStatus.FILLED, 5, 0.45)}))
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
        FakeConfirmer({"kalshi": (OrderStatus.FILLED, 5, 0.45), "poly": (OrderStatus.PARTIAL, 3, 0.40)}))
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
        FakeConfirmer({"kalshi": (OrderStatus.FILLED, 5, 0.45), "poly": (OrderStatus.PARTIAL, 3, 0.40)}))
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
    # The hedge band holds 3 of 5 -> PROPORTIONAL cover: arm 3//3 = 1 contract
    # (a maker fill arrives on a sweep; a band that barely covers pre-fill is gone
    # post-fill — 24h live: 62 killed hedges vs 18 clean).
    kalshi = FakeVenue("kalshi", [res("kalshi", Side.NO, OrderStatus.RESTING, 0, None)])
    poly = PreviewVenue("poly", [res("poly", Side.YES, OrderStatus.FILLED, 3, 0.40)], preview_filled=3)
    ex, risk = make_maker_exec([kalshi, poly], FakeConfirmer({"kalshi": (OrderStatus.FILLED, 1, 0.45)}))
    report = asyncio.run(ex.execute_maker(opp(yv="poly", nv="kalshi", max_contracts=5,
                                              yes_price=0.40, no_price=0.55)))
    assert report.status is ExecStatus.SUCCESS
    assert kalshi.calls[0][4] == 1                   # proportional: band 3 -> arm 3//3 = 1
    assert round(report.realized_pnl, 4) == round(1 * (1 - 0.40 - 0.55), 4)


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
    ex, risk = make_maker_exec([kalshi, poly], FakeConfirmer({"kalshi": (OrderStatus.FILLED, 5, 0.45)}))
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
    ex, risk = make_maker_exec([kalshi, poly], FakeConfirmer({"kalshi": (OrderStatus.FILLED, 5, 0.45)}))
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
    ex, risk = make_maker_exec([kalshi, poly], FakeConfirmer({"kalshi": (OrderStatus.FILLED, 5, 0.45)}))
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
    ex, risk = make_maker_exec([kalshi, poly], FakeConfirmer({"kalshi": (OrderStatus.FILLED, 5, 0.45)}))
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
    no = FakeVenue("poly", [res("poly", Side.NO, OrderStatus.REJECTED, 0, None),
                            res("poly", Side.NO, OrderStatus.KILLED, 0, None)])  # + recross
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
    no = FakeVenue("poly", [rej, res("poly", Side.NO, OrderStatus.KILLED, 0, None)])
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
    no = FakeVenue("poly", [res("poly", Side.NO, OrderStatus.KILLED, 0, None),
                            res("poly", Side.NO, OrderStatus.KILLED, 0, None)])  # + recross
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
    poly = FakeVenue("poly", [res("poly", Side.YES, OrderStatus.KILLED, 0, None),
                              res("poly", Side.YES, OrderStatus.KILLED, 0, None)])  # + recross
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
    # the hedge limit reaches the PnL-BREAKEVEN CEILING from leg1's actual fill
    # (1 - 0.55 - fees + recross_epsilon), superseding ask+buffer when higher; a
    # FOK fills at RESTING prices so the reach is free on an unmoved book
    assert poly.calls[0][3] == max(0.43, ex._breakeven_ceiling("kalshi", 0.55, "poly"))


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


def test_family_reliability_lends_starting_size_to_new_markets():
    # Markets are ephemeral: per-market learning never transfers, so 67% of live fires
    # were stuck at the 1-contract probe. A family with a healthy fill history lends its
    # NEW markets a real starting size (half its largest demonstrated fill).
    from bot.execution.executor import _family
    assert _family("kalshi", "KXVALORANTGAME-26JUL020400DKGENA-GENA") == ("kalshi", "KXVALORANTGAME")
    assert _family("polymarket_us", "aec-valorant-gena-dk-2026-07-02") == ("polymarket_us", "aec-valorant")

    ex, _ = make_exec([FakeVenue("kalshi", []), FakeVenue("poly", [])])
    ex.probe_contracts, ex.market_proven_fills, ex.market_max_fails = 1, 3, 2
    # no family history -> probe
    assert ex._reliability_cap("kalshi", "KXVALORANTGAME-NEW1-X") == 1.0
    # healthy family history (20 fills, 2 fails, max_fill 30) -> new market starts at 15
    ex._family_rel[("kalshi", "KXVALORANTGAME")] = [20, 2, 30.0]
    assert ex._reliability_cap("kalshi", "KXVALORANTGAME-NEW1-X") == 15.0
    # UNHEALTHY family (40% failure rate) -> back to probe
    ex._family_rel[("kalshi", "KXVALORANTGAME")] = [12, 8, 30.0]
    assert ex._reliability_cap("kalshi", "KXVALORANTGAME-NEW1-X") == 1.0
    # a market's OWN fail streak still excludes it (during the cooldown) regardless of family
    import time as _time
    ex._family_rel[("kalshi", "KXVALORANTGAME")] = [20, 2, 30.0]
    ex._market_rel[("kalshi", "KXVALORANTGAME-NEW1-X")] = (0, 2, 2, 0.0)
    ex._excluded_until[("kalshi", "KXVALORANTGAME-NEW1-X")] = _time.time() + 300
    assert ex._reliability_cap("kalshi", "KXVALORANTGAME-NEW1-X") == 0.0


def test_edge_weighted_capital_budget():
    # A thin 1c edge may take at most edge/full_budget of spendable cash; a 3c edge
    # takes it all. Bankroll stops FIFO-locking into small-pnl trades.
    ex, _ = make_exec([FakeVenue("kalshi", []), FakeVenue("poly", [])],
                      max_order_contracts=0)
    ex.edge_full_budget, ex.edge_budget_floor = 0.03, 0.25
    ex.balance_buffer = 1.0
    ex._balances = {"kalshi": 90.0, "poly": 300.0}
    # 1c edge, gross ~0.99: budget = 90 * (0.01/0.03) / 0.99 ≈ 30 contracts
    _, caps = ex._max_size(opp(max_contracts=1000, yes_price=0.44, no_price=0.55))
    assert abs(caps["edge_budget"] - 90 * (1/3) / 0.99) < 1.0
    # 3c edge: full budget ≈ 90/0.97
    _, caps = ex._max_size(opp(max_contracts=1000, yes_price=0.44, no_price=0.53))
    assert abs(caps["edge_budget"] - 90 / 0.97) < 1.0
    # sub-floor edge still gets the 25% floor, not zero
    _, caps = ex._max_size(opp(max_contracts=1000, yes_price=0.45, no_price=0.548))
    assert caps["edge_budget"] >= 90 * 0.25 / 0.998 - 1.0


def test_fresh_hedge_fast_path_skips_rest_reread():
    # With fresh WS quotes and a hedge leg showing >=2x the trade size, the executor must
    # NOT re-read the hedge book (the only network hop on the fire path); stale or thin
    # quotes keep the confirm.
    import time as _time
    from dataclasses import replace as _replace

    class CountingVenue(FakeVenue):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.book_reads = 0

        async def fetch_quote(self, market):
            self.book_reads += 1
            return await super().fetch_quote(market)

    def build(fresh, hedge_size):
        yes = CountingVenue("kalshi", [res("kalshi", Side.YES, OrderStatus.FILLED, 2, 0.40)])
        no = CountingVenue("poly", [res("poly", Side.NO, OrderStatus.FILLED, 2, 0.55)])
        ex, _ = make_exec([yes, no])
        ex.fresh_hedge_secs = 1.0
        o = opp()
        o = _replace(o, fresh_ts=_time.time() if fresh else 0.0,
                     yes_size=100.0, no_size=hedge_size)
        return ex, no, o

    ex, no, o = build(fresh=True, hedge_size=100.0)      # fresh + deep -> skip re-read
    assert asyncio.run(ex.execute(o)).status is ExecStatus.SUCCESS
    assert no.book_reads == 0

    ex, no, o = build(fresh=False, hedge_size=100.0)     # stale -> re-read
    assert asyncio.run(ex.execute(o)).status is ExecStatus.SUCCESS
    assert no.book_reads == 1

    ex, no, o = build(fresh=True, hedge_size=3.0)        # fresh but thin -> re-read
    asyncio.run(ex.execute(o))
    assert no.book_reads == 1


def test_size_aware_reliability_big_reject_caps_not_kills():
    # A FOK reject at 20 contracts is NOT phantom evidence — it means "no 20 of depth
    # right now". It must set a TEMPORARY ceiling (half the attempt), leave the phantom
    # streak untouched, and expire.
    import time as _time
    ex, _ = make_exec([FakeVenue("kalshi", []), FakeVenue("poly", [])])
    ex.probe_contracts = 1
    ex._market_rel[("poly", "P1")] = (5, 0, 0, 10.0)          # proven, ramp -> 30
    assert ex._reliability_cap("poly", "P1") == 30.0
    ex._record_market_reliability("poly", "P1", False, attempted=20.0)
    fills, fails, streak, _ = ex._market_rel[("poly", "P1")]
    assert (fails, streak) == (0, 0)                           # no phantom strike
    assert ex._reliability_cap("poly", "P1") == 10.0           # capped to attempt/2
    ex._size_ceiling[("poly", "P1")] = (10.0, _time.time() - 1)  # TTL expired
    assert ex._reliability_cap("poly", "P1") == 30.0           # back to ramped


def test_probe_reject_excludes_with_cooldown_then_reprobes():
    # Probe-size rejects ARE phantom evidence: max_fails of them exclude the market for a
    # cooldown; after it expires the market re-probes (never permanently dead), and a fill
    # clears everything.
    import time as _time
    ex, _ = make_exec([FakeVenue("kalshi", []), FakeVenue("poly", [])])
    ex.probe_contracts, ex.market_max_fails = 1, 2
    ex._record_market_reliability("poly", "P1", False, attempted=1.0)
    assert ex._reliability_cap("poly", "P1") == 1.0            # 1 strike -> still probes
    ex._record_market_reliability("poly", "P1", False, attempted=1.0)
    assert ex._reliability_cap("poly", "P1") == 0.0            # 2 strikes -> cooling down
    ex._excluded_until[("poly", "P1")] = _time.time() - 1      # cooldown over
    assert ex._reliability_cap("poly", "P1") == 1.0            # re-probes, not dead
    ex._record_market_reliability("poly", "P1", True, fill_size=1.0)
    fills, fails, streak, _ = ex._market_rel[("poly", "P1")]
    assert streak == 0 and ("poly", "P1") not in ex._excluded_until


def test_size_ladder_remembers_and_converges_to_phantom_strike():
    # Without memory, the family base RESETS the ladder after each ceiling TTL and a
    # persistent phantom loops 20 -> 10 -> (reset) -> 20 forever, never earning strikes
    # (observed live on gh-alka: book showed 170, killed even 5). The ladder must halve
    # from the REMEMBERED ceiling and convert to a phantom strike at probe size.
    ex, _ = make_exec([FakeVenue("kalshi", []), FakeVenue("poly", [])])
    ex.probe_contracts, ex.market_max_fails = 1, 2
    ex._family_rel[("polymarket_us", "aec-cs2")] = [50, 2, 40.0]   # healthy family, start 20

    m = "aec-cs2-gh-alka-2026-07-02"
    assert ex._reliability_cap("polymarket_us", m) == 20.0         # family start
    ex._record_market_reliability("polymarket_us", m, False, attempted=20.0)
    assert ex._reliability_cap("polymarket_us", m) == 10.0
    ex._record_market_reliability("polymarket_us", m, False, attempted=10.0)
    assert ex._reliability_cap("polymarket_us", m) == 5.0
    # simulate the capping TTL expiring — the family base would reset the cap, but the
    # ladder MEMORY must keep halving from the last ceiling on the next reject
    c, exp = ex._size_ceiling[("polymarket_us", m)]
    ex._size_ceiling[("polymarket_us", m)] = (c, exp - 3600 + 1800 + 10)  # expired, remembered
    ex._record_market_reliability("polymarket_us", m, False, attempted=20.0)
    assert ex._reliability_cap("polymarket_us", m) == 2.5          # min(20, 5)/2, not 10
    ex._record_market_reliability("polymarket_us", m, False, attempted=2.5)
    # 2.5/2 = 1.25 > probe -> one more rung
    ex._record_market_reliability("polymarket_us", m, False, attempted=1.25)
    # 1.25/2 <= probe -> converted to a PHANTOM STRIKE
    _, _, streak, _ = ex._market_rel[("polymarket_us", m)]
    assert streak == 1
    ex._record_market_reliability("polymarket_us", m, False, attempted=1.0)
    _, _, streak, _ = ex._market_rel[("polymarket_us", m)]
    assert streak == 2                                              # -> cooldown engaged
    assert ex._reliability_cap("polymarket_us", m) == 0.0


def test_recross_locks_at_breakeven_instead_of_unwinding():
    # leg2 FOK fails but the book still offers the hedge within breakeven+eps ->
    # re-take it: a ~$0 lock strictly dominates the unwind's guaranteed spread loss.
    yes = FakeVenue("kalshi", [res("kalshi", Side.YES, OrderStatus.FILLED, 2, 0.40)])
    no = FakeVenue("poly", [
        res("poly", Side.NO, OrderStatus.KILLED, 0, None),          # hedge FOK fails
        res("poly", Side.NO, OrderStatus.FILLED, 2, 0.59),          # recross fills at 0.59
    ])
    ex, risk = make_exec([yes, no])
    report = asyncio.run(ex.execute(opp()))
    assert report.status is ExecStatus.SUCCESS                       # locked, NOT unwound
    assert round(report.realized_pnl, 4) == round(2 * (1 - 0.40 - 0.59), 4)
    assert not risk.is_killed
    # the recross order was priced at the breakeven+eps ceiling (1-0.40+0.02)
    assert abs(no.calls[1][3] - 0.62) < 1e-9


def test_recross_skipped_when_ask_beyond_breakeven():
    # The hedge book has moved past breakeven+eps -> recross would lock a real loss,
    # so pay for the unwind instead (old behavior preserved).
    from bot.models import MarketQuote
    away = MarketQuote(venue="poly", market_id="P1", title="", no_ask=0.80, no_ask_size=100)
    yes = FakeVenue("kalshi", [
        res("kalshi", Side.YES, OrderStatus.FILLED, 2, 0.40),
        res("kalshi", Side.YES, OrderStatus.FILLED, 2, 0.38, action="sell"),
    ])
    no = QuotingVenue("poly", [res("poly", Side.NO, OrderStatus.KILLED, 0, None)], away)
    ex, risk = make_exec([yes, no])
    report = asyncio.run(ex.execute(opp()))
    assert report.status is ExecStatus.UNWOUND
    assert len(no.calls) == 1                        # no recross order was even attempted


def test_recross_ceiling_is_pnl_breakeven_not_price_breakeven():
    # With real fee models, the recross ceiling must net fees OUT so the worst salvage
    # is -epsilon/ct (a price-breakeven ceiling locked -(fees+eps): -$0.83 on a 20-lot).
    from bot.fees import KalshiFeeModel, PolymarketUSFeeModel
    yes = FakeVenue("poly", [res("poly", Side.YES, OrderStatus.FILLED, 20, 0.75)])
    no = FakeVenue("kalshi", [
        res("kalshi", Side.NO, OrderStatus.KILLED, 0, None),        # hedge fails
        res("kalshi", Side.NO, OrderStatus.FILLED, 20, 0.23),       # recross fills
    ])
    risk = RiskManager(RiskLimits(max_position_per_market=1e9, max_total_exposure=1e12))
    ex = Executor({"poly": yes, "kalshi": no}, risk,
                  fee_models={"kalshi": KalshiFeeModel(0.07),
                              "poly": PolymarketUSFeeModel(0.05)},
                  max_order_contracts=0, take_first_venue="poly", first_venue="kalshi")
    report = asyncio.run(ex.execute(opp(yv="poly", nv="kalshi", max_contracts=20,
                                        yes_price=0.75, no_price=0.22)))
    assert report.status is ExecStatus.SUCCESS
    # ceiling = 1 - 0.75 - fees(0.75)+fees(0.25) [~0.022] + 0.02 -> ~0.248, NOT 0.27
    sent = no.calls[1][3]
    assert sent < 0.26                                   # fee-aware, tighter than price+eps
    # and the locked pnl at the actual 0.23 fill is small-positive/near-zero, not -0.8ish
    assert report.realized_pnl > -0.45


# ---------------- capital recycler (auto-rebalance v2) ----------------

def _rec_exec(venues, **kw):
    ex, risk = make_exec(venues, max_order_contracts=0, **kw)
    ex.recycle_floor, ex.recycle_itm_bid, ex.recycle_max_cost = 40.0, 0.90, 0.03
    ex.recycle_max_contracts, ex.recycle_target = 50.0, 0.0
    ex.recycle_cooldown, ex.recycle_pair_cooldown = 0.0, 3600.0
    ex.recycle_max_settle_days = 0.0    # off by default; horizon test sets it
    ex.recycle_min_settle_hours = 0.0   # ditto (dedicated test covers it)
    return ex, risk


def _q(venue, mid, yes_ask=None, no_ask=None):
    from bot.models import MarketQuote
    return MarketQuote(venue=venue, market_id=mid, title="", yes_ask=yes_ask,
                       yes_ask_size=100, no_ask=no_ask, no_ask_size=100)


def test_recycle_trigger_requires_drain_and_funded_other():
    ex, _ = _rec_exec([FakeVenue("kalshi", []), FakeVenue("poly", [])])
    ex._balances = {"kalshi": 8.0, "poly": 380.0}
    assert ex.recycle_trigger() == ("kalshi", "poly")       # drained + 3x imbalance
    ex._balances = {"kalshi": 8.0, "poly": 20.0}            # imbalance < 3x
    assert ex.recycle_trigger() is None
    ex._balances = {"kalshi": 100.0, "poly": 380.0}         # nobody drained
    assert ex.recycle_trigger() is None
    ex.recycle_floor = 0.0                                  # disabled
    ex._balances = {"kalshi": 8.0, "poly": 380.0}
    assert ex.recycle_trigger() is None


def test_plan_recycle_selects_itm_on_drained_venue_only():
    ex, _ = _rec_exec([FakeVenue("kalshi", []), FakeVenue("poly", [])])
    ex._balances = {"kalshi": 8.0, "poly": 380.0}
    # hedged pair: kalshi long 20 YES (ITM: NO ask 0.05 -> YES bid 0.95),
    #              poly short 20 (long NO; OTM: yes_ask 0.96 -> NO bid 0.04)
    ex._positions = {("kalshi", "K1"): 20.0, ("poly", "p1"): -20.0}
    pm = {("kalshi", "K1"): ("poly", "p1"), ("poly", "p1"): ("kalshi", "K1")}
    quotes = {("kalshi", "K1"): _q("kalshi", "K1", no_ask=0.05),
              ("poly", "p1"): _q("poly", "p1", yes_ask=0.96)}
    acts = ex.plan_recycle("kalshi", pm, quotes)
    assert len(acts) == 1
    a = acts[0]
    assert a["itm"][:2] == ("kalshi", "K1") and a["itm"][3] == 0.95
    assert a["otm"][:2] == ("poly", "p1") and a["otm"][3] == 0.04
    assert a["qty"] == 20.0
    # give-up = 1 - 0.95 - 0.04 + 0 fees = 0.01 <= 0.03 cap
    assert abs(a["give_up_ct"] - 0.01) < 1e-6
    # the ITM leg on the FUNDED venue is never recycled
    assert ex.plan_recycle("poly", pm, quotes) == []


def test_plan_recycle_guards():
    ex, _ = _rec_exec([FakeVenue("kalshi", []), FakeVenue("poly", [])])
    ex._balances = {"kalshi": 8.0, "poly": 380.0}
    pm = {("kalshi", "K1"): ("poly", "p1"), ("poly", "p1"): ("kalshi", "K1")}
    # bid below ITM threshold -> skipped
    ex._positions = {("kalshi", "K1"): 20.0, ("poly", "p1"): -20.0}
    q_low = {("kalshi", "K1"): _q("kalshi", "K1", no_ask=0.15),
             ("poly", "p1"): _q("poly", "p1", yes_ask=0.90)}
    assert ex.plan_recycle("kalshi", pm, q_low) == []
    # give-up beyond cap (bid 0.90, otm bid 0.02 -> 0.08) -> skipped
    q_cost = {("kalshi", "K1"): _q("kalshi", "K1", no_ask=0.10),
              ("poly", "p1"): _q("poly", "p1", yes_ask=0.98)}
    assert ex.plan_recycle("kalshi", pm, q_cost) == []
    q_ok = {("kalshi", "K1"): _q("kalshi", "K1", no_ask=0.05),
            ("poly", "p1"): _q("poly", "p1", yes_ask=0.96)}
    # busy market -> skipped
    assert ex.plan_recycle("kalshi", pm, q_ok,
                           busy=frozenset({("kalshi", "K1")})) == []
    # counterpart flat and NOT confirmed settled -> skipped (possible naked)
    ex._positions = {("kalshi", "K1"): 20.0}
    assert ex.plan_recycle("kalshi", pm, q_ok) == []
    # ...but confirmed settled -> SOLO realization. NOTE a solo exit has no OTM bid to
    # recapture, so its give-up is (1 - bid) + fees: at bid 0.95 that's 5c > the 3c cap
    # (correctly rejected); it needs a deeper bid.
    assert ex.plan_recycle("kalshi", pm, q_ok,
                           settled_counterparts=frozenset({("poly", "p1")})) == []
    q_deep = {("kalshi", "K1"): _q("kalshi", "K1", no_ask=0.02),
              ("poly", "p1"): _q("poly", "p1", yes_ask=0.99)}
    acts = ex.plan_recycle("kalshi", pm, q_deep,
                           settled_counterparts=frozenset({("poly", "p1")}))
    assert len(acts) == 1 and acts[0]["solo"] and acts[0]["otm"] is None
    assert abs(acts[0]["give_up_ct"] - 0.02) < 1e-6
    # same-direction counterpart (not a hedge) -> skipped
    ex._positions = {("kalshi", "K1"): 20.0, ("poly", "p1"): 20.0}
    assert ex.plan_recycle("kalshi", pm, q_ok) == []
    # per-pass contract budget truncation
    ex._positions = {("kalshi", "K1"): 200.0, ("poly", "p1"): -200.0}
    acts = ex.plan_recycle("kalshi", pm, q_ok)
    assert acts and acts[0]["qty"] == 50.0                 # recycle_max_contracts


def test_recycle_sells_itm_first_then_otm_and_books_delta():
    kalshi = FakeVenue("kalshi", [
        res("kalshi", Side.YES, OrderStatus.FILLED, 20, 0.95, action="sell", requested=20),
    ])
    poly = FakeVenue("poly", [
        res("poly", Side.NO, OrderStatus.FILLED, 20, 0.04, action="sell", requested=20),
    ])
    store = Store(":memory:")
    ex, risk = _rec_exec([kalshi, poly], store=store)
    ex._balances = {"kalshi": 8.0, "poly": 380.0}
    ex._positions = {("kalshi", "K1"): 20.0, ("poly", "p1"): -20.0}
    a = {"event": "kalshi:K1|poly:p1",
         "itm": ("kalshi", "K1", Side.YES, 0.95),
         "otm": ("poly", "p1", Side.NO, 0.04), "qty": 20.0,
         "give_up_ct": 0.01, "solo": False}
    r = asyncio.run(ex._recycle_one(a))
    # ordering: ITM (kalshi) sell first, then OTM (poly)
    assert kalshi.calls[0][2] == "sell" and poly.calls[0][2] == "sell"
    # drained venue credited with the ITM proceeds
    assert abs(ex._balances["kalshi"] - (8.0 + 20 * 0.95)) < 1e-6
    # positions decremented to flat
    assert abs(ex._positions[("kalshi", "K1")]) < 1e-9
    assert abs(ex._positions[("poly", "p1")]) < 1e-9
    # delta = 20*0.95 + 20*0.04 - 20 = -0.20 (zero-fee models)
    assert abs(r["delta"] - (-0.20)) < 1e-6
    assert abs(risk.daily_pnl - (-0.20)) < 1e-6
    row = store.conn.execute(
        "SELECT amount, note FROM pnl ORDER BY id DESC LIMIT 1").fetchone()
    assert "early exit (capital recycle)" in row["note"]
    # rebuy cooldown armed
    assert ex._recycled_until[ex._rpair_key("kalshi", "K1", "poly", "p1")] > 0


def test_recycle_otm_unsold_registers_persistent_remnant():
    kalshi = FakeVenue("kalshi", [
        res("kalshi", Side.YES, OrderStatus.FILLED, 20, 0.95, action="sell", requested=20),
    ])
    poly = FakeVenue("poly", [
        res("poly", Side.NO, OrderStatus.KILLED, 0, None, action="sell", requested=20),
    ])
    store = Store(":memory:")
    ex, _ = _rec_exec([kalshi, poly], store=store)
    ex._balances = {"kalshi": 8.0, "poly": 380.0}
    ex._positions = {("kalshi", "K1"): 20.0, ("poly", "p1"): -20.0}
    a = {"event": "kalshi:K1|poly:p1",
         "itm": ("kalshi", "K1", Side.YES, 0.95),
         "otm": ("poly", "p1", Side.NO, 0.04), "qty": 20.0,
         "give_up_ct": 0.01, "solo": False}
    r = asyncio.run(ex._recycle_one(a))
    assert r["remnant"] == 20.0
    assert ex.recycled_remnants[("poly", "p1")] == 20.0
    assert store.recycle_remnants()[("poly", "p1")] == 20.0     # persisted
    # conservative delta: treats the held OTM as $0 -> 20*(0.95-1) = -1.00
    assert abs(r["delta"] - (-1.00)) < 1e-6
    # a fresh executor loads the remnant back (restart safety)
    ex2, _ = _rec_exec([FakeVenue("kalshi", []), FakeVenue("poly", [])], store=store)
    assert ex2.recycled_remnants[("poly", "p1")] == 20.0


def test_recycle_itm_killed_or_error_books_nothing():
    for status in (OrderStatus.KILLED, OrderStatus.ERROR):
        kalshi = FakeVenue("kalshi", [
            res("kalshi", Side.YES, status, 0, None, action="sell", requested=20)])
        poly = FakeVenue("poly", [])
        store = Store(":memory:")
        ex, risk = _rec_exec([kalshi, poly], store=store)
        ex._positions = {("kalshi", "K1"): 20.0, ("poly", "p1"): -20.0}
        a = {"event": "e", "itm": ("kalshi", "K1", Side.YES, 0.95),
             "otm": ("poly", "p1", Side.NO, 0.04), "qty": 20.0,
             "give_up_ct": 0.01, "solo": False}
        r = asyncio.run(ex._recycle_one(a))
        assert r is None and poly.calls == []               # no OTM attempt
        assert risk.daily_pnl == 0.0 and not risk.is_killed
        assert ex._positions[("kalshi", "K1")] == 20.0      # untouched


def test_execute_blocks_recycled_pair_rebuy():
    import time as _time
    yes = FakeVenue("kalshi", [res("kalshi", Side.YES, OrderStatus.FILLED, 2, 0.40)])
    no = FakeVenue("poly", [res("poly", Side.NO, OrderStatus.FILLED, 2, 0.55)])
    ex, _ = make_exec([yes, no])
    o = opp()
    ex._recycled_until[ex._rpair_key(o.buy_yes_venue, o.buy_yes_market,
                                     o.buy_no_venue, o.buy_no_market)] = _time.time() + 60
    rep = asyncio.run(ex.execute(o))
    assert rep.status is ExecStatus.SKIPPED and "recycled" in rep.reason
    # expired cooldown -> trades again
    ex._recycled_until.clear()
    assert asyncio.run(ex.execute(o)).status is ExecStatus.SUCCESS


def test_effective_balance_gates_only():
    # Pending payouts flip the STEERING gates but never the cash sizing caps.
    ex, _ = make_exec([FakeVenue("kalshi", []), FakeVenue("poly", [])],
                      max_order_contracts=0)
    ex.rebalance_floor = 40.0
    ex._balances = {"kalshi": 10.0, "poly": 100.0}
    o = opp(yv="kalshi", nv="poly", yes_price=0.17, no_price=0.80)  # kalshi = longshot
    assert ex._rebalance_skip(o) is not None                # drained -> reserved
    ex.set_pending({"kalshi": 75.0})                        # $75 landing in hours
    assert ex._rebalance_skip(o) is None                    # effectively funded
    # sizing caps unchanged: cash_yes still uses REAL $10
    _, caps = ex._max_size(o)
    assert caps["cash_yes"] <= 10.0 / 0.17 + 1e-6


# ---------------- early-profit exit (generalized recycler) ----------------

def _ee_exec(venues, store, **kw):
    ex, risk = make_exec(venues, max_order_contracts=0, store=store, **kw)
    ex.early_exit_enabled = True
    ex.early_exit_margin = 0.0
    ex.early_exit_cooldown = 0.0
    ex.early_exit_max_pairs = 8.0
    ex.early_exit_max_contracts = 100.0
    ex.early_exit_min_bid_depth = 0.0
    ex.early_exit_min_settle_days = 0.0     # undated test quotes -> disable horizon gate here
    return ex, risk


def test_entry_cost_for_pair_order_independent():
    store = Store(":memory:")
    from types import SimpleNamespace
    store.record_opportunity(SimpleNamespace(
        event_key="e", buy_yes_venue="kalshi", buy_yes_market="K1",
        buy_no_venue="poly", buy_no_market="p1", yes_price=0.40, no_price=0.55,
        edge_per_contract=0.05, max_contracts=10, total_profit=0.5), acted=True)
    assert store.entry_cost_for_pair("kalshi", "K1", "poly", "p1") == (0.40, 0.55)
    assert store.entry_cost_for_pair("poly", "p1", "kalshi", "K1") == (0.40, 0.55)  # order-independent
    assert store.entry_cost_for_pair("kalshi", "K1", "poly", "nope") is None


def test_plan_early_exit_fires_only_above_entry_plus_margin():
    store = Store(":memory:")
    ex, _ = _ee_exec([FakeVenue("kalshi", []), FakeVenue("poly", [])], store)
    # Held hedged pair: kalshi long 20 YES, poly long 20 NO. Entry cost 0.40+0.55=0.95.
    ex._positions = {("kalshi", "K1"): 20.0, ("poly", "p1"): -20.0}
    pm = {("kalshi", "K1"): ("poly", "p1"), ("poly", "p1"): ("kalshi", "K1")}
    pk = ex._rpair_key("kalshi", "K1", "poly", "p1")
    entry = {pk: (0.40, 0.55)}
    # Books dislocated favorably: YES@kalshi bid 0.60 (no_ask 0.40), NO@poly bid 0.50 (yes_ask 0.50)
    # exit_value = 0.60 + 0.50 - fees(0) = 1.10 > entry 0.95 -> fire
    good = {("kalshi", "K1"): _q("kalshi", "K1", no_ask=0.40),
            ("poly", "p1"): _q("poly", "p1", yes_ask=0.50)}
    acts = ex.plan_early_exit(pm, good, entry)
    assert len(acts) == 1 and acts[0]["qty"] == 20.0 and acts[0]["gain"] > 0.14
    # Quiet market BELOW entry: asks 0.60+0.50 -> exit 0.40+0.50=0.90 < entry 0.95 -> hold
    flat = {("kalshi", "K1"): _q("kalshi", "K1", no_ask=0.60),
            ("poly", "p1"): _q("poly", "p1", yes_ask=0.50)}
    assert ex.plan_early_exit(pm, flat, entry) == []
    # margin requirement: exit 0.98 (gain 0.03) meets a 0.03 margin exactly -> fires;
    # raise the margin to 0.05 and the same book no longer qualifies -> hold
    fair = {("kalshi", "K1"): _q("kalshi", "K1", no_ask=0.44),
            ("poly", "p1"): _q("poly", "p1", yes_ask=0.58)}   # exit 0.56+0.42=0.98, gain 0.03
    ex.early_exit_margin = 0.03
    assert len(ex.plan_early_exit(pm, fair, entry)) == 1
    ex.early_exit_margin = 0.05
    assert ex.plan_early_exit(pm, fair, entry) == []


def test_plan_early_exit_no_entry_cost_skips():
    store = Store(":memory:")
    ex, _ = _ee_exec([FakeVenue("kalshi", []), FakeVenue("poly", [])], store)
    ex._positions = {("kalshi", "K1"): 20.0, ("poly", "p1"): -20.0}
    pm = {("kalshi", "K1"): ("poly", "p1"), ("poly", "p1"): ("kalshi", "K1")}
    good = {("kalshi", "K1"): _q("kalshi", "K1", no_ask=0.20),
            ("poly", "p1"): _q("poly", "p1", yes_ask=0.20)}
    assert ex.plan_early_exit(pm, good, {}) == []            # no cost basis -> can't judge


def test_plan_early_exit_respects_busy_and_min_depth():
    store = Store(":memory:")
    ex, _ = _ee_exec([FakeVenue("kalshi", []), FakeVenue("poly", [])], store)
    ex._positions = {("kalshi", "K1"): 20.0, ("poly", "p1"): -20.0}
    pm = {("kalshi", "K1"): ("poly", "p1"), ("poly", "p1"): ("kalshi", "K1")}
    entry = {ex._rpair_key("kalshi", "K1", "poly", "p1"): (0.40, 0.55)}
    good = {("kalshi", "K1"): _q("kalshi", "K1", no_ask=0.30),
            ("poly", "p1"): _q("poly", "p1", yes_ask=0.30)}
    assert ex.plan_early_exit(pm, good, entry, busy=frozenset({("kalshi", "K1")})) == []
    # min depth gate: quote sizes are 100 (from _q); require 200 -> skipped
    ex.early_exit_min_bid_depth = 200.0
    assert ex.plan_early_exit(pm, good, entry) == []


def test_early_exit_books_realized_profit_via_recycle_one():
    # exit_value - entry = realized profit; _recycle_one books delta-vs-$1 which, added
    # to the entry-time lock, equals that realized profit.
    kalshi = FakeVenue("kalshi", [
        res("kalshi", Side.YES, OrderStatus.FILLED, 20, 0.60, action="sell", requested=20)])
    poly = FakeVenue("poly", [
        res("poly", Side.NO, OrderStatus.FILLED, 20, 0.50, action="sell", requested=20)])
    store = Store(":memory:")
    ex, risk = _ee_exec([kalshi, poly], store)
    ex._positions = {("kalshi", "K1"): 20.0, ("poly", "p1"): -20.0}
    a = {"event": "kalshi:K1|poly:p1 (early-exit)",
         "itm": ("kalshi", "K1", Side.YES, 0.60),
         "otm": ("poly", "p1", Side.NO, 0.50), "qty": 20.0, "gain": 0.15, "solo": False}
    r = asyncio.run(ex._recycle_one(a))
    # delta-vs-$1 = 20*0.60 + 20*0.50 - 20 = +2.00 (zero-fee). Entry lock was 20*(1-0.95)=+1.00.
    # total realized = +3.00 = 20*(exit 1.10 - entry 0.95). delta booked = +2.00.
    assert abs(r["delta"] - 2.00) < 1e-6
    assert abs(risk.daily_pnl - 2.00) < 1e-6
    row = store.conn.execute("SELECT note FROM pnl ORDER BY id DESC LIMIT 1").fetchone()
    assert "early exit" in row["note"] or "capital recycle" in row["note"]


def test_horizon_gate_blocks_thin_longdated_entry():
    import time as _time
    yes = FakeVenue("kalshi", [res("kalshi", Side.YES, OrderStatus.FILLED, 2, 0.40)])
    no = FakeVenue("poly", [res("poly", Side.NO, OrderStatus.FILLED, 2, 0.55)])
    ex, _ = make_exec([yes, no])
    ex.max_settle_days = 30.0
    ex.longdated_min_edge = 0.05
    o = opp()
    o.edge_per_contract = 0.02                    # thin
    o.settle_ts = _time.time() + 90 * 86400       # 90 days out
    assert asyncio.run(ex.execute(o)).status is ExecStatus.SKIPPED
    # capital-yield policy (2026-07-09): even a fat 8c edge is 0.09%/day over 90
    # days -> still skipped; the same edge settling within 8 days passes
    o.edge_per_contract = 0.08
    assert asyncio.run(ex.execute(o)).status is ExecStatus.SKIPPED
    o.settle_ts = _time.time() + 7 * 86400
    assert asyncio.run(ex.execute(o)).status is ExecStatus.SUCCESS
    # near-dated thin edge -> allowed (fresh venues so the scripted fills aren't drained)
    yes2 = FakeVenue("kalshi", [res("kalshi", Side.YES, OrderStatus.FILLED, 2, 0.40)])
    no2 = FakeVenue("poly", [res("poly", Side.NO, OrderStatus.FILLED, 2, 0.55)])
    ex2, _ = make_exec([yes2, no2])
    ex2.max_settle_days = 30.0; ex2.longdated_min_edge = 0.05
    # yield policy: 2c pays for two days, not five
    o2 = opp(); o2.edge_per_contract = 0.02; o2.settle_ts = _time.time() + 1.5 * 86400
    assert asyncio.run(ex2.execute(o2)).status is ExecStatus.SUCCESS


def test_early_exit_skips_near_dated_pairs():
    import time as _time
    store = Store(":memory:")
    ex, _ = _ee_exec([FakeVenue("kalshi", []), FakeVenue("poly", [])], store)
    ex.early_exit_min_settle_days = 3.0        # only unwind pairs locked >= 3 days
    ex._positions = {("kalshi", "K1"): 20.0, ("poly", "p1"): -20.0}
    pm = {("kalshi", "K1"): ("poly", "p1"), ("poly", "p1"): ("kalshi", "K1")}
    entry = {ex._rpair_key("kalshi", "K1", "poly", "p1"): (0.40, 0.55)}
    # profitable exit (0.60+0.50=1.10 > entry 0.95) but settles TOMORROW -> skip
    def q(mid, **kw):
        z = _q("kalshi" if mid == "K1" else "poly", mid, **kw)
        return z
    near_k = _q("kalshi", "K1", no_ask=0.40); near_k.close_time = _time.time() + 1 * 86400
    near_p = _q("poly", "p1", yes_ask=0.50);   near_p.close_time = _time.time() + 1 * 86400
    assert ex.plan_early_exit(pm, {("kalshi","K1"): near_k, ("poly","p1"): near_p}, entry) == []
    # same book but settles in 10 days -> eligible
    far_k = _q("kalshi", "K1", no_ask=0.40); far_k.close_time = _time.time() + 10 * 86400
    far_p = _q("poly", "p1", yes_ask=0.50);   far_p.close_time = _time.time() + 10 * 86400
    assert len(ex.plan_early_exit(pm, {("kalshi","K1"): far_k, ("poly","p1"): far_p}, entry)) == 1
    # unknown close_time -> conservative skip (can't confirm the fee is worth it)
    und_k = _q("kalshi", "K1", no_ask=0.40); und_p = _q("poly", "p1", yes_ask=0.50)
    assert ex.plan_early_exit(pm, {("kalshi","K1"): und_k, ("poly","p1"): und_p}, entry) == []


def test_recycle_skips_fardated_undecided_favorite():
    import time as _time
    store = Store(":memory:")
    ex, _ = _rec_exec([FakeVenue("kalshi", []), FakeVenue("poly", [])], store=store)
    ex.recycle_max_settle_days = 3.0
    ex.recycle_decided_bid = 0.98
    ex._balances = {"kalshi": 8.0, "poly": 380.0}
    ex._positions = {("kalshi", "K1"): 20.0, ("poly", "p1"): -20.0}
    pm = {("kalshi", "K1"): ("poly", "p1"), ("poly", "p1"): ("kalshi", "K1")}
    # pre-decision favorite (0.92 ITM) settling in 30 days (a months-out election) -> skip
    far = _q("kalshi", "K1", no_ask=0.08); far.close_time = _time.time() + 30 * 86400
    farc = _q("poly", "p1", yes_ask=0.94); farc.close_time = _time.time() + 30 * 86400
    assert ex.plan_recycle("kalshi", pm, {("kalshi","K1"): far, ("poly","p1"): farc}) == []
    # same far date but near-certain (0.99) -> decided, allowed
    cert = _q("kalshi", "K1", no_ask=0.01); cert.close_time = _time.time() + 30 * 86400
    certc = _q("poly", "p1", yes_ask=0.99); certc.close_time = _time.time() + 30 * 86400
    assert len(ex.plan_recycle("kalshi", pm, {("kalshi","K1"): cert, ("poly","p1"): certc})) == 1
    # near-dated favorite (settles today) at 0.92 -> decided/imminent, allowed
    near = _q("kalshi", "K1", no_ask=0.08); near.close_time = _time.time() + 3600
    nearc = _q("poly", "p1", yes_ask=0.94); nearc.close_time = _time.time() + 3600
    assert len(ex.plan_recycle("kalshi", pm, {("kalshi","K1"): near, ("poly","p1"): nearc})) == 1


def test_recycle_horizon_uses_id_date_when_close_time_missing():
    import time as _time
    store = Store(":memory:")
    ex, _ = _rec_exec([FakeVenue("kalshi", []), FakeVenue("poly", [])], store=store)
    ex.recycle_max_settle_days = 3.0
    ex.recycle_decided_bid = 0.98
    ex._balances = {"kalshi": 380.0, "poly": 8.0}
    # UFC-class pair: quote has NO close_time, but the kalshi id is dated 6 days out
    from datetime import datetime, timezone, timedelta
    d = (datetime.now(timezone.utc) + timedelta(days=6)).strftime("%y%b%d").upper()
    km = f"KXUFCFIGHT-{d}STEELL-STE"
    ex._positions = {("poly", "p1"): 5.0, ("kalshi", km): -5.0}
    pm = {("poly", "p1"): ("kalshi", km), ("kalshi", km): ("poly", "p1")}
    q_itm = _q("poly", "p1", no_ask=0.09)          # poly ITM bid 0.91 < decided 0.98
    q_otm = _q("kalshi", km, yes_ask=0.93)
    # drained venue = poly; ITM leg on poly at 0.91, undated quote, far-dated id -> SKIP
    acts = ex.plan_recycle("poly", pm, {("poly","p1"): q_itm, ("kalshi",km): q_otm})
    assert acts == []


def test_recycle_skips_pairs_settling_within_min_hours():
    import time as _time
    from datetime import datetime, timezone, timedelta
    store = Store(":memory:")
    ex, _ = _rec_exec([FakeVenue("kalshi", []), FakeVenue("poly", [])], store=store)
    ex.recycle_min_settle_hours = 24.0
    ex._balances = {"kalshi": 8.0, "poly": 380.0}
    ex._positions = {("kalshi", "K1"): 20.0, ("poly", "p1"): -20.0}
    pm = {("kalshi", "K1"): ("poly", "p1"), ("poly", "p1"): ("kalshi", "K1")}
    # DECIDED leg (0.99) settling in 3 hours -> pays $1 in hours, never recycle
    near = _q("kalshi", "K1", no_ask=0.01); near.close_time = _time.time() + 3 * 3600
    nearc = _q("poly", "p1", yes_ask=0.99); nearc.close_time = _time.time() + 3 * 3600
    assert ex.plan_recycle("kalshi", pm, {("kalshi","K1"): near, ("poly","p1"): nearc}) == []
    # same book settling in 2 days -> eligible
    far = _q("kalshi", "K1", no_ask=0.01); far.close_time = _time.time() + 2 * 86400
    farc = _q("poly", "p1", yes_ask=0.99); farc.close_time = _time.time() + 2 * 86400
    assert len(ex.plan_recycle("kalshi", pm, {("kalshi","K1"): far, ("poly","p1"): farc})) == 1
    # undated quote BUT the id carries a same-day date -> fail-closed skip
    d = datetime.now(timezone.utc).strftime("%y%b%d").upper()
    km = f"KXCS2GAME-{d}AB-CD"
    ex._positions = {("kalshi", km): 20.0, ("poly", "p2"): -20.0}
    pm2 = {("kalshi", km): ("poly", "p2"), ("poly", "p2"): ("kalshi", km)}
    q1 = _q("kalshi", km, no_ask=0.01); q2 = _q("poly", "p2", yes_ask=0.99)
    assert ex.plan_recycle("kalshi", pm2, {("kalshi",km): q1, ("poly","p2"): q2}) == []
    # no date derivable anywhere -> can't prove it isn't settling soon -> skip
    ex._positions = {("kalshi", "KNODATE-X"): 20.0, ("poly", "p3"): -20.0}
    pm3 = {("kalshi", "KNODATE-X"): ("poly", "p3"), ("poly", "p3"): ("kalshi", "KNODATE-X")}
    q3 = _q("kalshi", "KNODATE-X", no_ask=0.01); q4 = _q("poly", "p3", yes_ask=0.99)
    assert ex.plan_recycle("kalshi", pm3, {("kalshi","KNODATE-X"): q3, ("poly","p3"): q4}) == []


def test_confirmer_avg_converts_no_side():
    from bot.execution.executor import Executor
    from bot.models import Side
    # both venues' private fill events quote YES-side prices: a NO order's cost is
    # the complement (the 2026-07-07 audit found NO makers booked at 1-actual)
    assert Executor.confirmer_avg(Side.NO, 0.28) == 0.72
    assert Executor.confirmer_avg(Side.YES, 0.28) == 0.28
    assert Executor.confirmer_avg(Side.NO, None) is None


def test_maker_thin_top_deep_ladder_passes_hedge_gate():
    # 1 contract at top with 300 behind it at +1 tick, all inside the profit ceiling:
    # the old top-size gate skipped this as "thin hedge book"; band depth hedges it.
    from dataclasses import replace as _rp
    kalshi = FakeVenue("kalshi", [res("kalshi", Side.NO, OrderStatus.RESTING, 0, None)])
    poly = FakeVenue("poly", [res("poly", Side.YES, OrderStatus.FILLED, 5, 0.40)])
    ex, risk = make_maker_exec([kalshi, poly],
                               FakeConfirmer({"kalshi": (OrderStatus.FILLED, 5, 0.45)}))
    ex.min_leg_depth = 10                            # top size 1 would fail this gate
    o = opp(yv="poly", nv="kalshi", max_contracts=5, yes_price=0.40, no_price=0.55)
    o = _rp(o, yes_size=1.0, no_size=50.0,
            yes_levels=((0.40, 1), (0.41, 300)), no_levels=((0.55, 50),))
    report = asyncio.run(ex.execute_maker(o))
    assert report.status is ExecStatus.SUCCESS       # band 301 clears min 10

    # same book WITHOUT the ladder: the top-size fallback must still skip it
    kalshi2 = FakeVenue("kalshi", [res("kalshi", Side.NO, OrderStatus.RESTING, 0, None)])
    poly2 = FakeVenue("poly", [res("poly", Side.YES, OrderStatus.FILLED, 5, 0.40)])
    ex2, _ = make_maker_exec([kalshi2, poly2],
                             FakeConfirmer({"kalshi": (OrderStatus.FILLED, 5, 0.45)}))
    ex2.min_leg_depth = 10
    o2 = _rp(opp(yv="poly", nv="kalshi", max_contracts=5, yes_price=0.40, no_price=0.55),
             yes_size=1.0, no_size=50.0)
    report2 = asyncio.run(ex2.execute_maker(o2))
    assert report2.status is ExecStatus.SKIPPED and "thin hedge" in report2.reason


def test_suspect_leg_fires_first_and_reject_is_free():
    # kalshi leg unproven (no reliability history), poly leg proven -> kalshi fires
    # FIRST; its REJECT is a clean SKIP with the poly leg never touched (no unwind).
    kalshi = FakeVenue("kalshi", [res("kalshi", Side.NO, OrderStatus.REJECTED, 0, None)])
    poly = FakeVenue("poly", [])                       # must never be called
    ex, risk = make_exec([kalshi, poly], max_order_contracts=5)
    ex._market_rel[("poly", "P1")] = (10, 0, 0, 20.0)  # proven
    # kalshi K1 absent from _market_rel -> unproven -> suspect
    report = asyncio.run(ex.execute(opp(yv="poly", nv="kalshi", max_contracts=5,
                                        yes_price=0.40, no_price=0.55)))
    assert report.status is ExecStatus.SKIPPED
    assert poly.calls == []                            # nothing to unwind
    assert kalshi.calls and kalshi.calls[0][2] == "buy"


def test_rejected_probe_escalates_exclusion_immediately():
    from bot.execution.executor import Executor
    kalshi = FakeVenue("kalshi", [])
    poly = FakeVenue("poly", [])
    ex, _ = make_exec([kalshi, poly])
    ex.market_max_fails = 2
    ex._record_market_reliability("kalshi", "K1", ok=False, attempted=1.0, rejected=True)
    # streak jumped by 2 -> crossed max_fails on the FIRST refusal -> excluded now
    assert ("kalshi", "K1") in ex._excluded_until


def test_leg2_ceiling_fills_moved_book_instead_of_unwinding():
    # leg1 fills at 0.40; the hedge book ticked 0.55 -> 0.57 (inside the breakeven
    # ceiling). Old behavior: FOK at 0.55 KILLs -> recross round-trip or a -2-3c
    # unwind. New behavior: the FOK limit already reaches the ceiling, the venue
    # fills at the RESTING 0.57, and the pair settles SUCCESS at reduced pnl.
    poly = FakeVenue("poly", [res("poly", Side.YES, OrderStatus.FILLED, 2, 0.40)])
    kalshi = FakeVenue("kalshi", [res("kalshi", Side.NO, OrderStatus.FILLED, 2, 0.57)])
    ex = Executor({"kalshi": kalshi, "poly": poly},
                  RiskManager(RiskLimits(min_edge=0.01, max_position_per_market=1e9,
                                         max_total_exposure=1e12)),
                  fee_models={"kalshi": ZeroFeeModel(), "poly": ZeroFeeModel()},
                  max_order_contracts=0, hedge_buffer=0.0,
                  take_first_venue="poly")
    ex._market_rel[("poly", "P1")] = (5, 0, 0, 10.0)
    ex._market_rel[("kalshi", "K1")] = (5, 0, 0, 10.0)
    report = asyncio.run(ex.execute(opp(yv="poly", nv="kalshi",
                                        yes_price=0.40, no_price=0.55)))
    assert report.status is ExecStatus.SUCCESS
    # kalshi FOK was placed at the ceiling (>= the moved 0.57), not the stale 0.55
    assert kalshi.calls[0][3] >= 0.57
    # pnl reduced but the pair locked: 10 ct * (1 - 0.40 - 0.57) = 0.30
    assert abs(report.realized_pnl - 0.30) < 1e-9


def test_taker_leg2_partial_tops_up_and_locks():
    # The 3.85/20 incident: poly FOK partial-fills leg2. Instead of HALTing, the
    # executor tops up the remainder at the ceiling; the book refilled -> locked.
    poly = FakeVenue("poly", [res("poly", Side.YES, OrderStatus.FILLED, 20, 0.46)])
    kalshi = FakeVenue("kalshi", [
        res("kalshi", Side.NO, OrderStatus.PARTIAL, 4, 0.52),   # leg2 partial
        res("kalshi", Side.NO, OrderStatus.FILLED, 16, 0.47),   # top-up fills
    ])
    ex = Executor({"kalshi": kalshi, "poly": poly},
                  RiskManager(RiskLimits(min_edge=0.001, max_position_per_market=1e9,
                                         max_total_exposure=1e12)),
                  fee_models={"kalshi": ZeroFeeModel(), "poly": ZeroFeeModel()},
                  max_order_contracts=0, take_first_venue="poly")
    ex._market_rel[("poly", "P1")] = (5, 0, 0, 30.0)
    ex._market_rel[("kalshi", "K1")] = (5, 0, 0, 30.0)
    report = asyncio.run(ex.execute(opp(yv="poly", nv="kalshi", max_contracts=20,
                                        yes_price=0.46, no_price=0.49)))
    assert report.status is ExecStatus.SUCCESS         # locked, not halted
    assert len(kalshi.calls) == 2                       # partial + top-up
    assert not ex.risk.is_killed


def test_taker_leg2_partial_topup_fails_settles_matched_unwinds_rest():
    # top-up also fails -> settle the hedged 4, unwind the excess 16 — flat, no halt
    poly = FakeVenue("poly", [
        res("poly", Side.YES, OrderStatus.FILLED, 20, 0.46),    # leg1
        res("poly", Side.YES, OrderStatus.FILLED, 16, 0.45,     # unwind sell of excess
            action="sell", requested=16),
    ])
    kalshi = FakeVenue("kalshi", [
        res("kalshi", Side.NO, OrderStatus.PARTIAL, 4, 0.52),
        res("kalshi", Side.NO, OrderStatus.KILLED, 0, None),    # top-up killed
    ])
    ex = Executor({"kalshi": kalshi, "poly": poly},
                  RiskManager(RiskLimits(min_edge=0.001, max_position_per_market=1e9,
                                         max_total_exposure=1e12)),
                  fee_models={"kalshi": ZeroFeeModel(), "poly": ZeroFeeModel()},
                  max_order_contracts=0, take_first_venue="poly")
    ex._market_rel[("poly", "P1")] = (5, 0, 0, 30.0)
    ex._market_rel[("kalshi", "K1")] = (5, 0, 0, 30.0)
    report = asyncio.run(ex.execute(opp(yv="poly", nv="kalshi", max_contracts=20,
                                        yes_price=0.46, no_price=0.49)))
    assert report.status is ExecStatus.UNWOUND
    assert not ex.risk.is_killed                        # flat, no manual reconcile


def test_capital_yield_entry_gate():
    # policy 2026-07-09: edge must pay >= 1%/day of holding — 2c buys two days
    import time as _t
    from dataclasses import replace
    kalshi = FakeVenue("kalshi", []); poly = FakeVenue("poly", [])
    ex, _ = make_exec([kalshi, poly])
    ex.min_daily_yield = 0.01
    o = opp(yv="poly", nv="kalshi", yes_price=0.40, no_price=0.58)   # 2c edge
    o = replace(o, settle_ts=_t.time() + 5 * 86400)                  # 5 days out
    skip = ex._horizon_skip(o)
    assert skip and "capital yield" in skip
    o2 = replace(o, settle_ts=_t.time() + 1.5 * 86400)               # 1.5 days out
    assert ex._horizon_skip(o2) is None


def test_capital_yield_early_exit_accepts_bounded_haircut():
    # a pair locked at 0.98 settling in 100 days earns ~0.02%/day — the exit is
    # acceptable down to entry x (1 - 1%), not only at a profit
    from types import SimpleNamespace
    import time as _t
    kalshi = FakeVenue("kalshi", []); poly = FakeVenue("poly", [])
    ex, _ = make_exec([kalshi, poly])
    ex.min_daily_yield = 0.01
    ex.early_exit_min_bid_depth = 0.0
    ex.early_exit_margin = 0.005
    ex._positions = {("kalshi", "K1"): 10.0, ("poly", "P1"): -10.0}
    pair_map = {("kalshi", "K1"): ("poly", "P1"), ("poly", "P1"): ("kalshi", "K1")}
    far = _t.time() + 100 * 86400
    quotes = {("kalshi", "K1"): SimpleNamespace(no_ask=0.40, no_ask_size=50,
                                                yes_ask=None, yes_ask_size=0, close_time=far),
              ("poly", "P1"): SimpleNamespace(yes_ask=0.61, yes_ask_size=50,
                                              no_ask=None, no_ask_size=0, close_time=far)}
    # exit value = (1-0.40)+(1-0.61) = 0.99 -> above 0.98*(1-0.01)=0.9702, below
    # entry+margin -> old rule skipped it, yield rule takes it
    entry = {ex._rpair_key("kalshi", "K1", "poly", "P1"): (0.58, 0.40)}
    acts = ex.plan_early_exit(pair_map, quotes, entry)
    assert acts, "under-yielding lock must exit at a bounded haircut"


def test_leg2_error_reconciliation_uses_delta_not_absolute():
    # FRA-MAR Hakimi bug: attempt #2's errored leg2 saw attempt #1's 20 contracts
    # and declared "hedge landed" — 20 contracts went naked. The venue position is
    # PRE + THIS; only the delta above our tracked baseline is this order's fill.
    # NOTE opp() maps: buy_yes_market="K1" on yv, buy_no_market="P1" on nv —
    # with yv=poly/nv=kalshi the KALSHI market id is "P1".
    kalshi = FakeVenue("kalshi", [
        res("kalshi", Side.NO, OrderStatus.ERROR, 0, None),           # leg2 errors
        res("kalshi", Side.NO, OrderStatus.KILLED, 0, None),          # recross killed
    ])
    from types import SimpleNamespace
    async def snap():
        return SimpleNamespace(venue="kalshi", balance=1000.0, positions=[
            SimpleNamespace(market_id="P1", quantity=-20, is_open=True)])
    kalshi.account_snapshot = snap
    poly = FakeVenue("poly", [
        res("poly", Side.YES, OrderStatus.FILLED, 20, 0.40),          # leg1 fills
        res("poly", Side.YES, OrderStatus.FILLED, 20, 0.39,           # unwind of leg1
            action="sell", requested=20),
    ])
    ex, risk = make_exec([kalshi, poly], max_order_contracts=0)
    ex.take_first_venue = "poly"
    ex._market_rel[("poly", "K1")] = (5, 0, 0, 30.0)
    ex._market_rel[("kalshi", "P1")] = (5, 0, 0, 30.0)
    # baseline: we ALREADY hold 20 kalshi NO from attempt #1
    ex._positions[("kalshi", "P1")] = -20.0
    report = asyncio.run(ex.execute(opp(yv="poly", nv="kalshi", max_contracts=20,
                                        yes_price=0.40, no_price=0.55)))
    # delta = 20 - 20 = 0 -> hedge did NOT land -> must NOT settle as locked
    assert report.status is not ExecStatus.SUCCESS


def test_maker_slip_ewma_raises_arm_bar():
    # a hostile family (in-play ITF) accumulates observed hedge slip; the ARM bar
    # rises until the family prices itself out of resting — calm families keep
    # resting cheaply
    kalshi = FakeVenue("kalshi", [])
    poly = FakeVenue("poly", [])
    ex, _ = make_maker_exec([kalshi, poly], FakeConfirmer({}))
    ex.maker_arm_cushion = 0.005
    for _ in range(6):
        ex._note_maker_slip("poly", "aec-itfme-x-y-2026-07-09", 0.04)
    assert ex._family_slip("poly", "aec-itfme-a-b-2026-07-09") > 0.03   # family-wide
    assert ex._family_slip("poly", "aec-cs2-a-b-2026-07-09") == 0.0     # calm family
    # the raised bar shows up as a DEEPER rest price: the maker manufactures
    # arm-worth of edge including the family slip (rest_cap = 1 - hedge_ask -
    # fees - arm). Without slip it would rest ~0.57; with ~3.7c slip <= ~0.553.
    kalshi2 = FakeVenue("kalshi", [res("kalshi", Side.YES, OrderStatus.RESTING, 0, None)])
    poly2 = FakeVenue("poly", [])
    ex2, _ = make_maker_exec([kalshi2, poly2], FakeConfirmer({}))
    ex2.maker_arm_cushion = 0.005
    from bot.execution.executor import _family
    ex2._maker_slip = {_family("poly", "P1"): 0.037}
    asyncio.run(ex2.execute_maker(
        opp(yv="kalshi", nv="poly", max_contracts=5, yes_price=0.58, no_price=0.40)))
    rest_px = [c[3] for c in kalshi2.venues["kalshi"].calls] if hasattr(kalshi2, "venues") else               [c[3] for c in kalshi2.calls if c[2] == "buy"]
    assert rest_px and rest_px[0] <= 0.556, f"rest price must include slip, got {rest_px}"


def test_maker_degrades_to_taker_in_play():
    # in-play kalshi ticker (start time in the past) -> execute_maker routes to the
    # TAKER path; a pre-game/dateless ticker keeps the maker
    import time as _t
    from bot.execution.executor import Executor
    assert Executor._kalshi_start_ts("KXMLBTOTAL-26JUL091840ATHDET-9") is not None
    assert Executor._kalshi_start_ts("KXMLBDRAFTTOP-26-5-EBOO") is None
    kalshi = FakeVenue("kalshi", [res("kalshi", Side.NO, OrderStatus.KILLED, 0, None)])
    poly = FakeVenue("poly", [res("poly", Side.YES, OrderStatus.KILLED, 0, None)])
    ex, _ = make_maker_exec([kalshi, poly], FakeConfirmer({}))
    o = opp(yv="poly", nv="kalshi", max_contracts=5, yes_price=0.40, no_price=0.55)
    # in-play: buy_no_market must look like a live-game ticker
    o.buy_no_market = "KXMLBTOTAL-20JAN010000ATHDET-9"   # long past -> in play
    report = asyncio.run(ex.execute_maker(o))
    # taker path evidence: leg1 attempted as IOC BUY (killed -> clean skip), no rest
    assert report.status is ExecStatus.SKIPPED and "leg1" in report.reason


def test_fire_slip_raises_the_entry_bar():
    # a family whose fills consistently realize 1.5c under the detected edge must
    # demand that slip up front — thin edges stop firing there, fat ones still do
    kalshi = FakeVenue("kalshi", []); poly = FakeVenue("poly", [])
    ex, _ = make_exec([kalshi, poly])
    o = opp(yv="poly", nv="kalshi", max_contracts=10, yes_price=0.40, no_price=0.585)
    for _ in range(8):
        ex._note_fire_slip(o, (o.edge_per_contract or 0.015) - 0.015)
    assert ex._fire_slip_for(o) > 0.01
    report = asyncio.run(ex.execute(o))     # 1.5c edge < floor + slip -> skip
    assert report.status is ExecStatus.SKIPPED and "hedge" in report.reason
    assert kalshi.calls == [] and poly.calls == []
