"""Live two-leg arbitrage executor (LIVE_SMALL).

Turns a detected :class:`ArbOpportunity` into real orders, market-neutrally:

  1. Size the trade (min of available depth, per-order cap, and risk headroom).
  2. Place leg 1 (buy YES) as **fill-or-kill** — all-or-nothing, no partial.
     - KILLED/REJECTED  -> no position; abort cleanly.
     - ERROR (unknown)  -> HALT + kill switch (we can't tell if it filled).
  3. Place leg 2 (buy NO) as fill-or-kill.
     - FILLED            -> locked: $1 payout vs sub-$1 cost. Record profit.
     - KILLED/REJECTED   -> **auto-unwind leg 1** (sell it back, take the small loss).
     - ERROR/PARTIAL     -> HALT + kill switch (don't risk doubling up).

The "halt on unknown" rule is deliberate: auto-unwind only happens when leg 2
definitively did NOT fill. Any ambiguous state stops trading for human reconciliation
rather than guessing. Works for both cross-venue (two venues) and single-venue bundle
(same venue) arbs — the leg venues just happen to be equal in the bundle case.

This is the real-money path. It runs only when the runner is in a live mode AND the
venues have trading credentials. Validate against sandbox/demo first.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field, replace
from enum import Enum

from bot.data.store import Store
from bot.execution.orders import OrderResult, OrderStatus
from bot.execution.risk import RiskManager
from bot.fees import FeeModel, ZeroFeeModel
from bot.models import Side
from bot.strategies.arbitrage import ArbOpportunity
from bot.venues.base import RawMarket

log = logging.getLogger("bot.executor")


def _reject_reason(result) -> str:
    """Human-readable why-it-failed from an OrderResult's raw payload (HTTP status +
    venue error body), so a rejection isn't an opaque [REJECTED] in the log."""
    raw = getattr(result, "raw", None) or {}
    parts = []
    if raw.get("http_status"):
        parts.append(f"HTTP {raw['http_status']}")
    body = raw.get("body") or raw.get("error")
    if body:
        parts.append(str(body)[:200])
    return " ".join(parts) or "no detail"


class ExecStatus(str, Enum):
    SUCCESS = "SUCCESS"      # both legs filled — locked arb
    SKIPPED = "SKIPPED"      # didn't act (risk/size/leg-1 no fill)
    UNWOUND = "UNWOUND"      # leg 1 filled, leg 2 didn't, leg 1 sold back
    HALTED = "HALTED"        # ambiguous/unknown state — kill switch tripped
    QUARANTINED = "QUARANTINED"  # stuck leg recorded + market blacklisted; KEEP trading others


def scaled_hedge_buffer(buffer: float, depth: float, thin_depth: float,
                        deep_depth: float) -> float:
    """The hedge buffer to require for a given book ``depth``. The buffer is a price
    cushion that lets the second (hedge) leg fill through movement without unwinding — but
    a DEEP book fills at the touch and barely needs it, while a THIN book needs the full
    amount. Scale linearly from the full ``buffer`` at/below ``thin_depth`` to a one-tick
    floor at/above ``deep_depth``, so the bot can fire on the SMALLER (more frequent)
    divergence windows on deep markets while keeping full protection on thin ones.
    ``deep_depth <= 0`` disables scaling (always the full buffer)."""
    floor = min(buffer, 0.01)
    thin = max(thin_depth, 1.0)
    if deep_depth <= thin or depth <= thin:
        return buffer
    if depth >= deep_depth:
        return floor
    frac = (depth - thin) / (deep_depth - thin)
    return round(buffer - frac * (buffer - floor), 4)


@dataclass
class ExecutionReport:
    status: ExecStatus
    reason: str = ""
    legs: list[OrderResult] = field(default_factory=list)
    realized_pnl: float = 0.0

    def __str__(self) -> str:
        return f"[{self.status.value}] pnl={self.realized_pnl:+.2f} {self.reason}".strip()


class Executor:
    def __init__(
        self,
        venues: dict,
        risk: RiskManager,
        *,
        fee_models: dict[str, FeeModel] | None = None,
        store: Store | None = None,
        max_order_contracts: float = 2.0,
        unwind_slippage: float = 0.05,
        fill_confirmer=None,
        confirm_timeout: float = 5.0,
        balance_buffer: float = 0.99,
        min_venue_balance: float = 0.0,
        probe_contracts: float = 0.0,
        market_proven_fills: int = 3,
        market_max_fails: int = 2,
        min_lock_edge: float | None = None,
        leg2_slippage_share: float = 0.6,
        min_leg_depth: float = 0.0,
        depth_safety: float = 1.0,
        first_venue: str = "kalshi",
        take_first_venue: str | None = None,
        hedge_buffer: float = 0.0,
        maker_timeout: float = 5.0,
        maker_improvement: float = 0.01,
        maker_arm_cushion: float = 0.0,
        maker_poll: float = 0.0,
        hedge_retries: int = 1,
        maker_dynamic: bool = False,
        buffer_deep_depth: float = 0.0,
    ) -> None:
        self.venues = venues
        self.risk = risk
        self.fee_models = fee_models or {}
        self.store = store
        # Per-order contract ceiling; <= 0 means "no ceiling" (size up to balances,
        # depth, and risk caps). Kept as an optional safety cap.
        self.max_order_contracts = max_order_contracts
        self.unwind_slippage = unwind_slippage
        # Min resting depth (contracts) required on BOTH legs before firing. Thin books
        # are where a leg rejects and the other can't be hedged/unwound — skipping them
        # means we should never have to unwind. 0 = disabled.
        self.min_leg_depth = min_leg_depth
        # Fraction of shown top-of-book depth to actually trade (headroom so a FOK still
        # fills if the book thins between the quote and the order). 1.0 = use it all.
        self.depth_safety = depth_safety
        # The rejection-prone venue (Kalshi: thinner books / FOK insufficient resting
        # volume) is placed FIRST, so if it rejects there's no other leg to unwind.
        self.first_venue = first_venue
        # Which venue fires FIRST in a two-FOK hybrid TAKE — independent of the maker-rest
        # venue (``first_venue``). Set this to the REJECTION-PRONE venue so its failure is a
        # clean skip, not an unwind: live data shows Polymarket 500s the hedge buy on thin
        # markets, so firing it first turns those 500s into free skips instead of Kalshi
        # unwinds. Defaults to ``first_venue`` (unchanged behavior) when not set.
        self.take_first_venue = take_first_venue or first_venue
        self.fill_confirmer = fill_confirmer       # optional FillTracker (private WS)
        self.confirm_timeout = confirm_timeout
        # Available cash per venue, seeded from the startup snapshot and decremented as
        # legs fill. Used to size each arb to the most the funded balances allow.
        self.balance_buffer = balance_buffer       # leave headroom for fees/slippage
        self.min_venue_balance = min_venue_balance  # don't trade a leg whose venue is below this
        # Empirical fill-reliability (probe-then-scale): a market is PROBED at probe_contracts
        # until its FOK orders have FILLED market_proven_fills times (proving real depth), then
        # scaled to full size; a market that KILL/REJECTs market_max_fails times without proving
        # is excluded. The live, learned replacement for the volume proxy. 0 probe = disabled.
        self.probe_contracts = probe_contracts
        self.market_proven_fills = market_proven_fills
        self.market_max_fails = market_max_fails
        self._market_rel: dict[tuple, tuple] = (
            self.store.market_reliability() if self.store is not None else {})
        self._balances: dict[str, float] = {}
        # Bounded-aggressive limit pricing: how much edge to preserve as locked profit
        # (defaults to the risk min_edge floor).
        self.min_lock_edge = min_lock_edge
        self.leg2_slippage_share = min(max(leg2_slippage_share, 0.0), 1.0)
        # Price cushion reserved for the SECOND (hedge) leg so it fills through normal
        # book movement — and we ONLY fire when the edge can pay it AND still lock the
        # floor, so a thin edge that would just unwind never fires. 0 = no cushion.
        self.hedge_buffer = hedge_buffer
        # Maker mode: how long (s) a resting maker order may wait to fill before it
        # auto-expires (and we give up on that arb). The taker hedge fires the instant
        # the maker fills, so the naked window is ~one round-trip, not this whole time.
        self.maker_timeout = maker_timeout
        # How far INSIDE the ask to post the maker (>= one tick so post-only doesn't
        # reject it as crossing). Also captures this much extra edge on a fill.
        self.maker_improvement = maker_improvement
        # Edge required ABOVE the lock floor before RESTING a maker. The resting leg is
        # exposed to the taker drifting against it (adverse selection); without a cushion
        # a 1-2c edge fills into an adverse move and the forced hedge locks a loss. Mirrors
        # hedge_buffer on the taker path, but for the maker's drift-while-resting risk.
        self.maker_arm_cushion = maker_arm_cushion
        # While a maker rests, poll the taker leg every this-many seconds; if it drifts so
        # the hedge could no longer lock the floor, CANCEL the maker before it fills into
        # the adverse move. This is what makes arming THIN edges safe (the cushion is the
        # static guard at fire time; this is the dynamic guard while resting). 0 = disabled
        # (rest blindly until fill/expiry — only safe with a large arm cushion).
        self.maker_poll = maker_poll
        # After a maker fills, the forced hedge may hit a TRANSIENT venue error (e.g. a
        # Polymarket 500/timeout during a WS wobble). Re-place the hedge up to this many
        # times — but ONLY after reconciling the hedge venue to confirm nothing landed, so
        # a retry can never double up. Exhausting the retries falls through to the unwind.
        self.hedge_retries = max(0, int(hedge_retries))
        # Dynamic maker side: rest the maker on whichever leg is the liquidity bottleneck
        # (the THINNER book) and TAKE the deeper leg, instead of always resting on
        # first_venue. Lets a thin Polymarket leg be SOURCED via its own flow as a maker
        # rather than skipped because its book can't be taken. Requires the maker venue to
        # support post-only (both adapters do). False = always rest on first_venue.
        self.maker_dynamic = maker_dynamic
        # Depth at/above which a market is "deep" enough that the hedge fills at the touch,
        # so the hedge buffer scales down to one tick — letting the bot take the smaller
        # divergence windows on deep books. 0 = off (always the full hedge_buffer).
        self.buffer_deep_depth = buffer_deep_depth

    def _hedge_buffer_for(self, depth: float) -> float:
        """The depth-scaled hedge buffer for a fire of this size (see scaled_hedge_buffer)."""
        return scaled_hedge_buffer(
            self.hedge_buffer, depth, self.min_leg_depth, self.buffer_deep_depth)

    def set_balances(self, snapshots) -> None:
        """Seed available cash per venue from account snapshots (startup/refresh)."""
        for snap in snapshots:
            bal = getattr(snap, "balance", None)
            if bal is not None:
                self._balances[snap.venue] = float(bal)

    def _balance(self, venue: str) -> float | None:
        return self._balances.get(venue)

    def _spend(self, venue: str, amount: float) -> None:
        """Adjust tracked cash after a fill (negative ``amount`` credits it back)."""
        if venue in self._balances:
            self._balances[venue] = max(0.0, self._balances[venue] - amount)

    def _fee(self, venue: str) -> FeeModel:
        return self.fee_models.get(venue, ZeroFeeModel())

    async def _place(self, venue, market_id, side, action, price, contracts, tif) -> OrderResult:
        """Place an order, converting any raised exception into an ERROR result so a
        venue error never crashes the loop (it routes to the halt path instead)."""
        try:
            result = await venue.place_order(market_id, side, action, price, contracts, tif=tif)
        except Exception as exc:
            return OrderResult(
                getattr(venue, "name", "?"), market_id, side, action, contracts,
                status=OrderStatus.ERROR, raw={"error": str(exc)},
            )
        # Authoritative fill confirmation from the private WS, when available — avoids
        # depending on the synchronous REST response shape.
        if self.fill_confirmer is not None and result.order_id and result.status is not OrderStatus.ERROR:
            try:
                status, filled, avg = await self.fill_confirmer.confirm(
                    result.venue, result.order_id, contracts, self.confirm_timeout
                )
                # The synchronous REST result is authoritative for a terminal outcome.
                # Only let the stream OVERRIDE it to upgrade (e.g. REST KILLED -> a
                # confirmed FILL); never let a timed-out / non-terminal stream result
                # downgrade a terminal REST result into a spurious PARTIAL/KILLED.
                terminal = {OrderStatus.FILLED, OrderStatus.KILLED, OrderStatus.REJECTED}
                if status in terminal or result.status not in terminal:
                    result.status, result.filled = status, filled
                    # Confirmer price is raw venue convention (Polymarket = YES-side);
                    # prefer the per-venue-converted price from place_order.
                    if avg is not None and result.avg_price is None:
                        result.avg_price = avg
            except Exception as exc:  # confirmer failure -> keep REST result
                log.warning("fill confirm failed for %s: %s", result.order_id, exc)
        # Empirical fill-reliability: a FOK BUY that FILLED proves the market's depth is real;
        # a KILL/REJECT/ERROR/partial proves it's phantom (the naked-leg source). Record per
        # market (not on unwinds — sells, which we only do to recover) to drive probe sizing.
        if action == "buy" and self.probe_contracts > 0:
            self._record_market_reliability(
                getattr(venue, "name", "?"), market_id,
                result.status is OrderStatus.FILLED)
        return result

    def _record_market_reliability(self, venue: str, market_id: str, ok: bool) -> None:
        key = (venue, market_id)
        fills, fails, streak = self._market_rel.get(key, (0, 0, 0))
        self._market_rel[key] = (fills + (1 if ok else 0), fails + (0 if ok else 1),
                                 0 if ok else streak + 1)
        if self.store is not None:
            try:
                self.store.record_market_outcome(venue, market_id, ok)
            except Exception as exc:
                log.warning("market reliability write failed for %s: %s", market_id, exc)
        if not ok:
            f, x, s = self._market_rel[key]
            log.info("reliability: %s:%s FOK failed (%d fills/%d fails/%d streak) — "
                     "%s", venue, market_id, f, x, s,
                     "EXCLUDED (depth vanished)" if s >= self.market_max_fails
                     else "probe-gated until proven")

    async def _position_after_error(self, venue, market_id):
        """After an ambiguous leg ERROR, ask the venue whether a position actually
        resulted. Returns True (a position exists -> naked risk), False (flat -> the
        order didn't fill, safe to skip), or None (couldn't determine -> fail closed)."""
        snap_fn = getattr(venue, "account_snapshot", None)
        if snap_fn is None:
            return None
        try:
            snap = await snap_fn()
        except Exception as exc:
            log.warning("post-error reconcile failed for %s: %s", market_id, exc)
            return None
        for pos in getattr(snap, "positions", None) or []:
            if pos.market_id == market_id and pos.is_open:
                return True
        return False

    async def _hedge_qty_after_error(self, venue, market_id):
        """After an ambiguous leg-2 ERROR (e.g. a venue 500 on POST /orders), ask the
        venue HOW MANY contracts of the hedge actually landed. Returns the open quantity
        (>= 0.0) or None if it couldn't be determined (fail closed -> halt). Unlike
        ``_position_after_error`` this returns the size, so the executor can settle a fully
        hedged trade (the common '500 but it actually filled' case) instead of freezing."""
        snap_fn = getattr(venue, "account_snapshot", None)
        if snap_fn is None:
            return None
        try:
            snap = await snap_fn()
        except Exception as exc:
            log.warning("post-error hedge reconcile failed for %s: %s", market_id, exc)
            return None
        for pos in getattr(snap, "positions", None) or []:
            if pos.market_id == market_id and pos.is_open:
                return abs(pos.quantity)
        return 0.0

    async def _hedge_fillable(self, venue, leg, size) -> float:
        """How many contracts the hedge (taker) leg would actually fill at its limit, read
        from the REAL order book — re-fetched live right before we commit leg 1. A book that
        thinned or moved since the depth-confirm then caps the trade (or skips it) instead of
        leaving a naked remainder. Returns the resting top-of-book size on the buy side when
        the current ask is at/through our limit; 0.0 when there's no offer, the price moved
        past our limit, or the fetch fails (an unconfirmable hedge must NOT arm a leg).

        Why the live book and not an order preview: Polymarket's /v1/order/preview does NOT
        simulate matching — it echoes the order with cumQuantity=0 for everything, so it
        always read as "fills nothing" and blocked every trade. The /book depth is the
        authoritative, verified source (``--show-book`` matches it exactly)."""
        market_id, side = leg[1], leg[2]
        try:
            q = await venue.fetch_quote(RawMarket(market_id=market_id, title="", raw={}))
        except Exception as exc:
            log.warning("hedge fillable: book fetch failed for %s: %s", market_id, exc)
            return 0.0                               # can't confirm -> don't arm
        ask = q.yes_ask if side is Side.YES else q.no_ask
        depth = q.yes_ask_size if side is Side.YES else q.no_ask_size
        if ask is None or not depth:
            return 0.0                               # no resting offer -> can't hedge
        # The buy side's resting top-of-book depth. We don't gate on the stale leg limit
        # here: the taker leg fires FOK (a book that moved past the limit kills cleanly ->
        # unwind), and the maker leg REPRICES the hedge to the live ask on fill — so the
        # only thing this pre-check must establish is that real depth exists to hedge into.
        return float(depth)

    def _max_size(self, opp: ArbOpportunity,
                  depth_override: float | None = None) -> tuple[int, dict[str, float]]:
        """Largest whole-contract size that fits every hard limit at once:

          * available order-book depth (can't fill more than is quoted) — or
            ``depth_override`` when the binding depth isn't ``max_contracts`` (maker mode
            sizes against the HEDGE leg, since the resting leg adds liquidity),
          * funded cash on each leg's venue (YES leg needs cash on the YES venue at
            ``yes_price``; NO leg needs cash on the NO venue at ``no_price``),
          * the per-market and total exposure risk caps,
          * the optional per-order contract ceiling.

        Returns ``(size, caps)`` where ``caps`` is each constraint's contract limit
        (for logging why a size was chosen).
        """
        label = f"{opp.buy_yes_venue}:{opp.buy_yes_market}"
        # Trade only a fraction of shown depth (headroom for a thinning book on FOK).
        depth = depth_override if depth_override is not None else float(opp.max_contracts)
        caps: dict[str, float] = {"depth": depth * self.depth_safety}

        yb, nb = self._balance(opp.buy_yes_venue), self._balance(opp.buy_no_venue)
        # Min-venue-balance guard: a venue too drained to reliably fund/hedge a leg must NOT
        # trade — else we fire the first (Polymarket) leg and the second (Kalshi) leg rejects
        # for insufficient_balance, leaving a naked position (the Kalshi-at-$0.38 incident).
        # Cap that leg to 0 -> size 0 -> clean skip. Self-healing: resumes when funds return.
        floor = self.min_venue_balance
        if yb is not None and opp.yes_price > 0:
            caps["cash_yes"] = 0.0 if (floor > 0 and yb < floor) else (yb * self.balance_buffer) / opp.yes_price
        if nb is not None and opp.no_price > 0:
            caps["cash_no"] = 0.0 if (floor > 0 and nb < floor) else (nb * self.balance_buffer) / opp.no_price

        gross = opp.gross_cost
        if gross > 0:
            lim = self.risk.limits
            caps["per_market"] = max(0.0, lim.max_position_per_market - self.risk.position(label)) / gross
            caps["total"] = max(0.0, lim.max_total_exposure - self.risk.total_exposure) / gross

        if self.max_order_contracts and self.max_order_contracts > 0:
            caps["order_cap"] = float(self.max_order_contracts)

        # Empirical fill-reliability (the live replacement for the volume proxy): cap each
        # leg by what its market's real FOK history has proven. Unproven markets are bounded
        # to a tiny probe; markets that have repeatedly failed without ever filling are
        # excluded. The min across both legs governs — a phantom on EITHER leg can leave a
        # naked position. proven markets are unbounded here (other caps apply).
        if self.probe_contracts > 0:
            caps["reliability"] = min(
                self._reliability_cap(opp.buy_yes_venue, opp.buy_yes_market),
                self._reliability_cap(opp.buy_no_venue, opp.buy_no_market),
            )

        return math.floor(max(0.0, min(caps.values()))), caps

    def _reliability_cap(self, venue: str, market_id: str) -> float:
        """Contract ceiling a market's empirical FOK history earns it. Unproven markets get
        a tiny probe; proven markets RAMP up with accumulated fills (not a jump to full);
        a consecutive-fail streak (or repeated fails before proving) excludes."""
        fills, fails, streak = self._market_rel.get((venue, market_id), (0, 0, 0))
        # CONSECUTIVE fails exclude EVEN a once-proven market: a Valorant/tennis market that
        # filled early then had its depth drain as the game wound down kept firing into
        # vanished volume -> repeated hedge-reject unwinds. A live fill resets the streak.
        if streak >= self.market_max_fails:
            return 0.0
        if fills < self.market_proven_fills:
            if fails >= self.market_max_fails:
                return 0.0
            return float(self.probe_contracts)
        # Proven -> RAMP, don't jump. Three 2-contract probe fills prove "2 fill", not "20
        # do" — so a market whose depth is only good for small size must not immediately fire
        # full size and unwind (the -$0.60 Lamine-Yamal jump). Each fill past the proving bar
        # earns +probe_contracts; a fail trips the streak/exclusion before size grows large.
        return float(self.probe_contracts) * (fills - self.market_proven_fills + 1)

    def _leg_limits(self, opp: ArbOpportunity, first_side, second_side) -> tuple[float, float]:
        """Limit prices for the (first, second) legs that may pay worse than the quoted
        ask to fill on a moving book, but never enough to drop the locked profit below
        the floor.

        The buffer goes to the SECOND (hedge) leg: its failure forces an unwind, while a
        first-leg failure is a clean skip — so the hedge gets up to ``hedge_buffer`` of
        room to fill through adverse movement. Any leftover surplus (edge - floor -
        hedge) widens the first leg (improves its fill at no unwind cost). Worst case
        both fill at the limits -> combined cost = gross + surplus = 1 - floor, so the
        locked profit is >= floor by construction."""
        floor = self.min_lock_edge if self.min_lock_edge is not None else self.risk.limits.min_edge
        surplus = max(0.0, opp.edge_per_contract - floor)
        hedge = min(self._hedge_buffer_for(opp.max_contracts), surplus)
        leftover = surplus - hedge

        def px(side):
            return opp.yes_price if side is Side.YES else opp.no_price

        first_limit = min(0.99, round(px(first_side) + leftover, 4))
        second_limit = min(0.99, round(px(second_side) + hedge, 4))
        return first_limit, second_limit

    async def execute(self, opp: ArbOpportunity) -> ExecutionReport:
        if self.risk.is_killed:
            return ExecutionReport(ExecStatus.SKIPPED, f"kill switch: {self.risk.kill_reason}")

        # Liquidity guard: don't fire unless BOTH legs have real resting depth. Thin
        # books are where a leg rejects and we can't hedge/unwind — skip them outright
        # so we should never have to unwind. opp.max_contracts is the min leg depth.
        if self.min_leg_depth > 0 and opp.max_contracts < self.min_leg_depth:
            return ExecutionReport(
                ExecStatus.SKIPPED,
                f"thin book: depth {opp.max_contracts:g} < min {self.min_leg_depth:g} contracts",
            )

        size, caps = self._max_size(opp)
        if size < 1:
            binding = min(caps, key=caps.get)
            return ExecutionReport(
                ExecStatus.SKIPPED,
                f"size < 1 contract (binding: {binding}={caps[binding]:.3f})",
            )
        binding = min(caps, key=caps.get)
        log.info("sizing %s: %d contracts (binding: %s) caps=%s",
                 opp.event_key, size, binding,
                 {k: round(v, 2) for k, v in caps.items()})

        # Order the two legs so the REJECTION-PRONE venue (self.take_first_venue) fires
        # FIRST: if it rejects/errors, no other leg was taken -> a clean skip, no unwind.
        # The SECOND (hedge) leg is the RELIABLE one whose failure would force an unwind —
        # and it's the leg the hedge-fillable pre-check below confirms can fill. (Live data:
        # Polymarket 500s the hedge buy on thin markets, so it must go first; then a 500 is a
        # free skip instead of a Kalshi unwind.)
        tfv = self.take_first_venue
        if opp.buy_no_venue == tfv and opp.buy_yes_venue != tfv:
            first_vn, first_m, first_side = opp.buy_no_venue, opp.buy_no_market, Side.NO
            second_vn, second_m, second_side = opp.buy_yes_venue, opp.buy_yes_market, Side.YES
        else:
            first_vn, first_m, first_side = opp.buy_yes_venue, opp.buy_yes_market, Side.YES
            second_vn, second_m, second_side = opp.buy_no_venue, opp.buy_no_market, Side.NO

        # Don't fire an edge too thin to give the hedge leg a fill cushion — it would
        # just unwind. We only fire when the edge can pay the hedge buffer AND still lock
        # the floor, so the hedge fills through normal book movement (no unwind).
        floor = self.min_lock_edge if self.min_lock_edge is not None else self.risk.limits.min_edge
        req_buffer = self._hedge_buffer_for(opp.max_contracts)
        if opp.edge_per_contract < floor + req_buffer - 1e-9:
            return ExecutionReport(
                ExecStatus.SKIPPED,
                f"edge {opp.edge_per_contract:.3f} < lock {floor:.3f} + hedge "
                f"{req_buffer:.3f} (depth {opp.max_contracts:g}) — would risk an unwind",
            )
        first_limit, second_limit = self._leg_limits(opp, first_side, second_side)
        notional = size * (first_limit + second_limit)   # worst-case cost for the risk check
        label = f"{opp.buy_yes_venue}:{opp.buy_yes_market}"
        decision = self.risk.check(label, notional)
        if not decision.allowed:
            return ExecutionReport(ExecStatus.SKIPPED, f"risk: {decision.reason}")

        first = (first_vn, first_m, first_side, first_limit)
        second = (second_vn, second_m, second_side, second_limit)
        first_venue = self.venues.get(first[0])
        second_venue = self.venues.get(second[0])
        if first_venue is None or second_venue is None:
            return ExecutionReport(ExecStatus.SKIPPED, "venue not available")

        # Re-confirm the hedge (deep) leg's LIVE book before committing leg 1 and SIZE DOWN
        # to what it would actually fill. Polymarket can thin out between the depth-confirm
        # and the fire, so taking the full size would leave the hedge short (a naked
        # remainder) or fire a FOK into vanished liquidity (the 500 trigger). Sizing to the
        # live fillable depth keeps the trade fully hedged at whatever the leg supports —
        # capturing a small real arb instead of skipping it. A fetch failure -> 0 (skip).
        fillable = await self._hedge_fillable(second_venue, second, size)
        if fillable < size - 1e-9:
            capped = int(fillable + 1e-9)
            if capped < 1:
                self._audit("execute_skip_hedge", opp, size=size)
                log.info("STREAM skip %s — hedge unfillable: book depth <1 contract at limit",
                         opp.event_key)
                return ExecutionReport(
                    ExecStatus.SKIPPED, "hedge unfillable: book depth <1 contract at limit")
            log.info("STREAM %s: sizing down to hedge-fillable %d (book showed %d)",
                     opp.event_key, capped, size)
            size = capped

        self._audit("execute_start", opp, size=size)

        # Reserve the estimated leg cost in the cached balance SYNCHRONOUSLY (before any
        # await) so a CONCURRENT fast-loop execution sees the drain immediately and doesn't
        # over-commit a draining venue -> a leg's insufficient_balance reject -> unwind. The
        # cache otherwise only refreshes on the slow (~minutes) loop and decrements at settle,
        # so two arbs firing together both read the full balance. Released in finally; the
        # real per-fill accounting stays in _settle_success / _unwind.
        reservation: dict[str, float] = {}
        reservation[first[0]] = reservation.get(first[0], 0.0) + first[3] * size
        reservation[second[0]] = reservation.get(second[0], 0.0) + second[3] * size
        for _v, _amt in reservation.items():
            self._spend(_v, _amt)
        try:
            return await self._fire_and_settle(
                opp, size, first, second, first_venue, second_venue)
        finally:
            for _v, _amt in reservation.items():
                self._spend(_v, -_amt)

    async def _fire_and_settle(self, opp, size, first, second, first_venue, second_venue):
        """Fire leg 1 (rejection-prone) then leg 2 (hedge), then resolve the outcome:
        settle a locked arb, unwind a clean leg-2 failure, or halt on an ambiguous state.
        Split out of execute() so its balance reservation can wrap this in try/finally."""
        # ----- Leg 1: the rejection-prone leg, fill-or-kill -----
        leg1 = await self._place(
            first_venue, first[1], first[2], "buy", first[3], size, "fill_or_kill"
        )
        log.info("leg1 %s", leg1)
        if leg1.status is OrderStatus.ERROR:
            # An ambiguous order error (e.g. a transient network timeout) shouldn't
            # freeze the whole bot. Reconcile against the venue: if no position
            # resulted, it's a clean skip and we keep trading; only halt if a position
            # exists (real naked risk) or we can't tell (fail closed).
            left = await self._position_after_error(first_venue, first[1])
            if left is False:
                log.warning("leg1 ERROR but %s flat — no position, skipping (%s)",
                            first[1], _reject_reason(leg1))
                return ExecutionReport(
                    ExecStatus.SKIPPED,
                    f"leg1 ERROR, account flat — no position ({_reject_reason(leg1)})", [leg1])
            return self._halt(
                f"leg1 ERROR — {'position exists' if left else 'fill state unknown'} "
                f"({_reject_reason(leg1)})", [leg1])
        if not leg1.left_a_position:
            # Clean: the first leg didn't fill, so NO position was taken — just skip.
            return ExecutionReport(
                ExecStatus.SKIPPED,
                f"leg1 not filled ({leg1.status.value}: {_reject_reason(leg1)})", [leg1])
        if not leg1.filled_fully:
            # FoK didn't behave all-or-nothing (observed on Polymarket: a 0.62/6 fill).
            # The filled amount is KNOWN (not ambiguous), so unwind that portion and skip
            # rather than halting the whole bot and stranding a naked leg.
            log.warning("leg1 PARTIAL %g/%g — unwinding the filled portion", leg1.filled, size)
            return await self._unwind(opp, leg1, None, reason="leg1 partial fill")

        # ----- Leg 2: the hedge, fill-or-kill -----
        leg2 = await self._place(
            second_venue, second[1], second[2], "buy", second[3], size, "fill_or_kill"
        )
        log.info("leg2 %s", leg2)

        if leg2.status is OrderStatus.FILLED and leg2.filled_fully:
            return self._settle_success(opp, size, [leg1, leg2])

        if leg2.status in (OrderStatus.KILLED, OrderStatus.REJECTED) and leg2.filled <= 1e-9:
            # Definitively no leg-2 position -> safe to unwind leg 1. Carry WHY leg2
            # failed (HTTP status + venue body) into the unwind/log so a rejection is
            # diagnosable instead of an opaque [REJECTED].
            log.warning("leg2 %s — %s", leg2.status.value, _reject_reason(leg2))
            return await self._unwind(
                opp, leg1, leg2, reason=f"leg2 {leg2.status.value} ({_reject_reason(leg2)})"
            )

        if leg2.status is OrderStatus.ERROR:
            # A leg-2 ERROR (e.g. a transient Polymarket 500 on POST /orders) is ambiguous:
            # the hedge may or may not have landed. Rather than freeze the WHOLE bot on one
            # venue hiccup, reconcile against the venue (as leg 1 does) and act on the real
            # hedge state. The hedge order is synchronous fill-or-kill (not a resting order),
            # so the reconciled position is authoritative — it cannot land "later".
            qty = await self._hedge_qty_after_error(second_venue, second[1])
            if qty is None:
                # Couldn't read the account -> genuinely unknown -> fail closed (halt).
                return self._halt(
                    f"leg2 ambiguous ({leg2.status.value}: {_reject_reason(leg2)}; "
                    f"hedge state unreadable) — manual reconcile", [leg1, leg2])
            if qty >= size - 1e-9:
                # A 500 often means the server DID process the order then errored on the
                # response: the full hedge is on and the arb is locked -> settle it.
                log.warning("leg2 ERROR but %s holds %g contracts — hedge landed, settling (%s)",
                            second[1], qty, _reject_reason(leg2))
                leg2 = replace(
                    leg2, status=OrderStatus.FILLED, filled=size,
                    avg_price=leg2.avg_price if leg2.avg_price is not None else second[3])
                return self._settle_success(opp, size, [leg1, leg2])
            if qty <= 1e-9:
                # Hedge confirmed FLAT. The synchronous FOK didn't fill -> no leg-2 position
                # -> safe to unwind leg 1 and KEEP TRADING (one bad market doesn't freeze
                # the whole bot).
                log.warning("leg2 ERROR but %s flat — no hedge landed, unwinding leg1 (%s)",
                            second[1], _reject_reason(leg2))
                return await self._unwind(
                    opp, leg1, leg2,
                    reason=f"leg2 ERROR, hedge flat ({_reject_reason(leg2)})")
            # 0 < qty < size: a partial hedge — a known-but-mismatched naked remainder we
            # can't safely auto-resolve. Halt for manual reconciliation.
            return self._halt(
                f"leg2 ambiguous ({leg2.status.value}: {_reject_reason(leg2)}; "
                f"partial hedge {qty:g}/{size:g}) — manual reconcile", [leg1, leg2])

        # PARTIAL (or any other non-definitive state) on leg 2: a known-but-mismatched
        # fill we can't safely auto-resolve. Halt for manual reconciliation.
        return self._halt(
            f"leg2 ambiguous ({leg2.status.value}: {_reject_reason(leg2)}) — manual reconcile",
            [leg1, leg2],
        )

    async def _cancel_maker(self, venue, order_id) -> None:
        """Best-effort cancel of a resting maker. A GOOD_TILL_CANCEL maker does not
        self-expire, so the executor must cancel an unfilled one or it would rest unhedged.
        A cancel that races a fill is harmless — the fill confirmation stays authoritative.

        A 404/400 on the cancel means the order is no longer cancellable (already filled,
        expired, or cancelled) — which is exactly the state we wanted, so it's logged quietly
        WITH the venue's response body for diagnosis. Anything else (5xx, network) stays a
        warning. A genuinely stranded resting order that later fills is caught by the periodic
        naked-exposure reconciliation (EXEC_RECONCILE_HALT), the cross-cycle backstop."""
        cancel = getattr(venue, "cancel_order", None)
        if cancel is None or not order_id:
            return
        try:
            await cancel(order_id)
            return
        except Exception as exc:
            resp = getattr(exc, "response", None)
            code = getattr(resp, "status_code", None)
            body = ""
            if resp is not None:
                try:
                    body = resp.text[:200]
                except Exception:
                    body = ""
            if code in (400, 404):
                log.info("maker %s not cancellable (already filled/expired/cancelled): "
                         "HTTP %s %s", order_id, code, body)
            else:
                log.warning("maker cancel failed for %s: HTTP %s %s (%s)",
                            order_id, code, body, exc)

    async def _taker_ask(self, taker, taker_venue) -> float | None:
        """Current ask for the taker (hedge) leg from its live book, or None on failure."""
        try:
            q = await taker_venue.fetch_quote(RawMarket(market_id=taker[1], title="", raw={}))
        except Exception as exc:
            log.warning("maker drift check: taker quote failed for %s: %s", taker[1], exc)
            return None
        return q.yes_ask if taker[2] is Side.YES else q.no_ask

    async def _rest_with_drift_guard(
        self, m, maker, taker, maker_venue, taker_venue, maker_px, size, floor,
    ) -> tuple[float, float | None]:
        """Wait for the resting maker to fill/expire while polling the taker leg. If the
        taker drifts so the would-be hedge can no longer lock ``floor``, CANCEL the maker
        before it fills into the adverse move — the only safe way to arm thin edges. Returns
        ``(filled, avg_price)`` from the authoritative fill confirmation."""
        confirm = asyncio.ensure_future(
            self.fill_confirmer.confirm(m.venue, m.order_id, size, self.maker_timeout + 1.5))
        deadline = time.time() + self.maker_timeout
        cancelled = False
        try:
            while not confirm.done() and time.time() < deadline:
                done, _ = await asyncio.wait({confirm}, timeout=self.maker_poll)
                if confirm in done:
                    break
                ask = await self._taker_ask(taker, taker_venue)
                if ask is None:
                    continue
                fee = self._fee(maker[0]).fee(maker_px, size) + self._fee(taker[0]).fee(ask, size)
                edge_now = (size * (1.0 - maker_px - ask) - fee) / size
                if edge_now < floor - 1e-9:
                    log.info(
                        "maker adverse drift on %s: taker ask %.3f -> hedge edge %.3f < floor "
                        "%.3f, cancelling maker before fill", maker[1], ask, edge_now, floor)
                    await self._cancel_maker(maker_venue, m.order_id)
                    cancelled = True
                    break
        except asyncio.CancelledError:
            confirm.cancel()
            raise
        # A GOOD_TILL_CANCEL maker does not self-expire: if it neither filled nor was already
        # cancelled on drift, cancel it now — BEFORE awaiting the authoritative confirm, so a
        # fill racing the cancel is still caught — otherwise it would rest on the book unhedged.
        if not confirm.done() and not cancelled:
            await self._cancel_maker(maker_venue, m.order_id)
        # Whether we cancelled or not, the confirmation is authoritative: a cancel that
        # raced a fill still reports the real filled amount (which we then hedge).
        _, filled, avg = await confirm
        return filled, avg

    async def execute_maker(self, opp: ArbOpportunity) -> ExecutionReport:
        """Capture an edge by RESTING the fee-heavy leg (``first_venue``, e.g. Kalshi)
        as a maker — no slippage, lower fee — then TAKING the deep leg (Polymarket) the
        instant the maker fills. Lets THIN edges be captured (we capture the spread
        instead of paying it). An unfilled maker is cancelled at the timeout, so an
        uncrossed quote is a clean no-trade. The naked window is just the maker-fill ->
        taker round trip; a hedge that can't fully fill re-crosses, then unwinds the
        unhedged remainder — it never leaves a naked leg resting.
        """
        if self.risk.is_killed:
            return ExecutionReport(ExecStatus.SKIPPED, f"kill switch: {self.risk.kill_reason}")
        if self.fill_confirmer is None:
            return ExecutionReport(ExecStatus.SKIPPED, "maker mode needs a fill confirmer")

        # Maker leg = the first_venue (fee-heavy) side; taker = the deep, cheap side. The
        # HEDGE (taker) leg's depth is what binds in maker mode: the resting maker ADDS
        # liquidity (its own book depth doesn't constrain us), but we must cross the taker
        # to hedge a fill. So guard/size on the taker leg's own depth, not min(both legs).
        yes_leg = (opp.buy_yes_venue, opp.buy_yes_market, Side.YES, opp.yes_price)
        no_leg = (opp.buy_no_venue, opp.buy_no_market, Side.NO, opp.no_price)
        if self.maker_dynamic and opp.yes_size > 0 and opp.no_size > 0:
            # Rest the maker on the THIN leg (the bottleneck); TAKE the deep leg (it fills
            # reliably). hedge_depth is the taker leg's own size — what actually binds.
            if opp.yes_size <= opp.no_size:
                maker, taker, hedge_depth = yes_leg, no_leg, opp.no_size
            else:
                maker, taker, hedge_depth = no_leg, yes_leg, opp.yes_size
        elif opp.buy_no_venue == self.first_venue:
            maker, taker, hedge_depth = no_leg, yes_leg, (opp.yes_size or opp.max_contracts)
        elif opp.buy_yes_venue == self.first_venue:
            maker, taker, hedge_depth = yes_leg, no_leg, (opp.no_size or opp.max_contracts)
        else:
            return await self.execute(opp)   # neither leg on the maker venue -> taker path

        if self.min_leg_depth > 0 and hedge_depth < self.min_leg_depth:
            return ExecutionReport(
                ExecStatus.SKIPPED,
                f"thin hedge book: depth {hedge_depth:g} < min {self.min_leg_depth:g}")

        size, caps = self._max_size(opp, depth_override=hedge_depth)
        if size < 1:
            binding = min(caps, key=caps.get)
            return ExecutionReport(
                ExecStatus.SKIPPED, f"size < 1 (binding: {binding}={caps[binding]:.3f})")

        floor = self.min_lock_edge if self.min_lock_edge is not None else self.risk.limits.min_edge
        # Arm only when the edge clears the lock floor PLUS a drift cushion: the maker
        # rests exposed to the taker moving against it, and the post-fill hedge is forced
        # (we hold the maker fill), so a sub-cushion edge that drifts locks a guaranteed
        # loss. The cushion is the maker analog of the taker path's hedge buffer.
        arm = floor + self.maker_arm_cushion
        if opp.edge_per_contract < arm - 1e-9:
            return ExecutionReport(
                ExecStatus.SKIPPED,
                f"edge {opp.edge_per_contract:.3f} < lock {floor:.3f} + maker cushion "
                f"{self.maker_arm_cushion:.3f} — would risk an adverse-fill loss")

        maker_venue = self.venues.get(maker[0])
        taker_venue = self.venues.get(taker[0])
        if maker_venue is None or taker_venue is None:
            return ExecutionReport(ExecStatus.SKIPPED, "venue not available")

        # Don't arm a maker we can't hedge. Re-read the taker (hedge) leg's LIVE book and
        # SIZE DOWN to its real top-of-book depth; skip entirely if it can't fill at all
        # (else the post-fill hedge KILLs at the cap and forces an unwind). The taker path
        # does the same before committing leg 1. A fetch failure -> 0 (skip, never arm blind).
        fillable = await self._hedge_fillable(taker_venue, taker, size)
        if fillable < 1:
            return ExecutionReport(
                ExecStatus.SKIPPED,
                f"hedge unfillable: book depth {fillable:g} < 1 — would arm an unhedgeable maker")
        if fillable < size:
            size = int(fillable)

        notional = size * (opp.yes_price + opp.no_price)
        decision = self.risk.check(f"{opp.buy_yes_venue}:{opp.buy_yes_market}", notional)
        if not decision.allowed:
            return ExecutionReport(ExecStatus.SKIPPED, f"risk: {decision.reason}")

        self._audit("maker_start", opp, size=size)

        # ----- Rest the maker (post-only, self-expiring) -----
        # Post one tick INSIDE the ask (a cheaper bid) so it rests instead of crossing
        # (post-only rejects a marketable price). This also captures an extra tick of
        # edge: our cost is maker_px < the quoted ask, so the locked edge only grows.
        maker_px = max(0.01, round(maker[3] - self.maker_improvement, 4))
        try:
            m = await maker_venue.place_order(
                maker[1], maker[2], "buy", maker_px, size,
                tif="gtc", post_only=True, expiration_ts=int(time.time() + self.maker_timeout))
        except Exception as exc:
            return ExecutionReport(ExecStatus.SKIPPED, f"maker placement failed: {exc}")
        if m.status is OrderStatus.REJECTED:   # post-only would have crossed -> no position
            return ExecutionReport(
                ExecStatus.SKIPPED, f"maker would cross / rejected ({_reject_reason(m)})", [m])
        if m.status is OrderStatus.ERROR:
            left = await self._position_after_error(maker_venue, maker[1])
            if left is False:
                return ExecutionReport(ExecStatus.SKIPPED, f"maker ERROR but flat ({_reject_reason(m)})")
            return self._halt(f"maker ERROR — fill state unknown ({_reject_reason(m)})", [m])
        log.info("maker %s", m)

        # ----- Wait for the maker to fill (per-order, via the private WS) or expire -----
        filled = m.filled
        avg = m.avg_price
        if m.status is OrderStatus.RESTING and m.order_id:
            if self.maker_poll > 0:
                # Watch the taker while resting: cancel the maker if the taker drifts so the
                # hedge could no longer lock the floor — don't let it fill into a loss.
                filled, avg = await self._rest_with_drift_guard(
                    m, maker, taker, maker_venue, taker_venue, maker_px, size, floor)
            else:
                # No drift polling: wait up to the maker timeout for a fill, then CANCEL
                # (a GOOD_TILL_CANCEL maker won't self-expire) BEFORE the authoritative
                # confirm — so an unfilled maker can't rest unhedged, and a fill racing the
                # cancel is still caught by the confirm (which runs slightly past the timeout).
                confirm = asyncio.ensure_future(
                    self.fill_confirmer.confirm(m.venue, m.order_id, size, self.maker_timeout + 1.5))
                try:
                    await asyncio.wait({confirm}, timeout=self.maker_timeout)
                finally:
                    if not confirm.done():
                        await self._cancel_maker(maker_venue, m.order_id)
                _, filled, avg = await confirm
        # AUTHORITATIVE maker fill: the private-fill confirmer can UNDER-report a venue maker
        # partial — a Polymarket maker that filled 17/20 was booked as 0, leaving 17 naked
        # contracts and tripping a RECONCILE HALT. Re-read the order's true filled qty from
        # the venue and trust the larger value, so we never walk away from a real fill.
        fn = getattr(maker_venue, "order_filled_qty", None)
        if fn is not None and m.order_id:
            try:
                true_qty = await fn(m.order_id, size, maker[2])
            except Exception as exc:
                log.warning("maker true-fill read failed for %s: %s", m.order_id, exc)
                true_qty = None
            if true_qty is not None:
                if true_qty > filled + 1e-9:
                    log.warning("maker fill UNDER-REPORTED by confirmer (%g) — venue shows %g; "
                                "hedging the true fill (would have leaked naked)", filled, true_qty)
                filled = max(filled, true_qty)
            elif filled <= 1e-9:
                # Confirmer says unfilled AND the venue (which supports the read) couldn't be
                # reached -> ambiguous. Fail closed: halt rather than guess 0 and leak naked.
                return self._halt(
                    "maker fill state unreadable after cancel (confirmer 0, venue read "
                    "failed) — manual reconcile", [m])
        if filled <= 1e-9:
            return ExecutionReport(
                ExecStatus.SKIPPED, "maker unfilled — expired/cancelled, no trade", [m])

        # ----- Maker filled -> take the deep hedge immediately (IOC) -----
        maker_leg = OrderResult(
            venue=m.venue, market_id=maker[1], side=maker[2], action="buy",
            requested=filled, filled=filled,
            avg_price=avg if avg is not None else maker[3],
            order_id=m.order_id, status=OrderStatus.FILLED)
        # Re-fetch the hedge venue's LIVE book and cross the current ask: the maker may
        # have rested for seconds, so the opp's price is stale — a stale limit misses
        # and forces an unwind. We're committed once the maker filled, so cross to fill.
        taker_px = taker[3]
        try:
            fresh = await taker_venue.fetch_quote(
                RawMarket(market_id=taker[1], title="", raw={}))
        except Exception as exc:
            log.warning("hedge book refetch failed for %s: %s", taker[1], exc)
            fresh = None
        if fresh is not None:
            live_ask = fresh.yes_ask if taker[2] is Side.YES else fresh.no_ask
            if live_ask is not None:
                taker_px = live_ask
        # The maker has already filled — we hold a one-sided position, so we MUST hedge
        # (holding it naked is directional risk, not arbitrage). But if the taker drifted
        # past break-even while the maker rested, this hedge locks a guaranteed loss. Bound
        # it (hedging caps the loss; naked does not) but log it loudly — a recurring forced
        # loss means the arming cushion (maker_arm_cushion) is too small for the drift.
        maker_cost = maker_leg.avg_price if maker_leg.avg_price is not None else maker[3]
        if maker_cost + taker_px >= 1.0:
            log.warning(
                "maker FORCED-LOSS hedge on %s: maker filled %s@%.3f, taker ask now %.3f "
                "(combined %.3f > 1) — hedging to bound the loss; raise EXEC_MAKER_ARM_CUSHION",
                opp.event_key, maker[2].value, maker_cost, taker_px, maker_cost + taker_px)
        taker_limit = min(0.99, round(taker_px + self.hedge_buffer, 4))
        attempt = 0
        hedged = 0.0                 # contracts of the hedge confirmed filled so far
        hedge_notional = 0.0         # sum(filled_i * price_i) -> blended hedge avg price
        while True:
            need = round(filled - hedged, 6)
            hedge = await self._place(
                taker_venue, taker[1], taker[2], "buy", taker_limit, need, "immediate_or_cancel")
            log.info("maker-hedge %s", hedge)

            if hedge.status is OrderStatus.ERROR:
                # ----- hedge ERROR (ambiguous: e.g. a Polymarket 500/timeout) -----
                # We already HOLD the maker fill, so holding it naked into settlement is the
                # dangerous state. Reconcile the hedge venue for the TRUE open hedge quantity
                # (rules out a double-up on retry) and fold it into ``hedged``; only a venue
                # we can't read at all halts for manual reconciliation.
                qty = await self._hedge_qty_after_error(taker_venue, taker[1])
                if qty is None:
                    return self._halt(
                        f"maker hedge ambiguous (ERROR: {_reject_reason(hedge)}; "
                        f"hedge qty unreadable) — manual reconcile", [maker_leg, hedge])
                if qty > hedged:     # newly-landed contracts, priced at the limit we sent
                    hedge_notional += (qty - hedged) * (
                        hedge.avg_price if hedge.avg_price is not None else taker_limit)
                hedged = qty
                log.warning("maker hedge ERROR; reconciled %s open=%g of %g (%s)",
                            taker[1], hedged, filled, _reject_reason(hedge))
            elif hedge.left_a_position:
                # FILLED (full or partial) — a KNOWN fill amount we can trust.
                hedged += hedge.filled
                hedge_notional += hedge.filled * (
                    hedge.avg_price if hedge.avg_price is not None else taker_limit)

            # ----- shared decision: fully hedged -> settle; else re-cross or unwind -----
            if hedged >= filled - 1e-9:
                combined = replace(
                    hedge, side=taker[2], action="buy", requested=filled, filled=hedged,
                    status=OrderStatus.FILLED,
                    avg_price=(hedge_notional / hedged) if hedged > 1e-9 else taker_limit)
                return self._settle_success(opp, filled, [maker_leg, combined])

            if attempt < self.hedge_retries:
                # Not fully hedged, but the unfilled remainder is reconciled flat (no
                # double-up risk): RE-CROSS it at a fresh, current ask instead of giving up.
                # Most KILLED/PARTIAL hedges are transient thinning at our limit — a re-cross
                # usually locks the arb rather than eating the unwind spread.
                attempt += 1
                log.warning(
                    "maker hedge incomplete (%g/%g) — re-crossing remainder (%d/%d) (%s: %s)",
                    hedged, filled, attempt, self.hedge_retries,
                    hedge.status.value, _reject_reason(hedge))
                await asyncio.sleep(0.25)
                try:
                    fresh = await taker_venue.fetch_quote(
                        RawMarket(market_id=taker[1], title="", raw={}))
                    live = fresh.yes_ask if taker[2] is Side.YES else fresh.no_ask
                    if live is not None:
                        taker_limit = min(0.99, round(live + self._hedge_buffer_for(need), 4))
                except Exception as exc:
                    log.warning("hedge re-cross book refetch failed for %s: %s", taker[1], exc)
                continue

            # ----- retries exhausted: settle what's hedged, UNWIND the unhedged remainder -----
            # Ending flat-or-hedged always beats halting with a naked leg: the matched
            # ``hedged`` portion is a locked arb; the maker's unhedged excess is sold back.
            return await self._settle_partial_hedge(
                opp, maker_leg, filled, hedged, hedge_notional, hedge,
                reason=f"maker hedge incomplete ({hedged:g}/{filled:g}: {_reject_reason(hedge)})")

    # ---- outcomes ----
    def _settle_success(self, opp, size, legs) -> ExecutionReport:
        # Identify the legs by side (the placement order may put the NO leg first).
        yes_leg = next(leg for leg in legs if leg.side is Side.YES)
        no_leg = next(leg for leg in legs if leg.side is Side.NO)
        ya = yes_leg.avg_price if yes_leg.avg_price is not None else opp.yes_price
        na = no_leg.avg_price if no_leg.avg_price is not None else opp.no_price
        fees = self._fee(yes_leg.venue).fee(ya, size) + self._fee(no_leg.venue).fee(na, size)
        pnl = size * (1.0 - ya - na) - fees

        self.risk.record_fill(f"{yes_leg.venue}:{yes_leg.market_id}", ya * size)
        self.risk.record_pnl(pnl)
        # Decrement tracked cash so the next arb sizes against what's actually left.
        self._spend(yes_leg.venue, ya * size)
        self._spend(no_leg.venue, na * size)
        if self.store is not None:
            self.store.record_fill(yes_leg.venue, yes_leg.market_id, "YES", ya, size)
            self.store.record_fill(no_leg.venue, no_leg.market_id, "NO", na, size)
            self.store.record_pnl(pnl, note="arb locked")
            self.store.record_opportunity(opp, acted=True)
        self._audit("execute_success", opp, pnl=pnl)
        log.info("ARB LOCKED %s | pnl=%+.2f", opp.event_key, pnl)
        return ExecutionReport(ExecStatus.SUCCESS, "both legs filled", list(legs), pnl)

    async def _settle_partial_hedge(
        self, opp, maker_leg, maker_filled, hedged, hedge_notional, last_hedge, *, reason,
    ) -> ExecutionReport:
        """Couldn't fully hedge a maker fill after re-crossing. End flat-or-locked, never
        naked: SETTLE the matched (``hedged``) portion as a locked arb, then UNWIND the
        maker's unhedged excess (``maker_filled - hedged``). Returns UNWOUND with the
        combined pnl; if the unwind itself can't flatten, that path halts (correct — we
        genuinely can't get flat). The hedge leg has no excess: it's fully matched."""
        excess = round(maker_filled - hedged, 6)
        legs: list[OrderResult] = []
        pnl = 0.0
        if hedged > 1e-9:
            matched_hedge = replace(
                last_hedge, action="buy", requested=hedged, filled=hedged,
                status=OrderStatus.FILLED, avg_price=hedge_notional / hedged)
            matched_maker = replace(maker_leg, requested=hedged, filled=hedged)
            settled = self._settle_success(opp, hedged, [matched_maker, matched_hedge])
            legs += settled.legs
            pnl += settled.realized_pnl
        if excess > 1e-9:
            excess_leg = replace(maker_leg, requested=excess, filled=excess)
            unwound = await self._unwind(opp, excess_leg, None, reason=reason)
            if unwound.status is ExecStatus.HALTED:
                return unwound          # couldn't flatten the excess -> already halted
            legs += unwound.legs
            pnl += unwound.realized_pnl
        log.warning("PARTIAL-HEDGE %s: locked %g + unwound %g | pnl=%+.2f (%s)",
                    opp.event_key, hedged, excess, pnl, reason)
        return ExecutionReport(ExecStatus.UNWOUND, reason, legs, pnl)

    async def _unwind(self, opp, leg1, leg2, *, reason: str = "leg2 failed") -> ExecutionReport:
        """Sell the filled first leg back (IOC, slippage-tolerant) to return to flat.
        Used when the second leg cleanly fails and when the first leg partial-fills
        (``leg2`` is then ``None``). Works for whichever side leg 1 bought (the
        rejection-prone leg may be the NO leg when Kalshi is the NO venue)."""
        venue = self.venues[leg1.venue]
        side = leg1.side
        ref_price = opp.yes_price if side is Side.YES else opp.no_price
        buy_px = leg1.avg_price if leg1.avg_price is not None else ref_price
        # Cross the REAL best bid for that side so the IOC actually fills (a fixed haircut
        # off the buy price can sit above a thin book's bid and never fill -> stuck naked
        # + halt). YES bid = 1 - no_ask; NO bid = 1 - yes_ask. Floor at $0.01; fall back
        # to the haircut only if the book can't be read.
        sell_px = max(0.01, round(buy_px - self.unwind_slippage, 4))
        try:
            q = await venue.fetch_quote(RawMarket(market_id=leg1.market_id, title="", raw={}))
        except Exception as exc:
            log.warning("unwind book fetch failed for %s: %s", leg1.market_id, exc)
            q = None
        if q is not None:
            bid = None
            if side is Side.YES and q.no_ask is not None:
                bid = round(1.0 - q.no_ask, 4)
            elif side is Side.NO and q.yes_ask is not None:
                bid = round(1.0 - q.yes_ask, 4)
            if bid is not None:
                sell_px = max(0.01, min(sell_px, bid))    # take the bid, never above it
        unwind = await self._place(
            venue, leg1.market_id, side, "sell", sell_px, leg1.filled, "immediate_or_cancel"
        )
        log.warning("unwind %s", unwind)

        legs = [leg1] + ([leg2] if leg2 is not None else []) + [unwind]
        if not unwind.filled_fully:
            # Couldn't flatten -> we're still holding a one-sided position. This market is
            # illiquid enough that BOTH the hedge AND the unwind rejected — AUTO-BLACKLIST the
            # pair so the bot never re-fires it (it just stranded a naked leg) and a restart
            # comes up clean (the pair drops the watchlist -> reconcile treats the stuck leg as
            # benign untracked, not a re-halt). Then stop, surfacing both failure reasons.
            blacklisted = False
            if self.store is not None:
                try:
                    self.store.blacklist_pair(
                        opp.buy_yes_venue, opp.buy_yes_market,
                        opp.buy_no_venue, opp.buy_no_market,
                        reason=f"unwind failed (stuck naked leg): {reason}")
                    blacklisted = True
                    log.warning("auto-blacklisted %s after unwind failure (illiquid: hedge "
                                "AND unwind both rejected)", opp.event_key)
                except Exception as exc:
                    log.warning("auto-blacklist failed for %s: %s", opp.event_key, exc)
            # If we quarantined the market (blacklisted), the stuck leg is KNOWN, bounded, and
            # won't re-fire -> KEEP trading the rest of the book instead of freezing on it.
            # If blacklisting failed, fall back to a hard halt (fail closed).
            return self._halt(
                f"UNWIND FAILED ({reason}; unwind {unwind.status.value}: "
                f"{_reject_reason(unwind)}) — leg1 stranded, market "
                f"{'quarantined' if blacklisted else 'NOT blacklisted'}; manual reconcile",
                legs, trip=not blacklisted,
            )

        sell_avg = unwind.avg_price if unwind.avg_price is not None else sell_px
        pnl = leg1.filled * (sell_avg - buy_px) - self._fee(leg1.venue).fee(buy_px, leg1.filled)
        self.risk.record_pnl(pnl)
        # Net cash effect of buying leg1 then selling it back (a small loss).
        self._spend(leg1.venue, (buy_px - sell_avg) * leg1.filled)
        if self.store is not None:
            self.store.record_fill(leg1.venue, leg1.market_id, side.value, buy_px, leg1.filled)
            self.store.record_fill(
                unwind.venue, unwind.market_id, f"{side.value}_SELL", sell_avg, unwind.filled)
            self.store.record_pnl(pnl, note=f"unwind ({reason})")
        self._audit("execute_unwound", opp, pnl=pnl)
        log.warning("UNWOUND %s (%s) | pnl=%+.2f", opp.event_key, reason, pnl)
        return ExecutionReport(ExecStatus.UNWOUND, f"{reason}; leg1 unwound", legs, pnl)

    def _halt(self, reason: str, legs: list[OrderResult], *, trip: bool = True) -> ExecutionReport:
        # ``trip=True`` (default): ambiguous/unknown state -> trip the global kill switch and
        # stop everything (fail closed). ``trip=False`` (QUARANTINE): the stuck leg is KNOWN
        # and bounded and its market is already blacklisted, so record it but KEEP trading the
        # rest of the book — freezing the whole bot for a tiny stranded thin-prop leg is
        # disproportionate, and the reconcile still catches any unexpected/larger naked.
        if trip:
            self.risk.trip_kill_switch(f"executor halt: {reason}")
        # Record what we were HOLDING so the stuck state isn't invisible in the books. The
        # SETTLED pnl is unknown — but the fills are factual and the CASH that moved is real.
        # Record each filled leg + a PROVISIONAL pnl row (the net cash outflow, conservatively
        # treating held legs as not-yet-recovered); the operator reconciles vs settlement.
        cash = 0.0
        for leg in legs or []:
            if leg is None or leg.filled <= 1e-9:
                continue
            price = leg.avg_price if leg.avg_price is not None else 0.0
            is_sell = getattr(leg, "action", "buy") == "sell"
            side = f"{leg.side.value}_SELL" if is_sell else leg.side.value
            cash += (price * leg.filled) if is_sell else -(price * leg.filled)
            if self.store is not None:
                self.store.record_fill(leg.venue, leg.market_id, side, price, leg.filled)
        tag = "HALT" if trip else "QUARANTINE"
        if self.store is not None:
            self.store.record_pnl(cash, note=f"{tag} provisional, unreconciled ({reason})")
            self.store.audit("execute_halt" if trip else "execute_quarantine",
                             {"reason": reason, "cash": round(cash, 4),
                              "legs": [str(leg) for leg in legs]})
        log.critical("%s: %s | provisional cash %+.2f recorded for reconciliation",
                     tag, reason, cash)
        return ExecutionReport(ExecStatus.HALTED if trip else ExecStatus.QUARANTINED,
                               reason, legs, cash)

    def _audit(self, kind: str, opp: ArbOpportunity, **extra) -> None:
        if self.store is not None:
            self.store.audit(kind, {"event_key": opp.event_key, "opp": str(opp), **extra})
