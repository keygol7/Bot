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
        min_lock_edge: float | None = None,
        leg2_slippage_share: float = 0.6,
        min_leg_depth: float = 0.0,
        depth_safety: float = 1.0,
        first_venue: str = "kalshi",
        hedge_buffer: float = 0.0,
        maker_timeout: float = 5.0,
        maker_improvement: float = 0.01,
        maker_arm_cushion: float = 0.0,
        maker_poll: float = 0.0,
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
        self.fill_confirmer = fill_confirmer       # optional FillTracker (private WS)
        self.confirm_timeout = confirm_timeout
        # Available cash per venue, seeded from the startup snapshot and decremented as
        # legs fill. Used to size each arb to the most the funded balances allow.
        self.balance_buffer = balance_buffer       # leave headroom for fees/slippage
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
        return result

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

    async def _hedge_would_fill(self, venue, leg, size) -> bool:
        """Preview the hedge leg (when the venue supports it) to confirm it would fully
        fill at its limit before we commit leg 1. Returns True to proceed (fills, OR no
        preview support / a preview failure -> don't block on a best-effort check), False
        only when the venue affirmatively says it would NOT fully fill."""
        preview = getattr(venue, "preview_order", None)
        if preview is None:
            return True
        try:
            res = await preview(leg[1], leg[2], "buy", leg[3], size, tif="fill_or_kill")
        except Exception as exc:
            log.warning("hedge preview failed for %s: %s", leg[1], exc)
            return True
        if getattr(res, "status", None) is OrderStatus.ERROR:
            return True                          # couldn't preview -> proceed (best effort)
        return res.filled >= size - 1e-9

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
        if yb is not None and opp.yes_price > 0:
            caps["cash_yes"] = (yb * self.balance_buffer) / opp.yes_price
        if nb is not None and opp.no_price > 0:
            caps["cash_no"] = (nb * self.balance_buffer) / opp.no_price

        gross = opp.gross_cost
        if gross > 0:
            lim = self.risk.limits
            caps["per_market"] = max(0.0, lim.max_position_per_market - self.risk.position(label)) / gross
            caps["total"] = max(0.0, lim.max_total_exposure - self.risk.total_exposure) / gross

        if self.max_order_contracts and self.max_order_contracts > 0:
            caps["order_cap"] = float(self.max_order_contracts)

        return math.floor(max(0.0, min(caps.values()))), caps

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
        hedge = min(self.hedge_buffer, surplus)
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

        # Order the two legs so the rejection-prone venue (self.first_venue, e.g. Kalshi)
        # goes FIRST: if it rejects, no other leg was taken -> a clean skip, no unwind.
        # The SECOND (hedge) leg is the one whose failure forces an unwind.
        if opp.buy_no_venue == self.first_venue and opp.buy_yes_venue != self.first_venue:
            first_vn, first_m, first_side = opp.buy_no_venue, opp.buy_no_market, Side.NO
            second_vn, second_m, second_side = opp.buy_yes_venue, opp.buy_yes_market, Side.YES
        else:
            first_vn, first_m, first_side = opp.buy_yes_venue, opp.buy_yes_market, Side.YES
            second_vn, second_m, second_side = opp.buy_no_venue, opp.buy_no_market, Side.NO

        # Don't fire an edge too thin to give the hedge leg a fill cushion — it would
        # just unwind. We only fire when the edge can pay the hedge buffer AND still lock
        # the floor, so the hedge fills through normal book movement (no unwind).
        floor = self.min_lock_edge if self.min_lock_edge is not None else self.risk.limits.min_edge
        if opp.edge_per_contract < floor + self.hedge_buffer - 1e-9:
            return ExecutionReport(
                ExecStatus.SKIPPED,
                f"edge {opp.edge_per_contract:.3f} < lock {floor:.3f} + hedge "
                f"{self.hedge_buffer:.3f} — would risk an unwind",
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

        # Preview the hedge (deep) leg before committing leg 1: if it would NOT fully fill
        # (phantom/evaporated depth), skip the whole trade — no leg placed, no naked risk,
        # and we never fire a synchronous FOK into empty liquidity (the 500 trigger). Only
        # acts when the hedge venue supports preview; a preview failure proceeds as before.
        if not await self._hedge_would_fill(second_venue, second, size):
            self._audit("execute_skip_preview", opp, size=size)
            log.info("STREAM skip %s — hedge preview: leg2 would not fully fill %g",
                     opp.event_key, size)
            return ExecutionReport(
                ExecStatus.SKIPPED, "hedge preview: leg2 would not fully fill")

        self._audit("execute_start", opp, size=size)

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
                    try:
                        await maker_venue.cancel_order(m.order_id)
                    except Exception as exc:
                        log.warning("maker cancel failed for %s: %s", m.order_id, exc)
                    break
        except asyncio.CancelledError:
            confirm.cancel()
            raise
        # Whether we cancelled or not, the confirmation is authoritative: a cancel that
        # raced a fill still reports the real filled amount (which we then hedge).
        _, filled, avg = await confirm
        return filled, avg

    async def execute_maker(self, opp: ArbOpportunity) -> ExecutionReport:
        """Capture an edge by RESTING the fee-heavy leg (``first_venue``, e.g. Kalshi)
        as a maker — no slippage, lower fee — then TAKING the deep leg (Polymarket) the
        instant the maker fills. Lets THIN edges be captured (we capture the spread
        instead of paying it). The maker self-expires if unfilled, so an uncrossed quote
        is a clean no-trade. The naked window is just the maker-fill -> taker round trip.
        """
        if self.risk.is_killed:
            return ExecutionReport(ExecStatus.SKIPPED, f"kill switch: {self.risk.kill_reason}")
        if self.fill_confirmer is None:
            return ExecutionReport(ExecStatus.SKIPPED, "maker mode needs a fill confirmer")

        # Maker leg = the first_venue (fee-heavy) side; taker = the deep, cheap side. The
        # HEDGE (taker) leg's depth is what binds in maker mode: the resting maker ADDS
        # liquidity (its own book depth doesn't constrain us), but we must cross the taker
        # to hedge a fill. So guard/size on the taker leg's own depth, not min(both legs).
        if opp.buy_no_venue == self.first_venue:
            maker = (opp.buy_no_venue, opp.buy_no_market, Side.NO, opp.no_price)
            taker = (opp.buy_yes_venue, opp.buy_yes_market, Side.YES, opp.yes_price)
            hedge_depth = opp.yes_size or opp.max_contracts
        elif opp.buy_yes_venue == self.first_venue:
            maker = (opp.buy_yes_venue, opp.buy_yes_market, Side.YES, opp.yes_price)
            taker = (opp.buy_no_venue, opp.buy_no_market, Side.NO, opp.no_price)
            hedge_depth = opp.no_size or opp.max_contracts
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
                # Confirm timeout slightly past the maker's expiry, so by the time it
                # returns the order is terminal and ``filled`` is final.
                _, filled, avg = await self.fill_confirmer.confirm(
                    m.venue, m.order_id, size, self.maker_timeout + 1.5)
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
        hedge = await self._place(
            taker_venue, taker[1], taker[2], "buy", taker_limit, filled, "immediate_or_cancel")
        log.info("maker-hedge %s", hedge)
        if hedge.status is OrderStatus.FILLED and hedge.filled_fully:
            return self._settle_success(opp, filled, [maker_leg, hedge])
        if hedge.status in (OrderStatus.KILLED, OrderStatus.REJECTED) and hedge.filled <= 1e-9:
            log.warning("maker hedge %s — %s", hedge.status.value, _reject_reason(hedge))
            return await self._unwind(
                opp, maker_leg, hedge, reason=f"maker hedge {hedge.status.value} ({_reject_reason(hedge)})")
        return self._halt(
            f"maker hedge ambiguous ({hedge.status.value}: {_reject_reason(hedge)})",
            [maker_leg, hedge])

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
            # Couldn't flatten -> we're still holding a one-sided position. Stop
            # everything, surfacing both WHY we were unwinding and why the unwind failed.
            return self._halt(
                f"UNWIND FAILED ({reason}; unwind {unwind.status.value}: "
                f"{_reject_reason(unwind)}) — still holding leg1, manual action required",
                legs,
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

    def _halt(self, reason: str, legs: list[OrderResult]) -> ExecutionReport:
        self.risk.trip_kill_switch(f"executor halt: {reason}")
        if self.store is not None:
            self.store.audit("execute_halt", {"reason": reason, "legs": [str(leg) for leg in legs]})
        log.critical("HALT: %s", reason)
        return ExecutionReport(ExecStatus.HALTED, reason, legs)

    def _audit(self, kind: str, opp: ArbOpportunity, **extra) -> None:
        if self.store is not None:
            self.store.audit(kind, {"event_key": opp.event_key, "opp": str(opp), **extra})
