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

import logging
import math
from dataclasses import dataclass, field
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
        # (defaults to the risk min_edge floor), and how the spendable surplus is split
        # toward the completing NO leg (which we most want to fill once leg 1 commits us).
        self.min_lock_edge = min_lock_edge
        self.leg2_slippage_share = min(max(leg2_slippage_share, 0.0), 1.0)

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

    def _max_size(self, opp: ArbOpportunity) -> tuple[int, dict[str, float]]:
        """Largest whole-contract size that fits every hard limit at once:

          * available order-book depth (can't fill more than is quoted),
          * funded cash on each leg's venue (YES leg needs cash on the YES venue at
            ``yes_price``; NO leg needs cash on the NO venue at ``no_price``),
          * the per-market and total exposure risk caps,
          * the optional per-order contract ceiling.

        Returns ``(size, caps)`` where ``caps`` is each constraint's contract limit
        (for logging why a size was chosen).
        """
        label = f"{opp.buy_yes_venue}:{opp.buy_yes_market}"
        # Trade only a fraction of shown depth (headroom for a thinning book on FOK).
        caps: dict[str, float] = {"depth": float(opp.max_contracts) * self.depth_safety}

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

        return int(math.floor(max(0.0, min(caps.values())))), caps

    def _aggressive_limits(self, opp: ArbOpportunity) -> tuple[float, float]:
        """Per-leg limit prices that may pay worse than the quoted ask to fill on a
        moving book, but never enough to drop the locked edge below the floor.

        The detected edge above the floor is the ``surplus`` we can spend on slippage;
        it is split between the legs (favoring the completing NO leg). Worst case both
        legs fill at their limits -> combined cost = gross + surplus = 1 - floor, so the
        locked profit is >= floor by construction. Thin edges get ~no room (stay
        conservative); fat edges get room to chase the fill."""
        floor = self.min_lock_edge if self.min_lock_edge is not None else self.risk.limits.min_edge
        surplus = max(0.0, opp.edge_per_contract - floor)
        no_buf = surplus * self.leg2_slippage_share
        yes_buf = surplus - no_buf
        yes_limit = min(0.99, round(opp.yes_price + yes_buf, 4))
        no_limit = min(0.99, round(opp.no_price + no_buf, 4))
        return yes_limit, no_limit

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

        yes_limit, no_limit = self._aggressive_limits(opp)
        notional = size * (yes_limit + no_limit)   # worst-case cost for the risk check
        label = f"{opp.buy_yes_venue}:{opp.buy_yes_market}"
        decision = self.risk.check(label, notional)
        if not decision.allowed:
            return ExecutionReport(ExecStatus.SKIPPED, f"risk: {decision.reason}")

        # Order the two legs so the rejection-prone venue (self.first_venue) goes FIRST:
        # if it rejects, there is no other leg to unwind (a clean skip, $0). Each leg is
        # (venue_name, market, side, limit). Cross-venue, so one leg is on each venue.
        yes_leg = (opp.buy_yes_venue, opp.buy_yes_market, Side.YES, yes_limit)
        no_leg = (opp.buy_no_venue, opp.buy_no_market, Side.NO, no_limit)
        if no_leg[0] == self.first_venue and yes_leg[0] != self.first_venue:
            first, second = no_leg, yes_leg
        else:
            first, second = yes_leg, no_leg

        first_venue = self.venues.get(first[0])
        second_venue = self.venues.get(second[0])
        if first_venue is None or second_venue is None:
            return ExecutionReport(ExecStatus.SKIPPED, "venue not available")

        self._audit("execute_start", opp, size=size)

        # ----- Leg 1: the rejection-prone leg, fill-or-kill -----
        leg1 = await self._place(
            first_venue, first[1], first[2], "buy", first[3], size, "fill_or_kill"
        )
        log.info("leg1 %s", leg1)
        if leg1.status is OrderStatus.ERROR:
            return self._halt(f"leg1 ERROR — fill state unknown ({_reject_reason(leg1)})", [leg1])
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

        # ERROR or PARTIAL on leg 2: we cannot be sure of the hedge state. Do NOT
        # auto-unwind (risk of doubling up). Halt for manual reconciliation.
        return self._halt(
            f"leg2 ambiguous ({leg2.status.value}: {_reject_reason(leg2)}) — manual reconcile",
            [leg1, leg2],
        )

    # ---- outcomes ----
    def _settle_success(self, opp, size, legs) -> ExecutionReport:
        # Identify the legs by side (the placement order may put the NO leg first).
        yes_leg = next(l for l in legs if l.side is Side.YES)
        no_leg = next(l for l in legs if l.side is Side.NO)
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
            self.store.audit("execute_halt", {"reason": reason, "legs": [str(l) for l in legs]})
        log.critical("HALT: %s", reason)
        return ExecutionReport(ExecStatus.HALTED, reason, legs)

    def _audit(self, kind: str, opp: ArbOpportunity, **extra) -> None:
        if self.store is not None:
            self.store.audit(kind, {"event_key": opp.event_key, "opp": str(opp), **extra})
