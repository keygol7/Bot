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

log = logging.getLogger("bot.executor")


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
    ) -> None:
        self.venues = venues
        self.risk = risk
        self.fee_models = fee_models or {}
        self.store = store
        self.max_order_contracts = max_order_contracts
        self.unwind_slippage = unwind_slippage
        self.fill_confirmer = fill_confirmer       # optional FillTracker (private WS)
        self.confirm_timeout = confirm_timeout

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
                result.status, result.filled = status, filled
                # The confirmer is authoritative for status/filled, but its price is
                # raw venue convention (Polymarket reports the YES-side price). Prefer
                # the per-venue-converted price from place_order; only fall back to the
                # confirmer's when place_order returned none.
                if avg is not None and result.avg_price is None:
                    result.avg_price = avg
            except Exception as exc:  # confirmer failure -> keep REST result
                log.warning("fill confirm failed for %s: %s", result.order_id, exc)
        return result

    def _size(self, opp: ArbOpportunity) -> int:
        return int(math.floor(min(self.max_order_contracts, opp.max_contracts)))

    async def execute(self, opp: ArbOpportunity) -> ExecutionReport:
        if self.risk.is_killed:
            return ExecutionReport(ExecStatus.SKIPPED, f"kill switch: {self.risk.kill_reason}")

        size = self._size(opp)
        if size < 1:
            return ExecutionReport(ExecStatus.SKIPPED, "size < 1 contract")

        notional = size * opp.gross_cost
        label = f"{opp.buy_yes_venue}:{opp.buy_yes_market}"
        decision = self.risk.check(label, notional)
        if not decision.allowed:
            return ExecutionReport(ExecStatus.SKIPPED, f"risk: {decision.reason}")

        yes_venue = self.venues.get(opp.buy_yes_venue)
        no_venue = self.venues.get(opp.buy_no_venue)
        if yes_venue is None or no_venue is None:
            return ExecutionReport(ExecStatus.SKIPPED, "venue not available")

        self._audit("execute_start", opp, size=size)

        # ----- Leg 1: buy YES (fill-or-kill) -----
        leg1 = await self._place(
            yes_venue, opp.buy_yes_market, Side.YES, "buy", opp.yes_price, size, "fill_or_kill"
        )
        log.info("leg1 %s", leg1)
        if leg1.status is OrderStatus.ERROR:
            return self._halt("leg1 ERROR — fill state unknown", [leg1])
        if not leg1.left_a_position:
            return ExecutionReport(ExecStatus.SKIPPED, f"leg1 not filled ({leg1.status.value})", [leg1])
        if not leg1.filled_fully:  # PARTIAL on an all-or-nothing leg — shouldn't happen
            return self._halt("leg1 partial fill — unexpected", [leg1])

        # ----- Leg 2: buy NO (fill-or-kill) -----
        leg2 = await self._place(
            no_venue, opp.buy_no_market, Side.NO, "buy", opp.no_price, size, "fill_or_kill"
        )
        log.info("leg2 %s", leg2)

        if leg2.status is OrderStatus.FILLED and leg2.filled_fully:
            return self._settle_success(opp, size, leg1, leg2)

        if leg2.status in (OrderStatus.KILLED, OrderStatus.REJECTED) and leg2.filled <= 1e-9:
            # Definitively no leg-2 position -> safe to unwind leg 1.
            return await self._unwind(opp, leg1, leg2)

        # ERROR or PARTIAL on leg 2: we cannot be sure of the hedge state. Do NOT
        # auto-unwind (risk of doubling up). Halt for manual reconciliation.
        return self._halt(f"leg2 ambiguous ({leg2.status.value}) — manual reconcile", [leg1, leg2])

    # ---- outcomes ----
    def _settle_success(self, opp, size, leg1, leg2) -> ExecutionReport:
        ya = leg1.avg_price if leg1.avg_price is not None else opp.yes_price
        na = leg2.avg_price if leg2.avg_price is not None else opp.no_price
        fees = self._fee(leg1.venue).fee(ya, size) + self._fee(leg2.venue).fee(na, size)
        pnl = size * (1.0 - ya - na) - fees

        self.risk.record_fill(f"{leg1.venue}:{leg1.market_id}", ya * size)
        self.risk.record_pnl(pnl)
        if self.store is not None:
            self.store.record_fill(leg1.venue, leg1.market_id, "YES", ya, size)
            self.store.record_fill(leg2.venue, leg2.market_id, "NO", na, size)
            self.store.record_pnl(pnl, note="arb locked")
            self.store.record_opportunity(opp, acted=True)
        self._audit("execute_success", opp, pnl=pnl)
        log.info("ARB LOCKED %s | pnl=%+.2f", opp.event_key, pnl)
        return ExecutionReport(ExecStatus.SUCCESS, "both legs filled", [leg1, leg2], pnl)

    async def _unwind(self, opp, leg1, leg2) -> ExecutionReport:
        """Sell leg 1 back (IOC, slippage-tolerant) to return to flat."""
        venue = self.venues[leg1.venue]
        buy_px = leg1.avg_price if leg1.avg_price is not None else opp.yes_price
        sell_px = max(0.01, round(buy_px - self.unwind_slippage, 4))  # aggressive to cross
        unwind = await self._place(
            venue, leg1.market_id, Side.YES, "sell", sell_px, leg1.filled, "immediate_or_cancel"
        )
        log.warning("unwind %s", unwind)

        if not unwind.filled_fully:
            # Couldn't flatten -> we're still holding a one-sided position. Stop everything.
            return self._halt(
                "UNWIND FAILED — still holding leg1, manual action required",
                [leg1, leg2, unwind],
            )

        sell_avg = unwind.avg_price if unwind.avg_price is not None else sell_px
        pnl = leg1.filled * (sell_avg - buy_px) - self._fee(leg1.venue).fee(buy_px, leg1.filled)
        self.risk.record_pnl(pnl)
        if self.store is not None:
            self.store.record_fill(leg1.venue, leg1.market_id, "YES", buy_px, leg1.filled)
            self.store.record_fill(unwind.venue, unwind.market_id, "YES_SELL", sell_avg, unwind.filled)
            self.store.record_pnl(pnl, note="leg-failure unwind")
        self._audit("execute_unwound", opp, pnl=pnl)
        log.warning("UNWOUND %s | pnl=%+.2f", opp.event_key, pnl)
        return ExecutionReport(ExecStatus.UNWOUND, "leg2 failed; leg1 unwound", [leg1, leg2, unwind], pnl)

    def _halt(self, reason: str, legs: list[OrderResult]) -> ExecutionReport:
        self.risk.trip_kill_switch(f"executor halt: {reason}")
        if self.store is not None:
            self.store.audit("execute_halt", {"reason": reason, "legs": [str(l) for l in legs]})
        log.critical("HALT: %s", reason)
        return ExecutionReport(ExecStatus.HALTED, reason, legs)

    def _audit(self, kind: str, opp: ArbOpportunity, **extra) -> None:
        if self.store is not None:
            self.store.audit(kind, {"event_key": opp.event_key, "opp": str(opp), **extra})
