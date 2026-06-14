"""Real-time fill confirmation from the venues' private WebSocket streams.

A background consumer feeds execution events into a :class:`FillTracker`, which
aggregates per-order fill state. The executor calls ``confirm(...)`` after placing an
order to get the authoritative fill outcome from the stream (with a timeout), instead
of depending on the synchronous REST response shape.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from bot.execution.orders import OrderStatus

# Normalized terminal execution types (venue prefixes stripped).
_TERMINAL = {"FILL", "CANCELED", "REJECTED", "EXPIRED", "DONE_FOR_DAY"}


@dataclass
class FillEvent:
    venue: str
    order_id: str
    exec_type: str           # normalized: FILL | PARTIAL_FILL | CANCELED | REJECTED | ...
    last_shares: float = 0.0
    last_px: float | None = None


class _OrderState:
    def __init__(self) -> None:
        self.filled = 0.0
        self._px_num = 0.0
        self._px_den = 0.0
        self.terminal: str | None = None
        self.cond = asyncio.Condition()

    @property
    def avg_px(self) -> float | None:
        return self._px_num / self._px_den if self._px_den else None


class FillTracker:
    def __init__(self) -> None:
        self._orders: dict[tuple[str, str], _OrderState] = {}

    def _state(self, venue: str, order_id: str) -> _OrderState:
        return self._orders.setdefault((venue, order_id), _OrderState())

    async def apply(self, ev: FillEvent) -> None:
        """Fold one execution event into the order's aggregate state."""
        st = self._state(ev.venue, ev.order_id)
        async with st.cond:
            if ev.last_shares:
                st.filled += ev.last_shares
                if ev.last_px is not None:
                    st._px_num += ev.last_px * ev.last_shares
                    st._px_den += ev.last_shares
            if ev.exec_type in _TERMINAL:
                st.terminal = ev.exec_type
            st.cond.notify_all()

    async def confirm(
        self, venue: str, order_id: str, requested: float, timeout: float = 5.0
    ) -> tuple[OrderStatus, float, float | None]:
        """Wait (up to ``timeout``) for the order to reach a terminal state or fully
        fill, then return ``(status, filled, avg_price)``."""
        st = self._state(venue, order_id)

        def done() -> bool:
            return st.terminal is not None or st.filled >= requested - 1e-9

        async with st.cond:
            try:
                await asyncio.wait_for(st.cond.wait_for(done), timeout)
            except asyncio.TimeoutError:
                pass
            return self._classify(st, requested), st.filled, st.avg_px

    @staticmethod
    def _classify(st: _OrderState, requested: float) -> OrderStatus:
        if st.filled >= requested - 1e-9 and requested > 0:
            return OrderStatus.FILLED
        if st.terminal == "REJECTED" and st.filled <= 1e-9:
            return OrderStatus.REJECTED
        if st.terminal in ("CANCELED", "EXPIRED", "DONE_FOR_DAY") and st.filled <= 1e-9:
            return OrderStatus.KILLED
        if st.filled > 1e-9:
            return OrderStatus.PARTIAL
        return OrderStatus.ERROR   # nothing seen within the timeout -> unknown
