"""Normalized order result, shared across venues.

The executor reasons about fills in these venue-agnostic terms; each adapter maps
its raw create-order response into an :class:`OrderResult`. FoK (fill-or-kill) is the
default for arbitrage legs, so the practical outcomes are FILLED or KILLED — but the
type also models PARTIAL/REJECTED/ERROR defensively.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from enum import Enum

from bot.models import Side


class OrderStatus(str, Enum):
    FILLED = "FILLED"        # fully filled
    KILLED = "KILLED"        # FoK/IOC, nothing filled (no resulting position)
    PARTIAL = "PARTIAL"      # partially filled (shouldn't happen with FoK)
    REJECTED = "REJECTED"    # exchange rejected
    RESTING = "RESTING"      # maker order accepted and resting on the book (not yet filled)
    ERROR = "ERROR"          # network/parse error — treat position as UNKNOWN


@dataclass
class OrderResult:
    venue: str
    market_id: str
    side: Side
    action: str                 # "buy" or "sell"
    requested: float            # contracts requested
    filled: float = 0.0         # contracts filled
    avg_price: float | None = None
    order_id: str | None = None
    status: OrderStatus = OrderStatus.ERROR
    raw: dict = field(default_factory=dict)

    @property
    def filled_fully(self) -> bool:
        return self.status is OrderStatus.FILLED or self.filled >= self.requested - 1e-9

    @property
    def left_a_position(self) -> bool:
        """True if this order resulted in any position (so a leg failure must unwind it)."""
        return self.filled > 1e-9

    @property
    def notional(self) -> float:
        return (self.avg_price or 0.0) * self.filled

    def __str__(self) -> str:
        return (
            f"{self.action.upper()} {self.side.value} {self.venue}:{self.market_id} "
            f"{self.filled:g}/{self.requested:g}@{self.avg_price if self.avg_price is not None else '?'} "
            f"[{self.status.value}]"
        )


def order_error_result(venue, market_id, side, action, contracts, exc) -> "OrderResult":
    """Map a place-order exception to an :class:`OrderResult`, distinguishing a
    definitive rejection from a truly ambiguous failure.

    A server HTTP 4xx (e.g. 409 Conflict, 400, 422) means the order was NOT created —
    no position resulted — so it is REJECTED: the executor aborts that one trade
    cleanly (and can safely unwind the other leg) without tripping the kill switch.
    Anything else — network error, timeout, 5xx (the request may or may not have been
    processed) — is ERROR, which keeps the conservative "halt on unknown fill" rule.
    """
    resp = getattr(exc, "response", None)
    code = getattr(resp, "status_code", None)
    body = None
    if resp is not None:
        with contextlib.suppress(Exception):
            body = resp.text[:500]
    if isinstance(code, int) and 400 <= code < 500:
        return OrderResult(venue, market_id, side, action, contracts,
                           status=OrderStatus.REJECTED,
                           raw={"http_status": code, "body": body, "error": str(exc)})
    # Network error, timeout, or 5xx (the request may or may not have been processed) ->
    # ERROR (unknown fill state). Carry the HTTP status + server body when present: a 500
    # often returns a diagnostic message that says WHY the order was refused.
    return OrderResult(venue, market_id, side, action, contracts,
                       status=OrderStatus.ERROR,
                       raw={"http_status": code, "body": body, "error": str(exc)})
