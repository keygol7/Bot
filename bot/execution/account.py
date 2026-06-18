"""Account-state value objects shared by venues and the startup guard.

Kept in a tiny neutral module so venue adapters and the reconciliation guard can
both import them without a circular dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VenuePosition:
    """A non-flat market on a venue: a held contract position and/or resting orders."""

    market_id: str
    quantity: float = 0.0       # signed contracts held (sign = direction)
    resting_orders: int = 0     # open (unfilled) orders on this market

    @property
    def is_open(self) -> bool:
        return abs(self.quantity) > 1e-9 or self.resting_orders > 0


@dataclass
class AccountSnapshot:
    """A venue's reconcilable state at a point in time.

    ``balance is None`` means the balance could not be read (treated as unknown by
    the guard, which fails closed). ``positions`` lists only non-flat markets.
    """

    venue: str
    balance: float | None = None        # USD available to trade
    positions: list[VenuePosition] = field(default_factory=list)

    @property
    def open_positions(self) -> list[VenuePosition]:
        return [p for p in self.positions if p.is_open]
