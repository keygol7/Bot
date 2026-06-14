"""Common, venue-agnostic data models.

Every venue adapter normalizes its raw payloads into these types so the rest of
the bot never sees venue-specific shapes. Prices are expressed in dollars in the
closed interval [0, 1] — the price of a binary YES/NO contract that pays $1 if it
resolves in your favor. Sizes are in contracts.

Standard-library only (dataclasses) so the deterministic core stays import-light
and testable without third-party packages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Side(str, Enum):
    YES = "YES"
    NO = "NO"

    @property
    def opposite(self) -> "Side":
        return Side.NO if self is Side.YES else Side.YES


@dataclass(frozen=True)
class PriceLevel:
    """One level of an order book. `price` in [0, 1] dollars, `size` in contracts."""

    price: float
    size: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.price <= 1.0):
            raise ValueError(f"price must be in [0, 1], got {self.price}")
        if self.size < 0:
            raise ValueError(f"size must be >= 0, got {self.size}")


@dataclass
class MarketQuote:
    """A normalized snapshot of the takeable top-of-book for one binary market.

    `yes_ask` is the cost to *buy* one YES contract; `no_ask` the cost to *buy*
    one NO contract. Either may be ``None`` when that side has no resting
    liquidity. `event_key` is the cross-venue identifier the matcher assigns so
    quotes for the same real-world event line up across venues.
    """

    venue: str
    market_id: str
    title: str
    event_key: Optional[str] = None
    yes_ask: Optional[float] = None
    yes_ask_size: float = 0.0
    no_ask: Optional[float] = None
    no_ask_size: float = 0.0
    fee_rate: float = 0.0  # carried for reference; FeeModel does the real math
    timestamp: float = 0.0
    close_time: Optional[float] = None  # epoch seconds when the market resolves/closes

    @property
    def label(self) -> str:
        return f"{self.venue}:{self.market_id}"
