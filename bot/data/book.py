"""In-memory order books for the latency hot path.

The arbitrage detector reads top-of-book from here, never from the database. Each
venue adapter pushes normalized level snapshots in; readers pull best bid/ask and
build :class:`MarketQuote` objects. Standard-library only and intentionally simple:
correctness and low overhead over cleverness.

Prices are YES-contract prices in [0, 1]. The NO ask is derived as ``1 - yes_bid``
when a venue doesn't quote NO directly, but adapters that expose a real NO book
should set it explicitly via :meth:`update`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from bot.models import MarketQuote, PriceLevel


@dataclass
class BookSide:
    """Sorted levels for one side. ``descending`` for bids, ascending for asks."""

    descending: bool
    levels: list[PriceLevel] = field(default_factory=list)

    def replace(self, levels: list[PriceLevel]) -> None:
        self.levels = sorted(
            (lvl for lvl in levels if lvl.size > 0),
            key=lambda lvl: lvl.price,
            reverse=self.descending,
        )

    def best(self) -> PriceLevel | None:
        return self.levels[0] if self.levels else None


@dataclass
class OrderBook:
    venue: str
    market_id: str
    title: str = ""
    event_key: str | None = None
    yes_bids: BookSide = field(default_factory=lambda: BookSide(descending=True))
    yes_asks: BookSide = field(default_factory=lambda: BookSide(descending=False))
    no_asks: BookSide = field(default_factory=lambda: BookSide(descending=False))
    updated_at: float = 0.0

    def best_yes_ask(self) -> PriceLevel | None:
        return self.yes_asks.best()

    def best_yes_bid(self) -> PriceLevel | None:
        return self.yes_bids.best()

    def best_no_ask(self) -> PriceLevel | None:
        """Real NO book if present; otherwise synthesize from the YES bid."""
        direct = self.no_asks.best()
        if direct is not None:
            return direct
        yes_bid = self.yes_bids.best()
        if yes_bid is not None:
            return PriceLevel(price=round(1.0 - yes_bid.price, 6), size=yes_bid.size)
        return None

    def to_quote(self) -> MarketQuote:
        yes_ask = self.best_yes_ask()
        no_ask = self.best_no_ask()
        return MarketQuote(
            venue=self.venue,
            market_id=self.market_id,
            title=self.title,
            event_key=self.event_key,
            yes_ask=yes_ask.price if yes_ask else None,
            yes_ask_size=yes_ask.size if yes_ask else 0.0,
            no_ask=no_ask.price if no_ask else None,
            no_ask_size=no_ask.size if no_ask else 0.0,
            timestamp=self.updated_at,
        )


class BookStore:
    """Keyed collection of live books, ``(venue, market_id) -> OrderBook``."""

    def __init__(self) -> None:
        self._books: dict[tuple[str, str], OrderBook] = {}

    def update(
        self,
        venue: str,
        market_id: str,
        *,
        title: str | None = None,
        event_key: str | None = None,
        yes_bids: list[PriceLevel] | None = None,
        yes_asks: list[PriceLevel] | None = None,
        no_asks: list[PriceLevel] | None = None,
        ts: float | None = None,
    ) -> OrderBook:
        key = (venue, market_id)
        book = self._books.get(key)
        if book is None:
            book = OrderBook(venue=venue, market_id=market_id)
            self._books[key] = book
        if title is not None:
            book.title = title
        if event_key is not None:
            book.event_key = event_key
        if yes_bids is not None:
            book.yes_bids.replace(yes_bids)
        if yes_asks is not None:
            book.yes_asks.replace(yes_asks)
        if no_asks is not None:
            book.no_asks.replace(no_asks)
        book.updated_at = ts if ts is not None else time.time()
        return book

    def get(self, venue: str, market_id: str) -> OrderBook | None:
        return self._books.get((venue, market_id))

    def quotes(self) -> list[MarketQuote]:
        return [b.to_quote() for b in self._books.values()]

    def __len__(self) -> int:
        return len(self._books)
