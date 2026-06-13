"""The venue abstraction every adapter implements.

Keeping a single Protocol means the engine, matcher, and executor are written once
against ``Venue`` and never branch on which exchange they're talking to — which is
also what keeps the venue layer swappable as the regulatory picture shifts.

Read-only methods (``list_markets``, ``stream_order_book``) are what the first
deliverable exercises. The order methods are part of the contract but raise in
DRY_RUN / when unimplemented; live order placement lands in a later phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Protocol, runtime_checkable

from bot.fees import FeeModel
from bot.models import MarketQuote, Side


@dataclass
class RawMarket:
    """A lightly-typed market descriptor returned by ``list_markets``."""

    market_id: str
    title: str
    raw: dict


@runtime_checkable
class Venue(Protocol):
    name: str
    fee_model: FeeModel

    async def list_markets(self, limit: int = 200) -> list[RawMarket]:
        """Fetch open markets (read-only)."""
        ...

    async def stream_order_book(
        self, market_ids: list[str]
    ) -> AsyncIterator[MarketQuote]:
        """Yield normalized top-of-book updates over a WebSocket (read-only)."""
        ...

    async def place_order(
        self, market_id: str, side: Side, price: float, contracts: float
    ) -> dict:
        """Place an order. Not used in DRY_RUN; implemented in the live phase."""
        ...

    async def cancel_order(self, order_id: str) -> dict:
        ...

    async def get_positions(self) -> dict:
        ...


class OrderNotPermitted(RuntimeError):
    """Raised when an order is attempted in a mode/phase that forbids it."""
