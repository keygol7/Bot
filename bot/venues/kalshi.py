"""Kalshi venue adapter (read-only for this phase).

Normalization is the unit-tested core. The networked client imports ``httpx`` /
``websockets`` lazily so importing the ``normalize_*`` helpers needs no extra deps.

Kalshi conventions:
  - Prices are integer **cents**, 1..99. We convert to dollars (price/100).
  - The order book has a ``yes`` array (resting bids to buy YES) and a ``no`` array
    (resting bids to buy NO). To BUY YES you cross the best NO bid, so
    ``yes_ask = 1 - best_no_bid``; to BUY NO you cross the best YES bid, so
    ``no_ask = 1 - best_yes_bid``. Sizes come from the level you'd cross.
  - Kalshi charges a price-dependent fee -> :class:`KalshiFeeModel`.
"""

from __future__ import annotations

from typing import Any, AsyncIterator

from bot.fees import KalshiFeeModel
from bot.models import MarketQuote, PriceLevel, Side
from bot.venues.base import OrderNotPermitted, RawMarket

VENUE = "kalshi"


def _best(levels: list[list[Any]]) -> tuple[float, float] | None:
    """Best (highest-price) bid level from a Kalshi ``[[price_cents, size], ...]``
    array. Returns ``(price_cents, size)`` or ``None`` if empty."""
    if not levels:
        return None
    best = max(levels, key=lambda lvl: lvl[0])
    return float(best[0]), float(best[1])


def normalize_orderbook(
    ticker: str, title: str, orderbook: dict[str, Any], *, event_key: str | None = None
) -> MarketQuote:
    """Normalize a Kalshi ``orderbook`` dict into a :class:`MarketQuote`.

    ``orderbook`` is ``{"yes": [[price_cents, size], ...], "no": [...]}`` (either may
    be missing/empty). ``yes_ask`` is derived by crossing the best NO bid and vice
    versa, per Kalshi's book semantics.
    """
    yes_bids = orderbook.get("yes") or []
    no_bids = orderbook.get("no") or []

    yes_ask = no_ask = None
    yes_ask_size = no_ask_size = 0.0

    best_no = _best(no_bids)
    if best_no is not None:
        price_cents, size = best_no
        yes_ask = round((100.0 - price_cents) / 100.0, 4)
        yes_ask_size = size

    best_yes = _best(yes_bids)
    if best_yes is not None:
        price_cents, size = best_yes
        no_ask = round((100.0 - price_cents) / 100.0, 4)
        no_ask_size = size

    return MarketQuote(
        venue=VENUE,
        market_id=ticker,
        title=title,
        event_key=event_key,
        yes_ask=yes_ask,
        yes_ask_size=yes_ask_size,
        no_ask=no_ask,
        no_ask_size=no_ask_size,
    )


class KalshiVenue:
    """Read-only Kalshi client. ``cfg`` is a ``bot.config.settings.KalshiConfig``."""

    name = VENUE

    def __init__(self, cfg: Any, fee_rate: float = 0.07) -> None:
        self.cfg = cfg
        self.fee_model = KalshiFeeModel(rate=fee_rate)
        self._client = None  # lazy httpx.AsyncClient

    def _http(self):
        if self._client is None:
            import httpx  # lazy

            self._client = httpx.AsyncClient(base_url=self.cfg.api_base, timeout=10.0)
        return self._client

    async def list_markets(self, limit: int = 200) -> list[RawMarket]:
        resp = await self._http().get("/markets", params={"limit": limit, "status": "open"})
        resp.raise_for_status()
        data = resp.json()
        return [
            RawMarket(market_id=m["ticker"], title=m.get("title", ""), raw=m)
            for m in data.get("markets", [])
        ]

    async def fetch_orderbook(self, ticker: str, title: str = "") -> MarketQuote:
        resp = await self._http().get(f"/markets/{ticker}/orderbook")
        resp.raise_for_status()
        ob = resp.json().get("orderbook", {})
        return normalize_orderbook(ticker, title, ob)

    async def stream_order_book(self, market_ids: list[str]) -> AsyncIterator[MarketQuote]:
        # WebSocket streaming lands with the latency hot path; not exercised yet.
        raise NotImplementedError("Kalshi WS streaming is implemented in the live phase")
        yield  # pragma: no cover - makes this an async generator

    async def place_order(self, market_id: str, side: Side, price: float, contracts: float) -> dict:
        raise OrderNotPermitted("order placement is not enabled in this phase (DRY_RUN)")

    async def cancel_order(self, order_id: str) -> dict:
        raise OrderNotPermitted("order placement is not enabled in this phase (DRY_RUN)")

    async def get_positions(self) -> dict:
        raise NotImplementedError("positions endpoint lands with the live phase")

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
