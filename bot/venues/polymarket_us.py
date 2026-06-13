"""Polymarket US / QCEX venue adapter (read-only for this phase).

Polymarket runs a CLOB where each binary market has two outcome tokens (YES and NO),
each with its own book. To buy YES you take the best ask on the YES token; to buy NO
you take the best ask on the NO token. Prices are already in dollars (0..1).

Auth for private endpoints uses Ed25519 request signing (``sign_request``), imported
lazily via ``cryptography`` — read-only public reads don't need it, and the
normalization helpers stay dependency-free for testing.

Standard markets are ~zero-fee -> :class:`ZeroFeeModel`.
"""

from __future__ import annotations

import base64
import time
from typing import Any, AsyncIterator

from bot.fees import ZeroFeeModel
from bot.models import MarketQuote, PriceLevel, Side
from bot.venues.base import OrderNotPermitted, RawMarket

VENUE = "polymarket_us"


def normalize_clob_book(book: dict[str, Any]) -> tuple[PriceLevel | None, PriceLevel | None]:
    """Return ``(best_bid, best_ask)`` for one CLOB token book.

    ``book`` is ``{"bids": [{"price": "0.55", "size": "100"}, ...], "asks": [...]}``.
    Bids/asks may be unsorted; we pick the best (highest bid, lowest ask).
    """
    def lvl(entries: list[dict], *, highest: bool) -> PriceLevel | None:
        parsed = [
            PriceLevel(price=float(e["price"]), size=float(e["size"]))
            for e in (entries or [])
            if float(e.get("size", 0)) > 0
        ]
        if not parsed:
            return None
        return max(parsed, key=lambda p: p.price) if highest else min(parsed, key=lambda p: p.price)

    bids = book.get("bids") or []
    asks = book.get("asks") or []
    return lvl(bids, highest=True), lvl(asks, highest=False)


def build_quote(
    market_id: str,
    title: str,
    yes_book: dict[str, Any],
    no_book: dict[str, Any] | None = None,
    *,
    event_key: str | None = None,
) -> MarketQuote:
    """Combine the YES (and optional NO) token books into one :class:`MarketQuote`.

    If the NO book is absent, ``no_ask`` is synthesized from the YES bid
    (``1 - best_yes_bid``) so the detector still has both sides.
    """
    yes_bid, yes_ask = normalize_clob_book(yes_book)

    no_ask_price = no_ask_size = None
    if no_book is not None:
        _, no_ask_lvl = normalize_clob_book(no_book)
        if no_ask_lvl is not None:
            no_ask_price, no_ask_size = no_ask_lvl.price, no_ask_lvl.size
    if no_ask_price is None and yes_bid is not None:
        no_ask_price = round(1.0 - yes_bid.price, 6)
        no_ask_size = yes_bid.size

    return MarketQuote(
        venue=VENUE,
        market_id=market_id,
        title=title,
        event_key=event_key,
        yes_ask=yes_ask.price if yes_ask else None,
        yes_ask_size=yes_ask.size if yes_ask else 0.0,
        no_ask=no_ask_price,
        no_ask_size=no_ask_size or 0.0,
    )


def sign_request(private_key_pem: bytes, message: bytes) -> str:
    """Ed25519-sign ``message`` and return a base64 signature (QCEX private auth).

    Imports ``cryptography`` lazily; only needed for authenticated endpoints.
    """
    from cryptography.hazmat.primitives.serialization import load_pem_private_key

    key = load_pem_private_key(private_key_pem, password=None)
    signature = key.sign(message)  # type: ignore[call-arg]  # Ed25519 sign(data)
    return base64.b64encode(signature).decode()


class PolymarketUSVenue:
    """Read-only QCEX / Polymarket US client. ``cfg`` is a ``QcexConfig``."""

    name = VENUE

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.fee_model = ZeroFeeModel()
        self._client = None

    def _http(self):
        if self._client is None:
            import httpx  # lazy

            self._client = httpx.AsyncClient(base_url=self.cfg.api_base, timeout=10.0)
        return self._client

    async def list_markets(self, limit: int = 200) -> list[RawMarket]:
        resp = await self._http().get("/markets", params={"limit": limit})
        resp.raise_for_status()
        data = resp.json()
        markets = data.get("data", data) if isinstance(data, dict) else data
        return [
            RawMarket(
                market_id=m.get("condition_id") or m.get("id") or m.get("market_id"),
                title=m.get("question") or m.get("title", ""),
                raw=m,
            )
            for m in markets
        ]

    async def stream_order_book(self, market_ids: list[str]) -> AsyncIterator[MarketQuote]:
        raise NotImplementedError("QCEX WS streaming is implemented in the live phase")
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
