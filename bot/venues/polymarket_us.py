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

import json

from bot.fees import ZeroFeeModel
from bot.models import MarketQuote, PriceLevel, Side
from bot.venues.base import OrderNotPermitted, RawMarket
from bot.venues.ratelimit import AsyncRateLimiter

VENUE = "polymarket_us"


def extract_token_ids(raw: dict[str, Any]) -> tuple[str | None, str | None]:
    """Pull the (YES, NO) CLOB token ids out of a market payload.

    Handles both shapes seen in the wild:
      - ``tokens``: [{"token_id": "...", "outcome": "Yes"}, {...}]
      - ``clobTokenIds``: a list or JSON-encoded string of two ids ([yes, no]).
    Returns ``(None, None)`` if neither is present.
    """
    tokens = raw.get("tokens")
    if isinstance(tokens, list) and tokens:
        yes_id = no_id = None
        for t in tokens:
            outcome = str(t.get("outcome", "")).strip().lower()
            if outcome in ("yes", "true"):
                yes_id = t.get("token_id") or t.get("tokenId")
            elif outcome in ("no", "false"):
                no_id = t.get("token_id") or t.get("tokenId")
        if yes_id or no_id:
            return yes_id, no_id

    clob = raw.get("clobTokenIds") or raw.get("clob_token_ids")
    if isinstance(clob, str):
        try:
            clob = json.loads(clob)
        except (ValueError, TypeError):
            clob = None
    if isinstance(clob, (list, tuple)) and len(clob) >= 2:
        return clob[0], clob[1]
    return None, None


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


def load_ed25519_key(secret_key: str = "", pem_path: str = ""):
    """Load the Ed25519 signing key from a raw base64 secret (preferred) or a PEM.

    Polymarket US issues the secret as a base64 string; per their docs the private
    key is ``Ed25519PrivateKey.from_private_bytes(b64decode(secret)[:32])``. A PEM
    file containing the same key also works (lazy ``cryptography`` import).
    """
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    if secret_key:
        raw = base64.b64decode(secret_key + "=" * (-len(secret_key) % 4))
        return Ed25519PrivateKey.from_private_bytes(raw[:32])
    if pem_path:
        from cryptography.hazmat.primitives.serialization import load_pem_private_key

        with open(pem_path, "rb") as fh:
            return load_pem_private_key(fh.read(), password=None)
    raise ValueError("no QCEX secret_key or PEM path provided")


def build_auth_headers(
    key_id: str, private_key, method: str, path: str, timestamp_ms: str | None = None
) -> dict[str, str]:
    """Polymarket US authenticated-request headers (X-PM-* / Ed25519).

    Signs ``timestamp + METHOD + path`` (path excludes the query string). Used only
    for trading endpoints — public market-data reads need no auth.
    """
    ts = timestamp_ms if timestamp_ms is not None else str(int(time.time() * 1000))
    message = f"{ts}{method.upper()}{path}"
    signature = base64.b64encode(private_key.sign(message.encode())).decode()
    return {
        "X-PM-Access-Key": key_id,
        "X-PM-Timestamp": ts,
        "X-PM-Signature": signature,
        "Content-Type": "application/json",
    }


class PolymarketUSVenue:
    """Read-only QCEX / Polymarket US client. ``cfg`` is a ``QcexConfig``."""

    name = VENUE

    def __init__(self, cfg: Any, rate_per_min: float = 100.0) -> None:
        self.cfg = cfg
        self.fee_model = ZeroFeeModel()
        self._gateway_client = None  # public reads
        self._api_client = None      # authenticated trading (later)
        self._key = None
        self._limiter = AsyncRateLimiter(rate_per_min)

    @property
    def is_trading_configured(self) -> bool:
        """Order placement needs creds; reads (this phase) do not."""
        return getattr(self.cfg, "is_trading_configured", False)

    def _gateway(self):
        """HTTP client for the PUBLIC market-data gateway (no auth)."""
        if self._gateway_client is None:
            import httpx  # lazy

            self._gateway_client = httpx.AsyncClient(
                base_url=self.cfg.gateway_base, timeout=10.0
            )
        return self._gateway_client

    def _auth_headers(self, method: str, path: str) -> dict[str, str]:
        """X-PM-* signed headers for authenticated (trading) requests."""
        if self._key is None:
            self._key = load_ed25519_key(
                getattr(self.cfg, "secret_key", ""),
                getattr(self.cfg, "ed25519_private_key_path", ""),
            )
        return build_auth_headers(self.cfg.api_key_id, self._key, method, path)

    async def _fetch_book(self, token_id: str) -> dict[str, Any]:
        # NOTE: endpoint path/params pending confirmation from the public Markets
        # API reference (docs.polymarket.us). Reads hit the public gateway.
        await self._limiter.wait()
        resp = await self._gateway().get("/book", params={"token_id": token_id})
        resp.raise_for_status()
        return resp.json()

    async def fetch_quote(self, market: RawMarket) -> MarketQuote | None:
        """Fetch and combine the YES/NO token books into one quote.

        Returns ``None`` if the market exposes no usable token ids.
        """
        yes_id, no_id = extract_token_ids(market.raw)
        if not yes_id:
            return None
        yes_book = await self._fetch_book(yes_id)
        no_book = await self._fetch_book(no_id) if no_id else None
        return build_quote(market.market_id, market.title, yes_book, no_book)

    async def list_markets(self, limit: int = 200) -> list[RawMarket]:
        # NOTE: path/shape pending confirmation from the public Markets API reference.
        await self._limiter.wait()
        resp = await self._gateway().get("/markets", params={"limit": limit})
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
        for attr in ("_gateway_client", "_api_client"):
            client = getattr(self, attr)
            if client is not None:
                await client.aclose()
                setattr(self, attr, None)
