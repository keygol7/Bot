"""Polymarket US / QCEX venue adapter.

Market data comes from the PUBLIC gateway (`gateway.polymarket.us`) — no auth:
  - ``GET /v1/markets``            list markets (each carries bestBid/bestAsk)
  - ``GET /v1/markets/{slug}/bbo`` best bid/offer + depth for one market

Binary mapping (prices in dollars, 0..1): ``bestAsk`` is the cost to buy YES; the
NO ask is ``1 - bestBid`` (buying NO == taking the YES bid). Sizes come from
``askDepth`` / ``bidDepth``. Standard markets are ~zero-fee -> :class:`ZeroFeeModel`.

Trading (later) uses the authenticated API (`api.polymarket.us`) with X-PM-* Ed25519
headers — see :func:`build_auth_headers` / :func:`load_ed25519_key`. Read-only here.
"""

from __future__ import annotations

import base64
import time
from typing import Any, AsyncIterator

from bot.fees import ZeroFeeModel
from bot.models import MarketQuote, Side
from bot.venues.base import OrderNotPermitted, RawMarket
from bot.venues.ratelimit import AsyncRateLimiter

VENUE = "polymarket_us"


def _amount(value: Any) -> float | None:
    """Parse a price from a v1Amount object ({value,currency}) or a bare number/str."""
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("value")
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_bbo(
    slug: str, title: str, market_data: dict[str, Any], *, event_key: str | None = None
) -> MarketQuote:
    """Turn a ``v1MarketDataLite`` (BBO) payload into a :class:`MarketQuote`.

    ``bestAsk`` is the YES ask; the NO ask is synthesized as ``1 - bestBid``. Sizes
    are taken from ``askDepth`` (YES) and ``bidDepth`` (the YES bid backing the NO).
    """
    best_ask = _amount(market_data.get("bestAsk"))
    best_bid = _amount(market_data.get("bestBid"))
    ask_depth = float(market_data.get("askDepth") or 0)
    bid_depth = float(market_data.get("bidDepth") or 0)

    no_ask = round(1.0 - best_bid, 6) if best_bid is not None else None

    return MarketQuote(
        venue=VENUE,
        market_id=slug,
        title=title,
        event_key=event_key,
        yes_ask=best_ask,
        yes_ask_size=ask_depth,
        no_ask=no_ask,
        no_ask_size=bid_depth,
    )


def _long_side_outcome(m: dict[str, Any]) -> str | None:
    """The YES (``long``) side's outcome label for a Polymarket US market.

    Each market is binary; ``marketSides`` carries the two outcomes. We surface the
    long side's team name (or its description) to disambiguate otherwise-identical
    questions (e.g. many "World Series Champion" markets, one per team).
    """
    for side in m.get("marketSides") or []:
        if side.get("long"):
            team = side.get("team") or {}
            return team.get("name") or side.get("description")
    return None


def build_market_title(m: dict[str, Any]) -> str:
    """Descriptive, outcome-disambiguated title for matching."""
    question = m.get("question") or m.get("title") or ""
    outcome = _long_side_outcome(m)
    if outcome and outcome.lower() not in question.lower():
        return f"{question} - {outcome}"
    return question


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
    """Polymarket US client. Reads via the public gateway; trading needs creds."""

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

    async def list_markets(self, limit: int = 200) -> list[RawMarket]:
        await self._limiter.wait()
        resp = await self._gateway().get(
            "/v1/markets", params={"limit": limit, "active": "true", "closed": "false"}
        )
        resp.raise_for_status()
        out: list[RawMarket] = []
        for m in resp.json().get("markets", []):
            slug = m.get("slug") or m.get("id")
            if not slug:
                continue
            out.append(RawMarket(market_id=slug, title=build_market_title(m), raw=m))
        return out

    async def scan_quotes(self, limit: int = 500) -> list[MarketQuote]:
        """Phase 1: one /v1/markets call -> price-only quotes (bestBid/bestAsk).

        Sizes are not in the list payload, so this is the cheap wide scan; the sized
        quote comes from :meth:`fetch_quote` (BBO) for shortlisted markets only.
        """
        await self._limiter.wait()
        resp = await self._gateway().get(
            "/v1/markets", params={"limit": limit, "active": "true", "closed": "false"}
        )
        resp.raise_for_status()
        out: list[MarketQuote] = []
        for m in resp.json().get("markets", []):
            slug = m.get("slug") or m.get("id")
            if not slug:
                continue
            best_ask = _amount(m.get("bestAsk"))
            best_bid = _amount(m.get("bestBid"))
            out.append(
                MarketQuote(
                    venue=VENUE,
                    market_id=slug,
                    title=build_market_title(m),
                    yes_ask=best_ask,
                    yes_ask_size=0.0,
                    no_ask=round(1.0 - best_bid, 6) if best_bid is not None else None,
                    no_ask_size=0.0,
                )
            )
        return out

    async def fetch_quote(self, market: RawMarket) -> MarketQuote | None:
        """Deep (sized) quote: BBO for one market (by slug), used in phase 2."""
        await self._limiter.wait()
        resp = await self._gateway().get(f"/v1/markets/{market.market_id}/bbo")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        market_data = resp.json().get("marketData", {})
        if not market_data:
            return None
        return normalize_bbo(market.market_id, market.title, market_data)

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
