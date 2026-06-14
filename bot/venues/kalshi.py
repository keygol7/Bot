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

import base64
import time
from typing import Any, AsyncIterator
from urllib.parse import urlsplit

from bot.fees import KalshiFeeModel
from bot.models import MarketQuote, PriceLevel, Side
from bot.venues.base import OrderNotPermitted, RawMarket
from bot.venues.ratelimit import AsyncRateLimiter

VENUE = "kalshi"

# ---------------------------------------------------------------------------
# Request signing (optional). Kalshi authenticates API-key requests with an
# RSA-PSS signature over ``timestamp + METHOD + path``; the bot signs only when
# credentials are configured, so unauthenticated reads still work where allowed.
# Read-only here regardless — signing never enables order placement.
# ---------------------------------------------------------------------------


def load_private_key(path: str):
    """Load an RSA private key from a PEM file (lazy ``cryptography`` import)."""
    from cryptography.hazmat.primitives.serialization import load_pem_private_key

    with open(path, "rb") as fh:
        return load_pem_private_key(fh.read(), password=None)


def pss_sign(private_key, message: str) -> str:
    """RSA-PSS (SHA-256, MGF1-SHA256, digest-length salt) signature, base64-encoded."""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding

    signature = private_key.sign(
        message.encode(),
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode()


def build_signature_headers(
    api_key_id: str, private_key, method: str, path: str, timestamp_ms: str | None = None
) -> dict[str, str]:
    """Build the Kalshi auth headers for one request.

    ``path`` is the full request path Kalshi expects in the signed string
    (e.g. ``/trade-api/v2/markets``), excluding any query string.
    """
    ts = timestamp_ms if timestamp_ms is not None else str(int(time.time() * 1000))
    message = ts + method.upper() + path
    return {
        "KALSHI-ACCESS-KEY": api_key_id,
        "KALSHI-ACCESS-SIGNATURE": pss_sign(private_key, message),
        "KALSHI-ACCESS-TIMESTAMP": ts,
    }


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


def is_multivariate(ticker: str) -> bool:
    """True for Kalshi multivariate / parlay markets (ticker prefix ``KXMVE``).

    These combine many legs into one contract; their title is a comma-joined list
    of legs, so they are neither arbitrageable binaries nor useful for matching.
    """
    return ticker.upper().startswith("KXMVE")


def _cents_to_price(cents: Any) -> float | None:
    """Convert a Kalshi cents price (1..99) to dollars; 0/None -> None (no quote)."""
    if cents in (None, "", 0, 0.0):
        return None
    try:
        return round(float(cents) / 100.0, 4)
    except (TypeError, ValueError):
        return None


def build_summary_title(m: dict[str, Any]) -> str:
    """Outcome-disambiguated title for a Kalshi market.

    Kalshi returns one market per outcome but gives them the same ``title``
    (e.g. both sides of a game read "X vs Y Winner?"). ``yes_sub_title`` names the
    YES outcome (the team/side), so append it for matching.
    """
    title = m.get("title", "") or ""
    sub = m.get("yes_sub_title") or m.get("subtitle")
    suffix = f" - {sub}"
    # Always append the YES outcome (both sides of "X vs Y" share the title);
    # only guard against re-appending the exact suffix (idempotent).
    if sub and not title.endswith(suffix):
        return title + suffix
    return title


def normalize_summary(m: dict[str, Any], *, event_key: str | None = None) -> MarketQuote:
    """Build a price-only quote from a Kalshi /markets summary row (no per-market call).

    The summary already carries ``yes_ask`` / ``no_ask`` (cents). Sizes are not in the
    summary, so this is for the cheap wide scan; the sized quote comes from
    :meth:`KalshiVenue.fetch_orderbook` for shortlisted markets only.
    """
    return MarketQuote(
        venue=VENUE,
        market_id=m.get("ticker", ""),
        title=build_summary_title(m),
        event_key=event_key,
        yes_ask=_cents_to_price(m.get("yes_ask")),
        yes_ask_size=0.0,
        no_ask=_cents_to_price(m.get("no_ask")),
        no_ask_size=0.0,
    )


class KalshiVenue:
    """Read-only Kalshi client. ``cfg`` is a ``bot.config.settings.KalshiConfig``."""

    name = VENUE

    def __init__(self, cfg: Any, fee_rate: float = 0.07, rate_per_min: float = 55.0) -> None:
        self.cfg = cfg
        self.fee_model = KalshiFeeModel(rate=fee_rate)
        self._client = None  # lazy httpx.AsyncClient
        self._private_key = None
        self._base_path = urlsplit(cfg.api_base).path.rstrip("/")  # e.g. /trade-api/v2
        self._limiter = AsyncRateLimiter(rate_per_min)

    @property
    def authenticated(self) -> bool:
        return bool(getattr(self.cfg, "api_key_id", "")) and bool(
            getattr(self.cfg, "private_key_path", "")
        )

    def _http(self):
        if self._client is None:
            import httpx  # lazy

            self._client = httpx.AsyncClient(base_url=self.cfg.api_base, timeout=10.0)
        return self._client

    def _auth_headers(self, method: str, endpoint_path: str) -> dict[str, str]:
        """Signed headers when credentials are configured; empty dict otherwise."""
        if not self.authenticated:
            return {}
        if self._private_key is None:
            self._private_key = load_private_key(self.cfg.private_key_path)
        return build_signature_headers(
            self.cfg.api_key_id, self._private_key, method, self._base_path + endpoint_path
        )

    async def list_markets(self, limit: int = 200) -> list[RawMarket]:
        await self._limiter.wait()
        resp = await self._http().get(
            "/markets",
            params={"limit": limit, "status": "open", "mve_filter": "exclude"},
            headers=self._auth_headers("GET", "/markets"),
        )
        resp.raise_for_status()
        data = resp.json()
        return [
            RawMarket(market_id=m["ticker"], title=m.get("title", ""), raw=m)
            for m in data.get("markets", [])
        ]

    async def fetch_orderbook(self, ticker: str, title: str = "") -> MarketQuote:
        await self._limiter.wait()
        endpoint = f"/markets/{ticker}/orderbook"
        resp = await self._http().get(endpoint, headers=self._auth_headers("GET", endpoint))
        resp.raise_for_status()
        ob = resp.json().get("orderbook", {})
        return normalize_orderbook(ticker, title, ob)

    async def fetch_quote(self, market: RawMarket) -> MarketQuote:
        """Deep (sized) quote for one market — used in phase 2 for shortlisted markets."""
        return await self.fetch_orderbook(market.market_id, market.title)

    async def scan_quotes(self, limit: int = 500) -> list[MarketQuote]:
        """Phase 1: one /markets call -> price-only quotes for every open market."""
        await self._limiter.wait()
        resp = await self._http().get(
            "/markets",
            # mve_filter=exclude drops multivariate/parlay markets server-side, which
            # otherwise dominate the open feed. The client-side filter below is a backstop.
            params={"limit": limit, "status": "open", "mve_filter": "exclude"},
            headers=self._auth_headers("GET", "/markets"),
        )
        resp.raise_for_status()
        markets = resp.json().get("markets", [])
        # Drop multivariate/parlay markets — not arbitrageable, junk titles.
        return [
            normalize_summary(m)
            for m in markets
            if m.get("ticker") and not is_multivariate(m["ticker"])
        ]

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
