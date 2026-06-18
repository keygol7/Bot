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

import asyncio
import base64
import logging
import time
import uuid
from typing import Any, AsyncIterator, Optional
from urllib.parse import urlsplit

from bot.execution.orders import OrderResult, OrderStatus
from bot.fees import KalshiFeeModel
from bot.models import MarketQuote, PriceLevel, Side
from bot.timeutil import parse_iso8601
from bot.venues.base import OrderNotPermitted, RawMarket
from bot.venues.ratelimit import AsyncRateLimiter

VENUE = "kalshi"
log = logging.getLogger("bot.venues.kalshi")

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


def _bid_levels_dollars(orderbook: dict[str, Any], side: str) -> list[tuple[float, float]]:
    """Bid levels for ``side`` ('yes'/'no') as ``[(price_dollars, size), ...]``.

    Handles both Kalshi shapes: the current ``orderbook_fp`` with ``yes_dollars`` /
    ``no_dollars`` (dollar-string prices, fractional sizes) and the legacy
    ``{"yes": [[price_cents, size], ...]}``.
    """
    fp = orderbook.get(f"{side}_dollars")
    if fp is not None:
        return [(float(p), float(s)) for p, s in fp]
    return [(float(c) / 100.0, float(s)) for c, s in (orderbook.get(side) or [])]


def normalize_orderbook(
    ticker: str, title: str, orderbook: dict[str, Any], *, event_key: str | None = None
) -> MarketQuote:
    """Normalize a Kalshi orderbook dict into a :class:`MarketQuote`.

    Accepts the current ``orderbook_fp`` shape (``yes_dollars``/``no_dollars``, dollar
    prices) and the legacy cents shape. ``yes_ask`` is derived by crossing the best NO
    bid (cost to buy YES = 1 - best NO bid) and vice versa, per Kalshi's book semantics.
    """
    yes_bids = _bid_levels_dollars(orderbook, "yes")
    no_bids = _bid_levels_dollars(orderbook, "no")

    yes_ask = no_ask = None
    yes_ask_size = no_ask_size = 0.0

    if no_bids:
        price, size = max(no_bids, key=lambda lvl: lvl[0])
        yes_ask = round(1.0 - price, 4)
        yes_ask_size = size

    if yes_bids:
        price, size = max(yes_bids, key=lambda lvl: lvl[0])
        no_ask = round(1.0 - price, 4)
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


def parse_ticker(message: dict[str, Any]) -> MarketQuote | None:
    """Normalize a Kalshi WS ``ticker`` message into a top-of-book quote.

    ``msg`` carries ``market_ticker`` and ``yes_bid_dollars`` / ``yes_ask_dollars``.
    ``yes_ask`` is the cost to buy YES; the NO ask is ``1 - yes_bid``. The ticker
    channel has no depth, so sizes are 0 (depth comes from orderbook_delta or a REST
    fetch on shortlisted markets).
    """
    m = message.get("msg", message)
    ticker = m.get("market_ticker")
    if not ticker:
        return None

    def _px(v: Any) -> float | None:
        return float(v) if v not in (None, "") else None

    yes_ask = _px(m.get("yes_ask_dollars"))
    yes_bid = _px(m.get("yes_bid_dollars"))
    return MarketQuote(
        venue=VENUE, market_id=ticker, title="",
        yes_ask=yes_ask, yes_ask_size=0.0,
        no_ask=round(1.0 - yes_bid, 4) if yes_bid is not None else None,
        no_ask_size=0.0, timestamp=time.time(),
    )


def parse_fill(message: dict[str, Any]):
    """Normalize a Kalshi private ``fill`` message into a ``FillEvent`` (or None).

    NOTE: the exact fill-message field set should be confirmed against the demo
    environment; this parses defensively.
    """
    from bot.streaming.fills import FillEvent

    m = message.get("msg", message)
    order_id = m.get("order_id")
    if not order_id:
        return None
    cnt = m.get("count") or m.get("count_fp") or m.get("fill_count_fp")
    last_shares = float(cnt) if cnt not in (None, "") else 0.0
    px = None
    for k in ("yes_price_dollars", "no_price_dollars"):
        if m.get(k) not in (None, ""):
            px = float(m[k])
            break
    if px is None:
        for k in ("yes_price", "no_price"):
            if m.get(k) not in (None, ""):
                px = float(m[k]) / 100.0
                break
    return FillEvent("kalshi", str(order_id), "FILL", last_shares, px)


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
        close_time=parse_iso8601(m.get("close_time") or m.get("expiration_time")),
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
        body = resp.json()
        # Current API returns "orderbook_fp" (dollar prices, fractional sizes); fall
        # back to the legacy "orderbook" (cents) if present.
        ob = body.get("orderbook_fp") or body.get("orderbook") or {}
        return normalize_orderbook(ticker, title, ob)

    async def fetch_quote(self, market: RawMarket) -> MarketQuote:
        """Deep (sized) quote for one market — used in phase 2 for shortlisted markets."""
        return await self.fetch_orderbook(market.market_id, market.title)

    async def is_open(self, market_id: str) -> Optional[bool]:
        """Whether the market is still open for trading (vs closed/settled).

        Used by the streaming watchlist to keep open-but-illiquid markets while
        shedding settled ones — independent of current book depth. Returns ``None`` if
        the status can't be determined (caller keeps the market on uncertainty).
        """
        await self._limiter.wait()
        endpoint = f"/markets/{market_id}"
        try:
            resp = await self._http().get(endpoint, headers=self._auth_headers("GET", endpoint))
            resp.raise_for_status()
        except Exception:
            return None
        status = (resp.json().get("market") or {}).get("status")
        if not status:
            return None
        # Treat anything terminal as closed; everything else (open/active/...) as live.
        return status.lower() not in {
            "closed", "settled", "determined", "finalized", "cancelled", "expired",
        }

    async def scan_quotes(self, limit: int = 500) -> list[MarketQuote]:
        """Phase 1: price-only quotes for open markets, up to ``limit`` markets total.

        Kalshi caps a single /markets page at 1000 and the markets we care about can
        sit past the first page (e.g. UFC fights), so we follow the ``cursor`` until
        we've scanned ``limit`` markets or the feed is exhausted. ``limit`` is a TOTAL
        cap across pages, not a per-page size.
        """
        out: list[MarketQuote] = []
        cursor: str | None = None
        fetched = 0
        while fetched < limit:
            await self._limiter.wait()
            params = {
                # mve_filter=exclude drops multivariate/parlay markets server-side,
                # which otherwise dominate the feed. Client-side filter below backs it up.
                "limit": min(limit - fetched, 1000),
                "status": "open",
                "mve_filter": "exclude",
            }
            if cursor:
                params["cursor"] = cursor  # query only — not part of the signed path
            resp = await self._http().get(
                "/markets", params=params, headers=self._auth_headers("GET", "/markets"),
            )
            resp.raise_for_status()
            body = resp.json()
            markets = body.get("markets", [])
            fetched += len(markets)
            # Drop multivariate/parlay markets — not arbitrageable, junk titles.
            out.extend(
                normalize_summary(m)
                for m in markets
                if m.get("ticker") and not is_multivariate(m["ticker"])
            )
            cursor = body.get("cursor")
            if not cursor or not markets:
                break  # end of feed
        return out

    def _ws_auth_headers(self) -> dict[str, str]:
        """Auth headers for the WS handshake (signs GET + the WS path)."""
        if not self.authenticated:
            raise OrderNotPermitted("Kalshi credentials required for the WebSocket")
        if self._private_key is None:
            self._private_key = load_private_key(self.cfg.private_key_path)
        ws_path = urlsplit(self.cfg.ws_base).path or "/trade-api/ws/v2"
        return build_signature_headers(self.cfg.api_key_id, self._private_key, "GET", ws_path)

    async def stream_order_book(self, market_ids: list[str]) -> AsyncIterator[MarketQuote]:
        """Stream real-time top-of-book quotes via the ``ticker`` channel.

        Reconnects with exponential backoff. Yields a normalized MarketQuote per
        ticker update. ``market_ids`` filters to specific tickers (omit for all).
        """
        import json

        import websockets  # lazy

        backoff = 1.0
        while True:
            try:
                headers = self._ws_auth_headers()
                async with websockets.connect(
                    self.cfg.ws_base, additional_headers=headers, open_timeout=10
                ) as ws:
                    params: dict[str, Any] = {"channels": ["ticker"]}
                    if market_ids:
                        params["market_tickers"] = market_ids
                    await ws.send(json.dumps({"id": 1, "cmd": "subscribe", "params": params}))
                    backoff = 1.0  # reset on a healthy connection
                    async for raw in ws:
                        data = json.loads(raw)
                        if data.get("type") == "ticker":
                            quote = parse_ticker(data)
                            if quote is not None:
                                yield quote
                        elif data.get("type") == "error":
                            log.warning("kalshi ws error: %s", data.get("msg"))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("kalshi ws disconnected (%s); reconnecting in %.0fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def stream_private(self):
        """Stream private fills (FillEvents) from the ``fill`` channel."""
        import json

        import websockets  # lazy

        backoff = 1.0
        while True:
            try:
                headers = self._ws_auth_headers()
                async with websockets.connect(
                    self.cfg.ws_base, additional_headers=headers, open_timeout=10
                ) as ws:
                    await ws.send(json.dumps({"id": 1, "cmd": "subscribe",
                                              "params": {"channels": ["fill"]}}))
                    backoff = 1.0
                    async for raw in ws:
                        data = json.loads(raw)
                        if data.get("type") == "fill":
                            ev = parse_fill(data)
                            if ev is not None:
                                yield ev
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("kalshi private ws disconnected (%s); reconnecting in %.0fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def place_order(
        self, market_id: str, side: Side, action: str, price: float, contracts: float,
        *, tif: str = "fill_or_kill",
    ) -> OrderResult:
        """Place an order via legacy /portfolio/orders (explicit yes/no side).

        ``price`` is the cost to buy that ``side`` (dollars 0.01-0.99); converted to
        Kalshi cents. ``contracts`` is whole contracts. Requires credentials.
        NOTE: fill-response field mapping should be confirmed against the demo
        environment before live use.
        """
        if not self.authenticated:
            raise OrderNotPermitted("Kalshi credentials not configured")
        cents = int(round(price * 100))
        count = int(round(contracts))
        body: dict[str, Any] = {
            "ticker": market_id,
            "action": action,                 # "buy" | "sell"
            "side": side.value.lower(),        # "yes" | "no"
            "count": count,
            "time_in_force": tif,              # fill_or_kill | immediate_or_cancel | ...
            "client_order_id": str(uuid.uuid4()),
        }
        body["yes_price" if side is Side.YES else "no_price"] = cents

        await self._limiter.wait()
        try:
            resp = await self._http().post(
                "/portfolio/orders", json=body,
                headers=self._auth_headers("POST", "/portfolio/orders"),
            )
            resp.raise_for_status()
            order = resp.json().get("order", {})
        except Exception as exc:  # network/HTTP error -> position state UNKNOWN
            return OrderResult(VENUE, market_id, side, action, count, status=OrderStatus.ERROR,
                               raw={"error": str(exc)})

        filled = float(order.get("fill_count_fp") or order.get("fill_count") or 0)
        price_key = "yes_price_dollars" if side is Side.YES else "no_price_dollars"
        avg = order.get(price_key)
        status = (
            OrderStatus.FILLED if filled >= count - 1e-9
            else OrderStatus.KILLED if filled <= 1e-9
            else OrderStatus.PARTIAL
        )
        return OrderResult(
            venue=VENUE, market_id=market_id, side=side, action=action,
            requested=count, filled=filled,
            avg_price=float(avg) if avg not in (None, "") else None,
            order_id=order.get("order_id"), status=status, raw=order,
        )

    async def cancel_order(self, order_id: str) -> dict:
        if not self.authenticated:
            raise OrderNotPermitted("Kalshi credentials not configured")
        path = f"/portfolio/orders/{order_id}"
        await self._limiter.wait()
        resp = await self._http().request(
            "DELETE", path, headers=self._auth_headers("DELETE", path)
        )
        resp.raise_for_status()
        return resp.json()

    async def get_positions(self) -> dict:
        raise NotImplementedError("positions endpoint lands with the live phase")

    async def account_snapshot(self):
        """Balance + non-flat positions/resting orders, for the startup guard.

        Uses the authenticated portfolio endpoints: ``/portfolio/balance`` (cents)
        and ``/portfolio/positions`` (``market_positions`` with signed ``position``
        and ``resting_orders_count``). Raises on any HTTP error so the guard fails
        closed rather than assuming the account is flat.
        """
        from bot.execution.account import AccountSnapshot, VenuePosition

        await self._limiter.wait()
        resp = await self._http().get(
            "/portfolio/balance", headers=self._auth_headers("GET", "/portfolio/balance")
        )
        resp.raise_for_status()
        bal = resp.json().get("balance")
        balance = float(bal) / 100.0 if bal not in (None, "") else None

        await self._limiter.wait()
        endpoint = "/portfolio/positions"
        resp = await self._http().get(endpoint, headers=self._auth_headers("GET", endpoint))
        resp.raise_for_status()
        positions = []
        for mp in resp.json().get("market_positions") or []:
            qty = float(mp.get("position") or 0)
            resting = int(mp.get("resting_orders_count") or 0)
            pos = VenuePosition(mp.get("ticker", ""), qty, resting)
            if pos.is_open:
                positions.append(pos)
        return AccountSnapshot(self.name, balance, positions)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
