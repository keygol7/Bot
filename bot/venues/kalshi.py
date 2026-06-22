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
from typing import Any, AsyncIterator, ClassVar, NamedTuple, Optional
from urllib.parse import urlsplit

from bot.execution.orders import OrderResult, OrderStatus
from bot.fees import KalshiFeeModel
from bot.models import MarketQuote, Side
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
    """Normalize a Kalshi WS ``ticker`` message into a SIZED top-of-book quote.

    ``msg`` carries ``yes_bid_dollars`` / ``yes_ask_dollars`` AND the top-of-book sizes
    ``yes_bid_size_fp`` / ``yes_ask_size_fp`` (contracts at the best bid/ask). To buy
    YES you cross the best ask (``yes_ask``; size = ``yes_ask_size_fp``); to buy NO you
    cross the best YES bid, so ``no_ask = 1 - yes_bid`` and its size is the YES-bid size
    ``yes_bid_size_fp``. The ticker fires on any field change, so this is a fresh,
    sized top-of-book over the WS — no order-book-delta bookkeeping needed.
    """
    m = message.get("msg", message)
    ticker = m.get("market_ticker")
    if not ticker:
        return None

    def _px(v: Any) -> float | None:
        return float(v) if v not in (None, "") else None

    def _sz(v: Any) -> float:
        return float(v) if v not in (None, "") else 0.0

    yes_ask = _px(m.get("yes_ask_dollars"))
    yes_bid = _px(m.get("yes_bid_dollars"))
    return MarketQuote(
        venue=VENUE, market_id=ticker, title="",
        yes_ask=yes_ask, yes_ask_size=_sz(m.get("yes_ask_size_fp")),
        no_ask=round(1.0 - yes_bid, 4) if yes_bid is not None else None,
        no_ask_size=_sz(m.get("yes_bid_size_fp")),
        timestamp=time.time(),
    )


class LifecycleEvent(NamedTuple):
    """A parsed market_lifecycle_v2 update. ``state`` is "MARKET_STATE_OPEN" when the
    market is (re)tradeable, a non-open marker when paused/deactivated/closed, or None
    when the event doesn't change tradeability. ``terminal`` is True for determined/
    settled markets (prune them from the watchlist)."""

    market_ticker: str
    state: str | None
    terminal: bool


# Kalshi lifecycle event_type -> non-open state marker (anything != MARKET_STATE_OPEN
# blocks the fire). created/close_date_updated/price_level/metadata don't change
# tradeability (-> None, leave state as-is; unknown stays allowed).
_LIFECYCLE_BLOCK = {
    "deactivated": "KALSHI_DEACTIVATED",
    "determined": "KALSHI_DETERMINED",
    "settled": "KALSHI_SETTLED",
}
_LIFECYCLE_TERMINAL = {"determined", "settled"}


def parse_lifecycle(message: dict[str, Any]) -> LifecycleEvent | None:
    """Parse a ``market_lifecycle_v2`` message into a :class:`LifecycleEvent` (or None).

    ``is_deactivated`` (pause/unpause on an open market) takes precedence when present:
    True -> blocked, False -> open. Otherwise ``event_type`` decides: ``activated`` ->
    open; ``deactivated``/``determined``/``settled`` -> blocked (the last two terminal).
    """
    if message.get("type") != "market_lifecycle_v2":
        return None
    m = message.get("msg", message)
    ticker = m.get("market_ticker")
    if not ticker:
        return None
    event = m.get("event_type")
    is_deact = m.get("is_deactivated")
    terminal = event in _LIFECYCLE_TERMINAL
    if is_deact is not None:
        state = "MARKET_STATE_OPEN" if not is_deact else "KALSHI_PAUSED"
    elif event == "activated":
        state = "MARKET_STATE_OPEN"
    else:
        state = _LIFECYCLE_BLOCK.get(event)  # None for created/metadata/etc. (no change)
    return LifecycleEvent(ticker, state, terminal)


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
        self._limiter = AsyncRateLimiter(getattr(cfg, "read_rate_per_min", None) or rate_per_min)

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

    async def scan_quotes(
        self, limit: int = 500, *, max_close_ts: int | None = None
    ) -> list[MarketQuote]:
        """Phase 1: price-only quotes for open markets, up to ``limit`` markets total.

        Kalshi caps a single /markets page at 1000 and the markets we care about can
        sit past the first page (e.g. UFC fights), so we follow the ``cursor`` until
        we've scanned ``limit`` markets or the feed is exhausted. ``limit`` is a TOTAL
        cap across pages, not a per-page size. ``limit <= 0`` scans the ENTIRE board
        (paginate until the cursor is exhausted).

        ``max_close_ts`` (Unix seconds) enables a TARGETED scan: only markets closing
        at/before that time (the live/imminent set). It's sent as the documented
        ``max_close_ts`` query param AND enforced client-side on ``close_time`` as a
        fail-safe — so coverage is correct whether or not the server honors the param.
        Markets with no close_time are kept (don't drop a live market on missing data).
        """
        out: list[MarketQuote] = []
        cursor: str | None = None
        fetched = 0
        unbounded = limit <= 0                          # scan the whole board
        while unbounded or fetched < limit:
            await self._limiter.wait()
            params: dict[str, Any] = {
                # mve_filter=exclude drops multivariate/parlay markets server-side,
                # which otherwise dominate the feed. Client-side filter below backs it up.
                "limit": 1000 if unbounded else min(limit - fetched, 1000),
                "status": "open",
                "mve_filter": "exclude",
            }
            if max_close_ts is not None:
                params["max_close_ts"] = int(max_close_ts)
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
            for m in markets:
                if not m.get("ticker") or is_multivariate(m["ticker"]):
                    continue
                q = normalize_summary(m)
                # Client-side fail-safe for the targeted window (server may ignore the
                # param). Keep markets with no close_time — never drop on missing data.
                if (
                    max_close_ts is not None
                    and q.close_time is not None
                    and q.close_time > max_close_ts
                ):
                    continue
                out.append(q)
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
                    params: dict[str, Any] = {
                        "channels": ["ticker"],
                        # Get an immediate sized top-of-book on subscribe instead of
                        # waiting for the first field change (which could be a while on a
                        # quiet market) — primes the live book over the WS itself.
                        "send_initial_snapshot": True,
                    }
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

    # tif -> V2 time_in_force enum (GTT is internal-only and not a valid API value).
    _TIF_V2: ClassVar[dict[str, str]] = {
        "fill_or_kill": "fill_or_kill",
        "immediate_or_cancel": "immediate_or_cancel",
        "gtc": "good_till_canceled",
        "good_till_canceled": "good_till_canceled",
    }

    async def stream_lifecycle(self):
        """Stream ``market_lifecycle_v2`` events as :class:`LifecycleEvent`s.

        The channel has no market filter (it carries every market's lifecycle), so the
        consumer filters to the watchlist client-side. Reconnects with backoff.
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
                    await ws.send(json.dumps({"id": 1, "cmd": "subscribe",
                                              "params": {"channels": ["market_lifecycle_v2"]}}))
                    backoff = 1.0
                    async for raw in ws:
                        ev = parse_lifecycle(json.loads(raw))
                        if ev is not None:
                            yield ev
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("kalshi lifecycle ws disconnected (%s); reconnecting in %.0fs",
                            exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def place_order(
        self, market_id: str, side: Side, action: str, price: float, contracts: float,
        *, tif: str = "fill_or_kill", post_only: bool = False, expiration_ts: int | None = None,
    ) -> OrderResult:
        """Place an order via the V2 /portfolio/events/orders endpoint.

        The legacy /portfolio/orders was deprecated (no earlier than 2026-05-06). V2
        quotes everything from the YES side: ``bid`` buys YES, ``ask`` sells YES, and
        buying NO is economically selling YES at ``1 - price``. So:
          (YES, buy)  -> bid @ price          (NO, buy)  -> ask @ 1 - price
          (YES, sell) -> ask @ price          (NO, sell) -> bid @ 1 - price
        ``price`` is the cost/price of the requested ``side`` (dollars 0.01-0.99).

        ``post_only`` rests a MAKER order (rejected if it would cross), and
        ``expiration_ts`` (Unix seconds) auto-cancels it — so a maker that doesn't fill
        cleans itself up. A resting (unfilled) maker returns status ``RESTING``.
        Prices/counts are fixed-point dollar strings. Requires credentials.
        """
        if not self.authenticated:
            raise OrderNotPermitted("Kalshi credentials not configured")
        # Map (side, action) onto the YES-side book and the YES-side price.
        if side is Side.YES:
            book_side = "bid" if action == "buy" else "ask"
            yes_px = price
        else:
            book_side = "ask" if action == "buy" else "bid"
            yes_px = round(1.0 - price, 4)
        # Round to the cent (Kalshi tick) and clamp to a valid 0.01-0.99 quote.
        yes_px = min(max(round(yes_px, 2), 0.01), 0.99)
        count = float(contracts)
        # A maker rests, so it must be GTC (with an expiration to self-cancel).
        tif_v2 = "good_till_canceled" if post_only else self._TIF_V2.get(tif, "fill_or_kill")
        body: dict[str, Any] = {
            "ticker": market_id,
            "client_order_id": str(uuid.uuid4()),
            "side": book_side,                       # bid = buy YES, ask = sell YES
            "count": f"{count:.2f}",                 # contracts as a fixed-point string
            "price": f"{yes_px:.4f}",                # YES-side price, fixed-point dollars
            "time_in_force": tif_v2,
            "self_trade_prevention_type": "taker_at_cross",
            "post_only": bool(post_only),
        }
        if expiration_ts is not None:
            body["expiration_time"] = int(expiration_ts)

        endpoint = "/portfolio/events/orders"
        await self._limiter.wait()
        try:
            resp = await self._http().post(
                endpoint, json=body, headers=self._auth_headers("POST", endpoint),
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # 4xx -> REJECTED (no fill); else ERROR (unknown)
            from bot.execution.orders import order_error_result
            return order_error_result(VENUE, market_id, side, action, count, exc)

        filled = float(data.get("fill_count") or 0)
        avg = data.get("average_fill_price")
        avg_price = float(avg) if avg not in (None, "") else None
        # average_fill_price is the YES-side price; for a NO trade the cost is 1 - that.
        if avg_price is not None and side is Side.NO:
            avg_price = round(1.0 - avg_price, 4)
        order_id = data.get("order_id")
        if filled >= count - 1e-9 and count > 0:
            status = OrderStatus.FILLED
        elif post_only and filled <= 1e-9 and order_id:
            # Accepted maker, nothing filled yet -> resting on the book (NOT killed).
            status = OrderStatus.RESTING
        elif filled <= 1e-9:
            status = OrderStatus.KILLED
        else:
            status = OrderStatus.PARTIAL
        return OrderResult(
            venue=VENUE, market_id=market_id, side=side, action=action,
            requested=count, filled=filled, avg_price=avg_price,
            order_id=order_id, status=status, raw=data,
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
        body = resp.json()
        # Prefer the exact dollar string; fall back to the integer-cents field.
        bal_dollars = body.get("balance_dollars")
        bal_cents = body.get("balance")
        if bal_dollars not in (None, ""):
            balance = float(bal_dollars)
        elif bal_cents not in (None, ""):
            balance = float(bal_cents) / 100.0
        else:
            balance = None

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
