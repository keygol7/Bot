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

import asyncio
import base64
import logging
import time
from typing import Any, AsyncIterator

from bot.execution.orders import OrderResult, OrderStatus
from bot.fees import ZeroFeeModel
from bot.models import MarketQuote, Side
from bot.timeutil import parse_iso8601
from bot.venues.base import OrderNotPermitted, RawMarket
from bot.venues.ratelimit import AsyncRateLimiter

VENUE = "polymarket_us"
log = logging.getLogger("bot.venues.polymarket_us")


def parse_execution(message: dict[str, Any]):
    """Normalize a private-WS order update into a ``FillEvent`` (or ``None``)."""
    from bot.streaming.fills import FillEvent

    upd = message.get("orderSubscriptionUpdate")
    if not isinstance(upd, dict):
        return None
    ex = upd.get("execution") or {}
    order = ex.get("order") or {}
    order_id = order.get("id") or ex.get("id")
    if not order_id:
        return None
    etype = str(ex.get("type") or "").replace("EXECUTION_TYPE_", "")
    px = ex.get("lastPx") or {}
    last_px = float(px["value"]) if isinstance(px, dict) and px.get("value") not in (None, "") else None
    ls = ex.get("lastShares")
    last_shares = float(ls) if ls not in (None, "") else 0.0
    return FillEvent(VENUE, str(order_id), etype, last_shares, last_px)


def parse_market_data_lite(message: dict[str, Any]) -> MarketQuote | None:
    """Normalize a ``MARKET_DATA_LITE`` WS message into a top-of-book quote.

    The payload (`marketDataLite`) carries the same bestBid/bestAsk/askDepth/bidDepth
    shape as the REST BBO, so it reuses :func:`normalize_bbo`.
    """
    md = message.get("marketDataLite")
    if not isinstance(md, dict):
        return None
    slug = md.get("marketSlug")
    if not slug:
        return None
    return normalize_bbo(slug, "", md)


# (side, action) -> order intent. Price is always quoted on the YES/long side.
_INTENT = {
    (Side.YES, "buy"): "ORDER_INTENT_BUY_LONG",
    (Side.YES, "sell"): "ORDER_INTENT_SELL_LONG",
    (Side.NO, "buy"): "ORDER_INTENT_BUY_SHORT",
    (Side.NO, "sell"): "ORDER_INTENT_SELL_SHORT",
}
_TIF = {
    "fill_or_kill": "TIME_IN_FORCE_FILL_OR_KILL",
    "immediate_or_cancel": "TIME_IN_FORCE_IMMEDIATE_OR_CANCEL",
    "gtc": "TIME_IN_FORCE_GOOD_TILL_CANCEL",
}
_TERMINAL_FILLED = "ORDER_STATE_FILLED"
_REJECTED = {"ORDER_STATE_REJECTED"}
_KILLED = {"ORDER_STATE_CANCELED", "ORDER_STATE_EXPIRED"}


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


_BALANCE_KEYS = (
    "availableBalance", "available", "cashBalance", "cash", "buyingPower", "balance",
)
_POSITION_CONTAINER_KEYS = ("positions", "marketPositions", "market_positions")


def _account_balance(body: Any) -> float | None:
    """Parse available USD balance from a portfolio-balance payload (best-effort)."""
    if not isinstance(body, dict):
        return None
    sources = [body]
    nested = body.get("balance")
    if isinstance(nested, dict):
        sources.append(nested)   # e.g. {"balance": {"availableBalance": ...}}
    for src in sources:
        for key in _BALANCE_KEYS:
            if key in src and src[key] not in (None, ""):
                value = _amount(src[key])
                if value is not None:
                    return value
    return None


def _parse_positions(body: Any):
    """Parse non-flat positions/resting orders from a portfolio-positions payload.

    Raises ``ValueError`` on an unrecognized shape so the startup guard fails closed
    rather than reading an unparsed account as flat.
    """
    from bot.execution.account import VenuePosition

    if isinstance(body, list):
        rows = body
    elif isinstance(body, dict):
        rows = next(
            (body[k] for k in _POSITION_CONTAINER_KEYS if isinstance(body.get(k), list)),
            None,
        )
        if rows is None:
            raise ValueError(f"unrecognized positions payload keys: {sorted(body)}")
    else:
        raise ValueError("unrecognized positions payload")

    out = []
    for r in rows:
        slug = (r.get("marketSlug") or r.get("slug") or r.get("market_id")
                or r.get("ticker") or "")
        qty = _amount(r.get("quantity") or r.get("netQuantity") or r.get("size")
                      or r.get("position") or r.get("netSize")) or 0.0
        resting = int(r.get("openOrders") or r.get("restingOrders")
                      or r.get("resting_orders_count") or 0)
        pos = VenuePosition(slug, qty, resting)
        if pos.is_open:
            out.append(pos)
    return out


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
    suffix = f" - {outcome}"
    if outcome and not question.endswith(suffix):
        return question + suffix
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
        """Phase 1: paginate /v1/markets -> price-only quotes (bestBid/bestAsk).

        Sizes are not in the list payload, so this is the cheap wide scan; the sized
        quote comes from :meth:`fetch_quote` (BBO) for shortlisted markets only.

        ``limit`` is a TOTAL cap across pages. We page via ``offset`` and dedupe by
        slug; if the gateway ignores ``offset`` (returns the same page) we get no new
        slugs and stop, so this is safe whether or not paging is supported.
        """
        out: list[MarketQuote] = []
        seen: set[str] = set()
        offset = 0
        while len(out) < limit:
            await self._limiter.wait()
            page_size = min(limit - len(out), 500)
            resp = await self._gateway().get(
                "/v1/markets",
                params={"limit": page_size, "active": "true", "closed": "false",
                        "offset": offset},
            )
            resp.raise_for_status()
            markets = resp.json().get("markets", [])
            if not markets:
                break
            new = 0
            for m in markets:
                slug = m.get("slug") or m.get("id")
                if not slug or slug in seen:
                    continue
                seen.add(slug)
                new += 1
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
                        close_time=parse_iso8601(m.get("endDate")),
                    )
                )
            offset += len(markets)
            # No new slugs (offset ignored / end of feed), or a short page -> done.
            if new == 0 or len(markets) < page_size:
                break
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

    async def is_open(self, market_id: str):
        """Whether the market is still quotable. The gateway has no bare
        ``/v1/markets/{slug}`` detail endpoint (it 404s), so we use the ``/bbo``
        subpath that does exist: a 200 means the market is live (even if the book is
        momentarily empty). We NEVER return False here — only ``True`` (live) or
        ``None`` (unknown -> caller keeps it), so a gateway quirk can't wrongly drop a
        live market. Settled pairs are shed via the Kalshi leg's real status instead.
        """
        await self._limiter.wait()
        try:
            resp = await self._gateway().get(f"/v1/markets/{market_id}/bbo")
        except Exception:
            return None
        return True if resp.status_code == 200 else None

    async def stream_order_book(self, market_ids: list[str]) -> AsyncIterator[MarketQuote]:
        """Stream real-time top-of-book via the markets WS (MARKET_DATA_LITE).

        Requires credentials (the markets WS is on the authenticated API, unlike the
        public REST gateway). Subscribes in batches of 100 slugs; reconnects with
        exponential backoff; ignores heartbeats.
        """
        if not getattr(self.cfg, "is_trading_configured", False):
            raise OrderNotPermitted("Polymarket US credentials required for the WebSocket")
        import json

        import websockets  # lazy

        path = "/v1/ws/markets"
        backoff = 1.0
        while True:
            try:
                headers = self._auth_headers("GET", path)
                async with websockets.connect(
                    self.cfg.ws_markets, additional_headers=headers, open_timeout=10
                ) as ws:
                    chunks = (
                        [market_ids[i : i + 100] for i in range(0, len(market_ids), 100)]
                        if market_ids else [None]
                    )
                    for n, chunk in enumerate(chunks):
                        sub: dict[str, Any] = {
                            "requestId": f"md-{n}",
                            "subscriptionType": "SUBSCRIPTION_TYPE_MARKET_DATA_LITE",
                        }
                        if chunk:
                            sub["marketSlugs"] = chunk
                        await ws.send(json.dumps({"subscribe": sub}))
                    backoff = 1.0
                    async for raw in ws:
                        data = json.loads(raw)
                        if "heartbeat" in data:
                            continue
                        if data.get("error"):
                            log.warning("polymarket ws error: %s", data["error"])
                            continue
                        quote = parse_market_data_lite(data)
                        if quote is not None:
                            yield quote
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("polymarket ws disconnected (%s); reconnecting in %.0fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    def _api(self):
        """HTTP client for the authenticated trading API."""
        if self._api_client is None:
            import httpx  # lazy

            self._api_client = httpx.AsyncClient(base_url=self.cfg.api_base, timeout=10.0)
        return self._api_client

    async def stream_private(self):
        """Stream private order executions (FillEvents) from /v1/ws/private."""
        if not getattr(self.cfg, "is_trading_configured", False):
            raise OrderNotPermitted("Polymarket US credentials required for the private WS")
        import json

        import websockets  # lazy

        path = "/v1/ws/private"
        backoff = 1.0
        while True:
            try:
                headers = self._auth_headers("GET", path)
                async with websockets.connect(
                    self.cfg.ws_private, additional_headers=headers, open_timeout=10
                ) as ws:
                    await ws.send(json.dumps({"subscribe": {
                        "requestId": "ord", "subscriptionType": "SUBSCRIPTION_TYPE_ORDER"}}))
                    backoff = 1.0
                    async for raw in ws:
                        data = json.loads(raw)
                        if "heartbeat" in data:
                            continue
                        ev = parse_execution(data)
                        if ev is not None:
                            yield ev
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("polymarket private ws disconnected (%s); reconnecting in %.0fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def place_order(
        self, market_id: str, side: Side, action: str, price: float, contracts: float,
        *, tif: str = "fill_or_kill",
    ) -> OrderResult:
        """Place an order via POST /v1/orders (X-PM Ed25519 auth).

        ``price`` is the cost/value of the requested ``side``; the API always wants the
        YES/long price, so for NO we send ``1 - price``. ``manualOrderIndicator`` is
        AUTOMATIC (bot, regulatory). NOTE: confirm the create-response fill fields
        against the sandbox before live use.
        """
        if not getattr(self.cfg, "is_trading_configured", False):
            raise OrderNotPermitted("Polymarket US trading credentials not configured")
        yes_value = price if side is Side.YES else round(1.0 - price, 6)
        yes_value = min(max(yes_value, 0.01), 0.99)
        body = {
            "marketSlug": market_id,
            "type": "ORDER_TYPE_LIMIT",
            "price": {"value": f"{yes_value}", "currency": "USD"},
            "quantity": contracts,
            "tif": _TIF.get(tif, "TIME_IN_FORCE_FILL_OR_KILL"),
            "intent": _INTENT[(side, action)],
            "manualOrderIndicator": "MANUAL_ORDER_INDICATOR_AUTOMATIC",
            "synchronousExecution": True,
        }
        await self._limiter.wait()
        try:
            resp = await self._api().post(
                "/v1/orders", json=body, headers=self._auth_headers("POST", "/v1/orders")
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            return OrderResult(VENUE, market_id, side, action, contracts,
                               status=OrderStatus.ERROR, raw={"error": str(exc)})

        order = data.get("order", data)
        state = order.get("state") or order.get("orderState") or ""
        filled = float(order.get("cumQuantity") or order.get("filledQuantity") or 0)
        avg = order.get("avgPx", {})
        avg_price = None
        if isinstance(avg, dict) and avg.get("value") not in (None, ""):
            avg_price = float(avg["value"])
        if state == _TERMINAL_FILLED or filled >= contracts - 1e-9:
            status = OrderStatus.FILLED
        elif state in _REJECTED:
            status = OrderStatus.REJECTED
        elif filled <= 1e-9:
            status = OrderStatus.KILLED
        else:
            status = OrderStatus.PARTIAL
        return OrderResult(
            venue=VENUE, market_id=market_id, side=side, action=action,
            requested=contracts, filled=filled, avg_price=avg_price,
            order_id=order.get("id") or order.get("orderId"), status=status, raw=order,
        )

    async def cancel_order(self, order_id: str) -> dict:
        if not getattr(self.cfg, "is_trading_configured", False):
            raise OrderNotPermitted("Polymarket US trading credentials not configured")
        path = f"/v1/order/{order_id}/cancel"
        await self._limiter.wait()
        resp = await self._api().post(path, headers=self._auth_headers("POST", path))
        resp.raise_for_status()
        return resp.json()

    async def get_positions(self) -> dict:
        raise NotImplementedError("positions endpoint lands with the live phase")

    async def account_snapshot(self):
        """Balance + non-flat positions/resting orders, for the startup guard.

        Uses the authenticated ``/v1/portfolio/balance`` and
        ``/v1/portfolio/positions`` endpoints. Payload field names are parsed
        defensively; an unrecognized positions shape RAISES (so the guard fails
        closed and never mistakes an unparsed account for a flat one). The first
        live run logs the snapshot so the exact shape can be confirmed.
        """
        from bot.execution.account import AccountSnapshot

        if not getattr(self.cfg, "is_trading_configured", False):
            raise OrderNotPermitted("Polymarket US trading credentials not configured")

        await self._limiter.wait()
        path = "/v1/portfolio/balance"
        resp = await self._api().get(path, headers=self._auth_headers("GET", path))
        resp.raise_for_status()
        balance = _account_balance(resp.json())

        await self._limiter.wait()
        path = "/v1/portfolio/positions"
        resp = await self._api().get(path, headers=self._auth_headers("GET", path))
        resp.raise_for_status()
        positions = _parse_positions(resp.json())
        return AccountSnapshot(self.name, balance, positions)

    async def aclose(self) -> None:
        for attr in ("_gateway_client", "_api_client"):
            client = getattr(self, attr)
            if client is not None:
                await client.aclose()
                setattr(self, attr, None)
