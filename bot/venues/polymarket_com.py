"""Polymarket.com (international) CLOB V2 venue adapter.

This is a separate venue from Polymarket US/QCEX.  Discovery uses Gamma, books and
orders use the CLOB, positions use the Data API, and orders are signed locally by
the official ``py-clob-client-v2`` SDK. Optional slug and close-time filters can narrow
discovery, while the default scans the full open binary market universe.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

from bot.execution.orders import OrderResult, OrderStatus
from bot.fees import PolymarketComFeeModel
from bot.models import MarketQuote, Side
from bot.timeutil import parse_iso8601
from bot.venues.base import OrderNotPermitted, RawMarket
from bot.venues.ratelimit import AsyncRateLimiter

VENUE = "polymarket_com"
log = logging.getLogger("bot.venues.polymarket_com")

_MARKET_FIELDS = (
    "slug", "conditionId", "condition_id", "clobTokenIds", "outcomes",
    "orderPriceMinTickSize", "orderMinSize", "negRisk", "volume24hr",
    "endDate", "question", "title", "description", "bestAsk", "bestBid",
    "acceptingOrders", "closed", "active",
)


def _json_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except (TypeError, ValueError):
            return []
    return []


def _compact_market(market: dict[str, Any]) -> dict[str, Any]:
    """Drop Gamma's large nested event/series payload before retaining a row."""
    return {key: market[key] for key in _MARKET_FIELDS if key in market}


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _levels(value: Any, *, reverse: bool = False) -> tuple[tuple[float, float], ...]:
    out: list[tuple[float, float]] = []
    for level in value or []:
        if isinstance(level, dict):
            price, size = _number(level.get("price")), _number(level.get("size"))
        else:
            price = _number(getattr(level, "price", None))
            size = _number(getattr(level, "size", None))
        if price is not None and size is not None and 0 <= price <= 1 and size > 0:
            out.append((price, size))
    return tuple(sorted(out, key=lambda x: x[0], reverse=reverse))


def _market_title(market: dict[str, Any]) -> str:
    question = str(market.get("question") or market.get("title") or market.get("slug") or "")
    outcomes = _json_list(market.get("outcomes"))
    # The bot's YES side means the first CLOB outcome.  Make that proposition explicit
    # for Up/Down contracts so matching/rules checks do not mistake the neutral question
    # for a literal YES/NO label.
    if outcomes and str(outcomes[0]).lower() not in ("yes", "no"):
        return f"{question} - {outcomes[0]}"
    return question


def _market_meta(market: dict[str, Any]) -> dict[str, Any] | None:
    slug = market.get("slug")
    condition = market.get("conditionId") or market.get("condition_id")
    tokens = [str(x) for x in _json_list(market.get("clobTokenIds"))]
    outcomes = [str(x) for x in _json_list(market.get("outcomes"))]
    if not slug or not condition or len(tokens) != 2 or len(outcomes) != 2:
        return None
    return {
        "slug": str(slug),
        "condition_id": str(condition),
        "tokens": (tokens[0], tokens[1]),
        "outcomes": (outcomes[0], outcomes[1]),
        "tick": _number(market.get("orderPriceMinTickSize")) or 0.01,
        "min_size": _number(market.get("orderMinSize")) or 0.0,
        "neg_risk": bool(market.get("negRisk")),
        "volume24hr": _number(market.get("volume24hr")),
        "close_time": parse_iso8601(market.get("endDate")),
        "title": _market_title(market),
        "description": str(market.get("description") or ""),
    }


def normalize_market(market: dict[str, Any]) -> MarketQuote | None:
    """Normalize Gamma's first-outcome BBO into a phase-one quote."""
    meta = _market_meta(market)
    if meta is None:
        return None
    ask = _number(market.get("bestAsk"))
    bid = _number(market.get("bestBid"))
    return MarketQuote(
        venue=VENUE,
        market_id=meta["slug"],
        title=meta["title"],
        yes_ask=ask,
        no_ask=(round(1.0 - bid, 6) if bid is not None else None),
        close_time=meta["close_time"],
        state=("MARKET_STATE_OPEN" if market.get("acceptingOrders") is not False
               else "MARKET_STATE_CLOSED"),
    )


def normalize_books(
    market_id: str,
    title: str,
    yes_book: Any,
    no_book: Any,
    *,
    close_time: float | None = None,
) -> MarketQuote:
    """Combine the two outcome-token books into one binary ``MarketQuote``."""
    ya = _levels(yes_book.get("asks") if isinstance(yes_book, dict)
                 else getattr(yes_book, "asks", None))
    na = _levels(no_book.get("asks") if isinstance(no_book, dict)
                 else getattr(no_book, "asks", None))
    timestamps = []
    for book in (yes_book, no_book):
        value = book.get("timestamp") if isinstance(book, dict) else getattr(book, "timestamp", None)
        ts = _number(value)
        if ts is not None:
            timestamps.append(ts / 1000.0 if ts > 10_000_000_000 else ts)
    return MarketQuote(
        venue=VENUE,
        market_id=market_id,
        title=title,
        yes_ask=ya[0][0] if ya else None,
        yes_ask_size=ya[0][1] if ya else 0.0,
        no_ask=na[0][0] if na else None,
        no_ask_size=na[0][1] if na else 0.0,
        yes_ask_levels=ya,
        no_ask_levels=na,
        # A binary quote combines two independently streamed token books.  It is only
        # as fresh as the older side; using the newer timestamp can make a stale
        # opposite-outcome ask look synchronized and manufacture an edge.
        exchange_ts=min(timestamps) if timestamps else None,
        timestamp=time.time(),
        close_time=close_time,
        state="MARKET_STATE_OPEN",
    )


def parse_order_response(
    data: Any,
    requested: float,
    price: float,
    *,
    resting: bool = False,
) -> tuple[OrderStatus, float, float | None, str | None]:
    if not isinstance(data, dict):
        return OrderStatus.ERROR, 0.0, None, None
    order_id = data.get("orderID") or data.get("order_id") or data.get("id")
    status = str(data.get("status") or "").lower()
    error = str(data.get("errorMsg") or data.get("error") or "").lower()
    if status == "matched":
        return OrderStatus.FILLED, requested, price, str(order_id) if order_id else None
    if resting and status in ("live", "delayed") and order_id:
        return OrderStatus.RESTING, 0.0, None, str(order_id)
    if status in ("unmatched", "canceled", "cancelled", "expired"):
        return OrderStatus.KILLED, 0.0, None, str(order_id) if order_id else None
    if "fok" in error and ("fill" in error or "match" in error):
        return OrderStatus.KILLED, 0.0, None, str(order_id) if order_id else None
    if data.get("success") is False or error:
        return OrderStatus.REJECTED, 0.0, None, str(order_id) if order_id else None
    # A marketable crypto order may briefly report delayed.  It is ambiguous until an
    # authoritative order read says how much matched, so never guess that it filled.
    return OrderStatus.ERROR, 0.0, None, str(order_id) if order_id else None


def parse_private_order(message: dict[str, Any], previous: dict[str, float]):
    """Normalize cumulative user-channel order updates into delta ``FillEvent``s."""
    from bot.streaming.fills import FillEvent

    if str(message.get("event_type") or "").lower() != "order":
        return None
    order_id = message.get("id") or message.get("order_id")
    if not order_id:
        return None
    cumulative = _number(message.get("size_matched")) or 0.0
    old = previous.get(str(order_id), 0.0)
    previous[str(order_id)] = max(old, cumulative)
    delta = max(0.0, cumulative - old)
    original = _number(message.get("original_size")) or 0.0
    kind = str(message.get("type") or "").upper()
    if original > 0 and cumulative >= original - 1e-9:
        event_type = "FILL"
    elif kind in ("CANCELLATION", "CANCELED", "CANCELLED"):
        event_type = "CANCELED"
    elif delta > 0:
        event_type = "PARTIAL_FILL"
    else:
        return None
    return FillEvent(VENUE, str(order_id), event_type, delta, _number(message.get("price")))


class PolymarketComVenue:
    name = VENUE

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.fee_model = PolymarketComFeeModel()
        self._public_client = None
        self._sdk_client = None
        self._limiter = AsyncRateLimiter(getattr(cfg, "read_rate_per_min", 600.0))
        self._order_limiter = AsyncRateLimiter(600.0, burst=30)
        self._meta: dict[str, dict[str, Any]] = {}
        self._token_to_market: dict[str, tuple[str, Side]] = {}
        self._condition_to_market: dict[str, str] = {}
        self._private_cum: dict[str, float] = {}

    @property
    def is_trading_configured(self) -> bool:
        return bool(getattr(self.cfg, "is_trading_configured", False))

    def _public(self):
        if self._public_client is None:
            import httpx

            self._public_client = httpx.AsyncClient(timeout=10.0)
        return self._public_client

    def _private_key(self) -> str:
        value = (getattr(self.cfg, "private_key", "") or "").strip()
        if not value:
            path = Path(getattr(self.cfg, "private_key_path", "")).expanduser()
            if path.exists():
                value = path.read_text().strip()
        if not value:
            raise OrderNotPermitted("Polymarket.com private key not configured")
        return value

    def _sdk(self):
        if self._sdk_client is None:
            if not self.is_trading_configured:
                raise OrderNotPermitted("Polymarket.com trading credentials not configured")
            try:
                from py_clob_client_v2 import ApiCreds, ClobClient
            except ImportError as exc:  # pragma: no cover - deployment configuration
                raise RuntimeError(
                    "install the 'venues' extra (py-clob-client-v2 is required)"
                ) from exc
            creds = ApiCreds(
                api_key=self.cfg.api_key,
                api_secret=self.cfg.api_secret,
                api_passphrase=self.cfg.api_passphrase,
            )
            self._sdk_client = ClobClient(
                host=self.cfg.clob_base,
                chain_id=self.cfg.chain_id,
                key=self._private_key(),
                creds=creds,
                signature_type=self.cfg.signature_type,
                funder=self.cfg.funder_address,
                retry_on_error=False,
            )
        return self._sdk_client

    def _remember(self, market: dict[str, Any]) -> dict[str, Any] | None:
        meta = _market_meta(market)
        if meta is None:
            return None
        self._meta[meta["slug"]] = meta
        self._condition_to_market[meta["condition_id"]] = meta["slug"]
        self._token_to_market[meta["tokens"][0]] = (meta["slug"], Side.YES)
        self._token_to_market[meta["tokens"][1]] = (meta["slug"], Side.NO)
        return meta

    def _included(self, market: dict[str, Any], now: float, end: float | None) -> bool:
        slug = str(market.get("slug") or "").lower()
        prefixes = tuple(
            str(p).strip().lower()
            for p in (getattr(self.cfg, "market_slug_prefixes", ()) or ())
            if str(p).strip()
        )
        close = parse_iso8601(market.get("endDate"))
        in_scope = not prefixes or "*" in prefixes or any(
            slug.startswith(p) for p in prefixes
        )
        in_window = (
            (close is None or close >= now - 30)
            if end is None
            else close is not None and now - 30 <= close <= end
        )
        return bool(
            slug
            and in_scope
            and market.get("active") is not False
            and market.get("closed") is not True
            and market.get("acceptingOrders") is not False
            and in_window
        )

    async def _gamma_markets(self, limit: int = 0, max_close_ts: int | None = None) -> list[dict]:
        now = time.time()
        lookahead = max(0.0, float(self.cfg.lookahead_minutes))
        ends = []
        if lookahead > 0:
            ends.append(now + lookahead * 60.0)
        if max_close_ts is not None:
            ends.append(float(max_close_ts))
        end = min(ends) if ends else None
        if end is not None and end <= now:
            return []
        # Gamma currently caps this endpoint at 100 rows even when a larger limit is
        # requested. Using the real cap is important: otherwise a 100-row response looks
        # like the final page and silently truncates full-board discovery.
        page_size = 100
        cursor: str | None = None
        found: list[dict] = []
        while True:
            await self._limiter.wait()
            params: dict[str, Any] = {
                "closed": "false",
                "limit": page_size,
                "order": "id",
                "ascending": "false",
            }
            if cursor:
                params["after_cursor"] = cursor
            if end is not None:
                params["end_date_min"] = datetime.fromtimestamp(
                    now - 30, timezone.utc
                ).isoformat()
                params["end_date_max"] = datetime.fromtimestamp(
                    end, timezone.utc
                ).isoformat()
            resp = await self._public().get(
                f"{self.cfg.gamma_base}/markets/keyset", params=params
            )
            resp.raise_for_status()
            body = resp.json()
            if not isinstance(body, dict) or not isinstance(body.get("markets"), list):
                raise ValueError("unexpected Gamma keyset markets response")
            page = body["markets"]
            for market in page:
                if isinstance(market, dict) and self._included(market, now, end):
                    found.append(_compact_market(market))
                    if limit > 0 and len(found) >= limit:
                        return found[:limit]
            next_cursor = body.get("next_cursor")
            if not next_cursor or next_cursor == cursor:
                break
            cursor = str(next_cursor)
        return found

    async def list_markets(self, limit: int = 200) -> list[RawMarket]:
        markets = await self._gamma_markets(limit)
        out = []
        for market in markets:
            meta = self._remember(market)
            if meta:
                out.append(RawMarket(meta["slug"], meta["title"], market))
        return out

    async def scan_quotes(self, limit: int = 200, *, max_close_ts: int | None = None, **_) -> list[MarketQuote]:
        markets = await self._gamma_markets(limit, max_close_ts)
        out = []
        for market in markets:
            if self._remember(market):
                quote = normalize_market(market)
                if quote is not None:
                    out.append(quote)
        return out

    async def _ensure_meta(self, market_id: str) -> dict[str, Any] | None:
        if market_id in self._meta:
            return self._meta[market_id]
        await self._limiter.wait()
        resp = await self._public().get(f"{self.cfg.gamma_base}/markets/slug/{market_id}")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        body = resp.json()
        return self._remember(body) if isinstance(body, dict) else None

    async def fetch_quote(self, market: RawMarket) -> MarketQuote | None:
        meta = self._remember(market.raw) if market.raw else None
        meta = meta or await self._ensure_meta(market.market_id)
        if meta is None:
            return None
        await self._limiter.wait()
        yes, no = await asyncio.gather(
            asyncio.to_thread(self._public_book, meta["tokens"][0]),
            asyncio.to_thread(self._public_book, meta["tokens"][1]),
        )
        return normalize_books(
            meta["slug"], meta["title"], yes, no, close_time=meta["close_time"]
        )

    def _public_book(self, token_id: str):
        # The official SDK's L0 book method needs no key and knows the current V2 schema.
        try:
            from py_clob_client_v2 import ClobClient
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("py-clob-client-v2 is required for Polymarket.com") from exc
        return ClobClient(host=self.cfg.clob_base, chain_id=self.cfg.chain_id).get_order_book(token_id)

    async def is_open(self, market_id: str) -> bool | None:
        meta = await self._ensure_meta(market_id)
        if meta is None:
            return None
        await self._limiter.wait()
        resp = await self._public().get(f"{self.cfg.gamma_base}/markets/slug/{market_id}")
        if resp.status_code == 404:
            return False
        resp.raise_for_status()
        body = resp.json()
        return bool(
            isinstance(body, dict)
            and body.get("closed") is not True
            and body.get("acceptingOrders") is not False
        )

    async def stream_order_book(self, market_ids: list[str]) -> AsyncIterator[MarketQuote]:
        import websockets

        metas = [m for mid in market_ids if (m := await self._ensure_meta(mid)) is not None]
        assets = [token for m in metas for token in m["tokens"]]
        if not assets:
            return
        books: dict[str, dict[str, dict[float, float]]] = {}
        while True:
            ping_task = None
            try:
                async with websockets.connect(self.cfg.ws_market, ping_interval=None) as ws:
                    await ws.send(json.dumps({
                        "assets_ids": assets,
                        "type": "market",
                        "custom_feature_enabled": True,
                    }))

                    async def ping_loop():
                        while True:
                            await asyncio.sleep(10)
                            await ws.send("PING")

                    ping_task = asyncio.create_task(ping_loop())
                    async for raw in ws:
                        if raw == "PONG":
                            continue
                        payload = json.loads(raw)
                        messages = payload if isinstance(payload, list) else [payload]
                        for msg in messages:
                            if not isinstance(msg, dict):
                                continue
                            changed_slugs = self._apply_book_message(books, msg)
                            if not changed_slugs:
                                continue
                            for slug in changed_slugs:
                                meta = self._meta.get(slug)
                                if meta is None:
                                    continue
                                yes_token, no_token = meta["tokens"]
                                if yes_token not in books or no_token not in books:
                                    continue
                                yes_book = self._ws_book(yes_token, books[yes_token])
                                no_book = self._ws_book(no_token, books[no_token])
                                yield normalize_books(
                                    slug, meta["title"], yes_book, no_book,
                                    close_time=meta["close_time"],
                                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("Polymarket.com market WS disconnected: %s", exc)
                await asyncio.sleep(1.0)
            finally:
                if ping_task:
                    ping_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await ping_task

    def _apply_book_message(self, books: dict, msg: dict) -> set[str]:
        event = str(msg.get("event_type") or "").lower()
        changes = msg.get("price_changes") if event == "price_change" else [msg]
        changed_slugs: set[str] = set()
        message_ts = msg.get("timestamp")
        for item in changes or []:
            token = str(item.get("asset_id") or "")
            mapping = self._token_to_market.get(token)
            if mapping is None:
                continue
            if event == "book":
                books[token] = {
                    "bids": dict(_levels(item.get("bids"), reverse=True)),
                    "asks": dict(_levels(item.get("asks"))),
                    "timestamp": item.get("timestamp", message_ts),
                }
            elif event == "price_change":
                book = books.setdefault(token, {"bids": {}, "asks": {}, "timestamp": None})
                price, size = _number(item.get("price")), _number(item.get("size"))
                side = str(item.get("side") or "").upper()
                if price is None or size is None or side not in ("BUY", "SELL"):
                    continue
                ladder = book["bids" if side == "BUY" else "asks"]
                if size <= 0:
                    ladder.pop(price, None)
                else:
                    ladder[price] = size
                book["timestamp"] = item.get("timestamp", message_ts)
            else:
                continue
            changed_slugs.add(mapping[0])
        return changed_slugs

    @staticmethod
    def _ws_book(token: str, book: dict) -> dict:
        return {
            "asset_id": token,
            "timestamp": book.get("timestamp"),
            "bids": [{"price": p, "size": s} for p, s in book["bids"].items()],
            "asks": [{"price": p, "size": s} for p, s in book["asks"].items()],
        }

    async def stream_private(self):
        import websockets

        if not self.is_trading_configured:
            raise OrderNotPermitted("Polymarket.com trading credentials not configured")
        while True:
            ping_task = None
            try:
                async with websockets.connect(self.cfg.ws_user, ping_interval=None) as ws:
                    await ws.send(json.dumps({
                        "auth": {
                            "apiKey": self.cfg.api_key,
                            "secret": self.cfg.api_secret,
                            "passphrase": self.cfg.api_passphrase,
                        },
                        "type": "user",
                    }))

                    async def ping_loop():
                        while True:
                            await asyncio.sleep(10)
                            await ws.send("PING")

                    ping_task = asyncio.create_task(ping_loop())
                    async for raw in ws:
                        if raw == "PONG":
                            continue
                        payload = json.loads(raw)
                        for msg in payload if isinstance(payload, list) else [payload]:
                            if isinstance(msg, dict):
                                event = parse_private_order(msg, self._private_cum)
                                if event is not None:
                                    yield event
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("Polymarket.com user WS disconnected: %s", exc)
                await asyncio.sleep(1.0)
            finally:
                if ping_task:
                    ping_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await ping_task

    async def place_order(
        self,
        market_id: str,
        side: Side,
        action: str,
        price: float,
        contracts: float,
        *,
        tif: str = "fill_or_kill",
        post_only: bool = False,
        expiration_ts: int | None = None,
    ) -> OrderResult:
        if not self.is_trading_configured:
            raise OrderNotPermitted("Polymarket.com trading credentials not configured")
        meta = await self._ensure_meta(market_id)
        if meta is None:
            return OrderResult(VENUE, market_id, side, action, contracts,
                               status=OrderStatus.REJECTED,
                               raw={"error": "unknown market"})
        if contracts + 1e-9 < meta["min_size"]:
            return OrderResult(
                VENUE, market_id, side, action, contracts,
                status=OrderStatus.REJECTED,
                raw={"error": f"below market minimum size {meta['min_size']}"},
            )
        token = meta["tokens"][0 if side is Side.YES else 1]
        await self._order_limiter.wait()
        try:
            from py_clob_client_v2 import OrderArgs, OrderType, PartialCreateOrderOptions
            from py_clob_client_v2.order_builder.constants import BUY, SELL

            order_type = OrderType.GTC if post_only or tif == "gtc" else OrderType.FOK
            args = OrderArgs(
                token_id=token,
                price=price,
                size=contracts,
                side=BUY if action == "buy" else SELL,
                expiration=(expiration_ts or 0),
            )
            options = PartialCreateOrderOptions(
                tick_size=str(meta["tick"]),
                neg_risk=meta["neg_risk"],
            )
            data = await asyncio.to_thread(
                self._sdk().create_and_post_order,
                args,
                options,
                order_type,
                post_only,
            )
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            status = (OrderStatus.REJECTED if isinstance(status_code, int)
                      and 400 <= status_code < 500 else OrderStatus.ERROR)
            return OrderResult(
                VENUE, market_id, side, action, contracts, status=status,
                raw={"status_code": status_code, "error": str(exc)},
            )
        status, filled, avg, order_id = parse_order_response(
            data, contracts, price, resting=(order_type == OrderType.GTC)
        )
        if order_id and status is OrderStatus.ERROR:
            resolved = await self._get_order_result(order_id, market_id, side, action, contracts)
            if resolved is not None:
                return resolved
        return OrderResult(
            VENUE, market_id, side, action, contracts,
            filled=filled, avg_price=avg, order_id=order_id, status=status,
            raw=data if isinstance(data, dict) else {"response": data},
        )

    async def _get_order_result(
        self, order_id: str, market_id: str, side: Side, action: str, requested: float
    ) -> OrderResult | None:
        try:
            data = await asyncio.to_thread(self._sdk().get_order, order_id)
        except Exception as exc:
            log.warning("Polymarket.com get-order %s failed: %s", order_id, exc)
            return None
        if not isinstance(data, dict):
            return None
        filled = _number(data.get("size_matched")) or 0.0
        original = _number(data.get("original_size")) or requested
        raw_status = str(data.get("status") or "").upper()
        if filled >= requested - 1e-9:
            status = OrderStatus.FILLED
        elif filled > 1e-9:
            status = OrderStatus.PARTIAL
        elif raw_status in ("ORDER_STATUS_CANCELED", "CANCELED", "CANCELLED", "EXPIRED"):
            status = OrderStatus.KILLED
        elif raw_status in ("ORDER_STATUS_LIVE", "LIVE"):
            status = OrderStatus.RESTING
        elif raw_status in ("ORDER_STATUS_REJECTED", "REJECTED"):
            status = OrderStatus.REJECTED
        else:
            status = OrderStatus.ERROR
        return OrderResult(
            VENUE, market_id, side, action, original,
            filled=filled, avg_price=_number(data.get("price")), order_id=order_id,
            status=status, raw=data,
        )

    async def order_filled_qty(self, order_id: str, requested: float, side: Side) -> float | None:
        res = await self._get_order_result(order_id, "", side, "buy", requested)
        return None if res is None else res.filled

    async def cancel_order(self, order_id: str) -> dict:
        from py_clob_client_v2 import OrderPayload

        await self._order_limiter.wait()
        return await asyncio.to_thread(self._sdk().cancel_order, OrderPayload(order_id))

    async def market_rules(self, market_id: str) -> str | None:
        meta = await self._ensure_meta(market_id)
        if meta is None:
            return None
        return meta.get("description") or None

    @staticmethod
    def _collateral_balance(value: Any) -> float | None:
        if value in (None, ""):
            return None
        text = str(value)
        parsed = _number(value)
        if parsed is None:
            return None
        # CLOB balance responses use the collateral token's six-decimal base units.
        # A decimal response is already human-readable (accepted for forward compatibility).
        return parsed if "." in text else parsed / 1_000_000.0

    async def account_snapshot(self):
        from bot.execution.account import AccountSnapshot, VenuePosition
        from py_clob_client_v2 import AssetType, BalanceAllowanceParams

        if not self.is_trading_configured:
            raise OrderNotPermitted("Polymarket.com trading credentials not configured")
        balance_body, open_orders = await asyncio.gather(
            asyncio.to_thread(
                self._sdk().get_balance_allowance,
                BalanceAllowanceParams(asset_type=AssetType.COLLATERAL),
            ),
            asyncio.to_thread(self._sdk().get_open_orders),
        )
        balance = self._collateral_balance(
            balance_body.get("balance") if isinstance(balance_body, dict) else None
        )
        await self._limiter.wait()
        resp = await self._public().get(
            f"{self.cfg.data_base}/positions",
            params={"user": self.cfg.funder_address, "sizeThreshold": 0, "limit": 500},
        )
        resp.raise_for_status()
        body = resp.json()
        positions: dict[str, VenuePosition] = {}
        for row in body if isinstance(body, list) else []:
            size = _number(row.get("size")) or 0.0
            if abs(size) <= 1e-9:
                continue
            market_id = str(row.get("slug") or row.get("conditionId") or "")
            positions[market_id] = VenuePosition(
                market_id,
                quantity=size,
                cost=abs(_number(row.get("initialValue")) or 0.0),
            )
        for row in open_orders if isinstance(open_orders, list) else []:
            condition = str(row.get("market") or "")
            market_id = self._condition_to_market.get(condition, condition or "(unknown)")
            pos = positions.setdefault(market_id, VenuePosition(market_id))
            pos.resting_orders += 1
        return AccountSnapshot(VENUE, balance, list(positions.values()))

    async def aclose(self) -> None:
        if self._public_client is not None:
            await self._public_client.aclose()
            self._public_client = None
