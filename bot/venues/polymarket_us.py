"""Polymarket US / QCEX venue adapter.

Market data comes from the PUBLIC gateway (`gateway.polymarket.us`) — no auth:
  - ``GET /v1/markets``            list markets (each carries bestBid/bestAsk)
  - ``GET /v1/markets/{slug}/bbo`` best bid/offer + depth for one market

Binary mapping (prices in dollars, 0..1): ``bestAsk`` is the cost to buy YES; the
NO ask is ``1 - bestBid`` (buying NO == taking the YES bid). NOTE: ``askDepth`` /
``bidDepth`` in the BBO/lite payload are the NUMBER OF PRICE LEVELS, not contract
sizes, so real takeable size comes from the full ``/book`` (``offers``/``bids`` with
``qty``) — see :func:`normalize_book`. Taker trades pay ``0.05 * C * p * (1-p)``
(published schedule, eff. 2026-04-03) -> :class:`PolymarketUSFeeModel`.

Trading (later) uses the authenticated API (`api.polymarket.us`) with X-PM-* Ed25519
headers — see :func:`build_auth_headers` / :func:`load_ed25519_key`. Read-only here.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import math
import re
import time
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from bot.execution.orders import OrderResult, OrderStatus
from bot.fees import PolymarketUSFeeModel
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
    # lastPx is the YES/long-side price; for a SHORT (NO) order convert to the NO cost.
    intent = str(order.get("intent") or ex.get("intent") or "")
    if last_px is not None and "SHORT" in intent.upper():
        last_px = round(1.0 - last_px, 6)
    ls = ex.get("lastShares")
    last_shares = float(ls) if ls not in (None, "") else 0.0
    return FillEvent(VENUE, str(order_id), etype, last_shares, last_px)


def parse_market_data_lite(message: dict[str, Any]) -> MarketQuote | None:
    """Normalize a ``MARKET_DATA_LITE`` WS message into a top-of-book quote.

    The payload (`marketDataLite`) carries the same bestBid/bestAsk/askDepth/bidDepth
    shape as the REST BBO, so it reuses :func:`normalize_bbo` (price-only — the lite
    feed's depth fields are level counts, not sizes).
    """
    md = message.get("marketDataLite")
    if not isinstance(md, dict):
        return None
    slug = md.get("marketSlug")
    if not slug:
        return None
    return normalize_bbo(slug, "", md)


def parse_market_data(message: dict[str, Any]) -> MarketQuote | None:
    """Normalize a ``MARKET_DATA`` (full book) WS message into a SIZED quote.

    The payload (`marketData`) carries the same ``bids``/``offers`` (with real ``qty``)
    shape as the REST ``/book``, so it reuses :func:`normalize_book` — giving real
    top-of-book size over the stream (unlike the lite feed's level counts).
    """
    md = message.get("marketData")
    if not isinstance(md, dict):
        return None
    slug = md.get("marketSlug")
    if not slug:
        return None
    q = normalize_book(slug, "", md)
    q.timestamp = time.time()   # stamp WS arrival so the engine can gate on freshness
    return q


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


_SLUG_DATE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def _date_from_slug(slug: str) -> float | None:
    """Game date embedded in a per-game slug (e.g. ``aec-mlb-kc-cws-2026-06-26``) as a UTC
    timestamp, or None. Fallback for the matcher's resolve-date guard when endDate is null."""
    m = _SLUG_DATE.search(slug or "")
    if not m:
        return None
    try:
        return datetime(int(m[1]), int(m[2]), int(m[3]), tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


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


def _price_str(value: float) -> str:
    """Format a price as the API's ``Amount.value`` — a plain decimal string. Guards
    against float-repr artifacts and scientific notation (e.g. ``1e-06``) that a bare
    ``f"{x}"`` can emit, which the exchange would reject as a malformed price."""
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"


_BALANCE_KEYS = (
    "buyingPower", "currentBalance", "availableBalance", "available",
    "cashBalance", "cash", "balance",
)
_POSITION_CONTAINER_KEYS = ("positions", "availablePositions", "marketPositions",
                            "market_positions")


def _usd_balance_entry(body: Any) -> dict | None:
    """The USD entry from a ``/v1/account/balances`` payload (``{"balances": [...]}``)."""
    if not isinstance(body, dict):
        return None
    entries = body.get("balances")
    if not isinstance(entries, list):
        return None
    dicts = [e for e in entries if isinstance(e, dict)]
    usd = [e for e in dicts if (e.get("currency") or "USD") == "USD"]
    pool = usd or dicts
    return pool[0] if pool else None


def _account_balance(body: Any) -> float | None:
    """Available USD buying power from an account-balances payload.

    Primary shape is ``{"balances": [{"currency":"USD","buyingPower":...}]}``; a few
    flat/legacy shapes are also accepted for resilience.
    """
    entry = _usd_balance_entry(body)
    sources: list[dict] = []
    if entry is not None:
        sources.append(entry)
    if isinstance(body, dict):
        sources.append(body)
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


def _account_flatness(body: Any) -> tuple[float, float]:
    """``(asset_notional, open_orders_notional)`` from the USD balance entry.

    Per the API schema both are *notional dollar values* (``assetNotional`` =
    aggregate value of held securities; ``openOrders`` = aggregate notional of open
    orders), NOT counts — so they're floats and any positive value means non-flat.
    """
    entry = _usd_balance_entry(body) or {}
    asset_notional = _amount(entry.get("assetNotional")) or 0.0
    open_orders = _amount(entry.get("openOrders")) or 0.0
    return asset_notional, open_orders


def _parse_positions(body: Any):
    """Parse non-flat positions/resting orders from a portfolio-positions payload.

    Polymarket US returns ``{"positions": {...}|[...], "availablePositions": [...]}``;
    ``positions`` is an object keyed by market (empty ``{}`` when flat). Raises
    ``ValueError`` on an unrecognized shape so the startup guard fails closed rather
    than reading an unparsed account as flat.
    """
    from bot.execution.account import VenuePosition

    items: list[tuple[str | None, Any]] = []
    if isinstance(body, list):
        items = [(None, r) for r in body]
    elif isinstance(body, dict):
        container = next(
            (body[k] for k in _POSITION_CONTAINER_KEYS
             if isinstance(body.get(k), (list, dict))),
            None,
        )
        if container is None:
            raise ValueError(f"unrecognized positions payload keys: {sorted(body)}")
        items = (list(container.items()) if isinstance(container, dict)
                 else [(None, r) for r in container])
    else:
        raise ValueError("unrecognized positions payload")

    out = []
    for slug_hint, r in items:
        if not isinstance(r, dict):
            continue
        slug = (r.get("marketSlug") or r.get("slug") or r.get("market_id")
                or r.get("ticker") or slug_hint or "")
        qty = _amount(r.get("netPosition") or r.get("quantity") or r.get("netQuantity")
                      or r.get("size") or r.get("position") or r.get("netSize")
                      or r.get("netShares")) or 0.0
        resting = int(r.get("openOrders") or r.get("restingOrders")
                      or r.get("resting_orders_count") or 0)
        pos = VenuePosition(slug, qty, resting)
        if pos.is_open:
            out.append(pos)
    return out


_STATE_FILLED = "ORDER_STATE_FILLED"
_STATE_REJECTED = "ORDER_STATE_REJECTED"


def _parse_create_order_response(data: Any, requested: float, side: Side):
    """Parse a synchronous ``CreateOrderResponse`` -> (status, filled, avg_price, id).

    The real shape is ``{"id", "executions": [Execution, ...]}`` — there is NO
    top-level ``order``. With ``synchronousExecution`` the response carries the full
    execution list and each execution embeds the order snapshot at that point
    (``order.state`` / ``order.cumQuantity`` / ``order.avgPx``), so the LAST snapshot
    is the terminal outcome. This is authoritative for the order; the private-WS
    confirmer is only a fallback (and must never downgrade this terminal result).
    """
    if not isinstance(data, dict):
        return OrderStatus.ERROR, 0.0, None, None
    order_id = data.get("id") or data.get("orderId")
    executions = data.get("executions") or []
    terminal_order: dict = {}
    filled_from_exec = 0.0
    rejected = False
    for ex in executions:
        etype = str(ex.get("type") or "")
        if "REJECT" in etype:
            rejected = True
        shares = _amount(ex.get("lastShares"))
        if shares and "FILL" in etype:           # EXECUTION_TYPE_FILL / _PARTIAL_FILL
            filled_from_exec += shares
        snap = ex.get("order")
        if isinstance(snap, dict):
            terminal_order = snap

    state = str(terminal_order.get("state") or "")
    cum = terminal_order.get("cumQuantity")
    filled = float(cum) if cum not in (None, "") else filled_from_exec
    if not order_id:
        order_id = terminal_order.get("id")

    avg_price = _amount(terminal_order.get("avgPx"))
    # avgPx is the YES/long-side price; for a NO buy the cost is 1 - that.
    if avg_price is not None and side is Side.NO:
        avg_price = round(1.0 - avg_price, 6)

    if state == _STATE_FILLED or (requested > 0 and filled >= requested - 1e-9):
        status = OrderStatus.FILLED
    elif state == _STATE_REJECTED or (rejected and filled <= 1e-9):
        status = OrderStatus.REJECTED
    elif filled <= 1e-9:
        status = OrderStatus.KILLED      # FOK/IOC with no fill = canceled
    else:
        status = OrderStatus.PARTIAL     # IOC partial (FOK should never land here)
    return status, filled, avg_price, order_id


def _parse_order_snapshot(order: Any, requested: float, side: Side):
    """Parse a single ``Order`` object (from ``/v1/order/preview`` or
    ``GET /v1/order/{id}``) -> (status, filled, avg_price, id). For a preview the
    ``cumQuantity`` is the EXPECTED fill, so the same threshold logic tells us whether
    the order would fill fully."""
    if not isinstance(order, dict):
        return OrderStatus.ERROR, 0.0, None, None
    order_id = order.get("id")
    state = str(order.get("state") or "")
    cum = order.get("cumQuantity")
    filled = float(cum) if cum not in (None, "") else 0.0
    avg_price = _amount(order.get("avgPx"))
    if avg_price is not None and side is Side.NO:      # avgPx is YES-side; NO cost = 1 - it
        avg_price = round(1.0 - avg_price, 6)
    if state == _STATE_FILLED or (requested > 0 and filled >= requested - 1e-9):
        status = OrderStatus.FILLED
    elif state == _STATE_REJECTED:
        status = OrderStatus.REJECTED
    elif filled <= 1e-9:
        status = OrderStatus.KILLED
    else:
        status = OrderStatus.PARTIAL
    return status, filled, avg_price, order_id


# Market states that mean the market is terminally done (vs temporarily not trading).
# Used by is_open: only these drop a cached market; everything else stays live.
_TERMINAL_MARKET_STATES = frozenset(
    {"MARKET_STATE_EXPIRED", "MARKET_STATE_TERMINATED"}
)


def normalize_bbo(
    slug: str, title: str, market_data: dict[str, Any], *, event_key: str | None = None
) -> MarketQuote:
    """Turn a ``v1MarketDataLite`` (BBO) payload into a price-only :class:`MarketQuote`.

    ``bestAsk`` is the YES ask; the NO ask is synthesized as ``1 - bestBid``. The
    payload's ``askDepth``/``bidDepth`` are LEVEL COUNTS (not contract sizes), so this
    carries NO size — real takeable size needs the full ``/book`` (:func:`normalize_book`).
    Sizes are left at 0 so nothing mistakes a level count for available liquidity.
    """
    best_ask = _amount(market_data.get("bestAsk"))
    best_bid = _amount(market_data.get("bestBid"))

    no_ask = round(1.0 - best_bid, 6) if best_bid is not None else None

    return MarketQuote(
        venue=VENUE,
        market_id=slug,
        title=title,
        event_key=event_key,
        yes_ask=best_ask,
        yes_ask_size=0.0,
        no_ask=no_ask,
        no_ask_size=0.0,
    )


def _book_levels(entries: Any) -> list[tuple[float, float]]:
    """``[(price, qty), ...]`` from a ``/book`` side (``[{"px":{value},"qty":...}]``)."""
    out: list[tuple[float, float]] = []
    for e in entries or []:
        if not isinstance(e, dict):
            continue
        px = _amount(e.get("px"))
        qty = _amount(e.get("qty"))
        if px is not None and qty is not None:
            out.append((px, qty))
    return out


def normalize_book(
    slug: str, title: str, market_data: dict[str, Any], *, event_key: str | None = None
) -> MarketQuote:
    """Turn a full ``/book`` payload into a sized :class:`MarketQuote`.

    ``offers`` are sell orders (buy YES by crossing the best/lowest offer) and ``bids``
    are buy orders (buy NO by crossing the best/highest bid). Unlike the BBO, these
    carry real ``qty`` per level, so the top-of-book ``qty`` is the actual takeable
    size used to size an arb. ``yes_ask = best offer px`` (size = its qty);
    ``no_ask = 1 - best bid px`` (size = the bid's qty).
    """
    offers = _book_levels(market_data.get("offers"))
    bids = _book_levels(market_data.get("bids"))

    yes_ask = no_ask = None
    yes_ask_size = no_ask_size = 0.0
    if offers:
        px, qty = min(offers, key=lambda lvl: lvl[0])   # best (lowest) ask
        yes_ask, yes_ask_size = px, qty
    if bids:
        px, qty = max(bids, key=lambda lvl: lvl[0])      # best (highest) bid
        no_ask, no_ask_size = round(1.0 - px, 6), qty

    return MarketQuote(
        venue=VENUE,
        market_id=slug,
        title=title,
        event_key=event_key,
        yes_ask=yes_ask,
        yes_ask_size=yes_ask_size,
        no_ask=no_ask,
        no_ask_size=no_ask_size,
        state=market_data.get("state"),   # MARKET_STATE_OPEN / SUSPENDED / ... (for the gate)
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
        self.fee_model = PolymarketUSFeeModel()
        self._gateway_client = None  # public reads
        self._api_client = None      # authenticated trading (later)
        self._key = None
        self._limiter = AsyncRateLimiter(getattr(cfg, "read_rate_per_min", None) or rate_per_min)
        # ORDERS must never queue behind scan/depth READ tokens (a hedge leg waiting on
        # read tokens right after leg 1 fills = a widened naked window). Own bucket.
        self._order_limiter = AsyncRateLimiter(240.0, burst=20)
        # Per-slug order constraints (orderPriceMinTickSize / minimumTradeQty) captured
        # during scan_quotes, so place_order can round price/qty to valid increments
        # without a hot-path fetch. The docs warn NOT to infer these from slug/type.
        self._meta: dict[str, dict[str, float | None]] = {}
        self._descriptions: dict[str, str] = {}   # slug -> resolution rules text (from scan)

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

    def _market_to_quote(
        self, m: dict[str, Any], max_close_ts: int | None
    ) -> MarketQuote | None:
        """Build a phase-1 price-only ``MarketQuote`` from one raw market dict — works
        for both the flat ``/v1/markets`` feed and a nested ``/v1/events`` market. Records
        the order-constraint meta for ``place_order``. Returns ``None`` to skip (no slug,
        or closing past a targeted ``max_close_ts`` window; unknown close is kept).

        Event-nested markets carry no ``bestBid``/``bestAsk`` (prices come from ``/book``
        in phase 2), so their phase-1 ask is ``None`` — fine: matching is structural
        (title/slug) and the streaming watchlist re-fetches real depth before trading.
        """
        slug = m.get("slug") or m.get("id")
        if not slug:
            return None
        # Real 24h traded volume — the reliable "the hedge will fill" signal (the phantom
        # /book depth is not). Captured in meta so the executor can keep a resting MAKER off
        # thin markets (whose hedge 500s and goes naked) while still TAKING them (a 500 on a
        # taker leg is a clean skip via the leg-order fix). The optional universe-level gate
        # (min_volume_24h) drops them from the scan entirely; 0 = keep them, gate the maker.
        v24 = _amount(m.get("volume24hr"))
        floor = getattr(self.cfg, "min_volume_24h", 0.0) or 0.0
        if floor > 0 and v24 is not None and v24 < floor:
            return None
        # Capture order constraints for place_order (don't infer from slug/type).
        self._meta[slug] = {
            "tick": _amount(m.get("orderPriceMinTickSize")),
            "min_qty": _amount(m.get("minimumTradeQty")),
            "volume24hr": v24,
        }
        # Resolution rules text, free in the scan payload — the rules-verification
        # layer reads it via market_rules() (no per-market fetch needed).
        desc = m.get("description")
        if desc:
            self._descriptions[slug] = str(desc)
        # Per-game markets often carry endDate=None, which left the matcher's resolve-date
        # guard with no Polymarket date to compare -> it failed OPEN and matched same-teams
        # games on DIFFERENT dates (a Kalshi June-30 KC@CWS paired with a Poly June-26 one;
        # when June-26 settled the Kalshi leg was stranded). The game date is in the slug
        # (...-2026-06-26...), so fall back to it so the date-gap guard can actually fire.
        close_time = parse_iso8601(m.get("endDate")) or _date_from_slug(slug)
        # Targeted window: skip markets closing past it (keep unknown close).
        if max_close_ts is not None and close_time is not None and close_time > max_close_ts:
            return None
        best_ask = _amount(m.get("bestAsk"))
        best_bid = _amount(m.get("bestBid"))
        return MarketQuote(
            venue=VENUE,
            market_id=slug,
            title=build_market_title(m),
            yes_ask=best_ask,
            yes_ask_size=0.0,
            no_ask=round(1.0 - best_bid, 6) if best_bid is not None else None,
            no_ask_size=0.0,
            close_time=close_time,
        )

    async def _scan_event_markets(
        self, *, max_close_ts: int | None = None
    ) -> list[MarketQuote]:
        """Phase 1, part 2: the per-GAME markets (live moneylines, totals, spreads,
        player props) live NESTED under ``/v1/events`` — the flat ``/v1/markets`` feed is
        season futures + a few standalones and does NOT contain them. Enumerate active
        events and flatten each event's ``markets[]`` so the live game catalog is
        scannable (and matchable against Kalshi's per-game lines). Deduped by slug;
        prices are filled later from ``/book``."""
        out: list[MarketQuote] = []
        seen: set[str] = set()
        offset = 0
        while True:
            await self._limiter.wait()
            resp = await self._gateway().get(
                "/v1/events",
                params={"limit": 500, "active": "true", "closed": "false", "offset": offset},
            )
            resp.raise_for_status()
            events = resp.json().get("events", [])
            if not events:
                break
            new = 0
            for ev in events:
                for m in ev.get("markets") or []:
                    # Skip a settled sub-market inside an otherwise-live event.
                    if not m.get("active", True) or m.get("closed"):
                        continue
                    slug = m.get("slug") or m.get("id")
                    if not slug or slug in seen:
                        continue
                    seen.add(slug)
                    new += 1
                    q = self._market_to_quote(m, max_close_ts)
                    if q is not None:
                        out.append(q)
            offset += len(events)
            # Offset ignored (same page) or end of feed -> stop.
            if new == 0 or len(events) < 500:
                break
        return out

    async def scan_quotes(
        self, limit: int = 500, *, max_close_ts: int | None = None
    ) -> list[MarketQuote]:
        """Phase 1: price-only quotes for the whole board (sizes come from
        :meth:`fetch_quote` for shortlisted markets only). Two sources, merged + deduped:

        * ``/v1/markets`` — the flat feed (season futures + standalones), paginated by
          ``offset``. ``limit`` is a TOTAL cap across pages (``<= 0`` = the entire feed);
          if the gateway ignores ``offset`` we get no new slugs and stop.
        * ``/v1/events`` — the per-GAME markets nested under each event (live moneylines,
          props, totals). NOT capped by ``limit`` — these are the live game lines we most
          want to match, so we always take them all.

        ``max_close_ts`` (Unix seconds) enables a TARGETED scan: only markets closing
        at/before that time (from ``endDate``) are returned; markets with no close_time
        are kept (never drop a live market on missing data).
        """
        out: list[MarketQuote] = []
        seen: set[str] = set()
        offset = 0
        unbounded = limit <= 0                          # scan the whole feed
        # Server-side close-time filter (docs: endDateMax, ISO 8601). Cuts the feed at
        # the source; the client-side endDate filter below still backs it up.
        end_date_max = (
            datetime.fromtimestamp(max_close_ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            if max_close_ts is not None else None
        )
        while unbounded or len(out) < limit:
            await self._limiter.wait()
            page_size = 500 if unbounded else min(limit - len(out), 500)
            params: dict[str, Any] = {
                "limit": page_size, "active": "true", "closed": "false", "offset": offset,
            }
            if end_date_max is not None:
                params["endDateMax"] = end_date_max
            resp = await self._gateway().get("/v1/markets", params=params)
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
                new += 1  # progress through the feed (counts even if out-of-window)
                q = self._market_to_quote(m, max_close_ts)
                if q is not None:
                    out.append(q)
            offset += len(markets)
            # No new slugs (offset ignored / end of feed), or a short page -> done.
            if new == 0 or len(markets) < page_size:
                break

        # Merge in the per-game markets nested under /v1/events (the flat feed above
        # misses them). Isolated: an events-feed hiccup must not lose the /v1/markets scan.
        try:
            event_quotes = await self._scan_event_markets(max_close_ts=max_close_ts)
            added = 0
            for q in event_quotes:
                if q.market_id not in seen:
                    seen.add(q.market_id)
                    out.append(q)
                    added += 1
            log.info("polymarket scan: %d flat markets + %d per-game (events) = %d total",
                     len(out) - added, added, len(out))
        except Exception as exc:
            log.warning("polymarket /v1/events scan failed (%s); flat /v1/markets only", exc)
        return out

    async def fetch_quote(self, market: RawMarket) -> MarketQuote | None:
        """Deep (sized) quote: the full ``/book`` for one market (by slug), used in
        phase 2. The book carries real per-level ``qty`` (the BBO only carries level
        COUNTS), so this is the authoritative size for arb sizing."""
        await self._limiter.wait()
        resp = await self._gateway().get(f"/v1/markets/{market.market_id}/book")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        market_data = resp.json().get("marketData", {})
        if not market_data:
            return None
        return normalize_book(market.market_id, market.title, market_data)

    async def is_open(self, market_id: str):
        """Whether the market is still live (vs terminally expired/terminated).

        Reads the ``/book`` ``state`` field: only ``MARKET_STATE_EXPIRED`` /
        ``MARKET_STATE_TERMINATED`` count as closed (return ``False``). Everything else
        (open/pre-open/suspended/halted/closing-auction) is live (``True``), and any
        error / 404 / missing state is ``None`` (unknown -> caller keeps it) so a
        transient quirk can't wrongly drop a live market.
        """
        await self._limiter.wait()
        try:
            resp = await self._gateway().get(f"/v1/markets/{market_id}/book")
        except Exception:
            return None
        if resp.status_code != 200:
            return None
        state = (resp.json().get("marketData") or {}).get("state")
        if not state:
            return None
        return state not in _TERMINAL_MARKET_STATES

    async def stream_order_book(self, market_ids: list[str]) -> AsyncIterator[MarketQuote]:
        """Stream real-time SIZED top-of-book via the markets WS (MARKET_DATA full book).

        Requires credentials (the markets WS is on the authenticated API, unlike the
        public REST gateway). Subscribes in batches of 100 slugs; reconnects with
        exponential backoff; ignores heartbeats. Uses the full-book channel (not the
        lite BBO) so the streamed quote carries real per-level size.
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
                    self.cfg.ws_markets, additional_headers=headers, open_timeout=10,
                    ping_interval=20, ping_timeout=30, close_timeout=5,
                ) as ws:
                    chunks = (
                        [market_ids[i : i + 100] for i in range(0, len(market_ids), 100)]
                        if market_ids else [None]
                    )
                    for n, chunk in enumerate(chunks):
                        sub: dict[str, Any] = {
                            "requestId": f"md-{n}",
                            "subscriptionType": "SUBSCRIPTION_TYPE_MARKET_DATA",
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
                        quote = parse_market_data(data)
                        if quote is not None:
                            yield quote
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("polymarket ws disconnected (%s); reconnecting in %.0fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def capture_market_data(
        self, market_ids: list[str], *, limit: int = 6, idle_timeout: float = 20.0
    ) -> list[dict[str, Any]]:
        """Capture the first ``limit`` raw MARKET_DATA messages for diagnostics.

        Subscribes exactly like :meth:`stream_order_book` but returns the raw decoded JSON
        (not normalized), so a caller can see whether the channel sends a full snapshot or
        incremental deltas, and whether the streamed top-of-book matches REST ``/book``.
        """
        if not getattr(self.cfg, "is_trading_configured", False):
            raise OrderNotPermitted("Polymarket US credentials required for the WebSocket")
        import json

        import websockets  # lazy

        path = "/v1/ws/markets"
        out: list[dict[str, Any]] = []
        headers = self._auth_headers("GET", path)
        async with websockets.connect(
            self.cfg.ws_markets, additional_headers=headers, open_timeout=10
        ) as ws:
            await ws.send(json.dumps({"subscribe": {
                "requestId": "probe",
                "subscriptionType": "SUBSCRIPTION_TYPE_MARKET_DATA",
                "marketSlugs": market_ids,
            }}))
            while len(out) < limit:
                raw = await asyncio.wait_for(ws.recv(), timeout=idle_timeout)
                data = json.loads(raw)
                if "heartbeat" in data or data.get("error"):
                    continue
                out.append(data)
        return out

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
                    self.cfg.ws_private, additional_headers=headers, open_timeout=10,
                    ping_interval=20, ping_timeout=30, close_timeout=5,
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

    def _order_body(self, market_id: str, side: Side, action: str, price: float,
                    contracts: float, tif: str) -> dict:
        """Build the CreateOrderRequest body (shared by place + preview so a preview
        reflects the EXACT order we'd send). ``price`` is the cost of the requested side;
        the API wants the YES/long price, so for NO we send ``1 - price``. Price and
        quantity are snapped to the market's orderPriceMinTickSize / minimumTradeQty
        (captured during scan_quotes) — off-grid values get normalized or rejected."""
        yes_value = price if side is Side.YES else round(1.0 - price, 6)
        yes_value = min(max(yes_value, 0.01), 0.99)
        meta = self._meta.get(market_id) or {}
        tick, min_qty = meta.get("tick"), meta.get("min_qty")
        if tick and tick > 0:
            yes_value = round(round(yes_value / tick) * tick, 6)
            yes_value = min(max(yes_value, tick), round(1.0 - tick, 6))
        quantity: float = contracts
        if min_qty and min_qty > 0:
            snapped = math.floor(round(contracts / min_qty, 9)) * min_qty
            if snapped >= min_qty:          # keep original if snapping would zero it out
                quantity = round(snapped, 6)
        return {
            "marketSlug": market_id,
            "type": "ORDER_TYPE_LIMIT",
            "price": {"value": _price_str(yes_value), "currency": "USD"},
            "quantity": quantity,
            "tif": _TIF.get(tif, "TIME_IN_FORCE_FILL_OR_KILL"),
            "intent": _INTENT[(side, action)],
            "manualOrderIndicator": "MANUAL_ORDER_INDICATOR_AUTOMATIC",
        }

    async def place_order(
        self, market_id: str, side: Side, action: str, price: float, contracts: float,
        *, tif: str = "fill_or_kill", post_only: bool = False, expiration_ts: int | None = None,
    ) -> OrderResult:
        """Place an order via POST /v1/orders (X-PM Ed25519 auth).

        Taker orders use ``synchronousExecution`` so the response carries the terminal
        executions (with a submit-then-poll fallback). A ``post_only`` order is a resting
        MAKER: it's sent async with ``participateDontInitiate`` (rejected if it would
        immediately match) and a GOOD_TILL_CANCEL tif — the executor cancels it on
        timeout/drift. We deliberately do NOT use GOOD_TILL_DATE: Polymarket has GTD
        disabled venue-side (POST /v1/orders -> HTTP 400 "GTD orders are temporarily
        disabled"), and an explicit cancel doesn't depend on a venue feature. A resting
        maker comes back ``RESTING``. ``expiration_ts`` is accepted for signature
        compatibility (the executor passes its maker timeout) but no longer drives a
        server-side expiry — cleanup is the executor's cancel.
        """
        if not getattr(self.cfg, "is_trading_configured", False):
            raise OrderNotPermitted("Polymarket US trading credentials not configured")
        body = self._order_body(market_id, side, action, price, contracts, tif)
        if post_only:
            # Maker-only: rest on the book, never cross (the docs reject it if it would
            # immediately match). NO synchronousExecution — it rests; we confirm via the WS.
            # Time-in-force is GOOD_TILL_CANCEL; the executor cancels it on timeout/drift.
            # We do NOT use GOOD_TILL_DATE: Polymarket has it disabled venue-side (POST
            # /v1/orders -> HTTP 400 "GTD orders are temporarily disabled"), so a server-side
            # self-expiry isn't available — an explicit cancel is the cleanup, and it doesn't
            # depend on a venue feature that can come and go.
            body["participateDontInitiate"] = True
            body["tif"] = "TIME_IN_FORCE_GOOD_TILL_CANCEL"
        else:
            # Per the API schema maxBlockTime is an int64 (seconds) encoded as a string — a
            # bare "5", NOT a duration like "5s". Keep it under the 10s HTTP client timeout.
            body["synchronousExecution"] = True
            body["maxBlockTime"] = "5"
        await self._order_limiter.wait()
        try:
            resp = await self._api().post(
                "/v1/orders", json=body, headers=self._auth_headers("POST", "/v1/orders")
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # 4xx -> REJECTED (no fill); else ERROR (unknown)
            from bot.execution.orders import order_error_result
            # The server's error body is often opaque ("internal server error"), so log the
            # exact request we sent — it's what makes an order rejection diagnosable (and
            # replayable with curl) when the response tells us nothing.
            log.warning("polymarket order POST failed (%s); request body=%s", exc, body)
            return order_error_result(VENUE, market_id, side, action, contracts, exc)

        status, filled, avg_price, order_id = _parse_create_order_response(data, contracts, side)
        # A post-only maker that didn't immediately fill or reject is RESTING on the book
        # (it was sent async, so there are no executions yet) — surface that so the caller
        # waits for the fill via the WS confirmer rather than treating it as a no-trade.
        if post_only and order_id and status is OrderStatus.KILLED and filled <= 1e-9:
            status = OrderStatus.RESTING
        # Submit-then-poll fallback: a synchronous FOK should come back terminal, but if it
        # didn't (a non-terminal state with an order id), GET the order once for the real
        # terminal outcome rather than mis-reporting a fill/no-fill.
        if order_id and status is OrderStatus.PARTIAL:
            resolved = await self._get_order_result(order_id, contracts, side)
            if resolved is not None:
                return resolved
        return OrderResult(
            venue=VENUE, market_id=market_id, side=side, action=action,
            requested=contracts, filled=filled, avg_price=avg_price,
            order_id=order_id, status=status, raw=data,
        )

    async def preview_order(
        self, market_id: str, side: Side, action: str, price: float, contracts: float,
        *, tif: str = "fill_or_kill",
    ) -> OrderResult:
        """Preview an order via POST /v1/order/preview WITHOUT submitting it. The returned
        Order carries calculated values (``cumQuantity`` = expected fill), so the executor
        can confirm a hedge would fully fill before committing the first leg — avoiding a
        FOK fired into phantom liquidity (a naked leg + the synchronous-execution 500)."""
        if not getattr(self.cfg, "is_trading_configured", False):
            raise OrderNotPermitted("Polymarket US trading credentials not configured")
        body = self._order_body(market_id, side, action, price, contracts, tif)
        await self._limiter.wait()
        try:
            resp = await self._api().post(
                "/v1/order/preview", json={"request": body},
                headers=self._auth_headers("POST", "/v1/order/preview"),
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            from bot.execution.orders import order_error_result
            log.warning("polymarket order preview failed (%s); request body=%s", exc, body)
            return order_error_result(VENUE, market_id, side, action, contracts, exc)
        order = data.get("order") if isinstance(data, dict) else None
        status, filled, avg_price, order_id = _parse_order_snapshot(order, contracts, side)
        return OrderResult(
            venue=VENUE, market_id=market_id, side=side, action=action,
            requested=contracts, filled=filled, avg_price=avg_price,
            order_id=order_id, status=status, raw=data,
        )

    async def _get_order_result(self, order_id: str, requested: float,
                                side: Side) -> OrderResult | None:
        """GET /v1/order/{id} -> OrderResult, or None if it couldn't be read. The
        authoritative terminal state for an order we already have an id for."""
        path = f"/v1/order/{order_id}"
        await self._order_limiter.wait()
        try:
            resp = await self._api().get(path, headers=self._auth_headers("GET", path))
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            log.warning("polymarket get-order %s failed: %s", order_id, exc)
            return None
        order = data.get("order") if isinstance(data, dict) else None
        status, filled, avg_price, oid = _parse_order_snapshot(order, requested, side)
        return OrderResult(
            venue=VENUE, market_id=(order or {}).get("marketSlug", ""), side=side,
            action="buy", requested=requested, filled=filled, avg_price=avg_price,
            order_id=oid or order_id, status=status, raw=data,
        )

    async def order_filled_qty(self, order_id: str, requested: float, side: Side) -> float | None:
        """Authoritative filled contracts for one order from GET /v1/order/{id}, or None if
        it couldn't be read. The private-fill stream can MISS a maker partial; this reads the
        order's true ``cumQuantity`` so the executor never walks away from a real fill."""
        res = await self._get_order_result(order_id, requested, side)
        return None if res is None else res.filled

    async def cancel_order(self, order_id: str) -> dict:
        if not getattr(self.cfg, "is_trading_configured", False):
            raise OrderNotPermitted("Polymarket US trading credentials not configured")
        path = f"/v1/order/{order_id}/cancel"
        await self._order_limiter.wait()
        resp = await self._api().post(path, headers=self._auth_headers("POST", path))
        resp.raise_for_status()
        return resp.json()

    async def get_positions(self) -> dict:
        raise NotImplementedError("positions endpoint lands with the live phase")

    async def market_rules(self, slug: str) -> str | None:
        """Resolution rules text for a market, captured from the scan payload
        (Polymarket's ``description`` maps every outcome to its settlement condition).
        None if the market hasn't been scanned this process."""
        return self._descriptions.get(slug)

    async def settled_positions(self) -> dict:
        """Raw slug -> position dict INCLUDING recently-settled positions.

        ``?includeSettled=true`` keeps settled positions readable for a window after
        resolution (they then drop off entirely) — the only place Polymarket exposes a
        post-settlement ``realized``, which the settlement-truth auditor uses to infer
        which side actually paid. Read-only. The query string is excluded from the
        signed path (verified live: signing the bare path returns 200)."""
        if not getattr(self.cfg, "is_trading_configured", False):
            raise OrderNotPermitted("Polymarket US trading credentials not configured")
        path = "/v1/portfolio/positions"
        await self._limiter.wait()
        resp = await self._api().get(
            path + "?includeSettled=true", headers=self._auth_headers("GET", path))
        resp.raise_for_status()
        body = resp.json()
        return (body.get("positions") if isinstance(body, dict) else None) or {}

    async def account_snapshot(self):
        """Balance + non-flat positions/resting orders, for the startup guard.

        Reads ``/v1/account/balances`` (USD ``buyingPower``, plus ``assetNotional`` /
        ``openOrders`` as the authoritative flatness signal) and
        ``/v1/portfolio/positions`` (itemized detail). If the account is non-flat by
        the balance signal but the positions list parses empty, a synthetic
        account-level position is recorded so the guard still trips. An unrecognized
        positions shape raises (the guard then fails closed).
        """
        from bot.execution.account import AccountSnapshot, VenuePosition

        if not getattr(self.cfg, "is_trading_configured", False):
            raise OrderNotPermitted("Polymarket US trading credentials not configured")

        await self._limiter.wait()
        path = "/v1/account/balances"
        resp = await self._api().get(path, headers=self._auth_headers("GET", path))
        resp.raise_for_status()
        balance_body = resp.json()
        balance = _account_balance(balance_body)
        asset_notional, open_orders_notional = _account_flatness(balance_body)

        await self._limiter.wait()
        path = "/v1/portfolio/positions"
        resp = await self._api().get(path, headers=self._auth_headers("GET", path))
        resp.raise_for_status()
        positions = _parse_positions(resp.json())
        # Both notionals are dollars (not counts); any positive value = non-flat. If the
        # balance signal says non-flat but the positions list parsed empty, synthesize
        # an account-level entry so the guard still trips (resting_orders as a 0/1 flag).
        if not positions and (asset_notional > 1e-9 or open_orders_notional > 1e-9):
            positions = [VenuePosition(
                "(account-level)", quantity=asset_notional,
                resting_orders=1 if open_orders_notional > 1e-9 else 0,
            )]
        return AccountSnapshot(self.name, balance, positions)

    async def aclose(self) -> None:
        for attr in ("_gateway_client", "_api_client"):
            client = getattr(self, attr)
            if client is not None:
                await client.aclose()
                setattr(self, attr, None)
