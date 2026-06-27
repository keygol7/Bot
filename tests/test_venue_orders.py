"""Venue order request construction + fill-response mapping (mocked HTTP)."""

import asyncio
import json

import pytest

httpx = pytest.importorskip("httpx")

from bot.config.settings import KalshiConfig, QcexConfig
from bot.models import Side
from bot.venues.kalshi import KalshiVenue
from bot.venues.polymarket_us import PolymarketUSVenue


def _client(handler, base_url):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=base_url)


def test_kalshi_buy_yes_v2_request_and_fill():
    # V2 /portfolio/events/orders: buy YES -> bid @ price (fixed-point strings).
    cap = {}

    def handler(req):
        cap["url"] = req.url.path
        cap["body"] = json.loads(req.content)
        return httpx.Response(201, json={
            "order_id": "abc", "fill_count": "2.00", "remaining_count": "0.00",
            "average_fill_price": "0.40"})

    v = KalshiVenue(KalshiConfig(api_key_id="k", private_key_path="x"))
    v._client = _client(handler, v.cfg.api_base)
    v._auth_headers = lambda m, p: {}  # skip real signing for this test

    r = asyncio.run(v.place_order("KTICK", Side.YES, "buy", 0.40, 2))
    assert cap["url"].endswith("/portfolio/events/orders")
    assert cap["body"]["ticker"] == "KTICK" and cap["body"]["side"] == "bid"
    assert cap["body"]["price"] == "0.4000" and cap["body"]["count"] == "2.00"
    assert cap["body"]["time_in_force"] == "fill_or_kill"
    assert cap["body"]["self_trade_prevention_type"] == "taker_at_cross"
    assert r.status.value == "FILLED" and r.filled == 2.0 and r.avg_price == 0.40


def test_kalshi_buy_no_v2_sells_yes_at_one_minus_price():
    # Buy NO @ 0.55 == sell YES @ 0.45 -> side "ask", price "0.4500". The YES-side
    # average_fill_price (0.45) inverts back to a NO cost of 0.55.
    cap = {}

    def handler(req):
        cap["body"] = json.loads(req.content)
        return httpx.Response(201, json={
            "order_id": "x", "fill_count": "2.00", "remaining_count": "0.00",
            "average_fill_price": "0.45"})

    v = KalshiVenue(KalshiConfig(api_key_id="k", private_key_path="x"))
    v._client = _client(handler, v.cfg.api_base)
    v._auth_headers = lambda m, p: {}
    r = asyncio.run(v.place_order("KTICK", Side.NO, "buy", 0.55, 2))
    assert cap["body"]["side"] == "ask" and cap["body"]["price"] == "0.4500"
    assert r.status.value == "FILLED" and r.avg_price == 0.55


def test_kalshi_maker_order_rests():
    # post_only + expiration -> a resting maker (GTC); 0 fill with an order_id = RESTING.
    cap = {}

    def handler(req):
        cap["body"] = json.loads(req.content)
        return httpx.Response(201, json={
            "order_id": "m1", "fill_count": "0.00", "remaining_count": "37.00"})

    v = KalshiVenue(KalshiConfig(api_key_id="k", private_key_path="x"))
    v._client = _client(handler, v.cfg.api_base)
    v._auth_headers = lambda m, p: {}
    r = asyncio.run(v.place_order("KT", Side.NO, "buy", 0.72, 37,
                                  tif="gtc", post_only=True, expiration_ts=1782300000))
    assert cap["body"]["post_only"] is True
    assert cap["body"]["time_in_force"] == "good_till_canceled"
    # Self-expiry sent under BOTH keys: expiration_time is confirmed honored by a live
    # order record; expiration_ts is the documented field. Belt and suspenders so a maker
    # never rests un-expired (-> late naked fill).
    assert cap["body"]["expiration_time"] == 1782300000
    assert cap["body"]["expiration_ts"] == 1782300000
    assert r.status.value == "RESTING" and r.order_id == "m1" and r.filled == 0.0


def test_kalshi_v2_fok_no_fill_is_killed():
    def handler(req):
        return httpx.Response(201, json={
            "order_id": "x", "fill_count": "0.00", "remaining_count": "2.00"})

    v = KalshiVenue(KalshiConfig(api_key_id="k", private_key_path="x"))
    v._client = _client(handler, v.cfg.api_base)
    v._auth_headers = lambda m, p: {}
    r = asyncio.run(v.place_order("KTICK", Side.NO, "buy", 0.55, 2))
    assert r.status.value == "KILLED"   # fill_count 0 on FoK


def test_kalshi_v2_rejection_surfaces_body():
    # A 4xx (e.g. legacy-endpoint-style errors) -> REJECTED with the venue body.
    def handler(req):
        return httpx.Response(400, json={"error": {"code": "bad_request", "message": "x"}})

    v = KalshiVenue(KalshiConfig(api_key_id="k", private_key_path="x"))
    v._client = _client(handler, v.cfg.api_base)
    v._auth_headers = lambda m, p: {}
    r = asyncio.run(v.place_order("KTICK", Side.YES, "buy", 0.40, 2))
    assert r.status.value == "REJECTED"
    assert r.raw.get("http_status") == 400


def test_kalshi_cancel_uses_v2_events_orders_endpoint():
    # The legacy DELETE /portfolio/orders/{id} was deprecated -> 410; cancel must hit the
    # V2 path /portfolio/events/orders/{id} (mirrors the create endpoint family).
    cap = {}

    def handler(req):
        cap["method"] = req.method
        cap["url"] = req.url.path
        return httpx.Response(200, json={"order": {"order_id": "o1", "status": "canceled"}})

    v = KalshiVenue(KalshiConfig(api_key_id="k", private_key_path="x"))
    v._client = _client(handler, v.cfg.api_base)
    v._auth_headers = lambda m, p: {}
    asyncio.run(v.cancel_order("o1"))
    assert cap["method"] == "DELETE"
    assert cap["url"].endswith("/portfolio/events/orders/o1")


def test_kalshi_order_filled_qty_reads_fill_count_fp():
    # The order RECORD reports fills as the fixed-point string `fill_count_fp`, not
    # `fill_count` (null). order_filled_qty must read fill_count_fp -> the true fill.
    def handler(req):
        return httpx.Response(200, json={"order": {
            "order_id": "o", "status": "canceled",
            "fill_count": None, "fill_count_fp": "2.00"}})

    v = KalshiVenue(KalshiConfig(api_key_id="k", private_key_path="x"))
    v._client = _client(handler, v.cfg.api_base)
    v._auth_headers = lambda m, p: {}
    qty = asyncio.run(v.order_filled_qty("o", 2, Side.NO))
    assert qty == 2.0


def test_kalshi_order_filled_qty_zero_fill_is_zero_not_unreadable():
    # fill_count_fp "0.00" is an AUTHORITATIVE zero fill (clean no-trade) -> 0.0, NOT None.
    # Returning None here is what made the maker fix false-HALT on a cleanly-unfilled maker.
    def handler(req):
        return httpx.Response(200, json={"order": {
            "order_id": "o", "status": "canceled",
            "fill_count": None, "fill_count_fp": "0.00"}})

    v = KalshiVenue(KalshiConfig(api_key_id="k", private_key_path="x"))
    v._client = _client(handler, v.cfg.api_base)
    v._auth_headers = lambda m, p: {}
    qty = asyncio.run(v.order_filled_qty("o", 2, Side.NO))
    assert qty == 0.0 and qty is not None


def _exec_response(order_id, state, cum, last_shares, avg_yes, etype="EXECUTION_TYPE_FILL"):
    """Build a synchronous CreateOrderResponse ({id, executions:[Execution]})."""
    order = {"id": order_id, "state": state, "cumQuantity": cum}
    if avg_yes is not None:
        order["avgPx"] = {"value": str(avg_yes), "currency": "USD"}
    return {"id": order_id, "executions": [
        {"type": etype, "lastShares": str(last_shares), "order": order}]}


def test_polymarket_buy_no_request_and_fill():
    cap = {}

    def handler(req):
        cap["body"] = json.loads(req.content)
        # avgPx is YES-side: a NO bought at 0.55 fills at YES-side 0.45.
        return httpx.Response(200, json=_exec_response("o1", "ORDER_STATE_FILLED", 2, 2, "0.45"))

    cfg = QcexConfig(api_key_id="k", secret_key="c2VjcmV0")  # is_trading_configured -> True
    v = PolymarketUSVenue(cfg)
    v._api_client = _client(handler, cfg.api_base)
    v._auth_headers = lambda m, p: {}

    r = asyncio.run(v.place_order("slug", Side.NO, "buy", 0.55, 2))
    # Buy NO -> BUY_SHORT, YES-side price.value = 1 - 0.55 = 0.45
    assert cap["body"]["intent"] == "ORDER_INTENT_BUY_SHORT"
    assert cap["body"]["price"]["value"] == "0.45"
    assert cap["body"]["manualOrderIndicator"] == "MANUAL_ORDER_INDICATOR_AUTOMATIC"
    assert cap["body"]["tif"] == "TIME_IN_FORCE_FILL_OR_KILL"
    assert r.status.value == "FILLED" and r.filled == 2 and r.order_id == "o1"
    # avg_price is the NO cost (1 - YES-side 0.45 = 0.55), NOT the raw YES-side value.
    assert r.avg_price == 0.55


def test_polymarket_no_fill_avg_price_not_inverted():
    # Regression: the real UZB-COL trade. NO leg filled at $0.855 (YES-side 0.145);
    # it must record 0.855, not 0.145 (which inflated a 4c arb into a fake 75c one).
    def handler(req):
        return httpx.Response(200, json=_exec_response("o", "ORDER_STATE_FILLED", 2, 2, "0.145"))

    cfg = QcexConfig(api_key_id="k", secret_key="c2VjcmV0")
    v = PolymarketUSVenue(cfg)
    v._api_client = _client(handler, cfg.api_base)
    v._auth_headers = lambda m, p: {}
    r = asyncio.run(v.place_order("astatc-fwc-uzb-col", Side.NO, "buy", 0.855, 2))
    assert r.status.value == "FILLED"
    assert r.avg_price == 0.855          # 1 - 0.145, the true NO cost

    # YES legs are reported on the same side, so they pass through unchanged.
    def yes_handler(req):
        return httpx.Response(200, json=_exec_response("y", "ORDER_STATE_FILLED", 2, 2, "0.105"))

    v2 = PolymarketUSVenue(cfg)
    v2._api_client = _client(yes_handler, cfg.api_base)
    v2._auth_headers = lambda m, p: {}
    ry = asyncio.run(v2.place_order("slug", Side.YES, "buy", 0.105, 2))
    assert ry.avg_price == 0.105


def test_polymarket_fok_killed_when_no_fill():
    # FOK that couldn't fill -> terminal CANCELED, 0 filled -> KILLED (clean skip).
    def handler(req):
        return httpx.Response(200, json=_exec_response(
            "o", "ORDER_STATE_CANCELED", 0, 0, None, etype="EXECUTION_TYPE_CANCELED"))

    cfg = QcexConfig(api_key_id="k", secret_key="c2VjcmV0")
    v = PolymarketUSVenue(cfg)
    v._api_client = _client(handler, cfg.api_base)
    v._auth_headers = lambda m, p: {}
    r = asyncio.run(v.place_order("slug", Side.YES, "buy", 0.62, 6))
    assert r.status.value == "KILLED" and r.filled == 0


def test_polymarket_full_fill_from_multiple_executions():
    # A 6-ct fill arriving as a partial then a completing fill; the terminal order
    # snapshot (cumQuantity 6, state FILLED) -> FILLED 6, not a spurious partial.
    def handler(req):
        return httpx.Response(200, json={"id": "o", "executions": [
            {"type": "EXECUTION_TYPE_PARTIAL_FILL", "lastShares": "0.62",
             "order": {"id": "o", "state": "ORDER_STATE_PARTIALLY_FILLED", "cumQuantity": 0.62}},
            {"type": "EXECUTION_TYPE_FILL", "lastShares": "5.38",
             "order": {"id": "o", "state": "ORDER_STATE_FILLED", "cumQuantity": 6,
                       "avgPx": {"value": "0.88", "currency": "USD"}}},
        ]})

    cfg = QcexConfig(api_key_id="k", secret_key="c2VjcmV0")
    v = PolymarketUSVenue(cfg)
    v._api_client = _client(handler, cfg.api_base)
    v._auth_headers = lambda m, p: {}
    r = asyncio.run(v.place_order("slug", Side.YES, "buy", 0.88, 6))
    assert r.status.value == "FILLED" and r.filled == 6 and r.avg_price == 0.88


def test_polymarket_buy_yes_price_is_yes_side():
    cap = {}

    def handler(req):
        cap["body"] = json.loads(req.content)
        return httpx.Response(200, json=_exec_response(
            "o", "ORDER_STATE_CANCELED", 0, 0, None, etype="EXECUTION_TYPE_CANCELED"))

    cfg = QcexConfig(api_key_id="k", secret_key="c2VjcmV0")
    v = PolymarketUSVenue(cfg)
    v._api_client = _client(handler, cfg.api_base)
    v._auth_headers = lambda m, p: {}
    r = asyncio.run(v.place_order("slug", Side.YES, "buy", 0.62, 1))
    assert cap["body"]["intent"] == "ORDER_INTENT_BUY_LONG"
    assert cap["body"]["price"]["value"] == "0.62"
    # maxBlockTime is an int64 (seconds) encoded as a string per the API schema: a bare
    # "5", not a duration like "5s".
    assert cap["body"]["maxBlockTime"] == "5"
    assert r.status.value == "KILLED"


def test_kalshi_scan_quotes_paginates_cursor():
    # Two pages: cursor on page 1 -> page 2 -> no cursor (end). limit=1500 spans both.
    pages = [
        {"markets": [{"ticker": f"A{i}", "title": f"A{i}", "yes_bid": 40, "yes_ask": 41,
                      "no_bid": 59, "no_ask": 60} for i in range(1000)], "cursor": "CUR2"},
        {"markets": [{"ticker": f"B{i}", "title": f"B{i}", "yes_bid": 40, "yes_ask": 41,
                      "no_bid": 59, "no_ask": 60} for i in range(500)], "cursor": ""},
    ]
    seen_params = []

    def handler(req):
        seen_params.append(dict(req.url.params))
        page = pages[1] if req.url.params.get("cursor") == "CUR2" else pages[0]
        return httpx.Response(200, json=page)

    v = KalshiVenue(KalshiConfig(api_key_id="k", private_key_path="x"))
    v._client = _client(handler, v.cfg.api_base)
    v._auth_headers = lambda m, p: {}

    quotes = asyncio.run(v.scan_quotes(1500))
    assert len(quotes) == 1500                          # both pages collected
    assert {q.market_id for q in quotes} >= {"A0", "B0", "B499"}
    assert len(seen_params) == 2                         # exactly two HTTP calls
    assert seen_params[0].get("cursor") is None          # first page has no cursor
    assert seen_params[1]["cursor"] == "CUR2"            # second follows the cursor
    assert seen_params[1]["limit"] == "500"              # remaining cap, not a full page


def test_polymarket_scan_quotes_volume_gate_drops_thin_markets():
    # The liquidity gate drops low-24h-volume markets (the thin set whose Polymarket hedge
    # 500s -> naked maker) and keeps liquid ones — using real traded volume, not /book depth.
    def handler(req):
        if req.url.path.endswith("/v1/events"):
            return httpx.Response(200, json={"events": []})
        return httpx.Response(200, json={"markets": [
            {"slug": "liquid-game", "question": "Liquid", "bestAsk": "0.40", "bestBid": "0.38",
             "volume24hr": "9000"},
            {"slug": "thin-prop", "question": "Thin", "bestAsk": "0.02", "bestBid": "0.01",
             "volume24hr": "120"},
            {"slug": "no-vol-field", "question": "Unknown", "bestAsk": "0.40", "bestBid": "0.38"},
        ]})

    cfg = QcexConfig(min_volume_24h=1000)
    v = PolymarketUSVenue(cfg)
    v._gateway_client = _client(handler, cfg.gateway_base)
    ids = {q.market_id for q in asyncio.run(v.scan_quotes(5000))}
    assert "liquid-game" in ids       # vol 9000 >= 1000 -> kept
    assert "thin-prop" not in ids     # vol 120 < 1000 -> dropped (the 500-prone set)
    assert "no-vol-field" in ids      # missing volume -> fail open (kept)


def test_polymarket_close_time_falls_back_to_slug_date():
    # Per-game markets carry endDate=null; the game date is in the slug. close_time must
    # fall back to the slug date so the matcher's resolve-date guard has a date to compare
    # (else it fails open and pairs same-teams games on DIFFERENT dates -> stranded leg).
    from datetime import datetime, timezone

    def handler(req):
        if req.url.path.endswith("/v1/events"):
            return httpx.Response(200, json={"events": []})
        return httpx.Response(200, json={"markets": [
            {"slug": "aec-mlb-kc-cws-2026-06-26", "question": "KC@CWS",
             "bestAsk": "0.40", "bestBid": "0.38", "endDate": None},
        ]})

    cfg = QcexConfig()
    v = PolymarketUSVenue(cfg)
    v._gateway_client = _client(handler, cfg.gateway_base)
    q = next(q for q in asyncio.run(v.scan_quotes(5000)) if q.market_id == "aec-mlb-kc-cws-2026-06-26")
    assert q.close_time == datetime(2026, 6, 26, tzinfo=timezone.utc).timestamp()


def test_kalshi_account_snapshot_reads_position_fp():
    # The LIVE positions API carries the signed contract count as "position_fp" (decimal
    # string); the older "position" field is absent. Reading only "position" reported every
    # held position as flat -> the startup guard could trade on top of an open position.
    def handler(req):
        if req.url.path.endswith("/portfolio/balance"):
            return httpx.Response(200, json={"balance_dollars": "138.34"})
        return httpx.Response(200, json={"market_positions": [
            {"ticker": "KXMLBGAME-26JUN271610KCCWS-KC", "position_fp": "4.00",
             "resting_orders_count": 0},
            {"ticker": "KXMLBGAME-26JUN271910CHCMIL-CHC", "position_fp": "-4.00",
             "resting_orders_count": 0},
            {"ticker": "KX-FLAT", "position_fp": "0.00", "resting_orders_count": 0},
        ]})

    v = KalshiVenue(KalshiConfig(api_key_id="k", private_key_path="x"))
    v._client = _client(handler, v.cfg.api_base)
    v._auth_headers = lambda m, p: {}
    snap = asyncio.run(v.account_snapshot())
    held = {p.market_id: p.quantity for p in snap.positions}
    assert held == {"KXMLBGAME-26JUN271610KCCWS-KC": 4.0,
                    "KXMLBGAME-26JUN271910CHCMIL-CHC": -4.0}   # flat row dropped
    assert snap.balance == 138.34


def test_kalshi_scan_quotes_unbounded_scans_whole_board():
    # limit<=0 -> scan the ENTIRE feed: follow the cursor across pages until exhausted,
    # requesting full 1000-market pages (not a shrinking remainder).
    pages = [
        {"markets": [{"ticker": f"A{i}", "title": f"A{i}", "yes_bid": 40, "yes_ask": 41,
                      "no_bid": 59, "no_ask": 60} for i in range(1000)], "cursor": "CUR2"},
        {"markets": [{"ticker": f"B{i}", "title": f"B{i}", "yes_bid": 40, "yes_ask": 41,
                      "no_bid": 59, "no_ask": 60} for i in range(1000)], "cursor": "CUR3"},
        {"markets": [{"ticker": "C0", "title": "C0", "yes_bid": 40, "yes_ask": 41,
                      "no_bid": 59, "no_ask": 60}], "cursor": ""},
    ]
    seen = []

    def handler(req):
        seen.append(dict(req.url.params))
        cur = req.url.params.get("cursor")
        return httpx.Response(200, json=pages[{None: 0, "CUR2": 1, "CUR3": 2}[cur]])

    v = KalshiVenue(KalshiConfig(api_key_id="k", private_key_path="x"))
    v._client = _client(handler, v.cfg.api_base)
    v._auth_headers = lambda m, p: {}
    quotes = asyncio.run(v.scan_quotes(0))               # 0 = unbounded
    assert len(quotes) == 2001                            # all three pages
    assert all(p["limit"] == "1000" for p in seen)        # full pages, not a remainder
    assert len(seen) == 3


def test_kalshi_scan_quotes_stops_when_cursor_exhausted():
    # limit asks for 5000 but the feed ends after one short page with no cursor.
    def handler(req):
        return httpx.Response(200, json={"markets": [
            {"ticker": "A0", "title": "A0", "yes_bid": 40, "yes_ask": 41,
             "no_bid": 59, "no_ask": 60}], "cursor": ""})

    v = KalshiVenue(KalshiConfig(api_key_id="k", private_key_path="x"))
    v._client = _client(handler, v.cfg.api_base)
    v._auth_headers = lambda m, p: {}
    quotes = asyncio.run(v.scan_quotes(5000))
    assert len(quotes) == 1                              # no infinite loop on a short feed


def test_polymarket_scan_quotes_paginates_offset():
    def page(prefix, n):
        return {"markets": [{"slug": f"{prefix}{i}", "question": f"{prefix}{i}",
                             "bestAsk": "0.40", "bestBid": "0.38"} for i in range(n)]}
    seen_offsets = []

    def handler(req):
        # /v1/events is the per-game source (empty here — this test covers /v1/markets paging).
        if req.url.path.endswith("/v1/events"):
            return httpx.Response(200, json={"events": []})
        off = int(req.url.params.get("offset", "0"))
        seen_offsets.append(off)
        return httpx.Response(200, json=page("A", 500) if off == 0 else page("B", 100))

    cfg = QcexConfig()
    v = PolymarketUSVenue(cfg)
    v._gateway_client = _client(handler, cfg.gateway_base)

    quotes = asyncio.run(v.scan_quotes(1000))
    assert len(quotes) == 600                     # 500 + 100 across two pages
    assert seen_offsets == [0, 500]               # second call advanced the offset


def test_polymarket_scan_quotes_stops_when_offset_ignored():
    # Gateway ignores offset and returns the same first page -> dedupe, stop, no loop.
    def handler(req):
        return httpx.Response(200, json={"markets": [
            {"slug": f"A{i}", "question": f"A{i}", "bestAsk": "0.40", "bestBid": "0.38"}
            for i in range(500)]})

    cfg = QcexConfig()
    v = PolymarketUSVenue(cfg)
    v._gateway_client = _client(handler, cfg.gateway_base)
    quotes = asyncio.run(v.scan_quotes(5000))
    assert len(quotes) == 500                      # only the unique first page kept


def test_polymarket_scan_quotes_includes_per_game_event_markets():
    # The per-GAME markets live nested under /v1/events, not the flat /v1/markets feed.
    # scan_quotes must flatten them in (deduped) so live game lines are matchable.
    def handler(req):
        if req.url.path.endswith("/v1/events"):
            if int(req.url.params.get("offset", "0")) != 0:
                return httpx.Response(200, json={"events": []})
            return httpx.Response(200, json={"events": [
                {"ticker": "mlb-kc-tb-2026-06-25", "markets": [
                    {"slug": "aec-mlb-kc-tb-2026-06-25", "active": True, "closed": False,
                     "question": "Kansas City Royals vs. Tampa Bay Rays",
                     "endDate": "2026-06-25T23:59:00Z"},
                    {"slug": "astatc-mlb-kc-tb-2026-06-25-xi", "active": True, "closed": False,
                     "question": "Will KC vs TB go to extra innings?",
                     "endDate": "2026-07-09T16:10:00Z"},
                    # a settled sub-market inside the live event -> must be skipped
                    {"slug": "astatc-mlb-kc-tb-2026-06-25-yrfi", "active": True, "closed": True,
                     "question": "Yes run first inning?"},
                ]},
            ]})
        # flat /v1/markets feed: one futures market
        return httpx.Response(200, json={"markets": [
            {"slug": "tec-mlb-champ-2026-09-27-tb", "question": "World Series Champion",
             "bestAsk": "0.10", "bestBid": "0.08"}]})

    cfg = QcexConfig()
    v = PolymarketUSVenue(cfg)
    v._gateway_client = _client(handler, cfg.gateway_base)
    quotes = asyncio.run(v.scan_quotes(5000))
    ids = {q.market_id for q in quotes}
    assert "tec-mlb-champ-2026-09-27-tb" in ids          # flat feed kept
    assert "aec-mlb-kc-tb-2026-06-25" in ids             # per-game moneyline added
    assert "astatc-mlb-kc-tb-2026-06-25-xi" in ids       # per-game prop added
    assert "astatc-mlb-kc-tb-2026-06-25-yrfi" not in ids  # settled sub-market skipped


def test_kalshi_scan_quotes_targeted_close_window():
    # Targeted scan: max_close_ts is sent as a query param AND enforced client-side.
    # Near market is kept, far market is dropped, no-close-time market is kept.
    import time as _time
    from datetime import datetime, timezone

    now = _time.time()
    window = int(now + 2 * 86400)
    near = datetime.fromtimestamp(now + 3600, timezone.utc).isoformat()
    far = datetime.fromtimestamp(now + 10 * 86400, timezone.utc).isoformat()
    seen_params = []

    def handler(req):
        seen_params.append(dict(req.url.params))
        return httpx.Response(200, json={"markets": [
            {"ticker": "NEAR", "title": "near", "close_time": near},
            {"ticker": "FAR", "title": "far", "close_time": far},
            {"ticker": "NOCLOSE", "title": "noclose"},
        ], "cursor": ""})

    v = KalshiVenue(KalshiConfig(api_key_id="k", private_key_path="x"))
    v._client = _client(handler, v.cfg.api_base)
    v._auth_headers = lambda m, p: {}
    quotes = asyncio.run(v.scan_quotes(100, max_close_ts=window))
    ids = {q.market_id for q in quotes}
    assert ids == {"NEAR", "NOCLOSE"}                     # FAR filtered out
    assert seen_params[0]["max_close_ts"] == str(window)  # server-side param sent


def test_polymarket_scan_quotes_targeted_close_window():
    # Polymarket has no documented close-time query param -> client-side filter on endDate.
    import time as _time
    from datetime import datetime, timezone

    now = _time.time()
    window = int(now + 2 * 86400)
    near = datetime.fromtimestamp(now + 3600, timezone.utc).isoformat()
    far = datetime.fromtimestamp(now + 10 * 86400, timezone.utc).isoformat()

    def handler(req):
        return httpx.Response(200, json={"markets": [
            {"slug": "near", "question": "near", "bestAsk": "0.4", "bestBid": "0.38",
             "endDate": near},
            {"slug": "far", "question": "far", "bestAsk": "0.4", "bestBid": "0.38",
             "endDate": far},
            {"slug": "noclose", "question": "noclose", "bestAsk": "0.4", "bestBid": "0.38"},
        ]})

    cfg = QcexConfig()
    v = PolymarketUSVenue(cfg)
    v._gateway_client = _client(handler, cfg.gateway_base)
    quotes = asyncio.run(v.scan_quotes(100, max_close_ts=window))
    assert {q.market_id for q in quotes} == {"near", "noclose"}   # far filtered out


def test_kalshi_is_open_status():
    def handler(req):
        if "SETTLED" in str(req.url):
            return httpx.Response(200, json={"market": {"status": "settled"}})
        return httpx.Response(200, json={"market": {"status": "open"}})

    v = KalshiVenue(KalshiConfig(api_key_id="k", private_key_path="x"))
    v._client = _client(handler, v.cfg.api_base)
    v._auth_headers = lambda m, p: {}
    assert asyncio.run(v.is_open("KOPEN")) is True
    assert asyncio.run(v.is_open("KSETTLED")) is False


def test_polymarket_is_open_uses_book_state():
    # is_open reads /book state: terminal states -> False; live states -> True;
    # 404 / missing state -> None (unknown -> keep, never wrongly drop a live market).
    def handler(req):
        if "goneslug" in str(req.url):
            return httpx.Response(404, json={})
        if "deadslug" in str(req.url):
            return httpx.Response(200, json={"marketData": {"state": "MARKET_STATE_EXPIRED"}})
        if "haltedslug" in str(req.url):
            return httpx.Response(200, json={"marketData": {"state": "MARKET_STATE_HALTED"}})
        return httpx.Response(200, json={"marketData": {"state": "MARKET_STATE_OPEN"}})

    cfg = QcexConfig()
    v = PolymarketUSVenue(cfg)
    v._gateway_client = _client(handler, cfg.gateway_base)
    assert asyncio.run(v.is_open("liveslug")) is True
    assert asyncio.run(v.is_open("haltedslug")) is True   # temporary, not terminal -> keep
    assert asyncio.run(v.is_open("deadslug")) is False    # expired -> drop
    assert asyncio.run(v.is_open("goneslug")) is None     # unknown -> keep, never False


def test_polymarket_fetch_quote_uses_book_real_sizes():
    # Phase-2 deep fetch must hit /book (real qty), not /bbo (level counts).
    paths = []

    def handler(req):
        paths.append(req.url.path)
        return httpx.Response(200, json={"marketData": {
            "offers": [{"px": {"value": "0.56"}, "qty": "750"}],
            "bids": [{"px": {"value": "0.55"}, "qty": "1000"}],
        }})

    cfg = QcexConfig()
    v = PolymarketUSVenue(cfg)
    v._gateway_client = _client(handler, cfg.gateway_base)
    from bot.venues.base import RawMarket
    q = asyncio.run(v.fetch_quote(RawMarket(market_id="slug", title="t", raw={})))
    assert paths[0].endswith("/book")               # not /bbo
    assert q.yes_ask == 0.56 and q.yes_ask_size == 750
    assert round(q.no_ask, 6) == 0.45 and q.no_ask_size == 1000


def test_polymarket_scan_sends_end_date_max_and_captures_meta():
    import time as _time
    from datetime import datetime, timezone

    window = int(_time.time() + 2 * 86400)
    seen = []

    def handler(req):
        seen.append(dict(req.url.params))
        return httpx.Response(200, json={"markets": [
            {"slug": "m1", "question": "m1", "bestAsk": "0.4", "bestBid": "0.38",
             "orderPriceMinTickSize": "0.005", "minimumTradeQty": "0.01"},
        ]})

    cfg = QcexConfig()
    v = PolymarketUSVenue(cfg)
    v._gateway_client = _client(handler, cfg.gateway_base)
    asyncio.run(v.scan_quotes(50, max_close_ts=window))
    expected = datetime.fromtimestamp(window, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert seen[0]["endDateMax"] == expected          # server-side filter sent
    assert v._meta["m1"] == {"tick": 0.005, "min_qty": 0.01, "volume24hr": None}  # constraints + vol captured


def test_polymarket_place_order_snaps_to_tick_and_min_qty():
    cap = {}

    def handler(req):
        cap["body"] = json.loads(req.content)
        return httpx.Response(200, json=_exec_response(
            "o", "ORDER_STATE_FILLED", 4, 4, None))

    cfg = QcexConfig(api_key_id="k", secret_key="c2VjcmV0")
    v = PolymarketUSVenue(cfg)
    v._api_client = _client(handler, cfg.api_base)
    v._auth_headers = lambda m, p: {}
    v._meta["slug"] = {"tick": 0.05, "min_qty": 2.0}
    # price 0.62 snaps to nearest 0.05 tick -> 0.60; qty 5 floors to a 2.0 grid -> 4.
    asyncio.run(v.place_order("slug", Side.YES, "buy", 0.62, 5))
    assert cap["body"]["price"]["value"] == "0.6"
    assert cap["body"]["quantity"] == 4.0


def test_polymarket_place_order_no_meta_unchanged():
    # No captured constraints -> no rounding (whole contracts already valid).
    cap = {}

    def handler(req):
        cap["body"] = json.loads(req.content)
        return httpx.Response(200, json=_exec_response("o", "ORDER_STATE_FILLED", 3, 0.62, None))

    cfg = QcexConfig(api_key_id="k", secret_key="c2VjcmV0")
    v = PolymarketUSVenue(cfg)
    v._api_client = _client(handler, cfg.api_base)
    v._auth_headers = lambda m, p: {}
    asyncio.run(v.place_order("slug", Side.YES, "buy", 0.62, 3))
    assert cap["body"]["price"]["value"] == "0.62"
    assert cap["body"]["quantity"] == 3


def test_polymarket_preview_order_reports_expected_fill():
    # POST /v1/order/preview returns {order: {...}} with cumQuantity = expected fill.
    cap = {}

    def handler(req):
        cap["path"] = req.url.path
        cap["body"] = json.loads(req.content)
        return httpx.Response(200, json={"order": {
            "id": "pv", "state": "ORDER_STATE_NEW", "cumQuantity": 5,
            "avgPx": {"value": "0.60", "currency": "USD"}}})

    cfg = QcexConfig(api_key_id="k", secret_key="c2VjcmV0")
    v = PolymarketUSVenue(cfg)
    v._api_client = _client(handler, cfg.api_base)
    v._auth_headers = lambda m, p: {}
    r = asyncio.run(v.preview_order("slug", Side.YES, "buy", 0.60, 5))
    assert cap["path"].endswith("/v1/order/preview")
    assert cap["body"]["request"]["marketSlug"] == "slug"   # wrapped under `request`
    assert r.status.value == "FILLED" and r.filled == 5     # would fully fill


def test_polymarket_preview_partial_is_not_full_fill():
    def handler(req):
        return httpx.Response(200, json={"order": {
            "id": "pv", "state": "ORDER_STATE_NEW", "cumQuantity": 2}})

    cfg = QcexConfig(api_key_id="k", secret_key="c2VjcmV0")
    v = PolymarketUSVenue(cfg)
    v._api_client = _client(handler, cfg.api_base)
    v._auth_headers = lambda m, p: {}
    r = asyncio.run(v.preview_order("slug", Side.YES, "buy", 0.60, 5))
    assert r.filled == 2 and r.filled < 5                   # would NOT fully fill


def test_polymarket_place_order_polls_get_when_nonterminal():
    # A synchronous response that comes back non-terminal (PARTIAL) but with an id ->
    # GET /v1/order/{id} resolves the real terminal state (here, FILLED).
    def handler(req):
        if req.method == "GET" and "/v1/order/" in req.url.path:
            return httpx.Response(200, json={"order": {
                "id": "o9", "state": "ORDER_STATE_FILLED", "cumQuantity": 2,
                "avgPx": {"value": "0.40", "currency": "USD"}}})
        # POST /v1/orders: a non-terminal partial with an order id.
        return httpx.Response(200, json={"id": "o9", "executions": [
            {"type": "EXECUTION_TYPE_PARTIAL_FILL", "lastShares": "1",
             "order": {"id": "o9", "state": "ORDER_STATE_PARTIALLY_FILLED", "cumQuantity": 1}}]})

    cfg = QcexConfig(api_key_id="k", secret_key="c2VjcmV0")
    v = PolymarketUSVenue(cfg)
    v._api_client = _client(handler, cfg.api_base)
    v._auth_headers = lambda m, p: {}
    r = asyncio.run(v.place_order("slug", Side.YES, "buy", 0.40, 2))
    assert r.status.value == "FILLED" and r.filled == 2     # resolved via GET-by-id


def test_polymarket_post_only_maker_rests():
    # A post_only order is a resting MAKER: participateDontInitiate=true, GOOD_TILL_CANCEL
    # (the executor cancels it on timeout/drift — Polymarket has GTD disabled venue-side),
    # NO synchronousExecution. An accepted-but-unfilled maker -> RESTING.
    cap = {}

    def handler(req):
        cap["body"] = json.loads(req.content)
        return httpx.Response(200, json={"id": "mk1", "executions": []})   # rests, no fill

    cfg = QcexConfig(api_key_id="k", secret_key="c2VjcmV0")
    v = PolymarketUSVenue(cfg)
    v._api_client = _client(handler, cfg.api_base)
    v._auth_headers = lambda m, p: {}
    r = asyncio.run(v.place_order("slug", Side.YES, "buy", 0.62, 5,
                                  tif="gtc", post_only=True, expiration_ts=1782300000))
    assert cap["body"]["participateDontInitiate"] is True
    assert cap["body"]["tif"] == "TIME_IN_FORCE_GOOD_TILL_CANCEL"
    assert "goodTillTime" not in cap["body"]
    assert "synchronousExecution" not in cap["body"]   # a maker rests; not blocked
    assert r.status.value == "RESTING" and r.order_id == "mk1"


def test_kalshi_account_snapshot_balance_and_positions():
    def handler(req):
        if req.url.path.endswith("/portfolio/balance"):
            # Real shape: integer cents + exact dollar string (the latter preferred).
            return httpx.Response(200, json={"balance": 25000, "balance_dollars": "250.00"})
        if req.url.path.endswith("/portfolio/positions"):
            return httpx.Response(200, json={"market_positions": [
                {"ticker": "KHELD", "position": 5, "resting_orders_count": 0},
                {"ticker": "KREST", "position": 0, "resting_orders_count": 2},
                {"ticker": "KFLAT", "position": 0, "resting_orders_count": 0},  # dropped
            ]})
        return httpx.Response(404, json={})

    v = KalshiVenue(KalshiConfig(api_key_id="k", private_key_path="x"))
    v._client = _client(handler, v.cfg.api_base)
    v._auth_headers = lambda m, p: {}
    snap = asyncio.run(v.account_snapshot())
    assert snap.balance == 250.0
    assert {p.market_id for p in snap.open_positions} == {"KHELD", "KREST"}


def test_kalshi_account_snapshot_cents_fallback():
    # Older/edge payload with only integer cents -> divide by 100.
    def handler(req):
        if req.url.path.endswith("/portfolio/balance"):
            return httpx.Response(200, json={"balance": 11552})
        return httpx.Response(200, json={"market_positions": []})

    v = KalshiVenue(KalshiConfig(api_key_id="k", private_key_path="x"))
    v._client = _client(handler, v.cfg.api_base)
    v._auth_headers = lambda m, p: {}
    snap = asyncio.run(v.account_snapshot())
    assert snap.balance == 115.52 and snap.open_positions == []


def test_polymarket_account_snapshot_flat():
    # Real shapes: balances list (USD buyingPower) + empty positions object.
    def handler(req):
        if req.url.path.endswith("/account/balances"):
            return httpx.Response(200, json={"balances": [
                {"currency": "USD", "buyingPower": 142.53,
                 "assetNotional": 0, "openOrders": 0}]})
        if req.url.path.endswith("/portfolio/positions"):
            return httpx.Response(200, json={"positions": {}, "availablePositions": [],
                                             "eof": True})
        return httpx.Response(404, json={})

    cfg = QcexConfig(api_key_id="k", secret_key="c2VjcmV0")
    v = PolymarketUSVenue(cfg)
    v._api_client = _client(handler, cfg.api_base)
    v._auth_headers = lambda m, p: {}
    snap = asyncio.run(v.account_snapshot())
    assert snap.balance == 142.53 and snap.open_positions == []


def test_polymarket_account_snapshot_non_flat_via_balance_signal():
    # Positions endpoint empty, but the balance entry says held notional / open orders
    # -> a synthetic account-level position so the guard still trips.
    def handler(req):
        if req.url.path.endswith("/account/balances"):
            return httpx.Response(200, json={"balances": [
                {"currency": "USD", "buyingPower": 50.0,
                 "assetNotional": 120.0, "openOrders": 1}]})
        if req.url.path.endswith("/portfolio/positions"):
            return httpx.Response(200, json={"positions": {}, "availablePositions": []})
        return httpx.Response(404, json={})

    cfg = QcexConfig(api_key_id="k", secret_key="c2VjcmV0")
    v = PolymarketUSVenue(cfg)
    v._api_client = _client(handler, cfg.api_base)
    v._auth_headers = lambda m, p: {}
    snap = asyncio.run(v.account_snapshot())
    assert snap.balance == 50.0
    assert len(snap.open_positions) == 1
    assert snap.open_positions[0].resting_orders == 1


def test_place_order_requires_credentials():
    from bot.venues.base import OrderNotPermitted

    v = KalshiVenue(KalshiConfig(api_key_id="", private_key_path=""))
    with pytest.raises(OrderNotPermitted):
        asyncio.run(v.place_order("T", Side.YES, "buy", 0.5, 1))
