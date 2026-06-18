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


def test_kalshi_buy_yes_request_and_fill():
    cap = {}

    def handler(req):
        cap["body"] = json.loads(req.content)
        return httpx.Response(201, json={"order": {
            "order_id": "abc", "fill_count_fp": "2.00", "yes_price_dollars": "0.40"}})

    v = KalshiVenue(KalshiConfig(api_key_id="k", private_key_path="x"))
    v._client = _client(handler, v.cfg.api_base)
    v._auth_headers = lambda m, p: {}  # skip real signing for this test

    r = asyncio.run(v.place_order("KTICK", Side.YES, "buy", 0.40, 2))
    assert cap["body"]["ticker"] == "KTICK"
    assert cap["body"]["action"] == "buy" and cap["body"]["side"] == "yes"
    assert cap["body"]["yes_price"] == 40 and "no_price" not in cap["body"]
    assert cap["body"]["count"] == 2 and cap["body"]["time_in_force"] == "fill_or_kill"
    assert r.status.value == "FILLED" and r.filled == 2.0 and r.avg_price == 0.40


def test_kalshi_buy_no_uses_no_price_cents():
    cap = {}

    def handler(req):
        cap["body"] = json.loads(req.content)
        return httpx.Response(201, json={"order": {"order_id": "x", "fill_count_fp": "0.00"}})

    v = KalshiVenue(KalshiConfig(api_key_id="k", private_key_path="x"))
    v._client = _client(handler, v.cfg.api_base)
    v._auth_headers = lambda m, p: {}
    r = asyncio.run(v.place_order("KTICK", Side.NO, "buy", 0.55, 2))
    assert cap["body"]["side"] == "no" and cap["body"]["no_price"] == 55
    assert r.status.value == "KILLED"   # fill_count 0 on FoK


def test_polymarket_buy_no_request_and_fill():
    cap = {}

    def handler(req):
        cap["body"] = json.loads(req.content)
        return httpx.Response(200, json={"order": {
            "id": "o1", "state": "ORDER_STATE_FILLED", "cumQuantity": 2,
            "avgPx": {"value": "0.55", "currency": "USD"}}})

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
    assert r.status.value == "FILLED" and r.filled == 2


def test_polymarket_buy_yes_price_is_yes_side():
    cap = {}

    def handler(req):
        cap["body"] = json.loads(req.content)
        return httpx.Response(200, json={"order": {"id": "o", "state": "ORDER_STATE_CANCELED", "cumQuantity": 0}})

    cfg = QcexConfig(api_key_id="k", secret_key="c2VjcmV0")
    v = PolymarketUSVenue(cfg)
    v._api_client = _client(handler, cfg.api_base)
    v._auth_headers = lambda m, p: {}
    r = asyncio.run(v.place_order("slug", Side.YES, "buy", 0.62, 1))
    assert cap["body"]["intent"] == "ORDER_INTENT_BUY_LONG"
    assert cap["body"]["price"]["value"] == "0.62"
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


def test_polymarket_is_open_uses_bbo_never_false():
    # The gateway 404s the bare /v1/markets/{slug}; is_open must use /bbo and never
    # return False (a quirky 404 must not drop a live market -> None = keep).
    def handler(req):
        if "goneslug" in str(req.url):
            return httpx.Response(404, json={})
        return httpx.Response(200, json={"marketData": {"bestAsk": "0.4"}})

    cfg = QcexConfig()
    v = PolymarketUSVenue(cfg)
    v._gateway_client = _client(handler, cfg.gateway_base)
    assert asyncio.run(v.is_open("liveslug")) is True
    assert asyncio.run(v.is_open("goneslug")) is None     # unknown -> keep, never False


def test_kalshi_account_snapshot_balance_and_positions():
    def handler(req):
        if req.url.path.endswith("/portfolio/balance"):
            return httpx.Response(200, json={"balance": 25000})        # cents -> $250.00
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


def test_polymarket_account_snapshot_balance_and_positions():
    def handler(req):
        if req.url.path.endswith("/portfolio/balance"):
            return httpx.Response(200, json={"availableBalance": "300.50"})
        if req.url.path.endswith("/portfolio/positions"):
            return httpx.Response(200, json={"positions": [
                {"marketSlug": "held", "quantity": "4"},
                {"slug": "flat", "quantity": "0", "openOrders": 0},   # dropped
            ]})
        return httpx.Response(404, json={})

    cfg = QcexConfig(api_key_id="k", secret_key="c2VjcmV0")
    v = PolymarketUSVenue(cfg)
    v._api_client = _client(handler, cfg.api_base)
    v._auth_headers = lambda m, p: {}
    snap = asyncio.run(v.account_snapshot())
    assert snap.balance == 300.50
    assert [p.market_id for p in snap.open_positions] == ["held"]


def test_place_order_requires_credentials():
    from bot.venues.base import OrderNotPermitted

    v = KalshiVenue(KalshiConfig(api_key_id="", private_key_path=""))
    with pytest.raises(OrderNotPermitted):
        asyncio.run(v.place_order("T", Side.YES, "buy", 0.5, 1))
