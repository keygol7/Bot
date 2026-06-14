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


def test_place_order_requires_credentials():
    from bot.venues.base import OrderNotPermitted

    v = KalshiVenue(KalshiConfig(api_key_id="", private_key_path=""))
    with pytest.raises(OrderNotPermitted):
        asyncio.run(v.place_order("T", Side.YES, "buy", 0.5, 1))
