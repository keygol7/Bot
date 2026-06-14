"""Private fill stream: tracker aggregation/confirm + per-venue parsers."""

import asyncio

from bot.execution.orders import OrderStatus
from bot.streaming.fills import FillEvent, FillTracker
from bot.venues.kalshi import parse_fill
from bot.venues.polymarket_us import parse_execution


def test_tracker_confirms_full_fill():
    async def scenario():
        t = FillTracker()
        await t.apply(FillEvent("poly", "o1", "FILL", last_shares=2, last_px=0.55))
        return await t.confirm("poly", "o1", requested=2, timeout=0.1)

    status, filled, avg = asyncio.run(scenario())
    assert status is OrderStatus.FILLED and filled == 2 and avg == 0.55


def test_tracker_confirms_killed():
    async def scenario():
        t = FillTracker()
        await t.apply(FillEvent("poly", "o2", "CANCELED", last_shares=0))
        return await t.confirm("poly", "o2", requested=2, timeout=0.1)

    status, filled, _ = asyncio.run(scenario())
    assert status is OrderStatus.KILLED and filled == 0


def test_tracker_timeout_is_error():
    async def scenario():
        t = FillTracker()
        return await t.confirm("poly", "missing", requested=2, timeout=0.05)

    status, filled, _ = asyncio.run(scenario())
    assert status is OrderStatus.ERROR and filled == 0


def test_tracker_aggregates_partials_to_full():
    async def scenario():
        t = FillTracker()
        await t.apply(FillEvent("kalshi", "o3", "PARTIAL_FILL", last_shares=1, last_px=0.40))
        await t.apply(FillEvent("kalshi", "o3", "PARTIAL_FILL", last_shares=1, last_px=0.42))
        return await t.confirm("kalshi", "o3", requested=2, timeout=0.1)

    status, filled, avg = asyncio.run(scenario())
    assert status is OrderStatus.FILLED and filled == 2 and round(avg, 3) == 0.41


def test_parse_polymarket_execution():
    msg = {"orderSubscriptionUpdate": {"execution": {
        "id": "exec-1", "order": {"id": "order-123"},
        "lastShares": "0.25", "lastPx": {"value": "0.555"}, "type": "EXECUTION_TYPE_PARTIAL_FILL"}}}
    ev = parse_execution(msg)
    assert ev.venue == "polymarket_us" and ev.order_id == "order-123"
    assert ev.exec_type == "PARTIAL_FILL" and ev.last_shares == 0.25 and ev.last_px == 0.555


def test_parse_polymarket_non_execution_is_none():
    assert parse_execution({"heartbeat": {}}) is None
    assert parse_execution({"orderSubscriptionSnapshot": {"orders": []}}) is None


def test_parse_kalshi_fill():
    msg = {"type": "fill", "msg": {"order_id": "k-1", "count": 3, "yes_price": 40}}
    ev = parse_fill(msg)
    assert ev.venue == "kalshi" and ev.order_id == "k-1"
    assert ev.last_shares == 3 and ev.last_px == 0.40
