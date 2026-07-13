import asyncio
import json
from types import SimpleNamespace

from bot.config.settings import PolymarketComConfig, load_settings
from bot.data.store import Store
from bot.execution.orders import OrderStatus
from bot.models import Side
from bot.venues.polymarket_com import (
    PolymarketComVenue,
    _compact_market,
    normalize_books,
    normalize_market,
    parse_order_response,
    parse_private_order,
)


MARKET = {
    "slug": "btc-updown-5m-1784002500",
    "conditionId": "0xcondition",
    "question": "Bitcoin Up or Down - July 14, 12:15AM-12:20AM ET",
    "description": "Resolves Up if the ending BTC price is at least the starting price.",
    "outcomes": '["Up", "Down"]',
    "clobTokenIds": '["up-token", "down-token"]',
    "endDate": "2026-07-14T04:20:00Z",
    "bestAsk": 0.51,
    "bestBid": 0.49,
    "orderPriceMinTickSize": 0.01,
    "orderMinSize": 5,
    "acceptingOrders": True,
    "closed": False,
}


def test_normalize_market_maps_first_outcome_to_yes():
    q = normalize_market(MARKET)
    assert q.venue == "polymarket_com"
    assert q.market_id == MARKET["slug"]
    assert q.title.endswith("- Up")
    assert q.yes_ask == 0.51 and q.no_ask == 0.51
    assert q.state == "MARKET_STATE_OPEN"


def test_compact_market_drops_nested_gamma_payloads():
    compact = _compact_market({**MARKET, "events": [{"large": "payload"}]})
    assert compact["slug"] == MARKET["slug"]
    assert "events" not in compact


def test_normalize_two_token_books_uses_real_depth_and_sorts():
    q = normalize_books(
        MARKET["slug"], "BTC - Up",
        {"asks": [{"price": "0.53", "size": "3"},
                  {"price": "0.51", "size": "8"}], "timestamp": "1000000000000"},
        {"asks": [{"price": "0.50", "size": "11"}], "timestamp": "1000000000100"},
    )
    assert (q.yes_ask, q.yes_ask_size) == (0.51, 8.0)
    assert (q.no_ask, q.no_ask_size) == (0.50, 11.0)
    assert q.yes_ask_levels == ((0.51, 8.0), (0.53, 3.0))
    assert q.exchange_ts == 1_000_000_000.0
    assert q.state == "MARKET_STATE_OPEN"


def test_order_response_is_fail_closed_for_crypto_delay():
    assert parse_order_response(
        {"success": True, "status": "matched", "orderID": "o1"}, 5, 0.52
    )[:2] == (OrderStatus.FILLED, 5)
    assert parse_order_response(
        {"success": True, "status": "unmatched", "orderID": "o2"}, 5, 0.52
    )[0] is OrderStatus.KILLED
    assert parse_order_response(
        {"success": True, "status": "delayed", "orderID": "o3"}, 5, 0.52
    )[0] is OrderStatus.ERROR
    assert parse_order_response(
        {"success": True, "status": "live", "orderID": "o4"}, 5, 0.52,
        resting=True,
    )[0] is OrderStatus.RESTING


def test_private_order_updates_are_converted_from_cumulative_to_delta():
    seen = {}
    base = {
        "event_type": "order", "id": "o1", "original_size": "5", "price": "0.51",
    }
    first = parse_private_order({**base, "type": "UPDATE", "size_matched": "2"}, seen)
    second = parse_private_order({**base, "type": "UPDATE", "size_matched": "5"}, seen)
    assert first.exec_type == "PARTIAL_FILL" and first.last_shares == 2
    assert second.exec_type == "FILL" and second.last_shares == 3


def test_book_price_change_mutates_the_correct_outcome_token():
    cfg = SimpleNamespace(read_rate_per_min=600)
    venue = PolymarketComVenue(cfg)
    venue._remember(MARKET)
    books = {}
    venue._apply_book_message(books, {
        "event_type": "book", "asset_id": "up-token",
        "bids": [], "asks": [{"price": "0.51", "size": "8"}],
    })
    venue._apply_book_message(books, {
        "event_type": "price_change", "price_changes": [{
            "asset_id": "up-token", "side": "SELL", "price": "0.51", "size": "0"
        }, {
            "asset_id": "up-token", "side": "SELL", "price": "0.52", "size": "9"
        }],
    })
    assert books["up-token"]["asks"] == {0.52: 9.0}


def test_book_updates_keep_per_token_timestamps_and_all_changed_markets():
    cfg = SimpleNamespace(read_rate_per_min=600)
    venue = PolymarketComVenue(cfg)
    venue._remember(MARKET)
    books = {}
    venue._apply_book_message(books, {
        "event_type": "book", "asset_id": "up-token",
        "timestamp": "1783972800100", "bids": [],
        "asks": [{"price": "0.51", "size": "8"}],
    })
    venue._apply_book_message(books, {
        "event_type": "book", "asset_id": "down-token",
        "timestamp": "1783972800200", "bids": [],
        "asks": [{"price": "0.50", "size": "9"}],
    })
    changed = venue._apply_book_message(books, {
        "event_type": "price_change", "timestamp": "1783972800900",
        "price_changes": [{"asset_id": "up-token", "side": "SELL",
                           "price": "0.52", "size": "10"}],
    })
    assert changed == {MARKET["slug"]}
    yes = venue._ws_book("up-token", books["up-token"])
    no = venue._ws_book("down-token", books["down-token"])
    assert yes["timestamp"] == "1783972800900"
    assert no["timestamp"] == "1783972800200"
    q = normalize_books(MARKET["slug"], "BTC - Up", yes, no)
    assert q.exchange_ts == 1783972800.2


def test_config_is_separate_disabled_and_full_market_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("POLYMARKET_COM_ENABLED", raising=False)
    settings = load_settings(str(tmp_path / "missing.env"))
    assert settings.polymarket_com.enabled is False
    assert settings.polymarket_com.is_trading_configured is False
    assert settings.polymarket_com.market_slug_prefixes == ("*",)
    assert settings.polymarket_com.lookahead_minutes == 0

    key = tmp_path / "key"
    key.write_text("0xprivate")
    cfg = PolymarketComConfig(
        enabled=True, private_key_path=str(key), api_key="k", api_secret="s",
        api_passphrase="p", funder_address="0x" + "1" * 40, signature_type=3,
    )
    assert cfg.is_trading_configured is True


def test_full_market_scope_accepts_non_bitcoin_market():
    cfg = SimpleNamespace(read_rate_per_min=600, market_slug_prefixes=("*",))
    venue = PolymarketComVenue(cfg)
    market = {
        **MARKET,
        "slug": "will-example-candidate-win-2026",
        "endDate": "2026-12-31T00:00:00Z",
    }
    assert venue._included(market, 1_783_900_000, None)


def test_optional_prefix_scope_still_narrows_discovery():
    cfg = SimpleNamespace(
        read_rate_per_min=600,
        market_slug_prefixes=("btc-updown-15m-",),
    )
    venue = PolymarketComVenue(cfg)
    market = {**MARKET, "slug": "will-example-candidate-win-2026"}
    assert not venue._included(market, 1_783_900_000, None)


def test_full_market_discovery_uses_keyset_cursor():
    class Response:
        def __init__(self, body):
            self.body = body

        def raise_for_status(self):
            return None

        def json(self):
            return self.body

    class Public:
        def __init__(self):
            self.calls = []

        async def get(self, url, *, params):
            self.calls.append((url, params))
            market = {
                **MARKET,
                "slug": f"non-bitcoin-market-{len(self.calls)}",
                "endDate": "2099-12-31T00:00:00Z",
            }
            if len(self.calls) == 1:
                return Response({"markets": [market], "next_cursor": "cursor-2"})
            return Response({"markets": [market]})

    cfg = SimpleNamespace(
        read_rate_per_min=600,
        market_slug_prefixes=("*",),
        lookahead_minutes=0,
        gamma_base="https://gamma.example",
    )
    venue = PolymarketComVenue(cfg)
    public = Public()
    venue._public_client = public
    markets = asyncio.run(venue._gamma_markets())
    assert len(markets) == 2
    assert public.calls[0][0].endswith("/markets/keyset")
    assert "offset" not in public.calls[0][1]
    assert public.calls[1][1]["after_cursor"] == "cursor-2"


def test_minimum_order_is_rejected_before_sdk_call():
    cfg = SimpleNamespace(is_trading_configured=True, read_rate_per_min=600)
    venue = PolymarketComVenue(cfg)
    venue._remember(MARKET)
    result = asyncio.run(venue.place_order(MARKET["slug"], Side.YES, "buy", 0.51, 2))
    assert result.status is OrderStatus.REJECTED
    assert "minimum size 5" in result.raw["error"]


def test_deterministic_join_matches_exact_btc_15m_window(tmp_path):
    store = Store(str(tmp_path / "bot.db"))
    try:
        # 00:45 ET on July 13, 2026 is 04:45 UTC. Polymarket's suffix is the
        # 04:30 UTC start, exactly 15 minutes earlier.
        kalshi = "KXBTC15M-26JUL130045-45"
        poly = "btc-updown-15m-1783917000"
        store.upsert_market("kalshi", kalshi, "BTC price up in next 15 mins?")
        store.upsert_market("polymarket_com", poly, "Bitcoin Up or Down - Up")
        pairs = store.idparse_pairs()
        assert ("kalshi", kalshi, "polymarket_com", poly,
                f"kalshi:{kalshi}|polymarket_com:{poly}") in pairs
    finally:
        store.close()


def test_deterministic_join_rejects_adjacent_btc_window(tmp_path):
    store = Store(str(tmp_path / "bot.db"))
    try:
        store.upsert_market("kalshi", "KXBTC15M-26JUL130045-45", "BTC up?")
        # Starts at 04:45 UTC and ends at 05:00, so it is the next contract.
        store.upsert_market(
            "polymarket_com", "btc-updown-15m-1783917900", "Bitcoin Up or Down - Up"
        )
        assert not any(p[2] == "polymarket_com" for p in store.idparse_pairs())
    finally:
        store.close()


def test_deterministic_join_supports_all_shared_crypto_15m_families(tmp_path):
    store = Store(str(tmp_path / "bot.db"))
    try:
        for asset in ("BTC", "ETH", "SOL", "XRP", "DOGE", "HYPE", "BNB"):
            kalshi = f"KX{asset}15M-26JUL130045-45"
            poly = f"{asset.lower()}-updown-15m-1783917000"
            store.upsert_market("kalshi", kalshi, f"{asset} up?")
            store.upsert_market("polymarket_com", poly, f"{asset} Up or Down - Up")
        pairs = store.crypto_short_pairs()
        assert len(pairs) == 7
        assert {p[1][2:].split("15M-", 1)[0] for p in pairs} == {
            "BTC", "ETH", "SOL", "XRP", "DOGE", "HYPE", "BNB"
        }
    finally:
        store.close()


def test_deterministic_join_never_crosses_crypto_assets(tmp_path):
    store = Store(str(tmp_path / "bot.db"))
    try:
        store.upsert_market("kalshi", "KXETH15M-26JUL130045-45", "ETH up?")
        store.upsert_market(
            "polymarket_com", "btc-updown-15m-1783917000", "Bitcoin up?"
        )
        assert store.crypto_short_pairs() == []
    finally:
        store.close()


def test_market_fixture_stays_valid_json_strings():
    # Guard against accidentally changing the real Gamma wire shape in this fixture.
    assert json.loads(MARKET["outcomes"]) == ["Up", "Down"]
