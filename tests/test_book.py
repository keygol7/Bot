import pytest

from bot.data.book import BookStore, OrderBook
from bot.models import PriceLevel


def test_best_levels_sorted_correctly():
    store = BookStore()
    store.update(
        "kalshi", "A",
        title="Market A",
        yes_bids=[PriceLevel(0.40, 10), PriceLevel(0.42, 5), PriceLevel(0.38, 20)],
        yes_asks=[PriceLevel(0.46, 8), PriceLevel(0.45, 3), PriceLevel(0.47, 100)],
    )
    book = store.get("kalshi", "A")
    assert book.best_yes_bid().price == 0.42  # highest bid
    assert book.best_yes_ask().price == 0.45  # lowest ask


def test_no_ask_synthesized_from_yes_bid():
    book = OrderBook(venue="kalshi", market_id="A")
    book.yes_bids.replace([PriceLevel(0.40, 25)])
    no = book.best_no_ask()
    assert round(no.price, 6) == 0.60  # 1 - 0.40
    assert no.size == 25


def test_no_ask_prefers_real_book():
    book = OrderBook(venue="poly", market_id="A")
    book.yes_bids.replace([PriceLevel(0.40, 25)])
    book.no_asks.replace([PriceLevel(0.58, 12)])
    no = book.best_no_ask()
    assert no.price == 0.58 and no.size == 12  # real NO book wins over synthesis


def test_zero_size_levels_dropped():
    book = OrderBook(venue="kalshi", market_id="A")
    book.yes_asks.replace([PriceLevel(0.45, 0), PriceLevel(0.46, 5)])
    assert book.best_yes_ask().price == 0.46


def test_to_quote_roundtrip():
    store = BookStore()
    store.update(
        "poly", "B", title="Market B", event_key="E1",
        yes_asks=[PriceLevel(0.50, 40)],
        no_asks=[PriceLevel(0.49, 60)],
    )
    quote = store.get("poly", "B").to_quote()
    assert quote.venue == "poly"
    assert quote.event_key == "E1"
    assert quote.yes_ask == 0.50 and quote.yes_ask_size == 40
    assert quote.no_ask == 0.49 and quote.no_ask_size == 60
    assert len(store) == 1


def test_price_level_validation():
    with pytest.raises(ValueError):
        PriceLevel(1.5, 10)
    with pytest.raises(ValueError):
        PriceLevel(0.5, -1)
