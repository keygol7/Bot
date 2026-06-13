"""Tests for the pure venue normalization functions (no network, no SDKs)."""

from bot.venues.kalshi import normalize_orderbook
from bot.venues.polymarket_us import build_quote, normalize_clob_book


def test_kalshi_orderbook_normalization():
    # yes = resting bids to buy YES; no = resting bids to buy NO. Prices in cents.
    ob = {"yes": [[40, 100], [39, 50]], "no": [[55, 80], [54, 20]]}
    quote = normalize_orderbook("FED-MAR26", "Fed cut in March", ob, event_key="E1")
    # Buy YES by crossing best NO bid (55c): yes_ask = 1 - 0.55 = 0.45, size 80.
    assert quote.yes_ask == 0.45 and quote.yes_ask_size == 80
    # Buy NO by crossing best YES bid (40c): no_ask = 1 - 0.40 = 0.60, size 100.
    assert quote.no_ask == 0.60 and quote.no_ask_size == 100
    assert quote.venue == "kalshi" and quote.event_key == "E1"


def test_kalshi_orderbook_one_sided():
    ob = {"yes": [[42, 10]], "no": []}
    quote = normalize_orderbook("X", "X", ob)
    assert quote.yes_ask is None            # no NO bids -> can't buy YES
    assert quote.no_ask == 0.58 and quote.no_ask_size == 10


def test_clob_book_best_levels():
    book = {
        "bids": [{"price": "0.44", "size": "30"}, {"price": "0.43", "size": "10"}],
        "asks": [{"price": "0.47", "size": "5"}, {"price": "0.46", "size": "50"}],
    }
    best_bid, best_ask = normalize_clob_book(book)
    assert best_bid.price == 0.44 and best_bid.size == 30
    assert best_ask.price == 0.46 and best_ask.size == 50


def test_clob_skips_zero_size():
    book = {"bids": [], "asks": [{"price": "0.46", "size": "0"}, {"price": "0.48", "size": "7"}]}
    _, best_ask = normalize_clob_book(book)
    assert best_ask.price == 0.48


def test_build_quote_with_both_token_books():
    yes_book = {"bids": [{"price": "0.44", "size": "30"}], "asks": [{"price": "0.46", "size": "50"}]}
    no_book = {"bids": [{"price": "0.50", "size": "20"}], "asks": [{"price": "0.53", "size": "40"}]}
    quote = build_quote("0xcond", "Some market", yes_book, no_book, event_key="E2")
    assert quote.yes_ask == 0.46 and quote.yes_ask_size == 50
    assert quote.no_ask == 0.53 and quote.no_ask_size == 40
    assert quote.event_key == "E2"


def test_build_quote_synthesizes_no_from_yes_bid():
    yes_book = {"bids": [{"price": "0.44", "size": "30"}], "asks": [{"price": "0.46", "size": "50"}]}
    quote = build_quote("0xcond", "Some market", yes_book, no_book=None)
    assert quote.yes_ask == 0.46
    assert round(quote.no_ask, 6) == 0.56 and quote.no_ask_size == 30  # 1 - 0.44
