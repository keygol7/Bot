"""Tests for the pure venue normalization functions (no network, no SDKs)."""

from bot.venues.kalshi import (
    build_summary_title,
    is_multivariate,
    normalize_orderbook,
    normalize_summary,
)


def test_kalshi_title_appends_yes_sub_title():
    # Both sides of a game share the title; yes_sub_title names the YES outcome.
    m = {"ticker": "KXMLBGAME-26JUN16-LAA", "title": "Los Angeles A vs Arizona Winner?",
         "yes_sub_title": "Los Angeles A"}
    assert build_summary_title(m) == "Los Angeles A vs Arizona Winner? - Los Angeles A"


def test_kalshi_title_idempotent_suffix():
    # Always appends the outcome (disambiguates shared "X vs Y" titles)...
    m = {"title": "Will the Chargers win?", "yes_sub_title": "Chargers"}
    once = build_summary_title(m)
    assert once == "Will the Chargers win? - Chargers"
    # ...but re-normalizing an already-suffixed title doesn't double up.
    assert build_summary_title({"title": once, "yes_sub_title": "Chargers"}) == once
from bot.venues.polymarket_us import build_market_title, normalize_bbo


def test_polymarket_title_appends_outcome():
    m = {
        "question": "World Series Champion",
        "marketSides": [
            {"long": True, "team": {"name": "New York Yankees"}},
            {"long": False, "team": {"name": "Los Angeles Dodgers"}},
        ],
    }
    assert build_market_title(m) == "World Series Champion - New York Yankees"


def test_polymarket_title_appends_outcome_even_if_in_question():
    m = {"question": "Will the Chargers win?", "marketSides": [{"long": True, "description": "Chargers"}]}
    # The outcome disambiguates the YES side, so it's appended.
    assert build_market_title(m) == "Will the Chargers win? - Chargers"


def test_polymarket_title_falls_back_to_question():
    assert build_market_title({"question": "Plain question"}) == "Plain question"


def test_kalshi_multivariate_detection():
    assert is_multivariate("KXMVESPORTSMULTIGAMEEXTENDED-S2026E41-DAE")
    assert is_multivariate("KXMVECROSSCATEGORY-S2026B05-2A9")
    assert not is_multivariate("KXNBA-25DEC-LAL")   # ordinary single-event market
    assert not is_multivariate("FED-MAR26")


def test_kalshi_summary_price_only():
    # Phase-1 scan: prices from the /markets summary (cents), sizes unknown (0).
    m = {"ticker": "FED-MAR26", "title": "Fed cut", "yes_ask": 41, "no_ask": 60}
    quote = normalize_summary(m)
    assert quote.market_id == "FED-MAR26"
    assert quote.yes_ask == 0.41 and quote.no_ask == 0.60
    assert quote.yes_ask_size == 0.0 and quote.no_ask_size == 0.0


def test_kalshi_summary_zero_price_means_no_quote():
    m = {"ticker": "X", "title": "X", "yes_ask": 0, "no_ask": 55}
    quote = normalize_summary(m)
    assert quote.yes_ask is None
    assert quote.no_ask == 0.55


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


def test_polymarket_bbo_normalization():
    # Polymarket US BBO: bestAsk = YES ask; NO ask = 1 - bestBid. v1Amount objects.
    md = {
        "marketSlug": "fed-cuts-march-2026",
        "bestAsk": {"value": "0.62", "currency": "USD"},
        "bestBid": {"value": "0.60", "currency": "USD"},
        "askDepth": 40,
        "bidDepth": 75,
    }
    quote = normalize_bbo("fed-cuts-march-2026", "Fed cuts March 2026", md, event_key="E2")
    assert quote.venue == "polymarket_us" and quote.event_key == "E2"
    assert quote.yes_ask == 0.62 and quote.yes_ask_size == 40
    assert round(quote.no_ask, 6) == 0.40 and quote.no_ask_size == 75  # 1 - 0.60


def test_polymarket_bbo_handles_missing_sides():
    md = {"marketSlug": "x", "bestAsk": None, "bestBid": {"value": "0.30"}, "askDepth": 0, "bidDepth": 12}
    quote = normalize_bbo("x", "x", md)
    assert quote.yes_ask is None              # no ask -> can't buy YES
    assert round(quote.no_ask, 6) == 0.70 and quote.no_ask_size == 12


def test_polymarket_bbo_accepts_bare_numbers():
    # Some payloads may carry bare numeric prices instead of v1Amount objects.
    md = {"bestAsk": 0.55, "bestBid": 0.53, "askDepth": 5, "bidDepth": 8}
    quote = normalize_bbo("s", "s", md)
    assert quote.yes_ask == 0.55
    assert round(quote.no_ask, 6) == 0.47
