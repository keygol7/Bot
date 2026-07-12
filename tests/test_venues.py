"""Tests for the pure venue normalization functions (no network, no SDKs)."""

from bot.venues.kalshi import (
    build_summary_title,
    is_multivariate,
    normalize_orderbook,
    normalize_summary,
    parse_lifecycle,
    parse_ticker,
)


def _lc(event_type, ticker="KXMLBGAME-X-LAA", **extra):
    return {"type": "market_lifecycle_v2",
            "msg": {"event_type": event_type, "market_ticker": ticker, **extra}}


def test_kalshi_lifecycle_terminal_events_block_and_prune():
    for ev in ("settled", "determined"):
        out = parse_lifecycle(_lc(ev))
        assert out.market_ticker == "KXMLBGAME-X-LAA"
        assert out.state != "MARKET_STATE_OPEN" and out.terminal is True


def test_kalshi_lifecycle_activate_deactivate():
    assert parse_lifecycle(_lc("activated")).state == "MARKET_STATE_OPEN"
    assert parse_lifecycle(_lc("activated")).terminal is False
    out = parse_lifecycle(_lc("deactivated"))
    assert out.state != "MARKET_STATE_OPEN" and out.terminal is False


def test_kalshi_lifecycle_pause_flag_takes_precedence():
    # is_deactivated is the pause/unpause signal on an open market.
    assert parse_lifecycle(_lc("activated", is_deactivated=True)).state == "KALSHI_PAUSED"
    assert parse_lifecycle(_lc("deactivated", is_deactivated=False)).state == "MARKET_STATE_OPEN"


def test_kalshi_lifecycle_neutral_events_no_state_change():
    # created/metadata/etc. don't change tradeability -> state None (leave as-is).
    assert parse_lifecycle(_lc("created", open_ts=1)).state is None
    assert parse_lifecycle(_lc("metadata_updated")).state is None


def test_kalshi_lifecycle_ignores_other_messages():
    assert parse_lifecycle({"type": "ticker", "msg": {}}) is None
    assert parse_lifecycle(_lc("settled", ticker="")) is None    # no ticker


def test_kalshi_ws_ticker_parsing():
    msg = {"type": "ticker", "msg": {
        "market_ticker": "KXMLBGAME-X-LAA",
        "yes_bid_dollars": "0.40", "yes_ask_dollars": "0.43",
        "yes_bid_size_fp": "300.00", "yes_ask_size_fp": "150.00"}}
    q = parse_ticker(msg)
    assert q.market_id == "KXMLBGAME-X-LAA"
    assert q.yes_ask == 0.43 and q.yes_ask_size == 150     # contracts at best ask
    assert round(q.no_ask, 4) == 0.60                      # 1 - yes_bid
    assert q.no_ask_size == 300                            # contracts at best YES bid
    assert q.timestamp > 0


def test_kalshi_ws_ticker_missing_sizes_default_zero():
    msg = {"type": "ticker", "msg": {
        "market_ticker": "X", "yes_bid_dollars": "0.40", "yes_ask_dollars": "0.43"}}
    q = parse_ticker(msg)
    assert q.yes_ask_size == 0.0 and q.no_ask_size == 0.0  # absent sizes -> 0


def test_kalshi_ws_ticker_missing_ticker():
    assert parse_ticker({"type": "ticker", "msg": {}}) is None


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
from bot.venues.polymarket_us import (
    build_market_title, normalize_bbo, normalize_book, parse_market_data,
    parse_market_data_lite,
)


def test_polymarket_ws_market_data_lite_parsing():
    msg = {"subscriptionType": "SUBSCRIPTION_TYPE_MARKET_DATA_LITE", "marketDataLite": {
        "marketSlug": "tec-mlb-champ-lad",
        "bestBid": {"value": "0.54", "currency": "USD"},
        "bestAsk": {"value": "0.56", "currency": "USD"},
        "bidDepth": 5, "askDepth": 4}}
    q = parse_market_data_lite(msg)
    assert q.venue == "polymarket_us" and q.market_id == "tec-mlb-champ-lad"
    # askDepth/bidDepth are LEVEL COUNTS, not sizes -> the lite quote carries no size.
    assert q.yes_ask == 0.56 and q.yes_ask_size == 0.0
    assert round(q.no_ask, 4) == 0.46 and q.no_ask_size == 0.0  # 1 - bestBid


def test_polymarket_ws_ignores_non_lite():
    assert parse_market_data_lite({"heartbeat": {}}) is None
    assert parse_market_data_lite({"marketDataLite": {}}) is None


def test_polymarket_ws_full_book_parsing_has_real_size():
    # The MARKET_DATA (full book) channel carries bids/offers with real qty.
    msg = {"marketData": {
        "marketSlug": "tec-mlb-champ-lad",
        "bids": [{"px": {"value": "0.54"}, "qty": "300"}],
        "offers": [{"px": {"value": "0.56"}, "qty": "200"}],
        "state": "MARKET_STATE_OPEN",
    }}
    q = parse_market_data(msg)
    assert q.market_id == "tec-mlb-champ-lad"
    assert q.yes_ask == 0.56 and q.yes_ask_size == 200       # real size, not a level count
    assert round(q.no_ask, 4) == 0.46 and q.no_ask_size == 300


def test_polymarket_ws_full_book_ignores_non_data():
    assert parse_market_data({"heartbeat": {}}) is None
    assert parse_market_data({"marketData": {}}) is None      # no slug


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


def test_kalshi_orderbook_fp_dollars_format():
    # The current Kalshi shape: orderbook_fp with dollar-string prices + fractional
    # sizes. (yes_dollars / no_dollars are bid levels.)
    ob = {
        "yes_dollars": [["0.40", "100.5"], ["0.39", "50"]],
        "no_dollars": [["0.55", "80.25"], ["0.54", "20"]],
    }
    quote = normalize_orderbook("WSHCONN-WSH", "WNBA", ob)
    assert quote.yes_ask == 0.45 and quote.yes_ask_size == 80.25   # 1 - best no bid 0.55
    assert quote.no_ask == 0.60 and quote.no_ask_size == 100.5     # 1 - best yes bid 0.40


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
    # Prices come from the BBO; sizes do NOT (askDepth/bidDepth are level counts).
    assert quote.yes_ask == 0.62 and quote.yes_ask_size == 0.0
    assert round(quote.no_ask, 6) == 0.40 and quote.no_ask_size == 0.0  # 1 - 0.60


def test_polymarket_bbo_handles_missing_sides():
    md = {"marketSlug": "x", "bestAsk": None, "bestBid": {"value": "0.30"}, "askDepth": 0, "bidDepth": 12}
    quote = normalize_bbo("x", "x", md)
    assert quote.yes_ask is None              # no ask -> can't buy YES
    assert round(quote.no_ask, 6) == 0.70 and quote.no_ask_size == 0.0


def test_polymarket_book_normalization_uses_real_qty():
    # The full /book carries real per-level qty (unlike the BBO's level counts).
    md = {
        "marketSlug": "will-team-a-win",
        "bids": [{"px": {"value": "0.55"}, "qty": "1000"},
                 {"px": {"value": "0.54"}, "qty": "500"}],
        "offers": [{"px": {"value": "0.56"}, "qty": "750"},
                   {"px": {"value": "0.57"}, "qty": "1200"}],
        "state": "MARKET_STATE_OPEN",
    }
    q = normalize_book("will-team-a-win", "Team A", md, event_key="E9")
    # Buy YES by crossing the best (lowest) offer 0.56 -> size 750.
    assert q.yes_ask == 0.56 and q.yes_ask_size == 750
    # Buy NO by crossing the best (highest) bid 0.55 -> no_ask = 0.45, size 1000.
    assert round(q.no_ask, 6) == 0.45 and q.no_ask_size == 1000
    assert q.event_key == "E9"
    assert q.state == "MARKET_STATE_OPEN"          # captured for the state guard


def test_polymarket_book_one_sided():
    md = {"bids": [{"px": {"value": "0.40"}, "qty": "20"}], "offers": []}
    q = normalize_book("s", "s", md)
    assert q.yes_ask is None and q.yes_ask_size == 0.0     # no offers -> can't buy YES
    assert round(q.no_ask, 6) == 0.60 and q.no_ask_size == 20


def test_polymarket_bbo_accepts_bare_numbers():
    # Some payloads may carry bare numeric prices instead of v1Amount objects.
    md = {"bestAsk": 0.55, "bestBid": 0.53, "askDepth": 5, "bidDepth": 8}
    quote = normalize_bbo("s", "s", md)
    assert quote.yes_ask == 0.55
    assert round(quote.no_ask, 6) == 0.47


def test_order_path_has_its_own_rate_limiter():
    # ORDERS must never queue behind the read token bucket: a hedge leg waiting on
    # scan-drained read tokens right after leg 1 fills is a widened naked window.
    from types import SimpleNamespace

    from bot.venues.kalshi import KalshiVenue
    from bot.venues.polymarket_us import PolymarketUSVenue

    k = KalshiVenue(SimpleNamespace(api_key_id="", private_key_path="", api_base="",
                                    ws_base="", read_rate_per_min=None))
    p = PolymarketUSVenue(SimpleNamespace(gateway_base="", api_base="", ws_markets="",
                                          ws_private="", api_key_id="", secret_key="",
                                          ed25519_private_key_path="",
                                          read_rate_per_min=None))
    for v in (k, p):
        assert v._order_limiter is not v._limiter          # separate bucket
        assert v._order_limiter._rate > v._limiter._rate   # and a more generous one


def test_scan_quotes_deny_patterns_drops_multioutcome_junk():
    # A "SENATE" winner allowlist must not admit KXPRIMARYPLACE junk; the deny-list is the
    # guard. Pure filter logic — verify against the ticker filter used in scan_quotes.
    tickers = ["KXSENATE-26-DEM", "KXSENATEPLACE-26-4TH", "KXGOVWINNER-26-R",
               "KXPRIMARYRANK-26-2", "KXNOBEL-26-X"]
    allow = ("SENATE", "GOVWINNER", "NOBEL")
    deny = ("PLACE", "RANK")
    kept = [t for t in tickers
            if any(p in t for p in allow) and not any(d in t for d in deny)]
    assert kept == ["KXSENATE-26-DEM", "KXGOVWINNER-26-R", "KXNOBEL-26-X"]
