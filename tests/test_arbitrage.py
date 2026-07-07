from bot.fees import KalshiFeeModel, ZeroFeeModel
from bot.models import MarketQuote
from bot.strategies.arbitrage import (
    bundle_price_edge,
    cross_price_edge,
    detect_bundle,
    detect_cross_venue,
)


def q(venue, mid, yes_ask=None, ya=0.0, no_ask=None, na=0.0, event="E1"):
    return MarketQuote(
        venue=venue, market_id=mid, title=mid, event_key=event,
        yes_ask=yes_ask, yes_ask_size=ya, no_ask=no_ask, no_ask_size=na,
    )


def test_cross_venue_positive_edge_zero_fees():
    a = q("kalshi", "A", yes_ask=0.40, ya=100, no_ask=0.65, na=100)
    b = q("polymarket_us", "B", yes_ask=0.62, ya=100, no_ask=0.55, na=60)
    opps = detect_cross_venue(a, b)  # zero fees by default
    assert len(opps) == 1
    o = opps[0]
    # YES on A (0.40) + NO on B (0.55) = 0.95 -> 0.05 edge
    assert o.buy_yes_venue == "kalshi" and o.buy_no_venue == "polymarket_us"
    assert round(o.gross_cost, 4) == 0.95
    assert round(o.edge_per_contract, 4) == 0.05
    assert o.max_contracts == 60  # min(100, 60)
    assert round(o.total_profit, 4) == round(60 * 0.05, 4)
    assert round(o.notional, 4) == round(0.95 * 60, 4)


def test_cross_venue_other_direction_wins():
    # Cheap YES on B, cheap NO on A -> profitable direction is YES@B + NO@A
    a = q("kalshi", "A", yes_ask=0.70, ya=100, no_ask=0.42, na=100)
    b = q("polymarket_us", "B", yes_ask=0.50, ya=80, no_ask=0.70, na=100)
    opps = detect_cross_venue(a, b)
    assert len(opps) == 1
    o = opps[0]
    assert o.buy_yes_venue == "polymarket_us" and o.buy_no_venue == "kalshi"
    assert round(o.gross_cost, 4) == 0.92
    assert o.max_contracts == 80


def test_no_arb_when_sum_exceeds_one():
    a = q("kalshi", "A", yes_ask=0.55, ya=100, no_ask=0.55, na=100)
    b = q("polymarket_us", "B", yes_ask=0.55, ya=100, no_ask=0.55, na=100)
    assert detect_cross_venue(a, b) == []


def test_min_edge_filters_marginal_opportunities():
    a = q("kalshi", "A", yes_ask=0.49, ya=100, no_ask=0.60, na=100)
    b = q("polymarket_us", "B", yes_ask=0.60, ya=100, no_ask=0.50, na=100)
    # YES@A + NO@B = 0.99 -> 0.01 edge
    assert detect_cross_venue(a, b, min_edge=0.0)
    assert detect_cross_venue(a, b, min_edge=0.02) == []


def test_missing_side_yields_no_opportunity():
    a = q("kalshi", "A", yes_ask=0.40, ya=100)  # no NO side, no... still has yes
    b = q("polymarket_us", "B", no_ask=None, na=0)  # nothing takeable
    assert detect_cross_venue(a, b) == []


def test_fees_reduce_edge():
    a = q("kalshi", "A", yes_ask=0.47, ya=100, no_ask=0.60, na=100)
    b = q("polymarket_us", "B", yes_ask=0.60, ya=100, no_ask=0.50, na=100)
    # gross YES@A+NO@B = 0.97. Kalshi fee on the 0.47 leg eats into the 0.03.
    with_fees = detect_cross_venue(a, b, fee_a=KalshiFeeModel(), fee_b=ZeroFeeModel())
    without = detect_cross_venue(a, b)
    assert without[0].edge_per_contract > with_fees[0].edge_per_contract


def test_bundle_single_venue():
    qq = q("kalshi", "A", yes_ask=0.40, ya=100, no_ask=0.55, na=70)
    o = detect_bundle(qq)
    assert o is not None
    assert o.is_single_venue
    assert round(o.edge_per_contract, 4) == 0.05
    assert o.max_contracts == 70


def test_bundle_none_when_no_edge():
    qq = q("kalshi", "A", yes_ask=0.50, ya=100, no_ask=0.55, na=70)
    assert detect_bundle(qq) is None


# --- price-only edge helpers (size-independent, used by the two-phase scanner) ---

def test_bundle_price_edge_ignores_size():
    # size 0 -> detect_bundle finds nothing, but the price edge is still visible.
    qq = q("kalshi", "A", yes_ask=0.40, ya=0, no_ask=0.55, na=0)
    assert detect_bundle(qq) is None
    assert round(bundle_price_edge(qq), 4) == 0.05


def test_bundle_price_edge_none_when_side_missing():
    qq = q("kalshi", "A", yes_ask=0.40, ya=0, no_ask=None, na=0)
    assert bundle_price_edge(qq) is None


def test_cross_price_edge_best_direction():
    a = q("kalshi", "A", yes_ask=0.40, ya=0, no_ask=0.65, na=0)
    b = q("polymarket_us", "B", yes_ask=0.62, ya=0, no_ask=0.55, na=0)
    # YES@A + NO@B = 0.95 -> 0.05; the other direction is negative.
    assert round(cross_price_edge(a, b), 4) == 0.05


def test_cross_price_edge_negative_when_no_arb():
    a = q("kalshi", "A", yes_ask=0.55, ya=0, no_ask=0.55, na=0)
    b = q("polymarket_us", "B", yes_ask=0.55, ya=0, no_ask=0.55, na=0)
    assert cross_price_edge(a, b) < 0


def test_cross_price_edge_subtracts_fees():
    a = q("kalshi", "A", yes_ask=0.47, ya=0, no_ask=0.60, na=0)
    b = q("polymarket_us", "B", yes_ask=0.60, ya=0, no_ask=0.50, na=0)
    with_fee = cross_price_edge(a, b, fee_a=KalshiFeeModel(), fee_b=ZeroFeeModel())
    without = cross_price_edge(a, b)
    assert with_fee < without


def test_settle_ts_from_id():
    from bot.strategies.arbitrage import settle_ts_from_id
    from datetime import datetime, timezone
    # poly ISO date slug
    ts = settle_ts_from_id("ewc-usgub-ca-2026-11-03-stehil")
    assert datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d") == "2026-11-03"
    # kalshi YYMONDD ticker
    ts = settle_ts_from_id("KXWCGOAL-26JUL04PARFRA-FRAKMBAPP10-2")
    assert datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d") == "2026-07-04"
    # year-only kalshi election ticker -> no parseable date
    assert settle_ts_from_id("KXGOVCA-26-SHIL") == 0.0
    assert settle_ts_from_id("") == 0.0


def test_build_settle_ts_falls_back_to_id_date():
    # A long-dated election pair with NO close_time on either quote must still get a
    # settle_ts from the id (the horizon gate was blind to these before).
    from bot.strategies.arbitrage import detect_cross_venue
    from datetime import datetime, timezone
    a = q("kalshi", "KXGOVCA-26-SHIL", yes_ask=0.08, ya=100, no_ask=0.93, na=100)
    b = q("polymarket_us", "ewc-usgub-ca-2026-11-03-stehil",
          yes_ask=0.90, ya=100, no_ask=0.10, na=100)
    opps = detect_cross_venue(a, b)
    assert opps
    got = datetime.fromtimestamp(opps[0].settle_ts, timezone.utc).strftime("%Y-%m-%d")
    assert got == "2026-11-03"                          # from the poly leg's slug


def test_sweep_levels_takes_deeper_profitable_levels():
    from bot.fees import ZeroFeeModel
    from bot.strategies.arbitrage import sweep_levels
    z = ZeroFeeModel()
    # top: 5ct at .44; behind: 200ct at .46 — poly NO flat 300ct at .50
    yes = ((0.44, 5), (0.46, 200))
    no = ((0.50, 300),)
    py, pn, size = sweep_levels(yes, no, z, z, min_edge=0.0)
    assert (py, pn) == (0.46, 0.50)         # limit at the deeper level
    assert size == 205                      # cumulative across both levels
    # if level 2 is unprofitable (.51 + .50 > 1), stay at the top
    py, pn, size = sweep_levels(((0.44, 5), (0.51, 200)), no, z, z, min_edge=0.0)
    assert (py, pn, size) == (0.44, 0.50, 5)
    # min_edge gates how deep the sweep goes
    py, pn, size = sweep_levels(yes, no, z, z, min_edge=0.05)
    assert (py, pn, size) == (0.44, 0.50, 5)


def test_detect_cross_venue_sweeps_ladders():
    from bot.fees import ZeroFeeModel
    from bot.strategies.arbitrage import detect_cross_venue
    a = MarketQuote(venue="kalshi", market_id="K", title="k",
                    yes_ask=0.44, yes_ask_size=5,
                    yes_ask_levels=((0.44, 5), (0.46, 200)))
    b = MarketQuote(venue="poly", market_id="P", title="p",
                    no_ask=0.50, no_ask_size=300,
                    no_ask_levels=((0.50, 300),))
    opps = detect_cross_venue(a, b, fee_a=ZeroFeeModel(), fee_b=ZeroFeeModel())
    assert opps and opps[0].yes_price == 0.46 and opps[0].max_contracts == 205
    # profit maximized: 205 * .04 = 8.20 > top-only 5 * .06 = 0.30
    assert abs(opps[0].total_profit - 8.20) < 1e-9
