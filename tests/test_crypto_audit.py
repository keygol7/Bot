from bot.crypto_audit import (
    _asset, _margin_reference_spot, _poly_result, current_pair_ids,
    historical_pair_ids, source_adjustment,
)


def test_source_adjustment_is_directional_and_penalizes_uncertainty():
    mean, lower = source_adjustment(7, 8, 168)
    assert -0.006 < mean < -0.005
    assert lower < mean
    reverse_mean, _ = source_adjustment(8, 7, 168)
    assert round(reverse_mean, 8) == round(-mean, 8)


def test_source_adjustment_tiny_perfect_sample_is_not_zero_risk():
    mean, lower = source_adjustment(0, 0, 8)
    assert mean == 0
    assert lower < -0.01


def test_crypto_audit_parses_supported_assets_and_poly_resolution():
    assert _asset("KXHYPE15M-26JUL131700-00") == "HYPE"
    assert _asset("KXNEAR15M-26JUL131700-00") is None
    assert _poly_result({"closed": True, "outcomePrices": '["1", "0"]'}) == "yes"
    assert _poly_result({"closed": True, "outcomePrices": '["0", "1"]'}) == "no"
    assert _poly_result({"closed": True, "outcomePrices": '["0.5", "0.5"]'}) is None
    assert _poly_result({"closed": False, "outcomePrices": '["0.005", "0.995"]'}) is None


def test_current_pair_ids_do_not_wait_for_market_scan():
    pairs = current_pair_ids(1783976401)  # 2026-07-13 21:00 UTC / 17:00 ET start
    assert len(pairs) == 7
    assert pairs[0][1] == "KXBTC15M-26JUL131715-15"
    assert pairs[0][3] == "btc-updown-15m-1783976400"
    history = historical_pair_ids(1783976401, 1.0)
    assert len(history) == 28
    assert history[0][3] == "btc-updown-15m-1783975500"


def test_margin_reference_price_is_normalized_by_contract_size():
    assert round(_margin_reference_spot({
        "reference_price": {"price": "6.2086"}, "contract_size": "0.0001",
    })) == 62086
    assert _margin_reference_spot({}) is None
