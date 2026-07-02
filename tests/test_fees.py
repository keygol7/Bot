import math

import pytest

from bot.fees import KalshiFeeModel, PolymarketUSFeeModel, ZeroFeeModel


def test_polymarket_us_fee_matches_published_schedule():
    f = PolymarketUSFeeModel()                       # taker rate 0.05
    # Examples from the published Polymarket US fee schedule (eff. 2026-04-03):
    assert f.fee(0.10, 1000) == 4.50                 # 0.05*1000*0.10*0.90
    assert f.fee(0.65, 1000) == 11.38                # 0.05*1000*0.65*0.35 = 11.375 -> half-to-even
    assert f.fee(0.50, 1000) == 12.50                # max at the midpoint
    assert f.fee(0.50, 100) == 1.25                  # 100-lot table row
    assert f.fee(0.10, 100) == 0.45
    assert f.fee(0.30, 500) == f.fee(0.70, 500)      # symmetric around 0.5
    assert f.fee(0.99, 1) == 0.0                      # ~zero at the extremes / tiny trades
    # at mid-price the per-contract fee (1.25c) exceeds a 1c min-edge — the whole point
    assert f.fee(0.50, 100) / 100 > 0.01


def test_per_contract_rate_is_unrounded_for_edge_gating():
    # fee(p, 1) cent-quantizes (Kalshi ceils 0.0175->0.02, Poly rounds 0.0125->0.01,
    # 0.0024->0.00) — an error the size of a half-cent edge floor. per_contract_fee
    # must return the smooth rate so detectors gate honestly.
    from bot.fees import per_contract_fee
    k, p = KalshiFeeModel(0.07), PolymarketUSFeeModel(0.05)
    assert per_contract_fee(k, 0.50) == 0.07 * 0.25            # 0.0175, not 0.02
    assert per_contract_fee(p, 0.50) == 0.05 * 0.25            # 0.0125, not 0.01
    assert per_contract_fee(p, 0.05) == 0.05 * 0.05 * 0.95     # 0.0024, not 0.00
    assert per_contract_fee(ZeroFeeModel(), 0.5) == 0.0

    class LegacyModel:                                          # no per_contract attr
        def fee(self, price, contracts):
            return 0.42
    assert per_contract_fee(LegacyModel(), 0.5) == 0.42         # falls back to fee(p, 1)


def test_zero_fee_is_always_zero():
    f = ZeroFeeModel()
    assert f.fee(0.5, 100) == 0.0
    assert f.fee(0.01, 1) == 0.0


def test_kalshi_fee_at_mid_price():
    f = KalshiFeeModel(rate=0.07)
    # 0.07 * 1 * 0.5 * 0.5 = 0.0175 -> $1.75 cents -> rounds up to $0.02
    assert f.fee(0.5, 1) == 0.02
    # 0.07 * 100 * 0.25 = 1.75 -> exactly $1.75
    assert f.fee(0.5, 100) == 1.75


def test_kalshi_fee_smaller_near_extremes():
    f = KalshiFeeModel(rate=0.07)
    assert f.fee(0.99, 1) == 0.01   # 0.07*0.99*0.01=0.000693 -> rounds up to a cent
    assert f.fee(0.5, 1) > f.fee(0.9, 1)


def test_kalshi_fee_rounds_up_to_cent():
    f = KalshiFeeModel(rate=0.07)
    raw = 0.07 * 3 * 0.5 * 0.5  # 0.0525
    assert f.fee(0.5, 3) == math.ceil(raw * 100) / 100 == 0.06


def test_invalid_inputs():
    f = KalshiFeeModel()
    with pytest.raises(ValueError):
        f.fee(1.5, 1)
    with pytest.raises(ValueError):
        KalshiFeeModel(rate=-1)
