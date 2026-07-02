"""Per-venue trading fee models.

Fees are decisive for prediction-market arbitrage: an apparent edge that ignores
fees is usually not an edge at all. Each venue adapter supplies the right model so
the arbitrage detector can subtract real costs before signalling.

- Polymarket US standard markets: ~zero trading fee  -> ``ZeroFeeModel``.
- Kalshi: a price-dependent fee, ``ceil(rate * C * P * (1-P))`` rounded up to the
  next cent (rate ~0.07 on most markets) -> ``KalshiFeeModel``.

If/when published fee schedules change, update the model here; nothing else needs
to change.
"""

from __future__ import annotations

import math
from decimal import ROUND_HALF_EVEN, Decimal
from typing import Protocol


class FeeModel(Protocol):
    def fee(self, price: float, contracts: float) -> float:
        """Total fee in dollars to transact ``contracts`` at ``price`` (per fill)."""
        ...


def per_contract_fee(model, price: float) -> float:
    """UNROUNDED per-contract fee for EDGE GATING.

    ``model.fee(price, 1)`` quantizes to a whole cent at 1 contract (Kalshi ceils
    0.0175 -> $0.02; Polymarket banker's-rounds 0.0125 -> $0.01 and 0.0024 -> $0.00),
    an error the same magnitude as a half-cent edge floor — so detectors gating on it
    mis-rank marginal arbs in both directions. Gate on the smooth rate instead; the
    venue's rounded ``fee()`` still applies to settlement accounting at real size.
    Models may expose ``per_contract(price)``; anything else falls back to fee(p, 1).
    """
    fn = getattr(model, "per_contract", None)
    if fn is not None:
        return fn(price)
    return model.fee(price, 1)


class ZeroFeeModel:
    """No trading fee (Polymarket US standard markets)."""

    def fee(self, price: float, contracts: float) -> float:
        return 0.0

    def per_contract(self, price: float) -> float:
        return 0.0

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "ZeroFeeModel()"


class KalshiFeeModel:
    """Kalshi maker/taker fee: ``ceil(rate * C * P * (1-P))`` rounded up to a cent.

    The fee is largest near P=0.50 and vanishes near the extremes, which is why it
    has to be modelled per price rather than as a flat rate.
    """

    def __init__(self, rate: float = 0.07) -> None:
        if rate < 0:
            raise ValueError("fee rate must be >= 0")
        self.rate = rate

    def fee(self, price: float, contracts: float) -> float:
        if not (0.0 <= price <= 1.0):
            raise ValueError(f"price must be in [0, 1], got {price}")
        if contracts < 0:
            raise ValueError("contracts must be >= 0")
        raw = self.rate * contracts * price * (1.0 - price)
        # Kalshi rounds fees up to the next whole cent.
        return math.ceil(round(raw * 100, 9)) / 100.0

    def per_contract(self, price: float) -> float:
        """Unrounded per-contract rate for edge gating (see per_contract_fee)."""
        return self.rate * price * (1.0 - price)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"KalshiFeeModel(rate={self.rate})"


class PolymarketUSFeeModel:
    """Polymarket US (QCEX) taker fee: ``rate * C * p * (1-p)`` rounded to the NEAREST
    cent with banker's rounding (round half to even), per the published schedule
    (effective 2026-04-03). Same price-dependent shape as Kalshi's but rate 0.05 and
    nearest-cent rounding (Kalshi rounds UP). The bot always TAKES the Polymarket leg (it
    rests only Kalshi makers), so the taker rate applies; the maker rebate (-0.0125) is not
    captured. Fees near p=0 / p=1 round to $0.

    This is decisive for the edge gate: at mid-prices the taker fee is ~1.25c/contract,
    ABOVE a 1c min-edge — modelling it (vs the old ZeroFeeModel) stops the bot firing arbs
    that are net-negative after the real fee.
    """

    def __init__(self, rate: float = 0.05) -> None:
        if rate < 0:
            raise ValueError("fee rate must be >= 0")
        self.rate = rate

    def fee(self, price: float, contracts: float) -> float:
        if not (0.0 <= price <= 1.0):
            raise ValueError(f"price must be in [0, 1], got {price}")
        if contracts < 0:
            raise ValueError("contracts must be >= 0")
        raw = self.rate * contracts * price * (1.0 - price)
        # Polymarket rounds to the NEAREST cent, half-to-even (banker's rounding).
        return float(Decimal(str(raw)).quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN))

    def per_contract(self, price: float) -> float:
        """Unrounded per-contract rate for edge gating (see per_contract_fee)."""
        return self.rate * price * (1.0 - price)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"PolymarketUSFeeModel(rate={self.rate})"
