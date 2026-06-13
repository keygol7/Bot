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
from typing import Protocol


class FeeModel(Protocol):
    def fee(self, price: float, contracts: float) -> float:
        """Total fee in dollars to transact ``contracts`` at ``price`` (per fill)."""
        ...


class ZeroFeeModel:
    """No trading fee (Polymarket US standard markets)."""

    def fee(self, price: float, contracts: float) -> float:  # noqa: ARG002
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

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"KalshiFeeModel(rate={self.rate})"
