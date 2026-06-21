"""Run modes for the engine.

The mode gates whether the executor is allowed to place real orders. DRY_RUN is
the default and the only mode that requires no credentials; it detects and logs
opportunities but never sends an order.
"""

from __future__ import annotations

from enum import Enum


class RunMode(str, Enum):
    DRY_RUN = "DRY_RUN"        # monitor + simulate, no real orders
    LIVE_SMALL = "LIVE_SMALL"  # real orders behind tight caps + kill switch
    LIVE = "LIVE"              # real orders at full configured limits

    @property
    def places_real_orders(self) -> bool:
        return self in (RunMode.LIVE_SMALL, RunMode.LIVE)

    @classmethod
    def parse(cls, value: str | None) -> "RunMode":
        if not value:
            return cls.DRY_RUN
        try:
            return cls(value.strip().upper())
        except ValueError:
            raise ValueError(
                f"Unknown run mode {value!r}; expected one of "
                f"{', '.join(m.value for m in cls)}"
            ) from None
