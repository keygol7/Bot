"""Risk gating: position/exposure caps, a daily loss limit, and a kill switch.

Every prospective trade must pass :meth:`RiskManager.check` before the executor is
allowed to act. The kill switch is sticky — once tripped (manually or by breaching
the daily loss limit) the manager rejects all further opens until explicitly reset.

This is deliberately standard-library-only and side-effect-free apart from its own
in-memory state, so it is fully unit-testable without network or credentials.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskLimits:
    max_position_per_market: float = 200.0   # $ notional per market
    max_total_exposure: float = 5000.0       # $ notional across all markets
    max_daily_loss: float = 500.0            # $ realized loss before kill switch
    min_edge: float = 0.01                   # $ per-contract edge to act on an arb
    max_order_contracts: float = 2.0         # LIVE_SMALL per-order size cap (contracts)
    max_position_fraction: float = 1.0       # per-market cap as a fraction of total (concentration)

    def __post_init__(self) -> None:
        for name in (
            "max_position_per_market",
            "max_total_exposure",
            "max_daily_loss",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.min_edge < 0:
            raise ValueError("min_edge must be >= 0")


@dataclass
class RiskDecision:
    allowed: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.allowed


class RiskManager:
    def __init__(self, limits: RiskLimits | None = None) -> None:
        self.limits = limits or RiskLimits()
        self._positions: dict[str, float] = {}  # market label -> $ notional
        self._daily_pnl: float = 0.0
        self._killed: bool = False
        self._kill_reason: str = ""
        self._pnl_day: str = ""                 # UTC date the daily counter belongs to

    # ---- state inspection ----
    @property
    def total_exposure(self) -> float:
        return sum(self._positions.values())

    @property
    def daily_pnl(self) -> float:
        return self._daily_pnl

    @property
    def is_killed(self) -> bool:
        return self._killed

    @property
    def kill_reason(self) -> str:
        return self._kill_reason

    def position(self, market_label: str) -> float:
        return self._positions.get(market_label, 0.0)

    # ---- gating ----
    def check(self, market_label: str, notional: float) -> RiskDecision:
        """Can we add ``notional`` dollars of exposure to ``market_label``?"""
        if self._killed:
            return RiskDecision(False, f"kill switch active: {self._kill_reason}")
        if notional <= 0:
            return RiskDecision(False, "notional must be positive")

        new_market = self.position(market_label) + notional
        if new_market > self.limits.max_position_per_market:
            return RiskDecision(
                False,
                f"per-market cap exceeded: ${new_market:.2f} > "
                f"${self.limits.max_position_per_market:.2f}",
            )

        new_total = self.total_exposure + notional
        if new_total > self.limits.max_total_exposure:
            return RiskDecision(
                False,
                f"total exposure cap exceeded: ${new_total:.2f} > "
                f"${self.limits.max_total_exposure:.2f}",
            )

        return RiskDecision(True)

    # ---- state mutation ----
    def record_fill(self, market_label: str, notional: float) -> None:
        """Record committed capital after a fill. Negative ``notional`` releases it."""
        self._positions[market_label] = self.position(market_label) + notional
        if self._positions[market_label] <= 1e-9:
            self._positions.pop(market_label, None)

    def retain_markets(self, open_labels) -> float:
        """Release exposure for markets no longer OPEN on the venue.

        ``record_fill`` only ever ADDS notional, so settled/expired markets would pin
        their exposure forever — tightening the per-market/total caps monotonically
        until a restart. The balance poll calls this with the labels the venues still
        report as open positions; anything else has settled and its capital is back
        (or lost — either way no longer *at risk*). Returns the $ released.
        """
        keep = set(open_labels)
        stale = [label for label in self._positions if label not in keep]
        released = 0.0
        for label in stale:
            released += self._positions.pop(label)
        return released

    def record_pnl(self, amount: float) -> None:
        """Record realized PnL. A breach of the daily loss limit trips the kill switch.

        The counter self-rolls at UTC midnight so "daily" is a real day, not
        process-lifetime (nothing else in the live loop calls reset_daily). The kill
        switch stays sticky — a new day resets the COUNTER, never a tripped switch.
        """
        from datetime import datetime, timezone

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._pnl_day and self._pnl_day != today:
            self._daily_pnl = 0.0
        self._pnl_day = today
        self._daily_pnl += amount
        if self._daily_pnl <= -self.limits.max_daily_loss:
            self.trip_kill_switch(
                f"daily loss limit hit: ${self._daily_pnl:.2f} <= "
                f"-${self.limits.max_daily_loss:.2f}"
            )

    def trip_kill_switch(self, reason: str = "manual") -> None:
        self._killed = True
        self._kill_reason = reason

    def reset_kill_switch(self) -> None:
        self._killed = False
        self._kill_reason = ""

    def reset_daily(self) -> None:
        """Reset the daily PnL counter (call at the start of each trading day)."""
        self._daily_pnl = 0.0
