"""Startup balance + position reconciliation guard.

Before the streaming engine is allowed to place a single order, every trading
venue must be verified to be in a known, safe state:

  * funded at or above a configured minimum balance, and
  * flat — NO held positions and NO resting orders.

The bot is built to start flat; a leftover leg from a prior crash, an aborted
two-leg execution, or a manual trade is exactly the dangerous state that turns a
market-neutral arb into naked directional risk. This guard catches it on boot.

Fail-closed is the rule: any venue that cannot be reconciled — an endpoint error,
an unreadable balance, an unexpected position — trips the kill switch so the
executor rejects every trade until a human investigates. Trading on an
unverified account is never allowed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from bot.execution.account import AccountSnapshot

log = logging.getLogger("bot.startup_guard")


@dataclass
class GuardResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    snapshots: list[AccountSnapshot] = field(default_factory=list)


def _is_trading(venue) -> bool:
    return bool(getattr(venue, "is_trading_configured", False)
                or getattr(venue, "authenticated", False))


async def _snapshot(venue) -> AccountSnapshot | None:
    """Read a venue's account state, or ``None`` if it can't be read (fail closed)."""
    fn = getattr(venue, "account_snapshot", None)
    if fn is None:
        log.error("%s exposes no account_snapshot() — cannot reconcile", venue.name)
        return None
    try:
        return await fn()
    except Exception as exc:
        log.error("%s account_snapshot failed: %s", venue.name, exc)
        return None


async def reconcile_startup(
    venues,
    risk,
    *,
    min_balance: float = 0.0,
    allow_existing_positions: bool = False,
) -> GuardResult:
    """Verify every trading venue is funded and flat; trip the kill switch if not.

    Returns a :class:`GuardResult`. When ``ok`` is False the kill switch is already
    tripped (the executor will refuse to trade) and ``reasons`` explains why.
    """
    reasons: list[str] = []
    snapshots: list[AccountSnapshot] = []

    trading = [v for v in venues if _is_trading(v)]
    if not trading:
        reasons.append("no trading-configured venues to reconcile")

    for v in trading:
        snap = await _snapshot(v)
        if snap is None:
            reasons.append(f"{v.name}: account state unreadable — failing closed")
            continue
        snapshots.append(snap)

        if snap.balance is None:
            reasons.append(f"{v.name}: balance unknown — failing closed")
            log.warning("startup: %s balance UNKNOWN", v.name)
        else:
            log.info("startup: %s balance $%.2f", v.name, snap.balance)
            if snap.balance < min_balance:
                reasons.append(
                    f"{v.name}: balance ${snap.balance:.2f} < required ${min_balance:.2f}"
                )

        if not allow_existing_positions:
            open_pos = snap.open_positions
            if open_pos:
                detail = ", ".join(
                    f"{p.market_id}(qty={p.quantity:g},resting={p.resting_orders})"
                    for p in open_pos
                )
                reasons.append(
                    f"{v.name}: {len(open_pos)} pre-existing position(s)/order(s) — {detail}"
                )

    result = GuardResult(ok=not reasons, reasons=reasons, snapshots=snapshots)
    if not result.ok:
        risk.trip_kill_switch("startup reconciliation failed: " + "; ".join(reasons))
        if all("unreadable" in r for r in reasons):
            # venue outage/maintenance — the caller retries until it answers;
            # CRITICAL once a minute for a planned window just pages the operator
            log.warning("startup guard: venue(s) unreadable — %s", "; ".join(reasons))
        else:
            log.critical("STARTUP GUARD FAILED — trading disabled. %s",
                         "; ".join(reasons))
    else:
        log.info("startup guard passed: %d venue(s) funded and flat", len(snapshots))
    return result
