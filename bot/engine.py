"""The coordinating engine: matched quotes -> arb detection -> risk gating -> log.

This is the deterministic core of the DRY_RUN loop, written so it is unit-testable
without any network: feed it confirmed cross-venue pairs and it detects arbitrage,
applies the risk limits, persists every opportunity, and — only when the run mode
permits and risk allows — hands off to the executor. In DRY_RUN it records but never
trades.

The live data path (WebSocket ingest -> in-memory books -> these quotes) and the
latency-optimized executor attach around this core in later phases; keeping the
decision logic isolated here is what lets us prove it before risking capital.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from bot.data.store import Store
from bot.execution.risk import RiskManager
from bot.fees import FeeModel, ZeroFeeModel
from bot.modes import RunMode
from bot.models import MarketQuote
from bot.strategies.arbitrage import ArbOpportunity, detect_cross_venue

log = logging.getLogger("bot.engine")


@dataclass
class ScanResult:
    detected: list[ArbOpportunity] = field(default_factory=list)
    actionable: list[ArbOpportunity] = field(default_factory=list)
    skipped: list[tuple[ArbOpportunity, str]] = field(default_factory=list)

    @property
    def best_profit(self) -> float:
        return max((o.total_profit for o in self.actionable), default=0.0)


class Engine:
    def __init__(
        self,
        *,
        fee_models: dict[str, FeeModel],
        risk: RiskManager,
        store: Store | None = None,
        mode: RunMode = RunMode.DRY_RUN,
        min_edge: float = 0.01,
    ) -> None:
        self.fee_models = fee_models
        self.risk = risk
        self.store = store
        self.mode = mode
        self.min_edge = min_edge

    def _fee(self, venue: str) -> FeeModel:
        return self.fee_models.get(venue, ZeroFeeModel())

    def evaluate_pairs(
        self, pairs: list[tuple[MarketQuote, MarketQuote]]
    ) -> ScanResult:
        """Detect, gate, and record arbitrage across confirmed cross-venue pairs.

        Each pair MUST already be confirmed by the matcher as the same event with
        the same resolution criteria — the engine trusts that contract and only
        does price/risk work here.
        """
        result = ScanResult()

        for a, b in pairs:
            opps = detect_cross_venue(
                a, b,
                fee_a=self._fee(a.venue),
                fee_b=self._fee(b.venue),
                min_edge=self.min_edge,
            )
            for opp in opps:
                result.detected.append(opp)

                decision = self.risk.check(
                    f"{opp.buy_yes_venue}:{opp.buy_yes_market}", opp.notional
                )
                acted = False
                if not decision.allowed:
                    result.skipped.append((opp, decision.reason))
                elif self.mode.places_real_orders:
                    # Live execution attaches in a later phase. Until the executor
                    # exists, surface that we *would* act but cannot yet.
                    result.skipped.append((opp, "executor not enabled in this phase"))
                else:
                    # DRY_RUN: viable opportunity, recorded but not traded.
                    result.actionable.append(opp)

                if self.store is not None:
                    self.store.record_opportunity(opp, acted=acted)
                log.info("%s | %s", "ACT" if opp in result.actionable else "skip", opp)

        return result
