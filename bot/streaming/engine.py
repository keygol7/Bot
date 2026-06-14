"""Fast streaming engine: live quotes -> per-tick edge check -> execute."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from bot.execution.executor import Executor
from bot.fees import FeeModel, ZeroFeeModel
from bot.models import MarketQuote
from bot.strategies.arbitrage import ArbOpportunity

log = logging.getLogger("bot.streaming")


@dataclass(frozen=True)
class ConfirmedPair:
    """A same-event pair confirmed by the slow matcher. Direction is decided live."""

    event_key: str
    venue_a: str
    market_a: str
    venue_b: str
    market_b: str

    @property
    def key(self) -> tuple:
        return tuple(sorted([(self.venue_a, self.market_a), (self.venue_b, self.market_b)]))

    def markets(self) -> list[tuple[str, str]]:
        return [(self.venue_a, self.market_a), (self.venue_b, self.market_b)]


class LiveBook:
    """Latest top-of-book quote per (venue, market_id), fed from WS streams."""

    def __init__(self) -> None:
        self._q: dict[tuple[str, str], MarketQuote] = {}

    def update(self, q: MarketQuote) -> None:
        self._q[(q.venue, q.market_id)] = q

    def get(self, venue: str, market_id: str) -> MarketQuote | None:
        return self._q.get((venue, market_id))


class StreamingEngine:
    def __init__(
        self,
        *,
        executor: Executor,
        fee_models: dict[str, FeeModel] | None = None,
        min_edge: float = 0.01,
        cooldown: float = 5.0,
        livebook: LiveBook | None = None,
        clock=time.monotonic,
    ) -> None:
        self.executor = executor
        self.fee_models = fee_models or {}
        self.min_edge = min_edge
        self.cooldown = cooldown
        self.livebook = livebook or LiveBook()
        self.clock = clock
        self._pairs: dict[tuple, ConfirmedPair] = {}
        self._index: dict[tuple[str, str], set] = {}   # (venue,market) -> set of pair keys
        self._last_acted: dict[tuple, float] = {}

    def _fee(self, venue: str) -> FeeModel:
        return self.fee_models.get(venue, ZeroFeeModel())

    def set_pairs(self, pairs: list[ConfirmedPair]) -> None:
        self._pairs = {p.key: p for p in pairs}
        self._index = {}
        for p in pairs:
            for m in p.markets():
                self._index.setdefault(m, set()).add(p.key)

    @property
    def market_ids(self) -> dict[str, list[str]]:
        """Confirmed markets grouped by venue (what to subscribe each WS to)."""
        out: dict[str, list[str]] = {}
        for (venue, market) in self._index:
            out.setdefault(venue, []).append(market)
        return out

    def _best_direction(self, p: ConfirmedPair):
        """Best (edge, yes_quote, no_quote, size) over both arb directions, or None."""
        a = self.livebook.get(p.venue_a, p.market_a)
        b = self.livebook.get(p.venue_b, p.market_b)
        if a is None or b is None:
            return None
        best = None
        for yq, nq in ((a, b), (b, a)):  # buy YES@yq + NO@nq
            if yq.yes_ask is None or nq.no_ask is None:
                continue
            fee = self._fee(yq.venue).fee(yq.yes_ask, 1) + self._fee(nq.venue).fee(nq.no_ask, 1)
            edge = 1.0 - (yq.yes_ask + nq.no_ask) - fee
            size = min(yq.yes_ask_size, nq.no_ask_size)
            if best is None or edge > best[0]:
                best = (edge, yq, nq, size)
        return best

    def _build_opp(self, p, edge, yq, nq, size) -> ArbOpportunity:
        gross = yq.yes_ask + nq.no_ask
        return ArbOpportunity(
            event_key=p.event_key,
            buy_yes_venue=yq.venue, buy_yes_market=yq.market_id,
            buy_no_venue=nq.venue, buy_no_market=nq.market_id,
            yes_price=yq.yes_ask, no_price=nq.no_ask, gross_cost=gross,
            fee_per_pair=max(0.0, 1.0 - gross - edge), edge_per_contract=edge,
            max_contracts=size, total_fees=0.0, total_profit=edge * size, notional=gross * size,
        )

    async def on_quote(self, q: MarketQuote):
        """Process one live quote: update the book, then act on any pair it touches.

        Returns the last ExecutionReport produced (or None). The cooldown prevents
        re-firing the same pair on every tick.
        """
        self.livebook.update(q)
        report = None
        for key in self._index.get((q.venue, q.market_id), ()):
            p = self._pairs[key]
            ev = self._best_direction(p)
            if ev is None:
                continue
            edge, yq, nq, size = ev
            if edge <= self.min_edge or size < 1:
                continue
            if self.clock() - self._last_acted.get(key, -1e9) < self.cooldown:
                continue
            self._last_acted[key] = self.clock()
            opp = self._build_opp(p, edge, yq, nq, size)
            log.info("STREAM edge %.4f on %s -> executing", edge, p.event_key)
            report = await self.executor.execute(opp)
            log.info("STREAM exec %s | %s", p.event_key, report)
        return report

    async def _consume(self, venue) -> None:
        mids = self.market_ids.get(venue.name) or None
        if mids is None:
            return  # nothing confirmed for this venue this cycle
        async for q in venue.stream_order_book(mids):
            await self.on_quote(q)

    async def run(self, venues: list, refresh_specs, *, refresh_interval: float = 300.0) -> None:
        """Slow/fast loop: refresh confirmed pairs each interval, (re)subscribe the
        fast consumers to the current market set, and stream until the next refresh.

        ``refresh_specs`` is an async callable returning ``list[ConfirmedPair]``.
        """
        while True:
            try:
                pairs = await refresh_specs()
            except Exception as exc:
                log.warning("spec refresh failed (%s); keeping %d existing pairs",
                            exc, len(self._pairs))
                pairs = None
            if pairs:
                self.set_pairs(pairs)
            elif pairs is not None and self._pairs:
                # Successful refresh but empty (e.g. transient: no edge/markets this
                # cycle) — keep the last-good watchlist rather than going dark.
                log.info("refresh returned 0 pairs; keeping %d existing", len(self._pairs))
            log.info("streaming %d confirmed pairs across %d venues",
                     len(self._pairs), len(self.market_ids))
            consumers = [asyncio.create_task(self._consume(v)) for v in venues]
            try:
                await asyncio.sleep(refresh_interval)
            finally:
                for c in consumers:
                    c.cancel()
                await asyncio.gather(*consumers, return_exceptions=True)
