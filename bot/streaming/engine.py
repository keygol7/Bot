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


def _fmt_q(q) -> str:
    """Compact yes_ask/no_ask + sizes for a quote, for diagnostic logs."""
    if q is None:
        return "None"
    return (f"yes={q.yes_ask}@{q.yes_ask_size:g} no={q.no_ask}@{q.no_ask_size:g}")


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
    # Only a market in this state is tradeable; anything else (suspended, halted,
    # pre-open, closing auction, expired, terminated, or a Kalshi non-active lifecycle)
    # would reject or settle against us, so we don't fire on it.
    _OPEN_STATE = "MARKET_STATE_OPEN"

    def __init__(
        self,
        *,
        executor: Executor,
        fee_models: dict[str, FeeModel] | None = None,
        min_edge: float = 0.01,
        cooldown: float = 5.0,
        livebook: LiveBook | None = None,
        clock=time.monotonic,
        depth_fetch=None,
        max_ws_quote_age: float = 2.0,
    ) -> None:
        self.executor = executor
        self.fee_models = fee_models or {}
        self.min_edge = min_edge
        self.cooldown = cooldown
        self.livebook = livebook or LiveBook()
        self.clock = clock
        # If both crossing legs have a WS quote with real size no older than this many
        # seconds, fire off the live book and skip the per-fire REST depth re-fetch.
        # 0 disables (always re-fetch). Quote timestamps are wall-clock (time.time()),
        # so freshness is checked against time.time(), not the monotonic ``clock``.
        self.max_ws_quote_age = max_ws_quote_age
        # Async callable depth_fetch(venue, market_id) -> sized MarketQuote | None.
        # WS ticker feeds carry no size (Kalshi), so before firing on a price edge we
        # re-fetch real order-book depth (which also re-validates the price).
        self.depth_fetch = depth_fetch
        self._pairs: dict[tuple, ConfirmedPair] = {}
        self._index: dict[tuple[str, str], set] = {}   # (venue,market) -> set of pair keys
        self._last_acted: dict[tuple, float] = {}
        self._ws_counts: dict[str, int] = {}   # venue -> quotes seen since last refresh
        self._inflight: set = set()             # in-flight execute() tasks (cancel-shielded)
        # Escalating backoff for pairs whose orders keep failing (e.g. a venue that
        # rejects on a live/in-play market) so the engine stops hammering them.
        self._fail_counts: dict[tuple, int] = {}
        self._backoff_until: dict[tuple, float] = {}
        self._backoff_base = 60.0               # first penalty after a failed attempt
        self._backoff_cap = 1800.0              # max 30 min between retries
        # Latest known market state per (venue, market_id): from Polymarket's marketData
        # `state` (carried on the quote) and Kalshi's lifecycle channel. Used to skip
        # firing into a non-OPEN (halted/suspended/pre-open/settled) market.
        self._market_state: dict[tuple[str, str], str] = {}

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

    def _eval_direction(self, a: MarketQuote | None, b: MarketQuote | None):
        """Best (edge, yes_quote, no_quote, size) over both arb directions, or None."""
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

    def _best_direction(self, p: ConfirmedPair):
        """Best arb direction from the live WS book (prices; sizes may be 0)."""
        return self._eval_direction(
            self.livebook.get(p.venue_a, p.market_a),
            self.livebook.get(p.venue_b, p.market_b),
        )

    def watches(self, venue: str, market_id: str) -> bool:
        """True if (venue, market_id) is a leg of a current confirmed pair — used to
        filter the unfiltered Kalshi lifecycle firehose down to the watchlist."""
        return (venue, market_id) in self._index

    def set_market_state(self, venue: str, market_id: str, state: str | None) -> None:
        """Record a market's latest state (from a Polymarket quote or the Kalshi
        lifecycle channel). ``None`` clears it back to unknown."""
        if state:
            self._market_state[(venue, market_id)] = state
        else:
            self._market_state.pop((venue, market_id), None)

    def _quote_open(self, q) -> bool:
        """Whether this leg's market is tradeable: OPEN, or unknown (no state seen yet
        -> don't block, so a venue we lack state for still trades). A KNOWN non-OPEN
        state blocks the fire. Prefers the quote's own state, falling back to the
        last-seen state map (which the Kalshi lifecycle channel feeds)."""
        state = getattr(q, "state", None) or self._market_state.get((q.venue, q.market_id))
        return state is None or state == self._OPEN_STATE

    def _ws_book_fresh(self, yq, nq) -> bool:
        """True when BOTH crossing legs have a recent, sized WS quote — so the live
        book is trustworthy enough to fire on without a REST depth re-fetch. Only
        WS-fed quotes carry a timestamp (REST-primed quotes don't), so primed/stale
        snapshots correctly fall through to the re-fetch path."""
        if self.max_ws_quote_age <= 0:
            return False
        if yq.yes_ask_size <= 0 or nq.no_ask_size <= 0:
            return False
        now = time.time()
        for quote in (yq, nq):
            ts = getattr(quote, "timestamp", 0.0) or 0.0
            if ts <= 0 or now - ts > self.max_ws_quote_age:
                return False
        return True

    async def _confirm_depth(self, p: ConfirmedPair):
        """Re-fetch real order-book depth for both legs and recompute the best
        direction with true sizes + fresh prices. Returns the eval tuple or None."""
        if self.depth_fetch is None:
            return None
        try:
            da = await self.depth_fetch(p.venue_a, p.market_a)
            db = await self.depth_fetch(p.venue_b, p.market_b)
        except Exception as exc:
            log.warning("depth fetch failed for %s: %s", p.event_key, exc)
            return None
        ev = self._eval_direction(da, db)
        if ev is None:
            # A leg had no usable two-sided quote (illiquid / one-sided book). Show
            # what came back so a "price edge but never trades" pair is explainable.
            log.info("STREAM %s: depth not two-sided — %s=%s %s=%s",
                     p.event_key, p.venue_a, _fmt_q(da), p.venue_b, _fmt_q(db))
        return ev

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

    async def _act_on_pair(self, key):
        """Edge-check one confirmed pair from the live book; execute if it clears the
        threshold, cooldown, and a depth re-validation. Returns a report or None."""
        p = self._pairs.get(key)
        if p is None:
            return None
        ev = self._best_direction(p)
        if ev is None:
            return None
        edge, yq, nq, size = ev
        if edge <= self.min_edge:                # price-edge gate (size checked below)
            return None
        if self.clock() < self._backoff_until.get(key, 0.0):
            return None                           # market keeps rejecting -> backing off
        if self.clock() - self._last_acted.get(key, -1e9) < self.cooldown:
            return None
        self._last_acted[key] = self.clock()      # cooldown set now to avoid REST storms
        # Both legs now stream a sized top-of-book (Polymarket full book + Kalshi ticker
        # sizes). If the live book is fresh + sized, trust it and fire — no REST round
        # trip. Otherwise re-fetch the real book (covers sizeless/stale/primed quotes);
        # the executor's bounded-aggressive limits absorb any residual move.
        if self.depth_fetch is not None and not self._ws_book_fresh(yq, nq):
            log.info("STREAM %s: price edge %.4f -> confirming real depth", p.event_key, edge)
            ev = await self._confirm_depth(p)
            if ev is None:
                return None
            edge, yq, nq, size = ev
            if edge <= self.min_edge or size < 1:
                log.info("STREAM edge gone after depth check on %s (edge %.4f sz %g)",
                         p.event_key, edge, size)
                return None
        elif size < 1:
            return None
        else:
            log.info("STREAM %s: edge %.4f from fresh WS book (sz %g) -> executing",
                     p.event_key, edge, size)
        # State guard: never fire into a non-OPEN market (halted/suspended/pre-open/
        # closing-auction/settled) — it would reject or settle against us.
        if not (self._quote_open(yq) and self._quote_open(nq)):
            log.info("STREAM %s: skip — leg not OPEN (%s=%s %s=%s)", p.event_key,
                     yq.venue, self._market_state.get((yq.venue, yq.market_id)) or yq.state,
                     nq.venue, self._market_state.get((nq.venue, nq.market_id)) or nq.state)
            return None
        opp = self._build_opp(p, edge, yq, nq, size)
        log.info("STREAM edge %.4f sz %g on %s -> executing", edge, size, p.event_key)
        report = await self._execute_guarded(opp)
        log.info("STREAM exec %s | %s", p.event_key, report)
        self._note_outcome(key, p, report)
        return report

    def _note_outcome(self, key, p, report) -> None:
        """Escalating backoff for a pair whose orders keep failing (e.g. a venue that
        rejects orders on a live/in-play market). A real fill resets it; a failed
        attempt (an order was placed but didn't lock the arb) pushes the next retry
        out exponentially, so the engine stops hammering an untradeable market."""
        from bot.execution.executor import ExecStatus

        status = getattr(report, "status", None)
        if status is None:
            return                                # not an ExecutionReport (test stub)
        if status is ExecStatus.SUCCESS:
            self._fail_counts.pop(key, None)
            self._backoff_until.pop(key, None)
            return
        if not getattr(report, "legs", None):
            return                                # pre-order skip (risk/size) — not a failure
        n = self._fail_counts.get(key, 0) + 1
        self._fail_counts[key] = n
        delay = min(self._backoff_base * (2 ** (n - 1)), self._backoff_cap)
        self._backoff_until[key] = self.clock() + delay
        log.info("STREAM backing off %s for %.0fs after failed attempt #%d (%s)",
                 p.event_key, delay, n, report.reason)

    async def _execute_guarded(self, opp):
        """Run ``executor.execute`` so a refresh-boundary consumer cancellation can
        never abort a half-placed trade. The execution runs as a tracked task and is
        awaited via ``asyncio.shield``: if our caller (a consumer) is cancelled, the
        trade keeps running to a definitive outcome (locked / unwound / halted) and
        ``run()`` awaits it before the next cycle. Leaving an order placed-but-tracked
        is the whole point — an abandoned leg is the dangerous state, not a slow one."""
        task = asyncio.ensure_future(self.executor.execute(opp))
        self._inflight.add(task)
        task.add_done_callback(self._inflight.discard)
        return await asyncio.shield(task)

    async def on_quote(self, q: MarketQuote):
        """Process one live quote: update the book, then act on any pair it touches.

        Returns the last ExecutionReport produced (or None). The cooldown prevents
        re-firing the same pair on every tick.
        """
        self.livebook.update(q)
        if getattr(q, "state", None):
            self._market_state[(q.venue, q.market_id)] = q.state
        report = None
        for key in self._index.get((q.venue, q.market_id), ()):
            r = await self._act_on_pair(key)
            if r is not None:
                report = r
        return report

    async def prime_and_sweep(self):
        """Seed the live book with a REST snapshot of every watchlist market, then
        edge-check every pair once. Closes the gap where a venue's WS (Kalshi ticker)
        only emits on price *change*, so a stable-priced leg would otherwise never
        enter the book — leaving real edges undetected until the price happened to move.
        """
        if self.depth_fetch is None:
            return
        primed = 0
        for (venue, market) in list(self._index):
            try:
                q = await self.depth_fetch(venue, market)
            except Exception:
                continue
            if q is not None:
                self.livebook.update(q)
                primed += 1
        log.info("primed live book with %d/%d market snapshots", primed, len(self._index))
        for key in list(self._pairs):
            await self._act_on_pair(key)

    async def _consume(self, venue) -> None:
        mids = self.market_ids.get(venue.name) or None
        if mids is None:
            return  # nothing confirmed for this venue this cycle
        async for q in venue.stream_order_book(mids):
            self._ws_counts[venue.name] = self._ws_counts.get(venue.name, 0) + 1
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
            self._ws_counts = {}
            # Seed the book with REST snapshots so a quiet (non-ticking) leg doesn't
            # leave pairs blind, and catch any edge already present at refresh time.
            await self.prime_and_sweep()
            consumers = [asyncio.create_task(self._consume(v)) for v in venues]
            try:
                await asyncio.sleep(refresh_interval)
            finally:
                for c in consumers:
                    c.cancel()
                await asyncio.gather(*consumers, return_exceptions=True)
                # A trade may have fired in the last instant before the interval
                # ended; the consumer that launched it is now cancelled, but the
                # shielded execution survives. Wait it out so we never start a new
                # cycle (or exit) with an order still in flight.
                if self._inflight:
                    log.warning("waiting for %d in-flight execution(s) to settle "
                                "before refresh", len(self._inflight))
                    await asyncio.gather(*list(self._inflight), return_exceptions=True)
            # WS health: how many live ticks each venue delivered this interval. A
            # venue at 0 means its market WebSocket isn't feeding the fast path.
            counts = {v.name: self._ws_counts.get(v.name, 0) for v in venues}
            dead = [name for name, n in counts.items() if n == 0]
            if dead:
                log.warning("WS health: %s — NO quotes this interval from %s", counts, dead)
            else:
                log.info("WS health: %s quotes this interval", counts)
            await self.log_edge_snapshot()

    def edge_snapshot(self, top: int = 5) -> list[tuple]:
        """Current best edge per pair, computed from the live WS book.

        Returns ``[(edge, pair, yes_quote, no_quote, size), ...]`` sorted best-first,
        only for pairs that have a two-sided quote (both legs present in the book).
        This is proof that WS prices are matched to the right events: an entry can only
        exist if both of a pair's markets have a current quote in the livebook."""
        rows = []
        for p in self._pairs.values():
            ev = self._best_direction(p)
            if ev is None:
                continue
            edge, yq, nq, size = ev
            rows.append((edge, p, yq, nq, size))
        rows.sort(key=lambda r: r[0], reverse=True)
        return rows[:top]

    async def log_edge_snapshot(self, top: int = 5) -> None:
        snap = self.edge_snapshot(top)
        priced = sum(1 for p in self._pairs.values() if self._best_direction(p) is not None)
        if not snap:
            log.info("edge snapshot: 0/%d pairs have two-sided WS quotes yet "
                     "(book still warming up?)", len(self._pairs))
            return
        # The live WS book has no Kalshi size (ticker is sizeless), so re-fetch the real
        # order book for the shown rows — the displayed price/size/edge then reflect
        # actual depth, not the sizeless WS quote. (Bounded: only the top rows.)
        rows = []
        for edge, p, yq, nq, size in snap:
            ev = await self._confirm_depth(p)
            rows.append(ev if ev is not None else (edge, yq, nq, size))
        log.info("edge snapshot (real book): %d/%d pairs two-sided, top %d:",
                 priced, len(self._pairs), len(snap))
        for edge, yq, nq, size in rows:
            log.info("  %s yes=%.2f + %s no=%.2f = %.2f | edge=%+.3f sz=%g",
                     yq.venue, yq.yes_ask, nq.venue, nq.no_ask,
                     yq.yes_ask + nq.no_ask, edge, size)
