"""Fast streaming engine: live quotes -> per-tick edge check -> execute."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections import defaultdict, deque
from dataclasses import dataclass

from bot.execution.executor import Executor
from bot.fees import FeeModel, ZeroFeeModel, per_contract_fee
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
    """Latest top-of-book quote per (venue, market_id), fed from WS streams.

    Also keeps a short EVENT-TIME history per market (quotes stamped with
    exchange_ts, ~2s deep) so the fast feed can be REWOUND to the slow feed's
    timestamp: the venues' pipelines differ by ~300ms (kalshi) vs ~40ms (poly),
    and comparing books from the same event instant separates a standing
    dislocation from a skew artifact."""

    HISTORY_SECS = 2.5

    def __init__(self) -> None:
        self._q: dict[tuple[str, str], MarketQuote] = {}
        self._hist: dict[tuple[str, str], deque] = {}

    def update(self, q: MarketQuote) -> None:
        self._q[(q.venue, q.market_id)] = q
        ts = getattr(q, "exchange_ts", None)
        if ts:
            h = self._hist.setdefault((q.venue, q.market_id), deque())
            h.append((ts, q))
            cutoff = ts - self.HISTORY_SECS
            while h and h[0][0] < cutoff:
                h.popleft()

    def get(self, venue: str, market_id: str) -> MarketQuote | None:
        return self._q.get((venue, market_id))

    def asof(self, venue: str, market_id: str, ts: float) -> MarketQuote | None:
        """The venue's book as it stood AT exchange time ``ts`` (latest quote with
        exchange_ts <= ts), or None when history doesn't reach back that far."""
        h = self._hist.get((venue, market_id))
        if not h or h[0][0] > ts:
            return None
        best = None
        for qts, q in h:
            if qts <= ts:
                best = q
            else:
                break
        return best

    def prune(self, keep: set[tuple[str, str]]) -> None:
        """Drop quotes for markets no longer watched (settled pairs would pile up forever)."""
        for k in [k for k in self._q if k not in keep]:
            del self._q[k]
        for k in [k for k in self._hist if k not in keep]:
            del self._hist[k]


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
        max_plausible_edge: float = 0.0,
        empirical_min_obs: int = 0,
        empirical_sum_floor: float = 0.93,
        maker_eligible=None,
        cooldown: float = 5.0,
        livebook: LiveBook | None = None,
        clock=time.monotonic,
        depth_fetch=None,
        open_check=None,
        max_ws_quote_age: float = 2.0,
        min_leg_price: float = 0.0,
        store=None,
        maker_mode: bool = False,
        hybrid_take_depth: float = 0.0,
        hybrid_take_bar: float = 0.0,
        reconcile_halt: bool = True,
        edge_snapshot_top: int = 5,
        edge_persist_secs: float = 0.0,
        sync_window_secs: float = 0.0,
        prime_concurrency: int = 8,
        ws_trust_min: int = 3,
        ws_trust_eps: float = 0.01,
        require_rules_verify: bool = False,
    ) -> None:
        self.executor = executor
        self.fee_models = fee_models or {}
        self.min_edge = min_edge
        # Implausible-edge guard: a genuine cross-venue arb is bounded by arbitrage to a few
        # percent — an "edge" above this is the signature of a FALSE MATCH (two different
        # events whose independent prices momentarily summed < $1), not a profit. e.g. the
        # R6-vs-CoD mismatch showed a 39% "edge" (YES+NO ≈ 0.68). Skip those, never trade
        # them. 0 = disabled. This is the cheap fire-path half of the empirical matcher.
        self.max_plausible_edge = max_plausible_edge
        # EMPIRICAL same-event confirmation: a structural/LLM match is only a HYPOTHESIS;
        # two markets are truly the same event iff their prices behave as complements —
        # YES_A + NO_B stays near $1. We sample that sum on every tick (in-memory, per pair)
        # and only let a pair TRADE once it has >= empirical_min_obs samples whose MEAN sum
        # is >= empirical_sum_floor. A false match (R6 vs CoD: mean sum ~0.68) never qualifies;
        # a real pair (mean ~1.0) confirms within minutes. min_obs 0 = disabled (no gate).
        self.empirical_min_obs = empirical_min_obs
        self.empirical_sum_floor = empirical_sum_floor
        self._sum_obs: dict[tuple, deque] = defaultdict(lambda: deque(maxlen=200))
        # Optional callable (venue, market_id) -> bool: may this market host a resting MAKER?
        # Used to keep makers off thin Polymarket markets (hedge would 500 -> naked) while
        # still allowing them to be TAKEN. None = no gate (any market can rest a maker).
        self.maker_eligible = maker_eligible
        self.cooldown = cooldown
        self.livebook = livebook or LiveBook()
        self.clock = clock
        # If both crossing legs have a WS quote with real size no older than this many
        # seconds, fire off the live book and skip the per-fire REST depth re-fetch.
        # 0 disables (always re-fetch). Quote timestamps are wall-clock (time.time()),
        # so freshness is checked against time.time(), not the monotonic ``clock``.
        self.max_ws_quote_age = max_ws_quote_age
        # Skip a fire if either crossing leg's ask is at a price extreme (<= this, or
        # >= 1 - this). A binary leg at ~$0.01 is effectively resolved/settling: its
        # "edge" is a phantom and it has no real resting volume (Kalshi then rejects the
        # FOK). 0 = disabled.
        self.min_leg_price = min_leg_price
        # Optional Store: log each actionable edge (post-cooldown) + its outcome, so a
        # soak measures how often/how big real edges actually are.
        self.store = store
        # Maker mode: rest the fee-heavy leg as a maker and complete it asynchronously
        # (so a pending maker doesn't block the quote loop). One maker per pair at a time.
        self.maker_mode = maker_mode
        # Hybrid take-or-rest (maker mode only): when a depth-confirmed edge has at least
        # ``hybrid_take_depth`` contracts of real top-of-book size AND clears the taker bar
        # ``hybrid_take_bar`` (the raw lock floor + the taker hedge buffer — NOT the maker
        # arm bar that ``min_edge`` carries in maker mode), TAKE it immediately (cross both
        # books, lock now) instead of resting a maker that may never get crossed. Deep-but-
        # thin edges (below the taker bar) and shallow edges still rest a maker. A take_depth
        # of 0 disables it (pure maker mode — never auto-takes).
        self.hybrid_take_depth = hybrid_take_depth
        self.hybrid_take_bar = hybrid_take_bar
        # Periodic cross-venue reconciliation: a locked arb holds EQUAL contracts on both
        # legs, so a per-pair size imbalance is naked directional exposure (a hedge that
        # failed to land). When True, a naked position confirmed on two consecutive checks
        # trips the kill switch so the bot stops until a human flattens. False = warn only.
        self.reconcile_halt = reconcile_halt
        self._reconcile_tol = 1.0               # contracts; below this is rounding, not naked
        # Don't flag a pair traded within this many seconds: Poly fills land instantly but
        # Kalshi's /positions read lags, so a rapid burst shows a TRANSIENT imbalance that
        # settles once trading stops. Without this, a frequent reconcile (every 30s via the
        # balance poll) caught a burst mid-flight and FALSE-halted on a pair that was fully
        # hedged seconds later. A real stranded leg persists after trading quiesces.
        self._reconcile_grace = 25.0
        # Venue-DOWN guard: if a venue's snapshot comes back empty (its API is down/erroring,
        # as Polymarket did), EVERY hedge on it reads 0 and looks naked at once. Real hedge
        # failures are isolated (the executor halts on a single one), so this many pairs going
        # naked simultaneously with the SAME venue reading ZERO total positions is a stale read,
        # not simultaneous failures — skip the halt for those (the next healthy cycle re-checks).
        self._venue_down_min = 2
        self._imbalanced_prev: set = set()      # pairs imbalanced last check (persistence)
        # How many of the best edges the periodic snapshot logs each interval.
        self.edge_snapshot_top = edge_snapshot_top
        # Require an edge to persist this many seconds before acting (distinguishes a real
        # venue-lag from a fleeting cross-feed timing artifact). 0 = act on first sighting.
        self.edge_persist_secs = edge_persist_secs
        # SUB-100ms path: if BOTH legs ticked within this tight window (very recent AND
        # close to each other -> the cross-feed is synced NOW, not one venue leading), a
        # deep edge is real immediately and fires WITHOUT the persist wait. 0 = disabled.
        self.sync_window_secs = sync_window_secs
        # Max concurrent REST snapshot fetches in prime_and_sweep (the rest are paced by
        # the per-venue rate limiters). Higher = faster watchlist refresh.
        self.prime_concurrency = prime_concurrency
        self._maker_inflight: set = set()
        self._take_inflight: set = set()        # pairs with a spawned hybrid TAKE running
        # Pair keys with the STRONGEST match evidence (rules-verified identical, or
        # settlement-verified consistent) — allowed to fire FAT edges with no price
        # history. Fed by the slow loop from Store.verified_pair_keys().
        self.verified_pairs: set = set()
        # deterministic NAME-level identity (complete person-codes / full
        # participant alignment) — may overrule PRICE-LEVEL evidence
        self.identity_certain: set = set()
        # Async callable depth_fetch(venue, market_id) -> sized MarketQuote | None.
        # WS ticker feeds carry no size (Kalshi), so before firing on a price edge we
        # re-fetch real order-book depth (which also re-validates the price).
        self.depth_fetch = depth_fetch
        # open_check(venue, market) -> True/False/None: authoritative settled status from the
        # venue's status field (the orderbook quote omits it — Kalshi reports state=None even
        # when finalized). Lets the reconcile tell a settled-leg leftover from a stranded leg.
        self.open_check = open_check
        self._pairs: dict[tuple, ConfirmedPair] = {}
        self._index: dict[tuple[str, str], set] = {}   # (venue,market) -> set of pair keys
        self._last_acted: dict[tuple, float] = {}
        self._edge_since: dict[tuple, float] = {}   # when a pair's edge first crossed (persistence)
        self._ws_counts: dict[str, int] = {}   # venue -> quotes seen since last refresh
        self._inflight: set = set()             # in-flight execute() tasks (cancel-shielded)
        # Escalating backoff for pairs whose orders keep failing (e.g. a venue that
        # rejects on a live/in-play market) so the engine stops hammering them.
        self._fail_counts: dict[tuple, int] = {}
        self._backoff_until: dict[tuple, float] = {}
        self._backoff_base = 60.0               # first penalty after a failed attempt
        self._backoff_cap = 1800.0              # max 30 min between retries
        self._confirm_fails: dict = {}          # consecutive confirm-rejections per pair
        # WS-TRUST LADDER: a pair earns trust each time the REST confirm AGREES with
        # what its WS books claimed (rest edge >= ws edge - eps: the book was honest,
        # whatever the edge size). After ws_trust_min consecutive agreements, a fresh
        # WS edge fires WITHOUT the ~35ms REST round-trip. Trust resets to zero on any
        # lying confirm (rest edge short by > eps) and on any failed/unwound execution
        # (the FOK legs bound the residual risk while trust is provisional). 0 = off.
        self.ws_trust_min = ws_trust_min
        self.ws_trust_eps = ws_trust_eps
        self._ws_trust: dict = {}               # consecutive WS-vs-REST agreements
        self._feed_lag: dict = {}               # venue -> EWMA transport lag (secs)
        self._settled_leg_until: dict = {}      # pair key -> sticky settled-leg expiry
        self._rules_blocked: dict = {}          # pair key -> last rules_pending block ts
        # FAST LANE: fresh blocks push the pair here; a dedicated verifier task
        # (dryrun.rules_fast_lane) judges it within one LLM call instead of waiting
        # for the pass-based loop (hot pairs blocked ~1000 ticks over 15 min).
        self.rules_priority_q = None            # asyncio.Queue set by the host
        self._rules_enqueued: set = set()
        self._rules_block_logged: dict = {}
        self._fat_logged: dict = {}     # rate-limit for the block log line
        # ONE-WAY pairs (timing-scope rules divergence): may fire ONLY with YES on
        # this venue (the wider settlement window) — the other direction is the
        # naked SUI-COL shape. key -> required YES venue name.
        self.one_way_yes: dict = {}
        # Divergence EDGE FLOORS: pairs whose rules diverge in a class with real
        # expected cost (tennis retirement, cricket rain...) trade only when the
        # edge ALSO pays for that risk: requires edge >= min_edge + extra.
        self.pair_edge_floor: dict = {}
        # PRE-TRADE RULES GATE: pairs may not fire until their resolution rules have
        # been LLM-compared once (rules_checked, fed by the rules loop). Closes the
        # race where a fresh pair trades minutes before verification reaches it.
        self.require_rules_verify = require_rules_verify
        self.rules_checked: set = set()
        self._preview_backoff = 60.0            # pause a pair whose hedge can't fill (preview)
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
        # Co-movement evidence: (last_yes, last_no, last_ts) + paired-delta history.
        # corr(dYes_A, -dNo_B) ~ +1 = same event (a level far from $1 is a REAL
        # dislocation); ~0 = independent propositions (the "edge" is noise between
        # strangers — including the level-check blind spot where two unrelated
        # 50/50s sum to ~$1.00 chronically).
        self._comove_last: dict = {}
        self._comove: dict = {}
        # Prune per-pair / per-market state for pairs that left the watchlist — these maps
        # otherwise grow forever across refreshes (a steady memory leak on long uptimes).
        live_keys = set(self._pairs)
        for d in (self._sum_obs, self._edge_since, self._fail_counts,
                  self._backoff_until, self._last_acted):
            for k in [k for k in d if k not in live_keys]:
                del d[k]
        live_markets = set(self._index)
        for k in [k for k in self._market_state if k not in live_markets]:
            del self._market_state[k]
        self.livebook.prune(live_markets)

    @property
    def market_ids(self) -> dict[str, list[str]]:
        """Confirmed markets grouped by venue (what to subscribe each WS to)."""
        out: dict[str, list[str]] = {}
        for (venue, market) in self._index:
            out.setdefault(venue, []).append(market)
        return out

    def _eval_direction(self, a: MarketQuote | None, b: MarketQuote | None):
        """Best (edge, yes_quote, no_quote, size) over both arb directions, or None.

        DEPTH SWEEP: when a quote carries a full ask ladder, pick the limit-price
        pair maximizing total profit across levels (an FOK at the limit sweeps every
        better level at its own price, so the estimate is conservative). The returned
        quotes are COPIES re-priced at the chosen limits — the executor's FOK order
        placed at these prices is exactly the sweep."""
        if a is None or b is None:
            return None
        from dataclasses import replace as _rp

        from bot.strategies.arbitrage import sweep_levels
        best = None
        for yq, nq in ((a, b), (b, a)):  # buy YES@yq + NO@nq
            if yq.yes_ask is None or nq.no_ask is None:
                continue
            fee_y, fee_n = self._fee(yq.venue), self._fee(nq.venue)
            swept = sweep_levels(
                yq.yes_ask_levels or ((yq.yes_ask, yq.yes_ask_size),),
                nq.no_ask_levels or ((nq.no_ask, nq.no_ask_size),),
                fee_y, fee_n, min_edge=self.min_edge)
            if swept is None:
                continue
            py, pn, size = swept
            if (py, pn) != (yq.yes_ask, nq.no_ask):
                yq = _rp(yq, yes_ask=py, yes_ask_size=size)
                nq = _rp(nq, no_ask=pn, no_ask_size=size)
            # Unrounded per-contract rate: fee(p, 1) cent-quantizes (error ~ the edge floor).
            fee = per_contract_fee(fee_y, py) + per_contract_fee(fee_n, pn)
            edge = 1.0 - (py + pn) - fee
            if best is None or edge * size > best[0] * best[3] or (best[3] <= 0 and edge > best[0]):
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

    def note_feed_lag(self, q) -> None:
        """Rolling per-venue transport lag (arrival − exchange time). Diagnostic only —
        surfaces one venue's feed falling behind (reconnect storms, venue degradation)."""
        ex = getattr(q, "exchange_ts", None)
        ts = getattr(q, "timestamp", 0.0) or 0.0
        if not ex or ts <= 0:
            return
        lag = max(0.0, ts - ex)
        if lag > 5.0:
            # snapshot/quiet-book message (ts = last CHANGE, arrival = now) — not
            # transport. Real pipeline lag measures well under 1s on both venues
            # (sampled min 6ms kalshi / 39ms poly); 24s "lags" were poisoning the
            # EWMA the operator reads.
            return
        prev = self._feed_lag.get(q.venue, lag)
        self._feed_lag[q.venue] = 0.9 * prev + 0.1 * lag

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

    def _aligned_edge_ok(self, yq, nq) -> bool:
        """Event-time ALIGNMENT: rewind the fresher feed to the slower feed's
        exchange timestamp and re-check this direction's edge at that COMMON
        instant. The venues' pipelines are skewed (~300ms kalshi vs ~40ms poly),
        so the mixed 'latest of each' view can show a phantom edge whose other
        half is still inside the slow pipeline. An edge that ALSO holds in the
        aligned view is a STANDING dislocation — trustworthy like a synced tick.
        Conservative by construction: history gaps or missing timestamps -> False."""
        if not self._ws_book_fresh(yq, nq):
            return False
        ty = getattr(yq, "exchange_ts", None)
        tn = getattr(nq, "exchange_ts", None)
        if not ty or not tn:
            return False
        t_star = min(ty, tn)
        if time.time() - t_star > 1.5:              # slow side too old to align against
            return False
        ay = yq if ty <= t_star + 1e-9 else self.livebook.asof(
            yq.venue, yq.market_id, t_star)
        an = nq if tn <= t_star + 1e-9 else self.livebook.asof(
            nq.venue, nq.market_id, t_star)
        if ay is None or an is None:
            return False
        ev = self._eval_direction(ay, an)
        return (ev is not None and ev[0] > self.min_edge and ev[3] >= 1
                and ev[1].venue == yq.venue)        # same direction as the live view

    def _ws_synced(self, yq, nq) -> bool:
        """True when BOTH legs ticked within a TIGHT window — each very recent AND close to
        the other in time — so the cross-feed is synchronized RIGHT NOW. That rules out the
        cross-feed flicker (one venue leading the other, whose timestamps would be far apart
        or one stale), so a synced+sized edge is real and can fire in tens of ms WITHOUT the
        persist wait. 0 window disables (-> normal persist/confirm path). Stricter than
        :meth:`_ws_book_fresh`, which only bounds age (~2s) and ignores inter-leg skew."""
        w = self.sync_window_secs
        if w <= 0 or yq.yes_ask_size <= 0 or nq.no_ask_size <= 0:
            return False
        ts_y = getattr(yq, "timestamp", 0.0) or 0.0
        ts_n = getattr(nq, "timestamp", 0.0) or 0.0
        if ts_y <= 0 or ts_n <= 0:
            return False
        now = time.time()
        if now - ts_y > w or now - ts_n > w:        # both legs must be VERY recent
            return False
        # EVENT-TIME sync: when both venues stamp their books with exchange time
        # (kalshi ts_ms, poly transactTime), compare when the books actually CHANGED
        # at the exchanges — immune to transport jitter, which arrival times conflate
        # with real repricing lag. Falls back to arrival when either side lacks it.
        ex_y = getattr(yq, "exchange_ts", None)
        ex_n = getattr(nq, "exchange_ts", None)
        if ex_y and ex_n:
            return abs(ex_y - ex_n) <= w             # both repriced together at source
        return abs(ts_y - ts_n) <= w                 # arrival-time approximation

    async def _confirm_depth(self, p: ConfirmedPair, *, quiet: bool = False):
        """Re-fetch real order-book depth for both legs and recompute the best
        direction with true sizes + fresh prices. Returns the eval tuple or None.

        The two legs are fetched CONCURRENTLY (asyncio.gather), so the confirmed edge is
        sampled on a snapshot as near-simultaneous as the network allows. Fetching them
        sequentially would leave the books tens of ms apart — and on a fast in-play market
        that residual skew is itself a source of phantom edges (the very thing this confirm
        exists to reject).

        ``quiet`` suppresses the per-pair "depth not two-sided" line — set it for bulk
        snapshot confirms (where one-sided books are expected and 20+ such lines would be
        noise); on the trade path it stays on so a "price edge but never trades" pair is
        explainable."""
        if self.depth_fetch is None:
            return None
        try:
            da, db = await asyncio.gather(
                self.depth_fetch(p.venue_a, p.market_a),
                self.depth_fetch(p.venue_b, p.market_b),
            )
        except Exception as exc:
            log.warning("depth fetch failed for %s: %s", p.event_key, exc)
            return None
        # RESEED the live book with the REST truth: a stale/phantom WS book otherwise
        # keeps re-triggering this same confirm every cooldown until the venue happens
        # to tick (the single largest source of REST volume — 12.6k confirms/48h, 91%
        # rejecting). REST quotes carry no timestamp, so they can never qualify for
        # the fresh-WS fast path; the next real WS tick overwrites them.
        for q in (da, db):
            if q is not None:
                self.livebook.update(q)
        ev = self._eval_direction(da, db)
        if ev is None and not quiet:
            # A leg had no usable two-sided quote (illiquid / one-sided book). Show
            # what came back so a "price edge but never trades" pair is explainable.
            log.info("STREAM %s: depth not two-sided — %s=%s %s=%s",
                     p.event_key, p.venue_a, _fmt_q(da), p.venue_b, _fmt_q(db))
        return ev

    def _observe(self, p, edge, yq, nq, size, outcome: str) -> None:
        """Log one actionable edge (post-cooldown) + outcome for the soak distribution."""
        if self.store is None:
            return
        try:
            self.store.record_edge(
                p.event_key, yq.venue, nq.venue, yq.yes_ask, nq.no_ask, edge, size, outcome)
        except Exception as exc:
            log.warning("edge log failed for %s: %s", p.event_key, exc)

    def _comove_corr(self, key):
        """corr(dYes_A, -dNo_B) over paired ticks, or None below 8 observations.
        +1 = the legs reprice on the same news (same event); ~0 = strangers."""
        h = self._comove.get(key)
        if not h or len(h) < 8:
            return None
        xs = [d[0] for d in h]
        ys = [-d[1] for d in h]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        vx = sum((x - mx) ** 2 for x in xs)
        vy = sum((y - my) ** 2 for y in ys)
        if vx <= 0 or vy <= 0:
            return None
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        return cov / (vx * vy) ** 0.5

    def _maybe_blacklist(self, p, key, reason: str) -> None:
        """Persist a CONFIRMED false match to the store blacklist so it's excluded from
        matching across restarts. Only once the verdict is backed by enough samples (not a
        one-off data blip) — the empirical mean must be both available and far from $1."""
        if self.store is None or self.empirical_min_obs <= 0:
            return
        obs = self._sum_obs.get(key)
        n = len(obs) if obs else 0
        if n < self.empirical_min_obs:
            return                                   # not enough evidence yet — just skip
        mean_sum = sum(obs) / n
        if mean_sum >= self.empirical_sum_floor:
            return                                   # behaves like a real complement — don't blacklist
        corr = self._comove_corr(key)
        if corr is not None and corr >= 0.6:
            # The legs reprice TOGETHER on the same news — the level being far from
            # $1 is a genuine (unarbed) dislocation, not proof of conflict. Level
            # says whether there is money; CO-MOVEMENT says whether it is the same
            # event. Never blacklist a co-moving pair on price level alone.
            log.info("STREAM %s: cheap sum but legs CO-MOVE (corr %.2f over %d "
                     "paired ticks) — dislocation, not conflict; not blacklisting",
                     p.event_key, corr, len(self._comove.get(key) or ()))
            return
        if key in self.verified_pairs:
            # IDENTITY-CERTAIN pairs are never blacklisted on PRICE evidence: a
            # persistently cheap sum on a verified complement is wide/stale books
            # or a scope divergence — the rules/divergence engine owns those
            # (one-way, edge floors), not a permanent park. Audit 2026-07-09:
            # 137 ids+names-aligned pairs were price-blacklisted, incl. a pair we
            # HELD 48 locked contracts of and the FRA-MAR FTTS set.
            log.info("STREAM %s: cheap sum %.3f over %d on an identity-certain pair "
                     "— NOT blacklisting (divergence policy owns this)",
                     p.event_key, mean_sum, n)
            return
        try:
            self.store.blacklist_pair(
                p.venue_a, p.market_a, p.venue_b, p.market_b,
                reason=reason, mean_sum=round(mean_sum, 4), samples=n)
            log.warning("STREAM %s: BLACKLISTED as a false match (%s, mean sum %.3f over %d) "
                        "— excluded from matching from now on", p.event_key, reason, mean_sum, n)
        except Exception as exc:
            log.warning("blacklist write failed for %s: %s", p.event_key, exc)

    def _build_opp(self, p, edge, yq, nq, size) -> ArbOpportunity:
        from bot.strategies.arbitrage import _pair_settle_ts
        gross = yq.yes_ask + nq.no_ask
        ts_y = getattr(yq, "timestamp", 0.0) or 0.0
        ts_n = getattr(nq, "timestamp", 0.0) or 0.0
        return ArbOpportunity(
            event_key=p.event_key,
            buy_yes_venue=yq.venue, buy_yes_market=yq.market_id,
            buy_no_venue=nq.venue, buy_no_market=nq.market_id,
            yes_price=yq.yes_ask, no_price=nq.no_ask, gross_cost=gross,
            fee_per_pair=max(0.0, 1.0 - gross - edge), edge_per_contract=edge,
            max_contracts=size, total_fees=0.0, total_profit=edge * size, notional=gross * size,
            yes_size=yq.yes_ask_size, no_size=nq.no_ask_size,
            yes_levels=getattr(yq, "yes_ask_levels", None),
            no_levels=getattr(nq, "no_ask_levels", None),
            # Settlement horizon (close_time, falling back to the id-parsed date). The
            # fast path is where ALL live fires happen — omitting this left settle_ts=0
            # and the horizon gate silently OFF for streaming trades (the CA-gov pair
            # kept locking months-out capital at a ~1% edge straight through the gate).
            settle_ts=_pair_settle_ts(yq, nq),
            # Oldest leg quote's wall-clock ts, only when BOTH legs are WS-stamped —
            # lets the executor skip its hedge REST re-read on a fresh, deep book.
            fresh_ts=min(ts_y, ts_n) if ts_y > 0 and ts_n > 0 else 0.0,
        )

    async def _act_on_pair(self, key):
        """Edge-check one confirmed pair from the live book; execute if it clears the
        threshold, cooldown, and a depth re-validation. Returns a report or None."""
        p = self._pairs.get(key)
        if p is None:
            return None
        # When the kill switch is tripped the executor SKIPs every order anyway, so doing
        # the per-tick edge math + a REST depth-confirm round trip is pure waste — and a
        # post-halt confirm storm starves the WS keepalive (a cause of the hedge-venue
        # disconnect churn). Stop all work for the pair until a restart clears the halt.
        risk = getattr(self.executor, "risk", None)
        if risk is not None and getattr(risk, "is_killed", False):
            return None
        if self.maker_mode and (key in self._maker_inflight or key in self._take_inflight):
            return None                # already resting a maker / running a take for this pair
        ev = self._best_direction(p)
        if ev is None:
            return None
        edge, yq, nq, size = ev
        # EMPIRICAL observation: sample this pair's YES+NO sum on EVERY tick (incl. no-edge
        # ticks where the sum is >= 1), building the price-behavior history the same-event
        # gate below relies on. Cheap, in-memory, bounded.
        if self.empirical_min_obs > 0 and yq.yes_ask is not None and nq.no_ask is not None:
            self._sum_obs[key].append(yq.yes_ask + nq.no_ask)
            # paired deltas: only when BOTH legs changed since the last sample —
            # that's the co-movement experiment (same news moves both, or doesn't)
            last = self._comove_last.get(key)
            self._comove_last[key] = (yq.yes_ask, nq.no_ask)
            if last is not None:
                dy, dn = yq.yes_ask - last[0], nq.no_ask - last[1]
                if abs(dy) >= 0.01 and abs(dn) >= 0.01:
                    h = self._comove.setdefault(key, deque(maxlen=60))
                    h.append((dy, dn))
        if edge <= self.min_edge:                # price-edge gate (size checked below)
            self._edge_since.pop(key, None)      # edge gone -> reset persistence timer
            return None
        # Implausible-edge guard (empirical sanity): an edge this large can't be a real arb —
        # it means the two legs are NOT complements (a false same-event match). Skip + record
        # it so the empirical observer learns the pair's sum sits far from $1.
        # FAT-EDGE evidence escalation (was a hard ceiling): a huge "edge" is USUALLY the
        # signature of a false match (spread-vs-moneyline pairings show 10-44%), but a hard
        # cap also rejected genuine dislocations — money left on the table. Edge size doesn't
        # determine truth; PRICE HISTORY does. So above the threshold we don't reject — we
        # demand STRONGER empirical proof of complementarity: ~3x the samples and a mean
        # YES+NO sum >= ~0.97 (a true pair's history sits at ~$1 with a momentary dip; a
        # spread-pair's mean lives at 0.55-0.90 and never qualifies). A proven pair fires at
        # ANY edge; an unproven one keeps observing (and the empirical gate below still
        # blacklists persistent non-complements). With the empirical gate DISABLED there is
        # no evidence to escalate to, so the conservative hard skip is kept.
        if self.max_plausible_edge > 0 and edge > self.max_plausible_edge:
            if key in self.verified_pairs:
                # Definitional truth outranks price statistics: the pair's RESOLUTION
                # RULES were verified identical (or it has settled consistently before),
                # so a fat edge is a genuine dislocation — PROCEED with no history
                # needed (size/staleness/depth checks below still apply; a dust book
                # advertising +11c at 0.1 contracts dies at those, so this line says
                # "eligible", not "fired"). Rate-limited: a snapshot flood logged it
                # 12x/40ms.
                now_log = self.clock()
                if now_log - self._fat_logged.get(key, 0.0) > 60:
                    self._fat_logged[key] = now_log
                    log.warning("STREAM %s: FAT edge %+.3f on a verified pair — "
                                "eligible without price history (guards still apply)",
                                p.event_key, edge)
            elif self.empirical_min_obs <= 0:
                log.warning("STREAM %s: edge %+.3f > %.3f and no empirical gate to verify "
                            "complementarity — skipping (likely FALSE MATCH)",
                            p.event_key, edge, self.max_plausible_edge)
                self._observe(p, edge, yq, nq, size, "implausible_edge_false_match")
                self._backoff_until[key] = self.clock() + self._backoff_cap
                return None
            else:
                obs = self._sum_obs.get(key)
                n = len(obs) if obs else 0
                need_n = max(3 * self.empirical_min_obs, 30)
                need_mean = max(self.empirical_sum_floor, 0.97)
                # BAND, not floor: a true complement's ASK sum sits ~$1.00-1.05. A mean
                # far ABOVE $1 means chronically wide/illiquid books (observed live: a
                # LoL pair averaging 1.281 "fat-fired" when one book collapsed — a quote
                # pull, not a dislocation; the FOK then rejects/strands). Fat edges only
                # fire on TIGHT proven complements.
                max_mean = 1.06
                mean_sum = (sum(obs) / n) if n else 0.0
                corr = self._comove_corr(key)
                anti = corr is not None and corr < 0.2 and len(self._comove.get(key) or ()) >= 15
                if n < need_n or mean_sum < need_mean or mean_sum > max_mean or anti:
                    if n in (1, need_n // 2):        # occasional heartbeat, not tick spam
                        log.info("STREAM %s: fat edge %+.3f needs stronger proof — %d/%d "
                                 "samples, mean sum %.3f (need %.3f-%.2f), comove %s; "
                                 "observing",
                                 p.event_key, edge, n, need_n, mean_sum, need_mean,
                                 max_mean, f"{corr:.2f}" if corr is not None else "n/a")
                    self._observe(p, edge, yq, nq, size, "fat_edge_unproven")
                    # Evidence-based blacklisting still applies: a pair whose sum history
                    # sits far from $1 over enough samples is a false match, parked.
                    self._maybe_blacklist(p, key, f"implausible edge {edge:+.3f}")
                    return None
                log.warning("STREAM %s: FAT edge %+.3f on a PROVEN complement (mean sum "
                            "%.3f over %d samples) — genuine dislocation, firing",
                            p.event_key, edge, mean_sum, n)
        # EMPIRICAL same-event gate: only TRADE a pair once its observed YES+NO sum confirms
        # the legs are complements. Insufficient history -> OBSERVE, don't trade yet (a new
        # real pair confirms within minutes as ticks accrue). Mean sum below the floor -> a
        # false match (its prices don't sum to ~$1) -> never trade. This is the authority;
        # the structural/LLM match only nominated the pair.
        if self.empirical_min_obs > 0:
            obs = self._sum_obs.get(key)
            n = len(obs) if obs else 0
            if n < self.empirical_min_obs and key not in self.verified_pairs:
                # A rules/settlement-VERIFIED pair skips the observation wait entirely
                # (definitional truth needs no price history); everyone else observes.
                if n == 1 or n == self.empirical_min_obs // 2:   # occasional heartbeat
                    log.info("STREAM %s: observing (%d/%d samples) before trading",
                             p.event_key, n, self.empirical_min_obs)
                return None
            mean_sum = (sum(obs) / n) if n else 1.0
            if n >= self.empirical_min_obs and mean_sum < self.empirical_sum_floor:
                # Defense in depth: an empirical NON-complement blocks even a verified
                # pair (two truth signals disagreeing = investigate, never trade).
                if key in self.identity_certain:
                    # identity proven DETERMINISTICALLY (complete person-code /
                    # full name alignment: the EBOO draft roster, 27% vs 63%
                    # across venues = a real dislocation on a sleepy book).
                    # Identity outranks price level — PROCEED to trade; the
                    # reliability ladder probe-sizes the first fires and
                    # settlement adjudicates.
                    now_l = self.clock()
                    if now_l - self._rules_block_logged.get(("ic",) + tuple(key), 0.0) > 300:
                        self._rules_block_logged[("ic",) + tuple(key)] = now_l
                        log.warning("STREAM %s: cheap sum %.3f on an IDENTITY-CERTAIN "
                                    "pair — real dislocation (probe-sized by the "
                                    "ladder)", p.event_key, mean_sum)
                elif key in self.verified_pairs:
                    log.warning("STREAM %s: CONFLICT — pair is rules/settlement-verified "
                                "but its price history says non-complement (mean %.3f); "
                                "trusting the prices, not trading", p.event_key, mean_sum)
                else:
                    log.warning("STREAM %s: empirical reject — mean YES+NO sum %.3f < %.3f "
                                "over %d samples (legs not complementary -> FALSE MATCH)",
                                p.event_key, mean_sum, self.empirical_sum_floor, n)
                    self._maybe_blacklist(p, key, f"mean sum {mean_sum:.3f} < "
                                                  f"{self.empirical_sum_floor}")
                if key not in self.identity_certain:
                    self._observe(p, edge, yq, nq, size, "empirical_reject_false_match")
                    self._backoff_until[key] = self.clock() + self._backoff_cap
                    return None
        # SUB-100ms SYNC PATH: when BOTH legs just ticked within the tight sync window the
        # cross-feed is synchronized NOW, so this is a genuine edge — not a one-sided flicker.
        # Skip the persist wait entirely and take it in tens of ms. Gated to deep books (the
        # TAKE path; a stale top unwinds cleanly, bounded by the per-order cap). Everything
        # else keeps the persist guard below.
        vdown = getattr(self.executor, "venue_down", None)
        if vdown and (yq.venue in vdown or nq.venue in vdown):
            self._observe(p, edge, yq, nq, size, "venue_outage")
            return None                    # can't hedge into a downed venue
        req_yes = self.one_way_yes.get(key)
        if req_yes and yq.venue != req_yes:
            self._observe(p, edge, yq, nq, size, "unsafe_direction_timing_scope")
            return None       # only the windfall-shaped direction may trade
        extra = self.pair_edge_floor.get(key, 0.0)
        if extra > 0 and edge < self.min_edge + extra - 1e-9:
            self._observe(p, edge, yq, nq, size, "edge_below_divergence_floor")
            return None       # the edge must also pay for the divergence risk
        if (self.require_rules_verify and key not in self.rules_checked
                and key not in self.verified_pairs):
            # visible + prioritized: the rules loop verifies blocked-with-live-edge
            # pairs FIRST (a real 7c dislocation sat silently blocked for minutes on
            # 2026-07-07 while the loop worked the watchlist in arbitrary order)
            self._rules_blocked[key] = self.clock()
            if self.rules_priority_q is not None and key not in self._rules_enqueued:
                self._rules_enqueued.add(key)
                try:
                    self.rules_priority_q.put_nowait(p)
                except Exception:
                    self._rules_enqueued.discard(key)
            now = self.clock()
            if now - self._rules_block_logged.get(key, 0.0) > 300:
                self._rules_block_logged[key] = now
                log.info("STREAM %s: edge %.4f BLOCKED pending rules verification "
                         "(prioritized for next pass)", p.event_key, edge)
            self._observe(p, edge, yq, nq, size, "rules_pending")
            return None                       # never trade ahead of the rules pass
        deep = self.hybrid_take_depth > 0 and size >= self.hybrid_take_depth
        sync_fast_take = self.maker_mode and deep and self._ws_synced(yq, nq)
        if not sync_fast_take and self.maker_mode and deep and self.sync_window_secs > 0:
            if self._aligned_edge_ok(yq, nq):
                log.info("STREAM %s: edge holds at ALIGNED event time (skew-rewound) "
                         "-> sync take", p.event_key)
                sync_fast_take = True
        # Persistence filter: a cross-feed timing artifact (one venue's WS leading the other
        # for a beat) flickers — it appears for ~100ms and vanishes when the lagging leg
        # catches up. A REAL venue-lag edge persists for the duration of the lag (seconds).
        # So only act once the edge has held continuously for ``edge_persist_secs``, which
        # separates genuine lag from skew without a slow REST round-trip. 0 = disabled. The
        # sync path above proves sync directly, so it bypasses this wait.
        if not sync_fast_take and self.edge_persist_secs > 0:
            first = self._edge_since.get(key)
            if first is None:
                self._edge_since[key] = self.clock()
                return None                       # first sighting — wait for it to persist
            if self.clock() - first < self.edge_persist_secs:
                return None                       # not held long enough yet
        if self.clock() < self._backoff_until.get(key, 0.0):
            return None                           # market keeps rejecting -> backing off
        if self.clock() - self._last_acted.get(key, -1e9) < self.cooldown:
            return None
        self._last_acted[key] = self.clock()      # cooldown set now to avoid REST storms
        # LATENCY FAST PATH: a FRESH, SIZED WS book deep enough to TAKE fires WITHOUT the
        # REST depth-confirm round-trip — so we beat slower actors to the edge instead of
        # losing it in the ~35ms confirm window (the "edge gone after depth check" misses).
        # Limited to deep hybrid TAKEs (both legs FOK -> a stale WS top just unwinds, bounded
        # by the per-order cap), and the executor's hedge-fillable check still re-reads the
        # taker book before committing leg 1 — so no leg is fired truly blind. Maker rests
        # and thin/sizeless books still REST-confirm first (resting on a phantom edge is
        # wasteful, and a maker can't unwind as cleanly as a FOK take).
        ws_fresh = self._ws_book_fresh(yq, nq)
        # WS-TRUST LADDER: this pair's WS books have agreed with the REST truth
        # ws_trust_min times straight — fire on the fresh book without the REST
        # round-trip. FOK legs bound the downside; any lie or failed execution
        # resets trust and the pair goes back to confirming.
        trusted = (self.ws_trust_min > 0 and ws_fresh and size >= 1
                   and self._ws_trust.get(key, 0) >= self.ws_trust_min)
        fast_take = sync_fast_take or (self.maker_mode and ws_fresh and deep)
        # FAT edges on IDENTITY-UNCERTAIN pairs never skip the REST confirm — for
        # them a sudden implausible edge is usually a book event (quote pull) or a
        # false match, and one round-trip verifies the whole ladder. VERIFIED /
        # identity-certain pairs keep their earned fast paths at ANY edge size: a
        # true complement under $1 pays regardless of why the edge exists, FOK legs
        # + the ceiling hedge bound the downside, and the slippage feedback revokes
        # trust if the book turns out to be lying.
        if (self.max_plausible_edge > 0 and edge > self.max_plausible_edge
                and key not in self.verified_pairs):
            trusted = False
            fast_take = False
        extreme = False
        if self.min_leg_price > 0 and yq is not None and nq is not None:
            _lo, _hi = self.min_leg_price, 1.0 - self.min_leg_price
            extreme = not (_lo <= yq.yes_ask <= _hi and _lo <= nq.no_ask <= _hi)
        if extreme:
            # extreme-priced legs fire only through a REST confirm: phantom depth at
            # $0.01/$0.99 is exactly what the WS book lies about on settling markets
            trusted = False
            fast_take = False
        if (self.depth_fetch is not None and not fast_take and not trusted
                and (self.maker_mode or not ws_fresh)):
            ws_claim = edge
            log.info("STREAM %s: price edge %.4f -> confirming real depth", p.event_key, edge)
            ev = await self._confirm_depth(p)
            if ev is None:
                return None
            edge, yq, nq, size = ev
            # feed the ladder: honest book (rest within eps of the WS claim) climbs;
            # a lying book resets to zero
            if edge >= ws_claim - self.ws_trust_eps:
                self._ws_trust[key] = self._ws_trust.get(key, 0) + 1
            else:
                self._ws_trust[key] = 0
            if edge <= self.min_edge or size < 1:
                # CONVERT before backing off: the confirm just fetched BOTH full
                # ladders — exactly what a maker rest needs. No taker edge does not
                # mean no opportunity: the maker manufactures its price deeper in
                # the spread (executor caps the rest so a fill locks >= floor +
                # cushion). Resting is free; a cheap "no room"/thin skip falls
                # through to the normal backoff.
                maker_note = ""
                if (self.maker_mode and yq is not None and nq is not None
                        and (not self.require_rules_verify
                             or key in self.rules_checked)):
                    arm = self.min_edge + getattr(self.executor,
                                                  "maker_arm_cushion", 0.005)
                    opp_mk = self._build_opp(p, arm, yq, nq, max(size, 1))
                    mk = getattr(self.executor, "execute_maker", None)
                    if opp_mk is not None and callable(mk):
                        rep = await mk(opp_mk)
                        st = getattr(rep, "status", None)
                        if st is not None and st.name in ("SUCCESS", "UNWOUND", "HALTED"):
                            self._note_outcome(key, p, rep)
                            self._observe(p, edge, yq, nq, size, f"maker_{st.name}")
                            return rep
                        maker_note = (f"; maker: {getattr(rep, 'reason', st)}"
                                      if st is not None else "")
                n = self._confirm_fails.get(key, 0) + 1
                self._confirm_fails[key] = n
                delay = min(self._backoff_base * (2 ** (n - 1)), self._backoff_cap)
                self._backoff_until[key] = self.clock() + delay
                log.info("STREAM edge gone after depth check on %s (edge %.4f sz %g) — "
                         "backoff %.0fs (miss #%d)%s", p.event_key, edge, size, delay,
                         n, maker_note)
                self._observe(p, edge, yq, nq, size, "edge_gone_after_depth")
                return None
            self._confirm_fails.pop(key, None)     # WS agreed with REST — full cadence
        elif size < 1:
            return None
        else:
            log.info("STREAM %s: edge %.4f from fresh WS book (sz %g) -> fast executing "
                     "(%s)", p.event_key, edge, size,
                     "SYNC sub-100ms TAKE" if sync_fast_take
                     else "deep TAKE, no REST confirm" if fast_take
                     else f"TRUSTED WS x{self._ws_trust.get(key, 0)}" if trusted
                     else "fresh WS")
        # State guard: never fire into a non-OPEN market (halted/suspended/pre-open/
        # closing-auction/settled) — it would reject or settle against us.
        if not (self._quote_open(yq) and self._quote_open(nq)):
            log.info("STREAM %s: skip — leg not OPEN (%s=%s %s=%s)", p.event_key,
                     yq.venue, self._market_state.get((yq.venue, yq.market_id)) or yq.state,
                     nq.venue, self._market_state.get((nq.venue, nq.market_id)) or nq.state)
            self._observe(p, edge, yq, nq, size, "skip_not_open")
            return None
        # Price-extreme guard: a leg at ~$0.01/$0.99 is a settling/resolved market with
        # phantom depth (no real resting volume) — its edge is an artifact. Skip it.
        if self.min_leg_price > 0:
            # recompute on the CURRENT quotes — the REST confirm may have replaced
            # yq/nq since the pre-confirm extremeness check
            lo, hi = self.min_leg_price, 1.0 - self.min_leg_price
            extreme = not (lo <= yq.yes_ask <= hi and lo <= nq.no_ask <= hi)
            if extreme and key not in self.verified_pairs:
                # phantom-depth artifact of a settling market — unless the pair's
                # IDENTITY is certain: a verified complement at 0.02/0.95 is a real
                # blowout edge (the lifecycle gate handles non-OPEN separately, and
                # the forced REST confirm below proves the depth is real).
                log.info("STREAM %s: skip — leg at price extreme (yes=%.3f no=%.3f), "
                         "likely settling", p.event_key, yq.yes_ask, nq.no_ask)
                self._observe(p, edge, yq, nq, size, "skip_settling")
                return None
        opp = self._build_opp(p, edge, yq, nq, size)
        if self.maker_mode:
            # Hybrid take-or-rest: a depth-confirmed edge that has REAL top-of-book size
            # (>= hybrid_take_depth) AND clears the taker bar (lock floor + hedge buffer)
            # is takeable right now — cross both books and lock it immediately rather than
            # resting a maker that may never get crossed. Edges that are deep but thin
            # (below the taker bar) or shallow fall through to the maker. 0 = disabled.
            take_bar = (self.hybrid_take_bar(size) if callable(self.hybrid_take_bar)
                        else self.hybrid_take_bar)
            if (self.hybrid_take_depth > 0 and size >= self.hybrid_take_depth
                    and edge >= take_bar - 1e-9):
                log.info("STREAM edge %.4f sz %g on %s -> hybrid TAKE "
                         "(depth >= %g, clears taker bar %.4f)",
                         edge, size, p.event_key, self.hybrid_take_depth, take_bar)
                # Spawn the take OFF the quote loop (like makers): awaiting the two-leg
                # execution inline (~hundreds of ms) stalls THIS VENUE'S ENTIRE WS stream —
                # other pairs' edges go unevaluated exactly when game events move many
                # markets at once. Tracked + never cancelled; run() drains before resubscribe.
                self._take_inflight.add(key)
                task = asyncio.ensure_future(self._run_take(key, p, opp, edge, yq, nq, size))
                self._inflight.add(task)
                task.add_done_callback(self._inflight.discard)
                return None
            # Maker-volume gate: only REST a maker when the Polymarket HEDGE market is liquid
            # enough that its hedge will reliably fill. On a thin market the hedge 500s and
            # leaves the maker fill naked (the incident). Such markets stay TAKE-able above —
            # a 500 on a taker leg is a clean skip via the leg-order fix — just never rested.
            if self.maker_eligible is not None:
                poly_v, poly_m = ((p.venue_a, p.market_a) if p.venue_a == "polymarket_us"
                                  else (p.venue_b, p.market_b))
                if not self.maker_eligible(poly_v, poly_m):
                    log.info("STREAM %s: maker NOT rested — hedge market below volume floor "
                             "(thin -> TAKE-only, never naked)", p.event_key)
                    return None
            # Rest a maker and complete it asynchronously so a pending maker doesn't
            # block the quote loop (it may wait seconds to fill). Tracked + shielded.
            self._maker_inflight.add(key)
            log.info("STREAM edge %.4f sz %g on %s -> resting maker", edge, size, p.event_key)
            task = asyncio.ensure_future(self._run_maker(key, p, opp, edge, yq, nq, size))
            self._inflight.add(task)
            task.add_done_callback(self._inflight.discard)
            return None
        log.info("STREAM edge %.4f sz %g on %s -> executing", edge, size, p.event_key)
        report = await self._execute_guarded(opp)
        log.info("STREAM exec %s | %s", p.event_key, report)
        status = getattr(report, "status", None)
        self._observe(p, edge, yq, nq, size, status.value if status is not None else "executed")
        self._note_outcome(key, p, report)
        return report

    async def _run_take(self, key, p, opp, edge, yq, nq, size) -> None:
        """Run a hybrid TAKE to completion off the quote loop, then log + book it."""
        try:
            trig = max(getattr(yq, "timestamp", 0.0) or 0.0,
                       getattr(nq, "timestamp", 0.0) or 0.0)
            if trig > 0:
                lat_ms = (time.time() - trig) * 1000.0
                self._fire_lat_max = max(getattr(self, "_fire_lat_max", 0.0), lat_ms)
                log.info("STREAM %s: fire latency %.0fms (tick arrival -> order send)",
                         p.event_key, lat_ms)
            report = await self.executor.execute(opp)
            if (getattr(report, "status", None) is not None
                    and str(getattr(report, "reason", "")).startswith("leg1 not filled")):
                # The book moved in flight — but BOTH legs may have moved such that
                # the pair still clears the floor at a different price split. One
                # immediate re-evaluation from the live book (already updated by the
                # in-flight WS ticks); a genuine vanished edge fails it and skips.
                ev2 = self._eval_direction(self.livebook.get(p.venue_a, p.market_a),
                                           self.livebook.get(p.venue_b, p.market_b))
                if ev2 is not None and ev2[0] > self.min_edge and ev2[3] >= 1:
                    edge2, yq2, nq2, size2 = ev2
                    opp2 = self._build_opp(p, edge2, yq2, nq2, size2)
                    if opp2 is not None:
                        log.info("STREAM %s: leg1 miss — re-firing at fresh prices "
                                 "(edge %.4f sz %g)", p.event_key, ev2[0], ev2[3])
                        report = await self.executor.execute(opp2)
            log.info("STREAM exec %s | %s", p.event_key, report)
            status = getattr(report, "status", None)
            self._observe(p, edge, yq, nq, size,
                          status.value if status is not None else "executed")
            self._note_outcome(key, p, report)
        except Exception as exc:
            log.warning("take run failed for %s: %s", p.event_key, exc)
        finally:
            self._take_inflight.discard(key)

    async def drain(self) -> None:
        """Await all in-flight executions (makers + takes) to a definitive outcome."""
        while self._inflight:
            await asyncio.gather(*list(self._inflight), return_exceptions=True)

    async def _run_maker(self, key, p, opp, edge, yq, nq, size) -> None:
        """Run a maker execution to completion off the quote loop, then log + book it."""
        try:
            report = await self.executor.execute_maker(opp)
            log.info("STREAM maker %s | %s", p.event_key, report)
            status = getattr(report, "status", None)
            self._observe(p, edge, yq, nq, size,
                          ("maker_" + status.value) if status is not None else "maker")
            self._note_outcome(key, p, report)
        except Exception as exc:
            log.warning("maker run failed for %s: %s", p.event_key, exc)
        finally:
            self._maker_inflight.discard(key)

    def _note_outcome(self, key, p, report) -> None:
        """Escalating backoff for a pair whose orders keep failing (e.g. a venue that
        rejects orders on a live/in-play market). A real fill resets it; a genuinely
        hostile outcome (a rejected leg, an unwind, a halt) pushes the next retry out
        exponentially. A maker that simply RESTED and expired unfilled is NOT a failure —
        nobody crossed it yet — so it must not back off, or we'd cede the book most of the
        time on a persistent edge instead of re-resting to capture it (cooldown still
        throttles the re-rest)."""
        from bot.execution.executor import ExecStatus
        from bot.execution.orders import OrderStatus

        status = getattr(report, "status", None)
        reason = str(getattr(report, "reason", "") or "")
        if status is ExecStatus.SKIPPED and reason.startswith("horizon:"):
            # A horizon skip is decided by the settle DATE — it will not change for
            # weeks. Re-confirming the same real-but-long-dated edge every cooldown
            # burned 22 REST round-trips on one CA-governor pair in an evening; park
            # it for 6h instead (it re-evaluates when the horizon or edge changes).
            self._backoff_until[key] = self.clock() + 6 * 3600.0
            log.info("STREAM %s: horizon-skipped — parking 6h (%s)", p.event_key,
                     reason[:80])
            return
        if status is None:
            return                                # not an ExecutionReport (test stub)
        if status is ExecStatus.SUCCESS:
            self._fail_counts.pop(key, None)
            self._backoff_until.pop(key, None)
            # SLIPPAGE feedback: the breakeven-ceiling hedge makes fills succeed even
            # when the WS book lied about the price (the lock lands near 0 instead of
            # the detected edge). Fill-success is then NOT evidence of book honesty —
            # a lock realizing under half the floor revokes WS trust so the next fire
            # on this pair goes back through the REST confirm.
            try:
                legs = getattr(report, "legs", None) or []
                cts = max((l.filled or 0) for l in legs) if legs else 0
                per_ct = (report.realized_pnl or 0.0) / cts if cts else None
            except Exception:
                per_ct = None
            if per_ct is not None and per_ct < self.min_edge * 0.5:
                if self._ws_trust.pop(key, None) is not None:
                    log.info("STREAM %s: locked %.3f/ct << detected edge — WS book "
                             "overstated; trust revoked (next fire re-confirms)",
                             p.event_key, per_ct)
            return
        reason = getattr(report, "reason", "") or ""
        if status is ExecStatus.SKIPPED and "hedge unfillable" in reason:
            # The hedge can't fill at the edge price right now (thin top-of-book — a real,
            # fillable edge would have cleared the live-book check). Don't re-confirm it every
            # cooldown; back the pair off briefly so we stop hammering REST on an unfillable
            # edge. It re-enters naturally once the book actually supports the hedge.
            self._backoff_until[key] = self.clock() + self._preview_backoff
            return
        legs = getattr(report, "legs", None)
        if not legs:
            return                                # pre-order skip (risk/size) — not a failure
        rejected = any(getattr(leg, "status", None) is OrderStatus.REJECTED for leg in legs)
        if status is ExecStatus.SKIPPED and not rejected:
            # Benign no-trade: a maker rested and expired uncrossed. Clear any prior penalty
            # and let it re-rest (gated only by the cooldown) while the edge persists.
            self._fail_counts.pop(key, None)
            return
        # An UNWOUND outcome cost real money (crossed a spread to escape); it parks
        # the pair twice as hard as a clean failure — overnight thin books were
        # re-probing every ~30-45min for another 2-3c unwind, all night.
        bump = 2 if status is ExecStatus.UNWOUND else 1
        n = self._fail_counts.get(key, 0) + bump
        self._fail_counts[key] = n
        self._ws_trust.pop(key, None)           # a failed fire revokes WS trust too
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

        TOP-OF-LADDER FILTER: the per-event book feed delivers ~280 msg/s (30x the
        old conflated ticker) and most deltas touch DEEP levels that change nothing
        actionable. Evaluating every one saturated the CPU (90%) and starved WS
        pong reads (periodic 1011 disconnects). A delta that leaves the top ask,
        its size, and the second level unchanged cannot change any edge decision —
        update the book, skip the eval."""
        prev = self.livebook.get(q.venue, q.market_id)
        self.livebook.update(q)
        if prev is not None and q.exchange_ts is not None:
            def _head(x):
                yl = getattr(x, "yes_ask_levels", None)
                nl = getattr(x, "no_ask_levels", None)
                return (x.yes_ask, x.yes_ask_size, x.no_ask, x.no_ask_size,
                        yl[1] if yl and len(yl) > 1 else None,
                        nl[1] if nl and len(nl) > 1 else None)
            if _head(prev) == _head(q):
                self._ws_counts[q.venue] = self._ws_counts.get(q.venue, 0) + 1
                return None
        self.note_feed_lag(q)
        if getattr(q, "state", None):
            self._market_state[(q.venue, q.market_id)] = q.state
        report = None
        keys = self._index.get((q.venue, q.market_id), ())
        if len(keys) > 1:
            keys = sorted(keys, key=lambda k: -self._local_edge(k))
        for key in keys:
            r = await self._act_on_pair(key)
            if r is not None:
                report = r
        return report

    async def reconcile_positions(self, snapshots) -> list:
        """Cross-venue naked-exposure backstop, run each refresh cycle from the venue
        account snapshots. A locked arb holds EQUAL contracts on its two legs (YES on one
        venue, NO on the other), so for every confirmed pair the leg sizes should match;
        an imbalance is naked directional exposure — a hedge that never landed (how the
        Ruzic maker fill became a total loss). To ride out a transient mid-trade/settlement
        blip this warns on first sight and only trips the kill switch when the SAME pair is
        still imbalanced on the next check. Positions on markets not in any active pair are
        logged for visibility (their hedge can't be auto-verified). Returns the imbalances.

        A leg whose market has SETTLED before its hedge (the recurring poly-settles-first
        case: poly EXPIRED -> position 0, kalshi leg still open) is a REALIZED arb awaiting
        the other venue's settlement, not a stranded leg — verified by live market state and
        excluded from the halt (genuine stranded legs are caught at execution time).
        """
        pos: dict[tuple[str, str], float] = {}
        venue_open: dict[str, int] = {}          # per-venue count of open positions (for down-detect)
        for snap in snapshots or []:
            venue = getattr(snap, "venue", None)
            venue_open.setdefault(venue, 0)
            for vp in getattr(snap, "positions", None) or []:
                if getattr(vp, "is_open", False) and abs(getattr(vp, "quantity", 0.0)) > 1e-9:
                    pos[(venue, vp.market_id)] = abs(vp.quantity)
                    venue_open[venue] += 1

        imbalanced, paired_markets = [], set()
        for p in self._pairs.values():
            paired_markets.add((p.venue_a, p.market_a))
            paired_markets.add((p.venue_b, p.market_b))
            qa = pos.get((p.venue_a, p.market_a), 0.0)
            qb = pos.get((p.venue_b, p.market_b), 0.0)
            if abs(qa - qb) > self._reconcile_tol:
                # Skip pairs still actively trading — their venue positions haven't settled
                # (a transient burst imbalance, not a stranded leg). Caught once trading stops.
                if self.clock() - self._last_acted.get(p.key, -1e9) < self._reconcile_grace:
                    continue
                imbalanced.append((p, qa, qb))

        # Held positions on markets the watchlist no longer pairs. The live watchlist
        # forgets a pair once it's pruned (settled/thin), but a held position still has a
        # knowable counterpart in the store's ACTED-opportunity history — verify against
        # that instead of just logging (the TPZRL naked leg sat exactly in this blind
        # spot: its pair left the watchlist, so the naked Kalshi leg was only ever an
        # INFO line). A counterpart holding a matching size = hedged; a FLAT counterpart
        # joins the imbalance flow below, where the venue-down and settled-leg filters
        # still apply before any warn/persist/halt. Blacklisted (quarantined) pairs are
        # deliberately excluded — quarantine records the loss precisely so the bot does
        # NOT freeze on that market.
        untracked = [(v, m, q) for (v, m), q in pos.items() if (v, m) not in paired_markets]
        if untracked:
            hist = {}
            blacklisted = set()
            if self.store is not None:
                try:
                    hist = self.store.acted_pair_map()
                    blacklisted = self.store.blacklisted_keys()
                except Exception as exc:
                    log.warning("reconcile: pair-history read failed: %s", exc)
            unverified = []
            seen_keys = {p.key for p, _, _ in imbalanced}
            for v, m, q in untracked:
                counter = hist.get((v, m))
                if counter is None:
                    unverified.append((v, m, q))
                    continue
                qc = pos.get(counter, 0.0)
                if abs(q - qc) <= self._reconcile_tol:
                    continue                                   # hedged on the counterpart
                p = ConfirmedPair(f"{v}:{m}|{counter[0]}:{counter[1]} (historical)",
                                  v, m, counter[0], counter[1])
                bkey = (self.store._pair_key(v, m, counter[0], counter[1])
                        if self.store is not None else None)
                if bkey in blacklisted or p.key in seen_keys:
                    continue
                if self.clock() - self._last_acted.get(p.key, -1e9) < self._reconcile_grace:
                    continue                                   # mid-burst, not stranded
                seen_keys.add(p.key)
                imbalanced.append((p, q, qc))
            if unverified:
                log.info("RECONCILE: %d held position(s) with no known counterpart "
                         "(verify manually): %s", len(unverified),
                         ", ".join(f"{v}:{m}={q:g}" for v, m, q in unverified[:8]))

        # Drop imbalances caused by a venue-DOWN read: if a venue returned ZERO open positions
        # (its API is down/erroring) yet >= _venue_down_min pairs hold their hedge on it, every
        # one of those looks naked at once — a stale read, not simultaneous hedge failures. Skip
        # them (don't halt); the next healthy cycle re-checks. A single-pair naked, or nakeds
        # split across both venues, still flows through to the real halt logic.
        if imbalanced:
            zero_side = {}                        # venue -> # pairs whose leg on it reads ~0
            for p, qa, qb in imbalanced:
                if qa <= self._reconcile_tol:
                    zero_side[p.venue_a] = zero_side.get(p.venue_a, 0) + 1
                if qb <= self._reconcile_tol:
                    zero_side[p.venue_b] = zero_side.get(p.venue_b, 0) + 1
            down = {v for v, n in zero_side.items()
                    if n >= self._venue_down_min and venue_open.get(v, 0) == 0}
            if down:
                before = len(imbalanced)
                imbalanced = [
                    (p, qa, qb) for p, qa, qb in imbalanced
                    if not ((qa <= self._reconcile_tol and p.venue_a in down)
                            or (qb <= self._reconcile_tol and p.venue_b in down))]
                log.warning("RECONCILE: %s returned 0 positions while %d pair(s) hold a hedge "
                            "on it — treating as venue DOWN/stale read, NOT naked; skipping halt "
                            "for those this cycle", ",".join(sorted(down)), before - len(imbalanced))

        # Drop imbalances that are CAPITAL-RECYCLE remnants: the deliberately-held cheap
        # OTM leg of an early-exited pair (an upset-hedge, not a stranded hedge). The
        # registry is persisted, so a restart can't misread one as naked. Without this
        # filter every recycle with an unsold OTM would trip a RECONCILE HALT.
        if imbalanced:
            remnants = getattr(self.executor, "recycled_remnants", {}) or {}
            if remnants:
                kept = []
                for p, qa, qb in imbalanced:
                    surviving = ((p.venue_a, p.market_a) if qa > qb
                                 else (p.venue_b, p.market_b))
                    qty = max(qa, qb) - min(qa, qb)
                    if qty <= remnants.get(surviving, 0.0) + self._reconcile_tol:
                        log.info("RECONCILE: %s imbalance (Δ%g) is an intentional "
                                 "upset-hedge remnant from a capital recycle — not naked",
                                 p.event_key, qty)
                        continue
                    kept.append((p, qa, qb))
                imbalanced = kept

        # Drop imbalances where a leg's market has SETTLED — a realized arb leftover, not a
        # stranded hedge (those halt at execution time). Checked only on imbalance (rare).
        if imbalanced:
            kept = []
            now = self.clock()
            for p, qa, qb in imbalanced:
                # STICKY: a settled leg cannot un-settle. The check reads venue APIs
                # that intermittently fail; without the cache a read blip flips the
                # classification back to "naked", two blips in a row trip the kill
                # switch (2026-07-07: halt-looped on a settled ARG-EGY leg for 15min).
                if now < self._settled_leg_until.get(p.key, 0.0):
                    continue
                if await self._pair_leg_settled(p, qa, qb):
                    self._settled_leg_until[p.key] = now + 1800.0
                    log.info("RECONCILE: %s imbalanced (%s=%g vs %s=%g) but a leg has SETTLED "
                             "— realized arb awaiting the other venue, not naked",
                             p.event_key, p.venue_a, qa, p.venue_b, qb)
                    continue
                kept.append((p, qa, qb))
            imbalanced = kept

        # BLACKLISTED-pair remnants: a false match we flattened may leave one
        # bid-less leg (KXUCL-27-INT: 15ct/$1.05 cost, empty futures book — even a
        # 1c IOC found no buyer). The pair can never re-trade (blacklist), the cost
        # is sunk, a long can't lose more — halting the slate protects nothing.
        if imbalanced and self.store is not None:
            try:
                bl = self.store.blacklisted_keys()
            except Exception:
                bl = set()
            kept = []
            for p, qa, qb in imbalanced:
                if p.key in bl:
                    log.warning("RECONCILE: unbalanced remnant on BLACKLISTED pair %s "
                                "(%g vs %g) — sunk long on a dead pair, not naked risk",
                                p.event_key, qa, qb)
                    continue
                kept.append((p, qa, qb))
            imbalanced = kept

        # DUST exemption: a held-long remnant is worth qty x its best bid — when that
        # is ~zero (no bid / pennies), there is nothing left to protect: the cost is
        # sunk and a long can't lose more. Halting the whole bot over a worthless
        # leftover (2026-07-07: $0-bid ET-scope remnants halt-looped for 30 min while
        # the game finished) protects nothing and costs the entire slate.
        if imbalanced and self.depth_fetch is not None:
            kept = []
            for p, qa, qb in imbalanced:
                sv, sm = ((p.venue_a, p.market_a) if qa > qb
                          else (p.venue_b, p.market_b))
                qty = abs(qa - qb)
                try:
                    q = await self.depth_fetch(sv, sm)
                except Exception:
                    q = None
                bid = None
                if q is not None:
                    # held side unknown here; take the HIGHER side bid (conservative:
                    # overstates value, understates the exemption)
                    asks = (getattr(q, "yes_ask", None), getattr(q, "no_ask", None))
                    bids = [1.0 - a for a in asks if a is not None]
                    bid = max(bids) if bids else None
                if bid is None:
                    kept.append((p, qa, qb))    # unreadable book: NOT provably dust
                    continue
                value = qty * bid
                if value <= 1.0:
                    log.info("RECONCILE: %s imbalance (Δ%g) is DUST (mark value $%.2f, "
                             "bid %s) — cost sunk, nothing to protect; settling out",
                             p.event_key, qty, value, f"{bid:.2f}" if bid else "none")
                    continue
                kept.append((p, qa, qb))
            imbalanced = kept

        if not imbalanced:
            self._imbalanced_prev = set()
            return []
        for p, qa, qb in imbalanced:
            log.warning("RECONCILE: NAKED exposure on %s — %s=%g vs %s=%g (Δ%g contracts)",
                        p.event_key, p.venue_a, qa, p.venue_b, qb, abs(qa - qb))
        keys = {p.key for p, _, _ in imbalanced}
        repeat = keys & self._imbalanced_prev
        self._imbalanced_prev = keys
        if repeat and self.reconcile_halt:
            risk = getattr(self.executor, "risk", None)
            trip = getattr(risk, "trip_kill_switch", None)
            if trip is not None and not getattr(risk, "is_killed", False):
                trip(f"reconcile: persistent naked exposure on {len(repeat)} pair(s)")
            log.critical("RECONCILE HALT: naked exposure persisted on %d pair(s) — stopping "
                         "until flat. Manually flatten the unhedged leg(s), then restart.",
                         len(repeat))
        return imbalanced

    async def _pair_leg_settled(self, p, qa, qb) -> bool:
        """True if either of a pair's legs has SETTLED — so a 0-vs-N imbalance is a realized
        arb leftover, not a stranded hedge. Three settled signals, in order:
          - open_check is False  (Kalshi reports finalized/settled in its status field,
            though state=None in the quote);
          - a non-OPEN quote state (Poly reports MARKET_STATE_EXPIRED);
          - the leg's POSITION is 0 AND its market is now UNREADABLE (404 after the venue
            pruned the resolved market). A REAL stranded leg's market is still OPEN and reads
            fine, so it isn't caught here; a transient unread clears on the next 30s check."""
        for venue, market, qty in (
                (p.venue_a, p.market_a, qa), (p.venue_b, p.market_b, qb)):
            checked, is_open, state = False, None, None
            if self.open_check is not None:
                checked = True
                try:
                    is_open = await self.open_check(venue, market)
                except Exception:
                    is_open = None
                if is_open is False:               # authoritatively settled/closed
                    return True
            if self.depth_fetch is not None:
                checked = True
                try:
                    q = await self.depth_fetch(venue, market)
                except Exception:
                    q = None
                state = getattr(q, "state", None) if q is not None else None
                if state is not None and state != self._OPEN_STATE:
                    return True
            # 0-position leg whose market is UNREADABLE despite an attempted read (404 after
            # the venue pruned the resolved market) -> settled+pruned. Only when we actually
            # checked: with no checker configured, fail toward halt (a human verifies).
            if (checked and abs(qty) <= self._reconcile_tol
                    and is_open is None and state is None):
                return True
        return False

    def _local_edge(self, key) -> float:
        """Current edge for a pair from livebook quotes alone — pure local math,
        used only for PRIORITIZATION (fattest edges claim scarce capital first)."""
        p = self._pairs.get(key)
        if p is None:
            return -1.0
        ev = self._eval_direction(self.livebook.get(p.venue_a, p.market_a),
                                  self.livebook.get(p.venue_b, p.market_b))
        return ev[0] if ev is not None else -1.0

    async def prime_and_sweep(self, only_markets=None):
        """Seed the live book with a REST snapshot of watchlist markets, then
        edge-check every pair once. Closes the gap where a venue's WS (Kalshi ticker)
        only emits on price *change*, so a stable-priced leg would otherwise never
        enter the book — leaving real edges undetected until the price happened to move.

        ``only_markets``: prime just this subset (the NEW markets of a resubscribe
        diff) — carried-over markets already hold live books, and full primes of a
        ~3k-market watchlist took ~6 minutes."""
        if self.depth_fetch is None:
            return
        items = list(self._index) if only_markets is None else [
            m for m in self._index if m in only_markets]
        if only_markets is not None and not items:
            log.info("prime: no new markets — sweeping existing books only")
        # Fetch the snapshots CONCURRENTLY (bounded) instead of one-at-a-time: a sequential
        # sweep of ~140 legs takes >a minute, which both delays picking up newly-listed
        # markets and widens the window where the book is half-primed. Concurrency collapses
        # it to seconds; the per-venue rate limiters still pace the actual HTTP.
        sem = asyncio.Semaphore(max(1, self.prime_concurrency))

        async def _one(venue, market):
            async with sem:
                try:
                    return venue, market, await self.depth_fetch(venue, market)
                except Exception:
                    return venue, market, None

        primed = 0
        for _venue, _market, q in await asyncio.gather(*(_one(v, m) for (v, m) in items)):
            if q is not None:
                self.livebook.update(q)
                primed += 1
        log.info("primed live book with %d/%d market snapshots", primed, len(items))
        # FATTEST FIRST: capital is the binding constraint — when several standing
        # edges exist at once, allocation order decides which get funded. Rank by
        # the local livebook edge (no I/O) and act descending; skip pairs showing
        # no local edge at all (they have nothing to fire — also makes the sweep
        # ~10x cheaper than blind-evaluating every pair).
        ranked = sorted(((self._local_edge(k), k) for k in list(self._pairs)),
                        key=lambda t: -t[0])
        for edge_hint, key in ranked:
            if edge_hint <= 0:
                break
            await self._act_on_pair(key)
            # YIELD between pairs: this sweep runs right after (re)subscribe, and
            # 400+ back-to-back ladder evals starved the fresh sockets' pong reads —
            # both venues 1011-closed at exactly ping_interval+ping_timeout after
            # every resubscribe. sleep(0) lets the loop service IO each iteration.
            await asyncio.sleep(0)

    async def _consume(self, venue) -> None:
        mids = self.market_ids.get(venue.name) or None
        if mids is None:
            return  # nothing confirmed for this venue this cycle
        async for q in venue.stream_order_book(mids):
            self._ws_counts[venue.name] = self._ws_counts.get(venue.name, 0) + 1
            await self.on_quote(q)

    async def run(self, venues: list, refresh_specs, *, refresh_interval: float = 300.0) -> None:
        """Slow/fast loop: refresh confirmed pairs each interval WHILE the fast
        consumers keep streaming, hot-swap the watchlist on completion, and only
        (re)subscribe the WebSockets when the market set actually changed.

        The consumers used to be CANCELLED for the whole discovery pass, so the fast
        path was dark for its entire duration (measured ~8 min/cycle before the
        embedding cache) — every edge appearing during a refresh was invisible. Now
        the only dark windows are the first pass at boot (no watchlist yet) and the
        brief resubscribe when subscriptions change.

        ``refresh_specs`` is an async callable returning ``list[ConfirmedPair]``.
        """
        consumers: list = []
        subscribed: dict[str, tuple] = {}

        prime_task: list = []
        prev_markets: set = set()

        async def _resubscribe() -> None:
            for c in consumers:
                c.cancel()
            await asyncio.gather(*consumers, return_exceptions=True)
            consumers.clear()
            for t in prime_task:
                t.cancel()                     # stale prime for the OLD market set
            prime_task.clear()
            # A consumer we just cancelled may have spawned an execution; the shielded
            # task survives. Settle every order before resubscribing — never stream a
            # new market set with an order ambiguously in flight.
            if self._inflight:
                log.warning("waiting for %d in-flight execution(s) to settle "
                            "before resubscribe", len(self._inflight))
                await self.drain()
            # STREAM FIRST: the livebook retains last-good quotes and the kalshi
            # book channel self-primes via snapshots on subscribe — tearing the
            # consumers down for the whole REST prime left the fast path DARK ~6
            # minutes per resubscribe (17x in 3h observed = dark a third of the
            # evening). Prime runs in the BACKGROUND, and only for the markets
            # NEW to this subscription (carried-over books are already live).
            consumers.extend(asyncio.create_task(self._consume(v)) for v in venues)
            new_markets = set(self._index) - prev_markets
            prev_markets.clear()
            prev_markets.update(self._index)
            prime_task.append(asyncio.create_task(
                self.prime_and_sweep(only_markets=new_markets)))

        try:
            while True:
                # Discovery/scan/match runs CONCURRENTLY with the consumers (which
                # keep trading the last-good watchlist while this completes).
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
                new_subs = {v: tuple(sorted(m)) for v, m in self.market_ids.items()}
                if new_subs != subscribed or not consumers or any(c.done() for c in consumers):
                    subscribed = new_subs
                    await _resubscribe()
                self._ws_counts = {}
                await asyncio.sleep(refresh_interval)
                # WS health: how many live ticks each venue delivered this interval. A
                # venue at 0 means its market WebSocket isn't feeding the fast path.
                counts = {v.name: self._ws_counts.get(v.name, 0) for v in venues}
                dead = [name for name, n in counts.items() if n == 0]
                if dead:
                    log.warning("WS health: %s — NO quotes this interval from %s", counts, dead)
                else:
                    log.info("WS health: %s quotes this interval | max fire latency "
                             "%.0fms", counts, getattr(self, "_fire_lat_max", 0.0))
                    self._fire_lat_max = 0.0
                await self.log_edge_snapshot()
        finally:
            for c in consumers:
                c.cancel()
            await asyncio.gather(*consumers, return_exceptions=True)
            if self._inflight:
                await self.drain()

    def edge_snapshot(self, top: int | None = None) -> list[tuple]:
        """Current best edge per pair, computed from the live WS book.

        Returns ``[(edge, pair, yes_quote, no_quote, size), ...]`` sorted best-first,
        only for pairs that have a two-sided quote (both legs present in the book).
        This is proof that WS prices are matched to the right events: an entry can only
        exist if both of a pair's markets have a current quote in the livebook.

        Sizes here may be 0 (the Kalshi ticker is sizeless and an empty book side shows
        as a ``1 - bid`` artifact). This is the cheap WS *candidate* ordering; the
        tradeable view is :meth:`log_edge_snapshot`, which re-confirms real depth."""
        rows = []
        for p in self._pairs.values():
            ev = self._best_direction(p)
            if ev is None:
                continue
            edge, yq, nq, size = ev
            rows.append((edge, p, yq, nq, size))
        rows.sort(key=lambda r: r[0], reverse=True)
        return rows[: top if top is not None else self.edge_snapshot_top]

    async def log_edge_snapshot(self, top: int | None = None) -> None:
        """Log the top *tradeable* edges: ranked by REAL order-book depth, not the
        sizeless/sentinel WS book. Every priced pair is re-confirmed against its actual
        book (bounded concurrency) and only genuinely two-sided pairs with fillable size
        (>= 1 contract) qualify — so empty-book ``1.00``-sentinel rows that can never fill
        no longer dominate the snapshot. Ranking is by the confirmed edge."""
        top = top if top is not None else self.edge_snapshot_top
        # FULL set of pairs with a two-sided WS quote — NOT capped at edge_snapshot_top
        # (that cap is only for how many rows we DISPLAY). This is the honest "priced"
        # denominator: how many watchlist pairs are actually streaming a two-sided book.
        priced = [(ev, p) for p in self._pairs.values()
                  if (ev := self._best_direction(p)) is not None]
        if not priced:
            log.info("edge snapshot: 0/%d pairs have two-sided WS quotes yet "
                     "(book still warming up?)", len(self._pairs))
            return
        priced.sort(key=lambda r: r[0][0], reverse=True)   # best WS edge first
        if self.depth_fetch is None:
            # No depth source to confirm against — fall back to the raw WS book (sizes
            # may be 0 / sentinel; this path is for tests/diagnostics, not live trading).
            shown = priced[:top]
            log.info("edge snapshot (WS book): %d/%d pairs two-sided, top %d:",
                     len(priced), len(self._pairs), len(shown))
            for (edge, yq, nq, size), p in shown:
                log.info("  %s yes=%.2f + %s no=%.2f = %.2f | edge=%+.3f sz=%g",
                         yq.venue, yq.yes_ask, nq.venue, nq.no_ask,
                         yq.yes_ask + nq.no_ask, edge, size)
            return
        # Confirm real depth on the best WS candidates first, bounded so a large watchlist
        # can't flood the rate limiter / starve trade-path depth fetches. Keep only
        # genuinely two-sided pairs with fillable size (>= 1 contract).
        cap = max(top * 6, 24)
        sem = asyncio.Semaphore(self.prime_concurrency)

        async def _confirm(p):
            async with sem:
                return await self._confirm_depth(p, quiet=True)

        evs = await asyncio.gather(*(_confirm(p) for _, p in priced[:cap]))
        rows = [ev for ev in evs if ev is not None and ev[3] >= 1]  # ev[3] = fillable size
        rows.sort(key=lambda r: r[0], reverse=True)
        shown = rows[:top]
        checked = min(len(priced), cap)
        lag = " | feed lag " + " ".join(
            f"{v}={l*1000:.0f}ms" for v, l in sorted(self._feed_lag.items())) \
            if self._feed_lag else ""
        # "N of M depth-sampled": this line SAMPLES the top pairs by WS edge and
        # depth-verifies just those — it is a diagnostic, not a trading limit (the
        # fire path evaluates EVERY pair on every tick).
        log.info("edge snapshot: %d/%d pairs two-sided on WS; depth-sampled top %d: "
                 "%d verified two-sided, top %d%s:",
                 len(priced), len(self._pairs), checked, len(rows), len(shown), lag)
        for edge, yq, nq, size in shown:
            log.info("  %s yes=%.2f + %s no=%.2f = %.2f | edge=%+.3f sz=%g",
                     yq.venue, yq.yes_ask, nq.venue, nq.no_ask,
                     yq.yes_ask + nq.no_ask, edge, size)
