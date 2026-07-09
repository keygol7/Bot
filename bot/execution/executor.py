"""Live two-leg arbitrage executor (LIVE_SMALL).

Turns a detected :class:`ArbOpportunity` into real orders, market-neutrally:

  1. Size the trade (min of available depth, per-order cap, and risk headroom).
  2. Place leg 1 (buy YES) as **fill-or-kill** — all-or-nothing, no partial.
     - KILLED/REJECTED  -> no position; abort cleanly.
     - ERROR (unknown)  -> HALT + kill switch (we can't tell if it filled).
  3. Place leg 2 (buy NO) as fill-or-kill.
     - FILLED            -> locked: $1 payout vs sub-$1 cost. Record profit.
     - KILLED/REJECTED   -> **auto-unwind leg 1** (sell it back, take the small loss).
     - ERROR/PARTIAL     -> HALT + kill switch (don't risk doubling up).

The "halt on unknown" rule is deliberate: auto-unwind only happens when leg 2
definitively did NOT fill. Any ambiguous state stops trading for human reconciliation
rather than guessing. Works for both cross-venue (two venues) and single-venue bundle
(same venue) arbs — the leg venues just happen to be equal in the bundle case.

This is the real-money path. It runs only when the runner is in a live mode AND the
venues have trading credentials. Validate against sandbox/demo first.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field, replace
from enum import Enum

from bot.data.store import Store
from bot.execution.orders import OrderResult, OrderStatus
from bot.execution.risk import RiskManager
from bot.fees import FeeModel, ZeroFeeModel, per_contract_fee
from bot.models import Side
from bot.strategies.arbitrage import ArbOpportunity, usable_depth
from bot.venues.base import RawMarket

log = logging.getLogger("bot.executor")


def _family(venue: str, market_id: str) -> tuple[str, str]:
    """Reliability FAMILY of a market — the venue-side grouping whose fill behavior is
    correlated: Kalshi's series token (``KXVALORANTGAME``), Polymarket's slug family
    (``aec-valorant``). Individual markets are ephemeral; families persist, so fill
    evidence aggregated here transfers to each new market in the family."""
    if not market_id:
        return (venue, "")
    if market_id[:2].isupper():                       # Kalshi ticker
        return (venue, market_id.split("-", 1)[0])
    parts = market_id.split("-")                      # Poly slug: first two segments
    return (venue, "-".join(parts[:2]))


def _reject_reason(result) -> str:
    """Human-readable why-it-failed from an OrderResult's raw payload (HTTP status +
    venue error body). A KILL usually carries NO venue detail (FOK simply could not
    fill at the limit — a book race), so fall back to the ORDER context we always
    have: side/limit/size, which is what makes the log line diagnosable."""
    raw = getattr(result, "raw", None) or {}
    parts = []
    if raw.get("http_status"):
        parts.append(f"HTTP {raw['http_status']}")
    body = raw.get("body") or raw.get("error")
    if body:
        parts.append(str(body)[:200])
    if not parts:
        side = getattr(getattr(result, "side", None), "value", "?")
        req = getattr(result, "requested", None) or 0
        act = getattr(result, "action", "?")
        return (f"no venue detail (FOK could not fill: {act} {side} x{req:g} "
                f"— book race)")
    return " ".join(parts)


class ExecStatus(str, Enum):
    SUCCESS = "SUCCESS"      # both legs filled — locked arb
    SKIPPED = "SKIPPED"      # didn't act (risk/size/leg-1 no fill)
    UNWOUND = "UNWOUND"      # leg 1 filled, leg 2 didn't, leg 1 sold back
    HALTED = "HALTED"        # ambiguous/unknown state — kill switch tripped
    QUARANTINED = "QUARANTINED"  # stuck leg recorded + market blacklisted; KEEP trading others


def scaled_hedge_buffer(buffer: float, depth: float, thin_depth: float,
                        deep_depth: float) -> float:
    """The hedge buffer to require for a given book ``depth``. The buffer is a price
    cushion that lets the second (hedge) leg fill through movement without unwinding — but
    a DEEP book fills at the touch and barely needs it, while a THIN book needs the full
    amount. Scale linearly from the full ``buffer`` at/below ``thin_depth`` to a one-tick
    floor at/above ``deep_depth``, so the bot can fire on the SMALLER (more frequent)
    divergence windows on deep markets while keeping full protection on thin ones.
    ``deep_depth <= 0`` disables scaling (always the full buffer)."""
    floor = min(buffer, 0.01)
    thin = max(thin_depth, 1.0)
    if deep_depth <= thin or depth <= thin:
        return buffer
    if depth >= deep_depth:
        return floor
    frac = (depth - thin) / (deep_depth - thin)
    return round(buffer - frac * (buffer - floor), 4)


@dataclass
class ExecutionReport:
    status: ExecStatus
    reason: str = ""
    legs: list[OrderResult] = field(default_factory=list)
    realized_pnl: float = 0.0

    def __str__(self) -> str:
        return f"[{self.status.value}] pnl={self.realized_pnl:+.2f} {self.reason}".strip()


class Executor:
    def __init__(
        self,
        venues: dict,
        risk: RiskManager,
        *,
        fee_models: dict[str, FeeModel] | None = None,
        store: Store | None = None,
        max_order_contracts: float = 2.0,
        unwind_slippage: float = 0.05,
        fill_confirmer=None,
        confirm_timeout: float = 5.0,
        balance_buffer: float = 0.99,
        min_venue_balance: float = 0.0,
        probe_contracts: float = 0.0,
        market_proven_fills: int = 3,
        market_max_fails: int = 2,
        market_ramp_factor: float = 3.0,
        scarcity_balance: float = 0.0,
        scarcity_min_edge: float = 0.02,
        rebalance_floor: float = 0.0,
        edge_full_budget: float = 0.0,
        edge_budget_floor: float = 0.25,
        fresh_hedge_secs: float = 0.0,
        recross_epsilon: float = 0.02,
        recycle_floor: float = 0.0,
        recycle_itm_bid: float = 0.90,
        recycle_max_cost: float = 0.03,
        recycle_max_contracts: float = 50.0,
        recycle_target: float = 0.0,
        recycle_cooldown: float = 300.0,
        recycle_pair_cooldown: float = 3600.0,
        recycle_max_settle_days: float = 0.0,
        recycle_decided_bid: float = 0.98,
        recycle_min_settle_hours: float = 0.0,
        early_exit_enabled: bool = False,
        early_exit_margin: float = 0.0,
        early_exit_cooldown: float = 300.0,
        early_exit_max_pairs: float = 8.0,
        early_exit_max_contracts: float = 50.0,
        early_exit_min_bid_depth: float = 0.0,
        early_exit_min_settle_days: float = 3.0,
        max_settle_days: float = 0.0,
        longdated_min_edge: float = 0.0,
        min_lock_edge: float | None = None,
        leg2_slippage_share: float = 0.6,
        min_leg_depth: float = 0.0,
        depth_safety: float = 1.0,
        hedge_depth_fraction: float = 1.0,
        first_venue: str = "kalshi",
        take_first_venue: str | None = None,
        hedge_buffer: float = 0.0,
        maker_timeout: float = 5.0,
        maker_improvement: float = 0.01,
        maker_arm_cushion: float = 0.0,
        maker_poll: float = 0.0,
        hedge_retries: int = 1,
        maker_dynamic: bool = False,
        buffer_deep_depth: float = 0.0,
    ) -> None:
        self.venues = venues
        self.risk = risk
        self.fee_models = fee_models or {}
        self.store = store
        # Per-order contract ceiling; <= 0 means "no ceiling" (size up to balances,
        # depth, and risk caps). Kept as an optional safety cap.
        self.max_order_contracts = max_order_contracts
        self.unwind_slippage = unwind_slippage
        # Min resting depth (contracts) required on BOTH legs before firing. Thin books
        # are where a leg rejects and the other can't be hedged/unwound — skipping them
        # means we should never have to unwind. 0 = disabled.
        self.min_leg_depth = min_leg_depth
        # Fraction of shown top-of-book depth to actually trade (headroom so a FOK still
        # fills if the book thins between the quote and the order). 1.0 = use it all.
        self.depth_safety = depth_safety
        # Deep-cushion fraction applied to the LIVE hedge book read (see _hedge_fillable):
        # commit the FOK to only this fraction of shown hedge depth so a partial vanish can't
        # reject it. Deep books are unaffected (fraction still exceeds the trade size).
        self.hedge_depth_fraction = hedge_depth_fraction
        # The rejection-prone venue (Kalshi: thinner books / FOK insufficient resting
        # volume) is placed FIRST, so if it rejects there's no other leg to unwind.
        self.first_venue = first_venue
        # Which venue fires FIRST in a two-FOK hybrid TAKE — independent of the maker-rest
        # venue (``first_venue``). Set this to the REJECTION-PRONE venue so its failure is a
        # clean skip, not an unwind: live data shows Polymarket 500s the hedge buy on thin
        # markets, so firing it first turns those 500s into free skips instead of Kalshi
        # unwinds. Defaults to ``first_venue`` (unchanged behavior) when not set.
        self.take_first_venue = take_first_venue or first_venue
        self.fill_confirmer = fill_confirmer       # optional FillTracker (private WS)
        self.confirm_timeout = confirm_timeout
        # Available cash per venue, seeded from the startup snapshot and decremented as
        # legs fill. Used to size each arb to the most the funded balances allow.
        self.balance_buffer = balance_buffer       # leave headroom for fees/slippage
        self.min_venue_balance = min_venue_balance  # don't trade a leg whose venue is below this
        # Empirical fill-reliability (probe-then-scale): a market is PROBED at probe_contracts
        # until its FOK orders have FILLED market_proven_fills times (proving real depth), then
        # scaled to full size; a market that KILL/REJECTs market_max_fails times without proving
        # is excluded. The live, learned replacement for the volume proxy. 0 probe = disabled.
        self.probe_contracts = probe_contracts
        self.market_proven_fills = market_proven_fills
        self.market_max_fails = market_max_fails
        # Proven markets scale to ramp_factor x their largest demonstrated fill (see
        # _reliability_cap) — fast geometric scale-up on real depth, bounded by what's proven.
        self.market_ramp_factor = market_ramp_factor
        # Capital-scarcity edge gate (see _scarce_skip): reserve a nearly-drained venue's
        # last cash for the fattest edges instead of FIFO-locking it into a 1c arb.
        self.scarcity_balance = scarcity_balance
        self.scarcity_min_edge = scarcity_min_edge
        # Venue auto-balancing (see _rebalance_skip): below this floor, stop firing the arbs
        # that drain a venue fastest so it self-levels instead of one-way draining to idle.
        self.rebalance_floor = rebalance_floor
        # Edge-weighted capital budget (see _max_size): the edge (per contract) at which a
        # fire may use the FULL spendable balance; thinner edges get a proportional share
        # (floored at edge_budget_floor) so small-pnl trades can't FIFO-lock the bankroll.
        self.edge_full_budget = edge_full_budget
        self.edge_budget_floor = edge_budget_floor
        # Fresh-hedge fast path: skip the hedge REST re-read when the opp's backing quotes
        # are younger than this AND the hedge leg's WS depth is >= 2x the trade size (the
        # only network round-trip on the fire path, ~40ms). 0 = always re-read.
        self.fresh_hedge_secs = fresh_hedge_secs
        # Breakeven recross (see _recross_or_unwind): how far past leg1's breakeven the
        # failed hedge may be re-taken before we accept the unwind's guaranteed loss.
        self.recross_epsilon = recross_epsilon
        # Capital recycler (auto-rebalance v2): hedged pairs are portable capital — a
        # decided pair's ITM leg can be sold at its bid to free the drained venue's cash
        # NOW instead of waiting days for settlement. See recycle_capital.
        self.recycle_floor = recycle_floor
        self.recycle_itm_bid = recycle_itm_bid
        self.recycle_max_cost = recycle_max_cost
        self.recycle_max_contracts = recycle_max_contracts
        self.recycle_target = recycle_target
        self.recycle_cooldown = recycle_cooldown
        self.recycle_pair_cooldown = recycle_pair_cooldown
        self._recycled_until: dict[tuple, float] = {}   # pair key -> rebuy-cooldown expiry
        self._last_recycle_ts: float = 0.0
        # Don't recycle a pair settling more than this far out UNLESS its ITM leg is
        # near-certain (>= recycle_decided_bid). Blocks recycling an UNDECIDED long-dated
        # favorite (a months-out election at 0.92) — its price isn't a settled outcome,
        # and the rebalance gate wants settlement, not an early sale, to replenish the
        # venue. Near-dated games (decided/imminent) recycle at any ITM bid. 0 = no limit.
        self.recycle_max_settle_days = recycle_max_settle_days
        self.recycle_decided_bid = recycle_decided_bid
        # Don't recycle pairs settling WITHIN this window: they pay the full $1 in
        # hours anyway, so an early exit only donates the give-up (1-3c/ct). The
        # recycler's value is freeing capital locked for DAYS. 0 = off.
        self.recycle_min_settle_hours = recycle_min_settle_hours
        # Early-profit exit: generalizes the recycler to realize a hedged pair's locked
        # profit BEFORE settlement whenever the two venues dislocate favorably (both
        # exit bids recover >= entry cost + margin). No drained-venue precondition.
        self.early_exit_enabled = early_exit_enabled
        self.early_exit_margin = early_exit_margin
        self.early_exit_cooldown = early_exit_cooldown
        self.early_exit_max_pairs = early_exit_max_pairs
        self.early_exit_max_contracts = early_exit_max_contracts
        self.early_exit_min_bid_depth = early_exit_min_bid_depth
        # Only early-exit pairs settling at least this far out — a near-dated pair pays
        # full value in hours, so paying exit fees to unwind it early is pure leakage;
        # early-exit's value is freeing capital locked for DAYS/months, not hours.
        self.early_exit_min_settle_days = early_exit_min_settle_days
        self._last_early_exit_ts: float = 0.0
        # Capital-horizon gate: reject entries settling beyond max_settle_days unless the
        # edge clears longdated_min_edge (a fat edge justifies the long capital lock).
        self.max_settle_days = max_settle_days
        self.longdated_min_edge = longdated_min_edge
        # Held OTM remnants of recycled pairs (free upset-hedges, held to settlement).
        # Persisted so a restart doesn't make the reconcile read them as naked exposure.
        self.recycled_remnants: dict[tuple[str, str], float] = (
            self.store.recycle_remnants() if self.store is not None else {})
        # Known pending settlement payouts per venue (part B): counted by the GATING
        # balances only — you can't spend a pending payout, so sizing stays on cash.
        self._pending: dict[str, float] = {}
        self._market_rel: dict[tuple, tuple] = (
            self.store.market_reliability() if self.store is not None else {})
        # FAMILY-level reliability (aggregated from the per-market table): markets are
        # ephemeral (a game market lives hours and sees a handful of edges), so per-market
        # learning never transfers — 67% of live fires were stuck at the 1-contract probe
        # while every other cap allowed 40+. A family (KXVALORANTGAME, aec-cs2, ...) with a
        # long fill history lends its NEW markets a real starting size; per-market streak
        # exclusion still bounds any individual phantom book.
        self._family_rel: dict[tuple, list] = {}    # (venue, family) -> [fills, fails, max_fill]
        # Size-aware reliability state (in-memory; a restart re-probes, which is safe):
        self._size_ceiling: dict[tuple, tuple] = {}   # key -> (max size, expires_at)
        self._excluded_until: dict[tuple, float] = {} # key -> phantom-cooldown expiry
        self.size_ceiling_ttl = 120.0                 # seconds a big-size reject caps size
        for (venue, market_id), (fills, fails, _streak, max_fill) in self._market_rel.items():
            fam = self._family_rel.setdefault(_family(venue, market_id), [0, 0, 0.0])
            fam[0] += fills
            fam[1] += fails
            fam[2] = max(fam[2], max_fill)
        self._balances: dict[str, float] = {}
        # Net position per (venue, market_id), signed (+ = net long YES). Seeded from the
        # startup snapshot, refreshed by the 30s balance poll, and updated on each settled
        # fill — so the churn guard (_churn_skip) reads it without a hot-path venue call.
        self._positions: dict[tuple[str, str], float] = {}
        # Bounded-aggressive limit pricing: how much edge to preserve as locked profit
        # (defaults to the risk min_edge floor).
        self.min_lock_edge = min_lock_edge
        self.leg2_slippage_share = min(max(leg2_slippage_share, 0.0), 1.0)
        # Price cushion reserved for the SECOND (hedge) leg so it fills through normal
        # book movement — and we ONLY fire when the edge can pay it AND still lock the
        # floor, so a thin edge that would just unwind never fires. 0 = no cushion.
        self.hedge_buffer = hedge_buffer
        # Maker mode: how long (s) a resting maker order may wait to fill before it
        # auto-expires (and we give up on that arb). The taker hedge fires the instant
        # the maker fills, so the naked window is ~one round-trip, not this whole time.
        self.maker_timeout = maker_timeout
        # How far INSIDE the ask to post the maker (>= one tick so post-only doesn't
        # reject it as crossing). Also captures this much extra edge on a fill.
        self.maker_improvement = maker_improvement
        # Edge required ABOVE the lock floor before RESTING a maker. The resting leg is
        # exposed to the taker drifting against it (adverse selection); without a cushion
        # a 1-2c edge fills into an adverse move and the forced hedge locks a loss. Mirrors
        # hedge_buffer on the taker path, but for the maker's drift-while-resting risk.
        self.maker_arm_cushion = maker_arm_cushion
        # ADVERSE-SELECTION calibration: per-(venue, family) EWMA of post-fill
        # hedge slip (planned hedge px vs achieved; ceiling shortfall when killed).
        # A resting maker is filled precisely when the market sweeps its price —
        # the family's observed slip is the true cost of being picked off there,
        # and it joins the ARM bar so hostile books price themselves out.
        self._maker_slip: dict = {}
        # While a maker rests, poll the taker leg every this-many seconds; if it drifts so
        # the hedge could no longer lock the floor, CANCEL the maker before it fills into
        # the adverse move. This is what makes arming THIN edges safe (the cushion is the
        # static guard at fire time; this is the dynamic guard while resting). 0 = disabled
        # (rest blindly until fill/expiry — only safe with a large arm cushion).
        self.maker_poll = maker_poll
        # Venue-wide outage state: names of venues currently unreachable (the
        # engine stops firing pairs touching them); recovery tasks re-hedge parked
        # naked legs when the venue returns.
        self.venue_down: set = set()
        self._recovery_tasks: list = []
        # Capital-yield floor: a lock must earn at least this fraction of its
        # capital PER DAY of holding (0 disables). Policy 2026-07-09: 1%/day.
        import os as _os
        self.min_daily_yield = float(_os.getenv("RISK_MIN_DAILY_YIELD", "0.01"))
        # Optional callable (venue_name, market_id) -> latest streamed MarketQuote;
        # wired by the streaming host so rest-window drift checks read the WS book.
        self.live_quote = None
        # After a maker fills, the forced hedge may hit a TRANSIENT venue error (e.g. a
        # Polymarket 500/timeout during a WS wobble). Re-place the hedge up to this many
        # times — but ONLY after reconciling the hedge venue to confirm nothing landed, so
        # a retry can never double up. Exhausting the retries falls through to the unwind.
        self.hedge_retries = max(0, int(hedge_retries))
        # Dynamic maker side: rest the maker on whichever leg is the liquidity bottleneck
        # (the THINNER book) and TAKE the deeper leg, instead of always resting on
        # first_venue. Lets a thin Polymarket leg be SOURCED via its own flow as a maker
        # rather than skipped because its book can't be taken. Requires the maker venue to
        # support post-only (both adapters do). False = always rest on first_venue.
        self.maker_dynamic = maker_dynamic
        # Depth at/above which a market is "deep" enough that the hedge fills at the touch,
        # so the hedge buffer scales down to one tick — letting the bot take the smaller
        # divergence windows on deep books. 0 = off (always the full hedge_buffer).
        self.buffer_deep_depth = buffer_deep_depth

    def _hedge_buffer_for(self, depth: float) -> float:
        """The depth-scaled hedge buffer for a fire of this size (see scaled_hedge_buffer)."""
        return scaled_hedge_buffer(
            self.hedge_buffer, depth, self.min_leg_depth, self.buffer_deep_depth)

    def set_balances(self, snapshots) -> None:
        """Seed available cash AND net positions per venue from account snapshots
        (startup + the 30s poll). Cached positions drive the churn guard with no hot-path
        venue read. Positions are rebuilt per venue each call so a settled/closed market
        drops out of the cache rather than lingering."""
        for snap in snapshots:
            bal = getattr(snap, "balance", None)
            if bal is not None:
                self._balances[snap.venue] = float(bal)
            positions = getattr(snap, "positions", None)
            if positions is not None:
                for key in [k for k in self._positions if k[0] == snap.venue]:
                    del self._positions[key]
                for pos in positions:
                    if getattr(pos, "is_open", False):
                        self._positions[(snap.venue, pos.market_id)] = float(pos.quantity)
                # Prune recycled-pair remnants whose position is gone (settled/paid).
                for key in [k for k in self.recycled_remnants if k[0] == snap.venue]:
                    if key not in self._positions:
                        self.recycled_remnants.pop(key, None)
                        if self.store is not None:
                            try:
                                self.store.clear_recycle_remnant(*key)
                            except Exception as exc:
                                log.warning("remnant clear failed for %s: %s", key, exc)

    def _track_fill(self, yes_venue, yes_market, no_venue, no_market, size) -> None:
        """Update the cached net position after a settled hedge: +size YES on the yes-leg,
        -size (i.e. +NO) on the no-leg. Keeps the churn guard accurate between 30s polls."""
        yk = (yes_venue, yes_market); nk = (no_venue, no_market)
        self._positions[yk] = self._positions.get(yk, 0.0) + size
        self._positions[nk] = self._positions.get(nk, 0.0) - size

    def _churn_skip(self, opp) -> str | None:
        """Block re-trading a pair in the OPPOSITE direction to the hedge we already hold.
        Buying the reverse legs opens no new arb: on Kalshi it nets against (cancels) the
        existing contracts and forfeits their premium (~$0.90/ct) — the dominant settled
        loss (e.g. OMETSA -$25 over a 27-contract flip). Reads the in-memory position cache
        (30s poll + per-fill updates) -> no hot-path venue read, latency-neutral. Flat opens
        and same-direction adds are allowed; only a clear reverse hedge (BOTH legs already
        opposite) is blocked, so a legitimate open is never falsely gated."""
        tol = 0.5
        yes_net = self._positions.get((opp.buy_yes_venue, opp.buy_yes_market), 0.0)
        no_net = self._positions.get((opp.buy_no_venue, opp.buy_no_market), 0.0)
        # This opp buys YES on the yes-leg (net +) and NO on the no-leg (net -). We already
        # hold the opposite hedge iff we're net-long NO on the yes-leg AND net-long YES on
        # the no-leg — firing would unwind it at a premium-forfeiting churn loss.
        if yes_net < -tol and no_net > tol:
            return (f"churn guard: pair already hedged the other way "
                    f"(yes-leg {opp.buy_yes_market} net {yes_net:+.0f}, "
                    f"no-leg {opp.buy_no_market} net {no_net:+.0f}) — reverse trade forfeits premium")
        return None

    def _balance(self, venue: str) -> float | None:
        return self._balances.get(venue)

    def set_pending(self, pending: dict) -> None:
        """Known pending settlement payouts per venue (settled-legs awaiting the venue's
        payout run). Counted by the GATING balances only."""
        self._pending = dict(pending or {})

    def _effective_balance(self, venue: str) -> float | None:
        """Cash + known pending payouts — what the venue is ABOUT to have. Used by the
        steering gates (_rebalance_skip/_scarce_skip) so they stop starving a venue
        that has cash hours away; real order sizing stays on spendable cash."""
        b = self._balances.get(venue)
        if b is None:
            return None
        return b + self._pending.get(venue, 0.0)

    def _spend(self, venue: str, amount: float) -> None:
        """Adjust tracked cash after a fill (negative ``amount`` credits it back)."""
        if venue in self._balances:
            self._balances[venue] = max(0.0, self._balances[venue] - amount)

    def _fee(self, venue: str) -> FeeModel:
        return self.fee_models.get(venue, ZeroFeeModel())

    @staticmethod
    def confirmer_avg(side, avg):
        """Convert a raw private-WS fill average into the ORDER-side price. Both
        venues' fill events quote the YES side; for a NO order the order-side cost is
        1 - yes_avg. Booking the raw value recorded phantom prices on kalshi NO maker
        fills (a NO resting at 0.72 reported 0.28 -> +$4.20 booked on a +$0.37 pair;
        the settlement audit of 2026-07-07 caught it)."""
        if avg is None:
            return None
        return round(1.0 - avg, 4) if side is Side.NO else avg

    async def _place(self, venue, market_id, side, action, price, contracts, tif) -> OrderResult:
        """Place an order, converting any raised exception into an ERROR result so a
        venue error never crashes the loop (it routes to the halt path instead)."""
        try:
            result = await venue.place_order(market_id, side, action, price, contracts, tif=tif)
        except Exception as exc:
            return OrderResult(
                getattr(venue, "name", "?"), market_id, side, action, contracts,
                status=OrderStatus.ERROR, raw={"error": str(exc)},
            )
        # Authoritative fill confirmation from the private WS, when available — avoids
        # depending on the synchronous REST response shape.
        if self.fill_confirmer is not None and result.order_id and result.status is not OrderStatus.ERROR:
            try:
                status, filled, avg = await self.fill_confirmer.confirm(
                    result.venue, result.order_id, contracts, self.confirm_timeout
                )
                # The synchronous REST result is authoritative for a terminal outcome.
                # Only let the stream OVERRIDE it to upgrade (e.g. REST KILLED -> a
                # confirmed FILL); never let a timed-out / non-terminal stream result
                # downgrade a terminal REST result into a spurious PARTIAL/KILLED.
                terminal = {OrderStatus.FILLED, OrderStatus.KILLED, OrderStatus.REJECTED}
                if status in terminal or result.status not in terminal:
                    result.status, result.filled = status, filled
                    # Confirmer price is raw venue convention (Polymarket = YES-side);
                    # prefer the per-venue-converted price from place_order.
                    if avg is not None and result.avg_price is None:
                        result.avg_price = self.confirmer_avg(side, avg)
            except Exception as exc:  # confirmer failure -> keep REST result
                log.warning("fill confirm failed for %s: %s", result.order_id, exc)
        # Empirical fill-reliability: a FOK BUY that FILLED proves the market's depth is real;
        # a KILL/REJECT/ERROR/partial proves it's phantom (the naked-leg source). Record per
        # market (not on unwinds — sells, which we only do to recover) to drive probe sizing.
        if action == "buy" and self.probe_contracts > 0:
            ok = (result.filled or 0) > 1e-9        # IOC partials are real fills
            self._record_market_reliability(
                getattr(venue, "name", "?"), market_id, ok,
                result.filled if ok else 0.0, attempted=contracts,
                rejected=result.status is OrderStatus.REJECTED)
        return result

    def _record_market_reliability(self, venue: str, market_id: str, ok: bool,
                                   fill_size: float = 0.0, attempted: float = 0.0,
                                   rejected: bool = False) -> None:
        """Size-aware, self-healing reliability.

        A reject is only PHANTOM evidence when it happened at PROBE size — a reject at 20
        contracts says "no 20 of depth right now", not "this book is fake" (live data:
        markets with 2-4 REAL fills were permanently excluded after two momentarily-thin
        rejects mid-game). So:
          - reject at size > probe  -> a TEMPORARY size ceiling (half the attempt, short
            TTL): retry smaller, no streak, no persisted fail;
          - reject at probe size    -> the phantom streak; crossing max_fails EXCLUDES for
            a cooldown (5 min, doubling per extra strike, capped 30 min) after which the
            market re-probes at 1 contract — never a permanent ratchet (the old cap=0
            forever meant the streak could never reset: 41 markets were dead-listed, some
            with real fill history, spamming futile reliability=0 skips);
          - any FILL              -> clears the ceiling and resets the streak.
        """
        key = (venue, market_id)
        fills, fails, streak, max_fill = self._market_rel.get(key, (0, 0, 0, 0.0))
        if ok:
            self._market_rel[key] = (fills + 1, fails, 0, max(max_fill, fill_size))
            self._size_ceiling.pop(key, None)
            self._excluded_until.pop(key, None)
            fam = self._family_rel.setdefault(_family(venue, market_id), [0, 0, 0.0])
            fam[0] += 1
            fam[2] = max(fam[2], fill_size)
            if self.store is not None:
                try:
                    self.store.record_market_outcome(venue, market_id, True, fill_size)
                except Exception as exc:
                    log.warning("market reliability write failed for %s: %s", market_id, exc)
            return
        if attempted > self.probe_contracts + 1e-9 and not rejected:
            # (a REJECTED order is a venue REFUSAL — size-independent; it skips the
            # depth-ceiling ladder and counts as a strike directly)
            # Not phantom evidence by itself — the book couldn't fill THIS size right now.
            # But the halving ladder must REMEMBER: without memory, the family base resets
            # the size after each TTL and a persistent phantom loops 20 -> 10 -> (reset) ->
            # 20 forever, never reaching probe size, never earning strikes. So a new reject
            # halves from the REMEMBERED ceiling (kept ~30 min beyond its capping TTL), and
            # once the ladder walks down TO probe size it converts into a phantom strike —
            # phantoms converge to the cooldown; a FILL clears everything.
            now = time.time()
            prev_c, prev_exp = self._size_ceiling.get(key, (None, 0.0))
            remembered = (prev_c is not None
                          and now - (prev_exp - self.size_ceiling_ttl) < 1800.0)
            ref = min(attempted, prev_c) if remembered else attempted
            ceiling = ref / 2.0
            if ceiling > self.probe_contracts + 1e-9:
                self._size_ceiling[key] = (ceiling, now + self.size_ceiling_ttl)
                log.info("reliability: %s:%s FOK reject at %g — size-capped to %g for %.0fs "
                         "(not phantom evidence)", venue, market_id, attempted, ceiling,
                         self.size_ceiling_ttl)
                return
            # Ladder exhausted: even near-probe sizes reject -> fall through and count it
            # as a phantom strike (streak/cooldown below).
            self._size_ceiling[key] = (float(self.probe_contracts),
                                       now + self.size_ceiling_ttl)
        # Probe-size reject: real phantom evidence. A venue REFUSAL escalates
        # double — market state (not-yet-open etc.) won't change within a cooldown,
        # so re-probing every 5 min just drips unwinds.
        streak += 2 if rejected else 1
        self._market_rel[key] = (fills, fails + 1, streak, max_fill)
        fam = self._family_rel.setdefault(_family(venue, market_id), [0, 0, 0.0])
        fam[1] += 1
        if self.store is not None:
            try:
                self.store.record_market_outcome(venue, market_id, False, 0.0)
            except Exception as exc:
                log.warning("market reliability write failed for %s: %s", market_id, exc)
        if streak >= self.market_max_fails:
            cooldown = min(300.0 * (2 ** (streak - self.market_max_fails)), 1800.0)
            self._excluded_until[key] = time.time() + cooldown
            log.info("reliability: %s:%s probe FOK failed (%d fills/%d fails/%d streak) — "
                     "excluded for %.0fs, then re-probes", venue, market_id,
                     fills, fails + 1, streak, cooldown)
        else:
            log.info("reliability: %s:%s probe FOK failed (%d fills/%d fails/%d streak) — "
                     "probe-gated until proven", venue, market_id, fills, fails + 1, streak)

    async def _venue_down(self, venue) -> bool:
        """Venue-WIDE outage probe: distinguishes 'this request failed' from 'the
        venue is in maintenance' (poly 2-4am EST 2026-07-09: a mid-flight hedge
        503d, its position read 503d, and the fail-closed halt killed the whole
        bot for a scheduled outage). A cheap public read failing too = outage."""
        try:
            ms = await venue.list_markets(limit=1)
            return not ms
        except Exception:
            return True

    def _schedule_hedge_recovery(self, opp, size, first, second,
                                 first_venue, second_venue, maker_leg) -> None:
        """The hedge venue is DOWN with a naked maker fill held. Park a recovery
        task: poll until the venue answers, then complete the hedge for the true
        imbalance via the existing _complete_hedge machinery. No kill switch —
        new fires are stopped separately by the engine's venue_down gate."""
        self.venue_down.add(getattr(second_venue, "name", second[0]))

        async def _recover():
            for _ in range(240):                      # up to 4 hours
                await asyncio.sleep(60.0)
                if not await self._venue_down(second_venue):
                    break
            else:
                self.risk.trip_kill_switch("hedge venue never recovered")
                return
            self.venue_down.discard(getattr(second_venue, "name", second[0]))
            log.warning("VENUE RECOVERED: %s — completing the parked hedge for %s",
                        getattr(second_venue, "name", second[0]), second[1])
            try:
                report = await self._complete_hedge_after_leg1_error(
                    opp, size, first, second, first_venue, second_venue, maker_leg)
                log.warning("parked hedge completion: %s %s", report.status,
                            report.reason)
            except Exception as exc:
                self.risk.trip_kill_switch(f"parked hedge completion failed: {exc}")

        self._recovery_tasks.append(asyncio.ensure_future(_recover()))

    async def _position_after_error(self, venue, market_id):
        """After an ambiguous leg ERROR, ask the venue whether a position actually
        resulted. Returns True (a position exists -> naked risk), False (flat -> the
        order didn't fill, safe to skip), or None (couldn't determine -> fail closed)."""
        snap_fn = getattr(venue, "account_snapshot", None)
        if snap_fn is None:
            return None
        try:
            snap = await snap_fn()
        except Exception as exc:
            log.warning("post-error reconcile failed for %s: %s", market_id, exc)
            return None
        for pos in getattr(snap, "positions", None) or []:
            if pos.market_id == market_id and pos.is_open:
                return True
        return False

    async def _hedge_qty_after_error(self, venue, market_id):
        """After an ambiguous leg-2 ERROR (e.g. a venue 500 on POST /orders), ask the
        venue HOW MANY contracts of the hedge actually landed. Returns the open quantity
        (>= 0.0) or None if it couldn't be determined (fail closed -> halt). Unlike
        ``_position_after_error`` this returns the size, so the executor can settle a fully
        hedged trade (the common '500 but it actually filled' case) instead of freezing."""
        snap_fn = getattr(venue, "account_snapshot", None)
        if snap_fn is None:
            return None
        try:
            snap = await snap_fn()
        except Exception as exc:
            log.warning("post-error hedge reconcile failed for %s: %s", market_id, exc)
            return None
        for pos in getattr(snap, "positions", None) or []:
            if pos.market_id == market_id and pos.is_open:
                return abs(pos.quantity)
        return 0.0

    async def _hedge_fillable(self, venue, leg, size, ceiling: float | None = None) -> float:
        """How many contracts the hedge (taker) leg would actually fill at its limit, read
        from the REAL order book — re-fetched live right before we commit leg 1. A book that
        thinned or moved since the depth-confirm then caps the trade (or skips it) instead of
        leaving a naked remainder. Returns the resting top-of-book size on the buy side when
        the current ask is at/through our limit; 0.0 when there's no offer, the price moved
        past our limit, or the fetch fails (an unconfirmable hedge must NOT arm a leg).

        Why the live book and not an order preview: Polymarket's /v1/order/preview does NOT
        simulate matching — it echoes the order with cumQuantity=0 for everything, so it
        always read as "fills nothing" and blocked every trade. The /book depth is the
        authoritative, verified source (``--show-book`` matches it exactly)."""
        market_id, side = leg[1], leg[2]
        try:
            q = await venue.fetch_quote(RawMarket(market_id=market_id, title="", raw={}))
        except Exception as exc:
            log.warning("hedge fillable: book fetch failed for %s: %s", market_id, exc)
            return 0.0                               # can't confirm -> don't arm
        ask = q.yes_ask if side is Side.YES else q.no_ask
        depth = q.yes_ask_size if side is Side.YES else q.no_ask_size
        if ask is None or not depth:
            return 0.0                               # no resting offer -> can't hedge
        # LADDER depth within the price band: the hedge IOC sweeps every level priced
        # <= its limit, so count the cumulative size there — top-of-book alone calls a
        # 1-at-top/300-behind book unhedgeable. ``ceiling`` (when the caller knows the
        # pair's breakeven band) bounds the count to levels that stay profitable.
        levels = q.yes_ask_levels if side is Side.YES else q.no_ask_levels
        if levels:
            band_max = ceiling if ceiling is not None else min(0.99, ask + self.hedge_buffer)
            band = usable_depth(levels, band_max)
            if band > depth:
                depth = band
        # The buy side's resting top-of-book depth. We don't gate on the stale leg limit
        # here: the taker leg fires FOK (a book that moved past the limit kills cleanly ->
        # unwind), and the maker leg REPRICES the hedge to the live ask on fill — so the
        # only thing this pre-check must establish is that real depth exists to hedge into.
        # DEEP CUSHION: commit the FOK to only a FRACTION of the shown depth so a partial
        # vanish in the ~tens of ms before the order lands can't reject it (the dominant
        # unwind cause). Deep books (depth >> size) keep firing full size — the caller caps
        # at min(size, this); only thin hedges size down. Latency-neutral (this read already
        # happens on every path).
        return float(depth) * self.hedge_depth_fraction

    def _max_size(self, opp: ArbOpportunity,
                  depth_override: float | None = None) -> tuple[int, dict[str, float]]:
        """Largest whole-contract size that fits every hard limit at once:

          * available order-book depth (can't fill more than is quoted) — or
            ``depth_override`` when the binding depth isn't ``max_contracts`` (maker mode
            sizes against the HEDGE leg, since the resting leg adds liquidity),
          * funded cash on each leg's venue (YES leg needs cash on the YES venue at
            ``yes_price``; NO leg needs cash on the NO venue at ``no_price``),
          * the per-market and total exposure risk caps,
          * the optional per-order contract ceiling.

        Returns ``(size, caps)`` where ``caps`` is each constraint's contract limit
        (for logging why a size was chosen).
        """
        label = f"{opp.buy_yes_venue}:{opp.buy_yes_market}"
        # Trade only a fraction of shown depth (headroom for a thinning book on FOK).
        depth = depth_override if depth_override is not None else float(opp.max_contracts)
        caps: dict[str, float] = {"depth": depth * self.depth_safety}

        yb, nb = self._balance(opp.buy_yes_venue), self._balance(opp.buy_no_venue)
        # Min-venue-balance guard: a venue too drained to reliably fund/hedge a leg must NOT
        # trade — else we fire the first (Polymarket) leg and the second (Kalshi) leg rejects
        # for insufficient_balance, leaving a naked position (the Kalshi-at-$0.38 incident).
        # Cap that leg to 0 -> size 0 -> clean skip. Self-healing: resumes when funds return.
        floor = self.min_venue_balance
        if yb is not None and opp.yes_price > 0:
            caps["cash_yes"] = 0.0 if (floor > 0 and yb < floor) else (yb * self.balance_buffer) / opp.yes_price
        if nb is not None and opp.no_price > 0:
            caps["cash_no"] = 0.0 if (floor > 0 and nb < floor) else (nb * self.balance_buffer) / opp.no_price

        gross = opp.gross_cost
        if gross > 0:
            lim = self.risk.limits
            caps["per_market"] = max(0.0, lim.max_position_per_market - self.risk.position(label)) / gross
            caps["total"] = max(0.0, lim.max_total_exposure - self.risk.total_exposure) / gross

        if self.max_order_contracts and self.max_order_contracts > 0:
            caps["order_cap"] = float(self.max_order_contracts)

        # Empirical fill-reliability (the live replacement for the volume proxy): cap each
        # leg by what its market's real FOK history has proven. Unproven markets are bounded
        # to a tiny probe; markets that have repeatedly failed without ever filling are
        # excluded. The min across both legs governs — a phantom on EITHER leg can leave a
        # naked position. proven markets are unbounded here (other caps apply).
        if self.probe_contracts > 0:
            caps["reliability"] = min(
                self._reliability_cap(opp.buy_yes_venue, opp.buy_yes_market),
                self._reliability_cap(opp.buy_no_venue, opp.buy_no_market),
            )

        # EDGE-WEIGHTED CAPITAL BUDGET: weight each fire's cash share by edge quality so
        # the bankroll isn't FIFO-locked into small-pnl trades — a thin edge may take at
        # most a fraction of spendable cash (still trades, just smaller); a fat edge takes
        # it all. Pure arithmetic (no latency). 0 disables.
        if self.edge_full_budget > 0 and gross > 0:
            frac = min(1.0, max(self.edge_budget_floor,
                                opp.edge_per_contract / self.edge_full_budget))
            spendable = min((b for b in (yb, nb) if b is not None), default=None)
            if spendable is not None:
                caps["edge_budget"] = (spendable * self.balance_buffer * frac) / gross

        return math.floor(max(0.0, min(caps.values()))), caps

    def _reliability_cap(self, venue: str, market_id: str) -> float:
        """Contract ceiling a market's empirical FOK history earns it. Unproven markets get
        a tiny probe; once PROVEN real, a market scales on the largest size it has actually
        FILLED (geometric, fast) rather than a slow per-fill count; a consecutive-fail streak
        (or repeated fails before proving) excludes."""
        fills, fails, streak, max_fill = self._market_rel.get(
            (venue, market_id), (0, 0, 0, 0.0))
        key = (venue, market_id)
        now = time.time()
        # Phantom exclusion is a COOLDOWN, never a ratchet: while it lasts the market is
        # out; when it expires the market re-probes at probe size (a fill then resets the
        # streak; another probe-reject doubles the next cooldown). The old permanent cap=0
        # could never recover — the streak could only reset on a fill that could never fire.
        if streak >= self.market_max_fails:
            if now < self._excluded_until.get(key, 0.0):
                return 0.0
            return float(self.probe_contracts)
        if fills < self.market_proven_fills:
            # FAMILY INHERITANCE: markets are ephemeral (hours), so per-market learning
            # never transfers and 67% of fires were stuck at the probe. A family with a
            # long, healthy fill history (>= 10 fills, <= 30% failure rate) lends its new
            # markets a real starting size — half its largest demonstrated fill. A failed
            # FOK at that size is a $0 clean reject that only sets a TEMPORARY size
            # ceiling (below); only probe-size rejects count phantom strikes.
            ffills, ffails, fmax = self._family_rel.get(
                _family(venue, market_id), (0, 0, 0.0))
            attempts = ffills + ffails
            if ffills >= 10 and attempts > 0 and ffails / attempts <= 0.30:
                base = max(float(self.probe_contracts), fmax * 0.5)
            else:
                base = float(self.probe_contracts)
        else:
            # Proven real -> scale on DEMONSTRATED depth: up to ramp_factor x the largest
            # size a FOK has actually filled here. The live deep-cushion and order cap
            # still bind each actual fire.
            base = max(float(self.probe_contracts), max_fill * self.market_ramp_factor)
        # Temporary size ceiling from a recent bigger-size reject: retry smaller until it
        # expires or a fill clears it.
        ceiling, until = self._size_ceiling.get(key, (0.0, 0.0))
        if now < until:
            base = min(base, ceiling)
        return base

    @staticmethod
    def _kalshi_start_ts(market_id: str):
        """Game start time embedded in kalshi per-game tickers
        (KXMLBTOTAL-26JUL091840ATHDET-... -> 2026-07-09 18:40 UTC), or None for
        dateless/futures/draft tickers."""
        import re as _re
        from datetime import datetime as _dt, timezone as _tz
        m = _re.search(r"-(\d{2}[A-Z]{3}\d{2})(\d{4})[A-Z]", market_id or "")
        if not m:
            return None
        try:
            return _dt.strptime(m.group(1) + m.group(2), "%y%b%d%H%M").replace(
                tzinfo=_tz.utc).timestamp()
        except ValueError:
            return None

    def _family_slip(self, venue_name: str, market_id: str) -> float:
        return self._maker_slip.get(_family(venue_name, market_id), 0.0)

    def _note_maker_slip(self, venue_name: str, market_id: str, slip: float) -> None:
        """EWMA (alpha .25) of hedge slip observed after maker fills, floored at 0."""
        fam = _family(venue_name, market_id)
        prev = self._maker_slip.get(fam, 0.0)
        ew = 0.75 * prev + 0.25 * max(0.0, slip)
        self._maker_slip[fam] = ew
        log.info("maker slip %s/%s: %+0.3f -> ewma %.3f", fam[0], fam[1], slip, ew)

    def _breakeven_ceiling(self, leg1_venue: str, p1: float, leg2_venue: str) -> float:
        """The max hedge price at which filling still beats unwinding: PnL-breakeven
        net of BOTH legs' fees, plus recross_epsilon. Shared by the preemptive leg-2
        limit and the recross fallback so the two can never drift. A FOK at this
        limit fills at the venue's RESTING prices (the whole book <= ceiling), so it
        costs nothing when the book hasn't moved and beats a -2-3c unwind when it
        has."""
        fees_ct = (per_contract_fee(self._fee(leg1_venue), p1)
                   + per_contract_fee(self._fee(leg2_venue), min(0.99, 1.0 - p1)))
        return min(0.99, max(0.01, round(
            (1.0 - p1) - fees_ct + self.recross_epsilon, 4)))

    def _leg_limits(self, opp: ArbOpportunity, first_side, second_side) -> tuple[float, float]:
        """Limit prices for the (first, second) legs that may pay worse than the quoted
        ask to fill on a moving book, but never enough to drop the locked profit below
        the floor.

        The buffer goes to the SECOND (hedge) leg: its failure forces an unwind, while a
        first-leg failure is a clean skip — so the hedge gets up to ``hedge_buffer`` of
        room to fill through adverse movement. Any leftover surplus (edge - floor -
        hedge) widens the first leg (improves its fill at no unwind cost). Worst case
        both fill at the limits -> combined cost = gross + surplus = 1 - floor, so the
        locked profit is >= floor by construction."""
        floor = self.min_lock_edge if self.min_lock_edge is not None else self.risk.limits.min_edge
        surplus = max(0.0, opp.edge_per_contract - floor)
        hedge = min(self._hedge_buffer_for(opp.max_contracts), surplus)
        leftover = surplus - hedge

        def px(side):
            return opp.yes_price if side is Side.YES else opp.no_price

        first_limit = min(0.99, round(px(first_side) + leftover, 4))
        second_limit = min(0.99, round(px(second_side) + hedge, 4))
        return first_limit, second_limit

    def _scarce_skip(self, opp: ArbOpportunity) -> str | None:
        """When a venue's spendable cash is below the scarcity floor, reserve it for the
        fattest edges — skip a thin edge so the last capital isn't FIFO-locked into a ~1%
        arb when a 2-3% one may follow. Returns a skip reason, or None to proceed."""
        if self.scarcity_balance <= 0 or opp.edge_per_contract >= self.scarcity_min_edge:
            return None
        bals = [b for b in (self._effective_balance(opp.buy_yes_venue),
                            self._effective_balance(opp.buy_no_venue)) if b is not None]
        if bals and min(bals) < self.scarcity_balance:
            return (f"capital scarce (${min(bals):.0f} < ${self.scarcity_balance:.0f}); "
                    f"reserving for edge >= {self.scarcity_min_edge:.2f} "
                    f"(this edge {opp.edge_per_contract:.3f})")
        return None

    def _rebalance_skip(self, opp: ArbOpportunity) -> str | None:
        """Keep a draining venue alive: when one venue's cash is below the rebalance floor,
        skip arbs where THAT venue would hold the LONGSHOT leg (the cheaper side).

        Settlement cash flows to the venue holding the WINNER — and price ~= win
        probability. The original gate did the OPPOSITE (kept only cheap legs on the
        drained venue to conserve cash-at-purchase), which measurably created a death
        spiral: over 48h live, Kalshi's book became 109 cheap legs vs 17 expensive, its
        held sides won only 16/52 settlements, and the venue's settlement cash flow ran
        spent $281 / received $83 (net -$198) while Poly collected the $1 payouts. Holding
        the FAVORITE costs more per contract now (the cash caps bound that) but the median
        settlement REPLENISHES the drained venue within hours instead of bleeding it."""
        if self.rebalance_floor <= 0:
            return None
        legs = ((opp.buy_yes_venue, opp.yes_price), (opp.buy_no_venue, opp.no_price))
        for i, (venue, leg_price) in enumerate(legs):
            other_venue, other_price = legs[1 - i]
            # EFFECTIVE balance (cash + known pending payouts): a venue with $200 of
            # settled-legs paying out in hours is not actually drained — steering more
            # favorites onto it would overshoot.
            bal = self._effective_balance(venue)
            if bal is None or bal >= self.rebalance_floor or leg_price >= other_price:
                continue                    # funded, or already holding the favorite
            # Only RESERVE if the COUNTERPART venue is funded (>= floor) — i.e. there's a
            # funded side to steer the longshot toward. If BOTH venues are below the floor
            # there's no rebalance target, so reserving would just deadlock the bot into
            # idle (the scarcity gate + min_venue_balance still guard genuine shortfalls).
            other_bal = self._effective_balance(other_venue)
            if other_bal is not None and other_bal < self.rebalance_floor:
                continue
            return (f"rebalance: {venue} low (${bal:.0f} < ${self.rebalance_floor:.0f}) and "
                    f"its leg is the LONGSHOT (${leg_price:.2f} vs ${other_price:.2f}) — "
                    f"settlements must replenish it, so it only holds favorites now")
        return None

    # ------------------------------------------------------------------
    # Capital recycler (auto-rebalance v2). Cross-venue cash transfer cannot be
    # automated (separate regulated exchanges), but hedged pairs ARE portable capital:
    # a locked pair (+YES@A/+NO@B = guaranteed $1xn at settlement) whose event is
    # effectively DECIDED can be early-exited — sell the ITM leg at its bid FIRST
    # (recovers ~0.9xn on the drained venue NOW), then try the cheap OTM leg (pennies;
    # unsold = a held upset-hedge remnant, so the ordering is bounded: worst case
    # realizes give-up cents below settlement, with upside on an upset). NOTE: the
    # bound assumes a TRUE complement pair — a false match makes it unbounded, which
    # is gated upstream by the matching truth stack.
    # ------------------------------------------------------------------

    @staticmethod
    def _rpair_key(va: str, ma: str, vb: str, mb: str) -> tuple:
        """Order-independent pair key (mirrors Store._pair_key: sorted 2-tuples)."""
        return tuple(sorted([(va, ma), (vb, mb)]))

    def _recycle_skip(self, opp) -> str | None:
        """Rebuy-churn guard: a just-recycled pair at ~0.92/0.05 sums to ~0.97 and looks
        like a 3c edge — without this the engine re-buys exactly what the recycler just
        sold, burning fees in a loop."""
        key = self._rpair_key(opp.buy_yes_venue, opp.buy_yes_market,
                              opp.buy_no_venue, opp.buy_no_market)
        if time.time() < self._recycled_until.get(key, 0.0):
            return "pair was capital-recycled recently — rebuy blocked (fee-churn guard)"
        return None

    def _horizon_skip(self, opp) -> str | None:
        """CAPITAL-YIELD gate (policy 2026-07-09): capital may not be locked at under
        ``min_daily_yield`` (1%/day) — the edge must pay for every day it is held:
        edge >= days_to_settle x 1%. A 1c edge justifies a 1-day hold, 5c buys five
        days; a November election needs an edge no real arb has (the $112 CA-gov
        lock earned 0.015%/day for 4 months). Same-day settles always pass."""
        if not opp.settle_ts:
            return None
        days = max(0.0, (opp.settle_ts - time.time()) / 86400.0)
        if days <= 1.0 or self.min_daily_yield <= 0:
            legacy = None
        else:
            need = days * self.min_daily_yield
            if opp.edge_per_contract < need - 1e-9:
                return (f"horizon: settles in {days:.1f}d and edge "
                        f"{opp.edge_per_contract:.3f} < {need:.3f} needed for "
                        f"{self.min_daily_yield:.1%}/day capital yield")
            legacy = None
        if self.max_settle_days > 0 and days > self.max_settle_days \
                and opp.edge_per_contract < self.longdated_min_edge:
            return (f"horizon: settles in {days:.0f}d (> {self.max_settle_days:.0f}d) and "
                    f"edge {opp.edge_per_contract:.3f} < long-dated min "
                    f"{self.longdated_min_edge:.3f} — capital better used near-dated")
        return legacy

    def recycle_trigger(self) -> tuple[str, str] | None:
        """(drained, funded) when the recycler should act: one venue's REAL cash below
        the recycle floor while the other holds >= 3x its cash AND is itself funded.
        Real cash, not effective — pending payouts don't pay today's hedges."""
        if self.recycle_floor <= 0 or len(self._balances) < 2:
            return None
        items = sorted(self._balances.items(), key=lambda kv: kv[1])
        (drained, dbal), (funded, fbal) = items[0], items[-1]
        if (dbal < self.recycle_floor and fbal >= max(self.rebalance_floor, 1.0)
                and fbal >= 3.0 * max(dbal, 1e-9)):
            return drained, funded
        return None

    def plan_recycle(self, drained: str, pair_map: dict, quotes: dict, *,
                     busy: frozenset = frozenset(),
                     settled_counterparts: frozenset = frozenset()) -> list:
        """PURE selection: which held ITM legs on ``drained`` to early-exit, and how.

        ``quotes``: (venue, market) -> MarketQuote (bids derived as in _unwind:
        YES bid = 1 - no_ask, NO bid = 1 - yes_ask). ``settled_counterparts``: markets
        whose counterpart was AUTHORITATIVELY confirmed settled (open_check False) —
        those become solo realizations. A flat counterpart NOT confirmed settled is
        skipped entirely (a possible naked leg belongs to the reconcile, not us).
        Returns dicts sorted cheapest-give-up first, truncated to the per-pass contract
        budget and to the projected cash target."""
        actions = []
        target = self.recycle_target if self.recycle_target > 0 else 2 * self.recycle_floor
        projected = self._balances.get(drained, 0.0)
        for (venue, market), net in self._positions.items():
            if venue != drained or abs(net) < 1.0 or (venue, market) in busy:
                continue
            held_side = Side.YES if net > 0 else Side.NO
            q = quotes.get((venue, market))
            if q is None:
                continue
            itm_bid = None
            if held_side is Side.YES and q.no_ask is not None:
                itm_bid = round(1.0 - q.no_ask, 4)
            elif held_side is Side.NO and q.yes_ask is not None:
                itm_bid = round(1.0 - q.yes_ask, 4)
            if itm_bid is None or itm_bid < self.recycle_itm_bid:
                continue
            # Horizon guard: a far-dated pair's ITM bid is a pre-decision favorite price,
            # not a settled outcome — don't sell it early (churn vs the rebalance gate's
            # settlement plan). Recycle it only if near-certain (>= decided bid). Near-
            # dated pairs (decided/imminent) always pass.
            if self.recycle_max_settle_days > 0 or self.recycle_min_settle_hours > 0:
                close = q.close_time
                if not close:
                    # close_time is None for many markets (the CA-gov class); the id
                    # usually carries the date — same fallback as the horizon gate.
                    from bot.strategies.arbitrage import settle_ts_from_id
                    _c = pair_map.get((venue, market))
                    close = (settle_ts_from_id(market)
                             or (settle_ts_from_id(_c[1]) if _c else 0.0) or 0.0)
                # Settles WITHIN the min window (or date unknown -> can't prove it
                # doesn't): ride to settlement, never pay a give-up for hours.
                if self.recycle_min_settle_hours > 0:
                    if not close or (close - time.time()) < self.recycle_min_settle_hours * 3600.0:
                        continue
                # Far-dated pre-decision favorite: leave it (rebalance gate wants the
                # settlement flow); recycle far-dated only when near-certain.
                if (self.recycle_max_settle_days > 0 and itm_bid < self.recycle_decided_bid
                        and close and (close - time.time()) > self.recycle_max_settle_days * 86400.0):
                    continue
            counter = pair_map.get((venue, market))
            cnet = self._positions.get(counter, 0.0) if counter else 0.0
            solo = False
            otm = None                       # (venue, market, side, bid)
            if counter is None or abs(cnet) < 0.5:
                if counter is not None and counter in settled_counterparts:
                    solo = True              # counterpart settled&paid -> pure realization
                else:
                    continue                 # possible naked leg -> reconcile's job
            else:
                if (cnet > 0) == (net > 0) or abs(abs(net) - abs(cnet)) > 0.5:
                    continue                 # not a clean opposite hedge
                cq = quotes.get(counter)
                otm_side = Side.YES if cnet > 0 else Side.NO
                otm_bid = None
                if cq is not None:
                    if otm_side is Side.YES and cq.no_ask is not None:
                        otm_bid = round(1.0 - cq.no_ask, 4)
                    elif otm_side is Side.NO and cq.yes_ask is not None:
                        otm_bid = round(1.0 - cq.yes_ask, 4)
                otm = (counter[0], counter[1], otm_side,
                       otm_bid if otm_bid is not None and otm_bid >= 0.01 else None)
            qty = float(int(abs(net)))
            fees_ct = per_contract_fee(self._fee(venue), itm_bid)
            otm_bid_val = otm[3] if (otm and otm[3]) else 0.0
            if otm and otm[3]:
                fees_ct += per_contract_fee(self._fee(otm[0]), otm[3])
            give_up = round((1.0 - itm_bid - otm_bid_val) + fees_ct, 4)
            if give_up > self.recycle_max_cost + 1e-9:
                log.info("recycle: %s:%s ITM bid %.2f rejected — give-up %.3f/ct > cap %.3f",
                         venue, market, itm_bid, give_up, self.recycle_max_cost)
                continue
            actions.append({
                "event": f"{venue}:{market}" + (f"|{otm[0]}:{otm[1]}" if otm else " (solo)"),
                "itm": (venue, market, held_side, itm_bid), "otm": otm,
                "qty": qty, "give_up_ct": give_up, "solo": solo,
            })
        actions.sort(key=lambda a: a["give_up_ct"])
        out, budget = [], self.recycle_max_contracts
        for a in actions:
            if budget < 1.0 or projected >= target:
                break
            a["qty"] = min(a["qty"], float(int(budget)))
            if a["qty"] < 1.0:
                continue
            budget -= a["qty"]
            projected += a["qty"] * a["itm"][3]
            out.append(a)
        return out

    async def recycle_capital(self, *, pair_map: dict, quote_fetch, busy=frozenset(),
                              open_check=None, snapshot_fn=None) -> list:
        """Loop entrypoint (off the hot path). Returns reports of executed actions."""
        if self.risk.is_killed or self.recycle_floor <= 0:
            return []
        now = time.time()
        if now - self._last_recycle_ts < self.recycle_cooldown:
            return []
        trig = self.recycle_trigger()
        if trig is None:
            return []
        drained, funded = trig
        # Insurance re-read: a stale/mis-signed position cache could make us SELL
        # inventory we don't hold (opening real exposure). One REST call, off hot path.
        if snapshot_fn is not None:
            try:
                snap = await snapshot_fn(drained)
                if snap is not None:
                    self.set_balances([snap])
            except Exception as exc:
                log.warning("recycle: drained-venue snapshot failed (%s) — skipping pass", exc)
                return []
        # Candidate quotes: bounded reads for drained-venue holds + their counterparts.
        candidates = [(v, m) for (v, m), n in self._positions.items()
                      if v == drained and abs(n) >= 1.0 and (v, m) not in busy][:10]
        quotes: dict = {}
        settled: set = set()
        for key in candidates:
            try:
                quotes[key] = await quote_fetch(*key)
            except Exception:
                continue
            counter = pair_map.get(key)
            if counter is None:
                continue
            try:
                quotes[counter] = await quote_fetch(*counter)
            except Exception:
                quotes[counter] = None
            if abs(self._positions.get(counter, 0.0)) < 0.5 and open_check is not None:
                try:
                    if (await open_check(*counter)) is False:   # authoritative ONLY
                        settled.add(counter)
                except Exception:
                    pass
        actions = self.plan_recycle(drained, pair_map, quotes, busy=busy,
                                    settled_counterparts=frozenset(settled))
        reports = []
        for a in actions:
            r = await self._recycle_one(a)
            if r is not None:
                reports.append(r)
        if reports:
            self._last_recycle_ts = now
        return reports

    # ------------------------------------------------------------------
    # Early-profit exit — generalizes the recycler. The recycler exits when one leg is
    # ITM (~$1, the event decided) to free a DRAINED venue's capital. This exits ANY
    # held hedged pair, drained or not, when the market offers a profitable early
    # unwind: the two legs sit on DIFFERENT venues, so when the books dislocate the
    # other way (both exit bids rise on news) their sum can exceed the entry cost —
    # realizing the locked profit months before settlement. It NEVER exits below entry
    # (the trigger enforces >= entry_cost + margin), so a quiet pair simply stays held.
    # Reuses _recycle_one for the sell-both plumbing + accounting (delta-vs-$1, which
    # equals exit_value - entry_cost once the entry-time lock is included).
    # ------------------------------------------------------------------

    def plan_early_exit(self, pair_map: dict, quotes: dict, entry_costs: dict, *,
                        busy: frozenset = frozenset()) -> list:
        """PURE: which held hedged pairs to unwind early at a profit-vs-entry. Each pair
        is considered once (dedup by pair key). ``entry_costs``: pair key ->
        (yes_price, no_price) sum = what we paid. Bids derived as in plan_recycle."""
        actions, seen = [], set()
        for (venue, market), net in list(self._positions.items()):
            if abs(net) < 1.0 or (venue, market) in busy:
                continue
            counter = pair_map.get((venue, market))
            if counter is None or counter in busy:
                continue
            cnet = self._positions.get(counter, 0.0)
            # clean opposite hedge only
            if abs(cnet) < 1.0 or (cnet > 0) == (net > 0) or abs(abs(net) - abs(cnet)) > 0.5:
                continue
            key = self._rpair_key(venue, market, counter[0], counter[1])
            if key in seen:
                continue
            seen.add(key)
            entry = entry_costs.get(key)
            if entry is None:
                continue                        # no cost basis -> can't judge profit
            q, cq = quotes.get((venue, market)), quotes.get(counter)
            if q is None or cq is None:
                continue
            # Settlement-horizon gate: only unwind pairs LOCKED for a while. A pair
            # settling in hours pays full value imminently — paying exit fees to beat
            # that is leakage. Require a KNOWN close_time far enough out (unknown = skip,
            # since we can't confirm the fee is worth it).
            if self.early_exit_min_settle_days > 0:
                closes = [t for t in (q.close_time, cq.close_time) if t]
                nearest = min(closes) if closes else None
                if nearest is None or (nearest - time.time()) < (
                        self.early_exit_min_settle_days * 86400.0):
                    continue

            def _bid(quote, held):               # held side's sellable bid + its depth
                if held is Side.YES and quote.no_ask is not None:
                    return round(1.0 - quote.no_ask, 4), (quote.no_ask_size or 0.0)
                if held is Side.NO and quote.yes_ask is not None:
                    return round(1.0 - quote.yes_ask, 4), (quote.yes_ask_size or 0.0)
                return None, 0.0

            side = Side.YES if net > 0 else Side.NO
            cside = Side.YES if cnet > 0 else Side.NO
            bid, depth = _bid(q, side)
            cbid, cdepth = _bid(cq, cside)
            if bid is None or cbid is None:
                continue
            if depth < self.early_exit_min_bid_depth or cdepth < self.early_exit_min_bid_depth:
                continue                         # thin exit -> partial-fill/naked risk
            qty = float(int(min(abs(net), abs(cnet))))
            if qty < 1.0:
                continue
            exit_fees = (per_contract_fee(self._fee(venue), bid)
                         + per_contract_fee(self._fee(counter[0]), cbid))
            exit_value = round(bid + cbid - exit_fees, 4)
            entry_cost = entry[0] + entry[1]
            required = entry_cost + self.early_exit_margin
            if self.min_daily_yield > 0 and entry_cost > 0:
                closes = [t for t in (q.close_time, cq.close_time) if t]
                if closes:
                    days = max(1.0, (min(closes) - time.time()) / 86400.0)
                    rem_rate = (1.0 - entry_cost) / (entry_cost * days)
                    if rem_rate < self.min_daily_yield:
                        # CAPITAL-YIELD policy: this lock earns under 1%/day for the
                        # rest of its life — freeing the capital is worth up to one
                        # day's yield as a haircut (redeployment repays it in a day;
                        # the $112 CA-gov lock sat 4 months at 0.015%/day).
                        required = entry_cost * (1.0 - self.min_daily_yield)
            if exit_value < required - 1e-9:
                continue                         # market isn't offering an acceptable exit
            # itm = higher-bid leg (sold first); otm = cheaper leg (a partial leaves the
            # LEAST capital exposed as the remnant).
            legs = sorted([(bid, venue, market, side), (cbid, counter[0], counter[1], cside)],
                          reverse=True)
            (hi_bid, hv, hm, hs), (lo_bid, lv, lm, ls) = legs
            actions.append({
                "event": f"{hv}:{hm}|{lv}:{lm} (early-exit +{exit_value - entry_cost:.3f})",
                "itm": (hv, hm, hs, hi_bid),
                "otm": (lv, lm, ls, lo_bid if lo_bid >= 0.01 else None),
                "qty": qty, "gain": round(exit_value - entry_cost, 4), "solo": False,
            })
        actions.sort(key=lambda a: -a["gain"])   # bank the biggest gains first
        out, budget = [], self.early_exit_max_contracts
        for a in actions:
            if budget < 1.0:
                break
            a["qty"] = min(a["qty"], float(int(budget)))
            if a["qty"] < 1.0:
                continue
            budget -= a["qty"]
            out.append(a)
        return out

    async def early_exit(self, *, pair_map: dict, quote_fetch, store, busy=frozenset(),
                         snapshot_fn=None) -> list:
        """Loop entrypoint (off the hot path). Scans ALL held hedged pairs for a
        profitable early unwind. Returns reports of executed exits."""
        if self.risk.is_killed or not self.early_exit_enabled:
            return []
        now = time.time()
        if now - self._last_early_exit_ts < self.early_exit_cooldown:
            return []
        # Candidate pairs: held markets with a known held counterpart, deduped, bounded.
        cands, seen = [], set()
        for (v, m), n in list(self._positions.items()):
            if abs(n) < 1.0 or (v, m) in busy:
                continue
            counter = pair_map.get((v, m))
            if counter is None or counter in busy or abs(self._positions.get(counter, 0.0)) < 1.0:
                continue
            key = self._rpair_key(v, m, counter[0], counter[1])
            if key in seen:
                continue
            seen.add(key)
            cands.append(((v, m), counter))
            if len(cands) >= int(self.early_exit_max_pairs):
                break
        if not cands:
            return []
        quotes, entry_costs = {}, {}
        for leg, counter in cands:
            for key in (leg, counter):
                if key not in quotes:
                    try:
                        quotes[key] = await quote_fetch(*key)
                    except Exception:
                        quotes[key] = None
            pk = self._rpair_key(leg[0], leg[1], counter[0], counter[1])
            try:
                ec = store.entry_cost_for_pair(leg[0], leg[1], counter[0], counter[1])
            except Exception:
                ec = None
            if ec is not None:
                entry_costs[pk] = ec
        actions = self.plan_early_exit(pair_map, quotes, entry_costs, busy=busy)
        reports = []
        for a in actions:
            r = await self._recycle_one(a)
            if r is not None:
                reports.append(r)
        if reports:
            self._last_early_exit_ts = now
        return reports

    async def _recycle_one(self, a: dict):
        """Sell ITM first (bounded ordering), then match the OTM; book only the DELTA
        vs the $1xn the lock already assumed at entry."""
        iv, im, iside, ibid = a["itm"]
        venue = self.venues.get(iv)
        if venue is None:
            return None
        sell = await self._place(venue, im, iside, "sell", ibid, a["qty"],
                                 "immediate_or_cancel")
        log.info("recycle ITM sell %s", sell)
        if sell.status is OrderStatus.ERROR:
            # Ambiguous but SAFE either way (sold = cash; unsold = still hedged).
            # Book nothing; the 30s poll resyncs. Cooldown so we don't hammer.
            self._last_recycle_ts = time.time()
            return None
        k = sell.filled
        if k <= 1e-9:
            return None                          # book moved; retry next pass
        itm_avg = sell.avg_price if sell.avg_price is not None else ibid
        sign = 1.0 if iside is Side.YES else -1.0
        self._spend(iv, -(k * itm_avg))          # credit the drained venue NOW
        self._positions[(iv, im)] = self._positions.get((iv, im), 0.0) - k * sign
        if self.store is not None:
            self.store.record_fill(iv, im, f"{iside.value}_SELL", itm_avg, k)
        j, otm_avg = 0.0, 0.0
        if a["otm"] is not None:
            ov, om, oside, obid = a["otm"]
            oven = self.venues.get(ov)
            if oven is not None and obid is not None:
                osell = await self._place(oven, om, oside, "sell", obid, k,
                                          "immediate_or_cancel")
                log.info("recycle OTM sell %s", osell)
                if osell.filled > 1e-9 and osell.status is not OrderStatus.ERROR:
                    j = osell.filled
                    otm_avg = osell.avg_price if osell.avg_price is not None else obid
                    osign = 1.0 if oside is Side.YES else -1.0
                    self._spend(ov, -(j * otm_avg))
                    self._positions[(ov, om)] = (
                        self._positions.get((ov, om), 0.0) - j * osign)
                    if self.store is not None:
                        self.store.record_fill(ov, om, f"{oside.value}_SELL", otm_avg, j)
            remnant = round(k - j, 6)
            if remnant > 1e-9:
                self.recycled_remnants[(ov, om)] = (
                    self.recycled_remnants.get((ov, om), 0.0) + remnant)
                if self.store is not None:
                    try:
                        self.store.record_recycle_remnant(
                            ov, om, self.recycled_remnants[(ov, om)])
                    except Exception as exc:
                        log.warning("remnant persist failed for %s: %s", om, exc)
                log.warning("recycle: holding %g OTM remnant on %s:%s as upset-hedge "
                            "(unbooked windfall if it wins)", remnant, ov, om)
            self._recycled_until[self._rpair_key(iv, im, ov, om)] = (
                time.time() + self.recycle_pair_cooldown)
        # Delta vs the $1xk the entry-time lock assumed: realized k*itm + j*otm instead.
        fees = self._fee(iv).fee(itm_avg, k)
        if j > 1e-9:
            fees += self._fee(a["otm"][0]).fee(otm_avg, j)
        delta = round(k * itm_avg + j * otm_avg - k * 1.0 - fees, 6)
        self.risk.record_pnl(delta)
        if self.store is not None:
            self.store.record_pnl(delta, note=f"early exit (capital recycle): {a['event']}",
                                  event_key=a["event"])
        log.warning("RECYCLED %s: sold %g ITM@%.2f%s -> freed $%.2f on %s (delta %+.2f "
                    "vs settlement)", a["event"], k, itm_avg,
                    f" + {j:g} OTM@{otm_avg:.2f}" if j > 1e-9 else "",
                    k * itm_avg, iv, delta)
        return {"event": a["event"], "freed": k * itm_avg, "delta": delta,
                "remnant": round(k - j, 6) if a["otm"] is not None else 0.0}

    async def execute(self, opp: ArbOpportunity) -> ExecutionReport:
        if self.risk.is_killed:
            return ExecutionReport(ExecStatus.SKIPPED, f"kill switch: {self.risk.kill_reason}")
        scarce = (self._scarce_skip(opp) or self._rebalance_skip(opp)
                  or self._churn_skip(opp) or self._recycle_skip(opp)
                  or self._horizon_skip(opp))
        if scarce is not None:
            return ExecutionReport(ExecStatus.SKIPPED, scarce)

        # Liquidity guard: don't fire unless BOTH legs have real resting depth. Thin
        # books are where a leg rejects and we can't hedge/unwind — skip them outright
        # so we should never have to unwind. opp.max_contracts is the min leg depth.
        if self.min_leg_depth > 0 and opp.max_contracts < self.min_leg_depth:
            return ExecutionReport(
                ExecStatus.SKIPPED,
                f"thin book: depth {opp.max_contracts:g} < min {self.min_leg_depth:g} contracts",
            )

        size, caps = self._max_size(opp)
        if size < 1:
            binding = min(caps, key=caps.get)
            return ExecutionReport(
                ExecStatus.SKIPPED,
                f"size < 1 contract (binding: {binding}={caps[binding]:.3f})",
            )
        binding = min(caps, key=caps.get)
        log.info("sizing %s: %d contracts (binding: %s) caps=%s",
                 opp.event_key, size, binding,
                 {k: round(v, 2) for k, v in caps.items()})

        # Order the two legs so the REJECTION-PRONE venue (self.take_first_venue) fires
        # FIRST: if it rejects/errors, no other leg was taken -> a clean skip, no unwind.
        # The SECOND (hedge) leg is the RELIABLE one whose failure would force an unwind —
        # and it's the leg the hedge-fillable pre-check below confirms can fill. (Live data:
        # Polymarket 500s the hedge buy on thin markets, so it must go first; then a 500 is a
        # free skip instead of a Kalshi unwind.)
        # RELIABILITY-AWARE ordering first: when exactly one leg's market is suspect
        # (active fail streak, or never proven) while the other is proven, the SUSPECT
        # leg fires first — its reject is then a free skip instead of stranding the
        # other leg (2026-07-07: next-day kalshi tennis markets REJECTED FOKs while
        # quoting on WS; the static poly-first order paid an unwind per attempt).
        def _suspect(vn: str, mk: str) -> bool:
            fills, _fails, streak, _mx = self._market_rel.get((vn, mk), (0, 0, 0, 0.0))
            return streak > 0 or fills == 0

        def _family_fail_rate(vn: str, mk: str) -> float:
            fam = self._family_rel.get(_family(vn, mk))
            if not fam or (fam[0] + fam[1]) < 5:
                return -1.0                       # not enough family evidence
            return fam[1] / (fam[0] + fam[1])

        yes_susp = _suspect(opp.buy_yes_venue, opp.buy_yes_market)
        no_susp = _suspect(opp.buy_no_venue, opp.buy_no_market)
        tfv = self.take_first_venue
        if yes_susp != no_susp:
            no_first = no_susp
        else:
            # tie (both proven or both unproven): break by FAMILY fail rate — a
            # brand-new overnight kalshi ITF market inherits its family's refusal
            # history, so its first-ever attempt fires kalshi first (free skip)
            # instead of paying an unwind to discover the refusal
            yr = _family_fail_rate(opp.buy_yes_venue, opp.buy_yes_market)
            nr = _family_fail_rate(opp.buy_no_venue, opp.buy_no_market)
            if yr >= 0 and nr >= 0 and abs(yr - nr) > 0.15:
                no_first = nr > yr
            else:
                no_first = opp.buy_no_venue == tfv and opp.buy_yes_venue != tfv
        if no_first:
            first_vn, first_m, first_side = opp.buy_no_venue, opp.buy_no_market, Side.NO
            second_vn, second_m, second_side = opp.buy_yes_venue, opp.buy_yes_market, Side.YES
        else:
            first_vn, first_m, first_side = opp.buy_yes_venue, opp.buy_yes_market, Side.YES
            second_vn, second_m, second_side = opp.buy_no_venue, opp.buy_no_market, Side.NO

        # Don't fire an edge too thin to give the hedge leg a fill cushion — it would
        # just unwind. We only fire when the edge can pay the hedge buffer AND still lock
        # the floor, so the hedge fills through normal book movement (no unwind).
        floor = self.min_lock_edge if self.min_lock_edge is not None else self.risk.limits.min_edge
        req_buffer = self._hedge_buffer_for(opp.max_contracts)
        if opp.edge_per_contract < floor + req_buffer - 1e-9:
            return ExecutionReport(
                ExecStatus.SKIPPED,
                f"edge {opp.edge_per_contract:.3f} < lock {floor:.3f} + hedge "
                f"{req_buffer:.3f} (depth {opp.max_contracts:g}) — would risk an unwind",
            )
        first_limit, second_limit = self._leg_limits(opp, first_side, second_side)
        notional = size * (first_limit + second_limit)   # worst-case cost for the risk check
        label = f"{opp.buy_yes_venue}:{opp.buy_yes_market}"
        decision = self.risk.check(label, notional)
        if not decision.allowed:
            return ExecutionReport(ExecStatus.SKIPPED, f"risk: {decision.reason}")

        first = (first_vn, first_m, first_side, first_limit)
        second = (second_vn, second_m, second_side, second_limit)
        first_venue = self.venues.get(first[0])
        second_venue = self.venues.get(second[0])
        if first_venue is None or second_venue is None:
            return ExecutionReport(ExecStatus.SKIPPED, "venue not available")

        # Re-confirm the hedge (deep) leg's LIVE book before committing leg 1 and SIZE DOWN
        # to what it would actually fill. Polymarket can thin out between the depth-confirm
        # and the fire, so taking the full size would leave the hedge short (a naked
        # remainder) or fire a FOK into vanished liquidity (the 500 trigger). Sizing to the
        # live fillable depth keeps the trade fully hedged at whatever the leg supports —
        # capturing a small real arb instead of skipping it. A fetch failure -> 0 (skip).
        #
        # FRESH-HEDGE FAST PATH: this re-read is the ONLY network round-trip on the fire
        # path (~40ms). When the opp is backed by WS quotes younger than fresh_hedge_secs
        # AND the hedge leg's own WS depth is >= 2x the trade size (the deep-cushion margin
        # squared), the live book already told us the hedge fills — skip the re-read and
        # fire immediately. Thin or stale hedges keep the confirm.
        hedge_ws_size = opp.no_size if second[2] is Side.NO else opp.yes_size
        if (self.fresh_hedge_secs > 0 and opp.fresh_ts > 0
                and time.time() - opp.fresh_ts <= self.fresh_hedge_secs
                and hedge_ws_size * self.hedge_depth_fraction >= 2 * size):
            fillable = float(size)
        else:
            fillable = await self._hedge_fillable(second_venue, second, size,
                                                  ceiling=second[3])
        if fillable < size - 1e-9:
            capped = int(fillable + 1e-9)
            if capped < 1:
                self._audit("execute_skip_hedge", opp, size=size)
                log.info("STREAM skip %s — hedge unfillable: book depth <1 contract at limit",
                         opp.event_key)
                return ExecutionReport(
                    ExecStatus.SKIPPED, "hedge unfillable: book depth <1 contract at limit")
            log.info("STREAM %s: sizing down to hedge-fillable %d (book showed %d)",
                     opp.event_key, capped, size)
            size = capped

        self._audit("execute_start", opp, size=size)

        # Reserve the estimated leg cost in the cached balance SYNCHRONOUSLY (before any
        # await) so a CONCURRENT fast-loop execution sees the drain immediately and doesn't
        # over-commit a draining venue -> a leg's insufficient_balance reject -> unwind. The
        # cache otherwise only refreshes on the slow (~minutes) loop and decrements at settle,
        # so two arbs firing together both read the full balance. Released in finally; the
        # real per-fill accounting stays in _settle_success / _unwind.
        reservation: dict[str, float] = {}
        reservation[first[0]] = reservation.get(first[0], 0.0) + first[3] * size
        reservation[second[0]] = reservation.get(second[0], 0.0) + second[3] * size
        for _v, _amt in reservation.items():
            self._spend(_v, _amt)
        try:
            return await self._fire_and_settle(
                opp, size, first, second, first_venue, second_venue)
        finally:
            for _v, _amt in reservation.items():
                self._spend(_v, -_amt)

    async def _fire_and_settle(self, opp, size, first, second, first_venue, second_venue):
        """Fire leg 1 (rejection-prone) then leg 2 (hedge), then resolve the outcome:
        settle a locked arb, unwind a clean leg-2 failure, or halt on an ambiguous state.
        Split out of execute() so its balance reservation can wrap this in try/finally."""
        # ----- Leg 1: the rejection-prone leg, IMMEDIATE-OR-CANCEL -----
        # IOC, not FOK: an all-or-nothing leg1 killed whenever the book held less
        # than the full size at the limit — 115 clean-skip misses/day. IOC takes
        # whatever quantity actually rests <= limit; a partial routes through the
        # existing hedge-the-filled-portion path, zero fill stays a free skip.
        leg1 = await self._place(
            first_venue, first[1], first[2], "buy", first[3], size,
            "immediate_or_cancel"
        )
        log.info("leg1 %s%s", leg1,
                 f" reason={_reject_reason(leg1)}"
                 if leg1.status is OrderStatus.REJECTED else "")
        if leg1.status is OrderStatus.ERROR:
            # An ambiguous order error (e.g. a transient network timeout) shouldn't
            # freeze the whole bot. Reconcile against the venue: if no position
            # resulted, it's a clean skip and we keep trading; only halt if a position
            # exists (real naked risk) or we can't tell (fail closed).
            left = await self._position_after_error(first_venue, first[1])
            if left is False:
                log.warning("leg1 ERROR but %s flat — no position, skipping (%s)",
                            first[1], _reject_reason(leg1))
                return ExecutionReport(
                    ExecStatus.SKIPPED,
                    f"leg1 ERROR, account flat — no position ({_reject_reason(leg1)})", [leg1])
            if left is True:
                # The order EXECUTED (a real position exists) but its HTTP response failed —
                # so the Kalshi hedge never fired and this leg is naked. Rather than halt the
                # whole bot, COMPLETE THE HEDGE for the unhedged imbalance to lock the arb
                # (the same recovery a human does by hand). Only halt if it can't be done safely.
                return await self._complete_hedge_after_leg1_error(
                    opp, size, first, second, first_venue, second_venue, leg1)
            # left is None -> couldn't read the account -> genuinely unknown -> fail closed.
            return self._halt(
                f"leg1 ERROR — fill state unknown ({_reject_reason(leg1)})", [leg1])
        if not leg1.left_a_position:
            # Clean: the first leg didn't fill, so NO position was taken — just skip.
            return ExecutionReport(
                ExecStatus.SKIPPED,
                f"leg1 not filled ({leg1.status.value}: {_reject_reason(leg1)})", [leg1])
        if not leg1.filled_fully:
            # FoK didn't behave all-or-nothing (observed on Polymarket: a 0.62/6 fill).
            # We HOLD leg1.filled — a perfectly good position. The old behavior sold it
            # straight back (guaranteed spread+slippage loss; the single worst unwind
            # category, -$32 all-time at -$4.6 avg). Just HEDGE the amount we actually
            # hold instead: shrink the trade to leg1.filled and continue into the normal
            # leg-2 flow — every failure path below (recross -> unwind) still applies.
            log.warning("leg1 PARTIAL %g/%g — hedging the filled portion instead of "
                        "unwinding it", leg1.filled, size)
            size = leg1.filled

        # ----- Leg 2: the hedge, fill-or-kill AT THE BREAKEVEN CEILING -----
        # The limit reaches to PnL-breakeven computed from leg1's ACTUAL fill: a FOK
        # fills at resting prices, so an unmoved book gives the full edge, a book
        # that ticked against us fills between floor and ~breakeven (strictly better
        # than the unwind), and only a would-be-loss kills (recross backstop below).
        p1 = leg1.avg_price if leg1.avg_price is not None else first[3]
        ceiling2 = self._breakeven_ceiling(
            getattr(first_venue, "name", first[0]), p1,
            getattr(second_venue, "name", second[0]))
        # the ceiling is BOTH floor and CAP: reaching above the detected ask is
        # free (FOK fills at resting prices), but a detected ask ABOVE the ceiling
        # must never widen the limit past breakeven — a 10ct lock booked -7.5c/ct
        # (LYONBBB 2026-07-09) when the second leg filled at 0.73 against a 0.674
        # ceiling via the max() arm.
        leg2_limit = ceiling2
        log.info("leg2 limit %.3f (detected ask %.3f, ceiling %.3f, leg1 fill %.3f)",
                 leg2_limit, second[3], ceiling2, p1)
        leg2 = await self._place(
            second_venue, second[1], second[2], "buy", leg2_limit, size, "fill_or_kill"
        )
        log.info("leg2 %s%s", leg2,
                 f" reason={_reject_reason(leg2)}"
                 if leg2.status is OrderStatus.REJECTED else "")

        if leg2.status is OrderStatus.FILLED and leg2.filled_fully:
            return self._settle_success(opp, size, [leg1, leg2])

        if leg2.status in (OrderStatus.KILLED, OrderStatus.REJECTED) and leg2.filled <= 1e-9:
            # Definitively no leg-2 position. Try a breakeven RECROSS before unwinding —
            # unwinding is a GUARANTEED loss (cross leg1's spread + slippage), while the
            # hedge is usually still available a tick worse. See _recross_or_unwind.
            log.warning("leg2 %s — %s", leg2.status.value, _reject_reason(leg2))
            return await self._recross_or_unwind(
                opp, leg1, leg2, second, second_venue, size,
                reason=f"leg2 {leg2.status.value} ({_reject_reason(leg2)})"
            )

        if leg2.status is OrderStatus.ERROR:
            # A leg-2 ERROR (e.g. a transient Polymarket 500 on POST /orders) is ambiguous:
            # the hedge may or may not have landed. Rather than freeze the WHOLE bot on one
            # venue hiccup, reconcile against the venue (as leg 1 does) and act on the real
            # hedge state. The hedge order is synchronous fill-or-kill (not a resting order),
            # so the reconciled position is authoritative — it cannot land "later".
            qty = await self._hedge_qty_after_error(second_venue, second[1])
            if qty is None:
                # Couldn't read the account -> genuinely unknown -> fail closed (halt).
                return self._halt(
                    f"leg2 ambiguous ({leg2.status.value}: {_reject_reason(leg2)}; "
                    f"hedge state unreadable) — manual reconcile", [leg1, leg2])
            # DELTA, not absolute: the venue position includes PRE-EXISTING holdings —
            # attempt #2's errored leg read attempt #1's 20 contracts as "hedge
            # landed" and booked itself hedged while 20 poly went naked (FRA-MAR
            # Hakimi, 2026-07-09, caught by reconcile at Δ20). Baseline = our
            # tracked position from confirmed fills.
            pre = abs(self._positions.get(
                (getattr(second_venue, "name", second[0]), second[1]), 0.0))
            qty = max(0.0, qty - pre)
            if qty >= size - 1e-9:
                # A 500 often means the server DID process the order then errored on the
                # response: the full hedge is on and the arb is locked -> settle it.
                log.warning("leg2 ERROR but %s holds %g contracts — hedge landed, settling (%s)",
                            second[1], qty, _reject_reason(leg2))
                leg2 = replace(
                    leg2, status=OrderStatus.FILLED, filled=size,
                    avg_price=leg2.avg_price if leg2.avg_price is not None else second[3])
                return self._settle_success(opp, size, [leg1, leg2])
            if qty <= 1e-9:
                # Hedge confirmed FLAT. The synchronous FOK didn't fill -> no leg-2
                # position -> try the breakeven recross, then unwind. One bad market
                # doesn't freeze the whole bot either way.
                log.warning("leg2 ERROR but %s flat — no hedge landed, recross/unwind (%s)",
                            second[1], _reject_reason(leg2))
                return await self._recross_or_unwind(
                    opp, leg1, leg2, second, second_venue, size,
                    reason=f"leg2 ERROR, hedge flat ({_reject_reason(leg2)})")
            # 0 < qty < size: a partial hedge — a known-but-mismatched naked remainder we
            # can't safely auto-resolve. Halt for manual reconciliation.
            return self._halt(
                f"leg2 ambiguous ({leg2.status.value}: {_reject_reason(leg2)}; "
                f"partial hedge {qty:g}/{size:g}) — manual reconcile", [leg1, leg2])

        # PARTIAL on leg 2 with a KNOWN filled quantity is NOT ambiguous — Polymarket
        # FOKs are documented to partial-fill (the 0.62/6 and 3.85/20 incidents).
        # Resolve instead of halting: TOP-UP the remainder once at the same ceiling
        # limit (books often refill within ms — the 3.85/20 remainder completed
        # manually at a BETTER price seconds later), then settle the matched portion
        # and unwind any excess. Only a zero/unknown fill remains a manual halt.
        filled2 = leg2.filled or 0.0
        if filled2 > 1e-9:
            hedged = min(float(size), filled2)
            hedge_px = leg2.avg_price if leg2.avg_price is not None else second[3]
            hedge_notional = hedged * hedge_px
            need = round(size - hedged, 6)
            if need >= 1.0:
                topup = await self._place(second_venue, second[1], second[2], "buy",
                                          leg2_limit, need, "immediate_or_cancel")
                log.warning("leg2 top-up %s", topup)
                k = topup.filled or 0.0
                if k > 1e-9:
                    hedge_notional += k * (topup.avg_price
                                           if topup.avg_price is not None else leg2_limit)
                    hedged = round(hedged + k, 6)
            if hedged >= size - 1e-9:
                merged = replace(leg2, status=OrderStatus.FILLED, filled=float(size),
                                 avg_price=hedge_notional / size)
                log.warning("leg2 PARTIAL %g/%g completed via top-up — locked",
                            filled2, size)
                return self._settle_success(opp, size, [leg1, merged])
            return await self._settle_partial_hedge(
                opp, leg1, float(size), hedged, hedge_notional, leg2,
                reason=f"taker leg2 PARTIAL {filled2:g}/{size:g}")
        return self._halt(
            f"leg2 ambiguous ({leg2.status.value}: {_reject_reason(leg2)}) — manual reconcile",
            [leg1, leg2],
        )

    async def _complete_hedge_after_leg1_error(
        self, opp, size, first, second, first_venue, second_venue, leg1,
    ) -> ExecutionReport:
        """Leg 1 errored ambiguously but LEFT A POSITION (the order executed; only its HTTP
        response failed), so the hedge never fired and the leg is naked. COMPLETE THE HEDGE:
        read both legs' real positions, fire the hedge leg for the unhedged imbalance, and
        settle the locked arb. Returns SUCCESS on recovery, SKIPPED if already balanced, or
        HALT if it can't safely complete.

        We hedge ``first_pos - second_pos`` (the imbalance) rather than leg 1's reported size:
        a leg-1 FOK can partial-fill (observed on Polymarket), and the market may already carry
        HEDGED prior fills, so only the imbalance is the true unhedged amount — robust to both.
        Bounded to this trade's ``size`` so a bookkeeping surprise can't auto-fire a huge order."""
        fpos = await self._hedge_qty_after_error(first_venue, first[1])
        spos = await self._hedge_qty_after_error(second_venue, second[1])
        if fpos is None or spos is None:
            return self._halt(
                f"leg1 ERROR — position exists but couldn't read both legs to complete the "
                f"hedge ({_reject_reason(leg1)})", [leg1])
        unhedged = round(fpos - spos, 6)
        if unhedged < 1.0:
            # Legs already balanced (the position is covered by prior hedged fills) — no naked
            # remainder from this error. Safe to skip and keep trading.
            log.warning("leg1 ERROR on %s but legs already balanced (%g vs %g) — no naked, skipping",
                        first[1], fpos, spos)
            return ExecutionReport(
                ExecStatus.SKIPPED,
                f"leg1 ERROR, legs balanced ({fpos:g}/{spos:g}) — no naked ({_reject_reason(leg1)})",
                [leg1])
        if unhedged > size + 1e-9:
            # More naked than this trade asked for — an unexplained excess we shouldn't chase
            # automatically. Halt for manual reconciliation.
            return self._halt(
                f"leg1 ERROR — naked imbalance {unhedged:g} exceeds trade size {size:g} on "
                f"{first[1]} ({_reject_reason(leg1)})", [leg1])
        # Fire the hedge (reliable, second) leg for exactly the unhedged amount.
        hedge = await self._place(
            second_venue, second[1], second[2], "buy", second[3], unhedged, "fill_or_kill")
        log.info("leg1-ERROR recovery hedge %s", hedge)
        if hedge.status is OrderStatus.FILLED and hedge.filled_fully:
            # Reconstruct leg 1 at the recovered (now-balanced) size for settle accounting; its
            # fill price is unknown after the error, so use its limit (the FOK ceiling).
            leg1_recovered = replace(
                leg1, status=OrderStatus.FILLED, filled=unhedged,
                avg_price=leg1.avg_price if leg1.avg_price is not None else first[3])
            log.warning("leg1 ERROR RECOVERED on %s|%s: completed hedge of %g — LOCKED instead "
                        "of halting naked", first[1], second[1], unhedged)
            return self._settle_success(opp, unhedged, [leg1_recovered, hedge])
        # Hedge didn't complete -> still naked, and now we KNOW it (not ambiguous). Halt so the
        # remainder is reconciled by hand rather than left to drift.
        return self._halt(
            f"leg1 ERROR — hedge-completion failed ({hedge.status.value}: "
            f"{_reject_reason(hedge)}); {unhedged:g} still naked on {first[1]}", [leg1, hedge])

    async def _recross_or_unwind(self, opp, leg1, leg2, second, second_venue, size,
                                 *, reason: str) -> ExecutionReport:
        """A failed hedge doesn't mean the hedge is GONE — usually the book moved a tick.
        Unwinding leg 1 is a GUARANTEED loss (cross its spread + slippage, historically
        ~-$0.16 avg and 89% of gross profit all-time), so first RE-READ the hedge book and
        take it at up to BREAKEVEN + recross_epsilon: worst case we lock a ~$0 arb (or an
        epsilon loss strictly smaller than the expected unwind cost), best case the edge
        survived the tick. Only when the book truly can't hedge near breakeven do we pay
        for the unwind. One extra book read + at most one order, on the failure path only."""
        buy_px = leg1.avg_price if leg1.avg_price is not None else (
            opp.yes_price if leg1.side is Side.YES else opp.no_price)
        # PnL-breakeven ceiling, not price-breakeven (see _breakeven_ceiling).
        ceiling = self._breakeven_ceiling(leg1.venue, buy_px, second[0])
        ask = await self._taker_ask(second, second_venue)
        if ask is not None and ask <= ceiling + 1e-9:
            retry = await self._place(
                second_venue, second[1], second[2], "buy", ceiling, size, "fill_or_kill")
            log.warning("recross %s", retry)
            if retry.status is OrderStatus.FILLED and retry.filled_fully:
                log.warning("RECROSSED %s: hedge landed at %.3f (breakeven %.3f + eps) — "
                            "locked instead of unwinding", opp.event_key,
                            retry.avg_price if retry.avg_price is not None else ceiling,
                            1.0 - buy_px)
                return self._settle_success(opp, size, [leg1, retry])
            if retry.status is OrderStatus.ERROR:
                # Ambiguous retry: reconcile like the main leg-2 ERROR path — settle if it
                # actually landed, otherwise fall through to the unwind.
                qty = await self._hedge_qty_after_error(second_venue, second[1])
                if qty is not None and qty >= size - 1e-9:
                    landed = replace(retry, status=OrderStatus.FILLED, filled=size,
                                     avg_price=retry.avg_price if retry.avg_price is not None
                                     else ceiling)
                    return self._settle_success(opp, size, [leg1, landed])
                if qty is None or qty > 1e-9:
                    return self._halt(
                        f"recross ambiguous ({_reject_reason(retry)}; hedge state "
                        f"{'unreadable' if qty is None else f'partial {qty:g}/{size:g}'}) "
                        f"— manual reconcile", [leg1, retry])
        return await self._unwind(opp, leg1, leg2, reason=reason)

    async def _cancel_maker(self, venue, order_id) -> None:
        """Best-effort cancel of a resting maker. A GOOD_TILL_CANCEL maker does not
        self-expire, so the executor must cancel an unfilled one or it would rest unhedged.
        A cancel that races a fill is harmless — the fill confirmation stays authoritative.

        A 404/400 on the cancel means the order is no longer cancellable (already filled,
        expired, or cancelled) — which is exactly the state we wanted, so it's logged quietly
        WITH the venue's response body for diagnosis. Anything else (5xx, network) stays a
        warning. A genuinely stranded resting order that later fills is caught by the periodic
        naked-exposure reconciliation (EXEC_RECONCILE_HALT), the cross-cycle backstop."""
        cancel = getattr(venue, "cancel_order", None)
        if cancel is None or not order_id:
            return
        try:
            await cancel(order_id)
            return
        except Exception as exc:
            resp = getattr(exc, "response", None)
            code = getattr(resp, "status_code", None)
            body = ""
            if resp is not None:
                try:
                    body = resp.text[:200]
                except Exception:
                    body = ""
            if code in (400, 404):
                log.info("maker %s not cancellable (already filled/expired/cancelled): "
                         "HTTP %s %s", order_id, code, body)
            else:
                log.warning("maker cancel failed for %s: HTTP %s %s (%s)",
                            order_id, code, body, exc)

    async def _taker_ask(self, taker, taker_venue) -> float | None:
        """Current ask for the taker (hedge) leg — from the STREAMED live book when
        fresh (the WS delivers it at ~40-70ms lag, free), REST only as fallback.
        The drift guard was REST-polling the same book once a second per resting
        maker (155 fetches/5min observed on one pair)."""
        lq = self.live_quote
        if lq is not None:
            try:
                q = lq(getattr(taker_venue, "name", taker[0]), taker[1])
            except Exception:
                q = None
            ts = getattr(q, "timestamp", 0.0) or 0.0 if q is not None else 0.0
            if q is not None and time.time() - ts < 5.0:
                ask = q.yes_ask if taker[2] is Side.YES else q.no_ask
                if ask is not None:
                    return ask
        try:
            q = await taker_venue.fetch_quote(RawMarket(market_id=taker[1], title="", raw={}))
        except Exception as exc:
            log.warning("maker drift check: taker quote failed for %s: %s", taker[1], exc)
            return None
        return q.yes_ask if taker[2] is Side.YES else q.no_ask

    async def _rest_with_drift_guard(
        self, m, maker, taker, maker_venue, taker_venue, maker_px, size, floor,
    ) -> tuple[float, float | None]:
        """Wait for the resting maker to fill/expire while polling the taker leg. If the
        taker drifts so the would-be hedge can no longer lock ``floor``, CANCEL the maker
        before it fills into the adverse move — the only safe way to arm thin edges. Returns
        ``(filled, avg_price)`` from the authoritative fill confirmation."""
        confirm = asyncio.ensure_future(
            self.fill_confirmer.confirm(m.venue, m.order_id, size, self.maker_timeout + 1.5))
        deadline = time.time() + self.maker_timeout
        cancelled = False
        try:
            while not confirm.done() and time.time() < deadline:
                done, _ = await asyncio.wait({confirm}, timeout=self.maker_poll)
                if confirm in done:
                    break
                ask = await self._taker_ask(taker, taker_venue)
                if ask is None:
                    continue
                # Rested leg pays the MAKER fee; the taker (hedge) leg pays taker — mirror the
                # arm gate so we don't cancel a maker that armed on its true (maker-fee) edge.
                maker_fee_fn = getattr(maker_venue, "maker_fee_model", None) or self._fee(maker[0])
                fee = maker_fee_fn.fee(maker_px, size) + self._fee(taker[0]).fee(ask, size)
                edge_now = (size * (1.0 - maker_px - ask) - fee) / size
                if edge_now < floor - 1e-9:
                    log.info(
                        "maker adverse drift on %s: taker ask %.3f -> hedge edge %.3f < floor "
                        "%.3f, cancelling maker before fill", maker[1], ask, edge_now, floor)
                    await self._cancel_maker(maker_venue, m.order_id)
                    cancelled = True
                    break
        except asyncio.CancelledError:
            confirm.cancel()
            raise
        # A GOOD_TILL_CANCEL maker does not self-expire: if it neither filled nor was already
        # cancelled on drift, cancel it now — BEFORE awaiting the authoritative confirm, so a
        # fill racing the cancel is still caught — otherwise it would rest on the book unhedged.
        if not confirm.done() and not cancelled:
            await self._cancel_maker(maker_venue, m.order_id)
        # Whether we cancelled or not, the confirmation is authoritative: a cancel that
        # raced a fill still reports the real filled amount (which we then hedge).
        _, filled, avg = await confirm
        return filled, avg

    async def execute_maker(self, opp: ArbOpportunity) -> ExecutionReport:
        """Capture an edge by RESTING the fee-heavy leg (``first_venue``, e.g. Kalshi)
        as a maker — no slippage, lower fee — then TAKING the deep leg (Polymarket) the
        instant the maker fills. Lets THIN edges be captured (we capture the spread
        instead of paying it). An unfilled maker is cancelled at the timeout, so an
        uncrossed quote is a clean no-trade. The naked window is just the maker-fill ->
        taker round trip; a hedge that can't fully fill re-crosses, then unwinds the
        unhedged remainder — it never leaves a naked leg resting.
        """
        if self.risk.is_killed:
            return ExecutionReport(ExecStatus.SKIPPED, f"kill switch: {self.risk.kill_reason}")
        scarce = (self._scarce_skip(opp) or self._rebalance_skip(opp)
                  or self._churn_skip(opp) or self._recycle_skip(opp)
                  or self._horizon_skip(opp))
        if scarce is not None:
            return ExecutionReport(ExecStatus.SKIPPED, scarce)
        if self.fill_confirmer is None:
            return ExecutionReport(ExecStatus.SKIPPED, "maker mode needs a fill confirmer")

        # Maker leg = the first_venue (fee-heavy) side; taker = the deep, cheap side. The
        # HEDGE (taker) leg's depth is what binds in maker mode: the resting maker ADDS
        # liquidity (its own book depth doesn't constrain us), but we must cross the taker
        # to hedge a fill. So guard/size on the taker leg's own depth, not min(both legs).
        yes_leg = (opp.buy_yes_venue, opp.buy_yes_market, Side.YES, opp.yes_price)
        no_leg = (opp.buy_no_venue, opp.buy_no_market, Side.NO, opp.no_price)
        # HEDGE depth is the taker leg's LADDER depth inside the profit ceiling, not its
        # top level: the hedge is an IOC that sweeps every level priced <= the ceiling
        # (1 - maker price - floor - fees), so 1-at-top with 300 behind at +1 tick hedges
        # 301 profitably. This was the "thin hedge book" skip class (~540/day) — books
        # deep enough to hedge, judged only by their top. Falls back to top size when
        # the opp carries no ladders (REST-era callers, tests).
        floor0 = self.min_lock_edge if self.min_lock_edge is not None else self.risk.limits.min_edge

        def _band(levels, top_size, own_px, other_px, own_venue, other_venue):
            if not levels:
                return top_size
            fees_ct = (per_contract_fee(self._fee(other_venue), other_px)
                       + per_contract_fee(self._fee(own_venue), own_px))
            ceiling = min(0.99, max(0.01, round(1.0 - other_px - floor0 - fees_ct, 4)))
            return usable_depth(levels, ceiling)

        yes_band = _band(opp.yes_levels, opp.yes_size, opp.yes_price, opp.no_price,
                         opp.buy_yes_venue, opp.buy_no_venue)
        no_band = _band(opp.no_levels, opp.no_size, opp.no_price, opp.yes_price,
                        opp.buy_no_venue, opp.buy_yes_venue)
        if self.maker_dynamic and yes_band > 0 and no_band > 0:
            # Rest the maker on the THIN leg (the bottleneck); TAKE the deep leg (it fills
            # reliably). hedge_depth is the taker leg's usable band — what actually binds.
            if yes_band <= no_band:
                maker, taker, hedge_depth = yes_leg, no_leg, no_band
            else:
                maker, taker, hedge_depth = no_leg, yes_leg, yes_band
        elif opp.buy_no_venue == self.first_venue:
            maker, taker, hedge_depth = no_leg, yes_leg, (yes_band or opp.max_contracts)
        elif opp.buy_yes_venue == self.first_venue:
            maker, taker, hedge_depth = yes_leg, no_leg, (no_band or opp.max_contracts)
        else:
            return await self.execute(opp)   # neither leg on the maker venue -> taker path

        size, caps = self._max_size(opp, depth_override=hedge_depth)
        if self.min_leg_depth > 0:
            # Scale the depth bar with the size we would ACTUALLY fire: a 1-2ct
            # probe needs ~3 of hedge depth, not the full static minimum — the
            # static bar skipped ~260 probe-size opportunities/day on books that
            # could hedge them 3x over. Larger fires keep the full bar.
            need = max(3.0, min(float(self.min_leg_depth), 2.0 * max(size, 1)))
            if hedge_depth < need:
                top = opp.yes_size if taker[2] is Side.YES else opp.no_size
                return ExecutionReport(
                    ExecStatus.SKIPPED,
                    f"thin hedge book: band depth {hedge_depth:g} (top {top:g}) "
                    f"< min {need:g} (size {size})")
        if size < 1:
            binding = min(caps, key=caps.get)
            return ExecutionReport(
                ExecStatus.SKIPPED, f"size < 1 (binding: {binding}={caps[binding]:.3f})")

        maker_venue = self.venues.get(maker[0])
        taker_venue = self.venues.get(taker[0])
        if maker_venue is None or taker_venue is None:
            return ExecutionReport(ExecStatus.SKIPPED, "venue not available")

        floor = self.min_lock_edge if self.min_lock_edge is not None else self.risk.limits.min_edge
        # Re-price the edge with the RESTED leg at its MAKER fee (not the taker fee the matcher
        # charged it) and at the real order size: the Kalshi maker fee (0.0175) is ~4x below the
        # taker rate (0.07), so charging taker here would skip thin maker arbs that are genuinely
        # profitable. The taker (hedge) leg keeps its taker fee. _rest_with_drift_guard mirrors
        # this, so a maker armed on its real edge isn't immediately cancelled on the same basis.
        maker_fee_fn = getattr(maker_venue, "maker_fee_model", None) or self._fee(maker[0])
        fee = maker_fee_fn.fee(maker[3], size) + self._fee(taker[0]).fee(taker[3], size)
        # IN-PLAY = TAKER-ONLY: adverse selection on live-game books is intrinsic
        # (a resting quote fills mid-sweep; the hedge is already gone — net -$4.65
        # across two 4h samples even with the slip cushion). Pre-game and dateless
        # boards (draft, futures) keep the maker: slow books, the drift guard wins.
        kalshi_leg = (opp.buy_yes_market if opp.buy_yes_venue == "kalshi"
                      else opp.buy_no_market if opp.buy_no_venue == "kalshi" else None)
        start_ts = self._kalshi_start_ts(kalshi_leg) if kalshi_leg else None
        if start_ts is not None and time.time() >= start_ts:
            return await self.execute(opp)
        maker_edge = (size * (1.0 - maker[3] - taker[3]) - fee) / size
        # A maker CHOOSES its price: when one tick inside the ask doesn't clear the
        # lock floor + drift cushion, rest DEEPER in the spread at the price that
        # manufactures exactly that edge against the CURRENT hedge ask. Resting is
        # free (post-only; the preview + forced hedge only engage on a fill), so a
        # wide-spread pair with no taker edge is still a market-making opportunity —
        # this converts the edge-gone class instead of skipping it.
        arm = (floor + self.maker_arm_cushion
               + self._family_slip(taker[0], taker[1]))
        rest_cap = None
        if maker_edge < arm - 1e-9:
            fee_ct = fee / size
            rest_cap = round(1.0 - taker[3] - fee_ct - arm, 4)
            if rest_cap < 0.01:
                return ExecutionReport(
                    ExecStatus.SKIPPED,
                    f"maker edge {maker_edge:.3f} < lock {floor:.3f} + maker cushion "
                    f"{self.maker_arm_cushion:.3f} and no room to rest deeper")

        # Don't arm a maker we can't hedge. Re-read the taker (hedge) leg's LIVE book and
        # SIZE DOWN to its real top-of-book depth; skip entirely if it can't fill at all
        # (else the post-fill hedge KILLs at the cap and forces an unwind). The taker path
        # does the same before committing leg 1. A fetch failure -> 0 (skip, never arm blind).
        maker_px_planned = max(0.01, round(maker[3] - self.maker_improvement, 4))
        if rest_cap is not None:
            maker_px_planned = min(maker_px_planned, rest_cap)
        hedge_fees_ct = (per_contract_fee(maker_fee_fn, maker_px_planned)
                         + per_contract_fee(self._fee(taker[0]), taker[3]))
        hedge_ceiling = min(0.99, max(0.01, round(
            1.0 - maker_px_planned - floor - hedge_fees_ct, 4)))
        fillable = await self._hedge_fillable(taker_venue, taker, size, ceiling=hedge_ceiling)
        if fillable < 1:
            return ExecutionReport(
                ExecStatus.SKIPPED,
                f"hedge unfillable: book depth {fillable:g} < 1 — would arm an unhedgeable maker")
        # PROPORTIONAL cover: the hedge band must hold 3x the rest size — a fill
        # arrives on a book sweep, and a band that barely covers the size pre-fill
        # is gone post-fill (24h: 62 killed hedges vs 18 clean).
        if fillable < 3 * size:
            size = max(1, int(fillable // 3))

        notional = size * (opp.yes_price + opp.no_price)
        decision = self.risk.check(f"{opp.buy_yes_venue}:{opp.buy_yes_market}", notional)
        if not decision.allowed:
            return ExecutionReport(ExecStatus.SKIPPED, f"risk: {decision.reason}")

        self._audit("maker_start", opp, size=size)

        # ----- Rest the maker (post-only, self-expiring) -----
        # Post one tick INSIDE the ask (a cheaper bid) so it rests instead of crossing
        # (post-only rejects a marketable price). This also captures an extra tick of
        # edge: our cost is maker_px < the quoted ask, so the locked edge only grows.
        maker_px = max(0.01, round(maker[3] - self.maker_improvement, 4))
        if rest_cap is not None:
            maker_px = min(maker_px, rest_cap)
            log.info("maker resting DEEP at %.2f (spread ask %.2f) to manufacture "
                     "the %.3f floor edge", maker_px, maker[3], arm)
        try:
            m = await maker_venue.place_order(
                maker[1], maker[2], "buy", maker_px, size,
                tif="gtc", post_only=True, expiration_ts=int(time.time() + self.maker_timeout))
        except Exception as exc:
            return ExecutionReport(ExecStatus.SKIPPED, f"maker placement failed: {exc}")
        if m.status is OrderStatus.REJECTED:   # post-only would have crossed -> no position
            return ExecutionReport(
                ExecStatus.SKIPPED, f"maker would cross / rejected ({_reject_reason(m)})", [m])
        if m.status is OrderStatus.ERROR:
            left = await self._position_after_error(maker_venue, maker[1])
            if left is False:
                return ExecutionReport(ExecStatus.SKIPPED, f"maker ERROR but flat ({_reject_reason(m)})")
            return self._halt(f"maker ERROR — fill state unknown ({_reject_reason(m)})", [m])
        log.info("maker %s", m)

        # ----- Wait for the maker to fill (per-order, via the private WS) or expire -----
        filled = m.filled
        avg = m.avg_price
        if m.status is OrderStatus.RESTING and m.order_id:
            if self.maker_poll > 0:
                # Watch the taker while resting: cancel the maker if the taker drifts so the
                # hedge could no longer lock the floor — don't let it fill into a loss.
                filled, avg = await self._rest_with_drift_guard(
                    m, maker, taker, maker_venue, taker_venue, maker_px, size, floor)
            else:
                # No drift polling: wait up to the maker timeout for a fill, then CANCEL
                # (a GOOD_TILL_CANCEL maker won't self-expire) BEFORE the authoritative
                # confirm — so an unfilled maker can't rest unhedged, and a fill racing the
                # cancel is still caught by the confirm (which runs slightly past the timeout).
                confirm = asyncio.ensure_future(
                    self.fill_confirmer.confirm(m.venue, m.order_id, size, self.maker_timeout + 1.5))
                try:
                    await asyncio.wait({confirm}, timeout=self.maker_timeout)
                finally:
                    if not confirm.done():
                        await self._cancel_maker(maker_venue, m.order_id)
                _, filled, avg = await confirm
        # AUTHORITATIVE maker fill: the private-fill confirmer can UNDER-report a venue maker
        # partial — a Polymarket maker that filled 17/20 was booked as 0, leaving 17 naked
        # contracts and tripping a RECONCILE HALT. Re-read the order's true filled qty from
        # the venue and trust the larger value, so we never walk away from a real fill.
        fn = getattr(maker_venue, "order_filled_qty", None)
        if fn is not None and m.order_id:
            async def _read():
                try:
                    return await fn(m.order_id, size, maker[2])
                except Exception as exc:
                    log.warning("maker true-fill read failed for %s: %s", m.order_id, exc)
                    return None

            true_qty = await _read()
            if true_qty is None or true_qty <= filled + 1e-9:
                # CONFIRMATION RETRY: the order record is eventually consistent — a live
                # incident had fills land at T+0s, the record still read the OLD count in
                # this window, and 8 contracts leaked naked until the reconcile halted.
                # The maker is already cancelled here, so ~2.5s later the record is final.
                await asyncio.sleep(2.5)
                second = await _read()
                if second is not None:
                    true_qty = second if true_qty is None else max(true_qty, second)
            if true_qty is not None:
                if true_qty > filled + 1e-9:
                    log.warning("maker fill UNDER-REPORTED by confirmer (%g) — venue shows %g; "
                                "hedging the true fill (would have leaked naked)", filled, true_qty)
                filled = max(filled, true_qty)
            elif filled <= 1e-9:
                # Confirmer says unfilled AND the venue read failed. If the venue's
                # account plane is DOWN venue-wide (kalshi portfolio API flapping,
                # 2026-07-09 — two kill-switch halts in one morning), park a
                # recovery instead of halting: retry the read until the venue
                # answers; 0 -> clean, >0 -> hedge the true fill.
                if await self._venue_down(maker_venue):
                    log.critical("maker fill read: venue %s account plane DOWN — "
                                 "parking fill-state recovery (no kill switch)",
                                 maker[0])
                    self.venue_down.add(getattr(maker_venue, "name", maker[0]))

                    async def _fill_recover():
                        for _ in range(240):
                            await asyncio.sleep(60.0)
                            try:
                                tq = await _read()
                            except Exception:
                                tq = None
                            if tq is not None:
                                self.venue_down.discard(
                                    getattr(maker_venue, "name", maker[0]))
                                if tq <= 1e-9:
                                    log.warning("parked fill-state recovery: %s "
                                                "confirmed UNFILLED — clean", maker[1])
                                    return
                                log.warning("parked fill-state recovery: %s shows %g "
                                            "filled — completing hedge", maker[1], tq)
                                try:
                                    rep = await self._complete_hedge_after_leg1_error(
                                        opp, tq, (maker[0], maker[1], maker[2], maker_px),
                                        (taker[0], taker[1], taker[2], taker[3]),
                                        maker_venue, taker_venue, m)
                                    log.warning("fill-state recovery hedge: %s %s",
                                                rep.status, rep.reason)
                                except Exception as exc:
                                    self.risk.trip_kill_switch(
                                        f"fill-state recovery failed: {exc}")
                                return
                        self.risk.trip_kill_switch(
                            "maker venue never recovered for fill-state read")

                    self._recovery_tasks.append(asyncio.ensure_future(_fill_recover()))
                    return ExecutionReport(
                        ExecStatus.HALTED,
                        "maker fill state unreadable + venue down — recovery parked "
                        "(no kill switch)", [m])
                # venue healthy, read still failed -> genuinely ambiguous: fail closed
                return self._halt(
                    "maker fill state unreadable after cancel (confirmer 0, venue read "
                    "failed) — manual reconcile", [m])
        if filled <= 1e-9:
            return ExecutionReport(
                ExecStatus.SKIPPED, "maker unfilled — expired/cancelled, no trade", [m])

        # ----- Maker filled -> take the deep hedge immediately (IOC) -----
        maker_leg = OrderResult(
            venue=m.venue, market_id=maker[1], side=maker[2], action="buy",
            requested=filled, filled=filled,
            avg_price=(self.confirmer_avg(maker[2], avg)
                       if avg is not None else maker[3]),
            order_id=m.order_id, status=OrderStatus.FILLED)
        # Re-fetch the hedge venue's LIVE book and cross the current ask: the maker may
        # have rested for seconds, so the opp's price is stale — a stale limit misses
        # and forces an unwind. We're committed once the maker filled, so cross to fill.
        taker_px = taker[3]
        try:
            fresh = await taker_venue.fetch_quote(
                RawMarket(market_id=taker[1], title="", raw={}))
        except Exception as exc:
            log.warning("hedge book refetch failed for %s: %s", taker[1], exc)
            fresh = None
        if fresh is not None:
            live_ask = fresh.yes_ask if taker[2] is Side.YES else fresh.no_ask
            if live_ask is not None:
                taker_px = live_ask
        # The maker has already filled — we hold a one-sided position, so we MUST hedge
        # (holding it naked is directional risk, not arbitrage). But if the taker drifted
        # past break-even while the maker rested, this hedge locks a guaranteed loss. Bound
        # it (hedging caps the loss; naked does not) but log it loudly — a recurring forced
        # loss means the arming cushion (maker_arm_cushion) is too small for the drift.
        maker_cost = maker_leg.avg_price if maker_leg.avg_price is not None else maker[3]
        if maker_cost + taker_px >= 1.0:
            log.warning(
                "maker FORCED-LOSS hedge on %s: maker filled %s@%.3f, taker ask now %.3f "
                "(combined %.3f > 1) — hedging to bound the loss; raise EXEC_MAKER_ARM_CUSHION",
                opp.event_key, maker[2].value, maker_cost, taker_px, maker_cost + taker_px)
        # Reach to the PROFIT ceiling, not just top-ask + buffer: the pre-arm gate
        # counted ladder depth up to the ceiling, so the IOC limit must reach it or the
        # counted depth isn't takeable. IOC fills at actual level prices (never pays the
        # limit unless the book is there), and the drift guard above already bounded the
        # forced-loss case.
        fees_ct_fire = (per_contract_fee(maker_fee_fn, maker_cost)
                        + per_contract_fee(self._fee(taker[0]), taker_px))
        fire_ceiling = max(0.01, round(1.0 - maker_cost - floor - fees_ct_fire, 4))
        taker_limit = min(0.99, max(round(taker_px + self.hedge_buffer, 4), fire_ceiling))
        attempt = 0
        hedged = 0.0                 # contracts of the hedge confirmed filled so far
        hedge_notional = 0.0         # sum(filled_i * price_i) -> blended hedge avg price
        while True:
            need = round(filled - hedged, 6)
            hedge = await self._place(
                taker_venue, taker[1], taker[2], "buy", taker_limit, need, "immediate_or_cancel")
            log.info("maker-hedge %s", hedge)

            if hedge.status is OrderStatus.ERROR:
                # ----- hedge ERROR (ambiguous: e.g. a Polymarket 500/timeout) -----
                # We already HOLD the maker fill, so holding it naked into settlement is the
                # dangerous state. Reconcile the hedge venue for the TRUE open hedge quantity
                # (rules out a double-up on retry) and fold it into ``hedged``; only a venue
                # we can't read at all halts for manual reconciliation.
                qty = await self._hedge_qty_after_error(taker_venue, taker[1])
                if qty is None:
                    if await self._venue_down(taker_venue):
                        # VENUE OUTAGE (maintenance): the naked maker fill is real
                        # but the venue will return — park a recovery task that
                        # completes the hedge when it does; do NOT kill the bot
                        # over a scheduled maintenance window.
                        log.critical(
                            "maker hedge venue DOWN (%s) with a naked fill on %s — "
                            "parking hedge recovery, suspending fires on that venue",
                            _reject_reason(hedge), maker[1])
                        self._schedule_hedge_recovery(
                            opp, size, (maker[0], maker[1], maker[2], maker_px),
                            (taker[0], taker[1], taker[2], taker_limit),
                            maker_venue, taker_venue, maker_leg)
                        return ExecutionReport(
                            ExecStatus.HALTED,
                            "hedge venue down — recovery parked (no kill switch)",
                            [maker_leg, hedge])
                    return self._halt(
                        f"maker hedge ambiguous (ERROR: {_reject_reason(hedge)}; "
                        f"hedge qty unreadable) — manual reconcile", [maker_leg, hedge])
                if qty > hedged:     # newly-landed contracts, priced at the limit we sent
                    hedge_notional += (qty - hedged) * (
                        hedge.avg_price if hedge.avg_price is not None else taker_limit)
                hedged = qty
                log.warning("maker hedge ERROR; reconciled %s open=%g of %g (%s)",
                            taker[1], hedged, filled, _reject_reason(hedge))
            elif hedge.left_a_position:
                # FILLED (full or partial) — a KNOWN fill amount we can trust.
                hedged += hedge.filled
                hedge_notional += hedge.filled * (
                    hedge.avg_price if hedge.avg_price is not None else taker_limit)

            # ----- shared decision: fully hedged -> settle; else re-cross or unwind -----
            if hedged >= filled - 1e-9:
                combined = replace(
                    hedge, side=taker[2], action="buy", requested=filled, filled=hedged,
                    status=OrderStatus.FILLED,
                    avg_price=(hedge_notional / hedged) if hedged > 1e-9 else taker_limit)
                self._note_maker_slip(
                    taker[0], taker[1],
                    (combined.avg_price if combined.avg_price is not None
                     else taker_limit) - taker_px)
                return self._settle_success(opp, filled, [maker_leg, combined])

            if attempt < self.hedge_retries:
                # Not fully hedged, but the unfilled remainder is reconciled flat (no
                # double-up risk): RE-CROSS it at a fresh, current ask instead of giving up.
                # Most KILLED/PARTIAL hedges are transient thinning at our limit — a re-cross
                # usually locks the arb rather than eating the unwind spread.
                attempt += 1
                log.warning(
                    "maker hedge incomplete (%g/%g) — re-crossing remainder (%d/%d) (%s: %s)",
                    hedged, filled, attempt, self.hedge_retries,
                    hedge.status.value, _reject_reason(hedge))
                await asyncio.sleep(0.25)
                try:
                    fresh = await taker_venue.fetch_quote(
                        RawMarket(market_id=taker[1], title="", raw={}))
                    live = fresh.yes_ask if taker[2] is Side.YES else fresh.no_ask
                    if live is not None:
                        taker_limit = min(0.99, round(live + self._hedge_buffer_for(need), 4))
                except Exception as exc:
                    log.warning("hedge re-cross book refetch failed for %s: %s", taker[1], exc)
                continue

            # ----- retries exhausted: settle what's hedged, UNWIND the unhedged remainder -----
            # adverse-selection ledger: the hedge could not fill inside the ceiling
            # at all — book the full ceiling shortfall + a tick as the family slip
            self._note_maker_slip(taker[0], taker[1],
                                  (fire_ceiling - taker_px) + 0.01)
            # Ending flat-or-hedged always beats halting with a naked leg: the matched
            # ``hedged`` portion is a locked arb; the maker's unhedged excess is sold back.
            return await self._settle_partial_hedge(
                opp, maker_leg, filled, hedged, hedge_notional, hedge,
                reason=f"maker hedge incomplete ({hedged:g}/{filled:g}: {_reject_reason(hedge)})")

    # ---- outcomes ----
    def _settle_success(self, opp, size, legs) -> ExecutionReport:
        try:
            ya = next(l.avg_price for l in legs if l.side is Side.YES)
            na = next(l.avg_price for l in legs if l.side is Side.NO)
            if ya is not None and na is not None and (1.0 - ya - na) < -0.025:
                log.critical("NEGATIVE LOCK beyond tolerance: %s yes=%.3f no=%.3f "
                             "sum=%.3f size=%g — leg limits/detected prices in the "
                             "preceding lines; investigate the pricing path",
                             opp.event_key, ya, na, ya + na, size)
        except (StopIteration, TypeError):
            pass
        # Identify the legs by side (the placement order may put the NO leg first).
        yes_leg = next(leg for leg in legs if leg.side is Side.YES)
        no_leg = next(leg for leg in legs if leg.side is Side.NO)
        ya = yes_leg.avg_price if yes_leg.avg_price is not None else opp.yes_price
        na = no_leg.avg_price if no_leg.avg_price is not None else opp.no_price
        fees = self._fee(yes_leg.venue).fee(ya, size) + self._fee(no_leg.venue).fee(na, size)
        pnl = size * (1.0 - ya - na) - fees

        # Record the PAIR's full committed notional (both legs) under the same label
        # risk.check()/_max_size gate on — recording only the YES leg's half made
        # recorded exposure ~half of what the gates were sized against.
        self.risk.record_fill(f"{yes_leg.venue}:{yes_leg.market_id}", (ya + na) * size)
        self.risk.record_pnl(pnl)
        # Decrement tracked cash so the next arb sizes against what's actually left.
        self._spend(yes_leg.venue, ya * size)
        self._spend(no_leg.venue, na * size)
        # Update the cached net position so the churn guard sees this hedge immediately
        # (before the next 30s poll) and won't fire the reverse direction on it. Keyed off
        # the opportunity's legs — exactly what the guard reads — not the venue's echoed ids.
        self._track_fill(opp.buy_yes_venue, opp.buy_yes_market,
                         opp.buy_no_venue, opp.buy_no_market, size)
        if self.store is not None:
            self.store.record_fill(yes_leg.venue, yes_leg.market_id, "YES", ya, size)
            self.store.record_fill(no_leg.venue, no_leg.market_id, "NO", na, size)
            self.store.record_pnl(pnl, note="arb locked", event_key=opp.event_key)
            self.store.record_opportunity(opp, acted=True)
        self._audit("execute_success", opp, pnl=pnl)
        log.info("ARB LOCKED %s | pnl=%+.2f", opp.event_key, pnl)
        return ExecutionReport(ExecStatus.SUCCESS, "both legs filled", list(legs), pnl)

    async def _settle_partial_hedge(
        self, opp, maker_leg, maker_filled, hedged, hedge_notional, last_hedge, *, reason,
    ) -> ExecutionReport:
        """Couldn't fully hedge a maker fill after re-crossing. End flat-or-locked, never
        naked: SETTLE the matched (``hedged``) portion as a locked arb, then UNWIND the
        maker's unhedged excess (``maker_filled - hedged``). Returns UNWOUND with the
        combined pnl; if the unwind itself can't flatten, that path halts (correct — we
        genuinely can't get flat). The hedge leg has no excess: it's fully matched."""
        excess = round(maker_filled - hedged, 6)
        legs: list[OrderResult] = []
        pnl = 0.0
        if hedged > 1e-9:
            matched_hedge = replace(
                last_hedge, action="buy", requested=hedged, filled=hedged,
                status=OrderStatus.FILLED, avg_price=hedge_notional / hedged)
            matched_maker = replace(maker_leg, requested=hedged, filled=hedged)
            settled = self._settle_success(opp, hedged, [matched_maker, matched_hedge])
            legs += settled.legs
            pnl += settled.realized_pnl
        if excess > 1e-9:
            excess_leg = replace(maker_leg, requested=excess, filled=excess)
            unwound = await self._unwind(opp, excess_leg, None, reason=reason)
            if unwound.status is ExecStatus.HALTED:
                return unwound          # couldn't flatten the excess -> already halted
            legs += unwound.legs
            pnl += unwound.realized_pnl
        log.warning("PARTIAL-HEDGE %s: locked %g + unwound %g | pnl=%+.2f (%s)",
                    opp.event_key, hedged, excess, pnl, reason)
        return ExecutionReport(ExecStatus.UNWOUND, reason, legs, pnl)

    async def _unwind(self, opp, leg1, leg2, *, reason: str = "leg2 failed") -> ExecutionReport:
        """Sell the filled first leg back (IOC, slippage-tolerant) to return to flat.
        Used when the second leg cleanly fails and when the first leg partial-fills
        (``leg2`` is then ``None``). Works for whichever side leg 1 bought (the
        rejection-prone leg may be the NO leg when Kalshi is the NO venue)."""
        venue = self.venues[leg1.venue]
        side = leg1.side
        ref_price = opp.yes_price if side is Side.YES else opp.no_price
        buy_px = leg1.avg_price if leg1.avg_price is not None else ref_price
        # Cross the REAL best bid for that side so the IOC actually fills (a fixed haircut
        # off the buy price can sit above a thin book's bid and never fill -> stuck naked
        # + halt). YES bid = 1 - no_ask; NO bid = 1 - yes_ask. Floor at $0.01; fall back
        # to the haircut only if the book can't be read.
        sell_px = max(0.01, round(buy_px - self.unwind_slippage, 4))
        try:
            q = await venue.fetch_quote(RawMarket(market_id=leg1.market_id, title="", raw={}))
        except Exception as exc:
            log.warning("unwind book fetch failed for %s: %s", leg1.market_id, exc)
            q = None
        if q is not None:
            bid = None
            if side is Side.YES and q.no_ask is not None:
                bid = round(1.0 - q.no_ask, 4)
            elif side is Side.NO and q.yes_ask is not None:
                bid = round(1.0 - q.yes_ask, 4)
            if bid is not None:
                sell_px = max(0.01, min(sell_px, bid))    # take the bid, never above it
        unwind = await self._place(
            venue, leg1.market_id, side, "sell", sell_px, leg1.filled, "immediate_or_cancel"
        )
        log.warning("unwind %s", unwind)

        legs = [leg1] + ([leg2] if leg2 is not None else []) + [unwind]
        if not unwind.filled_fully:
            # Couldn't flatten -> we're still holding a one-sided position. This market is
            # illiquid enough that BOTH the hedge AND the unwind rejected — AUTO-BLACKLIST the
            # pair so the bot never re-fires it (it just stranded a naked leg) and a restart
            # comes up clean (the pair drops the watchlist -> reconcile treats the stuck leg as
            # benign untracked, not a re-halt). Then stop, surfacing both failure reasons.
            blacklisted = False
            if self.store is not None:
                try:
                    self.store.blacklist_pair(
                        opp.buy_yes_venue, opp.buy_yes_market,
                        opp.buy_no_venue, opp.buy_no_market,
                        reason=f"unwind failed (stuck naked leg): {reason}")
                    blacklisted = True
                    log.warning("auto-blacklisted %s after unwind failure (illiquid: hedge "
                                "AND unwind both rejected)", opp.event_key)
                except Exception as exc:
                    log.warning("auto-blacklist failed for %s: %s", opp.event_key, exc)
            # If we quarantined the market (blacklisted), the stuck leg is KNOWN, bounded, and
            # won't re-fire -> KEEP trading the rest of the book instead of freezing on it.
            # If blacklisting failed, fall back to a hard halt (fail closed).
            return self._halt(
                f"UNWIND FAILED ({reason}; unwind {unwind.status.value}: "
                f"{_reject_reason(unwind)}) — leg1 stranded, market "
                f"{'quarantined' if blacklisted else 'NOT blacklisted'}; manual reconcile",
                legs, trip=not blacklisted,
            )

        sell_avg = unwind.avg_price if unwind.avg_price is not None else sell_px
        pnl = leg1.filled * (sell_avg - buy_px) - self._fee(leg1.venue).fee(buy_px, leg1.filled)
        self.risk.record_pnl(pnl)
        # Net cash effect of buying leg1 then selling it back (a small loss).
        self._spend(leg1.venue, (buy_px - sell_avg) * leg1.filled)
        if self.store is not None:
            self.store.record_fill(leg1.venue, leg1.market_id, side.value, buy_px, leg1.filled)
            self.store.record_fill(
                unwind.venue, unwind.market_id, f"{side.value}_SELL", sell_avg, unwind.filled)
            self.store.record_pnl(pnl, note=f"unwind ({reason})", event_key=opp.event_key)
        self._audit("execute_unwound", opp, pnl=pnl)
        log.warning("UNWOUND %s (%s) | pnl=%+.2f", opp.event_key, reason, pnl)
        return ExecutionReport(ExecStatus.UNWOUND, f"{reason}; leg1 unwound", legs, pnl)

    def _halt(self, reason: str, legs: list[OrderResult], *, trip: bool = True) -> ExecutionReport:
        # ``trip=True`` (default): ambiguous/unknown state -> trip the global kill switch and
        # stop everything (fail closed). ``trip=False`` (QUARANTINE): the stuck leg is KNOWN
        # and bounded and its market is already blacklisted, so record it but KEEP trading the
        # rest of the book — freezing the whole bot for a tiny stranded thin-prop leg is
        # disproportionate, and the reconcile still catches any unexpected/larger naked.
        if trip:
            self.risk.trip_kill_switch(f"executor halt: {reason}")
        # Record what we were HOLDING so the stuck state isn't invisible in the books. The
        # SETTLED pnl is unknown — but the fills are factual and the CASH that moved is real.
        # Record each filled leg + a PROVISIONAL pnl row (the net cash outflow, conservatively
        # treating held legs as not-yet-recovered); the operator reconciles vs settlement.
        cash = 0.0
        for leg in legs or []:
            if leg is None or leg.filled <= 1e-9:
                continue
            price = leg.avg_price if leg.avg_price is not None else 0.0
            is_sell = getattr(leg, "action", "buy") == "sell"
            side = f"{leg.side.value}_SELL" if is_sell else leg.side.value
            cash += (price * leg.filled) if is_sell else -(price * leg.filled)
            if self.store is not None:
                self.store.record_fill(leg.venue, leg.market_id, side, price, leg.filled)
        tag = "HALT" if trip else "QUARANTINE"
        if self.store is not None:
            halt_key = "|".join(f"{l.venue}:{l.market_id}" for l in legs if l is not None) or None
            self.store.record_pnl(cash, note=f"{tag} provisional, unreconciled ({reason})",
                                  event_key=halt_key)
            self.store.audit("execute_halt" if trip else "execute_quarantine",
                             {"reason": reason, "cash": round(cash, 4),
                              "legs": [str(leg) for leg in legs]})
        log.critical("%s: %s | provisional cash %+.2f recorded for reconciliation",
                     tag, reason, cash)
        return ExecutionReport(ExecStatus.HALTED if trip else ExecStatus.QUARANTINED,
                               reason, legs, cash)

    def _audit(self, kind: str, opp: ArbOpportunity, **extra) -> None:
        if self.store is not None:
            self.store.audit(kind, {"event_key": opp.event_key, "opp": str(opp), **extra})
