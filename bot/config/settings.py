"""Typed settings loaded from environment variables (see ``.env.example``).

Standard-library only: a tiny ``.env`` parser plus dataclasses. No external config
library, so importing settings never drags in heavy dependencies.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from bot.execution.risk import RiskLimits
from bot.modes import RunMode

# Recognized fingerprint metrics (mirrors bot.matching.fingerprint). Used to validate
# MATCH_FINGERPRINT_METRICS so a malformed value can't silently filter out everything.
_VALID_FINGERPRINT_METRICS = frozenset(
    {"winner", "goals", "assists", "ga", "points", "saves", "shots"}
)


def _load_dotenv(path: str = ".env") -> None:
    """Populate ``os.environ`` from a ``.env`` file if present (without overriding
    values already set in the environment). Minimal ``KEY=VALUE`` parsing."""
    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if value[:1] in ('"', "'"):
            # Quoted: take content up to the matching quote, ignore any trailing comment.
            quote = value[0]
            end = value.find(quote, 1)
            value = value[1:end] if end != -1 else value[1:]
        elif value.startswith("#"):
            # The whole value is a comment (e.g. `KEY=# note`) -> empty value.
            value = ""
        else:
            # Unquoted: strip an inline comment (" #...").
            value = value.split(" #", 1)[0].rstrip()
        os.environ.setdefault(key, value)


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw in (None, ""):
        return default
    # Tolerate stray trailing text (e.g. an inline comment) on numeric values.
    return float(str(raw).split()[0])


@dataclass
class KalshiConfig:
    api_key_id: str = ""
    private_key_path: str = "secrets/kalshi_private_key.pem"
    api_base: str = "https://api.elections.kalshi.com/trade-api/v2"
    ws_base: str = "wss://api.elections.kalshi.com/trade-api/ws/v2"
    # Read request budget (req/min). Kalshi's basic tier allows ~10 reads/sec; 240 (4/s)
    # is a safe default well under that. Raise toward your account's real tier for a faster
    # watchlist refresh; lower it if you see HTTP 429s.
    read_rate_per_min: float = 240.0


@dataclass
class QcexConfig:
    # Reads use the PUBLIC gateway (no key). Trading uses the authenticated API.
    gateway_base: str = "https://gateway.polymarket.us"   # public market data
    api_base: str = "https://api.polymarket.us"           # authenticated trading
    ws_markets: str = "wss://api.polymarket.us/v1/ws/markets"
    ws_private: str = "wss://api.polymarket.us/v1/ws/private"
    # Trading credentials (only needed to place orders, not for the dry run):
    api_key_id: str = ""
    secret_key: str = ""                                   # raw base64 secret (preferred)
    ed25519_private_key_path: str = "secrets/qcex_ed25519.pem"  # PEM fallback
    use_sandbox: bool = False
    # Read request budget (req/min) for the public gateway. 300 (5/s) is a safe default;
    # raise toward the gateway's real limit for a faster refresh, lower it on HTTP 429s.
    read_rate_per_min: float = 300.0
    # Liquidity gate: drop markets whose 24h traded volume is below this from the scan. Real
    # volume (NOT the phantom /book depth) is the reliable signal for "the hedge will fill" —
    # thin markets are where Polymarket 500s the hedge and a resting maker goes naked. 0 = off.
    min_volume_24h: float = 0.0

    @property
    def is_trading_configured(self) -> bool:
        """True once order-placement credentials are present (key id + a secret)."""
        from pathlib import Path

        has_secret = bool(self.secret_key) or (
            bool(self.ed25519_private_key_path)
            and Path(self.ed25519_private_key_path).expanduser().exists()
        )
        return bool(self.api_key_id) and has_secret


@dataclass
class LLMConfig:
    base_url: str = "http://localhost:8000/v1"
    reasoning_model: str = "qwen2.5-instruct"
    embedding_model: str = "nomic-embed-text"


@dataclass
class Settings:
    run_mode: RunMode = RunMode.DRY_RUN
    kalshi: KalshiConfig = field(default_factory=KalshiConfig)
    qcex: QcexConfig = field(default_factory=QcexConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    risk: RiskLimits = field(default_factory=RiskLimits)
    db_path: str = "data/bot.db"
    # Startup reconciliation guard: minimum balance required per trading venue, and
    # whether to allow pre-existing positions (default: refuse to trade if not flat).
    startup_min_balance: float = 0.0
    startup_allow_positions: bool = False
    # When true, the per-market and total exposure caps are set from the live balance
    # check (sum of funded venue balances) at startup and each refresh — so "balances
    # are the limit" without hardcoding dollar caps. Overrides RISK_MAX_* exposure caps.
    risk_caps_from_balance: bool = False
    # When true, the streaming watchlist is gated by the structured fingerprint matcher
    # (are_complementary) instead of the series/title allowlists. Optionally restrict to
    # specific metrics (e.g. "winner") for a staged rollout; empty = all matchable.
    match_use_fingerprint: bool = False
    # When true AND match_use_fingerprint is on, the watchlist is the UNION of the
    # deterministic fingerprint sweep and the LLM match_verdicts cache (max recall: a pair
    # either matcher finds is admitted; a shared fan-out backstop drops pairs the two
    # disagree on). Requires STREAM_DISCOVERY=true so the verdict half stays populated.
    match_combine_verdicts: bool = False
    match_fingerprint_metrics: frozenset = field(default_factory=frozenset)
    # Fingerprint sweep recency: drop markets whose EVENT date is more than this many days
    # in the past. Settled games linger in the markets table (never pruned); without this
    # the sweep resurfaces every long-settled match. Uses the fingerprint's parsed event
    # date (not updated_at), so it's independent of scan cadence. 1 day keeps today's and
    # in-progress games. 0 = disabled (sweep regardless of event date).
    match_sweep_past_days: float = 1.0
    # Per-cycle cross-venue discovery (embedding shortlist + LLM confirm -> match_verdicts).
    # When the fingerprint sweep is authoritative (MATCH_USE_FINGERPRINT), the streamer
    # builds its watchlist directly from the scanned markets and IGNORES match_verdicts, so
    # this discovery pass is dead weight on the live path (it only feeds --inspect-matches /
    # --compare-filters). Off = skip it (faster cycles, no embedding/LLM calls); the scan
    # still populates the markets table the sweep needs.
    stream_discovery: bool = True
    # Targeted scan: only scan markets closing within this many days (the live/imminent
    # set), so today's games are always covered without a huge --limit (the matcher
    # embeds every scanned title, so scan size is the cost). 0 = disabled (scan all).
    scan_close_within_days: float = 0.0
    # KALSHI-ONLY targeted window: Kalshi has ~61k open markets (vs the ~5000 --limit cap),
    # and the first 5000 are dominated by far-future noise (midterm-election series etc.), so
    # ~73% of its per-game/match lines sit past the cap and never reach the matcher. Kalshi's
    # per-game tickers close IMMINENTLY (at game time), so a close-time window captures them
    # without scanning all 61k. Applied to KALSHI ONLY — Polymarket's per-game markets carry
    # far-future endDates, so a window would drop them (hence the separate, Kalshi-scoped knob).
    # 0 = disabled (fall back to the shared --limit/scan_close_within_days behavior).
    scan_kalshi_close_within_days: float = 0.0
    # KALSHI-ONLY per-game pattern allowlist (the memory-safe coverage fix). The matcher
    # embeds every scanned title, so memory tracks scan size. Kalshi's 61k board is mostly
    # non-arbable (elections/crypto/streaming); the arbable head-to-head markets are
    # identifiable by ticker substring (GAME/MATCH/FIGHT/quarter-WINNER/World-Cup props).
    # Scanning UNBOUNDED but keeping only tickers containing one of these returns ~3.2k
    # markets (FEWER embeds than the 5000 baseline) while covering ~3.4x the per-game lines
    # AND every currently-confirmed pair. Empty = disabled. Applied to KALSHI only.
    kalshi_scan_patterns: tuple[str, ...] = ()
    # Non-sports scan expansion: extra Kalshi allowlist patterns for BINARY winner markets
    # (elections/awards/macro/finance) unioned into kalshi_scan_patterns. Kept separate so
    # it's toggleable independently. Empty = sports only.
    kalshi_nonsport_patterns: tuple[str, ...] = ()
    # Scan deny-list: drop tickers containing any of these even if allowlisted — kills the
    # multi-outcome place/rank/spread bulk (KXPRIMARYPLACE etc.) a winner-pattern would
    # otherwise admit, so non-sports stays binary-only. Empty = no deny filter.
    kalshi_scan_deny: tuple[str, ...] = ()
    # Canonicalize-then-join matching (see bot/matching/canon.py): per-market cached LLM
    # extraction of the canonical contract from its RESOLUTION RULES, matched by a
    # deterministic join — the domain-agnostic third member of the watchlist union.
    # Only takes effect as extractions accumulate (the canon loop budgets a few per pass).
    match_use_canon: bool = True
    # Deterministic matching (MATCH_DETERMINISTIC): discovery uses the id-parse join
    # (bot/matching/idparse.py) and DROPS embeddings + per-pair LLM confirms entirely.
    # The rules-verify LLM stays (safety layer). Legacy verdict/canon caches remain
    # valid union members. Shadow-validated before enabling (bot.dryrun --match-shadow).
    match_deterministic: bool = False
    match_idparse_interval: float = 600.0  # seconds between deterministic join runs
    stream_require_rules_verify: bool = True  # pairs may not TRADE before a rules-LLM pass
    stream_ws_trust_min: int = 3      # consecutive honest confirms before WS fires unconfirmed (0=off)
    stream_ws_trust_eps: float = 0.01  # rest edge may lag the WS claim by this and stay "honest"
    # Max markets to pull per venue per scan (TOTAL across paginated pages). 0 = scan the
    # ENTIRE board (every available market), not just a page. Used by the streaming loop;
    # the matched/tradeable set is still bounded by Polymarket's small universe, so this
    # mainly governs how much of Kalshi is searched for a counterpart.
    scan_limit: int = 0
    # Liquidity guard: min resting depth (contracts) required on BOTH legs before the
    # executor fires. Thin books are where a leg rejects and can't be hedged/unwound;
    # skipping them means we should never have to unwind. 0 = disabled.
    exec_min_leg_depth: float = 0.0
    # Trust the WS live book (and skip the per-fire REST depth re-fetch) when both legs
    # have a sized quote no older than this many seconds. Both venues now stream a sized
    # top-of-book, so this removes the round-trip on the hot path. 0 = always re-fetch.
    stream_max_ws_quote_age: float = 2.0
    # Fraction of shown top-of-book depth to actually trade — headroom so a fill-or-kill
    # leg still fills if the book thins between the quote and the order (the cause of
    # Kalshi's "insufficient resting volume" rejections). 1.0 = use the full shown depth.
    exec_depth_fraction: float = 0.85
    # Fraction of the LIVE hedge top-of-book (re-read right before firing) we'll actually
    # commit the FOK to — a DEEP cushion so a partial vanish between the read and the order
    # can't reject it (the dominant unwind cause). 0.5 = take half the shown hedge depth ->
    # the book must lose >50% in ~tens of ms to reject. Deep books (depth >> size) are
    # unaffected (the fraction still exceeds the trade size); only thin hedges size down.
    # 1.0 = commit the full shown depth (old behavior, no extra cushion).
    exec_hedge_depth_fraction: float = 0.5
    # Skip a fire if either leg's ask is at a price extreme (<= this or >= 1 - this).
    # A binary at ~$0.01 is a settling/resolved market with phantom depth (no real
    # resting volume), so its "edge" is an artifact. 0 = disabled.
    stream_min_leg_price: float = 0.02
    # How many of the best edges the periodic snapshot logs each refresh interval.
    stream_edge_snapshot_top: int = 5
    # Require an edge to persist continuously this many seconds before acting. A cross-feed
    # timing artifact (one venue's WS leading the other) flickers and is gone in ~100ms; a
    # real venue-lag edge holds for seconds. So this filters skew-phantoms without a slow
    # REST confirm. 0 = act on first sighting (off). ~0.75 is a reasonable in-play value.
    stream_edge_persist_secs: float = 0.0
    # Sub-100ms sync path: if BOTH legs tick within this tight window (synced cross-feed,
    # not a one-sided flicker), a deep edge fires WITHOUT the persist wait. ~0.05 (50ms) is
    # a reasonable value; 0 = disabled (always use the persist path).
    stream_sync_window_secs: float = 0.0
    # Streaming slow-loop cadence (s): how often the watchlist is rebuilt + re-primed. Lower
    # = newly-listed markets enter the watchlist sooner, at the cost of more REST traffic.
    # The prime is now concurrent (below), so a lower value is feasible.
    stream_refresh_secs: float = 300.0
    # How often to re-read venue balances/positions, INDEPENDENT of the (slow, ~minutes)
    # discovery cycle. The cache otherwise only refreshed at the end of each discovery pass,
    # so a fresh deposit took minutes to register -> spurious low-balance skips after a
    # refill. A short tick picks up deposits/settlements (and re-runs the naked-leg
    # reconcile) within seconds. 0 = off (refresh only on the discovery cycle).
    stream_balance_refresh_secs: float = 30.0
    # Max concurrent REST snapshots when re-priming the watchlist each cycle (the rest are
    # paced by the per-venue rate limiters). Higher = faster refresh (seconds vs >a minute).
    stream_prime_concurrency: int = 8
    # Watchlist depth filter: drop pairs whose Polymarket (hedge-bottleneck) leg shows less
    # than this much top-of-book size, so the bot watches only markets it can actually
    # hedge. Costs one Polymarket /book probe per pair each refresh. 0 = off (watch all).
    stream_min_poly_depth: float = 0.0
    # Price cushion reserved for the hedge (second) leg so it fills through book
    # movement WITHOUT unwinding. The bot only fires when the edge can pay this AND
    # still lock RISK_MIN_EDGE, so thin edges that would unwind never fire. Effective
    # firing threshold = RISK_MIN_EDGE + this. 0 = off (chase thin edges, may unwind).
    # Ignored in maker mode (the maker captures the spread, so thin edges need no buffer).
    exec_hedge_buffer: float = 0.03
    # Which venue fires FIRST in a two-FOK hybrid TAKE (the rejection-prone one, so its
    # failure is a clean skip not an unwind). "" = same as the maker-rest venue. Set to
    # "polymarket_us" so its 500s on thin markets become free skips instead of Kalshi unwinds.
    exec_take_first_venue: str = ""
    # Fat-edge evidence threshold (NOT a hard ceiling): an edge above this is usually a
    # false same-event match (legs not complements), so it must be backed by STRONGER
    # empirical proof — ~3x the sum-observation samples with a mean YES+NO >= ~0.97 —
    # before it may fire. A proven complement fires at ANY edge (a genuine dislocation);
    # an unproven pair keeps observing and blacklists on evidence. If the empirical gate
    # is disabled (MATCH_EMPIRICAL_MIN_OBS=0) this falls back to a hard skip. 0 = off.
    exec_max_plausible_edge: float = 0.0
    # Empirical same-event confirmation: a pair must show >= this many YES+NO-sum samples
    # whose MEAN is >= match_empirical_sum_floor before it can TRADE (price behavior is the
    # authority, not the structural/LLM match). 0 = disabled. Floor ~0.93 separates real
    # arbs (mean ~1.0) from false matches (R6-vs-CoD mean ~0.68).
    match_empirical_min_obs: int = 0
    match_empirical_sum_floor: float = 0.93
    # Maker-volume gate: a market may host a resting MAKER only if its 24h volume clears this
    # (thin hedges 500 -> naked maker). Thin markets stay TAKE-able (a 500 there is a clean
    # skip via the leg-order fix). 0 = no gate. Distinct from QCEX_MIN_VOLUME_24H (universe).
    exec_maker_min_volume_24h: float = 0.0
    # Min venue balance ($) to fire a leg there: a drained venue can't fund its hedge ->
    # naked. Below this, that leg's trades are skipped (self-healing). 0 = off.
    exec_min_venue_balance: float = 0.0
    # Capital-scarcity edge gate: when a venue's spendable cash falls below
    # exec_scarcity_balance, RESERVE it for the fattest edges rather than locking the last
    # dollars into a thin (1c ~= 1%/cycle) arb. Below the floor, only edges >=
    # exec_scarcity_min_edge fire. Keeps scarce capital flowing to the best returns instead
    # of FIFO. 0 balance = off (no scarcity gate).
    exec_scarcity_balance: float = 0.0
    exec_scarcity_min_edge: float = 0.02
    # Edge-weighted capital budget: the per-contract edge at which a fire may use the FULL
    # spendable balance; thinner edges get edge/this (floored below) — so the bankroll isn't
    # FIFO-locked into small-pnl trades while thin edges still trade smaller. 0 = off.
    exec_edge_full_budget: float = 0.0
    exec_edge_budget_floor: float = 0.25
    # Fresh-hedge fast path: skip the hedge REST re-read (the only network hop on the fire
    # path, ~40ms) when the opp's WS quotes are younger than this AND the hedge leg's WS
    # depth is >= 2x the trade size. 0 = always re-read.
    exec_fresh_hedge_secs: float = 0.0
    # Breakeven recross: when a hedge FOK fails, re-take it at up to breakeven + this
    # before unwinding leg 1 (unwinding is a guaranteed spread+slippage loss; a ~$0 lock
    # or an epsilon loss strictly dominates it).
    exec_recross_epsilon: float = 0.02
    # ---- Capital recycler (auto-rebalance v2) ----
    # Cross-venue cash transfer can't be automated, but hedged pairs ARE portable
    # capital: a locked pair whose event is effectively decided can be EARLY-EXITED —
    # sell the ITM leg at its bid (recovers ~0.9x/contract on the drained venue NOW),
    # then the cheap OTM leg (unsold = a free upset-hedge remnant). Bounded give-up vs
    # waiting days for settlement. Armed when a venue's REAL cash < recycle_floor AND
    # the other venue holds >= 3x its cash. 0 = off.
    exec_recycle_floor: float = 0.0
    exec_recycle_itm_bid: float = 0.90      # candidate pre-filter; max_cost is the gate
    exec_recycle_max_cost: float = 0.03     # max give-up/contract vs $1, incl. sell fees
    exec_recycle_max_contracts: float = 50.0  # blast-radius bound per pass
    exec_recycle_target: float = 0.0        # stop once drained cash >= this; 0 -> 2x floor
    exec_recycle_interval_secs: float = 90.0
    exec_recycle_cooldown_secs: float = 300.0   # between passes that placed orders
    exec_recycle_pair_cooldown_secs: float = 3600.0  # rebuy guard (fee-churn loop)
    exec_recycle_max_settle_days: float = 3.0  # skip recycling far-dated undecided favorites
    exec_recycle_decided_bid: float = 0.98     # far-dated allowed only if this certain
    exec_recycle_min_settle_hours: float = 24.0  # settling sooner than this -> ride it out
    # Structural-imbalance alert: drained + nothing recyclable for this long -> a loud
    # log.critical telling the operator the exact manual transfer to make. 0 = off.
    exec_imbalance_alert_secs: float = 900.0
    # ---- Early-profit exit (generalizes the recycler) ----
    # Realize a hedged pair's locked profit BEFORE settlement whenever the two venues
    # dislocate favorably (both exit bids recover >= entry cost + margin). Never exits
    # below entry, so a quiet pair stays held. Frees capital months early on long-dated
    # markets. 0 margin = exit at breakeven-vs-entry; >0 requires real profit.
    exec_early_exit_enabled: bool = False
    exec_early_exit_margin: float = 0.0
    exec_early_exit_interval_secs: float = 300.0
    exec_early_exit_cooldown_secs: float = 300.0
    exec_early_exit_max_pairs: float = 8.0
    exec_early_exit_max_contracts: float = 50.0
    exec_early_exit_min_bid_depth: float = 0.0    # both legs need >= this sellable depth
    exec_early_exit_min_settle_days: float = 3.0  # only unwind pairs locked >= this long
    # Capital-horizon gate: reject entries settling beyond this many days unless the edge
    # clears exec_longdated_min_edge. Keeps thin edges from locking cash for months. 0=off.
    exec_max_settle_days: float = 0.0
    exec_longdated_min_edge: float = 0.0
    # Venue auto-balancing: when a venue's cash dips below exec_rebalance_floor, skip arbs
    # whose leg on THAT venue is the expensive (> $0.50) side, so new spend shifts to the
    # funded venue and the scarce side's cash lasts until settlements replenish it. Same edge
    # captured, just allocated to keep both venues fundable (cuts the "can't-fund" idle skips
    # from one-way draining). 0 = off.
    exec_rebalance_floor: float = 0.0
    # Empirical fill-reliability (probe-then-scale): the live, learned replacement for the
    # volume proxy. Untested markets trade at exec_probe_contracts until their FOK orders
    # have FILLED exec_market_proven_fills times (real depth proven), then scale to full
    # size; a market that KILL/REJECTs exec_market_max_fails times without proving is
    # excluded. 0 probe = disabled (no empirical gate).
    exec_probe_contracts: float = 0.0
    exec_market_proven_fills: int = 3
    exec_market_max_fails: int = 2
    # Once a market is proven real, the sizer allows up to this x the LARGEST size a FOK has
    # actually filled there — a fast geometric scale-up on demonstrated depth (vs the old slow
    # +1-per-fill ramp that lost edge), bounded by what's been proven so a phantom-at-size book
    # can't strand a big naked leg.
    exec_market_ramp_factor: float = 3.0
    # Maker mode: capture THIN edges by RESTING the fee-heavy (Kalshi) leg as a maker
    # (no slippage / lower fee), then TAKING the deep (Polymarket) leg the instant it
    # fills. The firing threshold drops to just RISK_MIN_EDGE (no hedge buffer needed).
    exec_maker_mode: bool = False
    # How long (s) a resting maker may wait to fill before it self-expires (no trade).
    exec_maker_timeout: float = 5.0
    # How far inside the ask to post the maker (>= one tick, so post-only doesn't reject
    # it as crossing). Also captures this much extra edge on a fill. Kalshi tick = $0.01.
    exec_maker_improvement: float = 0.01
    # Maker arming cushion: a resting maker is exposed to the TAKER leg drifting against
    # it while it waits to fill (adverse selection — it tends to fill exactly when the
    # market moves). If the taker drifts past the edge before the maker fills, the forced
    # hedge locks a guaranteed loss. So require this much edge ABOVE the lock floor before
    # arming a maker — a drift cushion analogous to the taker path's hedge buffer. Set it
    # at/above the typical taker drift over the maker timeout. 0 = no cushion (unsafe:
    # arms 1-2c edges that don't survive the rest window).
    exec_maker_arm_cushion: float = 0.04
    # While a maker rests, re-poll the taker leg every this-many seconds and CANCEL the
    # maker if the taker drifts so the hedge can no longer lock the edge floor — so a
    # resting maker never fills into an adverse move. This is the dynamic guard that makes
    # arming thin edges safe; with it on you can lower EXEC_MAKER_ARM_CUSHION. 0 = disabled
    # (rest blindly until fill/expiry). Smaller = tighter (less fill-before-cancel race) but
    # more REST polls per resting maker.
    exec_maker_poll: float = 1.0
    # Hybrid take-or-rest (maker mode only): when a depth-confirmed edge has at least this
    # many contracts of real top-of-book size AND clears the taker bar (RISK_MIN_EDGE +
    # EXEC_HEDGE_BUFFER), TAKE it immediately (cross both books, lock now) instead of
    # resting a maker that may never get crossed. Deep-but-thin edges (below the taker bar)
    # and shallow edges still rest a maker. 0 = disabled (pure maker mode, never auto-takes).
    exec_hybrid_take_depth: float = 0.0
    # After a maker fills, retry the forced hedge this many times on a TRANSIENT venue
    # error (e.g. a Polymarket 500/timeout) before unwinding — each retry only after
    # reconciling the hedge venue to confirm nothing landed (so it can't double up).
    exec_hedge_retries: int = 1
    # Periodic cross-venue reconciliation: each cycle, compare the two legs of every pair;
    # an imbalance is naked exposure (a hedge that never landed). True trips the kill switch
    # when a naked position persists across two checks; False only warns. Always logs.
    exec_reconcile_halt: bool = True
    # Dynamic maker side: rest the maker on whichever leg is the liquidity bottleneck (the
    # thinner book) and take the deeper leg, instead of always resting on Kalshi. Lets a
    # thin Polymarket leg be sourced via its own flow as a maker rather than skipped. Needs
    # EXEC_MAKER_MODE=true. False = always rest on Kalshi (the original behavior).
    exec_maker_dynamic: bool = False
    # Depth-scaled hedge buffer: at/above this book depth (contracts) the hedge fills at the
    # touch, so the required hedge buffer scales down to one tick — letting the bot fire on
    # the smaller, more frequent divergence windows on DEEP markets while keeping the full
    # buffer on thin books. Set near the depth where fills are reliable (e.g. 1000 given the
    # major markets are 50k-500k deep). 0 = off (always the full EXEC_HEDGE_BUFFER).
    exec_buffer_deep_depth: float = 0.0


def load_settings(dotenv_path: str = ".env") -> Settings:
    _load_dotenv(dotenv_path)
    env = os.environ.get

    return Settings(
        run_mode=RunMode.parse(env("BOT_RUN_MODE")),
        kalshi=KalshiConfig(
            api_key_id=env("KALSHI_API_KEY_ID", "") or "",
            private_key_path=env("KALSHI_API_PRIVATE_KEY_PATH", KalshiConfig.private_key_path),
            api_base=env("KALSHI_API_BASE", KalshiConfig.api_base),
            ws_base=env("KALSHI_WS_BASE", KalshiConfig.ws_base),
            read_rate_per_min=_env_float("KALSHI_READ_RATE_PER_MIN", KalshiConfig.read_rate_per_min),
        ),
        qcex=QcexConfig(
            gateway_base=env("QCEX_GATEWAY_BASE", QcexConfig.gateway_base),
            api_base=env("QCEX_API_BASE", QcexConfig.api_base),
            ws_markets=env("QCEX_WS_MARKETS", QcexConfig.ws_markets),
            ws_private=env("QCEX_WS_PRIVATE", QcexConfig.ws_private),
            api_key_id=env("QCEX_API_KEY_ID", "") or "",
            secret_key=env("QCEX_SECRET_KEY", "") or "",
            ed25519_private_key_path=env(
                "QCEX_ED25519_PRIVATE_KEY_PATH", QcexConfig.ed25519_private_key_path
            ),
            use_sandbox=(env("QCEX_USE_SANDBOX", "false") or "false").lower() == "true",
            read_rate_per_min=_env_float("QCEX_READ_RATE_PER_MIN", QcexConfig.read_rate_per_min),
            min_volume_24h=_env_float("QCEX_MIN_VOLUME_24H", 0.0),
        ),
        llm=LLMConfig(
            base_url=env("LLM_BASE_URL", LLMConfig.base_url),
            reasoning_model=env("LLM_REASONING_MODEL", LLMConfig.reasoning_model),
            embedding_model=env("LLM_EMBEDDING_MODEL", LLMConfig.embedding_model),
        ),
        risk=RiskLimits(
            max_position_per_market=_env_float("RISK_MAX_POSITION_PER_MARKET", 200.0),
            max_total_exposure=_env_float("RISK_MAX_TOTAL_EXPOSURE", 5000.0),
            max_daily_loss=_env_float("RISK_MAX_DAILY_LOSS", 500.0),
            min_edge=_env_float("RISK_MIN_EDGE", 0.01),
            max_order_contracts=_env_float("RISK_MAX_ORDER_CONTRACTS", 2.0),
            max_position_fraction=_env_float("RISK_MAX_POSITION_FRACTION", 1.0),
        ),
        db_path=env("BOT_DB_PATH", "data/bot.db"),
        startup_min_balance=_env_float("STARTUP_MIN_BALANCE", 0.0),
        startup_allow_positions=(
            env("STARTUP_ALLOW_POSITIONS", "false") or "false"
        ).lower() == "true",
        risk_caps_from_balance=(
            env("RISK_CAPS_FROM_BALANCE", "false") or "false"
        ).lower() == "true",
        match_use_fingerprint=(
            env("MATCH_USE_FINGERPRINT", "false") or "false"
        ).lower() == "true",
        match_combine_verdicts=(
            env("MATCH_COMBINE_VERDICTS", "false") or "false"
        ).lower() == "true",
        match_deterministic=(
            env("MATCH_DETERMINISTIC", "false") or "false"
        ).lower() == "true",
        match_idparse_interval=_env_float("MATCH_IDPARSE_INTERVAL", 600.0),
        stream_require_rules_verify=(
            env("STREAM_REQUIRE_RULES_VERIFY", "true") or "true"
        ).lower() == "true",
        stream_ws_trust_min=int(_env_float("STREAM_WS_TRUST_MIN", 3)),
        stream_ws_trust_eps=_env_float("STREAM_WS_TRUST_EPS", 0.01),
        match_use_canon=(
            env("MATCH_USE_CANON", "true") or "true"
        ).lower() == "true",
        # Keep ONLY recognized metric names — so a malformed value (e.g. an inline
        # comment captured as the value, or a stray token) degrades to "all matchable"
        # instead of silently filtering out every real metric.
        match_fingerprint_metrics=frozenset(
            tok for raw in (env("MATCH_FINGERPRINT_METRICS", "") or "").split(",")
            if (tok := raw.strip().lower()) in _VALID_FINGERPRINT_METRICS
        ),
        scan_close_within_days=_env_float("SCAN_CLOSE_WITHIN_DAYS", 0.0),
        scan_kalshi_close_within_days=_env_float("SCAN_KALSHI_CLOSE_WITHIN_DAYS", 0.0),
        kalshi_scan_patterns=tuple(
            p.strip().upper() for p in os.getenv("KALSHI_SCAN_PATTERNS", "").split(",") if p.strip()
        ),
        kalshi_nonsport_patterns=tuple(
            p.strip().upper() for p in os.getenv("KALSHI_NONSPORT_PATTERNS", "").split(",") if p.strip()
        ),
        kalshi_scan_deny=tuple(
            p.strip().upper() for p in os.getenv("KALSHI_SCAN_DENY", "").split(",") if p.strip()
        ),
        scan_limit=int(_env_float("SCAN_LIMIT", 0)),
        match_sweep_past_days=_env_float("MATCH_SWEEP_PAST_DAYS", 1.0),
        stream_discovery=(env("STREAM_DISCOVERY", "true") or "true").lower() == "true",
        exec_min_leg_depth=_env_float("EXEC_MIN_LEG_DEPTH", 0.0),
        stream_max_ws_quote_age=_env_float("STREAM_MAX_WS_QUOTE_AGE", 2.0),
        exec_depth_fraction=_env_float("EXEC_DEPTH_FRACTION", 0.85),
        exec_hedge_depth_fraction=_env_float("EXEC_HEDGE_DEPTH_FRACTION", 0.5),
        stream_min_leg_price=_env_float("STREAM_MIN_LEG_PRICE", 0.02),
        stream_edge_snapshot_top=int(_env_float("STREAM_EDGE_SNAPSHOT_TOP", 5)),
        stream_edge_persist_secs=_env_float("STREAM_EDGE_PERSIST_SECS", 0.0),
        stream_sync_window_secs=_env_float("STREAM_SYNC_WINDOW_SECS", 0.0),
        stream_refresh_secs=_env_float("STREAM_REFRESH_SECS", 300.0),
        stream_balance_refresh_secs=_env_float("STREAM_BALANCE_REFRESH_SECS", 30.0),
        stream_prime_concurrency=int(_env_float("STREAM_PRIME_CONCURRENCY", 8)),
        stream_min_poly_depth=_env_float("STREAM_MIN_POLY_DEPTH", 0.0),
        exec_hedge_buffer=_env_float("EXEC_HEDGE_BUFFER", 0.03),
        exec_take_first_venue=(env("EXEC_TAKE_FIRST_VENUE", "") or "").strip(),
        exec_max_plausible_edge=_env_float("EXEC_MAX_PLAUSIBLE_EDGE", 0.0),
        match_empirical_min_obs=int(_env_float("MATCH_EMPIRICAL_MIN_OBS", 0)),
        match_empirical_sum_floor=_env_float("MATCH_EMPIRICAL_SUM_FLOOR", 0.93),
        exec_maker_min_volume_24h=_env_float("EXEC_MAKER_MIN_VOLUME_24H", 0.0),
        exec_min_venue_balance=_env_float("EXEC_MIN_VENUE_BALANCE", 0.0),
        exec_scarcity_balance=_env_float("EXEC_SCARCITY_BALANCE", 0.0),
        exec_scarcity_min_edge=_env_float("EXEC_SCARCITY_MIN_EDGE", 0.02),
        exec_edge_full_budget=_env_float("EXEC_EDGE_FULL_BUDGET", 0.0),
        exec_edge_budget_floor=_env_float("EXEC_EDGE_BUDGET_FLOOR", 0.25),
        exec_fresh_hedge_secs=_env_float("EXEC_FRESH_HEDGE_SECS", 0.0),
        exec_recross_epsilon=_env_float("EXEC_RECROSS_EPSILON", 0.02),
        exec_recycle_floor=_env_float("EXEC_RECYCLE_FLOOR", 0.0),
        exec_recycle_itm_bid=_env_float("EXEC_RECYCLE_ITM_BID", 0.90),
        exec_recycle_max_cost=_env_float("EXEC_RECYCLE_MAX_COST", 0.03),
        exec_recycle_max_contracts=_env_float("EXEC_RECYCLE_MAX_CONTRACTS", 50.0),
        exec_recycle_target=_env_float("EXEC_RECYCLE_TARGET", 0.0),
        exec_recycle_interval_secs=_env_float("EXEC_RECYCLE_INTERVAL_SECS", 90.0),
        exec_recycle_cooldown_secs=_env_float("EXEC_RECYCLE_COOLDOWN_SECS", 300.0),
        exec_recycle_pair_cooldown_secs=_env_float("EXEC_RECYCLE_PAIR_COOLDOWN_SECS", 3600.0),
        exec_recycle_max_settle_days=_env_float("EXEC_RECYCLE_MAX_SETTLE_DAYS", 3.0),
        exec_recycle_decided_bid=_env_float("EXEC_RECYCLE_DECIDED_BID", 0.98),
        exec_recycle_min_settle_hours=_env_float("EXEC_RECYCLE_MIN_SETTLE_HOURS", 24.0),
        exec_imbalance_alert_secs=_env_float("EXEC_IMBALANCE_ALERT_SECS", 900.0),
        exec_early_exit_enabled=(env("EXEC_EARLY_EXIT_ENABLED", "false") or "false").lower() == "true",
        exec_early_exit_margin=_env_float("EXEC_EARLY_EXIT_MARGIN", 0.0),
        exec_early_exit_interval_secs=_env_float("EXEC_EARLY_EXIT_INTERVAL_SECS", 300.0),
        exec_early_exit_cooldown_secs=_env_float("EXEC_EARLY_EXIT_COOLDOWN_SECS", 300.0),
        exec_early_exit_max_pairs=_env_float("EXEC_EARLY_EXIT_MAX_PAIRS", 8.0),
        exec_early_exit_max_contracts=_env_float("EXEC_EARLY_EXIT_MAX_CONTRACTS", 50.0),
        exec_early_exit_min_bid_depth=_env_float("EXEC_EARLY_EXIT_MIN_BID_DEPTH", 0.0),
        exec_early_exit_min_settle_days=_env_float("EXEC_EARLY_EXIT_MIN_SETTLE_DAYS", 3.0),
        exec_max_settle_days=_env_float("EXEC_MAX_SETTLE_DAYS", 0.0),
        exec_longdated_min_edge=_env_float("EXEC_LONGDATED_MIN_EDGE", 0.0),
        exec_rebalance_floor=_env_float("EXEC_REBALANCE_FLOOR", 0.0),
        exec_probe_contracts=_env_float("EXEC_PROBE_CONTRACTS", 0.0),
        exec_market_proven_fills=int(_env_float("EXEC_MARKET_PROVEN_FILLS", 3)),
        exec_market_max_fails=int(_env_float("EXEC_MARKET_MAX_FAILS", 2)),
        exec_market_ramp_factor=_env_float("EXEC_MARKET_RAMP_FACTOR", 3.0),
        exec_maker_mode=(env("EXEC_MAKER_MODE", "false") or "false").lower() == "true",
        exec_maker_timeout=_env_float("EXEC_MAKER_TIMEOUT", 5.0),
        exec_maker_improvement=_env_float("EXEC_MAKER_IMPROVEMENT", 0.01),
        exec_maker_arm_cushion=_env_float("EXEC_MAKER_ARM_CUSHION", 0.04),
        exec_maker_poll=_env_float("EXEC_MAKER_POLL", 1.0),
        exec_hybrid_take_depth=_env_float("EXEC_HYBRID_TAKE_DEPTH", 0.0),
        exec_hedge_retries=int(_env_float("EXEC_HEDGE_RETRIES", 1)),
        exec_reconcile_halt=(env("EXEC_RECONCILE_HALT", "true") or "true").lower() == "true",
        exec_maker_dynamic=(env("EXEC_MAKER_DYNAMIC", "false") or "false").lower() == "true",
        exec_buffer_deep_depth=_env_float("EXEC_BUFFER_DEEP_DEPTH", 0.0),
    )
