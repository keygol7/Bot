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
    match_fingerprint_metrics: frozenset = field(default_factory=frozenset)
    # Targeted scan: only scan markets closing within this many days (the live/imminent
    # set), so today's games are always covered without a huge --limit (the matcher
    # embeds every scanned title, so scan size is the cost). 0 = disabled (scan all).
    scan_close_within_days: float = 0.0
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
    # Skip a fire if either leg's ask is at a price extreme (<= this or >= 1 - this).
    # A binary at ~$0.01 is a settling/resolved market with phantom depth (no real
    # resting volume), so its "edge" is an artifact. 0 = disabled.
    stream_min_leg_price: float = 0.02
    # Price cushion reserved for the hedge (second) leg so it fills through book
    # movement WITHOUT unwinding. The bot only fires when the edge can pay this AND
    # still lock RISK_MIN_EDGE, so thin edges that would unwind never fire. Effective
    # firing threshold = RISK_MIN_EDGE + this. 0 = off (chase thin edges, may unwind).
    # Ignored in maker mode (the maker captures the spread, so thin edges need no buffer).
    exec_hedge_buffer: float = 0.03
    # Maker mode: capture THIN edges by RESTING the fee-heavy (Kalshi) leg as a maker
    # (no slippage / lower fee), then TAKING the deep (Polymarket) leg the instant it
    # fills. The firing threshold drops to just RISK_MIN_EDGE (no hedge buffer needed).
    exec_maker_mode: bool = False
    # How long (s) a resting maker may wait to fill before it self-expires (no trade).
    exec_maker_timeout: float = 5.0
    # How far inside the ask to post the maker (>= one tick, so post-only doesn't reject
    # it as crossing). Also captures this much extra edge on a fill. Kalshi tick = $0.01.
    exec_maker_improvement: float = 0.01


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
        # Keep ONLY recognized metric names — so a malformed value (e.g. an inline
        # comment captured as the value, or a stray token) degrades to "all matchable"
        # instead of silently filtering out every real metric.
        match_fingerprint_metrics=frozenset(
            tok for raw in (env("MATCH_FINGERPRINT_METRICS", "") or "").split(",")
            if (tok := raw.strip().lower()) in _VALID_FINGERPRINT_METRICS
        ),
        scan_close_within_days=_env_float("SCAN_CLOSE_WITHIN_DAYS", 0.0),
        exec_min_leg_depth=_env_float("EXEC_MIN_LEG_DEPTH", 0.0),
        stream_max_ws_quote_age=_env_float("STREAM_MAX_WS_QUOTE_AGE", 2.0),
        exec_depth_fraction=_env_float("EXEC_DEPTH_FRACTION", 0.85),
        stream_min_leg_price=_env_float("STREAM_MIN_LEG_PRICE", 0.02),
        exec_hedge_buffer=_env_float("EXEC_HEDGE_BUFFER", 0.03),
        exec_maker_mode=(env("EXEC_MAKER_MODE", "false") or "false").lower() == "true",
        exec_maker_timeout=_env_float("EXEC_MAKER_TIMEOUT", 5.0),
        exec_maker_improvement=_env_float("EXEC_MAKER_IMPROVEMENT", 0.01),
    )
