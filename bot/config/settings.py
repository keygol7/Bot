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
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    return float(raw) if raw not in (None, "") else default


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
    )
