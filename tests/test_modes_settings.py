import pytest

from bot.config.settings import load_settings
from bot.modes import RunMode


def test_run_mode_parse_and_flags():
    assert RunMode.parse(None) is RunMode.DRY_RUN
    assert RunMode.parse("live_small") is RunMode.LIVE_SMALL
    assert not RunMode.DRY_RUN.places_real_orders
    assert RunMode.LIVE_SMALL.places_real_orders
    assert RunMode.LIVE.places_real_orders


def test_run_mode_parse_rejects_unknown():
    with pytest.raises(ValueError):
        RunMode.parse("turbo")


def test_load_settings_defaults(tmp_path, monkeypatch):
    # No .env present and a clean environment -> safe defaults.
    for key in (
        "BOT_RUN_MODE", "RISK_MAX_DAILY_LOSS", "RISK_MIN_EDGE",
        "RISK_MAX_POSITION_PER_MARKET", "RISK_MAX_TOTAL_EXPOSURE",
        "SCAN_CLOSE_WITHIN_DAYS", "EXEC_MIN_LEG_DEPTH", "STREAM_MAX_WS_QUOTE_AGE",
        "EXEC_DEPTH_FRACTION", "STREAM_MIN_LEG_PRICE", "EXEC_HEDGE_BUFFER",
    ):
        monkeypatch.delenv(key, raising=False)
    settings = load_settings(dotenv_path=str(tmp_path / "nonexistent.env"))
    assert settings.run_mode is RunMode.DRY_RUN
    assert settings.risk.max_daily_loss == 500.0
    assert settings.risk.min_edge == 0.01
    assert settings.qcex.use_sandbox is False
    assert settings.qcex.gateway_base == "https://gateway.polymarket.us"
    assert settings.qcex.is_trading_configured is False  # no creds by default
    assert settings.scan_close_within_days == 0.0         # targeted scan off by default
    assert settings.exec_min_leg_depth == 0.0             # liquidity guard off by default
    assert settings.stream_max_ws_quote_age == 2.0        # trust WS book within 2s
    assert settings.exec_depth_fraction == 0.85           # trade 85% of shown depth
    assert settings.stream_min_leg_price == 0.02          # skip settling-market extremes
    assert settings.exec_hedge_buffer == 0.03             # hedge cushion / no-unwind gate


def test_load_settings_reads_scan_close_within_days(tmp_path, monkeypatch):
    monkeypatch.setenv("SCAN_CLOSE_WITHIN_DAYS", "4")
    settings = load_settings(dotenv_path=str(tmp_path / "nonexistent.env"))
    assert settings.scan_close_within_days == 4.0


def test_load_settings_reads_env(tmp_path, monkeypatch):
    monkeypatch.setenv("BOT_RUN_MODE", "LIVE_SMALL")
    monkeypatch.setenv("RISK_MAX_DAILY_LOSS", "250")
    settings = load_settings(dotenv_path=str(tmp_path / "nonexistent.env"))
    assert settings.run_mode is RunMode.LIVE_SMALL
    assert settings.risk.max_daily_loss == 250.0


def test_dotenv_strips_inline_comments(tmp_path, monkeypatch):
    for k in ("RISK_MAX_POSITION_PER_MARKET", "RISK_MIN_EDGE", "QCEX_API_BASE", "BOT_RUN_MODE"):
        monkeypatch.delenv(k, raising=False)
    env = tmp_path / ".env"
    env.write_text(
        "RISK_MAX_POSITION_PER_MARKET=50     # $ notional per market\n"
        "RISK_MIN_EDGE=0.02  # min edge\n"
        'QCEX_API_BASE="https://api.polymarket.us"  # quoted value\n'
    )
    settings = load_settings(dotenv_path=str(env))
    assert settings.risk.max_position_per_market == 50.0   # comment stripped, parses
    assert settings.risk.min_edge == 0.02
    assert settings.qcex.api_base == "https://api.polymarket.us"
