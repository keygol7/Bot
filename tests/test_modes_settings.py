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
    ):
        monkeypatch.delenv(key, raising=False)
    settings = load_settings(dotenv_path=str(tmp_path / "nonexistent.env"))
    assert settings.run_mode is RunMode.DRY_RUN
    assert settings.risk.max_daily_loss == 500.0
    assert settings.risk.min_edge == 0.01
    assert settings.qcex.use_sandbox is False
    assert settings.qcex.gateway_base == "https://gateway.polymarket.us"
    assert settings.qcex.is_trading_configured is False  # no creds by default


def test_load_settings_reads_env(tmp_path, monkeypatch):
    monkeypatch.setenv("BOT_RUN_MODE", "LIVE_SMALL")
    monkeypatch.setenv("RISK_MAX_DAILY_LOSS", "250")
    settings = load_settings(dotenv_path=str(tmp_path / "nonexistent.env"))
    assert settings.run_mode is RunMode.LIVE_SMALL
    assert settings.risk.max_daily_loss == 250.0
