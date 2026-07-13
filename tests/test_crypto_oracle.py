from types import SimpleNamespace

from bot.crypto_oracle import CryptoOracleMonitor


START = 1783978200
EVENT = (
    "kalshi:KXBTC15M-26JUL131745-45|"
    "polymarket_com:btc-updown-15m-1783978200"
)


def _monitor(now: float, *, cf=100.20, chainlink=100.18):
    monitor = CryptoOracleMonitor(SimpleNamespace(), entry_last_s=15)
    monitor.thresholds[(START, "BTC")] = {
        "kalshi_open": 100.0, "pcom_open": 100.0,
    }
    monitor.cf_ticks["BTC"] = (cf, now)
    monitor.chainlink_ticks["BTC"] = (chainlink, now)
    for second in range(START + 840, int(now) + 1):
        monitor.cf_final_minute[(START, "BTC")].append((float(second), cf))
        monitor.chainlink_history["BTC"].append((float(second), chainlink))
    return monitor


def test_oracle_gate_accepts_fresh_well_separated_same_side_paths():
    now = START + 890
    assessment = _monitor(now).assess(EVENT, now=now)
    assert assessment.eligible
    assert assessment.path_aligned
    assert assessment.reason == "eligible"
    assert assessment.cf_samples >= 45
    assert assessment.min_distance_bps > assessment.path_gap_bps


def test_oracle_gate_rejects_opposite_source_outcomes():
    now = START + 890
    assessment = _monitor(now, cf=100.20, chainlink=99.98).assess(EVENT, now=now)
    assert not assessment.eligible
    assert not assessment.path_aligned
    assert assessment.reason == "source_paths_disagree"


def test_oracle_gate_rejects_early_entry_and_missing_reference():
    early = START + 600
    assert _monitor(early).assess(EVENT, now=early).reason == "too_early_for_source_lock"
    bnb = EVENT.replace("BTC", "BNB").replace("btc-", "bnb-")
    assert _monitor(START + 890).assess(bnb, now=START + 890).reason == "threshold_missing"


def test_oracle_gate_projects_unobserved_cf_average_slots():
    now = START + 890
    monitor = _monitor(now, cf=100.10, chainlink=100.18)
    # The observed partial average is +10bp, but the current CF reference has
    # reversed enough that filling the nine remaining slots at spot projects a
    # negative final average. It must disagree with positive Chainlink and reject.
    monitor.cf_ticks["BTC"] = (99.0, now)
    assessment = monitor.assess(EVENT, now=now)
    assert not assessment.eligible
    assert assessment.reason == "source_paths_disagree"
    assert assessment.cf_move_bps < 0 < assessment.chainlink_move_bps
