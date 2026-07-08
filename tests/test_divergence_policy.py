"""Divergence relevance: taxonomy, severities, policies."""

from bot.matching.divergence_policy import (
    classify_rationale, policy_for, sport_family,
)


def test_classify_legacy_rationales():
    assert classify_rationale("Market A includes extra time while B does not") == "timing_scope"
    assert classify_rationale("rules differ on withdrawals after a retirement") == "retirement_withdrawal"
    assert classify_rationale("handling of rain-affected matches differs") == "cancellation_postponement"
    assert classify_rationale("A voids, B settles at fair price") == "void_vs_fairprice"
    assert classify_rationale("different resolution times") == "settlement_time_only"
    assert classify_rationale("identical everything") == "none"


def test_sport_families():
    assert sport_family("KXITFWMATCH-26JUL07MAHCHA-MAH") == "tennis_itf"
    assert sport_family("KXATPMATCH-26JUL08SINDJO-SIN") == "tennis_tour"
    assert sport_family("KXWCGOAL-26JUL07SUICOL-X") == "soccer"
    assert sport_family("KXT20MATCH-26JUL10HAMWAR-HAM") == "cricket"


def test_policies_by_expected_cost():
    # soccer timing scope: asymmetric -> one_way regardless of cost
    p = policy_for("timing_scope", "KXWCFTTS-26JUL07SUICOL-COL")
    assert p.policy == "one_way"
    # ITF retirement: 5% x 50c = 2.5c expected -> edge_floor with 2x extra
    p = policy_for("retirement_withdrawal", "KXITFMATCH-26JUL08X-Y")
    assert p.policy == "edge_floor" and abs(p.extra_edge_ct - 0.05) < 1e-9
    # settlement-time-only: pure noise -> ignore
    p = policy_for("settlement_time_only", "KXLOLGAME-26JUL08X-Y")
    assert p.policy == "ignore" and p.extra_edge_ct == 0.0
    # different event: block
    assert policy_for("different_event", "KXANY-1").policy == "block"
    # esports cancellation: 1% x 50c = 0.5c -> edge_floor (above ignore threshold)
    p = policy_for("cancellation_postponement", "KXCS2GAME-26JUL08X-Y")
    assert p.policy == "edge_floor"


def test_calibration_override_only_raises():
    base = policy_for("retirement_withdrawal", "KXATPMATCH-1")   # tour prior 2%
    raised = policy_for("retirement_withdrawal", "KXATPMATCH-1", p_override=0.10)
    assert raised.expected_cost_ct > base.expected_cost_ct


def test_settlement_evidence_raises_severity():
    # a divergent settlement on a classified pair raises that class's branch rate,
    # which raises the edge floor for sibling pairs of the same class+sport
    from bot.data.store import Store
    st = Store(":memory:")
    st.record_rules_verdict("kalshi", "KXITFMATCH-26JUL08AB-A", "polymarket_us", "aec-itfme-ab",
                            identical=False, confidence=1.0,
                            rationale="rules differ on retirement handling",
                            divergence="retirement_withdrawal")
    st.record_rules_verdict("kalshi", "KXITFMATCH-26JUL08CD-C", "polymarket_us", "aec-itfme-cd",
                            identical=False, confidence=1.0,
                            rationale="rules differ on retirement handling",
                            divergence="retirement_withdrawal")
    base = st.pair_divergence_policies()
    key = st._pair_key("kalshi", "KXITFMATCH-26JUL08CD-C", "polymarket_us", "aec-itfme-cd")
    base_extra = base[key][1]
    # the AB pair settles DIVERGENTLY -> the branch hit -> Laplace (1+1)/(1+2)=0.67
    st.record_settlement_check("kalshi", "KXITFMATCH-26JUL08AB-A",
                               "polymarket_us", "aec-itfme-ab",
                               result_a="yes", result_b="no", consistent=False)
    raised = st.pair_divergence_policies()
    assert raised[key][1] > base_extra          # sibling's floor went UP
    # rates visible
    rates = st.divergence_branch_rates()
    assert rates[("retirement_withdrawal", "tennis_itf")] == (1, 1)
    st.close()
