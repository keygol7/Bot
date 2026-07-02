"""Rules-text verification: compare resolution criteria, fail closed."""

import json

from bot.data.store import Store
from bot.matching.rules_match import confirm_rules


def test_identical_rules_verdict():
    def llm(prompt):
        assert "RESOLUTION RULES" in prompt and "Gen.G" in prompt
        return json.dumps({"divergent_scenario": "none found", "identical": True,
                           "confidence": 0.95, "rationale": "same match, same winner"})
    v = confirm_rules(llm, venue_a="kalshi", title_a="DK vs GENG - GENG",
                      rules_a="If Gen.G wins the Jul 2 match, resolves Yes.",
                      venue_b="polymarket_us", title_b="GENG vs DK - Gen.G",
                      rules_b="If Gen.G wins, resolves to Gen.G.")
    assert v.identical and v.confidence == 0.95


def test_divergent_and_malformed_fail_closed():
    v = confirm_rules(lambda p: json.dumps({"identical": False, "confidence": 0.9,
                                            "rationale": "handicap vs moneyline"}),
                      venue_a="k", title_a="", rules_a="win", venue_b="p", title_b="",
                      rules_b="win by 2.5")
    assert not v.identical
    for bad in ("not json at all", '{"identical": "maybe"}', ""):
        v = confirm_rules(lambda p, b=bad: b, venue_a="k", title_a="", rules_a="r",
                          venue_b="p", title_b="", rules_b="r")
        assert not v.identical                       # fails CLOSED

    def boom(p):
        raise RuntimeError("llm down")
    assert not confirm_rules(boom, venue_a="k", title_a="", rules_a="r",
                             venue_b="p", title_b="", rules_b="r").identical


def test_verified_pair_keys_union():
    s = Store(":memory:")
    s.record_rules_verdict("kalshi", "K1", "polymarket_us", "p1",
                           identical=True, confidence=0.9, rationale="")
    s.record_rules_verdict("kalshi", "K2", "polymarket_us", "p2",
                           identical=False, confidence=0.9, rationale="diverges")
    s.record_settlement_check("kalshi", "K3", "polymarket_us", "p3",
                              result_a="yes", result_b="yes", consistent=True)
    keys = s.verified_pair_keys()
    assert tuple(sorted([("kalshi", "K1"), ("polymarket_us", "p1")])) in keys
    assert tuple(sorted([("kalshi", "K3"), ("polymarket_us", "p3")])) in keys
    assert len(keys) == 2                            # divergent K2 NOT verified
