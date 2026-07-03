"""Settlement ground truth: divergent settlements are PROOF of a false match."""

from types import SimpleNamespace

from bot.data.store import Store
from bot.matching.settlement_truth import audit_settlements, infer_poly_result


def pos(net, cost, realized):
    return {"netPosition": str(net), "cost": {"value": str(cost)},
            "realized": {"value": str(realized)}}


def test_infer_poly_result_long_and_short():
    # LONG 10 YES at $4: realized +6 => YES paid; realized -4 => NO paid.
    assert infer_poly_result(pos(10, 4.0, 6.0)) == "yes"
    assert infer_poly_result(pos(10, 4.0, -4.0)) == "no"
    # SHORT 10 (holding NO) at $4: realized +6 => NO paid; realized -4 => YES paid.
    assert infer_poly_result(pos(-10, 4.0, 6.0)) == "no"
    assert infer_poly_result(pos(-10, 4.0, -4.0)) == "yes"
    # fees/slippage tolerated near an endpoint
    assert infer_poly_result(pos(10, 4.0, 5.72)) == "yes"


def test_infer_poly_result_never_guesses():
    # realized far from BOTH endpoints (partial redemptions, weird economics) -> None.
    assert infer_poly_result(pos(10, 4.0, 1.0)) is None
    assert infer_poly_result(pos(0, 0, 0)) is None          # flat
    assert infer_poly_result({}) is None


def _store_with_pair(km, pm):
    store = Store(":memory:")
    store.record_opportunity(SimpleNamespace(
        event_key=f"{km}|{pm}", buy_yes_venue="kalshi", buy_yes_market=km,
        buy_no_venue="polymarket_us", buy_no_market=pm, yes_price=0.4, no_price=0.55,
        edge_per_contract=0.05, max_contracts=10, total_profit=0.5), acted=True)
    return store


def test_consistent_settlement_records_positive_evidence():
    store = _store_with_pair("K1", "p1")
    # kalshi says yes; poly held 5 NO at $3 and realized -3 (NO lost) => poly says yes.
    n = audit_settlements(
        [{"ticker": "K1", "market_result": "yes"}],
        {"p1": pos(-5, 3.0, -3.0)},
        store.acted_pair_map(), store)
    assert n == 1
    ok, bad = store.settlement_consistency()
    assert (ok, bad) == (1, 0)
    assert not store.blacklisted_keys()                     # true match -> no blacklist


def test_divergent_settlement_blacklists_ground_truth():
    store = _store_with_pair("K1", "p1")
    # kalshi says yes; poly's NO side PAID (realized = 5-3 = +2 on a 5-NO short)
    # => poly says no => the legs settled DIFFERENTLY: proven false match.
    n = audit_settlements(
        [{"ticker": "K1", "market_result": "yes"}],
        {"p1": pos(-5, 3.0, 2.0)},
        store.acted_pair_map(), store)
    assert n == 1
    ok, bad = store.settlement_consistency()
    assert (ok, bad) == (0, 1)
    assert store.blacklisted_keys()                         # ground truth -> blacklisted


def test_ambiguous_or_unknown_pairs_skipped():
    store = _store_with_pair("K1", "p1")
    n = audit_settlements(
        [{"ticker": "K1", "market_result": "yes"},          # ambiguous realized -> skip
         {"ticker": "K_unpaired", "market_result": "no"},    # no pair map entry -> skip
         {"ticker": "K1", "market_result": "scalar"}],       # non-binary -> skip
        {"p1": pos(10, 4.0, 1.0)},
        store.acted_pair_map(), store)
    assert n == 0
    # a checked pair is not re-checked
    store.record_settlement_check("kalshi", "K1", "polymarket_us", "p1",
                                  result_a="yes", result_b="yes", consistent=True)
    n = audit_settlements([{"ticker": "K1", "market_result": "yes"}],
                          {"p1": pos(10, 4.0, 6.0)}, store.acted_pair_map(), store)
    assert n == 0


def test_void_refund_never_classified_as_a_result():
    # A voided/refunded market realizes ~$0 — NOT a win or a loss. On a 42-lot costing
    # $32.40 (the Atreides forfeit shape), $0 sits 9.6 from the "won" endpoint; a loose
    # band would call that a win and record a false ground-truth verdict.
    assert infer_poly_result(pos(42, 32.40, 0.0)) is None
    # while genuine settlements (realized at an endpoint +- fees) still classify
    assert infer_poly_result(pos(42, 32.40, 9.60)) == "yes"     # won exactly
    assert infer_poly_result(pos(42, 32.40, 8.90)) == "yes"     # won minus fees
    assert infer_poly_result(pos(42, 32.40, -32.40)) == "no"    # lost
