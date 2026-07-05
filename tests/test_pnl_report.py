"""Accurate profit math — pinned to the numbers validated against venue truth (QOR)."""

from types import SimpleNamespace

from bot.analysis.pnl_report import Ledger, build_ledgers, kalshi_settled_pnl, locked_pairs_report
from bot.data.store import Store


def test_ledger_average_cost_sells():
    # The QOR poly leg, validated against the venue's cost.value (32.36 incl fees):
    led = Ledger()
    led.buy(0.32, 11); led.buy(0.32, 9)
    led.sell(0.31, 9)          # realized = 9*(0.31-0.32) = -0.09; basis shrinks at AVG
    led.sell(0.54, 11)         # realized += 11*(0.54-0.32) = +2.42; flat
    led.buy(0.59, 15); led.buy(0.56, 20); led.buy(0.69, 3.06); led.buy(0.73, 13)
    assert abs(led.net - 51.06) < 1e-9
    assert abs(led.cost - 31.65) < 0.02          # venue: 32.36 incl ~0.7 fees
    assert abs(led.realized - 2.33) < 0.01       # venue 'realized': 2.33  <- exact match
    # the WRONG method (netting sells at proceeds) gives 29.32 — must not equal ours
    assert abs(led.cost - 29.32) > 2.0


def test_ledger_never_oversells():
    led = Ledger()
    led.buy(0.50, 5)
    led.sell(0.60, 8)                             # clamped to 5
    assert led.net == 0 and abs(led.realized - 0.5) < 1e-9


def _fill(venue, mid, side, price, ct, ts):
    return {"venue": venue, "market_id": mid, "side": side, "price": price,
            "contracts": ct, "ts": ts}


def test_locked_pairs_report_uses_avg_cost():
    store = Store(":memory:")
    store.record_opportunity(SimpleNamespace(
        event_key="k:K1|p:p1", buy_yes_venue="polymarket_us", buy_yes_market="p1",
        buy_no_venue="kalshi", buy_no_market="K1", yes_price=0.6, no_price=0.37,
        edge_per_contract=0.03, max_contracts=10, total_profit=0.3), acted=True)
    # poly YES: buy 20 @0.32, sell 20 (flat), buy 51 @avg .62 | kalshi NO: buy 51 @avg .48
    fills = [
        ("polymarket_us", "p1", "YES", 0.32, 20, 1), ("polymarket_us", "p1", "YES_SELL", 0.4, 20, 2),
        ("polymarket_us", "p1", "YES", 0.62, 51, 3),
        ("kalshi", "K1", "NO", 0.48, 51, 3),
    ]
    for f in fills:
        store.conn.execute("INSERT INTO fills (venue, market_id, side, price, contracts,"
                           " notional, ts) VALUES (?,?,?,?,?,?,?)",
                           (f[0], f[1], f[2], f[3], f[4], f[3]*f[4], f[5]))
    store.conn.commit()
    reps = locked_pairs_report(store)
    assert len(reps) == 1
    r = reps[0]
    assert r.hedged == 51
    # locked = 51 - (51*0.62 + 51*0.48) = 51 - 56.1 = -5.1 (ex fees)
    assert abs(r.locked - (51 - 56.1)) < 1e-6
    # realized from the round-trip sell: 20*(0.40-0.32) = +1.6
    assert abs(r.realized - 1.6) < 1e-6


def test_kalshi_settled_math():
    rows = kalshi_settled_pnl([{
        "ticker": "T1", "market_result": "no", "yes_count_fp": "16.00",
        "no_count_fp": "16.00", "yes_total_cost_dollars": "0.80",
        "no_total_cost_dollars": "15.20", "fee_cost": "0.1067",
        "settled_time": "2026-07-04T23:40:44Z",
    }])
    # payout 16 (no won, held 16 NO) - 0.80 - 15.20 - 0.1067 = -0.1067
    assert rows[0][0] == "T1" and abs(rows[0][1] - (-0.1067)) < 1e-4


def test_pnl_event_key_attribution():
    store = Store(":memory:")
    store.record_pnl(1.5, note="arb locked", event_key="k:A|p:b")
    store.record_pnl(-0.2, note="unwind (x)", event_key="k:A|p:b")
    row = store.conn.execute(
        "SELECT ROUND(SUM(amount),2) s FROM pnl WHERE event_key='k:A|p:b'").fetchone()
    assert row["s"] == 1.3


def test_locked_report_open_mask_excludes_settled_and_uses_venue_qty():
    store = Store(":memory:")
    store.record_opportunity(SimpleNamespace(
        event_key="k:K1|p:p1", buy_yes_venue="polymarket_us", buy_yes_market="p1",
        buy_no_venue="kalshi", buy_no_market="K1", yes_price=0.6, no_price=0.37,
        edge_per_contract=0.03, max_contracts=10, total_profit=0.3), acted=True)
    store.record_opportunity(SimpleNamespace(
        event_key="k:K2|p:p2", buy_yes_venue="polymarket_us", buy_yes_market="p2",
        buy_no_venue="kalshi", buy_no_market="K2", yes_price=0.5, no_price=0.45,
        edge_per_contract=0.05, max_contracts=10, total_profit=0.5), acted=True)
    for f in [("polymarket_us","p1","YES",0.62,51,1), ("kalshi","K1","NO",0.48,51,1),
              ("polymarket_us","p2","YES",0.50,10,1), ("kalshi","K2","NO",0.45,10,1)]:
        store.conn.execute("INSERT INTO fills (venue, market_id, side, price, contracts,"
                           " notional, ts) VALUES (?,?,?,?,?,?,?)",
                           (f[0], f[1], f[2], f[3], f[4], f[3]*f[4], f[5]))
    store.conn.commit()
    # venue says: pair 1 open (51/51), pair 2 SETTLED (no positions)
    mask = {("polymarket_us","p1"): 51.0, ("kalshi","K1"): 51.0}
    reps = locked_pairs_report(store, open_positions=mask)
    assert len(reps) == 1 and reps[0].yes_leg == ("polymarket_us","p1")
    # venue qty is authoritative for the hedged count even if the ledger differs
    mask2 = {("polymarket_us","p1"): 40.0, ("kalshi","K1"): 51.0,
             ("polymarket_us","p2"): 10.0, ("kalshi","K2"): 10.0}
    reps2 = locked_pairs_report(store, open_positions=mask2)
    p1 = next(r for r in reps2 if r.yes_leg == ("polymarket_us","p1"))
    assert p1.hedged == 40 and p1.imbalance == 11
