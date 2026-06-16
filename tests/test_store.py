from bot.data.store import Store
from bot.models import MarketQuote
from bot.strategies.arbitrage import detect_cross_venue


def make_opp():
    a = MarketQuote("kalshi", "A", "A", "E1", yes_ask=0.40, yes_ask_size=100, no_ask=0.65, no_ask_size=100)
    b = MarketQuote("polymarket_us", "B", "B", "E1", yes_ask=0.62, yes_ask_size=100, no_ask=0.55, no_ask_size=60)
    return detect_cross_venue(a, b)[0]


def test_upsert_market_and_pnl():
    s = Store(":memory:")
    s.upsert_market("kalshi", "A", "Market A", "E1")
    s.upsert_market("kalshi", "A", "Market A renamed")  # upsert keeps event_key
    row = s.conn.execute("SELECT * FROM markets WHERE market_id='A'").fetchone()
    assert row["title"] == "Market A renamed"
    assert row["event_key"] == "E1"

    s.record_pnl(10.0, "arb close")
    s.record_pnl(-3.5, "slippage")
    assert round(s.total_pnl(), 2) == 6.5
    s.close()


def test_record_opportunity():
    s = Store(":memory:")
    opp = make_opp()
    rowid = s.record_opportunity(opp, acted=False)
    assert rowid == 1
    row = s.conn.execute("SELECT * FROM opportunities WHERE id=1").fetchone()
    assert row["buy_yes_venue"] == "kalshi"
    assert row["acted"] == 0
    s.close()


def test_verdict_cache_is_order_independent():
    s = Store(":memory:")
    s.cache_verdict(
        "kalshi", "A", "polymarket_us", "B",
        same_event=True, confidence=0.95, rationale="same election", event_key="E1",
    )
    # Look up with arguments in the opposite order -> same row.
    v = s.get_verdict("polymarket_us", "B", "kalshi", "A")
    assert v is not None
    assert v["same_event"] == 1
    assert round(v["confidence"], 2) == 0.95
    assert v["event_key"] == "E1"
    s.close()


def test_bulk_upsert_markets():
    s = Store(":memory:")
    rows = [("kalshi", f"K{i}", f"Market {i}", None) for i in range(200)]
    s.upsert_markets(rows)
    assert s.conn.execute("SELECT COUNT(*) c FROM markets").fetchone()["c"] == 200
    # Upsert again updates in place (no duplicates).
    s.upsert_markets([("kalshi", "K0", "Renamed", "E1")])
    row = s.conn.execute("SELECT * FROM markets WHERE market_id='K0'").fetchone()
    assert row["title"] == "Renamed" and row["event_key"] == "E1"
    assert s.conn.execute("SELECT COUNT(*) c FROM markets").fetchone()["c"] == 200
    s.upsert_markets([])  # empty is a no-op
    s.close()


def test_creates_parent_directory(tmp_path):
    db = tmp_path / "nested" / "dir" / "bot.db"
    s = Store(str(db))
    assert db.exists()
    s.upsert_market("kalshi", "A", "Market A")
    s.close()


def test_confirmed_pairs_returns_only_same_event():
    s = Store(":memory:")
    s.cache_verdict("kalshi", "K1", "polymarket_us", "P1", same_event=True, confidence=0.95, event_key="E1")
    s.cache_verdict("kalshi", "K2", "polymarket_us", "P2", same_event=False, confidence=0.1)
    pairs = s.confirmed_pairs()
    assert len(pairs) == 1
    va, ma, vb, mb, ek = pairs[0]
    assert {(va, ma), (vb, mb)} == {("kalshi", "K1"), ("polymarket_us", "P1")}
    assert ek == "E1"
    s.close()


def test_confirmed_pairs_applies_confidence_floor():
    # A same_event=True but LOW-confidence verdict must NOT reach the watchlist —
    # it isn't tradeable, so the streamer must never see it.
    s = Store(":memory:")
    s.cache_verdict("kalshi", "K1", "polymarket_us", "P1", same_event=True, confidence=0.95)
    s.cache_verdict("kalshi", "K2", "polymarket_us", "P2", same_event=True, confidence=0.60)
    pairs = s.confirmed_pairs()                       # default floor 0.85
    assert {(p[0], p[1]) for p in pairs} == {("kalshi", "K1")}
    assert len(s.confirmed_pairs(min_confidence=0.5)) == 2   # floor can be relaxed
    s.close()


def test_audit_log():
    s = Store(":memory:")
    s.audit("startup", {"mode": "DRY_RUN"})
    row = s.conn.execute("SELECT * FROM audit_log").fetchone()
    assert row["kind"] == "startup"
    assert "DRY_RUN" in row["payload"]
    s.close()
