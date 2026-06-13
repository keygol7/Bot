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


def test_creates_parent_directory(tmp_path):
    db = tmp_path / "nested" / "dir" / "bot.db"
    s = Store(str(db))
    assert db.exists()
    s.upsert_market("kalshi", "A", "Market A")
    s.close()


def test_audit_log():
    s = Store(":memory:")
    s.audit("startup", {"mode": "DRY_RUN"})
    row = s.conn.execute("SELECT * FROM audit_log").fetchone()
    assert row["kind"] == "startup"
    assert "DRY_RUN" in row["payload"]
    s.close()
