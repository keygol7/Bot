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


def test_record_edge_observation():
    s = Store(":memory:")
    s.record_edge("E1", "kalshi", "polymarket_us", 0.40, 0.55, 0.05, 50, "SUCCESS")
    s.record_edge("E1", "kalshi", "polymarket_us", 0.01, 0.90, 0.09, 0, "skip_settling")
    rows = s.conn.execute(
        "SELECT outcome, edge, size FROM edge_observations ORDER BY id").fetchall()
    assert [r["outcome"] for r in rows] == ["SUCCESS", "skip_settling"]
    assert rows[0]["edge"] == 0.05 and rows[0]["size"] == 50
    s.close()


def test_confirmed_pairs_fingerprint_recovers_llm_rejected_complement():
    # A real same-event winner pair the local LLM WRONGLY marked not-same-event. The
    # fingerprint-as-matcher must recover it; the old LLM-gated filter drops it.
    s = Store(":memory:")
    s.upsert_market("kalshi", "KXDOTA2GAME-26JUN21MODUSNAVI-NAVI",
                    "Will Natus Vincere win the MODUS vs. Natus Vincere Dota 2 match? - Natus Vincere")
    s.upsert_market("polymarket_us", "aec-dota2-navi-modus-2026-06-21",
                    "Who will win in the upcoming esports event Natus Vincere vs MODUS - Natus Vincere")
    s.cache_verdict("kalshi", "KXDOTA2GAME-26JUN21MODUSNAVI-NAVI",
                    "polymarket_us", "aec-dota2-navi-modus-2026-06-21",
                    same_event=False, confidence=0.2)        # LLM said NOT same-event
    assert s.confirmed_pairs() == []                          # LLM-gated filter drops it
    assert len(s.confirmed_pairs(use_fingerprint=True)) == 1  # fingerprint recovers it
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
    pairs = s.confirmed_pairs(safe_types_only=False)
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
    pairs = s.confirmed_pairs(safe_types_only=False)  # default floor 0.85
    assert {(p[0], p[1]) for p in pairs} == {("kalshi", "K1")}
    assert len(s.confirmed_pairs(min_confidence=0.5, safe_types_only=False)) == 2  # floor relaxed
    s.close()


def test_confirmed_pairs_drops_fanout_clusters():
    # The PGA trap: one Polymarket market "confirmed" against many distinct Kalshi
    # markets (and vice versa) is a multi-outcome cross-product, not 1:1 arb. All of
    # those pairs must be dropped; a clean 1:1 pair survives.
    s = Store(":memory:")
    # Fan-out cluster: K_a matches P1,P2,P3 — all high confidence.
    for p in ("P1", "P2", "P3"):
        s.cache_verdict("kalshi", "Kfield", "polymarket_us", p, same_event=True, confidence=1.0)
    # Clean 1:1 pair elsewhere.
    s.cache_verdict("kalshi", "Ksolo", "polymarket_us", "Psolo", same_event=True, confidence=1.0)

    pairs = s.confirmed_pairs(safe_types_only=False)  # default max_fanout=1
    assert {(p[0], p[1], p[2], p[3]) for p in pairs} == {
        ("kalshi", "Ksolo", "polymarket_us", "Psolo")
    }
    # Disabling the fan-out gate keeps everything (4 pairs).
    assert len(s.confirmed_pairs(max_fanout=None, safe_types_only=False)) == 4
    s.close()


def test_confirmed_pairs_drops_scope_mismatch():
    # A same_event=True 1:1 pair whose titles resolve on different scopes ("2nd half"
    # vs full match) must be dropped from the watchlist — fan-out can't catch a 1:1.
    s = Store(":memory:")
    s.upsert_market("kalshi", "K1", "Will Portugal win the 2nd Half? - Portugal")
    s.upsert_market("polymarket_us", "P1", "Will Portugal win the World Cup match? - Portugal")
    s.cache_verdict("kalshi", "K1", "polymarket_us", "P1", same_event=True, confidence=1.0)
    # A clean full-match pair survives.
    s.upsert_market("kalshi", "K2", "Will France win the match? - France")
    s.upsert_market("polymarket_us", "P2", "Will France win against Iraq? - France")
    s.cache_verdict("kalshi", "K2", "polymarket_us", "P2", same_event=True, confidence=1.0)

    pairs = s.confirmed_pairs(safe_types_only=False)
    assert {(p[0], p[1]) for p in pairs} == {("kalshi", "K2")}
    # The gate can be disabled.
    assert len(s.confirmed_pairs(drop_scope_mismatch=False, safe_types_only=False)) == 2
    s.close()


def test_confirmed_pairs_safe_types_only():
    # Exotic Kalshi series (announcer mention, group-stage undefeated) must be excluded
    # by the series allowlist even though they're same_event=1, 1:1, and the title reads
    # like a winner. A vetted game-winner series survives.
    s = Store(":memory:")
    Km = "KXWCMENTION-26JUN19BRAHTI-LFFL"
    s.upsert_market("kalshi", Km, "What will the announcers say during Brazil vs Haiti? - LFF")
    s.upsert_market("polymarket_us", "Pm", "Will Haiti win against Brazil? - Haiti")
    s.cache_verdict("kalshi", Km, "polymarket_us", "Pm", same_event=True, confidence=1.0)
    Ku = "KXWCGSUNDEFEATED-26-CPV"
    s.upsert_market("kalshi", Ku, "Will Cape Verde win all 3 of their group matches? - Cape Verde")
    s.upsert_market("polymarket_us", "Pu", "Will Cape Verde win vs Uruguay? - Cape Verde")
    s.cache_verdict("kalshi", Ku, "polymarket_us", "Pu", same_event=True, confidence=1.0)
    Kw = "KXWNBAGAME-26JUN17DALGS-DAL"
    s.upsert_market("kalshi", Kw, "Dallas vs Golden State winner? - Dallas")
    s.upsert_market("polymarket_us", "Pw", "Who will win Dallas vs Golden State? - Dallas")
    s.cache_verdict("kalshi", Kw, "polymarket_us", "Pw", same_event=True, confidence=1.0)

    pairs = s.confirmed_pairs()                      # safe_types_only=True default
    assert {(p[0], p[1]) for p in pairs} == {("kalshi", Kw)}
    assert len(s.confirmed_pairs(safe_types_only=False)) == 3   # gate can be relaxed
    s.close()


def test_confirmed_pairs_fingerprint_mode():
    # The fingerprint gate is strictly better than the allowlist: it ADMITS a clean
    # winner the title/series allowlist drops, and REJECTS a winner<->method-decision
    # pair the allowlist wrongly keeps.
    s = Store(":memory:")
    # (1) Tennis winner: Poly title lacks "win" so the allowlist drops it; fingerprint
    #     keeps it (vs -> winner, same player/date).
    s.upsert_market("kalshi", "KXATPMATCH-26JUN17DESHA-DE",
                    "Will Alex de Minaur win the de Minaur vs Shapovalov match? - Alex de Minaur")
    s.upsert_market("polymarket_us", "aec-atp-alemin-densha-2026-06-17",
                    "Alex de Minaur vs. Denis Shapovalov - Alex de Minaur")
    s.cache_verdict("kalshi", "KXATPMATCH-26JUN17DESHA-DE",
                    "polymarket_us", "aec-atp-alemin-densha-2026-06-17",
                    same_event=True, confidence=1.0)
    # (2) Fight winner <-> "go to a decision" method market: allowlist keeps (the word
    #     "draw" trips its winner regex); fingerprint rejects (no clean YES side).
    s.upsert_market("kalshi", "KXUFCFIGHT-26JUN20KAPHOR-HOR",
                    "Will Kyoji Horiguchi win the Kape vs Horiguchi MMA fight? - Kyoji Horiguchi")
    s.upsert_market("polymarket_us", "astatc-ufc-kyohor-mankap-2026-06-20-rov-dec",
                    "Will Kyoji Horiguchi vs. Manel Kape go to a decision, draw, or no contest? - Yes")
    s.cache_verdict("kalshi", "KXUFCFIGHT-26JUN20KAPHOR-HOR",
                    "polymarket_us", "astatc-ufc-kyohor-mankap-2026-06-20-rov-dec",
                    same_event=True, confidence=1.0)

    live = {(p[0], p[1]) for p in s.confirmed_pairs()}
    fp = {(p[0], p[1]) for p in s.confirmed_pairs(use_fingerprint=True)}
    assert ("kalshi", "KXATPMATCH-26JUN17DESHA-DE") not in live   # allowlist drops winner
    assert ("kalshi", "KXATPMATCH-26JUN17DESHA-DE") in fp         # fingerprint keeps it
    assert ("kalshi", "KXUFCFIGHT-26JUN20KAPHOR-HOR") in live     # allowlist keeps FP
    assert ("kalshi", "KXUFCFIGHT-26JUN20KAPHOR-HOR") not in fp   # fingerprint rejects it
    # Metric restriction for a staged rollout.
    assert s.confirmed_pairs(use_fingerprint=True, fingerprint_metrics=frozenset({"goals"})) == []
    s.close()


def test_drop_fanout_pairs_pure():
    from bot.data.store import drop_fanout_pairs

    pairs = [
        ("k", "A", "p", "1", "e"),   # A->1 only (clean)
        ("k", "B", "p", "2", "e"),   # B->2,3 (fan-out)
        ("k", "B", "p", "3", "e"),
    ]
    kept = drop_fanout_pairs(pairs, max_fanout=1)
    assert kept == [("k", "A", "p", "1", "e")]
    assert len(drop_fanout_pairs(pairs, max_fanout=5)) == 3   # tolerant threshold keeps all


def test_audit_log():
    s = Store(":memory:")
    s.audit("startup", {"mode": "DRY_RUN"})
    row = s.conn.execute("SELECT * FROM audit_log").fetchone()
    assert row["kind"] == "startup"
    assert "DRY_RUN" in row["payload"]
    s.close()
