"""Canonicalize-then-join: extraction fails closed, the join is exact on settlement."""

import json

from bot.data.store import Store
from bot.matching.canon import Canon, complementary, extract_canon


def mk(venue="kalshi", mid="K1", event_type="match", entities=("dplus", "gen.g global academy"),
       subject="gen.g global academy", metric="winner", comparator=None, value=None,
       period="full", date="2026-07-02", confidence=0.9):
    return Canon(venue=venue, market_id=mid, event_type=event_type,
                 entities=tuple(entities), subject=subject, metric=metric,
                 comparator=comparator, value=value, period=period, date=date,
                 confidence=confidence)


def test_extract_canon_parses_and_fails_closed():
    def llm(prompt):
        assert "resolution rules" in prompt.lower()
        return json.dumps({"event_type": "match", "entities": ["Dplus", "Gen.G Global Academy"],
                           "subject": "Gen.G Global Academy", "metric": "winner",
                           "comparator": None, "value": None, "period": "full",
                           "date": "2026-07-02", "confidence": 0.92})
    c = extract_canon(llm, venue="kalshi", market_id="K1", title="DK vs GENG - GENG",
                      rules="If Gen.G Global Academy wins the Jul 2 match, resolves Yes.")
    assert c is not None and c.subject == "gen.g global academy"
    assert c.entities == ("dplus", "gen.g global academy")     # normalized + sorted
    # anything unparseable -> None (fail closed)
    assert extract_canon(lambda p: "garbage", venue="k", market_id="m",
                         title="", rules="r") is None

    def boom(p):
        raise RuntimeError("down")
    assert extract_canon(boom, venue="k", market_id="m", title="", rules="r") is None


def test_join_matches_true_pair_with_name_variants():
    a = mk(venue="kalshi", entities=("dplus kia", "gen.g global academy"),
           subject="dplus kia")
    b = mk(venue="polymarket_us", mid="p1", entities=("dplus", "gen.g global academy"),
           subject="dplus")                                    # shorter variant, no sub-org marker
    assert complementary(a, b)                                 # token-subset subject aligns
    # BUT dropping a SUB-ORG marker is NOT a variant — "gen.g" could be the main org
    # (the BESTIA-class risk), so it must not align with "gen.g global academy".
    c = mk(venue="polymarket_us", mid="p2", entities=("dplus kia", "gen.g"),
           subject="gen.g")
    a2 = mk(venue="kalshi", subject="gen.g global academy")
    assert not complementary(a2, c)


def test_join_rejects_every_settlement_divergence():
    a = mk()
    assert not complementary(a, mk(venue="p", value=2.5, comparator=">="))  # handicap line
    assert not complementary(a, mk(venue="p", period="1h"))                 # half vs full
    assert not complementary(a, mk(venue="p", metric="goals"))              # different metric
    assert not complementary(a, mk(venue="p", date="2026-07-03"))           # different date
    assert not complementary(a, mk(venue="p", subject="dplus"))             # YES pays other side
    assert not complementary(a, mk(venue="p", entities=("liquid", "faze"),
                                   subject="gen.g global academy"))         # different event
    assert not complementary(a, mk(venue="p", confidence=0.4))              # low confidence
    assert not complementary(a, mk(venue="p", subject=None))                # draw/range YES


def test_join_is_domain_agnostic():
    # An election canonicalizes and joins exactly like a match — no sports vocabulary.
    a = mk(event_type="election", entities=("smith", "jones"), subject="smith",
           metric="winner", date="2026-11-03")
    b = mk(venue="polymarket_us", mid="p-el", event_type="election",
           entities=("john smith", "mary jones"), subject="john smith",
           metric="winner", date="2026-11-03")
    assert complementary(a, b)
    # a BTC threshold market joins only on the same comparator+value
    c1 = mk(event_type="price_threshold", entities=("btc",), subject="btc",
            metric="price_close", comparator=">=", value=100000.0, date="2026-12-31")
    c2 = mk(venue="polymarket_us", mid="p-btc", event_type="price_threshold",
            entities=("bitcoin btc",), subject="bitcoin btc", metric="price_close",
            comparator=">=", value=100000.0, date="2026-12-31")
    assert complementary(c1, c2)
    assert not complementary(c1, mk(venue="p", event_type="price_threshold",
                                    entities=("btc",), subject="btc", metric="price_close",
                                    comparator=">=", value=120000.0, date="2026-12-31"))


def test_store_canon_roundtrip_and_join():
    s = Store(":memory:")
    s.record_canon(mk(venue="kalshi", mid="K1"))
    s.record_canon(mk(venue="polymarket_us", mid="p1"))
    s.record_canon(mk(venue="polymarket_us", mid="p-spread",
                      comparator=">=", value=1.5))    # a line market
    assert s.canon_checked("kalshi", "K1")
    pairs = s.canon_pairs()
    assert len(pairs) == 1                                     # spread did NOT join
    assert pairs[0][:4] == ("kalshi", "K1", "polymarket_us", "p1")


def test_confirmed_pairs_union_includes_canon():
    s = Store(":memory:")
    s.record_canon(mk(venue="kalshi", mid="K1"))
    s.record_canon(mk(venue="polymarket_us", mid="p1"))
    got = s.confirmed_pairs(use_fingerprint=True, combine_verdicts=True, use_canon=True)
    assert ("kalshi", "K1", "polymarket_us", "p1") in {p[:4] for p in got}
    # and canon respects the blacklist backstop
    s.blacklist_pair("kalshi", "K1", "polymarket_us", "p1", reason="test")
    got = s.confirmed_pairs(use_fingerprint=True, combine_verdicts=True, use_canon=True)
    assert ("kalshi", "K1", "polymarket_us", "p1") not in {p[:4] for p in got}


def test_sub_org_teams_never_align():
    # An org and its academy/junior squad are DIFFERENT teams (the BESTIA incident:
    # 95 contracts "hedged" across the main org's and the Academy's games).
    a = mk(venue="kalshi", entities=("patins da ferrari", "bestia academy"),
           subject="bestia academy")
    b = mk(venue="polymarket_us", mid="p1", entities=("patins da ferrari", "bestia"),
           subject="bestia")
    assert not complementary(a, b)
    # but a genuine name variant WITHOUT a sub-org marker still aligns
    c = mk(venue="polymarket_us", mid="p2", entities=("patins da ferrari", "bestia academy"),
           subject="bestia academy esports")
    assert complementary(a, c)


def _canon(venue, mid, **kw):
    from bot.matching.canon import Canon
    d = dict(event_type="econ_release", entities=("cpi",), subject="cpi", metric="price_close",
             comparator=">=", value=5.0, period="full", date="2026-11-01", confidence=0.9)
    d.update(kw)
    return Canon(venue=venue, market_id=mid, **d)


def test_scalar_threshold_exact_value_guard():
    from bot.matching.canon import complementary
    a = _canon("kalshi", "KXCPIYOY-26NOV-T5.0")
    same = _canon("polymarket_us", "cpi-yoy-nov-5", value=5.0)
    diff = _canon("polymarket_us", "cpi-yoy-nov-49", value=4.9)   # adjacent bucket
    assert complementary(a, same) is True
    assert complementary(a, diff) is False                        # bucket-boundary guard


def test_nonsport_canon_pair_survives_confirmed_pairs():
    # A non-sports (KXCPI) canon pair must reach the watchlist — the sports-only series
    # allowlist lives in the verdict path and must NOT drop the canon union member.
    from bot.data.store import Store
    store = Store(":memory:")
    store.record_canon(_canon("kalshi", "KXCPIYOY-26NOV-T5.0"))
    store.record_canon(_canon("polymarket_us", "cpi-yoy-nov-2026-5", value=5.0))
    pairs = store.confirmed_pairs(use_fingerprint=True, combine_verdicts=True,
                                  use_canon=True, safe_types_only=True, max_fanout=None)
    keys = {store._pair_key(p[0], p[1], p[2], p[3]) for p in pairs}
    want = store._pair_key("kalshi", "KXCPIYOY-26NOV-T5.0",
                           "polymarket_us", "cpi-yoy-nov-2026-5")
    assert want in keys                                           # not dropped by allowlist


def test_lenient_json_repairs_trailing_comma_and_comments():
    from bot.matching.canon import _lenient_json
    assert _lenient_json('{"a": 1, "b": 2}') == {"a": 1, "b": 2}
    assert _lenient_json('{"a": 1, "b": 2,}') == {"a": 1, "b": 2}         # trailing comma
    assert _lenient_json('{"a": 1, // note\n "b": 2}') == {"a": 1, "b": 2}  # line comment
    assert _lenient_json('{"a": 1, /* why */ "b": 2}') == {"a": 1, "b": 2}   # block comment
    assert _lenient_json('{"d": "x", /* multi\nline */ "b": 2}') == {"d": "x", "b": 2}


# ---------------- canon audit fixes ----------------

def test_scalar_both_none_subject_joins():
    # THE structural bug: the prompt instructs subject=null for numeric ranges, but
    # complementary required subject alignment unconditionally -> identical CPI
    # contracts could NEVER join. Both-None must pass (identity via the exact fields).
    from bot.matching.canon import complementary
    a = _canon("kalshi", "K1", subject=None, metric="inflation_rate",
               comparator=">", value=3.7, date="2026-06-01")
    b = _canon("polymarket_us", "p1", subject=None, metric="inflation_rate",
               comparator=">", value=3.7, date="2026-06-01")
    assert complementary(a, b) is True
    # one-sided None still rejects (entity vs non-entity contract)
    c = _canon("polymarket_us", "p2", subject="cpi", metric="inflation_rate",
               comparator=">", value=3.7, date="2026-06-01")
    assert complementary(a, c) is False


def test_metric_normalization_closed_vocab():
    from bot.matching.canon import _normalize_metric
    assert _normalize_metric("cpi_increase", "") == "inflation_rate"     # synonym
    assert _normalize_metric("inflation_rate", "") == "inflation_rate"   # already canonical
    assert _normalize_metric("value", "Will CPI inflation be above 3.7%?") == "inflation_rate"
    assert _normalize_metric("increase", "US GDP growth in Q2") == "gdp_growth"
    assert _normalize_metric("bananas", "Some unrelated title") == "other"  # never invented


def test_date_normalization():
    from bot.matching.canon import _normalize_date
    assert _normalize_date("2026-XX-XX", "election") is None             # placeholder junk
    assert _normalize_date("2026-11-03", "election") == "2026-11-03"
    # econ joins at month granularity: 06-01 / 06-30 -> same bucket
    assert _normalize_date("2026-06-01", "econ_release") == "2026-06-01"
    assert _normalize_date("2026-06-30", "econ_release") == "2026-06-01"
    assert _normalize_date(None, "econ_release") is None


def test_legacy_rows_normalized_on_read_and_join():
    # Rows extracted BEFORE the normalization layer (free-form metric, day-level dates)
    # must join after normalize-on-read, without re-extraction.
    from bot.data.store import Store
    store = Store(":memory:")
    store.record_canon(_canon("kalshi", "KXCPIYOY-26JUN-T3.7", subject=None,
                              metric="cpi_increase", comparator=">", value=3.7,
                              date="2026-06-30"))
    store.record_canon(_canon("polymarket_us", "cpic-uscpi-june", subject=None,
                              metric="inflation_rate", comparator=">", value=3.7,
                              date="2026-06-01"))
    pairs = store.canon_pairs()
    assert len(pairs) == 1                                   # joined despite raw fields
    # adjacent bucket still never joins
    store.record_canon(_canon("polymarket_us", "cpic-uscpi-june-38", subject=None,
                              metric="inflation_rate", comparator=">", value=3.8,
                              date="2026-06-01"))
    assert len(store.canon_pairs()) == 1
