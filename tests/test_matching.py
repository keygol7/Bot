from bot.matching.embed import candidate_pairs, lexical_similarity
from bot.matching.llm_match import confirm_match, MatchVerdict
from bot.models import MarketQuote


def mq(venue, mid, title):
    return MarketQuote(venue=venue, market_id=mid, title=title)


def test_lexical_similarity_high_for_near_identical():
    s = lexical_similarity(
        "Will the Fed cut rates in March 2026?",
        "Fed cuts interest rates March 2026",
    )
    assert s > 0.3


def test_lexical_similarity_low_for_unrelated():
    s = lexical_similarity(
        "Will it rain in Seattle tomorrow?",
        "Will Bitcoin close above 100000 this year?",
    )
    assert s < 0.2


def test_candidate_pairs_filters_and_excludes_same_venue():
    a = [mq("kalshi", "K1", "Fed cuts rates in March 2026"),
         mq("kalshi", "K2", "Lakers win the championship")]
    b = [mq("polymarket_us", "P1", "Fed cuts interest rates March 2026"),
         mq("polymarket_us", "P2", "Unrelated weather market")]
    cands = candidate_pairs(a, b, threshold=0.3)
    assert cands
    assert cands[0].a.market_id == "K1" and cands[0].b.market_id == "P1"
    # Never pairs two markets from the same venue.
    assert all(c.a.venue != c.b.venue for c in cands)


def test_confirm_match_parses_true_verdict():
    a = mq("kalshi", "K1", "Fed cuts rates March 2026")
    b = mq("polymarket_us", "P1", "Fed cuts interest rates March 2026")
    fake = lambda prompt: '{"same_event": true, "confidence": 0.93, "rationale": "same"}'
    v = confirm_match(a, b, fake)
    assert v.same_event and v.confidence == 0.93
    assert v.tradeable(min_confidence=0.85)


def test_confirm_match_parses_yes_party_format():
    # New structured response (with yes_party fields) still parses to a verdict.
    a = mq("kalshi", "K1", "Allan Nascimento win the fight")
    b = mq("polymarket_us", "P1", "Mitch Raposo win by submission in Nascimento vs Raposo")
    fake = lambda prompt: ('{"yes_party_a": "Allan Nascimento", "yes_party_b": '
                           '"Mitch Raposo", "same_event": false, "confidence": 0.98, '
                           '"rationale": "YES pays for different fighters"}')
    v = confirm_match(a, b, fake)
    assert v.same_event is False and not v.tradeable()


def test_confirm_prompt_asks_for_yes_party():
    from bot.matching.llm_match import build_prompt
    a = mq("kalshi", "K1", "x")
    b = mq("polymarket_us", "P1", "y")
    prompt = build_prompt(a, b)
    assert "yes_party_a" in prompt and "YES party" in prompt


def test_confirm_match_fails_closed_on_garbage():
    a = mq("kalshi", "K1", "x")
    b = mq("polymarket_us", "P1", "y")
    v = confirm_match(a, b, lambda p: "I cannot answer that")
    assert v.same_event is False
    assert v.confidence == 0.0
    assert not v.tradeable()


def test_confirm_match_extracts_json_amid_prose():
    a = mq("kalshi", "K1", "x")
    b = mq("polymarket_us", "P1", "y")
    msg = 'Sure!\n{"same_event": false, "confidence": 0.7, "rationale": "different cutoff"}\nDone.'
    v = confirm_match(a, b, lambda p: msg)
    assert v.same_event is False and round(v.confidence, 2) == 0.7


def test_low_confidence_not_tradeable_even_if_same():
    v = MatchVerdict(same_event=True, confidence=0.6)
    assert not v.tradeable(min_confidence=0.85)
