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


def test_confirm_match_rejects_explicit_calendar_date_mismatch_without_llm():
    a = mq("kalshi", "K", "High temperature in Miami on Jul 13, 2026: 94-95F")
    b = mq("polymarket_com", "P", "Highest temperature in Miami on July 14: 94-95F")
    calls = []
    v = confirm_match(a, b, lambda prompt: calls.append(prompt) or
                      '{"same_event": true, "confidence": 1}')
    assert not v.same_event
    assert "date mismatch" in v.rationale
    assert calls == []


def test_confirm_match_rejects_crypto_strike_and_time_mismatches_without_llm():
    calls = []
    yes = lambda prompt: calls.append(prompt) or '{"same_event": true, "confidence": 1}'
    strike = confirm_match(
        mq("kalshi", "K1", "Ethereum price at Jul 13, 2026 at 5pm EDT? - $1,880 or above"),
        mq("polymarket_com", "P1", "Ethereum above 1,710 on July 13, 2AM ET?"), yes,
    )
    when = confirm_match(
        mq("kalshi", "K2", "Ethereum price at Jul 13, 2026 at 5pm EDT? - $1,760 or above"),
        mq("polymarket_com", "P2", "Ethereum above 1,760 on July 13, 2AM ET?"), yes,
    )
    assert not strike.same_event and "strike mismatch" in strike.rationale
    assert not when.same_event and "time mismatch" in when.rationale
    assert calls == []


def test_confirm_match_rejects_hidden_crypto_cutoff_from_close_time_without_llm():
    calls = []
    a = mq("kalshi", "KXBTCD-26JUL1317-T63999.99",
           "Bitcoin price on Jul 13, 2026? - $64,000 or above")
    b = mq("polymarket_com", "bitcoin-above-64k-on-july-13-2026",
           "Will the price of Bitcoin be above $64,000 on July 13?")
    a.close_time = 1_789_746_000  # 17:00 ET
    b.close_time = 1_789_728_000  # noon ET
    verdict = confirm_match(
        a, b, lambda prompt: calls.append(prompt) or
        '{"same_event": true, "confidence": 1}',
    )
    assert not verdict.same_event and "time mismatch" in verdict.rationale
    assert calls == []


def test_confirm_match_rejects_netflix_us_vs_global_without_llm():
    calls = []
    verdict = confirm_match(
        mq("kalshi", "KXNETFLIXRANKSHOW-26JUL13-WOR",
           "Top US Netflix Show on Jul 13, 2026? - Worst Neighbor Ever"),
        mq("polymarket_com", "worst-neighbor-top-global-netflix-show",
           "Will Worst Neighbor Ever be the top global Netflix show this week?"),
        lambda prompt: calls.append(prompt) or
        '{"same_event": true, "confidence": 1}',
    )
    assert not verdict.same_event and "scope mismatch" in verdict.rationale
    assert calls == []


def test_confirm_match_rejects_strict_vs_inclusive_threshold_without_llm():
    calls = []
    verdict = confirm_match(
        mq("kalshi", "KXART-TYR26-30000000",
           'Will "Gus" sell above $30000000?'),
        mq("polymarket_com", "gus-at-least-30m",
           'Will "Gus" sell for at least $30m?'),
        lambda prompt: calls.append(prompt) or
        '{"same_event": true, "confidence": 1}',
    )
    assert not verdict.same_event and "comparator mismatch" in verdict.rationale
    assert calls == []


def test_confirm_match_rejects_weather_range_mismatch_without_llm():
    calls = []
    v = confirm_match(
        mq("kalshi", "K", "Will high temp be >96° on Jul 13, 2026? - 97° or above"),
        mq("polymarket_com", "P", "Highest temperature 98°F or higher on July 13?"),
        lambda prompt: calls.append(prompt) or '{"same_event": true, "confidence": 1}',
    )
    assert not v.same_event and "weather range mismatch" in v.rationale
    assert calls == []


def test_confirm_match_accepts_equivalent_strict_weather_threshold_for_llm():
    calls = []
    v = confirm_match(
        mq("kalshi", "K", "Will high temp be >96° on Jul 13, 2026?"),
        mq("polymarket_com", "P", "Highest temperature 97°F or higher on July 13?"),
        lambda prompt: calls.append(prompt) or '{"same_event": true, "confidence": 1}',
    )
    assert v.same_event and len(calls) == 1


def test_obvious_rules_mismatch_rejects_nws_vs_wunderground():
    from bot.matching.rules_match import obvious_rules_mismatch

    reason = obvious_rules_mismatch(
        "Official value reported by the National Weather Service Climatological Report",
        "The resolution source is Wunderground at station KMIA",
    )
    assert reason and "settlement source mismatch" in reason


def test_obvious_rules_mismatch_rejects_crypto_price_sources():
    from bot.matching.rules_match import obvious_rules_mismatch

    reason = obvious_rules_mismatch(
        "Simple average of CF Benchmarks Bitcoin Real-Time Index (BRTI)",
        "Binance BTC/USDT one minute candle close",
    )
    assert reason and "CF Benchmarks" in reason and "Binance" in reason


def test_confirm_match_reads_weather_range_from_market_id_without_llm():
    calls = []
    v = confirm_match(
        mq("polymarket_us", "tc-temp-miahigh-2026-07-13-lt90f",
           "Highest temperature in Miami on July 13? - Yes"),
        mq("polymarket_com", "highest-temperature-in-miami-on-july-13-2026-102forhigher",
           "Will the highest temperature in Miami be 102°F or higher on July 13?"),
        lambda prompt: calls.append(prompt) or '{"same_event": true, "confidence": 1}',
    )
    assert not v.same_event and "weather range mismatch" in v.rationale
    assert calls == []
