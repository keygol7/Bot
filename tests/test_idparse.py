"""Deterministic id parsing + alignment — fixtures are REAL ids from both venues."""

from bot.matching.idparse import (
    code_aligns_tokens, keys_match, parse_kalshi, parse_poly, person_codes,
)

SER = {
    "nascar": {"title": "NASCAR Race", "tags": ["Motorsport"]},
    "f1fl": {"title": "F1 Fastest Lap", "tags": ["Motorsport"]},
    "atp": {"title": "ATP Challenger Match", "tags": ["Tennis"]},
    "cpi": {"title": "Inflation", "tags": ["Inflation"]},
    "wnba": {"title": "WNBA Game", "tags": ["Basketball"]},
    "wcg": {"title": "World Cup Goal", "tags": ["Soccer"]},
    "cs2": {"title": "CS2 Game", "tags": ["Esports"]},
}


def _match(kt, ktitle, meta, ps, ptitle):
    return keys_match(parse_kalshi(kt, ktitle, meta), parse_poly(ps, ptitle))


def test_nascar_driver_codes_match_despite_identical_poly_titles():
    # poly titles carry NO driver; the slug code chaell = first3+last3 of the kalshi
    # title's full name. This was unmatchable for embeddings (all titles identical).
    assert _match("KXNASCARRACE-EER26-CHEL",
                  "Will Chase Elliott win the eero 400? - Chase Elliott", SER["nascar"],
                  "tec-nascar-eero400-2026-07-05-w-chaell", "NASCAR eero 400 Winner - Yes")
    assert not _match("KXNASCARRACE-EER26-CHEL",
                      "Will Chase Elliott win the eero 400? - Chase Elliott", SER["nascar"],
                      "tec-nascar-eero400-2026-07-05-w-shavan", "NASCAR eero 400 Winner - Yes")


def test_f1_event_abbreviation_segmentation():
    # BRIGP segments over 'british grand prix' (bri+g+p); one-sided date tolerated
    # only with year agreement + event identity.
    assert _match("KXF1FASTLAP-BRIGP26-ALB",
                  "Will Alexander Albon set the fastest lap? - Alexander Albon", SER["f1fl"],
                  "aachc-f1-gbr-2026-07-05-fastlap-alealb",
                  "British Grand Prix Main Race Fastest Lap - Yes")
    assert not _match("KXF1FASTLAP-BRIGP26-ALB",
                      "Will Alexander Albon set the fastest lap? - Alexander Albon", SER["f1fl"],
                      "aachc-f1-gbr-2026-07-05-fastlap-lannor",
                      "British Grand Prix Main Race Fastest Lap - Yes")


def test_tennis_lastname_codes():
    # kalshi pair-code COMDON = last3+last3; poly codes fracom/matdon = first3+last3
    assert _match("KXATPCHALLENGERMATCH-26JUN30COMDON-COM",
                  "Will Francisco Comesana win? - Francisco Comesana", SER["atp"],
                  "aec-atp-fracom-matdon-2026-06-30",
                  "Francisco Comesana vs Matteo Donati - Francisco Comesana")


def test_cpi_scalar_thresholds_exact():
    assert _match("KXCPIYOY-26JUN-T3.6", "Will CPI inflation be above 3.6%?", SER["cpi"],
                  "cpic-uscpi-june-yoy-2026-07-14-gt3pt6pct", "CPI YoY in June - Yes")
    assert not _match("KXCPIYOY-26JUN-T3.6", "Will CPI inflation be above 3.6%?", SER["cpi"],
                      "cpic-uscpi-june-yoy-2026-07-14-gt3pt7pct", "CPI YoY in June - Yes")


def test_team_winner_and_bestia_guard():
    assert _match("KXWNBAGAME-26JUN25NYSEA-NY",
                  "Will New York win the New York vs Seattle game? - New York", SER["wnba"],
                  "aec-wnba-ny-sea-2026-06-25",
                  "New York Liberty vs Seattle Storm - New York Liberty")
    # org vs academy squad must NEVER align (the BESTIA incident)
    assert not _match("KXCS2GAME-26JUL021800BSTAPDAF-BSTA",
                      "BESTIA Academy vs. Patins da Ferrari - BESTIA Academy", SER["cs2"],
                      "aec-cs2-pdaf-bsta-2026-07-02",
                      "Patins da Ferrari vs BESTIA - BESTIA")


def test_player_prop_lines_and_players_distinguished():
    k = ("KXWCGOAL-26JUL04PARFRA-FRAKMBAPP10-2",
         "Will Kylian Mbappe score 2 or more goals? - Kylian Mbappe", SER["wcg"])
    assert _match(*k, "astatc-fwc-par-fra-2026-07-04-g-fwckylmba-gte2",
                  "France vs Paraguay: Kylian Mbappe goals - Yes")
    # same event + same line, different player -> never
    assert not _match(*k, "astatc-fwc-par-fra-2026-07-04-g-fwcousdem-gte2",
                      "France vs Paraguay: Ousmane Dembele goals - Yes")
    # same player, different line -> never
    assert not _match(*k, "astatc-fwc-par-fra-2026-07-04-g-fwckylmba-gte3",
                      "France vs Paraguay: Kylian Mbappe goals - Yes")


def test_novelty_series_fail_closed():
    mk = parse_kalshi("KXWCMENTION-26JUN22NORSEN-SHUT", "Norway vs Senegal - Shutout",
                      {"title": "World Cup Mention", "tags": ["Soccer"]})
    assert not mk.matchable


def test_unknown_metric_never_matches():
    a = parse_kalshi("KXMYSTERY-26JUL05-X", "Something inscrutable", None)
    b = parse_poly("zzz-mystery-2026-07-05-x", "Something inscrutable")
    assert not keys_match(a, b)


def test_helpers():
    assert "chaell" in person_codes({"chase", "elliott"})
    assert code_aligns_tokens("brigp", {"british", "grand", "prix"})
    assert code_aligns_tokens("comdon", {"fracom", "matdon"})
    assert not code_aligns_tokens("zzz", {"british", "grand", "prix"})


def test_ipo_undated_both_sides():
    # both ids undated (year tokens only); outcome exact carries the join
    assert _match("KXIPO-26-DATABRICKS", "Will Databricks IPO in 2026? - Databricks",
                  {"title": "IPO", "tags": ["Companies"]},
                  "ipcc-2026ipos-databricks", "Databricks IPO before 2027?")
    assert not _match("KXIPO-26-DATABRICKS", "Will Databricks IPO in 2026? - Databricks",
                      {"title": "IPO", "tags": ["Companies"]},
                      "ipcc-2026ipos-stripe", "Stripe IPO before 2027?")


def test_election_undated_kalshi_vs_dated_poly():
    # KXGOVCA-26-SHIL: sparse middle (year only) + candidate code; poly dated Nov 3
    assert _match("KXGOVCA-26-SHIL", "Will Steve Hilton win? - Steve Hilton",
                  {"title": "California Governor Election", "tags": ["Politics"]},
                  "ewc-usgub-ca-2026-11-03-shil", "California Governor Race - Steve Hilton")


def test_nfl_division_series_title_evidence():
    # empty middle: the series TITLE supplies event tokens (north <-> afcnorth)
    assert _match("KXNFLAFCNORTH-26-BAL", "Will Baltimore win the AFC North? - Baltimore",
                  {"title": "American Football Conference North Winner", "tags": ["Football"]},
                  "tec-nfl-afcnorth-2027-01-04-w-bal", "AFC North Winner - Yes")


def test_f5_scope_separates_from_full_game():
    # The 2026-07-07 live incident: poly F5 slugs matched kalshi FULL-game markets.
    full_k = ("KXMLBGAME-26JUL071835CHCBAL-BAL", "Chicago vs Baltimore Winner? - Baltimore",
              {"title": "MLB Game", "tags": ["Baseball"]})
    f5_k = ("KXMLBF5-26JUL071835CHCBAL-BAL", "First 5 Innings Winner? - Baltimore",
            {"title": "First 5 Innings Winner", "tags": ["Baseball"]})
    f5_p = ("atc-mlb-chc-bal-2026-07-07-f5-bal", "CHC vs BAL First 5 Innings - Baltimore")
    full_p = ("atc-mlb-chc-bal-2026-07-07-bal", "CHC vs BAL - Baltimore")
    assert not _match(*full_k, *f5_p)      # the incident pair: must NEVER match
    assert not _match(*f5_k, *full_p)      # inverse mismatch
    assert _match(*f5_k, *f5_p)            # legit F5 <-> F5 arb
    assert _match(*full_k, *full_p)        # legit full <-> full
