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


MLB = {
    "game": {"title": "Professional Baseball Game", "tags": ["Baseball"]},
    "f5": {"title": "First 5 Innings Winner", "tags": ["Baseball"]},
    "total": {"title": "Pro Baseball Total Points", "tags": ["Baseball"]},
    "f5total": {"title": "First 5 Innings Total", "tags": ["Baseball"]},
    "f5spread": {"title": "First 5 Innings Spread", "tags": ["Baseball"]},
    "ldr_rbi": {"title": "MLB RBIs Leader", "tags": ["Baseball"]},
    "alcy": {"title": "Pro Baseball American League Cy Young", "tags": ["Baseball"]},
    "drafttop": {"title": "Pro Baseball Top Pick", "tags": ["Baseball"]},
}


def test_mlb_totals_normalize_to_count_form():
    # kalshi outcome digit N = "over N-0.5" = ">= N"; poly bare 10pt5 = ">= 11"
    k = parse_kalshi("KXMLBF5TOTAL-26JUL071835CHCBAL-4",
                     "CHC vs BAL first 5 innings runs? - Over 3.5 runs", MLB["f5total"])
    p = parse_poly("tsc-mlb-chc-bal-2026-07-07-f5-3pt5",
                   "Cubs vs Orioles F5: O/U 3.5 - Over")
    assert (k.metric, k.thr_lo, set(k.scope)) == ("total", 4.0, {"f5"})
    assert (p.metric, p.thr_lo, set(p.scope)) == ("total", 4.0, {"f5"})
    assert keys_match(k, p)
    # different lines never match
    p2 = parse_poly("tsc-mlb-chc-bal-2026-07-07-f5-4pt5", "Cubs vs Orioles F5: O/U 4.5 - Over")
    assert not keys_match(k, p2)


def test_mlb_spread_margin_form_and_pos_fails_closed():
    k = parse_kalshi("KXMLBF5SPREAD-26JUL071835CHCBAL-BAL2",
                     "Baltimore wins first 5 innings by over 1.5 runs? - Baltimore -1.5",
                     MLB["f5spread"])
    p = parse_poly("asc-mlb-chc-bal-2026-07-07-f5-neg-1pt5",
                   "Will the Chicago Cubs cover -1.5 (F5)")
    # both margin>=2 form; outcome teams differ here (bal vs chc) so no match,
    # but a same-team pair aligns
    assert k.thr_lo == p.thr_lo == 2.0 and k.metric == p.metric == "spread"
    p_pos = parse_poly("asc-mlb-chc-bal-2026-07-07-f5-pos-1pt5",
                       "Will the Chicago Cubs cover +1.5 (F5)")
    assert not p_pos.matchable          # polarity-inverted: must fail closed


def test_mlb_leader_metric_never_crosses_props():
    lead = parse_kalshi("KXLEADERMLBRBI-26-AJUD", "MLB RBIs Leader - Aaron Judge", MLB["ldr_rbi"])
    p_lead = parse_poly("aachc-mlb-rbi-leader-aarjud", "MLB RBI Leader - Aaron Judge")
    assert lead.metric == p_lead.metric == "ldr_rbi"
    assert keys_match(lead, p_lead)
    # a game RBI prop shares the player but NOT the metric
    assert lead.metric != "rbi"


def test_mlb_cy_young_qualifier():
    k = parse_kalshi("KXMLBALCY-26-BWOO", "AL Cy Young - Bryan Woo", MLB["alcy"])
    p = parse_poly("tec-mlb-al-2026-11-27-cy-brywoo", "AL Cy Young Award - Yes")
    assert k.metric == p.metric == "cyyoung"
    assert keys_match(k, p)


def test_mlb_draft_top_scopes():
    k = parse_kalshi("KXMLBDRAFTTOP-26-10-AGRA", "Will A. Gray go top 10? - A. Gray", MLB["drafttop"])
    p10 = parse_poly("arankc-mlb-draft-2026-07-12-top10-andgra", "MLB Draft Top 10 - Andrew Gray")
    p5 = parse_poly("arankc-mlb-draft-2026-07-12-top5-andgra", "MLB Draft Top 5 - Andrew Gray")
    assert "top10" in k.scope and "top10" in p10.scope and "top5" in p5.scope
    assert keys_match(k, p10)
    assert not keys_match(k, p5)        # different cut lines never match


def test_cross_gender_tennis_never_matches():
    # BONWEI class: men's ITF vs women's ITF matched via a 3-letter name collision
    # ('wei'). Gendered circuits are different events regardless of id alignment.
    a = parse_kalshi("KXITFMATCH-26JUL08BONWEI-WEI",
                     "Will Wei win the Bond vs Wei match? - Wei",
                     {"title": "ITF Match", "tags": ["Tennis"]})
    b = parse_poly("aec-itfwo-sijwei-yinsun-2026-07-08",
                   "Sijia Wei vs Yin Sun - Sijia Wei")
    assert a.category == "atp" and b.category == "itfwo"
    assert not keys_match(a, b)
    # same-gender still matches
    c = parse_poly("aec-itfme-bonwei-xyz-2026-07-08", "Bond vs Wei - Wei")
    assert keys_match(a, c)
