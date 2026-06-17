from bot.matching.scope import _scoreline_tag, scope_mismatch, scope_tags


def test_second_half_vs_full_match_is_mismatch():
    # The exact live false positive.
    assert scope_mismatch(
        "Will Portugal win the 2nd Half? - Portugal",
        "Will Portugal win against Uzbekistan in the World Cup match? - Portugal",
    )


def test_same_full_match_is_not_mismatch():
    assert not scope_mismatch(
        "Will Portugal win against Uzbekistan? - Portugal",
        "Will Portugal win the match vs Uzbekistan? - Portugal",
    )


def test_both_second_half_is_not_mismatch():
    assert not scope_mismatch(
        "Will Portugal win the 2nd half?",
        "Portugal to win the second half",
    )


def test_method_of_victory_pairs_survive():
    # Legit UFC method markets carry no scope tag -> not a mismatch.
    assert not scope_mismatch(
        "Will Ciryl Gane win the fight by KO/TKO/DQ? - Ciryl Gane by KO/TKO/DQ",
        "Will Ciryl Gane win by KO, TKO, or DQ in Ciryl Gane vs Alex Pereira - Yes",
    )


def test_advance_vs_win_is_mismatch():
    assert scope_mismatch("Will France advance to the next round?",
                          "Will France win the match?")


def test_game_handicap_vs_match_winner_is_mismatch():
    assert scope_mismatch(
        "Will Joao Fonseca win at least 5.5 more games than Yannick Hanfmann? - Joao Fonseca -5.5 games",
        "Joao Fonseca vs. Yannick Hanfmann - Joao Fonseca",
    )


def test_set_winner_vs_match_winner_is_mismatch():
    assert scope_mismatch(
        "Will Suzan Lamens win set 2 in the Suzan Lamens vs Dalma Galfi match - Suzan Lamens",
        "Suzan Lamens vs. Dalma Galfi - Suzan Lamens",
    )


def test_goals_vs_goals_plus_assists_is_mismatch():
    assert scope_mismatch(
        "Virgil Van Dijk: 2+ goals - Virgil Van Dijk: 2+",
        "Will Virgil van Dijk record at least 2 goals+assists in NED vs JPN? - Yes",
    )


def test_assists_vs_goals_plus_assists_is_mismatch():
    assert scope_mismatch(
        "Sadio Mane: 1+ assists? - Sadio Mane: 1+",
        "Will Sadio Mane record at least 1 goals+assists in FRA vs SEN? - Yes",
    )


def test_same_metric_goals_is_not_mismatch():
    assert not scope_mismatch(
        "Ritsu Doan: 2+ goals - Ritsu Doan: 2+",
        "Will Ritsu Doan record at least 2 goals in NED vs JPN? - Yes",
    )


def test_score_or_assist_matches_goals_plus_assists():
    # Kalshi "score or assist" == Polymarket "1+ goals+assists" — legit, must survive.
    assert not scope_mismatch(
        "Frenkie De Jong: score or assist? - Frenkie De Jong",
        "Will Frenkie de Jong record at least 1 goals+assists in NED vs JPN? - Yes",
    )


def test_fight_win_vs_win_by_submission_is_mismatch():
    assert scope_mismatch(
        "Will Allan Nascimento win the Nascimento vs Raposo professional MMA fight? - Allan Nascimento",
        "Will Mitch Raposo win by submission in Allan Nascimento vs Mitch Raposo - Yes",
    )


def test_total_goals_metric_detected():
    assert "metric:goals" in scope_tags("Over 2.5 goals in the match?")


def test_exact_set_score_vs_match_winner_is_mismatch():
    # The live +43% false positive: Kalshi exact set score vs Polymarket match winner.
    assert scope_mismatch(
        "Will Brandon Nakashima win 2-1 in sets vs Buse? - Nakashima 2-1",
        "Brandon Nakashima vs. Ignacio Buse - Brandon Nakashima",
    )


def test_same_scoreline_is_not_mismatch():
    assert not scope_mismatch(
        "Will the final score be Sweden wins 2-0? - Sweden wins 2-0",
        "Will SWE vs TUN finish 2-0? - Yes",
    )


def test_different_scoreline_is_mismatch():
    assert scope_mismatch("Will SWE vs TUN finish 2-0? - Yes",
                          "Will SWE vs TUN finish 3-1? - Yes")


def test_year_in_title_is_not_a_scoreline():
    # 4-digit years must not be read as a scoreline.
    assert _scoreline_tag("World Cup match in 2026") is None


def test_scoreline_helper_imported():
    from bot.matching.scope import _scoreline_tag as _s
    assert _s("win 2-1 in sets") == "score:2-1"


def test_first_half_margin_vs_team_total_is_mismatch():
    # Kalshi 1H spread (margin) vs Polymarket 1H team total — different resolution.
    assert scope_mismatch(
        "Mexico wins by over 1.5 goals in the 1st Half? - Mexico wins the 1H by over 1.5 goals",
        "Will Mexico score more than 1.5 goals in the first half of MEX vs KOR? - Over",
    )


def test_win_fight_vs_go_the_distance_is_mismatch():
    assert scope_mismatch(
        "Will Melsik Baghdasaryan win the fight? - Melsik Baghdasaryan",
        "Will the fight go the distance in Melsik Baghdasaryan vs Murtazali Magomedov - Yes",
    )


def test_market_type_whitelist():
    from bot.matching.scope import is_tradeable_market_type as ok

    # Allowed: moneyline/draw winners, player props, fight method.
    assert ok("Will Haiti win against Brazil in the World Cup match? - Haiti")
    assert ok("Who will win in the upcoming esports event Spirit vs G2? - Spirit")
    assert ok("Cyle Larin: 2+ assists? - Cyle Larin: 2+")
    assert ok("Will Ciryl Gane win by KO/TKO/DQ? - Ciryl Gane by KO/TKO/DQ")
    assert ok("Will the fight end in a draw or no contest? - Draw/No Contest")

    # Excluded: novelty, halves, spread/margin, set, exact score, go-the-distance.
    assert not ok("What will the announcers say during Brazil vs Haiti? - Lincoln Financial Field")
    assert not ok("Will England win the 2nd Half? - England")
    assert not ok("Mexico wins by over 1.5 goals in the 1st Half? - Mexico")
    assert not ok("Will Suzan Lamens win set 2? - Suzan Lamens")
    assert not ok("Will the final score be Sweden wins 2-0? - Sweden wins 2-0")
    assert not ok("Will the fight go the distance? - Yes")
    # Futures/outright markets resolve over a whole tournament, not one match.
    assert not ok("Will Cape Verde be undefeated in the group stage? - Cape Verde")
    assert not ok("Will Brazil win the group? - Brazil")
    # ...but a single World Cup match winner is still allowed.
    assert ok("Will Portugal win against Congo DR in the World Cup match? - Portugal")


def test_kalshi_series_allowlist():
    from bot.matching.scope import is_allowed_kalshi_series as ok

    # Vetted series: match/game/fight winners and player props.
    assert ok("KXATPMATCH-26JUN17DIMGAN-DIM")
    assert ok("KXATPCHALLENGERMATCH-26JUN17ROTJAC-ROT")
    assert ok("KXDOTA2GAME-26JUN171900PLATNEM-NEM")
    assert ok("KXWNBAGAME-26JUN17DALGS-DAL")
    assert ok("KXUFCFIGHT-26JUN20BAGMAG-BAG")
    assert ok("KXWCGOAL-26JUN17PORCOD-CODCBAKAM11-1")
    assert ok("KXWCAST-26JUN17ENGCRO-ENGDRICE41-2")
    assert ok("KXNHLPTS-26JUN14CARVGK-VGKJEICHEL9-3")
    assert ok("KXWCSOA-26JUN17ENGCRO-ENGJBELLI5")

    # Exotic series excluded regardless of title wording.
    assert not ok("KXWCGSUNDEFEATED-26-CPV")
    assert not ok("KXWCMENTION-26JUN19TURPAR-HIST")
    assert not ok("KXWC2H-26JUN17ENGCRO-ENG")
    assert not ok("KXWC1HSPREAD-26JUN18MEXKOR-MEX2")
    assert not ok("KXWCSCORE-26JUN15IRINZL-IRI0NZL2")
    assert not ok("KXATPEXACTMATCH-26JUN17NAKBUS-NAK21")
    assert not ok("KXWTASETWINNER-26JUN15ZHESAK-1-ZHE")
    assert not ok("KXATPGSPREAD-26JUN14FONHAN-FON6")
