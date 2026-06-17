from bot.matching.scope import scope_mismatch, scope_tags


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
