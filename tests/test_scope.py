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


def test_total_goals_tag_detected():
    assert "total_goals" in scope_tags("Over 2.5 goals in the match?")
