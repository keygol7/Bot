"""Structured fingerprint matching: true arbs complement, false positives don't."""

from bot.matching.fingerprint import (
    are_complementary,
    from_kalshi,
    from_polymarket,
    kalshi_metric,
)


def _pair(kt, ktitle, ysub, pslug, ptitle, ed):
    return (from_kalshi(kt, ktitle, yes_sub_title=ysub),
            from_polymarket(pslug, ptitle, end_date=ed))


def test_kalshi_series_to_metric():
    assert kalshi_metric("KXWCGOAL-26JUN17X-Y") == "goals"
    assert kalshi_metric("KXCS2GAME-26JUN18X-Y") == "winner"
    assert kalshi_metric("KXUFCFIGHT-26JUN18X-Y") == "winner"
    assert kalshi_metric("KXATPMATCH-26JUN18X-Y") == "winner"
    assert kalshi_metric("KXNBAASTS-26JUN18X-Y") == "assists"
    # Novelty / futures / exotic -> explicitly unmatchable (no allow/deny list).
    assert kalshi_metric("KXWCMENTION-26JUN22X-Y") == "unmatchable"
    assert kalshi_metric("KXGSUNDEFEATED-X-Y") == "unmatchable"
    assert kalshi_metric("KX1HSPREAD-X-Y") == "unmatchable"


def test_true_goals_prop_is_complementary():
    k, p = _pair(
        "KXWCGOAL-26JUN17UZBCOL-UZBAFAYZU11-1", "Uzbekistan vs Colombia - Fayzullaev: 1+",
        "Abbosbek Fayzullaev: 1+",
        "astatc-fwc-uzb-col-2026-06-17-g-fwcabbfay-gte1",
        "Abbosbek Fayzullayev 1+ goals - Abbosbek Fayzullayev", "2026-06-17")
    assert k.metric == "goals" and p.metric == "goals" and k.threshold == 1 == p.threshold
    assert are_complementary(k, p)


def test_true_game_winner_is_complementary():
    # Polymarket game titles carry no literal "win" word — just "A vs B - Outcome".
    k, p = _pair(
        "KXCODGAME-26JUN191630C9NYLAT-LAT", "Cloud9 NY vs LA Thieves - Los Angeles Thieves",
        "Los Angeles Thieves",
        "aec-cod-lat-c9ny-2026-06-19", "Cloud9 New York vs Los Angeles Thieves - Los Angeles Thieves",
        "2026-06-19")
    assert k.metric == "winner" and p.metric == "winner"
    assert are_complementary(k, p)


def test_mention_novelty_not_complementary():
    k, p = _pair(
        "KXWCMENTION-26JUN22JORDZA-CAPT", "Jordan vs Algeria - Captain", "Captain",
        "atc-fwc-jor-alg-2026-06-22-jor", "Jordan to advance - Jordan", "2026-06-22")
    assert not are_complementary(k, p)             # unmatchable metric


def test_win_vs_advance_not_complementary():
    # Same teams/date, but "win the match" != "advance in tournament" (scope differs).
    k = from_kalshi("KXCS2GAME-26JUN22NORSEN-NOR", "Norway vs Senegal - Norway", yes_sub_title="Norway")
    p = from_polymarket("atc-fwc-nor-sen-2026-06-22-nor", "Norway to advance to knockout - Norway",
                        end_date="2026-06-22")
    assert "advance" in p.scope
    assert not are_complementary(k, p)


def test_threshold_mismatch_not_complementary():
    k, p = _pair(
        "KXWCGOAL-26JUN17UZBCOL-X-2", "Uzbekistan vs Colombia - Player: 2+", "Player: 2+",
        "x-g-player-gte1", "Player 1+ goals - Player", "2026-06-17")
    assert k.threshold == 2 and p.threshold == 1
    assert not are_complementary(k, p)


def test_different_metric_not_complementary():
    # goals vs goals-or-assists (the classic prop false positive): metrics differ.
    k = from_kalshi("KXWCGOAL-26JUN17X-P-1", "A vs B - Player: 1+", yes_sub_title="Player: 1+")
    p = from_polymarket("x-soa-player-gte1", "Player 1+ goals or assists - Player", end_date="2026-06-17")
    assert k.metric == "goals" and p.metric != "goals"
    assert not are_complementary(k, p)


def test_date_gap_not_complementary():
    k = from_kalshi("KXCODGAME-26JUN19X-LAT", "A vs LAT - LAT", yes_sub_title="LAT")
    p = from_polymarket("aec-cod-a-lat-2026-09-01", "A vs LAT - LAT", end_date="2026-09-01")
    assert not are_complementary(k, p)             # months apart -> not the same event


def test_unfingerprintable_fails_closed():
    k = from_kalshi("KXNObSeriesHere-26JUN18X-Y", "some novelty thing", yes_sub_title="Z")
    p = from_polymarket("weird-slug", "unrelated novelty", end_date=None)
    assert not k.matchable and not p.matchable
    assert not are_complementary(k, p)
