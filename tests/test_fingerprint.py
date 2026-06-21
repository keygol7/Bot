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


def test_polymarket_prop_subject_from_question():
    # Real format: Poly props read "Will <Player> record N+ <stat> ... - Yes". The
    # entity is in the question, not the Yes suffix. (No yes_sub_title, as in the DB.)
    k = from_kalshi("KXWCAST-26JUN18CANQAT-CANCLARIN17-2", "Cyle Larin: 2+ assists? - Cyle Larin: 2+")
    p = from_polymarket("astatc-fwc-can-qat-2026-06-18-a-fwccyllar-gte2",
                        "Will Cyle Larin record at least 2 assists in CAN vs QAT? - Yes")
    assert "larin" in k.subject and "larin" in p.subject
    assert are_complementary(k, p)


def test_prop_wrong_player_same_game_not_complementary():
    # Same game/metric/threshold but different players must still reject (the team codes
    # in the Poly subject can't cause a false align against the clean Kalshi player name).
    k = from_kalshi("KXWCGOAL-26JUN18CANQAT-CANCLARIN17-2", "Cyle Larin: 2+ goals - Cyle Larin: 2+")
    p = from_polymarket("astatc-fwc-can-qat-2026-06-18-g-fwcismkon-gte2",
                        "Will Ismael Kone record at least 2 goals in CAN vs QAT? - Yes")
    assert not are_complementary(k, p)


def test_fight_winner_vs_duration_not_complementary():
    # Fight WINNER must NOT match a "ends before round N" duration market (the Yes-side
    # matchup question names both fighters, so it has no clean YES side).
    k = from_kalshi("KXUFCFIGHT-26JUN20CUTSTI-STI",
                    "Will Navajo Stirling win the Cutelaba vs Stirling MMA fight? - Navajo Stirling")
    p = from_polymarket("astatc-ufc-ioncut-navsti-2026-06-20-rof-before-r3",
                        "Will Ion Cutelaba vs. Navajo Stirling end before round 3? - Yes")
    assert p.subject == frozenset()                # no entity recovered -> unmatchable side
    assert not are_complementary(k, p)


def test_real_fight_winner_is_complementary():
    k = from_kalshi("KXUFCFIGHT-26JUN20KAPHOR-HOR",
                    "Will Kyoji Horiguchi win the Kape vs Horiguchi MMA fight? - Kyoji Horiguchi")
    p = from_polymarket("aec-ufc-kyohor-mankap-2026-06-20",
                        "Kyoji Horiguchi vs. Manel Kape - Kyoji Horiguchi")
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


def test_fight_decision_prop_not_a_winner():
    # Kalshi "will X win" must NOT match Polymarket "go to a decision" (different event).
    k = from_kalshi("KXUFCFIGHT-26JUN20BAGMAG-BAG",
                    "Will Melsik Baghdasaryan win the fight? - Melsik Baghdasaryan")
    p = from_polymarket("astatc-ufc-melbag-murmag-2026-06-20-rov-dec",
                        "Will Melsik Baghdasaryan vs. Murtazali Magomedov go to a "
                        "decision, draw, or no contest? - Yes")
    assert k.metric == "winner" and p.metric == "unmatchable"
    assert not are_complementary(k, p)


def test_precision_rejects_cross_league_and_different_entities():
    # Cross-league: a Valorant "Brazil" team is not the World Cup "Brazil".
    kv = from_kalshi("KXVALORANTGAME-26JUN18TLELE-TL",
                     "Will Team Liquid Brazil win the Valorant match? - Team Liquid Brazil")
    pw = from_polymarket("atc-fwc-bra-hai-2026-06-19-bra",
                         "Will Brazil win against Haiti in the World Cup match? - Brazil")
    assert kv.league == "valorant" and pw.league == "soccer"
    assert not are_complementary(kv, pw)
    # Same surname, different first name (different player) -> reject.
    k1 = from_kalshi("KXWCGOAL-26JUN15KSAURU-URURARAUJ4-1", "Ronald Araujo: 1+ goals - Ronald Araujo: 1+")
    p1 = from_polymarket("astatc-fwc-ksa-uru-2026-06-15-goals-fwcmaxara-gte1",
                         "Will Maximiliano Araujo record at least 1 goals in KSA vs URU? - Yes")
    assert not are_complementary(k1, p1)
    # Two teams sharing a dropped suffix ("Gaming") must not align.
    kg = from_kalshi("KXDOTA2GAME-26JUN17LGDAMA-LGD", "Will LGD Gaming win - LGD Gaming")
    pg = from_polymarket("aec-dota2-agm-lgd-2026-06-17", "Who will win ... Amaru Gaming vs LGD - Amaru Gaming")
    assert not are_complementary(kg, pg)


def test_precision_keeps_spelling_variant_same_player():
    # Same player, surname spelled differently across venues -> still matches.
    k = from_kalshi("KXWCGOAL-26JUN17X-UZBAFAYZ-1", "Abbosbek Fayzullaev: 1+ goals - Abbosbek Fayzullaev: 1+")
    p = from_polymarket("astatc-fwc-x-2026-06-17-goals-y-gte1",
                        "Will Abbosbek Fayzullayev record at least 1 goals in X vs Y? - Yes")
    assert are_complementary(k, p)


def test_winner_same_team_different_game_not_complementary():
    # Same YES team (Dallas), DIFFERENT opponent -> different games, not a hedge. The
    # date gap can be inside the tolerance window (and doubleheaders share a date), so the
    # matchup (both teams) is what separates them.
    k = from_kalshi("KXWNBAGAME-26JUN15DALLV-DAL",
                    "Dallas Wings vs Las Vegas Aces - Dallas Wings", yes_sub_title="Dallas Wings")
    p = from_polymarket("aec-wnba-dal-gs-2026-06-18",
                        "Dallas Wings vs Golden State Valkyries - Dallas Wings", end_date="2026-06-18")
    assert not are_complementary(k, p)
    # Same game across venues still matches (matchups align).
    k2 = from_kalshi("KXWNBAGAME-26JUN18GSLV-GS",
                     "Golden State Valkyries vs Las Vegas Aces - Golden State Valkyries",
                     yes_sub_title="Golden State Valkyries")
    p2 = from_polymarket("aec-wnba-gs-lv-2026-06-18",
                         "Golden State Valkyries vs Las Vegas Aces - Golden State Valkyries",
                         end_date="2026-06-18")
    assert are_complementary(k2, p2)
    # Abbreviated team names ("Cloud9 NY" / "LA Thieves") must NOT trip the matchup check
    # against their spelled-out counterparts -> the true winner match is kept.
    k3 = from_kalshi("KXCODGAME-26JUN191630C9NYLAT-LAT",
                     "Cloud9 NY vs LA Thieves - Los Angeles Thieves", yes_sub_title="Los Angeles Thieves")
    p3 = from_polymarket("aec-cod-lat-c9ny-2026-06-19",
                         "Cloud9 New York vs Los Angeles Thieves - Los Angeles Thieves", end_date="2026-06-19")
    assert are_complementary(k3, p3)


def test_prop_cross_game_same_surname_not_complementary():
    # A "Rodri" assists prop (Spain, ESP vs KSA) must NOT bind to a "Brian Rodriguez"
    # assists prop (Uruguay, URU vs CPV) on the same day: same metric/threshold/date and
    # the surname prefix-aligns, but the structured event codes are different games.
    k = from_kalshi("KXWCAST-26JUN21ESPKSA-ESPRODR16-2", "Rodri: 2+ assists? - Rodri: 2+")
    p = from_polymarket("astatc-fwc-uru-cpv-2026-06-21-a-fwcbrirod-gte2",
                        "Will Brian Rodriguez record at least 2 assists in URU vs CPV? - Yes")
    assert k.event == frozenset({"espksa"}) and p.event == frozenset({"uru", "cpv"})
    assert not are_complementary(k, p)
    # Same player, SAME game still matches (event codes overlap: beliri contains bel).
    k2 = from_kalshi("KXWCGOAL-26JUN21BELIRI-BELLTROSS19-1", "Leandro Trossard: 1+ goals - Leandro Trossard: 1+")
    p2 = from_polymarket("astatc-fwc-bel-irn-2026-06-21-g-fwcleatro-gte1",
                         "Will Leandro Trossard record at least 1 goals in BEL vs IRN? - Yes")
    assert are_complementary(k2, p2)


def test_season_futures_winner_is_unmatchable():
    # A season/championship futures ("Will Jen win Love Island USA Season 8?") has no
    # single-game code in its ticker, so it's not a head-to-head winner -> unmatchable, and
    # can't bind to an unrelated tennis match via a name collision (Jen ~ Jeng).
    k = from_kalshi("KXLIUSAWINNERS-26-JEN", "Will Jen win Love Island USA Season 8? - Jen")
    assert k.event == frozenset() and k.metric == "unmatchable"
    p = from_polymarket("aec-itfwo-jujen-yekim-2026-06-21",
                        "Who will win in the upcoming tennis event Ju-Yun Jeng vs Ye Eun "
                        "Kim scheduled for June 21, 2026 at 5:30 AM UTC? - Ju-Yun Jeng")
    assert not are_complementary(k, p)
    # A real single-game winner keeps its game code and stays matchable.
    kw = from_kalshi("KXWNBAGAME-26JUN21GSLV-GS", "Golden State vs Las Vegas winner? - Golden State")
    assert kw.event == frozenset({"gslv"}) and kw.metric == "winner"


def test_first_goal_metric_matches_and_is_distinct():
    # "record the first goal" <-> "first to score": same event, distinct from a goal-COUNT
    # prop (so it can't false-match an anytime/N+ goals market).
    k = from_kalshi("KXFIRSTGOAL-26JUN18KORMEX-KOR",
                    "Will Korea Republic record the first goal of the game? - Korea Republic")
    p = from_polymarket("first-goal-kor-mex-2026-06-18",
                        "Will Korea Republic be the first to score a goal? - Yes",
                        end_date="2026-06-18")
    assert k.metric == "first_goal" and p.metric == "first_goal"
    assert are_complementary(k, p)
    # Must differ from a 2+ goals count prop for the same team:
    k2 = from_kalshi("KXWCGOAL-26JUN18KORMEX-KOR2", "Korea Republic: 2+ goals - Korea Republic: 2+")
    assert not are_complementary(p, k2)
