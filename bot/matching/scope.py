"""Deterministic scope/metric guard for cross-venue market matching.

A frequent false-positive class the local LLM rubber-stamps: two markets about the
same teams/event/date that actually resolve on a DIFFERENT scope, metric, or method —
e.g. "win the 2nd Half" vs "win the match", "win at least 5.5 more games" (handicap)
vs "win the match", "2+ goals" vs "2+ goals+assists", or "win by submission" vs
"win the fight". Same subject, same date, but not the same resolution, so
YES-on-one / NO-on-the-other is NOT a locked $1.

This catches the common cases structurally (no model needed): each title is reduced to
a set of tags (sub-period, handicap, set-winner, fight method, and the stat metric).
If the two titles' tag sets differ, they are different markets. Like the settlement-
date gate, it is a cheap backstop applied before the LLM and again when building the
streaming watchlist (so it also cleans already-cached verdicts).
"""

from __future__ import annotations

import re

# name -> regex marking a market scoped to a sub-period / different resolution than
# the plain "win the match/event/fight" question.
_SCOPE_PATTERNS: dict[str, str] = {
    "first_half": r"\b(1st|first)\s+half\b",
    "second_half": r"\b(2nd|second)\s+half\b",
    "halftime": r"\bhalf[\s-]?time\b|\bat\s+the\s+half\b",
    "first_period": r"\b(1st|first)\s+period\b",
    "advance": r"\badvance\b|\bto\s+qualif|\bqualify\b|\bto\s+reach\b",
    "extra_time": r"\bextra\s+time\b|\bover[\s-]?time\b",
    "penalties": r"\bpenalt",
    "regulation": r"\bregulation\b|\b90\s+minutes\b",
    "clean_sheet": r"\bclean\s+sheet\b",
    "both_score": r"\bboth\s+teams?\s+to\s+score\b|\bbtts\b",
    "exact_score": r"\bexact\s+(score|outcome)\b|\bcorrect\s+score\b",
    "to_nil": r"\bto\s+nil\b",
    # spread / alternative-line markets (tennis games handicap, etc.)
    "handicap": r"\bmore\s+games\s+than\b|\b-\s*\d+(\.\d+)?\s+games\b|"
                r"\bgames?\s+(handicap|spread)\b|\bhandicap\b|\bspread\b",
    # set-level tennis markets vs the full match
    "set_winner": r"\bwin\s+set\s+\d\b|\bset\s+\d\s+winner\b|\bset\s+winner\b|"
                  r"\bwin\s+the\s+\d(st|nd|rd|th)\s+set\b",
    # fight method-of-victory (must match on both sides or neither)
    "method_ko": r"\bko\s*/\s*tko\b|\bko\s*,\s*tko\b|\bby\s+ko\b|\bby\s+tko\b|"
                 r"\bknockout\b|\bko/tko/dq\b",
    "method_sub": r"\bby\s+submission\b|\bsubmission\b",
    "method_decision": r"\bby\s+decision\b|\bunanimous\s+decision\b",
    # winning MARGIN / spread ("wins by over 1.5 goals") — NOT a team total ("scores
    # over 1.5 goals"). One side margin, the other total -> different markets.
    "margin": r"\bwins?\s+by\s+(over\s+|under\s+|more\s+than\s+)?\d|\bgoal\s+(margin|spread)\b",
    # fight goes the distance (duration) — NOT "win the fight" (winner).
    "distance": r"\bgo(es)?\s+the\s+distance\b",
    # tournament/outright FUTURES (resolve over a whole event) — NOT a single match.
    # e.g. "undefeated in the group stage" vs "win the match".
    "futures": r"\bundefeated\b|\btop\s+scorer\b|\bgolden\s+boot\b|\bgroup\s+winner\b|"
               r"\bwin\s+the\s+group\b|\bto\s+reach\s+the\b|"
               r"\bto\s+win\s+the\s+(world\s+cup|tournament|title|trophy|cup)\b",
    # round-specific fight market ("win in Round 3") — NOT "win the fight".
    "round": r"\bin\s+round\s+\d\b|\bround\s+of\s+victory\b|\bby\s+round\s+\d\b|"
             r"\bwins?\s+in\s+round\b",
}

# The stat a player-prop market resolves on. Mutually exclusive, checked in order so
# "goals+assists" / "score or assist" is not mistaken for plain "goals" or "assists".
_METRIC_RULES: list[tuple[str, str]] = [
    ("ga", r"score\s+or\s+assist|goals?\s*\+\s*assists?|goals?\s+and\s+assists?|"
           r"goal\s+or\s+assist|goals?\s*/\s*assists?"),
    ("assists", r"\bassists?\b"),
    ("goals", r"\bgoals?\b"),
    ("points", r"\bpoints?\b"),
    ("saves", r"\bsaves?\b"),
    ("shots", r"\bshots?\b"),
]


def _metric_tag(t: str) -> str | None:
    for name, pat in _METRIC_RULES:
        if re.search(pat, t):
            return f"metric:{name}"
    return None


# A scoreline like "2-1", "2 - 0", "3-2" — marks an exact-score market. 1-2 digits so
# years ("2026") can't match. Encoded WITH its value so different scores also mismatch.
_SCORELINE_RE = re.compile(r"\b(\d{1,2})\s*[-–]\s*(\d{1,2})\b")


def _scoreline_tag(t: str) -> str | None:
    m = _SCORELINE_RE.search(t)
    return f"score:{m.group(1)}-{m.group(2)}" if m else None


def _threshold_tag(t: str) -> str | None:
    """The required count for a player prop: "2+ goals"/"at least 2"/"3+" -> thr:2/3;
    "score or assist" implies thr:1. Distinguishes "1+ assists" from "2+ assists"."""
    if re.search(r"\bscore\s+or\s+assist\b", t):
        return "thr:1"
    m = re.search(r"\b(\d+)\s*\+", t) or re.search(r"\bat\s+least\s+(\d+)\b", t)
    return f"thr:{int(m.group(1))}" if m else None


def scope_tags(title: str) -> frozenset[str]:
    """The set of scope/metric qualifiers present in ``title`` (lowercased match)."""
    if not title:
        return frozenset()
    t = title.lower()
    tags = {name for name, pat in _SCOPE_PATTERNS.items() if re.search(pat, t)}
    metric = _metric_tag(t)
    if metric:
        tags.add(metric)
    scoreline = _scoreline_tag(t)
    if scoreline:
        tags.add(scoreline)
    threshold = _threshold_tag(t)
    if threshold:
        tags.add(threshold)
    return frozenset(tags)


def scope_mismatch(title_a: str, title_b: str) -> bool:
    """True if the two titles carry DIFFERENT scope/metric qualifiers.

    Equal tag sets (including both empty = both plain "win" markets, or both the same
    sub-period/metric) are NOT a mismatch. Only a difference — e.g. one "2nd half" or
    "goals+assists" or "by submission", the other plain — is rejected.
    """
    return scope_tags(title_a) != scope_tags(title_b)


# Scope tags that mark an "exotic" market the matcher has repeatedly mishandled. Even
# when both sides agree, we exclude these from the safe tradeable set by default.
_EXOTIC_SCOPE = frozenset({
    "first_half", "second_half", "halftime", "first_period", "advance", "extra_time",
    "penalties", "regulation", "clean_sheet", "both_score", "exact_score", "to_nil",
    "handicap", "set_winner", "margin", "distance", "futures", "round",
})
# A market resolving on a clear winner/draw outcome (vs a novelty like "what will the
# announcers say", which matches none of these and is therefore excluded).
_WINNER_RE = re.compile(
    r"\bwins?\b|\bbeat\b|\bdefeat|\bwinner\b|who will win|\bto win\b|\bdraw\b|\bno contest\b"
)


def is_tradeable_market_type(title: str) -> bool:
    """Whitelist of market types the matcher handles reliably: plain moneyline
    winners (incl. draw/no-contest), same-metric player props (goals/assists/points/
    saves/shots), and fight method-of-victory. Everything else — halves, spreads/
    margins, set winners, exact scores, go-the-distance, announcer novelties, etc. —
    is excluded. This is a positive allowlist: an unrecognized type fails by default.
    """
    if not title:
        return False
    tags = scope_tags(title)
    if any(t in _EXOTIC_SCOPE for t in tags) or any(t.startswith("score:") for t in tags):
        return False
    if any(t.startswith("metric:") for t in tags):
        return True  # player prop with a stat metric (goals/assists/points/...)
    if any(t in {"method_ko", "method_sub", "method_decision"} for t in tags):
        return True  # fight method-of-victory (validated as reliable)
    return bool(_WINNER_RE.search(title.lower()))  # plain winner; novelties fail this


# Kalshi encodes the market TYPE in the ticker series (the token after "KX", before the
# first "-"). This is far more reliable than parsing free-text titles, which keep
# leaking exotic types ("win all 3 ... group stage" reads as a winner). Allow ONLY the
# series we've validated; an unknown series is excluded by default.
_ALLOWED_KALSHI_SERIES = re.compile(
    r"^(?:"
    r"(?:ATP|WTA|ITF|ITFW)(?:CHALLENGER)?MATCH"             # tennis match winners
    r"|[A-Z0-9]*GAME"                                       # game winners (esports, WNBA, ...)
    r"|UFCFIGHT"                                            # MMA fight winners
    r"|[A-Z0-9]*(?:GOALS?|ASTS?|ASSISTS?|PTS|POINTS?|SOA)"  # player props
    r")$"
)


def kalshi_series(ticker: str) -> str:
    """The Kalshi series token, e.g. 'KXWCGOAL-26JUN17...' -> 'WCGOAL'."""
    t = ticker[2:] if ticker.startswith("KX") else ticker
    return t.split("-", 1)[0]


def is_allowed_kalshi_series(ticker: str) -> bool:
    """True only for vetted Kalshi series (match/game/fight winners, player props).
    Excludes exotic series — MENTION, GSUNDEFEATED, 2H, 1HSPREAD, SCORE, EXACTMATCH,
    SETWINNER, GSPREAD, … — regardless of how their title reads."""
    return bool(_ALLOWED_KALSHI_SERIES.match(kalshi_series(ticker)))

