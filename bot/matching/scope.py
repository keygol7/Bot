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


def scope_tags(title: str) -> frozenset[str]:
    """The set of scope/metric qualifiers present in ``title`` (lowercased match)."""
    if not title:
        return frozenset()
    t = title.lower()
    tags = {name for name, pat in _SCOPE_PATTERNS.items() if re.search(pat, t)}
    metric = _metric_tag(t)
    if metric:
        tags.add(metric)
    return frozenset(tags)


def scope_mismatch(title_a: str, title_b: str) -> bool:
    """True if the two titles carry DIFFERENT scope/metric qualifiers.

    Equal tag sets (including both empty = both plain "win" markets, or both the same
    sub-period/metric) are NOT a mismatch. Only a difference — e.g. one "2nd half" or
    "goals+assists" or "by submission", the other plain — is rejected.
    """
    return scope_tags(title_a) != scope_tags(title_b)
