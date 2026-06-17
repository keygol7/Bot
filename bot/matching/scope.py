"""Deterministic scope/period guard for cross-venue market matching.

A frequent false-positive class the LLM rubber-stamps: two markets about the same
teams/event/date that actually resolve on a DIFFERENT scope — e.g. Kalshi's "Will
Portugal win the 2nd Half?" vs Polymarket's "Will Portugal win the match?". Same
subject, same date, but not the same resolution, so YES-on-one / NO-on-the-other is
NOT a locked $1.

This catches the common cases structurally (no model needed): if one title carries a
scope qualifier (half, to-advance, extra time, total goals, exact score, …) that the
other does not, they are different markets. Like the settlement-date gate, it is a
cheap backstop applied before the LLM and again when building the streaming watchlist.
"""

from __future__ import annotations

import re

# name -> regex marking a market scoped to a sub-period or a different resolution
# than the plain "win the match / win the event" question.
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
    "total_goals": r"\b(over|under)\b.{0,25}\bgoals?\b|\bgoals?\b.{0,25}\b(over|under)\b",
    "exact_score": r"\bexact\s+(score|outcome)\b|\bcorrect\s+score\b",
    "to_nil": r"\bto\s+nil\b|\bto\s+win\s+to\s+nil\b",
}


def scope_tags(title: str) -> frozenset[str]:
    """The set of scope qualifiers present in ``title`` (lowercased match)."""
    if not title:
        return frozenset()
    t = title.lower()
    return frozenset(name for name, pat in _SCOPE_PATTERNS.items() if re.search(pat, t))


def scope_mismatch(title_a: str, title_b: str) -> bool:
    """True if the two titles carry DIFFERENT scope qualifiers.

    Equal tag sets (including both empty = both plain "win" markets, or both the same
    sub-period) are NOT a mismatch. Only a difference — e.g. one "2nd half", the other
    full match — is rejected.
    """
    return scope_tags(title_a) != scope_tags(title_b)
