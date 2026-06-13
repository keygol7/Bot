"""Cheap, deterministic lexical pre-filter for cross-venue market matching.

The goal is to *shortlist* candidate pairs cheaply so the (more expensive) local
LLM only judges the ambiguous ones. This default uses token Jaccard similarity —
standard-library only, no model required — and is easily swapped for real embedding
cosine similarity later (same ``candidate_pairs`` interface).

This is a pre-filter, not a decision: a high lexical score still must be confirmed
by the LLM resolution check before any trade.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from bot.models import MarketQuote

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "to", "of", "in", "on", "for", "and", "or", "will",
    "be", "is", "are", "by", "at", "with", "than", "this", "that",
}


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS}


def lexical_similarity(a: str, b: str) -> float:
    """Jaccard similarity over normalized, stop-word-filtered tokens, in [0, 1]."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


@dataclass
class Candidate:
    a: MarketQuote
    b: MarketQuote
    score: float


def candidate_pairs(
    group_a: list[MarketQuote],
    group_b: list[MarketQuote],
    *,
    threshold: float = 0.4,
) -> list[Candidate]:
    """Shortlist cross-venue pairs whose titles score at or above ``threshold``.

    ``group_a`` and ``group_b`` are quotes from two different venues. Returns
    candidates sorted by descending similarity.
    """
    out: list[Candidate] = []
    for qa in group_a:
        for qb in group_b:
            if qa.venue == qb.venue:
                continue
            score = lexical_similarity(qa.title, qb.title)
            if score >= threshold:
                out.append(Candidate(a=qa, b=qb, score=score))
    out.sort(key=lambda c: c.score, reverse=True)
    return out
