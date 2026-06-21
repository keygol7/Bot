"""Cheap, deterministic lexical pre-filter for cross-venue market matching.

The goal is to *shortlist* candidate pairs cheaply so the (more expensive) local
LLM only judges the ambiguous ones. This default uses token Jaccard similarity —
standard-library only, no model required — and is easily swapped for real embedding
cosine similarity later (same ``candidate_pairs`` interface).

This is a pre-filter, not a decision: a high lexical score still must be confirmed
by the LLM resolution check before any trade.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable

from bot.models import MarketQuote

EmbedFn = Callable[[list[str]], list[list[float]]]

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


def cosine(u: list[float], v: list[float]) -> float:
    """Cosine similarity of two vectors, in [-1, 1] (0 if either is zero-length)."""
    dot = sum(a * b for a, b in zip(u, v, strict=False))
    nu = math.sqrt(sum(a * a for a in u))
    nv = math.sqrt(sum(b * b for b in v))
    if nu == 0 or nv == 0:
        return 0.0
    return dot / (nu * nv)


def semantic_candidate_pairs(
    group_a: list[MarketQuote],
    group_b: list[MarketQuote],
    embed_fn: EmbedFn,
    *,
    threshold: float = 0.8,
) -> list[Candidate]:
    """Shortlist cross-venue pairs by embedding **cosine** similarity.

    Far better than lexical overlap at matching the same event worded differently
    across venues. ``embed_fn`` maps a list of titles to vectors (one batch each
    side). Returns candidates at/above ``threshold``, highest similarity first.
    """
    # Drop blank titles — they can't match and an empty string makes some embedding
    # servers (e.g. Ollama) return 400.
    group_a = [q for q in group_a if q.title and q.title.strip()]
    group_b = [q for q in group_b if q.title and q.title.strip()]
    if not group_a or not group_b:
        return []
    vecs_a = embed_fn([q.title for q in group_a])
    vecs_b = embed_fn([q.title for q in group_b])

    sim = _similarity_matrix(vecs_a, vecs_b)  # sim[i][j] = cosine(a_i, b_j)
    out: list[Candidate] = []
    for i, qa in enumerate(group_a):
        row = sim[i]
        for j, qb in enumerate(group_b):
            if qa.venue == qb.venue:
                continue
            score = row[j]
            if score >= threshold:
                out.append(Candidate(a=qa, b=qb, score=float(score)))
    out.sort(key=lambda c: c.score, reverse=True)
    return out


def _similarity_matrix(vecs_a, vecs_b):
    """All-pairs cosine similarity. Uses numpy when available (orders of magnitude
    faster on large boards), else a pure-Python fallback."""
    try:
        import numpy as np

        a = np.asarray(vecs_a, dtype=float)
        b = np.asarray(vecs_b, dtype=float)
        a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
        b /= np.linalg.norm(b, axis=1, keepdims=True) + 1e-12
        return a @ b.T
    except ImportError:
        return [[cosine(va, vb) for vb in vecs_b] for va in vecs_a]
