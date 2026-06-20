"""Structured contract fingerprints for cross-venue matching (shadow / comparison).

The live filter (``bot/data/store.confirmed_pairs`` + ``bot/matching/scope``) gates by
*positive allowlists* — only a hand-curated set of Kalshi series and market types is
admitted, everything else dropped. That is safe but caps volume by throwing away whole
categories.

This module is the alternative: reduce each market to a structured
:class:`ContractFingerprint` (metric / scope / threshold / subject / date) and admit a
pair only when the two fingerprints are provable logical complements — same underlying
proposition, opposite sides. A false match (e.g. a broadcast-novelty "mention" market
vs a team-advance market) fails because the *metrics differ*, not because a series is
off a list — so any new category is admitted automatically when its structure matches.

This is deterministic and *fails closed*: a market we cannot fingerprint (unknown
metric) is non-matchable, exactly like today. It is wired only into the comparison
tooling for now; the live path is unchanged until the shadow report validates it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from bot.matching.scope import (
    _METRIC_RULES,
    _threshold_tag,
    kalshi_series,
    scope_tags,
)
from bot.timeutil import parse_iso8601

# Scope tags that are not a resolution *period/method* but a stat metric/score/threshold
# (handled separately in the fingerprint), so they're excluded from the scope set.
_NON_SCOPE_PREFIXES = ("metric:", "score:", "thr:")

# Metric a market resolves on. ``winner`` = moneyline/draw; player props carry their
# stat; ``unmatchable`` = no clean cross-venue complement (novelty/futures/exotic);
# ``unknown`` = could not classify (fails closed, same as unmatchable).
_UNMATCHABLE = "unmatchable"
_UNKNOWN = "unknown"

# Kalshi encodes the market type in the series token (more reliable than the title).
# Classify EVERY series into a canonical metric — including an explicit unmatchable
# bucket — instead of an allow/deny list. Order matters (first hit wins).
_KALSHI_SERIES_METRIC: list[tuple[str, str]] = [
    # --- no clean complement: novelties, futures, exotic lines ---
    ("MENTION", _UNMATCHABLE), ("UNDEFEATED", _UNMATCHABLE), ("GROUP", _UNMATCHABLE),
    ("SPREAD", _UNMATCHABLE), ("HANDICAP", _UNMATCHABLE), ("EXACT", _UNMATCHABLE),
    ("EXACTMATCH", _UNMATCHABLE), ("SETWINNER", _UNMATCHABLE), ("SCORE", _UNMATCHABLE),
    ("1H", _UNMATCHABLE), ("2H", _UNMATCHABLE), ("HALF", _UNMATCHABLE),
    # --- player props (stat metric) ---
    ("GOAL", "goals"), ("ASSIST", "assists"), ("ASTS", "assists"), ("AST", "assists"),
    ("SOA", "ga"), ("PTS", "points"), ("POINT", "points"), ("SAVE", "saves"),
    ("SHOT", "shots"),
    # --- moneyline / match winners ---
    ("FIGHT", "winner"), ("MATCH", "winner"), ("GAME", "winner"),
    ("WINNER", "winner"), ("MONEYLINE", "winner"), ("RESULT", "winner"),
]

_DATE_IN_TICKER = re.compile(r"(\d{2}[A-Z]{3}\d{2})")     # e.g. 26JUN17
_DATE_IN_SLUG = re.compile(r"(\d{4}-\d{2}-\d{2})")        # e.g. 2026-06-17
_WORD = re.compile(r"[a-z0-9]+")
# Very common stop-tokens in titles/outcomes that carry no entity signal.
_STOP = frozenset({
    "the", "to", "win", "wins", "winner", "vs", "v", "match", "game", "will",
    "score", "scores", "or", "and", "of", "a", "an", "at", "in", "on", "for",
    "yes", "no", "1", "2", "goals", "goal", "assist", "assists", "points", "point",
})


def _title_metric(title: str) -> str:
    """Metric inferred from a free-text title: a stat prop, else moneyline winner."""
    t = (title or "").lower()
    for name, pat in _METRIC_RULES:
        if re.search(pat, t):
            return "ga" if name == "ga" else name
    if re.search(r"\bwins?\b|\bbeat\b|\bdefeat|\bwinner\b|to win|\bdraw\b|\bno contest\b", t):
        return "winner"
    # A bare "Team A vs Team B - Outcome" matchup with no stat/exotic scope is a
    # moneyline winner (Polymarket game titles carry no literal "win" word). Any
    # advance/half/spread scope is still separated by the scope set downstream.
    if re.search(r"\bvs?\.?\b", t):
        return "winner"
    return _UNKNOWN


def kalshi_metric(ticker: str) -> str:
    series = kalshi_series(ticker).upper()
    for needle, metric in _KALSHI_SERIES_METRIC:
        if needle in series:
            return metric
    return _UNKNOWN


def _scope_only(title: str) -> frozenset[str]:
    """Scope/period/method tags only (drop metric/score/threshold, tracked separately)."""
    return frozenset(
        t for t in scope_tags(title) if not t.startswith(_NON_SCOPE_PREFIXES)
    )


def _subject_tokens(text: str) -> frozenset[str]:
    """Entity tokens (team/player) from an outcome label, minus stopwords & digits."""
    return frozenset(
        w for w in _WORD.findall((text or "").lower())
        if w not in _STOP and not w.isdigit() and len(w) > 1
    )


def _outcome_from_title(title: str) -> str:
    """The disambiguating outcome both venues append as ' - <outcome>'."""
    return title.rsplit(" - ", 1)[1] if title and " - " in title else (title or "")


@dataclass(frozen=True)
class ContractFingerprint:
    venue: str
    metric: str                  # winner / goals / ... / unmatchable / unknown
    scope: frozenset            # period/method qualifiers (1st_half, regulation, ...)
    threshold: int | None        # player-prop count (1+, 2+) or None
    subject: frozenset           # normalized YES-outcome entity tokens
    date: float | None           # event date (epoch), if parseable

    @property
    def matchable(self) -> bool:
        return self.metric not in (_UNMATCHABLE, _UNKNOWN)


def _threshold_int(title: str) -> int | None:
    tag = _threshold_tag((title or "").lower())
    return int(tag.split(":", 1)[1]) if tag else None


def from_kalshi(ticker: str, title: str, yes_sub_title: str = "",
                close_time: float | None = None) -> ContractFingerprint:
    metric = kalshi_metric(ticker)
    date = close_time
    if date is None:
        m = _DATE_IN_TICKER.search(ticker)
        if m:
            # 26JUN17 -> 2026-06-17 (assume 20YY); parse via the shared iso parser.
            from datetime import datetime, timezone
            try:
                dt = datetime.strptime(m.group(1), "%y%b%d").replace(tzinfo=timezone.utc)
                date = dt.timestamp()
            except ValueError:
                date = None
    subject = _subject_tokens(yes_sub_title) or _subject_tokens(_outcome_from_title(title))
    return ContractFingerprint(
        venue="kalshi", metric=metric, scope=_scope_only(title),
        threshold=_threshold_int(title), subject=subject, date=date,
    )


def from_polymarket(slug: str, title: str, end_date: str | None = None,
                    outcome: str = "") -> ContractFingerprint:
    metric = _title_metric(title)
    date = None
    if end_date:
        date = parse_iso8601(end_date)
    if date is None:
        m = _DATE_IN_SLUG.search(slug or "")
        if m:
            date = parse_iso8601(m.group(1))
    subject = _subject_tokens(outcome) or _subject_tokens(_outcome_from_title(title))
    return ContractFingerprint(
        venue="polymarket_us", metric=metric, scope=_scope_only(title),
        threshold=_threshold_int(title), subject=subject, date=date,
    )


def _subjects_align(a: frozenset, b: frozenset) -> bool:
    """The YES outcomes refer to the same entity. Exact token overlap, or a shared
    distinctive prefix (handles 'nor'/'norway', team-code vs full-name)."""
    if not a or not b:
        return False
    if a & b:
        return True
    for x in a:
        for y in b:
            lo, hi = (x, y) if len(x) <= len(y) else (y, x)
            if len(lo) >= 3 and hi.startswith(lo):
                return True
    return False


def complement_reason(a: ContractFingerprint, b: ContractFingerprint,
                      max_gap_days: float = 3.0) -> str:
    """``"ok"`` if complementary, else the first failing check (for shadow diagnostics)."""
    if not a.matchable:
        return f"a-{a.metric}"
    if not b.matchable:
        return f"b-{b.metric}"
    if a.metric != b.metric:
        return f"metric {a.metric}!={b.metric}"
    if a.scope != b.scope:
        return f"scope {sorted(a.scope)}!={sorted(b.scope)}"
    if a.threshold != b.threshold:
        return f"threshold {a.threshold}!={b.threshold}"
    if a.date is not None and b.date is not None and abs(a.date - b.date) > max_gap_days * 86400:
        return "date gap"
    if not _subjects_align(a.subject, b.subject):
        return f"subject {sorted(a.subject)}!={sorted(b.subject)}"
    return "ok"


def are_complementary(a: ContractFingerprint, b: ContractFingerprint,
                      max_gap_days: float = 3.0) -> bool:
    """True iff buying YES on ``a`` and NO on ``b`` is the same proposition, opposite
    sides: same (matchable) metric, scope, threshold, event date, and the YES outcomes
    refer to the same entity. Fails closed on anything unparseable."""
    return complement_reason(a, b, max_gap_days) == "ok"
