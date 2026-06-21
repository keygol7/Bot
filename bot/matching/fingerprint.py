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
# Player-prop stat metrics — these carry the entity in the question ("Will <Player>
# record N+ <stat>"), so the Yes-outcome subject may be recovered from the question.
_STAT_METRICS = frozenset({"goals", "assists", "ga", "points", "saves", "shots"})
# Metrics whose YES subject lives in the QUESTION (not the Yes/No suffix): the stat
# props plus first-to-score ("Will <Team> be the first to score a goal? - Yes").
_QUESTION_SUBJECT_METRICS = _STAT_METRICS | {"first_goal"}

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

# League/sport, extracted from the Kalshi series and the Polymarket slug. Two markets
# in DIFFERENT leagues are not the same event even if they share a team token (a Valorant
# "Brazil" team vs the World Cup "Brazil"; "Team Nemesis" in CS2 vs Dota2). First match
# wins, so more specific needles precede generic ones.
_LEAGUE_RULES: list[tuple[str, str]] = [
    ("VALORANT", "valorant"), ("DOTA2", "dota2"), ("DOTA", "dota2"),
    ("CS2", "cs2"), ("CSGO", "cs2"), ("LOL", "lol"), ("COD", "cod"),
    ("UFC", "mma"), ("BELLATOR", "mma"), ("PFL", "mma"),
    ("ATP", "tennis"), ("WTA", "tennis"), ("ITF", "tennis"), ("TENNIS", "tennis"),
    ("WNBA", "basketball"), ("NBA", "basketball"),
    ("MLB", "baseball"), ("NHL", "hockey"),
    ("FWC", "soccer"), ("WCGOAL", "soccer"), ("WCAST", "soccer"), ("WCFTTS", "soccer"),
    ("FIFA", "soccer"), ("SOCCER", "soccer"), ("WC", "soccer"),
]


def _league_of(text: str) -> str | None:
    s = (text or "").upper()
    for needle, league in _LEAGUE_RULES:
        if needle in s:
            return league
    return None


_DATE_IN_TICKER = re.compile(r"(\d{2}[A-Z]{3}\d{2})")     # e.g. 26JUN17
_DATE_IN_SLUG = re.compile(r"(\d{4}-\d{2}-\d{2})")        # e.g. 2026-06-17
_WORD = re.compile(r"[a-z0-9]+")
# Very common stop-tokens in titles/outcomes that carry no entity signal. Includes
# generic org/club words ("gaming", "team", "fc", ...) that otherwise let two DIFFERENT
# teams falsely align on a shared suffix (e.g. "LGD Gaming" vs "Amaru Gaming").
_STOP = frozenset({
    "the", "to", "win", "wins", "winner", "vs", "v", "match", "game", "will",
    "score", "scores", "or", "and", "of", "a", "an", "at", "in", "on", "for",
    "yes", "no", "1", "2", "goals", "goal", "assist", "assists", "points", "point",
    "be", "first",
    "gaming", "esports", "team", "club", "fc", "sc", "cf", "united", "city", "afc",
})


def _title_metric(title: str) -> str:
    """Metric inferred from a free-text title: a stat prop, else moneyline winner."""
    t = (title or "").lower()
    # First-to-score: a DISTINCT metric, checked before the goal-count rules so "the
    # first goal" isn't folded into "goals" (which would falsely match an anytime-goal
    # or N+ goals prop). "Will <Team> record the first goal" / "be the first to score".
    if re.search(r"\bfirst goal\b|first to score|record the first goal|opening goal", t):
        return "first_goal"
    for name, pat in _METRIC_RULES:
        if re.search(pat, t):
            return "ga" if name == "ga" else name
    # Fight method/duration props ("go to a decision", "go the distance", "by KO/
    # submission", "no contest") have NO clean cross-venue WINNER complement — a fighter
    # can win by KO *or* decision — so they must never match a moneyline winner. Mark
    # unmatchable (these phrasings don't occur in a soccer "draw" moneyline).
    if re.search(r"\bdecision\b|go the distance|\bno contest\b|"
                 r"by (ko|knockout|submission|tko)\b|method of", t):
        return _UNMATCHABLE
    if re.search(r"\bwins?\b|\bbeat\b|\bdefeat|\bwinner\b|to win|\bdraw\b", t):
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


# Template / sport / scheduling filler that surrounds the two teams in a matchup title
# ("Who will win in the upcoming <sport> event A vs B scheduled for <Month> ... UTC?",
# "Will <Y> win the A vs B professional MMA fight ...?"). Stripped so the matchup reduces
# to just the team/player tokens. Tournament place-names can't all be listed, but the
# subset alignment tolerates residue in the LARGER set, so only the common words matter.
_MATCHUP_STOP = _STOP | {
    "who", "upcoming", "event", "scheduled", "professional", "fight", "mma", "round",
    "matches", "stage", "group", "qualifiers", "challenger", "final", "semifinal",
    "quarterfinal", "playoff", "playoffs", "series", "leg",
    "basketball", "tennis", "soccer", "football", "baseball", "hockey", "esports",
    "dota", "valorant", "cod", "cs2", "lol", "mlb", "nba", "wnba", "nhl", "nfl", "ufc",
    "utc", "et", "am", "pm", "edt", "est",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
}


def _matchup_tokens(title: str) -> frozenset[str]:
    """Both teams/players of a matchup title (pre-' - '), minus template/sport filler.
    Used to tell two DIFFERENT games of the same team apart (same YES team, different
    opponent), which the YES-only subject cannot."""
    head = (title or "").rsplit(" - ", 1)[0]
    return frozenset(
        w for w in _WORD.findall(head.lower())
        if w not in _MATCHUP_STOP and not w.isdigit() and len(w) > 1
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
    league: str | None = None    # sport/league (valorant/soccer/...), if identifiable
    matchup: frozenset = frozenset()  # BOTH teams of a winner market (game disambiguation)

    @property
    def matchable(self) -> bool:
        return self.metric not in (_UNMATCHABLE, _UNKNOWN)


def _threshold_int(title: str) -> int | None:
    tag = _threshold_tag((title or "").lower())
    return int(tag.split(":", 1)[1]) if tag else None


def from_kalshi(ticker: str, title: str, yes_sub_title: str = "",
                close_time: float | None = None) -> ContractFingerprint:
    metric = kalshi_metric(ticker)
    # First-goal markets carry "GOAL" in the series (-> goals), but the title says "the
    # first goal", which is a DIFFERENT proposition from a goal-count prop. The title is
    # decisive here, so it overrides the series classification.
    if _title_metric(title) == "first_goal":
        metric = "first_goal"
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
        league=_league_of(kalshi_series(ticker)),
        matchup=_matchup_tokens(title) if metric == "winner" else frozenset(),
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
    if not subject and metric in _QUESTION_SUBJECT_METRICS:
        # Player props read "Will <Player> record N+ <stat> in A vs B? - Yes": the
        # entity is in the question, not the Yes/No suffix. Restricted to stat metrics:
        # a "A vs B end before round 3? - Yes" matchup has no clean YES side (the
        # question names BOTH parties), so it stays subject-less -> unmatchable, rather
        # than falsely aligning a duration/method market to a fight WINNER. Winners keep
        # their team/player suffix above (preserving YES polarity).
        subject = _subject_tokens(title.rsplit(" - ", 1)[0])
    return ContractFingerprint(
        venue="polymarket_us", metric=metric, scope=_scope_only(title),
        threshold=_threshold_int(title), subject=subject, date=date,
        league=_league_of(slug),
        matchup=_matchup_tokens(title) if metric == "winner" else frozenset(),
    )


def _common_prefix_len(x: str, y: str) -> int:
    n = 0
    for cx, cy in zip(x, y):
        if cx != cy:
            break
        n += 1
    return n


def _token_matches(x: str, large: frozenset) -> bool:
    """A token matches one in ``large`` if it's present, one is a >=3-char prefix of the
    other (team-code vs full-name, 'nor'/'norway'), or they share a >=5-char common
    prefix (spelling variants: 'fayzullaev'/'fayzullayev'). Different names with no shared
    prefix ('ronald' vs 'maximiliano') do NOT match."""
    for y in large:
        if x == y:
            return True
        if len(x) >= 3 and (y.startswith(x) or x.startswith(y)):
            return True
        if _common_prefix_len(x, y) >= 5:
            return True
    return False


def _matchup_conflict(a: frozenset, b: frozenset) -> bool:
    """Two winner matchups name DIFFERENT games. Both share the YES team; the discriminator
    is the OPPONENT. Reject only on a clear, MUTUAL mismatch: each side has a substantial
    (>=4 char) team token absent from the other. Short abbreviations ('NY', 'LA') are
    ignored, so an abbreviated venue title ('Cloud9 NY') still matches its spelled-out
    counterpart ('Cloud9 New York') — only full opponent names that genuinely differ
    (Las Vegas vs Golden State) trip it. Biased toward KEEP; the date gate is the other
    line of defense for different-day games."""
    a_only = [x for x in a if not _token_matches(x, b) and len(x) >= 4]
    b_only = [x for x in b if not _token_matches(x, a) and len(x) >= 4]
    return bool(a_only) and bool(b_only)


def _subjects_align(a: frozenset, b: frozenset) -> bool:
    """The YES outcomes refer to the same entity. The SMALLER (cleaner) subject must be
    fully covered by the larger — every one of its tokens matches. A single shared token
    is NOT enough, so two different multi-token entities that share one word (different
    players 'Ronald Araujo' vs 'Maximiliano Araujo', or teams sharing a dropped suffix)
    no longer falsely align."""
    if not a or not b:
        return False
    small, large = (a, b) if len(a) <= len(b) else (b, a)
    return all(_token_matches(x, large) for x in small)


def complement_reason(a: ContractFingerprint, b: ContractFingerprint,
                      max_gap_days: float = 3.0) -> str:
    """``"ok"`` if complementary, else the first failing check (for shadow diagnostics)."""
    if not a.matchable:
        return f"a-{a.metric}"
    if not b.matchable:
        return f"b-{b.metric}"
    if a.metric != b.metric:
        return f"metric {a.metric}!={b.metric}"
    if a.league and b.league and a.league != b.league:
        return f"league {a.league}!={b.league}"
    if a.scope != b.scope:
        return f"scope {sorted(a.scope)}!={sorted(b.scope)}"
    if a.threshold != b.threshold:
        return f"threshold {a.threshold}!={b.threshold}"
    if a.date is not None and b.date is not None and abs(a.date - b.date) > max_gap_days * 86400:
        return "date gap"
    # Same team, different game: a winner market names BOTH sides ("A vs B"), so when both
    # fingerprints carry a matchup with clearly different OPPONENTS, reject. The date
    # tolerance (max_gap_days) that legit cross-venue matches need is wide enough to admit
    # two different games of the same team within the window (and doubleheaders share a
    # date outright), so the date check alone can't separate them — this can.
    if a.metric == "winner" and a.matchup and b.matchup and _matchup_conflict(a.matchup, b.matchup):
        return f"matchup {sorted(a.matchup)}!={sorted(b.matchup)}"
    if not _subjects_align(a.subject, b.subject):
        return f"subject {sorted(a.subject)}!={sorted(b.subject)}"
    return "ok"


def are_complementary(a: ContractFingerprint, b: ContractFingerprint,
                      max_gap_days: float = 3.0) -> bool:
    """True iff buying YES on ``a`` and NO on ``b`` is the same proposition, opposite
    sides: same (matchable) metric, scope, threshold, event date, and the YES outcomes
    refer to the same entity. Fails closed on anything unparseable."""
    return complement_reason(a, b, max_gap_days) == "ok"
