"""Canonicalize-then-join matching: extract each market's canonical contract ONCE,
then match by deterministic join — instead of pairwise LLM judgments.

The pairwise design (shortlist pairs -> LLM judges each pair) is O(N^2) in LLM work,
so confirmations get rationed and recall starves — and each judgment sees only titles.
Here the LLM is asked ONE question per market, against its RESOLUTION RULES (the
contract): "extract the canonical event/subject/metric/threshold/period/date". The
answer is cached forever (a market's contract never changes), and matching becomes a
database join on canonical fields — O(N) extractions, domain-agnostic by construction:
an election, a CPI print, and a Valorant match all canonicalize into the same schema.

Precision comes from the join being EXACT on the fields that decide settlement
(metric, threshold, period, date) plus a token-overlap subject alignment for
cross-venue name variants. Downstream, every joined pair still passes the full truth
stack (id-scope structure, empirical sums, rules verification, settlement ground
truth, probe sizing) — the join nominates, verification disposes.

Fail-closed everywhere: an unparseable extraction -> no canon row -> not matchable
via this path (the fingerprint/LLM paths still apply).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Callable

log = logging.getLogger("bot.matching.canon")

CompleteFn = Callable[[str], str]

_PROMPT = """\
Extract the canonical contract of this prediction market from its resolution rules.
Any domain: sports, elections, economics, crypto, weather, awards, entertainment.

Market ({venue}): "{title}"
Resolution rules: {rules}

Fields (use null when not applicable):
- event_type: one of "match" (two parties compete), "election", "econ_release",
  "price_threshold", "award", "weather", "entertainment", "other"
- entities: ALL parties/instruments involved (both teams; all candidates named; the
  index/asset), lowercase canonical real-world names
- subject: the ONE entity the YES side pays for (lowercase). null if YES is not about
  a single entity (e.g. "draw", a numeric range).
- metric: EXACTLY one of "winner", "goals", "assists", "points", "price_close",
  "vote_share", "inflation_rate", "gdp_growth", "temperature", "count", "other".
  Never invent a new word: CPI/inflation markets -> "inflation_rate"; GDP -> "gdp_growth".
- comparator: ">=", "<=", ">", "<", "==" or null (winner markets have none).
  "above X" / "more than X" -> ">".  "X or above" / "at least X" -> ">=".
  These are DIFFERENT contracts — copy the rules' wording exactly.
- value: the numeric threshold/line (2.5 for a -2.5 handicap; 100000 for BTC>100k;
  1 for "1 or more goals") or null
- period: "full" (default), "1h", "2h", "et_included", "regulation", "set", "round",
  or null if unclear
- date: YYYY-MM-DD. For matches/awards: the event date. For econ_release: the LAST DAY
  of the MEASUREMENT period (June CPI -> 2026-06-30), NEVER the announcement/release
  date. Use null if unstated — never guess or use placeholders.

Example (econ): "Will CPI inflation be above 3.7% for June 2026?" ->
{{"event_type": "econ_release", "entities": ["us cpi"], "subject": null,
  "metric": "inflation_rate", "comparator": ">", "value": 3.7, "period": "full",
  "date": "2026-06-30", "confidence": 0.95}}
Example (election): "Will the Democratic candidate win the 2026 Ohio governor race?" ->
{{"event_type": "election", "entities": ["democratic party", "ohio governor race"],
  "subject": "democratic party", "metric": "winner", "comparator": null, "value": null,
  "period": "full", "date": "2026-11-03", "confidence": 0.9}}

Respond with ONLY a JSON object:
{{"event_type": "...", "entities": [...], "subject": "...", "metric": "...",
  "comparator": null, "value": null, "period": "full", "date": "YYYY-MM-DD",
  "confidence": <0.0-1.0>}}
"""

_WORD = re.compile(r"[a-z0-9]+")
_GENERIC = frozenset({"gaming", "esports", "team", "club", "fc", "sc", "cf", "the"})
# An org and its academy/junior/reserve squad are DIFFERENT teams (see fingerprint._SUB_ORG;
# live incident: BESTIA vs BESTIA Academy matched as one event). Residue containing one of
# these breaks name-variant alignment.
_SUB_ORG = frozenset({"academy", "jr", "junior", "youth", "reserve", "reserves",
                      "u17", "u18", "u19", "u20", "u21", "u23", "ii", "prospects"})


@dataclass
class Canon:
    venue: str
    market_id: str
    event_type: str
    entities: tuple[str, ...]
    subject: str | None
    metric: str
    comparator: str | None
    value: float | None
    period: str
    date: str | None
    confidence: float


def _tokens(name: str | None) -> frozenset[str]:
    return frozenset(w for w in _WORD.findall((name or "").lower())
                     if w not in _GENERIC and len(w) > 1)


# Closed metric vocabulary + synonym normalization. The model invents variants
# (cpi_increase / inflation_rate / increase / value / cpi for the SAME contract) and the
# join requires EXACT equality — free-form metrics put identical contracts in different
# buckets forever. Deterministic normalization beats prompt hope.
_METRICS = frozenset({"winner", "goals", "assists", "points", "price_close",
                      "vote_share", "inflation_rate", "gdp_growth", "temperature",
                      "count", "other"})
_METRIC_SYNONYMS = {
    "cpi": "inflation_rate", "cpi_increase": "inflation_rate",
    "cpi_change": "inflation_rate", "inflation": "inflation_rate",
    "gdp": "gdp_growth", "gdp_change": "gdp_growth", "growth": "gdp_growth",
    "win": "winner", "victory": "winner", "champion": "winner",
    "price": "price_close", "goal": "goals", "assist": "assists", "point": "points",
}
_TITLE_METRIC_HINTS = (            # generic metric + title keyword -> real metric
    (("cpi", "inflation"), "inflation_rate"),
    (("gdp",), "gdp_growth"),
)
_GENERIC_METRICS = frozenset({"other", "value", "increase", "change", "rate", "number"})
_COMPARATORS = frozenset({">=", "<=", ">", "<", "=="})
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _normalize_metric(metric: str, title: str) -> str:
    m = (metric or "other").lower().strip().replace(" ", "_")
    if m in _METRICS:
        return m
    if m in _METRIC_SYNONYMS:
        return _METRIC_SYNONYMS[m]
    if m in _GENERIC_METRICS or m not in _METRICS:
        t = (title or "").lower()
        for keys, real in _TITLE_METRIC_HINTS:
            if any(k in t for k in keys):
                return real
    return "other"


def _normalize_date(date: str | None, event_type: str) -> str | None:
    """Strict YYYY-MM-DD or None (placeholders like 2026-XX-XX are junk). Economic
    releases join at MONTH granularity: the model dates 'June CPI' as 06-01, 06-30 or
    the July release day — truncate to YYYY-MM-01 so one contract lands in one bucket."""
    if not date or not _DATE_RE.match(date):
        return None
    if event_type == "econ_release":
        return date[:7] + "-01"
    return date


def normalize_stored(metric: str | None, date: str | None,
                     event_type: str | None) -> tuple[str, str | None]:
    """Normalize a LEGACY canon row read back from the db (rows extracted before the
    normalization layer existed carry free-form metrics and raw dates). No title
    context here, so only the synonym map + date rules apply."""
    et = (event_type or "other").lower()
    m = (metric or "other").lower().strip().replace(" ", "_")
    if m not in _METRICS:
        m = _METRIC_SYNONYMS.get(m, "other")
    return m, _normalize_date(date if date and _DATE_RE.match(date) else None, et)


def _lenient_json(s: str) -> dict:
    """Parse LLM JSON, repairing the two failures we see in the wild: trailing commas
    (``... "x": 1, }``) and ``// line comments``. Local models emit these on ~5% of
    extractions (seen on the Love Island winner markets), which otherwise fail-close and
    the market never canonicalizes."""
    try:
        return json.loads(s)
    except ValueError:
        repaired = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)  # strip /* block */ comments
        repaired = re.sub(r"//[^\n]*", "", repaired)            # strip // line comments
        repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)      # strip trailing commas
        return json.loads(repaired)


def extract_canon(complete: CompleteFn, *, venue: str, market_id: str,
                  title: str, rules: str, attempts: int = 3) -> Canon | None:
    """One LLM extraction -> Canon, or None (fail closed) on anything unparseable.

    The local model is non-deterministic and emits malformed JSON on a minority of
    calls (unquoted keys, stray text) — the SAME prompt parses fine on a re-roll. Retry
    a few times before giving up so a good market isn't lost to a one-off bad generation.
    """
    prompt = _PROMPT.format(venue=venue, title=(title or "")[:300],
                            rules=(rules or "")[:1500])
    last_exc = None
    for attempt in range(attempts):
        c = _extract_once(complete, prompt, venue, market_id, title)
        if c is not None:
            return c
    log.warning("canon parse failed for %s after %d attempts", market_id, attempts)
    return None


def _extract_once(complete: CompleteFn, prompt: str, venue: str,
                  market_id: str, title: str = "") -> Canon | None:
    try:
        raw = complete(prompt)
    except Exception as exc:
        log.warning("canon extraction failed for %s: %s", market_id, exc)
        return None
    m = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not m:
        return None
    try:
        obj = _lenient_json(m.group(0))
        ents = tuple(sorted(str(e).lower().strip() for e in (obj.get("entities") or [])
                            if str(e).strip()))
        subject = obj.get("subject")
        subject = str(subject).lower().strip() if subject else None
        value = obj.get("value")
        event_type = str(obj.get("event_type") or "other").lower()
        comparator = str(obj["comparator"]).strip() if obj.get("comparator") else None
        if comparator is not None and comparator not in _COMPARATORS:
            comparator = None                       # unknown symbol -> not a threshold
        return Canon(
            venue=venue, market_id=market_id,
            event_type=event_type,
            entities=ents, subject=subject,
            metric=_normalize_metric(str(obj.get("metric") or "other"), title),
            comparator=comparator,
            value=(float(value) if value is not None else None),
            period=str(obj.get("period") or "full").lower(),
            date=_normalize_date(
                str(obj["date"])[:10] if obj.get("date") else None, event_type),
            confidence=float(obj.get("confidence") or 0.0),
        )
    except (ValueError, TypeError, KeyError) as exc:
        log.debug("canon parse attempt failed for %s: %s", market_id, exc)
        return None


def _subjects_align(a: str | None, b: str | None) -> bool:
    """Token-overlap subject alignment tolerant of cross-venue name variants
    ("Gen.G Global Academy" vs "geng academy"): one side's tokens must be a subset of
    the other's (residue tolerated only in the LONGER name — EXCEPT sub-org markers:
    "BESTIA" vs "BESTIA Academy" are different teams). Both-None never aligns."""
    if not a or not b:
        return False
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return False
    if not (ta <= tb or tb <= ta):
        return False
    small, large = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    return not (large - small) & _SUB_ORG


def _events_align(a: Canon, b: Canon) -> bool:
    """Same underlying event: entity sets share the event (subset either way — venues
    name entities with different completeness) and dates agree when both are stated."""
    ea, eb = set(a.entities), set(b.entities)
    if not ea or not eb:
        return False
    # Token-level containment per entity: every entity of the SMALLER set must align
    # with some entity of the larger (name-variant tolerant).
    small, big = (ea, eb) if len(ea) <= len(eb) else (eb, ea)
    for ent in small:
        if not any(_subjects_align(ent, other) for other in big):
            return False
    if a.date and b.date and a.date != b.date:
        return False
    return True


def complementary(a: Canon, b: Canon, *, min_confidence: float = 0.7) -> bool:
    """True when YES(a) and YES(b) are the SAME contract — exact on every field that
    decides settlement, name-variant tolerant on the subject, same underlying event."""
    if a.confidence < min_confidence or b.confidence < min_confidence:
        return False
    if a.event_type != b.event_type or a.metric != b.metric:
        return False
    if (a.period or "full") != (b.period or "full"):
        return False
    if a.comparator != b.comparator or a.value != b.value:
        return False                                  # a -2.5 line never joins a moneyline
    # Subject: YES must pay the same party. BOTH-None is legitimate — the prompt itself
    # instructs null for non-entity contracts (numeric ranges: "CPI above 3.7%", draws),
    # whose identity is fully carried by the remaining exact fields (metric, comparator,
    # value, period, date) + entity alignment below. Requiring alignment unconditionally
    # made scalar contracts UNMATCHABLE by construction (None never aligns). One-sided
    # None still rejects: an entity contract can't join a non-entity one.
    if a.subject is None and b.subject is None:
        pass
    elif not _subjects_align(a.subject, b.subject):
        return False
    return _events_align(a, b)
