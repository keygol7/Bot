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
- metric: what is measured — "winner", "goals", "assists", "points", "price_close",
  "vote_share", "temperature", "count", "other"
- comparator: ">=", "<=", "==" or null (winner markets have none)
- value: the numeric threshold/line (2.5 for a -2.5 handicap; 100000 for BTC>100k;
  1 for "1 or more goals") or null
- period: "full" (default), "1h", "2h", "et_included", "regulation", "set", "round",
  or null if unclear
- date: the event/measurement date as YYYY-MM-DD, null if unstated

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
        c = _extract_once(complete, prompt, venue, market_id)
        if c is not None:
            return c
    log.warning("canon parse failed for %s after %d attempts", market_id, attempts)
    return None


def _extract_once(complete: CompleteFn, prompt: str, venue: str,
                  market_id: str) -> Canon | None:
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
        return Canon(
            venue=venue, market_id=market_id,
            event_type=str(obj.get("event_type") or "other").lower(),
            entities=ents, subject=subject,
            metric=str(obj.get("metric") or "other").lower(),
            comparator=(str(obj["comparator"]) if obj.get("comparator") else None),
            value=(float(value) if value is not None else None),
            period=str(obj.get("period") or "full").lower(),
            date=(str(obj["date"])[:10] if obj.get("date") else None),
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
    if not _subjects_align(a.subject, b.subject):
        return False                                  # YES must pay the same party
    return _events_align(a, b)
