"""Rules-text verification: compare the two markets' RESOLUTION CRITERIA directly.

The title-based LLM confirm checks the label; this checks the contract. Both venues
publish the actual settlement rules (Kalshi ``rules_primary``/``rules_secondary``,
Polymarket's ``description`` maps each outcome to its condition), so the LLM can be
asked the question that actually determines PnL: *is there ANY outcome in which these
two contracts settle differently?* A pair that passes is definitionally the same bet —
strong enough evidence that the streaming engine lets it fire a fat edge with no price
history (see the fat-edge escalation), while a found divergence is a hard, cacheable
false-match verdict.

Same conventions as llm_match: ``complete(prompt) -> str`` injected, JSON parsed
defensively, fails CLOSED (unparseable/malformed -> not verified).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Callable

log = logging.getLogger("bot.matching.rules_match")

CompleteFn = Callable[[str], str]

_PROMPT = """\
You are comparing the RESOLUTION RULES of two prediction markets from different
exchanges to decide if they are the SAME CONTRACT — such that buying YES on one and
NO on the other locks a guaranteed $1 payout in EVERY possible outcome.

Market A ({venue_a}) — YES side: "{title_a}"
Resolution rules A: {rules_a}

Market B ({venue_b}) — YES side: "{title_b}"
Resolution rules B: {rules_b}

Think adversarially: enumerate concrete scenarios where the two contracts could
settle DIFFERENTLY, then CLASSIFY the divergence:

- "different_event": the two markets are NOT about the same real-world event or the
  same winning party — different teams (beware an org vs its ACADEMY/junior squad —
  those are different teams), different scheduled match/date/time, different metric,
  threshold or period. These can NEVER hedge each other.
- "tail_scenarios": SAME event and SAME winning party, but the rules differ in edge
  cases — cancellations, postponements, ties/draw handling, void/refund wording,
  overtime inclusion, settlement sources. The pair hedges in the normal outcome but
  may diverge in those tails.
- "none": you cannot construct any scenario where A's YES and B's YES settle
  differently.

Answer "identical": true ONLY for "none". Missing/ambiguous rules -> "tail_scenarios"
at low confidence, never "none".

Respond with ONLY a JSON object:
{{"divergent_scenario": "<the scenario, or 'none found'>",
  "divergence": "different_event"|"tail_scenarios"|"none",
  "identical": <true|false>, "confidence": <0.0-1.0>, "rationale": "<one sentence>"}}
"""


@dataclass
class RulesVerdict:
    identical: bool
    confidence: float
    rationale: str
    # True = "different_event": NOT the same event/party — never a hedge -> demote from
    # the watchlist. False = tail-scenario divergence (same event, differing void/tie
    # wording) or identical — stays tradeable (just no fat-edge privilege when divergent).
    material: bool = False


def confirm_rules(complete: CompleteFn, *, venue_a: str, title_a: str, rules_a: str,
                  venue_b: str, title_b: str, rules_b: str) -> RulesVerdict:
    """Ask the LLM whether two rules texts settle identically — and, when they don't,
    whether the divergence is MATERIAL (different event/party) or a tail scenario.
    Fails closed (not identical, not material)."""
    prompt = _PROMPT.format(
        venue_a=venue_a, title_a=(title_a or "")[:300], rules_a=(rules_a or "")[:1500],
        venue_b=venue_b, title_b=(title_b or "")[:300], rules_b=(rules_b or "")[:1500])
    try:
        raw = complete(prompt)
    except Exception as exc:
        log.warning("rules confirm failed: %s", exc)
        return RulesVerdict(False, 0.0, f"llm error: {exc}")
    m = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not m:
        return RulesVerdict(False, 0.0, "unparseable response")
    try:
        obj = json.loads(m.group(0))
        return RulesVerdict(
            identical=bool(obj.get("identical") is True),
            confidence=float(obj.get("confidence") or 0.0),
            rationale=str(obj.get("rationale") or "")[:400],
            material=(str(obj.get("divergence") or "") == "different_event"))
    except (ValueError, TypeError) as exc:
        return RulesVerdict(False, 0.0, f"bad json: {exc}")
