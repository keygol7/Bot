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
settle DIFFERENTLY. Consider: different events/dates, different winning parties,
draws/ties/cancellations/postponements handled differently, different thresholds or
periods (half vs full, handicap lines, overtime inclusion), different data sources or
settlement authorities, void/refund conditions.

Answer "identical": true ONLY if you cannot construct ANY scenario where A's YES and
B's YES settle differently. Any divergent scenario, or missing/ambiguous rules -> false.

Respond with ONLY a JSON object:
{{"divergent_scenario": "<the scenario, or 'none found'>",
  "identical": <true|false>, "confidence": <0.0-1.0>, "rationale": "<one sentence>"}}
"""


@dataclass
class RulesVerdict:
    identical: bool
    confidence: float
    rationale: str


def confirm_rules(complete: CompleteFn, *, venue_a: str, title_a: str, rules_a: str,
                  venue_b: str, title_b: str, rules_b: str) -> RulesVerdict:
    """Ask the LLM whether two rules texts settle identically. Fails closed."""
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
            rationale=str(obj.get("rationale") or "")[:400])
    except (ValueError, TypeError) as exc:
        return RulesVerdict(False, 0.0, f"bad json: {exc}")
