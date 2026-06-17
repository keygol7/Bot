"""Local-LLM confirmation that two shortlisted markets are the *same event with the
same resolution criteria*.

This is the safety gate that distinguishes a real risk-free arbitrage from a
"settlement mismatch" trap (two markets that look alike but resolve differently).
The LLM is used offline — its verdicts are cached so each pair is judged once.

The model is reached through a ``complete`` callable: ``complete(prompt) -> str``.
In production this wraps a local OpenAI-compatible endpoint (vLLM/Ollama) on
``localhost``; in tests a fake callable is injected, so this module needs no network
and no model. The verdict is parsed from JSON the model returns; parsing is
defensive and fails *closed* (``same_event=False``) so an unparseable answer can
never green-light a trade.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable

from bot.models import MarketQuote

CompleteFn = Callable[[str], str]

_PROMPT_TEMPLATE = """\
You decide whether two prediction markets are the SAME tradeable contract, so that
buying YES on one and NO on the other locks a guaranteed payout. Mechanical
differences (half vs match, handicaps, set winner, goals vs goals+assists, method of
victory) are ALREADY filtered out before you — your ONLY job is to verify the two
markets resolve on the SAME contest and the SAME winning party.

Work in two explicit steps, then decide:
  STEP 1 — For EACH market, state the exact PARTY the YES side pays out for (the team,
  player, or fighter who must win/achieve the outcome). Beware: a title often names
  BOTH sides of a matchup ("Allan Nascimento vs Mitch Raposo"); the YES party is the
  ONE the market resolves YES for, not merely a name that appears.
  STEP 2 — Confirm it is the SAME underlying contest (same competitors, same date).

Answer "same_event": true ONLY if BOTH YES parties are the SAME individual/team AND it
is the same contest. Different party, different match, or any doubt -> false. Allow for
spelling/transliteration variants of the SAME person (e.g. "Ghoddos"/"Ghoddoos",
"van Dijk"/"Van Dijk") — judge the real-world identity, not the exact string.

EXAMPLES (these are FALSE):
  - A: "Allan Nascimento win the fight" / B: "Mitch Raposo win ... in Nascimento vs
    Raposo" -> false (YES pays for different fighters).
  - A: "final score Draw 0-0" [IRQ vs NOR, Jun 16] / B: "SCO vs MAR finish Draw 0-0"
    [Jun 19] -> false (different match and date).
TRUE example:
  - A: "Will Ciryl Gane win by KO/TKO/DQ?" / B: "Ciryl Gane win by KO,TKO,DQ in Gane
    vs Pereira" -> true (same fighter, same contest).

Market A ({venue_a}): "{title_a}"  [resolves: {date_a}]
Market B ({venue_b}): "{title_b}"  [resolves: {date_b}]

Respond with ONLY a JSON object:
{{"yes_party_a": "<who YES pays in A>", "yes_party_b": "<who YES pays in B>",
  "same_event": <true|false>, "confidence": <0.0-1.0>, "rationale": "<one sentence>"}}
"""


def _fmt_date(ts: float | None) -> str:
    if ts is None:
        return "unknown"
    from datetime import datetime, timezone

    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


@dataclass
class MatchVerdict:
    same_event: bool          # same event AND same resolution criteria
    confidence: float         # 0..1
    rationale: str = ""

    def tradeable(self, min_confidence: float = 0.85) -> bool:
        """A pair is safe to arbitrage only if confirmed AND confident."""
        return self.same_event and self.confidence >= min_confidence


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model response. Raises on failure."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in model response")
    return json.loads(match.group(0))


def build_prompt(a: MarketQuote, b: MarketQuote) -> str:
    return _PROMPT_TEMPLATE.format(
        venue_a=a.venue, title_a=a.title, date_a=_fmt_date(a.close_time),
        venue_b=b.venue, title_b=b.title, date_b=_fmt_date(b.close_time),
    )


def confirm_match(a: MarketQuote, b: MarketQuote, complete: CompleteFn) -> MatchVerdict:
    """Ask the local model whether A and B are the same event/resolution.

    Fails closed: any error parsing the response yields ``same_event=False``.
    """
    try:
        raw = complete(build_prompt(a, b))
        data = _extract_json(raw)
        same = bool(data.get("same_event", False))
        confidence = float(data.get("confidence", 0.0))
        confidence = min(max(confidence, 0.0), 1.0)
        rationale = str(data.get("rationale", ""))
        return MatchVerdict(same_event=same, confidence=confidence, rationale=rationale)
    except Exception as exc:  # fail closed — never green-light on a bad parse
        return MatchVerdict(
            same_event=False, confidence=0.0, rationale=f"parse error: {exc}"
        )
