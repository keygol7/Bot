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
You are a careful prediction-market analyst. Two markets from different venues may
or may not refer to the SAME real-world event with the SAME resolution criteria.

Only answer "same_event": true if ALL of these hold:
  1. Same underlying outcome AND the same specific subject (same exact team / player /
     candidate — "Los Angeles Angels" is NOT "Los Angeles Dodgers").
  2. Same EVENT TYPE and scope. A single game is NOT a season-long championship; a
     regular-season matchup is NOT a "win the World Series" futures market; a primary
     is NOT a general election.
  3. Same resolution timing/date and sources, so that exactly one of (YES on A, NO on
     B) is guaranteed to pay out.

The two resolution dates below must describe the same event. If they differ
materially, or you are unsure, answer false. A wrong "true" causes real financial loss.

Market A ({venue_a}): "{title_a}"  [resolves: {date_a}]
Market B ({venue_b}): "{title_b}"  [resolves: {date_b}]

Respond with ONLY a JSON object:
{{"same_event": <true|false>, "confidence": <0.0-1.0>, "rationale": "<one sentence>"}}
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
