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
buying YES on one and NO on the other locks a guaranteed payout. Verify the complete
resolution proposition: subject, event/date/time, threshold or range, measurement
source, period, and YES outcome. Never assume mechanical differences were filtered.

Work in two explicit steps, then decide:
  STEP 1 — For EACH market, state the exact condition under which YES pays. For a
  contest, identify the one team/player/fighter (the YES party). For weather, prices,
  economics, or other props, include the exact metric, threshold/range, date, time,
  and period.
  STEP 2 — Confirm every material term is identical. A one-day, time-of-day, strike,
  range, comparator, measurement-source, or settlement-period difference is false.

Answer "same_event": true ONLY if BOTH YES conditions are logically identical.
Different party, date, time, threshold/range, comparator, source, period, or any doubt
-> false. Allow for
spelling/transliteration variants of the SAME person (e.g. "Ghoddos"/"Ghoddoos",
"van Dijk"/"Van Dijk") — judge the real-world identity, not the exact string.
CAUTION: an esports org and its ACADEMY/junior/youth squad are DIFFERENT teams playing
DIFFERENT matches ("BESTIA" vs "BESTIA Academy"). When one title carries an
Academy/Jr/youth marker and the other does not, answer true ONLY if you are confident
both refer to the SAME squad in the SAME contest (e.g. identical opponent and the
shorter name is clearly an abbreviation of the academy squad); any doubt -> false.

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
{{"yes_party_a": "<exact YES condition in A>", "yes_party_b": "<exact YES condition in B>",
  "same_event": <true|false>, "confidence": <0.0-1.0>, "rationale": "<one sentence>"}}
"""


def _fmt_date(ts: float | None) -> str:
    if ts is None:
        return "unknown"
    from datetime import datetime, timezone

    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7,
    "july": 7, "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12,
    "december": 12,
}
_DATE_RE = re.compile(
    r"\b(" + "|".join(_MONTHS) + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?"
    r"(?:,?\s+(20\d{2}))?\b",
    re.IGNORECASE,
)
_TIME_RE = re.compile(
    r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\s*(e[ds]t|et|utc)?\b",
    re.IGNORECASE,
)
_CRYPTO = ("bitcoin", "btc", "ethereum", "ether", "eth")


def _calendar_date(title: str) -> tuple[int | None, int, int] | None:
    match = _DATE_RE.search(title)
    if not match:
        return None
    return (int(match.group(3)) if match.group(3) else None,
            _MONTHS[match.group(1).lower().rstrip(".")], int(match.group(2)))


def _clock_time(title: str) -> tuple[int, int, str | None] | None:
    match = _TIME_RE.search(title)
    if not match:
        return None
    hour = int(match.group(1)) % 12 + (12 if match.group(3).lower() == "pm" else 0)
    zone = match.group(4).lower() if match.group(4) else None
    if zone in {"est", "edt", "et"}:
        zone = "et"
    return hour, int(match.group(2) or 0), zone


def _crypto_strike(title: str) -> float | None:
    lower = title.lower()
    if not any(re.search(rf"\b{asset}\b", lower) for asset in _CRYPTO):
        return None
    match = re.search(r"\$\s*([\d,]+(?:\.\d+)?)", title)
    if not match:
        match = re.search(r"\b(?:above|over|at least)\s+\$?([\d,]+(?:\.\d+)?)", title,
                          re.IGNORECASE)
    return float(match.group(1).replace(",", "")) if match else None


def _netflix_scope(text: str) -> str | None:
    """Return the explicitly named Netflix chart scope, when present."""
    lower = text.lower()
    if "netflix" not in lower:
        return None
    if "global" in lower:
        return "global"
    if re.search(r"\b(?:us|u\.s\.)\b", lower):
        return "us"
    return None


def _scaled_number(raw: str, suffix: str | None) -> float:
    value = float(raw.replace(",", ""))
    return value * {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}.get(
        (suffix or "").lower(), 1)


def _numeric_bound(text: str) -> tuple[str, float] | None:
    """Extract an explicit one-sided comparator and threshold from a title.

    This intentionally ignores bare amounts and ranges.  It exists to distinguish
    strict contracts (``above $30m``) from inclusive ones (``at least $30m``).
    """
    amount = r"\$?\s*([\d,]+(?:\.\d+)?)\s*([kmb])?"
    patterns = (
        ("gte", rf"\b(?:at\s+least|greater\s+than\s+or\s+equal\s+to)\s+{amount}"),
        ("lte", rf"\b(?:at\s+most|less\s+than\s+or\s+equal\s+to)\s+{amount}"),
        ("gt", rf"\b(?:above|over|greater\s+than)\s+{amount}"),
        ("lt", rf"\b(?:below|under|less\s+than)\s+{amount}"),
    )
    for comparator, pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return comparator, _scaled_number(match.group(1), match.group(2))
    return None


def _weather_interval(title: str) -> tuple[float, float] | None:
    lower = title.lower()
    if "temp" not in lower and "temperature" not in lower:
        return None
    encoded_range = re.search(r"(?:^|[-_])gte(\d{1,3})lt(\d{1,3})f?\b", lower)
    if encoded_range:
        return float(encoded_range.group(1)), float(encoded_range.group(2)) - 1.0
    encoded_floor = re.search(r"(?:^|[-_])gte(\d{1,3})f?\b", lower)
    if encoded_floor:
        return float(encoded_floor.group(1)), float("inf")
    encoded_ceiling = re.search(r"(?:^|[-_])lt(\d{1,3})f?\b", lower)
    if encoded_ceiling:
        return float("-inf"), float(encoded_ceiling.group(1)) - 1.0
    range_match = re.search(
        r"\b(\d{1,3})\s*°?\s*(?:-|to)\s*(\d{1,3})\s*°?\s*f?\b", lower)
    if range_match:
        return float(range_match.group(1)), float(range_match.group(2))
    strict_above = re.search(r">\s*(\d{1,3})\s*°", lower)
    if strict_above:
        return float(strict_above.group(1)) + 1.0, float("inf")
    inclusive_above = re.search(
        r"\b(\d{1,3})\s*°?\s*f?\s*(?:or\s+)?(?:above|higher)\b", lower)
    if inclusive_above:
        return float(inclusive_above.group(1)), float("inf")
    inclusive_below = re.search(
        r"\b(\d{1,3})\s*°?\s*f?\s*(?:or\s+)?(?:below|lower)\b", lower)
    if inclusive_below:
        return float("-inf"), float(inclusive_below.group(1))
    return None


def obvious_contract_mismatch(a: MarketQuote, b: MarketQuote) -> str | None:
    """Return a deterministic reason for an obvious title-level settlement mismatch.

    This deliberately handles only facts that can be extracted without interpretation.
    Unknown or one-sided details fall through to the semantic/rules verifier.
    """
    date_a, date_b = _calendar_date(a.title), _calendar_date(b.title)
    if date_a and date_b:
        year_mismatch = date_a[0] is not None and date_b[0] is not None \
            and date_a[0] != date_b[0]
        if year_mismatch or date_a[1:] != date_b[1:]:
            return f"explicit date mismatch: {date_a} != {date_b}"

    weather_a = _weather_interval(f"{a.title} {a.market_id}")
    weather_b = _weather_interval(f"{b.title} {b.market_id}")
    if weather_a is not None and weather_b is not None and weather_a != weather_b:
        return f"weather range mismatch: {weather_a} != {weather_b}"

    strike_a, strike_b = _crypto_strike(a.title), _crypto_strike(b.title)
    if strike_a is not None and strike_b is not None and abs(strike_a - strike_b) > 0.02:
        return f"crypto strike mismatch: {strike_a:g} != {strike_b:g}"

    # A time mismatch is material for price snapshots. Restrict this guard to crypto so
    # sports titles containing broadcast times do not get mistaken for settlement times.
    if strike_a is not None and strike_b is not None:
        time_a, time_b = _clock_time(a.title), _clock_time(b.title)
        if time_a and time_b and time_a != time_b:
            return f"crypto observation time mismatch: {time_a} != {time_b}"

        # Crypto snapshot titles frequently omit the observation time.  Both venues
        # close these contracts at the observation instant, so their authoritative
        # close timestamps expose the otherwise-hidden 03:00/noon/17:00 mismatch.
        if (a.close_time is not None and b.close_time is not None
                and abs(a.close_time - b.close_time) > 60):
            return "crypto observation time mismatch: market close timestamps differ"

    scope_a = _netflix_scope(f"{a.title} {a.market_id}")
    scope_b = _netflix_scope(f"{b.title} {b.market_id}")
    if scope_a and scope_b and scope_a != scope_b:
        return f"Netflix chart scope mismatch: {scope_a} != {scope_b}"

    bound_a, bound_b = _numeric_bound(a.title), _numeric_bound(b.title)
    if (bound_a is not None and bound_b is not None
            and abs(bound_a[1] - bound_b[1]) <= 0.02
            and bound_a[0] != bound_b[0]):
        return ("threshold comparator mismatch: "
                f"{bound_a[0]} {bound_a[1]:g} != {bound_b[0]} {bound_b[1]:g}")
    return None


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
    mismatch = obvious_contract_mismatch(a, b)
    if mismatch:
        return MatchVerdict(same_event=False, confidence=1.0, rationale=mismatch)
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
