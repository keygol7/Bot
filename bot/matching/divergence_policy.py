"""Divergence relevance: which rule divergences matter, and how much.

A rule divergence between two matched markets is relevant in proportion to
``P(the divergent branch actually happens) x loss-if-it-hits``, compared against the
1-2c/ct edge an arb earns. Industry practice (professional surebet services) treats
this as a per-sport TAXONOMY of enumerable rule classes with known severities —
tennis alone has four canonical retirement rules, and "mixed rules" pairs have a
quantified expected cost — not a free-form judgment call. The local LLM classifies
each divergence INTO this taxonomy (schema classification, which it does reliably);
this module owns the severity numbers and the resulting policy.

Policies:
- ``ignore``      expected cost is noise (< ~0.2c/ct): tradeable, unrestricted
- ``edge_floor``  tradeable only when the edge also pays for the divergence risk:
                  requires edge >= min_edge + EXTRA (extra = k x expected cost)
- ``one_way``     asymmetric divergence (one venue's settlement window/rule is a
                  superset): trade only with YES on the wider side (windfall shape)
- ``block``       never a hedge (different event, or unknown-high severity)

Priors are seeded from researched base rates and updated empirically: every settled
pair carrying a divergence class is a completed experiment (settlement_checks), and
the Beta posterior can only RAISE p above the prior until >= 20 observations exist.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# The enumerated divergence taxonomy. Keep in sync with the rules-verify prompt.
CLASSES = (
    "timing_scope", "retirement_withdrawal", "cancellation_postponement",
    "void_vs_fairprice", "tie_handling", "settlement_source",
    "settlement_time_only", "different_event", "none",
)

# Rationale keyword fallback for verdicts that predate the constrained prompt.
_CLASS_PATTERNS = (
    ("timing_scope", re.compile(
        r"extra time|overtime|penalty shootout|shootout|90 minutes|regulation", re.I)),
    ("retirement_withdrawal", re.compile(
        r"retire|withdraw|walkover|walk-over|disqualif", re.I)),
    ("cancellation_postponement", re.compile(
        r"cancel|postpon|abandon|rain|suspend|not completed|incomplete game", re.I)),
    ("void_vs_fairprice", re.compile(r"void|refund|fair price|last[- ]trade", re.I)),
    ("tie_handling", re.compile(r"\btie\b|\bties\b|draw handling|drawn", re.I)),
    ("settlement_source", re.compile(
        r"settlement source|official (scorer|source)|data provider|opta", re.I)),
    ("settlement_time_only", re.compile(
        r"resolution time|settlement time|resolve[sd]? .*time|different times", re.I)),
)


def classify_rationale(rationale: str) -> str:
    """Keyword classification for legacy free-text rationales (audit-proven)."""
    for cls, pat in _CLASS_PATTERNS:
        if pat.search(rationale or ""):
            return cls
    return "none"


# Sport family from the kalshi series tag / ticker (idparse series metadata tags).
_FAMILY_PATTERNS = (
    ("tennis_itf", re.compile(r"KXITF|CHALLENGER", re.I)),
    ("tennis_tour", re.compile(r"KXATP|KXWTA", re.I)),
    ("soccer", re.compile(r"KXWC|KXUCL|KXUEL|KXEPL|KXMLS|SOCCER", re.I)),
    ("cricket", re.compile(r"T20|CRICKET|KXIPL", re.I)),
    ("esports", re.compile(r"CS2|DOTA|LOL|VALORANT|KXR6|KXEWC", re.I)),
    ("baseball", re.compile(r"KXMLB|KXNPB|KXKBO|KXWBC", re.I)),
    ("basketball", re.compile(r"KXNBA|KXWNBA", re.I)),
)


def sport_family(kalshi_market_id: str) -> str:
    for fam, pat in _FAMILY_PATTERNS:
        if pat.search(kalshi_market_id or ""):
            return fam
    return "other"


@dataclass
class DivergencePolicy:
    policy: str            # ignore | edge_floor | one_way | block
    expected_cost_ct: float  # per-contract expected cost of the divergence
    extra_edge_ct: float   # additional edge required above min_edge (edge_floor only)


# (class, family) -> (p_branch, loss_ct). Researched priors:
# - soccer knockout ET/shootout branch ~20% (0-0 after 90' / decided beyond regulation)
# - tennis retirement: tour ~2%, ITF/challenger ~5% (low-tier retirements are common)
# - cricket rain/no-result ~8%; esports cancellations/remakes ~1%
# - loss_ct: asymmetric void-vs-settle shapes risk ~50c/ct; both-legs-lose timing
#   shapes ~100c/ct on the wrong direction (hence one_way, not edge_floor)
_SEVERITY: dict = {
    ("timing_scope", "soccer"): (0.20, 1.00),
    ("timing_scope", "baseball"): (0.08, 0.50),     # extras handling
    ("timing_scope", "esports"): (0.03, 0.50),      # map/series scope oddities
    ("retirement_withdrawal", "tennis_itf"): (0.05, 0.50),
    ("retirement_withdrawal", "tennis_tour"): (0.02, 0.50),
    ("cancellation_postponement", "cricket"): (0.08, 0.50),
    ("cancellation_postponement", "soccer"): (0.01, 0.50),
    ("cancellation_postponement", "tennis_itf"): (0.03, 0.50),
    ("cancellation_postponement", "esports"): (0.01, 0.50),
    ("cancellation_postponement", "baseball"): (0.02, 0.50),
    ("tie_handling", "baseball"): (0.02, 0.50),     # NPB real ties exist
    ("tie_handling", "soccer"): (0.05, 0.50),
    ("void_vs_fairprice", None): (0.02, 0.30),
    ("settlement_source", None): (0.01, 0.50),
    ("settlement_time_only", None): (0.0, 0.0),
    ("tie_handling", None): (0.01, 0.50),
    ("retirement_withdrawal", None): (0.03, 0.50),
    ("cancellation_postponement", None): (0.02, 0.50),
    ("timing_scope", None): (0.10, 1.00),
}

_IGNORE_BELOW_CT = 0.2      # expected cost below this is noise
_FLOOR_MULTIPLIER = 2.0     # required extra edge = k x expected cost


def policy_for(divergence: str, kalshi_market_id: str,
               p_override: float | None = None) -> DivergencePolicy:
    """The trading policy for one classified divergence on one pair.

    ``p_override`` lets the empirical calibrator raise the branch probability
    above the prior (never below, until enough observations exist — enforced by
    the caller)."""
    if divergence in ("", "none", None):
        return DivergencePolicy("ignore", 0.0, 0.0)
    if divergence == "different_event":
        return DivergencePolicy("block", 100.0, 0.0)
    fam = sport_family(kalshi_market_id)
    p, loss = _SEVERITY.get((divergence, fam)) or _SEVERITY.get((divergence, None)) \
        or (0.02, 0.50)
    if p_override is not None:
        p = max(p, p_override)
    cost_ct = p * loss * 100.0
    if divergence == "timing_scope":
        # asymmetric: the wider-window side as YES turns the branch into a windfall
        return DivergencePolicy("one_way", round(cost_ct, 2), 0.0)
    if cost_ct < _IGNORE_BELOW_CT:
        return DivergencePolicy("ignore", round(cost_ct, 2), 0.0)
    return DivergencePolicy("edge_floor", round(cost_ct, 2),
                            round(cost_ct * _FLOOR_MULTIPLIER, 2) / 100.0)
