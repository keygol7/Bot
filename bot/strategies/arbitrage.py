"""Cross-venue and single-venue binary-market arbitrage detection.

A binary market pays exactly $1 to whichever side (YES or NO) resolves true. So if
you can acquire one YES and one NO for a *combined* cost below $1 (after fees), the
$1 payout is locked in regardless of outcome — the difference is risk-free profit.

  - Cross-venue: buy YES on venue A and NO on venue B (or vice-versa).
  - Single-venue ("bundle"): buy YES and NO on the same venue when they sum to < $1.

This module is pure arithmetic over normalized :class:`MarketQuote` objects and the
per-venue :class:`FeeModel`. It performs no I/O and sits off the latency hot path's
network — detection runs against in-memory quotes.

IMPORTANT: detection only establishes a *price* edge. Acting on it safely also
requires that the two markets resolve on identical criteria — that check is the
matcher's job (see ``bot.matching``); never trade an unconfirmed cross-venue pair.
"""

from __future__ import annotations

from dataclasses import dataclass

from bot.fees import FeeModel, ZeroFeeModel, per_contract_fee
from bot.models import MarketQuote


@dataclass
class ArbOpportunity:
    """A detected arbitrage. ``edge_per_contract`` is profit per YES+NO pair after
    fees; ``total_profit`` is the realizable profit at ``max_contracts`` using the
    venues' actual (size-scaled) fees."""

    event_key: str | None
    buy_yes_venue: str
    buy_yes_market: str
    buy_no_venue: str
    buy_no_market: str
    yes_price: float
    no_price: float
    gross_cost: float          # yes_price + no_price (per pair, before fees)
    fee_per_pair: float        # fees for 1 YES + 1 NO
    edge_per_contract: float   # 1 - gross_cost - fee_per_pair
    max_contracts: float       # min liquidity across the two legs
    total_fees: float          # fees at max_contracts (size-scaled, accurate)
    total_profit: float        # max_contracts * (1 - gross_cost) - total_fees
    notional: float            # capital deployed = gross_cost * max_contracts
    yes_size: float = 0.0      # YES leg's own top-of-book depth (0 = unknown)
    no_size: float = 0.0       # NO leg's own top-of-book depth (0 = unknown)

    @property
    def is_single_venue(self) -> bool:
        return self.buy_yes_venue == self.buy_no_venue

    def __str__(self) -> str:
        kind = "BUNDLE" if self.is_single_venue else "CROSS"
        return (
            f"[{kind}] {self.event_key or '?'} | "
            f"YES@{self.buy_yes_venue}:{self.buy_yes_market}={self.yes_price:.3f} + "
            f"NO@{self.buy_no_venue}:{self.buy_no_market}={self.no_price:.3f} "
            f"(cost {self.gross_cost:.3f}) | edge/ct ${self.edge_per_contract:.4f} | "
            f"max {self.max_contracts:g} ct -> ${self.total_profit:.2f} profit"
        )


def _build(
    *,
    yes_q: MarketQuote,
    yes_fee: FeeModel,
    no_q: MarketQuote,
    no_fee: FeeModel,
) -> ArbOpportunity | None:
    """Build an opportunity for buying YES on ``yes_q`` and NO on ``no_q``.

    Returns ``None`` if either leg lacks a takeable price/size.
    """
    yes_price = yes_q.yes_ask
    no_price = no_q.no_ask
    if yes_price is None or no_price is None:
        return None

    max_contracts = min(yes_q.yes_ask_size, no_q.no_ask_size)
    if max_contracts <= 0:
        return None

    gross_cost = yes_price + no_price
    # Gate on the UNROUNDED per-contract rate: fee(price, 1) quantizes to a whole cent
    # (an error the size of the edge floor itself); settlement still books the venue's
    # rounded fee at real size via total_fees below.
    fee_per_pair = per_contract_fee(yes_fee, yes_price) + per_contract_fee(no_fee, no_price)
    edge_per_contract = 1.0 - gross_cost - fee_per_pair

    total_fees = (
        yes_fee.fee(yes_price, max_contracts)
        + no_fee.fee(no_price, max_contracts)
    )
    total_profit = max_contracts * (1.0 - gross_cost) - total_fees

    return ArbOpportunity(
        event_key=yes_q.event_key or no_q.event_key,
        buy_yes_venue=yes_q.venue,
        buy_yes_market=yes_q.market_id,
        buy_no_venue=no_q.venue,
        buy_no_market=no_q.market_id,
        yes_price=yes_price,
        no_price=no_price,
        gross_cost=gross_cost,
        fee_per_pair=fee_per_pair,
        edge_per_contract=edge_per_contract,
        max_contracts=max_contracts,
        total_fees=total_fees,
        total_profit=total_profit,
        notional=gross_cost * max_contracts,
    )


def detect_cross_venue(
    a: MarketQuote,
    b: MarketQuote,
    *,
    fee_a: FeeModel | None = None,
    fee_b: FeeModel | None = None,
    min_edge: float = 0.0,
) -> list[ArbOpportunity]:
    """Find risk-free arbs between two *matched* markets on different venues.

    Evaluates both legs (YES on A / NO on B, and NO on A / YES on B) and returns
    every opportunity whose per-contract edge strictly exceeds ``min_edge``,
    best first.
    """
    fee_a = fee_a or ZeroFeeModel()
    fee_b = fee_b or ZeroFeeModel()

    candidates = [
        _build(yes_q=a, yes_fee=fee_a, no_q=b, no_fee=fee_b),
        _build(yes_q=b, yes_fee=fee_b, no_q=a, no_fee=fee_a),
    ]
    opps = [o for o in candidates if o is not None and o.edge_per_contract > min_edge]
    opps.sort(key=lambda o: o.edge_per_contract, reverse=True)
    return opps


def detect_bundle(
    q: MarketQuote,
    *,
    fee: FeeModel | None = None,
    min_edge: float = 0.0,
) -> ArbOpportunity | None:
    """Single-venue arb: YES + NO on the same market summing to < $1 after fees."""
    fee = fee or ZeroFeeModel()
    opp = _build(yes_q=q, yes_fee=fee, no_q=q, no_fee=fee)
    if opp is not None and opp.edge_per_contract > min_edge:
        return opp
    return None


# --- Price-only edge checks (ignore size) -----------------------------------
# Used by the two-phase scanner: a cheap wide price scan returns top-of-book
# prices without depth, so these compute the per-contract edge from prices alone
# to shortlist which markets are worth a deeper (sized) fetch. Detection of an
# actionable, sizeable opportunity still uses detect_cross_venue / detect_bundle
# on the sized quotes.


def _pair_price_edge(
    yes_q: MarketQuote, yes_fee: FeeModel, no_q: MarketQuote, no_fee: FeeModel
) -> float | None:
    """Per-contract edge of buying YES on ``yes_q`` and NO on ``no_q``, prices only."""
    if yes_q.yes_ask is None or no_q.no_ask is None:
        return None
    fees = per_contract_fee(yes_fee, yes_q.yes_ask) + per_contract_fee(no_fee, no_q.no_ask)
    return 1.0 - (yes_q.yes_ask + no_q.no_ask) - fees


def cross_price_edge(
    a: MarketQuote, b: MarketQuote,
    fee_a: FeeModel | None = None, fee_b: FeeModel | None = None,
) -> float:
    """Best per-contract cross-venue edge across both directions (prices only).

    Returns ``-inf`` if neither direction has both legs quoted.
    """
    fee_a = fee_a or ZeroFeeModel()
    fee_b = fee_b or ZeroFeeModel()
    edges = [
        _pair_price_edge(a, fee_a, b, fee_b),   # YES@a + NO@b
        _pair_price_edge(b, fee_b, a, fee_a),   # YES@b + NO@a
    ]
    present = [e for e in edges if e is not None]
    return max(present) if present else float("-inf")


def bundle_price_edge(q: MarketQuote, fee: FeeModel | None = None) -> float | None:
    """Per-contract single-venue (YES+NO) edge from prices only."""
    fee = fee or ZeroFeeModel()
    return _pair_price_edge(q, fee, q, fee)
