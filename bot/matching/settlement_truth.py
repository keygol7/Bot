"""Settlement ground truth: verify matches from how pairs ACTUALLY settled.

Every settled pair is a completed experiment. Structural parsing and LLM confirmation
are predictions; the empirical price gate is correlation — but if a matched pair's two
legs ever settle to DIFFERENT outcomes, that is *proof* the pair was a false match
(and the "hedge" was never a hedge). Conversely, consistent settlements accumulate as
the strongest possible positive evidence for a pairing.

Sources:
  - Kalshi ``/portfolio/settlements``: authoritative per-market ``market_result``.
  - Polymarket keeps recently-settled positions readable for a window
    (``/v1/portfolio/positions?includeSettled=true``) with a ``realized`` field; which
    side paid is inferred from realized-vs-cost, INDEPENDENTLY of the Kalshi result —
    that independence is exactly what makes the comparison a test.

Divergent settlement -> record + blacklist the pair (ground truth beats every other
signal). Consistent -> record (positive evidence, queryable per pair/family).
Ambiguous poly economics (fees/partials blur which side paid) -> no verdict recorded,
never guess. Runs entirely off the trade path on a slow timer.
"""

from __future__ import annotations

import logging

log = logging.getLogger("bot.matching.settlement_truth")


def infer_poly_result(pos: dict) -> str | None:
    """Which side (``"yes"``/``"no"``) a settled Polymarket position says PAID.

    A position of |net| contracts on side S (net>0 = YES, net<0 = NO) that cost
    ``cost`` dollars realizes ``|net| - cost`` if S paid and ``-cost`` if it lost —
    symmetric for longs and shorts. ``realized`` is matched to the nearer endpoint;
    fees/slippage blur it, so we also require clear separation (>= a third of the
    payout) or return None (no guess). None for unsettled (realized 0 with the
    position's market still carrying value is indistinguishable from a loss, so we
    only run on positions the caller knows are settled).
    """
    try:
        net = float(pos.get("netPosition") or 0)
        cost = float((pos.get("cost") or {}).get("value") or 0)
        realized = float((pos.get("realized") or {}).get("value") or 0)
    except (TypeError, ValueError):
        return None
    n = abs(net)
    if n < 1e-9:
        return None
    won_r, lost_r = n - cost, -cost                    # realized if held side won / lost
    # TIGHT endpoint tolerance (fees/slippage only): a VOIDED/refunded market realizes
    # ~$0, which a loose span/3 band could misread as a "win" whenever the position's
    # cost is a large fraction of the payout (live case: a pre-match FORFEIT — Kalshi
    # honored it as a win; had Poly voided, realized 0 sat 9.6 from the win endpoint on
    # a 42-lot and would have passed a 14-dollar band). Void/odd economics -> no verdict.
    tol = max(0.05 * n, 0.50)
    d_won, d_lost = abs(realized - won_r), abs(realized - lost_r)
    if min(d_won, d_lost) > tol:
        return None                                     # void/refund/unclear -> never guess
    held_won = d_won < d_lost
    held_yes = net > 0
    return "yes" if held_yes == held_won else "no"


def audit_settlements(settlements, poly_positions: dict, pair_map: dict, store) -> int:
    """Compare settled Kalshi results against inferred Poly results for known pairs.

    ``settlements``: Kalshi settlement dicts (ticker, market_result). ``poly_positions``:
    raw slug -> position dict, INCLUDING settled ones. ``pair_map``: (venue, market) ->
    counterpart, from Store.acted_pair_map(). Records each newly-checkable pair once;
    a DIVERGENT settlement blacklists the pair. Returns how many new checks landed.
    """
    checked = 0
    for s in settlements:
        ticker = s.get("ticker")
        result_k = s.get("market_result")
        if not ticker or result_k not in ("yes", "no"):
            continue                                    # scalar/void -> not comparable
        counter = pair_map.get(("kalshi", ticker))
        if counter is None or counter[0] != "polymarket_us":
            continue
        slug = counter[1]
        pos = poly_positions.get(slug)
        if pos is None:
            continue                                    # already dropped from poly's window
        if store.settlement_checked("kalshi", ticker, "polymarket_us", slug):
            continue
        result_p = infer_poly_result(pos)
        if result_p is None:
            continue                                    # ambiguous economics -> never guess
        consistent = result_p == result_k
        store.record_settlement_check(
            "kalshi", ticker, "polymarket_us", slug,
            result_a=result_k, result_b=result_p, consistent=consistent)
        checked += 1
        if consistent:
            log.info("SETTLEMENT TRUTH: %s | %s settled CONSISTENT (%s) — match verified "
                     "by ground truth", ticker, slug, result_k)
        else:
            # Ground truth: the legs resolved differently -> this was never a hedge.
            log.warning("SETTLEMENT TRUTH: %s settled %s but %s settled %s — DIVERGENT: "
                        "proven FALSE MATCH, blacklisting", ticker, result_k, slug, result_p)
            try:
                store.blacklist_pair("kalshi", ticker, "polymarket_us", slug,
                                     reason=f"settled divergently: kalshi={result_k} "
                                            f"poly={result_p}")
            except Exception as exc:
                log.warning("settlement-truth blacklist failed for %s: %s", ticker, exc)
    return checked
