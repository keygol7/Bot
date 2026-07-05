"""Accurate locked/settled profit, from first principles.

Why this module exists (2026-07-05 audit): the bot's STORED data (fills at actual
average fill prices; entry-time 'arb locked' bookings) was verified correct to the
penny — but every ad-hoc analysis layered on top of it kept getting profit wrong, in
three specific ways this module makes impossible:

1. NAIVE SELL NETTING. Summing signed notional (sells at proceeds) is NOT a cost
   basis. Selling 11 of 20 contracts bought at $0.32 leaves 9 costing 9x$0.32 —
   regardless of the sale price (the sale's gain/loss is REALIZED, separately).
   Venues use this average-cost method (verified: Poly's cost.value == average-cost
   + fees to within rounding); analyses that netted at proceeds produced phantom
   costs several dollars off per pair.
2. MARK-vs-COST FIELD CONFUSION. Kalshi's ``market_exposure_dollars`` is the CURRENT
   MARK VALUE of the position, not what was paid (verified live: exposure == mark
   for a 51-lot whose fills cost differed by $7+). Treating it as cost basis once
   turned a -$5 position into a claimed +$1.24.
3. NO PAIR ATTRIBUTION. pnl rows said 'arb locked' with no event key; per-pair booked
   profit was unqueryable (now fixed: pnl.event_key).

Definitions used here:
- LOCKED (open hedged pair): min(|yes_net|, |no_net|) x $1 settlement payout minus the
  average-cost basis of BOTH open legs (ex post fills, buy fees included when the fee
  model is supplied). This is what the pair will realize at settlement if held.
- REALIZED (from sells): proceeds minus average cost of the contracts sold, minus sell
  fees — booked when the sell happens, independent of the remaining position.
- SETTLED: for Kalshi, the venue settlement record is authoritative
  (payout(count of winning side) - yes_cost - no_cost - fee). For Poly, the venue
  'realized' field when the settled position is still visible; otherwise the ledger's
  realized + terminal settlement of the remaining position at $1/$0.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Ledger:
    """Average-cost position ledger for ONE market side-aggregate.

    Positions are tracked as a single signed net (this bot only ever holds one side
    of a market at a time; venue netting collapses opposing fills the same way).
    ``realized`` accumulates sell-time gains/losses vs average cost.
    """

    net: float = 0.0            # contracts currently held
    cost: float = 0.0           # average-cost basis of the held contracts ($)
    realized: float = 0.0       # realized from sells vs avg cost ($, ex-fees)
    buys: float = 0.0           # lifetime bought contracts (diagnostics)
    sells: float = 0.0          # lifetime sold contracts (diagnostics)

    @property
    def avg(self) -> float:
        return self.cost / self.net if self.net > 1e-9 else 0.0

    def buy(self, price: float, qty: float) -> None:
        self.net += qty
        self.cost += qty * price
        self.buys += qty

    def sell(self, price: float, qty: float) -> None:
        """Average-cost sell: realized = proceeds - avg_cost*qty; basis shrinks
        pro-rata. Selling more than held is clamped (defensive; the venue would have
        rejected it)."""
        qty = min(qty, self.net) if self.net > 1e-9 else 0.0
        if qty <= 1e-12:
            return
        avg = self.cost / self.net
        self.realized += qty * (price - avg)
        self.cost -= qty * avg
        self.net -= qty
        self.sells += qty


def build_ledgers(fill_rows) -> dict:
    """(venue, market_id, side) -> Ledger from fills rows (dicts/sqlite rows with
    venue, market_id, side, price, contracts, ts), applied in time order. Sides are
    the stored 'YES'/'NO' with '_SELL' suffix marking exits."""
    ledgers: dict = {}
    for r in sorted(fill_rows, key=lambda r: r["ts"]):
        raw = r["side"] or ""
        base = raw.replace("_SELL", "")
        key = (r["venue"], r["market_id"], base)
        led = ledgers.setdefault(key, Ledger())
        qty = float(r["contracts"] or 0)
        px = float(r["price"] or 0)
        if raw.endswith("_SELL"):
            led.sell(px, qty)
        else:
            led.buy(px, qty)
    return ledgers


@dataclass
class PairReport:
    event_key: str
    yes_leg: tuple            # (venue, market_id)
    no_leg: tuple
    hedged: float             # contracts locked (min of both open legs)
    cost_basis: float         # avg-cost of both open legs ($, ex-fee)
    locked: float             # hedged*$1 - cost_basis  (settlement outcome if held)
    realized: float           # realized-so-far from sells on either leg ($, ex-fee)
    imbalance: float = 0.0    # |yes_net - no_net| (naked/remnant portion, not counted)


def locked_pairs_report(store, fee_models: dict | None = None,
                        open_positions: dict | None = None) -> list[PairReport]:
    """Per-pair LOCKED profit for every acted pair with open contracts on both legs.

    ``open_positions``: (venue, market_id) -> |contracts| currently held per the VENUE
    (from account snapshots). When provided it is authoritative for WHICH pairs are
    open and HOW MANY contracts are hedged — the fills ledger can't see settlements
    (nothing writes a sell row when a market pays out), so without the mask every
    long-settled pair looks open forever. The ledger supplies the avg-cost basis."""
    rows = store.conn.execute(
        "SELECT venue, market_id, side, price, contracts, ts FROM fills ORDER BY ts"
    ).fetchall()
    ledgers = build_ledgers(rows)

    # net + cost per (venue, market): collapse YES/NO ledgers into a side-aware view.
    def open_side(venue, market):
        y = ledgers.get((venue, market, "YES"))
        n = ledgers.get((venue, market, "NO"))
        if y and y.net > 1e-9:
            return "YES", y
        if n and n.net > 1e-9:
            return "NO", n
        return None, None

    reports = []
    seen = set()
    for r in store.conn.execute(
        "SELECT DISTINCT event_key, buy_yes_venue, buy_yes_market, buy_no_venue,"
        " buy_no_market FROM opportunities WHERE acted = 1"
    ):
        key = store._pair_key(r["buy_yes_venue"], r["buy_yes_market"],
                              r["buy_no_venue"], r["buy_no_market"])
        if key in seen:
            continue
        seen.add(key)
        ys, yl = open_side(r["buy_yes_venue"], r["buy_yes_market"])
        ns, nl = open_side(r["buy_no_venue"], r["buy_no_market"])
        if yl is None or nl is None:
            continue                                  # not open on both legs
        if open_positions is not None:
            vy = abs(open_positions.get((r["buy_yes_venue"], r["buy_yes_market"]), 0.0))
            vn = abs(open_positions.get((r["buy_no_venue"], r["buy_no_market"]), 0.0))
            if vy < 1e-9 or vn < 1e-9:
                continue                              # settled/closed per the venue
            hedged = min(vy, vn)                      # venue truth for the open count
            imb = abs(vy - vn)
        else:
            hedged = min(yl.net, nl.net)
            imb = abs(yl.net - nl.net)
        if hedged < 1e-9:
            continue
        # cost basis of the HEDGED portion (pro-rata of each leg's basis)
        basis = yl.avg * hedged + nl.avg * hedged
        fees = 0.0
        if fee_models:
            fy = fee_models.get(r["buy_yes_venue"])
            fn = fee_models.get(r["buy_no_venue"])
            if fy is not None:
                fees += fy.fee(yl.avg, hedged)
            if fn is not None:
                fees += fn.fee(nl.avg, hedged)
        realized = (yl.realized + nl.realized)
        reports.append(PairReport(
            event_key=r["event_key"] or f"{r['buy_yes_market']}|{r['buy_no_market']}",
            yes_leg=(r["buy_yes_venue"], r["buy_yes_market"]),
            no_leg=(r["buy_no_venue"], r["buy_no_market"]),
            hedged=round(hedged, 2),
            cost_basis=round(basis + fees, 4),
            locked=round(hedged * 1.0 - basis - fees, 4),
            realized=round(realized, 4),
            imbalance=round(imb, 2),
        ))
    reports.sort(key=lambda p: p.locked)
    return reports


def kalshi_settled_pnl(settlements, since_ts: float | None = None) -> list[tuple]:
    """(ticker, realized) per Kalshi settlement record — the venue-authoritative math:
    $1 x winning-side count held, minus both sides' total cost, minus fees. Records
    already net every buy/sell over the market's life (verified against the app)."""
    from datetime import datetime

    out = []
    for x in settlements:
        res = x.get("market_result")
        if res not in ("yes", "no"):
            continue
        if since_ts is not None:
            t = x.get("settled_time")
            ts = datetime.fromisoformat(str(t).replace("Z", "+00:00")).timestamp() if t else 0
            if ts < since_ts:
                continue
        pay = float(x["yes_count_fp"] if res == "yes" else x["no_count_fp"])
        realized = (pay - float(x["yes_total_cost_dollars"])
                    - float(x["no_total_cost_dollars"]) - float(x["fee_cost"]))
        out.append((x["ticker"], round(realized, 4)))
    return out
