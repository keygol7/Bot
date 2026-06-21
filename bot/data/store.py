"""SQLite persistence — kept strictly off the latency hot path.

Stores reference data and an audit trail: markets seen, detected opportunities,
fills, realized PnL, an append-only audit log, and cached cross-venue match
verdicts (so the local LLM judges each market pair once and the result is reused).

Standard-library ``sqlite3`` only. Use ``":memory:"`` for tests.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from bot.matching.scope import (
    is_allowed_kalshi_series,
    is_tradeable_market_type,
    scope_mismatch,
)


def drop_fanout_pairs(pairs: list[tuple], max_fanout: int = 1) -> list[tuple]:
    """Keep only ~1:1 same-event pairs; drop multi-outcome cross-products.

    ``pairs`` is ``[(venue_a, market_a, venue_b, market_b, event_key), ...]``. A pair
    survives only if BOTH of its markets appear in at most ``max_fanout`` pairs. A
    market matched to many distinct counterparties is a "which of N" multi-outcome
    event (a golf field, a "who wins" market) exploded into many near-identical binary
    titles — it can be the *same event* as at most one counterparty, so when it maps to
    several the matcher can't tell which, and ALL its pairs are unsafe to trade.

    This is the deterministic backstop for an LLM that rubber-stamps "same event" on
    same-tournament/different-subject titles. ``max_fanout=1`` is strictest (true 1:1).
    """
    deg_a = Counter((va, ma) for (va, ma, vb, mb, ek) in pairs)
    deg_b = Counter((vb, mb) for (va, ma, vb, mb, ek) in pairs)
    return [
        p for p in pairs
        if deg_a[(p[0], p[1])] <= max_fanout and deg_b[(p[2], p[3])] <= max_fanout
    ]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS markets (
    venue       TEXT NOT NULL,
    market_id   TEXT NOT NULL,
    title       TEXT,
    event_key   TEXT,
    updated_at  REAL,
    PRIMARY KEY (venue, market_id)
);

CREATE TABLE IF NOT EXISTS opportunities (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              REAL NOT NULL,
    event_key       TEXT,
    buy_yes_venue   TEXT,
    buy_yes_market  TEXT,
    buy_no_venue    TEXT,
    buy_no_market   TEXT,
    yes_price       REAL,
    no_price        REAL,
    edge_per_contract REAL,
    max_contracts   REAL,
    total_profit    REAL,
    acted           INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS edge_observations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    event_key   TEXT,
    yes_venue   TEXT,
    no_venue    TEXT,
    yes_price   REAL,
    no_price    REAL,
    edge        REAL,        -- per-contract edge after fees (can be negative post-depth)
    size        REAL,        -- contracts available at top of book
    outcome     TEXT         -- executed/unwound/skipped reason: the actionable result
);

CREATE TABLE IF NOT EXISTS fills (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    venue       TEXT,
    market_id   TEXT,
    side        TEXT,
    price       REAL,
    contracts   REAL,
    notional    REAL
);

CREATE TABLE IF NOT EXISTS pnl (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      REAL NOT NULL,
    amount  REAL,
    note    TEXT
);

CREATE TABLE IF NOT EXISTS match_verdicts (
    venue_a     TEXT NOT NULL,
    market_a    TEXT NOT NULL,
    venue_b     TEXT NOT NULL,
    market_b    TEXT NOT NULL,
    same_event  INTEGER NOT NULL,   -- 1 = same event AND same resolution
    confidence  REAL,
    rationale   TEXT,
    event_key   TEXT,
    ts          REAL,
    PRIMARY KEY (venue_a, market_a, venue_b, market_b)
);

CREATE TABLE IF NOT EXISTS audit_log (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      REAL NOT NULL,
    kind    TEXT,
    payload TEXT
);
"""


class Store:
    def __init__(self, path: str = ":memory:") -> None:
        # Create the parent directory for a file-backed DB (no-op for :memory:).
        if path != ":memory:":
            parent = Path(path).expanduser().parent
            if parent and not parent.exists():
                parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        # WAL + NORMAL sync: many small writes per cycle without an fsync per commit.
        try:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.OperationalError:
            pass  # e.g. :memory: — fall back to defaults
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ---- markets ----
    def upsert_market(
        self, venue: str, market_id: str, title: str, event_key: Optional[str] = None
    ) -> None:
        self.conn.execute(
            """INSERT INTO markets (venue, market_id, title, event_key, updated_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(venue, market_id) DO UPDATE SET
                 title=excluded.title,
                 event_key=COALESCE(excluded.event_key, markets.event_key),
                 updated_at=excluded.updated_at""",
            (venue, market_id, title, event_key, time.time()),
        )
        self.conn.commit()

    def upsert_markets(self, rows) -> None:
        """Bulk upsert markets in a single transaction (one commit for the batch).

        ``rows`` is an iterable of ``(venue, market_id, title, event_key)``. This is
        the hot path during the wide scan — per-row commits are far too slow.
        """
        now = time.time()
        payload = [(v, mid, title, ek, now) for (v, mid, title, ek) in rows]
        if not payload:
            return
        self.conn.executemany(
            """INSERT INTO markets (venue, market_id, title, event_key, updated_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(venue, market_id) DO UPDATE SET
                 title=excluded.title,
                 event_key=COALESCE(excluded.event_key, markets.event_key),
                 updated_at=excluded.updated_at""",
            payload,
        )
        self.conn.commit()

    # ---- opportunities ----
    def record_opportunity(self, opp: Any, acted: bool = False) -> int:
        cur = self.conn.execute(
            """INSERT INTO opportunities
               (ts, event_key, buy_yes_venue, buy_yes_market, buy_no_venue,
                buy_no_market, yes_price, no_price, edge_per_contract,
                max_contracts, total_profit, acted)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                time.time(), opp.event_key, opp.buy_yes_venue, opp.buy_yes_market,
                opp.buy_no_venue, opp.buy_no_market, opp.yes_price, opp.no_price,
                opp.edge_per_contract, opp.max_contracts, opp.total_profit,
                1 if acted else 0,
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def record_edge(
        self, event_key: str, yes_venue: str, no_venue: str, yes_price: float,
        no_price: float, edge: float, size: float, outcome: str,
    ) -> None:
        """Log one actionable edge observation (after the streaming guards) with its
        outcome, so a soak builds a distribution of how often/how big real edges are."""
        self.conn.execute(
            """INSERT INTO edge_observations
               (ts, event_key, yes_venue, no_venue, yes_price, no_price, edge, size, outcome)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (time.time(), event_key, yes_venue, no_venue, yes_price, no_price,
             edge, size, outcome),
        )
        self.conn.commit()

    # ---- fills / pnl ----
    def record_fill(
        self, venue: str, market_id: str, side: str,
        price: float, contracts: float,
    ) -> None:
        self.conn.execute(
            """INSERT INTO fills (ts, venue, market_id, side, price, contracts, notional)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (time.time(), venue, market_id, side, price, contracts, price * contracts),
        )
        self.conn.commit()

    def record_pnl(self, amount: float, note: str = "") -> None:
        self.conn.execute(
            "INSERT INTO pnl (ts, amount, note) VALUES (?, ?, ?)",
            (time.time(), amount, note),
        )
        self.conn.commit()

    def total_pnl(self) -> float:
        row = self.conn.execute("SELECT COALESCE(SUM(amount), 0) AS s FROM pnl").fetchone()
        return float(row["s"])

    # ---- match verdict cache ----
    @staticmethod
    def _pair_key(va: str, ma: str, vb: str, mb: str) -> tuple[str, str, str, str]:
        # Order-independent: a pair is the same regardless of argument order.
        return tuple(sorted([(va, ma), (vb, mb)]))[0] + tuple(sorted([(va, ma), (vb, mb)]))[1]

    def cache_verdict(
        self, va: str, ma: str, vb: str, mb: str, *,
        same_event: bool, confidence: float, rationale: str = "",
        event_key: Optional[str] = None,
    ) -> None:
        a, ma2, b, mb2 = self._pair_key(va, ma, vb, mb)
        self.conn.execute(
            """INSERT INTO match_verdicts
               (venue_a, market_a, venue_b, market_b, same_event, confidence,
                rationale, event_key, ts)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(venue_a, market_a, venue_b, market_b) DO UPDATE SET
                 same_event=excluded.same_event,
                 confidence=excluded.confidence,
                 rationale=excluded.rationale,
                 event_key=excluded.event_key,
                 ts=excluded.ts""",
            (a, ma2, b, mb2, 1 if same_event else 0, confidence, rationale,
             event_key, time.time()),
        )
        self.conn.commit()

    def confirmed_pairs(
        self, min_confidence: float = 0.85, max_fanout: Optional[int] = 1,
        drop_scope_mismatch: bool = True, safe_types_only: bool = True,
        use_fingerprint: bool = False, fingerprint_metrics: Optional[frozenset] = None,
    ) -> list[tuple]:
        """Cached tradeable pairs: (venue_a, market_a, venue_b, market_b, event_key).

        The durable source of truth for the streaming watchlist — independent of
        per-cycle embedding/LLM variance. Four gates are applied:

        1. The SAME confidence gate as ``MatchVerdict.tradeable`` (confirmed same-event
           AND ``confidence >= min_confidence``). Without it the streamer would trade
           every low-confidence "true" the model emitted.
        2. A scope/period gate (``drop_scope_mismatch``): drop pairs whose titles
           resolve on different scopes ("win 2nd half" vs "win the match"). Cleans
           existing cache entries the LLM rubber-stamped, with no re-seed needed.
        3. A market-type whitelist (``safe_types_only``): keep only the types the
           matcher handles reliably (moneyline/draw winners, same-metric player props,
           fight method-of-victory). Excludes halves, spreads, set winners, exact
           scores, go-the-distance, announcer novelties, etc. — the classes that have
           produced false positives.
        4. A fan-out gate (``max_fanout``): drop multi-outcome cross-products where a
           market maps to many counterparties (see :func:`drop_fanout_pairs`). Pass
           ``None`` to disable.
        """
        if use_fingerprint:
            return self._fingerprint_sweep(fingerprint_metrics, max_fanout)
        rows = self.conn.execute(
            """SELECT v.venue_a, v.market_a, v.venue_b, v.market_b, v.event_key,
                      ma.title AS title_a, mb.title AS title_b
               FROM match_verdicts v
               LEFT JOIN markets ma ON ma.venue=v.venue_a AND ma.market_id=v.market_a
               LEFT JOIN markets mb ON mb.venue=v.venue_b AND mb.market_id=v.market_b
               WHERE v.same_event=1 AND v.confidence >= ?""",
            (min_confidence,),
        ).fetchall()
        pairs = []
        for r in rows:
            ta, tb = r["title_a"] or "", r["title_b"] or ""
            if drop_scope_mismatch and scope_mismatch(ta, tb):
                continue
            if safe_types_only:
                if not (is_tradeable_market_type(ta) and is_tradeable_market_type(tb)):
                    continue
                # Kalshi series allowlist (more reliable than titles). The kalshi
                # leg must be a vetted series; unknown/exotic series are excluded.
                kalshi_mkt = (
                    r["market_a"] if r["venue_a"] == "kalshi"
                    else r["market_b"] if r["venue_b"] == "kalshi" else None
                )
                if kalshi_mkt is not None and not is_allowed_kalshi_series(kalshi_mkt):
                    continue
            pairs.append(
                (r["venue_a"], r["market_a"], r["venue_b"], r["market_b"], r["event_key"])
            )
        if max_fanout is not None:
            pairs = drop_fanout_pairs(pairs, max_fanout=max_fanout)
        return pairs

    def _fingerprint_sweep(
        self, fingerprint_metrics: Optional[frozenset], max_fanout: Optional[int],
    ) -> list[tuple]:
        """Authoritative matcher: fingerprint EVERY scanned cross-venue market pair and
        keep the provable complements — independent of the embedding/LLM shortlist, which
        only ever proposed a fraction of true pairs (the recall bottleneck). The
        deterministic fingerprint is the precision gate, so any structurally complementary
        pair becomes tradeable the moment both legs are scanned. Pairs are blocked by metric
        (a complement requires equal metric) so the sweep is O(sum of per-metric K*P), not a
        full O(N^2). One leg is always Kalshi; same-venue pairs are never formed."""
        from collections import defaultdict

        from bot.matching.fingerprint import (
            are_complementary, from_kalshi, from_polymarket,
        )

        rows = self.conn.execute(
            "SELECT venue, market_id, title FROM markets"
        ).fetchall()
        # Fingerprint each market once; drop the unmatchable ones up front, then bucket by
        # metric and Kalshi/other side so only plausible counterparties are compared.
        by_metric: dict[str, dict[str, list]] = defaultdict(
            lambda: {"kalshi": [], "other": []})
        for m in rows:
            venue = m["venue"]
            f = (from_kalshi(m["market_id"], m["title"] or "") if venue == "kalshi"
                 else from_polymarket(m["market_id"], m["title"] or ""))
            if not f.matchable:
                continue
            side = "kalshi" if venue == "kalshi" else "other"
            by_metric[f.metric][side].append((venue, m["market_id"], f))

        pairs = []
        for metric, sides in by_metric.items():
            if fingerprint_metrics and metric not in fingerprint_metrics:
                continue
            for va, ma, fa in sides["kalshi"]:
                for vb, mb, fb in sides["other"]:
                    if are_complementary(fa, fb):
                        pairs.append((va, ma, vb, mb, f"{va}:{ma}|{vb}:{mb}"))
        if max_fanout is not None:
            pairs = drop_fanout_pairs(pairs, max_fanout=max_fanout)
        return pairs

    def prune_market(self, venue: str, market_id: str) -> int:
        """Delete a settled/closed market and every cached verdict referencing it.

        Returns the number of verdict rows removed. Keeps the cache focused on live
        events so the watchlist build stops re-probing dead markets every cycle.
        Safe because a closed/settled status is terminal — the market won't reopen.
        """
        cur = self.conn.execute(
            """DELETE FROM match_verdicts
               WHERE (venue_a=? AND market_a=?) OR (venue_b=? AND market_b=?)""",
            (venue, market_id, venue, market_id),
        )
        n = cur.rowcount
        self.conn.execute(
            "DELETE FROM markets WHERE venue=? AND market_id=?", (venue, market_id)
        )
        self.conn.commit()
        return n

    def get_verdict(self, va: str, ma: str, vb: str, mb: str) -> Optional[sqlite3.Row]:
        a, ma2, b, mb2 = self._pair_key(va, ma, vb, mb)
        return self.conn.execute(
            """SELECT * FROM match_verdicts
               WHERE venue_a=? AND market_a=? AND venue_b=? AND market_b=?""",
            (a, ma2, b, mb2),
        ).fetchone()

    # ---- audit ----
    def audit(self, kind: str, payload: Any) -> None:
        self.conn.execute(
            "INSERT INTO audit_log (ts, kind, payload) VALUES (?, ?, ?)",
            (time.time(), kind, json.dumps(payload, default=str)),
        )
        self.conn.commit()
