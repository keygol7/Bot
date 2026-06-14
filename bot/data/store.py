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
from pathlib import Path
from typing import Any, Optional

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

    def confirmed_pairs(self) -> list[tuple]:
        """All cached same-event pairs: (venue_a, market_a, venue_b, market_b, event_key).

        This is the durable source of truth for the streaming watchlist — independent
        of per-cycle embedding/LLM variance."""
        rows = self.conn.execute(
            "SELECT venue_a, market_a, venue_b, market_b, event_key "
            "FROM match_verdicts WHERE same_event=1"
        ).fetchall()
        return [
            (r["venue_a"], r["market_a"], r["venue_b"], r["market_b"], r["event_key"])
            for r in rows
        ]

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
