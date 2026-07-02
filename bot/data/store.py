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
    id_scope_mismatch,
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

-- Empirical false-match learning loop: pairs CONFIRMED to not be the same event (their
-- prices don't behave as complements). Persists the engine's empirical verdict so a known
-- false match (e.g. R6 vs CoD) is excluded from matching across restarts — never re-observed,
-- never re-traded. Order-independent key (venue_a/market_a < venue_b/market_b).
CREATE TABLE IF NOT EXISTS match_blacklist (
    venue_a   TEXT NOT NULL,
    market_a  TEXT NOT NULL,
    venue_b   TEXT NOT NULL,
    market_b  TEXT NOT NULL,
    reason    TEXT,
    mean_sum  REAL,
    samples   INTEGER,
    ts        REAL,
    PRIMARY KEY (venue_a, market_a, venue_b, market_b)
);

-- Empirical per-market fill reliability: did a FOK order actually FILL (real depth) or
-- KILL/REJECT (phantom depth)? This is the live, learned replacement for the volume proxy —
-- a market is probed small until it proves it fills, then scaled up; one that keeps failing
-- its hedge is excluded. Persisted so the verdict survives restarts.
CREATE TABLE IF NOT EXISTS market_reliability (
    venue      TEXT NOT NULL,
    market_id  TEXT NOT NULL,
    fills      INTEGER NOT NULL DEFAULT 0,
    fails      INTEGER NOT NULL DEFAULT 0,
    streak     INTEGER NOT NULL DEFAULT 0,   -- CONSECUTIVE fails (reset to 0 on a fill)
    max_fill   REAL NOT NULL DEFAULT 0,       -- largest size a real FOK has FILLED (for scale-up)
    ts         REAL,
    PRIMARY KEY (venue, market_id)
);

-- Title-embedding cache. Market titles are stable strings, but the discovery cycle was
-- re-embedding the ENTIRE scanned board every pass (~3.4k calls/cycle -> ~8-minute
-- cycles). Persisting text-hash -> vector means each pass embeds only NEW titles, and a
-- restart doesn't re-pay the whole board. vec is packed float32 (array('f').tobytes()).
CREATE TABLE IF NOT EXISTS embedding_cache (
    model      TEXT NOT NULL,
    text_hash  TEXT NOT NULL,
    vec        BLOB NOT NULL,
    ts         REAL NOT NULL,
    PRIMARY KEY (model, text_hash)
);

-- Settlement ground truth: every settled pair is a completed experiment. If a matched
-- pair's two legs ever SETTLE DIFFERENTLY, that's PROOF of a false match (no statistics
-- needed) -> blacklist. Consistent settlements accumulate as positive evidence per pair.
CREATE TABLE IF NOT EXISTS settlement_checks (
    venue_a    TEXT NOT NULL,
    market_a   TEXT NOT NULL,
    venue_b    TEXT NOT NULL,
    market_b   TEXT NOT NULL,
    result_a   TEXT,               -- yes/no as settled on venue A
    result_b   TEXT,               -- yes/no as settled on venue B (inferred from realized)
    consistent INTEGER NOT NULL,   -- 1 = same outcome (true match), 0 = DIVERGED (false)
    ts         REAL NOT NULL,
    PRIMARY KEY (venue_a, market_a, venue_b, market_b)
);

-- Time/market indices: the settled-PnL reconciliation, reconcile pairing, and every
-- "last N hours" query scan these tables, which grow without bound.
CREATE INDEX IF NOT EXISTS idx_pnl_ts            ON pnl (ts);
CREATE INDEX IF NOT EXISTS idx_fills_ts          ON fills (ts);
CREATE INDEX IF NOT EXISTS idx_fills_market      ON fills (venue, market_id);
CREATE INDEX IF NOT EXISTS idx_opportunities_ts  ON opportunities (ts);
CREATE INDEX IF NOT EXISTS idx_opps_acted        ON opportunities (acted);
CREATE INDEX IF NOT EXISTS idx_embed_ts          ON embedding_cache (ts);
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
        # Migrations: add market_reliability columns to DBs created before they existed.
        _rel_cols = {r["name"] for r in
                     self.conn.execute("PRAGMA table_info(market_reliability)")}
        if "streak" not in _rel_cols:
            self.conn.execute(
                "ALTER TABLE market_reliability ADD COLUMN streak INTEGER NOT NULL DEFAULT 0")
        if "max_fill" not in _rel_cols:
            self.conn.execute(
                "ALTER TABLE market_reliability ADD COLUMN max_fill REAL NOT NULL DEFAULT 0")
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

    # ---- empirical false-match blacklist (Layer 2 learning loop) ----
    def blacklist_pair(self, va: str, ma: str, vb: str, mb: str, *,
                       reason: str = "", mean_sum: float | None = None,
                       samples: int | None = None) -> None:
        """Persist a CONFIRMED false match so it's excluded from matching forever (the
        engine calls this when a pair's price behavior empirically proves the legs aren't
        complements). Order-independent; idempotent (keeps the latest verdict)."""
        a, ma2, b, mb2 = self._pair_key(va, ma, vb, mb)
        self.conn.execute(
            """INSERT INTO match_blacklist
                 (venue_a, market_a, venue_b, market_b, reason, mean_sum, samples, ts)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(venue_a, market_a, venue_b, market_b) DO UPDATE SET
                 reason=excluded.reason, mean_sum=excluded.mean_sum,
                 samples=excluded.samples, ts=excluded.ts""",
            (a, ma2, b, mb2, reason, mean_sum, samples, time.time()),
        )
        self.conn.commit()

    def blacklisted_keys(self) -> set:
        """All confirmed false-match pairs as order-independent keys, for fast exclusion."""
        return {
            (r["venue_a"], r["market_a"], r["venue_b"], r["market_b"])
            for r in self.conn.execute(
                "SELECT venue_a, market_a, venue_b, market_b FROM match_blacklist")
        }

    def _drop_blacklisted(self, pairs: list[tuple]) -> list[tuple]:
        """Remove any confirmed false-match pairs from a watchlist result."""
        bl = self.blacklisted_keys()
        if not bl:
            return pairs
        return [p for p in pairs if self._pair_key(p[0], p[1], p[2], p[3]) not in bl]

    # ---- empirical per-market fill reliability (probe-then-scale) ----
    def record_market_outcome(self, venue: str, market_id: str, ok: bool,
                              fill_size: float = 0.0) -> None:
        """Record one FOK outcome for a market: ok=filled (real depth) / not (phantom).
        ``streak`` is the CONSECUTIVE-fail count (reset to 0 on a fill) — it catches a
        once-proven market whose depth later vanishes. ``max_fill`` tracks the LARGEST size
        a real FOK actually filled, so the sizer can scale up on demonstrated depth instead
        of a slow per-fill count."""
        mf = float(fill_size) if ok else 0.0
        self.conn.execute(
            """INSERT INTO market_reliability (venue, market_id, fills, fails, streak, max_fill, ts)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(venue, market_id) DO UPDATE SET
                 fills=fills+?, fails=fails+?,
                 streak=CASE WHEN ?=1 THEN 0 ELSE streak+1 END,
                 max_fill=MAX(max_fill, ?),
                 ts=excluded.ts""",
            (venue, market_id, 1 if ok else 0, 0 if ok else 1, 0 if ok else 1, mf, time.time(),
             1 if ok else 0, 0 if ok else 1, 1 if ok else 0, mf),
        )
        self.conn.commit()

    def market_reliability(self) -> dict:
        """All markets' (fills, fails, streak, max_fill) by (venue, market_id), for the gate."""
        return {
            (r["venue"], r["market_id"]): (r["fills"], r["fails"], r["streak"], r["max_fill"])
            for r in self.conn.execute(
                "SELECT venue, market_id, fills, fails, streak, max_fill FROM market_reliability")
        }

    def acted_pair_map(self) -> dict:
        """(venue, market) -> its historical counterpart, from every ACTED opportunity.

        The live watchlist forgets a pair once it's pruned (settled/thin), but a held
        position on a forgotten market still has a knowable counterpart here — used by
        the reconcile to verify 'unpaired' positions instead of just logging them
        (the TPZRL naked leg sat exactly in that blind spot)."""
        out: dict = {}
        for r in self.conn.execute(
            "SELECT DISTINCT buy_yes_venue, buy_yes_market, buy_no_venue, buy_no_market "
            "FROM opportunities WHERE acted = 1"):
            a = (r["buy_yes_venue"], r["buy_yes_market"])
            b = (r["buy_no_venue"], r["buy_no_market"])
            out[a] = b
            out[b] = a
        return out

    # ---- settlement ground truth ----

    def record_settlement_check(self, va: str, ma: str, vb: str, mb: str, *,
                                result_a: str, result_b: str, consistent: bool) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO settlement_checks "
            "(venue_a, market_a, venue_b, market_b, result_a, result_b, consistent, ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (va, ma, vb, mb, result_a, result_b, 1 if consistent else 0, time.time()))
        self.conn.commit()

    def settlement_checked(self, va: str, ma: str, vb: str, mb: str) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM settlement_checks WHERE venue_a=? AND market_a=? "
            "AND venue_b=? AND market_b=?", (va, ma, vb, mb)).fetchone() is not None

    def settlement_consistency(self) -> tuple[int, int]:
        """(consistent, divergent) totals — the ground-truth track record."""
        row = self.conn.execute(
            "SELECT SUM(consistent) c, SUM(1 - consistent) d FROM settlement_checks"
        ).fetchone()
        return (row["c"] or 0, row["d"] or 0)

    # ---- embedding cache (see schema comment) ----

    def embeddings_get(self, model: str, hashes: list[str]) -> dict[str, bytes]:
        """Cached packed-float32 vectors for ``hashes`` (missing ones absent)."""
        out: dict[str, bytes] = {}
        CHUNK = 500                                   # stay under SQLite's param limit
        for i in range(0, len(hashes), CHUNK):
            chunk = hashes[i:i + CHUNK]
            marks = ",".join("?" * len(chunk))
            for r in self.conn.execute(
                f"SELECT text_hash, vec FROM embedding_cache "
                f"WHERE model = ? AND text_hash IN ({marks})", [model, *chunk]):
                out[r["text_hash"]] = r["vec"]
        return out

    def embeddings_put(self, model: str, rows: list[tuple[str, bytes]]) -> None:
        """Persist ``(text_hash, packed_vec)`` rows (INSERT OR REPLACE)."""
        now = time.time()
        self.conn.executemany(
            "INSERT OR REPLACE INTO embedding_cache (model, text_hash, vec, ts) "
            "VALUES (?, ?, ?, ?)",
            [(model, h, v, now) for h, v in rows])
        self.conn.commit()

    def embeddings_prune(self, *, max_age_days: float = 30.0) -> int:
        """Drop cache entries older than ``max_age_days`` (titles that left the board)."""
        cutoff = time.time() - max_age_days * 86400
        cur = self.conn.execute("DELETE FROM embedding_cache WHERE ts < ?", (cutoff,))
        self.conn.commit()
        return cur.rowcount

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
        sweep_max_past_s: Optional[float] = None, combine_verdicts: bool = False,
    ) -> list[tuple]:
        """Cached tradeable pairs: (venue_a, market_a, venue_b, market_b, event_key).

        The durable source of truth for the streaming watchlist — independent of
        per-cycle embedding/LLM variance. Three matcher modes:

        * ``use_fingerprint=False`` — LLM ``match_verdicts`` cache only (with the
          scope/type/series + fan-out guards below).
        * ``use_fingerprint=True, combine_verdicts=False`` — deterministic fingerprint
          sweep only (``_fingerprint_sweep``); the verdict cache is ignored.
        * ``use_fingerprint=True, combine_verdicts=True`` — **UNION** of both, for the
          most matches without losing precision: a pair from EITHER matcher is admitted,
          deduped by order-independent pair key (a pair both find counts once = mutually
          confirmed), then a SINGLE shared fan-out backstop drops any market the two
          disagree on (mapped to >1 counterparty → ambiguous). Net recall ≥ either method
          alone; each survivor is vetted by its own precision bar and conflicts are caught.

        Verdict-side gates (also applied to the verdict half of the union):

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
        if use_fingerprint and not combine_verdicts:
            return self._drop_blacklisted(
                self._fingerprint_sweep(fingerprint_metrics, max_fanout, sweep_max_past_s))

        verdict = self._verdict_pairs(min_confidence, drop_scope_mismatch, safe_types_only)
        if not use_fingerprint:
            # LLM verdict cache only.
            return self._drop_blacklisted(
                drop_fanout_pairs(verdict, max_fanout=max_fanout)
                if max_fanout is not None else verdict)

        # UNION mode: merge the deterministic sweep with the LLM verdicts. Fingerprint
        # legs are kalshi-first and "kalshi" sorts before "polymarket_us", so both
        # sources put the Kalshi leg in slot A — the shared fan-out below counts degree
        # consistently across them. Pull the sweep WITHOUT its own fan-out so the backstop
        # runs once over the combined set (a market the two map to different counterparties
        # is ambiguous and both pairs drop). Dedup keeps the first occurrence per pair key.
        fp = self._fingerprint_sweep(fingerprint_metrics, None, sweep_max_past_s)
        merged: dict = {}
        for p in (*fp, *verdict):
            merged.setdefault(self._pair_key(p[0], p[1], p[2], p[3]), p)
        pairs = list(merged.values())
        if max_fanout is not None:
            pairs = drop_fanout_pairs(pairs, max_fanout=max_fanout)
        return self._drop_blacklisted(pairs)

    def _verdict_pairs(
        self, min_confidence: float, drop_scope_mismatch: bool, safe_types_only: bool,
    ) -> list[tuple]:
        """LLM ``match_verdicts`` confirmed same-event pairs with the scope/type/series
        precision guards applied. Pre-fan-out — the caller owns the fan-out backstop so it
        can run once over a union. See :meth:`confirmed_pairs` for gate descriptions."""
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
            # Identifier-level scope: filters CACHED verdicts too, so handicap/half
            # slugs the LLM already rubber-stamped (plain-matchup titles) drop out of
            # the watchlist instead of persisting as spread-vs-moneyline false matches.
            if drop_scope_mismatch and id_scope_mismatch(r["market_a"], r["market_b"]):
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
        return pairs

    def _fingerprint_sweep(
        self, fingerprint_metrics: Optional[frozenset], max_fanout: Optional[int],
        max_past_s: Optional[float] = None,
    ) -> list[tuple]:
        """Authoritative matcher: fingerprint EVERY scanned cross-venue market pair and
        keep the provable complements — independent of the embedding/LLM shortlist, which
        only ever proposed a fraction of true pairs (the recall bottleneck). The
        deterministic fingerprint is the precision gate, so any structurally complementary
        pair becomes tradeable the moment both legs are scanned. Pairs are blocked by metric
        (a complement requires equal metric) so the sweep is O(sum of per-metric K*P), not a
        full O(N^2). One leg is always Kalshi; same-venue pairs are never formed.

        ``max_past_s`` drops markets whose EVENT date is more than that many seconds in the
        past: settled games linger in the table (never pruned), and the sweep would
        otherwise resurface every long-settled match ever scanned. Event date comes from the
        fingerprint (parsed from the ticker/slug), so this is independent of scan cadence —
        unlike an updated_at bound, which assumes a tight re-scan that the streamer doesn't
        do. Markets with no parseable date are kept (fail open). ``None`` keeps everything."""
        from collections import defaultdict

        from bot.matching.fingerprint import (
            are_complementary, from_kalshi, from_polymarket,
        )

        rows = self.conn.execute(
            "SELECT venue, market_id, title FROM markets"
        ).fetchall()
        cutoff = (time.time() - max_past_s) if max_past_s is not None else None
        # Fingerprint each market once; drop the unmatchable (and long-settled) ones up
        # front, then bucket by metric and Kalshi/other side so only plausible
        # counterparties are compared.
        by_metric: dict[str, dict[str, list]] = defaultdict(
            lambda: {"kalshi": [], "other": []})
        for m in rows:
            venue = m["venue"]
            f = (from_kalshi(m["market_id"], m["title"] or "") if venue == "kalshi"
                 else from_polymarket(m["market_id"], m["title"] or ""))
            if not f.matchable:
                continue
            if cutoff is not None and f.date is not None and f.date < cutoff:
                continue                              # event already happened -> settled
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
