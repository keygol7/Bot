"""Persistent title-embedding cache.

Market titles are stable strings, but the discovery cycle was re-embedding the ENTIRE
scanned board on every pass (~3.4k embedding calls per cycle -> ~8-minute cycles that
kept the streaming fast path dark most of the time). Wrapping the embed function with
this cache means each pass embeds only titles it has never seen; everything else is a
dict lookup (in-memory LRU) or a SQLite read (across restarts).

Vectors are stored as packed float32 (``array('f')``) — ~3 KB each for a 768-dim model —
both in memory and in the ``embedding_cache`` table. ``array`` objects duck-type as
sequences of floats, so the cosine/matrix code downstream (including the numpy path)
consumes them unchanged.
"""

from __future__ import annotations

import hashlib
import logging
from array import array
from collections import OrderedDict

log = logging.getLogger("bot.matching.embed_cache")


def _key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class CachingEmbedFn:
    """Wrap an ``embed(texts) -> vectors`` callable with an in-memory LRU + SQLite cache.

    Drop-in for the raw embed_fn everywhere (same call signature). ``store`` may be None
    (in-memory only). ``max_memory`` bounds the LRU (~3 KB/entry at 768 dims).
    """

    def __init__(self, embed_fn, store=None, *, model: str = "", max_memory: int = 16000,
                 prune_days: float = 30.0, max_new_per_call: int = 0) -> None:
        self._embed = embed_fn
        self._store = store
        self._model = model or "default"
        self._mem: OrderedDict[str, array] = OrderedDict()
        self._max_memory = max_memory
        self._prune_days = prune_days
        self._pruned_once = False
        # Incremental-backfill budget: embed at most this many NEVER-SEEN titles per
        # call; the rest return None this cycle and get embedded on later cycles. An
        # unbounded backfill of a whole 62k board froze the event loop for ~10min and
        # ground a 4GB box into D-state memory-reclaim stalls (no swap, 189MB free).
        # 0 = unlimited.
        self._max_new = max_new_per_call

    def _remember(self, key: str, vec: array) -> None:
        self._mem[key] = vec
        self._mem.move_to_end(key)
        while len(self._mem) > self._max_memory:
            self._mem.popitem(last=False)

    def __call__(self, texts: list[str]) -> list:
        keys = [_key(t) for t in texts]
        out: dict[int, array] = {}
        misses: list[int] = []

        # 1) in-memory LRU
        for i, k in enumerate(keys):
            vec = self._mem.get(k)
            if vec is not None:
                self._mem.move_to_end(k)
                out[i] = vec
            else:
                misses.append(i)

        # 2) SQLite (survives restarts)
        if misses and self._store is not None:
            try:
                found = self._store.embeddings_get(self._model, [keys[i] for i in misses])
            except Exception as exc:                      # cache must never break matching
                log.warning("embedding cache read failed: %s", exc)
                found = {}
            still: list[int] = []
            for i in misses:
                blob = found.get(keys[i])
                if blob is not None:
                    vec = array("f")
                    vec.frombytes(blob)
                    out[i] = vec
                    self._remember(keys[i], vec)
                else:
                    still.append(i)
            misses = still

        # 3) the real embedder, only for never-seen titles (budgeted)
        deferred: list[int] = []
        if misses and self._max_new > 0 and len(misses) > self._max_new:
            deferred = misses[self._max_new:]
            misses = misses[:self._max_new]
            log.info("embeddings: deferring %d uncached titles to later cycles "
                     "(budget %d/cycle)", len(deferred), self._max_new)
        if misses:
            vecs = self._embed([texts[i] for i in misses])
            rows: list[tuple[str, bytes]] = []
            for i, v in zip(misses, vecs, strict=True):
                vec = array("f", v)
                out[i] = vec
                self._remember(keys[i], vec)
                rows.append((keys[i], vec.tobytes()))
            if self._store is not None and rows:
                try:
                    self._store.embeddings_put(self._model, rows)
                    # One prune per process: drop entries for titles long off the board.
                    if not self._pruned_once:
                        self._pruned_once = True
                        dropped = self._store.embeddings_prune(max_age_days=self._prune_days)
                        if dropped:
                            log.info("embedding cache: pruned %d stale entries", dropped)
                except Exception as exc:
                    log.warning("embedding cache write failed: %s", exc)
            log.info("embeddings: %d/%d titles were new (rest served from cache)",
                     len(misses), len(texts))

        return [out.get(i) for i in range(len(texts))]   # None = deferred this cycle
