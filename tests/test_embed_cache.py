"""The embedding cache must eliminate repeat embed calls (the ~8-minute-cycle fix)."""

from array import array

from bot.data.store import Store
from bot.matching.embed import cosine, semantic_candidate_pairs
from bot.matching.embed_cache import CachingEmbedFn
from bot.models import MarketQuote


class CountingEmbedder:
    def __init__(self):
        self.calls = 0
        self.texts_embedded = 0

    def __call__(self, texts):
        self.calls += 1
        self.texts_embedded += len(texts)
        # deterministic fake vector per text
        return [[float(len(t)), float(sum(map(ord, t)) % 97), 1.0] for t in texts]


def test_repeat_titles_served_from_memory():
    inner = CountingEmbedder()
    fn = CachingEmbedFn(inner, None)
    v1 = fn(["Lakers vs Celtics", "Yankees vs Mets"])
    v2 = fn(["Lakers vs Celtics", "Yankees vs Mets"])          # second pass: all cached
    assert inner.texts_embedded == 2                            # only the first pass embedded
    assert [list(a) for a in v1] == [list(b) for b in v2]


def test_partial_miss_embeds_only_new_titles():
    inner = CountingEmbedder()
    fn = CachingEmbedFn(inner, None)
    fn(["A game", "B game"])
    fn(["B game", "C game"])                                    # only "C game" is new
    assert inner.texts_embedded == 3
    # order preserved regardless of hit/miss mix
    out = fn(["C game", "A game"])
    assert list(out[0]) == [6.0, float(sum(map(ord, "C game")) % 97), 1.0]


def test_sqlite_cache_survives_restart():
    store = Store(":memory:")
    inner = CountingEmbedder()
    fn = CachingEmbedFn(inner, store, model="m")
    fn(["Persistent title"])
    # a fresh wrapper (new process) with an empty LRU must hit SQLite, not the embedder
    inner2 = CountingEmbedder()
    fn2 = CachingEmbedFn(inner2, store, model="m")
    out = fn2(["Persistent title"])
    assert inner2.texts_embedded == 0
    assert isinstance(out[0], array) and len(out[0]) == 3


def test_lru_bound_holds():
    inner = CountingEmbedder()
    fn = CachingEmbedFn(inner, None, max_memory=3)
    fn([f"title {i}" for i in range(10)])
    assert len(fn._mem) == 3                                    # bounded


def test_cached_vectors_work_in_the_matcher():
    # array('f') vectors must flow through cosine + semantic_candidate_pairs unchanged.
    inner = CountingEmbedder()
    fn = CachingEmbedFn(inner, None)
    qa = MarketQuote(venue="kalshi", market_id="K", title="Lakers vs Celtics",
                     yes_ask=0.5, yes_ask_size=1, no_ask=0.5, no_ask_size=1)
    qb = MarketQuote(venue="polymarket_us", market_id="P", title="Lakers vs Celtics",
                     yes_ask=0.5, yes_ask_size=1, no_ask=0.5, no_ask_size=1)
    fn([qa.title])                                              # pre-warm -> cached path
    pairs = semantic_candidate_pairs([qa], [qb], fn, threshold=0.9)
    assert pairs and pairs[0].score > 0.99                      # identical titles match
    v = fn([qa.title])[0]
    assert abs(cosine(v, v) - 1.0) < 1e-9
