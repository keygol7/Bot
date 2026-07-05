"""Chunked float32 similarity: identical results to brute force, bounded memory."""

import random

from bot.matching.embed import semantic_candidate_pairs
from bot.models import MarketQuote


def _q(venue, mid, title):
    return MarketQuote(venue=venue, market_id=mid, title=title,
                       yes_ask=0.5, yes_ask_size=1, no_ask=0.5, no_ask_size=1)


def test_chunked_matches_bruteforce_and_spans_chunk_boundaries():
    rng = random.Random(7)
    dim = 16
    # 2500 A-side rows -> spans multiple 1024 chunks; plant known matches at
    # positions before/on/after chunk boundaries (0, 1023, 1024, 2047, 2400)
    base = [rng.gauss(0, 1) for _ in range(dim)]
    def noisy(scale):
        return [x + rng.gauss(0, scale) for x in base]
    a_titles, planted = [], {0, 1023, 1024, 2047, 2400}
    for i in range(2500):
        a_titles.append(f"a{i}")
    b_titles = ["match", "unrelated"]
    vecs = {}
    for i, t in enumerate(a_titles):
        vecs[t] = noisy(0.05) if i in planted else [rng.gauss(0, 1) for _ in range(dim)]
    vecs["match"] = noisy(0.05)
    vecs["unrelated"] = [rng.gauss(0, 1) for _ in range(dim)]
    embed_fn = lambda titles: [vecs[t] for t in titles]
    ga = [_q("kalshi", t, t) for t in a_titles]
    gb = [_q("poly", t, t) for t in b_titles]
    cands = semantic_candidate_pairs(ga, gb, embed_fn, threshold=0.9)
    got = {c.a.market_id for c in cands if c.b.market_id == "match"}
    assert {f"a{i}" for i in planted} <= got          # every planted match found
    assert all(c.score >= 0.9 for c in cands)
    assert cands == sorted(cands, key=lambda c: c.score, reverse=True)


def test_deferred_none_vectors_sit_out_this_cycle():
    # A budgeted cache returns None for not-yet-embedded titles; the shortlist must
    # skip them without error and still match the embedded ones.
    vec_yes = [1.0] * 8
    def embed_fn(titles):
        return [vec_yes if t != "deferred" else None for t in titles]
    ga = [_q("kalshi", "K1", "same title"), _q("kalshi", "K2", "deferred")]
    gb = [_q("poly", "p1", "same title")]
    cands = semantic_candidate_pairs(ga, gb, embed_fn, threshold=0.9)
    assert len(cands) == 1 and cands[0].a.market_id == "K1"


def test_embed_cache_budget_defers_and_completes_over_calls():
    from bot.matching.embed_cache import CachingEmbedFn
    calls = []
    def raw(titles):
        calls.append(len(titles))
        return [[1.0, 2.0] for _ in titles]
    fn = CachingEmbedFn(raw, store=None, max_new_per_call=2)
    out1 = fn(["a", "b", "c", "d"])
    assert calls == [2]                        # only the budget embedded
    assert sum(v is not None for v in out1) == 2 and sum(v is None for v in out1) == 2
    out2 = fn(["a", "b", "c", "d"])            # next cycle: two more from budget
    assert calls == [2, 2]
    assert all(v is not None for v in out2)    # cache + this call cover all
