"""Semantic matcher tests: cosine + embedding-based candidate pairing (no network)."""

import pytest

from bot.matching.embed import cosine, semantic_candidate_pairs
from bot.models import MarketQuote


def mq(venue, mid, title):
    return MarketQuote(venue=venue, market_id=mid, title=title)


def test_cosine_basic():
    assert cosine([1, 0], [1, 0]) == 1.0
    assert cosine([1, 0], [0, 1]) == 0.0
    assert round(cosine([1, 1], [1, 0]), 4) == 0.7071
    assert cosine([0, 0], [1, 1]) == 0.0   # zero vector -> 0, no div error


def test_semantic_pairs_uses_embeddings_and_excludes_same_venue():
    # Fake embedder: map known titles to vectors so we control similarity.
    vectors = {
        "Fed cuts rates March 2026": [1.0, 0.0, 0.0],
        "Will the Fed lower interest rates by March?": [0.96, 0.10, 0.0],  # ~same dir
        "Lakers win the title": [0.0, 1.0, 0.0],                          # unrelated
    }

    def fake_embed(texts):
        return [vectors[t] for t in texts]

    a = [mq("kalshi", "K1", "Fed cuts rates March 2026"),
         mq("kalshi", "K2", "Lakers win the title")]
    b = [mq("polymarket_us", "P1", "Will the Fed lower interest rates by March?")]

    pairs = semantic_candidate_pairs(a, b, fake_embed, threshold=0.8)
    assert len(pairs) == 1
    assert pairs[0].a.market_id == "K1" and pairs[0].b.market_id == "P1"
    assert pairs[0].score > 0.9
    assert all(c.a.venue != c.b.venue for c in pairs)


def test_semantic_pairs_empty_when_below_threshold():
    def fake_embed(texts):
        return [[1.0, 0.0]] * len(texts) if texts and texts[0].startswith("a") else [[0.0, 1.0]] * len(texts)

    a = [mq("kalshi", "K1", "alpha event")]
    b = [mq("polymarket_us", "P1", "beta event")]
    assert semantic_candidate_pairs(a, b, fake_embed, threshold=0.5) == []


def test_semantic_pairs_empty_groups():
    assert semantic_candidate_pairs([], [mq("p", "1", "x")], lambda t: []) == []


def test_embedding_client_parses_response():
    httpx = pytest.importorskip("httpx")
    from bot.matching.embed_client import EmbeddingClient

    def handler(request):
        body = request.read()
        import json
        n = len(json.loads(body)["input"])
        return httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2, 0.3]} for _ in range(n)]})

    client = EmbeddingClient(
        "http://win-pc:11434/v1", "nomic-embed-text",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    vecs = client.embed(["a", "b", "c"])
    assert len(vecs) == 3 and vecs[0] == [0.1, 0.2, 0.3]
