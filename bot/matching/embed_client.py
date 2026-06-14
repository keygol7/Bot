"""Client for the on-prem embedding model (OpenAI-compatible: vLLM or Ollama).

Provides ``embed(texts) -> list[vector]`` for the semantic market matcher. Talks to
``LLM_BASE_URL`` on localhost/LAN; nothing leaves the box. The HTTP client is
injectable so tests can supply an ``httpx`` MockTransport.
"""

from __future__ import annotations

from typing import Any, Callable


class EmbeddingClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        client: Any = None,
        timeout: float = 60.0,
        batch_size: int = 64,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.batch_size = batch_size
        self._client = client

    def _http(self):
        if self._client is None:
            import httpx  # lazy

            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text (batched)."""
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            resp = self._http().post(
                f"{self.base_url}/embeddings", json={"model": self.model, "input": batch}
            )
            resp.raise_for_status()
            data = resp.json().get("data", [])
            out.extend(item["embedding"] for item in data)
        return out

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


def make_embed_fn(llm_cfg: Any) -> Callable[[list[str]], list[list[float]]]:
    """Build an ``embed(texts) -> vectors`` callable from an ``LLMConfig``."""
    client = EmbeddingClient(llm_cfg.base_url, llm_cfg.embedding_model)
    return client.embed
