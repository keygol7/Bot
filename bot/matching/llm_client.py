"""Client for the on-prem reasoning model (OpenAI-compatible: vLLM or Ollama).

Provides the ``complete(prompt) -> str`` callable that
``bot.matching.llm_match.confirm_match`` expects. Talks to ``LLM_BASE_URL`` on
localhost; nothing leaves the box. Kept out of the package ``__init__`` so importing
the matching package never requires ``httpx``.

The HTTP client is injectable so tests can supply an ``httpx`` MockTransport — no
network and no model needed to exercise the parsing path.
"""

from __future__ import annotations

from typing import Any, Callable


class LocalLLMClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        client: Any = None,   # optional preconfigured httpx.Client (tests)
        timeout: float = 60.0,
        temperature: float = 0.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self._client = client

    def _http(self):
        if self._client is None:
            import httpx  # lazy

            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def complete(self, prompt: str) -> str:
        """Single-turn completion. Returns the assistant message text."""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        resp = self._http().post(f"{self.base_url}/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def list_models(self) -> list[str]:
        """Return the model ids the endpoint reports (GET /models)."""
        resp = self._http().get(f"{self.base_url}/models")
        resp.raise_for_status()
        data = resp.json()
        return [m.get("id") for m in data.get("data", []) if m.get("id")]

    def check(self) -> tuple[bool, str]:
        """Probe the endpoint: reachable, model present, and able to generate.

        Returns ``(ok, message)`` with a human-readable diagnosis — used by the
        runner's ``--check-llm`` so a misconfigured/unreachable model fails fast
        with a clear message instead of mid-soak.
        """
        try:
            models = self.list_models()
        except Exception as exc:
            return False, (
                f"cannot reach LLM at {self.base_url}: {exc}\n"
                f"  - is the model server running and listening on the LAN?\n"
                f"  - is LLM_BASE_URL correct (host IP + /v1)?\n"
                f"  - is the port open in the server's firewall?"
            )
        note = "" if self.model in models else (
            f"  WARNING: model '{self.model}' not in available models {models}\n"
        )
        try:
            reply = self.complete("Reply with exactly: OK")
        except Exception as exc:
            return False, f"reachable at {self.base_url} but completion failed: {exc}"
        return True, (
            f"LLM reachable at {self.base_url}\n{note}"
            f"  available models: {models}\n"
            f"  sample reply: {reply!r}"
        )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


def make_complete_fn(llm_cfg: Any) -> Callable[[str], str]:
    """Build a ``complete(prompt) -> str`` callable from an ``LLMConfig``."""
    client = LocalLLMClient(llm_cfg.base_url, llm_cfg.reasoning_model)
    return client.complete
