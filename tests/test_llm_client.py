import json

import pytest

httpx = pytest.importorskip("httpx")

from bot.matching.llm_client import LocalLLMClient


def test_complete_parses_openai_style_response():
    captured = {}

    def handler(request: "httpx.Request") -> "httpx.Response":
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "hello world"}}]}
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    llm = LocalLLMClient("http://localhost:8000/v1", "qwen2.5-instruct", client=client)

    assert llm.complete("say hi") == "hello world"
    assert captured["url"].endswith("/v1/chat/completions")
    assert captured["body"]["model"] == "qwen2.5-instruct"
    assert captured["body"]["messages"][0]["content"] == "say hi"


def test_complete_raises_on_http_error():
    client = httpx.Client(
        transport=httpx.MockTransport(lambda req: httpx.Response(500, text="boom"))
    )
    llm = LocalLLMClient("http://localhost:8000/v1", "m", client=client)
    with pytest.raises(httpx.HTTPStatusError):
        llm.complete("hi")


def _routed(models):
    def handler(request: "httpx.Request") -> "httpx.Response":
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": m} for m in models]})
        return httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}]})
    return httpx.MockTransport(handler)


def test_check_ok_when_reachable_and_model_present():
    client = httpx.Client(transport=_routed(["qwen2.5:14b-instruct"]))
    llm = LocalLLMClient("http://win-pc:11434/v1", "qwen2.5:14b-instruct", client=client)
    ok, msg = llm.check()
    assert ok is True
    assert "reachable" in msg and "OK" in msg


def test_check_warns_when_model_missing_but_still_ok():
    client = httpx.Client(transport=_routed(["some-other-model"]))
    llm = LocalLLMClient("http://win-pc:11434/v1", "qwen2.5:14b-instruct", client=client)
    ok, msg = llm.check()
    assert ok is True  # endpoint works; just a model-name warning
    assert "WARNING" in msg


def test_check_fails_when_unreachable():
    def boom(request):
        raise httpx.ConnectError("connection refused")
    client = httpx.Client(transport=httpx.MockTransport(boom))
    llm = LocalLLMClient("http://win-pc:11434/v1", "m", client=client)
    ok, msg = llm.check()
    assert ok is False
    assert "cannot reach LLM" in msg
