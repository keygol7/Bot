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
