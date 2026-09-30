import json

import httpx
import pytest

from agentops.errors import InvalidRequestError, ProviderUnavailableError, RateLimitError
from agentops.llm import Message
from agentops.llm.gemini import GeminiProvider


def _provider(handler):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return GeminiProvider("test-key", "gemini-test-model", client=client)


def test_generate_returns_text_and_usage():
    def handler(request):
        assert request.url.params["key"] == "test-key"
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 4},
            },
        )

    response = _provider(handler).generate([Message(role="user", content="hi")])

    assert response.text == '{"ok": true}'
    assert response.usage.tokens_in == 10
    assert response.usage.tokens_out == 4
    assert response.provider == "gemini"


def test_system_message_becomes_system_instruction():
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.read())
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": "ok"}]}}], "usageMetadata": {}},
        )

    _provider(handler).generate(
        [Message(role="system", content="be terse"), Message(role="user", content="hi")]
    )

    body = captured["body"]
    assert body["systemInstruction"]["parts"][0]["text"] == "be terse"
    assert body["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]


def test_assistant_role_maps_to_model():
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.read())
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": "ok"}]}}], "usageMetadata": {}},
        )

    _provider(handler).generate(
        [Message(role="user", content="hi"), Message(role="assistant", content="hello")]
    )

    roles = [c["role"] for c in captured["body"]["contents"]]
    assert roles == ["user", "model"]


def test_429_raises_rate_limit_error_with_retry_after():
    def handler(request):
        return httpx.Response(429, json={}, headers={"Retry-After": "2"})

    with pytest.raises(RateLimitError) as exc_info:
        _provider(handler).generate([Message(role="user", content="hi")])
    assert exc_info.value.retry_after_s == 2.0


def test_500_raises_provider_unavailable_error():
    def handler(request):
        return httpx.Response(500, json={})

    with pytest.raises(ProviderUnavailableError):
        _provider(handler).generate([Message(role="user", content="hi")])


def test_400_raises_invalid_request_error():
    def handler(request):
        return httpx.Response(400, json={"error": "bad"})

    with pytest.raises(InvalidRequestError):
        _provider(handler).generate([Message(role="user", content="hi")])


def test_no_candidates_raises_invalid_request_error():
    def handler(request):
        return httpx.Response(
            200,
            json={"candidates": [], "promptFeedback": {"blockReason": "SAFETY"}},
        )

    with pytest.raises(InvalidRequestError, match="SAFETY"):
        _provider(handler).generate([Message(role="user", content="hi")])
