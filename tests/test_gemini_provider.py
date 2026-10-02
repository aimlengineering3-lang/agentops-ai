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


def test_429_reads_retry_delay_from_the_body_retry_info():
    def handler(request):
        body = {
            "error": {
                "code": 429,
                "message": "You exceeded your current quota.",
                "details": [
                    {"@type": "type.googleapis.com/google.rpc.Help", "links": []},
                    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "15s"},
                ],
            }
        }
        return httpx.Response(429, json=body)

    with pytest.raises(RateLimitError) as exc_info:
        _provider(handler).generate([Message(role="user", content="hi")])

    assert exc_info.value.retry_after_s == 16.0  # 15s + 1s margin


def test_429_falls_back_to_retry_in_text_when_no_retry_info():
    def handler(request):
        body = {"error": {"code": 429, "message": "Quota exceeded.\nPlease retry in 15.5s."}}
        return httpx.Response(429, json=body)

    with pytest.raises(RateLimitError) as exc_info:
        _provider(handler).generate([Message(role="user", content="hi")])

    assert exc_info.value.retry_after_s == 16.5


def test_429_with_unusable_body_has_no_retry_after():
    def handler(request):
        return httpx.Response(429, text="not json")

    with pytest.raises(RateLimitError) as exc_info:
        _provider(handler).generate([Message(role="user", content="hi")])

    assert exc_info.value.retry_after_s is None


def test_429_on_a_daily_quota_benches_the_model_instead_of_a_short_retry():
    def handler(request):
        body = {
            "error": {
                "code": 429,
                "message": "You exceeded your current quota. Please retry in 12s.",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [
                            {"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}
                        ],
                    },
                    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "12s"},
                ],
            }
        }
        return httpx.Response(429, json=body)

    with pytest.raises(RateLimitError) as exc_info:
        _provider(handler).generate([Message(role="user", content="hi")])

    assert exc_info.value.retry_after_s == 3600.0  # not the misleading 12s


def test_provider_name_is_configurable_for_multiple_models():
    provider = GeminiProvider("k", "model-b", name="gemini:model-b")
    assert provider.name == "gemini:model-b"
    assert GeminiProvider("k", "model-a").name == "gemini"
