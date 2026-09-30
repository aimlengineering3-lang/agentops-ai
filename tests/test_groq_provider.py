import json

import httpx
import pytest

from agentops.errors import InvalidRequestError, ProviderTimeoutError, RateLimitError
from agentops.llm import Message
from agentops.llm.groq import GroqProvider


def _provider(handler):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return GroqProvider("test-key", "groq-test-model", client=client)


def test_generate_returns_text_and_usage():
    def handler(request):
        assert request.headers["authorization"] == "Bearer test-key"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "hello"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
        )

    response = _provider(handler).generate([Message(role="user", content="hi")])

    assert response.text == "hello"
    assert response.usage.tokens_in == 5
    assert response.usage.tokens_out == 2
    assert response.provider == "groq"


def test_json_output_sets_response_format():
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.read())
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}], "usage": {}})

    _provider(handler).generate([Message(role="user", content="hi")], json_output=True)

    assert captured["body"]["response_format"] == {"type": "json_object"}


def test_429_raises_rate_limit_error():
    def handler(request):
        return httpx.Response(429, json={})

    with pytest.raises(RateLimitError):
        _provider(handler).generate([Message(role="user", content="hi")])


def test_timeout_raises_provider_timeout_error():
    def handler(request):
        raise httpx.ReadTimeout("timed out", request=request)

    with pytest.raises(ProviderTimeoutError):
        _provider(handler).generate([Message(role="user", content="hi")])


def test_empty_choices_raises_invalid_request_error():
    def handler(request):
        return httpx.Response(200, json={"choices": [], "usage": {}})

    with pytest.raises(InvalidRequestError):
        _provider(handler).generate([Message(role="user", content="hi")])
