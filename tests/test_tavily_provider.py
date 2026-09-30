import json

import httpx
import pytest

from agentops.errors import InvalidRequestError, RateLimitError
from agentops.search.tavily import TavilyProvider


def _provider(handler):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return TavilyProvider("test-key", client=client)


def test_search_returns_results():
    def handler(request):
        body = json.loads(request.read())
        assert body["api_key"] == "test-key"
        assert body["query"] == "enterprise ai platforms"
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Example",
                        "url": "https://example.com/a",
                        "content": "some text",
                        "score": 0.9,
                    }
                ]
            },
        )

    results = _provider(handler).search("enterprise ai platforms")

    assert len(results) == 1
    assert results[0].url == "https://example.com/a"
    assert results[0].provider == "tavily"
    assert results[0].score == 0.9


def test_advanced_depth_is_sent_through():
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.read())
        return httpx.Response(200, json={"results": []})

    _provider(handler).search("x", depth="advanced")

    assert captured["body"]["search_depth"] == "advanced"


def test_no_results_returns_empty_list():
    def handler(request):
        return httpx.Response(200, json={"results": []})

    assert _provider(handler).search("obscure query") == []


def test_429_raises_rate_limit_error():
    def handler(request):
        return httpx.Response(429, json={})

    with pytest.raises(RateLimitError):
        _provider(handler).search("x")


def test_400_raises_invalid_request_error():
    def handler(request):
        return httpx.Response(400, json={})

    with pytest.raises(InvalidRequestError):
        _provider(handler).search("x")
