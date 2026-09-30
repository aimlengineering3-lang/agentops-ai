import json

import httpx
import pytest

from agentops.errors import ProviderUnavailableError, RateLimitError
from agentops.search.serper import SerperProvider


def _provider(handler):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return SerperProvider("test-key", client=client)


def test_search_returns_results():
    def handler(request):
        assert request.headers["x-api-key"] == "test-key"
        body = json.loads(request.read())
        assert body["q"] == "enterprise ai platforms"
        return httpx.Response(
            200,
            json={
                "organic": [
                    {"title": "Example", "link": "https://example.com/a", "snippet": "some text"}
                ]
            },
        )

    results = _provider(handler).search("enterprise ai platforms")

    assert len(results) == 1
    assert results[0].url == "https://example.com/a"
    assert results[0].content == "some text"
    assert results[0].provider == "serper"


def test_results_are_capped_at_max_results():
    def handler(request):
        organic = [{"title": f"r{i}", "link": f"https://example.com/{i}"} for i in range(10)]
        return httpx.Response(200, json={"organic": organic})

    results = _provider(handler).search("x", max_results=3)

    assert len(results) == 3


def test_no_organic_key_returns_empty_list():
    def handler(request):
        return httpx.Response(200, json={})

    assert _provider(handler).search("x") == []


def test_429_raises_rate_limit_error():
    def handler(request):
        return httpx.Response(429, json={})

    with pytest.raises(RateLimitError):
        _provider(handler).search("x")


def test_500_raises_provider_unavailable_error():
    def handler(request):
        return httpx.Response(503, json={})

    with pytest.raises(ProviderUnavailableError):
        _provider(handler).search("x")
