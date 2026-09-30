import pytest

from agentops.errors import InvalidRequestError, ProviderTimeoutError, RateLimitError
from agentops.search import SearchResult, SearchRouter
from agentops.testing.fakes import FakeSearch


def _hit(url: str = "https://a.com") -> SearchResult:
    return SearchResult(title="t", url=url, content="c", provider="fake")


def test_uses_primary_when_it_succeeds():
    primary = FakeSearch({"q": [_hit()]}, name="primary")
    fallback = FakeSearch({"q": [_hit()]}, name="fallback")
    router = SearchRouter([primary, fallback])

    results = router.search("q")

    assert results[0].provider == "fake"
    assert fallback.calls == []


def test_fails_over_on_retryable_error():
    primary = FakeSearch(failures=[RateLimitError("slow down", provider="primary")], name="primary")
    fallback = FakeSearch({"q": [_hit()]}, name="fallback")
    router = SearchRouter([primary, fallback])

    results = router.search("q")

    assert len(results) == 1
    assert fallback.calls == ["q"]


def test_does_not_fail_over_on_non_retryable_error():
    primary = FakeSearch(failures=[InvalidRequestError("bad", provider="primary")], name="primary")
    fallback = FakeSearch({"q": [_hit()]}, name="fallback")
    router = SearchRouter([primary, fallback])

    with pytest.raises(InvalidRequestError):
        router.search("q")
    assert fallback.calls == []


def test_raises_after_last_provider_also_fails():
    primary = FakeSearch(failures=[ProviderTimeoutError("t", provider="primary")], name="primary")
    fallback = FakeSearch(
        failures=[ProviderTimeoutError("t", provider="fallback")], name="fallback"
    )
    router = SearchRouter([primary, fallback])

    with pytest.raises(ProviderTimeoutError):
        router.search("q")


def test_router_requires_at_least_one_provider():
    with pytest.raises(ValueError):
        SearchRouter([])
