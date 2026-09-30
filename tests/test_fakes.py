import pytest

from agentops.errors import ProviderTimeoutError, RateLimitError
from agentops.llm import LLMProvider, Message
from agentops.search import SearchProvider, SearchResult
from agentops.testing import FakeLLM, FakeSearch

USER = [Message(role="user", content="hello")]


def make_result(url: str = "https://a.com") -> SearchResult:
    return SearchResult(title="t", url=url, content="c", provider="fake-search")


def test_fake_llm_returns_script_in_order_and_records_calls():
    llm = FakeLLM(["one", "two"])
    assert llm.generate(USER).text == "one"
    assert llm.generate(USER).text == "two"
    assert len(llm.calls) == 2


def test_fake_llm_raises_scripted_exception_then_recovers():
    llm = FakeLLM([RateLimitError("slow down", provider="fake"), "ok"])
    with pytest.raises(RateLimitError):
        llm.generate(USER)
    assert llm.generate(USER).text == "ok"


def test_fake_llm_fails_loudly_when_script_is_exhausted():
    with pytest.raises(AssertionError, match="exhausted"):
        FakeLLM([]).generate(USER)


def test_fakes_satisfy_provider_protocols():
    assert isinstance(FakeLLM([]), LLMProvider)
    assert isinstance(FakeSearch(), SearchProvider)


def test_fake_search_returns_canned_results_and_empty_for_unknown_query():
    hit = make_result()
    search = FakeSearch({"q1": [hit]})
    assert search.search("q1") == [hit]
    assert search.search("other") == []
    assert search.calls == ["q1", "other"]


def test_fake_search_respects_max_results():
    hits = [make_result(f"https://a.com/{i}") for i in range(5)]
    assert len(FakeSearch({"q": hits}).search("q", max_results=2)) == 2


def test_fake_search_scripted_failure_then_success():
    hit = make_result()
    search = FakeSearch({"q": [hit]}, failures=[ProviderTimeoutError("t", provider="fake-search")])
    with pytest.raises(ProviderTimeoutError):
        search.search("q")
    assert search.search("q") == [hit]
