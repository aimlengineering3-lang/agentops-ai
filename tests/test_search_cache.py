from agentops.search import InMemorySearchCache, SearchResult, cache_key


def test_cache_key_normalises_query_case_and_whitespace():
    assert cache_key("  Enterprise AI  ", depth="basic", max_results=4) == cache_key(
        "enterprise ai", depth="basic", max_results=4
    )


def test_cache_key_differs_by_depth_and_max_results():
    a = cache_key("q", depth="basic", max_results=4)
    b = cache_key("q", depth="advanced", max_results=4)
    c = cache_key("q", depth="basic", max_results=5)
    assert len({a, b, c}) == 3


def test_miss_then_hit():
    cache = InMemorySearchCache()
    key = cache_key("q", depth="basic", max_results=4)
    assert cache.get(key) is None

    hit = SearchResult(title="t", url="https://a.com", content="c", provider="tavily")
    cache.set(key, [hit])

    assert cache.get(key) == [hit]
