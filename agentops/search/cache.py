from typing import Protocol, runtime_checkable

from .base import SearchResult


@runtime_checkable
class SearchCache(Protocol):
    def get(self, key: str) -> list[SearchResult] | None: ...
    def set(self, key: str, results: list[SearchResult]) -> None: ...


class InMemorySearchCache:
    """Per-process cache, keyed by normalized (depth, max_results, query).

    Good enough for one run and one demo session. Phase 6 swaps in a
    Postgres-backed cache behind the same SearchCache protocol, so callers
    (LLMResearcher) never change.
    """

    def __init__(self) -> None:
        self._store: dict[str, list[SearchResult]] = {}

    def get(self, key: str) -> list[SearchResult] | None:
        return self._store.get(key)

    def set(self, key: str, results: list[SearchResult]) -> None:
        self._store[key] = list(results)


def cache_key(query: str, *, depth: str, max_results: int) -> str:
    return f"{depth}:{max_results}:{query.strip().lower()}"
