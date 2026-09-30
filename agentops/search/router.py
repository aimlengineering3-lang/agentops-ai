import time
from collections.abc import Callable, Sequence

from agentops.errors import ProviderError

from .base import SearchDepth, SearchProvider, SearchResult


class SearchRouter:
    """Tries search providers in order; fails over to the next one on a retryable error.

    Mirrors LLMRouter on purpose: the same failover shape for LLMs and search
    means one pattern to explain, not two.
    """

    def __init__(
        self,
        providers: Sequence[SearchProvider],
        *,
        sleep: Callable[[float], None] = time.sleep,
        backoff_s: float = 0.0,
    ) -> None:
        if not providers:
            raise ValueError("SearchRouter needs at least one provider")
        self._providers = list(providers)
        self._sleep = sleep
        self._backoff_s = backoff_s

    def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        depth: SearchDepth = "basic",
        timeout_s: float = 15.0,
    ) -> list[SearchResult]:
        for index, provider in enumerate(self._providers):
            is_last = index == len(self._providers) - 1
            try:
                return provider.search(
                    query, max_results=max_results, depth=depth, timeout_s=timeout_s
                )
            except ProviderError as exc:
                if not exc.retryable or is_last:
                    raise
                if self._backoff_s:
                    self._sleep(self._backoff_s)
        raise AssertionError("unreachable: loop above always returns or raises")
