from typing import Any

import httpx

from agentops.http_errors import classify_http_status, classify_transport_error, parse_retry_after

from .base import SearchDepth, SearchResult

_BASE_URL = "https://api.tavily.com/search"


class TavilyProvider:
    """Tavily's AI-native search API, called directly over REST via httpx."""

    name = "tavily"

    def __init__(self, api_key: str, *, client: httpx.Client | None = None) -> None:
        self._api_key = api_key
        self._client = client or httpx.Client()

    def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        depth: SearchDepth = "basic",
        timeout_s: float = 15.0,
    ) -> list[SearchResult]:
        body: dict[str, Any] = {
            "api_key": self._api_key,
            "query": query,
            "search_depth": depth,
            "max_results": max_results,
        }
        try:
            resp = self._client.post(_BASE_URL, json=body, timeout=timeout_s)
        except httpx.HTTPError as exc:
            raise classify_transport_error(exc, provider=self.name) from exc

        error = classify_http_status(
            resp.status_code,
            provider=self.name,
            retry_after_s=parse_retry_after(resp.headers.get("Retry-After")),
        )
        if error is not None:
            raise error

        results = resp.json().get("results") or []
        return [
            SearchResult(
                title=r.get("title", ""),
                url=r["url"],
                content=r.get("content", ""),
                score=r.get("score"),
                provider=self.name,
            )
            for r in results
        ]
