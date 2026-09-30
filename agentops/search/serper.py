import httpx

from agentops.http_errors import classify_http_status, classify_transport_error, parse_retry_after

from .base import SearchDepth, SearchResult

_BASE_URL = "https://google.serper.dev/search"


class SerperProvider:
    """Serper's raw Google Search API, called directly over REST via httpx.

    Serper has no notion of "depth" (it always returns Google's organic results);
    `depth` is accepted for SearchProvider interface compatibility but unused here.
    """

    name = "serper"

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
        try:
            resp = self._client.post(
                _BASE_URL,
                headers={"X-API-KEY": self._api_key, "Content-Type": "application/json"},
                json={"q": query, "num": max_results},
                timeout=timeout_s,
            )
        except httpx.HTTPError as exc:
            raise classify_transport_error(exc, provider=self.name) from exc

        error = classify_http_status(
            resp.status_code,
            provider=self.name,
            retry_after_s=parse_retry_after(resp.headers.get("Retry-After")),
        )
        if error is not None:
            raise error

        organic = resp.json().get("organic") or []
        return [
            SearchResult(
                title=r.get("title", ""),
                url=r["link"],
                content=r.get("snippet", ""),
                score=None,
                provider=self.name,
            )
            for r in organic[:max_results]
        ]
