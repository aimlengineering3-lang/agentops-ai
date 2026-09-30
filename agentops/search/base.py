from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel

SearchDepth = Literal["basic", "advanced"]


class SearchResult(BaseModel):
    """Raw, untrusted result from a search provider.

    Deliberately permissive: real pages have long titles and content. Limits are
    enforced later, when a result is converted into an Evidence record.
    """

    title: str
    url: str
    content: str = ""
    score: float | None = None
    provider: str


@runtime_checkable
class SearchProvider(Protocol):
    name: str

    def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        depth: SearchDepth = "basic",
        timeout_s: float = 15.0,
    ) -> list[SearchResult]: ...
