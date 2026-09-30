from collections import deque
from collections.abc import Iterable, Sequence

from agentops.llm import LLMResponse, Message, Usage
from agentops.search import SearchDepth, SearchResult


class FakeLLM:
    """Scripted LLM: returns the next scripted item; raises it if it is an Exception."""

    def __init__(
        self,
        script: Iterable[str | Exception],
        *,
        name: str = "fake-llm",
        model: str = "fake-model",
    ) -> None:
        self.name = name
        self.model = model
        self._script = deque(script)
        self.calls: list[list[Message]] = []

    def generate(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.2,
        max_output_tokens: int = 2048,
        json_output: bool = False,
        timeout_s: float = 30.0,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        if not self._script:
            # Fail loudly: silently returning "" would hide a wrong number of LLM calls.
            raise AssertionError("FakeLLM script exhausted: more LLM calls than the test expected")
        item = self._script.popleft()
        if isinstance(item, Exception):
            raise item
        tokens_in = sum(len(m.content) for m in messages) // 4  # rough estimate
        return LLMResponse(
            text=item,
            usage=Usage(tokens_in=tokens_in, tokens_out=len(item) // 4),
            provider=self.name,
            model=self.model,
            latency_ms=1,
        )


class FakeSearch:
    """Search provider returning canned results per query; can raise scripted failures first."""

    def __init__(
        self,
        results: dict[str, list[SearchResult]] | None = None,
        *,
        failures: Iterable[Exception] = (),
        name: str = "fake-search",
    ) -> None:
        self.name = name
        self._results = results or {}
        self._failures = deque(failures)
        self.calls: list[str] = []

    def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        depth: SearchDepth = "basic",
        timeout_s: float = 15.0,
    ) -> list[SearchResult]:
        self.calls.append(query)
        if self._failures:
            raise self._failures.popleft()
        return list(self._results.get(query, []))[:max_results]
