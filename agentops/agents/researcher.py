from pydantic import BaseModel, Field

from agentops.contracts import AgentName, EventType, RunState, Subtask
from agentops.errors import MalformedOutputError, ProviderError
from agentops.llm import Generator, Message
from agentops.orchestrator import RunContext
from agentops.search import SearchCache, SearchDepth, SearchProvider, SearchResult, cache_key

from .evidence import evidence_from_claim, evidence_from_search_result
from .llm_call import call_llm_structured
from .tool_call import call_search

_QUERY_SYSTEM_PROMPT = """\
You are the Researcher in an autonomous research and analysis agent. Given one
subtask, propose 2 to 4 distinct, high-quality web search queries that would
find verifiable, up-to-date evidence for it. Return ONLY a JSON object
matching this schema -- no prose, no markdown fences:

{"queries": ["<query 1>", "<query 2>", ...]}

Rules:
- 2 to 4 queries, each a short search-engine-style query, not a full sentence.
- Queries must be genuinely different angles on the subtask, not near-duplicates.
- Do not invent a source or a specific fact; only propose what to search for.
"""

_EXTRACTION_SYSTEM_PROMPT = """\
You are the Researcher in an autonomous research and analysis agent. You are
given a numbered list of raw web search results for one subtask. For each
result that is actually relevant, extract the claim it supports in your own
words. Return ONLY JSON, no prose:

{
  "items": [
    {"source_index": <int, index from the list below>,
     "extracted_claim": "<1-2 sentence claim in your own words, grounded in
       the result's content, not invented>",
     "snippet": "<short supporting excerpt, your own words is fine>"}
  ]
}

Rules:
- source_index MUST be an index from the list below. Never invent one.
- Skip results that are irrelevant, duplicate, or too thin to support a claim.
- Never state something the result does not support.
"""


class SearchQueries(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=4)


class _ExtractedItem(BaseModel):
    source_index: int = Field(ge=0)
    extracted_claim: str = Field(min_length=1, max_length=1000)
    snippet: str = Field(default="", max_length=2000)


class _ExtractionResult(BaseModel):
    items: list[_ExtractedItem] = Field(default_factory=list)


def _format_raw_results(pairs: list[tuple[SearchResult, str]]) -> str:
    lines = [
        f"[{i}] query={query!r} title={result.title!r} url={result.url} content={result.content!r}"
        for i, (result, query) in enumerate(pairs)
    ]
    return "\n".join(lines) or "(no results)"


class LLMResearcher:
    """Researcher agent, two LLM calls per subtask:

    1. propose search queries for the subtask (SearchQueries)
    2. extract a grounded claim, in the model's own words, from the raw results
       (source_index -> Evidence, same never-invent-an-id grounding the Analyst
       and Finalizer use for their own citations)

    A cache hit skips the real search call and its tool-call budget; a failed
    extraction call falls back to a deterministic, verbatim conversion so a bad
    LLM response never loses evidence outright.
    """

    def __init__(
        self,
        generator: Generator,
        search_router: SearchProvider,
        *,
        cache: SearchCache | None = None,
        results_per_query: int = 4,
        max_evidence_per_subtask: int = 6,
        max_raw_results_for_extraction: int = 12,
        depth: SearchDepth = "basic",
        temperature: float = 0.2,
    ) -> None:
        self._generator = generator
        self._router = search_router
        self._cache = cache
        self._results_per_query = results_per_query
        self._max_evidence = max_evidence_per_subtask
        self._max_raw = max_raw_results_for_extraction
        self._depth = depth
        self._temperature = temperature

    def research(self, state: RunState, ctx: RunContext, subtask: Subtask) -> None:
        queries = self._make_queries(state, ctx, subtask)
        pairs = self._fetch_raw_results(state, ctx, subtask, queries)
        self._extract_evidence(state, ctx, subtask, pairs)

    # -- step 1: queries -------------------------------------------------------------------

    def _make_queries(self, state: RunState, ctx: RunContext, subtask: Subtask) -> list[str]:
        messages = [
            Message(role="system", content=_QUERY_SYSTEM_PROMPT),
            Message(
                role="user",
                content=(
                    f"Subtask: {subtask.description}\n"
                    f"Acceptance criteria: {subtask.acceptance_criteria}"
                ),
            ),
        ]
        parsed = call_llm_structured(
            state,
            ctx,
            AgentName.RESEARCHER,
            self._generator,
            messages,
            SearchQueries,
            temperature=self._temperature,
        )
        return parsed.queries

    # -- step 2: search, cache-aware ---------------------------------------------------------

    def _fetch_raw_results(
        self, state: RunState, ctx: RunContext, subtask: Subtask, queries: list[str]
    ) -> list[tuple[SearchResult, str]]:
        """Raw (result, query) pairs, deduped by URL across queries."""
        seen_urls: set[str] = set()
        pairs: list[tuple[SearchResult, str]] = []
        for query in queries:
            for result in self._search_one(state, ctx, subtask, query):
                if result.url in seen_urls:
                    continue
                seen_urls.add(result.url)
                pairs.append((result, query))
        return pairs

    def _search_one(
        self, state: RunState, ctx: RunContext, subtask: Subtask, query: str
    ) -> list[SearchResult]:
        key = cache_key(query, depth=self._depth, max_results=self._results_per_query)
        if self._cache is not None:
            cached = self._cache.get(key)
            if cached is not None:
                ctx.bus.emit(
                    state,
                    AgentName.RESEARCHER,
                    EventType.TOOL_CALL,
                    f"Cache hit: {query!r}",
                    subtask_id=subtask.id,
                    success=True,
                    data={"query": query, "cached": True},
                )
                return cached

        try:
            results = call_search(
                state,
                ctx,
                AgentName.RESEARCHER,
                self._router,
                query,
                subtask_id=subtask.id,
                max_results=self._results_per_query,
                depth=self._depth,
                timeout_s=state.limits.search_timeout_s,
            )
        except ProviderError:
            return []  # call_search already traced the failure; move on to the next query

        if self._cache is not None:
            self._cache.set(key, results)
        return results

    # -- step 3: extraction ------------------------------------------------------------------

    def _extract_evidence(
        self,
        state: RunState,
        ctx: RunContext,
        subtask: Subtask,
        pairs: list[tuple[SearchResult, str]],
    ) -> None:
        if not pairs:
            state.limitations.append(f"Subtask {subtask.id}: no search results to extract from")
            return

        limited = pairs[: self._max_raw]
        messages = [
            Message(role="system", content=_EXTRACTION_SYSTEM_PROMPT),
            Message(role="user", content=_format_raw_results(limited)),
        ]
        try:
            parsed = call_llm_structured(
                state,
                ctx,
                AgentName.RESEARCHER,
                self._generator,
                messages,
                _ExtractionResult,
                temperature=self._temperature,
            )
        except (MalformedOutputError, ProviderError) as exc:
            state.limitations.append(
                f"Subtask {subtask.id}: evidence extraction failed, using raw results ({exc})"
            )
            self._add_raw_fallback(state, subtask, limited)
            return

        collected = 0
        for item in parsed.items:
            if collected >= self._max_evidence:
                break
            if not 0 <= item.source_index < len(limited):
                continue  # model referenced a result we never showed it
            result, query = limited[item.source_index]
            evidence = evidence_from_claim(
                result,
                subtask_id=subtask.id,
                retrieval_query=query,
                extracted_claim=item.extracted_claim,
                snippet=item.snippet,
            )
            state.add_evidence(evidence)
            collected += 1

    def _add_raw_fallback(
        self, state: RunState, subtask: Subtask, pairs: list[tuple[SearchResult, str]]
    ) -> None:
        for result, query in pairs[: self._max_evidence]:
            evidence = evidence_from_search_result(
                result, subtask_id=subtask.id, retrieval_query=query
            )
            state.add_evidence(evidence)
