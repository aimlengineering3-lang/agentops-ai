import json

from agentops.agents.researcher import LLMResearcher
from agentops.contracts import EventType, RunState, Subtask
from agentops.errors import ProviderTimeoutError
from agentops.orchestrator import BudgetGuard, EventBus, RunContext
from agentops.search import InMemorySearchCache, SearchResult
from agentops.testing.fakes import FakeLLM, FakeSearch

OBJECTIVE = "Compare three approaches for building an enterprise AI platform"

QUERIES_JSON = json.dumps({"queries": ["enterprise ai platform 2026", "ai platform comparison"]})


def _subtask() -> Subtask:
    return Subtask(
        id="s1",
        description="Research current enterprise AI platform options",
        acceptance_criteria=["at least two independent sources"],
    )


def _context(state: RunState) -> RunContext:
    bus = EventBus()
    return RunContext(bus=bus, guard=BudgetGuard(state, bus))


def _hits(provider: str, n: int, prefix: str = "a") -> list[SearchResult]:
    return [
        SearchResult(
            title=f"t{i}", url=f"https://{prefix}{i}.com", content=f"c{i}", provider=provider
        )
        for i in range(n)
    ]


def _extraction_for(n: int, *, offset: int = 0) -> str:
    """An extraction response that accepts source_index 0..n-1 as-is."""
    items = [
        {
            "source_index": i,
            "extracted_claim": f"Claim from result {i + offset}",
            "snippet": f"snippet {i + offset}",
        }
        for i in range(n)
    ]
    return json.dumps({"items": items})


def test_researcher_collects_evidence_across_queries():
    state = RunState(objective=OBJECTIVE)
    ctx = _context(state)
    subtask = _subtask()
    search = FakeSearch(
        {
            "enterprise ai platform 2026": _hits("fake", 2, "a"),
            "ai platform comparison": _hits("fake", 2, "b"),
        }
    )
    llm = FakeLLM([QUERIES_JSON, _extraction_for(4)])
    researcher = LLMResearcher(llm, search)

    researcher.research(state, ctx, subtask)

    assert len(state.evidence_for("s1")) == 4
    assert state.usage.tool_calls == 2  # one real search call per query
    assert state.usage.llm_calls == 2  # queries + extraction
    assert any(e.event_type == EventType.TOOL_RESULT for e in state.events)
    first_evidence = state.evidence_for("s1")[0]
    assert first_evidence.extracted_claim.startswith("Claim from result")


def test_researcher_dedupes_same_url_across_queries():
    state = RunState(objective=OBJECTIVE)
    ctx = _context(state)
    subtask = _subtask()
    dup = _hits("fake", 1, "same")
    search = FakeSearch({"enterprise ai platform 2026": dup, "ai platform comparison": dup})
    llm = FakeLLM([QUERIES_JSON, _extraction_for(1)])
    researcher = LLMResearcher(llm, search)

    researcher.research(state, ctx, subtask)

    assert len(state.evidence_for("s1")) == 1


def test_researcher_continues_after_one_query_fails():
    state = RunState(objective=OBJECTIVE)
    ctx = _context(state)
    subtask = _subtask()
    search = FakeSearch(
        {"ai platform comparison": _hits("fake", 2, "b")},
        failures=[ProviderTimeoutError("t", provider="fake")],
    )
    llm = FakeLLM([QUERIES_JSON, _extraction_for(2)])
    researcher = LLMResearcher(llm, search)

    researcher.research(state, ctx, subtask)

    assert len(state.evidence_for("s1")) == 2
    failed_events = [
        e for e in state.events if e.event_type == EventType.TOOL_RESULT and e.success is False
    ]
    assert len(failed_events) == 1


def test_researcher_respects_max_evidence_per_subtask():
    state = RunState(objective=OBJECTIVE)
    ctx = _context(state)
    subtask = _subtask()
    search = FakeSearch(
        {
            "enterprise ai platform 2026": _hits("fake", 4, "a"),
            "ai platform comparison": _hits("fake", 4, "b"),
        }
    )
    llm = FakeLLM([QUERIES_JSON, _extraction_for(8)])
    researcher = LLMResearcher(llm, search, max_evidence_per_subtask=3)

    researcher.research(state, ctx, subtask)

    assert len(state.evidence_for("s1")) == 3


def test_cache_hit_does_not_spend_tool_call_budget():
    state = RunState(objective=OBJECTIVE)
    ctx = _context(state)
    subtask = _subtask()
    cache = InMemorySearchCache()
    search = FakeSearch({"ai platform comparison": _hits("fake", 1, "b")})
    llm = FakeLLM([QUERIES_JSON, _extraction_for(2)])
    researcher = LLMResearcher(llm, search, cache=cache)

    # Pre-warm the cache for the first query, as if a previous run already searched it.
    from agentops.search import cache_key

    key = cache_key("enterprise ai platform 2026", depth="basic", max_results=4)
    cache.set(key, _hits("fake", 1, "a"))

    researcher.research(state, ctx, subtask)

    # Only the second query ("ai platform comparison") should hit the real provider.
    assert state.usage.tool_calls == 1
    cache_hit_events = [e for e in state.events if e.data.get("cached") is True]
    assert len(cache_hit_events) == 1


def test_extraction_failure_falls_back_to_raw_results():
    state = RunState(objective=OBJECTIVE)
    ctx = _context(state)
    subtask = _subtask()
    search = FakeSearch(
        {
            "enterprise ai platform 2026": _hits("fake", 1, "a"),
            "ai platform comparison": _hits("fake", 1, "b"),
        }
    )
    # generate_structured makes one repair attempt, so extraction needs two bad replies
    # before it gives up and LLMResearcher falls back to the raw results.
    llm = FakeLLM([QUERIES_JSON, "not json", "still not json"])
    researcher = LLMResearcher(llm, search)

    researcher.research(state, ctx, subtask)

    # extraction failed twice (one repair attempt) -> fallback keeps the raw results
    assert len(state.evidence_for("s1")) == 2
    assert any("extraction failed" in note for note in state.limitations)


def test_no_search_results_records_a_limitation_and_skips_extraction():
    state = RunState(objective=OBJECTIVE)
    ctx = _context(state)
    subtask = _subtask()
    search = FakeSearch({})  # every query returns nothing
    llm = FakeLLM([QUERIES_JSON])  # no extraction call scripted -> proves it is skipped

    researcher = LLMResearcher(llm, search)
    researcher.research(state, ctx, subtask)

    assert state.evidence_for("s1") == []
    assert any("no search results" in note for note in state.limitations)
