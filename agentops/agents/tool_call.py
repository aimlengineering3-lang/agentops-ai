from agentops.contracts import AgentName, EventType, RunState
from agentops.orchestrator import RunContext
from agentops.search import SearchDepth, SearchProvider, SearchResult


def call_search(
    state: RunState,
    ctx: RunContext,
    agent: AgentName,
    provider: SearchProvider,
    query: str,
    *,
    subtask_id: str,
    max_results: int = 5,
    depth: SearchDepth = "basic",
    timeout_s: float = 15.0,
) -> list[SearchResult]:
    """The one place every search-calling agent goes through: budget, then call, then trace.

    Mirrors call_llm_structured: guarantees ctx.guard.before_tool_call() is never skipped,
    and every call leaves a TOOL_CALL/TOOL_RESULT pair in the trace, success or failure.
    Errors are NOT caught here -- this function only does bookkeeping; deciding what a
    failure means for the subtask (retry, skip, limitation) is the calling agent's job.
    """
    ctx.guard.before_tool_call()
    ctx.bus.emit(
        state,
        agent,
        EventType.TOOL_CALL,
        f"search: {query!r}",
        subtask_id=subtask_id,
        data={"query": query, "depth": depth},
    )
    try:
        results = provider.search(query, max_results=max_results, depth=depth, timeout_s=timeout_s)
    except Exception:
        ctx.bus.emit(
            state,
            agent,
            EventType.TOOL_RESULT,
            f"search failed: {query!r}",
            subtask_id=subtask_id,
            success=False,
        )
        raise
    ctx.bus.emit(
        state,
        agent,
        EventType.TOOL_RESULT,
        f"search returned {len(results)} result(s): {query!r}",
        subtask_id=subtask_id,
        success=True,
        data={"count": len(results)},
    )
    return results
