from collections.abc import Sequence
from typing import TypeVar

from pydantic import BaseModel

from agentops.contracts import AgentName, EventType, RunState
from agentops.errors import MalformedOutputError
from agentops.llm import Generator, Message, generate_structured
from agentops.orchestrator import RunContext

SchemaT = TypeVar("SchemaT", bound=BaseModel)


def call_llm_structured(
    state: RunState,
    ctx: RunContext,
    agent: AgentName,
    generator: Generator,
    messages: Sequence[Message],
    schema: type[SchemaT],
    *,
    temperature: float = 0.2,
    max_output_tokens: int = 2048,
    timeout_s: float = 30.0,
) -> SchemaT:
    """The one place every LLM-calling agent goes through: budget, then call, then trace.

    Guarantees ctx.guard.before_llm_call() and ctx.guard.record_tokens() are never
    skipped by a forgetful agent implementation, and that every LLM call leaves an
    EventType.LLM_CALL entry in the trace, whether it succeeded or not.
    """
    ctx.guard.before_llm_call()
    try:
        parsed, response = generate_structured(
            generator,
            messages,
            schema,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            timeout_s=timeout_s,
        )
    except MalformedOutputError:
        ctx.bus.emit(
            state,
            agent,
            EventType.LLM_CALL,
            f"{schema.__name__} generation failed schema validation",
            success=False,
        )
        raise

    ctx.guard.record_tokens(response.usage)
    ctx.bus.emit(
        state,
        agent,
        EventType.LLM_CALL,
        f"{schema.__name__} generated via {response.provider}",
        success=True,
        data={"provider": response.provider, "model": response.model},
    )
    return parsed
