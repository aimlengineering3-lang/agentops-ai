from typing import Protocol

from agentops.contracts import Critique, Plan, RunState, Subtask

from .context import RunContext


class Planner(Protocol):
    def plan(self, state: RunState, ctx: RunContext) -> Plan: ...


class Researcher(Protocol):
    """Adds Evidence to `state` for one subtask."""

    def research(self, state: RunState, ctx: RunContext, subtask: Subtask) -> None: ...


class Analyst(Protocol):
    """Adds Findings to `state` for one subtask."""

    def analyze(self, state: RunState, ctx: RunContext, subtask: Subtask) -> None: ...


class Critic(Protocol):
    def critique(self, state: RunState, ctx: RunContext, *, iteration: int) -> Critique: ...


class Finalizer(Protocol):
    def finalize(self, state: RunState, ctx: RunContext) -> None: ...
