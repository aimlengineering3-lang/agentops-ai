from agentops.contracts import AgentName, Plan, RunState
from agentops.llm import Generator, Message
from agentops.orchestrator import RunContext

from .llm_call import call_llm_structured

_SYSTEM_PROMPT = """\
You are the Planner in an autonomous research and analysis agent. Given an
objective, break it into 2 to 6 independent, verifiable subtasks and return
ONLY a JSON object matching this schema -- no prose, no markdown fences:

{
  "subtasks": [
    {
      "id": "<short unique slug, e.g. 's1'>",
      "description": "<5-500 chars: what this subtask must investigate>",
      "acceptance_criteria": ["<at least one concrete, checkable criterion>"],
      "depends_on": ["<ids of subtasks that must finish first, or [] if none>"]
    }
  ]
}

Rules:
- 2 to 6 subtasks. Fewer, well-scoped subtasks beat many shallow ones.
- Every acceptance_criteria entry must be something a Critic can check later
  (e.g. "at least two independent sources"), never a vague goal like "do
  good research".
- depends_on must only reference other subtask ids in this same plan. Most
  objectives need no dependencies -- use them only when a subtask genuinely
  cannot start before another finishes.
- Do not include "status" or "attempts" in your output; the system sets those.

Example, for the objective "Compare web frameworks for a small team":
{
  "subtasks": [
    {
      "id": "s1",
      "description": "Research FastAPI, Django and Flask: learning curve, ecosystem",
      "acceptance_criteria": ["at least two independent sources per framework"],
      "depends_on": []
    },
    {
      "id": "s2",
      "description": "Compare the three frameworks for a small team's needs",
      "acceptance_criteria": ["explicit trade-off comparison across all three"],
      "depends_on": ["s1"]
    }
  ]
}
"""


class LLMPlanner:
    """Turns an objective into a Plan by asking an LLM, via the shared structured-call helper."""

    def __init__(self, generator: Generator, *, temperature: float = 0.2) -> None:
        self._generator = generator
        self._temperature = temperature

    def plan(self, state: RunState, ctx: RunContext) -> Plan:
        messages = [
            Message(role="system", content=_SYSTEM_PROMPT),
            Message(role="user", content=f"Objective: {state.objective}"),
        ]
        return call_llm_structured(
            state,
            ctx,
            AgentName.PLANNER,
            self._generator,
            messages,
            Plan,
            temperature=self._temperature,
        )
