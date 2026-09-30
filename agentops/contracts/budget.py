from pydantic import BaseModel, Field


class BudgetLimits(BaseModel):
    max_subtasks: int = Field(default=6, gt=0)
    max_critic_cycles: int = Field(default=2, gt=0)
    max_retries_per_subtask: int = Field(default=2, ge=0)
    max_tool_calls: int = Field(default=25, gt=0)
    max_llm_calls: int = Field(default=40, gt=0)
    max_wall_clock_s: int = Field(default=240, gt=0)
    llm_timeout_s: int = Field(default=30, gt=0)
    search_timeout_s: int = Field(default=15, gt=0)


class BudgetUsage(BaseModel):
    critic_cycles: int = 0
    tool_calls: int = 0
    llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    elapsed_s: float = 0.0
    retries_by_subtask: dict[str, int] = Field(default_factory=dict)

    def exhausted(self, limits: BudgetLimits) -> list[str]:
        """Names of budgets that are used up. Empty list means we may continue."""
        checks = {
            "tool_calls": self.tool_calls >= limits.max_tool_calls,
            "llm_calls": self.llm_calls >= limits.max_llm_calls,
            "wall_clock": self.elapsed_s >= limits.max_wall_clock_s,
        }
        return [name for name, hit in checks.items() if hit]

    def can_run_critic_cycle(self, limits: BudgetLimits) -> bool:
        """Loop control, not a hard stop: running out of cycles means 'finalize', not 'abort'."""
        return self.critic_cycles < limits.max_critic_cycles

    def can_retry(self, subtask_id: str, limits: BudgetLimits) -> bool:
        return self.retries_by_subtask.get(subtask_id, 0) < limits.max_retries_per_subtask

    def record_retry(self, subtask_id: str) -> None:
        self.retries_by_subtask[subtask_id] = self.retries_by_subtask.get(subtask_id, 0) + 1
