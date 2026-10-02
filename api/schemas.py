from pydantic import BaseModel, Field

from agentops.contracts import BudgetLimits


class CreateRunRequest(BaseModel):
    # Same bounds as RunState.objective, so a bad request is a 422 here, not a crash later.
    objective: str = Field(min_length=10, max_length=2000)
    limits: BudgetLimits | None = None
