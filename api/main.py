from datetime import UTC, datetime
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException

from agentops.config import Settings, get_settings
from agentops.contracts import BudgetLimits, Event, RunState
from agentops.errors import ConfigError
from agentops.runs import RunManager

from .dependencies import get_run_manager
from .schemas import CreateRunRequest

app = FastAPI(title="AgentOps AI", version="0.1.0")

RunManagerDep = Annotated[RunManager, Depends(get_run_manager)]
SettingsDep = Annotated[Settings, Depends(get_settings)]

# Hard ceilings for caller-supplied limits: the public API must not let one request spend
# the whole free-tier quota. The agent's own defaults are well below these.
_LIMIT_CEILINGS = {
    "max_subtasks": 8,
    "max_critic_cycles": 3,
    "max_retries_per_subtask": 3,
    "max_tool_calls": 40,
    "max_llm_calls": 60,
    "max_wall_clock_s": 600,
}


def clamp_limits(limits: BudgetLimits | None) -> BudgetLimits | None:
    if limits is None:
        return None
    update = {name: min(getattr(limits, name), cap) for name, cap in _LIMIT_CEILINGS.items()}
    return limits.model_copy(update=update)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/runs", response_model=RunState, status_code=201)
def create_run(
    payload: CreateRunRequest, manager: RunManagerDep, settings: SettingsDep
) -> RunState:
    # Daily cap protects the free-tier quotas. list_runs is newest-first, so if the newest
    # `cap` runs all started today, the cap is reached.
    today = datetime.now(UTC).date()
    recent = manager.list_runs(limit=settings.daily_run_cap)
    if sum(1 for r in recent if r.created_at.astimezone(UTC).date() == today) >= (
        settings.daily_run_cap
    ):
        raise HTTPException(
            status_code=429,
            detail=f"Daily run limit ({settings.daily_run_cap}) reached; try again tomorrow.",
        )
    try:
        return manager.start_run(payload.objective, clamp_limits(payload.limits))
    except ConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/runs", response_model=list[RunState])
def list_runs(manager: RunManagerDep, limit: int = 25) -> list[RunState]:
    return manager.list_runs(limit=limit)


@app.get("/runs/{run_id}", response_model=RunState)
def get_run(run_id: str, manager: RunManagerDep) -> RunState:
    state = manager.get_run(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"no run with id {run_id}")
    return state


@app.get("/runs/{run_id}/events", response_model=list[Event])
def get_run_events(run_id: str, manager: RunManagerDep, after: int = 0) -> list[Event]:
    state = manager.get_run(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"no run with id {run_id}")
    return [e for e in state.events if e.seq > after]
