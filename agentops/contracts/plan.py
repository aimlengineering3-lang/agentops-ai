from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .enums import SubtaskStatus


class Subtask(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    id: str
    description: str = Field(min_length=5, max_length=500)
    acceptance_criteria: list[str] = Field(min_length=1)
    depends_on: list[str] = Field(default_factory=list)
    status: SubtaskStatus = SubtaskStatus.PENDING
    attempts: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _no_self_dependency(self) -> Self:
        if self.id in self.depends_on:
            raise ValueError(f"subtask {self.id} depends on itself")
        return self


class Plan(BaseModel):
    subtasks: list[Subtask] = Field(min_length=1, max_length=10)

    @property
    def subtask_ids(self) -> set[str]:
        return {s.id for s in self.subtasks}

    def get(self, subtask_id: str) -> Subtask:
        for subtask in self.subtasks:
            if subtask.id == subtask_id:
                return subtask
        raise KeyError(subtask_id)

    def execution_order(self) -> list[str]:
        """Topological order (Kahn). Raises ValueError on a dependency cycle."""
        remaining = {s.id: set(s.depends_on) for s in self.subtasks}
        order: list[str] = []
        while remaining:
            ready = [sid for sid, deps in remaining.items() if not deps]
            if not ready:
                raise ValueError(f"dependency cycle among: {sorted(remaining)}")
            for sid in ready:
                order.append(sid)
                del remaining[sid]
            for deps in remaining.values():
                deps.difference_update(ready)
        return order

    @model_validator(mode="after")
    def _validate_graph(self) -> Self:
        ids = [s.id for s in self.subtasks]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate subtask ids")
        known = set(ids)
        for subtask in self.subtasks:
            missing = set(subtask.depends_on) - known
            if missing:
                raise ValueError(f"{subtask.id} depends on unknown subtasks: {sorted(missing)}")
        self.execution_order()  # cycle check
        return self
