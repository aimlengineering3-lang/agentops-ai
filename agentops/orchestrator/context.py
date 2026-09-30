from dataclasses import dataclass

from .budget import BudgetGuard
from .event_bus import EventBus


@dataclass(frozen=True)
class RunContext:
    """Per-run services handed to every agent, so agents never build their own."""

    bus: EventBus
    guard: BudgetGuard
