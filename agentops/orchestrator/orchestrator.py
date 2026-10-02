import time
from collections.abc import Callable

from agentops.contracts import (
    AgentName,
    Critique,
    CritiqueAction,
    CritiqueVerdict,
    EventType,
    RunState,
    RunStatus,
    SubtaskStatus,
)
from agentops.contracts.ids import utc_now
from agentops.errors import BudgetExceededError, MalformedOutputError, ProviderError

from .budget import BudgetGuard
from .context import RunContext
from .event_bus import EventBus
from .protocols import Analyst, Critic, Finalizer, Planner, Researcher

_ORCH = AgentName.ORCHESTRATOR


class Orchestrator:
    """Deterministic control loop: plan -> per-subtask research+analysis -> critic loop -> finalize.

    Plain code, not an LLM: only code may decide whether the run keeps looping.
    Agents do the work; the orchestrator owns routing, budgets and stopping.
    """

    def __init__(
        self,
        *,
        planner: Planner,
        researcher: Researcher,
        analyst: Analyst,
        critic: Critic,
        finalizer: Finalizer,
        bus: EventBus | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._planner = planner
        self._researcher = researcher
        self._analyst = analyst
        self._critic = critic
        self._finalizer = finalizer
        self._bus = bus or EventBus()
        self._clock = clock

    def run(self, state: RunState) -> RunState:
        ctx = RunContext(bus=self._bus, guard=BudgetGuard(state, self._bus, self._clock))
        state.status = RunStatus.RUNNING
        self._bus.emit(state, _ORCH, EventType.RUN_STARTED, "Run started")

        try:
            order = self._make_plan(state, ctx)
        except ProviderError as exc:
            # No plan means there is nothing to research or report on, so this is the one
            # place a provider failure legitimately ends the run as FAILED -- but with a
            # recorded reason and a closed trace, not a silent crash.
            self._fail_run(state, f"Planning failed: {type(exc).__name__}: {exc}")
            return state
        try:
            for subtask_id in order:
                self._run_subtask(state, ctx, subtask_id, research=True)
            self._critique_loop(state, ctx)
        except BudgetExceededError as exc:
            # A budget running out mid-run is a normal, expected outcome (spec §11),
            # not a bug: stop the work loop but still let the run finish and report
            # what it has instead of propagating a crash out of run().
            state.limitations.append(f"Run stopped early: {exc}")

        self._finalize(state, ctx)
        return state

    # -- steps ---------------------------------------------------------------------------

    def _make_plan(self, state: RunState, ctx: RunContext) -> list[str]:
        plan = self._planner.plan(state, ctx)
        if len(plan.subtasks) > state.limits.max_subtasks:
            raise MalformedOutputError(
                f"plan has {len(plan.subtasks)} subtasks, limit is {state.limits.max_subtasks}"
            )
        state.plan = plan
        order = plan.execution_order()
        self._bus.emit(
            state,
            _ORCH,
            EventType.PLAN,
            f"Plan accepted: {len(order)} subtasks",
            success=True,
            data={"execution_order": order},
        )
        return order

    def _run_subtask(
        self, state: RunState, ctx: RunContext, subtask_id: str, *, research: bool
    ) -> None:
        assert state.plan is not None
        ctx.guard.check()
        subtask = state.plan.get(subtask_id)
        subtask.status = SubtaskStatus.RUNNING
        subtask.attempts += 1
        try:
            if research:
                self._researcher.research(state, ctx, subtask)
            self._analyst.analyze(state, ctx, subtask)
        except (ProviderError, MalformedOutputError) as exc:
            # One subtask's provider outage (rate limit, timeout, bad output) must not sink
            # the whole run: mark it failed, record why, and let the remaining subtasks,
            # the critic and the finalizer work with what exists. BudgetExceededError is
            # deliberately NOT caught here -- it stops the whole run (see run()).
            subtask.status = SubtaskStatus.FAILED
            state.limitations.append(f"Subtask {subtask_id} failed: {type(exc).__name__}: {exc}")
            self._bus.emit(
                state,
                _ORCH,
                EventType.ERROR,
                f"Subtask {subtask_id} failed: {type(exc).__name__}",
                subtask_id=subtask_id,
                success=False,
                data={"error": type(exc).__name__},
            )
            return
        subtask.status = SubtaskStatus.DONE

    def _critique_loop(self, state: RunState, ctx: RunContext) -> None:
        while True:
            ctx.guard.check()
            critique = self._critic.critique(state, ctx, iteration=state.usage.critic_cycles + 1)
            state.critiques.append(critique)
            state.usage.critic_cycles += 1
            passed = critique.verdict == CritiqueVerdict.PASS
            self._bus.emit(
                state,
                _ORCH,
                EventType.CRITIQUE,
                f"Critic {critique.verdict.value}: {len(critique.issues)} issue(s)",
                success=passed,
                data={"iteration": critique.iteration},
            )
            if passed:
                return

            if not state.usage.can_run_critic_cycle(state.limits):
                # No cycle left to verify a retry, so do not retry: record and move on.
                state.limitations.extend(f"Unresolved: {i.detail}" for i in critique.issues)
                return

            retries = self._plan_retries(state, critique)
            if not retries:
                return
            for subtask_id, research in retries:
                self._run_subtask(state, ctx, subtask_id, research=research)

    def _plan_retries(self, state: RunState, critique: Critique) -> list[tuple[str, bool]]:
        """Turn critic issues into concrete next steps. This mapping is what makes it agentic.

        Returns [(subtask_id, needs_research)]. Several issues on one subtask collapse into a
        single retry, and RESEARCH_MORE wins over REANALYZE (research is followed by analysis).
        """
        assert state.plan is not None
        wanted: dict[str, bool] = {}
        for issue in critique.issues:
            match issue.action:
                case CritiqueAction.RESEARCH_MORE | CritiqueAction.REANALYZE:
                    if issue.subtask_id not in state.plan.subtask_ids:
                        state.limitations.append(f"Unresolved (unknown subtask): {issue.detail}")
                        continue
                    sid = issue.subtask_id
                    needs_research = issue.action == CritiqueAction.RESEARCH_MORE
                    wanted[sid] = wanted.get(sid, False) or needs_research
                case CritiqueAction.ACCEPT_WITH_LIMITATIONS:
                    state.limitations.append(issue.detail)
                case CritiqueAction.REPLAN:
                    # Re-planning arrives in Phase 5; until then it is an honest limitation.
                    state.limitations.append(f"Re-plan needed but not supported: {issue.detail}")
                case CritiqueAction.FIX_OUTPUT:
                    pass  # the Finalizer reads the last critique and repairs the output

        retries: list[tuple[str, bool]] = []
        for subtask_id, needs_research in wanted.items():
            if not state.usage.can_retry(subtask_id, state.limits):
                state.limitations.append(f"Subtask {subtask_id}: retry budget used up")
                continue
            state.usage.record_retry(subtask_id)
            scope = "research + analysis" if needs_research else "analysis"
            self._bus.emit(
                state,
                _ORCH,
                EventType.RETRY,
                f"Retrying {subtask_id}: {scope}",
                subtask_id=subtask_id,
            )
            retries.append((subtask_id, needs_research))
        return retries

    def _fail_run(self, state: RunState, reason: str) -> None:
        state.limitations.append(reason)
        state.status = RunStatus.FAILED
        state.finished_at = utc_now()
        self._bus.emit(state, _ORCH, EventType.ERROR, reason, success=False)
        self._bus.emit(
            state,
            _ORCH,
            EventType.RUN_FINISHED,
            f"Run finished: {state.status.value}",
            success=False,
        )

    def _finalize(self, state: RunState, ctx: RunContext) -> None:
        ctx.guard.enter_finalization()
        try:
            self._finalizer.finalize(state, ctx)
        except BudgetExceededError as exc:
            # The Finalizer may itself call an LLM and hit the same wall; still
            # finish the run gracefully instead of crashing here too.
            state.limitations.append(f"Finalizer could not complete: {exc}")
        clean = (
            bool(state.critiques)
            and state.critiques[-1].verdict == CritiqueVerdict.PASS
            and not state.limitations
        )
        state.status = RunStatus.COMPLETED if clean else RunStatus.COMPLETED_WITH_LIMITATIONS
        state.finished_at = utc_now()
        self._bus.emit(
            state,
            _ORCH,
            EventType.RUN_FINISHED,
            f"Run finished: {state.status.value}",
            success=True,
        )
