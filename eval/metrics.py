"""Scoring for AgentOps runs. Pure functions over a finished RunState: no network, no LLM.

Every metric is something the run record can actually show. Notes on what they do NOT prove:
- `citation_resolution`, `supported_rate` and `evidence_backed_rate` measure STRUCTURAL
  grounding (a cited id exists in the collected evidence). They do not prove that the
  evidence semantically entails the claim; that would need an LLM judge or human review.
- `topic_coverage` is a keyword check against task-specific expected topics: a cheap proxy
  for "did the report address what was asked", not a quality score.
"""

import statistics
from typing import Any

from pydantic import BaseModel, Field

from agentops.contracts import (
    CritiqueVerdict,
    EventType,
    RunState,
    RunStatus,
    SubtaskStatus,
)

_FINISHED = {RunStatus.COMPLETED, RunStatus.COMPLETED_WITH_LIMITATIONS}


class EvalTask(BaseModel):
    id: str
    category: str
    objective: str
    expected_topics: list[str] = Field(default_factory=list)
    # Tasks with no good answer (a made-up standard): the right outcome is an honest,
    # limited report, not confident findings.
    expect_limitations: bool = False


class RunScore(BaseModel):
    task_id: str
    category: str
    status: str
    completed: bool  # run reached a report-producing end state
    clean: bool  # completed with no limitations
    report_present: bool
    validation_passed: bool
    subtask_completion: float | None
    evidence_coverage: float | None  # share of subtasks with >= 1 evidence item
    n_evidence: int
    n_source_domains: int
    n_findings: int
    citation_resolution: float | None  # share of cited ids that exist in collected evidence
    supported_rate: float | None  # share of report findings the citation verifier kept
    evidence_backed_rate: float | None  # share of findings with >= 1 resolvable citation
    topic_coverage: float | None
    tool_success_rate: float | None
    duplicate_searches: int  # identical live (non-cached) queries repeated within the run
    retries: int
    critic_cycles: int
    critic_recovered: bool | None  # first critique failed, last passed (None if never failed)
    llm_repairs: int
    injection_drops: int
    llm_calls: int
    tool_calls: int
    tokens_in: int
    tokens_out: int
    wall_s: float
    n_limitations: int
    limitations_expectation_met: bool | None  # only for tasks with expect_limitations


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def _report_text(state: RunState) -> str:
    report = state.report
    if report is None:
        return ""
    parts = [report.executive_summary, report.comparison or ""]
    parts += [f.statement for f in report.key_findings]
    parts += report.trade_offs + report.risks + report.implementation_considerations
    return "\n".join(parts).lower()


def score_run(state: RunState, task: EvalTask, wall_s: float) -> RunScore:
    report = state.report
    subtasks = state.plan.subtasks if state.plan else []
    findings = report.key_findings if report else []

    cited = [eid for f in findings for eid in f.evidence_ids]
    resolved = [eid for eid in cited if eid in state.evidence]
    backed = sum(1 for f in findings if any(e in state.evidence for e in f.evidence_ids))

    with_evidence = sum(1 for s in subtasks if state.evidence_for(s.id))
    text = _report_text(state)
    topics_hit = sum(1 for t in task.expected_topics if t.lower() in text)

    tool_results = [e for e in state.events if e.event_type == EventType.TOOL_RESULT]
    live_queries = [
        e.data.get("query")
        for e in state.events
        if e.event_type == EventType.TOOL_CALL and not e.data.get("cached") and e.data.get("query")
    ]

    verdicts = [c.verdict for c in state.critiques]
    failed_once = CritiqueVerdict.FAIL in verdicts
    recovered = (verdicts[-1] == CritiqueVerdict.PASS) if failed_once else None

    limitations_met = None
    if task.expect_limitations:
        # Honest degradation: limitations recorded, or a report with no/low-confidence claims.
        limitations_met = bool(state.limitations) or not findings

    return RunScore(
        task_id=task.id,
        category=task.category,
        status=state.status.value,
        completed=state.status in _FINISHED,
        clean=state.status == RunStatus.COMPLETED,
        report_present=report is not None,
        validation_passed=bool(report and report.validation_passed),
        subtask_completion=_ratio(
            sum(1 for s in subtasks if s.status == SubtaskStatus.DONE), len(subtasks)
        ),
        evidence_coverage=_ratio(with_evidence, len(subtasks)),
        n_evidence=len(state.evidence),
        n_source_domains=len({e.domain for e in state.evidence.values() if e.domain}),
        n_findings=len(findings),
        citation_resolution=_ratio(len(resolved), len(cited)),
        supported_rate=_ratio(sum(1 for f in findings if f.supported), len(findings)),
        evidence_backed_rate=_ratio(backed, len(findings)),
        topic_coverage=_ratio(topics_hit, len(task.expected_topics)),
        tool_success_rate=_ratio(sum(1 for e in tool_results if e.success), len(tool_results)),
        duplicate_searches=len(live_queries) - len(set(live_queries)),
        retries=sum(1 for e in state.events if e.event_type == EventType.RETRY),
        critic_cycles=state.usage.critic_cycles,
        critic_recovered=recovered,
        llm_repairs=state.usage.llm_repairs,
        injection_drops=sum(
            1 for e in state.events if e.event_type == EventType.VALIDATION and e.success is False
        ),
        llm_calls=state.usage.llm_calls,
        tool_calls=state.usage.tool_calls,
        tokens_in=state.usage.tokens_in,
        tokens_out=state.usage.tokens_out,
        wall_s=round(wall_s, 1),
        n_limitations=len(state.limitations),
        limitations_expectation_met=limitations_met,
    )


def _mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return round(statistics.fmean(present), 3) if present else None


def summarize(scores: list[RunScore]) -> dict[str, Any]:
    """Aggregate over a benchmark run. Means skip tasks where a metric is not applicable."""
    n = len(scores)
    if n == 0:
        return {"n_tasks": 0}
    walls = sorted(s.wall_s for s in scores)
    recovered = [s.critic_recovered for s in scores if s.critic_recovered is not None]
    expected = [
        s.limitations_expectation_met for s in scores if s.limitations_expectation_met is not None
    ]
    return {
        "n_tasks": n,
        "completion_rate": round(sum(s.completed for s in scores) / n, 3),
        "clean_completion_rate": round(sum(s.clean for s in scores) / n, 3),
        "validation_pass_rate": round(sum(s.validation_passed for s in scores) / n, 3),
        "mean_subtask_completion": _mean([s.subtask_completion for s in scores]),
        "mean_evidence_coverage": _mean([s.evidence_coverage for s in scores]),
        "mean_citation_resolution": _mean([s.citation_resolution for s in scores]),
        "mean_supported_rate": _mean([s.supported_rate for s in scores]),
        "mean_evidence_backed_rate": _mean([s.evidence_backed_rate for s in scores]),
        "mean_topic_coverage": _mean([s.topic_coverage for s in scores]),
        "mean_tool_success_rate": _mean([s.tool_success_rate for s in scores]),
        "total_duplicate_searches": sum(s.duplicate_searches for s in scores),
        "runs_with_retries": sum(1 for s in scores if s.retries),
        "critic_recovery": f"{sum(recovered)}/{len(recovered)}" if recovered else "n/a",
        "total_llm_repairs": sum(s.llm_repairs for s in scores),
        "total_injection_drops": sum(s.injection_drops for s in scores),
        "honest_degradation": f"{sum(expected)}/{len(expected)}" if expected else "n/a",
        "mean_llm_calls": _mean([float(s.llm_calls) for s in scores]),
        "mean_tool_calls": _mean([float(s.tool_calls) for s in scores]),
        "mean_tokens_in": _mean([float(s.tokens_in) for s in scores]),
        "mean_tokens_out": _mean([float(s.tokens_out) for s in scores]),
        "latency_p50_s": round(statistics.median(walls), 1),
        "latency_max_s": walls[-1],
    }
