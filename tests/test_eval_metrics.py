import json
from pathlib import Path

from agentops.contracts import (
    AgentName,
    Critique,
    CritiqueAction,
    CritiqueIssue,
    CritiqueVerdict,
    EventType,
    Evidence,
    IssueType,
    Plan,
    Report,
    ReportFinding,
    ReportMetadata,
    RunState,
    RunStatus,
    Severity,
    SourceType,
    Subtask,
    SubtaskStatus,
)
from eval.metrics import EvalTask, score_run, summarize

OBJECTIVE = "Compare three approaches for an enterprise AI platform"


def _evidence(subtask_id: str, url: str) -> Evidence:
    return Evidence(
        subtask_id=subtask_id,
        source_type=SourceType.WEB,
        title="t",
        url=url,
        extracted_claim="claim",
        provider="fake",
    )


def _state() -> tuple[RunState, Evidence, Evidence]:
    state = RunState(objective=OBJECTIVE, status=RunStatus.COMPLETED)
    state.plan = Plan(
        subtasks=[
            Subtask(id="s1", description="First task", acceptance_criteria=["a"]),
            Subtask(id="s2", description="Second task", acceptance_criteria=["a"]),
        ]
    )
    state.plan.get("s1").status = SubtaskStatus.DONE
    state.plan.get("s2").status = SubtaskStatus.FAILED
    ev1 = _evidence("s1", "https://a.com/x")
    ev2 = _evidence("s1", "https://b.com/y")
    state.add_evidence(ev1)
    state.add_evidence(ev2)
    return state, ev1, ev2


def _report(state: RunState, findings: list[ReportFinding]) -> Report:
    return Report(
        objective=OBJECTIVE,
        executive_summary="Security and cost drive the choice.",
        key_findings=findings,
        validation_passed=True,
        metadata=ReportMetadata(
            run_id=state.run_id,
            duration_s=1,
            llm_calls=1,
            tool_calls=1,
            tokens_in=1,
            tokens_out=1,
            critic_cycles=1,
        ),
    )


def test_grounding_and_coverage_metrics():
    state, ev1, ev2 = _state()
    state.report = _report(
        state,
        [
            ReportFinding(statement="Backed", evidence_ids=[ev1.id, ev2.id], confidence=0.9),
            ReportFinding(
                statement="Bad cite", evidence_ids=["ev_missing"], confidence=0.5, supported=False
            ),
        ],
    )
    task = EvalTask(
        id="t", category="c", objective=OBJECTIVE, expected_topics=["security", "latency"]
    )

    score = score_run(state, task, wall_s=12.34)

    assert score.completed and score.clean and score.report_present
    assert score.subtask_completion == 0.5
    assert score.evidence_coverage == 0.5  # only s1 has evidence
    assert score.n_source_domains == 2
    assert score.citation_resolution == 2 / 3
    assert score.supported_rate == 0.5
    assert score.evidence_backed_rate == 0.5
    assert score.topic_coverage == 0.5  # "security" present, "latency" not
    assert score.wall_s == 12.3


def test_tool_use_retry_and_recovery_metrics():
    state, _, _ = _state()
    bus_events = [
        (EventType.TOOL_CALL, None, {"query": "q1"}),
        (EventType.TOOL_RESULT, True, {}),
        (EventType.TOOL_CALL, None, {"query": "q1"}),  # same live query again: wasted call
        (EventType.TOOL_RESULT, False, {}),
        (EventType.TOOL_CALL, True, {"query": "q1", "cached": True}),  # cache hit: not counted
        (EventType.RETRY, None, {}),
        (EventType.VALIDATION, False, {}),
    ]
    for event_type, success, data in bus_events:
        state.add_event(AgentName.RESEARCHER, event_type, "e", success=success, data=data)
    issue = CritiqueIssue(
        type=IssueType.INSUFFICIENT_EVIDENCE,
        subtask_id="s1",
        severity=Severity.MEDIUM,
        action=CritiqueAction.RESEARCH_MORE,
        detail="thin",
    )
    state.critiques = [
        Critique(iteration=1, verdict=CritiqueVerdict.FAIL, issues=[issue]),
        Critique(iteration=2, verdict=CritiqueVerdict.PASS),
    ]
    task = EvalTask(id="t", category="c", objective=OBJECTIVE)

    score = score_run(state, task, wall_s=1.0)

    assert score.tool_success_rate == 0.5
    assert score.duplicate_searches == 1
    assert score.retries == 1
    assert score.critic_recovered is True
    assert score.injection_drops == 1


def test_run_without_a_report_scores_as_incomplete_not_as_a_crash():
    state = RunState(objective=OBJECTIVE, status=RunStatus.FAILED)
    task = EvalTask(id="t", category="c", objective=OBJECTIVE, expect_limitations=True)

    score = score_run(state, task, wall_s=0.5)

    assert not score.completed and not score.report_present
    assert score.citation_resolution is None
    assert score.limitations_expectation_met is True  # no findings = nothing fabricated


def test_summarize_aggregates_and_skips_inapplicable_metrics():
    state, ev1, _ = _state()
    finding = ReportFinding(statement="ok", evidence_ids=[ev1.id], confidence=0.9)
    state.report = _report(state, [finding])
    task = EvalTask(id="t", category="c", objective=OBJECTIVE)
    good = score_run(state, task, wall_s=10)
    failed = score_run(RunState(objective=OBJECTIVE, status=RunStatus.FAILED), task, wall_s=2)

    summary = summarize([good, failed])

    assert summary["n_tasks"] == 2
    assert summary["completion_rate"] == 0.5
    assert summary["mean_citation_resolution"] == 1.0  # the failed run has no citations: skipped
    assert summary["critic_recovery"] == "n/a"
    assert summary["latency_max_s"] == 10.0
    assert summarize([]) == {"n_tasks": 0}


def test_benchmark_task_file_is_valid_and_ids_are_unique():
    raw = json.loads((Path(__file__).parent.parent / "eval" / "tasks.json").read_text("utf-8"))
    tasks = [EvalTask.model_validate(item) for item in raw]

    assert len({t.id for t in tasks}) == len(tasks)
    assert any(t.expect_limitations for t in tasks)  # keeps one hallucination probe
