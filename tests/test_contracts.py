import pytest
from pydantic import ValidationError

from agentops.contracts import (
    AgentName,
    BudgetLimits,
    BudgetUsage,
    Critique,
    CritiqueAction,
    CritiqueIssue,
    CritiqueVerdict,
    Event,
    EventType,
    Evidence,
    Finding,
    FindingKind,
    IssueType,
    Plan,
    Report,
    ReportFinding,
    ReportMetadata,
    RunState,
    Severity,
    SourceType,
    Subtask,
)

OBJECTIVE = "Evaluate three approaches for building an enterprise AI platform."


def make_subtask(sid: str, depends_on: list[str] | None = None) -> Subtask:
    return Subtask(
        id=sid,
        description=f"Investigate topic {sid}",
        acceptance_criteria=["at least two independent sources"],
        depends_on=depends_on or [],
    )


def make_plan() -> Plan:
    return Plan(subtasks=[make_subtask("t1"), make_subtask("t2", ["t1"])])


def make_web_evidence(subtask_id: str = "t1", url: str = "https://www.Example.com/a") -> Evidence:
    return Evidence(
        subtask_id=subtask_id,
        source_type=SourceType.WEB,
        title="Example",
        url=url,
        extracted_claim="X is Y.",
        provider="tavily",
    )


# ---- Evidence -------------------------------------------------------------


def test_web_evidence_domain_is_normalised():
    assert make_web_evidence().domain == "example.com"


@pytest.mark.parametrize("bad_url", ["javascript:alert(1)", "file:///etc/passwd", "not a url"])
def test_web_evidence_rejects_non_http_urls(bad_url):
    with pytest.raises(ValidationError):
        make_web_evidence(url=bad_url)


def test_file_evidence_needs_no_url():
    evidence = Evidence(
        subtask_id="t1",
        source_type=SourceType.FILE,
        title="notes.txt",
        extracted_claim="Budget is 10k.",
        provider="file",
    )
    assert evidence.domain is None


# ---- Plan -----------------------------------------------------------------


def test_plan_execution_order_respects_dependencies():
    plan = Plan(subtasks=[make_subtask("t2", ["t1"]), make_subtask("t1")])
    assert plan.execution_order() == ["t1", "t2"]


def test_plan_rejects_duplicate_ids():
    with pytest.raises(ValidationError):
        Plan(subtasks=[make_subtask("t1"), make_subtask("t1")])


def test_plan_rejects_unknown_dependency():
    with pytest.raises(ValidationError):
        Plan(subtasks=[make_subtask("t1", ["ghost"])])


def test_plan_rejects_cycle():
    with pytest.raises(ValidationError):
        Plan(subtasks=[make_subtask("t1", ["t2"]), make_subtask("t2", ["t1"])])


def test_subtask_rejects_self_dependency():
    with pytest.raises(ValidationError):
        make_subtask("t1", ["t1"])


# ---- Finding / Critique / Event ------------------------------------------


def test_evidence_finding_must_cite_evidence():
    with pytest.raises(ValidationError):
        Finding(subtask_id="t1", statement="X", kind=FindingKind.EVIDENCE, confidence=0.5)


def test_fail_verdict_needs_issues():
    with pytest.raises(ValidationError):
        Critique(iteration=1, verdict=CritiqueVerdict.FAIL)


def test_pass_verdict_cannot_have_high_severity_issue():
    issue = CritiqueIssue(
        type=IssueType.CONTRADICTION,
        severity=Severity.HIGH,
        action=CritiqueAction.RESEARCH_MORE,
        detail="Sources disagree.",
    )
    with pytest.raises(ValidationError):
        Critique(iteration=1, verdict=CritiqueVerdict.PASS, issues=[issue])


def test_event_summary_is_length_limited():
    with pytest.raises(ValidationError):
        Event(
            run_id="r",
            seq=1,
            agent=AgentName.PLANNER,
            event_type=EventType.PLAN,
            summary="x" * 301,
        )


# ---- Budget ---------------------------------------------------------------


def test_budget_reports_exhausted_limits():
    limits = BudgetLimits(max_tool_calls=2)
    assert BudgetUsage().exhausted(limits) == []
    assert BudgetUsage(tool_calls=2).exhausted(limits) == ["tool_calls"]


def test_retry_budget_is_per_subtask():
    limits = BudgetLimits(max_retries_per_subtask=2)
    usage = BudgetUsage()
    usage.record_retry("t1")
    usage.record_retry("t1")
    assert not usage.can_retry("t1", limits)
    assert usage.can_retry("t2", limits)


# ---- RunState -------------------------------------------------------------


def test_event_sequence_numbers_increase():
    state = RunState(objective=OBJECTIVE)
    first = state.add_event(AgentName.ORCHESTRATOR, EventType.RUN_STARTED, "start")
    second = state.add_event(AgentName.PLANNER, EventType.PLAN, "planned")
    assert (first.seq, second.seq) == (1, 2)
    assert first.run_id == state.run_id


def test_duplicate_evidence_is_rejected():
    state = RunState(objective=OBJECTIVE, plan=make_plan())
    evidence = make_web_evidence()
    state.add_evidence(evidence)
    with pytest.raises(ValueError):
        state.add_evidence(evidence)


def test_evidence_for_unknown_subtask_is_rejected():
    state = RunState(objective=OBJECTIVE, plan=make_plan())
    with pytest.raises(ValueError):
        state.add_evidence(make_web_evidence(subtask_id="ghost"))


def test_finding_citing_unknown_evidence_is_rejected():
    state = RunState(objective=OBJECTIVE, plan=make_plan())
    finding = Finding(
        subtask_id="t1",
        statement="X is Y",
        kind=FindingKind.EVIDENCE,
        evidence_ids=["ev_does_not_exist"],
        confidence=0.9,
    )
    with pytest.raises(ValueError):
        state.add_finding(finding)


def test_state_survives_json_round_trip():
    state = RunState(objective=OBJECTIVE, plan=make_plan())
    evidence = make_web_evidence()
    state.add_evidence(evidence)
    state.add_finding(
        Finding(
            subtask_id="t1",
            statement="X is Y",
            kind=FindingKind.EVIDENCE,
            evidence_ids=[evidence.id],
            confidence=0.8,
        )
    )
    state.add_event(AgentName.RESEARCHER, EventType.TOOL_RESULT, "1 result", subtask_id="t1")

    restored = RunState.model_validate_json(state.model_dump_json())
    assert restored == state


# ---- Report -----------------------------------------------------------------


def make_report_metadata(**overrides) -> ReportMetadata:
    defaults = dict(
        run_id="run_test",
        duration_s=12.5,
        llm_calls=3,
        tool_calls=5,
        tokens_in=1000,
        tokens_out=400,
        critic_cycles=1,
        providers_used=["gemini", "tavily"],
    )
    defaults.update(overrides)
    return ReportMetadata(**defaults)


def test_report_with_no_findings_needs_limitations():
    with pytest.raises(ValidationError):
        Report(
            objective=OBJECTIVE,
            executive_summary="Nothing could be established.",
            validation_passed=False,
            metadata=make_report_metadata(),
        )


def test_report_with_no_findings_but_limitations_is_valid():
    report = Report(
        objective=OBJECTIVE,
        executive_summary="Research failed before any finding was produced.",
        limitations=["Search provider was unreachable for every subtask."],
        validation_passed=False,
        metadata=make_report_metadata(),
    )
    assert report.key_findings == []


def test_report_with_findings_is_valid():
    report = Report(
        objective=OBJECTIVE,
        executive_summary="Three approaches were compared on cost, security and complexity.",
        key_findings=[
            ReportFinding(
                statement="Approach A is cheapest.",
                evidence_ids=["ev_1"],
                confidence=0.8,
            )
        ],
        trade_offs=["A is cheap but less secure."],
        validation_passed=True,
        metadata=make_report_metadata(),
    )
    assert report.key_findings[0].supported is True  # default, until the citation verifier runs
