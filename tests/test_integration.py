"""End-to-end tests: the real Orchestrator wired with all five real agents (Planner,
Researcher, Analyst, Critic, Finalizer), driven entirely by FakeLLM + FakeSearch so no
network or API key is needed. These are the closest thing to "run the actual app" that
the test suite has before FastAPI/Streamlit exist.
"""

import json

from agentops.agents import HybridCritic, LLMAnalyst, LLMFinalizer, LLMPlanner, LLMResearcher
from agentops.contracts import EventType, RunState, RunStatus
from agentops.orchestrator import Orchestrator
from agentops.search import SearchResult
from agentops.testing.fakes import FakeLLM, FakeSearch

OBJECTIVE = "Compare FastAPI and Flask for a small engineering team"


def _result(title: str, url: str) -> SearchResult:
    return SearchResult(title=title, url=url, content="relevant content", provider="fake-search")


def _build_orchestrator(llm: FakeLLM, search: FakeSearch) -> Orchestrator:
    return Orchestrator(
        planner=LLMPlanner(llm),
        researcher=LLMResearcher(llm, search),
        analyst=LLMAnalyst(llm),
        critic=HybridCritic(llm),
        finalizer=LLMFinalizer(llm),
    )


def test_full_run_happy_path_clean_completion():
    plan = json.dumps(
        {
            "subtasks": [
                {
                    "id": "s1",
                    "description": "Compare FastAPI and Flask for a small team",
                    "acceptance_criteria": ["at least two independent sources"],
                    "depends_on": [],
                }
            ]
        }
    )
    queries = json.dumps({"queries": ["fastapi vs flask performance", "flask small team fit"]})
    extraction = json.dumps(
        {
            "items": [
                {
                    "source_index": 0,
                    "extracted_claim": "FastAPI is async-first and benchmarks faster.",
                    "snippet": "s",
                },
                {
                    "source_index": 1,
                    "extracted_claim": "Flask suits small teams due to its simplicity.",
                    "snippet": "s",
                },
            ]
        }
    )
    analysis = json.dumps(
        {
            "findings": [
                {
                    "statement": "FastAPI benchmarks faster than Flask.",
                    "kind": "evidence",
                    "evidence_indices": [0],
                    "confidence": 0.8,
                },
                {
                    "statement": "Trade-off: FastAPI performance vs Flask simplicity.",
                    "kind": "analysis",
                    "evidence_indices": [0, 1],
                    "confidence": 0.6,
                },
            ]
        }
    )
    critique = json.dumps({"issues": []})
    final_report = json.dumps(
        {
            "executive_summary": "FastAPI wins on performance; Flask wins on simplicity.",
            "key_findings": [
                {
                    "statement": "FastAPI benchmarks faster than Flask.",
                    "finding_indices": [0],
                    "confidence": 0.8,
                },
                {
                    "statement": "Performance vs simplicity trade-off for small teams.",
                    "finding_indices": [0, 1],
                    "confidence": 0.6,
                },
            ],
            "comparison": "FastAPI vs Flask",
            "trade_offs": ["Performance vs simplicity"],
            "risks": [],
            "implementation_considerations": [],
        }
    )
    llm = FakeLLM([plan, queries, extraction, analysis, critique, final_report])
    search = FakeSearch(
        {
            "fastapi vs flask performance": [_result("Benchmark", "https://a.example/bench")],
            "flask small team fit": [_result("Guide", "https://b.example/guide")],
        }
    )
    orchestrator = _build_orchestrator(llm, search)
    state = RunState(objective=OBJECTIVE)

    orchestrator.run(state)

    assert state.status == RunStatus.COMPLETED
    assert state.limitations == []
    assert len(state.evidence) == 2
    assert len(state.findings) == 2
    assert len(state.critiques) == 1
    assert state.critiques[0].verdict.value == "pass"

    assert state.report is not None
    assert len(state.report.key_findings) == 2
    assert state.report.validation_passed is True

    assert state.usage.llm_calls == 6
    assert state.usage.tool_calls == 2

    types = [e.event_type for e in state.events]
    assert types[0] == EventType.RUN_STARTED
    assert types[-1] == EventType.RUN_FINISHED
    assert EventType.RETRY not in types
    assert types.count(EventType.TOOL_CALL) == 2
    assert types.count(EventType.TOOL_RESULT) == 2
    assert types.count(EventType.LLM_CALL) == 6


def test_full_run_with_a_critic_triggered_research_retry():
    plan = json.dumps(
        {
            "subtasks": [
                {
                    "id": "s1",
                    "description": "Compare FastAPI and Flask for a small team",
                    "acceptance_criteria": ["at least two independent sources"],
                    "depends_on": [],
                }
            ]
        }
    )
    queries_round1 = json.dumps({"queries": ["query one"]})
    extraction_round1 = json.dumps(
        {"items": [{"source_index": 0, "extracted_claim": "Claim A", "snippet": "s"}]}
    )
    analysis_round1 = json.dumps(
        {
            "findings": [
                {
                    "statement": "Claim A finding.",
                    "kind": "evidence",
                    "evidence_indices": [0],
                    "confidence": 0.7,
                }
            ]
        }
    )
    critique_round1 = json.dumps({"issues": []})  # deterministic check alone triggers the fail

    queries_round2 = json.dumps({"queries": ["query two"]})
    extraction_round2 = json.dumps(
        {"items": [{"source_index": 0, "extracted_claim": "Claim B", "snippet": "s"}]}
    )
    analysis_round2 = json.dumps(
        {
            "findings": [
                {
                    "statement": "Claim A finding.",
                    "kind": "evidence",
                    "evidence_indices": [0],
                    "confidence": 0.7,
                },
                {
                    "statement": "Claim B finding.",
                    "kind": "evidence",
                    "evidence_indices": [1],
                    "confidence": 0.75,
                },
            ]
        }
    )
    critique_round2 = json.dumps({"issues": []})
    final_report = json.dumps(
        {
            "executive_summary": "Combined synthesis of claims A and B.",
            "key_findings": [
                {"statement": "Combined finding.", "finding_indices": [0, 1, 2], "confidence": 0.75}
            ],
            "comparison": None,
            "trade_offs": [],
            "risks": [],
            "implementation_considerations": [],
        }
    )
    llm = FakeLLM(
        [
            plan,
            queries_round1,
            extraction_round1,
            analysis_round1,
            critique_round1,
            queries_round2,
            extraction_round2,
            analysis_round2,
            critique_round2,
            final_report,
        ]
    )
    search = FakeSearch(
        {
            "query one": [_result("A", "https://a.example/one")],
            "query two": [_result("B", "https://b.example/two")],
        }
    )
    orchestrator = _build_orchestrator(llm, search)
    state = RunState(objective=OBJECTIVE)

    orchestrator.run(state)

    # a single research-only retry happened, resolved cleanly -- no leftover limitations
    assert state.status == RunStatus.COMPLETED
    assert state.limitations == []
    assert len(state.critiques) == 2
    assert [c.verdict.value for c in state.critiques] == ["fail", "pass"]
    assert [c.iteration for c in state.critiques] == [1, 2]
    assert state.usage.retries_by_subtask == {"s1": 1}

    assert len(state.evidence) == 2  # one from each round
    assert len(state.findings) == 3  # 1 from round 1 + 2 from round 2

    assert state.report is not None
    assert state.report.validation_passed is True

    types = [e.event_type for e in state.events]
    assert types.count(EventType.RETRY) == 1
    assert types.count(EventType.TOOL_CALL) == 2
    assert types.count(EventType.LLM_CALL) == 10
