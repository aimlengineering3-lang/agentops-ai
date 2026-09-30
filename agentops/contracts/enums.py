from enum import StrEnum


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    COMPLETED_WITH_LIMITATIONS = "completed_with_limitations"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"


TERMINAL_RUN_STATUSES = frozenset(
    {
        RunStatus.COMPLETED,
        RunStatus.COMPLETED_WITH_LIMITATIONS,
        RunStatus.FAILED,
        RunStatus.INTERRUPTED,
        RunStatus.CANCELLED,
    }
)


class SubtaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class SourceType(StrEnum):
    WEB = "web"
    FILE = "file"


class EvidenceStatus(StrEnum):
    CANDIDATE = "candidate"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    CONTRADICTED = "contradicted"


class FindingKind(StrEnum):
    EVIDENCE = "evidence"  # stated directly by a source
    ANALYSIS = "analysis"  # derived from evidence
    CONCLUSION = "conclusion"  # generated judgment


class AgentName(StrEnum):
    ORCHESTRATOR = "orchestrator"
    PLANNER = "planner"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    CRITIC = "critic"
    FINALIZER = "finalizer"


class EventType(StrEnum):
    RUN_STARTED = "run_started"
    PLAN = "plan"
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    RETRY = "retry"
    CRITIQUE = "critique"
    VALIDATION = "validation"
    BUDGET = "budget"
    ERROR = "error"
    RUN_FINISHED = "run_finished"


class CritiqueVerdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"


class IssueType(StrEnum):
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    MISSING_SUBTASK = "missing_subtask"
    WEAK_SOURCE_COVERAGE = "weak_source_coverage"
    CONTRADICTION = "contradiction"
    INCOMPLETE_OUTPUT = "incomplete_output"
    INVALID_OUTPUT = "invalid_output"
    UNSUPPORTED_CLAIM = "unsupported_claim"


class CritiqueAction(StrEnum):
    RESEARCH_MORE = "research_more"
    REANALYZE = "reanalyze"
    REPLAN = "replan"
    FIX_OUTPUT = "fix_output"
    ACCEPT_WITH_LIMITATIONS = "accept_with_limitations"


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
