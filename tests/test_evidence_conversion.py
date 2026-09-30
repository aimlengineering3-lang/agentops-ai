from agentops.agents.evidence import evidence_from_search_result
from agentops.contracts import SourceType
from agentops.search import SearchResult


def test_converts_basic_fields():
    result = SearchResult(
        title="Enterprise AI Platforms 2026",
        url="https://Example.com/report",
        content="A detailed comparison of platforms.",
        provider="tavily",
    )
    evidence = evidence_from_search_result(result, subtask_id="s1", retrieval_query="enterprise ai")

    assert evidence.subtask_id == "s1"
    assert evidence.source_type == SourceType.WEB
    assert evidence.url == "https://Example.com/report"
    assert evidence.domain == "example.com"
    assert evidence.retrieval_query == "enterprise ai"
    assert evidence.provider == "tavily"
    assert evidence.extracted_claim == "A detailed comparison of platforms."


def test_truncates_overlong_content_and_keeps_within_evidence_limits():
    result = SearchResult(
        title="t" * 500,
        url="https://a.com",
        content="c" * 3000,
        provider="serper",
    )
    evidence = evidence_from_search_result(result, subtask_id="s1", retrieval_query="q")

    assert len(evidence.title) <= 300
    assert len(evidence.extracted_claim) <= 1000
    assert len(evidence.snippet) <= 2000
    assert evidence.title.endswith("…")


def test_falls_back_to_title_when_content_is_empty():
    result = SearchResult(title="Only a title", url="https://a.com", content="", provider="tavily")
    evidence = evidence_from_search_result(result, subtask_id="s1", retrieval_query="q")
    assert evidence.extracted_claim == "Only a title"
