from agentops.contracts import Evidence, SourceType
from agentops.search import SearchResult

# Must stay <= the corresponding field limits on Evidence (contracts/evidence.py).
# Kept here, next to the only code that truncates into those fields.
_TITLE_MAX = 300
_CLAIM_MAX = 1000
_SNIPPET_MAX = 2000


def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def evidence_from_search_result(
    result: SearchResult, *, subtask_id: str, retrieval_query: str
) -> Evidence:
    """Convert one raw, untrusted search result into Evidence with no LLM involved.

    Used as the Researcher's fallback when the extraction LLM call fails: the raw
    content becomes the claim verbatim, so a bad LLM day never loses evidence outright.
    """
    claim_source = result.content or result.title or "(search result had no content)"
    return evidence_from_claim(
        result,
        subtask_id=subtask_id,
        retrieval_query=retrieval_query,
        extracted_claim=claim_source,
        snippet=result.content,
    )


def evidence_from_claim(
    result: SearchResult,
    *,
    subtask_id: str,
    retrieval_query: str,
    extracted_claim: str,
    snippet: str,
) -> Evidence:
    """Build Evidence from a search result plus an already-produced claim/snippet.

    Used for the normal path, where the Researcher's extraction LLM call has turned
    a raw result into a claim in its own words rather than a verbatim content dump.
    """
    return Evidence(
        subtask_id=subtask_id,
        source_type=SourceType.WEB,
        title=_truncate(result.title or result.url, _TITLE_MAX),
        url=result.url,
        extracted_claim=_truncate(extracted_claim, _CLAIM_MAX),
        snippet=_truncate(snippet or result.content, _SNIPPET_MAX),
        retrieval_query=retrieval_query,
        provider=result.provider,
    )
