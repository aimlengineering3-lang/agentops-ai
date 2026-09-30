from datetime import datetime
from typing import Self
from urllib.parse import urlparse

from pydantic import BaseModel, Field, computed_field, model_validator

from .enums import EvidenceStatus, SourceType
from .ids import new_id, utc_now


class Evidence(BaseModel):
    """One atomic, source-attributed claim collected by a tool."""

    id: str = Field(default_factory=lambda: new_id("ev"))
    subtask_id: str
    source_type: SourceType
    title: str = Field(max_length=300)
    url: str | None = None
    extracted_claim: str = Field(min_length=1, max_length=1000)
    snippet: str = Field(default="", max_length=2000)
    retrieved_at: datetime = Field(default_factory=utc_now)
    retrieval_query: str | None = None
    provider: str
    status: EvidenceStatus = EvidenceStatus.CANDIDATE

    @model_validator(mode="after")
    def _web_evidence_needs_http_url(self) -> Self:
        if self.source_type == SourceType.WEB:
            parsed = urlparse(self.url or "")
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("web evidence requires a valid http(s) url")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def domain(self) -> str | None:
        if not self.url:
            return None
        return urlparse(self.url).netloc.lower().removeprefix("www.")
