from collections.abc import Sequence
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class Usage(BaseModel):
    tokens_in: int = Field(default=0, ge=0)
    tokens_out: int = Field(default=0, ge=0)


class LLMResponse(BaseModel):
    text: str
    usage: Usage = Field(default_factory=Usage)
    provider: str
    model: str
    latency_ms: int = Field(default=0, ge=0)
    # Provider requests behind this response: 2 when a schema-repair retry was needed.
    attempts: int = Field(default=1, ge=1)


@runtime_checkable
class LLMProvider(Protocol):
    """Anything with these members is an LLM provider (Gemini, Groq, a fake...)."""

    name: str

    def generate(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.2,
        max_output_tokens: int = 2048,
        json_output: bool = False,
        timeout_s: float = 30.0,
    ) -> LLMResponse: ...
