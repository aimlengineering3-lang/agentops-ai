from collections.abc import Sequence
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from agentops.errors import MalformedOutputError

from .base import LLMResponse, Message, Usage

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class Generator(Protocol):
    """Anything that can turn messages into a response: an LLMProvider or an LLMRouter."""

    def generate(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.2,
        max_output_tokens: int = 2048,
        json_output: bool = False,
        timeout_s: float = 30.0,
    ) -> LLMResponse: ...


def generate_structured(
    generator: Generator,
    messages: Sequence[Message],
    schema: type[SchemaT],
    *,
    temperature: float = 0.2,
    max_output_tokens: int = 2048,
    timeout_s: float = 30.0,
) -> tuple[SchemaT, LLMResponse]:
    """Call `generator`, parse the response as `schema`, with one bounded repair attempt.

    Returns (parsed_object, last_llm_response). When a repair was needed, the returned
    response's `usage` is the SUM of both attempts and `attempts` is 2, so the caller
    records the true token cost. Raises MalformedOutputError (carrying both attempts'
    usage) if output still fails validation after the repair attempt -- the caller decides
    what "step failed, degrade gracefully" means for it.
    """
    attempt_messages = list(messages)
    first = response = generator.generate(
        attempt_messages,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        json_output=True,
        timeout_s=timeout_s,
    )
    try:
        return schema.model_validate_json(response.text), response
    except ValidationError as first_error:
        repair_prompt = (
            "Your previous response did not match the required JSON schema. "
            f"Validation error:\n{first_error}\n\n"
            "Reply again with ONLY corrected JSON that satisfies the schema. "
            "No prose, no markdown fences."
        )
        attempt_messages = [*attempt_messages, Message(role="user", content=repair_prompt)]
        response = generator.generate(
            attempt_messages,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            json_output=True,
            timeout_s=timeout_s,
        )
        combined = Usage(
            tokens_in=first.usage.tokens_in + response.usage.tokens_in,
            tokens_out=first.usage.tokens_out + response.usage.tokens_out,
        )
        try:
            parsed = schema.model_validate_json(response.text)
        except ValidationError as second_error:
            raise MalformedOutputError(
                "model output failed schema validation twice "
                f"(after one repair attempt): {second_error}",
                tokens_in=combined.tokens_in,
                tokens_out=combined.tokens_out,
                attempts=2,
            ) from second_error
        return parsed, response.model_copy(update={"usage": combined, "attempts": 2})
