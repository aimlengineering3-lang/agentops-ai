import time
from collections.abc import Sequence
from typing import Any

import httpx

from agentops.errors import InvalidRequestError
from agentops.http_errors import classify_http_status, classify_transport_error, parse_retry_after

from .base import LLMResponse, Message, Usage

_BASE_URL = "https://api.groq.com/openai/v1"


class GroqProvider:
    """Groq's OpenAI-compatible chat completions API, called directly over REST via httpx."""

    name = "groq"

    def __init__(self, api_key: str, model: str, *, client: httpx.Client | None = None) -> None:
        self._api_key = api_key
        self._model = model
        self._client = client or httpx.Client()

    def generate(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.2,
        max_output_tokens: int = 2048,
        json_output: bool = False,
        timeout_s: float = 30.0,
    ) -> LLMResponse:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_output_tokens,
        }
        if json_output:
            body["response_format"] = {"type": "json_object"}

        started = time.monotonic()
        try:
            resp = self._client.post(
                f"{_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=body,
                timeout=timeout_s,
            )
        except httpx.HTTPError as exc:
            raise classify_transport_error(exc, provider=self.name) from exc
        latency_ms = int((time.monotonic() - started) * 1000)

        error = classify_http_status(
            resp.status_code,
            provider=self.name,
            retry_after_s=parse_retry_after(resp.headers.get("Retry-After")),
        )
        if error is not None:
            raise error

        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise InvalidRequestError("groq returned no choices", provider=self.name)

        usage = data.get("usage", {})
        return LLMResponse(
            text=choices[0]["message"]["content"],
            usage=Usage(
                tokens_in=usage.get("prompt_tokens", 0),
                tokens_out=usage.get("completion_tokens", 0),
            ),
            provider=self.name,
            model=self._model,
            latency_ms=latency_ms,
        )
