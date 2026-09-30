import time
from collections.abc import Sequence
from typing import Any

import httpx

from agentops.errors import InvalidRequestError
from agentops.http_errors import classify_http_status, classify_transport_error, parse_retry_after

from .base import LLMResponse, Message, Usage

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class GeminiProvider:
    """Google Gemini Developer API (AI Studio), called directly over REST via httpx.

    A raw REST call instead of the google-genai SDK: one fewer dependency, and every
    request/response shape is visible here instead of hidden behind an SDK whose
    model names and behaviour have churned repeatedly this year (master doc, S21).
    """

    name = "gemini"

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
        # Gemini has no "system" role in contents: system text is a separate field,
        # and "assistant" is called "model" here.
        system_text = "\n".join(m.content for m in messages if m.role == "system")
        contents = [
            {
                "role": "model" if m.role == "assistant" else "user",
                "parts": [{"text": m.content}],
            }
            for m in messages
            if m.role != "system"
        ]
        generation_config: dict[str, Any] = {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        }
        if json_output:
            generation_config["responseMimeType"] = "application/json"
        body: dict[str, Any] = {"contents": contents, "generationConfig": generation_config}
        if system_text:
            body["systemInstruction"] = {"parts": [{"text": system_text}]}

        started = time.monotonic()
        try:
            resp = self._client.post(
                f"{_BASE_URL}/models/{self._model}:generateContent",
                params={"key": self._api_key},
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
        candidates = data.get("candidates") or []
        if not candidates:
            block_reason = data.get("promptFeedback", {}).get("blockReason", "no candidates")
            raise InvalidRequestError(
                f"gemini returned no output: {block_reason}", provider=self.name
            )

        text = "".join(part.get("text", "") for part in candidates[0]["content"]["parts"])
        usage_meta = data.get("usageMetadata", {})
        return LLMResponse(
            text=text,
            usage=Usage(
                tokens_in=usage_meta.get("promptTokenCount", 0),
                tokens_out=usage_meta.get("candidatesTokenCount", 0),
            ),
            provider=self.name,
            model=self._model,
            latency_ms=latency_ms,
        )
