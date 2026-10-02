import re
import time
from collections.abc import Sequence
from typing import Any

import httpx

from agentops.errors import InvalidRequestError
from agentops.http_errors import classify_http_status, classify_transport_error, parse_retry_after

from .base import LLMResponse, Message, Usage

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
_RETRY_IN = re.compile(r"retry in ([0-9]+(?:\.[0-9]+)?)s", re.IGNORECASE)
_RETRY_MARGIN_S = 1.0  # Gemini's stated delay is exact; retrying on the boundary often 429s again
# A DAILY quota that is gone will not come back in the seconds Gemini's RetryInfo suggests
# (that hint is just the per-request retry window), so bench the model for a while instead.
_DAILY_QUOTA_COOLDOWN_S = 3600.0


def _is_daily_quota_failure(detail: dict) -> bool:
    """True for a QuotaFailure detail whose violated quota is a per-day one."""
    return any(
        isinstance(v, dict) and "PerDay" in str(v.get("quotaId", ""))
        for v in detail.get("violations") or []
    )


def _body_retry_delay(resp: httpx.Response) -> float | None:
    """Gemini reports how long to wait in the 429 BODY (RetryInfo, or "Please retry in 15s"),
    not in a Retry-After header. Returns seconds, or None if the body says nothing usable."""
    try:
        data = resp.json()
    except ValueError:
        return None
    error = data.get("error") if isinstance(data, dict) else None
    if not isinstance(error, dict):
        return None
    for detail in error.get("details") or []:
        if isinstance(detail, dict) and _is_daily_quota_failure(detail):
            return _DAILY_QUOTA_COOLDOWN_S
    for detail in error.get("details") or []:
        if isinstance(detail, dict) and str(detail.get("@type", "")).endswith("RetryInfo"):
            delay = detail.get("retryDelay")  # e.g. "15s" or "15.009679695s"
            if isinstance(delay, str) and delay.endswith("s"):
                try:
                    return float(delay[:-1]) + _RETRY_MARGIN_S
                except ValueError:
                    pass
    match = _RETRY_IN.search(str(error.get("message", "")))
    return float(match.group(1)) + _RETRY_MARGIN_S if match else None


class GeminiProvider:
    """Google Gemini Developer API (AI Studio), called directly over REST via httpx.

    A raw REST call instead of the google-genai SDK: one fewer dependency, and every
    request/response shape is visible here instead of hidden behind an SDK whose
    model names and behaviour have churned repeatedly this year (master doc, S21).
    """

    name = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        client: httpx.Client | None = None,
        name: str = "gemini",
    ) -> None:
        # Free-tier quotas are per model, so the same key can back several GeminiProviders;
        # each needs its own name so the router tracks their cooldowns separately.
        self.name = name
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

        retry_after = parse_retry_after(resp.headers.get("Retry-After"))
        if retry_after is None and resp.status_code == 429:
            retry_after = _body_retry_delay(resp)
        error = classify_http_status(
            resp.status_code, provider=self.name, retry_after_s=retry_after
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
