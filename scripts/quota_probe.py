"""Probe each provider's real free-tier status with ONE minimal request per provider.

Run manually, never in CI: `python scripts/quota_probe.py`. This never loops or
retries -- a single small call per provider is enough to read the rate-limit headers
(or the error) that provider actually returns right now, instead of trusting blog
posts that go stale within weeks (master doc, S21).
"""

import sys
from dataclasses import dataclass

import httpx

from agentops.config import get_settings

_RATE_LIMIT_HEADER_HINTS = ("rate", "limit", "quota", "retry")


@dataclass
class ProbeResult:
    provider: str
    ok: bool
    detail: str
    rate_limit_headers: dict[str, str]


def _interesting_headers(headers: httpx.Headers) -> dict[str, str]:
    return {
        key: value
        for key, value in headers.items()
        if any(hint in key.lower() for hint in _RATE_LIMIT_HEADER_HINTS)
    }


def probe_gemini(api_key: str, model: str) -> ProbeResult:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {"contents": [{"role": "user", "parts": [{"text": "Reply with just: ok"}]}]}
    try:
        resp = httpx.post(url, params={"key": api_key}, json=body, timeout=15.0)
    except httpx.HTTPError as exc:
        return ProbeResult("gemini", False, f"transport error: {exc}", {})
    return ProbeResult(
        "gemini",
        resp.status_code < 300,
        f"HTTP {resp.status_code}: {resp.text[:200]}",
        _interesting_headers(resp.headers),
    )


def probe_groq(api_key: str, model: str) -> ProbeResult:
    url = "https://api.groq.com/openai/v1/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with just: ok"}],
        "max_tokens": 5,
    }
    try:
        resp = httpx.post(
            url, headers={"Authorization": f"Bearer {api_key}"}, json=body, timeout=15.0
        )
    except httpx.HTTPError as exc:
        return ProbeResult("groq", False, f"transport error: {exc}", {})
    return ProbeResult(
        "groq",
        resp.status_code < 300,
        f"HTTP {resp.status_code}: {resp.text[:200]}",
        _interesting_headers(resp.headers),
    )


def probe_tavily(api_key: str) -> ProbeResult:
    try:
        resp = httpx.post(
            "https://api.tavily.com/search",
            json={"api_key": api_key, "query": "test", "max_results": 1},
            timeout=15.0,
        )
    except httpx.HTTPError as exc:
        return ProbeResult("tavily", False, f"transport error: {exc}", {})
    return ProbeResult(
        "tavily",
        resp.status_code < 300,
        f"HTTP {resp.status_code}: {resp.text[:200]}",
        _interesting_headers(resp.headers),
    )


def probe_serper(api_key: str) -> ProbeResult:
    try:
        resp = httpx.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": "test", "num": 1},
            timeout=15.0,
        )
    except httpx.HTTPError as exc:
        return ProbeResult("serper", False, f"transport error: {exc}", {})
    return ProbeResult(
        "serper",
        resp.status_code < 300,
        f"HTTP {resp.status_code}: {resp.text[:200]}",
        _interesting_headers(resp.headers),
    )


def _print_result(result: ProbeResult) -> None:
    status = "OK" if result.ok else "FAILED"
    print(f"\n[{result.provider}] {status}")
    print(f"  {result.detail}")
    if result.rate_limit_headers:
        print("  rate-limit headers:")
        for key, value in result.rate_limit_headers.items():
            print(f"    {key}: {value}")
    else:
        print("  (no rate-limit headers in this response)")


def main() -> int:
    settings = get_settings()
    results: list[ProbeResult] = []

    if not (settings.gemini_api_key and settings.gemini_model):
        print("[gemini] SKIPPED (GEMINI_API_KEY or GEMINI_MODEL not set in .env)")
    else:
        results.append(
            probe_gemini(settings.gemini_api_key.get_secret_value(), settings.gemini_model)
        )

    if not (settings.groq_api_key and settings.groq_model):
        print("[groq] SKIPPED (GROQ_API_KEY or GROQ_MODEL not set in .env)")
    else:
        results.append(probe_groq(settings.groq_api_key.get_secret_value(), settings.groq_model))

    if not settings.tavily_api_key:
        print("[tavily] SKIPPED (TAVILY_API_KEY not set in .env)")
    else:
        results.append(probe_tavily(settings.tavily_api_key.get_secret_value()))

    if not settings.serper_api_key:
        print("[serper] SKIPPED (SERPER_API_KEY not set in .env)")
    else:
        results.append(probe_serper(settings.serper_api_key.get_secret_value()))

    for result in results:
        _print_result(result)

    if not results:
        print("\nNo providers had both a key and (where needed) a model set. Nothing probed.")
        return 0
    return 1 if any(not r.ok for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
