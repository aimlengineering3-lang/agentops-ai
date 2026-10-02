"""Check every configured Gemini model with one tiny request each.

Usage (from the project folder, venv active):  python scripts/check_gemini_models.py

Reads GEMINI_API_KEY, GEMINI_MODEL and GEMINI_EXTRA_MODELS from .env. Prints one line per
model: OK, or the HTTP status and, for a 429, which quota ran out (per-minute vs per-day).
The API key is never printed. Costs one request per model.
"""

import httpx

from agentops.config import get_settings

URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def check(model: str, key: str) -> str:
    try:
        resp = httpx.post(
            URL.format(model=model),
            params={"key": key},
            json={"contents": [{"parts": [{"text": "Reply with the single word: ok"}]}]},
            timeout=30,
        )
    except httpx.HTTPError as exc:
        return f"NETWORK ERROR ({type(exc).__name__})"
    if resp.status_code == 200:
        return "OK - works"
    if resp.status_code == 429:
        quota_ids = []
        try:
            for detail in resp.json()["error"].get("details", []):
                for violation in detail.get("violations", []) or []:
                    quota_ids.append(violation.get("quotaId", "?"))
        except (ValueError, KeyError, AttributeError):
            pass
        which = ", ".join(quota_ids) or "unknown quota"
        return f"429 quota exhausted ({which})"
    if resp.status_code == 404:
        return "404 model name not found - check the spelling"
    return f"HTTP {resp.status_code}: {resp.text[:150]}"


def main() -> None:
    settings = get_settings()
    if not settings.gemini_api_key:
        print("GEMINI_API_KEY is not set in .env")
        return
    key = settings.gemini_api_key.get_secret_value()
    extras = [m.strip() for m in settings.gemini_extra_models.split(",") if m.strip()]
    models = list(dict.fromkeys([settings.gemini_model, *extras]))
    for model in models:
        print(f"{model:40s} {check(model, key)}")


if __name__ == "__main__":
    main()
