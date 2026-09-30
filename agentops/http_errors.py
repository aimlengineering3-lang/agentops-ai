import httpx

from agentops.errors import (
    InvalidRequestError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RateLimitError,
)

_RETRYABLE_SERVER_STATUSES = {500, 502, 503, 504}


def parse_retry_after(value: str | None) -> float | None:
    """Parse a Retry-After header's seconds form. Returns None for anything else
    (missing, HTTP-date form, or garbage) rather than raising."""
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def classify_http_status(
    status_code: int, *, provider: str, retry_after_s: float | None = None
) -> Exception | None:
    """Map an HTTP status code to the matching ProviderError, or None for a 2xx."""
    if 200 <= status_code < 300:
        return None
    if status_code == 429:
        return RateLimitError(
            f"{provider} rate limit (HTTP 429)", provider=provider, retry_after_s=retry_after_s
        )
    if status_code in _RETRYABLE_SERVER_STATUSES:
        return ProviderUnavailableError(
            f"{provider} server error (HTTP {status_code})", provider=provider
        )
    return InvalidRequestError(f"{provider} request error (HTTP {status_code})", provider=provider)


def classify_transport_error(exc: Exception, *, provider: str) -> Exception:
    """Map an httpx transport-level failure (no HTTP response at all) to a ProviderError."""
    if isinstance(exc, httpx.TimeoutException):
        return ProviderTimeoutError(f"{provider} request timed out", provider=provider)
    return ProviderUnavailableError(f"{provider} unreachable: {exc}", provider=provider)
