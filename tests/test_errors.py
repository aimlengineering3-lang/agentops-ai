from agentops.errors import (
    InvalidRequestError,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RateLimitError,
)


def test_retryable_flags():
    assert RateLimitError("x").retryable
    assert ProviderTimeoutError("x").retryable
    assert ProviderUnavailableError("x").retryable
    assert not InvalidRequestError("x").retryable


def test_rate_limit_carries_retry_after_and_provider():
    err = RateLimitError("slow down", provider="gemini", retry_after_s=12.5)
    assert err.retry_after_s == 12.5
    assert err.provider == "gemini"


def test_all_provider_errors_share_a_base_class():
    for cls in (
        RateLimitError,
        ProviderTimeoutError,
        ProviderUnavailableError,
        InvalidRequestError,
    ):
        assert issubclass(cls, ProviderError)
