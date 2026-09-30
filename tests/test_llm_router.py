import pytest

from agentops.errors import InvalidRequestError, ProviderTimeoutError, RateLimitError
from agentops.llm import Message
from agentops.llm.router import LLMRouter
from agentops.testing.fakes import FakeLLM

MESSAGES = [Message(role="user", content="hello")]


def test_uses_primary_when_it_succeeds():
    primary = FakeLLM(["primary answer"], name="primary")
    fallback = FakeLLM(["fallback answer"], name="fallback")
    router = LLMRouter([primary, fallback])

    response = router.generate(MESSAGES)

    assert response.text == "primary answer"
    assert response.provider == "primary"
    assert len(fallback.calls) == 0  # fallback was never touched


def test_fails_over_on_retryable_error():
    primary = FakeLLM([RateLimitError("rate limited", provider="primary")], name="primary")
    fallback = FakeLLM(["fallback answer"], name="fallback")
    router = LLMRouter([primary, fallback])

    response = router.generate(MESSAGES)

    assert response.text == "fallback answer"
    assert response.provider == "fallback"


def test_does_not_fail_over_on_non_retryable_error():
    primary = FakeLLM([InvalidRequestError("bad request", provider="primary")], name="primary")
    fallback = FakeLLM(["fallback answer"], name="fallback")
    router = LLMRouter([primary, fallback])

    with pytest.raises(InvalidRequestError):
        router.generate(MESSAGES)
    assert len(fallback.calls) == 0  # a bad request fails the same way everywhere


def test_raises_after_last_provider_also_fails():
    primary = FakeLLM([ProviderTimeoutError("timeout", provider="primary")], name="primary")
    fallback = FakeLLM([ProviderTimeoutError("timeout", provider="fallback")], name="fallback")
    router = LLMRouter([primary, fallback])

    with pytest.raises(ProviderTimeoutError):
        router.generate(MESSAGES)


def test_backoff_sleep_is_called_between_retries():
    primary = FakeLLM([RateLimitError("rate limited", provider="primary")], name="primary")
    fallback = FakeLLM(["fallback answer"], name="fallback")
    sleeps: list[float] = []
    router = LLMRouter([primary, fallback], sleep=sleeps.append, backoff_s=0.5)

    router.generate(MESSAGES)

    assert sleeps == [0.5]


def test_router_requires_at_least_one_provider():
    with pytest.raises(ValueError):
        LLMRouter([])
