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


def test_failover_is_logged_with_reason(caplog):
    primary = FakeLLM([RateLimitError("429 quota", provider="primary")], name="primary")
    fallback = FakeLLM(["fallback answer"], name="fallback")
    router = LLMRouter([primary, fallback])

    with caplog.at_level("WARNING", logger="agentops.llm.router"):
        router.generate(MESSAGES)

    assert "primary" in caplog.text
    assert "RateLimitError" in caplog.text
    assert "failing over" in caplog.text


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_rate_limited_provider_is_skipped_during_cooldown():
    clock = _Clock()
    primary = FakeLLM([RateLimitError("429", provider="primary", retry_after_s=30)], name="primary")
    fallback = FakeLLM(["one", "two"], name="fallback")
    router = LLMRouter([primary, fallback], max_wait_s=5, clock=clock, sleep=clock.sleep)

    assert router.generate(MESSAGES).text == "one"
    assert router.generate(MESSAGES).text == "two"

    assert len(primary.calls) == 1  # not hammered again while cooling down
    assert clock.sleeps == []


def test_waits_for_the_soonest_provider_when_all_are_cooling_down():
    clock = _Clock()
    only = FakeLLM(
        [RateLimitError("429", provider="only", retry_after_s=3), "recovered"], name="only"
    )
    router = LLMRouter([only], max_wait_s=10, clock=clock, sleep=clock.sleep)

    response = router.generate(MESSAGES)

    assert response.text == "recovered"
    assert clock.sleeps == [3]


def test_never_waits_longer_than_max_wait():
    clock = _Clock()
    only = FakeLLM([RateLimitError("429", provider="only", retry_after_s=120)], name="only")
    router = LLMRouter([only], max_wait_s=10, clock=clock, sleep=clock.sleep)

    with pytest.raises(RateLimitError):
        router.generate(MESSAGES)

    assert clock.sleeps == []  # bounded: it gave up instead of sleeping 120s


def test_repeated_failures_without_retry_after_back_off_longer():
    clock = _Clock()
    primary = FakeLLM(
        [RateLimitError("429", provider="primary"), RateLimitError("429", provider="primary")],
        name="primary",
    )
    fallback = FakeLLM(["a", "b", "c"], name="fallback")
    router = LLMRouter([primary, fallback], max_wait_s=5, clock=clock, sleep=clock.sleep)

    router.generate(MESSAGES)  # primary fails -> cooldown 30s
    clock.now += 31
    router.generate(MESSAGES)  # primary retried, fails again -> cooldown 60s
    clock.now += 31
    router.generate(MESSAGES)  # still cooling (60s), so it is skipped

    assert len(primary.calls) == 2
