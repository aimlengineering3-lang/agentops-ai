import logging
import time
from collections.abc import Callable, Sequence

from agentops.errors import ProviderError, RateLimitError

from .base import LLMProvider, LLMResponse, Message

logger = logging.getLogger(__name__)

_RATE_LIMIT_COOLDOWN_S = 30.0  # used when a 429 carries no Retry-After header
_OTHER_COOLDOWN_S = 10.0  # 503s and timeouts
_MAX_COOLDOWN_S = 300.0


class LLMRouter:
    """Tries providers in order; fails over to the next one on a retryable error.

    Not an agent: this is plumbing. Every LLM-calling agent goes through one of
    these instead of holding a provider directly, so failover logic is never
    duplicated per agent (Planner, Researcher, Analyst, Critic, Finalizer).

    With max_wait_s > 0 the router also becomes quota-aware (free tiers rate-limit hard):
    - a provider that fails is "cooling down" (Retry-After if given, else a growing
      default) and is skipped instead of being hammered again on every call;
    - if every provider is cooling down, the router waits for the soonest one -- but only
      if that wait is at most max_wait_s; otherwise it raises immediately (bounded).
    With max_wait_s == 0 (default) it is the plain, stateless failover router.
    """

    def __init__(
        self,
        providers: Sequence[LLMProvider],
        *,
        sleep: Callable[[float], None] = time.sleep,
        backoff_s: float = 0.0,
        max_wait_s: float = 0.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not providers:
            raise ValueError("LLMRouter needs at least one provider")
        self._providers = list(providers)
        self._sleep = sleep
        self._backoff_s = backoff_s
        self._max_wait_s = max_wait_s
        self._clock = clock
        self._cooling_until: dict[str, float] = {}
        self._failures: dict[str, int] = {}

    def generate(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.2,
        max_output_tokens: int = 2048,
        json_output: bool = False,
        timeout_s: float = 30.0,
    ) -> LLMResponse:
        resilient = self._max_wait_s > 0
        last_exc: ProviderError | None = None
        # Resilient mode gets at most ONE waited second pass per call: bounded by design.
        for _ in range(2 if resilient else 1):
            candidates = self._available()
            if not candidates:
                wait = self._soonest_wait()
                if wait > self._max_wait_s:
                    raise last_exc or RateLimitError(
                        f"all LLM providers are cooling down (~{wait:.0f}s left)"
                    )
                logger.warning("All LLM providers cooling down; waiting %.1fs", wait)
                self._sleep(wait)
                candidates = self._available() or list(self._providers)

            for index, provider in enumerate(candidates):
                is_last = index == len(candidates) - 1
                name = self._name(provider)
                try:
                    response = provider.generate(
                        messages,
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                        json_output=json_output,
                        timeout_s=timeout_s,
                    )
                except ProviderError as exc:
                    if not exc.retryable:
                        raise
                    last_exc = exc
                    if resilient:
                        self._record_failure(name, exc)
                    # Failover used to be invisible: the trace only names the provider
                    # that finally answered. Log why one was skipped (message only,
                    # never request content or keys).
                    logger.warning(
                        "LLM provider %s failed (%s: %s)%s",
                        name,
                        type(exc).__name__,
                        exc,
                        "; failing over" if not is_last else "",
                    )
                    if not is_last and self._backoff_s:
                        self._sleep(self._backoff_s)
                    continue
                self._failures.pop(name, None)
                self._cooling_until.pop(name, None)
                return response

        assert last_exc is not None  # every path above either returned or recorded a failure
        raise last_exc

    # -- cooldown bookkeeping -----------------------------------------------------------

    @staticmethod
    def _name(provider: LLMProvider) -> str:
        return getattr(provider, "name", type(provider).__name__)

    def _available(self) -> list[LLMProvider]:
        now = self._clock()
        return [p for p in self._providers if self._cooling_until.get(self._name(p), 0.0) <= now]

    def _soonest_wait(self) -> float:
        now = self._clock()
        until = min(self._cooling_until.get(self._name(p), 0.0) for p in self._providers)
        return max(0.0, until - now)

    def _record_failure(self, name: str, exc: ProviderError) -> None:
        failures = self._failures.get(name, 0) + 1
        self._failures[name] = failures
        if isinstance(exc, RateLimitError) and exc.retry_after_s is not None:
            cooldown = exc.retry_after_s  # the provider told us exactly how long
        else:
            base = _RATE_LIMIT_COOLDOWN_S if isinstance(exc, RateLimitError) else _OTHER_COOLDOWN_S
            # Repeated failures back off (a provider whose DAILY quota is gone stays benched).
            cooldown = min(base * 2 ** (failures - 1), _MAX_COOLDOWN_S)
        self._cooling_until[name] = self._clock() + cooldown
