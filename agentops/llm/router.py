import time
from collections.abc import Callable, Sequence

from agentops.errors import ProviderError

from .base import LLMProvider, LLMResponse, Message


class LLMRouter:
    """Tries providers in order; fails over to the next one on a retryable error.

    Not an agent: this is plumbing. Every LLM-calling agent goes through one of
    these instead of holding a provider directly, so failover logic is never
    duplicated per agent (Planner, Researcher, Analyst, Critic, Finalizer).
    """

    def __init__(
        self,
        providers: Sequence[LLMProvider],
        *,
        sleep: Callable[[float], None] = time.sleep,
        backoff_s: float = 0.0,
    ) -> None:
        if not providers:
            raise ValueError("LLMRouter needs at least one provider")
        self._providers = list(providers)
        self._sleep = sleep
        self._backoff_s = backoff_s

    def generate(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.2,
        max_output_tokens: int = 2048,
        json_output: bool = False,
        timeout_s: float = 30.0,
    ) -> LLMResponse:
        for index, provider in enumerate(self._providers):
            is_last = index == len(self._providers) - 1
            try:
                return provider.generate(
                    messages,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    json_output=json_output,
                    timeout_s=timeout_s,
                )
            except ProviderError as exc:
                if not exc.retryable or is_last:
                    raise
                if self._backoff_s:
                    self._sleep(self._backoff_s)
        raise AssertionError("unreachable: loop above always returns or raises")
