from typing import ClassVar


class AgentOpsError(Exception):
    """Base class for all application errors."""


class ConfigError(AgentOpsError):
    """A required setting (API key, model name) is missing."""


class ProviderError(AgentOpsError):
    """An external provider (LLM, search) failed.

    `retryable` tells the failover router whether trying again or switching
    provider can help. Errors caused by our own bad request cannot be fixed by retrying.
    """

    retryable: ClassVar[bool] = False

    def __init__(self, message: str, *, provider: str | None = None) -> None:
        super().__init__(message)
        self.provider = provider


class RateLimitError(ProviderError):
    retryable = True

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        retry_after_s: float | None = None,
    ) -> None:
        super().__init__(message, provider=provider)
        self.retry_after_s = retry_after_s


class ProviderTimeoutError(ProviderError):
    retryable = True


class ProviderUnavailableError(ProviderError):
    retryable = True


class InvalidRequestError(ProviderError):
    retryable = False


class MalformedOutputError(AgentOpsError):
    """Model output could not be parsed or validated against the expected schema."""


class BudgetExceededError(AgentOpsError):
    """A run budget (tool calls, LLM calls, time, ...) is used up."""


class GuardrailViolation(AgentOpsError):
    """A security or safety rule blocked an action."""
