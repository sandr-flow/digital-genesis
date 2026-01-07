"""AI provider gateway for selecting and wiring implementations."""

from __future__ import annotations

from services.ai.base import AIProvider
from services.ai.limits import AsyncRateLimiter
from services.ai.providers.gemini import GeminiProvider
from services.ai.providers.mistral import MistralProvider


class AIProviderGateway:
    """Gateway for AI provider selection and initialization."""

    def __init__(self, provider_name: str, config: dict, rate_limit_rps: float, timeout_seconds: float):
        self._provider_name = (provider_name or "").lower()
        self._config = config
        self._rate_limiter = AsyncRateLimiter(rate_limit_rps)
        self._timeout_seconds = timeout_seconds

    def get_provider(self) -> AIProvider:
        """Return a concrete AI provider instance."""
        if self._provider_name == "gemini":
            return GeminiProvider(self._config, self._rate_limiter, self._timeout_seconds)
        if self._provider_name == "mistral":
            return MistralProvider(self._config, self._rate_limiter, self._timeout_seconds)
        raise ValueError(f"Unsupported AI provider: {self._provider_name}")
