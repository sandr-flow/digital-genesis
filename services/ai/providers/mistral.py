"""Mistral provider adapter."""

from __future__ import annotations

import asyncio
import json

from services.ai.base import AIProvider, ChatModel, ChatSession, TextModel, StructuredModel, normalize_history
from services.ai.limits import AsyncRateLimiter, request_with_retry

try:
    from mistralai import Mistral
except ImportError:  # pragma: no cover - handled at runtime
    Mistral = None


class MistralChatSession(ChatSession):
    """Stateless chat session emulation for Mistral."""

    def __init__(
        self,
        client: Mistral,
        model_name: str,
        system_prompt: str,
        history: list[dict],
        rate_limiter: AsyncRateLimiter,
        timeout_seconds: float,
    ):
        self._client = client
        self._model_name = model_name
        self._rate_limiter = rate_limiter
        self._timeout_seconds = timeout_seconds
        self._messages: list[dict] = []
        if system_prompt:
            self._messages.append({"role": "system", "content": system_prompt})
        self._messages.extend(normalize_history(history))

    async def send_message_async(self, text: str) -> str:
        self._messages.append({"role": "user", "content": text})
        
        response = await request_with_retry(
            lambda: asyncio.to_thread(
                self._client.chat.complete,
                model=self._model_name,
                messages=self._messages,
            ),
            self._rate_limiter,
            self._timeout_seconds,
        )
        content = response.choices[0].message.content
        self._messages.append({"role": "assistant", "content": content})
        return content


class MistralChatModel(ChatModel):
    """Chat model wrapper for Mistral."""

    def __init__(
        self,
        client: Mistral,
        model_name: str,
        system_prompt: str,
        rate_limiter: AsyncRateLimiter,
        timeout_seconds: float,
    ):
        self._client = client
        self._model_name = model_name
        self._system_prompt = system_prompt
        self._rate_limiter = rate_limiter
        self._timeout_seconds = timeout_seconds

    def start_chat(self, history: list[dict]) -> ChatSession:
        return MistralChatSession(
            self._client,
            self._model_name,
            system_prompt=self._system_prompt,
            history=history,
            rate_limiter=self._rate_limiter,
            timeout_seconds=self._timeout_seconds,
        )


class MistralTextModel(TextModel):
    """Text generation wrapper for Mistral."""

    def __init__(self, client: Mistral, model_name: str, rate_limiter: AsyncRateLimiter, timeout_seconds: float):
        self._client = client
        self._model_name = model_name
        self._rate_limiter = rate_limiter
        self._timeout_seconds = timeout_seconds

    async def generate_content_async(self, prompt: str) -> str:
        response = await request_with_retry(
            lambda: asyncio.to_thread(
                self._client.chat.complete,
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
            ),
            self._rate_limiter,
            self._timeout_seconds,
        )
        return response.choices[0].message.content


class MistralStructuredModel(StructuredModel):
    """Structured output wrapper for Mistral."""

    def __init__(self, client: Mistral, model_name: str, rate_limiter: AsyncRateLimiter, timeout_seconds: float):
        self._client = client
        self._model_name = model_name
        self._rate_limiter = rate_limiter
        self._timeout_seconds = timeout_seconds

    async def generate_content_async(self, prompt: str, response_schema: dict) -> dict:
        """Generate structured content using Mistral's JSON schema mode.
        
        Args:
            prompt: User prompt.
            response_schema: JSON schema for response validation.
            
        Returns:
            Parsed JSON response matching the schema.
        """
        messages = [{"role": "user", "content": prompt}]
        
        # Use json_schema mode for guaranteed schema compliance
        # Mistral API expects the schema in response_format
        response = await request_with_retry(
            lambda: asyncio.to_thread(
                self._client.chat.complete,
                model=self._model_name,
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "cognitive_assets_extraction",
                        "strict": True,
                        "schema": response_schema
                    }
                },
            ),
            self._rate_limiter,
            self._timeout_seconds,
        )
        return json.loads(response.choices[0].message.content)


class MistralProvider(AIProvider):
    """Mistral provider implementation."""

    def __init__(self, provider_config: dict, rate_limiter: AsyncRateLimiter, timeout_seconds: float):
        if Mistral is None:
            raise RuntimeError("mistralai is not installed. Add it to requirements.txt.")
        self._config = provider_config
        self._rate_limiter = rate_limiter
        self._timeout_seconds = timeout_seconds
        self._client = Mistral(api_key=self._config.get("api_key"))

    def create_chat_model(self, system_prompt: str) -> ChatModel:
        model_name = self._config.get("chat_model")
        return MistralChatModel(
            self._client,
            model_name,
            system_prompt,
            self._rate_limiter,
            self._timeout_seconds,
        )

    def create_reflection_model(self) -> TextModel:
        model_name = self._config.get("reflection_model")
        return MistralTextModel(self._client, model_name, self._rate_limiter, self._timeout_seconds)

    def create_backup_reflection_model(self) -> TextModel | None:
        model_name = self._config.get("backup_model")
        if not model_name:
            return None
        return MistralTextModel(self._client, model_name, self._rate_limiter, self._timeout_seconds)

    def create_concepts_model(self) -> StructuredModel:
        model_name = self._config.get("concepts_model")
        return MistralStructuredModel(self._client, model_name, self._rate_limiter, self._timeout_seconds)
