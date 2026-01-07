"""Gemini provider adapter."""

from __future__ import annotations

import json
import threading

import google.generativeai as genai

import config
from services.ai.base import AIProvider, ChatModel, ChatSession, TextModel, StructuredModel
from services.ai.limits import AsyncRateLimiter, request_with_timeout


class GeminiChatSession(ChatSession):
    """Chat session wrapper for Gemini."""

    def __init__(self, session, rate_limiter: AsyncRateLimiter, timeout_seconds: float):
        self._session = session
        self._rate_limiter = rate_limiter
        self._timeout_seconds = timeout_seconds

    async def send_message_async(self, text: str) -> str:
        await self._rate_limiter.wait()
        response = await request_with_timeout(
            self._session.send_message_async(text),
            self._timeout_seconds
        )
        return response.text


class GeminiChatModel(ChatModel):
    """Chat model wrapper for Gemini."""

    def __init__(self, model, rate_limiter: AsyncRateLimiter, timeout_seconds: float):
        self._model = model
        self._rate_limiter = rate_limiter
        self._timeout_seconds = timeout_seconds

    def start_chat(self, history: list[dict]) -> ChatSession:
        session = self._model.start_chat(history=history)
        return GeminiChatSession(session, self._rate_limiter, self._timeout_seconds)


class GeminiTextModel(TextModel):
    """Text model wrapper for Gemini."""

    def __init__(self, model, rate_limiter: AsyncRateLimiter, timeout_seconds: float):
        self._model = model
        self._rate_limiter = rate_limiter
        self._timeout_seconds = timeout_seconds

    async def generate_content_async(self, prompt: str) -> str:
        await self._rate_limiter.wait()
        response = await request_with_timeout(
            self._model.generate_content_async(prompt),
            self._timeout_seconds
        )
        return response.text


class GeminiStructuredModel(StructuredModel):
    """Structured output model wrapper for Gemini."""

    def __init__(self, model, rate_limiter: AsyncRateLimiter, timeout_seconds: float):
        self._model = model
        self._rate_limiter = rate_limiter
        self._timeout_seconds = timeout_seconds

    async def generate_content_async(self, prompt: str, response_schema: dict) -> dict:
        await self._rate_limiter.wait()
        response = await request_with_timeout(
            self._model.generate_content_async(
                prompt,
                generation_config={
                    "response_mime_type": "application/json",
                    "response_schema": response_schema,
                }
            ),
            self._timeout_seconds
        )
        return json.loads(response.text)


class GeminiProvider(AIProvider):
    """Gemini provider implementation."""

    def __init__(self, provider_config: dict, rate_limiter: AsyncRateLimiter, timeout_seconds: float):
        self._config = provider_config
        self._rate_limiter = rate_limiter
        self._timeout_seconds = timeout_seconds
        self._config_lock = threading.Lock()

    def _configure(self, api_key: str) -> None:
        with self._config_lock:
            genai.configure(api_key=api_key)

    def create_chat_model(self, system_prompt: str) -> ChatModel:
        api_key = self._config.get("api_key")
        self._configure(api_key)
        model = genai.GenerativeModel(
            model_name=self._config.get("chat_model"),
            system_instruction=system_prompt,
            safety_settings=config.SAFETY_SETTINGS
        )
        return GeminiChatModel(model, self._rate_limiter, self._timeout_seconds)

    def create_reflection_model(self) -> TextModel:
        api_key = self._config.get("api_key")
        self._configure(api_key)
        model = genai.GenerativeModel(
            model_name=self._config.get("reflection_model"),
            safety_settings=config.SAFETY_SETTINGS
        )
        return GeminiTextModel(model, self._rate_limiter, self._timeout_seconds)

    def create_backup_reflection_model(self) -> TextModel | None:
        backup_model_name = self._config.get("backup_model")
        if not backup_model_name:
            return None
        api_key = self._config.get("api_key")
        self._configure(api_key)
        model = genai.GenerativeModel(
            model_name=backup_model_name,
            safety_settings=config.SAFETY_SETTINGS
        )
        return GeminiTextModel(model, self._rate_limiter, self._timeout_seconds)

    def create_concepts_model(self) -> StructuredModel:
        api_key = self._config.get("api_key")
        self._configure(api_key)
        model = genai.GenerativeModel(
            model_name=self._config.get("concepts_model"),
            safety_settings=config.SAFETY_SETTINGS
        )
        return GeminiStructuredModel(model, self._rate_limiter, self._timeout_seconds)
