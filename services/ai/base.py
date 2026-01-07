"""Base abstractions for AI provider integration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable


Message = dict[str, str]


def normalize_history(history: Iterable[dict]) -> list[Message]:
    """Normalize history into a list of role/content messages."""
    normalized: list[Message] = []
    for item in history:
        role = item.get("role")
        content = item.get("content") or item.get("text") or ""
        if not role:
            continue
        normalized.append({"role": role, "content": content})
    return normalized


class ChatSession(ABC):
    """Conversation session abstraction."""

    @abstractmethod
    async def send_message_async(self, text: str) -> str:
        """Send a message and return the model response."""


class ChatModel(ABC):
    """Chat model abstraction."""

    @abstractmethod
    def start_chat(self, history: list[Message]) -> ChatSession:
        """Start a chat session with optional history."""


class TextModel(ABC):
    """Text generation abstraction."""

    @abstractmethod
    async def generate_content_async(self, prompt: str) -> str:
        """Generate a text response."""


class StructuredModel(ABC):
    """Structured output abstraction."""

    @abstractmethod
    async def generate_content_async(self, prompt: str, response_schema: dict) -> dict:
        """Generate a JSON-structured response."""


class AIProvider(ABC):
    """Provider abstraction for AI model families."""

    @abstractmethod
    def create_chat_model(self, system_prompt: str) -> ChatModel:
        """Create a chat model."""

    @abstractmethod
    def create_reflection_model(self) -> TextModel:
        """Create a reflection text model."""

    @abstractmethod
    def create_backup_reflection_model(self) -> TextModel | None:
        """Create a backup reflection model if available."""

    @abstractmethod
    def create_concepts_model(self) -> StructuredModel:
        """Create a structured output model."""
