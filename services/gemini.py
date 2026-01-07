"""Gemini API client module."""

import logging
import threading
import google.generativeai as genai
import config


class GeminiClient:
    """Wrapper for Gemini API.

    Manages model creation with thread-safe configuration.
    """
    
    def __init__(self):
        """Initialize Gemini client with thread-safe lock."""
        self._config_lock = threading.Lock()
    
    def create_chat_model(self):
        """Create a model for dialogue with system prompt.

        Returns:
            GenerativeModel for dialogue.
        """
        with self._config_lock:
            genai.configure(api_key=config.GEMINI_API_KEY)
            return genai.GenerativeModel(
                model_name=config.GEMINI_MODEL_NAME,
                system_instruction=config.SYSTEM_PROMPT,
                safety_settings=config.SAFETY_SETTINGS
            )
    
    def create_reflection_model(self):
        """Create a model for reflection.

        Returns:
            GenerativeModel for reflection.
        """
        with self._config_lock:
            genai.configure(api_key=config.GEMINI_API_KEY)
            return genai.GenerativeModel(
                model_name=config.GEMINI_MODEL_NAME,
                safety_settings=config.SAFETY_SETTINGS
            )
    
    def create_backup_reflection_model(self):
        """Create a backup model for reflection.

        Returns:
            GenerativeModel backup for reflection.
        """
        with self._config_lock:
            genai.configure(api_key=config.GEMINI_API_KEY)
            return genai.GenerativeModel(
                model_name=config.GEMINI_BACKUP_MODEL_NAME,
                safety_settings=config.SAFETY_SETTINGS
            )
    
    def create_concepts_model(self):
        """Create a model for cognitive asset extraction.

        Returns:
            GenerativeModel for concept extraction or None.
        """
        if not config.GEMINI_API_KEY:
            logging.error("Concepts model requires GEMINI_API_KEY to be set.")
            return None
        
        logging.info(f"Creating concepts model '{config.GEMINI_CONCEPTS_MODEL_NAME}'")
        
        with self._config_lock:
            genai.configure(api_key=config.GEMINI_API_KEY)
            return genai.GenerativeModel(
                model_name=config.GEMINI_CONCEPTS_MODEL_NAME,
                safety_settings=config.SAFETY_SETTINGS
            )


# Global client instance
gemini_client = GeminiClient()
