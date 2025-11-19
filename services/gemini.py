# services/gemini.py
"""
Клиент для работы с Gemini API
"""

import logging
import google.generativeai as genai
import config


class GeminiClient:
    """
    Обёртка для работы с Gemini API.
    Управляет созданием моделей и обработкой ошибок.
    """
    
    def __init__(self):
        """Инициализирует клиент Gemini"""
        self._main_api_configured = False
        self._concepts_api_configured = False
        
    def _configure_main_api(self):
        """Настраивает основной API ключ"""
        if not self._main_api_configured:
            genai.configure(api_key=config.GEMINI_API_KEY)
            self._main_api_configured = True
            
    def _configure_concepts_api(self):
        """Настраивает API ключ для концептов"""
        if not self._concepts_api_configured:
            genai.configure(api_key=config.GEMINI_CONCEPTS_API_KEY)
            self._concepts_api_configured = True
    
    def create_chat_model(self):
        """
        Создаёт модель для диалога с системным промптом
        
        Returns:
            GenerativeModel: Модель для диалога
        """
        self._configure_main_api()
        return genai.GenerativeModel(
            config.GEMINI_MODEL_NAME,
            system_instruction=config.SYSTEM_PROMPT,
            safety_settings=config.SAFETY_SETTINGS
        )
    
    def create_reflection_model(self):
        """
        Создаёт модель для рефлексии
        
        Returns:
            GenerativeModel: Модель для рефлексии
        """
        self._configure_main_api()
        return genai.GenerativeModel(
            model_name=config.GEMINI_MODEL_NAME,
            safety_settings=config.SAFETY_SETTINGS
        )
    
    def create_backup_reflection_model(self):
        """
        Создаёт резервную модель для рефлексии
        
        Returns:
            GenerativeModel: Резервная модель для рефлексии
        """
        self._configure_main_api()
        return genai.GenerativeModel(
            model_name=config.GEMINI_BACKUP_MODEL_NAME,
            safety_settings=config.SAFETY_SETTINGS
        )
    
    def create_concepts_model(self):
        """
        Создаёт модель для извлечения когнитивных активов
        
        Returns:
            GenerativeModel: Модель для извлечения концептов
        """
        if not config.GEMINI_CONCEPTS_API_KEY:
            logging.error("API-ключ для концептов (GEMINI_CONCEPTS_API_KEY) не найден!")
            return None
            
        self._configure_concepts_api()
        logging.info(f"Создание модели '{config.GEMINI_CONCEPTS_MODEL_NAME}' для активов")
        
        return genai.GenerativeModel(
            model_name=config.GEMINI_CONCEPTS_MODEL_NAME,
            safety_settings=config.SAFETY_SETTINGS
        )


# Глобальный экземпляр клиента
gemini_client = GeminiClient()
