# main.py
"""
Точка входа для Telegram-бота "Цифровой Генезис"
Архитектура v4.0 - Модульная структура
"""

import asyncio
import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from aiogram import Bot, Dispatcher

import config
from services.logging_config import setup_logging, get_concepts_logger
from services.gemini import gemini_client
from core.ltm import ltm
from core.graph import graph_manager
from core.reflection.engine import ReflectionEngine
from handlers import commands, messages


# --- Настройка ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

if not config.TELEGRAM_BOT_TOKEN or not config.GEMINI_API_KEY:
    raise ValueError(
        "Необходимо установить TELEGRAM_BOT_TOKEN и GEMINI_API_KEY в .env файле"
    )

# Инициализация бота и диспетчера
bot = Bot(token=config.TELEGRAM_BOT_TOKEN)
dp = Dispatcher()

# Словарь пользовательских чатов
user_chats = {}

# Настройка логирования
setup_logging()
concepts_logger = get_concepts_logger()

# Движок рефлексии
reflection_engine = ReflectionEngine(ltm)


async def main():
    """Основная асинхронная функция, которая запускает все компоненты."""
    
    # Проверяем настройки концептов при запуске
    concepts_logger.info("=== ДИАГНОСТИКА СИСТЕМЫ КОНЦЕПТОВ ===")
    concepts_logger.info(
        f"GEMINI_CONCEPTS_API_KEY установлен: "
        f"{bool(getattr(config, 'GEMINI_CONCEPTS_API_KEY', None))}"
    )
    concepts_logger.info(
        f"GEMINI_CONCEPTS_MODEL_NAME: "
        f"{getattr(config, 'GEMINI_CONCEPTS_MODEL_NAME', 'НЕ ЗАДАН')}"
    )
    concepts_logger.info(
        f"CONCEPT_EXTRACTION_PROMPT длина: "
        f"{len(getattr(config, 'CONCEPT_EXTRACTION_PROMPT', ''))}"
    )
    concepts_logger.info("=======================================")

    # Настраиваем зависимости для обработчиков
    commands.set_user_chats(user_chats)
    messages.set_dependencies(user_chats, ltm)
    
    # Регистрируем роутеры обработчиков
    dp.include_router(commands.router)
    dp.include_router(messages.router)

    # Настройка планировщика фоновых задач
    scheduler = AsyncIOScheduler(timezone="Europe/Moscow")
    scheduler.add_job(
        reflection_engine.run_cycle,
        'interval',
        seconds=config.REFLECTION_INTERVAL_SECONDS,
        id='reflection_job'
    )
    scheduler.add_job(
        graph_manager.save_graph,
        'interval',
        seconds=config.GRAPH_SAVE_INTERVAL_SECONDS,
        id='save_graph_job'
    )
    scheduler.start()
    logging.info("Планировщик фоновых задач запущен.")

    try:
        logging.info("Запуск поллинга Telegram...")
        await dp.start_polling(bot)
    finally:
        logging.info("Остановка поллинга...")
        scheduler.shutdown()
        logging.info("Планировщик остановлен.")
        logging.info("Выполняется финальное сохранение графа...")
        graph_manager.save_graph()
        logging.info("Система 'Цифровой Генезис' остановлена.")


if __name__ == '__main__':
    logging.info("Запуск системы 'Цифровой Генезис'...")
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logging.info("Бот остановлен вручную.")