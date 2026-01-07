"""Entry point for the Digital Genesis Telegram bot.

This module initializes the bot, sets up handlers, configures
the reflection scheduler, and starts the polling loop.
"""

import asyncio
import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from aiogram import Bot, Dispatcher

import config
from services.logging_config import setup_logging, get_concepts_logger
from services.ai.gateway import AIProviderGateway
from core.ltm import LTM_Manager
from core.graph import graph_manager
from core.reflection.engine import ReflectionEngine
from handlers import commands, messages


# --- Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

provider_config = config.AI_PROVIDER_CONFIG.get(config.AI_PROVIDER, {})
provider_api_key = provider_config.get("api_key")

if not config.TELEGRAM_BOT_TOKEN or not provider_api_key:
    raise ValueError(
        "TELEGRAM_BOT_TOKEN and the provider API key must be set in the .env file"
    )

# Initialize bot and dispatcher
bot = Bot(token=config.TELEGRAM_BOT_TOKEN)
dp = Dispatcher()

# User chat sessions
user_chats = {}

# Setup logging
setup_logging()
concepts_logger = get_concepts_logger()

# AI provider gateway and instances
gateway = AIProviderGateway(
    config.AI_PROVIDER,
    provider_config,
    config.AI_RATE_LIMIT_RPS,
    config.AI_REQUEST_TIMEOUT_SECONDS,
)
ai_provider = gateway.get_provider()

# LTM and reflection engine
ltm = LTM_Manager(ai_provider)
reflection_engine = ReflectionEngine(ltm, ai_provider)


async def main():
    """Start all bot components and run the polling loop."""
    
    # specific Concept system diagnostics at startup
    concepts_logger.info("=== CONCEPT SYSTEM DIAGNOSTICS ===")
    concepts_logger.info(f"AI_PROVIDER: {config.AI_PROVIDER}")
    concepts_logger.info(
        f"AI_CONCEPTS_MODEL_NAME: "
        f"{provider_config.get('concepts_model', 'NOT SET')}"
    )
    concepts_logger.info(
        f"CONCEPT_EXTRACTION_PROMPT length: "
        f"{len(getattr(config, 'CONCEPT_EXTRACTION_PROMPT', ''))}"
    )
    concepts_logger.info("=======================================")

    # Set dependencies for handlers
    commands.set_user_chats(user_chats)
    messages.set_dependencies(user_chats, ltm, ai_provider)
    
    # Register handler routers
    dp.include_router(commands.router)
    dp.include_router(messages.router)

    # Background task scheduler setup
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
    logging.info("Background task scheduler started.")

    try:
        logging.info("Starting Telegram polling...")
        await dp.start_polling(bot)
    finally:
        logging.info("Stopping polling...")
        scheduler.shutdown()
        logging.info("Scheduler stopped.")
        logging.info("Performing final graph save...")
        graph_manager.save_graph()
        logging.info("System 'Digital Genesis' stopped.")


if __name__ == '__main__':
    logging.info("Starting 'Digital Genesis' system...")
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logging.info("Bot stopped manually.")
