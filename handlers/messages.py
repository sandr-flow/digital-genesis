"""Text message handler for the Telegram bot."""

import logging
import statistics
import asyncio
from aiogram import Router, F, Bot
from aiogram.types import Message
from aiogram.enums import ChatAction, ParseMode
from aiogram.exceptions import TelegramBadRequest

import config
from services.logging_config import get_thought_logger, get_concepts_logger
from services.gemini import gemini_client
from utils.formatters import convert_to_telegram_markdown
from utils.keyboards import get_persistent_keyboard

# Message router
router = Router()

# Global references (set externally)
user_chats = {}
ltm = None
concepts_logger = None
thought_process_logger = None


def set_dependencies(chats_dict, ltm_instance):
    """Set dependencies for the message handler.

    Args:
        chats_dict: User chat sessions dictionary.
        ltm_instance: Long-term memory manager instance.
    """
    global user_chats, ltm, concepts_logger, thought_process_logger
    user_chats = chats_dict
    ltm = ltm_instance
    concepts_logger = get_concepts_logger()
    thought_process_logger = get_thought_logger()


async def run_concepts_extraction_with_wait(user_record_id: str, bot_record_id: str):
    """Run asset extraction for both records and wait for completion.

    Args:
        user_record_id: User message record ID.
        bot_record_id: Bot response record ID.
    """
    concepts_logger.info("Запуск параллельного извлечения активов для диалоговой пары...")

    async def safe_extract_assets(parent_id: str, description: str):
        """Safely extract assets with full error logging."""
        concepts_logger.info(f"=== НАЧАЛО ИЗВЛЕЧЕНИЯ АКТИВОВ ===")
        concepts_logger.info(f"Parent ID: {parent_id}")
        concepts_logger.info(f"Description: {description}")

        try:
            await ltm.extract_and_process_assets(parent_id=parent_id)
            concepts_logger.info(f"✓ Успешно завершено извлечение активов для {parent_id} ({description})")
        except Exception as e:
            concepts_logger.error(f"✗ ОШИБКА при извлечении активов для {parent_id} ({description}): {e}", exc_info=True)
            logging.error(f"КРИТИЧЕСКАЯ ОШИБКА АКТИВОВ [{parent_id}]: {e}", exc_info=True)

        concepts_logger.info(f"=== КОНЕЦ ИЗВЛЕЧЕНИЯ АКТИВОВ ===")

    # Создаем задачи
    user_task = asyncio.create_task(
        safe_extract_assets(user_record_id, "USER_MESSAGE"),
        name=f"extract_user_{user_record_id}"
    )
    bot_task = asyncio.create_task(
        safe_extract_assets(bot_record_id, "BOT_RESPONSE"),
        name=f"extract_bot_{bot_record_id}"
    )

    # Ждем завершения обеих задач
    try:
        await asyncio.gather(user_task, bot_task, return_exceptions=True)
        concepts_logger.info("Извлечение активов для диалоговой пары завершено")
    except Exception as e:
        concepts_logger.error(f"Ошибка при групповом извлечении активов: {e}", exc_info=True)


@router.message(F.text & (F.text != "🔄 Сброс контекста"))
async def handle_text_message(message: Message, bot: Bot):
    """Handle incoming text messages.

    Args:
        message: Incoming user message.
        bot: Bot instance for sending actions.
    """
    user_id = message.from_user.id
    user_text = message.text

    # Получаем или создаём сессию чата
    chat_session = user_chats.get(user_id)
    if not chat_session:
        logging.info(f"Для пользователя {user_id} не найдена активная сессия STM. Создаю новую.")
        model = gemini_client.create_chat_model()
        chat_session = model.start_chat(history=[])
        user_chats[user_id] = chat_session

    try:
        await bot.send_chat_action(message.chat.id, ChatAction.TYPING)

        # Поиск в долгосрочной памяти
        thought_memories, _ = ltm.search_and_update(
            query_text=user_text,
            n_results=config.THOUGHT_SEARCH_RESULT_COUNT,
            where_filter={"role": "internal"}
        )
        dialogue_memories, dialogue_access_counts = ltm.search_and_update(
            query_text=user_text,
            n_results=config.DIALOGUE_SEARCH_RESULT_COUNT,
            where_filter={"role": {"$in": ["user", config.AI_ROLE_NAME]}}
        )

        # Формирование контекста
        all_relevant_memories = list(dict.fromkeys(thought_memories + dialogue_memories))
        final_prompt = user_text
        if all_relevant_memories:
            memories_str = "\n".join(f"- {mem}" for mem in all_relevant_memories)
            final_prompt = config.MEMORY_PROMPT_TEMPLATE.format(
                memories=memories_str, 
                user_text=user_text
            )

        thought_process_logger.info(f"--- START DIALOGUE TURN (User: {user_id}) ---")
        thought_process_logger.info(f"User query: '{user_text}'")
        thought_process_logger.info(f"Final prompt sent to LLM:\n---\n{final_prompt}\n---")

        # Получение ответа от LLM
        response = await chat_session.send_message_async(final_prompt)
        bot_response_original = response.text

        # Отправка ответа с поддержкой Markdown
        bot_response_formatted = convert_to_telegram_markdown(bot_response_original)
        try:
            await message.answer(
                bot_response_formatted, 
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=get_persistent_keyboard()
            )
        except TelegramBadRequest as e:
            logging.warning(f"Ошибка парсинга Markdown: {e}. Отправка как простой текст.")
            await message.answer(
                bot_response_original,
                reply_markup=get_persistent_keyboard()
            )

        # Сохранение в долгосрочную память
        bot_response_ac = round(statistics.median(dialogue_access_counts)) if dialogue_access_counts else 0
        user_record_id, bot_record_id = await ltm.save_dialogue_pair(
            user_text=user_text,
            bot_text=bot_response_original,
            bot_response_access_count=bot_response_ac
        )

        thought_process_logger.info(f"Bot response: '{bot_response_original}'")
        thought_process_logger.info(f"Записи сохранены: User ID={user_record_id}, Bot ID={bot_record_id}")

        # Извлечение концептов
        concepts_logger.info("Инициируется извлечение концептов для диалоговой пары...")
        await run_concepts_extraction_with_wait(user_record_id, bot_record_id)

        thought_process_logger.info(f"--- END DIALOGUE TURN ---")

    except Exception as e:
        logging.error(f"Критическая ошибка при обработке сообщения от {user_id}: {e}", exc_info=True)
        await message.answer(
            "Произошла ошибка во время обработки вашего запроса. Попробуйте начать заново с команды /start."
        )
        if user_id in user_chats:
            del user_chats[user_id]
