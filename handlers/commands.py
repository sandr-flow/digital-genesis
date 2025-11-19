# handlers/commands.py
"""
Обработчики команд Telegram бота
"""

import logging
from aiogram import Router
from aiogram.filters import CommandStart
from aiogram.types import Message

# Создаём роутер для команд
router = Router()

# Словарь пользовательских чатов (будет передаваться извне)
user_chats = {}


def set_user_chats(chats_dict):
    """Устанавливает ссылку на словарь пользовательских чатов"""
    global user_chats
    user_chats = chats_dict


@router.message(CommandStart())
async def handle_start(message: Message):
    """
    Обработчик команды /start
    Очищает старую сессию и приветствует пользователя
    
    Args:
        message: Входящее сообщение от пользователя
    """
    user_id = message.from_user.id
    
    if user_id in user_chats:
        del user_chats[user_id]
        logging.info(f"Пользователь {user_id} начал новый диалог. Старая сессия STM удалена.")
    
    await message.answer(
        "Цифровой Генезис: Этап 4.0. Концептуальное Ядро активно. Системы в норме."
    )
