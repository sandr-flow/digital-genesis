"""Telegram bot command handlers."""

import logging
from aiogram import Router
from utils.keyboards import get_persistent_keyboard
from aiogram.filters import CommandStart, Command
from aiogram.types import Message, ReplyKeyboardRemove

# Command router
router = Router()

# User chat sessions (set externally)
user_chats = {}


def set_user_chats(chats_dict):
    """Set reference to user chat sessions dictionary."""
    global user_chats
    user_chats = chats_dict


@router.message(CommandStart())
async def handle_start(message: Message):
    """Handle /start command. Clears old session and greets user.

    Args:
        message: Incoming user message.
    """
    user_id = message.from_user.id
    
    if user_id in user_chats:
        del user_chats[user_id]
        logging.info(f"User {user_id} started new dialogue. Old STM session deleted.")
    
    await message.answer(
        "Digital Genesis: System ready.",
        reply_markup=get_persistent_keyboard()
    )


@router.message(Command("reset_keyboard"))
async def handle_reset_keyboard(message: Message):
    """Force reset the Reply keyboard."""
    await message.answer(
        "Keyboard forcibly reset.",
        reply_markup=ReplyKeyboardRemove()
    )


@router.message(lambda message: message.text == "🔄 Сброс контекста")
async def handle_reset_context(message: Message):
    """Handle context reset button press (ReplyKeyboard)."""
    user_id = message.from_user.id
    
    if user_id in user_chats:
        del user_chats[user_id]
        await message.answer(
            "✅ Контекст диалога сброшен",
            reply_markup=get_persistent_keyboard()
        )
        logging.info(f"Пользователь {user_id} сбросил контекст через кнопку.")
    else:
        await message.answer(
            "ℹ️ Контекст уже пуст",
            reply_markup=get_persistent_keyboard()
        )

