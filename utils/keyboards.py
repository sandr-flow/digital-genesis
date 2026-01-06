\"\"\"Keyboard utilities for Telegram bot.\"\"\"

from aiogram.types import ReplyKeyboardMarkup, KeyboardButton


def get_persistent_keyboard() -> ReplyKeyboardMarkup:
    \"\"\"Return a persistent ReplyKeyboard displayed at the bottom of the screen.

    Contains the context reset button.
    \"\"\"
    keyboard = ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="🔄 Сброс контекста")]
        ],
        resize_keyboard=True,
        persistent=True
    )
    return keyboard
