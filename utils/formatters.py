\"\"\"Text formatting utilities.\"\"\"

import re


def convert_to_telegram_markdown(text: str) -> str:
    \"\"\"Convert common Markdown formats to Telegram legacy Markdown.

    Converts **bold** (from Gemini) to *bold* (Telegram legacy format).
    Also handles __italic__ to _italic_.

    Args:
        text: Source text with Markdown formatting.

    Returns:
        Text converted for Telegram.
    \"\"\"
    # Replace **bold** with *bold*
    # Use non-greedy search (.*?) to handle multiple highlights in one line
    text = re.sub(r'\\*\\*(.*?)\\*\\*', r'*\\1*', text)
    
    # Replace __italic__ with _italic__ (if model uses it)
    text = re.sub(r'__(.*?)__', r'_\\1_', text)

    # NOTE: 'Markdown' mode doesn't require escaping most characters,
    # unlike 'MarkdownV2', making it more robust for LLM output.
    # We intentionally don't escape other characters.
    return text
