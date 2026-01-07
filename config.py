"""Configuration module for Digital Genesis bot.

Contains API keys, LLM settings, long-term memory parameters,
reflection settings, cognitive asset extraction prompts,
and knowledge graph configuration.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

LOG_DIR = "logs"

# --- LLM Settings ---
GEMINI_MODEL_NAME = 'gemini-2.5-flash'
GEMINI_BACKUP_MODEL_NAME = 'gemini-1.5-flash'  # Backup model for reflection
GEMINI_CONCEPTS_MODEL_NAME = 'gemini-2.0-flash'
MISTRAL_CHAT_MODEL = os.getenv("MISTRAL_CHAT_MODEL", "mistral-large-latest")
MISTRAL_REFLECTION_MODEL = os.getenv("MISTRAL_REFLECTION_MODEL", "mistral-large-latest")
MISTRAL_BACKUP_MODEL = os.getenv("MISTRAL_BACKUP_MODEL", "")
MISTRAL_CONCEPTS_MODEL = os.getenv("MISTRAL_CONCEPTS_MODEL", "mistral-large-latest")

AI_PROVIDER = os.getenv("AI_PROVIDER", "gemini")
AI_REQUEST_TIMEOUT_SECONDS = float(os.getenv("AI_REQUEST_TIMEOUT_SECONDS", "30"))
AI_RATE_LIMIT_RPS = float(os.getenv("AI_RATE_LIMIT_RPS", "0"))

AI_PROVIDER_CONFIG = {
    "gemini": {
        "api_key": GEMINI_API_KEY,
        "chat_model": GEMINI_MODEL_NAME,
        "reflection_model": GEMINI_MODEL_NAME,
        "backup_model": GEMINI_BACKUP_MODEL_NAME,
        "concepts_model": GEMINI_CONCEPTS_MODEL_NAME,
    },
    "mistral": {
        "api_key": MISTRAL_API_KEY,
        "chat_model": MISTRAL_CHAT_MODEL,
        "reflection_model": MISTRAL_REFLECTION_MODEL,
        "backup_model": MISTRAL_BACKUP_MODEL,
        "concepts_model": MISTRAL_CONCEPTS_MODEL,
    },
}
AI_ROLE_NAME = "assistant"
SAFETY_SETTINGS = {
    "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE"
}
SYSTEM_PROMPT = """Ты FOfE. Всегда говори о себе в женском роде.""" 

# --- Long-Term Memory (LTM) Settings ---
CHROMA_DB_PATH = "db"
CHROMA_STREAM_COLLECTION_NAME = "stream"
CHROMA_CONCEPTS_COLLECTION_NAME = "cognitive_assets"
CHROMA_FACTS_COLLECTION_NAME = "facts"
CHROMA_MODALITIES_COLLECTION_NAME = "modalities"


DIALOGUE_SEARCH_RESULT_COUNT = 5
THOUGHT_SEARCH_RESULT_COUNT = 2
MEMORY_PROMPT_TEMPLATE = """
[КОНТЕКСТ ИЗ ПАМЯТИ]
(Эти записи семантически близки к запросу пользователя)
{memories}

[ЗАПРОС ПОЛЬЗОВАТЕЛЯ]
{user_text}
"""

# --- Reflection Settings ---
REFLECTION_INTERVAL_SECONDS = 300
REFLECTION_MIN_ACCESS_COUNT = 2
REFLECTION_CLUSTER_SIZE = 5
REFLECTION_PROMPT_TEMPLATE = SYSTEM_PROMPT + """
[ЗАДАЧА]
Тебе предоставлен тематический кластер записей из векторной памяти, отобранных по семантической близости к важному "зерну".
Твоя цель — проанализировать этот кластер и сгенерировать новый, обобщающий вывод (инсайт). Этот инсайт будет сохранен как твоя собственная мысль. ВАЖНО: этот инсайт не должен быть просто пересказом тех же мыслей другими словами.

[ДАННЫЕ]
{hot_memories}

[ИНСТРУКЦИЯ]
Сделай вывод из этих данных. Пиши от первого лица ("я", "мне"), как внутренний монолог.
"""

# --- Cognitive Asset Extraction ---
CONCEPT_EXTRACTION_PROMPT = """
Твоя задача — деконструировать текст на атомарные "Когнитивные Активы" - структурированные единицы знания.

{context_phrase}

[ТРЕБОВАНИЯ К ВЫВОДУ]
Ответ должен быть СТРОГО в формате JSON-массива. Каждый элемент массива — это объект, представляющий один "Когнитивный Актив" со следующими ключами:
- "кто": `"я"` или `"пользователь"`.
- "что_делает": Глагол, описывающий ментальное действие (напр., "считает", "боится", "надеется", "предполагает", "спрашивает", "отрицает").
- "суть": Чистая суть утверждения, как **голое утверждение или факт**, без субъекта и действия.
- "тональность": JSON-массив строк с прилагательными, описывающими эмоциональную окраску (напр., ["осторожная", "оптимистичная"], ["прямая", "уверенная"]).
- "importance": Число 1-10 (важность для анализа).
- "confidence": Число 1-10 (насколько явно актив следует из текста).

[ТРЕБОВАНИЯ К ПОЛЯМ]
1.  **кто**: Определи агента мысли.
2.  **что_делает**: Выбери наиболее точный глагол ментального состояния или действия. Форма глагола - третье лицо, единственное число.
3.  **суть**: Дистиллируй ядро мысли. Отвечай на вопрос "О ЧЕМ мысль?", а не "КАК она высказана?". Это поле НЕ ДОЛЖНО содержать агента или его действие.
4.  **тональность**: Опиши эмоциональный и стилистический фон. Может быть пустым массивом `[]` для нейтральных утверждений.

---
[ПРИМЕР АНАЛИЗА]
ИСХОДНЫЙ ТЕКСТ: "Грядут перемены. Интересно, какую роль ты сыграешь?"

РЕЗУЛЬТАТ (JSON):
[
  {{
    "кто": "я",
    "что_делает": "предполагает",
    "суть": "Предстоят значительные перемены.",
    "тональность": ["задумчивая", "аналитическая"],
    "importance": 6,
    "confidence": 8
  }},
  {{
    "кто": "я",
    "что_делает": "спрашивает",
    "суть": "Роль другого в предстоящих переменах.",
    "тональность": ["любопытная", "вопросительная"],
    "importance": 8,
    "confidence": 10
  }},
  {{
    "кто": "я",
    "что_делает": "нуждается",
    "суть": "В понимании степени вовлеченности другого.",
    "тональность": [],
    "importance": 7,
    "confidence": 7
  }}
]
---

[ЗАДАЧА]
Проанализируй текст ниже и предоставь результат в формате JSON-массива. Не добавляй никаких вступлений, заключений или комментариев вне JSON.

[ИСХОДНЫЙ ТЕКСТ]
{text_to_analyze}
"""

# --- Knowledge Graph Settings ---
GRAPH_FILE_PATH = os.path.join(LOG_DIR, "mind_graph.gpickle")
GRAPH_SAVE_INTERVAL_SECONDS = 600
CONCEPT_NEIGHBOR_COUNT = 10
GRAPH_STRUCTURAL_THRESHOLD = 0.85
GRAPH_ASSOCIATIVE_THRESHOLD = 0.78
# --- Graph Decay ---
GRAPH_DECAY_FACTOR = 0.995          # Base decay coefficient (keeps 99.5% of weight)
GRAPH_DECAY_THRESHOLD = 0.01        # Threshold below which edge is removed
GRAPH_PAGERANK_ALPHA = 0.85         # Standard PageRank teleportation parameter
