"""Reflection engine module - background thinking system."""

import logging
import statistics
from services.logging_config import get_thought_logger, get_reflections_logger, get_concepts_logger
from services.gemini import gemini_client
import config


class ReflectionEngine:
    """Reflection engine for background system thinking.

    Analyzes hot records and generates insights.
    """
    
    def __init__(self, ltm_manager):
        """Initialize the reflection engine.

        Args:
            ltm_manager: Long-term memory manager instance.
        """
        self.ltm = ltm_manager
        self.thought_logger = get_thought_logger()
        self.reflections_logger = get_reflections_logger()
        self.concepts_logger = get_concepts_logger()
    
    async def run_cycle(self):
        """Execute one reflection cycle with full error handling."""
        try:
            self.thought_logger.info("--- START FOCUSED REFLECTION CYCLE ---")
            self.concepts_logger.info("🔄 РЕФЛЕКСИЯ: Поиск горячих записей для рефлексии...")

            # Проверяем наличие необходимых конфигураций
            if not hasattr(config, 'REFLECTION_MIN_ACCESS_COUNT'):
                self.concepts_logger.error("🔄 РЕФЛЕКСИЯ: REFLECTION_MIN_ACCESS_COUNT не найден в конфигурации")
                return

            if not hasattr(config, 'REFLECTION_CLUSTER_SIZE'):
                self.concepts_logger.error("🔄 РЕФЛЕКСИЯ: REFLECTION_CLUSTER_SIZE не найден в конфигурации")
                return

            if not hasattr(config, 'REFLECTION_PROMPT_TEMPLATE'):
                self.concepts_logger.error("🔄 РЕФЛЕКСИЯ: REFLECTION_PROMPT_TEMPLATE не найден в конфигурации")
                return

            # Получаем "зерно" для рефлексии
            seed = self.ltm.get_random_hot_record_as_seed(config.REFLECTION_MIN_ACCESS_COUNT)
            if not seed:
                self.thought_logger.info("No hot records to serve as a seed. Skipping reflection.")
                self.concepts_logger.info("🔄 РЕФЛЕКСИЯ: Горячие записи не найдены, пропуск цикла")
                return

            self.thought_logger.info(f"Reflection seed chosen: '{seed['doc'][:80]}...'")
            self.concepts_logger.info(f"🔄 РЕФЛЕКСИЯ: Выбрано зерно рефлексии: ID={seed['id']}")

            # Формируем семантический кластер
            reflection_cluster = self.ltm.get_semantic_cluster(
                seed_doc=seed['doc'], 
                cluster_size=config.REFLECTION_CLUSTER_SIZE
            )
            if not reflection_cluster:
                self.thought_logger.info("Could not form a semantic cluster around the seed. Skipping.")
                self.concepts_logger.info("🔄 РЕФЛЕКСИЯ: Не удалось сформировать семантический кластер")
                return

            self.concepts_logger.info(f"🔄 РЕФЛЕКСИЯ: Сформирован кластер из {len(reflection_cluster)} записей")

            # Формируем промпт для рефлексии
            try:
                memories_for_prompt = []
                for mem in reflection_cluster:
                    role = mem.get('role', 'unknown')
                    access_count = mem.get('access_count', 0)
                    doc = mem.get('doc', '')
                    memories_for_prompt.append(f"[{role.capitalize()} (ac={access_count})]: {doc}")

                memories_str = "\n".join(f"- {mem}" for mem in memories_for_prompt)
                reflection_prompt = config.REFLECTION_PROMPT_TEMPLATE.format(hot_memories=memories_str)
                self.concepts_logger.info(f"🔄 РЕФЛЕКСИЯ: Сформирован промпт длиной {len(reflection_prompt)} символов")
            except Exception as e:
                self.concepts_logger.error(f"🔄 РЕФЛЕКСИЯ: Ошибка при формировании промпта: {e}", exc_info=True)
                return

            # Генерируем мысль
            thought_text = await self._generate_thought(reflection_prompt)
            if not thought_text:
                return

            # Сохраняем и обрабатываем результат
            await self._save_and_process(thought_text, reflection_cluster)

            self.thought_logger.info("--- END FOCUSED REFLECTION CYCLE ---")
            self.concepts_logger.info("🔄 РЕФЛЕКСИЯ: Цикл рефлексии завершен успешно")

        except Exception as e:
            logging.error(f"КРИТИЧЕСКАЯ ОШИБКА в цикле рефлексии: {e}", exc_info=True)
            self.concepts_logger.error(f"🔄 РЕФЛЕКСИЯ: КРИТИЧЕСКАЯ ОШИБКА цикла: {e}", exc_info=True)

    async def _generate_thought(self, reflection_prompt: str) -> str | None:
        """Generate a thought using the main or backup model.

        Args:
            reflection_prompt: Reflection prompt.

        Returns:
            Generated thought text or None on error.
        """
        thought_text = None
        
        try:
            self.concepts_logger.info("🔄 РЕФЛЕКСИЯ: Отправка запроса к основной модели...")
            
            reflection_model = gemini_client.create_reflection_model()
            response = await reflection_model.generate_content_async(reflection_prompt)
            thought_text = response.text
            self.concepts_logger.info("🔄 РЕФЛЕКСИЯ: Получен ответ от основной модели")

        except Exception as e:
            logging.error(f"Reflection error with main model: {e}", exc_info=True)
            self.concepts_logger.warning(f"🔄 РЕФЛЕКСИЯ: Ошибка основной модели, переключение на резервную: {e}")

            # Проверяем наличие резервной модели
            if not hasattr(config, 'GEMINI_BACKUP_MODEL_NAME'):
                self.concepts_logger.error("🔄 РЕФЛЕКСИЯ: GEMINI_BACKUP_MODEL_NAME не найден в конфигурации")
                return None

            try:
                backup_model = gemini_client.create_backup_reflection_model()
                response = await backup_model.generate_content_async(reflection_prompt)
                thought_text = response.text
                self.concepts_logger.info("🔄 РЕФЛЕКСИЯ: Получен ответ от резервной модели")
            except Exception as e2:
                logging.error(f"Reflection failed with backup model: {e2}", exc_info=True)
                self.concepts_logger.error(f"🔄 РЕФЛЕКСИЯ: Критическая ошибка, цикл прерван: {e2}")
                return None

        return thought_text

    async def _save_and_process(self, thought_text: str, reflection_cluster: list):
        """Save reflection and process the cluster.

        Args:
            thought_text: Generated thought text.
            reflection_cluster: Cluster of records that spawned the thought.
        """
        if not thought_text or not thought_text.strip():
            self.concepts_logger.warning("🔄 РЕФЛЕКСИЯ: Получен пустой или некорректный текст мысли")
            return

        self.thought_logger.info(f"Generated thought: '{thought_text}'")
        self.reflections_logger.info(thought_text)
        self.concepts_logger.info(f"🔄 РЕФЛЕКСИЯ: Сгенерирована мысль длиной {len(thought_text)} символов")

        try:
            # Рассчитываем начальный счетчик доступа
            parent_counts = [mem.get('access_count', 0) for mem in reflection_cluster]
            initial_thought_ac = round(statistics.median(parent_counts)) if parent_counts else 0

            # Сохраняем рефлексию
            reflection_id = await self.ltm.save_reflection(
                reflection_text=thought_text,
                initial_access_count=initial_thought_ac
            )
            self.concepts_logger.info(f"🔄 РЕФЛЕКСИЯ: Рефлексия сохранена с ID={reflection_id}")

            # Извлечение активов для рефлексии
            self.concepts_logger.info("🔄 РЕФЛЕКСИЯ: Запуск извлечения активов...")
            await self._safe_extract_assets(reflection_id, "REFLECTION")

            # Охлаждение записей
            cluster_ids = [rec.get('id') for rec in reflection_cluster if rec.get('id')]
            if cluster_ids:
                self.ltm.cooldown_records_by_ids(cluster_ids)
                self.concepts_logger.info(f"🔄 РЕФЛЕКСИЯ: Выполнено охлаждение {len(cluster_ids)} записей кластера")
            else:
                self.concepts_logger.warning("🔄 РЕФЛЕКСИЯ: Нет ID для охлаждения записей")

        except Exception as e:
            self.concepts_logger.error(f"🔄 РЕФЛЕКСИЯ: Ошибка при сохранении рефлексии: {e}", exc_info=True)

    async def _safe_extract_assets(self, parent_id: str, description: str):
        """Safely extract assets with full error logging.

        Args:
            parent_id: Parent record ID.
            description: Record description for logging.
        """
        self.concepts_logger.info(f"=== НАЧАЛО ИЗВЛЕЧЕНИЯ АКТИВОВ ===")
        self.concepts_logger.info(f"Parent ID: {parent_id}")
        self.concepts_logger.info(f"Description: {description}")

        try:
            await self.ltm.extract_and_process_assets(parent_id=parent_id)
            self.concepts_logger.info(f"✓ Успешно завершено извлечение активов для {parent_id} ({description})")
        except Exception as e:
            self.concepts_logger.error(
                f"✗ ОШИБКА при извлечении активов для {parent_id} ({description}): {e}", 
                exc_info=True
            )
            logging.error(f"КРИТИЧЕСКАЯ ОШИБКА АКТИВОВ [{parent_id}]: {e}", exc_info=True)

        self.concepts_logger.info(f"=== КОНЕЦ ИЗВЛЕЧЕНИЯ АКТИВОВ ===")
