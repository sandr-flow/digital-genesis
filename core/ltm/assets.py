# core/ltm/assets.py
"""
Модуль для извлечения и обработки когнитивных активов
"""

import logging
import json
import asyncio
import hashlib
from collections import defaultdict

import config
from core.graph import graph_manager


class AssetExtractor:
    """
    Экстрактор когнитивных активов.
    Отвечает за извлечение активов из текста, их сохранение и обновление графа.
    """
    
    def __init__(self, stream_collection, assets_collection, fact_manager, gemini_client):
        """
        Инициализирует экстрактор активов
        
        Args:
            stream_collection: Коллекция основного потока
            assets_collection: Коллекция активов
            fact_manager: Менеджер фактов и модальностей
            gemini_client: Клиент Gemini API
        """
        self.stream_collection = stream_collection
        self.assets_collection = assets_collection
        self.fact_manager = fact_manager
        self.gemini_client = gemini_client
        self._concepts_model_instance = None
    
    @staticmethod
    def _get_hash(text: str) -> str:
        """Вычисляет SHA-256 хеш от текста"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    @staticmethod
    def _get_context_phrase_for_llm(role: str) -> str:
        """
        Формирует контекстную фразу для LLM в зависимости от роли
        
        Args:
            role: Роль автора текста
            
        Returns:
            Контекстная фраза
        """
        if role == 'user':
            return "Проанализируй этот текст. Это реплика твоего собеседника (пользователя)."
        if role == 'internal':
            return "Проанализируй этот текст. Это твоя внутренняя мысль, результат рефлексии."
        if role == config.AI_ROLE_NAME:
            return "Проанализируй этот текст. Это твоя собственная реплика в диалоге."
        return f"Проанализируй этот текст. Это реплика, произнесенная {role}."
    
    def _get_concepts_model(self):
        """
        Получает или создаёт модель для извлечения концептов
        
        Returns:
            Модель Gemini или None
        """
        if self._concepts_model_instance is None:
            self._concepts_model_instance = self.gemini_client.create_concepts_model()
            if self._concepts_model_instance:
                logging.info("LTM: Независимая модель для активов успешно создана.")
        return self._concepts_model_instance
    
    async def extract_and_process_assets(self, parent_id: str):
        """
        Извлекает и обрабатывает когнитивные активы из родительской записи
        Использует Structured Output API для гарантированного получения JSON
        
        Args:
            parent_id: ID родительской записи в stream
        """
        logging.info(f"LTM: Начинаю извлечение Когнитивных Активов для родителя {parent_id}")
        
        concepts_model = self._get_concepts_model()
        if not concepts_model:
            logging.warning(f"Извлечение активов для {parent_id} пропущено: модель не инициализирована.")
            return

        parent_record = self.stream_collection.get(ids=[parent_id], include=["documents", "metadatas"])
        if not parent_record['ids']:
            logging.error(f"Не удалось найти родительскую запись {parent_id} для извлечения активов.")
            return

        raw_text = parent_record['documents'][0]
        role = parent_record['metadatas'][0].get('role', 'unknown')

        try:
            context_phrase = self._get_context_phrase_for_llm(role)
            
            # Формируем упрощенный промпт для Structured Output
            prompt = f"""{context_phrase}

Деконструируй текст на атомарные "Когнитивные Активы" - структурированные единицы знания.

Для каждого актива определи:
- кто: агент мысли ("я" или "пользователь")
- что_делает: глагол ментального действия в третьем лице единственного числа (считает, боится, надеется, предполагает, спрашивает, отрицает)
- суть: чистая суть утверждения как голый факт, без субъекта и действия
- тональность: эмоциональная окраска (массив прилагательных)
- importance: важность для анализа (1-10)
- confidence: насколько явно актив следует из текста (1-10)

ТЕКСТ ДЛЯ АНАЛИЗА:
{raw_text}
"""
            
            # Используем Structured Output с JSON schema
            response = await concepts_model.generate_content_async(
                prompt,
                generation_config={
                    "response_mime_type": "application/json",
                    "response_schema": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "кто": {
                                    "type": "string",
                                    "enum": ["я", "пользователь"],
                                    "description": "Агент мысли"
                                },
                                "что_делает": {
                                    "type": "string",
                                    "description": "Глагол ментального действия (3 лицо, ед. число)"
                                },
                                "суть": {
                                    "type": "string",
                                    "description": "Чистая суть утверждения без субъекта и действия"
                                },
                                "тональность": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Эмоциональная окраска (прилагательные)"
                                },
                                "importance": {
                                    "type": "integer",
                                    "description": "Важность для анализа (1-10)"
                                },
                                "confidence": {
                                    "type": "integer",
                                    "description": "Насколько явно актив следует из текста (1-10)"
                                }
                            },
                            "required": ["кто", "что_делает", "суть", "тональность", "importance", "confidence"]
                        }
                    }
                }
            )
            
            # С Structured Output JSON парсится автоматически
            assets_data = json.loads(response.text)
            
            if not isinstance(assets_data, list):
                logging.warning(f"LTM: Ответ модели не является JSON-массивом для {parent_id}.")
                return

            logging.info(f"LTM: Извлечено {len(assets_data)} Когнитивных Активов для {parent_id}.")

            # Обрабатываем каждый актив
            for asset_data in assets_data:
                # С Structured Output все ключи гарантированно присутствуют
                fact_text = asset_data.get("суть")
                modality_text = asset_data.get("что_делает")
                
                if not fact_text or not modality_text:
                    logging.warning(f"LTM: Пропущен Актив с пустой 'сутью' или 'что_делает': {asset_data}")
                    continue

                # Создаём или получаем факт и модальность
                fact_id = self.fact_manager.get_or_create_fact(fact_text)
                modality_id = self.fact_manager.get_or_create_modality(modality_text)
                
                # Сохраняем актив
                asset_id = self._add_or_update_cognitive_asset(asset_data, parent_id, fact_id, modality_id)
                logging.info(f"LTM: Актив сохранен/обновлен с ID: {asset_id}")

                # Обновляем граф
                await self._rebuild_graph_for_asset(asset_id)

        except json.JSONDecodeError as e:
            logging.error(f"LTM: Не удалось распарсить JSON из ответа модели: {e}", exc_info=True)
        except Exception as e:
            logging.error(f"Ошибка при извлечении/обработке активов для {parent_id}: {e}", exc_info=True)
            self._concepts_model_instance = None
            logging.warning("LTM: Экземпляр модели активов сброшен из-за ошибки.")

    def _add_or_update_cognitive_asset(self, asset_data: dict, parent_id: str, fact_id: str, modality_id: str) -> str:
        """
        Добавляет или обновляет когнитивный актив
        
        Args:
            asset_data: Данные актива
            parent_id: ID родительской записи
            fact_id: ID факта
            modality_id: ID модальности
            
        Returns:
            ID актива
        """
        asset_hash = self._get_hash(f"{fact_id}:{modality_id}")
        existing = self.assets_collection.get(where={"hash": {"$eq": asset_hash}})
        
        if existing['ids']:
            return existing['ids'][0]
        else:
            new_asset_id = f"asset_{asset_hash[:16]}"
            metadata = {
                "parent_id": parent_id,
                "hash": asset_hash,
                "fact_id": fact_id,
                "modality_id": modality_id,
                "кто": asset_data['кто'],
                "тональность": json.dumps(asset_data['тональность']),
                "importance": asset_data['importance'],
                "confidence": asset_data['confidence'],
            }
            
            # Формируем текст актива для векторизации
            asset_text = f"[{asset_data['кто']}] -> [{asset_data['что_делает']}] -> ([{asset_data['суть']}])"
            
            self.assets_collection.add(
                documents=[asset_text],
                metadatas=[metadata],
                ids=[new_asset_id]
            )
            logging.info(f"LTM: Создан новый Когнитивный Актив {new_asset_id}")
            return new_asset_id

    async def _rebuild_graph_for_asset(self, asset_id: str):
        """
        Асинхронно и эффективно обновляет граф на основе семантической близости фактов.
        
        Ключевые улучшения:
        - Истинно асинхронная работа с помощью asyncio.to_thread для блокирующих вызовов БД
        - Решена проблема "N+1 запроса" за счет пакетной загрузки данных
        - Исключены избыточные обновления рёбер для одной и той же пары узлов
        - Используются потокобезопасные методы graph_manager
        
        Args:
            asset_id: ID когнитивного актива
        """
        logging.debug(f"LTM: Начинаю обновление графа для Когнитивного Актива {asset_id}")

        # --- Шаг 1: Загрузка исходных данных об активе и его факте ---
        try:
            asset_record = await asyncio.to_thread(
                self.assets_collection.get, ids=[asset_id], include=["metadatas"]
            )
            if not asset_record.get('ids'):
                logging.warning(f"Не удалось найти актив {asset_id} для обновления графа.")
                return

            current_asset_metadata = asset_record['metadatas'][0]
            fact_id = current_asset_metadata.get('fact_id')
            parent_id = current_asset_metadata.get('parent_id')
            if not fact_id or not parent_id:
                logging.warning(f"Актив {asset_id} не имеет fact_id или parent_id. Пропуск.")
                return

            fact_record = await asyncio.to_thread(
                self.fact_manager.facts_collection.get, ids=[fact_id], include=["documents"]
            )
            if not fact_record.get('ids'):
                logging.error(f"LTM: Не удалось найти текст для факта {fact_id[:16]}. Обновление графа невозможно.")
                return
            fact_text = fact_record['documents'][0]
        except Exception as e:
            logging.error(f"LTM: Ошибка на этапе загрузки исходных данных для актива {asset_id}: {e}")
            return

        # --- Шаг 2: Поиск семантически близких фактов-соседей ---
        logging.debug(f"LTM: Ищем семантических соседей для факта '{fact_text[:50]}...'")
        neighbors = await asyncio.to_thread(
            self.fact_manager.facts_collection.query,
            query_texts=[fact_text],
            n_results=config.CONCEPT_NEIGHBOR_COUNT + 1,
            include=["distances"]
        )
        if not neighbors.get('ids') or not neighbors['ids'][0]:
            logging.warning(f"LTM: Не найдено семантических соседей для факта {fact_id[:16]}.")
            return

        # Фильтруем соседей и собираем их ID и дистанции
        neighbor_fact_data = []
        for i in range(len(neighbors['ids'][0])):
            neighbor_fact_id = neighbors['ids'][0][i]
            distance = neighbors['distances'][0][i]
            
            if neighbor_fact_id == fact_id:
                continue
            # config.GRAPH_ASSOCIATIVE_THRESHOLD - это порог схожести, а distance - расстояние
            # 1.0 - distance = схожесть
            if (1.0 - distance) < config.GRAPH_ASSOCIATIVE_THRESHOLD:
                continue
            neighbor_fact_data.append({"id": neighbor_fact_id, "distance": distance})

        if not neighbor_fact_data:
            logging.debug(f"LTM: Не найдено достаточно близких соседей для факта {fact_id[:16]}.")
            return

        # --- Шаг 3: Пакетная загрузка всех активов, связанных с найденными соседями ---
        all_neighbor_fact_ids = [n['id'] for n in neighbor_fact_data]
        all_linked_assets_records = await asyncio.to_thread(
            self.assets_collection.get,
            where={"fact_id": {"$in": all_neighbor_fact_ids}},
            include=["metadatas"]
        )

        # Группируем найденные активы по fact_id для удобного и быстрого доступа
        assets_by_fact_id = defaultdict(list)
        if all_linked_assets_records.get('ids'):
            for meta in all_linked_assets_records['metadatas']:
                assets_by_fact_id[meta['fact_id']].append(meta)

        # --- Шаг 4: Обработка и создание/обновление рёбер в графе ---
        tasks = []

        # Итерируемся по заранее отфильтрованному и подготовленному списку соседей
        for neighbor_data in neighbor_fact_data:
            neighbor_fact_id = neighbor_data['id']
            linked_assets_for_neighbor = assets_by_fact_id.get(neighbor_fact_id, [])

            if not linked_assets_for_neighbor:
                continue

            similarity = 1.0 - neighbor_data['distance']

            # Собираем уникальные ID родительских узлов для данного соседа
            neighbor_parent_ids = {
                meta.get('parent_id') for meta in linked_assets_for_neighbor
            }

            for neighbor_parent_id in neighbor_parent_ids:
                if not neighbor_parent_id or parent_id == neighbor_parent_id:
                    continue

                # Находим метаданные одного из активов-соседей
                neighbor_asset_metadata = next(
                    (meta for meta in linked_assets_for_neighbor if meta.get('parent_id') == neighbor_parent_id),
                    None
                )
                if not neighbor_asset_metadata:
                    continue

                # Создаем асинхронную задачу для обновления ребра
                task = asyncio.create_task(
                    graph_manager.add_or_update_edge(
                        parent_id,
                        neighbor_parent_id,
                        similarity,
                        current_asset_metadata,
                        neighbor_asset_metadata
                    )
                )
                tasks.append(task)

        # Дожидаемся выполнения всех задач по обновлению графа
        if tasks:
            await asyncio.gather(*tasks)
            logging.info(f"Граф обновлен для актива {asset_id}. Добавлено/обновлено {len(tasks)} рёбер.")
