"""Long-term memory (LTM) manager module.

Coordinates all LTM submodules and ChromaDB collections.
"""

import chromadb
import uuid
import time
import logging
import hashlib
import asyncio

from config import (
    CHROMA_DB_PATH, CHROMA_STREAM_COLLECTION_NAME,
    CHROMA_CONCEPTS_COLLECTION_NAME,
    CHROMA_FACTS_COLLECTION_NAME,
    CHROMA_MODALITIES_COLLECTION_NAME,
    AI_ROLE_NAME
)
from core.graph import graph_manager
from services.gemini import gemini_client
from .search import SearchManager
from .facts import FactManager
from .assets import AssetExtractor


class LTM_Manager:
    """Main long-term memory manager.

    Coordinates ChromaDB collections and delegates tasks to submodules.
    """
    
    def __init__(self):
        """Initialize the LTM manager and all submodules."""
        try:
            # Инициализация ChromaDB клиента
            self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            
            # Создаём или получаем коллекции
            self.stream_collection = self.client.get_or_create_collection(
                name=CHROMA_STREAM_COLLECTION_NAME
            )
            self.assets_collection = self.client.get_or_create_collection(
                name=CHROMA_CONCEPTS_COLLECTION_NAME
            )
            self.facts_collection = self.client.get_or_create_collection(
                name=CHROMA_FACTS_COLLECTION_NAME
            )
            self.modalities_collection = self.client.get_or_create_collection(
                name=CHROMA_MODALITIES_COLLECTION_NAME
            )

            logging.info(
                f"LTM Manager: Инициализировано 4 коллекции: "
                f"Stream: {self.stream_collection.count()}, "
                f"Assets: {self.assets_collection.count()}, "
                f"Facts: {self.facts_collection.count()}, "
                f"Modalities: {self.modalities_collection.count()}."
            )

            # Инициализация подмодулей
            self.search_manager = SearchManager(self.stream_collection)
            self.fact_manager = FactManager(self.facts_collection, self.modalities_collection)
            self.asset_extractor = AssetExtractor(
                self.stream_collection,
                self.assets_collection,
                self.fact_manager,
                gemini_client
            )
            
        except Exception as e:
            logging.critical(
                f"LTM Manager: Не удалось инициализировать ChromaDB! Ошибка: {e}",
                exc_info=True
            )
            raise

    @staticmethod
    def _get_hash(text: str) -> str:
        """Compute SHA-256 hash of text."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    async def _add_to_stream(self, text: str, role: str, initial_ac: int = 0) -> str:
        """Add a record to the main stream.

        Args:
            text: Record text content.
            role: Role (user, assistant, internal).
            initial_ac: Initial access count.

        Returns:
            ID of the created or existing record.
        """
        text_hash = self._get_hash(text)
        
        # Обернем I/O в to_thread для полной асинхронности
        existing = await asyncio.to_thread(
            self.stream_collection.get,
            where={"hash": {"$eq": text_hash}},
            limit=1
        )
        if existing and existing['ids']:
            logging.debug(f"LTM Stream: Найдена существующая запись с хешем {text_hash[:16]}...")
            return existing['ids'][0]

        new_id = f"{role}_{str(uuid.uuid4())}"
        metadata = {
            "role": role,
            "timestamp": time.time(),
            "access_count": initial_ac,
            "hash": text_hash
        }
        
        # Обернем I/O в to_thread
        await asyncio.to_thread(
            self.stream_collection.add,
            documents=[text],
            metadatas=[metadata],
            ids=[new_id]
        )

        # Добавляем узел в граф
        await graph_manager.add_node_if_not_exists(
            new_id,
            role=role,
            timestamp=metadata["timestamp"]
        )

        logging.info(f"LTM Stream: Создана новая запись ID {new_id} для роли '{role}'.")
        return new_id

    async def save_dialogue_pair(self, user_text: str, bot_text: str, bot_response_access_count: int = 0) -> tuple[str, str]:
        """Save a dialogue pair (user and bot messages) to long-term memory.

        Args:
            user_text: User message text.
            bot_text: Bot response text.
            bot_response_access_count: Initial access count for bot response.

        Returns:
            Tuple of (user record ID, bot record ID).
        """
        # Запускаем создание обеих записей параллельно
        user_task = asyncio.create_task(
            self._add_to_stream(text=user_text, role="user", initial_ac=0)
        )
        bot_task = asyncio.create_task(
            self._add_to_stream(text=bot_text, role=AI_ROLE_NAME, initial_ac=bot_response_access_count)
        )

        user_id, bot_id = await asyncio.gather(user_task, bot_task)
        return user_id, bot_id

    async def save_reflection(self, reflection_text: str, initial_access_count: int = 0) -> str:
        """Save a reflection result to long-term memory.

        Args:
            reflection_text: Reflection text content.
            initial_access_count: Initial access count.

        Returns:
            Reflection record ID.
        """
        reflection_id = await self._add_to_stream(
            text=reflection_text,
            role="internal",
            initial_ac=initial_access_count
        )
        return reflection_id

    # Delegate search methods
    def search_and_update(self, query_text: str, n_results: int, where_filter: dict = None) -> tuple[list[str], list[int]]:
        """Delegate search to SearchManager."""
        return self.search_manager.search_and_update(query_text, n_results, where_filter)

    def get_random_hot_record_as_seed(self, min_access_count: int) -> dict | None:
        """Delegate seed retrieval to SearchManager."""
        return self.search_manager.get_random_hot_record_as_seed(min_access_count)

    def get_semantic_cluster(self, seed_doc: str, cluster_size: int) -> list[dict]:
        """Delegate cluster formation to SearchManager."""
        return self.search_manager.get_semantic_cluster(seed_doc, cluster_size)

    def cooldown_records_by_ids(self, ids: list[str]):
        """Delegate record cooldown to SearchManager."""
        self.search_manager.cooldown_records_by_ids(ids)

    def get_records_by_ids(self, ids: list[str]) -> list[dict] | None:
        """Delegate record retrieval by ID to SearchManager."""
        return self.search_manager.get_records_by_ids(ids)

    # Delegate asset extraction methods
    async def extract_and_process_assets(self, parent_id: str):
        """Delegate asset extraction to AssetExtractor."""
        await self.asset_extractor.extract_and_process_assets(parent_id)
