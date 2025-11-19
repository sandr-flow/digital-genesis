# core/ltm/search.py
"""
Модуль поиска и обновления записей в долгосрочной памяти
"""

import logging
import random


class SearchManager:
    """
    Менеджер поиска в долгосрочной памяти.
    Отвечает за поиск релевантных записей, формирование кластеров и охлаждение.
    """
    
    def __init__(self, stream_collection):
        """
        Инициализирует менеджер поиска
        
        Args:
            stream_collection: Коллекция ChromaDB для основного потока
        """
        self.stream_collection = stream_collection
    
    def search_and_update(self, query_text: str, n_results: int, where_filter: dict = None) -> tuple[list[str], list[int]]:
        """
        Ищет в основной коллекции stream релевантные записи, обновляет их счетчик
        доступа и возвращает их в виде отформатированных строк.
        
        Args:
            query_text: Текст запроса для поиска
            n_results: Количество результатов
            where_filter: Фильтр для поиска (опционально)
            
        Returns:
            Кортеж (список отформатированных воспоминаний, список счетчиков доступа)
        """
        if n_results == 0:
            return [], []
        if where_filter is None:
            where_filter = {}

        try:
            results = self.stream_collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where_filter
            )

            if not results or not results['ids'] or not results['ids'][0]:
                return [], []

            retrieved_ids = results['ids'][0]
            docs = results['documents'][0]
            metadatas = results['metadatas'][0]

            memories = []
            counts = []
            ids_to_update = []
            metas_to_update = []

            for i in range(len(retrieved_ids)):
                meta = metadatas[i]
                ac = meta.get('access_count', 0)

                counts.append(ac)
                # Форматируем строку для подачи в промпт
                memories.append(f"[{meta.get('role', 'unknown').capitalize()} в прошлом (ac={ac})]: {docs[i]}")

                # Готовим данные для обновления
                meta['access_count'] = ac + 1
                ids_to_update.append(retrieved_ids[i])
                metas_to_update.append(meta)

            # Выполняем пакетное обновление метаданных
            if ids_to_update:
                self.stream_collection.update(ids=ids_to_update, metadatas=metas_to_update)

            return memories, counts
        except Exception as e:
            logging.error(f"LTM Search: Ошибка во время поиска: {e}", exc_info=True)
            return [], []
    
    def get_random_hot_record_as_seed(self, min_access_count: int) -> dict | None:
        """
        Получает случайную "горячую" запись для использования в качестве зерна рефлексии.
        Вероятность выбора пропорциональна счетчику доступа.
        
        Args:
            min_access_count: Минимальный счетчик доступа
            
        Returns:
            Словарь с данными записи или None, если записи не найдены
        """
        if min_access_count <= 0:
            min_access_count = 1
            
        try:
            hot_records = self.stream_collection.get(
                where={"access_count": {"$gte": min_access_count}},
                include=["documents", "metadatas"]
            )
            if not hot_records.get('ids'):
                return None

            population = [
                {
                    "id": hot_records['ids'][i],
                    "doc": hot_records['documents'][i],
                    "meta": hot_records['metadatas'][i]
                }
                for i in range(len(hot_records['ids']))
            ]
            weights = [rec['meta'].get('access_count', 0) for rec in population]
            if not population:
                return None

            return random.choices(population=population, weights=weights, k=1)[0]
        except Exception as e:
            logging.error(f"LTM Seed: Ошибка при поиске 'зерна': {e}")
            return None

    def get_semantic_cluster(self, seed_doc: str, cluster_size: int) -> list[dict]:
        """
        Формирует семантический кластер записей вокруг заданного "зерна"
        
        Args:
            seed_doc: Текст "зерна" для поиска
            cluster_size: Размер кластера
            
        Returns:
            Список словарей с данными записей кластера
        """
        try:
            results = self.stream_collection.query(
                query_texts=[seed_doc],
                n_results=cluster_size,
                include=["documents", "metadatas"]
            )
            if not results or not results['ids'][0]:
                return []

            return [
                {
                    "id": results['ids'][0][i],
                    "doc": results['documents'][0][i],
                    "role": results['metadatas'][0][i].get('role', 'unknown'),
                    "access_count": results['metadatas'][0][i].get('access_count', 0)
                }
                for i in range(len(results['ids'][0]))
            ]
        except Exception as e:
            logging.error(f"LTM Cluster: Ошибка при поиске кластера: {e}")
            return []

    def cooldown_records_by_ids(self, ids: list[str]):
        """
        Охлаждает записи, уменьшая их счетчик доступа вдвое
        
        Args:
            ids: Список ID записей для охлаждения
        """
        if not ids:
            return
            
        try:
            records = self.stream_collection.get(ids=ids, include=["metadatas"])
            if not records.get('ids'):
                return

            ids_to_update = []
            metas_to_update = []
            
            for i in range(len(records['ids'])):
                meta = records['metadatas'][i]
                current_ac = meta.get('access_count', 0)
                if current_ac > 0:
                    meta['access_count'] = current_ac // 2
                    ids_to_update.append(records['ids'][i])
                    metas_to_update.append(meta)
                    
            if ids_to_update:
                self.stream_collection.update(ids=ids_to_update, metadatas=metas_to_update)
        except Exception as e:
            logging.error(f"LTM Cooldown: Ошибка при 'охлаждении': {e}")

    def get_records_by_ids(self, ids: list[str]) -> list[dict] | None:
        """
        Получает записи по списку ID
        
        Args:
            ids: Список ID записей
            
        Returns:
            Список словарей с данными записей или None
        """
        if not ids:
            return None
            
        try:
            records = self.stream_collection.get(ids=ids, include=["documents", "metadatas"])
            if not records or not records.get('ids'):
                return None
                
            result_list = []
            for i in range(len(records['ids'])):
                result_list.append({
                    "id": records['ids'][i],
                    "doc": records['documents'][i],
                    "role": records['metadatas'][i].get('role', 'unknown'),
                    "access_count": records['metadatas'][i].get('access_count', 0),
                    "timestamp": records['metadatas'][i].get('timestamp', 0)
                })
            return result_list
        except Exception as e:
            logging.error(f"LTM: Ошибка при получении записей по ID: {e}")
            return None
