# core/ltm/facts.py
"""
Модуль для работы с фактами и модальностями
"""

import logging
import hashlib


class FactManager:
    """
    Менеджер для работы с фактами и модальностями.
    Обеспечивает дедупликацию и управление этими сущностями.
    """
    
    def __init__(self, facts_collection, modalities_collection):
        """
        Инициализирует менеджер фактов
        
        Args:
            facts_collection: Коллекция ChromaDB для фактов
            modalities_collection: Коллекция ChromaDB для модальностей
        """
        self.facts_collection = facts_collection
        self.modalities_collection = modalities_collection
    
    @staticmethod
    def _get_hash(text: str) -> str:
        """
        Вычисляет SHA-256 хеш от текста
        
        Args:
            text: Исходный текст
            
        Returns:
            Хеш текста
        """
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def get_or_create_fact(self, fact_text: str) -> str:
        """
        Получает существующий факт или создаёт новый.
        Использует хеш для дедупликации.
        
        Args:
            fact_text: Текст факта
            
        Returns:
            ID факта
        """
        fact_id = self._get_hash(fact_text)
        
        if not self.facts_collection.get(ids=[fact_id])['ids']:
            self.facts_collection.add(documents=[fact_text], ids=[fact_id])
            logging.info(f"LTM: Создан новый Факт '{fact_text[:50]}...' (ID: {fact_id[:16]})")
            
        return fact_id

    def get_or_create_modality(self, modality_text: str) -> str:
        """
        Получает существующую модальность или создаёт новую.
        Модальность представляет ментальное действие (считает, боится, надеется и т.д.)
        
        Args:
            modality_text: Текст модальности (глагол)
            
        Returns:
            ID модальности
        """
        modality_id = self._get_hash(modality_text)
        
        if not self.modalities_collection.get(ids=[modality_id])['ids']:
            # Добавляем префикс для лучшей векторизации
            hydrated_text = f"ментальное действие: {modality_text}"
            self.modalities_collection.add(
                documents=[hydrated_text],
                metadatas=[{"original_text": modality_text}],
                ids=[modality_id]
            )
            logging.info(f"LTM: Создана новая Модальность '{modality_text}' (ID: {modality_id[:16]})")
            
        return modality_id
