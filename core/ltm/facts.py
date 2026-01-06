"""Facts and modalities management module."""

import logging
import hashlib


class FactManager:
    """Manager for facts and modalities.

    Handles deduplication and management of these entities.
    """
    
    def __init__(self, facts_collection, modalities_collection):
        """Initialize the fact manager.

        Args:
            facts_collection: ChromaDB collection for facts.
            modalities_collection: ChromaDB collection for modalities.
        """
        self.facts_collection = facts_collection
        self.modalities_collection = modalities_collection
    
    @staticmethod
    def _get_hash(text: str) -> str:
        """Compute SHA-256 hash of text.

        Args:
            text: Source text.

        Returns:
            Text hash.
        """
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def get_or_create_fact(self, fact_text: str) -> str:
        """Get an existing fact or create a new one.

        Uses hash for deduplication.

        Args:
            fact_text: Fact text.

        Returns:
            Fact ID.
        """
        fact_id = self._get_hash(fact_text)
        
        if not self.facts_collection.get(ids=[fact_id])['ids']:
            self.facts_collection.add(documents=[fact_text], ids=[fact_id])
            logging.info(f"LTM: Создан новый Факт '{fact_text[:50]}...' (ID: {fact_id[:16]})")
            
        return fact_id

    def get_or_create_modality(self, modality_text: str) -> str:
        """Get an existing modality or create a new one.

        A modality represents a mental action (believes, fears, hopes, etc.).

        Args:
            modality_text: Modality text (verb).

        Returns:
            Modality ID.
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
