"""Cognitive asset extraction and processing module."""

import logging
import json
import asyncio
import hashlib
from collections import defaultdict

import config
from core.graph import graph_manager


class AssetExtractor:
    """Cognitive asset extractor.

    Handles asset extraction from text, storage, and graph updates.
    """
    
    def __init__(self, stream_collection, assets_collection, fact_manager, gemini_client):
        """Initialize the asset extractor.

        Args:
            stream_collection: Main stream collection.
            assets_collection: Assets collection.
            fact_manager: Facts and modalities manager.
            gemini_client: Gemini API client.
        """
        self.stream_collection = stream_collection
        self.assets_collection = assets_collection
        self.fact_manager = fact_manager
        self.gemini_client = gemini_client
        self._concepts_model_instance = None
    
    @staticmethod
    def _get_hash(text: str) -> str:
        """Compute SHA-256 hash of text."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    @staticmethod
    def _get_context_phrase_for_llm(role: str) -> str:
        """Generate a context phrase for LLM based on role.

        Args:
            role: Text author role.

        Returns:
            Context phrase.
        """
        if role == 'user':
            return "Проанализируй этот текст. Это реплика твоего собеседника (пользователя)."
        if role == 'internal':
            return "Проанализируй этот текст. Это твоя внутренняя мысль, результат рефлексии."
        if role == config.AI_ROLE_NAME:
            return "Проанализируй этот текст. Это твоя собственная реплика в диалоге."
        return f"Проанализируй этот текст. Это реплика, произнесенная {role}."
    
    def _get_concepts_model(self):
        """Get or create the concepts extraction model.

        Returns:
            Gemini model or None.
        """
        if self._concepts_model_instance is None:
            self._concepts_model_instance = self.gemini_client.create_concepts_model()
            if self._concepts_model_instance:
                logging.info("LTM: Independent model for assets created successfully.")
        return self._concepts_model_instance
    
    async def extract_and_process_assets(self, parent_id: str):
        """Extract and process cognitive assets from a parent record.

        Uses Structured Output API for guaranteed JSON response.

        Args:
            parent_id: Parent record ID in stream.
        """
        logging.info(f"LTM: Starting Cognitive Asset extraction for parent {parent_id}")
        
        concepts_model = self._get_concepts_model()
        if not concepts_model:
            logging.warning(f"Asset extraction for {parent_id} skipped: model not initialized.")
            return

        parent_record = self.stream_collection.get(ids=[parent_id], include=["documents", "metadatas"])
        if not parent_record['ids']:
            logging.error(f"Could not find parent record {parent_id} for asset extraction.")
            return

        raw_text = parent_record['documents'][0]
        role = parent_record['metadatas'][0].get('role', 'unknown')

        try:
            context_phrase = self._get_context_phrase_for_llm(role)
            
            # Form simplified prompt for Structured Output
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
            
            # Use Structured Output with JSON schema
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
            
            # With Structured Output, JSON parses automatically
            assets_data = json.loads(response.text)
            
            if not isinstance(assets_data, list):
                logging.warning(f"LTM: Model response is not a JSON array for {parent_id}.")
                return

            logging.info(f"LTM: Extracted {len(assets_data)} Cognitive Assets for {parent_id}.")

            # Process each asset
            for asset_data in assets_data:
                # With Structured Output all keys are guaranteed present
                fact_text = asset_data.get("суть")
                modality_text = asset_data.get("что_делает")
                
                if not fact_text or not modality_text:
                    logging.warning(f"LTM: Skipped Asset with empty 'essence' or 'action': {asset_data}")
                    continue

                # Create or get fact and modality
                fact_id = self.fact_manager.get_or_create_fact(fact_text)
                modality_id = self.fact_manager.get_or_create_modality(modality_text)
                
                # Save asset
                asset_id = self._add_or_update_cognitive_asset(asset_data, parent_id, fact_id, modality_id)
                logging.info(f"LTM: Asset saved/updated with ID: {asset_id}")

                # Update graph
                await self._rebuild_graph_for_asset(asset_id)

        except json.JSONDecodeError as e:
            logging.error(f"LTM: Failed to parse JSON from model response: {e}", exc_info=True)
        except Exception as e:
            logging.error(f"Error extracting/processing assets for {parent_id}: {e}", exc_info=True)
            self._concepts_model_instance = None
            logging.warning("LTM: Assets model instance reset due to error."))

    def _add_or_update_cognitive_asset(self, asset_data: dict, parent_id: str, fact_id: str, modality_id: str) -> str:
        """Add or update a cognitive asset.

        Args:
            asset_data: Asset data.
            parent_id: Parent record ID.
            fact_id: Fact ID.
            modality_id: Modality ID.

        Returns:
            Asset ID.
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
            
            # Form asset text for vectorization
            asset_text = f"[{asset_data['кто']}] -> [{asset_data['что_делает']}] -> ([{asset_data['суть']}])"
            
            self.assets_collection.add(
                documents=[asset_text],
                metadatas=[metadata],
                ids=[new_asset_id]
            )
            logging.info(f"LTM: Created new Cognitive Asset {new_asset_id}")
            return new_asset_id

    async def _rebuild_graph_for_asset(self, asset_id: str):
        """Asynchronously update the graph based on semantic fact similarity.

        Key improvements:
        - Truly async with asyncio.to_thread for blocking DB calls
        - Solved N+1 query problem with batch data loading
        - Eliminated redundant edge updates for same node pairs
        - Uses thread-safe graph_manager methods

        Args:
            asset_id: Cognitive asset ID.
        """
        logging.debug(f"LTM: Starting graph update for Cognitive Asset {asset_id}")

        # --- Step 1: Load initial asset and fact data ---
        try:
            asset_record = await asyncio.to_thread(
                self.assets_collection.get, ids=[asset_id], include=["metadatas"]
            )
            if not asset_record.get('ids'):
                logging.warning(f"Could not find asset {asset_id} for graph update.")
                return

            current_asset_metadata = asset_record['metadatas'][0]
            fact_id = current_asset_metadata.get('fact_id')
            parent_id = current_asset_metadata.get('parent_id')
            if not fact_id or not parent_id:
                logging.warning(f"Asset {asset_id} has no fact_id or parent_id. Skipping.")
                return

            fact_record = await asyncio.to_thread(
                self.fact_manager.facts_collection.get, ids=[fact_id], include=["documents"]
            )
            if not fact_record.get('ids'):
                logging.error(f"LTM: Could not find text for fact {fact_id[:16]}. Graph update impossible.")
                return
            fact_text = fact_record['documents'][0]
        except Exception as e:
            logging.error(f"LTM: Error loading initial data for asset {asset_id}: {e}")
            return

        # --- Step 2: Search for semantically similar neighbor facts ---
        logging.debug(f"LTM: Searching for semantic neighbors for fact '{fact_text[:50]}...'")
        neighbors = await asyncio.to_thread(
            self.fact_manager.facts_collection.query,
            query_texts=[fact_text],
            n_results=config.CONCEPT_NEIGHBOR_COUNT + 1,
            include=["distances"]
        )
        if not neighbors.get('ids') or not neighbors['ids'][0]:
            logging.warning(f"LTM: No semantic neighbors found for fact {fact_id[:16]}.")
            return

        # Filter neighbors and collect their IDs and distances
        neighbor_fact_data = []
        for i in range(len(neighbors['ids'][0])):
            neighbor_fact_id = neighbors['ids'][0][i]
            distance = neighbors['distances'][0][i]
            
            if neighbor_fact_id == fact_id:
                continue
            # config.GRAPH_ASSOCIATIVE_THRESHOLD is similarity threshold, distance is distance
            # 1.0 - distance = similarity
            if (1.0 - distance) < config.GRAPH_ASSOCIATIVE_THRESHOLD:
                continue
            neighbor_fact_data.append({"id": neighbor_fact_id, "distance": distance})

        if not neighbor_fact_data:
            logging.debug(f"LTM: No sufficiently close neighbors found for fact {fact_id[:16]}.")
            return

        # --- Step 3: Batch load all assets linked to found neighbors ---
        all_neighbor_fact_ids = [n['id'] for n in neighbor_fact_data]
        all_linked_assets_records = await asyncio.to_thread(
            self.assets_collection.get,
            where={"fact_id": {"$in": all_neighbor_fact_ids}},
            include=["metadatas"]
        )

        # Group found assets by fact_id for convenient and fast access
        assets_by_fact_id = defaultdict(list)
        if all_linked_assets_records.get('ids'):
            for meta in all_linked_assets_records['metadatas']:
                assets_by_fact_id[meta['fact_id']].append(meta)

        # --- Step 4: Process and create/update edges in graph ---
        tasks = []

        # Iterate through pre-filtered and prepared neighbor list
        for neighbor_data in neighbor_fact_data:
            neighbor_fact_id = neighbor_data['id']
            linked_assets_for_neighbor = assets_by_fact_id.get(neighbor_fact_id, [])

            if not linked_assets_for_neighbor:
                continue

            similarity = 1.0 - neighbor_data['distance']

            # Collect unique parent node IDs for this neighbor
            neighbor_parent_ids = {
                meta.get('parent_id') for meta in linked_assets_for_neighbor
            }

            for neighbor_parent_id in neighbor_parent_ids:
                if not neighbor_parent_id or parent_id == neighbor_parent_id:
                    continue

                # Find metadata of one of the neighbor assets
                neighbor_asset_metadata = next(
                    (meta for meta in linked_assets_for_neighbor if meta.get('parent_id') == neighbor_parent_id),
                    None
                )
                if not neighbor_asset_metadata:
                    continue

                # Create async task for edge update
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

        # Wait for all graph update tasks to complete
        if tasks:
            await asyncio.gather(*tasks)
            logging.info(f"Graph updated for asset {asset_id}. Added/updated {len(tasks)} edges.")
