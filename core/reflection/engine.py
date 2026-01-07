"""Reflection engine module - background thinking system."""

import logging
import statistics
from services.logging_config import get_thought_logger, get_reflections_logger, get_concepts_logger
from services.ai.base import AIProvider
import config


class ReflectionEngine:
    """Reflection engine for background system thinking.

    Analyzes hot records and generates insights.
    """
    
    def __init__(self, ltm_manager, provider: AIProvider):
        """Initialize the reflection engine.

        Args:
            ltm_manager: Long-term memory manager instance.
            provider: AI provider instance.
        """
        self.ltm = ltm_manager
        self.provider = provider
        self.thought_logger = get_thought_logger()
        self.reflections_logger = get_reflections_logger()
        self.concepts_logger = get_concepts_logger()
    
    async def run_cycle(self):
        """Execute one reflection cycle with full error handling."""
        try:
            self.thought_logger.info("--- START FOCUSED REFLECTION CYCLE ---")
            self.concepts_logger.info("REFLECTION: Searching for hot records...")

            # Check required configurations
            if not hasattr(config, 'REFLECTION_MIN_ACCESS_COUNT'):
                self.concepts_logger.error("REFLECTION: REFLECTION_MIN_ACCESS_COUNT not found in config")
                return

            if not hasattr(config, 'REFLECTION_CLUSTER_SIZE'):
                self.concepts_logger.error("REFLECTION: REFLECTION_CLUSTER_SIZE not found in config")
                return

            if not hasattr(config, 'REFLECTION_PROMPT_TEMPLATE'):
                self.concepts_logger.error("REFLECTION: REFLECTION_PROMPT_TEMPLATE not found in config")
                return

            # Get reflection seed
            seed = self.ltm.get_random_hot_record_as_seed(config.REFLECTION_MIN_ACCESS_COUNT)
            if not seed:
                self.thought_logger.info("No hot records to serve as a seed. Skipping reflection.")
                self.concepts_logger.info("REFLECTION: No hot records found, skipping cycle")
                return

            self.thought_logger.info(f"Reflection seed chosen: '{seed['doc'][:80]}...'")
            self.concepts_logger.info(f"REFLECTION: Seed chosen: ID={seed['id']}")

            # Form semantic cluster
            reflection_cluster = self.ltm.get_semantic_cluster(
                seed_doc=seed['doc'], 
                cluster_size=config.REFLECTION_CLUSTER_SIZE
            )
            if not reflection_cluster:
                self.thought_logger.info("Could not form a semantic cluster around the seed. Skipping.")
                self.concepts_logger.info("REFLECTION: Could not form semantic cluster")
                return

            self.concepts_logger.info(f"REFLECTION: Cluster formed with {len(reflection_cluster)} records")

            # Form reflection prompt
            try:
                memories_for_prompt = []
                for mem in reflection_cluster:
                    role = mem.get('role', 'unknown')
                    access_count = mem.get('access_count', 0)
                    doc = mem.get('doc', '')
                    memories_for_prompt.append(f"[{role.capitalize()} (ac={access_count})]: {doc}")

                memories_str = "\n".join(f"- {mem}" for mem in memories_for_prompt)
                reflection_prompt = config.REFLECTION_PROMPT_TEMPLATE.format(hot_memories=memories_str)
                self.concepts_logger.info(f"REFLECTION: Prompt formed, length {len(reflection_prompt)} chars")
            except Exception as e:
                self.concepts_logger.error(f"REFLECTION: Error forming prompt: {e}", exc_info=True)
                return

            # Generate thought
            thought_text = await self._generate_thought(reflection_prompt)
            if not thought_text:
                return

            # Save and process result
            await self._save_and_process(thought_text, reflection_cluster)

            self.thought_logger.info("--- END FOCUSED REFLECTION CYCLE ---")
            self.concepts_logger.info("REFLECTION: Cycle completed successfully")

        except Exception as e:
            logging.error(f"CRITICAL ERROR in reflection cycle: {e}", exc_info=True)
            self.concepts_logger.error(f"REFLECTION: CRITICAL ERROR: {e}", exc_info=True)

    async def _generate_thought(self, reflection_prompt: str) -> str | None:
        """Generate a thought using the main or backup model.

        Args:
            reflection_prompt: Reflection prompt.

        Returns:
            Generated thought text or None on error.
        """
        thought_text = None
        
        try:
            self.concepts_logger.info("REFLECTION: Sending request to main model...")
            
            reflection_model = self.provider.create_reflection_model()
            thought_text = await reflection_model.generate_content_async(reflection_prompt)
            self.concepts_logger.info("REFLECTION: Response received from main model")

        except Exception as e:
            logging.error(f"Reflection error with main model: {e}", exc_info=True)
            self.concepts_logger.warning(f"REFLECTION: Main model error, switching to backup: {e}")

            # Check backup model availability
            try:
                backup_model = self.provider.create_backup_reflection_model()
                if not backup_model:
                    self.concepts_logger.error("REFLECTION: Backup model not configured")
                    return None
                thought_text = await backup_model.generate_content_async(reflection_prompt)
                self.concepts_logger.info("REFLECTION: Response received from backup model")
            except Exception as e2:
                logging.error(f"Reflection failed with backup model: {e2}", exc_info=True)
                self.concepts_logger.error(f"REFLECTION: Critical error, cycle aborted: {e2}")
                return None

        return thought_text

    async def _save_and_process(self, thought_text: str, reflection_cluster: list):
        """Save reflection and process the cluster.

        Args:
            thought_text: Generated thought text.
            reflection_cluster: Cluster of records that spawned the thought.
        """
        if not thought_text or not thought_text.strip():
            self.concepts_logger.warning("REFLECTION: Empty or invalid thought text received")
            return

        self.thought_logger.info(f"Generated thought: '{thought_text}'")
        self.reflections_logger.info(thought_text)
        self.concepts_logger.info(f"REFLECTION: Thought generated, length {len(thought_text)} chars")

        try:
            # Calculate initial access count
            parent_counts = [mem.get('access_count', 0) for mem in reflection_cluster]
            initial_thought_ac = round(statistics.median(parent_counts)) if parent_counts else 0

            # Save reflection
            reflection_id = await self.ltm.save_reflection(
                reflection_text=thought_text,
                initial_access_count=initial_thought_ac
            )
            self.concepts_logger.info(f"REFLECTION: Saved with ID={reflection_id}")

            # Extract assets for reflection
            self.concepts_logger.info("REFLECTION: Starting asset extraction...")
            await self._safe_extract_assets(reflection_id, "REFLECTION")

            # Cool down records
            cluster_ids = [rec.get('id') for rec in reflection_cluster if rec.get('id')]
            if cluster_ids:
                self.ltm.cooldown_records_by_ids(cluster_ids)
                self.concepts_logger.info(f"REFLECTION: Cooled down {len(cluster_ids)} cluster records")
            else:
                self.concepts_logger.warning("REFLECTION: No IDs for record cooldown")

        except Exception as e:
            self.concepts_logger.error(f"REFLECTION: Error saving reflection: {e}", exc_info=True)

    async def _safe_extract_assets(self, parent_id: str, description: str):
        """Safely extract assets with full error logging.

        Args:
            parent_id: Parent record ID.
            description: Record description for logging.
        """
        self.concepts_logger.info(f"=== START ASSET EXTRACTION ===")
        self.concepts_logger.info(f"Parent ID: {parent_id}")
        self.concepts_logger.info(f"Description: {description}")

        try:
            await self.ltm.extract_and_process_assets(parent_id=parent_id)
            self.concepts_logger.info(f"Asset extraction completed for {parent_id} ({description})")
        except Exception as e:
            self.concepts_logger.error(
                f"ERROR extracting assets for {parent_id} ({description}): {e}", 
                exc_info=True
            )
            logging.error(f"CRITICAL ASSET ERROR [{parent_id}]: {e}", exc_info=True)

        self.concepts_logger.info(f"=== END ASSET EXTRACTION ===")
