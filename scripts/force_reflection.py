"""Force reflection script (interactive version).

Manually triggers reflection cycles on specific record IDs.
"""

import asyncio
import logging
import sys
import google.generativeai as genai
import statistics
import os

import config
from ltm import ltm

# --- Logger setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
genai.configure(api_key=config.GEMINI_API_KEY)

thought_process_logger = logging.getLogger("ThoughtProcess")
thought_process_logger.setLevel(logging.INFO)
thought_process_logger.propagate = False
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - [ForceReflection] - %(message)s'))
thought_process_logger.addHandler(handler)

os.makedirs(config.LOG_DIR, exist_ok=True)
reflections_logger = logging.getLogger("Reflections")
reflections_logger.setLevel(logging.INFO)
reflections_logger.propagate = False
reflections_file_handler = logging.FileHandler(
    os.path.join(config.LOG_DIR, "reflections.log"),
    encoding='utf-8'
)
# Use special formatter to distinguish forced thoughts
reflections_file_handler.setFormatter(
    logging.Formatter('%(asctime)s\n[Forced Reflection from ID: %(seed_id)s]\n%(message)s\n' + '-' * 80)
)
reflections_logger.addHandler(reflections_file_handler)


async def force_reflection_on_id(seed_id: str):
    """
    Main forced reflection logic for a single ID.
    """
    thought_process_logger.info(f"--- START FORCED REFLECTION CYCLE (Seed ID: {seed_id}) ---")

    # 1. Get "seed" by ID
    seed_data = ltm.get_records_by_ids([seed_id])
    if not seed_data:
        thought_process_logger.error(f"Could not find record with ID: {seed_id}")
        print(f"\nError: Record with ID '{seed_id}' not found in database.")
        return

    seed = {
        "id": seed_data[0]['id'],
        "doc": seed_data[0]['doc'],
        "meta": {
            'role': seed_data[0]['role'],
            'access_count': seed_data[0]['access_count'],
            'timestamp': seed_data[0].get('timestamp')
        }
    }

    thought_process_logger.info(
        f"Reflection seed loaded: '{seed['doc'][:80]}...' (ac={seed['meta'].get('access_count')})")

    # 2. Find semantic cluster
    reflection_cluster = ltm.get_semantic_cluster(
        seed_doc=seed['doc'],
        cluster_size=config.REFLECTION_CLUSTER_SIZE
    )
    if not reflection_cluster:
        thought_process_logger.info("Could not form a semantic cluster around the seed. Skipping.")
        print("\nCould not form a semantic cluster around this 'seed'.")
        return

    # 3. Prepare prompt
    memories_for_prompt = [f"[{mem['role'].capitalize()} (ac={mem['access_count']})]: {mem['doc']}" for mem in
                           reflection_cluster]
    memories_str = "\n".join(f"- {mem}" for mem in memories_for_prompt)
    reflection_prompt = config.REFLECTION_PROMPT_TEMPLATE.format(hot_memories=memories_str)

    thought_process_logger.info(f"Formed a cluster of {len(reflection_cluster)} memories.")
    thought_process_logger.info(f"Reflection prompt sent to LLM:\n---\n{reflection_prompt}\n---")

    try:
        # 4. Act of Reflection
        print("\nSending reflection request to LLM...")
        reflection_model = genai.GenerativeModel(config.GEMINI_MODEL_NAME)
        response = await reflection_model.generate_content_async(reflection_prompt)
        thought_text = response.text

        # 5. Calculate "weight"
        parent_counts = [mem['access_count'] for mem in reflection_cluster]
        initial_thought_ac = round(statistics.median(parent_counts)) if parent_counts else 0

        thought_process_logger.info(f"Generated thought: '{thought_text}'")
        print(f"\nGenerated new thought:\n---\n{thought_text}\n---")

        # 6. Write to diary
        class SeedIdAdapter(logging.LoggerAdapter):
            def process(self, msg, kwargs):
                kwargs['extra'] = {'seed_id': self.extra['seed_id']}
                return msg, kwargs

        adapter = SeedIdAdapter(reflections_logger, {'seed_id': seed_id})
        adapter.info(thought_text)

        # 7. Save "Thought" to LTM
        ltm.save_reflection(reflection_text=thought_text, initial_access_count=initial_thought_ac)

        # 8. "Cooldown" memory
        cluster_ids_to_cooldown = [rec['id'] for rec in reflection_cluster]
        ltm.cooldown_records_by_ids(cluster_ids_to_cooldown)

        print(f"\nForced reflection for ID {seed_id} completed successfully.")
        print(f"New thought saved to LTM and to {os.path.join(config.LOG_DIR, 'reflections.log')}")

    except Exception as e:
        logging.error(f"Act of Reflection: Error generating or saving thought: {e}. Memory 'cooldown' cancelled.")
        print(f"\nError occurred during reflection: {e}")

    thought_process_logger.info("--- END FORCED REFLECTION CYCLE ---")


async def interactive_reflection_session():
    """
    Start an interactive session, requesting ID from user in a loop.
    """
    print("--- Interactive Forced Reflection Session ---")
    print("Enter record ID to start reflection.")
    print("To exit, enter 'exit' or 'quit'.\n")

    while True:
        # Request ID from user
        target_id = input("Enter seed record ID: ").strip()

        # Check exit command
        if target_id.lower() in ['exit', 'quit', 'q']:
            print("Ending session.")
            break

        # Check that ID is not empty
        if not target_id:
            continue

        # Run async reflection function
        await force_reflection_on_id(target_id)

        print("\n" + "=" * 50 + "\n")  # Separator for next iteration


if __name__ == "__main__":
    # Start interactive session
    try:
        asyncio.run(interactive_reflection_session())
    except KeyboardInterrupt:
        print("\nSession interrupted by user.")
        sys.exit(0)