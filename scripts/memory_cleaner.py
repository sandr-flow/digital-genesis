"""Memory cleaner script.

Cleans and optimizes LTM by removing short/unused messages and merging duplicates.
"""

import chromadb
import pandas as pd
from tqdm import tqdm
import time

# --- Settings ---
SHORT_MESSAGE_THRESHOLD_CHARS = 15
# Delete short messages with access_count <= this value
SHORT_MESSAGE_MAX_AC = 1

SEMANTIC_DUPLICATE_THRESHOLD_DISTANCE = 0.2
# --- Database configuration ---
CHROMA_DB_PATH = "db"
CHROMA_COLLECTION_NAME = "stream"


def confirm_action(prompt: str) -> bool:
    """Request user confirmation for an action."""
    while True:
        response = input(f"{prompt} (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            return True
        if response in ['n', 'no']:
            return False
        print("Please enter 'y' or 'n'.")


def run_memory_hygiene():
    """Main function for LTM cleanup and optimization."""
    print("--- Starting Memory Hygiene Procedure ---")

    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME)

        # --- Stage 1: Sanitary cleanup ---
        clean_short_messages(collection)

        # --- Stage 2: Intelligent merging ---
        merge_semantic_duplicates(collection)

        print("\n--- Memory Hygiene Procedure completed. ---")

    except Exception as e:
        print(f"\nCritical error occurred: {e}")


def clean_short_messages(collection: chromadb.Collection):
    """Find and offer to delete short and 'cold' messages."""
    print("\n--- Stage 1: Searching for short and unused messages ---")

    try:
        # Get all data. May be slow for large databases, but reliable.
        all_data = collection.get(include=["metadatas", "documents"])

        ids_to_delete = []
        short_records_for_display = []

        for i in range(len(all_data['ids'])):
            doc = all_data['documents'][i]
            meta = all_data['metadatas'][i]
            if (len(doc) < SHORT_MESSAGE_THRESHOLD_CHARS and
                    meta.get('access_count', 0) <= SHORT_MESSAGE_MAX_AC):
                ids_to_delete.append(all_data['ids'][i])
                short_records_for_display.append({
                    'id': all_data['ids'][i],
                    'access_count': meta.get('access_count', 0),
                    'document': doc
                })

        if not ids_to_delete:
            print("No short/unused messages found for deletion.")
            return

        print(f"Found {len(ids_to_delete)} candidate records for deletion:")
        df = pd.DataFrame(short_records_for_display)
        print(df.to_string())

        if confirm_action("\nDelete these records?"):
            collection.delete(ids=ids_to_delete)
            print(f"Successfully deleted {len(ids_to_delete)} records.")
        else:
            print("Deletion cancelled.")

    except Exception as e:
        print(f"Error during sanitary cleanup: {e}")


def merge_semantic_duplicates(collection: chromadb.Collection):
    """Find, offer and perform merging of semantic duplicates."""
    print("\n--- Stage 2: Searching and merging semantic duplicates ---")

    all_data = collection.get(include=["metadatas", "documents"])
    if not all_data or not all_data['ids']:
        print("Database is empty, merging not possible.")
        return

    processed_ids = set()

    for i in tqdm(range(len(all_data['ids'])), desc="Analyzing duplicates"):
        record_id = all_data['ids'][i]
        if record_id in processed_ids:
            continue

        query_doc = all_data['documents'][i]

        results = collection.query(
            query_texts=[query_doc],
            n_results=10,  # Search for more neighbors for reliability
            include=["documents", "metadatas", "distances"]
        )

        group_candidates = []
        for j in range(len(results['ids'][0])):
            dist = results['distances'][0][j]
            if dist < SEMANTIC_DUPLICATE_THRESHOLD_DISTANCE:
                group_candidates.append({
                    'id': results['ids'][0][j],
                    'distance': round(dist, 4),
                    'metadata': results['metadatas'][0][j],
                    'document': results['documents'][0][j]
                })

        if len(group_candidates) <= 1:
            processed_ids.add(record_id)
            continue

        print("\n\n-----------------------------------------------------")
        print(f"Found group of {len(group_candidates)} semantic duplicates:")
        df_group = pd.DataFrame([{
            'id': rec['id'],
            'role': rec['metadata'].get('role'),
            'ac': rec['metadata'].get('access_count'),
            'timestamp': pd.to_datetime(rec['metadata'].get('timestamp'), unit='s'),
            'document': rec['document']
        } for rec in group_candidates])
        print(df_group.to_string())

        # Determine strategy
        internal_count = sum(1 for rec in group_candidates if rec['metadata'].get('role') == 'internal')
        is_thought_group = internal_count / len(group_candidates) > 0.5

        leader = None
        if is_thought_group:
            print("\nStrategy: 'Novelty' (keep the freshest thought).")
            leader = max(group_candidates, key=lambda x: x['metadata'].get('timestamp', 0))
        else:
            print("\nStrategy: 'Popularity' (keep record with max access_count).")
            leader = max(group_candidates, key=lambda x: x['metadata'].get('access_count', 0))

        ids_to_delete = [rec['id'] for rec in group_candidates if rec['id'] != leader['id']]
        total_ac = sum(rec['metadata'].get('access_count', 0) for rec in group_candidates)

        print("\nOperation plan:")
        print(f"  - Keep ID: {leader['id']}")
        print(f"  - Text: '{leader['document'][:80]}...'")
        print(f"  - Assign new access_count: {total_ac}")
        print(f"  - Delete IDs: {ids_to_delete}")

        if confirm_action("Apply this merge?"):
            try:
                # Update leader
                leader_meta = leader['metadata']
                leader_meta['access_count'] = total_ac
                collection.update(ids=[leader['id']], metadatas=[leader_meta])

                # Delete the rest
                if ids_to_delete:
                    collection.delete(ids=ids_to_delete)

                print("Merge successfully completed.")
            except Exception as e:
                print(f"Error during merge: {e}")
        else:
            print("Merge cancelled for this group.")

        # Add all IDs from group to processed to avoid re-analyzing
        for rec in group_candidates:
            processed_ids.add(rec['id'])


if __name__ == "__main__":
    # Install dependencies: pip install pandas tqdm
    run_memory_hygiene()