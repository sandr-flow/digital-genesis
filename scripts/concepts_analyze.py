"""Concepts inspector script.

Analyzes the "Conceptual Core" for semantic duplicates.
"""

import chromadb
import pandas as pd
from tqdm import tqdm
import json  # For JSON string handling
import os

# --- Inspector settings ---
# Semantic similarity threshold for duplicate concept detection.
# Distance is the metric, lower means closer.
# For concepts, threshold should be stricter than for stream,
# as we're looking for nearly synonymous statements.
# Good starting value: 0.1
SEMANTIC_DUPLICATE_THRESHOLD_DISTANCE = 0.1

# --- Project configuration ---
# Use variables from config.py for consistency
try:
    from config import CHROMA_DB_PATH, CHROMA_CONCEPTS_COLLECTION_NAME
except ImportError:
    print("Could not import config.py. Using default values.")
    CHROMA_DB_PATH = "db"
    CHROMA_CONCEPTS_COLLECTION_NAME = "concepts"


def inspect_concepts_hygiene():
    """
    Analyze "Conceptual Core" (concepts collection)
    for semantic duplicates.
    """
    print("--- Conceptual Core Hygiene Inspector ---")

    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_collection(name=CHROMA_CONCEPTS_COLLECTION_NAME)

        # Get all data once to avoid constant database calls
        all_data = collection.get(include=["metadatas", "documents"])

        if not all_data or not all_data['ids']:
            print("Concepts collection is empty.")
            return

        print(f"Total concepts in database: {len(all_data['ids'])}")

        # --- Main analysis: Semantic duplicates of concepts ---
        find_semantic_duplicates(collection, all_data)

    except Exception as e:
        print(f"Critical error occurred: {e}", exc_info=True)


def find_semantic_duplicates(collection: chromadb.Collection, all_data: dict):
    """Find and display groups of semantically similar concepts."""
    print("\n--- Searching for semantic duplicates ---")
    print(f"Distance threshold: < {SEMANTIC_DUPLICATE_THRESHOLD_DISTANCE}\n")

    # Create map for quick metadata access by ID
    metadata_map = {all_data['ids'][i]: all_data['metadatas'][i] for i in range(len(all_data['ids']))}
    document_map = {all_data['ids'][i]: all_data['documents'][i] for i in range(len(all_data['ids']))}

    processed_ids = set()
    duplicate_groups_found = 0

    for record_id in tqdm(all_data['ids'], desc="Analyzing concepts"):
        if record_id in processed_ids:
            continue

        query_doc = document_map[record_id]

        # Search for 5 nearest neighbors. Usually enough for concepts.
        results = collection.query(
            query_texts=[query_doc],
            n_results=5,
            include=["metadatas", "distances"]
        )

        current_group = []

        # Collect all candidates that pass the threshold
        for j in range(len(results['ids'][0])):
            dist = results['distances'][0][j]
            res_id = results['ids'][0][j]

            if dist < SEMANTIC_DUPLICATE_THRESHOLD_DISTANCE:
                # Get metadata and parent count
                meta = results['metadatas'][0][j]
                try:
                    # parent_ids is stored as JSON string
                    num_parents = len(json.loads(meta.get('parent_ids', '[]')))
                except (json.JSONDecodeError, TypeError):
                    num_parents = 0

                current_group.append({
                    'id': res_id,
                    'distance': round(dist, 4),
                    'num_parents': num_parents,
                    'document': collection.get(ids=[res_id])['documents'][0]
                })

        # If group has more than one element, we found duplicates
        if len(current_group) > 1:
            duplicate_groups_found += 1
            print(f"\n\n--- Found Duplicate Group #{duplicate_groups_found} ---")
            df_group = pd.DataFrame(current_group)
            # Sort within group by distance for clarity
            print(df_group.sort_values(by='distance').to_string(index=False))

            # Add all IDs from found group to processed to avoid re-checking
            for rec in current_group:
                processed_ids.add(rec['id'])

    if duplicate_groups_found == 0:
        print("\nNo semantic duplicate groups matching threshold found.")


if __name__ == "__main__":
    # Make sure you have dependencies installed: pip install pandas tqdm chromadb
    inspect_concepts_hygiene()