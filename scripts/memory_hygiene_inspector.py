"""Memory hygiene inspector script.

Analyzes LTM for short records and semantic duplicates.
"""

import chromadb
import pandas as pd
from tqdm import tqdm  # Library for nice progress bars

# --- Inspector settings ---
# Records shorter than this number of characters will be considered deletion candidates.
SHORT_MESSAGE_THRESHOLD_CHARS = 15

# Semantic similarity threshold for duplicate detection.
# Distance is the metric, lower means closer. 0 = identical.
# For standard Chroma model (L2-normalized) good threshold is < 0.2
SEMANTIC_DUPLICATE_THRESHOLD_DISTANCE = 0.2

# --- Database configuration ---
CHROMA_DB_PATH = "db"
CHROMA_COLLECTION_NAME = "stream"


def inspect_memory_hygiene():
    """
    Analyze LTM for short records and semantic duplicates.
    """
    print("--- Memory Hygiene Inspector ---")

    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME)

        # Get all data once to avoid constant database calls
        all_data = collection.get(include=["metadatas", "documents"])

        if not all_data or not all_data['ids']:
            print("Database is empty.")
            return

        print(f"Total records in database: {len(all_data['ids'])}")

        # --- Analysis #1: Short messages ---
        find_short_messages(all_data)

        # --- Analysis #2: Semantic duplicates ---
        find_semantic_duplicates(collection, all_data)

    except Exception as e:
        print(f"An error occurred: {e}")


def find_short_messages(all_data: dict):
    """Find and display too short records."""
    print("\n--- Analysis 1: Short messages ---")
    print(f"Length threshold: {SHORT_MESSAGE_THRESHOLD_CHARS} characters.\n")

    short_records = []
    for i in range(len(all_data['ids'])):
        doc = all_data['documents'][i]
        if len(doc) < SHORT_MESSAGE_THRESHOLD_CHARS:
            meta = all_data['metadatas'][i]
            short_records.append({
                'id': all_data['ids'][i],
                'access_count': meta.get('access_count', 0),
                'role': meta.get('role', 'N/A'),
                'document': doc
            })

    if not short_records:
        print("No short messages matching criteria found.")
        return

    df = pd.DataFrame(short_records)
    df_sorted = df.sort_values(by='access_count', ascending=False)

    print("Found short records (deletion candidates):")
    print(df_sorted.to_string())


def find_semantic_duplicates(collection: chromadb.Collection, all_data: dict):
    """Find and display groups of semantically similar records."""
    print("\n--- Analysis 2: Semantic duplicates ---")
    print(f"Distance threshold: < {SEMANTIC_DUPLICATE_THRESHOLD_DISTANCE}\n")

    processed_ids = set()
    duplicate_groups = []

    # Use tqdm for visibility, as the process may be long
    for i in tqdm(range(len(all_data['ids'])), desc="Searching duplicates"):
        record_id = all_data['ids'][i]

        if record_id in processed_ids:
            continue

        query_doc = all_data['documents'][i]

        # Search for 5 nearest neighbors, including the record itself
        results = collection.query(
            query_texts=[query_doc],
            n_results=5,
            include=["documents", "metadatas", "distances"]
        )

        current_group = []

        # results['distances'][0] is the list of distances for our single query
        distances = results['distances'][0]
        result_ids = results['ids'][0]

        for j in range(len(result_ids)):
            dist = distances[j]
            res_id = result_ids[j]

            # If neighbor is very close (and not the record itself)
            if dist < SEMANTIC_DUPLICATE_THRESHOLD_DISTANCE:
                if not current_group:
                    # Add original to group only if at least one duplicate found
                    original_meta = all_data['metadatas'][i]
                    current_group.append({
                        'id': record_id,
                        'distance': 0.0,  # Distance to itself
                        'access_count': original_meta.get('access_count', 0),
                        'document': query_doc
                    })

                # Check that this is not the record itself (its distance will be 0)
                if res_id != record_id:
                    res_meta = results['metadatas'][0][j]
                    current_group.append({
                        'id': res_id,
                        'distance': round(dist, 4),
                        'access_count': res_meta.get('access_count', 0),
                        'document': results['documents'][0][j]
                    })

        if len(current_group) > 1:
            duplicate_groups.append(current_group)
            # Add all IDs from found group to processed to avoid re-checking
            for rec in current_group:
                processed_ids.add(rec['id'])

    if not duplicate_groups:
        print("No semantic duplicate groups found.")
        return

    print("Found the following semantic duplicate groups:")
    for i, group in enumerate(duplicate_groups):
        print(f"\n--- Group {i + 1} ---")
        df_group = pd.DataFrame(group)
        print(df_group.to_string())


if __name__ == "__main__":
    # Install dependencies: pip install pandas tqdm
    inspect_memory_hygiene()