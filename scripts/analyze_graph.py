"""Export database to CSV.

Exports all ChromaDB collections to CSV files for Excel.
"""

import chromadb
import csv
import os
import json
import logging

try:
    from config import (
        CHROMA_DB_PATH,
        CHROMA_STREAM_COLLECTION_NAME,
        CHROMA_CONCEPTS_COLLECTION_NAME, # cognitive_assets
        CHROMA_FACTS_COLLECTION_NAME,
        CHROMA_MODALITIES_COLLECTION_NAME
    )
except ImportError:
    print("Error: Could not import settings from config.py.")
    exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
OUTPUT_DIR = "export"

def process_collection(collection: chromadb.Collection, output_filename: str):
    """
    Extract all data from collection and save to CSV file,
    optimized for Microsoft Excel.
    """
    collection_name = collection.name
    logging.info(f"Starting processing collection: '{collection_name}'...")

    try:
        data = collection.get(include=["documents", "metadatas"])
    except Exception as e:
        logging.error(f"Could not get data from collection '{collection_name}': {e}")
        return

    if not data or not data['ids']:
        logging.warning(f"Collection '{collection_name}' is empty. File will not be created.")
        return

    # --- Prepare data and headers ---
    all_metadata_keys = set()
    for metadata in data['metadatas']:
        if metadata:
            all_metadata_keys.update(metadata.keys())

    sorted_metadata_keys = sorted(list(all_metadata_keys))
    # Ensure document is always second if present
    headers = ['id']
    if 'document' in sorted_metadata_keys:
        headers.append('document')
        sorted_metadata_keys.remove('document')
    headers.extend(sorted_metadata_keys)


    rows_to_write = []
    for i in range(len(data['ids'])):
        row = {'id': data['ids'][i]}
        # Add document if present
        if data['documents'] and data['documents'][i] is not None:
             row['document'] = data['documents'][i]

        metadata = data['metadatas'][i] or {}

        # Unpack JSON fields for better readability
        for key in ['parent_ids', 'тональность']:
             if key in metadata and isinstance(metadata.get(key), str):
                try:
                    items_list = json.loads(metadata[key])
                    metadata[key] = ", ".join(items_list)
                except (json.JSONDecodeError, TypeError):
                    pass # Leave as is if invalid JSON or not a list

        row.update(metadata)
        rows_to_write.append(row)

    # --- Sort and adapt data for each collection ---
    if collection_name == CHROMA_STREAM_COLLECTION_NAME:
        rows_to_write.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
    elif collection_name == CHROMA_FACTS_COLLECTION_NAME:
        # Sort facts alphabetically for convenience
        rows_to_write.sort(key=lambda x: x.get('document', ''))
    elif collection_name == CHROMA_CONCEPTS_COLLECTION_NAME: # Cognitive assets
        # Rename 'document' to 'debug_info' for clarity
        if 'document' in headers:
            headers[headers.index('document')] = 'debug_info'
            for row in rows_to_write:
                if 'document' in row:
                    row['debug_info'] = row.pop('document')
        # Sort by parent ID to see assets from one reply together
        rows_to_write.sort(key=lambda x: x.get('parent_id', ''))
    elif collection_name == CHROMA_MODALITIES_COLLECTION_NAME:
        # Rename 'document' to 'hydrated_text'
        if 'document' in headers:
            headers[headers.index('document')] = 'hydrated_text'
            for row in rows_to_write:
                if 'document' in row:
                    row['hydrated_text'] = row.pop('document')
        # Sort by original text
        rows_to_write.sort(key=lambda x: x.get('original_text', ''))


    # --- Write to CSV file ---
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    try:
        with open(output_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
            # Ensure all keys from rows_to_write are in headers
            all_keys = set()
            for row in rows_to_write:
                all_keys.update(row.keys())
            final_headers = sorted(list(all_keys), key=lambda x: (x!='id', x!='document', x))

            writer = csv.DictWriter(csvfile, fieldnames=final_headers, delimiter=';', extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows_to_write)

        logging.info(
            f"Successfully exported {len(rows_to_write)} records from '{collection_name}' to file: {output_path}")

    except Exception as e:
        logging.error(f"Error writing CSV for '{collection_name}': {e}", exc_info=True)


def main():
    """Main export function."""
    print(f"--- Starting ChromaDB to CSV export ---")
    if not os.path.exists(CHROMA_DB_PATH):
        logging.error(f"Database directory not found: {CHROMA_DB_PATH}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection_names_map = {
            "stream": CHROMA_STREAM_COLLECTION_NAME,
            "assets": CHROMA_CONCEPTS_COLLECTION_NAME,
            "facts": CHROMA_FACTS_COLLECTION_NAME,
            "modalities": CHROMA_MODALITIES_COLLECTION_NAME
        }

        db_collections = {c.name for c in client.list_collections()}

        for name, const_name in collection_names_map.items():
            if const_name in db_collections:
                collection = client.get_collection(name=const_name)
                process_collection(collection, f"{name}.csv")
            else:
                logging.warning(f"Collection '{const_name}' not found in database. Skipping.")

        print(f"\n--- Export complete. Files saved to '{os.path.abspath(OUTPUT_DIR)}' ---")

    except Exception as e:
        logging.critical(f"Critical error during export: {e}", exc_info=True)

if __name__ == "__main__":
    main()