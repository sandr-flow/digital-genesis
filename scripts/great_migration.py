"""ChromaDB diagnostics script.

Tests ChromaDB connection and embedding model functionality.
"""

import chromadb
import time
import os

# Make sure the database path matches your config.py
DB_PATH = "db"
COLLECTION_NAME = "test_collection"

print(f"--- STARTING CHROMA DB DIAGNOSTICS ---")
print(f"Database path: {os.path.abspath(DB_PATH)}")

try:
    # 1. Initialize client
    client = chromadb.PersistentClient(path=DB_PATH)
    print("Step 1: ChromaDB client initialized successfully.")

    # 2. Create or get test collection
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    print(f"Step 2: Collection '{COLLECTION_NAME}' obtained/created. Current record count: {collection.count()}")

    # Clean up test record if leftover from previous run
    collection.delete(ids=["test-doc-1"])

    # 3. Add one document. At this point ChromaDB SHOULD start
    # downloading/initializing the embedding model.
    print("Step 3: Adding test document 'This is a test'. Please wait, this may take up to a minute...")
    collection.add(
        documents=["This is a test"],
        metadatas=[{'source': 'test'}],
        ids=["test-doc-1"]
    )
    print("Step 3.1: .add() command executed. Starting result verification.")

    # 4. Wait and check if embedding appeared
    embedding_result = None
    for i in range(45):
        print(f"  Attempt {i + 1}/45: Checking for embedding...")
        record = collection.get(ids=["test-doc-1"], include=["embeddings"])
        if record and record['embeddings'] and record['embeddings'][0] is not None:
            print("\n[SUCCESS] Embedding created successfully.")
            embedding_result = record['embeddings'][0]
            break
        time.sleep(1)

    # 5. Output final result
    print("\n--- DIAGNOSTICS RESULT ---")
    if embedding_result:
        print("Status: OK")
        print(f"ChromaDB is working correctly and was able to create an embedding.")
        print(f"Embedding fragment: {embedding_result[:5]}...")
    else:
        print("[FAILED]")
        print("ChromaDB COULD NOT create an embedding in 45 seconds.")
        print("Most likely cause: no internet access or firewall blocking,")
        print("required for downloading embedding model from Hugging Face.")

    # Cleanup
    collection.delete(ids=["test-doc-1"])
    print("\n--- DIAGNOSTICS COMPLETE ---")

except Exception as e:
    print(f"\n[CRITICAL ERROR] during diagnostics: {e}")