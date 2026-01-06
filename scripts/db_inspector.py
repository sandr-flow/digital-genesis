"""Graph rebuild script.

Rebuilds the entire graph based on data from ChromaDB.
"""

import asyncio
import logging
import os
from tqdm import tqdm

# Ensure our modules are properly imported
from ltm import ltm, LTM_Manager
from graph_manager import graph_manager
from config import GRAPH_FILE_PATH

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


async def main():
    """
    Main function for complete graph rebuild based on ChromaDB data.
    """
    logging.info("--- STARTING GRAPH REBUILD ---")

    # 1. Check and remove old graph file
    if os.path.exists(GRAPH_FILE_PATH):
        logging.warning(f"Found old graph file '{GRAPH_FILE_PATH}'. Removing it for clean rebuild.")
        try:
            os.remove(GRAPH_FILE_PATH)
            # Important: need to recreate graph_manager instance to load empty graph
            # This is the simplest way to "reset" its state.
            # In a real app, there would be a reset() function.
            global graph_manager
            graph_manager = type(graph_manager)(graph_path=GRAPH_FILE_PATH)
            logging.info("Old graph removed, graph manager reset.")
        except OSError as e:
            logging.error(f"Could not remove old graph file: {e}")
            return
    else:
        logging.info("Old graph file not found, starting from scratch.")

    # 2. Get all "Cognitive Assets" from database
    logging.info("Loading all Cognitive Assets from database...")
    try:
        # ltm.assets_collection.get() without parameters returns all records (with default limit, but enough for hundreds/thousands)
        all_assets = ltm.assets_collection.get(include=["metadatas"])
        asset_ids = all_assets.get('ids', [])

        if not asset_ids:
            logging.error("No Cognitive Assets found in database. Rebuild not possible.")
            return

        logging.info(f"Found {len(asset_ids)} assets to process.")
    except Exception as e:
        logging.error(f"Error getting assets from ChromaDB: {e}")
        return

    # 3. Recreate graph nodes (important since we deleted the old graph)
    logging.info("Recreating graph nodes from main 'stream' collection...")
    all_stream_records = ltm.stream_collection.get(include=["metadatas"])
    for i, node_id in enumerate(all_stream_records['ids']):
        meta = all_stream_records['metadatas'][i]
        graph_manager.add_node_if_not_exists(node_id, role=meta.get('role'), timestamp=meta.get('timestamp'))
    logging.info(f"Created {graph_manager.graph.number_of_nodes()} nodes.")

    # 4. Iterative edge rebuild for each asset
    logging.info("Starting graph edge rebuild. This may take a while...")

    # Use tqdm for a nice progress bar
    for asset_id in tqdm(asset_ids, desc="Processing assets"):
        try:
            # Call the "secret" internal function from ltm that does exactly what we need!
            await ltm._rebuild_graph_for_asset(asset_id)
        except Exception as e:
            logging.warning(f"Error processing asset {asset_id}: {e}")

    logging.info("--- EDGE REBUILD COMPLETE ---")
    logging.info(
        f"New graph contains: {graph_manager.graph.number_of_nodes()} nodes and {graph_manager.graph.number_of_edges()} edges.")

    # 5. Save the new graph
    logging.info("Saving new graph to disk...")
    graph_manager.save_graph()
    logging.info(f"New graph successfully saved to '{GRAPH_FILE_PATH}'.")


if __name__ == "__main__":
    asyncio.run(main())