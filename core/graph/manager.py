"""Knowledge graph manager module.

Handles graph loading, saving, and modification with thread-safe async access.
"""

import networkx as nx
import os
import pickle
import logging
import asyncio
from config import GRAPH_STRUCTURAL_THRESHOLD


class GraphManager:
    """Knowledge graph manager.

    Handles graph loading, saving, and modification.
    Thread-safe for use in async environments.
    """
    
    def __init__(self, graph_path: str):
        """Initialize the graph manager.

        Loads the graph and creates a lock for safe async access.

        Args:
            graph_path: Path to the graph file.
        """
        self.graph_path = graph_path
        self.graph = self._load_graph()
        self.lock = asyncio.Lock()  # Lock for thread-safety
        logging.info(
            f"Graph Manager: Graph loaded from '{self.graph_path}'. "
            f"Nodes: {self.graph.number_of_nodes()}, "
            f"Edges: {self.graph.number_of_edges()}"
        )

    def _load_graph(self) -> nx.Graph:
        """Load graph from disk with robust error handling.

        Returns:
            NetworkX graph object.

        Raises:
            RuntimeError: If graph file exists but cannot be loaded.
        """
        # Check if main file exists
        if not os.path.exists(self.graph_path):
            # Main file missing - check for backup before creating new graph
            backup_path = f"{self.graph_path}.bak"
            if os.path.exists(backup_path):
                logging.warning(
                    f"Graph Manager: Main file not found, but backup exists at {backup_path}. "
                    "Attempting to restore from backup."
                )
                try:
                    with open(backup_path, 'rb') as f:
                        graph = pickle.load(f)
                    # Restore backup as main file
                    os.replace(backup_path, self.graph_path)
                    logging.info("Graph Manager: Successfully restored from backup.")
                    return graph
                except Exception as e:
                    logging.error(f"Graph Manager: Failed to load backup: {e}")
                    # Fall through to create new graph
            
            logging.info(
                f"Graph Manager: Graph file not found at {self.graph_path}. "
                "Creating new graph."
            )
            return nx.Graph()

        # File exists - attempt to load
        try:
            with open(self.graph_path, 'rb') as f:
                graph = pickle.load(f)
            logging.info(f"Graph Manager: Graph loaded from {self.graph_path}")
            return graph
        except Exception as e:
            # File exists but corrupted - preserve it for manual recovery
            corrupted_path = f"{self.graph_path}.corrupted"
            logging.error(
                f"Graph Manager: Failed to load graph from {self.graph_path}: {e}. "
                f"Renaming to {corrupted_path} for manual recovery."
            )
            os.rename(self.graph_path, corrupted_path)

            # Try loading backup if available
            backup_path = f"{self.graph_path}.bak"
            if os.path.exists(backup_path):
                logging.info(f"Graph Manager: Attempting to load backup from {backup_path}")
                try:
                    with open(backup_path, 'rb') as f:
                        return pickle.load(f)
                except Exception as backup_error:
                    logging.error(f"Graph Manager: Backup also corrupted: {backup_error}")

            raise RuntimeError(
                f"Cannot load graph from {self.graph_path}. "
                f"Corrupted file saved to {corrupted_path}. "
                "Please restore from backup or delete to start fresh."
            )

    def _sync_save_graph(self, temp_path: str, backup_path: str):
        """Synchronous graph save operation for thread pool execution.
        
        This method contains blocking I/O operations and should only be called
        from asyncio.to_thread() to avoid blocking the event loop.
        
        Args:
            temp_path: Path to temporary file.
            backup_path: Path to backup file.
        """
        # Write to temporary file first (blocking I/O)
        with open(temp_path, 'wb') as f:
            pickle.dump(self.graph, f, pickle.HIGHEST_PROTOCOL)

        # Create backup of old graph before replacing
        if os.path.exists(self.graph_path):
            os.replace(self.graph_path, backup_path)

        # Atomic replace - if this fails, backup is still intact
        os.replace(temp_path, self.graph_path)

    async def save_graph(self):
        """Save the current graph state to disk atomically.

        Thread-safe: Acquires lock before serialization to prevent race conditions.
        Non-blocking: Runs blocking I/O in thread pool to avoid freezing event loop.
        Atomic: Writes to temp file, then replaces original to prevent corruption.
        Creates backup of previous graph before overwriting.
        """
        async with self.lock:  # Prevent concurrent modifications during serialization
            temp_path = f"{self.graph_path}.tmp"
            backup_path = f"{self.graph_path}.bak"
            
            try:
                os.makedirs(os.path.dirname(self.graph_path), exist_ok=True)

                # Run blocking I/O in thread pool to avoid blocking event loop
                await asyncio.to_thread(self._sync_save_graph, temp_path, backup_path)

                logging.info(
                    f"Graph Manager: Graph saved atomically to '{self.graph_path}'. "
                    f"Nodes: {self.graph.number_of_nodes()}, "
                    f"Edges: {self.graph.number_of_edges()}"
                )
            except Exception as e:
                logging.error(f"Graph Manager: Failed to save graph: {e}", exc_info=True)
                # Clean up temp file if it exists
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass  # Best effort cleanup

    async def add_node_if_not_exists(self, node_id: str, **attrs):
        """Add a node if it doesn't exist and update its attributes.

        Args:
            node_id: Node ID.
            **attrs: Node attributes.
        """
        async with self.lock:
            if not self.graph.has_node(node_id):
                self.graph.add_node(node_id, **attrs)
            else:
                # Direct update is faster than nx.set_node_attributes for single node
                self.graph.nodes[node_id].update(attrs)

    async def add_or_update_edge(
        self,
        node1_id: str,
        node2_id: str,
        similarity_score: float,
        asset1_meta: dict,
        asset2_meta: dict
    ):
        """Add or update an edge with weight based on semantic similarity.

        Weight is calculated from similarity, importance, and confidence
        of the cognitive assets that created the edge.

        Args:
            node1_id: First node ID.
            node2_id: Second node ID.
            similarity_score: Semantic similarity score.
            asset1_meta: First asset metadata.
            asset2_meta: Second asset metadata.
        """
        if node1_id == node2_id:
            return

        # Acquire lock for safe graph modification
        async with self.lock:
            # Extract scores from metadata
            imp1 = asset1_meta.get('importance', 5)
            conf1 = asset1_meta.get('confidence', 5)
            imp2 = asset2_meta.get('importance', 5)
            conf2 = asset2_meta.get('confidence', 5)

            # Calculate final weight
            # Formula: similarity * avg(importance*confidence / 100)
            # Divide by 100 since imp and conf are 1-10, product up to 100
            weight_modifier = ((imp1 * conf1) + (imp2 * conf2)) / 200.0
            final_weight = similarity_score * weight_modifier

            link_type = 'structural' if similarity_score > GRAPH_STRUCTURAL_THRESHOLD else 'associative'

            if self.graph.has_edge(node1_id, node2_id):
                edge = self.graph[node1_id][node2_id]
                # Add new weighted weight to cumulative
                edge['cumulative_weight'] = edge.get('cumulative_weight', 0) + final_weight
                edge['shared_concepts_count'] = edge.get('shared_concepts_count', 0) + 1
                edge['max_similarity'] = max(edge.get('max_similarity', 0), similarity_score)
                # If link became structural, upgrade its status
                if link_type == 'structural' and edge.get('type') == 'associative':
                    edge['type'] = 'structural'
            else:
                self.graph.add_edge(
                    node1_id,
                    node2_id,
                    type=link_type,
                    max_similarity=similarity_score,
                    shared_concepts_count=1,
                    cumulative_weight=final_weight  # Initial weight
                )
