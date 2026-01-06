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
        """Load graph from disk or create a new one if file not found.

        Returns:
            NetworkX graph object.
        """
        if os.path.exists(self.graph_path):
            try:
                with open(self.graph_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logging.error(
                    f"Graph Manager: Error loading graph from {self.graph_path}: {e}. "
                    "Creating new graph."
                )
                return nx.Graph()
        else:
            logging.info(
                f"Graph Manager: Graph file not found at {self.graph_path}. "
                "Creating new graph."
            )
            return nx.Graph()

    def save_graph(self):
        """Save the current graph state to disk.

        This operation is synchronous as pickle lacks native async API.
        Call with asyncio.to_thread in async code.
        """
        try:
            os.makedirs(os.path.dirname(self.graph_path), exist_ok=True)
            # No lock during save - assuming this is called infrequently (e.g., on timer)
            # and not concurrently with modifications
            with open(self.graph_path, 'wb') as f:
                pickle.dump(self.graph, f, pickle.HIGHEST_PROTOCOL)
            logging.info(
                f"Graph Manager: Graph saved to '{self.graph_path}'. "
                f"Nodes: {self.graph.number_of_nodes()}, "
                f"Edges: {self.graph.number_of_edges()}"
            )
        except Exception as e:
            logging.error(f"Graph Manager: Failed to save graph: {e}")

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
                # Update attributes if node already exists
                nx.set_node_attributes(self.graph, {node_id: attrs})

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
