"""Graph fallback server.

Alternative server that reads data directly from the graph file.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import pickle
import networkx as nx
import sys
import os

# Add project root to path for config.py import
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Try to import config
try:
    from config import GRAPH_FILE_PATH, AI_ROLE_NAME
except ImportError:
    print("Warning: config.py not found, using default values")
    GRAPH_FILE_PATH = "logs/mind_graph.gpickle"
    AI_ROLE_NAME = "assistant"

app = FastAPI(title="Graph Memory API", description="API for retrieving node data from NetworkX graph")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable for graph
graph = None
graph_error = None


def load_graph():
    """Load graph at server startup."""
    global graph, graph_error
    try:
        if not os.path.exists(GRAPH_FILE_PATH):
            graph_error = f"Graph file not found: {GRAPH_FILE_PATH}"
            print(f"Error: {graph_error}")
            return False

        with open(GRAPH_FILE_PATH, 'rb') as f:
            graph = pickle.load(f)

        print(f"Graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        return True

    except Exception as e:
        graph_error = f"Error loading graph: {str(e)}"
        print(f"Error: {graph_error}")
        return False


# Load graph on startup
load_graph()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Graph Memory API Server",
        "graph_loaded": graph is not None,
        "nodes": graph.number_of_nodes() if graph else 0,
        "edges": graph.number_of_edges() if graph else 0
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "graph_available": graph is not None,
        "graph_error": graph_error,
        "graph_file": GRAPH_FILE_PATH
    }


@app.get("/reload")
async def reload_graph():
    """Reload graph."""
    success = load_graph()
    return {
        "success": success,
        "graph_available": graph is not None,
        "error": graph_error
    }


@app.get("/nodes")
async def list_nodes():
    """List all nodes."""
    if graph is None:
        raise HTTPException(status_code=503, detail="Graph unavailable")

    nodes_info = []
    for node_id, attrs in graph.nodes(data=True):
        nodes_info.append({
            "id": node_id,
            "role": attrs.get('role', 'unknown'),
            "attributes": attrs
        })

    return {"nodes": nodes_info, "count": len(nodes_info)}


@app.get("/memory/{node_id}")
async def get_memory(node_id: str):
    """Get memory data for a specific node."""
    if graph is None:
        raise HTTPException(status_code=503, detail=f"Graph unavailable: {graph_error}")

    try:
        # Check if node exists
        if node_id not in graph.nodes():
            raise HTTPException(status_code=404, detail=f"Node {node_id} not found in graph")

        # Get node attributes
        attrs = graph.nodes[node_id]
        role = attrs.get('role', 'unknown')

        # Get neighbors
        neighbors = list(graph.neighbors(node_id))

        # Get edges
        edges_info = []
        for neighbor in neighbors:
            edge_data = graph.get_edge_data(node_id, neighbor, {})
            edges_info.append({
                "to": neighbor,
                "type": edge_data.get('type', 'unknown'),
                "weight": edge_data.get('cumulative_weight', 1.0)
            })

        # Try to find text content in attributes
        content = attrs.get('content', attrs.get('text', attrs.get('document', 'Content not found')))

        # Format HTML response
        html_content = f"""
        <div style="font-family: Arial, sans-serif;">
            <h3 style="color: #333; margin-top: 0;">Node: {node_id}</h3>
            <p><strong>Role:</strong> <span style="color: {'#5CB85C' if role == 'user' else '#D9534F' if role == AI_ROLE_NAME else '#5BC0DE'};">{role.capitalize()}</span></p>
            <p><strong>Neighbors:</strong> {len(neighbors)}</p>

            <hr style="margin: 15px 0;">

            <h4>Content:</h4>
            <div style="background: #f5f5f5; padding: 10px; border-radius: 5px; max-height: 300px; overflow-y: auto; border-left: 4px solid #ddd;">
                {content}
            </div>

            <hr style="margin: 15px 0;">

            <h4>All attributes:</h4>
            <div style="background: #f9f9f9; padding: 10px; border-radius: 5px; max-height: 200px; overflow-y: auto; font-family: monospace; font-size: 12px;">
                {format_attributes(attrs)}
            </div>

            {f'''
            <hr style="margin: 15px 0;">
            <h4>Connections ({len(edges_info)}):</h4>
            <div style="max-height: 150px; overflow-y: auto;">
                {''.join([f"<div style='margin: 5px 0; padding: 5px; background: #f0f0f0; border-radius: 3px;'><strong>{edge['to']}</strong> ({edge['type']}) - weight: {edge['weight']:.2f}</div>" for edge in edges_info[:10]])}
                {f"<div style='color: #666; font-style: italic;'>... and {len(edges_info) - 10} more</div>" if len(edges_info) > 10 else ""}
            </div>
            ''' if edges_info else ''}
        </div>
        """

        return HTMLResponse(content=html_content)

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting node data {node_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


def format_attributes(attrs):
    """Format attributes for display."""
    if not attrs:
        return "No attributes"

    formatted = []
    for key, value in attrs.items():
        if isinstance(value, str) and len(value) > 100:
            value = value[:100] + "..."
        formatted.append(f"<strong>{key}:</strong> {value}")

    return "<br>".join(formatted)


if __name__ == "__main__":
    import uvicorn

    print("Starting Graph Memory API Server...")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")