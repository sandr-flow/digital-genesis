"""Memory API server.

FastAPI server for serving 'memories' by node ID.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import sys
import os

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Safe LTM import with all possible error handling
ltm = None
ltm_error = None

try:
    from core.ltm import ltm as ltm_instance

    ltm = ltm_instance
    print("LTM imported successfully")
except ImportError as e:
    ltm_error = f"ImportError: {e}"
    print(f"Warning: Could not import ltm module: {e}")
except TypeError as e:
    ltm_error = f"TypeError during LTM initialization: {e}"
    print(f"Warning: LTM initialization error: {e}")
    print("Likely a dependency or environment issue")
except Exception as e:
    ltm_error = f"General error: {e}"
    print(f"Warning: Unexpected error loading LTM: {e}")

print(f"LTM Status: {'Available' if ltm else 'Unavailable'}")

app = FastAPI(title="Memory API", description="API for retrieving graph node data")

# Allow CORS for local HTML viewing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Memory API Server running successfully"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "ltm_available": ltm is not None,
        "ltm_error": ltm_error if ltm is None else None
    }


@app.get("/memory/{node_id}")
async def get_memory(node_id: str):
    """Get memory data for a specific node."""
    if ltm is None:
        raise HTTPException(status_code=503, detail="LTM module unavailable")

    try:
        # Get data from collection
        result = ltm.stream_collection.get(
            ids=[node_id],
            include=["documents", "metadatas"]
        )

        if not result["ids"] or len(result["ids"]) == 0:
            raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

        # Extract data
        doc = result["documents"][0] if result["documents"] else "No document"
        meta = result["metadatas"][0] if result["metadatas"] else {}

        role = meta.get("role", "unknown").capitalize()
        access_count = meta.get("access_count", 0)

        # Format HTML response
        html_content = f"""
        <div style="font-family: Arial, sans-serif;">
            <h3 style="color: #333; margin-top: 0;">Node: {node_id}</h3>
            <p><strong>Role:</strong> {role}</p>
            <p><strong>Access count:</strong> {access_count}</p>
            <hr style="margin: 15px 0;">
            <div style="background: #f5f5f5; padding: 10px; border-radius: 5px; max-height: 400px; overflow-y: auto;">
                <strong>Content:</strong><br>
                {doc}
            </div>
        </div>
        """

        return HTMLResponse(content=html_content)

    except Exception as e:
        print(f"Error getting node data {node_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    print("Starting Memory API Server...")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
