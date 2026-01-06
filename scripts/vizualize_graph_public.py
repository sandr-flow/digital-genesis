"""Public graph visualization script without node content access.

Standalone version that works without a server - all metadata is embedded in HTML.
Node content is hidden for privacy.
"""

import pickle
import networkx as nx
from pyvis.network import Network
import logging
import os
import json
import sys
import webbrowser
import math
from datetime import datetime

# Add project root to path for config import
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    from config import GRAPH_FILE_PATH, AI_ROLE_NAME
except ImportError:
    logging.warning("Could not import config.py, using default values")
    GRAPH_FILE_PATH = "logs/mind_graph.gpickle"
    AI_ROLE_NAME = "assistant"


def prepare_node_metadata(graph):
    """Prepare node metadata for embedding in HTML."""
    node_metadata = {}
    
    for node_id, attrs in graph.nodes(data=True):
        role = attrs.get('role', 'unknown')
        # Replace FOFE with assistant for display
        display_role = 'assistant' if role == 'FOFE' else role
        neighbors = list(graph.neighbors(node_id))
        
        # Collect edge information
        edges_info = []
        for neighbor in neighbors:
            edge_data = graph.get_edge_data(node_id, neighbor, {})
            edges_info.append({
                "to": neighbor,
                "type": edge_data.get('type', 'unknown'),
                "weight": edge_data.get('cumulative_weight', 1.0)
            })
        
        # Format safe attributes
        safe_metadata = {}
        if 'role' in attrs:
            # Show assistant instead of FOFE
            safe_metadata['role'] = 'assistant' if attrs['role'] == 'FOFE' else attrs['role']
        if 'timestamp' in attrs:
            try:
                dt = datetime.fromtimestamp(attrs['timestamp'])
                safe_metadata['timestamp'] = dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                safe_metadata['timestamp'] = str(attrs['timestamp'])
        
        node_metadata[node_id] = {
            "role": display_role,
            "neighbors_count": len(neighbors),
            "edges": edges_info,
            "metadata": safe_metadata
        }
    
    return node_metadata


def load_graph():
    """Load graph from file."""
    try:
        if not os.path.exists(GRAPH_FILE_PATH):
            logging.error(f"Graph file '{GRAPH_FILE_PATH}' not found")
            return None

        with open(GRAPH_FILE_PATH, 'rb') as f:
            graph = pickle.load(f)
        logging.info(f"Graph loaded from '{GRAPH_FILE_PATH}'")
        return graph
    except Exception as e:
        logging.error(f"Error loading graph: {e}")
        return None


def prepare_data_for_js(graph):
    """Prepare graph data for JavaScript."""
    nodes_data = []
    edges_data = []

    color_map = {
        'internal': '#5BC0DE',
        'user': '#5CB85C',
        AI_ROLE_NAME: '#D9534F',
        'FOFE': '#D9534F',  # FOFE also displayed as Assistant
    }

    for node_id, attrs in graph.nodes(data=True):
        role = attrs.get('role')
        if role is None:
            continue
            
        nodes_data.append({
            "id": node_id,
            "label": ' ',
            "color": color_map.get(role, '#808080'),
            "size": 10,
            "title": f"ID: {node_id}\nRole: {role}",
            "role": role
        })

    valid_node_ids = {n['id'] for n in nodes_data}

    min_weight, max_weight = (1.0, 1.0)
    if graph.number_of_edges() > 0:
        weights = [attrs.get('cumulative_weight', 1.0) for u, v, attrs in graph.edges(data=True)]
        min_weight = min(weights) if weights else 1.0
        max_weight = max(weights) if weights else 1.0

    for u, v, attrs in graph.edges(data=True):
        if u not in valid_node_ids or v not in valid_node_ids:
            continue
            
        edge_type = attrs.get('type', 'unknown')
        weight = attrs.get('cumulative_weight', 1.0)
        width = max(0.5, min(weight * 0.2, 3.0))

        edges_data.append({
            "from": u,
            "to": v,
            "color": 'rgba(0,0,0,0.3)' if edge_type == 'structural' else 'rgba(100,100,100,0.3)',
            "width": width,
            "dashes": edge_type == 'associative',
            "title": f"Type: {edge_type}\nWeight: {weight:.2f}",
            "type": edge_type,
            "weight": weight
        })

    min_weight_rounded = math.floor(min_weight * 10) / 10
    max_weight_rounded = math.ceil(max_weight * 10) / 10

    return nodes_data, edges_data, min_weight_rounded, max_weight_rounded


def create_pyvis_network(nodes_data, edges_data):
    """Create a pyvis network from prepared data."""
    net = Network(
        height='95vh',
        width='100%',
        notebook=False,
        cdn_resources='in_line',
        bgcolor='#ffffff',
        font_color='#000000'
    )

    if not nodes_data:
        return net

    node_ids = [d['id'] for d in nodes_data]
    node_properties = {
        k: [d[k] for d in nodes_data]
        for k in nodes_data[0]
        if k not in ['id', 'role']
    }

    net.add_nodes(node_ids, **node_properties)

    for edge in edges_data:
        net.add_edge(edge['from'], edge['to'],
                     color=edge['color'], width=edge['width'],
                     dashes=edge['dashes'], title=edge['title'])

    return net


def configure_physics(net):
    """Configure physics options for graph layout."""
    physics_config = {
        "edges": {"smooth": {"type": "continuous", "roundness": 0.2}},
        "physics": {
            "enabled": True,
            "solver": "barnesHut",
            "barnesHut": {
                "gravitationalConstant": -8000, "centralGravity": 0.1, "springLength": 200,
                "springConstant": 0.04, "damping": 0.95, "avoidOverlap": 0.2
            },
            "stabilization": {"enabled": True, "iterations": 1000, "updateInterval": 25, "fit": True},
            "timestep": 0.5, "adaptiveTimestep": True, "maxVelocity": 30, "minVelocity": 0.1
        },
        "interaction": {"hover": True, "selectConnectedEdges": False, "dragNodes": True, "dragView": True,
                        "zoomView": True}
    }
    net.set_options(json.dumps(physics_config))


def add_custom_js(html_content, all_nodes_json, all_edges_json, node_metadata_json, min_weight, max_weight):
    """Add custom JS with filtering and controls (standalone version without server)."""
    custom_js = f"""<style>
#loading-overlay{{position:fixed;top:0;left:0;width:100%;height:100%;background:#ffffff;z-index:9999;display:flex;flex-direction:column;align-items:center;justify-content:center;font-family:'Segoe UI',Arial,sans-serif;}}
#loading-overlay .spinner{{border:8px solid #f3f3f3;border-top:8px solid #007bff;border-radius:50%;width:60px;height:60px;animation:spin 1s linear infinite;margin-bottom:20px;}}
@keyframes spin{{0%{{transform:rotate(0deg);}}100%{{transform:rotate(360deg);}}}}
#loading-overlay h2{{color:#333;margin:10px 0;}}
#loading-overlay p{{color:#666;font-size:14px;}}
#loading-progress{{font-size:24px;font-weight:bold;color:#007bff;margin-bottom:10px;}}
#memory-panel{{position:fixed;top:10px;right:10px;width:35%;height:90vh;overflow:auto;border:2px solid #ddd;border-radius:8px;padding:15px;background:#fff;font-family:'Segoe UI',Arial,sans-serif;font-size:14px;z-index:1000;box-shadow:0 4px 6px rgba(0,0,0,.1);display:none;opacity:0;transition:opacity 0.3s ease-in;}}
#memory-panel.visible{{display:block;opacity:1;}}
#memory-panel h3{{margin-top:0;color:#333;border-bottom:2px solid #eee;padding-bottom:10px;}}
.loading{{text-align:center;color:#666;font-style:italic;}}
.error{{color:#d9534f;background:#f2dede;padding:10px;border-radius:4px;border:1px solid #ebccd1;}}
#controls-container{{position:fixed;bottom:10px;left:10px;z-index:1000;display:none;flex-direction:column;gap:10px;opacity:0;transition:opacity 0.3s ease-in;}}
#controls-container.visible{{display:flex;opacity:1;}}
.control-panel{{background:rgba(255,255,255,0.9);padding:10px;border-radius:8px;border:1px solid #ddd;font-family:'Segoe UI',Arial,sans-serif;font-size:12px;box-shadow:0 2px 4px rgba(0,0,0,.1);}}
.control-panel h4{{margin:0 0 8px 0;font-size:13px;border-bottom:1px solid #eee;padding-bottom:5px;}}
.control-panel button{{margin:2px;padding:5px 10px;border:1px solid #ccc;border-radius:4px;background:#f8f9fa;cursor:pointer;font-size:11px;}}
.control-panel button:hover{{background:#e9ecef;}}
.control-panel button.active{{background:#007bff;color:white;}}
.filter-group label{{display:inline-block;margin-right:10px;user-select:none;}}
#weight-slider-container label{{display:block;margin-bottom:5px;}}
</style>
<div id="loading-overlay">
    <div id="loading-progress">0%</div>
    <div class="spinner"></div>
    <h2>Loading graph...</h2>
    <p>Please wait</p>
</div>
<div id="memory-panel"><div class="loading">Click on a node to view metadata</div></div>
<div id="controls-container">
    <div id="filter-controls" class="control-panel">
        <h4>Filtering</h4>
        <div class="filter-group">
            <strong>Node roles:</strong>
            <label><input type="checkbox" class="filter-cb" id="filter-role-user" checked> User</label>
            <label><input type="checkbox" class="filter-cb" id="filter-role-{AI_ROLE_NAME}" checked> Assistant</label>
            <label><input type="checkbox" class="filter-cb" id="filter-role-internal" checked> Internal</label>
        </div>
        <div class="filter-group" style="margin-top:8px;">
            <strong>Edge types:</strong>
            <label><input type="checkbox" class="filter-cb" id="filter-type-structural" checked> Structural</label>
            <label><input type="checkbox" class="filter-cb" id="filter-type-associative" checked> Associative</label>
        </div>
        <div id="weight-slider-container" style="margin-top:8px;">
            <label for="filter-weight">Min. edge weight: <span id="weight-value">1.00</span></label>
            <input type="range" id="filter-weight" min="{min_weight}" max="{max_weight}" value="1.0" step="0.1" style="width:100%;">
        </div>
        <div class="filter-group" style="margin-top:8px; border-top: 1px solid #eee; padding-top: 8px;">
             <label><input type="checkbox" class="filter-cb" id="filter-hide-isolated"> Hide isolated nodes</label>
        </div>
        <div style="margin-top:10px; text-align:right;">
            <button id="apply-filters-btn">Apply</button>
            <button id="reset-filters-btn">Reset</button>
        </div>
    </div>
    <div id="physics-controls" class="control-panel">
        <h4>Physics controls</h4>
        <button id="physics-toggle" class="active">Physics ON</button>
        <button id="fit-btn">Fit all</button>
    </div>
</div>

<script>
document.addEventListener('DOMContentLoaded', function() {{
    const allNodes = {all_nodes_json};
    const allEdges = {all_edges_json};
    const nodeMetadata = {node_metadata_json};
    let currentNodeId = null;
    let physicsEnabled = true;

    // Function to display node metadata (without server)
    function displayNodeMetadata(nodeId) {{
        const metadata = nodeMetadata[nodeId];
        if (!metadata) {{
            return '<div class="error"><strong>Node not found</strong><br>ID: ' + nodeId + '</div>';
        }}

        const role = metadata.role;
        const roleColor = role === 'user' ? '#5CB85C' : (role === '{AI_ROLE_NAME}' ? '#D9534F' : '#5BC0DE');
        
        let metadataHtml = '';
        for (const [key, value] of Object.entries(metadata.metadata)) {{
            metadataHtml += '<strong>' + key + ':</strong> ' + value + '<br>';
        }}
        if (!metadataHtml) {{
            metadataHtml = 'No public attributes';
        }}

        let edgesHtml = '';
        if (metadata.edges && metadata.edges.length > 0) {{
            const displayEdges = metadata.edges.slice(0, 10);
            edgesHtml = displayEdges.map(edge => 
                '<div style="margin: 5px 0; padding: 5px; background: #f0f0f0; border-radius: 3px;"><strong>' + edge.to.substring(0, 40) + '...</strong> (' + edge.type + ') - weight: ' + edge.weight.toFixed(2) + '</div>'
            ).join('');
            if (metadata.edges.length > 10) {{
                edgesHtml += '<div style="color: #666; font-style: italic;">... and ' + (metadata.edges.length - 10) + ' more</div>';
            }}
        }}

        return '<div style="font-family: Arial, sans-serif;">' +
            '<h3 style="color: #333; margin-top: 0;">Node: ' + nodeId + '</h3>' +
            '<p><strong>Role:</strong> <span style="color: ' + roleColor + ';">' + role.charAt(0).toUpperCase() + role.slice(1) + '</span></p>' +
            '<p><strong>Neighbors:</strong> ' + metadata.neighbors_count + '</p>' +
            '<hr style="margin: 15px 0;">' +
            '<h4>Content:</h4>' +
            '<div style="background: #f9f9f9; padding: 20px; border-radius: 5px; border-left: 4px solid #007bff; text-align: center;">' +
                '<p style="color: #666; margin: 0; font-size: 14px;"><strong>Node content is available to developers only</strong></p>' +
                '<p style="color: #999; margin: 10px 0 0 0; font-size: 12px;">This information contains private data and is not intended for public viewing.</p>' +
            '</div>' +
            '<hr style="margin: 15px 0;">' +
            '<h4>Metadata:</h4>' +
            '<div style="background: #f9f9f9; padding: 10px; border-radius: 5px; font-family: monospace; font-size: 12px;">' + metadataHtml + '</div>' +
            (metadata.edges && metadata.edges.length > 0 ? 
                '<hr style="margin: 15px 0;"><h4>Connections (' + metadata.edges.length + '):</h4><div style="max-height: 150px; overflow-y: auto;">' + edgesHtml + '</div>' 
            : '') +
        '</div>';
    }}

    function applyFilters() {{
        const assistantChecked = document.getElementById('filter-role-{AI_ROLE_NAME}').checked;
        const visibleRoles = {{
            'user': document.getElementById('filter-role-user').checked,
            '{AI_ROLE_NAME}': assistantChecked,
            'FOFE': assistantChecked,
            'internal': document.getElementById('filter-role-internal').checked
        }};
        const visibleTypes = {{
            'structural': document.getElementById('filter-type-structural').checked,
            'associative': document.getElementById('filter-type-associative').checked
        }};
        const minWeight = parseFloat(document.getElementById('filter-weight').value);
        const hideIsolated = document.getElementById('filter-hide-isolated').checked;

        let preliminaryNodes = allNodes.filter(node => visibleRoles[node.role]);
        const preliminaryNodeIds = new Set(preliminaryNodes.map(n => n.id));

        const filteredEdges = allEdges.filter(edge => 
            visibleTypes[edge.type] &&
            edge.weight >= minWeight &&
            preliminaryNodeIds.has(edge.from) &&
            preliminaryNodeIds.has(edge.to)
        );

        let finalNodes;
        if (hideIsolated) {{
            const connectedNodeIds = new Set();
            filteredEdges.forEach(edge => {{
                connectedNodeIds.add(edge.from);
                connectedNodeIds.add(edge.to);
            }});
            finalNodes = preliminaryNodes.filter(node => connectedNodeIds.has(node.id));
        }} else {{
            finalNodes = preliminaryNodes;
        }}

        nodes.clear();
        edges.clear();
        nodes.add(finalNodes);
        edges.add(filteredEdges);
        
        // Automatically enable physics when filters change
        if (typeof network !== 'undefined' && !physicsEnabled) {{
            network.setOptions({{ physics: {{ enabled: true }} }});
            physicsEnabled = true;
            const toggleBtn = document.getElementById("physics-toggle");
            if (toggleBtn) {{
                toggleBtn.textContent = "Physics ON";
                toggleBtn.classList.add("active");
            }}
        }}
    }}

    document.getElementById('apply-filters-btn').addEventListener('click', applyFilters);

    document.getElementById('reset-filters-btn').addEventListener('click', function() {{
        document.querySelectorAll('.filter-cb').forEach(cb => cb.checked = cb.id !== 'filter-hide-isolated');
        document.getElementById('filter-hide-isolated').checked = false;

        const weightSlider = document.getElementById('filter-weight');
        weightSlider.value = 1.0;
        document.getElementById('weight-value').textContent = '1.00';

        nodes.clear();
        edges.clear();
        nodes.add(allNodes);
        edges.add(allEdges);
        
        // Automatically enable physics on reset
        if (typeof network !== 'undefined' && !physicsEnabled) {{
            network.setOptions({{ physics: {{ enabled: true }} }});
            physicsEnabled = true;
            const toggleBtn = document.getElementById("physics-toggle");
            if (toggleBtn) {{
                toggleBtn.textContent = "Physics ON";
                toggleBtn.classList.add("active");
            }}
        }}
    }});

    document.getElementById('filter-weight').addEventListener('input', function() {{
        document.getElementById('weight-value').textContent = parseFloat(this.value).toFixed(2);
    }});

    setTimeout(() => {{
        if (typeof network === 'undefined') {{
            console.error("Pyvis 'network' object not found!");
            return;
        }}

        network.on("stabilizationIterationsDone", function() {{
            setTimeout(() => {{
                network.setOptions({{ physics: {{ enabled: false }} }});
                physicsEnabled = false;
                const toggleBtn = document.getElementById("physics-toggle");
                toggleBtn.textContent = "Physics OFF";
                toggleBtn.classList.remove("active");
                
                const loadingOverlay = document.getElementById(\"loading-overlay\");\n                const controlsContainer = document.getElementById(\"controls-container\");
                
                if (loadingOverlay) loadingOverlay.style.display = \"none\";
                if (controlsContainer) {{
                    controlsContainer.style.display = "flex";
                    setTimeout(() => controlsContainer.classList.add("visible"), 10);
                }}
            }}, 500);
        }});

        // Update stabilization progress
        network.on("stabilizationProgress", function(params) {{
            const progress = Math.round((params.iterations / params.total) * 100);
            const progressEl = document.getElementById("loading-progress");
            if (progressEl) progressEl.textContent = progress + "%";
        }});

        document.getElementById("physics-toggle").addEventListener("click", function() {{
            physicsEnabled = !physicsEnabled;
            network.setOptions({{ physics: {{ enabled: physicsEnabled }} }});
            this.textContent = physicsEnabled ? "Physics ON" : "Physics OFF";
            this.classList.toggle("active", physicsEnabled);
        }});

        document.getElementById("fit-btn").addEventListener("click", function() {{
            network.fit();
        }});

        network.on("click", function(e) {{
            const panel = document.getElementById("memory-panel");
            if (e.nodes.length > 0) {{
                let nodeId = e.nodes[0];
                currentNodeId = nodeId;
                panel.innerHTML = displayNodeMetadata(nodeId);
                // Show panel only when clicking on a node
                panel.style.display = "block";
                setTimeout(() => panel.classList.add("visible"), 10);
            }} else {{
                // Hide panel when clicking on empty space
                panel.classList.remove("visible");
                setTimeout(() => {{ panel.style.display = "none"; }}, 300);
                currentNodeId = null;
            }}
        }});

        network.on("hoverNode", function() {{ document.body.style.cursor = "pointer"; }});
        network.on("blurNode", function() {{ document.body.style.cursor = "default"; }});

    }}, 100);
}});
</script>"""

    if "</body>" in html_content:
        return html_content.replace("</body>", custom_js + "</body>")
    return html_content + custom_js


def visualize_interactive_with_graph(graph):
    """Create interactive public graph visualization."""
    logging.info("--- Starting public graph visualizer ---")

    if graph is None:
        return False

    if graph.number_of_nodes() == 0:
        logging.warning("Graph is empty. Visualization not possible.")
        return False

    logging.info(f"Graph contains {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")

    # Prepare graph data
    nodes_data, edges_data, min_w, max_w = prepare_data_for_js(graph)
    
    # Prepare node metadata
    node_metadata = prepare_node_metadata(graph)

    net = create_pyvis_network(nodes_data, edges_data)

    configure_physics(net)

    html_path = "public_graph_visualization.html"
    html_content = net.generate_html()

    all_nodes_json = json.dumps(nodes_data)
    all_edges_json = json.dumps(edges_data)
    node_metadata_json = json.dumps(node_metadata)
    html_content = add_custom_js(html_content, all_nodes_json, all_edges_json, node_metadata_json, min_w, max_w)

    try:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        full_path = os.path.abspath(html_path)
        logging.info(f"Public HTML file saved: {full_path}")

        try:
            webbrowser.open(f"file://{full_path}")
            logging.info("Page opened in browser")
        except Exception as e:
            logging.warning(f"Could not open browser: {e}")
            logging.info(f"Open manually: file://{full_path}")

        return True

    except Exception as e:
        logging.error(f"Error saving HTML: {e}")
        return False


def main():
    """Main entry point for public visualization script."""
    logging.info("Starting public graph visualization system")
    
    # Load graph
    graph = load_graph()
    if graph is None:
        logging.error("Failed to load graph")
        return
    
    # Create visualization
    viz_created = visualize_interactive_with_graph(graph)

    if viz_created:
        logging.info("Public system started successfully!")
        logging.info("Use filter panel at bottom left to control graph display.")
        logging.info("Click on nodes to view metadata (content is hidden)")
        logging.info("HTML file works standalone, without a server")
    else:
        logging.error("Error creating visualization")


if __name__ == "__main__":
    main()
