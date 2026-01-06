"""Animated graph visualization in banner format for Notion.

Nodes and edges appear gradually, physics is active for cluster formation.
"""

import pickle
import networkx as nx
import logging
import os
import json
import webbrowser
import random

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import sys

# Add project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from config import GRAPH_FILE_PATH, AI_ROLE_NAME
except ImportError:
    logging.warning("Could not import config.py, using default values")
    GRAPH_FILE_PATH = "logs/mind_graph.gpickle"
    AI_ROLE_NAME = "assistant"


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


def prepare_banner_data(graph):
    """
    Prepare graph data for animated visualization.
    Returns nodes, edges, and node->edges mapping for synchronous addition.
    """
    nodes_data = []
    edges_data = []
    
    # Color scheme for white background — saturated colors
    color_map = {
        'internal': {'background': '#6366f1', 'border': '#4338ca'},  # Indigo
        'user': {'background': '#10b981', 'border': '#047857'},       # Emerald
        AI_ROLE_NAME: {'background': '#f43f5e', 'border': '#be123c'}, # Rose
        'FOFE': {'background': '#f43f5e', 'border': '#be123c'}        # Rose (merged with assistant)
    }
    default_color = {'background': '#64748b', 'border': '#334155'}  # Slate
    
    for node_id, attrs in graph.nodes(data=True):
        if graph.degree(node_id) == 0:
            continue
        
        role = attrs.get('role')
        # Exclude nodes without role
        if role is None:
            continue
            
        colors = color_map.get(role, default_color)
        
        nodes_data.append({
            "id": node_id,
            "x": random.randint(-100, 100),  # Random initial X position
            "y": random.randint(-100, 100),  # Random initial Y position
            "color": {
                "background": colors['background'],
                "border": colors['border']
            },
            "size": random.randint(6, 14),
            "borderWidth": 0,
            "shape": "dot"
        })
    
    # Create mapping: node -> list of edges where this node is the source
    node_to_edges = {}
    
    # Minimum edge weight for display (can be configured)
    MIN_EDGE_WEIGHT = 0.8
    
    for u, v, attrs in graph.edges(data=True):
        weight = attrs.get('cumulative_weight', 1.0)
        
        # Filter edges by weight
        if weight < MIN_EDGE_WEIGHT:
            continue
            
        edge = {
            "from": u,
            "to": v,
            "color": {"color": "rgba(100, 116, 139, 0.22)"},  # More visible edges
            "width": max(0.5, min(weight * 0.25, 1.5)),  # Slightly thicker lines
            "smooth": {"type": "continuous", "roundness": 0.2}
        }
        edges_data.append(edge)
        
        # Bind edge to source node
        if u not in node_to_edges:
            node_to_edges[u] = []
        node_to_edges[u].append(edge)
    
    return nodes_data, edges_data, node_to_edges


def generate_banner_html(nodes_data, edges_data, node_to_edges):
    """Generate HTML page with animated banner visualization."""
    
    nodes_json = json.dumps(nodes_data, ensure_ascii=False)
    edges_json = json.dumps(edges_data, ensure_ascii=False)
    node_to_edges_json = json.dumps(node_to_edges, ensure_ascii=False)
    
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Graph Clustering Animation</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        html, body {{
            width: 100%;
            height: 100%;
            overflow: hidden;
            background: #ffffff;
        }}
        
        #graph-container {{
            width: 100%;
            height: 100%;
        }}
        
        /* Hide all vis.js interface elements */
        .vis-navigation,
        .vis-manipulation,
        .vis-close,
        .vis-edit-mode,
        .vis-button {{
            display: none !important;
        }}
    </style>
</head>
<body>
    <div id="graph-container"></div>
    
    <script>
        // Все данные
        const allNodesData = {nodes_json};
        const allEdgesData = {edges_json};
        const nodeToEdges = {node_to_edges_json};
        
        // DataSets для vis.js (изначально пустые)
        const nodes = new vis.DataSet([]);
        const edges = new vis.DataSet([]);
        
        // Контейнер
        const container = document.getElementById('graph-container');
        
        // ОПТИМИЗАЦИЯ: Опции сети — отключены тени, оптимизированы параметры физики
        const options = {{
            nodes: {{
                font: {{ size: 0 }},
                // ОПТИМИЗАЦИЯ: Тени отключены для повышения FPS
                shadow: false
            }},
            edges: {{
                shadow: false,
                selectionWidth: 0,
                // ОПТИМИЗАЦИЯ: Упрощённое сглаживание рёбер
                smooth: {{
                    enabled: true,
                    type: 'continuous',
                    roundness: 0.2
                }}
            }},
            physics: {{
                enabled: true,
                // Barnes-Hut быстрее для больших графов
                solver: 'barnesHut',
                barnesHut: {{
                    gravitationalConstant: -2000,
                    centralGravity: 0.08,
                    springLength: 120,
                    springConstant: 0.015,
                    damping: 0.4,           // Уменьшено для менее вязкого движения
                    avoidOverlap: 0.3
                }},
                stabilization: {{
                    enabled: false
                }},
                timestep: 0.35,             // Увеличен для более плавной симуляции
                maxVelocity: 25,            // Увеличена максимальная скорость
                minVelocity: 0.1
            }},
            interaction: {{
                dragNodes: false,
                dragView: false,
                zoomView: false,
                selectable: false,
                hover: false,
                keyboard: false,
                navigationButtons: false,
                // ОПТИМИЗАЦИЯ: Отключена подсветка рёбер
                hideEdgesOnDrag: true,
                hideEdgesOnZoom: true
            }}
        }};
        
        // Создаём сеть
        const network = new vis.Network(container, {{ nodes, edges }}, options);
        
        // ========================================
        // ПЛАВНОЕ НЕПРЕРЫВНОЕ МАСШТАБИРОВАНИЕ
        // Вычисляем bounding box каждый кадр и плавно интерполируем камеру
        // ========================================
        const SMOOTH_SCALE_CONFIG = {{
            padding: 80,           // Отступ от краёв в пикселях
            lerpFactor: 0.08,      // Скорость интерполяции (увеличено для быстрой реакции)
            minScale: 0.1,         // Минимальный масштаб (снижен для больших графов)
            maxScale: 2.0          // Максимальный масштаб
        }};
        
        // Текущие значения камеры (для плавной интерполяции)
        let currentScale = 1.0;
        let currentCenterX = 0;
        let currentCenterY = 0;
        let isFirstFrame = true;
        
        // Линейная интерполяция
        function lerp(current, target, factor) {{
            return current + (target - current) * factor;
        }}
        
        // Функция плавного масштабирования (вызывается каждый кадр)
        function smoothScaleAnimation() {{
            const positions = network.getPositions();
            const nodeIds = Object.keys(positions);
            
            // Если нет узлов — пропускаем
            if (nodeIds.length === 0) {{
                requestAnimationFrame(smoothScaleAnimation);
                return;
            }}
            
            // Вычисляем bounding box всех узлов
            let minX = Infinity, maxX = -Infinity;
            let minY = Infinity, maxY = -Infinity;
            
            for (const nodeId of nodeIds) {{
                const pos = positions[nodeId];
                if (pos.x < minX) minX = pos.x;
                if (pos.x > maxX) maxX = pos.x;
                if (pos.y < minY) minY = pos.y;
                if (pos.y > maxY) maxY = pos.y;
            }}
            
            // Размеры bounding box графа
            const graphWidth = maxX - minX;
            const graphHeight = maxY - minY;
            const graphCenterX = (minX + maxX) / 2;
            const graphCenterY = (minY + maxY) / 2;
            
            // Размеры контейнера
            const canvasWidth = container.clientWidth;
            const canvasHeight = container.clientHeight;
            
            if (canvasWidth === 0 || canvasHeight === 0) {{
                requestAnimationFrame(smoothScaleAnimation);
                return;
            }}
            
            // Вычисляем целевой масштаб, чтобы граф помещался с отступами
            const availableWidth = canvasWidth - SMOOTH_SCALE_CONFIG.padding * 2;
            const availableHeight = canvasHeight - SMOOTH_SCALE_CONFIG.padding * 2;
            
            let targetScale = 1.0;
            if (graphWidth > 0 && graphHeight > 0) {{
                const scaleX = availableWidth / graphWidth;
                const scaleY = availableHeight / graphHeight;
                targetScale = Math.min(scaleX, scaleY);
            }}
            
            // Ограничиваем масштаб
            targetScale = Math.max(SMOOTH_SCALE_CONFIG.minScale, 
                                   Math.min(SMOOTH_SCALE_CONFIG.maxScale, targetScale));
            
            // Первый кадр — мгновенно устанавливаем значения
            if (isFirstFrame) {{
                currentScale = targetScale;
                currentCenterX = graphCenterX;
                currentCenterY = graphCenterY;
                isFirstFrame = false;
            }} else {{
                // Плавная интерполяция
                currentScale = lerp(currentScale, targetScale, SMOOTH_SCALE_CONFIG.lerpFactor);
                currentCenterX = lerp(currentCenterX, graphCenterX, SMOOTH_SCALE_CONFIG.lerpFactor);
                currentCenterY = lerp(currentCenterY, graphCenterY, SMOOTH_SCALE_CONFIG.lerpFactor);
            }}
            
            // Применяем камеру
            network.moveTo({{
                position: {{ x: currentCenterX, y: currentCenterY }},
                scale: currentScale,
                animation: false  // Без встроенной анимации — мы сами делаем плавность
            }});
            
            // Следующий кадр
            requestAnimationFrame(smoothScaleAnimation);
        }}
        
        // Запускаем плавное масштабирование
        requestAnimationFrame(smoothScaleAnimation);
        
        // Параметры анимации
        const ANIMATION_CONFIG = {{
            nodeDelay: 350,       // Задержка между добавлением нодов (мс)
            nodeBatchSize: 1,     // По 1 ноду за раз для плавности
            connectedNodeFrequency: 3  // Каждый N-й узел должен иметь связь с существующими
        }};
        
        // Перемешиваем узлы для хаотичного появления
        const remainingNodes = [...allNodesData].sort(() => Math.random() - 0.5);
        const nodeQueue = [];  // Очередь узлов для добавления
        
        // Строим индекс соседей для быстрого поиска связанных узлов
        const nodeNeighbors = {{}};
        for (const [sourceId, sourceEdges] of Object.entries(nodeToEdges)) {{
            if (!nodeNeighbors[sourceId]) nodeNeighbors[sourceId] = new Set();
            for (const edge of sourceEdges) {{
                nodeNeighbors[sourceId].add(edge.to);
                if (!nodeNeighbors[edge.to]) nodeNeighbors[edge.to] = new Set();
                nodeNeighbors[edge.to].add(sourceId);
            }}
        }}
        
        let nodeCounter = 0;
        // ОПТИМИЗАЦИЯ: Используем Set для быстрого поиска
        let addedNodeIds = new Set();
        let addedEdgeIds = new Set();
        
        // ОПТИМИЗАЦИЯ: Предвычисляем обратный индекс рёбер (target -> edges)
        const edgesByTarget = {{}};
        for (const [sourceId, sourceEdges] of Object.entries(nodeToEdges)) {{
            for (const edge of sourceEdges) {{
                if (!edgesByTarget[edge.to]) {{
                    edgesByTarget[edge.to] = [];
                }}
                edgesByTarget[edge.to].push({{ ...edge, sourceId }});
            }}
        }}
        
        // Функция выбора следующего узла для добавления
        function pickNextNode() {{
            nodeCounter++;
            
            // Каждый N-й узел — пытаемся найти связанный с уже добавленными
            if (nodeCounter % ANIMATION_CONFIG.connectedNodeFrequency === 0 && addedNodeIds.size > 0) {{
                // Ищем узел, который связан с уже добавленными
                for (let i = 0; i < remainingNodes.length; i++) {{
                    const candidate = remainingNodes[i];
                    const neighbors = nodeNeighbors[candidate.id];
                    
                    if (neighbors) {{
                        for (const neighborId of neighbors) {{
                            if (addedNodeIds.has(neighborId)) {{
                                // Нашли связанный узел — удаляем из remainingNodes и возвращаем
                                remainingNodes.splice(i, 1);
                                return candidate;
                            }}
                        }}
                    }}
                }}
            }}
            
            // Если не нашли связанный или не пора искать — берём первый из оставшихся
            if (remainingNodes.length > 0) {{
                return remainingNodes.shift();
            }}
            
            return null;
        }}
        
        // Функция добавления узлов с их рёбрами
        function addNodesWithEdges() {{
            if (remainingNodes.length === 0) {{
                console.log('Все элементы добавлены. Физика формирует кластеры...');
                return;
            }}
            
            const nodeBatch = [];
            const edgeBatch = [];
            
            for (let i = 0; i < ANIMATION_CONFIG.nodeBatchSize && remainingNodes.length > 0; i++) {{
                const node = pickNextNode();
                if (!node) break;
                
                nodeBatch.push(node);
                addedNodeIds.add(node.id);
                
                // Добавляем исходящие рёбра к уже существующим узлам
                const nodeEdges = nodeToEdges[node.id] || [];
                for (const edge of nodeEdges) {{
                    const edgeId = edge.from + '-' + edge.to;
                    if (addedNodeIds.has(edge.to) && !addedEdgeIds.has(edgeId)) {{
                        edgeBatch.push({{ ...edge, id: edgeId }});
                        addedEdgeIds.add(edgeId);
                    }}
                }}
                
                // ОПТИМИЗАЦИЯ: Используем предвычисленный индекс для входящих рёбер
                const incomingEdges = edgesByTarget[node.id] || [];
                for (const edge of incomingEdges) {{
                    const edgeId = edge.from + '-' + edge.to;
                    if (addedNodeIds.has(edge.sourceId) && !addedEdgeIds.has(edgeId)) {{
                        edgeBatch.push({{ ...edge, id: edgeId }});
                        addedEdgeIds.add(edgeId);
                    }}
                }}
            }}
            
            // Добавляем узлы
            if (nodeBatch.length > 0) {{
                nodes.add(nodeBatch);
            }}
            
            // Добавляем рёбра
            if (edgeBatch.length > 0) {{
                edges.add(edgeBatch);
            }}
            
            setTimeout(addNodesWithEdges, ANIMATION_CONFIG.nodeDelay);
        }}
        

        
        // Запускаем анимацию
        setTimeout(() => {{
            addNodesWithEdges();
        }}, 300);
    </script>
</body>
</html>'''
    
    return html_content


def create_banner_visualization():
    """Main function to create banner visualization."""
    logging.info("--- Creating banner graph visualization ---")
    
    graph = load_graph()
    if graph is None:
        return False
    
    if graph.number_of_nodes() == 0:
        logging.warning("Graph is empty. Visualization not possible.")
        return False
    
    logging.info(f"Graph contains {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    
    nodes_data, edges_data, node_to_edges = prepare_banner_data(graph)
    html_content = generate_banner_html(nodes_data, edges_data, node_to_edges)
    
    # Save HTML
    html_path = "graph_banner_visualization.html"
    try:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        full_path = os.path.abspath(html_path)
        logging.info(f"HTML file saved: {full_path}")
        
        # Open in browser
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


if __name__ == "__main__":
    logging.info("Starting banner visualization creation")
    
    if create_banner_visualization():
        logging.info("Banner visualization created successfully!")
        logging.info("Physics remains active — observe cluster formation")
    else:
        logging.error("Error creating visualization")
