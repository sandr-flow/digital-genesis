# Digital Genesis

**Digital Genesis** is an advanced AI bot driven by Long-Term Memory (LTM), associative graph reasoning, and background self-reflection capabilities. It moves beyond stateless interactions by maintaining a persistent "stream of consciousness" and a conceptual graph, allowing it to remember facts, form associations, and evolve its understanding over time.

## Table of Contents

- [Background](#background)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Visualizations](#visualizations)
- [License](#license)

## Background

Traditional chatbots often suffer from amnesia or limited context windows. Digital Genesis addresses this by implementing a dual-memory architecture:
1.  **Vector Memory (LTM):** Uses **ChromaDB** to store and retrieve "memories" (facts, thoughts, dialogues) based on semantic similarity.
2.  **Graph Memory (NetworkX):** Maintains a knowledge graph of concepts and their relationships, enabling associative reasoning and role tracking.

Powered by **Google Gemini** (via `google-generativeai`), the bot acts not just as a responder but as an autonomous entity that "thinks," "reflects," and "dreams" (via background reflection cycles).

## Key Features

-   **Deep LTM**: Persistent storage of every interaction, thought, and fact.
-   **Associative Reasoning**: Graph-based connections allow the bot to link related concepts even if they weren't mentioned in the current context.
-   **Self-Reflection**: A background process (`Force Reflection`) that analyzes past memories to form new, higher-level concepts and eliminate conversational noise.
-   **Telegram Interface**: A robust, async bot interface built with `aiogram 3.x`.
-   **Interactive Visualizations**: Generate HTML-based interactive graphs of the bot's memory space.

## Project Structure

```text
digital_genesis/
├── core/
│   ├── graph/           # Knowledge Graph logic (NetworkX)
│   ├── ltm/             # Long-Term Memory (LTM) & ChromaDB manager
│   └── reflection/      # Background reflection engine
├── handlers/            # Telegram bot command and message handlers
├── scripts/             # Utility toolset
│   ├── analyze_graph.py           # Analyzes graph structure and connectivity
│   ├── concepts_analyze.py        # Hygiene inspector for Conceptual Core
│   ├── db_inspector.py            # Low-level ChromaDB inspection tool
│   ├── force_reflection.py        # Manually triggers reflection cycles
│   ├── graph_fallback_server.py   # API server reading directly from graph file
│   ├── memory_api_server.py       # API server for debugging specific memories
│   ├── memory_cleaner.py          # Tool to prune short/duplicate memories
│   ├── memory_hygiene_inspector.py# Analyzes LTM for quality and duplicates
│   └── vizualize_graph_*.py       # Various graph visualization generators
├── services/            # External integrations (Gemini, Logging)
├── utils/               # Helpers (Formatters, Keyboards)
├── .env.example         # Template for environment variables
├── config.py            # Central configuration
├── main.py              # Application entry point
└── requirements.txt     # Python dependencies
```

## Installation

### Prerequisites
- Python 3.10+
- A Google Cloud Project with Gemini API access ([Google AI Studio](https://aistudio.google.com/))
- A Telegram Bot Token ([BotFather](https://t.me/BotFather))

### Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/digital_genesis.git
    cd digital_genesis
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # Windows:
    .\venv\Scripts\activate
    # Linux/Mac:
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

1.  **Environment Variables:**
    Copy `.env.example` to `.env` and fill in your credentials:
    ```bash
    cp .env.example .env
    ```

    | Variable | Description |
    | :--- | :--- |
    | `GEMINI_API_KEY` | Your Google Gemini API Key |
    | `TELEGRAM_BOT_TOKEN` | Your Telegram Bot Token |
    | `ADMIN_ID` | Telegram User ID of the bot admin |

2.  **App Config:**
    Adjust internal settings in `config.py` if necessary (e.g., model names, memory thresholds).

## Usage

Start the bot:
```bash
python main.py
```

The bot will initialize:
1.  Connect to ChromaDB (LTM).
2.  Load the Knowledge Graph.
3.  Start the Telegram polling loop.

### Scripts usage
Run utilities from the root directory:
```bash
# Generate a visual map of the brain
python -m scripts.vizualize_graph

# Analyze memory health
python -m scripts.memory_hygiene_inspector

# Force a reflection cycle on a specific memory ID
python -m scripts.force_reflection
```

## Visualizations

### 🌐 Live Demo
Explore the **[Interactive RAG Pipeline Visualization](https://endearing-belekoy-40e114.netlify.app/)** to see how the bot processes information.

### 📂 Local Artifacts
The project generates interactive HTML files to visualize the bot's internal state:
-   `interactive_graph_visualization.html`: Full interactive graph (PyVis).
-   `graph_banner_visualization.html`: A simplified, banner-like view.
-   `public_graph_visualization.html`: Sanitized view for public demonstration.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
