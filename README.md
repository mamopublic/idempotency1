# Idempotency Experiment Runner

A robust, self-healing framework for running iterative generative AI experiments. It tests the stability ("idempotency") of LLM-generated artifacts (Mermaid diagrams) by cycling them through a **Description -> Code -> Render -> Description** loop and measuring semantic and visual convergence over time.

## Key Features

- **🔄 Batch Processing**: Run multiple prompts in parallel with isolated environments.
- **🛡️ Self-Healing Generation**: Automatically detects Mermaid rendering errors (syntax issues, timeouts) and uses an LLM-based feedback loop to fix the code on the fly.
- **📊 Comprehensive Reporting**: Generates detailed **Markdown** and **PDF** reports for each batch, including:
    - Aggregate statistics (Convergence speed, Stability scores, Cost).
    - Embedded visualizations (Semantic evolution, Visual convergence).
    - Detailed logs of every experiment.
- **💰 Cost Approximation**: Tracks token usage and estimates costs based on configurable model pricing.
- **📉 Advanced Metrics**: Analytics for **Semantic** (text embedding), **Visual** (image embedding), and **Cross-Modal** similarity.

## Setup

### Prerequisites
- **Python 3.10+**
- **Node.js & npm** (Required for Mermaid rendering)

### Installation

1.  **Clone & Install Python Deps**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Install Node Deps**:
    ```bash
    npm install
    ```
    *This installs `@mermaid-js/mermaid-cli` locally for rendering.*

3.  **Environment Variables**:
    Create a `.env` file (or use `env.sh`):
    ```bash
    export OPENROUTER_API_KEY="sk-or-..."
    ```

## Usage

### 1. Batch Mode (Recommended)
Run a suite of experiments defined in a file (line-separated prompts or JSON list).

```bash
python main.py --batch batch_prompts.txt
```

**Output**: Creates a timestamped folder in `experiments/` (e.g., `experiments/batch_20251208_185012/`) containing:
- `batch_report.pdf` & `batch_report.md`: The full scientific report.
- `batch_dashboard_*.png`: Aggregate visualizations.
- `prompt_X/`: Subfolders for individual experiment data (`trajectory.json`, images).

### 2. Single Experiment
Run a quick test for a single prompt.

```bash
python main.py --prompt "A distributed payment gateway architecture" --name "payment_test"
```

## Configuration (`config/config.yaml`)

Control every aspect of the experiment via `config/config.yaml`.

### Experiment Settings
```yaml
experiment:
  iterations: 10          # Number of cycles per prompt
  output_dir: "experiments"
```

### Models & Parameters
```yaml
models:
  text_model: "google/gemini-2.0-flash-001"
  vision_model: "google/gemini-2.0-flash-001"
  generation_params:
    text_temperature: 0.1   # Keep low for code stability
    vision_temperature: 0.7 # Higher for descriptive variety
  embedding_model: "google/siglip-so400m-patch14-384" # or Titan
```

### Cost Tracking
Define rates per 1M tokens to estimate spend.
```yaml
costs:
  google/gemini-2.0-flash-001:
    input: 0.10
    output: 0.40
```

## Architecture

*   **`src/experiment.py`**: Core logic. Manages the loop and the **retry/repair** mechanism.
*   **`src/batch.py`**: Orchestrates parallel/sequential runs and triggers reporting.
*   **`src/llm.py`**: Handles OpenRouter API calls and the `fix_diagram_code` agent.
*   **`src/diagram.py`**: Wrapper for the Mermaid CLI renderer.
*   **`src/analysis.py`**: Embeddings engine (SigLIP/Titan) and similarity math.
