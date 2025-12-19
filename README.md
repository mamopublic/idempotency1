# Idempotency 1: Generative Stability Experiment

> **Status**: Active Research Phase
> **Focus**: Information Theory, Generative Stability, Attractor Dynamics

## Abstract

Idempotency 1 is an experimental framework designed to investigate the **stability of information** in iterative generative loops. By cycling semantic concepts through multiple modalities—**Text Description $\to$ Code Structure $\to$ Visual Rendering $\to$ Text Description**—we aim to quantify information loss, semantic drift, and the emergence of stable "attractor" states in Large Multimodal Models (LMMs).

This project treats the generative loop as a **dynamical system**, asking whether rigorous architectural diagrams converge to a stable fixed point (idempotency) or diverge into chaos (semantic collapse) when subjected to repeated lossy compression/decompression by AI models.

## Key Research Questions

1.  **Attractor Dynamics**: Do initial prompts converge to specific, stable architectural patterns ("basins of attraction"), regardless of minor phrasing variations?
2.  **Semantic Drift**: How does the creativity temperature of the Vision-Language Model affect the rate of information decay?
3.  **Self-Correction**: Can a feedback loop with strict syntax constraints (Mermaid.js) act as a forcing function to stabilize "hallucinated" details?

## Getting Started

### Prerequisites
- **Python 3.10+**
- **Node.js & npm** (Required for Mermaid rendering interaction)

### Installation

1.  **Clone & Install Python Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Install Node Dependencies**:
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

### 1. Batch Experiments (Recommended)
Run a suite of experiments defined in the prompts configuration. This is the primary method for gathering statistical data.

```bash
python main.py --batch config/prompts/batch_prompts.txt
```

**Output**: Creates a structured dataset in `experiments/` containing:
- **Scientific Report**: `batch_report.pdf` (Convergence rates, Stability scores).
- **Visualizations**: `batch_dashboard_*.png` (Semantic evolution graphs).
- **Raw Data**: `trajectory.json` (Full text/image history for every iteration).

### 2. Single Probe
Run a quick probe for a single concept to test a hypothesis.

```bash
# Probing the stability of a "Microservices Payment System" concept
python main.py --prompt "A distributed payment gateway architecture with sharded databases" --name "payment_probe_1"
```

## Research Tools

The repository includes specialized tools for performing controlled experiments:

- **Hyperparameter Sweeps**: Vary models and temperatures to test sensitivity.
    - `python tools/run_sweep.py`
- **Dry Run**: Verify the pipeline integrity before long-running experiments.
    - `python tools/run_dry_run.py`

## Documentation & Roadmap

- **[Methodology & Architecture](docs/project_overview.md)**: Deep dive into the experimental loop, metric definitions (SigLIP/Titan embeddings), and system architecture.
- **[Research Roadmap](RESEARCH_TODO.md)**: Open questions, planned experiments (Temperature Studies, Information Bottleneck), and publication goals.
