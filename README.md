# Generative Stability: Attractor Dynamics in LMMs

> **Research Framework**: Investigating the fixed-point stability and information entropy of Large Multimodal Models (LMMs) through Iterated Function Systems (IFS).

## Abstract

This project provides a rigorous experimental framework to quantify the **semantic stability** of multimodal generative loops. By routing information through a cycle of **Natural Language $\to$ Formal Code $\to$ Visual Manifestation $\to$ Natural Language**, we treat the AI model as a transformer in a high-dimensional semantic space. 

Our core research objective is to determine if concepts in latent space possess **stable attractors** (points of idempotency) or if they exhibit **divergent drift** (semantic collapse) when subjected to repeated lossy cross-modal translation.

## Research Dimensions

1.  **Attractor Dynamics**: Mapping the "basins of attraction" in embedding space. Does a "Cloud Architecture" prompt always collapse to the same canonical diagram?
2.  **Semantic Entropy**: Using k-means clustering and Shannon entropy on SigLIP embeddings to measure the "crystallization" vs. "diffusion" of information over iterations.
3.  **Cross-Modal Information Bottlenecks**: Quantifying information loss at the Vision $\to$ Text transition as a function of model temperature and prompt complexity.

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
- **[Prompt Matrix](docs/prompt_matrix.md)**: Formal description of the 3x3x3 Scientific Matrix (27 prompts) used in our experiments.
- **[Research Roadmap](RESEARCH_TODO.md)**: Open questions, planned experiments (Temperature Studies, Information Bottleneck), and publication goals.
