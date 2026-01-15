# Generative Stability: Attractor Dynamics in LMMs

> **Research Framework**: Investigating the fixed-point stability and information entropy of Large Multimodal Models (LMMs) through Iterated Function Systems (IFS).

## Abstract

This project provides a rigorous experimental framework to quantify the **semantic stability** of multimodal generative loops. By routing information through a cycle of **Natural Language $\to$ Formal Code $\to$ Visual Manifestation $\to$ Natural Language**, we treat the AI model as a transformer in a high-dimensional semantic space. 

Our core research objective is to determine if concepts in latent space possess **stable attractors** (points of idempotency) or if they exhibit **divergent drift** (semantic collapse) when subjected to repeated lossy cross-modal translation.

## Research Dimensions

1.  **Attractor Dynamics**: Mapping the "basins of attraction" in embedding space. Does a "Cloud Architecture" prompt always collapse to the same canonical diagram?
2.  **Semantic Entropy**: Using k-means clustering and Shannon entropy on SigLIP embeddings to measure the "crystallization" vs. "diffusion" of information over iterations.
3.  **Cross-Modal Information Bottlenecks**: Quantifying information loss at the Vision $\to$ Text transition as a function of model temperature and prompt complexity.

## Preliminary Findings

### Observed Behavior: Gradual Stabilization Without Fixed-Point Attractors

Based on temperature sweep experiments (n=108, spanning vision temperatures 0.1-1.0):

**Key Results:**
- **Visual representations stabilize rapidly** (mean: 2.1 steps, 96.5% first-last similarity)
- **Semantic descriptions drift longer** (mean: 26.8 steps, 56.2% first-last similarity)  
- **No evidence of discrete attractor basins** - concepts exhibit continuous drift rather than convergence to fixed points
- **Temperature independence**: High vision temperature (1.0) shows similar stability patterns to low (0.1)

### Theoretical Implications

1. **Lossy Compression Dynamics**: The vision→text bottleneck demonstrates continuous information loss rather than convergence to canonical forms. The system acts as a **progressive filter**, not a **point attractor**.

2. **Structural vs. Stochastic Compression**: Temperature independence suggests compression is driven by the architectural constraints of the feedback loop itself, not by model randomness.

3. **Research Question Status**:
   - ❌ **Strong attractors** (rejected) - No evidence of fixed-point convergence
   - ❌ **Limit cycles** (not observed) - No oscillation between discrete states  
   - ✅ **Continuous semantic drift** - Concepts exhibit bounded wandering in semantic space

### Significance

**The absence of attractors is itself a finding**: Multimodal iterative loops may not "crystallize" concepts as hypothesized, but rather exhibit **controlled drift** within bounded regions of semantic space. This has implications for:

- **Retrieval-Augmented Generation**: Repeated reformulation may gradually degrade query semantics
- **Agentic Workflows**: Multi-step reasoning chains may experience semantic drift proportional to chain length
- **Prompt Engineering**: "Canonical" prompt forms may not exist in latent space as discrete attractors

This work provides quantitative evidence that multimodal transformations are inherently **lossy and non-convergent**, challenging assumptions about semantic stability in LLM systems.

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

## Example Results

See **[examples/sample_sweep/](examples/sample_sweep/)** for a complete example temperature sweep experiment demonstrating:

- **Cross-temperature analysis**: 108 experiments (27 prompts × 4 temperatures)
- **Visual vs. semantic stability patterns**: Rapid visual convergence (2.1 steps) vs. prolonged semantic drift (26.8 steps)
- **Temperature independence**: Similar patterns across temperature range 0.1-1.0
- **Complete data formats**: Full trajectory.json and metrics.json examples for reproducibility

The example includes all visualizations, reports (MD + PDF), and one complete trajectory showcasing the experimental methodology.

## Documentation & Roadmap

- **[Methodology & Architecture](docs/project_overview.md)**: Deep dive into the experimental loop, metric definitions (SigLIP/Titan embeddings), and system architecture.
- **[Prompt Matrix](docs/prompt_matrix.md)**: Formal description of the 3x3x3 Scientific Matrix (27 prompts) used in our experiments.
- **[Research Roadmap](RESEARCH_TODO.md)**: Open questions, planned experiments (Temperature Studies, Information Bottleneck), and publication goals.
- **[Example Sweep](examples/sample_sweep/)**: Complete example run with all artifacts and analysis.

## Acknowledgments

Developed with AI assistance (Cline + Google DeepMind's Antigravity).
