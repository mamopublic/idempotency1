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

### Observed Behavior: Register Shift at the Bottleneck, Structural Stability Within the Loop

Based on two replicated temperature sweep experiments (n=108 each, spanning vision temperatures 0.1–1.0):

**Key Results (sweep `vision_sweep_20260225_205550`, 108 runs, 30 iterations each):**

| Metric | Mean | Min | Max |
|---|---|---|---|
| **Semantic stability — seed→last** (style gap) | 56.9% | 35.9% | 73.2% |
| **Semantic stability — iter 1→last** (intra-loop) | **80.4%** | 49.4% | 100% |
| **Visual stability — iter 1→last** | **97.3%** | 63.0% | 100% |

**Interpretation — there are two phenomena at play, not one:**

1. **Register shift at the bottleneck (~57% seed→last)**: The ~43% gap between the human-written seed and the final output is *not* continuous semantic drift. It is a one-time, irreversible **style transformation** that happens at iteration 1, when the VLM first re-describes the diagram in its own verbose register (e.g., "a pink rectangle labelled..."). The LLM generating Mermaid from this VLM output then compresses back to the same structural diagram, so the **Mermaid code converges** even though the *text prompts* remain stylistically distant from the seed.

2. **Intra-loop stability (~80% iter 1→last)**: Once the system is in the VLM's output register, it is considerably more stable. Step-by-step semantic similarity between consecutive VLM descriptions sits at 0.85–0.98; the LLM consistently recovers the same structural diagram from these descriptions. The loop has **found a structural attractor** — not in the seed's vocabulary, but in the VLM's idiomatic representation of the structure.

3. **Visual representations are highly stable (97.3%)**: The rendered diagram changes very little after the first few iterations, regardless of text prompt variance.

**Temperature effect** (mild, contrary to initial hypothesis):

| T | Semantic intra-loop |
|---|---|
| 0.1 | 85.5% |
| 0.4 | 81.1% |
| 0.7 | 79.0% |
| 1.0 | 75.8% |

Higher vision temperature reduces intra-loop semantic stability by ~10 percentage points across the range. This is real but modest; structural convergence (Mermaid topology) remains consistent across all temperatures.

**Replication note**: A second sweep run with Mermaid cosmetics normalization (stripping whitespace/style from `.mmd` files before rendering) produced nearly identical aggregate numbers, confirming the structural attractor is driven by LLM topology compression, not cosmetic code features.

### Revised Research Question Status

- ✅ **Structural attractor found** — The LLM reliably maps semantically equivalent VLM descriptions back to the same Mermaid topology, acting as a topological compressor.
- ✅ **Register shift at the bottleneck** — The VLM induces a one-time, stable style transform from terse human seed to verbose visual narration.
- ❌ **Fixed-point in prompt space** (not observed) — The VLM's text output is never identical to the human seed; seed→last similarity (~57%) reflects a permanent vocabulary gap, not ongoing drift.
- ❌ **Strong limit cycles** (not observed) — The system does not oscillate discretely; it wanders within a bounded region of the VLM's output register.

### Significance

The loop is best described as a **two-stage compression**: a one-time register normalization (seed → VLM style) followed by ongoing structural stabilization (VLM style → Mermaid topology). The "irreducible kernel" is the diagram's topological skeleton — the set of nodes and edges — which survives every cycle regardless of how the VLM phrases its description.

Implications:
- **Retrieval-Augmented Generation**: A single prompt-to-image-to-prompt cycle changes phrasing significantly (~43% style shift), but subsequent cycles are relatively stable. The first reformulation is the dangerous one.
- **Agentic Workflows**: Multi-step reasoning chains may undergo a rapid initial style normalization rather than proportional drift — worth measuring at step 1 specifically.
- **Prompt Engineering**: Structural invariants (topology of a diagram, entity relationships) appear to have genuine attractor basins; surface phrasing does not.

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
- **Two-stage compression**: register shift at iteration 1 (seed→VLM style, ~57% similarity), then intra-loop stability (iter 1→last, ~80%); visual stability stays high throughout (~97%)
- **Mild temperature effect**: semantic intra-loop stability decreases from 85.5% (T=0.1) to 75.8% (T=1.0); visual topology remains stable across all temperatures
- **Complete data formats**: Full trajectory.json and metrics.json examples for reproducibility

The example includes all visualizations, reports (MD + PDF), and one complete trajectory showcasing the experimental methodology.

## Documentation & Roadmap

- **[Methodology & Architecture](docs/project_overview.md)**: Deep dive into the experimental loop, metric definitions (SigLIP/Titan embeddings), and system architecture.
- **[Prompt Matrix](docs/prompt_matrix.md)**: Formal description of the 3x3x3 Scientific Matrix (27 prompts) used in our experiments.
- **[Research Roadmap](RESEARCH_TODO.md)**: Open questions, planned experiments (Temperature Studies, Information Bottleneck), and publication goals.
- **[Example Sweep](examples/sample_sweep/)**: Complete example run with all artifacts and analysis.

## Acknowledgments

Developed with AI assistance (Cline + Google DeepMind's Antigravity).
