# Formal Framework: Iterated Multimodal Stability

This document outlines the theoretical and architectural underpinnings of the Idempotency Experiment. This research exists at the intersection of **Dynamical Systems Theory** and **Multimodal Latent Space Analysis**. 

We characterize the interaction between Vision-Language Models (VLM) and Large Language Models (LLM) as a lossy feedback loop, specifically an **Iterated Function System (IFS)**, to determine the limits of semantic stability.

### The Feedback Loop (The "Iterated Function System")

The experiment is modeled as an Iterated Function System (IFS) where the state $S_t$ at iteration $t$ transforms into $S_{t+1}$ through four distinct phases:

1.  **Codification ($f_{code}$)**: **Text $\to$ Code**
    *   **Input**: Natural language architectural description.
    *   **Transformation**: An LLM (e.g., Gemini 2.0 Flash) converts the ambiguous text description into strict, formal syntax (Mermaid.js).
    *   **Constraint**: The syntax requirements act as a "rigid body" constraint, forcing the model to make specific structural decisions.

2.  **Manifestation ($f_{render}$)**: **Code $\to$ Image**
    *   **Input**: Mermaid source code (`.mmd`).
    *   **Transformation**: Deterministic rendering via `mermaid-cli` (`mmdc`) to a PNG image.
    *   **Role**: This step is purely deterministic; it serves as the "grounding" of the abstract code into visual reality.

3.  **Observation ($f_{vision}$)**: **Image $\to$ Text**
    *   **Input**: The rendered PNG diagram.
    *   **Transformation**: A Vision-Language Model (VLM) observes the image and describes the architecture it sees.
    *   **Entropy**: This is the primary source of entropy (or "semantic drift"). The VLM must interpret the visual signs back into concepts, often adding or losing detail based on `vision_temperature`.

4.  **Iteration**: The output text becomes the input for $t+1$.

$$ S_{t+1} = f_{vision}(f_{render}(f_{code}(S_t))) $$

### Metrics & Measurement

To quantify stability we work in a high-dimensional semantic vector space and compute the following measures. All similarity values live in $[0, 1]$; all distance values are $1 - \text{sim}$.

#### 1. Pointwise Similarity Measures

Computed at every iteration $t$ and stored both as step-by-step sequences and as full $N \times N$ pairwise matrices.

*   **Character-level Text Similarity** (`text_sim_prev`): The `difflib.SequenceMatcher` ratio between the raw text strings at $t$ and $t-1$. A coarse, model-free baseline that catches exact wording changes and catches cases where semantic similarity might be falsely high (e.g. paraphrase of the same words).

*   **Semantic Similarity** (`semantic_sim_prev` / `semantic_matrix`): Cosine similarity between the *text embeddings* of the prompt at $t$ and $t-1$ (and in the full matrix, every pair $(i, j)$).
    *   Embedding backend: **SigLIP** (`google/siglip-so400m-patch14-384`, local) or **Amazon Titan** (`amazon.titan-embed-image-v1`, remote via Bedrock).
    *   $\text{Sem-Sim}(t, t-1) = \frac{E_t^{\text{text}} \cdot E_{t-1}^{\text{text}}}{\|E_t^{\text{text}}\| \cdot \|E_{t-1}^{\text{text}}\|}$

*   **Visual Similarity** (`visual_sim_prev` / `visual_matrix`): Cosine similarity between the *image embeddings* of the rendered diagram at $t$ and $t-1$. Captures visual convergence independently of the text loop — two diagrams can look nearly identical even if their textual descriptions differ.

*   **Cross-Modal Alignment** (`cross_modal_sim` / `cross_modal_distance`): Cosine similarity between the text embedding $T_t$ and the image embedding $I_t$ *at the same iteration*, measuring how faithfully the rendered diagram preserves the textual intent of the prompt that generated it.

#### 2. Derived Distance Metrics

The three distance channels are computed as $d = 1 - \text{sim}$ and reported on their own dashboard:

*   **Semantic Embedding Distance** (`semantic_distance_prev`)
*   **Visual Embedding Distance** (`visual_distance_prev`)
*   **Cross-Modal Distance** (`cross_modal_distance`)

Distances make divergence dynamics more visually legible (upward spikes = sudden drift) and are used in the Embedding Distances dashboard panel.

#### 3. Convergence & Stability Summary Statistics

Derived from the full matrices and series at the batch level:

*   **Convergence Speed**: The first iteration $t^*$ at which `semantic_sim_prev` (or `visual_sim_prev`) $\geq 0.95$. If the run never reaches that threshold, the convergence step is set to $N+1$ (did not converge).
*   **First-Last Stability**: $\text{sem\_matrix}[0][-1]$ and $\text{vis\_matrix}[1][-1]$ — the similarity between the very first and very last states, summarising total semantic drift over the run.

#### 4. Sliding Window Analysis

Operates on the *embedding sequence* (not the raw scalars) to reveal local dynamics that step-by-step similarity scores can miss.

For a window of size $k = 10$ sliding across the $N$ embedding vectors:

*   **Window Center** $C_k$: The mean embedding of the $k$ vectors in window $k$.
*   **Window Center Drift** (`text_window_drifts` / `visual_window_drifts`): $\|C_k - C_{k-1}\|_2$ — how much the "average semantic position" moves between consecutive windows. A system approaching a fixed point will show drift collapsing toward zero; a limit cycle will sustain periodic drift.
*   **Window Radius** (`text_window_radii` / `visual_window_radii`): $\max_{t \in \text{window}} \|E_t - C_k\|_2$ — the maximum distance from any point in the window to its center. Measures local *dispersion*: a small radius means the system is tightly clustered in that window; a large radius means high local volatility even if the mean is stable.

Both quantities are computed separately for text and image embedding streams, producing four dashboard panels (`window_drift_text`, `window_radius_text`, `window_drift_visual`, `window_radius_visual`).

#### 5. Semantic Entropy

A global, distributional measure of how spread out the system's states are across the model's latent space, computed across an entire temperature sweep rather than within a single run.

The pipeline:

1.  **Fit a global k-means model** ($K = 16$ clusters, one model per temperature per modality) on all embeddings collected from every prompt run at that temperature.
2.  **Soft cluster assignment**: For each embedding $E_t$, compute distances to all $K$ centroids and convert to a probability distribution $p$ via softmax with temperature $\tau = 0.1$:
    $$p_i = \frac{\exp(-d_i / \tau)}{\sum_j \exp(-d_j / \tau)}$$
    Lower $\tau$ produces harder assignments (winner-take-all); the value 0.1 yields near-hard assignments while remaining differentiable.
3.  **Shannon Entropy**:
    $$H(E_t) = -\sum_{i=1}^{K} p_i \log p_i \quad [\text{nats}]$$
    Maximum entropy $\log K \approx 2.77$ nats corresponds to a uniform distribution across all clusters (maximal semantic wandering). Entropy $\approx 0$ means the embedding is firmly assigned to a single cluster (semantic lock-in).

Entropy is computed per step, averaged across runs, and plotted as:
*   **Entropy over time** — how semantic diversity evolves across iterations for each temperature.
*   **Final entropy vs temperature** — whether higher vision temperature systematically increases latent-space diversity at convergence.

## Theoretical Framework

### Attractor Dynamics
We hypothesize that for a given concept (e.g., "Load Balancer"), there exists a **basin of attraction** in the model's latent space.
*   **Strong Attractor**: Regardless of initial phrasing, the system rapidly converges to a specific canonical diagram (e.g., a specific box-and-arrow arrangement) and stays there.
*   **Limit Cycles**: The system oscillates between two or three approximately stable states.
*   **Chaos**: The system drifts endlessly, constantly adding or hallucinating new components without stabilizing.

### Information Compression
The loop acts as a lossy compression algorithm. Details that are not "salient" enough to survive the Vision $\to$ Text translation are stripped away. We aim to identify the **irreducible kernel** of information that survives repeated cycling.

## System Architecture

The codebase is organized to support robust, uninterruptible experimentation.

### Core Modules (`src/`)

*   **`experiment.py` (The Controller)**:
    *   Orchestrates the $N$-step loop.
    *   Maintains the "Trajectory" (the time-series history of all artifacts).
    *   **Self-Healing**: Wraps the rendering step in a retry loop. If `mermaid-cli` fails (syntax error), it feeds the error message *back* to an LLM agent (`fix_diagram_code`) to repair the code before proceeding.
*   **`llm.py` (The Agent)**:
    *   Abstracts provider interactions (OpenRouter/OpenAI).
    *   implements the temperature controls specific to each phase (low temp for code, variable temp for vision).
*   **`analysis/` (The Observer)**:
    *   **`similarity.py`**: Hosts the embedding engines. Supports pluggable backends (Local HuggingFace SigLIP vs. Remote AWS Titan) depending on compute availability.
    *   **`convergence.py`**: Calculates the stability matrices and detects convergence events (e.g., "Stopped changing at Step 14").
*   **`batch.py` (The Scaler)**:
    *   Manages concurrent or sequential execution of multiple prompt trajectories (see the **[Prompt Matrix](prompt_matrix.md)** for the experimental set).
    *   Aggregates results into the unified scientific PDF report.

### Directory Structure

```
├── main.py               # CLI Entry Point
├── config/
│   ├── config.yaml       # Experiment Parameters (Iterations, Models, Costs)
│   └── prompts/          # Batch Prompt Sets (The "Dataset")
├── experiments/          # Output Artifacts (Reports, Graphs, JSON Data)
├── src/                  # Core Python Logic
└── tools/                # Specialized Research Scripts (Sweeps, Dry Runs)
    ├── run_sweep.py      # Primary research workflow: temperature sweeps
    ├── analyze_sweep.py  # Cross-temperature completion analysis
    └── run_dry_run.py    # Verification/testing runs
```

### Experiment Output Hierarchy

**Temperature Sweep (Primary Research Mode):**
```
experiments/vision_sweep_YYYYMMDD_HHMMSS/
├── completion_analysis.md              # Cross-temperature summary
├── semantic_entropy_*_over_time_p2.png # Aggregate time-series plots
├── semantic_entropy_*_vs_temperature_p2.png # Temperature comparison plots
├── vision_temp_0.1/                    # Low temperature batch
│   ├── batch_report.md/pdf             # Per-temperature analysis
│   ├── batch_dashboard_*.png           # Temperature-specific visualizations
│   ├── semantic_entropy_*_trajectories_p2.png # Trajectory plots
│   └── prompt_N/                       # Individual experiment trajectories
│       ├── trajectory.json             # Complete iteration history
│       └── metrics.json                # Convergence metrics
├── vision_temp_0.4/                    # Medium-low temperature
├── vision_temp_0.7/                    # Medium-high temperature
└── vision_temp_1.0/                    # High temperature batch
```

**Single Batch (Legacy Mode):**
```
experiments/batch_YYYYMMDD_HHMMSS/
└── [Same structure as vision_temp_X/ subdirectories]
```

The sweep structure enables systematic study of how vision model temperature (description creativity) affects semantic stability while holding text generation temperature constant.

**Example:** See [../examples/sample_sweep/](../examples/sample_sweep/) for a complete example demonstrating this structure with real experimental data, visualizations, and detailed analysis.
