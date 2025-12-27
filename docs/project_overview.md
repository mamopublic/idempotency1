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

To quantify stability, we define a distance metric $d(S_i, S_j)$ in a high-dimensional semantic vector space.

*   **Semantic Stability**: The cosine similarity between the embeddings of the prompt at iteration $t$ and $t-1$.
    *   Embedding Model: **SigLIP** (So400m) or **Amazon Titan**.
    *   $Score = \frac{E_t \cdot E_{t-1}}{\|E_t\| \|E_{t-1}\|}$
*   **Visual Convergence**: The consine similarity between the image embeddings of the rendered diagram at $t$ and $t-1$.
*   **Cross-Modal Alignment**: The similarity between the text embedding $T_t$ and the image embedding $I_t$, measuring how well the visual output represents the textual intent.

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
```
