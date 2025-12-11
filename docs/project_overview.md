# Project Overview

This project implements an **Idempotency Experiment Runner** that tests the stability of generative AI models when converting Text to Diagrams and back to Text.

## Core Logic: The Feedback Loop

The core functionality is encapsulated in the `ExperimentRunner` class (`src/experiment.py`). It executes a cycle of generation and extraction to measure information loss or drift over multiple iterations.

### The Loop
For a given number of iterations (controlled by `config.yaml`):

1.  **Text-to-Diagram**:
    *   **Input**: A text prompt (e.g., "A simple e-commerce system architecture").
    *   **Process**: An LLM (via OpenRouter) generates Mermaid.js code representing this system.
    *   **Output**: `.mmd` file.

2.  **Rendering**:
    *   **Input**: `.mmd` file.
    *   **Process**: The `MermaidGenerator` uses the local `mmdc` CLI tool to render the diagram.
    *   **Output**: `.png` image.

3.  **Vision-to-Text**:
    *   **Input**: `.png` image.
    *   **Process**: A Vision-Language Model (VLM) views the image and extracts a text description of the system architecture.
    *   **Output**: A new text prompt.

4.  **Loop**:
    *   This extracted text becomes the **Input** for the next iteration.

## Analysis & Metrics

The project uses a **decoupled architecture**:
1.  **Generation Phase**: Runs the loop and saves raw data to `trajectory.json`.
2.  **Analysis Phase**: Processes the trajectory to compute global similarity matrices (simultaneously comparing all iterations against all others).

### Metrics Matrices
The analysis (`src/post_process.py`) produces `metrics.json` containing:

*   **Text Matrix**:
    *   *Method*: Character-level similarity (`difflib`).
    *   *Description*: A square matrix where cell `[i][j]` is the text similarity between Prompt *i* and Prompt *j*.

*   **Semantic Matrix**:
    *   *Method*: Text Embeddings (SigLIP/Titan) + Cosine Similarity.
    *   *Description*: A square matrix measuring conceptual similarity between all pairs of prompts.

*   **Visual Matrix**:
    *   *Method*: Image Embeddings (SigLIP/Titan) + Cosine Similarity.
    *   *Description*: A square matrix measuring visual similarity between all pairs of generated images. Cell `[0][0]` is typically 0 if no initial image exists.

## Architecture

*   **`main.py`**: Entry point. Runs experiment generation then triggers post-processing.
*   **`src/experiment.py`**: Orchestrates the generative loop. Saves `trajectory.json` and images.
*   **`src/post_process.py`**: Loads a completed experiment and computes similarity matrices.
*   **`src/llm.py`**: Handles Interactions with OpenRouter API (Text and Vision models).
*   **`src/diagram.py`**: A wrapper around the `mmdc` tool for rendering images.
*   **`src/analysis.py`**: Logic for embedding models and computing similarity scores (Text, Image, Visual).
*   **`src/batch.py`**: Logic for running multiple experiments from a file.
*   **`src/config_loader.py`**: Utilities for reading YAML configuration.

## Configuration & Tuning

The behavior of the generative models is controlled via `config.yaml`.

*   **`text_temperature`** (Default: `0.1`): Controls the randomness of the Code Generation step.
    *   *Rationale*: Set this **low** because generating formal syntax (Mermaid.js) requires precision. High temperatures increase the risk of invalid syntax or hallucinated formatting.
*   **`vision_temperature`** (Default: `0.7`): Controls the randomness of the Vision Extraction step.
    *   *Rationale*: Set this **higher** because describing an image is a creative task. A very low temperature (0.0) can lead to terse, repetitive descriptions. A bit of randomness encourages richer, more natural prompts for the subsequent iteration.
