# Sample Sweep Experiment

This directory contains a cleaned example of a complete temperature sweep experiment.

## Overview

- **Experiment Date**: December 18, 2025
- **Configuration**: Vision temperature varied [0.1, 0.4, 0.7, 1.0], text temperature fixed at 0.1
- **Prompts**: 27 architectural concepts across 4 temperatures (108 total experiments)
- **Included**: One temperature bucket (1.0) with one complete trajectory (prompt_12) as example

## Structure

```
sample_sweep/
├── README.md                       # This file
├── completion_analysis.md          # Summary across all 4 temperatures
├── semantic_entropy_*_p2.png       # Cross-temperature aggregate plots (4 files)
└── vision_temp_1.0/                # Example of one temperature bucket
    ├── batch_report.md             # Detailed analysis report (markdown)
    ├── batch_report.pdf            # Detailed analysis report (PDF)
    ├── batch_dashboard_*.png       # Visualizations (8 files)
    ├── semantic_entropy_*.png      # Trajectory plots (8 files)
    ├── semantic_entropy_*.json     # Statistics (2 files)
    └── prompt_12/                  # Example complete trajectory
        ├── trajectory.json         # 30 iterations of text/code/images
        ├── metrics.json            # Convergence analysis
        └── run_config.json         # Experiment parameters
```

## Key Findings

This experiment demonstrates the core research findings:

### Visual vs. Semantic Stability
- **Visual representations stabilize rapidly**: Mean convergence at 2.1 steps with 96.5% first-last similarity
- **Semantic descriptions drift longer**: Mean convergence at 26.8 steps with 56.2% first-last similarity
- **No discrete attractor basins**: Concepts exhibit continuous drift rather than fixed-point convergence

### Temperature Independence
- High vision temperature (1.0) shows similar stability patterns to low temperature (0.1)
- Suggests compression is driven by architectural constraints, not model randomness

### Implications
- Multimodal loops demonstrate **controlled drift** within bounded semantic regions
- Challenges assumptions about semantic stability in LLM systems
- Relevant for RAG, agentic workflows, and prompt engineering

See `completion_analysis.md` for full cross-temperature analysis and `batch_report.pdf` for detailed metrics.

## Files Explained

### Sweep Level
- **completion_analysis.md**: Summary of completion rates, costs, and failures across all temperatures
- **semantic_entropy_*_over_time_p2.png**: Time-series evolution across temperatures
- **semantic_entropy_*_vs_temperature_p2.png**: Direct temperature comparison plots

### Temperature Level (vision_temp_1.0)
- **batch_report.md/pdf**: Complete analysis of all 27 prompts at this temperature
- **batch_dashboard_*.png**: Semantic/visual analysis, distance matrices, trajectories
- **semantic_entropy_*.png**: Per-trajectory evolution plots showing convergence patterns

### Prompt Level (prompt_12)
- **trajectory.json**: Complete history of 30 iterations (text → code → image → text)
- **metrics.json**: Convergence metrics, similarity matrices, step-by-step analysis
- **run_config.json**: Exact parameters used for this experiment

## Reproducibility

To reproduce this experiment:

```bash
cd idempotency1
python tools/run_sweep.py
```

This will create a new sweep with current date/time in `experiments/`.

To regenerate just the reports from existing data:

```bash
# For single temperature
python tools/regenerate_report.py experiments/vision_sweep_YYYYMMDD_HHMMSS/vision_temp_1.0

# For all temperatures in a sweep
for temp_dir in experiments/vision_sweep_YYYYMMDD_HHMMSS/vision_temp_*/; do
    python tools/regenerate_report.py "$temp_dir"
done
```

## Data Format

The `trajectory.json` file demonstrates the complete data format used throughout the system. Each iteration contains:
- `prompt`: Text description of the architecture
- `mermaid_code`: Generated formal diagram code
- `image_path`: Path to rendered diagram
- `timestamp`: When this iteration completed
- `tokens_used`: LLM token consumption
- `cost`: API cost for this iteration

The `metrics.json` file shows the analysis structure:
- `series`: Step-by-step similarity scores
- `semantic_matrix`: Full N×N similarity matrix for text embeddings
- `visual_matrix`: Full N×N similarity matrix for image embeddings
- `convergence_step`: When similarity threshold (0.95) was first reached

## Citation

If you use this work or methodology:

```
Generative Stability: Attractor Dynamics in Large Multimodal Models
Investigation of semantic stability in iterative multimodal feedback loops
December 2025
```

## License

MIT License - See main repository LICENSE file
