import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_batch_dashboard(batch_dir):
    """Generates batch-level charts for all experiments in the directory."""
    
    # 1. Collect Data
    data = []
    run_names = []
    
    # Iterate through batch directory looking for metrics.json
    for item in sorted(os.listdir(batch_dir)):
        item_path = os.path.join(batch_dir, item)
        if os.path.isdir(item_path):
            metrics_path = os.path.join(item_path, "metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
                    data.append(metrics)
                    run_names.append(item)
                    
    if not data:
        print(f"No valid experiment data found in {batch_dir}")
        return

    # 2. Extract Series
    # Structure: [Exp1_Series, Exp2_Series, ...] where Series is list of steps
    
    # Helper to extract metric list for one experiment
    def get_series(metric_key, series_data):
        # Skip Iteration 0 if metric is 0 (it usually is for 'prev' metrics)
        return [s.get(metric_key, 0.0) for s in series_data if s["iteration"] > 0]

    # Helper to get first vs last
    def get_first_last(matrix, is_visual=False):
        # Matrix is square. 
        # Text/Semantic: Compare 0 (initial) vs -1 (last)
        # Visual: Compare 1 (first image) vs -1 (last image). Iter 0 has no image.
        start_idx = 1 if is_visual else 0
        if not matrix or len(matrix) <= start_idx: 
            return 0.0
        return matrix[start_idx][-1]

    # --- Prepare Plot Data ---
    sem_evo = [get_series("semantic_sim_prev", d["series"]) for d in data]
    vis_evo = [get_series("visual_sim_prev", d["series"]) for d in data]
    
    sem_dist = [get_series("semantic_distance_prev", d["series"]) for d in data]
    vis_dist = [get_series("visual_distance_prev", d["series"]) for d in data]
    cm_dist = [get_series("cross_modal_distance", d["series"]) for d in data]
    
    sem_fl = [get_first_last(d["semantic_matrix"], False) for d in data]
    vis_fl = [get_first_last(d["visual_matrix"], True) for d in data]

    # Convergence (Iter > 0.95)
    def get_convergence(series):
        threshold = 0.95
        for i, val in enumerate(series):
            if val >= threshold:
                return i + 1 # 1-based iteration
        return len(series) + 1 # Didn't converge

    sem_conv = [get_convergence(s) for s in sem_evo]
    vis_conv = [get_convergence(s) for s in vis_evo]

    # --- Generate Plots ---
    
    def create_figure(metric_name, evolutions, convergences, first_lasts):
        fig, axes = plt.subplots(1, 3, figsize=(24, 6))
        
        # 1. Evolution (Line Plot)
        ax = axes[0]
        for i, series in enumerate(evolutions):
            # Pad or truncate to match length? Usually just plot
            iterations = range(1, len(series) + 1)
            ax.plot(iterations, series, marker='o', label=run_names[i])
        
        ax.set_title(f"{metric_name}: Evolution (Sim vs Prev)")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Similarity")
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Convergence (Bar Plot)
        ax = axes[1]
        x = range(len(run_names))
        ax.bar(x, convergences, color='skyblue')
        ax.set_xticks(x)
        ax.set_xticklabels(run_names, rotation=45, ha='right')
        ax.set_title("Convergence Speed (Iter > 0.95)")
        ax.set_ylabel("Iteration")
        
        # 3. First vs Last (Bar Plot)
        ax = axes[2]
        ax.bar(x, first_lasts, color='lightgreen')
        ax.set_xticks(x)
        ax.set_xticklabels(run_names, rotation=45, ha='right')
        ax.set_title(f"Stability: First vs Last {metric_name}")
        ax.set_ylabel("Similarity")
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        return fig

    # Semantic Dashboard
    fig_sem = create_figure("Semantic", sem_evo, sem_conv, sem_fl)
    fig_sem.savefig(os.path.join(batch_dir, "batch_dashboard_semantic.png"))
    plt.close(fig_sem)
    
    # Visual Dashboard
    fig_vis = create_figure("Visual", vis_evo, vis_conv, vis_fl)
    fig_vis.savefig(os.path.join(batch_dir, "batch_dashboard_visual.png"))
    plt.close(fig_vis)
    
    # Distances Dashboard
    fig_dist, axes = plt.subplots(1, 3, figsize=(24, 6))
    
    # 1. Semantic Distances
    ax = axes[0]
    for i, series in enumerate(sem_dist):
        iterations = range(1, len(series) + 1)
        ax.plot(iterations, series, marker='x', linestyle='--', label=run_names[i])
    ax.set_title("Semantic Embedding Distances (Step vs Prev)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Distance (1 - Sim)")
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Visual Distances
    ax = axes[1]
    for i, series in enumerate(vis_dist):
        iterations = range(1, len(series) + 1)
        ax.plot(iterations, series, marker='x', linestyle='--', label=run_names[i])
    ax.set_title("Visual Embedding Distances (Step vs Prev)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Distance (1 - Sim)")
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Cross-Modal Distances
    ax = axes[2]
    for i, series in enumerate(cm_dist):
        iterations = range(1, len(series) + 1)
        ax.plot(iterations, series, marker='x', linestyle='--', label=run_names[i])
    ax.set_title("Cross-Modal Distances (Prompt vs Image)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Distance (1 - Sim)")
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig_dist.suptitle("Embedding Distances Analysis")
    plt.tight_layout()
    fig_dist.savefig(os.path.join(batch_dir, "batch_dashboard_distances.png"))
    plt.close(fig_dist)
    
    print(f"Batch dashboards saved to {batch_dir}")
