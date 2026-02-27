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

    # Helper to get SECOND vs last (intra-loop stability: first VLM output vs final output)
    def get_second_last(matrix, is_visual=False):
        # Semantic: Compare 1 (first LLM-output prompt) vs -1 (last)
        # Visual:   Compare 2 (second image) vs -1 (last)
        start_idx = 2 if is_visual else 1
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

    sem_sl = [get_second_last(d["semantic_matrix"], False) for d in data]
    vis_sl = [get_second_last(d["visual_matrix"], True) for d in data]

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
    
    def create_figure(metric_name, evolutions, convergences, first_lasts, second_lasts=None):
        fig, axes = plt.subplots(1, 3, figsize=(24, 6))
        
        # 1. Evolution (Line Plot)
        ax = axes[0]
        for i, series in enumerate(evolutions):
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
        x = np.arange(len(run_names))
        ax.bar(x, convergences, color='skyblue')
        ax.set_xticks(x)
        ax.set_xticklabels(run_names, rotation=45, ha='right')
        ax.set_title("Convergence Speed (Iter > 0.95)")
        ax.set_ylabel("Iteration")
        
        # 3. Stability Bar Chart (grouped when second_lasts is provided)
        ax = axes[2]
        if second_lasts is not None:
            # Grouped bar: seed-to-last (dark green) + loop-to-last (light green)
            width = 0.38
            bars1 = ax.bar(x - width / 2, first_lasts,  width, color='#2ecc71', label='Seed → Last (style gap)')
            bars2 = ax.bar(x + width / 2, second_lasts, width, color='#a8e6cf', label='Iter 1 → Last (intra-loop)')
            ax.legend(fontsize=8)
            ax.set_title(f"Stability: {metric_name}\n(seed→last vs iter1→last)")
        else:
            ax.bar(x, first_lasts, color='lightgreen')
            ax.set_title(f"Stability: First vs Last {metric_name}")
        ax.set_xticks(x)
        ax.set_xticklabels(run_names, rotation=45, ha='right')
        ax.set_ylabel("Similarity")
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        return fig

    # Semantic Dashboard — pass second_lasts to show grouped stability bars
    fig_sem = create_figure("Semantic", sem_evo, sem_conv, sem_fl, second_lasts=sem_sl)
    fig_sem.savefig(os.path.join(batch_dir, "batch_dashboard_p1_semantic.png"))
    plt.close(fig_sem)
    
    # Visual Dashboard — pass second_lasts to show grouped stability bars
    fig_vis = create_figure("Visual", vis_evo, vis_conv, vis_fl, second_lasts=vis_sl)
    fig_vis.savefig(os.path.join(batch_dir, "batch_dashboard_p1_visual.png"))
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
    fig_dist.savefig(os.path.join(batch_dir, "batch_dashboard_p1_distances.png"))
    plt.close(fig_dist)
    
    # Trajectory Plot (2D Embedding Space)
    fig_traj, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 1. Semantic Trajectory (Semantic Sim vs Iteration)
    ax = axes[0]
    for i, d in enumerate(data):
        # Extract semantic similarity to initial prompt over time
        if "semantic_matrix" in d and d["semantic_matrix"]:
            matrix = np.array(d["semantic_matrix"])
            # Similarity of each iteration to iteration 0
            traj = matrix[0, :]  # Row 0 = similarities to initial
            iterations = range(len(traj))
            
            # Plot with markers at start/end
            ax.plot(iterations, traj, marker='o', markersize=3, alpha=0.7, label=run_names[i])
            ax.scatter([0], [traj[0]], s=100, marker='s', edgecolors='black', linewidths=2, zorder=5)  # Start
            ax.scatter([len(traj)-1], [traj[-1]], s=100, marker='*', edgecolors='black', linewidths=2, zorder=5)  # End
    
    ax.set_title("Semantic Trajectory\n(Similarity to Initial Prompt)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Semantic Similarity")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    # 2. 2D Phase Space (Semantic vs Visual Stability)
    ax = axes[1]
    for i, d in enumerate(data):
        # Extract final 10 iterations for stability analysis
        if "semantic_matrix" in d and "visual_matrix" in d:
            sem_matrix = np.array(d["semantic_matrix"])
            vis_matrix = np.array(d["visual_matrix"])
            
            n = len(sem_matrix)
            if n > 10:
                # Last 10 iterations
                sem_vals = [sem_matrix[0, j] for j in range(n-10, n)]  # Sim to initial
                vis_vals = [vis_matrix[1, j] for j in range(max(1, n-10), n)]  # Sim to first image
                
                # Plot trajectory
                ax.plot(sem_vals, vis_vals, marker='o', markersize=4, alpha=0.6, label=run_names[i])
                ax.scatter([sem_vals[0]], [vis_vals[0]], s=100, marker='s', edgecolors='black', linewidths=2, zorder=5)
                ax.scatter([sem_vals[-1]], [vis_vals[-1]], s=100, marker='*', edgecolors='black', linewidths=2, zorder=5)
    
    ax.set_title("Phase Space (Last 10 Iterations)\n■ = Start, ★ = End")
    ax.set_xlabel("Semantic Similarity (to Initial)")
    ax.set_ylabel("Visual Similarity (to First Image)")
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_traj.savefig(os.path.join(batch_dir, "batch_dashboard_p1_trajectory.png"), dpi=150)
    plt.close(fig_traj)
    
    # --- Window Analysis Dashboards ---
    # Collect window metrics from each experiment
    text_drifts = []
    text_radii = []
    vis_drifts = []
    vis_radii = []
    
    for d in data:
        if "text_window_drifts" in d:
            text_drifts.append(d["text_window_drifts"])
            text_radii.append(d["text_window_radii"])
        if "visual_window_drifts" in d:
            vis_drifts.append(d["visual_window_drifts"])
            vis_radii.append(d["visual_window_radii"])
    
    # Text Window Drift
    if text_drifts:
        fig_wd, ax = plt.subplots(1, 1, figsize=(12, 6))
        for i, drifts in enumerate(text_drifts):
            indices = range(len(drifts))
            ax.plot(indices, drifts, marker='o', markersize=3, alpha=0.7, label=run_names[i])
        ax.set_title("Text Embedding: Window Center Drift")
        ax.set_xlabel("Window Index")
        ax.set_ylabel("Drift ||C_k - C_{k-1}||")
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig_wd.savefig(os.path.join(batch_dir, "batch_dashboard_p1_window_drift_text.png"))
        plt.close(fig_wd)
    
    # Text Window Radius
    if text_radii:
        fig_wr, ax = plt.subplots(1, 1, figsize=(12, 6))
        for i, radii in enumerate(text_radii):
            indices = range(len(radii))
            ax.plot(indices, radii, marker='s', markersize=3, alpha=0.7, label=run_names[i])
        ax.set_title("Text Embedding: Window Radius (Dispersion)")
        ax.set_xlabel("Window Index")
        ax.set_ylabel("Radius (Max Dist to Center)")
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig_wr.savefig(os.path.join(batch_dir, "batch_dashboard_p1_window_radius_text.png"))
        plt.close(fig_wr)
    
    # Visual Window Drift
    if vis_drifts:
        fig_vd, ax = plt.subplots(1, 1, figsize=(12, 6))
        for i, drifts in enumerate(vis_drifts):
            indices = range(len(drifts))
            ax.plot(indices, drifts, marker='o', markersize=3, alpha=0.7, label=run_names[i])
        ax.set_title("Visual Embedding: Window Center Drift")
        ax.set_xlabel("Window Index")
        ax.set_ylabel("Drift ||C_k - C_{k-1}||")
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig_vd.savefig(os.path.join(batch_dir, "batch_dashboard_p1_window_drift_visual.png"))
        plt.close(fig_vd)
    
    # Visual Window Radius
    if vis_radii:
        fig_vr, ax = plt.subplots(1, 1, figsize=(12, 6))
        for i, radii in enumerate(vis_radii):
            indices = range(len(radii))
            ax.plot(indices, radii, marker='s', markersize=3, alpha=0.7, label=run_names[i])
        ax.set_title("Visual Embedding: Window Radius (Dispersion)")
        ax.set_xlabel("Window Index")
        ax.set_ylabel("Radius (Max Dist to Center)")
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig_vr.savefig(os.path.join(batch_dir, "batch_dashboard_p1_window_radius_visual.png"))
        plt.close(fig_vr)
    
    print(f"Batch dashboards saved to {batch_dir}")
