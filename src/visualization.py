import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.gridspec import GridSpec

def plot_dashboard(exp_dir):
    """Generates the 5-graph dashboard for an experiment."""
    metrics_path = os.path.join(exp_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        print(f"Error: metrics.json not found in {exp_dir}")
        return

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Prepare Data
    series = data.get("series", [])
    if not series:
        print("Error: No series data found in metrics.json")
        return

    iterations = [s["iteration"] for s in series]
    text_prev = [s["text_sim_prev"] for s in series]
    sem_prev = [s["semantic_sim_prev"] for s in series]
    vis_prev = [s["visual_sim_prev"] for s in series]
    cross_sim = [s["cross_modal_sim"] for s in series]

    text_matrix = np.array(data["text_matrix"])
    sem_matrix = np.array(data["semantic_matrix"])
    vis_matrix = np.array(data["visual_matrix"])

    # Setup Figure
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 1])

    # 1. Trajectory Line Plot (Full Width)
    ax_line = fig.add_subplot(gs[0, :])
    ax_line.plot(iterations, text_prev, label="Text Sim (vs Prev)", marker='o')
    ax_line.plot(iterations, sem_prev, label="Semantic Sim (vs Prev)", marker='s')
    ax_line.plot(iterations, vis_prev, label="Visual Sim (vs Prev)", marker='^')
    ax_line.plot(iterations, cross_sim, label="Cross-Modal Sim", marker='x', linestyle='--')
    ax_line.set_title("Trajectory of Similarities")
    ax_line.set_xlabel("Iteration")
    ax_line.set_ylabel("Similarity Score")
    ax_line.set_ylim(0, 1.1)
    ax_line.legend()
    ax_line.grid(True, alpha=0.3)

    # 2. Matrices (Middle Row)
    ax_mat1 = fig.add_subplot(gs[1, 0])
    sns.heatmap(text_matrix, annot=True, cmap="YlGnBu", ax=ax_mat1, vmin=0, vmax=1)
    ax_mat1.set_title("Text Similarity Matrix")
    
    ax_mat2 = fig.add_subplot(gs[1, 1])
    sns.heatmap(sem_matrix, annot=True, cmap="YlGnBu", ax=ax_mat2, vmin=0, vmax=1)
    ax_mat2.set_title("Semantic Similarity Matrix")

    ax_mat3 = fig.add_subplot(gs[1, 2])
    sns.heatmap(vis_matrix, annot=True, cmap="YlGnBu", ax=ax_mat3, vmin=0, vmax=1)
    ax_mat3.set_title("Visual Similarity Matrix")

    # 3. Summary Bar Chart (Bottom Row, Full Width)
    ax_bar = fig.add_subplot(gs[2, :])
    x = np.arange(len(iterations))
    width = 0.2
    
    rects1 = ax_bar.bar(x - 1.5*width, text_prev, width, label='Text (vs Prev)')
    rects2 = ax_bar.bar(x - 0.5*width, sem_prev, width, label='Semantic (vs Prev)')
    rects3 = ax_bar.bar(x + 0.5*width, vis_prev, width, label='Visual (vs Prev)')
    rects4 = ax_bar.bar(x + 1.5*width, cross_sim, width, label='Cross-Modal')

    ax_bar.set_ylabel('Similarity')
    ax_bar.set_title('Step-wise Metrics Summary')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f"Iter {i}" for i in iterations])
    ax_bar.legend()
    ax_bar.set_ylim(0, 1.1)

    plt.tight_layout()
    
    output_path = os.path.join(exp_dir, "dashboard.png")
    plt.savefig(output_path)
    plt.close()
    
    print(f"Dashboard saved to {output_path}")
