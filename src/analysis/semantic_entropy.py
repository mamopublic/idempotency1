"""
Semantic Entropy Analysis Module

This module implements semantic entropy analysis for prompt→diagram→prompt cycles.

Semantic entropy measures the diversity/uncertainty in the semantic space by:
1. Clustering embeddings using k-means (one model per temperature)
2. Computing soft assignments to clusters via softmax over distances
3. Calculating entropy of the assignment distribution

Key concepts:
- Global clustering: One k-means model per temperature, fitted on all runs
- Soft assignment: Distances to centroids → probability distribution (via softmax)
- Tau parameter: Controls softness of assignments (lower = harder assignments)
- Separate analysis for text and image embeddings
"""

import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from sklearn.cluster import KMeans, MiniBatchKMeans
from collections import defaultdict


# ============================================================================
# Clustering Functions
# ============================================================================

def fit_semantic_clusters(
    all_embs: np.ndarray,
    n_clusters: int = 16,
    random_state: int = 42,
    use_minibatch: bool = True
) -> KMeans:
    """
    Fit k-means clustering on embeddings.
    
    Args:
        all_embs: Array of shape (N, d) containing embeddings
        n_clusters: Number of clusters
        random_state: Random seed for reproducibility
        use_minibatch: Use MiniBatchKMeans for efficiency (recommended for N > 10k)
    
    Returns:
        Fitted KMeans or MiniBatchKMeans model
    """
    # Sanity check: verify embeddings are unit-normalized (SigLIP property)
    norms = np.linalg.norm(all_embs, axis=1)
    mean_norm = np.mean(norms)
    std_norm = np.std(norms)
    
    if not np.allclose(norms, 1.0, atol=1e-3):
        print(f"  WARNING: Embeddings not unit-normalized! Mean norm: {mean_norm:.6f}, std: {std_norm:.6f}")
        print(f"  Expected: ~1.0 for all embeddings (SigLIP should normalize)")
        # Check for problematic embeddings
        bad_indices = np.where(np.abs(norms - 1.0) > 0.1)[0]
        if len(bad_indices) > 0:
            print(f"  Found {len(bad_indices)} embeddings with norm far from 1.0")
            print(f"  Sample bad norms: {norms[bad_indices[:5]]}")
    else:
        print(f"  ✓ Embeddings are unit-normalized (mean norm: {mean_norm:.6f})")
    
    # Check for NaN or Inf
    if np.isnan(all_embs).any():
        raise ValueError("Embeddings contain NaN values!")
    if np.isinf(all_embs).any():
        raise ValueError("Embeddings contain Inf values!")
    
    if use_minibatch and len(all_embs) > 10000:
        print(f"  Using MiniBatchKMeans for {len(all_embs)} samples")
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            batch_size=1024,
            n_init=3
        )
    else:
        print(f"  Using KMeans for {len(all_embs)} samples")
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10
        )
    
    kmeans.fit(all_embs)
    print(f"  Fitted {n_clusters} clusters, inertia: {kmeans.inertia_:.2f}")
    return kmeans


def collect_all_embeddings(
    run_dirs: List[Path],
    embedding_type: str = "text",
    max_samples: Optional[int] = None
) -> np.ndarray:
    """
    Collect all embeddings from multiple run directories.
    
    Args:
        run_dirs: List of paths to run directories
        embedding_type: "text" or "image"
        max_samples: Optional limit on total samples (for efficiency)
    
    Returns:
        Array of shape (N, d) with stacked embeddings
    """
    all_embs = []
    
    for run_dir in run_dirs:
        emb_file = run_dir / "embeddings.npz"
        if not emb_file.exists():
            continue
        
        try:
            data = np.load(emb_file, allow_pickle=True)
            if embedding_type == "text":
                embs = data["text_embeddings"]
            elif embedding_type == "image":
                embs = data["image_embeddings"]
            else:
                raise ValueError(f"Unknown embedding_type: {embedding_type}")
            
            # Handle different array structures
            # embs might be:
            # - 1D array of objects (each object is an embedding array)
            # - 2D array (n_steps, embedding_dim)
            # - 3D array (1, n_steps, embedding_dim)
            
            # Convert to list and filter None values
            if embs.dtype == object:
                # Array of objects - extract each embedding
                valid_embs = [e for e in embs if e is not None and isinstance(e, np.ndarray)]
            else:
                # Numeric array - reshape if needed
                if embs.ndim == 3:
                    # Squeeze out batch dimension
                    embs = embs.reshape(-1, embs.shape[-1])
                valid_embs = [embs[i] for i in range(len(embs))]
            
            if valid_embs:
                # Stack into 2D array
                valid_embs = np.vstack(valid_embs)
                all_embs.append(valid_embs)
                
        except Exception as e:
            print(f"  Warning: Failed to load {emb_file}: {e}")
    
    if not all_embs:
        raise ValueError(f"No {embedding_type} embeddings found in {len(run_dirs)} runs")
    
    # Stack all embeddings
    all_embs = np.vstack(all_embs)
    print(f"  Collected {len(all_embs)} {embedding_type} embeddings, shape: {all_embs.shape}")
    
    # Subsample if requested
    if max_samples and len(all_embs) > max_samples:
        indices = np.random.choice(len(all_embs), max_samples, replace=False)
        all_embs = all_embs[indices]
        print(f"  Subsampled to {max_samples} embeddings")
    
    return all_embs


def save_kmeans_model(model: KMeans, path: Path) -> None:
    """Save k-means model to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"  Saved k-means model to {path}")


def load_kmeans_model(path: Path) -> KMeans:
    """Load k-means model from disk."""
    model = joblib.load(path)
    print(f"  Loaded k-means model from {path}")
    return model


# ============================================================================
# Entropy Computation
# ============================================================================

def distances_to_probabilities(distances: np.ndarray, tau: float = 0.1) -> np.ndarray:
    """
    Convert distances to probability distribution via softmax.
    
    Args:
        distances: Array of shape (n_clusters,) with distances to centroids
        tau: Temperature parameter (lower = harder assignments)
    
    Returns:
        Probability distribution over clusters
    """
    # Negative distances as logits (closer = higher probability)
    logits = -distances / tau
    
    # Softmax with numerical stability
    logits_max = np.max(logits)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / np.sum(exp_logits)
    
    return probs


def semantic_entropy_from_probs(p: np.ndarray, eps: float = 1e-12) -> float:
    """
    Compute Shannon entropy H(p) = -sum p_i log p_i.
    
    Args:
        p: Probability distribution
        eps: Small constant to avoid log(0)
    
    Returns:
        Entropy in nats (natural log)
    """
    # Clip probabilities to avoid log(0)
    p_clipped = np.clip(p, eps, 1.0)
    entropy = -np.sum(p * np.log(p_clipped))
    return entropy


def compute_semantic_entropy_for_trajectory(
    embs: np.ndarray,
    kmeans: KMeans,
    tau: float = 0.1
) -> np.ndarray:
    """
    Compute semantic entropy for each step in a trajectory.
    
    Args:
        embs: Array of shape (num_steps, d) with embeddings (assumed unit-normalized)
        kmeans: Fitted k-means model
        tau: Softmax temperature
    
    Returns:
        Array of shape (num_steps,) with entropy values
    """
    entropies = []
    centers = kmeans.cluster_centers_
    
    for emb in embs:
        # Compute distances to all cluster centers
        distances = np.linalg.norm(centers - emb, axis=1)
        
        # Convert to probabilities
        probs = distances_to_probabilities(distances, tau)
        
        # Compute entropy
        entropy = semantic_entropy_from_probs(probs)
        entropies.append(entropy)
    
    return np.array(entropies)


# ============================================================================
# Per-Run Analysis
# ============================================================================

def compute_semantic_entropy_for_runs(
    run_dirs: List[Path],
    kmeans: KMeans,
    embedding_type: str = "text",
    tau: float = 0.1
) -> Dict[str, Any]:
    """
    Compute semantic entropy for multiple runs.
    
    Args:
        run_dirs: List of run directories
        kmeans: Fitted k-means model
        embedding_type: "text" or "image"
        tau: Softmax temperature
    
    Returns:
        Dict with:
          - "run_ids": List of run names
          - "entropies": List of entropy arrays (one per run)
    """
    run_ids = []
    entropies_list = []
    
    for run_dir in run_dirs:
        emb_file = run_dir / "embeddings.npz"
        if not emb_file.exists():
            continue
        
        try:
            data = np.load(emb_file, allow_pickle=True)
            if embedding_type == "text":
                embs_raw = data["text_embeddings"]
            else:
                embs_raw = data["image_embeddings"]
            
            # Handle object dtype arrays
            if embs_raw.dtype == object:
                # Filter out None values and stack
                valid_embs = [e for e in embs_raw if e is not None and isinstance(e, np.ndarray)]
                if not valid_embs:
                    continue
                embs = np.vstack(valid_embs)
            else:
                # Numeric array - reshape if needed
                if embs_raw.ndim == 3:
                    embs = embs_raw.reshape(-1, embs_raw.shape[-1])
                else:
                    embs = embs_raw
            
            # Compute entropy trajectory
            entropies = compute_semantic_entropy_for_trajectory(embs, kmeans, tau)
            
            # Save to disk
            entropy_file = run_dir / f"semantic_entropy_{embedding_type}.npy"
            np.save(entropy_file, entropies)
            
            run_ids.append(run_dir.name)
            entropies_list.append(entropies)
            
        except Exception as e:
            print(f"  Warning: Failed to process {run_dir}: {e}")
    
    return {
        "run_ids": run_ids,
        "entropies": entropies_list
    }


# ============================================================================
# Aggregation & Plotting
# ============================================================================

def aggregate_entropy_stats(entropies_list: List[np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Aggregate entropy statistics across runs.
    Handles variable-length trajectories by averaging over available data at each step.
    
    Args:
        entropies_list: List of entropy arrays (one per run)
    
    Returns:
        Dict with "mean", "std", "min", "max" arrays over steps
    """
    if not entropies_list:
        return {"mean": np.array([]), "std": np.array([]), "min": np.array([]), "max": np.array([]), "n_runs": 0}
    
    max_len = max(len(e) for e in entropies_list)
    
    means = np.zeros(max_len)
    stds = np.zeros(max_len)
    mins = np.zeros(max_len)
    maxs = np.zeros(max_len)
    
    for t in range(max_len):
        # Collect values at step t for all runs that have at least t+1 steps
        vals = [e[t] for e in entropies_list if len(e) > t]
        
        if vals:
            means[t] = np.mean(vals)
            stds[t] = np.std(vals)
            mins[t] = np.min(vals)
            maxs[t] = np.max(vals)
        else:
            # Should not happen given how max_len is computed, but for safety:
            means[t] = stds[t] = mins[t] = maxs[t] = np.nan
            
    return {
        "mean": means,
        "std": stds,
        "min": mins,
        "max": maxs,
        "n_runs": len(entropies_list)
    }


def plot_semantic_entropy_over_time(
    entropy_stats_by_T: Dict[float, Dict[str, np.ndarray]],
    embedding_type: str,
    out_path: Path,
    n_clusters: int = 16
) -> None:
    """
    Plot semantic entropy evolution over iterations for multiple temperatures.
    
    Args:
        entropy_stats_by_T: Dict mapping temperature → stats dict
        embedding_type: "text" or "image"
        out_path: Output file path
        n_clusters: Number of clusters (for reference line)
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Sort temperatures for consistent colors
    temps = sorted(entropy_stats_by_T.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(temps)))
    
    for temp, color in zip(temps, colors):
        stats = entropy_stats_by_T[temp]
        if len(stats["mean"]) == 0:
            continue
        
        steps = np.arange(len(stats["mean"]))
        
        # Plot mean with shaded std
        ax.plot(steps, stats["mean"], color=color, linewidth=2, label=f"T={temp}")
        ax.fill_between(
            steps,
            stats["mean"] - stats["std"],
            stats["mean"] + stats["std"],
            color=color,
            alpha=0.2
        )
    
    # Reference line: maximum entropy (uniform distribution)
    max_entropy = np.log(n_clusters)
    ax.axhline(max_entropy, color='gray', linestyle='--', linewidth=1, alpha=0.5, label=f"Max entropy (log {n_clusters})")
    
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Semantic Entropy (nats)", fontsize=12)
    ax.set_title(f"Semantic Entropy Evolution - {embedding_type.capitalize()} Embeddings", fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved plot to {out_path}")


def plot_entropy_vs_temperature(
    entropy_stats_by_T: Dict[float, Dict[str, np.ndarray]],
    embedding_type: str,
    out_path: Path
) -> None:
    """
    Plot final entropy vs temperature.
    
    Args:
        entropy_stats_by_T: Dict mapping temperature → stats dict
        embedding_type: "text" or "image"
        out_path: Output file path
    """
    temps = []
    final_entropies = []
    final_stds = []
    
    for temp in sorted(entropy_stats_by_T.keys()):
        stats = entropy_stats_by_T[temp]
        if len(stats["mean"]) == 0:
            continue
        
        temps.append(temp)
        final_entropies.append(stats["mean"][-1])
        final_stds.append(stats["std"][-1])
    
    if not temps:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.errorbar(temps, final_entropies, yerr=final_stds, marker='o', markersize=8, 
                linewidth=2, capsize=5, capthick=2)
    
    ax.set_xlabel("Temperature", fontsize=12)
    ax.set_ylabel("Final Semantic Entropy (nats)", fontsize=12)
    ax.set_title(f"Final Entropy vs Temperature - {embedding_type.capitalize()} Embeddings", fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved plot to {out_path}")


def plot_individual_trajectories(
    run_entropies: List[np.ndarray],
    run_ids: List[str],
    embedding_type: str,
    temp: float,
    out_path: Path,
    n_clusters: int = 16
) -> None:
    """
    Plot all individual trajectories for a single temperature.
    
    Args:
        run_entropies: List of entropy arrays
        run_ids: List of run names
        embedding_type: "text" or "image"
        temp: Temperature value
        out_path: Output file path
        n_clusters: Number of clusters
    """
    if not run_entropies:
        return
        
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for i, (entropies, run_id) in enumerate(zip(run_entropies, run_ids)):
        steps = np.arange(len(entropies))
        # Use a faint line for each trajectory
        ax.plot(steps, entropies, alpha=0.3, linewidth=1, label=None)
        
    # Plot the mean as a thick line
    max_len = max(len(e) for e in run_entropies)
    means = np.zeros(max_len)
    for t in range(max_len):
        vals = [e[t] for e in run_entropies if len(e) > t]
        means[t] = np.mean(vals)
    
    ax.plot(np.arange(max_len), means, color='black', linewidth=3, label='Mean', linestyle='--')
    
    # Reference line
    max_entropy = np.log(n_clusters)
    ax.axhline(max_entropy, color='red', linestyle=':', linewidth=1, alpha=0.5, label=f"Max entropy (log {n_clusters})")
    
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Semantic Entropy (nats)", fontsize=12)
    ax.set_title(f"Individual Entropy Trajectories - {embedding_type.capitalize()} (T={temp})", fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved individual trajectories plot to {out_path}")


# ============================================================================
# Main Entry Point
# ============================================================================

def analyze_semantic_entropy(
    sweep_dir: Path,
    n_clusters: int = 16,
    tau: float = 0.1,
    embedding_types: List[str] = ["text", "image"],
    force_refit: bool = False
) -> None:
    """
    Run complete semantic entropy analysis on a sweep directory.
    
    Args:
        sweep_dir: Path to sweep directory
        n_clusters: Number of k-means clusters
        tau: Softmax temperature
        embedding_types: List of embedding types to analyze
        force_refit: Force refitting k-means even if model exists
    """
    sweep_dir = Path(sweep_dir)
    models_dir = sweep_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Discover temperature directories
    temp_dirs = sorted([d for d in sweep_dir.iterdir() 
                       if d.is_dir() and d.name.startswith("vision_temp_")])
    
    if not temp_dirs:
        print(f"No temperature directories found in {sweep_dir}")
        return
    
    # Group runs by temperature
    temp_to_runs = {}
    for temp_dir in temp_dirs:
        temp_str = temp_dir.name.replace("vision_temp_", "")
        temp = float(temp_str)
        
        run_dirs = sorted([d for d in temp_dir.iterdir() 
                          if d.is_dir() and d.name.startswith("prompt_")])
        temp_to_runs[temp] = run_dirs
    
    print(f"Found {len(temp_to_runs)} temperatures with {sum(len(r) for r in temp_to_runs.values())} total runs")
    
    # Process each embedding type
    for emb_type in embedding_types:
        print(f"\n{'='*60}")
        print(f"Analyzing {emb_type.upper()} embeddings")
        print(f"{'='*60}")
        
        entropy_stats_by_T = {}
        
        # Process each temperature
        for temp in sorted(temp_to_runs.keys()):
            print(f"\nTemperature: {temp}")
            run_dirs = temp_to_runs[temp]
            
            # K-means model path
            model_path = models_dir / f"kmeans_{emb_type}_T{temp}.pkl"
            
            # Fit or load k-means
            if model_path.exists() and not force_refit:
                kmeans = load_kmeans_model(model_path)
            else:
                print(f"  Fitting k-means with {n_clusters} clusters...")
                all_embs = collect_all_embeddings(run_dirs, emb_type)
                kmeans = fit_semantic_clusters(all_embs, n_clusters)
                save_kmeans_model(kmeans, model_path)
            
            # Compute entropy for all runs
            print(f"  Computing entropy for {len(run_dirs)} runs...")
            results = compute_semantic_entropy_for_runs(run_dirs, kmeans, emb_type, tau)
            
            # Aggregate statistics
            stats = aggregate_entropy_stats(results["entropies"])
            stats["n_clusters"] = n_clusters
            stats["tau"] = tau
            
            # Save stats
            stats_file = temp_to_runs[temp][0].parent / f"semantic_entropy_{emb_type}_stats.json"
            with open(stats_file, "w") as f:
                # Convert numpy arrays to lists for JSON
                stats_json = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                             for k, v in stats.items()}
                json.dump(stats_json, f, indent=2)
            
            entropy_stats_by_T[temp] = stats
            
            # Print summary if we have data
            if len(stats['mean']) > 0:
                print(f"  Mean final entropy: {stats['mean'][-1]:.3f} ± {stats['std'][-1]:.3f}")
            else:
                print(f"  Warning: No entropy data collected for T={temp}")
                
            # Generate individual trajectories plot for this temp
            ind_plot_path = run_dirs[0].parent / f"semantic_entropy_{emb_type}_trajectories_p2.png"
            plot_individual_trajectories(results["entropies"], results["run_ids"], emb_type, temp, ind_plot_path, n_clusters)
        
        # Generate plots
        print(f"\nGenerating plots for {emb_type} embeddings...")
        plot_path = sweep_dir / f"semantic_entropy_{emb_type}_over_time_p2.png"
        plot_semantic_entropy_over_time(entropy_stats_by_T, emb_type, plot_path, n_clusters)
        
        plot_path = sweep_dir / f"semantic_entropy_{emb_type}_vs_temperature_p2.png"
        plot_entropy_vs_temperature(entropy_stats_by_T, emb_type, plot_path)
    
    print(f"\n{'='*60}")
    print("Semantic entropy analysis complete!")
    print(f"{'='*60}")
