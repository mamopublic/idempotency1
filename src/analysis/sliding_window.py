import numpy as np
import matplotlib.pyplot as plt
import os

class SlidingWindowAnalyzer:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def compute_window_metrics(self, embeddings, window_size=10):
        """
        Computes sliding window metrics:
        1. Window Center (mean of embeddings in window)
        2. Window Radius (max distance from center to any point in window)
        3. Center Drift (distance between consecutive window centers)
        
        Args:
            embeddings (np.ndarray): Array of shape (n_steps, embedding_dim)
            window_size (int): Size of the sliding window
            
        Returns:
            dict: {
                "centers": list of centers,
                "radii": list of radii,
                "drifts": list of drift values,
                "indices": list of start indices k
            }
        """
        n_steps = len(embeddings)
        if n_steps < window_size:
            print(f"Warning: Not enough steps ({n_steps}) for window size {window_size}")
            return None

        centers = []
        radii = []
        indices = []

        # 1. Compute Centers and Radii for each window
        # Window k: [k, k+window_size)
        for k in range(n_steps - window_size + 1):
            window_embs = embeddings[k : k + window_size]
            
            # Center: Mean embedding
            center_k = np.mean(window_embs, axis=0)
            centers.append(center_k)
            
            # Radius: Max euclidean distance from center
            # dists = ||E_t - C_k||
            dists = np.linalg.norm(window_embs - center_k, axis=1)
            radius_k = np.max(dists)
            radii.append(radius_k)
            
            indices.append(k)

        centers = np.array(centers)
        
        # 2. Compute Drift (distance between C_k and C_{k-1})
        drifts = []
        for k in range(len(centers)):
            if k == 0:
                drifts.append(0.0) # No drift for first window
            else:
                drift = np.linalg.norm(centers[k] - centers[k-1])
                drifts.append(drift)

        return {
            "centers": centers,
            "radii": radii,
            "drifts": drifts,
            "indices": indices
        }

    def plot_sliding_window_metrics(self, metrics, label_prefix, run_name=None):
        """Generates plots for drift and radius."""
        if not metrics:
            return

        indices = metrics["indices"]
        drifts = metrics["drifts"]
        radii = metrics["radii"]
        
        # Use middle step for x-axis labeling? Or just window index.
        # User prompt said "x-axis = window index or middle step"
        # Let's use Window Index k
        
        # 1. Plot Center Drift
        plt.figure(figsize=(10, 6))
        plt.plot(indices, drifts, marker='o', label='Center Drift')
        plt.title(f"{label_prefix} Center Drift (Window Size = 10)")
        plt.xlabel("Window Start Index")
        plt.ylabel("Drift ||C_k - C_{k-1}||")
        plt.grid(True, alpha=0.3)
        if run_name:
            plt.suptitle(f"Run: {run_name}")
        
        output_path_drift = os.path.join(self.output_dir, f"window_drift_{label_prefix.lower()}.png")
        plt.savefig(output_path_drift)
        plt.close()
        
        # 2. Plot Radius
        plt.figure(figsize=(10, 6))
        plt.plot(indices, radii, marker='s', color='orange', label='Window Radius')
        plt.title(f"{label_prefix} Window Radius (Dispersion)")
        plt.xlabel("Window Start Index")
        plt.ylabel("Radius (Max Dist to Center)")
        plt.grid(True, alpha=0.3)
        if run_name:
            plt.suptitle(f"Run: {run_name}")
            
        output_path_radius = os.path.join(self.output_dir, f"window_radius_{label_prefix.lower()}.png")
        plt.savefig(output_path_radius)
        plt.close()
        
        print(f"Saved sliding window plots to {self.output_dir}")
