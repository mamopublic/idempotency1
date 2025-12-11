import os
import json
import numpy as np
from .analysis import SimilarityEngine

class ExperimentAnalyzer:
    def __init__(self, config):
        self.config = config
        self.similarity = SimilarityEngine(config)

    def analyze(self, exp_dir):
        """Analyzes a completed experiment directory."""
        print(f"Analyzing experiment: {exp_dir}")
        
        traj_path = os.path.join(exp_dir, "trajectory.json")
        if not os.path.exists(traj_path):
            print(f"Error: trajectory.json not found in {exp_dir}")
            return

        with open(traj_path, "r", encoding="utf-8") as f:
            trajectory = json.load(f)

        # Filter out iterations with missing data (e.g. failed rendering)
        valid_steps = [step for step in trajectory if step.get("prompt")]
        
        n_steps = len(valid_steps)
        metrics = {
            "text_matrix": np.zeros((n_steps, n_steps)).tolist(),
            "semantic_matrix": np.zeros((n_steps, n_steps)).tolist(),
            "visual_matrix": np.zeros((n_steps, n_steps)).tolist(),
            "trajectory": valid_steps
        }

        print(f"Computing metrics for {n_steps} steps...")
        
        # 1. Text & Semantic Matrices (Prompt vs Prompt)
        for i in range(n_steps):
            for j in range(n_steps):
                p1 = valid_steps[i]["prompt"]
                p2 = valid_steps[j]["prompt"]
                
                # Optimization: Similarity is symmetric
                if i <= j:
                    text_sim = self.similarity.compute_text_similarity(p1, p2)
                    sem_sim = self.similarity.compute_semantic_similarity(p1, p2)
                    
                    metrics["text_matrix"][i][j] = text_sim
                    metrics["text_matrix"][j][i] = text_sim
                    
                    metrics["semantic_matrix"][i][j] = sem_sim
                    metrics["semantic_matrix"][j][i] = sem_sim

        # 2. Visual Matrix (Image vs Image)
        # Only valid for steps that have images (Iteration 0 usually doesn't have an image)
        for i in range(n_steps):
            for j in range(n_steps):
                img1 = valid_steps[i].get("image_path")
                img2 = valid_steps[j].get("image_path")
                
                if img1 and img2 and os.path.exists(img1) and os.path.exists(img2):
                    if i <= j:
                        vis_sim = self.similarity.compute_visual_similarity(img1, img2)
                        metrics["visual_matrix"][i][j] = vis_sim
                        metrics["visual_matrix"][j][i] = vis_sim
                else:
                    # If any image is missing (e.g. Iteration 0), decide on value. 0.0 or None?
                    # Using 0.0 for now for matrix completeness
                    metrics["visual_matrix"][i][j] = 0.0

        # 3. Sequential & Cross-Modal Metrics (for Graphs)
        series = []
        for i in range(n_steps):
            step_data = {
                "iteration": valid_steps[i]["iteration"],
                "cross_modal_sim": 0.0,
                "text_sim_prev": 0.0,
                "semantic_sim_prev": 0.0,
                "visual_sim_prev": 0.0
            }
            
            # Cross-Modal: Prompt(i) vs Image(i)
            # Use visual_matrix logic or recompute? Recompute is safer as matrix is image vs image
            p_i = valid_steps[i]["prompt"]
            img_i = valid_steps[i].get("image_path")
            if img_i and os.path.exists(img_i):
                cm_sim = self.similarity.compute_cross_modal_similarity(p_i, img_i)
                step_data["cross_modal_sim"] = cm_sim
                step_data["cross_modal_distance"] = 1.0 - cm_sim
            else:
                step_data["cross_modal_distance"] = 1.0 # Max distance if no image
            
            # Sequential: i vs i-1
            if i > 0:
                # Text/Semantic: Prompt(i) vs Prompt(i-1)
                # We can grab from the matrices we just computed!
                step_data["text_sim_prev"] = metrics["text_matrix"][i][i-1]
                step_data["semantic_sim_prev"] = metrics["semantic_matrix"][i][i-1]
                step_data["semantic_distance_prev"] = 1.0 - step_data["semantic_sim_prev"]
                
                # Visual: Image(i) vs Image(i-1)
                step_data["visual_sim_prev"] = metrics["visual_matrix"][i][i-1]
                step_data["visual_distance_prev"] = 1.0 - step_data["visual_sim_prev"]
            else:
                # Iteration 0 has no "prev".
                step_data["semantic_distance_prev"] = 0.0 # Or 1.0? Usually 0 diff from self, but undefined from prev. Using 0 for graph continuity or 1? 
                # Let's use 0.0 for "start point" or just None/skip in graphing.
                step_data["visual_distance_prev"] = 0.0
            
            series.append(step_data)
        
        metrics["series"] = series

        # Save Metrics
        metrics_path = os.path.join(exp_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print(f"Analysis finished. Metrics matrix saved to {metrics_path}")
        return metrics
