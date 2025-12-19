import os
import json
import numpy as np
from src.analysis.similarity import SimilarityEngine

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
        print("  [1/3] Computing semantic embeddings...")
        
        # Storage for embeddings (for reuse in future analyses)
        text_embeddings = []
        image_embeddings = []
        
        # 1. Text & Semantic Matrices (Prompt vs Prompt)
        for i in range(n_steps):
            if i % 5 == 0 and i > 0:
                print(f"    Progress: {i}/{n_steps} prompts processed")
            
            p_i = valid_steps[i]["prompt"]
            # Compute and store text embedding once
            text_emb_i = self.similarity.model.encode_text(p_i)
            text_embeddings.append(text_emb_i.cpu().numpy() if hasattr(text_emb_i, 'cpu') else text_emb_i)
            
            for j in range(n_steps):
                p_j = valid_steps[j]["prompt"]
                
                # Optimization: Similarity is symmetric
                if i <= j:
                    text_sim = self.similarity.compute_text_similarity(p_i, p_j)
                    sem_sim = self.similarity.compute_semantic_similarity(p_i, p_j)
                    
                    metrics["text_matrix"][i][j] = text_sim
                    metrics["text_matrix"][j][i] = text_sim
                    
                    metrics["semantic_matrix"][i][j] = sem_sim
                    metrics["semantic_matrix"][j][i] = sem_sim

        print("  [2/3] Computing visual embeddings...")
        # 2. Visual Matrix (Image vs Image)
        for i in range(n_steps):
            if i % 5 == 0 and i > 0:
                print(f"    Progress: {i}/{n_steps} images processed")
            
            img_i = valid_steps[i].get("image_path")
            # Compute and store image embedding once
            if img_i and os.path.exists(img_i):
                img_emb_i = self.similarity.model.encode_image(img_i)
                image_embeddings.append(img_emb_i.cpu().numpy() if hasattr(img_emb_i, 'cpu') else img_emb_i)
            else:
                image_embeddings.append(None)
            
            for j in range(n_steps):
                img_j = valid_steps[j].get("image_path")
                
                if img_i and img_j and os.path.exists(img_i) and os.path.exists(img_j):
                    if i <= j:
                        vis_sim = self.similarity.compute_visual_similarity(img_i, img_j)
                        metrics["visual_matrix"][i][j] = vis_sim
                        metrics["visual_matrix"][j][i] = vis_sim
                else:
                    metrics["visual_matrix"][i][j] = 0.0

        print("  [3/3] Computing cross-modal similarities...")
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
            p_i = valid_steps[i]["prompt"]
            img_i = valid_steps[i].get("image_path")
            if img_i and os.path.exists(img_i):
                cm_sim = self.similarity.compute_cross_modal_similarity(p_i, img_i)
                step_data["cross_modal_sim"] = cm_sim
                step_data["cross_modal_distance"] = 1.0 - cm_sim
            else:
                step_data["cross_modal_distance"] = 1.0
            
            # Sequential: Compare to previous step
            if i > 0:
                p_prev = valid_steps[i-1]["prompt"]
                img_prev = valid_steps[i-1].get("image_path")
                
                step_data["text_sim_prev"] = self.similarity.compute_text_similarity(p_i, p_prev)
                step_data["semantic_sim_prev"] = self.similarity.compute_semantic_similarity(p_i, p_prev)
                
                # Distances (1 - similarity)
                step_data["semantic_distance_prev"] = 1.0 - step_data["semantic_sim_prev"]
                
                if img_i and img_prev and os.path.exists(img_i) and os.path.exists(img_prev):
                    step_data["visual_sim_prev"] = self.similarity.compute_visual_similarity(img_i, img_prev)
                    step_data["visual_distance_prev"] = 1.0 - step_data["visual_sim_prev"]
                else:
                    step_data["visual_distance_prev"] = 1.0
            
            series.append(step_data)
        
        metrics["series"] = series
        
        # Save metrics
        metrics_path = os.path.join(exp_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        
        # Save embeddings for reuse
        embeddings_path = os.path.join(exp_dir, "embeddings.npz")
        np.savez_compressed(
            embeddings_path,
            text_embeddings=np.array([e for e in text_embeddings if e is not None]),
            image_embeddings=np.array([e for e in image_embeddings if e is not None]),
            iterations=[step["iteration"] for step in valid_steps]
        )
        
        print(f"Analysis finished. Metrics matrix saved to {metrics_path}")
        print(f"Embeddings saved to {embeddings_path}")
        return metrics

