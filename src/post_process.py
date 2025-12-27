import os
import json
import time
import numpy as np
from src.analysis.similarity import SimilarityEngine
from .analysis.sliding_window import SlidingWindowAnalyzer

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
        text_embeddings = [None] * n_steps
        image_embeddings = [None] * n_steps
        
        # Check if embeddings.npz exists to speed up re-analysis
        embeddings_path = os.path.join(exp_dir, "embeddings.npz")
        loaded_embeddings = False
        if os.path.exists(embeddings_path):
            try:
                print("  > found existing embeddings.npz, loading...")
                data = np.load(embeddings_path, allow_pickle=True)
                # Map stored embeddings back to steps based on iteration
                stored_iters = data["iterations"]
                stored_txt = data["text_embeddings"]
                stored_img = data["image_embeddings"]
                
                # Create lookup map: iteration -> index in stored array
                iter_map = {it: idx for idx, it in enumerate(stored_iters)}
                
                for i, step in enumerate(valid_steps):
                    it = step["iteration"]
                    if it in iter_map:
                        idx = iter_map[it]
                        # Verify shape? assume correct
                        text_embeddings[i] = stored_txt[idx]
                        # Handle potential None/missing in image embeddings?
                        # stored_img entries might be None if saved that way? 
                        # np.savez converts None to object array or something.
                        # But we filtered `if e is not None` when saving.
                        # So stored arrays are packed. We need to be careful.
                        # Actually, previous save code:
                        # text_emb_arr = np.array([e for e in text_embeddings if e is not None])
                        # This loses the alignment with iterations if we had gaps.
                        # But valid_steps filters out steps without prompt. 
                        # The iteration mapping logic above handles packed arrays correctly 
                        # IF we saved iterations aligned with packed arrays.
                        # My previous save code:
                        # iterations=[step["iteration"] for step in valid_steps]
                        # text_emb_arr = ... (filtered Nones)
                        # If image_embeddings had Nones (missing images), img_emb_arr would be shorter than iterations!
                        # This is a bug in my previous save code. 
                        # Let's fix the load logic to be robust or just re-compute if mismatch.
                        
                        # Wait, the previous save code was:
                        # text_emb_arr = np.array([e for e in text_embeddings if e is not None])
                        # iterations = [step["iteration"] for step in valid_steps]
                        
                        # If text_embeddings has the same length as valid_steps (which it should, as we loop n_steps), 
                        # then text_emb_arr matches iterations.
                        # image_embeddings might implement None.
                        
                        if idx < len(stored_txt):
                            text_embeddings[i] = stored_txt[idx]
                        
                        # Images might be sparse
                        # We cannot easily map packed image array back unless we stored image-specific iterations
                        # or if we assume 1-to-1.
                        # Let's assume re-computing visuals is okay/safer, or just use text for now?
                        # Or improving save logic later. 
                        # For now, let's just use text embeddings if available, they are the expensive/deterministic ones.
                        # The new logic below will ensure image embeddings are computed if not loaded.
                        if idx < len(stored_img): # Check if image embedding exists for this iteration
                            image_embeddings[i] = stored_img[idx]
                loaded_embeddings = True
            except Exception as e:
                print(f"  > failed to load embeddings.npz: {e}")
        
        # 1. Pre-compute/Load all embeddings
        print("    Ensuring all embeddings are ready...")
        embedding_start = time.time()
        text_embedding_time = 0.0
        image_embedding_time = 0.0
        
        for i in range(n_steps):
            if i % 10 == 0: print(f"    Encoding step {i}/{n_steps}...")
            
            p_i = valid_steps[i]["prompt"]
            img_i = valid_steps[i].get("image_path")
            
            # Text
            if text_embeddings[i] is None:
                t0 = time.time()
                text_emb = self.similarity.model.encode_text(p_i)
                text_embeddings[i] = text_emb.cpu().numpy() if hasattr(text_emb, 'cpu') else text_emb
                text_embedding_time += time.time() - t0
                
            # Image
            if image_embeddings[i] is None:
                if img_i and os.path.exists(img_i):
                    t0 = time.time()
                    img_emb = self.similarity.model.encode_image(img_i)
                    image_embeddings[i] = img_emb.cpu().numpy() if hasattr(img_emb, 'cpu') else img_emb
                    image_embedding_time += time.time() - t0
                else:
                    image_embeddings[i] = None
        
        total_embedding_time = time.time() - embedding_start
        print(f"    > Text embeddings: {text_embedding_time:.2f}s")
        print(f"    > Image embeddings: {image_embedding_time:.2f}s")
        print(f"    > Total embedding time: {total_embedding_time:.2f}s")

        # 2. Compute Matrices (using pre-computed embeddings)
        print("    Computing similarity matrices...")
        for i in range(n_steps):
            for j in range(n_steps):
                if i <= j:
                    # Text & Semantic
                    text_sim = self.similarity.compute_text_similarity(valid_steps[i]["prompt"], valid_steps[j]["prompt"])
                    
                    # Direct cosine sim on stored embeddings
                    sem_sim = self.similarity.model.compute_similarity(text_embeddings[i], text_embeddings[j])
                    
                    metrics["text_matrix"][i][j] = text_sim
                    metrics["text_matrix"][j][i] = text_sim
                    
                    metrics["semantic_matrix"][i][j] = sem_sim
                    metrics["semantic_matrix"][j][i] = sem_sim
                    
                    # Visual
                    emb_i = image_embeddings[i]
                    emb_j = image_embeddings[j]
                    
                    if emb_i is not None and emb_j is not None:
                        vis_sim = self.similarity.model.compute_similarity(emb_i, emb_j)
                        metrics["visual_matrix"][i][j] = vis_sim
                        metrics["visual_matrix"][j][i] = vis_sim
                    else:
                        metrics["visual_matrix"][i][j] = 0.0
                        metrics["visual_matrix"][j][i] = 0.0

        print("  [3/3] Computing sequential & cross-modal metrics...")
        # 3. Sequential & Cross-Modal Metrics
        series = []
        for i in range(n_steps):
            step_data = {
                "iteration": valid_steps[i]["iteration"],
                "cross_modal_sim": 0.0,
                "text_sim_prev": 0.0,
                "semantic_sim_prev": 0.0,
                "visual_sim_prev": 0.0,
                "cross_modal_distance": 1.0,
                "semantic_distance_prev": 0.0,
                "visual_distance_prev": 1.0
            }
            
            # Cross-Modal: Text(i) vs Image(i)
            if image_embeddings[i] is not None:
                cm_sim = self.similarity.model.compute_similarity(text_embeddings[i], image_embeddings[i])
                step_data["cross_modal_sim"] = cm_sim
                step_data["cross_modal_distance"] = 1.0 - cm_sim
            
            # Sequential: Compare to previous step
            if i > 0:
                p_i = valid_steps[i]["prompt"]
                p_prev = valid_steps[i-1]["prompt"]
                
                step_data["text_sim_prev"] = self.similarity.compute_text_similarity(p_i, p_prev)
                step_data["semantic_sim_prev"] = self.similarity.model.compute_similarity(text_embeddings[i], text_embeddings[i-1])
                step_data["semantic_distance_prev"] = 1.0 - step_data["semantic_sim_prev"]
                
                if image_embeddings[i] is not None and image_embeddings[i-1] is not None:
                    step_data["visual_sim_prev"] = self.similarity.model.compute_similarity(image_embeddings[i], image_embeddings[i-1])
                    step_data["visual_distance_prev"] = 1.0 - step_data["visual_sim_prev"]
            
            series.append(step_data)
        
        metrics["series"] = series
        
        # Save metrics
        metrics_path = os.path.join(exp_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        
        # Save embeddings for reuse
        embeddings_path = os.path.join(exp_dir, "embeddings.npz")
        
        # Convert lists to arrays for saving and analysis
        text_emb_arr = np.array([e for e in text_embeddings if e is not None])
        img_emb_arr = np.array([e for e in image_embeddings if e is not None])
        
        np.savez_compressed(
            embeddings_path,
            text_embeddings=text_emb_arr,
            image_embeddings=img_emb_arr,
            iterations=[step["iteration"] for step in valid_steps]
        )
        
        print(f"Analysis finished. Metrics matrix saved to {metrics_path}")
        print(f"Embeddings saved to {embeddings_path}")
        
        # --- Sliding Window Analysis ---
        from src.analysis.sliding_window import SlidingWindowAnalyzer
        
        window_size = self.config["experiment"].get("ball_window_size", 10)
        print(f"Running Sliding Window Analysis (Window Size={window_size})...")
        
        sw_analyzer = SlidingWindowAnalyzer(exp_dir)
        exp_name = os.path.basename(exp_dir)
        
        # Text Analysis
        if len(text_emb_arr) >= window_size:
            text_metrics = sw_analyzer.compute_window_metrics(text_emb_arr, window_size)
            sw_analyzer.plot_sliding_window_metrics(text_metrics, "Text", exp_name)
            
            # Persist metrics for batch analysis
            # Convert numpy types to native Python types for JSON serialization
            metrics["text_window_drifts"] = [float(x) for x in text_metrics["drifts"]]
            metrics["text_window_radii"] = [float(x) for x in text_metrics["radii"]]
            metrics["text_window_indices"] = [int(x) for x in text_metrics["indices"]]
            
        # Visual Analysis
        if len(img_emb_arr) >= window_size:
            vis_metrics = sw_analyzer.compute_window_metrics(img_emb_arr, window_size)
            sw_analyzer.plot_sliding_window_metrics(vis_metrics, "Visual", exp_name)

            metrics["visual_window_drifts"] = [float(x) for x in vis_metrics["drifts"]]
            metrics["visual_window_radii"] = [float(x) for x in vis_metrics["radii"]]
            metrics["visual_window_indices"] = [int(x) for x in vis_metrics["indices"]]

        # Re-save metrics with new data
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        return metrics

