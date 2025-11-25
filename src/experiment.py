import os
import json
import shutil
from datetime import datetime
from .llm import OpenRouterClient
from .diagram import MermaidGenerator
from .analysis import SimilarityEngine

class ExperimentRunner:
    def __init__(self, config, prompts):
        self.config = config
        self.prompts = prompts
        self.llm = OpenRouterClient(config)
        self.diagram_gen = MermaidGenerator(config)
        self.similarity = SimilarityEngine(config)
        
        self.output_dir = config["experiment"]["output_dir"]
        self.iterations = config["experiment"]["iterations"]

    def run(self, initial_prompt, experiment_name=None):
        """Runs the Prompt -> Diagram -> Prompt loop."""
        
        if not experiment_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"exp_{timestamp}"
            
        exp_dir = os.path.join(self.output_dir, experiment_name)
        traj_dir = os.path.join(exp_dir, "trajectory")
        os.makedirs(traj_dir, exist_ok=True)
        
        current_prompt = initial_prompt
        metrics = []
        
        print(f"Starting experiment: {experiment_name}")
        print(f"Initial Prompt: {initial_prompt[:50]}...")

        for i in range(1, self.iterations + 1):
            print(f"\n--- Iteration {i} ---")
            
            # 1. Generate Diagram Code (Text -> Code)
            print("Generating diagram code...")
            mermaid_code = self.llm.generate_diagram_code(
                current_prompt, 
                self.prompts["diagram_generation"]
            )
            
            # Save Code
            code_path = os.path.join(traj_dir, f"iter_{i}.mmd")
            with open(code_path, "w", encoding="utf-8") as f:
                f.write(mermaid_code)
                
            # 2. Render Diagram (Code -> Image)
            print("Rendering diagram...")
            image_path = os.path.join(traj_dir, f"iter_{i}.png")
            try:
                self.diagram_gen.render(mermaid_code, image_path)
            except Exception as e:
                print(f"Rendering failed: {e}")
                # Break or continue? For now, break as vision needs image
                break

            # 3. Extract Prompt (Image -> Text)
            print("Extracting prompt from image...")
            next_prompt = self.llm.extract_prompt_from_image(
                image_path, 
                self.prompts["vision_extraction"]
            )
            
            # 4. Compute Metrics
            text_sim = self.similarity.compute_text_similarity(current_prompt, next_prompt)
            sem_sim = self.similarity.compute_semantic_similarity(current_prompt, next_prompt)
            cross_sim = self.similarity.compute_cross_modal_similarity(current_prompt, image_path)
            
            print(f"Text Similarity: {text_sim:.4f}")
            print(f"Semantic Similarity: {sem_sim:.4f}")
            print(f"Cross-Modal Similarity: {cross_sim:.4f}")
            
            metrics.append({
                "iteration": i,
                "input_prompt": current_prompt,
                "mermaid_code": mermaid_code,
                "extracted_prompt": next_prompt,
                "text_similarity": text_sim,
                "semantic_similarity": sem_sim,
                "cross_modal_similarity": cross_sim
            })
            
            # Update for next loop
            current_prompt = next_prompt
            
        # Save Metrics
        metrics_path = os.path.join(exp_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
            
        print(f"\nExperiment finished. Results saved to {exp_dir}")
        return metrics
