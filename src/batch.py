import os
import json
from concurrent.futures import ThreadPoolExecutor
from .experiment import ExperimentRunner

class BatchRunner:
    def __init__(self, config, prompts):
        self.config = config
        self.prompts = prompts
        self.runner = ExperimentRunner(config, prompts)

    def run_batch(self, batch_file):
        """Runs experiments for each prompt in the batch file."""
        if not os.path.exists(batch_file):
            raise FileNotFoundError(f"Batch file not found: {batch_file}")
            
        with open(batch_file, "r", encoding="utf-8") as f:
            # Assume one prompt per line, or JSON list
            try:
                data = json.load(f)
                if isinstance(data, list):
                    prompts = data
                else:
                    prompts = [line.strip() for line in f if line.strip()]
            except json.JSONDecodeError:
                f.seek(0)
                prompts = [line.strip() for line in f if line.strip()]

        print(f"Found {len(prompts)} prompts in batch.")
        
        # Sequential for now to avoid Rate Limits, can be parallelized later
        results = []
        for idx, prompt in enumerate(prompts):
            exp_name = f"batch_exp_{idx+1}"
            print(f"Running batch item {idx+1}/{len(prompts)}: {exp_name}")
            metrics = self.runner.run(prompt, experiment_name=exp_name)
            results.append({
                "experiment": exp_name,
                "final_semantic_similarity": metrics[-1]["semantic_similarity"] if metrics else 0.0
            })
            
        return results
