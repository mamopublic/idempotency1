import os
import json
from datetime import datetime
from .experiment import ExperimentRunner
from .post_process import ExperimentAnalyzer
from .visualization import plot_dashboard
from .batch_visualization import plot_batch_dashboard
from .batch_reporting import generate_batch_report

class BatchRunner:
    def __init__(self, config):
        self.config = config
        self.runner = ExperimentRunner(config)

    def run_batch(self, batch_file):
        """Runs experiments for each prompt in the batch file."""
        if not os.path.exists(batch_file):
            print(f"Batch file not found: {batch_file}")
            return
            
        with open(batch_file, "r", encoding="utf-8") as f:
            # Assume one prompt per line, or JSON list
            try:
                data = json.load(f)
                if isinstance(data, list):
                    prompts = data
                else:
                    f.seek(0)
                    prompts = [line.strip() for line in f if line.strip()]
            except json.JSONDecodeError:
                f.seek(0)
                prompts = [line.strip() for line in f if line.strip()]

        print(f"Found {len(prompts)} prompts in batch.")
        
        # Create Batch Output Directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_dir = os.path.join(self.config["experiment"]["output_dir"], f"batch_{timestamp}")
        os.makedirs(batch_dir, exist_ok=True)
        print(f"Batch Output Directory: {batch_dir}")
        
        results = []
        for idx, prompt in enumerate(prompts):
            exp_name = f"prompt_{idx+1}"
            print(f"Running batch item {idx+1}/{len(prompts)}: {exp_name}")
            
            # Run Experiment
            exp_path = self.runner.run(prompt, experiment_name=exp_name, output_dir=batch_dir)
            
            # Post-Process & Visualize
            analyzer = ExperimentAnalyzer(self.config)
            analyzer.analyze(exp_path)
            plot_dashboard(exp_path)
            
            results.append(exp_path)
            
        # Generate Batch Dashboard
        try:
            plot_batch_dashboard(batch_dir)
        except Exception as e:
            print(f"Failed to generate batch dashboard: {e}")
            
        # Generate Batch Report
        try:
            generate_batch_report(batch_dir, self.config)
        except Exception as e:
            print(f"Failed to generate batch report: {e}")
            
        return results
