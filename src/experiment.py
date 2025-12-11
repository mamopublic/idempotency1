import os
import json
import shutil
from datetime import datetime
from .llm import OpenRouterClient
from .diagram import MermaidGenerator

class ExperimentRunner:
    def __init__(self, config):
        self.config = config
        self.prompts = config.get("system_prompts", {})
        if not self.prompts:
             print("Warning: No system_prompts found in config.")
             
        self.llm = OpenRouterClient(config)
        self.diagram_gen = MermaidGenerator(config)
        
        self.default_output_dir = config["experiment"]["output_dir"]
        self.iterations = config["experiment"]["iterations"]

    def run(self, initial_prompt, experiment_name=None, output_dir=None):
        """Runs the Prompt -> Diagram -> Prompt loop."""
        
        if not experiment_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"exp_{timestamp}"
            
        base_dir = output_dir if output_dir else self.default_output_dir
        exp_dir = os.path.join(base_dir, experiment_name)
        traj_dir = os.path.join(exp_dir, "trajectory")
        os.makedirs(traj_dir, exist_ok=True)
        
        # Save Run Config
        run_config = {
            "initial_prompt": initial_prompt,
            "system_prompts": self.prompts,
            "models": self.config.get("models", {}),
            "experiment_settings": self.config.get("experiment", {}),
            "timestamp": datetime.now().isoformat()
        }
        with open(os.path.join(exp_dir, "run_config.json"), "w", encoding="utf-8") as f:
            json.dump(run_config, f, indent=2)
        
        current_prompt = initial_prompt
        trajectory = []
        
        # Initial state (Iteration 0)
        trajectory.append({
            "iteration": 0,
            "prompt": current_prompt,
            "mermaid_code": None,
            "image_path": None
        })
        
        cost_tracker = {
            "input_tokens": 0,
            "output_tokens": 0,
            "estimated_cost": 0.0
        }
        
        print(f"Starting experiment: {experiment_name}")
        print(f"Initial Prompt: {initial_prompt[:50]}...")

        for i in range(1, self.iterations + 1):
            print(f"\n--- Iteration {i} ---")
            
            # 1. Generate Diagram Code (Text -> Code)
            print("Generating diagram code...")
            mermaid_code, text_usage = self.llm.generate_diagram_code(
                current_prompt, 
                self.prompts["diagram_generation"]
            )
            
            # Update Cost (Text Model)
            self._update_cost(cost_tracker, text_usage, self.config["models"]["text_model"])
            
            # Save Code
            code_path = os.path.join(traj_dir, f"iter_{i}.mmd")
            with open(code_path, "w", encoding="utf-8") as f:
                f.write(mermaid_code)
                
            # 2. Render Diagram (Code -> Image)
            print("Rendering diagram...")
            image_path = os.path.join(traj_dir, f"iter_{i}.png")
            
            render_success = False
            max_retries = 3
            current_code = mermaid_code
            
            for attempt in range(max_retries + 1):
                try:
                    self.diagram_gen.render(current_code, image_path)
                    render_success = True
                    # If we fixed it, save the fixed version
                    if attempt > 0:
                        print(f"  > Fix successful on attempt {attempt}!")
                        # Overwrite the file with the fixed code so trajectory is correct
                        with open(code_path, "w", encoding="utf-8") as f:
                            f.write(current_code)
                        # Also update local var for trajectory record
                        mermaid_code = current_code
                    break
                except Exception as e:
                    print(f"  > Rendering failed (Attempt {attempt}/{max_retries}): {e}")
                    if attempt < max_retries:
                        print("  > Attempting to fix code with LLM...")
                        try:
                            fixed_code, fix_usage = self.llm.fix_diagram_code(current_code, str(e))
                            # Track cost of fixing
                            self._update_cost(cost_tracker, fix_usage, self.config["models"]["text_model"])
                            current_code = fixed_code
                        except Exception as fix_error:
                            print(f"  > Fix generation failed: {fix_error}")
                            break # Fixer failed, abort
                    else:
                        print("  > Max retries reached. Aborting step.")
            
            if not render_success:
                break

            # 3. Extract Prompt (Image -> Text)
            print("Extracting prompt from image...")
            next_prompt, vision_usage = self.llm.extract_prompt_from_image(
                image_path, 
                self.prompts["vision_extraction"]
            )
            
            # Update Cost (Vision Model)
            self._update_cost(cost_tracker, vision_usage, self.config["models"]["vision_model"])
            
            # Record step (include usage for granular analysis if needed)
            trajectory.append({
                "iteration": i,
                "prompt": next_prompt,
                "mermaid_code": mermaid_code,
                "image_path": image_path,
                "usage": {
                    "text_gen": text_usage,
                    "vision_ext": vision_usage
                }
            })
            
            # Update for next loop
            current_prompt = next_prompt
            
        # Save Trajectory
        traj_path = os.path.join(exp_dir, "trajectory.json")
        with open(traj_path, "w", encoding="utf-8") as f:
            json.dump(trajectory, f, indent=2)
            
        # Save Cost Report
        cost_path = os.path.join(exp_dir, "cost_report.json")
        with open(cost_path, "w", encoding="utf-8") as f:
            json.dump(cost_tracker, f, indent=2)
            
        print(f"\nExperiment generation finished. Trajectory saved to {exp_dir}")
        print(f"Estimated Cost: ${cost_tracker['estimated_cost']:.4f}")
        return exp_dir

    def _update_cost(self, tracker, usage, model_name):
        """Updates the cost tracker based on usage and model rates."""
        if not usage:
            return
            
        in_tokens = usage.get("prompt_tokens", 0)
        out_tokens = usage.get("completion_tokens", 0)
        
        tracker["input_tokens"] += in_tokens
        tracker["output_tokens"] += out_tokens
        
        rates = self.config.get("costs", {}).get(model_name)
        if rates:
            input_cost = (in_tokens / 1_000_000) * rates.get("input", 0.0)
            output_cost = (out_tokens / 1_000_000) * rates.get("output", 0.0)
            tracker["estimated_cost"] += (input_cost + output_cost)
