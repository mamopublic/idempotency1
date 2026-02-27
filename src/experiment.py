import os
import json
import shutil
import time
from datetime import datetime
from .llm import OpenRouterClient
from .diagram import MermaidGenerator
from .mermaid_normalizer import normalize_mermaid_code

class ExperimentRunner:
    def __init__(self, config):
        self.config = config
        
        # Select prompt set based on use_tight_prompts flag
        use_tight = config.get("prompts", {}).get("use_tight_prompts", False)
        if use_tight and "tight_system_prompts" in config:
            self.prompts = config["tight_system_prompts"]
            print("Using tight system prompts.")
        else:
            self.prompts = config.get("system_prompts", {})
        
        if not self.prompts:
             print("Warning: No system_prompts found in config.")
             
        self.llm = OpenRouterClient(config)
        self.diagram_gen = MermaidGenerator(config)
        
        self.default_output_dir = config["experiment"]["output_dir"]
        self.iterations = config["experiment"]["iterations"]
        self.normalize_mermaid = config["experiment"].get("normalize_mermaid", False)

    def run(self, initial_prompt, experiment_name=None, output_dir=None, config_path=None):
        """Runs the Prompt -> Diagram -> Prompt loop.
        
        Args:
            initial_prompt: The starting concept/description
            experiment_name: Optional name for the experiment directory
            output_dir: Optional override for the base output directory
            config_path: Optional path to the config file used, for snapshotting
        """
        
        if not experiment_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"exp_{timestamp}"
            
        base_dir = output_dir if output_dir else self.default_output_dir
        exp_dir = os.path.join(base_dir, experiment_name)
        traj_dir = os.path.join(exp_dir, "trajectory")
        os.makedirs(traj_dir, exist_ok=True)
        
        # --- Snapshot the config file used for this run ---
        if config_path and os.path.isfile(config_path):
            shutil.copy(config_path, os.path.join(exp_dir, "config_snapshot.yaml"))
        
        # Save Run Config (JSON summary, kept for backwards compatibility)
        use_tight = self.config.get("prompts", {}).get("use_tight_prompts", False)
        run_config = {
            "initial_prompt": initial_prompt,
            "system_prompts": self.prompts,
            "prompts_variant": "tight" if use_tight else "standard",
            "normalize_mermaid": self.normalize_mermaid,
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
        
        # Initialize cost tracker
        cost_tracker = {
            "total_tokens": 0,
            "total_cost": 0.0,
            "rescue_tokens": 0,
            "rescue_cost": 0.0
        }
        
        # Initialize profiling tracker
        profiling = {
            "text_generation_time": 0.0,
            "rendering_time": 0.0,
            "vision_extraction_time": 0.0,
            "total_iteration_time": 0.0
        }
        
        print(f"Starting experiment: {experiment_name}")
        print(f"Initial Prompt: {initial_prompt[:50]}...")
        print(f"Mermaid normalization: {'ON' if self.normalize_mermaid else 'OFF'}")
        print(f"Prompts: {'tight' if use_tight else 'standard'}")

        for i in range(1, self.iterations + 1):
            print(f"\n--- Iteration {i} ---")
            
            # 1. Generate Diagram Code (Text -> Code)
            print("Generating diagram code...")
            t0 = time.time()
            mermaid_code, text_usage = self.llm.generate_diagram_code(
                current_prompt, 
                self.prompts["diagram_generation"]
            )
            text_gen_time = time.time() - t0
            profiling["text_generation_time"] += text_gen_time
            print(f"  > Text generation took {text_gen_time:.2f}s")
            
            # Update Cost (Text Model)
            self._update_cost(cost_tracker, text_usage, self.config["models"]["text_model"])
            
            # --- Option A: Normalize Mermaid code to remove cosmetic variance ---
            if self.normalize_mermaid:
                mermaid_code_raw = mermaid_code
                mermaid_code = normalize_mermaid_code(mermaid_code)
                if mermaid_code != mermaid_code_raw:
                    print("  > Mermaid code normalized (cosmetic variance removed).")
            
            # Save Code
            code_path = os.path.join(traj_dir, f"iter_{i}.mmd")
            with open(code_path, "w", encoding="utf-8") as f:
                f.write(mermaid_code)
                
            # 2. Render Diagram (Code -> Image)
            print("Rendering diagram...")
            image_path = os.path.join(traj_dir, f"iter_{i}.png")
            
            render_success = False
            max_retries = 5
            current_code = mermaid_code
            rescue_model = self.config["models"].get("rescue_model")
            rescue_retries = self.config["models"].get("rescue_retries", 3)
            rescue_strategy = self.config["models"].get("rescue_strategy", "fix")
            
            # Track rescue metadata for this iteration
            rescue_meta = {
                "rescue_invoked": False,
                "rescue_strategy": None,
                "rescue_attempts": 0
            }
            
            # Phase 1: Try with primary model
            render_start = time.time()
            for attempt in range(max_retries + 1):
                try:
                    self.diagram_gen.render(current_code, image_path)
                    render_success = True
                    render_time = time.time() - render_start
                    profiling["rendering_time"] += render_time
                    print(f"  > Rendering took {render_time:.2f}s")
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
                        print("  > Attempting to fix code with primary LLM...")
                        try:
                            fixed_code, fix_usage = self.llm.fix_diagram_code(current_code, str(e))
                            # Track cost of fixing
                            self._update_cost(cost_tracker, fix_usage, self.config["models"]["text_model"])
                            current_code = fixed_code
                        except Exception as fix_error:
                            print(f"  > Fix generation failed: {fix_error}")
                            break # Fixer failed, abort
                    else:
                        # Phase 2: Escalate to rescue model if available
                        if rescue_model:
                            print(f"  > Primary model exhausted. Escalating to rescue model ({rescue_model}, strategy={rescue_strategy})...")
                            rescue_meta["rescue_invoked"] = True
                            rescue_meta["rescue_strategy"] = rescue_strategy
                            
                            for rescue_attempt in range(rescue_retries):
                                rescue_meta["rescue_attempts"] += 1
                                try:
                                    print(f"  > Rescue attempt {rescue_attempt+1}/{rescue_retries}...")
                                    
                                    if rescue_strategy == "regenerate":
                                        # Regenerate from scratch using the current prompt
                                        print("  > Rescue strategy: regenerate from current prompt.")
                                        fixed_code, fix_usage = self.llm.generate_diagram_code(
                                            current_prompt,
                                            self.prompts["diagram_generation"],
                                            override_model=rescue_model
                                        )
                                    else:
                                        # Default "fix": patch the broken code
                                        fixed_code, fix_usage = self.llm.fix_diagram_code(
                                            current_code, str(e), override_model=rescue_model
                                        )
                                    
                                    # Track rescue cost separately
                                    self._update_cost(cost_tracker, fix_usage, rescue_model, is_rescue=True)
                                    current_code = fixed_code
                                    
                                    # Apply normalization to rescue-generated code too
                                    if self.normalize_mermaid:
                                        current_code = normalize_mermaid_code(current_code)
                                    
                                    # Try rendering with rescue fix
                                    self.diagram_gen.render(current_code, image_path)
                                    render_success = True
                                    print(f"  > Rescue successful on attempt {rescue_attempt+1}!")
                                    with open(code_path, "w", encoding="utf-8") as f:
                                        f.write(current_code)
                                    mermaid_code = current_code
                                    break
                                except Exception as rescue_error:
                                    print(f"  > Rescue attempt {rescue_attempt+1} failed: {rescue_error}")
                                    if rescue_attempt == rescue_retries - 1:
                                        print("  > Rescue model exhausted. Aborting step.")
                        else:
                            print("  > Max retries reached. Aborting step.")
            
            if not render_success:
                break

            # 3. Extract Prompt (Image -> Text)
            print("Extracting prompt from image...")
            t0 = time.time()
            next_prompt, vision_usage = self.llm.extract_prompt_from_image(
                image_path, 
                self.prompts["vision_extraction"]
            )
            vision_time = time.time() - t0
            profiling["vision_extraction_time"] += vision_time
            print(f"  > Vision extraction took {vision_time:.2f}s")
            
            # Update Cost (Vision Model)
            self._update_cost(cost_tracker, vision_usage, self.config["models"]["vision_model"])
            
            # Record step (include usage and rescue metadata)
            trajectory.append({
                "iteration": i,
                "prompt": next_prompt,
                "mermaid_code": mermaid_code,
                "image_path": image_path,
                "usage": {
                    "text_gen": text_usage,
                    "vision_ext": vision_usage
                },
                **rescue_meta  # rescue_invoked, rescue_strategy, rescue_attempts
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
        
        # Save Profiling Report
        profiling_path = os.path.join(exp_dir, "profiling.json")
        with open(profiling_path, "w", encoding="utf-8") as f:
            json.dump(profiling, f, indent=2)
            
        print(f"\nExperiment generation finished. Trajectory saved to {exp_dir}")
        print(f"Estimated Cost: ${cost_tracker['total_cost']:.4f}")
        print(f"\nProfiling Summary:")
        print(f"  Text Generation: {profiling['text_generation_time']:.1f}s")
        print(f"  Rendering:       {profiling['rendering_time']:.1f}s")
        print(f"  Vision Extract:  {profiling['vision_extraction_time']:.1f}s")
        total_time = sum([profiling['text_generation_time'], profiling['rendering_time'], profiling['vision_extraction_time']])
        print(f"  Total:           {total_time:.1f}s ({total_time/60:.1f}min)")
        return exp_dir

    def _update_cost(self, cost_tracker, usage, model_name, is_rescue=False):
        """Updates the cost tracker with token usage.
        Args:
            cost_tracker: The cost tracking dict
            usage: Token usage dict from LLM response
            model_name: Name of the model used
            is_rescue: Whether this was a rescue model call
        """
        if not usage or "total_tokens" not in usage:
            return
            
        # Get cost rates
        costs = self.config.get("costs", {})
        model_costs = costs.get(model_name, {"input": 0, "output": 0})
        
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        
        # Calculate cost (rates are per 1M tokens)
        input_cost = (input_tokens / 1_000_000) * model_costs["input"]
        output_cost = (output_tokens / 1_000_000) * model_costs["output"]
        total_cost = input_cost + output_cost
        
        # Update tracker
        cost_tracker["total_tokens"] += usage.get("total_tokens", 0)
        cost_tracker["total_cost"] += total_cost
        
        # Track rescue costs separately
        if is_rescue:
            cost_tracker["rescue_cost"] += total_cost
            cost_tracker["rescue_tokens"] += usage.get("total_tokens", 0)
