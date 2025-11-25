import argparse
import os
from src.config_loader import load_config, load_prompts
from src.experiment import ExperimentRunner
from src.batch import BatchRunner

def main():
    parser = argparse.ArgumentParser(description="Idempotency Experiment Runner")
    parser.add_argument("--prompt", type=str, help="Initial prompt for a single experiment")
    parser.add_argument("--batch", type=str, help="Path to a batch file (txt or json) containing prompts")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--prompts", type=str, default="config/prompts.yaml", help="Path to prompts file")
    parser.add_argument("--name", type=str, help="Name for the experiment (optional)")
    
    args = parser.parse_args()
    
    # Load Config
    try:
        config = load_config(args.config)
        prompts = load_prompts(args.prompts)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return

    # Run
    if args.batch:
        runner = BatchRunner(config, prompts)
        runner.run_batch(args.batch)
    elif args.prompt:
        runner = ExperimentRunner(config, prompts)
        runner.run(args.prompt, experiment_name=args.name)
    else:
        print("Error: Must provide either --prompt or --batch")
        parser.print_help()

if __name__ == "__main__":
    main()
