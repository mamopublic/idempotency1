import argparse
import os
from src.config_loader import load_config
from src.experiment import ExperimentRunner
from src.batch import BatchRunner
from src.post_process import ExperimentAnalyzer
from src.visualization import plot_dashboard

def main():
    parser = argparse.ArgumentParser(description="Idempotency Experiment Runner")
    parser.add_argument("--prompt", type=str, help="Initial prompt for a single experiment")
    parser.add_argument("--batch", type=str, help="Path to a batch file (txt or json) containing prompts")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--prompts", type=str, default="config/prompts.yaml", help="Path to prompts file")
    parser.add_argument("--name", type=str, help="Name for the experiment (optional)")
    parser.add_argument("--text-temp", type=float, help="Override text model temperature")
    parser.add_argument("--vision-temp", type=float, help="Override vision model temperature")
    parser.add_argument("--output-dir", type=str, help="Override output directory for batch run")
    
    args = parser.parse_args()
    
    # Load Config
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return

    # Apply Overrides
    if args.text_temp is not None:
        if 'models' not in config: config['models'] = {}
        if 'generation_params' not in config['models']: config['models']['generation_params'] = {}
        config['models']['generation_params']['text_temperature'] = args.text_temp
        
    if args.vision_temp is not None:
        if 'models' not in config: config['models'] = {}
        if 'generation_params' not in config['models']: config['models']['generation_params'] = {}
        config['models']['generation_params']['vision_temperature'] = args.vision_temp

    # Run
    if args.batch:
        runner = BatchRunner(config)
        runner.run_batch(args.batch, output_dir=args.output_dir)
    elif args.prompt:
        runner = ExperimentRunner(config)
        exp_dir = runner.run(args.prompt, experiment_name=args.name, output_dir=args.output_dir)
        
        # Post-Processing Analysis
        analyzer = ExperimentAnalyzer(config)
        analyzer.analyze(exp_dir)
        
        # Visualization
        plot_dashboard(exp_dir)
    else:
        print("Error: Must provide either --prompt or --batch")
        parser.print_help()

if __name__ == "__main__":
    main()
