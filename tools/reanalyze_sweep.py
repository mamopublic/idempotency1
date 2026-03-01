#!/usr/bin/env python3
"""
Re-analyze Sweep
Usage: python tools/reanalyze_sweep.py <sweep_directory>

Example:
    python tools/reanalyze_sweep.py experiments/vision_sweep_20251214_165641

This script will:
1. Iterate through all experiment folders in the sweep directory.
2. Re-run `ExperimentAnalyzer.analyze()` (re-using embeddings if available).
   - This includes the new "Sliding Ball Window" analysis.
3. Re-run `generate_batch_report` for each temperature batch.
4. Re-run `tools/analyze_sweep.py` to update the top-level completion analysis.
"""

import sys
import os
import subprocess
import glob

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.post_process import ExperimentAnalyzer
from src.batch_reporting import generate_batch_report
from src.analysis.convergence import plot_batch_dashboard
from src.config_loader import load_config

def main():
    if len(sys.argv) != 2:
        print("Usage: python tools/reanalyze_sweep.py <sweep_directory>")
        sys.exit(1)
        
    sweep_dir = sys.argv[1]
    if not os.path.exists(sweep_dir):
        print(f"Error: Directory not found: {sweep_dir}")
        sys.exit(1)

    # Load config: prefer a config.yaml saved alongside the sweep (for provenance),
    # then fall back to the project default config/config.yaml.
    sweep_config = os.path.join(sweep_dir, "config.yaml")
    if os.path.exists(sweep_config):
        config_path = sweep_config
        print(f"Using sweep-local config: {sweep_config}")
    else:
        config_path = "config/config.yaml"
        print(f"No sweep-local config found; using project default: {config_path}")

    try:
        config = load_config(config_path)
        print(f"Loaded config. normalize_mermaid={config['experiment'].get('normalize_mermaid', False)}, "
              f"ball_window_size={config['experiment'].get('ball_window_size', 10)}")
    except Exception as e:
        print(f"Warning: Could not load config: {e}")
        return

    analyzer = ExperimentAnalyzer(config)

    # 1. Iterate through temperature/batch directories
    subdirs = [d for d in os.listdir(sweep_dir) if os.path.isdir(os.path.join(sweep_dir, d))]
    
    # Sort to process orderly
    subdirs.sort()
    
    for batch_name in subdirs:
        batch_dir = os.path.join(sweep_dir, batch_name)
        
        # Check if it looks like a batch folder (contains prompt_* folders)
        prompt_dirs = [d for d in os.listdir(batch_dir) if d.startswith("prompt_") and os.path.isdir(os.path.join(batch_dir, d))]
        
        if not prompt_dirs:
            print(f"Skipping {batch_name} (no prompt_ folders found)")
            continue
            
        print(f"\n=== Re-analyzing Batch: {batch_name} ===")
        
        # Process each experiment
        for prompt_dir in sorted(prompt_dirs):
            exp_dir = os.path.join(batch_dir, prompt_dir)
            try:
                analyzer.analyze(exp_dir)
            except Exception as e:
                print(f"Error analyzing {prompt_dir}: {e}")
                
        # Clean up old dashboard files (without _p1 naming)
        print(f"Cleaning up old dashboard files in {batch_name}...")
        old_patterns = [
            "batch_dashboard_semantic.png",
            "batch_dashboard_visual.png", 
            "batch_dashboard_distances.png",
            "batch_dashboard_trajectory.png"
        ]
        for pattern in old_patterns:
            old_file = os.path.join(batch_dir, pattern)
            if os.path.exists(old_file):
                os.remove(old_file)
                print(f"  Removed: {pattern}")
                
        # Regenerate batch dashboard (metrics aggregation and plotting)
        print(f"Regenerating batch dashboard for {batch_name}...")
        try:
            plot_batch_dashboard(batch_dir)
        except Exception as e:
            print(f"Error generating dashboard for {batch_name}: {e}")
                
        # Regenerate batch report
        print(f"Regenerating batch report for {batch_name}...")
        try:
            generate_batch_report(batch_dir, config)
        except Exception as e:
            print(f"Error generating report for {batch_name}: {e}")

    # 2. Update Sweep Analysis
    print("\n=== Updating Sweep Completion Analysis ===")
    try:
        cmd = ["python3", "tools/analyze_sweep.py", sweep_dir]
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running sweep analysis: {e}")
    except Exception as e:
        print(f"Error: {e}")

    print("\nRe-analysis complete.")

if __name__ == "__main__":
    main()
