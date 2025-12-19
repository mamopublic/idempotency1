#!/usr/bin/env python3
"""
Backfill embeddings for existing experiments that don't have embeddings.npz

Usage:
    python tools/backfill_embeddings.py <sweep_directory>

Example:
    python tools/backfill_embeddings.py experiments/sweep_20251213_091729
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config_loader import load_config
from src.post_process import ExperimentAnalyzer


def backfill_sweep(sweep_dir):
    """Backfill embeddings for all experiments in a sweep directory"""
    
    if not os.path.exists(sweep_dir):
        print(f"Error: Directory {sweep_dir} does not exist")
        return False
    
    # Load config
    config = load_config("config/config.yaml")
    analyzer = ExperimentAnalyzer(config)
    
    # Find all experiment directories
    experiments = []
    for root, dirs, files in os.walk(sweep_dir):
        if "trajectory.json" in files:
            experiments.append(root)
    
    if not experiments:
        print(f"No experiments found in {sweep_dir}")
        return False
    
    print(f"Found {len(experiments)} experiments")
    
    # Process each experiment
    processed = 0
    skipped = 0
    
    for exp_dir in sorted(experiments):
        embeddings_file = os.path.join(exp_dir, "embeddings.npz")
        
        if os.path.exists(embeddings_file):
            skipped += 1
            continue
        
        rel_path = os.path.relpath(exp_dir, sweep_dir)
        print(f"Processing {rel_path}...")
        
        try:
            analyzer.analyze(exp_dir)
            processed += 1
        except Exception as e:
            print(f"  Error: {e}")
    
    print(f"\nDone!")
    print(f"  Processed: {processed}")
    print(f"  Skipped (already had embeddings): {skipped}")
    print(f"  Total: {len(experiments)}")
    
    return True


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python tools/backfill_embeddings.py <sweep_directory>")
        print("Example: python tools/backfill_embeddings.py experiments/sweep_20251213_091729")
        sys.exit(1)
    
    sweep_dir = sys.argv[1]
    success = backfill_sweep(sweep_dir)
    sys.exit(0 if success else 1)
