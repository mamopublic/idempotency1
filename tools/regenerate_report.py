#!/usr/bin/env python3
"""
Regenerate Batch Report
Usage: python tools/regenerate_report.py <batch_directory>

Example:
    python tools/regenerate_report.py experiments/vision_sweep_20251218_100919/vision_temp_0.1
"""

import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.batch_reporting import generate_batch_report
from src.config_loader import load_config

def main():
    if len(sys.argv) != 2:
        print("Usage: python tools/regenerate_report.py <batch_directory>")
        sys.exit(1)
        
    batch_dir = sys.argv[1]
    if not os.path.exists(batch_dir):
        print(f"Error: Directory not found: {batch_dir}")
        sys.exit(1)
        
    # Try to load config.yaml as a baseline
    try:
        config = load_config("config/config.yaml")
    except Exception as e:
        print(f"Warning: Could not load config.yaml ({e}). Using empty config (report metadata may be missing).")
        config = {"models": {"text_model": "?", "vision_model": "?", "generation_params": {}, "embedding_model": "?"}, "experiment": {"iterations": "?"}}

    print(f"Regenerating report for: {batch_dir}")
    generate_batch_report(batch_dir, config)
    print("Done.")

if __name__ == "__main__":
    main()
