#!/usr/bin/env python3
"""
Dry Run Script
Executes a small-scale end-to-end test of the entire pipeline.
- Uses `batch_dry_run.txt` (only 2 prompts)
- Uses `config/config_dryrun.yaml` (only 3 iterations)
- Sweeps only 2 vision temperatures [0.1, 1.0] to test variability handling.
- Runs analysis at the end.
"""

import subprocess
import os
import time
from datetime import datetime

# Configuration
VISION_TEMPERATURES = [0.1, 1.0]  # Just min and max
TEXT_TEMPERATURE = 0.1
BATCH_FILE = "config/prompts/batch_dry_run.txt"
CONFIG_FILE = "config/config_dryrun.yaml"
BASE_OUTPUT_DIR = "experiments/dryrun_" + datetime.now().strftime("%Y%m%d_%H%M%S")

def run_dryrun():
    print(f"Starting DRY RUN Verification.")
    print(f"Batch File: {BATCH_FILE}")
    print(f"Config File: {CONFIG_FILE} (Iterations: 3)")
    print(f"Output Directory: {BASE_OUTPUT_DIR}")
    print(f"Vision Temperatures: {VISION_TEMPERATURES}")
    print("-" * 50)
    
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    
    for vision_temp in VISION_TEMPERATURES:
        print(f"\n>>> Running Batch with Vision Temperature: {vision_temp}")
        
        temp_dir = os.path.join(BASE_OUTPUT_DIR, f"vision_temp_{vision_temp}")
        
        cmd = [
            "python", "main.py",
            "--batch", BATCH_FILE,
            "--config", CONFIG_FILE,
            "--text-temp", str(TEXT_TEMPERATURE),
            "--vision-temp", str(vision_temp),
            "--output-dir", temp_dir
        ]
        
        try:
            start_time = time.time()
            subprocess.run(cmd, check=True)
            duration = time.time() - start_time
            print(f">>> Completed Vision Temp {vision_temp} in {duration:.2f} seconds.")
        except subprocess.CalledProcessError as e:
            print(f"!!! Error running vision temp {vision_temp}: {e}")
            
    print("\n" + "=" * 50)
    print(f"Dry Run Complete. All results saved to {BASE_OUTPUT_DIR}")
    
    # Generate completion analysis
    print("\nGenerating completion analysis...")
    try:
        subprocess.run(["python", "tools/analyze_sweep.py", BASE_OUTPUT_DIR], check=True)
    except Exception as e:
        print(f"Warning: Failed to generate analysis: {e}")

    print("\nDone. Verify artifacts manually in:", BASE_OUTPUT_DIR)
    print("Check for: ")
    print("  - embeddings.npz (in each prompt folder)")
    print("  - batch_dashboard_p1_*.png (in each vision_temp folder)")
    print("  - completion_analysis.md (in base folder)")

if __name__ == "__main__":
    run_dryrun()
