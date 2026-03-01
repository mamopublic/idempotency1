#!/usr/bin/env python3
"""
Run a Vision Temperature Sweep.

Usage:
    python tools/run_sweep.py <config_file>

Examples:
    python tools/run_sweep.py config/config_standard.yaml
    python tools/run_sweep.py config/config_tight_normalize.yaml

The experiment output directory is automatically named from the config filename
and a timestamp, e.g.:
    experiments/vision_sweep_20260228_165641_tight_normalize/
"""

import argparse
import subprocess
import os
import sys
import time
import shutil
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VISION_TEMPERATURES = [0.1, 0.4, 0.7, 1.0]
TEXT_TEMPERATURE = 0.1  # Fixed: deterministic code generation


def derive_label(config_file: str) -> str:
    """Turn config filename into a short human-readable label for the sweep dir."""
    name = os.path.basename(config_file)          # e.g. config_tight_normalize.yaml
    name = os.path.splitext(name)[0]              # e.g. config_tight_normalize
    name = name.replace("config_", "").replace("config", "")  # e.g. tight_normalize
    return name.strip("_") or "default"


def run_sweep(config_file: str):
    label = derive_label(config_file)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = f"experiments/vision_sweep_{timestamp}_{label}"

    print(f"Starting Vision Temperature Sweep.")
    print(f"Config:           {config_file}")
    print(f"Label:            {label}")
    print(f"Output Directory: {base_output_dir}")
    print(f"Vision Temperatures: {VISION_TEMPERATURES}")
    print(f"Text Temperature (Fixed): {TEXT_TEMPERATURE}")
    print("-" * 60)

    os.makedirs(base_output_dir, exist_ok=True)

    # ── Copy the config into the sweep root for provenance ──────────────────
    dest_config = os.path.join(base_output_dir, "config.yaml")
    shutil.copy2(config_file, dest_config)
    print(f"Config saved to: {dest_config}")

    for vision_temp in VISION_TEMPERATURES:
        print(f"\n>>> Running Batch with Vision Temperature: {vision_temp}")

        temp_dir = os.path.join(base_output_dir, f"vision_temp_{vision_temp}")

        cmd = [
            "python", "main.py",
            "--batch", "config/prompts/batch_stage2_robust.txt",
            "--config", config_file,
            "--text-temp", str(TEXT_TEMPERATURE),
            "--vision-temp", str(vision_temp),
            "--output-dir", temp_dir,
        ]

        try:
            start_time = time.time()
            subprocess.run(cmd, check=True)
            duration = time.time() - start_time

            import json
            timing_data = {
                "vision_temperature": vision_temp,
                "text_temperature": TEXT_TEMPERATURE,
                "config_file": config_file,
                "duration_seconds": duration,
                "duration_minutes": round(duration / 60, 2),
                "start_time": datetime.fromtimestamp(start_time).isoformat(),
                "end_time": datetime.now().isoformat(),
            }
            with open(os.path.join(temp_dir, "timing.json"), "w") as f:
                json.dump(timing_data, f, indent=2)

            print(f">>> Completed Vision Temp {vision_temp} in {duration/60:.1f} min.")
        except subprocess.CalledProcessError as e:
            print(f"!!! Error running vision temp {vision_temp}: {e}")
            continue

    print("\n" + "=" * 60)
    print(f"Sweep complete. Results: {base_output_dir}")

    print("\nGenerating completion analysis...")
    try:
        subprocess.run(
            ["python", "tools/analyze_sweep.py", base_output_dir], check=True
        )
    except Exception as e:
        print(f"Warning: Failed to generate analysis: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Run a vision temperature sweep.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "config_file",
        help="Path to the YAML config file (e.g. config/config_tight_normalize.yaml)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config_file):
        print(f"Error: config file not found: {args.config_file}")
        sys.exit(1)

    run_sweep(args.config_file)


if __name__ == "__main__":
    main()
