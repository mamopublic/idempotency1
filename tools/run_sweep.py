import subprocess
import os
import time
from datetime import datetime

# Configuration
VISION_TEMPERATURES = [0.1, 0.4, 0.7, 1.0]  # Sweep vision temp (description creativity)
TEXT_TEMPERATURE = 0.1  # Fixed (deterministic code generation)
BATCH_FILE = "config/prompts/batch_stage2_robust.txt"
CONFIG_FILE = "config/config.yaml"
BASE_OUTPUT_DIR = "experiments/vision_sweep_" + datetime.now().strftime("%Y%m%d_%H%M%S")

def run_sweep():
    print(f"Starting Vision Temperature Sweep.")
    print(f"Batch File: {BATCH_FILE}")
    print(f"Output Directory: {BASE_OUTPUT_DIR}")
    print(f"Vision Temperatures: {VISION_TEMPERATURES}")
    print(f"Text Temperature (Fixed): {TEXT_TEMPERATURE}")
    print("-" * 50)
    
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    
    for vision_temp in VISION_TEMPERATURES:
        print(f"\n>>> Running Batch with Vision Temperature: {vision_temp}")
        
        # Define output directory for this temperature
        temp_dir = os.path.join(BASE_OUTPUT_DIR, f"vision_temp_{vision_temp}")
        
        # Construct command
        cmd = [
            "python", "main.py",
            "--batch", BATCH_FILE,
            "--config", CONFIG_FILE,
            "--text-temp", str(TEXT_TEMPERATURE),
            "--vision-temp", str(vision_temp),
            "--output-dir", temp_dir
        ]
        
        # Execute
        try:
            start_time = time.time()
            subprocess.run(cmd, check=True)
            duration = time.time() - start_time
            
            # Save timing data
            timing_data = {
                "vision_temperature": vision_temp,
                "text_temperature": TEXT_TEMPERATURE,
                "duration_seconds": duration,
                "duration_minutes": duration / 60,
                "start_time": datetime.fromtimestamp(start_time).isoformat(),
                "end_time": datetime.now().isoformat()
            }
            timing_file = os.path.join(temp_dir, "timing.json")
            with open(timing_file, "w") as f:
                import json
                json.dump(timing_data, f, indent=2)
            
            print(f">>> Completed Vision Temp {vision_temp} in {duration:.2f} seconds ({duration/60:.1f} minutes).")
        except subprocess.CalledProcessError as e:
            print(f"!!! Error running vision temp {vision_temp}: {e}")
            # Continue to next temperature? Or stop? 
            # For a sweep, usually better to try continuing.
            continue
            
    print("\n" + "=" * 50)
    print(f"Vision Temperature Sweep Complete. All results saved to {BASE_OUTPUT_DIR}")
    
    # Generate completion analysis
    print("\nGenerating completion analysis...")
    try:
        subprocess.run(["python", "tools/analyze_sweep.py", BASE_OUTPUT_DIR], check=True)
    except Exception as e:
        print(f"Warning: Failed to generate analysis: {e}")

if __name__ == "__main__":
    run_sweep()
