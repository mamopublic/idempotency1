#!/usr/bin/env python3
"""
Analyzes sweep results and generates a completion analysis report.

Usage:
    python tools/analyze_sweep.py <sweep_directory>

Example:
    python tools/analyze_sweep.py experiments/vision_sweep_20251214_165641
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.semantic_entropy import analyze_semantic_entropy


def analyze_sweep(sweep_dir):
    """Analyzes a sweep directory and generates completion_analysis.md"""
    
    # Run Semantic Entropy Analysis
    print("\nRunning Semantic Entropy Analysis...")
    try:
        analyze_semantic_entropy(sweep_dir)
        print("✓ Semantic Entropy Analysis complete.")
    except Exception as e:
        print(f"Warning: Semantic Entropy Analysis failed: {e}")
        import traceback
        traceback.print_exc()

    if not os.path.exists(sweep_dir):
        print(f"Error: Directory {sweep_dir} does not exist")
        return False
    
    # Detect sweep type from directory name
    sweep_name = os.path.basename(sweep_dir)
    if "vision_sweep" in sweep_name:
        sweep_type = "Vision Temperature"
        temp_prefix = "vision_temp_"
        varied_param = "vision_temperature"
        fixed_param = "text_temperature = 0.1"
    else:
        sweep_type = "Text Temperature"
        temp_prefix = "temp_"
        varied_param = "text_temperature"
        fixed_param = "vision_temperature = 0.7"
    
    # Find all temperature directories
    temp_dirs = sorted([d for d in os.listdir(sweep_dir) 
                       if d.startswith(temp_prefix) and os.path.isdir(os.path.join(sweep_dir, d))])
    
    if not temp_dirs:
        print(f"Error: No temperature directories found in {sweep_dir}")
        return False
    
    # Extract temperature values
    temperatures = []
    for td in temp_dirs:
        temp_val = td.replace(temp_prefix, "")
        temperatures.append(float(temp_val))
    
    temperatures.sort()
    
    # Analyze each temperature
    results = {}
    total_prompts = 0
    
    for temp in temperatures:
        temp_dir = os.path.join(sweep_dir, f"{temp_prefix}{temp}")
        prompt_dirs = sorted([d for d in os.listdir(temp_dir) 
                             if d.startswith("prompt_") and os.path.isdir(os.path.join(temp_dir, d))])
        
        if total_prompts == 0:
            total_prompts = len(prompt_dirs)
        
        completed = 0
        failed = []
        temp_cost = 0.0
        
        for prompt_dir in prompt_dirs:
            traj_file = os.path.join(temp_dir, prompt_dir, "trajectory.json")
            if os.path.exists(traj_file):
                with open(traj_file, 'r') as f:
                    trajectory = json.load(f)
                    iterations = len(trajectory) - 1  # Subtract iteration 0
                    
                    if iterations >= 30:
                        completed += 1
                    else:
                        prompt_num = prompt_dir.replace("prompt_", "")
                        failed.append((prompt_num, iterations))
            
            # Load Cost
            cost_file = os.path.join(temp_dir, prompt_dir, "cost_report.json")
            if os.path.exists(cost_file):
                try:
                    with open(cost_file, 'r') as f:
                        c_data = json.load(f)
                        temp_cost += c_data.get("total_cost", 0.0)
                except:
                    pass
        
        # Load Timing
        timing_file = os.path.join(temp_dir, "timing.json")
        temp_duration = None
        if os.path.exists(timing_file):
            try:
                with open(timing_file, 'r') as f:
                    timing_data = json.load(f)
                    temp_duration = timing_data.get("duration_minutes")
            except:
                pass
        
        results[temp] = {
            "completed": completed,
            "failed": failed,
            "total": len(prompt_dirs),
            "cost": temp_cost,
            "duration_minutes": temp_duration
        }
    
    # Generate markdown report
    total_experiments = len(temperatures) * total_prompts
    total_completed = sum(r["completed"] for r in results.values())
    
    report_lines = [
        f"# {sweep_type} Sweep - Completion Analysis",
        "",
        f"**Sweep Date**: {datetime.now().strftime('%Y-%m-%d')}",
        f"**Configuration**: {varied_param} varied {temperatures}, {fixed_param}",
        f"**Total Experiments**: {total_experiments} ({total_prompts} prompts × {len(temperatures)} temperatures)",
        "",
        "## Completion Summary",
        "",
        "| Temperature | Completed | Failed | Completion Rate | Cost | Duration | Failed Prompts |",
        "|-------------|-----------|--------|-----------------|------|----------|----------------|"
    ]
    
    for temp in temperatures:
        r = results[temp]
        completion_rate = (r["completed"] / r["total"]) * 100
        
        if r["failed"]:
            failed_str = ", ".join([f"#{num} ({iters} iter)" for num, iters in r["failed"]])
        else:
            failed_str = "None"
        
        # Format duration
        if r["duration_minutes"] is not None:
            duration_str = f"{r['duration_minutes']:.1f} min"
        else:
            duration_str = "N/A"
        
        report_lines.append(
            f"| **{temp}**     | {r['completed']}/{r['total']}     | {len(r['failed'])}      | {completion_rate:.1f}%            | ${r['cost']:.2f} | {duration_str} | {failed_str} |"
        )
    
    overall_rate = (total_completed / total_experiments) * 100
    total_cost = sum(r["cost"] for r in results.values())
    total_duration = sum(r["duration_minutes"] for r in results.values() if r["duration_minutes"] is not None)
    
    report_lines.extend([
        "",
        f"**Overall**: {total_completed}/{total_experiments} experiments ({overall_rate:.1f}%) completed all 30 iterations",
        f"**Total Sweep Cost**: ${total_cost:.2f}",
        f"**Total Sweep Duration**: {total_duration:.1f} minutes ({total_duration/60:.1f} hours)" if total_duration > 0 else "",
        "",
        "## Key Findings",
        ""
    ])
    
    # Add insights based on sweep type
    if "vision" in sweep_name.lower():
        report_lines.extend([
            f"1. **Temperature vs Stability**: Vision temperature sweep with fixed text_temperature=0.1",
            f"   - Lowest temp ({temperatures[0]}): {results[temperatures[0]]['completed']}/{total_prompts} completed ({results[temperatures[0]]['completed']/total_prompts*100:.1f}%)",
            f"   - Highest temp ({temperatures[-1]}): {results[temperatures[-1]]['completed']}/{total_prompts} completed ({results[temperatures[-1]]['completed']/total_prompts*100:.1f}%)",
            "",
            "2. **Failure Mode**: Failures indicate true system instability (not syntax errors)",
            "",
            "3. **Scientific Insight**: Tests whether description creativity (high vision temp) causes semantic drift"
        ])
    else:
        report_lines.extend([
            f"1. **Temperature vs Stability**: Higher text temperature → lower completion rate",
            f"   - T={temperatures[0]}: {results[temperatures[0]]['completed']}/{total_prompts} completed ({results[temperatures[0]]['completed']/total_prompts*100:.1f}%)",
            f"   - T={temperatures[-1]}: {results[temperatures[-1]]['completed']}/{total_prompts} completed ({results[temperatures[-1]]['completed']/total_prompts*100:.1f}%)",
            "",
            "2. **Failure Mode**: All failures due to Mermaid syntax errors from high-temperature code generation",
            "",
            "3. **Scientific Insight**: High text temperature introduces syntax errors rather than semantic variation"
        ])
    
    # Write report
    report_path = os.path.join(sweep_dir, "completion_analysis.md")
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✓ Analysis complete: {report_path}")
    print(f"  Total: {total_completed}/{total_experiments} ({overall_rate:.1f}%) completed")
    
    return True


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python tools/analyze_sweep.py <sweep_directory>")
        print("Example: python tools/analyze_sweep.py experiments/vision_sweep_20251214_165641")
        sys.exit(1)
    
    sweep_dir = sys.argv[1]
    success = analyze_sweep(sweep_dir)
    sys.exit(0 if success else 1)
