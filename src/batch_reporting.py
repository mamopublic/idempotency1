import os
import json
import subprocess
import shutil
from datetime import datetime
import numpy as np

def generate_batch_report(batch_dir, config):
    """Generates a Markdown report for the batch run."""
    
    report_path = os.path.join(batch_dir, "batch_report.md")
    
    # --- 1. Collect Data ---
    runs = []
    total_cost = 0.0
    
    for item in sorted(os.listdir(batch_dir)):
        item_path = os.path.join(batch_dir, item)
        if os.path.isdir(item_path):
            run_data = {
                "name": item,
                "metrics": None,
                "config": None,
                "cost": 0.0
            }
            
            # Load Metrics
            metrics_path = os.path.join(item_path, "metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, "r", encoding="utf-8") as f:
                    run_data["metrics"] = json.load(f)
            
            # Load Run Config
            run_config_path = os.path.join(item_path, "run_config.json")
            if os.path.exists(run_config_path):
                with open(run_config_path, "r", encoding="utf-8") as f:
                    run_data["config"] = json.load(f)
                    
            # Load Cost
            cost_path = os.path.join(item_path, "cost_report.json")
            if os.path.exists(cost_path):
                with open(cost_path, "r", encoding="utf-8") as f:
                    cost_data = json.load(f)
                    run_data["cost"] = cost_data.get("estimated_cost", 0.0)
                    total_cost += run_data["cost"]
            
            if run_data["metrics"]:
                runs.append(run_data)
                
    if not runs:
        print("No valid runs found for reporting.")
        return

    # --- 2. Build Report Content ---
    
    lines = []
    
    # Title & Header
    lines.append(f"# Batch Experiment Report")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Batch Directory:** `{os.path.basename(batch_dir)}`")
    lines.append("")
    
    # 1. Experimental Configuration
    lines.append("## 1. Experimental Setup")
    lines.append("| Parameter | Value |")
    lines.append("|---|---|")
    lines.append(f"| **Models** | Text: `{config['models']['text_model']}` <br> Vision: `{config['models']['vision_model']}` |")
    lines.append(f"| **Temperatures** | Text: `{config['models']['generation_params']['text_temperature']}` <br> Vision: `{config['models']['generation_params']['vision_temperature']}` |")
    lines.append(f"| **Iterations** | {config['experiment']['iterations']} |")
    lines.append(f"| **Embedding Model** | `{config['models']['embedding_model']}` |")
    lines.append("")

    # 2. Batch-Level Findings
    lines.append("## 2. Batch Analysis")
    
    # Aggregates
    avg_cost = total_cost / len(runs)
    
    # Calculate Mean Convergence & Stability
    sem_conv_steps = []
    vis_conv_steps = []
    sem_stability = []
    vis_stability = []
    
    for r in runs:
        # Re-derive stats or use pre-calculated? 
        # Calculating simple stats from series/matrices
        m = r["metrics"]
        
        # Stability (First-Last)
        if m.get("semantic_matrix"):
            sem_stability.append(m["semantic_matrix"][0][-1])
        if m.get("visual_matrix") and len(m["visual_matrix"]) > 1:
            # Vis matrix 1st image is at index 1 (Iter 1)
            vis_stability.append(m["visual_matrix"][1][-1])

        # Convergence (>0.95 for first time) purely from series
        def find_conv(key):
            for i, step in enumerate(m["series"]):
                if i == 0: continue # Skip first
                if step.get(key, 0) >= 0.95:
                    return i + 1
            return len(m["series"]) + 1 # Didn't converge

        sem_conv_steps.append(find_conv("semantic_sim_prev"))
        vis_conv_steps.append(find_conv("visual_sim_prev"))

    lines.append("### Aggregate Statistics")
    lines.append("| Metric | Mean | Min | Max |")
    lines.append("|---|---|---|---|")
    lines.append(f"| **Semantic Stability** (First-Last) | {np.mean(sem_stability):.3f} | {np.min(sem_stability):.3f} | {np.max(sem_stability):.3f} |")
    lines.append(f"| **Visual Stability** (First-Last) | {np.mean(vis_stability):.3f} | {np.min(vis_stability):.3f} | {np.max(vis_stability):.3f} |")
    lines.append(f"| **Semantic Convergence** (Step) | {np.mean(sem_conv_steps):.1f} | {np.min(sem_conv_steps)} | {np.max(sem_conv_steps)} |")
    lines.append(f"| **Visual Convergence** (Step) | {np.mean(vis_conv_steps):.1f} | {np.min(vis_conv_steps)} | {np.max(vis_conv_steps)} |")
    lines.append(f"| **Total Cost** | ${total_cost:.4f} (Avg: ${avg_cost:.4f}/run) | - | - |")
    lines.append("")
    
    # Dashboards
    lines.append("### Visualizations")
    lines.append("#### Semantic Analysis")
    lines.append("![Semantic Dashboard](batch_dashboard_semantic.png)")
    lines.append("")
    lines.append("#### Visual Analysis")
    lines.append("![Visual Dashboard](batch_dashboard_visual.png)")
    lines.append("")
    lines.append("#### Embedding Distances")
    lines.append("![Distances Dashboard](batch_dashboard_distances.png)")
    lines.append("")

    # 3. Individual Experiment Summary
    lines.append("## 3. Individual Experiments")
    lines.append("| Run | Initial Prompt | Sem Stab. | Vis Stab. | Cost |")
    lines.append("|---|---|---|---|---|")
    
    for i, r in enumerate(runs):
        prompt_snippet = r["config"]["initial_prompt"][:60].replace("\n", " ") + "..." if r["config"] else "N/A"
        s_stab = sem_stability[i] if i < len(sem_stability) else 0.0
        v_stab = vis_stability[i] if i < len(vis_stability) else 0.0
        
        # Link to subfolder? Markdown relative link
        run_link = f"[{r['name']}](./{r['name']}/trajectory.json)"
        
        lines.append(f"| {run_link} | {prompt_snippet} | {s_stab:.3f} | {v_stab:.3f} | ${r['cost']:.4f} |")

    lines.append("")
    
    # 4. Detailed Experiment Log (Full Prompts)
    lines.append("## 4. Detailed Experiment Log")
    for r in runs:
        lines.append(f"### {r['name']}")
        if r["config"]:
            lines.append("**Initial Prompt:**")
            lines.append("> " + r["config"]["initial_prompt"].replace("\n", "\n> "))
            lines.append("")
        lines.append(f"- **Cost:** ${r['cost']:.4f}")
        lines.append(f"- **Data:** [trajectory.json](./{r['name']}/trajectory.json), [metrics.json](./{r['name']}/metrics.json)")
        lines.append("---")
    
    lines.append("")
    
    # Write Report
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        
    print(f"Batch report generated: {report_path}")
    
    # Generate PDF
    print("Generating PDF report...")
    try:
        # Check if npx is available
        if shutil.which("npx"):
            cmd = ["npx", "-y", "md-to-pdf", report_path]
            result = subprocess.run(
                cmd, 
                check=False, 
                shell=False,
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            if result.returncode == 0:
                print(f"PDF report generated: {report_path.replace('.md', '.pdf')}")
            else:
                print(f"PDF generation failed: {result.stderr.decode('utf-8')}")
        else:
            print("npx not found, skipping PDF generation.")
    except Exception as e:
        print(f"Error generating PDF: {e}")
