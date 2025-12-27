#!/usr/bin/env python3
"""
Semantic Entropy Analysis Tool

Analyzes semantic entropy evolution in prompt→diagram→prompt cycles.

Usage:
    python tools/analyze_semantic_entropy.py <sweep_directory> [options]

Example:
    python tools/analyze_semantic_entropy.py experiments/vision_sweep_20251218_100919 \\
        --n-clusters 16 \\
        --tau 0.1 \\
        --embedding-types text image

This will:
1. Fit k-means models for each temperature (or load existing)
2. Compute semantic entropy for each run
3. Generate plots showing entropy evolution and temperature dependence
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.semantic_entropy import analyze_semantic_entropy


def main():
    parser = argparse.ArgumentParser(
        description="Analyze semantic entropy in prompt→diagram→prompt cycles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze both text and image embeddings
  python tools/analyze_semantic_entropy.py experiments/vision_sweep_20251218_100919

  # Analyze only text embeddings with custom parameters
  python tools/analyze_semantic_entropy.py experiments/vision_sweep_20251218_100919 \\
      --n-clusters 24 --tau 0.05 --embedding-types text

  # Force refitting k-means models
  python tools/analyze_semantic_entropy.py experiments/vision_sweep_20251218_100919 \\
      --force-refit
        """
    )
    
    parser.add_argument(
        "sweep_dir",
        type=str,
        help="Path to sweep directory (e.g., experiments/vision_sweep_20251218_100919)"
    )
    
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=16,
        help="Number of k-means clusters (default: 16)"
    )
    
    parser.add_argument(
        "--tau",
        type=float,
        default=0.1,
        help="Softmax temperature for soft assignment (default: 0.1, lower = harder)"
    )
    
    parser.add_argument(
        "--embedding-types",
        nargs="+",
        choices=["text", "image"],
        default=["text", "image"],
        help="Embedding types to analyze (default: text image)"
    )
    
    parser.add_argument(
        "--force-refit",
        action="store_true",
        help="Force refitting k-means even if models exist"
    )
    
    args = parser.parse_args()
    
    # Validate sweep directory
    sweep_path = Path(args.sweep_dir)
    if not sweep_path.exists():
        print(f"Error: Sweep directory not found: {sweep_path}")
        sys.exit(1)
    
    if not sweep_path.is_dir():
        print(f"Error: Not a directory: {sweep_path}")
        sys.exit(1)
    
    # Run analysis
    print(f"Semantic Entropy Analysis")
    print(f"=" * 60)
    print(f"Sweep directory: {sweep_path}")
    print(f"N clusters: {args.n_clusters}")
    print(f"Tau (softmax temp): {args.tau}")
    print(f"Embedding types: {', '.join(args.embedding_types)}")
    print(f"Force refit: {args.force_refit}")
    print(f"=" * 60)
    
    try:
        analyze_semantic_entropy(
            sweep_dir=sweep_path,
            n_clusters=args.n_clusters,
            tau=args.tau,
            embedding_types=args.embedding_types,
            force_refit=args.force_refit
        )
        print("\n✓ Analysis complete!")
        
    except Exception as e:
        print(f"\n✗ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
