#!/usr/bin/env python3
"""
Master pipeline script for single-trial spike-field coupling analysis.

This script runs the complete workflow:
1. Simulate single-trial data with known ground truth
2. Compute traditional methods (PLV, SFC) with permutation tests
3. Compute CT-SSMT LFP-only baseline
4. Run joint inference
5. Generate journal-ready comparison figures

Usage:
    # Run full pipeline:
    python run_pipeline.py --output ./output/

    # Run with custom seed:
    python run_pipeline.py --output ./output/ --seed 123

    # Skip simulation (use existing data):
    python run_pipeline.py --output ./output/ --skip_simulation --data ./data/sim.pkl
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path


def run_command(cmd, description):
    """Run a shell command and handle errors."""
    print("\n" + "="*70)
    print(f"STEP: {description}")
    print("="*70)
    print(f"Command: {cmd}")
    print("-"*70)
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=False)
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"\nERROR: {description} failed with return code {result.returncode}")
        return False
    
    print(f"\n✓ Completed in {elapsed:.1f}s")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Run complete single-trial spike-field analysis pipeline'
    )
    parser.add_argument('--output', type=str, default='./output/',
                        help='Output directory for all results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--skip_simulation', action='store_true',
                        help='Skip simulation step (use existing data)')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to existing simulation data (if skipping simulation)')
    parser.add_argument('--skip_traditional', action='store_true',
                        help='Skip traditional methods (PLV/SFC)')
    parser.add_argument('--skip_ctssmt', action='store_true',
                        help='Skip CT-SSMT LFP-only baseline')
    parser.add_argument('--skip_joint', action='store_true',
                        help='Skip joint inference')
    parser.add_argument('--skip_figures', action='store_true',
                        help='Skip figure generation')
    parser.add_argument('--n_permutations', type=int, default=500,
                        help='Number of permutations for PLV/SFC')
    parser.add_argument('--n_warmup', type=int, default=300,
                        help='MCMC warmup iterations')
    parser.add_argument('--n_samples', type=int, default=500,
                        help='MCMC samples')
    parser.add_argument('--duration', type=float, default=10.0,
                        help='Simulation duration (seconds)')
    parser.add_argument('--S', type=int, default=5,
                        help='Number of units')
    
    args = parser.parse_args()
    
    # Get script directory
    script_dir = Path(__file__).parent.resolve()
    
    # Create output directories
    output_dir = Path(args.output).resolve()
    data_dir = output_dir / 'data'
    results_dir = output_dir / 'results'
    figures_dir = output_dir / 'figures'
    
    for d in [output_dir, data_dir, results_dir, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("SINGLE-TRIAL SPIKE-FIELD COUPLING ANALYSIS PIPELINE")
    print("="*70)
    print(f"Output directory: {output_dir}")
    print(f"Seed: {args.seed}")
    print(f"Skip simulation: {args.skip_simulation}")
    print(f"Skip traditional: {args.skip_traditional}")
    print(f"Skip CT-SSMT: {args.skip_ctssmt}")
    print(f"Skip joint: {args.skip_joint}")
    print(f"Skip figures: {args.skip_figures}")
    print("="*70)
    
    start_total = time.time()
    
    # Step 1: Simulation
    if args.skip_simulation:
        if args.data:
            sim_path = Path(args.data).resolve()
        else:
            sim_path = data_dir / 'sim_single_trial.pkl'
        print(f"\nUsing existing data: {sim_path}")
        if not sim_path.exists():
            print(f"ERROR: Data file not found: {sim_path}")
            sys.exit(1)
    else:
        sim_path = data_dir / 'sim_single_trial.pkl'
        cmd = (
            f"python {script_dir / 'simulate_single_trial.py'} "
            f"--output {sim_path} "
            f"--seed {args.seed} "
            f"--duration {args.duration} "
            f"--S {args.S}"
        )
        if not run_command(cmd, "Simulate single-trial data"):
            sys.exit(1)
    
    # Step 2: Traditional methods (PLV/SFC)
    trad_path = results_dir / 'traditional_methods.pkl'
    if args.skip_traditional:
        print("\nSkipping traditional methods (PLV/SFC)")
    else:
        cmd = (
            f"python {script_dir / 'compute_traditional_methods_single.py'} "
            f"--data {sim_path} "
            f"--output {trad_path} "
            f"--method both "
            f"--n_permutations {args.n_permutations} "
            f"--seed {args.seed}"
        )
        if not run_command(cmd, "Compute PLV and SFC with permutation tests"):
            sys.exit(1)
    
    # Step 3: CT-SSMT LFP-only
    ctssmt_path = results_dir / 'ctssmt_lfp_only.pkl'
    if args.skip_ctssmt:
        print("\nSkipping CT-SSMT LFP-only baseline")
    else:
        cmd = (
            f"python {script_dir / 'compute_ctssmt_lfp_only_single.py'} "
            f"--input {sim_path} "
            f"--output {ctssmt_path}"
        )
        if not run_command(cmd, "Compute CT-SSMT LFP-only baseline"):
            sys.exit(1)
    
    # Step 4: Joint inference
    joint_path = results_dir / 'joint_inference.pkl'
    if args.skip_joint:
        print("\nSkipping joint inference")
    else:
        cmd = (
            f"python {script_dir / 'run_joint_inference_single.py'} "
            f"--input {sim_path} "
            f"--output {joint_path} "
            f"--n_warmup {args.n_warmup} "
            f"--n_samples {args.n_samples} "
            f"--seed {args.seed}"
        )
        if not run_command(cmd, "Run joint inference"):
            sys.exit(1)
    
    # Step 5: Generate figures
    if args.skip_figures:
        print("\nSkipping figure generation")
    else:
        # Check that required results exist
        missing = []
        if not trad_path.exists():
            missing.append(str(trad_path))
        if not joint_path.exists():
            missing.append(str(joint_path))
        
        if missing:
            print(f"\nWARNING: Cannot generate figures - missing results:")
            for m in missing:
                print(f"  - {m}")
            print("Run with appropriate steps enabled first.")
        else:
            cmd = (
                f"python {script_dir / 'generate_figures.py'} "
                f"--data {sim_path} "
                f"--plv {trad_path} "
                f"--joint {joint_path} "
                f"--output {figures_dir}"
            )
            if ctssmt_path.exists():
                cmd += f" --ctssmt {ctssmt_path}"
            
            if not run_command(cmd, "Generate comparison figures"):
                sys.exit(1)
    
    # Summary
    elapsed_total = time.time() - start_total
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"Total time: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
    print(f"\nOutputs:")
    print(f"  Data:     {data_dir}")
    print(f"  Results:  {results_dir}")
    print(f"  Figures:  {figures_dir}")
    
    # List generated files
    print("\nGenerated files:")
    for d in [data_dir, results_dir, figures_dir]:
        for f in sorted(d.glob('*')):
            size = f.stat().st_size
            if size > 1024*1024:
                size_str = f"{size/1024/1024:.1f} MB"
            elif size > 1024:
                size_str = f"{size/1024:.1f} KB"
            else:
                size_str = f"{size} B"
            print(f"  {f.relative_to(output_dir)}: {size_str}")
    
    print("\n" + "="*70)
    print("SUCCESS!")
    print("="*70)


if __name__ == '__main__':
    main()
