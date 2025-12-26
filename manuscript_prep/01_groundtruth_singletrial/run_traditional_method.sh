#!/bin/bash
#SBATCH --job-name=trad_single
#SBATCH --output=slurm/traditional_methods_single_%A_%a.out
#SBATCH --error=slurm/traditional_methods_single_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --partition=ou_bcs_normal
#SBATCH --array=0

source ~/miniconda3/etc/profile.d/conda.sh
conda activate jax312

python compute_traditional_methods_single.py \
    --data ./data/sim.pkl \
    --output ./results/traditional_methods_single.pkl \
    --method both \
    --n_permutations 500 \
    --seed 42