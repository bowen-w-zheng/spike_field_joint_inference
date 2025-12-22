#!/bin/bash
#SBATCH --job-name=spectrogram
#SBATCH --output=slurm/traditional_methods_%A_%a.out
#SBATCH --error=slurm/traditional_methods_%A_%a.err
#SBATCH --time=04:30:00
#SBATCH --mem=32G
#SBATCH --partition=ou_bcs_normal
#SBATCH --array=0
source ~/miniconda3/etc/profile.d/conda.sh
conda activate jax312
python compute_traditional_methods.py --data ./data/sim_with_trials.pkl --output ./results/plv_results.pkl --n_permutations 400 --method plv