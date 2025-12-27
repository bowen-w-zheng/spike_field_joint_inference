#!/usr/bin/env python3
"""
Plot Beta Posterior Scatter for Spike-Field Coupling

Generates 2D scatter plots of βR vs βI posterior samples showing:
- Coupled bands (top row): should show posterior away from origin
- Uncoupled bands (bottom row): should show posterior centered at origin

Usage:
    python plot_beta_posterior.py \
        --sim ./data/sim_single_trial.pkl \
        --joint ./results/joint.pkl \
        --output ./figures/ \
        --unit 0
"""

import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Ellipse
from scipy import stats

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.size'] = 10

# Colors
C = {'coupled': '#F18F01', 'uncoupled': '#888888'}

# Parameters
BURNIN = 0.5


def plot_beta_posterior_scatter(beta_R_samples, beta_I_samples, coupled, freq_hz, unit_idx, ax):
    """
    Plot 2D scatter of βR vs βI posterior samples.
    
    Parameters
    ----------
    beta_R_samples : (N,) array of βR samples
    beta_I_samples : (N,) array of βI samples  
    coupled : bool - whether this is a coupled pair
    freq_hz : frequency in Hz
    unit_idx : unit index
    ax : matplotlib axes
    """
    # Posterior mean
    mean_R = np.mean(beta_R_samples)
    mean_I = np.mean(beta_I_samples)
    
    # Plot samples
    ax.scatter(beta_R_samples, beta_I_samples, alpha=0.3, s=10, c='tab:blue')
    
    # Plot posterior mean
    ax.scatter([mean_R], [mean_I], c='red', s=100, marker='x', linewidths=2, 
               label=f'E[β]', zorder=10)
    
    # Add origin
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
    ax.scatter([0], [0], c='black', s=80, marker='o', zorder=9, label='Origin')
    
    # Compute 95% confidence ellipse
    cov = np.cov(beta_R_samples, beta_I_samples)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    
    # Chi-squared value for 95% confidence with 2 DOF
    chi2_val = stats.chi2.ppf(0.95, 2)
    width, height = 2 * np.sqrt(eigenvalues * chi2_val)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    
    ellipse = Ellipse((mean_R, mean_I), width, height, angle=angle,
                      fill=False, color='red', linestyle='-', linewidth=1.5,
                      label='95% CI')
    ax.add_patch(ellipse)
    
    # Title with coupling status
    status = "Coupled" if coupled else "Uncoupled"
    color = C['coupled'] if coupled else C['uncoupled']
    ax.set_title(f'Unit {unit_idx}, {freq_hz:.0f} Hz\n({status})', 
                 fontsize=10, color=color, fontweight='bold')
    ax.set_xlabel('βR')
    ax.set_ylabel('βI')
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend(fontsize=8, loc='upper right')


def main():
    parser = argparse.ArgumentParser(
        description='Plot beta posterior scatter for spike-field coupling'
    )
    parser.add_argument('--sim', type=str, required=True,
                        help='Path to simulation data')
    parser.add_argument('--joint', type=str, required=True,
                        help='Path to joint inference results')
    parser.add_argument('--output', type=str, default='./figures/',
                        help='Output directory')
    parser.add_argument('--unit', type=int, default=None,
                        help='Unit index to plot (default: first unit with couplings)')
    parser.add_argument('--n_coupled', type=int, default=3,
                        help='Number of coupled bands to plot')
    parser.add_argument('--n_uncoupled', type=int, default=3,
                        help='Number of uncoupled bands to plot')
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # =========================================================================
    # Load Data
    # =========================================================================
    print("="*60)
    print("LOADING DATA")
    print("="*60)
    
    # Simulation data
    print(f"Loading simulation from: {args.sim}")
    with open(args.sim, "rb") as f:
        data = pickle.load(f)
    
    freqs_true = np.asarray(data["freqs_hz"], float)
    masks = np.asarray(data["masks"], bool)
    S, J_true = masks.shape
    
    print(f"  Units: {S}, True freqs: {freqs_true} Hz")
    print(f"  Coupling mask shape: {masks.shape}")
    
    # Joint results
    print(f"\nLoading Joint from: {args.joint}")
    with open(args.joint, 'rb') as f:
        joint_res = pickle.load(f)
    
    FREQS = np.asarray(joint_res['freqs'], dtype=float)
    B = len(FREQS)
    print(f"  Frequency grid: {B} bands ({FREQS[0]:.0f}-{FREQS[-1]:.0f} Hz)")
    
    # Map true freqs to dense grid
    idx_map = np.array([np.argmin(np.abs(FREQS - f)) for f in freqs_true])
    
    # Extract beta from trace
    trace = joint_res['trace']['beta']
    n_samp = trace.shape[0]
    burn = int(BURNIN * n_samp)
    post = trace[burn:]
    
    J_inf = (trace.shape[2] - 1) // 2
    
    # Extract βR and βI samples
    bR = post[:, :, 1:1+J_inf]
    bI = post[:, :, 1+J_inf:1+2*J_inf]
    
    print(f"  Posterior samples: {post.shape[0]} (after {burn} burn-in)")
    print(f"  βR shape: {bR.shape}, βI shape: {bI.shape}")
    
    # =========================================================================
    # Select Unit
    # =========================================================================
    if args.unit is not None:
        unit_idx = args.unit
        if unit_idx >= S:
            print(f"  WARNING: Unit {unit_idx} out of range, using unit 0")
            unit_idx = 0
    else:
        # Find first unit with couplings
        unit_idx = 0
        for s in range(S):
            if masks[s, :].sum() > 0:
                unit_idx = s
                break
    
    print(f"\n  Selected unit: {unit_idx}")
    print(f"  Couplings for unit {unit_idx}: {masks[unit_idx, :]}")
    
    # =========================================================================
    # Find Coupled and Uncoupled Frequency Indices
    # =========================================================================
    coupled_freq_indices = []
    uncoupled_freq_indices = []
    
    # From true signal frequencies
    for jt, ft in enumerate(freqs_true):
        j_dense = idx_map[jt]
        if masks[unit_idx, jt]:
            coupled_freq_indices.append((j_dense, ft, True))  # (index, freq, is_signal)
        else:
            uncoupled_freq_indices.append((j_dense, ft, True))
    
    # Add noise bands (frequencies not in freqs_true)
    noise_freqs = [f for f in FREQS if not any(np.abs(f - ft) < 1.5 for ft in freqs_true)]
    for f in noise_freqs:
        j_dense = np.argmin(np.abs(FREQS - f))
        uncoupled_freq_indices.append((j_dense, f, False))
    
    # Select up to n of each
    coupled_freq_indices = coupled_freq_indices[:args.n_coupled]
    uncoupled_freq_indices = uncoupled_freq_indices[:args.n_uncoupled]
    
    n_coupled = len(coupled_freq_indices)
    n_uncoupled = len(uncoupled_freq_indices)
    
    print(f"\n  Plotting {n_coupled} coupled bands: {[f for _, f, _ in coupled_freq_indices]} Hz")
    print(f"  Plotting {n_uncoupled} uncoupled bands: {[f for _, f, _ in uncoupled_freq_indices]} Hz")
    
    # =========================================================================
    # Plot
    # =========================================================================
    print("\n" + "="*60)
    print("GENERATING PLOT")
    print("="*60)
    
    n_cols = max(n_coupled, n_uncoupled, 3)
    fig, axes = plt.subplots(2, n_cols, figsize=(4*n_cols, 8))
    
    # Plot coupled bands (top row)
    for i, (j_dense, freq_hz, is_signal) in enumerate(coupled_freq_indices):
        ax = axes[0, i]
        beta_R_samples = bR[:, unit_idx, j_dense]
        beta_I_samples = bI[:, unit_idx, j_dense]
        plot_beta_posterior_scatter(beta_R_samples, beta_I_samples, 
                                    coupled=True, freq_hz=freq_hz, 
                                    unit_idx=unit_idx, ax=ax)
        
        # Print stats
        mag = np.sqrt(beta_R_samples.mean()**2 + beta_I_samples.mean()**2)
        print(f"  Coupled {freq_hz:.0f} Hz: |E[β]| = {mag:.4f}")
    
    # Hide unused coupled axes
    for i in range(n_coupled, n_cols):
        axes[0, i].axis('off')
    
    # Plot uncoupled bands (bottom row)
    for i, (j_dense, freq_hz, is_signal) in enumerate(uncoupled_freq_indices):
        ax = axes[1, i]
        beta_R_samples = bR[:, unit_idx, j_dense]
        beta_I_samples = bI[:, unit_idx, j_dense]
        plot_beta_posterior_scatter(beta_R_samples, beta_I_samples, 
                                    coupled=False, freq_hz=freq_hz, 
                                    unit_idx=unit_idx, ax=ax)
        
        # Print stats
        mag = np.sqrt(beta_R_samples.mean()**2 + beta_I_samples.mean()**2)
        label = "signal" if is_signal else "noise"
        print(f"  Uncoupled {freq_hz:.0f} Hz ({label}): |E[β]| = {mag:.4f}")
    
    # Hide unused uncoupled axes
    for i in range(n_uncoupled, n_cols):
        axes[1, i].axis('off')
    
    # Row labels
    if n_coupled > 0:
        axes[0, 0].annotate('Coupled', xy=(-0.35, 0.5), xycoords='axes fraction',
                            fontsize=14, fontweight='bold', color=C['coupled'],
                            rotation=90, va='center', ha='center')
    if n_uncoupled > 0:
        axes[1, 0].annotate('Uncoupled', xy=(-0.35, 0.5), xycoords='axes fraction',
                            fontsize=14, fontweight='bold', color=C['uncoupled'],
                            rotation=90, va='center', ha='center')
    
    plt.suptitle(f'β Posterior Samples (Unit {unit_idx})\n'
                 f'Red ellipse = 95% CI, × = posterior mean', 
                 fontsize=12, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, top=0.90)
    
    # Save
    out_path = f'{args.output}/beta_posterior_scatter.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.savefig(out_path.replace('.png', '.pdf'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n  Saved beta_posterior_scatter.png/pdf")
    
    # =========================================================================
    # Summary Statistics
    # =========================================================================
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    # Compute Wald statistics for selected bands
    print(f"\nWald test results for Unit {unit_idx}:")
    print(f"  {'Freq (Hz)':>10} {'Status':>12} {'|E[β]|':>10} {'W':>10} {'p-value':>12}")
    print("-" * 60)
    
    all_indices = coupled_freq_indices + uncoupled_freq_indices
    for j_dense, freq_hz, is_signal in all_indices:
        br = bR[:, unit_idx, j_dense]
        bi = bI[:, unit_idx, j_dense]
        mu = np.array([br.mean(), bi.mean()])
        Sig = np.cov(np.column_stack([br, bi]), rowvar=False) + 1e-10*np.eye(2)
        W = mu @ np.linalg.solve(Sig, mu)
        pval = 1 - stats.chi2.cdf(W, df=2)
        mag = np.sqrt(mu[0]**2 + mu[1]**2)
        
        is_coupled = any(j == j_dense for j, _, _ in coupled_freq_indices)
        status = "Coupled" if is_coupled else "Uncoupled"
        sig_marker = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        
        print(f"  {freq_hz:>10.0f} {status:>12} {mag:>10.4f} {W:>10.2f} {pval:>10.4f} {sig_marker}")
    
    print("\n" + "="*60)
    print(f"Figures saved to: {args.output}")
    print("="*60)


if __name__ == '__main__':
    main()