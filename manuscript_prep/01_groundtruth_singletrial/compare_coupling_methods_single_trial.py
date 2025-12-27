#!/usr/bin/env python3
"""
Spike-Field Coupling Detection: Method Comparison (Single Trial)

This script matches the trial-structure notebook:
  compare_coupling_methods_complete_mark_signal_only.ipynb

Expected input files (produced by the repo scripts):
- sim:         simulate_single_trial.py              -> data['freqs_hz'], data['masks'], data['beta_mag'], data['beta_phase']
- joint:       run_joint_inference_single.py         -> output['freqs'], output['trace']['beta'], output['coupling'], output['beta_standardized']
- traditional: compute_traditional_methods_single.py -> results['plv']['values/pval/phase'], results['sfc']['values/pval']

Usage:
    python compare_coupling_methods_single_trial.py \
        --sim ./data/sim_single_trial.pkl \
        --joint ./results/joint.pkl \
        --traditional ./results/traditional_methods_single.pkl \
        --output ./figures/
"""

import os
import sys
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patheffects as patheffects
from scipy import stats

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn not installed, ROC/PR curves will be skipped")


# =============================================================================
# FIGURE STYLE
# =============================================================================

def set_style(font_size=9):
    """Set publication-quality figure style."""
    if HAS_SEABORN:
        sns.set(style="ticks", context="paper", font="sans-serif",
                rc={"font.size": font_size, "axes.titlesize": font_size,
                    "axes.labelsize": font_size, "axes.linewidth": 0.5,
                    "lines.linewidth": 1.5, "xtick.labelsize": font_size,
                    "ytick.labelsize": font_size, "legend.fontsize": font_size,
                    "legend.frameon": False})
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42


# Colors
C = {'plv': '#2E86AB', 'sfc': '#A23B72', 'joint': '#F18F01', 'joint_std': '#8B4513'}

# Parameters
ALPHA = 0.05
BURNIN = 0.5


# =============================================================================
# HEATMAP PLOTTING
# =============================================================================

def plot_effect_heatmap_row(ax, values, freqs, title, true_freqs, masks, 
                            log_scale=False, cmap='Reds', vmax_percentile=99):
    """Plot single effect size heatmap row with ★ markers."""
    
    if log_scale:
        plot_values = np.log10(values + 1)
        vmax = np.percentile(plot_values[np.isfinite(plot_values)], vmax_percentile)
    else:
        plot_values = values
        vmax = np.percentile(plot_values[np.isfinite(plot_values)], vmax_percentile)
    
    freq_step = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    extent = [freqs[0] - freq_step/2, freqs[-1] + freq_step/2, 
              values.shape[0] - 0.5, -0.5]
    
    im = ax.imshow(plot_values, aspect='auto', cmap=cmap,
                   vmin=0, vmax=vmax, extent=extent)
    
    ax.set_ylabel('Unit')
    ax.set_title(title, fontweight='bold')
    
    for f in true_freqs:
        ax.axvline(f, color='cyan', linestyle='--', alpha=0.5, lw=1)
    
    for s in range(masks.shape[0]):
        for j, f in enumerate(true_freqs):
            if j < masks.shape[1] and masks[s, j]:
                ax.text(f, s, '★', ha='center', va='center',
                       fontsize=12, color='white', fontweight='bold',
                       path_effects=[patheffects.withStroke(linewidth=2, foreground='black')])
    
    return im


def plot_pval_heatmap_row(ax, pvals, freqs, title, true_freqs, masks, 
                          cmap='hot_r', vmin=0, vmax=10):
    """Plot single p-value heatmap row with ★ markers."""
    
    log_p = -np.log10(np.clip(pvals, 1e-20, 1))
    
    freq_step = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    extent = [freqs[0] - freq_step/2, freqs[-1] + freq_step/2,
              pvals.shape[0] - 0.5, -0.5]
    
    im = ax.imshow(log_p, aspect='auto', cmap=cmap,
                   vmin=vmin, vmax=vmax, extent=extent)
    
    ax.set_ylabel('Unit')
    ax.set_title(title, fontweight='bold')
    
    for f in true_freqs:
        ax.axvline(f, color='cyan', linestyle='--', alpha=0.5, lw=1)
    
    for s in range(masks.shape[0]):
        for j, f in enumerate(true_freqs):
            if j < masks.shape[1] and masks[s, j]:
                ax.text(f, s, '★', ha='center', va='center',
                       fontsize=12, color='white', fontweight='bold',
                       path_effects=[patheffects.withStroke(linewidth=2, foreground='black')])
    
    return im


def metrics(y_true, pval, alpha=ALPHA):
    """Compute detection metrics using ALL bands."""
    y_pred = (pval < alpha).flatten()
    y = y_true.flatten()
    
    TP = (y_pred & y).sum()
    TN = (~y_pred & ~y).sum()
    FP = (y_pred & ~y).sum()
    FN = (~y_pred & y).sum()
    
    sens = TP/(TP+FN) if TP+FN > 0 else 0
    spec = TN/(TN+FP) if TN+FP > 0 else 0
    prec = TP/(TP+FP) if TP+FP > 0 else 0
    f1 = 2*prec*sens/(prec+sens) if prec+sens > 0 else 0
    
    return {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
            'sens': sens, 'spec': spec, 'prec': prec, 'f1': f1}


def circ_diff(a, b):
    """Compute circular difference between angles."""
    return np.arctan2(np.sin(a-b), np.cos(a-b))


def compute_wald_stats(bR, bI, S, J):
    """Compute Wald statistics and p-values from posterior samples."""
    W = np.zeros((S, J))
    pval = np.zeros((S, J))
    
    for s in range(S):
        for j in range(J):
            br, bi = bR[:, s, j], bI[:, s, j]
            mu = np.array([br.mean(), bi.mean()])
            Sig = np.cov(np.column_stack([br, bi]), rowvar=False) + 1e-10*np.eye(2)
            W[s, j] = mu @ np.linalg.solve(Sig, mu)
            pval[s, j] = 1 - stats.chi2.cdf(W[s, j], df=2)
    
    return W, pval


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compare spike-field coupling detection methods (single trial)'
    )
    parser.add_argument('--sim', type=str, required=True,
                        help='Path to simulation data')
    parser.add_argument('--joint', type=str, required=True,
                        help='Path to joint inference results')
    parser.add_argument('--traditional', type=str, default=None,
                        help='Path to PLV+SFC results')
    parser.add_argument('--plv', type=str, default=None,
                        help='Path to PLV results (alternative to --traditional)')
    parser.add_argument('--sfc', type=str, default=None,
                        help='Path to SFC results (alternative to --traditional)')
    parser.add_argument('--output', type=str, default='./figures/',
                        help='Output directory')
    
    args = parser.parse_args()
    
    if args.traditional:
        args.plv = args.traditional
        args.sfc = args.traditional
    elif not (args.plv and args.sfc):
        parser.error("Either --traditional OR both --plv and --sfc are required")
    
    set_style()
    os.makedirs(args.output, exist_ok=True)
    OUT = args.output
    
    # =========================================================================
    # 1. Load Data
    # =========================================================================
    print("="*60)
    print("1. LOADING DATA")
    print("="*60)
    
    # Simulation data
    print(f"Loading simulation from: {args.sim}")
    with open(args.sim, "rb") as f:
        data = pickle.load(f)
    
    freqs_true = np.asarray(data["freqs_hz"], float)
    freqs_coupled = np.asarray(data.get("freqs_hz_coupled", freqs_true), float)
    masks = np.asarray(data["masks"], bool)
    
    if 'beta_mag' in data:
        beta_mag_true = np.asarray(data['beta_mag'])
        beta_phase_true = np.asarray(data['beta_phase'])
    else:
        beta = np.asarray(data["beta_true"])
        J = len(freqs_true)
        bR, bI = beta[:, 1:1+J], beta[:, 1+J:1+2*J]
        beta_mag_true = np.sqrt(bR**2 + bI**2)
        beta_phase_true = np.arctan2(bI, bR)
    
    S, J_true = masks.shape
    print(f"  Units: {S}, True freqs: {freqs_true} Hz")
    print(f"  True couplings: {masks.sum()}, Non-couplings: {(~masks).sum()}")
    
    # PLV results
    print(f"\nLoading PLV from: {args.plv}")
    with open(args.plv, 'rb') as f:
        plv_res = pickle.load(f)
    plv_val = plv_res['plv']['values']
    plv_pval = plv_res['plv']['pval']
    plv_phase = plv_res['plv']['phase']
    print(f"  PLV: {plv_val.shape}")
    
    # SFC results
    print(f"\nLoading SFC from: {args.sfc}")
    with open(args.sfc, 'rb') as f:
        sfc_res = pickle.load(f)
    sfc_val = sfc_res['sfc']['values']
    sfc_pval = sfc_res['sfc']['pval']
    print(f"  SFC: {sfc_val.shape}")
    
    # Joint results
    print(f"\nLoading Joint from: {args.joint}")
    with open(args.joint, 'rb') as f:
        joint_res = pickle.load(f)
    
    FREQS = np.asarray(joint_res['freqs'], dtype=float)
    B = len(FREQS)
    print(f"  Frequency grid: {B} bands ({FREQS[0]:.0f}-{FREQS[-1]:.0f} Hz)")
    
    idx_map = np.array([np.argmin(np.abs(FREQS - f)) for f in freqs_true])
    
    # Extract ORIGINAL beta from trace
    trace = joint_res['trace']['beta']
    n_samp = trace.shape[0]
    burn = int(BURNIN * n_samp)
    post = trace[burn:]
    
    J_inf = (trace.shape[2] - 1) // 2
    
    bR = post[:, :, 1:1+J_inf]
    bI = post[:, :, 1+J_inf:1+2*J_inf]
    mR, mI = bR.mean(0), bI.mean(0)
    joint_mag = np.sqrt(mR**2 + mI**2)
    joint_phase = np.arctan2(mI, mR)
    
    W, joint_pval = compute_wald_stats(bR, bI, S, J_inf)
    
    print(f"  Joint (original) |β|: range [{joint_mag.min():.4f}, {joint_mag.max():.4f}]")
    print(f"  Wald W: [{W.min():.1f}, {W.max():.1f}]")
    
    # Extract STANDARDIZED beta
    beta_standardized = joint_res.get('beta_standardized', None)
    if beta_standardized is not None:
        # beta_standardized is (S, 1+2*J) - point estimate, not trace
        bR_std = beta_standardized[:, 1:1+J_inf]
        bI_std = beta_standardized[:, 1+J_inf:1+2*J_inf]
        joint_mag_std = np.sqrt(bR_std**2 + bI_std**2)
        joint_phase_std = np.arctan2(bI_std, bR_std)
        
        # For Wald on standardized, we need to use the trace variance structure
        # but with standardized means - approximate by scaling
        scale_factors = joint_res.get('latent_scale_factors', None)
        if scale_factors is not None:
            # Standardized Wald: same W statistic structure but with standardized beta
            W_std = np.zeros((S, J_inf))
            joint_pval_std = np.zeros((S, J_inf))
            
            for s in range(S):
                for j in range(J_inf):
                    # Use standardized point estimate
                    mu_std = np.array([bR_std[s, j], bI_std[s, j]])
                    # Scale the covariance by the same factor
                    br, bi = bR[:, s, j], bI[:, s, j]
                    Sig = np.cov(np.column_stack([br, bi]), rowvar=False) + 1e-10*np.eye(2)
                    # Scale covariance to standardized space
                    scale_j = scale_factors[j] * scale_factors[J_inf + j] if j + J_inf < len(scale_factors) else 1.0
                    Sig_std = Sig  # Keep same relative structure
                    W_std[s, j] = mu_std @ np.linalg.solve(Sig_std + 1e-10*np.eye(2), mu_std)
                    joint_pval_std[s, j] = 1 - stats.chi2.cdf(W_std[s, j], df=2)
        else:
            W_std = W  # Fallback
            joint_pval_std = joint_pval
        
        print(f"  Joint (standardized) |β|: range [{joint_mag_std.min():.4f}, {joint_mag_std.max():.4f}]")
        print(f"  Wald W (std): [{W_std.min():.1f}, {W_std.max():.1f}]")
        HAS_STANDARDIZED = True
    else:
        print("  WARNING: beta_standardized not found in results")
        HAS_STANDARDIZED = False
    
    # =========================================================================
    # 2. Create Ground Truth Matrix
    # =========================================================================
    print("\n" + "="*60)
    print("2. GROUND TRUTH")
    print("="*60)
    
    y_true = np.zeros((S, B), dtype=bool)
    for s in range(S):
        for jt, ft in enumerate(freqs_true):
            if masks[s, jt]:
                y_true[s, idx_map[jt]] = True
    
    n_pos = y_true.sum()
    n_neg = (~y_true).sum()
    print(f"Full ground truth: {S}×{B} = {S*B} pairs")
    print(f"  Positive (coupled): {n_pos}")
    print(f"  Negative (not coupled): {n_neg}")
    
    # =========================================================================
    # 3. Effect Size Heatmaps - ORIGINAL BETA
    # =========================================================================
    print("\n" + "="*60)
    print("3. EFFECT SIZE HEATMAPS (Original β)")
    print("="*60)
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    effect_data = [plv_val, sfc_val, joint_mag, W]
    effect_names = ['PLV', 'SFC', 'Joint |E[β]| (original)', 'Joint Wald W (original)']
    log_flags = [False, False, False, True]
    
    for ax, val, name, use_log in zip(axes, effect_data, effect_names, log_flags):
        im = plot_effect_heatmap_row(ax, val, FREQS, name, freqs_true, masks, 
                                      log_scale=use_log, cmap='Reds')
        cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
        if use_log:
            cbar.set_label('log₁₀(W + 1)')
    
    axes[-1].set_xlabel('Frequency (Hz)')
    
    plt.suptitle('Effect Size - Original β (★ = true coupling)', fontsize=12, y=0.995)
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    plt.savefig(f'{OUT}/heatmap_effect_original.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{OUT}/heatmap_effect_original.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved heatmap_effect_original.png/pdf")
    
    # =========================================================================
    # 3b. Effect Size Heatmaps - STANDARDIZED BETA
    # =========================================================================
    if HAS_STANDARDIZED:
        print("\n" + "="*60)
        print("3b. EFFECT SIZE HEATMAPS (Standardized β)")
        print("="*60)
        
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        
        effect_data_std = [plv_val, sfc_val, joint_mag_std, W_std]
        effect_names_std = ['PLV', 'SFC', 'Joint |E[β]| (standardized)', 'Joint Wald W (standardized)']
        
        for ax, val, name, use_log in zip(axes, effect_data_std, effect_names_std, log_flags):
            im = plot_effect_heatmap_row(ax, val, FREQS, name, freqs_true, masks, 
                                          log_scale=use_log, cmap='Reds')
            cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
            if use_log:
                cbar.set_label('log₁₀(W + 1)')
        
        axes[-1].set_xlabel('Frequency (Hz)')
        
        plt.suptitle('Effect Size - Standardized β (★ = true coupling)', fontsize=12, y=0.995)
        plt.tight_layout()
        plt.subplots_adjust(top=0.94)
        plt.savefig(f'{OUT}/heatmap_effect_standardized.png', dpi=150, bbox_inches='tight')
        plt.savefig(f'{OUT}/heatmap_effect_standardized.pdf', dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved heatmap_effect_standardized.png/pdf")
    
    # =========================================================================
    # 3c. Side-by-side comparison of Original vs Standardized
    # =========================================================================
    if HAS_STANDARDIZED:
        print("\n" + "="*60)
        print("3c. SIDE-BY-SIDE: Original vs Standardized β")
        print("="*60)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        
        # Original |β|
        im1 = plot_effect_heatmap_row(axes[0, 0], joint_mag, FREQS, 
                                       'Joint |E[β]| (Original)', freqs_true, masks,
                                       log_scale=False, cmap='Reds')
        fig.colorbar(im1, ax=axes[0, 0], fraction=0.02, pad=0.01)
        
        # Standardized |β|
        im2 = plot_effect_heatmap_row(axes[0, 1], joint_mag_std, FREQS,
                                       'Joint |E[β]| (Standardized)', freqs_true, masks,
                                       log_scale=False, cmap='Reds')
        fig.colorbar(im2, ax=axes[0, 1], fraction=0.02, pad=0.01)
        
        # Original Wald W
        im3 = plot_effect_heatmap_row(axes[1, 0], W, FREQS,
                                       'Wald W (Original)', freqs_true, masks,
                                       log_scale=True, cmap='Reds')
        cbar3 = fig.colorbar(im3, ax=axes[1, 0], fraction=0.02, pad=0.01)
        cbar3.set_label('log₁₀(W + 1)')
        
        # Standardized Wald W
        im4 = plot_effect_heatmap_row(axes[1, 1], W_std, FREQS,
                                       'Wald W (Standardized)', freqs_true, masks,
                                       log_scale=True, cmap='Reds')
        cbar4 = fig.colorbar(im4, ax=axes[1, 1], fraction=0.02, pad=0.01)
        cbar4.set_label('log₁₀(W + 1)')
        
        axes[1, 0].set_xlabel('Frequency (Hz)')
        axes[1, 1].set_xlabel('Frequency (Hz)')
        
        plt.suptitle('Comparison: Original vs Standardized β (★ = true coupling)', fontsize=12, y=0.995)
        plt.tight_layout()
        plt.subplots_adjust(top=0.94)
        plt.savefig(f'{OUT}/heatmap_effect_comparison.png', dpi=150, bbox_inches='tight')
        plt.savefig(f'{OUT}/heatmap_effect_comparison.pdf', dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved heatmap_effect_comparison.png/pdf")
    
    # =========================================================================
    # 4. P-value Heatmaps - ORIGINAL
    # =========================================================================
    print("\n" + "="*60)
    print("4. P-VALUE HEATMAPS (Original)")
    print("="*60)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True)
    
    pval_data = [plv_pval, sfc_pval, joint_pval]
    pval_names = ['PLV (permutation)', 'SFC (permutation)', 'Joint Wald (original)']
    
    for ax, pv, name in zip(axes, pval_data, pval_names):
        log_p = -np.log10(np.clip(pv, 1e-10, 1))
        vmax_data = np.percentile(log_p[np.isfinite(log_p)], 99)
        vmax_data = max(vmax_data, -np.log10(ALPHA) + 1)
        
        im = plot_pval_heatmap_row(ax, pv, FREQS, name, freqs_true, masks, 
                                    cmap='hot_r', vmin=0, vmax=vmax_data)
        cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
        cbar.set_label('-log₁₀(p)')
        
        sig_line = -np.log10(ALPHA)
        if sig_line <= vmax_data:
            cbar.ax.axhline(sig_line, color='cyan', lw=2)
            cbar.ax.text(1.5, sig_line, f'α={ALPHA}', color='cyan', fontsize=7, va='center')
    
    axes[-1].set_xlabel('Frequency (Hz)')
    
    plt.suptitle('P-values - Original (★ = true coupling)', fontsize=12, y=0.995)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig(f'{OUT}/heatmap_pval_original.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{OUT}/heatmap_pval_original.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved heatmap_pval_original.png/pdf")
    
    # =========================================================================
    # 4b. P-value Heatmaps - STANDARDIZED
    # =========================================================================
    if HAS_STANDARDIZED:
        print("\n" + "="*60)
        print("4b. P-VALUE HEATMAPS (Standardized)")
        print("="*60)
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True)
        
        pval_data_std = [plv_pval, sfc_pval, joint_pval_std]
        pval_names_std = ['PLV (permutation)', 'SFC (permutation)', 'Joint Wald (standardized)']
        
        for ax, pv, name in zip(axes, pval_data_std, pval_names_std):
            log_p = -np.log10(np.clip(pv, 1e-10, 1))
            vmax_data = np.percentile(log_p[np.isfinite(log_p)], 99)
            vmax_data = max(vmax_data, -np.log10(ALPHA) + 1)
            
            im = plot_pval_heatmap_row(ax, pv, FREQS, name, freqs_true, masks, 
                                        cmap='hot_r', vmin=0, vmax=vmax_data)
            cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
            cbar.set_label('-log₁₀(p)')
            
            sig_line = -np.log10(ALPHA)
            if sig_line <= vmax_data:
                cbar.ax.axhline(sig_line, color='cyan', lw=2)
                cbar.ax.text(1.5, sig_line, f'α={ALPHA}', color='cyan', fontsize=7, va='center')
        
        axes[-1].set_xlabel('Frequency (Hz)')
        
        plt.suptitle('P-values - Standardized (★ = true coupling)', fontsize=12, y=0.995)
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        plt.savefig(f'{OUT}/heatmap_pval_standardized.png', dpi=150, bbox_inches='tight')
        plt.savefig(f'{OUT}/heatmap_pval_standardized.pdf', dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved heatmap_pval_standardized.png/pdf")
    # =========================================================================
    # 5. ROC & PR Curves
    # =========================================================================
    if HAS_SKLEARN:
        print("\n" + "="*60)
        print("5. ROC & PR CURVES")
        print("="*60)
        
        y = y_true.flatten()
        
        s_plv = -np.log10(np.clip(plv_pval, 1e-10, 1)).flatten()
        s_sfc = -np.log10(np.clip(sfc_pval, 1e-10, 1)).flatten()
        
        # Use standardized if available, otherwise original
        if HAS_STANDARDIZED:
            s_joint = -np.log10(np.clip(joint_pval_std, 1e-10, 1)).flatten()
            joint_label = 'Joint'
        else:
            s_joint = -np.log10(np.clip(joint_pval, 1e-10, 1)).flatten()
            joint_label = 'Joint'
        
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        
        # ROC
        ax = axes[0]
        for score, name, c in [(s_plv, 'PLV', C['plv']), 
                                (s_sfc, 'SFC', C['sfc']), 
                                (s_joint, joint_label, C['joint'])]:
            fpr, tpr, _ = roc_curve(y, score)
            a = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=c, lw=2, label=f'{name} (AUC={a:.3f})')
        
        ax.plot([0,1], [0,1], 'k--', lw=1)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc='lower right')
        ax.set_aspect('equal')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # PR
        ax = axes[1]
        for score, name, c in [(s_plv, 'PLV', C['plv']), 
                                (s_sfc, 'SFC', C['sfc']), 
                                (s_joint, joint_label, C['joint'])]:
            prec, rec, _ = precision_recall_curve(y, score)
            ap = average_precision_score(y, score)
            ax.plot(rec, prec, color=c, lw=2, label=f'{name} (AP={ap:.3f})')
        
        ax.axhline(y.mean(), color='gray', ls='--', lw=1, label=f'Baseline ({y.mean():.2f})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc='upper right')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(f'{OUT}/roc_pr_curves.png', dpi=150, bbox_inches='tight')
        plt.savefig(f'{OUT}/roc_pr_curves.pdf', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("\nAUC:")
        print(f"  PLV:   {auc(*roc_curve(y, s_plv)[:2]):.3f}")
        print(f"  SFC:   {auc(*roc_curve(y, s_sfc)[:2]):.3f}")
        print(f"  Joint: {auc(*roc_curve(y, s_joint)[:2]):.3f}")
        print("  Saved roc_pr_curves.png/pdf")
    
    # =========================================================================
    # 6. Detection Metrics
    # =========================================================================
    print("\n" + "="*60)
    print("6. DETECTION METRICS")
    print("="*60)
    
    m_plv = metrics(y_true, plv_pval)
    m_sfc = metrics(y_true, sfc_pval)
    
    # Use standardized if available, otherwise original
    if HAS_STANDARDIZED:
        m_joint = metrics(y_true, joint_pval_std)
    else:
        m_joint = metrics(y_true, joint_pval)
    
    print(f"\nDetection Metrics @ α = {ALPHA}")
    print(f"Total: {S*B} pairs, Pos: {n_pos}, Neg: {n_neg}")
    print()
    print(f"{'':12} {'PLV':>10} {'SFC':>10} {'Joint':>10}")
    print("-"*45)
    print(f"{'TP':12} {m_plv['TP']:>10} {m_sfc['TP']:>10} {m_joint['TP']:>10}")
    print(f"{'FP':12} {m_plv['FP']:>10} {m_sfc['FP']:>10} {m_joint['FP']:>10}")
    print(f"{'TN':12} {m_plv['TN']:>10} {m_sfc['TN']:>10} {m_joint['TN']:>10}")
    print(f"{'FN':12} {m_plv['FN']:>10} {m_sfc['FN']:>10} {m_joint['FN']:>10}")
    print("-"*45)
    print(f"{'Sensitivity':12} {m_plv['sens']:>10.3f} {m_sfc['sens']:>10.3f} {m_joint['sens']:>10.3f}")
    print(f"{'Specificity':12} {m_plv['spec']:>10.3f} {m_sfc['spec']:>10.3f} {m_joint['spec']:>10.3f}")
    print(f"{'Precision':12} {m_plv['prec']:>10.3f} {m_sfc['prec']:>10.3f} {m_joint['prec']:>10.3f}")
    print(f"{'F1':12} {m_plv['f1']:>10.3f} {m_sfc['f1']:>10.3f} {m_joint['f1']:>10.3f}")
    
    # Bar plot
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))
    
    for ax, key, name in zip(axes, ['sens', 'spec', 'prec', 'f1'],
                              ['Sensitivity', 'Specificity', 'Precision', 'F1']):
        vals = [m_plv[key], m_sfc[key], m_joint[key]]
        bars = ax.bar(['PLV', 'SFC', 'Joint'], vals, 
                      color=[C['plv'], C['sfc'], C['joint']], width=0.6)
        ax.set_ylim([0, 1.1])
        ax.set_title(name)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.03, f'{v:.2f}', 
                    ha='center', fontsize=9)
    
    plt.suptitle(f'Detection Metrics (α = {ALPHA})', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUT}/metrics_bars.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{OUT}/metrics_bars.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved metrics_bars.png/pdf")
    # =========================================================================
    # 7. Phase Recovery
    # =========================================================================
    print("\n" + "="*60)
    print("7. PHASE RECOVERY")
    print("="*60)
    
    coupled = masks.flatten()
    true_ph = beta_phase_true.flatten()[coupled]
    joint_ph = joint_phase[:, idx_map].flatten()[coupled]
    plv_ph = plv_phase[:, idx_map].flatten()[coupled]
    
    err_joint = circ_diff(joint_ph, true_ph)
    err_plv = circ_diff(plv_ph, true_ph)
    
    mae_joint = np.abs(err_joint).mean()
    mae_plv = np.abs(err_plv).mean()
    
    if HAS_STANDARDIZED:
        joint_ph_std = joint_phase_std[:, idx_map].flatten()[coupled]
        err_joint_std = circ_diff(joint_ph_std, true_ph)
        mae_joint_std = np.abs(err_joint_std).mean()
    
    print(f"\nPhase MAE (coupled pairs, n={coupled.sum()}):")
    print(f"  Joint (orig): {np.degrees(mae_joint):.1f}°")
    if HAS_STANDARDIZED:
        print(f"  Joint (std):  {np.degrees(mae_joint_std):.1f}°")
    print(f"  PLV:          {np.degrees(mae_plv):.1f}°")
    
    if HAS_STANDARDIZED:
        fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    
    # Joint scatter (original)
    ax = axes[0]
    ax.scatter(np.degrees(true_ph), np.degrees(joint_ph), 
               c=C['joint'], s=50, alpha=0.7, edgecolors='white', lw=0.5)
    ax.plot([-180, 180], [-180, 180], 'k--', lw=1)
    ax.set_xlabel('True Phase (°)')
    ax.set_ylabel('Estimated Phase (°)')
    ax.set_title(f'Joint Orig (MAE={np.degrees(mae_joint):.1f}°)')
    ax.set_xlim([-180, 180])
    ax.set_ylim([-180, 180])
    ax.set_aspect('equal')
    
    if HAS_STANDARDIZED:
        # Joint scatter (standardized)
        ax = axes[1]
        ax.scatter(np.degrees(true_ph), np.degrees(joint_ph_std), 
                   c=C['joint_std'], s=50, alpha=0.7, edgecolors='white', lw=0.5)
        ax.plot([-180, 180], [-180, 180], 'k--', lw=1)
        ax.set_xlabel('True Phase (°)')
        ax.set_ylabel('Estimated Phase (°)')
        ax.set_title(f'Joint Std (MAE={np.degrees(mae_joint_std):.1f}°)')
        ax.set_xlim([-180, 180])
        ax.set_ylim([-180, 180])
        ax.set_aspect('equal')
        
        plv_ax_idx = 2
        hist_ax_idx = 3
    else:
        plv_ax_idx = 1
        hist_ax_idx = 2
    
    # PLV scatter
    ax = axes[plv_ax_idx]
    ax.scatter(np.degrees(true_ph), np.degrees(plv_ph),
               c=C['plv'], s=50, alpha=0.7, edgecolors='white', lw=0.5)
    ax.plot([-180, 180], [-180, 180], 'k--', lw=1)
    ax.set_xlabel('True Phase (°)')
    ax.set_ylabel('Estimated Phase (°)')
    ax.set_title(f'PLV (MAE={np.degrees(mae_plv):.1f}°)')
    ax.set_xlim([-180, 180])
    ax.set_ylim([-180, 180])
    ax.set_aspect('equal')
    
    # Error histogram
    ax = axes[hist_ax_idx]
    bins = np.linspace(-180, 180, 25)
    ax.hist(np.degrees(err_joint), bins, alpha=0.7, color=C['joint'], 
            label=f'Joint Orig ({np.degrees(mae_joint):.1f}°)', edgecolor='white')
    if HAS_STANDARDIZED:
        ax.hist(np.degrees(err_joint_std), bins, alpha=0.5, color=C['joint_std'],
                label=f'Joint Std ({np.degrees(mae_joint_std):.1f}°)', edgecolor='white')
    ax.hist(np.degrees(err_plv), bins, alpha=0.5, color=C['plv'],
            label=f'PLV ({np.degrees(mae_plv):.1f}°)', edgecolor='white')
    ax.axvline(0, color='k', ls='--', lw=1)
    ax.set_xlabel('Phase Error (°)')
    ax.set_ylabel('Count')
    ax.set_title('Error Distribution')
    ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{OUT}/phase_recovery.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{OUT}/phase_recovery.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved phase_recovery.png/pdf")
    # =========================================================================
    # 8. Summary
    # =========================================================================
    print("\n" + "="*60)
    print("8. SUMMARY")
    print("="*60)
    
    if HAS_SKLEARN:
        print(f"\nROC-AUC:")
        print(f"  PLV:   {auc(*roc_curve(y, s_plv)[:2]):.3f}")
        print(f"  SFC:   {auc(*roc_curve(y, s_sfc)[:2]):.3f}")
        print(f"  Joint: {auc(*roc_curve(y, s_joint)[:2]):.3f}")

    print(f"\nDetection @ α={ALPHA}:")
    print(f"  {'':10} {'Sens':>8} {'Spec':>8} {'Prec':>8} {'F1':>8}")
    print(f"  {'PLV':10} {m_plv['sens']:>8.3f} {m_plv['spec']:>8.3f} {m_plv['prec']:>8.3f} {m_plv['f1']:>8.3f}")
    print(f"  {'SFC':10} {m_sfc['sens']:>8.3f} {m_sfc['spec']:>8.3f} {m_sfc['prec']:>8.3f} {m_sfc['f1']:>8.3f}")
    print(f"  {'Joint':10} {m_joint['sens']:>8.3f} {m_joint['spec']:>8.3f} {m_joint['prec']:>8.3f} {m_joint['f1']:>8.3f}")
    
    print(f"\nPhase MAE:")
    print(f"  PLV:   {np.degrees(mae_plv):.1f}°")
    print(f"  Joint: {np.degrees(mae_joint):.1f}°")
    
    # Save results
    results = {
        'config': {'freqs': FREQS, 'freqs_true': freqs_true, 'alpha': ALPHA},
        'ground_truth': {'masks': masks, 'y_true': y_true},
        'plv': {'val': plv_val, 'pval': plv_pval, 'phase': plv_phase, 'metrics': m_plv},
        'sfc': {'val': sfc_val, 'pval': sfc_pval, 'metrics': m_sfc},
        'joint': {
            'mag': joint_mag_std if HAS_STANDARDIZED else joint_mag,
            'pval': joint_pval_std if HAS_STANDARDIZED else joint_pval,
            'phase': joint_phase_std if HAS_STANDARDIZED else joint_phase,
            'W': W_std if HAS_STANDARDIZED else W,
            'metrics': m_joint,
            'standardized': HAS_STANDARDIZED,
        },
        'phase': {'mae_joint': mae_joint, 'mae_plv': mae_plv},
    }
    
    # Also store original for reference
    if HAS_STANDARDIZED:
        results['joint_original'] = {
            'mag': joint_mag, 'pval': joint_pval, 'phase': joint_phase, 'W': W
        }
    
    if HAS_SKLEARN:
        results['plv']['auc'] = auc(*roc_curve(y, s_plv)[:2])
        results['sfc']['auc'] = auc(*roc_curve(y, s_sfc)[:2])
        results['joint']['auc'] = auc(*roc_curve(y, s_joint)[:2])
    
    with open(f'{OUT}/comparison_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved comparison_results.pkl")
    
    print("\n" + "="*60)
    print(f"All figures saved to: {OUT}")
    print("="*60)
   
    with open(f'{OUT}/comparison_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved comparison_results.pkl")
    
    print("\n" + "="*60)
    print(f"All figures saved to: {OUT}")
    print("="*60)


if __name__ == '__main__':
    main()