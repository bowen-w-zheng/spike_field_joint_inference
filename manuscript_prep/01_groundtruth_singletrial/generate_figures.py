#!/usr/bin/env python3
"""
Generate journal-ready figures comparing spike-field coupling methods.

This script creates publication-quality figures including:
1. Effect size heatmaps (PLV, SFC, Joint |β|)
2. P-value heatmaps  
3. ROC and Precision-Recall curves
4. Phase recovery analysis
5. Summary statistics

Usage:
    python generate_figures.py --data ./data/sim.pkl \
        --plv ./results/traditional_methods.pkl \
        --joint ./results/joint.pkl \
        --output ./figures/

Requirements:
    - matplotlib
    - numpy
    - scipy
    - sklearn
"""

import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import stats
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Style Configuration
# =============================================================================

# Method colors
COLORS = {
    'PLV': '#2E86AB',
    'SFC': '#A23B72', 
    'Joint': '#F18F01',
}

def set_style():
    """Set publication-quality style."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 12,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })


# =============================================================================
# Data Loading and Preparation
# =============================================================================

def load_results(data_path, plv_path, joint_path, ctssmt_path=None):
    """Load all results files."""
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    with open(plv_path, 'rb') as f:
        trad = pickle.load(f)
    
    with open(joint_path, 'rb') as f:
        joint = pickle.load(f)
    
    ctssmt = None
    if ctssmt_path and os.path.exists(ctssmt_path):
        with open(ctssmt_path, 'rb') as f:
            ctssmt = pickle.load(f)
    
    return data, trad, joint, ctssmt


def prepare_ground_truth(data, freqs_dense):
    """
    Prepare ground truth arrays on dense frequency grid.
    
    Returns
    -------
    y_true : (S, B) boolean mask of true couplings on dense grid
    mag_true : (S, B) true coupling magnitudes
    phase_true : (S, B) true coupling phases
    """
    freqs_true = data['freqs_hz']  # True signal frequencies
    masks = data['masks']  # (S, J_true) coupling mask
    beta_mag = data['beta_mag']  # (S, J_true)
    beta_phase = data['beta_phase']  # (S, J_true)
    
    S = masks.shape[0]
    B = len(freqs_dense)
    
    y_true = np.zeros((S, B), dtype=bool)
    mag_true = np.zeros((S, B))
    phase_true = np.zeros((S, B))
    
    # Map true frequencies to dense grid
    for j_true, f_true in enumerate(freqs_true):
        # Find closest frequency in dense grid
        j_dense = np.argmin(np.abs(freqs_dense - f_true))
        
        for s in range(S):
            if masks[s, j_true]:
                y_true[s, j_dense] = True
                mag_true[s, j_dense] = beta_mag[s, j_true]
                phase_true[s, j_dense] = beta_phase[s, j_true]
    
    return y_true, mag_true, phase_true


def get_method_results(trad, joint, freqs_dense):
    """
    Extract results from all methods.
    
    Returns dict with:
        {method: {'effect': (S, B), 'pval': (S, B), 'phase': (S, B)}}
    """
    results = {}
    
    # PLV
    if 'plv' in trad:
        results['PLV'] = {
            'effect': trad['plv']['values'],
            'pval': trad['plv']['pval'],
            'phase': trad['plv']['phase'],
        }
    
    # SFC
    if 'sfc' in trad:
        results['SFC'] = {
            'effect': trad['sfc']['values'],
            'pval': trad['sfc']['pval'],
            'phase': None,
        }
    
    # Joint
    if 'coupling' in joint:
        results['Joint'] = {
            'effect': joint['coupling']['beta_mag_mean'],
            'pval': joint['wald']['pval'],
            'phase': joint['coupling']['beta_phase_mean'],
        }
    elif 'trace' in joint:
        # Compute from trace
        beta = joint['trace']['beta']
        S, P = beta.shape[1], beta.shape[2]
        J = (P - 1) // 2
        
        beta_R = beta[:, :, 1:1+J]
        beta_I = beta[:, :, 1+J:1+2*J]
        
        mag = np.sqrt(beta_R**2 + beta_I**2).mean(axis=0)
        
        results['Joint'] = {
            'effect': mag,
            'pval': joint['wald']['pval'],
            'phase': np.arctan2(beta_I.mean(0), beta_R.mean(0)),
        }
    
    return results


# =============================================================================
# Figure 1: Effect Size Heatmaps
# =============================================================================

def plot_effect_heatmap(ax, effect, freqs, y_true, freqs_true, title, 
                        vmin=None, vmax=None, log_scale=False, cmap='viridis'):
    """Plot effect size heatmap with ground truth markers."""
    S, B = effect.shape
    
    if log_scale:
        effect_plot = np.log10(effect + 1e-6)
        if vmin is None:
            vmin = np.percentile(effect_plot[effect_plot > -5], 5)
        if vmax is None:
            vmax = np.percentile(effect_plot, 95)
    else:
        effect_plot = effect
        if vmax is None:
            vmax = np.percentile(effect, 95)
        if vmin is None:
            vmin = 0
    
    im = ax.imshow(effect_plot, aspect='auto', origin='lower',
                   extent=[freqs[0], freqs[-1], -0.5, S-0.5],
                   vmin=vmin, vmax=vmax, cmap=cmap)
    
    # Mark true couplings
    for s in range(S):
        for j, f in enumerate(freqs):
            if y_true[s, j]:
                ax.plot(f, s, '*', color='red', markersize=8, 
                        markeredgecolor='white', markeredgewidth=0.5)
    
    # Mark true frequency bands
    for f in freqs_true:
        ax.axvline(f, color='red', linestyle='--', alpha=0.3, linewidth=0.8)
    
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Unit')
    ax.set_title(title)
    ax.set_yticks(range(S))
    
    return im


def figure_effect_heatmaps(results, freqs, y_true, freqs_true, save_path=None):
    """Create effect size heatmap comparison figure."""
    methods = ['PLV', 'SFC', 'Joint']
    available = [m for m in methods if m in results]
    n_methods = len(available)
    
    fig, axes = plt.subplots(1, n_methods, figsize=(4*n_methods, 3.5))
    if n_methods == 1:
        axes = [axes]
    
    for ax, method in zip(axes, available):
        effect = results[method]['effect']
        
        # Use log scale for Joint β
        log_scale = (method == 'Joint')
        title = f'{method}' + (' (log₁₀|β|)' if log_scale else '')
        
        im = plot_effect_heatmap(ax, effect, freqs, y_true, freqs_true,
                                 title=title, log_scale=log_scale)
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# Figure 2: P-value Heatmaps
# =============================================================================

def plot_pval_heatmap(ax, pval, freqs, y_true, freqs_true, title, alpha=0.05):
    """Plot -log10(p-value) heatmap."""
    S, B = pval.shape
    
    # -log10 transform
    pval_plot = -np.log10(np.clip(pval, 1e-10, 1))
    
    vmax = np.percentile(pval_plot, 95)
    vmin = 0
    
    im = ax.imshow(pval_plot, aspect='auto', origin='lower',
                   extent=[freqs[0], freqs[-1], -0.5, S-0.5],
                   vmin=vmin, vmax=vmax, cmap='hot')
    
    # Mark significance threshold
    sig_threshold = -np.log10(alpha)
    
    # Mark true couplings
    for s in range(S):
        for j, f in enumerate(freqs):
            if y_true[s, j]:
                ax.plot(f, s, '*', color='cyan', markersize=8,
                        markeredgecolor='white', markeredgewidth=0.5)
    
    # Mark true frequency bands
    for f in freqs_true:
        ax.axvline(f, color='cyan', linestyle='--', alpha=0.3, linewidth=0.8)
    
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Unit')
    ax.set_title(f'{title} (-log₁₀ p)')
    ax.set_yticks(range(S))
    
    return im


def figure_pval_heatmaps(results, freqs, y_true, freqs_true, save_path=None):
    """Create p-value heatmap comparison figure."""
    methods = ['PLV', 'SFC', 'Joint']
    available = [m for m in methods if m in results]
    n_methods = len(available)
    
    fig, axes = plt.subplots(1, n_methods, figsize=(4*n_methods, 3.5))
    if n_methods == 1:
        axes = [axes]
    
    for ax, method in zip(axes, available):
        pval = results[method]['pval']
        im = plot_pval_heatmap(ax, pval, freqs, y_true, freqs_true, method)
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# Figure 3: ROC and PR Curves
# =============================================================================

def compute_roc_pr(y_true, scores):
    """Compute ROC and PR curves."""
    y_flat = y_true.ravel().astype(int)
    s_flat = scores.ravel()
    
    # Remove NaN
    valid = ~np.isnan(s_flat)
    y_flat = y_flat[valid]
    s_flat = s_flat[valid]
    
    if y_flat.sum() == 0 or y_flat.sum() == len(y_flat):
        return None, None, None, None, None, None
    
    # ROC
    fpr, tpr, _ = roc_curve(y_flat, s_flat)
    roc_auc = auc(fpr, tpr)
    
    # PR
    precision, recall, _ = precision_recall_curve(y_flat, s_flat)
    pr_auc = average_precision_score(y_flat, s_flat)
    
    return fpr, tpr, roc_auc, precision, recall, pr_auc


def figure_roc_pr(results, y_true, save_path=None):
    """Create ROC and PR curve comparison figure."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    methods = ['PLV', 'SFC', 'Joint']
    available = [m for m in methods if m in results]
    
    for method in available:
        effect = results[method]['effect']
        
        fpr, tpr, roc_auc, prec, rec, pr_auc = compute_roc_pr(y_true, effect)
        
        if fpr is not None:
            color = COLORS.get(method, 'gray')
            
            # ROC
            axes[0].plot(fpr, tpr, color=color, linewidth=2,
                        label=f'{method} (AUC={roc_auc:.3f})')
            
            # PR
            axes[1].plot(rec, prec, color=color, linewidth=2,
                        label=f'{method} (AP={pr_auc:.3f})')
    
    # ROC plot formatting
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve')
    axes[0].legend(loc='lower right')
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1])
    
    # PR plot formatting
    baseline = y_true.sum() / y_true.size
    axes[1].axhline(baseline, color='k', linestyle='--', linewidth=1, alpha=0.5,
                   label=f'Baseline ({baseline:.3f})')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend(loc='upper right')
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# Figure 4: Phase Recovery
# =============================================================================

def circular_mean_error(true_phase, est_phase, mask):
    """Compute circular mean absolute error."""
    diff = np.angle(np.exp(1j * (true_phase[mask] - est_phase[mask])))
    return np.abs(diff).mean(), np.abs(diff).std()


def figure_phase_recovery(results, phase_true, y_true, freqs, freqs_true, save_path=None):
    """Create phase recovery comparison figure."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    methods_with_phase = []
    for method in ['PLV', 'Joint']:
        if method in results and results[method]['phase'] is not None:
            methods_with_phase.append(method)
    
    if not methods_with_phase:
        print("No methods with phase estimates available")
        return None
    
    # Left: Scatter plot of true vs estimated phase
    ax = axes[0]
    
    for method in methods_with_phase:
        est_phase = results[method]['phase']
        color = COLORS.get(method, 'gray')
        
        # Only at true coupling locations
        mask = y_true
        true_flat = phase_true[mask]
        est_flat = est_phase[mask]
        
        ax.scatter(np.degrees(true_flat), np.degrees(est_flat),
                  c=color, alpha=0.6, s=50, label=method, edgecolors='white')
    
    ax.plot([-180, 180], [-180, 180], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel('True Phase (°)')
    ax.set_ylabel('Estimated Phase (°)')
    ax.set_title('Phase Recovery at True Couplings')
    ax.set_xlim([-180, 180])
    ax.set_ylim([-180, 180])
    ax.legend()
    ax.set_aspect('equal')
    
    # Right: Bar plot of circular MAE
    ax = axes[1]
    
    maes = []
    stds = []
    for method in methods_with_phase:
        est_phase = results[method]['phase']
        mae, std = circular_mean_error(phase_true, est_phase, y_true)
        maes.append(np.degrees(mae))
        stds.append(np.degrees(std))
    
    x = np.arange(len(methods_with_phase))
    colors = [COLORS.get(m, 'gray') for m in methods_with_phase]
    
    bars = ax.bar(x, maes, yerr=stds, capsize=5, color=colors, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(methods_with_phase)
    ax.set_ylabel('Circular MAE (°)')
    ax.set_title('Phase Estimation Error')
    ax.set_ylim([0, max(maes) * 1.3 if maes else 90])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# Figure 5: Detection Summary
# =============================================================================

def compute_detection_metrics(y_true, pval, alpha=0.05):
    """Compute detection metrics at significance threshold."""
    sig = pval < alpha
    
    tp = (sig & y_true).sum()
    fp = (sig & ~y_true).sum()
    fn = (~sig & y_true).sum()
    tn = (~sig & ~y_true).sum()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    fdr = fp / (tp + fp) if (tp + fp) > 0 else 0
    
    return {
        'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'PPV': ppv,
        'FDR': fdr,
    }


def figure_detection_summary(results, y_true, save_path=None):
    """Create detection summary bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    methods = ['PLV', 'SFC', 'Joint']
    available = [m for m in methods if m in results]
    
    metrics_list = []
    for method in available:
        pval = results[method]['pval']
        metrics = compute_detection_metrics(y_true, pval)
        metrics['method'] = method
        metrics_list.append(metrics)
    
    # Left: Sensitivity and Specificity
    ax = axes[0]
    x = np.arange(len(available))
    width = 0.35
    
    sens = [m['Sensitivity'] for m in metrics_list]
    spec = [m['Specificity'] for m in metrics_list]
    colors = [COLORS.get(m, 'gray') for m in available]
    
    bars1 = ax.bar(x - width/2, sens, width, label='Sensitivity', 
                   color=colors, alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, spec, width, label='Specificity',
                   color=colors, alpha=0.4, edgecolor='black', hatch='//')
    
    ax.set_xticks(x)
    ax.set_xticklabels(available)
    ax.set_ylabel('Rate')
    ax.set_title('Detection Performance (α=0.05)')
    ax.legend()
    ax.set_ylim([0, 1])
    
    # Add value labels
    for bar, val in zip(bars1, sens):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    for bar, val in zip(bars2, spec):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    # Right: FDR and PPV
    ax = axes[1]
    
    ppv = [m['PPV'] for m in metrics_list]
    fdr = [m['FDR'] for m in metrics_list]
    
    bars1 = ax.bar(x - width/2, ppv, width, label='PPV',
                   color=colors, alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, fdr, width, label='FDR',
                   color=colors, alpha=0.4, edgecolor='black', hatch='//')
    
    ax.set_xticks(x)
    ax.set_xticklabels(available)
    ax.set_ylabel('Rate')
    ax.set_title('Precision Metrics (α=0.05)')
    ax.legend()
    ax.set_ylim([0, 1])
    
    for bar, val in zip(bars1, ppv):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    for bar, val in zip(bars2, fdr):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# Summary Table
# =============================================================================

def print_summary_table(results, y_true, phase_true):
    """Print summary statistics table."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    methods = ['PLV', 'SFC', 'Joint']
    available = [m for m in methods if m in results]
    
    header = f"{'Metric':<20}" + "".join(f"{m:>15}" for m in available)
    print(header)
    print("-"*80)
    
    # ROC AUC
    row = f"{'ROC AUC':<20}"
    for method in available:
        effect = results[method]['effect']
        _, _, roc_auc, _, _, _ = compute_roc_pr(y_true, effect)
        row += f"{roc_auc:>15.3f}" if roc_auc else f"{'N/A':>15}"
    print(row)
    
    # PR AUC
    row = f"{'PR AUC':<20}"
    for method in available:
        effect = results[method]['effect']
        _, _, _, _, _, pr_auc = compute_roc_pr(y_true, effect)
        row += f"{pr_auc:>15.3f}" if pr_auc else f"{'N/A':>15}"
    print(row)
    
    # Detection metrics
    for metric in ['Sensitivity', 'Specificity', 'PPV', 'FDR']:
        row = f"{metric:<20}"
        for method in available:
            pval = results[method]['pval']
            metrics = compute_detection_metrics(y_true, pval)
            row += f"{metrics[metric]:>15.3f}"
        print(row)
    
    # Phase error (for methods with phase)
    row = f"{'Phase MAE (°)':<20}"
    for method in available:
        if results[method]['phase'] is not None:
            mae, _ = circular_mean_error(phase_true, results[method]['phase'], y_true)
            row += f"{np.degrees(mae):>15.1f}"
        else:
            row += f"{'N/A':>15}"
    print(row)
    
    print("="*80)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate comparison figures')
    parser.add_argument('--data', type=str, required=True,
                        help='Simulated data path')
    parser.add_argument('--plv', type=str, required=True,
                        help='Traditional methods results path')
    parser.add_argument('--joint', type=str, required=True,
                        help='Joint inference results path')
    parser.add_argument('--ctssmt', type=str, default=None,
                        help='CT-SSMT LFP-only results path (optional)')
    parser.add_argument('--output', type=str, default='./figures/',
                        help='Output directory for figures')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Set style
    set_style()
    
    # Load data
    print("Loading results...")
    data, trad, joint, ctssmt = load_results(
        args.data, args.plv, args.joint, args.ctssmt
    )
    
    # Get frequency grid from traditional methods
    freqs_dense = trad['config']['freqs']
    
    # Prepare ground truth
    print("Preparing ground truth...")
    y_true, mag_true, phase_true = prepare_ground_truth(data, freqs_dense)
    freqs_true = data['freqs_hz']
    
    print(f"  Dense grid: {len(freqs_dense)} frequencies")
    print(f"  True signal bands: {freqs_true}")
    print(f"  True couplings: {y_true.sum()} / {y_true.size}")
    
    # Get method results
    print("Extracting method results...")
    results = get_method_results(trad, joint, freqs_dense)
    print(f"  Available methods: {list(results.keys())}")
    
    # Generate figures
    print("\nGenerating figures...")
    
    # Figure 1: Effect heatmaps
    fig1 = figure_effect_heatmaps(
        results, freqs_dense, y_true, freqs_true,
        save_path=os.path.join(args.output, 'fig1_effect_heatmaps.png')
    )
    plt.close(fig1)
    
    # Figure 2: P-value heatmaps
    fig2 = figure_pval_heatmaps(
        results, freqs_dense, y_true, freqs_true,
        save_path=os.path.join(args.output, 'fig2_pval_heatmaps.png')
    )
    plt.close(fig2)
    
    # Figure 3: ROC and PR curves
    fig3 = figure_roc_pr(
        results, y_true,
        save_path=os.path.join(args.output, 'fig3_roc_pr_curves.png')
    )
    plt.close(fig3)
    
    # Figure 4: Phase recovery
    fig4 = figure_phase_recovery(
        results, phase_true, y_true, freqs_dense, freqs_true,
        save_path=os.path.join(args.output, 'fig4_phase_recovery.png')
    )
    if fig4:
        plt.close(fig4)
    
    # Figure 5: Detection summary
    fig5 = figure_detection_summary(
        results, y_true,
        save_path=os.path.join(args.output, 'fig5_detection_summary.png')
    )
    plt.close(fig5)
    
    # Print summary table
    print_summary_table(results, y_true, phase_true)
    
    # Save summary to file
    summary_path = os.path.join(args.output, 'summary.txt')
    with open(summary_path, 'w') as f:
        import sys
        old_stdout = sys.stdout
        sys.stdout = f
        print_summary_table(results, y_true, phase_true)
        sys.stdout = old_stdout
    print(f"\nSaved summary to: {summary_path}")
    
    print(f"\nAll figures saved to: {args.output}")


if __name__ == '__main__':
    main()
