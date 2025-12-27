#!/usr/bin/env python3
"""
Compare Ground Truth vs Multitaper vs LFP-only CT-SSMT vs Joint Inference.

Generates:
1. Spectrogram comparison (4 panels) with matched scaling
2. Correlation over time (line plots per frequency)
3. Correlation box plot (grouped by frequency)
4. Correlation heatmaps
5. Time series snapshots at fine resolution with uncertainty bands

Usage:
    python compare_spectral_dynamics.py --sim ./data/sim_single_trial.pkl --joint ./results/joint.pkl --output ./spectral_dynamics_figures/
"""
import os
import sys
import pickle
import argparse
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import pearsonr

# Setup project root for imports
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import mne
from src.utils_multitaper import derotate_tfr_align_start

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.size'] = 10

# =============================================================================
# COLORS AND CONFIG - ALIGNED WITH NOTEBOOK
# =============================================================================
# Ground truth
C_GT = '#2ecc71'  # Green

# Method config matching notebook conventions
METHOD_CONFIG = {
    'mt': {
        'color': '#7f8c8d',      # Gray
        'label': 'Multi-taper',
        'linewidth': 1.5,
    },
    'lfp': {
        'color': '#3498db',      # Blue
        'label': 'LFP-only CT-SSMT',
        'linewidth': 2,
    },
    'spk': {
        'color': '#e74c3c',      # Red/Orange
        'label': 'Joint Inference',
        'linewidth': 2,
    },
}

# Aliases for backward compatibility
METHOD_CONFIG['multitaper'] = METHOD_CONFIG['mt']
METHOD_CONFIG['lfp_only'] = METHOD_CONFIG['lfp']
METHOD_CONFIG['joint'] = METHOD_CONFIG['spk']


def db_scale(power, ref=None):
    if ref is None:
        ref = np.nanmean(power)
    return 10 * np.log10(power / ref + 1e-10)


def find_optimal_scale(gt, est):
    """Find alpha such that alpha*est best matches gt (least squares)."""
    valid = (gt > 0) & (est > 0) & ~np.isnan(gt) & ~np.isnan(est)
    if valid.sum() < 10:
        return 1.0
    gt_valid = gt[valid]
    est_valid = est[valid]
    alpha = np.sum(gt_valid * est_valid) / (np.sum(est_valid**2) + 1e-10)
    return alpha


def fine_to_amplitude_JT(Z_fine, J, M):
    """Convert fine state (1, T, 2*J*M) to amplitude (J, T)."""
    T = Z_fine.shape[1]
    amplitude = np.zeros((J, T))
    
    for j in range(J):
        amp_tapers = np.zeros((M, T))
        for m in range(M):
            col_re = 2 * (j * M + m)
            col_im = col_re + 1
            amp_tapers[m, :] = np.sqrt(Z_fine[0, :, col_re]**2 + Z_fine[0, :, col_im]**2)
        amplitude[j, :] = amp_tapers.mean(axis=0)
    
    return amplitude


def fine_to_amplitude_with_uncertainty(Z_fine, Z_var_fine, J, M):
    """
    Convert fine state to amplitude with uncertainty.
    
    Parameters
    ----------
    Z_fine : (1, T, 2*J*M) - mean estimates
    Z_var_fine : (1, T, 2*J*M) - variance estimates
    J, M : dimensions
    
    Returns
    -------
    amplitude : (J, T) - mean amplitude
    amplitude_std : (J, T) - standard deviation of amplitude
    """
    T = Z_fine.shape[1]
    amplitude = np.zeros((J, T))
    amplitude_var = np.zeros((J, T))
    
    for j in range(J):
        amp_tapers = np.zeros((M, T))
        var_tapers = np.zeros((M, T))
        
        for m in range(M):
            col_re = 2 * (j * M + m)
            col_im = col_re + 1
            
            # Mean values
            x = Z_fine[0, :, col_re]
            y = Z_fine[0, :, col_im]
            
            # Variances
            var_x = Z_var_fine[0, :, col_re]
            var_y = Z_var_fine[0, :, col_im]
            
            # Amplitude
            amp = np.sqrt(x**2 + y**2)
            amp_tapers[m, :] = amp
            
            # Propagate variance using delta method:
            # Var(|z|) ≈ (x²σ²_x + y²σ²_y) / (x² + y²)
            denom = x**2 + y**2 + 1e-10
            var_amp = (x**2 * var_x + y**2 * var_y) / denom
            var_tapers[m, :] = var_amp
        
        # Average over tapers
        amplitude[j, :] = amp_tapers.mean(axis=0)
        # Variance of mean = mean of variances / M (assuming independence)
        amplitude_var[j, :] = var_tapers.mean(axis=0) / M
    
    amplitude_std = np.sqrt(amplitude_var)
    return amplitude, amplitude_std


def downsample_to_blocks(data_JT, block_size):
    """Downsample (J, T) to (J, K) by averaging over blocks."""
    J, T = data_JT.shape
    K = T // block_size
    data_JK = np.zeros((J, K))
    for k in range(K):
        start = k * block_size
        end = start + block_size
        data_JK[:, k] = data_JT[:, start:end].mean(axis=1)
    return data_JK


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim', type=str, required=True)
    parser.add_argument('--joint', type=str, required=True)
    parser.add_argument('--output', type=str, default='./figures/')
    parser.add_argument('--window_sec', type=float, default=20.0, help='Correlation window size')
    parser.add_argument('--n_snapshots', type=int, default=4)
    parser.add_argument('--snapshot_sec', type=float, default=10.0)
    parser.add_argument('--ci_level', type=float, default=0.95, help='Confidence interval level')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    
    # CI multiplier (e.g., 1.96 for 95%)
    from scipy.stats import norm
    ci_mult = norm.ppf((1 + args.ci_level) / 2)

    # =========================================================================
    # LOAD DATA
    # =========================================================================
    print("Loading data...")
    with open(args.sim, 'rb') as f:
        sim = pickle.load(f)
    
    with open(args.joint, 'rb') as f:
        res = pickle.load(f)
    
    print(f"  Available keys in results: {list(res.keys())}")
    
    # Simulation parameters
    freqs_coupled = np.asarray(sim.get('freqs_hz_coupled', sim.get('freqs_hz', [])), float)
    freqs_all_signal = np.asarray(sim['freqs_hz'], float)
    fs = sim.get('fs', 1000.0)
    delta_spk = sim.get('delta_spk', 0.001)
    duration = len(sim['LFP']) / fs
    
    # Results parameters
    freqs = res['freqs']
    window_sec = res['window_sec']
    J = len(freqs)
    M = res.get('n_tapers', 3)
    NW = res.get('NW', 2.0)
    
    # Fine time parameters
    T_fine = int(duration / delta_spk)
    block_samples = int(window_sec / delta_spk)
    K = T_fine // block_samples
    
    # Sampling rates
    fs_fine = 1.0 / delta_spk  # 1000 Hz for CT-SSMT
    
    print(f"  Duration: {duration:.1f}s, T_fine: {T_fine}, K: {K}")
    print(f"  J: {J}, M: {M}, NW: {NW}")
    print(f"  fs: {fs} Hz, fs_fine: {fs_fine} Hz")
    print(f"  Signal frequencies: {freqs_all_signal}")
    print(f"  Coupled frequencies: {freqs_coupled}")

    # =========================================================================
    # EXTRACT AMPLITUDE AT FINE RESOLUTION
    # =========================================================================
    
    # 1. Ground truth amplitude
    A_gt = sim['A_t']  # (n_signals, T_fine)
    freqs_signal = sim['freqs_hz']
    
    # Map to analysis frequency grid
    amp_gt_fine = np.zeros((J, T_fine))
    signal_freq_to_j = {}
    for i, f_sig in enumerate(freqs_signal):
        j_idx = np.argmin(np.abs(freqs - f_sig))
        if np.abs(freqs[j_idx] - f_sig) < 1.5 and i < A_gt.shape[0]:
            amp_gt_fine[j_idx, :] = A_gt[i, :T_fine]
            signal_freq_to_j[f_sig] = j_idx
    
    print(f"  Ground truth shape: {amp_gt_fine.shape}")
    
    # 2. LFP-only CT-SSMT amplitude (from fine resolution)
    if 'Z_fine_em' in res and res['Z_fine_em'] is not None:
        Z_fine_em = res['Z_fine_em']
        amp_lfp_fine = fine_to_amplitude_JT(Z_fine_em, J, M)
        print(f"  LFP-only shape: {amp_lfp_fine.shape}")
    else:
        print("  WARNING: Z_fine_em not found")
        amp_lfp_fine = np.zeros((J, T_fine))
    
    # 3. Joint inference amplitude WITH UNCERTAINTY
    if 'Z_fine_joint' in res and res['Z_fine_joint'] is not None:
        Z_fine_joint = res['Z_fine_joint']
        Z_var_fine_joint = res.get('Z_var_fine_joint', None)
        
        if Z_var_fine_joint is not None:
            amp_joint_fine, amp_joint_std = fine_to_amplitude_with_uncertainty(
                Z_fine_joint, Z_var_fine_joint, J, M
            )
            print(f"  Joint shape: {amp_joint_fine.shape} (with uncertainty)")
        else:
            amp_joint_fine = fine_to_amplitude_JT(Z_fine_joint, J, M)
            amp_joint_std = None
            print(f"  Joint shape: {amp_joint_fine.shape} (no uncertainty)")
    else:
        print("  WARNING: Z_fine_joint not found")
        amp_joint_fine = np.zeros((J, T_fine))
        amp_joint_std = None
    
    # 4. Multitaper amplitude - compute at FULL resolution
    print("  Computing multitaper at full resolution...")
    lfp = sim['LFP']
    n_tapers = int(2 * NW - 1)
    
    tfr_raw = mne.time_frequency.tfr_array_multitaper(
        lfp[None, None, :],
        sfreq=fs,
        freqs=freqs,
        n_cycles=freqs * window_sec,
        time_bandwidth=2 * NW,
        output='complex',
        zero_mean=False,
    )
    
    # Reshape to (J, M, T)
    if tfr_raw.ndim == 5:
        tfr = tfr_raw[0, 0, :, :, :].transpose(1, 0, 2)
    else:
        tfr = tfr_raw[0, 0, :, :][:, None, :]
    
    J_tfr, M_tfr, T_tfr = tfr.shape
    M_samples = int(window_sec * fs)
    
    # Derotate
    tfr = derotate_tfr_align_start(tfr, freqs, fs, M_tfr, M_samples)
    
    # Taper scaling - use first taper only
    tapers, _ = mne.time_frequency.multitaper.dpss_windows(M_samples, NW, Kmax=n_tapers)
    scaling = 2.0 / tapers[0].sum()
    tfr = tfr * scaling
    
    # Average over tapers to get amplitude (J, T)
    amp_mt_fine = np.abs(tfr).mean(axis=1)  # (J, T)
    print(f"  Multitaper shape: {amp_mt_fine.shape}")
    
    # Ensure all same length
    T_min = min(amp_gt_fine.shape[1], amp_lfp_fine.shape[1], 
                amp_joint_fine.shape[1], amp_mt_fine.shape[1])
    amp_gt_fine = amp_gt_fine[:, :T_min]
    amp_lfp_fine = amp_lfp_fine[:, :T_min]
    amp_joint_fine = amp_joint_fine[:, :T_min]
    amp_mt_fine = amp_mt_fine[:, :T_min]
    if amp_joint_std is not None:
        amp_joint_std = amp_joint_std[:, :T_min]
    T_fine = T_min
    
    print(f"  Final T_fine: {T_fine}")
    
    # Signal band indices
    idx_sig = list(signal_freq_to_j.values())
    idx_coupled = []
    for f_sig in freqs_coupled:
        j_idx = np.argmin(np.abs(freqs - f_sig))
        if np.abs(freqs[j_idx] - f_sig) < 1.5:
            idx_coupled.append(j_idx)
    idx_coupled = list(set(idx_coupled))
    
    print(f"  Signal band indices: {idx_sig}")
    print(f"  Coupled band indices: {idx_coupled}")

    # =========================================================================
    # FIND OPTIMAL SCALING
    # =========================================================================
    print("\nFinding optimal scales...")
    
    gt_flat = amp_gt_fine[idx_sig, :].flatten()
    
    scale_mt = find_optimal_scale(gt_flat, amp_mt_fine[idx_sig, :].flatten())
    scale_lfp = find_optimal_scale(gt_flat, amp_lfp_fine[idx_sig, :].flatten())
    scale_joint = find_optimal_scale(gt_flat, amp_joint_fine[idx_sig, :].flatten())
    
    print(f"  Multi-taper scale: {scale_mt:.4f}")
    print(f"  LFP-only scale: {scale_lfp:.4f}")
    print(f"  Joint scale: {scale_joint:.4f}")
    
    amp_mt_scaled = amp_mt_fine * scale_mt
    amp_lfp_scaled = amp_lfp_fine * scale_lfp
    amp_joint_scaled = amp_joint_fine * scale_joint
    if amp_joint_std is not None:
        amp_joint_std_scaled = amp_joint_std * scale_joint

    # =========================================================================
    # COMPUTE CORRELATIONS OVER TIME WINDOWS
    # =========================================================================
    print(f"\nComputing correlations (window = {args.window_sec}s)...")
    
    time_window = args.window_sec
    time_bins = int(time_window * fs_fine)  # samples per window at fine resolution
    
    total_duration = T_fine / fs_fine
    n_windows = int(total_duration / time_window)
    time_centers = np.arange(n_windows) * time_window + time_window / 2
    
    print(f"  Analyzing {n_windows} windows of {time_window}s each")
    print(f"  Total duration: {n_windows * time_window}s")
    
    methods = ['mt', 'lfp', 'spk']
    method_data = {
        'mt': amp_mt_scaled,
        'lfp': amp_lfp_scaled,
        'spk': amp_joint_scaled,
    }
    
    # Initialize storage
    correlations = {m: np.zeros((len(idx_sig), n_windows)) for m in methods}
    correlation_pvals = {m: np.zeros((len(idx_sig), n_windows)) for m in methods}
    
    # Process each time window
    for win_idx in range(n_windows):
        start_sample = int(win_idx * time_bins)
        end_sample = int(start_sample + time_bins)
        
        if end_sample > T_fine:
            break
        
        for freq_idx, j in enumerate(idx_sig):
            # Ground truth power
            gt_amp = amp_gt_fine[j, start_sample:end_sample]
            gt_power = gt_amp ** 2
            
            if gt_power.std() < 1e-10:
                continue
            
            for method in methods:
                est_amp = method_data[method][j, start_sample:end_sample]
                est_power = est_amp ** 2
                
                if est_power.std() < 1e-10:
                    continue
                
                try:
                    corr, pval = pearsonr(gt_power, est_power)
                    correlations[method][freq_idx, win_idx] = corr
                    correlation_pvals[method][freq_idx, win_idx] = pval
                except:
                    pass

    # =========================================================================
    # FIGURE 1: SPECTROGRAM COMPARISON
    # =========================================================================
    print("\nGenerating spectrogram comparison...")
    
    K_plot = T_fine // block_samples
    power_gt_block = downsample_to_blocks(amp_gt_fine**2, block_samples)
    power_mt_block = downsample_to_blocks(amp_mt_scaled**2, block_samples)
    power_lfp_block = downsample_to_blocks(amp_lfp_scaled**2, block_samples)
    power_joint_block = downsample_to_blocks(amp_joint_scaled**2, block_samples)
    
    # Convert to dB
    def to_db(power):
        return 10 * np.log10(power + 1e-10)
    
    db_gt = to_db(power_gt_block)
    db_mt = to_db(power_mt_block)
    db_lfp = to_db(power_lfp_block)
    db_joint = to_db(power_joint_block)
    
    # Set zeros to nan for ground truth
    db_gt[power_gt_block == 0] = np.nan
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True, sharey=True)
    
    extent = [0, duration, freqs[0], freqs[-1]]
    
    panels = [
        ('gt', db_gt, 'Ground Truth', C_GT),
        ('mt', db_mt, METHOD_CONFIG['mt']['label'], METHOD_CONFIG['mt']['color']),
        ('lfp', db_lfp, METHOD_CONFIG['lfp']['label'], METHOD_CONFIG['lfp']['color']),
        ('spk', db_joint, METHOD_CONFIG['spk']['label'], METHOD_CONFIG['spk']['color']),
    ]
    
    print("  Color scale (dB) for each panel:")
    
    for ax, (key, db_power, title, color) in zip(axes, panels):
        # Compute vmin/vmax for this panel
        valid = db_power[~np.isnan(db_power)]
        if len(valid) > 0:
            vmax = np.percentile(valid, 100)
            vmin = vmax - 40
        else:
            vmin, vmax = -50, 50
        
        print(f"    {title:30s}: vmin={vmin:+.1f} dB, vmax={vmax:+.1f} dB")
        
        im = ax.imshow(db_power, aspect='auto', origin='lower',
                       extent=extent, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(title, fontweight='bold', color=color)
        
        # Mark true signal frequencies
        for f in freqs_all_signal:
            ax.axhline(f, color='white', ls='--', alpha=0.7, lw=1)
    
    axes[-1].set_xlabel('Time (s)')
    
    plt.tight_layout()
    plt.savefig(f'{args.output}/spectrogram_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{args.output}/spectrogram_comparison.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved spectrogram_comparison.png/pdf")

    # =========================================================================
    # FIGURE 2: CORRELATION OVER TIME (Line plots per frequency)
    # =========================================================================
    print("\nGenerating correlation over time plots...")
    
    fig, axes = plt.subplots(len(idx_sig), 1, figsize=(12, 3*len(idx_sig)), sharex=True)
    if len(idx_sig) == 1:
        axes = [axes]
    
    for freq_idx, (ax, j) in enumerate(zip(axes, idx_sig)):
        freq_val = freqs[j]
        
        for method in methods:
            config = METHOD_CONFIG[method]
            corr_values = correlations[method][freq_idx, :]
            
            ax.plot(time_centers, corr_values,
                   color=config['color'],
                   lw=config['linewidth'],
                   marker='o',
                   markersize=5,
                   label=config['label'],
                   alpha=0.8)
            
            # Mark significant correlations (p < 0.05)
            pvals = correlation_pvals[method][freq_idx, :]
            significant = pvals < 0.05
            
            if np.any(significant):
                ax.scatter(time_centers[significant],
                          corr_values[significant],
                          color=config['color'],
                          s=50,
                          zorder=5,
                          edgecolors='white',
                          linewidth=1)
        
        ax.set_ylabel('Correlation with GT')
        ax.set_title(f'Frequency: {freq_val:.1f} Hz')
        ax.set_ylim([-1, 1])
        ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.3)
        ax.axhline(-0.5, color='gray', linestyle=':', alpha=0.3)
        ax.grid(True, alpha=0.3)
        
        if freq_idx == 0:
            ax.legend(loc='upper right', fontsize='small')
    
    axes[-1].set_xlabel('Time (center of window, seconds)')
    fig.suptitle(f'Correlation with Ground Truth (window = {time_window}s)', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f'{args.output}/correlation_over_time.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{args.output}/correlation_over_time.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved correlation_over_time.png/pdf")

    # =========================================================================
    # FIGURE 3: BOX PLOT (Grouped by frequency)
    # =========================================================================
    print("\nGenerating correlation box plot...")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    box_data = []
    box_positions = []
    box_colors = []
    tick_labels = []
    
    x_pos = 0
    freq_spacing = len(methods) + 1
    
    for freq_idx, j in enumerate(idx_sig):
        freq_val = freqs[j]
        
        for i, method in enumerate(methods):
            box_data.append(correlations[method][freq_idx, :])
            box_positions.append(x_pos + i)
            box_colors.append(METHOD_CONFIG[method]['color'])
        
        tick_labels.append(f'{freq_val:.1f} Hz')
        x_pos += freq_spacing
    
    bp = ax.boxplot(box_data, positions=box_positions, widths=0.6,
                    patch_artist=True, showfliers=True)
    
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xticks([i * freq_spacing + (len(methods)-1)/2 for i in range(len(idx_sig))])
    ax.set_xticklabels(tick_labels)
    
    ax.set_ylabel('Correlation with Ground Truth')
    ax.set_xlabel('Signal Frequency')
    ax.set_title(f'Distribution of Correlations across {n_windows} Time Windows ({time_window}s each)')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.3)
    ax.grid(True, alpha=0.3, axis='y')
    
    legend_elements = [plt.Rectangle((0,0), 1, 1,
                                      facecolor=METHOD_CONFIG[m]['color'],
                                      alpha=0.7,
                                      label=METHOD_CONFIG[m]['label'])
                       for m in methods]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'{args.output}/correlation_boxplot.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{args.output}/correlation_boxplot.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved correlation_boxplot.png/pdf")

    # =========================================================================
    # FIGURE 4: CORRELATION HEATMAPS
    # =========================================================================
    print("\nGenerating correlation heatmaps...")
    
    fig, axes = plt.subplots(1, len(methods), figsize=(15, 4))
    
    for idx, (method, ax) in enumerate(zip(methods, axes)):
        corr_matrix = correlations[method]  # (n_frequencies, n_windows)
        
        im = ax.imshow(corr_matrix, aspect='auto', cmap='RdBu_r',
                       vmin=-1, vmax=1, interpolation='nearest')
        
        ax.set_yticks(range(len(idx_sig)))
        ax.set_yticklabels([f'{freqs[j]:.1f} Hz' for j in idx_sig])
        
        n_ticks = min(10, n_windows)
        tick_indices = np.linspace(0, n_windows-1, n_ticks, dtype=int)
        ax.set_xticks(tick_indices)
        ax.set_xticklabels([f'{time_centers[i]:.0f}' for i in tick_indices])
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency' if idx == 0 else '')
        ax.set_title(METHOD_CONFIG[method]['label'])
        
        if idx == len(methods) - 1:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Correlation', rotation=270, labelpad=20)
    
    fig.suptitle(f'Correlation Heatmaps: Each Method vs Ground Truth ({time_window}s windows)',
                 fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'{args.output}/correlation_heatmaps.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{args.output}/correlation_heatmaps.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved correlation_heatmaps.png/pdf")

    # =========================================================================
    # FIGURE 5: TIME SERIES SNAPSHOTS WITH UNCERTAINTY BANDS
    # =========================================================================
    print("\nGenerating time series snapshots with uncertainty...")
    
    snapshot_samples = int(args.snapshot_sec / delta_spk)
    n_signals = len(idx_sig)
    
    if n_signals > 0 and T_fine > snapshot_samples:
        total_samples = T_fine - snapshot_samples
        if args.n_snapshots > 1:
            snapshot_starts = np.linspace(0, total_samples, args.n_snapshots, dtype=int)
        else:
            snapshot_starts = [total_samples // 2]
        
        fig, axes = plt.subplots(n_signals, args.n_snapshots,
                                 figsize=(4 * args.n_snapshots, 3 * n_signals),
                                 squeeze=False)
        
        for col, start_sample in enumerate(snapshot_starts):
            end_sample = start_sample + snapshot_samples
            t_local = np.arange(snapshot_samples) * delta_spk
            
            for row, j in enumerate(idx_sig):
                ax = axes[row, col]
                
                # Ground truth
                gt = amp_gt_fine[j, start_sample:end_sample]
                if gt.max() > 0:
                    ax.plot(t_local, gt, color=C_GT, lw=1.5,
                           label='Ground Truth', alpha=0.9)
                
                # Multi-taper
                ax.plot(t_local, amp_mt_scaled[j, start_sample:end_sample],
                       color=METHOD_CONFIG['mt']['color'], 
                       lw=METHOD_CONFIG['mt']['linewidth'], 
                       alpha=0.5,
                       label=METHOD_CONFIG['mt']['label'])
                
                # LFP-only
                ax.plot(t_local, amp_lfp_scaled[j, start_sample:end_sample],
                       color=METHOD_CONFIG['lfp']['color'], 
                       lw=METHOD_CONFIG['lfp']['linewidth'],
                       label=METHOD_CONFIG['lfp']['label'])
                
                # Joint with uncertainty band
                joint_mean = amp_joint_scaled[j, start_sample:end_sample]
                ax.plot(t_local, joint_mean,
                       color=METHOD_CONFIG['spk']['color'], 
                       lw=METHOD_CONFIG['spk']['linewidth'],
                       label=METHOD_CONFIG['spk']['label'])
                
                # Add uncertainty band for Joint if available
                if amp_joint_std is not None:
                    joint_std = amp_joint_std_scaled[j, start_sample:end_sample]
                    lower = joint_mean - ci_mult * joint_std
                    upper = joint_mean + ci_mult * joint_std
                    lower = np.maximum(lower, 0)  # Amplitude can't be negative
                    
                    ax.fill_between(t_local, lower, upper,
                                   color=METHOD_CONFIG['spk']['color'],
                                   alpha=0.2,
                                   label=f'{int(args.ci_level*100)}% CI')
                
                ax.set_xlim(0, args.snapshot_sec)
                ax.set_ylim(0, None)
                
                if row == 0:
                    start_sec = start_sample * delta_spk
                    ax.set_title(f't = {start_sec:.0f}-{start_sec + args.snapshot_sec:.0f} s',
                                fontweight='bold')
                if row == n_signals - 1:
                    ax.set_xlabel('Time (s)')
                if col == 0:
                    ax.set_ylabel(f'{freqs[j]:.0f} Hz\nAmplitude')
                
                if row == 0 and col == 0:
                    ax.legend(loc='upper right', fontsize=7, framealpha=0.9)
        
        plt.tight_layout()
        plt.savefig(f'{args.output}/timeseries_snapshots.png', dpi=150, bbox_inches='tight')
        plt.savefig(f'{args.output}/timeseries_snapshots.pdf', dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved timeseries_snapshots.png/pdf")
        if amp_joint_std is not None:
            print(f"  Note: Shaded region shows {int(args.ci_level*100)}% CI for Joint Inference")

    # =========================================================================
    # SUMMARY STATISTICS
    # =========================================================================
    print("\n" + "=" * 60)
    print("MEAN CORRELATIONS WITH GROUND TRUTH")
    print("=" * 60)
    
    for freq_idx, j in enumerate(idx_sig):
        freq_val = freqs[j]
        print(f"\nFrequency: {freq_val:.1f} Hz")
        print("-" * 40)
        
        for method in methods:
            corr_values = correlations[method][freq_idx, :]
            mean_corr = np.mean(corr_values)
            std_corr = np.std(corr_values)
            median_corr = np.median(corr_values)
            
            pvals = correlation_pvals[method][freq_idx, :]
            n_significant = np.sum(pvals < 0.05)
            
            print(f"  {METHOD_CONFIG[method]['label']:25s}: "
                  f"μ={mean_corr:+.3f}, σ={std_corr:.3f}, "
                  f"median={median_corr:+.3f}, "
                  f"sig={n_significant}/{n_windows}")
    
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    print(f"  Optimal scales: MT={scale_mt:.3f}, LFP={scale_lfp:.3f}, Joint={scale_joint:.3f}")
    
    for method in methods:
        all_corrs = correlations[method].flatten()
        print(f"  {METHOD_CONFIG[method]['label']:25s}: "
              f"mean={np.mean(all_corrs):+.3f}, median={np.median(all_corrs):+.3f}")
    
    print(f"\nFigures saved to {args.output}")


if __name__ == '__main__':
    main()