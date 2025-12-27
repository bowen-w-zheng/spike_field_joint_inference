#!/usr/bin/env python3
"""
CLI wrapper for src/run_joint_inference_single_trial.py

UPDATED: Stores latent spectral dynamics for comparison:
- Y_cube: Raw input
- Z_mean_em: LFP-only CT-SSMT estimates  
- Z_mean_joint: Joint inference estimates
- beta_standardized: Beta in standardized units

Usage:
    python run_joint_inference_single.py --input ./data/sim_single_trial.pkl --output ./results/joint.pkl
"""
import sys
import os
import pickle
import argparse
import numpy as np
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import mne
from src.run_joint_inference_single_trial import (
    run_joint_inference_single_trial, 
    SingleTrialInferenceConfig
)
from src.simulate_single_trial import build_history_design_single
from src.utils_multitaper import derotate_tfr_align_start


def convert_fine_to_JMK(Z_fine, Z_var_fine, J, M, K, window_sec, delta_spk):
    """
    Convert fine state (1, T, 2*J*M) to block format (J, M, K) complex.
    
    Parameters
    ----------
    Z_fine : (1, T_fine, 2*J*M) real array
    Z_var_fine : (1, T_fine, 2*J*M) real array
    J, M : dimensions
    K : number of blocks
    window_sec : block duration
    delta_spk : fine time step
    
    Returns
    -------
    Z_mean : (J, M, K) complex
    Z_var : (J, M, K) real
    """
    T_fine = Z_fine.shape[1]
    block_size = int(window_sec / delta_spk)
    
    # Reconstruct complex from Re/Im layout: [Re_j0m0, Im_j0m0, Re_j0m1, ...]
    Z_complex = np.zeros((J, M, T_fine), dtype=complex)
    Z_var_out = np.zeros((J, M, T_fine), dtype=float)
    
    for j in range(J):
        for m in range(M):
            col_re = 2 * (j * M + m)
            col_im = col_re + 1
            Z_complex[j, m, :] = Z_fine[0, :, col_re] + 1j * Z_fine[0, :, col_im]
            Z_var_out[j, m, :] = Z_var_fine[0, :, col_re] + Z_var_fine[0, :, col_im]
    
    # Downsample to block centers
    K_actual = min(K, T_fine // block_size)
    Z_mean = Z_complex[:, :, ::block_size][:, :, :K_actual]
    Z_var = Z_var_out[:, :, ::block_size][:, :, :K_actual]
    
    return Z_mean, Z_var


def main():
    parser = argparse.ArgumentParser(
        description='Single-trial spike-field joint inference'
    )
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    
    # Frequency grid
    parser.add_argument('--freq_min', type=float, default=1.0)
    parser.add_argument('--freq_max', type=float, default=61.0)
    parser.add_argument('--freq_step', type=float, default=2.0)
    parser.add_argument('--window_sec', type=float, default=2.0)
    parser.add_argument('--NW', type=float, default=2.0,
                        help='Time-bandwidth product (n_tapers = 2*NW-1)')
    
    # MCMC parameters
    parser.add_argument('--fixed_iter', type=int, default=500)
    parser.add_argument('--n_refreshes', type=int, default=5)
    parser.add_argument('--inner_steps', type=int, default=100)
    parser.add_argument('--trace_thin', type=int, default=2)
    
    # EM parameters
    parser.add_argument('--em_max_iter', type=int, default=2000)
    
    # Wald test
    parser.add_argument('--no_wald_selection', action='store_true')
    parser.add_argument('--wald_alpha', type=float, default=0.05)
    
    # Beta shrinkage
    parser.add_argument('--no_shrinkage', action='store_true')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input}")
    with open(args.input, 'rb') as f:
        data = pickle.load(f)
    
    lfp = data['LFP']
    spikes = data['spikes']
    delta_spk = data.get('delta_spk', 0.001)
    fs = data.get('fs', 1000.0)
    
    print(f"  LFP shape: {lfp.shape}")
    print(f"  Spikes shape: {spikes.shape}")
    
    # Frequency grid
    freqs = np.arange(args.freq_min, args.freq_max, args.freq_step)
    J = len(freqs)
    n_tapers = int(2 * args.NW - 1)
    M = n_tapers
    
    print(f"  Using NW={args.NW}, n_tapers={n_tapers}")
    print(f"  Frequency grid: {J} bands ({freqs[0]:.0f}-{freqs[-1]:.0f} Hz)")
    
    # Compute spectrogram (matching compute_ctssmt_lfp_only_single.py)
    print("Computing multitaper spectrogram...")
    tfr_raw = mne.time_frequency.tfr_array_multitaper(
        lfp[None, None, :],
        sfreq=fs,
        freqs=freqs,
        n_cycles=freqs * args.window_sec,
        time_bandwidth=2 * args.NW,
        output='complex',
        zero_mean=False,
    )
    
    print(f"  tfr_raw shape from MNE: {tfr_raw.shape}")
    
    # Reshape to (J, M, T)
    if tfr_raw.ndim == 5:
        tfr = tfr_raw[0, 0, :, :, :].transpose(1, 0, 2)
    else:
        tfr = tfr_raw[0, 0, :, :][:, None, :]
    
    print(f"  tfr shape after reshape: {tfr.shape}")
    
    J, M, T = tfr.shape
    M_samples = int(args.window_sec * fs)
    
    # Derotate
    tfr = derotate_tfr_align_start(tfr, freqs, fs, M, M_samples)
    
    # Taper scaling - use first taper only
    tapers, _ = mne.time_frequency.multitaper.dpss_windows(M_samples, args.NW, Kmax=n_tapers)
    scaling = 2.0 / tapers[0].sum()
    tfr = tfr * scaling
    
    # Downsample to blocks
    Y_cube = tfr[:, :, ::M_samples]
    J, M_tapers, K = Y_cube.shape
    print(f"  Y_cube shape: {Y_cube.shape} (J={J}, M={M_tapers}, K={K})")
    
    # History design matrix
    print("Building history design matrix...")
    H_hist = build_history_design_single(spikes, n_lags=20)
    print(f"  H_hist shape: {H_hist.shape}")
    
    # Configure inference
    config = SingleTrialInferenceConfig(
        fixed_iter=args.fixed_iter,
        n_refreshes=args.n_refreshes,
        inner_steps_per_refresh=args.inner_steps,
        trace_thin=args.trace_thin,
        use_wald_band_selection=not args.no_wald_selection,
        wald_alpha=args.wald_alpha,
        use_beta_shrinkage=not args.no_shrinkage,
        em_kwargs=dict(max_iter=args.em_max_iter),
    )
    
    # Run inference
    print("Running joint inference...")
    beta, gamma, theta, trace = run_joint_inference_single_trial(
        Y_cube=Y_cube,
        spikes_ST=spikes,
        H_STL=H_hist,
        all_freqs=freqs,
        delta_spk=delta_spk,
        window_sec=args.window_sec,
        config=config,
    )
    
    print(f"  beta shape: {beta.shape}")
    print(f"  gamma shape: {gamma.shape}")
    
    # =================================================================
    # EXTRACT LATENT DYNAMICS FROM TRACE
    # =================================================================
    print("Extracting latent dynamics...")
    
    # EM (LFP-only) estimates
    if hasattr(trace, 'Z_fine_em') and trace.Z_fine_em is not None:
        Z_mean_em, Z_var_em = convert_fine_to_JMK(
            trace.Z_fine_em, trace.Z_var_em, J, M, K, args.window_sec, delta_spk
        )
        Z_smooth_em = Z_mean_em.mean(axis=1)  # (J, K)
        print(f"  Z_mean_em shape: {Z_mean_em.shape}")
    else:
        Z_mean_em, Z_var_em, Z_smooth_em = None, None, None
        print("  WARNING: Z_fine_em not found in trace")
    
    # Joint inference estimates
    if hasattr(trace, 'Z_fine_joint') and trace.Z_fine_joint is not None:
        Z_mean_joint, Z_var_joint = convert_fine_to_JMK(
            trace.Z_fine_joint, trace.Z_var_joint, J, M, K, args.window_sec, delta_spk
        )
        Z_smooth_joint = Z_mean_joint.mean(axis=1)  # (J, K)
        print(f"  Z_mean_joint shape: {Z_mean_joint.shape}")
    else:
        Z_mean_joint, Z_var_joint, Z_smooth_joint = None, None, None
        print("  WARNING: Z_fine_joint not found in trace")
    
    # Beta standardized
    beta_standardized = getattr(trace, 'beta_standardized', None)
    if beta_standardized is not None:
        print(f"  beta_standardized shape: {beta_standardized.shape}")
    
    # =================================================================
    # BUILD OUTPUT
    # =================================================================
    S = beta.shape[0]
    beta_R = beta[:, 1:1+J]
    beta_I = beta[:, 1+J:1+2*J]
    beta_mag = np.sqrt(beta_R**2 + beta_I**2)
    beta_phase = np.arctan2(beta_I, beta_R)
    
    output = {
        'Y_cube': np.asarray(Y_cube),
        # Raw input
        'Z_fine_em': np.asarray(trace.Z_fine_em) if hasattr(trace, 'Z_fine_em') and trace.Z_fine_em is not None else None,
        'Z_var_fine_em': np.asarray(trace.Z_var_em) if hasattr(trace, 'Z_var_em') and trace.Z_var_em is not None else None,
        
        # Joint inference - FINE RESOLUTION
        'Z_fine_joint': np.asarray(trace.Z_fine_joint) if hasattr(trace, 'Z_fine_joint') and trace.Z_fine_joint is not None else None,
        'Z_var_fine_joint': np.asarray(trace.Z_var_joint) if hasattr(trace, 'Z_var_joint') and trace.Z_var_joint is not None else None,
        
        # LFP-only CT-SSMT (from EM) - BLOCK RESOLUTION (keep for backward compatibility)
        'Z_smooth_em': Z_smooth_em,
        'Z_mean_em': Z_mean_em,
        'Z_var_em': Z_var_em,
        
        # Joint inference - BLOCK RESOLUTION (keep for backward compatibility)
        'Z_smooth_joint': Z_smooth_joint,
        'Z_mean_joint': Z_mean_joint,
        'Z_var_joint': Z_var_joint,
        
        
        # For backward compatibility
        'Z_smooth': Z_smooth_joint if Z_smooth_joint is not None else Z_smooth_em,
        
        # Coupling parameters
        'beta': beta,
        'beta_standardized': beta_standardized,
        'gamma': gamma,
        'latent_scale_factors': getattr(trace, 'latent_scale_factors', None),
        
        # EM (CT-SSMT) parameters
        'theta_em': {
            'lam': np.asarray(theta.lam),
            'sig_v': np.asarray(theta.sig_v),
            'sig_eps': np.asarray(theta.sig_eps),
        },
        
        # Coupling summary
        'coupling': {
            'beta_mag': beta_mag,
            'beta_phase': beta_phase,
        },
        
        # Trace (thinned)
        'trace': {
            'beta': np.stack(trace.beta) if trace.beta else None,
            'gamma': np.stack(trace.gamma) if trace.gamma else None,
        },
        
        # Metadata
        'freqs': freqs,
        'window_sec': args.window_sec,
        'NW': args.NW,
        'n_tapers': n_tapers,
        'delta_spk': delta_spk,
        'fs': fs,
        'config': {
            'fixed_iter': args.fixed_iter,
            'n_refreshes': args.n_refreshes,
            'inner_steps': args.inner_steps,
            'use_wald_selection': not args.no_wald_selection,
            'wald_alpha': args.wald_alpha,
            'use_beta_shrinkage': not args.no_shrinkage,
        },
    }
    
    # Add Wald test results
    if hasattr(trace, 'wald_significant_mask'):
        output['wald'] = {
            'significant_mask': np.asarray(trace.wald_significant_mask),
            'W_stats': np.asarray(trace.wald_W_stats),
            'pval': np.asarray(trace.wald_p_values),
        }
        n_sig = trace.wald_significant_mask.sum()
        print(f"  Wald test: {n_sig}/{J} bands significant")
    
    # Add shrinkage factors
    if hasattr(trace, 'shrinkage_factors') and trace.shrinkage_factors:
        output['shrinkage_factors'] = np.stack(trace.shrinkage_factors)
    
    # Add ground truth if available
    gt_keys = ['beta_mag', 'beta_phase', 'masks', 'freqs_hz']
    if any(k in data for k in gt_keys):
        output['ground_truth'] = {k: data[k] for k in gt_keys if k in data}
    
    # Save
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(output, f)
    
    print(f"\nSaved to {args.output}")
    print(f"  Output keys: {list(output.keys())}")


if __name__ == '__main__':
    main()