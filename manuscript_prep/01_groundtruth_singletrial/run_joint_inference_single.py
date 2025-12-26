#!/usr/bin/env python3
"""
CLI wrapper for src/run_joint_inference_single_trial.py

Usage:
    python run_joint_inference_single.py --input ./data/sim.pkl --output ./results/joint.pkl
    python run_joint_inference_single.py --input ./data/sim.pkl --output ./results/joint.pkl --n_refreshes 5 --inner_steps 200
    
    # Disable Wald gating:
    python run_joint_inference_single.py --input ./data/sim.pkl --output ./results/joint.pkl --no_wald_selection
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
# UPDATED: Import from run_joint_inference_single_trial
from src.run_joint_inference_single_trial import (
    run_joint_inference_single_trial, 
    SingleTrialInferenceConfig
)
from src.simulate_single_trial import build_history_design_single
from src.utils_multitaper import derotate_tfr_align_start


def main():
    parser = argparse.ArgumentParser(
        description='Single-trial spike-field joint inference'
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Input pickle file with LFP and spikes')
    parser.add_argument('--output', type=str, required=True,
                        help='Output pickle file for results')
    
    # Frequency grid
    parser.add_argument('--freq_min', type=float, default=1.0)
    parser.add_argument('--freq_max', type=float, default=61.0)
    parser.add_argument('--freq_step', type=float, default=2.0)
    parser.add_argument('--window_sec', type=float, default=0.4)
    
    # MCMC parameters - USE CORRECT NAMES from SingleTrialInferenceConfig
    parser.add_argument('--fixed_iter', type=int, default=500,
                        help='Warmup iterations')
    parser.add_argument('--n_refreshes', type=int, default=5,
                        help='Number of KF refresh passes')
    parser.add_argument('--inner_steps', type=int, default=100,
                        help='Inner PG-Gibbs steps per refresh')
    parser.add_argument('--trace_thin', type=int, default=2,
                        help='Thinning factor for trace')
    
    # EM parameters
    parser.add_argument('--em_max_iter', type=int, default=500,
                        help='Max EM iterations')
    
    # Wald test band selection
    parser.add_argument('--no_wald_selection', action='store_true',
                        help='Disable Wald test band selection')
    parser.add_argument('--wald_alpha', type=float, default=0.05,
                        help='Significance level for Wald test')
    
    # Beta shrinkage
    parser.add_argument('--no_shrinkage', action='store_true',
                        help='Disable beta shrinkage')
    
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
    print(f"  Frequency grid: {len(freqs)} bands ({freqs[0]:.0f}-{freqs[-1]:.0f} Hz)")
    
    # Compute spectrogram
    print("Computing multitaper spectrogram...")
    tfr = mne.time_frequency.tfr_array_multitaper(
        lfp[None, None, :], 
        sfreq=fs, 
        freqs=freqs,
        n_cycles=freqs * args.window_sec, 
        time_bandwidth=2, 
        output='complex', 
        zero_mean=False
    ).squeeze()[:, None, :]  # (J, 1, T)
    
    M = int(args.window_sec * fs)
    tfr = derotate_tfr_align_start(tfr, freqs, fs, 1, M)
    tapers, _ = mne.time_frequency.multitaper.dpss_windows(M, 1, Kmax=1)
    Y_cube = (tfr * (2.0 / tapers.sum(axis=1)))[:, :, ::M]
    print(f"  Y_cube shape: {Y_cube.shape}")
    
    # History design matrix
    print("Building history design matrix...")
    H_hist = build_history_design_single(spikes, n_lags=20)
    print(f"  H_hist shape: {H_hist.shape}")
    
    # Configure inference - USE CORRECT PARAMETER NAMES
    config = SingleTrialInferenceConfig(
        fixed_iter=args.fixed_iter,
        n_refreshes=args.n_refreshes,
        inner_steps_per_refresh=args.inner_steps,
        trace_thin=args.trace_thin,
        # Wald test band selection
        use_wald_band_selection=not args.no_wald_selection,
        wald_alpha=args.wald_alpha,
        # Beta shrinkage
        use_beta_shrinkage=not args.no_shrinkage,
        # EM kwargs
        em_kwargs=dict(max_iter=args.em_max_iter),
    )
    
    # Run inference - USE CORRECT FUNCTION NAME AND ARGUMENT NAMES
    print("Running joint inference...")
    beta, gamma, theta, trace = run_joint_inference_single_trial(
        Y_cube=Y_cube,
        spikes_ST=spikes,         # RENAMED from 'spikes'
        H_STL=H_hist,             # RENAMED from 'H_hist'
        all_freqs=freqs,          # RENAMED from 'freqs_hz'
        delta_spk=delta_spk,
        window_sec=args.window_sec,
        config=config,
    )
    
    print(f"  beta shape: {beta.shape}")
    print(f"  gamma shape: {gamma.shape}")
    
    # Build output dict
    output = {
        'beta': beta,
        'gamma': gamma,
        'theta': {
            'lam': np.asarray(theta.lam), 
            'sig_v': np.asarray(theta.sig_v), 
            'sig_eps': np.asarray(theta.sig_eps)
        },
        'trace': {
            'beta': np.stack(trace.beta) if trace.beta else None,
            'gamma': np.stack(trace.gamma) if trace.gamma else None,
        },
        'freqs': freqs,
        'config': {
            'fixed_iter': args.fixed_iter,
            'n_refreshes': args.n_refreshes,
            'inner_steps': args.inner_steps,
            'use_wald_selection': not args.no_wald_selection,
            'wald_alpha': args.wald_alpha,
            'use_beta_shrinkage': not args.no_shrinkage,
        },
    }
    
    # Add Wald test results if available
    if hasattr(trace, 'wald_significant_mask'):
        output['wald'] = {
            'significant_mask': np.asarray(trace.wald_significant_mask),
            'W_stats': np.asarray(trace.wald_W_stats),
            'pval': np.asarray(trace.wald_p_values),
        }
        n_sig = trace.wald_significant_mask.sum()
        print(f"  Wald test: {n_sig}/{len(freqs)} bands significant")
    
    # Add shrinkage factors if available
    if hasattr(trace, 'shrinkage_factors') and trace.shrinkage_factors:
        output['shrinkage_factors'] = np.stack(trace.shrinkage_factors)
    
    # Add coupling summary
    S, P = beta.shape
    J = (P - 1) // 2
    beta_R = beta[:, 1:1+J]
    beta_I = beta[:, 1+J:1+2*J]
    beta_mag = np.sqrt(beta_R**2 + beta_I**2)
    beta_phase = np.arctan2(beta_I, beta_R)
    
    output['coupling'] = {
        'beta_mag_mean': beta_mag,
        'beta_phase_mean': beta_phase,
    }
    
    # Add ground truth if available
    gt_keys = ['beta_mag', 'beta_phase', 'masks', 'freqs_hz']
    if any(k in data for k in gt_keys):
        output['ground_truth'] = {k: data[k] for k in gt_keys if k in data}
    
    # Save
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(output, f)
    
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()