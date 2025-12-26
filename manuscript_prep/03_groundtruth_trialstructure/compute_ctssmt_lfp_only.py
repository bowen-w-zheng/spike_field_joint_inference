#!/usr/bin/env python3
"""
Compute CT-SSMT (LFP-only) inference for comparison with joint inference.

Uses the existing em_ct_hier_jax and upsample_ct_hier_fine functions.

FIXED: Properly computes trial-specific dynamics Z = X + D (not just X).

Usage:
    python compute_ctssmt_lfp_only.py --input ./data/sim_with_trials.pkl --output ./results/ctssmt_lfp_only.pkl
"""

import sys
import os
import pickle
import numpy as np
import argparse
from dataclasses import dataclass, asdict
from simulate_trial_data import TrialSimConfig
# Add the project root to the working directory and python path
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


@dataclass
class CTSSMTConfig:
    """Configuration for CT-SSMT inference."""
    # EM parameters
    max_iter: int = 5000
    tol: float = 1e-3
    sig_eps_init: float = 5.0
    log_every: int = 500
    
    # Spectrogram parameters
    window_sec: float = 0.4
    NW_product: int = 1
    
    # Frequency grid
    freq_min: float = 1.0
    freq_max: float = 61.0
    freq_step: float = 2.0


def compute_multitaper_spectrogram(lfp: np.ndarray, fs: float, freqs: np.ndarray,
                                   window_sec: float, NW_product: int):
    """
    Compute complex multitaper spectrogram with derotation.
    
    Returns
    -------
    Y_trials : (R, J, M, K)
        Complex spectrogram (trials, freqs, tapers, time bins)
    tfr_raw : (R, J, T)
        Raw TFR before downsampling (for comparison)
    """
    import mne
    from src.utils_multitaper import derotate_tfr_align_start
    
    R, T = lfp.shape
    
    # Compute TFR
    tfr_raw = mne.time_frequency.tfr_array_multitaper(
        lfp[:, None, :],
        sfreq=fs,
        freqs=freqs,
        n_cycles=freqs * window_sec,
        time_bandwidth=2 * NW_product,
        output="complex",
        zero_mean=False,
    ).squeeze()  # (R, J, T)
    
    # Store raw TFR for comparison
    tfr_raw_full = tfr_raw.copy()
    
    # Add taper dimension: (R, J, M, T) where M=1
    tfr_raw_mt = tfr_raw[:, :, None, :]
    
    # Derotate
    M = int(round(window_sec * fs))
    decim = 1
    tfr = derotate_tfr_align_start(tfr_raw_mt, freqs, fs, decim, M)
    
    # Scale by taper normalization
    tapers, _ = mne.time_frequency.multitaper.dpss_windows(M, NW_product, Kmax=1)
    scaling_factor = 2.0 / tapers.sum(axis=1)
    tfr = tfr * scaling_factor
    
    # Downsample to non-overlapping windows
    tfr_ds = tfr[:, :, :, ::M]
    
    return tfr_ds, tfr


def run_ctssmt_lfp_only(lfp: np.ndarray, fs: float, config: CTSSMTConfig,
                        delta_spk: float = 0.001) -> dict:
    """
    Run CT-SSMT inference using only LFP data.
    
    Uses em_ct_hier_jax for EM and upsample_ct_hier_fine for upsampling.
    
    IMPORTANT: Returns trial-specific dynamics Z_r = X + D_r, not just shared X.
    """
    from src.em_ct_hier_jax import em_ct_hier_jax
    from src.upsample_ct_hier_fine import upsample_ct_hier_fine
    
    R, T = lfp.shape
    
    # Frequency grid
    freqs = np.arange(config.freq_min, config.freq_max, config.freq_step, dtype=float)
    J = len(freqs)
    
    print(f"CT-SSMT (LFP-only): R={R} trials, J={J} freqs")
    print(f"  Frequencies: {freqs[0]:.1f} - {freqs[-1]:.1f} Hz")
    
    # Compute spectrogram
    print("Computing multitaper spectrogram...")
    Y_trials, tfr = compute_multitaper_spectrogram(
        lfp, fs, freqs, config.window_sec, config.NW_product
    )
    print(f"  Y_trials shape: {Y_trials.shape}")  # (R, J, M, K)
    print(f"  tfr_raw shape: {tfr.shape}")    # (R, J, M, T)
    
    R, J, M, K = Y_trials.shape
    
    # Run EM
    print("Running EM (em_ct_hier_jax)...")
    em_kwargs = dict(
        max_iter=config.max_iter,
        tol=config.tol,
        sig_eps_init=config.sig_eps_init,
        log_every=config.log_every
    )
    res = em_ct_hier_jax(Y_trials=Y_trials, db=config.window_sec, **em_kwargs)
    print("  EM complete")
    
    # Extract OU parameters
    lam_X = np.asarray(res.lam_X, float).reshape(J, M)
    sig_v_X = np.asarray(res.sigv_X, float).reshape(J, M)
    lam_D = np.asarray(res.lam_D, float).reshape(J, M)
    sig_v_D = np.asarray(res.sigv_D, float).reshape(J, M)
    
    if hasattr(res, "sig_eps_jmr"):
        sig_eps_jmr = np.asarray(res.sig_eps_jmr, float)
    elif hasattr(res, "sig_eps_mr"):
        sig_eps_mr = np.asarray(res.sig_eps_mr, float)
        sig_eps_jmr = np.broadcast_to(sig_eps_mr[None, :, :], (J, M, R))
    else:
        sig_eps_jmr = np.full((J, M, R), 5.0, float)
    
    print(f"  θ_X: λ range [{lam_X.min():.4f}, {lam_X.max():.4f}]")
    print(f"  θ_D: λ range [{lam_D.min():.4f}, {lam_D.max():.4f}]")
    
    # Upsample to fine grid
    print("Upsampling to fine grid...")
    ups = upsample_ct_hier_fine(
        Y_trials=Y_trials, 
        res=res, 
        delta_spk=delta_spk,
        win_sec=config.window_sec, 
        offset_sec=0.0, 
        T_f=None
    )
    
    # =========================================================================
    # CRITICAL FIX: Extract BOTH X (shared) and D (trial-specific) and sum them
    # =========================================================================
    
    # Check what attributes are available
    print(f"  Upsampled object attributes: {[a for a in dir(ups) if not a.startswith('_')]}")
    
    # Extract X (shared) and D (trial deviations)
    if hasattr(ups, 'X_mean') and hasattr(ups, 'D_mean'):
        X_mean = np.asarray(ups.X_mean)  # (J, M, T_fine) complex
        D_mean = np.asarray(ups.D_mean)  # (R, J, M, T_fine) complex
        X_var = np.asarray(ups.X_var)    # (J, M, T_fine) real
        D_var = np.asarray(ups.D_var)    # (R, J, M, T_fine) real
        
        print(f"  X_mean shape: {X_mean.shape}")
        print(f"  D_mean shape: {D_mean.shape}")
        
        # Compute trial-specific latent: Z_r = X + D_r
        # X is (J, M, T_fine), D is (R, J, M, T_fine)
        Z_mean_fine = X_mean[None, :, :, :] + D_mean  # (R, J, M, T_fine)
        Z_var_fine = X_var[None, :, :, :] + D_var     # (R, J, M, T_fine)
        
        print(f"  Z_mean_fine (X+D) shape: {Z_mean_fine.shape}")
        
        # Also store X separately for comparison
        X_mean_fine = X_mean
        
    elif hasattr(ups, 'Z_mean'):
        # Fallback if only Z_mean is available
        print("  Warning: Only Z_mean available, may not be trial-specific!")
        Z_mean_fine = np.asarray(ups.Z_mean)
        Z_var_fine = np.asarray(ups.Z_var) if hasattr(ups, 'Z_var') else None
        X_mean_fine = None
        print(f"  Z_mean_fine shape: {Z_mean_fine.shape}")
    else:
        raise ValueError(f"Upsampled object has neither X_mean/D_mean nor Z_mean! "
                        f"Available: {[a for a in dir(ups) if not a.startswith('_')]}")
    
    # Average across tapers
    Z_smooth_tapered = Z_mean_fine.mean(axis=2)  # (R, J, T_fine)
    T_fine = Z_smooth_tapered.shape[-1]
    
    print(f"  Z_smooth_tapered (taper-averaged) shape: {Z_smooth_tapered.shape}")
    
    # Resample to LFP resolution if needed
    if T_fine > T:
        ds = T_fine // T
        Z_smooth_full = Z_smooth_tapered[:, :, ::ds][:, :, :T]
    elif T_fine < T:
        # Upsample (shouldn't happen normally)
        from scipy.interpolate import interp1d
        t_fine = np.linspace(0, 1, T_fine)
        t_lfp = np.linspace(0, 1, T)
        Z_smooth_full = np.zeros((R, J, T), dtype=complex)
        for r in range(R):
            for j in range(J):
                f_re = interp1d(t_fine, Z_smooth_tapered[r, j, :].real, kind='linear')
                f_im = interp1d(t_fine, Z_smooth_tapered[r, j, :].imag, kind='linear')
                Z_smooth_full[r, j, :] = f_re(t_lfp) + 1j * f_im(t_lfp)
    else:
        Z_smooth_full = Z_smooth_tapered
    
    print(f"  Z_smooth_full shape: {Z_smooth_full.shape}")
    
    # Verify trial-specificity: check variance across trials
    trial_var = np.var(np.abs(Z_smooth_full), axis=0).mean()
    print(f"  Trial variance (should be > 0 for trial-specific): {trial_var:.6f}")
    
    # Squeeze tfr if it has taper dimension
    if tfr.ndim == 4:
        tfr = tfr.squeeze(axis=2)  # (R, J, T)
    for j, freq in enumerate(freqs):
        print(f"{freq:.0f} Hz: λ_X = {lam_X[j, 0]:.4f}")
    return {
        'Z_smooth_full': Z_smooth_full,     # (R, J, T) - trial-specific latent Z = X + D
        'Z_mean_fine': Z_mean_fine,         # (R, J, M, T_fine) - fine grid with tapers
        'Z_var_fine': Z_var_fine,           # (R, J, M, T_fine) - variance
        'X_mean_fine': X_mean_fine,         # (J, M, T_fine) - shared process only (for comparison)
        'tfr': tfr,                         # (R, J, T) - raw multitaper for comparison
        'Y_trials': Y_trials,               # (R, J, M, K) - downsampled spectrogram
        'params': {
            'lam_X': lam_X,
            'sig_v_X': sig_v_X,
            'lam_D': lam_D,
            'sig_v_D': sig_v_D,
            'sig_eps': sig_eps_jmr,
        },
        'freqs': freqs,
        'config': asdict(config),
        'block_size': int(round(config.window_sec * fs)),
    }


def main():
    parser = argparse.ArgumentParser(description='Compute CT-SSMT (LFP-only) inference')
    parser.add_argument('--input', type=str, required=True,
                        help='Input path for simulated data')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for results')
    parser.add_argument('--max_iter', type=int, default=5000,
                        help='Max EM iterations')
    parser.add_argument('--freq_min', type=float, default=1.0,
                        help='Minimum frequency')
    parser.add_argument('--freq_max', type=float, default=61.0,
                        help='Maximum frequency')
    parser.add_argument('--freq_step', type=float, default=2.0,
                        help='Frequency step')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input}")
    with open(args.input, 'rb') as f:
        data = pickle.load(f)
    
    lfp = data['LFP']
    delta_spk = data.get('delta_spk', 0.001)
    fs = 1.0 / delta_spk
    
    print(f"  LFP: {lfp.shape}")
    print(f"  Sampling rate: {fs} Hz")
    
    # Configuration
    config = CTSSMTConfig(
        max_iter=args.max_iter,
        freq_min=args.freq_min,
        freq_max=args.freq_max,
        freq_step=args.freq_step,
    )
    
    # Run CT-SSMT
    results = run_ctssmt_lfp_only(lfp, fs, config, delta_spk=delta_spk)
    
    # Add ground truth info if available
    if 'Z_lat' in data:
        results['ground_truth'] = {
            'Z_lat': data['Z_lat'],
            'freqs_hz': data['freqs_hz'],
        }
    
    # Save
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to {args.output}")
    print(f"  Z_smooth_full shape: {results['Z_smooth_full'].shape}")
    
    # Quick sanity check
    print("\nSanity check - trial variance at each frequency:")
    for j_idx in [0, 5, 10, 15, 20, 25]:
        if j_idx < results['Z_smooth_full'].shape[1]:
            freq = results['freqs'][j_idx]
            var_j = np.var(np.abs(results['Z_smooth_full'][:, j_idx, :]), axis=0).mean()
            print(f"  {freq:.0f} Hz: trial variance = {var_j:.4f}")


if __name__ == '__main__':
    main()
    