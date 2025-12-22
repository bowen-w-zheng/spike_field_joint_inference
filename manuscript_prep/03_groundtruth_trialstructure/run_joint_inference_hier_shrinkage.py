#!/usr/bin/env python3
"""
Run HIERARCHICAL joint inference algorithm on trial data.
WITH BETA SHRINKAGE to fix low-frequency artifacts.

Usage:
    python run_joint_inference_hier_shrinkage.py --input ./data/sim_with_trials.pkl --output ./results/joint_inference.pkl
    
To disable shrinkage:
    python run_joint_inference_hier_shrinkage.py --input ./data/sim_with_trials.pkl --output ./results/joint.pkl --no_shrinkage
"""
import sys
import os
import pickle
import numpy as np
import argparse
import gc
from dataclasses import dataclass, asdict
from typing import Optional
from simulate_trial_data import TrialSimConfig
# Add path to joint inference source code
import pathlib

# Add the project root to the working directory and python path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Import the SHRINKAGE module
from src.run_joint_inference_trials_hier_shrinkage import (
    run_joint_inference_trials_hier,
    InferenceTrialsHierConfig,
)


@dataclass
class JointInferenceHierConfig:
    """Configuration for hierarchical joint inference."""
    # EM parameters
    max_iter: int = 5000
    tol: float = 1e-3
    sig_eps_init: float = 5.0
    log_every: int = 1000

    # MCMC parameters
    fixed_iter: int = 500
    n_refreshes: int = 10
    inner_steps_per_refresh: int = 100

    # Regularization
    omega_floor: float = 1e-3
    sigma_u: float = 0.1

    # Spectrogram
    window_sec: float = 0.4
    NW_product: int = 1

    # Frequency grid
    freq_min: float = 1.0
    freq_max: float = 61.0
    freq_step: float = 2.0

    # Memory optimization
    trace_thin: int = 2
    save_checkpoints: bool = False

    # Beta shrinkage
    use_beta_shrinkage: bool = True
    beta_shrinkage_burn_in: float = 0.5

    # Band-specific spike weighting
    use_spike_band_weights: bool = True
    spike_weight_floor: float = 0.01
    spike_weight_aggregation: str = "mean"
    verbose_band_weights: bool = True


def compute_multitaper_spectrogram(lfp: np.ndarray, fs: float, freqs: np.ndarray,
                                   window_sec: float, NW_product: int) -> np.ndarray:
    """Compute complex multitaper spectrogram with derotation."""
    import mne
    from src.utils_multitaper import derotate_tfr_align_start
    
    R, T = lfp.shape
    
    tfr_raw = mne.time_frequency.tfr_array_multitaper(
        lfp[:, None, :],
        sfreq=fs,
        freqs=freqs,
        n_cycles=freqs * window_sec,
        time_bandwidth=2 * NW_product,
        output="complex",
        zero_mean=False,
    ).squeeze()
    
    tfr_raw = tfr_raw[:, :, None, :]
    
    M = int(round(window_sec * fs))
    decim = 1
    tfr = derotate_tfr_align_start(tfr_raw, freqs, fs, decim, M)
    
    tapers, _ = mne.time_frequency.multitaper.dpss_windows(M, NW_product, Kmax=1)
    scaling_factor = 2.0 / tapers.sum(axis=1)
    tfr = tfr * scaling_factor
    
    tfr_ds = tfr[:, :, :, ::M]
    
    return tfr_ds


def build_history_design(spikes: np.ndarray, n_lags: int = 20) -> np.ndarray:
    """Build spike history design matrix."""
    R, S, T = spikes.shape
    H = np.zeros((S, R, T, n_lags), dtype=np.float32)
    
    for s in range(S):
        for r in range(R):
            for lag in range(n_lags):
                if lag + 1 < T:
                    H[s, r, lag+1:, lag] = spikes[r, s, :T-lag-1]
    
    return H


def trace_to_dict(trace, thin: int = 1) -> dict:
    """
    Convert Trace object to pickle-safe format with thinning.
    """
    trace_out = {}
    
    # Beta trace with thinning
    if hasattr(trace, 'beta') and len(trace.beta) > 0:
        beta_list = trace.beta[::thin]
        trace_out['beta'] = np.asarray(beta_list)
        print(f"  [TRACE] beta: {len(trace.beta)} -> {len(beta_list)} samples (thin={thin})")
    
    # Gamma trace with thinning
    if hasattr(trace, 'gamma') and len(trace.gamma) > 0:
        gamma_list = trace.gamma[::thin]
        trace_out['gamma'] = np.asarray(gamma_list)
        print(f"  [TRACE] gamma: {len(trace.gamma)} -> {len(gamma_list)} samples (thin={thin})")
    
    # X_fine and D_fine (final estimates)
    if hasattr(trace, 'X_fine') and len(trace.X_fine) > 0:
        trace_out['X_fine_final'] = np.asarray(trace.X_fine[-1])
        print(f"  [TRACE] X_fine_final: {trace_out['X_fine_final'].shape}")
    
    if hasattr(trace, 'D_fine') and len(trace.D_fine) > 0:
        trace_out['D_fine_final'] = np.asarray(trace.D_fine[-1])
        print(f"  [TRACE] D_fine_final: {trace_out['D_fine_final'].shape}")
    
    # Variance estimates
    if hasattr(trace, 'X_var_fine') and len(trace.X_var_fine) > 0:
        trace_out['X_var_fine_final'] = np.asarray(trace.X_var_fine[-1])
        print(f"  [TRACE] X_var_fine_final: {trace_out['X_var_fine_final'].shape}")
    
    if hasattr(trace, 'D_var_fine') and len(trace.D_var_fine) > 0:
        trace_out['D_var_fine_final'] = np.asarray(trace.D_var_fine[-1])
        print(f"  [TRACE] D_var_fine_final: {trace_out['D_var_fine_final'].shape}")
    
    # Averaged estimates across refreshes
    if hasattr(trace, 'X_fine_avg'):
        trace_out['X_fine_avg'] = np.asarray(trace.X_fine_avg)
        print(f"  [TRACE] X_fine_avg: {trace_out['X_fine_avg'].shape}")
    
    if hasattr(trace, 'D_fine_avg'):
        trace_out['D_fine_avg'] = np.asarray(trace.D_fine_avg)
        print(f"  [TRACE] D_fine_avg: {trace_out['D_fine_avg'].shape}")
    
    # Latent (for backward compatibility)
    if hasattr(trace, 'latent') and len(trace.latent) > 0:
        trace_out['latent'] = [np.asarray(trace.latent[-1])]
    
    # Scale factors
    if hasattr(trace, 'latent_scale_factors'):
        trace_out['latent_scale_factors'] = np.asarray(trace.latent_scale_factors)
    
    # Shrinkage factors
    if hasattr(trace, 'shrinkage_factors') and len(trace.shrinkage_factors) > 0:
        trace_out['shrinkage_factors'] = np.asarray(trace.shrinkage_factors)
        # print(f"  [TRACE] shrinkage_factors: {trace_out['shrinkage_factors'].shape}")

    # Spike band weights
    if hasattr(trace, 'spike_band_weights') and len(trace.spike_band_weights) > 0:
        trace_out['spike_band_weights'] = np.asarray(trace.spike_band_weights)
        print(f"  [TRACE] spike_band_weights: {trace_out['spike_band_weights'].shape}")

    return trace_out


def theta_to_dict(theta) -> dict:
    """Convert OUParams to pickle-safe dict."""
    return {
        'lam': np.asarray(theta.lam),
        'sig_v': np.asarray(theta.sig_v),
        'sig_eps': np.asarray(theta.sig_eps),
    }


def run_joint_inference_hier_wrapper(
    lfp: np.ndarray,
    spikes: np.ndarray,
    delta_spk: float,
    config: JointInferenceHierConfig,
    ground_truth_freqs: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
) -> dict:
    """Run hierarchical joint inference on trial data."""
    fs = 1.0 / delta_spk
    R, S, T = spikes.shape
    
    freqs_dense = np.arange(config.freq_min, config.freq_max, config.freq_step, dtype=float)
    J = len(freqs_dense)
    
    print(f"Hierarchical Joint Inference (WITH BETA SHRINKAGE)")
    print(f"  R={R} trials, S={S} units, J={J} freqs")
    print(f"  Frequency range: {freqs_dense[0]:.1f} - {freqs_dense[-1]:.1f} Hz")
    print(f"  Memory: trace_thin={config.trace_thin}")
    print(f"  Beta shrinkage: {config.use_beta_shrinkage}")
    if config.use_beta_shrinkage:
        print(f"  Shrinkage burn-in: {config.beta_shrinkage_burn_in}")
    print(f"  Spike band weights: {config.use_spike_band_weights}")
    if config.use_spike_band_weights:
        print(f"  Spike weight floor: {config.spike_weight_floor}")
        print(f"  Spike weight aggregation: {config.spike_weight_aggregation}")
    
    # Compute spectrogram
    print("Computing multitaper spectrogram...")
    Y_trials = compute_multitaper_spectrogram(
        lfp, fs, freqs_dense, config.window_sec, config.NW_product
    )
    print(f"  Y_trials shape: {Y_trials.shape}")
    
    # Build history design matrix
    print("Building history design matrix...")
    H_SRTL = build_history_design(spikes, n_lags=20)
    print(f"  H_SRTL shape: {H_SRTL.shape}")
    
    # Prepare spikes: (R, S, T) -> (S, R, T)
    spikes_SRT = np.swapaxes(spikes, 0, 1).astype(np.float32)
    
    # Configure inference
    inf_config = InferenceTrialsHierConfig(
        fixed_iter=config.fixed_iter,
        n_refreshes=config.n_refreshes,
        inner_steps_per_refresh=config.inner_steps_per_refresh,
        omega_floor=config.omega_floor,
        sigma_u=config.sigma_u,
        trace_thin=config.trace_thin,
        # Beta shrinkage
        use_beta_shrinkage=config.use_beta_shrinkage,
        beta_shrinkage_burn_in=config.beta_shrinkage_burn_in,
        # Band-specific spike weighting
        use_spike_band_weights=config.use_spike_band_weights,
        spike_weight_floor=config.spike_weight_floor,
        spike_weight_aggregation=config.spike_weight_aggregation,
        verbose_band_weights=config.verbose_band_weights,
        em_kwargs=dict(
            max_iter=config.max_iter,
            tol=config.tol,
            sig_eps_init=config.sig_eps_init,
            log_every=config.log_every
        ),
    )
    
    # Run inference
    print("Running hierarchical joint inference...")
    beta, gamma, theta_X, theta_D, trace = run_joint_inference_trials_hier(
        Y_trials=Y_trials,
        spikes_SRT=spikes_SRT,
        H_SRTL=H_SRTL,
        all_freqs=freqs_dense,
        delta_spk=delta_spk,
        window_sec=config.window_sec,
        offset_sec=0.0,
        config=inf_config
    )
    
    print(f"Results:")
    print(f"  beta shape: {beta.shape}")
    print(f"  gamma shape: {gamma.shape}")
    
    # Convert trace to pickle-safe format
    print("Converting trace to dict...")
    trace_dict = trace_to_dict(trace, thin=config.trace_thin)
    
    gc.collect()
    
    return {
        'beta': np.asarray(beta),
        'gamma': np.asarray(gamma),
        'theta_X': theta_to_dict(theta_X),
        'theta_D': theta_to_dict(theta_D),
        'trace': trace_dict,
        'freqs_dense': freqs_dense,
        'config': asdict(config),
    }


def extract_coupling_from_trace(trace_dict: dict, freqs_dense: np.ndarray, 
                                burnin_frac: float = 0.2,
                                thin: int = 1) -> dict:
    """Extract coupling magnitude and phase from MCMC trace."""
    trace_beta = trace_dict['beta']
    n_samples, S, D = trace_beta.shape
    J = len(freqs_dense)
    
    burnin = int(burnin_frac * n_samples)
    trace_beta = trace_beta[burnin::thin]
    n_samples = trace_beta.shape[0]
    
    R_slice = slice(1, 1 + J)
    I_slice = slice(1 + J, 1 + 2*J)
    
    beta_R = trace_beta[:, :, R_slice]
    beta_I = trace_beta[:, :, I_slice]
    
    beta_mag = np.sqrt(beta_R**2 + beta_I**2)
    beta_mag_mean = np.mean(beta_mag, axis=0)
    beta_mag_std = np.std(beta_mag, axis=0)
    
    beta_phase = np.arctan2(beta_I, beta_R)
    
    beta_phase_mean = np.zeros((S, J))
    beta_phase_R = np.zeros((S, J))
    
    for s in range(S):
        for j in range(J):
            phases = beta_phase[:, s, j]
            z = np.mean(np.exp(1j * phases))
            beta_phase_mean[s, j] = np.angle(z)
            beta_phase_R[s, j] = np.abs(z)
    
    return {
        'beta_mag_mean': beta_mag_mean,
        'beta_mag_std': beta_mag_std,
        'beta_phase_mean': beta_phase_mean,
        'beta_phase_R': beta_phase_R,
        'n_samples': n_samples,
    }


def compute_wald_significance(trace_dict: dict, freqs_dense: np.ndarray,
                              burnin_frac: float = 0.2) -> dict:
    """Compute Wald test significance for coupling parameters."""
    from scipy import stats
    
    trace_beta = trace_dict['beta']
    n_samples, S, D = trace_beta.shape
    J = len(freqs_dense)
    
    burnin = int(burnin_frac * n_samples)
    trace_beta = trace_beta[burnin:]
    
    R_slice = slice(1, 1 + J)
    I_slice = slice(1 + J, 1 + 2*J)
    
    beta_R = trace_beta[:, :, R_slice]
    beta_I = trace_beta[:, :, I_slice]
    
    beta_R_mean = np.mean(beta_R, axis=0)
    beta_I_mean = np.mean(beta_I, axis=0)
    beta_R_var = np.var(beta_R, axis=0)
    beta_I_var = np.var(beta_I, axis=0)
    
    W = np.zeros((S, J))
    pvals = np.zeros((S, J))
    
    for s in range(S):
        for j in range(J):
            if beta_R_var[s, j] > 1e-10 and beta_I_var[s, j] > 1e-10:
                W[s, j] = (beta_R_mean[s, j]**2 / beta_R_var[s, j] + 
                          beta_I_mean[s, j]**2 / beta_I_var[s, j])
                pvals[s, j] = 1 - stats.chi2.cdf(W[s, j], df=2)
            else:
                W[s, j] = np.nan
                pvals[s, j] = np.nan
    
    return {'W': W, 'pval_wald': pvals}


def main():
    parser = argparse.ArgumentParser(
        description='Run hierarchical joint inference with beta shrinkage'
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Input path for simulated data')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for results')
    parser.add_argument('--max_iter', type=int, default=10000,
                        help='Max EM iterations')
    parser.add_argument('--fixed_iter', type=int, default=300,
                        help='Fixed MCMC iterations')
    parser.add_argument('--n_refreshes', type=int, default=5,
                        help='Number of MCMC refreshes')
    parser.add_argument('--trace_thin', type=int, default=1,
                        help='Thinning factor for trace')
    parser.add_argument('--no_shrinkage', action='store_true',
                        help='Disable beta shrinkage')
    parser.add_argument('--shrinkage_burn_in', type=float, default=0.5,
                        help='Burn-in fraction for shrinkage computation')
    parser.add_argument('--no_spike_band_weights', action='store_true',
                        help='Disable band-specific spike weighting')
    parser.add_argument('--spike_weight_floor', type=float, default=0.01,
                        help='Minimum weight for spike band weighting')
    parser.add_argument('--spike_weight_aggregation', type=str, default='mean',
                        choices=['mean', 'min', 'max'],
                        help='How to aggregate shrinkage across neurons')
    parser.add_argument('--quiet_band_weights', action='store_true',
                        help='Suppress band weight diagnostics')

    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input}")
    with open(args.input, 'rb') as f:
        data = pickle.load(f)
    
    lfp = data['LFP']
    spikes = data['spikes']
    delta_spk = data['delta_spk']
    
    print(f"  LFP: {lfp.shape}, Spikes: {spikes.shape}")
    
    # Config
    config = JointInferenceHierConfig(
        max_iter=args.max_iter,
        fixed_iter=args.fixed_iter,
        n_refreshes=args.n_refreshes,
        trace_thin=args.trace_thin,
        use_beta_shrinkage=not args.no_shrinkage,
        beta_shrinkage_burn_in=args.shrinkage_burn_in,
        use_spike_band_weights=not args.no_spike_band_weights,
        spike_weight_floor=args.spike_weight_floor,
        spike_weight_aggregation=args.spike_weight_aggregation,
        verbose_band_weights=not args.quiet_band_weights,
    )
    
    # Run inference
    results = run_joint_inference_hier_wrapper(
        lfp, spikes, delta_spk, config,
        ground_truth_freqs=data.get('freqs_hz'),
        output_path=args.output,
    )
    
    # Extract summaries
    print("\nExtracting coupling summaries...")
    coupling = extract_coupling_from_trace(results['trace'], results['freqs_dense'])
    results['coupling'] = coupling
    
    # Wald significance
    print("Computing Wald significance...")
    wald = compute_wald_significance(results['trace'], results['freqs_dense'])
    results['wald'] = wald
    
    # Add ground truth if available
    if 'beta_mag' in data and 'beta_phase' in data:
        results['ground_truth'] = {
            'beta_mag': data['beta_mag'],
            'beta_phase': data['beta_phase'],
            'masks': data.get('masks'),
            'freqs_hz': data.get('freqs_hz'),
        }
    
    # Save
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nSaved to {args.output}")
    
    # Print summary
    print("\n" + "="*60)
    print("SAVED TRACE CONTENTS:")
    print("="*60)
    for key, val in results['trace'].items():
        if isinstance(val, np.ndarray):
            print(f"  {key}: {val.shape}")
        elif isinstance(val, list):
            print(f"  {key}: list of {len(val)} items")
        else:
            print(f"  {key}: {type(val)}")


if __name__ == '__main__':
    main()