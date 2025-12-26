#!/usr/bin/env python3
"""
Compute traditional spike-field coupling measures (PLV, SFC) for single-trial data.

For single-trial, we use surrogate/shuffle tests rather than across-trial permutation.
The permutation strategy is circular shifting of spikes relative to LFP.

Usage:
    python compute_traditional_methods_single.py --data ./data/sim_single_trial.pkl \
        --output ./results/traditional_methods.pkl

    # Or compute only one method:
    python compute_traditional_methods_single.py --data ./data/sim_single_trial.pkl \
        --output ./results/plv_results.pkl --method plv
"""

import os
import sys
import pickle
import argparse
import numpy as np
from scipy.signal import butter, filtfilt, hilbert, csd, welch


# =============================================================================
# Configuration
# =============================================================================
FS = 1000.0              # Sampling rate
WINDOW_SEC = 0.4         # Window duration
NW_PRODUCT = 1           # Time-bandwidth product
FREQS_DENSE = np.arange(1, 61, 2, dtype=float)  # [1, 3, 5, ..., 59] Hz
BANDWIDTH = 2 * NW_PRODUCT / WINDOW_SEC  # Hz


# =============================================================================
# PLV Computation (Single Trial)
# =============================================================================
def compute_plv_single(lfp, spikes, freqs, fs, bandwidth, n_permutations=500, seed=42):
    """
    Compute PLV with circular shift permutation test for single-trial data.
    
    Parameters
    ----------
    lfp : (T,) array
        LFP signal
    spikes : (S, T_fine) array
        Spike trains (binary)
    freqs : (B,) array
        Frequencies to analyze
    fs : float
        Sampling rate of LFP
    bandwidth : float
        Bandwidth for bandpass filter (Hz)
    n_permutations : int
        Number of permutations for significance test
    seed : int
        Random seed
    
    Returns
    -------
    plv : (S, B) array
        PLV values
    plv_pval : (S, B) array
        Permutation test p-values
    preferred_phase : (S, B) array
        Circular mean of spike phases
    """
    T = len(lfp)
    S, T_fine = spikes.shape
    B = len(freqs)
    
    ds = T_fine // T
    nyq = fs / 2
    
    plv = np.zeros((S, B))
    plv_pval = np.ones((S, B))
    preferred_phase = np.zeros((S, B))
    
    for j, f in enumerate(freqs):
        # Design bandpass filter
        low = max(f - bandwidth / 2, 0.5)
        high = min(f + bandwidth / 2, nyq - 1)
        
        if low >= high or high >= nyq:
            continue
        
        try:
            b, a = butter(2, [low / nyq, high / nyq], btype='band')
        except ValueError:
            continue
        
        # Filter LFP and compute phase
        lfp_filt = filtfilt(b, a, lfp)
        lfp_phase = np.angle(hilbert(lfp_filt))
        
        for s in range(S):
            # Downsample spikes to LFP resolution
            spk_fine = spikes[s, :]
            spk_coarse = spk_fine.reshape(-1, ds).max(axis=1)
            spike_idx = np.where(spk_coarse > 0)[0]
            
            n_spikes = len(spike_idx)
            if n_spikes < 5:
                continue
            
            # Get spike phases
            all_phases = lfp_phase[spike_idx]
            
            # Observed PLV
            z_obs = np.exp(1j * all_phases)
            plv[s, j] = np.abs(z_obs.mean())
            preferred_phase[s, j] = np.angle(z_obs.mean())
            
            # Permutation test: circular shift spikes
            null_plv = np.zeros(n_permutations)
            rng = np.random.default_rng(seed=seed + s * B + j)
            
            for perm in range(n_permutations):
                # Circular shift by random amount
                shift = rng.integers(T_fine // 4, 3 * T_fine // 4)
                spk_shifted = np.roll(spk_fine, shift)
                spk_coarse_perm = spk_shifted.reshape(-1, ds).max(axis=1)
                spike_idx_perm = np.where(spk_coarse_perm > 0)[0]
                
                if len(spike_idx_perm) > 0:
                    perm_phases = lfp_phase[spike_idx_perm]
                    null_plv[perm] = np.abs(np.exp(1j * perm_phases).mean())
            
            # P-value: proportion of null >= observed
            plv_pval[s, j] = (np.sum(null_plv >= plv[s, j]) + 1) / (n_permutations + 1)
    
    return plv, plv_pval, preferred_phase


# =============================================================================
# SFC Computation (Single Trial)
# =============================================================================
def compute_sfc_single(lfp, spikes, freqs, fs, window_sec, n_permutations=500, seed=42):
    """
    Compute SFC with circular shift permutation test for single-trial data.
    
    Parameters
    ----------
    lfp : (T,) array
    spikes : (S, T_fine) array
    freqs : (B,) array
    fs : float
    window_sec : float
    n_permutations : int
    seed : int
    
    Returns
    -------
    sfc : (S, B) array
        Coherence values (0-1)
    sfc_pval : (S, B) array
        Permutation test p-values
    """
    T = len(lfp)
    S, T_fine = spikes.shape
    B = len(freqs)
    
    ds = T_fine // T
    nperseg = int(window_sec * fs)
    
    # Downsample spikes to LFP resolution
    spikes_ds = spikes.reshape(S, T, ds).mean(axis=-1)
    
    sfc = np.zeros((S, B))
    sfc_pval = np.ones((S, B))
    
    def compute_coherence(x, y):
        """Compute coherence between LFP and spike train."""
        f_csd, Pxy = csd(x, y, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
        _, Pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
        _, Pyy = welch(y, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
        
        Pxy_interp = np.interp(freqs, f_csd, Pxy)
        Pxx_interp = np.interp(freqs, f_csd, Pxx)
        Pyy_interp = np.interp(freqs, f_csd, Pyy)
        
        denom = np.sqrt(Pxx_interp * Pyy_interp)
        coh = np.zeros(B)
        valid = denom > 1e-10
        coh[valid] = np.abs(Pxy_interp[valid]) / denom[valid]
        return coh
    
    for s in range(S):
        print(f"  SFC: unit {s+1}/{S}")
        
        # Observed SFC
        sfc[s] = compute_coherence(lfp, spikes_ds[s])
        
        # Permutation test
        null_sfc = np.zeros((n_permutations, B))
        rng = np.random.default_rng(seed=seed + s)
        
        for perm in range(n_permutations):
            # Circular shift
            shift = rng.integers(T // 4, 3 * T // 4)
            spikes_perm = np.roll(spikes_ds[s], shift)
            null_sfc[perm] = compute_coherence(lfp, spikes_perm)
        
        # P-value per frequency
        for j in range(B):
            sfc_pval[s, j] = (np.sum(null_sfc[:, j] >= sfc[s, j]) + 1) / (n_permutations + 1)
    
    return sfc, sfc_pval


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Compute PLV and SFC for single-trial data'
    )
    parser.add_argument('--data', type=str, required=True,
                        help='Input path for simulated data')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for results')
    parser.add_argument('--method', type=str, default='both',
                        choices=['plv', 'sfc', 'both'],
                        help='Which method to compute (default: both)')
    parser.add_argument('--n_permutations', type=int, default=500,
                        help='Number of permutations (default: 500)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data}")
    with open(args.data, 'rb') as f:
        data = pickle.load(f)
    
    lfp = data['LFP']
    spikes = data['spikes']
    
    T = len(lfp)
    S, T_fine = spikes.shape
    
    print(f"  LFP: ({T},)")
    print(f"  Spikes: {spikes.shape}")
    print(f"  Frequencies: {len(FREQS_DENSE)} from {FREQS_DENSE[0]} to {FREQS_DENSE[-1]} Hz")
    print(f"  Permutations: {args.n_permutations}")
    
    results = {
        'config': {
            'freqs': FREQS_DENSE,
            'fs': FS,
            'window_sec': WINDOW_SEC,
            'bandwidth': BANDWIDTH,
            'nw_product': NW_PRODUCT,
            'n_permutations': args.n_permutations,
            'seed': args.seed,
        }
    }
    
    # Compute PLV
    if args.method in ['plv', 'both']:
        print("\n" + "="*60)
        print("Computing PLV with permutation test...")
        print("="*60)
        
        plv, plv_pval, plv_phase = compute_plv_single(
            lfp, spikes, FREQS_DENSE, FS, BANDWIDTH,
            n_permutations=args.n_permutations, seed=args.seed
        )
        
        results['plv'] = {
            'values': plv,
            'pval': plv_pval,
            'phase': plv_phase,
        }
        
        print(f"  PLV shape: {plv.shape}")
        print(f"  PLV range: [{plv.min():.4f}, {plv.max():.4f}]")
        print(f"  Significant (p<0.05): {(plv_pval < 0.05).sum()} / {plv_pval.size}")
    
    # Compute SFC
    if args.method in ['sfc', 'both']:
        print("\n" + "="*60)
        print("Computing SFC with permutation test...")
        print("="*60)
        
        sfc, sfc_pval = compute_sfc_single(
            lfp, spikes, FREQS_DENSE, FS, WINDOW_SEC,
            n_permutations=args.n_permutations, seed=args.seed
        )
        
        results['sfc'] = {
            'values': sfc,
            'pval': sfc_pval,
        }
        
        print(f"  SFC shape: {sfc.shape}")
        print(f"  SFC range: [{sfc.min():.4f}, {sfc.max():.4f}]")
        print(f"  Significant (p<0.05): {(sfc_pval < 0.05).sum()} / {sfc_pval.size}")
    
    # Save results
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(results, f)
    
    print("\n" + "="*60)
    print(f"Results saved to {args.output}")
    print("="*60)


if __name__ == '__main__':
    main()
