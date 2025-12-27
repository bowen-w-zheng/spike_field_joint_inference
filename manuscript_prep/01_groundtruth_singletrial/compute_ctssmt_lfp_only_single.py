#!/usr/bin/env python3
"""
CLI wrapper for src/em_ct_single_jax.py
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

import jax.numpy as jnp
import mne
from src.em_ct_single_jax import em_ct_single_jax
from src.utils_multitaper import derotate_tfr_align_start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--max_iter', type=int, default=500)
    parser.add_argument('--tol', type=float, default=1e-6)
    parser.add_argument('--freq_min', type=float, default=1.0)
    parser.add_argument('--freq_max', type=float, default=61.0)
    parser.add_argument('--freq_step', type=float, default=2.0)
    parser.add_argument('--window_sec', type=float, default=2)
    parser.add_argument('--NW', type=float, default=2.0, help='Time-bandwidth product (n_tapers = 2*NW-1)')
    args = parser.parse_args()

    # Load
    with open(args.input, 'rb') as f:
        data = pickle.load(f)
    lfp = data['LFP']
    fs = data.get('fs', 1000.0)
    freqs = np.arange(args.freq_min, args.freq_max, args.freq_step)

    # Number of tapers
    n_tapers = int(2 * args.NW - 1)
    print(f"Using NW={args.NW}, n_tapers={n_tapers}")

    # Spectrogram
    tfr_raw = mne.time_frequency.tfr_array_multitaper(
        lfp[None, None, :],
        sfreq=fs,
        freqs=freqs,
        n_cycles=freqs * args.window_sec,
        time_bandwidth=2 * args.NW,
        output='complex',
        zero_mean=False,
    )
    
    # MNE output shape: (n_epochs, n_channels, n_freqs, n_times) for 1 taper
    #                   (n_epochs, n_channels, n_tapers, n_freqs, n_times) for multiple tapers
    print(f"tfr_raw shape from MNE: {tfr_raw.shape}")
    
    # Reshape to (J, M, T)
    if tfr_raw.ndim == 5:
        # Multiple tapers: (1, 1, n_tapers, n_freqs, n_times) -> (n_freqs, n_tapers, n_times)
        tfr = tfr_raw[0, 0, :, :, :].transpose(1, 0, 2)  # (J, M, T)
    else:
        # Single taper: (1, 1, n_freqs, n_times) -> (n_freqs, 1, n_times)
        tfr = tfr_raw[0, 0, :, :][:, None, :]  # (J, 1, T)
    
    print(f"tfr shape after reshape: {tfr.shape}")
    
    J, M, T = tfr.shape
    M_samples = int(args.window_sec * fs)
    
    # Derotate
    tfr = derotate_tfr_align_start(tfr, freqs, fs, M, M_samples)

    # Taper scaling
    tapers, _ = mne.time_frequency.multitaper.dpss_windows(M_samples, args.NW, Kmax=n_tapers)
    scaling = 2.0 / tapers[0].sum()
    tfr = tfr * scaling

    # Downsample to blocks
    Y_cube = tfr[:, :, ::M_samples]  # (J, M, K)
    J, M_tapers, K = Y_cube.shape
    print(f"Y_cube shape: {Y_cube.shape} (J={J}, M={M_tapers}, K={K})")

    # Run EM
    result = em_ct_single_jax(
        Y=jnp.asarray(Y_cube),
        db=args.window_sec,
        max_iter=args.max_iter,
        tol=args.tol,
        verbose=True,
        log_every=50,
    )

    # Convert to numpy
    Z_mean = np.asarray(result.Z_mean)
    Z_var = np.asarray(result.Z_var)
    lam = np.asarray(result.lam)
    sigv = np.asarray(result.sigv)
    sig_eps = np.asarray(result.sig_eps)
    ll_hist = np.asarray(result.ll_hist)

    # Z_smooth: average across tapers
    Z_smooth = Z_mean.mean(axis=1)

    # Save
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump({
            'Y_cube': np.asarray(Y_cube),
            'Z_smooth': Z_smooth,
            'Z_mean': Z_mean,
            'Z_var': Z_var,
            'params': {
                'lam': lam,
                'sigv': sigv,
                'sig_eps': sig_eps,
            },
            'll_hist': ll_hist,
            'freqs': freqs,
            'window_sec': args.window_sec,
            'fs': fs,
            'NW': args.NW,
            'n_tapers': n_tapers,
        }, f)

    print(f"\nSaved to {args.output}")
    print(f"  λ range: [{lam.min():.4f}, {lam.max():.4f}]")
    print(f"  σ_ε: {sig_eps.mean():.4f}")
    ll_nonzero = ll_hist[ll_hist != 0]
    if len(ll_nonzero) > 0:
        print(f"  Final LL: {ll_nonzero[-1]:.4e}")


if __name__ == '__main__':
    main()