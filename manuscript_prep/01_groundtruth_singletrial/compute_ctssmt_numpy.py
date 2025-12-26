#!/usr/bin/env python3
"""
Run NumPy em_ct for comparison with JAX version.
Usage:
    python compute_ctssmt_numpy.py --input ./data/sim.pkl --output ./results/ctssmt_numpy.pkl
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
from src.em_ct import em_ct
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
    parser.add_argument('--window_sec', type=float, default=0.4)
    parser.add_argument('--freeze_lam_iters', type=int, default=50)
    args = parser.parse_args()

    # Load
    with open(args.input, 'rb') as f:
        data = pickle.load(f)
    lfp = data['LFP']
    fs = data.get('fs', 1000.0)
    freqs = np.arange(args.freq_min, args.freq_max, args.freq_step)

    # Spectrogram (same preprocessing as JAX version)
    NW = 1
    tfr = mne.time_frequency.tfr_array_multitaper(
        lfp[None, None, :],
        sfreq=fs,
        freqs=freqs,
        n_cycles=freqs * args.window_sec,
        time_bandwidth=2 * NW,
        output='complex',
        zero_mean=False
    ).squeeze()[:, None, :]  # (J, 1, T)

    M = int(args.window_sec * fs)
    tfr = derotate_tfr_align_start(tfr, freqs, fs, 1, M)

    # Taper scaling
    tapers, _ = mne.time_frequency.multitaper.dpss_windows(M, NW, Kmax=1)
    scaling = 2.0 / tapers.sum(axis=1)
    tfr = tfr * scaling[None, :, None]

    # Downsample to blocks
    Y_cube = tfr[:, :, ::M]  # (J, M, K)
    J, M_tapers, K = Y_cube.shape
    print(f"Y_cube shape: {Y_cube.shape} (J={J}, M={M_tapers}, K={K})")

    # Run NumPy EM
    print(f"\nRunning NumPy em_ct (freeze_lam_iters={args.freeze_lam_iters})...")
    lam, sig_v, sig_eps, ll_hist, xs, Ps = em_ct(
        Y_cube,
        db=args.window_sec,
        max_iter=args.max_iter,
        tol=args.tol,
        verbose=True,
        freeze_lam_iters=args.freeze_lam_iters,
        return_moments=True,
    )

    print(f"\nResults:")
    print(f"  Iterations: {len(ll_hist)}")
    print(f"  λ range: [{lam.min():.4f}, {lam.max():.4f}]")
    print(f"  σ_ε: {sig_eps}")
    print(f"  Final LL: {ll_hist[-1]:.4e}")

    # Print lambda per frequency
    print(f"\nLambda per frequency:")
    freqs_true = np.asarray(data.get('freqs_hz', []))
    for j in range(J):
        lam_val = lam[j, 0] if lam.ndim > 1 else lam[j]
        marker = " <-- SIGNAL" if any(np.abs(freqs[j] - freqs_true) < 1) else ""
        print(f"  {freqs[j]:5.1f} Hz: λ = {lam_val:6.4f}{marker}")

    # Save
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump({
            'Y_cube': np.asarray(Y_cube),
            'Z_smooth': xs.mean(axis=1),  # (J, K)
            'Z_mean': xs,                  # (J, M, K)
            'Z_var': Ps,                   # (J, M, K)
            'params': {
                'lam': lam,
                'sigv': sig_v,
                'sig_eps': sig_eps,
            },
            'll_hist': ll_hist,
            'freqs': freqs,
            'window_sec': args.window_sec,
            'fs': fs,
            'method': 'numpy_em_ct',
        }, f)

    print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()