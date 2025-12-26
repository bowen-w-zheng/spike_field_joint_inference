#!/usr/bin/env python3
"""
CLI wrapper for src/em_ct_single_jax.py

Usage:
    python compute_ctssmt_lfp_only_single.py --input ./data/sim.pkl --output ./results/ctssmt.pkl
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
    parser.add_argument('--freq_min', type=float, default=1.0)
    parser.add_argument('--freq_max', type=float, default=61.0)
    parser.add_argument('--freq_step', type=float, default=2.0)
    parser.add_argument('--window_sec', type=float, default=0.4)
    args = parser.parse_args()

    # Load
    with open(args.input, 'rb') as f:
        data = pickle.load(f)
    
    lfp = data['LFP']
    fs = data.get('fs', 1000.0)
    freqs = np.arange(args.freq_min, args.freq_max, args.freq_step)

    # Spectrogram
    tfr = mne.time_frequency.tfr_array_multitaper(
        lfp[None, None, :], sfreq=fs, freqs=freqs,
        n_cycles=freqs * args.window_sec, time_bandwidth=2, output='complex', zero_mean=False
    ).squeeze()[:, None, :]
    M = int(args.window_sec * fs)
    tfr = derotate_tfr_align_start(tfr, freqs, fs, 1, M)
    tapers, _ = mne.time_frequency.multitaper.dpss_windows(M, 1, Kmax=1)
    Y_cube = (tfr * (2.0 / tapers.sum(axis=1)))[:, :, ::M]

    # Run EM - THIS IS THE ONLY INFERENCE CALL
    result = em_ct_single_jax(Y=jnp.asarray(Y_cube), db=args.window_sec, max_iter=args.max_iter, verbose=True, log_every=100)

    # Save
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump({
            'Z_smooth': np.asarray(result.Z_mean).mean(axis=1),  # (J, K)
            'Z_mean': np.asarray(result.Z_mean),
            'Z_var': np.asarray(result.Z_var),
            'params': {'lam': np.asarray(result.lam), 'sig_v': np.asarray(result.sigv), 'sig_eps': np.asarray(result.sig_eps)},
            'freqs': freqs,
            'window_sec': args.window_sec,
        }, f)
    
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()