#!/usr/bin/env python3
"""
Compare Raw vs CT-SSMT (NumPy and/or JAX versions).
"""
import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

C = {'raw': '#888888', 'numpy': '#2E86AB', 'jax': '#F18F01'}


def db_scale(power):
    return 10 * np.log10(power / power.mean() + 1e-10)


def compute_roughness(x):
    return np.mean(np.abs(np.diff(x, n=2)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim', type=str, required=True)
    parser.add_argument('--numpy', type=str, required=True, help='NumPy em_ct results')
    parser.add_argument('--jax', type=str, default=None, help='JAX em_ct results (optional)')
    parser.add_argument('--output', type=str, default='./figures/')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load simulation
    with open(args.sim, 'rb') as f:
        sim = pickle.load(f)
    freqs_true = np.asarray(sim['freqs_hz'], float)
    fs = sim.get('fs', 1000.0)
    duration = len(sim['LFP']) / fs

    # Load NumPy results (required)
    with open(args.numpy, 'rb') as f:
        res_np = pickle.load(f)
    
    assert 'Y_cube' in res_np, "Y_cube not found"
    assert 'Z_mean' in res_np, "Z_mean not found"

    Y_cube = res_np['Y_cube']
    Z_np = res_np['Z_mean']
    freqs = res_np['freqs']
    lam_np = res_np['params']['lam']

    J, M, K = Y_cube.shape
    time_blocks = np.arange(K) * (duration / K)

    # Compute powers
    power_raw = (np.abs(Y_cube)**2).mean(axis=1)
    power_np = (np.abs(Z_np)**2).mean(axis=1)

    # Load JAX results (optional)
    power_jax = None
    lam_jax = None
    if args.jax and os.path.exists(args.jax):
        with open(args.jax, 'rb') as f:
            res_jax = pickle.load(f)
        Z_jax = res_jax['Z_mean']
        power_jax = (np.abs(Z_jax)**2).mean(axis=1)
        lam_jax = res_jax['params']['lam']

    # Print lambda comparison
    print("Lambda values:")
    print(f"{'Freq':>6} {'NumPy':>10}", end="")
    if lam_jax is not None:
        print(f" {'JAX':>10}", end="")
    print()
    
    for j in range(J):
        lam_np_val = lam_np[j, 0] if lam_np.ndim > 1 else lam_np[j]
        marker = " <-- SIG" if any(np.abs(freqs[j] - freqs_true) < 1) else ""
        print(f"{freqs[j]:5.1f}Hz {lam_np_val:10.4f}", end="")
        if lam_jax is not None:
            lam_jax_val = lam_jax[j, 0] if lam_jax.ndim > 1 else lam_jax[j]
            print(f" {lam_jax_val:10.4f}", end="")
        print(marker)

    # === Figure: Spectrogram comparison ===
    n_rows = 3 if power_jax is not None else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 3 * n_rows), sharex=True, sharey=True)
    
    vmin, vmax = np.percentile(db_scale(power_raw), [5, 95])
    extent = [0, duration, freqs[0], freqs[-1]]

    # Raw
    axes[0].imshow(db_scale(power_raw), aspect='auto', origin='lower',
                   extent=extent, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_ylabel('Frequency (Hz)')
    axes[0].set_title('Raw (Y_cube)', fontweight='bold')
    for f in freqs_true:
        axes[0].axhline(f, color='white', ls='--', alpha=0.7, lw=1)

    # NumPy
    axes[1].imshow(db_scale(power_np), aspect='auto', origin='lower',
                   extent=extent, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_title('CT-SSMT NumPy (em_ct)', fontweight='bold')
    for f in freqs_true:
        axes[1].axhline(f, color='white', ls='--', alpha=0.7, lw=1)

    # JAX (if available)
    if power_jax is not None:
        im = axes[2].imshow(db_scale(power_jax), aspect='auto', origin='lower',
                           extent=extent, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[2].set_ylabel('Frequency (Hz)')
        axes[2].set_title('CT-SSMT JAX (em_ct_single_jax)', fontweight='bold')
        for f in freqs_true:
            axes[2].axhline(f, color='white', ls='--', alpha=0.7, lw=1)
    else:
        im = axes[1].images[0]

    axes[-1].set_xlabel('Time (s)')
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label='Power (dB)')
    plt.tight_layout()
    plt.savefig(f'{args.output}/spectrogram_comparison.png', dpi=150)
    plt.savefig(f'{args.output}/spectrogram_comparison.pdf', dpi=150)
    plt.close()

    # === Metrics ===
    rough_raw = np.mean([compute_roughness(power_raw[j]) for j in range(J)])
    rough_np = np.mean([compute_roughness(power_np[j]) for j in range(J)])

    idx_sig = [np.argmin(np.abs(freqs - f)) for f in freqs_true]
    idx_noise = [j for j in range(J) if j not in idx_sig]

    def compute_snr(power):
        return 10 * np.log10(power.mean(axis=1)[idx_sig].mean() / 
                            power.mean(axis=1)[idx_noise].mean() + 1e-10)

    snr_raw = compute_snr(power_raw)
    snr_np = compute_snr(power_np)

    print(f"\n{'Method':<15} {'Roughness':>12} {'SNR (dB)':>10}")
    print("-" * 40)
    print(f"{'Raw':<15} {rough_raw:>12.2f} {snr_raw:>10.1f}")
    print(f"{'NumPy':<15} {rough_np:>12.2f} {snr_np:>10.1f}")
    print(f"{'NumPy vs Raw':<15} {100*(1-rough_np/rough_raw):>11.1f}% {snr_np-snr_raw:>+9.1f}")

    if power_jax is not None:
        rough_jax = np.mean([compute_roughness(power_jax[j]) for j in range(J)])
        snr_jax = compute_snr(power_jax)
        print(f"{'JAX':<15} {rough_jax:>12.2f} {snr_jax:>10.1f}")
        print(f"{'JAX vs Raw':<15} {100*(1-rough_jax/rough_raw):>11.1f}% {snr_jax-snr_raw:>+9.1f}")

    print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()