#!/usr/bin/env python3
"""
Diagnostic: Test taper count AND window size combinations.
"""
import sys
import numpy as np
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import mne
from src.utils_multitaper import derotate_tfr_align_start


def test_config(lfp, fs, freqs, window_sec, NW):
    """Test a specific configuration."""
    n_tapers = int(2 * NW - 1)
    M_samples = int(window_sec * fs)
    
    print(f"\n{'='*60}")
    print(f"CONFIG: window_sec={window_sec}, NW={NW}, n_tapers={n_tapers}, M_samples={M_samples}")
    print('='*60)
    
    # Step 1: MNE TFR
    tfr_raw = mne.time_frequency.tfr_array_multitaper(
        lfp[None, None, :],
        sfreq=fs,
        freqs=freqs,
        n_cycles=freqs * window_sec,
        time_bandwidth=2 * NW,
        output='complex',
        zero_mean=False,
    )
    print(f"1. MNE tfr_raw shape: {tfr_raw.shape}")
    print(f"   |tfr_raw|: min={np.abs(tfr_raw).min():.4f}, max={np.abs(tfr_raw).max():.4f}")
    
    # Step 2: Reshape to (J, M, T)
    if tfr_raw.ndim == 5:
        tfr = tfr_raw[0, 0, :, :, :].transpose(1, 0, 2)
    else:
        tfr = tfr_raw[0, 0, :, :][:, None, :]
    print(f"2. After reshape: {tfr.shape}")
    print(f"   |tfr|: min={np.abs(tfr).min():.4f}, max={np.abs(tfr).max():.4f}")
    
    J, M, T = tfr.shape
    
    # Step 3: Derotate
    tfr_derot = derotate_tfr_align_start(tfr, freqs, fs, M, M_samples)
    print(f"3. After derotate: {tfr_derot.shape}")
    print(f"   |tfr_derot|: min={np.abs(tfr_derot).min():.4f}, max={np.abs(tfr_derot).max():.4f}")
    
    # Check for NaN/Inf
    if np.any(np.isnan(tfr_derot)) or np.any(np.isinf(tfr_derot)):
        print("   *** WARNING: NaN or Inf detected after derotate! ***")
    
    # Step 4: Taper scaling
    tapers, _ = mne.time_frequency.multitaper.dpss_windows(M_samples, NW, Kmax=n_tapers)
    print(f"4. Tapers shape: {tapers.shape}")
    print(f"   Taper sums: {tapers.sum(axis=1)}")
    scaling = 2.0 / tapers.sum(axis=1)
    print(f"   Scaling factors: {scaling}")
    
    tfr_scaled = tfr_derot * scaling[None, :, None]
    print(f"   |tfr_scaled|: min={np.abs(tfr_scaled).min():.4f}, max={np.abs(tfr_scaled).max():.4f}")
    
    # Check for NaN/Inf
    if np.any(np.isnan(tfr_scaled)) or np.any(np.isinf(tfr_scaled)):
        print("   *** WARNING: NaN or Inf detected after scaling! ***")
    
    # Step 5: Downsample
    Y_cube = tfr_scaled[:, :, ::M_samples]
    print(f"5. Y_cube shape: {Y_cube.shape}")
    print(f"   |Y_cube|: min={np.abs(Y_cube).min():.4f}, max={np.abs(Y_cube).max():.4f}")
    
    # Step 6: Power and roughness
    power = (np.abs(Y_cube)**2).mean(axis=1)
    roughness = np.mean([np.mean(np.abs(np.diff(power[j], n=2))) for j in range(power.shape[0])])
    print(f"6. Power: min={power.min():.4f}, max={power.max():.4f}, mean={power.mean():.4f}")
    print(f"   Roughness: {roughness:.4f}")
    
    return roughness


def main():
    import pickle
    with open('./data/sim.pkl', 'rb') as f:
        data = pickle.load(f)
    
    lfp = data['LFP']
    fs = data.get('fs', 1000.0)
    freqs = np.arange(1.0, 61.0, 2.0)
    
    print(f"LFP: shape={lfp.shape}, range=[{lfp.min():.2f}, {lfp.max():.2f}]")
    
    results = {}
    
    # Test different configurations
    for window_sec in [0.4, 2.0]:
        for NW in [1.0, 2.0]:
            key = f"window={window_sec}, NW={NW}"
            try:
                roughness = test_config(lfp, fs, freqs, window_sec, NW)
                results[key] = roughness
            except Exception as e:
                print(f"ERROR: {e}")
                results[key] = None
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    for key, val in results.items():
        status = "OK" if val is not None and val < 1e6 else "BROKEN"
        print(f"  {key}: roughness={val:.4f} [{status}]" if val else f"  {key}: ERROR")


if __name__ == '__main__':
    main()