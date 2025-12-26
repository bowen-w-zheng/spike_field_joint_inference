#!/usr/bin/env python3
"""
Example: Single-Trial Spike-Field Joint Inference
"""
import numpy as np
import sys
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import mne
from src.simulate_single_trial import (
    SingleTrialSimConfig, 
    simulate_single_trial, 
    build_history_design_single
)
from src.run_joint_inference_single_trial import (
    run_joint_inference_single_trial, 
    SingleTrialInferenceConfig
)
from src.utils_multitaper import derotate_tfr_align_start


def main():
    sim_config = SingleTrialSimConfig(
        freqs_hz=np.array([11.0, 27.0, 43.0]),
        freqs_hz_extra=np.array([]),
        S=4,
        k_active=2,
        duration_sec=120.0,
        fs=1000.0,
        delta_spk=0.001,
        half_bw_hz=np.array([0.05, 0.05, 0.05]),
        sigma_v=np.array([4.0, 4.0, 4.0]),
        sigma_eps=np.array([15.0, 15.0, 15.0]),
        b0_mu=-2.0,
        b0_sd=0.4,
        beta_mag_lo=0.02,
        beta_mag_hi=0.15,
    )
    
    print("="*60)
    print("SINGLE-TRIAL SPIKE-FIELD JOINT INFERENCE EXAMPLE")
    print("="*60)
    
    print("\n1. Simulating data...")
    data = simulate_single_trial(sim_config, seed=42)
    
    freqs_inf = np.arange(1, 61, 2)  # Dense inference grid (30 freqs)
    freqs_sim = data['freqs_hz']      # Simulation frequencies (3 freqs)
    J_inf = len(freqs_inf)
    J_sim = len(freqs_sim)
    S = data['spikes'].shape[0]
    
    print(f"   LFP shape: {data['LFP'].shape}")
    print(f"   Spikes shape: {data['spikes'].shape}")
    print(f"   Inference frequencies: {J_inf} bands from {freqs_inf[0]} to {freqs_inf[-1]} Hz")
    print(f"   Simulation frequencies: {freqs_sim}")
    
    print("\n   Ground truth coupling:")
    for s in range(S):
        coupled = []
        for j in range(J_sim):
            if data['masks'][s, j]:
                f = freqs_sim[j]
                mag = data['beta_mag'][s, j]
                phase = np.degrees(data['beta_phase'][s, j])
                coupled.append(f"{f:.0f}Hz (|β|={mag:.3f}, φ={phase:.0f}°)")
        print(f"   Unit {s}: {', '.join(coupled) if coupled else 'none'}")
    
    print("\n2. Computing multitaper spectrogram...")
    lfp = data['LFP']
    fs = 1000.0
    window_sec = 0.4
    NW_product = 2  # M = 2*NW - 1 = 3 tapers
    
    # Compute TFR
    tfr_out = mne.time_frequency.tfr_array_multitaper(
        lfp[None, None, :],
        sfreq=fs,
        freqs=freqs_inf,
        n_cycles=freqs_inf * window_sec,
        time_bandwidth=2 * NW_product,
        output='complex',
        zero_mean=False,
    )
    
    # Handle MNE output shape: (epochs, channels, tapers, freqs, times)
    tfr_out = np.asarray(tfr_out)
    print(f"   Raw TFR from MNE: {tfr_out.shape}")
    
    if tfr_out.ndim == 5:
        # (epochs=1, channels=1, tapers, freqs, times) -> (J, M, T)
        tfr_raw = tfr_out[0, 0]  # (tapers, freqs, times)
        tfr_raw = tfr_raw.transpose(1, 0, 2)  # -> (freqs, tapers, times) = (J, M, T)
    elif tfr_out.ndim == 4:
        # (epochs=1, channels=1, freqs, times) -> (J, 1, T)
        tfr_raw = tfr_out[0, 0, :, None, :]
    else:
        raise ValueError(f"Unexpected TFR shape: {tfr_out.shape}")
    
    J, M_tapers, T = tfr_raw.shape
    print(f"   tfr_raw: {tfr_raw.shape} (J={J}, M={M_tapers}, T={T})")
    
    # Derotate
    block_size = int(round(window_sec * fs))
    tfr = derotate_tfr_align_start(tfr_raw, freqs_inf, fs, 1, block_size)
    
    # Scale by taper normalization
    tapers, _ = mne.time_frequency.multitaper.dpss_windows(block_size, NW_product, Kmax=1)
    scaling_factor = 2.0 / tapers.sum(axis=1)  # (M_tapers,)
    tfr = tfr * scaling_factor[None, :, None]  # Broadcast: (J, M, T)
    
    # Downsample to blocks: (J, M, K)
    Y_cube = tfr[:, :, ::block_size]
    print(f"   Y_cube shape: {Y_cube.shape}")
    
    print("\n3. Building history design matrix...")
    H = build_history_design_single(data['spikes'], n_lags=20)
    print(f"   H shape: {H.shape}")
    print(f"Y_cube: max={np.abs(Y_cube).max():.6e}, mean={np.abs(Y_cube).mean():.6e}")
    print("\n4. Running joint inference...")
    inf_config = SingleTrialInferenceConfig(
        fixed_iter=200,
        n_refreshes=3,
        inner_steps_per_refresh=100,
        use_beta_shrinkage=True,
        use_wald_band_selection=True,
        wald_alpha=0.05,
        verbose_wald=True,
    )
    
    beta, gamma, theta, trace = run_joint_inference_single_trial(
        Y_cube=Y_cube,
        spikes_ST=data['spikes'],
        H_STL=H,
        all_freqs=freqs_inf,
        delta_spk=data['delta_spk'],
        window_sec=window_sec,
        config=inf_config,
    )
    
    print("\n" + "="*60)
    print("RESULTS: Inferred vs Ground Truth")
    print("="*60)
    
    if hasattr(trace, 'wald_significant_mask'):
        n_sig = trace.wald_significant_mask.sum()
        sig_freqs = [f"{freqs_inf[j]:.0f}Hz" for j in range(J_inf) if trace.wald_significant_mask[j]]
        print(f"\nWald test: {n_sig}/{J_inf} bands significant (α=0.05)")
        print(f"Significant bands: {', '.join(sig_freqs) if sig_freqs else 'none'}")
    
    # Map simulation frequencies to inference grid indices
    sim_to_inf = [np.argmin(np.abs(freqs_inf - f)) for f in freqs_sim]
    
    for s in range(S):
        print(f"\nUnit {s}:")
        for j_sim, j_inf in enumerate(sim_to_inf):
            freq = freqs_sim[j_sim]
            
            # Inferred (SEPARATED layout: [β₀, βR₀..βR_{J-1}, βI₀..βI_{J-1}])
            betaR = beta[s, 1 + j_inf]
            betaI = beta[s, 1 + J_inf + j_inf]
            inf_mag = np.sqrt(betaR**2 + betaI**2)
            inf_phase = np.degrees(np.arctan2(betaI, betaR))
            
            # Ground truth
            gt_mag = data['beta_mag'][s, j_sim]
            gt_phase = np.degrees(data['beta_phase'][s, j_sim])
            is_coupled = data['masks'][s, j_sim]
            
            wald_sig = ""
            if hasattr(trace, 'wald_p_values'):
                p = trace.wald_p_values[s, j_inf]
                wald_sig = f" [p={p:.3f}{'*' if p < 0.05 else ''}]"
            
            status = "COUPLED" if is_coupled else "uncoupled"
            print(f"  {freq:.0f} Hz [{status}]{wald_sig}:")
            print(f"    Inferred:     |β|={inf_mag:.4f}, φ={inf_phase:+6.1f}°")
            print(f"    Ground truth: |β|={gt_mag:.4f}, φ={gt_phase:+6.1f}°")
    
    print("\n" + "="*60)
    print("DONE")
    print("="*60)
    
    return beta, gamma, theta, trace, data


if __name__ == '__main__':
    beta, gamma, theta, trace, data = main()