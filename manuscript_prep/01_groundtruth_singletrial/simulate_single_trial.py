#!/usr/bin/env python3
"""
Single-trial LFP + multi-unit spikes simulation.

Generates ground-truth data for evaluating spike-field coupling methods.

COEFFICIENT LAYOUT (SEPARATED):
  β = [β₀, βR₀..βR_{J-1}, βI₀..βI_{J-1}]

Usage:
    python simulate_single_trial.py --output ./data/sim_single_trial.pkl
"""

from __future__ import annotations
import math
import pickle
import pathlib
import argparse
from dataclasses import dataclass, field, asdict
from typing import Tuple, Optional
import numpy as np


# ========================== Utilities ==========================

def nearest_idx(centres: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Nearest-neighbour indices mapping time points t -> centres."""
    c = np.asarray(centres, float).ravel()
    tt = np.asarray(t, float).ravel()
    ir = np.searchsorted(c, tt, side="left")
    il = np.clip(ir - 1, 0, c.size - 1)
    ir = np.clip(ir, 0, c.size - 1)
    use_r = np.abs(c[ir] - tt) < np.abs(c[il] - tt)
    return np.where(use_r, ir, il).astype(np.int32)


def pad_param(x, J: int, name: str) -> np.ndarray:
    """Pad/extend a parameter array to length J."""
    x = np.asarray(x, float).ravel()
    if x.size == 1:
        return np.full(J, float(x[0]))
    if x.size == J:
        return x
    if x.size < J:
        return np.concatenate([x, np.full(J - x.size, x[-1])])
    raise ValueError(f"{name} has length {x.size}, expected 1 or <= {J}")


def random_beta_with_mask(
    rng: np.random.Generator,
    J: int,
    k_active: int,
    mag_lo: float,
    mag_hi: float,
    b0_mu: float,
    b0_sd: float,
    allowed_idx: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate β with SEPARATED layout: [β₀, βR₀..βR_{J-1}, βI₀..βI_{J-1}]."""
    if allowed_idx is None:
        candidates = np.arange(J, dtype=int)
    else:
        candidates = np.asarray(allowed_idx, dtype=int).ravel()
        if candidates.size == 0:
            raise ValueError("allowed_idx is empty")

    mask = np.zeros(J, dtype=bool)
    active = rng.choice(candidates, size=min(k_active, candidates.size), replace=False)
    mask[active] = True

    b0 = rng.normal(loc=b0_mu, scale=b0_sd)

    betaR = np.zeros(J, dtype=float)
    betaI = np.zeros(J, dtype=float)
    beta_mag = np.zeros(J, dtype=float)
    beta_phase = np.zeros(J, dtype=float)

    for j in active:
        mag = rng.uniform(mag_lo, mag_hi)
        phase = rng.uniform(-np.pi, np.pi)
        beta_mag[j] = mag
        beta_phase[j] = phase
        betaR[j] = mag * np.cos(phase)
        betaI[j] = mag * np.sin(phase)

    beta = np.concatenate([[b0], betaR, betaI])
    return beta, mask, beta_mag, beta_phase


def make_gamma_history(n_lags: int, delta_spk: float, scale: float = 1.5, tau: float = 0.03) -> np.ndarray:
    """Simple negative exponential history (inhibition)."""
    lag_times = (np.arange(1, n_lags + 1) * delta_spk)
    return -scale * np.exp(-lag_times / tau)


# ========================== Configuration ==========================

@dataclass
class SingleTrialSimConfig:
    """Configuration for single-trial simulation."""
    
    # Signal bands
    freqs_hz: np.ndarray = field(default_factory=lambda: np.array([11.0, 19.0, 27.0, 43.0], float))
    freqs_hz_extra: np.ndarray = field(default_factory=lambda: np.array([7.0, 35.0], float))
    
    # Units
    S: int = 5
    k_active: int = 3
    
    # Time grids
    fs: float = 1000.0
    duration_sec: float = 10.0
    delta_spk: float = 0.001
    
    # OU parameters
    half_bw_hz: np.ndarray = field(default_factory=lambda: np.array([0.05, 0.05, 0.05, 0.05], float))
    sigma_v: np.ndarray = field(default_factory=lambda: np.array([4.0, 4.0, 4.0, 4.0], float))
    
    # Observation noise
    sigma_eps: np.ndarray = field(default_factory=lambda: np.array([15.0, 15.0, 15.0, 15.0], float))
    sigma_eps_other: float = 15.0
    
    # Dense frequency grid
    noise_fmax_hz: int = 60
    
    # Spike model
    n_lags: int = 20
    hist_scale: float = 1.5
    hist_tau: float = 0.03
    b0_mu: float = -2.0
    b0_sd: float = 0.4
    beta_mag_lo: float = 0.02
    beta_mag_hi: float = 0.15
    
    # Output
    out_path: str = "./data/sim_single_trial.pkl"


# ========================== Main Simulation ==========================

def simulate_single_trial(cfg: SingleTrialSimConfig, seed: int | None = None) -> dict:
    """Single-trial simulator with ground-truth spike-field coupling."""
    rng = np.random.default_rng(np.random.randint(0, 1_000_000) if seed is None else seed)
    
    # Signal bands
    freqs_c = np.asarray(cfg.freqs_hz, float).ravel()
    freqs_u = np.asarray(cfg.freqs_hz_extra, float).ravel()
    freqs_hz = np.concatenate([freqs_c, freqs_u])
    Jc = freqs_c.size
    J = freqs_hz.size
    S = int(cfg.S)
    
    # Time grids
    fs = float(cfg.fs)
    dt = 1.0 / fs
    T = int(round(float(cfg.duration_sec) * fs))
    time = np.arange(T) * dt
    
    delta_spk = float(cfg.delta_spk)
    T_fine = int(round(float(cfg.duration_sec) / delta_spk))
    t_fine = np.arange(T_fine) * delta_spk
    k_idx = nearest_idx(time, t_fine)
    
    print("Single-trial simulation setup:")
    print(f"  Units: S={S}")
    print(f"  Coupled signal bands (Jc={Jc}): {freqs_c}")
    print(f"  Extra uncoupled signal bands (Ju={freqs_u.size}): {freqs_u}")
    print(f"  Total signal bands (J={J}): {freqs_hz}")
    print(f"  LFP: T={T} samples at fs={fs} Hz")
    print(f"  Spikes: T_fine={T_fine} samples at {1/delta_spk} Hz")
    
    # OU parameters
    half_bw_hz = pad_param(cfg.half_bw_hz, J, "half_bw_hz")
    sigma_v = pad_param(cfg.sigma_v, J, "sigma_v")
    sigma_eps_sig = pad_param(cfg.sigma_eps, J, "sigma_eps")
    
    lam = math.pi * half_bw_hz
    sigma_eps_other = float(cfg.sigma_eps_other)
    
    # Dense frequency grid
    noise_fmax_hz = int(cfg.noise_fmax_hz)
    if noise_fmax_hz >= int(fs // 2):
        raise ValueError(f"noise_fmax_hz={noise_fmax_hz} must be < Nyquist {fs/2:.1f} Hz")
    
    freqs_dense = np.arange(1, noise_fmax_hz + 1, dtype=float)
    F = freqs_dense.size
    
    print(f"  Dense frequency grid: F={F} frequencies (1 to {noise_fmax_hz} Hz)")
    
    # Map dense frequency -> signal band index
    freq_to_j = {int(f): j for j, f in enumerate(freqs_hz.astype(int))}
    dense_to_j = np.array([freq_to_j.get(int(f), -1) for f in freqs_dense], dtype=int)
    is_signal_row = dense_to_j >= 0
    idx_sig = np.where(is_signal_row)[0]
    
    # 1) Generate OU latents
    print("Generating OU latent processes...")
    Z_lat = np.zeros((J, T), dtype=np.complex128)
    sqrt_dt_over_2 = math.sqrt(dt / 2.0)
    
    for j in range(J):
        for n in range(1, T):
            Z_lat[j, n] = (
                (1.0 - lam[j] * dt) * Z_lat[j, n - 1]
                + sigma_v[j] * sqrt_dt_over_2
                * (rng.standard_normal() + 1j * rng.standard_normal())
            )
    
    # 2) Phase-rotated latents for spike predictors
    print("Computing derotated latents for spike predictors...")
    phase = 2.0 * np.pi * freqs_hz[:, None] * t_fine[None, :]
    rot = np.exp(1j * phase).astype(np.complex128)
    
    Z_fine = np.take(Z_lat, k_idx, axis=1)
    Ztil = Z_fine * rot
    Ztil_R = Ztil.real
    Ztil_I = Ztil.imag
    
    # 3) Generate β (SEPARATED layout)
    print("Generating coupling parameters...")
    allowed_idx = np.arange(Jc, dtype=int)
    
    beta_true_list = []
    masks_list = []
    beta_mag_list = []
    beta_phase_list = []
    
    for s in range(S):
        beta, mask, beta_mag, beta_phase = random_beta_with_mask(
            rng, J, cfg.k_active, cfg.beta_mag_lo, cfg.beta_mag_hi,
            cfg.b0_mu, cfg.b0_sd, allowed_idx=allowed_idx,
        )
        beta_true_list.append(beta)
        masks_list.append(mask)
        beta_mag_list.append(beta_mag)
        beta_phase_list.append(beta_phase)
    
    beta_true = np.stack(beta_true_list)
    masks = np.stack(masks_list)
    beta_mag = np.stack(beta_mag_list)
    beta_phase = np.stack(beta_phase_list)
    
    print(f"  beta_true shape: {beta_true.shape} (SEPARATED layout)")
    
    if freqs_u.size > 0:
        assert np.all(~masks[:, Jc:]), "Uncoupled signal bands must have zero coupling mask."
    
    # 4) Spike history kernel
    n_lags = int(cfg.n_lags)
    gamma_true = make_gamma_history(n_lags, delta_spk, cfg.hist_scale, cfg.hist_tau)
    
    # 5) Generate spikes
    print("Generating spikes...")
    spikes = np.zeros((S, T_fine), dtype=np.uint8)
    linpred = np.zeros((S, T_fine), dtype=float)
    
    betaR_all = beta_true[:, 1:1 + J]
    betaI_all = beta_true[:, 1 + J:1 + 2 * J]
    
    for s in range(S):
        b0 = beta_true[s, 0]
        bR = betaR_all[s]
        bI = betaI_all[s]
        
        psi_bands = (bR[:, None] * Ztil_R + bI[:, None] * Ztil_I).sum(axis=0)
        
        h = np.zeros(n_lags, dtype=float)
        for t in range(T_fine):
            hist_term = (h @ gamma_true) if n_lags > 0 else 0.0
            psi = b0 + psi_bands[t] + hist_term
            p = 1.0 / (1.0 + np.exp(-np.clip(psi, -30, 30)))
            y = rng.binomial(1, p)
            spikes[s, t] = y
            linpred[s, t] = psi
            if n_lags > 0:
                h[1:] = h[:-1]
                h[0] = y
    
    total_spikes = spikes.sum(axis=1)
    print(f"  Spikes per unit: {total_spikes}")
    print(f"  Mean firing rate: {spikes.mean() / delta_spk:.2f} Hz")
    
    # 6) Embed into dense grid and add broadband noise
    print("Generating LFP with broadband noise...")
    
    Z_all = (
        rng.standard_normal((F, T)) +
        1j * rng.standard_normal((F, T))
    ) * sigma_eps_other
    
    if idx_sig.size > 0:
        sig_sig = sigma_eps_sig[dense_to_j[idx_sig]]
        noise_sig = (
            rng.standard_normal((idx_sig.size, T)) +
            1j * rng.standard_normal((idx_sig.size, T))
        ) * sig_sig[:, None]
        
        Z_lat_sig = Z_lat[dense_to_j[idx_sig], :]
        Z_all[idx_sig, :] = Z_lat_sig + noise_sig
    
    # 7) Synthesize LFP
    carrier_dense = np.exp(2j * np.pi * freqs_dense[:, None] * time[None, :])
    carrier_sig = np.exp(2j * np.pi * freqs_hz[:, None] * time[None, :])
    
    LFP_clean = np.sum((carrier_sig * Z_lat).real, axis=0)
    LFP = np.sum((carrier_dense * Z_all).real, axis=0)
    
    print("Simulation complete.")
    
    return {
        "time": time, "t_fine": t_fine, "k_idx": k_idx,
        "freqs_hz_coupled": freqs_c, "freqs_hz_extra": freqs_u,
        "freqs_hz": freqs_hz, "freqs_dense": freqs_dense,
        "Z_lat": Z_lat, "Z_all": Z_all,
        "Ztil_R": Ztil_R, "Ztil_I": Ztil_I,
        "beta_true": beta_true, "masks": masks,
        "beta_mag": beta_mag, "beta_phase": beta_phase,
        "gamma_true": gamma_true,
        "spikes": spikes, "linpred": linpred,
        "LFP_clean": LFP_clean, "LFP": LFP,
        "half_bw_hz": half_bw_hz, "sigma_v": sigma_v,
        "sigma_eps_signal": sigma_eps_sig, "sigma_eps_other": sigma_eps_other,
        "lam": lam, "delta_spk": delta_spk, "fs": fs,
        "config": asdict(cfg),
    }


def build_history_design_single(spikes: np.ndarray, n_lags: int = 20) -> np.ndarray:
    """Build spike history design matrix for single-trial."""
    S, T = spikes.shape
    H = np.zeros((S, T, n_lags), dtype=np.float32)
    for s in range(S):
        for lag in range(n_lags):
            if lag + 1 < T:
                H[s, lag+1:, lag] = spikes[s, :T-lag-1]
    return H


# ========================== Main ==========================

def main():
    parser = argparse.ArgumentParser(description='Simulate single-trial spike-field data')
    parser.add_argument('--output', type=str, default='./data/sim_single_trial.pkl',
                        help='Output path')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--duration', type=float, default=10.0, help='Duration (sec)')
    parser.add_argument('--S', type=int, default=5, help='Number of units')
    parser.add_argument('--k_active', type=int, default=3, help='Active bands per unit')
    
    args = parser.parse_args()
    
    cfg = SingleTrialSimConfig(
        duration_sec=args.duration,
        S=args.S,
        k_active=args.k_active,
        out_path=args.output,
    )
    
    data = simulate_single_trial(cfg, seed=args.seed)
    
    # Save
    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(data, f)
    print(f"\nSaved to: {out_path}")
    
    # Print ground truth
    print("\n" + "=" * 60)
    print("GROUND TRUTH COUPLING")
    print("=" * 60)
    
    freqs_hz = data["freqs_hz"]
    freqs_c = data["freqs_hz_coupled"]
    Jc = len(freqs_c)
    masks = data["masks"]
    beta_mag = data["beta_mag"]
    beta_phase = data["beta_phase"]
    S = masks.shape[0]
    
    for s in range(S):
        print(f"\nUnit {s}:")
        for j, freq in enumerate(freqs_hz):
            tag = "(couplable)" if j < Jc else "(signal-only)"
            if masks[s, j]:
                print(f"  {freq:.0f} Hz {tag}: |β|={beta_mag[s, j]:.4f}, φ={np.degrees(beta_phase[s, j]):.1f}°")
            else:
                print(f"  {freq:.0f} Hz {tag}: no coupling")


if __name__ == "__main__":
    main()
