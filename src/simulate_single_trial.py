#!/usr/bin/env python3
"""
Single-trial LFP + multi-unit spikes simulation.

This is the single-trial (R=1) version of simulate_trial_data.py.
Uses EXACTLY the same coefficient layout and data generation logic.

COEFFICIENT LAYOUT (SEPARATED) - SAME AS TRIAL-STRUCTURED:
  β = [β₀, βR₀..βR_{J-1}, βI₀..βI_{J-1}]
where J = total number of signal bands (coupled + uncoupled).

Data shapes (single-trial versions of trial-structured):
  Trial-structured (R trials)  →  Single-trial (R=1)
  ─────────────────────────────────────────────────────
  LFP: (R, T)                  →  LFP: (T,)
  spikes: (R, S, T_fine)       →  spikes: (S, T_fine)
  Z_lat: (R, J, T)             →  Z_lat: (J, T)
  Ztil_R/I: (R, J, T_fine)     →  Ztil_R/I: (J, T_fine)
  
  beta_true: (S, 1+2J)         →  beta_true: (S, 1+2J)  [SAME]
  masks: (S, J)                →  masks: (S, J)         [SAME]
  beta_mag: (S, J)             →  beta_mag: (S, J)      [SAME]
  beta_phase: (S, J)           →  beta_phase: (S, J)    [SAME]
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Tuple, Optional
import numpy as np


# ========================== Utilities ==========================
# These are IDENTICAL to simulate_trial_data.py

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
    """
    Generate β = [b0, βR(0..J-1), βI(0..J-1)] with exactly k_active non-zero 
    complex entries chosen from allowed_idx.
    
    LAYOUT (SEPARATED): [β₀, βR₀..βR_{J-1}, βI₀..βI_{J-1}]
    
    Returns:
        beta: (1+2J,) array with SEPARATED layout
        mask: (J,) boolean array indicating which bands have coupling
        beta_mag: (J,) magnitude |β| for each band
        beta_phase: (J,) phase angle for each band
    """
    if allowed_idx is None:
        candidates = np.arange(J, dtype=int)
    else:
        candidates = np.asarray(allowed_idx, dtype=int).ravel()
        if candidates.size == 0:
            raise ValueError("allowed_idx is empty")

    # Choose active bands from candidates only
    mask = np.zeros(J, dtype=bool)
    active = rng.choice(candidates, size=min(k_active, candidates.size), replace=False)
    mask[active] = True

    # Baseline
    b0 = rng.normal(loc=b0_mu, scale=b0_sd)

    # Separate real/imag parts - SEPARATED layout
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

    # SEPARATED layout: [β₀, βR₀..βR_{J-1}, βI₀..βI_{J-1}]
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
    
    # --- Signal bands ---
    # Bands that CAN couple to spikes (nonzero beta allowed)
    freqs_hz: np.ndarray = field(default_factory=lambda: np.array([11.0, 19.0, 27.0, 43.0], float))
    # Extra bands that are TRUE signals in LFP but NEVER couple to spikes
    freqs_hz_extra: np.ndarray = field(default_factory=lambda: np.array([7.0, 35.0], float))
    
    # Units
    S: int = 5               # Number of units
    k_active: int = 3        # Number of active (coupled) bands per unit
    
    # Time grids
    fs: float = 1000.0
    duration_sec: float = 10.0
    delta_spk: float = 0.001  # 1 ms spike resolution
    
    # OU parameters
    half_bw_hz: np.ndarray = field(default_factory=lambda: np.array([0.05, 0.05, 0.05, 0.05], float))
    sigma_v: np.ndarray = field(default_factory=lambda: np.array([4.0, 4.0, 4.0, 4.0], float))
    
    # Observation noise
    sigma_eps: np.ndarray = field(default_factory=lambda: np.array([15.0, 15.0, 15.0, 15.0], float))
    sigma_eps_other: float = 15.0
    
    # Dense frequency grid for broadband noise
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
    """
    Single-trial simulator - MATCHES trial-structured version for R=1.
    
    Uses EXACTLY the same logic as simulate_trial_data.py but without
    the trial dimension.
    
    Returns dict with:
        LFP: (T,) noisy LFP
        LFP_clean: (T,) clean LFP (signal only)
        spikes: (S, T_fine) spike trains
        Z_lat: (J, T) latent complex processes
        Ztil_R: (J, T_fine) phase-rotated latent (real)
        Ztil_I: (J, T_fine) phase-rotated latent (imag)
        beta_true: (S, 1+2J) SEPARATED layout
        masks: (S, J)
        beta_mag: (S, J)
        beta_phase: (S, J)
        gamma_true: (n_lags,)
    """
    rng = np.random.default_rng(np.random.randint(0, 1_000_000) if seed is None else seed)
    
    # ---------- Signal bands ----------
    freqs_c = np.asarray(cfg.freqs_hz, float).ravel()            # couplable
    freqs_u = np.asarray(cfg.freqs_hz_extra, float).ravel()      # uncoupled-but-signal
    freqs_hz = np.concatenate([freqs_c, freqs_u])                # all signal bands
    Jc = freqs_c.size
    J = freqs_hz.size
    S = int(cfg.S)
    
    # ---------- Time grids ----------
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
    
    # ---------- OU parameters ----------
    half_bw_hz = pad_param(cfg.half_bw_hz, J, "half_bw_hz")
    sigma_v = pad_param(cfg.sigma_v, J, "sigma_v")
    sigma_eps_sig = pad_param(cfg.sigma_eps, J, "sigma_eps")
    
    lam = math.pi * half_bw_hz
    sigma_eps_other = float(cfg.sigma_eps_other)
    
    # ---------- Dense frequency grid ----------
    noise_fmax_hz = int(cfg.noise_fmax_hz)
    if noise_fmax_hz >= int(fs // 2):
        raise ValueError(f"noise_fmax_hz={noise_fmax_hz} must be < Nyquist {fs/2:.1f} Hz")
    
    freqs_dense = np.arange(1, noise_fmax_hz + 1, dtype=float)
    F = freqs_dense.size
    
    # Map dense frequency index -> signal band index
    freq_to_j = {int(f): j for j, f in enumerate(freqs_hz.astype(int))}
    dense_to_j = np.array([freq_to_j.get(int(f), -1) for f in freqs_dense], dtype=int)
    is_signal_row = dense_to_j >= 0
    idx_sig = np.where(is_signal_row)[0]
    
    # ---------- 1) OU latent processes (no trial structure) ----------
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
    
    # ---------- 2) Spike predictors: derotate latents on fine grid ----------
    print("Computing derotated latents for spike predictors...")
    phase = 2.0 * np.pi * freqs_hz[:, None] * t_fine[None, :]  # (J, T_fine)
    rot = np.exp(1j * phase).astype(np.complex128)
    
    Z_fine = np.take(Z_lat, k_idx, axis=1)  # (J, T_fine)
    Ztil = Z_fine * rot
    Ztil_R = Ztil.real  # (J, T_fine)
    Ztil_I = Ztil.imag  # (J, T_fine)
    
    # ---------- 3) Generate β (ONLY coupled subset eligible) ----------
    print("Generating coupling parameters...")
    allowed_idx = np.arange(Jc, dtype=int)  # only first Jc bands are couplable
    
    beta_true_list = []
    masks_list = []
    beta_mag_list = []
    beta_phase_list = []
    
    for s in range(S):
        beta, mask, beta_mag, beta_phase = random_beta_with_mask(
            rng,
            J,
            cfg.k_active,
            cfg.beta_mag_lo,
            cfg.beta_mag_hi,
            cfg.b0_mu,
            cfg.b0_sd,
            allowed_idx=allowed_idx,
        )
        beta_true_list.append(beta)
        masks_list.append(mask)
        beta_mag_list.append(beta_mag)
        beta_phase_list.append(beta_phase)
    
    beta_true = np.stack(beta_true_list)      # (S, 1+2J)
    masks = np.stack(masks_list)              # (S, J)
    beta_mag = np.stack(beta_mag_list)        # (S, J)
    beta_phase = np.stack(beta_phase_list)    # (S, J)
    
    print(f"  beta_true shape: {beta_true.shape} (SEPARATED layout)")
    
    # Sanity check
    if freqs_u.size > 0:
        assert np.all(~masks[:, Jc:]), "Uncoupled signal bands must have zero coupling mask."
    
    # ---------- 4) Spike history kernel ----------
    n_lags = int(cfg.n_lags)
    gamma_true = make_gamma_history(n_lags, delta_spk, cfg.hist_scale, cfg.hist_tau)
    
    # ---------- 5) Generate spikes ----------
    print("Generating spikes...")
    spikes = np.zeros((S, T_fine), dtype=np.uint8)
    linpred = np.zeros((S, T_fine), dtype=float)
    
    # Extract real/imag from SEPARATED layout
    betaR_all = beta_true[:, 1:1 + J]          # (S, J)
    betaI_all = beta_true[:, 1 + J:1 + 2 * J]  # (S, J)
    
    for s in range(S):
        b0 = beta_true[s, 0]
        bR = betaR_all[s]  # (J,)
        bI = betaI_all[s]  # (J,)
        
        # Linear predictor from all bands (uncoupled have bR=bI=0)
        psi_bands = (bR[:, None] * Ztil_R + bI[:, None] * Ztil_I).sum(axis=0)  # (T_fine,)
        
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
    
    # ---------- 6) Embed into dense grid and add broadband noise ----------
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
    
    # ---------- 7) Synthesize LFP ----------
    carrier_dense = np.exp(2j * np.pi * freqs_dense[:, None] * time[None, :])
    carrier_sig = np.exp(2j * np.pi * freqs_hz[:, None] * time[None, :])
    
    LFP_clean = np.sum((carrier_sig * Z_lat).real, axis=0)  # (T,)
    LFP = np.sum((carrier_dense * Z_all).real, axis=0)      # (T,)
    
    print("Simulation complete.")
    
    return {
        # Time grids
        "time": time,
        "t_fine": t_fine,
        "k_idx": k_idx,
        
        # Frequencies
        "freqs_hz_coupled": freqs_c,     # (Jc,)
        "freqs_hz_extra": freqs_u,       # (Ju,)
        "freqs_hz": freqs_hz,            # (J,)
        "freqs_dense": freqs_dense,      # (F,)
        
        # Latent processes - (J, T) not (R, J, T)
        "Z_lat": Z_lat,        # (J, T)
        "Z_all": Z_all,        # (F, T)
        
        # Derotated latents - (J, T_fine) not (R, J, T_fine)
        "Ztil_R": Ztil_R,      # (J, T_fine)
        "Ztil_I": Ztil_I,      # (J, T_fine)
        
        # Coupling parameters - SEPARATED layout (SAME as trial-structured)
        "beta_true": beta_true,    # (S, 1+2J)
        "masks": masks,            # (S, J)
        "beta_mag": beta_mag,      # (S, J)
        "beta_phase": beta_phase,  # (S, J)
        "gamma_true": gamma_true,  # (n_lags,)
        
        # Spikes - (S, T_fine) not (R, S, T_fine)
        "spikes": spikes,          # (S, T_fine)
        "linpred": linpred,        # (S, T_fine)
        
        # LFP - (T,) not (R, T)
        "LFP_clean": LFP_clean,    # (T,)
        "LFP": LFP,                # (T,)
        
        # Parameters
        "half_bw_hz": half_bw_hz,
        "sigma_v": sigma_v,
        "sigma_eps_signal": sigma_eps_sig,
        "sigma_eps_other": sigma_eps_other,
        "lam": lam,
        
        # Config
        "delta_spk": delta_spk,
        "fs": fs,
        "config": cfg,
    }


def build_history_design_single(spikes: np.ndarray, n_lags: int = 20) -> np.ndarray:
    """
    Build spike history design matrix for single-trial.
    
    Parameters
    ----------
    spikes : (S, T) spike trains
    n_lags : number of history lags
    
    Returns
    -------
    H : (S, T, n_lags) history design matrix
    """
    S, T = spikes.shape
    H = np.zeros((S, T, n_lags), dtype=np.float32)
    
    for s in range(S):
        for lag in range(n_lags):
            if lag + 1 < T:
                H[s, lag+1:, lag] = spikes[s, :T-lag-1]
    
    return H


# ========================== Script Entry ==========================

if __name__ == "__main__":
    import pickle
    import pathlib
    
    cfg = SingleTrialSimConfig()
    data = simulate_single_trial(cfg, seed=None)
    
    # Save
    out_path = pathlib.Path(cfg.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(data, f)
    print(f"\nSaved single-trial dataset to: {out_path}")
    
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
    
    # Verification
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    
    print(f"beta_true layout: SEPARATED [β₀, βR₀..βR_{{J-1}}, βI₀..βI_{{J-1}}]")
    print(f"beta_true shape: {data['beta_true'].shape}")
    print(f"  Index 0: β₀ (intercept)")
    print(f"  Index 1:{1+len(freqs_hz)}: βR (real parts)")
    print(f"  Index {1+len(freqs_hz)}:{1+2*len(freqs_hz)}: βI (imag parts)")
