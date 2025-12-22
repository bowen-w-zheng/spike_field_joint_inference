# src/joint_inference_core_trial_fast.py
# FIXED: Removed _demodulate_Z_for_spikes which was causing the rotation bug
from __future__ import annotations
from typing import Optional, Sequence, Tuple, Any
from dataclasses import dataclass
import numpy as np

# JAX-friendly imports (kept optional in case single-trial core already handles JAX)
import jax
import jax.numpy as jnp

from src.params import OUParams
from src.joint_inference_core import joint_kf_rts_moments  # single-trial smoother


# ─────────────────────────────────────────────────────────────────────────────
# Return container (keeps backward-compat .m_s/.P_s pointing to Z_m_s/Z_P_s)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrialMoments:
    """
    Per-trial fine-grid moments for Z (= X + D_r), with optional components.

    Shapes:
      Z_m_s : (R, T, 2*J*M)
      Z_P_s : (R, T, 2*J*M)
      X_m_s : (T, 2*J*M)        # optional, shared across trials
      X_P_s : (T, 2*J*M)        # optional
      D_m_s : (R, T, 2*J*M)     # optional, per trial
      D_P_s : (R, T, 2*J*M)     # optional
    """
    Z_m_s: np.ndarray
    Z_P_s: np.ndarray
    X_m_s: Optional[np.ndarray] = None
    X_P_s: Optional[np.ndarray] = None
    D_m_s: Optional[np.ndarray] = None
    D_P_s: Optional[np.ndarray] = None

    # Backward-compat aliases used by existing runner code
    @property
    def m_s(self) -> np.ndarray:
        return self.Z_m_s

    @property
    def P_s(self) -> np.ndarray:
        return self.Z_P_s


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _pool_lfp_trials(
    Y_trials: np.ndarray,
    sig_eps_trials: np.ndarray,
    eps: float = 1e-20
) -> Tuple[np.ndarray, np.ndarray]:
    var = np.asarray(sig_eps_trials, float) ** 2
    w = 1.0 / np.maximum(var, eps)                      # (R,J,M)
    wsum = w.sum(axis=0)                                # (J,M)
    Ynum = (w[..., None] * Y_trials).sum(axis=0)        # (J,M,K)
    Y_pool = Ynum / np.maximum(wsum[..., None], eps)
    sig_pool = np.sqrt(1.0 / np.maximum(wsum, eps))     # (J,M)
    return Y_pool, sig_pool


def _pool_spike_pg_exact(
    spikes_SRT: np.ndarray,
    omega_SRT: np.ndarray,
    H_SRTL: Optional[np.ndarray],
    gamma_shared: Optional[np.ndarray],          # (S,L) or None
    omega_floor: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    S, R, T = spikes_SRT.shape
    ω = np.maximum(np.asarray(omega_SRT, float), omega_floor)    # (S,R,T)
    κ = np.asarray(spikes_SRT, float) - 0.5                      # (S,R,T)
    ωST = np.maximum(ω.sum(axis=1), omega_floor)                 # (S,T)

    if (H_SRTL is not None) and (gamma_shared is not None):
        H = np.asarray(H_SRTL, float)                            # (S,R,T,L)
        g = np.asarray(gamma_shared, float)                      # (S,L)
        h_SRT = np.einsum('srtl,sl->srt', H, g)                  # (S,R,T)
        hbar_ST = (ω * h_SRT).sum(axis=1) / ωST                  # (S,T)
    else:
        hbar_ST = np.zeros_like(ωST)

    κ_sum_ST = κ.sum(axis=1)                                     # (S,T)
    κ_eff_prime_ST = κ_sum_ST - ωST * hbar_ST                    # (S,T)
    spikes_eff_ST = κ_eff_prime_ST + 0.5                         # (S,T)

    H_eff_STL = np.zeros((S, T, 1), float)
    gamma_eff_SL = np.zeros((S, 1), float)
    return spikes_eff_ST, ωST, H_eff_STL, gamma_eff_SL


def _extract_single_hist(H_hist: Optional[np.ndarray], *, S: int, T: int) -> np.ndarray:
    if H_hist is None:
        return np.zeros((S, T, 0), float)
    arr = np.asarray(H_hist, float)
    if arr.ndim == 3:
        return arr
    if arr.ndim == 4:
        if arr.shape[1] != 1:
            raise ValueError('Expected a single trial in H_hist when pool_spike_trials is False')
        return arr[:, 0, :, :]
    raise ValueError('Unsupported H_hist shape for single-trial extraction')


def _gamma_default(S: int, L: int) -> np.ndarray:
    return np.zeros((S, L), float)


def _separated_to_interleaved(beta_sep: np.ndarray) -> np.ndarray:
    """
    Convert beta from SEPARATED layout to INTERLEAVED layout.
    
    SEPARATED (used by simulation & Gibbs sampler):
        [β₀, βR_0, βR_1, ..., βR_{J-1}, βI_0, βI_1, ..., βI_{J-1}]
    
    INTERLEAVED (expected by joint_kf_rts_moments):
        [β₀, βR_0, βI_0, βR_1, βI_1, ..., βR_{J-1}, βI_{J-1}]
    
    Parameters
    ----------
    beta_sep : (S, 1+2J) array in separated layout
    
    Returns
    -------
    beta_int : (S, 1+2J) array in interleaved layout
    """
    S, P = beta_sep.shape
    J = (P - 1) // 2
    beta_int = np.zeros_like(beta_sep)
    beta_int[:, 0] = beta_sep[:, 0]  # β₀
    for j in range(J):
        beta_int[:, 1 + 2*j] = beta_sep[:, 1 + j]          # βR_j
        beta_int[:, 1 + 2*j + 1] = beta_sep[:, 1 + J + j]  # βI_j
    return beta_int


# ─────────────────────────────────────────────────────────────────────────────
# Main trial-aware refresh (always supports per-trial Z; optional X/D)
# ─────────────────────────────────────────────────────────────────────────────

def joint_kf_rts_moments_trials_fast(
    Y_trials: np.ndarray,
    theta: OUParams,
    delta_spk: float,
    win_sec: float,
    offset_sec: float,
    beta: np.ndarray,                      # (S,P) or (S,R,P) -> reduced to (S,P)
    gamma_shared: Optional[np.ndarray],    # (S,L)
    spikes: np.ndarray,                    # (S,R,T)
    omega: np.ndarray,                     # (S,R,T)
    coupled_bands_idx: Sequence[int],
    freqs_for_phase: Sequence[float],
    sidx: Any,
    H_hist: Optional[np.ndarray],          # (S,R,T,L)
    *,
    sigma_u: float = 0.0,
    omega_floor: float = 1e-6,
    sig_eps_trials: Optional[np.ndarray] = None,  # (R,J,M)
    pool_lfp_trials: bool = False,
    pool_spike_trials: bool = False,
) -> TrialMoments:
    """
    Trial-aware KF/RTS smoother.
    
    IMPORTANT: Returns UNROTATED state Z. The caller must apply e^{+iωt} rotation
    via _rotate_reim_for_spikes to get Z̃ for spike regression.
    
    The internal KF uses equation (17) from the paper:
        h_{n,spk} = (1/M) [ (βR cos θ + βI sin θ) 1_M^T , (-βR sin θ + βI cos θ) 1_M^T ]
    which correctly handles the rotation within the Kalman update.
    """
    # Convert/validate LFP inputs
    Y_arr = np.asarray(Y_trials)
    if Y_arr.ndim != 4:
        raise ValueError("Y_trials must have shape (R, J, M, K).")
    R, J, M, K = Y_arr.shape

    # per-trial obs noise
    if sig_eps_trials is None:
        se = np.asarray(theta.sig_eps, float)                    # (J,M)
        sig_eps_use = np.broadcast_to(se[None, ...], (R, J, M))  # (R,J,M)
    else:
        sig_eps_use = np.asarray(sig_eps_trials, float)          # (R,J,M)

    # optional pooling (LFP)
    if pool_lfp_trials:
        Y_use, sig_pool = _pool_lfp_trials(Y_arr, sig_eps_use)
        theta_use = OUParams(lam=np.asarray(theta.lam, float),
                             sig_v=np.asarray(theta.sig_v, float),
                             sig_eps=np.asarray(sig_pool, float))
        Y_mode = "pooled"    # (J,M,K)
    else:
        Y_mode = "per-trial"  # (R,J,M,K)

    # spikes / ω
    spikes_arr = np.asarray(spikes, float)
    omega_arr = np.maximum(np.asarray(omega, float), omega_floor)
    if spikes_arr.ndim != 3:
        raise ValueError("spikes must have shape (S, R, T).")
    S, R2, T = spikes_arr.shape
    if R2 != R:
        raise ValueError(f"spikes R={R2} does not match Y_trials R={R}.")

    # history designs
    if pool_spike_trials:
        spikes_eff, omega_eff, H_eff, gamma_eff = _pool_spike_pg_exact(
            spikes_SRT=spikes_arr, omega_SRT=omega_arr, H_SRTL=H_hist,
            gamma_shared=gamma_shared, omega_floor=omega_floor,
        )
        spikes_list = [spikes_eff]     # (S,T)
        omega_list  = [omega_eff]      # (S,T)
        H_list      = [H_eff]          # (S,T,L_eff)
        gamma_list  = [np.zeros((S, 1), float)]
        R_eff = 1
    else:
        if H_hist is None:
            H_arr = np.zeros((S, R, T, 0), float)
        else:
            H_arr = np.asarray(H_hist, float)
            if H_arr.shape[:3] != (S, R, T):
                raise ValueError(f"H_hist must have shape (S,R,T,L); got {H_arr.shape}")
        spikes_list = [spikes_arr[:, r, :] for r in range(R)]
        omega_list  = [omega_arr[:, r, :]  for r in range(R)]
        H_list      = [H_arr[:, r, :, :]   for r in range(R)]
        gamma_eff   = np.asarray(gamma_shared, float) if gamma_shared is not None else _gamma_default(S, H_arr.shape[-1])
        gamma_list  = [gamma_eff for _ in range(R)]
        R_eff = R

    # Reduce β if given per trial, then convert to interleaved layout
    beta_use = np.median(beta, axis=1) if beta.ndim == 3 else np.asarray(beta, float)  # (S,P)
    beta_use = _separated_to_interleaved(beta_use)  # Convert to interleaved for joint_kf_rts_moments

    # Single-trial smoother caller
    def _run_single_trial(
        Y_r: np.ndarray, theta_r: OUParams,
        spikes_ST: np.ndarray, omega_ST: np.ndarray, H_STL: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        mom = joint_kf_rts_moments(
            Y_cube=Y_r,
            theta=theta_r,
            delta_spk=delta_spk,
            win_sec=win_sec,
            offset_sec=offset_sec,
            beta=beta_use,               # (S,P)
            gamma=gamma_shared if pool_spike_trials else gamma_eff,
            spikes=spikes_ST,            # (S,T)
            omega=omega_ST,              # (S,T)
            coupled_bands_idx=coupled_bands_idx,
            freqs_for_phase=freqs_for_phase,   # core uses rotation internally via h_spk
            sidx=sidx,
            H_hist=H_STL,                # (S,T,L)
            sigma_u=sigma_u,
            omega_floor=omega_floor,
        )
        return np.asarray(mom.m_s), np.asarray(mom.P_s)  # ensure NumPy outward

    Z_m_list: list[np.ndarray] = []
    Z_P_list: list[np.ndarray] = []

    if Y_mode == "pooled":
        Y_r = np.ascontiguousarray(Y_use)                      # (J,M,K)
        theta_r = theta_use
        m_s, P_s = _run_single_trial(Y_r, theta_r, spikes_list[0], omega_list[0], H_list[0])
        Z_m = np.broadcast_to(m_s[None, :, :], (R, m_s.shape[0], m_s.shape[1]))
        Z_P = np.broadcast_to(P_s[None, :, :], (R, P_s.shape[0], P_s.shape[1]))
        Z_m_list.append(Z_m)
        Z_P_list.append(Z_P)
    else:
        for r in range(R):
            Y_r = np.ascontiguousarray(Y_arr[r])               # (J,M,K)
            theta_r = OUParams(
                lam=np.asarray(theta.lam, float),
                sig_v=np.asarray(theta.sig_v, float),
                sig_eps=np.asarray(sig_eps_use[r], float)
            )
            m_s, P_s = _run_single_trial(Y_r, theta_r, spikes_list[r], omega_list[r], H_list[r])
            Z_m_list.append(m_s[None, :, :])   # prepend trial axis
            Z_P_list.append(P_s[None, :, :])

    # Stack per-trial Z moments
    Z_m_s = np.concatenate(Z_m_list, axis=0)    # (R,T,2*J*M)
    Z_P_s = np.concatenate(Z_P_list, axis=0)    # (R,T,2*J*M)

    return TrialMoments(Z_m_s=Z_m_s, Z_P_s=Z_P_s)