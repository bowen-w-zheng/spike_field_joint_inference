# src/joint_inference_core_trial_fast.py
# Simple two-pass hierarchical KF refresh
from __future__ import annotations
from typing import Optional, Sequence, Tuple, Any
from dataclasses import dataclass
import numpy as np

from src.params import OUParams
from src.joint_inference_core import joint_kf_rts_moments
from src.utils_common import separated_to_interleaved


@dataclass
class TrialMoments:
    """
    Per-trial fine-grid moments for Z (= X + D_r).

    Shapes:
      Z_m_s : (R, T, 2*J*M)
      Z_P_s : (R, T, 2*J*M)
    """
    Z_m_s: np.ndarray
    Z_P_s: np.ndarray
    X_m_s: Optional[np.ndarray] = None
    X_P_s: Optional[np.ndarray] = None
    D_m_s: Optional[np.ndarray] = None
    D_P_s: Optional[np.ndarray] = None

    @property
    def m_s(self) -> np.ndarray:
        return self.Z_m_s

    @property
    def P_s(self) -> np.ndarray:
        return self.Z_P_s


def _pool_lfp_trials(
    Y_trials: np.ndarray,
    sig_eps_trials: np.ndarray,
    eps: float = 1e-20
) -> Tuple[np.ndarray, np.ndarray]:
    var = np.asarray(sig_eps_trials, float) ** 2
    w = 1.0 / np.maximum(var, eps)
    wsum = w.sum(axis=0)
    Ynum = (w[..., None] * Y_trials).sum(axis=0)
    Y_pool = Ynum / np.maximum(wsum[..., None], eps)
    sig_pool = np.sqrt(1.0 / np.maximum(wsum, eps))
    return Y_pool, sig_pool


def _pool_spike_pg_exact(
    spikes_SRT: np.ndarray,
    omega_SRT: np.ndarray,
    H_SRTL: Optional[np.ndarray],
    gamma_shared: Optional[np.ndarray],
    omega_floor: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    S, R, T = spikes_SRT.shape
    ω = np.maximum(np.asarray(omega_SRT, float), omega_floor)
    κ = np.asarray(spikes_SRT, float) - 0.5
    ωST = np.maximum(ω.sum(axis=1), omega_floor)

    if (H_SRTL is not None) and (gamma_shared is not None):
        H = np.asarray(H_SRTL, float)
        g = np.asarray(gamma_shared, float)
        h_SRT = np.einsum('srtl,sl->srt', H, g)
        hbar_ST = (ω * h_SRT).sum(axis=1) / ωST
    else:
        hbar_ST = np.zeros_like(ωST)

    κ_sum_ST = κ.sum(axis=1)
    κ_eff_prime_ST = κ_sum_ST - ωST * hbar_ST
    spikes_eff_ST = κ_eff_prime_ST + 0.5

    H_eff_STL = np.zeros((S, T, 1), float)
    gamma_eff_SL = np.zeros((S, 1), float)
    return spikes_eff_ST, ωST, H_eff_STL, gamma_eff_SL


def _gamma_default(S: int, L: int) -> np.ndarray:
    return np.zeros((S, L), float)


def joint_kf_rts_moments_trials_fast(
    Y_trials: np.ndarray,
    theta: OUParams,
    delta_spk: float,
    win_sec: float,
    offset_sec: float,
    beta: np.ndarray,
    gamma_shared: Optional[np.ndarray],
    spikes: np.ndarray,
    omega: np.ndarray,
    coupled_bands_idx: Sequence[int],
    freqs_for_phase: Sequence[float],
    sidx: Any,
    H_hist: Optional[np.ndarray],
    *,
    sigma_u: float = 0.0,
    omega_floor: float = 1e-6,
    sig_eps_trials: Optional[np.ndarray] = None,
    pool_lfp_trials: bool = False,
    pool_spike_trials: bool = False,
) -> TrialMoments:
    """
    Trial-aware KF/RTS smoother.

    Returns UNROTATED state Z. Caller must apply rotation for spike regression.
    """
    Y_arr = np.asarray(Y_trials)
    if Y_arr.ndim != 4:
        raise ValueError("Y_trials must have shape (R, J, M, K).")
    R, J, M, K = Y_arr.shape

    if sig_eps_trials is None:
        se = np.asarray(theta.sig_eps, float)
        sig_eps_use = np.broadcast_to(se[None, ...], (R, J, M))
    else:
        sig_eps_use = np.asarray(sig_eps_trials, float)

    if pool_lfp_trials:
        Y_use, sig_pool = _pool_lfp_trials(Y_arr, sig_eps_use)
        theta_use = OUParams(lam=np.asarray(theta.lam, float),
                             sig_v=np.asarray(theta.sig_v, float),
                             sig_eps=np.asarray(sig_pool, float))
        Y_mode = "pooled"
    else:
        Y_mode = "per-trial"

    spikes_arr = np.asarray(spikes, float)
    omega_arr = np.maximum(np.asarray(omega, float), omega_floor)
    if spikes_arr.ndim != 3:
        raise ValueError("spikes must have shape (S, R, T).")
    S, R2, T = spikes_arr.shape
    if R2 != R:
        raise ValueError(f"spikes R={R2} does not match Y_trials R={R}.")

    if pool_spike_trials:
        spikes_eff, omega_eff, H_eff, gamma_eff = _pool_spike_pg_exact(
            spikes_SRT=spikes_arr, omega_SRT=omega_arr, H_SRTL=H_hist,
            gamma_shared=gamma_shared, omega_floor=omega_floor,
        )
        spikes_list = [spikes_eff]
        omega_list = [omega_eff]
        H_list = [H_eff]
        gamma_list = [np.zeros((S, 1), float)]
    else:
        if H_hist is None:
            H_arr = np.zeros((S, R, T, 0), float)
        else:
            H_arr = np.asarray(H_hist, float)
            if H_arr.shape[:3] != (S, R, T):
                raise ValueError(f"H_hist must have shape (S,R,T,L); got {H_arr.shape}")
        spikes_list = [spikes_arr[:, r, :] for r in range(R)]
        omega_list = [omega_arr[:, r, :] for r in range(R)]
        H_list = [H_arr[:, r, :, :] for r in range(R)]
        gamma_eff = np.asarray(gamma_shared, float) if gamma_shared is not None else _gamma_default(S, H_arr.shape[-1])
        gamma_list = [gamma_eff for _ in range(R)]

    beta_use = np.median(beta, axis=1) if beta.ndim == 3 else np.asarray(beta, float)
    beta_use = separated_to_interleaved(beta_use)

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
            beta=beta_use,
            gamma=gamma_shared if pool_spike_trials else gamma_eff,
            spikes=spikes_ST,
            omega=omega_ST,
            coupled_bands_idx=coupled_bands_idx,
            freqs_for_phase=freqs_for_phase,
            sidx=sidx,
            H_hist=H_STL,
            sigma_u=sigma_u,
            omega_floor=omega_floor,
        )
        return np.asarray(mom.m_s), np.asarray(mom.P_s)

    Z_m_list: list[np.ndarray] = []
    Z_P_list: list[np.ndarray] = []

    if Y_mode == "pooled":
        Y_r = np.ascontiguousarray(Y_use)
        theta_r = theta_use
        m_s, P_s = _run_single_trial(Y_r, theta_r, spikes_list[0], omega_list[0], H_list[0])
        Z_m = np.broadcast_to(m_s[None, :, :], (R, m_s.shape[0], m_s.shape[1]))
        Z_P = np.broadcast_to(P_s[None, :, :], (R, P_s.shape[0], P_s.shape[1]))
        Z_m_list.append(Z_m)
        Z_P_list.append(Z_P)
    else:
        for r in range(R):
            Y_r = np.ascontiguousarray(Y_arr[r])
            theta_r = OUParams(
                lam=np.asarray(theta.lam, float),
                sig_v=np.asarray(theta.sig_v, float),
                sig_eps=np.asarray(sig_eps_use[r], float)
            )
            m_s, P_s = _run_single_trial(Y_r, theta_r, spikes_list[r], omega_list[r], H_list[r])
            Z_m_list.append(m_s[None, :, :])
            Z_P_list.append(P_s[None, :, :])

    Z_m_s = np.concatenate(Z_m_list, axis=0)
    Z_P_s = np.concatenate(Z_P_list, axis=0)

    return TrialMoments(Z_m_s=Z_m_s, Z_P_s=Z_P_s)
