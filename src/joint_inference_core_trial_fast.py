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
from src.utils_common import separated_to_interleaved


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


# ─────────────────────────────────────────────────────────────────────────────
# Deviation mode helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_X_at_block_centers(
    X_fine: np.ndarray,      # (T_fine, 2*J*M)
    delta_spk: float,
    win_sec: float,
    offset_sec: float,
    K: int,                  # number of LFP blocks
) -> np.ndarray:
    """Extract X̂ values at LFP block centers for residual computation.

    Returns: (K, 2*J*M) array of X values at block centers
    """
    T_fine = X_fine.shape[0]
    centres_sec = offset_sec + np.arange(K) * win_sec
    t_idx = np.clip(np.round((centres_sec - offset_sec) / delta_spk).astype(int), 0, T_fine - 1)
    return X_fine[t_idx, :]  # (K, 2*J*M)


def _compute_X_tilde_at_spike_times(
    X_fine: np.ndarray,           # (T_fine, 2*J*M)
    freqs_for_phase: np.ndarray,  # (J,)
    delta_spk: float,
    offset_sec: float,
    J: int,
    M: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute rotated, taper-averaged X̃ for spike pseudo-observation subtraction.

    X̃ = e^{iωt} * (1/M) Σ_m X_m

    Returns:
        X_tilde_R: (T_fine, J) - real part of e^{iωt} * X̄
        X_tilde_I: (T_fine, J) - imag part
    """
    T_fine = X_fine.shape[0]

    # Reshape X_fine to (T_fine, J, M, 2) for Re/Im access
    X_reshaped = X_fine.reshape(T_fine, J, M, 2)
    X_re = X_reshaped[..., 0]  # (T_fine, J, M)
    X_im = X_reshaped[..., 1]  # (T_fine, J, M)

    # Taper average
    X_bar_re = X_re.mean(axis=2)  # (T_fine, J)
    X_bar_im = X_im.mean(axis=2)  # (T_fine, J)

    # Rotation: X̃ = e^{iωt} * X_bar
    t = offset_sec + np.arange(T_fine) * delta_spk
    freqs = np.asarray(freqs_for_phase)
    theta = 2.0 * np.pi * freqs[None, :] * t[:, None]  # (T_fine, J)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # e^{iθ}(R + iI) = (R cos θ - I sin θ) + i(R sin θ + I cos θ)
    X_tilde_R = X_bar_re * cos_theta - X_bar_im * sin_theta  # (T_fine, J)
    X_tilde_I = X_bar_re * sin_theta + X_bar_im * cos_theta  # (T_fine, J)

    return X_tilde_R, X_tilde_I


def _compute_beta_X_contribution(
    X_fine: np.ndarray,           # (T_fine, 2*J*M)
    beta_interleaved: np.ndarray, # (S, 1+2*J) interleaved [β₀, βR_0, βI_0, ...]
    freqs_for_phase: np.ndarray,  # (J,)
    delta_spk: float,
    offset_sec: float,
    J: int,
    M: int,
) -> np.ndarray:
    """Compute β·X̃ contribution to subtract from spike pseudo-observations.

    Returns: (S, T_fine) array of β·X̃ values
    """
    S = beta_interleaved.shape[0]
    T_fine = X_fine.shape[0]

    # Get X̃ (rotated, taper-averaged)
    X_tilde_R, X_tilde_I = _compute_X_tilde_at_spike_times(
        X_fine, freqs_for_phase, delta_spk, offset_sec, J, M
    )  # (T_fine, J)

    # Extract β_R and β_I from interleaved format
    # Layout: [β₀, βR_0, βI_0, βR_1, βI_1, ...]
    beta_R_sj = np.zeros((S, J), dtype=float)
    beta_I_sj = np.zeros((S, J), dtype=float)
    for j in range(J):
        beta_R_sj[:, j] = beta_interleaved[:, 1 + 2*j]
        beta_I_sj[:, j] = beta_interleaved[:, 1 + 2*j + 1]

    # β·X̃ for each (s, t): Σ_j (β_R,j * X̃_R,j + β_I,j * X̃_I,j)
    beta_X_contribution = np.einsum('sj,tj->st', beta_R_sj, X_tilde_R) + \
                          np.einsum('sj,tj->st', beta_I_sj, X_tilde_I)

    return beta_X_contribution  # (S, T_fine)

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
    X_fine_estimate: Optional[np.ndarray] = None,  # (T_fine, 2*J*M) from Pass 1
    X_var_estimate: Optional[np.ndarray] = None,   # (T_fine, 2*J*M) variance of X̂
    estimate_deviation: bool = False,              # If True, estimate D (deviation) not Z
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
    beta_use = separated_to_interleaved(beta_use)  # Convert to interleaved for joint_kf_rts_moments

    # === DEVIATION MODE SETUP ===
    # Compute quantities to subtract when estimating D instead of Z
    X_block_JMK = None        # X̂ at LFP block centers (J, M, K)
    spike_psi_offset = None   # β·X̃ contribution to spike pseudo-obs (S, T_fine)

    if estimate_deviation:
        if X_fine_estimate is None:
            raise ValueError("X_fine_estimate required when estimate_deviation=True")

        freqs_arr = np.asarray(freqs_for_phase, dtype=float)

        # 1. For LFP: compute X̂ at block centers
        X_at_blocks = _extract_X_at_block_centers(
            X_fine_estimate, delta_spk, win_sec, offset_sec, K
        )  # (K, 2*J*M)

        # Reshape to (K, J, M, 2) then to (J, M, K) complex for subtraction
        X_block_reshaped = X_at_blocks.reshape(K, J, M, 2)
        X_block_complex = X_block_reshaped[..., 0] + 1j * X_block_reshaped[..., 1]  # (K,J,M)
        X_block_JMK = np.moveaxis(X_block_complex, 0, 2)  # (J, M, K)

        # 2. For spikes: compute β·X̃ to subtract from pseudo-observation
        spike_psi_offset = _compute_beta_X_contribution(
            X_fine_estimate, beta_use, freqs_arr, delta_spk, offset_sec, J, M
        )  # (S, T_fine)

    # Single-trial smoother caller
    def _run_single_trial(
        Y_r: np.ndarray, theta_r: OUParams,
        spikes_ST: np.ndarray, omega_ST: np.ndarray, H_STL: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # In deviation mode, subtract X from LFP
        Y_r_use = Y_r
        if estimate_deviation and X_block_JMK is not None:
            Y_r_use = Y_r - X_block_JMK  # Y_residual = Y - X̂

        mom = joint_kf_rts_moments(
            Y_cube=Y_r_use,
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
            spike_psi_offset=spike_psi_offset,  # β·X̃ to subtract from yspk
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