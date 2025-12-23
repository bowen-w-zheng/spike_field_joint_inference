# run_joint_inference_trials_hier_shrinkage.py
# Trial-aware runner with HIERARCHICAL EM warm start (X + D decomposition)
# 
# WITH BETA SHRINKAGE to fix low-frequency artifacts:
#   shrink[j] = |E[β_j]|² / (|E[β_j]|² + Var(β_j))
#   Attenuates uncertain frequency components before KF refresh.
#
# TWO-PASS HIERARCHICAL KF REFRESH:
#   Pass 1: Pooled observations → X (shared component)
#   Pass 2: Per-trial observations → Z_r (full per-trial latent)  
#   Then: D_r = Z_r - X
#
# This properly separates:
#   - X: dynamics shared across all trials (informed by pooled observations)
#   - D_r: trial-specific deviations (informed by each trial's own observations)
# 
# MEMORY-OPTIMIZED VERSION:
# - Only keeps latest X_fine, D_fine, latent (not every refresh)
# - Thins beta/gamma trace during accumulation
# - Forces garbage collection after each refresh

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple, Dict, Any
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import sys
import os
import gc

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.params import OUParams
from src.utils_joint import Trace
from src.state_index import StateIndex
from tqdm.auto import tqdm

from src.joint_inference_core_trial_fast import joint_kf_rts_moments_trials_fast
from src.beta_sampler_trials_jax_test import TrialBetaConfig, gibbs_update_beta_trials_shared_Xrtp_vectorized
from src.polyagamma_jax import sample_pg_saddle_single


# ========================= JAX helpers =========================

@jax.jit
def _sample_omega_pg_batch(key, psi: jnp.ndarray, omega_floor: float) -> jnp.ndarray:
    N = psi.shape[0]
    keys = jr.split(key, N)
    omega = jax.vmap(lambda k, z: sample_pg_saddle_single(k, 1.0, z))(keys, psi)
    return jnp.maximum(omega, omega_floor)


@jax.jit
def _compute_psi_all(
    X_RTP: jnp.ndarray,
    beta_S: jnp.ndarray,
    gamma_SRL: jnp.ndarray,
    H_SRTL: jnp.ndarray
) -> jnp.ndarray:
    base_SRT = jnp.einsum('rtp,sp->srt', X_RTP, beta_S)
    hist_SRT = jnp.einsum('srtk,srk->srt', H_SRTL, gamma_SRL)
    return base_SRT + hist_SRT


@jax.jit
def _sample_omega_pg_matrix(
    key: "jr.KeyArray",
    psi_SRT: jnp.ndarray,
    omega_floor: float,
) -> jnp.ndarray:
    S, R, T = psi_SRT.shape
    keys = jr.split(key, S * R)
    omega_flat = jax.vmap(
        lambda k, psi: _sample_omega_pg_batch(k, psi, omega_floor)
    )(keys, psi_SRT.reshape(-1, T))
    return omega_flat.reshape(S, R, T)


# ========================= Standardization =========================

def _standardize_latents(
    lat_reim_RTP: np.ndarray,
    var_reim_RTP: np.ndarray,
    min_std: float = 0.01,
    scale_factors: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize each frequency band to have approximately unit std."""
    R, T, twoJ = lat_reim_RTP.shape
    
    lat_std = np.zeros_like(lat_reim_RTP)
    var_std = np.zeros_like(var_reim_RTP)
    
    if scale_factors is None:
        scale_factors = np.zeros(twoJ, dtype=float)
        for j in range(twoJ):
            std_j = lat_reim_RTP[:, :, j].std()
            scale_factors[j] = max(std_j, min_std)
    
    for j in range(twoJ):
        lat_std[:, :, j] = lat_reim_RTP[:, :, j] / scale_factors[j]
        var_std[:, :, j] = var_reim_RTP[:, :, j] / (scale_factors[j] ** 2)
    
    return lat_std, var_std, scale_factors


def _rescale_beta(
    beta: np.ndarray,
    scale_factors: np.ndarray,
) -> np.ndarray:
    """Rescale β from standardized to original units."""
    beta_rescaled = beta.copy()
    if beta.ndim == 2:
        beta_rescaled[:, 1:] = beta_rescaled[:, 1:] / scale_factors[None, :]
    elif beta.ndim == 3:
        beta_rescaled[:, :, 1:] = beta_rescaled[:, :, 1:] / scale_factors[None, None, :]
    else:
        raise ValueError(f"Unexpected beta shape: {beta.shape}")
    return beta_rescaled


# ========================= Beta Shrinkage =========================

def _apply_beta_shrinkage(
    beta_samples: np.ndarray,
    burn_in_frac: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply empirical Bayes shrinkage to β estimates.
    
    shrink[j] = |E[β_j]|² / (|E[β_j]|² + Var(β_j))
    
    This attenuates uncertain frequency components (high variance relative to mean).
    Frequencies with strong signal have shrink ≈ 1, noise frequencies have shrink ≈ 0.
    
    Args:
        beta_samples: (n_samples, S, P) where P = 1 + 2*J
        burn_in_frac: fraction of samples to discard as burn-in
        
    Returns:
        beta_shrunk: (S, P) shrunk mean estimates
        shrink_factors: (S, J) shrinkage factors per frequency (for diagnostics)
    """
    n_samples = beta_samples.shape[0]
    burn_in = int(burn_in_frac * n_samples)
    samples = beta_samples[burn_in:]
    
    beta_mean = samples.mean(axis=0)  # (S, P)
    beta_var = samples.var(axis=0)    # (S, P)
    
    S, P = beta_mean.shape
    J = (P - 1) // 2
    
    beta_shrunk = beta_mean.copy()
    shrink_factors = np.ones((S, J), dtype=float)
    
    for j in range(J):
        idx_re = 1 + j
        idx_im = 1 + J + j
        
        # |E[β_j]|² = E[β_R]² + E[β_I]²
        mag_sq = beta_mean[:, idx_re]**2 + beta_mean[:, idx_im]**2
        
        # Var(β_j) = Var(β_R) + Var(β_I)
        var_sum = beta_var[:, idx_re] + beta_var[:, idx_im]
        
        # Shrinkage factor: SNR / (1 + SNR)
        shrink = mag_sq / (mag_sq + var_sum + 1e-12)
        shrink_factors[:, j] = shrink
        
        # Apply shrinkage (preserves phase, attenuates magnitude)
        beta_shrunk[:, idx_re] *= shrink
        beta_shrunk[:, idx_im] *= shrink
    
    return beta_shrunk, shrink_factors


def _print_shrinkage_diagnostics(
    shrink_factors: np.ndarray,
    freqs_hz: Sequence[float],
    max_neurons: int = 2,
) -> None:
    """Print shrinkage diagnostics for each frequency."""
    S, J = shrink_factors.shape
    # print("[SHRINKAGE] Beta shrinkage diagnostics:")
    for s in range(min(S, max_neurons)):
        for j in range(J):
            freq = freqs_hz[j]
            sf = shrink_factors[s, j]
            # status = "SIGNAL" if sf > 0.5 else "(noise)"
            # print(f"  s={s} f={freq:.1f}Hz: shrink={sf:.3f} {status}")


# ========================= Wald Test Band Selection =========================

def _wald_test_band_selection(
    beta_samples: np.ndarray,
    J: int,
    alpha: float = 0.05,
    burn_in_frac: float = 0.5,
    verbose: bool = True,
    freqs_hz: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Wald test to select bands with significant spike coupling.

    Tests H0: β_j = 0 (no coupling) for each band j.

    Args:
        beta_samples: (n_samples, S, P) MCMC samples, P = 1 + 2*J
        J: Number of frequency bands
        alpha: Significance level (default 0.05)
        burn_in_frac: Fraction of samples to discard as burn-in
        verbose: Print diagnostic output
        freqs_hz: Frequency labels for diagnostics

    Returns:
        significant_mask: (J,) bool array - True if band is significantly coupled
        W_stats: (S, J) Wald statistics per neuron/band
        p_values: (S, J) p-values per neuron/band
    """
    from scipy import stats

    n_samples, S, P = beta_samples.shape

    # Apply burn-in
    burn_in = int(burn_in_frac * n_samples)
    samples = beta_samples[burn_in:]
    n_post = samples.shape[0]

    # Extract β_R and β_I for each band
    # Layout: [β₀, βR_0, ..., βR_{J-1}, βI_0, ..., βI_{J-1}]
    beta_R = samples[:, :, 1:1+J]        # (n_post, S, J)
    beta_I = samples[:, :, 1+J:1+2*J]    # (n_post, S, J)

    # Posterior means and variances
    mean_R = beta_R.mean(axis=0)  # (S, J)
    mean_I = beta_I.mean(axis=0)  # (S, J)
    var_R = beta_R.var(axis=0)    # (S, J)
    var_I = beta_I.var(axis=0)    # (S, J)

    # Wald statistic: W = β̂_R²/Var(β_R) + β̂_I²/Var(β_I) ~ χ²(2)
    W_stats = np.zeros((S, J))
    p_values = np.zeros((S, J))

    for s in range(S):
        for j in range(J):
            if var_R[s, j] > 1e-10 and var_I[s, j] > 1e-10:
                W_stats[s, j] = (mean_R[s, j]**2 / var_R[s, j] +
                                mean_I[s, j]**2 / var_I[s, j])
                p_values[s, j] = 1 - stats.chi2.cdf(W_stats[s, j], df=2)
            else:
                W_stats[s, j] = 0.0
                p_values[s, j] = 1.0

    # Band is significant if ANY neuron shows significance
    significant_mask = (p_values < alpha).any(axis=0)  # (J,)

    if verbose:
        n_sig = significant_mask.sum()
        print(f"[WALD] Band selection (α={alpha}, burn-in={burn_in_frac}):")
        print(f"[WALD]   Using {n_post} post-burn-in samples")
        print(f"[WALD]   Significant bands: {n_sig}/{J}")

        if freqs_hz is not None:
            sig_freqs = [f"{freqs_hz[j]:.1f}Hz" for j in range(J) if significant_mask[j]]
            nonsig_freqs = [f"{freqs_hz[j]:.1f}Hz" for j in range(J) if not significant_mask[j]]

            if sig_freqs:
                print(f"[WALD]   Coupled (spike+LFP): {', '.join(sig_freqs)}")
            if nonsig_freqs and len(nonsig_freqs) <= 10:
                print(f"[WALD]   Uncoupled (LFP-only): {', '.join(nonsig_freqs)}")
            elif nonsig_freqs:
                print(f"[WALD]   Uncoupled (LFP-only): {len(nonsig_freqs)} bands")

        # Per-band details (only for significant bands)
        if n_sig > 0 and n_sig <= 10:
            print(f"[WALD]   Significant band details:")
            for j in range(J):
                if significant_mask[j]:
                    min_p = p_values[:, j].min()
                    max_W = W_stats[:, j].max()
                    freq_str = f"{freqs_hz[j]:.1f}Hz" if freqs_hz is not None else f"band {j}"
                    print(f"[WALD]     {freq_str}: W_max={max_W:.2f}, p_min={min_p:.2e}")

    return significant_mask, W_stats, p_values


def _zero_nonsignificant_beta(
    beta: np.ndarray,
    significant_mask: np.ndarray,
    J: int,
) -> np.ndarray:
    """
    Set β coefficients to zero for bands without significant coupling.

    Args:
        beta: (S, P) array where P = 1 + 2*J
        significant_mask: (J,) boolean - True for significant bands
        J: Number of frequency bands

    Returns:
        beta_masked: (S, P) with non-significant bands zeroed
    """
    beta_masked = beta.copy()

    for j in range(J):
        if not significant_mask[j]:
            beta_masked[:, 1 + j] = 0.0        # β_R,j = 0
            beta_masked[:, 1 + J + j] = 0.0    # β_I,j = 0

    return beta_masked


def _select_significant_neurons(
    p_values: np.ndarray,
    alpha: float = 0.05,
    verbose: bool = True,
) -> np.ndarray:
    """
    Select neurons that show significant coupling to at least one band.

    Args:
        p_values: (S, J) p-values from Wald test
        alpha: Significance level
        verbose: Print diagnostics

    Returns:
        neuron_mask: (S,) bool - True for neurons to include in KF refresh
    """
    S, J = p_values.shape

    # Neuron is significant if coupled to ANY band
    neuron_mask = (p_values < alpha).any(axis=1)  # (S,)

    if verbose:
        n_sig = neuron_mask.sum()
        print(f"[WALD] Neuron selection: {n_sig}/{S} neurons have significant coupling")

        if n_sig < S:
            excluded = np.where(~neuron_mask)[0]
            if len(excluded) <= 10:
                print(f"[WALD]   Excluded neurons (no coupling): {excluded.tolist()}")
            else:
                print(f"[WALD]   Excluded neurons: {len(excluded)} total")

    return neuron_mask


# ========================= Prior helpers =========================

def _reduce_mu(mu_raw: Optional[np.ndarray], L: int) -> np.ndarray:
    if mu_raw is None:
        return np.zeros(L, dtype=float)
    mu_arr = np.asarray(mu_raw, float)
    if mu_arr.ndim == 1:
        assert mu_arr.shape[0] == L
        return mu_arr
    if mu_arr.ndim == 2:
        assert mu_arr.shape[1] == L
        return mu_arr.mean(axis=0)
    raise ValueError(f"mu prior has unsupported shape: {mu_arr.shape}")


def _prec_from_sigma(Sigma_raw: Optional[np.ndarray], L: int) -> np.ndarray:
    if Sigma_raw is None:
        return np.eye(L, dtype=float) * 1e-6
    Sigma_arr = np.asarray(Sigma_raw, float)
    if Sigma_arr.ndim == 2:
        base = Sigma_arr
    elif Sigma_arr.ndim == 3:
        base = Sigma_arr.mean(axis=0)
    else:
        raise ValueError(f"Sigma prior has unsupported shape: {Sigma_arr.shape}")
    base = base + 1e-8 * np.eye(L, dtype=float)
    return np.linalg.inv(base)


def _slice_mu(mu_prior, s, S, R):
    if mu_prior is None:
        return None
    mu_arr = np.asarray(mu_prior, float)
    if mu_arr.ndim == 1:
        return mu_arr
    if mu_arr.ndim == 2:
        if mu_arr.shape[0] == S:
            return mu_arr[s]
        if mu_arr.shape[0] == R:
            return mu_arr
    if mu_arr.ndim == 3:
        return mu_arr[s]
    raise ValueError(f"mu prior has unsupported shape: {mu_arr.shape}")


def _slice_sigma(Sigma_prior, s, S, R):
    if Sigma_prior is None:
        return None
    Sigma_arr = np.asarray(Sigma_prior, float)
    if Sigma_arr.ndim == 2:
        return Sigma_arr
    if Sigma_arr.ndim == 3:
        if Sigma_arr.shape[0] == S:
            return Sigma_arr[s]
        if Sigma_arr.shape[0] == R:
            return Sigma_arr
    if Sigma_arr.ndim == 4:
        return Sigma_arr[s]
    raise ValueError(f"Sigma prior has unsupported shape: {Sigma_arr.shape}")


def _prepare_gamma_priors(mu_prior, Sigma_prior, S, R, L, dtype):
    mu_out = np.zeros((S, L), dtype=float)
    prec_out = np.zeros((S, L, L), dtype=float)
    for s in range(S):
        mu_raw = _slice_mu(mu_prior, s, S, R)
        Sig_raw = _slice_sigma(Sigma_prior, s, S, R)
        mu_out[s] = _reduce_mu(mu_raw, L)
        prec_out[s] = _prec_from_sigma(Sig_raw, L)
    return jnp.asarray(prec_out, dtype=dtype), jnp.asarray(mu_out, dtype=dtype)


def _rotate_reim_for_spikes(
    lat_reim_RTP: np.ndarray,
    var_reim_RTP: np.ndarray,
    freqs_hz: Sequence[float],
    delta_spk: float,
    offset_sec: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Modulate complex latents by e^{+i 2π f t} for spike coupling.
    
    This gives Z̃ = e^{+iωt} Z, whose phase equals the instantaneous
    oscillation phase. Thus φ_β directly indicates preferred LFP phase:
      φ_β = 0°   → spikes prefer peak
      φ_β = 180° → spikes prefer trough
    """
    R, T, twoJ = lat_reim_RTP.shape
    J = twoJ // 2

    t = offset_sec + np.arange(T, dtype=float) * float(delta_spk)
    freqs = np.asarray(freqs_hz, float).reshape(1, J)
    phi = 2.0 * np.pi * t[:, None] * freqs

    cos = np.cos(phi)[None, :, :]
    sin = np.sin(phi)[None, :, :]

    Re = lat_reim_RTP[:, :, :J]
    Im = lat_reim_RTP[:, :, J:]
    Vr = var_reim_RTP[:, :, :J]
    Vi = var_reim_RTP[:, :, J:]

    # e^{+iωt}(R + iI) = (R cos - I sin) + i(R sin + I cos)
    Re_rot = Re * cos - Im * sin
    Im_rot = Re * sin + Im * cos
    Vr_rot = Vr * (cos ** 2) + Vi * (sin ** 2)
    Vi_rot = Vr * (sin ** 2) + Vi * (cos ** 2)

    return np.concatenate([Re_rot, Im_rot], axis=2), np.concatenate([Vr_rot, Vi_rot], axis=2)


# ========================= X/D Decomposition =========================

def _decompose_Z_to_XD_precision_weighted(
    Z_mean: np.ndarray,
    Z_var: np.ndarray,
    theta_D: OUParams,
    J: int,
    M: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Precision-weighted decomposition of Z into X (shared) and D (trial-specific).
    
    Each trial gives us: m_r = X + D_r + noise, with total variance Σ_D + P_r
    where Σ_D is the stationary variance of the D process from theta_D.
    
    Optimal (precision-weighted) estimate of X:
        X = [Σ_r (Σ_D + P_r)^{-1}]^{-1} [Σ_r (Σ_D + P_r)^{-1} m_r]
    
    This weights trials with lower uncertainty (smaller P_r) more heavily,
    and accounts for expected D variance from the OU prior.
    
    Args:
        Z_mean: (R, T, D) - KF posterior means per trial
        Z_var: (R, T, D) - KF posterior variances per trial
        theta_D: OU parameters for the D (trial-deviation) process
        J: number of frequency bands
        M: number of tapers
        
    Returns:
        X_mean: (T, D) - shared component mean
        X_var: (T, D) - shared component variance  
        D_mean: (R, T, D) - trial-specific deviation means
        D_var: (R, T, D) - trial-specific deviation variances
    """
    R, T, dim = Z_mean.shape
    
    # Compute stationary variance of D from OU parameters
    # Var(D) = σ²_v / (2λ) for each (j, m) pair
    lam_D = np.asarray(theta_D.lam, float).flatten()
    sigv_D = np.asarray(theta_D.sig_v, float).flatten()
    
    # Handle different shapes of theta_D parameters
    if lam_D.size == J:
        # (J,) -> repeat for each taper
        lam_D = np.repeat(lam_D, M)
        sigv_D = np.repeat(sigv_D, M)
    elif lam_D.size == J * M:
        # (J, M) flattened -> already correct
        pass
    else:
        raise ValueError(f"Unexpected theta_D.lam shape: {theta_D.lam.shape}")
    
    # Stationary variance for each dimension
    # dim = 2 * J * M, layout: [Re_00, Im_00, Re_01, Im_01, ...]
    Sigma_D = np.zeros(dim, dtype=float)
    for jm in range(J * M):
        var_D_jm = sigv_D[jm]**2 / (2 * np.maximum(lam_D[jm], 1e-6))
        # Same variance for real and imaginary parts
        Sigma_D[2 * jm] = var_D_jm
        Sigma_D[2 * jm + 1] = var_D_jm
    
    # Precision-weighted estimation of X
    X_mean = np.zeros((T, dim), dtype=float)
    X_var = np.zeros((T, dim), dtype=float)
    
    for d in range(dim):
        # Total variance for each trial: Var(D) + Var(Z_r|data)
        # Shape: (R, T)
        total_var = Sigma_D[d] + Z_var[:, :, d]  # (R, T)
        
        # Precision (inverse variance) for each trial
        precision = 1.0 / np.maximum(total_var, 1e-10)  # (R, T)
        
        # Total precision across trials at each time point
        total_precision = precision.sum(axis=0)  # (T,)
        
        # Precision-weighted mean: X = Σ_r (prec_r * m_r) / Σ_r prec_r
        X_mean[:, d] = (precision * Z_mean[:, :, d]).sum(axis=0) / total_precision
        
        # Variance of the estimate: Var(X) = 1 / Σ_r prec_r
        X_var[:, d] = 1.0 / total_precision
    
    # D_r = Z_r - X
    D_mean = Z_mean - X_mean[None, :, :]  # (R, T, D)
    
    # Variance of D_r estimate
    # Var(D_r) ≈ Var(Z_r|data) + Var(X) since they're approximately independent
    # (conservative estimate that doesn't underestimate uncertainty)
    D_var = Z_var + X_var[None, :, :]  # (R, T, D)
    
    return X_mean, X_var, D_mean, D_var


# ========================= Config =========================

@dataclass
class InferenceTrialsHierConfig:
    """Configuration for hierarchical trial-aware joint inference."""
    fixed_iter: int = 150
    beta0_window: int = 100
    n_refreshes: int = 3
    inner_steps_per_refresh: int = 100
    omega_floor: float = 1e-3
    sigma_u: float = 0.05
    tau2_intercept: float = 100.0 ** 2
    tau2_gamma: float = 25.0 ** 2
    a0_ard: float = 1e-2
    b0_ard: float = 1e-2
    use_exact_cov: bool = False
    em_kwargs: Dict[str, Any] = field(default_factory=dict)
    key_jax: Optional["jr.KeyArray"] = None
    freeze_beta0: bool = False
    # Standardization
    standardize_latents: bool = True
    min_latent_std: float = 0.01
    # Memory optimization
    trace_thin: int = 2
    # Beta shrinkage
    use_beta_shrinkage: bool = True
    beta_shrinkage_burn_in: float = 0.5
    # Wald test band selection
    use_wald_band_selection: bool = True
    wald_alpha: float = 0.05
    wald_burn_in_frac: float = 0.5
    verbose_wald: bool = True


# ========================= EM adapters =========================

def _theta_from_em_hier(res, J, M, Rtr):
    """Extract θ_X (shared) and θ_D (per-trial residual) from hierarchical EM.

    Returns:
        theta_X: OUParams for shared component X
        theta_D: OUParams for per-trial residual D
        sig_eps_trials: (R, J, M) per-trial observation noise

    Note: theta_Z was removed because Pass 2 now estimates D directly using
    residual observations (Y - X, spikes - β·X̃) instead of estimating Z.
    """
    lam_X = np.asarray(getattr(res, "lam_X"), float).reshape(J, M)
    sigv_X = np.asarray(getattr(res, "sigv_X"), float).reshape(J, M)
    lam_D = np.asarray(getattr(res, "lam_D"), float).reshape(J, M)
    sigv_D = np.asarray(getattr(res, "sigv_D"), float).reshape(J, M)

    if hasattr(res, "sig_eps_jmr"):
        sig_eps_jmr = np.asarray(res.sig_eps_jmr, float)
        if sig_eps_jmr.shape[0] == 1:
            sig_eps_jmr = np.broadcast_to(sig_eps_jmr, (J, sig_eps_jmr.shape[1], sig_eps_jmr.shape[2]))
        sig_eps_trials = np.moveaxis(sig_eps_jmr, 2, 0)  # (R, J, M)
    elif hasattr(res, "sig_eps_mr"):
        sig_eps_mr = np.asarray(res.sig_eps_mr, float)
        sig_eps_trials = np.broadcast_to(sig_eps_mr.T[:, None, :], (Rtr, J, M))
    else:
        sig_eps_trials = np.full((Rtr, J, M), 5.0, float)

    # Pool for shared theta
    var_rm = sig_eps_trials ** 2
    w_rm = 1.0 / np.maximum(var_rm, 1e-20)
    var_pool = 1.0 / np.maximum(w_rm.sum(axis=0), 1e-20)

    theta_X = OUParams(lam=lam_X, sig_v=sigv_X, sig_eps=np.sqrt(var_pool))
    theta_D = OUParams(lam=lam_D, sig_v=sigv_D, sig_eps=np.sqrt(var_pool))

    return theta_X, theta_D, sig_eps_trials


def _extract_XD_from_upsampled_hier(upsampled, J, M):
    """Extract X (shared) and D (per-trial) from hierarchical upsampling."""
    # X: shared process
    Xm = np.asarray(upsampled.X_mean)  # (J, M, Tf) complex
    Xv = np.asarray(upsampled.X_var)   # (J, M, Tf) real
    
    # D: per-trial deviations
    Dm = np.asarray(upsampled.D_mean)  # (R, J, M, Tf) complex
    Dv = np.asarray(upsampled.D_var)   # (R, J, M, Tf) real
    
    J_, M_, Tf = Xm.shape
    R = Dm.shape[0]
    assert (J_, M_) == (J, M)
    
    # Convert to fine state format (T, 2*J*M)
    X_fine = np.zeros((Tf, 2 * J * M), float)
    X_var_fine = np.zeros((Tf, 2 * J * M), float)
    
    for j in range(J):
        for m in range(M):
            col = 2 * (j * M + m)
            X_fine[:, col] = Xm[j, m, :].real
            X_fine[:, col + 1] = Xm[j, m, :].imag
            X_var_fine[:, col] = Xv[j, m, :]
            X_var_fine[:, col + 1] = Xv[j, m, :]
    
    D_fine = np.zeros((R, Tf, 2 * J * M), float)
    D_var_fine = np.zeros((R, Tf, 2 * J * M), float)
    
    for r in range(R):
        for j in range(J):
            for m in range(M):
                col = 2 * (j * M + m)
                D_fine[r, :, col] = Dm[r, j, m, :].real
                D_fine[r, :, col + 1] = Dm[r, j, m, :].imag
                D_var_fine[r, :, col] = Dv[r, j, m, :]
                D_var_fine[r, :, col + 1] = Dv[r, j, m, :]
    
    return X_fine, X_var_fine, D_fine, D_var_fine


def _reim_from_fine_trials(mu_fine_RTP, var_fine_RTP, J, M):
    """Convert fine state to [Re | Im] format averaged over tapers."""
    R, T, _ = mu_fine_RTP.shape
    tmp = mu_fine_RTP.reshape(R, T, J, M, 2)
    vtmp = var_fine_RTP.reshape(R, T, J, M, 2)
    mu_re = tmp[..., 0].mean(axis=3)
    mu_im = tmp[..., 1].mean(axis=3)
    vr_re = vtmp[..., 0].mean(axis=3) / M
    vr_im = vtmp[..., 1].mean(axis=3) / M
    return np.concatenate([mu_re, mu_im], axis=2), np.concatenate([vr_re, vr_im], axis=2)


# ========================= Main runner =========================

def run_joint_inference_trials_hier(
    Y_trials: np.ndarray,
    spikes_SRT: np.ndarray,
    H_SRTL: np.ndarray,
    all_freqs: Sequence[float],
    *,
    delta_spk: float,
    window_sec: float,
    offset_sec: float = 0.0,
    beta_init: Optional[np.ndarray] = None,
    gamma_prior_mu: Optional[np.ndarray] = None,
    gamma_prior_Sigma: Optional[np.ndarray] = None,
    config: Optional[InferenceTrialsHierConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, OUParams, OUParams, Trace]:
    """
    Hierarchical trial-aware joint inference of spike-field coupling.
    
    WITH BETA SHRINKAGE: Applies empirical Bayes shrinkage to β before KF refresh.
    This attenuates uncertain frequency components, preventing low-frequency artifacts.
    
    TWO-PASS HIERARCHICAL KF REFRESH:
      Pass 1: Pooled observations → X (shared component)
      Pass 2: Per-trial observations → Z_r (full per-trial latent)  
      Then: D_r = Z_r - X
    
    Returns β in ORIGINAL (non-standardized) units.
    """
    if config is None:
        config = InferenceTrialsHierConfig()
    key = config.key_jax or jr.PRNGKey(0)

    print("[HIER-JOINT] Starting hierarchical joint inference...")
    print(f"[HIER-JOINT] freeze_beta0 = {config.freeze_beta0}")
    print(f"[HIER-JOINT] standardize_latents = {config.standardize_latents}")
    print(f"[HIER-JOINT] use_beta_shrinkage = {config.use_beta_shrinkage}")
    print(f"[HIER-JOINT] use_wald_band_selection = {config.use_wald_band_selection}")
    print(f"[HIER-JOINT] trace_thin = {config.trace_thin}")
    print(f"[HIER-JOINT] Using TWO-PASS hierarchical KF refresh:")
    print(f"[HIER-JOINT]   Pass 1: Pooled → X (shared)")
    print(f"[HIER-JOINT]   Pass 2: Per-trial → Z_r, then D_r = Z_r - X")

    Rtr, J, M, K = Y_trials.shape
    S, R, T_f = spikes_SRT.shape
    assert Rtr == R
    Rlags = H_SRTL.shape[-1]
    P = 1 + 2 * J
    print(f"[HIER-JOINT] Data dimensions: R={R}, S={S}, J={J}, M={M}, T={T_f}, Rlags={Rlags}")

    # 0) Hierarchical EM warm start
    from src.em_ct_hier_jax import em_ct_hier_jax
    em_kwargs = dict(max_iter=5000, tol=1e-3, sig_eps_init=5.0, log_every=1000)
    if config.em_kwargs:
        em_kwargs.update(config.em_kwargs)
    print("[HIER-JOINT] Running hierarchical EM warm start...")
    res = em_ct_hier_jax(Y_trials=Y_trials, db=window_sec, **em_kwargs)
    theta_X, theta_D, sig_eps_trials = _theta_from_em_hier(res, J=J, M=M, Rtr=Rtr)
    print("[HIER-JOINT] Hierarchical EM complete")
    print(f"[HIER-JOINT] θ_X: λ range [{theta_X.lam.min():.2f}, {theta_X.lam.max():.2f}]")
    print(f"[HIER-JOINT] θ_D: λ range [{theta_D.lam.min():.2f}, {theta_D.lam.max():.2f}]")

    # 1) Upsample X and D to fine grid
    from src.upsample_ct_hier_fine import upsample_ct_hier_fine
    print("[HIER-JOINT] Upsampling EM latents to fine grid...")
    ups = upsample_ct_hier_fine(
        Y_trials=Y_trials, res=res, delta_spk=delta_spk,
        win_sec=window_sec, offset_sec=offset_sec, T_f=None
    )
    X_fine, X_var_fine, D_fine, D_var_fine = _extract_XD_from_upsampled_hier(ups, J=J, M=M)
    
    # Z = X + D for regression (THIS IS THE KEY: treat as single latent)
    Z_fine = X_fine[None, :, :] + D_fine  # (R, T, 2*J*M)
    Z_var_fine = X_var_fine[None, :, :] + D_var_fine
    
    lat_reim_RTP, var_reim_RTP = _reim_from_fine_trials(Z_fine, Z_var_fine, J=J, M=M)
    T0 = min(T_f, lat_reim_RTP.shape[1])
    lat_reim_RTP = lat_reim_RTP[:, :T0]
    var_reim_RTP = var_reim_RTP[:, :T0]
    
    # Store initial X_fine and D_fine (from EM, will be updated after KF refresh)
    X_fine_current = X_fine[:T0].copy()
    D_fine_current = D_fine[:, :T0].copy()
    X_var_fine_current = X_var_fine[:T0].copy()
    D_var_fine_current = D_var_fine[:, :T0].copy()
    
    # For averaging across refreshes
    X_fine_accum = np.zeros_like(X_fine_current)
    D_fine_accum = np.zeros_like(D_fine_current)
    n_accum = 0

    # Modulate for spike coupling (e^{+iωt} convention)
    lat_reim_RTP, var_reim_RTP = _rotate_reim_for_spikes(
        lat_reim_RTP, var_reim_RTP, all_freqs,
        delta_spk=delta_spk, offset_sec=offset_sec
    )

    # Standardize
    if config.standardize_latents:
        lat_reim_RTP, var_reim_RTP, latent_scale_factors = _standardize_latents(
            lat_reim_RTP, var_reim_RTP, min_std=config.min_latent_std
        )
        print(f"[HIER-JOINT] Latent scale factors (min={latent_scale_factors.min():.4f}, "
              f"max={latent_scale_factors.max():.4f})")
    else:
        latent_scale_factors = np.ones(2 * J, dtype=float)
        print("[HIER-JOINT] Latent standardization DISABLED")
    
    latent_scales_jax = jnp.asarray(latent_scale_factors)

    spikes_SRT = spikes_SRT[:, :, :T0]
    H_SRTL = H_SRTL[:, :, :T0, :]
    spikes_SRT_jax = jnp.asarray(spikes_SRT)
    H_SRTL_jax = jnp.asarray(H_SRTL)

    X_RTP = np.concatenate([np.ones((R, T0, 1)), lat_reim_RTP], axis=2)
    X_jax = jnp.asarray(X_RTP)
    V_SRTB = jnp.broadcast_to(jnp.asarray(var_reim_RTP)[None, ...], (S, R, T0, 2 * J))

    # 2) Init β, γ
    beta = np.zeros((S, P), float) if beta_init is None else np.asarray(beta_init, float)
    assert beta.shape == (S, P)
    beta = jnp.asarray(beta)

    if gamma_prior_mu is None:
        gamma = np.zeros((S, R, Rlags), float)
    else:
        gamma = np.asarray(gamma_prior_mu, float)
        gamma = gamma[None, None, :] if gamma.ndim == 1 else (gamma[:, None, :] if gamma.ndim == 2 else gamma)
        gamma = np.broadcast_to(gamma, (S, R, Rlags))
    gamma = jnp.asarray(gamma)

    tau2_lat = jnp.ones((S, 2 * J), dtype=X_jax.dtype)
    Prec_gamma_init, mu_gamma_init = _prepare_gamma_priors(
        gamma_prior_mu, gamma_prior_Sigma, S, R, Rlags, X_jax.dtype
    )

    # 3) Trace - MEMORY OPTIMIZED
    trace = Trace()
    trace.theta_X = [theta_X]
    trace.theta_D = [theta_D]
    trace.X_fine = []
    trace.D_fine = []
    trace.shrinkage_factors = []  # store shrinkage diagnostics
    
    # Track D variance fraction over refreshes
    D_fraction = 0.0

    # 4) Warmup loop
    tb_cfg = TrialBetaConfig(
        omega_floor=config.omega_floor,
        tau2_intercept=config.tau2_intercept,
        tau2_gamma=config.tau2_gamma,
        a0_ard=config.a0_ard,
        b0_ard=config.b0_ard,
    )
    beta_hist, gamma_hist = [], []
    print("[HIER-JOINT] Starting warmup loop...")
    warmup_iter = tqdm(range(config.fixed_iter), desc="Warmup (trial PG-Gibbs)")
    for it in warmup_iter:
        psi_SRT = _compute_psi_all(X_jax, beta, gamma, H_SRTL_jax)
        key_pg, key = jr.split(key)
        omega_SRT = _sample_omega_pg_matrix(key_pg, psi_SRT, config.omega_floor)

        key, beta, gamma_shared, tau2_lat = gibbs_update_beta_trials_shared_Xrtp_vectorized(
            key, X_jax, H_SRTL_jax, spikes_SRT_jax, omega_SRT, V_SRTB,
            Prec_gamma_init, mu_gamma_init, tau2_lat, latent_scales_jax, tb_cfg
        )
        gamma = jnp.broadcast_to(gamma_shared[:, None, :], (S, R, gamma_shared.shape[1]))
        
        if it % config.trace_thin == 0:
            beta_hist.append(np.asarray(beta))
            gamma_hist.append(np.asarray(gamma))

    beta_hist = np.stack(beta_hist, axis=0)
    gamma_hist = np.stack(gamma_hist, axis=0)
    print(f"[HIER-JOINT] Warmup complete ({len(beta_hist)} samples stored)")

    # Freeze β₀ if requested
    if config.freeze_beta0:
        w = min(config.beta0_window // config.trace_thin, len(beta_hist))
        w = max(w, 1)
        beta0_fixed = np.mean(beta_hist[-w:, :, 0], axis=0)
        beta0_fixed_jax = jnp.asarray(beta0_fixed, dtype=beta.dtype)
        beta = beta.at[:, 0].set(beta0_fixed_jax)
        print(f"[HIER-JOINT] Froze β₀ to median of last {w} warmup samples")
    else:
        beta0_fixed_jax = None

    # Bootstrap γ posterior
    mu_g_post = gamma_hist.mean(axis=0)
    Sig_g_post = np.zeros((S, R, Rlags, Rlags), float)
    for s in range(S):
        for r in range(R):
            gh = gamma_hist[:, s, r, :]
            ctr = gh - gh.mean(axis=0, keepdims=True)
            Sg = (ctr.T @ ctr) / max(gh.shape[0] - 1, 1)
            Sig_g_post[s, r] = Sg + 1e-6 * np.eye(Rlags)

    Prec_g_lock = np.zeros_like(Sig_g_post)
    for s in range(S):
        for r in range(R):
            d = np.clip(np.diag(Sig_g_post[s, r]), 1e-10, None)
            Prec_g_lock[s, r] = np.diag(1e6 / d)

    # Store warmup samples to trace
    for i in range(len(beta_hist)):
        trace.beta.append(beta_hist[i])
        trace.gamma.append(gamma_hist[i])
    
    del beta_hist, gamma_hist
    gc.collect()
    print(f"[HIER-JOINT] Warmup trace stored, memory freed")

    # =================================================================
    # WALD TEST FOR BAND SELECTION (after warmup)
    # =================================================================
    if config.use_wald_band_selection:
        print("[HIER-JOINT] Performing Wald test for band selection...")

        # Stack all warmup beta samples
        beta_samples_warmup = np.stack(trace.beta, axis=0)  # (n_warmup, S, P)

        significant_mask, W_stats, p_values = _wald_test_band_selection(
            beta_samples=beta_samples_warmup,
            J=J,
            alpha=config.wald_alpha,
            burn_in_frac=config.wald_burn_in_frac,
            verbose=config.verbose_wald,
            freqs_hz=all_freqs,
        )

        # Store band-level results in trace
        trace.wald_significant_mask = significant_mask
        trace.wald_W_stats = W_stats
        trace.wald_p_values = p_values

        n_significant = significant_mask.sum()
        print(f"[HIER-JOINT] {n_significant}/{J} bands will receive spike updates")

        # Neuron selection: only neurons with coupling to at least one band
        neuron_mask = _select_significant_neurons(
            p_values=p_values,
            alpha=config.wald_alpha,
            verbose=config.verbose_wald,
        )
        trace.wald_neuron_mask = neuron_mask
    else:
        significant_mask = np.ones(J, dtype=bool)  # All bands coupled
        neuron_mask = np.ones(S, dtype=bool)        # All neurons included
        print("[HIER-JOINT] Wald test disabled - all bands/neurons receive spike updates")

    # 5) Refresh passes using NON-HIERARCHICAL KF (treats Z = X + D jointly)
    sidx = StateIndex(J, M)
    print(f"[HIER-JOINT] Starting {config.n_refreshes} refresh passes (non-hierarchical KF)...")
    
    for rr in range(config.n_refreshes):
        print(f"[HIER-JOINT] Refresh {rr + 1}/{config.n_refreshes}")
        inner_beta_hist, inner_gamma_hist = [], []

        inner_iter = tqdm(
            range(config.inner_steps_per_refresh),
            desc=f"Refresh {rr + 1} inner PG steps", leave=False
        )
        for it in inner_iter:
            gamma_samp = np.zeros((S, R, Rlags), float)
            for s in range(S):
                for r in range(R):
                    chol = np.linalg.cholesky(Sig_g_post[s, r])
                    gamma_samp[s, r] = mu_g_post[s, r] + chol @ np.random.randn(Rlags)

            gamma_samp_jax = jnp.asarray(gamma_samp)
            key_pg, key = jr.split(key)
            omega_SRT = _sample_omega_pg_matrix(
                key_pg, _compute_psi_all(X_jax, beta, gamma_samp_jax, H_SRTL_jax),
                config.omega_floor
            )

            key, beta, gamma_shared, tau2_lat = gibbs_update_beta_trials_shared_Xrtp_vectorized(
                key, X_jax, H_SRTL_jax, spikes_SRT_jax, omega_SRT, V_SRTB,
                jnp.asarray(Prec_g_lock), jnp.asarray(gamma_samp), tau2_lat, latent_scales_jax, tb_cfg
            )
            gamma = jnp.broadcast_to(gamma_shared[:, None, :], (S, R, gamma_shared.shape[1]))

            if config.freeze_beta0:
                beta = beta.at[:, 0].set(beta0_fixed_jax)

            if it % config.trace_thin == 0:
                inner_beta_hist.append(np.asarray(beta))
                inner_gamma_hist.append(np.asarray(gamma))

        inner_beta_hist = np.stack(inner_beta_hist, axis=0)
        inner_gamma_hist = np.stack(inner_gamma_hist, axis=0)

        # =================================================================
        # BETA SHRINKAGE
        # =================================================================
        if config.use_beta_shrinkage:
            beta_mean, shrink_factors = _apply_beta_shrinkage(
                inner_beta_hist, burn_in_frac=config.beta_shrinkage_burn_in
            )
            _print_shrinkage_diagnostics(shrink_factors, all_freqs, max_neurons=2)
            trace.shrinkage_factors.append(shrink_factors)
        else:
            beta_mean = np.mean(inner_beta_hist, axis=0)
        
        gamma_shared_for_refresh = np.median(inner_gamma_hist, axis=0).mean(axis=1)
        beta = jnp.asarray(beta_mean)

        # Rescale β to original units for KF refresh
        beta_mean_original = _rescale_beta(beta_mean, latent_scale_factors) if config.standardize_latents else beta_mean

        # =================================================================
        # APPLY WALD MASK: Zero β for non-significant bands
        # =================================================================
        if config.use_wald_band_selection:
            beta_for_kf = _zero_nonsignificant_beta(
                beta_mean_original, significant_mask, J
            )
            if config.verbose_wald and rr == 0:  # Only print on first refresh
                n_active = significant_mask.sum()
                print(f"[HIER-JOINT] KF refresh using {n_active}/{J} coupled bands")
        else:
            beta_for_kf = beta_mean_original

        # =================================================================
        # TWO-PASS HIERARCHICAL KF REFRESH
        # Pass 1: Pooled observations → X (shared component)
        # Pass 2: Residual observations → D_r (per-trial deviation)
        # Then: Z_r = X + D_r
        # =================================================================

        # -----------------------------------------------------------------
        # FILTER: Only include significant neurons in spike observations
        # -----------------------------------------------------------------
        if neuron_mask.all():
            # All neurons significant, no filtering needed
            spikes_for_kf = spikes_SRT
            omega_for_kf = np.asarray(omega_SRT)
            beta_for_kf_filtered = beta_for_kf
            gamma_for_kf = np.asarray(gamma_shared_for_refresh)
            H_hist_for_kf = H_SRTL
        else:
            # Filter to significant neurons only
            sig_neuron_idx = np.where(neuron_mask)[0]
            spikes_for_kf = spikes_SRT[sig_neuron_idx]
            omega_for_kf = np.asarray(omega_SRT)[sig_neuron_idx]
            beta_for_kf_filtered = beta_for_kf[sig_neuron_idx]
            gamma_for_kf = np.asarray(gamma_shared_for_refresh)[sig_neuron_idx]
            H_hist_for_kf = H_SRTL[sig_neuron_idx] if H_SRTL is not None else None
            if rr == 0:
                print(f"[HIER-JOINT] Filtering to {len(sig_neuron_idx)}/{S} significant neurons")

        # -----------------------------------------------------------------
        # PASS 1: Estimate X (shared) using POOLED observations
        # Since E[D_r] = 0, pooling gives optimal estimate of X
        # -----------------------------------------------------------------
        print(f"[HIER-JOINT] Refresh {rr + 1}: Pass 1 - estimating X (pooled)")

        mom_X = joint_kf_rts_moments_trials_fast(
            Y_trials=Y_trials,
            theta=theta_X,  # Use X dynamics (slower, shared)
            delta_spk=delta_spk,
            win_sec=window_sec,
            offset_sec=offset_sec,
            beta=beta_for_kf_filtered,
            gamma_shared=gamma_for_kf,
            spikes=spikes_for_kf,
            omega=omega_for_kf,
            coupled_bands_idx=np.arange(J, dtype=np.int64),
            freqs_for_phase=np.asarray(all_freqs, float),
            sidx=sidx,
            H_hist=H_hist_for_kf,
            sigma_u=config.sigma_u,
            sig_eps_trials=sig_eps_trials,
            pool_lfp_trials=True,   # Pool LFP for X
            pool_spike_trials=True, # Pool spikes for X
        )

        # When pooled, all trials have same estimate - take first
        X_fine_updated = mom_X.m_s[0, :T0, :]   # (T, 2*J*M)
        X_var_updated = mom_X.P_s[0, :T0, :]    # (T, 2*J*M)

        # -----------------------------------------------------------------
        # PASS 2: Estimate D_r (deviation) directly using RESIDUAL observations
        # Residual LFP: Y_r - X̂
        # Residual spikes: yspk - β·X̃
        # Uses θ_D dynamics (faster, trial-specific)
        # -----------------------------------------------------------------
        print(f"[HIER-JOINT] Refresh {rr + 1}: Pass 2 - estimating D_r (deviation mode)")

        mom_D = joint_kf_rts_moments_trials_fast(
            Y_trials=Y_trials,
            theta=theta_D,  # Use D dynamics (faster, trial-specific)
            delta_spk=delta_spk,
            win_sec=window_sec,
            offset_sec=offset_sec,
            beta=beta_for_kf_filtered,
            gamma_shared=gamma_for_kf,
            spikes=spikes_for_kf,
            omega=omega_for_kf,
            coupled_bands_idx=np.arange(J, dtype=np.int64),
            freqs_for_phase=np.asarray(all_freqs, float),
            sidx=sidx,
            H_hist=H_hist_for_kf,
            sigma_u=config.sigma_u,
            sig_eps_trials=sig_eps_trials,
            pool_lfp_trials=False,   # Per-trial LFP residuals
            pool_spike_trials=False, # Per-trial spike residuals
            X_fine_estimate=X_fine_updated,   # X̂ from Pass 1
            X_var_estimate=X_var_updated,     # Var(X̂) from Pass 1
            estimate_deviation=True,          # Enable deviation mode
        )

        # D_r is the direct output (already the residual)
        D_fine_updated = mom_D.m_s[:, :T0, :]  # (R, T, 2*J*M)
        D_var_updated = mom_D.P_s[:, :T0, :]   # (R, T, 2*J*M)

        # Reconstruct Z_r = X + D_r for downstream use
        Z_fine_kf = X_fine_updated[None, :, :] + D_fine_updated  # (R, T, 2*J*M)
        Z_var_kf = X_var_updated[None, :, :] + D_var_updated     # (R, T, 2*J*M)
        
        # Compute and print D contribution statistics
        D_power = np.mean(D_fine_updated**2)
        X_power = np.mean(X_fine_updated**2)
        D_fraction = 100 * D_power / (D_power + X_power) if (D_power + X_power) > 0 else 0
        print(f"[HIER-JOINT] X/D decomposition: D variance fraction = {D_fraction:.1f}%")
        
        # Debug: print cross-trial variability
        Z_trial_std = np.std(Z_fine_kf, axis=0).mean()
        print(f"[HIER-JOINT] Z cross-trial std (mean): {Z_trial_std:.6f}")
        
        # Update current X and D
        X_fine_current = X_fine_updated.copy()
        D_fine_current = D_fine_updated.copy()
        X_var_fine_current = X_var_updated.copy()
        D_var_fine_current = D_var_updated.copy()
        
        # Accumulate for averaging
        X_fine_accum += X_fine_current
        D_fine_accum += D_fine_current
        n_accum += 1
        
        # Rebuild regressors from smoothed Z (per-trial)
        # Use reconstructed Z = X + D from the two passes
        lat_reim_RTP, var_reim_RTP = _reim_from_fine_trials(Z_fine_kf, Z_var_kf, J=J, M=M)
        T0 = min(T0, lat_reim_RTP.shape[1])
        lat_reim_RTP = lat_reim_RTP[:, :T0]
        var_reim_RTP = var_reim_RTP[:, :T0]
        
        # The KF already derotates internally for spike observation, 
        # but the STATE it returns is UNROTATED. We need to rotate for regression.
        lat_reim_RTP, var_reim_RTP = _rotate_reim_for_spikes(
            lat_reim_RTP, var_reim_RTP, all_freqs,
            delta_spk=delta_spk, offset_sec=offset_sec
        )

        # Re-standardize using SAME scale factors
        if config.standardize_latents:
            lat_reim_RTP, var_reim_RTP, _ = _standardize_latents(
                lat_reim_RTP, var_reim_RTP,
                min_std=config.min_latent_std,
                scale_factors=latent_scale_factors
            )

        X_RTP = np.concatenate([np.ones((R, T0, 1)), lat_reim_RTP], axis=2)
        X_jax = jnp.asarray(X_RTP)
        V_SRTB = jnp.broadcast_to(jnp.asarray(var_reim_RTP)[None, ...], (S, R, T0, 2 * J))
        spikes_SRT_jax = jnp.asarray(spikes_SRT[:, :, :T0])
        H_SRTL_jax = jnp.asarray(H_SRTL[:, :, :T0, :])

        # Store to trace
        trace.theta_X.append(theta_X)
        trace.theta_D.append(theta_D)
        
        for i in range(len(inner_beta_hist)):
            trace.beta.append(inner_beta_hist[i])
            trace.gamma.append(inner_gamma_hist[i])

        # Update γ posterior
        mu_g_post = inner_gamma_hist.mean(axis=0)
        Sig_g_post = np.zeros((S, R, Rlags, Rlags), float)
        for s in range(S):
            for r in range(R):
                gh = inner_gamma_hist[:, s, r, :]
                ctr = gh - gh.mean(axis=0, keepdims=True)
                Sg = (ctr.T @ ctr) / max(gh.shape[0] - 1, 1)
                Sig_g_post[s, r] = Sg + 1e-6 * np.eye(Rlags)
        Prec_g_lock = np.zeros_like(Sig_g_post)
        for s in range(S):
            for r in range(R):
                d = np.clip(np.diag(Sig_g_post[s, r]), 1e-10, None)
                Prec_g_lock[s, r] = np.diag(1e6 / d)

        del inner_beta_hist, inner_gamma_hist
        gc.collect()
        
        print(f"[HIER-JOINT] Refresh {rr + 1} complete (trace has {len(trace.beta)} samples)")

    # =================================================================
    # STORE FINAL X_fine and D_fine (updated with spike information!)
    # =================================================================
    trace.X_fine = [X_fine_current]
    trace.D_fine = [D_fine_current]
    trace.X_var_fine = [X_var_fine_current]
    trace.D_var_fine = [D_var_fine_current]
    trace.latent = [lat_reim_RTP]
    
    # Averaged estimates across refreshes
    if n_accum > 0:
        trace.X_fine_avg = X_fine_accum / n_accum
        trace.D_fine_avg = D_fine_accum / n_accum
    else:
        trace.X_fine_avg = X_fine_current
        trace.D_fine_avg = D_fine_current

    # =================================================================
    # RESCALE β to original units
    # =================================================================
    beta_final = np.asarray(beta)
    if config.standardize_latents:
        beta_final = _rescale_beta(beta_final, latent_scale_factors)
        for i in range(len(trace.beta)):
            trace.beta[i] = _rescale_beta(trace.beta[i], latent_scale_factors)
    
    trace.latent_scale_factors = latent_scale_factors
    
    print("[HIER-JOINT] Inference complete")
    print(f"[HIER-JOINT] Total trace samples: {len(trace.beta)}")
    print(f"[HIER-JOINT] Final D variance fraction: {D_fraction:.1f}%")
    if config.use_beta_shrinkage:
        print(f"[HIER-JOINT] Beta shrinkage was ENABLED")

    return beta_final, np.asarray(gamma), theta_X, theta_D, trace