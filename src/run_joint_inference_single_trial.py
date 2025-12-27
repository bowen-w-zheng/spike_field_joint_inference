# run_joint_inference_single_trial.py
# DIRECT COPY of run_joint_inference_trials_hier_shrinkage.py
# ONLY CHANGES: R=1 hardcoded, removed trial loops, shape adjustments
#
# All functions copied VERBATIM from document 7 (trial-structured version)

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

from src.joint_inference_core import joint_kf_rts_moments
from src.beta_sampler_trials_jax_test import TrialBetaConfig, gibbs_update_beta_trials_shared_Xrtp_vectorized
from src.polyagamma_jax import sample_pg_saddle_single

jax.config.update("jax_enable_x64", True)


# ========================= JAX helpers =========================
# VERBATIM from run_joint_inference_trials_hier_shrinkage.py

@jax.jit
def _sample_omega_pg_batch(key, psi: jnp.ndarray, omega_floor: float) -> jnp.ndarray:
    N = psi.shape[0]
    keys = jr.split(key, N)
    omega = jax.vmap(lambda k, z: sample_pg_saddle_single(k, 1.0, z))(keys, psi)
    return jnp.maximum(omega, omega_floor)


@jax.jit
def _build_design_jax(latent_reim: jnp.ndarray) -> jnp.ndarray:
    T = latent_reim.shape[0]
    return jnp.concatenate([jnp.ones((T, 1), dtype=latent_reim.dtype), latent_reim], axis=1)


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

def _reim_from_fine_single(mu_fine_RTP, var_fine_RTP, J, M):
    """Convert fine state to [Re | Im] format averaged over tapers."""
    R, T, _ = mu_fine_RTP.shape
    tmp = mu_fine_RTP.reshape(R, T, J, M, 2)
    vtmp = var_fine_RTP.reshape(R, T, J, M, 2)
    mu_re = tmp[..., 0].mean(axis=3)
    mu_im = tmp[..., 1].mean(axis=3)
    vr_re = vtmp[..., 0].mean(axis=3) / M
    vr_im = vtmp[..., 1].mean(axis=3) / M
    return np.concatenate([mu_re, mu_im], axis=2), np.concatenate([vr_re, vr_im], axis=2)

# ========================= Standardization =========================
# VERBATIM from run_joint_inference_trials_hier_shrinkage.py

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
# VERBATIM from run_joint_inference_trials_hier_shrinkage.py

def _apply_beta_shrinkage(
    beta_samples: np.ndarray,
    burn_in_frac: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply empirical Bayes shrinkage to β estimates.
    
    shrink[j] = |E[β_j]|² / (|E[β_j]|² + Var(β_j))
    
    This attenuates uncertain frequency components (high variance relative to mean).
    Frequencies with strong signal have shrink ≈ 1, noise frequencies have shrink ≈ 0.
    """
    n_samples = beta_samples.shape[0]
    burn_in = int(burn_in_frac * n_samples)
    samples = beta_samples[burn_in:]
    
    beta_mean = samples.mean(axis=0)
    beta_var = samples.var(axis=0)
    
    S, P = beta_mean.shape
    J = (P - 1) // 2
    
    beta_shrunk = beta_mean.copy()
    shrink_factors = np.ones((S, J), dtype=float)
    
    for j in range(J):
        idx_re = 1 + j
        idx_im = 1 + J + j
        
        mag_sq = beta_mean[:, idx_re]**2 + beta_mean[:, idx_im]**2
        var_sum = beta_var[:, idx_re] + beta_var[:, idx_im]
        
        shrink = mag_sq / (mag_sq + var_sum + 1e-12)
        shrink_factors[:, j] = shrink
        
        beta_shrunk[:, idx_re] *= shrink
        beta_shrunk[:, idx_im] *= shrink
    
    return beta_shrunk, shrink_factors


def _print_shrinkage_diagnostics(
    shrink_factors: np.ndarray,
    freqs_hz: Sequence[float],
    max_neurons: int = 2,
) -> None:
    """Print shrinkage diagnostics for each frequency."""
    pass  # Silenced for cleaner output


# ========================= Wald Test Band Selection =========================
# VERBATIM from run_joint_inference_trials_hier_shrinkage.py

def _wald_test_band_selection(
    beta_samples: np.ndarray,
    J: int,
    alpha: float = 0.05,
    burn_in_frac: float = 0.5,
    verbose: bool = True,
    freqs_hz: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Wald test to select (neuron, band) pairs with significant spike coupling.

    Tests H0: β_j = 0 (no coupling) for each (neuron, band) pair.

    Returns:
        significant_mask: (J,) bool - True if ANY neuron is significantly coupled to band
        W_stats: (S, J) Wald statistics per (neuron, band)
        p_values: (S, J) p-values per (neuron, band)
    """
    from scipy import stats

    n_samples, S, P = beta_samples.shape

    burn_in = int(burn_in_frac * n_samples)
    samples = beta_samples[burn_in:]
    n_post = samples.shape[0]

    # Layout: [β₀, βR_0, ..., βR_{J-1}, βI_0, ..., βI_{J-1}]
    beta_R = samples[:, :, 1:1+J]        # (n_post, S, J)
    beta_I = samples[:, :, 1+J:1+2*J]    # (n_post, S, J)

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
        n_sig_bands = significant_mask.sum()
        n_sig_pairs = (p_values < alpha).sum()
        total_pairs = S * J
        print(f"[WALD] Band selection (α={alpha}, burn-in={burn_in_frac}):")
        print(f"[WALD]   Using {n_post} post-burn-in samples")
        print(f"[WALD]   Significant (neuron,band) pairs: {n_sig_pairs}/{total_pairs}")
        print(f"[WALD]   Bands with ANY significant coupling: {n_sig_bands}/{J}")

        if freqs_hz is not None:
            sig_freqs = [f"{freqs_hz[j]:.1f}Hz" for j in range(J) if significant_mask[j]]
            nonsig_freqs = [f"{freqs_hz[j]:.1f}Hz" for j in range(J) if not significant_mask[j]]

            if sig_freqs:
                print(f"[WALD]   Coupled bands (joint KF): {', '.join(sig_freqs)}")
            if nonsig_freqs and len(nonsig_freqs) <= 10:
                print(f"[WALD]   Uncoupled bands (LFP-only): {', '.join(nonsig_freqs)}")
            elif nonsig_freqs:
                print(f"[WALD]   Uncoupled bands (LFP-only): {len(nonsig_freqs)} bands")

    return significant_mask, W_stats, p_values


def _zero_nonsignificant_beta_per_pair(
    beta: np.ndarray,
    p_values: np.ndarray,
    J: int,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Zero β coefficients for non-significant (neuron, band) pairs.
    
    This is PAIR-SPECIFIC: each (s, j) pair is gated independently.
    """
    S, P = beta.shape
    beta_masked = beta.copy()
    significant_pairs = p_values < alpha  # (S, J)
    
    n_zeroed = 0
    for s in range(S):
        for j in range(J):
            if not significant_pairs[s, j]:
                beta_masked[s, 1 + j] = 0.0        # β_R,j = 0
                beta_masked[s, 1 + J + j] = 0.0    # β_I,j = 0
                n_zeroed += 1
    
    n_sig = significant_pairs.sum()
    print(f"[GATING] β masking: {n_sig}/{S*J} pairs significant, {n_zeroed} zeroed")
    
    return beta_masked, significant_pairs


def _apply_true_band_gating(
    Z_fine_kf: np.ndarray,       # (R, T, 2*J*M) from joint KF
    Z_var_kf: np.ndarray,        # (R, T, 2*J*M) from joint KF
    Z_fine_lfponly: np.ndarray,  # (R, T, 2*J*M) from LFP-only (EM)
    Z_var_lfponly: np.ndarray,   # (R, T, 2*J*M) from LFP-only (EM)
    significant_mask: np.ndarray, # (J,) boolean - True if ANY neuron coupled
    J: int,
    M: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    TRUE per-band gating: Use joint KF estimates ONLY for coupled bands.
    For uncoupled bands (no significant neurons), keep the LFP-only estimates.
    """
    # Start with LFP-only estimates
    Z_gated = Z_fine_lfponly.copy()
    Z_var_gated = Z_var_lfponly.copy()
    
    n_coupled = 0
    n_uncoupled = 0
    
    for j in range(J):
        if significant_mask[j]:
            # This band IS coupled - use joint KF estimates
            n_coupled += 1
            for m in range(M):
                col_re = 2 * (j * M + m)
                col_im = col_re + 1
                Z_gated[:, :, col_re] = Z_fine_kf[:, :, col_re]
                Z_gated[:, :, col_im] = Z_fine_kf[:, :, col_im]
                Z_var_gated[:, :, col_re] = Z_var_kf[:, :, col_re]
                Z_var_gated[:, :, col_im] = Z_var_kf[:, :, col_im]
        else:
            # This band is NOT coupled - keep LFP-only (already in Z_gated)
            n_uncoupled += 1
    
    print(f"[GATING] State gating: {n_coupled} bands from joint KF, {n_uncoupled} bands from LFP-only")
    
    return Z_gated, Z_var_gated


def _select_significant_neurons(
    p_values: np.ndarray,
    alpha: float = 0.05,
    verbose: bool = True,
) -> np.ndarray:
    """
    Select neurons that show significant coupling to at least one band.
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


def _zero_nonsignificant_beta_per_neuron(
    beta: np.ndarray,
    p_values: np.ndarray,
    J: int,
    alpha: float = 0.05,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Zero β[s,j] when neuron s is NOT significantly coupled to band j.
    """
    S, P = beta.shape
    assert P == 1 + 2 * J, f"Expected P={1+2*J}, got {P}"
    assert p_values.shape == (S, J), f"Expected p_values shape ({S},{J}), got {p_values.shape}"

    beta_masked = beta.copy()
    significant_pairs = p_values < alpha  # (S, J) boolean

    for s in range(S):
        for j in range(J):
            if not significant_pairs[s, j]:
                # SEPARATED layout: βR at 1+j, βI at 1+J+j
                beta_masked[s, 1 + j] = 0.0          # β_R[s,j] = 0
                beta_masked[s, 1 + J + j] = 0.0      # β_I[s,j] = 0

    if verbose:
        n_zeroed = (~significant_pairs).sum()
        n_sig = significant_pairs.sum()
        print(f"[GATING] {n_sig}/{S*J} (neuron,band) pairs significant, {n_zeroed} zeroed")

    return beta_masked, significant_pairs


# ========================= Prior helpers =========================
# VERBATIM from run_joint_inference_trials_hier_shrinkage.py

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
    """Modulate complex latents by e^{+i 2π f t} for spike coupling."""
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


# ========================= Config =========================
# VERBATIM from run_joint_inference_trials_hier_shrinkage.py (renamed class only)

@dataclass
class SingleTrialInferenceConfig:
    """Configuration for single-trial joint inference."""
    fixed_iter: int = 500
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
    # Wald test band selection (TRUE GATING)
    use_wald_band_selection: bool = True
    wald_alpha: float = 0.05
    wald_burn_in_frac: float = 0.5
    verbose_wald: bool = True


# ========================= EM adapters =========================
# ADAPTED for single-trial (no X/D decomposition)

def _theta_from_em_single(res, J, M):
    """Extract θ from single-trial EM result."""
    lam = np.asarray(getattr(res, "lam"), float).reshape(J, M)
    sigv = np.asarray(getattr(res, "sigv"), float).reshape(J, M)
    sig_eps = np.asarray(getattr(res, "sig_eps"), float).reshape(J, M)
    
    return OUParams(lam=lam, sig_v=sigv, sig_eps=sig_eps)


def _extract_Z_from_upsampled_single(upsampled, J, M):
    """Extract Z from single-trial EM upsampling."""
    # For single-trial EM, Z_mean is (J, M, T) complex
    Zm = np.asarray(upsampled.Z_mean)  # (J, M, Tf) complex
    Zv = np.asarray(upsampled.Z_var)   # (J, M, Tf) real
    
    J_, M_, Tf = Zm.shape
    assert (J_, M_) == (J, M)
    
    # Convert to fine state format (T, 2*J*M)
    Z_fine = np.zeros((Tf, 2 * J * M), float)
    Z_var_fine = np.zeros((Tf, 2 * J * M), float)
    
    for j in range(J):
        for m in range(M):
            col = 2 * (j * M + m)
            Z_fine[:, col] = Zm[j, m, :].real
            Z_fine[:, col + 1] = Zm[j, m, :].imag
            Z_var_fine[:, col] = Zv[j, m, :]
            Z_var_fine[:, col + 1] = Zv[j, m, :]
    
    # Add R=1 dimension for compatibility: (1, T, 2*J*M)
    return Z_fine[None, :, :], Z_var_fine[None, :, :]




# ========================= Main runner =========================
# ADAPTED from run_joint_inference_trials_hier for R=1

def run_joint_inference_single_trial(
    Y_cube: np.ndarray,              # (J, M, K) - CHANGED from Y_trials (R, J, M, K)
    spikes_ST: np.ndarray,           # (S, T) - CHANGED from spikes_SRT (S, R, T)
    H_STL: np.ndarray,               # (S, T, L) - CHANGED from H_SRTL (S, R, T, L)
    all_freqs: Sequence[float],
    *,
    delta_spk: float,
    window_sec: float,
    offset_sec: float = 0.0,
    beta_init: Optional[np.ndarray] = None,
    gamma_prior_mu: Optional[np.ndarray] = None,
    gamma_prior_Sigma: Optional[np.ndarray] = None,
    config: Optional[SingleTrialInferenceConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, OUParams, Trace]:
    """
    Single-trial joint inference of spike-field coupling.
    
    DIRECT PORT of run_joint_inference_trials_hier with R=1.
    
    WITH TRUE BAND GATING:
    - Wald test identifies significant (neuron, band) pairs
    - Pair-specific β masking for spike observations
    - TRUE band gating: uncoupled bands keep LFP-only (EM) estimates
    
    Returns β in ORIGINAL (non-standardized) units.
    """
    if config is None:
        config = SingleTrialInferenceConfig()
    key = config.key_jax or jr.PRNGKey(0)

    print("[SINGLE-TRIAL] Starting joint inference...")
    print(f"[SINGLE-TRIAL] freeze_beta0 = {config.freeze_beta0}")
    print(f"[SINGLE-TRIAL] standardize_latents = {config.standardize_latents}")
    print(f"[SINGLE-TRIAL] use_beta_shrinkage = {config.use_beta_shrinkage}")
    print(f"[SINGLE-TRIAL] use_wald_band_selection = {config.use_wald_band_selection}")
    print(f"[SINGLE-TRIAL] trace_thin = {config.trace_thin}")
    if config.use_wald_band_selection:
        print(f"[SINGLE-TRIAL] TRUE GATING ENABLED:")
        print(f"[SINGLE-TRIAL]   - Pair-specific β masking")
        print(f"[SINGLE-TRIAL]   - Uncoupled bands keep LFP-only estimates")

    # SINGLE TRIAL: R=1
    R = 1
    J, M, K = Y_cube.shape
    S, T_f = spikes_ST.shape
    Rlags = H_STL.shape[-1]
    P = 1 + 2 * J
    print(f"[SINGLE-TRIAL] Data dimensions: R={R}, S={S}, J={J}, M={M}, T={T_f}, Rlags={Rlags}")

    # Reshape inputs to have R dimension for compatibility
    Y_trials = Y_cube[None, :, :, :]  # (1, J, M, K)
    spikes_SRT = spikes_ST[:, None, :]  # (S, 1, T)
    H_SRTL = H_STL[:, None, :, :]  # (S, 1, T, L)

    # 0) Single-trial EM warm start
    from src.em_ct_single_jax import em_ct_single_jax
    em_kwargs = dict(max_iter=1000, tol=1e-6, sig_eps_init=10.0, verbose=True, log_every=100)
    if config.em_kwargs:
        em_kwargs.update(config.em_kwargs)
    print("[SINGLE-TRIAL] Running EM warm start...")
    res = em_ct_single_jax(Y=jnp.asarray(Y_cube), db=window_sec, **em_kwargs)
    theta = _theta_from_em_single(res, J=J, M=M)
    print("[SINGLE-TRIAL] EM complete")
    print(f"[SINGLE-TRIAL] θ: λ range [{theta.lam.min():.4f}, {theta.lam.max():.4f}]")

    # 1) Upsample Z to fine grid
    from src.upsample_ct_single_fine import upsample_ct_single_fine
    print("[SINGLE-TRIAL] Upsampling EM latents to fine grid...")
    ups = upsample_ct_single_fine(
        Y=Y_cube, res=res, delta_spk=delta_spk,
        win_sec=window_sec, offset_sec=offset_sec, T_f=None
    )
    print(f"DEBUG: ups.Z_mean shape={ups.Z_mean.shape}, max={np.abs(ups.Z_mean).max():.6e}, min={np.abs(ups.Z_mean).min():.6e}")

    Z_fine, Z_var_fine = _extract_Z_from_upsampled_single(ups, J=J, M=M)
    print(f"DEBUG: Z_fine shape={Z_fine.shape}, max={np.abs(Z_fine).max():.6e}")

    # =====================================================================
    # STORE EM ESTIMATES AS LFP-ONLY REFERENCE (NEVER MODIFY THESE)
    # =====================================================================
    Z_fine_em = Z_fine.copy()  # (1, T, 2*J*M)
    Z_var_em = Z_var_fine.copy()
    print(f"[SINGLE-TRIAL] Stored EM estimates as LFP-only reference (shape: {Z_fine_em.shape})")
    # Store in trace for output
    lat_reim_RTP, var_reim_RTP = _reim_from_fine_single(Z_fine, Z_var_fine, J=J, M=M)
    T0 = min(T_f, lat_reim_RTP.shape[1])
    lat_reim_RTP = lat_reim_RTP[:, :T0]
    var_reim_RTP = var_reim_RTP[:, :T0]
    print(f"DEBUG: lat_reim (before rotation) shape={lat_reim_RTP.shape}, max={np.abs(lat_reim_RTP).max():.6e}")

    # Modulate for spike coupling (e^{+iωt} convention)
    lat_reim_RTP, var_reim_RTP = _rotate_reim_for_spikes(
        lat_reim_RTP, var_reim_RTP, all_freqs,
        delta_spk=delta_spk, offset_sec=offset_sec
    )
    print(f"DEBUG: lat_reim (after rotation) shape={lat_reim_RTP.shape}, max={np.abs(lat_reim_RTP).max():.6e}")
    # Standardize
    if config.standardize_latents:
        lat_reim_RTP, var_reim_RTP, latent_scale_factors = _standardize_latents(
            lat_reim_RTP, var_reim_RTP, min_std=config.min_latent_std
        )
        print(f"[SINGLE-TRIAL] Latent scale factors (min={latent_scale_factors.min():.4f}, "
              f"max={latent_scale_factors.max():.4f})")
    else:
        latent_scale_factors = np.ones(2 * J, dtype=float)
        print("[SINGLE-TRIAL] Latent standardization DISABLED")
    
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

    # 3) Trace
    trace = Trace()
    trace.theta = [theta]
    trace.shrinkage_factors = []
    trace.Z_fine_em = Z_fine_em
    trace.Z_var_em = Z_var_em

    # 4) Warmup loop
    tb_cfg = TrialBetaConfig(
        omega_floor=config.omega_floor,
        tau2_intercept=config.tau2_intercept,
        tau2_gamma=config.tau2_gamma,
        a0_ard=config.a0_ard,
        b0_ard=config.b0_ard,
    )
    beta_hist, gamma_hist = [], []
    print("[SINGLE-TRIAL] Starting warmup loop...")
    warmup_iter = tqdm(range(config.fixed_iter), desc="Warmup (PG-Gibbs)")
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
    print(f"[SINGLE-TRIAL] Warmup complete ({len(beta_hist)} samples stored)")

    # Freeze β₀ if requested
    if config.freeze_beta0:
        w = min(config.beta0_window // config.trace_thin, len(beta_hist))
        w = max(w, 1)
        beta0_fixed = np.mean(beta_hist[-w:, :, 0], axis=0)
        beta0_fixed_jax = jnp.asarray(beta0_fixed, dtype=beta.dtype)
        beta = beta.at[:, 0].set(beta0_fixed_jax)
        print(f"[SINGLE-TRIAL] Froze β₀ to median of last {w} warmup samples")
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
    print(f"[SINGLE-TRIAL] Warmup trace stored, memory freed")

    # =================================================================
    # WALD TEST FOR BAND SELECTION (after warmup)
    # =================================================================
    if config.use_wald_band_selection:
        print("[SINGLE-TRIAL] Performing Wald test for band selection...")

        # Stack all warmup samples
        beta_samples_warmup = np.stack(trace.beta, axis=0)

        # Perform Wald test
        significant_mask, W_stats, p_values = _wald_test_band_selection(
            beta_samples=beta_samples_warmup,
            J=J,
            alpha=config.wald_alpha,
            burn_in_frac=config.wald_burn_in_frac,
            verbose=config.verbose_wald,
            freqs_hz=all_freqs,
        )

        # Store in trace
        trace.wald_significant_mask = significant_mask
        trace.wald_W_stats = W_stats
        trace.wald_p_values = p_values

        n_significant = significant_mask.sum()
        print(f"[SINGLE-TRIAL] {n_significant}/{J} bands will use joint KF")
        print(f"[SINGLE-TRIAL] {J - n_significant}/{J} bands will keep LFP-only (EM) estimates")
    else:
        # All bands are "significant" -> all go through joint KF
        significant_mask = np.ones(J, dtype=bool)
        p_values = np.zeros((S, J))  # All significant
        print("[SINGLE-TRIAL] Wald test disabled - all bands use joint KF")

    # 5) Refresh passes
    sidx = StateIndex(J, M)
    print(f"[SINGLE-TRIAL] Starting {config.n_refreshes} refresh passes...")
    
    for rr in range(config.n_refreshes):
        print(f"[SINGLE-TRIAL] Refresh {rr + 1}/{config.n_refreshes}")
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
        # PAIR-SPECIFIC β MASKING: Zero non-significant (neuron, band) pairs
        # =================================================================
        if config.use_wald_band_selection:
            beta_for_kf, significant_pairs = _zero_nonsignificant_beta_per_pair(
                beta_mean_original, p_values, J, alpha=config.wald_alpha
            )
        else:
            beta_for_kf = beta_mean_original

        # Convert to INTERLEAVED for joint_kf_rts_moments
        from src.utils_common import separated_to_interleaved
        beta_interleaved = separated_to_interleaved(beta_for_kf)

        # =================================================================
        # KF REFRESH (single-trial: no pooling)
        # =================================================================
        print(f"[SINGLE-TRIAL] Refresh {rr + 1}: Running KF smoother...")

        mom = joint_kf_rts_moments(
            Y_cube=Y_cube,
            theta=theta,
            delta_spk=delta_spk,
            win_sec=window_sec,
            offset_sec=offset_sec,
            beta=beta_interleaved,
            gamma=np.asarray(gamma_shared_for_refresh),
            spikes=np.asarray(spikes_SRT_jax[:, 0, :]),  # (S, T)
            omega=np.asarray(omega_SRT[:, 0, :]),  # (S, T)
            coupled_bands_idx=np.arange(J, dtype=np.int64),
            freqs_for_phase=np.asarray(all_freqs, float),
            sidx=sidx,
            H_hist=np.asarray(H_SRTL_jax[:, 0, :, :]),  # (S, T, L)
            sigma_u=config.sigma_u,
            omega_floor=config.omega_floor,
        )

        Z_fine_kf = mom.m_s[None, :, :]   # (1, T, 2*J*M)
        Z_var_kf = mom.P_s[None, :, :]    # (1, T, 2*J*M)

        # =================================================================
        # TRUE BAND GATING: Keep LFP-only (EM) estimates for uncoupled bands
        # =================================================================
        T_kf = Z_fine_kf.shape[1]
        T0_new = min(T0, T_kf)
        
        if config.use_wald_band_selection:
            Z_fine_gated, Z_var_gated = _apply_true_band_gating(
                Z_fine_kf=Z_fine_kf[:, :T0_new, :],
                Z_var_kf=Z_var_kf[:, :T0_new, :],
                Z_fine_lfponly=Z_fine_em[:, :T0_new, :],
                Z_var_lfponly=Z_var_em[:, :T0_new, :],
                significant_mask=significant_mask,
                J=J, M=M
            )
        else:
            Z_fine_gated = Z_fine_kf[:, :T0_new, :]
            Z_var_gated = Z_var_kf[:, :T0_new, :]

        # Rebuild regressors from GATED smoothed Z
        lat_reim_RTP, var_reim_RTP = _reim_from_fine_single(Z_fine_gated, Z_var_gated, J=J, M=M)
        T0 = T0_new
        lat_reim_RTP = lat_reim_RTP[:, :T0]
        var_reim_RTP = var_reim_RTP[:, :T0]
        
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
        trace.theta.append(theta)
        
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
        # Store final joint KF estimates (will be overwritten each refresh, final one kept)
        trace.Z_fine_joint = Z_fine_gated
        trace.Z_var_joint = Z_var_gated
        print(f"[SINGLE-TRIAL] Refresh {rr + 1} complete (trace has {len(trace.beta)} samples)")

    # =================================================================
    # RESCALE β to original units
    # =================================================================
# =================================================================
    # RESCALE β to original units
    # =================================================================
    beta_final = np.asarray(beta)
    trace.beta_standardized = beta_final.copy()  # Store standardized version
    if config.standardize_latents:
        beta_final = _rescale_beta(beta_final, latent_scale_factors)
    trace.latent_scale_factors = latent_scale_factors
    trace.latent.append(lat_reim_RTP)
    
    print("[SINGLE-TRIAL] Inference complete")
    print(f"[SINGLE-TRIAL] Total trace samples: {len(trace.beta)}")
    if config.use_wald_band_selection:
        print(f"[SINGLE-TRIAL]   - {significant_mask.sum()}/{J} bands used joint KF")
        print(f"[SINGLE-TRIAL]   - {J - significant_mask.sum()}/{J} bands kept LFP-only")

    # Return gamma squeezed to (S, L) for single trial
    gamma_final = np.asarray(gamma)[:, 0, :]  # (S, R, L) -> (S, L)
    
    return beta_final, gamma_final, theta, trace