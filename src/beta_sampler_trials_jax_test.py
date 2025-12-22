# beta_sampler_trials_jax.py
# JAX-native β–γ PG–Gaussian sampler for trial-aware inference
# β shared across trials, γ shared across trials per unit
# Fully vectorized, JIT-compiled, device-resident
#
# STANDARDIZATION SUPPORT:
# This sampler supports working in standardized predictor space while
# maintaining exact Gibbs sampling. When latent_scales are provided:
# - Prior precision uses σ²τ² (scaled prior variance)
# - ARD update uses (β'/σ)² to update τ² in original units
# - Output β' is in standardized units (caller must rescale)
#
# MATHEMATICAL JUSTIFICATION:
# Original model: ψ = β₀ + Σⱼ βⱼ Zⱼ with βⱼ|τ²ⱼ ~ N(0,τ²ⱼ)
# Standardized:   ψ = β₀ + Σⱼ β'ⱼ Z'ⱼ where Z'ⱼ = Zⱼ/σⱼ, β'ⱼ = σⱼβⱼ
# Induced prior:  β'ⱼ|τ²ⱼ ~ N(0, σⱼ²τ²ⱼ)
# ARD update:     τ²ⱼ|β'ⱼ ~ InvGamma(a₀+0.5, b₀ + (β'ⱼ/σⱼ)²/2)
#
# This is EXACT - we sample from the correct posterior in transformed coordinates.

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp

# 64-bit numerics for stable Cholesky in near-singular blocks
jax.config.update("jax_enable_x64", True)

# -----------------------------------------------------------------------------
# Config (registered as JAX pytree for passing through transformations)
# -----------------------------------------------------------------------------

@dataclass
class TrialBetaConfig:
    """Configuration for trial-aware β/γ sampler (JAX-compatible pytree).
    
    Attributes
    ----------
    omega_floor : float
        Minimum value for PG samples (numerical stability).
    tau2_intercept : float
        Prior variance for β₀ (intercept). Large = weak prior.
    tau2_gamma : float
        Ridge regularization for γ (history coefficients).
    a0_ard, b0_ard : float
        ARD prior hyperparameters: τ² ~ InvGamma(a0, b0).
    """
    omega_floor: float = 1e-3
    tau2_intercept: float = 1e4
    tau2_gamma: float = 25.0 ** 2
    a0_ard: float = 1e-2
    b0_ard: float = 1e-2

    def _tree_flatten(self):
        children = (self.omega_floor, self.tau2_intercept, self.tau2_gamma,
                    self.a0_ard, self.b0_ard)
        aux_data = None
        return children, aux_data

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children)


jax.tree_util.register_pytree_node(
    TrialBetaConfig,
    TrialBetaConfig._tree_flatten,
    TrialBetaConfig._tree_unflatten
)

# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

@jax.jit
def build_design_jax(latent_reim: jnp.ndarray) -> jnp.ndarray:
    """
    Prepend intercept column - pure JAX.
    latent_reim: (T, 2B)
    returns:     (T, P) where P = 1 + 2B
    """
    T = latent_reim.shape[0]
    return jnp.concatenate([jnp.ones((T, 1), dtype=latent_reim.dtype), latent_reim], axis=1)


def _reduce_gamma_priors_to_S(
    Prec_gamma: jnp.ndarray,
    mu_gamma: jnp.ndarray,
    S: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Reduce potentially (S,R,L,L) and (S,R,L) priors to per-unit (S,L,L) and (S,L).
    """
    if Prec_gamma.ndim == 2:
        L = Prec_gamma.shape[0]
        Prec_gamma = jnp.broadcast_to(Prec_gamma, (S, L, L))
    elif Prec_gamma.ndim == 3:
        pass
    elif Prec_gamma.ndim == 4:
        Prec_eff = Prec_gamma.sum(axis=1)
    else:
        raise ValueError(f"Unsupported Prec_gamma shape: {Prec_gamma.shape}")

    if mu_gamma.ndim == 1:
        L = mu_gamma.shape[0]
        mu_gamma = jnp.broadcast_to(mu_gamma, (S, L))
    elif mu_gamma.ndim == 2:
        pass
    elif mu_gamma.ndim == 3:
        if Prec_gamma.ndim != 4:
            mu_gamma = mu_gamma.mean(axis=1)
    else:
        raise ValueError(f"Unsupported mu_gamma shape: {mu_gamma.shape}")

    if Prec_gamma.ndim == 3:
        Prec_eff = Prec_gamma

    if mu_gamma.ndim == 3:
        num = jnp.einsum('srlk,srl->sk', Prec_gamma, mu_gamma)
        mu_eff = jsp.linalg.solve(Prec_eff, num, assume_a='pos')
        return Prec_eff, mu_eff
    else:
        return Prec_eff, mu_gamma


# -----------------------------------------------------------------------------
# Single-unit kernels with standardization support
# -----------------------------------------------------------------------------

def _beta_gamma_joint_unit_sharedX(
    key: jr.KeyArray,
    X: jnp.ndarray,             # (T, P) shared across trials
    H_rtl: jnp.ndarray,         # (R, T, L)
    spikes_rt: jnp.ndarray,     # (R, T)
    omega_rt: jnp.ndarray,      # (R, T)
    V_rtb: jnp.ndarray,         # (R, T, 2B)  EIV variances for latent cols only
    Prec_gamma_ll: jnp.ndarray, # (L, L)
    mu_gamma_l: jnp.ndarray,    # (L,)
    tau2_lat_b: jnp.ndarray,    # (2B,) - τ² in ORIGINAL units
    latent_scales_b: jnp.ndarray,  # (2B,) - σ_j scale factors (1.0 if no standardization)
    cfg: TrialBetaConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Joint Gaussian sample for θ=[β;γ] with shared design X.
    
    When latent_scales != 1, assumes X contains standardized predictors Z' = Z/σ.
    The prior on β' is N(0, σ²τ²) to maintain equivalence with original model.
    ARD updates τ² using (β'/σ)² to keep τ² in original units.
    """
    key1, key2 = jr.split(key)

    R, T, L = H_rtl.shape
    T_x, P = X.shape
    assert T_x == T, "Design matrix and spikes time dimension mismatch"
    twoB = P - 1
    dtype = X.dtype

    omega_rt = jnp.maximum(omega_rt, cfg.omega_floor)
    kappa_rt = spikes_rt - 0.5

    # Prior precision for β' (standardized coefficients)
    # If β ~ N(0, τ²), then β' = σβ ~ N(0, σ²τ²)
    # So precision on β' is 1/(σ²τ²)
    sigma2_b = latent_scales_b ** 2
    tau2_effective = sigma2_b * jnp.maximum(tau2_lat_b, 1e-12)
    
    Prec_beta_diag = jnp.zeros((P,), dtype=dtype)
    Prec_beta_diag = Prec_beta_diag.at[0].set(1.0 / cfg.tau2_intercept)
    Prec_beta_diag = Prec_beta_diag.at[1:].set(1.0 / tau2_effective)

    # γ prior precision with minimum ridge
    Prec_gamma_ll = Prec_gamma_ll + (1.0 / cfg.tau2_gamma) * jnp.eye(L, dtype=dtype)

    # Accumulate across trials with einsums
    A_bb = jnp.einsum('tp,t,tq->pq', X, omega_rt.sum(axis=0), X, optimize=True)
    A_bg = jnp.einsum('tp,rt,rtl->pl', X, omega_rt, H_rtl, optimize=True)
    A_gg = jnp.einsum('rtl,rt,rtm->lm', H_rtl, omega_rt, H_rtl, optimize=True)

    # RHS
    b_b = jnp.einsum('tp,t->p', X, kappa_rt.sum(axis=0), optimize=True)
    b_g = jnp.einsum('rtl,rt->l', H_rtl, kappa_rt, optimize=True) + Prec_gamma_ll @ mu_gamma_l

    # EIV correction for β latent cols
    diag_add = jnp.einsum('rtb,rt->b', V_rtb, omega_rt, optimize=True)
    A_bb = A_bb + jnp.diag(Prec_beta_diag)
    A_gg = A_gg + Prec_gamma_ll
    A_bb = A_bb.at[1:, 1:].add(jnp.diag(diag_add))

    # Assemble joint precision
    dim = P + L
    Prec = jnp.zeros((dim, dim), dtype=dtype)
    Prec = Prec.at[:P, :P].set(A_bb)
    Prec = Prec.at[:P, P:].set(A_bg)
    Prec = Prec.at[P:, :P].set(A_bg.T)
    Prec = Prec.at[P:, P:].set(A_gg)

    Prec = 0.5 * (Prec + Prec.T) + (1e-8 if dtype == jnp.float64 else 1e-6) * jnp.eye(dim, dtype=dtype)

    # Solve & sample
    chol = jnp.linalg.cholesky(Prec)
    v = jsp.linalg.solve_triangular(chol, jnp.concatenate([b_b, b_g]), lower=True)
    mean = jsp.linalg.solve_triangular(chol.T, v, lower=False)
    eps = jr.normal(key1, (dim,), dtype=dtype)
    theta = mean + jsp.linalg.solve_triangular(chol.T, eps, lower=False)

    beta = theta[:P]
    gamma = theta[P:]

    # ARD update: keep τ² in original units
    # If β' = σβ, then β = β'/σ, and we update τ² using β² = (β'/σ)²
    beta_lat = beta[1:]  # β' in standardized units
    beta_original = beta_lat / latent_scales_b  # β in original units
    
    alpha_post = cfg.a0_ard + 0.5
    beta_post = cfg.b0_ard + 0.5 * (beta_original ** 2)
    gdraw = jr.gamma(key2, alpha_post, shape=(twoB,), dtype=dtype)
    tau2_new = beta_post / jnp.maximum(gdraw, 1e-12)

    return beta, gamma, tau2_new


def _beta_gamma_joint_unit_pertrial(
    key: jr.KeyArray,
    X_rtp: jnp.ndarray,        # (R, T, P) per-trial design
    H_rtl: jnp.ndarray,        # (R, T, L)
    spikes_rt: jnp.ndarray,    # (R, T)
    omega_rt: jnp.ndarray,     # (R, T)
    V_rtb: jnp.ndarray,        # (R, T, 2B)
    Prec_gamma_ll: jnp.ndarray,# (L, L)
    mu_gamma_l: jnp.ndarray,   # (L,)
    tau2_lat_b: jnp.ndarray,   # (2B,) - τ² in ORIGINAL units
    latent_scales_b: jnp.ndarray,  # (2B,) - σ_j scale factors
    cfg: TrialBetaConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Joint Gaussian sample for θ=[β;γ] with per-trial design X_rtp.
    
    Supports standardized predictors with exact prior transformation.
    """
    key1, key2 = jr.split(key)

    R, T, P = X_rtp.shape
    R2, T2, L = H_rtl.shape
    assert (R2, T2) == (R, T)
    twoB = P - 1
    dtype = X_rtp.dtype

    omega_rt = jnp.maximum(omega_rt, cfg.omega_floor)
    kappa_rt = spikes_rt - 0.5

    # Prior precision for β' (standardized coefficients)
    sigma2_b = latent_scales_b ** 2
    tau2_effective = sigma2_b * jnp.maximum(tau2_lat_b, 1e-12)
    
    Prec_beta_diag = jnp.zeros((P,), dtype=dtype)
    Prec_beta_diag = Prec_beta_diag.at[0].set(1.0 / cfg.tau2_intercept)
    Prec_beta_diag = Prec_beta_diag.at[1:].set(1.0 / tau2_effective)

    # γ prior precision with minimum ridge
    Prec_gamma_ll = Prec_gamma_ll + (1.0 / cfg.tau2_gamma) * jnp.eye(L, dtype=dtype)

    # Blocks via einsum
    A_bb = jnp.einsum('rtp,rt,rtq->pq', X_rtp, omega_rt, X_rtp, optimize=True)
    A_bg = jnp.einsum('rtp,rt,rtl->pl', X_rtp, omega_rt, H_rtl, optimize=True)
    A_gg = jnp.einsum('rtl,rt,rtm->lm', H_rtl, omega_rt, H_rtl, optimize=True)

    # RHS
    b_b = jnp.einsum('rtp,rt->p', X_rtp, kappa_rt, optimize=True)
    b_g = jnp.einsum('rtl,rt->l', H_rtl, kappa_rt, optimize=True) + Prec_gamma_ll @ mu_gamma_l

    # EIV correction
    diag_add = jnp.einsum('rtb,rt->b', V_rtb, omega_rt, optimize=True)
    A_bb = A_bb + jnp.diag(Prec_beta_diag)
    A_gg = A_gg + Prec_gamma_ll
    A_bb = A_bb.at[1:, 1:].add(jnp.diag(diag_add))

    # Assemble joint precision
    dim = P + L
    Prec = jnp.zeros((dim, dim), dtype=dtype)
    Prec = Prec.at[:P, :P].set(A_bb)
    Prec = Prec.at[:P, P:].set(A_bg)
    Prec = Prec.at[P:, :P].set(A_bg.T)
    Prec = Prec.at[P:, P:].set(A_gg)

    Prec = 0.5 * (Prec + Prec.T) + (1e-8 if dtype == jnp.float64 else 1e-6) * jnp.eye(dim, dtype=dtype)

    # Solve & sample
    chol = jnp.linalg.cholesky(Prec)
    v = jsp.linalg.solve_triangular(chol, jnp.concatenate([b_b, b_g]), lower=True)
    mean = jsp.linalg.solve_triangular(chol.T, v, lower=False)
    eps = jr.normal(key1, (dim,), dtype=dtype)
    theta = mean + jsp.linalg.solve_triangular(chol.T, eps, lower=False)

    beta = theta[:P]
    gamma = theta[P:]

    # ARD update: keep τ² in original units
    beta_lat = beta[1:]
    beta_original = beta_lat / latent_scales_b
    
    alpha_post = cfg.a0_ard + 0.5
    beta_post = cfg.b0_ard + 0.5 * (beta_original ** 2)
    gdraw = jr.gamma(key2, alpha_post, shape=(twoB,), dtype=dtype)
    tau2_new = beta_post / jnp.maximum(gdraw, 1e-12)

    return beta, gamma, tau2_new


# JIT-compile kernels
_beta_gamma_joint_unit_sharedX_jit = jax.jit(_beta_gamma_joint_unit_sharedX)
_beta_gamma_joint_unit_pertrial_jit = jax.jit(_beta_gamma_joint_unit_pertrial)


# -----------------------------------------------------------------------------
# Vectorized samplers across units
# -----------------------------------------------------------------------------

def _prepare_priors_S(
    Prec_gamma_S_like: jnp.ndarray,
    mu_gamma_S_like: jnp.ndarray,
    S: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Normalize priors into (S,L,L) and (S,L)."""
    if Prec_gamma_S_like.ndim == 2:
        L = Prec_gamma_S_like.shape[0]
        Prec_gamma_S = jnp.tile(Prec_gamma_S_like[None, ...], (S, 1, 1))
    elif Prec_gamma_S_like.ndim == 3:
        Prec_gamma_S = Prec_gamma_S_like
    elif Prec_gamma_S_like.ndim == 4:
        Prec_gamma_S = Prec_gamma_S_like
    else:
        raise ValueError(f"Unsupported Prec_gamma shape: {Prec_gamma_S_like.shape}")

    if mu_gamma_S_like.ndim == 1:
        L = mu_gamma_S_like.shape[0]
        mu_gamma_S = jnp.tile(mu_gamma_S_like[None, :], (S, 1))
    elif mu_gamma_S_like.ndim == 2:
        mu_gamma_S = mu_gamma_S_like
    elif mu_gamma_S_like.ndim == 3:
        mu_gamma_S = mu_gamma_S_like
    else:
        raise ValueError(f"Unsupported mu_gamma shape: {mu_gamma_S_like.shape}")

    if Prec_gamma_S.ndim == 4 and mu_gamma_S.ndim == 3:
        S_, R_, L, _ = Prec_gamma_S.shape
        assert S_ == S
        Prec_eff = Prec_gamma_S.sum(axis=1)
        num = jnp.einsum('srlk,srl->sk', Prec_gamma_S, mu_gamma_S)
        mu_eff = jsp.linalg.solve(Prec_eff, num, assume_a='pos')
        return Prec_eff, mu_eff

    if Prec_gamma_S.ndim == 4 and mu_gamma_S.ndim == 2:
        Prec_eff = Prec_gamma_S.sum(axis=1)
        return Prec_eff, mu_gamma_S

    return Prec_gamma_S, mu_gamma_S


def _gibbs_update_beta_trials_shared_vectorized(
    key: jr.KeyArray,
    X: jnp.ndarray,
    H_S_rtl: jnp.ndarray,
    spikes_S_rt: jnp.ndarray,
    omega_S_rt: jnp.ndarray,
    V_S_rtb: jnp.ndarray,
    Prec_gamma_S_ll_like: jnp.ndarray,
    mu_gamma_S_l_like: jnp.ndarray,
    tau2_lat_S_b: jnp.ndarray,
    latent_scales_b: jnp.ndarray,  # (2B,) shared across units
    cfg: TrialBetaConfig,
) -> Tuple[jr.KeyArray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    S = H_S_rtl.shape[0]
    Prec_gamma_S_ll, mu_gamma_S_l = _prepare_priors_S(Prec_gamma_S_ll_like, mu_gamma_S_l_like, S)

    keys = jr.split(key, S)
    beta_S, gamma_S, tau2_S = jax.vmap(
        lambda k, H_rtl, spk_rt, om_rt, V_rtb, Prec_g, mu_g, tau2_b:
            _beta_gamma_joint_unit_sharedX_jit(
                k, X, H_rtl, spk_rt, om_rt, V_rtb, Prec_g, mu_g, tau2_b, latent_scales_b, cfg
            )
    )(keys, H_S_rtl, spikes_S_rt, omega_S_rt, V_S_rtb, Prec_gamma_S_ll, mu_gamma_S_l, tau2_lat_S_b)

    return jr.fold_in(key, 1), beta_S, gamma_S, tau2_S


gibbs_update_beta_trials_shared_vectorized = jax.jit(
    _gibbs_update_beta_trials_shared_vectorized,
    static_argnums=()
)


def _gibbs_update_beta_trials_shared_Xrtp_vectorized(
    key: jr.KeyArray,
    X_RTP: jnp.ndarray,
    H_S_rtl: jnp.ndarray,
    spikes_S_rt: jnp.ndarray,
    omega_S_rt: jnp.ndarray,
    V_S_rtb: jnp.ndarray,
    Prec_gamma_S_ll_like: jnp.ndarray,
    mu_gamma_S_l_like: jnp.ndarray,
    tau2_lat_S_b: jnp.ndarray,
    latent_scales_b: jnp.ndarray,  # (2B,) shared across units
    cfg: TrialBetaConfig,
) -> Tuple[jr.KeyArray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    S = H_S_rtl.shape[0]
    Prec_gamma_S_ll, mu_gamma_S_l = _prepare_priors_S(Prec_gamma_S_ll_like, mu_gamma_S_l_like, S)

    keys = jr.split(key, S)
    beta_S, gamma_S, tau2_S = jax.vmap(
        lambda k, H_rtl, spk_rt, om_rt, V_rtb, Prec_g, mu_g, tau2_b:
            _beta_gamma_joint_unit_pertrial_jit(
                k, X_RTP, H_rtl, spk_rt, om_rt, V_rtb, Prec_g, mu_g, tau2_b, latent_scales_b, cfg
            )
    )(keys, H_S_rtl, spikes_S_rt, omega_S_rt, V_S_rtb, Prec_gamma_S_ll, mu_gamma_S_l, tau2_lat_S_b)

    return jr.fold_in(key, 1), beta_S, gamma_S, tau2_S


gibbs_update_beta_trials_shared_Xrtp_vectorized = jax.jit(
    _gibbs_update_beta_trials_shared_Xrtp_vectorized,
    static_argnums=()
)


# -----------------------------------------------------------------------------
# Backward compatibility wrapper
# -----------------------------------------------------------------------------

def gibbs_update_beta_trials_shared(
    key: jr.KeyArray,
    *,
    latent_reim: jnp.ndarray,
    spikes: jnp.ndarray,
    omega: jnp.ndarray,
    H_hist: jnp.ndarray,
    Sigma_gamma: Optional[jnp.ndarray] = None,
    mu_gamma: Optional[jnp.ndarray] = None,
    var_latent_reim: Optional[jnp.ndarray] = None,
    tau2_lat: Optional[jnp.ndarray] = None,
    latent_scales: Optional[jnp.ndarray] = None,  # NEW: scale factors
    config: Optional[TrialBetaConfig] = None,
) -> Tuple[jr.KeyArray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Backward-compatible wrapper for single-unit API."""
    if config is None:
        config = TrialBetaConfig()

    latent_reim = jnp.asarray(latent_reim)
    spikes = jnp.asarray(spikes)
    omega = jnp.asarray(omega)
    H_hist = jnp.asarray(H_hist)

    T, twoJ = latent_reim.shape
    R, T_check, L = H_hist.shape
    assert T == T_check, "Time dimension mismatch"

    X = build_design_jax(latent_reim)
    P = X.shape[1]

    if var_latent_reim is None:
        V_rtb = jnp.zeros((R, T, twoJ), dtype=latent_reim.dtype)
    else:
        var_latent_reim = jnp.asarray(var_latent_reim)
        V_rtb = jnp.tile(var_latent_reim[None, ...], (R, 1, 1))

    if Sigma_gamma is None:
        Prec_gamma_like = jnp.eye(L, dtype=latent_reim.dtype)
    else:
        Sigma_gamma = jnp.asarray(Sigma_gamma)
        if Sigma_gamma.ndim == 2:
            Prec_gamma_like = jnp.linalg.inv(Sigma_gamma + 1e-8 * jnp.eye(L, dtype=Sigma_gamma.dtype))
        elif Sigma_gamma.ndim == 3:
            Prec_gamma_like = jnp.linalg.inv(Sigma_gamma + 1e-8 * jnp.eye(L, dtype=Sigma_gamma.dtype)[None, ...])
        else:
            raise ValueError(f"Sigma_gamma has unexpected shape: {Sigma_gamma.shape}")

    if mu_gamma is None:
        mu_gamma_like = jnp.zeros((L,), dtype=latent_reim.dtype)
    else:
        mu_gamma_like = jnp.asarray(mu_gamma)

    tau2_lat_b = jnp.ones(twoJ, dtype=latent_reim.dtype) if (tau2_lat is None) else jnp.asarray(tau2_lat)
    
    # Scale factors default to 1.0 (no standardization)
    if latent_scales is None:
        latent_scales_b = jnp.ones(twoJ, dtype=latent_reim.dtype)
    else:
        latent_scales_b = jnp.asarray(latent_scales)

    H_S_rtl = H_hist[None, ...]
    spikes_S_rt = spikes[None, ...]
    omega_S_rt = omega[None, ...]
    V_S_rtb = V_rtb[None, ...]
    Prec_gamma_Sll = Prec_gamma_like[None, ...] if Prec_gamma_like.ndim == 2 else Prec_gamma_like[None, ...]
    mu_gamma_Sl = mu_gamma_like[None, ...] if mu_gamma_like.ndim == 1 else mu_gamma_like[None, ...]
    tau2_lat_Sb = tau2_lat_b[None, ...]

    key_out, beta_S, gamma_S, tau2_S = _gibbs_update_beta_trials_shared_vectorized(
        key, X, H_S_rtl, spikes_S_rt, omega_S_rt, V_S_rtb,
        Prec_gamma_Sll, mu_gamma_Sl, tau2_lat_Sb, latent_scales_b, config
    )

    beta = beta_S[0]
    gamma_unit = gamma_S[0]
    tau2_new = tau2_S[0]
    gamma_bcast = jnp.tile(gamma_unit[None, :], (R, 1))
    return key_out, beta, gamma_bcast, tau2_new
