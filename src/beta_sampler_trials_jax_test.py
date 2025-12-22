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
from typing import Tuple

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


# -----------------------------------------------------------------------------
# Single-unit kernel with standardization support (per-trial design)
# -----------------------------------------------------------------------------

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


# JIT-compile kernel
_beta_gamma_joint_unit_pertrial_jit = jax.jit(_beta_gamma_joint_unit_pertrial)


# -----------------------------------------------------------------------------
# Vectorized sampler across units
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
