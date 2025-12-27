# em_ct_single_jax.py
"""
EM for single-trial CT-SSMT - FIXED to match NumPy em_ct exactly.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
from jax import lax
from dataclasses import dataclass

jax.config.update("jax_enable_x64", True)
EPS = 1e-12


@dataclass
class EMSingleResult:
    lam: jnp.ndarray
    sigv: jnp.ndarray
    sig_eps: jnp.ndarray
    ll_hist: jnp.ndarray
    Z_mean: jnp.ndarray
    Z_var: jnp.ndarray


# ============================================================
# RTS Smoother for single (j, m) chain
# ============================================================

def _rtss_single_jm(Y_jm, phi, q, R, z0, P0):
    """RTS smoother for a single (freq, taper) chain."""
    
    def filter_step(carry, y_k):
        z, P = carry
        z_pred = phi * z
        P_pred = phi * phi * P + q
        S = P_pred + R
        Kg = P_pred / S
        z_filt = z_pred + Kg * (y_k - z_pred)
        P_filt = (1.0 - Kg) * P_pred
        return (z_filt, P_filt), (z_filt, P_filt, z_pred, P_pred)
    
    _, (xf, Pf, xp, Pp) = lax.scan(filter_step, (z0, P0), Y_jm)
    
    def smooth_step(carry, inputs):
        xs_next, Ps_next = carry
        xf_k, Pf_k = inputs
        P_pred = phi * phi * Pf_k + q
        G = Pf_k * phi / P_pred
        xs_k = xf_k + G * (xs_next - phi * xf_k)
        Ps_k = Pf_k + G * G * (Ps_next - P_pred)
        return (xs_k, Ps_k), (xs_k, Ps_k)
    
    _, (xs_rev, Ps_rev) = lax.scan(
        smooth_step, (xf[-1], Pf[-1]), (xf[:-1], Pf[:-1]), reverse=True
    )
    xs = jnp.concatenate([xs_rev, xf[-1:]])
    Ps = jnp.concatenate([Ps_rev, Pf[-1:]])
    
    # EM sufficient statistics per (j, m) - NO summing across tapers
    Csum = jnp.sum(xs[1:] * jnp.conj(xs[:-1]))
    Rprev = jnp.sum(jnp.abs(xs[:-1])**2 + Ps[:-1])
    Rnext = jnp.sum(jnp.abs(xs[1:])**2 + Ps[1:])
    
    # For Q monitor
    E_ZZ = jnp.sum(jnp.abs(xs)**2 + Ps)
    E_YZ = jnp.real(jnp.sum(Y_jm * jnp.conj(xs)))
    
    diff = xs[1:] - phi * xs[:-1]
    E_dZ = jnp.sum(
        jnp.abs(diff)**2 + Ps[1:] + (phi**2) * Ps[:-1]
        - 2 * phi * jnp.real(xs[1:] * jnp.conj(xs[:-1]))
    )
    
    return xs, Ps, Csum, Rprev, Rnext, E_ZZ, E_YZ, E_dZ


# vmap over m (tapers) for fixed j
_rtss_single_j = jax.vmap(
    _rtss_single_jm,
    in_axes=(0, 0, 0, 0, 0, 0),  # All inputs vary over m
    out_axes=(0, 0, 0, 0, 0, 0, 0, 0)
)

# vmap over j (freqs)
_rtss_all_jm = jax.vmap(
    _rtss_single_j,
    in_axes=(0, 0, 0, 0, 0, 0),  # All inputs vary over j
    out_axes=(0, 0, 0, 0, 0, 0, 0, 0)
)


@jax.jit
def _rtss_ou_full(Y, phi_jm, q_jm, R_jm, z0_jm, P0_jm):
    """
    Full RTS smoother for (J, M, K) data.
    All parameters are (J, M).
    Returns statistics as (J, M) - per frequency AND per taper.
    """
    xs, Ps, Csum, Rprev, Rnext, E_ZZ, E_YZ, E_dZ = _rtss_all_jm(
        Y, phi_jm, q_jm, R_jm, z0_jm, P0_jm
    )
    return xs, Ps, Csum, Rprev, Rnext, E_ZZ, E_YZ, E_dZ


# ============================================================
# M-steps (matching NumPy exactly - all (J, M))
# ============================================================

@jax.jit
def _m_step_dynamics(Csum_jm, Rprev_jm, Rnext_jm, K, db, update_lam, lam_old_jm):
    """
    M-step for OU dynamics - per (j, m).
    All inputs/outputs are (J, M).
    """
    # Phi from |Csum|/Rprev per (j, m)
    Phi = jnp.clip(jnp.abs(Csum_jm) / (Rprev_jm + EPS), EPS, 1.0 - EPS)
    
    # Only update lambda if past freeze period
    lam_new = jnp.where(update_lam, -jnp.log(Phi) / db, lam_old_jm)
    
    # Q estimate per (j, m): (Rnext - Phi*Csum.conj()).real / K
    Qhat = (Rnext_jm - Phi * jnp.conj(Csum_jm)).real / K
    Qhat = jnp.maximum(Qhat, EPS)
    
    # sigma_v per (j, m)
    sigv_new = jnp.sqrt(2.0 * lam_new * Qhat / jnp.maximum(1.0 - Phi**2, EPS))
    
    return lam_new, sigv_new, Qhat


@jax.jit
def _m_step_obs_noise(Y, xs, Ps):
    """M-step for observation noise -> (M,) matching NumPy."""
    resid2 = jnp.abs(Y - xs)**2 + Ps  # (J, M, K)
    sig_eps = jnp.sqrt(jnp.mean(resid2, axis=(0, 2)) + EPS)  # (M,)
    return sig_eps


@jax.jit
def _compute_Q(E_YY, E_YZ, E_ZZ, E_dZ, Rbar, Qbar, K, J, M):
    """Q function matching NumPy em_ct."""
    ll = -(E_YY - 2*E_YZ + E_ZZ) / (Rbar + EPS) \
         - E_dZ / (Qbar + EPS) \
         - K * J * M * jnp.log(Rbar + EPS) \
         - K * J * jnp.log(Qbar + EPS)
    return ll


# ============================================================
# Main EM function
# ============================================================

def em_ct_single_jax(
    Y: jnp.ndarray,
    db: float,
    *,
    max_iter: int = 500,
    tol: float = 1e-3,
    sig_eps_init: float = 5.0,
    verbose: bool = True,
    log_every: int = 50,
    freeze_lam_iters: int = 50,
) -> EMSingleResult:
    """
    EM for single-trial CT-SSMT - matches NumPy em_ct exactly.
    
    Key: All dynamics parameters (lam, sigv) are (J, M) - per frequency AND per taper.
    """
    Y = jnp.asarray(Y)
    J, M, K = Y.shape
    
    # Initialize (J, M) parameters (matching NumPy)
    lam = jnp.full((J, M), 0.1)
    sigv = jnp.full((J, M), 1.0)
    sig_eps = jnp.full((M,), sig_eps_init)
    
    ll_hist = []
    Z_mean, Z_var = None, None
    
    for it in range(max_iter):
        # Compute phi, q per (j, m)
        phi_jm = jnp.exp(-lam * db)  # (J, M)
        q_jm = sigv**2 * (1.0 - phi_jm**2) / (2.0 * lam + EPS)  # (J, M)
        
        # Observation noise variance per (j, m)
        R_jm = jnp.broadcast_to(sig_eps[None, :]**2, (J, M))  # (J, M)
        
        # Diffuse prior
        z0_jm = jnp.zeros((J, M), dtype=Y.dtype)
        P0_jm = jnp.full((J, M), 1e3)
        
        # E-step: returns (J, M) statistics
        xs, Ps, Csum, Rprev, Rnext, E_ZZ, E_YZ, E_dZ = _rtss_ou_full(
            Y, phi_jm, q_jm, R_jm, z0_jm, P0_jm
        )
        Z_mean = xs
        Z_var = Ps.real
        
        # M-step: dynamics per (j, m)
        update_lam = it >= freeze_lam_iters
        lam, sigv, Qhat = _m_step_dynamics(Csum, Rprev, Rnext, K, db, update_lam, lam)
        
        # M-step: observation noise -> (M,)
        sig_eps = _m_step_obs_noise(Y, xs, Ps)
        
        # Q monitor
        E_YY = jnp.sum(jnp.abs(Y)**2)
        Rbar = jnp.mean(sig_eps**2)
        Qbar = jnp.mean(Qhat)
        E_YZ_sum = jnp.sum(E_YZ)
        E_ZZ_sum = jnp.sum(E_ZZ)
        E_dZ_sum = jnp.sum(E_dZ)
        
        ll = _compute_Q(E_YY, E_YZ_sum, E_ZZ_sum, E_dZ_sum, Rbar, Qbar, K, J, M)
        ll_hist.append(float(ll))
        
        if verbose and (it % log_every == 0 or it == max_iter - 1):
            print(f"[EM-CT-JAX] iter {it:4d}  Q = {ll:.4e}")
        
        if it == freeze_lam_iters and verbose:
            print(f"[EM-CT-JAX] Unfreezing lambda at iter {it}")
        
        # Convergence
        if it > 0 and abs(ll_hist[-1] - ll_hist[-2]) < tol:
            if verbose:
                print(f"[EM-CT-JAX] Converged at iter {it}  Q = {ll:.4e}")
            break
    
    # Output sig_eps as (J, M) for API consistency
    sig_eps_out = jnp.broadcast_to(sig_eps[None, :], (J, M))
    
    return EMSingleResult(
        lam=lam,
        sigv=sigv,
        sig_eps=sig_eps_out,
        ll_hist=jnp.array(ll_hist),
        Z_mean=Z_mean,
        Z_var=Z_var,
    )