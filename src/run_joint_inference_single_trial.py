"""
JAX-native joint inference with latent standardization and beta shrinkage.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap, lax
from functools import partial
import gc

from src.joint_inference_core import JointMoments
from src.params import OUParams
from src.priors import gamma_prior_simple
from src.polyagamma_jax import sample_pg_saddle_single
from src.utils_joint import Trace


# ============================================================================
# Pure JAX utility functions
# ============================================================================

@jax.jit
def _sample_omega_pg_batch(key, psi: jnp.ndarray, omega_floor: float) -> jnp.ndarray:
    N = psi.shape[0]
    keys = jr.split(key, N)
    omega = jax.vmap(lambda k, z: sample_pg_saddle_single(k, 1.0, z))(keys, psi)
    return jnp.maximum(omega, omega_floor)


@jax.jit
def _build_design_jax(latent_reim: jnp.ndarray) -> jnp.ndarray:
    T = latent_reim.shape[0]
    return jnp.concatenate([jnp.ones((T, 1)), latent_reim], axis=1)


@jax.jit
def _gibbs_update_beta_gamma_jax(
    key, beta, gamma, tau2_lat, X, H, spikes, V,
    Prec_gamma, mu_gamma, omega, tau2_intercept, a0_ard, b0_ard,
):
    key1, key2 = jr.split(key)
    T, P = X.shape
    R = H.shape[1]
    twoB = P - 1

    kappa = spikes - 0.5
    Prec_beta_diag = jnp.zeros(P)
    Prec_beta_diag = Prec_beta_diag.at[0].set(1.0 / tau2_intercept)
    Prec_beta_diag = Prec_beta_diag.at[1:].set(1.0 / jnp.maximum(tau2_lat, 1e-12))

    sqrt_omega = jnp.sqrt(omega)[:, None]
    Xw = sqrt_omega * X
    Hw = sqrt_omega * H

    Prec_beta_block = Xw.T @ Xw + jnp.diag(Prec_beta_diag)
    diag_add = V.T @ omega
    Prec_beta_block = Prec_beta_block.at[1:, 1:].add(jnp.diag(diag_add))

    Prec_gamma_block = Hw.T @ Hw + Prec_gamma
    Prec_cross = Xw.T @ Hw

    Prec = jnp.zeros((P + R, P + R))
    Prec = Prec.at[:P, :P].set(Prec_beta_block)
    Prec = Prec.at[:P, P:].set(Prec_cross)
    Prec = Prec.at[P:, :P].set(Prec_cross.T)
    Prec = Prec.at[P:, P:].set(Prec_gamma_block)

    h_beta = X.T @ kappa
    h_gamma = H.T @ kappa + Prec_gamma @ mu_gamma
    h = jnp.concatenate([h_beta, h_gamma])

    Prec = 0.5 * (Prec + Prec.T) + 1e-8 * jnp.eye(P + R)
    L = jnp.linalg.cholesky(Prec)
    v = jax.scipy.linalg.solve_triangular(L, h, lower=True)
    mean = jax.scipy.linalg.solve_triangular(L.T, v, lower=False)

    eps = jr.normal(key1, shape=(P + R,))
    theta = mean + jax.scipy.linalg.solve_triangular(L.T, eps, lower=False)

    beta_new = theta[:P]
    gamma_new = theta[P:]

    beta_lat = beta_new[1:]
    alpha_post = a0_ard + 0.5
    beta_post = b0_ard + 0.5 * (beta_lat ** 2)
    tau2_lat_new = 1.0 / jr.gamma(key2, alpha_post, shape=(twoB,)) * beta_post

    return beta_new, gamma_new, tau2_lat_new


_gibbs_update_vectorized = vmap(
    _gibbs_update_beta_gamma_jax,
    in_axes=(0, 0, 0, 0, None, 0, 0, None, 0, 0, 0, None, None, None)
)


@partial(jax.jit, static_argnames=['n_iter'])
def _warmup_loop_scan(
    key, beta_init, gamma_init, tau2_init, X, H_all, spikes_all, V,
    Prec_gamma_all, mu_gamma_all, omega_floor, tau2_intercept, a0_ard, b0_ard, n_iter,
):
    S = beta_init.shape[0]

    def scan_fn(carry, key_iter):
        beta, gamma, tau2 = carry
        psi_all = vmap(lambda b, g, h: X @ b + h @ g)(beta, gamma, H_all)
        keys_omega = jr.split(key_iter, S)
        omega_all = vmap(lambda k, psi: _sample_omega_pg_batch(k, psi, omega_floor))(keys_omega, psi_all)
        keys_gibbs = jr.split(jr.fold_in(key_iter, 1), S)
        beta_new, gamma_new, tau2_new = _gibbs_update_vectorized(
            keys_gibbs, beta, gamma, tau2, X, H_all, spikes_all, V,
            Prec_gamma_all, mu_gamma_all, omega_all, tau2_intercept, a0_ard, b0_ard
        )
        return (beta_new, gamma_new, tau2_new), (beta_new, gamma_new)

    keys = jr.split(key, n_iter)
    init_carry = (beta_init, gamma_init, tau2_init)
    final_carry, history = lax.scan(scan_fn, init_carry, keys)
    return final_carry, history


@partial(jax.jit, static_argnames=['n_iter'])
def _inner_loop_scan(
    key, beta_init, gamma_init, tau2_init, beta0_fixed, X, H_all, spikes_all, V,
    Prec_gamma_lock, mu_gamma_post, Sigma_gamma_post, omega_floor, tau2_intercept,
    a0_ard, b0_ard, n_iter,
):
    S = beta_init.shape[0]
    R = gamma_init.shape[1]

    def scan_fn(carry, key_iter):
        beta, gamma, tau2 = carry
        keys_gamma = jr.split(key_iter, S)

        def sample_mvn(key, mu, Sigma):
            L = jnp.linalg.cholesky(Sigma + 1e-6 * jnp.eye(R))
            z = jr.normal(key, shape=(R,))
            return mu + L @ z

        gamma_samp = vmap(sample_mvn)(keys_gamma, mu_gamma_post, Sigma_gamma_post)
        psi_all = vmap(lambda b, g, h: X @ b + h @ g)(beta, gamma_samp, H_all)
        keys_omega = jr.split(jr.fold_in(key_iter, 1), S)
        omega_all = vmap(lambda k, psi: _sample_omega_pg_batch(k, psi, omega_floor))(keys_omega, psi_all)
        keys_gibbs = jr.split(jr.fold_in(key_iter, 2), S)
        beta_new, _, tau2_new = _gibbs_update_vectorized(
            keys_gibbs, beta, gamma_samp, tau2, X, H_all, spikes_all, V,
            Prec_gamma_lock, gamma_samp, omega_all, tau2_intercept, a0_ard, b0_ard
        )
        beta_new = beta_new.at[:, 0].set(beta0_fixed)
        return (beta_new, gamma_samp, tau2_new), (beta_new, gamma_samp)

    keys = jr.split(key, n_iter)
    init_carry = (beta_init, gamma_init, tau2_init)
    final_carry, history = lax.scan(scan_fn, init_carry, keys)
    return final_carry, history


# ============================================================================
# Standardization and Shrinkage
# ============================================================================

def _standardize_latents(lat_reim, var_reim, scale_factors=None):
    """Standardize latents to unit variance."""
    T, twoJ = lat_reim.shape
    if scale_factors is None:
        scale_factors = np.zeros(twoJ)
        for j in range(twoJ):
            std_j = np.std(lat_reim[:, j])
            scale_factors[j] = std_j if std_j > 1e-10 else 1.0
    lat_scaled = lat_reim / scale_factors[None, :]
    var_scaled = var_reim / (scale_factors[None, :] ** 2)
    return lat_scaled, var_scaled, scale_factors


def _unstandardize_beta(beta, scale_factors):
    """Convert beta back to original units."""
    if beta.ndim == 1:
        beta_orig = beta.copy()
        beta_orig[1:] = beta[1:] / scale_factors
    else:
        beta_orig = beta.copy()
        beta_orig[:, 1:] = beta[:, 1:] / scale_factors[None, :]
    return beta_orig


def _apply_beta_shrinkage(beta_samples, burn_in_frac=0.5):
    """Apply empirical Bayes shrinkage. INTERLEAVED format."""
    n_samples, S, P = beta_samples.shape
    J = (P - 1) // 2
    burn = int(burn_in_frac * n_samples)
    post = beta_samples[burn:]

    shrinkage = np.zeros((S, J))
    beta_shrunk = np.median(post, axis=0)

    for s in range(S):
        for j in range(J):
            idx_re = 1 + 2*j
            idx_im = 2 + 2*j
            mean_re = post[:, s, idx_re].mean()
            mean_im = post[:, s, idx_im].mean()
            mean_mag_sq = mean_re**2 + mean_im**2
            var_re = post[:, s, idx_re].var()
            var_im = post[:, s, idx_im].var()
            var_total = var_re + var_im
            shrinkage[s, j] = mean_mag_sq / (mean_mag_sq + var_total + 1e-12)
            beta_shrunk[s, idx_re] *= shrinkage[s, j]
            beta_shrunk[s, idx_im] *= shrinkage[s, j]

    return shrinkage, beta_shrunk


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class InferenceConfig:
    fixed_iter: int = 100
    beta0_window: int = 100
    n_refreshes: int = 3
    inner_steps_per_refresh: int = 100
    omega_floor: float = 1e-3
    sigma_u: float = 0.05
    pg_jax: bool = True
    use_standardization: bool = True
    use_shrinkage: bool = True
    shrinkage_burn_in: float = 0.5


# ============================================================================
# Main function
# ============================================================================

def run_joint_inference_jax(
    Y_cube_block: np.ndarray,
    params0: OUParams,
    spikes: np.ndarray,
    H_hist: np.ndarray,
    all_freqs: np.ndarray,
    build_design: Callable,
    extract_band_reim_with_var: Callable,
    gibbs_update_beta_robust: Callable,
    joint_kf_rts_moments: Callable,
    em_theta_from_joint: Callable,
    config: Optional[InferenceConfig] = None,
    *,
    delta_spk: float,
    window_sec: float,
    rng_pg: np.random.Generator = np.random.default_rng(0),
    key_jax=None,
    offset_sec: float = 0.0,
):
    if key_jax is None:
        with jax.default_device(jax.devices("cpu")[0]):
            key_jax = jr.PRNGKey(0)

    if config is None:
        config = InferenceConfig()

    sigma_u = config.sigma_u
    use_standardization = config.use_standardization
    use_shrinkage = config.use_shrinkage
    shrinkage_burn_in = config.shrinkage_burn_in

    print("[JAX-V2] Starting inference...")
    print(f"[JAX-V2] Standardization={use_standardization}, Shrinkage={use_shrinkage}")

    J, M, K = Y_cube_block.shape
    single_spike_mode = (spikes.ndim == 1)

    if single_spike_mode:
        spikes_S = spikes[None, :]
        H_hist_S = H_hist[None, :, :]
    else:
        spikes_S = spikes
        H_hist_S = H_hist

    S, T_total = spikes_S.shape
    R = H_hist_S.shape[2]

    theta = OUParams(
        lam=params0.lam,
        sig_v=params0.sig_v,
        sig_eps=np.broadcast_to(params0.sig_eps, (J, M))
    )

    from src.ou_fine import kalman_filter_rts_ffbs_fine
    fine0 = kalman_filter_rts_ffbs_fine(
        Y_cube_block, theta, delta_spk=delta_spk, win_sec=window_sec, offset_sec=offset_sec
    )

    lat_reim_np, var_reim_np = extract_band_reim_with_var(
        mu_fine=np.asarray(fine0.mu)[:-1],
        var_fine=np.asarray(fine0.var)[:-1],
        coupled_bands=all_freqs, freqs_hz=all_freqs, delta_spk=delta_spk, J=J, M=M
    )

    scale_factors = None
    if use_standardization:
        lat_reim_np, var_reim_np, scale_factors = _standardize_latents(lat_reim_np, var_reim_np)
        print(f"[JAX-V2] Scale factors: [{scale_factors.min():.4f}, {scale_factors.max():.4f}]")

    lat_reim_jax = jnp.asarray(lat_reim_np)
    design_np = np.asarray(build_design(lat_reim_jax))
    T_design = min(int(design_np.shape[0]), H_hist_S.shape[1], spikes_S.shape[1])

    X_jax = jnp.array(design_np[:T_design], dtype=jnp.float64)
    V_jax = jnp.array(var_reim_np[:T_design], dtype=jnp.float64)
    spikes_jax = jnp.array(spikes_S[:, :T_design], dtype=jnp.float64)
    H_jax = jnp.array(H_hist_S[:, :T_design, :], dtype=jnp.float64)

    B = len(all_freqs)
    P = 1 + 2*B

    beta_jax = jnp.zeros((S, P), dtype=jnp.float64)
    gamma_jax = jnp.zeros((S, R), dtype=jnp.float64)
    tau2_lat_jax = jnp.ones((S, 2*B), dtype=jnp.float64)
    a0_ard, b0_ard = 1e-2, 1e-2

    mu_g, Sig_g = gamma_prior_simple(n_lags=R, strong_neg=-2.5, mild_neg=-0.5, k_mild=4, tau_gamma=1.5)
    Prec_gamma_all = jnp.array([np.linalg.pinv(Sig_g) for _ in range(S)])
    mu_gamma_all = jnp.array([mu_g for _ in range(S)])

    print(f"[JAX-V2] S={S}, T={T_design}, B={B}, R={R}, P={P}")

    trace = Trace()
    trace.theta.append(theta)
    trace.latent.append(lat_reim_jax)
    trace.fine_latent.append(np.asarray(fine0.mu))

    # Warmup
    import time
    t0 = time.time()
    key_warmup, key_jax = jr.split(key_jax)
    (beta_jax, gamma_jax, tau2_lat_jax), (beta_history, gamma_history) = _warmup_loop_scan(
        key_warmup, beta_jax, gamma_jax, tau2_lat_jax, X_jax, H_jax, spikes_jax, V_jax,
        Prec_gamma_all, mu_gamma_all, config.omega_floor, 100.0**2, a0_ard, b0_ard, config.fixed_iter
    )
    beta_jax.block_until_ready()
    print(f"[JAX-V2] Warmup: {time.time()-t0:.2f}s")

    beta_np = np.array(beta_jax)
    beta_hist_np = np.array(beta_history)
    gamma_hist_np = np.array(gamma_history)

    beta0_window = min(config.beta0_window, config.fixed_iter)
    beta0_fixed = np.median(beta_hist_np[-beta0_window:, :, 0], axis=0)
    beta_np[:, 0] = beta0_fixed
    beta_jax = jnp.array(beta_np)

    mu_g_post = np.zeros((S, R), dtype=np.float64)
    Sig_g_post = np.zeros((S, R, R), dtype=np.float64)
    Sig_g_lock = np.zeros((S, R, R), dtype=np.float64)

    for s in range(S):
        gh = gamma_hist_np[:, s, :]
        mu_s = gh.mean(axis=0)
        ctr = gh - mu_s[None, :]
        Sg = (ctr.T @ ctr) / max(gh.shape[0] - 1, 1) + 1e-6 * np.eye(R)
        mu_g_post[s] = mu_s
        Sig_g_post[s] = Sg
        Sig_g_lock[s] = np.diag(1e-6 * np.clip(np.diag(Sg), 1e-10, None))

    for i in range(config.fixed_iter):
        trace.beta.append(beta_hist_np[i])
        trace.gamma.append(gamma_hist_np[i])

    beta0_fixed_jax = jnp.array(beta0_fixed)
    mu_gamma_post_jax = jnp.array(mu_g_post)
    Sigma_gamma_post_jax = jnp.array(Sig_g_post)
    Prec_gamma_lock_jax = jnp.array([np.linalg.pinv(Sig_g_lock[s]) for s in range(S)])

    from src.state_index import StateIndex
    sidx = StateIndex(J, M)

    # Refresh passes
    for r in range(config.n_refreshes):
        print(f"[JAX-V2] Refresh {r+1}/{config.n_refreshes}")

        key_inner, key_jax = jr.split(key_jax)
        (beta_jax, gamma_jax, tau2_lat_jax), (beta_history, gamma_history) = _inner_loop_scan(
            key_inner, beta_jax, gamma_jax, tau2_lat_jax, beta0_fixed_jax, X_jax, H_jax,
            spikes_jax, V_jax, Prec_gamma_lock_jax, mu_gamma_post_jax, Sigma_gamma_post_jax,
            config.omega_floor, 100.0**2, a0_ard, b0_ard, config.inner_steps_per_refresh
        )
        beta_jax.block_until_ready()

        beta_np = np.array(beta_jax)
        gamma_np = np.array(gamma_jax)
        beta_hist_inner = np.array(beta_history)

        # Shrinkage
        if use_shrinkage:
            shrinkage_factors, beta_median = _apply_beta_shrinkage(beta_hist_inner, shrinkage_burn_in)
            if r == 0:
                print(f"[JAX-V2] Shrinkage: [{shrinkage_factors.min():.3f}, {shrinkage_factors.max():.3f}]")
        else:
            beta_median = np.median(beta_hist_inner, axis=0)

        # Rescale for KF
        if use_standardization:
            beta_median_orig = _unstandardize_beta(beta_median, scale_factors)
        else:
            beta_median_orig = beta_median

        for i in range(config.inner_steps_per_refresh):
            trace.beta.append(np.array(beta_history[i]))
            trace.gamma.append(np.array(gamma_history[i]))

        # Omega for refresh
        key_omega_refresh, key_jax = jr.split(key_jax)
        keys_gamma_refresh = jr.split(key_omega_refresh, S)

        def sample_mvn(key, mu, Sigma):
            L = jnp.linalg.cholesky(Sigma + 1e-6 * jnp.eye(R))
            return mu + L @ jr.normal(key, shape=(R,))

        gamma_refresh_jax = vmap(sample_mvn)(keys_gamma_refresh, mu_gamma_post_jax, Sigma_gamma_post_jax)
        psi_refresh_all = vmap(lambda b, g, h: X_jax @ b + h @ g)(jnp.array(beta_median), gamma_refresh_jax, H_jax)
        keys_omega_all = jr.split(jr.fold_in(key_omega_refresh, 1), S)
        omega_refresh_jax = vmap(lambda k, psi: _sample_omega_pg_batch(k, psi, config.omega_floor))(keys_omega_all, psi_refresh_all)

        omega_refresh = np.array(omega_refresh_jax)
        gamma_np = np.array(gamma_refresh_jax)

        # KF refresh
        mom = joint_kf_rts_moments(
            Y_cube=Y_cube_block, theta=theta,
            delta_spk=delta_spk, win_sec=window_sec, offset_sec=offset_sec,
            beta=beta_median_orig, gamma=gamma_np, spikes=spikes_S, omega=omega_refresh,
            coupled_bands_idx=np.arange(J, dtype=np.int64),
            freqs_for_phase=np.asarray(all_freqs, np.float64),
            sidx=sidx, H_hist=H_hist_S, sigma_u=sigma_u
        )

        lat_reim_np, var_reim_np = extract_band_reim_with_var(
            mu_fine=mom.m_s, var_fine=mom.P_s,
            coupled_bands=all_freqs, freqs_hz=all_freqs, delta_spk=delta_spk, J=J, M=M
        )

        if use_standardization:
            lat_reim_np, var_reim_np, _ = _standardize_latents(lat_reim_np, var_reim_np, scale_factors)

        lat_reim_jax = jnp.asarray(lat_reim_np)
        design_np = np.asarray(build_design(lat_reim_jax))
        X_jax = jnp.array(np.ascontiguousarray(design_np[:T_design], dtype=np.float64))
        V_jax = jnp.array(np.ascontiguousarray(var_reim_np[:T_design], dtype=np.float64))
        beta_jax = jnp.array(beta_np)
        gamma_jax = jnp.array(gamma_np)

        trace.theta.append(theta)
        gc.collect()

    trace.latent.append(lat_reim_jax)
    trace.fine_latent.append(mom.m_s)

    beta_final = np.array(beta_jax)
    gamma_final = np.array(gamma_jax)

    if use_standardization:
        beta_final = _unstandardize_beta(beta_final, scale_factors)

    print("[JAX-V2] Done!")

    if single_spike_mode:
        return beta_final[0], gamma_final[0], theta, trace
    else:
        return beta_final, gamma_final, theta, trace


