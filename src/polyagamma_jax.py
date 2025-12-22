"""
Fast Polya-Gamma sampler implementation in JAX with GPU support.

This module implements the Devroye and Saddle point methods for sampling
from the Polya-Gamma distribution PG(h, z), optimized for GPU acceleration.

References:
-----------
- Polson, Scott, and Windle (2013). "Bayesian inference for logistic models
  using Pólya–Gamma latent variables."
- Windle et al. (2014). "Sampling Polya-Gamma random variates: alternate
  and approximate techniques."
- Devroye (2009). "On exact simulation algorithms for some distributions
  related to Jacobi theta functions."
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import random
from functools import partial


# ==============================================================================
# Constants
# ==============================================================================

PI = jnp.pi
PI_2 = PI / 2.0
PI2_8 = PI * PI / 8.0
LOGPI_2 = jnp.log(PI / 2.0)
SQRT_2PI = jnp.sqrt(2.0 * PI)
T = 0.64  # Truncation point for Devroye method


# ==============================================================================
# Helper Functions
# ==============================================================================

@jax.jit
def tanh_x(x):
    """
    Compute f(x) = tanh(x) / x in the range [0, infinity).

    Uses rational polynomial approximation from Cody and Waite (1980)
    for x <= 4.95, and 1/x approximation for larger x.
    """
    # Coefficients for rational polynomial
    p0 = -0.16134119023996228053e+04
    p1 = -0.99225929672236083313e+02
    p2 = -0.96437492777225469787e+00
    q0 = 0.48402357071988688686e+04
    q1 = 0.22337720718962312926e+04
    q2 = 0.11274474380534949335e+03

    x2 = x * x

    # For x > 4.95, use 1/x approximation
    result_large = 1.0 / x

    # For x <= 4.95, use rational polynomial
    result_small = 1.0 + x2 * ((p2 * x2 + p1) * x2 + p0) / \
                             (((x2 + q2) * x2 + q1) * x2 + q0)

    return jnp.where(x > 4.95, result_large, result_small)


@jax.jit
def tan_x(x):
    """Compute f(x) = tan(x) / x."""
    return jnp.tan(x) / x


@jax.jit
def cumulant(u, log_cosh_z):
    """Compute K(t), the cumulant generating function of X."""
    def neg_case():
        return log_cosh_z - jnp.log(jnp.cosh(jnp.sqrt(-2.0 * u)))

    def pos_case():
        return log_cosh_z - jnp.log(jnp.cos(jnp.sqrt(2.0 * u)))

    def zero_case():
        return log_cosh_z

    return jax.lax.cond(
        u < 0.0,
        neg_case,
        lambda: jax.lax.cond(u > 0.0, pos_case, zero_case)
    )


@jax.jit
def cumulant_prime(u):
    """Compute K'(t) and K''(t), the first and second derivatives of the CGF."""
    s = 2.0 * u

    def compute_derivatives():
        f = jax.lax.cond(
            s < 0.0,
            lambda: tanh_x(jnp.sqrt(-s)),
            lambda: jax.lax.cond(s > 0.0, lambda: tan_x(jnp.sqrt(s)), lambda: 1.0)
        )
        fprime = f * f + (1.0 - f) / s
        return f, fprime

    return compute_derivatives()


def select_starting_guess(x):
    """Select the starting guess for Newton's method given a value of x."""
    return jnp.select(
        [x <= 0.25, x <= 0.5, x <= 1.0, x <= 1.5, x <= 2.5, x <= 4.0],
        [-9.0, -1.78, -0.147, 0.345, 0.72, 0.95],
        default=1.15
    )


@jax.jit
def newton_raphson(arg, x0, max_iter=25, atol=1e-5, rtol=1e-5):
    """Solve for the root of (f(u) = K'(t) - arg) using Newton's method."""
    def cond_fn(state):
        i, x, x_old, converged = state
        return (i < max_iter) & (~converged)

    def body_fn(state):
        i, x, x_old, converged = state
        f, fprime = cumulant_prime(x)
        fval = f - arg

        # Check convergence
        converged_abs = jnp.abs(fval) <= atol
        converged_deriv = fprime <= atol
        converged_rel = jnp.abs(x - x_old) <= atol + rtol * jnp.abs(x_old)
        converged_now = converged_abs | converged_deriv | converged_rel

        # Newton step
        x_new = jnp.where(converged_now, x, x - fval / fprime)

        return i + 1, x_new, x, converged_now

    _, x_final, _, _ = jax.lax.while_loop(cond_fn, body_fn, (0, x0, x0 + 1.0, False))

    # Get final derivative
    _, fprime_final = cumulant_prime(x_final)

    return x_final, fprime_final


@jax.jit
def pgm_lgamma(z):
    """Compute logarithm of the gamma function."""
    return jsp.special.gammaln(z)


@jax.jit
def log_norm_cdf(x):
    """Compute the logarithm of the standard normal CDF."""
    return jnp.log1p(-0.5 * jsp.special.erfc(x / jnp.sqrt(2.0)))


# ==============================================================================
# Incomplete Gamma Functions
# ==============================================================================

@jax.jit
def confluent_x_smaller(p, x, max_iter=100):
    """Compute G(p, x) using continued fraction for x <= p."""
    eps = jnp.finfo(jnp.float64).eps
    tiny = jnp.finfo(jnp.float64).tiny

    a = jnp.float64(1.0)
    b = jnp.float64(p)
    r = jnp.float64(-(p - 1.0) * x)
    s = jnp.float64(0.5 * x)

    f = a / b
    c = a / tiny
    d = jnp.float64(1.0) / b

    def cond_fn(state):
        n, f, c, d, converged = state
        return (n < max_iter) & (~converged)

    def body_fn(state):
        n, f, c, d, _ = state

        # Compute a_n
        a_n = jnp.where(n & 1, s * jnp.float64(n - 1), r - s * jnp.float64(n))
        b_n = b + jnp.float64(n - 2)

        # Modified Lentz step
        c_new = b_n + a_n / jnp.maximum(c, tiny)
        d_new = jnp.float64(1.0) / jnp.maximum(a_n * d + b_n, tiny)
        delta = c_new * d_new
        f_new = f * delta

        converged = jnp.abs(delta - jnp.float64(1.0)) < eps

        return n + 1, f_new, c_new, d_new, converged

    _, f_final, _, _, _ = jax.lax.while_loop(
        cond_fn, body_fn, (2, f, c, d, False)
    )

    return f_final


@jax.jit
def confluent_p_smaller(p, x, max_iter=100):
    """Compute G(p, x) using continued fraction for x > p."""
    eps = jnp.finfo(jnp.float64).eps
    tiny = jnp.finfo(jnp.float64).tiny

    a = jnp.float64(1.0)
    b = jnp.float64(x - p + 1.0)

    f = a / b
    c = a / tiny
    d = jnp.float64(1.0) / b

    def cond_fn(state):
        n, f, c, d, converged = state
        return (n < max_iter) & (~converged)

    def body_fn(state):
        n, f, c, d, _ = state

        # Compute a_n and b_n
        a_n = jnp.float64(n) * (p - jnp.float64(n))
        b_n = b + jnp.float64(2.0 * (n - 1))

        # Modified Lentz step
        c_new = b_n + a_n / jnp.maximum(c, tiny)
        d_new = jnp.float64(1.0) / jnp.maximum(a_n * d + b_n, tiny)
        delta = c_new * d_new
        f_new = f * delta

        converged = jnp.abs(delta - jnp.float64(1.0)) < eps

        return n + 1, f_new, c_new, d_new, converged

    _, f_final, _, _, _ = jax.lax.while_loop(
        cond_fn, body_fn, (1, f, c, d, False)
    )

    return f_final


@jax.jit
def upper_incomplete_gamma(p, x, normalized=False):
    """Compute the upper incomplete gamma function using continued fractions."""
    max_exp = 88.7228

    # Use continued fractions for all cases
    def general_case_normalized():
        x_smaller = p >= x
        f = jax.lax.cond(
            x_smaller,
            lambda: confluent_x_smaller(p, x),
            lambda: confluent_p_smaller(p, x)
        )
        out = f * jnp.exp(-x + p * jnp.log(x) - pgm_lgamma(p))
        return jnp.where(x_smaller, 1.0 - out, out)

    def general_case_unnormalized():
        x_smaller = p >= x
        f = jax.lax.cond(
            x_smaller,
            lambda: confluent_x_smaller(p, x),
            lambda: confluent_p_smaller(p, x)
        )
        lgam = pgm_lgamma(p)
        exp_lgam = jnp.where(lgam >= max_exp, jnp.exp(max_exp), jnp.exp(lgam))
        arg = jnp.clip(-x + p * jnp.log(x) - lgam, -max_exp, max_exp)

        return jax.lax.cond(
            x_smaller,
            lambda: (1.0 - f * jnp.exp(arg)) * exp_lgam,
            lambda: f * jnp.exp(arg)
        )

    return jax.lax.cond(normalized, general_case_normalized, general_case_unnormalized)


# ==============================================================================
# Devroye Sampler
# ==============================================================================

@jax.jit
def piecewise_coef(n, x, logx, z, k):
    """Compute a_n(x|t), the nth term of the alternating sum S_n(x|t)."""
    def large_x_case():
        b = PI * (n + 0.5)
        return b * jnp.exp(-0.5 * x * b * b)

    def small_x_case():
        a = n + 0.5
        return jnp.exp(-1.5 * (LOGPI_2 + logx) - 2.0 * a * a / x) * (PI * a)

    return jax.lax.cond(x > T, large_x_case, small_x_case)


@jax.jit
def random_right_bounded_invgauss(key, z, z2, k, max_iter=10000):
    """Sample from an Inverse-Gaussian(1/z, 1) truncated on {x | x < T}."""
    def small_z_case(key):
        """Case when z < 1.5625 (i.e., 1/z < T)"""
        def cond_fn(state):
            key, x, accepted, iter_count = state
            return (~accepted) & (iter_count < max_iter)

        def body_fn(state):
            key, x, _, iter_count = state

            # Sample two exponentials
            key, subkey1, subkey2, subkey3 = random.split(key, 4)
            e1 = random.exponential(subkey1)
            e2 = random.exponential(subkey2)

            # Check first condition
            accept1 = e1 * e1 <= 3.125 * e2  # 2 / T = 3.125

            # Compute x
            x_prop = T / ((1.0 + T * e1) ** 2)

            # Check second condition (if z > 0)
            u = random.uniform(subkey3)
            accept2 = (z <= 0.0) | (u <= jnp.exp(-0.5 * z2 * x_prop))

            accepted = accept1 & accept2
            x_new = jnp.where(accepted, x_prop, x)

            return key, x_new, accepted, iter_count + 1

        _, x_final, _, _ = jax.lax.while_loop(
            cond_fn, body_fn, (key, 0.0, False, 0)
        )
        return key, x_final

    def large_z_case(key):
        """Case when z >= 1.5625 (i.e., 1/z >= T)"""
        def cond_fn(state):
            key, x, accepted, iter_count = state
            return (~accepted) & (iter_count < max_iter)

        def body_fn(state):
            key, x, _, iter_count = state

            key, subkey1, subkey2 = random.split(key, 3)
            y = random.normal(subkey1)
            u = random.uniform(subkey2)

            w = (z + 0.5 * y * y) / z2
            x_prop = w - jnp.sqrt(jnp.abs(w * w - 1.0 / z2))

            # Adjust if needed
            x_prop = jnp.where(u * (1.0 + x_prop * z) > 1.0, 1.0 / (x_prop * z2), x_prop)

            accepted = x_prop < T
            x_new = jnp.where(accepted, x_prop, x)

            return key, x_new, accepted, iter_count + 1

        _, x_final, _, _ = jax.lax.while_loop(
            cond_fn, body_fn, (key, 0.0, False, 0)
        )
        return key, x_final

    return jax.lax.cond(z < 1.5625, small_z_case, large_z_case, key)


@jax.jit
def random_jacobi_star(key, z, z2, k, proposal_probability, max_iter=10000):
    """Generate a random sample J*(1, z) using the Devroye method."""
    def cond_fn(state):
        key, x, accepted, iter_count = state
        return (~accepted) & (iter_count < max_iter)

    def body_fn(state):
        key, x, _, iter_count = state

        # Choose proposal
        key, subkey1, subkey2 = random.split(key, 3)
        u_prop = random.uniform(subkey1)
        use_invgauss = u_prop < proposal_probability

        # Sample from inverse-Gaussian or exponential
        def sample_invgauss(key):
            return random_right_bounded_invgauss(key, z, z2, k)

        def sample_exponential(key):
            key, subkey = random.split(key)
            x_prop = T + random.exponential(subkey) / k
            return key, x_prop

        key, x_prop = jax.lax.cond(use_invgauss, sample_invgauss, sample_exponential, key)
        logx_prop = jnp.log(x_prop)

        # Acceptance-rejection with alternating sum
        s0 = piecewise_coef(0, x_prop, logx_prop, z, k)
        key, subkey = random.split(key)
        u = random.uniform(subkey) * s0

        s1 = s0 - piecewise_coef(1, x_prop, logx_prop, z, k)
        accept_early = u <= s1

        # If not accepted early, continue alternating sum
        def alternating_sum(carry, i):
            s, u, accepted, sign = carry
            coef = piecewise_coef(i, x_prop, logx_prop, z, k)
            s_new = s + sign * coef
            sign_new = -sign

            # Accept if u <= s and sign is negative
            # Reject if u > s and sign is positive
            accept_now = (u <= s_new) & (sign < 0)
            reject_now = (u > s_new) & (sign > 0)

            accepted_new = accepted | accept_now | reject_now

            return (s_new, u, accepted_new, sign_new), accept_now

        # Run alternating sum for a few more terms if needed
        (s_final, _, accepted_sum, _), accepts = jax.lax.scan(
            alternating_sum, (s1, u, False, 1.0), jnp.arange(2, 12)
        )

        # If still not decided after 12 terms, accept (very rare)
        accepted_final = accept_early | jnp.any(accepts) | (~accepted_sum)

        x_new = jnp.where(accepted_final, x_prop, x)

        return key, x_new, accepted_final, iter_count + 1

    key, x_final, _, _ = jax.lax.while_loop(
        cond_fn, body_fn, (key, 0.0, False, 0)
    )

    return key, x_final


def sample_pg_devroye_single(key, h, z):
    """
    Sample from PG(h, z) using the Devroye method (for integer h).

    Parameters
    ----------
    key : PRNGKey
        Random key
    h : int
        Shape parameter (must be positive integer)
    z : float
        Exponential tilting parameter

    Returns
    -------
    float
        Sample from PG(h, z)
    """
    z_half = 0.5 * jnp.abs(z)

    # Set sampling parameters
    z2 = z_half * z_half
    k = PI2_8 + 0.5 * z2

    # Compute proposal probability
    a = 0.8838834764831844  # 1 / sqrt(2 * T)
    t_two = 0.565685424949238  # sqrt(T / 2)
    b = z_half * t_two
    ez = jnp.exp(z_half)

    p = jsp.special.erfc(a - b) / ez + jsp.special.erfc(a + b) * ez
    q = PI_2 * jnp.exp(-k * T) / k
    proposal_probability_nz = p / (p + q)
    proposal_probability = jnp.where(z_half > 0, proposal_probability_nz, 0.4223027567786595)

    # Sum h independent samples of J*(1, z)
    h_int = int(h)

    def body_fn(i, carry):
        key, result = carry
        key, subkey = random.split(key)
        key, sample = random_jacobi_star(subkey, z_half, z2, k, proposal_probability)
        return key, result + sample

    _, result = jax.lax.fori_loop(0, h_int, body_fn, (key, 0.0))

    return 0.25 * result


# ==============================================================================
# Saddle Point Sampler
# ==============================================================================

@jax.jit
def invgauss_logcdf(x, mu, lam):
    """Calculate the log CDF of an Inverse-Gaussian distribution."""
    qm = x / mu
    tm = mu / lam
    r = jnp.sqrt(x / lam)
    a = log_norm_cdf((qm - 1.0) / r)
    b = 2.0 / tm + log_norm_cdf(-(qm + 1.0) / r)

    return a + jnp.log1p(jnp.exp(b - a))


@jax.jit
def random_left_bounded_gamma(key, a, b, t):
    """Sample from Gamma(a, rate=b) truncated on {x | x > t}."""
    def a_greater_1(key):
        """Dagpunar (1978) algorithm for a > 1"""
        bt = t * b
        amin1 = a - 1.0
        bmina = bt - a
        c0 = 0.5 * (bmina + jnp.sqrt(bmina * bmina + 4.0 * bt)) / bt
        one_minus_c0 = 1.0 - c0
        log_m = amin1 * (jnp.log(amin1 / one_minus_c0) - 1.0)

        max_iter = 1000

        def cond_fn(state):
            key, x, accepted, iter_count = state
            return (~accepted) & (iter_count < max_iter)

        def body_fn(state):
            key, x, _, iter_count = state

            key, subkey1, subkey2 = random.split(key, 3)
            x_prop = bt + random.exponential(subkey1) / c0
            threshold = amin1 * jnp.log(x_prop) - x_prop * one_minus_c0 - log_m

            u = random.uniform(subkey2)
            accepted = jnp.log(1.0 - u) > threshold
            x_new = jnp.where(accepted, x_prop, x)

            return key, x_new, accepted, iter_count + 1

        _, x_final, _, _ = jax.lax.while_loop(
            cond_fn, body_fn, (key, 0.0, False, 0)
        )
        return key, t * (x_final / bt)

    def a_equals_1(key):
        """Simple exponential shift for a == 1"""
        key, subkey = random.split(key)
        return key, t + random.exponential(subkey) / b

    def a_less_1(key):
        """Philippe (1997) algorithm A4 for a < 1"""
        amin1 = a - 1.0
        tb = t * b

        max_iter = 1000

        def cond_fn(state):
            key, x, accepted, iter_count = state
            return (~accepted) & (iter_count < max_iter)

        def body_fn(state):
            key, x, _, iter_count = state

            key, subkey1, subkey2 = random.split(key, 3)
            x_prop = 1.0 + random.exponential(subkey1) / tb

            u = random.uniform(subkey2)
            accepted = jnp.log(1.0 - u) > amin1 * jnp.log(x_prop)
            x_new = jnp.where(accepted, x_prop, x)

            return key, x_new, accepted, iter_count + 1

        _, x_final, _, _ = jax.lax.while_loop(
            cond_fn, body_fn, (key, 0.0, False, 0)
        )
        return key, t * x_final

    # Select algorithm based on a
    return jax.lax.cond(
        a > 1.0,
        a_greater_1,
        lambda key: jax.lax.cond(a == 1.0, a_equals_1, a_less_1, key),
        key
    )


@jax.jit
def saddle_point(x, h, z_half, log_cosh_z, sqrt_h2pi):
    """Compute the saddle point estimate at x."""
    x0 = select_starting_guess(x)
    u, fprime = newton_raphson(x, x0)
    t = u + 0.5 * z_half * z_half

    return jnp.exp(h * (cumulant(u, log_cosh_z) - t * x)) * sqrt_h2pi / jnp.sqrt(fprime)


@jax.jit
def bounding_kernel(x, xc, logxc, h, left_tangent_slope, left_tangent_intercept,
                   right_tangent_slope, right_tangent_intercept,
                   left_kernel_coef, right_kernel_coef):
    """Compute k(x|h,z), the bounding kernel of the saddle point approximation."""
    def right_case():
        point = right_tangent_slope * x + right_tangent_intercept
        return jnp.exp(h * (logxc + point) + (h - 1.0) * jnp.log(x)) * right_kernel_coef

    def left_case():
        point = left_tangent_slope * x + left_tangent_intercept
        return jnp.exp(0.5 * h * (1.0 / xc - 1.0 / x) + h * point - 1.5 * jnp.log(x)) * left_kernel_coef

    return jax.lax.cond(x > xc, right_case, left_case)


def sample_pg_saddle_single(key, h, z):
    """
    Sample from PG(h, z) using the Saddle point method.

    Parameters
    ----------
    key : PRNGKey
        Random key
    h : float
        Shape parameter (h >= 1)
    z : float
        Exponential tilting parameter

    Returns
    -------
    float
        Sample from PG(h, z)
    """
    z_half = 0.5 * jnp.abs(z)

    # Set sampling parameters
    xl = jnp.where(z_half > 0, tanh_x(z_half), 1.0)
    logxl = jnp.where(z_half > 0, jnp.log(xl), 0.0)
    half_z2 = jnp.where(z_half > 0, 0.5 * z_half * z_half, 0.0)
    log_cosh_z = jnp.where(z_half > 0, jnp.log(jnp.cosh(z_half)), 0.0)

    xc = 2.75 * xl
    xr = 3.0 * xl

    xc_inv = 1.0 / xc
    xl_inv = 1.0 / xl
    ul = -half_z2

    # Compute tangent parameters
    x0_r = select_starting_guess(xr)
    ur, fprime_r = newton_raphson(xr, x0_r)

    x0_c = select_starting_guess(xc)
    uc, fprime_c = newton_raphson(xc, x0_c)

    tr = ur + half_z2

    left_tangent_slope = -0.5 * (xl_inv * xl_inv)
    left_tangent_intercept = cumulant(ul, log_cosh_z) - 0.5 * xc_inv + xl_inv
    logxc = jnp.log(2.75) + logxl
    right_tangent_slope = -tr - 1.0 / xr
    right_tangent_intercept = cumulant(ur, log_cosh_z) + 1.0 - jnp.log(3.0) - logxl + logxc

    alpha_r = fprime_c * (xc_inv * xc_inv)
    alpha_l = xc_inv * alpha_r

    sqrt_alpha = 1.0 / jnp.sqrt(alpha_l)
    sqrt_h2pi = jnp.sqrt(h / (2.0 * PI))
    left_kernel_coef = sqrt_h2pi * sqrt_alpha
    right_kernel_coef = sqrt_h2pi / jnp.sqrt(alpha_r)

    # Compute proposal probabilities
    sqrt_rho = jnp.sqrt(-2.0 * left_tangent_slope)
    sqrt_rho_inv = 1.0 / sqrt_rho
    mu2 = sqrt_rho_inv * sqrt_rho_inv

    p = jnp.exp(h * (0.5 / xc + left_tangent_intercept - sqrt_rho) +
                invgauss_logcdf(xc, sqrt_rho_inv, h)) * sqrt_alpha

    hrho = -h * right_tangent_slope
    q = upper_incomplete_gamma(h, hrho * xc, False) * right_kernel_coef * \
        jnp.exp(h * (right_tangent_intercept - jnp.log(hrho)))

    proposal_probability = p / (p + q)

    # Acceptance-rejection sampling
    max_iter = 1000

    def cond_fn(state):
        key, x, accepted, iter_count = state
        return (~accepted) & (iter_count < max_iter)

    def body_fn(state):
        key, x_prev, _, iter_count = state

        # Choose proposal
        key, subkey1 = random.split(key)
        u_prop = random.uniform(subkey1)
        use_invgauss = u_prop < proposal_probability

        # Sample from inverse-Gaussian or gamma
        def sample_invgauss(key):
            max_iter_ig = 1000

            def cond_fn_ig(state):
                key, x, accepted, iter_count_ig = state
                return (~accepted) & (iter_count_ig < max_iter_ig)

            def body_fn_ig(state):
                key, x, _, iter_count_ig = state

                key, subkey1, subkey2 = random.split(key, 3)
                y = random.normal(subkey1)
                w = sqrt_rho_inv + 0.5 * mu2 * y * y / h
                x_prop = w - jnp.sqrt(jnp.abs(w * w - mu2))

                u = random.uniform(subkey2)
                x_prop = jnp.where(u * (1.0 + x_prop * sqrt_rho) > 1.0, mu2 / x_prop, x_prop)

                accepted = x_prop < xc
                x_new = jnp.where(accepted, x_prop, x)

                return key, x_new, accepted, iter_count_ig + 1

            _, x_final, _, _ = jax.lax.while_loop(
                cond_fn_ig, body_fn_ig, (key, 0.0, False, 0)
            )
            return key, x_final

        def sample_gamma(key):
            return random_left_bounded_gamma(key, h, hrho, xc)

        key, x_prop = jax.lax.cond(use_invgauss, sample_invgauss, sample_gamma, key)

        # Acceptance test
        key, subkey = random.split(key)
        u = random.uniform(subkey)

        bk = bounding_kernel(x_prop, xc, logxc, h, left_tangent_slope, left_tangent_intercept,
                           right_tangent_slope, right_tangent_intercept,
                           left_kernel_coef, right_kernel_coef)
        sp = saddle_point(x_prop, h, z_half, log_cosh_z, sqrt_h2pi)

        accepted = u * bk <= sp
        x_new = jnp.where(accepted, x_prop, x_prev)

        return key, x_new, accepted, iter_count + 1

    key, x_final, _, _ = jax.lax.while_loop(
        cond_fn, body_fn, (key, 0.0, False, 0)
    )

    return 0.25 * h * x_final


# ==============================================================================
# Normal Approximation
# ==============================================================================

@jax.jit
def sample_pg_normal_single(key, h, z):
    """
    Sample from PG(h, z) using Normal approximation (for large h > 50).

    Uses truncated normal to ensure samples are always positive, as required
    for Polya-Gamma distributions.
    """
    def z_zero_case():
        mean = 0.25 * h
        stdev = jnp.sqrt(h / 24.0)
        return mean, stdev

    def z_nonzero_case():
        x = jnp.tanh(0.5 * z)
        mean = 0.5 * h * x / z
        # Ensure variance is non-negative (can be negative due to numerical issues)
        variance = jnp.maximum(0.0, 0.25 * h * (jnp.sinh(z) - z) * (1.0 - x * x) / (z * z * z))
        stdev = jnp.sqrt(variance)
        return mean, stdev

    mean, stdev = jax.lax.cond(z == 0.0, z_zero_case, z_nonzero_case)

    # Sample from truncated normal to ensure positivity
    max_iter = 100

    def cond_fn(state):
        key, sample, accepted, iter_count = state
        return (~accepted) & (iter_count < max_iter)

    def body_fn(state):
        key, sample, _, iter_count = state
        key, subkey = random.split(key)
        sample_prop = mean + random.normal(subkey) * stdev
        accepted = sample_prop > 0.0
        sample_new = jnp.where(accepted, sample_prop, sample)
        return key, sample_new, accepted, iter_count + 1

    _, sample_final, _, _ = jax.lax.while_loop(
        cond_fn, body_fn, (key, 0.0, False, 0)
    )

    # If we still don't have a positive sample (very unlikely for large h),
    # fall back to using the mean
    return jnp.where(sample_final > 0.0, sample_final, mean)
