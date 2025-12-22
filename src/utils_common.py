"""
utils_common.py - Shared utilities for spike-field joint inference.

This module contains utilities that are used across multiple modules:
- Block-to-fine time mapping functions
- Array normalization functions
- Beta coefficient layout conversion functions
"""
from __future__ import annotations
from typing import Tuple
import numpy as np
import jax.numpy as jnp

EPS = 1e-12


# ==============================================================================
# Block-to-Fine Time Mapping
# ==============================================================================

def centres_from_win(K: int, win_sec: float, offset_sec: float) -> np.ndarray:
    """
    Compute block centre times from window parameters.

    Parameters
    ----------
    K : int
        Number of blocks/windows
    win_sec : float
        Window length in seconds (also the hop/stride between windows)
    offset_sec : float
        Time offset to the first window start

    Returns
    -------
    centres_sec : ndarray, shape (K,)
        Centre times in seconds for each block
    """
    return offset_sec + np.arange(K, dtype=float) * float(win_sec)


def map_blocks_to_fine(centres_sec: np.ndarray, delta_spk: float, T_f: int) -> np.ndarray:
    """
    Map block centres to fine-grid indices.

    Parameters
    ----------
    centres_sec : ndarray, shape (K,)
        Block centre times in seconds
    delta_spk : float
        Fine-grid time step in seconds
    T_f : int
        Number of fine-grid time points

    Returns
    -------
    idx : ndarray, shape (K,), dtype int64
        Fine-grid indices for each block centre, clipped to [0, T_f-1]
    """
    idx = np.round(np.asarray(centres_sec) / float(delta_spk)).astype(np.int64)
    return np.clip(idx, 0, T_f - 1)


def build_t2k(centres_sec: np.ndarray, delta_spk: float, T_f: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (T_f, max_k_per_t) index table for block-to-fine mapping.

    Creates a lookup table that maps each fine-grid time point to the
    block indices that fall at that time. This is needed for the KF
    observation model where blocks provide observations at specific times.

    Parameters
    ----------
    centres_sec : ndarray, shape (K,)
        Block centre times in seconds
    delta_spk : float
        Fine-grid time step in seconds
    T_f : int
        Number of fine-grid time points

    Returns
    -------
    t2k : ndarray, shape (T_f, max_k_per_t), dtype int32
        Lookup table mapping fine-grid times to block indices.
        Unused slots are filled with -1.
    kcount : ndarray, shape (T_f,), dtype int32
        Number of blocks at each fine-grid time point
    """
    t_idx = map_blocks_to_fine(centres_sec, delta_spk, T_f)
    T_f = int(T_f)

    # Build buckets: for each fine time, which blocks map to it
    buckets = [[] for _ in range(T_f)]
    for k, t in enumerate(t_idx):
        buckets[int(t)].append(k)

    kcount = np.array([len(b) for b in buckets], dtype=np.int32)
    max_k = int(kcount.max(initial=0))

    # Create padded lookup table
    t2k = np.full((T_f, max_k), -1, dtype=np.int32)
    for t in range(T_f):
        row = buckets[t]
        if row:
            t2k[t, :len(row)] = np.asarray(row, dtype=np.int32)

    return t2k, kcount


# ==============================================================================
# Array Normalization
# ==============================================================================

def normalize_Y_to_RJMK(Y: np.ndarray, J: int, M: int) -> np.ndarray:
    """
    Normalize Y array to (R, J, M, K) shape.

    Handles both (R, J, M, K) and (R, M, J, K) input formats.

    Parameters
    ----------
    Y : ndarray, shape (R, ?, ?, K)
        Input array with trial-frequency-taper-time dimensions
    J : int
        Number of frequency bands
    M : int
        Number of tapers

    Returns
    -------
    Y_out : ndarray, shape (R, J, M, K)
        Normalized array with consistent dimension ordering
    """
    Y = np.asarray(Y)
    if Y.ndim != 4:
        raise ValueError("Y must be 4D: (R,J,M,K) or (R,M,J,K)")

    R, A, B, K = Y.shape
    if (A, B) == (J, M):
        return Y
    if (A, B) == (M, J):
        return np.swapaxes(Y, 1, 2)

    raise ValueError(f"Cannot normalize shape {Y.shape} to (R,{J},{M},K)")


def normalize_Y_to_RMJK_jax(Y: jnp.ndarray) -> jnp.ndarray:
    """
    Ensure (R, M, J, K) ordering from either (R, M, J, K) or (R, J, M, K).

    JAX version of the normalizer used in EM algorithms.

    Parameters
    ----------
    Y : jax.Array, shape (R, ?, ?, K)
        Input array

    Returns
    -------
    Y_out : jax.Array, shape (R, M, J, K)
        Array with (R, M, J, K) ordering
    """
    if Y.ndim != 4:
        raise ValueError("Y must have 4 dims: (R,M,J,K) or (R,J,M,K)")

    # Heuristic: M (tapers) is typically <= 16, J (bands) is typically larger
    # If dim 1 > 16 or dim 2 < dim 1, assume it's (R,J,M,K) and swap
    if Y.shape[1] > 16 or Y.shape[2] < Y.shape[1]:
        return jnp.swapaxes(Y, 1, 2)
    return Y


# ==============================================================================
# Beta Layout Conversion
# ==============================================================================

def separated_to_interleaved(beta_sep: np.ndarray) -> np.ndarray:
    """
    Convert beta from SEPARATED layout to INTERLEAVED layout.

    SEPARATED (used by simulation & Gibbs sampler):
        [beta_0, betaR_0, betaR_1, ..., betaR_{J-1}, betaI_0, betaI_1, ..., betaI_{J-1}]

    INTERLEAVED (expected by joint_kf_rts_moments):
        [beta_0, betaR_0, betaI_0, betaR_1, betaI_1, ..., betaR_{J-1}, betaI_{J-1}]

    Parameters
    ----------
    beta_sep : ndarray, shape (S, 1+2J)
        Beta coefficients in separated layout

    Returns
    -------
    beta_int : ndarray, shape (S, 1+2J)
        Beta coefficients in interleaved layout
    """
    S, P = beta_sep.shape
    J = (P - 1) // 2

    beta_int = np.zeros_like(beta_sep)
    beta_int[:, 0] = beta_sep[:, 0]  # beta_0 (intercept)

    for j in range(J):
        beta_int[:, 1 + 2*j] = beta_sep[:, 1 + j]          # betaR_j
        beta_int[:, 1 + 2*j + 1] = beta_sep[:, 1 + J + j]  # betaI_j

    return beta_int


def interleaved_to_separated(beta_int: np.ndarray) -> np.ndarray:
    """
    Convert beta from INTERLEAVED layout to SEPARATED layout.

    INTERLEAVED:
        [beta_0, betaR_0, betaI_0, betaR_1, betaI_1, ..., betaR_{J-1}, betaI_{J-1}]

    SEPARATED:
        [beta_0, betaR_0, betaR_1, ..., betaR_{J-1}, betaI_0, betaI_1, ..., betaI_{J-1}]

    Parameters
    ----------
    beta_int : ndarray, shape (S, 1+2J)
        Beta coefficients in interleaved layout

    Returns
    -------
    beta_sep : ndarray, shape (S, 1+2J)
        Beta coefficients in separated layout
    """
    S, P = beta_int.shape
    J = (P - 1) // 2

    beta_sep = np.zeros_like(beta_int)
    beta_sep[:, 0] = beta_int[:, 0]  # beta_0 (intercept)

    for j in range(J):
        beta_sep[:, 1 + j] = beta_int[:, 1 + 2*j]          # betaR_j
        beta_sep[:, 1 + J + j] = beta_int[:, 1 + 2*j + 1]  # betaI_j

    return beta_sep
