# src/upsample_ct_single_fine.py
"""
Upsample single-trial CT-SSMT latents to fine (spike-resolution) grid.

Adapted from upsample_ct_hier_fine.py for single-trial case (no X/D decomposition).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

EPS = 1e-12


@dataclass
class UpsampleSingleResult:
    """Result of single-trial upsampling."""
    Z_mean: np.ndarray      # (J, M, Tf) complex
    Z_var: np.ndarray       # (J, M, Tf) real
    t_idx_of_k: np.ndarray  # (K,) block center indices
    centres_sec: np.ndarray # (K,) block center times


def centres_from_win(K: int, win_sec: float, offset_sec: float) -> np.ndarray:
    """Compute block centre times."""
    return offset_sec + np.arange(K, dtype=float) * float(win_sec)


def map_blocks_to_fine(centres_sec: np.ndarray, delta_spk: float, T_f: int) -> np.ndarray:
    """Map block centres to fine-grid indices."""
    idx = np.round(np.asarray(centres_sec) / float(delta_spk)).astype(np.int64)
    return np.clip(idx, 0, T_f - 1)


def build_t2k(centres_sec: np.ndarray, delta_spk: float, T_f: int):
    """Build fine-time to block-index lookup table."""
    t_idx = map_blocks_to_fine(centres_sec, delta_spk, T_f)
    T_f = int(T_f)

    buckets = [[] for _ in range(T_f)]
    for k, t in enumerate(t_idx):
        buckets[int(t)].append(k)

    kcount = np.array([len(b) for b in buckets], dtype=np.int32)
    max_k = int(kcount.max(initial=1))

    t2k = np.full((T_f, max_k), -1, dtype=np.int32)
    for t in range(T_f):
        row = buckets[t]
        if row:
            t2k[t, :len(row)] = np.asarray(row, dtype=np.int32)

    return t2k, kcount


def _smooth_fine_ou_complex(phi, q, R, yk, t2k, kcount, z0, P0):
    """
    Kalman smoother on fine grid with block observations.
    
    Pure Python/NumPy implementation (no Numba required).
    """
    Tf = t2k.shape[0]
    
    m_p = np.empty(Tf, np.complex128)
    P_p = np.empty(Tf, np.float64)
    m_f = np.empty(Tf, np.complex128)
    P_f = np.empty(Tf, np.float64)
    
    z = complex(z0)
    P = float(P0)
    
    for t in range(Tf):
        # Predict
        z_pred = phi * z
        P_pred = phi * phi * P + q
        
        z_upd = z_pred
        P_upd = P_pred
        
        # Update with any observations at this time
        kc = kcount[t]
        for i in range(kc):
            k = t2k[t, i]
            if k < 0:
                break
            y = yk[k]
            S = P_upd + R
            K = P_upd / max(S, 1e-18)
            z_upd = z_upd + K * (y - z_upd)
            P_upd = max((1.0 - K) * P_upd, 1e-16)
        
        m_p[t] = z_pred
        P_p[t] = P_pred
        m_f[t] = z_upd
        P_f[t] = P_upd
        z = z_upd
        P = P_upd
    
    # RTS smoother backward pass
    m_s = np.empty(Tf, np.complex128)
    P_s = np.empty(Tf, np.float64)
    m_s[-1] = m_f[-1]
    P_s[-1] = P_f[-1]
    
    for t in range(Tf - 2, -1, -1):
        denom = max(P_p[t + 1], 1e-16)
        Jt = (P_f[t] * phi) / denom
        m_s[t] = m_f[t] + Jt * (m_s[t + 1] - m_p[t + 1])
        P_s[t] = max(P_f[t] + Jt * Jt * (P_s[t + 1] - P_p[t + 1]), 1e-16)
    
    return m_s, P_s


def upsample_ct_single_fine(
    *,
    Y: np.ndarray,           # (J, M, K) complex
    res,                      # EMSingleResult
    delta_spk: float,
    win_sec: float,
    offset_sec: float = 0.0,
    T_f: Optional[int] = None,
) -> UpsampleSingleResult:
    """
    Upsample single-trial CT-SSMT latents to fine grid.
    
    Parameters
    ----------
    Y : (J, M, K) complex
        Multitaper spectrogram
    res : EMSingleResult
        EM result with lam, sigv, sig_eps, Z_mean, Z_var
    delta_spk : float
        Fine grid time step (seconds)
    win_sec : float
        Block window duration (seconds)
    offset_sec : float
        Time offset to first block
    T_f : int, optional
        Fine grid length (auto-computed if None)
        
    Returns
    -------
    UpsampleSingleResult
        Z_mean: (J, M, Tf) complex smoothed means
        Z_var: (J, M, Tf) real smoothed variances
    """
    Y = np.asarray(Y)
    if Y.ndim != 3:
        raise ValueError(f"Y must be (J, M, K), got {Y.shape}")
    
    J, M, K = Y.shape
    
    # Extract EM parameters
    lam = np.asarray(res.lam, float)
    sigv = np.asarray(res.sigv, float)
    sig_eps = np.asarray(res.sig_eps, float)
    
    # Handle shape - could be (J,M) or (J,) depending on EM output
    if lam.ndim == 1:
        lam = np.broadcast_to(lam[:, None], (J, M))
        sigv = np.broadcast_to(sigv[:, None], (J, M))
        sig_eps = np.broadcast_to(sig_eps[:, None], (J, M))
    
    assert lam.shape == (J, M), f"lam shape mismatch: {lam.shape} vs ({J}, {M})"
    
    # Get initial states if available
    if hasattr(res, 'x0') and res.x0 is not None:
        x0 = np.asarray(res.x0, np.complex128)
        if x0.ndim == 1:
            x0 = np.broadcast_to(x0[:, None], (J, M))
    else:
        x0 = np.zeros((J, M), dtype=np.complex128)
    
    if hasattr(res, 'P0') and res.P0 is not None:
        P0 = np.asarray(res.P0, float)
        if P0.ndim == 1:
            P0 = np.broadcast_to(P0[:, None], (J, M))
    else:
        P0 = (sigv**2 / np.maximum(2.0 * lam, EPS))
    
    # Fine grid setup
    centres_sec = centres_from_win(K, win_sec, offset_sec)
    t_end = centres_sec[-1] + 0.5 * float(win_sec)
    
    if T_f is None:
        T_f = int(round(t_end / float(delta_spk)))
    assert T_f > 0, "T_f must be positive"
    
    t_idx_of_k = map_blocks_to_fine(centres_sec, delta_spk, T_f)
    t2k, kcount = build_t2k(centres_sec, delta_spk, T_f)
    
    # OU parameters at fine step
    phi = np.exp(-lam * float(delta_spk))
    q = (sigv**2) * (1.0 - np.exp(-2.0 * lam * float(delta_spk))) / np.maximum(2.0 * lam, EPS)
    R = sig_eps**2
    
    # Upsample each (j, m) chain
    Z_mean = np.zeros((J, M, T_f), dtype=np.complex128)
    Z_var = np.zeros((J, M, T_f), dtype=np.float64)
    
    Y_complex = np.asarray(Y, dtype=np.complex128)
    
    for j in range(J):
        for m in range(M):
            xs, Ps = _smooth_fine_ou_complex(
                float(phi[j, m]),
                float(q[j, m]),
                float(R[j, m]),
                Y_complex[j, m, :],
                t2k, kcount,
                complex(x0[j, m]),
                float(P0[j, m])
            )
            Z_mean[j, m] = xs
            Z_var[j, m] = Ps
    
    return UpsampleSingleResult(
        Z_mean=Z_mean,
        Z_var=Z_var,
        t_idx_of_k=t_idx_of_k,
        centres_sec=centres_sec,
    )
