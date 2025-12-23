from dataclasses import dataclass
import numpy as np
from typing import NamedTuple, Optional

from src.utils_common import centres_from_win, build_t2k

class JointMoments(NamedTuple):
    m_f:    np.ndarray  # (T_f, d) filtered means
    P_f:    np.ndarray  # (T_f, d) filtered diag variances
    P_pred: np.ndarray  # (T_f, d) one-step predicted diag variances
    m_s:    np.ndarray  # (T_f, d) smoothed means
    P_s:    np.ndarray  # (T_f, d) smoothed diag variances



try:
    import numba as nb
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False
# ---------- Numba RTS smoother (diag OU, exact) ----------
if _HAS_NUMBA:
    @nb.njit(cache=True, fastmath=False)
    def _rts_smoother_numba(F_jm, m_p, P_p, m_f, P_f):
        """
        F_jm : (J,M)   per-(band,taper) state transition (exp(-lam*Δ))
        m_p  : (T,d)   predicted means
        P_p  : (T,d)   predicted variances (diag)
        m_f  : (T,d)   filtered  means
        P_f  : (T,d)   filtered  variances (diag)
        returns: (m_s, P_s) smoothed means/vars (diag), same shapes as m_f/P_f
        """
        J, M = F_jm.shape
        d    = 2 * J * M
        T    = m_f.shape[0]

        # build Fd (length-d) once: real/imag share the same F
        Fd = np.empty(d)
        idx = 0
        for j in range(J):
            for m in range(M):
                F = F_jm[j, m]
                Fd[idx]   = F      # real
                Fd[idx+1] = F      # imag
                idx += 2

        m_s = np.zeros_like(m_f)
        P_s = np.zeros_like(P_f)

        # init last
        for i in range(d):
            m_s[T-1, i] = m_f[T-1, i]
            P_s[T-1, i] = P_f[T-1, i]

        # RTS backward (elementwise because covariance is diagonal)
        for t in range(T - 2, -1, -1):
            for i in range(d):
                denom = P_p[t+1, i]
                if denom < 1e-16:
                    denom = 1e-16
                C = (P_f[t, i] * Fd[i]) / denom
                m_s[t, i] = m_f[t, i] + C * (m_s[t+1, i] - m_p[t+1, i])
                P_s[t, i] = P_f[t, i] + C*C * (P_s[t+1, i] - P_p[t+1, i])
                if P_s[t, i] < 1e-16:
                    P_s[t, i] = 1e-16
        return m_s, P_s

def _compute_X_contribution_to_spikes(
    X_fine: np.ndarray,           # (T_f, 2*J*M) fine-grid X estimate
    beta_S: np.ndarray,           # (S, 1+2B) in INTERLEAVED format [β₀, βR_0, βI_0, βR_1, βI_1, ...]
    freqs_for_phase: np.ndarray,  # (J,) Hz
    coupled_bands_idx: np.ndarray,  # (B,) indices into 0..J-1
    delta_spk: float,
    offset_sec: float,
    J: int,
    M: int,
) -> np.ndarray:
    """
    Compute β · X̃ for each time bin (the X contribution to spike linear predictor).

    X̃ = e^{+iωt} X̄ where X̄ is taper-averaged X.

    Returns:
        X_contrib: (S, T_f) - contribution to subtract from spike pseudo-obs
    """
    T_f = X_fine.shape[0]
    S = beta_S.shape[0]
    B = len(coupled_bands_idx)

    # Reshape X to (T, J, M, 2) for easier indexing
    X_reshaped = X_fine.reshape(T_f, J, M, 2)

    # Taper-average: (T, J, 2)
    X_avg = X_reshaped.mean(axis=2)  # (T, J, 2)

    # Time array (with offset)
    t = offset_sec + np.arange(T_f, dtype=np.float64) * delta_spk

    # For each coupled band, rotate by e^{+iωt}
    X_contrib = np.zeros((S, T_f), dtype=np.float64)

    # beta layout (INTERLEAVED): [β₀, βR_0, βI_0, βR_1, βI_1, ...]
    for b_idx, j in enumerate(coupled_bands_idx):
        freq = freqs_for_phase[j]
        phi = 2.0 * np.pi * freq * t  # (T,)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        # X for this band: Re and Im parts
        X_re = X_avg[:, j, 0]  # (T,)
        X_im = X_avg[:, j, 1]  # (T,)

        # X̃ = e^{+iωt} X = (X_re cos - X_im sin) + i(X_re sin + X_im cos)
        X_tilde_re = X_re * cos_phi - X_im * sin_phi  # (T,)
        X_tilde_im = X_re * sin_phi + X_im * cos_phi  # (T,)

        # β indices in interleaved format
        beta_R_idx = 1 + 2 * b_idx
        beta_I_idx = 1 + 2 * b_idx + 1

        # Contribution: βR · X̃_re + βI · X̃_im for each neuron
        for s in range(S):
            X_contrib[s, :] += (beta_S[s, beta_R_idx] * X_tilde_re +
                               beta_S[s, beta_I_idx] * X_tilde_im)

    return X_contrib  # (S, T_f)


def joint_kf_rts_moments(
    Y_cube: np.ndarray,              # (J, M, K) complex, derotated & scaled
    theta,                           # OUParams(.lam, .sig_v, .sig_eps) (J,M)
    delta_spk: float,
    win_sec: float,
    offset_sec: float,
    beta: np.ndarray,                # (1+2B,)  OR  (S, 1+2B)
    gamma: np.ndarray,               # (R,)     OR  (S, R)
    spikes: np.ndarray,              # (T_f,)   OR  (S, T_f)
    omega: np.ndarray,               # (T_f,)   OR  (S, T_f)
    coupled_bands_idx,               # Sequence[int]
    freqs_for_phase,                 # Sequence[float]
    sidx,                            # StateIndex (unused, kept for API)
    H_hist: np.ndarray,              # (T_f, R) OR  (S, T_f, R)
    *,
    sigma_u: float = 0.0,
    omega_floor: float = 1e-6,
    X_fine_estimate: Optional[np.ndarray] = None,  # (T_f, 2*J*M) for residual mode
    estimate_residual: bool = False,               # If True, estimate D (residual of Z - X)
) -> JointMoments:
    J, M, K = Y_cube.shape

    # ----- normalize to multi-shape (backward compatible) -----
    single = (spikes.ndim == 1)
    if single:
        spikes_S = spikes[None, :]
        omega_S  = omega[None, :]
        H_S      = H_hist[None, :, :]
        beta_S   = beta[None, :]
        gamma_S  = gamma[None, :]
    else:
        spikes_S = spikes
        omega_S  = omega
        H_S      = H_hist
        beta_S   = beta
        gamma_S  = gamma

    S, T_f = spikes_S.shape
    R      = H_S.shape[2]
    B      = (beta_S.shape[1] - 1) // 2

    lam    = np.asarray(theta.lam,     dtype=np.float64)
    sig_v  = np.asarray(theta.sig_v,   dtype=np.float64)
    sig_ep = np.asarray(theta.sig_eps, dtype=np.float64)      # (J,M)
    sig2   = (sig_ep**2).astype(np.float64)

    # ----- OU diagonal dynamics (length-d vectors) -----
    eps = 1e-12
    F_jm  = np.exp(-lam * delta_spk)                          # (J,M)
    Q_jm  = (sig_v**2) * (1.0 - np.exp(-2*lam*delta_spk)) / np.maximum(2*lam, eps)
    P0_jm = (sig_v**2) / np.maximum(2*lam, eps)

    # ----- window centres → fine-time map (T_f, max_k_per_t) with -1 padding -----
    centres_sec = centres_from_win(K, win_sec, offset_sec)
    t2k, kcount = build_t2k(centres_sec, delta_spk, T_f)

    # === RESIDUAL MODE: LFP ===
    # Subtract X estimate from LFP observations at window centers
    if estimate_residual and X_fine_estimate is not None:
        # Compute fine-time index for each window center
        k_to_t = np.clip(np.round((centres_sec - offset_sec) / delta_spk).astype(int), 0, T_f - 1)

        # Reshape X_fine to (T, J, M, 2) and extract values at window centers
        X_at_centres = X_fine_estimate[k_to_t].reshape(K, J, M, 2)  # (K, J, M, 2)
        X_complex = X_at_centres[..., 0] + 1j * X_at_centres[..., 1]  # (K, J, M)

        # Create residual Y_cube: Y_residual = Y - X_hat
        Y_cube = Y_cube - X_complex.transpose(1, 2, 0)  # (J, M, K)

    # ----- per-train spike pseudo-observations -----
    ωeff = np.maximum(np.asarray(omega_S, np.float64), omega_floor)       # (S,T_f)
    κ    = np.asarray(spikes_S, np.float64) - 0.5                          # (S,T_f)
    hist = np.einsum('str,sr->st', np.asarray(H_S[:,:T_f,:], np.float64), np.asarray(gamma_S, np.float64))  # (S,T)
    yspk = np.clip((κ / ωeff) - beta_S[:, [0]] - np.where(np.isfinite(hist), hist, 0.0), -1e6, 1e6)         # (S,T)
    Rspk = (1.0 / ωeff) + float(sigma_u)**2                                                                     # (S,T)

    # === RESIDUAL MODE: SPIKES ===
    # Subtract X contribution from spike pseudo-observations
    if estimate_residual and X_fine_estimate is not None:
        X_spike_contrib = _compute_X_contribution_to_spikes(
            X_fine=X_fine_estimate,
            beta_S=beta_S,
            freqs_for_phase=np.asarray(freqs_for_phase, np.float64),
            coupled_bands_idx=np.asarray(coupled_bands_idx, dtype=int),
            delta_spk=delta_spk,
            offset_sec=offset_sec,
            J=J,
            M=M,
        )
        yspk = yspk - X_spike_contrib  # (S, T_f)

    # ----- carriers + per-time weights a_s(t), b_s(t); NO Hspk -----
    freqs_for_phase = np.asarray(freqs_for_phase, np.float64)
    t = np.arange(T_f, dtype=np.float64) * float(delta_spk)
    phi = 2.0 * np.pi * freqs_for_phase.reshape(J,1) * t.reshape(1,T_f)
    c_tbl = np.cos(phi)                                 # (J,T_f)
    s_tbl = np.sin(phi)                                 # (J,T_f)

    beta_pairs_sub = np.asarray(beta_S[:, 1:], np.float64).reshape(S, B, 2)  # (S,B,2)
    beta_pairs = np.zeros((S, J, 2), np.float64)
    idx = np.asarray(coupled_bands_idx, dtype=int)
    beta_pairs[:, idx, :] = beta_pairs_sub

    # a multiplies Re; b multiplies Im; (S,T,J), repeated across tapers (÷M)
    a_tbl = (beta_pairs[..., 0][:, None, :] * c_tbl.T[None, :, :] +
             beta_pairs[..., 1][:, None, :] * s_tbl.T[None, :, :]) / float(M)
    b_tbl = (-beta_pairs[..., 0][:, None, :] * s_tbl.T[None, :, :] +
              beta_pairs[..., 1][:, None, :] * c_tbl.T[None, :, :]) / float(M)

    # real/imag observations as real arrays
    Yre = np.ascontiguousarray(np.real(Y_cube), dtype=np.float64)      # (J,M,K)
    Yim = np.ascontiguousarray(np.imag(Y_cube), dtype=np.float64)

    # allocate outputs (flattened d = 2*J*M)
    d = 2 * J * M
    m_p = np.zeros((T_f, d), np.float64)
    P_p = np.zeros((T_f, d), np.float64)
    m_f = np.zeros((T_f, d), np.float64)
    P_f = np.zeros((T_f, d), np.float64)
    m_s = np.zeros((T_f, d), np.float64)
    P_s = np.zeros((T_f, d), np.float64)

    # ---- compiled forward + RTS ----
    if _HAS_NUMBA:
        m_p, P_p, m_f, P_f = _forward_filter_numba_multi(
            Yre, Yim, F_jm, Q_jm, P0_jm, sig2, t2k, kcount,
            a_tbl, b_tbl, yspk, Rspk
        )
        m_s, P_s = _rts_smoother_numba(F_jm, m_p, P_p, m_f, P_f)
    else:
        m_p, P_p, m_f, P_f = _forward_filter_numpy_multi(
            Yre, Yim, F_jm, Q_jm, P0_jm, sig2, t2k, kcount,
            a_tbl, b_tbl, yspk, Rspk
        )
        m_s, P_s = _rts_smoother_numpy(F_jm, m_p, P_p, m_f, P_f)

    return JointMoments(m_f=m_f, P_f=P_f, P_pred=m_p, m_s=m_s, P_s=P_s)

# ------------------------- NUMBA kernels (multi-spike) -------------------------
if _HAS_NUMBA:
    @nb.njit(cache=True, fastmath=False)
    def _forward_filter_numba_multi(
        Yre, Yim, F_jm, Q_jm, P0_jm, sig2, t2k, kcount, a_tbl, b_tbl, yspk, Rspk
    ):
        J, M, K = Yre.shape
        S, T_f = yspk.shape
        d = 2 * J * M

        mu_re = np.zeros((J, M))
        mu_im = np.zeros((J, M))
        P_re  = P0_jm.copy()
        P_im  = P0_jm.copy()

        m_p = np.zeros((T_f, d))
        P_p = np.zeros((T_f, d))
        m_f = np.zeros((T_f, d))
        P_f = np.zeros((T_f, d))

        for t in range(T_f):
            # predict
            mp_re = F_jm * mu_re
            mp_im = F_jm * mu_im
            Pp_re = (F_jm * F_jm) * P_re + Q_jm
            Pp_im = (F_jm * F_jm) * P_im + Q_jm

            # LFP centre updates (all (J,M))
            mu_re_u = mp_re.copy(); mu_im_u = mp_im.copy()
            P_re_u  = Pp_re.copy(); P_im_u  = Pp_im.copy()

            kc = kcount[t]
            for i in range(kc):
                k = t2k[t, i]
                if k < 0: break
                # real
                for j in range(J):
                    for m in range(M):
                        S_ = P_re_u[j, m] + sig2[j, m]
                        K_ = P_re_u[j, m] / (S_ + 1e-18)
                        inov = Yre[j, m, k] - mu_re_u[j, m]
                        mu_re_u[j, m] += K_ * inov
                        P_re_u[j, m] = (1.0 - K_) * P_re_u[j, m]
                # imag
                for j in range(J):
                    for m in range(M):
                        S_ = P_im_u[j, m] + sig2[j, m]
                        K_ = P_im_u[j, m] / (S_ + 1e-18)
                        inov = Yim[j, m, k] - mu_im_u[j, m]
                        mu_im_u[j, m] += K_ * inov
                        P_im_u[j, m] = (1.0 - K_) * P_im_u[j, m]

            # apply S spike pseudo-rows (sequential; conditionally independent)
            # precompute taper sums once per t
            sum_m_re = np.zeros(J); sum_m_im = np.zeros(J)
            sum_P_re = np.zeros(J); sum_P_im = np.zeros(J)

            for j in range(J):
                s_mr = 0.0; s_mi = 0.0; s_Pr = 0.0; s_Pi = 0.0
                for m in range(M):
                    s_mr += mu_re_u[j, m]
                    s_mi += mu_im_u[j, m]
                    s_Pr += P_re_u[j, m]
                    s_Pi += P_im_u[j, m]
                sum_m_re[j] = s_mr
                sum_m_im[j] = s_mi
                sum_P_re[j] = s_Pr
                sum_P_im[j] = s_Pi

            for s in range(S):
                arow = a_tbl[s, t]   # (J,)
                brow = b_tbl[s, t]   # (J,)

                # yhat, S_spk
                yhat = 0.0
                for j in range(J):
                    yhat += arow[j]*sum_m_re[j] + brow[j]*sum_m_im[j]
                inov = yspk[s, t] - yhat

                S_spk = 0.0
                for j in range(J):
                    S_spk += (arow[j]*arow[j])*sum_P_re[j] + (brow[j]*brow[j])*sum_P_im[j]
                S_spk += Rspk[s, t]

                # gains & updates
                for j in range(J):
                    aj = arow[j]; bj = brow[j]
                    for m in range(M):
                        K_re = (P_re_u[j, m] * aj) / (S_spk + 1e-18)
                        K_im = (P_im_u[j, m] * bj) / (S_spk + 1e-18)
                        mu_re_u[j, m] += K_re * inov
                        mu_im_u[j, m] += K_im * inov
                        P_re_u[j, m]  = max(P_re_u[j, m] - K_re * (aj * P_re_u[j, m]), 1e-16)
                        P_im_u[j, m]  = max(P_im_u[j, m] - K_im * (bj * P_im_u[j, m]), 1e-16)

                # update the sums after each spike row (exact)
                for j in range(J):
                    s_mr = 0.0; s_mi = 0.0; s_Pr = 0.0; s_Pi = 0.0
                    for m in range(M):
                        s_mr += mu_re_u[j, m]
                        s_mi += mu_im_u[j, m]
                        s_Pr += P_re_u[j, m]
                        s_Pi += P_im_u[j, m]
                    sum_m_re[j] = s_mr
                    sum_m_im[j] = s_mi
                    sum_P_re[j] = s_Pr
                    sum_P_im[j] = s_Pi

            # flatten & store
            for j in range(J):
                for m in range(M):
                    base = (j*M + m) * 2
                    m_p[t, base]   = mp_re[j, m]
                    m_p[t, base+1] = mp_im[j, m]
                    P_p[t, base]   = Pp_re[j, m]
                    P_p[t, base+1] = Pp_im[j, m]
                    m_f[t, base]   = mu_re_u[j, m]
                    m_f[t, base+1] = mu_im_u[j, m]
                    P_f[t, base]   = P_re_u[j, m]
                    P_f[t, base+1] = P_im_u[j, m]

            mu_re = mu_re_u; mu_im = mu_im_u
            P_re  = P_re_u;  P_im  = P_im_u

        return m_p, P_p, m_f, P_f

# ------------------------- NumPy fallback (multi-spike) -------------------------
def _forward_filter_numpy_multi(
    Yre, Yim, F_jm, Q_jm, P0_jm, sig2, t2k, kcount, a_tbl, b_tbl, yspk, Rspk
):
    J, M, K = Yre.shape
    S, T_f  = yspk.shape
    d = 2 * J * M

    mu_re = np.zeros((J, M)); mu_im = np.zeros((J, M))
    P_re  = P0_jm.copy();     P_im  = P0_jm.copy()

    m_p = np.zeros((T_f, d)); P_p = np.zeros((T_f, d))
    m_f = np.zeros((T_f, d)); P_f = np.zeros((T_f, d))

    for t in range(T_f):
        mp_re = F_jm * mu_re; mp_im = F_jm * mu_im
        Pp_re = (F_jm*F_jm) * P_re + Q_jm
        Pp_im = (F_jm*F_jm) * P_im + Q_jm

        mu_re_u = mp_re.copy(); mu_im_u = mp_im.copy()
        P_re_u  = Pp_re.copy(); P_im_u  = Pp_im.copy()

        kc = kcount[t]
        for i in range(kc):
            k = t2k[t, i]
            if k < 0: break
            # real
            S_re = P_re_u + sig2
            K_re = P_re_u / (S_re + 1e-18)
            inov_re = Yre[..., k] - mu_re_u
            mu_re_u = mu_re_u + K_re * inov_re
            P_re_u  = (1.0 - K_re) * P_re_u
            # imag
            S_im = P_im_u + sig2
            K_im = P_im_u / (S_im + 1e-18)
            inov_im = Yim[..., k] - mu_im_u
            mu_im_u = mu_im_u + K_im * inov_im
            P_im_u  = (1.0 - K_im) * P_im_u

        # sums over tapers (once)
        sum_m_re = mu_re_u.sum(axis=1); sum_m_im = mu_im_u.sum(axis=1)
        sum_P_re = P_re_u.sum(axis=1);  sum_P_im = P_im_u.sum(axis=1)

        # apply S spike rows
        for s in range(S):
            a = a_tbl[s, t]; b = b_tbl[s, t]
            yhat  = (a * sum_m_re + b * sum_m_im).sum()
            S_spk = (a*a * sum_P_re + b*b * sum_P_im).sum() + Rspk[s, t]
            inov  = yspk[s, t] - yhat

            K_re = (P_re_u * a[:, None]) / (S_spk + 1e-18)
            K_im = (P_im_u * b[:, None]) / (S_spk + 1e-18)
            mu_re_u = mu_re_u + K_re * inov
            mu_im_u = mu_im_u + K_im * inov
            P_re_u  = np.maximum(P_re_u - K_re * (a[:, None] * P_re_u), 1e-16)
            P_im_u  = np.maximum(P_im_u - K_im * (b[:, None] * P_im_u), 1e-16)

            # refresh sums for next spike row
            sum_m_re = mu_re_u.sum(axis=1); sum_m_im = mu_im_u.sum(axis=1)
            sum_P_re = P_re_u.sum(axis=1);  sum_P_im = P_im_u.sum(axis=1)

        # flatten & store
        base = 0
        for j in range(J):
            for m in range(M):
                m_p[t, base]   = mp_re[j, m]
                m_p[t, base+1] = mp_im[j, m]
                P_p[t, base]   = Pp_re[j, m]
                P_p[t, base+1] = Pp_im[j, m]
                m_f[t, base]   = mu_re_u[j, m]
                m_f[t, base+1] = mu_im_u[j, m]
                P_f[t, base]   = P_re_u[j, m]
                P_f[t, base+1] = P_im_u[j, m]
                base += 2

        mu_re, mu_im = mu_re_u, mu_im_u
        P_re,  P_im  = P_re_u,  P_im_u

    return m_p, P_p, m_f, P_f



# ---------- NumPy fallback RTS smoother (same math) ----------
def _rts_smoother_numpy(F_jm, m_p, P_p, m_f, P_f):
    J, M = F_jm.shape
    d    = 2 * J * M
    T    = m_f.shape[0]

    # vectorized Fd
    Fd = np.repeat(F_jm[..., None], 2, axis=-1).reshape(-1)

    m_s = np.zeros_like(m_f)
    P_s = np.zeros_like(P_f)
    m_s[-1] = m_f[-1]
    P_s[-1] = P_f[-1]

    for t in range(T - 2, -1, -1):
        denom = np.maximum(P_p[t+1], 1e-16)
        C     = (P_f[t] * Fd) / denom
        m_s[t] = m_f[t] + C * (m_s[t+1] - m_p[t+1])
        P_s[t] = np.maximum(P_f[t] + C*C * (P_s[t+1] - P_p[t+1]), 1e-16)
    return m_s, P_s


