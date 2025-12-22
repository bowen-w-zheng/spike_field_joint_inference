"""
utils_multitaper.py - Multitaper TFR utilities.
"""
import numpy as np


def derotate_tfr_align_start(Y_cplx, freqs_hz, sfreq, hop_samples, win_samples, t0=0.0):
    """
    Derotate TFR aligning to window start.

    Parameters
    ----------
    Y_cplx : ndarray
        Complex TFR array, shape (F, K, T) or (R, F, K, T)
    freqs_hz : ndarray
        Frequency axis (Hz)
    sfreq : float
        Sampling frequency (Hz)
    hop_samples : int
        Stride between consecutive frames (samples)
    win_samples : int
        Window length (samples)
    t0 : float
        Time offset (seconds)

    Returns
    -------
    ndarray
        Derotated TFR, same shape as input
    """
    if Y_cplx.ndim == 3:
        F, K, T = Y_cplx.shape
        # use STARTS, not centers
        starts_sec = t0 + (np.arange(T) * hop_samples) / sfreq
        rot = np.exp(-1j * 2*np.pi * freqs_hz[:, None, None] * starts_sec[None, None, :])
        return Y_cplx * rot
    elif Y_cplx.ndim == 4:
        R, F, K, T = Y_cplx.shape
        starts_sec = t0 + (np.arange(T) * hop_samples) / sfreq
        rot = np.exp(-1j * 2*np.pi * freqs_hz[None, :, None, None] * starts_sec[None, None, None, :])
        return Y_cplx * rot
    else:
        raise ValueError(f"Y_cplx with ndim={Y_cplx.ndim} unsupported for derotation")
