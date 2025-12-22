import numpy as np
import mne 

def derotate_tfr(Y_cplx: np.ndarray,
                 freqs_hz: np.ndarray,
                 sfreq: float,
                 hop_samples: int,
                 win_samples: int,
                 t0: float = 0.0):
    """
    Y_cplx: (..., n_freqs, n_tapers, n_frames) from tfr_array_multitaper
    freqs_hz: frequency axis used in TFR (length = n_freqs)
    sfreq: sampling rate (Hz)
    hop_samples: stride between consecutive frames (samples); with decim=M, hop=M
    win_samples: window length (samples); with n_cycles=f*window_sec, win≈M
    t0: optional time offset (s)
    Supports Y_cplx of shape (n_freqs, n_tapers, n_frames) or
    (n_trials, n_freqs, n_tapers, n_frames)
    """
    if Y_cplx.ndim == 3:
        F, K, T = Y_cplx.shape
        centers_sec = t0 + (np.arange(T) * hop_samples + (win_samples - 1) / 2.0) / sfreq
        rot = np.exp(-1j * 2 * np.pi * freqs_hz[:, None, None] * centers_sec[None, None, :])
        return Y_cplx * rot
    elif Y_cplx.ndim == 4:
        R, F, K, T = Y_cplx.shape
        centers_sec = t0 + (np.arange(T) * hop_samples + (win_samples - 1) / 2.0) / sfreq
        # Broadcast trial axis
        rot = np.exp(-1j * 2 * np.pi * freqs_hz[None, :, None, None] * centers_sec[None, None, None, :])
        return Y_cplx * rot
    else:
        raise ValueError(f"Y_cplx with ndim={Y_cplx.ndim} unsupported for derotation")


def derotate_tfr_align_start(Y_cplx, freqs_hz, sfreq, hop_samples, win_samples, t0=0.0):
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


def fine_to_cube_complex(fine_latent: np.ndarray, J: int, M: int) -> np.ndarray:
    """
    fine_latent: (T_f, d) with d == 2*J*M, flattened as (J, M, 2) in C-order
    returns:     (J, M, T_f) complex array
    """
    T_f, d = fine_latent.shape
    assert d == 2 * J * M, f"expected d=2*J*M={2*J*M}, got {d}"
    # reshape back to (T_f, J, M, 2), then split Re/Im and move time last
    tmp = fine_latent.reshape(T_f, J, M, 2, order="C")
    Z = (tmp[..., 0] + 1j * tmp[..., 1]).transpose(1, 2, 0)  # (J, M, T_f)
    return Z


# Compute the spectral coefficients with the option to return scaled power spectrogram 
def compute_spectral_coefficients(lfp, fs, freqs, window_sec, NW_product):
    """
    Compute the spectral coefficients with the option to return scaled power spectrogram 
    lfp: (T,)
    fs: float
    freqs: (F,)
    window_sec: float
    NW_product: float
    return_power: bool
    (F, K, T): (frequency, taper, time)
    """
    tfr_raw = mne.time_frequency.tfr_array_multitaper(
        lfp[None, None, :],
        sfreq=fs,
        freqs=freqs,
        n_cycles=freqs * window_sec,
        time_bandwidth=2 * NW_product,
        output="complex",
        zero_mean=False,
    )[0, 0]   # the direct output is (n_trials, n_channels, n_tapers, n_freqs, n_timepoints)
    # we need to reshape it to (n_tapers, n_freqs, n_timepoints)
    # Time axis for TFR
    tfr_time = np.arange(tfr_raw.shape[2]) / fs

    # Reorder to (F, K, T) exactly as in the OU script
    tfr = tfr_raw.copy()
    tfr_raw = np.swapaxes(tfr_raw, 0, 1)   # (F, K, T)
    tfr     = np.swapaxes(tfr,     0, 1)   # (F, K, T)

    # --- 3.2 Derotate and DPSS scaling ------------------------------------------
    M     = int(round(window_sec * fs))
    hop   = 1      # decim=1 in previous code
    tfr   = derotate_tfr(tfr, freqs, fs, hop, M)
    tapers, _ = mne.time_frequency.multitaper.dpss_windows(M, NW_product, Kmax=1)
    scaling_factor = 2.0 / tapers.sum(axis=1)
    tfr     = tfr * scaling_factor

    # Non-overlapping windows for convenience (stride = M samples)
    tfr_ds      = tfr[:, :, ::M]
    tfr_time_ds = tfr_time[::M]

    return tfr_ds, tfr_time_ds
    
def compute_spectral_coefficients_trial(lfp_trials, fs, freqs, window_sec, NW_product, return_power=False):
    # Assuming there is more than one trial
    # ────────── 2) Multitaper spectrogram (complex) ──────────
    tfr_raw = mne.time_frequency.tfr_array_multitaper(
        lfp_trials[:,None,:],
        sfreq=fs,
        freqs=freqs,
        n_cycles=freqs * window_sec,
        time_bandwidth=2 * NW_product,
        output="complex",
        zero_mean=False,
    )  # (R, C, K, F, T): (trials, tapers, frequencies, timepoints)]
    tfr_raw = np.squeeze(tfr_raw, axis=1) # (R, K, F, T): (trials, tapers, frequencies, timepoints)
    # reshape it to (R, F, K, T, axis swap to (R, F, K, T))
    tfr_raw = np.swapaxes(tfr_raw, 1, 2) 
    tfr_time = np.arange(tfr_raw.shape[-1]) / fs
    tfr      = tfr_raw.copy()
    # Derotate and scale each taper
    M = int(round(window_sec * fs))
    tfr = derotate_tfr_align_start(tfr, freqs, fs,1, M)

    tapers, _ = mne.time_frequency.multitaper.dpss_windows(M, NW_product, Kmax=1)
    scaling_factor = 2.0 / tapers.sum(axis=1)
    tfr_raw = tfr_raw * scaling_factor
    tfr     = tfr * scaling_factor

    # Non-overlapping windows
    tfr_ds      = tfr[:, :, :, ::M]
    tfr_time = np.arange(tfr_raw.shape[-1]) / fs
    tfr_time_ds = tfr_time[::M]

    print(f"\nY_cube_trials shape: {tfr_ds.shape} (should be (n_trials, freqs, tapers, timepoints))")
    
    # calculate the power for each trial 
    return tfr_ds, tfr_time_ds

def compute_scaled_power_spectrogram(tfr_ds, freqs, fs, lfp):
    # P_est_scaled scale the spectrogram such that the unit is var/hz
    P_est      = np.abs(tfr_ds) ** 2                # (F, K, T)
    P_est_mean = P_est.mean(axis=1)              # (F, T)

    df     = freqs[1] - freqs[0]
    var_x  = np.var(lfp)
    var_spec_frames = (2.0 / fs) * np.sum(P_est_mean, axis=0) * df   # (T,)
    mean_var_spec   = np.mean(var_spec_frames) + 1e-30
    c_global        = var_x / mean_var_spec
    P_est_scaled = c_global * P_est_mean                # (F, T)
    return P_est_mean, P_est_scaled

