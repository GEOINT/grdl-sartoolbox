# -*- coding: utf-8 -*-
"""
Signal Analysis - STFT and pulse signal processing.

Implements Short-Time Fourier Transform and deramp operations
for SAR pulse analysis.

Attribution
-----------
Ported from MATLAB SAR Toolbox (https://github.com/ngageoint/MATLAB_SAR)
Original: NGA/IDT PulseExplorer

License
-------
MIT License
Copyright (c) 2024 geoint.org
"""

from typing import Tuple, Optional
import numpy as np
from scipy.signal import spectrogram
from scipy.signal.windows import kaiser


def stft(
    data: np.ndarray,
    sample_rate: float = 1.0,
    center: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Short-Time Fourier Transform using sliding window spectrogram.

    Parameters
    ----------
    data : np.ndarray
        1D input signal (real or complex).
    sample_rate : float
        Sampling rate (Hz). Default 1.0.
    center : bool
        If True, fftshift for DC-centered output. Default True.

    Returns
    -------
    Sxx : np.ndarray
        STFT magnitude, shape (num_freqs, num_times).
    t : np.ndarray
        Time axis (seconds).
    f : np.ndarray
        Frequency axis (Hz).
    """
    n = len(data)

    # FFT size: 4x next power of 2 of sqrt(n)
    nfft = 4 * (2 ** int(np.ceil(np.log2(max(np.sqrt(n), 4)))))

    # Kaiser window with ~97% of nfft length
    win_len = max(4, int(0.97 * nfft))
    if win_len % 2 == 0:
        win_len -= 1  # Make odd for symmetry
    win = kaiser(win_len, beta=5)

    # 90% overlap
    noverlap = int(0.9 * len(win))

    f, t, Sxx = spectrogram(
        data, fs=sample_rate, window=win,
        nperseg=len(win), noverlap=noverlap, nfft=nfft,
        return_onesided=False, mode='complex'
    )

    if center:
        # Shift DC to center
        Sxx = np.fft.fftshift(Sxx, axes=0)
        f = np.fft.fftshift(f)

    return np.abs(Sxx), t, f


def reramp(
    pulses: np.ndarray,
    sample_rate: float,
    deramp_rate: float
) -> Tuple[np.ndarray, float]:
    """
    Apply quadratic phase ramp (re-ramp) to SAR pulse data.

    Undoes deramp processing by applying the conjugate quadratic phase.
    Upsamples if needed to avoid aliasing.

    Parameters
    ----------
    pulses : np.ndarray
        Pulse data, shape (num_samples,) or (num_samples, num_pulses).
    sample_rate : float
        Input sampling rate (Hz).
    deramp_rate : float
        Deramp chirp rate (Hz/s).

    Returns
    -------
    output : np.ndarray
        Re-ramped pulse data (possibly upsampled).
    dt : float
        New sample spacing (seconds).
    """
    if pulses.ndim == 1:
        pulses = pulses[:, np.newaxis]
        squeeze = True
    else:
        squeeze = False

    num_samples, num_pulses = pulses.shape

    # Determine required upsample factor
    deramp_bw = abs(deramp_rate) * num_samples / sample_rate
    total_bw = sample_rate + deramp_bw
    upsample_factor = max(1, int(np.ceil(total_bw / sample_rate)))

    # Upsample via FFT interpolation
    if upsample_factor > 1:
        new_n = num_samples * upsample_factor
        upsampled = np.zeros((new_n, num_pulses), dtype=pulses.dtype)

        for p in range(num_pulses):
            # Zero-mean
            pulse = pulses[:, p] - np.mean(pulses[:, p])
            # FFT interpolation
            ft = np.fft.fft(pulse)
            padded_ft = np.zeros(new_n, dtype=ft.dtype)
            half = num_samples // 2
            padded_ft[:half] = ft[:half]
            padded_ft[-half:] = ft[-half:]
            upsampled[:, p] = np.fft.ifft(padded_ft) * upsample_factor
    else:
        new_n = num_samples
        upsampled = pulses.copy()

    # New sample spacing
    dt = 1.0 / (sample_rate * upsample_factor)

    # Create time vector centered on pulse
    t = (np.arange(new_n) - new_n / 2) * dt

    # Quadratic phase ramp
    phase_ramp = np.exp(1j * np.pi * deramp_rate * t ** 2)

    # Apply ramp to each pulse
    output = upsampled * phase_ramp[:, np.newaxis]

    if squeeze:
        output = output.ravel()

    return output, dt


__all__ = ["stft", "reramp"]
