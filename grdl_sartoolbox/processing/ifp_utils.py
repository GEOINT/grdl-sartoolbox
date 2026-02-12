# -*- coding: utf-8 -*-
"""
Image Formation Processing (IFP) Utilities.

Supporting functions for SAR image formation algorithms including
range deskew, resolution/extent computation, and pulse selection.

Attribution
-----------
Ported from MATLAB SAR Toolbox (https://github.com/ngageoint/MATLAB_SAR)
Original: Wade Schwartzkopf and Tom Krauss, NGA/IDT

License
-------
MIT License
Copyright (c) 2024 geoint.org
"""

from typing import Tuple, Optional
from dataclasses import dataclass
import numpy as np

from grdl_sartoolbox.utils.constants import SPEED_OF_LIGHT


@dataclass
class ResolutionExtent:
    """
    SAR resolution and extent parameters.

    Attributes
    ----------
    range_resolution : float
        Range resolution (meters).
    azimuth_resolution : float
        Azimuth (cross-range) resolution (meters).
    range_extent : float
        Maximum unambiguous range extent (meters).
    azimuth_extent : float
        Maximum unambiguous azimuth extent (meters).
    delta_azimuth : float
        Angular spacing between pulses (radians).
    total_azimuth : float
        Total angular aperture (radians).
    """
    range_resolution: float
    azimuth_resolution: float
    range_extent: float
    azimuth_extent: float
    delta_azimuth: float
    total_azimuth: float


def pulse_info_to_resolution_extent(
    range_vectors: np.ndarray,
    center_frequency: float,
    delta_frequency: float,
    bandwidth: float,
    num_samples: int
) -> ResolutionExtent:
    """
    Compute SAR resolution and extent from pulse parameters.

    Parameters
    ----------
    range_vectors : np.ndarray
        Unit range vectors for each pulse, shape (num_pulses, 3).
    center_frequency : float
        Center frequency (Hz).
    delta_frequency : float
        Frequency step between samples (Hz).
    bandwidth : float
        Total signal bandwidth (Hz).
    num_samples : int
        Number of range samples per pulse.

    Returns
    -------
    ResolutionExtent
        Computed resolution and extent parameters.
    """
    num_pulses = range_vectors.shape[0]

    # Normalize range vectors
    norms = np.linalg.norm(range_vectors, axis=1)
    unit_vectors = range_vectors / norms[:, np.newaxis]

    # Compute angular aperture from first to last pulse
    dot_product = np.clip(
        np.dot(unit_vectors[0], unit_vectors[-1]), -1.0, 1.0
    )
    total_azimuth = np.arccos(dot_product)

    # Angular spacing between consecutive pulses
    if num_pulses > 1:
        delta_azimuth = total_azimuth / (num_pulses - 1)
    else:
        delta_azimuth = 0.0

    # Range resolution: c / (2 * BW)
    range_resolution = SPEED_OF_LIGHT / (2.0 * bandwidth)

    # Azimuth resolution: c / (2 * theta * fc) [for spotlight]
    wavelength = SPEED_OF_LIGHT / center_frequency
    if total_azimuth > 0:
        azimuth_resolution = wavelength / (2.0 * total_azimuth)
    else:
        azimuth_resolution = float('inf')

    # Range extent: c / (2 * delta_f)
    if delta_frequency > 0:
        range_extent = SPEED_OF_LIGHT / (2.0 * delta_frequency)
    else:
        range_extent = float('inf')

    # Azimuth extent: wavelength / (2 * delta_azimuth)
    if delta_azimuth > 0:
        azimuth_extent = wavelength / (2.0 * delta_azimuth)
    else:
        azimuth_extent = float('inf')

    return ResolutionExtent(
        range_resolution=range_resolution,
        azimuth_resolution=azimuth_resolution,
        range_extent=range_extent,
        azimuth_extent=azimuth_extent,
        delta_azimuth=delta_azimuth,
        total_azimuth=total_azimuth
    )


def deskew_rvp(
    pulses: np.ndarray,
    sampling_rate: float,
    chirp_rate: float,
    pad: int = 0,
    time_shift: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, float]:
    """
    Apply range deskew (residual video phase compensation).

    Removes residual video phase (RVP) caused by stretch processing
    via frequency-dependent time shifts in the Fourier domain.

    Parameters
    ----------
    pulses : np.ndarray
        Pulse data, shape (num_samples, num_pulses) or (num_samples,).
    sampling_rate : float
        Sampling rate (Hz).
    chirp_rate : float
        Chirp rate (Hz/s).
    pad : int
        Zero-padding factor (0 = no padding). Default 0.
    time_shift : np.ndarray, optional
        Per-pulse time shifts for dejitter (seconds).

    Returns
    -------
    output : np.ndarray
        Deskewed pulses.
    t_start : float
        Start time of output (seconds).
    """
    if pulses.ndim == 1:
        pulses = pulses[:, np.newaxis]

    num_samples, num_pulses = pulses.shape

    # Padded size
    n_pad = num_samples + pad

    # Frequency axis
    freq = np.fft.fftfreq(n_pad, d=1.0 / sampling_rate)

    # Quadratic phase correction for RVP
    # H(f) = exp(j * pi * f^2 / chirp_rate)
    rvp_correction = np.exp(1j * np.pi * freq ** 2 / chirp_rate)

    # Process each pulse
    output = np.zeros((n_pad, num_pulses), dtype=pulses.dtype)

    for p in range(num_pulses):
        # Zero-pad
        padded = np.zeros(n_pad, dtype=pulses.dtype)
        padded[:num_samples] = pulses[:, p]

        # Apply correction in frequency domain
        ft = np.fft.fft(padded)

        # Add dejitter if provided
        if time_shift is not None:
            linear_phase = np.exp(-1j * 2 * np.pi * freq * time_shift[p])
            ft *= linear_phase

        ft *= rvp_correction
        output[:, p] = np.fft.ifft(ft)

    # Compute output time start
    t_start = -0.5 / sampling_rate * pad

    if output.shape[1] == 1:
        output = output.ravel()

    return output, t_start


def pfa_inverse(
    complex_data: np.ndarray,
    center_freq: float,
    sample_spacing: Tuple[float, float],
    imp_resp_bw: Tuple[float, float],
    fft_sgn: Tuple[int, int] = (-1, -1)
) -> np.ndarray:
    """
    Inverse Polar Format Algorithm.

    Transforms rectangular-sampled image data back to polar-sampled
    (frequency, angle) domain.

    Parameters
    ----------
    complex_data : np.ndarray
        Complex SAR image, shape (rows, cols).
    center_freq : float
        Center frequency (Hz).
    sample_spacing : tuple of float
        (row_spacing, col_spacing) in meters.
    imp_resp_bw : tuple of float
        (row_bandwidth, col_bandwidth) impulse response bandwidth.
    fft_sgn : tuple of int
        FFT sign convention for each dimension.

    Returns
    -------
    np.ndarray
        Polar-sampled phase history data.
    """
    rows, cols = complex_data.shape

    # Transform to k-space via 2D FFT
    k_data = complex_data.copy()

    # Row dimension
    if fft_sgn[0] == -1:
        k_data = np.fft.fftshift(np.fft.fft(k_data, axis=0), axes=0)
    else:
        k_data = np.fft.fftshift(np.fft.ifft(k_data, axis=0), axes=0)

    # Column dimension
    if fft_sgn[1] == -1:
        k_data = np.fft.fftshift(np.fft.fft(k_data, axis=1), axes=1)
    else:
        k_data = np.fft.fftshift(np.fft.ifft(k_data, axis=1), axes=1)

    # Compute k-space coordinates (rectangular)
    k_v = np.fft.fftshift(np.fft.fftfreq(rows, d=sample_spacing[0]))
    k_u = np.fft.fftshift(np.fft.fftfreq(cols, d=sample_spacing[1]))

    # Trim to impulse response bandwidth
    bw_row = imp_resp_bw[0]
    bw_col = imp_resp_bw[1]
    valid_rows = np.abs(k_v) <= bw_row / 2
    valid_cols = np.abs(k_u) <= bw_col / 2

    k_data_trimmed = k_data[np.ix_(valid_rows, valid_cols)]

    return k_data_trimmed


__all__ = [
    "ResolutionExtent",
    "pulse_info_to_resolution_extent",
    "deskew_rvp",
    "pfa_inverse",
]
