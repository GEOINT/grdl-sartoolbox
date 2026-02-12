# -*- coding: utf-8 -*-
"""
Radar Cross-Section (RCS) Computation.

Computes calibrated radar cross-section from complex SAR image data.

Attribution
-----------
Ported from MATLAB SAR Toolbox (https://github.com/ngageoint/MATLAB_SAR)
Original: NGA/IDT RCS Tool

License
-------
MIT License
Copyright (c) 2024 geoint.org
"""

import numpy as np
from typing import Optional


def compute_rcs(
    complex_data: np.ndarray,
    oversample_ratio: float = 1.0,
    cal_sf: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None
) -> float:
    """
    Compute calibrated Radar Cross-Section for a region of interest.

    Parameters
    ----------
    complex_data : np.ndarray
        Complex SAR image data, shape (rows, cols).
    oversample_ratio : float
        Zero-pad oversampling ratio (accounts for sinc^2 spreading).
        Default 1.0 (no oversampling).
    cal_sf : np.ndarray, optional
        Per-pixel calibration scale factor. If None, uses unit calibration.
    mask : np.ndarray, optional
        Binary mask for region of interest (True = include).
        If None, uses entire image.

    Returns
    -------
    float
        Total RCS in square meters (linear scale, not dB).
    """
    if mask is None:
        mask = np.ones(complex_data.shape, dtype=bool)

    if cal_sf is None:
        cal_sf = np.ones(complex_data.shape, dtype=np.float64)

    # Compute power (magnitude squared)
    power = np.abs(complex_data) ** 2

    # Apply calibration and mask
    calibrated_power = power * cal_sf

    # Sum over masked region, accounting for oversampling
    # Oversampled data concentrates energy differently
    total_rcs = np.sum(calibrated_power[mask]) / (oversample_ratio ** 2)

    return float(total_rcs)


def compute_rcs_db(
    complex_data: np.ndarray,
    oversample_ratio: float = 1.0,
    cal_sf: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None
) -> float:
    """
    Compute RCS in dBsm (decibels relative to one square meter).

    Parameters
    ----------
    complex_data : np.ndarray
        Complex SAR image data.
    oversample_ratio : float
        Zero-pad oversampling ratio.
    cal_sf : np.ndarray, optional
        Per-pixel calibration scale factor.
    mask : np.ndarray, optional
        Binary mask for region of interest.

    Returns
    -------
    float
        RCS in dBsm.
    """
    rcs = compute_rcs(complex_data, oversample_ratio, cal_sf, mask)
    return float(10.0 * np.log10(max(rcs, 1e-30)))


__all__ = ["compute_rcs", "compute_rcs_db"]
