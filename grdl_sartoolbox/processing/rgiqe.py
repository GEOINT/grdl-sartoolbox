# -*- coding: utf-8 -*-
"""
Radar Generalized Image Quality Equation (RGIQE).

Computes SAR image quality metrics including information density
and RNIIRS (Radar National Imagery Interpretability Rating Scale).

Attribution
-----------
Ported from MATLAB SAR Toolbox (https://github.com/ngageoint/MATLAB_SAR)
Based on NGA RGIQE methodology.

License
-------
MIT License
Copyright (c) 2024 geoint.org
"""

from typing import Optional, Tuple, Dict
from dataclasses import dataclass
import numpy as np


@dataclass
class RGIQEResult:
    """
    RGIQE computation result.

    Attributes
    ----------
    information_density : float
        Image information density (bits/m^2).
    rniirs : float
        Radar NIIRS estimate (1-9 scale).
    snr_db : float
        Signal-to-noise ratio in dB.
    bandwidth_area : float
        2D bandwidth area (cycles^2/m^2).
    range_resolution : float
        Range resolution (meters).
    azimuth_resolution : float
        Azimuth resolution (meters).
    graze_angle : float
        Grazing angle (radians).
    nesz_db : float
        Noise equivalent sigma-zero (dB).
    """
    information_density: float
    rniirs: float
    snr_db: float
    bandwidth_area: float
    range_resolution: float
    azimuth_resolution: float
    graze_angle: float
    nesz_db: float


def compute_rgiqe(
    range_resolution: float,
    azimuth_resolution: float,
    graze_angle: float,
    nesz_db: float,
    signal_sigma_db: float = -10.0,
    multi_noise: float = 0.0
) -> RGIQEResult:
    """
    Compute RGIQE metrics for SAR image quality assessment.

    Parameters
    ----------
    range_resolution : float
        Range impulse response width (meters).
    azimuth_resolution : float
        Azimuth impulse response width (meters).
    graze_angle : float
        Grazing angle (radians).
    nesz_db : float
        Noise equivalent sigma-zero (dB).
    signal_sigma_db : float
        Signal sigma-zero level (dB). Default -10.
    multi_noise : float
        Multiplicative noise ratio (0 to 1). Default 0.

    Returns
    -------
    RGIQEResult
        Computed metrics including RNIIRS estimate.
    """
    # SNR computation
    snr_linear = 10 ** ((signal_sigma_db - nesz_db) / 10.0)

    # Account for multiplicative noise
    if multi_noise > 0:
        effective_snr = snr_linear / (1 + multi_noise * snr_linear)
    else:
        effective_snr = snr_linear

    snr_db = 10 * np.log10(max(effective_snr, 1e-10))

    # 2D bandwidth area (cycles^2/m^2)
    # Ground range resolution accounts for grazing angle
    ground_range_res = range_resolution / np.sin(max(graze_angle, 0.01))
    bandwidth_area = 1.0 / (ground_range_res * azimuth_resolution)

    # Shannon-Hartley information density (bits/m^2)
    info_density = bandwidth_area * np.log2(1.0 + effective_snr)

    # Map to RNIIRS (1-9 scale) via empirical polynomial fit
    log_info = np.log10(max(info_density, 1e-10))

    # Piecewise mapping: linear for very low, polynomial for normal range
    if log_info < 0:
        rniirs = max(1.0, 1.0 + log_info)
    else:
        # Empirical fit coefficients (approximation of NGA RGIQE curve)
        # RNIIRS ≈ a0 + a1*log10(info) + a2*log10(info)^2
        a0 = 3.7
        a1 = 1.6
        a2 = -0.1
        rniirs = a0 + a1 * log_info + a2 * log_info ** 2

    rniirs = np.clip(rniirs, 0.0, 9.0)

    return RGIQEResult(
        information_density=float(info_density),
        rniirs=float(rniirs),
        snr_db=float(snr_db),
        bandwidth_area=float(bandwidth_area),
        range_resolution=float(range_resolution),
        azimuth_resolution=float(azimuth_resolution),
        graze_angle=float(graze_angle),
        nesz_db=float(nesz_db)
    )


__all__ = ["RGIQEResult", "compute_rgiqe"]
