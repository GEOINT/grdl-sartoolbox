# -*- coding: utf-8 -*-
"""
SAR Remap Functions - Convert complex SAR data to displayable images.

Implements various remap/contrast-stretch functions for SAR image
visualization, including density, NRL, PEDF, log, and linear remaps.

Attribution
-----------
Ported from MATLAB SAR Toolbox (https://github.com/ngageoint/MATLAB_SAR)
Original: Wade Schwartzkopf, NGA/IDT

License
-------
MIT License
Copyright (c) 2024 geoint.org
"""

from typing import List, Optional, Callable, Dict
import numpy as np


def amplitude_to_density(
    data: np.ndarray,
    dmin: float = 30.0,
    mmult: float = 40.0,
    data_mean: Optional[float] = None
) -> np.ndarray:
    """
    Convert amplitude data to density values for visualization.

    This is the core remap algorithm used by density_remap, brighter_remap,
    darker_remap, and high_contrast_remap.

    Parameters
    ----------
    data : np.ndarray
        Input SAR data (complex or real).
    dmin : float
        Minimum density value (controls brightness). Default 30.
    mmult : float
        Multiplier for high threshold (controls contrast). Default 40.
    data_mean : float, optional
        Precomputed mean of |data|. If None, computed from data.

    Returns
    -------
    np.ndarray
        Density values as float64 array (typically 0-255 range).
    """
    eps = 1e-5
    A = np.abs(data).astype(np.float32)

    if data_mean is None:
        finite_mask = np.isfinite(A)
        data_mean = float(np.mean(A[finite_mask])) if np.any(finite_mask) else 1.0

    Cl = 0.8 * data_mean
    Ch = mmult * Cl
    m = (255.0 - dmin) / np.log10(Ch / max(Cl, eps))
    b = dmin - m * np.log10(max(Cl, eps))

    D = m * np.log10(np.maximum(A, eps)) + b
    return D


def density_remap(data: np.ndarray) -> np.ndarray:
    """
    Standard density remap for SAR visualization.

    Parameters
    ----------
    data : np.ndarray
        Complex or real SAR data.

    Returns
    -------
    np.ndarray
        uint8 image (0-255).
    """
    return np.clip(amplitude_to_density(data), 0, 255).astype(np.uint8)


def brighter_remap(data: np.ndarray) -> np.ndarray:
    """
    Brighter density remap (higher Dmin = 60).

    Parameters
    ----------
    data : np.ndarray
        Complex or real SAR data.

    Returns
    -------
    np.ndarray
        uint8 image (0-255).
    """
    return np.clip(amplitude_to_density(data, dmin=60, mmult=40), 0, 255).astype(np.uint8)


def darker_remap(data: np.ndarray) -> np.ndarray:
    """
    Darker density remap (Dmin = 0).

    Parameters
    ----------
    data : np.ndarray
        Complex or real SAR data.

    Returns
    -------
    np.ndarray
        uint8 image (0-255).
    """
    return np.clip(amplitude_to_density(data, dmin=0, mmult=40), 0, 255).astype(np.uint8)


def high_contrast_remap(data: np.ndarray) -> np.ndarray:
    """
    High contrast density remap (lower Mmult = 4).

    Parameters
    ----------
    data : np.ndarray
        Complex or real SAR data.

    Returns
    -------
    np.ndarray
        uint8 image (0-255).
    """
    return np.clip(amplitude_to_density(data, dmin=30, mmult=4), 0, 255).astype(np.uint8)


def linear_remap(data: np.ndarray) -> np.ndarray:
    """
    Simple linear remap (magnitude for complex, passthrough for real).

    Parameters
    ----------
    data : np.ndarray
        Complex or real SAR data.

    Returns
    -------
    np.ndarray
        float64 array of magnitudes (or real values).
    """
    if np.iscomplexobj(data):
        return np.abs(data).astype(np.float64)
    return data.astype(np.float64)


def log_remap(data: np.ndarray, span_db: float = 50.0) -> np.ndarray:
    """
    Logarithmic (dB) remap for SAR data.

    Parameters
    ----------
    data : np.ndarray
        Complex or real SAR data.
    span_db : float
        Dynamic range in dB. Default 50.

    Returns
    -------
    np.ndarray
        uint8 image (0-255).
    """
    A = np.abs(data).astype(np.float32)
    mean_val = float(np.mean(A))
    if mean_val > 0:
        A = 10.0 * A / mean_val

    # Ensure minimum of 1 for log
    A = A + max(1.0 - float(np.min(A)), 0.0)
    x = 20.0 * np.log10(np.maximum(A, 1e-10))

    # Compute center from RMS
    rcent = 10.0 * np.log10(max(float(np.mean(A ** 2)), 1e-10))

    disp_min = max(float(np.min(x)), rcent - span_db / 2.0)
    disp_max = min(float(np.max(x)), rcent + span_db / 2.0)

    if disp_max > disp_min:
        out = 255.0 * (x - disp_min) / (disp_max - disp_min)
    else:
        out = np.zeros_like(x)

    return np.clip(out, 0, 255).astype(np.uint8)


def _percentile(data: np.ndarray, p: float) -> float:
    """Compute percentile using linear interpolation."""
    return float(np.percentile(data.ravel(), p))


def nrl_remap(
    data: np.ndarray,
    a: float = 1.0,
    c: float = 220.0
) -> np.ndarray:
    """
    NRL lin-log style remap.

    Linear mapping below the knee point, logarithmic above.

    Parameters
    ----------
    data : np.ndarray
        Complex or real SAR data.
    a : float
        Scale factor for 99th percentile (input knee). Default 1.0.
    c : float
        Output knee value in the lin-log curve. Default 220.

    Returns
    -------
    np.ndarray
        uint8 image (0-255).
    """
    A = np.abs(data).astype(np.float32)
    Amin = float(np.min(A))
    P99 = _percentile(A, 99)
    Amax = float(np.max(A))

    knee = a * P99
    if knee <= Amin or Amax <= Amin:
        return np.zeros(data.shape, dtype=np.uint8)

    # Log scale coefficient
    log_denom = np.log10(max((Amax - Amin) / (knee - Amin), 1e-10))
    b = (255.0 - c) / max(log_denom, 1e-10)

    out = np.zeros(data.shape, dtype=np.float64)

    # Linear region
    linear_mask = A <= knee
    out[linear_mask] = (A[linear_mask] - Amin) * c / (knee - Amin)

    # Log region
    log_mask = ~linear_mask
    ratio = (A[log_mask] - Amin) / (knee - Amin)
    out[log_mask] = b * np.log10(np.maximum(ratio, 1e-10)) + c

    return np.clip(out, 0, 255).astype(np.uint8)


def pedf_remap(data: np.ndarray) -> np.ndarray:
    """
    Piecewise Extended Density Format (PEDF) remap.

    Standard density remap with compression of high values.

    Parameters
    ----------
    data : np.ndarray
        Complex or real SAR data.

    Returns
    -------
    np.ndarray
        uint8 image (0-255).
    """
    D = amplitude_to_density(data)
    D[D > 128] = 0.5 * (D[D > 128] + 128.0)
    return np.clip(D, 0, 255).astype(np.uint8)


# Registry of all available remap functions
_REMAP_REGISTRY: Dict[str, Callable] = {
    'density': density_remap,
    'brighter': brighter_remap,
    'darker': darker_remap,
    'highcontrast': high_contrast_remap,
    'linear': linear_remap,
    'log': log_remap,
    'nrl': nrl_remap,
    'pedf': pedf_remap,
}


def get_remap_list() -> List[str]:
    """
    Return list of available remap function names.

    Returns
    -------
    list of str
        Available remap function names.
    """
    return list(_REMAP_REGISTRY.keys())


def get_remap_function(name: str) -> Callable:
    """
    Get a remap function by name.

    Parameters
    ----------
    name : str
        Remap function name (e.g., 'density', 'nrl', 'log').

    Returns
    -------
    callable
        The remap function.

    Raises
    ------
    KeyError
        If name is not a recognized remap function.
    """
    if name.lower() not in _REMAP_REGISTRY:
        raise KeyError(
            f"Unknown remap '{name}'. Available: {get_remap_list()}"
        )
    return _REMAP_REGISTRY[name.lower()]


__all__ = [
    "amplitude_to_density",
    "density_remap",
    "brighter_remap",
    "darker_remap",
    "high_contrast_remap",
    "linear_remap",
    "log_remap",
    "nrl_remap",
    "pedf_remap",
    "get_remap_list",
    "get_remap_function",
]
