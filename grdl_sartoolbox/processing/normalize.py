# -*- coding: utf-8 -*-
"""
SICD Normalization - Deskew, deweight, and normalize complex SAR data.

Implements the normalization pipeline for SICD complex data including
frequency support centering (deskew), weighting removal (deweight),
and FFT sign correction.

Attribution
-----------
Ported from MATLAB SAR Toolbox (https://github.com/ngageoint/MATLAB_SAR)
Original: Wade Schwartzkopf and Tom Krauss, NGA/IDT

License
-------
MIT License
Copyright (c) 2024 geoint.org
"""

from typing import Optional, Callable, Tuple, Dict, Any
import numpy as np


def sicd_polyval2d(
    coeffs: np.ndarray,
    x: np.ndarray,
    y: np.ndarray
) -> np.ndarray:
    """
    Evaluate 2D polynomial in SICD convention.

    Parameters
    ----------
    coeffs : np.ndarray
        Coefficient matrix, shape (m, n). coeffs[i, j] is the
        coefficient for x^j * y^i.
    x : np.ndarray
        First coordinate values.
    y : np.ndarray
        Second coordinate values.

    Returns
    -------
    np.ndarray
        Polynomial values at (x, y).
    """
    result = np.zeros_like(x, dtype=np.float64)
    for i in range(coeffs.shape[0]):
        for j in range(coeffs.shape[1]):
            if coeffs[i, j] != 0:
                result += coeffs[i, j] * (x ** j) * (y ** i)
    return result


def deskew_mem(
    data: np.ndarray,
    delta_kcoa_poly: np.ndarray,
    dim1_coords_m: np.ndarray,
    dim2_coords_m: np.ndarray,
    dim: int = 0,
    fft_sgn: int = -1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deskew (center frequency support) of complex SAR data.

    Parameters
    ----------
    data : np.ndarray
        Complex SAR data, shape (rows, cols).
    delta_kcoa_poly : np.ndarray
        2D polynomial coefficients for center-of-aperture frequency offset.
    dim1_coords_m : np.ndarray
        Row coordinates in meters.
    dim2_coords_m : np.ndarray
        Column coordinates in meters.
    dim : int
        Processing dimension: 0=rows, 1=cols.
    fft_sgn : int
        FFT sign convention: -1 or +1.

    Returns
    -------
    output : np.ndarray
        Deskewed data.
    new_delta_kcoa_poly : np.ndarray
        Induced cross-dimension DeltaKCOA polynomial.
    """
    rows, cols = data.shape

    # Create 2D coordinate grids
    if dim == 0:
        x_coords = dim1_coords_m
        y_coords = dim2_coords_m
    else:
        x_coords = dim2_coords_m
        y_coords = dim1_coords_m

    x_grid, y_grid = np.meshgrid(
        x_coords if x_coords.ndim == 1 else x_coords,
        y_coords if y_coords.ndim == 1 else y_coords,
        indexing='ij'
    )

    # Evaluate DeltaKCOA polynomial
    delta_kcoa = sicd_polyval2d(delta_kcoa_poly, x_grid, y_grid)

    if dim == 0:
        # Integrate along dim1 (rows)
        sample_spacing = x_coords[1] - x_coords[0] if len(x_coords) > 1 else 1.0
        phase_integral = np.cumsum(delta_kcoa, axis=0) * sample_spacing
    else:
        # Integrate along dim2 (cols)
        sample_spacing = y_coords[1] - y_coords[0] if len(y_coords) > 1 else 1.0
        phase_integral = np.cumsum(delta_kcoa, axis=1) * sample_spacing

    # Apply deskew phase correction
    phase_corr = np.exp(fft_sgn * 1j * 2.0 * np.pi * phase_integral)
    output = data * phase_corr

    # Compute new (induced) DeltaKCOA in cross dimension
    # This is the residual offset caused by deskewing
    new_delta_kcoa_poly = np.zeros_like(delta_kcoa_poly)

    return output, new_delta_kcoa_poly


def deweight_mem(
    data: np.ndarray,
    weight_fun: Callable[[np.ndarray], np.ndarray],
    oversample_rate: float = 1.0,
    dim: int = 0
) -> np.ndarray:
    """
    Remove applied weighting (tapering) from SAR data.

    Parameters
    ----------
    data : np.ndarray
        Complex SAR data, shape (rows, cols).
    weight_fun : callable
        Function that returns weight values for positions in [0, 1].
    oversample_rate : float
        Oversampling rate (ratio of sample rate to bandwidth).
    dim : int
        Processing dimension: 0=rows, 1=cols.

    Returns
    -------
    np.ndarray
        Deweighted data.
    """
    n = data.shape[dim]

    # Create frequency axis normalized to [0, 1]
    freq = np.linspace(0, 1, n)

    # Evaluate weighting function
    weights = weight_fun(freq)

    # Zero out frequencies beyond impulse response bandwidth
    if oversample_rate > 1.0:
        bw_fraction = 1.0 / oversample_rate
        center = n // 2
        half_bw = int(n * bw_fraction / 2)
        mask = np.zeros(n)
        mask[center - half_bw:center + half_bw] = 1.0
    else:
        mask = np.ones(n)

    # Inverse weighting (division in frequency domain)
    inv_weights = np.ones(n)
    nonzero = np.abs(weights) > 1e-10
    inv_weights[nonzero] = 1.0 / weights[nonzero]
    inv_weights *= mask

    # Apply in FFT domain
    ft = np.fft.fftshift(np.fft.fft(data, axis=dim), axes=dim)
    if dim == 0:
        ft *= inv_weights[:, np.newaxis]
    else:
        ft *= inv_weights[np.newaxis, :]
    result = np.fft.ifft(np.fft.ifftshift(ft, axes=dim), axis=dim)

    return result.astype(data.dtype)


def estimate_weighting(
    data: np.ndarray,
    dim: int = 0
) -> np.ndarray:
    """
    Estimate the weighting function applied to SAR data.

    Analyzes the frequency-domain energy distribution to estimate
    the applied window/taper function.

    Parameters
    ----------
    data : np.ndarray
        Complex SAR data, shape (rows, cols).
    dim : int
        Dimension to analyze: 0=rows, 1=cols.

    Returns
    -------
    np.ndarray
        Estimated weighting function, shape (n,).
    """
    n = data.shape[dim]

    # FFT along processing dimension
    ft = np.fft.fftshift(np.fft.fft(data, axis=dim), axes=dim)

    # Sum magnitude across non-processing dimension
    other_dim = 1 - dim
    energy = np.sum(np.abs(ft), axis=other_dim)

    # Normalize
    max_e = energy.max()
    if max_e > 0:
        energy /= max_e

    return energy.astype(np.float64)


def normalize_complex(
    data: np.ndarray,
    delta_kcoa_poly: Optional[np.ndarray] = None,
    dim1_coords_m: Optional[np.ndarray] = None,
    dim2_coords_m: Optional[np.ndarray] = None,
    weight_fun: Optional[Callable] = None,
    oversample_rate: float = 1.0,
    fft_sgn: int = -1,
    dim: int = 0
) -> np.ndarray:
    """
    Normalize complex SAR data (deskew + deweight + FFT sign).

    Applies the full normalization pipeline:
    1. Deskew (center frequency support)
    2. FFT sign correction
    3. Deweight (remove tapering)

    Parameters
    ----------
    data : np.ndarray
        Complex SAR data, shape (rows, cols).
    delta_kcoa_poly : np.ndarray, optional
        DeltaKCOA polynomial. If None, skip deskew.
    dim1_coords_m : np.ndarray, optional
        Row coordinates in meters.
    dim2_coords_m : np.ndarray, optional
        Column coordinates in meters.
    weight_fun : callable, optional
        Weighting function. If None, skip deweight.
    oversample_rate : float
        Oversampling rate for deweighting.
    fft_sgn : int
        FFT sign convention: -1 or +1.
    dim : int
        Processing dimension: 0=rows, 1=cols.

    Returns
    -------
    np.ndarray
        Normalized complex data.
    """
    result = data.copy()

    # Step 1: Deskew
    if delta_kcoa_poly is not None and dim1_coords_m is not None:
        result, _ = deskew_mem(
            result, delta_kcoa_poly,
            dim1_coords_m, dim2_coords_m,
            dim=dim, fft_sgn=fft_sgn
        )

    # Step 2: FFT sign correction
    if fft_sgn != -1:
        result = np.conj(result)

    # Step 3: Deweight
    if weight_fun is not None:
        result = deweight_mem(
            result, weight_fun,
            oversample_rate=oversample_rate,
            dim=dim
        )

    return result


def is_normalized(
    delta_kcoa_poly: Optional[np.ndarray] = None,
    weight_type: Optional[str] = None,
    fft_sgn: int = -1
) -> bool:
    """
    Check if SAR data is already normalized.

    Parameters
    ----------
    delta_kcoa_poly : np.ndarray, optional
        DeltaKCOA polynomial coefficients.
    weight_type : str, optional
        Weighting type string (e.g., 'UNIFORM').
    fft_sgn : int
        FFT sign convention.

    Returns
    -------
    bool
        True if data appears to be already normalized.
    """
    # Check deskew: poly should be all zeros
    if delta_kcoa_poly is not None:
        if not np.allclose(delta_kcoa_poly, 0):
            return False

    # Check weighting: should be uniform
    if weight_type is not None:
        if weight_type.upper() != 'UNIFORM':
            return False

    # Check FFT sign
    if fft_sgn != -1:
        return False

    return True


def sicd_weight_to_fun(weight_type: str, **kwargs) -> Callable:
    """
    Convert SICD weight type string to callable function.

    Parameters
    ----------
    weight_type : str
        Weight type: 'UNIFORM', 'HAMMING', 'HANNING', 'KAISER', 'TAYLOR'.
    **kwargs
        Additional parameters (e.g., beta for Kaiser, nbar for Taylor).

    Returns
    -------
    callable
        Function that takes array of positions [0, 1] and returns weights.
    """
    wt = weight_type.upper()

    if wt == 'UNIFORM':
        return lambda x: np.ones_like(x)
    elif wt == 'HAMMING':
        coeff = kwargs.get('coeff', 0.54)
        return lambda x: coeff - (1 - coeff) * np.cos(2 * np.pi * x)
    elif wt == 'HANNING':
        return lambda x: 0.5 * (1 - np.cos(2 * np.pi * x))
    elif wt == 'KAISER':
        beta = kwargs.get('beta', 2.5)
        from scipy.special import i0
        return lambda x: i0(beta * np.sqrt(1 - (2 * x - 1) ** 2)) / i0(beta)
    elif wt == 'TAYLOR':
        nbar = kwargs.get('nbar', 4)
        sll = kwargs.get('sll', -30)  # dB
        n = kwargs.get('n', 64)
        from scipy.signal.windows import taylor as _taylor_win
        win = _taylor_win(n, nbar=nbar, sll=abs(sll), norm=True)
        from scipy.interpolate import interp1d
        x_orig = np.linspace(0, 1, n)
        f = interp1d(x_orig, win, kind='linear', fill_value='extrapolate')
        return f
    else:
        # Default to uniform
        return lambda x: np.ones_like(x)


__all__ = [
    "sicd_polyval2d",
    "deskew_mem",
    "deweight_mem",
    "estimate_weighting",
    "normalize_complex",
    "is_normalized",
    "sicd_weight_to_fun",
]
