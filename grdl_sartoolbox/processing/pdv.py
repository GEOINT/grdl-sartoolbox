# -*- coding: utf-8 -*-
"""
Phase Difference Visualization (PDV).

Computes phase gradient of complex SAR imagery for visualization
of terrain slope and man-made features.

Attribution
-----------
Ported from MATLAB SAR Toolbox (https://github.com/ngageoint/MATLAB_SAR)
Original MATLAB implementation by NGA/IDT

License
-------
MIT License
Copyright (c) 2024 geoint.org
"""

import numpy as np
from scipy import ndimage
from typing import Optional, Tuple


def pdv_mem(
    complex_image: np.ndarray,
    delta_x: float = 0.25,
    filter_size: Tuple[int, int] = (5, 5),
    filter_type: str = 'mean',
    dim: int = 0
) -> np.ndarray:
    """
    Compute Phase Difference Visualization from complex SAR image.

    Computes the phase gradient (derivative) of a complex image in the
    specified dimension via frequency-domain conjugate multiplication.

    Parameters
    ----------
    complex_image : np.ndarray
        Complex SAR image, shape (rows, cols).
    delta_x : float
        Shift amount in pixels for phase difference. Default 0.25.
    filter_size : tuple of int
        Size of smoothing filter (rows, cols). Default (5, 5).
    filter_type : str
        Smoothing filter type: 'mean' or 'median'. Default 'mean'.
    dim : int
        Dimension for phase gradient: 0=rows, 1=cols.

    Returns
    -------
    np.ndarray
        Phase gradient image (radians/pixel), dtype float32.
    """
    if complex_image.ndim != 2:
        raise ValueError(f"Expected 2D image, got {complex_image.ndim}D")

    rows, cols = complex_image.shape
    n = rows if dim == 0 else cols

    # Create phase ramp in frequency domain
    freq = np.arange(n, dtype=np.float64) - n / 2
    phase_ramp = np.exp(1j * np.pi * delta_x * freq / (n / 2))

    # Apply shift via FFT
    ph = np.fft.fftshift(np.fft.fft(complex_image, axis=dim), axes=dim)

    if dim == 0:
        shifted_ph = ph * phase_ramp[:, np.newaxis]
    else:
        shifted_ph = ph * phase_ramp[np.newaxis, :]

    shifted = np.fft.ifft(np.fft.ifftshift(shifted_ph, axes=dim), axis=dim)

    # Phase difference via conjugate multiplication
    phase_diff = np.angle(np.conj(complex_image) * shifted) / delta_x

    # Apply smoothing filter
    if filter_type == 'median':
        phase_diff = ndimage.median_filter(
            phase_diff.astype(np.float64), size=filter_size
        )
    else:  # mean
        phase_diff = ndimage.uniform_filter(
            phase_diff.astype(np.float64), size=filter_size
        )

    return phase_diff.astype(np.float32)


__all__ = ["pdv_mem"]
