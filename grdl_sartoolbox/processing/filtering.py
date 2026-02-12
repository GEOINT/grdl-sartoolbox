# -*- coding: utf-8 -*-
"""
SAR Image Filtering - Apodization and upsampling.

Implements:
- 2D Spatially Variant Apodization (SVA) for sidelobe control
- FFT-based image upsampling/downsampling

Attribution
-----------
Ported from MATLAB SAR Toolbox (https://github.com/ngageoint/MATLAB_SAR)

License
-------
MIT License
Copyright (c) 2024 geoint.org
"""

import numpy as np


def _apodize_real(data: np.ndarray, k_h: int, k_v: int) -> np.ndarray:
    """
    Apply spatially variant apodization to real-valued data.

    For each pixel, computes weighted sums of neighbors and selects
    the result with minimum absolute value (or zero if sign changes).
    """
    rows, cols = data.shape
    result = np.zeros_like(data)

    for m in range(k_v, rows - k_v):
        for n in range(k_h, cols - k_h):
            val = data[m, n]

            # Three candidate weighted sums
            c1 = val + 0.5 * (data[m, n - k_h] + data[m, n + k_h])
            c2 = val + 0.5 * (data[m - k_v, n] + data[m + k_v, n])
            c3 = val + 0.25 * (
                data[m, n - k_h] + data[m, n + k_h] +
                data[m - k_v, n] + data[m + k_v, n]
            )

            candidates = [val, c1, c2, c3]

            # Check for sign changes (zero crossing)
            signs = [np.sign(c) for c in candidates]
            if len(set(s for s in signs if s != 0)) > 1:
                result[m, n] = 0.0
            else:
                # Select minimum absolute value
                result[m, n] = min(candidates, key=abs)

    return result


def apodize_2d(
    image: np.ndarray,
    k_h: int = 1,
    k_v: int = 1
) -> np.ndarray:
    """
    Apply 2D spatially variant apodization for sidelobe control.

    Processes complex image by applying apodization separately to
    real and imaginary parts.

    Parameters
    ----------
    image : np.ndarray
        Complex SAR image, shape (rows, cols).
    k_h : int
        Horizontal kernel offset (pixels). Default 1.
    k_v : int
        Vertical kernel offset (pixels). Default 1.

    Returns
    -------
    np.ndarray
        Apodized image, same shape and dtype as input.
    """
    if np.iscomplexobj(image):
        real_out = _apodize_real(image.real.copy(), k_h, k_v)
        imag_out = _apodize_real(image.imag.copy(), k_h, k_v)
        return (real_out + 1j * imag_out).astype(image.dtype)
    else:
        return _apodize_real(image.copy(), k_h, k_v).astype(image.dtype)


def upsample_image(
    image: np.ndarray,
    row_factor: float = 2.0,
    col_factor: float = 2.0
) -> np.ndarray:
    """
    Upsample (or downsample) image using FFT zero-padding.

    Parameters
    ----------
    image : np.ndarray
        Input image (real or complex), shape (rows, cols).
    row_factor : float
        Row upsampling factor. >1 upsamples, <1 downsamples.
    col_factor : float
        Column upsampling factor. >1 upsamples, <1 downsamples.

    Returns
    -------
    np.ndarray
        Resampled image.
    """
    rows, cols = image.shape
    new_rows = int(np.round(rows * row_factor))
    new_cols = int(np.round(cols * col_factor))

    # Forward FFT
    ft = np.fft.fftshift(np.fft.fft2(image))

    # Zero-pad or crop in frequency domain
    padded = np.zeros((new_rows, new_cols), dtype=ft.dtype)

    # Copy centered frequency content
    r_start_src = max(0, (rows - new_rows) // 2)
    r_start_dst = max(0, (new_rows - rows) // 2)
    c_start_src = max(0, (cols - new_cols) // 2)
    c_start_dst = max(0, (new_cols - cols) // 2)

    r_copy = min(rows, new_rows)
    c_copy = min(cols, new_cols)

    padded[r_start_dst:r_start_dst + r_copy,
           c_start_dst:c_start_dst + c_copy] = \
        ft[r_start_src:r_start_src + r_copy,
           c_start_src:c_start_src + c_copy]

    # Inverse FFT with energy normalization
    result = np.fft.ifft2(np.fft.ifftshift(padded))
    result *= (new_rows * new_cols) / (rows * cols)

    if not np.iscomplexobj(image):
        result = np.real(result)

    return result.astype(image.dtype)


__all__ = ["apodize_2d", "upsample_image"]
