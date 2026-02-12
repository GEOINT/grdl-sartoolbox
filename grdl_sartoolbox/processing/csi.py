# -*- coding: utf-8 -*-
"""
Color Sub-aperture Image (CSI) visualization.

Creates color subaperture images that encode frequency/subaperture
information in RGB channels, useful for detecting moving targets and
understanding SAR frequency content.

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
from typing import Optional


def _jet_wrapped(n: int) -> np.ndarray:
    """
    Create jet-like colormap that wraps around for circular filtering.

    Parameters
    ----------
    n : int
        Number of entries.

    Returns
    -------
    np.ndarray
        RGB colormap, shape (n, 3).
    """
    x = np.linspace(0, 1, n)

    # Red channel: ramp up 0.35-0.65, flat 0.65-0.9, ramp down 0.9-1.0, ramp up 0-0.1
    r = np.interp(x, [0, 0.1, 0.35, 0.65, 0.9, 1.0], [0.5, 0, 0, 1, 1, 0.5])
    # Green channel: ramp up 0.1-0.35, flat 0.35-0.65, ramp down 0.65-0.9
    g = np.interp(x, [0, 0.1, 0.35, 0.65, 0.9, 1.0], [0, 0, 1, 1, 0, 0])
    # Blue channel: ramp up 0.65-0.9, flat 0.9-1.0, ramp down 0-0.1, flat 0.1-0.35
    b = np.interp(x, [0, 0.1, 0.35, 0.65, 0.9, 1.0], [0.5, 1, 1, 0, 0, 0.5])

    return np.column_stack([r, g, b])


def csi_mem(
    complex_image: np.ndarray,
    dim: int = 0,
    fill: float = 1.0,
    platform_dir: str = 'right',
    full_res: bool = True
) -> np.ndarray:
    """
    Create Color Sub-aperture Image (CSI) from complex SAR data.

    Transforms complex SAR image into a color visualization that encodes
    frequency/subaperture information in RGB channels. Moving targets
    appear as colored streaks, while stationary targets appear white/gray.

    Parameters
    ----------
    complex_image : np.ndarray
        Complex SAR image, shape (rows, cols).
    dim : int
        Dimension to decompose: 0=azimuth (rows), 1=range (cols).
    fill : float
        Fill fraction of bandwidth (0 to 1). Default 1.0.
    platform_dir : str
        Platform direction: 'right' or 'left'. Affects color ordering.
    full_res : bool
        If True, preserve original resolution intensity. Default True.

    Returns
    -------
    np.ndarray
        RGB image, shape (rows, cols, 3), dtype float32.
        Values normalized to [0, 1].
    """
    if complex_image.ndim != 2:
        raise ValueError(f"Expected 2D image, got {complex_image.ndim}D")

    rows, cols = complex_image.shape
    n = rows if dim == 0 else cols

    # Transform to phase history domain
    ph = np.fft.fftshift(np.fft.fft(complex_image, axis=dim), axes=dim)

    # Create color filters
    cmap = _jet_wrapped(n)

    # Adjust for platform direction
    if platform_dir == 'left':
        cmap = cmap[::-1]

    # Apply fill factor (zero-pad center if fill < 1)
    if fill < 1.0:
        center = n // 2
        half_fill = int(n * fill / 2)
        mask = np.zeros(n)
        mask[center - half_fill:center + half_fill] = 1.0
        if dim == 0:
            mask_2d = mask[:, np.newaxis]
        else:
            mask_2d = mask[np.newaxis, :]
        ph = ph * mask_2d

    # Apply color filters and inverse transform
    rgb = np.zeros((rows, cols, 3), dtype=np.complex128)

    for c in range(3):
        if dim == 0:
            colored = ph * cmap[:, c][:, np.newaxis]
        else:
            colored = ph * cmap[:, c][np.newaxis, :]

        # Inverse transform
        colored = np.fft.ifft(np.fft.ifftshift(colored, axes=dim), axis=dim)
        rgb[:, :, c] = colored

    # Convert to magnitude
    rgb_mag = np.abs(rgb).astype(np.float64)

    if full_res:
        # Preserve original intensity, keep subaperture color
        orig_mag = np.abs(complex_image)
        color_mag = np.sqrt(np.sum(rgb_mag ** 2, axis=2))
        scale = np.zeros_like(color_mag)
        nonzero = color_mag > 0
        scale[nonzero] = orig_mag[nonzero] / color_mag[nonzero]
        rgb_mag *= scale[:, :, np.newaxis]

    # Normalize to [0, 1]
    max_val = rgb_mag.max()
    if max_val > 0:
        rgb_mag /= max_val

    return rgb_mag.astype(np.float32)


__all__ = ["csi_mem"]
