# -*- coding: utf-8 -*-
"""
Miscellaneous Utilities - Fast running mean and local sum.

Provides efficient running mean (boxcar filter) and integral image
based local sum computations.

Attribution
-----------
Ported from MATLAB SAR Toolbox (https://github.com/ngageoint/MATLAB_SAR)
Original: Neil Hodgson (fastrunmean), Wade Schwartzkopf (local_sum)

License
-------
MIT License
Copyright (c) 2024 geoint.org
"""

from typing import Tuple, Union
import numpy as np


def fast_running_mean(
    data: np.ndarray,
    window: Union[Tuple[int, ...], int],
    pad_type: str = 'zeros'
) -> np.ndarray:
    """
    Compute fast running mean using recursive algorithm.

    Uses an efficient recursive (integral-image-like) algorithm whose
    cost is independent of window size, proportional only to input size.

    Parameters
    ----------
    data : np.ndarray
        2D or 3D input array.
    window : int or tuple of int
        Window size for each dimension. Must be odd in each dimension.
    pad_type : str
        Padding type: 'zeros' or 'mean'. Default 'zeros'.

    Returns
    -------
    np.ndarray
        Running mean of input data, same shape as input.

    Raises
    ------
    ValueError
        If window dimensions don't match data dimensions or window sizes are even.
    """
    if np.isscalar(window):
        window = (int(window),) * data.ndim
    else:
        window = tuple(int(w) for w in window)

    if len(window) != data.ndim:
        raise ValueError(
            f"Window dimensions ({len(window)}) must match data dimensions ({data.ndim})"
        )

    for w in window:
        if w % 2 == 0:
            raise ValueError(f"Window sizes must be odd, got {window}")

    data = data.astype(np.float64)

    if data.ndim == 2:
        return _fast_running_mean_2d(data, window, pad_type)
    elif data.ndim == 3:
        return _fast_running_mean_3d(data, window, pad_type)
    else:
        raise ValueError(f"Only 2D and 3D arrays supported, got {data.ndim}D")


def _fast_running_mean_2d(
    data: np.ndarray,
    window: Tuple[int, int],
    pad_type: str
) -> np.ndarray:
    """2D fast running mean implementation."""
    nx, ny = data.shape
    winx, winy = window
    hwinx = (winx - 1) // 2
    hwiny = (winy - 1) // 2

    # Pad input
    padded_shape = (nx + 2 * hwinx, ny + 2 * hwiny)
    if pad_type == 'zeros':
        padded = np.zeros(padded_shape)
    elif pad_type == 'mean':
        padded = np.full(padded_shape, np.mean(data))
    else:
        raise ValueError(f"Unknown pad_type: {pad_type}")

    padded[hwinx:hwinx + nx, hwiny:hwiny + ny] = data

    # Recursive running sum along X
    cx = np.zeros((nx, ny + winy - 1))
    cx[0, :] = np.sum(padded[:winx, :], axis=0)
    for n in range(1, nx):
        cx[n, :] = cx[n - 1, :] - padded[n - 1, :] + padded[n + winx - 1, :]

    # Recursive running sum along Y
    cy = np.zeros((nx, ny))
    cy[:, 0] = np.sum(cx[:, :winy], axis=1)
    for n in range(1, ny):
        cy[:, n] = cy[:, n - 1] - cx[:, n - 1] + cx[:, n + winy - 1]

    return cy / (winx * winy)


def _fast_running_mean_3d(
    data: np.ndarray,
    window: Tuple[int, int, int],
    pad_type: str
) -> np.ndarray:
    """3D fast running mean implementation."""
    nx, ny, nz = data.shape
    winx, winy, winz = window
    hwinx = (winx - 1) // 2
    hwiny = (winy - 1) // 2
    hwinz = (winz - 1) // 2

    padded_shape = (nx + 2 * hwinx, ny + 2 * hwiny, nz + 2 * hwinz)
    if pad_type == 'zeros':
        padded = np.zeros(padded_shape)
    elif pad_type == 'mean':
        padded = np.full(padded_shape, np.mean(data))
    else:
        raise ValueError(f"Unknown pad_type: {pad_type}")

    padded[hwinx:hwinx + nx, hwiny:hwiny + ny, hwinz:hwinz + nz] = data

    # X pass
    cx = np.zeros((nx, ny + winy - 1, nz + winz - 1))
    cx[0, :, :] = np.sum(padded[:winx, :, :], axis=0)
    for n in range(1, nx):
        cx[n, :, :] = cx[n - 1, :, :] - padded[n - 1, :, :] + padded[n + winx - 1, :, :]

    # Y pass
    cy = np.zeros((nx, ny, nz + winz - 1))
    cy[:, 0, :] = np.sum(cx[:, :winy, :], axis=1)
    for n in range(1, ny):
        cy[:, n, :] = cy[:, n - 1, :] - cx[:, n - 1, :] + cx[:, n + winy - 1, :]

    # Z pass
    cz = np.zeros((nx, ny, nz))
    cz[:, :, 0] = np.sum(cy[:, :, :winz], axis=2)
    for n in range(1, nz):
        cz[:, :, n] = cz[:, :, n - 1] - cy[:, :, n - 1] + cy[:, :, n + winz - 1]

    return cz / (winx * winy * winz)


def local_sum(
    data: np.ndarray,
    window_size: Union[Tuple[int, int], int]
) -> np.ndarray:
    """
    Fast 2D local sum using integral images.

    Uses the Viola-Jones integral image approach for O(1) per-pixel
    local sum computation, independent of window size.

    Parameters
    ----------
    data : np.ndarray
        2D input array.
    window_size : int or tuple of int
        Window size as (rows, cols) or scalar for square window.

    Returns
    -------
    np.ndarray
        Local sum values, same shape as input.

    References
    ----------
    Viola and Jones, "Robust real-time face detection", International
    Journal of Computer Vision, 2004.
    """
    if np.isscalar(window_size):
        window_size = (int(window_size), int(window_size))
    else:
        window_size = (int(window_size[0]), int(window_size[1]))

    data = np.asarray(data, dtype=np.float64)
    pad_rows = window_size[0] // 2
    pad_cols = window_size[1] // 2

    # Pad with zeros: extra +1 on pre-pad for integral image
    padded = np.pad(data, ((pad_rows + 1, pad_rows), (pad_cols + 1, pad_cols)),
                    mode='constant', constant_values=0)

    # Create integral image
    integral = np.cumsum(np.cumsum(padded, axis=0), axis=1)

    # Compute local sums using four corners of integral image
    rows, cols = data.shape
    value = (integral[:rows, :cols]
             - integral[:rows, window_size[1]:]
             - integral[window_size[0]:, :cols]
             + integral[window_size[0]:, window_size[1]:])

    return value


__all__ = [
    "fast_running_mean",
    "local_sum",
]
