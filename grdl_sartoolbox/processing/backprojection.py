# -*- coding: utf-8 -*-
"""
Backprojection - Basic time-domain SAR image formation algorithm.

Implements the backprojection algorithm for forming SAR images from
phase history (CPHD) data. Backprojection is a time-domain algorithm
that projects range profiles onto an image grid.

Attribution
-----------
Ported from MATLAB SAR Toolbox (https://github.com/ngageoint/MATLAB_SAR)
Original implementation by LeRoy Gorham, Air Force Research Laboratory
Modified by Wade Schwartzkopf (NGA) and Daniel Andre (Dstl)

Reference
---------
Gorham, L.A. and Moore, L.J., "SAR image formation toolbox for MATLAB,"
Algorithms for Synthetic Aperture Radar Imagery XVII, SPIE (2010).

Dependencies
------------
numpy - Array operations, FFT
scipy.interpolate - Range profile interpolation

Author
------
Claude Sonnet 4.5 (automated port)

License
-------
MIT License
Copyright (c) 2024 geoint.org

Created
-------
2026-02-11

Modified
--------
2026-02-11
"""

# Standard library
from typing import Annotated, Optional, Tuple
from dataclasses import dataclass

# Third-party
import numpy as np
from scipy import interpolate

# GRDL internal - note: not using ImageProcessor base for this specialized algorithm
from grdl_sartoolbox.utils.constants import SPEED_OF_LIGHT


# ===================================================================
# Data Structures
# ===================================================================

@dataclass
class SensorPosition:
    """3D position of sensor (transmitter or receiver)."""
    X: np.ndarray  # X-position at each pulse (meters)
    Y: np.ndarray  # Y-position at each pulse (meters)
    Z: np.ndarray  # Z-position at each pulse (meters)

    def __post_init__(self):
        """Validate that all arrays have same length."""
        if not (len(self.X) == len(self.Y) == len(self.Z)):
            raise ValueError(
                f"Position arrays must have same length. "
                f"Got X:{len(self.X)}, Y:{len(self.Y)}, Z:{len(self.Z)}"
            )


@dataclass
class BackprojectionData:
    """
    Input data for backprojection algorithm.

    Attributes
    ----------
    phdata : np.ndarray
        Phase history data (complex), shape (num_samples, num_pulses).
        Fast time in rows, slow time in columns. Frequency domain data.
    delta_f : np.ndarray
        Frequency step size for each pulse (Hz), shape (num_pulses,).
    min_f : np.ndarray
        Start frequency for each pulse (Hz), shape (num_pulses,).
    tx_pos : SensorPosition
        Transmitter position at each pulse (meters).
    R0 : np.ndarray
        Bistatic range to motion compensation point (meters), shape (num_pulses,).
    x_mat : np.ndarray
        X-position of each pixel (meters), shape (rows, cols).
    y_mat : np.ndarray
        Y-position of each pixel (meters), shape (rows, cols).
    z_mat : np.ndarray
        Z-position of each pixel (meters), shape (rows, cols).
    rcv_pos : Optional[SensorPosition]
        Receiver position at each pulse. If None, monostatic (Rcv = Tx) assumed.
    nfft : Optional[int]
        FFT size for range profile. If None, auto-computed as 2^(3+nextpow2(num_samples)).
    """
    phdata: np.ndarray
    delta_f: np.ndarray
    min_f: np.ndarray
    tx_pos: SensorPosition
    R0: np.ndarray
    x_mat: np.ndarray
    y_mat: np.ndarray
    z_mat: np.ndarray
    rcv_pos: Optional[SensorPosition] = None
    nfft: Optional[int] = None

    def __post_init__(self):
        """Validate input data."""
        num_samples, num_pulses = self.phdata.shape

        # Validate pulse-wise data
        if len(self.delta_f) != num_pulses:
            raise ValueError(f"delta_f length {len(self.delta_f)} != num_pulses {num_pulses}")
        if len(self.min_f) != num_pulses:
            raise ValueError(f"min_f length {len(self.min_f)} != num_pulses {num_pulses}")
        if len(self.R0) != num_pulses:
            raise ValueError(f"R0 length {len(self.R0)} != num_pulses {num_pulses}")
        if len(self.tx_pos.X) != num_pulses:
            raise ValueError(f"tx_pos length {len(self.tx_pos.X)} != num_pulses {num_pulses}")

        # Validate grid
        if not (self.x_mat.shape == self.y_mat.shape == self.z_mat.shape):
            raise ValueError(
                f"Grid arrays must have same shape. "
                f"x:{self.x_mat.shape}, y:{self.y_mat.shape}, z:{self.z_mat.shape}"
            )

        # Default to monostatic if no receiver position
        if self.rcv_pos is None:
            self.rcv_pos = self.tx_pos

        # Auto-compute FFT size if not provided
        if self.nfft is None:
            # Use power-of-2 FFT with extra zero-padding for interpolation quality
            self.nfft = 2 ** (3 + int(np.ceil(np.log2(num_samples))))


# ===================================================================
# Backprojection Algorithm
# ===================================================================

def backproject_basic(
    data: BackprojectionData,
    verbose: bool = False
) -> np.ndarray:
    """
    Perform basic backprojection SAR image formation.

    This function implements the time-domain backprojection algorithm for
    SAR image formation. For each pulse, it:
    1. Forms a range profile via IFFT of the phase history data
    2. Calculates the differential (bistatic) range to each pixel
    3. Applies phase correction
    4. Interpolates the range profile and accumulates into the image

    Parameters
    ----------
    data : BackprojectionData
        Input phase history data and geometry.
    verbose : bool
        If True, print progress messages. Default False.

    Returns
    -------
    np.ndarray
        Complex SAR image, same shape as input grid (x_mat, y_mat, z_mat).

    Examples
    --------
    >>> # Create synthetic data for a point target
    >>> num_pulses = 100
    >>> num_samples = 256
    >>> phdata = np.random.randn(num_samples, num_pulses) + \\
    ...          1j*np.random.randn(num_samples, num_pulses)
    >>> delta_f = np.ones(num_pulses) * 1e6  # 1 MHz steps
    >>> min_f = np.ones(num_pulses) * 9e9    # 9 GHz start
    >>> tx_x = np.linspace(-50, 50, num_pulses)  # Linear flight path
    >>> tx_y = np.zeros(num_pulses)
    >>> tx_z = np.ones(num_pulses) * 1000    # 1km altitude
    >>> tx_pos = SensorPosition(tx_x, tx_y, tx_z)
    >>> R0 = np.ones(num_pulses) * 1000      # Range to scene center
    >>>
    >>> # Image grid
    >>> x = np.linspace(-10, 10, 128)
    >>> y = np.linspace(-10, 10, 128)
    >>> x_mat, y_mat = np.meshgrid(x, y)
    >>> z_mat = np.zeros_like(x_mat)
    >>>
    >>> # Form image
    >>> bp_data = BackprojectionData(
    ...     phdata=phdata, delta_f=delta_f, min_f=min_f,
    ...     tx_pos=tx_pos, R0=R0,
    ...     x_mat=x_mat, y_mat=y_mat, z_mat=z_mat
    ... )
    >>> image = backproject_basic(bp_data)
    >>> image.shape
    (128, 128)

    Notes
    -----
    - This is a straightforward implementation; optimized versions use
      fast factorized backprojection or GPU acceleration
    - Linear interpolation is used; for better quality, increase nfft
    - Monostatic case is assumed if rcv_pos is not provided
    """
    num_samples, num_pulses = data.phdata.shape

    # Maximum scene extent in range (two-way: transmit + receive)
    mean_delta_f = np.mean(data.delta_f)
    max_extent_range = SPEED_OF_LIGHT / mean_delta_f

    # Range vector for interpolation (meters)
    # Centered at zero with extent based on FFT size
    range_vector = np.linspace(
        -data.nfft / 2,
        data.nfft / 2 - 1,
        data.nfft
    ) * max_extent_range / data.nfft

    # Initialize output image
    complex_image = np.zeros(data.x_mat.shape, dtype=np.complex128)

    # Loop through each pulse
    for pulse_idx in range(num_pulses):
        if verbose and (pulse_idx % 10 == 0):
            print(f"Processing pulse {pulse_idx + 1}/{num_pulses}")

        # Form range profile with zero padding via IFFT
        # MATLAB: rc = fftshift(ifft(data.phdata(:,ii), data.Nfft))
        range_profile = np.fft.fftshift(
            np.fft.ifft(data.phdata[:, pulse_idx], n=data.nfft)
        )

        # Calculate differential (bistatic) range to each pixel
        # Range from transmitter to pixel
        tx_range = np.sqrt(
            (data.tx_pos.X[pulse_idx] - data.x_mat) ** 2 +
            (data.tx_pos.Y[pulse_idx] - data.y_mat) ** 2 +
            (data.tx_pos.Z[pulse_idx] - data.z_mat) ** 2
        )

        # Range from receiver to pixel
        rcv_range = np.sqrt(
            (data.rcv_pos.X[pulse_idx] - data.x_mat) ** 2 +
            (data.rcv_pos.Y[pulse_idx] - data.y_mat) ** 2 +
            (data.rcv_pos.Z[pulse_idx] - data.z_mat) ** 2
        )

        # Differential range (subtract motion compensation point range)
        dR = tx_range + rcv_range - 2.0 * data.R0[pulse_idx]

        # Phase correction for residual range
        # MATLAB: phCorr = exp(1i*2*pi*data.minF(ii)/c*dR)
        phase_corr = np.exp(
            1j * 2.0 * np.pi * data.min_f[pulse_idx] / SPEED_OF_LIGHT * dR
        )

        # Find pixels within range swath
        # MATLAB: I = find(and(dR > min(range_vector), dR < max(range_vector)))
        valid_mask = (dR > range_vector.min()) & (dR < range_vector.max())

        # Interpolate range profile at pixel ranges
        # MATLAB: interp1(range_vector, rc, dR(I), 'linear')
        # Use scipy for 1D interpolation (faster than np.interp for this use case)
        interp_func = interpolate.interp1d(
            range_vector,
            range_profile,
            kind='linear',
            bounds_error=False,
            fill_value=0.0
        )

        # Get interpolated values for valid pixels
        dR_valid = dR[valid_mask]
        interp_values = interp_func(dR_valid)

        # Accumulate into image with phase correction
        # MATLAB: complex_image(I) = complex_image(I) + interp1(...) .* phCorr(I)
        update = np.zeros_like(complex_image, dtype=np.complex128)
        update[valid_mask] = interp_values * phase_corr[valid_mask]
        complex_image += update

    return complex_image.astype(np.complex64)


# ===================================================================
# Convenience Functions
# ===================================================================

def create_image_grid(
    center: Tuple[float, float, float],
    size: Tuple[float, float],
    resolution: Tuple[float, float],
    ground_plane_z: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a 2D image grid for backprojection.

    Parameters
    ----------
    center : tuple of float
        Scene center (x, y, z) in meters.
    size : tuple of float
        Scene size (width_x, width_y) in meters.
    resolution : tuple of float
        Pixel spacing (dx, dy) in meters.
    ground_plane_z : float, optional
        Z-height for ground plane. If None, uses center[2].

    Returns
    -------
    x_mat : np.ndarray
        X-coordinates of pixels (rows, cols).
    y_mat : np.ndarray
        Y-coordinates of pixels (rows, cols).
    z_mat : np.ndarray
        Z-coordinates of pixels (rows, cols).

    Examples
    --------
    >>> # Create 100m x 100m grid with 1m resolution
    >>> x, y, z = create_image_grid(
    ...     center=(0, 0, 0),
    ...     size=(100, 100),
    ...     resolution=(1.0, 1.0)
    ... )
    >>> x.shape
    (100, 100)
    """
    cx, cy, cz = center
    width_x, width_y = size
    dx, dy = resolution

    if ground_plane_z is not None:
        cz = ground_plane_z

    # Number of pixels
    nx = int(np.round(width_x / dx))
    ny = int(np.round(width_y / dy))

    # Create 1D arrays
    x = np.linspace(cx - width_x/2, cx + width_x/2, nx)
    y = np.linspace(cy - width_y/2, cy + width_y/2, ny)

    # Create 2D grids
    x_mat, y_mat = np.meshgrid(x, y)
    z_mat = np.full_like(x_mat, cz)

    return x_mat, y_mat, z_mat


__all__ = [
    "backproject_basic",
    "BackprojectionData",
    "SensorPosition",
    "create_image_grid",
]
