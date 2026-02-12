# -*- coding: utf-8 -*-
"""
Coherent Change Detection - Detect changes between co-registered SAR image pairs.

Implements the Coherent Change Detection (CCD) algorithm as described in
Jakowatz, et al., "Spotlight-mode Synthetic Aperture Radar: A Signal Processing
Approach" (equation 5.102).

Also includes:
- Noise-aware CCD (ccdnoisemem) - ML estimator accounting for noise
- SCCM (Signal Correlation Change Metric) - Mitchell's improved CCD metric
- Angle-aware CCD (ccdmem_angle) - CCD with rotation support

Attribution
-----------
Ported from MATLAB SAR Toolbox (https://github.com/ngageoint/MATLAB_SAR)
Original MATLAB implementation by Wade Schwartzkopf, NGA/IDT

License
-------
MIT License
Copyright (c) 2024 geoint.org
"""

from typing import Annotated, Tuple, Optional
import numpy as np
from scipy import signal
from scipy.ndimage import rotate as ndimage_rotate

from grdl.image_processing.base import ImageProcessor
from grdl.image_processing.params import Desc, Range
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ImageModality, ProcessorCategory


# ===================================================================
# Helpers
# ===================================================================

def _validate_complex_2d(image: np.ndarray, param_name: str = "image") -> None:
    """Validate that *image* is a 2D complex numpy array."""
    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"{param_name} must be a numpy ndarray, got {type(image).__name__}"
        )
    if not np.iscomplexobj(image):
        raise TypeError(
            f"{param_name} must be complex-valued (complex64 or complex128), "
            f"got {image.dtype}. Pass complex SAR imagery from the reader."
        )
    if image.ndim != 2:
        raise ValueError(
            f"{param_name} must be 2D (rows, cols), got {image.ndim}D "
            f"with shape {image.shape}"
        )


def _ensure_window_tuple(corr_window_size):
    """Ensure window size is a 2-tuple (rows, cols)."""
    if np.isscalar(corr_window_size):
        return (int(corr_window_size), int(corr_window_size))
    return (int(corr_window_size[0]), int(corr_window_size[1]))


# ===================================================================
# Standalone CCD functions
# ===================================================================

def ccd_mem(
    reference_image: np.ndarray,
    match_image: np.ndarray,
    corr_window_size: int = 7
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Basic coherent change detection between two registered complex images.

    Ported from MATLAB ccdmem.m.

    Parameters
    ----------
    reference_image : np.ndarray
        First complex image, shape (rows, cols).
    match_image : np.ndarray
        Second complex image, shape (rows, cols).
    corr_window_size : int or tuple of int
        Correlation window size. Scalar or (rows, cols).

    Returns
    -------
    ccd_out : np.ndarray
        Coherence magnitude in [0, 1], dtype float64.
    phase_out : np.ndarray
        Phase in radians [-pi, pi], dtype float64.
    """
    win = _ensure_window_tuple(corr_window_size)
    window = np.ones(win, dtype=np.float64)

    conjf_times_g = signal.convolve2d(
        np.conj(reference_image) * match_image,
        window, mode='same', boundary='symm'
    )
    f_squared = signal.convolve2d(
        np.abs(reference_image) ** 2,
        window, mode='same', boundary='symm'
    )
    g_squared = signal.convolve2d(
        np.abs(match_image) ** 2,
        window, mode='same', boundary='symm'
    )

    denom = np.sqrt(f_squared * g_squared)
    ccd_out = np.abs(conjf_times_g) / np.where(denom > 0, denom, 1.0)
    ccd_out[~np.isfinite(ccd_out)] = 0.0
    ccd_out = np.clip(ccd_out, 0.0, 1.0)

    phase_out = np.angle(conjf_times_g)
    phase_out[~np.isfinite(phase_out)] = 0.0

    return ccd_out, phase_out


def ccd_noise_mem(
    reference_image: np.ndarray,
    match_image: np.ndarray,
    corr_window_size: int = 7,
    reference_noise_var: float = 0.0,
    match_noise_var: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Noise-aware coherent change detection (maximum likelihood estimator).

    Implements CCD equation from Wahl, Jakowatz, Yocky, "A New Optimal
    Change Estimator for SAR Coherent Change Detection". When noise
    variances are zero, reverts to traditional CCD.

    Ported from MATLAB ccdnoisemem.m.

    Parameters
    ----------
    reference_image : np.ndarray
        First complex image, shape (rows, cols).
    match_image : np.ndarray
        Second complex image, shape (rows, cols).
    corr_window_size : int or tuple of int
        Correlation window size. Scalar or (rows, cols).
    reference_noise_var : float
        Noise variance of reference image. Default 0 (traditional CCD).
    match_noise_var : float
        Noise variance of match image. Default 0 (traditional CCD).

    Returns
    -------
    ccd_out : np.ndarray
        Coherence magnitude in [0, 1], dtype float64.
    phase_out : np.ndarray
        Phase in radians [-pi, pi], dtype float64.
    """
    if reference_noise_var == 0.0 and match_noise_var == 0.0:
        return ccd_mem(reference_image, match_image, corr_window_size)

    win = _ensure_window_tuple(corr_window_size)
    window = np.ones(win, dtype=np.float64)
    n_pixels = win[0] * win[1]

    conjf_times_g = signal.convolve2d(
        np.conj(reference_image) * match_image,
        window, mode='same', boundary='symm'
    )
    f_squared = signal.convolve2d(
        np.abs(reference_image) ** 2,
        window, mode='same', boundary='symm'
    )
    g_squared = signal.convolve2d(
        np.abs(match_image) ** 2,
        window, mode='same', boundary='symm'
    )

    # Noise-corrected CCD: 2|<f*g>| / (|f|^2 + |g|^2 - N*(sigma_f + sigma_g))
    denom = f_squared + g_squared - n_pixels * (reference_noise_var + match_noise_var)
    ccd_out = 2.0 * np.abs(conjf_times_g) / np.where(denom > 0, denom, 1.0)
    ccd_out[~np.isfinite(ccd_out)] = 0.0
    # Values outside [0,1] are set to 1 (denotes uncertainty)
    ccd_out[(ccd_out < 0) | (ccd_out > 1)] = 1.0

    phase_out = np.angle(conjf_times_g)
    phase_out[~np.isfinite(phase_out)] = 0.0

    return ccd_out, phase_out


def sccm(
    reference_image: np.ndarray,
    match_image: np.ndarray,
    corr_window_size: int = 7,
    reference_noise_var: float = 0.0,
    match_noise_var: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Signal Correlation Change Metric (SCCM).

    Computes SCCM as described in Tom Mitchell's "An Improved CCD Metric"
    (2016 March Tech Forum). SCCM better separates signal correlation
    from noise effects compared to traditional CCD.

    Ported from MATLAB SCCM.m.

    Parameters
    ----------
    reference_image : np.ndarray
        First complex image, shape (rows, cols).
    match_image : np.ndarray
        Second complex image, shape (rows, cols).
    corr_window_size : int or tuple of int
        Correlation window size.
    reference_noise_var : float
        Noise variance of reference image.
    match_noise_var : float
        Noise variance of match image.

    Returns
    -------
    sccm_out : np.ndarray
        SCCM values. Negative values (-100) indicate no-signal pixels.
    angle_out : np.ndarray
        Phase angle from standard CCD.
    """
    win = _ensure_window_tuple(corr_window_size)
    window = np.ones(win, dtype=np.float64)
    n_pixels = win[0] * win[1]

    # Standard CCD for the angle map
    _, angle_out = ccd_mem(reference_image, match_image, corr_window_size)

    # Cross-correlations
    r_star_m = signal.convolve2d(
        np.conj(reference_image) * match_image,
        window, mode='same', boundary='symm'
    )
    m_star_r = signal.convolve2d(
        np.conj(match_image) * reference_image,
        window, mode='same', boundary='symm'
    )

    # Auto-correlations
    r_squared = signal.convolve2d(
        np.conj(reference_image) * reference_image,
        window, mode='same', boundary='symm'
    ).real
    m_squared = signal.convolve2d(
        np.conj(match_image) * match_image,
        window, mode='same', boundary='symm'
    ).real

    # Signal variances (subtract noise)
    sig_r = r_squared / (2 * n_pixels) - reference_noise_var
    sig_m = m_squared / (2 * n_pixels) - match_noise_var

    # SCCM
    denom = 4 * n_pixels * np.sqrt(np.maximum(sig_r * sig_m, 0))
    sccm_out = np.real(r_star_m + m_star_r) / np.where(denom > 0, denom, 1.0)

    # Flag no-signal pixels
    sccm_out[sig_r <= 0] = -100.0
    sccm_out[sig_m <= 0] = -100.0

    return sccm_out, angle_out


def ccd_mem_angle(
    reference_image: np.ndarray,
    match_image: np.ndarray,
    corr_window_size: int = 7,
    angle: float = 0.0,
    rotate_output: bool = True,
    noise_var: float = 0.0,
    metric: str = 'ccd'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    CCD with rotation support for oriented correlation windows.

    Rotates images before computing CCD, then optionally rotates the
    result back to the original orientation.

    Ported from MATLAB ccdmem_angle.m.

    Parameters
    ----------
    reference_image : np.ndarray
        First complex image, shape (rows, cols).
    match_image : np.ndarray
        Second complex image, shape (rows, cols).
    corr_window_size : int or tuple of int
        Correlation window size.
    angle : float
        Rotation angle in degrees.
    rotate_output : bool
        If True, rotate output back to original orientation.
    noise_var : float
        Noise variance (same for both images). Default 0.
    metric : str
        CCD metric: 'ccd' (traditional), 'noise' (noise-aware),
        or 'sccm' (Mitchell's metric).

    Returns
    -------
    ccd_out : np.ndarray
        Coherence/SCCM values.
    phase_out : np.ndarray
        Phase values.
    """
    win = _ensure_window_tuple(corr_window_size)

    # Handle 90/270 degree rotations as simple array ops
    needs_rotate = True
    if angle % 90 == 0:
        needs_rotate = False
        if angle == 90 or angle == 270:
            win = (win[1], win[0])  # Flip window dimensions

    # Rotate if needed (non-axis-aligned angle)
    if needs_rotate:
        ny1, nx1 = reference_image.shape
        ref_rot = ndimage_rotate(reference_image.real, angle, reshape=True) + \
                  1j * ndimage_rotate(reference_image.imag, angle, reshape=True)
        match_rot = ndimage_rotate(match_image.real, angle, reshape=True) + \
                    1j * ndimage_rotate(match_image.imag, angle, reshape=True)
    else:
        ref_rot = reference_image
        match_rot = match_image

    # Compute CCD with selected metric
    if noise_var == 0:
        ccd_out, phase_out = ccd_mem(ref_rot, match_rot, win)
    elif metric.lower() == 'sccm':
        ccd_out, phase_out = sccm(ref_rot, match_rot, win, noise_var, noise_var)
    else:
        ccd_out, phase_out = ccd_noise_mem(ref_rot, match_rot, win, noise_var, noise_var)

    # Rotate output back if requested
    if rotate_output:
        if needs_rotate:
            ccd_rot = ndimage_rotate(ccd_out, -angle, reshape=True)
            phase_rot = ndimage_rotate(phase_out, -angle, reshape=True)
            ny2, nx2 = ccd_rot.shape
            xoff = (nx2 - nx1) // 2
            yoff = (ny2 - ny1) // 2
            ccd_out = ccd_rot[yoff:yoff + ny1, xoff:xoff + nx1]
            phase_out = phase_rot[yoff:yoff + ny1, xoff:xoff + nx1]
        elif angle == 90:
            ccd_out = np.rot90(ccd_out, 1)
            phase_out = np.rot90(phase_out, 1)
        elif angle == 270:
            ccd_out = np.rot90(ccd_out, 3)
            phase_out = np.rot90(phase_out, 3)

    return ccd_out, phase_out


# ===================================================================
# CoherentChangeDetection Processor
# ===================================================================

@processor_version('1.0.0')
@processor_tags(
    modalities=[ImageModality.SAR],
    category=ProcessorCategory.ANALYZE,
    description='Coherent Change Detection between co-registered SAR image pairs'
)
class CoherentChangeDetection(ImageProcessor):
    """
    Coherent Change Detection (CCD) between two co-registered SAR images.

    CCD computes the normalized cross-correlation (coherence) between two complex
    SAR images using a sliding window. The coherence magnitude ranges from 0 to 1,
    where 1 indicates perfect correlation (no change) and 0 indicates complete
    decorrelation (change).

    Parameters
    ----------
    window_size : int
        Size of the square correlation window in pixels. Must be >= 3 and odd.
    """

    __gpu_compatible__ = True

    window_size: Annotated[
        int,
        Range(min=3, max=51),
        Desc('Correlation window size (must be odd)')
    ] = 7

    def __init__(self, window_size: int = 7) -> None:
        if window_size < 3:
            raise ValueError(f"window_size must be >= 3, got {window_size}")
        if window_size % 2 == 0:
            raise ValueError(f"window_size must be odd, got {window_size}")
        self._window_size = window_size

    @property
    def window_size(self) -> int:
        """Correlation window size in pixels."""
        return self._window_size

    def apply(
        self,
        reference_image: np.ndarray,
        match_image: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        Compute coherence magnitude between two complex SAR images.

        Parameters
        ----------
        reference_image : np.ndarray
            First complex image, shape (rows, cols).
        match_image : np.ndarray
            Second complex image, shape (rows, cols).

        Returns
        -------
        np.ndarray
            Coherence magnitude in [0, 1], dtype float32.
        """
        _validate_complex_2d(reference_image, "reference_image")
        _validate_complex_2d(match_image, "match_image")

        if reference_image.shape != match_image.shape:
            raise ValueError(
                f"Images must have same shape. Got reference: {reference_image.shape}, "
                f"match: {match_image.shape}"
            )

        params = self._resolve_params(kwargs)
        window_size = params['window_size']

        ccd_out, _ = ccd_mem(reference_image, match_image, window_size)
        return ccd_out.astype(np.float32)

    def apply_with_phase(
        self,
        reference_image: np.ndarray,
        match_image: np.ndarray,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute both coherence magnitude and phase.

        Returns
        -------
        coherence : np.ndarray
            Coherence magnitude in [0, 1], dtype float32.
        phase : np.ndarray
            Phase in radians [-pi, pi], dtype float32.
        """
        _validate_complex_2d(reference_image, "reference_image")
        _validate_complex_2d(match_image, "match_image")

        if reference_image.shape != match_image.shape:
            raise ValueError(
                f"Images must have same shape. Got reference: {reference_image.shape}, "
                f"match: {match_image.shape}"
            )

        params = self._resolve_params(kwargs)
        window_size = params['window_size']

        ccd_out, phase_out = ccd_mem(reference_image, match_image, window_size)
        return ccd_out.astype(np.float32), phase_out.astype(np.float32)


__all__ = [
    "CoherentChangeDetection",
    "ccd_mem",
    "ccd_noise_mem",
    "sccm",
    "ccd_mem_angle",
]
