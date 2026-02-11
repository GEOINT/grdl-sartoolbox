# -*- coding: utf-8 -*-
"""
Coherent Change Detection - Detect changes between co-registered SAR image pairs.

Implements the Coherent Change Detection (CCD) algorithm as described in
Jakowatz, et al., "Spotlight-mode Synthetic Aperture Radar: A Signal Processing
Approach" (equation 5.102).

CCD computes the normalized cross-correlation (coherence) between two complex
SAR images over a sliding window. High coherence indicates stable regions, while
low coherence indicates change.

Attribution
-----------
Ported from MATLAB SAR Toolbox (https://github.com/ngageoint/MATLAB_SAR)
Original MATLAB implementation by Wade Schwartzkopf, NGA/IDT

Dependencies
------------
numpy - Array operations and complex number handling
scipy.signal - 2D convolution for windowed correlation

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
from typing import Annotated, Tuple, Optional

# Third-party
import numpy as np
from scipy import signal

# GRDL internal
from grdl.image_processing.base import ImageProcessor
from grdl.image_processing.params import Desc, Range
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ImageModality, ProcessorCategory


# ===================================================================
# Helpers
# ===================================================================

def _validate_complex_2d(image: np.ndarray, param_name: str = "image") -> None:
    """
    Validate that *image* is a 2D complex numpy array.

    Parameters
    ----------
    image : np.ndarray
        Array to validate.
    param_name : str
        Name of parameter for error messages.

    Raises
    ------
    TypeError
        If *image* is not a numpy array or not complex-valued.
    ValueError
        If *image* is not 2D.
    """
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


# ===================================================================
# CoherentChangeDetection
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

    The algorithm implements the CCD equation from Jakowatz et al.:

        coherence(i,j) = |sum(conj(f) * g)| / sqrt(sum(|f|^2) * sum(|g|^2))

    where f is the reference image, g is the match image, and the sums are computed
    over a local window of size window_size × window_size centered at pixel (i,j).

    Parameters
    ----------
    window_size : int
        Size of the square correlation window in pixels. Must be >= 3 and odd.
        Larger windows provide more stable coherence estimates but reduce spatial
        resolution. Default is 7.

    Raises
    ------
    ValueError
        If *window_size* is < 3 or even.

    Examples
    --------
    >>> from grdl.IO.sar import SICDReader
    >>> from grdl_sartoolbox.processing.ccd import CoherentChangeDetection
    >>>
    >>> # Load two co-registered SAR images
    >>> with SICDReader('image1.nitf') as reader:
    ...     image1 = reader.read()  # complex64 array
    >>> with SICDReader('image2.nitf') as reader:
    ...     image2 = reader.read()  # complex64 array
    >>>
    >>> # Compute coherence
    >>> ccd = CoherentChangeDetection(window_size=11)
    >>> coherence = ccd.apply(image1, image2)  # float32 in [0, 1]
    >>>
    >>> # Optionally get both coherence and phase
    >>> coherence, phase = ccd.apply_with_phase(image1, image2)

    Notes
    -----
    - Input images must be co-registered (aligned) for meaningful results
    - The coherence is computed as magnitude; phase is optionally available
    - Edge pixels within window_size//2 of borders use 'same' mode padding
    - Division by zero is handled by setting coherence to 0 at those pixels
    """

    __gpu_compatible__ = True  # Pure numpy/scipy operations

    # -- Annotated scalar field for GUI introspection (__param_specs__) --
    window_size: Annotated[
        int,
        Range(min=3, max=51),
        Desc('Correlation window size (must be odd)')
    ] = 7

    def __init__(self, window_size: int = 7) -> None:
        """
        Initialize CoherentChangeDetection processor.

        Parameters
        ----------
        window_size : int
            Size of the square correlation window. Must be >= 3 and odd.
            Default is 7.

        Raises
        ------
        ValueError
            If window_size is < 3 or even.
        """
        if window_size < 3:
            raise ValueError(
                f"window_size must be >= 3, got {window_size}"
            )
        if window_size % 2 == 0:
            raise ValueError(
                f"window_size must be odd for symmetric window, got {window_size}"
            )

        self._window_size = window_size

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def window_size(self) -> int:
        """Correlation window size in pixels."""
        return self._window_size

    # ------------------------------------------------------------------
    # Core algorithm
    # ------------------------------------------------------------------

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
            First complex image, shape (rows, cols). Complex64 or complex128.
        match_image : np.ndarray
            Second complex image, shape (rows, cols). Must match reference shape.
        **kwargs
            Optional parameter overrides (e.g., window_size=11).

        Returns
        -------
        np.ndarray
            Coherence magnitude, shape (rows, cols), dtype float32.
            Values in [0, 1] where 1 = perfect correlation, 0 = no correlation.

        Raises
        ------
        TypeError
            If images are not complex numpy arrays.
        ValueError
            If images are not 2D or have different shapes.

        Examples
        --------
        >>> image1 = np.random.randn(512, 512) + 1j*np.random.randn(512, 512)
        >>> image2 = image1.copy()  # Identical images
        >>> ccd = CoherentChangeDetection(window_size=7)
        >>> coherence = ccd.apply(image1, image2)
        >>> np.mean(coherence)  # Should be ~1.0 for identical images
        0.999...
        """
        # Validate inputs
        _validate_complex_2d(reference_image, "reference_image")
        _validate_complex_2d(match_image, "match_image")

        if reference_image.shape != match_image.shape:
            raise ValueError(
                f"Images must have same shape. Got reference: {reference_image.shape}, "
                f"match: {match_image.shape}"
            )

        # Resolve parameters (allows runtime override via kwargs)
        params = self._resolve_params(kwargs)
        window_size = params['window_size']

        # Create uniform window (ones matrix)
        window = np.ones((window_size, window_size), dtype=np.float64)

        # Port of MATLAB ccdmem.m algorithm:
        # conjf_times_g = conv2(conj(reference_image).*match_image, ones(window_size), 'same')
        # f_squared = conv2(abs(reference_image).^2, ones(window_size), 'same')
        # g_squared = conv2(abs(match_image).^2, ones(window_size), 'same')
        # ccd_out = abs(conjf_times_g)./sqrt(f_squared.*g_squared)
        # ccd_out(~isfinite(ccd_out))=0

        # Compute windowed cross-correlation: sum(conj(f) * g)
        conjf_times_g = signal.convolve2d(
            np.conj(reference_image) * match_image,
            window,
            mode='same',
            boundary='symm'
        )

        # Compute windowed power: sum(|f|^2) and sum(|g|^2)
        f_squared = signal.convolve2d(
            np.abs(reference_image) ** 2,
            window,
            mode='same',
            boundary='symm'
        )

        g_squared = signal.convolve2d(
            np.abs(match_image) ** 2,
            window,
            mode='same',
            boundary='symm'
        )

        # Compute coherence: |cross-correlation| / sqrt(power_f * power_g)
        # Add small epsilon to denominator to avoid division by zero
        epsilon = np.finfo(np.float64).tiny
        denominator = np.sqrt(f_squared * g_squared + epsilon)
        coherence = np.abs(conjf_times_g) / denominator

        # Handle any remaining non-finite values (NaN, Inf)
        coherence = np.where(np.isfinite(coherence), coherence, 0.0)

        # Clamp to [0, 1] to handle floating-point precision issues
        coherence = np.clip(coherence, 0.0, 1.0)

        # Return as float32 (typical for image data)
        return coherence.astype(np.float32)

    def apply_with_phase(
        self,
        reference_image: np.ndarray,
        match_image: np.ndarray,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute both coherence magnitude and phase between two SAR images.

        The phase represents the interferometric phase difference between
        the two images, useful for interferometric SAR (InSAR) applications.

        Parameters
        ----------
        reference_image : np.ndarray
            First complex image, shape (rows, cols).
        match_image : np.ndarray
            Second complex image, shape (rows, cols).
        **kwargs
            Optional parameter overrides (e.g., window_size=11).

        Returns
        -------
        coherence : np.ndarray
            Coherence magnitude in [0, 1], dtype float32.
        phase : np.ndarray
            Phase in radians, range [-π, π], dtype float32.

        Examples
        --------
        >>> ccd = CoherentChangeDetection(window_size=7)
        >>> coherence, phase = ccd.apply_with_phase(image1, image2)
        >>> print(f"Mean coherence: {np.mean(coherence):.3f}")
        >>> print(f"Phase range: [{np.min(phase):.2f}, {np.max(phase):.2f}]")
        """
        # Validate inputs
        _validate_complex_2d(reference_image, "reference_image")
        _validate_complex_2d(match_image, "match_image")

        if reference_image.shape != match_image.shape:
            raise ValueError(
                f"Images must have same shape. Got reference: {reference_image.shape}, "
                f"match: {match_image.shape}"
            )

        # Resolve parameters
        params = self._resolve_params(kwargs)
        window_size = params['window_size']

        # Create uniform window
        window = np.ones((window_size, window_size), dtype=np.float64)

        # Compute windowed cross-correlation
        conjf_times_g = signal.convolve2d(
            np.conj(reference_image) * match_image,
            window,
            mode='same',
            boundary='symm'
        )

        # Compute windowed power
        f_squared = signal.convolve2d(
            np.abs(reference_image) ** 2,
            window,
            mode='same',
            boundary='symm'
        )

        g_squared = signal.convolve2d(
            np.abs(match_image) ** 2,
            window,
            mode='same',
            boundary='symm'
        )

        # Coherence magnitude
        epsilon = np.finfo(np.float64).tiny
        denominator = np.sqrt(f_squared * g_squared + epsilon)
        coherence = np.abs(conjf_times_g) / denominator
        coherence = np.where(np.isfinite(coherence), coherence, 0.0)

        # Clamp to [0, 1] to handle floating-point precision issues
        coherence = np.clip(coherence, 0.0, 1.0)

        # Phase (angle of complex cross-correlation)
        phase = np.angle(conjf_times_g)
        phase = np.where(np.isfinite(phase), phase, 0.0)

        return coherence.astype(np.float32), phase.astype(np.float32)
