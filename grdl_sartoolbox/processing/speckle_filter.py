# -*- coding: utf-8 -*-
"""
Speckle Filtering - Adaptive filters for SAR speckle noise reduction.

Implements Lee, Kuan, and Boxcar speckle filters for SAR imagery.
These filters reduce multiplicative speckle noise while preserving
edges and features.

Attribution
-----------
Ported from MATLAB SAR Toolbox (https://github.com/ngageoint/MATLAB_SAR)
Original MATLAB implementation by Tom Braun, NGA/IBR

Dependencies
------------
numpy - Array operations
scipy.ndimage - Uniform filtering for windowed statistics

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
from typing import Annotated, Optional, Literal
from enum import Enum

# Third-party
import numpy as np
from scipy import ndimage

# GRDL internal
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.params import Desc, Range, Options
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ImageModality, ProcessorCategory


# ===================================================================
# Filter Type Enum
# ===================================================================

class SpeckleFilterType(str, Enum):
    """Speckle filter types."""
    LEE = "Lee"
    KUAN = "Kuan"
    BOXCAR = "Boxcar"


# ===================================================================
# Helpers
# ===================================================================

def _validate_sar_image(image: np.ndarray, param_name: str = "image") -> None:
    """
    Validate SAR image input.

    Parameters
    ----------
    image : np.ndarray
        Array to validate (can be real or complex).
    param_name : str
        Name for error messages.

    Raises
    ------
    TypeError
        If not a numpy array.
    ValueError
        If not 2D.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"{param_name} must be a numpy ndarray, got {type(image).__name__}"
        )
    if image.ndim != 2:
        raise ValueError(
            f"{param_name} must be 2D (rows, cols), got {image.ndim}D "
            f"with shape {image.shape}"
        )


def _uniform_filter_2d(data: np.ndarray, radius: int) -> np.ndarray:
    """
    Apply uniform (boxcar) filter using scipy.

    Parameters
    ----------
    data : np.ndarray
        2D input array.
    radius : int
        Filter radius (window is 2*radius+1).

    Returns
    -------
    np.ndarray
        Filtered array.
    """
    size = 2 * radius + 1
    return ndimage.uniform_filter(data, size=size, mode='reflect')


def _estimate_enl(variance_coeff: np.ndarray, radius: int = 2) -> float:
    """
    Estimate Equivalent Number of Looks (ENL).

    Assumes ~10% of the scene is homogeneous and estimates ENL
    from the variance coefficient in those regions.

    Parameters
    ----------
    variance_coeff : np.ndarray
        Coefficient of variation squared (Ci²).
    radius : int
        Edge pixels to skip.

    Returns
    -------
    float
        Estimated ENL.
    """
    # Skip edge pixels
    rows, cols = variance_coeff.shape
    if rows <= 2 * radius or cols <= 2 * radius:
        # Image too small, use all pixels
        interior = variance_coeff
    else:
        interior = variance_coeff[radius:-radius, radius:-radius]

    # Sort and take lowest 10% (assumed homogeneous)
    sorted_values = np.sort(interior.ravel())
    n_samples = max(1, int(len(sorted_values) * 0.1))
    homogeneous_samples = sorted_values[:n_samples]

    # ENL = 1 / mean(Ci²) for homogeneous regions
    mean_ci2 = np.mean(homogeneous_samples)
    if mean_ci2 > 0:
        return 1.0 / mean_ci2
    else:
        return 1.0  # Default to single look


# ===================================================================
# LeeFilter
# ===================================================================

@processor_version('1.0.0')
@processor_tags(
    modalities=[ImageModality.SAR],
    category=ProcessorCategory.FILTERS,
    description='Lee adaptive speckle filter for SAR imagery'
)
class LeeFilter(ImageTransform):
    """
    Lee adaptive speckle filter for SAR imagery.

    The Lee filter is an adaptive filter that reduces speckle noise
    while preserving edges. It estimates local statistics (mean and
    variance) within a sliding window and computes a weighted average
    based on the local coefficient of variation.

    The filter operates on amplitude (magnitude) data. If complex input
    is provided, the magnitude is computed automatically.

    Filter equation:
        output = mean + W * (input - mean)
        where W = Ct² / Ci²
        Ct² = (Ci² - Cn²)  [Lee filter]
        Ci² = variance / mean²
        Cn² = 1 / ENL

    Parameters
    ----------
    radius : int
        Filter window radius in pixels. Window size is (2*radius+1)².
        Larger windows provide more smoothing but reduce resolution.
        Default is 2 (5×5 window).
    enl : float
        Equivalent Number of Looks. If 0, automatically estimated from
        the image assuming ~10% homogeneous regions. Default is 0 (auto).
    filter_type : str
        Type of filter: "Lee", "Kuan", or "Boxcar". Default is "Lee".

    Examples
    --------
    >>> from grdl.IO.sar import SICDReader
    >>> from grdl_sartoolbox.processing import LeeFilter
    >>>
    >>> with SICDReader('image.nitf') as reader:
    ...     image = reader.read()  # complex64
    >>>
    >>> # Apply Lee filter
    >>> lee = LeeFilter(radius=3, enl=0)  # Auto-estimate ENL
    >>> filtered = lee.apply(image)  # Returns amplitude (real)
    >>>
    >>> # Or with explicit ENL
    >>> lee = LeeFilter(radius=2, enl=4.0)
    >>> filtered = lee.apply(image)

    Notes
    -----
    - Input can be real (amplitude) or complex (magnitude computed)
    - Output is always real-valued amplitude
    - Bias correction is applied to preserve mean intensity
    - Edge pixels use reflection boundary conditions
    """

    __gpu_compatible__ = False  # uses scipy.ndimage (no CuPy support)

    # Annotated parameters
    radius: Annotated[
        int,
        Range(min=1, max=10),
        Desc('Filter window radius (window is 2*radius+1)')
    ] = 2

    enl: Annotated[
        float,
        Range(min=0.0, max=100.0),
        Desc('Equivalent Number of Looks (0=auto-estimate)')
    ] = 0.0

    filter_type: Annotated[
        str,
        Options('Lee', 'Kuan', 'Boxcar'),
        Desc('Filter type')
    ] = 'Lee'

    def __init__(
        self,
        radius: int = 2,
        enl: float = 0.0,
        filter_type: str = 'Lee'
    ) -> None:
        """
        Initialize Lee speckle filter.

        Parameters
        ----------
        radius : int
            Filter radius (default 2 for 5×5 window).
        enl : float
            Equivalent number of looks (0 = auto-estimate).
        filter_type : str
            Filter type: "Lee", "Kuan", or "Boxcar".
        """
        if radius < 1:
            raise ValueError(f"radius must be >= 1, got {radius}")
        if enl < 0:
            raise ValueError(f"enl must be >= 0, got {enl}")
        if filter_type not in ('Lee', 'Kuan', 'Boxcar'):
            raise ValueError(
                f"filter_type must be 'Lee', 'Kuan', or 'Boxcar', got {filter_type!r}"
            )

        self._radius = radius
        self._enl = enl
        self._filter_type = filter_type

    @property
    def radius(self) -> int:
        """Filter window radius."""
        return self._radius

    @property
    def enl(self) -> float:
        """Equivalent number of looks."""
        return self._enl

    @property
    def filter_type(self) -> str:
        """Filter type."""
        return self._filter_type

    def apply(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply speckle filter to SAR image.

        Parameters
        ----------
        image : np.ndarray
            Input SAR image (rows, cols). Can be real (amplitude) or
            complex (magnitude computed automatically).
        **kwargs
            Optional parameter overrides (radius, enl, filter_type).

        Returns
        -------
        np.ndarray
            Filtered amplitude image, dtype float32.

        Raises
        ------
        TypeError
            If image is not a numpy array.
        ValueError
            If image is not 2D.

        Examples
        --------
        >>> image = np.random.randn(512, 512) + 1j*np.random.randn(512, 512)
        >>> lee = LeeFilter(radius=2, enl=4)
        >>> filtered = lee.apply(image)
        >>> filtered.shape
        (512, 512)
        """
        # Validate input
        _validate_sar_image(image, "image")

        # Resolve parameters
        params = self._resolve_params(kwargs)
        radius = params['radius']
        enl = params['enl']
        filter_type = params['filter_type']

        # Convert to amplitude (power)
        # MATLAB: I = double(abs(I)).^2
        amplitude = np.abs(image).astype(np.float64)
        power = amplitude ** 2

        # For boxcar filter, just apply uniform filter
        if filter_type == 'Boxcar':
            filtered_power = _uniform_filter_2d(power, radius)
        else:
            # Compute local statistics using uniform filter
            # Mean: m = sum(I) / N
            mean_power = _uniform_filter_2d(power, radius)

            # Variance: v = sum((I - m)²) / N
            variance = _uniform_filter_2d((power - mean_power) ** 2, radius)

            # Coefficient of variation squared: Ci² = variance / mean²
            # Avoid division by zero
            epsilon = np.finfo(np.float64).tiny
            ci2 = variance / (mean_power ** 2 + epsilon)

            # Estimate or use provided ENL
            if enl <= 0:
                enl_estimated = _estimate_enl(ci2, radius)
            else:
                enl_estimated = enl

            # Noise variance coefficient: Cn² = 1 / ENL
            cn2 = 1.0 / enl_estimated

            # Compute texture variance coefficient: Ct²
            if filter_type == 'Lee':
                # Lee: Ct² = Ci² - Cn²
                ct2 = ci2 - cn2
            else:  # Kuan
                # Kuan: Ct² = (Ci² - Cn²) / (1 + Cn²)
                ct2 = (ci2 - cn2) / (1.0 + cn2)

            # Clamp Ct² to small positive value to avoid negative weights
            ct2 = np.maximum(ct2, 1e-20)

            # Compute weight: W = min(1, Ct² / Ci²)
            weight = np.minimum(1.0, ct2 / (ci2 + epsilon))

            # Apply filter: output = mean + W * (input - mean)
            filtered_power = mean_power + weight * (power - mean_power)

        # Bias correction: scale to preserve mean intensity
        # MATLAB: fil = fil .* (mean(mean(I)) ./ mean(mean(fil)))
        original_mean = np.mean(power)
        filtered_mean = np.mean(filtered_power)
        if filtered_mean > 0:
            filtered_power *= original_mean / filtered_mean

        # Convert back to amplitude: sqrt(power)
        # MATLAB: fil = sqrt(fil)
        filtered_amplitude = np.sqrt(np.maximum(filtered_power, 0.0))

        return filtered_amplitude.astype(np.float32)


# Export for convenience
__all__ = ["LeeFilter", "SpeckleFilterType"]
