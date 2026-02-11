# -*- coding: utf-8 -*-
"""
Physical Constants - Standard physical and geodetic constants for SAR processing.

Provides commonly used constants including:
- Speed of light
- Unit conversions
- WGS-84 ellipsoid parameters

Attribution
-----------
Ported from MATLAB SAR Toolbox (https://github.com/ngageoint/MATLAB_SAR)

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

# ===================================================================
# Physical Constants
# ===================================================================

#: Speed of light in vacuum (meters per second)
#: Exact value as defined by SI units
SPEED_OF_LIGHT = 299792458.0  # m/s

# ===================================================================
# Unit Conversions
# ===================================================================

#: Feet to meters conversion factor
FEET_TO_METERS = 0.3048  # m/ft

#: Meters to feet conversion factor
METERS_TO_FEET = 1.0 / FEET_TO_METERS  # ft/m

#: Nautical miles to meters conversion factor
NAUTICAL_MILES_TO_METERS = 1852.0  # m/nmi

#: Meters to nautical miles conversion factor
METERS_TO_NAUTICAL_MILES = 1.0 / NAUTICAL_MILES_TO_METERS  # nmi/m

#: Degrees to radians conversion factor
DEG_TO_RAD = 0.017453292519943295  # π/180

#: Radians to degrees conversion factor
RAD_TO_DEG = 57.29577951308232  # 180/π

# ===================================================================
# WGS-84 Ellipsoid Parameters
# ===================================================================

#: WGS-84 semi-major axis (equatorial radius) in meters
WGS84_A = 6378137.0  # m

#: WGS-84 semi-minor axis (polar radius) in meters
WGS84_B = 6356752.314245  # m

#: WGS-84 flattening (f = (a-b)/a)
WGS84_F = (WGS84_A - WGS84_B) / WGS84_A  # ~1/298.257223563

#: WGS-84 first eccentricity squared (e² = (a²-b²)/a²)
WGS84_E2 = (WGS84_A**2 - WGS84_B**2) / WGS84_A**2  # ~0.00669437999014

#: WGS-84 second eccentricity squared (e'² = (a²-b²)/b²)
WGS84_EP2 = (WGS84_A**2 - WGS84_B**2) / WGS84_B**2  # ~0.00673949674228

# ===================================================================
# SAR-Specific Constants
# ===================================================================

#: Default oversampling ratio for SAR image formation
DEFAULT_OVERSAMPLING_RATIO = 1.5

#: Speed of light divided by 2 (for two-way propagation in radar)
#: Commonly used in SAR range calculations
TWO_WAY_SPEED_OF_LIGHT = SPEED_OF_LIGHT / 2.0  # m/s

# ===================================================================
# Mathematical Constants
# ===================================================================

#: Pi (for convenience, also available as math.pi or np.pi)
PI = 3.141592653589793

#: Two times Pi (2π)
TWO_PI = 2.0 * PI

# ===================================================================
# Helper Functions
# ===================================================================

def wavelength_from_frequency(frequency_hz: float) -> float:
    """
    Compute wavelength from frequency.

    Parameters
    ----------
    frequency_hz : float
        Electromagnetic frequency in Hertz.

    Returns
    -------
    float
        Wavelength in meters.

    Examples
    --------
    >>> # X-band SAR at 10 GHz
    >>> wavelength_from_frequency(10e9)
    0.0299792458
    """
    return SPEED_OF_LIGHT / frequency_hz


def frequency_from_wavelength(wavelength_m: float) -> float:
    """
    Compute frequency from wavelength.

    Parameters
    ----------
    wavelength_m : float
        Wavelength in meters.

    Returns
    -------
    float
        Frequency in Hertz.

    Examples
    --------
    >>> # 3cm wavelength (X-band)
    >>> frequency_from_wavelength(0.03)
    9993081933.333334
    """
    return SPEED_OF_LIGHT / wavelength_m


def range_resolution(bandwidth_hz: float) -> float:
    """
    Compute range resolution from signal bandwidth.

    The theoretical range resolution is c/(2*B) where c is the speed
    of light and B is the signal bandwidth.

    Parameters
    ----------
    bandwidth_hz : float
        Signal bandwidth in Hertz.

    Returns
    -------
    float
        Range resolution in meters.

    Examples
    --------
    >>> # 600 MHz bandwidth
    >>> range_resolution(600e6)
    0.24982704833333333
    """
    return TWO_WAY_SPEED_OF_LIGHT / bandwidth_hz


def doppler_frequency_to_velocity(doppler_hz: float, wavelength_m: float) -> float:
    """
    Convert Doppler frequency shift to radial velocity.

    Parameters
    ----------
    doppler_hz : float
        Doppler frequency shift in Hertz.
    wavelength_m : float
        Radar wavelength in meters.

    Returns
    -------
    float
        Radial velocity in meters per second (positive = approaching).

    Examples
    --------
    >>> # 100 Hz Doppler shift at X-band (3cm wavelength)
    >>> doppler_frequency_to_velocity(100.0, 0.03)
    1.5
    """
    return doppler_hz * wavelength_m / 2.0


# ===================================================================
# Constants Dictionary (for programmatic access)
# ===================================================================

CONSTANTS = {
    'SPEED_OF_LIGHT': SPEED_OF_LIGHT,
    'FEET_TO_METERS': FEET_TO_METERS,
    'METERS_TO_FEET': METERS_TO_FEET,
    'NAUTICAL_MILES_TO_METERS': NAUTICAL_MILES_TO_METERS,
    'METERS_TO_NAUTICAL_MILES': METERS_TO_NAUTICAL_MILES,
    'DEG_TO_RAD': DEG_TO_RAD,
    'RAD_TO_DEG': RAD_TO_DEG,
    'WGS84_A': WGS84_A,
    'WGS84_B': WGS84_B,
    'WGS84_F': WGS84_F,
    'WGS84_E2': WGS84_E2,
    'WGS84_EP2': WGS84_EP2,
    'DEFAULT_OVERSAMPLING_RATIO': DEFAULT_OVERSAMPLING_RATIO,
    'TWO_WAY_SPEED_OF_LIGHT': TWO_WAY_SPEED_OF_LIGHT,
    'PI': PI,
    'TWO_PI': TWO_PI,
}

__all__ = [
    # Physical constants
    'SPEED_OF_LIGHT',
    'TWO_WAY_SPEED_OF_LIGHT',
    # Unit conversions
    'FEET_TO_METERS',
    'METERS_TO_FEET',
    'NAUTICAL_MILES_TO_METERS',
    'METERS_TO_NAUTICAL_MILES',
    'DEG_TO_RAD',
    'RAD_TO_DEG',
    # WGS-84 parameters
    'WGS84_A',
    'WGS84_B',
    'WGS84_F',
    'WGS84_E2',
    'WGS84_EP2',
    # SAR-specific
    'DEFAULT_OVERSAMPLING_RATIO',
    # Mathematical
    'PI',
    'TWO_PI',
    # Helper functions
    'wavelength_from_frequency',
    'frequency_from_wavelength',
    'range_resolution',
    'doppler_frequency_to_velocity',
    # Dictionary
    'CONSTANTS',
]
