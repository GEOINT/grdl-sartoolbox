# -*- coding: utf-8 -*-
"""
Utilities - Constants and helper functions.

Physical constants, WGS-84 parameters, and utility functions for SAR processing.

License
-------
MIT License
Copyright (c) 2024 geoint.org
"""

from grdl_sartoolbox.utils.constants import (
    SPEED_OF_LIGHT,
    TWO_WAY_SPEED_OF_LIGHT,
    FEET_TO_METERS,
    METERS_TO_FEET,
    WGS84_A,
    WGS84_B,
    WGS84_E2,
    wavelength_from_frequency,
    frequency_from_wavelength,
    range_resolution,
    doppler_frequency_to_velocity,
)

from grdl_sartoolbox.utils.misc import (
    fast_running_mean,
    local_sum,
)

__all__ = [
    "SPEED_OF_LIGHT",
    "TWO_WAY_SPEED_OF_LIGHT",
    "FEET_TO_METERS",
    "METERS_TO_FEET",
    "WGS84_A",
    "WGS84_B",
    "WGS84_E2",
    "wavelength_from_frequency",
    "frequency_from_wavelength",
    "range_resolution",
    "doppler_frequency_to_velocity",
    "fast_running_mean",
    "local_sum",
]
