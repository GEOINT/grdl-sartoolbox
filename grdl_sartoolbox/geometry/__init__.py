# -*- coding: utf-8 -*-
"""
SAR Geometry - Geometry analysis and projection utilities.

Supplements GRDL's geolocation capabilities with SAR-specific geometry
analysis functions.

License
-------
MIT License
Copyright (c) 2024 geoint.org

Created
-------
2026-02-11
"""

# Import geometry analysis functions
from grdl_sartoolbox.geometry.analysis import (
    SARGeometry,
    wgs84_normal,
    compute_sar_geometry,
)

__all__ = [
    "SARGeometry",
    "wgs84_normal",
    "compute_sar_geometry",
]
