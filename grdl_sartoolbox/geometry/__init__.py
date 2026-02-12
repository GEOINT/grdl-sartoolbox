# -*- coding: utf-8 -*-
"""
SAR Geometry - Coordinate transforms, analysis, and projection utilities.

Provides:
- Coordinate transformations (ECF, geodetic, NED, geocentric, RIC)
- SAR geometry analysis (azimuth, grazing angle, etc.)
- Image-to-ground and ground-to-image projections

License
-------
MIT License
Copyright (c) 2024 geoint.org
"""

from grdl_sartoolbox.geometry.analysis import (
    SARGeometry,
    wgs84_normal,
    compute_sar_geometry,
)

from grdl_sartoolbox.geometry.coordinates import (
    ecf_to_geodetic,
    geodetic_to_ecf,
    ecf_ned_rotation_matrix,
    ecf_to_ned,
    ned_to_ecf,
    ecf_to_geocentric,
    ric_ecf_matrix,
)

from grdl_sartoolbox.geometry.projections import (
    point_to_ground_plane,
    point_to_hae,
    point_to_hae_newton,
)

__all__ = [
    # Analysis
    "SARGeometry",
    "wgs84_normal",
    "compute_sar_geometry",
    # Coordinates
    "ecf_to_geodetic",
    "geodetic_to_ecf",
    "ecf_ned_rotation_matrix",
    "ecf_to_ned",
    "ned_to_ecf",
    "ecf_to_geocentric",
    "ric_ecf_matrix",
    # Projections
    "point_to_ground_plane",
    "point_to_hae",
    "point_to_hae_newton",
]
