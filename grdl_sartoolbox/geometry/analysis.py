# -*- coding: utf-8 -*-
"""
SAR Geometry Analysis - Compute collection geometry from sensor vectors.

Provides utilities for computing SAR-specific geometry parameters such as
azimuth angle, grazing angle, squint angle, layover angle, and Doppler
cone angle from sensor position and velocity vectors.

Attribution
-----------
Ported from MATLAB SAR Toolbox (https://github.com/ngageoint/MATLAB_SAR)
Original implementation by Wade Schwartzkopf (NGA/IDT)

Dependencies
------------
numpy - Vector operations
dataclasses - Geometry result structure

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
from typing import Optional, Tuple
from dataclasses import dataclass

# Third-party
import numpy as np

# GRDL internal
from grdl_sartoolbox.utils.constants import WGS84_A, WGS84_B


# ===================================================================
# Data Structures
# ===================================================================

@dataclass
class SARGeometry:
    """
    SAR collection geometry parameters.

    All angles are in radians unless otherwise specified.

    Attributes
    ----------
    azimuth : float
        Sensor azimuth angle (0 to 2π radians, measured clockwise from north).
    graze : float
        Grazing angle (angle between range vector and ground plane).
    slope : float
        Slope angle (angle between slant plane normal and ground plane normal).
    squint : float
        Squint angle (angle between velocity and range vectors in ground plane).
    layover : float
        Layover angle (rotation of slant plane from vertical).
    multipath : float
        Multipath angle (angle of ground bounce reflection).
    dca : float
        Doppler Cone Angle (angle between velocity and range vectors).
    tilt : float
        Slant plane tilt angle.
    track : float
        Ground track angle (direction of ground track in tangent plane).
    felev : float
        Flight elevation angle (angle of velocity above ground plane).
    right : int
        Flight direction: -1 for left-looking, +1 for right-looking, 0 for unknown.
    ascend : int
        Flight direction: +1 for ascending, -1 for descending, 0 for level.
    """
    azimuth: float
    graze: float
    slope: float
    squint: float
    layover: float
    multipath: float
    dca: float
    tilt: float
    track: float
    felev: float
    right: int
    ascend: int


# ===================================================================
# Helper Functions
# ===================================================================

def _project_onto_plane(vector: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """
    Project vector onto plane defined by normal.

    Parameters
    ----------
    vector : np.ndarray
        Vector to project, shape (3,) or (N, 3).
    normal : np.ndarray
        Plane normal vector, shape (3,).

    Returns
    -------
    np.ndarray
        Projected vector, same shape as input vector.
    """
    normal_unit = normal / np.linalg.norm(normal)
    if vector.ndim == 1:
        return vector - np.dot(vector, normal_unit) * normal_unit
    else:
        # Batch projection for (N, 3) arrays
        dots = np.dot(vector, normal_unit)  # (N,)
        return vector - dots[:, np.newaxis] * normal_unit


# ===================================================================
# WGS-84 Normal
# ===================================================================

def wgs84_normal(
    ecef: np.ndarray
) -> np.ndarray:
    """
    Compute normal vector to WGS-84 ellipsoid at given ECEF point.

    The normal is perpendicular to the tangent plane of the WGS-84
    ellipsoid at the specified point. This is used for ground plane
    calculations in SAR geometry.

    Parameters
    ----------
    ecef : np.ndarray
        ECEF coordinates (meters). Can be:
        - 1D array of shape (3,): single point [X, Y, Z]
        - 2D array of shape (3, N): N points as columns [[X...], [Y...], [Z...]]
        - 2D array of shape (N, 3): N points as rows [[X, Y, Z], ...]

    Returns
    -------
    np.ndarray
        Unit normal vector(s) to WGS-84 ellipsoid, same shape as input.

    Examples
    --------
    >>> # Single point
    >>> ecef = np.array([4000e3, 3000e3, 4000e3])  # Example ECEF point
    >>> normal = wgs84_normal(ecef)
    >>> normal.shape
    (3,)
    >>> np.isclose(np.linalg.norm(normal), 1.0)
    True

    >>> # Multiple points as columns
    >>> ecef = np.array([[4000e3, 5000e3], [3000e3, 2000e3], [4000e3, 3500e3]])
    >>> normals = wgs84_normal(ecef)
    >>> normals.shape
    (3, 2)

    >>> # Multiple points as rows
    >>> ecef = np.array([[4000e3, 3000e3, 4000e3], [5000e3, 2000e3, 3500e3]])
    >>> normals = wgs84_normal(ecef)
    >>> normals.shape
    (2, 3)

    Notes
    -----
    - For an ellipsoid with semi-axes a (equatorial) and b (polar), the
      normal at point (x, y, z) is proportional to (x/a², y/a², z/b²)
    - The returned vector is normalized to unit length
    - WGS-84 parameters: a=6378137 m, b=6356752.314245179 m

    References
    ----------
    MATLAB SAR Toolbox: Geometry/wgs_84_norm.m
    """
    ecef = np.asarray(ecef, dtype=np.float64)

    # Determine input format
    # For ambiguous (3, 3) arrays, row format (N=3 points, 3 coords each) is
    # assumed, consistent with the (N, 3) convention used throughout this
    # codebase (TxPos, RcvPos, SRPPos, etc.).  To use column format with a
    # (3, 3) array, transpose before calling.
    if ecef.ndim == 1:
        # Single point: (3,)
        x, y, z = ecef
        single_point = True
        row_format = False
    elif ecef.ndim == 2 and ecef.shape[1] == 3:
        # Row format: (N, 3) — also covers the ambiguous (3, 3) case
        x, y, z = ecef[:, 0], ecef[:, 1], ecef[:, 2]
        single_point = False
        row_format = True
    elif ecef.ndim == 2 and ecef.shape[0] == 3:
        # Column format: (3, N) where N != 3
        x, y, z = ecef
        single_point = False
        row_format = False
    else:
        raise ValueError(
            f"Invalid ECEF shape {ecef.shape}. Expected (3,), (3, N), or (N, 3)"
        )

    # Compute gradient of ellipsoid equation: x²/a² + y²/a² + z²/b² = 1
    # Normal is proportional to gradient: (x/a², y/a², z/b²)
    a_sq = WGS84_A ** 2
    b_sq = WGS84_B ** 2

    nx = x / a_sq
    ny = y / a_sq
    nz = z / b_sq

    # Normalize to unit vector
    mag = np.sqrt(nx**2 + ny**2 + nz**2)
    nx = nx / mag
    ny = ny / mag
    nz = nz / mag

    # Return in same format as input
    if single_point:
        return np.array([nx, ny, nz])
    elif row_format:
        return np.column_stack([nx, ny, nz])
    else:
        # Column format: (3, N)
        return np.array([nx, ny, nz])


# ===================================================================
# SAR Geometry Computation
# ===================================================================

def compute_sar_geometry(
    aim: np.ndarray,
    position: np.ndarray,
    velocity: np.ndarray,
    normal: Optional[np.ndarray] = None
) -> SARGeometry:
    """
    Compute SAR collection geometry from pointing vectors.

    Calculates comprehensive SAR geometry parameters including azimuth,
    grazing angle, squint angle, layover angle, and Doppler cone angle
    from the sensor position, velocity, and ground reference point.

    Parameters
    ----------
    aim : np.ndarray
        Ground reference point in ECEF coordinates (meters), shape (3,).
    position : np.ndarray
        Platform position at center of aperture in ECEF (meters), shape (3,).
    velocity : np.ndarray
        Platform velocity in ECEF (meters/second), shape (3,).
    normal : np.ndarray, optional
        Ground plane normal vector. If None, uses WGS-84 tangent plane
        normal at the aim point. Shape (3,).

    Returns
    -------
    SARGeometry
        Comprehensive geometry structure with all computed angles and parameters.

    Examples
    --------
    >>> # Typical SAR geometry
    >>> aim = np.array([4000e3, 3000e3, 3500e3])  # Ground point
    >>> position = np.array([4010e3, 3020e3, 3550e3])  # Platform at 50km altitude
    >>> velocity = np.array([0, 7000, 100])  # Moving ~7km/s northward
    >>>
    >>> geom = compute_sar_geometry(aim, position, velocity)
    >>>
    >>> # Check azimuth (should be ~0 for northward flight)
    >>> np.degrees(geom.azimuth)  # Close to 0° (north)
    >>>
    >>> # Check if right-looking
    >>> geom.right
    -1  # Left-looking (negative)
    >>>
    >>> # Grazing angle
    >>> np.degrees(geom.graze)  # Typical 20-60°

    Notes
    -----
    The function computes:

    - **Azimuth**: Sensor azimuth angle (0-360°) measured clockwise from north
    - **Graze**: Grazing angle between range vector and ground plane
    - **Slope**: Slope angle (≥ graze) between slant plane and ground
    - **Squint**: Angle between velocity and range vectors in ground plane
    - **Layover**: Rotation angle of slant plane from vertical
    - **DCA**: Doppler Cone Angle between velocity and range vectors
    - **Tilt**: Slant plane tilt angle
    - **Track**: Ground track angle in tangent plane at aim point
    - **Felev**: Flight elevation angle above ground plane
    - **Multipath**: Ground bounce reflection angle
    - **Right**: ±1 for left/right-looking collection
    - **Ascend**: ±1 for ascending/descending pass

    Conventions
    -----------
    - All input vectors are in ECEF coordinates
    - All output angles are in radians
    - Azimuth: 0 = north, π/2 = east, π = south, 3π/2 = west
    - Right = -1 for left-looking, +1 for right-looking
    - Ascend = +1 for ascending orbit, -1 for descending

    References
    ----------
    MATLAB SAR Toolbox: Geometry/vect2geom.m
    """
    aim = np.asarray(aim, dtype=np.float64)
    position = np.asarray(position, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)

    if aim.shape != (3,) or position.shape != (3,) or velocity.shape != (3,):
        raise ValueError(
            f"aim, position, and velocity must be 1D arrays of shape (3,). "
            f"Got shapes: aim={aim.shape}, position={position.shape}, velocity={velocity.shape}"
        )

    # Default to WGS-84 tangent plane normal if not provided
    if normal is None:
        normal = wgs84_normal(aim)
    else:
        normal = np.asarray(normal, dtype=np.float64)
        if normal.shape != (3,):
            raise ValueError(f"normal must be shape (3,), got {normal.shape}")

    # Range from aim point to platform
    R = position - aim
    R_mag = np.linalg.norm(R)
    R_unit = R / R_mag

    P_mag = np.linalg.norm(position)
    P_unit = position / P_mag

    # Ground plane unit vectors (tangent plane at aim point)
    KDP_unit = normal / np.linalg.norm(normal)

    # Project range onto ground plane
    JDP = _project_onto_plane(R_unit, KDP_unit)
    JDP_unit = JDP / np.linalg.norm(JDP)

    # Cross product to get third orthogonal direction
    IDP_unit = np.cross(JDP_unit, KDP_unit)

    # === Sensor Azimuth ===
    # Project north direction onto ground plane
    north = np.array([0.0, 0.0, 1.0])
    ProjN = _project_onto_plane(north, KDP_unit)
    ProjN_norm = np.linalg.norm(ProjN)

    # Handle edge case: if ground plane is horizontal (normal is vertical),
    # north projection is zero, azimuth is undefined
    if ProjN_norm < 1e-10:
        # Default to 0 azimuth when undefined
        azimuth = 0.0
    else:
        ProjN_unit = ProjN / ProjN_norm

        # Azimuth angle using atan2 for full 360° range
        azimuth = np.arctan2(
            np.dot(np.cross(JDP_unit, ProjN_unit), KDP_unit),
            np.dot(ProjN_unit, JDP_unit)
        )
        if azimuth < 0:
            azimuth += 2 * np.pi

    # === Trajectory and Flight Direction ===
    V_mag = np.linalg.norm(velocity)
    TRAJ_unit = velocity / V_mag

    # Slant plane normal (cross product of range and trajectory)
    SLANT = np.cross(R_unit, TRAJ_unit)
    SLANT_unit = SLANT / np.linalg.norm(SLANT)

    # Determine sense (left/right looking)
    sense = np.sign(np.dot(SLANT_unit, KDP_unit))
    SLANT_unit = sense * SLANT_unit

    # === Geometry Angles ===

    # Slope angle (>= graze)
    # Clamp dot products to [-1, 1] to avoid NaN from floating-point overshoot
    slope = np.arccos(np.clip(np.dot(SLANT_unit, KDP_unit), -1.0, 1.0))

    # Grazing angle
    graze = np.arcsin(np.clip(np.dot(R_unit, KDP_unit), -1.0, 1.0))

    # Squint angle
    Vproj = _project_onto_plane(velocity, position)
    Vproj_unit = Vproj / np.linalg.norm(Vproj)
    Rproj = _project_onto_plane(-R, position)
    Rproj_unit = Rproj / np.linalg.norm(Rproj)
    squint = np.arctan2(
        np.dot(np.cross(Vproj_unit, Rproj_unit), P_unit),
        np.dot(Rproj_unit, Vproj_unit)
    )

    # Layover angle
    TMP = np.cross(SLANT_unit, KDP_unit)
    TMP_unit = TMP / np.linalg.norm(TMP)
    layover = np.arcsin(np.clip(np.dot(JDP_unit, TMP_unit), -1.0, 1.0))

    # Doppler cone angle
    dca = -sense * np.arccos(np.clip(np.dot(-R_unit, TRAJ_unit), -1.0, 1.0))

    # Ground track angle in tangent plane
    TRACK = _project_onto_plane(TRAJ_unit, KDP_unit)
    TRACK_unit = TRACK / np.linalg.norm(TRACK)
    track = -sense * np.arccos(np.clip(np.dot(-JDP_unit, TRACK_unit), -1.0, 1.0))

    # Flight elevation
    felev = np.arcsin(np.clip(np.dot(KDP_unit, TRAJ_unit), -1.0, 1.0))

    # Slant plane tilt angle
    tilt = -np.arccos(np.clip(np.cos(slope) / np.cos(graze), -1.0, 1.0)) * np.sign(layover)

    # Multipath angle
    multipath = -np.arctan(np.tan(tilt) * np.sin(graze))

    # === Flight Direction Flags ===

    # Left/right looking
    if sense < 0:
        right = 1  # Right-looking
    elif sense > 0:
        right = -1  # Left-looking
    else:
        right = 0  # Unknown

    # Ascending/descending
    if velocity[2] > 0:
        ascend = 1  # Ascending
    elif velocity[2] < 0:
        ascend = -1  # Descending
    else:
        ascend = 0  # Level

    return SARGeometry(
        azimuth=azimuth,
        graze=graze,
        slope=slope,
        squint=squint,
        layover=layover,
        multipath=multipath,
        dca=dca,
        tilt=tilt,
        track=track,
        felev=felev,
        right=right,
        ascend=ascend
    )


__all__ = [
    "SARGeometry",
    "wgs84_normal",
    "compute_sar_geometry",
]
