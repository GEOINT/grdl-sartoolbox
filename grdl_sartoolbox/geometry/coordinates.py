# -*- coding: utf-8 -*-
"""
Coordinate Transformations - ECF, geodetic, NED, geocentric, and RIC frames.

Implements all coordinate transformations from the MATLAB SAR Toolbox
Geometry/coordinates module.

Attribution
-----------
Ported from MATLAB SAR Toolbox (https://github.com/ngageoint/MATLAB_SAR)
Original: Sean Hatch, Wade Schwartzkopf, Rocco Corsetti, NGA

License
-------
MIT License
Copyright (c) 2024 geoint.org
"""

from typing import Tuple, Optional
import numpy as np

from grdl_sartoolbox.utils.constants import WGS84_A, WGS84_B, WGS84_E2


# ===================================================================
# ECF <-> Geodetic
# ===================================================================

def ecf_to_geodetic(
    x: np.ndarray,
    y: Optional[np.ndarray] = None,
    z: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert ECF (Earth Centered Fixed) coordinates to geodetic.

    Uses the closed-form algorithm from Zhu, J. "Conversion of Earth-centered,
    Earth-fixed coordinates to geodetic coordinates." IEEE Transactions on
    Aerospace and Electronic Systems, 30(3), 1994.

    Parameters
    ----------
    x : np.ndarray
        ECF X coordinate(s) in meters, or a 3-element/3xN/Nx3 array of
        full ECF positions.
    y : np.ndarray, optional
        ECF Y coordinate(s) in meters.
    z : np.ndarray, optional
        ECF Z coordinate(s) in meters.

    Returns
    -------
    lat : np.ndarray
        Geodetic latitude in degrees.
    lon : np.ndarray
        Geodetic longitude in degrees.
    alt : np.ndarray
        Altitude above WGS-84 ellipsoid in meters.
    """
    x = np.asarray(x, dtype=np.float64)

    # Handle vector/matrix input
    if y is None and z is None:
        if x.ndim == 1 and x.size == 3:
            x, y, z = x[0], x[1], x[2]
        elif x.ndim == 2 and x.shape[0] == 3:
            x, y, z = x[0], x[1], x[2]
        elif x.ndim == 2 and x.shape[1] == 3:
            x, y, z = x[:, 0], x[:, 1], x[:, 2]
        else:
            raise ValueError(f"Invalid ECF shape {x.shape}. Expected (3,), (3,N), or (N,3)")
    else:
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)

    # Constants
    e2 = WGS84_E2
    a = WGS84_A
    e4 = e2 * e2
    ome2 = 1.0 - e2
    a2 = a * a
    b = WGS84_B
    b2 = b * b
    e_b2 = (a2 - b2) / b2

    z2 = z * z
    r2 = x * x + y * y
    r = np.sqrt(r2)

    # Check for valid solution
    valid = (a * r) ** 2 + (b * z) ** 2 > (a2 - b2) ** 2

    lon = np.full_like(x, np.nan, dtype=np.float64)
    lat = np.full_like(x, np.nan, dtype=np.float64)
    alt = np.full_like(x, np.nan, dtype=np.float64)

    lon[valid] = np.degrees(np.arctan2(y[valid], x[valid]))

    F = 54.0 * b2 * z2
    G = r2 + ome2 * z2 - e2 * (a2 - b2)
    c = e4 * F * r2 / (G * G * G)
    s = (1.0 + c + np.sqrt(c * c + 2.0 * c)) ** (1.0 / 3.0)
    templ = s + 1.0 / s + 1.0
    P = F / (3.0 * templ * templ * G * G)
    Q = np.sqrt(1.0 + 2.0 * e4 * P)
    r0 = (-P * e2 * r / (1.0 + Q) +
          np.sqrt(np.abs(0.5 * a2 * (1.0 + 1.0 / Q) -
                         P * ome2 * z2 / (Q * (1.0 + Q)) -
                         0.5 * P * r2)))
    temp2 = r - e2 * r0
    U = np.sqrt(temp2 * temp2 + z2)
    V = np.sqrt(temp2 * temp2 + ome2 * z2)
    z0 = b2 * z / (a * V)

    lat[valid] = np.degrees(np.arctan2(z[valid] + e_b2 * z0[valid], r[valid]))
    alt[valid] = U[valid] * (1.0 - b2 / (a * V[valid]))

    return lat, lon, alt


def geodetic_to_ecf(
    lat: np.ndarray,
    lon: Optional[np.ndarray] = None,
    alt: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert geodetic coordinates to ECF (Earth Centered Fixed).

    Parameters
    ----------
    lat : np.ndarray
        Geodetic latitude in degrees, or a 3-element/3xN/Nx3 array
        of [lat, lon, alt].
    lon : np.ndarray, optional
        Geodetic longitude in degrees.
    alt : np.ndarray, optional
        Altitude above WGS-84 ellipsoid in meters.

    Returns
    -------
    x : np.ndarray
        ECF X coordinate(s) in meters.
    y : np.ndarray
        ECF Y coordinate(s) in meters.
    z : np.ndarray
        ECF Z coordinate(s) in meters.
    """
    lat = np.asarray(lat, dtype=np.float64)

    if lon is None and alt is None:
        if lat.ndim == 1 and lat.size == 3:
            lat, lon, alt = lat[0], lat[1], lat[2]
        elif lat.ndim == 2 and lat.shape[0] == 3:
            lat, lon, alt = lat[0], lat[1], lat[2]
        elif lat.ndim == 2 and lat.shape[1] == 3:
            lat, lon, alt = lat[:, 0], lat[:, 1], lat[:, 2]
        else:
            raise ValueError(f"Invalid LLA shape. Expected (3,), (3,N), or (N,3)")
    else:
        lon = np.asarray(lon, dtype=np.float64)
        alt = np.asarray(alt, dtype=np.float64)

    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    R = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)

    x = (R + alt) * cos_lat * cos_lon
    y = (R + alt) * cos_lat * sin_lon
    z = (R + alt - WGS84_E2 * R) * sin_lat

    return x, y, z


# ===================================================================
# ECF <-> NED (North-East-Down)
# ===================================================================

def ecf_ned_rotation_matrix(orp_ecf: np.ndarray) -> np.ndarray:
    """
    Compute the ECF to NED rotation matrix at a given ECF position.

    Parameters
    ----------
    orp_ecf : np.ndarray
        Origin Reference Point in ECF coordinates, shape (3,).

    Returns
    -------
    np.ndarray
        3x3 rotation matrix from ECF to NED frame.
    """
    orp_ecf = np.asarray(orp_ecf, dtype=np.float64).ravel()
    lat, lon, _ = ecf_to_geodetic(orp_ecf)
    lat = float(lat)
    lon = float(lon)

    lat_rad = np.radians(-90.0 - lat)
    lon_rad = np.radians(lon)

    # Rotation about Y axis (latitude)
    Ry = np.array([
        [np.cos(lat_rad), 0, -np.sin(lat_rad)],
        [0, 1, 0],
        [np.sin(lat_rad), 0, np.cos(lat_rad)]
    ])

    # Rotation about Z axis (longitude)
    Rz = np.array([
        [np.cos(lon_rad), np.sin(lon_rad), 0],
        [-np.sin(lon_rad), np.cos(lon_rad), 0],
        [0, 0, 1]
    ])

    return Ry @ Rz


def ecf_to_ned(
    ecf_value: np.ndarray,
    orp_ecf: np.ndarray,
    is_position: bool = True
) -> np.ndarray:
    """
    Convert ECF coordinates to North-East-Down (NED) frame.

    Parameters
    ----------
    ecf_value : np.ndarray
        ECF coordinates, shape (3,) or (3, N).
    orp_ecf : np.ndarray
        Origin Reference Point in ECF, shape (3,).
    is_position : bool
        If True (default), subtract ORP before rotation (position coordinate).
        If False, only rotate (direction/velocity vector).

    Returns
    -------
    np.ndarray
        NED coordinates, same shape as ecf_value.
    """
    ecf_value = np.asarray(ecf_value, dtype=np.float64)
    orp_ecf = np.asarray(orp_ecf, dtype=np.float64).ravel()

    if is_position:
        ecf_value = ecf_value - orp_ecf.reshape(3, 1) if ecf_value.ndim == 2 else ecf_value - orp_ecf

    rot_mat = ecf_ned_rotation_matrix(orp_ecf)
    return rot_mat @ ecf_value


def ned_to_ecf(
    ned_value: np.ndarray,
    orp_ecf: np.ndarray,
    is_position: bool = True
) -> np.ndarray:
    """
    Convert North-East-Down (NED) coordinates to ECF.

    Parameters
    ----------
    ned_value : np.ndarray
        NED coordinates, shape (3,) or (3, N).
    orp_ecf : np.ndarray
        Origin Reference Point in ECF, shape (3,).
    is_position : bool
        If True (default), add ORP after rotation (position coordinate).
        If False, only rotate (direction/velocity vector).

    Returns
    -------
    np.ndarray
        ECF coordinates, same shape as ned_value.
    """
    ned_value = np.asarray(ned_value, dtype=np.float64)
    orp_ecf = np.asarray(orp_ecf, dtype=np.float64).ravel()

    rot_mat = ecf_ned_rotation_matrix(orp_ecf)
    ecf_value = rot_mat.T @ ned_value

    if is_position:
        if ecf_value.ndim == 2:
            ecf_value = ecf_value + orp_ecf.reshape(3, 1)
        else:
            ecf_value = ecf_value + orp_ecf

    return ecf_value


# ===================================================================
# ECF -> Geocentric
# ===================================================================

def ecf_to_geocentric(pos_ecf: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert ECF coordinates to geocentric latitude, longitude, altitude.

    Unlike geodetic coordinates, geocentric latitude is measured from
    the center of the Earth.

    Parameters
    ----------
    pos_ecf : np.ndarray
        ECF position, shape (3,).

    Returns
    -------
    lat : float
        Geocentric latitude in degrees.
    lon : float
        Geocentric longitude in degrees.
    alt : float
        Geocentric altitude in meters.
    """
    pos = np.asarray(pos_ecf, dtype=np.float64).ravel()
    x, y, z = pos[0], pos[1], pos[2]

    a = WGS84_A
    b = WGS84_B
    a2 = a * a
    b2 = b * b

    r2 = x * x + y * y
    r = np.sqrt(r2)

    if r2 > 0 or z * z > 0:
        lon = np.degrees(np.arctan2(y, x))
        lat = np.degrees(np.arctan2(z, r))
        R_ellip = np.sqrt(1.0 / (r2 / a2 + z * z / b2)) * np.sqrt(r2 + z * z)
        alt = np.sqrt(r2 + z * z) - R_ellip
    else:
        lon, lat, alt = 0.0, 0.0, 0.0

    return float(lat), float(lon), float(alt)


# ===================================================================
# RIC Frame
# ===================================================================

def ric_ecf_matrix(
    r_arp: np.ndarray,
    v_arp: np.ndarray,
    frame_type: str = 'ecf'
) -> np.ndarray:
    """
    Compute the RIC (Radial-InTrack-CrossTrack) to ECF transformation matrix.

    Parameters
    ----------
    r_arp : np.ndarray
        ARP position in ECF, shape (3,).
    v_arp : np.ndarray
        ARP velocity in ECF, shape (3,).
    frame_type : str
        Frame type: 'ecf' or 'eci'. Default 'ecf'.

    Returns
    -------
    np.ndarray
        3x3 transformation matrix [R, I, C] where R, I, C are unit vectors.
    """
    r_arp = np.asarray(r_arp, dtype=np.float64).ravel()
    v_arp = np.asarray(v_arp, dtype=np.float64).ravel()

    if frame_type.lower() == 'eci':
        w_e = 7.292115e-5  # Earth rotation rate (rad/s)
        vi = v_arp + np.cross(np.array([0.0, 0.0, w_e]), r_arp)
    else:
        vi = v_arp

    R_hat = r_arp / np.linalg.norm(r_arp)
    C_vec = np.cross(r_arp, vi)
    C_hat = C_vec / np.linalg.norm(C_vec)
    I_hat = np.cross(C_hat, R_hat)

    return np.column_stack([R_hat, I_hat, C_hat])


__all__ = [
    "ecf_to_geodetic",
    "geodetic_to_ecf",
    "ecf_ned_rotation_matrix",
    "ecf_to_ned",
    "ned_to_ecf",
    "ecf_to_geocentric",
    "ric_ecf_matrix",
]
