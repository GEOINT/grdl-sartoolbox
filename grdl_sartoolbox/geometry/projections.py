# -*- coding: utf-8 -*-
"""
SAR Projections - Image-to-ground and ground-to-image projection functions.

Implements the SICD sensor model projection functions from the MATLAB SAR
Toolbox, including R/Rdot contour intersections with ground planes, HAE
surfaces, and DEM surfaces.

Attribution
-----------
Ported from MATLAB SAR Toolbox (https://github.com/ngageoint/MATLAB_SAR)
Original: Thomas McDowall (Harris), Wade Schwartzkopf (NGA/IDT),
          Rocco Corsetti (NGA/IB)

License
-------
MIT License
Copyright (c) 2024 geoint.org
"""

from typing import Tuple, Optional
import numpy as np

from grdl_sartoolbox.utils.constants import WGS84_A, WGS84_B
from grdl_sartoolbox.geometry.analysis import wgs84_normal
from grdl_sartoolbox.geometry.coordinates import ecf_to_geodetic, geodetic_to_ecf


def point_to_ground_plane(
    r_tgt_coa: np.ndarray,
    rdot_tgt_coa: np.ndarray,
    arp_coa: np.ndarray,
    varp_coa: np.ndarray,
    gref: np.ndarray,
    gpn: np.ndarray
) -> np.ndarray:
    """
    Project R/Rdot contour to a ground plane.

    Solves for the intersection of an R/Rdot contour and a ground plane
    defined by a reference point and normal vector.

    Parameters
    ----------
    r_tgt_coa : np.ndarray
        Range to ARP at COA, shape (N,) or scalar.
    rdot_tgt_coa : np.ndarray
        Range rate at COA, shape (N,) or scalar.
    arp_coa : np.ndarray
        ARP position at COA, shape (3,) or (3, N).
    varp_coa : np.ndarray
        ARP velocity at COA, shape (3,) or (3, N).
    gref : np.ndarray
        Ground plane reference point, shape (3,) or (3, N).
    gpn : np.ndarray
        Ground plane normal vector, shape (3,) or (3, N).

    Returns
    -------
    np.ndarray
        Ground Plane Point(s), shape (3,) or (3, N).
    """
    r_tgt_coa = np.atleast_1d(np.asarray(r_tgt_coa, dtype=np.float64))
    rdot_tgt_coa = np.atleast_1d(np.asarray(rdot_tgt_coa, dtype=np.float64))
    arp_coa = np.asarray(arp_coa, dtype=np.float64)
    varp_coa = np.asarray(varp_coa, dtype=np.float64)
    gref = np.asarray(gref, dtype=np.float64)
    gpn = np.asarray(gpn, dtype=np.float64)

    n_pts = r_tgt_coa.size

    # Ensure column vectors for multi-point case
    if arp_coa.ndim == 1:
        arp_coa = np.tile(arp_coa[:, np.newaxis], (1, n_pts))
    if varp_coa.ndim == 1:
        varp_coa = np.tile(varp_coa[:, np.newaxis], (1, n_pts))
    if gref.ndim == 1:
        gref = np.tile(gref[:, np.newaxis], (1, n_pts))
    if gpn.ndim == 1:
        gpn = np.tile(gpn[:, np.newaxis], (1, n_pts))

    # Unit normal
    uZ = gpn / np.sqrt(np.sum(gpn**2, axis=0, keepdims=True))

    # ARP distance from plane
    ARPz = np.sum((arp_coa - gref) * uZ, axis=0)
    ARPz[ARPz > r_tgt_coa] = np.nan

    # ARP ground plane nadir
    AGPN = arp_coa - ARPz[np.newaxis, :] * uZ

    # Ground plane distance from nadir to range circle
    G = np.sqrt(r_tgt_coa**2 - ARPz**2)

    cos_GRAZ = G / r_tgt_coa
    sin_GRAZ = ARPz / r_tgt_coa

    # Velocity components
    VMag = np.sqrt(np.sum(varp_coa**2, axis=0))
    Vz = np.sum(varp_coa * uZ, axis=0)
    Vx = np.sqrt(VMag**2 - Vz**2)

    # Orient X such that Vx > 0
    uX = (varp_coa - Vz[np.newaxis, :] * uZ) / Vx[np.newaxis, :]
    uY = np.cross(uZ, uX, axis=0)

    # Cosine of azimuth angle
    cos_AZ = (-rdot_tgt_coa + Vz * sin_GRAZ) / (Vx * cos_GRAZ)
    cos_AZ = np.clip(cos_AZ, -1.0, 1.0)

    # Determine look direction
    LOOK = np.sign(np.sum(gpn * np.cross(arp_coa - gref, varp_coa, axis=0), axis=0))
    sin_AZ = LOOK * np.sqrt(1.0 - cos_AZ**2)

    # Ground Plane Point
    GPP = AGPN + (G * cos_AZ)[np.newaxis, :] * uX + (G * sin_AZ)[np.newaxis, :] * uY

    if n_pts == 1:
        return GPP.ravel()
    return GPP


def point_to_hae(
    r_tgt_coa: np.ndarray,
    rdot_tgt_coa: np.ndarray,
    arp_coa: np.ndarray,
    varp_coa: np.ndarray,
    scp: np.ndarray,
    hae0: float,
    delta_hae_max: float = 1.0,
    n_lim: int = 3
) -> np.ndarray:
    """
    Project R/Rdot contour to a constant height above ellipsoid (HAE) surface.

    Iteratively projects to ground planes, refining until the height
    converges to within delta_hae_max of the target HAE.

    Parameters
    ----------
    r_tgt_coa : np.ndarray
        Range to ARP at COA, scalar or shape (N,).
    rdot_tgt_coa : np.ndarray
        Range rate at COA, scalar or shape (N,).
    arp_coa : np.ndarray
        ARP position at COA, shape (3,) or (3, N).
    varp_coa : np.ndarray
        ARP velocity at COA, shape (3,) or (3, N).
    scp : np.ndarray
        Scene Center Point, shape (3,).
    hae0 : float
        Target height above ellipsoid (meters).
    delta_hae_max : float
        Height convergence threshold (meters). Default 1.0.
    n_lim : int
        Maximum iterations. Default 3.

    Returns
    -------
    np.ndarray
        Surface Projection Point(s), shape (3,) or (3, N).
    """
    r_tgt_coa = np.atleast_1d(np.asarray(r_tgt_coa, dtype=np.float64))
    rdot_tgt_coa = np.atleast_1d(np.asarray(rdot_tgt_coa, dtype=np.float64))
    arp_coa = np.asarray(arp_coa, dtype=np.float64)
    varp_coa = np.asarray(varp_coa, dtype=np.float64)
    scp = np.asarray(scp, dtype=np.float64).ravel()

    n_pts = r_tgt_coa.size

    if arp_coa.ndim == 1:
        arp_coa = np.tile(arp_coa[:, np.newaxis], (1, n_pts))
    if varp_coa.ndim == 1:
        varp_coa = np.tile(varp_coa[:, np.newaxis], (1, n_pts))

    iters = 0
    delta_HAE = np.full(n_pts, np.inf)

    # (1) Ground plane normal at SCP
    uGPN = wgs84_normal(scp)  # shape (3,)
    scp_lat, scp_lon, scp_alt = ecf_to_geodetic(scp)
    scp_alt = float(scp_alt)

    # Initial ground reference
    GREF = scp[:, np.newaxis] - (scp_alt - hae0) * uGPN[:, np.newaxis]
    uGPN_col = uGPN[:, np.newaxis]

    while np.all(np.abs(delta_HAE) > delta_hae_max) and iters <= n_lim:
        # (2) Project to ground plane
        GPP = point_to_ground_plane(
            r_tgt_coa, rdot_tgt_coa, arp_coa, varp_coa,
            GREF if GREF.ndim == 2 else GREF.ravel(),
            uGPN_col if uGPN_col.ndim == 2 else uGPN_col.ravel()
        )
        if GPP.ndim == 1:
            GPP = GPP[:, np.newaxis]

        # (3) Update ground plane normal and height
        for i in range(n_pts):
            gpp_n = wgs84_normal(GPP[:, i])
            uGPN_col[:, 0] = gpp_n if n_pts == 1 else gpp_n

        gpp_lat, gpp_lon, gpp_alt = ecf_to_geodetic(GPP[0], GPP[1], GPP[2])
        delta_HAE = np.atleast_1d(gpp_alt) - hae0

        GREF = GPP - delta_HAE[np.newaxis, :] * uGPN_col
        iters += 1

    # (5) Final slant plane projection
    LOOK = np.sign(np.sum(
        uGPN_col * np.cross(varp_coa, GPP - arp_coa, axis=0),
        axis=0
    ))
    SPN = LOOK[np.newaxis, :] * np.cross(varp_coa, GPP - arp_coa, axis=0)
    uSPN = SPN / np.sqrt(np.sum(SPN**2, axis=0, keepdims=True))

    SF = np.sum(uGPN_col * uSPN, axis=0)
    SLP = GPP - (delta_HAE / SF)[np.newaxis, :] * uSPN

    # (7) Adjust HAE to exact target
    slp_lat, slp_lon, slp_alt = ecf_to_geodetic(SLP[0], SLP[1], SLP[2])
    spp_x, spp_y, spp_z = geodetic_to_ecf(slp_lat, slp_lon, np.full_like(slp_lat, hae0))

    SPP = np.array([spp_x, spp_y, spp_z])
    if n_pts == 1:
        return SPP.ravel()
    return SPP


def point_to_hae_newton(
    r_tgt_coa: np.ndarray,
    rdot_tgt_coa: np.ndarray,
    arp_coa: np.ndarray,
    varp_coa: np.ndarray,
    scp: np.ndarray,
    hae0: float,
    delta_max: float = 1e-3,
    n_lim: int = 10
) -> np.ndarray:
    """
    Project R/Rdot contour to HAE surface using Newton's method.

    More suitable for spotlight-mode data. Solves the simultaneous
    equations for range, Doppler, and ellipsoid surface.

    Parameters
    ----------
    r_tgt_coa : np.ndarray
        Range to ARP at COA, scalar or shape (N,).
    rdot_tgt_coa : np.ndarray
        Range rate at COA, scalar or shape (N,).
    arp_coa : np.ndarray
        ARP position at COA, shape (3,) or (3, N).
    varp_coa : np.ndarray
        ARP velocity at COA, shape (3,) or (3, N).
    scp : np.ndarray
        Scene Center Point (initial guess), shape (3,).
    hae0 : float
        Target height above ellipsoid (meters).
    delta_max : float
        Convergence threshold. Default 1e-3.
    n_lim : int
        Maximum iterations. Default 10.

    Returns
    -------
    np.ndarray
        Surface Projection Point(s), shape (3,) or (3, N).
    """
    r_tgt_coa = np.atleast_1d(np.asarray(r_tgt_coa, dtype=np.float64))
    rdot_tgt_coa = np.atleast_1d(np.asarray(rdot_tgt_coa, dtype=np.float64))
    scp = np.asarray(scp, dtype=np.float64).ravel()

    n_pts = r_tgt_coa.size
    a = WGS84_A
    b = WGS84_B

    SPP = np.zeros((3, n_pts))

    for idx in range(n_pts):
        # Get per-point ARP/VARP
        if arp_coa.ndim == 2:
            R_ARP = arp_coa[:, idx]
            V_ARP = varp_coa[:, idx]
        else:
            R_ARP = np.asarray(arp_coa, dtype=np.float64).ravel()
            V_ARP = np.asarray(varp_coa, dtype=np.float64).ravel()

        Range_obs = r_tgt_coa[idx]
        Doppler_obs = -rdot_tgt_coa[idx]

        R0 = scp.copy()
        h = hae0

        iters = 0
        delta = np.inf

        while np.max(np.abs(delta)) > delta_max and iters < n_lim:
            F1 = np.zeros((3, 3))
            F2 = np.zeros((3, 3))
            F3 = np.zeros((3, 3))

            for i in range(3):
                for j, eps in enumerate([-0.5, 0.0, 0.5]):
                    R = R0.copy()
                    R[i] += eps
                    X, Y, Z = R

                    r_est = R - R_ARP
                    Range_est = np.linalg.norm(r_est)
                    Doppler_est = np.dot(V_ARP, r_est) / Range_est

                    F1[i, j] = Range_obs - Range_est
                    F2[i, j] = Doppler_obs - Doppler_est
                    F3[i, j] = ((X**2 + Y**2) / (a + h)**2 +
                                Z**2 / (b + h)**2 - 1)

            # Numerical Jacobian
            dF1 = (F1[:, 2] - F1[:, 0])
            dF2 = (F2[:, 2] - F2[:, 0])
            dF3 = (F3[:, 2] - F3[:, 0])

            f = -np.array([F1[0, 1], F2[0, 1], F3[0, 1]])
            B = np.array([dF1, dF2, dF3])

            try:
                delta = np.linalg.solve(B, f)
            except np.linalg.LinAlgError:
                break

            R0 += delta
            iters += 1

        SPP[:, idx] = R0

    if n_pts == 1:
        return SPP.ravel()
    return SPP


def point_slant_to_ground(
    points: np.ndarray,
    metadata: dict,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project slant plane pixel coordinates to geodetic ground coordinates.

    Wrapper that converts 1-based pixel indices to geodetic lat/lon/alt
    via the SICD projection model.

    Parameters
    ----------
    points : np.ndarray
        Pixel coordinates (1-based), shape (2,) or (2, N) as [row, col].
    metadata : dict
        SICD metadata dictionary with required fields for projection.
    **kwargs
        Additional keyword arguments passed to the projection function.

    Returns
    -------
    lat : np.ndarray
        Geodetic latitude(s) in degrees.
    lon : np.ndarray
        Geodetic longitude(s) in degrees.
    alt : np.ndarray
        Height above ellipsoid in meters.
    """
    # Convert 1-based to 0-based
    points_0based = np.asarray(points, dtype=np.float64) - 1
    # This would call point_image_to_ground and then ecf_to_geodetic
    # For now, raise NotImplementedError as it requires full SICD metadata parsing
    raise NotImplementedError(
        "point_slant_to_ground requires full SICD metadata parsing. "
        "Use GRDL's SICDGeolocation for this functionality."
    )


def point_ground_to_slant(
    points: np.ndarray,
    metadata: dict,
    **kwargs
) -> np.ndarray:
    """
    Project geodetic ground coordinates to slant plane pixel coordinates.

    Wrapper that converts geodetic lat/lon/alt to 1-based pixel indices
    via the SICD projection model.

    Parameters
    ----------
    points : np.ndarray
        Geodetic coordinates [lat, lon, alt] in degrees/meters.
    metadata : dict
        SICD metadata dictionary.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    np.ndarray
        Pixel coordinates (1-based) as [row, col].
    """
    raise NotImplementedError(
        "point_ground_to_slant requires full SICD metadata parsing. "
        "Use GRDL's SICDGeolocation for this functionality."
    )


__all__ = [
    "point_to_ground_plane",
    "point_to_hae",
    "point_to_hae_newton",
    "point_slant_to_ground",
    "point_ground_to_slant",
]
