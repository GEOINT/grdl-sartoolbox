# -*- coding: utf-8 -*-
"""
DTED Reader - Read DTED (Digital Terrain Elevation Data) files.

Reads elevation data in the DTED format as described in
MIL-PRF-89020B (23 May 2000).

Attribution
-----------
Ported from MATLAB SAR Toolbox (https://github.com/ngageoint/MATLAB_SAR)
Original: Tom Krauss, NGA/IDT

License
-------
MIT License
Copyright (c) 2024 geoint.org
"""

from typing import Tuple, Optional
from dataclasses import dataclass, field
import struct
import numpy as np


@dataclass
class DTEDMetadata:
    """
    DTED file metadata.

    Attributes
    ----------
    sec_class : str
        Security classification.
    lat_origin : float
        Latitude origin in decimal degrees.
    lon_origin : float
        Longitude origin in decimal degrees.
    lat_spacing : float
        Latitude spacing in decimal degrees.
    lon_spacing : float
        Longitude spacing in decimal degrees.
    num_lat_lines : int
        Number of latitude points.
    num_lon_lines : int
        Number of longitude points.
    vert_datum : str
        Vertical datum identifier.
    horiz_datum : str
        Horizontal datum identifier.
    abs_hor_acc : float
        Absolute horizontal accuracy.
    abs_vert_acc : float
        Absolute vertical accuracy.
    """
    sec_class: str = ''
    lat_origin: float = 0.0
    lon_origin: float = 0.0
    lat_spacing: float = 0.0
    lon_spacing: float = 0.0
    num_lat_lines: int = 0
    num_lon_lines: int = 0
    vert_datum: str = ''
    horiz_datum: str = ''
    abs_hor_acc: float = 0.0
    abs_vert_acc: float = 0.0


def _parse_dms(deg_str: str, min_str: str, sec_str: str, dir_char: str) -> float:
    """Parse degree-minute-second strings to decimal degrees."""
    deg = float(deg_str) if deg_str.strip() else 0.0
    mins = float(min_str) if min_str.strip() else 0.0
    secs = float(sec_str) if sec_str.strip() else 0.0
    dd = abs(deg) + mins / 60.0 + secs / 3600.0
    if dir_char.upper() in ('S', 'W'):
        dd = -dd
    if deg < 0:
        dd = -dd
    return dd


def read_dted(
    filename: str,
    ll: Optional[Tuple[float, float]] = None,
    ur: Optional[Tuple[float, float]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, DTEDMetadata]:
    """
    Read a DTED elevation data file.

    Parameters
    ----------
    filename : str
        Path to the DTED file.
    ll : tuple of float, optional
        Lower-left corner (lat, lon) in degrees to constrain reading.
    ur : tuple of float, optional
        Upper-right corner (lat, lon) in degrees to constrain reading.

    Returns
    -------
    elevations : np.ndarray
        Elevation data in meters above MSL, shape (num_lons, num_lats).
    lats : np.ndarray
        Latitude values in decimal degrees.
    lons : np.ndarray
        Longitude values in decimal degrees.
    meta : DTEDMetadata
        File metadata.

    Raises
    ------
    FileNotFoundError
        If the file cannot be opened.
    ValueError
        If the file is not in DTED format.
    """
    meta = DTEDMetadata()

    with open(filename, 'rb') as f:
        # Check UHL header
        uhl_hdr = f.read(3).decode('ascii')
        if uhl_hdr != 'UHL':
            raise ValueError(
                'File not DTED format or unsupported (compressed) DTED format'
            )

        # Skip to DSI record (offset 80)
        f.seek(80)
        dsi_hdr = f.read(3).decode('ascii')
        if dsi_hdr != 'DSI':
            raise ValueError('DSI record not found: not DTED format')

        meta.sec_class = f.read(1).decode('ascii')
        f.read(2)   # sec_mark
        f.read(27)  # sec_hand
        f.read(26)  # fill

        f.read(5)   # dma_desc
        f.read(23)  # fill

        f.read(2)   # data_edition
        f.read(1)   # match_version
        f.read(4)   # maint_date
        f.read(4)   # match_date
        f.read(4)   # maint_desc_code
        f.read(8)   # producer_code
        f.read(16)  # fill

        f.read(9)   # product_vers
        f.read(2)   # fill

        f.read(4)   # product_date
        meta.vert_datum = f.read(3).decode('ascii').strip()
        meta.horiz_datum = f.read(5).decode('ascii').strip()
        f.read(10)  # dig_coll_sys
        f.read(4)   # comp_date
        f.read(22)  # fill

        # Latitude origin
        lat_deg_s = f.read(2).decode('ascii')
        lat_min_s = f.read(2).decode('ascii')
        lat_sec_s = f.read(4).decode('ascii')
        lat_dir = f.read(1).decode('ascii')
        meta.lat_origin = _parse_dms(lat_deg_s, lat_min_s, lat_sec_s, lat_dir)

        # Longitude origin
        lon_deg_s = f.read(3).decode('ascii')
        lon_min_s = f.read(3).decode('ascii')
        lon_sec_s = f.read(3).decode('ascii')
        lon_dir = f.read(1).decode('ascii')
        meta.lon_origin = _parse_dms(lon_deg_s, lon_min_s, lon_sec_s, lon_dir)

        # Corner coordinates (skip)
        f.read(60)

        f.read(9)  # orient_angle

        lat_spacing_raw = f.read(4).decode('ascii').strip()
        lon_spacing_raw = f.read(4).decode('ascii').strip()
        meta.lat_spacing = float(lat_spacing_raw) / (10 * 60 * 60) if lat_spacing_raw else 0.0
        meta.lon_spacing = float(lon_spacing_raw) / (10 * 60 * 60) if lon_spacing_raw else 0.0

        num_lat_s = f.read(4).decode('ascii').strip()
        num_lon_s = f.read(4).decode('ascii').strip()
        meta.num_lat_lines = int(num_lat_s) if num_lat_s else 0
        meta.num_lon_lines = int(num_lon_s) if num_lon_s else 0

        f.read(2)    # part_cell_ind
        f.read(357)  # fill

        # Accuracy record
        acc_hdr = f.read(3).decode('ascii')
        if acc_hdr != 'ACC':
            raise ValueError('ACC record not found: not DTED format')

        abs_hor_s = f.read(4).decode('ascii').strip()
        abs_vert_s = f.read(4).decode('ascii').strip()
        meta.abs_hor_acc = float(abs_hor_s) if abs_hor_s else 0.0
        meta.abs_vert_acc = float(abs_vert_s) if abs_vert_s else 0.0
        f.read(4)   # pt2pt_hor
        f.read(4)   # pt2pt_vert
        f.read(36)  # fill
        f.read(2)   # mult_acc_out_flg

        # Build lat/lon arrays
        lats = np.arange(meta.num_lat_lines) * meta.lat_spacing + meta.lat_origin
        lons = np.arange(meta.num_lon_lines) * meta.lon_spacing + meta.lon_origin

        # Constrain to region if specified
        if ll is not None and ur is not None:
            lat_lo = np.searchsorted(lats, ll[0])
            lat_hi = np.searchsorted(lats, ur[0], side='right') - 1
            lon_lo = np.searchsorted(lons, ll[1])
            lon_hi = np.searchsorted(lons, ur[1], side='right') - 1
        else:
            lat_lo, lat_hi = 0, meta.num_lat_lines - 1
            lon_lo, lon_hi = 0, meta.num_lon_lines - 1

        lats = lats[lat_lo:lat_hi + 1]
        lons = lons[lon_lo:lon_hi + 1]

        # Data record layout:
        # 1 byte sentinel + 3 byte count + 2 byte lon + 2 byte lat + N*2 byte elev + 4 byte checksum
        data_record_length = 1 + 3 + 2 + 2 + 2 * meta.num_lat_lines + 4

        # Seek to first record
        f.seek(3428 + lon_lo * data_record_length)

        n_lon_read = lon_hi - lon_lo + 1
        bulk_data = np.frombuffer(
            f.read(data_record_length * n_lon_read),
            dtype=np.uint8
        ).reshape(n_lon_read, data_record_length)

        # Extract elevation bytes (skip header: 8 bytes, skip checksum: 4 bytes)
        high_bytes = bulk_data[:, 8:data_record_length - 4:2].astype(np.int16)
        low_bytes = bulk_data[:, 9:data_record_length - 4:2].astype(np.int16)

        # Handle sign bit
        sign = np.ones_like(high_bytes)
        neg_mask = high_bytes > 128
        high_bytes[neg_mask] -= 128
        sign[neg_mask] = -1

        # Compute elevations
        all_elevations = sign * (high_bytes * 256 + low_bytes)

        # Extract requested latitude range
        elevations = all_elevations[:, lat_lo:lat_hi + 1].astype(np.float64)

        # Clean up voids
        elevations[elevations < -50] = -50.0

    return elevations, lats, lons, meta


__all__ = ["read_dted", "DTEDMetadata"]
