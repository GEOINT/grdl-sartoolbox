# -*- coding: utf-8 -*-
"""
Polar Format Algorithm (PFA) - FFT-based SAR image formation algorithm.

Implements the Polar Format Algorithm for forming SAR images from Complex
Phase History Data (CPHD). PFA is a frequency-domain algorithm that maps
polar-sampled k-space data onto a rectangular grid via interpolation, then
applies 2D FFTs to form the image.

Attribution
-----------
Ported from MATLAB SAR Toolbox (https://github.com/ngageoint/MATLAB_SAR)
Original implementation by Wade Schwartzkopf and Tom Krauss (NGA/IDT)
Modified by LeRoy Gorham (AFRL)

Reference
---------
Carrara, W.G., Goodman, R.S., and Majewski, R.M., "Spotlight Synthetic
Aperture Radar: Signal Processing Algorithms," Artech House, 1995.

Dependencies
------------
numpy - Array operations, FFT
scipy.interpolate - Polar to rectangular interpolation

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
from scipy import interpolate

# GRDL internal
from grdl_sartoolbox.utils.constants import SPEED_OF_LIGHT
from grdl_sartoolbox.geometry.analysis import wgs84_normal


# ===================================================================
# Data Structures
# ===================================================================

@dataclass
class NarrowbandData:
    """
    Narrowband metadata for CPHD data.

    Attributes
    ----------
    TxPos : np.ndarray
        Transmitter position (meters), shape (num_pulses, 3) [X, Y, Z].
    RcvPos : np.ndarray
        Receiver position (meters), shape (num_pulses, 3) [X, Y, Z].
    SRPPos : np.ndarray
        Stabilization Reference Point position (meters), shape (num_pulses, 3).
    SC0 : np.ndarray
        Start frequency for each pulse (Hz), shape (num_pulses,).
    SCSS : np.ndarray
        Frequency step size for each pulse (Hz), shape (num_pulses,).
    """
    TxPos: np.ndarray
    RcvPos: np.ndarray
    SRPPos: np.ndarray
    SC0: np.ndarray
    SCSS: np.ndarray

    def __post_init__(self):
        """Validate narrowband data."""
        num_pulses = len(self.SC0)

        if self.TxPos.shape != (num_pulses, 3):
            raise ValueError(f"TxPos must be shape ({num_pulses}, 3), got {self.TxPos.shape}")
        if self.RcvPos.shape != (num_pulses, 3):
            raise ValueError(f"RcvPos must be shape ({num_pulses}, 3), got {self.RcvPos.shape}")
        if self.SRPPos.shape != (num_pulses, 3):
            raise ValueError(f"SRPPos must be shape ({num_pulses}, 3), got {self.SRPPos.shape}")
        if len(self.SCSS) != num_pulses:
            raise ValueError(f"SCSS length {len(self.SCSS)} != num_pulses {num_pulses}")


# ===================================================================
# Helper Functions
# ===================================================================

def _sinc_interp(x: np.ndarray, s: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Sinc interpolation.

    Parameters
    ----------
    x : np.ndarray
        Data to interpolate (row vector).
    s : np.ndarray
        Sample positions (row vector).
    u : np.ndarray
        Output positions (row vector).

    Returns
    -------
    np.ndarray
        Interpolated data at positions u.
    """
    T = s[1] - s[0]  # Sample spacing
    sincM = u[:, np.newaxis] - s[np.newaxis, :]  # (len(u), len(s))
    y = x @ np.sinc(sincM / T).T
    return y


def _pfa_bistatic_pos(
    tx_pos: np.ndarray,
    rcv_pos: np.ndarray,
    srp: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute equivalent monostatic parameters for bistatic collection.

    Parameters
    ----------
    tx_pos : np.ndarray
        Transmitter positions, shape (num_pulses, 3).
    rcv_pos : np.ndarray
        Receiver positions, shape (num_pulses, 3).
    srp : np.ndarray
        Stabilization Reference Point, shape (num_pulses, 3).

    Returns
    -------
    bi_pos : np.ndarray
        Equivalent bistatic positions, shape (num_pulses, 3).
    freq_scale : np.ndarray
        Frequency scaling factor for bistatic, shape (num_pulses,).
    """
    tx_range = tx_pos - srp
    rcv_range = rcv_pos - srp

    tx_mag = np.linalg.norm(tx_range, axis=1)
    rcv_mag = np.linalg.norm(rcv_range, axis=1)

    tx_unit = tx_range / tx_mag[:, np.newaxis]
    rcv_unit = rcv_range / rcv_mag[:, np.newaxis]

    bisector = tx_unit + rcv_unit
    bisector = bisector / np.linalg.norm(bisector, axis=1)[:, np.newaxis]

    # Equivalent bistatic position
    bi_pos = bisector * (tx_mag + rcv_mag)[:, np.newaxis]

    # Frequency scaling: cos(beta/2) where beta is bistatic angle
    dot_product = np.sum(tx_unit * rcv_unit, axis=1)
    # Clamp to avoid numerical issues with arccos
    dot_product = np.clip(dot_product, -1.0, 1.0)
    bistatic_angle = np.arccos(dot_product)
    freq_scale = np.cos(bistatic_angle / 2)

    return bi_pos, freq_scale


def _pfa_polar_coords(
    pos: np.ndarray,
    scp: np.ndarray,
    coa_pos: np.ndarray,
    ipn: np.ndarray,
    fpn: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute polar coordinates for PFA.

    Parameters
    ----------
    pos : np.ndarray
        Sensor positions, shape (num_pulses, 3).
    scp : np.ndarray
        Scene center point, shape (3,).
    coa_pos : np.ndarray
        Center of aperture position, shape (3,).
    ipn : np.ndarray
        Image plane normal unit vector, shape (3,).
    fpn : np.ndarray
        Focus plane normal unit vector, shape (3,).

    Returns
    -------
    k_a : np.ndarray
        Polar angles (radians), shape (num_pulses,).
    k_sf : np.ndarray
        Frequency scale factors, shape (num_pulses,).
    """
    def project(points, line_dir, point_in_plane, plane_normal):
        """Project points onto plane."""
        d = (np.dot(point_in_plane - points, plane_normal) /
             np.dot(line_dir, plane_normal))
        return points + d[:, np.newaxis] * line_dir

    # Project positions to image plane
    ip_pos = project(pos, fpn, scp, ipn)
    ip_coa_pos = project(coa_pos.reshape(1, 3), fpn, scp, ipn)[0]

    # Image plane coordinate system
    ipx = ip_coa_pos - scp
    ipx = ipx / np.linalg.norm(ipx)
    ipy = np.cross(ipx, ipn)

    # Polar angle (counter-clockwise)
    ip_range_vectors = ip_pos - scp
    k_a = -np.arctan2(np.dot(ip_range_vectors, ipy),
                      np.dot(ip_range_vectors, ipx))

    # Frequency scale factor
    range_vectors = pos - scp
    range_vectors = range_vectors / np.linalg.norm(range_vectors, axis=1)[:, np.newaxis]
    sin_graze = np.dot(range_vectors, fpn)

    ip_range_vectors = ip_range_vectors / np.linalg.norm(ip_range_vectors, axis=1)[:, np.newaxis]
    sin_graze_ip = np.dot(ip_range_vectors, fpn)

    k_sf = np.sqrt(1 - sin_graze**2) / np.sqrt(1 - sin_graze_ip**2)

    return k_a, k_sf


def _pfa_inscribed_rectangle_coords(
    k_a: np.ndarray,
    k_r0: np.ndarray,
    bw: np.ndarray
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Compute inscribed rectangle coordinates in k-space.

    Parameters
    ----------
    k_a : np.ndarray
        Polar angles, shape (num_pulses,).
    k_r0 : np.ndarray
        Start frequency for each pulse, shape (num_pulses,).
    bw : np.ndarray
        Bandwidth for each pulse, shape (num_pulses,).

    Returns
    -------
    k_v_bounds : tuple of float
        (min, max) bounds in V dimension.
    k_u_bounds : tuple of float
        (min, max) bounds in U dimension.
    """
    # Bottom of inscribed rectangle
    k_v_min = np.max(k_r0 * np.cos(k_a))

    # Sides (angular bounds)
    k_u_bounds = (k_v_min * np.tan(np.min(k_a)),
                  k_v_min * np.tan(np.max(k_a)))

    # Top of inscribed rectangle
    k_r_high = k_r0 + bw
    k_u_high = k_r_high * np.sin(k_a)
    k_v_high = k_r_high * np.cos(k_a)

    valid = (k_u_high >= k_u_bounds[0]) & (k_u_high <= k_u_bounds[1])
    if np.any(valid):
        k_v_max = np.min(k_v_high[valid])
    else:
        # If no valid points, use minimum of all high frequencies
        k_v_max = np.min(k_v_high)

    k_v_bounds = (k_v_min, k_v_max)

    return k_v_bounds, k_u_bounds


def _pfa_interp_range(
    phase_history: np.ndarray,
    k_a: np.ndarray,
    k_r0: np.ndarray,
    k_r_ss: np.ndarray,
    new_v: np.ndarray,
    interp_type: str = 'linear'
) -> np.ndarray:
    """
    Range interpolation (radial) for PFA.

    Parameters
    ----------
    phase_history : np.ndarray
        Complex phase history, shape (num_samples, num_pulses).
    k_a : np.ndarray
        Polar angles, shape (num_pulses,).
    k_r0 : np.ndarray
        Start frequency, shape (num_pulses,) or scalar.
    k_r_ss : np.ndarray
        Frequency step, shape (num_pulses,) or scalar.
    new_v : np.ndarray
        V coordinates to interpolate to, shape (num_samples_new,).
    interp_type : str
        Interpolation type: 'linear' or 'sinc'.

    Returns
    -------
    np.ndarray
        Interpolated phase history, shape (num_samples_new, num_pulses).
    """
    num_samples, num_pulses = phase_history.shape

    # Expand scalar arguments
    k_r0 = np.atleast_1d(k_r0)
    k_r_ss = np.atleast_1d(k_r_ss)
    if len(k_r0) == 1:
        k_r0 = np.full(num_pulses, k_r0[0])
    if len(k_r_ss) == 1:
        k_r_ss = np.full(num_pulses, k_r_ss[0])

    output = np.zeros((len(new_v), num_pulses), dtype=phase_history.dtype)

    for pulse in range(num_pulses):
        # Radial positions
        k_r = k_r0[pulse] + k_r_ss[pulse] * np.arange(num_samples)

        # V coordinates (rectangular)
        k_v = k_r * np.cos(k_a[pulse])

        # Interpolate
        if interp_type == 'sinc':
            output[:, pulse] = _sinc_interp(phase_history[:, pulse], k_v, new_v)
        else:
            f = interpolate.interp1d(k_v, phase_history[:, pulse],
                                    kind=interp_type, bounds_error=False,
                                    fill_value=0.0)
            output[:, pulse] = f(new_v)

    return output


def _pfa_interp_azimuth(
    phase_history: np.ndarray,
    k_a: np.ndarray,
    v_coords: np.ndarray,
    new_u: np.ndarray,
    interp_type: str = 'linear'
) -> np.ndarray:
    """
    Azimuth interpolation (cross-range) for PFA.

    Parameters
    ----------
    phase_history : np.ndarray
        Phase history after range interpolation, shape (num_pulses, num_samples).
        Each row is a pulse.
    k_a : np.ndarray
        Polar angles, shape (num_pulses,).
    v_coords : np.ndarray
        V coordinates for each sample, shape (num_samples,).
    new_u : np.ndarray
        U coordinates to interpolate to, shape (num_samples_new,).
    interp_type : str
        Interpolation type: 'linear' or 'sinc'.

    Returns
    -------
    np.ndarray
        Interpolated phase history, shape (num_samples_new, num_samples).
    """
    num_pulses, num_samples = phase_history.shape
    output = np.zeros((len(new_u), num_samples), dtype=phase_history.dtype)

    for sample in range(num_samples):
        # U coordinates
        k_u = np.tan(k_a) * v_coords[sample]

        # Interpolate
        if interp_type == 'sinc':
            output[:, sample] = _sinc_interp(phase_history[:, sample], k_u, new_u)
        else:
            f = interpolate.interp1d(k_u, phase_history[:, sample],
                                    kind=interp_type, bounds_error=False,
                                    fill_value=0.0)
            output[:, sample] = f(new_u)

    return output


def _pfa_fft_zeropad_1d(
    data: np.ndarray,
    sample_rate: float = 1.5
) -> np.ndarray:
    """
    Zero-padded 1D FFT along first dimension.

    Parameters
    ----------
    data : np.ndarray
        Input data, shape (rows, cols).
    sample_rate : float
        Samples per IPR (Impulse Response). Controls zero-padding.

    Returns
    -------
    np.ndarray
        FFT result, shape (rows * sample_rate, cols).
    """
    rows, cols = data.shape

    # Zero-pad array
    zeropad_size = int(np.floor(rows * sample_rate))
    zeropad = np.zeros((zeropad_size, cols), dtype=data.dtype)

    # Insert data into center
    start_index = int(np.floor(rows * (sample_rate - 1) / 2))
    zeropad[start_index:start_index + rows, :] = data

    # FFT: ifftshift -> ifft -> fftshift
    zeropad = np.fft.ifftshift(zeropad, axes=0)
    result = np.fft.fftshift(np.fft.ifft(zeropad, axis=0), axes=0)

    return result


# ===================================================================
# Main PFA Functions
# ===================================================================

def pfa_mem(
    phase_history: np.ndarray,
    nbdata: NarrowbandData,
    sample_rate: float = 1.5,
    verbose: bool = False
) -> np.ndarray:
    """
    Polar Format Algorithm (PFA) for SAR image formation.

    Implements the frequency-domain PFA algorithm that maps polar-sampled
    k-space data onto a rectangular grid via interpolation, then applies
    2D FFTs to form the complex SAR image.

    Parameters
    ----------
    phase_history : np.ndarray
        Complex phase history data, shape (num_samples, num_pulses).
        Each column is a pulse (fast time along rows, slow time along cols).
    nbdata : NarrowbandData
        Narrowband metadata including sensor positions and frequencies.
    sample_rate : float, optional
        Samples per IPR for zero-padding. Default is 1.5.
    verbose : bool, optional
        Print progress messages. Default is False.

    Returns
    -------
    np.ndarray
        Complex SAR image, shape determined by sample_rate.

    Raises
    ------
    ValueError
        If SRPPos varies (non-spotlight data not supported).

    Examples
    --------
    >>> # Generate synthetic CPHD data
    >>> num_samples, num_pulses = 512, 256
    >>> phase_history = np.random.randn(num_samples, num_pulses) + \
    ...                 1j * np.random.randn(num_samples, num_pulses)
    >>>
    >>> # Create narrowband metadata
    >>> tx_pos = np.random.randn(num_pulses, 3) * 1000
    >>> nbdata = NarrowbandData(
    ...     TxPos=tx_pos,
    ...     RcvPos=tx_pos,  # Monostatic
    ...     SRPPos=np.tile([0, 0, 0], (num_pulses, 1)),
    ...     SC0=np.ones(num_pulses) * 10e9,
    ...     SCSS=np.ones(num_pulses) * 1e6
    ... )
    >>>
    >>> image = pfa_mem(phase_history, nbdata)
    >>> image.shape  # Depends on sample_rate

    Notes
    -----
    - Currently only supports spotlight data (constant SRP)
    - Uses linear interpolation by default (sinc available but slower)
    - Handles bistatic collections via equivalent monostatic position
    - Units in k-space are cycles/meter (consistent with SICD)

    References
    ----------
    Carrara, W.G., Goodman, R.S., and Majewski, R.M., "Spotlight Synthetic
    Aperture Radar: Signal Processing Algorithms," Artech House, 1995.
    """
    if verbose:
        print(f"PFA: Processing {phase_history.shape[1]} pulses...")

    # Check for spotlight data (constant SRP)
    if not np.allclose(nbdata.SRPPos, nbdata.SRPPos[0], atol=1e-6):
        raise ValueError("Non-spotlight data not supported (SRPPos must be constant)")

    scp = nbdata.SRPPos[0]

    # Compute focus plane normal at actual SCP location
    fpn = wgs84_normal(scp)  # Focus plane normal (WGS-84 ellipsoid normal)

    # Compute equivalent bistatic positions
    bi_pos, bi_freq_scale = _pfa_bistatic_pos(nbdata.TxPos, nbdata.RcvPos, nbdata.SRPPos)

    # Shift coordinate system so SCP is at origin
    bi_pos = bi_pos - scp  # Shift positions
    scp_origin = np.array([0.0, 0.0, 0.0])

    ref_pulse = phase_history.shape[1] // 2
    arp_coa = bi_pos[ref_pulse]
    arp_coa_vel = bi_pos[ref_pulse + 1] - bi_pos[ref_pulse - 1]

    # Slant plane unit normal
    srv = arp_coa
    slant_cross = np.cross(srv, arp_coa_vel)
    slant_cross_mag = np.linalg.norm(slant_cross)
    if slant_cross_mag < 1e-10:
        raise ValueError(
            "Degenerate geometry: range and velocity vectors are parallel "
            "at center of aperture. Cannot determine slant plane."
        )
    look = np.sign(np.dot(fpn, slant_cross))
    if look == 0:
        look = 1.0  # Default to right-looking when slant plane is orthogonal to fpn
    ipn = look * slant_cross
    ipn = ipn / np.linalg.norm(ipn)

    # Compute polar coordinates
    k_a, k_sf = _pfa_polar_coords(bi_pos, scp_origin, arp_coa, ipn, fpn)

    # RF to radial frequency conversion
    rf_to_rad = (2.0 / SPEED_OF_LIGHT) * k_sf * bi_freq_scale
    k_r0 = nbdata.SC0 * rf_to_rad
    k_r_ss = nbdata.SCSS * rf_to_rad

    # Compute inscribed rectangle bounds
    bw = k_r_ss * (phase_history.shape[0] - 1)
    k_v_bounds, k_u_bounds = _pfa_inscribed_rectangle_coords(k_a, k_r0, bw)

    # New grid coordinates
    new_v = np.linspace(k_v_bounds[0], k_v_bounds[1], phase_history.shape[0])
    new_u = np.linspace(k_u_bounds[0], k_u_bounds[1], phase_history.shape[1])

    if verbose:
        print(f"PFA: Interpolating to rectangular grid...")

    # Interpolate range (radial)
    data_block = _pfa_interp_range(phase_history, k_a, k_r0, k_r_ss, new_v)

    # Interpolate azimuth (cross-range)
    data_block = _pfa_interp_azimuth(data_block.T, k_a, new_v, new_u)

    if verbose:
        print(f"PFA: Performing 2D FFT...")

    # 2D FFT with zero-padding
    data_block = _pfa_fft_zeropad_1d(data_block.T, sample_rate)  # FFT V
    data_block = _pfa_fft_zeropad_1d(data_block.T, sample_rate)  # FFT U

    return data_block.T.astype(np.complex64)


__all__ = [
    "NarrowbandData",
    "pfa_mem",
]
