# -*- coding: utf-8 -*-
"""
Amplitude Change Detection and Image Registration.

Implements:
- Amplitude Change Detection (ACD) RGB visualization
- Subpixel image registration via DFT cross-correlation

Attribution
-----------
Ported from MATLAB SAR Toolbox (https://github.com/ngageoint/MATLAB_SAR)
Original: Wade Schwartzkopf (NGA/IDT), Manuel Guizar-Sicairos et al.

License
-------
MIT License
Copyright (c) 2024 geoint.org
"""

from typing import Tuple, Optional
import numpy as np


def dft_registration(
    buf1ft: np.ndarray,
    buf2ft: np.ndarray,
    usfac: int = 1
) -> Tuple[float, float, float, float]:
    """
    Subpixel image registration via DFT cross-correlation.

    Efficient subpixel image registration by cross-correlation.
    Based on Manuel Guizar-Sicairos, Samuel T. Thurman, and James R.
    Fienup, "Efficient subpixel image registration algorithms,"
    Optics Letters 33, 156-158 (2008).

    Parameters
    ----------
    buf1ft : np.ndarray
        FFT of reference image.
    buf2ft : np.ndarray
        FFT of image to register.
    usfac : int
        Upsampling factor. Images are registered to within
        1/usfac of a pixel. Default 1 (whole pixel).

    Returns
    -------
    error : float
        Translation invariant normalized RMS error.
    diffphase : float
        Global phase difference between images.
    row_shift : float
        Row shift (pixels).
    col_shift : float
        Column shift (pixels).
    """
    nr, nc = buf1ft.shape

    if usfac == 0:
        # Simple cross-correlation (no registration)
        CCmax = np.sum(buf1ft * np.conj(buf2ft))
        rfzero = np.sum(np.abs(buf1ft) ** 2)
        rgzero = np.sum(np.abs(buf2ft) ** 2)
        error = 1.0 - np.abs(CCmax) ** 2 / (rgzero * rfzero)
        diffphase = np.angle(CCmax)
        return float(error), float(diffphase), 0.0, 0.0

    # Whole-pixel shift via cross-correlation
    CC = np.fft.ifft2(buf1ft * np.conj(buf2ft))
    max_idx = np.unravel_index(np.argmax(np.abs(CC)), CC.shape)
    CCmax = CC[max_idx]

    rloc, cloc = float(max_idx[0]), float(max_idx[1])

    # Handle wrap-around
    if rloc > nr // 2:
        rloc -= nr
    if cloc > nc // 2:
        cloc -= nc

    if usfac == 1:
        rfzero = np.sum(np.abs(buf1ft) ** 2) / (nr * nc)
        rgzero = np.sum(np.abs(buf2ft) ** 2) / (nr * nc)
        error = 1.0 - np.abs(CCmax) ** 2 / (rgzero * rfzero * nr * nc)
        diffphase = np.angle(CCmax)
        return float(error), float(diffphase), rloc, cloc

    # Refine estimate with matrix-multiply DFT
    # Initial estimate from whole-pixel shift
    row_shift = rloc
    col_shift = cloc

    # Refine to within 2 pixels using upsampled DFT
    # Then refine to within 1/usfac pixels
    dftshift = np.fix(np.ceil(usfac * 1.5))

    # Matrix multiply DFT around current shift estimate
    CC = _dftups(
        buf1ft * np.conj(buf2ft),
        int(np.ceil(usfac * 1.5)),
        int(np.ceil(usfac * 1.5)),
        usfac,
        dftshift - row_shift * usfac,
        dftshift - col_shift * usfac
    ) / (nr * nc * usfac ** 2)

    max_idx = np.unravel_index(np.argmax(np.abs(CC)), CC.shape)
    CCmax = CC[max_idx]

    rloc, cloc = float(max_idx[0]), float(max_idx[1])
    rloc -= dftshift
    cloc -= dftshift

    row_shift += rloc / usfac
    col_shift += cloc / usfac

    # Compute error
    rfzero = np.sum(np.abs(buf1ft) ** 2) / (nr * nc)
    rgzero = np.sum(np.abs(buf2ft) ** 2) / (nr * nc)
    error = 1.0 - np.abs(CCmax) ** 2 / (rgzero * rfzero)
    error = np.abs(error)
    diffphase = np.angle(CCmax)

    return float(error), float(diffphase), float(row_shift), float(col_shift)


def dft_register_image(
    buf1ft: np.ndarray,
    buf2ft: np.ndarray,
    usfac: int = 1
) -> np.ndarray:
    """
    Register and return the shifted second image.

    Parameters
    ----------
    buf1ft : np.ndarray
        FFT of reference image.
    buf2ft : np.ndarray
        FFT of image to register.
    usfac : int
        Upsampling factor.

    Returns
    -------
    np.ndarray
        Registered version of second image (spatial domain).
    """
    error, diffphase, row_shift, col_shift = dft_registration(buf1ft, buf2ft, usfac)
    nr, nc = buf2ft.shape

    # Apply shift in frequency domain
    Nr = np.fft.ifftshift(np.arange(-np.fix(nr / 2), np.ceil(nr / 2)))
    Nc = np.fft.ifftshift(np.arange(-np.fix(nc / 2), np.ceil(nc / 2)))
    Nc, Nr = np.meshgrid(Nc, Nr)

    greg = buf2ft * np.exp(
        1j * 2 * np.pi * (-row_shift * Nr / nr - col_shift * Nc / nc)
    )
    greg *= np.exp(1j * diffphase)

    return np.fft.ifft2(greg)


def _dftups(
    inp: np.ndarray,
    nor: int,
    noc: int,
    usfac: int = 1,
    roff: float = 0,
    coff: float = 0
) -> np.ndarray:
    """
    Upsampled DFT by matrix multiplies.

    Parameters
    ----------
    inp : np.ndarray
        Input data (2D).
    nor, noc : int
        Number of output rows and columns.
    usfac : int
        Upsampling factor.
    roff, coff : float
        Row and column offsets.
    """
    nr, nc = inp.shape

    # Compute kernels
    kernc = np.exp(
        (-1j * 2 * np.pi / (nc * usfac)) *
        np.outer(
            np.fft.ifftshift(np.arange(nc)) - np.floor(nc / 2),
            np.arange(noc) - coff
        )
    )
    kernr = np.exp(
        (-1j * 2 * np.pi / (nr * usfac)) *
        np.outer(
            np.arange(nor) - roff,
            np.fft.ifftshift(np.arange(nr)) - np.floor(nr / 2)
        )
    )

    return kernr @ inp @ kernc


def acd_rgb(
    reference_image: np.ndarray,
    match_image: np.ndarray,
    reference_color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    match_color: Tuple[float, float, float] = (0.0, 1.0, 1.0)
) -> np.ndarray:
    """
    Create RGB amplitude change detection visualization.

    Assigns reference and match images to different color channels,
    producing a color composite where changes appear as color differences.

    Parameters
    ----------
    reference_image : np.ndarray
        First SAR image (complex or real).
    match_image : np.ndarray
        Second SAR image (complex or real).
    reference_color : tuple
        RGB color for reference image (default red).
    match_color : tuple
        RGB color for match image (default cyan).

    Returns
    -------
    np.ndarray
        RGB image, shape (rows, cols, 3), dtype float32.
    """
    amp_ref = np.abs(reference_image).astype(np.float64)
    amp_match = np.abs(match_image).astype(np.float64)

    # Normalize to [0, 1]
    max_val = max(amp_ref.max(), amp_match.max())
    if max_val > 0:
        amp_ref /= max_val
        amp_match /= max_val

    rows, cols = amp_ref.shape
    rgb = np.zeros((rows, cols, 3), dtype=np.float64)

    for c in range(3):
        rgb[:, :, c] = (
            amp_ref * reference_color[c] +
            amp_match * match_color[c]
        )

    # Clip to [0, 1]
    rgb = np.clip(rgb, 0.0, 1.0)

    return rgb.astype(np.float32)


__all__ = [
    "dft_registration",
    "dft_register_image",
    "acd_rgb",
]
