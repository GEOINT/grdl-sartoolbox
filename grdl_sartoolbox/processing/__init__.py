# -*- coding: utf-8 -*-
"""
SAR Processing Algorithms - Core SAR image processing algorithms.

This module contains Python ports of key SAR processing algorithms from
the MATLAB SAR Toolbox, including:

- Coherent Change Detection (CCD, SCCM, noise-aware CCD)
- Amplitude Change Detection (ACD)
- Polar Format Algorithm (PFA)
- Backprojection
- SAR-specific filtering (Lee, Frost, Kuan, apodization)
- Color Sub-aperture Image (CSI)
- Phase Difference Visualization (PDV)
- SICD normalization (deskew, deweight)
- RGIQE (Radar Generalized Image Quality Equation)
- IFP utilities (resolution, deskew, PFA inverse)
- RCS computation
- Signal analysis (STFT, reramp)

License
-------
MIT License
Copyright (c) 2024 geoint.org
"""

# CCD
from grdl_sartoolbox.processing.ccd import (
    CoherentChangeDetection,
    ccd_mem,
    ccd_noise_mem,
    sccm,
    ccd_mem_angle,
)

# Speckle filtering
from grdl_sartoolbox.processing.speckle_filter import LeeFilter

# Image formation
from grdl_sartoolbox.processing.backprojection import (
    backproject_basic,
    BackprojectionData,
    SensorPosition,
    create_image_grid,
)
from grdl_sartoolbox.processing.pfa import (
    pfa_mem,
    NarrowbandData,
)

# ACD
from grdl_sartoolbox.processing.acd import (
    dft_registration,
    dft_register_image,
    acd_rgb,
)

# CSI & PDV
from grdl_sartoolbox.processing.csi import csi_mem
from grdl_sartoolbox.processing.pdv import pdv_mem

# Filtering extras
from grdl_sartoolbox.processing.filtering import apodize_2d, upsample_image

# Normalization
from grdl_sartoolbox.processing.normalize import (
    normalize_complex,
    deskew_mem,
    deweight_mem,
    estimate_weighting,
    is_normalized,
    sicd_weight_to_fun,
)

# RGIQE
from grdl_sartoolbox.processing.rgiqe import (
    RGIQEResult,
    compute_rgiqe,
)

# IFP utilities
from grdl_sartoolbox.processing.ifp_utils import (
    ResolutionExtent,
    pulse_info_to_resolution_extent,
    deskew_rvp,
    pfa_inverse,
)

# RCS
from grdl_sartoolbox.processing.rcs import compute_rcs, compute_rcs_db

# Signal analysis
from grdl_sartoolbox.processing.signal_analysis import stft, reramp

__all__ = [
    # CCD
    "CoherentChangeDetection",
    "ccd_mem",
    "ccd_noise_mem",
    "sccm",
    "ccd_mem_angle",
    # Speckle
    "LeeFilter",
    # Image formation
    "backproject_basic",
    "BackprojectionData",
    "SensorPosition",
    "create_image_grid",
    "pfa_mem",
    "NarrowbandData",
    # ACD
    "dft_registration",
    "dft_register_image",
    "acd_rgb",
    # CSI & PDV
    "csi_mem",
    "pdv_mem",
    # Filtering
    "apodize_2d",
    "upsample_image",
    # Normalization
    "normalize_complex",
    "deskew_mem",
    "deweight_mem",
    "estimate_weighting",
    "is_normalized",
    "sicd_weight_to_fun",
    # RGIQE
    "RGIQEResult",
    "compute_rgiqe",
    # IFP
    "ResolutionExtent",
    "pulse_info_to_resolution_extent",
    "deskew_rvp",
    "pfa_inverse",
    # RCS
    "compute_rcs",
    "compute_rcs_db",
    # Signal analysis
    "stft",
    "reramp",
]
