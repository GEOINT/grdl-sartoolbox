# -*- coding: utf-8 -*-
"""
SAR Processing Algorithms - Core SAR image processing algorithms.

This module contains Python ports of key SAR processing algorithms from
the MATLAB SAR Toolbox, including:

- Coherent Change Detection (CCD)
- Polar Format Algorithm (PFA)
- Backprojection
- SAR-specific filtering (Lee, Frost, Kuan)

All processors inherit from grdl.image_processing.base.ImageProcessor and
follow the GRDL processor pattern with versioning, tags, and annotated
parameters.

License
-------
MIT License
Copyright (c) 2024 geoint.org

Created
-------
2026-02-11
"""

# Import processors
from grdl_sartoolbox.processing.ccd import CoherentChangeDetection
from grdl_sartoolbox.processing.speckle_filter import LeeFilter
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

__all__ = [
    "CoherentChangeDetection",
    "LeeFilter",
    "backproject_basic",
    "BackprojectionData",
    "SensorPosition",
    "create_image_grid",
    "pfa_mem",
    "NarrowbandData",
]
