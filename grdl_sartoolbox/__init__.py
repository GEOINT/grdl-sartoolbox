# -*- coding: utf-8 -*-
"""
grdl-sartoolbox - Python port of NGA MATLAB SAR Toolbox.

Pure-NumPy/SciPy reimplementations of SAR processing algorithms from
the MATLAB SAR Toolbox (https://github.com/ngageoint/MATLAB_SAR).

Modules
-------
processing : Core SAR processing algorithms (CCD, PFA, filtering, etc.)
geometry : Geometry analysis and projection utilities
utils : Constants and helper functions

License
-------
MIT License
Copyright (c) 2024 geoint.org

Created
-------
2026-02-11
"""

__version__ = "0.1.0"

from grdl_sartoolbox import processing, geometry, utils

__all__ = ["processing", "geometry", "utils", "__version__"]
