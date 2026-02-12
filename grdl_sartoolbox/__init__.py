# -*- coding: utf-8 -*-
"""
grdl-sartoolbox - Python port of NGA MATLAB SAR Toolbox.

Pure-NumPy/SciPy reimplementations of SAR processing algorithms from
the MATLAB SAR Toolbox (https://github.com/ngageoint/MATLAB_SAR).

Modules
-------
processing : Core SAR processing algorithms (CCD, PFA, filtering, etc.)
geometry : Geometry analysis, coordinate transforms, and projections
visualization : Remap and display functions for SAR imagery
io : File format readers (DTED) and writers (KML, shapefile)
utils : Constants and helper functions

License
-------
MIT License
Copyright (c) 2024 geoint.org
"""

__version__ = "0.2.0"

from grdl_sartoolbox import processing, geometry, visualization, utils

__all__ = ["processing", "geometry", "visualization", "io", "utils", "__version__"]
