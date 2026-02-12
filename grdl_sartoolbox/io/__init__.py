# -*- coding: utf-8 -*-
"""
SAR I/O - File format readers and writers.

Provides readers for elevation data (DTED) and writers for
geospatial output formats (KML, shapefiles).

Note: Complex SAR format readers (SICD, CPHD, CRSD, SIDD) are
provided by the GRDL core library.

License
-------
MIT License
Copyright (c) 2024 geoint.org
"""

from grdl_sartoolbox.io.dted import read_dted, DTEDMetadata
from grdl_sartoolbox.io.kml import (
    KMLWriter,
    get_kml_color,
    get_kml_date_string,
)
from grdl_sartoolbox.io.shapefile import (
    ShapeWriter,
    write_polygon_kml,
    write_point_kml,
)

__all__ = [
    "read_dted",
    "DTEDMetadata",
    "KMLWriter",
    "get_kml_color",
    "get_kml_date_string",
    "ShapeWriter",
    "write_polygon_kml",
    "write_point_kml",
]
