# -*- coding: utf-8 -*-
"""
Shapefile Writer - Write geospatial annotations in multiple formats.

Supports writing geographic shapes (points, polygons, lines) in
KML, GRFX (SOCET), and RVL (RemoteView) formats.

Attribution
-----------
Ported from MATLAB SAR Toolbox (https://github.com/ngageoint/MATLAB_SAR)
Original: Tim Cox, NGA

License
-------
MIT License
Copyright (c) 2024 geoint.org
"""

from typing import List, Tuple, Optional, Union
import os

from grdl_sartoolbox.io.kml import KMLWriter, get_kml_color


class ShapeWriter:
    """
    Multi-format shapefile writer.

    Supports KML, GRFX (SOCET GXP), and RVL (RemoteView) formats.

    Parameters
    ----------
    filename : str
        Output filename (extension determines format).
    format : str
        Output format: 'kml', 'grfx', or 'rvl'.
    name : str
        Document name.
    """

    def __init__(self, filename: str, fmt: str = 'kml', name: str = 'Untitled'):
        self.filename = filename
        self.fmt = fmt.lower()
        self.name = name
        self._fid = None
        self._kml_writer = None

        if self.fmt == 'kml':
            self._kml_writer = KMLWriter(filename, name=name)
        elif self.fmt == 'grfx':
            self._fid = open(filename, 'w')
            self._fid.write('<?xml version="1.0" ?>\n')
            self._fid.write('<Graphics appVersionMajor="2" appVersionMinor="3" '
                           'appVersionBuild="1"><GmGraphicsLayout m_has_changed="true" '
                           'm_manage_memory="true">\n')
            self._fid.write('<GmLayerList name="m_layer_list">\n')
            self._fid.write('<GmLayer m_layer_name="GRFXLayer" m_manage_memory="true" '
                           'm_refresh_indices="true" m_exclusive="false">\n')
            self._fid.write('<GrGmGraphicList name="m_graphic_list">\n\n')
        elif self.fmt == 'rvl':
            self._fid = open(filename, 'w')
            self._fid.write('<<\n')
            self._fid.write('Layer\t<<\n')
            self._fid.write('DisplayList\t[\n\n')
        else:
            raise ValueError(f"Unsupported format: {fmt}. Use 'kml', 'grfx', or 'rvl'")

    def add_point(
        self,
        name: str,
        lat: float,
        lon: float,
        alt: float = 0.0,
        color: str = 'red',
        description: str = ''
    ) -> None:
        """Add a point annotation."""
        if self.fmt == 'kml':
            self._kml_writer.add_placemark(
                name, lat, lon, alt, description, color
            )
        elif self.fmt == 'grfx':
            self._fid.write(f'<!-- Point: {name} -->\n')
            self._fid.write(f'<GrGmPoint lat="{lat}" lon="{lon}" />\n')
        elif self.fmt == 'rvl':
            self._fid.write(f'Point\t{lat}\t{lon}\t{name}\n')

    def add_polygon(
        self,
        name: str,
        coords: List[Tuple[float, float]],
        color: str = 'blue',
        transparency: float = 0.5,
        description: str = ''
    ) -> None:
        """Add a polygon annotation."""
        if self.fmt == 'kml':
            coords_3d = [(c[0], c[1], 0.0) if len(c) == 2 else c for c in coords]
            self._kml_writer.add_polygon(
                name, coords_3d, color, transparency, description
            )
        elif self.fmt == 'grfx':
            self._fid.write(f'<!-- Polygon: {name} -->\n')
            self._fid.write('<GrGmPolygon>\n')
            for lat, lon in coords:
                self._fid.write(f'  <vertex lat="{lat}" lon="{lon}" />\n')
            self._fid.write('</GrGmPolygon>\n')
        elif self.fmt == 'rvl':
            self._fid.write(f'Polygon\t{name}\n')
            for lat, lon in coords:
                self._fid.write(f'\t{lat}\t{lon}\n')
            self._fid.write('End\n')

    def close(self) -> None:
        """Close the file and write any footers."""
        if self.fmt == 'kml':
            self._kml_writer.close()
        elif self.fmt == 'grfx':
            self._fid.write('</GrGmGraphicList>\n')
            self._fid.write('</GmLayer>\n')
            self._fid.write('</GmLayerList>\n')
            self._fid.write('</GmGraphicsLayout>\n')
            self._fid.write('</Graphics>\n')
            self._fid.close()
        elif self.fmt == 'rvl':
            self._fid.write(']\n')
            self._fid.write('>>\n')
            self._fid.write('>>\n')
            self._fid.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def write_polygon_kml(
    filename: str,
    name: str,
    coords: List[Tuple[float, float]],
    color: str = 'blue',
    transparency: float = 0.5
) -> None:
    """
    Write a single polygon to a KML file.

    Convenience function for simple polygon output.

    Parameters
    ----------
    filename : str
        Output KML filename.
    name : str
        Polygon name.
    coords : list of tuple
        List of (lat, lon) tuples.
    color : str
        Fill color.
    transparency : float
        Fill transparency.
    """
    with ShapeWriter(filename, fmt='kml', name=name) as writer:
        writer.add_polygon(name, coords, color, transparency)


def write_point_kml(
    filename: str,
    name: str,
    lat: float,
    lon: float,
    alt: float = 0.0,
    color: str = 'red'
) -> None:
    """
    Write a single point to a KML file.

    Parameters
    ----------
    filename : str
        Output KML filename.
    name : str
        Placemark name.
    lat : float
        Latitude in degrees.
    lon : float
        Longitude in degrees.
    alt : float
        Altitude in meters.
    color : str
        Marker color.
    """
    with ShapeWriter(filename, fmt='kml', name=name) as writer:
        writer.add_point(name, lat, lon, alt, color)


__all__ = [
    "ShapeWriter",
    "write_polygon_kml",
    "write_point_kml",
]
