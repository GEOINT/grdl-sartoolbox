# -*- coding: utf-8 -*-
"""
KML Writer - Generate KML files for Google Earth visualization.

Provides utilities for creating KML files with placemarks, polygons,
and other geographic annotations from SAR analysis results.

Attribution
-----------
Ported from MATLAB SAR Toolbox (https://github.com/ngageoint/MATLAB_SAR)
Original: Tim Cox, NGA

License
-------
MIT License
Copyright (c) 2024 geoint.org
"""

from typing import Optional, Tuple, List, Union
from datetime import datetime
import xml.etree.ElementTree as ET
from xml.dom import minidom


def get_kml_color(
    color: Union[str, Tuple[float, float, float]],
    transparency: float = 1.0
) -> str:
    """
    Convert a color specification to KML format (TTBBGGRR hex string).

    Parameters
    ----------
    color : str or tuple of float
        Color name ('red', 'green', 'blue', etc.) or RGB tuple (0-1 range).
    transparency : float
        Transparency value (0=transparent, 1=opaque). Default 1.0.

    Returns
    -------
    str
        KML color string in TTBBGGRR format.
    """
    trans_int = int(transparency * 255)
    trans_hex = f'{trans_int:02x}'

    if isinstance(color, (list, tuple)):
        r = f'{int(color[0] * 255):02x}'
        g = f'{int(color[1] * 255):02x}'
        b = f'{int(color[2] * 255):02x}'
        return f'{trans_hex}{b}{g}{r}'

    color_map = {
        'red': '0000ff',
        'orange': '0066ff',
        'yellow': '00ffff',
        'green': '00ff00',
        'cyan': 'ffff00',
        'blue': 'ff0000',
        'magenta': 'ff00ff',
        'white': 'ffffff',
        'black': '000000',
    }

    color_str = color_map.get(color.lower(), 'ffffff')
    return f'{trans_hex}{color_str}'


def get_kml_date_string(dt: datetime) -> str:
    """
    Format a datetime object as a KML date string.

    Parameters
    ----------
    dt : datetime
        Datetime to format.

    Returns
    -------
    str
        KML-formatted date string (ISO 8601).
    """
    return dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'


class KMLWriter:
    """
    Writer for KML (Keyhole Markup Language) files.

    Supports creating KML documents with placemarks, polygons,
    lines, and ground overlays.

    Examples
    --------
    >>> writer = KMLWriter('output.kml', name='SAR Analysis')
    >>> writer.add_placemark('Target', lat=38.0, lon=-77.0, alt=0)
    >>> writer.add_polygon('Coverage', coords=[(38,−77), (38,−76), (39,−76), (39,−77)])
    >>> writer.close()
    """

    def __init__(self, filename: str, name: str = 'Untitled'):
        """
        Create a new KML file.

        Parameters
        ----------
        filename : str
            Output KML filename.
        name : str
            Document name displayed in Google Earth.
        """
        self.filename = filename
        self.kml = ET.Element('kml', xmlns='http://www.opengis.net/kml/2.2')
        self.document = ET.SubElement(self.kml, 'Document')
        name_elem = ET.SubElement(self.document, 'name')
        name_elem.text = name

    def add_placemark(
        self,
        name: str,
        lat: float,
        lon: float,
        alt: float = 0.0,
        description: str = '',
        color: str = 'red'
    ) -> None:
        """
        Add a point placemark to the KML document.

        Parameters
        ----------
        name : str
            Placemark name.
        lat : float
            Latitude in degrees.
        lon : float
            Longitude in degrees.
        alt : float
            Altitude in meters. Default 0.
        description : str
            Placemark description.
        color : str
            Marker color name.
        """
        pm = ET.SubElement(self.document, 'Placemark')
        name_elem = ET.SubElement(pm, 'name')
        name_elem.text = name

        if description:
            desc_elem = ET.SubElement(pm, 'description')
            desc_elem.text = description

        # Style
        style = ET.SubElement(pm, 'Style')
        icon_style = ET.SubElement(style, 'IconStyle')
        color_elem = ET.SubElement(icon_style, 'color')
        color_elem.text = get_kml_color(color)

        point = ET.SubElement(pm, 'Point')
        coords = ET.SubElement(point, 'coordinates')
        coords.text = f'{lon},{lat},{alt}'

    def add_polygon(
        self,
        name: str,
        coords: List[Tuple[float, float, float]],
        color: str = 'blue',
        transparency: float = 0.5,
        description: str = ''
    ) -> None:
        """
        Add a polygon to the KML document.

        Parameters
        ----------
        name : str
            Polygon name.
        coords : list of tuple
            List of (lat, lon, alt) tuples defining the polygon boundary.
        color : str or tuple
            Fill color.
        transparency : float
            Fill transparency (0-1).
        description : str
            Polygon description.
        """
        pm = ET.SubElement(self.document, 'Placemark')
        name_elem = ET.SubElement(pm, 'name')
        name_elem.text = name

        if description:
            desc_elem = ET.SubElement(pm, 'description')
            desc_elem.text = description

        # Style
        style = ET.SubElement(pm, 'Style')
        poly_style = ET.SubElement(style, 'PolyStyle')
        color_elem = ET.SubElement(poly_style, 'color')
        color_elem.text = get_kml_color(color, transparency)

        # Polygon geometry
        polygon = ET.SubElement(pm, 'Polygon')
        outer = ET.SubElement(polygon, 'outerBoundaryIs')
        ring = ET.SubElement(outer, 'LinearRing')
        coord_elem = ET.SubElement(ring, 'coordinates')

        coord_strings = []
        for c in coords:
            if len(c) == 3:
                coord_strings.append(f'{c[1]},{c[0]},{c[2]}')
            else:
                coord_strings.append(f'{c[1]},{c[0]},0')
        coord_elem.text = ' '.join(coord_strings)

    def add_line(
        self,
        name: str,
        coords: List[Tuple[float, float, float]],
        color: str = 'red',
        width: float = 2.0
    ) -> None:
        """
        Add a line string to the KML document.

        Parameters
        ----------
        name : str
            Line name.
        coords : list of tuple
            List of (lat, lon, alt) tuples.
        color : str or tuple
            Line color.
        width : float
            Line width in pixels.
        """
        pm = ET.SubElement(self.document, 'Placemark')
        name_elem = ET.SubElement(pm, 'name')
        name_elem.text = name

        style = ET.SubElement(pm, 'Style')
        line_style = ET.SubElement(style, 'LineStyle')
        color_elem = ET.SubElement(line_style, 'color')
        color_elem.text = get_kml_color(color)
        width_elem = ET.SubElement(line_style, 'width')
        width_elem.text = str(width)

        linestring = ET.SubElement(pm, 'LineString')
        coord_elem = ET.SubElement(linestring, 'coordinates')
        coord_strings = []
        for c in coords:
            if len(c) == 3:
                coord_strings.append(f'{c[1]},{c[0]},{c[2]}')
            else:
                coord_strings.append(f'{c[1]},{c[0]},0')
        coord_elem.text = ' '.join(coord_strings)

    def close(self) -> None:
        """Write the KML document to file."""
        rough_string = ET.tostring(self.kml, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        pretty = reparsed.toprettyxml(indent='  ', encoding='UTF-8')

        with open(self.filename, 'wb') as f:
            f.write(pretty)


__all__ = [
    "KMLWriter",
    "get_kml_color",
    "get_kml_date_string",
]
