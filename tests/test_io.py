# -*- coding: utf-8 -*-
"""Tests for IO modules (DTED, KML, Shapefile)."""
import os
import tempfile
import numpy as np
import pytest
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
from datetime import datetime


class TestKMLColor:
    def test_red(self):
        color = get_kml_color('red')
        assert len(color) == 8
        assert color.startswith('ff')  # Full opacity
        assert color == 'ff0000ff'

    def test_rgb_tuple(self):
        color = get_kml_color((1.0, 0.0, 0.0), 0.5)
        assert len(color) == 8
        # 50% transparency
        assert color[:2] == '7f' or color[:2] == '80'

    def test_transparency(self):
        color = get_kml_color('blue', 0.0)
        assert color[:2] == '00'  # Fully transparent


class TestKMLDateString:
    def test_format(self):
        dt = datetime(2024, 1, 15, 12, 30, 45, 123456)
        s = get_kml_date_string(dt)
        assert '2024-01-15T12:30:45' in s
        assert s.endswith('Z')


class TestKMLWriter:
    def test_write_kml(self):
        with tempfile.NamedTemporaryFile(suffix='.kml', delete=False) as f:
            fname = f.name

        try:
            writer = KMLWriter(fname, name='Test')
            writer.add_placemark('Point1', lat=38.0, lon=-77.0)
            writer.add_polygon('Area1', coords=[(38, -77, 0), (38, -76, 0), (39, -76, 0)])
            writer.add_line('Line1', coords=[(38, -77), (39, -76)])
            writer.close()

            assert os.path.exists(fname)
            with open(fname, 'rb') as f:
                content = f.read().decode('utf-8')
            assert 'Point1' in content
            assert 'Area1' in content
            assert 'kml' in content
        finally:
            os.unlink(fname)


class TestShapeWriter:
    def test_kml_format(self):
        with tempfile.NamedTemporaryFile(suffix='.kml', delete=False) as f:
            fname = f.name

        try:
            with ShapeWriter(fname, fmt='kml', name='Test') as w:
                w.add_point('P1', 38.0, -77.0)
                w.add_polygon('Poly1', [(38, -77), (38, -76), (39, -76)])

            assert os.path.exists(fname)
        finally:
            os.unlink(fname)

    def test_grfx_format(self):
        with tempfile.NamedTemporaryFile(suffix='.grfx', delete=False) as f:
            fname = f.name

        try:
            with ShapeWriter(fname, fmt='grfx', name='Test') as w:
                w.add_point('P1', 38.0, -77.0)
                w.add_polygon('Poly1', [(38, -77), (38, -76)])

            with open(fname) as f:
                content = f.read()
            assert 'GrGmGraphicList' in content
        finally:
            os.unlink(fname)

    def test_rvl_format(self):
        with tempfile.NamedTemporaryFile(suffix='.rvl', delete=False) as f:
            fname = f.name

        try:
            with ShapeWriter(fname, fmt='rvl', name='Test') as w:
                w.add_point('P1', 38.0, -77.0)

            with open(fname) as f:
                content = f.read()
            assert 'Layer' in content
        finally:
            os.unlink(fname)

    def test_unsupported_format(self):
        with pytest.raises(ValueError):
            ShapeWriter('test.xyz', fmt='xyz')


class TestConvenienceFunctions:
    def test_write_polygon_kml(self):
        with tempfile.NamedTemporaryFile(suffix='.kml', delete=False) as f:
            fname = f.name

        try:
            write_polygon_kml(fname, 'Test', [(38, -77), (38, -76), (39, -76)])
            assert os.path.exists(fname)
        finally:
            os.unlink(fname)

    def test_write_point_kml(self):
        with tempfile.NamedTemporaryFile(suffix='.kml', delete=False) as f:
            fname = f.name

        try:
            write_point_kml(fname, 'Test', 38.0, -77.0)
            assert os.path.exists(fname)
        finally:
            os.unlink(fname)
