# -*- coding: utf-8 -*-
"""Tests for coordinate transformation functions."""
import numpy as np
import pytest
from grdl_sartoolbox.geometry.coordinates import (
    ecf_to_geodetic,
    geodetic_to_ecf,
    ecf_ned_rotation_matrix,
    ecf_to_ned,
    ned_to_ecf,
    ecf_to_geocentric,
    ric_ecf_matrix,
)


class TestECFGeodesicRoundtrip:
    """Test ECF <-> Geodetic roundtrip conversions."""

    def test_origin_point(self):
        """Test conversion at lat=0, lon=0."""
        lat, lon, alt = 0.0, 0.0, 0.0
        x, y, z = geodetic_to_ecf(lat, lon, alt)
        lat2, lon2, alt2 = ecf_to_geodetic(x, y, z)
        assert abs(float(lat2) - lat) < 1e-10
        assert abs(float(lon2) - lon) < 1e-10
        assert abs(float(alt2) - alt) < 1e-3

    def test_north_pole(self):
        lat, lon, alt = 90.0, 0.0, 0.0
        x, y, z = geodetic_to_ecf(lat, lon, alt)
        lat2, lon2, alt2 = ecf_to_geodetic(x, y, z)
        assert abs(float(lat2) - lat) < 1e-10
        assert abs(float(alt2) - alt) < 1e-3

    def test_south_pole(self):
        lat, lon, alt = -90.0, 0.0, 0.0
        x, y, z = geodetic_to_ecf(lat, lon, alt)
        lat2, lon2, alt2 = ecf_to_geodetic(x, y, z)
        assert abs(float(lat2) - lat) < 1e-10

    def test_arbitrary_point(self):
        lat, lon, alt = 38.8977, -77.0365, 100.0  # Washington DC
        x, y, z = geodetic_to_ecf(lat, lon, alt)
        lat2, lon2, alt2 = ecf_to_geodetic(x, y, z)
        assert abs(float(lat2) - lat) < 1e-8
        assert abs(float(lon2) - lon) < 1e-8
        assert abs(float(alt2) - alt) < 1e-3

    def test_vector_input(self):
        lats = np.array([0.0, 45.0, -30.0])
        lons = np.array([0.0, 90.0, -120.0])
        alts = np.array([0.0, 1000.0, 5000.0])
        x, y, z = geodetic_to_ecf(lats, lons, alts)
        lat2, lon2, alt2 = ecf_to_geodetic(x, y, z)
        np.testing.assert_allclose(lat2, lats, atol=1e-8)
        np.testing.assert_allclose(lon2, lons, atol=1e-8)
        np.testing.assert_allclose(alt2, alts, atol=1e-3)

    def test_array_3_input_ecf(self):
        """Test single (3,) vector input."""
        x, y, z = geodetic_to_ecf(38.0, -77.0, 0.0)
        ecf = np.array([float(x), float(y), float(z)])
        lat, lon, alt = ecf_to_geodetic(ecf)
        assert abs(float(lat) - 38.0) < 1e-8
        assert abs(float(lon) - (-77.0)) < 1e-8

    def test_array_3x1_input_ecf(self):
        """Test (3,1) column vector input."""
        x, y, z = geodetic_to_ecf(38.0, -77.0, 0.0)
        ecf = np.array([[float(x)], [float(y)], [float(z)]])
        lat, lon, alt = ecf_to_geodetic(ecf)
        assert abs(float(lat) - 38.0) < 1e-8


class TestNED:
    def test_ned_roundtrip(self):
        """ECF -> NED -> ECF roundtrip."""
        orp = np.array([4000e3, 3000e3, 4000e3])
        point_ecf = orp + np.array([100.0, 200.0, 300.0])

        ned = ecf_to_ned(point_ecf, orp, is_position=True)
        point_back = ned_to_ecf(ned, orp, is_position=True)

        np.testing.assert_allclose(point_back, point_ecf, atol=1e-6)

    def test_ned_rotation_orthogonal(self):
        """Rotation matrix should be orthogonal."""
        orp = np.array([4000e3, 3000e3, 4000e3])
        R = ecf_ned_rotation_matrix(orp)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)

    def test_ned_direction_vector(self):
        """Test with direction vector (no ORP offset)."""
        orp = np.array([4000e3, 3000e3, 4000e3])
        vec = np.array([1.0, 0.0, 0.0])
        ned = ecf_to_ned(vec, orp, is_position=False)
        ecf_back = ned_to_ecf(ned, orp, is_position=False)
        np.testing.assert_allclose(ecf_back, vec, atol=1e-10)


class TestGeocentric:
    def test_origin(self):
        ecf = np.array([6378137.0, 0.0, 0.0])  # On equator
        lat, lon, alt = ecf_to_geocentric(ecf)
        assert abs(lat) < 1e-10
        assert abs(lon) < 1e-10
        assert abs(alt) < 1.0  # Near zero altitude

    def test_north_pole(self):
        ecf = np.array([0.0, 0.0, 6356752.314245])
        lat, lon, alt = ecf_to_geocentric(ecf)
        assert abs(lat - 90.0) < 1e-6


class TestRIC:
    def test_ric_matrix_orthogonal(self):
        r_arp = np.array([7000e3, 0.0, 0.0])
        v_arp = np.array([0.0, 7500.0, 0.0])
        T = ric_ecf_matrix(r_arp, v_arp, 'ecf')
        # Columns should be orthonormal
        for i in range(3):
            assert abs(np.linalg.norm(T[:, i]) - 1.0) < 1e-10
        np.testing.assert_allclose(T[:, 0] @ T[:, 1], 0.0, atol=1e-10)
        np.testing.assert_allclose(T[:, 0] @ T[:, 2], 0.0, atol=1e-10)
        np.testing.assert_allclose(T[:, 1] @ T[:, 2], 0.0, atol=1e-10)

    def test_ric_radial_direction(self):
        r_arp = np.array([7000e3, 0.0, 0.0])
        v_arp = np.array([0.0, 7500.0, 0.0])
        T = ric_ecf_matrix(r_arp, v_arp, 'ecf')
        R_hat = T[:, 0]
        expected = np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(R_hat, expected, atol=1e-10)
