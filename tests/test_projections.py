# -*- coding: utf-8 -*-
"""Tests for SAR projection functions."""
import numpy as np
import pytest
from grdl_sartoolbox.geometry.projections import (
    point_to_ground_plane,
    point_to_hae,
    point_to_hae_newton,
)
from grdl_sartoolbox.geometry.coordinates import geodetic_to_ecf


class TestPointToGroundPlane:
    def test_basic_projection(self):
        """Basic ground plane projection."""
        # SCP on Earth surface
        scp_x, scp_y, scp_z = geodetic_to_ecf(38.0, -77.0, 0.0)
        scp = np.array([float(scp_x), float(scp_y), float(scp_z)])

        # ARP 500km above SCP
        arp = scp + np.array([0.0, 0.0, 500e3])
        varp = np.array([7500.0, 0.0, 0.0])

        R = np.linalg.norm(arp - scp)
        Rdot = np.dot(varp, (scp - arp)) / R

        gpn = scp / np.linalg.norm(scp)  # Normal pointing outward

        gpp = point_to_ground_plane(
            np.array([R]), np.array([Rdot]),
            arp[:, np.newaxis], varp[:, np.newaxis],
            scp, gpn
        )
        assert gpp.shape == (3,)
        # Result should be near SCP
        dist = np.linalg.norm(gpp - scp)
        assert dist < 1e6  # Within reasonable distance

    def test_multi_point(self):
        """Multiple point projection."""
        scp_x, scp_y, scp_z = geodetic_to_ecf(38.0, -77.0, 0.0)
        scp = np.array([float(scp_x), float(scp_y), float(scp_z)])

        arp = scp + np.array([0.0, 0.0, 500e3])
        varp = np.array([7500.0, 0.0, 0.0])

        R = np.linalg.norm(arp - scp)
        Rdot = np.dot(varp, (scp - arp)) / R

        # Two points with same parameters
        r_arr = np.array([R, R * 1.001])
        rdot_arr = np.array([Rdot, Rdot])
        arp_2d = np.tile(arp[:, np.newaxis], (1, 2))
        varp_2d = np.tile(varp[:, np.newaxis], (1, 2))

        gpn = scp / np.linalg.norm(scp)

        gpp = point_to_ground_plane(r_arr, rdot_arr, arp_2d, varp_2d, scp, gpn)
        assert gpp.shape == (3, 2)


class TestPointToHAE:
    def test_basic_hae_projection(self):
        """Basic HAE surface projection."""
        scp_x, scp_y, scp_z = geodetic_to_ecf(38.0, -77.0, 100.0)
        scp = np.array([float(scp_x), float(scp_y), float(scp_z)])

        arp = scp + np.array([0.0, 100e3, 400e3])
        varp = np.array([7500.0, 0.0, 100.0])

        R = np.linalg.norm(arp - scp)
        Rdot = np.dot(varp, (scp - arp)) / R

        spp = point_to_hae(
            np.array([R]), np.array([Rdot]),
            arp, varp, scp, hae0=100.0
        )
        assert spp.shape == (3,)
        assert np.all(np.isfinite(spp))


class TestPointToHAENewton:
    def test_basic_newton(self):
        """Basic Newton HAE projection."""
        scp_x, scp_y, scp_z = geodetic_to_ecf(38.0, -77.0, 100.0)
        scp = np.array([float(scp_x), float(scp_y), float(scp_z)])

        arp = scp + np.array([0.0, 100e3, 400e3])
        varp = np.array([7500.0, 0.0, 100.0])

        R = np.linalg.norm(arp - scp)
        Rdot = np.dot(varp, (scp - arp)) / R

        spp = point_to_hae_newton(
            np.array([R]), np.array([Rdot]),
            arp, varp, scp, hae0=100.0
        )
        assert spp.shape == (3,)
        assert np.all(np.isfinite(spp))
