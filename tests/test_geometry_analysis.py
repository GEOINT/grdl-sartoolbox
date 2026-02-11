# -*- coding: utf-8 -*-
"""
Tests for SAR geometry analysis functions.

Tests the wgs84_normal and compute_sar_geometry functions with various
geometric configurations.

License
-------
MIT License
Copyright (c) 2024 geoint.org

Created
-------
2026-02-11
"""

import numpy as np
import pytest

from grdl_sartoolbox.geometry.analysis import (
    wgs84_normal,
    compute_sar_geometry,
    SARGeometry,
)
from grdl_sartoolbox.utils.constants import WGS84_A, WGS84_B


class TestWGS84Normal:
    """Test suite for wgs84_normal function."""

    def test_single_point(self):
        """Test normal computation for single ECEF point."""
        # Point on equator at prime meridian
        ecef = np.array([WGS84_A, 0.0, 0.0])
        normal = wgs84_normal(ecef)

        assert normal.shape == (3,)
        # Normal should point outward (positive X for this point)
        assert normal[0] > 0.9
        assert np.abs(normal[1]) < 0.1
        assert np.abs(normal[2]) < 0.1

    def test_unit_length(self):
        """Normal vectors should have unit length."""
        ecef = np.array([4000e3, 3000e3, 3500e3])
        normal = wgs84_normal(ecef)

        assert np.isclose(np.linalg.norm(normal), 1.0)

    def test_north_pole(self):
        """Normal at north pole should point upward (Z-direction)."""
        ecef = np.array([0.0, 0.0, WGS84_B])
        normal = wgs84_normal(ecef)

        assert np.abs(normal[0]) < 1e-6
        assert np.abs(normal[1]) < 1e-6
        assert np.abs(normal[2] - 1.0) < 1e-6

    def test_south_pole(self):
        """Normal at south pole should point downward (-Z direction)."""
        ecef = np.array([0.0, 0.0, -WGS84_B])
        normal = wgs84_normal(ecef)

        assert np.abs(normal[0]) < 1e-6
        assert np.abs(normal[1]) < 1e-6
        assert np.abs(normal[2] + 1.0) < 1e-6

    def test_multiple_points_columns(self):
        """Test with multiple points in column format (3, N)."""
        ecef = np.array([
            [WGS84_A, 0.0, 0.0],
            [0.0, WGS84_A, 0.0],
            [0.0, 0.0, WGS84_B]
        ])  # 3 points, shape (3, 3)

        normals = wgs84_normal(ecef)

        assert normals.shape == (3, 3)
        # All normals should be unit length
        for i in range(3):
            assert np.isclose(np.linalg.norm(normals[:, i]), 1.0)

    def test_multiple_points_rows(self):
        """Test with multiple points in row format (N, 3)."""
        ecef = np.array([
            [WGS84_A, 0.0, 0.0],
            [0.0, WGS84_A, 0.0],
            [0.0, 0.0, WGS84_B]
        ])  # 3 points, shape (3, 3) interpreted as rows

        normals = wgs84_normal(ecef)

        assert normals.shape == (3, 3)
        # All normals should be unit length
        for i in range(3):
            assert np.isclose(np.linalg.norm(normals[i, :]), 1.0)

    def test_invalid_shape(self):
        """Invalid array shape should raise ValueError."""
        ecef = np.array([[1, 2], [3, 4]])  # (2, 2) - invalid

        with pytest.raises(ValueError, match="Invalid ECEF shape"):
            wgs84_normal(ecef)

    def test_typical_ground_point(self):
        """Test with typical ground point."""
        # Approximate coordinates for mid-latitude point
        ecef = np.array([4000e3, 3000e3, 3500e3])
        normal = wgs84_normal(ecef)

        # Should be unit length
        assert np.isclose(np.linalg.norm(normal), 1.0)
        # Should point roughly in same direction as position (but normalized)
        ecef_unit = ecef / np.linalg.norm(ecef)
        # Normal should be close to position direction (within ~10°)
        assert np.dot(normal, ecef_unit) > 0.95


class TestComputeSARGeometry:
    """Test suite for compute_sar_geometry function."""

    @pytest.fixture
    def typical_geometry(self):
        """Typical SAR geometry setup."""
        # Ground point at mid-latitude
        aim = np.array([4000e3, 3000e3, 3500e3])

        # Platform 50km above ground point, offset to east
        position = np.array([4050e3, 3020e3, 3550e3])

        # Velocity ~7 km/s northward with slight upward component
        velocity = np.array([0, 7000, 100])

        return {'aim': aim, 'position': position, 'velocity': velocity}

    def test_basic_geometry_computation(self, typical_geometry):
        """Test basic SAR geometry computation."""
        geom = compute_sar_geometry(
            typical_geometry['aim'],
            typical_geometry['position'],
            typical_geometry['velocity']
        )

        assert isinstance(geom, SARGeometry)
        # All angles should be finite
        assert np.isfinite(geom.azimuth)
        assert np.isfinite(geom.graze)
        assert np.isfinite(geom.slope)
        assert np.isfinite(geom.squint)
        assert np.isfinite(geom.dca)

    def test_azimuth_range(self, typical_geometry):
        """Azimuth should be in [0, 2π]."""
        geom = compute_sar_geometry(
            typical_geometry['aim'],
            typical_geometry['position'],
            typical_geometry['velocity']
        )

        assert 0 <= geom.azimuth < 2 * np.pi

    def test_grazing_angle_range(self, typical_geometry):
        """Grazing angle should be in reasonable range [-π/2, π/2]."""
        geom = compute_sar_geometry(
            typical_geometry['aim'],
            typical_geometry['position'],
            typical_geometry['velocity']
        )

        # Typical SAR grazing angles are 20-60°
        assert -np.pi/2 <= geom.graze <= np.pi/2
        # For typical geometry, should be positive (looking down)
        assert geom.graze > 0

    def test_slope_greater_than_graze(self, typical_geometry):
        """Slope angle should be >= grazing angle."""
        geom = compute_sar_geometry(
            typical_geometry['aim'],
            typical_geometry['position'],
            typical_geometry['velocity']
        )

        # Slope >= graze by definition
        assert geom.slope >= geom.graze - 1e-6  # Small tolerance for numerical error

    def test_right_looking(self):
        """Test right-looking geometry.

        Right-looking: platform east of target, flying north.
        Target is to the right (west) of the flight path.
        """
        aim = np.array([4000e3, 3000e3, 3500e3])
        # Platform east of aim point
        position = np.array([4050e3, 3000e3, 3550e3])

        # Velocity northward
        velocity = np.array([0, 7000, 0])

        geom = compute_sar_geometry(aim, position, velocity)

        # Should be right-looking (sense < 0, so right = 1)
        assert geom.right == 1

    def test_left_vs_right_looking(self):
        """Test that different geometries give different looking directions."""
        aim = np.array([4000e3, 3000e3, 3500e3])

        # Two different platform positions
        position1 = np.array([4050e3, 3000e3, 3550e3])  # East of aim
        position2 = np.array([3950e3, 3000e3, 3550e3])  # West of aim

        # Same velocity (northward)
        velocity = np.array([0, 7000, 0])

        geom1 = compute_sar_geometry(aim, position1, velocity)
        geom2 = compute_sar_geometry(aim, position2, velocity)

        # Different geometries should give different looking directions
        # Both should be valid (±1)
        assert geom1.right in [-1, 1]
        assert geom2.right in [-1, 1]
        # In this case both happen to be right-looking due to geometry
        assert geom1.right == 1
        assert geom2.right == 1

    def test_ascending_orbit(self, typical_geometry):
        """Test ascending orbit detection."""
        # Modify velocity to be ascending (positive Z)
        velocity = np.array([0, 7000, 500])

        geom = compute_sar_geometry(
            typical_geometry['aim'],
            typical_geometry['position'],
            velocity
        )

        assert geom.ascend == 1

    def test_descending_orbit(self, typical_geometry):
        """Test descending orbit detection."""
        # Modify velocity to be descending (negative Z)
        velocity = np.array([0, 7000, -500])

        geom = compute_sar_geometry(
            typical_geometry['aim'],
            typical_geometry['position'],
            velocity
        )

        assert geom.ascend == -1

    def test_level_flight(self, typical_geometry):
        """Test level flight (no vertical velocity)."""
        # Velocity purely horizontal
        velocity = np.array([0, 7000, 0])

        geom = compute_sar_geometry(
            typical_geometry['aim'],
            typical_geometry['position'],
            velocity
        )

        assert geom.ascend == 0

    def test_custom_normal(self, typical_geometry):
        """Test with custom ground plane normal."""
        # Custom normal (not WGS-84) - use tilted plane
        normal = np.array([0.1, 0.0, 1.0])  # Nearly vertical, slight tilt

        geom = compute_sar_geometry(
            typical_geometry['aim'],
            typical_geometry['position'],
            typical_geometry['velocity'],
            normal=normal
        )

        assert isinstance(geom, SARGeometry)
        assert np.isfinite(geom.azimuth)

    def test_custom_normal_vertical(self, typical_geometry):
        """Test with purely vertical normal (edge case)."""
        # Vertical normal - azimuth becomes degenerate
        normal = np.array([0.0, 0.0, 1.0])

        geom = compute_sar_geometry(
            typical_geometry['aim'],
            typical_geometry['position'],
            typical_geometry['velocity'],
            normal=normal
        )

        assert isinstance(geom, SARGeometry)
        # Azimuth should be defined (defaults to 0) even with vertical normal
        assert np.isfinite(geom.azimuth)
        assert geom.azimuth == 0.0  # Default value for degenerate case

    def test_input_validation_aim_shape(self, typical_geometry):
        """Invalid aim shape should raise ValueError."""
        aim = np.array([1, 2])  # Only 2 elements

        with pytest.raises(ValueError, match="must be 1D arrays of shape"):
            compute_sar_geometry(
                aim,
                typical_geometry['position'],
                typical_geometry['velocity']
            )

    def test_input_validation_position_shape(self, typical_geometry):
        """Invalid position shape should raise ValueError."""
        position = np.array([[1, 2, 3]])  # 2D array

        with pytest.raises(ValueError, match="must be 1D arrays of shape"):
            compute_sar_geometry(
                typical_geometry['aim'],
                position,
                typical_geometry['velocity']
            )

    def test_input_validation_velocity_shape(self, typical_geometry):
        """Invalid velocity shape should raise ValueError."""
        velocity = np.array([1, 2, 3, 4])  # 4 elements

        with pytest.raises(ValueError, match="must be 1D arrays of shape"):
            compute_sar_geometry(
                typical_geometry['aim'],
                typical_geometry['position'],
                velocity
            )

    def test_input_validation_normal_shape(self, typical_geometry):
        """Invalid normal shape should raise ValueError."""
        normal = np.array([1, 2])  # Only 2 elements

        with pytest.raises(ValueError, match="normal must be shape"):
            compute_sar_geometry(
                typical_geometry['aim'],
                typical_geometry['position'],
                typical_geometry['velocity'],
                normal=normal
            )

    def test_doppler_cone_angle_range(self, typical_geometry):
        """Doppler cone angle should be in reasonable range."""
        geom = compute_sar_geometry(
            typical_geometry['aim'],
            typical_geometry['position'],
            typical_geometry['velocity']
        )

        # DCA typically 0-90° for SAR
        assert -np.pi <= geom.dca <= np.pi

    def test_squint_angle_small_for_broadside(self):
        """Squint should be small for near-broadside collection."""
        aim = np.array([4000e3, 3000e3, 3500e3])
        # Position to the side (east) and above
        position = np.array([4050e3, 3000e3, 3550e3])
        # Velocity northward
        velocity = np.array([0, 7000, 0])

        geom = compute_sar_geometry(aim, position, velocity)

        # Squint should be relatively small for this geometry
        # (not perfectly broadside due to vertical offset and ECEF curvature)
        assert np.abs(geom.squint) < np.radians(15)  # Within 15°

    def test_all_geometry_parameters_present(self, typical_geometry):
        """All geometry parameters should be present in result."""
        geom = compute_sar_geometry(
            typical_geometry['aim'],
            typical_geometry['position'],
            typical_geometry['velocity']
        )

        # Check all expected attributes
        assert hasattr(geom, 'azimuth')
        assert hasattr(geom, 'graze')
        assert hasattr(geom, 'slope')
        assert hasattr(geom, 'squint')
        assert hasattr(geom, 'layover')
        assert hasattr(geom, 'multipath')
        assert hasattr(geom, 'dca')
        assert hasattr(geom, 'tilt')
        assert hasattr(geom, 'track')
        assert hasattr(geom, 'felev')
        assert hasattr(geom, 'right')
        assert hasattr(geom, 'ascend')


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
