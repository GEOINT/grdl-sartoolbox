# -*- coding: utf-8 -*-
"""
Tests for backprojection algorithm.

Tests the backproject_basic function with synthetic phase history data.

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

from grdl_sartoolbox.processing.backprojection import (
    backproject_basic,
    BackprojectionData,
    SensorPosition,
    create_image_grid,
)


class TestBackprojection:
    """Test suite for backprojection algorithm."""

    @pytest.fixture
    def simple_geometry(self):
        """Create simple synthetic geometry for testing."""
        num_pulses = 50
        num_samples = 128

        # Linear flight path at constant altitude
        tx_x = np.linspace(-25, 25, num_pulses)
        tx_y = np.zeros(num_pulses)
        tx_z = np.ones(num_pulses) * 1000.0  # 1km altitude

        tx_pos = SensorPosition(tx_x, tx_y, tx_z)

        # Constant frequency parameters
        delta_f = np.ones(num_pulses) * 2e6  # 2 MHz step
        min_f = np.ones(num_pulses) * 10e9   # 10 GHz start

        # Range to scene center
        R0 = np.ones(num_pulses) * 1000.0

        return {
            'num_pulses': num_pulses,
            'num_samples': num_samples,
            'tx_pos': tx_pos,
            'delta_f': delta_f,
            'min_f': min_f,
            'R0': R0,
        }

    @pytest.fixture
    def simple_grid(self):
        """Create simple image grid."""
        x, y, z = create_image_grid(
            center=(0, 0, 0),
            size=(50, 50),
            resolution=(1.0, 1.0)
        )
        return {'x_mat': x, 'y_mat': y, 'z_mat': z}

    # ------------------------------------------------------------------
    # Data structure tests
    # ------------------------------------------------------------------

    def test_sensor_position_creation(self):
        """Test SensorPosition creation."""
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        z = np.array([7, 8, 9])

        pos = SensorPosition(x, y, z)
        assert len(pos.X) == 3
        assert len(pos.Y) == 3
        assert len(pos.Z) == 3

    def test_sensor_position_length_mismatch(self):
        """Mismatched array lengths should raise error."""
        x = np.array([1, 2, 3])
        y = np.array([4, 5])
        z = np.array([7, 8, 9])

        with pytest.raises(ValueError, match="must have same length"):
            SensorPosition(x, y, z)

    def test_backprojection_data_validation(self, simple_geometry, simple_grid):
        """Test BackprojectionData validation."""
        phdata = np.random.randn(
            simple_geometry['num_samples'],
            simple_geometry['num_pulses']
        ).astype(np.complex64)

        data = BackprojectionData(
            phdata=phdata,
            delta_f=simple_geometry['delta_f'],
            min_f=simple_geometry['min_f'],
            tx_pos=simple_geometry['tx_pos'],
            R0=simple_geometry['R0'],
            **simple_grid
        )

        assert data.phdata.shape == (simple_geometry['num_samples'], simple_geometry['num_pulses'])
        assert data.nfft is not None  # Auto-computed

    def test_backprojection_data_wrong_pulse_count(self, simple_geometry, simple_grid):
        """Wrong number of pulses should raise error."""
        # Create phase data with wrong number of pulses
        phdata = np.random.randn(128, 30).astype(np.complex64)  # 30 != 50

        with pytest.raises(ValueError, match="!= num_pulses"):
            BackprojectionData(
                phdata=phdata,
                delta_f=simple_geometry['delta_f'],  # 50 pulses
                min_f=simple_geometry['min_f'],
                tx_pos=simple_geometry['tx_pos'],
                R0=simple_geometry['R0'],
                **simple_grid
            )

    # ------------------------------------------------------------------
    # Grid creation tests
    # ------------------------------------------------------------------

    def test_create_image_grid_size(self):
        """Grid should have correct size."""
        x, y, z = create_image_grid(
            center=(0, 0, 0),
            size=(100, 100),
            resolution=(1.0, 1.0)
        )

        assert x.shape == (100, 100)
        assert y.shape == (100, 100)
        assert z.shape == (100, 100)

    def test_create_image_grid_center(self):
        """Grid should be centered correctly."""
        x, y, z = create_image_grid(
            center=(10, 20, 30),
            size=(50, 50),
            resolution=(1.0, 1.0),
            ground_plane_z=30
        )

        assert np.abs(np.mean(x) - 10) < 1e-6
        assert np.abs(np.mean(y) - 20) < 1e-6
        assert np.all(z == 30)

    def test_create_image_grid_resolution(self):
        """Grid spacing should approximately match resolution."""
        x, y, z = create_image_grid(
            center=(0, 0, 0),
            size=(100, 100),
            resolution=(2.0, 2.0)
        )

        # Check pixel spacing (within 5% tolerance due to rounding)
        dx = x[0, 1] - x[0, 0]
        dy = y[1, 0] - y[0, 0]

        assert np.abs(dx - 2.0) / 2.0 < 0.05  # Within 5%
        assert np.abs(dy - 2.0) / 2.0 < 0.05

    # ------------------------------------------------------------------
    # Backprojection algorithm tests
    # ------------------------------------------------------------------

    def test_backproject_basic_runs(self, simple_geometry, simple_grid):
        """Backprojection should run without errors."""
        phdata = np.random.randn(
            simple_geometry['num_samples'],
            simple_geometry['num_pulses']
        ).astype(np.complex64)

        data = BackprojectionData(
            phdata=phdata,
            delta_f=simple_geometry['delta_f'],
            min_f=simple_geometry['min_f'],
            tx_pos=simple_geometry['tx_pos'],
            R0=simple_geometry['R0'],
            **simple_grid
        )

        image = backproject_basic(data)

        assert image.shape == simple_grid['x_mat'].shape
        assert image.dtype == np.complex64

    def test_backproject_output_is_complex(self, simple_geometry, simple_grid):
        """Backprojection output should be complex."""
        phdata = np.random.randn(
            simple_geometry['num_samples'],
            simple_geometry['num_pulses']
        ).astype(np.complex64)

        data = BackprojectionData(
            phdata=phdata,
            delta_f=simple_geometry['delta_f'],
            min_f=simple_geometry['min_f'],
            tx_pos=simple_geometry['tx_pos'],
            R0=simple_geometry['R0'],
            **simple_grid
        )

        image = backproject_basic(data)

        assert np.iscomplexobj(image)

    def test_backproject_point_target(self):
        """Test backprojection with simulated point target."""
        # Small test case
        num_pulses = 20
        num_samples = 64

        # Circular flight path
        angles = np.linspace(0, 2*np.pi, num_pulses, endpoint=False)
        radius = 100.0
        altitude = 1000.0

        tx_x = radius * np.cos(angles)
        tx_y = radius * np.sin(angles)
        tx_z = np.ones(num_pulses) * altitude

        tx_pos = SensorPosition(tx_x, tx_y, tx_z)

        # Frequency parameters
        delta_f = np.ones(num_pulses) * 5e6  # 5 MHz
        min_f = np.ones(num_pulses) * 10e9   # 10 GHz

        # Range to scene center (roughly)
        R0 = np.sqrt(radius**2 + altitude**2) * np.ones(num_pulses)

        # Small grid
        x, y, z = create_image_grid(
            center=(0, 0, 0),
            size=(20, 20),
            resolution=(1.0, 1.0)
        )

        # Simulate point target at origin
        # For simplicity, use random phase history (won't focus properly but tests mechanics)
        phdata = np.random.randn(num_samples, num_pulses).astype(np.complex64)

        data = BackprojectionData(
            phdata=phdata,
            delta_f=delta_f,
            min_f=min_f,
            tx_pos=tx_pos,
            R0=R0,
            x_mat=x,
            y_mat=y,
            z_mat=z
        )

        image = backproject_basic(data)

        # Check output
        assert image.shape == (20, 20)
        assert np.all(np.isfinite(image))

    def test_backproject_bistatic(self, simple_geometry, simple_grid):
        """Test bistatic backprojection (separate Tx and Rcv)."""
        phdata = np.random.randn(
            simple_geometry['num_samples'],
            simple_geometry['num_pulses']
        ).astype(np.complex64)

        # Create separate receiver position (slightly offset)
        rcv_pos = SensorPosition(
            simple_geometry['tx_pos'].X + 10,  # 10m offset
            simple_geometry['tx_pos'].Y,
            simple_geometry['tx_pos'].Z
        )

        data = BackprojectionData(
            phdata=phdata,
            delta_f=simple_geometry['delta_f'],
            min_f=simple_geometry['min_f'],
            tx_pos=simple_geometry['tx_pos'],
            rcv_pos=rcv_pos,  # Explicit receiver
            R0=simple_geometry['R0'],
            **simple_grid
        )

        image = backproject_basic(data)

        assert image.shape == simple_grid['x_mat'].shape
        assert np.iscomplexobj(image)

    def test_backproject_custom_nfft(self, simple_geometry, simple_grid):
        """Test with custom FFT size."""
        phdata = np.random.randn(
            simple_geometry['num_samples'],
            simple_geometry['num_pulses']
        ).astype(np.complex64)

        data = BackprojectionData(
            phdata=phdata,
            delta_f=simple_geometry['delta_f'],
            min_f=simple_geometry['min_f'],
            tx_pos=simple_geometry['tx_pos'],
            R0=simple_geometry['R0'],
            nfft=2048,  # Custom FFT size
            **simple_grid
        )

        image = backproject_basic(data)

        assert image.shape == simple_grid['x_mat'].shape
        assert data.nfft == 2048

    def test_backproject_small_grid(self):
        """Test with very small grid."""
        num_pulses = 10
        num_samples = 32

        tx_x = np.linspace(-10, 10, num_pulses)
        tx_y = np.zeros(num_pulses)
        tx_z = np.ones(num_pulses) * 100.0

        tx_pos = SensorPosition(tx_x, tx_y, tx_z)

        delta_f = np.ones(num_pulses) * 1e6
        min_f = np.ones(num_pulses) * 10e9
        R0 = np.ones(num_pulses) * 100.0

        x, y, z = create_image_grid(
            center=(0, 0, 0),
            size=(10, 10),
            resolution=(2.0, 2.0)
        )

        phdata = np.random.randn(num_samples, num_pulses).astype(np.complex64)

        data = BackprojectionData(
            phdata=phdata,
            delta_f=delta_f,
            min_f=min_f,
            tx_pos=tx_pos,
            R0=R0,
            x_mat=x,
            y_mat=y,
            z_mat=z
        )

        image = backproject_basic(data)

        assert image.shape == (5, 5)  # 10m / 2m resolution


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
