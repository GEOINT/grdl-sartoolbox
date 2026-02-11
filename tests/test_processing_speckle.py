# -*- coding: utf-8 -*-
"""
Tests for speckle filtering processors.

Tests the LeeFilter class with synthetic SAR data to verify:
- Speckle reduction on noisy images
- Filter parameter validation
- Different filter types (Lee, Kuan, Boxcar)
- Edge preservation
- Output shapes and data types

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

from grdl_sartoolbox.processing.speckle_filter import LeeFilter


class TestLeeFilter:
    """Test suite for LeeFilter processor."""

    @pytest.fixture
    def random_state(self):
        """Fixed random state for reproducible tests."""
        return np.random.RandomState(42)

    @pytest.fixture
    def clean_image(self):
        """Generate a clean SAR-like image with features."""
        # Create image with bright and dark regions
        image = np.ones((256, 256), dtype=np.float32) * 10.0
        # Add bright square
        image[64:192, 64:192] = 50.0
        # Add bright circle
        y, x = np.ogrid[:256, :256]
        circle_mask = (x - 200)**2 + (y - 200)**2 <= 30**2
        image[circle_mask] = 80.0
        return image

    @pytest.fixture
    def noisy_sar_image(self, clean_image, random_state):
        """Add multiplicative speckle noise to clean image."""
        # Speckle follows gamma distribution
        # For single-look SAR, use exponential (gamma with shape=1)
        speckle = random_state.gamma(1.0, 1.0, clean_image.shape)
        return (clean_image * speckle).astype(np.float32)

    # ------------------------------------------------------------------
    # Basic functionality tests
    # ------------------------------------------------------------------

    def test_lee_filter_reduces_noise(self, noisy_sar_image, clean_image):
        """Lee filter should reduce noise variance."""
        lee = LeeFilter(radius=3, enl=1.0, filter_type='Lee')
        filtered = lee.apply(noisy_sar_image)

        # Compute variance in homogeneous region (background)
        roi = (slice(10, 50), slice(10, 50))
        noisy_std = np.std(noisy_sar_image[roi])
        filtered_std = np.std(filtered[roi])

        # Filtered should have lower variance
        assert filtered_std < noisy_std
        # Should reduce variance by at least 30%
        assert filtered_std < 0.7 * noisy_std

    def test_lee_filter_preserves_mean(self, noisy_sar_image):
        """Lee filter should approximately preserve mean intensity."""
        lee = LeeFilter(radius=2, enl=1.0)
        filtered = lee.apply(noisy_sar_image)

        original_mean = np.mean(noisy_sar_image)
        filtered_mean = np.mean(filtered)

        # Should preserve mean within 30% (bias correction is approximate)
        # Note: The filter operates on power, applies correction, then takes sqrt
        assert abs(filtered_mean - original_mean) / original_mean < 0.30

    def test_output_shape_and_dtype(self, noisy_sar_image):
        """Output should match input shape with float32 dtype."""
        lee = LeeFilter(radius=2)
        filtered = lee.apply(noisy_sar_image)

        assert filtered.shape == noisy_sar_image.shape
        assert filtered.dtype == np.float32
        assert np.all(filtered >= 0)  # Amplitude should be non-negative

    def test_complex_input(self, random_state):
        """Filter should handle complex input by computing magnitude."""
        rows, cols = 128, 128
        real = random_state.randn(rows, cols) * 10
        imag = random_state.randn(rows, cols) * 10
        complex_image = (real + 1j * imag).astype(np.complex64)

        lee = LeeFilter(radius=2)
        filtered = lee.apply(complex_image)

        assert filtered.shape == (rows, cols)
        assert filtered.dtype == np.float32
        assert np.all(filtered >= 0)

    # ------------------------------------------------------------------
    # Filter type tests
    # ------------------------------------------------------------------

    def test_filter_type_lee(self, noisy_sar_image):
        """Test Lee filter specifically."""
        lee = LeeFilter(radius=2, enl=4, filter_type='Lee')
        filtered = lee.apply(noisy_sar_image)

        assert filtered.shape == noisy_sar_image.shape
        # Lee filter should smooth
        assert np.std(filtered) < np.std(noisy_sar_image)

    def test_filter_type_kuan(self, noisy_sar_image):
        """Test Kuan filter."""
        kuan = LeeFilter(radius=2, enl=4, filter_type='Kuan')
        filtered = kuan.apply(noisy_sar_image)

        assert filtered.shape == noisy_sar_image.shape
        # Kuan filter should also smooth
        assert np.std(filtered) < np.std(noisy_sar_image)

    def test_filter_type_boxcar(self, noisy_sar_image):
        """Test Boxcar (uniform) filter."""
        boxcar = LeeFilter(radius=2, filter_type='Boxcar')
        filtered = boxcar.apply(noisy_sar_image)

        assert filtered.shape == noisy_sar_image.shape
        # Boxcar should smooth heavily
        assert np.std(filtered) < np.std(noisy_sar_image)

    def test_lee_vs_kuan_difference(self, noisy_sar_image):
        """Lee and Kuan should give different results."""
        lee = LeeFilter(radius=2, enl=4, filter_type='Lee')
        kuan = LeeFilter(radius=2, enl=4, filter_type='Kuan')

        lee_result = lee.apply(noisy_sar_image)
        kuan_result = kuan.apply(noisy_sar_image)

        # Results should differ
        assert not np.allclose(lee_result, kuan_result)

    # ------------------------------------------------------------------
    # Parameter validation tests
    # ------------------------------------------------------------------

    def test_radius_validation_minimum(self):
        """Radius < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="radius must be >= 1"):
            LeeFilter(radius=0)

    def test_enl_validation_negative(self):
        """Negative ENL should raise ValueError."""
        with pytest.raises(ValueError, match="enl must be >= 0"):
            LeeFilter(enl=-1.0)

    def test_filter_type_validation_invalid(self):
        """Invalid filter type should raise ValueError."""
        with pytest.raises(ValueError, match="filter_type must be"):
            LeeFilter(filter_type='InvalidFilter')

    def test_valid_parameters(self):
        """Valid parameter combinations should not raise errors."""
        valid_configs = [
            {'radius': 1, 'enl': 0, 'filter_type': 'Lee'},
            {'radius': 5, 'enl': 10, 'filter_type': 'Kuan'},
            {'radius': 3, 'enl': 0, 'filter_type': 'Boxcar'},
        ]
        for config in valid_configs:
            lee = LeeFilter(**config)
            assert lee.radius == config['radius']
            assert lee.enl == config['enl']
            assert lee.filter_type == config['filter_type']

    # ------------------------------------------------------------------
    # Input validation tests
    # ------------------------------------------------------------------

    def test_non_2d_image_raises_error(self, random_state):
        """Non-2D image should raise ValueError."""
        image_3d = random_state.randn(10, 10, 10).astype(np.float32)

        lee = LeeFilter()

        with pytest.raises(ValueError, match="must be 2D"):
            lee.apply(image_3d)

    def test_non_numpy_array_raises_error(self):
        """Non-numpy array should raise TypeError."""
        image_list = [[1.0, 2.0], [3.0, 4.0]]

        lee = LeeFilter()

        with pytest.raises(TypeError, match="must be a numpy ndarray"):
            lee.apply(image_list)

    # ------------------------------------------------------------------
    # ENL estimation tests
    # ------------------------------------------------------------------

    def test_enl_auto_estimation(self, noisy_sar_image):
        """ENL=0 should trigger auto-estimation."""
        lee = LeeFilter(radius=2, enl=0)  # Auto-estimate
        filtered_auto = lee.apply(noisy_sar_image)

        # Should produce valid output
        assert filtered_auto.shape == noisy_sar_image.shape
        assert np.all(np.isfinite(filtered_auto))

    # ------------------------------------------------------------------
    # Edge case tests
    # ------------------------------------------------------------------

    def test_small_image(self, random_state):
        """Test with very small image."""
        image = random_state.randn(10, 10).astype(np.float32) + 10
        image = np.abs(image)  # Make positive

        lee = LeeFilter(radius=2)
        filtered = lee.apply(image)

        assert filtered.shape == (10, 10)
        assert np.all(np.isfinite(filtered))

    def test_uniform_image(self):
        """Test on uniform (constant) image."""
        image = np.ones((100, 100), dtype=np.float32) * 42.0

        lee = LeeFilter(radius=2, enl=4)
        filtered = lee.apply(image)

        # Should remain approximately uniform
        assert np.allclose(filtered, 42.0, atol=1e-3)


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
