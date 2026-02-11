# -*- coding: utf-8 -*-
"""
Tests for Coherent Change Detection (CCD) processor.

Tests the CoherentChangeDetection class with synthetic data to verify:
- Identical images produce high coherence (~1.0)
- Uncorrelated images produce low coherence (~0.0)
- Parameter validation
- Edge cases (small images, large windows)
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

from grdl_sartoolbox.processing.ccd import CoherentChangeDetection


class TestCoherentChangeDetection:
    """Test suite for CoherentChangeDetection processor."""

    @pytest.fixture
    def random_state(self):
        """Fixed random state for reproducible tests."""
        return np.random.RandomState(42)

    @pytest.fixture
    def synthetic_image(self, random_state):
        """Generate a synthetic complex SAR image."""
        rows, cols = 512, 512
        real = random_state.randn(rows, cols)
        imag = random_state.randn(rows, cols)
        return (real + 1j * imag).astype(np.complex64)

    # ------------------------------------------------------------------
    # Basic functionality tests
    # ------------------------------------------------------------------

    def test_identical_images_high_coherence(self, synthetic_image):
        """CCD on identical images should produce very high coherence (~1.0)."""
        image1 = synthetic_image
        image2 = image1.copy()

        ccd = CoherentChangeDetection(window_size=7)
        coherence = ccd.apply(image1, image2)

        assert coherence.shape == image1.shape
        assert coherence.dtype == np.float32
        assert np.all(coherence >= 0.0) and np.all(coherence <= 1.0)
        # Mean coherence should be very close to 1.0 for identical images
        assert np.mean(coherence) > 0.98
        # Most pixels should have coherence > 0.95
        assert np.sum(coherence > 0.95) / coherence.size > 0.95

    def test_uncorrelated_images_low_coherence(self, random_state):
        """CCD on uncorrelated images should produce low coherence (~0.0)."""
        rows, cols = 512, 512

        # Generate two independent random complex images
        image1 = (
            random_state.randn(rows, cols) + 1j * random_state.randn(rows, cols)
        ).astype(np.complex64)
        image2 = (
            random_state.randn(rows, cols) + 1j * random_state.randn(rows, cols)
        ).astype(np.complex64)

        ccd = CoherentChangeDetection(window_size=7)
        coherence = ccd.apply(image1, image2)

        assert coherence.shape == image1.shape
        assert coherence.dtype == np.float32
        # Mean coherence should be low for uncorrelated images
        # (typically < 0.3 for window_size=7)
        assert np.mean(coherence) < 0.35

    def test_partial_change(self, synthetic_image, random_state):
        """CCD should detect partial changes in an image."""
        image1 = synthetic_image.copy()
        image2 = image1.copy()

        # Modify a region in image2 to simulate change
        rows, cols = image1.shape
        change_region = (
            slice(rows//4, 3*rows//4),
            slice(cols//4, 3*cols//4)
        )
        image2[change_region] = (
            random_state.randn(rows//2, cols//2) +
            1j * random_state.randn(rows//2, cols//2)
        ).astype(np.complex64)

        ccd = CoherentChangeDetection(window_size=11)
        coherence = ccd.apply(image1, image2)

        # Unchanged regions should have high coherence
        unchanged_region = (slice(0, rows//8), slice(0, cols//8))
        assert np.mean(coherence[unchanged_region]) > 0.9

        # Changed regions should have low coherence
        changed_region = (slice(rows//2, rows//2 + 50), slice(cols//2, cols//2 + 50))
        assert np.mean(coherence[changed_region]) < 0.5

    # ------------------------------------------------------------------
    # apply_with_phase tests
    # ------------------------------------------------------------------

    def test_apply_with_phase(self, synthetic_image):
        """Test that apply_with_phase returns both coherence and phase."""
        image1 = synthetic_image
        image2 = image1.copy()

        ccd = CoherentChangeDetection(window_size=7)
        coherence, phase = ccd.apply_with_phase(image1, image2)

        # Check shapes and dtypes
        assert coherence.shape == image1.shape
        assert phase.shape == image1.shape
        assert coherence.dtype == np.float32
        assert phase.dtype == np.float32

        # Coherence bounds
        assert np.all(coherence >= 0.0) and np.all(coherence <= 1.0)

        # Phase bounds (should be in [-π, π])
        assert np.all(phase >= -np.pi) and np.all(phase <= np.pi)

        # For identical images, phase should be near zero
        assert np.abs(np.mean(phase)) < 0.1

    # ------------------------------------------------------------------
    # Parameter validation tests
    # ------------------------------------------------------------------

    def test_window_size_validation_minimum(self):
        """Window size < 3 should raise ValueError."""
        with pytest.raises(ValueError, match="window_size must be >= 3"):
            CoherentChangeDetection(window_size=1)

    def test_window_size_validation_even(self):
        """Even window size should raise ValueError."""
        with pytest.raises(ValueError, match="window_size must be odd"):
            CoherentChangeDetection(window_size=8)

    def test_window_size_valid_values(self):
        """Valid window sizes should not raise errors."""
        valid_sizes = [3, 5, 7, 9, 11, 21, 31]
        for size in valid_sizes:
            ccd = CoherentChangeDetection(window_size=size)
            assert ccd.window_size == size

    # ------------------------------------------------------------------
    # Input validation tests
    # ------------------------------------------------------------------

    def test_non_complex_image_raises_error(self, random_state):
        """Non-complex image should raise TypeError."""
        real_image = random_state.randn(100, 100).astype(np.float32)
        complex_image = (real_image + 1j * real_image).astype(np.complex64)

        ccd = CoherentChangeDetection()

        with pytest.raises(TypeError, match="must be complex-valued"):
            ccd.apply(real_image, complex_image)

        with pytest.raises(TypeError, match="must be complex-valued"):
            ccd.apply(complex_image, real_image)

    def test_non_2d_image_raises_error(self, random_state):
        """Non-2D image should raise ValueError."""
        image_3d = random_state.randn(10, 10, 10).astype(np.complex64)
        image_2d = random_state.randn(10, 10).astype(np.complex64)

        ccd = CoherentChangeDetection()

        with pytest.raises(ValueError, match="must be 2D"):
            ccd.apply(image_3d, image_2d)

    def test_mismatched_shapes_raises_error(self, random_state):
        """Images with different shapes should raise ValueError."""
        image1 = random_state.randn(100, 100).astype(np.complex64)
        image2 = random_state.randn(50, 50).astype(np.complex64)

        ccd = CoherentChangeDetection()

        with pytest.raises(ValueError, match="must have same shape"):
            ccd.apply(image1, image2)

    def test_non_numpy_array_raises_error(self):
        """Non-numpy array should raise TypeError."""
        image_list = [[1+1j, 2+2j], [3+3j, 4+4j]]
        image_array = np.array(image_list, dtype=np.complex64)

        ccd = CoherentChangeDetection()

        with pytest.raises(TypeError, match="must be a numpy ndarray"):
            ccd.apply(image_list, image_array)

    # ------------------------------------------------------------------
    # Edge case tests
    # ------------------------------------------------------------------

    def test_small_image(self, random_state):
        """Test with very small image (smaller than window)."""
        image1 = random_state.randn(10, 10).astype(np.complex64)
        image2 = image1.copy()

        ccd = CoherentChangeDetection(window_size=7)
        coherence = ccd.apply(image1, image2)

        assert coherence.shape == (10, 10)
        assert np.all(np.isfinite(coherence))

    def test_large_window(self, random_state):
        """Test with large window size."""
        image1 = random_state.randn(256, 256).astype(np.complex64)
        image2 = image1.copy()

        ccd = CoherentChangeDetection(window_size=31)
        coherence = ccd.apply(image1, image2)

        assert coherence.shape == (256, 256)
        assert np.mean(coherence) > 0.98

    def test_single_pixel_differences(self, synthetic_image):
        """Test sensitivity to single pixel changes."""
        image1 = synthetic_image.copy()
        image2 = image1.copy()

        # Change a single pixel
        image2[256, 256] = 1000 + 1000j

        ccd = CoherentChangeDetection(window_size=7)
        coherence = ccd.apply(image1, image2)

        # The changed pixel's neighborhood should have lower coherence
        # but not necessarily 0 since it's averaged over the window
        assert coherence[256, 256] < np.mean(coherence)

    def test_zero_images(self):
        """Test with all-zero images (edge case for division by zero)."""
        image1 = np.zeros((100, 100), dtype=np.complex64)
        image2 = np.zeros((100, 100), dtype=np.complex64)

        ccd = CoherentChangeDetection(window_size=7)
        coherence = ccd.apply(image1, image2)

        # Should handle division by zero gracefully
        assert coherence.shape == (100, 100)
        assert np.all(coherence == 0.0)
        assert np.all(np.isfinite(coherence))

    # ------------------------------------------------------------------
    # Parameter override tests (via kwargs)
    # ------------------------------------------------------------------

    def test_parameter_override_via_kwargs(self, synthetic_image, random_state):
        """Test that window_size can be overridden via kwargs."""
        image1 = synthetic_image.copy()
        image2 = synthetic_image.copy()

        # Add some noise to image2 to create partial correlation
        # This makes window size effects more visible
        noise = (random_state.randn(*image2.shape) +
                1j * random_state.randn(*image2.shape)) * 0.3
        image2 = image2 + noise.astype(np.complex64)

        ccd = CoherentChangeDetection(window_size=7)

        # Use default window_size=7
        coherence1 = ccd.apply(image1, image2)

        # Override with window_size=21 via kwargs (larger window smooths more)
        coherence2 = ccd.apply(image1, image2, window_size=21)

        # Results should differ due to different window sizes
        # Larger windows smooth more, so we expect different results
        assert not np.allclose(coherence1, coherence2)

    # ------------------------------------------------------------------
    # Data type tests
    # ------------------------------------------------------------------

    def test_complex128_input(self, random_state):
        """Test with complex128 (double precision) input."""
        image1 = random_state.randn(100, 100).astype(np.complex128)
        image2 = image1.copy()

        ccd = CoherentChangeDetection(window_size=7)
        coherence = ccd.apply(image1, image2)

        assert coherence.dtype == np.float32
        assert np.mean(coherence) > 0.98

    def test_output_always_float32(self, synthetic_image):
        """Output should always be float32 regardless of input precision."""
        # Test with complex64
        image_c64 = synthetic_image.astype(np.complex64)
        ccd = CoherentChangeDetection()
        coherence_c64 = ccd.apply(image_c64, image_c64)
        assert coherence_c64.dtype == np.float32

        # Test with complex128
        image_c128 = synthetic_image.astype(np.complex128)
        coherence_c128 = ccd.apply(image_c128, image_c128)
        assert coherence_c128.dtype == np.float32


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
