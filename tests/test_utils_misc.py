# -*- coding: utf-8 -*-
"""Tests for utility functions (fast_running_mean, local_sum)."""
import numpy as np
import pytest
from grdl_sartoolbox.utils.misc import fast_running_mean, local_sum


class TestFastRunningMean:
    def test_2d_uniform(self):
        """Running mean of uniform array should be the same value."""
        data = np.ones((20, 20)) * 5.0
        result = fast_running_mean(data, (3, 3), 'zeros')
        # Interior should be exactly 5
        np.testing.assert_allclose(result[2:-2, 2:-2], 5.0, atol=1e-10)

    def test_2d_zeros_padding(self):
        data = np.ones((10, 10))
        result = fast_running_mean(data, (3, 3), 'zeros')
        assert result.shape == (10, 10)
        # Center should be close to 1
        assert abs(result[5, 5] - 1.0) < 1e-10
        # Corners should be less due to zero padding
        assert result[0, 0] < 1.0

    def test_2d_mean_padding(self):
        data = np.ones((10, 10)) * 3.0
        result = fast_running_mean(data, (3, 3), 'mean')
        assert result.shape == (10, 10)
        np.testing.assert_allclose(result, 3.0, atol=1e-10)

    def test_3d(self):
        data = np.ones((10, 10, 10)) * 2.0
        result = fast_running_mean(data, (3, 3, 3), 'zeros')
        assert result.shape == (10, 10, 10)
        # Interior should be 2
        np.testing.assert_allclose(result[2:-2, 2:-2, 2:-2], 2.0, atol=1e-10)

    def test_scalar_window(self):
        data = np.ones((10, 10))
        result = fast_running_mean(data, 5, 'zeros')
        assert result.shape == (10, 10)

    def test_even_window_raises(self):
        data = np.ones((10, 10))
        with pytest.raises(ValueError):
            fast_running_mean(data, (4, 4), 'zeros')

    def test_dimension_mismatch_raises(self):
        data = np.ones((10, 10))
        with pytest.raises(ValueError):
            fast_running_mean(data, (3, 3, 3), 'zeros')


class TestLocalSum:
    def test_basic(self):
        data = np.ones((10, 10))
        result = local_sum(data, (3, 3))
        assert result.shape == (10, 10)
        # Center should be 9 (3x3 window of ones)
        assert abs(result[5, 5] - 9.0) < 1e-10

    def test_scalar_window(self):
        data = np.ones((10, 10))
        result = local_sum(data, 5)
        assert result.shape == (10, 10)
        # Center should be 25 (5x5)
        assert abs(result[5, 5] - 25.0) < 1e-10

    def test_rectangular_window(self):
        data = np.ones((20, 20))
        result = local_sum(data, (3, 5))
        assert result.shape == (20, 20)
        # Center should be 15 (3x5)
        assert abs(result[10, 10] - 15.0) < 1e-10

    def test_agrees_with_convolution(self):
        """Local sum should agree with conv2d for interior pixels."""
        rng = np.random.RandomState(42)
        data = rng.randn(20, 20)
        win = (5, 5)
        ls = local_sum(data, win)

        from scipy.signal import convolve2d
        conv = convolve2d(data, np.ones(win), mode='same')

        # Interior pixels should match well
        np.testing.assert_allclose(
            ls[3:-3, 3:-3], conv[3:-3, 3:-3], atol=1e-8
        )
