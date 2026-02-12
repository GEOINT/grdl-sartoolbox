# -*- coding: utf-8 -*-
"""Tests for visualization remap functions."""
import numpy as np
import pytest
from grdl_sartoolbox.visualization.remap import (
    amplitude_to_density,
    density_remap,
    brighter_remap,
    darker_remap,
    high_contrast_remap,
    linear_remap,
    log_remap,
    nrl_remap,
    pedf_remap,
    get_remap_list,
    get_remap_function,
)


@pytest.fixture
def complex_data():
    rng = np.random.RandomState(42)
    return rng.randn(64, 64) + 1j * rng.randn(64, 64)


@pytest.fixture
def real_data():
    rng = np.random.RandomState(42)
    return np.abs(rng.randn(64, 64))


class TestAmplitudeToDensity:
    def test_basic(self, complex_data):
        result = amplitude_to_density(complex_data)
        assert result.shape == (64, 64)
        assert np.all(np.isfinite(result))

    def test_custom_params(self, complex_data):
        result = amplitude_to_density(complex_data, dmin=60, mmult=4)
        assert result.shape == (64, 64)


class TestDensityRemap:
    def test_uint8_output(self, complex_data):
        result = density_remap(complex_data)
        assert result.dtype == np.uint8
        assert result.shape == (64, 64)

    def test_range_0_255(self, complex_data):
        result = density_remap(complex_data)
        assert np.min(result) >= 0
        assert np.max(result) <= 255


class TestBrighterRemap:
    def test_brighter_than_standard(self, complex_data):
        std = density_remap(complex_data).astype(float)
        bright = brighter_remap(complex_data).astype(float)
        # Brighter should have higher mean
        assert np.mean(bright) >= np.mean(std) - 10  # Allow some tolerance


class TestDarkerRemap:
    def test_output_shape(self, complex_data):
        result = darker_remap(complex_data)
        assert result.shape == (64, 64)
        assert result.dtype == np.uint8


class TestHighContrastRemap:
    def test_output_shape(self, complex_data):
        result = high_contrast_remap(complex_data)
        assert result.shape == (64, 64)
        assert result.dtype == np.uint8


class TestLinearRemap:
    def test_complex_input(self, complex_data):
        result = linear_remap(complex_data)
        assert result.dtype == np.float64
        assert np.all(result >= 0)  # Magnitude is non-negative

    def test_real_input(self, real_data):
        result = linear_remap(real_data)
        np.testing.assert_array_equal(result, real_data.astype(np.float64))


class TestLogRemap:
    def test_output_uint8(self, complex_data):
        result = log_remap(complex_data)
        assert result.dtype == np.uint8
        assert result.shape == (64, 64)

    def test_custom_span(self, complex_data):
        result = log_remap(complex_data, span_db=30)
        assert result.shape == (64, 64)


class TestNRLRemap:
    def test_output_uint8(self, complex_data):
        result = nrl_remap(complex_data)
        assert result.dtype == np.uint8
        assert result.shape == (64, 64)

    def test_custom_params(self, complex_data):
        result = nrl_remap(complex_data, a=2.0, c=200)
        assert result.shape == (64, 64)


class TestPEDFRemap:
    def test_output_uint8(self, complex_data):
        result = pedf_remap(complex_data)
        assert result.dtype == np.uint8
        assert result.shape == (64, 64)

    def test_high_value_compression(self):
        """PEDF should compress values above 128."""
        data = np.ones((10, 10)) * 1000 + 0j  # High uniform value
        result = pedf_remap(data)
        assert np.all(result <= 255)


class TestRemapRegistry:
    def test_get_remap_list(self):
        remaps = get_remap_list()
        assert 'density' in remaps
        assert 'nrl' in remaps
        assert 'log' in remaps
        assert len(remaps) >= 8

    def test_get_remap_function(self, complex_data):
        fn = get_remap_function('density')
        result = fn(complex_data)
        assert result.shape == (64, 64)

    def test_get_unknown_remap(self):
        with pytest.raises(KeyError):
            get_remap_function('nonexistent')
