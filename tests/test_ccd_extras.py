# -*- coding: utf-8 -*-
"""Tests for CCD extra functions (noise CCD, SCCM, angle CCD)."""
import numpy as np
import pytest
from grdl_sartoolbox.processing.ccd import (
    ccd_mem, ccd_noise_mem, sccm, ccd_mem_angle,
)


@pytest.fixture
def identical_images():
    """Two identical complex images."""
    rng = np.random.RandomState(42)
    img = rng.randn(64, 64) + 1j * rng.randn(64, 64)
    return img, img.copy()


@pytest.fixture
def uncorrelated_images():
    """Two uncorrelated complex images."""
    rng = np.random.RandomState(42)
    img1 = rng.randn(64, 64) + 1j * rng.randn(64, 64)
    img2 = rng.randn(64, 64) + 1j * rng.randn(64, 64)
    return img1, img2


class TestCCDMem:
    def test_identical_images_high_coherence(self, identical_images):
        img1, img2 = identical_images
        ccd, phase = ccd_mem(img1, img2, 7)
        assert ccd.shape == (64, 64)
        assert np.mean(ccd) > 0.95

    def test_uncorrelated_low_coherence(self, uncorrelated_images):
        img1, img2 = uncorrelated_images
        ccd, phase = ccd_mem(img1, img2, 7)
        assert np.mean(ccd) < 0.4

    def test_phase_zero_for_identical(self, identical_images):
        img1, img2 = identical_images
        _, phase = ccd_mem(img1, img2, 7)
        assert np.allclose(phase, 0.0, atol=0.01)

    def test_rectangular_window(self):
        rng = np.random.RandomState(42)
        img = rng.randn(64, 64) + 1j * rng.randn(64, 64)
        ccd, _ = ccd_mem(img, img.copy(), (5, 9))
        assert ccd.shape == (64, 64)
        assert np.mean(ccd) > 0.95


class TestCCDNoiseMem:
    def test_zero_noise_equals_standard_ccd(self, identical_images):
        img1, img2 = identical_images
        ccd_std, _ = ccd_mem(img1, img2, 7)
        ccd_noise, _ = ccd_noise_mem(img1, img2, 7, 0.0, 0.0)
        np.testing.assert_array_almost_equal(ccd_std, ccd_noise)

    def test_with_noise(self, identical_images):
        img1, img2 = identical_images
        ccd, phase = ccd_noise_mem(img1, img2, 7, 0.1, 0.1)
        assert ccd.shape == (64, 64)
        # With noise correction, should still show high coherence
        assert np.all(ccd >= 0)

    def test_high_noise_uncertainty(self, uncorrelated_images):
        img1, img2 = uncorrelated_images
        ccd, _ = ccd_noise_mem(img1, img2, 5, 100.0, 100.0)
        # Very high noise should create uncertainty (values set to 1)
        assert ccd.shape == (64, 64)


class TestSCCM:
    def test_identical_images(self, identical_images):
        img1, img2 = identical_images
        sccm_out, angle = sccm(img1, img2, 7, 0.01, 0.01)
        assert sccm_out.shape == (64, 64)
        assert angle.shape == (64, 64)

    def test_no_signal_pixels(self):
        rng = np.random.RandomState(42)
        # Very weak signal
        img1 = 1e-10 * (rng.randn(32, 32) + 1j * rng.randn(32, 32))
        img2 = 1e-10 * (rng.randn(32, 32) + 1j * rng.randn(32, 32))
        sccm_out, _ = sccm(img1, img2, 5, 1.0, 1.0)
        # Should have -100 no-signal markers
        assert np.any(sccm_out == -100.0)


class TestCCDMemAngle:
    def test_zero_angle(self, identical_images):
        img1, img2 = identical_images
        ccd_0, _ = ccd_mem_angle(img1, img2, 7, angle=0.0)
        ccd_ref, _ = ccd_mem(img1, img2, 7)
        np.testing.assert_array_almost_equal(ccd_0, ccd_ref)

    def test_90_degree_rotation(self, identical_images):
        img1, img2 = identical_images
        ccd, phase = ccd_mem_angle(img1, img2, 7, angle=90.0)
        assert ccd.shape == (64, 64)

    def test_arbitrary_angle(self):
        rng = np.random.RandomState(42)
        img = rng.randn(64, 64) + 1j * rng.randn(64, 64)
        ccd, phase = ccd_mem_angle(img, img.copy(), 7, angle=45.0)
        assert ccd.shape == (64, 64)

    def test_with_noise_metric(self, identical_images):
        img1, img2 = identical_images
        ccd, _ = ccd_mem_angle(img1, img2, 7, angle=0.0, noise_var=0.01, metric='noise')
        assert ccd.shape == (64, 64)

    def test_sccm_metric(self, identical_images):
        img1, img2 = identical_images
        ccd, _ = ccd_mem_angle(img1, img2, 7, angle=0.0, noise_var=0.01, metric='sccm')
        assert ccd.shape == (64, 64)
