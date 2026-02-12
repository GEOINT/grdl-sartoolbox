# -*- coding: utf-8 -*-
"""Tests for ACD, CSI, PDV, filtering, normalize, RGIQE, IFP, RCS, signal analysis."""
import numpy as np
import pytest


class TestACD:
    def test_dft_registration_no_shift(self):
        from grdl_sartoolbox.processing.acd import dft_registration
        rng = np.random.RandomState(42)
        img = rng.randn(64, 64) + 1j * rng.randn(64, 64)
        result = dft_registration(np.fft.fft2(img), np.fft.fft2(img))
        assert abs(result[1]) < 0.5  # row shift near 0
        assert abs(result[2]) < 0.5  # col shift near 0

    def test_dft_register_image(self):
        from grdl_sartoolbox.processing.acd import dft_register_image
        rng = np.random.RandomState(42)
        img = rng.randn(64, 64) + 1j * rng.randn(64, 64)
        # dft_register_image expects FFT'd images
        fft1 = np.fft.fft2(img)
        fft2 = np.fft.fft2(img)
        registered = dft_register_image(fft1, fft2)
        assert registered.shape == img.shape

    def test_acd_rgb(self):
        from grdl_sartoolbox.processing.acd import acd_rgb
        rng = np.random.RandomState(42)
        img1 = rng.randn(32, 32) + 1j * rng.randn(32, 32)
        img2 = img1 * 1.5  # Scaled version
        rgb = acd_rgb(img1, img2)
        assert rgb.shape == (32, 32, 3)
        assert rgb.dtype == np.float32  # Returns float32, not uint8


class TestCSI:
    def test_csi_mem_basic(self):
        from grdl_sartoolbox.processing.csi import csi_mem
        rng = np.random.RandomState(42)
        data = rng.randn(64, 64) + 1j * rng.randn(64, 64)
        rgb = csi_mem(data)
        assert rgb.shape == (64, 64, 3)
        assert rgb.dtype == np.float32

    def test_csi_mem_dim1(self):
        from grdl_sartoolbox.processing.csi import csi_mem
        rng = np.random.RandomState(42)
        data = rng.randn(64, 64) + 1j * rng.randn(64, 64)
        rgb = csi_mem(data, dim=1)
        assert rgb.shape == (64, 64, 3)


class TestPDV:
    def test_pdv_mem_basic(self):
        from grdl_sartoolbox.processing.pdv import pdv_mem
        rng = np.random.RandomState(42)
        data = rng.randn(64, 64) + 1j * rng.randn(64, 64)
        result = pdv_mem(data)
        assert result.shape == (64, 64)
        # PDV returns real-valued phase gradient, not complex
        assert np.all(np.isfinite(result))

    def test_pdv_mem_dim1(self):
        from grdl_sartoolbox.processing.pdv import pdv_mem
        rng = np.random.RandomState(42)
        data = rng.randn(64, 64) + 1j * rng.randn(64, 64)
        result = pdv_mem(data, dim=1)
        assert result.shape == (64, 64)


class TestFiltering:
    def test_apodize_2d(self):
        from grdl_sartoolbox.processing.filtering import apodize_2d
        rng = np.random.RandomState(42)
        data = rng.randn(64, 64) + 1j * rng.randn(64, 64)
        result = apodize_2d(data)
        assert result.shape == (64, 64)
        assert np.iscomplexobj(result)

    def test_upsample_image(self):
        from grdl_sartoolbox.processing.filtering import upsample_image
        rng = np.random.RandomState(42)
        data = rng.randn(32, 32) + 1j * rng.randn(32, 32)
        result = upsample_image(data, row_factor=2, col_factor=2)
        assert result.shape == (64, 64)


class TestNormalize:
    def test_is_normalized_uniform(self):
        from grdl_sartoolbox.processing.normalize import is_normalized
        assert is_normalized(weight_type='UNIFORM') is True

    def test_is_normalized_nonuniform(self):
        from grdl_sartoolbox.processing.normalize import is_normalized
        assert is_normalized(weight_type='HAMMING') is False

    def test_sicd_polyval2d(self):
        from grdl_sartoolbox.processing.normalize import sicd_polyval2d
        # coeffs[i,j] is coeff for x^j * y^i
        # For constant poly, result should be scalar-broadcast
        coeffs = np.array([[5.0]])
        x = np.array([1.0, 2.0])
        y = np.array([0.5])
        result = sicd_polyval2d(coeffs, x, y)
        np.testing.assert_allclose(result, 5.0)

    def test_estimate_weighting(self):
        from grdl_sartoolbox.processing.normalize import estimate_weighting
        rng = np.random.RandomState(42)
        data = rng.randn(64, 128) + 1j * rng.randn(64, 128)
        weights = estimate_weighting(data, dim=0)
        assert weights.shape == (64,)
        assert np.all(weights >= 0)

    def test_sicd_weight_to_fun_uniform(self):
        from grdl_sartoolbox.processing.normalize import sicd_weight_to_fun
        fn = sicd_weight_to_fun('UNIFORM')
        x = np.linspace(0, 1, 100)
        w = fn(x)
        np.testing.assert_array_almost_equal(w, 1.0)

    def test_sicd_weight_to_fun_hamming(self):
        from grdl_sartoolbox.processing.normalize import sicd_weight_to_fun
        fn = sicd_weight_to_fun('HAMMING')
        x = np.linspace(0, 1, 100)
        w = fn(x)
        assert w.shape == (100,)
        assert np.max(w) > 0

    def test_normalize_complex_passthrough(self):
        from grdl_sartoolbox.processing.normalize import normalize_complex
        rng = np.random.RandomState(42)
        data = rng.randn(32, 32) + 1j * rng.randn(32, 32)
        result = normalize_complex(data)
        np.testing.assert_array_almost_equal(result, data)


class TestRGIQE:
    def test_compute_rgiqe_basic(self):
        from grdl_sartoolbox.processing.rgiqe import compute_rgiqe, RGIQEResult
        result = compute_rgiqe(
            range_resolution=0.5,
            azimuth_resolution=0.5,
            graze_angle=np.radians(45),
            nesz_db=-20.0
        )
        assert isinstance(result, RGIQEResult)
        assert 0 <= result.rniirs <= 9
        assert result.information_density > 0
        assert result.snr_db > 0

    def test_compute_rgiqe_low_resolution(self):
        from grdl_sartoolbox.processing.rgiqe import compute_rgiqe
        result = compute_rgiqe(
            range_resolution=10.0,
            azimuth_resolution=10.0,
            graze_angle=np.radians(30),
            nesz_db=-15.0
        )
        assert result.rniirs < 5  # Low res should have lower RNIIRS


class TestIFPUtils:
    def test_resolution_extent(self):
        from grdl_sartoolbox.processing.ifp_utils import pulse_info_to_resolution_extent
        # Simple 10-pulse synthetic aperture
        angles = np.linspace(-0.01, 0.01, 10)
        range_vecs = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(10)])
        result = pulse_info_to_resolution_extent(
            range_vectors=range_vecs,
            center_frequency=10e9,
            delta_frequency=1e6,
            bandwidth=100e6,
            num_samples=1024
        )
        assert result.range_resolution > 0
        assert result.azimuth_resolution > 0

    def test_deskew_rvp(self):
        from grdl_sartoolbox.processing.ifp_utils import deskew_rvp
        rng = np.random.RandomState(42)
        pulses = rng.randn(128) + 1j * rng.randn(128)
        output, t_start = deskew_rvp(pulses, 1e6, 1e12)
        assert output.shape[0] >= 128

    def test_pfa_inverse(self):
        from grdl_sartoolbox.processing.ifp_utils import pfa_inverse
        rng = np.random.RandomState(42)
        data = rng.randn(64, 64) + 1j * rng.randn(64, 64)
        result = pfa_inverse(
            data,
            center_freq=10e9,
            sample_spacing=(0.5, 0.5),
            imp_resp_bw=(0.8, 0.8)
        )
        assert result.ndim == 2


class TestRCS:
    def test_compute_rcs(self):
        from grdl_sartoolbox.processing.rcs import compute_rcs
        rng = np.random.RandomState(42)
        data = rng.randn(32, 32) + 1j * rng.randn(32, 32)
        rcs = compute_rcs(data)
        assert rcs > 0

    def test_compute_rcs_with_mask(self):
        from grdl_sartoolbox.processing.rcs import compute_rcs
        rng = np.random.RandomState(42)
        data = rng.randn(32, 32) + 1j * rng.randn(32, 32)
        mask = np.zeros((32, 32), dtype=bool)
        mask[10:20, 10:20] = True
        rcs_full = compute_rcs(data)
        rcs_masked = compute_rcs(data, mask=mask)
        assert rcs_masked < rcs_full

    def test_compute_rcs_db(self):
        from grdl_sartoolbox.processing.rcs import compute_rcs_db
        rng = np.random.RandomState(42)
        data = rng.randn(32, 32) + 1j * rng.randn(32, 32)
        rcs_db = compute_rcs_db(data)
        assert np.isfinite(rcs_db)


class TestSignalAnalysis:
    def test_stft(self):
        from grdl_sartoolbox.processing.signal_analysis import stft
        t = np.linspace(0, 1, 1024)
        sig = np.exp(2j * np.pi * 100 * t)
        Sxx, times, freqs = stft(sig, sample_rate=1024)
        assert Sxx.ndim == 2
        assert len(times) == Sxx.shape[1]
        assert len(freqs) == Sxx.shape[0]

    def test_reramp_1d(self):
        from grdl_sartoolbox.processing.signal_analysis import reramp
        rng = np.random.RandomState(42)
        pulse = rng.randn(128) + 1j * rng.randn(128)
        output, dt = reramp(pulse, 1e6, 1e12)
        assert output.ndim == 1
        assert dt > 0

    def test_reramp_2d(self):
        from grdl_sartoolbox.processing.signal_analysis import reramp
        rng = np.random.RandomState(42)
        pulses = rng.randn(128, 4) + 1j * rng.randn(128, 4)
        output, dt = reramp(pulses, 1e6, 1e12)
        assert output.shape[1] == 4
