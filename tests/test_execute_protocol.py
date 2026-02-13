# -*- coding: utf-8 -*-
"""
Execute protocol tests for grdl-sartoolbox processors.

Verifies that LeeFilter and CoherentChangeDetection work correctly
with the ``execute(metadata, source, **kwargs)`` protocol added to
the ImageProcessor hierarchy.

License
-------
MIT License
Copyright (c) 2024 geoint.org

Created
-------
2026-02-12
"""

from __future__ import annotations

import numpy as np
import pytest

from grdl.IO.models.base import ImageMetadata
from grdl.image_processing.base import ImageProcessor, ImageTransform

from grdl_sartoolbox.processing.ccd import CoherentChangeDetection
from grdl_sartoolbox.processing.speckle_filter import LeeFilter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def meta():
    return ImageMetadata(
        format='test', rows=64, cols=64, dtype='float32', bands=1,
    )


@pytest.fixture
def sar_amplitude(meta):
    """64x64 synthetic SAR amplitude image."""
    rng = np.random.RandomState(42)
    return (rng.rand(meta.rows, meta.cols) * 100 + 10).astype(np.float32)


@pytest.fixture
def sar_complex_pair():
    """Pair of 64x64 complex SAR images (reference + match)."""
    rng = np.random.RandomState(42)
    ref = (rng.randn(64, 64) + 1j * rng.randn(64, 64)).astype(np.complex64)
    match = ref.copy()  # identical → high coherence
    return ref, match


# ---------------------------------------------------------------------------
# LeeFilter execute() protocol
# ---------------------------------------------------------------------------

class TestLeeFilterExecuteProtocol:

    def test_is_image_transform(self):
        """LeeFilter should be an ImageTransform subclass."""
        assert issubclass(LeeFilter, ImageTransform)
        assert issubclass(LeeFilter, ImageProcessor)

    def test_execute_returns_tuple(self, meta, sar_amplitude):
        """execute() returns (result, metadata) tuple."""
        lee = LeeFilter(radius=2, enl=1.0)
        out = lee.execute(meta, sar_amplitude)
        assert isinstance(out, tuple) and len(out) == 2

    def test_execute_result_is_ndarray(self, meta, sar_amplitude):
        """execute() result is an ndarray."""
        lee = LeeFilter(radius=2, enl=1.0)
        result, _ = lee.execute(meta, sar_amplitude)
        assert isinstance(result, np.ndarray)
        assert result.shape == sar_amplitude.shape
        assert result.dtype == np.float32

    def test_execute_metadata_updated(self, meta, sar_amplitude):
        """Returned metadata reflects output shape and dtype."""
        lee = LeeFilter(radius=2, enl=1.0)
        result, out_meta = lee.execute(meta, sar_amplitude)
        assert isinstance(out_meta, ImageMetadata)
        assert out_meta.dtype == 'float32'
        assert out_meta.rows == result.shape[0]
        assert out_meta.cols == result.shape[1]
        assert out_meta.bands == 1

    def test_metadata_property_available(self, meta, sar_amplitude):
        """self.metadata is set during apply()."""
        lee = LeeFilter(radius=2, enl=1.0)
        assert lee.metadata is None  # before execute
        lee.execute(meta, sar_amplitude)
        assert lee.metadata is meta  # after execute

    def test_execute_matches_apply(self, meta, sar_amplitude):
        """execute() produces same result as direct apply()."""
        lee = LeeFilter(radius=2, enl=1.0)
        result_execute, _ = lee.execute(meta, sar_amplitude)

        lee2 = LeeFilter(radius=2, enl=1.0)
        result_apply = lee2.apply(sar_amplitude)

        np.testing.assert_array_equal(result_execute, result_apply)

    def test_execute_with_kwargs_override(self, meta, sar_amplitude):
        """Parameter overrides via kwargs work through execute()."""
        lee = LeeFilter(radius=2, enl=1.0)
        result_r2, _ = lee.execute(meta, sar_amplitude)
        result_r4, _ = lee.execute(meta, sar_amplitude, radius=4)
        assert not np.array_equal(result_r2, result_r4)

    def test_gpu_compatible_false(self):
        """LeeFilter should not claim GPU compatibility (uses scipy)."""
        lee = LeeFilter()
        assert not lee.__gpu_compatible__

    def test_supports_gpu_transfer_false(self):
        """dispatch.supports_gpu_transfer() returns False for LeeFilter."""
        from grdl_rt.execution.dispatch import supports_gpu_transfer
        lee = LeeFilter()
        assert not supports_gpu_transfer(lee)


# ---------------------------------------------------------------------------
# CoherentChangeDetection execute() protocol
# ---------------------------------------------------------------------------

class TestCCDExecuteProtocol:

    def test_is_image_processor(self):
        """CCD should be an ImageProcessor but NOT ImageTransform."""
        assert issubclass(CoherentChangeDetection, ImageProcessor)
        assert not issubclass(CoherentChangeDetection, ImageTransform)

    def test_has_explicit_execute(self):
        """CCD defines its own execute() (not just the base fallback)."""
        assert 'execute' in CoherentChangeDetection.__dict__

    def test_execute_returns_tuple(self, meta, sar_complex_pair):
        """execute() returns (result, metadata) tuple."""
        ref, match = sar_complex_pair
        ccd = CoherentChangeDetection(window_size=7)
        out = ccd.execute(meta, ref, match_image=match)
        assert isinstance(out, tuple) and len(out) == 2

    def test_execute_result_is_coherence(self, meta, sar_complex_pair):
        """execute() result is a float32 coherence map."""
        ref, match = sar_complex_pair
        ccd = CoherentChangeDetection(window_size=7)
        result, _ = ccd.execute(meta, ref, match_image=match)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == ref.shape
        assert np.all(result >= 0.0) and np.all(result <= 1.0)

    def test_execute_metadata_updated(self, meta, sar_complex_pair):
        """Returned metadata reflects coherence output."""
        ref, match = sar_complex_pair
        ccd = CoherentChangeDetection(window_size=7)
        result, out_meta = ccd.execute(meta, ref, match_image=match)
        assert isinstance(out_meta, ImageMetadata)
        assert out_meta.dtype == 'float32'
        assert out_meta.rows == result.shape[0]
        assert out_meta.cols == result.shape[1]
        assert out_meta.bands == 1

    def test_metadata_property_available(self, meta, sar_complex_pair):
        """self.metadata is set during apply()."""
        ref, match = sar_complex_pair
        ccd = CoherentChangeDetection(window_size=7)
        assert ccd.metadata is None
        ccd.execute(meta, ref, match_image=match)
        assert ccd.metadata is meta

    def test_execute_matches_apply(self, meta, sar_complex_pair):
        """execute() produces same result as direct apply()."""
        ref, match = sar_complex_pair
        ccd = CoherentChangeDetection(window_size=7)
        result_execute, _ = ccd.execute(meta, ref, match_image=match)

        ccd2 = CoherentChangeDetection(window_size=7)
        result_apply = ccd2.apply(ref, match)

        np.testing.assert_array_equal(result_execute, result_apply)

    def test_execute_with_kwargs_override(self, meta, sar_complex_pair):
        """window_size override via kwargs works through execute()."""
        ref, match = sar_complex_pair
        ccd = CoherentChangeDetection(window_size=7)
        result_w7, _ = ccd.execute(meta, ref, match_image=match)
        result_w11, _ = ccd.execute(meta, ref, match_image=match, window_size=11)
        assert not np.array_equal(result_w7, result_w11)

    def test_identical_images_high_coherence(self, meta, sar_complex_pair):
        """Identical images via execute() should give high coherence."""
        ref, match = sar_complex_pair
        ccd = CoherentChangeDetection(window_size=7)
        result, _ = ccd.execute(meta, ref, match_image=match)
        assert np.mean(result) > 0.98

    def test_gpu_compatible_false(self):
        """CCD should not claim GPU compatibility (uses scipy)."""
        ccd = CoherentChangeDetection()
        assert not ccd.__gpu_compatible__

    def test_supports_gpu_transfer_false(self):
        """dispatch.supports_gpu_transfer() returns False for CCD."""
        from grdl_rt.execution.dispatch import supports_gpu_transfer
        ccd = CoherentChangeDetection()
        assert not supports_gpu_transfer(ccd)
