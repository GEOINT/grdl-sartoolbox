# -*- coding: utf-8 -*-
"""
Tests for Polar Format Algorithm (PFA).

Tests the pfa_mem function and supporting functions with synthetic
phase history data.

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

from grdl_sartoolbox.processing.pfa import (
    pfa_mem,
    NarrowbandData,
    _pfa_bistatic_pos,
    _pfa_polar_coords,
    _pfa_inscribed_rectangle_coords,
    _pfa_fft_zeropad_1d,
    _sinc_interp,
)


class TestNarrowbandData:
    """Test suite for NarrowbandData structure."""

    def test_narrowband_data_creation(self):
        """Test NarrowbandData creation with valid inputs."""
        num_pulses = 50

        nbdata = NarrowbandData(
            TxPos=np.random.randn(num_pulses, 3) * 1000,
            RcvPos=np.random.randn(num_pulses, 3) * 1000,
            SRPPos=np.tile([0, 0, 0], (num_pulses, 1)),
            SC0=np.ones(num_pulses) * 10e9,
            SCSS=np.ones(num_pulses) * 1e6
        )

        assert nbdata.TxPos.shape == (num_pulses, 3)
        assert nbdata.RcvPos.shape == (num_pulses, 3)
        assert nbdata.SRPPos.shape == (num_pulses, 3)
        assert len(nbdata.SC0) == num_pulses
        assert len(nbdata.SCSS) == num_pulses

    def test_narrowband_data_validation_tx_pos(self):
        """Wrong TxPos shape should raise ValueError."""
        num_pulses = 50

        with pytest.raises(ValueError, match="TxPos must be shape"):
            NarrowbandData(
                TxPos=np.random.randn(num_pulses, 2),  # Wrong shape
                RcvPos=np.random.randn(num_pulses, 3),
                SRPPos=np.tile([0, 0, 0], (num_pulses, 1)),
                SC0=np.ones(num_pulses) * 10e9,
                SCSS=np.ones(num_pulses) * 1e6
            )

    def test_narrowband_data_validation_scss_length(self):
        """Wrong SCSS length should raise ValueError."""
        num_pulses = 50

        with pytest.raises(ValueError, match="SCSS length"):
            NarrowbandData(
                TxPos=np.random.randn(num_pulses, 3),
                RcvPos=np.random.randn(num_pulses, 3),
                SRPPos=np.tile([0, 0, 0], (num_pulses, 1)),
                SC0=np.ones(num_pulses) * 10e9,
                SCSS=np.ones(30) * 1e6  # Wrong length
            )


class TestHelperFunctions:
    """Test suite for PFA helper functions."""

    def test_sinc_interp_basic(self):
        """Test basic sinc interpolation."""
        # Simple test: interpolate a constant (interior points)
        x = np.ones(10)
        s = np.arange(10, dtype=float)
        u = np.linspace(2, 7, 10)  # Use interior points only

        y = _sinc_interp(x, s, u)

        # Interior values should be close to 1
        assert np.allclose(y, 1.0, atol=0.15)  # Sinc ringing can cause deviations

    def test_pfa_bistatic_pos_monostatic(self):
        """Bistatic position for monostatic should be in direction of pos."""
        num_pulses = 10
        pos = np.random.randn(num_pulses, 3) * 1000
        srp = np.tile([0, 0, 0], (num_pulses, 1))

        bi_pos, freq_scale = _pfa_bistatic_pos(pos, pos, srp)

        # For monostatic, bistatic pos should be in same direction as pos
        assert bi_pos.shape == (num_pulses, 3)

        # Check direction (normalized vectors should match)
        pos_unit = pos / np.linalg.norm(pos, axis=1)[:, np.newaxis]
        bi_pos_unit = bi_pos / np.linalg.norm(bi_pos, axis=1)[:, np.newaxis]
        assert np.allclose(pos_unit, bi_pos_unit, rtol=1e-5)

        # Magnitude should be 2*|pos| for monostatic
        assert np.allclose(np.linalg.norm(bi_pos, axis=1),
                          2 * np.linalg.norm(pos, axis=1), rtol=1e-5)

        # Frequency scale should be 1.0 for monostatic
        assert np.allclose(freq_scale, 1.0, rtol=1e-5)

    def test_pfa_polar_coords_basic(self):
        """Test polar coordinate computation."""
        num_pulses = 20

        # Simple linear flight path
        pos = np.zeros((num_pulses, 3))
        pos[:, 0] = np.linspace(-10, 10, num_pulses)  # X varies
        pos[:, 2] = 100  # Constant altitude

        scp = np.array([0, 0, 0])
        coa_pos = pos[num_pulses // 2]
        ipn = np.array([0, 0, 1])  # Vertical
        fpn = np.array([0, 0, 1])  # Same as ipn for simplicity

        k_a, k_sf = _pfa_polar_coords(pos, scp, coa_pos, ipn, fpn)

        # Check outputs
        assert k_a.shape == (num_pulses,)
        assert k_sf.shape == (num_pulses,)
        assert np.all(np.isfinite(k_a))
        assert np.all(np.isfinite(k_sf))

        # Center pulse should have k_a ≈ 0
        assert np.abs(k_a[num_pulses // 2]) < 0.1

    def test_pfa_inscribed_rectangle_coords(self):
        """Test inscribed rectangle coordinate computation."""
        num_pulses = 30

        # Symmetric angles around zero
        k_a = np.linspace(-0.2, 0.2, num_pulses)
        k_r0 = np.ones(num_pulses) * 1e8  # 100 MHz
        bw = np.ones(num_pulses) * 1e7    # 10 MHz bandwidth

        k_v_bounds, k_u_bounds = _pfa_inscribed_rectangle_coords(k_a, k_r0, bw)

        # Check bounds are valid
        assert len(k_v_bounds) == 2
        assert len(k_u_bounds) == 2
        assert k_v_bounds[0] < k_v_bounds[1]
        assert k_u_bounds[0] < k_u_bounds[1]

    def test_pfa_fft_zeropad_1d(self):
        """Test zero-padded 1D FFT."""
        data = np.random.randn(64, 32) + 1j * np.random.randn(64, 32)
        sample_rate = 1.5

        result = _pfa_fft_zeropad_1d(data, sample_rate)

        # Output should be larger due to zero-padding
        assert result.shape[0] == int(np.floor(64 * sample_rate))
        assert result.shape[1] == 32
        assert np.iscomplexobj(result)


class TestPFAAlgorithm:
    """Test suite for main PFA algorithm."""

    @pytest.fixture
    def synthetic_spotlight_data(self):
        """Generate synthetic spotlight CPHD data."""
        num_samples = 128
        num_pulses = 64

        # Random phase history
        phase_history = (np.random.randn(num_samples, num_pulses) +
                        1j * np.random.randn(num_samples, num_pulses))
        phase_history = phase_history.astype(np.complex64)

        # Realistic ground point (approximate mid-latitude location)
        srp_ground = np.array([4000e3, 3000e3, 3500e3])  # ECEF meters

        # Linear flight path at altitude above ground point
        angles = np.linspace(-0.3, 0.3, num_pulses)
        radius = 1000.0
        altitude = 5000.0

        # Platform positions relative to ground
        tx_x = srp_ground[0] + radius * np.sin(angles)
        tx_y = srp_ground[1] + np.zeros(num_pulses)
        tx_z = srp_ground[2] + np.ones(num_pulses) * altitude

        tx_pos = np.column_stack([tx_x, tx_y, tx_z])

        # Constant SRP (spotlight) - same ground point for all pulses
        srp = np.tile(srp_ground, (num_pulses, 1))

        # Frequency parameters
        sc0 = np.ones(num_pulses) * 10e9    # 10 GHz start
        scss = np.ones(num_pulses) * 100e6  # 100 MHz step

        nbdata = NarrowbandData(
            TxPos=tx_pos,
            RcvPos=tx_pos,  # Monostatic
            SRPPos=srp,
            SC0=sc0,
            SCSS=scss
        )

        return phase_history, nbdata

    def test_pfa_mem_runs(self, synthetic_spotlight_data):
        """PFA should run without errors."""
        phase_history, nbdata = synthetic_spotlight_data

        image = pfa_mem(phase_history, nbdata)

        # Output should be complex
        assert np.iscomplexobj(image)
        assert image.dtype == np.complex64
        # Size should be larger due to zero-padding
        assert image.shape[0] >= phase_history.shape[0]
        assert image.shape[1] >= phase_history.shape[1]

    def test_pfa_mem_with_sample_rate(self, synthetic_spotlight_data):
        """PFA with different sample rates."""
        phase_history, nbdata = synthetic_spotlight_data

        image1 = pfa_mem(phase_history, nbdata, sample_rate=1.0)
        image2 = pfa_mem(phase_history, nbdata, sample_rate=2.0)

        # Different sample rates should give different sizes
        assert image1.shape[0] < image2.shape[0]
        assert image1.shape[1] < image2.shape[1]

    def test_pfa_mem_non_spotlight_raises_error(self):
        """Non-spotlight data (varying SRP) should raise ValueError."""
        num_samples = 64
        num_pulses = 32

        phase_history = (np.random.randn(num_samples, num_pulses) +
                        1j * np.random.randn(num_samples, num_pulses))

        # Varying SRP (not spotlight)
        tx_pos = np.random.randn(num_pulses, 3) * 1000
        srp = np.random.randn(num_pulses, 3) * 10  # Varies

        nbdata = NarrowbandData(
            TxPos=tx_pos,
            RcvPos=tx_pos,
            SRPPos=srp,
            SC0=np.ones(num_pulses) * 10e9,
            SCSS=np.ones(num_pulses) * 1e6
        )

        with pytest.raises(ValueError, match="Non-spotlight data not supported"):
            pfa_mem(phase_history, nbdata)

    def test_pfa_mem_output_is_finite(self, synthetic_spotlight_data):
        """PFA output should have no NaN or Inf values."""
        phase_history, nbdata = synthetic_spotlight_data

        image = pfa_mem(phase_history, nbdata)

        assert np.all(np.isfinite(image))

    def test_pfa_mem_verbose(self, synthetic_spotlight_data, capsys):
        """PFA with verbose=True should print progress."""
        phase_history, nbdata = synthetic_spotlight_data

        image = pfa_mem(phase_history, nbdata, verbose=True)

        captured = capsys.readouterr()
        assert "PFA:" in captured.out
        assert "pulses" in captured.out

    def test_pfa_mem_bistatic(self):
        """Test PFA with bistatic geometry."""
        num_samples = 64
        num_pulses = 32

        phase_history = (np.random.randn(num_samples, num_pulses) +
                        1j * np.random.randn(num_samples, num_pulses))

        # Realistic ground point
        srp_ground = np.array([4000e3, 3000e3, 3500e3])

        # Separate Tx and Rcv positions above ground
        tx_pos = srp_ground + np.random.randn(num_pulses, 3) * 1000
        tx_pos[:, 2] += 5000  # Ensure altitude above ground
        rcv_pos = tx_pos + np.array([10, 0, 0])  # 10m offset

        srp = np.tile(srp_ground, (num_pulses, 1))

        nbdata = NarrowbandData(
            TxPos=tx_pos,
            RcvPos=rcv_pos,
            SRPPos=srp,
            SC0=np.ones(num_pulses) * 10e9,
            SCSS=np.ones(num_pulses) * 1e6
        )

        image = pfa_mem(phase_history, nbdata)

        assert np.iscomplexobj(image)
        assert np.all(np.isfinite(image))


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
