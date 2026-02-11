# -*- coding: utf-8 -*-
"""
Tests for constants module.

Tests physical constants, unit conversions, and helper functions.

License
-------
MIT License
Copyright (c) 2024 geoint.org

Created
-------
2026-02-11
"""

import pytest
import math

from grdl_sartoolbox.utils import constants


class TestPhysicalConstants:
    """Test physical constants values."""

    def test_speed_of_light(self):
        """Speed of light should be exact SI value."""
        assert constants.SPEED_OF_LIGHT == 299792458.0

    def test_two_way_speed_of_light(self):
        """Two-way speed of light for radar."""
        assert constants.TWO_WAY_SPEED_OF_LIGHT == constants.SPEED_OF_LIGHT / 2.0

    def test_pi_value(self):
        """Pi should be close to math.pi."""
        assert abs(constants.PI - math.pi) < 1e-15

    def test_two_pi(self):
        """Two pi should be 2 * pi."""
        assert abs(constants.TWO_PI - 2 * math.pi) < 1e-15


class TestUnitConversions:
    """Test unit conversion constants."""

    def test_feet_to_meters(self):
        """Feet to meters conversion."""
        assert constants.FEET_TO_METERS == 0.3048

    def test_meters_to_feet(self):
        """Meters to feet conversion."""
        assert abs(constants.METERS_TO_FEET - 1.0 / 0.3048) < 1e-10

    def test_feet_meters_reciprocal(self):
        """Feet/meters conversions should be reciprocals."""
        assert abs(
            constants.FEET_TO_METERS * constants.METERS_TO_FEET - 1.0
        ) < 1e-10

    def test_deg_to_rad(self):
        """Degrees to radians conversion."""
        expected = math.pi / 180.0
        assert abs(constants.DEG_TO_RAD - expected) < 1e-15

    def test_rad_to_deg(self):
        """Radians to degrees conversion."""
        expected = 180.0 / math.pi
        assert abs(constants.RAD_TO_DEG - expected) < 1e-10

    def test_deg_rad_reciprocal(self):
        """Deg/rad conversions should be reciprocals."""
        assert abs(
            constants.DEG_TO_RAD * constants.RAD_TO_DEG - 1.0
        ) < 1e-10


class TestWGS84Parameters:
    """Test WGS-84 ellipsoid parameters."""

    def test_wgs84_semi_major_axis(self):
        """WGS-84 equatorial radius."""
        assert constants.WGS84_A == 6378137.0

    def test_wgs84_semi_minor_axis(self):
        """WGS-84 polar radius."""
        assert abs(constants.WGS84_B - 6356752.314245) < 1e-6

    def test_wgs84_flattening(self):
        """WGS-84 flattening parameter."""
        f_calculated = (constants.WGS84_A - constants.WGS84_B) / constants.WGS84_A
        assert abs(constants.WGS84_F - f_calculated) < 1e-15
        # WGS-84 flattening is approximately 1/298.257223563
        assert abs(constants.WGS84_F - 1.0 / 298.257223563) < 1e-10

    def test_wgs84_first_eccentricity_squared(self):
        """WGS-84 first eccentricity squared."""
        e2_calculated = (
            (constants.WGS84_A**2 - constants.WGS84_B**2) / constants.WGS84_A**2
        )
        assert abs(constants.WGS84_E2 - e2_calculated) < 1e-15

    def test_wgs84_second_eccentricity_squared(self):
        """WGS-84 second eccentricity squared."""
        ep2_calculated = (
            (constants.WGS84_A**2 - constants.WGS84_B**2) / constants.WGS84_B**2
        )
        assert abs(constants.WGS84_EP2 - ep2_calculated) < 1e-15


class TestHelperFunctions:
    """Test helper conversion functions."""

    def test_wavelength_from_frequency_xband(self):
        """Wavelength from X-band frequency (10 GHz)."""
        wavelength = constants.wavelength_from_frequency(10e9)
        expected = constants.SPEED_OF_LIGHT / 10e9
        assert abs(wavelength - expected) < 1e-10
        # X-band wavelength should be around 3 cm
        assert abs(wavelength - 0.0299792458) < 1e-6

    def test_frequency_from_wavelength_xband(self):
        """Frequency from X-band wavelength (3 cm)."""
        frequency = constants.frequency_from_wavelength(0.03)
        expected = constants.SPEED_OF_LIGHT / 0.03
        assert abs(frequency - expected) < 1e-6
        # Should be close to 10 GHz
        assert abs(frequency - 10e9) < 1e8

    def test_wavelength_frequency_roundtrip(self):
        """Wavelength/frequency conversions should be inverses."""
        freq = 9.6e9  # 9.6 GHz (X-band)
        wavelength = constants.wavelength_from_frequency(freq)
        freq_back = constants.frequency_from_wavelength(wavelength)
        assert abs(freq - freq_back) < 1e-3

    def test_range_resolution(self):
        """Range resolution from bandwidth."""
        # 600 MHz bandwidth
        res = constants.range_resolution(600e6)
        expected = constants.TWO_WAY_SPEED_OF_LIGHT / 600e6
        assert abs(res - expected) < 1e-10
        # Should be approximately 25 cm
        assert abs(res - 0.25) < 0.01

    def test_range_resolution_high_bandwidth(self):
        """Higher bandwidth gives better resolution."""
        res_1ghz = constants.range_resolution(1e9)
        res_2ghz = constants.range_resolution(2e9)
        # 2 GHz should give half the resolution (better)
        assert abs(res_2ghz - res_1ghz / 2.0) < 1e-10

    def test_doppler_frequency_to_velocity(self):
        """Doppler shift to velocity conversion."""
        # 100 Hz Doppler at X-band (3cm wavelength)
        vel = constants.doppler_frequency_to_velocity(100.0, 0.03)
        expected = 100.0 * 0.03 / 2.0  # 1.5 m/s
        assert abs(vel - expected) < 1e-10
        assert abs(vel - 1.5) < 1e-10

    def test_doppler_velocity_zero_doppler(self):
        """Zero Doppler shift gives zero velocity."""
        vel = constants.doppler_frequency_to_velocity(0.0, 0.03)
        assert vel == 0.0

    def test_doppler_velocity_negative(self):
        """Negative Doppler indicates receding."""
        vel_pos = constants.doppler_frequency_to_velocity(100.0, 0.03)
        vel_neg = constants.doppler_frequency_to_velocity(-100.0, 0.03)
        assert abs(vel_pos + vel_neg) < 1e-10


class TestConstantsDictionary:
    """Test the CONSTANTS dictionary."""

    def test_constants_dict_exists(self):
        """CONSTANTS dictionary should exist."""
        assert hasattr(constants, 'CONSTANTS')
        assert isinstance(constants.CONSTANTS, dict)

    def test_constants_dict_has_speed_of_light(self):
        """Dictionary should contain speed of light."""
        assert 'SPEED_OF_LIGHT' in constants.CONSTANTS
        assert constants.CONSTANTS['SPEED_OF_LIGHT'] == constants.SPEED_OF_LIGHT

    def test_constants_dict_has_wgs84(self):
        """Dictionary should contain WGS-84 parameters."""
        assert 'WGS84_A' in constants.CONSTANTS
        assert 'WGS84_E2' in constants.CONSTANTS

    def test_constants_dict_values_match(self):
        """Dictionary values should match module constants."""
        for key, value in constants.CONSTANTS.items():
            module_value = getattr(constants, key)
            assert value == module_value


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
