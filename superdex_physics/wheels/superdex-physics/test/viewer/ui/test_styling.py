# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
from superdex.physics.viewer.ui.styling import (
    adjust_color,
    color_to_u32,
    make_color_rgb,
    make_color_rgb_from_u8,
    make_color_rgba,
    make_color_rgba_from_u8,
    replace_color_alpha,
    u32_to_color,
)

########################################################################################


class TestColorConstruction(unittest.TestCase):
    """Test class for color construction functions."""

    def test_float_color_construction(self):
        """Test RGB and RGBA color creation from float components."""
        # RGB with typical values
        rgb = make_color_rgb(0.5, 0.25, 0.75)
        assert np.allclose(rgb, [0.5, 0.25, 0.75])
        assert len(rgb) == 3

        # RGBA with explicit alpha
        rgba = make_color_rgba(0.5, 0.25, 0.75, 0.8)
        assert np.allclose(rgba, [0.5, 0.25, 0.75, 0.8])
        assert len(rgba) == 4

        # RGBA with default alpha
        rgba_default = make_color_rgba(0.5, 0.25, 0.75)
        assert np.allclose(rgba_default, [0.5, 0.25, 0.75, 1.0])

        # Boundary values
        assert np.allclose(make_color_rgb(0.0, 0.0, 0.0), [0.0, 0.0, 0.0])
        assert np.allclose(make_color_rgb(1.0, 1.0, 1.0), [1.0, 1.0, 1.0])

    def test_u8_color_construction(self):
        """Test RGB and RGBA color creation from 8-bit integer components."""
        # RGB with typical values
        rgb = make_color_rgb_from_u8(128, 64, 192)
        assert np.allclose(rgb, [128 / 255, 64 / 255, 192 / 255])
        assert len(rgb) == 3

        # RGBA with explicit alpha
        rgba = make_color_rgba_from_u8(128, 64, 192, 200)
        assert np.allclose(rgba, [128 / 255, 64 / 255, 192 / 255, 200 / 255])
        assert len(rgba) == 4

        # RGBA with default alpha
        rgba_default = make_color_rgba_from_u8(128, 64, 192)
        assert rgba_default[3] == 1.0

        # Boundary values
        assert np.allclose(make_color_rgb_from_u8(0, 0, 0), [0.0, 0.0, 0.0])
        assert np.allclose(make_color_rgb_from_u8(255, 255, 255), [1.0, 1.0, 1.0])


class TestColorConversion(unittest.TestCase):
    """Test class for color conversion functions."""

    def test_color_to_u32(self):
        """Test converting color arrays to U32 format."""
        # RGB to U32 (implicit full alpha)
        rgb = np.array([1.0, 0.5, 0.25])
        u32_rgb = color_to_u32(rgb)
        expected_rgb = (255 << 24) | (63 << 16) | (127 << 8) | 255
        assert u32_rgb == expected_rgb

        # RGBA to U32
        rgba = np.array([1.0, 0.5, 0.25, 0.5])
        u32_rgba = color_to_u32(rgba)
        expected_rgba = (127 << 24) | (63 << 16) | (127 << 8) | 255
        assert u32_rgba == expected_rgba

        # Boundary values
        black = color_to_u32(np.array([0.0, 0.0, 0.0, 1.0]))
        expected_black = 0xFF000000
        assert black == expected_black
        white = color_to_u32(np.array([1.0, 1.0, 1.0, 1.0]))
        expected_white = 0xFFFFFFFF
        assert white == expected_white

    def test_u32_to_color(self):
        """Test converting U32 format to RGBA color arrays."""
        # Typical conversion
        u32 = (204 << 24) | (63 << 16) | (127 << 8) | 255
        color = u32_to_color(u32)
        expected = make_color_rgba_from_u8(255, 127, 63, 204)
        assert np.allclose(color, expected)
        assert len(color) == 4

        # Boundary values
        assert np.allclose(u32_to_color(0), [0.0, 0.0, 0.0, 0.0])
        assert np.allclose(u32_to_color(0xFFFFFFFF), [1.0, 1.0, 1.0, 1.0])

    def test_color_u32_roundtrip(self):
        """Test that converting color to U32 and back preserves values."""
        original = make_color_rgba(0.5, 0.25, 0.75, 0.8)
        u32 = color_to_u32(original)
        recovered = u32_to_color(u32)
        # Allow small tolerance due to quantization to 8-bit
        assert np.allclose(recovered, original, atol=1 / 255)


class TestColorManipulation(unittest.TestCase):
    """Test class for color manipulation functions."""

    def test_replace_alpha(self):
        """Test replacing alpha channel on various color formats."""
        # RGB array - adds alpha
        rgb = np.array([0.5, 0.25, 0.75])
        with_alpha = replace_color_alpha(rgb, 0.6)
        assert np.allclose(with_alpha, [0.5, 0.25, 0.75, 0.6])

        # RGBA array - replaces alpha
        rgba = np.array([0.5, 0.25, 0.75, 0.9])
        new_alpha = replace_color_alpha(rgba, 0.3)
        assert np.allclose(new_alpha, [0.5, 0.25, 0.75, 0.3])

        # U32 format
        u32 = (200 << 24) | (100 << 16) | (150 << 8) | 255
        u32_new = replace_color_alpha(u32, 0.5)
        expected_u32 = (127 << 24) | (100 << 16) | (150 << 8) | 255
        assert u32_new == expected_u32
        assert isinstance(u32_new, int)

        # Boundary values
        assert replace_color_alpha(rgba, 0.0)[3] == 0.0
        assert replace_color_alpha(rgba, 1.0)[3] == 1.0

    def test_adjust_color_arrays(self):
        """Test adjusting color brightness and offset on arrays."""
        color = np.array([0.5, 0.25, 0.75, 0.8])

        # Brighten (factor > 1)
        bright = adjust_color(color, factor=2.0)
        assert np.allclose(bright, [1.0, 0.5, 1.0, 0.8])

        # Darken (factor < 1)
        dark = adjust_color(color, factor=0.5)
        assert np.allclose(dark, [0.25, 0.125, 0.375, 0.8])

        # Add offset
        offset = adjust_color(np.array([0.5, 0.25, 0.75]), factor=1.0, offset=0.1)
        assert np.allclose(offset, [0.6, 0.35, 0.85])

        # Combined factor and offset
        combined = adjust_color(np.array([0.5, 0.25, 0.75]), factor=0.8, offset=0.1)
        assert np.allclose(combined, [0.5, 0.3, 0.7])

        # Clamping negative values
        clamped = adjust_color(np.array([0.2, 0.3, 0.4]), factor=1.0, offset=-0.5)
        assert np.all(clamped >= 0.0)

        # Identity (default parameters)
        identity = adjust_color(color, factor=1.0, offset=0.0)
        assert np.allclose(identity, color)

    def test_adjust_color_u32(self):
        """Test adjusting U32 color format."""
        u32 = (255 << 24) | (100 << 16) | (150 << 8) | 200
        adjusted = adjust_color(u32, factor=1.5)

        assert isinstance(adjusted, int)

        # Verify RGB components are brighter, alpha unchanged
        adjusted_rgba = u32_to_color(adjusted)
        original_rgba = u32_to_color(u32)
        assert adjusted_rgba[0] > original_rgba[0]
        assert adjusted_rgba[1] > original_rgba[1]
        assert adjusted_rgba[2] > original_rgba[2]
        assert np.isclose(adjusted_rgba[3], original_rgba[3], atol=1 / 255)


class TestColorTypeConsistency(unittest.TestCase):
    """Test type consistency and edge cases."""

    def test_return_types(self):
        """Test that functions return expected types."""
        # Arrays return numpy arrays
        assert isinstance(make_color_rgb(0.5, 0.5, 0.5), np.ndarray)
        assert isinstance(make_color_rgba(0.5, 0.5, 0.5, 0.5), np.ndarray)

        # U32 functions return integers
        assert isinstance(color_to_u32(np.array([0.5, 0.5, 0.5])), int)

    def test_array_dimensions(self):
        """Test that color operations preserve expected dimensions."""
        assert make_color_rgb(0.5, 0.5, 0.5).shape == (3,)
        assert make_color_rgba(0.5, 0.5, 0.5, 0.5).shape == (4,)
        assert replace_color_alpha(np.array([0.5, 0.5, 0.5]), 0.8).shape == (4,)

    def test_extreme_values(self):
        """Test color operations with extreme values."""
        color = np.array([0.5, 0.5, 0.5])

        # Very bright adjustment
        bright = adjust_color(color, factor=10.0)
        assert np.all(bright >= 0.0)

        # Very dark with negative offset
        dark = adjust_color(color, factor=0.1, offset=-1.0)
        assert np.all(dark >= 0.0)


########################################################################################

if __name__ == "__main__":
    unittest.main()
