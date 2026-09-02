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

import dataclasses
import unittest
import warnings

from superdex.physics.utils.deprecation import deprecated, DeprecatedField

########################################################################################


class TestDeprecation(unittest.TestCase):
    """Test class for deprecation functionality."""

    def test_deprecated_decorator(self):
        """Test that the deprecated decorator issues warnings."""

        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            @deprecated("Use NewClass instead")
            class OldClass:
                pass

            # Instantiate the deprecated class to trigger warning
            _ = OldClass()

            # Check that a warning was raised
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "Use NewClass instead" in str(w[0].message)

    def test_deprecated_field(self):
        """Test that DeprecatedField issues warnings when accessed or modified."""

        @dataclasses.dataclass
        class TestClass:
            normal_field: int = 0
            deprecated_field: int = DeprecatedField(42, "Use normal_field instead")

        # Test initialization with default value (should not warn)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            instance = TestClass()
            assert len(w) == 0

        # Test initialization with non-default value (should warn)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            instance = TestClass(deprecated_field=10)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "Field 'deprecated_field' is deprecated" in str(w[0].message)
            assert "Use normal_field instead" in str(w[0].message)

        # Test accessing the field (should warn)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            value = instance.deprecated_field
            assert value == 10  # Value should be correctly retrieved
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "Field 'deprecated_field' is deprecated" in str(w[0].message)

        # Test modifying the field (should warn)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            instance.deprecated_field = 20
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "Field 'deprecated_field' is deprecated" in str(w[0].message)

        # Verify the value was correctly set
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert instance.deprecated_field == 20

    def test_deprecated_field_custom_warning(self):
        """Test DeprecatedField with custom warning category."""

        class CustomWarning(Warning):
            pass

        @dataclasses.dataclass
        class TestClass:
            field: int = DeprecatedField(0, category=CustomWarning)

        # Test that the custom warning category is used
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = TestClass(field=10)
            assert len(w) == 1
            assert issubclass(w[0].category, CustomWarning)

    def test_deprecated_field_default_value(self):
        """Test that DeprecatedField returns the default value when not set."""

        @dataclasses.dataclass
        class TestClass:
            field: int = DeprecatedField(42)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            instance = TestClass()
            assert instance.field == 42


########################################################################################

if __name__ == "__main__":
    unittest.main()
