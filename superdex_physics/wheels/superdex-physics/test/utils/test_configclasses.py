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

from superdex.physics.utils.configclasses import configclass

########################################################################################


class TestConfigClasses(unittest.TestCase):
    """Test class for configclass decorator functionality."""

    def test_configclass_decorator(self):
        """Test that the configclass decorator enforces keyword-only arguments."""

        @configclass
        class TestConfig:
            field1: int
            field2: str = "default"

        # Test initialization with keyword arguments (should work)
        config = TestConfig(field1=42, field2="test")
        assert config.field1 == 42
        assert config.field2 == "test"

        # Test initialization with default values
        config = TestConfig(field1=10)
        assert config.field1 == 10
        assert config.field2 == "default"

        # Test initialization without keyword arguments (should fail)
        with self.assertRaises(TypeError):
            config = TestConfig(42, "test")

    def test_configclass_is_dataclass(self):
        """Test that configclass creates a valid dataclass."""

        @configclass
        class TestConfig:
            field1: int
            field2: str = "default"

        # Verify it's a dataclass
        assert dataclasses.is_dataclass(TestConfig)

        # Test dataclass functionality
        config = TestConfig(field1=42)
        config_dict = dataclasses.asdict(config)
        assert config_dict == {"field1": 42, "field2": "default"}

    def test_configclass_inheritance(self):
        """Test that configclass works with inheritance."""

        @configclass
        class BaseConfig:
            base_field: int

        @configclass
        class DerivedConfig(BaseConfig):
            derived_field: str

        # Test initialization with all required fields
        config = DerivedConfig(base_field=10, derived_field="test")
        assert config.base_field == 10
        assert config.derived_field == "test"

        # Missing required fields should fail
        with self.assertRaises(TypeError):
            config = DerivedConfig(derived_field="test")


########################################################################################

if __name__ == "__main__":
    unittest.main()
