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
import json
import unittest

import numpy as np
from superdex.physics.utils.json import ExtendedJSONEncoder

########################################################################################


@dataclasses.dataclass
class DataClass:
    field1: int
    field2: str
    field3: list


class TestExtendedJSONEncoder(unittest.TestCase):
    """Test class for ExtendedJSONEncoder functionality."""

    def test_extended_json_encoder_with_numpy_array(self):
        """Test that ExtendedJSONEncoder correctly encodes numpy arrays."""

        # Create a numpy array
        array = np.array([1, 2, 3, 4, 5])

        # Encode using the ExtendedJSONEncoder
        encoded = json.dumps({"data": array}, cls=ExtendedJSONEncoder)

        # Decode and verify
        decoded = json.loads(encoded)
        assert decoded["data"] == [1, 2, 3, 4, 5]

    def test_extended_json_encoder_with_numpy_scalar(self):
        """Test that ExtendedJSONEncoder correctly encodes numpy scalar values."""

        # Create numpy scalar values
        int_scalar = np.int32(42)
        float_scalar = np.float32(3.14)

        # Encode using the ExtendedJSONEncoder
        encoded = json.dumps(
            {"int": int_scalar, "float": float_scalar}, cls=ExtendedJSONEncoder
        )

        # Decode and verify
        decoded = json.loads(encoded)
        assert decoded["int"] == 42
        assert np.isclose(decoded["float"], 3.14)

    def test_extended_json_encoder_with_dataclass(self):
        """Test that ExtendedJSONEncoder correctly encodes dataclasses."""

        # Create a dataclass instance
        data = DataClass(field1=42, field2="test", field3=[1, 2, 3])

        # Encode using the ExtendedJSONEncoder
        encoded = json.dumps(data, cls=ExtendedJSONEncoder)

        # Decode and verify
        decoded = json.loads(encoded)
        assert decoded["field1"] == 42
        assert decoded["field2"] == "test"
        assert decoded["field3"] == [1, 2, 3]

    def test_extended_json_encoder_with_nested_structures(self):
        """Test that ExtendedJSONEncoder correctly encodes nested structures."""

        # Create a complex nested structure
        nested_data = {
            "array": np.array([1, 2, 3]),
            "dataclass": DataClass(
                field1=42, field2="test", field3=[np.int64(1), np.float32(2.5)]
            ),
            "mixed": [np.array([4, 5]), {"value": np.float64(3.14)}],
        }

        # Encode using the ExtendedJSONEncoder
        encoded = json.dumps(nested_data, cls=ExtendedJSONEncoder)

        # Decode and verify
        decoded = json.loads(encoded)
        assert decoded["array"] == [1, 2, 3]
        assert decoded["dataclass"]["field1"] == 42
        assert decoded["dataclass"]["field2"] == "test"
        assert decoded["dataclass"]["field3"] == [1, 2.5]
        assert decoded["mixed"][0] == [4, 5]
        assert decoded["mixed"][1]["value"] == 3.14

    def test_extended_json_encoder_with_unsupported_type(self):
        """Test that ExtendedJSONEncoder raises TypeError for unsupported types."""

        # Create an object of an unsupported type
        class UnsupportedType:
            pass

        unsupported = UnsupportedType()

        # Encoding should raise TypeError
        with self.assertRaises(TypeError):
            json.dumps(unsupported, cls=ExtendedJSONEncoder)


########################################################################################

if __name__ == "__main__":
    unittest.main()
