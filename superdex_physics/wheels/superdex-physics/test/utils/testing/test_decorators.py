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

from superdex.physics.utils.testing.decorators import skip, skip_if
from superdex.physics.utils.testing.environment import IS_INTERNAL_CI

########################################################################################


class TestDecorators(unittest.TestCase):
    """Test class for testing functionality."""

    @skip_if(IS_INTERNAL_CI, "Unable to test in internal CI")
    def test_skip_decorator(self):
        """Test that the skip decorator skips tests."""

        @skip("Skipping for testing")
        def test_skipped():
            self.fail("This test should be skipped")

        # Run the test and verify that it was skipped.
        with self.assertRaises(unittest.SkipTest):
            test_skipped()

    @skip_if(IS_INTERNAL_CI, "Unable to test in internal CI")
    def test_skip_if_decorator(self):
        """Test that the skip_if decorator skips tests based on a condition."""

        @skip_if(True, "Skipping for testing")
        def test_skipped():
            self.fail("This test should be skipped")

        # Run the test and verify that it was skipped.
        with self.assertRaises(unittest.SkipTest):
            test_skipped()

        @skip_if(False, "Skipping for testing")
        def test_not_skipped():
            pass

        # Run the test and verify that it was not skipped.
        test_not_skipped()


########################################################################################

if __name__ == "__main__":
    unittest.main()
