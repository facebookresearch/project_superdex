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

import inspect
import logging
import unittest

import superdex.physics as sdp
from superdex.physics import LogChannel
from superdex.physics.utils.logging import forward_mochi_logs_to_logger
from superdex.physics.utils.testing.testcases import MochiContextTestCase


########################################################################################


class TestLogging(MochiContextTestCase):
    """Test class for logging functionality."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Set up a test logger with a handler to capture log records.
        self.log_records = []
        self.test_handler = logging.Handler()
        self.test_handler.emit = lambda record: self.log_records.append(record)
        self.logger = logging.getLogger("mochi")
        self.logger.addHandler(self.test_handler)
        self.logger.setLevel(logging.DEBUG)

    def tearDown(self):
        """Clean up after each test method."""
        sdp.set_log_callback(None)
        self.logger.removeHandler(self.test_handler)

    def test_log_callback(self):
        """Test that log_callback correctly handles the level logs."""

        # Set the log callback to forward logs to the test logger.
        forward_mochi_logs_to_logger(self.logger)

        # Call log_callback directly with different channels.
        # Note verbose disabled by default so we don't test it.
        current_line = inspect.currentframe().f_lineno
        sdp.log("Test info message", LogChannel.INFO)
        sdp.log("Test warning message", LogChannel.WARNING)
        sdp.log("Test error message", LogChannel.ERROR)

        # Verify the log records were created correctly.
        self.assertEqual(len(self.log_records), 3)

        # Verify the contents of the records.
        record = self.log_records[0]
        self.assertEqual(record.levelno, logging.INFO)
        self.assertEqual(record.msg, "Test info message")
        self.assertEqual(record.mochi_lineno, current_line + 1)

        record = self.log_records[1]
        self.assertEqual(record.levelno, logging.WARNING)
        self.assertEqual(record.msg, "Test warning message")
        self.assertEqual(record.mochi_lineno, current_line + 2)

        record = self.log_records[2]
        self.assertEqual(record.levelno, logging.ERROR)
        self.assertEqual(record.msg, "Test error message")
        self.assertEqual(record.mochi_lineno, current_line + 3)


########################################################################################

if __name__ == "__main__":
    unittest.main()
