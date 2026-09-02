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

import logging
import unittest
from unittest.mock import MagicMock, patch

import superdex.physics as sdp
from superdex.physics.utils.context import MochiContext


########################################################################################


class TestMochiContext(unittest.TestCase):
    """Test class for MochiContext context manager."""

    @patch("superdex.physics.utils.context.sdp")
    def test_context_initializes_and_shuts_down(self, mock_sdp: MagicMock):
        """Test that MochiContext properly initializes and shuts down physics."""
        with MochiContext():
            mock_sdp.initialize.assert_called_once_with(num_worker_threads=-1)

        mock_sdp.shutdown.assert_called_once()

    @patch("superdex.physics.utils.context.sdp")
    def test_context_with_custom_worker_threads(self, mock_sdp: MagicMock):
        """Test that MochiContext respects custom num_worker_threads."""
        with MochiContext(num_worker_threads=4):
            mock_sdp.initialize.assert_called_once_with(num_worker_threads=4)

    @patch("superdex.physics.utils.context.sdp")
    def test_context_returns_self(self, mock_sdp: MagicMock):
        """Test that __enter__ returns the MochiContext instance."""
        ctx = MochiContext()
        with ctx as entered:
            self.assertIs(entered, ctx)

    @patch("superdex.physics.utils.context.sdp")
    def test_context_shuts_down_on_exception(self, mock_sdp: MagicMock):
        """Test that MochiContext shuts down even when an exception occurs."""
        with self.assertRaises(RuntimeError):
            with MochiContext():
                raise RuntimeError("Test exception")

        mock_sdp.shutdown.assert_called_once()

    @patch("superdex.physics.utils.context.forward_mochi_logs_to_logger")
    @patch("superdex.physics.utils.context.configure_logger")
    @patch("superdex.physics.utils.context.sdp")
    def test_context_configures_logging_when_logger_provided(
        self,
        mock_sdp: MagicMock,
        mock_configure_logger: MagicMock,
        mock_forward_logs: MagicMock,
    ):
        """Test that MochiContext configures logging when a logger is provided."""
        test_logger = logging.getLogger("test_mochi")

        with MochiContext(logger=test_logger):
            mock_configure_logger.assert_called_once()
            mock_forward_logs.assert_called_once_with(test_logger)

    @patch("superdex.physics.utils.context.forward_mochi_logs_to_logger")
    @patch("superdex.physics.utils.context.configure_logger")
    @patch("superdex.physics.utils.context.sdp")
    def test_context_does_not_configure_logging_without_logger(
        self,
        mock_sdp: MagicMock,
        mock_configure_logger: MagicMock,
        mock_forward_logs: MagicMock,
    ):
        """Test that MochiContext does not configure logging when no logger is provided."""
        with MochiContext():
            mock_configure_logger.assert_not_called()
            mock_forward_logs.assert_not_called()


########################################################################################


class TestMochiContextIntegration(unittest.TestCase):
    """Integration tests for MochiContext with actual Mochi initialization."""

    def test_context_integration(self):
        """Test MochiContext with actual Mochi initialization."""
        self.assertFalse(sdp.is_initialized())

        with MochiContext(num_worker_threads=0):
            self.assertTrue(sdp.is_initialized())

        self.assertFalse(sdp.is_initialized())


########################################################################################

if __name__ == "__main__":
    unittest.main()
