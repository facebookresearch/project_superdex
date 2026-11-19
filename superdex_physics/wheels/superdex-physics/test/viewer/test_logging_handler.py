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

from superdex.physics.viewer.logging_handler import LoggingHandler


class TestConsumeNewErrors(unittest.TestCase):
    def setUp(self) -> None:
        self.handler = LoggingHandler(max_messages=4)
        self.logger = logging.getLogger(f"{__name__}.{self.id()}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        self.logger.addHandler(self.handler)

    def tearDown(self) -> None:
        self.logger.removeHandler(self.handler)

    def test_reports_each_error_once(self) -> None:
        self.logger.info("ready")
        self.logger.warning("slow")
        self.assertFalse(self.handler.consume_new_errors())
        self.logger.error("diverged")
        self.assertTrue(self.handler.consume_new_errors())
        self.assertFalse(self.handler.consume_new_errors())

    def test_critical_counts_as_an_error(self) -> None:
        self.logger.critical("crashed")
        self.assertTrue(self.handler.consume_new_errors())

    def test_an_error_already_dropped_by_the_buffer_is_still_reported(self) -> None:
        self.logger.error("diverged")
        for index in range(10):  # More than the buffer keeps.
            self.logger.info("step %d", index)
        self.assertTrue(self.handler.consume_new_errors())

    def test_clearing_drops_an_unreported_error(self) -> None:
        self.logger.error("diverged")
        self.handler.clear()
        self.assertFalse(self.handler.consume_new_errors())
