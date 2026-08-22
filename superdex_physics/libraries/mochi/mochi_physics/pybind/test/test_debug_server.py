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

from test.conftest import mochi, MochiTestBase


class TestDebugServer(MochiTestBase):
    def test_start_and_stop(self):
        server = mochi.get_debug_server()

        # The server does not start automatically.
        self.assertFalse(server.has_started())
        self.assertFalse(server.has_connection())

        # Attempt to start the server using the default port. Internally, it will attempt
        # to bind that TCP port, or the next available port. This attempt may fail entirely
        # on CI machines with restricted privileges, but the attempt to start should still
        # be safe.
        server.start()

        # has_started() returns true because start() was called, even if the underlying TCP
        # socket can't actually accept connections on this machine.
        self.assertTrue(server.has_started())

        # We can call get_port(). It might return a valid port, or it might return zero on
        # failure. We don't assert the value to avoid failures on CI.
        self.assertIsInstance(server.get_port(), int)
        self.assertFalse(server.has_connection())

        # Stop
        server.stop()
        self.assertFalse(server.has_started())
        self.assertFalse(server.has_connection())
