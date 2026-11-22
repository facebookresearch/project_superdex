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

import subprocess
import sys
import unittest

from superdex import physics as mochi


class MochiPhysicsFreeThreadedTest(unittest.TestCase):
    def tearDown(self):
        if mochi.is_initialized():
            mochi.shutdown()

    def test_free_threaded_interpreter(self):
        self.assertTrue(hasattr(sys, "_is_gil_enabled"))
        self.assertFalse(sys._is_gil_enabled())

    def test_initialized_module_exits_cleanly(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys\n"
                "from superdex import physics as mochi\n"
                "mochi.initialize(num_worker_threads=0)\n"
                "assert mochi.is_initialized()\n"
                "assert not sys._is_gil_enabled()\n",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("Exception ignored in atexit callback", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
