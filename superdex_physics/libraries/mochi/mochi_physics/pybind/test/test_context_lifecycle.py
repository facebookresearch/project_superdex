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


class ContextLifecycleTest(MochiTestBase):
    """Covers the physics module's context lifecycle invariants.

    The ordered dependent-context teardown now lives in the shared pybind-core C++ library
    (RegisterContextDependent / RunContextDependentTeardowns), covered by the C++
    `pybind_helpers_test`. The real dependent-teardown-before-context-destroy path is covered by
    `superdex_robotics/pybind/test/internal/test_bot_scene.py::RoboticsContextTeardownTest`. These tests only
    exercise the Python-visible lifecycle surface: initialize / shutdown / is_initialized.
    """

    def tearDown(self) -> None:
        # Leave the context initialized for the rest of the (bundled) test session; the
        # conftest initializes it once at import and other tests rely on it.
        if not mochi.is_initialized():
            mochi.initialize(num_worker_threads=0)

    def test_shutdown_leaves_context_uninitialized(self) -> None:
        if not mochi.is_initialized():
            mochi.initialize(num_worker_threads=0)

        mochi.shutdown()

        self.assertFalse(mochi.is_initialized())

    def test_initialize_requires_shutdown_first(self) -> None:
        # initialize() performs no dependent teardown by design: InitGlobalContext raises
        # if a context already exists, so re-initializing requires shutdown() first (which
        # drains dependents). This documents/guards that invariant.
        if not mochi.is_initialized():
            mochi.initialize(num_worker_threads=0)
        with self.assertRaisesRegex(RuntimeError, "already been initialized"):
            mochi.initialize(num_worker_threads=0)

    def test_reinitialize_after_shutdown_works(self) -> None:
        if not mochi.is_initialized():
            mochi.initialize(num_worker_threads=0)

        mochi.shutdown()
        self.assertFalse(mochi.is_initialized())

        mochi.initialize(num_worker_threads=0)
        self.assertTrue(mochi.is_initialized())
