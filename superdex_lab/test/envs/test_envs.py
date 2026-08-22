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

"""Instantiate-and-step smoke tests for every discovered environment.

This is the auto-discovering successor to the hand-maintained per-env smoke tests: one
test is generated per discovered env and per gym config variant, so adding an env or a
variant needs no change here, and an env absent from a build simply produces no test
rather than needing a per-environment gate.

Test-only variants are included deliberately -- being crash-checked here is the entire
reason they exist (see :attr:`superdex.lab.gym.utils.env_discovery.EnvEntry.test_only`).

This file covers only "does it run". The discovery mechanism itself is tested in
``test_env_discovery``, shared-scene and lifetime behavior in ``test_scene_sharing``, and
env-specific reward/termination semantics in a test module sitting next to the env
itself.
"""

import time
import unittest
from typing import Callable

from superdex.lab.gym.utils.env_discovery import discover_envs, EnvEntry

########################################################################################

_MAX_STEPS = 20
_MAX_TIME = 5.0


class TestEnvs(unittest.TestCase):
    """Instantiate-and-step smoke tests, generated from environment discovery."""

    def test_discovery_is_non_empty(self) -> None:
        """Guards the generated tests below: an empty discovery would otherwise produce
        zero test methods and pass silently."""
        self.assertTrue(discover_envs(), "env discovery found no environments")


def _run_env(env) -> None:
    """Reset and step an environment a few times (bounded by time)."""
    with env:
        env.reset()
        start_time = time.time()
        for _ in range(_MAX_STEPS):
            env.step(env.action_space.sample())
            if time.time() - start_time > _MAX_TIME:
                break


def _make_entry_test(entry: EnvEntry) -> Callable[[TestEnvs], None]:
    def test(self: TestEnvs) -> None:
        _run_env(entry.make_env())

    test.__doc__ = f"Instantiate and step {entry.env_id}."
    return test


def _register_generated_tests() -> None:
    for entry in discover_envs():
        setattr(TestEnvs, f"test_{entry.short_name}", _make_entry_test(entry))


_register_generated_tests()

########################################################################################

if __name__ == "__main__":
    unittest.main()
