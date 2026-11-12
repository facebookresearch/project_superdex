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

from apps.envs.sample_runner import sample_runner


class _FakeEnv:
    """Ends an episode every ``episode_length`` steps and records closing."""

    def __init__(self, episode_length: int) -> None:
        self.episode_length = episode_length
        self.steps = 0
        self.resets = 0
        self.closed = False

    def reset(self) -> None:
        self.resets += 1

    def step(self, _action: object) -> tuple[None, float, bool, bool, dict[str, str]]:
        self.steps += 1
        done = self.steps % self.episode_length == 0
        return None, 1.0, done, False, {}

    def close(self) -> None:
        self.closed = True


class _FakeMochiEnv:
    """Reports a close request once the env has taken ``close_after_steps`` steps."""

    def __init__(self, env: _FakeEnv, close_after_steps: int | None) -> None:
        self.env = env
        self.close_after_steps = close_after_steps

    def user_requested_close(self) -> bool:
        return (
            self.close_after_steps is not None
            and self.env.steps >= self.close_after_steps
        )


def _run(close_after_steps: int | None, num_episodes: int = 10) -> _FakeEnv:
    env = _FakeEnv(episode_length=5)
    sample_runner(
        env,
        _FakeMochiEnv(env, close_after_steps),
        action_sampler=lambda _env, _mochi_env: 0,
        num_episodes=num_episodes,
    )
    return env


class TestSampleRunner(unittest.TestCase):
    def test_runs_all_episodes_without_close_request(self) -> None:
        env = _run(close_after_steps=None, num_episodes=3)
        self.assertEqual(15, env.steps)
        self.assertTrue(env.closed)

    def test_stops_and_closes_when_user_closes_window(self) -> None:
        env = _run(close_after_steps=7)
        self.assertEqual(7, env.steps)
        self.assertTrue(env.closed)

    def test_close_request_before_first_step_runs_nothing(self) -> None:
        env = _run(close_after_steps=0)
        self.assertEqual(0, env.steps)
        self.assertEqual(1, env.resets)
        self.assertTrue(env.closed)
