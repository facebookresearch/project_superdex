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

import contextlib
import pathlib
import sys
import tempfile
import unittest
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from unittest import mock

from apps.envs import benchmark, run_sample

_ENV_ID = "superdex_gym/CartPole-v0"


class _ActionSpace:
    def sample(self) -> int:
        return 0


class _FakeEnv:
    def __init__(self, events: list[str], *, fail_reset: bool = False) -> None:
        self.action_space = _ActionSpace()
        self.events = events
        self.fail_reset = fail_reset
        self.closed = False

    def get_simulation_frequency(self) -> int:
        self.events.append("simulation_frequency")
        return 100

    def get_control_frequency(self) -> int:
        self.events.append("control_frequency")
        return 20

    def call(self, name: str) -> tuple[int, ...]:
        return (getattr(self, name)(),)

    def reset(self) -> None:
        self.events.append("reset")
        if self.fail_reset:
            raise RuntimeError("reset failed")

    def step(self, action: int) -> None:
        self.events.append(f"step:{action}")

    def close(self) -> None:
        self.events.append("close")
        self.closed = True

    def __enter__(self) -> "_FakeEnv":
        return self

    def __exit__(self, *ignored: object) -> bool:
        self.close()
        return False


class _FakeSection:
    def __init__(self) -> None:
        self.mean = 10.0
        self.info: dict[str, object] = {}


class _FakeProfiler:
    def __init__(self) -> None:
        self.sections: dict[str, _FakeSection] = {}

    def enter(self, name: str) -> contextlib.AbstractContextManager[None]:
        self.sections.setdefault(name, _FakeSection())
        return contextlib.nullcontext()

    def print_summary(self) -> None:
        pass


class _FakeWork:
    def __init__(self, values: list[tuple[int, int]]) -> None:
        self.values = values

    def __iter__(self) -> Iterator[tuple[int, int]]:
        return iter(self.values)

    def set_description(self, description: str) -> None:
        del description


@contextlib.contextmanager
def _patched_benchmark(
    temp_dir: str,
    make_env: Callable[..., _FakeEnv],
    *,
    make_vector: Callable[..., _FakeEnv] | None = None,
    now: Callable[[], float] | None = None,
) -> Iterator[None]:
    with contextlib.ExitStack() as stack:
        patchers = [
            mock.patch.object(
                benchmark, "__file__", str(pathlib.Path(temp_dir) / "benchmark.py")
            ),
            mock.patch.object(benchmark.gym, "make", side_effect=make_env),
            mock.patch.object(
                benchmark, "unwrap_mochi_env", side_effect=lambda env: env
            ),
            mock.patch.object(benchmark, "Profiler", _FakeProfiler),
            mock.patch.object(benchmark, "tqdm", side_effect=_FakeWork),
            mock.patch.object(benchmark, "get_logical_cpu_count", return_value=8),
            mock.patch.object(benchmark, "get_used_ram", return_value=1_000.0),
            mock.patch.object(benchmark, "get_total_ram", return_value=10_000.0),
        ]
        if make_vector is not None:
            patchers.append(
                mock.patch.object(benchmark, "HybridVectorEnv", side_effect=make_vector)
            )
        if now is not None:
            patchers.append(mock.patch.object(benchmark.time, "time", side_effect=now))
        for patcher in patchers:
            stack.enter_context(patcher)
        yield


class BenchmarkTest(unittest.TestCase):
    def _run_benchmark(
        self,
        envs: list[_FakeEnv],
        events: list[str],
        topologies: list[tuple[int, int]],
    ) -> None:
        clock = iter(range(100))

        def now() -> float:
            events.append("time")
            return float(next(clock))

        def make_env(env_id: str, **cfg: object) -> _FakeEnv:
            self.assertEqual(_ENV_ID, env_id)
            self.assertEqual(0, cfg["num_worker_threads"])
            events.append("construct")
            env = _FakeEnv(events)
            envs.append(env)
            return env

        def make_vector(env_fns: list[object], num_envs_per_worker: int) -> _FakeEnv:
            self.assertGreater(num_envs_per_worker, 0)
            events.append("vector_construct")
            dummy_env = env_fns[0]()
            dummy_env.close()
            env = _FakeEnv(events)
            envs.append(env)
            return env

        with tempfile.TemporaryDirectory() as temp_dir:
            with _patched_benchmark(
                temp_dir, make_env, make_vector=make_vector, now=now
            ):
                benchmark.run_benchmark(
                    _ENV_ID,
                    topologies,
                    min_iterations=1,
                    max_iterations=1,
                    min_time=0.0,
                    max_time=0.0,
                    write_to_file=False,
                )

    def test_uses_first_measured_environment_for_frequencies(self) -> None:
        events: list[str] = []
        envs: list[_FakeEnv] = []

        self._run_benchmark(envs, events, [(1, 1), (1, 1)])

        self.assertEqual(2, events.count("construct"))
        self.assertEqual(1, events.count("simulation_frequency"))
        self.assertEqual(1, events.count("control_frequency"))
        time_indexes = [index for index, event in enumerate(events) if event == "time"]
        self.assertLess(time_indexes[0], events.index("construct"))
        self.assertLess(time_indexes[1], events.index("simulation_frequency"))
        self.assertEqual(2, events.count("close"))
        self.assertTrue(all(env.closed for env in envs))

    def test_vector_first_uses_measured_dummy_for_frequencies(self) -> None:
        events: list[str] = []
        envs: list[_FakeEnv] = []

        self._run_benchmark(envs, events, [(2, 1)])

        self.assertEqual(1, events.count("construct"))
        self.assertEqual(1, events.count("simulation_frequency"))
        time_indexes = [index for index, event in enumerate(events) if event == "time"]
        self.assertLess(time_indexes[0], events.index("vector_construct"))
        self.assertLess(events.index("vector_construct"), events.index("construct"))
        self.assertLess(time_indexes[1], events.index("simulation_frequency"))
        self.assertEqual(2, events.count("close"))
        self.assertTrue(all(env.closed for env in envs))

    def test_closes_measured_environment_when_reset_fails(self) -> None:
        events: list[str] = []
        env = _FakeEnv(events, fail_reset=True)

        def make_env(*ignored: object, **also_ignored: object) -> _FakeEnv:
            return env

        with tempfile.TemporaryDirectory() as temp_dir:
            with _patched_benchmark(temp_dir, make_env):
                with self.assertRaisesRegex(RuntimeError, "reset failed"):
                    benchmark.run_benchmark(
                        _ENV_ID,
                        [(1, 1)],
                        min_iterations=0,
                        max_iterations=0,
                        min_time=0.0,
                        max_time=0.0,
                        write_to_file=False,
                    )

        self.assertTrue(env.closed)
        self.assertEqual(1, events.count("close"))

    def test_main_forwards_canonical_environment_id(self) -> None:
        argv = [
            "benchmark.py",
            "--env",
            _ENV_ID,
            "--num_workers",
            "1",
            "--num_envs_per_worker",
            "1",
        ]
        with (
            mock.patch.object(
                benchmark,
                "get_env_specs",
                return_value=(SimpleNamespace(id=_ENV_ID),),
            ),
            mock.patch.object(benchmark, "get_physical_core_count", return_value=None),
            mock.patch.object(benchmark, "run_benchmark") as run_benchmark,
            mock.patch.object(benchmark.sys, "argv", argv),
        ):
            benchmark.main()

        self.assertEqual(_ENV_ID, run_benchmark.call_args.args[0])


class RunSampleTest(unittest.TestCase):
    def test_main_forwards_canonical_environment_id(self) -> None:
        argv = ["run_sample.py", _ENV_ID, "--num_episodes", "1"]
        with (
            mock.patch.object(
                run_sample,
                "get_env_specs",
                return_value=(SimpleNamespace(id=_ENV_ID),),
            ),
            mock.patch.object(run_sample, "configure_logger"),
            mock.patch.object(run_sample, "run_sample") as run,
            mock.patch.object(sys, "argv", argv),
        ):
            run_sample.main()

        self.assertEqual(_ENV_ID, run.call_args.kwargs["env_id"])
