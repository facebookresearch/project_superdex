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

"""
Environment benchmarking script for SuperDex Gym.

This script provides a flexible benchmarking tool for testing environment performance
across different worker and environment configurations. It supports configurable
timing constraints, iteration limits, and flexible argument parsing for comprehensive
performance testing. The environment to benchmark is selected by its discovered short
name (see :mod:`superdex.lab.gym.utils.env_discovery`). Benchmarkable environments must accept
``num_worker_threads``, ``use_shared_scenes``, and ``render_mode`` configuration overrides,
and their config must expose numeric ``simulation_frequency`` and ``control_frequency``
attributes used to calculate simulation FPS.
"""

import argparse
import itertools
import json
import pathlib
import sys
import time

import psutil
from superdex.lab.gym.utils.env_discovery import get_env_short_names, register_all_envs
from superdex.lab.gym.utils.vector import HybridVectorEnv
from superdex.physics.utils.profiling import Profiler
from tqdm import tqdm

########################################################################################

DEFAULT_OUTPUT_FILENAME = "benchmark.json"


def get_logical_cpu_count() -> int:
    """Get the number of logical CPUs available on the system."""
    return psutil.cpu_count(logical=True) or 1


def get_physical_core_count() -> int | None:
    """Get the number of physical CPU cores available on the system."""
    return psutil.cpu_count(logical=False)


def get_used_ram() -> float:
    """Get the amount of used RAM in bytes."""
    return float(psutil.virtual_memory().used)


def get_total_ram() -> float:
    """Get the total amount of RAM in bytes."""
    return float(psutil.virtual_memory().total)


def make_section_name(num_workers: int, num_envs_per_worker: int) -> str:
    """Generate a descriptive section name for benchmark results."""
    return f"{num_workers} workers × {num_envs_per_worker} envs/worker = {num_workers * num_envs_per_worker} envs"


def run_benchmark(
    env_name: str,
    num_envs_and_envs_per_worker: list[tuple[int, int]],
    min_iterations: int,
    max_iterations: int,
    min_time: float,
    max_time: float,
    write_to_file: bool,
    output_filename: str = DEFAULT_OUTPUT_FILENAME,
):
    """Run benchmarks for all specified worker/environment combinations.

    The selected environment's config must accept the benchmark overrides and expose
    numeric ``simulation_frequency`` and ``control_frequency`` attributes. These
    frequencies are required to derive simulation FPS from the measured control FPS.
    """
    # Display system information for context.
    print(f"Logical CPUs: {get_logical_cpu_count()}")
    print(f"Memory: {1e-9 * get_used_ram():.2f} /  {1e-9 * get_total_ram()} GB")

    # Setup output directory and file path for benchmark results.
    base_path = pathlib.Path(__file__).parent.resolve()
    output_path = base_path / "output" / output_filename
    output_path.parent.mkdir(exist_ok=True)

    # Resolve the environment to benchmark from the discovery registry.
    entries = get_env_short_names()
    if env_name not in entries:
        available = ", ".join(sorted(entries))
        raise ValueError(f"Unknown env: {env_name}. Available: {available}")
    entry = entries[env_name]

    # Configure environment settings for consistent benchmarking.
    cfg = entry.cfg_cls(
        **{
            **entry.cfg_kwargs,
            "num_worker_threads": 0,
            "use_shared_scenes": True,
            "render_mode": None,
        }
    )

    def env_builder():
        """Factory function to create environment instances."""
        return entry.env_cls(cfg)

    # Initialize profiler to track performance metrics.
    profiler = Profiler()
    work = tqdm(num_envs_and_envs_per_worker)

    # Iterate through each worker/environment combination.
    for num_workers, num_envs_per_worker in work:
        num_envs = num_workers * num_envs_per_worker
        section_name = make_section_name(num_workers, num_envs_per_worker)
        work.set_description(section_name)

        # Track initialization time and memory usage.
        start_time = time.time()
        start_ram = get_used_ram()

        # Create environment(s) - single env for reference, vectorized for scaling.
        if num_envs == 1:
            env = env_builder()  # Single environment without vectorization overhead.
        else:
            # Use HybridVectorEnv for parallel environment execution.
            env = HybridVectorEnv([env_builder] * num_envs, num_envs_per_worker)
        action_space = env.action_space

        # Initialize environments and measure setup time/memory.
        env.reset()
        initialization_time = time.time() - start_time
        memory_used = get_used_ram() - start_ram
        system_wide_memory_used = 100 * get_used_ram() / get_total_ram()

        # Display initialization metrics.
        print()
        print(f"Initialization took {initialization_time} s.")
        print(f"Approx. memory used by the environments: {1e-9 * memory_used:.2f} GB")
        print(f"System-wide memory usage: {system_wide_memory_used:.2f}%")

        # Execute benchmark with timing and iteration constraints.
        with env:
            elapsed_time = 0
            start_time = time.time()
            report_interval = 1  # Report progress every second.
            next_report_time = report_interval

            steps_count = 0  # Total environment steps performed.
            iterations = 0  # Number of env.step() calls made.

            while True:
                # Apply stopping conditions based on time and iteration limits.
                if iterations >= min_iterations and elapsed_time >= min_time:
                    # Minimum requirements met, continue until max limits.
                    if iterations >= max_iterations:
                        break  # Both max iterations and min time satisfied.
                    if elapsed_time >= max_time:
                        break  # Hard time limit reached.

                # Periodic progress reporting.
                if elapsed_time >= next_report_time:
                    print(
                        f"- Performed {iterations} iterations ({steps_count} steps), "
                        f"elapsed {elapsed_time:.2f}s"
                    )
                    next_report_time = elapsed_time + report_interval

                # Execute one environment step with profiling.
                with profiler.enter(section_name):
                    env.step(action_space.sample())

                # Update counters and timing.
                iterations += 1
                steps_count += num_envs  # Each iteration steps all environments.
                elapsed_time = time.time() - start_time

        # Calculate and store performance metrics.
        section = profiler.sections[section_name]
        control_fps = (1000 * num_envs) / section.mean
        sim_fps = (cfg.simulation_frequency * control_fps) / cfg.control_frequency

        # Store comprehensive benchmark results.
        section.info["init_time"] = f"{initialization_time:.2f}s"
        section.info["sim_fps"] = f"{sim_fps:.2f}"
        section.info["control_fps"] = f"{control_fps:.2f}"
        section.info["samples"] = steps_count
        section.info["mem_usage"] = f"{1e-9 * memory_used:.2f} GB"

        # Display results and optionally save to file.
        print()
        profiler.print_summary()
        print()

        if write_to_file:
            with open(output_path, "w") as handle:
                json.dump(profiler.to_dict(extended=True), handle, indent=4)

    # Display final summary of all benchmark runs.
    profiler.print_summary()


########################################################################################


def parse_positive_int_args(args: list[str]) -> list[int]:
    """Parse positive integers from single values or comma-separated lists."""
    result = []
    for arg in args:
        for value in arg.split(","):
            try:
                parsed = int(value.strip())
            except ValueError:
                raise ValueError(f"invalid integer value: {value!r}") from None
            if parsed <= 0:
                raise ValueError(f"value must be positive: {parsed}")
            result.append(parsed)

    return result


def main():
    """Main entry point for the benchmark script."""
    # Discover and register environments so the CLI can list and build them. Test-only
    # variants are excluded, so they are not offered nor accepted here.
    register_all_envs()
    available_envs = sorted(get_env_short_names())
    if not available_envs:
        raise RuntimeError(
            "No environments were discovered, so there is nothing to benchmark. This "
            "usually means the env packages failed to import or were excluded from the "
            "build."
        )
    parser = argparse.ArgumentParser(
        description="Run environment benchmarks with configurable parameters."
    )

    env_selection = parser.add_mutually_exclusive_group(required=True)
    env_selection.add_argument(
        "--env",
        type=str,
        choices=available_envs,
        help=(f"Environment to benchmark. One of: {', '.join(available_envs)}"),
    )
    env_selection.add_argument(
        "--all",
        action="store_true",
        help="Benchmark every discovered environment.",
    )

    # Worker arguments
    parser.add_argument(
        "--num_workers",
        type=str,
        nargs="+",
        required=True,
        help=(
            "Number of workers. One or more positive integers, separated by spaces "
            "and/or commas."
        ),
    )
    parser.add_argument(
        "--num_envs_per_worker",
        type=str,
        nargs="+",
        required=True,
        help=(
            "Number of environments per worker. "
            "One or more positive integers, separated by spaces and/or commas."
        ),
    )
    parser.add_argument(
        "--min_iterations",
        type=int,
        default=10,
        help="Minimum number of batched step() calls per test case (default: 10). "
        "Each iteration steps all environments in parallel.",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=100,
        help="Maximum number of batched step() calls per test case (default: 100). "
        "Each iteration steps all environments in parallel.",
    )
    parser.add_argument(
        "--min_time",
        type=float,
        default=5.0,
        help="Minimum runtime per test case in seconds (default: 5.0). "
        "Test will continue until both min_iterations and min_time are satisfied.",
    )
    parser.add_argument(
        "--max_time",
        type=float,
        default=60.0,
        help="Maximum runtime per test case in seconds (default: 60.0). "
        "Test will stop when either max_iterations or max_time is reached, "
        "but only after min_iterations and min_time are both satisfied.",
    )
    parser.add_argument(
        "--write_to_file",
        action="store_true",
        help=(
            "Write results to JSON; --all writes one file per environment "
            "(default: False)."
        ),
    )

    args = parser.parse_args()

    # Parse worker and environment configuration arguments.
    try:
        num_workers_list = parse_positive_int_args(args.num_workers)
    except ValueError as error:
        parser.error(f"argument --num_workers: {error}")
    try:
        num_envs_per_worker_list = parse_positive_int_args(args.num_envs_per_worker)
    except ValueError as error:
        parser.error(f"argument --num_envs_per_worker: {error}")

    physical_cores = get_physical_core_count()
    if physical_cores is not None:
        excessive_worker_counts = sorted(
            {count for count in num_workers_list if count > physical_cores}
        )
        if excessive_worker_counts:
            counts = ", ".join(str(count) for count in excessive_worker_counts)
            print(
                f"Warning: requested worker counts ({counts}) above the physical CPU "
                f"core count ({physical_cores}). Workers may share cores through SMT "
                "or scheduler oversubscription.",
                file=sys.stderr,
            )

    # Generate all combinations of worker/environment configurations to test.
    benchmarks = list(itertools.product(num_workers_list, num_envs_per_worker_list))

    if args.all:
        env_names = available_envs
    else:
        assert args.env is not None
        env_names = [args.env]

    for env_name in env_names:
        # Display planned benchmark configurations.
        print(f"Benchmarking env '{env_name}' for {len(benchmarks)} combinations:")
        for num_workers, num_envs_per_worker in benchmarks:
            print(f"- {make_section_name(num_workers, num_envs_per_worker)}")
        print()

        # Execute benchmarks with specified timing and iteration constraints.
        run_benchmark(
            env_name,
            benchmarks,
            min_iterations=args.min_iterations,
            max_iterations=args.max_iterations,
            min_time=args.min_time,
            max_time=args.max_time,
            write_to_file=args.write_to_file,
            output_filename=(
                f"benchmark_{env_name}.json" if args.all else DEFAULT_OUTPUT_FILENAME
            ),
        )


if __name__ == "__main__":
    main()
