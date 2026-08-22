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

"""Sample environment runner for SuperDex Gym environments.

This script provides a command-line interface for running any environment discovered by
:mod:`superdex.lab.gym.utils.env_discovery`, including JSON config variants. The set of
available samples is determined automatically, so environments absent from the current
build simply do not appear.

Usage:
    python run_sample.py <sample_name> [options]

Examples:
    python run_sample.py cart_pole --action_sampler random --num_episodes 5
    python run_sample.py half_cheetah --action_sampler sweep --video
"""

import argparse
import cProfile
import pathlib
import warnings
from typing import Any

from superdex.lab.gym.envs.mochi_env import MochiEnv
from superdex.lab.gym.utils.env_discovery import get_env_short_names, register_all_envs
from superdex.physics.utils.logging import configure_logger
from superdex.physics.viewer import VIEWER_AVAILABLE
from superdex.physics.viewer.utils import AnimationWriter

# Support both direct-script and package-module execution.
try:
    from sample_runner import random_action, sample_runner, sweep_action, zero_action
except ImportError:
    from .sample_runner import random_action, sample_runner, sweep_action, zero_action

########################################################################################


def make_env(sample: str, common_env_cfg: dict[str, Any]) -> MochiEnv:
    """Build a discovered environment (or config variant) by its short name.

    The variant's JSON config kwargs are applied first, then overridden by the shared
    ``common_env_cfg``, which carries only runtime settings the CLI owns (render mode,
    render size, start paused, profiling) -- never task configuration.
    """
    entries = get_env_short_names()
    if sample not in entries:
        available = ", ".join(sorted(entries))
        raise ValueError(f"Unknown sample: {sample}. Available: {available}")
    entry = entries[sample]
    cfg = entry.cfg_cls(**{**entry.cfg_kwargs, **common_env_cfg})
    return entry.env_cls(cfg)


########################################################################################


def run_sample(
    sample: str,
    action_sampler: str,
    num_episodes: int,
    video_size: str | None,
    video_path: pathlib.Path | None,
    start_paused: bool,
    profile: bool,
):
    """Run a sample environment with the specified configuration and action sampler.

    This function creates an environment based on the sample name, selects an action
    sampling strategy, and runs the specified number of episodes while profiling
    the execution performance.
    """

    # Check if video rendering is enabled and determine the render mode.
    if VIEWER_AVAILABLE:
        render_mode = "human" if video_path is None else "rgb_array"
    else:
        warnings.warn(
            "Polyscope is not installed in the current environment, or it's an "
            "incompatible version. Please install Polyscope >= 2.5.0 to enable the "
            "renderer. Falling back to render mode None...",
            stacklevel=2,
        )
        render_mode = None
        video_path = None
        video_size = None

    # Parse video size.
    if video_size is not None:
        video_size = tuple(int(x) for x in video_size.split("x"))
        if len(video_size) != 2:
            raise ValueError(f"Invalid video size: {video_size}")

    # Setup common parameters. These are runtime/presentation settings owned by the CLI,
    # so they are merged over the entry config and win. Task configuration deliberately
    # does not appear here: steps_per_episode would restate each env class default while
    # silently overriding a horizon supplied by a config variant, and num_worker_threads
    # already defaults to 0.
    common_env_cfg = {
        "render_mode": render_mode,
        "render_size": video_size,
        "start_paused": start_paused,
        "profile": True,
    }

    # Create the environment to run and select the action sampler function.
    # fmt: off
    action_samplers = {
        "zero": zero_action,
        "random": random_action,
        "sweep": sweep_action,
    }
    # fmt: on

    if action_sampler not in action_samplers:
        raise ValueError(f"Unknown action sampler: {action_sampler}")
    env = make_env(sample, common_env_cfg)
    action_sampler_fn = action_samplers[action_sampler]

    # Initialize animation writer.
    animation_writer = None
    if video_path is not None:
        fps = env.get_control_frequency()
        animation_writer = AnimationWriter(video_path, fps, "mp4")

    # With the environment instantiated, we can now step it.
    # See the definition of `sample_runner` for more details.
    if profile:
        pr = cProfile.Profile()
        pr.enable()
        sample_runner(env, action_sampler_fn, num_episodes, animation_writer)
        pr.disable()
        print()
        print("cProfile summary")
        pr.print_stats(sort="cumulative")
    else:
        sample_runner(env, action_sampler_fn, num_episodes, animation_writer)

    # Print the environment's profiler summary (if available).
    profiler = env.get_profiler()
    if profiler.enabled:
        print()
        print("Environment profiler summary")
        profiler.print_summary()


########################################################################################


def main():
    # Discover and register the available environments up front so the CLI can list them.
    # Test-only variants are excluded, so they are not offered nor accepted here.
    register_all_envs()
    available_samples = sorted(get_env_short_names())

    # Parse command line arguments.
    parser = argparse.ArgumentParser(
        description="Run SuperDex Gym sample environments with different action sampling strategies."
    )
    parser.add_argument(
        "sample",
        type=str,
        choices=available_samples,
        help="Name of the sample environment to run. One of: "
        + ", ".join(available_samples),
    )
    parser.add_argument(
        "--action_sampler",
        type=str,
        default="sweep",
        help="Action sampling strategy to use (zero, random, sweep)",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=10, help="Number of episodes to run"
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Enable video recording of the environment. Setting this argument will "
        "make rendering happen offscreen.",
    )
    parser.add_argument(
        "--video_size",
        type=str,
        default=None,
        help=("Size of the video output (e.g., 1280x720, 1920x1080)"),
    )
    parser.add_argument(
        "--video_path",
        type=pathlib.Path,
        default=None,
        help=(
            "Path to save video output (defaults to output/ folder if ony --video is "
            "used). This argument implies --video"
        ),
    )
    parser.add_argument(
        "--start_paused",
        action="store_true",
        help="Start the environment in paused state",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable performance profiling during execution",
    )
    args = parser.parse_args()

    # Setup logging.
    configure_logger()

    # Retrieve video path.
    # If not specified, then put it into a child "output" folder.
    video_path = args.video_path
    if video_path is not None:
        args.video = True
    if args.video and video_path is None:
        base_path = pathlib.Path(__file__).parent.resolve()
        video_path = base_path / "output"

    # Run the sample.
    run_sample(
        sample=args.sample,
        action_sampler=args.action_sampler,
        num_episodes=args.num_episodes,
        video_size=args.video_size,
        video_path=video_path,
        start_paused=args.start_paused,
        profile=args.profile,
    )


if __name__ == "__main__":
    main()
