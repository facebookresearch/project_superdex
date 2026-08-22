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

"""Action sampling utilities and environment runner for SuperDex Gym environments."""

import logging
from typing import Callable, TypeAlias

import numpy as np
import numpy.typing as npt
from superdex.lab.gym.envs.mochi_env import MochiEnv
from superdex.physics.viewer.utils import AnimationWriter

logger = logging.getLogger(__name__)

########################################################################################

ActionSampler: TypeAlias = Callable[[MochiEnv], npt.NDArray[float]]
"""Type alias for a function that samples actions given a MochiEnv environment."""

########################################################################################


def zero_action(env: MochiEnv) -> npt.NDArray[float]:
    """Return a zero action vector for the given environment."""
    return np.zeros(env.action_space.shape, dtype=np.float32)


def random_action(env: MochiEnv) -> npt.NDArray[float]:
    """Return a random action sampled from the environment's action space."""
    return env.action_space.sample()


def sweep_action(env: MochiEnv) -> npt.NDArray[float]:
    """Generate a sweeping sinusoidal action that cycles through action dimensions.
    This function creates an action where only one dimension is active at a time,
    cycling through all dimensions with a sinusoidal value over time."""
    time = env._step_count * env._control_dt
    action_shape = env.action_space.shape
    period_length = 4.0
    index = int(time / period_length) % action_shape[0]
    value = 0.5 * np.pi * np.sin(time / period_length * 2 * np.pi)
    action = np.zeros(action_shape)
    action[index] = value
    return action


########################################################################################


def sample_runner(
    env: MochiEnv,
    action_sampler: ActionSampler = sweep_action,
    num_episodes: int = 10,
    animation_writer: AnimationWriter | None = None,
):
    """Run multiple episodes using a specified action sampling strategy. This function
    runs a specified number of episodes in the given environment, using the provided
    action sampler to generate actions at each step. It tracks episode returns and
    prints episode completion information.
    """

    episode = 0
    episode_return = 0.0
    env.reset()

    if animation_writer:
        animation_writer.add(env.render())

    while episode < num_episodes:
        # Sample an action from the given action sampler.
        action = action_sampler(env)

        # Send the computed action to the env.
        _, reward, terminated, truncated, info = env.step(action)
        episode_return += reward

        # If we are rendering to video, then write the current frame to the video.
        if animation_writer:
            animation_writer.add(env.render())

        # Is the episode `done`? -> Reset.
        if terminated or truncated:
            reason = "Unavailable"
            reason = info.get("terminated_reason", reason)
            reason = info.get("truncated_reason", reason)
            logger.info(f"Episode over: {reason}. Total reward = {episode_return}.")

            env.reset()
            episode += 1
            episode_return = 0.0

            if animation_writer:
                animation_writer.write(f"video_{episode:03d}")
                animation_writer.add(env.render())

    logger.info(f"Done running {episode} episodes.")
    env.close()

    if animation_writer:
        logger.info("Waiting for remaining videos to be written to disk...")
        animation_writer.flush()
