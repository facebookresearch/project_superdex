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

import argparse
import json
import pathlib

import gymnasium as gym
from checkpoint_policy import restore_policy
from superdex.physics.viewer import VIEWER_AVAILABLE
from superdex.physics.viewer.utils import AnimationWriter
from utils import register_envs

########################################################################################


def run_inference(
    checkpoint_path: pathlib.Path,
    num_episodes: int,
    explore_during_inference: bool,
    video_path: pathlib.Path | None,
):
    # Register MochiGym envs.
    register_envs(import_sample_envs=True)

    # Determine paths to a) the training params and b) the policy checkpoint.
    path_to_training_params = checkpoint_path.parent / "params.json"

    # Load the training params. Figure out the environment name and configuration.
    with open(path_to_training_params, "r") as handle:
        training_params = json.load(handle)
    env_name = training_params["env"]
    env_cfg = training_params["env_config"]

    # Restore the policy together with its connector pipelines, so that observation
    # normalization and action rescaling match what training did.
    policy = restore_policy(checkpoint_path)

    # Setup environment configuration. If the user wishes to render the environment to
    # video, set the render mode "rgb_array" (off-screen). Otherwise, set the render
    # mode to "human", which will create a rendering window. Note we opt to handle
    # animation here through "rgb_array" in order to support arbitrary environments
    # other than those based in MochiEnv.
    is_mochi_environment = "superdex_gym" in env_name
    is_renderer_available = not is_mochi_environment or VIEWER_AVAILABLE
    animation_writer = None

    if is_renderer_available:
        env_cfg["render_mode"] = "human"
        if video_path is not None:
            video_path.parent.mkdir(parents=True, exist_ok=True)
            animation_writer = AnimationWriter(
                output_path=video_path, fps=30, fmt="mp4"
            )
            env_cfg["render_mode"] = "rgb_array"
    else:
        print("Renderer not available, setting render mode to None...")
        env_cfg["render_mode"] = None

    # Create an env and do inference with it.
    # MochiGym environment configuration is specified through the "cfg" argument.
    # Other environments like Gymnasium's are specified through keyword arguments.
    if "superdex_gym" in env_name:
        env_cfg = {"cfg": env_cfg}
    env = gym.make(env_name, **env_cfg)
    obs, info = env.reset()
    episode = policy.new_episode(
        obs,
        info=info,
        observation_space=env.observation_space,
        action_space=env.action_space,
    )
    episode_index = 0
    episode_return = 0.0

    while episode_index != num_episodes:
        # The connector pipelines operate on lists of episodes, so B=1 here.
        raw_actions, env_actions, extra_outs = policy.compute_actions(
            [episode], explore=explore_during_inference
        )

        # Step with the transformed action, but record the raw policy-space action below.
        obs, reward, terminated, truncated, info = env.step(env_actions[0])
        episode_return += reward

        # Keep the episode up to date: the connectors read the latest observation from
        # it, and stateful modules read their carried-over state from it.
        episode.add_env_step(
            obs,
            raw_actions[0],
            reward,
            infos=info,
            terminated=terminated,
            truncated=truncated,
            extra_model_outputs={k: v[0] for k, v in extra_outs.items()},
        )

        # If we are rendering to video, then write the current frame to the video.
        if animation_writer is not None:
            animation_writer.add(env.render())

        # Is the episode `done`? -> Reset.
        if terminated or truncated:
            reason = "Unavailable"
            reason = info.get("terminated_reason", reason)
            reason = info.get("truncated_reason", reason)
            print(f"Episode done: {reason}. Total reward = {episode_return}")

            if animation_writer is not None:
                animation_writer.write(f"inference_{episode_index:03d}")

            obs, info = env.reset()
            episode = policy.new_episode(
                obs,
                info=info,
                observation_space=env.observation_space,
                action_space=env.action_space,
            )
            episode_index += 1
            episode_return = 0.0

    print(f"Done performing action inference through {episode_index} Episodes")
    env.close()

    if animation_writer is not None:
        print("Waiting for remaining videos to be written to disk...")
        animation_writer.flush()


########################################################################################


if __name__ == "__main__":
    # Parse command line arguments.
    parser = argparse.ArgumentParser(
        description="Run inference using a trained RLlib policy checkpoint."
    )
    parser.add_argument(
        "checkpoint_path",
        type=pathlib.Path,
        help="Path to the checkpoint directory containing the trained policy",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=10,
        help="Number of episodes to run for inference",
    )
    parser.add_argument(
        "--explore_during_inference",
        action="store_true",
        help="Use exploration policy instead of deterministic inference policy",
    )
    parser.add_argument(
        "--video", action="store_true", help="Enable video recording of the inference"
    )
    parser.add_argument(
        "--video_path",
        type=pathlib.Path,
        default=None,
        help=(
            "Path to save video output (defaults to checkpoint path if only --video is "
            "used). This argument implies --video"
        ),
    )
    args = parser.parse_args()

    # Retrieve video path.
    # If not specified, then put it in the checkpoint path.
    video_path = args.video_path
    if video_path is not None:
        args.video = True
    if args.video and video_path is None:
        video_path = args.checkpoint_path

    # Run inference.
    run_inference(
        checkpoint_path=args.checkpoint_path,
        num_episodes=args.num_episodes,
        explore_during_inference=args.explore_during_inference,
        video_path=video_path,
    )
