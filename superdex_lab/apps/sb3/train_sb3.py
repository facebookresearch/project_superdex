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

"""Train and evaluate a Stable-Baselines3 policy on SuperDex Gymnasium environments.

Usage:
    python train_sb3.py [--env_id <env_id>] [--total_timesteps <steps>]

Example:
    python train_sb3.py --env_id superdex_gym/CartPole-v0 --total_timesteps 50000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import superdex.lab  # Registers superdex_gym canonical environments
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

########################################################################################


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate an SB3 PPO policy on SuperDex Gym."
    )
    parser.add_argument(
        "--env_id",
        type=str,
        default="superdex_gym/CartPole-v0",
        help="Canonical environment ID to train.",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=50_000,
        help="Total environment interaction timesteps.",
    )
    parser.add_argument(
        "--log_dir",
        type=Path,
        default=Path("output/sb3"),
        help="Output directory for checkpoints and tensorboard logs.",
    )
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes after training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)

    # 1. Environment initialization
    env = gym.make(args.env_id)

    # 2. Policy setup
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=str(args.log_dir / "tensorboard"),
        device="auto",
    )

    # 3. Training
    model.learn(total_timesteps=args.total_timesteps)

    # 4. Save checkpoint
    checkpoint_path = args.log_dir / f"ppo_{args.env_id.replace('/', '_')}.zip"
    model.save(str(checkpoint_path))

    # 5. Evaluation
    eval_env = gym.make(args.env_id)
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=args.eval_episodes
    )
    print(
        f"Evaluation finished over {args.eval_episodes} episodes: "
        f"Mean Reward = {mean_reward:.2f} +/- {std_reward:.2f}"
    )

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
