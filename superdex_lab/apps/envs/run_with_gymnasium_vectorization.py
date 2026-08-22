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
SuperDex Gym environments currently do not support native vectorization, meaning they can
only run one simulation scene at a time. While this is sufficient for inference, policy
training often requires running multiple environments in parallel to accelerate data
collection.

To address this limitation, our training scripts utilize Ray's capabilities for
transparent and efficient vectorization. However, this approach may not be suitable for
all users, as it is library-specific and deeply integrated within RLlib's training
framework.

An alternative is to use environment vectorization wrappers. These wrappers allow us to
run multiple environments in parallel while maintaining a consistent interface with the
original environments. They also handle packing and unpacking of observations and
actions to and from the main process, enabling interaction with the environments as if
they were running in the main process.

In this script, we demonstrate how to use one of these wrappers with SuperDex Gym
environments. Specifically, we demonstrate the use of our `HybridVectorEnv` wrapper,
which combines Gymnasium's AsyncVectorEnv and SyncVectorEnv to enable vectorized
environment execution. This hybrid approach balances parallelism with efficiency by
reducing process communication overhead while still leveraging multiple cores.

Note: This wrapper is provided for convenience. For more advanced use cases or different
vectorization strategies, consider using Gymnasium's native wrappers directly:
- AsyncVectorEnv: https://gymnasium.farama.org/api/vector/async_vector_env/
- SyncVectorEnv: https://gymnasium.farama.org/api/vector/sync_vector_env/
"""

from typing import Any, Type

import numpy as np
from superdex.lab.gym.envs import MochiEnv
from superdex.lab.gym.envs.benchmarks.cartpole_env import CartPoleEnv
from superdex.lab.gym.utils.vector import HybridVectorEnv

########################################################################################


def run_with_gymnasium_vectorization(
    cls: Type[MochiEnv],
    cfg: dict[str, Any],
    num_environments: int,
    num_environments_per_worker: int,
    num_steps: int,
):
    """
    Runs a number of instances of the given environment using our HybridVectorEnv
    wrapper, which combines async and sync vectorization. Instances are run for the
    given number of steps.
    """

    assert issubclass(cls, MochiEnv), "The given class must inherit MochiEnv"

    # Generate hybrid (async + sync) vectorized environment. This creates multiple
    # async worker processes, where each worker runs a SyncVectorEnv with multiple
    # environments.
    env_creators = [lambda: cls(cfg) for _ in range(num_environments)]
    env = HybridVectorEnv(env_creators, num_envs_per_worker=num_environments_per_worker)
    print(f"Created HybridVectorEnv with {num_environments} environments.")
    print(
        f"Running {env.num_workers} async workers, with {env.num_envs_per_worker} "
        "environments per worker."
    )
    print("Observation space shape:", env.observation_space.shape)
    print("Action space shape:", env.action_space.shape)

    # Perform first reset. Here we can provide a global reset seed for all environments,
    # or a list of reset seeds for each environment. See the documentation for more
    # details regarding how the observations and info dictionaries are packed.
    # https://gymnasium.farama.org/api/vector/#gymnasium.vector.VectorEnv
    #
    # IMPORTANT: HybridVectorEnv does not support reset masks. If you need to force
    # reset specific environments, use Gymnasium's AsyncVectorEnv or SyncVectorEnv
    # directly.
    reset_seeds = list(range(num_environments))
    _, infos = env.reset(seed=reset_seeds)

    # Run the environment for some steps.
    returns = np.zeros(num_environments)
    for step in range(num_steps):
        # Sample random action and step using the flattened action space.
        # NOTE: No need to call env.reset() here for terminated/truncated environments,
        # the HybridVectorEnv will automatically reset them. See
        # https://farama.org/Vector-Autoreset-Mode for details on autoreset modes.
        action = env.action_space.sample()
        _, rewards, terminateds, truncateds, infos = env.step(action)
        returns += rewards

        # Report progress.
        for i, (ret, term, trun) in enumerate(zip(returns, terminateds, truncateds)):
            if term or trun:
                wat = "terminated" if term else "truncated"
                why = infos[f"{wat}_reason"][i]
                print(f"Environment {i} {wat} at step {step} with return {ret}: {why}.")
                returns[i] = 0.0

    # Close the environment.
    env.close()
    del env

    # Print the final returns.
    for i, ret in enumerate(returns):
        print(f"Environment {i} finished with return {ret}.")
    print()


########################################################################################

if __name__ == "__main__":
    run_with_gymnasium_vectorization(
        cls=CartPoleEnv,
        cfg={"render_mode": None},
        num_environments=9,
        num_environments_per_worker=3,
        num_steps=200,
    )
