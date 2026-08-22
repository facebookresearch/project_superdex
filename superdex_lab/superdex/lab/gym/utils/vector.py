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

from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Any, Callable, NamedTuple, Self, Sequence

import numpy as np
import numpy.typing as npt
from gymnasium import Env, spaces
from gymnasium.vector import AsyncVectorEnv, AutoresetMode, SyncVectorEnv, VectorWrapper
from superdex.lab.gym.envs import Action, Info, Observation, ResetResult
from superdex.physics.utils.decorators import override_from
from superdex.physics.viewer import RenderFrame

########################################################################################


class BatchedStepResult(NamedTuple):
    """Batched version of the step result."""

    observation: Observation
    reward: npt.NDArray[float]
    terminated: npt.NDArray[bool]
    truncated: npt.NDArray[bool]
    info: Info


BatchedResetResult = ResetResult
"""Batched version of the reset result."""

BatchedRenderFrame = tuple[RenderFrame, ...]
"""Batched version of the render frame."""

########################################################################################


class HybridVectorEnv(VectorWrapper):
    """
    A hybrid vectorized environment that combines async and sync vectorization.

    This class creates multiple async workers using AsyncVectorEnv, where each worker
    runs a SyncVectorEnv containing multiple environments. This enables to run multiple
    environments in parallel, while also allowing for synchronous execution within each
    worker.
    """

    ####################################################################################
    # Nested classes
    ####################################################################################

    class _InnerVectorEnv(SyncVectorEnv):
        """Inner vectorization wrapper handling the serial execution of environment
        within a worker. Currently this is simply a `SyncVectorEnv`, but future versions
        of the wrapper could use a different implementation."""

        pass

    class _OuterVectorEnv(AsyncVectorEnv):
        """Outer vectorization wrapper handling the parallel execution of environments
        across workers. In the current version this is simply a batch-aware
        `AsyncVectorEnv`, but future versions of the wrapper could use more advanced
        implementations."""

        def __init__(
            self,
            env_fns: Sequence[Callable[[], Env]],
            num_inner_envs: int,
            *args,
            **kwargs,
        ):
            self.num_envs = len(env_fns)
            self.num_inner_envs = num_inner_envs
            super().__init__(env_fns, *args, **kwargs)

        @override_from(AsyncVectorEnv)
        def _add_info(
            self, vector_infos: Info, env_info: Info, env_num: int
        ) -> dict[str, Any]:
            # Process each key-value pair from the worker's info dictionary. Note this
            # has been pre-processed already by the inner sequential vectorization,
            # therefore it's already in batched format.
            num_envs = self.num_envs * self.num_inner_envs
            for info_key, info_value in env_info.items():
                # Skip internal mask keys - we'll regenerate them during processing.
                if info_key.startswith("_") and info_key[1:] in env_info:
                    continue

                # Recursively handle nested dictionary structures.
                if isinstance(info_value, dict):
                    if info_key not in vector_infos:
                        vector_infos[info_key] = {}
                    self._add_info(vector_infos[info_key], info_value, env_num)
                else:
                    # Handle array data with its associated presence mask.
                    mask_key = f"_{info_key}"
                    validity_mask = env_info[mask_key]

                    # Initialize global arrays on first encounter of this key.
                    if info_key not in vector_infos:
                        # Create array to hold data from all environments.
                        vector_infos[info_key] = np.empty(
                            (num_envs, *info_value.shape[1:]),
                            dtype=info_value.dtype,
                        )
                        # Create mask to track which environments provided this datum.
                        vector_infos[mask_key] = np.zeros(
                            (num_envs,),
                            dtype=bool,
                        )

                    # Copy worker's data into the appropriate slice of the global arrays.
                    worker_start_idx = env_num * self.num_inner_envs
                    worker_end_idx = worker_start_idx + self.num_inner_envs
                    vector_infos[info_key][worker_start_idx:worker_end_idx] = info_value
                    vector_infos[mask_key][worker_start_idx:worker_end_idx] = (
                        validity_mask
                    )

            return vector_infos

    ####################################################################################
    # Constructor
    ####################################################################################

    def __init__(
        self,
        env_fns: Sequence[Callable[[], Env]],
        num_envs_per_worker: int = 1,
        *args,
        autoreset_mode: str | AutoresetMode = AutoresetMode.NEXT_STEP,
        inner_env_cls: type | None = None,
        inner_env_kwargs: dict | None = None,
        **kwargs,
    ):
        """
        Initializes the hybrid vectorized environment. The argument `env_fns` specifies
        the factory functions generating each environment to be vectorized. These
        environments are distributed across multiple workers, where each worker runs a
        `num_envs_per_worker` environmenst sequentially. Note that the total number of
        environments must be divisible by the number of environments per worker (i.e.
        `len(env_fns) % num_envs_per_worker == 0`).

        `inner_env_cls` overrides the inner per-subprocess VectorEnv class (default:
        `_InnerVectorEnv`). Must be picklable by name (module-level) since
        AsyncVectorEnv ships the factory to each subprocess. `inner_env_kwargs` is
        a dict of additional kwargs forwarded to `inner_env_cls(...)`.
        """

        # Validate that environments can be evenly distributed across workers.
        num_envs = len(env_fns)
        if num_envs % num_envs_per_worker != 0:
            raise ValueError(
                "The total number of environments must be divisible by the number of "
                "environments per process."
            )

        # Create environment functions for each async worker.
        # Each worker will run a SyncVectorEnv with multiple environments.
        worker_env_fns = []
        num_workers = num_envs // num_envs_per_worker
        # Bind inner class/kwargs into the closure so the picklable factory
        # carries them to each AsyncVectorEnv subprocess.
        inner_cls = (
            inner_env_cls
            if inner_env_cls is not None
            else HybridVectorEnv._InnerVectorEnv
        )
        inner_kwargs = inner_env_kwargs or {}

        for worker_idx in range(num_workers):
            # Calculate which environments this worker will handle.
            env_start_idx = worker_idx * num_envs_per_worker
            env_end_idx = env_start_idx + num_envs_per_worker
            worker_env_batch = env_fns[env_start_idx:env_end_idx]
            worker_inner_kwargs = dict(inner_kwargs)

            # Append the function instantiating the inner VectorEnv for this worker.
            def create_worker_env(
                env_batch=worker_env_batch,
                inner_cls=inner_cls,
                inner_kwargs=worker_inner_kwargs,
            ):
                return inner_cls(
                    env_batch, autoreset_mode=autoreset_mode, **inner_kwargs
                )

            worker_env_fns.append(create_worker_env)

        # Initialize the outer AsyncVectorEnv with our worker environments. Note we
        # disable autoreset at the async level because it will be handled by the inner
        # SyncVectorEnvs following the supplied autoreset mode.
        if kwargs.pop("context", None) not in (None, "spawn"):
            warnings.warn(
                "You have specified a multiprocessing context other than 'spawn'. "
                "Spawning child processes via methods other than spawn (e.g. 'fork') "
                "will result in a deadlock if the Mochi context is set to run with "
                "worker threads. Falling back to 'spawn' instead to prevent this.",
                category=RuntimeWarning,
                stacklevel=2,
            )

        env = HybridVectorEnv._OuterVectorEnv(
            worker_env_fns,
            num_envs_per_worker,
            *args,
            **kwargs,
            autoreset_mode=AutoresetMode.DISABLED,
            context="spawn",
        )
        super().__init__(env)

        # Determine flattened and single observation/action spaces.
        # This hides the nesting from users, allowing them to work with single batch
        # dimension tensors. Note we assumes all environments have identical Box spaces
        # (continuous spaces) with the same boundaries.
        def flatten_batch_dims(original_space: spaces.Box) -> spaces.Box:
            """Convert a Box space to its flattened equivalent."""
            return spaces.Box(
                low=original_space.low.reshape(-1, *original_space.low.shape[2:]),
                high=original_space.high.reshape(-1, *original_space.high.shape[2:]),
                dtype=original_space.dtype,
                seed=deepcopy(original_space.np_random),
            )

        def drop_leading_dim(original_space: spaces.Box) -> spaces.Box:
            """Drops the leading batch dimension of the given space."""
            return spaces.Box(
                low=original_space.low[0],
                high=original_space.high[0],
                dtype=original_space.dtype,
                seed=deepcopy(original_space.np_random),
            )

        self._observation_space = flatten_batch_dims(self.env.observation_space)
        self._action_space = flatten_batch_dims(self.env.action_space)
        self._single_observation_space = drop_leading_dim(
            self.env.single_observation_space
        )
        self._single_action_space = drop_leading_dim(self.env.single_action_space)
        self._num_envs_per_worker = num_envs_per_worker

    ####################################################################################
    # Methods
    ####################################################################################

    @override_from(VectorWrapper)
    def reset(
        self,
        *,
        seed: int | Sequence[int | None] | None = None,
        options: dict[str, Any] | None = None,
    ) -> BatchedResetResult:
        """
        Reset all parallel environments and return a batch of initial observations and
        info. A random seed can be provided to control the environment reset sequence.
        This seed can be a single value, or a sequence of values (one for each
        environment). Additional reset options can also be provided. Note that the
        reset mask is not supported for this vectorization.
        """
        if options is not None and "reset_mask" in options:
            raise ValueError("Reset mask is not supported for this vectorization.")

        # Distribute seeds across workers if a full seed sequence is provided
        # Each worker gets a subset of seeds for its environments
        processed_seed = seed
        if isinstance(seed, Sequence):
            if len(seed) != self.num_envs:
                raise ValueError(
                    f"Expected seed sequence to match the num_envs={self.num_envs} "
                    f"but got a sequence of length len={len(seed)} instead."
                )
            processed_seed = [
                seed[worker_start_idx : worker_start_idx + self._num_envs_per_worker]
                for worker_start_idx in range(0, len(seed), self._num_envs_per_worker)
            ]

        # Delegate to inner AsyncVectorEnv for actual reset execution.
        observations, info = self.env.reset(seed=processed_seed, options=options)

        # Flatten observations from [num_workers, envs_per_worker, ...] to [num_envs, ...]
        # This removes the nested batch dimension for easier downstream processing.
        return BatchedResetResult(
            observation=observations.reshape(-1, *observations.shape[2:]),
            info=info,
        )

    @override_from(VectorWrapper)
    def step(self, actions: Action) -> BatchedStepResult:
        """
        Execute one step in all environments using the provided actions.
        """

        if not isinstance(actions, np.ndarray):
            raise ValueError(
                f"Expected actions to be a numpy array but got {type(actions)}."
            )
        if actions.shape != self.action_space.shape:
            raise ValueError(
                f"Expected actions.shape={self.action_space.shape}, but got "
                f"shape={actions.shape} instead."
            )

        # Execute step across all worker environments
        (observations, rewards, terminated, truncated, info) = self.env.step(
            actions.reshape(self.env.action_space.shape)
        )

        # Flatten all results from nested batch format to single batch format.
        # Convert from [num_workers, envs_per_worker, ...] to [num_envs, ...].
        return BatchedStepResult(
            observation=observations.reshape(-1, *observations.shape[2:]),
            reward=rewards.reshape(-1),
            terminated=terminated.reshape(-1),
            truncated=truncated.reshape(-1),
            info=info,
        )

    ####################################################################################
    # Properties
    ####################################################################################

    @property
    def num_envs(self) -> int:
        """Total number of individual environments across all workers."""
        return self.env.num_envs * self._num_envs_per_worker

    @property
    def num_workers(self) -> int:
        """Number of async worker processes."""
        return self.env.num_envs

    @property
    def num_envs_per_worker(self) -> int:
        """Number of environments per worker. Computed as `num_envs // num_workers`."""
        return self._num_envs_per_worker

    ####################################################################################
    # Other operators
    ####################################################################################

    def __enter__(self) -> Self:
        """Enter the context manager and return the vectorized environment."""
        return self

    def __exit__(self, *ignored) -> bool:
        """Exit the context manager and close the vectorized environment."""
        self.close()
        return False

    def __repr__(self) -> str:
        """Returns a string representation of the hybrid vector environment."""
        if self.spec is None:
            return f"{self.__class__.__name__}(num_envs={self.num_envs}, num_workers={self.num_workers})"
        else:
            return f"{self.__class__.__name__}({self.spec.id}, num_envs={self.num_envs}, num_workers={self.num_workers})"
