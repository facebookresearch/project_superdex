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

import numpy as np
from gymnasium import Env, spaces

########################################################################################


class BasicMockEnv(Env):
    """Mock environment for testing purposes."""

    metadata = {"render_modes": [None]}

    def __init__(self, env_id: int = 0, max_steps: int = 10):
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )
        self.render_mode = None
        self._env_id = env_id
        self._max_steps = max_steps
        self._step_count = 0
        self._reset_count = 0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self._step_count = 0
        self._reset_count += 1
        observation = np.random.random(4).astype(np.float32)
        info = {"env_id": self._env_id, "reset_count": self._reset_count}
        return observation, info

    def step(self, action):
        self._step_count += 1
        observation = np.random.random(4).astype(np.float32)
        reward = np.random.random()
        terminated = self._step_count >= self._max_steps
        truncated = False
        info = {"env_id": self._env_id, "step_count": self._step_count}
        return observation, reward, terminated, truncated, info

    def close(self):
        pass
