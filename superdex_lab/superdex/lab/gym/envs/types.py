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
Common types and classes involved in the definition of the SuperDex Gym environments
and derived classes.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy.typing as npt
from gymnasium import spaces

#######################################################################################

ObservationSpaceStructure = spaces.Dict
"""
Dictionary describing the structure observation space of the environment. The keys of
the dictionary are the names of the observations, and the values are the spaces of the
observations (shape, boundaries, data type, etc.). The observation space structure is
flattened into a single observation space before being used by the environment.
"""

ObservationSpace = spaces.Box
"""
Space object corresponding to valid observations. All valid observations should be
contained within the space defined by the observation space.
"""

StructuredObservation = dict[str, npt.ArrayLike]
"""
A structured sample of observation space. Its structure follows the same structure
defined by the observation space structure.
"""

Observation = npt.NDArray[float]
"""
A sample of the observation space. Obtained by flattening a structured observation
following an observation space structure.
"""

#######################################################################################


ActionSpaceStructure = spaces.Dict
"""
Dictionary describing the structure action space of the environment. The keys of the
dictionary are the names of the actions, and the values are the spaces of the actions
(shape, boundaries, data type, etc.). The action space structure is flattened into a
single action space before being used by the environment.
"""

ActionSpace = spaces.Box
"""
Space object corresponding to valid actions. All valid actions should be contained
within the space defined by the action space.
"""

StructuredAction = dict[str, npt.ArrayLike]
"""
A structured sample of action space. Its structure follows the same structure defined
by the action space structure. Obtained by unflattening the action with the action
space structure.
"""

Action = npt.NDArray[float]
"""
A sample of the action space.
"""

#######################################################################################

Info = dict[str, Any]
"""
Dictionary containing additional information about the environment.
"""

RewardTerms = dict[str, float]
"""
Dictionary containing different reward terms. These terms are summed together to obtain
the final reward.
"""

#######################################################################################


class StructuredStepResult(NamedTuple):
    """
    Structured result of a step in the environment, as generated internally by the
    environment. Consists of the structured observation, the reward, the termination
    flag, the truncation flag, and the info dictionary. Note that some terms might be
    missing depending if the environment was just created or reset. In these cases, only
    the observation and the info dictionary are guaranteed to be set to non-None values.
    """

    action: StructuredAction | None
    """Action taken by the agent."""
    observation: StructuredObservation
    """Next observation sample after taking the step."""
    reward: RewardTerms | None
    """Reward obtained from the last step."""
    terminated: bool | None
    """Flag indicating if the episode is terminated."""
    truncated: bool | None
    """Flag indicating if the episode must be truncated."""
    info: Info
    """Additional information about the step."""


class StepResult(NamedTuple):
    """
    Result of a step in the environment, as consumed by Gymnasium's Env interface.
    Consists of the observation, the reward, the termination flag, the truncation flag,
    and the info dictionary.
    """

    observation: Observation
    """Next observation sample after taking the step."""
    reward: float
    """Reward obtained from the last step."""
    terminated: bool
    """Flag indicating if the episode is terminated."""
    truncated: bool
    """Flag indicating if the episode must be truncated."""
    info: Info
    """Additional information about the step."""


class ResetResult(NamedTuple):
    """
    Result of a reset in the environment. Consists of the observation and the info
    dictionary.
    """

    observation: Observation
    """Initial observation sample after resetting the environment."""
    info: dict[str, Any]
    """Additional information about the reset."""


#######################################################################################
