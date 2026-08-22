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
A collection of functions for transforming between different representations of
rotations and transformations.
"""

import numpy as np
import numpy.typing as npt

########################################################################################
# Angle utilities
########################################################################################


def pi_minus_pi_cap(x: npt.NDArray[float]) -> npt.NDArray[float]:
    """Wraps an angle in the range [-pi, pi)."""
    return np.mod(x + np.pi, 2 * np.pi) - np.pi


def angular_distance(
    x: npt.NDArray[float], y: npt.NDArray[float]
) -> npt.NDArray[float]:
    """Computes the (signed) angular distance between two angles."""
    return pi_minus_pi_cap(y - x)


def unwrap_angle_sequence(x: npt.NDArray[float]) -> npt.NDArray[float]:
    """Unwraps a sequence of angles."""
    out = np.copy(x)
    out[1:] = out[0] + np.cumsum(angular_distance(x[:-1], x[1:]))
    return out
