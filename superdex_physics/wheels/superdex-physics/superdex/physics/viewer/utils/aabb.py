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

import dataclasses

import numpy as np
import numpy.typing as npt

########################################################################################


@dataclasses.dataclass
class AABB:
    """
    Class representing an axis-aligned bounding box.
    """

    ####################################################################################
    # Members
    ####################################################################################

    # Public members.
    min: npt.NDArray[float]
    """Minimum coordinates of the bounding box."""
    max: npt.NDArray[float]
    """Maximum coordinates of the bounding box."""

    ####################################################################################
    # Constructors
    ####################################################################################

    @staticmethod
    def empty() -> AABB:
        """Generates an empty AABB."""
        return AABB(np.zeros(3), np.zeros(3))

    @staticmethod
    def from_points(points: npt.NDArray[float]) -> AABB:
        """Generates an AABB from the given points."""
        return AABB(np.min(points, axis=0), np.max(points, axis=0))

    @staticmethod
    def from_aabbs(aabbs: list[AABB]) -> AABB:
        """Generates an AABB from the given AABBs."""
        return AABB(
            np.min([aabb.min for aabb in aabbs], axis=0),
            np.max([aabb.max for aabb in aabbs], axis=0),
        )

    ####################################################################################
    # Properties
    ####################################################################################

    @property
    def center(self) -> npt.NDArray[float]:
        """Returns the center of the bounding box."""
        return (self.min + self.max) / 2

    @property
    def extents(self) -> npt.NDArray[float]:
        """Returns the extents of the bounding box."""
        return self.max - self.min

    @property
    def is_empty(self) -> bool:
        """Returns true if the bounding box is empty. A bounding box is considered empty
        if it's been collapsed to a single point (i.e. min == max)."""
        return not np.all(self.min < self.max)

    ####################################################################################
    # Methods
    ####################################################################################

    def compute_from_points(self, points: npt.NDArray[float]) -> None:
        """Updates the given bounding box to contain the given points."""
        self.min[:] = np.min(points, axis=0)
        self.max[:] = np.max(points, axis=0)

    def compute_from_aabbs(self, aabbs: list[AABB]) -> None:
        """Updates the given bounding box to contain the given AABBs."""
        self.min[:] = np.min([aabb.min for aabb in aabbs], axis=0)
        self.max[:] = np.max([aabb.max for aabb in aabbs], axis=0)

    def compute_from_transformed_aabb(
        self, other: AABB, transform: npt.NDArray[float]
    ) -> None:
        """Updates the given bounding box to contain the given transformed AABB."""
        extents = other.extents
        dx = extents[0] * transform[:3, 0]
        dy = extents[1] * transform[:3, 1]
        dz = extents[2] * transform[:3, 2]
        corner = transform[:3, :3] @ other.min + transform[:3, 3]
        points = np.array(
            [
                corner,
                corner + dx,
                corner + dy,
                corner + dx + dy,
                corner + dz,
                corner + dx + dz,
                corner + dy + dz,
                corner + dx + dy + dz,
            ]
        )
        self.min[:] = np.min(points, axis=0)
        self.max[:] = np.max(points, axis=0)

    def set_empty(self) -> None:
        """Sets the bounding box to be empty."""
        self.min[:] = 0
        self.max[:] = 0
