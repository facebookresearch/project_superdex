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
Actor updater for UnrealCV-based scene synchronization.

This class handles the updating of mochi actors via UnrealCV, including
transform updates for rigid bodies
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from superdex.physics import Actor
from superdex.physics.utils.coordinate_systems import CoordinateTransform

from .unrealcv_updater import UnrealCVUpdater

if TYPE_CHECKING:
    from ..unrealcv_client import UnrealCVClient

logger = logging.getLogger(__name__)

########################################################################################


class UnrealCVActorUpdater(UnrealCVUpdater):
    """
    Updater for mochi actors via UnrealCV.

    This updater handles the synchronization of mochi actor state (transforms and
    optionally mesh geometry) with Unreal Engine via the UnrealCV protocol.
    """

    ####################################################################################
    # Members
    ####################################################################################

    _actor: Actor
    """The mochi actor being updated."""

    _meters_to_cm: float
    """Scale factor for converting meters to centimeters."""

    ####################################################################################
    # Constructor
    ####################################################################################

    def __init__(
        self,
        actor: Actor,
        ue_actor_name: str,
        client: "UnrealCVClient",
        coordinate_transform: CoordinateTransform,
        meters_to_cm: float = 100.0,
    ):
        """
        Initialize the actor updater.

        Args:
            actor: The mochi actor to update.
            ue_actor_name: Name of the corresponding actor in Unreal Engine.
            client: Reference to the UnrealCV client.
            coordinate_transform: Converter for coordinate system transformations.
            meters_to_cm: Scale factor for converting meters to centimeters.
        """
        super().__init__(
            name=actor.get_name(),
            ue_actor_name=ue_actor_name,
            client=client,
            coordinate_transform=coordinate_transform,
        )

        self._actor = actor
        self._meters_to_cm = meters_to_cm

    ####################################################################################
    # Update Methods
    ####################################################################################

    def get_transform_data(
        self,
    ) -> (
        tuple[
            str,
            npt.NDArray[np.floating] | None,
            npt.NDArray[np.floating] | None,
        ]
        | None
    ):
        """
        Get the transform data for batched updates.

        Returns:
            Tuple of (ue_actor_name, position_ue, rotation_ue), or None
            if this actor has no transform or is not visible.
            - ue_actor_name: The UE actor name
            - position_ue: Position in UE coordinates, or None if no transform
            - rotation_ue: Rotation in UE coordinates, or None if no transform
        """
        if not self._is_visible:
            return None

        if not self._actor.has_root_transform():
            return None

        position_ue, rotation_ue = self._get_actor_transform_ue()
        return (self._ue_actor_name, position_ue, rotation_ue)

    def remove(self):
        """Hide the actor in Unreal Engine."""
        self.set_visible(False)

    ####################################################################################
    # Transform Handling
    ####################################################################################

    def _get_actor_transform_ue(
        self,
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """
        Get the actor's transform in UE coordinates.

        Returns:
            Tuple of (position, rotation) where:
            - position is [x, y, z] in centimeters
            - rotation is [x, y, z, w] quaternion
        """

        # Get transform from mochi
        actor_xform = self._actor.get_root_transform()

        # Convert to target coordinates
        position_ue = self._coordinate_transform.position_to_target(
            np.array(actor_xform.translation), scale=self._meters_to_cm
        )
        rotation_ue = self._coordinate_transform.rotation_to_target(
            np.array(actor_xform.rotation)
        )

        return position_ue, rotation_ue

    ####################################################################################
    # Actor Properties
    ####################################################################################

    def get_actor(self) -> Actor:
        """Returns the mochi actor."""
        return self._actor

    def get_aabb(self):
        """
        Returns the axis-aligned bounding box of this actor in world coordinates.

        Uses the mochi actor's get_aabb_world() for efficient AABB computation
        directly from the physics engine.
        """
        from superdex.physics.viewer.utils.aabb import AABB

        try:
            mochi_aabb = self._actor.get_aabb_world()
            aabb = AABB(
                min=np.array([mochi_aabb.min[0], mochi_aabb.min[1], mochi_aabb.min[2]]),
                max=np.array([mochi_aabb.max[0], mochi_aabb.max[1], mochi_aabb.max[2]]),
            )
            return aabb
        except Exception:
            return AABB.empty()
