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

import numpy as np
import numpy.typing as npt
import superdex.physics as mochi
from superdex.physics import Actor, ActorType
from superdex.physics.utils.coordinate_systems import CoordinateTransform
from superdex.physics.utils.decorators import override_from
from superdex.physics.utils.transformations import make_transform, transformrt_to_numpy
from superdex.physics.viewer.renderers.mesh_renderer import MeshRenderer

########################################################################################


class ActorRenderer(MeshRenderer):
    """
    Specialization of the MeshRenderer handling the rendering of an Actor.
    """

    ####################################################################################
    # Members
    ####################################################################################

    # Actor properties.
    _actor: Actor
    _show_axes_at_com: bool

    ####################################################################################
    # Constants
    ####################################################################################

    _TAG: str = "[Actor]"
    """Tag used to identify ActorRenderer-related render structure."""

    ####################################################################################
    # Constructor
    ####################################################################################

    def __init__(
        self,
        actor: Actor,
        coordinate_transform: CoordinateTransform,
    ):
        """Initializes the actor renderer."""

        # Initialize ActorRenderer-specific members.
        assert not actor.get_surface_mesh().is_empty()
        self._actor = actor
        self._show_axes_at_com = False

        # Rely on the base class to initialize the render structure.
        # Use a unique identifier combining handle and name for the Polyscope group name
        unique_name = f"{self._actor.get_name()}_h{self._actor.get_handle().value}"
        super().__init__(
            name=unique_name,
            coordinate_transform=coordinate_transform,
            coordinates=self._get_actor_coordinates(),
            faces=self._get_actor_faces(),
            transform=self._get_actor_transform(),
        )

    ####################################################################################
    # Functions handling the render structure creation and update
    ####################################################################################

    @override_from(MeshRenderer)
    def update(self) -> None:
        """Updates the actor, querying its new state in the simulation and updating the
        corresponding render structure."""

        # Update underlying mesh renderer.
        transform = self._get_actor_transform()
        if transform is not None:
            self.set_transform(transform)
        if self._actor.get_type() in (ActorType.SOFT, ActorType.SHELL):
            self.set_local_coordinates(self._get_actor_coordinates())
        super().update()

        # Special case: if the actor is rigid, is not static, and the user requested to
        # show the axes on the center of mass, override the axes renderer transform.
        if (
            self._actor.get_type() == ActorType.RIGID
            and not self._actor.is_static()
            and self._show_axes_at_com
        ):
            transform = self._get_actor_com_transform()
            self.get_axes_renderer().set_transform(transform)
            self.get_axes_renderer().update()

    ####################################################################################
    # Actor geometry and topology management
    ####################################################################################

    def _get_actor_transform(self) -> npt.NDArray[float] | None:
        """Gets the rigid actor's transform."""
        if not self._actor.has_root_transform():
            return None
        pos, rotvec = transformrt_to_numpy(self._actor.get_root_transform())
        return make_transform(pos, rotvec)

    def _get_actor_com_transform(self) -> npt.NDArray[float]:
        """Gets the rigid actor's transform centered on the center of mass."""
        assert self._actor.get_type() == ActorType.RIGID and not self._actor.is_static()
        pos, rotvec = transformrt_to_numpy(self._actor.get_center_of_mass_transform())
        return make_transform(pos, rotvec)

    def _get_actor_coordinates(self) -> npt.NDArray[float]:
        """Gets the Actor's surface mesh coordinates."""
        self._actor.register_query_and_compute(mochi.QueryType.SURFACE_NODE_POSITIONS)
        return np.asarray(
            self._actor.get_surface_mesh_node_positions_local(), dtype=np.float32
        ).reshape(-1, 3)

    def _get_actor_faces(self) -> npt.NDArray[np.int32]:
        """Gets the Actor's surface mesh connectivity."""
        return np.asarray(
            self._actor.get_surface_mesh().connectivity, dtype=np.int32
        ).reshape(-1, 3)

    ####################################################################################
    # Actor properties
    ####################################################################################

    def get_actor(self) -> Actor:
        """Returns the actor."""
        return self._actor

    def are_frame_axes_at_com(self) -> bool:
        """Returns whether the axes describing the frame of reference is being displayed
        at the actor's center of mass."""
        return self._show_axes_at_com

    def set_show_axes_at_com(self, enabled: bool) -> None:
        """Sets whether to locate the axes describing the frame of reference at center
        of mass of the actor or at the root transform. Note that this change will only
        be reflected in the next update."""
        enabled = bool(enabled)
        self._show_axes_at_com = enabled
