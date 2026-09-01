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
from superdex.physics import Actor, ActorType, TransformRT
from superdex.physics.utils.coordinate_systems import CoordinateTransform
from superdex.physics.utils.decorators import override_from
from superdex.physics.utils.transformations import make_transform, transformrt_to_numpy
from superdex.physics.viewer.renderers.actor_renderer import ActorRenderer
from superdex.physics.viewer.renderers.mesh_renderer import MeshRenderer

########################################################################################

# Bot render meshes (.glb) are exported with their Y and Z axes flipped relative to the
# collision meshes and each link's render_model_rotation (a Y-up <-> Z-up convention
# baked into the asset-export pipeline, confirmed intentional by the bot-assets owner).
# Undo it so render_model_rotation places the visual mesh exactly like the collision
# mesh: (x, y, z) -> (x, -z, y). This is a +90 deg rotation about X (det +1), so triangle
# winding is preserved and no face flip is needed.
_GLB_YZ_UNFLIP: npt.NDArray[np.float32] = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float32
)


def _load_glb_mesh(
    glb_path: str, scale: npt.ArrayLike
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]:
    """Loads a ``.glb`` file into ``(coordinates, faces)``.

    Node transforms are baked into the vertex positions (via trimesh's
    ``force="mesh"`` concatenation) and the per-axis render scale is baked in as well,
    so the resulting geometry can be uploaded once and thereafter driven by a rigid 4x4
    transform alone. The bot export's Y/Z axis flip is also undone here (see
    @ref _GLB_YZ_UNFLIP), so the mesh lands in the same frame as the collision mesh.
    """
    import trimesh

    loaded = trimesh.load(glb_path, force="mesh")
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"GLB did not load as a single mesh: {glb_path}")

    coordinates = np.asarray(loaded.vertices, dtype=np.float32).reshape(-1, 3)
    faces = np.asarray(loaded.faces, dtype=np.int32).reshape(-1, 3)
    scale = np.asarray(scale, dtype=np.float32).reshape(3)
    # Scale first (per-axis, in the model frame), then unflip Y/Z into the shape frame.
    coordinates = coordinates * scale
    coordinates = coordinates @ _GLB_YZ_UNFLIP.T
    return coordinates, faces


########################################################################################


class GlbActorRenderer(ActorRenderer):
    """
    Renders an actor using an external ``.glb`` visual mesh instead of its physics
    surface mesh.

    The GLB vertices are uploaded once (with the render scale baked in); each frame only
    a 4x4 transform is pushed, composing the actor's root transform with the fixed
    local render transform. Subclassing
    :class:`~superdex.physics.viewer.renderers.ActorRenderer` makes this a drop-in wherever
    renderers are typed ``ActorRenderer | StaticPlaneRenderer`` (UI, scene bounds).
    """

    ####################################################################################
    # Members
    ####################################################################################

    _local_transform: TransformRT
    """Transform from the render-mesh frame to the actor's root frame."""

    ####################################################################################
    # Constants
    ####################################################################################

    _TAG: str = "[GlbActor]"
    """Tag used to identify GlbActorRenderer-related render structure."""

    ####################################################################################
    # Constructor
    ####################################################################################

    def __init__(
        self,
        actor: Actor,
        glb_path: str,
        local_transform: TransformRT,
        scale: npt.ArrayLike,
        coordinate_transform: CoordinateTransform,
    ):
        """Initializes the GLB actor renderer.

        Args:
            actor: The actor whose root transform drives the render mesh.
            glb_path: Absolute path to the ``.glb`` visual mesh.
            local_transform: Transform from the render-mesh frame to the actor's root
                frame.
            scale: Per-axis render scale, baked into the vertices at load time.
            coordinate_transform: Transform to the viewer coordinate system.
        """

        # Load the GLB geometry up front so a failure surfaces before any render
        # structure is created (the viewer falls back to the physics mesh on failure).
        coordinates, faces = _load_glb_mesh(glb_path, scale)

        # Initialize ActorRenderer-specific members. We bypass ActorRenderer.__init__
        # (which builds geometry from the physics surface mesh) and drive MeshRenderer
        # directly with the GLB geometry.
        self._actor = actor
        self._show_axes_at_com = False
        self._local_transform = local_transform

        unique_name = f"{actor.get_name()}_h{actor.get_handle().value}"
        MeshRenderer.__init__(
            self,
            name=unique_name,
            coordinate_transform=coordinate_transform,
            coordinates=coordinates,
            faces=faces,
            transform=self._get_glb_transform(),
        )

    ####################################################################################
    # Functions handling the render structure creation and update
    ####################################################################################

    @override_from(ActorRenderer)
    def update(self) -> None:
        """Updates the render mesh transform from the actor's current root transform.

        The GLB geometry is static, so (unlike
        :class:`~superdex.physics.viewer.renderers.ActorRenderer`) we never re-upload
        vertices -- the per-frame cost is just a 4x4 transform.
        """
        transform = self._get_glb_transform()
        if transform is not None:
            self.set_transform(transform)
        MeshRenderer.update(self)

        # The mesh includes the visual model's local offset, while the axes describe
        # the physical actor's root or center-of-mass frame.
        show_axes_at_com = (
            self._actor.get_type() == ActorType.RIGID
            and not self._actor.is_static()
            and self._show_axes_at_com
        )
        axes_transform = (
            self._get_actor_com_transform()
            if show_axes_at_com
            else self._get_actor_transform()
        )
        if axes_transform is not None:
            axes_renderer = self.get_axes_renderer()
            axes_renderer.set_transform(axes_transform)
            axes_renderer.update()

    def _get_glb_transform(self) -> npt.NDArray[float] | None:
        """Composes ``world_from_mesh = root_transform * local_transform`` as a 4x4."""
        if not self._actor.has_root_transform():
            return None
        world_from_mesh = self._actor.get_root_transform() * self._local_transform
        pos, rotvec = transformrt_to_numpy(world_from_mesh)
        return make_transform(pos, rotvec)
