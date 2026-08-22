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
from superdex.physics import Actor
from superdex.physics.utils.coordinate_systems import CoordinateTransform
from superdex.physics.utils.decorators import override_from
from superdex.physics.utils.transformations import make_transform
from superdex.physics.viewer.renderers.grid_renderer import GridRenderer
from superdex.physics.viewer.utils.aabb import AABB


class StaticPlaneRenderer(GridRenderer):
    """Renderer for static plane collision shapes using GridRenderer."""

    _SIZE = 1e3
    _PERIOD = 1.0
    _COLOR_1 = (0.85, 0.85, 0.85)
    _COLOR_2 = (0.95, 0.95, 0.95)

    def __init__(self, actor: Actor, coordinate_transform: CoordinateTransform):
        self._actor = actor
        self._plane_aabb = AABB.empty()

        center, normal = self._infer_plane_from_aabb()

        super().__init__(
            name=f"{actor.get_name()}_h{actor.get_handle().value}",
            coordinate_transform=coordinate_transform,
            size=self._SIZE,
            center=center,
            axes="xz",
            period=self._PERIOD,
            style="checker",
            color_1=np.asarray(self._COLOR_1, dtype=float),
            color_2=np.asarray(self._COLOR_2, dtype=float),
            double_sided=True,
        )
        GridRenderer.update(self)  # Set up checker pattern

        # Apply rotation for arbitrary normal
        rotvec = self._rotation_from_normal(normal)
        transform = make_transform(center, rotvec, self._SIZE)
        assert self._render_struct is not None
        self._render_struct.set_transform(
            coordinate_transform.source_to_target @ transform
        )

    def _infer_plane_from_aabb(self) -> tuple[npt.NDArray, npt.NDArray]:
        """Infer plane center and normal from AABB."""
        center = np.zeros(3, dtype=np.float32)
        normal = np.array([0, 1, 0], dtype=np.float32)

        try:
            aabb = self._actor.get_aabb_world()
            min_pt = np.array([aabb.min[0], aabb.min[1], aabb.min[2]], dtype=np.float32)
            max_pt = np.array([aabb.max[0], aabb.max[1], aabb.max[2]], dtype=np.float32)
            for i in range(3):
                if np.isfinite(min_pt[i]) or np.isfinite(max_pt[i]):
                    normal[:] = 0.0
                    normal[i] = -1.0 if np.isfinite(min_pt[i]) else 1.0
                    center[:] = 0.0
                    center[i] = min_pt[i] if np.isfinite(min_pt[i]) else max_pt[i]
                    break
            self._plane_aabb.min = center.copy()
            self._plane_aabb.max = center.copy()
        except Exception:
            pass
        return center, normal

    def _rotation_from_normal(self, normal: npt.NDArray) -> npt.NDArray:
        """Compute rotation vector to align Z-up quad with given normal."""
        z_up = np.array([0, 0, 1], dtype=np.float32)
        dot = float(np.dot(z_up, normal))
        if abs(dot) > 0.999:
            return np.array([np.pi, 0, 0] if dot < 0 else [0, 0, 0], dtype=np.float32)
        axis = np.cross(z_up, normal)
        axis /= np.linalg.norm(axis)
        return (axis * np.arccos(np.clip(dot, -1, 1))).astype(np.float32)

    @override_from(GridRenderer)
    def update(self):
        pass

    def get_actor(self) -> Actor:
        return self._actor

    def get_aabb(self) -> AABB:
        return self._plane_aabb

    def set_show_axes_at_com(self, enabled: bool):
        pass

    def is_surface_enabled(self) -> bool:
        return self._render_struct is not None and self._render_struct.is_enabled()

    def set_enable_surface(self, enabled: bool):
        if self._render_struct is not None:
            self._render_struct.set_enabled(bool(enabled))

    def are_edges_enabled(self) -> bool:
        return False

    def set_enable_edges(self, enabled: bool):
        pass

    def are_nodes_enabled(self) -> bool:
        return False

    def set_enable_nodes(self, enabled: bool):
        pass

    def are_axes_enabled(self) -> bool:
        return False

    def set_enable_axes(self, enabled: bool):
        pass
