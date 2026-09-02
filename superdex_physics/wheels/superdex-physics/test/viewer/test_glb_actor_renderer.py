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

import os
import tempfile
import unittest
from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np
import superdex.physics as sdp
import trimesh
from superdex.physics.viewer.renderers.glb_actor_renderer import (
    _GLB_YZ_UNFLIP,
    _load_glb_mesh,
    GlbActorRenderer,
)
from superdex.physics.viewer.renderers.mesh_renderer import MeshRenderer

########################################################################################


class GlbActorRendererTest(TestCase):
    """Tests for the GLB loading / frame-conversion used by GlbActorRenderer."""

    def test_yz_unflip_maps_axes(self) -> None:
        # The bot export flips Y/Z; the unflip must map (x, y, z) -> (x, -z, y).
        points = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]], dtype=np.float32)
        expected = np.array([[1.0, -3.0, 2.0], [-4.0, 6.0, 5.0]], dtype=np.float32)
        result = points @ _GLB_YZ_UNFLIP.T
        np.testing.assert_allclose(result, expected)

    def test_yz_unflip_is_proper_rotation(self) -> None:
        # A +90 deg rotation about X (det +1) preserves triangle winding; a reflection
        # (det -1) would silently invert normals.
        self.assertAlmostEqual(float(np.linalg.det(_GLB_YZ_UNFLIP)), 1.0, places=5)

    def test_load_glb_mesh_unflips_and_scales(self) -> None:
        # Asymmetric box so the sign of each mapped axis is observable from the bounds.
        box = trimesh.creation.box(extents=(2.0, 4.0, 6.0))
        box.apply_translation((1.0, 10.0, 100.0))
        # glb bounds: x in [0, 2], y in [8, 12], z in [97, 103].

        tmp_dir = tempfile.mkdtemp()
        glb_path = os.path.join(tmp_dir, "box.glb")
        try:
            box.export(glb_path)

            # Scale x by 3 (model frame), then expect the Y/Z unflip: (x, -z, y).
            coords, faces = _load_glb_mesh(glb_path, np.array([3.0, 1.0, 1.0]))
        finally:
            if os.path.exists(glb_path):
                os.remove(glb_path)
            os.rmdir(tmp_dir)

        self.assertEqual(coords.shape[1], 3)
        self.assertEqual(faces.shape[1], 3)

        lo = coords.min(axis=0)
        hi = coords.max(axis=0)
        # x scaled by 3: [0, 6]; new_y = -z: [-103, -97]; new_z = y: [8, 12].
        np.testing.assert_allclose(lo, [0.0, -103.0, 8.0], atol=1e-4)
        np.testing.assert_allclose(hi, [6.0, -97.0, 12.0], atol=1e-4)

    def test_update_separates_mesh_and_root_axes_transforms(self) -> None:
        mesh_transform = np.eye(4, dtype=np.float32)
        mesh_transform[0, 3] = 10.0
        root_transform = np.eye(4, dtype=np.float32)
        root_transform[1, 3] = 20.0
        self._assert_update_transforms(mesh_transform, root_transform, at_com=False)

    def test_update_separates_mesh_and_com_axes_transforms(self) -> None:
        mesh_transform = np.eye(4, dtype=np.float32)
        mesh_transform[0, 3] = 10.0
        com_transform = np.eye(4, dtype=np.float32)
        com_transform[2, 3] = 30.0
        self._assert_update_transforms(mesh_transform, com_transform, at_com=True)

    def _assert_update_transforms(
        self,
        mesh_transform: np.ndarray,
        axes_transform: np.ndarray,
        *,
        at_com: bool,
    ) -> None:
        renderer = object.__new__(GlbActorRenderer)
        renderer._show_axes_at_com = at_com
        actor = MagicMock()
        actor.get_type.return_value = sdp.ActorType.RIGID
        actor.is_static.return_value = False
        renderer._actor = actor
        renderer._get_glb_transform = MagicMock(return_value=mesh_transform)
        renderer._get_actor_transform = MagicMock(return_value=axes_transform)
        renderer._get_actor_com_transform = MagicMock(return_value=axes_transform)
        renderer.set_transform = MagicMock()
        axes_renderer = MagicMock()
        renderer.get_axes_renderer = MagicMock(return_value=axes_renderer)

        with patch.object(MeshRenderer, "update") as mesh_update:
            renderer.update()

        renderer.set_transform.assert_called_once_with(mesh_transform)
        mesh_update.assert_called_once_with(renderer)
        axes_renderer.set_transform.assert_called_once_with(axes_transform)
        axes_renderer.update.assert_called_once_with()
        if at_com:
            renderer._get_actor_com_transform.assert_called_once_with()
            renderer._get_actor_transform.assert_not_called()
        else:
            renderer._get_actor_transform.assert_called_once_with()
            renderer._get_actor_com_transform.assert_not_called()

    def test_com_mode_falls_back_to_root_for_static_actor(self) -> None:
        mesh_transform = np.eye(4, dtype=np.float32)
        root_transform = np.eye(4, dtype=np.float32)
        root_transform[1, 3] = 20.0
        renderer = object.__new__(GlbActorRenderer)
        renderer._show_axes_at_com = True
        actor = MagicMock()
        actor.get_type.return_value = sdp.ActorType.RIGID
        actor.is_static.return_value = True
        renderer._actor = actor
        renderer._get_glb_transform = MagicMock(return_value=mesh_transform)
        renderer._get_actor_transform = MagicMock(return_value=root_transform)
        renderer._get_actor_com_transform = MagicMock()
        renderer.set_transform = MagicMock()
        axes_renderer = MagicMock()
        renderer.get_axes_renderer = MagicMock(return_value=axes_renderer)

        with patch.object(MeshRenderer, "update"):
            renderer.update()

        axes_renderer.set_transform.assert_called_once_with(root_transform)
        renderer._get_actor_com_transform.assert_not_called()


########################################################################################

if __name__ == "__main__":
    unittest.main()
