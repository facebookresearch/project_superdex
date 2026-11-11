#!/usr/bin/env python3
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

"""Tests for mochi_mesh tetrahedralization Python bindings."""

import numpy as np
from test.conftest import (
    create_cube_faces_closed,
    create_cube_faces_with_hole,
    create_cube_vertices,
    mochi_mesh,
    MochiMeshTestBase,
)


class TetrahedralizationTest(MochiMeshTestBase):
    """Tests for tetrahedralization bindings."""

    def test_tetrahedralize_closed_cube(self):
        params = mochi_mesh.TetrahedralizationParams()
        params.edge_size = 0.5
        params.facet_distance_bound = 0.1
        params.facet_angle_bound = 20.0

        self.assertEqual(params.facet_distance_bound, 0.1)
        self.assertEqual(params.facet_angle_bound, 20.0)

        verts, tets = mochi_mesh.tetrahedralize(
            create_cube_vertices(), create_cube_faces_closed(), params
        )

        self.assertEqual(verts.shape[1], 3)
        self.assertEqual(tets.shape[1], 4)
        self.assertGreater(tets.shape[0], 0)
        self.assertEqual(np.unique(tets).size, verts.shape[0])

    def test_tetrahedralize_with_default_params(self):
        _, tets = mochi_mesh.tetrahedralize(
            create_cube_vertices(), create_cube_faces_closed()
        )
        self.assertGreater(tets.shape[0], 0)

    def test_open_mesh_raises(self):
        with self.assertRaises(mochi_mesh.Error):
            mochi_mesh.tetrahedralize(
                create_cube_vertices(), create_cube_faces_with_hole()
            )

    def test_tetrahedralize_constrained_keeps_input_vertices_first(self):
        params = mochi_mesh.ConstrainedTetrahedralizationParams()
        params.edge_size = 0.5
        params.iterations = 3

        self.assertEqual(params.iterations, 3)
        self.assertFalse(params.strict)

        input_verts = create_cube_vertices()
        verts, tets = mochi_mesh.tetrahedralize_constrained(
            input_verts, create_cube_faces_closed(), params
        )

        self.assertEqual(tets.shape[1], 4)
        self.assertGreater(tets.shape[0], 0)
        self.assertEqual(np.unique(tets).size, verts.shape[0])
        np.testing.assert_array_equal(verts[: input_verts.shape[0]], input_verts)

    def test_tetrahedralize_constrained_strict_raises_when_vertices_needed(self):
        params = mochi_mesh.ConstrainedTetrahedralizationParams()
        params.strict = True
        # The cube's cospherical vertices need added boundary vertices.
        with self.assertRaises(mochi_mesh.Error):
            mochi_mesh.tetrahedralize_constrained(
                create_cube_vertices(), create_cube_faces_closed(), params
            )
