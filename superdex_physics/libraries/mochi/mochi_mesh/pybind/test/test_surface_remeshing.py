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

"""Tests for mochi_mesh surface remeshing Python bindings."""

from test.conftest import (
    create_cube_faces_closed,
    create_cube_faces_with_hole,
    create_cube_vertices,
    mochi_mesh,
    MochiMeshTestBase,
)


class SurfaceRemeshingTest(MochiMeshTestBase):
    """Tests for surface remeshing bindings."""

    def test_remesh_closed_cube(self):
        vertices = create_cube_vertices()
        faces = create_cube_faces_closed()

        params = mochi_mesh.SurfaceRemeshingParams()
        params.relative_to_mesh_size = False
        params.edge_size = 1.0

        result_verts, result_faces = mochi_mesh.remesh_surface(vertices, faces, params)

        self.assertEqual(result_verts.ndim, 2)
        self.assertEqual(result_verts.shape[1], 3)
        self.assertEqual(result_faces.ndim, 2)
        self.assertEqual(result_faces.shape[1], 3)
        self.assertGreater(result_verts.shape[0], 0)
        self.assertGreater(result_faces.shape[0], 0)

    def test_remesh_with_default_params(self):
        vertices = create_cube_vertices()
        faces = create_cube_faces_closed()

        result_verts, result_faces = mochi_mesh.remesh_surface(vertices, faces)

        self.assertGreater(result_verts.shape[0], 0)
        self.assertGreater(result_faces.shape[0], 0)

    def test_remesh_open_mesh(self):
        vertices = create_cube_vertices()
        faces = create_cube_faces_with_hole()

        params = mochi_mesh.SurfaceRemeshingParams()
        params.relative_to_mesh_size = False
        params.edge_size = 1.0

        result_verts, result_faces = mochi_mesh.remesh_surface(vertices, faces, params)

        self.assertGreater(result_verts.shape[0], 0)
        self.assertGreater(result_faces.shape[0], 0)

    def test_none_method(self):
        vertices = create_cube_vertices()
        faces = create_cube_faces_closed()
        params = mochi_mesh.SurfaceRemeshingParams()
        params.method = mochi_mesh.RemeshMethod.NONE
        params.relative_to_mesh_size = False
        params.edge_size = 1.0
        result_verts, result_faces = mochi_mesh.remesh_surface(vertices, faces, params)
        self.assertGreater(result_verts.shape[0], 0)
        self.assertGreater(result_faces.shape[0], 0)

    def test_compute_mesh_statistics(self):
        vertices = create_cube_vertices()
        faces = create_cube_faces_closed()

        stats = mochi_mesh.compute_mesh_statistics(vertices, faces)

        self.assertEqual(stats.num_vertices, 8)
        self.assertEqual(stats.num_faces, 12)
        self.assertGreater(stats.edge_lengths.min, 0.0)
        self.assertGreater(stats.angles.min, 0.0)
        self.assertLess(stats.angles.max, 180.0)
        self.assertTrue(stats.is_closed)
        self.assertEqual(stats.hausdorff_distance, -1.0)

    def test_compute_mesh_statistics_with_reference(self):
        vertices = create_cube_vertices()
        faces = create_cube_faces_closed()

        stats = mochi_mesh.compute_mesh_statistics(
            vertices, faces, ref_vertices=vertices, ref_faces=faces
        )

        self.assertGreaterEqual(stats.hausdorff_distance, 0.0)
