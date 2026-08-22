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

import math
import os
import tempfile

from test.conftest import assets_dir, mochi, MochiTestBase, np, np_real, requires_hdf5


class TestModel(MochiTestBase):
    @requires_hdf5
    def test_model_load_from_x(self):
        files_to_load = ["cube/cube_mesh.mochi.json", "cube/cube_mesh.mochi.h5"]
        for path in files_to_load:
            full_path = os.path.join(assets_dir, path)

            # model.load_from_file
            model = mochi.model.load_from_file(path=full_path)
            self.assertIsNotNone(model.mesh)
            self.assertEqual(4, model.mesh.nodes_per_element)
            self.assertNotEqual(0, len(model.mesh.coordinates))
            self.assertNotEqual(0, len(model.mesh.connectivity))

            # model.load_from_file_unchecked
            model2 = mochi.model.load_from_file_unchecked(path=full_path)
            mochi.model.auto_correct(model2)
            mochi.model.validate(model2)
            self.assertEqual(model, model2)

            # model.load_from_bytes
            data = None
            with open(full_path, "rb") as f:
                data = f.read()
            model2 = mochi.model.load_from_bytes(data)
            self.assertEqual(model, model2)

            # model.load_from_bytes_unchecked
            model2 = mochi.model.load_from_bytes_unchecked(data)
            mochi.model.auto_correct(model2)
            mochi.model.validate(model2)
            self.assertEqual(model, model2)

    @requires_hdf5
    def test_model_save_to_x(self):
        files_to_load = ["cube/cube_mesh.mochi.json", "cube/cube_mesh.mochi.h5"]
        for path in files_to_load:
            # model.save_to_json_string
            full_path = os.path.join(assets_dir, path)
            model = mochi.model.load_from_file(full_path)
            json = mochi.model.save_to_json_string(model)
            json_bytes = bytes(json, "utf-8")
            model2 = mochi.model.load_from_bytes(json_bytes)
            self.assertEqual(model, model2)  # Round trip was lossless
            shape = mochi.load_shape_from_bytes(
                json_bytes
            )  # Prove that Mochi can a ShapeHandle from it too
            self.assertTrue(shape.is_valid())
            mochi.release_shape(shape)

            # model.save_to_file
            for fmt in [mochi.FileFormat.JSON, mochi.FileFormat.H5]:
                for useKeywords in [False, True]:
                    with (
                        tempfile.NamedTemporaryFile(
                            suffix=".json" if (fmt == mochi.FileFormat.JSON) else ".h5",
                            prefix="mochi_physics_pybind_test_",
                            delete=True,  # default, explicit for clarity
                            delete_on_close=False,  # keeps file on disk until 'with' block exits
                        ) as tmp
                    ):
                        path = tmp.name
                        tmp.close()  # close the handle so we can overwrite this the temp file
                        if useKeywords:
                            mochi.model.save_to_file(data=model, path=path, format=fmt)
                        else:
                            mochi.model.save_to_file(model, path, fmt)
                        model2 = mochi.model.load_from_file(path=path)
                        self.assertEqual(model, model2)
                        shape = mochi.load_shape_from_file(file_path=path)
                        self.assertTrue(shape.is_valid())
                        mochi.release_shape(shape)

    @requires_hdf5
    def test_model_auto_correct_and_validate(self):
        # Load a model with a visual mesh
        model = mochi.model.load_from_file(
            os.path.join(assets_dir, "duck/duck_coarse.mochi.h5")
        )
        self.assertIsNotNone(model.visual_mesh)
        self.assertIsNotNone(model.visual_mesh.skinning)
        mochi.model.validate(model)  # Starts valid

        # Double all the skinning weights
        for i in range(len(model.visual_mesh.skinning.weights)):
            model.visual_mesh.skinning.weights[i] = (
                model.visual_mesh.skinning.weights[i] * 2
            )

        # It should now fail validation
        try:
            mochi.model.validate(model)
        except mochi.Error:
            pass

        # But auto_correct should fix it
        mochi.model.auto_correct(model)
        mochi.model.validate(model)

    def test_model_bake_transform(self):
        # Load a model with a mesh
        src_model = mochi.model.load_from_file(
            os.path.join(assets_dir, "cube/cube_mesh.mochi.json")
        )

        # Copy the original mesh coordinates into an Nx3 numpy array
        num_nodes = src_model.mesh.get_num_nodes()
        src_coords_flat = np.array(src_model.mesh.coordinates, dtype=np_real)
        src_coords = src_coords_flat.reshape((num_nodes, 3))

        model = mochi.ModelData(src_model)  # Deep copy

        # Apply a non-uniform scale via BakeTransform
        scale = [0.1, 0.2, 0.3]
        mochi.model.bake_transform(data=model, scale=scale)

        # Compute the expected coordinates and compare
        actual_coords = np.array(model.mesh.coordinates, dtype=np_real).reshape(
            (num_nodes, 3)
        )
        expected_coords = src_coords * np.array(scale, dtype=np_real)
        np.testing.assert_allclose(actual_coords, expected_coords, atol=1e-6)

        for useRT in [False, True]:
            # Repeat, but this time bake in a 90 degree rotation and translation
            model = mochi.ModelData(src_model)  # Deep copy
            rotation = mochi.Quaternion.rotation_z(math.pi / 2)
            translation = [1, 2, 3]
            if useRT:
                mochi.model.bake_transform(
                    data=model, transform=mochi.TransformRT(rotation, translation)
                )
            else:
                mochi.model.bake_transform(
                    data=model, rotation=rotation, translation=translation
                )

            # A 90-degree rotation around Z maps (x, y, z) -> (-y, x, z),
            # followed by the translation.
            actual_coords = np.array(model.mesh.coordinates, dtype=np_real).reshape(
                (num_nodes, 3)
            )
            rot_coords = np.column_stack(
                (-src_coords[:, 1], src_coords[:, 0], src_coords[:, 2])
            )
            expected_coords = rot_coords + np.array(translation, dtype=np_real)
            np.testing.assert_allclose(actual_coords, expected_coords, atol=1e-6)

    @requires_hdf5
    def test_model_bake_sdf(self):
        model = mochi.model.load_from_file(
            os.path.join(assets_dir, "duck/duck_coarse.mochi.h5")
        )
        model.sdf = None  # Discard the SDF if it came with one

        # Bake with default parameters
        mochi.model.bake_sdf(model)
        self.assertIsNotNone(model.sdf)
        self.assertNotEqual(0, len(model.sdf.values))

        # Remove the SDF from the model, but hold onto it for now
        prev_sdf = mochi.GridSdfData(model.sdf)  # Deep copy
        model.sdf = None

        # Repeat with non-default parameters
        boundary_padding = 0.5
        mochi.model.bake_sdf(
            model,
            params=mochi.GridSdfParams(boundary_padding_dist=boundary_padding),
        )
        self.assertIsNotNone(model.sdf)
        self.assertNotEqual(0, len(model.sdf.values))
        for i in range(3):
            # Expect greater grid resolution due to boundary_padding_dist
            self.assertGreaterEqual(model.sdf.dims[i], prev_sdf.dims[i])

    def _make_cube_ply_bytes(self):
        lines = [
            "ply",
            "format ascii 1.0",
            "element vertex 8",
            "property float x",
            "property float y",
            "property float z",
            "element face 12",
            "property list uchar int vertex_indices",
            "end_header",
            "0 0 0",
            "2 0 0",
            "2 2 0",
            "0 2 0",
            "0 0 2",
            "2 0 2",
            "2 2 2",
            "0 2 2",
            "3 0 2 1",
            "3 0 3 2",
            "3 4 5 6",
            "3 4 6 7",
            "3 0 1 5",
            "3 0 5 4",
            "3 2 7 6",
            "3 2 3 7",
            "3 0 4 7",
            "3 0 7 3",
            "3 1 6 5",
            "3 1 2 6",
        ]
        return "\n".join(lines).encode("utf-8")

    def _make_cube_off_bytes(self):
        lines = [
            "OFF",
            "8 12 0",
            "0 0 0",
            "2 0 0",
            "2 2 0",
            "0 2 0",
            "0 0 2",
            "2 0 2",
            "2 2 2",
            "0 2 2",
            "3 0 2 1",
            "3 0 3 2",
            "3 4 5 6",
            "3 4 6 7",
            "3 0 1 5",
            "3 0 5 4",
            "3 2 7 6",
            "3 2 3 7",
            "3 0 4 7",
            "3 0 7 3",
            "3 1 6 5",
            "3 1 2 6",
        ]
        return "\n".join(lines).encode("utf-8")

    def _make_cube_obj_bytes(self):
        lines = [
            "v 0 0 0",
            "v 2 0 0",
            "v 2 2 0",
            "v 0 2 0",
            "v 0 0 2",
            "v 2 0 2",
            "v 2 2 2",
            "v 0 2 2",
            "f 1 3 2",
            "f 1 4 3",
            "f 5 6 7",
            "f 5 7 8",
            "f 1 2 6",
            "f 1 6 5",
            "f 3 8 7",
            "f 3 4 8",
            "f 1 5 8",
            "f 1 8 4",
            "f 2 7 6",
            "f 2 3 7",
        ]
        return "\n".join(lines).encode("utf-8")

    def _expect_cube_mesh(self, model, expected_num_nodes):
        self.assertIsNotNone(model.mesh)
        self.assertEqual(3, model.mesh.nodes_per_element)
        self.assertEqual(expected_num_nodes, model.mesh.get_num_nodes())
        self.assertEqual(12, model.mesh.get_num_elements())

    def test_mesh_file_type_enum(self):
        self.assertNotEqual(mochi.MeshFileType.PLY, mochi.MeshFileType.LEGACY)
        for value in [
            mochi.MeshFileType.LEGACY,
            mochi.MeshFileType.PLY,
            mochi.MeshFileType.OFF,
            mochi.MeshFileType.STL,
            mochi.MeshFileType.OBJ,
        ]:
            self.assertIsNotNone(value)

    def test_load_from_bytes_without_format(self):
        full_path = os.path.join(assets_dir, "cube/cube_mesh.mochi.json")
        with open(full_path, "rb") as f:
            data = f.read()
        model = mochi.model.load_from_bytes(data)
        self.assertIsNotNone(model.mesh)
        self.assertEqual(4, model.mesh.nodes_per_element)

    def test_load_from_bytes_unchecked_without_format(self):
        full_path = os.path.join(assets_dir, "cube/cube_mesh.mochi.json")
        with open(full_path, "rb") as f:
            data = f.read()
        model = mochi.model.load_from_bytes_unchecked(data)
        self.assertIsNotNone(model.mesh)
        self.assertEqual(4, model.mesh.nodes_per_element)

    def test_load_from_bytes_with_ply_format(self):
        data = self._make_cube_ply_bytes()
        model = mochi.model.load_from_bytes(data, format=mochi.MeshFileType.PLY)
        self._expect_cube_mesh(model, 8)

    def test_load_from_bytes_with_off_format(self):
        data = self._make_cube_off_bytes()
        model = mochi.model.load_from_bytes(data, format=mochi.MeshFileType.OFF)
        self._expect_cube_mesh(model, 8)

    def test_load_from_bytes_with_obj_format(self):
        data = self._make_cube_obj_bytes()
        model = mochi.model.load_from_bytes(data, format=mochi.MeshFileType.OBJ)
        self._expect_cube_mesh(model, 8)

    def test_load_from_bytes_unchecked_with_ply_format(self):
        data = self._make_cube_ply_bytes()
        model = mochi.model.load_from_bytes_unchecked(
            data, format=mochi.MeshFileType.PLY
        )
        self._expect_cube_mesh(model, 8)

    def test_load_shape_from_bytes_without_format(self):
        full_path = os.path.join(assets_dir, "cube/cube_mesh.mochi.json")
        with open(full_path, "rb") as f:
            data = f.read()
        shape = mochi.load_shape_from_bytes(data)
        self.assertTrue(shape.is_valid())
        mochi.release_shape(shape)

    def test_load_shape_from_bytes_with_ply_format(self):
        data = self._make_cube_ply_bytes()
        shape = mochi.load_shape_from_bytes(data, format=mochi.MeshFileType.PLY)
        self.assertTrue(shape.is_valid())
        mochi.release_shape(shape)

    def test_load_shape_from_bytes_with_off_format(self):
        data = self._make_cube_off_bytes()
        shape = mochi.load_shape_from_bytes(data, format=mochi.MeshFileType.OFF)
        self.assertTrue(shape.is_valid())
        mochi.release_shape(shape)

    def test_model_flip_winding_order(self):
        model = mochi.model.load_from_file(
            os.path.join(assets_dir, "cube/cube_mesh.mochi.json")
        )
        self.assertIsNotNone(model.mesh)
        self.assertEqual(4, model.mesh.nodes_per_element)
        self.assertNotEqual(0, len(model.mesh.connectivity))
        model2 = mochi.ModelData(model)  # Deep copy
        mochi.model.flip_winding_order(model2)
        self.assertNotEqual(model, model2)

        # Flipping must negate each tetrahedron's signed volume.
        coords = np.array(model.mesh.coordinates).reshape(-1, 3)
        c1 = model.mesh.connectivity
        c2 = model2.mesh.connectivity
        n = model.mesh.nodes_per_element
        for i in range(0, len(c1), n):
            a = coords[c1[i]]
            e1 = coords[c1[i + 1]] - a
            e2 = coords[c1[i + 2]] - a
            e3 = coords[c1[i + 3]] - a
            v1 = np.dot(e1, np.cross(e2, e3))

            a2 = coords[c2[i]]
            f1 = coords[c2[i + 1]] - a2
            f2 = coords[c2[i + 2]] - a2
            f3 = coords[c2[i + 3]] - a2
            v2 = np.dot(f1, np.cross(f2, f3))

            self.assertGreater(abs(v1), 1e-6)
            self.assertAlmostEqual(v2, -v1, places=6)

        mochi.model.flip_winding_order(model2)
        self.assertEqual(model, model2)  # Equal again

        # The MeshData overload must produce the same result as the ModelData overload.
        model3 = mochi.ModelData(model)  # Deep copy
        mochi.model.flip_winding_order(model3)
        mesh = self._copy_mesh(model.mesh)
        mochi.model.flip_winding_order(mesh)
        self.assertEqual(model3.mesh, mesh)

    @staticmethod
    def _copy_mesh(mesh):
        # ModelData.mesh is an optional field, so Python reads it back by value. Rebuild the
        # MeshData explicitly to get an object the MeshData overloads can mutate in place.
        return mochi.MeshData(
            nodes_per_element=mesh.nodes_per_element,
            coordinates=mesh.coordinates,
            connectivity=mesh.connectivity,
            skinning=mesh.skinning,
        )

    def test_model_bake_coordinate_space_transform(self):
        # Mochi's default space is X-forward, Y-left, Z-up in meters. Unreal is X-forward,
        # Y-right, Z-up in centimeters, so the conversion maps (x, y, z) -> (x, -y, z) * 100.
        mochi_space = mochi.CoordinateSpace.default()
        unreal_space = mochi.CoordinateSpace.unreal()

        src_model = mochi.model.load_from_file(
            os.path.join(assets_dir, "cube/cube_mesh.mochi.json")
        )
        src_coords = np.array(src_model.mesh.coordinates, dtype=np_real).reshape(-1, 3)
        expected_coords = 100.0 * np.column_stack(
            (src_coords[:, 0], -src_coords[:, 1], src_coords[:, 2])
        )

        # ModelData overload
        model = mochi.ModelData(src_model)  # Deep copy
        mochi.model.bake_coordinate_space_transform(model, mochi_space, unreal_space)
        actual_coords = np.array(model.mesh.coordinates, dtype=np_real).reshape(-1, 3)
        np.testing.assert_allclose(actual_coords, expected_coords, rtol=1e-6)

        # Unreal is left handed, so the winding order must have been reversed too.
        self.assertNotEqual(
            list(src_model.mesh.connectivity), list(model.mesh.connectivity)
        )

        # MeshData overload must agree with the ModelData overload.
        mesh = self._copy_mesh(src_model.mesh)
        mochi.model.bake_coordinate_space_transform(mesh, mochi_space, unreal_space)
        self.assertEqual(model.mesh, mesh)

        # Converting back must restore the original model.
        mochi.model.bake_coordinate_space_transform(model, unreal_space, mochi_space)
        np.testing.assert_allclose(
            np.array(model.mesh.coordinates, dtype=np_real).reshape(-1, 3),
            src_coords,
            rtol=1e-6,
        )
        self.assertEqual(
            list(src_model.mesh.connectivity), list(model.mesh.connectivity)
        )

        # An invalid coordinate space is reported as an error.
        invalid_space = mochi.CoordinateSpace(
            axes=mochi.CoordinateSpaceAxes.FLU, units_per_meter=0.0
        )
        with self.assertRaises(mochi.Error):
            mochi.model.bake_coordinate_space_transform(
                model, mochi_space, invalid_space
            )
        with self.assertRaises(mochi.Error):
            mochi.model.bake_coordinate_space_transform(
                mesh, mochi_space, invalid_space
            )
