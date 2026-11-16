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

import numpy as np
from test.conftest import MochiTestBase, sdp, sdr


class TactileObservationTest(MochiTestBase):
    def test_rect_projection_options_drive_mapping(self) -> None:
        layout = sdr.make_tactile_taxel_layout([[1.0, 0.5, 1.0]])
        coordinates = []
        for projection_type in (
            sdr.TactileRectProjectionType.AZIMUTHAL_EQUAL_AREA,
            sdr.TactileRectProjectionType.AZIMUTHAL_EQUIDISTANT,
            sdr.TactileRectProjectionType.OCTAHEDRAL,
        ):
            projection = sdr.build_tactile_rect_projection(
                layout,
                sdr.TactileHemisphereRectParams(
                    width=16, height=16, projection=projection_type
                ),
            )
            coordinates.append(np.asarray(projection.coordinates)[0])

        self.assertFalse(np.allclose(coordinates[0], coordinates[1]))
        self.assertFalse(np.allclose(coordinates[0], coordinates[2]))
        self.assertFalse(np.allclose(coordinates[1], coordinates[2]))

        outside_layout = sdr.make_tactile_taxel_layout(
            [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]
        )
        with self.assertRaises(sdp.Error):
            sdr.build_tactile_rect_projection(
                outside_layout, sdr.TactileHemisphereRectParams(width=16, height=16)
            )

        excluded = sdr.build_tactile_rect_projection(
            outside_layout,
            sdr.TactileHemisphereRectParams(
                width=16, height=16, outside_policy=sdr.TactileRectOutsidePolicy.EXCLUDE
            ),
        )
        self.assertEqual(excluded.selected_taxel_count, 1)
        self.assertEqual(int(np.asarray(excluded.taxel_cells)[1]), -1)

    def test_force_only_layer_two_conserves_layer_one_sum(self) -> None:
        layer_one = sdr.TactileTaxelForces(
            forces=[[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0], [0.5, -0.5, 1.0]]
        )

        layer_two = sdr.aggregate_tactile_force(layer_one)

        np.testing.assert_allclose(layer_two.force, [-2.5, 6.5, -2.0])

    def test_layout_rejects_non_finite_positions(self) -> None:
        with self.assertRaises(sdp.Error):
            sdr.make_tactile_taxel_layout([[0.0, 0.0, np.inf]])

    def test_hemisphere_rect_maps_tip_and_rim_and_conserves_force(self) -> None:
        layout = sdr.make_tactile_taxel_layout(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        layer_one = sdr.TactileTaxelForces(
            forces=[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
        )
        params = sdr.TactileHemisphereRectParams(
            width=4,
            height=2,
            rasterization=sdr.TactileRectRasterizationType.NEAREST_CELL,
        )

        grid = sdr.map_tactile_hemisphere_to_rect(layout, layer_one, params)

        forces = np.asarray(grid.forces).reshape(2, 4, 3)
        counts = np.asarray(grid.taxel_counts).reshape(2, 4)
        np.testing.assert_allclose(forces[0, 2], [0.0, 0.0, 3.0])
        np.testing.assert_allclose(forces[1, 2], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(forces[1, 3], [0.0, 2.0, 0.0])
        self.assertEqual(int(counts.sum()), 3)
        np.testing.assert_allclose(forces.sum(axis=(0, 1)), [1.0, 2.0, 3.0])

    def test_conservative_voronoi_fills_grid_and_conserves_force(self) -> None:
        layout = sdr.make_tactile_taxel_layout([[0.0, 0.0, 1.0]])
        layer_one = sdr.TactileTaxelForces(forces=[[16.0, 32.0, 48.0]])
        params = sdr.TactileHemisphereRectParams(width=4, height=4)

        projection = sdr.build_tactile_rect_projection(layout, params)
        grid = sdr.rasterize_tactile_rect_projection(projection, layer_one)

        self.assertEqual(
            projection.rasterization,
            sdr.TactileRectRasterizationType.CONSERVATIVE_VORONOI,
        )
        self.assertEqual(projection.voronoi_oversampling, 4)
        forces = np.asarray(grid.forces).reshape(4, 4, 3)
        np.testing.assert_allclose(
            forces, np.broadcast_to([1.0, 2.0, 3.0], forces.shape)
        )
        np.testing.assert_allclose(forces.sum(axis=(0, 1)), layer_one.forces[0])

    def test_rect_projection_exposes_even_center_and_surface_selection(self) -> None:
        layout = sdr.make_tactile_taxel_layout([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
        params = sdr.TactileHemisphereRectParams(
            width=256,
            height=256,
            rasterization=sdr.TactileRectRasterizationType.NEAREST_CELL,
            outside_policy=sdr.TactileRectOutsidePolicy.EXCLUDE,
        )

        projection = sdr.build_tactile_rect_projection(layout, params)

        np.testing.assert_allclose(np.asarray(projection.coordinates)[0], [0.0, 0.0])
        self.assertEqual(int(np.asarray(projection.taxel_cells)[0]), 128 * 256 + 128)
        self.assertEqual(int(np.asarray(projection.taxel_cells)[1]), -1)
        self.assertEqual(projection.selected_taxel_count, 1)
