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

from test.conftest import assets_dir, mochi, MochiTestBase, requires_internal_assets


class TestPrefab(MochiTestBase):
    @requires_internal_assets
    def test_add_to_scene_returns_result(self):
        """Verify that add_to_scene returns an AddToSceneResult with actors and constraints."""
        prefab = mochi.prefab.load_from_file(
            os.path.join(assets_dir, "samples/rigid_minimal_cube_on_plane.mochi_scene"),
            assets_dir,
        )
        scene = mochi.create_scene("test_add_to_scene_result")
        result = mochi.prefab.add_to_scene(prefab=prefab, scene=scene)

        self.assertIsInstance(result, mochi.prefab.AddToSceneResult)
        self.assertGreater(len(result.actors), 0)
        self.assertEqual(len(result.actors), scene.get_num_actors())

        mochi.destroy_scene(scene)

    @requires_internal_assets
    def test_add_to_scene_result_filter(self):
        """Proof-of-life that AddToSceneResult.filter() bindings work for both overloads.

        Detailed ordering and correctness behavior is covered by the C++ tests.
        """
        # Inline JSON with one rigid actor and one constraint, so both filter overloads
        # have something to find and something to exclude.
        prefab_json = """{
            "actors": {
                "rigid": [{
                    "name": "Box",
                    "shape": "cube/cube_minimal.mochi.json",
                    "colliderType": "Box"
                }]
            },
            "constraints": {
                "rigidPivotPosition": [{
                    "actor": "Box",
                    "stiffness": 1000.0,
                    "targetPosition": [0, 1, 0]
                }]
            }
        }"""
        prefab = mochi.prefab.load_from_json_string(prefab_json, assets_dir)

        scene = mochi.create_scene("test_filter")
        result = mochi.prefab.add_to_scene(prefab=prefab, scene=scene)

        # Filter by ActorType: matching type returns the actor; non-matching returns empty.
        rigids = result.filter(mochi.ActorType.RIGID)
        self.assertEqual(1, len(rigids))
        self.assertEqual(mochi.ActorType.RIGID, rigids[0].get_type())
        self.assertEqual(0, len(result.filter(mochi.ActorType.SOFT)))

        # Filter by ConstraintType: matching type returns the constraint; non-matching is empty.
        positions = result.filter(mochi.ConstraintType.RIGID_PIVOT_POSITION)
        self.assertEqual(1, len(positions))
        self.assertEqual(
            mochi.ConstraintType.RIGID_PIVOT_POSITION, positions[0].get_type()
        )
        self.assertEqual(
            0, len(result.filter(mochi.ConstraintType.RIGID_SPHERICAL_JOINT))
        )

        mochi.destroy_scene(scene)

    def test_prefab_scene_params(self):
        # Test a few default values to prove the C++ constructor ran
        params = mochi.prefab.SceneParams()
        self.assertIsNone(params.comment)
        self.assertEqual("", params.description)
        self.assertIsNone(params.gravity)
        self.assertIsNone(params.solver)

        # Test assignments
        params.comment = "My comment"
        self.assertEqual("My comment", params.comment)
        params.description = "Good stuff"
        self.assertEqual("Good stuff", params.description)
        params.gravity = [0, 0, 0]
        self.assertEqual(mochi.Real3(), params.gravity)
        params.solver = mochi.SolverParams()
        self.assertEqual(mochi.SolverParams(), params.solver)

        # Test keyword argument constructor with all fields in shuffled order.
        params_kw_all = mochi.prefab.SceneParams(
            gravity=[1, 2, 3],
            comment="MyComment",
            description="MyDescription",
            solver=mochi.SolverParams(
                linear_solver=mochi.LinearSolverParams(abs_tol=1e-12)
            ),
        )
        self.assertEqual("MyComment", params_kw_all.comment)
        self.assertEqual("MyDescription", params_kw_all.description)
        self.assertAlmostEqual(mochi.Real3(1, 2, 3), params_kw_all.gravity)
        self.assertAlmostEqual(1e-12, params_kw_all.solver.linear_solver.abs_tol)

        # Test keyword argument constructor with some fields (rest use defaults)
        params_kw_partial = mochi.prefab.SceneParams(
            comment="MyComment2", gravity=[0, -5, 0]
        )
        self.assertAlmostEqual(mochi.Real3(0, -5, 0), params_kw_partial.gravity)
        self.assertEqual("MyComment2", params_kw_partial.comment)
        self.assertEqual("", params_kw_partial.description)
        self.assertIsNone(params_kw_partial.solver)

    @requires_internal_assets
    def test_ensure_fully_loaded(self):
        # Disable file caching so each load creates a distinct shape handle
        was_cache_enabled = mochi.is_file_cache_enabled()
        mochi.enable_file_cache(False)
        try:
            self.assertEqual(0, mochi.get_num_shapes())

            # Create a child prefab file with one rigid actor (duck shape)
            with tempfile.TemporaryDirectory(
                prefix="mochi_pybind_ensure_fully_loaded_"
            ) as temp_dir:
                child_json = """{
                    "actors": {
                        "rigid": [{
                            "name": "ChildActor",
                            "shape": "duck/duck_coarse_mesh.mochi.json"
                        }]
                    }
                }"""
                child_path = os.path.join(temp_dir, "child.mochi_scene")
                with open(child_path, "w") as f:
                    f.write(child_json)

                # Create a parent prefab with one rigid actor (cube shape)
                # and a nested reference to the child
                parent_json = """{
                    "actors": {
                        "rigid": [{
                            "name": "ParentActor",
                            "shape": "cube/cube_minimal.mochi.json"
                        }]
                    },
                    "prefabs": [{
                        "name": "child",
                        "path": "./child.mochi_scene"
                    }]
                }"""
                parent_path = os.path.join(temp_dir, "parent.mochi_scene")
                with open(parent_path, "w") as f:
                    f.write(parent_json)

                # Shallow load — no nested prefabs or shapes loaded yet
                prefab = mochi.prefab.shallow_load_from_file(parent_path)
                self.assertEqual(1, len(prefab.actors.rigid))
                self.assertEqual(1, len(prefab.prefabs))
                self.assertFalse(prefab.actors.rigid[0].shape.is_valid())
                self.assertEqual(0, mochi.get_num_shapes())

                # EnsureFullyLoaded should load the nested prefab and both shapes
                mochi.prefab.ensure_fully_loaded(prefab, assets_dir)

                # Verify shapes were loaded (parent + child)
                self.assertTrue(prefab.actors.rigid[0].shape.is_valid())
                self.assertEqual(2, mochi.get_num_shapes())

                # Verify the nested prefab was loaded by adding to a scene
                scene = mochi.create_scene("test")
                mochi.prefab.add_to_scene(prefab, scene)
                self.assertEqual(2, scene.get_num_actors())
                mochi.destroy_scene(scene)

                # Call EnsureFullyLoaded again — no redundant loads
                mochi.prefab.ensure_fully_loaded(prefab, assets_dir)
                self.assertEqual(2, mochi.get_num_shapes())
        finally:
            mochi.enable_file_cache(was_cache_enabled)

    @requires_internal_assets
    def test_add_to_scene_positional_args_with_params(self):
        """Verify add_to_scene works with positional arguments including PrefabParams."""
        prefab = mochi.prefab.load_from_file(
            os.path.join(assets_dir, "samples/rigid_minimal_cube_on_plane.mochi_scene"),
            assets_dir,
        )
        scene = mochi.create_scene("test_positional_with_params")
        params = mochi.prefab.PrefabParams(name="positional_test")
        result = mochi.prefab.add_to_scene(prefab, scene, params)

        self.assertIsInstance(result, mochi.prefab.AddToSceneResult)
        self.assertGreater(len(result.actors), 0)
        mochi.destroy_scene(scene)

    @requires_internal_assets
    def test_add_to_scene_calling_conventions(self):
        """Verify add_to_scene works with both positional and keyword arguments."""
        prefab = mochi.prefab.load_from_file(
            os.path.join(assets_dir, "samples/rigid_minimal_cube_on_plane.mochi_scene"),
            assets_dir,
        )
        scene = mochi.create_scene("test")
        params = mochi.prefab.PrefabParams(name="keyword_test")
        for result in [
            mochi.prefab.add_to_scene(prefab, scene, params),  # positional arguments
            mochi.prefab.add_to_scene(
                prefab=prefab, scene=scene, params=params
            ),  # keyword arguments
        ]:
            self.assertIsInstance(result, mochi.prefab.AddToSceneResult)
            self.assertGreater(len(result.actors), 0)
        mochi.destroy_scene(scene)

    @requires_internal_assets
    def test_add_to_scene_procedural_contact_filter(self):
        """Proof-of-life that procedural ContactFilter bindings work for all 4 categories.

        Builds a ContactFilter with one entry per category (actor/layer x
        symmetric/asymmetric), attaches it to a prefab loaded from JSON, and
        verifies add_to_scene applies it. Detailed semantics are covered by C++
        tests; this just exercises the Python bindings.
        """
        # Base prefab: 4 rigid actors in 4 distinct layers, no contact filter yet.
        prefab_json = """{
            "actors": {
                "rigid": [
                    {"name": "Floor", "layer": "LayerA", "isStatic": true,
                     "colliderType": "Box", "shape": "cube/cube_minimal.mochi.json"},
                    {"name": "CubeA", "layer": "LayerB",
                     "colliderType": "Box", "shape": "cube/cube_minimal.mochi.json"},
                    {"name": "CubeB", "layer": "LayerC",
                     "colliderType": "Box", "shape": "cube/cube_minimal.mochi.json"},
                    {"name": "CubeC", "layer": "LayerD",
                     "colliderType": "Box", "shape": "cube/cube_minimal.mochi.json"}
                ]
            }
        }"""
        prefab = mochi.prefab.load_from_json_string(prefab_json, assets_dir)
        self.assertIsNone(prefab.contact_filter)

        # Procedurally build a ContactFilter exercising all 4 categories.
        self.assertTrue(mochi.prefab.ActorContactEntry().include_nested_actors)
        contact_filter = mochi.prefab.ContactFilter()
        contact_filter.actor_contact_symmetric = [
            mochi.prefab.ActorContactEntry(
                enable=False, actors=["CubeA", "CubeB"], include_nested_actors=False
            ),
        ]
        contact_filter.actor_contact_asymmetric = [
            mochi.prefab.ActorContactEntry(
                enable=False, actors=["CubeC", "Floor"], include_nested_actors=False
            ),
        ]
        contact_filter.layer_contact_symmetric = [
            mochi.prefab.LayerContactEntry(enable=False, layers=["LayerB", "LayerC"]),
        ]
        contact_filter.layer_contact_asymmetric = [
            mochi.prefab.LayerContactEntry(enable=False, layers=["LayerD", "LayerA"]),
        ]
        prefab.contact_filter = contact_filter
        self.assertFalse(
            prefab.contact_filter.actor_contact_symmetric[0].include_nested_actors
        )

        # Round-trip check: each category came back with the value we set.
        self.assertIsNotNone(prefab.contact_filter)
        self.assertEqual(
            ["CubeA", "CubeB"],
            list(prefab.contact_filter.actor_contact_symmetric[0].actors),
        )
        self.assertEqual(
            ["CubeC", "Floor"],
            list(prefab.contact_filter.actor_contact_asymmetric[0].actors),
        )
        self.assertEqual(
            ["LayerB", "LayerC"],
            list(prefab.contact_filter.layer_contact_symmetric[0].layers),
        )
        self.assertEqual(
            ["LayerD", "LayerA"],
            list(prefab.contact_filter.layer_contact_asymmetric[0].layers),
        )

        # add_to_scene must accept the procedurally-built filter and apply it.
        scene = mochi.create_scene("test_procedural_contact_filter")
        mochi.prefab.add_to_scene(prefab=prefab, scene=scene)

        # Layer effects are observable from Python — verify both layer categories
        # took effect (the actor-side bindings are exercised but not separately
        # observable in Python; their semantics are covered by C++ tests).
        self.assertFalse(scene.is_layer_contact_enabled("LayerB", "LayerC"))
        self.assertFalse(scene.is_layer_contact_enabled("LayerC", "LayerB"))
        self.assertFalse(scene.is_layer_contact_enabled("LayerD", "LayerA"))
        self.assertTrue(scene.is_layer_contact_enabled("LayerA", "LayerD"))
        self.assertTrue(scene.is_layer_contact_enabled("LayerA", "LayerB"))

        mochi.destroy_scene(scene)

    @requires_internal_assets
    def test_add_to_scene_contact_pair_params_override(self):
        prefab_json = """{
            "actors": {
                "rigid": [
                    {"name": "ActorA", "colliderType": "Box",
                     "shape": "cube/cube_minimal.mochi.json"},
                    {"name": "ActorB", "colliderType": "Box",
                     "shape": "cube/cube_minimal.mochi.json"}
                ]
            }
        }"""
        prefab = mochi.prefab.load_from_json_string(prefab_json, assets_dir)
        self.assertIsNone(prefab.contact_pair_params_overrides)

        entry = mochi.prefab.ContactPairParamsOverrideEntry(
            actors=["ActorA", "ActorB"],
            params_override=mochi.ContactPairParamsOverride(
                penalty_coefficient=2e9,
                coulomb_friction_coefficient=0.0,
            ),
        )
        prefab.contact_pair_params_overrides = [entry]
        self.assertEqual(
            ["ActorA", "ActorB"],
            list(prefab.contact_pair_params_overrides[0].actors),
        )

        scene = mochi.create_scene("test_contact_pair_params_override")
        result = mochi.prefab.add_to_scene(prefab=prefab, scene=scene)
        actors = {actor.get_name(): actor.get_handle() for actor in result.actors}
        stored = scene.get_contact_pair_params_override(
            actors["ActorB"], actors["ActorA"]
        )
        self.assertAlmostEqual(2e9, stored.penalty_coefficient)
        self.assertEqual(0.0, stored.coulomb_friction_coefficient)
        self.assertIsNone(stored.viscous_friction_coefficient)

        mochi.destroy_scene(scene)

    @requires_internal_assets
    def test_add_to_scene_file_path_overload(self):
        """Verify the file-path overload of add_to_scene with default and custom params."""
        prefab_path = os.path.join(
            assets_dir, "samples/rigid_minimal_cube_on_plane.mochi_scene"
        )

        # File-path overload with default params omitted.
        scene = mochi.create_scene("test_file_path_default")
        result = mochi.prefab.add_to_scene(
            prefab_path=prefab_path, root_path=assets_dir, scene=scene
        )
        self.assertGreater(len(result.actors), 0)
        mochi.destroy_scene(scene)

        # File-path overload with keyword params.
        scene = mochi.create_scene("test_file_path_with_params")
        params = mochi.prefab.PrefabParams(name="file_path_test")
        result = mochi.prefab.add_to_scene(
            prefab_path=prefab_path,
            root_path=assets_dir,
            scene=scene,
            params=params,
        )
        self.assertGreater(len(result.actors), 0)
        mochi.destroy_scene(scene)
