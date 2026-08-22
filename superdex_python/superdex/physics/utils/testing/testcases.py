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

import unittest
from contextlib import contextmanager
from typing import Generator

import superdex.physics as mochi
from superdex.physics import Actor, Quaternion, Real3, Scene, TransformRT
from superdex.physics.paths import resolve_asset

########################################################################################


class MochiContextTestCase(unittest.TestCase):
    """Base class for physics unit tests that require an initialized SuperDex Physics context."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize SuperDex Physics before running any tests."""
        if not mochi.is_initialized():
            mochi.initialize(num_worker_threads=0)

    @classmethod
    def tearDownClass(cls) -> None:
        """Shut down SuperDex Physics after all tests have completed."""
        if mochi.is_initialized():
            mochi.shutdown()


########################################################################################


def _add_rigid_object(
    scene: Scene,
    name: str,
    relative_shape_path: str,
    position: Real3 | None = None,
    rotation: Quaternion | None = None,
) -> Actor:
    """Adds a rigid object to the scene."""

    # Load the shape asset.
    asset_path = resolve_asset(relative_shape_path)
    shape = mochi.load_shape_from_file(str(asset_path))

    # Create the rigid actor.
    actor = scene.create_rigid_actor(name=name, shape=shape)

    # Set the transform to the desired location.
    world_from_local = TransformRT.identity()
    if position is not None:
        world_from_local.translation = position
    if rotation is not None:
        world_from_local.rotation = rotation
    if actor is None:
        raise RuntimeError(f"Failed to create rigid actor: {name}")
    actor.set_root_transform(world_from_local)
    return actor


def add_rigid_cube(
    scene: Scene,
    name: str,
    position: Real3 | None = None,
    rotation: Quaternion | None = None,
) -> Actor:
    """Adds a rigid cube to the scene."""
    CUBE_SHAPE_PATH = "cube/cube_mesh.mochi.h5"
    return _add_rigid_object(scene, name, CUBE_SHAPE_PATH, position, rotation)


def add_rigid_sphere(
    scene: Scene,
    name: str,
    position: Real3 | None = None,
    rotation: Quaternion | None = None,
) -> Actor:
    """Adds a rigid sphere to the scene."""
    SPHERE_SHAPE_PATH = "sphere/icosphere_3subdiv.1.mochi.json"
    return _add_rigid_object(scene, name, SPHERE_SHAPE_PATH, position, rotation)


########################################################################################


@contextmanager
def make_empty_scene() -> Generator[Scene, None, None]:
    """Creates an empty scene and returns it."""
    scene = mochi.create_scene("")
    if scene is None:
        raise RuntimeError("Failed to create empty scene")
    yield scene
    mochi.destroy_scene(scene)


@contextmanager
def make_single_rigid_cube_scene() -> Generator[Scene, None, None]:
    """Creates a scene with a single rigid cube."""
    scene = mochi.create_scene("")
    if scene is None:
        raise RuntimeError("Failed to create scene")
    add_rigid_cube(scene, "Cube")
    yield scene
    mochi.destroy_scene(scene)
