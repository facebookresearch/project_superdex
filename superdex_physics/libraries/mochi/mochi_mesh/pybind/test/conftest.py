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

import numpy as np

try:
    import mochi_mesh  # @manual

    np_real = np.float32
except ImportError:
    import mochi_mesh_double as mochi_mesh  # @manual  # noqa: F401

    np_real = np.float64


def create_cube_vertices() -> np.ndarray:
    """Cube vertices (8 vertices, unit cube)."""
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np_real,
    )


def create_cube_faces_closed() -> np.ndarray:
    """12 triangles forming a watertight cube."""
    return np.array(
        [
            [0, 2, 1],
            [0, 3, 2],  # Bottom
            [4, 5, 6],
            [4, 6, 7],  # Top
            [0, 1, 5],
            [0, 5, 4],  # Front
            [2, 7, 6],
            [2, 3, 7],  # Back
            [0, 4, 7],
            [0, 7, 3],  # Left
            [1, 6, 5],
            [1, 2, 6],  # Right
        ],
        dtype=np.int32,
    )


def create_cube_faces_with_hole() -> np.ndarray:
    """10 triangles — top face removed to create a hole."""
    return np.array(
        [
            [0, 2, 1],
            [0, 3, 2],  # Bottom
            [0, 1, 5],
            [0, 5, 4],  # Front
            [2, 7, 6],
            [2, 3, 7],  # Back
            [0, 4, 7],
            [0, 7, 3],  # Left
            [1, 6, 5],
            [1, 2, 6],  # Right
        ],
        dtype=np.int32,
    )


class MochiMeshTestBase(unittest.TestCase):
    """Shared test base class for mochi_mesh pybind tests."""

    pass
