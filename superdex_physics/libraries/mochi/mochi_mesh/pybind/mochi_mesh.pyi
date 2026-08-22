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

from enum import IntEnum
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt

class RemeshMethod(IntEnum):
    NONE: int
    ALPHA_WRAP: int
    ACVD: int
    SURFACE_DELAUNAY: int

class SurfaceRemeshingParams:
    method: RemeshMethod
    edge_size: float
    detect_features: bool
    relative_to_mesh_size: bool
    alpha_wrap_relative_alpha: float
    alpha_wrap_relative_offset: float
    smoothing_iterations: int
    relaxation_steps_per_iteration: int
    tangential_relaxation_iterations: int
    angle_smoothing_iterations: int
    sharp_feature_angle: float
    protect_constraints: bool
    relax_constraints: bool
    use_adaptive_sizing: bool
    adaptive_sizing_tolerance: float
    min_edge_size_factor: float
    max_edge_size_factor: float
    target_vertex_count: int
    acvd_gradation_factor: float
    facet_angle_bound: float
    facet_distance_bound: float
    repair_mesh: bool
    def __init__(self) -> None: ...

class DistributionStatistics:
    mean: float
    standard_deviation: float
    min: float
    max: float

class MeshStatistics:
    num_vertices: int
    num_faces: int
    edge_lengths: DistributionStatistics
    angles: DistributionStatistics
    hausdorff_distance: float
    is_closed: bool

class Error(RuntimeError): ...

def remesh_surface(
    vertices: npt.NDArray[np.floating],
    faces: npt.NDArray[np.integer],
    params: SurfaceRemeshingParams = ...,
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.integer]]: ...
def compute_mesh_statistics(
    vertices: npt.NDArray[np.floating],
    faces: npt.NDArray[np.integer],
    ref_vertices: Optional[npt.NDArray[np.floating]] = ...,
    ref_faces: Optional[npt.NDArray[np.integer]] = ...,
) -> MeshStatistics: ...
def reconstruct_surface_from_sdf(
    dims: npt.NDArray[np.integer],
    values: npt.NDArray[np.floating],
    bounds_min: npt.NDArray[np.floating],
    bounds_max: npt.NDArray[np.floating],
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.integer]]: ...
