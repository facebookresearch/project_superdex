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

"""
A collection of functions for transforming between different representations of
rotations and transformations.
"""

import numpy as np
import numpy.typing as npt
import superdex.physics as mochi
from scipy.spatial import transform as tr

########################################################################################
# Transformation utilities
########################################################################################


def rotvec_to_quat(rotvec: npt.ArrayLike) -> npt.NDArray[float]:
    """
    Converts a rotation vector to a quaternion.
    """
    return tr.Rotation.from_rotvec(rotvec).as_quat()


def rotvec_to_mat(rotvec: npt.ArrayLike) -> npt.NDArray[float]:
    """
    Converts a rotation vector to a rotation matrix.
    """
    return tr.Rotation.from_rotvec(rotvec).as_matrix()


def make_transform(
    position: npt.ArrayLike,
    rotvec: npt.ArrayLike,
    scale: float = 1.0,
) -> npt.NDArray[float]:
    """
    Creates a homogeneous transformation matrix from the given position, rotation vector,
    and uniform scale factor.
    """

    mat = np.eye(4, dtype=np.float32)
    mat[:3, 3] = position
    mat[:3, :3] = scale * rotvec_to_mat(rotvec)
    return mat


def apply_linear_map(
    linear_map: npt.ArrayLike, points: npt.ArrayLike
) -> npt.NDArray[float]:
    """
    Applies a linear transformation to a set of points.

    Performs matrix multiplication to transform points using a linear map,
    which can represent rotations, scaling, shearing, or any linear transformation.
    Supports both single points and batches of points.

    Args:
        linear_map: A (N, N) transformation matrix representing the linear map.
            For 3D transformations, this is typically a (3, 3) matrix.
        points: Points to transform. Can be:
            - A single point with shape (N,)
            - Batch of points with shape (..., N)

    Returns:
        Transformed points with the same shape as the input points.
    """
    linear_map = np.asarray(linear_map)
    points = np.asarray(points)
    return (linear_map @ points[..., None]).squeeze(-1)


def apply_affine_transform(
    transform: npt.ArrayLike, points: npt.ArrayLike
) -> npt.NDArray[float]:
    """
    Applies an affine transformation to a set of points.

    An affine transformation consists of a linear transformation (rotation, scaling,
    shearing) followed by a translation. This function extracts the linear and
    translation components from a homogeneous transformation matrix and applies
    them to the input points. Supports both single points and batches of points.

    Args:
        transform: A homogeneous transformation matrix with shape (N+1, N+1).
            For 3D transformations, this is a (4, 4) matrix where:
            - transform[:3, :3] is the linear part (rotation/scale/shear)
            - transform[:3, 3] is the translation vector
        points: Points to transform. Can be:
            - A single point with shape (N,)
            - Batch of points with shape (..., N)

    Returns:
        Transformed points with the same shape as the input points.
    """
    transform = np.asarray(transform)
    return apply_linear_map(transform[:-1, :-1], points) + transform[:-1, -1]


def is_pure_rotation(transform: npt.NDArray) -> bool:
    """
    Checks if a linear part of a transformation matrix is describes a pure rotation.
    """
    rtol = 1e-5
    atol = 1e-6
    linear = transform[:3, :3]
    return np.isclose(np.linalg.det(linear), 1.0, rtol=rtol, atol=atol) and np.allclose(
        linear.T @ linear, np.eye(3), rtol=rtol, atol=atol
    )


def orthonormalize_transform(transform: npt.NDArray) -> npt.NDArray[float]:
    """
    Ensures that the linear part of a transformation matrix is orthonormalized, thus
    representing a pure rotation. This is done by applying a Gram-Schmidt
    orthogonalization procedure to the columns of the linear part of the matrix.
    """

    # Extract the X and Y axes from the linear part of the transformation matrix.
    xaxis, yaxis = transform[:3, :2].T

    # Perform the Gram-Schmidt orthogonalization procedure.
    # 1. Normalize the X axis.
    # 2. Orthogonalize the Y axis against the X axis.
    # 3. Find the Z axis by taking the cross product of the X and Y axes.
    xaxis = xaxis / np.linalg.norm(xaxis)
    yaxis = yaxis - np.dot(xaxis, yaxis) * xaxis
    yaxis = yaxis / np.linalg.norm(yaxis)
    zaxis = np.cross(xaxis, yaxis)

    # Return transformation matrix with the computed orthonormalized linear part.
    transform = np.copy(transform)
    transform[:3, 0] = xaxis
    transform[:3, 1] = yaxis
    transform[:3, 2] = zaxis
    return transform


def transformrt_to_numpy(transform: mochi.TransformRT) -> npt.NDArray[float]:
    """Convert a TransformRT to numpy array."""
    return np.stack(
        (
            np.asarray(transform.translation),
            np.asarray(transform.rotation.to_rotation_vector()),
        )
    )
