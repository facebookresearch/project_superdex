/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/lagrange_polynomials.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/span.h>

#include <utility>
#include <vector>

namespace mochi::triangular {

/**
  The base functions defined over the unit three-dimensional.

  Cf.Automated Solution of Differential Equations by
  the Finite Element Method Ch.3

  Dof numbering for linear

  Param Crds. | Dof num
  (0, 0) | 0
  (1, 0) | 1
  (0, 1) | 2
*/
template <int kPolyOrder_ = 1>
class BarycentricBasisTriangular final {
  // WARNING: In order to implement higher order polynomials we need to figure
  // out a better node numbering scheme. Most of the infrastructure is in place
  static_assert(kPolyOrder_ == 1, "Higher order polynomials not yet implemented");

 public:
  /**
    The value of the Lagrange-type interpolation.

    Returns the value of the Lagrange-type interpolation cf.
    Hughes App 3.1

    Args:
      index (int): the index of base function
      y (float): the scalar value in the rage of [0,1] of the barycentric coordinate

  Returns:
      float: the value of the function
  */
  static constexpr real GetTValue(int index, real y) {
    if (index == 0) {
      return 1_r;
    }

    // The nodes of the one dimensional polynomial
    // over the parametric interval [0,1]
    Real4 paramOneDim = {};
    MOCHI_ASSERT_VERBOSE(index < paramOneDim.size(), "buffer too small");
    for (int i = 0; i < index + 1; ++i) {
      paramOneDim[i] = static_cast<real>(i) / static_cast<real>(index);
    }

    return GetLagrangeBasis(
        Span<real const>{paramOneDim.data(), size_t(index) + 1},
        index,
        index,
        y / kOneDimKnot[index]);
  }

  /**
    The derivative of the Lagrange-type interpolation.

    The derivative of the Lagrange-type interpolation
    cf. Hughes App 3.1

    Args:
      index (int): the index of base function
      y (float): the scalar value in the rage of
          [0,1] of the barycentric coordinate

    Returns:
      float: the value of the derivative of the interpolant
  */
  static constexpr real GetDtValue(int index, real y) {
    if (index == 0) {
      return 0_r;
    }

    // The nodes of the one dimensional polynomial
    // over the parametric interval [0,1]
    MOCHI_ASSERT_VERBOSE(index < 4, "buffer too small");
    real paramOneDim[4] = {};
    for (int i = 0; i <= index; ++i) {
      paramOneDim[i] = static_cast<real>(i) / static_cast<real>(index);
    }

    return GetDLagrangeBasis(
               {paramOneDim, size_t(index + 1)}, index, index, (y / kOneDimKnot[index])) *
        (1_r / kOneDimKnot[index]);
  }

  /**
    Gets the value of the basis function.

    Get Gets the value of the basis function of index baseIndex evaluated
    a point in parametric space with parametric coordinates x

    Args:
      baseIndex (int): the index referring to the base to be evaluated
      x (array): the array of parametric coordinates values

    Returns:
      float: the value of the basis function
  */
  static constexpr real GetValue(int baseIndex, Real2 const& x) {
    //  Get the barycentric coordinates
    real const sumX = x[0] + x[1];
    Real3 const lam = {x[0], x[1], 1_r - sumX};
    return //
        GetTValue(kIndices[baseIndex][0], lam[0]) * //
        GetTValue(kIndices[baseIndex][1], lam[1]) * //
        GetTValue(kIndices[baseIndex][2], lam[2]);
  }

  /**
    Gets the gradient of the basis function.

    Gets the gradient of the basis function parametric coordinates
    (not barycentric) of base baseIndex evaluated at a point in
    parametric space with parametric coordinates x.

    Args:
      baseIndex (int): the index referring to the base to be evaluated
      x (ndarray): the 1D array of parametric coordinates (floats) values

    Returns:
      ndarray: 1D array of len 2 of floats containing the derivative in each
        of the directions of parametric space
  */
  static constexpr Real2 GetDValue(int baseIndex, Real2 const& x) {
    real const sumX = x[0] + x[1];
    Real3 const lam = {x[0], x[1], 1_r - sumX};
    real val_i = GetTValue(kIndices[baseIndex][0], lam[0]);
    real val_j = GetTValue(kIndices[baseIndex][1], lam[1]);
    real val_k = GetTValue(kIndices[baseIndex][2], lam[2]);
    return Real2{
        GetDtValue(kIndices[baseIndex][0], lam[0]) * val_j * val_k +
            val_i * val_j * GetDtValue(kIndices[baseIndex][2], lam[2]) * (-1_r),
        val_i * GetDtValue(kIndices[baseIndex][1], lam[1]) * val_k +
            val_i * val_j * GetDtValue(kIndices[baseIndex][2], lam[2]) * (-1_r)};
  }

  static constexpr int kVertexDof = 3; // The number of vertex dof of a triangle
  static constexpr int kPolyOrder = kPolyOrder_; // The polynomial order
  static constexpr int kNumDofs = // The number of degrees of freedom
      (kPolyOrder + 1) * (kPolyOrder + 2) / 2;
  static constexpr int kEdgeDof = // The number of edge dof
      kPolyOrder - 1;
  static constexpr int kFaceDof = // The face numer dof
      (kPolyOrder - 1) * (kPolyOrder - 2) / 2;
  static constexpr int kInteriorDof = 0; // The number of interior dof

  // Shorthand
  using IndicesType = NdArray<int, kNumDofs, 3>;
  using DofNodesType = NdArray<real, kNumDofs, 3>;
  using OneDimKnotType = NdArray<real, kPolyOrder + 1>;

  // Compile-time algorithm to compute kIndices and kDofNodes
  static constexpr std::pair<IndicesType, DofNodesType> PrecomputeIndicesAndNodes() {
    IndicesType indices = {};
    DofNodesType dofNodes = {};
    int count = 0;
    real const po = static_cast<real>(kPolyOrder);
    for (int i = 0; i < (kPolyOrder + 1); ++i) {
      for (int j = 0; j < (kPolyOrder + 1 - i); ++j) {
        Int3 const tri = {j, i, 1 - i - j};
        indices[count] = tri;
        dofNodes[count] = Real3{tri[0] / po, tri[1] / po, tri[2] / po};
        ++count;
      }
    }
    return std::make_pair(indices, dofNodes);
  }

  // Compile-time algorithm to compute kOneDimKnot
  static constexpr OneDimKnotType PrecomputeOneDimKnot() {
    OneDimKnotType oneDimKnot = {};
    for (int i = 0; i <= kPolyOrder; ++i) {
      oneDimKnot[i] = static_cast<real>(i) / static_cast<real>(kPolyOrder);
    }
    return oneDimKnot;
  }

  static constexpr IndicesType kIndices = PrecomputeIndicesAndNodes().first;
  static constexpr DofNodesType kDofNodes = PrecomputeIndicesAndNodes().second;
  static constexpr OneDimKnotType kOneDimKnot = PrecomputeOneDimKnot();
};

} // namespace mochi::triangular
