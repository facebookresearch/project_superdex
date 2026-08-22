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

#include <mochi_core/elements/finite_element_utils.h>
#include <mochi_core/elements/tetrahedral/basis_functions_evaluated.h>
#include <mochi_core/elements/tetrahedral/simplex_quadrature.h>
#include <mochi_core/geometry/geometry_utils.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/span.h>

namespace mochi::tetrahedral {

/**
The lagrange Tetrahedral element.
  In the mindset of keeping memory demand as low as possible
  and high efficiency, the element assumes the same interpolant
  for each of the fields; a reasonable assumption in the case of
  elasticity. This code is not intended to be as general as
  possible rather striking a balance between somewhat
  efficient and versatile.
*/
template <int kPolyOrder_, int kNumQuadPoints_ = 1>
class Pk3DElement final {
  static_assert(kPolyOrder_ == 1, "Higher order polynomials not yet implemented");

 public:
  using Basis = BarycentricBasisTetrahedra<kPolyOrder_>;
  using BasisEvaluated = BasisFunctionsEvaluated<kPolyOrder_, kNumQuadPoints_>;

  TetrahedralQuadrature<kNumQuadPoints_> const& quadrature;
  BasisEvaluated const basisEvaluatedParametric;

  static constexpr int kPolyOrder = kPolyOrder_;
  static constexpr int kNumQuadPoints = kNumQuadPoints_;
  static constexpr int kSpaceDimParam = 3;
  static constexpr int kSpaceDim = 3;
  static constexpr int kNumDofs = Basis::kNumDofs;
  static constexpr Basis kBasis = {};

  /**
    The constructor for a lagrange tet element

    Args:
      elementIndex (int): the index of the element
      coordinates (ndarray): reference to the whole set of coordinates of the mesh nodes (passed by
        reference)

      connectivity (ndarray): reference to the connectivity of the whole mesh (passed by reference)
  */
  Pk3DElement(
      int elementIndex,
      Span<Real3 const> coordinates,
      Span<Int4 const> connectivity,
      TetrahedralQuadrature<kNumQuadPoints> const& quad = kTetrahedralQuadrature1,
      bool withInitialization = true)
      : quadrature(quad),
        basisEvaluatedParametric(quadrature),
        elementIndex(elementIndex),
        coordinates(coordinates),
        connectivity(connectivity),
        basisEvaluated(basisEvaluatedParametric.basisEvaluated) {
    if (withInitialization) {
      // Construct the array containing the physical coordinates corresponding to the coordinates of
      // each degree of freedom
      InterpolateInteriorNodes();

      Initialize();
    }
  }

 private:
  void Initialize() {
    // Tabulate the isoparametric map and its derivative at the quad points
    QuadratureEvaluateMap();

    // Tabulate basis functions
    QuadratureEvaluateBasis();
  }

  /**
    Evaluate basis and derivatives at quadrature points and caches them
  */
  void QuadratureEvaluateMap() {
    for (int q = 0; q < kNumQuadPoints; ++q) {
      // Here we are using an isoparametric map
      mapEvaluated[q] = {};
      dMapEvaluated[q] = {};

      EvaluateField<kSpaceDim, kNumDofs>(
          basisEvaluatedParametric.basisEvaluated[q], nodesCrdsPhys, &mapEvaluated[q]);

      // Note this is slight abuse of this function as technically should be called with the
      // derivative of the basis wrt to reference coordinates not parametric coordinates.
      // We need the induced basis from the isoparametric mapping (dMapEvaluated) to compute the
      // element basis derivative wrt reference coordinates ( dBasisEvaluated).
      EvaluateFieldGradient<kSpaceDim, kNumDofs, kSpaceDimParam>(
          basisEvaluatedParametric.dBasisEvaluated[q], nodesCrdsPhys, &dMapEvaluated[q]);

      // Get the jacobian
      dMapEvaluatedDet[q] = Det(dMapEvaluated[q]);

      // Store the inverse
      dMapEvaluatedInv[q] = Invert(dMapEvaluated[q], dMapEvaluatedDet[q]);

      // Compute quadrature weight
      quadWeights[q] = quadrature.weights[q] * dMapEvaluatedDet[q];
    }
  }

  /**
    Evaluate basis and derivatives at quadrature points
  */
  void QuadratureEvaluateBasis() {
    for (int q = 0; q < kNumQuadPoints; ++q) {
      for (int f = 0; f < kNumDofs; ++f) {
        // Compute gradient with respect to physical coordinates
        dBasisEvaluated[q][f] = DotMatVec(
            Transpose(dMapEvaluatedInv[q]), basisEvaluatedParametric.dBasisEvaluated[q][f]);
      }
    }
  }

  /**
    Computes the nodal position in physical space of interior nodes.

    Only relevant for higher order elements this method interpolates linearly the mid and interior
    nodes on the physical domain. If we want to specify the value we can set the nodes using
    SetInteriorNodesCoordinates.
  */
  void InterpolateInteriorNodes() {
    constexpr int kNumVertex = 4;

    // Here we are simply interpolating linearly the physical coordinates of the interior nodes this
    // could be more general
    for (int i = 0; i < kNumVertex; ++i) {
      nodesCrdsPhys[i] = coordinates[connectivity[elementIndex][i]];
    }
    if constexpr (kPolyOrder > 1) {
      auto nodeCrdsParams = PrecomputeNodesCrdsParam();
      for (int i = kNumVertex; i < kNumDofs; ++i) {
        nodesCrdsPhys[i] = nodeCrdsParams[0] * nodeCrdsParams[0] +
            nodeCrdsParams[1] * nodeCrdsParams[1] + nodeCrdsParams[2] * nodeCrdsParams[2] +
            (1_r - nodeCrdsParams[0] - nodeCrdsParams[1] - nodeCrdsParams[2]) * nodeCrdsParams[3];
      }
    }
  }

  // Compile-time algorithm to compute kNodesCrdsParam
  static constexpr auto PrecomputeNodesCrdsParam() {
    // Get the coordinates of the nodes in the parametric domain this are the nodes of the exterior
    // vertices, edge nodes, as well as interior nodes. Here we remove the last column that
    // correspond to the additional barycentric coordinate
    NdArray<real, kNumDofs, kSpaceDimParam> nodesCrdsParam = {};
    for (size_t i = 0; i < kBasis.kDofNodes.size(); ++i) { // for each row
      auto const& row = kBasis.kDofNodes[i];
      for (size_t j = 0; j < row.size() - 1; ++j) { // for each column except the last one
        nodesCrdsParam[i][j] = kBasis.kDofNodes[i][j];
      }
    }
    return nodesCrdsParam;
  }

 public:
  // Coordinates of the nodes in the parametric domain
  static constexpr NdArray<real, kNumDofs, 3> kNodesCrdsParam = PrecomputeNodesCrdsParam();

  // Set by constructor
  int elementIndex;
  Span<Real3 const> coordinates;
  Span<Int4 const> connectivity;

  // The nodal position in physical space of interior nodes.
  NdArray<real, kNumDofs, kSpaceDim> nodesCrdsPhys;

  // The array of mapped quadrature points into physical space
  NdArray<real, kNumQuadPoints, kSpaceDim> mapEvaluated;

  // The array of the tangent map from parametric to physical evaluated at the quadrature points
  NdArray<real, kNumQuadPoints, kSpaceDim, kSpaceDimParam> dMapEvaluated;

  // The array of the inverse of the tangent map from parametric to physical evaluated at the
  // quadrature points
  NdArray<real, kNumQuadPoints, kSpaceDimParam, kSpaceDim> dMapEvaluatedInv;

  // The array of the determinant of the tangent map from parametric to physical evaluated at the
  // quadrature points
  NdArray<real, kNumQuadPoints> dMapEvaluatedDet;

  // The basis functions evaluated at quad points
  NdArray<real, kNumQuadPoints, kNumDofs> basisEvaluated;

  // dbasis evaluated at quad pts
  NdArray<real, kNumQuadPoints, kNumDofs, kSpaceDim> dBasisEvaluated;

  // The quadrature weights for volume integration
  NdArray<real, kNumQuadPoints> quadWeights = {};

  inline int GetElementIndex() const {
    return elementIndex;
  }

  /// @brief Get the node indices of the element.
  Int4 const& Nodes() const {
    return connectivity[elementIndex];
  }

  // Evaluate the inverse isoparametric map for a point x in physical space.
  inline NdArray<real, 3> GetInvMap(NdArray<real, 3> const& x) const {
    Int4 corners = connectivity[elementIndex];
    auto coords = BarycentricCoords4(
        coordinates[corners[0]],
        coordinates[corners[1]],
        coordinates[corners[2]],
        coordinates[corners[3]],
        x);
    return {coords[1], coords[2], coords[3]}; // Drop the first coordinate
  }

  // it overloads the assumption that the edge nodes lie along a line allowing for curved boundaries
  void SetInteriorNodesCoordinates(
      NdArray<real, Pk3DElement<kPolyOrder>::kNumDofs, Pk3DElement<kPolyOrder>::kSpaceDim> const&
          nodes_crds_phys) {
    static_assert(
        (nodes_crds_phys.dims[0] == kNumDofs) && (nodes_crds_phys.dims[1] == kSpaceDim),
        "There must be a coordinate of space dim for each node, including vertex nodes");
    nodesCrdsPhys = nodes_crds_phys;

    Initialize();
  }
};

} // namespace mochi::tetrahedral
