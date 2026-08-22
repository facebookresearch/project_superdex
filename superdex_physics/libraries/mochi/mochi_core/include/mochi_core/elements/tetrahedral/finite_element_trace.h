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

#include <mochi_core/elements/tetrahedral/basis_functions_evaluated.h>
#include <mochi_core/elements/tetrahedral/finite_element.h>
#include <mochi_core/elements/tetrahedral/simplex_quadrature.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/simd.h>
#include <mochi_core/utils/vmatrix.h>

#include <cmath>

namespace mochi::tetrahedral {

// @TODO[MAURIZIO] Merge this into Pk3DElement

/**
  A class for the trace of a tetrahedral element
*/
template <class ElementT_, int kNumQuadPoints_>
class Pk3DElementTrace final {
 public:
  static constexpr int kNumQuadPoints = kNumQuadPoints_;
  static constexpr int kNumNodes = 3;
  using ElementT = ElementT_;
  using BasisEvaluatedT =
      tetrahedral::BasisFunctionsEvaluated<ElementT::kPolyOrder, kNumQuadPoints>;

  Pk3DElementTrace(
      ElementT const& baseElement,
      int faceNum,
      TetrahedralQuadrature<kNumQuadPoints> const& quad)
      : baseElement(baseElement),
        faceNum(faceNum),
        quadrature(quad),
        basisEvaluatedParametric(quadrature),
        nodesCrdsPhys(baseElement.nodesCrdsPhys),
        basisEvaluated(basisEvaluatedParametric.basisEvaluated) {
    // Tabulate the isoparametric map and its derivative at the quad points
    QuadratureEvaluateMap();

    // Tabulate basis functions
    QuadratureEvaluateBasis();
  }

  inline ElementT const& GetBaseElement() const {
    return baseElement;
  }

  inline int GetElementIndex() const {
    return baseElement.GetElementIndex();
  }

  /// @brief Get the node indices of the element.
  Int3 Nodes() const {
    return {
        baseElement.connectivity[baseElement.elementIndex][TetFaces::kIndices[faceNum][0]],
        baseElement.connectivity[baseElement.elementIndex][TetFaces::kIndices[faceNum][1]],
        baseElement.connectivity[baseElement.elementIndex][TetFaces::kIndices[faceNum][2]],
    };
  }

  inline Int3 LocalNodes() const {
    return TetFaces::kIndices[faceNum];
  }

  /**
    Evaluate map and dmap for one quadrature point given nodal positions
  */
  void QuadraturePointEvaluateMap(
      int qpoint,
      NdArray<real, 4, 3> const& nodesCrds,
      NdArray<real, 3>& outMap,
      NdArray<real, 3, 3>& outDMap) const {
    // Here we are using an isoparametric map
    outMap = {};
    outDMap = {};

    for (int f = 0; f < kNumDofs; ++f) {
      // Get the map of the quadrature point
      outMap += basisEvaluatedParametric.basisEvaluated[qpoint][f] * nodesCrds[f];

      // Get the tangent map at the quadrature point
      outDMap += Outer(nodesCrds[f], basisEvaluatedParametric.dBasisEvaluated[qpoint][f]);
    }
  }

  // SIMD overload (see scalar version above)
  void QuadraturePointEvaluateMap(
      int qpoint,
      VMatrix4x3r const& nodesCrds,
      Vec4r& outMap,
      VMatrix3x3r& outDMap) const {
    static_assert(kNumDofs == 4, "Unexpected number of DoFs");

    // clang-format off
    // Get the map of the quadrature point
    Vec4r basisEval = Load<Vec4r>(basisEvaluatedParametric.basisEvaluated[qpoint].data());
    outMap = Broadcast<0>(basisEval) * nodesCrds[0] + Broadcast<1>(basisEval) * nodesCrds[1] + Broadcast<2>(basisEval) * nodesCrds[2] + Broadcast<3>(basisEval) * nodesCrds[3];

    // Get the tangent map at the quadrature point
    outDMap = Outer3(nodesCrds[0], Load<Vec4r>(basisEvaluatedParametric.dBasisEvaluated[qpoint][0].data()));
    outDMap += Outer3(nodesCrds[1], Load<Vec4r>(basisEvaluatedParametric.dBasisEvaluated[qpoint][1].data()));
    outDMap += Outer3(nodesCrds[2], Load<Vec4r>(basisEvaluatedParametric.dBasisEvaluated[qpoint][2].data()));
    outDMap += Outer3(nodesCrds[3], Load<3, Vec4r>(basisEvaluatedParametric.dBasisEvaluated[qpoint][3].data()));
    // clang-format on
  }

  /**
    Evaluate weight and normal for one quadrature point given the inverse of the map
  */
  void QuadraturePointEvaluateWeightNormal(
      int qpoint,
      real dMapDet,
      NdArray<real, 3, 3> const& dMapInv,
      real& outWeight,
      NdArray<real, 3>& outNormal) const {
    // Compute quadrature weights using Nanson's formula for metric change
    outWeight = 0_r;
    outNormal = {};
    for (int j = 0; j < kSpaceDim; ++j) {
      for (int i = 0; i < kSpaceDim; ++i) {
        outNormal[j] += dMapDet * dMapInv[i][j] * kReferenceNormals[faceNum][i];
      }
      outWeight +=
          Sqr(outNormal[j]); // the square of the component of the Piola transform of the norm
    }
    outWeight = std::sqrt(outWeight); // the norm of the Piola transform of the norm
    // Normalize the normal
    for (int i = 0; i < kSpaceDim; ++i) {
      outNormal[i] /= outWeight;
    }
    outWeight *= quadrature.weights[qpoint]; // scale by the area of the face
  }

  // SIMD overload (see scalar version above)
  void QuadraturePointEvaluateWeightNormal(
      int qpoint,
      real dMapDet,
      VMatrix3x3r const& dMapInv,
      real& outWeight,
      Vec4r& outNormal) const {
    // Compute quadrature weights using Nanson's formula for metric change
    static_assert(kSpaceDim == 3, "3D coordinates assumed by this implementation");

    Vec4r scaledRefNormal = dMapDet * ToSimd(kReferenceNormals[faceNum]);
    outNormal = dMapInv[0] * Broadcast<0>(scaledRefNormal) +
        dMapInv[1] * Broadcast<1>(scaledRefNormal) + dMapInv[2] * Broadcast<2>(scaledRefNormal);

    // Normalize the normal
    Vec4r weight = VNorm<3>(outNormal);
    outNormal /= weight;

    outWeight = Get0(weight);
    outWeight *= quadrature.weights[qpoint]; // scale by the area of the face
  }

 private:
  /**
    Evaluate basis, derivatives, metric changes, map and dmap at quadrature points and caches them
  */
  void QuadratureEvaluateMap() {
    for (int q = 0; q < kNumQuadPoints; ++q) {
      // Get the map and the tangent map at the quadrature point
      QuadraturePointEvaluateMap(q, nodesCrdsPhys, mapEvaluated[q], dMapEvaluated[q]);

      // Get the jacobian
      dMapEvaluatedDet[q] = Det(dMapEvaluated[q]);

      // Store the inverse
      dMapEvaluatedInv[q] = Invert(dMapEvaluated[q], dMapEvaluatedDet[q]);

      // Get the quadrature weight and the normal at the quadrature point
      QuadraturePointEvaluateWeightNormal(
          q, dMapEvaluatedDet[q], dMapEvaluatedInv[q], quadWeights[q], normals[q]);
    }
  }

  /**
    Evaluate basis derivatives at quadrature points
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

 public:
  ElementT const& baseElement;
  int const faceNum;
  TetrahedralQuadrature<kNumQuadPoints> const& quadrature;
  BasisEvaluatedT const basisEvaluatedParametric;

  // Static constants
  static constexpr int kSpaceDim = 3;
  static constexpr int kSpaceDimParam = 3;
  static constexpr int kNumDofs = BasisEvaluatedT::kBasis.kNumDofs;
  static_assert(kNumDofs == 4, "Unexpected number of DoFs");

  // The normals to the parametric element faces (for each of the 4 face we have a 3 dimensional
  // normal)
  static constexpr real kSqrt3 = 1.732050807568877_r;
  static constexpr NdArray<real, 4, 3> kReferenceNormals = {
      Real3{0_r, 0_r, -1_r},
      Real3{0_r, -1_r, 0_r},
      Real3{-1_r, 0_r, 0_r},
      Real3{1_r / kSqrt3, 1_r / kSqrt3, 1_r / kSqrt3}};

  // The reference area of each of the faces
  static constexpr NdArray<real, 4> kReferenceFaceAreas = {
      1_r / 2_r,
      1_r / 2_r,
      1_r / 2_r,
      kSqrt3 / 2_r};

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

  // Basis evaluate at quadrature points
  NdArray<real, kNumQuadPoints, kNumDofs> basisEvaluated;

  // dbasis evaluated at quad pts
  NdArray<real, kNumQuadPoints, kNumDofs, kSpaceDim> dBasisEvaluated;

  // The quadrature weights for surface integration
  NdArray<real, kNumQuadPoints> quadWeights = {};

  // The face normal
  NdArray<real, kNumQuadPoints, kSpaceDim> normals = {};
};

} // namespace mochi::tetrahedral
