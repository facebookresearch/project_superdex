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
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/vmatrix.h>

namespace mochi {

// A class for evaluating fields and field gradients at arbitrary sample points
// in the reference configuration.
template <typename ElementT>
class EvalPoint {
 public:
  static constexpr int kNumEleNodes = ElementT::kNumDofs;
  static constexpr int kSpaceDim = ElementT::kSpaceDim;
  static_assert(kSpaceDim == 3, "Expected 3D coordinates");

  using Basis = typename ElementT::Basis;

  EvalPoint() = default;

  EvalPoint(ElementT const& element, Real3 const& physPosition)
      : _elementIndex(element.GetElementIndex()), _mapEvaluated(physPosition) {
    EvaluateMapIsoparametric(element);
    EvaluateBasisIsoparametric(element);
  }

  // Gets the position of this evaluation point in the reference configuration
  Real3 GetPositionReference() const {
    return _mapEvaluated;
  }

  // Gets the element index
  int GetElementIndex() const {
    return _elementIndex;
  }

  // A default constructed EvalPoint is not valid
  bool IsValid() const {
    return _elementIndex >= 0;
  }

  // Evaluates a field at this evaluation point
  real Eval(NdArray<real, kNumEleNodes> const& dofs) const {
    return Dot(dofs, _basisEvaluated);
  }

  // Evaluates multiple fields at this evaluation point
  template <size_t D0>
  NdArray<real, D0> Eval(NdArray<real, D0, kNumEleNodes> const& dofs) const {
    return DotMatVec(dofs, _basisEvaluated);
  }

  // Evaluates the gradient of a field at this evaluation point
  Real3 DEval(NdArray<real, kNumEleNodes> const& dofs) const {
    return DotVecMat(dofs, _dbasisEvaluated);
  }

  // Evaluates the gradients of multiple fields at this evaluation point
  template <size_t D0>
  NdArray<real, D0, kSpaceDim> DEval(NdArray<real, D0, kNumEleNodes> const& dofs) const {
    return Dot(dofs, _dbasisEvaluated);
  }

 private:
  int _elementIndex = -1;
  NdArray<real, kSpaceDim> _invMapEvaluated;
  NdArray<real, kSpaceDim> _mapEvaluated;
  NdArray<real, kSpaceDim, kSpaceDim> _dmapEvaluated;
  NdArray<real, kNumEleNodes> _basisEvaluated;
  NdArray<real, kNumEleNodes, kSpaceDim> _dbasisEvaluated;
  NdArray<real, kSpaceDim, kSpaceDim> _dmapEvaluatedInv;
  real _dmapEvaluatedDet = 0_r;

  // Evaluate basis and derivatives at requested point
  void EvaluateBasisIsoparametric(ElementT const& element);

  // Evaluate basis and derivatives at requested point and caches them
  void EvaluateMapIsoparametric(ElementT const& element);
};

/*****************************************************************************************
  Inlines
*/

// Evaluate basis and derivatives at requested point
template <typename ElementT>
inline void EvalPoint<ElementT>::EvaluateBasisIsoparametric(ElementT const& /*element*/) {
  // Get the number of quadrature points
  for (int f = 0; f < kNumEleNodes; ++f) {
    // Compute the value of the basis in physical coordinates
    _basisEvaluated[f] = Basis::GetValue(f, _invMapEvaluated);

    // Compute gradient with respect to physical coordinates
    _dbasisEvaluated[f] =
        DotMatVec(Transpose(_dmapEvaluatedInv), Basis::GetDValue(f, _invMapEvaluated));
  }
}

// Evaluate parametric map, dmap, dmap inv and jacobian at requested point and cache them
template <typename ElementT>
inline void EvalPoint<ElementT>::EvaluateMapIsoparametric(ElementT const& element) {
  _invMapEvaluated = element.GetInvMap(_mapEvaluated);

  _dmapEvaluated = {};

  // Here we are using an isoparametric map
  for (int f = 0; f < kNumEleNodes; ++f) {
    // Get the tangent map at the quadrature point
    _dmapEvaluated += Outer(element.nodesCrdsPhys[f], Basis::GetDValue(f, _invMapEvaluated));
  }

  // Get the jacobian
  _dmapEvaluatedDet = Det(_dmapEvaluated);

  // Store the inverse
  _dmapEvaluatedInv = Invert(_dmapEvaluated, _dmapEvaluatedDet);
}
} // namespace mochi
