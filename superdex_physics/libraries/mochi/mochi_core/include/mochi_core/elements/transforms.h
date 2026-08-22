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

#include <mochi_core/linear_algebra/matrix.h>

#include <mochi_core/linear_algebra/strided_matrix.h>

namespace mochi {
using namespace krylov;

template <typename Scalar, int nNodes, int kLeadDim, int kStride>
MOCHI_ANY StridedMatrix<Scalar, 3, nNodes> GetCoordinates(
    Span<int const> nodes,
    StridedMatrixView<Scalar, 3, kDynamic, kStride, Direction::ColMajor, kLeadDim> const&
        allNodeCoord) {
  StridedMatrix<Scalar, 3, nNodes> elemCoords;
  for (int i = 0; i < nNodes; ++i) {
    elemCoords.Col(i) = allNodeCoord.Col(nodes[i]);
  }
  return elemCoords;
}

template <typename Scalar>
struct DeformationMap {
  StridedMatrix<Scalar, 3, 3> F;
  StridedMatrix<Scalar, 3, 3> dxi_dX;
  Scalar J;
};

template <typename Scalar, int nNodes>
MOCHI_ANY DeformationMap<Scalar> ComputeDefGradient(
    StridedMatrix<Scalar, 3, nNodes>& disp,
    StridedMatrix<Scalar, 3, nNodes>& refCoords,
    StridedMatrixView<Scalar const, nNodes, 3> const& dBari_dxi) {
  StridedMatrix<Scalar, 3, 3> dx_dxi = disp * dBari_dxi;
  StridedMatrix<Scalar, 3, 3> dX_dxi = refCoords * dBari_dxi;
  auto [dxi_dX, J] = Invert<true>(dX_dxi);
  Scalar zero{0}, one{1};
  StridedMatrix<Scalar, 3, 3> I{{one, zero, zero}, {zero, one, zero}, {zero, zero, one}};

  return {StridedMatrix<Scalar, 3, 3>{I + dx_dxi * dxi_dX}, dxi_dX, J};
}

} // namespace mochi
