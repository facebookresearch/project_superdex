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

#include <mochi_core/linear_algebra/krylov/tools/tensor_functions.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/utils/matrix_concepts.h>
#include <mochi_core/utils/debug.h>

#include <algorithm>

namespace mochi::krylov {

template <typename Scalar>
class RecyclingBin {
 public:
  explicit RecyclingBin(int retainedVectors) : retainedDirections(retainedVectors), P(), AP() {
    MOCHI_ASSERT_VERBOSE(
        retainedDirections > 0,
        "Need to request a positive number of vectors (%d <= 0)",
        retainedDirections);
  }

  ~RecyclingBin() = default;

  /// @brief Insert a new direction (and its image Ap = A*p) into the container
  template <typename VectorP, typename VectorAP>
  void Insert(VectorP const& p, VectorAP const& Ap);

  /// @brief Returns the retained directions
  auto GetRetainedDirections() {
    return P.LeftCols(std::min(retainedDirections, numStoredVectors));
  }

  auto GetRetainedMappedDirections() {
    return AP.LeftCols(std::min(retainedDirections, numStoredVectors));
  }

 protected:
  int numStoredVectors = 0;
  int retainedDirections = 0;

  Matrix<Scalar> P, AP;
};
} // namespace mochi::krylov

//
// Implementation of functions
//

namespace mochi::krylov {

template <typename Scalar>
template <typename VectorP, typename VectorAP>
void RecyclingBin<Scalar>::Insert(VectorP const& p, VectorAP const& Ap) {
  if (numStoredVectors == 0) {
    P.Resize(NumRows(p), retainedDirections);
    AP.Resize(NumRows(Ap), retainedDirections);
  }
  // Exit if we have all the directions
  if (numStoredVectors >= retainedDirections) {
    return;
  }
  //
  if constexpr (mochi::IsCuda<VectorP>) {
    CudaVectorView<Scalar const> pview(p.data(), NumRows(p));
    CudaVectorView<Scalar const> Apview(Ap.data(), NumRows(Ap));
    auto const pTAp = pview.Dot(Apview);
    if (pTAp <= Scalar(0)) {
      return; // Skip — direction is not A-conjugate (non-SPD or degenerate)
    }
    auto const scaling = Scalar(1) / Sqrt(pTAp);
    P.Col(numStoredVectors) = pview; // Convert from Cuda to CPU
    AP.Col(numStoredVectors) = Apview; // Convert from Cuda to CPU
    P.Col(numStoredVectors) *= scaling;
    AP.Col(numStoredVectors) *= scaling;
  } else {
    ColumnVectorView<Scalar const> pview(p.data(), NumRows(p));
    ColumnVectorView<Scalar const> Apview(Ap.data(), NumRows(Ap));
    auto const pTAp = pview.Dot(Apview);
    if (pTAp <= Scalar(0)) {
      return; // Skip — direction is not A-conjugate (non-SPD or degenerate)
    }
    auto const scaling = Scalar(1) / Sqrt(pTAp);
    P.Col(numStoredVectors) = scaling * pview;
    AP.Col(numStoredVectors) = scaling * Apview;
  }
  //
  numStoredVectors += 1;
}

} // namespace mochi::krylov
