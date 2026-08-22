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

#include <mochi_core/linear_algebra/any_matrix.h>
#include <mochi_core/solvers/interaction_matrix_info.h>

#include <vector>

namespace mochi {

/// @brief Composite matrix structure to represent an actor matrix with interaction matrices
/// @tparam Scalar type
template <typename Scalar>
struct ActorPseudoMatrix {
  /// @brief Actor offset.
  int offset;

  /// @brief Actor matrix free of interaction
  AnyMatrixView<Scalar const> const& actorMatrix;

  /// @brief Vector of interaction matrices.
  /// @note It is possible that some interaction matrices do not interact with the actor matrix.
  std::vector<AnyInteractionMatrixViewInfo<Scalar const>> const& interactionMatrices;

  /// @brief Returns the number of rows in the matrix
  /// @return Integral value
  auto Rows() const {
    return GetNumRows(actorMatrix);
  }

  /// @brief Returns the number of columns in the matrix
  /// @return Integral value
  auto Cols() const {
    return GetNumCols(actorMatrix);
  }
};

} // namespace mochi
