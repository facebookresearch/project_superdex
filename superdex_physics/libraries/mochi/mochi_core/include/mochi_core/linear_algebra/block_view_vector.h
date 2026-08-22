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

namespace mochi {

/// @brief Block view of a column vector
/// @remarks
/// The column vector has exactly 1 column.
/// The number of rows for that vector is a multiple of kBlockSize.
template <typename Scalar, int kBlockSize, typename CRIdx = int>
struct BlockViewVector {
  BlockViewVector(Scalar* v, CRIdx numBlocks)
      : _vec{v, numBlocks * kBlockSize, 1}, _numBlocks(numBlocks) {}

  [[nodiscard]] CRIdx BlockRows() const {
    return _numBlocks;
  }

  [[nodiscard]] CRIdx Cols() const {
    return 1;
  }

  auto operator[](CRIdx i) {
    return _vec.template Block<kBlockSize, 1>(i * kBlockSize, 0, kBlockSize, 1);
  }

  auto operator[](CRIdx i) const {
    return _vec.template Block<kBlockSize, 1>(i * kBlockSize, 0, kBlockSize, 1);
  }

  void SetZero() {
    _vec.SetZero();
  }

  Scalar* Data() {
    return _vec.Data();
  }

  Scalar const* Data() const {
    return _vec.Data();
  }

 private:
  ColumnVectorView<Scalar, krylov::kDynamic> _vec;
  CRIdx _numBlocks;
};

} // namespace mochi
