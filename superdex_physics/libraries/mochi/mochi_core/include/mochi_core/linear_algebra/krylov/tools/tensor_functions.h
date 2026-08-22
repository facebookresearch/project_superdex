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

#include <mochi_core/linear_algebra/krylov/tools/custom_matrix_traits.h>
#include <mochi_core/linear_algebra/utils/matrix_concepts.h>
#include <mochi_core/mochi_platform.h>
#include <mochi_core/utils/array_utils.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/task_scheduler.h>
#include <mochi_core/utils/time.h>

#include <array>
#include <atomic>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace mochi::krylov {

template <typename MatrixType, typename VectorIn, typename VectorOut>
void Apply(MatrixType const& A, VectorIn const& v, VectorOut&& Av) {
  if constexpr (IsMatrix<MatrixType>) {
    Av = A * v;
  } else if constexpr (IsLinearOperator<MatrixType>) {
    A.Apply(v, Av);
  } else {
    A(v, Av);
  }
}

// TODO: Introduce (Block)SparseMatrix::MiddleRows and deprecate this function.
template <typename MatrixType, typename VectorIn, typename VectorOut, typename Idx>
void ApplyToRange(
    MatrixType const& A,
    VectorIn const& v,
    VectorOut&& Av,
    Idx rowBegin,
    Idx rowEnd) {
  if constexpr (IsMatrix<MatrixType>) {
    Av.MiddleRows(rowBegin, rowEnd - rowBegin) = A.MiddleRows(rowBegin, rowEnd - rowBegin) * v;
  } else {
    static_assert(IsLinearOperator<MatrixType>, "Unsupported matrix type");
    A.ApplyToRange(v, Av, rowBegin, rowEnd);
  }
}

// TODO: Introduce (Block)SparseMatrix::MiddleRows and deprecate this function.
template <typename MatrixType, typename VectorIn, typename VectorOut, typename Idx>
void ApplyAddToRange(
    MatrixType const& A,
    VectorIn const& v,
    VectorOut&& Av,
    Idx rowBegin,
    Idx rowEnd) {
  if constexpr (IsMatrix<MatrixType>) {
    Av.MiddleRows(rowBegin, rowEnd - rowBegin) += A.MiddleRows(rowBegin, rowEnd - rowBegin) * v;
  } else {
    static_assert(
        IsSparseMatrix<MatrixType> || IsBlockSparseMatrix<MatrixType>, "Unsupported matrix type");
    MOCHI_ASSERT_VERBOSE(v.Cols() == Av.Cols(), "Inconsistent number of columns.");
    using DestAccessorType = decltype(mochi::details::GetAccessor(Av));
    A.AccessorApplyToRange(
        mochi::details::GetAccessor(v),
        mochi::details::DestinationAccessor<mochi::details::DestOp::Add, DestAccessorType>(
            mochi::details::GetAccessor(Av)),
        rowBegin,
        rowEnd,
        v.Cols());
  }
}

template <typename MatrixType, typename VectorIn, typename VectorOut>
void Solve(MatrixType const& P, VectorIn const& v, VectorOut&& Pv) {
  P(v, Pv);
}

template <typename MatrixType, typename VectorInOut>
auto UpperSolveInPlace(MatrixType const& mat, VectorInOut&& outMatV)
    -> decltype(mat.solveInPlace(outMatV)) {
  return mat.solveInPlace(outMatV);
}

template <typename MatrixType>
auto Block(MatrixType& mat, int p, int q, int r, int s) -> decltype(mat.block(p, q, r, s)) {
  return mat.block(p, q, r, s);
}

template <typename MatrixType>
auto Block(MatrixType& mat, int p, int q, int r, int s) -> decltype(mat.Block(p, q, r, s)) {
  return mat.Block(p, q, r, s);
}

template <typename MatrixType>
auto Col(MatrixType& mat, int i) -> decltype(mat.col(i)) {
  return mat.col(i);
}

template <typename MatrixType>
auto Col(MatrixType& mat, int i) -> decltype(mat.Col(i)) {
  return mat.Col(i);
}

template <typename MatrixType>
auto LeftCols(MatrixType& mat, int i) -> decltype(mat.leftCols(i)) {
  return mat.leftCols(i);
}

template <typename MatrixType>
auto LeftCols(MatrixType& mat, int i) -> decltype(mat.LeftCols(i)) {
  return mat.LeftCols(i);
}

template <typename MatrixType>
auto NumRows(MatrixType& mat) -> decltype(mat.rows()) {
  return mat.rows();
}

template <typename MatrixType>
auto NumRows(MatrixType& mat) -> decltype(mat.Rows()) {
  return mat.Rows();
}

template <typename T>
void SetZero(T& t) {
  if constexpr (std::is_arithmetic_v<T>) {
    t = T{};
  } else {
    customization::SetZero(t);
  }
}

template <typename T>
void SetZero(T&& t) {
  if constexpr (std::is_arithmetic_v<T>) {
    t = T{};
  } else {
    customization::SetZero(t);
  }
}

template <typename MatrixType>
auto TopRows(MatrixType& mat, int i) -> decltype(mat.topRows(i)) {
  return mat.topRows(i);
}

template <typename MatrixType>
auto TopRows(MatrixType& mat, int i) -> decltype(mat.TopRows(i)) {
  return mat.TopRows(i);
}

/** Dot product functor object that works out of the box.
 * Similar objects can be created for other types of vector/matrices.
 */
struct UsualDot {
  /// @brief Dot product between two vectors.
  ///
  /// @tparam VectorLeft Type of input vector on the left
  /// @tparam VectorRight Type of input vector on the right
  /// @param[in] x Vector on the left (with 1 column)
  /// @param[in] y Vector on the right (with 1 column)
  /// @returns Scalar for the result of dot product between x and y
  /// (mathematically it is x^T y)
  ///
  /// @note The result is expected to be a scalar.
  template <typename VectorLeft, typename VectorRight>
  auto operator()(VectorLeft const& x, VectorRight const& y) {
    return x.Dot(y);
  }

  /// @brief Norm of a vector (consistent with the dot product)
  ///
  /// @tparam VectorIn Type of input vector
  /// @param[in] x Vector (with 1 column)
  /// @returns Scalar for the norm of x
  ///
  template <typename VectorIn>
  auto Norm(VectorIn const& x) {
    return Sqrt((*this)(x, x));
  }

  /// @brief Square of the norm of a vector.
  ///
  /// @tparam VectorIn Type of input vector
  /// @param[in] x Vector (with 1 column)
  /// @returns Scalar for the squared norm of x
  ///
  template <typename VectorIn>
  auto NormSqr(VectorIn const& x) {
    return (*this)(x, x);
  }

  /// @brief Dot product between two matrices.
  ///
  /// @tparam VectorLeft Type of input matrix on the left
  /// @tparam VectorRight Type of input matrix on the right
  /// @param[in] A Matrix on the left (with 1 column)
  /// @param[in] B Matrix on the right (with 1 column)
  /// @returns Matrix for the result of x^T y
  ///
  /// @note The result is expected to be stored as a matrix.
  /// In particular, when x and y have exactly 1 column, the result is
  /// a matrix of size 1 x 1 (and not a scalar).
  template <typename MatrixLeft, typename MatrixRight>
  auto MatrixWise(MatrixLeft const& x, MatrixRight const& y) {
    return Dot(x, y);
  }
};

/* Utility to perform dot products using an existing pool of parallel workers. It can be used to
 * perform an arbitrary number of dot products, one after another. The dot product is a runtime
 * argument so that subsequent dot products can be of different type.
 *
 * WARNING: Each worker must have a COPY (not a reference) of the parallel dot object.
 *
 * EXAMPLE:
 *     ParallelDot<real> parDot(5);
 *     UsualDot dotType1 = {};
 *     MyCustomDot dotType2 = {};
 *     for (int i = 0; i < 5; ++i) {
 *       scheduler.AddTask([parDot]() { // Capture parDot BY VALUE
 *         // Do work
 *         parDot.Dot(dotType1, x1, x2, rowStart, rowEnd, workerIdx);
 *         // Do more work
 *         parDot.Dot(dotType2, x3, x4, rowStart, rowEnd, workerIdx);
 *       });
 *     }
 */
template <typename Scalar>
class ParallelDot final {
 public:
  // By using two counters/workspaces and switching between them from one dot product to another,
  // dot products can be performed with a single countdown instead of two.
  static constexpr int kStride = 2;

  ParallelDot() = delete;
  ParallelDot(ParallelDot const&) = default;
  ParallelDot(ParallelDot&&) noexcept = default;
  MOCHI_DECLARE_NO_ASSIGN(ParallelDot);

  explicit ParallelDot(int numWorkers) : _numWorkers(numWorkers) {
    MOCHI_ASSERT_VERBOSE(_numWorkers > 0, "Number of workers must be positive.");
    for (auto& ws : *_workspace) {
      ws.resize(_numWorkers);
    }
    (*_count)[0] = _numWorkers; // Mark 1st dot product as ready.
  }

  // Reduce the number of workers that use the parallel dot.
  void ReduceNumWorkers(int numWorkers, bool isMaster) {
    MOCHI_ASSERT_VERBOSE(numWorkers <= _numWorkers, "Invalid new number of workers.");
    if (isMaster && numWorkers < _numWorkers) {
      (*_count)[_idx] -= (_numWorkers - numWorkers);
    }
    _numWorkers = numWorkers;
  }

  template <typename DotType, typename Vec1, typename Vec2, typename Idx>
  Scalar Dot(DotType& dot, Vec1 const& v1, Vec2 const& v2, Idx rowStart, Idx rowEnd, int workerIdx)
      const {
    MOCHI_ASSERT_VERBOSE((workerIdx >= 0) && (workerIdx < _numWorkers));
    MOCHI_ASSERT_VERBOSE((rowStart >= 0) && (rowStart <= rowEnd));
    auto& workspace = (*_workspace)[_idx];
    auto& count = (*_count)[_idx];

    // Compute contribution from this worker.
    auto const partialResult = static_cast<Scalar>(dot(
        v1.MiddleRows(rowStart, rowEnd - rowStart), v2.MiddleRows(rowStart, rowEnd - rowStart)));

    // Wait for previous dot product to be completed.
    auto nonZeroCounter = [&count]() { return count != 0; };
    BusyWaitFor(nonZeroCounter);

    // Update shared workspace, decrease counter and wait for all other workers to be done.
    workspace[workerIdx] = partialResult;
    int const newCount = --count;
    MOCHI_ASSERT_VERBOSE(newCount >= 0);
    bool const isLast = (newCount == 0);
    auto zeroCounter = [&count]() { return count == 0; };
    BusyWaitFor(zeroCounter);

    // Add contributions from all workers.
    Scalar const result = HSum(Span(workspace.data(), _numWorkers));

    // Switch index for the next dot product.
    _idx = (_idx + 1) % kStride;

    // Increase counter for the next dot product to indicate it's ready.
    if (isLast) {
      auto& nextCounter = (*_count)[_idx];
      MOCHI_ASSERT_VERBOSE(nextCounter == 0);
      nextCounter += _numWorkers;
    }
    return result;
  }

 private:
  int _numWorkers = {};
  std::shared_ptr<std::array<std::vector<Scalar>, kStride>> const _workspace =
      std::make_shared<std::array<std::vector<Scalar>, kStride>>();
  std::shared_ptr<std::array<std::atomic<int>, kStride>> const _count =
      std::make_shared<std::array<std::atomic<int>, kStride>>();
  mutable int _idx = 0;
};

} // namespace mochi::krylov
