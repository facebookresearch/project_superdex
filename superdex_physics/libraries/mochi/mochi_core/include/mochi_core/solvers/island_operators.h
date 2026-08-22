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
#include <mochi_core/linear_algebra/krylov/preconditioner.h>
#include <mochi_core/linear_algebra/krylov/tools/tensor_functions.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/mochi_platform.h>
#include <mochi_core/solvers/actor_preconditioner.h>
#include <mochi_core/solvers/interaction_matrix_info.h>

#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace mochi {

// Forward declaration.
template <typename T>
struct IslandOperators;

template <typename T>
struct ActorPrecApplyer {
  int offset;
  int size;
  std::reference_wrapper<ActorPreconditioner<T>> prec;

  void operator()(ColumnVectorView<T const> x, ColumnVectorView<T> y) const {
    prec.get().Solve(x.MiddleRows(offset, size), y.MiddleRows(offset, size));
  }

  void ConcurrentSolve(
      ColumnVectorView<T const> x,
      ColumnVectorView<T> Px,
      ParallelWorkerInfo const& data) const {
    MOCHI_ASSERT_VERBOSE(data.rBegin >= 0 && data.rBegin <= data.rEnd, "Invalid row range.");
    if ((offset < data.rEnd) && (data.rBegin < offset + size)) {
      prec.get().ConcurrentSolve(
          x.MiddleRows(offset, size),
          Px.MiddleRows(offset, size),
          ParallelWorkerInfo{
              .workerId = data.workerId,
              .numWorkers = data.numWorkers,
              .rBegin = Max(data.rBegin - offset, 0),
              .rEnd = Min(data.rEnd - offset, size),
              .barrier = data.barrier});
    }
  }
};

template <typename T>
struct PerActorPrec final : Preconditioner<T> {
  static constexpr auto kType = PreconditionerType::PerActor;

  std::vector<ActorPrecApplyer<T>> actorPrecs;

  explicit PerActorPrec(std::vector<ActorPrecApplyer<T>>&& actorPrecs)
      : actorPrecs(std::move(actorPrecs)) {}

  void Solve(ColumnVectorView<T const> x, ColumnVectorView<T> y) const override {
    // TODO[T175051452]: Introduce efficient parallelization.
    for (auto const& prec : actorPrecs) {
      prec(x, y);
    }
  }

  void ConcurrentSolve(
      ColumnVectorView<T const> x,
      ColumnVectorView<T> Px,
      ParallelWorkerInfo const& data) const override {
    MOCHI_ASSERT_VERBOSE(data.rBegin >= 0 && data.rBegin <= data.rEnd, "Invalid row range.");
    for (auto const& prec : actorPrecs) {
      if ((prec.offset < data.rEnd) && (data.rBegin < prec.offset + prec.size)) {
        prec.ConcurrentSolve(x, Px, data);
      }
    }
  }

  void Update(IslandOperators<T> const& A) {
    // Note that MakePerActorPrec updates the actor preconditioners if they already exist.
    *this = std::move(A.MakePerActorPrec());
  }

  constexpr PreconditionerType GetType() const override {
    return kType;
  }
};

/*******************************************************************
  IslandOperators
*/
template <typename T>
struct IslandOperators {
 public:
  using Scalar = T;
  using NonConstScalar = std::remove_const_t<T>;

 private:
  /// @brief Vector of views for the actor matrices
  ///
  /// @note Exactly one pair of DOF offset and matrix for each actor.
  /// Together, they cover the full range of DOFs with no overlap and no gaps.
  /// @note The pairs should be sorted by increasing offset.
  std::vector<std::pair<int, AnyMatrixView<T const>>> _actorMatrices;

  /// @brief Vector of views for the interaction matrices.
  ///
  /// @note If any of the actors interact with each other, then there will be one or more
  /// interaction matrices. Examples include sync contact and static constraints.
  /// @note If an interaction matrix is not square or has different row and col offsets, then it
  /// must NOT overlap with the block diagonal corresponding to an actor, i.e. it must be an
  /// off-diagonal submatrix for the interaction between two actors.
  ///
  std::vector<AnyInteractionMatrixViewInfo<T const>> _interactionMatrices;

  /// @brief Vector of actor preconditioners.
  ///
  /// @note Actors are stored in the same order as in _actorMatrices.
  /// @note The vector can be empty, in which case the per-actor preconditioner cannot be
  /// constructed.
  std::vector<std::reference_wrapper<std::unique_ptr<ActorPreconditioner<T>>>>
      _actorPreconditioners;

  // Optional per-actor preconditioner type hints. If non-empty, must match the size of
  // _actorMatrices. If empty or for actors with std::nullopt, the preconditioner type is
  // auto-selected.
  std::vector<std::optional<PreconditionerType>> _actorPreconditionerTypeHints;

  // Number of workers to apply the IslandOperators to a dense matrix. Optimized for the application
  // on a column vector. Lazily evaluated the first time 'Apply' is called.
  mutable int _numWorkers = 0;

  // Range of rows that each of the workers is responsible for in 'Apply'. Lazily evaluated the
  // first time 'Apply' is called.
  mutable std::vector<int> _workerRowRanges = {};

 public:
  // Construct IslandOperators and enforce policies
  IslandOperators(
      std::vector<std::pair<int, AnyMatrixView<T const>>> actorMatrices,
      std::vector<AnyInteractionMatrixViewInfo<T const>> interactionMatrices,
      std::vector<std::reference_wrapper<std::unique_ptr<ActorPreconditioner<T>>>>
          actorPreconditioners,
      std::vector<std::optional<PreconditionerType>> actorPreconditionerTypeHints = {})
      : _actorMatrices(std::move(actorMatrices)),
        _interactionMatrices(std::move(interactionMatrices)),
        _actorPreconditioners(std::move(actorPreconditioners)),
        _actorPreconditionerTypeHints(std::move(actorPreconditionerTypeHints)) {
    // TODO[pabfer]: Store actor matrices and actor preconditioners is a single struct containing
    // the actor offset, the actor matrix and the actor preconditioner.
#if MOCHI_ASSERT_VERBOSE_ENABLED
    int expectedOffset = 0;
    for (auto const& [offset, anyMat] : _actorMatrices) {
      MOCHI_ASSERT_VERBOSE(
          offset == expectedOffset,
          "Actor matrices should be sorted by offset and there should be no gaps or overlap between them.");
      MOCHI_ASSERT_VERBOSE(GetNumRows(anyMat) != 0, "Actor matrix should not be empty");
      expectedOffset += GetNumRows(anyMat);
    }
    int totalNumDofs = expectedOffset;
    for (auto const& [rOffset, cOffset, anyMat, _] : _interactionMatrices) {
      MOCHI_ASSERT_VERBOSE(
          rOffset >= 0 && cOffset >= 0 && rOffset + GetNumRows(anyMat) <= totalNumDofs &&
              cOffset + GetNumCols(anyMat) <= totalNumDofs,
          "Interaction matrix is out of range.");
      MOCHI_ASSERT_VERBOSE(GetNumRows(anyMat) != 0, "Interaction matrix should not be empty");
    }
    MOCHI_ASSERT_VERBOSE(
        _actorPreconditioners.empty() || (_actorPreconditioners.size() == _actorMatrices.size()),
        "The vector of actor preconditioners must be empty or the same size as the vector of actor matrices.");
    MOCHI_ASSERT_VERBOSE(
        _actorPreconditionerTypeHints.empty() ||
            (_actorPreconditionerTypeHints.size() == _actorMatrices.size()),
        "The vector of preconditioner type hints must be empty or the same size as the vector of actor matrices.");
#endif // MOCHI_ASSERT_VERBOSE_ENABLED
  }

  // Construct or update the per-actor preconditioner.
  auto MakePerActorPrec() const {
    auto const numActors = isize(_actorMatrices);
    MOCHI_ASSERT(
        _actorPreconditioners.size() == numActors,
        "Actor preconditioners are not set. The per-actor preconditioner cannot be constructed.");

    // Construct or update the preconditioner of each actor.
    // TODO[T175051452]: Optimize parallelization.
    ParallelForN("PerActorPrecUpdate", numActors, 1, [this](int aIndex) {
      auto const& [offset, A] = _actorMatrices[aIndex];
      auto& actorPrec = _actorPreconditioners[aIndex].get();
      bool const hasTypeHint = !_actorPreconditionerTypeHints.empty() &&
          _actorPreconditionerTypeHints[aIndex].has_value();
      if (!actorPrec) {
        // Preconditioner doesn't exist yet. Construct it.
        PreconditionerType precType = {};
        if (hasTypeHint) {
          // Use preconditioner type hint.
          precType = _actorPreconditionerTypeHints[aIndex].value();
        } else {
          // Auto-select based on matrix type.
          // TODO[T247922835]: Improve criteria to select the preconditioner type.
          static_assert(
              std::variant_size_v<decltype(A)> == 4,
              "Please update the if statement below if the actor matrix types change");
          if (std::holds_alternative<MatrixView<T const>>(A)) {
            // Use symmetric inverse preconditioner for dense actors.
            precType = PreconditionerType::SymInverse;
          } else if (
              std::holds_alternative<BlockSparseMatrixView<T const, 3>>(A) ||
              std::holds_alternative<BlockSparseMatrixView<T const, 4>>(A)) {
            // Use block Jacobi preconditioner for block sparse actors.
            precType = PreconditionerType::BlockJacobi;
          } else {
            MOCHI_ASSERT_VERBOSE(std::holds_alternative<SparseMatrixView<T const>>(A));
            // Use Jacobi preconditioner for sparse actors.
            precType = PreconditionerType::Jacobi;
          }
        }
        actorPrec = CreateActorPreconditioner(precType, offset, A, _interactionMatrices);
      } else {
        // Preconditioner already exists. Update it.
        MOCHI_ASSERT(
            !hasTypeHint || *_actorPreconditionerTypeHints[aIndex] == actorPrec->GetType(),
            "Inconsistent preconditioner type.");
        actorPrec->Update({offset, A, _interactionMatrices});
      }
    });

    // Construct the per-actor preconditioner.
    std::vector<ActorPrecApplyer<T>> actorPrecApplyiers;
    actorPrecApplyiers.reserve(numActors);
    for (int aIndex = 0; aIndex < numActors; ++aIndex) {
      auto const& [offset, A] = _actorMatrices[aIndex];
      actorPrecApplyiers.push_back({offset, GetNumRows(A), *_actorPreconditioners[aIndex].get()});
    }

    return PerActorPrec<T>{std::move(actorPrecApplyiers)};
  }

  // Return the number of rows of the global system.
  [[nodiscard]] int Rows() const;

  // Return the number of columns of the global system.
  [[nodiscard]] int Cols() const;

  // Const reference to the actor matrices.
  [[nodiscard]] auto const& GetActorMatrices() const;

  // Const reference to the interaction matrices.
  [[nodiscard]] auto const& GetInteractionMatrices() const;

  // Condense all matrices into a single global SparseMatrix
  [[nodiscard]] SparseMatrix<T> FullSparseMatrix() const;

  // Condense all matrices into a single global BlockSparseMatrix. It requires all actor and
  // interaction matrices to be blockable. Otherwise, an empty matrix is returned, even if the
  // condensed global matrix is blockable.
  template <int kBlockSize>
  [[nodiscard]] BlockSparseMatrix<T, kBlockSize> FullBlockSparseMatrix() const;

  // Condense all matrices into a single global matrix. The matrix is returned with the storage type
  // that is most efficient for the linear solver, e.g. block sparse format is preferred over sparse
  // format whenever possible.
  [[nodiscard]] AnyMatrix<T> CondenseFullMatrix() const;

  // True if all actor and interaction matrices are blockable. False otherwise, even if the
  // condensed global matrix is blockable.
  template <int kBlockSize>
  [[nodiscard]] bool IsBlockable() const;

  /// @brief Application of the IslandOperators on a dense matrix, including a column vector.
  /// @note If the number of FLOPs is sufficiently large and a TaskScheduler is available and in
  /// multi-threaded mode (both global and thread-local), the computation is performed in parallel.
  template <typename InVector, typename OutVector>
  void Apply(InVector const& x, OutVector&& Ax) const;

  /// @brief Application of the row subset [rowBegin, rowEnd) of the IslandOperators on a dense
  /// matrix, including a column vector.
  /// @note If the number of FLOPs is sufficiently large and a TaskScheduler is available and in
  /// multi-threaded mode (both global and thread-local), the computation is performed in parallel.
  template <typename InVector, typename OutVector, typename Idx>
  void ApplyToRange(InVector const& x, OutVector&& Ax, Idx rowBegin, Idx rowEnd) const;
};

template <typename T>
int IslandOperators<T>::Rows() const {
  if (_actorMatrices.empty()) {
    return 0;
  } else {
    auto const& [offset, lastMatrix] = _actorMatrices.back();
    return offset + GetNumRows(lastMatrix);
  }
}

template <typename T>
int IslandOperators<T>::Cols() const {
  if (_actorMatrices.empty()) {
    return 0;
  } else {
    auto const& [offset, lastMatrix] = _actorMatrices.back();
    return offset + GetNumCols(lastMatrix);
  }
}

template <typename T>
auto const& IslandOperators<T>::GetActorMatrices() const {
  return _actorMatrices;
}

template <typename T>
auto const& IslandOperators<T>::GetInteractionMatrices() const {
  return _interactionMatrices;
}

template <typename T>
template <typename InVector, typename OutVector>
void IslandOperators<T>::Apply(InVector const& x, OutVector&& Ax) const {
  if (Rows() == 0)
    MOCHI_UNLIKELY {
      return;
    }
  if (_workerRowRanges.empty()) {
    constexpr int kNumNzPerWorker = 25000; // Empirically chosen value. Optimized for x.Cols() == 1.
    _numWorkers =
        Max(1,
            Min(2 * (TaskScheduler::StaticGetNumOtherThreads() + 1),
                GetNumValues(*this) / kNumNzPerWorker,
                Rows()));
    _workerRowRanges = GetRowRangesPerWorker(*this, _numWorkers);
  }
  MOCHI_ASSERT_VERBOSE(_numWorkers > 0 && _workerRowRanges.size() == _numWorkers + 1);
  ParallelForN("IslandOperatorsApply", _numWorkers, 1, [&, this](int workerIdx) {
    // Disable nested parallelization. IslandOperators::Apply owns the parallelization strategy and
    // distribution of work.
    TaskScheduler::PushLocalSingleThreadedMode();
    ApplyToRange(x, Ax, _workerRowRanges[workerIdx], _workerRowRanges[workerIdx + 1]);
    TaskScheduler::PopLocalSingleThreadedMode();
  });
}

template <typename T>
template <typename InVector, typename OutVector, typename Idx>
void IslandOperators<T>::ApplyToRange(InVector const& x, OutVector&& Ax, Idx rowBegin, Idx rowEnd)
    const {
  MOCHI_ASSERT_VERBOSE(
      rowBegin >= 0 && rowBegin <= rowEnd && rowEnd <= this->Rows(), "Invalid row range.");
  for (auto const& [offset, Op] : _actorMatrices) {
    auto const numRows = GetNumRows(Op);
    if ((offset < rowEnd) && (offset + numRows > rowBegin)) {
      auto const actualRowBegin = Max(offset, rowBegin);
      auto const actualRowEnd = Min(offset + numRows, rowEnd);
      auto offsetCopy = offset; // prevents a compiler warning
      std::visit(
          [&](auto const& A) {
            krylov::ApplyToRange(
                A,
                x.MiddleRows(offsetCopy, A.Cols()),
                Ax.MiddleRows(offsetCopy, A.Rows()),
                actualRowBegin - offsetCopy,
                actualRowEnd - offsetCopy);
          },
          Op);
    }
  }

  for (auto const& [rOffset, cOffset, Op, _] : _interactionMatrices) {
    auto const numRows = GetNumRows(Op);
    if (rOffset < rowEnd && rOffset + numRows > rowBegin) {
      auto const actualRowBegin = Max(rOffset, rowBegin);
      auto const actualRowEnd = Min(rOffset + numRows, rowEnd);
      std::visit(
          [&, rowOffset = rOffset, colOffset = cOffset](auto const& I) {
            ApplyAddToRange(
                I,
                x.MiddleRows(colOffset, I.Cols()),
                Ax.MiddleRows(rowOffset, I.Rows()),
                actualRowBegin - rowOffset,
                actualRowEnd - rowOffset);
          },
          Op);
    }
  }
}

template <typename T>
std::vector<int> GetRowRangesPerWorker(IslandOperators<T> const& ops, int numWorkers) {
  MOCHI_ASSERT_VERBOSE(numWorkers > 0, "Invalid number of workers.");
  int interactionBlockSize = 1;
  for (auto const& [rowOffset, colOffset, anyMat, symmetricPair] : ops.GetInteractionMatrices()) {
    static_assert(std::variant_size<AnyMatrixView<T const>>::value == 4, "Please update this code");
    if (std::holds_alternative<BlockSparseMatrixView<T const, 3>>(anyMat)) {
      interactionBlockSize = 3;
      break;
    } else if (std::holds_alternative<BlockSparseMatrixView<T const, 4>>(anyMat)) {
      interactionBlockSize = 4;
      break;
    }
  }

  auto const numRows = ops.Rows();
  std::vector<int> workerRowRanges(numWorkers + 1, 0);
  int i = 0;
  for (auto const& [offset, Op] : ops.GetActorMatrices()) {
    // The logic below could use the interaction block size of only the interaction matrices that
    // affect the actor (not of all the interaction matrices). That would make this function more
    // expensive but lead to slightly better load balancing.
    auto const numActorRows = GetNumRows(Op);
    int const actorBlockSize = static_cast<int>(numActorRows / GetNumBlockRows(Op));
    int const blockSizeLcm = std::lcm(interactionBlockSize, actorBlockSize);
    MOCHI_ASSERT_VERBOSE(
        blockSizeLcm % Min(interactionBlockSize, actorBlockSize) == 0,
        "Unsupported case. The least common multiple must be the max.");
    while (i < numWorkers) {
      auto const candidateRow = static_cast<int>(static_cast<int64_t>(i) * numRows / numWorkers);
      if (candidateRow - offset <= numActorRows) {
        // Ensure row range is consistent with the block size.
        workerRowRanges[i++] = offset + ((candidateRow - offset) / blockSizeLcm) * blockSizeLcm;
      } else {
        break;
      }
    }
  }
  MOCHI_ASSERT(i == numWorkers, "Unexpected index.");
  workerRowRanges[numWorkers] = numRows;
  return workerRowRanges;
}

/*******************************************************************
  IslandOperatorsOwningLite
*/
/// @brief Owning light class for island operators.
///
/// @note This class is used, for example, when converting an @ref IslandOperators object
/// to a different floating point representation.
/// @note IslandOperatorsOwningLite only contains actor matrices and interaction matrices,
/// but not all other @ref IslandOperators properties such as actor preconditioner.
/// @note This class provides only one member function, namely @ref AsConstView to convert
/// into an @ref IslandOperators object.
template <typename Scalar_>
struct IslandOperatorsOwningLite {
 public:
  using Scalar = Scalar_;
  using NonConstScalar = std::remove_const_t<Scalar>;

  /// @brief Vector of "owned" actor matrices.
  ///
  /// @note See @ref IslandOperators<Scalar>::_actorMatrices for further details.
  ///
  std::vector<std::pair<int, AnyMatrix<NonConstScalar>>> actorMatrices;

  /// @brief Vector of "owned" interaction matrices.
  ///
  /// @note See @ref IslandOperators<Scalar>::_interactionMatrices for further details.
  ///
  std::vector<AnyInteractionMatrixInfo<NonConstScalar>> interactionMatrices;

  IslandOperators<NonConstScalar> AsConstView() const {
    std::vector<std::pair<int, AnyMatrixView<Scalar const>>> newActorView;
    newActorView.reserve(actorMatrices.size());
    for (auto const& [offset, anyMat] : actorMatrices) {
      auto anyMatView = mochi::AsConstView(anyMat);
      newActorView.emplace_back(offset, anyMatView);
    }
    std::vector<AnyInteractionMatrixViewInfo<Scalar const>> newInteractionView;
    newInteractionView.reserve(interactionMatrices.size());
    for (auto const& interData : interactionMatrices) {
      newInteractionView.emplace_back(
          interData.rowOffset,
          interData.colOffset,
          mochi::AsConstView(interData.matrix),
          interData.symmetricPair);
    }
    return IslandOperators<NonConstScalar>(
        std::move(newActorView), std::move(newInteractionView), {});
  }
};
} // namespace mochi

namespace mochi::details {
template <typename T>
constexpr bool IsIslandOperatorsOwningLiteDef<IslandOperatorsOwningLite<T>> = true;
} // namespace mochi::details

/*******************************************************************
  Functions
*/

namespace mochi {

/// @brief Get the number of non-zeros in an IslandOperators.
/// @note If a non-zero is present in multiple actors and/or interaction matrices, it's counted
/// multiple times.
template <typename T>
int GetNumValues(IslandOperators<T> const& ops) {
  int numValues = 0;
  for (auto const& [_, Op] : ops.GetActorMatrices()) {
    numValues += GetNumValues(Op);
  }
  for (auto const& [rOffset, cOffset, Op, symmetricPair] : ops.GetInteractionMatrices()) {
    numValues += GetNumValues(Op);
  }
  return numValues;
}

/// @brief Approximate number of FLOPs to apply the IslandOperators to a column vector.
template <typename T>
inline auto FlopsPerApply(IslandOperators<T> const& ops) {
  return 2 * GetNumValues(ops);
}

} // namespace mochi

namespace mochi::details {
template <typename T>
constexpr bool IsIslandOperatorsDef<IslandOperators<T>> = true;
} // namespace mochi::details

namespace mochi {

// Implemented for float & double in island_operators.cpp
extern template struct IslandOperators<float>;
extern template struct IslandOperators<double>;

} // namespace mochi
