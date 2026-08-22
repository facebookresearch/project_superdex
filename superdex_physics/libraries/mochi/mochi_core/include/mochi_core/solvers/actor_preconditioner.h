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

#include <mochi_core/linear_algebra/actor_pseudo_matrix.h>
#include <mochi_core/linear_algebra/any_matrix.h>
#include <mochi_core/linear_algebra/krylov/amg/amg_prec.h>
#include <mochi_core/linear_algebra/krylov/block_jacobi_prec.h>
#include <mochi_core/linear_algebra/krylov/relaxed_ilu_prec.h>
#include <mochi_core/linear_algebra/krylov/sym_inverse_prec.h>
#include <mochi_core/linear_algebra/utils/matrix_conversions.h>
#include <mochi_core/solvers/interaction_matrix_info.h>
#include <mochi_core/solvers/linear_solver_params.h>

#include <memory>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace mochi {

/** @brief Abstract class for the preconditioner of an actor.
 *
 * REQUIREMENTS: All child classes must satisfy the following requirements:
 * - ConcurrentSolve must NOT perform synchronization (synchronization may be problematic if workers
 *   are responsible for a subset of the rows of multiple actors).
 * - It must be safe to reuse the preconditioner across multiple linear solves. This implies all the
 *   data must either be owned by the preconditioner or be a reference/view to an object that will
 *   outlive the preconditioner.
 */
template <typename T>
struct ActorPreconditioner {
  virtual ~ActorPreconditioner() = default;

  /** @brief Apply the preconditioner to a column vector.
   */
  virtual void Solve(ColumnVectorView<T const> x, ColumnVectorView<T> Px) const = 0;

  /** @brief Concurrent application of the preconditioner to a column vector by a pool of workers.
   * The calling worker is responsible for applying its own preconditioner contribution.
   *
   * @param[in] x Input column vector.
   * @param[out] Px Output column vector.
   * @param[in] data Parallel information for each worker.
   *
   * @note Marked as pure virtual to ensure that all preconditioners in the per-actor framework are
   * compatible with parallel solvers.
   */
  virtual void ConcurrentSolve(
      ColumnVectorView<T const> x,
      ColumnVectorView<T> Px,
      ParallelWorkerInfo const& data) const = 0;

  /** @brief Update the preconditioner's data with new values of the involved matrices.
   *
   * @note The update assumes that the sparsity (for the numerical entries of interest) is
   * unchanged. If the sparsity (for the numerical entries of interest) has changed, the
   * preconditioner has to be recreated.
   */
  virtual void Update(ActorPseudoMatrix<T> const& actorMatrix) = 0;

  /**
   * @brief Get the preconditioner type.
   *
   * @note Returns the underlying preconditioner type used for this specific actor (e.g.,
   * PreconditionerType::Jacobi, PreconditionerType::AMG). It does not return
   * PreconditionerType::PerActor, which is the domain decomposition wrapper that applies per-actor
   * preconditioners to each subdomain.
   */
  virtual constexpr PreconditionerType GetType() const = 0;
};

template <typename T, int kBlockSize>
class BlockJacobiActorPrec : public ActorPreconditioner<T> {
 public:
  static_assert(kBlockSize > 0, "Preconditioner block size must be positive");

  explicit BlockJacobiActorPrec(ActorPseudoMatrix<T> const& A) : prec(A) {}

  void Solve(ColumnVectorView<T const> x, ColumnVectorView<T> Px) const override {
    prec.Solve(x, Px);
  }

  void ConcurrentSolve(
      ColumnVectorView<T const> x,
      ColumnVectorView<T> Px,
      ParallelWorkerInfo const& data) const override {
    prec.ConcurrentSolve(x, Px, data);
  }

  void Update(ActorPseudoMatrix<T> const& actorMatrix) override {
    prec.Update(actorMatrix);
  }

  constexpr PreconditionerType GetType() const override {
    return kBlockSize > 1 ? PreconditionerType::BlockJacobi : PreconditionerType::Jacobi;
  }

  krylov::BlockJacobiPrec<T, kBlockSize> prec;
};

/** @brief Symmetric inverse actor preconditioner. */
template <typename T>
class SymInverseActorPrec : public ActorPreconditioner<T> {
 public:
  explicit SymInverseActorPrec(ActorPseudoMatrix<T> const& A) : prec(A) {}

  void Solve(ColumnVectorView<T const> x, ColumnVectorView<T> Px) const override {
    prec.Solve(x, Px);
  }

  void ConcurrentSolve(
      ColumnVectorView<T const> x,
      ColumnVectorView<T> Px,
      ParallelWorkerInfo const& data) const override {
    prec.ConcurrentSolve(x, Px, data);
  }

  void Update(ActorPseudoMatrix<T> const& A) override {
    prec.Update(A);
  }

  constexpr PreconditionerType GetType() const override {
    return PreconditionerType::SymInverse;
  }

  krylov::SymInversePrec<T> prec;
};

/**
 * @brief Class for the AMG actor preconditioner.
 *
 * @note Per-actor AMG preconditioner is only supported for actors whose @ref ActorPseudoMatrix can
 * be converted into a block sparse matrix of the desired block size.
 */
template <typename T, int kBlockSize>
class AMGActorPrec : public ActorPreconditioner<T> {
 public:
  static_assert(kBlockSize > 0, "Preconditioner block size must be positive");

  explicit AMGActorPrec(ActorPseudoMatrix<T> const& A, krylov::AMGOptions<T> const& options = {}) {
    Afine = ToBlockSparseMatrix<kBlockSize, true>(A);
    prec = std::make_unique<krylov::AMGPrec<T, kBlockSize>>(Afine, options);
  }

  void Solve(ColumnVectorView<T const> x, ColumnVectorView<T> Px) const override {
    prec->Solve(x, Px);
  }

  void ConcurrentSolve(
      ColumnVectorView<T const> x,
      ColumnVectorView<T> Px,
      ParallelWorkerInfo const& data) const override {
    prec->ConcurrentSolve(x, Px, data);
  }

  /** @brief Update the preconditioner for the input pseudo-matrix.
   *
   * @param[in] actorMatrix Novel actor pseudo-matrix to update the preconditioner.
   *
   * @note See the sparsity pattern assumptions in @ref ActorPreconditioner::Update.
   */
  void Update(ActorPseudoMatrix<T> const& actorMatrix) override {
#if MOCHI_ASSERT_VERBOSE_ENABLED
    // Expensive check to verify the sparsity is unchanged
    auto Anew = ToBlockSparseMatrix<kBlockSize, true>(actorMatrix);
    MOCHI_ASSERT_VERBOSE(Anew.Pointers() == Afine.Pointers(), "Sparsity pattern mismatch.");
    MOCHI_ASSERT_VERBOSE(Anew.Indices() == Afine.Indices(), "Sparsity pattern mismatch.");
#endif
    // Verify that the actor matrix is block sparse with the correct block size
    using BSpMatrixView = BlockSparseMatrixView<T const, kBlockSize>;
    MOCHI_ASSERT(
        std::holds_alternative<BSpMatrixView>(actorMatrix.actorMatrix),
        "Actor matrix type not supported for AMG actor preconditioner.");
    MOCHI_ASSERT_VERBOSE(actorMatrix.Rows() == actorMatrix.Cols(), "Expected square actor matrix.");

    // Assume that the sparsity pattern is unchanged.
    // Re-use the previous integral arrays to avoid new allocations.
    // Copy the numerical values for the actor matrix.
    auto const& bsp = std::get<BSpMatrixView>(actorMatrix.actorMatrix);
    auto srcValues = bsp.Values();
    std::copy(srcValues.begin(), srcValues.end(), Afine.Values().begin());
    details::AddInteractionToBlockSparseMatrix<true>(
        actorMatrix.interactionMatrices, Afine, actorMatrix.offset);

    prec->Update(Afine);
  }

  constexpr PreconditionerType GetType() const override {
    return PreconditionerType::AMG;
  }

  BlockSparseMatrix<T, kBlockSize> Afine;
  std::unique_ptr<krylov::AMGPrec<T, kBlockSize>> prec;
};

/** @brief Class for the colored SSOR actor preconditioner.
 *
 * @note Per-actor colored SSOR preconditioner is only supported for actors whose @ref
 * ActorPseudoMatrix can be converted into a block sparse matrix of the desired block size.
 */
template <typename T, int kBlockSize>
class ColoredSSORActorPrec : public ActorPreconditioner<T> {
 public:
  static_assert(kBlockSize > 0, "Preconditioner block size must be positive");

  explicit ColoredSSORActorPrec(ActorPseudoMatrix<T> const& A_) {
    A = ToBlockSparseMatrix<kBlockSize, true>(A_);
    prec = std::make_unique<krylov::ColoredSSORPrec<BlockSparseMatrix<T, kBlockSize>>>(A);
  }

  void Solve(ColumnVectorView<T const> x, ColumnVectorView<T> y) const override {
    prec->Solve(x, y);
  }

  void ConcurrentSolve(
      ColumnVectorView<T const> x,
      ColumnVectorView<T> Px,
      ParallelWorkerInfo const& data) const override {
    prec->ConcurrentSolve(x, Px, data);
  }

  /** @brief Update the preconditioner for the input pseudo-matrix.
   *
   * @param[in] actorMatrix Novel actor pseudo-matrix to update the preconditioner.
   *
   * @note See the sparsity pattern assumptions in @ref ActorPreconditioner::Update.
   */
  void Update(ActorPseudoMatrix<T> const& actorMatrix) override {
#if MOCHI_ASSERT_VERBOSE_ENABLED
    // Expensive check to verify the sparsity is unchanged
    auto Anew = ToBlockSparseMatrix<kBlockSize, true>(actorMatrix);
    MOCHI_ASSERT_VERBOSE(Anew.Pointers() == A.Pointers(), "Sparsity pattern mismatch.");
    MOCHI_ASSERT_VERBOSE(Anew.Indices() == A.Indices(), "Sparsity pattern mismatch.");
#endif
    // Verify that the actor matrix is block sparse with the correct block size
    using BSpMatrixView = BlockSparseMatrixView<T const, kBlockSize>;
    MOCHI_ASSERT(
        std::holds_alternative<BSpMatrixView>(actorMatrix.actorMatrix),
        "Actor matrix type not supported for colored SSOR actor preconditioner.");
    MOCHI_ASSERT_VERBOSE(actorMatrix.Rows() == actorMatrix.Cols(), "Expected square actor matrix.");

    // Assume that the sparsity pattern is unchanged.
    // Re-use the previous integral arrays to avoid new allocations.
    // Copy the numerical values for the actor matrix.
    auto const& bsp = std::get<BSpMatrixView>(actorMatrix.actorMatrix);
    auto srcValues = bsp.Values();
    std::copy(srcValues.begin(), srcValues.end(), A.Values().begin());
    details::AddInteractionToBlockSparseMatrix<true>(
        actorMatrix.interactionMatrices, A, actorMatrix.offset);

    prec->Update(A);
  }

  constexpr PreconditionerType GetType() const override {
    return PreconditionerType::ColoredSSOR;
  }

  BlockSparseMatrix<T, kBlockSize> A;
  std::unique_ptr<krylov::ColoredSSORPrec<BlockSparseMatrix<T, kBlockSize>>> prec;
};

/** @brief Class for the ILU0 actor preconditioner.
 *
 * @note Per-actor ILU0 preconditioner is only supported for actors whose @ref
 * ActorPseudoMatrix can be converted into a block sparse matrix of the desired block size.
 */
template <typename T, int kBlockSize>
class ILU0ActorPrec : public ActorPreconditioner<T> {
 public:
  static_assert(kBlockSize > 0, "Preconditioner block size must be positive");

  explicit ILU0ActorPrec(ActorPseudoMatrix<T> const& A_) {
    A = ToBlockSparseMatrix<kBlockSize, true>(A_);
    prec = std::make_unique<krylov::RelaxedILUPrec<BlockSparseMatrix<T, kBlockSize>>>(
        A, /*fillInLevel*/ 0, /*alphaRelax*/ T{0});
  }

  void Solve(ColumnVectorView<T const> x, ColumnVectorView<T> Px) const override {
    prec->Solve(x, Px);
  }

  void ConcurrentSolve(
      ColumnVectorView<T const> x,
      ColumnVectorView<T> Px,
      ParallelWorkerInfo const& data) const override {
    prec->ConcurrentSolve(x, Px, data);
  }

  /** @brief Update the preconditioner for the input pseudo-matrix.
   *
   * @param[in] actorMatrix Novel actor pseudo-matrix to update the preconditioner.
   *
   * @note See the sparsity pattern assumptions in @ref ActorPreconditioner::Update.
   */
  void Update(ActorPseudoMatrix<T> const& actorMatrix) override {
#if MOCHI_ASSERT_VERBOSE_ENABLED
    // Expensive check to verify the sparsity is unchanged
    auto Anew = ToBlockSparseMatrix<kBlockSize, true>(actorMatrix);
    MOCHI_ASSERT_VERBOSE(Anew.Pointers() == A.Pointers(), "Sparsity pattern mismatch.");
    MOCHI_ASSERT_VERBOSE(Anew.Indices() == A.Indices(), "Sparsity pattern mismatch.");
#endif
    // Verify that the actor matrix is block sparse with the correct block size
    using BSpMatrixView = BlockSparseMatrixView<T const, kBlockSize>;
    MOCHI_ASSERT(
        std::holds_alternative<BSpMatrixView>(actorMatrix.actorMatrix),
        "Actor matrix type not supported for ILU0 actor preconditioner.");
    MOCHI_ASSERT_VERBOSE(actorMatrix.Rows() == actorMatrix.Cols(), "Expected square actor matrix.");

    // Assume that the sparsity pattern is unchanged.
    // Re-use the previous integral arrays to avoid new allocations.
    // Copy the numerical values for the actor matrix.
    auto const& bsp = std::get<BSpMatrixView>(actorMatrix.actorMatrix);
    auto srcValues = bsp.Values();
    std::copy(srcValues.begin(), srcValues.end(), A.Values().begin());
    details::AddInteractionToBlockSparseMatrix<true>(
        actorMatrix.interactionMatrices, A, actorMatrix.offset);

    prec->Update(A);
  }

  constexpr PreconditionerType GetType() const override {
    return PreconditionerType::ILU0;
  }

  BlockSparseMatrix<T, kBlockSize> A;
  std::unique_ptr<krylov::RelaxedILUPrec<BlockSparseMatrix<T, kBlockSize>>> prec;
};

template <typename T>
auto CreateActorPreconditioner(
    PreconditionerType precType,
    int offset,
    AnyMatrixView<T const> actorMatrix,
    std::vector<AnyInteractionMatrixViewInfo<T const>> const& interactionMatrices) {
  static_assert(std::is_same_v<T, std::remove_const_t<T>>);
  return std::visit(
      [offset, precType, &interactionMatrices](auto const& A) {
        static_assert(
            std::variant_size_v<decltype(actorMatrix)> == 4,
            "Please update the if statement below if the actor matrix types change");
        using MatType = std::decay_t<decltype(A)>;
        ActorPseudoMatrix<T> actorPseudoMatrix = {offset, A, interactionMatrices};
        if (precType == PreconditionerType::SymInverse) {
          return std::unique_ptr<ActorPreconditioner<T>>{
              new SymInverseActorPrec<T>{std::move(actorPseudoMatrix)}};
        } else if (precType == PreconditionerType::AMG) {
          MOCHI_ASSERT(
              (std::is_same_v<MatType, BlockSparseMatrixView<T const, 3>>),
              "Per-actor AMG preconditioner is only enabled for block sparse actors with block size of 3.");
          return std::unique_ptr<ActorPreconditioner<T>>{
              new AMGActorPrec<T, 3>{std::move(actorPseudoMatrix)}};
        } else if (precType == PreconditionerType::BlockJacobi) {
          if constexpr (std::is_same_v<MatType, BlockSparseMatrixView<T const, 3>>) {
            return std::unique_ptr<ActorPreconditioner<T>>{
                new BlockJacobiActorPrec<T, 3>{std::move(actorPseudoMatrix)}};
          } else {
            MOCHI_ASSERT(
                (std::is_same_v<MatType, BlockSparseMatrixView<T const, 4>>),
                "Per-actor block Jacobi preconditioner is only enabled for block sparse actors.");
            return std::unique_ptr<ActorPreconditioner<T>>{
                new BlockJacobiActorPrec<T, 4>{std::move(actorPseudoMatrix)}};
          }
        } else if (precType == PreconditionerType::ColoredSSOR) {
          MOCHI_ASSERT(
              (std::is_same_v<MatType, BlockSparseMatrixView<T const, 3>>),
              "Per-actor colored SSOR preconditioner is only enabled for block sparse actors with block size of 3.");
          return std::unique_ptr<ActorPreconditioner<T>>{
              new ColoredSSORActorPrec<T, 3>{std::move(actorPseudoMatrix)}};
        } else if (precType == PreconditionerType::ILU0) {
          MOCHI_ASSERT(
              (std::is_same_v<MatType, BlockSparseMatrixView<T const, 4>>),
              "Per-actor ILU0 preconditioner is only enabled for block sparse actors with block size of 4.");
          return std::unique_ptr<ActorPreconditioner<T>>{
              new ILU0ActorPrec<T, 4>{std::move(actorPseudoMatrix)}};
        } else if (precType == PreconditionerType::Jacobi) {
          return std::unique_ptr<ActorPreconditioner<T>>{
              new BlockJacobiActorPrec<T, 1>{std::move(actorPseudoMatrix)}};
        } else [[unlikely]] {
          MOCHI_ASSERT(
              false,
              "Per-actor preconditioner type (%i) not supported.",
              static_cast<int>(precType));
          return std::unique_ptr<ActorPreconditioner<T>>{nullptr};
        }
      },
      actorMatrix);
}

} // namespace mochi
