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

/** @brief Execution model for concurrent application of an actor preconditioner. */
enum class ActorPreconditionerParallelMode {
  /** Apply the complete actor preconditioner serially on one worker. */
  SingleWorker,

  /** Apply disjoint actor-local row ranges without worker synchronization. */
  IndependentRows,

  /** Apply actor-local work using a dedicated synchronized worker team. */
  SynchronizedTeam,

  /** Number of execution modes. Not a valid execution mode. */
  Count,
};

/** @brief Concurrent application requirements for an actor preconditioner. */
struct ActorPreconditionerParallelism {
  /** @brief Execution mode. Must be set to a valid value before use. */
  ActorPreconditionerParallelMode mode = ActorPreconditionerParallelMode::Count;

  /** @brief Number of actor-local rows in each indivisible work block.
   *
   * @note Ignored for @ref ActorPreconditionerParallelMode::SingleWorker. For other modes, it must
   * be positive and divide the actor row count; every row range passed to @ref ConcurrentSolve
   * begins and ends on a block boundary.
   * @note For @ref ActorPreconditionerParallelMode::IndependentRows, every worker row boundary from
   * @ref GetRowRangesPerWorker that falls inside the actor must also lie on a block boundary.
   * Built-in actor preconditioners satisfy this.
   */
  int rowBlockSize = 1;
};

/** @brief Estimated cost of concurrently applying an actor preconditioner.
 *
 * @details @ref PerActorPrec uses these estimates to assign actor rows to workers so that all
 * workers finish each preconditioner application at about the same time. Estimates may change
 * after @ref ActorPreconditioner::Update.
 *
 * The costs use common FLOP-like work units. @ref PerActorPrec converts synchronization time to
 * the same units. For an actor containing @c numBlockRows aligned block rows, its estimated
 * duration on @c p workers is @c fixedCost+parallelCost*ceil(numBlockRows/p)/numBlockRows, plus
 * an estimated cost for each of the @ref numTeamBarriers barrier waits.
 *
 * @note @ref fixedCost and @ref parallelCost must be finite and nonnegative, and their sum must
 * be finite and positive; therefore, the all-zero default is invalid.
 * @note @ref ActorPreconditionerParallelMode::SingleWorker requires @ref maxUsefulWorkers to equal
 * one and @ref numTeamBarriers to equal zero.
 * @note @ref ActorPreconditionerParallelMode::IndependentRows requires @ref fixedCost to equal
 * zero and @ref numTeamBarriers to equal zero.
 */
struct ActorPreconditionerCost {
  /** Estimated work unaffected by worker count. Must be finite and nonnegative. */
  double fixedCost = 0.0;
  /** Estimated work divided among assigned workers. Must be finite and nonnegative. */
  double parallelCost = 0.0;
  /** Maximum structurally useful worker count. Must be positive. */
  int maxUsefulWorkers = 0;
  /** Team-wide barriers in one concurrent application. Must be nonnegative. */
  int numTeamBarriers = 0;
};

namespace details {

[[nodiscard]] constexpr double BlockMatVecCost(int numNonZeroBlocks, int blockSize) {
  // Use the conventional leading-order estimate of two FLOPs per matrix entry. The scheduler
  // needs consistent work units, not an exact arithmetic operation count.
  return 2.0 * static_cast<double>(numNonZeroBlocks) * blockSize * blockSize;
}

} // namespace details

/** @brief Abstract class for the preconditioner of an actor.
 *
 * REQUIREMENTS: All child classes must satisfy the following requirements:
 * - @ref ActorPreconditionerParallelMode::SingleWorker preconditioners are applied through @ref
 *   Solve by exactly one worker.
 * - In @ref ActorPreconditionerParallelMode::IndependentRows and
 *   @ref ActorPreconditionerParallelMode::SynchronizedTeam, `[data.rBegin, data.rEnd)` is the
 *   actor-local output range assigned to this call. Across one application, these ranges are
 *   disjoint and cover every actor row. A call may read any row of `x` but must write only this
 *   range of `Px`.
 * - In @ref ActorPreconditionerParallelMode::IndependentRows, `data.workerId` is the caller's index
 *   in the full solve worker group, `data.numWorkers` is the full group size, and `data.barrier`
 *   spans the full group. Not every group member is guaranteed to receive a call, so the barrier
 *   must not be used.
 * - In @ref ActorPreconditionerParallelMode::SynchronizedTeam, every member of the assigned team
 *   receives a call. `data.workerId` is the caller's zero-based team-local index,
 *   `data.numWorkers` is the team size, and `data.barrier` spans exactly that team. Every member
 *   must execute the same sequence of barrier waits. The preconditioner must support every team
 *   size from one through the minimum of @ref ActorPreconditionerCost::maxUsefulWorkers and the
 *   actor row count divided by @ref ActorPreconditionerParallelism::rowBlockSize.
 * - @ref ActorPreconditionerParallelism::rowBlockSize is ignored for
 *   @ref ActorPreconditionerParallelMode::SingleWorker. For other modes, it must be positive,
 *   divide the actor row count, and divide the start and end rows passed to @ref ConcurrentSolve.
 * - It must be safe to reuse the preconditioner across multiple linear solves. This implies all the
 *   data must either be owned by the preconditioner or be a reference/view to an object that will
 *   outlive the preconditioner.
 * - Workers may finish their preconditioner tasks at different times, but all remain in the
 *   enclosing concurrent solve. Actor preconditioner operations must not throw.
 * - An instance must not participate in multiple linear solves concurrently.
 */
template <typename T>
struct ActorPreconditioner {
  virtual ~ActorPreconditioner() = default;

  /** @brief Apply the preconditioner to a column vector. */
  virtual void Solve(ColumnVectorView<T const> x, ColumnVectorView<T> Px) const = 0;

  /** @brief Concurrent application of the preconditioner to a column vector by a pool of workers.
   * The calling worker is responsible for applying its own preconditioner contribution.
   *
   * @param[in] x Input column vector.
   * @param[out] Px Output column vector.
   * @param[in] data Parallel information for the calling worker.
   *
   * @note Not called for @ref ActorPreconditionerParallelMode::SingleWorker preconditioners.
   * Implementations selecting another mode must override this method.
   */
  virtual void ConcurrentSolve(
      ColumnVectorView<T const> /*x*/,
      ColumnVectorView<T> /*Px*/,
      ParallelWorkerInfo const& /*data*/) const {
    MOCHI_ASSERT(false, "Parallel solve not supported for this actor preconditioner.");
  }

  /** @brief Update the preconditioner's data with new matrix values.
   *
   * @note Matrix dimensions and relevant sparsity must remain unchanged. Recreate the
   * preconditioner if either changes.
   */
  virtual void Update(ActorPseudoMatrix<T> const& actorMatrix) = 0;

  /**
   * @brief Return the concrete preconditioner type for this actor.
   *
   * @note This is never @ref PreconditionerType::PerActor, which identifies the wrapper.
   */
  virtual constexpr PreconditionerType GetType() const = 0;

  /** @brief Get the concurrent-solve requirements.
   *
   * @return Requirements that remain constant for the lifetime of this preconditioner.
   *
   * @note Defaults to serial application by one worker.
   */
  [[nodiscard]] virtual constexpr ActorPreconditionerParallelism GetConcurrentSolveRequirements()
      const {
    return {ActorPreconditionerParallelMode::SingleWorker, 1};
  }

  /** @brief Return the estimated cost of a concurrent solve.
   *
   * @note Must run in constant time.
   */
  [[nodiscard]] virtual ActorPreconditionerCost GetConcurrentSolveCost() const = 0;
};

/**
 * @brief Block Jacobi actor preconditioner.
 *
 * @note Only valid for symmetric actors.
 */
template <typename T, int kBlockSize>
class BlockJacobiActorPrec : public ActorPreconditioner<T> {
 public:
  static_assert(kBlockSize > 0, "Preconditioner block size must be positive");

  explicit BlockJacobiActorPrec(ActorPseudoMatrix<T> const& A)
      : prec(A), _numBlockRows(A.Rows() / kBlockSize) {
    MOCHI_ASSERT_VERBOSE(
        A.Rows() > 0 && A.Rows() % kBlockSize == 0,
        "Actor row count must be positive and divisible by the block size.");
  }

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
    MOCHI_ASSERT_VERBOSE(
        actorMatrix.Rows() == _numBlockRows * kBlockSize, "Actor row count changed.");
    prec.Update(actorMatrix);
  }

  [[nodiscard]] constexpr ActorPreconditionerParallelism GetConcurrentSolveRequirements()
      const override {
    return {ActorPreconditionerParallelMode::IndependentRows, kBlockSize};
  }

  [[nodiscard]] ActorPreconditionerCost GetConcurrentSolveCost() const override {
    double const parallelCost = details::BlockMatVecCost(_numBlockRows, kBlockSize);
    return ActorPreconditionerCost{
        .fixedCost = 0.0,
        .parallelCost = parallelCost,
        .maxUsefulWorkers = _numBlockRows,
        .numTeamBarriers = 0};
  }

  constexpr PreconditionerType GetType() const override {
    return kBlockSize > 1 ? PreconditionerType::BlockJacobi : PreconditionerType::Jacobi;
  }

  krylov::BlockJacobiPrec<T, kBlockSize, /*kIsSymmetric*/ true> prec;

 private:
  int _numBlockRows;
};

/** @brief Symmetric inverse actor preconditioner. */
template <typename T>
class SymInverseActorPrec : public ActorPreconditioner<T> {
 public:
  explicit SymInverseActorPrec(ActorPseudoMatrix<T> const& A) : prec(A), _numRows(A.Rows()) {
    MOCHI_ASSERT_VERBOSE(_numRows > 0, "Actor row count must be positive.");
  }

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
    MOCHI_ASSERT_VERBOSE(A.Rows() == _numRows, "Actor row count changed.");
    prec.Update(A);
  }

  [[nodiscard]] constexpr ActorPreconditionerParallelism GetConcurrentSolveRequirements()
      const override {
    return {ActorPreconditionerParallelMode::IndependentRows, 1};
  }

  [[nodiscard]] ActorPreconditionerCost GetConcurrentSolveCost() const override {
    double const parallelCost =
        static_cast<double>(_numRows) * (2.0 * static_cast<double>(_numRows) - 1.0);
    return ActorPreconditionerCost{
        .fixedCost = 0.0,
        .parallelCost = parallelCost,
        .maxUsefulWorkers = _numRows,
        .numTeamBarriers = 0};
  }

  constexpr PreconditionerType GetType() const override {
    return PreconditionerType::SymInverse;
  }

  krylov::SymInversePrec<T> prec;

 private:
  int _numRows;
};

/**
 * @brief AMG actor preconditioner.
 *
 * @note Requires @ref ActorPseudoMatrix to be convertible to a block-sparse matrix with block size
 * @c kBlockSize.
 */
template <typename T, int kBlockSize>
class AMGActorPrec : public ActorPreconditioner<T> {
 public:
  static_assert(kBlockSize > 0, "Preconditioner block size must be positive");

  explicit AMGActorPrec(ActorPseudoMatrix<T> const& A) {
    MOCHI_ASSERT_VERBOSE(A.Rows() > 0, "Actor row count must be positive.");
    // GetConcurrentSolveCost() is calibrated for these settings.
    krylov::AMGOptions<T> const options{
        .smoother = krylov::Smoother::BlockJacobi,
        .numPreSmoothingSteps = 1,
        .numPostSmoothingSteps = 1,
    };
    Afine = ToBlockSparseMatrix<kBlockSize, MissingSparsityPolicy::AddAbsToDiagonal>(A);
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

  /** @brief Update numerical values using the existing sparsity structure. */
  void Update(ActorPseudoMatrix<T> const& actorMatrix) override {
#if MOCHI_ASSERT_VERBOSE_ENABLED
    // Reconstruct only for the verbose sparsity-pattern check.
    auto Anew =
        ToBlockSparseMatrix<kBlockSize, MissingSparsityPolicy::AddAbsToDiagonal>(actorMatrix);
    MOCHI_ASSERT_VERBOSE(Anew.Pointers() == Afine.Pointers(), "Sparsity pattern mismatch.");
    MOCHI_ASSERT_VERBOSE(Anew.Indices() == Afine.Indices(), "Sparsity pattern mismatch.");
#endif
    using BSpMatrixView = BlockSparseMatrixView<T const, kBlockSize>;
    MOCHI_ASSERT(
        std::holds_alternative<BSpMatrixView>(actorMatrix.actorMatrix),
        "Actor matrix type not supported for AMG actor preconditioner.");
    MOCHI_ASSERT_VERBOSE(actorMatrix.Rows() == actorMatrix.Cols(), "Expected square actor matrix.");

    auto const& bsp = std::get<BSpMatrixView>(actorMatrix.actorMatrix);
    auto srcValues = bsp.Values();
    std::copy(srcValues.begin(), srcValues.end(), Afine.Values().begin());
    details::AddInteractionToBlockSparseMatrix<MissingSparsityPolicy::AddAbsToDiagonal>(
        actorMatrix.interactionMatrices, Afine, actorMatrix.offset);
    prec->Update(Afine);
  }

  [[nodiscard]] constexpr ActorPreconditionerParallelism GetConcurrentSolveRequirements()
      const override {
    return {ActorPreconditionerParallelMode::SynchronizedTeam, kBlockSize};
  }

  [[nodiscard]] ActorPreconditionerCost GetConcurrentSolveCost() const override {
    int const numBlockRows = Afine.BlockRows();
    double const fineMatVecCost =
        details::BlockMatVecCost(static_cast<int>(Afine.NumNonZeroBlocks()), kBlockSize);
    // Across four mesh resolutions, AMG work is 2.20-2.23x fine matvecs for soft meshes
    // and 1.62-1.64x for shell meshes. Worker-0-only shares are 6-7% and 3-4%, respectively.
    // Scale the (conservative) soft case to the observed ~3x runtime of AMG solve w.r.t. matvec:
    // 0.2 fixed and 2.8 parallel matvec-equivalents.
    // TODO(T290274539): Derive the costs from the hierarchy instead of fixed multiples of the
    // fine-level cost. Only worker 0 runs the levels below the first coarsening, so fixedCost is
    // their work, which varies across meshes. Also model the cache-line transfers after each
    // barrier: workers read fine-level rows that other workers wrote near their range boundaries,
    // and worker 0 gathers the first coarse-level vector from all workers. These grow with the
    // worker count, so small AMGs otherwise get more workers than pay off. See D121889727 and
    // D121889728.
    double const fixedCost = 0.2 * fineMatVecCost;
    double const parallelCost = 2.8 * fineMatVecCost;
    return ActorPreconditionerCost{
        .fixedCost = fixedCost,
        .parallelCost = parallelCost,
        .maxUsefulWorkers = numBlockRows,
        .numTeamBarriers = prec->NumConcurrentSolveBarriers()};
  }

  constexpr PreconditionerType GetType() const override {
    return PreconditionerType::AMG;
  }

  BlockSparseMatrix<T, kBlockSize> Afine;
  std::unique_ptr<krylov::AMGPrec<T, kBlockSize>> prec;
};

/**
 * @brief Colored SSOR actor preconditioner.
 *
 * @note Requires @ref ActorPseudoMatrix to be convertible to a block-sparse matrix with block size
 * @c kBlockSize.
 */
template <typename T, int kBlockSize>
class ColoredSSORActorPrec : public ActorPreconditioner<T> {
 public:
  static_assert(kBlockSize > 0, "Preconditioner block size must be positive");

  explicit ColoredSSORActorPrec(ActorPseudoMatrix<T> const& A_) {
    MOCHI_ASSERT_VERBOSE(A_.Rows() > 0, "Actor row count must be positive.");
    // With symmetric values and a strictly positive diagonal, colored SSOR with the default omega =
    // 1 is SPD even if A is indefinite. Diagonal compensation is therefore unnecessary.
    A = ToBlockSparseMatrix<kBlockSize, MissingSparsityPolicy::Discard>(A_);
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

  /** @brief Update numerical values using the existing sparsity structure. */
  void Update(ActorPseudoMatrix<T> const& actorMatrix) override {
#if MOCHI_ASSERT_VERBOSE_ENABLED
    // Reconstruct only for the verbose sparsity-pattern check.
    auto Anew = ToBlockSparseMatrix<kBlockSize, MissingSparsityPolicy::Discard>(actorMatrix);
    MOCHI_ASSERT_VERBOSE(Anew.Pointers() == A.Pointers(), "Sparsity pattern mismatch.");
    MOCHI_ASSERT_VERBOSE(Anew.Indices() == A.Indices(), "Sparsity pattern mismatch.");
#endif
    using BSpMatrixView = BlockSparseMatrixView<T const, kBlockSize>;
    MOCHI_ASSERT(
        std::holds_alternative<BSpMatrixView>(actorMatrix.actorMatrix),
        "Actor matrix type not supported for colored SSOR actor preconditioner.");
    MOCHI_ASSERT_VERBOSE(actorMatrix.Rows() == actorMatrix.Cols(), "Expected square actor matrix.");

    auto const& bsp = std::get<BSpMatrixView>(actorMatrix.actorMatrix);
    auto srcValues = bsp.Values();
    std::copy(srcValues.begin(), srcValues.end(), A.Values().begin());
    details::AddInteractionToBlockSparseMatrix<MissingSparsityPolicy::Discard>(
        actorMatrix.interactionMatrices, A, actorMatrix.offset);
    prec->Update(A);
  }

  [[nodiscard]] constexpr ActorPreconditionerParallelism GetConcurrentSolveRequirements()
      const override {
    return {ActorPreconditionerParallelMode::SynchronizedTeam, kBlockSize};
  }

  [[nodiscard]] ActorPreconditionerCost GetConcurrentSolveCost() const override {
    int const numBlockRows = A.BlockRows();
    auto const numNonZeroBlocks = static_cast<int>(A.NumNonZeroBlocks());
    double const totalCost =
        details::BlockMatVecCost(numNonZeroBlocks, kBlockSize) + 2.0 * A.Rows();
    int const numColors = prec->NumColors();
    MOCHI_ASSERT_VERBOSE(
        numColors > 0 && numColors <= numBlockRows, "Invalid colored SSOR color count.");

    // Colors execute sequentially, while rows within each color execute in parallel. Approximate
    // the critical path as one average block row per color, assuming work is balanced across rows.
    // Team-barrier costs are modeled separately.
    double const fixedFraction = static_cast<double>(numColors) / static_cast<double>(numBlockRows);
    double const fixedCost = totalCost * fixedFraction;
    double const parallelCost = totalCost - fixedCost;
    return ActorPreconditionerCost{
        .fixedCost = fixedCost,
        .parallelCost = parallelCost,
        .maxUsefulWorkers = prec->MaxBlockRowsPerColor(),
        .numTeamBarriers = prec->NumConcurrentSolveBarriers()};
  }

  constexpr PreconditionerType GetType() const override {
    return PreconditionerType::ColoredSSOR;
  }

  BlockSparseMatrix<T, kBlockSize> A;
  std::unique_ptr<krylov::ColoredSSORPrec<BlockSparseMatrix<T, kBlockSize>>> prec;
};

/**
 * @brief ILU0 actor preconditioner.
 *
 * @note Requires @ref ActorPseudoMatrix to be convertible to a block-sparse matrix with block size
 * @c kBlockSize.
 */
template <typename T, int kBlockSize>
class ILU0ActorPrec : public ActorPreconditioner<T> {
 public:
  static_assert(kBlockSize > 0, "Preconditioner block size must be positive");

  explicit ILU0ActorPrec(ActorPseudoMatrix<T> const& A_) {
    MOCHI_ASSERT_VERBOSE(A_.Rows() > 0, "Actor row count must be positive.");
    A = ToBlockSparseMatrix<kBlockSize, MissingSparsityPolicy::Discard>(A_);
    prec = std::make_unique<krylov::RelaxedILUPrec<BlockSparseMatrix<T, kBlockSize>>>(
        A, /*fillInLevel*/ 0, /*alphaRelax*/ T{0});
  }

  void Solve(ColumnVectorView<T const> x, ColumnVectorView<T> Px) const override {
    prec->Solve(x, Px);
  }

  /** @brief Update numerical values using the existing sparsity structure. */
  void Update(ActorPseudoMatrix<T> const& actorMatrix) override {
#if MOCHI_ASSERT_VERBOSE_ENABLED
    // Reconstruct only for the verbose sparsity-pattern check.
    auto Anew = ToBlockSparseMatrix<kBlockSize, MissingSparsityPolicy::Discard>(actorMatrix);
    MOCHI_ASSERT_VERBOSE(Anew.Pointers() == A.Pointers(), "Sparsity pattern mismatch.");
    MOCHI_ASSERT_VERBOSE(Anew.Indices() == A.Indices(), "Sparsity pattern mismatch.");
#endif
    using BSpMatrixView = BlockSparseMatrixView<T const, kBlockSize>;
    MOCHI_ASSERT(
        std::holds_alternative<BSpMatrixView>(actorMatrix.actorMatrix),
        "Actor matrix type not supported for ILU0 actor preconditioner.");
    MOCHI_ASSERT_VERBOSE(actorMatrix.Rows() == actorMatrix.Cols(), "Expected square actor matrix.");

    auto const& bsp = std::get<BSpMatrixView>(actorMatrix.actorMatrix);
    auto srcValues = bsp.Values();
    std::copy(srcValues.begin(), srcValues.end(), A.Values().begin());
    details::AddInteractionToBlockSparseMatrix<MissingSparsityPolicy::Discard>(
        actorMatrix.interactionMatrices, A, actorMatrix.offset);
    prec->Update(A);
  }

  [[nodiscard]] ActorPreconditionerCost GetConcurrentSolveCost() const override {
    double const solveCost =
        details::BlockMatVecCost(static_cast<int>(A.NumNonZeroBlocks()), kBlockSize) + A.Rows();
    return ActorPreconditionerCost{
        .fixedCost = solveCost, .parallelCost = 0.0, .maxUsefulWorkers = 1, .numTeamBarriers = 0};
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
