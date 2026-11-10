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

#include <mochi_core/linear_algebra/block_sparse_matrix.h>
#include <mochi_core/linear_algebra/krylov/amg/amg_prec.h>
#include <mochi_core/linear_algebra/krylov/amg/coarsening.h>
#include <mochi_core/linear_algebra/krylov/block_jacobi_prec.h>
#include <mochi_core/linear_algebra/krylov/preconditioner.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/matrix_operations.h>
#include <mochi_core/linear_algebra/utils/matrix_conversions.h>
#include <mochi_core/mochi_config.h>
#include <mochi_core/utils/array_utils.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/container_utils.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/graph.h>
#include <mochi_core/utils/graph_utils.h>
#include <mochi_core/utils/graph_views.h>
#include <mochi_core/utils/log.h>
#include <mochi_core/utils/simd.h>
#include <mochi_core/utils/span.h>
#include <mochi_core/utils/task_scheduler.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>

namespace mochi::krylov {

namespace details {
template <typename Scalar>
struct RigidModeTransfer;

template <typename Scalar>
struct CheckedBlockJacobi6;
} // namespace details

/**
 * @brief Smoothed-aggregation AMG preconditioner whose coarse levels capture the rigid body modes
 * of each aggregate: three translations and three rotations.
 *
 * @details For matrices with 3 DoFs per node and node positions, e.g., soft and shell actors. The
 * aggregates of each level become the nodes of the next, each with 6 DoFs: a translation t and a
 * rotation θ stored as φ = ℓ θ, where ℓ is the aggregate's RMS radius, over its members' distances
 * to its centroid and their own radii. A fine node i follows aggregate a by its rigid motion, so
 * the interpolation block of weight w is [w I | -[q]x] at level 0 and [[w I, -[q]x], [0, r I]] at
 * coarser levels, with q = w (x_i - c_a) / ℓ_a and r = w ℓ_i / ℓ_a, where c_a is the aggregate's
 * centroid, and x_i and ℓ_i are the fine node's position and radius. Without decoupling (below),
 * the interpolation reproduces all six rigid motions at every level in exact arithmetic.
 *
 * - Aggregation and weights: the pattern aggregation and smoothed weights of @ref AMGPrec. At level
 *   0, weights below @ref kTruncation are dropped and each row is renormalized. The pure rows of an
 *   aggregate are those that interpolate from it alone. The members of every aggregate whose pure
 *   rows have a margin (@ref details::RigidModeMarginAtLeast) below @ref kMemberOnlyMargin at the
 *   reference positions interpolate only from it, with weight 1, a decision made at construction.
 * - Degenerate positions: when the pure rows of a level-0 aggregate are nearly collinear at the
 *   current positions (see @ref DecoupleMargin), its rotation DoFs are decoupled. They do not
 *   interpolate, and their diagonal block of the first coarse matrix is J, the identity times the
 *   mean diagonal entry of the block's translation part in P^T A_0 P, so that A_1 = P^T A_0 P + J,
 *   and the interpolation reproduces only the aggregate's translations. With this, in exact
 *   arithmetic, no coarse matrix is singular for an SPD A and any positions. The coarser
 *   interpolations need no such tests: each aggregate's root, whose neighbors all belong to it
 *   (@ref details::Aggregate), interpolates from it alone with weight 1 and an invertible block
 *   (r > 0). Since J scales with A, the preconditioner of α A is that of A divided by α.
 * - Numerical acceptance: each update checks the coarse levels from the first. The smoother blocks
 *   of a smoothed level, the coarsest matrix, and the inverses the cycle applies must be positive
 *   definite to working precision (@ref details::IsPositiveDefinite). A smoothed level that fails
 *   is stiffened: a bound on its Galerkin product's rounding error is added to its diagonal (@ref
 *   details::AddGalerkinRoundingBound). It is kept if it is then finite and passes, and the coarser
 *   levels are built from it. A coarsest level that fails is solved on its translation block alone,
 *   with zero rotation correction, if it is finite and that block passes. Either repair also needs
 *   the cached P^T A that restricts to the level, if any, to be finite. A level that still fails is
 *   dropped with the coarser ones, with a logged warning, and the level above smooths without
 *   coarse correction. This covers rounding, which the geometric tests cannot; the cycle is then
 *   positive definite if the damping keeps every smoother convergent
 *   (@ref AMGOptions::relaxationFactor).
 *
 *   The checks certify only what keeps the cycle positive definite: positive definite smoother
 *   blocks and a positive semidefinite coarsest correction. The stiffening certifies that a level
 *   dominates the exact product of its stored finer level and interpolation, under the assumptions
 *   of @ref details::AddGalerkinRoundingBound: an exactly symmetric finer level, finite arithmetic
 *   rounded to nearest, no subnormal numbers and bounded sum lengths. None certifies accuracy. A
 *   smoothed level needs no more: its smoother is damped, the coarser levels correct what it
 *   misses, and when the exact product is positive definite, the exact inverse of the stiffened
 *   level is bounded above by that product's exact inverse, so its correction never exceeds the
 *   exact product's. The coarsest level is not stiffened: it is inverted explicitly, with nothing
 *   coarser to correct it, and when its finer level is indefinite, which the smoother checks allow,
 *   the stiffening need not make it positive definite, so a stiffened coarsest could be worse than
 *   none. Its translations are the fallback because stiff point terms, such as pins and contact,
 *   cancel in floating point in the rotations about each aggregate's centroid but not in the
 *   translations. Stiff springs within an aggregate can still cancel in them; if the translation
 *   block then fails too, dropping the level is the safe choice. Even when it passes, translations
 *   can be a poor coarse space: where stiff contact holds an aggregate on a curved surface, its
 *   soft motion along the curve is a rotation, which translations can't reproduce, so the fallback
 *   can take more iterations than dropping the level.
 * - Cycle: a V(1,1) cycle with damped block Jacobi smoothing, the automatic damping of @ref
 *   AMGOptions, restriction of the fine residual, and the explicit inverse of the coarsest matrix,
 *   or of its translation block. The coarse levels run on the first worker of @ref ConcurrentSolve.
 * - Every coarse matrix is stored exactly symmetric.
 *
 * @warning The matrix storage passed to the constructor or latest @ref Update call must outlive
 * this preconditioner or the next @ref Update call.
 */
template <typename Scalar>
struct RigidBodyModesAMGPrec : Preconditioner<Scalar> {
  static_assert(!std::is_const_v<Scalar>, "Implementation assumes Scalar is non-const.");
  static constexpr auto kType = PreconditionerType::AMG;

  using BSpMatrixView = BlockSparseMatrixView<Scalar const, 3>;

  /**
   * @brief Build the preconditioner.
   *
   * @param[in] A SPD matrix, with a symmetric pattern that includes the diagonal blocks.
   * @param[in] referencePositions Reference node positions, 3 per block row. They decide the
   * level-0 weights.
   * @param[in] displacements Node displacements from the reference positions, 3 per block row.
   *
   * @note Positions are in the axes of A's DoFs, with any origin and length unit. Both spans are
   * read only during the call.
   */
  RigidBodyModesAMGPrec(
      BSpMatrixView A,
      Span<Scalar const> referencePositions,
      Span<Scalar const> displacements);

  /**
   * @brief Update the preconditioner. Equivalent to constructing it from the same arguments.
   *
   * @param[in] A SPD matrix, with the pattern of the matrix passed to the constructor.
   * @param[in] referencePositions The reference positions passed to the constructor.
   * @param[in] displacements Node displacements from the reference positions, 3 per block row.
   *
   * @note The interpolation coefficients are rebuilt only if the positions changed.
   */
  void
  Update(BSpMatrixView A, Span<Scalar const> referencePositions, Span<Scalar const> displacements);

  void Solve(ColumnVectorView<Scalar const> b, ColumnVectorView<Scalar> x) const override;

  /// @brief Concurrent application by a team of workers, which wait @ref NumConcurrentSolveBarriers
  /// times on their barrier.
  ///
  /// @note Row ranges must begin and end at multiples of 3. Consecutive calls on one instance must
  /// be separated by a synchronization of the whole team, such as the dot products of PCG.
  void ConcurrentSolve(
      ColumnVectorView<Scalar const> b,
      ColumnVectorView<Scalar> x,
      ParallelWorkerInfo const& data) const override;

  constexpr PreconditionerType GetType() const override {
    return kType;
  }

  /// @brief Return how many times each worker waits on its barrier in one @ref ConcurrentSolve
  /// call: 5, or 1 when the latest update accepted no coarse level.
  [[nodiscard]] int NumConcurrentSolveBarriers() const {
    return _numAcceptedCoarseLevels > 0 ? 5 : 1;
  }

 protected:
  using Transfer = details::RigidModeTransfer<Scalar>;
  using CoarseMatrix = BlockSparseMatrix<Scalar, 6, int, int>;

  /// @brief Weight of the neighbors in the smoothed aggregate weights, as in @ref AMGPrec.
  static constexpr double kSmoothingWeight = AMGOptions<double>{}.prolongationSmoothingWeight;

  /// @brief Level-0 weights below this value are dropped.
  static constexpr double kTruncation = 0.15;
  static_assert(
      kTruncation < 1.0 - kSmoothingWeight,
      "A row's weight on its own aggregate, at least 1 - kSmoothingWeight, must be kept.");

  /// @brief Level-0 reference-position margin below which an aggregate's members interpolate only
  /// from it. Since the margin is at most σ_min / σ_max (@ref details::RigidModeMarginAtLeast),
  /// every aggregate whose pure rows have σ_min / σ_max below it switches.
  static constexpr double kMemberOnlyMargin = 0.1;

  /// @brief Level-0 margin below which an aggregate's rotation DoFs are decoupled: the weakest unit
  /// combination of its six rigid-mode coordinates then interpolates its pure rows with a squared
  /// norm below 100 epsilon times a bound on the strongest one's (@ref
  /// details::RigidModeMarginAtLeast). The acceptance checks on the coarse matrices and their
  /// inverses cover stiffness weighting and other rounding.
  [[nodiscard]] static double DecoupleMargin() {
    return 10.0 * Sqrt(static_cast<double>(std::numeric_limits<Scalar>::epsilon()));
  }

  /// @brief Coarse level s + 1, for s = 0, 1, ...: its interpolation to level s, its matrix, and
  /// the state of its smoothing and of the cycle.
  struct CoarseLevel {
    /// @brief Interpolation P from this level to the next finer one.
    Transfer transfer;
    /// @brief The level's matrix.
    CoarseMatrix A;
    /// @brief P^T A for the interpolation P from the next coarser level. Empty at the coarsest
    /// level.
    CoarseMatrix PtA;

    /// @brief Smoother. Unused at the coarsest level.
    details::CheckedBlockJacobi6<Scalar> smoother;
    Scalar relaxationFactor = 0;

    /// @brief Right-hand side and solution of the cycle, which @ref ConcurrentSolve overwrites.
    mutable ColumnVector<Scalar> b;
    mutable ColumnVector<Scalar> x;
    /// @brief Temporaries of the cycle. Empty at the coarsest level.
    mutable ColumnVector<Scalar> u;
    mutable ColumnVector<Scalar> r;
  };

  /// @brief Set the level-0 weights described in the class documentation.
  ///
  /// @param[in] A Fine matrix.
  /// @param[in] partition Aggregate of each fine node.
  /// @param[in] referencePositions Reference positions, 3 per node.
  /// @param[in,out] T Level-0 transfer with its aggregates.
  static void SetLevel0Weights(
      BSpMatrixView A,
      DynamicArray<int> const& partition,
      Span<double const> referencePositions,
      Transfer& T);

  /// @brief Build the aggregates, the level-0 weights, the interpolation patterns, the coarse
  /// matrix patterns and the vectors of the cycle.
  void BuildHierarchy(Span<double const> referencePositions);

  /// @brief Cache x = reference + displacement and return whether it changed, bit for bit.
  bool SetPositions(Span<Scalar const> referencePositions, Span<Scalar const> displacements);

  /// @brief Rebuild the aggregates' geometry, the decoupled aggregates and the interpolation
  /// coefficients from the positions.
  void UpdateInterpolation();

  /// @brief Recompute the coarse matrices, the accepted coarse levels, their smoothers, the damping
  /// and the coarsest inverse.
  void UpdateNumerics();

  /// @brief V-cycle at coarse level s + 1 on a single worker, from and to its vectors b and x.
  void CoarseCycle(int s) const;

  BSpMatrixView _A0;
  BlockJacobiPrec<Scalar, 3, /*kIsSymmetric*/ true> _fineSmoother;
  Scalar _fineRelaxationFactor = 0;

  /// @brief Random RHS of the damping estimates, drawn with the larger of the fine and first coarse
  /// sizes. Each level uses its leading entries.
  ColumnVector<Scalar> _spectralRadiusRhs;

  /// @brief Pattern of P^T A_0 for the interpolation P from level 1.
  Graph<int, int> _PtA0Pattern = {};
  /// @brief Level-0 aggregates whose rotation DoFs are decoupled.
  DynamicArray<bool> _decoupled = {};
  /// @brief Cached node positions x = reference + displacement.
  DynamicArray<double> _positions = {};

  DynamicArray<CoarseLevel> _levels = {};
  /// @brief Number of coarse levels, from the first, that the latest update accepted.
  int _numAcceptedCoarseLevels = 0;
  /// @brief Inverse of the coarsest matrix, or of its translation block, padded with zero rotation
  /// rows and columns.
  Matrix<Scalar> _coarsestInverse;

  /// @brief Fine-level temporaries of the cycle, which @ref ConcurrentSolve overwrites.
  mutable ColumnVector<Scalar> _fineU;
  mutable ColumnVector<Scalar> _fineR;
};

namespace details {

/// @brief Interpolation coefficients of one (fine node, aggregate) pair. See @ref
/// RigidBodyModesAMGPrec.
template <typename Scalar>
struct RigidModeCoefficients {
  Scalar w;
  Scalar q[3];
  Scalar r;
};

/// @brief Interpolation P from a coarse level, with 6 DoFs per node, to the next finer level, whose
/// aggregates its nodes are.
template <typename Scalar>
struct RigidModeTransfer {
  int numNodes = 0;
  int numAggregates = 0;
  /// @brief Members of each aggregate, in increasing order.
  Graph<int, int> members;
  /// @brief Interpolation by fine node, in CSR format, with aggregates in increasing order: the
  /// weights, in double, and the coefficients computed from them.
  DynamicArray<int> pointers;
  DynamicArray<int> aggregates;
  DynamicArray<double> weights;
  DynamicArray<RigidModeCoefficients<Scalar>> coefficients;
  /// @brief Restriction by aggregate, in CSR format, with fine nodes in increasing order: the index
  /// of each entry in @ref coefficients, and a copy of the coefficients in this order.
  DynamicArray<int> restrictionPointers;
  DynamicArray<int> restrictionNodes;
  DynamicArray<int> restrictionEntries;
  DynamicArray<RigidModeCoefficients<Scalar>> restrictionCoefficients;
  /// @brief Centroid (3 coordinates) and RMS radius of each aggregate.
  DynamicArray<double> centroids;
  DynamicArray<double> radii;
};

// TODO: Merge FactorLDLt and kernel::Factor into a single kernel at least as fast as both, and use
// it everywhere.
/// @brief Factor a symmetric matrix in place as L D L^T, with L unit lower triangular: L replaces
/// the strict lower triangle and D^{-1} the diagonal. Reads the lower triangle.
///
/// @param[in,out] a Row-major n x n matrix.
/// @param[in] n Size.
/// @return Whether the matrix is positive definite to working precision, as for @ref
/// IsPositiveDefinite. Fails for NaN.
///
/// @note Several times faster than @ref IsPositiveDefinite on 6x6 blocks, and slower on the
/// coarsest matrix.
template <typename Scalar>
[[nodiscard]] bool FactorLDLt(Scalar* a, int n) {
  Scalar const margin = Scalar(n) * std::numeric_limits<Scalar>::epsilon();
  for (int k = 0; k < n; ++k) {
    Scalar* rowK = a + k * n;
    for (int j = 0; j < k; ++j) {
      Scalar const* rowJ = a + j * n;
      Scalar sum = rowK[j];
      for (int m = 0; m < j; ++m) {
        sum -= rowK[m] * a[m * n + m] * rowJ[m];
      }
      rowK[j] = sum / rowJ[j];
    }
    Scalar pivot = rowK[k];
    for (int j = 0; j < k; ++j) {
      pivot -= rowK[j] * rowK[j] * a[j * n + j];
    }
    if (!(pivot > margin * rowK[k])) {
      return false;
    }
    rowK[k] = pivot;
  }

  for (int k = 0; k < n; ++k) {
    a[k * n + k] = Scalar(1) / a[k * n + k];
  }
  return true;
}

/// @brief The inverse Y^T D^{-1} Y, with Y = L^{-1}, of a 6x6 matrix from its row-major factors by
/// @ref FactorLDLt, as a row-major matrix.
template <typename Scalar>
void InverseFromLDLt6(Scalar const* factors, Scalar* inverse) {
  Scalar y[6][6] = {};
  for (int j = 0; j < 6; ++j) {
    y[j][j] = Scalar(1);
    for (int i = j + 1; i < 6; ++i) {
      Scalar sum = -factors[6 * i + j];
      for (int k = j + 1; k < i; ++k) {
        sum -= factors[6 * i + k] * y[k][j];
      }
      y[i][j] = sum;
    }
  }

  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j <= i; ++j) {
      Scalar sum = 0;
      for (int k = i; k < 6; ++k) {
        sum += y[k][i] * factors[6 * k + k] * y[k][j];
      }
      inverse[6 * i + j] = sum;
      inverse[6 * j + i] = sum;
    }
  }
}

/// @brief Whether the margin σ_min / sqrt(λ) of the rigid-mode matrix B of points, the stack of the
/// 3x6 blocks [I | -[q_i]x], is at least @p margin, from its sums, where λ = λ_max([[n, |s|], [|s|,
/// Σ|q_i|^2]]) for n = @p count and s = Σq_i. Fails for fewer than 3 points and for NaN.
///
/// @details A unit rigid motion (t, φ) moves q_i by t + φ × q_i, so Cauchy-Schwarz gives
/// σ_max^2 <= λ: the margin is at most σ_min / σ_max. With G = B^T B, the margin is at least
/// @p margin exactly when G - δ I is positive semidefinite, for δ = @p margin^2 λ. The test checks
/// that G - δ I is positive definite to working precision (@ref FactorLDLt) through the 3x3 Schur
/// complement of its translation block (@p count - δ) I. In exact arithmetic, the margin is 0
/// exactly when the points are collinear.
///
/// @param[in] count Number of points.
/// @param[in] sum Sum of the q_i.
/// @param[in] sumOuter Upper triangle of the sum of q_i q_i^T: xx, xy, xz, yy, yz, zz.
/// @param[in] margin Required margin, positive.
[[nodiscard]] inline bool RigidModeMarginAtLeast(
    int count,
    double const (&sum)[3],
    double const (&sumOuter)[6],
    double margin) {
  if (count < 3) {
    return false;
  }

  // G = [[count I, -[sum]x], [[sum]x, trace I - sumOuter]], the sum of the blocks' Gram matrices.
  double const trace = sumOuter[0] + sumOuter[3] + sumOuter[5];
  double const sumSq = Sqr(sum[0]) + Sqr(sum[1]) + Sqr(sum[2]);
  double const lambda = 0.5 * (count + trace + Sqrt(Sqr(count - trace) + 4.0 * sumSq));
  double const shift = Sqr(margin) * lambda;
  double const translation = count - shift;
  if (!(translation > 0.0)) {
    return false;
  }

  double const invTranslation = 1.0 / translation;
  double schur[3][3] = {
      {trace - sumOuter[0] - shift - (Sqr(sum[1]) + Sqr(sum[2])) * invTranslation, 0.0, 0.0},
      {-sumOuter[1] + sum[0] * sum[1] * invTranslation,
       trace - sumOuter[3] - shift - (Sqr(sum[0]) + Sqr(sum[2])) * invTranslation,
       0.0},
      {-sumOuter[2] + sum[0] * sum[2] * invTranslation,
       -sumOuter[4] + sum[1] * sum[2] * invTranslation,
       trace - sumOuter[5] - shift - (Sqr(sum[0]) + Sqr(sum[1])) * invTranslation}};
  return FactorLDLt(&schur[0][0], 3);
}

/// @brief Whether the margin of the rigid-mode matrix of an aggregate's pure rows, the rows that
/// interpolate from it alone (@ref RigidModeMarginAtLeast), is at least @p margin, with q_i =
/// (x_i - c) / ℓ for the aggregate's centroid c and RMS radius ℓ. Fails for NaN.
///
/// @details With at least 3 pure rows that are not collinear, the aggregate's 6 columns of the
/// interpolation have full rank, and no other aggregate's columns can cancel them. The margin is
/// invariant to translations, rotations and uniform scaling of the positions.
///
/// @param[in] T Transfer with the aggregate's geometry.
/// @param[in] a Aggregate.
/// @param[in] positions Fine node positions, 3 per node.
/// @param[in] margin Required margin.
template <typename Scalar>
[[nodiscard]] bool PureRowMarginAtLeast(
    RigidModeTransfer<Scalar> const& T,
    int a,
    Span<double const> positions,
    double margin) {
  double const* centroid = &T.centroids[3 * a];
  double const invRadius = 1.0 / T.radii[a];
  int count = 0;
  double sum[3] = {0.0, 0.0, 0.0};
  double sumOuter[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  for (int const i : T.members[a]) {
    if (T.pointers[i + 1] - T.pointers[i] != 1) {
      continue;
    }
    double q[3];
    for (int d = 0; d < 3; ++d) {
      q[d] = (positions[3 * i + d] - centroid[d]) * invRadius;
      sum[d] += q[d];
    }
    sumOuter[0] += q[0] * q[0];
    sumOuter[1] += q[0] * q[1];
    sumOuter[2] += q[0] * q[2];
    sumOuter[3] += q[1] * q[1];
    sumOuter[4] += q[1] * q[2];
    sumOuter[5] += q[2] * q[2];
    ++count;
  }
  return RigidModeMarginAtLeast(count, sum, sumOuter, margin);
}

/// @brief Minimum rows per task, for @p numRows rows that hold @p numItems items, so that a task
/// gets about @p itemsPerTask items.
[[nodiscard]] inline int RowsPerTask(int numRows, int64_t numItems, int64_t itemsPerTask) {
  return Clamp(
      static_cast<int>((itemsPerTask * static_cast<int64_t>(numRows)) / Max<int64_t>(1, numItems)),
      1,
      Max(1, numRows));
}

/// @brief Set the aggregates of a transfer, and their members, from a partition of the fine nodes.
template <typename Scalar>
void SetAggregates(
    DynamicArray<int> const& partition,
    int numAggregates,
    RigidModeTransfer<Scalar>& T) {
  T.numNodes = isize(partition);
  T.numAggregates = numAggregates;
  T.members = GraphFromAssignments<int, int>(partition);
  MOCHI_ASSERT_VERBOSE(
      T.members.size() == numAggregates, "Expected one member list per aggregate.");
}

/// @brief Set the weights of a transfer to the smoothed aggregate weights of @ref AMGPrec (@ref
/// Smoothing), in double.
///
/// @param[in] A Matrix of the fine nodes, whose pattern defines their neighbors.
/// @param[in] partition Aggregate of each fine node.
/// @param[in] smoothingWeight Weight of the neighbors.
/// @param[in,out] T Transfer with its aggregates.
template <typename MatrixType, typename Scalar>
void SetSmoothedWeights(
    MatrixType const& A,
    DynamicArray<int> const& partition,
    double smoothingWeight,
    RigidModeTransfer<Scalar>& T) {
  auto const W = Smoothing(A, T.numAggregates, partition, smoothingWeight);
  T.pointers = DynamicArray<int>(W.Pointers().begin(), W.Pointers().end());
  T.aggregates = DynamicArray<int>(W.Indices().begin(), W.Indices().end());
  T.weights = DynamicArray<double>(W.Values().begin(), W.Values().end());
}

/// @brief Keep only the weights for which @p keep(fine node, aggregate, weight) holds, renormalized
/// so that each row sums to one.
template <typename KeepFn, typename Scalar>
void FilterWeights(KeepFn const& keep, RigidModeTransfer<Scalar>& T) {
  int numKept = 0;
  for (int i = 0; i < T.numNodes; ++i) {
    int const rowBegin = T.pointers[i];
    int const rowEnd = T.pointers[i + 1];
    int const keptBegin = numKept;
    double sum = 0.0;
    for (int k = rowBegin; k < rowEnd; ++k) {
      if (keep(i, T.aggregates[k], T.weights[k])) {
        T.aggregates[numKept] = T.aggregates[k];
        T.weights[numKept] = T.weights[k];
        sum += T.weights[k];
        ++numKept;
      }
    }
    MOCHI_ASSERT_VERBOSE(numKept > keptBegin, "Every row must keep a weight.");
    for (int k = keptBegin; k < numKept; ++k) {
      T.weights[k] /= sum;
    }
    T.pointers[i] = keptBegin;
  }

  T.pointers[T.numNodes] = numKept;
  T.aggregates.resize(numKept);
  T.weights.resize(numKept);
}

/// @brief Set the restriction pattern of a transfer from its interpolation pattern.
template <typename Scalar>
void SetRestrictionPattern(RigidModeTransfer<Scalar>& T) {
  int const numEntries = isize(T.aggregates);
  T.restrictionPointers = DynamicArray<int>(T.numAggregates + 1, 0);
  for (int const a : T.aggregates) {
    ++T.restrictionPointers[a + 1];
  }

  for (int a = 0; a < T.numAggregates; ++a) {
    T.restrictionPointers[a + 1] += T.restrictionPointers[a];
  }

  T.restrictionNodes.resize_noinit(numEntries);
  T.restrictionEntries.resize_noinit(numEntries);
  DynamicArray<int> next(T.restrictionPointers.begin(), T.restrictionPointers.end() - 1);
  for (int i = 0; i < T.numNodes; ++i) {
    for (int k = T.pointers[i]; k < T.pointers[i + 1]; ++k) {
      int const slot = next[T.aggregates[k]]++;
      T.restrictionNodes[slot] = i;
      T.restrictionEntries[slot] = k;
    }
  }
}

/// @brief Graph of a transfer's interpolation pattern: from each fine node to its aggregates.
template <typename Scalar>
auto InterpolationGraph(RigidModeTransfer<Scalar> const& T) {
  return Graph<int const, int const, mochi::details::SpanCI>(T.pointers, T.aggregates);
}

/// @brief Graph of a transfer's restriction pattern: from each aggregate to its fine nodes.
template <typename Scalar>
auto RestrictionGraph(RigidModeTransfer<Scalar> const& T) {
  return Graph<int const, int const, mochi::details::SpanCI>(
      T.restrictionPointers, T.restrictionNodes);
}

/// @brief Set the centroid and RMS radius of each aggregate of a transfer from its members'
/// positions and radii. A radius that is not positive and finite is replaced by 1.
///
/// @param[in] positions Fine node positions, 3 per node.
/// @param[in] radii Fine node radii, or empty if they are zero (level 0).
/// @param[in,out] T Transfer.
template <typename Scalar>
void SetAggregateGeometry(
    Span<double const> positions,
    Span<double const> radii,
    RigidModeTransfer<Scalar>& T) {
  T.centroids.resize_noinit(3 * T.numAggregates);
  T.radii.resize_noinit(T.numAggregates);

  constexpr int kMinFlopsPerTask = 100000; // 20 μs @ 5 GFLOP/s.
  constexpr int kFlopsPerMember = 12; // 3 for the centroid and 9 for the squared distance.
  int const minAggregatesPerTask =
      RowsPerTask(T.numAggregates, T.numNodes, kMinFlopsPerTask / kFlopsPerMember);
  ParallelForN("RigidModeAMGGeometry", T.numAggregates, minAggregatesPerTask, [&](int a) {
    auto const members = T.members[a];
    double const invCount = 1.0 / isize(members);
    double centroid[3] = {0.0, 0.0, 0.0};
    for (int const i : members) {
      for (int d = 0; d < 3; ++d) {
        centroid[d] += positions[3 * i + d];
      }
    }
    for (int d = 0; d < 3; ++d) {
      centroid[d] *= invCount;
      T.centroids[3 * a + d] = centroid[d];
    }

    double sumSqr = 0.0;
    for (int const i : members) {
      for (int d = 0; d < 3; ++d) {
        sumSqr += Sqr(positions[3 * i + d] - centroid[d]);
      }
      if (!radii.empty()) {
        sumSqr += Sqr(radii[i]);
      }
    }
    double const radius = Sqrt(sumSqr * invCount);
    T.radii[a] = (IsFinite(radius) && radius > 0.0) ? radius : 1.0;
  });
}

/// @brief Set the interpolation coefficients of a transfer, q = w (x_i - c_a) / ℓ_a and r = w ℓ_i /
/// ℓ_a, computed in double and rounded once, and their copies in restriction order.
///
/// @param[in] positions Fine node positions, 3 per node.
/// @param[in] radii Fine node radii, or empty if they are zero (level 0).
/// @param[in] decoupled Aggregates whose q are zero, or empty.
/// @param[in,out] T Transfer with its aggregates' geometry.
template <typename Scalar>
void SetCoefficients(
    Span<double const> positions,
    Span<double const> radii,
    Span<bool const> decoupled,
    RigidModeTransfer<Scalar>& T) {
  T.coefficients.resize_noinit(T.aggregates.size());
  T.restrictionCoefficients.resize_noinit(T.aggregates.size());

  constexpr int kMinOpsPerTask = 30000; // 20 μs @ 1.5 G operations/s.
  constexpr int kFlopsPerWeight = 7; // A division for the scale and 6 for q.
  int const minNodesPerTask =
      RowsPerTask(T.numNodes, isize(T.aggregates), kMinOpsPerTask / kFlopsPerWeight);
  ParallelForN("RigidModeAMGCoefficients", T.numNodes, minNodesPerTask, [&](int i) {
    for (int k = T.pointers[i]; k < T.pointers[i + 1]; ++k) {
      int const a = T.aggregates[k];
      double const w = T.weights[k];
      double const scale = w / T.radii[a];
      bool const zeroRotation = !decoupled.empty() && decoupled[a];
      auto& p = T.coefficients[k];
      p.w = static_cast<Scalar>(w);
      for (int d = 0; d < 3; ++d) {
        p.q[d] = zeroRotation
            ? Scalar(0)
            : static_cast<Scalar>((positions[3 * i + d] - T.centroids[3 * a + d]) * scale);
      }
      p.r = radii.empty() ? Scalar(0) : static_cast<Scalar>(radii[i] * scale);
    }
  });

  constexpr int kAccessesPerEntry = 3; // Index, source and destination.
  ParallelForN(
      "RigidModeAMGRestrictionCoefficients",
      isize(T.coefficients),
      kMinOpsPerTask / kAccessesPerEntry,
      [&](int k) { T.restrictionCoefficients[k] = T.coefficients[T.restrictionEntries[k]]; });
}

/// @brief Index of a block column, which must be present, in a block row with sorted columns.
[[nodiscard]] inline int BlockIndex(Span<int const> columns, int column) {
  int const index = static_cast<int>(std::ranges::lower_bound(columns, column) - columns.begin());
  MOCHI_ASSERT_VERBOSE(
      index < isize(columns) && columns[index] == column, "Expected the block column.");
  return index;
}

/// @brief Make a block-sparse matrix with a symmetric pattern and sorted columns exactly symmetric
/// by copying its upper triangle, diagonal blocks included, onto its lower triangle.
template <typename Scalar, int kBlockSize>
void MirrorUpperTriangle(BlockSparseMatrix<Scalar, kBlockSize, int, int>& M) {
  int const numRows = M.BlockRows();
  constexpr int kMinOpsPerTask = 30000; // 20 μs @ 1.5 G operations/s.
  // Per block, on average: the lower blocks, half of all blocks, each binary-search their upper
  // transpose (about 4 steps) and copy its 36 values, so (4 + 36) / 2 = 20.
  constexpr int kAccessesPerBlock = 20;
  ParallelForRange(
      "RigidModeAMGMirror",
      0,
      numRows,
      RowsPerTask(numRows, M.NumNonZeroBlocks(), kMinOpsPerTask / kAccessesPerBlock),
      numRows,
      [&](int rowBegin, int rowEnd) {
        for (int a = rowBegin; a < rowEnd; ++a) {
          auto const columns = M.Indices(a);
          auto row = M.Values(a);
          int const diagonal = BlockIndex(columns, a);
          for (int k = 0; k < diagonal; ++k) {
            int const b = columns[k];
            row[k] = M.Values(b)[BlockIndex(M.Indices(b), a)].Transpose();
          }

          auto block = row[diagonal];
          for (int r = 1; r < kBlockSize; ++r) {
            for (int c = 0; c < r; ++c) {
              block(r, c) = block(c, r);
            }
          }
        }
      });
}

/// @brief A_next = P^T A P for the interpolation P from the next coarser level, with 3 or 6 DoFs
/// per fine node, the block size of A. Row a of P^T A, with the pattern @p ptAPattern, is
/// accumulated in scratch storage, copied to @p ptA unless it is null, and multiplied by P right
/// away.
///
/// @note @ref AddGalerkinRoundingBound bounds this product's rounding from its summation
/// structure, the nodes each entry sums over in each stage and the products each node adds, so a
/// change to the summation must update the bound.
template <typename Scalar, typename MatrixType, typename PatternType>
void Galerkin(
    RigidModeTransfer<Scalar> const& T,
    MatrixType const& A,
    PatternType const& ptAPattern,
    std::type_identity_t<BlockSparseMatrix<Scalar, 6, int, int>>* ptA,
    BlockSparseMatrix<Scalar, 6, int, int>& Ac) {
  constexpr int kFineDofs = MatrixType::kBlockSize;
  // Vectors of 3 columns per row of a block of A or P^T A.
  constexpr int kHalves = kFineDofs / 3;
  int const numAggregates = T.numAggregates;
  int const numNodes = A.BlockRows();
  ParallelForRange(
      "RigidModeAMGGalerkin",
      0,
      numAggregates,
      // About 30 ns per P^T A block, so 250 blocks take about 8 µs.
      RowsPerTask(numAggregates, ptAPattern.NumTargets(), 250),
      numAggregates,
      [&](int aggregateBegin, int aggregateEnd) {
        using V4 = Simd<Scalar, 4>;
        // Index of each fine node's block in row a of P^T A, or -1 if it has none.
        DynamicArray<int> fineSlot(numNodes, -1);
        // Row a of P^T A: per block, its 3 translation rows, then its 3 rotation rows.
        DynamicArray<V4> rowPtA;
        for (int a = aggregateBegin; a < aggregateEnd; ++a) {
          auto const columns = ptAPattern[a];
          int const numBlocks = isize(columns);
          for (int k = 0; k < numBlocks; ++k) {
            fineSlot[columns[k]] = k;
          }
          rowPtA.resize_noinit(6 * kHalves * static_cast<size_t>(numBlocks));
          std::fill(rowPtA.begin(), rowPtA.end(), SimdZero<V4>());

          for (int t = T.restrictionPointers[a]; t < T.restrictionPointers[a + 1]; ++t) {
            int const i = T.restrictionNodes[t];
            auto const& p = T.restrictionCoefficients[t];
            V4 const w = Broadcast<V4>(p.w);
            V4 const q0 = Broadcast<V4>(p.q[0]);
            V4 const q1 = Broadcast<V4>(p.q[1]);
            V4 const q2 = Broadcast<V4>(p.q[2]);
            [[maybe_unused]] V4 const r = Broadcast<V4>(p.r);
            auto const columnsA = A.Indices(i);
            Scalar const* rowA = A.Values(i).Data();
            int const ldA = kFineDofs * isize(columnsA);
            for (int kk = 0; kk < isize(columnsA); ++kk) {
              V4* o = rowPtA.data() + 6 * kHalves * fineSlot[columnsA[kk]];
              for (int h = 0; h < kHalves; ++h) {
                Scalar const* m = rowA + kFineDofs * kk + 3 * h;
                V4 const m0 = Load<3, V4>(m);
                V4 const m1 = Load<3, V4>(m + ldA);
                V4 const m2 = Load<3, V4>(m + 2 * ldA);
                // Translation rows w m_t, rotation rows [q]x m_t + r m_r.
                o[h] += w * m0;
                o[kHalves + h] += w * m1;
                o[2 * kHalves + h] += w * m2;
                if constexpr (kFineDofs == 3) {
                  o[3] += q1 * m2 - q2 * m1;
                  o[4] += q2 * m0 - q0 * m2;
                  o[5] += q0 * m1 - q1 * m0;
                } else {
                  o[3 * kHalves + h] += q1 * m2 - q2 * m1 + r * Load<3, V4>(m + 3 * ldA);
                  o[4 * kHalves + h] += q2 * m0 - q0 * m2 + r * Load<3, V4>(m + 4 * ldA);
                  o[5 * kHalves + h] += q0 * m1 - q1 * m0 + r * Load<3, V4>(m + 5 * ldA);
                }
              }
            }
          }

          if (ptA != nullptr) {
            Scalar* out = ptA->Values(a).Data();
            int const ld = kFineDofs * numBlocks;
            for (int k = 0; k < numBlocks; ++k) {
              for (int row = 0; row < 6; ++row) {
                for (int h = 0; h < kHalves; ++h) {
                  Store<3>(
                      out + row * ld + kFineDofs * k + 3 * h, rowPtA[(6 * k + row) * kHalves + h]);
                }
              }
            }
          }

          auto const columnsC = Ac.Indices(a);
          int const numBlocksC = isize(columnsC);
          Scalar* out = Ac.Values(a).Data();
          int const ldC = 6 * numBlocksC;
          // Blocks b >= a of row a of A_next, each summed over the fine nodes that interpolate from
          // b. A row (m_t, m_r) of a P^T A block maps to (w m_t, q x m_t + r m_r), with q x m_t =
          // q.yzx m_t.zxy - q.zxy m_t.yzx.
          for (int k = BlockIndex(columnsC, a); k < numBlocksC; ++k) {
            int const b = columnsC[k];
            V4 translation[6];
            V4 rotation[6];
            for (int row = 0; row < 6; ++row) {
              translation[row] = SimdZero<V4>();
              rotation[row] = SimdZero<V4>();
            }

            // TODO: Explore precomputing, once per hierarchy, the contributing fine nodes of each
            // block of A_next, so this loop skips the nodes absent from row a of P^T A.
            for (int t = T.restrictionPointers[b]; t < T.restrictionPointers[b + 1]; ++t) {
              int const slot = fineSlot[T.restrictionNodes[t]];
              if (slot < 0) {
                continue;
              }
              auto const& p = T.restrictionCoefficients[t];
              V4 const w = Broadcast<V4>(p.w);
              V4 const q = Load<3, V4>(p.q);
              V4 const qyzx = Shuffle<1, 2, 0, 3>(q);
              V4 const qzxy = Shuffle<2, 0, 1, 3>(q);
              [[maybe_unused]] V4 const r = Broadcast<V4>(p.r);
              V4 const* m = rowPtA.data() + 6 * kHalves * slot;
              for (int row = 0; row < 6; ++row) {
                V4 const mt = m[row * kHalves];
                translation[row] += w * mt;
                if constexpr (kFineDofs == 3) {
                  rotation[row] += qyzx * Shuffle<2, 0, 1, 3>(mt) - qzxy * Shuffle<1, 2, 0, 3>(mt);
                } else {
                  rotation[row] += qyzx * Shuffle<2, 0, 1, 3>(mt) - qzxy * Shuffle<1, 2, 0, 3>(mt) +
                      r * m[row * kHalves + 1];
                }
              }
            }

            for (int row = 0; row < 6; ++row) {
              Store<3>(out + row * ldC + 6 * k, translation[row]);
              Store<3>(out + row * ldC + 6 * k + 3, rotation[row]);
            }
          }

          for (int const j : columns) {
            fineSlot[j] = -1;
          }
        }
      });

  MirrorUpperTriangle(Ac);
}

/// @brief Add to the diagonal of A_next, as @ref Galerkin computes it from @p T, @p A and
/// @p ptAPattern, a bound on its rounding error, so that A_next ⪰ P^T A P, the exact product of the
/// stored A and P.
///
/// @details Row r gets δ_r = α z_r, with z = |P|^T |A| |P| 1, α = (γ_G + u (1 + γ_G)) / (1 - u)^2,
/// γ_G = γ(m_1) + γ(m_2) + γ(m_1) γ(m_2), γ(m) = m u / (1 - m u) and u the unit roundoff of Scalar.
/// @ref Galerkin sums each entry of row a of P^T A over the nodes of R(a) (the restriction of
/// aggregate a) whose row of A holds the entry's column, and each entry of block (a, b) of A_next
/// over the nodes of R(b) in the pattern of row a of P^T A: at most |R(b)|. Each node adds at most
/// p products, p = 2 from 3 fine DoFs and 3 from 6, so with n_1 and n_2 the most nodes per sum of
/// each stage, at most m_i = p n_i roundings reach a product in any summation order. @ref Galerkin
/// computes the upper triangle and mirrors it, so for an exactly symmetric A, its result G
/// satisfies |G - P^T A P| <= γ_G |P|^T |A| |P| entrywise, and Gershgorin's theorem gives
/// G + diag(γ_G z) ⪰ P^T A P; α also covers rounding δ_r to Scalar and adding it. Rows that do not
/// interpolate, such as decoupled rotations, get no shift, so a term J set there after
/// @ref Galerkin carries over: A_next ⪰ P^T A P + J. Assumes finite arithmetic, rounding to
/// nearest, no subnormal numbers and m_i u < 1. Non-finite entries of A or P make A_next
/// non-finite.
template <typename Scalar, typename MatrixType, typename PatternType>
void AddGalerkinRoundingBound(
    RigidModeTransfer<Scalar> const& T,
    MatrixType const& A,
    PatternType const& ptAPattern,
    BlockSparseMatrix<Scalar, 6, int, int>& Ac) {
  constexpr int kFineDofs = MatrixType::kBlockSize;
  constexpr int kMinOpsPerTask = 30000; // 20 μs @ 1.5 G operations/s.
  int const numNodes = A.BlockRows();
  int const numAggregates = T.numAggregates;
  // TODO: Use Abs once it clears the sign bit like std::fabs instead of comparing with zero, which
  // can branch on the entries' signs. That needs a constexpr std::fabs (C++23).
  auto const absolute = [](Scalar x) { return std::fabs(static_cast<double>(x)); };

  // v = |P| 1. A translation row of an interpolation block [w I | -[q]x] holds |w| and the other
  // two components of |q|; a rotation row, |r|.
  DynamicArray<double> v(kFineDofs * numNodes, 0.0);
  // About 20 absolute values and sums per coefficient: 18 from 3 fine DoFs, 24 from 6.
  constexpr int kOpsPerCoefficient = 20;
  ParallelForN(
      "RigidModeAMGBoundRowSums",
      numNodes,
      RowsPerTask(numNodes, isize(T.coefficients), kMinOpsPerTask / kOpsPerCoefficient),
      [&](int i) {
        for (int k = T.pointers[i]; k < T.pointers[i + 1]; ++k) {
          auto const& p = T.coefficients[k];
          for (int d = 0; d < 3; ++d) {
            int const d1 = (d + 1) % 3;
            int const d2 = (d + 2) % 3;
            v[kFineDofs * i + d] += absolute(p.w) + absolute(p.q[d1]) + absolute(p.q[d2]);
            if constexpr (kFineDofs == 6) {
              v[kFineDofs * i + 3 + d] += absolute(p.r);
            }
          }
        }
      });

  // y = |A| v.
  DynamicArray<double> y(kFineDofs * numNodes, 0.0);
  // An absolute value, a product and a sum per entry of a block of A.
  constexpr int kOpsPerBlock = 3 * kFineDofs * kFineDofs;
  ParallelForN(
      "RigidModeAMGBoundProduct",
      numNodes,
      RowsPerTask(numNodes, A.NumNonZeroBlocks(), kMinOpsPerTask / kOpsPerBlock),
      [&](int i) {
        auto const columns = A.Indices(i);
        auto const blocks = A.Values(i);
        for (int k = 0; k < isize(columns); ++k) {
          double const* vj = &v[kFineDofs * columns[k]];
          for (int r = 0; r < kFineDofs; ++r) {
            for (int c = 0; c < kFineDofs; ++c) {
              y[kFineDofs * i + r] += absolute(blocks[k](r, c)) * vj[c];
            }
          }
        }
      });

  // z = |P|^T y, and for each aggregate a, the most nodes of R(a) whose rows of A hold one column,
  // counted as in Galerkin's first stage.
  DynamicArray<double> z(6 * numAggregates, 0.0);
  DynamicArray<int> n1PerAggregate(numAggregates, 0);
  // About 20 operations per block of P^T A: an increment and a maximum from each of the several
  // restriction nodes whose rows of A hold its column, its reset, and a share of z's 27 or 36
  // operations per restriction node.
  constexpr int kOpsPerPtABlock = 20;
  ParallelForRange(
      "RigidModeAMGBoundRestriction",
      0,
      numAggregates,
      RowsPerTask(numAggregates, ptAPattern.NumTargets(), kMinOpsPerTask / kOpsPerPtABlock),
      numAggregates,
      [&](int aggregateBegin, int aggregateEnd) {
        // Number of nodes of R(a) whose row of A holds each column.
        DynamicArray<int> count(numNodes, 0);
        for (int a = aggregateBegin; a < aggregateEnd; ++a) {
          for (int t = T.restrictionPointers[a]; t < T.restrictionPointers[a + 1]; ++t) {
            int const i = T.restrictionNodes[t];
            for (int const j : A.Indices(i)) {
              n1PerAggregate[a] = Max(n1PerAggregate[a], ++count[j]);
            }
            auto const& p = T.restrictionCoefficients[t];
            double const* yi = &y[kFineDofs * i];
            for (int d = 0; d < 3; ++d) {
              int const d1 = (d + 1) % 3;
              int const d2 = (d + 2) % 3;
              z[6 * a + d] += absolute(p.w) * yi[d];
              z[6 * a + 3 + d] += absolute(p.q[d1]) * yi[d2] + absolute(p.q[d2]) * yi[d1];
              if constexpr (kFineDofs == 6) {
                z[6 * a + 3 + d] += absolute(p.r) * yi[3 + d];
              }
            }
          }
          for (int const j : ptAPattern[a]) {
            count[j] = 0;
          }
        }
      });

  // α from n_1 and n_2.
  int n1 = 0;
  int n2 = 0;
  for (int a = 0; a < numAggregates; ++a) {
    n1 = Max(n1, n1PerAggregate[a]);
    n2 = Max(n2, T.restrictionPointers[a + 1] - T.restrictionPointers[a]);
  }

  static_assert(kFineDofs == 3 || kFineDofs == 6, "Expected 3 or 6 DoFs per fine node.");
  constexpr int kProductsPerNode = (kFineDofs == 3) ? 2 : 3;
  // v, y, z and α are nonnegative doubles, and the only subtractions, 1 - m u and 1 - u in α, are
  // exact. v, y and z sum at most 3, 6 and 3 terms per entry of an int-indexed array, which has
  // fewer than 2^31 entries, so each δ_r goes through fewer than 2^35 roundings in any order: a
  // relative error below 2^-18. This factor, exact in double, thus keeps each δ_r at least α z_r.
  constexpr double kDoublePrecisionFactor = 1.0 + 0x1p-16;
  double const u = 0.5 * static_cast<double>(std::numeric_limits<Scalar>::epsilon());
  int const m1 = kProductsPerNode * n1;
  int const m2 = kProductsPerNode * n2;
  MOCHI_ASSERT_VERBOSE(Max(m1, m2) * u < 1.0, "Too many products for the rounding bound.");
  auto const gamma = [u](int m) { return m * u / (1.0 - m * u); };
  double const gammaG = gamma(m1) + gamma(m2) + gamma(m1) * gamma(m2);
  double const alpha = (gammaG + u * (1.0 + gammaG)) / Sqr(1.0 - u) * kDoublePrecisionFactor;

  // δ = α z on the diagonal.
  for (int a = 0; a < numAggregates; ++a) {
    auto block = Ac.Values(a)[BlockIndex(Ac.Indices(a), a)];
    for (int r = 0; r < 6; ++r) {
      block(r, r) += static_cast<Scalar>(alpha * z[6 * a + r]);
    }
  }
}

/// @brief Block Jacobi preconditioner with 6x6 blocks, as @ref BlockJacobiPrec, whose update checks
/// that the blocks and their computed inverses are positive definite.
template <typename Scalar>
struct CheckedBlockJacobi6 {
  /// @brief Invert the diagonal blocks of A, and return whether they and their inverses pass @ref
  /// FactorLDLt's check. A tiny pivot passes, but its reciprocal can overflow.
  [[nodiscard]] bool Update(BlockSparseMatrix<Scalar, 6, int, int> const& A) {
    int const numBlocks = A.BlockRows();
    inverses.resize(numBlocks);
    for (int a = 0; a < numBlocks; ++a) {
      // The blocks and their inverses are symmetric, so their column-major storage is also the
      // row-major storage that FactorLDLt and InverseFromLDLt6 use.
      Matrix<Scalar, 6, 6> factors = A.Values(a)[BlockIndex(A.Indices(a), a)];
      if (!FactorLDLt(factors.Data(), 6)) {
        return false;
      }
      InverseFromLDLt6(factors.Data(), inverses[a].Data());
      factors = inverses[a];
      if (!FactorLDLt(factors.Data(), 6)) {
        return false;
      }
    }
    return true;
  }

  void Solve(ColumnVectorView<Scalar const> x, ColumnVectorView<Scalar> y) const {
    ApplyBlockDiagonal<Scalar>(MakeConstSpan(inverses), x, y);
  }

  DynamicArray<Matrix<Scalar, 6, 6>> inverses;
};

/// @brief Whether a matrix with 6 DoFs per node is finite and its translation block, over the
/// first 3 DoFs of every node, passes @ref InvertIfPositiveDefinite. If so, sets @p inverse to that
/// block's inverse on the translation DoFs and to zero elsewhere.
///
/// @note The entries outside the translation block must be finite too: at level 1, which has no
/// cached P^T A to check, non-finite interpolation coefficients appear only there, and would make
/// the restricted residual non-finite, which zero rotation rows and columns don't mask.
template <typename Scalar>
[[nodiscard]] bool InvertTranslationsIfPositiveDefinite(
    Matrix<Scalar> const& A,
    Matrix<Scalar>& inverse) {
  MOCHI_ASSERT_VERBOSE(A.Rows() == A.Cols() && A.Rows() % 6 == 0, "Expected 6 DoFs per node.");
  if (!IsFinite(MakeConstSpan(A))) {
    return false;
  }

  int const numNodes = A.Rows() / 6;
  Matrix<Scalar> translations(3 * numNodes, 3 * numNodes);
  for (int b = 0; b < numNodes; ++b) {
    for (int a = 0; a < numNodes; ++a) {
      translations.template Block<3, 3>(3 * a, 3 * b, 3, 3) =
          A.template Block<3, 3>(6 * a, 6 * b, 3, 3);
    }
  }

  Matrix<Scalar> translationInverse;
  if (!InvertIfPositiveDefinite(translations, translationInverse)) {
    return false;
  }

  inverse = Matrix<Scalar>::Zero(A.Rows(), A.Cols());
  for (int b = 0; b < numNodes; ++b) {
    for (int a = 0; a < numNodes; ++a) {
      inverse.template Block<3, 3>(6 * a, 6 * b, 3, 3) =
          translationInverse.template Block<3, 3>(3 * a, 3 * b, 3, 3);
    }
  }
  return true;
}

/// @brief e_i = Σ_a P(i, a) h_a for the fine nodes [nodeBegin, nodeEnd), with kFineDofs = 3 or 6
/// DoFs per fine node.
template <int kFineDofs, typename Scalar>
void InterpolateRigidModes(
    RigidModeTransfer<Scalar> const& T,
    std::type_identity_t<ColumnVectorView<Scalar const>> h,
    std::type_identity_t<ColumnVectorView<Scalar>> e,
    int nodeBegin,
    int nodeEnd) {
  using V4 = Simd<Scalar, 4>;
  for (int i = nodeBegin; i < nodeEnd; ++i) {
    V4 translation = SimdZero<V4>();
    V4 rotation = SimdZero<V4>();
    for (int k = T.pointers[i]; k < T.pointers[i + 1]; ++k) {
      auto const& p = T.coefficients[k];
      Scalar const* ha = h.Data() + 6 * T.aggregates[k];
      V4 const hr = Load<3, V4>(ha + 3);
      translation += Broadcast<V4>(p.w) * Load<3, V4>(ha) - Cross3(Load<3, V4>(p.q), hr);
      if constexpr (kFineDofs == 6) {
        rotation += Broadcast<V4>(p.r) * hr;
      }
    }

    Store<3>(e.Data() + kFineDofs * i, translation);
    if constexpr (kFineDofs == 6) {
      Store<3>(e.Data() + kFineDofs * i + 3, rotation);
    }
  }
}

/// @brief g_a = Σ_i P(i, a)^T x_i for the aggregates [aggregateBegin, aggregateEnd), with kFineDofs
/// = 3 or 6 DoFs per fine node.
template <int kFineDofs, typename Scalar>
void RestrictRigidModes(
    RigidModeTransfer<Scalar> const& T,
    std::type_identity_t<ColumnVectorView<Scalar const>> x,
    std::type_identity_t<ColumnVectorView<Scalar>> g,
    int aggregateBegin,
    int aggregateEnd) {
  using V4 = Simd<Scalar, 4>;
  for (int a = aggregateBegin; a < aggregateEnd; ++a) {
    V4 translation = SimdZero<V4>();
    V4 rotation = SimdZero<V4>();
    for (int k = T.restrictionPointers[a]; k < T.restrictionPointers[a + 1]; ++k) {
      auto const& p = T.restrictionCoefficients[k];
      Scalar const* xi = x.Data() + kFineDofs * T.restrictionNodes[k];
      V4 const xt = Load<3, V4>(xi);
      translation += Broadcast<V4>(p.w) * xt;
      rotation += Cross3(Load<3, V4>(p.q), xt);
      if constexpr (kFineDofs == 6) {
        rotation += Broadcast<V4>(p.r) * Load<3, V4>(xi + 3);
      }
    }

    Store<3>(g.Data() + 6 * a, translation);
    Store<3>(g.Data() + 6 * a + 3, rotation);
  }
}

} // namespace details

template <typename Scalar>
RigidBodyModesAMGPrec<Scalar>::RigidBodyModesAMGPrec(
    BSpMatrixView A,
    Span<Scalar const> referencePositions,
    Span<Scalar const> displacements)
    : _A0(A), _fineSmoother(A) {
  MOCHI_ASSERT_VERBOSE(_A0.BlockRows() == _A0.BlockCols(), "Expected a square matrix.");
  MOCHI_ASSERT_VERBOSE(
      isize(referencePositions) == _A0.Rows() && isize(displacements) == _A0.Rows(),
      "Expected 3 position coordinates per block row.");

#if MOCHI_ASSERT_VERBOSE_ENABLED
  for (int i = 0; i < _A0.BlockRows(); ++i) {
    auto const columns = _A0.Indices(i);
    MOCHI_ASSERT_VERBOSE(Contains(columns, i), "Expected diagonal blocks.");
    for (int const j : columns) {
      MOCHI_ASSERT_VERBOSE(Contains(_A0.Indices(j), i), "Expected a symmetric pattern.");
    }
  }
#endif // MOCHI_ASSERT_VERBOSE_ENABLED

  DynamicArray<double> const reference(referencePositions.begin(), referencePositions.end());
  BuildHierarchy(MakeConstSpan(reference));

  // Level 1 has 6 rows per aggregate and at most one aggregate per node, so at most twice A's rows.
  // A deeper level has 6 rows per aggregate of 6-row nodes, so it never grows.
  _spectralRadiusRhs = ColumnVector<Scalar>(Max(_A0.Rows(), _levels.front().A.Rows()));
  details::SetXorShiftRandom(AsView(_spectralRadiusRhs));

  _positions.resize(_A0.Rows());
  SetPositions(referencePositions, displacements);
  UpdateInterpolation();
  UpdateNumerics();
}

template <typename Scalar>
void RigidBodyModesAMGPrec<Scalar>::Update(
    BSpMatrixView A,
    Span<Scalar const> referencePositions,
    Span<Scalar const> displacements) {
  MOCHI_ASSERT_VERBOSE(
      A.Pointers() == _A0.Pointers() && A.Indices() == _A0.Indices(), "Pattern mismatch.");
  MOCHI_ASSERT_VERBOSE(
      isize(referencePositions) == _A0.Rows() && isize(displacements) == _A0.Rows(),
      "Expected 3 position coordinates per block row.");

#if MOCHI_ASSERT_VERBOSE_ENABLED
  {
    auto const [partition, numAggregates] = details::Aggregate(AsGraphView(A));
    Transfer level0;
    details::SetAggregates(partition, numAggregates, level0);
    DynamicArray<double> const reference(referencePositions.begin(), referencePositions.end());
    SetLevel0Weights(A, partition, MakeConstSpan(reference), level0);
    auto const& T0 = _levels.front().transfer;
    MOCHI_ASSERT_VERBOSE(
        level0.pointers == T0.pointers && level0.aggregates == T0.aggregates,
        "The reference positions must decide the level-0 weights of construction.");
  }
#endif // MOCHI_ASSERT_VERBOSE_ENABLED

  _A0 = A;
  _fineSmoother.Update(_A0);
  if (SetPositions(referencePositions, displacements)) {
    UpdateInterpolation();
  }
  UpdateNumerics();
}

template <typename Scalar>
void RigidBodyModesAMGPrec<Scalar>::SetLevel0Weights(
    BSpMatrixView A,
    DynamicArray<int> const& partition,
    Span<double const> referencePositions,
    Transfer& T) {
  details::SetSmoothedWeights(A, partition, kSmoothingWeight, T);
  details::FilterWeights([](int, int, double weight) { return weight >= kTruncation; }, T);
  details::SetAggregateGeometry(referencePositions, {}, T);

  DynamicArray<bool> memberOnly(T.numAggregates);
  // About 40 ns per aggregate, so 500 aggregates take about 20 µs.
  ParallelForN("RigidModeAMGMemberOnly", T.numAggregates, 500, [&](int a) {
    memberOnly[a] = !details::PureRowMarginAtLeast(T, a, referencePositions, kMemberOnlyMargin);
  });

  details::FilterWeights(
      [&](int i, int aggregate, double) {
        int const a = partition[i];
        return !memberOnly[a] || aggregate == a;
      },
      T);
}

template <typename Scalar>
void RigidBodyModesAMGPrec<Scalar>::BuildHierarchy(Span<double const> referencePositions) {
  auto const graph0 = AsGraphView(_A0);
  auto const [partition0, numAggregates0] = details::Aggregate(graph0);
  _levels.emplace_back();
  auto& level1 = _levels.back();
  details::SetAggregates(partition0, numAggregates0, level1.transfer);
  SetLevel0Weights(_A0, partition0, referencePositions, level1.transfer);
  details::SetRestrictionPattern(level1.transfer);
  _decoupled = DynamicArray<bool>(numAggregates0, false);
  _PtA0Pattern = Traverse(details::RestrictionGraph(level1.transfer), graph0).SortTargets();
  level1.A = CoarseMatrix(
      numAggregates0,
      Traverse(_PtA0Pattern, details::InterpolationGraph(level1.transfer)).SortTargets());

  // Coarser levels, while they reduce the node count to more than one node, as in AMGPrec.
  while (true) {
    auto& current = _levels.back();
    int const numNodes = current.A.BlockRows();
    auto const graph = AsGraphView(current.A);
    auto const [partition, numAggregates] = details::Aggregate(graph);
    if (numAggregates == numNodes || numAggregates <= 1) {
      break;
    }

    CoarseLevel next;
    details::SetAggregates(partition, numAggregates, next.transfer);
    details::SetSmoothedWeights(current.A, partition, kSmoothingWeight, next.transfer);
    details::SetRestrictionPattern(next.transfer);
    auto ptAPattern = Traverse(details::RestrictionGraph(next.transfer), graph).SortTargets();
    next.A = CoarseMatrix(
        numAggregates,
        Traverse(ptAPattern, details::InterpolationGraph(next.transfer)).SortTargets());
    current.PtA = CoarseMatrix(numNodes, std::move(ptAPattern));
    _levels.push_back(std::move(next));
  }

  _fineU = ColumnVector<Scalar>(_A0.Rows());
  _fineR = ColumnVector<Scalar>(_A0.Rows());
  for (int s = 0; s < isize(_levels); ++s) {
    auto& level = _levels[s];
    int const numRows = level.A.Rows();
    level.b = ColumnVector<Scalar>(numRows);
    level.x = ColumnVector<Scalar>(numRows);
    if (s + 1 < isize(_levels)) {
      level.u = ColumnVector<Scalar>(numRows);
      level.r = ColumnVector<Scalar>(numRows);
    }
  }
}

template <typename Scalar>
bool RigidBodyModesAMGPrec<Scalar>::SetPositions(
    Span<Scalar const> referencePositions,
    Span<Scalar const> displacements) {
  bool changed = false;
  for (int k = 0; k < isize(_positions); ++k) {
    double const x =
        static_cast<double>(referencePositions[k]) + static_cast<double>(displacements[k]);
    // Compares bits: under MSVC /fp:fast, NaN != x is false for every x.
    if (std::bit_cast<uint64_t>(x) != std::bit_cast<uint64_t>(_positions[k])) {
      _positions[k] = x;
      changed = true;
    }
  }
  return changed;
}

template <typename Scalar>
void RigidBodyModesAMGPrec<Scalar>::UpdateInterpolation() {
  auto const positions = MakeConstSpan(_positions);
  auto& T0 = _levels.front().transfer;
  details::SetAggregateGeometry(positions, {}, T0);

  double const decoupleMargin = DecoupleMargin();
  // About 40 ns per aggregate, so 500 aggregates take about 20 µs.
  ParallelForN("RigidModeAMGDecouple", T0.numAggregates, 500, [&](int a) {
    _decoupled[a] = !details::PureRowMarginAtLeast(T0, a, positions, decoupleMargin);
  });
  details::SetCoefficients(positions, {}, MakeConstSpan(_decoupled), T0);

  for (int s = 1; s < isize(_levels); ++s) {
    auto const& fine = _levels[s - 1].transfer;
    auto& T = _levels[s].transfer;
    details::SetAggregateGeometry(MakeConstSpan(fine.centroids), MakeConstSpan(fine.radii), T);
    details::SetCoefficients(MakeConstSpan(fine.centroids), MakeConstSpan(fine.radii), {}, T);
  }
}

template <typename Scalar>
void RigidBodyModesAMGPrec<Scalar>::UpdateNumerics() {
  constexpr AMGOptions<Scalar> kOptions = {};
  constexpr Scalar kDefaultRelaxationFactor = AMGPrec<Scalar, 3>::kDefaultRelaxationFactor;
  auto const rhs = AsConstView(_spectralRadiusRhs);
  // TODO: Explore overlapping the fine level's setup with the coarse levels' setup, which are
  // independent of each other, so the setup scales at high worker counts.
  _fineRelaxationFactor =
      details::AutoRelaxationFactor(_A0, _fineSmoother, kOptions, 0, rhs.TopRows(_A0.Rows()))
          .value_or(kDefaultRelaxationFactor);

  auto& level1 = _levels.front();
  details::Galerkin(level1.transfer, _A0, _PtA0Pattern, nullptr, level1.A);
  for (int a = 0; a < isize(_decoupled); ++a) {
    if (_decoupled[a]) {
      auto block = level1.A.Values(a)[details::BlockIndex(level1.A.Indices(a), a)];
      Scalar const scale = (block(0, 0) + block(1, 1) + block(2, 2)) / 3;
      block.template Block<3, 3>(3, 3, 3, 3).SetIdentity();
      block.template Block<3, 3>(3, 3, 3, 3) *= scale;
    }
  }

  int const numLevels = isize(_levels);
  // Whether the cached P^T A that the cycle applies to restrict to coarse level s + 1 is finite.
  // The restriction to level 1 applies none.
  auto const restrictsWithFinitePtA = [this](int s) {
    return s == 0 || IsFinite(_levels[s - 1].PtA.Values());
  };

  // Each coarse level is formed once the levels above it are accepted.
  _numAcceptedCoarseLevels = 0;
  for (int s = 0; s + 1 < numLevels; ++s) {
    auto& current = _levels[s];
    if (!current.smoother.Update(current.A)) {
      if (s == 0) {
        details::AddGalerkinRoundingBound(current.transfer, _A0, _PtA0Pattern, current.A);
      } else {
        auto const& finer = _levels[s - 1];
        details::AddGalerkinRoundingBound(
            current.transfer, finer.A, AsGraphView(finer.PtA), current.A);
      }
      if (!IsFinite(current.A.Values()) || !restrictsWithFinitePtA(s) ||
          !current.smoother.Update(current.A)) {
        MOCHI_LOG_WARNING(
            "AMG: coarse level %d fails its positive-definiteness or finiteness checks, even once "
            "stiffened. It and the coarser levels are dropped, so convergence may slow.",
            s + 1);
        return;
      }
    }
    _numAcceptedCoarseLevels = s + 1;
    current.relaxationFactor =
        details::AutoRelaxationFactor(
            current.A, current.smoother, kOptions, s + 1, rhs.TopRows(current.A.Rows()))
            .value_or(kDefaultRelaxationFactor);
    auto& next = _levels[s + 1];
    details::Galerkin(next.transfer, current.A, AsGraphView(current.PtA), &current.PtA, next.A);
  }

  auto const coarsest = ToMatrix(_levels.back().A);
  if (details::InvertIfPositiveDefinite(coarsest, _coarsestInverse) ||
      (restrictsWithFinitePtA(numLevels - 1) &&
       details::InvertTranslationsIfPositiveDefinite(coarsest, _coarsestInverse))) {
    // TODO: Explore regularizing the coarsest inverse when the accepted matrix still has
    // eigenvalues below its rounding floor. See D128281511.
    _numAcceptedCoarseLevels = numLevels;
  } else {
    MOCHI_LOG_WARNING(
        "AMG: coarse level %d, the coarsest, fails its positive-definiteness or finiteness checks, "
        "even on its translations. It is dropped, so convergence may slow.",
        numLevels);
  }
}

template <typename Scalar>
void RigidBodyModesAMGPrec<Scalar>::Solve(
    ColumnVectorView<Scalar const> b,
    ColumnVectorView<Scalar> x) const {
  ConcurrentSolve(b, x, ParallelWorkerInfo{0, 1, 0, x.Rows(), ParallelBarrier(1)});
}

template <typename Scalar>
void RigidBodyModesAMGPrec<Scalar>::ConcurrentSolve(
    ColumnVectorView<Scalar const> b,
    ColumnVectorView<Scalar> x,
    ParallelWorkerInfo const& data) const {
  Preconditioner<Scalar>::ValidateInputOutput(_A0.Rows(), b, x);
  MOCHI_ASSERT_VERBOSE(
      data.rBegin % 3 == 0 && data.rEnd % 3 == 0,
      "Row ranges must begin and end at multiples of 3.");

  Scalar const omega = _fineRelaxationFactor;
  int const rowBegin = data.rBegin;
  int const rowEnd = data.rEnd;
  int const numRows = rowEnd - rowBegin;
  auto& u = _fineU;
  auto& r = _fineR;
  auto const setResidual = [&]() {
    _A0.ApplyToRange(u, r, rowBegin, rowEnd);
    r.MiddleRows(rowBegin, numRows) =
        b.MiddleRows(rowBegin, numRows) - r.MiddleRows(rowBegin, numRows);
  };

  // TODO: Explore an additive cycle at the finest level.
  // Pre-smoothing from zero: u = omega D^{-1} b.
  _fineSmoother.ConcurrentSolve(b, u, data);
  u.MiddleRows(rowBegin, numRows) *= omega;
  data.BarrierWait(); // Wait for 'u' to be up-to-date.

  if (_numAcceptedCoarseLevels > 0) {
    // Coarse correction: u += P h, where h, in level1.x, is the coarse cycle of g = P^T (b - A u),
    // in level1.b.
    auto const& level1 = _levels.front();
    auto const& T0 = level1.transfer;
    // TODO: Explore storing the finest P^T A, so the restriction needs no product with A_0.
    setResidual();
    data.BarrierWait(); // Wait for 'r' to be up-to-date.
    int const aggregateBegin = (data.workerId * T0.numAggregates) / data.numWorkers;
    int const aggregateEnd = ((data.workerId + 1) * T0.numAggregates) / data.numWorkers;
    details::RestrictRigidModes<3>(T0, r, level1.b, aggregateBegin, aggregateEnd);
    data.BarrierWait(); // Wait for 'g' to be up-to-date.
    if (data.workerId == 0) {
      CoarseCycle(0);
    }
    data.BarrierWait(); // Wait for 'h' to be up-to-date.
    details::InterpolateRigidModes<3>(T0, level1.x, r, rowBegin / 3, rowEnd / 3);
    u.MiddleRows(rowBegin, numRows) += r.MiddleRows(rowBegin, numRows);
    data.BarrierWait(); // Wait for 'u' to be up-to-date.
  }

  // Post-smoothing: x = u + omega D^{-1} (b - A u).
  setResidual();
  _fineSmoother.ConcurrentSolve(r, x, data);
  x.MiddleRows(rowBegin, numRows) =
      u.MiddleRows(rowBegin, numRows) + omega * x.MiddleRows(rowBegin, numRows);
}

template <typename Scalar>
void RigidBodyModesAMGPrec<Scalar>::CoarseCycle(int s) const {
  auto const& level = _levels[s];
  if (s + 1 == isize(_levels)) {
    level.x = _coarsestInverse * level.b;
    return;
  }

  Scalar const omega = level.relaxationFactor;

  // Pre-smoothing from zero: u = omega D^{-1} b.
  level.smoother.Solve(level.b, level.u);
  level.u *= omega;

  if (s + 1 < _numAcceptedCoarseLevels) {
    // Coarse correction: u += P h, where h is the coarse cycle of P^T b - (P^T A) u. The coarser
    // level's x holds (P^T A) u until its cycle overwrites it.
    auto const& next = _levels[s + 1];
    details::RestrictRigidModes<6>(next.transfer, level.b, next.b, 0, next.transfer.numAggregates);
    level.PtA.Apply(level.u, next.x);
    next.b -= next.x;
    CoarseCycle(s + 1);
    details::InterpolateRigidModes<6>(next.transfer, next.x, level.r, 0, next.transfer.numNodes);
    level.u += level.r;
  }

  // Post-smoothing: x = u + omega D^{-1} (b - A u).
  level.A.Apply(level.u, level.r);
  level.r = level.b - level.r;
  level.smoother.Solve(level.r, level.x);
  level.x = level.u + omega * level.x;
}

} // namespace mochi::krylov
