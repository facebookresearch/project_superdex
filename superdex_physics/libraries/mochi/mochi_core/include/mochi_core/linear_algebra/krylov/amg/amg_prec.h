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
#include <mochi_core/linear_algebra/krylov/amg/amg_level.h>
#include <mochi_core/linear_algebra/krylov/block_jacobi_prec.h>
#include <mochi_core/linear_algebra/krylov/colored_ssor_prec.h>
#include <mochi_core/linear_algebra/krylov/identity_prec.h>
#include <mochi_core/linear_algebra/krylov/preconditioner.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/matrix_operations.h>
#include <mochi_core/linear_algebra/utils/matrix_concepts.h>
#include <mochi_core/linear_algebra/utils/matrix_conversions.h>
#include <mochi_core/mochi_platform.h>
#include <mochi_core/utils/array_utils.h>
#include <mochi_core/utils/graph_views.h>
#include <mochi_core/utils/rand_utils.h>

#include <concepts>
#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>
#include <variant>

namespace mochi::krylov {

/// @brief Enumeration for selecting the smoother at each level
enum class Smoother : char {
  ApproximateJacobi = 0, ///< Approximation to block Jacobi (for post-smoothing)
  BlockJacobi = 1, ///< Block Jacobi
  SSOR = 2, ///< Colored block SSOR
};

/// @brief Configuration options for the AMG preconditioner.
template <typename T>
struct AMGOptions {
  static_assert(!std::is_const_v<T>, "Implementation assumes scalar type is non-const.");

  /// @brief Smoother used at each level.
  Smoother smoother = Smoother::BlockJacobi;

  /// @brief Safety factor applied to the estimated largest eigenvalue of D^{-1} A. Must be
  /// positive and finite.
  ///
  /// @note For @ref Smoother::BlockJacobi, D is the block diagonal of each smoothed level. A
  /// value > 1 reduces the relaxation factors and helps compensate for underestimating
  /// lambda_max(D^{-1} A).
  /// @note Only used with @ref Smoother::BlockJacobi when @ref relaxationFactor is negative.
  T spectralRadiusSafetyFactor = T(1.1);

  /// @brief Number of PCG iterations for estimating the spectral radius of D^{-1} A. Must be
  /// positive.
  ///
  /// @note The spectral radius is estimated at each smoothed level via a PCG Lanczos procedure
  /// (see @ref details::EstimateSpectralRadius).
  /// @note Only used with @ref Smoother::BlockJacobi when @ref relaxationFactor is auto-computed.
  int spectralRadiusMaxIters = 8;

  /// @brief Relaxation factor for the smoother correction: x += omega * M^{-1} * r.
  ///
  /// @details For @ref Smoother::BlockJacobi, M^{-1} is D^{-1}, where D is the block diagonal of A.
  /// If A and D are SPD, weighted Jacobi is stable for
  ///     0 < omega < 2 / lambda_max(D^{-1} A).
  /// Symmetric multigrid SPD analyses often use the equivalent sufficient condition
  ///     (2 / omega) D - A > 0
  /// for the Jacobi smoother.
  ///
  /// @note Applied at every smoothing iteration for all smoother types.
  /// @note Non-negative values are used directly at every level. Negative values enable automatic
  /// per-level damping for @ref Smoother::BlockJacobi. For each smoothed level, AMG estimates the
  /// largest eigenvalue of the block-Jacobi-scaled operator D^{-1} A and sets the relaxation factor
  /// to
  ///     omega = 4 / (3 * safety_factor * lambda_est),
  /// where safety_factor is @ref spectralRadiusSafetyFactor. This avoids choosing one fixed damping
  /// value for all levels.
  /// @note Per-level factors are intentional. The optimal relaxation factors can vary by level and
  /// need not be monotone. See Yang, "On the use of relaxation parameters in hybrid smoothers",
  /// UCRL-JC-151575 (2003).
  /// @note The automatic factor is not a certificate. The PCG/Lanczos estimate is deterministic,
  /// but finite-iteration Ritz values can underestimate the true largest eigenvalue. The safety
  /// factor makes the chosen omega more conservative, but auto mode does not prove that
  /// (2 / omega) D - A is positive definite, and therefore does not prove that the AMG V-cycle is
  /// SPD. Use an explicit non-negative @ref relaxationFactor, or a larger @ref
  /// spectralRadiusSafetyFactor, when a conservative damping policy is required.
  /// @note Negative values use 2/3 for non-BlockJacobi smoothers.
  /// @note If auto-estimation fails at a level, the preconditioner logs a warning and uses 2/3 for
  /// that level, also in @ref AMGPrec::Update instead of keeping the previous factor, because the
  /// preconditioner must be free of hysteresis: updating it with a matrix must yield the same
  /// preconditioner as constructing it from that matrix. Callers rely on this, e.g., to reconstruct
  /// a preconditioner instead of persisting it.
  T relaxationFactor = T(-1);

  /// @brief Number of pre-smoothing iterations.
  int numPreSmoothingSteps = 1;

  /// @brief Number of post-smoothing iterations.
  ///
  /// @note For @ref Smoother::ApproximateJacobi, only 1 post-smoothing iteration is done regardless
  /// of this value.
  int numPostSmoothingSteps = 1;

  /// @brief Prolongation smoothing weight used during aggregation.
  T prolongationSmoothingWeight = T(2.0 / 3.0);
};

/// @brief Algebraic Multi-Grid (AMG) preconditioner.
///
/// @details This class implements a preconditioner similar to the one described in "Algebraic
/// multigrid by smoothed aggregation for second and fourth order elliptic problems" by Vanek,
/// Mandel and Brezina.
///
/// @warning The matrix storage passed to the constructor or latest @ref Update call must outlive
/// this preconditioner or the next @ref Update call.
///
/// @note The default smoother is the weighted block Jacobi iteration with relaxation factors
/// auto-computed from the spectral radius of D^{-1} A at each smoothed level.
/// @note The coarsest problem is inverted numerically.
template <typename Scalar, int kDofsPerNode>
struct AMGPrec : Preconditioner<Scalar> {
  static constexpr auto kType = PreconditionerType::AMG;
  static constexpr Scalar kDefaultRelaxationFactor = Scalar(2.0 / 3.0);
  static_assert(!std::is_const_v<Scalar>, "Implementation assumes Scalar is non-const.");
  static_assert(kDofsPerNode > 0, "Preconditioner block size must be positive");

  struct Workspace {
    std::unique_ptr<Scalar[]> data;
    // r and z workspaces are shared across levels.
    size_t rOffset;
    size_t zOffset;
    // g and h workspaces must be different for each level.
    DynamicArray<size_t> gOffset;
    DynamicArray<size_t> hOffset;
  };

  template <
      template <typename, typename...> typename Storage,
      typename InputIdx,
      typename InputScalar>
  explicit AMGPrec(
      BlockSparseMatrix<InputScalar, kDofsPerNode, InputIdx, InputIdx, Storage> const& A,
      AMGOptions<Scalar> const& options = {});

  template <
      template <typename, typename...> typename Storage,
      typename InputIdx,
      typename InputScalar>
  explicit AMGPrec(
      SparseMatrix<InputScalar, InputIdx, InputIdx, Storage> const& A,
      AMGOptions<Scalar> const& options = {});

  template <typename MatrixType>
  explicit AMGPrec(MatrixType const& /*A*/) {
    MOCHI_ASSERT(false, "AMG preconditioner is only supported for (Block)SparseMatrix types.");
  }

  template <typename VectorIn, typename VectorOut>
  void operator()(VectorIn const& x, VectorOut&& Px) const;

  void Solve(ColumnVectorView<Scalar const> x, ColumnVectorView<Scalar> Px) const override {
    operator()(x, Px);
  }

  /// @brief Concurrent AMG solve by a pool of workers. Only supported for column vectors.
  ///
  /// @param[in] x Input column vector.
  /// @param[out] Px Output column vector.
  /// @param[in] data Parallel information for each worker.
  ///
  /// @note It's the responsibility of the caller to ensure each worker is allocated to a different
  /// thread to prevent deadlocks, which can be accomplished through @ref
  /// TaskScheduler::BatchEnqueueOnAvailableWorkers.
  void ConcurrentSolve(
      ColumnVectorView<Scalar const> x,
      ColumnVectorView<Scalar> Px,
      ParallelWorkerInfo const& data) const override;

  /// @brief Return how many times each worker waits on @c data.barrier in one
  /// @ref ConcurrentSolve call.
  [[nodiscard]] int NumConcurrentSolveBarriers() const;

  constexpr PreconditionerType GetType() const override {
    return kType;
  }

  /// @brief Update the projected matrices from a block sparse matrix.
  ///
  /// @param[in] A Novel block sparse matrix to build the preconditioner upon. Its scalar type must
  /// match the preconditioner's `Scalar` (modulo top-level `const`) and its block size must equal
  /// @ref kDofsPerNode.
  ///
  /// @note The current implementation does not update the aggregation step. It assumes that the
  /// sparsity pattern of A remains unchanged.
  template <IsBlockSparseMatrix MatA>
  void Update(MatA const& A);

  /// @brief Update the projected matrices from a sparse matrix.
  ///
  /// @param[in] A Novel sparse matrix to build the preconditioner upon. Its scalar type must match
  /// the preconditioner's `Scalar` (modulo top-level `const`). Only valid when `kDofsPerNode == 1`.
  ///
  /// @note The current implementation does not update the aggregation step. It assumes that the
  /// sparsity pattern of A remains unchanged.
  template <IsSparseMatrix MatA>
  void Update(MatA const& A) {
    static_assert(
        kDofsPerNode == 1,
        "AMGPrec::Update(SparseMatrix) is only supported for kDofsPerNode == 1.");
    static_assert(
        std::is_same_v<std::remove_const_t<typename MatA::Scalar>, Scalar>,
        "AMGPrec::Update: matrix Scalar must match the preconditioner Scalar.");
    Update(AsBlockSparseMatrixConstView(A));
  }

  template <bool kWithInitialX = false, typename VectorIn, typename VectorOut>
  void VCycle(VectorIn const& b, VectorOut& x, int s) const;

 protected:
  /// @brief Return the current relaxation factor for the smoother at a level.
  ///
  /// @param[in] level Zero-based smoothed level. Level 0 is the finest level. The direct coarsest
  /// solve has no relaxation factor.
  [[nodiscard]] Scalar GetRelaxationFactor(int level) const {
    MOCHI_ASSERT_VERBOSE(
        level >= 0 && level < isize(_relaxationFactors), "Invalid AMG relaxation level.");
    return _relaxationFactors[level];
  }

  DynamicArray<AMGLevel<Scalar, kDofsPerNode>> _coarsenings = {};
  using SmootherType = typename std::variant<
      BlockJacobiPrec<Scalar, kDofsPerNode, /*kIsSymmetric*/ true>,
      ColoredSSORPrec<BlockSparseMatrixView<Scalar const, kDofsPerNode, int const, int const>>>;
  DynamicArray<SmootherType> _relaxOps = {};
  RowMatrix<Scalar> _coarseInverse;
  BlockSparseMatrixView<Scalar const, kDofsPerNode, int const, int const> _Af;
  AMGOptions<Scalar> _options = {};
  DynamicArray<Scalar> _relaxationFactors = {};
  Workspace _workspace = {};

  template <bool kWithInitialX, typename VectorIn, typename VectorOut>
  void VCycle(VectorIn const& b, VectorOut& x, int s, ParallelWorkerInfo const& data) const;

  /// @brief Return whether @ref Solve, @ref operator() and @ref ConcurrentSolve run
  /// @ref FusedFineCycle.
  [[nodiscard]] bool UsesFusedFineCycle() const {
    return _options.smoother == Smoother::BlockJacobi && _options.numPreSmoothingSteps == 1 &&
        _options.numPostSmoothingSteps == 1;
  }

  /**
   * @brief @ref VCycle at level 0 from a zero initial guess, specialized to one block Jacobi pre-
   * and post-smoothing step.
   *
   * @details Drops two of the barriers of @ref VCycle. Pre-smoothing from zero, y = omega D^{-1} b,
   * is row-local, so it needs no barrier before it. The post-smoothing residual reads the corrected
   * iterate from a separate buffer, so the smoothed result can be written to @p x without waiting
   * for the other workers to finish reading it.
   *
   * @param[in] b RHS.
   * @param[out] x Output vector.
   * @param[in] data Parallel worker info.
   */
  void FusedFineCycle(
      ColumnVectorView<Scalar const> b,
      ColumnVectorView<Scalar> x,
      ParallelWorkerInfo const& data) const;

  /**
   * @brief Coarse-grid correction e = P h on this worker's rows, where h is @ref VCycle at level
   * s + 1 applied to Pt (b - A_s x).
   *
   * @details Uses three barriers, the first of which publishes @p x. @p e is written last, so it
   * may alias the r/z workspace that the coarse V-cycle uses.
   *
   * @param[in] b RHS at level @p s.
   * @param[in] x Iterate at level @p s.
   * @param[out] e Correction at level @p s.
   * @param[in] s Level.
   * @param[in] data Parallel worker info.
   */
  template <typename VectorB, typename VectorX>
  void CoarseCorrection(
      VectorB const& b,
      VectorX const& x,
      ColumnVectorView<Scalar> e,
      int s,
      ParallelWorkerInfo const& data) const;

  /**
   * @brief Smoothing.
   *
   * @param[in] numIter Number of smoothing iterations
   * @param[in] b RHS to smooth out
   * @param[out] x Output vector
   * @param r Temporary vector
   * @param z Temporary vector
   * @param[in] s Level
   * @param[in] data Parallel worker info
   */
  template <bool kWithInitialX, typename VectorIn, typename VectorOut, typename VectorTmp>
  void Smoothing(
      int numIter,
      VectorIn const& b,
      VectorOut& x,
      VectorTmp& r,
      VectorTmp& z,
      int s,
      ParallelWorkerInfo const& data) const;

  void MakeCoarsenings();

  void DefineRelaxations();

  void Initialize();

  void ValidateOptions() const;

  void RefreshRelaxationFactors();

  void FactorCoarsest();

  void CreateWorkspace();
};

namespace details {

template <typename Scalar>
void SetXorShiftRandom(ColumnVectorView<Scalar> out) {
  constexpr uint32_t kSeed = 0x9E3779B9u;
  auto generator = XorShift32Generator(kSeed);
  constexpr auto kScale = Scalar(2) / static_cast<Scalar>(std::numeric_limits<uint32_t>::max());
  for (int i = 0; i < out.Rows(); ++i) {
    out(i) = static_cast<Scalar>(generator()) * kScale - Scalar(1);
  }
}

/// @brief Estimate the spectral radius of D^{-1} A using the power method with Rayleigh quotient.
///
/// @details The power iteration uses D-normalization.
///
/// @param[in] A Matrix operator. Must support Apply(v, Av) and Rows().
/// @param[in] invDiag Diagonal preconditioner D^{-1}. Must support Solve(in, out).
/// @param[in] numIterations Number of power iterations. Must be positive.
/// @param[in,out] v Initial vector. Must have the same size as A.Rows().
/// @return Finite-iteration Rayleigh-quotient estimate of lambda_max(D^{-1} A), or 0 if the
/// estimate is not positive and finite (logs a warning).
///
/// @note The D-normalized Rayleigh quotient interpretation assumes A and D are SPD.
/// @note For SPD A and SPD D in exact arithmetic, this Rayleigh quotient is no larger than the true
/// largest eigenvalue. In finite precision, treat it as an estimate rather than a certificate.
template <typename Scalar, typename MatrixOp, typename PrecOp>
Scalar EstimateSpectralRadiusPowerMethod(
    MatrixOp const& A,
    PrecOp const& invDiag,
    int numIterations,
    ColumnVectorView<Scalar> v) {
  static_assert(!std::is_const_v<Scalar>, "Scalar type should not be const");
  MOCHI_ASSERT_VERBOSE(numIterations > 0, "Number of power iterations must be positive.");
  MOCHI_ASSERT_VERBOSE(v.Rows() == A.Rows(), "Initial vector size must match matrix rows.");
  Matrix<Scalar> W(v.Rows(), 2);
  auto Av = W.Col(0);
  auto invDAv = W.Col(1);
  // Power iteration: 1 step to normalize + numIterations iterations.
  for (int iter = 0; iter <= numIterations; ++iter) {
    Apply(A, v, Av);
    invDiag.Solve(Av, invDAv);
    auto dNormSqr = invDAv.Dot(Av);
    if (!IsFinite(dNormSqr) || dNormSqr <= Scalar(0)) {
      MOCHI_LOG_WARNING("Could not normalize vector for AMG spectral estimate.");
      return Scalar(0);
    }
    v = invDAv * (Scalar(1.0) / Sqrt(dNormSqr));
  }
  // Rayleigh quotient: lambda = v^T * A * v (v is D-normalized)
  Apply(A, v, Av);
  Scalar const maxEstimate = v.Dot(Av);
  if (!IsFinite(maxEstimate) || maxEstimate <= 0) {
    MOCHI_LOG_WARNING(
        "Could not estimate a positive finite largest eigenvalue for AMG relaxation factor.");
    return Scalar(0);
  }
  return maxEstimate;
}

/// @brief Estimate the spectral radius of D^{-1} A using PCG and the Lanczos tridiagonal matrix.
///
/// @details Runs preconditioned CG on the system (A, D) with zero initial guess and a random RHS.
/// The Lanczos/Ritz interpretation assumes A and D are SPD. The CG coefficients (alpha, beta)
/// define a symmetric tridiagonal matrix T whose eigenvalues (Ritz values) approximate the
/// eigenvalues of D^{-1} A. Runs at most min(numIterations, A.Rows()) iterations unless the
/// represented residual becomes exactly zero first. Non-positive or non-finite recurrence values
/// make the estimate fail. The largest eigenvalue of T is computed via @ref
/// SelfAdjointEigenDecomposition.
///
/// @param[in] A Matrix operator. Must support Apply(v, Av) and Rows().
/// @param[in] invDiag Diagonal preconditioner D^{-1}. Must support Solve(in, out).
/// @param[in] numIterations Requested PCG iteration budget, clamped to the scalar dimension of A.
/// The completed iteration count determines the dimension of T.
/// @return Deterministic finite-iteration Ritz estimate of lambda_max(D^{-1} A), or 0 if the
/// estimate fails.
///
/// @note The Lanczos interpretation assumes A and D are SPD. The returned value is used to choose a
/// damping factor. It is not guaranteed to be an upper bound on the true largest eigenvalue.
template <typename MatrixOp, typename PrecOp>
auto EstimateSpectralRadius(MatrixOp const& A, PrecOp const& invDiag, int numIterations) {
  using Scalar = typename MatrixOp::NonConstScalar;
  MOCHI_ASSERT_VERBOSE(numIterations > 0, "Number of iterations must be positive.");

  int const n = A.Rows();
  if (n <= 0) {
    return Scalar(0);
  }

  int const maxIterations = Min(numIterations, n);
  // Workspace: columns are r, z, p, Ap
  Matrix<Scalar> W(n, 4);
  auto r = W.Col(0);
  auto z = W.Col(1);
  auto p = W.Col(2);
  auto Ap = W.Col(3);

  // Storage for the CG coefficients
  DynamicArray<Scalar> alphas;
  DynamicArray<Scalar> betas;
  alphas.reserve(maxIterations);
  betas.reserve(maxIterations - 1);

  // Initial residual r_0 is random (zero initial guess)
  SetXorShiftRandom(r);

  // z_0 = D^{-1} r_0
  invDiag.Solve(r, z);
  Scalar rTz = z.Dot(r);
  if (!IsFinite(rTz) || rTz <= Scalar(0)) {
    MOCHI_LOG_WARNING(
        "Could not estimate the largest eigenvalue when building AMG (non-positive or non-finite initial r^T z).");
    return Scalar(0);
  }

  // p_0 = z_0
  p = z;

  int m = 0;
  for (int iter = 0; iter < maxIterations; ++iter) {
    Apply(A, p, Ap);
    Scalar const pTAp = p.Dot(Ap);
    if (!IsFinite(pTAp) || pTAp <= Scalar(0)) {
      MOCHI_LOG_WARNING(
          "Could not estimate the largest eigenvalue when building AMG (non-positive or non-finite p^T A p).");
      return Scalar(0);
    }
    Scalar const alpha = rTz / pTAp;
    if (!IsFinite(alpha) || alpha <= Scalar(0)) {
      MOCHI_LOG_WARNING(
          "Could not estimate the largest eigenvalue when building AMG (non-positive or non-finite alpha).");
      return Scalar(0);
    }
    alphas.push_back(alpha);
    ++m;
    if (iter + 1 == maxIterations) {
      break;
    }

    // r_{i+1} = r_i - alpha * A * p
    r -= alpha * Ap;

    // z_{i+1} = D^{-1} r_{i+1}
    invDiag.Solve(r, z);
    Scalar const rTzNew = z.Dot(r);
    if (!IsFinite(rTzNew)) {
      MOCHI_LOG_WARNING(
          "Could not estimate the largest eigenvalue when building AMG (non-finite updated r^T z).");
      return Scalar(0);
    }
    if (rTzNew <= Scalar(0)) {
      auto const rValues = r.GetConstSpan();
      if (!IsFinite(rValues)) {
        MOCHI_LOG_WARNING(
            "Could not estimate the largest eigenvalue when building AMG (non-finite updated residual).");
        return Scalar(0);
      }
      if (rTzNew == Scalar(0) && MaxAbs(rValues) == Scalar(0)) {
        break;
      }
      MOCHI_LOG_WARNING(
          "Could not estimate the largest eigenvalue when building AMG (non-positive updated r^T z with nonzero residual).");
      return Scalar(0);
    }
    Scalar const beta = rTzNew / rTz;
    if (!IsFinite(beta) || beta <= Scalar(0)) {
      MOCHI_LOG_WARNING(
          "Could not estimate the largest eigenvalue when building AMG (non-positive or non-finite beta).");
      return Scalar(0);
    }
    betas.push_back(beta);
    rTz = rTzNew;

    // p_{i+1} = z_{i+1} + beta * p_i
    p = z + beta * p;
  }

  MOCHI_ASSERT_VERBOSE(m > 0, "Expected at least one PCG iteration.");

  // Build the symmetric tridiagonal matrix T from the CG coefficients.
  // T(j,j)   = 1/alpha_j + beta_{j-1}/alpha_{j-1}
  // T(j,j+1) = sqrt(beta_j) / alpha_j
  auto T = Matrix<Scalar>::Zero(m, m);
  Scalar const invAlpha0 = Scalar(1) / alphas[0];
  if (!IsFinite(invAlpha0) || invAlpha0 <= Scalar(0)) {
    MOCHI_LOG_WARNING("Could not build a positive finite AMG Ritz matrix.");
    return Scalar(0);
  }
  T(0, 0) = invAlpha0;
  for (int j = 1; j < m; ++j) {
    Scalar const invAlpha = Scalar(1) / alphas[j];
    Scalar const invAlphaPrev = Scalar(1) / alphas[j - 1];
    Scalar const betaOverAlpha = betas[j - 1] * invAlphaPrev;
    Scalar const diag = invAlpha + betaOverAlpha;
    Scalar const offDiag = Sqrt(betas[j - 1]) * invAlphaPrev;
    if (invAlpha <= Scalar(0) || betaOverAlpha <= Scalar(0) || !IsFinite(diag) ||
        !IsFinite(offDiag)) {
      MOCHI_LOG_WARNING("Could not build a positive finite AMG Ritz matrix.");
      return Scalar(0);
    }
    T(j, j) = diag;
    T(j, j - 1) = offDiag;
    T(j - 1, j) = offDiag;
  }

  // Eigendecomposition of T
#if MOCHI_USE_EIGEN
  Matrix<Scalar> V(m, m);
  ColumnVector<Scalar> D(m);
  SelfAdjointEigenDecomposition(AsConstView(T), V, D);
  // Eigenvalues are in ascending order. The largest is the last.
  Scalar const maxEstimate = D(m - 1, 0);
#else
  ColumnVector<Scalar> w(T.Rows());
  SetXorShiftRandom(AsView(w));
  krylov::IdentityPrec<Scalar> N(T);
  constexpr int kMinFallbackPowerIterations = 16;
  auto const maxEstimate = EstimateSpectralRadiusPowerMethod<Scalar>(
      T, N, Max(kMinFallbackPowerIterations, T.Rows()), w);
#endif
  if (!IsFinite(maxEstimate) || maxEstimate <= 0) {
    MOCHI_LOG_WARNING(
        "Could not estimate a positive finite largest eigenvalue for AMG relaxation factor.");
    return Scalar(0);
  }
  return maxEstimate;
}

} // namespace details

template <typename Scalar, int kDofsPerNode>
template <
    template <typename, typename...> typename Storage,
    typename InputIdx,
    typename InputScalar>
AMGPrec<Scalar, kDofsPerNode>::AMGPrec(
    BlockSparseMatrix<InputScalar, kDofsPerNode, InputIdx, InputIdx, Storage> const& A,
    AMGOptions<Scalar> const& options)
    : _Af(AsConstView(A)), _options(options) {
  static_assert(std::is_same_v<InputScalar const, Scalar const>, "Incompatible scalar types");
  static_assert(
      std::same_as<std::remove_const_t<InputIdx>, int>, "AMGPrec requires int matrix indices.");
  Initialize();
}

template <typename Scalar, int kDofsPerNode>
template <
    template <typename, typename...> typename Storage,
    typename InputIdx,
    typename InputScalar>
AMGPrec<Scalar, kDofsPerNode>::AMGPrec(
    SparseMatrix<InputScalar, InputIdx, InputIdx, Storage> const& A,
    AMGOptions<Scalar> const& options)
    : _options(options) {
  static_assert(std::is_same_v<InputScalar const, Scalar const>, "Incompatible scalar types");
  static_assert(
      std::same_as<std::remove_const_t<InputIdx>, int>, "AMGPrec requires int matrix indices.");
  //
  if constexpr (kDofsPerNode == 1) {
    BlockSparseMatrixView<Scalar const, 1, InputIdx const, InputIdx const> Aview{
        A.Cols(), Span{A.Pointers()}, Span{A.Indices()}, Span{A.Values()}};
    _Af.Reset(Aview);
    Initialize();
  } else {
    MOCHI_ASSERT(false, "AMG not implemented for this type");
  }
}

template <typename Scalar, int kDofsPerNode>
template <typename VectorIn, typename VectorOut>
void AMGPrec<Scalar, kDofsPerNode>::operator()(VectorIn const& x, VectorOut&& Px) const {
  ConcurrentSolve(x, Px, ParallelWorkerInfo{0, 1, 0, Px.Rows(), ParallelBarrier(1)});
}

template <typename Scalar, int kDofsPerNode>
void AMGPrec<Scalar, kDofsPerNode>::ConcurrentSolve(
    ColumnVectorView<Scalar const> x,
    ColumnVectorView<Scalar> Px,
    ParallelWorkerInfo const& data) const {
  Preconditioner<Scalar>::ValidateInputOutput(_Af.Rows(), x, Px);
  if (UsesFusedFineCycle()) {
    FusedFineCycle(x, Px, data);
    return;
  }
  Px.MiddleRows(data.rBegin, data.rEnd - data.rBegin).SetZero();
  VCycle<false>(x, Px, 0, data);
}

template <typename Scalar, int kDofsPerNode>
template <bool kWithInitialX, typename VectorIn, typename VectorOut, typename VectorTmp>
void AMGPrec<Scalar, kDofsPerNode>::Smoothing(
    int numIter,
    VectorIn const& b,
    VectorOut& x,
    VectorTmp& r,
    VectorTmp& z,
    int s,
    ParallelWorkerInfo const& data) const {
  auto const rowBegin = data.rBegin;
  auto const rowEnd = data.rEnd;
  auto const numRows = rowEnd - rowBegin;

  for (int iter = 0; iter < numIter; ++iter) {
    if (iter == 0 && !kWithInitialX) {
      r.MiddleRows(rowBegin, numRows) = b.MiddleRows(rowBegin, numRows);
    } else {
      data.BarrierWait(); // Wait for 'x' to be up-to-date.
      if (s == 0) {
        _Af.ApplyToRange(x, r, rowBegin, rowEnd);
      } else {
        _coarsenings[s - 1].PtAP.ApplyToRange(x, r, rowBegin, rowEnd);
      }
      r.MiddleRows(rowBegin, numRows) =
          b.MiddleRows(rowBegin, numRows) - r.MiddleRows(rowBegin, numRows);
    }
    data.BarrierWait(); // Wait for 'r' to be up-to-date (and prevent 'x' from being modified before
                        // ApplyToRange is complete).
    switch (_options.smoother) {
      default:
      case Smoother::BlockJacobi: {
        std::get<0>(_relaxOps[s]).ConcurrentSolve(r, z, data);
        break;
      }
      case Smoother::SSOR: {
        std::get<1>(_relaxOps[s]).ConcurrentSolve(r, z, data);
        break;
      }
    }
    x.MiddleRows(rowBegin, numRows) += _relaxationFactors[s] * z.MiddleRows(rowBegin, numRows);
  }
}

template <typename Scalar, int kDofsPerNode>
template <bool kWithInitialX, typename VectorIn, typename VectorOut>
void AMGPrec<Scalar, kDofsPerNode>::VCycle(VectorIn const& b, VectorOut& x, int s) const {
  VCycle<kWithInitialX>(b, x, s, ParallelWorkerInfo{0, 1, 0, x.Rows(), ParallelBarrier(1)});
}

template <typename Scalar, int kDofsPerNode>
template <bool kWithInitialX, typename VectorIn, typename VectorOut>
void AMGPrec<Scalar, kDofsPerNode>::VCycle(
    VectorIn const& b,
    VectorOut& x,
    int s,
    ParallelWorkerInfo const& data) const {
  MOCHI_ASSERT_VERBOSE(b.Rows() == x.Rows() && b.Cols() == x.Cols(), "Inconsistent dimensions.");
  MOCHI_ASSERT_VERBOSE(
      data.rBegin >= 0 && data.rBegin <= data.rEnd && data.rEnd <= x.Rows(), "Invalid row range.");
  MOCHI_ASSERT(b.Cols() == 1, "VCycle is only supported for column vectors.");
  MOCHI_ASSERT(
      (data.rBegin % kDofsPerNode == 0) && (data.rEnd % kDofsPerNode == 0),
      "Start and end rows must be a multiple of the number of DoFs per node.");

  auto const numRows = data.rEnd - data.rBegin;
  if (s == isize(_coarsenings)) {
    x.MiddleRows(data.rBegin, numRows) = _coarseInverse.MiddleRows(data.rBegin, numRows) * b;
    return;
  }

  auto const n = x.Rows(); // Fine size
  ColumnVectorView<Scalar> r(_workspace.data.get() + _workspace.rOffset, n);
  ColumnVectorView<Scalar> z(_workspace.data.get() + _workspace.zOffset, n);

  Smoothing<kWithInitialX>(_options.numPreSmoothingSteps, b, x, r, z, s, data);

  CoarseCorrection(b, x, r, s, data);
  x.MiddleRows(data.rBegin, numRows) += r.MiddleRows(data.rBegin, numRows);
  //
  if (_options.smoother == Smoother::ApproximateJacobi) {
    // TODO(T185403857): Parallelize TransposeApply.
    if (data.workerId == 0) {
      ColumnVectorView<Scalar const> h(
          _workspace.data.get() + _workspace.hOffset[s], _coarsenings[s].PtAP.Rows());
      _coarsenings[s].PtA.TransposeApply(h, z);
    }
    //
    // Next lines are equivalent to
    // Smoothing<false>(1, "-z", x, r, z, s);
    //
    data.BarrierWait(); // Wait for 'z' to be up-to-date.
    std::get<0>(_relaxOps[s]).ConcurrentSolve(z, r, data);
    x.MiddleRows(data.rBegin, numRows) -=
        _relaxationFactors[s] * r.MiddleRows(data.rBegin, numRows);
  } else {
    Smoothing<true>(_options.numPostSmoothingSteps, b, x, r, z, s, data);
  }
}

template <typename Scalar, int kDofsPerNode>
void AMGPrec<Scalar, kDofsPerNode>::FusedFineCycle(
    ColumnVectorView<Scalar const> b,
    ColumnVectorView<Scalar> x,
    ParallelWorkerInfo const& data) const {
  MOCHI_ASSERT_VERBOSE(UsesFusedFineCycle(), "Fused cycle requires 1+1 block Jacobi smoothing.");

  auto const& smoother = std::get<0>(_relaxOps.front());
  auto const omega = _relaxationFactors.front();
  auto const n = x.Rows();
  auto const rowBegin = data.rBegin;
  auto const rowEnd = data.rEnd;
  auto const numRows = rowEnd - rowBegin;
  ColumnVectorView<Scalar> u(_workspace.data.get() + _workspace.rOffset, n);
  ColumnVectorView<Scalar> r(_workspace.data.get() + _workspace.zOffset, n);

  // 'x' holds the pre-smoothed iterate y until post-smoothing.
  smoother.ConcurrentSolve(b, x, data);
  x.MiddleRows(rowBegin, numRows) *= omega;
  CoarseCorrection(b, x, u, 0, data);
  u.MiddleRows(rowBegin, numRows) += x.MiddleRows(rowBegin, numRows);
  data.BarrierWait(); // Wait for 'u' to be up-to-date.
  _Af.ApplyToRange(u, r, rowBegin, rowEnd);
  r.MiddleRows(rowBegin, numRows) =
      b.MiddleRows(rowBegin, numRows) - r.MiddleRows(rowBegin, numRows);
  smoother.ConcurrentSolve(r, x, data);
  x.MiddleRows(rowBegin, numRows) =
      u.MiddleRows(rowBegin, numRows) + omega * x.MiddleRows(rowBegin, numRows);
}

template <typename Scalar, int kDofsPerNode>
template <typename VectorB, typename VectorX>
void AMGPrec<Scalar, kDofsPerNode>::CoarseCorrection(
    VectorB const& b,
    VectorX const& x,
    ColumnVectorView<Scalar> e,
    int s,
    ParallelWorkerInfo const& data) const {
  auto const nodeBegin = data.rBegin / kDofsPerNode;
  auto const nodeEnd = data.rEnd / kDofsPerNode;

  // Uniform division of rows in the coarser level among workers.
  // TODO(T290274539): Divide by the stored values of PtA and the restriction instead; preliminary
  // analysis on synthetic meshes found both gains and losses.
  auto const N = _coarsenings[s].PtAP.Rows(); // Coarse size
  MOCHI_ASSERT_VERBOSE(N % kDofsPerNode == 0, "Inconsistent coarse size.");
  auto const numCoarseNodes = N / kDofsPerNode;
  auto const coarseNodeBegin = (data.workerId * numCoarseNodes) / data.numWorkers;
  auto const coarseNodeEnd = ((data.workerId + 1) * numCoarseNodes) / data.numWorkers;
  auto const coarseRowBegin = coarseNodeBegin * kDofsPerNode;
  auto const coarseRowEnd = coarseNodeEnd * kDofsPerNode;
  auto const numCoarseRows = coarseRowEnd - coarseRowBegin;

  ColumnVectorView<Scalar> g(_workspace.data.get() + _workspace.gOffset[s], N);
  ColumnVectorView<Scalar> h(_workspace.data.get() + _workspace.hOffset[s], N);

  data.BarrierWait(); // Wait for 'x' to be up-to-date.
  _coarsenings[s].T.RestrictToNodeRange(b, g, coarseNodeBegin, coarseNodeEnd);
  _coarsenings[s].PtA.ApplyToRange(x, h, coarseRowBegin, coarseRowEnd);
  g.MiddleRows(coarseRowBegin, numCoarseRows) -= h.MiddleRows(coarseRowBegin, numCoarseRows);
  //
  h.MiddleRows(coarseRowBegin, numCoarseRows).SetZero();
  data.BarrierWait(); // Wait for 'g' and 'h' to be up-to-date.
  if (data.workerId == 0) {
    // TODO(T185403857): Use VCycle with a subset of the workers.
    VCycle<false>(g, h, s + 1, ParallelWorkerInfo{0, 1, 0, N, ParallelBarrier(1)});
  }
  //
  data.BarrierWait(); // Wait for 'h' to be up-to-date.
  _coarsenings[s].T.InterpolateToNodeRange(h, e, nodeBegin, nodeEnd);
}

template <typename Scalar, int kDofsPerNode>
int AMGPrec<Scalar, kDofsPerNode>::NumConcurrentSolveBarriers() const {
  if (UsesFusedFineCycle()) {
    // CoarseCorrection's three barriers, plus FusedFineCycle's wait for 'u'.
    return 4;
  }
  int const numPre = _options.numPreSmoothingSteps;
  int const numPost = _options.numPostSmoothingSteps;
  int numBarriers = 0;
  if (numPre > 0) {
    // The first pre-smoothing step waits once. Later steps also publish the previous x.
    numBarriers += 2 * numPre - 1;
  }
  // CoarseCorrection's three barriers.
  numBarriers += 3;
  switch (_options.smoother) {
    case Smoother::ApproximateJacobi:
      // Publish the worker-0 transpose result.
      ++numBarriers;
      break;
    case Smoother::BlockJacobi:
      // Each post-smoothing step publishes x and then r.
      numBarriers += 2 * numPost;
      break;
    case Smoother::SSOR: {
      auto const& finestSmoother = std::get<1>(_relaxOps.front());
      // Post-smoothing has the same outer x/r barriers as block Jacobi.
      numBarriers += 2 * numPost;
      // Every smoothing step also executes the finest-level colored SSOR barriers.
      numBarriers += (numPre + numPost) * finestSmoother.NumConcurrentSolveBarriers();
      break;
    }
  }
  return numBarriers;
}

template <typename Scalar, int kDofsPerNode>
template <IsBlockSparseMatrix MatA>
void AMGPrec<Scalar, kDofsPerNode>::Update(MatA const& A) {
  static_assert(
      MatA::kBlockSize == kDofsPerNode,
      "AMGPrec::Update: matrix block size must equal kDofsPerNode.");
  static_assert(
      std::is_same_v<typename MatA::NonConstScalar, Scalar>,
      "AMGPrec::Update: matrix Scalar must match the preconditioner Scalar.");
  // Expensive check to verify the sparsity is unchanged
  MOCHI_ASSERT_VERBOSE(A.Pointers() == _Af.Pointers());
  MOCHI_ASSERT_VERBOSE(A.Indices() == _Af.Indices());
  //
  // The current implementation does not update the aggregation step
  //
  for (int i = 0; i < _coarsenings.size(); ++i) {
    auto& [T, PtA, PtAP] = _coarsenings[i];
    if (i == 0) {
      krylov::details::SparseMatProduct(T.Pt, A, PtA);
    } else {
      krylov::details::SparseMatProduct(T.Pt, _coarsenings[i - 1].PtAP, PtA);
    }
    krylov::details::SparseMatProduct(PtA, T.P, PtAP);
  }
  _Af.Reset(AsConstView(A));
  //
  FactorCoarsest();
  DefineRelaxations();
  RefreshRelaxationFactors();
}

template <typename Scalar, int kDofsPerNode>
void AMGPrec<Scalar, kDofsPerNode>::FactorCoarsest() {
  auto const& coarsest = _coarsenings.back().PtAP;
  int const numBlocks = coarsest.BlockRows();
  int const numDofs = coarsest.Rows();
  //
  // TODO Explore whether the conversion routines in `matrix_conversions.h` can be re-used.
  //
  auto dense = RowMatrix<Scalar>::Zero(numDofs, numDofs);
  for (int i = 0; i < numBlocks; ++i) {
    auto row = coarsest.Values(i);
    auto indices = coarsest.Indices(i);
    for (int j = 0; j < isize(indices); ++j) {
      dense.template Block<kDofsPerNode, kDofsPerNode>(
          i * kDofsPerNode, indices[j] * kDofsPerNode, kDofsPerNode, kDofsPerNode) = row[j];
    }
  }
  //
  // Enforce the symmetry for the coarsest matrix
  // TODO Explore why the coarsest matrix is not symmetric
  //
  for (int ii = 0; ii < dense.Rows(); ++ii) {
    for (int jj = 0; jj < ii; ++jj) {
      dense(ii, jj) = dense(jj, ii);
    }
  }
  _coarseInverse = SymInverse(dense);
}

template <typename Scalar, int kDofsPerNode>
void AMGPrec<Scalar, kDofsPerNode>::MakeCoarsenings() {
  _coarsenings.clear();
  _coarsenings.reserve(4);
  _coarsenings.emplace_back(details::Coarsen(_Af, _options.prolongationSmoothingWeight));
  auto currentSize = _coarsenings.back().PtAP.BlockRows();
  while (currentSize > 1) {
    auto& last = _coarsenings.back();
    auto&& nextLevel = details::Coarsen(last.PtAP, _options.prolongationSmoothingWeight);
    if ((nextLevel.PtAP.BlockRows() == currentSize) || (nextLevel.PtAP.BlockRows() == 1)) {
      break;
    }
    currentSize = nextLevel.PtAP.BlockRows();
    _coarsenings.emplace_back(nextLevel);
  }
}

template <typename Scalar, int kDofsPerNode>
void AMGPrec<Scalar, kDofsPerNode>::DefineRelaxations() {
  _relaxOps.clear();
  _relaxOps.reserve(_coarsenings.size() + 1);
  switch (_options.smoother) {
    default:
    case Smoother::ApproximateJacobi:
    case Smoother::BlockJacobi: {
      _relaxOps.emplace_back(std::in_place_index<0>, _Af);
      for (int i = 0; i < _coarsenings.size(); ++i) {
        _relaxOps.emplace_back(std::in_place_index<0>, _coarsenings[i].PtAP);
      }
      break;
    }
    case Smoother::SSOR: {
      _relaxOps.emplace_back(std::in_place_index<1>, AsConstView(_Af));
      for (int i = 0; i < _coarsenings.size(); ++i) {
        _relaxOps.emplace_back(std::in_place_index<1>, AsConstView(_coarsenings[i].PtAP));
      }
      break;
    }
  }
}

template <typename Scalar, int kDofsPerNode>
void AMGPrec<Scalar, kDofsPerNode>::Initialize() {
  ValidateOptions();
  if (_options.smoother == Smoother::ApproximateJacobi) {
    MOCHI_LOG_WARNING_ONCE("Smoother::ApproximateJacobi is not verified. Use at your own risk.");
  }
  MakeCoarsenings();
  DefineRelaxations();
  RefreshRelaxationFactors();
  FactorCoarsest();
  CreateWorkspace();
}

template <typename Scalar, int kDofsPerNode>
void AMGPrec<Scalar, kDofsPerNode>::ValidateOptions() const {
  MOCHI_ASSERT_VERBOSE(IsFinite(_options.relaxationFactor), "Relaxation factor must be finite.");
  MOCHI_ASSERT_VERBOSE(
      IsFinite(_options.spectralRadiusSafetyFactor) &&
          _options.spectralRadiusSafetyFactor > Scalar(0),
      "Spectral radius safety factor (AMGOptions::spectralRadiusSafetyFactor) must be positive and finite.");
  MOCHI_ASSERT_VERBOSE(
      _options.spectralRadiusMaxIters > 0,
      "Number of spectral radius estimate iterations (AMGOptions::spectralRadiusMaxIters) must be positive.");
  MOCHI_ASSERT_VERBOSE(
      _options.numPreSmoothingSteps >= 0,
      "Number of pre-smoothing steps (AMGOptions::numPreSmoothingSteps) must not be negative.");
  MOCHI_ASSERT_VERBOSE(
      _options.numPostSmoothingSteps >= 0,
      "Number of post-smoothing steps (AMGOptions::numPostSmoothingSteps) must not be negative.");
  MOCHI_ASSERT_VERBOSE(
      IsFinite(_options.prolongationSmoothingWeight),
      "Prolongation smoothing weight (AMGOptions::prolongationSmoothingWeight) must be finite.");
}

template <typename Scalar, int kDofsPerNode>
void AMGPrec<Scalar, kDofsPerNode>::RefreshRelaxationFactors() {
  int const numSmoothingLevels = isize(_coarsenings);
  bool const autoRelaxation = _options.relaxationFactor < Scalar(0);
  _relaxationFactors.clear();
  _relaxationFactors.resize(
      numSmoothingLevels, autoRelaxation ? kDefaultRelaxationFactor : _options.relaxationFactor);
  if (!autoRelaxation || _options.smoother != Smoother::BlockJacobi) {
    return;
  }

  // Estimate lambda_max(D^{-1} A) separately for each smoothed level.
  for (int level = 0; level < numSmoothingLevels; ++level) {
    auto const estimate = (level == 0)
        ? details::EstimateSpectralRadius(
              _Af, std::get<0>(_relaxOps[level]), _options.spectralRadiusMaxIters)
        : details::EstimateSpectralRadius(
              _coarsenings[level - 1].PtAP,
              std::get<0>(_relaxOps[level]),
              _options.spectralRadiusMaxIters);
    if (estimate == Scalar(0)) {
      MOCHI_LOG_WARNING(
          "AMG auto relaxation factor skipped at level %d: Spectral-radius estimate was not positive and finite. Using the default relaxation factor.",
          level);
      continue;
    }

    // Weighted Jacobi is stable for 0 < omega < 2 / lambda_max(D^{-1} A), assuming A and its block
    // diagonal D are SPD. We choose the standard AMG smoothing value omega = 4 / (3 lambda_max),
    // where lambda_max = lambda_max(D^{-1} A), not the stability limit: this minimizes
    // max |1 - omega lambda| on the model high-frequency interval [lambda_max / 2, lambda_max]. The
    // safety factor makes the choice more conservative because the Ritz value is only an estimate
    // and is computed independently on each smoothed operator.
    auto const relaxationFactor =
        Scalar(4.0 / 3.0) / (_options.spectralRadiusSafetyFactor * estimate);
    if (!IsFinite(relaxationFactor) || relaxationFactor <= Scalar(0)) {
      MOCHI_LOG_WARNING(
          "AMG auto relaxation factor skipped at level %d: Computed relaxation factor was not positive and finite. Using the default relaxation factor.",
          level);
      continue;
    }
    _relaxationFactors[level] = relaxationFactor;
  }
}

template <typename Scalar, int kDofsPerNode>
void AMGPrec<Scalar, kDofsPerNode>::CreateWorkspace() {
  MOCHI_ASSERT_VERBOSE(
      _workspace.gOffset.empty() && _workspace.hOffset.empty() && !_coarsenings.empty());
  size_t wsSize = 0;
  _workspace.rOffset = wsSize;
  wsSize += _Af.Rows();
  _workspace.zOffset = wsSize;
  wsSize += _Af.Rows();
  //
  _workspace.gOffset.reserve(_coarsenings.size());
  _workspace.hOffset.reserve(_coarsenings.size());
  for (auto const& coarsening : _coarsenings) {
    _workspace.gOffset.push_back(wsSize);
    wsSize += coarsening.PtAP.Rows();
    _workspace.hOffset.push_back(wsSize);
    wsSize += coarsening.PtAP.Rows();
  }
  //
  _workspace.data = std::make_unique<Scalar[]>(wsSize);
}

} // namespace mochi::krylov
