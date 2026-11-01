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

#include <mochi_core/linear_algebra/block_sparse_matrix.h>
#include <mochi_core/linear_algebra/block_sparse_matrix_utils.h>
#include <mochi_core/linear_algebra/krylov/amg/amg_prec.h>
#include <mochi_core/linear_algebra/krylov/amg/coarsening.h>
#include <mochi_core/linear_algebra/krylov/amg/rigid_body_modes_amg_prec.h>
#include <mochi_core/linear_algebra/krylov/pcg.h>
#include <mochi_core/linear_algebra/sparse_matrix.h>
#include <mochi_core/linear_algebra/sparse_matrix_utils.h>
#include <mochi_core/linear_algebra/utils/matrix_conversions.h>
#include <mochi_core/test/linear_algebra_test_helpers.h>
#include <mochi_core/test/log_suppression.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/array_utils.h>
#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/container_utils.h>
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/graph.h>
#include <mochi_core/utils/log.h>
#include <mochi_core/utils/span.h>
#include <mochi_core/utils/task_scheduler.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

using namespace mochi;

namespace test::amg {

template <typename Fn>
static bool ParallelFor(int numWorkers, Fn const& forEach) {
  TaskSemaphore sem;
  TaskScheduler::BatchTaskFn task = [sem, &forEach](int workerIdx, int /*numWorkers*/) {
    TaskScheduler::PushLocalSingleThreadedMode();
    forEach(workerIdx);
    TaskScheduler::PopLocalSingleThreadedMode();
    sem.Done();
  };
  TaskScheduler scheduler(numWorkers);
  if (scheduler.BatchEnqueueOnAvailableWorkers(
          sem,
          std::move(task),
          /*minWorkers*/ numWorkers,
          /*targetWorkers*/ numWorkers,
          /*includeSelf*/ true) == numWorkers) {
    sem.Wait();
    return true;
  } else {
    return false;
  }
}

// Whether two vectors are finite and agree to 10 epsilon of the first one's largest entry.
[[nodiscard]] static bool NearEqualVectors(
    ColumnVector<real> const& a,
    ColumnVector<real> const& b) {
  return IsFinite(a.GetConstSpan()) && IsFinite(b.GetConstSpan()) &&
      mochi::test::NearEqualMatrices(
             a, b, 10 * std::numeric_limits<real>::epsilon() * MaxAbs(a.GetConstSpan()));
}

template <typename PrecType>
static void TestConcurrentSolve(PrecType const& p, int numRows, int blockSize) {
  EXPECT_TRUE(numRows % blockSize == 0);
  auto const numBlocks = numRows / blockSize;

  ColumnVector<real> x(numRows), Px1(numRows), Px2(numRows);
  bool tested = false;
  for (auto numWorkers : {1, 2, 3, 4, 5, 6}) {
    if (numWorkers > TaskScheduler::GetNumSupportedLogicalProcessors()) {
      continue;
    }

    ParallelBarrier barrier(numWorkers);
    x.SetRandom(numWorkers);
    Px1.SetRandom(numWorkers + 1);
    Px2.SetRandom(numWorkers + 2);
    bool const success = ParallelFor(numWorkers, [&](int workerId) {
      auto rBegin = blockSize * ((numBlocks * workerId) / numWorkers);
      auto rEnd = blockSize * ((numBlocks * (workerId + 1)) / numWorkers);
      auto workerBarrier = barrier; // Worker copy.
      p.ConcurrentSolve(x, Px2, {workerId, numWorkers, rBegin, rEnd, workerBarrier});
    });
    if (success) {
      p.Solve(x, Px1);
      EXPECT_TRUE(NearEqualVectors(Px1, Px2));
      tested = true;
    }
  }
  EXPECT_TRUE(tested);
}

[[nodiscard]] static Matrix<real> MakeRelaxationFactorStressMatrix(int n, real alternatingWeight) {
  Matrix<real> A(n, n);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      real const wi = (i % 2 == 0) ? 1.0_r : -1.0_r;
      real const wj = (j % 2 == 0) ? 1.0_r : -1.0_r;
      A(i, j) = 1.0_r + alternatingWeight * wi * wj;
    }
    A(i, i) += 0.1_r;
  }
  return A;
}

[[nodiscard]] static ColumnVector<real> MakeAlternatingVector(int n) {
  ColumnVector<real> v(n);
  for (int i = 0; i < n; ++i) {
    v(i, 0) = (i % 2 == 0) ? 1.0_r : -1.0_r;
  }
  return v;
}

[[nodiscard]] static auto MakeWeightedTridiagonalBlockSparseMatrix(int n, real offDiagonalWeight) {
  auto A = Matrix<real>::Zero(n, n);
  for (int i = 0; i < n; ++i) {
    A(i, i) = 2.0_r;
    if (i > 0) {
      A(i, i - 1) = -offDiagonalWeight;
    }
    if (i + 1 < n) {
      A(i, i + 1) = -offDiagonalWeight;
    }
  }
  return ToBlockSparseMatrix<1>(A, true);
}

[[nodiscard]] static auto MakeSpectralEstimateFailureMatrix(int n, real offDiagonalWeight) {
  auto A = MakeWeightedTridiagonalBlockSparseMatrix(n, offDiagonalWeight);
  for (auto& value : A.Values()) {
    value = -value;
  }
  return A;
}

// For the normalized 3-point Laplacian, the eigenvalues of D^{-1}A are
// 1 - cos(k*pi/(n + 1)), so lambda_max = 1 + cos(pi/(n + 1)).
[[nodiscard]] static real Normalized3PtLaplacianLambdaMax(int n) {
  return 1.0_r + std::cos(kPI / real(n + 1));
}

static void ExpectSpectralRadiusEstimateBoundsLambdaMax(real estimate, int n, real safetyFactor) {
  real const lambdaMax = Normalized3PtLaplacianLambdaMax(n);
  EXPECT_LE(lambdaMax, safetyFactor * estimate);
  EXPECT_LE(estimate, lambdaMax * (1.0_r + 1.0e-4_r));
}

// The AMGs' damping-estimate RHS, so that estimates and damping factors match theirs.
[[nodiscard]] static ColumnVector<real> RandomRhs(int numRows) {
  ColumnVector<real> rhs(numRows);
  krylov::details::SetXorShiftRandom(AsView(rhs));
  return rhs;
}

template <typename MatrixType>
[[nodiscard]] static real EstimateAutoRelaxationFactor(
    MatrixType const& A,
    krylov::AMGOptions<real> const& options) {
  krylov::BlockJacobiPrec<real, 1> invDiag(A);
  auto const estimate = krylov::details::EstimateSpectralRadius(
      A, invDiag, options.spectralRadiusMaxIters, AsConstView(RandomRhs(A.Rows())));
  EXPECT_GT(estimate, 0_r);
  return real(4.0 / 3.0) / (options.spectralRadiusSafetyFactor * estimate);
}

class ZeroAfterFirstSolvePrec {
 public:
  template <typename Input, typename Output>
  void Solve(Input const& r, Output&& z) const {
    if (_numSolves++ == 0) {
      z = r;
    } else {
      z.SetZero();
    }
  }

 private:
  mutable int _numSolves = 0;
};

class OverflowingBetaPrec {
 public:
  template <typename Input, typename Output>
  void Solve(Input const& r, Output&& z) const {
    real target = 1_r / 64_r;
    if (_numSolves++ > 0) {
      target = std::numeric_limits<real>::max() / 16_r;
    }
    int maxIndex = 0;
    for (int i = 1; i < r.Rows(); ++i) {
      if (Abs(r(i)) > Abs(r(maxIndex))) {
        maxIndex = i;
      }
    }
    z.SetZero();
    z(maxIndex) = target / r(maxIndex);
  }

 private:
  mutable int _numSolves = 0;
};

class RitzOverflowOp {
 public:
  explicit RitzOverflowOp(int numRows) : _numRows(numRows) {}

  int Rows() const {
    return _numRows;
  }

  bool IsFirstApply() const {
    return _numApplies++ == 0;
  }

 private:
  int _numRows;
  mutable int _numApplies = 0;
};

template <typename Input, typename Output>
void Apply(RitzOverflowOp const& A, Input const& p, Output&& Ap) {
  if (!A.IsFirstApply()) {
    Ap = p;
    return;
  }
  int maxIndex = 0;
  for (int i = 1; i < p.Rows(); ++i) {
    if (Abs(p(i)) > Abs(p(maxIndex))) {
      maxIndex = i;
    }
  }
  Ap.SetZero();
  Ap(maxIndex) = (std::numeric_limits<real>::max() / 4_r) / p(maxIndex);
}

class RitzOverflowPrec {
 public:
  template <typename Input, typename Output>
  void Solve(Input const& r, Output&& z) const {
    int maxIndex = 0;
    for (int i = 1; i < r.Rows(); ++i) {
      if (Abs(r(i)) > Abs(r(maxIndex))) {
        maxIndex = i;
      }
    }
    z.SetZero();
    if (_numSolves++ == 0) {
      z(maxIndex) = 1_r / r(maxIndex);
      return;
    }
    real const rNormSqr = r.Dot(r);
    z = r * (8_r / rNormSqr);
  }

 private:
  mutable int _numSolves = 0;
};

class AMG1Access : public krylov::AMGPrec<real, 1> {
 public:
  using krylov::AMGPrec<real, 1>::AMGPrec;
  using krylov::AMGPrec<real, 1>::GetRelaxationFactor;

  auto const& GetCoarsenedMatrixForRelaxationLevel(int level) const {
    MOCHI_ASSERT_VERBOSE(level > 0 && level <= isize(_coarsenings), "Invalid AMG level.");
    return _coarsenings[level - 1].PtAP;
  }
};

TEST(AMG, ConcurrentSolveBarriersReflectOptions) {
  auto A = MakeWeightedTridiagonalBlockSparseMatrix(15, 0.5_r);
  auto numBarriers = [&](krylov::AMGOptions<real> const& options) {
    auto suppressWarnings = mochi::test::SuppressLogWarning();
    return krylov::AMGPrec<real, 1>(A, options).NumConcurrentSolveBarriers();
  };

  krylov::AMGOptions<real> options;
  EXPECT_EQ(4, numBarriers(options));

  options.numPostSmoothingSteps = 2;
  EXPECT_EQ(8, numBarriers(options));

  options.numPreSmoothingSteps = 2;
  options.numPostSmoothingSteps = 1;
  EXPECT_EQ(8, numBarriers(options));

  options.numPreSmoothingSteps = 0;
  options.numPostSmoothingSteps = 0;
  EXPECT_EQ(3, numBarriers(options));

  options = {};
  options.smoother = krylov::Smoother::ApproximateJacobi;
  options.numPostSmoothingSteps = 7;
  EXPECT_EQ(5, numBarriers(options));

  options = {};
  options.smoother = krylov::Smoother::SSOR;
  krylov::ColoredSSORPrec<decltype(A)> finestSmoother(A);
  EXPECT_EQ(6 + 2 * finestSmoother.NumConcurrentSolveBarriers(), numBarriers(options));
}

template <typename MatrixType>
static void ExpectFusedFineCycleMatchesGenericVCycle(MatrixType const& A) {
  krylov::AMGPrec<real, MatrixType::kBlockSize> prec(A);
  ColumnVector<real> b(A.Rows()), fused(A.Rows()), generic(A.Rows());
  b.SetRandom(1);
  prec(b, fused);
  generic.SetZero();
  prec.template VCycle<false>(b, generic, 0);
  EXPECT_TRUE(mochi::test::NearEqualMatrices(generic, fused));
}

TEST(AMG, FusedFineCycleMatchesGenericVCycle) {
  ExpectFusedFineCycleMatchesGenericVCycle(MakeWeightedTridiagonalBlockSparseMatrix(15, 0.5_r));
  ExpectFusedFineCycleMatchesGenericVCycle(mochi::test::MakeBlockSparseMatrix<real, 3>(4, 4, 4));
}

TEST(AMG, ConcurrentSolveForEachCycle) {
  auto const A = mochi::test::MakeBlockSparseMatrix<real, 3>(4, 4, 4);
  auto suppressWarnings = mochi::test::SuppressLogWarning();
  // The defaults run FusedFineCycle; the others run the generic VCycle.
  krylov::AMGOptions<real> defaults, twoPostSteps, ssor, approximateJacobi;
  twoPostSteps.numPostSmoothingSteps = 2;
  ssor.smoother = krylov::Smoother::SSOR;
  approximateJacobi.smoother = krylov::Smoother::ApproximateJacobi;
  for (auto const& options : {defaults, twoPostSteps, ssor, approximateJacobi}) {
    TestConcurrentSolve(krylov::AMGPrec<real, 3>(A, options), A.Rows(), 3);
  }
}

// Dense matrix of the linear map apply(x, y): y = M x.
template <typename ApplyFn>
[[nodiscard]] static Matrix<double> MakeDense(int numRows, int numCols, ApplyFn const& apply) {
  Matrix<double> M(numRows, numCols);
  ColumnVector<real> x(numCols);
  ColumnVector<real> y(numRows);
  for (int j = 0; j < numCols; ++j) {
    x.SetZero();
    x(j) = 1_r;
    apply(x, y);
    for (int i = 0; i < numRows; ++i) {
      M(i, j) = y(i);
    }
  }
  return M;
}

// Largest absolute entry, or NaN, which fails every comparison, if an entry is not finite.
[[nodiscard]] static double MaxAbs(Matrix<double> const& A) {
  Span<double const> const entries(A.Data(), A.StorageSize());
  return IsFinite(entries) ? mochi::MaxAbs(entries) : std::numeric_limits<double>::quiet_NaN();
}

[[nodiscard]] static double MaxAbsDifference(Matrix<double> const& A, Matrix<double> const& B) {
  return MaxAbs(Matrix<double>(A - B));
}

// Smallest pivot of the LDL^T factorization of a symmetric matrix, or its first pivot that is not
// positive: positive exactly when the matrix is positive definite. Reads the lower triangle.
[[nodiscard]] static double MinPivot(Matrix<double> A) {
  double minPivot = std::numeric_limits<double>::infinity();
  for (int k = 0; k < A.Rows(); ++k) {
    if (!(A(k, k) > 0.0)) {
      return A(k, k);
    }
    minPivot = Min(minPivot, A(k, k));
    for (int i = k + 1; i < A.Rows(); ++i) {
      for (int j = k + 1; j <= i; ++j) {
        A(i, j) -= A(i, k) * (A(j, k) / A(k, k));
      }
    }
  }
  return minPivot;
}

// Inputs of an AMG preconditioner: the matrix and, for RigidBodyModesAMGPrec, the node positions.
template <typename MatrixType>
struct AMGProblem {
  MatrixType A;
  DynamicArray<real> referencePositions = {};
  DynamicArray<real> displacements = {};
};

template <typename MatrixType>
[[nodiscard]] static Matrix<double> ToDense(MatrixType const& A) {
  return ToMatrix(StaticCast<BlockSparseMatrix<double, MatrixType::kBlockSize>>(A));
}

// AMGPrec and RigidBodyModesAMGPrec, with dense views of their hierarchies.
template <int kBlockSize, typename MatrixType = BlockSparseMatrix<real, kBlockSize>>
class AMGPrecUnderTest : public krylov::AMGPrec<real, kBlockSize> {
 public:
  using Base = krylov::AMGPrec<real, kBlockSize>;
  static constexpr int kDofsPerNode = kBlockSize;
  static constexpr bool kExactlySymmetricCoarseMatrices = false;

  explicit AMGPrecUnderTest(AMGProblem<MatrixType> const& problem) : Base(problem.A) {}

  void Update(AMGProblem<MatrixType> const& problem) {
    Base::Update(problem.A);
  }

  [[nodiscard]] int NumCoarseLevels() const {
    return isize(this->_coarsenings);
  }

  [[nodiscard]] Matrix<double> LevelMatrix(int s) const {
    return s == 0 ? ToDense(this->_Af) : ToDense(this->_coarsenings[s - 1].PtAP);
  }

  [[nodiscard]] Matrix<double> StoredPtA(int s) const {
    return ToDense(this->_coarsenings[s].PtA);
  }

  // A zero coarsest inverse drops the coarse correction to the coarsest level.
  [[nodiscard]] int NumAcceptedCoarseLevels() const {
    return NumCoarseLevels() - (mochi::MaxAbs(MakeConstSpan(this->_coarseInverse)) == 0 ? 1 : 0);
  }

  void SolveCoarsest(ColumnVector<real> const& b, ColumnVector<real>& x) const {
    x = this->_coarseInverse * b;
  }

  void Interpolate(int s, ColumnVector<real> const& h, ColumnVector<real>& e) const {
    this->_coarsenings[s].T.Interpolate(h, e);
  }

  void Restrict(int s, ColumnVector<real> const& u, ColumnVector<real>& g) const {
    this->_coarsenings[s].T.Restrict(u, g);
  }
};

class RigidModesAMGUnderTest : public krylov::RigidBodyModesAMGPrec<real> {
 public:
  using Base = krylov::RigidBodyModesAMGPrec<real>;
  using Base::_decoupled;
  using Base::_levels;
  static constexpr int kDofsPerNode = 3;
  static constexpr bool kExactlySymmetricCoarseMatrices = true;

  explicit RigidModesAMGUnderTest(AMGProblem<BlockSparseMatrix<real, 3>> const& problem)
      : Base(
            problem.A,
            MakeConstSpan(problem.referencePositions),
            MakeConstSpan(problem.displacements)) {}

  void Update(AMGProblem<BlockSparseMatrix<real, 3>> const& problem) {
    Base::Update(
        problem.A, MakeConstSpan(problem.referencePositions), MakeConstSpan(problem.displacements));
  }

  [[nodiscard]] int NumCoarseLevels() const {
    return isize(_levels);
  }

  [[nodiscard]] Matrix<double> LevelMatrix(int s) const {
    return s == 0 ? ToDense(_A0) : ToDense(_levels[s - 1].A);
  }

  [[nodiscard]] Matrix<double> StoredPtA(int s) const {
    return s == 0 ? Matrix<double>() : ToDense(_levels[s - 1].PtA);
  }

  [[nodiscard]] int NumAcceptedCoarseLevels() const {
    return _numAcceptedCoarseLevels;
  }

  void SolveCoarsest(ColumnVector<real> const& b, ColumnVector<real>& x) const {
    x = _coarsestInverse * b;
  }

  void Interpolate(int s, ColumnVector<real> const& h, ColumnVector<real>& e) const {
    auto const& T = _levels[s].transfer;
    if (s == 0) {
      krylov::details::InterpolateRigidModes<3>(T, h, e, 0, T.numNodes);
    } else {
      krylov::details::InterpolateRigidModes<6>(T, h, e, 0, T.numNodes);
    }
  }

  void Restrict(int s, ColumnVector<real> const& u, ColumnVector<real>& g) const {
    auto const& T = _levels[s].transfer;
    if (s == 0) {
      krylov::details::RestrictRigidModes<3>(T, u, g, 0, T.numAggregates);
    } else {
      krylov::details::RestrictRigidModes<6>(T, u, g, 0, T.numAggregates);
    }
  }
};

// X is an accurate inverse of A: |A X - I| <= tol |A| |X| entrywise, as for a backward stable
// inverse. With translationsOnly, A has 6 DoFs per node, and X is such an inverse of its
// translation block, over the first 3 DoFs of every node, and zero elsewhere.
template <typename MatA, typename MatX>
static void
ExpectAccurateInverse(MatA const& A, MatX const& X, double tol, bool translationsOnly = false) {
  int const n = A.Rows();
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (translationsOnly && (i % 6 >= 3 || j % 6 >= 3)) {
        EXPECT_EQ(double(X(i, j)), 0.0);
        continue;
      }
      double residual = (i == j) ? -1.0 : 0.0;
      double scale = 0.0;
      for (int k = 0; k < n; ++k) {
        residual += double(A(i, k)) * double(X(k, j));
        scale += Abs(double(A(i, k)) * double(X(k, j)));
      }
      EXPECT_LE(Abs(residual), tol * scale);
    }
  }
}

// PCG iterations to a relative residual of 1e-5 for A x = A v from zero, or kMaxPCGIterations
// without convergence.
constexpr int kMaxPCGIterations = 200;
template <typename MatrixType, typename Prec>
[[nodiscard]] static int NumPCGIterations(MatrixType const& A, Prec const& prec) {
  ColumnVector<real> v(A.Rows());
  v.SetRandom(7);
  ColumnVector<real> const b = A * v;
  ColumnVector<real> x = ColumnVector<real>::Zero(A.Rows());
  krylov::StatusResidualL2<krylov::UsualDot, real> stopper{1e-5_r, 0_r, 1e10_r};
  auto const apply = [&prec](auto const& r, auto& z) { prec.Solve(r, z); };
  auto const status = krylov::PCG(A, b, x, apply, kMaxPCGIterations, stopper);
  return status.convergence == LinearSolverConvergenceStatus::Converged ? status.numIterDone
                                                                        : kMaxPCGIterations;
}

// Properties every AMG preconditioner has, given the coarse level kept after stiffening, or 0 if
// none, and whether the coarsest level is solved on its translations alone. Returns its PCG
// iteration count.
template <typename AMG, typename MatrixType>
static int ExpectAMGProperties(
    AMGProblem<MatrixType> const& problem,
    int stiffenedLevel = 0,
    bool translationsOnly = false) {
  AMG prec(problem);
  // The rounding of sums of up to a few hundred products in real.
  real const tol = 1000 * std::numeric_limits<real>::epsilon();
  // The levels down to the first one that still fails the numerical checks after any repair, below
  // which none is formed.
  int const numFormedLevels = Min(prec.NumAcceptedCoarseLevels() + 1, prec.NumCoarseLevels());
  for (int s = 0; s < numFormedLevels; ++s) {
    // The restriction is the interpolation's transpose. The stored P^T A and, unless stiffened, the
    // coarse matrix are the Galerkin products, with J on the coarse DoFs that do not interpolate,
    // the rotations of a 6-DoF node: the identity times the mean diagonal entry of the node's
    // translation block.
    auto const A = prec.LevelMatrix(s);
    auto const coarseA = prec.LevelMatrix(s + 1);
    auto const P = MakeDense(
        A.Rows(), coarseA.Rows(), [&](auto const& h, auto& e) { prec.Interpolate(s, h, e); });
    auto const R = MakeDense(
        coarseA.Rows(), A.Rows(), [&](auto const& u, auto& g) { prec.Restrict(s, u, g); });
    Matrix<double> const Pt = P.Transpose();
    EXPECT_EQ(MaxAbsDifference(R, Pt), 0.0);
    Matrix<double> const PtA = Pt * A;
    // Rounding relative to the largest product, since the sums cancel on rigid motions.
    double const productScale = MaxAbs(P) * MaxAbs(A);
    auto const storedPtA = prec.StoredPtA(s);
    if (storedPtA.Rows() > 0) {
      EXPECT_LE(MaxAbsDifference(storedPtA, PtA), tol * productScale);
    }
    Matrix<double> galerkin = PtA * P;
    DynamicArray<int> columns;
    for (int c = 0; c < P.Cols(); ++c) {
      if (P.Col(c).Dot(P.Col(c)) > 0.0) {
        columns.push_back(c);
      } else {
        int const node = c - c % 6;
        galerkin(c, c) +=
            (galerkin(node, node) + galerkin(node + 1, node + 1) + galerkin(node + 2, node + 2)) /
            3.0;
      }
    }
    if (s + 1 == stiffenedLevel) {
      // A stiffened level dominates the exact product: the difference is positive definite, J on
      // the DoFs that do not interpolate.
      EXPECT_GT(MinPivot(coarseA - PtA * P), 0.0);
    } else {
      EXPECT_LE(MaxAbsDifference(coarseA, galerkin), tol * productScale * MaxAbs(P));
    }
    if constexpr (AMG::kExactlySymmetricCoarseMatrices) {
      Matrix<double> const coarseAt = coarseA.Transpose();
      EXPECT_EQ(MaxAbsDifference(coarseA, coarseAt), 0.0);
    }
    // The other columns of P are numerically independent: their normalized Gram matrix is positive
    // definite by more than its rounding.
    Matrix<double> gram(isize(columns), isize(columns));
    for (int c0 = 0; c0 < isize(columns); ++c0) {
      for (int c1 = 0; c1 < isize(columns); ++c1) {
        auto const p0 = P.Col(columns[c0]);
        auto const p1 = P.Col(columns[c1]);
        gram(c0, c1) = p0.Dot(p1) / Sqrt(p0.Dot(p0) * p1.Dot(p1));
      }
    }
    EXPECT_GT(MinPivot(gram), std::numeric_limits<real>::epsilon());
  }
  double symmetryTol = Sqrt(double(std::numeric_limits<real>::epsilon()));
  if (prec.NumAcceptedCoarseLevels() == prec.NumCoarseLevels()) {
    // The coarsest inverse is accurate, on the translation block alone with translationsOnly.
    auto const coarsest = prec.LevelMatrix(prec.NumCoarseLevels());
    int const m = coarsest.Rows();
    auto const inverse = MakeDense(m, m, [&](auto const& u, auto& v) { prec.SolveCoarsest(u, v); });
    ExpectAccurateInverse(coarsest, inverse, tol, translationsOnly);
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < m; ++j) {
        symmetryTol =
            Max(symmetryTol,
                m * std::numeric_limits<real>::epsilon() * Abs(inverse(i, j)) *
                    Sqrt(coarsest(i, i) * coarsest(j, j)));
      }
    }
  }

  // The preconditioner M is positive definite, and symmetric to rounding, which the explicit
  // inverse X of the coarsest matrix C, or of its translation block, amplifies by up to
  // max |X_ij| sqrt(C_ii C_jj). The eigenvalues of M A, at most 1 for a V-cycle with symmetric
  // smoothing and coarse corrections no larger than the Galerkin ones, stay below 1.5:
  // 1.5 M - M A M is positive definite.
  auto const A = prec.LevelMatrix(0);
  int const n = A.Rows();
  auto const M = MakeDense(n, n, [&](auto const& r, auto& z) { prec.Solve(r, z); });
  Matrix<double> const Mt = M.Transpose();
  EXPECT_LE(MaxAbsDifference(M, Mt), symmetryTol * MaxAbs(M));
  EXPECT_GT(MinPivot(M), 0.0);
  Matrix<double> const AM = A * M;
  Matrix<double> const MAM = M * AM;
  Matrix<double> const contraction = 1.5 * M - MAM;
  EXPECT_GT(MinPivot(contraction), 0.0);
  TestConcurrentSolve(prec, n, AMG::kDofsPerNode);
  int const numPCGIterations = NumPCGIterations(problem.A, prec);
  EXPECT_LT(numPCGIterations, kMaxPCGIterations);

  // The preconditioner scales with A: M(4 A) = M(A) / 4, exactly, since 4 is a power of two.
  ColumnVector<real> r(n);
  r.SetRandom(11);
  AMGProblem<MatrixType> scaled = problem;
  for (auto& value : scaled.A.Values()) {
    value *= 4_r;
  }
  ColumnVector<real> z(n);
  ColumnVector<real> zScaled(n);
  prec.Solve(r, z);
  AMG(scaled).Solve(r, zScaled);
  zScaled *= 4_r;
  EXPECT_SPAN_EQ(z.GetConstSpan(), zScaled.GetConstSpan());

  // Update is equivalent to construction: with the diagonal blocks scaled by 2, 3 or 4, which no
  // damping factor can make up for in a stale smoother, then new positions, then the same inputs,
  // then the original ones.
  AMGProblem<MatrixType> updated = problem;
  for (int i = 0; i < updated.A.Rows() / AMG::kDofsPerNode; ++i) {
    updated.A.Values(i)[krylov::details::BlockIndex(updated.A.Indices(i), i)] *= real(2 + i % 3);
  }
  AMGProblem<MatrixType> moved = updated;
  for (int k = 0; k < isize(moved.displacements); ++k) {
    moved.displacements[k] += 0.01_r * Sin(real(k));
  }
  AMGProblem<MatrixType> const* const sequence[] = {&updated, &moved, &moved, &problem};
  for (auto const* inputs : sequence) {
    prec.Update(*inputs);
    AMG const fresh(*inputs);
    ColumnVector<real> zFresh(n);
    prec.Solve(r, z);
    fresh.Solve(r, zFresh);
    EXPECT_SPAN_EQ(z.GetConstSpan(), zFresh.GetConstSpan());
  }
  return numPCGIterations;
}

// Axial springs of unit stiffness between the nodes of each group, plus mass I.
[[nodiscard]] static BlockSparseMatrix<real, 3> MakeSpringMatrix(
    DynamicArray<std::array<double, 3>> const& x,
    DynamicArray<DynamicArray<int>> const& groups,
    real mass) {
  int const n = isize(x);
  auto A = Matrix<real>::Zero(3 * n, 3 * n);
  for (int k = 0; k < 3 * n; ++k) {
    A(k, k) = mass;
  }
  DynamicArray<uint8_t> coupled(n * n, 0);
  for (auto const& group : groups) {
    for (int const i : group) {
      for (int const j : group) {
        if (i >= j || coupled[i * n + j] != 0) {
          continue;
        }
        coupled[i * n + j] = 1;
        double const d[3] = {x[j][0] - x[i][0], x[j][1] - x[i][1], x[j][2] - x[i][2]};
        double const dd = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        for (int r = 0; r < 3; ++r) {
          for (int c = 0; c < 3; ++c) {
            auto const k = static_cast<real>(d[r] * d[c] / dd);
            A(3 * i + r, 3 * i + c) += k;
            A(3 * j + r, 3 * j + c) += k;
            A(3 * i + r, 3 * j + c) -= k;
            A(3 * j + r, 3 * i + c) -= k;
          }
        }
      }
    }
  }
  return ToBlockSparseMatrix<3>(A, /*pruneZeros*/ true);
}

// Rotation by 30 degrees about the axis (1, 2, 3) / sqrt(14).
[[nodiscard]] static std::array<double, 3> RotateAboutSkewAxis(std::array<double, 3> const& x) {
  double const axis[3] = {1.0 / Sqrt(14.0), 2.0 / Sqrt(14.0), 3.0 / Sqrt(14.0)};
  double const c = Cos(kPI / 6.0);
  double const s = Sin(kPI / 6.0);
  double const dot = axis[0] * x[0] + axis[1] * x[1] + axis[2] * x[2];
  double const cross[3] = {
      axis[1] * x[2] - axis[2] * x[1],
      axis[2] * x[0] - axis[0] * x[2],
      axis[0] * x[1] - axis[1] * x[0]};
  std::array<double, 3> y{};
  for (int k = 0; k < 3; ++k) {
    y[k] = c * x[k] + s * cross[k] + (1.0 - c) * dot * axis[k];
  }
  return y;
}

// Anchors node 0 with stiffness 1 / e, e = epsilon / 128.
static void PinNode0(BlockSparseMatrix<real, 3>& A) {
  real const e = std::numeric_limits<real>::epsilon() / 128;
  for (int d = 0; d < 3; ++d) {
    A.Values(0)[krylov::details::BlockIndex(A.Indices(0), 0)](d, d) += 1_r / e;
  }
}

enum class AMGFixture {
  Laplacian1D,
  TwoLaplacians1D,
  TwoLaplacians2D,
  Bar,
  MovedBar,
  FloatingBar,
  HeavierFloatingBar,
  PinnedBar,
  NonFiniteBar,
  Strip,
  MovedStrip,
  CollinearCurrent,
  CollinearReference,
  TenNodes,
  NearlyCollinear,
  CoupledNearlyCollinear,
  RoundsToSingular,
  RoundsToSingularChain,
  TinyScale,
  TinyScaleChain,
};

// Fixtures with positions. Bar: on cubic cells, whose springs are exact in floating point. Moved:
// rotated about a skew axis and translated by 1000 along x. Floating: without the mass term, but
// with mass 8 epsilon at three corner nodes, so that the coarsest matrix, on which the rigid
// motions of the whole bar have that energy, rounds to a singular one, while the springs keep the
// first coarse level's smoother blocks, which are all that its acceptance checks, positive
// definite; the matrix is SPD as represented. Since the bar's translations are among those motions,
// the coarsest translation block rounds to singular too. Heavier floating: mass 24 epsilon there,
// so that AMGPrec's coarsest matrix passes the positive-definiteness check but its explicit inverse
// fails it. Pinned: node 0 anchored with stiffness 1 / e, so that the first coarse block of its
// aggregate rounds to rank 3 until that level is stiffened. Non-finite: a NaN and an infinite
// displacement coordinate at nodes 40 and 41, as from a diverged step, so that the first coarse
// level is not finite, even once stiffened. Collinear: the strip, with its current positions
// collapsed onto a skew line, or its reference positions onto a narrow strip along it, whose
// aggregates' margins, 0.047 to 0.058, lie between 0 and the member-only threshold.
[[nodiscard]] static AMGProblem<BlockSparseMatrix<real, 3>> MakeProblemWithPositions(
    AMGFixture fixture) {
  using enum AMGFixture;
  bool const nearlyCollinear = fixture == NearlyCollinear || fixture == CoupledNearlyCollinear;
  bool const tinyScale = fixture == TinyScale || fixture == TinyScaleChain;
  bool const roundsToSingular = fixture == RoundsToSingular || fixture == RoundsToSingularChain;
  bool const floating = fixture == FloatingBar || fixture == HeavierFloatingBar;
  real const e = std::numeric_limits<real>::epsilon() / 128;
  DynamicArray<std::array<double, 3>> reference;
  DynamicArray<DynamicArray<int>> groups;
  if (fixture == Bar || fixture == MovedBar || floating || fixture == PinnedBar ||
      fixture == NonFiniteBar) {
    auto const [coordinates, tets] =
        mochi::test::CreateMinimalTetMeshUnitGrid({5_r, 1_r, 1_r}, {10, 2, 2});
    for (auto const& x : coordinates) {
      reference.push_back({x[0], x[1], x[2]});
    }
    for (auto const& tet : tets) {
      groups.push_back({tet[0], tet[1], tet[2], tet[3]});
    }
  } else if (nearlyCollinear) {
    // Three nodes, the middle one sqrt(epsilon) off the line of the others, so that the Galerkin
    // product loses the energy of their rotation about it, with a centroid penalty. Coupled: 100
    // sqrt(epsilon) off, 2.9 times the decoupling margin.
    double const offset = (fixture == NearlyCollinear ? 1.0 : 100.0) *
        Sqrt(double(std::numeric_limits<real>::epsilon()));
    reference = {{-1.0, -1.0, 0.0}, {0.0, offset, 0.0}, {1.0, 1.0, 0.0}};
    groups.push_back({0, 1, 2});
  } else if (tinyScale) {
    // Three nearly collinear nodes on the z axis, whose x and y stiffness is 100 times the smallest
    // normal number, so that the energy of the coarse rotation about their line, the last coarse
    // DoF, is subnormal and its reciprocal overflows, while their translations stay normal. Chain:
    // four of these joined into a path, whose first coarse level fails even once stiffened.
    for (int g = 0; g < (fixture == TinyScale ? 1 : 4); ++g) {
      reference.push_back({0.0, 0.0, 3.0 * g - 1.0});
      reference.push_back({0.01, 0.0, 3.0 * g});
      reference.push_back({0.0, 0.0, 3.0 * g + 1.0});
      groups.push_back({3 * g, 3 * g + 1, 3 * g + 2});
      if (g > 0) {
        groups.push_back({3 * g - 1, 3 * g + 2});
      }
    }
  } else if (roundsToSingular) {
    // diag(1, 6e, 6e, 6e) with off-diagonal -e on four nodes, whose coarse matrix rounds to a
    // singular one while its translation block stays positive definite. Chain: ten of these, each
    // with its diagonal-1 node at its centroid, joined into a path and moved as MovedBar, so that
    // the first coarse level passes and the second, a smoothed level, rounds to singular.
    constexpr std::array<double, 2> kSingleOffsets[] = {
        {1.0, 0.0}, {-1.0, 0.0}, {0.0, 1.0}, {0.0, -1.0}};
    constexpr std::array<double, 2> kChainOffsets[] = {
        {0.0, 0.0}, {1.0, 0.0}, {-0.5, 1.0}, {-0.5, -1.0}};
    auto const& offsets = (fixture == RoundsToSingular) ? kSingleOffsets : kChainOffsets;
    for (int g = 0; g < (fixture == RoundsToSingular ? 1 : 10); ++g) {
      for (auto const& offset : offsets) {
        reference.push_back({3.0 * g + offset[0], offset[1], 0.0});
      }
      groups.push_back({4 * g, 4 * g + 1, 4 * g + 2, 4 * g + 3});
      if (g > 0) {
        groups.push_back({4 * g - 1, 4 * g + 3});
      }
    }
  } else if (fixture == TenNodes) {
    // Ten nodes coupled in windows of five consecutive nodes, whose two aggregates have 2 pure rows
    // each, at positions where the truncated smoothed interpolation is rank deficient.
    constexpr double kBeta[] = {0, 0, 2.0 / 9, 2.0 / 7, 1.0 / 3, 2.0 / 3, 5.0 / 7, 7.0 / 9, 1, 1};
    for (int i = 0; i < 10; ++i) {
      reference.push_back({kBeta[i], 0.0, double(i % 2)});
    }
    for (int i = 0; i + 4 < 10; ++i) {
      groups.push_back({i, i + 1, i + 2, i + 3, i + 4});
    }
    // Seven isolated nodes, which form their own aggregates, so that level 1, with 6 rows per
    // aggregate, has more rows than the fine level.
    for (int i = 0; i < 7; ++i) {
      reference.push_back({2.0 + i, 0.0, 0.0});
    }
  } else {
    // A 12 x 3 strip of triangles bent into a quarter arc, with the bending stencil: the nodes of
    // every two triangles that share an edge.
    constexpr int kLength = 12;
    constexpr int kWidth = 3;
    for (int i = 0; i < kLength; ++i) {
      double const angle = (0.5 * kPI * i) / (kLength - 1);
      double const radius = (kLength - 1) / (0.5 * kPI);
      for (int j = 0; j < kWidth; ++j) {
        reference.push_back({radius * Cos(angle), radius * Sin(angle), double(j)});
      }
    }
    DynamicArray<std::array<int, 3>> triangles;
    for (int i = 0; i + 1 < kLength; ++i) {
      for (int j = 0; j + 1 < kWidth; ++j) {
        int const n00 = kWidth * i + j;
        triangles.push_back({n00, n00 + kWidth, n00 + kWidth + 1});
        triangles.push_back({n00, n00 + kWidth + 1, n00 + 1});
      }
    }
    for (auto const& s : triangles) {
      for (auto const& t : triangles) {
        auto const isShared = [&](int i) { return Contains(t, i); };
        if (std::count_if(s.begin(), s.end(), isShared) >= 2) {
          groups.push_back({s[0], s[1], s[2], t[0], t[1], t[2]});
        }
      }
    }
  }
  auto positions = reference;
  if (fixture == MovedBar || fixture == MovedStrip || fixture == RoundsToSingularChain) {
    for (auto& x : positions) {
      x = RotateAboutSkewAxis(x);
      x[0] += 1000.0;
    }
  }
  AMGProblem<BlockSparseMatrix<real, 3>> problem{
      .A = MakeSpringMatrix(positions, groups, floating ? 0_r : 1e-3_r)};
  int const n = isize(reference);
  if (floating) {
    constexpr std::array<double, 3> kCorners[] = {
        {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    real const cornerMass =
        (fixture == FloatingBar ? 8 : 24) * std::numeric_limits<real>::epsilon();
    for (int i = 0; i < n; ++i) {
      if (Contains(kCorners, reference[i])) {
        for (int d = 0; d < 3; ++d) {
          problem.A.Values(i)[krylov::details::BlockIndex(problem.A.Indices(i), i)](d, d) +=
              cornerMass;
        }
      }
    }
  }
  if (fixture == PinnedBar) {
    PinNode0(problem.A);
  }
  if (nearlyCollinear || tinyScale || roundsToSingular) {
    real const tiny = 100 * std::numeric_limits<real>::min();
    auto A = ToMatrix(problem.A);
    for (int i = 0; i < 3 * n; ++i) {
      auto const columns = problem.A.Indices(i / 3);
      for (int j = 0; j < 3 * n; ++j) {
        bool const coupled = Contains(columns, j / 3);
        if (nearlyCollinear) {
          A(i, j) += (i % 3 == j % 3) ? 1_r : 0_r;
        } else if (tinyScale) {
          real const scale = (i % 3 == 2) ? 1_r : tiny;
          A(i, j) = (i % 3 == j % 3 && coupled) ? scale * (real(i == j) + 0.01_r) : 0_r;
        } else if (i == j) {
          A(i, j) = (i / 3) % 4 == 0 ? 1_r : 6 * e;
        } else {
          A(i, j) = (i % 3 == j % 3 && coupled) ? -e : 0_r;
        }
      }
    }
    problem.A = ToBlockSparseMatrix<3>(A, /*pruneZeros*/ true);
  }
  if (fixture == CollinearCurrent || fixture == CollinearReference) {
    for (int i = 0; i < n; ++i) {
      int const k = i / 3;
      double const y = 2.0 * k + (fixture == CollinearReference ? 0.5 * (i % 3) : 0.0);
      (fixture == CollinearCurrent ? positions : reference)[i] = {double(k), y, 3.0 * k};
    }
  }
  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < 3; ++k) {
      problem.referencePositions.push_back(static_cast<real>(reference[i][k]));
      problem.displacements.push_back(static_cast<real>(positions[i][k] - reference[i][k]));
    }
  }
  if (fixture == NonFiniteBar) {
    problem.displacements[3 * 40] = std::numeric_limits<real>::quiet_NaN();
    problem.displacements[3 * 41 + 1] = kInf;
  }
  return problem;
}

// Whether the margin sigma_min / sqrt(lambda) of the stacked blocks B = [I | -[q]x], with
// q = (x_i - centroid) / radius for the given nodes, is at least the given one, where lambda =
// lambda_max([[n, |s|], [|s|, S]]) for n nodes, s = sum q and S = sum |q|^2: whether
// G - margin^2 lambda I is positive definite for the Gram matrix G = B^T B; false for fewer than
// 3 nodes. Also expects lambda to bound the largest eigenvalue of G.
[[nodiscard]] static bool StackedBlocksMarginAtLeast(
    DynamicArray<int> const& nodes,
    DynamicArray<double> const& x,
    double const* centroid,
    double radius,
    double margin) {
  int const n = isize(nodes);
  if (n < 3) {
    return false;
  }
  auto gram = Matrix<double>::Zero(6, 6);
  double s[3] = {0.0, 0.0, 0.0};
  double trace = 0.0;
  for (int const i : nodes) {
    double q[3];
    for (int d = 0; d < 3; ++d) {
      q[d] = (x[3 * i + d] - centroid[d]) / radius;
      s[d] += q[d];
      trace += Sqr(q[d]);
    }
    double const B[3][6] = {
        {1, 0, 0, 0, q[2], -q[1]}, {0, 1, 0, -q[2], 0, q[0]}, {0, 0, 1, q[1], -q[0], 0}};
    for (int r = 0; r < 6; ++r) {
      for (int c = 0; c < 6; ++c) {
        gram(r, c) += B[0][r] * B[0][c] + B[1][r] * B[1][c] + B[2][r] * B[2][c];
      }
    }
  }
  double const lambda =
      0.5 * (n + trace + Sqrt(Sqr(n - trace) + 4.0 * (Sqr(s[0]) + Sqr(s[1]) + Sqr(s[2]))));
  Matrix<double> bound = gram;
  bound *= -1.0;
  for (int k = 0; k < 6; ++k) {
    bound(k, k) += (1.0 + 1e-12) * lambda;
    gram(k, k) -= Sqr(margin) * lambda;
  }
  EXPECT_TRUE(krylov::details::IsPositiveDefinite(bound));
  return krylov::details::IsPositiveDefinite(gram);
}

// RigidBodyModesAMGPrec's own properties: its geometry, weights and decoupled aggregates match
// their definitions, and its interpolation reproduces every rigid motion, except the rotations of
// the rows that interpolate from decoupled aggregates. Returns the numbers of member-only and
// decoupled level-0 aggregates.
[[nodiscard]] static std::pair<int, int> ExpectRigidModesDefinitions(
    AMGProblem<BlockSparseMatrix<real, 3>> const& problem,
    RigidModesAMGUnderTest const& prec) {
  // The coefficients round to real once, and each interpolated entry sums a few products of them;
  // the positions, in double, round at their magnitude.
  real const tol = 50 * std::numeric_limits<real>::epsilon();
  double const positionTol = 50 * std::numeric_limits<double>::epsilon();
  int const n = problem.A.BlockRows();
  DynamicArray<double> const reference(
      problem.referencePositions.begin(), problem.referencePositions.end());
  // Positions and radii of the nodes of each level, from the finest.
  DynamicArray<double> x(3 * n);
  double center[3] = {0.0, 0.0, 0.0};
  for (int k = 0; k < 3 * n; ++k) {
    x[k] = double(problem.referencePositions[k]) + double(problem.displacements[k]);
    center[k % 3] += x[k] / n;
  }
  DynamicArray<double> radii(n, 0.0);
  int numMemberOnly = 0;
  int numDecoupled = 0;
  for (int s = 0; s < prec.NumCoarseLevels(); ++s) {
    auto const& T = prec._levels[s].transfer;
    int const numFine = T.numNodes;
    int const numCoarse = T.numAggregates;
    DynamicArray<int> partition(numFine);
    for (int a = 0; a < numCoarse; ++a) {
      for (int const i : T.members[a]) {
        partition[i] = a;
      }
    }
    // Centroids and RMS radii of the aggregates, computed in double, as in the preconditioner.
    auto const geometry = [&](DynamicArray<double> const& positions,
                              DynamicArray<double>& centroids,
                              DynamicArray<double>& coarseRadii) {
      centroids = DynamicArray<double>(3 * numCoarse, 0.0);
      coarseRadii = DynamicArray<double>(numCoarse, 0.0);
      DynamicArray<int> counts(numCoarse, 0);
      for (int i = 0; i < numFine; ++i) {
        ++counts[partition[i]];
        for (int d = 0; d < 3; ++d) {
          centroids[3 * partition[i] + d] += positions[3 * i + d];
        }
      }
      for (int k = 0; k < 3 * numCoarse; ++k) {
        centroids[k] /= counts[k / 3];
      }
      for (int i = 0; i < numFine; ++i) {
        int const a = partition[i];
        double const* c = &centroids[3 * a];
        coarseRadii[a] += (Sqr(positions[3 * i] - c[0]) + Sqr(positions[3 * i + 1] - c[1]) +
                           Sqr(positions[3 * i + 2] - c[2]) + Sqr(radii[i])) /
            counts[a];
      }
      for (auto& radius : coarseRadii) {
        radius = radius > 0.0 ? Sqrt(radius) : 1.0;
      }
    };
    DynamicArray<double> centroids;
    DynamicArray<double> coarseRadii;
    geometry(x, centroids, coarseRadii);
    for (int k = 0; k < 3 * numCoarse; ++k) {
      EXPECT_NEAR(centroids[k], T.centroids[k], 1e-12 * (1.0 + Abs(centroids[k])));
    }
    for (int a = 0; a < numCoarse; ++a) {
      EXPECT_NEAR(coarseRadii[a], T.radii[a], 1e-12 * coarseRadii[a]);
    }

    // Weights, also in double: smoothed aggregate weights; at level 0, truncated below 0.15 and
    // renormalized, and member-only for the aggregates whose pure rows have a margin below 0.1 at
    // the reference positions.
    auto W = Matrix<double>::Zero(numFine, numCoarse);
    for (int i = 0; i < numFine; ++i) {
      auto const neighbors = (s == 0) ? problem.A.Indices(i) : prec._levels[s - 1].A.Indices(i);
      int const degree = isize(neighbors);
      for (int const j : neighbors) {
        W(i, partition[j]) += (degree == 1) ? 1.0 : (j == i ? 1.0 / 3.0 : 2.0 / 3.0 / (degree - 1));
      }
    }
    // The rows that interpolate from their own aggregate alone, by aggregate.
    auto const pureRows = [&]() {
      DynamicArray<DynamicArray<int>> rows(numCoarse);
      for (int i = 0; i < numFine; ++i) {
        int numNonzeros = 0;
        for (int a = 0; a < numCoarse; ++a) {
          numNonzeros += W(i, a) > 0.0 ? 1 : 0;
        }
        if (numNonzeros == 1) {
          rows[partition[i]].push_back(i);
        }
      }
      return rows;
    };
    if (s == 0) {
      for (int i = 0; i < numFine; ++i) {
        double sum = 0.0;
        for (int a = 0; a < numCoarse; ++a) {
          W(i, a) = W(i, a) >= 0.15 ? W(i, a) : 0.0;
          sum += W(i, a);
        }
        for (int a = 0; a < numCoarse; ++a) {
          W(i, a) /= sum;
        }
      }
      DynamicArray<double> referenceCentroids;
      DynamicArray<double> referenceRadii;
      geometry(reference, referenceCentroids, referenceRadii);
      auto const truncatedPureRows = pureRows();
      for (int a = 0; a < numCoarse; ++a) {
        if (!StackedBlocksMarginAtLeast(
                truncatedPureRows[a],
                reference,
                &referenceCentroids[3 * a],
                referenceRadii[a],
                0.1)) {
          ++numMemberOnly;
          for (int const i : T.members[a]) {
            for (int b = 0; b < numCoarse; ++b) {
              W(i, b) = (b == a) ? 1.0 : 0.0;
            }
          }
        }
      }
      // Rotations are decoupled where the final pure rows are nearly collinear.
      auto const finalPureRows = pureRows();
      for (int a = 0; a < numCoarse; ++a) {
        bool const decoupled = !StackedBlocksMarginAtLeast(
            finalPureRows[a],
            x,
            &centroids[3 * a],
            coarseRadii[a],
            10 * Sqrt(double(std::numeric_limits<real>::epsilon())));
        EXPECT_EQ(decoupled, prec._decoupled[a]);
        numDecoupled += decoupled ? 1 : 0;
      }
    }
    auto actualW = Matrix<double>::Zero(numFine, numCoarse);
    for (int i = 0; i < numFine; ++i) {
      for (int k = T.pointers[i]; k < T.pointers[i + 1]; ++k) {
        actualW(i, T.aggregates[k]) = T.weights[k];
      }
    }
    EXPECT_LE(MaxAbsDifference(W, actualW), 1e-12);

    // Rigid motions, with rotations about the actor's center.
    double const maxCoordinate = mochi::MaxAbs(MakeConstSpan(x));
    int const fineDofs = (s == 0) ? 3 : 6;
    auto const P = MakeDense(fineDofs * numFine, 6 * numCoarse, [&](auto const& h, auto& e) {
      prec.Interpolate(s, h, e);
    });
    for (int motion = 0; motion < 6; ++motion) {
      double t[3] = {0.0, 0.0, 0.0};
      double theta[3] = {0.0, 0.0, 0.0};
      (motion < 3 ? t : theta)[motion % 3] = 1.0;
      auto const rigidMotion = [&](DynamicArray<double> const& positions,
                                   DynamicArray<double> const& nodeRadii,
                                   int numDofs) {
        auto u = Matrix<double>::Zero(numDofs * isize(nodeRadii), 1);
        for (int i = 0; i < isize(nodeRadii); ++i) {
          for (int d = 0; d < 3; ++d) {
            int const d1 = (d + 1) % 3;
            int const d2 = (d + 2) % 3;
            u(numDofs * i + d, 0) = t[d] + theta[d1] * (positions[3 * i + d2] - center[d2]) -
                theta[d2] * (positions[3 * i + d1] - center[d1]);
            if (numDofs == 6) {
              u(numDofs * i + 3 + d, 0) = nodeRadii[i] * theta[d];
            }
          }
        }
        return u;
      };
      auto const fineMotion = rigidMotion(x, radii, fineDofs);
      Matrix<double> interpolated = P * rigidMotion(centroids, coarseRadii, 6);
      if (s == 0 && motion >= 3) {
        for (int i = 0; i < numFine; ++i) {
          for (int k = T.pointers[i]; k < T.pointers[i + 1]; ++k) {
            if (prec._decoupled[T.aggregates[k]]) {
              for (int d = 0; d < 3; ++d) {
                interpolated(3 * i + d, 0) = fineMotion(3 * i + d, 0);
              }
            }
          }
        }
      }
      EXPECT_LE(
          MaxAbsDifference(interpolated, fineMotion),
          tol * MaxAbs(fineMotion) + positionTol * maxCoordinate);
    }
    x = std::move(centroids);
    radii = std::move(coarseRadii);
  }
  return {numMemberOnly, numDecoupled};
}

template <typename Row>
[[nodiscard]] static std::string RowName(testing::TestParamInfo<Row> const& info) {
  return info.param.name;
}

// AMGPrec on a fixture, and whether it keeps the coarse correction to the coarsest level.
struct AMGRow {
  char const* name;
  AMGFixture fixture;
  bool coarsestAccepted;
};

class AMGTest : public testing::TestWithParam<AMGRow> {};

template <typename AMG, typename Problem>
static void ExpectAMGPrecProperties(Problem const& problem, bool coarsestAccepted) {
  ExpectAMGProperties<AMG>(problem);
  AMG const prec(problem);
  EXPECT_EQ(prec.NumAcceptedCoarseLevels() == prec.NumCoarseLevels(), coarsestAccepted);
}

TEST_P(AMGTest, Properties) {
  using enum AMGFixture;
  AMGFixture const fixture = GetParam().fixture;
  bool const coarsestAccepted = GetParam().coarsestAccepted;
  // Dropping the coarsest level warns: only the rows that drop it silence warnings.
  auto const suppressWarnings = mochi::test::SuppressLogWarning();
  EnableLogChannel(LogChannel::Warning, coarsestAccepted);
  if (fixture == Laplacian1D) {
    using Sparse = SparseMatrix<real, int, int>;
    ExpectAMGPrecProperties<AMGPrecUnderTest<1, Sparse>>(
        AMGProblem<Sparse>{.A = mochi::test::Make3ptLaplacianMatrix(15)}, coarsestAccepted);
  } else if (fixture == TwoLaplacians1D || fixture == TwoLaplacians2D) {
    int const ny = (fixture == TwoLaplacians1D) ? 1 : 3;
    auto A = mochi::test::Create2dLaplacianMatrix<real, 2>(fixture == TwoLaplacians1D ? 15 : 7, ny);
    if (ny == 1) {
      for (int i = 0; i < A.Rows(); ++i) {
        A(i, i) = 2_r;
      }
    }
    ExpectAMGPrecProperties<AMGPrecUnderTest<2>>(
        AMGProblem<BlockSparseMatrix<real, 2>>{.A = ToBlockSparseMatrix<2>(A, /*pruneZeros*/ true)},
        coarsestAccepted);
  } else {
    ExpectAMGPrecProperties<AMGPrecUnderTest<3>>(
        MakeProblemWithPositions(fixture), coarsestAccepted);
  }
}

constexpr AMGRow kAMGRows[] = {
    {.name = "Laplacian1D", .fixture = AMGFixture::Laplacian1D, .coarsestAccepted = true},
    {.name = "TwoLaplacians1D", .fixture = AMGFixture::TwoLaplacians1D, .coarsestAccepted = true},
    {.name = "TwoLaplacians2D", .fixture = AMGFixture::TwoLaplacians2D, .coarsestAccepted = true},
    {.name = "Bar", .fixture = AMGFixture::Bar, .coarsestAccepted = true},
    {.name = "Strip", .fixture = AMGFixture::Strip, .coarsestAccepted = true},
    {.name = "FloatingBar", .fixture = AMGFixture::FloatingBar, .coarsestAccepted = false},
    {.name = "HeavierFloatingBar",
     .fixture = AMGFixture::HeavierFloatingBar,
     .coarsestAccepted = false},
};
INSTANTIATE_TEST_SUITE_P(AMG, AMGTest, testing::ValuesIn(kAMGRows), RowName<AMGRow>);

// RigidBodyModesAMGPrec on a fixture: the expected numbers of level-0 aggregates with member-only
// weights and with decoupled rotations, of coarse levels and of accepted coarse levels, the coarse
// level kept after stiffening, if any, whether the coarsest level is solved on its translations
// alone, and the most PCG iterations it may take: on the bar and the strip, about a third above the
// measured counts.
struct RigidModesRow {
  char const* name;
  AMGFixture fixture;
  int numMemberOnly;
  int numDecoupled;
  int numCoarseLevels;
  int numAcceptedCoarseLevels;
  int stiffenedLevel = 0;
  bool translationsOnly = false;
  int maxPCGIterations = kMaxPCGIterations;
};

class RigidModesAMGTest : public testing::TestWithParam<RigidModesRow> {};

TEST_P(RigidModesAMGTest, Properties) {
  auto const& row = GetParam();
  // Only the rows that drop a coarse level log warnings:
  // count them, and forward every other channel.
  int numWarnings = 0;
  LogFn const logFn = GetLogCallback();
  SetLogCallback([&](LogChannel channel, char const* message, char const* file, int line) {
    if (channel == LogChannel::Warning) {
      ++numWarnings;
    } else {
      logFn(channel, message, file, line);
    }
  });
  MOCHI_DEFER(SetLogCallback(logFn));
  auto const problem = MakeProblemWithPositions(row.fixture);
  int const numPCGIterations = ExpectAMGProperties<RigidModesAMGUnderTest>(
      problem, row.stiffenedLevel, row.translationsOnly);
  RigidModesAMGUnderTest const prec(problem);
  auto const [numMemberOnly, numDecoupled] = ExpectRigidModesDefinitions(problem, prec);
  EXPECT_EQ(numMemberOnly, row.numMemberOnly);
  EXPECT_EQ(numDecoupled, row.numDecoupled);
  EXPECT_EQ(prec.NumCoarseLevels(), row.numCoarseLevels);
  EXPECT_EQ(prec.NumAcceptedCoarseLevels(), row.numAcceptedCoarseLevels);
  EXPECT_EQ(prec.NumConcurrentSolveBarriers(), row.numAcceptedCoarseLevels > 0 ? 5 : 1);
  if (row.numAcceptedCoarseLevels == 0) {
    // The smoother-only cycle: x = u + omega D^{-1} (b - A u), with u = omega D^{-1} b and the
    // automatic damping of AMGPrec.
    krylov::BlockJacobiPrec<real, 3, /*kIsSymmetric*/ true> const smoother(problem.A);
    real const omega = krylov::details::AutoRelaxationFactor(
                           problem.A,
                           smoother,
                           krylov::AMGOptions<real>{},
                           0,
                           AsConstView(RandomRhs(problem.A.Rows())))
                           .value_or(krylov::AMGPrec<real, 3>::kDefaultRelaxationFactor);
    ColumnVector<real> b(problem.A.Rows());
    b.SetRandom(3);
    ColumnVector<real> u(b.Rows());
    ColumnVector<real> x(b.Rows());
    smoother.Solve(b, u);
    u *= omega;
    ColumnVector<real> r = problem.A * u;
    r = b - r;
    smoother.Solve(r, x);
    x = u + omega * x;
    ColumnVector<real> z(b.Rows());
    prec.Solve(b, z);
    EXPECT_TRUE(NearEqualVectors(x, z));
  }
  EXPECT_LE(numPCGIterations, row.maxPCGIterations);
  EXPECT_EQ(numWarnings > 0, row.numAcceptedCoarseLevels < row.numCoarseLevels);
}

constexpr RigidModesRow kRigidModesRows[] = {
    {.name = "Bar",
     .fixture = AMGFixture::Bar,
     .numMemberOnly = 5,
     .numDecoupled = 0,
     .numCoarseLevels = 2,
     .numAcceptedCoarseLevels = 2,
     .maxPCGIterations = 16},
    {.name = "MovedBar",
     .fixture = AMGFixture::MovedBar,
     .numMemberOnly = 5,
     .numDecoupled = 0,
     .numCoarseLevels = 2,
     .numAcceptedCoarseLevels = 2,
     .maxPCGIterations = 16},
    {.name = "FloatingBar",
     .fixture = AMGFixture::FloatingBar,
     .numMemberOnly = 5,
     .numDecoupled = 0,
     .numCoarseLevels = 2,
     .numAcceptedCoarseLevels = 1},
    {.name = "PinnedBar",
     .fixture = AMGFixture::PinnedBar,
     .numMemberOnly = 5,
     .numDecoupled = 0,
     .numCoarseLevels = 2,
     .numAcceptedCoarseLevels = 2,
     .stiffenedLevel = 1},
    {.name = "Strip",
     .fixture = AMGFixture::Strip,
     .numMemberOnly = 0,
     .numDecoupled = 0,
     .numCoarseLevels = 1,
     .numAcceptedCoarseLevels = 1,
     .maxPCGIterations = 25},
    {.name = "MovedStrip",
     .fixture = AMGFixture::MovedStrip,
     .numMemberOnly = 0,
     .numDecoupled = 0,
     .numCoarseLevels = 1,
     .numAcceptedCoarseLevels = 1,
     .maxPCGIterations = 25},
    {.name = "CollinearCurrent",
     .fixture = AMGFixture::CollinearCurrent,
     .numMemberOnly = 0,
     .numDecoupled = 3,
     .numCoarseLevels = 1,
     .numAcceptedCoarseLevels = 1},
    {.name = "CollinearReference",
     .fixture = AMGFixture::CollinearReference,
     .numMemberOnly = 3,
     .numDecoupled = 0,
     .numCoarseLevels = 1,
     .numAcceptedCoarseLevels = 1},
    {.name = "TenNodes",
     .fixture = AMGFixture::TenNodes,
     .numMemberOnly = 9,
     .numDecoupled = 7,
     .numCoarseLevels = 2,
     .numAcceptedCoarseLevels = 2},
    {.name = "NearlyCollinear",
     .fixture = AMGFixture::NearlyCollinear,
     .numMemberOnly = 1,
     .numDecoupled = 1,
     .numCoarseLevels = 1,
     .numAcceptedCoarseLevels = 1},
    {.name = "CoupledNearlyCollinear",
     .fixture = AMGFixture::CoupledNearlyCollinear,
     .numMemberOnly = 1,
     .numDecoupled = 0,
     .numCoarseLevels = 1,
     .numAcceptedCoarseLevels = 1},
    {.name = "RoundsToSingular",
     .fixture = AMGFixture::RoundsToSingular,
     .numMemberOnly = 0,
     .numDecoupled = 0,
     .numCoarseLevels = 1,
     .numAcceptedCoarseLevels = 1,
     .translationsOnly = true},
    {.name = "RoundsToSingularChain",
     .fixture = AMGFixture::RoundsToSingularChain,
     .numMemberOnly = 0,
     .numDecoupled = 0,
     .numCoarseLevels = 3,
     .numAcceptedCoarseLevels = 3,
     .stiffenedLevel = 2},
    {.name = "TinyScale",
     .fixture = AMGFixture::TinyScale,
     .numMemberOnly = 1,
     .numDecoupled = 0,
     .numCoarseLevels = 1,
     .numAcceptedCoarseLevels = 1,
     .translationsOnly = true},
    {.name = "TinyScaleChain",
     .fixture = AMGFixture::TinyScaleChain,
     .numMemberOnly = 4,
     .numDecoupled = 0,
     .numCoarseLevels = 2,
     .numAcceptedCoarseLevels = 0},
};
INSTANTIATE_TEST_SUITE_P(
    AMG,
    RigidModesAMGTest,
    testing::ValuesIn(kRigidModesRows),
    RowName<RigidModesRow>);

// Update moves between full acceptance, no accepted coarse level, a dropped coarsest and a
// stiffened level, as construction does.
TEST(AMG, RigidModesAcceptanceTransitions) {
  using enum AMGFixture;
  constexpr std::pair<AMGFixture, int> kSequence[] = {
      {Bar, 2}, {NonFiniteBar, 0}, {FloatingBar, 1}, {PinnedBar, 2}, {Bar, 2}};
  // Dropping a coarse level warns.
  auto const suppressWarnings = mochi::test::SuppressLogWarning();
  DynamicArray<AMGProblem<BlockSparseMatrix<real, 3>>> problems;
  for (auto const& [fixture, numAccepted] : kSequence) {
    problems.push_back(MakeProblemWithPositions(fixture));
  }
  RigidModesAMGUnderTest prec(problems.front());
  for (int k = 1; k < isize(problems); ++k) {
    prec.Update(problems[k]);
    RigidModesAMGUnderTest const fresh(problems[k]);
    EXPECT_EQ(prec.NumAcceptedCoarseLevels(), kSequence[k].second);
    EXPECT_EQ(prec.NumConcurrentSolveBarriers(), fresh.NumConcurrentSolveBarriers());
    int const n = problems[k].A.Rows();
    ColumnVector<real> r(n);
    ColumnVector<real> z(n);
    ColumnVector<real> zFresh(n);
    r.SetRandom(k);
    prec.Solve(r, z);
    fresh.Solve(r, zFresh);
    EXPECT_SPAN_EQ(z.GetConstSpan(), zFresh.GetConstSpan());
    TestConcurrentSolve(prec, n, 3);
  }
}

// With a bound task scheduler, construction and an update with moved positions match the
// single-threaded results. The bar, 2084 x 2 x 2 nodes, is just long enough for each parallel loop
// of the setup over the fine nodes, their weights or their aggregates to split into several tasks.
// Node 0 is pinned as in the pinned bar, so that its first coarse level is stiffened and kept.
TEST(AMG, RigidModesParallelSetup) {
  constexpr Int3 kCells{2083, 1, 1};
  AMGProblem<BlockSparseMatrix<real, 3>> problem{
      .A = mochi::test::MakeBlockSparseMatrix<real, 3>(kCells[0], kCells[1], kCells[2])};
  PinNode0(problem.A);
  // The matrix's nodes, in its order, on unit cells.
  auto const nodes =
      mochi::test::CreateMinimalTetMeshUnitGrid(StaticCast<Real3>(kCells), kCells).first;
  for (auto const& x : nodes) {
    for (int d = 0; d < 3; ++d) {
      problem.referencePositions.push_back(x[d]);
      problem.displacements.push_back(0_r);
    }
  }
  AMGProblem<BlockSparseMatrix<real, 3>> moved = problem;
  for (int k = 0; k < isize(moved.displacements); ++k) {
    moved.displacements[k] = 0.01_r * Sin(real(k));
  }

  int const n = problem.A.Rows();
  ColumnVector<real> r(n);
  r.SetRandom(1);
  // The Solve outputs after construction and after the update.
  auto const solveOutputs = [&]() {
    ColumnVector<real> z(2 * n);
    RigidModesAMGUnderTest prec(problem);
    EXPECT_GT(prec.NumAcceptedCoarseLevels(), 0);
    prec.Solve(r, z.TopRows(n));
    prec.Update(moved);
    EXPECT_GT(prec.NumAcceptedCoarseLevels(), 0);
    prec.Solve(r, z.BottomRows(n));
    return z;
  };
  ColumnVector<real> const serial = solveOutputs();
  TaskScheduler scheduler(2);
  EXPECT_TRUE(NearEqualVectors(serial, solveOutputs()));
}

// The margin of the centered points (+-1, 0, 0) and (0, +-a, 0), normalized by their RMS radius,
// is a / sqrt(1 + a^2): their Gram matrix has eigenvalues 4 a^2 / (1 + a^2), 4 / (1 + a^2) and 4
// (four times), and with s = 0 and S = 4, the bound lambda is 4, the largest. Off their centroid,
// after two shifts, the decision at every margin is that of the Gram matrix of the stacked blocks.
TEST(AMG, RigidModeMargin) {
  using krylov::details::RigidModeMarginAtLeast;
  double const origin[3] = {0.0, 0.0, 0.0};
  DynamicArray<double> q(12);
  double sum[3];
  double sumOuter[6];
  auto const setSums = [&]() {
    Fill(MakeSpan(sum), 0.0);
    Fill(MakeSpan(sumOuter), 0.0);
    for (int i = 0; i < 4; ++i) {
      double const* qi = &q[3 * i];
      for (int d = 0; d < 3; ++d) {
        sum[d] += qi[d];
      }
      sumOuter[0] += qi[0] * qi[0];
      sumOuter[1] += qi[0] * qi[1];
      sumOuter[2] += qi[0] * qi[2];
      sumOuter[3] += qi[1] * qi[1];
      sumOuter[4] += qi[1] * qi[2];
      sumOuter[5] += qi[2] * qi[2];
    }
  };
  for (double const threshold : {0.1, 10 * Sqrt(double(std::numeric_limits<real>::epsilon()))}) {
    // Just above the threshold, just below it, and collinear.
    for (double const ratio : {1.1, 0.9, 0.0}) {
      double const margin = ratio * threshold;
      double const a = margin / Sqrt(1.0 - Sqr(margin));
      double const radius = Sqrt(0.5 * (1.0 + Sqr(a)));
      std::array<double, 3> const points[] = {
          {1.0, 0.0, 0.0}, {-1.0, 0.0, 0.0}, {0.0, a, 0.0}, {0.0, -a, 0.0}};
      for (int i = 0; i < 4; ++i) {
        auto const qi = RotateAboutSkewAxis(
            {points[i][0] / radius, points[i][1] / radius, points[i][2] / radius});
        std::ranges::copy(qi, &q[3 * i]);
      }
      setSums();
      EXPECT_EQ(RigidModeMarginAtLeast(4, sum, sumOuter, threshold), ratio >= 1.0);
      auto const centered = q;
      for (auto const& shift : {std::array{0.1, 0.2, 0.3}, std::array{-0.1, 0.2, 0.3}}) {
        for (int k = 0; k < 12; ++k) {
          q[k] = centered[k] + shift[k % 3];
        }
        setSums();
        for (double m = 1e-7; m < 0.5; m *= 1.01) {
          EXPECT_EQ(
              RigidModeMarginAtLeast(4, sum, sumOuter, m),
              StackedBlocksMarginAtLeast({0, 1, 2, 3}, q, origin, 1.0, m));
        }
      }
    }
  }
  // Points far off their centroid: the shifted translation block is negative, and the Schur
  // complement would pass at a margin below the threshold.
  std::ranges::copy(
      std::array{20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0, 10.0, 10.0, 10.0}, q.begin());
  setSums();
  EXPECT_EQ(
      RigidModeMarginAtLeast(4, sum, sumOuter, 0.1),
      StackedBlocksMarginAtLeast({0, 1, 2, 3}, q, origin, 1.0, 0.1));
  // Fewer than 3 points, and NaN, fail.
  EXPECT_FALSE(RigidModeMarginAtLeast(2, sum, sumOuter, 0.1));
  sum[0] = std::numeric_limits<double>::quiet_NaN();
  sumOuter[0] = sum[0];
  EXPECT_FALSE(RigidModeMarginAtLeast(3, sum, sumOuter, 0.1));
}

// The rounding bound on a hand-built pattern: identity blocks of A on the diagonal and between
// nodes 0 and 1, and coefficients w = 1, r = 1 and, for node i, q the i-th unit axis, with nodes 0
// and 1 interpolating from both aggregates and node 2 from aggregate 0 alone. So n_1 = 2 for both
// aggregates, n_2 = 3, from aggregate 0, and z = |P|^T |A| |P| 1 is (14, 14, 17, 10, 10, 12) and
// (12, 12, 16, 8, 8, 12) from 3 fine DoFs, or (14, 14, 17, 19, 19, 21) and (12, 12, 16, 16, 16, 20)
// from 6. With p = 2 or 3, each shift is at least alpha z and within 0.1% of it; one count more or
// less moves it by over 10%.
TEST(AMG, RigidModesRoundingBound) {
  using Coefficients = krylov::details::RigidModeCoefficients<real>;
  Coefficients const e0{.w = 1_r, .q = {1_r, 0_r, 0_r}, .r = 1_r};
  Coefficients const e1{.w = 1_r, .q = {0_r, 1_r, 0_r}, .r = 1_r};
  Coefficients const e2{.w = 1_r, .q = {0_r, 0_r, 1_r}, .r = 1_r};
  krylov::details::RigidModeTransfer<real> T;
  T.numNodes = 3;
  T.numAggregates = 2;
  T.pointers = {0, 2, 4, 5};
  T.aggregates = {0, 1, 0, 1, 0};
  T.coefficients = {e0, e0, e1, e1, e2};
  T.restrictionPointers = {0, 3, 5};
  T.restrictionNodes = {0, 1, 2, 0, 1};
  T.restrictionCoefficients = {e0, e1, e2, e0, e1};

  // Row a of P^T A has a block for each node of R(a).
  Graph<int, int> const ptAPattern({0, 3, 5}, {0, 1, 2, 0, 1});
  auto const coupled = [](int k) {
    Matrix<real> A(3 * k, 3 * k);
    A.SetIdentity();
    A.Block(0, k, k, k).SetIdentity();
    A.Block(k, 0, k, k).SetIdentity();
    return A;
  };

  double const u = 0.5 * double(std::numeric_limits<real>::epsilon());
  auto const gamma = [u](int m) { return m * u / (1.0 - m * u); };
  auto const expectShift = [&](auto const& A, std::array<std::array<double, 6>, 2> const& z) {
    auto Ac = ToBlockSparseMatrix<6>(Matrix<real>::Zero(12, 12), /*pruneZeros*/ false);
    krylov::details::AddGalerkinRoundingBound(T, A, ptAPattern, Ac);

    int const p = (A.kBlockSize == 3) ? 2 : 3;
    double const gammaG = gamma(2 * p) + gamma(3 * p) + gamma(2 * p) * gamma(3 * p);
    double const alpha = (gammaG + u * (1.0 + gammaG)) / Sqr(1.0 - u);
    for (int a = 0; a < 2; ++a) {
      auto const shift = Ac.Values(a)[krylov::details::BlockIndex(Ac.Indices(a), a)];
      for (int r = 0; r < 6; ++r) {
        EXPECT_GE(shift(r, r), alpha * z[a][r]);
        EXPECT_LE(shift(r, r), 1.001 * alpha * z[a][r]);
      }
    }
  };

  expectShift(
      ToBlockSparseMatrix<3>(coupled(3), /*pruneZeros*/ true),
      {{{14.0, 14.0, 17.0, 10.0, 10.0, 12.0}, {12.0, 12.0, 16.0, 8.0, 8.0, 12.0}}});
  expectShift(
      ToBlockSparseMatrix<6>(coupled(6), /*pruneZeros*/ true),
      {{{14.0, 14.0, 17.0, 19.0, 19.0, 21.0}, {12.0, 12.0, 16.0, 16.0, 16.0, 20.0}}});
}

// The positive-definiteness checks of RigidBodyModesAMGPrec accept a pivot only above n epsilon
// times its row's diagonal entry, and the inverses from the 6x6 factors, from the coarsest matrix's
// check and from its translation block's are accurate.
TEST(AMG, RigidModesFactorChecks) {
  real const eps = std::numeric_limits<real>::epsilon();
  real const nan = std::numeric_limits<real>::quiet_NaN();
  for (auto const& [delta, accepted] :
       {std::pair{1_r, true},
        std::pair{4 * eps, true},
        std::pair{eps, false},
        std::pair{-0.5_r, false},
        std::pair{nan, false}}) {
    // [[1, 1], [1, 1 + delta]], whose second pivot is delta.
    RowMatrix<real> A(2, 2);
    A(0, 0) = 1_r;
    A(0, 1) = 1_r;
    A(1, 0) = 1_r;
    A(1, 1) = 1_r + delta;
    real factors[4] = {A(0, 0), A(0, 1), A(1, 0), A(1, 1)};
    EXPECT_EQ(krylov::details::IsPositiveDefinite(A), accepted);
    Matrix<real> inverse;
    EXPECT_EQ(krylov::details::InvertIfPositiveDefinite(A, inverse), accepted);
    EXPECT_EQ(krylov::details::FactorLDLt(factors, 2), accepted);
    // The same pivot, scaled like the margin, last in a 6x6 smoother block, which fails even when
    // the inverse computed from its factors would pass.
    Matrix<real> smootherBlock(6, 6);
    smootherBlock.SetIdentity();
    smootherBlock(4, 5) = 1_r;
    smootherBlock(5, 4) = 1_r;
    smootherBlock(5, 5) = 1_r + 3 * delta;
    krylov::details::CheckedBlockJacobi6<real> smoother;
    EXPECT_EQ(
        smoother.Update(ToBlockSparseMatrix<6>(smootherBlock, /*pruneZeros*/ false)), accepted);
    EXPECT_EQ(krylov::details::IsPositiveDefinite(smootherBlock), accepted);
  }

  real block[36];
  for (int k = 0; k < 36; ++k) {
    block[k] = (k / 6 == k % 6) ? 7_r : 1_r / real(1 + Abs(k / 6 - k % 6));
  }
  real factors[36];
  std::copy(block, block + 36, factors);
  ASSERT_TRUE(krylov::details::FactorLDLt(factors, 6));
  real inverse[36];
  krylov::details::InverseFromLDLt6(factors, inverse);
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      double product = 0.0;
      for (int k = 0; k < 6; ++k) {
        product += double(block[6 * i + k]) * double(inverse[6 * k + j]);
      }
      // The block's condition number is below 2, so its inverse is accurate to a few epsilon.
      EXPECT_NEAR(product, i == j ? 1.0 : 0.0, 10 * eps);
    }
  }

  auto const diagonallyDominant = [](int n) {
    Matrix<real> M(n, n);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        M(i, j) = (i == j) ? real(n) : 1_r / real(1 + Abs(i - j));
      }
    }
    return M;
  };

  // From kSmallLDLtInverseSizeThreshold rows, the inverse comes from LDLt's factors.
  int const n = kSmallLDLtInverseSizeThreshold;
  Matrix<real> const M = diagonallyDominant(n);
  Matrix<real> coarsestInverse;
  ASSERT_TRUE(krylov::details::InvertIfPositiveDefinite(M, coarsestInverse));
  ExpectAccurateInverse(M, coarsestInverse, n * eps);

  // The translation-only solve of three 6-DoF nodes is accurate, and fails if an entry outside the
  // translation block, here the last one, is not finite.
  Matrix<real> const coarsest = diagonallyDominant(18);
  ASSERT_TRUE(krylov::details::InvertTranslationsIfPositiveDefinite(coarsest, coarsestInverse));
  ExpectAccurateInverse(coarsest, coarsestInverse, 18 * eps, /*translationsOnly*/ true);
  for (real const nonFinite : {nan, kInf}) {
    Matrix<real> withNonFinite = coarsest;
    withNonFinite(17, 17) = nonFinite;
    EXPECT_FALSE(
        krylov::details::InvertTranslationsIfPositiveDefinite(withNonFinite, coarsestInverse));
  }
}

TEST(AMG, Partition1D) {
  int p = 3;
  int n = int(std::pow(3.0, p) - 2);
  auto As = mochi::test::Make3ptLaplacianMatrix(n);
  auto nToN = AsGraphView(As);
  auto [partition, numParts] = krylov::details::Aggregate(nToN);
  //
  EXPECT_EQ(numParts, (n + 2) / 3);
  //
  // Partition will be composed of:
  // - one patch with 2 nodes;
  // - several patches with 3 nodes;
  // - one final patch with 2 nodes
  //
  int c = 0;
  EXPECT_EQ(partition[0], c);
  EXPECT_EQ(partition[1], c);
  c += 1;
  for (int i = 2; i + 3 <= n; i += 3, c += 1) {
    EXPECT_EQ(partition[i], c);
    EXPECT_EQ(partition[i + 1], c);
    EXPECT_EQ(partition[i + 2], c);
  }
  EXPECT_EQ(partition[n - 2], c);
  EXPECT_EQ(partition[n - 1], c);
}

TEST(AMG, Partition2D) {
  int px = 4;
  int nx = int(std::pow(3.0, px) - 2);
  int py = 3;
  int ny = int(std::pow(3.0, py) - 2);
  int n = nx * ny;
  auto A = Matrix<real>::Zero(n, n);
  for (int iy = 0; iy < ny; ++iy) {
    for (int ix = 0; ix < nx; ++ix) {
      int node = ix + iy * nx;
      if (iy > 0) {
        if (ix > 0) {
          A(node, node - nx - 1) = -1.0_r;
        }
        A(node, node - nx) = -1.0_r;
        if (ix + 1 < nx) {
          A(node, node - nx + 1) = -1.0_r;
        }
      }
      //
      if (ix > 0) {
        A(node, node - 1) = -1.0_r;
      }
      A(node, node) = 4.0_r;
      if (ix + 1 < nx) {
        A(node, node + 1) = -1.0_r;
      }
      //
      if (iy + 1 < ny) {
        if (ix > 0) {
          A(node, node + nx - 1) = -1.0_r;
        }
        A(node, node + nx) = -1.0_r;
        if (ix + 1 < nx) {
          A(node, node + nx + 1) = -1.0_r;
        }
      }
    }
  }
  auto Af = ToBlockSparseMatrix<1>(A, true);
  auto nToN = AsGraphView(Af);
  auto [partition, numParts] = krylov::details::Aggregate(nToN);
  //
  // The partition should be a "tensor"-product of 1D partitions.
  //
  int ncx = (nx + 2) / 3;
  int ncy = (ny + 2) / 3;
  EXPECT_EQ(numParts, ncx * ncy);
  //
  for (int jy = 0; jy < ny; ++jy) {
    for (int ix = 0; ix < nx; ++ix) {
      int node = ix + jy * nx;
      int coarse = (ix + 1) / 3 + ((jy + 1) / 3) * ncx;
      EXPECT_EQ(coarse, partition[node]);
    }
  }
}

TEST(AMG, Laplace2DRates) {
  int px = 5;
  int nx = int(std::pow(3.0, px) - 2);
  int py = 5;
  int ny = int(std::pow(3.0, py) - 2);
  int n = nx * ny;
  DynamicArray<int> rowPtr(n + 1, 0);
  DynamicArray<int> colIdx;
  colIdx.reserve(9 * n);
  DynamicArray<real> values;
  values.reserve(9 * n);
  for (int iy = 0; iy < ny; ++iy) {
    for (int ix = 0; ix < nx; ++ix) {
      int node = ix + iy * nx;
      if (iy > 0) {
        if (ix > 0) {
          colIdx.push_back(node - nx - 1);
          values.push_back(-0.25_r);
        }
        colIdx.push_back(node - nx);
        values.push_back(-0.5_r);
        if (ix + 1 < nx) {
          colIdx.push_back(node - nx + 1);
          values.push_back(-0.25_r);
        }
      }
      //
      if (ix > 0) {
        colIdx.push_back(node - 1);
        values.push_back(-0.5_r);
      }
      //--- Any diagonal entry greater than 3.0 makes the matrix diagonal-dominant
      colIdx.push_back(node);
      values.push_back(4.0_r);
      if (ix + 1 < nx) {
        colIdx.push_back(node + 1);
        values.push_back(-0.5_r);
      }
      //
      if (iy + 1 < ny) {
        if (ix > 0) {
          colIdx.push_back(node + nx - 1);
          values.push_back(-0.25_r);
        }
        colIdx.push_back(node + nx);
        values.push_back(-0.5_r);
        if (ix + 1 < nx) {
          colIdx.push_back(node + nx + 1);
          values.push_back(-0.25_r);
        }
      }
      //
      rowPtr[node + 1] = isize(colIdx);
    }
  }
  BlockSparseMatrix<real, 1, int, int> Af(n, rowPtr, colIdx, values);
  auto b = ColumnVector<real>::Zero(Af.Rows());
  ColumnVector<real> x(Af.Rows());
  x.SetRandom(123);
  ColumnVector<real> x0 = x;
  {
    // Use the default weighted Jacobi iteration with auto-computed relaxation factor and {(1, 0),
    // (1, 1), (2, 1)} smoothing steps.
    std::vector<int> preSteps({1, 1, 2});
    std::vector<int> postSteps({0, 1, 1});
    //
    // Table 4.2 of "A multigrid tutorial" by W. Briggs, V. Henson, and S. McCormick
    // suggests, for fixed w = 2/3 weighted Jacobi, a geometric multigrid convergence factor of 0.49
    // for (1, 0), 0.35 for (1, 1), and 0.24 for (2, 1). Here we are using smoothed aggregation
    // with auto-damped block Jacobi, and the average convergence factor is slightly worse.
    //
    std::vector<real> refRatio({0.55_r, 0.36_r, 0.25_r});
    for (int i = 0; i < isize(preSteps); ++i) {
      krylov::AMGOptions<real> options;
      options.numPreSmoothingSteps = preSteps[i];
      options.numPostSmoothingSteps = postSteps[i];
      krylov::AMGPrec<real, 1> prec(Af, options);
      x = x0;
      real avgReduction = 0.0_r;
      real oldNorm = x.Norm();
      for (int k = 0; k < 10; ++k) {
        prec.VCycle<true>(b, x, 0);
        real currentNorm = x.Norm();
        avgReduction += currentNorm / oldNorm;
        oldNorm = currentNorm;
      }
      avgReduction /= 10.0_r;
      //
      EXPECT_LT(avgReduction, refRatio[i]);
    }
  }
  {
    // Use the weighted (w = 2/3) colored SSOR iteration with {(1, 0), (1, 1), (2, 1)}
    // smoothing steps
    std::vector<int> preSteps({1, 1, 2});
    std::vector<int> postSteps({0, 1, 1});
    //
    // Table 4.2 of "A multigrid tutorial" by W. Briggs, V. Henson, and S. McCormick
    // suggests for weighted (w = 2/3) G-S
    // a convergence factor for geometric multigrid of 0.33 for (1, 0), of 0.14 for (1, 1),
    // and of 0.08 for (2, 1).
    // Here we are using smoothed aggregation with colored SSOR and the average convergence factor
    // is slightly worse
    //
    std::vector<real> refRatio({0.37_r, 0.17_r, 0.08_r});
    for (int i = 0; i < isize(preSteps); ++i) {
      krylov::AMGOptions<real> options;
      options.smoother = krylov::Smoother::SSOR;
      options.numPreSmoothingSteps = preSteps[i];
      options.numPostSmoothingSteps = postSteps[i];
      krylov::AMGPrec<real, 1> prec(Af, options);
      x = x0;
      real avgReduction = 0.0_r;
      real oldNorm = x.Norm();
      for (int k = 0; k < 10; ++k) {
        prec.VCycle<true>(b, x, 0);
        real currentNorm = x.Norm();
        avgReduction += currentNorm / oldNorm;
        oldNorm = currentNorm;
      }
      avgReduction /= 10.0_r;
      //
      EXPECT_LT(avgReduction, refRatio[i]);
    }
  }
}

template <int kBlockSize>
void TestMatMatProduct() {
  auto A = mochi::test::MakeBlockSparseMatrix<real, kBlockSize>(2, 2, 2);
  auto Ad = ToMatrix(A);
  //
  int nB = A.BlockRows();
  DynamicArray<int> rowPtr(nB + 1, 0);
  DynamicArray<int> cIdx;
  cIdx.reserve(9 * nB);
  for (int i = 0; i < nB; ++i) {
    std::array<int, 9> const shift{-11, -7, -5, -3, 0, 2, 4, 6, 8};
    for (auto k : shift) {
      if ((k + i >= 0) && (k + i < nB / 2)) {
        cIdx.push_back(k + i);
      }
    }
    rowPtr[i + 1] = isize(cIdx);
  }
  DynamicArray<real> values(cIdx.size());
  ColumnVectorView<real> vv(values.data(), isize(cIdx));
  vv.SetRandom(33, -1.0_r, 1.0_r);
  SparseMatrix<real> B(nB / 2, rowPtr, cIdx, values);
  //
  auto Bkron = Matrix<real>::Zero(B.Rows() * kBlockSize, B.Cols() * kBlockSize);
  for (int i = 0; i < B.Rows(); ++i) {
    auto col = B.Indices(i);
    auto val = B.Values(i);
    for (int k = 0; k < isize(col); ++k) {
      Bkron.Block(i * kBlockSize, col[k] * kBlockSize, kBlockSize, kBlockSize).SetIdentity();
      Bkron.Block(i * kBlockSize, col[k] * kBlockSize, kBlockSize, kBlockSize) *= val[k];
    }
  }
  //
  //--- Compute the product AB = A * B
  //
  auto gB = AsGraphView(B);
  auto gAB = Traverse(AsGraphView(A), gB).SortTargets();
  BlockSparseMatrix<real, kBlockSize> AB(B.Cols(), gAB);
  krylov::details::SparseMatProduct(A, B, AB);
  Matrix<real> ABd = Ad * Bkron;
  auto result = ToMatrix(AB);
  EXPECT_EQ(ABd.Rows(), result.Rows());
  EXPECT_EQ(ABd.Cols(), result.Cols());
  auto tol = real(Ad.Cols()) * std::numeric_limits<real>::epsilon();
  for (int ii = 0; ii < ABd.Rows(); ++ii) {
    for (int jj = 0; jj < ABd.Cols(); ++jj) {
      EXPECT_NEAR_RTOL(ABd(ii, jj), result(ii, jj), tol);
    }
  }
  {
    auto Bdense = ToMatrix(Bkron);
    Matrix<real> BtAd = Bdense.Transpose() * Ad;
    auto Bt = Transpose(B);
    auto gBtA = Traverse(AsGraphView(Bt), AsGraphView(A)).SortTargets();
    BlockSparseMatrix<real, kBlockSize> BtA(A.BlockCols(), gBtA);
    krylov::details::SparseMatProduct(Bt, A, BtA);
    for (int ii = 0; ii < BtAd.Rows(); ++ii) {
      for (int jj = 0; jj < BtAd.Cols(); ++jj) {
        EXPECT_NEAR_RTOL(BtAd(ii, jj), BtA(ii, jj), tol);
      }
    }
  }
  //--- Reset AB to 'reference' values
  AB = ToBlockSparseMatrix<kBlockSize>(ABd, true);
  auto Bt = Transpose(B);
  auto gBt = Reverse(gB);
  auto gBtAB = Traverse(gBt, gAB).SortTargets();
  BlockSparseMatrix<real, kBlockSize, int, int> BtAB(AB.BlockCols(), gBtAB);
  krylov::details::SparseMatProduct(Bt, AB, BtAB);
  //
  result = ToMatrix(BtAB);
  Matrix<real> BtABd = Bkron.Transpose() * ABd;
  EXPECT_EQ(BtABd.Rows(), result.Rows());
  EXPECT_EQ(BtABd.Cols(), result.Cols());
  tol = real(Bkron.Rows()) * std::numeric_limits<real>::epsilon();
  for (int ii = 0; ii < BtABd.Rows(); ++ii) {
    for (int jj = 0; jj < BtABd.Cols(); ++jj) {
      EXPECT_NEAR_RTOL(BtABd(ii, jj), result(ii, jj), tol);
    }
  }
}

TEST(AMG, MatMatProduct) {
  TestMatMatProduct<1>();
  TestMatMatProduct<2>();
  TestMatMatProduct<3>();
  TestMatMatProduct<4>();
}

TEST(AMG, EstimateSpectralRadiusPowerMethod1D) {
  krylov::AMGOptions<real> const options = {};
  for (int n : {15, 31, 63}) {
    auto A = mochi::test::Make3ptLaplacianMatrix(n);
    krylov::BlockJacobiPrec<real, 1> invDiag(A);
    ColumnVector<real> v(A.Rows());
    v.SetRandom(123);
    auto estimate =
        krylov::details::EstimateSpectralRadiusPowerMethod<real>(A, invDiag, n, AsView(v));
    ExpectSpectralRadiusEstimateBoundsLambdaMax(estimate, n, options.spectralRadiusSafetyFactor);
  }
}

TEST(AMG, EstimateSpectralRadiusPCGExactResidualZero) {
  Matrix<real> A(4, 4);
  A.SetIdentity();
  krylov::IdentityPrec<real> invDiag(A);
  EXPECT_EQ(
      krylov::details::EstimateSpectralRadius(A, invDiag, 8, AsConstView(RandomRhs(A.Rows()))),
      1_r);
}

TEST(AMG, EstimateSpectralRadiusPCGRejectsZeroQuadraticWithNonzeroResidual) {
  Matrix<real> A = Matrix<real>::Zero(2, 2);
  A(0, 0) = 1_r;
  A(1, 1) = 2_r;
  ZeroAfterFirstSolvePrec invDiag;
  auto suppressWarnings = mochi::test::SuppressLogWarning();
  EXPECT_EQ(
      krylov::details::EstimateSpectralRadius(A, invDiag, 2, AsConstView(RandomRhs(A.Rows()))),
      0_r);
}

TEST(AMG, EstimateSpectralRadiusPCGRejectsNonFiniteBeta) {
  constexpr int n = 64;
  Matrix<real> A(n, n);
  A.SetIdentity();
  OverflowingBetaPrec invDiag;
  auto suppressWarnings = mochi::test::SuppressLogWarning();
  EXPECT_EQ(
      krylov::details::EstimateSpectralRadius(A, invDiag, 2, AsConstView(RandomRhs(A.Rows()))),
      0_r);
}

TEST(AMG, EstimateSpectralRadiusPCGRejectsInvalidRitzMatrix) {
  constexpr int n = 64;
  RitzOverflowOp A(n);
  RitzOverflowPrec invDiag;
  auto suppressWarnings = mochi::test::SuppressLogWarning();
  EXPECT_EQ(
      krylov::details::EstimateSpectralRadius(A, invDiag, 2, AsConstView(RandomRhs(A.Rows()))),
      0_r);
}

TEST(AMG, EstimateSpectralRadiusPCG1D) {
  krylov::AMGOptions<real> const options = {};
  for (int n : {15, 31, 63}) {
    auto A = mochi::test::Make3ptLaplacianMatrix(n);
    krylov::BlockJacobiPrec<real, 1> invDiag(A);
    auto estimate = krylov::details::EstimateSpectralRadius(
        A, invDiag, options.spectralRadiusMaxIters, AsConstView(RandomRhs(n)));
    ExpectSpectralRadiusEstimateBoundsLambdaMax(estimate, n, options.spectralRadiusSafetyFactor);
  }
}

TEST(AMG, EstimateSpectralRadiusPCGBlockSize2_1D) {
  constexpr int kBlock = 2;
  krylov::AMGOptions<real> const options = {};
  for (int nNodes : {8, 16, 32}) {
    int n = kBlock * nNodes;
    auto A = Matrix<real>::Zero(n, n);
    // Interleave two independent 1D Laplacians in a block-size-2 matrix. The
    // block-Jacobi normalization gives the same spectrum as the scalar case.
    for (int i = 0; i < nNodes; ++i) {
      if (i > 0) {
        for (int j = 0; j < kBlock; ++j) {
          A(kBlock * i + j, kBlock * (i - 1) + j) = -1.0_r * real(j + 1);
        }
      }
      for (int j = 0; j < kBlock; ++j) {
        A(kBlock * i + j, kBlock * i + j) = 2.0_r * real(j + 1);
      }
      if (i + 1 < nNodes) {
        for (int j = 0; j < kBlock; ++j) {
          A(kBlock * i + j, kBlock * (i + 1) + j) = -1.0_r * real(j + 1);
        }
      }
    }
    auto Af = ToBlockSparseMatrix<kBlock>(A, true);
    krylov::BlockJacobiPrec<real, kBlock> invDiag(Af);
    auto estimate = krylov::details::EstimateSpectralRadius(
        Af, invDiag, options.spectralRadiusMaxIters, AsConstView(RandomRhs(Af.Rows())));
    ExpectSpectralRadiusEstimateBoundsLambdaMax(
        estimate, nNodes, options.spectralRadiusSafetyFactor);
  }
}

/// @brief Test that auto-computed relaxation factor fixes a non-SPD preconditioner.
///
/// Constructs A = e*e^T + 2*q*q^T + 0.1*I (n=8) where e=[1,...,1] and q=[1,-1,...,1,-1].
/// The complete graph forces single-aggregate coarsening (P = e), so the coarse space
/// captures only the constant mode. With omega = 2/3, the smoother amplifies the q-mode
/// (eigenvalue -2.46), making the V-cycle non-SPD. Auto-computed omega ~ 0.23 fixes this.
TEST(AMG, RelaxationFactorAutoCompute) {
  constexpr int n = 8;
  auto A = MakeRelaxationFactorStressMatrix(n, 2.0_r);
  auto Af = ToBlockSparseMatrix<1>(A, true);
  auto const r = MakeAlternatingVector(n);
  ColumnVector<real> z(n);
  {
    krylov::AMGOptions<real> options;
    options.relaxationFactor = 2.0_r / 3.0_r;
    krylov::AMGPrec<real, 1> prec(Af, options);
    prec(r, z);
    EXPECT_LT(r.Dot(z), 0.0_r);
  }
  {
    AMG1Access prec(Af);
    EXPECT_LT(prec.GetRelaxationFactor(0), AMG1Access::kDefaultRelaxationFactor);
    prec(r, z);
    EXPECT_GT(r.Dot(z), 0.0_r);
  }
}

TEST(AMG, ConstructorOptionsAutoRelaxationFactorComputedPerLevel) {
  constexpr int n = 15;
  krylov::AMGOptions<real> const options = {};
  auto Af = MakeWeightedTridiagonalBlockSparseMatrix(n, 0.5_r);
  auto level0 = krylov::details::Coarsen(Af, options.prolongationSmoothingWeight);
  auto level1 = krylov::details::Coarsen(level0.PtAP, options.prolongationSmoothingWeight);
  ASSERT_NE(level1.PtAP.BlockRows(), level0.PtAP.BlockRows());
  ASSERT_NE(level1.PtAP.BlockRows(), 1);

  AMG1Access prec(Af, options);

  real const expectedLevel0 = EstimateAutoRelaxationFactor(Af, options);
  real const expectedLevel1 = EstimateAutoRelaxationFactor(level0.PtAP, options);
  EXPECT_GT(std::abs(expectedLevel0 - expectedLevel1), 1.0e-5_r);
  EXPECT_NEAR_RTOL(expectedLevel0, prec.GetRelaxationFactor(0), 1.0e-5_r);
  EXPECT_NEAR_RTOL(expectedLevel1, prec.GetRelaxationFactor(1), 1.0e-5_r);
}

TEST(AMG, ConstructorOptionsFixedRelaxationFactor) {
  constexpr int n = 15;
  auto Af = MakeWeightedTridiagonalBlockSparseMatrix(n, 0.5_r);
  krylov::AMGOptions<real> options;
  auto level0 = krylov::details::Coarsen(Af, options.prolongationSmoothingWeight);
  auto level1 = krylov::details::Coarsen(level0.PtAP, options.prolongationSmoothingWeight);
  ASSERT_NE(level1.PtAP.BlockRows(), level0.PtAP.BlockRows());
  ASSERT_NE(level1.PtAP.BlockRows(), 1);

  options.relaxationFactor = 0.5_r;

  {
    AMG1Access prec(Af, options);
    EXPECT_EQ(prec.GetRelaxationFactor(0), 0.5_r);
    EXPECT_EQ(prec.GetRelaxationFactor(1), 0.5_r);
  }
  {
    options.smoother = krylov::Smoother::SSOR;
    AMG1Access prec(Af, options);
    EXPECT_EQ(prec.GetRelaxationFactor(0), 0.5_r);
    EXPECT_EQ(prec.GetRelaxationFactor(1), 0.5_r);
  }
}

TEST(AMG, ConstructorOptionsSpectralRadiusMaxItersAffectInitialRelaxationFactor) {
  constexpr int n = 8;
  auto A = MakeRelaxationFactorStressMatrix(n, 2.0_r);
  auto Af = ToBlockSparseMatrix<1>(A, true);

  krylov::AMGOptions<real> optionsFew;
  optionsFew.spectralRadiusMaxIters = 2;
  AMG1Access precFew(Af, optionsFew);
  real const relaxationFactorFew = precFew.GetRelaxationFactor(0);

  krylov::AMGOptions<real> optionsAtDimension;
  optionsAtDimension.spectralRadiusMaxIters = n;
  AMG1Access precAtDimension(Af, optionsAtDimension);
  real const relaxationFactorAtDimension = precAtDimension.GetRelaxationFactor(0);

  krylov::AMGOptions<real> optionsMany;
  optionsMany.spectralRadiusMaxIters = 32;
  AMG1Access precMany(Af, optionsMany);
  real const relaxationFactorMany = precMany.GetRelaxationFactor(0);

  EXPECT_EQ(relaxationFactorMany, relaxationFactorAtDimension);
  EXPECT_GT(relaxationFactorFew, 0.0_r);
  EXPECT_LT(relaxationFactorFew, AMG1Access::kDefaultRelaxationFactor);
  EXPECT_GT(relaxationFactorMany, 0.0_r);
  EXPECT_LT(relaxationFactorMany, AMG1Access::kDefaultRelaxationFactor);
  EXPECT_LT(relaxationFactorMany, relaxationFactorFew);
}

TEST(AMG, ConstructorOptionsSpectralRadiusSafetyFactorAffectsInitialRelaxationFactor) {
  constexpr int n = 8;
  auto A = MakeRelaxationFactorStressMatrix(n, 2.0_r);
  auto Af = ToBlockSparseMatrix<1>(A, true);

  krylov::AMGOptions<real> optionsLow;
  optionsLow.spectralRadiusSafetyFactor = 1.0_r;
  AMG1Access precLow(Af, optionsLow);

  krylov::AMGOptions<real> optionsHigh;
  optionsHigh.spectralRadiusSafetyFactor = 2.0_r;
  AMG1Access precHigh(Af, optionsHigh);

  EXPECT_GT(precLow.GetRelaxationFactor(0), 0.0_r);
  EXPECT_NEAR_RTOL(
      precHigh.GetRelaxationFactor(0), precLow.GetRelaxationFactor(0) / 2.0_r, 1.0e-5_r);
}

TEST(AMG, UpdateRecomputesAutoRelaxationFactor) {
  constexpr int n = 15;
  krylov::AMGOptions<real> const options = {};
  auto Af = MakeWeightedTridiagonalBlockSparseMatrix(n, 0.5_r);
  auto level0 = krylov::details::Coarsen(Af, options.prolongationSmoothingWeight);
  auto level1 = krylov::details::Coarsen(level0.PtAP, options.prolongationSmoothingWeight);
  ASSERT_NE(level1.PtAP.BlockRows(), level0.PtAP.BlockRows());
  ASSERT_NE(level1.PtAP.BlockRows(), 1);
  AMG1Access prec(Af, options);
  real const level1RelaxationFactorBeforeUpdate = prec.GetRelaxationFactor(1);

  auto AfUpdated = MakeWeightedTridiagonalBlockSparseMatrix(n, 0.9_r);
  prec.Update(AfUpdated);

  EXPECT_NEAR_RTOL(
      EstimateAutoRelaxationFactor(AfUpdated, options), prec.GetRelaxationFactor(0), 1.0e-5_r);
  EXPECT_GT(Abs(prec.GetRelaxationFactor(1) - level1RelaxationFactorBeforeUpdate), 1.0e-5_r);
  EXPECT_NEAR_RTOL(
      EstimateAutoRelaxationFactor(prec.GetCoarsenedMatrixForRelaxationLevel(1), options),
      prec.GetRelaxationFactor(1),
      1.0e-5_r);
}

TEST(AMG, AutoRelaxationFallbackUsesDefaultFactor) {
  constexpr int n = 15;
  krylov::AMGOptions<real> const options = {};
  auto Af = MakeWeightedTridiagonalBlockSparseMatrix(n, 0.5_r);
  auto level0 = krylov::details::Coarsen(Af, options.prolongationSmoothingWeight);
  auto level1 = krylov::details::Coarsen(level0.PtAP, options.prolongationSmoothingWeight);
  ASSERT_NE(level0.PtAP.BlockRows(), level1.PtAP.BlockRows());
  ASSERT_NE(1, level1.PtAP.BlockRows());

  auto AfEstimateFails = MakeSpectralEstimateFailureMatrix(n, 0.5_r);
  {
    auto suppressWarnings = mochi::test::SuppressLogWarning();
    AMG1Access prec(AfEstimateFails, options);
    EXPECT_EQ(AMG1Access::kDefaultRelaxationFactor, prec.GetRelaxationFactor(0));
    EXPECT_EQ(AMG1Access::kDefaultRelaxationFactor, prec.GetRelaxationFactor(1));
  }

  AMG1Access prec(Af, options);
  ASSERT_NE(AMG1Access::kDefaultRelaxationFactor, prec.GetRelaxationFactor(0));
  ASSERT_NE(AMG1Access::kDefaultRelaxationFactor, prec.GetRelaxationFactor(1));

  {
    auto suppressWarnings = mochi::test::SuppressLogWarning();
    prec.Update(AfEstimateFails);
  }

  EXPECT_EQ(AMG1Access::kDefaultRelaxationFactor, prec.GetRelaxationFactor(0));
  EXPECT_EQ(AMG1Access::kDefaultRelaxationFactor, prec.GetRelaxationFactor(1));
}

TEST(AMG, UpdatePreservesExplicitRelaxationFactor) {
  constexpr int n = 15;
  auto Af = MakeWeightedTridiagonalBlockSparseMatrix(n, 0.5_r);
  krylov::AMGOptions<real> options;
  auto level0 = krylov::details::Coarsen(Af, options.prolongationSmoothingWeight);
  auto level1 = krylov::details::Coarsen(level0.PtAP, options.prolongationSmoothingWeight);
  ASSERT_NE(level1.PtAP.BlockRows(), level0.PtAP.BlockRows());
  ASSERT_NE(level1.PtAP.BlockRows(), 1);

  options.relaxationFactor = 0.5_r;
  AMG1Access prec(Af, options);

  auto AfUpdated = MakeWeightedTridiagonalBlockSparseMatrix(n, 0.9_r);
  prec.Update(AfUpdated);

  EXPECT_EQ(prec.GetRelaxationFactor(0), 0.5_r);
  EXPECT_EQ(prec.GetRelaxationFactor(1), 0.5_r);
}

TEST(AMG, ConstructorOptionsNonJacobiUsesDefaultRelaxationFactor) {
  constexpr int n = 15;
  auto Af = MakeWeightedTridiagonalBlockSparseMatrix(n, 0.5_r);

  krylov::AMGOptions<real> options;
  auto level0 = krylov::details::Coarsen(Af, options.prolongationSmoothingWeight);
  auto level1 = krylov::details::Coarsen(level0.PtAP, options.prolongationSmoothingWeight);
  ASSERT_NE(level1.PtAP.BlockRows(), level0.PtAP.BlockRows());
  ASSERT_NE(level1.PtAP.BlockRows(), 1);

  options.smoother = krylov::Smoother::SSOR;
  AMG1Access prec(Af, options);
  EXPECT_EQ(prec.GetRelaxationFactor(0), AMG1Access::kDefaultRelaxationFactor);
  EXPECT_EQ(prec.GetRelaxationFactor(1), AMG1Access::kDefaultRelaxationFactor);
}

//
// TODO Add unit test(s) to verify the aggregation routine
// TODO Add unit test(s) to verify the smoothing process
// TODO Add unit test(s) to verify ApproximateJacobi smoother
//

} // namespace test::amg
