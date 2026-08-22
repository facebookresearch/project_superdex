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

#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/solvers/linear_solver.h>
#include <mochi_core/test/mochi_test_helpers.h>

#include <gtest/gtest.h>

#include <limits>
#include <vector>

#include "krylov_solver_test_helpers.h"

#if MOCHI_USE_CUDA

#include <mochi_core/linear_algebra/cuda/cuda_bsr_matrix.h>
#include <mochi_core/linear_algebra/cuda/cuda_csr_matrix.h>
#include <mochi_core/linear_algebra/cuda/cuda_gmres.h>
#include <mochi_core/linear_algebra/cuda/cuda_lib.h>
#include <mochi_core/linear_algebra/cuda/cuda_pcg.h>

using namespace mochi;

template <typename Scalar, LinearSolverType kSolverType, typename StopCriterion>
void CudaIterativeSolverTest() {
  using Vector = ColumnVector<Scalar>;

  Scalar constexpr kAbsTol = KrylovTestConstants<Scalar>::kAbsTol;
  Scalar constexpr kRelDivTol = KrylovTestConstants<Scalar>::kRelDivTol;

  for (int itest = 0; itest < 2; ++itest) {
    auto problem = GetTestProblem<Scalar>(itest);
    int const matrixSize = problem.matrix.Rows();

    int maxIter = 1000;

    ColumnVector<Scalar> b(problem.rhs);
    CudaVector<Scalar> d_b(b);

    ColumnVector<Scalar> ref(problem.solution);
    CudaVector<Scalar> d_ref(ref);

    Vector sol(matrixSize);
    CudaVector<Scalar> d_x(sol);

    auto A = ToMatrix(problem.matrix);

    auto Acsr = ToSparseMatrix(A, true);
    krylov::CudaCsrMatrix<Scalar> d_Acsr(Acsr);

    std::function<void(CudaVectorView<Scalar, -1> x, CudaVectorView<Scalar, -1> Px)> P =
        [](CudaVectorView<Scalar, -1> y, CudaVectorView<Scalar, -1> Py) {
          details::CudaDeviceCopy(Py.data(), y.data(), y.Rows());
        };

    //--- Input data is specified with 7 digits of accuracy (hence -> float)
    auto resRelTol = Scalar(std::numeric_limits<float>::epsilon() * Scalar(matrixSize));
    Scalar solTol = resRelTol * KrylovTestConstants<Scalar>::kCondMat[itest];

    [[maybe_unused]] StopCriterion stopCriterion{resRelTol, kAbsTol, kRelDivTol};
    LinearSolverStatus info;

    d_x.SetZero();
    if constexpr (kSolverType == LinearSolverType::CudaCG) {
      info = krylov::CudaPCG(d_Acsr, d_b, d_x, P, maxIter, stopCriterion);
    } else if constexpr (kSolverType == LinearSolverType::CudaGMRES) {
      info = krylov::CudaGMRes(d_Acsr, d_b, d_x, P, maxIter, 1e-28, resRelTol, 1e16);
    }
    EXPECT_LE(info.numIterDone, matrixSize);
    [[maybe_unused]] int const fullIter = info.numIterDone;
    sol = d_x;
    Vector diff = ref - sol;
    EXPECT_LT(diff.Norm(), solTol * ref.Norm());

    if constexpr (kSolverType == LinearSolverType::CudaCG) {
      // With Fletcher-Reeves formula
      d_x.SetZero();
      info = krylov::CudaPCG(
          d_Acsr, d_b, d_x, P, maxIter, stopCriterion, true, VerbosityLevel::Warning, true);
      EXPECT_LE(info.numIterDone, matrixSize);
      sol = d_x;
      diff = ref - sol;
      EXPECT_LT(diff.Norm(), solTol * ref.Norm());
    }

    // With the solution as initial guess
    if constexpr (kSolverType == LinearSolverType::CudaCG) {
      info = krylov::CudaPCG(d_Acsr, d_b, d_x, P, maxIter, stopCriterion);
    } else if constexpr (kSolverType == LinearSolverType::CudaGMRES) {
      d_x = ref;
      info = krylov::CudaGMRes(d_Acsr, d_b, d_x, P, maxIter, 1e-28, resRelTol, 1e16);
    }
    EXPECT_LE(info.numIterDone, 1);
    sol = d_x;
    diff = ref - sol;
    EXPECT_LT(diff.Norm(), solTol * ref.Norm());

    // With arbitrary non-zero initial guess
    sol = Scalar(0.3) * ref;
    d_x = sol;
    if constexpr (kSolverType == LinearSolverType::CudaCG) {
      info = krylov::CudaPCG(d_Acsr, d_b, d_x, P, maxIter, stopCriterion);
    } else if constexpr (kSolverType == LinearSolverType::CudaGMRES) {
      info = krylov::CudaGMRes(d_Acsr, d_b, d_x, P, maxIter, 1e-28, resRelTol, 1e16);
    }
    EXPECT_LE(info.numIterDone, matrixSize);
    sol = d_x;
    diff = ref - sol;
    EXPECT_LT(diff.Norm(), solTol * ref.Norm());

    if constexpr (kSolverType == LinearSolverType::CudaGMRES) {
      // Activate a restart before original convergence
      if (fullIter > 1) {
        d_x.SetZero();
        info = krylov::CudaGMRes(
            d_Acsr, d_b, d_x, P, maxIter, 1e-28, resRelTol, 1e16, 64, VerbosityLevel::Warning);
        sol = d_x;
        diff = ref - sol;
        EXPECT_GE(info.numIterDone, fullIter);
        EXPECT_LT(info.relativeResidualNorm, resRelTol);
        EXPECT_LT(diff.Norm(), solTol * ref.Norm());
      }
    }

    if (matrixSize % 2 == 0) {
      auto Absr = ToBlockSparseMatrix<2, Scalar, int, int>(Acsr);
      krylov::CudaBsrMatrix<Scalar, 2, int, int> d_Absr(Absr);
      d_x.SetZero();
      if constexpr (kSolverType == LinearSolverType::CudaCG) {
        info = krylov::CudaPCG(d_Absr, d_b, d_x, P, maxIter, stopCriterion);
      } else if constexpr (kSolverType == LinearSolverType::CudaGMRES) {
        info = krylov::CudaGMRes(d_Absr, d_b, d_x, P, maxIter, 1e-28, resRelTol, 1e16);
      }
      EXPECT_LE(info.numIterDone, matrixSize);
      sol = d_x;
      diff = ref - sol;
      EXPECT_LT(diff.Norm(), solTol * ref.Norm());
    }

    if (matrixSize % 4 == 0) {
      auto Absr = ToBlockSparseMatrix<4, Scalar, int, int>(Acsr);
      krylov::CudaBsrMatrix<Scalar, 4, int, int> d_Absr(Absr);
      d_x.SetZero();
      if constexpr (kSolverType == LinearSolverType::CudaCG) {
        info = krylov::CudaPCG(d_Absr, d_b, d_x, P, maxIter, stopCriterion);
      } else if constexpr (kSolverType == LinearSolverType::CudaGMRES) {
        info = krylov::CudaGMRes(d_Absr, d_b, d_x, P, maxIter, 1e-28, resRelTol, 1e16);
      }
      EXPECT_LE(info.numIterDone, matrixSize);
      sol = d_x;
      diff = ref - sol;
      EXPECT_LT(diff.Norm(), solTol * ref.Norm());
    }
  }
}

TEST(KrylovSolver, CudaPcg) {
  CudaIterativeSolverTest<
      real,
      LinearSolverType::CudaCG,
      krylov::StatusResidualL2<krylov::UsualDot, real>>();
  CudaIterativeSolverTest<
      real,
      LinearSolverType::CudaCG,
      krylov::StatusPreconditionedResidualL2<krylov::UsualDot, real>>();
  CudaIterativeSolverTest<
      real,
      LinearSolverType::CudaCG,
      krylov::StatusResidualPreconditionerInduced<krylov::UsualDot, real>>();
}

TEST(KrylovSolver, CudaPcgFallback) {
  using Scalar = real;

  int const n = 3;
  auto M = Matrix<Scalar>::Zero(n, n);
  M(0, 0) = Scalar(4);
  M(0, 1) = Scalar(-1);
  M(1, 0) = Scalar(-1);
  M(1, 1) = Scalar(4);
  M(1, 2) = Scalar(-1);
  M(2, 1) = Scalar(-1);
  M(2, 2) = Scalar(4);
  ColumnVector<Scalar> b(n);
  b.SetConstant(Scalar(1));
  auto sM = ToSparseMatrix(M, true);
  auto d_M = ToCuda(sM);
  auto d_b = ToCuda(b);
  CudaVector<Scalar> d_x(n);
  d_x.SetZero();
  auto P = IdentityPreconditioner();
  krylov::StatusPreconditionedResidualL2<krylov::UsualDot, Scalar> stopCriterion(
      static_cast<Scalar>(1.0e-05), static_cast<Scalar>(1.0e-37), static_cast<Scalar>(1.0e10));
  //
  bool wasWarningEnabled = IsLogChannelEnabled(LogChannel::Warning);
  EnableLogChannel(LogChannel::Warning, false);
  MOCHI_DEFER(EnableLogChannel(LogChannel::Warning, wasWarningEnabled));
  //--- With a burst of 8, the PCG would yield not-finite residual entries
  //--- Uses the fallback
  auto status = krylov::CudaPCG(d_M, d_b, d_x, P, 1000, stopCriterion);
  EXPECT_TRUE(status.converged);
  EXPECT_LT(status.numIterDone, n);
}

#if MOCHI_PLATFORM_LINUX
TEST(KrylovSolver, DISABLED_CudaGMRes) {
#else
TEST(KrylovSolver, CudaGMRes) {
#endif // MOCHI_PLATFORM_LINUX
  CudaIterativeSolverTest<
      real,
      LinearSolverType::CudaGMRES,
      krylov::StatusResidualL2<krylov::UsualDot, real>>();
}

#endif // MOCHI_USE_CUDA
