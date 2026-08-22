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
#include <mochi_core/linear_algebra/sparse_matrix.h>
#include <mochi_core/linear_algebra/sparse_matrix_utils.h>
#include <mochi_core/linear_algebra/utils/matrix_conversions.h>
#include <mochi_core/test/linear_algebra_test_helpers.h>
#include <mochi_core/test/log_suppression.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/task_scheduler.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <memory>
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
      p.operator()(x, Px1);
      EXPECT_TRUE(mochi::test::NearEqualMatrices(Px1, Px2));
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

template <typename MatrixType>
[[nodiscard]] static real EstimateAutoRelaxationFactor(
    MatrixType const& A,
    krylov::AMGOptions<real> const& options) {
  krylov::BlockJacobiPrec<real, 1> invDiag(A);
  auto const estimate =
      krylov::details::EstimateSpectralRadius(A, invDiag, options.spectralRadiusMaxIters);
  EXPECT_GT(estimate, 0_r);
  return real(4.0 / 3.0) / (options.spectralRadiusSafetyFactor * estimate);
}

class AMG1Access : public krylov::AMGPrec<real, 1> {
 public:
  using krylov::AMGPrec<real, 1>::AMGPrec;
  using krylov::AMGPrec<real, 1>::GetRelaxationFactor;

  auto const& GetCoarsenedMatrixForRelaxationLevel(int level) const {
    MOCHI_ASSERT_VERBOSE(level > 0 && level <= isize(_coarsenings), "Invalid AMG level.");
    return _coarsenings[level - 1].PtAP;
  }

  friend class AMG1Test;
  FRIEND_TEST(AMG1Test, Example1);
};

/// @brief Class to test the AMG preconditioner with 1D Laplace equation
class AMG1Test : public testing::Test {
 protected:
  using PrecType = AMG1Access;
  static constexpr int kBlock = 1;
  void SetUp() override {
    int p = 4;
    int n = int(std::pow(2.0, p) - 1);
    _Af = mochi::test::Make3ptLaplacianMatrix(n);
    _p = std::make_unique<PrecType>(_Af);
  }
  SparseMatrix<real, int, int> _Af;
  std::unique_ptr<PrecType> _p = nullptr;
};

TEST_F(AMG1Test, Example1) {
  auto Adense = ToMatrix(_Af);
  std::vector<Matrix<real>> Al;
  Al.reserve(_p->_coarsenings.size() + 1);
  Al.push_back(Adense);
  for (int i = 0; i < _p->_coarsenings.size(); ++i) {
    auto [T, PtA, PtAP] = _p->_coarsenings[i];
    auto Pdense = ToMatrix(T.P);
    auto PtAdense = ToMatrix(PtA);
    Matrix<real> result = Transpose(Pdense) * Al[i];
    EXPECT_TRUE(mochi::test::NearEqualMatrices(PtAdense, result));
    auto Acoarse = ToMatrix(PtAP);
    result = PtAdense * Pdense;
    real tol = std::numeric_limits<real>::epsilon() * Acoarse.Norm();
    //--- Verify the coarser matrix
    EXPECT_TRUE(mochi::test::NearEqualMatrices(Acoarse, result, tol));
    Al.push_back(Acoarse);
  }
  //
  {
    auto const& coarsest = Al[Al.size() - 1];
    EXPECT_EQ(coarsest.Rows(), _p->_coarseInverse.Rows());
    EXPECT_EQ(coarsest.Cols(), _p->_coarseInverse.Cols());
    Matrix<real> identity(coarsest.Rows(), coarsest.Cols());
    identity.SetIdentity();
    Matrix<real> product = coarsest * _p->_coarseInverse;
    EXPECT_TRUE(mochi::test::NearEqualMatrices(identity, product));
  }
  //
  {
    ColumnVector<real> r(_Af.Rows());
    r.SetRandom(3);
    ColumnVector<real> z(_Af.Rows());
    _p->operator()(r, z);
    real energy = -r.Dot(z);
    ColumnVector<real> Az = Adense * z;
    energy += 0.5_r * z.Dot(Az);
    //--- Trivial test because solution has negative energy
    EXPECT_LT(energy, 0.0_r);
  }
  // Test 'ConcurrentSolve' method.
  TestConcurrentSolve(*_p, _Af.Rows(), kBlock);
}

class AMG2Access : public krylov::AMGPrec<real, 2> {
 public:
  using krylov::AMGPrec<real, 2>::AMGPrec;
  friend class AMG2Test;
  FRIEND_TEST(AMG2Test, Example2);
};

/// @brief Class to test the AMG preconditioner with 2 1D Laplace equations
class AMG2Test : public testing::Test {
 protected:
  using PrecType = AMG2Access;
  static constexpr int kBlock = 2;
  void SetUp() override {
    int p = 4;
    int n = kBlock * int(std::pow(2.0, p) - 1);
    auto A = Matrix<real>::Zero(n, n);
    for (int i = 0; i < n / kBlock; ++i) {
      if (i > 0) {
        for (int j = 0; j < kBlock; ++j) {
          A(kBlock * i + j, kBlock * (i - 1) + j) = -1.0_r;
        }
      }
      for (int j = 0; j < kBlock; ++j) {
        A(kBlock * i + j, kBlock * i + j) = 2.0_r;
      }
      if (i + 1 < n / kBlock) {
        for (int j = 0; j < kBlock; ++j) {
          A(kBlock * i + j, kBlock * (i + 1) + j) = -1.0_r;
        }
      }
    }
    _Af = ToBlockSparseMatrix<kBlock>(A, true);
    _p = std::make_unique<PrecType>(_Af);
  }
  BlockSparseMatrix<real, kBlock, int, int> _Af;
  std::unique_ptr<PrecType> _p = nullptr;
};

TEST_F(AMG2Test, Example2) {
  {
    auto const& coarsest = _p->_coarsenings.back().PtAP;
    EXPECT_EQ(coarsest.Rows(), _p->_coarseInverse.Rows());
    EXPECT_EQ(coarsest.Cols(), _p->_coarseInverse.Cols());
    Matrix<real> identity(coarsest.Rows(), coarsest.Cols());
    identity.SetIdentity();
    Matrix<real> product = coarsest * _p->_coarseInverse;
    EXPECT_TRUE(mochi::test::NearEqualMatrices(identity, product));
  }
  //
  {
    ColumnVector<real> r(_Af.Rows());
    r.SetRandom(3);
    ColumnVector<real> z(_Af.Rows());
    _p->operator()(r, z);
    real energy = -r.Dot(z);
    ColumnVector<real> Az = _Af * z;
    energy += 0.5_r * z.Dot(Az);
    //--- Trivial test because solution has negative energy
    EXPECT_LT(energy, 0.0_r);
  }
  // Test 'ConcurrentSolve' method.
  TestConcurrentSolve(*_p, _Af.Rows(), kBlock);
}

class AMG3Access : public krylov::AMGPrec<real, 2> {
 public:
  using krylov::AMGPrec<real, 2>::AMGPrec;
  friend class AMG3Test;
  FRIEND_TEST(AMG3Test, Example3);
};

/// @brief Class to test the AMG preconditioner with 2 2D Laplace equations
class AMG3Test : public testing::Test {
 protected:
  using PrecType = AMG3Access;
  static constexpr int kBlock = 2;
  void SetUp() override {
    int px = 3;
    int nx = int(std::pow(2.0, px) - 1);
    int py = 2;
    int ny = int(std::pow(2.0, py) - 1);
    int n = nx * ny * kBlock;
    auto A = Matrix<real>::Zero(n, n);
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        int node = ix + iy * nx;
        if (iy > 0) {
          for (int k = 0; k < kBlock; ++k) {
            A(k + kBlock * node, k + kBlock * (node - nx)) = -1.0_r;
          }
        }
        //
        if (ix > 0) {
          for (int k = 0; k < kBlock; ++k) {
            A(k + kBlock * node, k + kBlock * (node - 1)) = -1.0_r;
          }
        }
        for (int k = 0; k < kBlock; ++k) {
          A(k + kBlock * node, k + kBlock * node) = 4.0_r;
        }
        if (ix + 1 < nx) {
          for (int k = 0; k < kBlock; ++k) {
            A(k + kBlock * node, k + kBlock * (node + 1)) = -1.0_r;
          }
        }
        //
        if (iy + 1 < ny) {
          for (int k = 0; k < kBlock; ++k) {
            A(k + kBlock * node, k + kBlock * (node + nx)) = -1.0_r;
          }
        }
      }
    }
    _Af = ToBlockSparseMatrix<kBlock>(A, true);
    _p = std::make_unique<PrecType>(_Af);
  }
  BlockSparseMatrix<real, kBlock, int, int> _Af;
  std::unique_ptr<PrecType> _p = nullptr;
};

TEST_F(AMG3Test, Example3) {
  {
    auto const& coarsest = _p->_coarsenings.back().PtAP;
    EXPECT_EQ(coarsest.Rows(), _p->_coarseInverse.Rows());
    EXPECT_EQ(coarsest.Cols(), _p->_coarseInverse.Cols());
    Matrix<real> identity(coarsest.Rows(), coarsest.Cols());
    identity.SetIdentity();
    Matrix<real> product = coarsest * _p->_coarseInverse;
    EXPECT_TRUE(mochi::test::NearEqualMatrices(identity, product));
  }
  //
  {
    ColumnVector<real> r(_Af.Rows());
    r.SetRandom(3);
    ColumnVector<real> z(_Af.Rows());
    _p->operator()(r, z);
    real energy = -r.Dot(z);
    ColumnVector<real> Az = _Af * z;
    energy += 0.5_r * z.Dot(Az);
    //--- Trivial test because solution has negative energy
    EXPECT_LT(energy, 0.0_r);
    //
    //--- Update the AMG preconditioner with the same matrix
    //
    _p->Update(_Af);
    ColumnVector<real> znew(z);
    _p->operator()(r, znew);
    EXPECT_TRUE(mochi::test::NearEqualMatrices(z, znew));
  }
  // Test 'ConcurrentSolve' method.
  TestConcurrentSolve(*_p, _Af.Rows(), kBlock);
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

TEST(AMG, EstimateSpectralRadiusPCG1D) {
  krylov::AMGOptions<real> const options = {};
  for (int n : {15, 31, 63}) {
    auto A = mochi::test::Make3ptLaplacianMatrix(n);
    krylov::BlockJacobiPrec<real, 1> invDiag(A);
    auto estimate =
        krylov::details::EstimateSpectralRadius(A, invDiag, options.spectralRadiusMaxIters);
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
    auto estimate =
        krylov::details::EstimateSpectralRadius(Af, invDiag, options.spectralRadiusMaxIters);
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

  krylov::AMGOptions<real> optionsMany;
  optionsMany.spectralRadiusMaxIters = 32;
  AMG1Access precMany(Af, optionsMany);
  real const relaxationFactorMany = precMany.GetRelaxationFactor(0);

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

TEST(AMG, AutoRelaxationFallbackKeepsCurrentFactor) {
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
  real const level0RelaxationFactorBeforeUpdate = prec.GetRelaxationFactor(0);
  real const level1RelaxationFactorBeforeUpdate = prec.GetRelaxationFactor(1);

  {
    auto suppressWarnings = mochi::test::SuppressLogWarning();
    prec.Update(AfEstimateFails);
  }

  EXPECT_EQ(level0RelaxationFactorBeforeUpdate, prec.GetRelaxationFactor(0));
  EXPECT_EQ(level1RelaxationFactorBeforeUpdate, prec.GetRelaxationFactor(1));
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
