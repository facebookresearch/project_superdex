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

#if MOCHI_USE_CUDA

#include <mochi_core/linear_algebra/base_tools.h>
#include <mochi_core/linear_algebra/block_sparse_matrix.h>
#include <mochi_core/linear_algebra/cuda/cuda_api.h>
#include <mochi_core/linear_algebra/cuda/cuda_block_jacobi_prec.h>
#include <mochi_core/linear_algebra/cuda/cuda_bsr_matrix.h>
#include <mochi_core/linear_algebra/cuda/cuda_csr_matrix.h>
#include <mochi_core/linear_algebra/cuda/cuda_gmres_kernels.h>
#include <mochi_core/linear_algebra/cuda/cuda_lib.h>
#include <mochi_core/linear_algebra/cuda/cuda_matrix.h>
#include <mochi_core/linear_algebra/cuda/cuda_sparse_factorization.h>
#include <mochi_core/linear_algebra/krylov/gmres.h>
#include <mochi_core/linear_algebra/krylov/pcg.h>
#include <mochi_core/linear_algebra/krylov_interop.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/strided_matrix.h>
#include <mochi_core/solvers/krylov_solver.h>
#include <mochi_core/solvers/linear_solver.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/sparsity_utils.h>

#include "matrix_expression_test.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <utility>
#include <vector>

using namespace mochi;

template <typename Scalar>
static void TestTranspose() {
  struct Sizes {
    int rows, cols;
  };
  Sizes combinations[] = {{16, 12}, {256, 256}, {256, 317}, {256, 320}, {260, 512}, {260, 1133}};
  for (auto size : combinations) {
    Matrix<Scalar> A(size.rows, size.cols);
    A.SetRandom(676);
    Matrix<Scalar> A_transpose = Matrix<Scalar>::Zero(size.cols, size.rows);
    A_transpose = A.Transpose();
    CudaMatrix<Scalar> Ac(size.rows, size.cols);
    CudaMatrix<Scalar> Atc(size.cols, size.rows);
    Ac = A;
    // Atc = A_transpose;
    CudaTranspose(
        CudaMatrixView<Scalar, krylov::kDynamic, krylov::kDynamic>{Atc},
        CudaMatrixView<Scalar const, krylov::kDynamic, krylov::kDynamic>{Ac});
    A_transpose = Atc;
    A_transpose -= A.Transpose();
    Scalar error = A_transpose.Norm();
    EXPECT_TRUE(error == Scalar{0});
  }
}

namespace mochi {

template <typename Scalar>
void DoCudaCuSquare3x3(CudaMatrix<Scalar>& A, CudaMatrix<Scalar>& B);

template <typename Scalar, int kStride>
void DoCudaCuSquare3x3(CudaVector<Scalar>& A, CudaVector<Scalar>& B);

} // namespace mochi

template <typename Scalar, int kStride>
static void TestStridedMatrixCuda() {
  Matrix<Scalar, 3, 3> Am{{10, 3, 5}, {2, 12, 4}, {-1, -2, 16}};
  Matrix<Scalar, 3, 3> Bm = Am * Am;

  CudaMatrix<Scalar> Acuda(3, 3);
  CudaMatrix<Scalar> Bcuda(3, 3);
  Acuda = Am;

  DoCudaCuSquare3x3(Acuda, Bcuda);
  Matrix<Scalar, 3, 3> Bhost;
  Bhost = Bcuda;
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      EXPECT_EQ(Bhost(row, col), Bm(row, col));
    }
  }

  StridedMatrix<Scalar, 3, 3> A1;
  A1 = Am;
  ColumnVector<Scalar> Astorage(3 * 3 * kStride * 5);
  StridedView<Scalar, 3, 3, kStride> AV{3, 3, Astorage.GetSpan()};
  for (int block = 0; block < 5 * kStride; ++block) {
    auto Astrided = AV[block];
    Astrided = static_cast<Scalar>(block % 7 + 1) * A1;
  }
  ColumnVector<Scalar> Bstorage(3 * 3 * kStride * 5);
  StridedView<Scalar, 3, 3, kStride> BV{3, 3, Bstorage.GetSpan()};

  CudaVector<Scalar> AstorageCuda(3 * 3 * kStride * 5);
  CudaVector<Scalar> BstorageCuda(3 * 3 * kStride * 5);

  AstorageCuda = Astorage;
  DoCudaCuSquare3x3<Scalar, kStride>(AstorageCuda, BstorageCuda);
  Bstorage = BstorageCuda;

  for (int block = 0; block < 5 * kStride; ++block) {
    auto Bstrided = BV[block];
    Matrix<Scalar, 3, 3> Res =
        static_cast<Scalar>(block % 7 + 1) * static_cast<Scalar>(block % 7 + 1) * Am * Am;
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        EXPECT_EQ(Bstrided(row, col), Res(row, col));
      }
    }
  }
}

template <typename Scalar, krylov::Direction aDir, krylov::Direction bDir, krylov::Direction cDir>
static void TestLargeMM() {
  struct Sizes {
    int m, n, k;
  };

  Sizes combinations[] = {
      {16, 16, 16},
      {16, 32, 16},
      {16, 16, 32},
      {32, 16, 16},
      {13, 5, 7},
      {13, 7, 5},
      {7, 5, 13},
      {5, 7, 13},
      {7, 13, 5},
      {5, 13, 7},
      {13, 5, 1},
      {13, 1, 5},
      {1, 5, 13},
      {5, 1, 13},
      {1, 13, 5},
      {5, 13, 1},
      {33, 34, 256}};
  for (auto [m, n, k] : combinations) {
    Matrix<Scalar, krylov::kDynamic, krylov::kDynamic, aDir> A(m, k);
    Matrix<Scalar, krylov::kDynamic, krylov::kDynamic, bDir> B(k, n);
    A.SetRandom(123);
    B.SetRandom(234);
    Matrix<Scalar, krylov::kDynamic, krylov::kDynamic, cDir> C = A * B;
    //
    CudaMatrix<Scalar, krylov::kDynamic, krylov::kDynamic, aDir> Ag(A);
    CudaMatrix<Scalar, krylov::kDynamic, krylov::kDynamic, bDir> Bg(B);
    CudaMatrix<Scalar, krylov::kDynamic, krylov::kDynamic, cDir> Cg(C.Rows(), C.Cols());
    Cg = Ag * Bg;
    //
    Matrix<Scalar, krylov::kDynamic, krylov::kDynamic, cDir> Ch(Cg);
    EXPECT_TRUE(test::IsNearRTol(Ch, C, test::GetTol<Scalar>(2 * k)));
    //
    A.SetRandom(345);
    B.SetRandom(456);
    C += Scalar(2.0) * A * B;
    //
    Ag = A;
    Bg = B;
    Cg += Scalar(2.0) * Ag * Bg;
    //
    Ch = Cg;
    EXPECT_TRUE(test::IsNear(C, Ch, test::GetTol<Scalar>(2 * k + m * n)));
    //
    C -= Scalar(2.0) * A * B;
    Cg -= Scalar(2.0) * Ag * Bg;
    //
    Ch = Cg;
    EXPECT_TRUE(test::IsNearRTol(Ch, C, test::GetTol<Scalar>(2 * k + m * n)));
    //
    Matrix<Scalar, krylov::kDynamic, krylov::kDynamic, cDir> F(C);
    F.SetRandom(789);
    C = F + Scalar(10.0 / 3.0) * A * B - F;
    CudaMatrix<Scalar, krylov::kDynamic, krylov::kDynamic, cDir> Fg(F);
    Cg = Fg + Scalar(10.0 / 3.0) * Ag * Bg - Fg;
    //
    Ch = Cg;
    EXPECT_TRUE(test::IsNearRTol(Ch, C, test::GetTol<Scalar>(2 * k + 3 * m * n)));
    //
    {
      constexpr auto kDyn = krylov::kDynamic;
      Matrix<Scalar, kDyn, kDyn, aDir, krylov::Ownership::Owner, kDyn> AA(m, k, m + k);
      Matrix<Scalar, kDyn, kDyn, bDir, krylov::Ownership::Owner, kDyn> BB(k, n, k + n);
      AA.SetRandom(123);
      BB.SetRandom(234);
      Matrix<Scalar, kDyn, kDyn, cDir, krylov::Ownership::Owner, kDyn> CC(m, n, m + n);
      CC = AA * BB;
      Matrix<Scalar, kDyn, kDyn, aDir, krylov::Ownership::Cuda, kDyn> AAg(m, k, m + k);
      AAg = AA;
      Matrix<Scalar, kDyn, kDyn, bDir, krylov::Ownership::Cuda, kDyn> BBg(k, n, k + n);
      BBg = BB;
      Matrix<Scalar, kDyn, kDyn, cDir, krylov::Ownership::Cuda, kDyn> CCg(m, n, m + n);
      CCg = AAg * BBg;
      Matrix<Scalar, kDyn, kDyn, cDir, krylov::Ownership::Owner, kDyn> CCh(CCg);
      EXPECT_TRUE(test::IsNearRTol(CC, CCh, test::GetTol<Scalar>(2 * k)));
    }
  }
}

template <
    typename Scalar,
    int m,
    int n,
    int k,
    krylov::Direction kDir = krylov::Direction::ColMajor>
static void TestFixedSizeProduct() {
  Matrix<Scalar, m, k, kDir> A;
  Matrix<Scalar, k, n, kDir> B;
  Matrix<Scalar, m, n, kDir> C;
  A.SetRandom(static_cast<unsigned int>(m + n + k));
  B.SetRandom(static_cast<unsigned int>(2 + m - n - k));
  C = A * B;
  //
  CudaMatrix<Scalar, m, k, kDir> Ad(A);
  CudaMatrix<Scalar, k, n, kDir> Bd(B);
  CudaMatrix<Scalar, m, n, kDir> Cd;
  Cd = Ad * Bd;
  //
  Matrix<Scalar, m, n, kDir> Ch(Cd);
  EXPECT_TRUE(test::IsNearRTol(C, Ch, test::GetTol<Scalar>(2 * k)));
  //
  Scalar scale{(Scalar)3.14159};
  C = scale * A * B;
  Cd = scale * Ad * Bd;
  //
  Ch = Cd;
  EXPECT_TRUE(test::IsNearRTol(C, Ch, scale * test::GetTol<Scalar>(2 * k)));
}

namespace {

template <
    typename HostMatrixType,
    typename HostInputType,
    typename HostOutputType,
    typename DeviceMatrixType>
void TestMatrixApply(
    HostMatrixType const& C,
    HostInputType const& V,
    HostOutputType& W,
    DeviceMatrixType const& dC) {
  using Scalar = std::remove_const_t<typename HostInputType::Scalar>;
  C.Apply(V, W);
  auto gpuV = ToCuda(V);
  auto gpuW = ToCuda(W);
  dC.Apply(gpuV, gpuW);
  HostOutputType h_W(gpuW); // same orientation as gpuW
  HostOutputType diffMat(h_W.Rows(), h_W.Cols());
  diffMat = h_W - W;
  EXPECT_GT(h_W.Norm(), Scalar(0));
  EXPECT_GT(W.Norm(), Scalar(0));
  EXPECT_LT(diffMat.Norm(), Scalar(2) * std::numeric_limits<Scalar>::epsilon() * W.Norm());
}

} // namespace

template <
    typename Scalar,
    typename CRIdx,
    typename Ptr,
    krylov::Direction kDirX,
    krylov::Direction kDirAX>
static void TestSparseMatrixVectorProduct() {
  constexpr int numCols = 6;
  constexpr int numRows = 4;
  //
  DynamicArray<Ptr> rowPtr({0, 1, 3, 3, 6});
  DynamicArray<CRIdx> colIdx({0, 0, 1, 0, 2, 5});
  DynamicArray<Scalar> values({Scalar(1), Scalar(2), Scalar(3), Scalar(4), Scalar(5), Scalar(6)});
  SparseMatrix<Scalar, CRIdx, Ptr> C(numCols, rowPtr, colIdx, values);
  //--- Convert to CSR matrix on GPU
  krylov::CudaCsrMatrix<Scalar, CRIdx, Ptr> gpuC(C);
  //---- Compare mat-vec product
  {
    ColumnVector<Scalar, numCols> x;
    x(0, 0) = Scalar(1);
    for (int ii = 1; ii < numCols; ++ii) {
      x(ii, 0) = -Scalar(2) * x(ii - 1, 0);
    }
    ColumnVector<Scalar, numRows> y;
    TestMatrixApply(C, x, y, gpuC);
  }
  //--- Compare Sparse * Mat product
  {
    Matrix<Scalar, numCols, 3, kDirX> V;
    V.SetRandom(123);
    Matrix<Scalar, numRows, 3, kDirAX> W;
    TestMatrixApply(C, V, W, gpuC);
  }
}

template <typename Scalar, typename CRIdx, typename Ptr>
static void TestSparseMatrixVectorProduct() {
  TestSparseMatrixVectorProduct<
      Scalar,
      CRIdx,
      Ptr,
      krylov::Direction::ColMajor,
      krylov::Direction::ColMajor>();
  TestSparseMatrixVectorProduct<
      Scalar,
      CRIdx,
      Ptr,
      krylov::Direction::ColMajor,
      krylov::Direction::RowMajor>();
  TestSparseMatrixVectorProduct<
      Scalar,
      CRIdx,
      Ptr,
      krylov::Direction::RowMajor,
      krylov::Direction::ColMajor>();
  TestSparseMatrixVectorProduct<
      Scalar,
      CRIdx,
      Ptr,
      krylov::Direction::RowMajor,
      krylov::Direction::RowMajor>();
}

template <typename Scalar, typename CRIdx, typename Ptr, krylov::Direction kDir>
static void TestSparseMatrixTransposeVectorProduct() {
  constexpr int numCols = 6;
  constexpr int numRows = 4;
  DynamicArray<Ptr> rowPtr({0, 1, 3, 3, 6});
  DynamicArray<CRIdx> colIdx({0, 0, 1, 0, 2, 5});
  DynamicArray<Scalar> values({Scalar(1), Scalar(2), Scalar(3), Scalar(4), Scalar(5), Scalar(6)});
  SparseMatrix<Scalar, CRIdx, Ptr> C(numCols, rowPtr, colIdx, values);
  //--- Convert to CSR matrix on GPU
  krylov::CudaCsrMatrix<Scalar, CRIdx, Ptr> gpuC(C);
  //--- Check Transpose(Sparse) * Vec
  {
    ColumnVector<Scalar, numRows> x;
    x.SetRandom(357);
    ColumnVector<Scalar, numCols> y;
    C.TransposeApply(x, y);
    CudaVector<Scalar, numRows> gpuX(x);
    CudaVector<Scalar, numCols> gpuY(y.Rows());
    gpuC.TransposeApply(gpuX, gpuY);
    ColumnVector<Scalar, numCols> h_y(gpuY);
    Matrix<Scalar, krylov::kDynamic, krylov::kDynamic> diff(h_y.Rows(), 1);
    diff = h_y - y;
    EXPECT_GT(h_y.Norm(), Scalar(0));
    EXPECT_GT(y.Norm(), Scalar(0));
    EXPECT_LT(diff.Norm(), Scalar(2) * std::numeric_limits<Scalar>::epsilon() * y.Norm());
  }
  //--- Compare Transpose(Sparse) * Mat product
  {
    Matrix<Scalar, numRows, 3, kDir> V;
    V.SetRandom(123);
    Matrix<Scalar, numCols, 3, kDir> W;
    C.TransposeApply(V, W);
    CudaMatrix<Scalar, numRows, 3, kDir> gpuV(V);
    auto gpuW = CudaMatrix<Scalar, numCols, 3, kDir>::Zero();
    gpuC.TransposeApply(gpuV, gpuW);
    Matrix<Scalar, krylov::kDynamic, krylov::kDynamic, kDir> h_W(gpuW);
    Matrix<Scalar, krylov::kDynamic, krylov::kDynamic, kDir> diffMat(h_W.Rows(), h_W.Cols());
    diffMat = h_W - W;
    EXPECT_GT(h_W.Norm(), Scalar(0));
    EXPECT_GT(W.Norm(), Scalar(0));
    EXPECT_LT(diffMat.Norm(), Scalar(2) * std::numeric_limits<Scalar>::epsilon() * W.Norm());
  }
}

template <typename Scalar, typename CRIdx, typename Ptr>
static void TestSparseMatrixTransposeVectorProduct() {
  TestSparseMatrixTransposeVectorProduct<Scalar, CRIdx, Ptr, krylov::Direction::ColMajor>();
}

template <typename Scalar, int kBlockSize>
static void TestBlockSparseMatrixVectorProduct() {
  constexpr krylov::Direction kDir = krylov::Direction::ColMajor;
  [[maybe_unused]] int numBlockRows = 4;
  int numBlockCols = 6;
  DynamicArray<int> rowPtr({0, 1, 3, 3, 6});
  MOCHI_ASSERT_VERBOSE(
      rowPtr.size() == numBlockRows + 1, "Incompatible number of block rows (%d)", numBlockRows);
  DynamicArray<int> colIdx({0, 0, 1, 0, 2, 5});
  DynamicArray<Scalar> values(colIdx.size() * kBlockSize * kBlockSize);
  auto vecValues = AsView(values);
  vecValues.SetRandom(117);
  BlockSparseMatrix<Scalar, kBlockSize> C(numBlockCols, rowPtr, colIdx, values);
  //--- Convert to BSR matrix on GPU
  krylov::CudaBsrMatrix<Scalar, kBlockSize, int, int, krylov::Direction::ColMajor> gpuC(C);
  krylov::CudaBsrMatrix<Scalar, kBlockSize, int, int, krylov::Direction::RowMajor> gpuCRow(C);
  //---- Compare mat-vec product
  {
    ColumnVector<Scalar> x(C.Cols(), 1);
    x.SetRandom(123);
    ColumnVector<Scalar> y(C.Rows(), 1);
    TestMatrixApply(C, x, y, gpuC);
    y.SetZero();
    TestMatrixApply(C, x, y, gpuCRow);
  }
  //--- Compare Sparse * Mat product
  {
    Matrix<Scalar, krylov::kDynamic, krylov::kDynamic, kDir> V(C.Cols(), 3);
    V.SetRandom(123);
    Matrix<Scalar, krylov::kDynamic, krylov::kDynamic, kDir> W(C.Rows(), 3);
    TestMatrixApply(C, V, W, gpuC);
    W.SetZero();
    TestMatrixApply(C, V, W, gpuCRow);
  }
}

TEST(CudaAlgebra, CopyHost2Host) {
  {
    Matrix<
        real,
        krylov::kDynamic,
        krylov::kDynamic,
        krylov::Direction::RowMajor,
        krylov::Ownership::Owner,
        krylov::kDynamic>
        I0(4, 4, 6);
    I0.SetConstant(real(1.23));

    Matrix<real, krylov::kDynamic, krylov::kDynamic, krylov::Direction::RowMajor> I1(2, 2);
    I1.SetConstant(real(-1));

    details::CudaCopy2D(I0.data(), I0.LeadDim(), I1.data(), I1.LeadDim(), I1.Cols(), I1.Rows());

    EXPECT_NEAR_EQ(I0(0, 0), real(-1));
    EXPECT_NEAR_EQ(I0(1, 0), real(-1));
    EXPECT_NEAR_EQ(I0(0, 1), real(-1));
    EXPECT_NEAR_EQ(I0(1, 1), real(-1));
    EXPECT_NEAR_EQ(I0(0, 2), real(1.23));
    EXPECT_NEAR_EQ(I0(0, 3), real(1.23));
    EXPECT_NEAR_EQ(I0(1, 2), real(1.23));
    EXPECT_NEAR_EQ(I0(1, 3), real(1.23));
    EXPECT_NEAR_EQ(I0(2, 0), real(1.23));
    EXPECT_NEAR_EQ(I0(2, 1), real(1.23));
    EXPECT_NEAR_EQ(I0(2, 2), real(1.23));
    EXPECT_NEAR_EQ(I0(2, 3), real(1.23));
    EXPECT_NEAR_EQ(I0(3, 0), real(1.23));
    EXPECT_NEAR_EQ(I0(3, 1), real(1.23));
    EXPECT_NEAR_EQ(I0(3, 2), real(1.23));
    EXPECT_NEAR_EQ(I0(3, 3), real(1.23));
  }
  {
    Matrix<real> I0(3, 4);
    I0.SetConstant(real(1.23));
    //--- Use a leading dimension larger than the storage for I0
    Matrix<
        real,
        krylov::kDynamic,
        krylov::kDynamic,
        krylov::Direction::ColMajor,
        krylov::Ownership::Owner,
        krylov::kDynamic>
        I1(2, 2, 16);
    I1.SetConstant(real(-1));

    details::CudaCopy2D(I0.data(), I0.LeadDim(), I1.data(), I1.LeadDim(), I1.Rows(), I1.Cols());

    EXPECT_NEAR_EQ(I0(0, 0), real(-1));
    EXPECT_NEAR_EQ(I0(0, 1), real(-1));
    EXPECT_NEAR_EQ(I0(0, 2), real(1.23));
    EXPECT_NEAR_EQ(I0(0, 3), real(1.23));
    EXPECT_NEAR_EQ(I0(1, 0), real(-1));
    EXPECT_NEAR_EQ(I0(1, 1), real(-1));
    EXPECT_NEAR_EQ(I0(1, 2), real(1.23));
    EXPECT_NEAR_EQ(I0(1, 3), real(1.23));
    EXPECT_NEAR_EQ(I0(2, 0), real(1.23));
    EXPECT_NEAR_EQ(I0(2, 1), real(1.23));
    EXPECT_NEAR_EQ(I0(2, 2), real(1.23));
    EXPECT_NEAR_EQ(I0(2, 3), real(1.23));
  }
}

TEST(CudaAlgebra, CudaMatrixEqCudaMatrix) {
  {
    Matrix<real> Ch(5, 7);
    Ch.SetConstant(real(1.23));
    CudaMatrix<real> Cd(5, 7);
    Cd = Ch;
    //
    Matrix<real> Dh(5, 7);
    Dh.SetRandom(456);
    CudaMatrix<real> Dd(Dh);
    // Copy two matrices on the device
    Cd = Dd;
    // Transfer device matrix back to host
    Ch = Cd;
    //
    EXPECT_TRUE(test::IsNear(Ch, Dh, std::numeric_limits<real>::epsilon() * 2));
  }
}

//--- This include has to be after the 'using namespace mochi'
#include <mochi_core/linear_algebra/cuda/cuda_block_jacobi_prec.h>
#include "data/krylov_solver_test_data.h"

TEST(CudaAlgebra, DenseTransfer) {
  int n = 46;
  Matrix<real> Iref(n, n / 2);
  for (int i = 0; i < Iref.Rows(); ++i) {
    for (int j = 0; j < Iref.Cols(); ++j) {
      Iref(i, j) = (i == j) ? 1.0_r : ((i < j) ? -0.25_r : 0.125_r);
    }
  }

  CudaMatrix<real> Id(Iref);
  Matrix<real> Ih(Id);
  EXPECT_TRUE(test::IsNear(Ih, Iref, std::numeric_limits<real>::epsilon() * 2));
}

TEST(CudaAlgebra, DenseZero) {
  int n = 46;
  Matrix<real> Iref(n, n / 2);
  for (int i = 0; i < Iref.Rows(); ++i) {
    for (int j = 0; j < Iref.Cols(); ++j) {
      Iref(i, j) = (i == j) ? 1.0_r : ((i < j) ? -0.25_r : 0.125_r);
    }
  }

  auto Id = CudaMatrix<real>::Zero(Iref.Rows(), Iref.Cols());
  Matrix<real> Ih(Id);
  int countDiff = 0;
  auto const tol = std::numeric_limits<real>::epsilon();
  for (int ii = 0; ii < Iref.Rows(); ++ii) {
    for (int jj = 0; jj < Iref.Cols(); ++jj) {
      countDiff += (abs(Ih(ii, jj)) > tol);
    }
  }
  EXPECT_EQ(0, countDiff);
}

TEST(CudaAlgebra, DenseScalarMultiplication) {
  int n = 27;
  auto Iref = Matrix<real>::Zero(n, n / 3);
  for (int i = 0; i < Iref.Rows(); ++i) {
    for (int j = 0; j < Iref.Cols(); ++j) {
      Iref(i, j) = (i == j) ? 1.0_r : ((i < j) ? -0.25_r : 0.125_r);
    }
  }
  auto const alpha = real(4.0);
  Matrix<real> aIref = alpha * Iref;
  //--- Transfer to the device
  CudaMatrix<real> Id(Iref.Rows(), Iref.Cols());
  Id = Iref;
  Id = alpha * Id;
  //--- Transfer back
  Matrix<real> Ih(Id);
  EXPECT_TRUE(test::IsNear(Ih, aIref, 4 * std::numeric_limits<real>::epsilon()));
}

TEST(CudaAlgebra, MatrixScaling) {
  int m = 133;
  int n = 59;
  {
    Matrix<real> A(m, n);
    Matrix<real> B(m, n);
    A.SetRandom(m);
    B.SetRandom(n);
    auto alpha = real(1.414213562373095);
    for (int ii = 0; ii < m; ++ii) {
      for (int jj = 0; jj < n; ++jj) {
        B(ii, jj) = alpha * A(ii, jj);
      }
    }
    CudaMatrix<real> Ag(A);
    Ag *= alpha;
    A = Ag;
    EXPECT_TRUE(test::IsNearRTol(A, B, test::GetTol<real>(m * n)));
  }
  {
    Matrix<real> A(m, n);
    Matrix<real> B(m, n);
    A.SetRandom(m);
    B.SetRandom(n);
    auto alpha = real(1.414213562373095);
    for (int ii = 0; ii < m; ++ii) {
      for (int jj = 0; jj < n; ++jj) {
        B(ii, jj) = A(ii, jj) / alpha;
      }
    }
    CudaMatrix<real> Ag(A);
    Ag /= alpha;
    A = Ag;
    EXPECT_TRUE(test::IsNearRTol(A, B, test::GetTol<real>(m * n)));
  }
}

TEST(CudaAlgebra, Dot) {
  for (int m : {5, 59, 99}) {
    Matrix<real> a(m, 1);
    a.SetRandom(m);
    CudaMatrix<real> ag(a);
    Matrix<real> b(m, 1);
    b.SetRandom(1);
    CudaMatrix<real> bg(b);
    EXPECT_NEAR_RTOL(a.Dot(b), ag.Dot(bg), 2 * std::numeric_limits<real>::epsilon());
  }
}

TEST(CudaAlgebra, NormSqr) {
  for (int m : {1, 133}) {
    for (int n : {1, 59}) {
      Matrix<real> A(m, n);
      A.SetRandom(m + n);
      CudaMatrix<real> Ag(A);
      EXPECT_NEAR_RTOL(A.NormSqr(), Ag.NormSqr(), 2 * std::numeric_limits<real>::epsilon());
    }
  }
}

TEST(CudaAlgebra, CudaTranspose) {
  TestTranspose<float>();
}

TEST(CudaAlgebra, StridedMatrixDevice) {
  TestStridedMatrixCuda<float, 1>();
  TestStridedMatrixCuda<float, 32>();
}

TEST(CudaAlgebra, LargeMatrixProduct) {
  constexpr krylov::Direction col = krylov::Direction::ColMajor;
  constexpr krylov::Direction row = krylov::Direction::RowMajor;
  TestLargeMM<real, col, col, col>();
  TestLargeMM<real, row, col, col>();
  TestLargeMM<real, col, row, col>();
  TestLargeMM<real, col, col, row>();
  TestLargeMM<real, row, row, col>();
  TestLargeMM<real, row, col, row>();
  TestLargeMM<real, col, row, row>();
  TestLargeMM<real, row, row, row>();
}

TEST(CudaAlgebra, FixedMatrixProduct) {
  TestFixedSizeProduct<real, 4, 4, 4>();
  TestFixedSizeProduct<real, 8, 8, 8>();
  TestFixedSizeProduct<real, 4, 4, 8>();
  TestFixedSizeProduct<real, 3, 1, 3>();
}

TEST(CudaAlgebra, SparseMatrixVectorProduct) {
  // All supported specializations of CudaCsrMatrix::Apply are tested.
  TestSparseMatrixVectorProduct<float, int, int>();
  TestSparseMatrixVectorProduct<float, int64_t, int64_t>();
  TestSparseMatrixVectorProduct<double, int, int>();
  TestSparseMatrixVectorProduct<double, int64_t, int64_t>();
}

TEST(CudaAlgebra, TransposeSparseMatrixVectorProduct) {
  // All supported specializations of CudaCsrMatrix::TransposeApply are tested.
  TestSparseMatrixTransposeVectorProduct<float, int, int>();
  TestSparseMatrixTransposeVectorProduct<float, int64_t, int64_t>();
  TestSparseMatrixTransposeVectorProduct<double, int, int>();
  TestSparseMatrixTransposeVectorProduct<double, int64_t, int64_t>();
}

TEST(CudaAlgebra, BlockSparseMatrixVectorProduct) {
  /// TODO Add testing for row & column storage of X and AX
  TestBlockSparseMatrixVectorProduct<float, 2>();
  TestBlockSparseMatrixVectorProduct<float, 3>();
  TestBlockSparseMatrixVectorProduct<float, 4>();
  TestBlockSparseMatrixVectorProduct<float, 5>();
  TestBlockSparseMatrixVectorProduct<double, 2>();
  TestBlockSparseMatrixVectorProduct<double, 3>();
  TestBlockSparseMatrixVectorProduct<double, 4>();
  TestBlockSparseMatrixVectorProduct<double, 5>();
}

//////

template <typename Scalar>
static void PcgOnCuda_impl() {
  DynamicArray<int> rowPtr, colIdx;
  DynamicArray<Scalar> values, bValues, solValues;

  int matrixSize = kPsdMatrixSize[0];

  rowPtr.resize(matrixSize + 1, 0);
  rowPtr[matrixSize] = kPsdMatrixNnz;
  int ir = kPsdMatrixRows[0];
  for (int ipos = 0; ipos < kPsdMatrixNnz; ++ipos) {
    if (kPsdMatrixRows[ipos] > ir) {
      rowPtr[ir + 1] = ipos;
      ir = kPsdMatrixRows[ipos];
    }
  }

  colIdx.assign(kPsdMatrixCols, kPsdMatrixCols + std::size(kPsdMatrixCols));
  values.assign(kPsdMatrixVals, kPsdMatrixVals + std::size(kPsdMatrixVals));
  bValues.assign(kPsdMatrixRhs, kPsdMatrixRhs + std::size(kPsdMatrixRhs));
  solValues.assign(kPsdMatrixOut, kPsdMatrixOut + std::size(kPsdMatrixOut));

  ColumnVectorView<Scalar> b(bValues.data(), matrixSize);
  ColumnVectorView<Scalar> ref(solValues.data(), matrixSize);
  SparseMatrix<Scalar, int, int> A(matrixSize, rowPtr, colIdx, values);

  auto relativeTol = std::numeric_limits<Scalar>::epsilon();
  {
    CudaVector<Scalar> gpuB(b);
    auto gpuSol = CudaVector<Scalar>::Zero(matrixSize);
    krylov::CudaCsrMatrix<Scalar, int, int> gpuA(A);
    auto opP = [](auto const& x, auto& Px) { Px = x; };
    auto opA = [&gpuA](auto const& x, auto& Ax) { gpuA.Apply(x, Ax); };
    krylov::StatusResidualL2 stopper{relativeTol, Scalar(1.e-17), Scalar(1.e10)};
    [[maybe_unused]] auto info = krylov::PCG(opA, gpuB, gpuSol, opP, matrixSize, stopper);

    ColumnVector<Scalar> diff(ref);
    ColumnVector<Scalar> xSol(gpuSol);
    diff = ref - xSol;
    EXPECT_NEAR_RTOL(diff.Norm(), Scalar{0}, Scalar(2e-2) * ref.Norm());
  }

  for (auto precType : {PreconditionerType::None, PreconditionerType::Jacobi}) {
    int size = kPsdMatrixSize[0];
    // Build system matrix.
    std::vector<NdArray<int, 2>> pattern;
    pattern.reserve(kPsdMatrixNnz);
    for (int i = 0; i < kPsdMatrixNnz; ++i) {
      pattern.emplace_back(kPsdMatrixRows[i], kPsdMatrixCols[i]);
    }
    //
    SparseMatrix<Scalar> inputMat(size, MakeSparsityGraph(std::move(pattern)));
    std::copy(kPsdMatrixVals, kPsdMatrixVals + kPsdMatrixNnz, inputMat.Values().data());
    // Build RHS vector.
    ColumnVector<Scalar> inputVec(size);
    std::copy(kPsdMatrixRhs, kPsdMatrixRhs + size, inputVec.data());
    // Build proposed solution
    ColumnVector<Scalar> expected(size);
    std::copy(kPsdMatrixOut, kPsdMatrixOut + size, expected.data());
    //
    KrylovSolverParams pcgParams;
    pcgParams.solverType = LinearSolverType::CudaCG;
    pcgParams.preconditionerType = precType;
    pcgParams.relTol = static_cast<double>(relativeTol);
    pcgParams.absTol = 1.e-17;
    pcgParams.relDivTol = 1.e10;
    LinearSolver<Scalar> pcgSolver(pcgParams);
    EXPECT_EQ(pcgSolver.GetParams().solverType, LinearSolverType::CudaCG);
    auto outputVec = ColumnVector<Scalar>::Zero(size);
    [[maybe_unused]] auto pcgStatus = pcgSolver.Solve(inputMat, inputVec, outputVec);
    ColumnVector<Scalar> delta = expected - outputVec;
    EXPECT_NEAR_RTOL(delta.Norm(), Scalar{0}, Scalar(2e-2) * expected.Norm());
  }
}

TEST(CudaAlgebra, PcgOnCuda) {
  PcgOnCuda_impl<float>();
  PcgOnCuda_impl<double>();
}

template <typename Scalar>
static void GMResOnCuda_impl() {
  DynamicArray<int> rowPtr, colIdx;
  DynamicArray<Scalar> values, bValues, solValues;

  int matrixSize = kPsdMatrixSize[0];
  rowPtr.resize(matrixSize + 1, 0);
  rowPtr[matrixSize] = kPsdMatrixNnz;
  int ir = kPsdMatrixRows[0];
  for (int ipos = 0; ipos < kPsdMatrixNnz; ++ipos) {
    if (kPsdMatrixRows[ipos] > ir) {
      rowPtr[ir + 1] = ipos;
      ir = kPsdMatrixRows[ipos];
    }
  }

  colIdx.assign(kPsdMatrixCols, kPsdMatrixCols + std::size(kPsdMatrixCols));
  values.assign(kPsdMatrixVals, kPsdMatrixVals + std::size(kPsdMatrixVals));
  bValues.assign(kPsdMatrixRhs, kPsdMatrixRhs + std::size(kPsdMatrixRhs));
  solValues.assign(kPsdMatrixOut, kPsdMatrixOut + std::size(kPsdMatrixOut));

  ColumnVectorView<Scalar> b(bValues.data(), matrixSize);
  ColumnVectorView<Scalar> ref(solValues.data(), matrixSize);
  SparseMatrix<Scalar, int, int> A(matrixSize, rowPtr, colIdx, values);

  auto relativeTol = std::numeric_limits<Scalar>::epsilon();
  {
    CudaVector<Scalar> gpuB(b);
    auto gpuSol = CudaVector<Scalar>::Zero(matrixSize);
    krylov::CudaCsrMatrix<Scalar, int, int> gpuA(A);
    auto opP = [](auto const& x, auto& Px) { Px = x; };
    auto opA = [&gpuA](auto const& x, auto& Ax) { gpuA.Apply(x, Ax); };
    krylov::StatusImplicitResidualNorm<Scalar> stopper{relativeTol, Scalar(1.e-17), Scalar(1.e10)};
    [[maybe_unused]] auto info = krylov::GMRes(opA, gpuB, gpuSol, opP, matrixSize, stopper);
    ColumnVector<Scalar> diff(ref);
    ColumnVector<Scalar> xSol(gpuSol);
    diff = ref - xSol;
    EXPECT_NEAR_RTOL(diff.Norm(), Scalar{0}, Scalar(2.e-2) * ref.Norm());
  }

  for (auto precType : {PreconditionerType::None, PreconditionerType::Jacobi}) {
    int size = kPsdMatrixSize[0];
    // Build system matrix.
    std::vector<NdArray<int, 2>> pattern;
    pattern.reserve(kPsdMatrixNnz);
    for (int i = 0; i < kPsdMatrixNnz; ++i) {
      pattern.emplace_back(kPsdMatrixRows[i], kPsdMatrixCols[i]);
    }
    //
    SparseMatrix<Scalar> inputMat(size, MakeSparsityGraph(std::move(pattern)));
    std::copy(kPsdMatrixVals, kPsdMatrixVals + kPsdMatrixNnz, inputMat.Values().data());
    // Build RHS vector.
    ColumnVector<Scalar> inputVec(size);
    std::copy(kPsdMatrixRhs, kPsdMatrixRhs + size, inputVec.data());
    // Build proposed solution
    ColumnVector<Scalar> expected(size);
    std::copy(kPsdMatrixOut, kPsdMatrixOut + size, expected.data());
    //
    KrylovSolverParams gmresParams;
    gmresParams.solverType = LinearSolverType::CudaGMRES;
    gmresParams.preconditionerType = precType;
    gmresParams.relTol = static_cast<double>(relativeTol);
    gmresParams.absTol = 1.e-17;
    gmresParams.relDivTol = 1.e10;
    LinearSolver<Scalar> gmresSolver(gmresParams);
    EXPECT_EQ(gmresSolver.GetParams().solverType, LinearSolverType::CudaGMRES);
    auto outputVec = ColumnVector<Scalar>::Zero(size);
    [[maybe_unused]] auto gmresStatus = gmresSolver.Solve(inputMat, inputVec, outputVec);
    ColumnVector<Scalar> delta = expected - outputVec;
    EXPECT_NEAR_RTOL(delta.Norm(), Scalar{0}, Scalar(2.e-2) * expected.Norm());
  }
}

// TODO(T239710450): platform010 fails in buck with error "device kernel image is invalid".
#if MOCHI_PLATFORM_LINUX
TEST(CudaAlgebra, DISABLED_GMResOnCuda) {
#else
TEST(CudaAlgebra, GMResOnCuda) {
#endif
  GMResOnCuda_impl<float>();
  GMResOnCuda_impl<double>();
}

//
// Test the preconditioners CudaBlockJacobiPrec
//

namespace mochi::test {

template <typename Scalar, typename Matrix>
static void EvalBlockJacobiPrec(Matrix& A) {
  int nrows = A.Rows();
  ColumnVector<Scalar> hb(nrows);
  for (int ii = 0; ii < nrows; ++ii) {
    hb(ii, 0) = Scalar(ii) + 1;
  }
  CudaVector<Scalar> b(hb);
  CudaVector<Scalar> jb(nrows);
  {
    krylov::CudaJacobiPrec<Scalar> J(A);
    J(b, jb);
    ColumnVector<Scalar> h_jb(jb);
    EXPECT_NEAR_EQ(h_jb[0], Scalar(0.5));
    EXPECT_NEAR_EQ(h_jb[1], Scalar(2) / Scalar(3));
    EXPECT_NEAR_EQ(h_jb[2], Scalar(3) / Scalar(4));
    EXPECT_NEAR_EQ(h_jb[3], Scalar(4) / Scalar(2));
    EXPECT_NEAR_EQ(h_jb[4], Scalar(5) / Scalar(2));
    EXPECT_NEAR_EQ(h_jb[5], Scalar(6) / Scalar(2));
  }
  {
    krylov::CudaBlockJacobiPrec<Scalar, 2> J(A);
    J(b, jb);
    ColumnVector<Scalar> h_jb(jb);
    EXPECT_NEAR_EQ(h_jb[0], Scalar(1));
    EXPECT_NEAR_EQ(h_jb[1], Scalar(1));
    EXPECT_NEAR_RTOL(h_jb[2], Scalar(1.4285714), 0.0001);
    EXPECT_NEAR_RTOL(h_jb[3], Scalar(2.7142857), 0.0001);
    EXPECT_NEAR_RTOL(h_jb[4], Scalar(5.3333333), 0.0001);
    EXPECT_NEAR_RTOL(h_jb[5], Scalar(5.6666666), 0.0001);
  }
  {
    krylov::CudaBlockJacobiPrec<Scalar, 3> J(A);
    J(b, jb);
    ColumnVector<Scalar> h_jb(jb);
    EXPECT_NEAR_RTOL(h_jb[0], Scalar(1.22222222), 0.0001);
    EXPECT_NEAR_RTOL(h_jb[1], Scalar(1.44444444), 0.0001);
    EXPECT_NEAR_RTOL(h_jb[2], Scalar(1.11111111), 0.0001);
    EXPECT_NEAR_EQ(h_jb[3], Scalar(7));
    EXPECT_NEAR_EQ(h_jb[4], Scalar(10));
    EXPECT_NEAR_EQ(h_jb[5], Scalar(8));
  }
  {
    krylov::CudaBlockJacobiPrec<Scalar, 6> J(A);
    J(b, jb);
    ColumnVector<Scalar> h_jb(jb);
    EXPECT_NEAR_RTOL(h_jb[0], Scalar(1.7719298), 0.0001);
    EXPECT_NEAR_RTOL(h_jb[1], Scalar(2.5438596), 0.0001);
    EXPECT_NEAR_RTOL(h_jb[2], Scalar(3.8596491), 0.0001);
    EXPECT_NEAR_RTOL(h_jb[3], Scalar(9.8947368), 0.0001);
    EXPECT_NEAR_RTOL(h_jb[4], Scalar(11.92982456), 0.0001);
    EXPECT_NEAR_RTOL(h_jb[5], Scalar(8.96491228), 0.0001);
  }
}

} // namespace mochi::test

template <typename Scalar, typename CRIdx, typename Ptr>
static void TestCudaBlockJacobiBsr() {
  {
    DynamicArray<Ptr> rp{0, 2, 5, 7};
    DynamicArray<CRIdx> ci{0, 1, 0, 1, 2, 1, 2};
    DynamicArray<Scalar> va{2, -1, 0,  0, -1, 3, -1, 0,  0, -1, 4, -1, 0,  0,
                            0, 0,  -1, 2, -1, 0, 0,  -1, 2, -1, 0, 0,  -1, 2};
    int matrixSize = 6;
    BlockSparseMatrix<Scalar, 2, CRIdx, Ptr> C(matrixSize / 2, rp, ci, va);
    krylov::CudaBsrMatrix<Scalar, 2, CRIdx, Ptr> dC(C);
    test::EvalBlockJacobiPrec<Scalar>(dC);
  }
  {
    DynamicArray<Ptr> rp{0, 2, 4};
    DynamicArray<CRIdx> ci{0, 1, 0, 1};
    DynamicArray<Scalar> va{2, -1, 0,  0, 0,  0, -1, 3, -1, 0,  0, 0,  0, -1, 4, -1, 0,  0,
                            0, 0,  -1, 2, -1, 0, 0,  0, 0,  -1, 2, -1, 0, 0,  0, 0,  -1, 2};
    int matrixSize = 6;
    BlockSparseMatrix<Scalar, 3, CRIdx, Ptr> C(matrixSize / 3, rp, ci, va);
    krylov::CudaBsrMatrix<Scalar, 3, CRIdx, Ptr> dC(C);
    test::EvalBlockJacobiPrec<Scalar>(dC);
  }
}

TEST(CudaAlgebra, CudaBlockJacobiBsr) {
  // All supported specializations of CudaJacobiPrec with CudaBsrMatrix are tested.
  TestCudaBlockJacobiBsr<float, int, int>();
  TestCudaBlockJacobiBsr<double, int, int>();
} // TEST(CudaAlgebra, CudaBlockJacobiBsr)

template <typename Scalar, typename CRIdx, typename Ptr>
static void TestCudaBlockJacobiCsr() {
  DynamicArray<Ptr> rp{0, 2, 5, 8, 11, 14, 16};
  DynamicArray<CRIdx> ci{0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5};
  DynamicArray<Scalar> va{2, -1, -1, 3, -1, -1, 4, -1, -1, 2, -1, -1, 2, -1, -1, 2};
  SparseMatrix<Scalar, CRIdx, Ptr> C(6, rp, ci, va);
  krylov::CudaCsrMatrix<Scalar, CRIdx, Ptr> dC(C);
  test::EvalBlockJacobiPrec<Scalar>(dC);
}

TEST(CudaAlgebra, CudaBlockJacobiCsr) {
  // All supported specializations of CudaJacobiPrec with CudaCsrMatrix are tested.
  TestCudaBlockJacobiCsr<float, int, int>();
  TestCudaBlockJacobiCsr<float, int64_t, int64_t>();
  TestCudaBlockJacobiCsr<double, int, int>();
  TestCudaBlockJacobiCsr<double, int64_t, int64_t>();
} // TEST(CudaAlgebra, CudaBlockJacobi)

// TODO(T239710450): platform010 fails in buck with error "device kernel image is invalid".
#if MOCHI_PLATFORM_LINUX
TEST(CudaAlgebra, DISABLED_CudaGMResKernelNorm) {
#else
TEST(CudaAlgebra, CudaGMResKernelNorm) {
#endif
  using Scalar = real;
  cudaStream_t stream1 = nullptr;
  MOCHI_CUDA_CHECK(cudaStreamCreate(&stream1));
  MOCHI_DEFER(cudaStreamDestroy(stream1));
  MOCHI_CUDA_CHECK(cudaDeviceSynchronize());
  MOCHI_CUDA_CHECK_LAST();
  for (int n : {32, 64, 99, 1001}) {
    ColumnVector<Scalar> h_x(n);
    h_x.SetRandom(123);
    Scalar h_norm = h_x.Norm();
    CudaVector<Scalar> d_x(h_x);
    CudaVector<Scalar> d_buffer(n);
    CudaVector<Scalar> d_norm(1);
    MOCHI_CUDA_CHECK(cudaDeviceSynchronize());
    MOCHI_CUDA_CHECK_LAST();
    mochi::details::gmres::NormL2(n, d_x.data(), d_norm.data(), d_buffer.data(), n, stream1);
    MOCHI_CUDA_CHECK_LAST();
    MOCHI_CUDA_CHECK(cudaStreamSynchronize(stream1));
    MOCHI_CUDA_CHECK(cudaDeviceSynchronize());
    Scalar h_d_norm = 0;
    MOCHI_CUDA_CHECK(cudaMemcpy(&h_d_norm, d_norm.data(), sizeof(Scalar), cudaMemcpyDeviceToHost));
    MOCHI_CUDA_CHECK(cudaDeviceSynchronize());
    MOCHI_CUDA_CHECK_LAST();
    EXPECT_NEAR_RTOL(h_norm, h_d_norm, Scalar(2 * n) * std::numeric_limits<Scalar>::epsilon());
  }
} // TEST(CudaAlgebra, CudaGMResKernelNorm)

// TODO(T239710450): platform010 fails in buck with error "device kernel image is invalid".
#if MOCHI_PLATFORM_LINUX
TEST(CudaAlgebra, DISABLED_CudaGMResKernelGemv) {
#else
TEST(CudaAlgebra, CudaGMResKernelGemv) {
#endif
  using Scalar = real;
  cudaStream_t stream1 = nullptr;
  MOCHI_CUDA_CHECK(cudaStreamCreate(&stream1));
  MOCHI_DEFER(cudaStreamDestroy(stream1));
  MOCHI_CUDA_CHECK(cudaDeviceSynchronize());
  MOCHI_CUDA_CHECK_LAST();
  for (int n : {64, 99, 1024, 2345}) {
    for (int k : {1, 2, 4, 43}) {
      ColumnVector<Scalar> h_x(k);
      h_x.SetRandom(34);
      Matrix<Scalar> h_Q(n, k);
      h_Q.SetRandom(23);
      //
      ColumnVector<Scalar> h_y(n);
      h_y.SetConstant(Scalar(1));
      ColumnVector<Scalar> y0(h_y);
      //
      int* d_k = nullptr;
      MOCHI_CUDA_CHECK(cudaMalloc((void**)&d_k, sizeof(int)));
      MOCHI_DEFER(cudaFree(d_k));
      MOCHI_CUDA_CHECK(cudaMemcpy(d_k, &k, sizeof(int), cudaMemcpyDefault));
      MOCHI_CUDA_CHECK_LAST();
      //
      CudaVector<Scalar> d_x(k);
      d_x = h_x;
      CudaMatrix<Scalar> d_Q(n, k);
      d_Q = h_Q;
      CudaVector<Scalar> d_y(n);
      d_y = h_y;
      MOCHI_CUDA_CHECK(cudaDeviceSynchronize());
      MOCHI_CUDA_CHECK_LAST();
      mochi::details::gmres::Gemv(
          n, d_k, d_Q.Data(), d_Q.LeadDim(), d_x.Data(), d_y.Data(), stream1);
      MOCHI_CUDA_CHECK_LAST();
      MOCHI_CUDA_CHECK(cudaStreamSynchronize(stream1));
      MOCHI_CUDA_CHECK(cudaDeviceSynchronize());
      MOCHI_CUDA_CHECK_LAST();
      ColumnVector<Scalar> h_d_y(n);
      h_d_y = d_y;
      MOCHI_CUDA_CHECK(cudaDeviceSynchronize());
      MOCHI_CUDA_CHECK_LAST();
      h_y -= h_Q * h_x;
      for (int ii = 0; ii < h_y.Rows(); ++ii) {
        EXPECT_NEAR_RTOL(
            h_y(ii), h_d_y(ii), Scalar(2 * k + 1) * std::numeric_limits<Scalar>::epsilon());
      }
    }
  }
} // TEST(CudaAlgebra, CudaGMResKernelGemv)

namespace mochi::test {
template <typename FactorizationType>
void CudaDirectSolveMultipleRHS() {
  //
  // Example extracted from
  // https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuDSS/simple/simple.cpp
  //

  int n = 5;
  int nnz = 11;

  std::vector<int> csr_offsets(n + 1);
  std::vector<int> csr_columns(nnz);
  std::vector<real> csr_values(nnz);

  ColumnVector<real> x_values(n), b_values(n);

  /* Initialize host memory for A and b */
  int i = 0;
  csr_offsets[i++] = 0;
  csr_offsets[i++] = 2;
  csr_offsets[i++] = 4;
  csr_offsets[i++] = 8;
  csr_offsets[i++] = 9;
  csr_offsets[i++] = 11;

  i = 0;
  csr_columns[i++] = 0;
  csr_columns[i++] = 2;
  csr_columns[i++] = 1;
  csr_columns[i++] = 2;
  csr_columns[i++] = 0;
  csr_columns[i++] = 1;
  csr_columns[i++] = 2;
  csr_columns[i++] = 4;
  csr_columns[i++] = 3;
  csr_columns[i++] = 2;
  csr_columns[i++] = 4;

  i = 0;
  csr_values[i++] = 4.0;
  csr_values[i++] = 1.0;
  csr_values[i++] = 3.0;
  csr_values[i++] = 2.0;
  csr_values[i++] = 1.0;
  csr_values[i++] = 2.0;
  csr_values[i++] = 5.0;
  csr_values[i++] = 1.0;
  csr_values[i++] = 1.0;
  csr_values[i++] = 1.0;
  csr_values[i++] = 2.0;

  /* Note: Right-hand side b is initialized with values which correspond
     to the exact solution vector {1, 2, 3, 4, 5} */
  i = 0;
  b_values[i++] = 7.0;
  b_values[i++] = 12.0;
  b_values[i++] = 25.0;
  b_values[i++] = 4.0;
  b_values[i++] = 13.0;

  for (i = 0; i < n; ++i) {
    x_values(i) = real(i + 1);
  }

  SparseMatrix<real, int, int, std::vector> A(n, csr_offsets, csr_columns, csr_values);
  auto const tol = real(200.0 * std::numeric_limits<real>::epsilon());
  {
    CudaVector<real> xCuda(x_values), bCuda(b_values);
    xCuda.SetZero();
    //--- Use SparseMatrix (host) as input
    FactorizationType direct(A);
    direct(bCuda, xCuda);
    ColumnVector<real> diff(xCuda);
    diff = diff - x_values;
    EXPECT_LT(diff.Norm(), x_values.Norm() * tol);
  }
  {
    krylov::CudaCsrMatrix<real> ACuda(A);
    CudaVector<real> xCuda(x_values), bCuda(b_values);
    xCuda.SetZero();
    //--- Use CudaCsrMatrix (device) as input
    FactorizationType direct(ACuda);
    direct(bCuda, xCuda);
    ColumnVector<real> diff(xCuda);
    diff = diff - x_values;
    EXPECT_LT(diff.Norm(), x_values.Norm() * tol);
    //--- Solve it again
    xCuda.SetZero();
    diff.SetZero();
    direct(bCuda, xCuda);
    diff = xCuda;
    diff = diff - x_values;
    EXPECT_LT(diff.Norm(), x_values.Norm() * tol);
    //--- Solve multiple right hand sides
    Matrix<real> y(5, 2);
    y.SetConstant(real(1));
    y(1, 1) = real(-1);
    y(3, 1) = real(-1);
    Matrix<real> f(5, 2);
    f(0, 0) = 5.0;
    f(1, 0) = 5.0;
    f(2, 0) = 9.0;
    f(3, 0) = 1.0;
    f(4, 0) = 3.0;
    f(0, 1) = 5.0;
    f(1, 1) = -1.0;
    f(2, 1) = 5.0;
    f(3, 1) = -1.0;
    f(4, 1) = 3.0;
    CudaMatrix<real> d_f(f), d_u(5, 2);
    direct(d_f, d_u);
    Matrix<real> u(d_u);
    u = u - y;
    EXPECT_LT(u.Norm(), y.Norm() * tol);
  }
}
} // namespace mochi::test

TEST(CudaAlgebra, CudaSparseDirectSolveCsrInput) {
  mochi::test::CudaDirectSolveMultipleRHS<krylov::CudaSparseCholesky<real>>();
  mochi::test::CudaDirectSolveMultipleRHS<krylov::CudaSparseLU<real>>();
#if MOCHI_USE_CUDSS
  mochi::test::CudaDirectSolveMultipleRHS<krylov::CudaSparseLDLt<real>>();
#endif
}

TEST(CudaAlgebra, CudaSparseCholesky) {
  //
  // Example extracted from
  // https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuDSS/simple/simple.cpp
  //

  int n = 5;

  std::vector<int> bsr_offsets(2, 0);
  bsr_offsets[1] = 1;
  std::vector<int> bsr_columns(1, 0);
  std::vector<real> bsr_values(
      {4, 0, 1, 0, 0, 0, 3, 2, 0, 0, 1, 2, 5, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 2});

  ColumnVector<real> x_values(n), b_values(n);

  /* Note: Right-hand side b is initialized with values which correspond
     to the exact solution vector {1, 2, 3, 4, 5} */
  b_values[0] = 7.0;
  b_values[1] = 12.0;
  b_values[2] = 25.0;
  b_values[3] = 4.0;
  b_values[4] = 13.0;

  for (int i = 0; i < n; ++i) {
    x_values(i) = real(i + 1);
  }

  BlockSparseMatrix<real, 5, int, int, std::vector> A(1, bsr_offsets, bsr_columns, bsr_values);
  auto const tol = real(200.0 * std::numeric_limits<real>::epsilon());
  {
    krylov::CudaBsrMatrix<real, 5, int, int, krylov::Direction::ColMajor> ACuda(A);
    CudaVector<real> xCuda(x_values), bCuda(b_values);
    xCuda.SetZero();
    //--- Use CudaBsrMatrix (device) as input
    krylov::CudaSparseCholesky<real> Chol(ACuda);
    Chol(bCuda, xCuda);
    ColumnVector<real> diff(xCuda);
    diff = diff - x_values;
    EXPECT_LT(diff.Norm(), x_values.Norm() * tol);
    //--- Solve it again
    xCuda.SetZero();
    diff.SetZero();
    Chol(bCuda, xCuda);
    diff = xCuda;
    diff = diff - x_values;
    EXPECT_LT(diff.Norm(), x_values.Norm() * tol);
  }
  {
    krylov::CudaBsrMatrix<real, 5, int, int, krylov::Direction::RowMajor> ACuda(A);
    CudaVector<real> xCuda(x_values), bCuda(b_values);
    xCuda.SetZero();
    //--- Use CudaBsrMatrix (device) as input
    krylov::CudaSparseCholesky<real> Chol(ACuda);
    Chol(bCuda, xCuda);
    ColumnVector<real> diff(xCuda);
    diff = diff - x_values;
    EXPECT_LT(diff.Norm(), x_values.Norm() * tol);
    //--- Solve it again
    xCuda.SetZero();
    diff.SetZero();
    Chol(bCuda, xCuda);
    diff = xCuda;
    diff = diff - x_values;
    EXPECT_LT(diff.Norm(), x_values.Norm() * tol);
  }
}

namespace mochi::test {

template <typename FactorizationType>
void CudaDirectSolveBsrInput() {
  static constexpr int kBlock = 2;
  int const px = 3;
  auto const nx = int(std::pow(2.0, px) - 1);
  int const py = 2;
  auto const ny = int(std::pow(2.0, py) - 1);
  int const n = nx * ny * kBlock;
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
  auto Absp = ToBlockSparseMatrix<kBlock>(A, true);

  ColumnVector<real> x(n);
  x.SetRandom(123);

  ColumnVector<real> f(n);
  Absp.Apply(x, f);

  {
    CudaVector<real> xCuda(x), bCuda(f);
    xCuda.SetZero();
    //--- Use BsrMatrix (host) as input
    FactorizationType direct(Absp);
    auto const tol = real(200.0 * std::numeric_limits<real>::epsilon());
    direct(bCuda, xCuda);
    ColumnVector<real> diff(xCuda);
    diff = diff - x;
    EXPECT_LT(diff.Norm(), x.Norm() * tol);
    //--- Solve it again
    xCuda.SetZero();
    diff.SetZero();
    direct(bCuda, xCuda);
    diff = xCuda;
    diff = diff - x;
    EXPECT_LT(diff.Norm(), x.Norm() * tol);
  }
  {
    CudaVector<real> xCuda(x), bCuda(f);
    xCuda.SetZero();
    //--- Use the default ColMajor orientation for each individual block
    //--- Use CudaBsrMatrix (device) as input
    krylov::CudaBsrMatrix<real, kBlock> ACuda(Absp);
    FactorizationType direct(ACuda);
    auto const tol = real(200.0 * std::numeric_limits<real>::epsilon());
    direct(bCuda, xCuda);
    ColumnVector<real> diff(xCuda);
    diff = diff - x;
    EXPECT_LT(diff.Norm(), x.Norm() * tol);
  }
  {
    CudaVector<real> xCuda(x), bCuda(f);
    xCuda.SetZero();
    krylov::CudaBsrMatrix<real, kBlock, int, int, krylov::Direction::RowMajor> ACuda(Absp);
    //--- Use CudaBsrMatrix (device) as input
    FactorizationType direct(ACuda);
    auto const tol = real(200.0 * std::numeric_limits<real>::epsilon());
    direct(bCuda, xCuda);
    ColumnVector<real> diff(xCuda);
    diff = diff - x;
    EXPECT_LT(diff.Norm(), x.Norm() * tol);
  }
}
} // namespace mochi::test

TEST(CudaAlgebra, CudaSparseDirectSolveBsrInput) {
  mochi::test::CudaDirectSolveBsrInput<krylov::CudaSparseCholesky<real>>();
  mochi::test::CudaDirectSolveBsrInput<krylov::CudaSparseLU<real>>();
#if MOCHI_USE_CUDSS
  mochi::test::CudaDirectSolveBsrInput<krylov::CudaSparseLDLt<real>>();
#endif
}

#if MOCHI_USE_CUDSS
TEST(CudaAlgebra, CudaSparseLDLtNegPosEigenvalues) {
  //
  // Example extracted from
  // https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuDSS/simple/simple.cpp
  //

  int n = 5;
  int nnz = 11;

  std::vector<int> csr_offsets(n + 1);
  std::vector<int> csr_columns(nnz);
  std::vector<real> csr_values(nnz);

  ColumnVector<real> x_values(n), b_values(n);
  for (int i = 0; i < n; ++i) {
    x_values(i) = real(i + 1);
  }

  /* Initialize host memory for A and b */
  int i = 0;
  csr_offsets[i++] = 0;
  csr_offsets[i++] = 2;
  csr_offsets[i++] = 4;
  csr_offsets[i++] = 8;
  csr_offsets[i++] = 9;
  csr_offsets[i++] = 11;

  //
  i = 0;
  csr_columns[i++] = 0;
  csr_columns[i++] = 2;
  csr_columns[i++] = 1;
  csr_columns[i++] = 2;
  csr_columns[i++] = 0;
  csr_columns[i++] = 1;
  csr_columns[i++] = 2;
  csr_columns[i++] = 4;
  csr_columns[i++] = 3;
  csr_columns[i++] = 2;
  csr_columns[i++] = 4;

  //
  i = 0;
  csr_values[i++] = 1.0;
  csr_values[i++] = 1.0;
  csr_values[i++] = 0.0;
  csr_values[i++] = 2.0;
  csr_values[i++] = 1.0;
  csr_values[i++] = 2.0;
  csr_values[i++] = 2.0;
  csr_values[i++] = 1.0;
  csr_values[i++] = -2.0;
  csr_values[i++] = 1.0;
  csr_values[i++] = -1.0;
  //
  b_values[0] = 4.0;
  b_values[1] = 6.0;
  b_values[2] = 16.0;
  b_values[3] = -8.0;
  b_values[4] = -2.0;
  //
  SparseMatrix<real, int, int, std::vector> H(n, csr_offsets, csr_columns, csr_values);
  krylov::CudaCsrMatrix<real> HCuda(H);
  auto const tol = real(200.0 * std::numeric_limits<real>::epsilon());
  //
  CudaVector<real> xCuda(x_values), bCuda(b_values);
  xCuda.SetZero();
  krylov::CudaSparseLDLt<real> LDLt(HCuda);
  LDLt(bCuda, xCuda);
  ColumnVector<real> diff(xCuda);
  diff = diff - x_values;
  auto const xNorm = x_values.Norm();
  EXPECT_LT(diff.Norm(), xNorm * tol);
  //--- Solve it again -- which does not do the factorization
  xCuda.SetZero();
  diff.SetZero();
  LDLt(bCuda, xCuda);
  diff = xCuda;
  diff = diff - x_values;
  EXPECT_LT(diff.Norm(), xNorm * tol);
}
#endif

TEST(CudaAlgebra, CudaSparseLUNonSymmetric) {
  //
  // Example extracted from
  // https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuDSS/simple/simple.cpp
  //

  int n = 5;
  int nnz = 11;

  std::vector<int> csr_offsets(n + 1);
  std::vector<int> csr_columns(nnz);
  std::vector<real> csr_values(nnz);

  ColumnVector<real> x_values(n), b_values(n);

  /* Initialize host memory for A and b */
  int i = 0;
  csr_offsets[i++] = 0;
  csr_offsets[i++] = 2;
  csr_offsets[i++] = 4;
  csr_offsets[i++] = 7;
  csr_offsets[i++] = 8;
  csr_offsets[i++] = 11;

  i = 0;
  csr_columns[i++] = 0;
  csr_columns[i++] = 2;
  csr_columns[i++] = 1;
  csr_columns[i++] = 2;
  csr_columns[i++] = 1;
  csr_columns[i++] = 2;
  csr_columns[i++] = 4;
  csr_columns[i++] = 3;
  csr_columns[i++] = 0;
  csr_columns[i++] = 2;
  csr_columns[i++] = 4;

  i = 0;
  csr_values[i++] = 4.0;
  csr_values[i++] = 1.0;
  csr_values[i++] = 3.0;
  csr_values[i++] = 2.0;
  csr_values[i++] = 2.0;
  csr_values[i++] = 5.0;
  csr_values[i++] = 1.0;
  csr_values[i++] = 1.0;
  csr_values[i++] = 1.0;
  csr_values[i++] = 1.0;
  csr_values[i++] = 2.0;

  /* Note: Right-hand side b is initialized with values which correspond
     to the exact solution vector {1, 2, 3, 4, 5} */
  i = 0;
  b_values[i++] = 7.0;
  b_values[i++] = 12.0;
  b_values[i++] = 24.0;
  b_values[i++] = 4.0;
  b_values[i++] = 14.0;

  for (i = 0; i < n; ++i) {
    x_values(i) = real(i + 1);
  }

  SparseMatrix<real, int, int, std::vector> K(n, csr_offsets, csr_columns, csr_values);
  krylov::CudaCsrMatrix<real> KCuda(K);
  auto const tol = real(200.0 * std::numeric_limits<real>::epsilon());

  CudaVector<real> xCuda(x_values), bCuda(b_values);
  xCuda.SetZero();
  krylov::CudaSparseLU<real> LU(KCuda);
  LU(bCuda, xCuda);
  ColumnVector<real> diff(xCuda);
  diff = diff - x_values;
  EXPECT_LT(diff.Norm(), x_values.Norm() * tol);
  //--- Solve it again -- which does not do the factorization
  xCuda.SetZero();
  diff.SetZero();
  LU(bCuda, xCuda);
  diff = xCuda;
  diff = diff - x_values;
  EXPECT_LT(diff.Norm(), x_values.Norm() * tol);
}

#endif // MOCHI_USE_CUDA
