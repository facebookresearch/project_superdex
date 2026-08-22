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

#include <mochi_core/linear_algebra/matrix_operations.h>

#if MOCHI_USE_EIGEN
#include <Eigen/Dense>
#endif // MOCHI_USE_EIGEN

namespace mochi::krylov {

#if MOCHI_USE_EIGEN
namespace details {

template <typename Scalar>
static void GeneralizedSelfAdjointEigenDecomposition(
    MatrixView<Scalar> const& A,
    MatrixView<Scalar> const& M,
    Matrix<Scalar>& V,
    ColumnVector<Scalar>& D) {
  MOCHI_ASSERT_VERBOSE(A.Rows() == A.Cols(), "Input matrix is not square");
  MOCHI_ASSERT_VERBOSE(M.Rows() == M.Cols(), "Input matrix is not square");
  MOCHI_ASSERT_VERBOSE(A.Rows() == M.Rows(), "Input matrices dimensions do not match");
  using EMatType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  Eigen::Map<EMatType const> eA(A.data(), A.Rows(), A.Cols());
  Eigen::Map<EMatType const> eM(M.data(), M.Rows(), M.Cols());
  Eigen::GeneralizedSelfAdjointEigenSolver<EMatType> pencilSolver(eA, eM);
  //
  V.Resize(A.Rows(), A.Cols());
  Eigen::Map<EMatType> matV(V.Data(), V.Rows(), V.Cols());
  matV = pencilSolver.eigenvectors();
  //
  auto eigVal = pencilSolver.eigenvalues();
  D.Resize(A.Rows());
  for (int i = 0; i < A.Rows(); ++i) {
    D(i) = eigVal(i);
  }
}

template <typename Scalar>
static void SelfAdjointEigenDecomposition(
    MatrixView<Scalar const> const& A,
    Matrix<Scalar>& V,
    ColumnVector<Scalar>& D) {
  using EMatType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  MOCHI_ASSERT_VERBOSE(A.Rows() == A.Cols(), "Input matrix must be square.");
  Eigen::Map<EMatType const> eA(A.Data(), A.Rows(), A.Cols());
  Eigen::SelfAdjointEigenSolver<EMatType> eigenSolver(eA);

  V.Resize(A.Rows(), A.Cols());
  Eigen::Map<EMatType> matV(V.Data(), V.Rows(), V.Cols());
  matV = eigenSolver.eigenvectors();

  auto eigVals = eigenSolver.eigenvalues();
  D.Resize(A.Rows());
  for (int i = 0; i < A.Rows(); ++i) {
    D(i) = eigVals(i);
  }
}

} // namespace details
#endif // MOCHI_USE_EIGEN

void GeneralizedSelfAdjointEigenDecomposition(
    [[maybe_unused]] MatrixView<float> const& A,
    [[maybe_unused]] MatrixView<float> const& M,
    [[maybe_unused]] Matrix<float>& V,
    [[maybe_unused]] ColumnVector<float>& D) {
  MOCHI_ASSERT_EIGEN();
#if MOCHI_USE_EIGEN
  details::GeneralizedSelfAdjointEigenDecomposition<float>(A, M, V, D);
#endif
}

void GeneralizedSelfAdjointEigenDecomposition(
    [[maybe_unused]] MatrixView<double> const& A,
    [[maybe_unused]] MatrixView<double> const& M,
    [[maybe_unused]] Matrix<double>& V,
    [[maybe_unused]] ColumnVector<double>& D) {
  MOCHI_ASSERT_EIGEN();
#if MOCHI_USE_EIGEN
  details::GeneralizedSelfAdjointEigenDecomposition<double>(A, M, V, D);
#endif
}

void SelfAdjointEigenDecomposition(
    [[maybe_unused]] MatrixView<float const> const& A,
    [[maybe_unused]] Matrix<float>& V,
    [[maybe_unused]] ColumnVector<float>& D) {
  MOCHI_ASSERT_EIGEN();
#if MOCHI_USE_EIGEN
  details::SelfAdjointEigenDecomposition<float>(A, V, D);
#endif
}

void SelfAdjointEigenDecomposition(
    [[maybe_unused]] MatrixView<double const> const& A,
    [[maybe_unused]] Matrix<double>& V,
    [[maybe_unused]] ColumnVector<double>& D) {
  MOCHI_ASSERT_EIGEN();
#if MOCHI_USE_EIGEN
  details::SelfAdjointEigenDecomposition<double>(A, V, D);
#endif
}

} // namespace mochi::krylov
