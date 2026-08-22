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
#include <mochi_core/linear_algebra/cuda/cuda_bsr_matrix.h>
#include <mochi_core/linear_algebra/cuda/cuda_csr_matrix.h>
#include <mochi_core/linear_algebra/cuda/cuda_matrix.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/matrix_expressions.h>
#include <mochi_core/linear_algebra/sparse_matrix.h>
#include <mochi_core/linear_algebra/strided_matrix.h>
#include <mochi_core/linear_algebra/utils/matrix_concepts.h>

#include <gtest/gtest.h>

using namespace mochi;

TEST(MatrixConcepts, ScalarType) {
  static_assert(std::is_same_v<ScalarType<Matrix<float>>, float>);
  static_assert(std::is_same_v<ScalarType<Matrix<double, 3, 3>>, double>);
  static_assert(std::is_same_v<ScalarType<MatrixView<int>>, int>);
  static_assert(std::is_same_v<ScalarType<MatrixView<int64_t const, 2>>, int64_t>);
  static_assert(std::is_same_v<ScalarType<SparseMatrix<float>>, float>);
  static_assert(std::is_same_v<ScalarType<SparseMatrixView<double>>, double>);
  static_assert(std::is_same_v<ScalarType<BlockSparseMatrix<int, 3>>, int>);
  static_assert(std::is_same_v<ScalarType<BlockSparseMatrixView<int64_t const, 4>>, int64_t>);
  static_assert(std::is_same_v<ScalarType<StridedMatrix<float, 3, 3, 6>>, float>);
  static_assert(std::is_same_v<ScalarType<StridedMatrixView<double const, 4, 4, 12>>, double>);
#if MOCHI_USE_CUDA
  static_assert(std::is_same_v<ScalarType<CudaMatrix<real>>, real>);
  static_assert(std::is_same_v<ScalarType<CudaMatrixView<real, 3, 3>>, real>);
  static_assert(std::is_same_v<ScalarType<CudaMatrixView<real const>>, real>);
  static_assert(std::is_same_v<ScalarType<krylov::CudaCsrMatrix<real>>, real>);
  static_assert(std::is_same_v<ScalarType<krylov::CudaBsrMatrix<real, 3>>, real>);
#endif // MOCHI_USE_CUDA
}

TEST(MatrixConcepts, IsMatrixLike) {
  static_assert(!IsMatrixLike<real>);
  static_assert(IsMatrixLike<Matrix<real>>);
  static_assert(IsMatrixLike<Matrix<real, 5, 7>>);
  static_assert(IsMatrixLike<MatrixView<real>>);
  static_assert(IsMatrixLike<MatrixView<real const>>);
  static_assert(IsMatrixLike<StridedMatrix<real, 3, 3, 6>>);
  static_assert(IsMatrixLike<StridedMatrixView<real const, 3, 3, 1>>);
  static_assert(!IsMatrixLike<SparseMatrix<real>>);
  static_assert(!IsMatrixLike<SparseMatrixView<real>>);
  static_assert(!IsMatrixLike<BlockSparseMatrix<real, 3>>);
  static_assert(!IsMatrixLike<BlockSparseMatrixView<real, 3>>);
#if MOCHI_USE_CUDA
  static_assert(IsMatrixLike<CudaMatrix<real>>);
  static_assert(IsMatrixLike<CudaMatrix<real, 5, 7>>);
  static_assert(!IsMatrixLike<krylov::CudaCsrMatrix<real>>);
  static_assert(!IsMatrixLike<krylov::CudaBsrMatrix<real, 3>>);
#endif // MOCHI_USE_CUDA

  {
    using L = Matrix<real>;
    using Ta = details::BinOp<L, L, ops::Add>;
    static_assert(IsMatrixLike<Ta>);
    using Ea = details::ScaledExpr<real, Ta>;
    static_assert(IsMatrixLike<Ea>);
    using Tm = details::BinOp<L, L, ops::Mul>;
    static_assert(IsMatrixLike<Tm>);
    using Em = details::ScaledExpr<real, Tm>;
    static_assert(IsMatrixLike<Em>);
    using Ts = details::BinOp<L, L, ops::Sub>;
    static_assert(IsMatrixLike<Ts>);
    using Es = details::ScaledExpr<real, Ts>;
    static_assert(IsMatrixLike<Es>);
  }
}

TEST(MatrixConcepts, IsSparseMatrix) {
  static_assert(!IsSparseMatrix<real>);
  static_assert(!IsSparseMatrix<Matrix<real>>);
  static_assert(!IsSparseMatrix<Matrix<real, 5, 7>>);
  static_assert(!IsSparseMatrix<MatrixView<real>>);
  static_assert(!IsSparseMatrix<MatrixView<real, 5, 7>>);
  static_assert(!IsSparseMatrix<StridedMatrix<real, 3, 3, 6>>);
  static_assert(!IsSparseMatrix<StridedMatrixView<real const, 3, 3, 1>>);
  static_assert(IsSparseMatrix<SparseMatrix<real>>);
  static_assert(IsSparseMatrix<SparseMatrixView<real>>);
  static_assert(!IsSparseMatrix<BlockSparseMatrix<real, 3>>);
  static_assert(!IsSparseMatrix<BlockSparseMatrixView<real, 3>>);
#if MOCHI_USE_CUDA
  static_assert(!IsSparseMatrix<CudaMatrix<real>>);
  static_assert(!IsSparseMatrix<CudaMatrix<real, 5, 7>>);
  static_assert(IsSparseMatrix<krylov::CudaCsrMatrix<real>>);
  static_assert(!IsSparseMatrix<krylov::CudaBsrMatrix<real, 3>>);
#endif // MOCHI_USE_CUDA
}

TEST(MatrixConcepts, IsBlockSparseMatrix) {
  static_assert(!IsBlockSparseMatrix<real>);
  static_assert(!IsBlockSparseMatrix<Matrix<real>>);
  static_assert(!IsBlockSparseMatrix<Matrix<real, 5, 7>>);
  static_assert(!IsBlockSparseMatrix<MatrixView<real>>);
  static_assert(!IsBlockSparseMatrix<MatrixView<real, 5, 7>>);
  static_assert(!IsBlockSparseMatrix<StridedMatrix<real, 3, 3, 6>>);
  static_assert(!IsBlockSparseMatrix<StridedMatrixView<real const, 3, 3, 1>>);
  static_assert(!IsBlockSparseMatrix<SparseMatrix<real>>);
  static_assert(!IsBlockSparseMatrix<SparseMatrixView<real>>);
  static_assert(IsBlockSparseMatrix<BlockSparseMatrix<real, 3>>);
  static_assert(IsBlockSparseMatrix<BlockSparseMatrixView<real, 3>>);
#if MOCHI_USE_CUDA
  static_assert(!IsBlockSparseMatrix<CudaMatrix<real>>);
  static_assert(!IsBlockSparseMatrix<CudaMatrix<real, 5, 7>>);
  static_assert(!IsBlockSparseMatrix<krylov::CudaCsrMatrix<real>>);
  static_assert(IsBlockSparseMatrix<krylov::CudaBsrMatrix<real, 3>>);
#endif // MOCHI_USE_CUDA
}

TEST(MatrixConcepts, IsMatrixExpr) {
  static_assert(!IsMatrixExpr<Matrix<real>>);
  static_assert(!IsMatrixExpr<Matrix<real, 5, 7>>);
  static_assert(!IsMatrixExpr<MatrixView<real>>);
  static_assert(!IsMatrixExpr<MatrixView<real const>>);
  static_assert(!IsMatrixExpr<StridedMatrix<real, 3, 3, 6>>);
  static_assert(!IsMatrixExpr<StridedMatrixView<real const, 3, 3, 1>>);
  static_assert(!IsMatrixExpr<SparseMatrix<real>>);
  static_assert(!IsMatrixExpr<SparseMatrixView<real>>);
  static_assert(!IsMatrixExpr<BlockSparseMatrix<real, 3>>);
  static_assert(!IsMatrixExpr<BlockSparseMatrixView<real, 3>>);
#if MOCHI_USE_CUDA
  static_assert(!IsMatrixExpr<CudaMatrix<real>>);
  static_assert(!IsMatrixExpr<CudaMatrix<real, 5, 7>>);
  static_assert(!IsMatrixExpr<krylov::CudaCsrMatrix<real>>);
  static_assert(!IsMatrixExpr<krylov::CudaBsrMatrix<real, 3>>);
#endif // MOCHI_USE_CUDA

  {
    using L = Matrix<real>;
    using Ta = details::BinOp<L, L, ops::Add>;
    static_assert(IsMatrixExpr<Ta>);
    using Ea = details::ScaledExpr<real, Ta>;
    static_assert(IsMatrixExpr<Ea>);
    using Tm = details::BinOp<L, L, ops::Mul>;
    static_assert(IsMatrixExpr<Tm>);
    using Em = details::ScaledExpr<real, Tm>;
    static_assert(IsMatrixExpr<Em>);
    using Ts = details::BinOp<L, L, ops::Sub>;
    static_assert(IsMatrixExpr<Ts>);
    using Es = details::ScaledExpr<real, Ts>;
    static_assert(IsMatrixExpr<Es>);
  }
}

TEST(MatrixConcepts, IsMatrix) {
  static_assert(!IsMatrix<real>);
  static_assert(IsMatrix<Matrix<real>>);
  static_assert(IsMatrix<Matrix<real, 5, 7>>);
  static_assert(IsMatrix<MatrixView<real>>);
  static_assert(IsMatrix<MatrixView<real const>>);
  static_assert(!IsMatrix<StridedMatrix<real, 3, 3, 6>>);
  static_assert(!IsMatrix<StridedMatrixView<real const, 3, 3, 1>>);
  static_assert(!IsMatrix<SparseMatrix<real>>);
  static_assert(!IsMatrix<SparseMatrixView<real>>);
  static_assert(!IsMatrix<BlockSparseMatrix<real, 3>>);
  static_assert(!IsMatrix<BlockSparseMatrixView<real, 3>>);
#if MOCHI_USE_CUDA
  static_assert(IsMatrix<CudaMatrix<real>>);
  static_assert(IsMatrix<CudaMatrix<real, 5, 7>>);
  static_assert(!IsMatrix<krylov::CudaCsrMatrix<real>>);
  static_assert(!IsMatrix<krylov::CudaBsrMatrix<real, 3>>);
#endif // MOCHI_USE_CUDA

  {
    using L = Matrix<real>;
    using Ta = details::BinOp<L, L, ops::Add>;
    static_assert(!IsMatrix<Ta>);
    using Ea = details::ScaledExpr<real, Ta>;
    static_assert(!IsMatrix<Ea>);
    using Tm = details::BinOp<L, L, ops::Mul>;
    static_assert(!IsMatrix<Tm>);
    using Em = details::ScaledExpr<real, Tm>;
    static_assert(!IsMatrix<Em>);
    using Ts = details::BinOp<L, L, ops::Sub>;
    static_assert(!IsMatrix<Ts>);
    using Es = details::ScaledExpr<real, Ts>;
    static_assert(!IsMatrix<Es>);
  }
}

TEST(MatrixConcepts, IsCuda) {
  static_assert(!IsCuda<Matrix<real>>);
  static_assert(!IsCuda<Matrix<real, 5, 7>>);
  static_assert(!IsCuda<MatrixView<real>>);
  static_assert(!IsCuda<MatrixView<real const>>);
  static_assert(!IsCuda<SparseMatrix<real>>);
  static_assert(!IsCuda<SparseMatrixView<real>>);
  static_assert(!IsCuda<BlockSparseMatrix<real, 3>>);
  static_assert(!IsCuda<BlockSparseMatrixView<real, 3>>);
#if MOCHI_USE_CUDA
  static_assert(IsCuda<CudaMatrix<real>>);
  static_assert(IsCuda<CudaMatrix<real, 5, 7>>);
  static_assert(IsCuda<krylov::CudaCsrMatrix<real>>);
  static_assert(IsCuda<krylov::CudaBsrMatrix<real, 3>>);
#endif // MOCHI_USE_CUDA
}

TEST(MatrixConcepts, IsLinearOperator) {
  static_assert(!IsLinearOperator<real>);
  static_assert(IsLinearOperator<Matrix<real>>);
  static_assert(IsLinearOperator<Matrix<real, 5, 7>>);
  static_assert(IsLinearOperator<MatrixView<real>>);
  static_assert(IsLinearOperator<MatrixView<real const>>);
  static_assert(IsLinearOperator<StridedMatrix<real, 3, 3, 6>>);
  static_assert(IsLinearOperator<StridedMatrixView<real const, 3, 3, 1>>);
  static_assert(IsLinearOperator<SparseMatrix<real>>);
  static_assert(IsLinearOperator<SparseMatrixView<real>>);
  static_assert(IsLinearOperator<BlockSparseMatrix<real, 3>>);
  static_assert(IsLinearOperator<BlockSparseMatrixView<real, 3>>);
#if MOCHI_USE_CUDA
  static_assert(IsLinearOperator<CudaMatrix<real>>);
  static_assert(IsLinearOperator<CudaMatrix<real, 5, 7>>);
  static_assert(IsLinearOperator<krylov::CudaCsrMatrix<real>>);
  static_assert(IsLinearOperator<krylov::CudaBsrMatrix<real, 3>>);
#endif // MOCHI_USE_CUDA
}

TEST(MatrixConcepts, IsStridedMatrix) {
  static_assert(!IsStridedMatrix<real>);
  static_assert(!IsStridedMatrix<Matrix<real>>);
  static_assert(!IsStridedMatrix<Matrix<real, 5, 7>>);
  static_assert(!IsStridedMatrix<MatrixView<real>>);
  static_assert(!IsStridedMatrix<MatrixView<real const>>);
  static_assert(IsStridedMatrix<StridedMatrix<real, 3, 3, 6>>);
  static_assert(IsStridedMatrix<StridedMatrixView<real const, 3, 3, 1>>);
  static_assert(!IsStridedMatrix<SparseMatrix<real>>);
  static_assert(!IsStridedMatrix<SparseMatrixView<real>>);
  static_assert(!IsStridedMatrix<BlockSparseMatrix<real, 3>>);
  static_assert(!IsStridedMatrix<BlockSparseMatrixView<real, 3>>);
#if MOCHI_USE_CUDA
  static_assert(!IsStridedMatrix<CudaMatrix<real>>);
  static_assert(!IsStridedMatrix<CudaMatrix<real, 5, 7>>);
  static_assert(!IsStridedMatrix<krylov::CudaCsrMatrix<real>>);
  static_assert(!IsStridedMatrix<krylov::CudaBsrMatrix<real, 3>>);
#endif // MOCHI_USE_CUDA
}
