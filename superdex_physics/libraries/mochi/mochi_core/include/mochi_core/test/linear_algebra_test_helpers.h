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
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/sparse_matrix.h>
#include <mochi_core/linear_algebra/utils/matrix_conversions.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/rand_utils.h>

#include <algorithm>
#include <numeric>
#include <set>
#include <type_traits>
#include <utility>

namespace mochi::test {

/**
 * @brief Create a @ref SparseMatrix for the 3-point 1D Laplacian on a uniform grid with homogeneous
 * Dirichlet conditions.
 *
 * @param[in] nx Number of rows in the matrix (so nx must be > 0)
 * @return Sparse matrix
 *
 * @note The uniform mesh size will be 1 / (nx + 1).
 */
inline auto Make3ptLaplacianMatrix(int nx) {
  MOCHI_ASSERT_VERBOSE(nx > 0);
  //
  DynamicArray<int> rowPtr(nx + 1);
  rowPtr[0] = 0;
  DynamicArray<int> colIdx(3 * nx - 2);
  DynamicArray<real> values(3 * nx - 2);
  //
  if (nx == 1) {
    rowPtr[1] = 1;
    colIdx[0] = 0;
    values[0] = 4_r;
    return SparseMatrix<real, int, int>{
        nx, std::move(rowPtr), std::move(colIdx), std::move(values)};
  }
  //
  rowPtr[1] = 2;
  colIdx[0] = 0;
  colIdx[1] = 1;
  values[0] = 2_r;
  values[1] = -1_r;
  for (int i = 1; i + 1 < nx; ++i) {
    int const pos = rowPtr[i];
    rowPtr[i + 1] = pos + 3;
    colIdx[pos] = i - 1;
    colIdx[pos + 1] = i;
    colIdx[pos + 2] = i + 1;
    values[pos] = -1_r;
    values[pos + 1] = 2_r;
    values[pos + 2] = -1_r;
  }
  int const pos = rowPtr[nx - 1];
  rowPtr[nx] = pos + 2;
  colIdx[pos] = nx - 2;
  colIdx[pos + 1] = nx - 1;
  values[pos] = -1_r;
  values[pos + 1] = 2_r;
  //
  auto const oneOverH = real(nx + 1);
  for (auto& val : values) {
    val *= oneOverH;
  }
  return SparseMatrix<real, int, int>{nx, std::move(rowPtr), std::move(colIdx), std::move(values)};
}

/**
 * @brief Create a 2D Laplacian SPD matrix as a dense matrix.
 *
 * @details Creates a block-diagonal structure where each node in the 2D grid has a 4 on the
 * diagonal and -1 for each neighbor (5-point stencil).
 *
 * @tparam Scalar Scalar type (e.g., float, double).
 * @tparam kBlockSize Block size for each grid node.
 * @param[in] nx Number of grid points in the x-direction.
 * @param[in] ny Number of grid points in the y-direction.
 * @return Dense SPD matrix of size (nx * ny * kBlockSize) x (nx * ny * kBlockSize).
 */
template <typename Scalar, int kBlockSize>
Matrix<Scalar> Create2dLaplacianMatrix(int nx, int ny) {
  int const n = nx * ny * kBlockSize;

  auto A = Matrix<Scalar>::Zero(n, n);
  for (int iy = 0; iy < ny; ++iy) {
    for (int ix = 0; ix < nx; ++ix) {
      int node = ix + iy * nx;
      for (int k = 0; k < kBlockSize; ++k) {
        int row = k + kBlockSize * node;
        A(row, row) = Scalar(4);
        if (iy > 0) {
          A(row, k + kBlockSize * (node - nx)) = Scalar(-1);
        }
        if (ix > 0) {
          A(row, k + kBlockSize * (node - 1)) = Scalar(-1);
        }
        if (ix + 1 < nx) {
          A(row, k + kBlockSize * (node + 1)) = Scalar(-1);
        }
        if (iy + 1 < ny) {
          A(row, k + kBlockSize * (node + nx)) = Scalar(-1);
        }
      }
    }
  }
  return A;
}

/**
 * @brief Create the graph of node-to-node connectivity for a brick [0, 1] x [0, 1] x [0, 1]
 * discretized with 8-noded hexahedra (forming an orthogonal grid).
 *
 * @param[in] nx Number of elements in the x-direction
 * @param[in] ny Number of elements in the y-direction
 * @param[in] nz Number of elements in the z-direction
 * @param[out] rowPtr Array similar to the row pointer array in a CSR matrix
 * @param[out] nodeIdx Array similar to the column index pointer array in a CSR matrix
 */
void MakeGraphBrick(int nx, int ny, int nz, DynamicArray<int>& rowPtr, DynamicArray<int>& nodeIdx);

/**
 * @brief Create a block sparse matrix (with block size kBlockSize) where the graph of
 * block-to-block connectivity is generated with the function @ref MakeGraphBrick. The output block
 * sparse matrix will be symmetric positive definite. The entries of the matrix are random numbers.
 *
 * @param[in] nx Number of elements in the x-direction
 * @param[in] ny Number of elements in the y-direction
 * @param[in] nz Number of elements in the z-direction
 * @param[out] rowPtr Array of block-row pointer
 * @param[out] colIdx Array of block-column indices
 * @param[out] values Array of numerical values for the block sparse matrix.
 */
template <typename Scalar, int kBlockSize>
void MakeBlockSparseData(
    int nx,
    int ny,
    int nz,
    DynamicArray<int>& rowPtr,
    DynamicArray<int>& colIdx,
    DynamicArray<Scalar>& values) {
  //--- Pointers and indices.
  MakeGraphBrick(nx, ny, nz, rowPtr, colIdx);

  //-- Max number of non-zeros per row.
  int const numBlockRows = isize(rowPtr) - 1;
  int const numBlockCols = numBlockRows;
  int maxNnzPerRow = 0;
  for (int ir = 0; ir < numBlockRows; ++ir) {
    maxNnzPerRow = Max(maxNnzPerRow, kBlockSize * (rowPtr[ir + 1] - rowPtr[ir]));
  }

  //--- Values.
  values.resize(colIdx.size() * kBlockSize * kBlockSize);
  BlockSparseMatrixView<Scalar, kBlockSize, int, int> view{
      numBlockCols, MakeSpan(rowPtr), MakeSpan(colIdx), MakeSpan(values)};
  view.SetZero();

  //--- Make matrix symmetric, positive-definite.
  for (int ir = 0; ir + 1 < isize(rowPtr); ++ir) {
    for (int kr = rowPtr[ir]; kr < rowPtr[ir + 1]; ++kr) {
      int const ic = colIdx[kr];
      if (ic < ir) {
        bool found = false;
        for (int kc = rowPtr[ic]; kc < rowPtr[ic + 1]; ++kc) {
          if (colIdx[kc] == ir) {
            found = true;
            auto L = view.Values(ir)[kr - rowPtr[ir]];
            auto U = view.Values(ic)[kc - rowPtr[ic]];
            U.SetRandom(kr + kc, Scalar(-1), Scalar(1));
            L = U.Transpose();
            break;
          }
        }
        MOCHI_ASSERT(found);
      } else if (ic == ir) {
        auto D = view.Values(ir)[kr - rowPtr[ir]];
        RowMatrix<Scalar> A = D;
        A.SetRandom(ir, Scalar(-1), Scalar(1));
        D = A.Transpose() * A;
        Scalar const Dnorm = D.Norm();
        for (int ii = 0; ii < kBlockSize; ++ii) {
          D(ii, ii) += Scalar(maxNnzPerRow) + Dnorm;
        }
        break;
      }
    }
  }
}

template <typename Scalar, typename ColIdx, typename Ptr, int kBlockSize>
void BlockSparseDataTargetNnz(
    ColIdx numBlockRows,
    ColIdx numNonZeroBlocksPerRow,
    DynamicArray<Ptr>& rowPtr,
    DynamicArray<ColIdx>& colIdx,
    DynamicArray<Scalar>& values) {
  ColIdx const numBlockCols = numBlockRows;
  ColIdx const nnzPerBlockRow = kBlockSize * kBlockSize * numNonZeroBlocksPerRow;
  rowPtr.clear();
  rowPtr.reserve(numBlockRows + 1);
  colIdx.clear();
  colIdx.reserve(nnzPerBlockRow * numBlockRows);
  values.clear();
  values.reserve(nnzPerBlockRow * numBlockRows);
  for (int ii = 0; ii < numBlockRows; ++ii) {
    // Row pointer
    rowPtr.push_back(ii * numNonZeroBlocksPerRow);

    // Column indices
    std::set<ColIdx> tmpIdx;
    for (int jj = 0; jj < numNonZeroBlocksPerRow; ++jj) {
      // Evenly distributed blocks within the row (~worst case)
      tmpIdx.insert(
          static_cast<ColIdx>(
              ii + static_cast<int64_t>(jj) * numBlockCols / numNonZeroBlocksPerRow) %
          numBlockCols);
    }
    for (auto idx : tmpIdx) {
      colIdx.push_back(idx);
    }
    // Values
    for (int jj = 0; jj < nnzPerBlockRow; ++jj) {
      values.push_back(Scalar(ii + jj - static_cast<int>(numBlockRows) / 2));
    }
  }
  rowPtr.push_back(numBlockRows * numNonZeroBlocksPerRow);
}

/**
 * @brief Create a square block sparse matrix (with block size kBlockSize)
 * and nx * ny * nz block rows.
 *
 * @param[in] nx Number of elements in the x-direction
 * @param[in] ny Number of elements in the y-direction
 * @param[in] nz Number of elements in the z-direction
 * @return Block sparse matrix
 *
 * @see MakeBlockSparseData
 */
template <typename Scalar, int kBlockSize>
BlockSparseMatrix<Scalar, kBlockSize> MakeBlockSparseMatrix(int nx, int ny, int nz) {
  DynamicArray<int> rowPtr, colIdx;
  DynamicArray<Scalar> values;
  MakeBlockSparseData<Scalar, kBlockSize>(nx, ny, nz, rowPtr, colIdx, values);
  int const numBlockCols = isize(rowPtr) - 1;
  return {numBlockCols, std::move(rowPtr), std::move(colIdx), std::move(values)};
}

template <typename Scalar, int kBlockSize>
SparseMatrix<Scalar> MakeSparseMatrixWithBlockStructure(int nx, int ny, int nz) {
  DynamicArray<int> bRowPtr, bColIdx;
  DynamicArray<Scalar> bValues;
  MakeBlockSparseData<Scalar, kBlockSize>(nx, ny, nz, bRowPtr, bColIdx, bValues);
  int const numBlockCols = isize(bRowPtr) - 1;
  BlockSparseMatrix<Scalar, kBlockSize> ABSp(
      numBlockCols, std::move(bRowPtr), std::move(bColIdx), std::move(bValues));
  return ToSparseMatrix(ABSp);
}

template <typename Scalar, int kBlockSize, typename CRIdx = int, typename Ptr = int>
BlockSparseMatrix<Scalar, kBlockSize, CRIdx, Ptr> MakeBlockSparseMatrixWithTargetNnz(
    CRIdx numBlockRows,
    CRIdx numNonZeroBlocksPerRow) {
  auto const numBlockCols = numBlockRows;
  DynamicArray<Ptr> rowPtr;
  DynamicArray<CRIdx> colIdx;
  DynamicArray<Scalar> values;
  BlockSparseDataTargetNnz<Scalar, CRIdx, Ptr, kBlockSize>(
      numBlockRows, numNonZeroBlocksPerRow, rowPtr, colIdx, values);
  return {numBlockCols, std::move(rowPtr), std::move(colIdx), std::move(values)};
}

template <typename Scalar, typename CRIdx = int, typename Ptr = int>
SparseMatrix<Scalar, CRIdx, Ptr> MakeSparseMatrixWithTargetNnz(
    CRIdx numRows,
    CRIdx numNonZerosPerRow) {
  return ToSparseMatrix(
      MakeBlockSparseMatrixWithTargetNnz<Scalar, 1, CRIdx, Ptr>(numRows, numNonZerosPerRow));
}

/**
 * @brief Conversion from @ref Matrix to @ref NdArray.
 * @note The implementation is not optimized.
 */
template <
    int N,
    int M,
    typename Scalar,
    int kRowsAtCompileTime,
    int kColsAtCompileTime,
    krylov::Direction kMajorDirection,
    krylov::Ownership kOwnership,
    int kLeadingDim>
auto ToNdArray(
    Matrix<
        Scalar,
        kRowsAtCompileTime,
        kColsAtCompileTime,
        kMajorDirection,
        kOwnership,
        kLeadingDim> const& mat) {
  MOCHI_ASSERT(mat.Rows() == N);
  MOCHI_ASSERT(mat.Cols() == M);
  NdArray<Scalar, N, M> result;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      result[i][j] = mat(i, j);
    }
  }
  return result;
}

// Given two SparseMatrix, return the maximum value of: abs(matA(r, c) - matB(r, c))
// They do not need to have the same sparsity pattern.
template <
    typename ScalarA,
    typename CRIdxA,
    typename PtrA,
    template <typename, typename...> typename StorageA,
    typename ScalarB,
    typename CRIdxB,
    typename PtrB,
    template <typename, typename...> typename StorageB>
auto MaxAbsDifference(
    SparseMatrix<ScalarA, CRIdxA, PtrA, StorageA> const& matA,
    SparseMatrix<ScalarB, CRIdxB, PtrB, StorageB> const& matB) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT((matA.Rows() == matB.Rows()) && (matA.Cols() == matB.Cols()), "Size mismatch");
  auto const numRows = matA.Rows();
  if ((matA.Pointers() == matB.Pointers()) && (matA.Indices() == matB.Indices())) {
    // These matrices have the same sparsity.
    return MaxAbsDifference(matA.Values(), matB.Values());
  } else {
    // Compare them row-by-row
    using NonConstScalar = std::remove_const_t<ScalarA>;
    using NonConstCRIdxA = std::remove_const_t<CRIdxA>;
    auto maxDiff = NonConstScalar(0);
    for (NonConstCRIdxA r = 0; r < numRows; ++r) {
      auto indicesA = matA.Indices(r);
      auto indicesB = matB.Indices(r);
      auto valuesA = matA.Values(r);
      auto valuesB = matB.Values(r);
      int ai = 0;
      int bi = 0;
      while ((ai < isize(indicesA)) || (bi < isize(indicesB))) {
        if (ai == isize(indicesA)) {
          // All remaining values of matA on this row are zero
          maxDiff = Max(maxDiff, MaxAbs<ScalarB, PtrB>(valuesB.subspan(bi)));
          bi = isize(indicesB); // done
        } else if (bi == isize(indicesB)) {
          // All remaining values of matB on this row are zero
          maxDiff = Max(maxDiff, MaxAbs<ScalarA, PtrA>(valuesA.subspan(ai)));
          ai = isize(indicesA); // done
        } else if (indicesA[ai] < indicesB[bi]) {
          // matB(r, indicesA[ai]) is zero
          maxDiff = Max(maxDiff, Abs(valuesA[ai]));
          ++ai;
        } else if (indicesA[ai] > indicesB[bi]) {
          // matA(r, indicesB[bi]) is zero
          maxDiff = Max(maxDiff, Abs(valuesB[bi]));
          ++bi;
        } else {
          // indicesA[ai] == indices[bi], which means this column index is in the sparsity pattern
          // of both matrices.
          maxDiff = Max(maxDiff, Abs(valuesA[ai] - valuesB[bi]));
          ++ai;
          ++bi;
        }
      }
    }
    return maxDiff;
  }
}

/**
 * @brief Generate a random permutation of integers [0, n-1].
 *
 * @tparam RandomEngine Random engine type.
 * @param[in] n Size of permutation.
 * @param[in,out] rng Random engine (state will be modified).
 * @return Array containing a random permutation.
 */
template <typename RandomEngine>
DynamicArray<int> CreateRandomPermutation(int n, RandomEngine& rng) {
  DynamicArray<int> perm;
  perm.resize_noinit(n);
  std::iota(perm.begin(), perm.end(), 0);
  std::shuffle(perm.begin(), perm.end(), rng);
  return perm;
}

/**
 * @brief Apply symmetric permutation P * A * P^T to a dense matrix at the block level.
 *
 * @tparam kBlockSize Size of each block for the permutation.
 * @param[in] A Input symmetric matrix of size (numBlocks * kBlockSize) x (numBlocks * kBlockSize).
 * @param[in] blockPerm Permutation array of block indices where blockPerm[i] gives the new block
 * position for block row/col i. Size must be numBlocks = A.Rows() / kBlockSize.
 * @return Permuted matrix where result block (blockPerm[i], blockPerm[j]) = A block (i, j).
 */
template <int kBlockSize, typename Scalar>
Matrix<Scalar> ApplySymmetricPermutation(Matrix<Scalar> const& A, Span<int const> blockPerm) {
  int const n = A.Rows();
  int const numBlocks = n / kBlockSize;
  MOCHI_ASSERT(A.Cols() == n && n % kBlockSize == 0 && isize(blockPerm) == numBlocks);

  auto B = Matrix<Scalar>::Zero(n, n);
  for (int bi = 0; bi < numBlocks; ++bi) {
    for (int bj = 0; bj < numBlocks; ++bj) {
      auto srcBlock = A.Block(bi * kBlockSize, bj * kBlockSize, kBlockSize, kBlockSize);
      auto dstBlock =
          B.Block(blockPerm[bi] * kBlockSize, blockPerm[bj] * kBlockSize, kBlockSize, kBlockSize);
      dstBlock = srcBlock;
    }
  }
  return B;
}

/**
 * @brief Create a dense block-banded symmetric matrix with random values.
 *
 * @details The matrix has a block-banded structure where blocks outside the band are zero.
 * Diagonal blocks are made SPD via A^T * A pattern. Off-diagonal blocks have random values
 * and are symmetric (A(i,j) = A(j,i)^T).
 *
 * @tparam Scalar Scalar type (e.g., float, double).
 * @tparam kBlockSize Block size for the band structure.
 * @tparam RandomEngine Random engine type.
 * @param[in] numBlockRows Number of block rows (matrix size = numBlockRows * kBlockSize).
 * @param[in] bandwidth Number of block diagonals on each side (e.g. 0 = diagonal only, 1 =
 * tridiagonal).
 * @param[in,out] rng Random engine (state will be modified).
 * @param[in] enforceDiagonallyDominant If true, ensures strict diagonal dominance for guaranteed
 * SPD.
 * @return Dense symmetric matrix ready for conversion to sparse.
 */
template <typename Scalar, int kBlockSize, typename RandomEngine>
Matrix<Scalar> CreateBlockBandedMatrix(
    int numBlockRows,
    int bandwidth,
    RandomEngine& rng,
    bool enforceDiagonallyDominant) {
  MOCHI_ASSERT(numBlockRows > 0 && bandwidth >= 0, "Invalid parameters.");
  int const n = numBlockRows * kBlockSize;
  auto A = Matrix<Scalar>::Zero(n, n);

  // Fill block bands with random values.
  for (int bi = 0; bi < numBlockRows; ++bi) {
    for (int bj = Max(0, bi - bandwidth); bj <= Min(numBlockRows - 1, bi + bandwidth); ++bj) {
      auto block = A.template Block<kBlockSize, kBlockSize>(
          bi * kBlockSize, bj * kBlockSize, kBlockSize, kBlockSize);
      if (bi == bj) {
        // Diagonal block: make it SPD via A^T * A pattern.
        Matrix<Scalar, kBlockSize, kBlockSize> tmp;
        for (int i = 0; i < kBlockSize; ++i) {
          for (int j = 0; j < kBlockSize; ++j) {
            tmp(i, j) = RandomUniformValue(rng, Scalar(-1), Scalar(1));
          }
        }
        block = tmp.Transpose() * tmp;
        // Add small positive value to ensure non-singular
        for (int k = 0; k < kBlockSize; ++k) {
          block(k, k) += Scalar(0.1);
        }
      } else if (bi > bj) {
        // Lower triangular: random values
        for (int i = 0; i < kBlockSize; ++i) {
          for (int j = 0; j < kBlockSize; ++j) {
            block(i, j) = RandomUniformValue(rng, Scalar(-1), Scalar(1));
          }
        }
        // Mirror to upper triangular for symmetry
        auto upperBlock = A.template Block<kBlockSize, kBlockSize>(
            bj * kBlockSize, bi * kBlockSize, kBlockSize, kBlockSize);
        upperBlock = block.Transpose();
      }
    }
  }

  if (enforceDiagonallyDominant) {
    // Strict diagonal dominance guarantees SPD.
    for (int i = 0; i < n; ++i) {
      Scalar rowSum = Scalar(0);
      for (int j = 0; j < n; ++j) {
        if (i != j) {
          rowSum += Abs(A(i, j));
        }
      }
      A(i, i) += rowSum + Scalar(1);
    }
  }

  return A;
}

} // namespace mochi::test
