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
#include <mochi_core/linear_algebra/block_sparse_matrix.h>
#include <mochi_core/linear_algebra/cuda/cuda_bsr_matrix.h>
#include <mochi_core/linear_algebra/cuda/cuda_csr_matrix.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/matrix_operations.h>
#include <mochi_core/linear_algebra/sparse_matrix.h>
#include <mochi_core/utils/container_utils.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/profile.h>
#include <mochi_core/utils/span.h>

#include <algorithm>
#include <numeric>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace mochi::details {

template <
    typename Scalar,
    int kRowsAtCompileTime,
    int kColsAtCompileTime,
    krylov::Direction kMajorDirection,
    krylov::Ownership kOwnership,
    int kLeadingDim>
bool IsZero(
    Matrix<
        Scalar,
        kRowsAtCompileTime,
        kColsAtCompileTime,
        kMajorDirection,
        kOwnership,
        kLeadingDim> const& mat) {
  static_assert(!krylov::IsCuda(kOwnership), "Utility not supported for CUDA matrices");
  if constexpr (kMajorDirection == mochi::krylov::Direction::RowMajor) {
    for (int i = 0; i < mat.Rows(); ++i) {
      auto matRow = mat.Row(i);
      for (int j = 0; j < mat.Cols(); ++j) {
        if (matRow(0, j)) {
          return false;
        }
      }
    }
  } else {
    static_assert(kMajorDirection == mochi::krylov::Direction::ColMajor);
    for (int j = 0; j < mat.Cols(); ++j) {
      auto matCol = mat.Col(j);
      for (int i = 0; i < mat.Rows(); ++i) {
        if (matCol(i)) {
          return false;
        }
      }
    }
  }
  return true;
}
} // namespace mochi::details

namespace mochi {

template <typename T>
struct IslandOperators;

template <typename T>
struct LowRankAugmentedMatrix;

/// @brief Convert from BlockSparseMatrix to CudaBsrMatrix.
template <
    typename Scalar,
    int kBlockSize,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
auto ToCuda(BlockSparseMatrix<Scalar, kBlockSize, CRIdx, Ptr, Storage> const& bSpMat) {
  return krylov::CudaBsrMatrix(bSpMat);
}

/// @brief Convert from SparseMatrix to CudaCsrMatrix.
template <
    typename Scalar,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
auto ToCuda(SparseMatrix<Scalar, CRIdx, Ptr, Storage> const& spMat) {
  return krylov::CudaCsrMatrix(spMat);
}

/// @brief Convert from CPU (dense) Matrix to CUDA (dense) Matrix.
template <
    typename Scalar,
    int kRowsAtCompileTime,
    int kColsAtCompileTime,
    krylov::Direction kMajorDirection,
    krylov::Ownership kOwnership,
    int kLeadingDim>
auto ToCuda(
    Matrix<
        Scalar,
        kRowsAtCompileTime,
        kColsAtCompileTime,
        kMajorDirection,
        kOwnership,
        kLeadingDim> const& mat) {
  static_assert(!krylov::IsCuda(kOwnership), "ToCuda not supported for CUDA input matrices");
  using NonConstScalar = std::remove_const_t<Scalar>;
  return CudaMatrix<
      NonConstScalar,
      kRowsAtCompileTime,
      kColsAtCompileTime,
      kMajorDirection,
      kLeadingDim>(mat);
}

//
// ToBlockSparseMatrix functions
//

/// @brief Convert from BlockSparseMatrix with some block size to BlockSparseMatrix with a different
/// block size.
///
/// @note It is a pass-through if the block sizes are the same.
/// @warning Not optimized when the block sizes are different.
template <
    int kBlockSizeOut,
    typename Scalar,
    int kBlockSizeIn,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
auto ToBlockSparseMatrix(
    BlockSparseMatrix<Scalar, kBlockSizeIn, CRIdx, Ptr, Storage> const& bSpMat) {
  static_assert(kBlockSizeOut > 0, "Inappropriate output block size");
  if constexpr (kBlockSizeIn == kBlockSizeOut) {
    return AsConstView(bSpMat);
  } else {
    // Conversion when the block sizes differ
    // The current implementation is slow as it operates two conversions.
    // Do not use in "performance" code
    // When modifying the implementation, update the function description accordingly.
    auto spMat = ToSparseMatrix(bSpMat);
    return ToBlockSparseMatrix<kBlockSizeOut>(spMat);
  }
}

/// @brief Convert from (dense) Matrix to BlockSparseMatrix.
/// @param[in] mat Input dense matrix.
/// @param[in] pruneZeros Boolean flag to indicate whether to prune the zeros or not.
/// Default value is false, i.e. no pruning.
///
/// @note The number of rows for the input matrix must be a multiple of the block size.
/// @note The number of columns for the input matrix must be a multiple of the block size.
/// @note The implementation does not support CudaMatrix as input.
template <
    int kBlockSizeOutput,
    typename Scalar,
    int kRowsAtCompileTime,
    int kColsAtCompileTime,
    krylov::Direction kMajorDirection,
    krylov::Ownership kOwnership,
    int kLeadingDim>
auto ToBlockSparseMatrix(
    Matrix<
        Scalar,
        kRowsAtCompileTime,
        kColsAtCompileTime,
        kMajorDirection,
        kOwnership,
        kLeadingDim> const& mat,
    bool pruneZeros = false) {
  static_assert(kBlockSizeOutput > 0, "Inappropriate output block size");
  static_assert(!krylov::IsCuda(kOwnership), "Utility not supported for CUDA matrices");
  using NonConstScalar = std::remove_const_t<Scalar>;
  MOCHI_ASSERT(
      mat.Rows() % kBlockSizeOutput == 0,
      "Incompatible number of rows (%ld) for blocksize (%d)",
      static_cast<long int>(mat.Rows()),
      kBlockSizeOutput);
  MOCHI_ASSERT(
      mat.Cols() % kBlockSizeOutput == 0,
      "Incompatible number of columns (%ld) for blocksize (%d)",
      static_cast<long int>(mat.Cols()),
      kBlockSizeOutput);
  int const numBlockRows = mat.Rows() / kBlockSizeOutput;
  int const numBlockCols = mat.Cols() / kBlockSizeOutput;
  DynamicArray<int> pointers;
  pointers.reserve(numBlockRows + 1);
  if (pruneZeros) {
    pointers.push_back(0);
    int count = 0;
    for (int br = 0; br < numBlockRows; ++br) {
      for (int bc = 0; bc < numBlockCols; ++bc) {
        auto const block = mat.Block(
            br * kBlockSizeOutput, bc * kBlockSizeOutput, kBlockSizeOutput, kBlockSizeOutput);
        if (!details::IsZero(block)) {
          count += 1;
        }
      }
      pointers.push_back(count);
    }
  } else {
    for (int br = 0; br < numBlockRows; ++br) {
      pointers.push_back(br * numBlockCols);
    }
    pointers.push_back(numBlockRows * numBlockCols);
  }
  DynamicArray<int> indices;
  indices.reserve(pointers[numBlockRows]);
  DynamicArray<NonConstScalar> values;
  if (pruneZeros) {
    values.resize(pointers[numBlockRows] * kBlockSizeOutput * kBlockSizeOutput, 0);
    for (int br = 0; br < numBlockRows; ++br) {
      int count = 0;
      for (int bc = 0; bc < numBlockCols; ++bc) {
        auto const block = mat.Block(
            br * kBlockSizeOutput, bc * kBlockSizeOutput, kBlockSizeOutput, kBlockSizeOutput);
        if (details::IsZero(block)) {
          continue;
        }
        indices.push_back(bc);
        //--- Use accessor-based assignment
        auto const shift =
            pointers[br] * kBlockSizeOutput * kBlockSizeOutput + count * kBlockSizeOutput;
        auto const leadDim = (pointers[br + 1] - pointers[br]) * kBlockSizeOutput;
        MatrixView<
            NonConstScalar,
            kBlockSizeOutput,
            kBlockSizeOutput,
            krylov::Direction::RowMajor,
            krylov::kDynamic>
            entryView(values.data() + shift, kBlockSizeOutput, kBlockSizeOutput, leadDim);
        entryView = block;
        count += 1;
      }
    }
  } else {
    indices.resize_noinit(numBlockRows * numBlockCols);
    auto firstRowIndices = Span(indices.data(), numBlockCols);
    std::iota(firstRowIndices.begin(), firstRowIndices.end(), 0);
    for (int br = 1; br < numBlockRows; ++br) {
      std::copy(firstRowIndices.begin(), firstRowIndices.end(), indices.data() + br * numBlockCols);
    }
    values.resize(mat.Rows() * mat.Cols());
    auto valuesAsRowMatrix = RowMatrixView<NonConstScalar>{values.data(), mat.Rows(), mat.Cols()};
    valuesAsRowMatrix = mat;
  }
  return BlockSparseMatrix<NonConstScalar, kBlockSizeOutput, int, int>(
      numBlockCols, std::move(pointers), std::move(indices), std::move(values));
}

/// @brief Convert from SparseMatrix to BlockSparseMatrix.
/// @tparam kBlockSizeOutput Block size for output matrix (must be > 0)
/// @param spmat Input sparse matrix (dimensions must be divisible by block size)
/// @return Block sparse matrix representation
///
/// @note The number of rows for the input matrix must be a multiple of the block size.
/// @note The number of columns for the input matrix must be a multiple of the block size.
template <
    int kBlockSizeOutput,
    typename Scalar,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
auto ToBlockSparseMatrix(SparseMatrix<Scalar, CRIdx, Ptr, Storage> const& spmat) {
  static_assert(kBlockSizeOutput > 0, "Inappropriate output block size");
  //
  using NonConstScalar = std::remove_const_t<Scalar>;
  using NonConstIdx = std::remove_const_t<CRIdx>;
  using NonConstPtr = std::remove_const_t<Ptr>;
  if (IsBlockable<kBlockSizeOutput>(spmat)) {
    auto bStructure = BlockedStructure<kBlockSizeOutput>(spmat);
    auto spValues = spmat.Values();
    DynamicArray<NonConstScalar> values(spValues.begin(), spValues.end());
    return BlockSparseMatrix<NonConstScalar, kBlockSizeOutput, NonConstIdx, NonConstPtr>(
        bStructure.nBlockCols,
        std::move(bStructure.ptr),
        std::move(bStructure.ndIndices),
        std::move(values));
  } else if ((spmat.Rows() % kBlockSizeOutput == 0) && (spmat.Cols() % kBlockSizeOutput == 0)) {
    //
    //--- Conversion will need to add 0 values
    //
    auto const numBlockRows = spmat.Rows() / kBlockSizeOutput;
    DynamicArray<NonConstPtr> rowPtr(numBlockRows + 1, 0);
    DynamicArray<NonConstIdx> colIdx;
    //--- Reserve memory space for worst-case scenario
    colIdx.reserve(spmat.NumNonZeros());
    NonConstIdx currentBlock = 0;
    std::unordered_set<NonConstIdx> blockCols;
    for (NonConstIdx ir = 0; ir < spmat.Rows(); ++ir) {
      auto myBlock = ir / kBlockSizeOutput;
      if (myBlock > currentBlock) {
        //--- Flush previous block columns
        rowPtr[currentBlock + 1] = rowPtr[currentBlock] + NonConstPtr(blockCols.size());
        Append(colIdx, blockCols);
        std::sort(colIdx.data() + rowPtr[currentBlock], colIdx.data() + rowPtr[currentBlock + 1]);
        blockCols.clear();
        currentBlock = myBlock;
      }
      auto cIdx = spmat.Indices(ir);
      for (size_t k = 0; k < cIdx.size(); ++k) {
        blockCols.insert(NonConstIdx(cIdx[k] / kBlockSizeOutput));
      }
    }
    //--- Final block-row
    rowPtr[currentBlock + 1] = rowPtr[currentBlock] + NonConstPtr(blockCols.size());
    Append(colIdx, blockCols);
    std::sort(colIdx.data() + rowPtr[currentBlock], colIdx.data() + rowPtr[currentBlock + 1]);
    //--- Copy values
    DynamicArray<NonConstScalar> values(
        rowPtr[numBlockRows] * kBlockSizeOutput * kBlockSizeOutput, NonConstScalar(0));
    BlockSparseMatrix<NonConstScalar, kBlockSizeOutput, NonConstIdx, NonConstPtr> bSpMat(
        spmat.Cols() / kBlockSizeOutput, std::move(rowPtr), std::move(colIdx), std::move(values));
    for (NonConstIdx ir = 0; ir < spmat.Rows(); ++ir) {
      auto myBlockRow = ir / kBlockSizeOutput;
      auto myLocalRow = ir - myBlockRow * kBlockSizeOutput;
      auto inputIdx = spmat.Indices(ir);
      auto inputVal = spmat.Values(ir);
      auto myBlockValues = bSpMat.Values(myBlockRow);
      auto myBlockIndices = bSpMat.Indices(myBlockRow);
      int blockShift = 0;
      for (size_t k = 0; k < inputIdx.size(); ++k) {
        auto myBlockCol = inputIdx[k] / kBlockSizeOutput;
        auto myLocalCol = inputIdx[k] - myBlockCol * kBlockSizeOutput;
        auto ptr = std::find(myBlockIndices.begin() + blockShift, myBlockIndices.end(), myBlockCol);
        MOCHI_ASSERT_VERBOSE(ptr != myBlockIndices.end(), "Incorrect column block index");
        blockShift = int(ptr - myBlockIndices.begin());
        myBlockValues[blockShift](myLocalRow, myLocalCol) = inputVal[k];
      }
    }
    MOCHI_ASSERT_VERBOSE(bSpMat.Rows() == spmat.Rows(), "Non-matching dimensions");
    MOCHI_ASSERT_VERBOSE(bSpMat.Cols() == spmat.Cols(), "Non-matching dimensions");
    return bSpMat;
  } else {
    MOCHI_ASSERT(
        spmat.Rows() % kBlockSizeOutput == 0,
        "Incompatible number of rows (%ld) for blocksize (%d)",
        static_cast<long int>(spmat.Rows()),
        kBlockSizeOutput);
    MOCHI_ASSERT(
        spmat.Cols() % kBlockSizeOutput == 0,
        "Incompatible number of columns (%ld) for blocksize (%d)",
        static_cast<long int>(spmat.Cols()),
        kBlockSizeOutput);
  }
  return BlockSparseMatrix<NonConstScalar, kBlockSizeOutput, NonConstIdx, NonConstPtr>();
}

/// @brief Convert from ActorPseudoMatrix to block sparse matrix.
/// @tparam kBlockSize Block size
/// @tparam kSkipMissingSparsityEntries If true, interaction matrix entries that fall outside the
/// actor matrix's sparsity pattern are silently dropped. If false (default), it is invalid to have
/// such entries (asserts in debug builds; undefined behavior in optimized builds). Skipping is
/// useful for preconditioners, which are approximate by nature and do not require the interaction
/// sparsity to be a subset of the actor sparsity.
/// @tparam Scalar Type for the numerical values
/// @param[in] in Actor pseudo-matrix
///
/// @note The actor matrix has to be a block sparse matrix (block size kBlockSize x kBlockSize).
/// @note Only the block size 3 is currently supported.
/// @note Interaction matrices that overlap with the actor matrix must be square and have the same
/// row and col offsets.
/// @note When the interaction matrix is sparse and with an overlapping (with the actor matrix)
/// "component", the overlapping "component" must have a block structure (block size kBlockSize x
/// kBlockSize).
template <int kBlockSize, bool kSkipMissingSparsityEntries = false, typename Scalar>
BlockSparseMatrix<std::remove_const_t<Scalar>, kBlockSize, int, int> ToBlockSparseMatrix(
    ActorPseudoMatrix<Scalar> const& in);

//
// ToMatrix functions
//

/// @brief Convert any matrix type to a column-major dense matrix.
///
/// @warning Not optimized.
template <typename SrcMatrixT>
auto ToMatrix(SrcMatrixT const& src) {
  using Scalar = typename SrcMatrixT::NonConstScalar;
  Matrix<Scalar> dst(src.Rows(), src.Cols());
  for (int r = 0; r < src.Rows(); ++r) {
    for (int c = 0; c < src.Cols(); ++c) {
      dst(r, c) = src(r, c);
    }
  }
  return dst;
}

/// @brief Convert from BlockSparseMatrix to column-major dense matrix.
template <
    typename Scalar,
    int kBlockSize,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
auto ToMatrix(BlockSparseMatrix<Scalar, kBlockSize, CRIdx, Ptr, Storage> const& bSpMat) {
  using NonConstScalar = std::remove_const_t<Scalar>;
  using NonConstIdx = std::remove_const_t<CRIdx>;
  //--- Note that dMat is column-oriented by default
  auto dMat = Matrix<NonConstScalar>::Zero(bSpMat.Rows(), bSpMat.Cols());
  for (NonConstIdx ib = 0; ib < bSpMat.BlockRows(); ++ib) {
    auto bColIdx = bSpMat.Indices(ib);
    auto bValues = bSpMat.Values(ib);
    for (size_t k = 0; k < bColIdx.size(); ++k) {
      dMat.Block(ib * kBlockSize, bColIdx[k] * kBlockSize, kBlockSize, kBlockSize) = bValues[k];
    }
  }
  return dMat;
}

/// @brief Convert from SparseMatrix to column-major dense matrix.
template <
    typename Scalar,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
auto ToMatrix(SparseMatrix<Scalar, CRIdx, Ptr, Storage> const& spmat) {
  using NonConstScalar = std::remove_const_t<Scalar>;
  using NonConstIdx = std::remove_const_t<CRIdx>;
  //--- Note that dMat is column-oriented by default
  auto dmat = Matrix<NonConstScalar>::Zero(spmat.Rows(), spmat.Cols());
  for (NonConstIdx ir = 0; ir < spmat.Rows(); ++ir) {
    auto colIdx = spmat.Indices(ir);
    auto values = spmat.Values(ir);
    for (size_t k = 0; k < colIdx.size(); ++k) {
      dmat(ir, colIdx[k]) = values[k];
    }
  }
  return dmat;
}

/// @brief Convert from AnyMatrixView to column-major dense matrix.
template <typename Scalar>
auto ToMatrix(AnyMatrixView<Scalar> const& anyMat) {
  return std::visit([](auto const& mat) { return ToMatrix(mat); }, anyMat);
}

/// @brief Convert from ActorPseudoMatrix to column-major dense matrix.
///
/// @note Interaction matrices that overlap with the actor matrix must be square and have the same
/// row and col offset. Other layouts are not supported.
/// @warning Not optimized. TODO[T175051452]: Optimize implementation.
template <typename Scalar>
auto ToMatrix(ActorPseudoMatrix<Scalar> const& in) {
  auto dense = ToMatrix(in.actorMatrix);
  MOCHI_ASSERT_VERBOSE(dense.Rows() == dense.Cols(), "Expected square actor matrix.");
  for (auto const& [rOffset, cOffset, anyMatrix, _] : in.interactionMatrices) {
    int const rBegin = Max(in.offset, rOffset);
    int const cBegin = Max(in.offset, cOffset);
    int const rEnd = Min(in.offset + dense.Rows(), rOffset + GetNumRows(anyMatrix));
    int const cEnd = Min(in.offset + dense.Cols(), cOffset + GetNumCols(anyMatrix));
    int const rLen = rEnd - rBegin;
    int const cLen = cEnd - cBegin;
    if (rLen > 0 && cLen > 0) {
      MOCHI_ASSERT(
          rOffset == cOffset && GetNumRows(anyMatrix) == GetNumCols(anyMatrix),
          "Unsupported interaction matrix layout.");
      dense.Block(rBegin - in.offset, cBegin - in.offset, rLen, cLen) +=
          krylov::GetBlockDiagonal<krylov::kDynamic, krylov::Direction::ColMajor>(
              anyMatrix, rBegin - rOffset, rLen);
    }
  }
  return dense;
}

/// @brief Convert from IslandOperators to column-major dense matrix.
///
/// @warning Not optimized.
template <typename Scalar>
auto ToMatrix(IslandOperators<Scalar> const& islOp) {
  auto fullMat = islOp.CondenseFullMatrix();
  return std::visit([](auto const& A) { return ToMatrix(A); }, fullMat);
}

/// @brief Convert from LowRankAugmentedMatrix to column-major dense matrix.
template <typename T>
auto ToMatrix(LowRankAugmentedMatrix<T> const& A) {
  return A.GetAugmentedMatrix();
}

//
// ToSparseMatrix functions
//

/// @brief Convert from BlockSparseMatrix to SparseMatrix.
template <
    typename Scalar,
    int kBlockSize,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
auto ToSparseMatrix(BlockSparseMatrix<Scalar, kBlockSize, CRIdx, Ptr, Storage> const& mat) {
  MOCHI_PROFILE_SCOPE();
  using NonConstScalar = std::remove_const_t<Scalar>;
  using NonConstCRIdx = std::remove_const_t<CRIdx>;
  using NonConstPtr = std::remove_const_t<Ptr>;

  DynamicArray<NonConstPtr> pointers;
  DynamicArray<NonConstCRIdx> indices;
  DynamicArray<NonConstScalar> values;

  // Pointers (row offsets)
  auto blockPointers = mat.Pointers();
  NonConstPtr count = 0;
  pointers.reserve(mat.BlockRows() * kBlockSize + 1);
  pointers.push_back(count);
  for (int br = 0; br < mat.BlockRows(); ++br) {
    auto numBlocksInRow = blockPointers[br + 1] - blockPointers[br];
    auto numValuesInRow = numBlocksInRow * kBlockSize;
    for (int i = 0; i < kBlockSize; ++i) {
      count += numValuesInRow;
      pointers.push_back(count);
    }
  }
  MOCHI_ASSERT(count == mat.NumNonZeroBlocks() * kBlockSize * kBlockSize);
  // Indices
  indices.reserve(count);
  for (int br = 0; br < mat.BlockRows(); ++br) {
    auto blockColsInBlockRow = mat.Indices(br);
    for (int i = 0; i < kBlockSize; ++i) {
      for (auto bc : blockColsInBlockRow) {
        for (int j = 0; j < kBlockSize; ++j) {
          indices.push_back(bc * kBlockSize + j);
        }
      }
    }
  }
  // Values
  values.assign(mat.Values().begin(), mat.Values().end());
  //
  return SparseMatrix<NonConstScalar, NonConstCRIdx, NonConstPtr>{
      mat.BlockCols() * kBlockSize, std::move(pointers), std::move(indices), std::move(values)};
}

/// @brief  Convert from (dense) Matrix to SparseMatrix.
/// @param[in] mat Input dense matrix.
/// @param[in] pruneZeros Boolean flag to indicate whether to prune the zeros or not.
/// Default value is false, i.e. no pruning.
template <
    typename Scalar,
    int kRowsAtCompileTime,
    int kColsAtCompileTime,
    krylov::Direction kMajorDirection,
    krylov::Ownership kOwnership,
    int kLeadingDim>
auto ToSparseMatrix(
    Matrix<
        Scalar,
        kRowsAtCompileTime,
        kColsAtCompileTime,
        kMajorDirection,
        kOwnership,
        kLeadingDim> const& mat,
    bool pruneZeros = false) {
  static_assert(!krylov::IsCuda(kOwnership), "Utility not supported for CUDA matrices");
  MOCHI_PROFILE_SCOPE();
  using NonConstScalar = std::remove_const_t<Scalar>;
  int const numRows = mat.Rows();
  int const numCols = mat.Cols();
  auto const numIndices = numRows * numCols;

  DynamicArray<int> pointers;
  DynamicArray<int> indices;
  DynamicArray<NonConstScalar> values;

  if (numIndices == 0) {
    pointers.resize(numRows + 1, 0);
    return SparseMatrix<NonConstScalar>(
        numCols, std::move(pointers), std::move(indices), std::move(values));
  }

  // Pointers
  pointers.reserve(numRows + 1);
  pointers.push_back(0);
  if (pruneZeros) {
    if constexpr (kMajorDirection == krylov::Direction::RowMajor) {
      int count = 0;
      for (int r = 0; r < numRows; ++r) {
        auto matRow = mat.Row(r);
        for (int c = 0; c < numCols; ++c) {
          count += int((matRow(0, c) != Scalar(0)));
        }
        pointers.push_back(count);
      }
    } else {
      static_assert(kMajorDirection == krylov::Direction::ColMajor);
      pointers.resize(numRows + 1, 0);
      for (int c = 0; c < numCols; ++c) {
        auto matCol = mat.Col(c);
        for (int r = 0; r < numRows; ++r) {
          if (matCol(r)) {
            pointers[r + 1] += 1;
          }
        }
      }
      for (int r = 0; r < numRows; ++r) {
        pointers[r + 1] += pointers[r];
      }
    }
  } else {
    //-- Case without pruning
    int count = 0;
    for (int r = 0; r < numRows; ++r) {
      count += numCols;
      pointers.push_back(count);
    }
  }
  indices.reserve(pointers[numRows]);
  values.reserve(pointers[numRows]);
  if (pruneZeros) {
    if constexpr (kMajorDirection == krylov::Direction::RowMajor) {
      for (int r = 0; r < numRows; ++r) {
        auto matRow = mat.Row(r);
        for (int c = 0; c < numCols; ++c) {
          if (matRow(0, c)) {
            indices.push_back(c);
            values.push_back(matRow(0, c));
          }
        }
      }
    } else {
      static_assert(kMajorDirection == krylov::Direction::ColMajor);
      std::vector<int> currentPos(pointers.begin(), pointers.begin() + numRows);
      indices.resize(pointers[numRows], 0);
      values.resize(pointers[numRows], 0);
      for (int c = 0; c < numCols; ++c) {
        for (int r = 0; r < numRows; ++r) {
          if (mat(r, c)) {
            indices[currentPos[r]] = c;
            values[currentPos[r]] = mat(r, c);
            currentPos[r] += 1;
          }
        }
      }
    }
  } else {
    //--- Conversion without pruning
    // Indices
    indices.resize_noinit(numRows * numCols);
    auto firstRow = Span(indices.data(), numCols);
    std::iota(firstRow.begin(), firstRow.end(), 0);
    for (int r = 1; r < numRows; ++r) {
      std::copy(firstRow.begin(), firstRow.end(), indices.begin() + r * numCols);
    }
    // Values
    values.resize(numIndices);
    MatrixView<
        NonConstScalar,
        krylov::kDynamic,
        krylov::kDynamic,
        krylov::Direction::RowMajor,
        krylov::kDynamic>
        rowBasedValues(values.data(), numRows, numCols, numCols);
    //--- Use accessor-based assignment
    rowBasedValues = mat;
  }
  return SparseMatrix<NonConstScalar>{
      numCols, std::move(pointers), std::move(indices), std::move(values)};
}

/// @brief Pass-through. ToSparseMatrix can be called with an input that is already a SparseMatrix.
/// In that case, it returns a const view. This makes it easier to use ToSparseMatrix in generic
/// code.
template <
    typename Scalar,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
auto ToSparseMatrix(SparseMatrix<Scalar, CRIdx, Ptr, Storage> const& spmat) {
  return AsConstView(spmat);
}

//
// AsBlockSparseMatrixView functions
//

/// @brief Create a BlockSparseMatrixView<Scalar const, 1> from a SparseMatrix.
///
/// @param[in] mat Input sparse matrix.
/// @return Const BlockSparseMatrixView with block size 1 pointing to the same data.
template <
    typename Scalar,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
auto AsBlockSparseMatrixConstView(SparseMatrix<Scalar, CRIdx, Ptr, Storage> const& mat) {
  return BlockSparseMatrixView<Scalar const, 1, CRIdx const, Ptr const>{
      mat.Cols(), mat.Pointers(), mat.Indices(), mat.Values()};
}

} // namespace mochi

#include "matrix_conversions_inl.h"
