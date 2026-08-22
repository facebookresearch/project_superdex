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

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <mochi_core/linear_algebra/base_enums.h>
#include <mochi_core/linear_algebra/cuda/cuda_api.h>
#include <mochi_core/linear_algebra/cuda/cuda_lib.h>

namespace mochi::details {

/// \brief Extract diagonal blocks of size (blockSize x blockSize)
/// from a sparse matrix stored in CSR or BSR format.
///
/// \tparam PtrIdx Typename for pointer in the row or block-row entries
/// \tparam Index Typename for column indices
/// \tparam T Typename for scalar values
/// \param nBlocks Number of blocks of size (blockSize x blockSize) to store
/// \param blockSize Block size of the output diagonal blocks
/// \param bsrBlockSize Block size for the input matrix
///        When `bsrBlockSize` is equal to 1, the input matrix
///        is expected to be stored as a CSR matrix.
/// \param bsrRowPtr Pointer array for the row non-zero entries (in CSR storage)
/// and for the blockrow non-zero entries (in BSR storage)
/// \param bsrColIdxPointer array for the row non-zero entries (in CSR storage)
///// and for the blockrow non-zero entries (in BSR storage)
/// \param bsrValues Pointer array for the row non-zero entries (in CSR storage)
///// and for the blockrow non-zero entries (in BSR storage)
/// \param diagValues Array of diagonal blocks stored consecutively for
/// strided batched operations
///
/// \note The input matrix should have nBlocks * blockSize rows.
template <typename T, typename Index, typename PtrIdx, mochi::krylov::Direction kBlockStorage>
MOCHI_GPU_KERNEL void CudaGrabDiagonalBlocks(
    size_t nBlocks,
    int blockSize,
    int bsrBlockSize,
    PtrIdx const* bsrRowPtr,
    Index const* bsrColIdx,
    T const* bsrValues,
    T* diagValues) {
  if ((bsrBlockSize == 1) && (blockSize == 1)) {
    //--- Treat the case of diagonal preconditioner
    for (size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < nBlocks;
         i += gridDim.x * blockDim.x) {
      for (PtrIdx k = bsrRowPtr[i]; k < bsrRowPtr[i + 1]; ++k) {
        auto j = bsrColIdx[k];
        if (j == i) {
          diagValues[i] = bsrValues[k];
          break;
        }
      } // for (PtrIdx k = bsrRowPtr[i]; k < bsrRowPtr[i+1]; ++k)
    }
  } else {
    //--- Note that the input matrix has nBlockRows * blockSize rows.
    //--- But the matrix could be stored in BSR format with bsrBlockSize blocks
    for (size_t ib = threadIdx.x + blockIdx.x * blockDim.x; ib < nBlocks;
         ib += gridDim.x * blockDim.x) {
      auto const start = ib * blockSize;
      auto* outputBlock = diagValues + ib * blockSize * blockSize;
      for (int r = 0; r < blockSize; ++r) {
        auto myRow = start + r;
        auto const bsrRowBlock = myRow / bsrBlockSize;
        auto const bsrLocalRow = myRow - bsrRowBlock * bsrBlockSize;
        for (auto k = bsrRowPtr[bsrRowBlock]; k < bsrRowPtr[bsrRowBlock + 1]; ++k) {
          // Get the current block in the Cuda BSR storage
          // When bsrBlockSize is equal to 1, Cuda BSR storage and CSR storage match.
          auto const* thisBlockValues = bsrValues + k * bsrBlockSize * bsrBlockSize;
          for (int j = 0; j < bsrBlockSize; ++j) {
            auto const jcol = bsrColIdx[k] * bsrBlockSize + j;
            if ((jcol >= start) && (jcol < start + blockSize)) {
              if constexpr (kBlockStorage == krylov::Direction::ColMajor) {
                // Get value in a column-major, store it in column-major.
                outputBlock[r + (jcol - start) * blockSize] =
                    thisBlockValues[bsrLocalRow + j * bsrBlockSize];
              } else {
                // Get value in a row-major, store it in column-major.
                outputBlock[r + (jcol - start) * blockSize] =
                    thisBlockValues[bsrLocalRow * bsrBlockSize + j];
              }
            }
          }
        }
      } // for (int r = 0; r < blockSize; ++r)
    }
  }
}

template <typename Scalar, typename ColIdx, typename RowPtr, mochi::krylov::Direction kBlockStorage>
void ExtractInverseDiagBlocks(
    size_t nBlockRows,
    int blockSize,
    int bsrBlockSize,
    RowPtr const* bsrRowPtr,
    ColIdx const* bsrColIdx,
    Scalar const* bsrValues,
    Scalar* diagValues) {
  int numThreads = kNThreads;
  int n = int((nBlockRows + numThreads - 1) / numThreads);
  //
  size_t lenBlock = size_t(blockSize) * size_t(blockSize);
  size_t lenDiagBlocks = size_t(nBlockRows) * lenBlock;
  Scalar* copyDiag;
  cudaMalloc((void**)&copyDiag, sizeof(Scalar) * lenDiagBlocks);
  cudaMemset(copyDiag, 0, sizeof(Scalar) * lenDiagBlocks);
  //
  // --- Creating the array of pointers needed as input to the batched getrf
  //
  auto** h_inout_pointers = (Scalar**)malloc(nBlockRows * sizeof(Scalar*));
  for (size_t ii = 0; ii < nBlockRows; ++ii)
    h_inout_pointers[ii] = copyDiag + ii * lenBlock;

  Scalar** d_inout_pointers;
  cudaMalloc((void**)&d_inout_pointers, nBlockRows * sizeof(Scalar*));
  cudaMemcpy(
      d_inout_pointers, h_inout_pointers, nBlockRows * sizeof(Scalar*), cudaMemcpyHostToDevice);

  CudaGrabDiagonalBlocks<Scalar, ColIdx, RowPtr, kBlockStorage><<<n, numThreads>>>(
      nBlockRows, blockSize, bsrBlockSize, bsrRowPtr, bsrColIdx, bsrValues, copyDiag);

  //--- Invert each block using  batched operation
  for (size_t ii = 0; ii < nBlockRows; ++ii) {
    h_inout_pointers[ii] = diagValues + ii * lenBlock;
  }

  Scalar** rd_inout_pointers;
  cudaMalloc((void**)&rd_inout_pointers, nBlockRows * sizeof(Scalar*));
  cudaMemcpy(
      rd_inout_pointers, h_inout_pointers, nBlockRows * sizeof(Scalar*), cudaMemcpyHostToDevice);
  free(h_inout_pointers);

  int* d_PivotArray;
  cudaMalloc((void**)&d_PivotArray, blockSize * nBlockRows * sizeof(int));

  int* d_InfoArray;
  cudaMalloc((void**)&d_InfoArray, nBlockRows * sizeof(int));

  auto handle = reinterpret_cast<cublasHandle_t>(mochi::details::GetCuBLASHandle());
  if constexpr (std::is_same_v<Scalar const, double const>) {
    cublasDgetrfBatched(
        handle,
        blockSize,
        d_inout_pointers,
        blockSize,
        d_PivotArray,
        d_InfoArray,
        static_cast<int>(nBlockRows));
    cublasDgetriBatched(
        handle,
        blockSize,
        d_inout_pointers,
        blockSize,
        d_PivotArray,
        rd_inout_pointers,
        blockSize,
        d_InfoArray,
        static_cast<int>(nBlockRows));
  } else if constexpr (std::is_same_v<Scalar const, float const>) {
    cublasSgetrfBatched(
        handle,
        blockSize,
        d_inout_pointers,
        blockSize,
        d_PivotArray,
        d_InfoArray,
        static_cast<int>(nBlockRows));
    cublasSgetriBatched(
        handle,
        blockSize,
        d_inout_pointers,
        blockSize,
        d_PivotArray,
        rd_inout_pointers,
        blockSize,
        d_InfoArray,
        static_cast<int>(nBlockRows));
  } else {
    static_assert(
        std::is_same_v<Scalar const, float const> || std::is_same_v<Scalar const, double const>,
        "Scalar type not supported");
  }
  //--- Free the memory
  cudaFree((void*)d_inout_pointers);
  cudaFree((void*)rd_inout_pointers);
  cudaFree((void*)d_PivotArray);
  cudaFree((void*)d_InfoArray);
  cudaFree((void*)copyDiag);
}

#define MOCHI_INSTANTIATE_EXTRACT_INVERSE_DIAG_BLOCKS(  \
    SCALAR_TYPE, COL_IDX_TYPE, ROW_PTR_TYPE, DIRECTION) \
  template void ExtractInverseDiagBlocks<               \
      SCALAR_TYPE,                                      \
      COL_IDX_TYPE,                                     \
      ROW_PTR_TYPE,                                     \
      mochi::krylov::Direction::DIRECTION>(             \
      size_t nBlockRows,                                \
      int blockSize,                                    \
      int bsrBlockSize,                                 \
      ROW_PTR_TYPE const* bsrRowPtr,                    \
      COL_IDX_TYPE const* bsrColIdx,                    \
      SCALAR_TYPE const* bsrValues,                     \
      SCALAR_TYPE* diagValues);

MOCHI_INSTANTIATE_EXTRACT_INVERSE_DIAG_BLOCKS(float, int, int, ColMajor)
MOCHI_INSTANTIATE_EXTRACT_INVERSE_DIAG_BLOCKS(float, int64_t, int64_t, ColMajor)
MOCHI_INSTANTIATE_EXTRACT_INVERSE_DIAG_BLOCKS(float, int, int, RowMajor)
MOCHI_INSTANTIATE_EXTRACT_INVERSE_DIAG_BLOCKS(float, int64_t, int64_t, RowMajor)

MOCHI_INSTANTIATE_EXTRACT_INVERSE_DIAG_BLOCKS(double, int, int, ColMajor)
MOCHI_INSTANTIATE_EXTRACT_INVERSE_DIAG_BLOCKS(double, int64_t, int64_t, ColMajor)
MOCHI_INSTANTIATE_EXTRACT_INVERSE_DIAG_BLOCKS(double, int, int, RowMajor)
MOCHI_INSTANTIATE_EXTRACT_INVERSE_DIAG_BLOCKS(double, int64_t, int64_t, RowMajor)

#undef MOCHI_INSTANTIATE_EXTRACT_INVERSE_DIAG_BLOCKS

/// \brief Cuda kernel to evaluate C = A .* B (entry-wise vector Hadamard product)
///
/// \tparam Scalar type
/// \param[in] A_val Pointer to the input vector A
/// \param[in] B_val Pointer to the input vector B
/// \param[out] C_val Pointer to the output vector C
/// \param[in] dim Length of vector
///
/// \note With the `restrict` flag, the call `EntryProduct(A, x, x)`
/// leads to undefined behavior.
///
template <typename Scalar>
__global__ static void EntryProduct(
    Scalar const* MOCHI_RESTRICT A_val,
    Scalar const* MOCHI_RESTRICT B_val,
    Scalar* MOCHI_RESTRICT C_val,
    int dim) {
  auto const gid = blockIdx.x * blockDim.x + threadIdx.x;
  auto const stride = blockDim.x * gridDim.x;
  for (auto i = gid; i < dim; i += stride) {
    C_val[i] = A_val[i] * B_val[i];
  }
}

template <typename Scalar>
void ApplyDiagonal(
    int nRows,
    Scalar const* D,
    Scalar const* x,
    int ldx,
    int colx,
    Scalar* y,
    int ldy,
    int coly) {
  static_assert(std::is_same_v<Scalar, double> || std::is_same_v<Scalar, float>);
  MOCHI_ASSERT_VERBOSE(colx == coly, "Incompatible number of columns for input and output");
  auto blasHandle = reinterpret_cast<cublasHandle_t>(mochi::details::GetCuBLASHandle());
  if (colx == 1) {
    constexpr int blockSize = kNThreads;
    int nBlocks = (nRows + blockSize - 1) / blockSize;
    cudaStream_t myStream = nullptr;
    cublasGetStream(blasHandle, &myStream);
    EntryProduct<<<nBlocks, blockSize, 0, myStream>>>(D, x, y, nRows);
  } else {
    constexpr auto one = int(1);
    if constexpr (std::is_same_v<Scalar, double>) {
      MOCHI_CUBLAS_CHECK(
          cublasDdgmm(blasHandle, CUBLAS_SIDE_LEFT, nRows, colx, x, ldx, D, one, y, ldy));
    } else {
      MOCHI_CUBLAS_CHECK(
          cublasSdgmm(blasHandle, CUBLAS_SIDE_LEFT, nRows, colx, x, ldx, D, one, y, ldy));
    }
  }
}

template void ApplyDiagonal<double>(
    int nRows,
    double const* D,
    double const* x,
    int ldx,
    int colx,
    double* y,
    int ldy,
    int coly);

template void ApplyDiagonal<float>(
    int nRows,
    float const* D,
    float const* x,
    int ldx,
    int colx,
    float* y,
    int ldy,
    int coly);

/// \brief Cuda kernel to evaluate y = A * x (where A represents a block-diagonal matrix)
///
/// \note With the `restrict` flag, the call `BlockDiagonalProduct`
/// may lead to undefined behavior (when x == y).
///
template <typename Scalar>
__global__ static void BlockDiagonalProduct(
    int nRowBlocks,
    int aRows,
    int aCols,
    Scalar const* MOCHI_RESTRICT A,
    int lda,
    int strideA,
    Scalar const* MOCHI_RESTRICT x,
    int strideX,
    Scalar* MOCHI_RESTRICT y,
    int strideY) {
  auto const gid = blockIdx.x * blockDim.x + threadIdx.x;
  auto const stride = blockDim.x * gridDim.x;
  for (auto i = gid; i < nRowBlocks; i += stride) {
    auto const* mat = A + i * strideA;
    for (int ia = 0; ia < aRows; ++ia) {
      y[i * strideY + ia] = 0;
      for (int ja = 0; ja < aCols; ++ja) {
        y[i * strideY + ia] += mat[ia + ja * lda] * x[i * strideX + ja];
      }
    }
  }
}

template <typename Scalar>
void ApplyBatchedDiagonal(
    int nRowBlocks,
    int aRows,
    int aCols,
    Scalar const* A,
    int lda,
    int strideA,
    Scalar const* x,
    int ldx,
    int colx,
    int strideX,
    Scalar* y,
    int ldy,
    int coly,
    int strideY) {
  static_assert(std::is_same_v<Scalar, double> || std::is_same_v<Scalar, float>);
  MOCHI_ASSERT_VERBOSE(colx == coly, "Incompatible number of columns for input and output");
  auto blasHandle = reinterpret_cast<cublasHandle_t>(mochi::details::GetCuBLASHandle());
  auto const one = Scalar(1), zero = Scalar(0);
  MOCHI_CUBLAS_CHECK(cublasSetPointerMode(blasHandle, CUBLAS_POINTER_MODE_HOST));
  if (colx == 1) {
    constexpr int blockSize = kNThreads;
    int nBlocks = (nRowBlocks + blockSize - 1) / blockSize;
    cudaStream_t myStream = nullptr;
    cublasGetStream(blasHandle, &myStream);
    BlockDiagonalProduct<<<nBlocks, blockSize, 0, myStream>>>(
        nRowBlocks, aRows, aCols, A, lda, strideA, x, strideX, y, strideY);
  } else {
    if constexpr (std::is_same_v<Scalar, double>) {
      cublasDgemmStridedBatched(
          blasHandle,
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          aRows,
          colx,
          aCols,
          &one,
          A,
          lda,
          strideA,
          x,
          ldx,
          strideX,
          &zero,
          y,
          ldy,
          strideY,
          nRowBlocks);
    } else {
      cublasSgemmStridedBatched(
          blasHandle,
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          aRows,
          colx,
          aCols,
          &one,
          A,
          lda,
          strideA,
          x,
          ldx,
          strideX,
          &zero,
          y,
          ldy,
          strideY,
          nRowBlocks);
    }
  }
}

template void ApplyBatchedDiagonal<double>(
    int nRowBlocks,
    int aRows,
    int aCols,
    double const* A,
    int lda,
    int strideA,
    double const* x,
    int ldx,
    int colx,
    int strideX,
    double* y,
    int ldy,
    int coly,
    int strideY);

template void ApplyBatchedDiagonal<float>(
    int nRowBlocks,
    int aRows,
    int aCols,
    float const* A,
    int lda,
    int strideA,
    float const* x,
    int ldx,
    int colx,
    int strideX,
    float* y,
    int ldy,
    int coly,
    int strideY);

} // namespace mochi::details
