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

#include <mochi_core/linear_algebra/cuda/cuda_lib.h>
#include <mochi_core/linear_algebra/cuda/cuda_sparse_factorization.h>

template <typename Scalar, mochi::krylov::Direction kBlockDir>
MOCHI_GPU_KERNEL void kernelBsrToCsr(
    int blockSize,
    int n,
    int const* MOCHI_RESTRICT d_blockPtr,
    int const* MOCHI_RESTRICT d_blockColIdx,
    Scalar const* MOCHI_RESTRICT d_blockValues,
    int* MOCHI_RESTRICT d_rowPtr,
    int* MOCHI_RESTRICT d_colIdx,
    Scalar* MOCHI_RESTRICT d_values) {
  int const globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (globalIdx == 0) {
    d_rowPtr[0] = 0;
  }
  for (int ir = globalIdx; ir < n; ir += gridDim.x * blockDim.x) {
    int const iBlock = ir / blockSize;
    int const local = ir - (iBlock * blockSize);
    auto const start = d_blockPtr[iBlock];
    auto const localNumBlocks = d_blockPtr[iBlock + 1] - start;
    auto const rowStart = start * blockSize * blockSize + local * blockSize * localNumBlocks;
    d_rowPtr[ir + 1] = rowStart + blockSize * localNumBlocks;
    for (int k = 0; k < localNumBlocks; ++k) {
      auto const myBlockCol = d_blockColIdx[start + k];
      for (int j = 0; j < blockSize; ++j) {
        d_colIdx[rowStart + k * blockSize + j] = myBlockCol * blockSize + j;
      }
    }
    //
    for (int k = 0; k < localNumBlocks; ++k) {
      auto const* myBlockValues = &d_blockValues[(start + k) * blockSize * blockSize];
      for (int j = 0; j < blockSize; ++j) {
        if constexpr (kBlockDir == mochi::krylov::Direction::ColMajor) {
          d_values[rowStart + k * blockSize + j] = myBlockValues[local + j * blockSize];
        } else {
          // Each block is stored in a row-major fashion
          d_values[rowStart + k * blockSize + j] = myBlockValues[local * blockSize + j];
        }
      }
    }
  }
}

namespace mochi::details::cuda {

template <typename Scalar, krylov::Direction kBlockDir>
void ConvertMatrixDeviceBsrToDeviceCsr(
    int blockSize,
    int n,
    int const* d_blockPtr,
    int const* d_blockColIdx,
    Scalar const* d_blockValues,
    int* d_rowPtr,
    int* d_colIdx,
    Scalar* d_values) {
  int const nBlocks = (n + mochi::details::kNThreads - 1) / mochi::details::kNThreads;
  kernelBsrToCsr<Scalar, kBlockDir><<<nBlocks, mochi::details::kNThreads>>>(
      blockSize, n, d_blockPtr, d_blockColIdx, d_blockValues, d_rowPtr, d_colIdx, d_values);
  cudaDeviceSynchronize();
}

template void ConvertMatrixDeviceBsrToDeviceCsr<double, krylov::Direction::ColMajor>(
    int blockSize,
    int n,
    int const* d_blockPtr,
    int const* d_blockColIdx,
    double const* d_blockValues,
    int* d_rowPtr,
    int* d_colIdx,
    double* d_values);

template void ConvertMatrixDeviceBsrToDeviceCsr<double, krylov::Direction::RowMajor>(
    int blockSize,
    int n,
    int const* d_blockPtr,
    int const* d_blockColIdx,
    double const* d_blockValues,
    int* d_rowPtr,
    int* d_colIdx,
    double* d_values);

template void ConvertMatrixDeviceBsrToDeviceCsr<float, krylov::Direction::ColMajor>(
    int blockSize,
    int n,
    int const* d_blockPtr,
    int const* d_blockColIdx,
    float const* d_blockValues,
    int* d_rowPtr,
    int* d_colIdx,
    float* d_values);

template void ConvertMatrixDeviceBsrToDeviceCsr<float, krylov::Direction::RowMajor>(
    int blockSize,
    int n,
    int const* d_blockPtr,
    int const* d_blockColIdx,
    float const* d_blockValues,
    int* d_rowPtr,
    int* d_colIdx,
    float* d_values);

void ConvertGraphHostBsrToDeviceCsr(
    int blockSize,
    int n,
    int const* h_blockRowPtr,
    int const* h_blockColIdx,
    int* d_rowPtr,
    int* d_colIdx) {
  std::vector<int> h_rowPtr(n + 1);
  int const nnz = blockSize * blockSize * h_blockRowPtr[n / blockSize];
  std::vector<int> h_colIdx(nnz);
  //
  ConvertGraphHostBsrToHostCsr(
      blockSize, n, h_blockRowPtr, h_blockColIdx, h_rowPtr.data(), h_colIdx.data());
  //
  MOCHI_CUDA_CHECK(
      cudaMemcpy(d_rowPtr, h_rowPtr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
  MOCHI_CUDA_CHECK(
      cudaMemcpy(d_colIdx, h_colIdx.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
}

void ConvertGraphHostBsrToHostCsr(
    int blockSize,
    int n,
    int const* h_blockRowPtr,
    int const* h_blockColIdx,
    int* h_rowPtr,
    int* h_colIdx) {
  h_rowPtr[0] = 0;
  //
  for (int ir = 0; ir < n; ir += 1) {
    int const iBlock = ir / blockSize;
    int const local = ir - (iBlock * blockSize);
    auto const start = h_blockRowPtr[iBlock];
    auto const localNumBlocks = h_blockRowPtr[iBlock + 1] - start;
    auto const rowStart = start * blockSize * blockSize + local * blockSize * localNumBlocks;
    h_rowPtr[ir + 1] = rowStart + blockSize * localNumBlocks;
    for (int k = 0; k < localNumBlocks; ++k) {
      auto const myBlockCol = h_blockColIdx[start + k];
      for (int j = 0; j < blockSize; ++j) {
        h_colIdx[rowStart + k * blockSize + j] = myBlockCol * blockSize + j;
      }
    }
  }
}

} // namespace mochi::details::cuda

#endif
