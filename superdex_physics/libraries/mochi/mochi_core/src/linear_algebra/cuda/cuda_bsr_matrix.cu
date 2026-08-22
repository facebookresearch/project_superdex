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

#include <mochi_core/linear_algebra/base_enums.h>

namespace mochi::krylov::details {

template <typename Scalar, krylov::Direction kMajorDir>
MOCHI_GPU_KERNEL void Convert(
    int blockSize,
    int nb,
    int const* MOCHI_RESTRICT rowPtr,
    Scalar const* MOCHI_RESTRICT csrVal,
    Scalar* MOCHI_RESTRICT bsrVal) {
  for (int rb = blockIdx.x; rb < nb; rb += gridDim.x) {
    // Number of block columns in this block row
    int const numBlockCols = rowPtr[rb + 1] - rowPtr[rb];
    int const totalCols = numBlockCols * blockSize;
    // Base pointers for this block row
    auto* bsrBase = bsrVal + blockSize * blockSize * rowPtr[rb];
    auto const* csrBase = csrVal + blockSize * blockSize * rowPtr[rb];
    for (int iy = threadIdx.y; iy < blockSize; iy += blockDim.y) {
      for (int jx = threadIdx.x; jx < totalCols; jx += blockDim.x) {
        int const localBlock = jx / blockSize;
        int const localCol = jx % blockSize;
        if constexpr (kMajorDir == mochi::krylov::Direction::ColMajor) {
          bsrBase[iy + localCol * blockSize + localBlock * blockSize * blockSize] =
              csrBase[jx + iy * totalCols];
        } else {
          bsrBase[localCol + iy * blockSize + localBlock * blockSize * blockSize] =
              csrBase[jx + iy * totalCols];
        }
      }
    }
  }
}

template <typename Scalar, krylov::Direction kMajorDir>
void ConvertCsrToBsr(
    int blockSize,
    int nBlockRows,
    int maxNNZBlockPerRow,
    int* d_csrRowPtr,
    Scalar const* d_csrVal,
    Scalar* d_bsrVal) {
  auto const nBlockColsX = (unsigned int)(maxNNZBlockPerRow);
  dim3 blockDim{nBlockColsX, (unsigned int)(blockSize)};
  Convert<Scalar, kMajorDir>
      <<<nBlockRows, blockDim>>>(blockSize, nBlockRows, d_csrRowPtr, d_csrVal, d_bsrVal);
}

template void ConvertCsrToBsr<double, krylov::Direction::ColMajor>(
    int blockSize,
    int nBlockRows,
    int maxNNZBlockPerRow,
    int* d_csrRowPtr,
    double const* d_csrVal,
    double* d_bsrVal);

template void ConvertCsrToBsr<double, krylov::Direction::RowMajor>(
    int blockSize,
    int nBlockRows,
    int maxNNZBlockPerRow,
    int* d_csrRowPtr,
    double const* d_csrVal,
    double* d_bsrVal);

template void ConvertCsrToBsr<float, krylov::Direction::ColMajor>(
    int blockSize,
    int nBlockRows,
    int maxNNZBlockPerRow,
    int* d_csrRowPtr,
    float const* d_csrVal,
    float* d_bsrVal);

template void ConvertCsrToBsr<float, krylov::Direction::RowMajor>(
    int blockSize,
    int nBlockRows,
    int maxNNZBlockPerRow,
    int* d_csrRowPtr,
    float const* d_csrVal,
    float* d_bsrVal);

} // namespace mochi::krylov::details
