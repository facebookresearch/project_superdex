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

#include <mochi_core/linear_algebra/cuda/cuda_lib.h>

namespace mochi::details {

/// \brief Simple kernel for linear combinations when all terms have a stride
/// of 1.
template <typename Scalar>
MOCHI_GPU_KERNEL void
CudaScaledAssign(int N, Scalar* dest, int nPos, PackedScaleData<Scalar const> posSrc) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    Scalar t = 0;
    for (int p = 0; p < nPos; ++p)
      t += posSrc.v[p].scale * posSrc.v[p].v[i];
    dest[i] = t;
  }
}

template <typename Scalar>
void DoCudaScaledAssign(int N, Scalar* dest, int nPos, PackedScaleData<Scalar const>& posSrc) {
  int blockSize = kNThreads;
  int nBlocks = (N + blockSize - 1) / blockSize;
  CudaScaledAssign<<<nBlocks, blockSize>>>(N, dest, nPos, posSrc);
#if MOCHI_DEBUG
  [[maybe_unused]] auto error = cudaPeekAtLastError();
#endif
}

template void
DoCudaScaledAssign<double>(int N, double* dest, int nPos, PackedScaleData<double const>& posSrc);

template void
DoCudaScaledAssign<float>(int N, float* dest, int nPos, PackedScaleData<float const>& posSrc);

/// \details Destination rowstride must be 1.
template <typename Scalar>
MOCHI_GPU_KERNEL void CudaScaled2DAssign(
    int nRows,
    int nCols,
    MatDataDest<Scalar> dest,
    int numTerms,
    PackedScaleData<Scalar const> terms) {
  auto r = blockIdx.x * kTileDim + threadIdx.x; // Row for the thread
  auto c = blockIdx.y * kTileDim + threadIdx.y; // First column for the thread.

  if (r < nRows)
    for (int j = 0; j < kTileDim; j += kBlockRows) {
      if (c + j < nCols) {
        Scalar t = 0;
        for (int i = 0; i < numTerms; ++i) {
          t += terms.v[i].scale *
              terms.v[i].v[r * terms.v[i].rowStride + (c + j) * terms.v[i].colStride];
        }
        dest.v[r + (j + c) * dest.colStride] = t;
      }
    }
}

template <typename Scalar>
void DoCudaScaled2DAssign(
    int nRows,
    int nCols,
    MatDataDest<Scalar> dest,
    int numTerms,
    PackedScaleData<Scalar const>& terms) {
  static_assert(kTileDim % kBlockRows == 0);
  dim3 nBlocks((nRows + kTileDim - 1) / kTileDim, (nCols + kTileDim - 1) / kTileDim, 1);
  dim3 threadsPerBlock(kTileDim, kBlockRows);
  CudaScaled2DAssign<<<nBlocks, threadsPerBlock>>>(nRows, nCols, dest, numTerms, terms);
#if MOCHI_DEBUG
  [[maybe_unused]] auto error = cudaPeekAtLastError();
#endif
}

template void DoCudaScaled2DAssign<double>(
    int nRows,
    int nCols,
    MatDataDest<double> dest,
    int numTerms,
    PackedScaleData<double const>& terms);

template void DoCudaScaled2DAssign<float>(
    int nRows,
    int nCols,
    MatDataDest<float> dest,
    int numTerms,
    PackedScaleData<float const>& terms);

} // namespace mochi::details
