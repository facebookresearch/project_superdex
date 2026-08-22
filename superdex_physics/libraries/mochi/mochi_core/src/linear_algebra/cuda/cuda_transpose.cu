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

#include <mochi_core/linear_algebra/matrix.h>

#include <mochi_core/linear_algebra/cuda/cuda_matrix.h>
#include <mochi_core/linear_algebra/strided_matrix.h>

namespace mochi::details {
/** \brief Return a StridedMatrixView in Column Major format spanning a given CudaMatrix data.
 *
 * The returned matrix may be the transposed of the input if the input was Row Major.
 * It is up to the caller to be aware of this fact.
 */
template <
    typename Scalar,
    int kRowsAtCompileTime,
    int kColsAtCompileTime,
    krylov::Direction kMajorDirection,
    krylov::Ownership kOwnership,
    int kLeadingDim>
static auto ToColMajStridedMatrix(
    Matrix<
        Scalar,
        kRowsAtCompileTime,
        kColsAtCompileTime,
        kMajorDirection,
        kOwnership,
        kLeadingDim>& A) {
  if constexpr (kMajorDirection == krylov::Direction::ColMajor) {
    return StridedMatrixView<
        Scalar,
        kRowsAtCompileTime,
        kColsAtCompileTime,
        1,
        Direction::ColMajor,
        kLeadingDim>(A.Data(), A.Rows(), A.Cols(), A.LeadDim());
  } else {
    // Return a transposed view of A.
    return StridedMatrixView<
        Scalar,
        kColsAtCompileTime,
        kRowsAtCompileTime,
        1,
        Direction::ColMajor,
        kLeadingDim>(A.Data(), A.Cols(), A.Rows(), A.LeadDim());
  }
}

/** \brief Return a StridedMatrixView in Column Major format spanning a given CudaMatrix data.
 *
 * The returned matrix may be the transposed of the input if the input was Row Major.
 * It is up to the caller to be aware of this fact.
 */
template <
    typename Scalar,
    int kRowsAtCompileTime,
    int kColsAtCompileTime,
    krylov::Direction kMajorDirection,
    krylov::Ownership kOwnership,
    int kLeadingDim>
auto ToConstColMajStridedMatrix(
    Matrix<
        Scalar,
        kRowsAtCompileTime,
        kColsAtCompileTime,
        kMajorDirection,
        kOwnership,
        kLeadingDim> const& A) {
  if constexpr (kMajorDirection == krylov::Direction::ColMajor) {
    return StridedMatrixView<
        Scalar const,
        kRowsAtCompileTime,
        kColsAtCompileTime,
        1,
        Direction::ColMajor,
        kLeadingDim>(A.Data(), A.Rows(), A.Cols(), A.LeadDim());
  } else {
    // Return a transposed view of A.
    return StridedMatrixView<
        Scalar const,
        kColsAtCompileTime,
        kRowsAtCompileTime,
        1,
        Direction::ColMajor,
        kLeadingDim>(A.Data(), A.Cols(), A.Rows(), A.LeadDim());
  }
}

static constexpr int TILE_DIM = 32;
static constexpr int BLOCK_ROWS = 8;

// This is optimized for Column major matrices
// Note the StridedMatrixView must be copied and not referenced as addresses do not match between
// host and device.
template <typename Scalar>
MOCHI_GPU_KERNEL void TransposeCoalesced(
    StridedMatrixView<Scalar, -1, -1> out,
    StridedMatrixView<Scalar const, -1, -1> in) {
  __shared__ Scalar tile[TILE_DIM][TILE_DIM + 1];

  // Source upper left corner of the tile. For the destination, they transpose.
  int baseRow = blockIdx.x * TILE_DIM;
  int baseCol = blockIdx.y * TILE_DIM;
  {
    int rowIn = baseRow + threadIdx.x;
    int colIn = baseCol + threadIdx.y;
    if (rowIn < in.Rows()) {
      for (int j = 0; (j < TILE_DIM) && (colIn < in.Cols());) {
        tile[threadIdx.y + j][threadIdx.x] = in(rowIn, colIn);
        j += BLOCK_ROWS;
        colIn += BLOCK_ROWS;
      }
    }
  }

  __syncthreads();

  {
    int rowOut = baseCol + threadIdx.x;
    int colOut = baseRow + threadIdx.y;
    if (rowOut < in.Cols()) {
      for (int j = 0; j < TILE_DIM && colOut < in.Rows();) {
        out(rowOut, colOut) = tile[threadIdx.x][threadIdx.y + j];
        j += BLOCK_ROWS;
        colOut += BLOCK_ROWS;
      }
    }
  }
}

template <typename Scalar>
void DoCudaTranspose(CudaMatrixView<Scalar> out, CudaMatrixView<Scalar const> in) {
  MOCHI_ASSERT_VERBOSE(
      in.Rows() == out.Cols() && in.Cols() == out.Rows(), "Transpose sizes are incompatible.");
  dim3 threadsPerBlock(details::TILE_DIM, details::BLOCK_ROWS);
  dim3 numBlocks(
      (details::TILE_DIM - 1 + in.Rows()) / details::TILE_DIM,
      (details::TILE_DIM - 1 + in.Cols()) / details::TILE_DIM);
  auto cuIn = ToConstColMajStridedMatrix(in);
  auto cuOut = ToColMajStridedMatrix(out);
  details::TransposeCoalesced<Scalar><<<numBlocks, threadsPerBlock>>>(cuOut, cuIn);
}

template void DoCudaTranspose<float>(CudaMatrixView<float> out, CudaMatrixView<float const> in);
template void DoCudaTranspose<double>(CudaMatrixView<double> out, CudaMatrixView<double const> in);
template void DoCudaTranspose<int>(CudaMatrixView<int> out, CudaMatrixView<int const> in);
} // namespace mochi::details
