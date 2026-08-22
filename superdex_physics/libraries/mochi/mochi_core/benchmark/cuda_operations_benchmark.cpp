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

#include "config.h"

#include <mochi_core/linear_algebra/cuda/cuda_lib.h>
#include <mochi_core/linear_algebra/utils/matrix_conversions.h>
#include <mochi_core/test/linear_algebra_test_helpers.h>

#include <cuda_runtime_api.h>

using namespace mochi;

namespace mochi_benchmark {

constexpr auto kColMajor = krylov::Direction::ColMajor;
constexpr auto kRowMajor = krylov::Direction::RowMajor;

/****************************************************************************************
  Cuda conversion from ColumnVector to CudaVector (2 vectors)
*/

template <typename Scalar, int kNx>
static void CudaVectorCopyConstructor(benchmark::State& state) {
  // Use the number of DOFs for an elastic brick [0, 1] x [0, 1] x [0, 1] discretized with
  // 8-noded hexahedra (forming an orthogonal grid).
  int constexpr kBlockSize = 3;
  int constexpr kNy = kNx + 1;
  int constexpr kNz = kNx + 2;
  int constexpr n = kBlockSize * kNx * kNy * kNz;
  ColumnVector<Scalar> b(n);
  b.SetRandom(123);
  ColumnVector<Scalar> x(n);
  x.SetRandom(234);
  // Warm-up to initialize the Cuda operations
  CudaVector<Scalar> bWarm(b);
  cudaDeviceSynchronize();

  for (auto _ : state) {
    cudaDeviceSynchronize();
    CudaVector<Scalar> bCuda(b);
    CudaVector<Scalar> xCuda(x);
    cudaDeviceSynchronize();
    MOCHI_NO_DISCARD_IN_LOOP(bCuda);
    MOCHI_NO_DISCARD_IN_LOOP(xCuda);
  }
}

// Size of vector: 3 * 4896
BENCHMARK_TEMPLATE(CudaVectorCopyConstructor, real, 16)
    ->Name("Cuda/VectorCopyConstructor/TwoVectors/Length14688");
// Size of vector: 3 * 9240
BENCHMARK_TEMPLATE(CudaVectorCopyConstructor, real, 20)
    ->Name("Cuda/VectorCopyConstructor/TwoVectors/Length27720");
// Size of vector: 3 * 19656
BENCHMARK_TEMPLATE(CudaVectorCopyConstructor, real, 26)
    ->Name("Cuda/VectorCopyConstructor/TwoVectors/Length58968");
// Size of vector: 3 * 29760
BENCHMARK_TEMPLATE(CudaVectorCopyConstructor, real, 30)
    ->Name("Cuda/VectorCopyConstructor/TwoVectors/Length89280");
// Size of vector: 3 * 39270
BENCHMARK_TEMPLATE(CudaVectorCopyConstructor, real, 33)
    ->Name("Cuda/VectorCopyConstructor/TwoVectors/Length117810");

/****************************************************************************************
Cuda conversion from BlockSparseMatrix to CudaBSRMatrix
*/

template <typename Scalar, int kNx, krylov::Direction kDir>
static void BlockSparseToCudaBsr(benchmark::State& state) {
  int constexpr kBlockSize = 3;
  BlockSparseMatrix<Scalar, kBlockSize> h_A;
  // Create a block sparse matrix (with block size kBlockSize)
  // where the graph of block-to-block connectivity is generated
  // with a brick [0, 1] x [0, 1] x [0, 1] discretized with
  // 8-noded hexahedra (forming an orthogonal grid).
  int constexpr kNy = kNx + 1;
  int constexpr kNz = kNx + 2;
  h_A = test::MakeBlockSparseMatrix<Scalar, kBlockSize>(kNx, kNy, kNz);
  // Warm-up to initialize the Cuda operations
  auto AWarm = ToCuda(h_A);
  cudaDeviceSynchronize();

  for (auto x : state) {
    cudaDeviceSynchronize();
    auto ACuda = ToCuda(h_A);
    cudaDeviceSynchronize();
    MOCHI_NO_DISCARD_IN_LOOP(ACuda);
  }
}

// Number of rows in matrix: 3 *  4896
BENCHMARK_TEMPLATE(BlockSparseToCudaBsr, real, 16, kColMajor)
    ->Name("Cuda/BlockSparseToCudaBsr/Nx16/ColMajor");
// Number of rows in matrix: 3 *  9240
BENCHMARK_TEMPLATE(BlockSparseToCudaBsr, real, 20, kColMajor)
    ->Name("Cuda/BlockSparseToCudaBsr/Nx20/ColMajor");
// Number of rows in matrix: 3 * 19656
BENCHMARK_TEMPLATE(BlockSparseToCudaBsr, real, 26, kColMajor)
    ->Name("Cuda/BlockSparseToCudaBsr/Nx26/ColMajor");
// Number of rows in matrix: 3 * 29760
BENCHMARK_TEMPLATE(BlockSparseToCudaBsr, real, 30, kColMajor)
    ->Name("Cuda/BlockSparseToCudaBsr/Nx30/ColMajor");
// Number of rows in matrix: 3 * 39270
BENCHMARK_TEMPLATE(BlockSparseToCudaBsr, real, 33, kColMajor)
    ->Name("Cuda/BlockSparseToCudaBsr/Nx33/ColMajor");

BENCHMARK_TEMPLATE(BlockSparseToCudaBsr, real, 16, kRowMajor)
    ->Name("Cuda/BlockSparseToCudaBsr/Nx16/RowMajor");
BENCHMARK_TEMPLATE(BlockSparseToCudaBsr, real, 20, kRowMajor)
    ->Name("Cuda/BlockSparseToCudaBsr/Nx20/RowMajor");
BENCHMARK_TEMPLATE(BlockSparseToCudaBsr, real, 26, kRowMajor)
    ->Name("Cuda/BlockSparseToCudaBsr/Nx26/RowMajor");
BENCHMARK_TEMPLATE(BlockSparseToCudaBsr, real, 30, kRowMajor)
    ->Name("Cuda/BlockSparseToCudaBsr/Nx30/RowMajor");
BENCHMARK_TEMPLATE(BlockSparseToCudaBsr, real, 33, kRowMajor)
    ->Name("Cuda/BlockSparseToCudaBsr/Nx33/RowMajor");

/// This benchmark is a baseline for the conversion from BlockSparseMatrix to CudaBSRMatrix.
/// It does not use the ToCuda() function, but instead it copy-constructs the Cuda vectors.
/// It yields the minimal work but it is not a conversion from CSR to BSR storage.
template <typename Scalar, int kNx>
static void BlockSparseToCudaBsrBaseline(benchmark::State& state) {
  int constexpr kBlockSize = 3;
  BlockSparseMatrix<Scalar, kBlockSize> h_A;
  // Create a block sparse matrix (with block size kBlockSize)
  // where the graph of block-to-block connectivity is generated
  // with a brick [0, 1] x [0, 1] x [0, 1] discretized with
  // 8-noded hexahedra (forming an orthogonal grid).
  int constexpr kNy = kNx + 1;
  int constexpr kNz = kNx + 2;
  h_A = test::MakeBlockSparseMatrix<Scalar, kBlockSize>(kNx, kNy, kNz);
  auto const h_val = AsConstView(h_A.Values());
  // Warm-up to initialize the Cuda operations
  auto AWarm = ToCuda(h_A);
  cudaDeviceSynchronize();

  for (auto _ : state) {
    cudaDeviceSynchronize();
    ////////////////
    CudaVector<int> d_bsrPtr(AsConstView(h_A.Pointers()));
    CudaVector<int> d_bsrIdx(AsConstView(h_A.Indices()));
    CudaVector<Scalar> d_bsrVal(h_val.size());
    CudaVector<Scalar> csrVal(h_val);
    // The next line does not convert correctly when kBlockSize > 1
    // But it touches all the entries once.
    cudaMemcpy(
        d_bsrVal.data(), csrVal.data(), sizeof(Scalar) * h_val.size(), cudaMemcpyDeviceToDevice);
    ////////////////
    cudaDeviceSynchronize();
    MOCHI_NO_DISCARD_IN_LOOP(d_bsrPtr);
    MOCHI_NO_DISCARD_IN_LOOP(d_bsrIdx);
    MOCHI_NO_DISCARD_IN_LOOP(d_bsrVal);
    MOCHI_NO_DISCARD_IN_LOOP(csrVal);
  }
}

// Number of rows in matrix: 3 *  4896
BENCHMARK_TEMPLATE(BlockSparseToCudaBsrBaseline, real, 16)
    ->Name("Cuda/BlockSparseToCudaBsr/CopyBaseline/Nx16");
// Number of rows in matrix: 3 *  9240
BENCHMARK_TEMPLATE(BlockSparseToCudaBsrBaseline, real, 20)
    ->Name("Cuda/BlockSparseToCudaBsr/CopyBaseline/Nx20");
// Number of rows in matrix: 3 * 19656
BENCHMARK_TEMPLATE(BlockSparseToCudaBsrBaseline, real, 26)
    ->Name("Cuda/BlockSparseToCudaBsr/CopyBaseline/Nx26");
// Number of rows in matrix: 3 * 29760
BENCHMARK_TEMPLATE(BlockSparseToCudaBsrBaseline, real, 30)
    ->Name("Cuda/BlockSparseToCudaBsr/CopyBaseline/Nx30");
// Number of rows in matrix: 3 * 39270
BENCHMARK_TEMPLATE(BlockSparseToCudaBsrBaseline, real, 33)
    ->Name("Cuda/BlockSparseToCudaBsr/CopyBaseline/Nx33");

} // namespace mochi_benchmark

#endif
