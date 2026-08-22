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

#include "config.h"

#include <mochi_core/linear_algebra/block_sparse_matrix.h>
#include <mochi_core/linear_algebra/krylov/amg/amg_prec.h>
#include <mochi_core/linear_algebra/krylov/block_jacobi_prec.h>
#include <mochi_core/linear_algebra/krylov/block_ssor_prec.h>
#include <mochi_core/linear_algebra/krylov/colored_ssor_prec.h>
#include <mochi_core/linear_algebra/krylov/incomplete_cholesky_prec.h>
#include <mochi_core/linear_algebra/krylov/relaxed_ilu_prec.h>
#include <mochi_core/linear_algebra/krylov/ssor_prec.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/sparse_matrix.h>
#include <mochi_core/linear_algebra/utils/matrix_conversions.h>
#include <mochi_core/test/linear_algebra_test_helpers.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/task_scheduler.h>

#include <type_traits>

using namespace mochi;

namespace mochi_benchmark {

/****************************************************************************************
  Construction of (block) SSOR preconditioner
*/

template <typename Scalar, int kMatBlockSize>
static void SSORConstructionFromSparseMatrix(benchmark::State& state) {
  int nx = state.range(0), ny = state.range(1), nz = state.range(2);
  auto A = test::MakeSparseMatrixWithBlockStructure<Scalar, kMatBlockSize>(nx, ny, nz);
  for (auto x : state) {
    krylov::SSORPrec<Scalar, SparseMatrix<Scalar>> P(A);
    MOCHI_NO_DISCARD_IN_LOOP(P);
  }
}

template <typename Scalar, int kMatBlockSize>
static void SSORConstructionFromBlockSparseMatrix(benchmark::State& state) {
  int nx = state.range(0), ny = state.range(1), nz = state.range(2);
  auto A = test::MakeBlockSparseMatrix<Scalar, kMatBlockSize>(nx, ny, nz);
  for (auto x : state) {
    krylov::SSORPrec<Scalar, BlockSparseMatrix<Scalar, kMatBlockSize>> P(A);
    MOCHI_NO_DISCARD_IN_LOOP(P);
  }
}

template <typename Scalar, int kMatBlockSize>
static void ColoredSSORConstructionFromSparseMatrix(benchmark::State& state) {
  int nx = int(state.range(0)), ny = int(state.range(1)), nz = int(state.range(2));
  int const numThreads = int(state.range(3));
  MOCHI_ASSERT(numThreads > 0, "Invalid number of threads.");
  if (numThreads > TaskScheduler::GetNumSupportedLogicalProcessors()) {
    MOCHI_SKIP_BENCHMARK(state);
    return;
  }
  TaskScheduler scheduler(numThreads - 1); // Caller not included in TaskScheduler's pool
  auto A = test::MakeSparseMatrixWithBlockStructure<Scalar, kMatBlockSize>(nx, ny, nz);
  for (auto x : state) {
    krylov::ColoredSSORPrec<SparseMatrix<Scalar>> P(A);
    MOCHI_NO_DISCARD_IN_LOOP(P);
  }
}

template <typename Scalar, int kMatBlockSize>
static void ColoredSSORConstructionFromBlockSparseMatrix(benchmark::State& state) {
  int nx = int(state.range(0)), ny = int(state.range(1)), nz = int(state.range(2));
  int const numThreads = int(state.range(3));
  MOCHI_ASSERT(numThreads > 0, "Invalid number of threads.");
  if (numThreads > TaskScheduler::GetNumSupportedLogicalProcessors()) {
    MOCHI_SKIP_BENCHMARK(state);
    return;
  }
  TaskScheduler scheduler(numThreads - 1); // Caller not included in TaskScheduler's pool
  auto A = test::MakeBlockSparseMatrix<Scalar, kMatBlockSize>(nx, ny, nz);
  for (auto x : state) {
    krylov::ColoredSSORPrec<BlockSparseMatrix<Scalar, kMatBlockSize>> P(A);
    MOCHI_NO_DISCARD_IN_LOOP(P);
  }
}

template <typename Scalar, int kMatBlockSize, int kPrecBlockSize>
static void BlockSSORConstructionFromSparseMatrix(benchmark::State& state) {
  int nx = int(state.range(0)), ny = int(state.range(1)), nz = int(state.range(2));
  auto A = test::MakeSparseMatrixWithBlockStructure<Scalar, kMatBlockSize>(nx, ny, nz);
  for (auto x : state) {
    krylov::BlockSSORPrec<Scalar, kPrecBlockSize, SparseMatrix<Scalar>> P(A);
    MOCHI_NO_DISCARD_IN_LOOP(P);
  }
}

template <typename Scalar, int kMatBlockSize, int kPrecBlockSize>
static void BlockSSORConstructionFromBlockSparseMatrix(benchmark::State& state) {
  int nx = int(state.range(0)), ny = int(state.range(1)), nz = int(state.range(2));
  auto A = test::MakeBlockSparseMatrix<Scalar, kMatBlockSize>(nx, ny, nz);
  for (auto x : state) {
    krylov::BlockSSORPrec<Scalar, kPrecBlockSize, BlockSparseMatrix<Scalar, kMatBlockSize>> P(A);
    MOCHI_NO_DISCARD_IN_LOOP(P);
  }
}

// SSOR construction. Benchmark performance when the input matrix is sparse and block sparse with
// block size of 3.
BENCHMARK_TEMPLATE(SSORConstructionFromSparseMatrix, float, 3)
    ->Name("Preconditioner/SSOR/Construction/SparseMatrix")
    ->ArgNames({"nx", "ny", "nz"})
    ->Args({13, 17, 19});
BENCHMARK_TEMPLATE(SSORConstructionFromBlockSparseMatrix, float, 3)
    ->Name("Preconditioner/SSOR/Construction/BlockSparseMatrix")
    ->ArgNames({"nx", "ny", "nz"})
    ->Args({13, 17, 19});

// Colored SSOR construction. Benchmark performance when the input matrix is sparse and block sparse
// with block size of 3.
BENCHMARK_TEMPLATE(ColoredSSORConstructionFromSparseMatrix, float, 3)
    ->Name("Preconditioner/ColoredSSOR/Construction/SparseMatrix")
    ->ArgNames({"nx", "ny", "nz", "threads"})
    ->ArgsProduct({{13}, {17}, {19}, {1, 2, 4, 8}})
    ->UseRealTime();
BENCHMARK_TEMPLATE(ColoredSSORConstructionFromBlockSparseMatrix, float, 3)
    ->Name("Preconditioner/ColoredSSOR/Construction/BlockSparseMatrix")
    ->ArgNames({"nx", "ny", "nz", "threads"})
    ->ArgsProduct({{13}, {17}, {19}, {1, 2, 4, 8}})
    ->UseRealTime();

// Block SSOR construction with block size of 3. Benchmark performance when the input matrix is
// sparse and block sparse with block size of 3.
BENCHMARK_TEMPLATE(BlockSSORConstructionFromSparseMatrix, float, 3, 3)
    ->Name("Preconditioner/BlockSSOR/Construction/SparseMatrix")
    ->ArgNames({"nx", "ny", "nz"})
    ->Args({13, 17, 19});
BENCHMARK_TEMPLATE(BlockSSORConstructionFromBlockSparseMatrix, float, 3, 3)
    ->Name("Preconditioner/BlockSSOR/Construction/BlockSparseMatrix")
    ->ArgNames({"nx", "ny", "nz"})
    ->Args({13, 17, 19});

template <typename Scalar, int kMatBlockSize>
static void ColoredSSORUpdateForSparseMatrix(benchmark::State& state) {
  int nx = int(state.range(0)), ny = int(state.range(1)), nz = int(state.range(2));
  int const numThreads = int(state.range(3));
  MOCHI_ASSERT(numThreads > 0, "Invalid number of threads.");
  if (numThreads > TaskScheduler::GetNumSupportedLogicalProcessors()) {
    MOCHI_SKIP_BENCHMARK(state);
    return;
  }
  TaskScheduler scheduler(numThreads - 1); // Caller not included in TaskScheduler's pool
  auto A = test::MakeSparseMatrixWithBlockStructure<Scalar, kMatBlockSize>(nx, ny, nz);
  krylov::ColoredSSORPrec<SparseMatrix<Scalar>> P(A);
  for (auto x : state) {
    P.Update(A);
    MOCHI_NO_DISCARD_IN_LOOP(P);
  }
}

template <typename Scalar, int kMatBlockSize>
static void ColoredSSORUpdateForBlockSparseMatrix(benchmark::State& state) {
  int nx = int(state.range(0)), ny = int(state.range(1)), nz = int(state.range(2));
  int const numThreads = int(state.range(3));
  MOCHI_ASSERT(numThreads > 0, "Invalid number of threads.");
  if (numThreads > TaskScheduler::GetNumSupportedLogicalProcessors()) {
    MOCHI_SKIP_BENCHMARK(state);
    return;
  }
  TaskScheduler scheduler(numThreads - 1); // Caller not included in TaskScheduler's pool
  auto A = test::MakeBlockSparseMatrix<Scalar, kMatBlockSize>(nx, ny, nz);
  krylov::ColoredSSORPrec<BlockSparseMatrix<Scalar, kMatBlockSize>> P(A);
  for (auto x : state) {
    P.Update(A);
    MOCHI_NO_DISCARD_IN_LOOP(P);
  }
}

// Colored SSOR update. Benchmark performance when the input matrix is sparse and block sparse
// with block size of 3.
BENCHMARK_TEMPLATE(ColoredSSORUpdateForSparseMatrix, float, 3)
    ->Name("Preconditioner/ColoredSSOR/Update/SparseMatrix")
    ->ArgNames({"nx", "ny", "nz", "threads"})
    ->ArgsProduct({{13}, {17}, {19}, {1, 2, 4, 8}})
    ->UseRealTime();
BENCHMARK_TEMPLATE(ColoredSSORUpdateForBlockSparseMatrix, float, 3)
    ->Name("Preconditioner/ColoredSSOR/Update/BlockSparseMatrix")
    ->ArgNames({"nx", "ny", "nz", "threads"})
    ->ArgsProduct({{13}, {17}, {19}, {1, 2, 4, 8}})
    ->UseRealTime();

/****************************************************************************************
  Construction of (block) Jacobi preconditioner
*/

template <typename Scalar, int kMatBlockSize, int kPrecBlockSize>
static void BlockJacobiConstructionFromSparseMatrix(benchmark::State& state) {
  int nx = state.range(0), ny = state.range(1), nz = state.range(2);
  auto A = test::MakeSparseMatrixWithBlockStructure<Scalar, kMatBlockSize>(nx, ny, nz);
  for (auto x : state) {
    krylov::BlockJacobiPrec<Scalar, kPrecBlockSize> P(A);
    MOCHI_NO_DISCARD_IN_LOOP(P);
  }
}

template <typename Scalar, int kMatBlockSize, int kPrecBlockSize>
static void BlockJacobiConstructionFromBlockSparseMatrix(benchmark::State& state) {
  int nx = state.range(0), ny = state.range(1), nz = state.range(2);
  auto A = test::MakeBlockSparseMatrix<Scalar, kMatBlockSize>(nx, ny, nz);
  for (auto x : state) {
    krylov::BlockJacobiPrec<Scalar, kPrecBlockSize> P(A);
    MOCHI_NO_DISCARD_IN_LOOP(P);
  }
}

// Jacobi construction. Benchmark performance when the input matrix is sparse and block sparse with
// block size of 3.
BENCHMARK_TEMPLATE(BlockJacobiConstructionFromSparseMatrix, float, 3, 1)
    ->Name("Preconditioner/Jacobi/Construction/SparseMatrix")
    ->ArgNames({"nx", "ny", "nz"})
    ->Args({13, 17, 19});
BENCHMARK_TEMPLATE(BlockJacobiConstructionFromBlockSparseMatrix, float, 3, 1)
    ->Name("Preconditioner/Jacobi/Construction/BlockSparseMatrix")
    ->ArgNames({"nx", "ny", "nz"})
    ->Args({13, 17, 19});

// Block Jacobi construction with block size of 3. Benchmark performance when the input matrix is
// sparse and block sparse with block size of 3.
BENCHMARK_TEMPLATE(BlockJacobiConstructionFromSparseMatrix, float, 3, 3)
    ->Name("Preconditioner/BlockJacobi/Construction/SparseMatrix")
    ->ArgNames({"nx", "ny", "nz"})
    ->Args({13, 17, 19});
BENCHMARK_TEMPLATE(BlockJacobiConstructionFromBlockSparseMatrix, float, 3, 3)
    ->Name("Preconditioner/BlockJacobi/Construction/BlockSparseMatrix")
    ->ArgNames({"nx", "ny", "nz"})
    ->Args({13, 17, 19});

/****************************************************************************************
  Construction and update of AMG preconditioner
*/

template <typename Scalar, int kMatBlockSize>
static void AMGConstructionFromBlockSparseMatrix(benchmark::State& state) {
  auto nx = static_cast<int>(state.range(0));
  auto ny = static_cast<int>(state.range(1));
  auto nz = static_cast<int>(state.range(2));
  auto numThreads = static_cast<int>(state.range(3));
  MOCHI_ASSERT(numThreads > 0, "Invalid number of threads.");
  if (numThreads > TaskScheduler::GetNumSupportedLogicalProcessors()) {
    MOCHI_SKIP_BENCHMARK(state);
    return;
  }
  auto A = test::MakeBlockSparseMatrix<Scalar, kMatBlockSize>(nx, ny, nz);
  TaskScheduler scheduler(numThreads - 1); // Caller not included in TaskScheduler's pool
  for (auto _ : state) {
    krylov::AMGPrec<Scalar, kMatBlockSize> P(A);
    MOCHI_NO_DISCARD_IN_LOOP(P);
  }
}

BENCHMARK_TEMPLATE(AMGConstructionFromBlockSparseMatrix, float, 3)
    ->Name("Preconditioner/AMG/Construction/BlockSparseMatrix")
    ->ArgNames({"nx", "ny", "nz", "threads"})
    ->Args({13, 17, 19, 1})
    ->Args({13, 17, 19, 2})
    ->Args({13, 17, 19, 4})
    ->Args({13, 17, 19, 8})
    ->UseRealTime();

/****************************************************************************************
  "Sparse Matrix" x "Block Sparse Matrix" (for AMG operation P^T * A)
*/

template <typename Scalar, int kBlockSize, typename CRIdx, typename Ptr>
static void AMGPtAProduct(benchmark::State& state) {
  auto nx = static_cast<int>(state.range(0));
  auto ny = static_cast<int>(state.range(1));
  auto nz = static_cast<int>(state.range(2));
  auto numThreads = static_cast<int>(state.range(3));
  MOCHI_ASSERT(numThreads > 0, "Invalid number of threads.");
  if (numThreads > TaskScheduler::GetNumSupportedLogicalProcessors()) {
    MOCHI_SKIP_BENCHMARK(state);
    return;
  }
  auto A = test::MakeBlockSparseMatrix<Scalar, kBlockSize>(nx, ny, nz);
  // Extract the node-to-node graph
  auto nToN = AsGraphView(A);
  // Create the tentative prolongation
  auto [partition, numAgg] = details::Aggregate(nToN);
  // Smoothen the prolongation
  auto P = details::Smoothing(A, numAgg, partition, Scalar(2.0 / 3.0));
  auto Pt = Transpose(P);
  auto gPtA = Traverse(AsGraphView(Pt), nToN).SortTargets();
  BlockSparseMatrix<Scalar, kBlockSize, CRIdx, CRIdx> PtA(A.BlockCols(), gPtA);
  TaskScheduler scheduler(numThreads - 1); // Caller not included in TaskScheduler's pool
  //
  for (auto _ : state) {
    krylov::details::SparseMatProduct(Pt, A, PtA);
    MOCHI_NO_DISCARD_IN_LOOP(PtA);
  }
  benchmark::DoNotOptimize(PtA);
  //--- Compute the number of FLOPs
  double operations = 0;
  using Idx = std::remove_const_t<CRIdx>;
  for (Idx i = 0; i < Pt.Rows(); ++i) {
    for (auto ptColIdx : Pt.Indices(i)) {
      auto const AcolIndices = A.Indices(ptColIdx);
      operations += double(2 * kBlockSize * kBlockSize) * AcolIndices.size();
    }
  }
  state.counters["FLOPs"] =
      benchmark::Counter(double(state.iterations()) * operations, benchmark::Counter::kIsRate);
}

// Benchmarks for performance on fixed grid with multiple threads
BENCHMARK_TEMPLATE(AMGPtAProduct, float, 3, int, int)
    ->Name("Preconditioner/AMG/PtAProduct/BlockSparseMatrix")
    ->ArgNames({"nx", "ny", "nz", "threads"})
    ->ArgsProduct({{13}, {17}, {19}, {1, 2, 4, 8}})
    ->UseRealTime();

template <typename Scalar, int kMatBlockSize>
static void AMGUpdateFromBlockSparseMatrix(benchmark::State& state) {
  auto nx = static_cast<int>(state.range(0));
  auto ny = static_cast<int>(state.range(1));
  auto nz = static_cast<int>(state.range(2));
  auto numThreads = static_cast<int>(state.range(3));
  MOCHI_ASSERT(numThreads > 0, "Invalid number of threads.");
  if (numThreads > TaskScheduler::GetNumSupportedLogicalProcessors()) {
    MOCHI_SKIP_BENCHMARK(state);
    return;
  }
  auto A = test::MakeBlockSparseMatrix<Scalar, kMatBlockSize>(nx, ny, nz);
  krylov::AMGPrec<Scalar, kMatBlockSize> P(A);
  TaskScheduler scheduler(numThreads - 1); // Caller not included in TaskScheduler's pool
  for (auto _ : state) {
    P.Update(A);
    MOCHI_NO_DISCARD_IN_LOOP(P);
  }
}

BENCHMARK_TEMPLATE(AMGUpdateFromBlockSparseMatrix, float, 3)
    ->Name("Preconditioner/AMG/Update/BlockSparseMatrix")
    ->ArgNames({"nx", "ny", "nz", "threads"})
    ->Args({13, 17, 19, 1})
    ->Args({13, 17, 19, 2})
    ->Args({13, 17, 19, 4})
    ->Args({13, 17, 19, 8})
    ->UseRealTime();

/****************************************************************************************
Construction of IC-0 preconditioner
*/

template <typename Scalar, int kMatBlockSize>
static void IC0ConstructionFromSparseMatrix(benchmark::State& state) {
  auto nx = int(state.range(0)), ny = int(state.range(1)), nz = int(state.range(2));
  auto A = test::MakeSparseMatrixWithBlockStructure<Scalar, kMatBlockSize>(nx, ny, nz);
  for (auto _ : state) {
    krylov::IncompleteCholeskyPrec<SparseMatrix<Scalar>> P(
        A, /*fillInLevel*/ 0, /*alphaShift*/ Scalar{0});
    MOCHI_NO_DISCARD_IN_LOOP(P);
  }
}

template <typename Scalar, int kMatBlockSize>
static void IC0ConstructionFromBlockSparseMatrix(benchmark::State& state) {
  auto nx = int(state.range(0)), ny = int(state.range(1)), nz = int(state.range(2));
  auto A = test::MakeBlockSparseMatrix<Scalar, kMatBlockSize>(nx, ny, nz);
  for (auto _ : state) {
    krylov::IncompleteCholeskyPrec<BlockSparseMatrix<Scalar, kMatBlockSize>> P(
        A, /*fillInLevel*/ 0, /*alphaShift*/ Scalar{0});
    MOCHI_NO_DISCARD_IN_LOOP(P);
  }
}

BENCHMARK_TEMPLATE(IC0ConstructionFromSparseMatrix, float, 3)
    ->Name("Preconditioner/IC0/Construction/SparseMatrix")
    ->ArgNames({"nx", "ny", "nz"})
    ->Args({13, 17, 19});
BENCHMARK_TEMPLATE(IC0ConstructionFromBlockSparseMatrix, float, 3)
    ->Name("Preconditioner/IC0/Construction/BlockSparseMatrix")
    ->ArgNames({"nx", "ny", "nz"})
    ->Args({13, 17, 19});

/****************************************************************************************
Construction of relaxed ILU-0 preconditioner
*/

template <typename Scalar, int kMatBlockSize>
static void ILU0ConstructionFromSparseMatrix(benchmark::State& state) {
  int nx = state.range(0), ny = state.range(1), nz = state.range(2);
  auto A = test::MakeSparseMatrixWithBlockStructure<Scalar, kMatBlockSize>(nx, ny, nz);
  for (auto x : state) {
    krylov::RelaxedILUPrec<SparseMatrix<Scalar>> P(A, /*fillInLevel*/ 0, /*alphaRelax*/ Scalar{0});
    MOCHI_NO_DISCARD_IN_LOOP(P);
  }
}

template <typename Scalar, int kMatBlockSize>
static void ILU0ConstructionFromBlockSparseMatrix(benchmark::State& state) {
  int nx = state.range(0), ny = state.range(1), nz = state.range(2);
  auto A = test::MakeBlockSparseMatrix<Scalar, kMatBlockSize>(nx, ny, nz);
  for (auto x : state) {
    krylov::RelaxedILUPrec<BlockSparseMatrix<Scalar, kMatBlockSize>> P(
        A, /*fillInLevel*/ 0, /*alphaRelax*/ Scalar{0});
    MOCHI_NO_DISCARD_IN_LOOP(P);
  }
}

BENCHMARK_TEMPLATE(ILU0ConstructionFromSparseMatrix, float, 3)
    ->Name("Preconditioner/ILU0/Construction/SparseMatrix")
    ->ArgNames({"nx", "ny", "nz"})
    ->Args({13, 17, 19});
BENCHMARK_TEMPLATE(ILU0ConstructionFromBlockSparseMatrix, float, 3)
    ->Name("Preconditioner/ILU0/Construction/BlockSparseMatrix")
    ->ArgNames({"nx", "ny", "nz"})
    ->Args({13, 17, 19});

/****************************************************************************************
Sparse and block sparse matrix-vector products (for reference)
*/

template <typename Scalar, int kMatBlockSize>
static void ReferenceSparseMatVecProduct(benchmark::State& state) {
  int nx = state.range(0), ny = state.range(1), nz = state.range(2);
  auto A = test::MakeSparseMatrixWithBlockStructure<Scalar, kMatBlockSize>(nx, ny, nz);
  ColumnVector<Scalar> b(A.Rows()), c(A.Rows());
  b.SetRandom(1);
  for (auto x : state) {
    c = A * b;
    MOCHI_NO_DISCARD_IN_LOOP(c);
  }
  state.counters["FLOPs"] = benchmark::Counter(
      state.iterations() * 2 * A.NumNonZeros(), // Only leading term
      benchmark::Counter::kIsRate);
  benchmark::DoNotOptimize(c);
}

template <typename Scalar, int kMatBlockSize>
static void ReferenceBlockSparseMatVecProduct(benchmark::State& state) {
  int nx = state.range(0), ny = state.range(1), nz = state.range(2);
  auto A = test::MakeBlockSparseMatrix<Scalar, kMatBlockSize>(nx, ny, nz);
  ColumnVector<Scalar> b(A.Rows()), c(A.Rows());
  b.SetRandom(1);
  for (auto x : state) {
    c = A * b;
    MOCHI_NO_DISCARD_IN_LOOP(c);
  }
  state.counters["FLOPs"] = benchmark::Counter(
      state.iterations() * 2 * A.NumNonZeros(), // Only leading term
      benchmark::Counter::kIsRate);
  benchmark::DoNotOptimize(c);
}

// SparseMatrix x Vector
BENCHMARK_TEMPLATE(ReferenceSparseMatVecProduct, float, 3)
    ->Name("Preconditioner/Reference/SparseMatVec/SparseMatrix")
    ->ArgNames({"nx", "ny", "nz"})
    ->Args({13, 17, 19});

// BlockSparseMatrix x Vector.
BENCHMARK_TEMPLATE(ReferenceBlockSparseMatVecProduct, float, 3)
    ->Name("Preconditioner/Reference/SparseMatVec/BlockSparseMatrix")
    ->ArgNames({"nx", "ny", "nz"})
    ->Args({13, 17, 19});

/****************************************************************************************
  Application of (block) SSOR preconditioner
*/

template <typename Scalar, int kMatBlockSize>
static void SSORSolveFromSparseMatrix(benchmark::State& state) {
  int nx = state.range(0), ny = state.range(1), nz = state.range(2);
  auto A = test::MakeSparseMatrixWithBlockStructure<Scalar, kMatBlockSize>(nx, ny, nz);
  ColumnVector<Scalar> b(A.Rows()), c(A.Rows());
  b.SetRandom(1);
  krylov::SSORPrec<Scalar, SparseMatrix<Scalar>> P(A);
  for (auto x : state) {
    P(b, c);
    MOCHI_NO_DISCARD_IN_LOOP(c);
  }
  state.counters["FLOPs"] = benchmark::Counter(
      state.iterations() * 2 * A.NumNonZeros(), // Only leading term
      benchmark::Counter::kIsRate);
  benchmark::DoNotOptimize(c);
}

template <typename Scalar, int kMatBlockSize>
static void ColoredSSORSolveFromSparseMatrix(benchmark::State& state) {
  auto nx = static_cast<int>(state.range(0));
  auto ny = static_cast<int>(state.range(1));
  auto nz = static_cast<int>(state.range(2));
  auto numThreads = static_cast<int>(state.range(3));
  MOCHI_ASSERT(numThreads > 0, "Invalid number of threads.");
  if (numThreads > TaskScheduler::GetNumSupportedLogicalProcessors()) {
    MOCHI_SKIP_BENCHMARK(state);
    return;
  }
  auto A = test::MakeSparseMatrixWithBlockStructure<Scalar, kMatBlockSize>(nx, ny, nz);
  ColumnVector<Scalar> b(A.Rows()), c(A.Rows());
  b.SetRandom(1);
  krylov::ColoredSSORPrec<SparseMatrix<Scalar>> P(A);
  TaskScheduler scheduler(numThreads - 1); // Caller not included in TaskScheduler's pool
  //
  for (auto _ : state) {
    P(b, c);
    MOCHI_NO_DISCARD_IN_LOOP(c);
  }
  state.counters["FLOPs"] = benchmark::Counter(
      state.iterations() * 2 * A.NumNonZeros(), // Only leading term
      benchmark::Counter::kIsRate);
  benchmark::DoNotOptimize(c);
}

template <typename Scalar, int kMatBlockSize>
static void SSORSolveFromBlockSparseMatrix(benchmark::State& state) {
  int nx = state.range(0), ny = state.range(1), nz = state.range(2);
  auto A = test::MakeBlockSparseMatrix<Scalar, kMatBlockSize>(nx, ny, nz);
  ColumnVector<Scalar> b(A.Rows()), c(A.Rows());
  b.SetRandom(1);
  krylov::SSORPrec<Scalar, BlockSparseMatrix<Scalar, kMatBlockSize>> P(A);
  for (auto x : state) {
    P(b, c);
    MOCHI_NO_DISCARD_IN_LOOP(c);
  }
  state.counters["FLOPs"] = benchmark::Counter(
      state.iterations() * 2 * A.NumNonZeros(), // Only leading term
      benchmark::Counter::kIsRate);
  benchmark::DoNotOptimize(c);
}

template <typename Scalar, int kMatBlockSize>
static void ColoredSSORSolveFromBlockSparseMatrix(benchmark::State& state) {
  auto nx = static_cast<int>(state.range(0));
  auto ny = static_cast<int>(state.range(1));
  auto nz = static_cast<int>(state.range(2));
  auto numThreads = static_cast<int>(state.range(3));
  MOCHI_ASSERT(numThreads > 0, "Invalid number of threads.");
  if (numThreads > TaskScheduler::GetNumSupportedLogicalProcessors()) {
    MOCHI_SKIP_BENCHMARK(state);
    return;
  }
  auto A = test::MakeBlockSparseMatrix<Scalar, kMatBlockSize>(nx, ny, nz);
  ColumnVector<Scalar> b(A.Rows()), c(A.Rows());
  b.SetRandom(1);
  krylov::ColoredSSORPrec<BlockSparseMatrix<Scalar, kMatBlockSize>> P(A);
  TaskScheduler scheduler(numThreads - 1); // Caller not included in TaskScheduler's pool
  //
  for (auto _ : state) {
    P(b, c);
    MOCHI_NO_DISCARD_IN_LOOP(c);
  }
  state.counters["FLOPs"] = benchmark::Counter(
      state.iterations() * 2 * A.NumNonZeros(), // Only leading term
      benchmark::Counter::kIsRate);
  benchmark::DoNotOptimize(c);
}

template <typename Scalar, int kMatBlockSize, int kPrecBlockSize>
static void BlockSSORSolveFromSparseMatrix(benchmark::State& state) {
  int nx = state.range(0), ny = state.range(1), nz = state.range(2);
  auto A = test::MakeSparseMatrixWithBlockStructure<Scalar, kMatBlockSize>(nx, ny, nz);
  ColumnVector<Scalar> b(A.Rows()), c(A.Rows());
  b.SetRandom(1);
  krylov::BlockSSORPrec<Scalar, kPrecBlockSize, SparseMatrix<Scalar>> P(A);
  for (auto x : state) {
    P(b, c);
    MOCHI_NO_DISCARD_IN_LOOP(c);
  }
  state.counters["FLOPs"] = benchmark::Counter(
      state.iterations() * 2 * A.NumNonZeros(), // Only leading term
      benchmark::Counter::kIsRate);
  benchmark::DoNotOptimize(c);
}

template <typename Scalar, int kMatBlockSize, int kPrecBlockSize>
static void BlockSSORSolveFromBlockSparseMatrix(benchmark::State& state) {
  int nx = state.range(0), ny = state.range(1), nz = state.range(2);
  auto A = test::MakeBlockSparseMatrix<Scalar, kMatBlockSize>(nx, ny, nz);
  ColumnVector<Scalar> b(A.Rows()), c(A.Rows());
  b.SetRandom(1);
  krylov::BlockSSORPrec<Scalar, kPrecBlockSize, BlockSparseMatrix<Scalar, kMatBlockSize>> P(A);
  for (auto x : state) {
    P(b, c);
    MOCHI_NO_DISCARD_IN_LOOP(c);
  }
  state.counters["FLOPs"] = benchmark::Counter(
      state.iterations() * 2 * A.NumNonZeros(), // Only leading term
      benchmark::Counter::kIsRate);
  benchmark::DoNotOptimize(c);
}

// SSOR solve. Benchmark performance when the input matrix is sparse and block sparse with block
// size of 3.
BENCHMARK_TEMPLATE(SSORSolveFromSparseMatrix, float, 3)
    ->Name("Preconditioner/SSOR/Solve/SparseMatrix")
    ->ArgNames({"nx", "ny", "nz"})
    ->Args({13, 17, 19});
BENCHMARK_TEMPLATE(SSORSolveFromBlockSparseMatrix, float, 3)
    ->Name("Preconditioner/SSOR/Solve/BlockSparseMatrix")
    ->ArgNames({"nx", "ny", "nz"})
    ->Args({13, 17, 19});

BENCHMARK_TEMPLATE(ColoredSSORSolveFromSparseMatrix, float, 3)
    ->Name("Preconditioner/ColoredSSOR/Solve/SparseMatrix")
    ->ArgNames({"nx", "ny", "nz", "threads"})
    ->Args({13, 17, 19, 1})
    ->Args({13, 17, 19, 2})
    ->Args({13, 17, 19, 4})
    ->Args({13, 17, 19, 8})
    ->UseRealTime();

BENCHMARK_TEMPLATE(ColoredSSORSolveFromBlockSparseMatrix, float, 3)
    ->Name("Preconditioner/ColoredSSOR/Solve/BlockSparseMatrix")
    ->ArgNames({"nx", "ny", "nz", "threads"})
    ->Args({13, 17, 19, 1})
    ->Args({13, 17, 19, 2})
    ->Args({13, 17, 19, 4})
    ->Args({13, 17, 19, 8})
    ->UseRealTime();

// Block SSOR solve with block size of 3. Benchmark performance when the input matrix is sparse and
// block sparse with block size of 3.
BENCHMARK_TEMPLATE(BlockSSORSolveFromSparseMatrix, float, 3, 3)
    ->Name("Preconditioner/BlockSSOR/Solve/SparseMatrix")
    ->ArgNames({"nx", "ny", "nz"})
    ->Args({13, 17, 19});
BENCHMARK_TEMPLATE(BlockSSORSolveFromBlockSparseMatrix, float, 3, 3)
    ->Name("Preconditioner/BlockSSOR/Solve/BlockSparseMatrix")
    ->ArgNames({"nx", "ny", "nz"})
    ->Args({13, 17, 19});

/****************************************************************************************
  Application of (block) Jacobi preconditioner
*/

template <typename Scalar, int kMatBlockSize, int kPrecBlockSize>
static void BlockJacobiSolve(benchmark::State& state) {
  int nx = state.range(0), ny = state.range(1), nz = state.range(2);
  auto A = test::MakeBlockSparseMatrix<Scalar, kMatBlockSize>(nx, ny, nz);
  ColumnVector<Scalar> b(A.Rows()), c(A.Rows());
  b.SetRandom(1);
  krylov::BlockJacobiPrec<Scalar, kPrecBlockSize> P(A);
  for (auto x : state) {
    P(b, c);
    MOCHI_NO_DISCARD_IN_LOOP(c);
  }
  state.counters["FLOPs"] = benchmark::Counter(
      state.iterations() * A.Rows() * (2 * kPrecBlockSize - 1), benchmark::Counter::kIsRate);
  benchmark::DoNotOptimize(c);
}

// Jacobi solve. Performance is independent of the format of the input matrix.
BENCHMARK_TEMPLATE(BlockJacobiSolve, float, 3, 1)
    ->Name("Preconditioner/Jacobi/Solve")
    ->ArgNames({"nx", "ny", "nz"})
    ->Args({13, 17, 19});

// Block Jacobi solve. Performance is independent of the format of the input matrix.
BENCHMARK_TEMPLATE(BlockJacobiSolve, float, 3, 3)
    ->Name("Preconditioner/BlockJacobi/Solve")
    ->ArgNames({"nx", "ny", "nz"})
    ->Args({13, 17, 19});

/****************************************************************************************
  Application of AMG preconditioner
*/

template <typename Scalar, int kBlockSize, krylov::Smoother kSmoother>
static void AMGSolveFromBlockSparseMatrix(benchmark::State& state) {
  auto nx = static_cast<int>(state.range(0));
  auto ny = static_cast<int>(state.range(1));
  auto nz = static_cast<int>(state.range(2));
  auto numThreads = static_cast<int>(state.range(3));
  MOCHI_ASSERT(numThreads > 0, "Invalid number of threads.");
  if (numThreads > TaskScheduler::GetNumSupportedLogicalProcessors()) {
    MOCHI_SKIP_BENCHMARK(state);
    return;
  }
  auto A = test::MakeBlockSparseMatrix<Scalar, kBlockSize>(nx, ny, nz);
  ColumnVector<Scalar> b(A.Rows()), c(A.Rows());
  b.SetRandom(1);
  krylov::AMGOptions<Scalar> options{.smoother = kSmoother};
  krylov::AMGPrec<Scalar, kBlockSize> P(A, options);
  TaskScheduler scheduler(numThreads - 1); // Caller not included in TaskScheduler's pool
  for (auto _ : state) {
    P(b, c);
    MOCHI_NO_DISCARD_IN_LOOP(c);
  }
  benchmark::DoNotOptimize(c);
}

BENCHMARK_TEMPLATE(AMGSolveFromBlockSparseMatrix, float, 3, krylov::Smoother::BlockJacobi)
    ->Name("Preconditioner/AMG/Solve/BlockSparseMatrix/BlockJacobiSmoother")
    ->ArgNames({"nx", "ny", "nz", "threads"})
    ->ArgsProduct({{13}, {17}, {19}, {1, 2, 4, 8}})
    ->UseRealTime();
BENCHMARK_TEMPLATE(AMGSolveFromBlockSparseMatrix, float, 3, krylov::Smoother::SSOR)
    ->Name("Preconditioner/AMG/Solve/BlockSparseMatrix/SSORSmoother")
    ->ArgNames({"nx", "ny", "nz", "threads"})
    ->ArgsProduct({{13}, {17}, {19}, {1, 2, 4, 8}})
    ->UseRealTime();

/****************************************************************************************
Application of IC-0 preconditioner
*/

template <typename Scalar, int kBlockSize>
static void IC0SolveFromBlockSparseMatrix(benchmark::State& state) {
  auto nx = state.range(0), ny = state.range(1), nz = state.range(2);
  auto A = test::MakeBlockSparseMatrix<Scalar, kBlockSize>(int(nx), int(ny), int(nz));
  ColumnVector<Scalar> b(A.Rows()), c(A.Rows());
  b.SetRandom(1);
  krylov::IncompleteCholeskyPrec<BlockSparseMatrix<Scalar, kBlockSize>> P(
      A, /*fillInLevel*/ 0, /*alphaShift*/ Scalar{0});
  for (auto _ : state) {
    P(b, c);
    MOCHI_NO_DISCARD_IN_LOOP(c);
  }
  state.counters["FLOPs"] = benchmark::Counter(
      state.iterations() * 2 * A.NumNonZeros(), // Only leading term
      benchmark::Counter::kIsRate);
  benchmark::DoNotOptimize(c);
}

template <typename Scalar>
static void IC0SolveFromSparseMatrix(benchmark::State& state) {
  auto nx = int(state.range(0)), ny = int(state.range(1)), nz = int(state.range(2));
  constexpr int kBlockSize = 3; // used to represent elasticity-like problem
  //
  auto A = test::MakeSparseMatrixWithBlockStructure<Scalar, kBlockSize>(nx, ny, nz);
  krylov::IncompleteCholeskyPrec<SparseMatrix<Scalar>> P(
      A, /*fillInLevel*/ 0, /*alphaShift*/ Scalar{0});
  //
  ColumnVector<Scalar> b(A.Rows()), c(A.Rows());
  b.SetRandom(1);
  //
  for (auto _ : state) {
    P(b, c);
    MOCHI_NO_DISCARD_IN_LOOP(c);
  }
  state.counters["FLOPs"] = benchmark::Counter(
      state.iterations() * 2 * A.NumNonZeros(), // Only leading term
      benchmark::Counter::kIsRate);
  benchmark::DoNotOptimize(c);
}

BENCHMARK_TEMPLATE(IC0SolveFromSparseMatrix, float)
    ->Name("Preconditioner/IC0/Solve/SparseMatrix")
    ->ArgNames({"nx", "ny", "nz"})
    ->Args({13, 17, 19});

BENCHMARK_TEMPLATE(IC0SolveFromBlockSparseMatrix, float, 3)
    ->Name("Preconditioner/IC0/Solve/BlockSparseMatrix")
    ->ArgNames({"nx", "ny", "nz"})
    ->Args({13, 17, 19});

/****************************************************************************************
Application of ILU-0 preconditioner
*/

template <typename Scalar, int kBlockSize>
static void ILU0SolveFromBlockSparseMatrix(benchmark::State& state) {
  int nx = state.range(0), ny = state.range(1), nz = state.range(2);
  auto A = test::MakeBlockSparseMatrix<Scalar, kBlockSize>(nx, ny, nz);
  ColumnVector<Scalar> b(A.Rows()), c(A.Rows());
  b.SetRandom(1);
  krylov::RelaxedILUPrec<BlockSparseMatrix<Scalar, kBlockSize>> P(
      A, /*fillInLevel*/ 0, /*alphaRelax*/ Scalar{0});
  for (auto x : state) {
    P(b, c);
    MOCHI_NO_DISCARD_IN_LOOP(c);
  }
  state.counters["FLOPs"] = benchmark::Counter(
      state.iterations() * 2 * A.NumNonZeros(), // Only leading term
      benchmark::Counter::kIsRate);
  benchmark::DoNotOptimize(c);
}

template <typename Scalar>
static void ILU0SolveFromSparseMatrix(benchmark::State& state) {
  int nx = int(state.range(0)), ny = int(state.range(1)), nz = int(state.range(2));
  constexpr int kBlockSize = 3; // used to represent elasticity-like problem
  auto A = test::MakeSparseMatrixWithBlockStructure<Scalar, kBlockSize>(nx, ny, nz);
  ColumnVector<Scalar> b(A.Rows()), c(A.Rows());
  b.SetRandom(1);
  krylov::RelaxedILUPrec<SparseMatrix<Scalar>> P(A, /*fillInLevel*/ 0, /*alphaRelax*/ Scalar{0});
  for (auto x : state) {
    P(b, c);
    MOCHI_NO_DISCARD_IN_LOOP(c);
  }
  state.counters["FLOPs"] = benchmark::Counter(
      state.iterations() * 2 * A.NumNonZeros(), // Only leading term
      benchmark::Counter::kIsRate);
  benchmark::DoNotOptimize(c);
}

BENCHMARK_TEMPLATE(ILU0SolveFromSparseMatrix, float)
    ->Name("Preconditioner/ILU0/Solve/SparseMatrix")
    ->ArgNames({"nx", "ny", "nz"})
    ->Args({13, 17, 19});
BENCHMARK_TEMPLATE(ILU0SolveFromBlockSparseMatrix, float, 3)
    ->Name("Preconditioner/ILU0/Solve/BlockSparseMatrix")
    ->ArgNames({"nx", "ny", "nz"})
    ->Args({13, 17, 19});

} // namespace mochi_benchmark
