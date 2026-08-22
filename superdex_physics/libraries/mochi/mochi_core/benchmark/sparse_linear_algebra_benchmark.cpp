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
#include <mochi_core/linear_algebra/block_sparse_matrix_utils.h>
#include <mochi_core/linear_algebra/krylov/sparse_ldlt.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/sparse_matrix.h>
#include <mochi_core/linear_algebra/sparse_matrix_utils.h>
#include <mochi_core/linear_algebra/utils/matrix_conversions.h>
#include <mochi_core/test/linear_algebra_test_helpers.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/graph.h>
#include <mochi_core/utils/graph_views.h>
#include <mochi_core/utils/task_scheduler.h>

#include <algorithm>
#include <type_traits>
#include <utility>

using namespace mochi;

namespace mochi_benchmark {

constexpr auto kColMajor = krylov::Direction::ColMajor;
constexpr auto kRowMajor = krylov::Direction::RowMajor;
constexpr auto kDynamic = krylov::kDynamic;

/****************************************************************************************
  "Block Sparse Matrix" x "Vector"
*/

template <typename Scalar, typename CRIdx, typename Ptr, int kBlockSize, krylov::Direction kDir>
static void BlockSparseMatVecProduct(benchmark::State& state) {
  auto const numBlockRows = CRIdx(state.range(0)), numNonZeroBlocksPerRow = CRIdx(state.range(1));
  auto const numRhsCols = int(state.range(2));
  auto const A = test::MakeBlockSparseMatrixWithTargetNnz<Scalar, kBlockSize, CRIdx, Ptr>(
      numBlockRows, numNonZeroBlocksPerRow);
  auto const numBlockCols = A.BlockCols();
  Matrix<Scalar, kDynamic, kDynamic, kDir> b(kBlockSize * numBlockCols, numRhsCols),
      c(kBlockSize * numBlockRows, numRhsCols);
  b.SetRandom(1);
  for (auto x : state) {
    c = A * b;
    MOCHI_NO_DISCARD_IN_LOOP(c);
  }
  state.counters["FLOPs"] = benchmark::Counter(
      state.iterations() * (2 * A.NumNonZeros() - A.Rows()) * numRhsCols,
      benchmark::Counter::kIsRate);
  benchmark::DoNotOptimize(c);
}

// Benchmarks for performance vs. number of non-zero blocks per row.
BENCHMARK_TEMPLATE(BlockSparseMatVecProduct, float, int, int, 3, kColMajor)
    ->Name("SparseLA/BlockSparseMatVec/NnzSweep/Block3/ColMajor")
    ->ArgNames({"blockRows", "nnzBlocksPerRow", "rhsCols"})
    ->ArgsProduct({{5000}, benchmark::CreateDenseRange(1, 16, /*step=*/1), {1}})
    ->ArgsProduct({{5000}, {100, 1000}, {1}});

BENCHMARK_TEMPLATE(BlockSparseMatVecProduct, float, int, int, 4, kColMajor)
    ->Name("SparseLA/BlockSparseMatVec/NnzSweep/Block4/ColMajor")
    ->ArgNames({"blockRows", "nnzBlocksPerRow", "rhsCols"})
    ->ArgsProduct({{5000}, benchmark::CreateDenseRange(1, 16, /*step=*/1), {1}})
    ->ArgsProduct({{5000}, {100, 1000}, {1}});

// Dense matrix represented as block sparse matrix.
BENCHMARK_TEMPLATE(BlockSparseMatVecProduct, float, int, int, 3, kColMajor)
    ->Name("SparseLA/BlockSparseMatVec/DenseCase/Block3/ColMajor")
    ->ArgNames({"blockRows", "nnzBlocksPerRow", "rhsCols"})
    ->Args({17, 17, 1})
    ->Args({42, 42, 1})
    ->Args({167, 167, 1});

// Use a brick-mesh for the block sparse matrix.
template <typename Scalar, int kMatBlockSize>
static void BlockSparseMatVecProductCube(benchmark::State& state) {
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
  TaskScheduler scheduler(numThreads - 1); // Caller not included in TaskScheduler's pool
  for (auto _ : state) {
    c = A * b;
    MOCHI_NO_DISCARD_IN_LOOP(c);
  }
  state.counters["FLOPs"] = benchmark::Counter(
      state.iterations() * (2 * A.NumNonZeros() - A.Rows()), benchmark::Counter::kIsRate);
  benchmark::DoNotOptimize(c);
}

// Strong scaling benchmarks.
BENCHMARK_TEMPLATE(BlockSparseMatVecProductCube, float, 3)
    ->Name("SparseLA/BlockSparseMatVec/Cube/Block3")
    ->ArgNames({"nx", "ny", "nz", "threads"})
    ->ArgsProduct({{16}, {17}, {19}, {1, 2, 4, 8}})
    ->UseRealTime();

/****************************************************************************************
  "Block Sparse Matrix" x "Vector" with block sparse matrix represented as sparse matrix
*/

template <typename Scalar, typename CRIdx, typename Ptr, int kBlockSize, krylov::Direction kDir>
static void BlockSparseMatVecProductAsSparse(benchmark::State& state) {
  auto const numBlockRows = CRIdx(state.range(0)), numNonZeroBlocksPerRow = CRIdx(state.range(1));
  auto const numRhsCols = int(state.range(2));
  auto const ABSp = test::MakeBlockSparseMatrixWithTargetNnz<Scalar, kBlockSize, CRIdx, Ptr>(
      numBlockRows, numNonZeroBlocksPerRow);
  SparseMatrix<Scalar, CRIdx, Ptr> A = ToSparseMatrix(ABSp);
  Matrix<Scalar, kDynamic, kDynamic, kDir> b(A.Cols(), numRhsCols), c(A.Rows(), numRhsCols);
  b.SetRandom(1);
  for (auto x : state) {
    c = A * b;
    MOCHI_NO_DISCARD_IN_LOOP(c);
  }
  state.counters["FLOPs"] = benchmark::Counter(
      state.iterations() * (2 * A.NumNonZeros() - A.Rows()) * numRhsCols,
      benchmark::Counter::kIsRate);
  benchmark::DoNotOptimize(c);
}

// Benchmarks for performance vs. number of non-zeros per row.
BENCHMARK_TEMPLATE(BlockSparseMatVecProductAsSparse, float, int, int, 3, kColMajor)
    ->Name("SparseLA/BlockSparseAsSparseMatVec/NnzSweep/Block3/ColMajor")
    ->ArgNames({"blockRows", "nnzBlocksPerRow", "rhsCols"})
    ->ArgsProduct({{5000}, {1, 4, 7, 10, 13, 16, 100, 1000}, {1}});

BENCHMARK_TEMPLATE(BlockSparseMatVecProductAsSparse, float, int, int, 4, kColMajor)
    ->Name("SparseLA/BlockSparseAsSparseMatVec/NnzSweep/Block4/ColMajor")
    ->ArgNames({"blockRows", "nnzBlocksPerRow", "rhsCols"})
    ->ArgsProduct({{5000}, {1, 4, 7, 10, 13, 16, 100, 1000}, {1}});

// Use a brick-mesh for the block sparse matrix.
template <typename Scalar, int kMatBlockSize>
static void BlockSparseMatVecProductAsSparseCube(benchmark::State& state) {
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
  TaskScheduler scheduler(numThreads - 1); // Caller not included in TaskScheduler's pool
  for (auto _ : state) {
    c = A * b;
    MOCHI_NO_DISCARD_IN_LOOP(c);
  }
  state.counters["FLOPs"] = benchmark::Counter(
      state.iterations() * (2 * A.NumNonZeros() - A.Rows()), benchmark::Counter::kIsRate);
  benchmark::DoNotOptimize(c);
}

// Strong scaling benchmarks.
BENCHMARK_TEMPLATE(BlockSparseMatVecProductAsSparseCube, float, 3)
    ->Name("SparseLA/BlockSparseAsSparseMatVec/Cube/Block3")
    ->ArgNames({"nx", "ny", "nz", "threads"})
    ->ArgsProduct({{16}, {17}, {19}, {1, 2, 4, 8}})
    ->UseRealTime();

/****************************************************************************************
  "Sparse Matrix" x "Vector"
*/

template <typename Scalar, typename CRIdx, typename Ptr, krylov::Direction kDir>
static void SparseMatVecProduct(benchmark::State& state) {
  BlockSparseMatVecProductAsSparse<Scalar, CRIdx, Ptr, 1, kDir>(state);
}

// Benchmarks for performance vs. number of non-zeros per row.
BENCHMARK_TEMPLATE(SparseMatVecProduct, float, int, int, kColMajor)
    ->Name("SparseLA/SparseMatVec/NnzSweep/ColMajor")
    ->ArgNames({"rows", "nnzPerRow", "rhsCols"})
    ->ArgsProduct({{15000}, {3, 12, 21, 30, 39, 48, 300, 3000}, {1}});

// Dense matrix represented as sparse matrix.
BENCHMARK_TEMPLATE(SparseMatVecProduct, float, int, int, kColMajor)
    ->Name("SparseLA/SparseMatVec/DenseCase/ColMajor")
    ->ArgNames({"rows", "nnzPerRow", "rhsCols"})
    ->Args({51, 51, 1})
    ->Args({126, 126, 1})
    ->Args({501, 501, 1});

// Use a brick-mesh for the sparse matrix.
template <typename Scalar>
static void SparseMatVecProductCube(benchmark::State& state) {
  auto nx = static_cast<int>(state.range(0));
  auto ny = static_cast<int>(state.range(1));
  auto nz = static_cast<int>(state.range(2));
  auto numThreads = static_cast<int>(state.range(3));
  MOCHI_ASSERT(numThreads > 0, "Invalid number of threads.");
  if (numThreads > TaskScheduler::GetNumSupportedLogicalProcessors()) {
    MOCHI_SKIP_BENCHMARK(state);
    return;
  }
  auto A = test::MakeSparseMatrixWithBlockStructure<Scalar, 1>(nx, ny, nz);
  ColumnVector<Scalar> b(A.Rows()), c(A.Rows());
  b.SetRandom(1);
  TaskScheduler scheduler(numThreads - 1); // Caller not included in TaskScheduler's pool
  for (auto _ : state) {
    c = A * b;
    MOCHI_NO_DISCARD_IN_LOOP(c);
  }
  state.counters["FLOPs"] = benchmark::Counter(
      state.iterations() * (2 * A.NumNonZeros() - A.Rows()), benchmark::Counter::kIsRate);
  benchmark::DoNotOptimize(c);
}

// Strong scaling benchmarks.
BENCHMARK_TEMPLATE(SparseMatVecProductCube, float)
    ->Name("SparseLA/SparseMatVec/Cube")
    ->ArgNames({"nx", "ny", "nz", "threads"})
    ->ArgsProduct({{48}, {17}, {19}, {1, 2, 4, 8}})
    ->UseRealTime();

/****************************************************************************************
  "Block Sparse Matrix" x "Dense Matrix"
*/

// Benchmarks for performance vs. number of columns of the dense RHS matrix.
BENCHMARK_TEMPLATE(BlockSparseMatVecProduct, float, int, int, 3, kColMajor)
    ->Name("SparseLA/BlockSparseMatDenseMat/RhsColsSweep/Block3/ColMajor")
    ->ArgNames({"blockRows", "nnzBlocksPerRow", "rhsCols"})
    ->ArgsProduct({{5000}, {30}, {1, 4, 8}});

BENCHMARK_TEMPLATE(BlockSparseMatVecProduct, float, int, int, 3, kRowMajor)
    ->Name("SparseLA/BlockSparseMatDenseMat/RhsColsSweep/Block3/RowMajor")
    ->ArgNames({"blockRows", "nnzBlocksPerRow", "rhsCols"})
    ->ArgsProduct({{5000}, {30}, {1, 4, 8}});

/****************************************************************************************
  "Sparse Matrix" x "Dense Matrix"
*/

// Benchmarks for performance vs. number of columns of the dense RHS matrix.
BENCHMARK_TEMPLATE(BlockSparseMatVecProductAsSparse, float, int, int, 3, kColMajor)
    ->Name("SparseLA/BlockSparseAsSparseMatDenseMat/RhsColsSweep/Block3/ColMajor")
    ->ArgNames({"blockRows", "nnzBlocksPerRow", "rhsCols"})
    ->ArgsProduct({{5000}, {30}, {1, 4, 8}});

BENCHMARK_TEMPLATE(BlockSparseMatVecProductAsSparse, float, int, int, 3, kRowMajor)
    ->Name("SparseLA/BlockSparseAsSparseMatDenseMat/RhsColsSweep/Block3/RowMajor")
    ->ArgNames({"blockRows", "nnzBlocksPerRow", "rhsCols"})
    ->ArgsProduct({{5000}, {30}, {1, 4, 8}});

BENCHMARK_TEMPLATE(SparseMatVecProduct, float, int, int, kColMajor)
    ->Name("SparseLA/SparseMatDenseMat/RhsColsSweep/ColMajor")
    ->ArgNames({"rows", "nnzPerRow", "rhsCols"})
    ->ArgsProduct({{15000}, {90}, {1, 4, 8}});

BENCHMARK_TEMPLATE(SparseMatVecProduct, float, int, int, kRowMajor)
    ->Name("SparseLA/SparseMatDenseMat/RhsColsSweep/RowMajor")
    ->ArgNames({"rows", "nnzPerRow", "rhsCols"})
    ->ArgsProduct({{15000}, {90}, {1, 4, 8}});

/****************************************************************************************
  Transpose("Block Sparse Matrix") x "Vector"
*/

template <typename Scalar, typename CRIdx, typename Ptr, int kBlockSize, krylov::Direction kDir>
static void BlockSparseMatVecTransposeProduct(benchmark::State& state) {
  auto const numBlockRows = CRIdx(state.range(0)), numNonZeroBlocksPerRow = CRIdx(state.range(1));
  auto const numRhsCols = int(state.range(2));
  auto const A = test::MakeBlockSparseMatrixWithTargetNnz<Scalar, kBlockSize, CRIdx, Ptr>(
      numBlockRows, numNonZeroBlocksPerRow);
  auto const numBlockCols = A.BlockCols();
  Matrix<Scalar, kDynamic, kDynamic, kDir> b(kBlockSize * numBlockRows, numRhsCols),
      c(kBlockSize * numBlockCols, numRhsCols);
  b.SetRandom(1);
  for (auto x : state) {
    A.TransposeApply(b, c);
    MOCHI_NO_DISCARD_IN_LOOP(c);
  }
  state.counters["FLOPs"] = benchmark::Counter(
      state.iterations() * (2 * A.NumNonZeros() - A.Cols()) * numRhsCols,
      benchmark::Counter::kIsRate);
  benchmark::DoNotOptimize(c);
}

// Benchmarks for performance vs. number of non-zero blocks per row.
BENCHMARK_TEMPLATE(BlockSparseMatVecTransposeProduct, float, int, int, 3, kColMajor)
    ->Name("SparseLA/BlockSparseTransposeMatVec/NnzSweep/Block3/ColMajor")
    ->ArgNames({"blockRows", "nnzBlocksPerRow", "rhsCols"})
    ->ArgsProduct({{5000}, {1, 4, 7, 10, 13, 16, 100, 1000}, {1}});
BENCHMARK_TEMPLATE(BlockSparseMatVecTransposeProduct, float, int, int, 3, kRowMajor)
    ->Name("SparseLA/BlockSparseTransposeMatVec/NnzSweep/Block3/RowMajor")
    ->ArgNames({"blockRows", "nnzBlocksPerRow", "rhsCols"})
    ->ArgsProduct({{5000}, {1, 4, 7, 10, 13, 16, 100, 1000}, {1}});

/****************************************************************************************
  Transpose("Sparse Matrix") x "Vector"
*/

template <typename Scalar, typename CRIdx, typename Ptr, krylov::Direction kDir>
static void SparseMatVecTransposeProduct(benchmark::State& state) {
  auto const numRows = CRIdx(state.range(0)), numNonZerosPerRow = CRIdx(state.range(1));
  auto const numRhsCols = int(state.range(2));
  auto const A =
      test::MakeSparseMatrixWithTargetNnz<Scalar, CRIdx, Ptr>(numRows, numNonZerosPerRow);
  Matrix<Scalar, kDynamic, kDynamic, kDir> b(A.Rows(), numRhsCols), c(A.Cols(), numRhsCols);
  b.SetRandom(1);
  for (auto x : state) {
    A.TransposeApply(b, c);
    MOCHI_NO_DISCARD_IN_LOOP(c);
  }
  state.counters["FLOPs"] = benchmark::Counter(
      state.iterations() * (2 * A.NumNonZeros() - A.Cols()) * numRhsCols,
      benchmark::Counter::kIsRate);
  benchmark::DoNotOptimize(c);
}

// Benchmarks for performance vs. number of non-zeros per row.
BENCHMARK_TEMPLATE(SparseMatVecTransposeProduct, float, int, int, kColMajor)
    ->Name("SparseLA/SparseTransposeMatVec/NnzSweep/ColMajor")
    ->ArgNames({"rows", "nnzPerRow", "rhsCols"})
    ->ArgsProduct({{15000}, {3, 12, 21, 30, 39, 48, 300, 3000}, {1}});
BENCHMARK_TEMPLATE(SparseMatVecTransposeProduct, float, int, int, kRowMajor)
    ->Name("SparseLA/SparseTransposeMatVec/NnzSweep/RowMajor")
    ->ArgNames({"rows", "nnzPerRow", "rhsCols"})
    ->ArgsProduct({{15000}, {3, 12, 21, 30, 39, 48, 300, 3000}, {1}});

/****************************************************************************************
  "Block Sparse Matrix" x "Block Sparse Matrix" (itself)
*/

template <typename Scalar, int kBlockSize, typename CRIdx, typename Ptr>
static void BlockSparseMatMatProduct(benchmark::State& state) {
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
  auto gA = AsGraphView(A);
  auto gAA = Traverse(gA, gA).SortTargets();
  BlockSparseMatrix<Scalar, kBlockSize, CRIdx, Ptr> AA(A.BlockCols(), std::move(gAA));
  TaskScheduler scheduler(numThreads - 1); // Caller not included in TaskScheduler's pool
  for (auto _ : state) {
    details::SparseMatProduct(A, A, AA);
    MOCHI_NO_DISCARD_IN_LOOP(AA);
  }
  benchmark::DoNotOptimize(AA);
  //--- Compute the number of FLOPs
  double operations = 0;
  using Idx = std::remove_const_t<CRIdx>;
  auto const& B = A;
  for (Idx i = 0; i < A.BlockRows(); ++i) {
    auto ArowIndices = A.Indices(i);
    for (auto aColIdx : ArowIndices) {
      auto BrowIndices = B.Indices(aColIdx);
      operations += double((2 * kBlockSize - 1) * kBlockSize * kBlockSize) * BrowIndices.size();
    }
  }
  state.counters["FLOPs"] =
      benchmark::Counter(double(state.iterations()) * operations, benchmark::Counter::kIsRate);
}

// Benchmarks for performance on fixed grid with multiple threads
BENCHMARK_TEMPLATE(BlockSparseMatMatProduct, float, 3, int, int)
    ->Name("SparseLA/BlockSparseMatMat/Cube/Block3")
    ->ArgNames({"nx", "ny", "nz", "threads"})
    ->ArgsProduct({{16}, {17}, {19}, {1, 2, 4, 8}})
    ->UseRealTime();

/****************************************************************************************
  "Sparse Matrix" x "Sparse Matrix" (itself)
*/

template <typename Scalar, int kBlockSize, typename CRIdx, typename Ptr>
static void SparseMatMatProduct(benchmark::State& state) {
  auto nx = static_cast<int>(state.range(0));
  auto ny = static_cast<int>(state.range(1));
  auto nz = static_cast<int>(state.range(2));
  auto numThreads = static_cast<int>(state.range(3));
  MOCHI_ASSERT(numThreads > 0, "Invalid number of threads.");
  if (numThreads > TaskScheduler::GetNumSupportedLogicalProcessors()) {
    MOCHI_SKIP_BENCHMARK(state);
    return;
  }
  auto A = test::MakeSparseMatrixWithBlockStructure<Scalar, kBlockSize>(nx, ny, nz);
  auto gA = AsGraphView(A);
  auto gAA = Traverse(gA, gA).SortTargets();
  SparseMatrix<Scalar, CRIdx, Ptr> AA(A.Cols(), std::move(gAA));
  TaskScheduler scheduler(numThreads - 1); // Caller not included in TaskScheduler's pool
  for (auto _ : state) {
    details::SparseMatProduct(A, A, AA);
    MOCHI_NO_DISCARD_IN_LOOP(AA);
  }
  benchmark::DoNotOptimize(AA);
  //--- Compute the number of FLOPs
  double operations = 0;
  using Idx = std::remove_const_t<CRIdx>;
  auto const& B = A;
  for (Idx i = 0; i < A.Rows(); ++i) {
    auto ArowIndices = A.Indices(i);
    for (auto aColIdx : ArowIndices) {
      auto const BrowIndices = B.Indices(aColIdx);
      operations += 2.0 * BrowIndices.size();
    }
  }
  state.counters["FLOPs"] =
      benchmark::Counter(double(state.iterations()) * operations, benchmark::Counter::kIsRate);
}

// Benchmarks for performance on fixed grid with multiple threads
BENCHMARK_TEMPLATE(SparseMatMatProduct, float, 3, int, int)
    ->Name("SparseLA/SparseMatMat/Cube/Block3")
    ->ArgNames({"nx", "ny", "nz", "threads"})
    ->ArgsProduct({{16}, {17}, {19}, {1, 2, 4, 8}})
    ->UseRealTime();

/****************************************************************************************
  Construction of sparse LDLt factorization
*/

template <typename Scalar, int kMatBlockSize>
static void SparseLDLtConstructionFromSparseMatrix(benchmark::State& state) {
  int nx = state.range(0), ny = state.range(1), nz = state.range(2);
  auto A = mochi::test::MakeSparseMatrixWithBlockStructure<Scalar, kMatBlockSize>(nx, ny, nz);
  for (auto x : state) {
    int info = 0;
    krylov::SparseLDLt P(A, info);
    MOCHI_NO_DISCARD_IN_LOOP(P);
  }
}

BENCHMARK_TEMPLATE(SparseLDLtConstructionFromSparseMatrix, float, 3)
    ->Name("SparseLA/SparseLDLt/Construction/SparseMatrix")
    ->ArgNames({"nx", "ny", "nz"})
    ->Args({13, 17, 19});

/****************************************************************************************
  Resolution from sparse LDLt factorization
*/

template <typename Scalar>
static void SparseLDLtSolveFromSparseMatrix(benchmark::State& state) {
  int nx = int(state.range(0)), ny = int(state.range(1)), nz = int(state.range(2));
  constexpr int kBlockSize = 3; // used to represent elasticity-like problem
  auto A = mochi::test::MakeSparseMatrixWithBlockStructure<Scalar, kBlockSize>(nx, ny, nz);
  mochi::ColumnVector<Scalar> b(A.Rows()), c(A.Rows());
  c.SetRandom(1);
  int info = 0;
  krylov::SparseLDLt P(A, info);
  for (auto x : state) {
    b = c;
    P.LeftSolveInPlace(b);
    MOCHI_NO_DISCARD_IN_LOOP(b);
  }
  benchmark::DoNotOptimize(b);
}

BENCHMARK_TEMPLATE(SparseLDLtSolveFromSparseMatrix, float)
    ->Name("SparseLA/SparseLDLt/Solve/SparseMatrix")
    ->ArgNames({"nx", "ny", "nz"})
    ->Args({13, 17, 19});

/****************************************************************************************
  Construction of sparse LDLt factorization from a block sparse matrix
*/

template <typename Scalar, int kMatBlockSize>
static void SparseLDLtConstructionFromBlockSparseMatrix(benchmark::State& state) {
  int nx = state.range(0), ny = state.range(1), nz = state.range(2);
  auto A = mochi::test::MakeBlockSparseMatrix<Scalar, kMatBlockSize>(nx, ny, nz);
  for (auto x : state) {
    int info = 0;
    krylov::SparseLDLt P(A, info);
    MOCHI_NO_DISCARD_IN_LOOP(P);
  }
}

BENCHMARK_TEMPLATE(SparseLDLtConstructionFromBlockSparseMatrix, float, 3)
    ->Name("SparseLA/SparseLDLt/Construction/BlockSparseMatrix")
    ->ArgNames({"nx", "ny", "nz"})
    ->Args({13, 17, 19});

/****************************************************************************************
  Resolution from sparse LDLt factorization for a block sparse matrix
*/

template <typename Scalar>
static void SparseLDLtSolveFromBlockSparseMatrix(benchmark::State& state) {
  int nx = int(state.range(0)), ny = int(state.range(1)), nz = int(state.range(2));
  constexpr int kBlockSize = 3; // used to represent elasticity-like problem
  auto A = mochi::test::MakeBlockSparseMatrix<Scalar, kBlockSize>(nx, ny, nz);
  mochi::ColumnVector<Scalar> b(A.Rows()), c(A.Rows());
  c.SetRandom(1);
  int info = 0;
  krylov::SparseLDLt P(A, info);
  for (auto x : state) {
    b = c;
    P.LeftSolveInPlace(b);
    MOCHI_NO_DISCARD_IN_LOOP(b);
  }
  benchmark::DoNotOptimize(b);
}

BENCHMARK_TEMPLATE(SparseLDLtSolveFromBlockSparseMatrix, float)
    ->Name("SparseLA/SparseLDLt/Solve/BlockSparseMatrix")
    ->ArgNames({"nx", "ny", "nz"})
    ->Args({13, 17, 19});

/****************************************************************************
  Sort graph targets.
*/

static void SortGraphTargets(benchmark::State& state) {
  auto nx = static_cast<int>(state.range(0));
  auto ny = static_cast<int>(state.range(1));
  auto nz = static_cast<int>(state.range(2));
  auto numThreads = static_cast<int>(state.range(3));
  MOCHI_ASSERT(numThreads > 0, "Invalid number of threads.");
  if (numThreads > TaskScheduler::GetNumSupportedLogicalProcessors()) {
    MOCHI_SKIP_BENCHMARK(state);
    return;
  }
  TaskScheduler scheduler(numThreads - 1); // Caller not included in TaskScheduler's pool
  auto A = test::MakeBlockSparseMatrix<real, 3>(nx, ny, nz);
  auto gA = AsGraphView(A);
  auto gAA = Traverse(gA, gA);
  for (auto _ : state) {
    auto gAAtmp = std::move(gAA).SortTargets();
    std::swap(gAAtmp, gAA);
    MOCHI_NO_DISCARD_IN_LOOP(gAA);
  }
  benchmark::DoNotOptimize(gAA);
}

BENCHMARK(SortGraphTargets)
    ->Name("SparseLA/SortGraphTargets/Cube")
    ->ArgNames({"nx", "ny", "nz", "threads"})
    ->ArgsProduct({{16}, {17}, {19}, {1, 2, 4, 8}})
    ->UseRealTime();

} // namespace mochi_benchmark
