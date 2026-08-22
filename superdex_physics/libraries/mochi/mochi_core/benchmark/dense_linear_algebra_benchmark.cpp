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

#include <mochi_core/linear_algebra/ldlt.h>
#include <mochi_core/linear_algebra/lu.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/matrix_operations.h>
#include <mochi_core/linear_algebra/qr.h>

#if MOCHI_USE_EIGEN
#include <Eigen/Dense>
#endif // MOCHI_USE_EIGEN

using namespace mochi;

namespace mochi_benchmark {

constexpr auto kColMajor = krylov::Direction::ColMajor;
constexpr auto kRowMajor = krylov::Direction::RowMajor;
constexpr auto kDynamic = krylov::kDynamic;

/****************************************************************************************
  "Square matrix" x "square matrix"
*/

template <
    typename Scalar,
    int kM,
    krylov::Direction kDirC,
    krylov::Direction kDirA,
    krylov::Direction kDirB,
    bool kIsCompileTime>
static void MatMatProduct(benchmark::State& state) {
  constexpr auto kMAtCT = kIsCompileTime ? kM : kDynamic;
  Matrix<Scalar, kMAtCT, kMAtCT, kDirC> C(kM, kM);
  Matrix<Scalar, kMAtCT, kMAtCT, kDirA> A(kM, kM);
  Matrix<Scalar, kMAtCT, kMAtCT, kDirB> B(kM, kM);
  A.SetRandom(123);
  B.SetRandom(456);
  for (auto x : state) {
    C = A * B;
    MOCHI_NO_DISCARD_IN_LOOP(C);
  }
  state.counters["FLOPs"] =
      benchmark::Counter(state.iterations() * (2 * kM - 1) * kM * kM, benchmark::Counter::kIsRate);
  benchmark::DoNotOptimize(C);
}

// Compile-time-size matrices
BENCHMARK_TEMPLATE(MatMatProduct, float, 3, kColMajor, kColMajor, kColMajor, true)
    ->Name("DenseLA/MatMat/FixedSize/M3/OutCol/ACol/BCol");
BENCHMARK_TEMPLATE(MatMatProduct, float, 3, kRowMajor, kRowMajor, kRowMajor, true)
    ->Name("DenseLA/MatMat/FixedSize/M3/OutRow/ARow/BRow");
BENCHMARK_TEMPLATE(MatMatProduct, float, 3, kColMajor, kRowMajor, kColMajor, true)
    ->Name("DenseLA/MatMat/FixedSize/M3/OutCol/ARow/BCol");
BENCHMARK_TEMPLATE(MatMatProduct, float, 3, kRowMajor, kColMajor, kRowMajor, true)
    ->Name("DenseLA/MatMat/FixedSize/M3/OutRow/ACol/BRow");
BENCHMARK_TEMPLATE(MatMatProduct, float, 6, kColMajor, kColMajor, kColMajor, true)
    ->Name("DenseLA/MatMat/FixedSize/M6/OutCol/ACol/BCol");
BENCHMARK_TEMPLATE(MatMatProduct, float, 6, kRowMajor, kRowMajor, kRowMajor, true)
    ->Name("DenseLA/MatMat/FixedSize/M6/OutRow/ARow/BRow");
BENCHMARK_TEMPLATE(MatMatProduct, float, 6, kColMajor, kRowMajor, kColMajor, true)
    ->Name("DenseLA/MatMat/FixedSize/M6/OutCol/ARow/BCol");
BENCHMARK_TEMPLATE(MatMatProduct, float, 6, kRowMajor, kColMajor, kRowMajor, true)
    ->Name("DenseLA/MatMat/FixedSize/M6/OutRow/ACol/BRow");
BENCHMARK_TEMPLATE(MatMatProduct, float, 12, kColMajor, kColMajor, kColMajor, true)
    ->Name("DenseLA/MatMat/FixedSize/M12/OutCol/ACol/BCol");
BENCHMARK_TEMPLATE(MatMatProduct, float, 12, kRowMajor, kRowMajor, kRowMajor, true)
    ->Name("DenseLA/MatMat/FixedSize/M12/OutRow/ARow/BRow");
BENCHMARK_TEMPLATE(MatMatProduct, float, 12, kColMajor, kRowMajor, kColMajor, true)
    ->Name("DenseLA/MatMat/FixedSize/M12/OutCol/ARow/BCol");
BENCHMARK_TEMPLATE(MatMatProduct, float, 12, kRowMajor, kColMajor, kRowMajor, true)
    ->Name("DenseLA/MatMat/FixedSize/M12/OutRow/ACol/BRow");
BENCHMARK_TEMPLATE(MatMatProduct, float, 24, kColMajor, kColMajor, kColMajor, true)
    ->Name("DenseLA/MatMat/FixedSize/M24/OutCol/ACol/BCol");
BENCHMARK_TEMPLATE(MatMatProduct, float, 24, kRowMajor, kRowMajor, kRowMajor, true)
    ->Name("DenseLA/MatMat/FixedSize/M24/OutRow/ARow/BRow");
BENCHMARK_TEMPLATE(MatMatProduct, float, 24, kColMajor, kRowMajor, kColMajor, true)
    ->Name("DenseLA/MatMat/FixedSize/M24/OutCol/ARow/BCol");
BENCHMARK_TEMPLATE(MatMatProduct, float, 24, kRowMajor, kColMajor, kRowMajor, true)
    ->Name("DenseLA/MatMat/FixedSize/M24/OutRow/ACol/BRow");
BENCHMARK_TEMPLATE(MatMatProduct, float, 48, kColMajor, kColMajor, kColMajor, true)
    ->Name("DenseLA/MatMat/FixedSize/M48/OutCol/ACol/BCol");
BENCHMARK_TEMPLATE(MatMatProduct, float, 48, kRowMajor, kRowMajor, kRowMajor, true)
    ->Name("DenseLA/MatMat/FixedSize/M48/OutRow/ARow/BRow");
BENCHMARK_TEMPLATE(MatMatProduct, float, 48, kColMajor, kRowMajor, kColMajor, true)
    ->Name("DenseLA/MatMat/FixedSize/M48/OutCol/ARow/BCol");
BENCHMARK_TEMPLATE(MatMatProduct, float, 48, kRowMajor, kColMajor, kRowMajor, true)
    ->Name("DenseLA/MatMat/FixedSize/M48/OutRow/ACol/BRow");

// Runtime-size (aka dynamic) matrices
BENCHMARK_TEMPLATE(MatMatProduct, float, 50, kColMajor, kColMajor, kColMajor, false)
    ->Name("DenseLA/MatMat/DynamicSize/M50/OutCol/ACol/BCol");
BENCHMARK_TEMPLATE(MatMatProduct, float, 50, kRowMajor, kRowMajor, kRowMajor, false)
    ->Name("DenseLA/MatMat/DynamicSize/M50/OutRow/ARow/BRow");
BENCHMARK_TEMPLATE(MatMatProduct, float, 50, kColMajor, kRowMajor, kColMajor, false)
    ->Name("DenseLA/MatMat/DynamicSize/M50/OutCol/ARow/BCol");
BENCHMARK_TEMPLATE(MatMatProduct, float, 50, kRowMajor, kColMajor, kRowMajor, false)
    ->Name("DenseLA/MatMat/DynamicSize/M50/OutRow/ACol/BRow");
BENCHMARK_TEMPLATE(MatMatProduct, float, 125, kColMajor, kColMajor, kColMajor, false)
    ->Name("DenseLA/MatMat/DynamicSize/M125/OutCol/ACol/BCol");
BENCHMARK_TEMPLATE(MatMatProduct, float, 125, kRowMajor, kRowMajor, kRowMajor, false)
    ->Name("DenseLA/MatMat/DynamicSize/M125/OutRow/ARow/BRow");
BENCHMARK_TEMPLATE(MatMatProduct, float, 125, kColMajor, kRowMajor, kColMajor, false)
    ->Name("DenseLA/MatMat/DynamicSize/M125/OutCol/ARow/BCol");
BENCHMARK_TEMPLATE(MatMatProduct, float, 125, kRowMajor, kColMajor, kRowMajor, false)
    ->Name("DenseLA/MatMat/DynamicSize/M125/OutRow/ACol/BRow");
BENCHMARK_TEMPLATE(MatMatProduct, float, 500, kColMajor, kColMajor, kColMajor, false)
    ->Name("DenseLA/MatMat/DynamicSize/M500/OutCol/ACol/BCol");
BENCHMARK_TEMPLATE(MatMatProduct, float, 500, kRowMajor, kRowMajor, kRowMajor, false)
    ->Name("DenseLA/MatMat/DynamicSize/M500/OutRow/ARow/BRow");
BENCHMARK_TEMPLATE(MatMatProduct, float, 500, kColMajor, kRowMajor, kColMajor, false)
    ->Name("DenseLA/MatMat/DynamicSize/M500/OutCol/ARow/BCol");
BENCHMARK_TEMPLATE(MatMatProduct, float, 500, kRowMajor, kColMajor, kRowMajor, false)
    ->Name("DenseLA/MatMat/DynamicSize/M500/OutRow/ACol/BRow");
BENCHMARK_TEMPLATE(MatMatProduct, float, 500, kRowMajor, kColMajor, kColMajor, false)
    ->Name("DenseLA/MatMat/DynamicSize/M500/OutRow/ACol/BCol");
BENCHMARK_TEMPLATE(MatMatProduct, float, 500, kColMajor, kRowMajor, kRowMajor, false)
    ->Name("DenseLA/MatMat/DynamicSize/M500/OutCol/ARow/BRow");
BENCHMARK_TEMPLATE(MatMatProduct, float, 500, kRowMajor, kRowMajor, kColMajor, false)
    ->Name("DenseLA/MatMat/DynamicSize/M500/OutRow/ARow/BCol");
BENCHMARK_TEMPLATE(MatMatProduct, float, 500, kColMajor, kColMajor, kRowMajor, false)
    ->Name("DenseLA/MatMat/DynamicSize/M500/OutCol/ACol/BRow");

/****************************************************************************************
  "Square matrix" x "vector"
*/

template <typename Scalar, int kM, krylov::Direction kDir>
static void MatVecProduct(benchmark::State& state) {
  Matrix<Scalar, kDynamic, kDynamic, kDir> A(kM, kM);
  ColumnVector<Scalar, kDynamic> b(kM), c(kM);
  A.SetRandom(123);
  b.SetRandom(456);
  for (auto x : state) {
    c = A * b;
    MOCHI_NO_DISCARD_IN_LOOP(c);
  }
  state.counters["FLOPs"] =
      benchmark::Counter(state.iterations() * (2 * kM - 1) * kM, benchmark::Counter::kIsRate);
  benchmark::DoNotOptimize(c);
}

BENCHMARK_TEMPLATE(MatVecProduct, float, 50, kColMajor)->Name("DenseLA/MatVec/M50/ColMajor");
BENCHMARK_TEMPLATE(MatVecProduct, float, 50, kRowMajor)->Name("DenseLA/MatVec/M50/RowMajor");
BENCHMARK_TEMPLATE(MatVecProduct, float, 125, kColMajor)->Name("DenseLA/MatVec/M125/ColMajor");
BENCHMARK_TEMPLATE(MatVecProduct, float, 125, kRowMajor)->Name("DenseLA/MatVec/M125/RowMajor");
BENCHMARK_TEMPLATE(MatVecProduct, float, 500, kColMajor)->Name("DenseLA/MatVec/M500/ColMajor");
BENCHMARK_TEMPLATE(MatVecProduct, float, 500, kRowMajor)->Name("DenseLA/MatVec/M500/RowMajor");

/****************************************************************************************
  "Square matrix" x "tall matrix"
*/

template <
    typename Scalar,
    int kM,
    int kN,
    krylov::Direction kDirC,
    krylov::Direction kDirA,
    krylov::Direction kDirB>
static void MatTallMatProduct(benchmark::State& state) {
  Matrix<Scalar, kDynamic, kDynamic, kDirC> C(kM, kN);
  Matrix<Scalar, kDynamic, kDynamic, kDirA> A(kM, kM);
  Matrix<Scalar, kDynamic, kDynamic, kDirB> B(kM, kN);
  A.SetRandom(123);
  B.SetRandom(456);
  for (auto x : state) {
    C = A * B;
    MOCHI_NO_DISCARD_IN_LOOP(C);
  }
  state.counters["FLOPs"] =
      benchmark::Counter(state.iterations() * (2 * kM - 1) * kM * kN, benchmark::Counter::kIsRate);
  benchmark::DoNotOptimize(C);
}

BENCHMARK_TEMPLATE(MatTallMatProduct, float, 50, 2, kColMajor, kColMajor, kColMajor)
    ->Name("DenseLA/MatTallMat/M50/N2/OutCol/ACol/BCol");
BENCHMARK_TEMPLATE(MatTallMatProduct, float, 50, 2, kRowMajor, kRowMajor, kRowMajor)
    ->Name("DenseLA/MatTallMat/M50/N2/OutRow/ARow/BRow");
BENCHMARK_TEMPLATE(MatTallMatProduct, float, 50, 2, kColMajor, kRowMajor, kColMajor)
    ->Name("DenseLA/MatTallMat/M50/N2/OutCol/ARow/BCol");
BENCHMARK_TEMPLATE(MatTallMatProduct, float, 50, 2, kRowMajor, kColMajor, kRowMajor)
    ->Name("DenseLA/MatTallMat/M50/N2/OutRow/ACol/BRow");
BENCHMARK_TEMPLATE(MatTallMatProduct, float, 50, 5, kColMajor, kColMajor, kColMajor)
    ->Name("DenseLA/MatTallMat/M50/N5/OutCol/ACol/BCol");
BENCHMARK_TEMPLATE(MatTallMatProduct, float, 50, 5, kRowMajor, kRowMajor, kRowMajor)
    ->Name("DenseLA/MatTallMat/M50/N5/OutRow/ARow/BRow");
BENCHMARK_TEMPLATE(MatTallMatProduct, float, 50, 5, kColMajor, kRowMajor, kColMajor)
    ->Name("DenseLA/MatTallMat/M50/N5/OutCol/ARow/BCol");
BENCHMARK_TEMPLATE(MatTallMatProduct, float, 50, 5, kRowMajor, kColMajor, kRowMajor)
    ->Name("DenseLA/MatTallMat/M50/N5/OutRow/ACol/BRow");
BENCHMARK_TEMPLATE(MatTallMatProduct, float, 50, 10, kColMajor, kColMajor, kColMajor)
    ->Name("DenseLA/MatTallMat/M50/N10/OutCol/ACol/BCol");
BENCHMARK_TEMPLATE(MatTallMatProduct, float, 50, 10, kRowMajor, kRowMajor, kRowMajor)
    ->Name("DenseLA/MatTallMat/M50/N10/OutRow/ARow/BRow");
BENCHMARK_TEMPLATE(MatTallMatProduct, float, 50, 10, kColMajor, kRowMajor, kColMajor)
    ->Name("DenseLA/MatTallMat/M50/N10/OutCol/ARow/BCol");
BENCHMARK_TEMPLATE(MatTallMatProduct, float, 50, 10, kRowMajor, kColMajor, kRowMajor)
    ->Name("DenseLA/MatTallMat/M50/N10/OutRow/ACol/BRow");
BENCHMARK_TEMPLATE(MatTallMatProduct, float, 500, 2, kColMajor, kColMajor, kColMajor)
    ->Name("DenseLA/MatTallMat/M500/N2/OutCol/ACol/BCol");
BENCHMARK_TEMPLATE(MatTallMatProduct, float, 500, 2, kRowMajor, kRowMajor, kRowMajor)
    ->Name("DenseLA/MatTallMat/M500/N2/OutRow/ARow/BRow");
BENCHMARK_TEMPLATE(MatTallMatProduct, float, 500, 2, kColMajor, kRowMajor, kColMajor)
    ->Name("DenseLA/MatTallMat/M500/N2/OutCol/ARow/BCol");
BENCHMARK_TEMPLATE(MatTallMatProduct, float, 500, 2, kRowMajor, kColMajor, kRowMajor)
    ->Name("DenseLA/MatTallMat/M500/N2/OutRow/ACol/BRow");
BENCHMARK_TEMPLATE(MatTallMatProduct, float, 500, 5, kColMajor, kColMajor, kColMajor)
    ->Name("DenseLA/MatTallMat/M500/N5/OutCol/ACol/BCol");
BENCHMARK_TEMPLATE(MatTallMatProduct, float, 500, 5, kRowMajor, kRowMajor, kRowMajor)
    ->Name("DenseLA/MatTallMat/M500/N5/OutRow/ARow/BRow");
BENCHMARK_TEMPLATE(MatTallMatProduct, float, 500, 5, kColMajor, kRowMajor, kColMajor)
    ->Name("DenseLA/MatTallMat/M500/N5/OutCol/ARow/BCol");
BENCHMARK_TEMPLATE(MatTallMatProduct, float, 500, 5, kRowMajor, kColMajor, kRowMajor)
    ->Name("DenseLA/MatTallMat/M500/N5/OutRow/ACol/BRow");
BENCHMARK_TEMPLATE(MatTallMatProduct, float, 500, 10, kColMajor, kColMajor, kColMajor)
    ->Name("DenseLA/MatTallMat/M500/N10/OutCol/ACol/BCol");
BENCHMARK_TEMPLATE(MatTallMatProduct, float, 500, 10, kRowMajor, kRowMajor, kRowMajor)
    ->Name("DenseLA/MatTallMat/M500/N10/OutRow/ARow/BRow");
BENCHMARK_TEMPLATE(MatTallMatProduct, float, 500, 10, kColMajor, kRowMajor, kColMajor)
    ->Name("DenseLA/MatTallMat/M500/N10/OutCol/ARow/BCol");
BENCHMARK_TEMPLATE(MatTallMatProduct, float, 500, 10, kRowMajor, kColMajor, kRowMajor)
    ->Name("DenseLA/MatTallMat/M500/N10/OutRow/ACol/BRow");

/****************************************************************************************
  Complementary projection of a vector
*/

template <typename Scalar, int m, int k, krylov::Direction kDir>
static void ComplementaryProjection(benchmark::State& state) {
  static_assert(
      k <= m, "Subspace dimensionality must not be greater than full space dimensionality.");
  Matrix<Scalar, kDynamic, kDynamic, kDir> Q(m, k);
  ColumnVector<Scalar> b(m);
  Q.SetRandom(123);
  b.SetRandom(234);
  for (auto x : state) {
    b -= Q * (Transpose(Q) * b);
  }
  state.counters["FLOPs"] = benchmark::Counter(
      state.iterations() * ((2 * m - 1) * k + (2 * k - 1) * m + m), benchmark::Counter::kIsRate);
  benchmark::DoNotOptimize(b);
}

// Relevant benchmarks for deflated CG.
BENCHMARK_TEMPLATE(ComplementaryProjection, float, 100, 8, kColMajor)
    ->Name("DenseLA/ComplementaryProjection/M100/K8/ColMajor");
BENCHMARK_TEMPLATE(ComplementaryProjection, float, 100, 8, kRowMajor)
    ->Name("DenseLA/ComplementaryProjection/M100/K8/RowMajor");
BENCHMARK_TEMPLATE(ComplementaryProjection, float, 100, 16, kColMajor)
    ->Name("DenseLA/ComplementaryProjection/M100/K16/ColMajor");
BENCHMARK_TEMPLATE(ComplementaryProjection, float, 100, 16, kRowMajor)
    ->Name("DenseLA/ComplementaryProjection/M100/K16/RowMajor");
BENCHMARK_TEMPLATE(ComplementaryProjection, float, 1000, 8, kColMajor)
    ->Name("DenseLA/ComplementaryProjection/M1000/K8/ColMajor");
BENCHMARK_TEMPLATE(ComplementaryProjection, float, 1000, 8, kRowMajor)
    ->Name("DenseLA/ComplementaryProjection/M1000/K8/RowMajor");
BENCHMARK_TEMPLATE(ComplementaryProjection, float, 1000, 16, kColMajor)
    ->Name("DenseLA/ComplementaryProjection/M1000/K16/ColMajor");
BENCHMARK_TEMPLATE(ComplementaryProjection, float, 1000, 16, kRowMajor)
    ->Name("DenseLA/ComplementaryProjection/M1000/K16/RowMajor");
BENCHMARK_TEMPLATE(ComplementaryProjection, float, 10000, 8, kColMajor)
    ->Name("DenseLA/ComplementaryProjection/M10000/K8/ColMajor");
BENCHMARK_TEMPLATE(ComplementaryProjection, float, 10000, 8, kRowMajor)
    ->Name("DenseLA/ComplementaryProjection/M10000/K8/RowMajor");
BENCHMARK_TEMPLATE(ComplementaryProjection, float, 10000, 16, kColMajor)
    ->Name("DenseLA/ComplementaryProjection/M10000/K16/ColMajor");
BENCHMARK_TEMPLATE(ComplementaryProjection, float, 10000, 16, kRowMajor)
    ->Name("DenseLA/ComplementaryProjection/M10000/K16/RowMajor");

/****************************************************************************************
  LDLt factorization of symmetric matrix
*/

template <
    typename Scalar,
    int kM,
    krylov::Direction kDir,
    LDLtEquilibration kEquilibration,
    bool kIsCompileTime>
static void LDLtFactorization(benchmark::State& state) {
  constexpr auto kMAtCT = kIsCompileTime ? kM : kDynamic;
  Matrix<Scalar, kMAtCT, kMAtCT, kDir> A(kM, kM), A2(kM, kM);
  A.SetRandom(123);
  for (int i = 0; i < kM; ++i) {
    for (int j = 0; j < i; ++j) {
      A(i, j) = A(j, i);
    }
  }
  int info = 0;
  for (auto x : state) {
    // Reset the matrix to avoid factorization breakdown due to composing factorizations. The
    // assignment has minor impact on performance measurement.
    A2 = A;
    LDLt<Scalar, kMAtCT, kMAtCT, kEquilibration> ldlt(A2, info);
    MOCHI_NO_DISCARD_IN_LOOP(ldlt);
  }
  state.counters["FLOPs"] = benchmark::Counter(
      state.iterations() *
          (static_cast<int64_t>(kM) * kM * kM / 3 + 2 * kM * kM), // Dropping the linear term
      benchmark::Counter::kIsRate);
}

BENCHMARK_TEMPLATE(LDLtFactorization, float, 8, kColMajor, LDLtEquilibration::None, false)
    ->Name("DenseLA/LDLtFactorization/DynamicSize/M8/ColMajor/NoEquilibration");
BENCHMARK_TEMPLATE(LDLtFactorization, float, 8, kRowMajor, LDLtEquilibration::None, false)
    ->Name("DenseLA/LDLtFactorization/DynamicSize/M8/RowMajor/NoEquilibration");
BENCHMARK_TEMPLATE(LDLtFactorization, float, 16, kColMajor, LDLtEquilibration::None, false)
    ->Name("DenseLA/LDLtFactorization/DynamicSize/M16/ColMajor/NoEquilibration");
BENCHMARK_TEMPLATE(LDLtFactorization, float, 16, kRowMajor, LDLtEquilibration::None, false)
    ->Name("DenseLA/LDLtFactorization/DynamicSize/M16/RowMajor/NoEquilibration");
BENCHMARK_TEMPLATE(LDLtFactorization, float, 50, kColMajor, LDLtEquilibration::None, false)
    ->Name("DenseLA/LDLtFactorization/DynamicSize/M50/ColMajor/NoEquilibration");
BENCHMARK_TEMPLATE(LDLtFactorization, float, 50, kRowMajor, LDLtEquilibration::None, false)
    ->Name("DenseLA/LDLtFactorization/DynamicSize/M50/RowMajor/NoEquilibration");
BENCHMARK_TEMPLATE(LDLtFactorization, float, 125, kColMajor, LDLtEquilibration::None, false)
    ->Name("DenseLA/LDLtFactorization/DynamicSize/M125/ColMajor/NoEquilibration");
BENCHMARK_TEMPLATE(LDLtFactorization, float, 125, kRowMajor, LDLtEquilibration::None, false)
    ->Name("DenseLA/LDLtFactorization/DynamicSize/M125/RowMajor/NoEquilibration");
BENCHMARK_TEMPLATE(LDLtFactorization, float, 500, kColMajor, LDLtEquilibration::None, false)
    ->Name("DenseLA/LDLtFactorization/DynamicSize/M500/ColMajor/NoEquilibration");
BENCHMARK_TEMPLATE(LDLtFactorization, float, 500, kRowMajor, LDLtEquilibration::None, false)
    ->Name("DenseLA/LDLtFactorization/DynamicSize/M500/RowMajor/NoEquilibration");

BENCHMARK_TEMPLATE(LDLtFactorization, float, 8, kColMajor, LDLtEquilibration::Diagonal, false)
    ->Name("DenseLA/LDLtFactorization/DynamicSize/M8/ColMajor/DiagonalEquilibration");
BENCHMARK_TEMPLATE(LDLtFactorization, float, 8, kRowMajor, LDLtEquilibration::Diagonal, false)
    ->Name("DenseLA/LDLtFactorization/DynamicSize/M8/RowMajor/DiagonalEquilibration");
BENCHMARK_TEMPLATE(LDLtFactorization, float, 16, kColMajor, LDLtEquilibration::Diagonal, false)
    ->Name("DenseLA/LDLtFactorization/DynamicSize/M16/ColMajor/DiagonalEquilibration");
BENCHMARK_TEMPLATE(LDLtFactorization, float, 16, kRowMajor, LDLtEquilibration::Diagonal, false)
    ->Name("DenseLA/LDLtFactorization/DynamicSize/M16/RowMajor/DiagonalEquilibration");
BENCHMARK_TEMPLATE(LDLtFactorization, float, 50, kColMajor, LDLtEquilibration::Diagonal, false)
    ->Name("DenseLA/LDLtFactorization/DynamicSize/M50/ColMajor/DiagonalEquilibration");
BENCHMARK_TEMPLATE(LDLtFactorization, float, 50, kRowMajor, LDLtEquilibration::Diagonal, false)
    ->Name("DenseLA/LDLtFactorization/DynamicSize/M50/RowMajor/DiagonalEquilibration");
BENCHMARK_TEMPLATE(LDLtFactorization, float, 125, kColMajor, LDLtEquilibration::Diagonal, false)
    ->Name("DenseLA/LDLtFactorization/DynamicSize/M125/ColMajor/DiagonalEquilibration");
BENCHMARK_TEMPLATE(LDLtFactorization, float, 125, kRowMajor, LDLtEquilibration::Diagonal, false)
    ->Name("DenseLA/LDLtFactorization/DynamicSize/M125/RowMajor/DiagonalEquilibration");
BENCHMARK_TEMPLATE(LDLtFactorization, float, 500, kColMajor, LDLtEquilibration::Diagonal, false)
    ->Name("DenseLA/LDLtFactorization/DynamicSize/M500/ColMajor/DiagonalEquilibration");
BENCHMARK_TEMPLATE(LDLtFactorization, float, 500, kRowMajor, LDLtEquilibration::Diagonal, false)
    ->Name("DenseLA/LDLtFactorization/DynamicSize/M500/RowMajor/DiagonalEquilibration");

/****************************************************************************************
  LU factorization of non-symmetric matrix
*/

template <typename Scalar, int kM, krylov::Direction kDir, PermuteAlg kAlg, bool kIsCompileTime>
static void LUFactorization(benchmark::State& state) {
  constexpr auto kMAtCT = kIsCompileTime ? kM : kDynamic;
  Matrix<Scalar, kMAtCT, kMAtCT, kDir> A(kM, kM), A2(kM, kM);
  A.SetRandom(123);
  for (auto x : state) {
    // Reset the matrix to avoid factorization breakdown due to composing factorizations. The
    // assignment has minor impact on performance measurement.
    A2 = A;
    LU<Scalar, kMAtCT, kMAtCT, kAlg> LU(A2);
    MOCHI_NO_DISCARD_IN_LOOP(LU);
  }
  state.counters["FLOPs"] = benchmark::Counter(
      state.iterations() *
          (static_cast<int64_t>(2 * kM) * kM * kM / 3 + 4 * kM * kM), // Dropping the linear term
      benchmark::Counter::kIsRate);
}

BENCHMARK_TEMPLATE(LUFactorization, float, 8, kColMajor, PermuteAlg::None, false)
    ->Name("DenseLA/LUFactorization/DynamicSize/M8/ColMajor/NoPivot");
BENCHMARK_TEMPLATE(LUFactorization, float, 8, kRowMajor, PermuteAlg::None, false)
    ->Name("DenseLA/LUFactorization/DynamicSize/M8/RowMajor/NoPivot");
BENCHMARK_TEMPLATE(LUFactorization, float, 16, kColMajor, PermuteAlg::None, false)
    ->Name("DenseLA/LUFactorization/DynamicSize/M16/ColMajor/NoPivot");
BENCHMARK_TEMPLATE(LUFactorization, float, 16, kRowMajor, PermuteAlg::None, false)
    ->Name("DenseLA/LUFactorization/DynamicSize/M16/RowMajor/NoPivot");
BENCHMARK_TEMPLATE(LUFactorization, float, 50, kColMajor, PermuteAlg::None, false)
    ->Name("DenseLA/LUFactorization/DynamicSize/M50/ColMajor/NoPivot");
BENCHMARK_TEMPLATE(LUFactorization, float, 50, kRowMajor, PermuteAlg::None, false)
    ->Name("DenseLA/LUFactorization/DynamicSize/M50/RowMajor/NoPivot");
BENCHMARK_TEMPLATE(LUFactorization, float, 125, kColMajor, PermuteAlg::None, false)
    ->Name("DenseLA/LUFactorization/DynamicSize/M125/ColMajor/NoPivot");
BENCHMARK_TEMPLATE(LUFactorization, float, 125, kRowMajor, PermuteAlg::None, false)
    ->Name("DenseLA/LUFactorization/DynamicSize/M125/RowMajor/NoPivot");
BENCHMARK_TEMPLATE(LUFactorization, float, 500, kColMajor, PermuteAlg::None, false)
    ->Name("DenseLA/LUFactorization/DynamicSize/M500/ColMajor/NoPivot");
BENCHMARK_TEMPLATE(LUFactorization, float, 500, kRowMajor, PermuteAlg::None, false)
    ->Name("DenseLA/LUFactorization/DynamicSize/M500/RowMajor/NoPivot");

BENCHMARK_TEMPLATE(LUFactorization, float, 8, kColMajor, PermuteAlg::PartialRow, false)
    ->Name("DenseLA/LUFactorization/DynamicSize/M8/ColMajor/PartialRowPivot");
BENCHMARK_TEMPLATE(LUFactorization, float, 8, kRowMajor, PermuteAlg::PartialRow, false)
    ->Name("DenseLA/LUFactorization/DynamicSize/M8/RowMajor/PartialRowPivot");
BENCHMARK_TEMPLATE(LUFactorization, float, 16, kColMajor, PermuteAlg::PartialRow, false)
    ->Name("DenseLA/LUFactorization/DynamicSize/M16/ColMajor/PartialRowPivot");
BENCHMARK_TEMPLATE(LUFactorization, float, 16, kRowMajor, PermuteAlg::PartialRow, false)
    ->Name("DenseLA/LUFactorization/DynamicSize/M16/RowMajor/PartialRowPivot");
BENCHMARK_TEMPLATE(LUFactorization, float, 50, kColMajor, PermuteAlg::PartialRow, false)
    ->Name("DenseLA/LUFactorization/DynamicSize/M50/ColMajor/PartialRowPivot");
BENCHMARK_TEMPLATE(LUFactorization, float, 50, kRowMajor, PermuteAlg::PartialRow, false)
    ->Name("DenseLA/LUFactorization/DynamicSize/M50/RowMajor/PartialRowPivot");
BENCHMARK_TEMPLATE(LUFactorization, float, 125, kColMajor, PermuteAlg::PartialRow, false)
    ->Name("DenseLA/LUFactorization/DynamicSize/M125/ColMajor/PartialRowPivot");
BENCHMARK_TEMPLATE(LUFactorization, float, 125, kRowMajor, PermuteAlg::PartialRow, false)
    ->Name("DenseLA/LUFactorization/DynamicSize/M125/RowMajor/PartialRowPivot");
BENCHMARK_TEMPLATE(LUFactorization, float, 500, kColMajor, PermuteAlg::PartialRow, false)
    ->Name("DenseLA/LUFactorization/DynamicSize/M500/ColMajor/PartialRowPivot");
BENCHMARK_TEMPLATE(LUFactorization, float, 500, kRowMajor, PermuteAlg::PartialRow, false)
    ->Name("DenseLA/LUFactorization/DynamicSize/M500/RowMajor/PartialRowPivot");

BENCHMARK_TEMPLATE(LUFactorization, float, 8, kColMajor, PermuteAlg::Rook, false)
    ->Name("DenseLA/LUFactorization/DynamicSize/M8/ColMajor/RookPivot");
BENCHMARK_TEMPLATE(LUFactorization, float, 8, kRowMajor, PermuteAlg::Rook, false)
    ->Name("DenseLA/LUFactorization/DynamicSize/M8/RowMajor/RookPivot");
BENCHMARK_TEMPLATE(LUFactorization, float, 16, kColMajor, PermuteAlg::Rook, false)
    ->Name("DenseLA/LUFactorization/DynamicSize/M16/ColMajor/RookPivot");
BENCHMARK_TEMPLATE(LUFactorization, float, 16, kRowMajor, PermuteAlg::Rook, false)
    ->Name("DenseLA/LUFactorization/DynamicSize/M16/RowMajor/RookPivot");
BENCHMARK_TEMPLATE(LUFactorization, float, 50, kColMajor, PermuteAlg::Rook, false)
    ->Name("DenseLA/LUFactorization/DynamicSize/M50/ColMajor/RookPivot");
BENCHMARK_TEMPLATE(LUFactorization, float, 50, kRowMajor, PermuteAlg::Rook, false)
    ->Name("DenseLA/LUFactorization/DynamicSize/M50/RowMajor/RookPivot");
BENCHMARK_TEMPLATE(LUFactorization, float, 125, kColMajor, PermuteAlg::Rook, false)
    ->Name("DenseLA/LUFactorization/DynamicSize/M125/ColMajor/RookPivot");
BENCHMARK_TEMPLATE(LUFactorization, float, 125, kRowMajor, PermuteAlg::Rook, false)
    ->Name("DenseLA/LUFactorization/DynamicSize/M125/RowMajor/RookPivot");
BENCHMARK_TEMPLATE(LUFactorization, float, 500, kColMajor, PermuteAlg::Rook, false)
    ->Name("DenseLA/LUFactorization/DynamicSize/M500/ColMajor/RookPivot");
BENCHMARK_TEMPLATE(LUFactorization, float, 500, kRowMajor, PermuteAlg::Rook, false)
    ->Name("DenseLA/LUFactorization/DynamicSize/M500/RowMajor/RookPivot");

/****************************************************************************************
  LDLt solve in place
*/

enum class SolveSide { Left, Right };

template <
    typename Scalar,
    int kM,
    bool kIsCompileTime,
    LDLtEquilibration kEquilibration,
    SolveSide kSide>
static void LDLtSolveInPlace(benchmark::State& state) {
  constexpr auto kMAtCT = kIsCompileTime ? kM : kDynamic;
  Matrix<Scalar, kMAtCT, kMAtCT> A(kM, kM);
  ColumnVector<Scalar, kMAtCT> b(kM);
  A.SetRandom(123);
  b.SetRandom(456);
  for (int i = 0; i < kM; ++i) {
    for (int j = 0; j < i; ++j) {
      A(i, j) = A(j, i);
    }
  }
  int info = 0;
  LDLt<Scalar, kMAtCT, kMAtCT, kEquilibration> ldlt(A, info);
  for (auto x : state) {
    if constexpr (kSide == SolveSide::Left) {
      ldlt.LeftSolveInPlace(b);
      MOCHI_NO_DISCARD_IN_LOOP(b);
    } else {
      static_assert(kSide == SolveSide::Right);
      auto bT = b.Transpose(); // View creation has minor impact on performance measurement.
      ldlt.RightSolveInPlace(bT);
      MOCHI_NO_DISCARD_IN_LOOP(bT);
    }
  }
  state.counters["FLOPs"] =
      benchmark::Counter(state.iterations() * (2 * kM - 1) * kM, benchmark::Counter::kIsRate);
  benchmark::DoNotOptimize(b);
}

// clang-format off
BENCHMARK_TEMPLATE(LDLtSolveInPlace, float, 8, false, LDLtEquilibration::None, SolveSide::Left)->Name("DenseLA/LDLtSolveInPlace/DynamicSize/M8/NoEquilibration/Left");
BENCHMARK_TEMPLATE(LDLtSolveInPlace, float, 8, false, LDLtEquilibration::None, SolveSide::Right)->Name("DenseLA/LDLtSolveInPlace/DynamicSize/M8/NoEquilibration/Right");
BENCHMARK_TEMPLATE(LDLtSolveInPlace, float, 16, false, LDLtEquilibration::None, SolveSide::Left)->Name("DenseLA/LDLtSolveInPlace/DynamicSize/M16/NoEquilibration/Left");
BENCHMARK_TEMPLATE(LDLtSolveInPlace, float, 16, false, LDLtEquilibration::None, SolveSide::Right)->Name("DenseLA/LDLtSolveInPlace/DynamicSize/M16/NoEquilibration/Right");
BENCHMARK_TEMPLATE(LDLtSolveInPlace, float, 50, false, LDLtEquilibration::None, SolveSide::Left)->Name("DenseLA/LDLtSolveInPlace/DynamicSize/M50/NoEquilibration/Left");
BENCHMARK_TEMPLATE(LDLtSolveInPlace, float, 50, false, LDLtEquilibration::None, SolveSide::Right)->Name("DenseLA/LDLtSolveInPlace/DynamicSize/M50/NoEquilibration/Right");
BENCHMARK_TEMPLATE(LDLtSolveInPlace, float, 125, false, LDLtEquilibration::None, SolveSide::Left)->Name("DenseLA/LDLtSolveInPlace/DynamicSize/M125/NoEquilibration/Left");
BENCHMARK_TEMPLATE(LDLtSolveInPlace, float, 125, false, LDLtEquilibration::None, SolveSide::Right)->Name("DenseLA/LDLtSolveInPlace/DynamicSize/M125/NoEquilibration/Right");
BENCHMARK_TEMPLATE(LDLtSolveInPlace, float, 500, false, LDLtEquilibration::None, SolveSide::Left)->Name("DenseLA/LDLtSolveInPlace/DynamicSize/M500/NoEquilibration/Left");
BENCHMARK_TEMPLATE(LDLtSolveInPlace, float, 500, false, LDLtEquilibration::None, SolveSide::Right)->Name("DenseLA/LDLtSolveInPlace/DynamicSize/M500/NoEquilibration/Right");

BENCHMARK_TEMPLATE(LDLtSolveInPlace, float, 8, false, LDLtEquilibration::Diagonal, SolveSide::Left)->Name("DenseLA/LDLtSolveInPlace/DynamicSize/M8/DiagonalEquilibration/Left");
BENCHMARK_TEMPLATE(LDLtSolveInPlace, float, 16, false, LDLtEquilibration::Diagonal, SolveSide::Left)->Name("DenseLA/LDLtSolveInPlace/DynamicSize/M16/DiagonalEquilibration/Left");
BENCHMARK_TEMPLATE(LDLtSolveInPlace, float, 50, false, LDLtEquilibration::Diagonal, SolveSide::Left)->Name("DenseLA/LDLtSolveInPlace/DynamicSize/M50/DiagonalEquilibration/Left");
BENCHMARK_TEMPLATE(LDLtSolveInPlace, float, 125, false, LDLtEquilibration::Diagonal, SolveSide::Left)->Name("DenseLA/LDLtSolveInPlace/DynamicSize/M125/DiagonalEquilibration/Left");
BENCHMARK_TEMPLATE(LDLtSolveInPlace, float, 500, false, LDLtEquilibration::Diagonal, SolveSide::Left)->Name("DenseLA/LDLtSolveInPlace/DynamicSize/M500/DiagonalEquilibration/Left");
// clang-format on

/****************************************************************************************
  LU solve in place
*/
template <typename Scalar, int kM, PermuteAlg kAlg, bool kIsCompileTime, SolveSide kSide>
static void LUSolveInPlace(benchmark::State& state) {
  constexpr auto kMAtCT = kIsCompileTime ? kM : kDynamic;
  Matrix<Scalar, kMAtCT, kMAtCT> A(kM, kM);
  ColumnVector<Scalar, kMAtCT> b(kM);
  A.SetRandom(123);
  b.SetRandom(456);
  LU<Scalar, kMAtCT, kMAtCT, kAlg> lu(A);
  for (auto x : state) {
    if constexpr (kSide == SolveSide::Left) {
      lu.LeftSolveInPlace(b);
      MOCHI_NO_DISCARD_IN_LOOP(b);
    } else {
      static_assert(kSide == SolveSide::Right);
      auto bT = b.Transpose(); // View creation has minor impact on performance measurement.
      lu.RightSolveInPlace(bT);
      MOCHI_NO_DISCARD_IN_LOOP(bT);
    }
  }
  state.counters["FLOPs"] =
      benchmark::Counter(state.iterations() * (2 * kM - 1) * kM, benchmark::Counter::kIsRate);
  benchmark::DoNotOptimize(b);
}

// clang-format off
BENCHMARK_TEMPLATE(LUSolveInPlace, float, 8, PermuteAlg::None, false, SolveSide::Left)->Name("DenseLA/LUSolveInPlace/DynamicSize/M8/NoPivot/Left");
BENCHMARK_TEMPLATE(LUSolveInPlace, float, 8, PermuteAlg::None, false, SolveSide::Right)->Name("DenseLA/LUSolveInPlace/DynamicSize/M8/NoPivot/Right");
BENCHMARK_TEMPLATE(LUSolveInPlace, float, 16, PermuteAlg::None, false, SolveSide::Left)->Name("DenseLA/LUSolveInPlace/DynamicSize/M16/NoPivot/Left");
BENCHMARK_TEMPLATE(LUSolveInPlace, float, 16, PermuteAlg::None, false, SolveSide::Right)->Name("DenseLA/LUSolveInPlace/DynamicSize/M16/NoPivot/Right");
BENCHMARK_TEMPLATE(LUSolveInPlace, float, 50, PermuteAlg::None, false, SolveSide::Left)->Name("DenseLA/LUSolveInPlace/DynamicSize/M50/NoPivot/Left");
BENCHMARK_TEMPLATE(LUSolveInPlace, float, 50, PermuteAlg::None, false, SolveSide::Right)->Name("DenseLA/LUSolveInPlace/DynamicSize/M50/NoPivot/Right");
BENCHMARK_TEMPLATE(LUSolveInPlace, float, 125, PermuteAlg::None, false, SolveSide::Left)->Name("DenseLA/LUSolveInPlace/DynamicSize/M125/NoPivot/Left");
BENCHMARK_TEMPLATE(LUSolveInPlace, float, 125, PermuteAlg::None, false, SolveSide::Right)->Name("DenseLA/LUSolveInPlace/DynamicSize/M125/NoPivot/Right");
BENCHMARK_TEMPLATE(LUSolveInPlace, float, 500, PermuteAlg::None, false, SolveSide::Left)->Name("DenseLA/LUSolveInPlace/DynamicSize/M500/NoPivot/Left");
BENCHMARK_TEMPLATE(LUSolveInPlace, float, 500, PermuteAlg::None, false, SolveSide::Right)->Name("DenseLA/LUSolveInPlace/DynamicSize/M500/NoPivot/Right");

BENCHMARK_TEMPLATE(LUSolveInPlace, float, 8, PermuteAlg::PartialRow, false, SolveSide::Left)->Name("DenseLA/LUSolveInPlace/DynamicSize/M8/PartialRowPivot/Left");
BENCHMARK_TEMPLATE(LUSolveInPlace, float, 16, PermuteAlg::PartialRow, false, SolveSide::Left)->Name("DenseLA/LUSolveInPlace/DynamicSize/M16/PartialRowPivot/Left");
BENCHMARK_TEMPLATE(LUSolveInPlace, float, 50, PermuteAlg::PartialRow, false, SolveSide::Left)->Name("DenseLA/LUSolveInPlace/DynamicSize/M50/PartialRowPivot/Left");
BENCHMARK_TEMPLATE(LUSolveInPlace, float, 125, PermuteAlg::PartialRow, false, SolveSide::Left)->Name("DenseLA/LUSolveInPlace/DynamicSize/M125/PartialRowPivot/Left");
BENCHMARK_TEMPLATE(LUSolveInPlace, float, 500, PermuteAlg::PartialRow, false, SolveSide::Left)->Name("DenseLA/LUSolveInPlace/DynamicSize/M500/PartialRowPivot/Left");

BENCHMARK_TEMPLATE(LUSolveInPlace, float, 8, PermuteAlg::Rook, false, SolveSide::Left)->Name("DenseLA/LUSolveInPlace/DynamicSize/M8/RookPivot/Left");
BENCHMARK_TEMPLATE(LUSolveInPlace, float, 16, PermuteAlg::Rook, false, SolveSide::Left)->Name("DenseLA/LUSolveInPlace/DynamicSize/M16/RookPivot/Left");
BENCHMARK_TEMPLATE(LUSolveInPlace, float, 50, PermuteAlg::Rook, false, SolveSide::Left)->Name("DenseLA/LUSolveInPlace/DynamicSize/M50/RookPivot/Left");
BENCHMARK_TEMPLATE(LUSolveInPlace, float, 125, PermuteAlg::Rook, false, SolveSide::Left)->Name("DenseLA/LUSolveInPlace/DynamicSize/M125/RookPivot/Left");
BENCHMARK_TEMPLATE(LUSolveInPlace, float, 500, PermuteAlg::Rook, false, SolveSide::Left)->Name("DenseLA/LUSolveInPlace/DynamicSize/M500/RookPivot/Left");
// clang-format on

/****************************************************************************************
  Matrix inverse
*/

template <typename Scalar, int kM, krylov::Direction kDir, bool kIsCompileTime>
static void Inverse(benchmark::State& state) {
  constexpr auto kMAtCT = kIsCompileTime ? kM : kDynamic;
  Matrix<Scalar, kMAtCT, kMAtCT, kDir> A(kM, kM), Ainv(kM, kM);
  A.SetRandom(123);
  for (auto x : state) {
    Ainv = Inverse(A);
    MOCHI_NO_DISCARD_IN_LOOP(Ainv);
  }
  // LU factorization (dropping the linear term): 2 * m^3 / 3 + 4 * m^2 FLOPs.
  // Solve (dropping the linear term): (m^3 / 3 + m^2 / 2) + (m^3) = 4 * m^3 / 3 + m^2 / 2 FLOPs.
  state.counters["FLOPs"] = benchmark::Counter(
      state.iterations() * (static_cast<int64_t>(2) * kM * kM * kM + 9 * kM * kM / 2),
      benchmark::Counter::kIsRate);
  benchmark::DoNotOptimize(Ainv);
}

// Relevant benchmarks for block Jacobi preconditioner.
BENCHMARK_TEMPLATE(Inverse, float, 3, kColMajor, true)
    ->Name("DenseLA/Inverse/FixedSize/M3/ColMajor");
BENCHMARK_TEMPLATE(Inverse, float, 3, kRowMajor, true)
    ->Name("DenseLA/Inverse/FixedSize/M3/RowMajor");

// Other benchmarks.
BENCHMARK_TEMPLATE(Inverse, float, 8, kColMajor, false)
    ->Name("DenseLA/Inverse/DynamicSize/M8/ColMajor");
BENCHMARK_TEMPLATE(Inverse, float, 8, kRowMajor, false)
    ->Name("DenseLA/Inverse/DynamicSize/M8/RowMajor");
BENCHMARK_TEMPLATE(Inverse, float, 16, kColMajor, false)
    ->Name("DenseLA/Inverse/DynamicSize/M16/ColMajor");
BENCHMARK_TEMPLATE(Inverse, float, 16, kRowMajor, false)
    ->Name("DenseLA/Inverse/DynamicSize/M16/RowMajor");
BENCHMARK_TEMPLATE(Inverse, float, 50, kColMajor, false)
    ->Name("DenseLA/Inverse/DynamicSize/M50/ColMajor");
BENCHMARK_TEMPLATE(Inverse, float, 50, kRowMajor, false)
    ->Name("DenseLA/Inverse/DynamicSize/M50/RowMajor");
BENCHMARK_TEMPLATE(Inverse, float, 125, kColMajor, false)
    ->Name("DenseLA/Inverse/DynamicSize/M125/ColMajor");
BENCHMARK_TEMPLATE(Inverse, float, 125, kRowMajor, false)
    ->Name("DenseLA/Inverse/DynamicSize/M125/RowMajor");
BENCHMARK_TEMPLATE(Inverse, float, 500, kColMajor, false)
    ->Name("DenseLA/Inverse/DynamicSize/M500/ColMajor");
BENCHMARK_TEMPLATE(Inverse, float, 500, kRowMajor, false)
    ->Name("DenseLA/Inverse/DynamicSize/M500/RowMajor");

/****************************************************************************************
  Symmetric matrix inverse
*/

template <typename Scalar, int kM, krylov::Direction kDir, bool kIsCompileTime>
static void SymInverse(benchmark::State& state) {
  constexpr auto kMAtCT = kIsCompileTime ? kM : kDynamic;
  Matrix<Scalar, kMAtCT, kMAtCT, kDir> A(kM, kM), Ainv(kM, kM);
  A.SetRandom(123);
  for (int i = 0; i < kM; ++i) {
    for (int j = 0; j < i; ++j) {
      A(i, j) = A(j, i);
    }
  }
  for (auto x : state) {
    Ainv = SymInverse(A);
    MOCHI_NO_DISCARD_IN_LOOP(Ainv);
  }
  // LDLt factorization (dropping the linear term): m^3 / 3 + 2 * m^2 FLOPs.
  // Solve (exploiting symmetry, dropping the linear term): 2 * (m^3 / 3 + m^2 / 2) + m^2 / 2 FLOPs.
  // The m^2 / 2 term refers to copying the lower triangular part to the upper triangular part (or
  // vice versa) and are not actual FLOPs.
  state.counters["FLOPs"] = benchmark::Counter(
      state.iterations() * (static_cast<int64_t>(1) * kM * kM * kM + 7 * kM * kM / 2),
      benchmark::Counter::kIsRate);
  benchmark::DoNotOptimize(Ainv);
}

// Relevant benchmarks for block Jacobi preconditioner.
BENCHMARK_TEMPLATE(SymInverse, float, 3, kColMajor, true)
    ->Name("DenseLA/SymInverse/FixedSize/M3/ColMajor");
BENCHMARK_TEMPLATE(SymInverse, float, 3, kRowMajor, true)
    ->Name("DenseLA/SymInverse/FixedSize/M3/RowMajor");

// Other benchmarks.
BENCHMARK_TEMPLATE(SymInverse, float, 8, kColMajor, false)
    ->Name("DenseLA/SymInverse/DynamicSize/M8/ColMajor");
BENCHMARK_TEMPLATE(SymInverse, float, 8, kRowMajor, false)
    ->Name("DenseLA/SymInverse/DynamicSize/M8/RowMajor");
BENCHMARK_TEMPLATE(SymInverse, float, 16, kColMajor, false)
    ->Name("DenseLA/SymInverse/DynamicSize/M16/ColMajor");
BENCHMARK_TEMPLATE(SymInverse, float, 16, kRowMajor, false)
    ->Name("DenseLA/SymInverse/DynamicSize/M16/RowMajor");
BENCHMARK_TEMPLATE(SymInverse, float, 50, kColMajor, false)
    ->Name("DenseLA/SymInverse/DynamicSize/M50/ColMajor");
BENCHMARK_TEMPLATE(SymInverse, float, 50, kRowMajor, false)
    ->Name("DenseLA/SymInverse/DynamicSize/M50/RowMajor");
BENCHMARK_TEMPLATE(SymInverse, float, 125, kColMajor, false)
    ->Name("DenseLA/SymInverse/DynamicSize/M125/ColMajor");
BENCHMARK_TEMPLATE(SymInverse, float, 125, kRowMajor, false)
    ->Name("DenseLA/SymInverse/DynamicSize/M125/RowMajor");
BENCHMARK_TEMPLATE(SymInverse, float, 500, kColMajor, false)
    ->Name("DenseLA/SymInverse/DynamicSize/M500/ColMajor");
BENCHMARK_TEMPLATE(SymInverse, float, 500, kRowMajor, false)
    ->Name("DenseLA/SymInverse/DynamicSize/M500/RowMajor");

/****************************************************************************************
  QR factorization
*/

template <
    typename Scalar,
    int kM,
    int kN,
    krylov::Direction kDir,
    bool kIsCompileTime,
    QrAlgorithm kAlgorithm>
static void QRFactorization(benchmark::State& state) {
  constexpr auto kMAtCT = kIsCompileTime ? kM : kDynamic;
  constexpr auto kNAtCT = kIsCompileTime ? kN : kDynamic;
  Matrix<Scalar, kMAtCT, kNAtCT, kDir> A(kM, kN);
  A.SetRandom(123);
  for (auto _ : state) {
    ThinQR<Scalar, kMAtCT, kNAtCT, kAlgorithm> qr(A);
    MOCHI_NO_DISCARD_IN_LOOP(qr);
  }
  state.counters["FLOPs"] = benchmark::Counter(
      state.iterations() *
          (static_cast<int64_t>(2) * kM * kN * kN + kM * kN -
           kN * kN / 2), // Dropping the linear term
      benchmark::Counter::kIsRate);
}

BENCHMARK_TEMPLATE(QRFactorization, float, 50, 50, kColMajor, false, QrAlgorithm::MGS)
    ->Name("DenseLA/QRFactorization/DynamicSize/M50/N50/ColMajor/MGS");
BENCHMARK_TEMPLATE(QRFactorization, float, 50, 50, kColMajor, false, QrAlgorithm::CGS)
    ->Name("DenseLA/QRFactorization/DynamicSize/M50/N50/ColMajor/CGS");
BENCHMARK_TEMPLATE(QRFactorization, float, 50, 50, kColMajor, false, QrAlgorithm::ICGS)
    ->Name("DenseLA/QRFactorization/DynamicSize/M50/N50/ColMajor/ICGS");
BENCHMARK_TEMPLATE(QRFactorization, float, 50, 50, kRowMajor, false, QrAlgorithm::MGS)
    ->Name("DenseLA/QRFactorization/DynamicSize/M50/N50/RowMajor/MGS");
BENCHMARK_TEMPLATE(QRFactorization, float, 50, 50, kRowMajor, false, QrAlgorithm::CGS)
    ->Name("DenseLA/QRFactorization/DynamicSize/M50/N50/RowMajor/CGS");
BENCHMARK_TEMPLATE(QRFactorization, float, 50, 50, kRowMajor, false, QrAlgorithm::ICGS)
    ->Name("DenseLA/QRFactorization/DynamicSize/M50/N50/RowMajor/ICGS");
BENCHMARK_TEMPLATE(QRFactorization, float, 125, 125, kColMajor, false, QrAlgorithm::MGS)
    ->Name("DenseLA/QRFactorization/DynamicSize/M125/N125/ColMajor/MGS");
BENCHMARK_TEMPLATE(QRFactorization, float, 125, 125, kColMajor, false, QrAlgorithm::CGS)
    ->Name("DenseLA/QRFactorization/DynamicSize/M125/N125/ColMajor/CGS");
BENCHMARK_TEMPLATE(QRFactorization, float, 125, 125, kColMajor, false, QrAlgorithm::ICGS)
    ->Name("DenseLA/QRFactorization/DynamicSize/M125/N125/ColMajor/ICGS");
BENCHMARK_TEMPLATE(QRFactorization, float, 125, 125, kRowMajor, false, QrAlgorithm::MGS)
    ->Name("DenseLA/QRFactorization/DynamicSize/M125/N125/RowMajor/MGS");
BENCHMARK_TEMPLATE(QRFactorization, float, 125, 125, kRowMajor, false, QrAlgorithm::CGS)
    ->Name("DenseLA/QRFactorization/DynamicSize/M125/N125/RowMajor/CGS");
BENCHMARK_TEMPLATE(QRFactorization, float, 125, 125, kRowMajor, false, QrAlgorithm::ICGS)
    ->Name("DenseLA/QRFactorization/DynamicSize/M125/N125/RowMajor/ICGS");
BENCHMARK_TEMPLATE(QRFactorization, float, 500, 500, kColMajor, false, QrAlgorithm::MGS)
    ->Name("DenseLA/QRFactorization/DynamicSize/M500/N500/ColMajor/MGS");
BENCHMARK_TEMPLATE(QRFactorization, float, 500, 500, kColMajor, false, QrAlgorithm::CGS)
    ->Name("DenseLA/QRFactorization/DynamicSize/M500/N500/ColMajor/CGS");
BENCHMARK_TEMPLATE(QRFactorization, float, 500, 500, kColMajor, false, QrAlgorithm::ICGS)
    ->Name("DenseLA/QRFactorization/DynamicSize/M500/N500/ColMajor/ICGS");
BENCHMARK_TEMPLATE(QRFactorization, float, 500, 500, kRowMajor, false, QrAlgorithm::MGS)
    ->Name("DenseLA/QRFactorization/DynamicSize/M500/N500/RowMajor/MGS");
BENCHMARK_TEMPLATE(QRFactorization, float, 500, 500, kRowMajor, false, QrAlgorithm::CGS)
    ->Name("DenseLA/QRFactorization/DynamicSize/M500/N500/RowMajor/CGS");
BENCHMARK_TEMPLATE(QRFactorization, float, 500, 500, kRowMajor, false, QrAlgorithm::ICGS)
    ->Name("DenseLA/QRFactorization/DynamicSize/M500/N500/RowMajor/ICGS");

/****************************************************************************************
  Dot product of vectors
*/
enum class Impl { Mochi, Eigen, Scalar, VDot, MulAdd };

template <typename Scalar, int m, Impl kImpl>
static void Dot(benchmark::State& state) {
  using VT = Simd<Scalar>;
  ColumnVector<Scalar> a(m), b(m);
  a.SetRandom(1);
  b.SetRandom(2);
  Scalar result = {};
  for (auto x : state) {
    if constexpr (kImpl == Impl::Mochi) {
      result = a.Dot(b);
    } else if constexpr (kImpl == Impl::Eigen) {
      MOCHI_ASSERT_EIGEN();
#if MOCHI_USE_EIGEN
      using EMatType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1, Eigen::DontAlign>;
      auto aEigen = Eigen::Map<EMatType>(a.data(), m);
      auto bEigen = Eigen::Map<EMatType>(b.data(), m);
      result = aEigen.dot(bEigen);
#endif
    } else if constexpr (kImpl == Impl::Scalar) {
      result = 0;
      for (int i = 0; i < m; ++i) {
        result += a[i] * b[i];
      }
    } else if constexpr (kImpl == Impl::VDot) {
      int i = 0;
      result = 0;
      for (; i + VT::kSize <= m; i += VT::kSize) {
        result += Dot(Load<VT>(&a[i]), Load<VT>(&b[i]));
      }
      for (; i < m; ++i) {
        result += a[i] * b[i];
      }
    } else if constexpr (kImpl == Impl::MulAdd) {
      int i = 0;
      VT temp = {};
      for (; i + VT::kSize <= m; i += VT::kSize) {
        temp = MulAdd(Load<VT>(&a[i]), Load<VT>(&b[i]), temp);
      }
      result = HSum(temp);
      for (; i < m; ++i) {
        result += a[i] * b[i];
      }
    }
    MOCHI_NO_DISCARD_IN_LOOP(result);
  }
  state.counters["FLOPs"] =
      benchmark::Counter(state.iterations() * (2 * m - 1), benchmark::Counter::kIsRate);
  benchmark::DoNotOptimize(result);
}

BENCHMARK_TEMPLATE(Dot, float, 8, Impl::Mochi)->Name("DenseLA/Dot/M8/Mochi");
#if MOCHI_USE_EIGEN
BENCHMARK_TEMPLATE(Dot, float, 8, Impl::Eigen)->Name("DenseLA/Dot/M8/Eigen");
#endif // MOCHI_USE_EIGEN
BENCHMARK_TEMPLATE(Dot, float, 8, Impl::Scalar)->Name("DenseLA/Dot/M8/Scalar");
BENCHMARK_TEMPLATE(Dot, float, 8, Impl::VDot)->Name("DenseLA/Dot/M8/VDot");
BENCHMARK_TEMPLATE(Dot, float, 8, Impl::MulAdd)->Name("DenseLA/Dot/M8/MulAdd");
BENCHMARK_TEMPLATE(Dot, float, 10, Impl::Mochi)->Name("DenseLA/Dot/M10/Mochi");
#if MOCHI_USE_EIGEN
BENCHMARK_TEMPLATE(Dot, float, 10, Impl::Eigen)->Name("DenseLA/Dot/M10/Eigen");
#endif // MOCHI_USE_EIGEN
BENCHMARK_TEMPLATE(Dot, float, 10, Impl::Scalar)->Name("DenseLA/Dot/M10/Scalar");
BENCHMARK_TEMPLATE(Dot, float, 10, Impl::VDot)->Name("DenseLA/Dot/M10/VDot");
BENCHMARK_TEMPLATE(Dot, float, 10, Impl::MulAdd)->Name("DenseLA/Dot/M10/MulAdd");
BENCHMARK_TEMPLATE(Dot, float, 100, Impl::Mochi)->Name("DenseLA/Dot/M100/Mochi");
#if MOCHI_USE_EIGEN
BENCHMARK_TEMPLATE(Dot, float, 100, Impl::Eigen)->Name("DenseLA/Dot/M100/Eigen");
#endif // MOCHI_USE_EIGEN
BENCHMARK_TEMPLATE(Dot, float, 100, Impl::Scalar)->Name("DenseLA/Dot/M100/Scalar");
BENCHMARK_TEMPLATE(Dot, float, 100, Impl::VDot)->Name("DenseLA/Dot/M100/VDot");
BENCHMARK_TEMPLATE(Dot, float, 100, Impl::MulAdd)->Name("DenseLA/Dot/M100/MulAdd");
BENCHMARK_TEMPLATE(Dot, float, 1000, Impl::Mochi)->Name("DenseLA/Dot/M1000/Mochi");
#if MOCHI_USE_EIGEN
BENCHMARK_TEMPLATE(Dot, float, 1000, Impl::Eigen)->Name("DenseLA/Dot/M1000/Eigen");
#endif // MOCHI_USE_EIGEN
BENCHMARK_TEMPLATE(Dot, float, 1000, Impl::Scalar)->Name("DenseLA/Dot/M1000/Scalar");
BENCHMARK_TEMPLATE(Dot, float, 1000, Impl::VDot)->Name("DenseLA/Dot/M1000/VDot");
BENCHMARK_TEMPLATE(Dot, float, 1000, Impl::MulAdd)->Name("DenseLA/Dot/M1000/MulAdd");
BENCHMARK_TEMPLATE(Dot, float, 10000, Impl::Mochi)->Name("DenseLA/Dot/M10000/Mochi");
#if MOCHI_USE_EIGEN
BENCHMARK_TEMPLATE(Dot, float, 10000, Impl::Eigen)->Name("DenseLA/Dot/M10000/Eigen");
#endif // MOCHI_USE_EIGEN
BENCHMARK_TEMPLATE(Dot, float, 10000, Impl::Scalar)->Name("DenseLA/Dot/M10000/Scalar");
BENCHMARK_TEMPLATE(Dot, float, 10000, Impl::VDot)->Name("DenseLA/Dot/M10000/VDot");
BENCHMARK_TEMPLATE(Dot, float, 10000, Impl::MulAdd)->Name("DenseLA/Dot/M10000/MulAdd");

/****************************************************************************************
  Frobenius norm of a vector
*/

template <typename Scalar, int m, Impl kImpl>
static void Norm(benchmark::State& state) {
  ColumnVector<Scalar> a(m);
  a.SetRandom(1);
  Scalar norm = {};
  for (auto x : state) {
    if constexpr (kImpl == Impl::Mochi) {
      norm = a.Norm();
    } else if constexpr (kImpl == Impl::Eigen) {
      MOCHI_ASSERT_EIGEN();
#if MOCHI_USE_EIGEN
      using EMatType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1, Eigen::DontAlign>;
      norm = Eigen::Map<EMatType>(a.data(), m).norm();
#endif
    }
    MOCHI_NO_DISCARD_IN_LOOP(norm);
  }
  state.counters["FLOPs"] =
      benchmark::Counter(state.iterations() * 2 * m, benchmark::Counter::kIsRate);
  benchmark::DoNotOptimize(norm);
}

BENCHMARK_TEMPLATE(Norm, float, 100, Impl::Mochi)->Name("DenseLA/Norm/M100/Mochi");
#if MOCHI_USE_EIGEN
BENCHMARK_TEMPLATE(Norm, float, 100, Impl::Eigen)->Name("DenseLA/Norm/M100/Eigen");
#endif // MOCHI_USE_EIGEN
BENCHMARK_TEMPLATE(Norm, float, 1000, Impl::Mochi)->Name("DenseLA/Norm/M1000/Mochi");
#if MOCHI_USE_EIGEN
BENCHMARK_TEMPLATE(Norm, float, 1000, Impl::Eigen)->Name("DenseLA/Norm/M1000/Eigen");
#endif // MOCHI_USE_EIGEN
BENCHMARK_TEMPLATE(Norm, float, 10000, Impl::Mochi)->Name("DenseLA/Norm/M10000/Mochi");
#if MOCHI_USE_EIGEN
BENCHMARK_TEMPLATE(Norm, float, 10000, Impl::Eigen)->Name("DenseLA/Norm/M10000/Eigen");
#endif // MOCHI_USE_EIGEN

/****************************************************************************************
  Matrix assignment
*/

template <typename Scalar, int m, krylov::Direction kDir, bool kIsCompileTime>
static void Assignment(benchmark::State& state) {
  constexpr int mAtCT = kIsCompileTime ? m : kDynamic;
  Matrix<Scalar, mAtCT, mAtCT, kDir> A(m, m), B(m, m);
  A.SetRandom(1);
  B.SetRandom(2);
  for (auto x : state) {
    A = B;
    MOCHI_NO_DISCARD_IN_LOOP(A);
  }
  state.counters["Values/second"] =
      benchmark::Counter(state.iterations() * m * m, benchmark::Counter::kIsRate);
  benchmark::DoNotOptimize(A);
}

BENCHMARK_TEMPLATE(Assignment, float, 10, kColMajor, true)
    ->Name("DenseLA/Assignment/FixedSize/M10/ColMajor");
BENCHMARK_TEMPLATE(Assignment, float, 10, kRowMajor, true)
    ->Name("DenseLA/Assignment/FixedSize/M10/RowMajor");
BENCHMARK_TEMPLATE(Assignment, float, 100, kColMajor, false)
    ->Name("DenseLA/Assignment/DynamicSize/M100/ColMajor");
BENCHMARK_TEMPLATE(Assignment, float, 100, kRowMajor, false)
    ->Name("DenseLA/Assignment/DynamicSize/M100/RowMajor");
BENCHMARK_TEMPLATE(Assignment, float, 1000, kColMajor, false)
    ->Name("DenseLA/Assignment/DynamicSize/M1000/ColMajor");
BENCHMARK_TEMPLATE(Assignment, float, 1000, kRowMajor, false)
    ->Name("DenseLA/Assignment/DynamicSize/M1000/RowMajor");

template <typename Scalar, int m, krylov::Direction kDir, bool kIsCompileTime>
static void AssignmentPlus(benchmark::State& state) {
  constexpr int mAtCT = kIsCompileTime ? m : kDynamic;
  Matrix<Scalar, mAtCT, mAtCT, kDir> A(m, m), B(m, m);
  A.SetRandom(1);
  B.SetRandom(2);
  for (auto x : state) {
    A += B;
    MOCHI_NO_DISCARD_IN_LOOP(A);
  }
  state.counters["FLOPs"] =
      benchmark::Counter(state.iterations() * m * m, benchmark::Counter::kIsRate);
  benchmark::DoNotOptimize(A);
}

BENCHMARK_TEMPLATE(AssignmentPlus, float, 10, kColMajor, true)
    ->Name("DenseLA/AssignmentPlus/FixedSize/M10/ColMajor");
BENCHMARK_TEMPLATE(AssignmentPlus, float, 10, kRowMajor, true)
    ->Name("DenseLA/AssignmentPlus/FixedSize/M10/RowMajor");
BENCHMARK_TEMPLATE(AssignmentPlus, float, 100, kColMajor, false)
    ->Name("DenseLA/AssignmentPlus/DynamicSize/M100/ColMajor");
BENCHMARK_TEMPLATE(AssignmentPlus, float, 100, kRowMajor, false)
    ->Name("DenseLA/AssignmentPlus/DynamicSize/M100/RowMajor");
BENCHMARK_TEMPLATE(AssignmentPlus, float, 1000, kColMajor, false)
    ->Name("DenseLA/AssignmentPlus/DynamicSize/M1000/ColMajor");
BENCHMARK_TEMPLATE(AssignmentPlus, float, 1000, kRowMajor, false)
    ->Name("DenseLA/AssignmentPlus/DynamicSize/M1000/RowMajor");

} // namespace mochi_benchmark
