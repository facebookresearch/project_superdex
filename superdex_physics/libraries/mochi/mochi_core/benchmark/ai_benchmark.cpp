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

#include <mochi_core/ai/mlp.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/task_scheduler.h>

#include <limits>
#include <utility>

using namespace mochi;
using namespace mochi::ai;

namespace mochi_benchmark {

/****************************************************************************************
  Activation Functions
*/

template <typename Scalar, typename ActivationType, krylov::Direction kDir, bool kComputeDerivative>
static void ApplyActivationHelper(benchmark::State& state) {
  auto const numRows = static_cast<int>(state.range(0));
  auto const numCols = static_cast<int>(state.range(1));
  Matrix<Scalar, krylov::kDynamic, krylov::kDynamic, kDir> Z(numRows, numCols),
      dfdZ(numRows, numCols);
  Z.SetRandom(1, Scalar(-1), Scalar(1));
  ActivationType f = {};
  for (auto x : state) {
    if constexpr (kComputeDerivative) {
      mochi::ai::details::ActivationInPlace(f, Z, dfdZ);
      MOCHI_NO_DISCARD_IN_LOOP(dfdZ);

    } else {
      mochi::ai::details::ActivationInPlace(f, Z);
    }
    MOCHI_NO_DISCARD_IN_LOOP(Z);
  }
  benchmark::DoNotOptimize(Z);
  benchmark::DoNotOptimize(dfdZ);

  state.counters["Values/second"] =
      benchmark::Counter(state.iterations() * numRows * numCols, benchmark::Counter::kIsRate);
}

template <typename Scalar, typename ActivationType, krylov::Direction kDir>
static void ApplyActivation(benchmark::State& state) {
  ApplyActivationHelper<Scalar, ActivationType, kDir, false>(state);
}

template <typename Scalar, typename ActivationType, krylov::Direction kDir>
static void ApplyActivationWithDerivative(benchmark::State& state) {
  ApplyActivationHelper<Scalar, ActivationType, kDir, true>(state);
}

// clang-format off
BENCHMARK_TEMPLATE(ApplyActivation, float, IdentityActivation<float>, krylov::Direction::ColMajor)
    ->Name("AI/ApplyActivation/Identity")
    ->ArgNames({"rows", "cols"})
    ->Args({60, 2048});
BENCHMARK_TEMPLATE(ApplyActivation, float, ReLUActivation<float>, krylov::Direction::ColMajor)
    ->Name("AI/ApplyActivation/ReLU")
    ->ArgNames({"rows", "cols"})
    ->Args({60, 2048});
BENCHMARK_TEMPLATE(ApplyActivation, float, ELUActivation<float>, krylov::Direction::ColMajor)
    ->Name("AI/ApplyActivation/ELU")
    ->ArgNames({"rows", "cols"})
    ->Args({60, 2048});

BENCHMARK_TEMPLATE(ApplyActivationWithDerivative, float, IdentityActivation<float>, krylov::Direction::ColMajor)
    ->Name("AI/ApplyActivationWithDerivative/Identity")
    ->ArgNames({"rows", "cols"})
    ->Args({60, 2048});
BENCHMARK_TEMPLATE(ApplyActivationWithDerivative, float, ReLUActivation<float>, krylov::Direction::ColMajor)
    ->Name("AI/ApplyActivationWithDerivative/ReLU")
    ->ArgNames({"rows", "cols"})
    ->Args({60, 2048});
BENCHMARK_TEMPLATE(ApplyActivationWithDerivative, float, ELUActivation<float>, krylov::Direction::ColMajor)
    ->Name("AI/ApplyActivationWithDerivative/ELU")
    ->ArgNames({"rows", "cols"})
    ->Args({60, 2048});
// clang-format on

/****************************************************************************************
  MLP Forward and Jacobian
*/

template <typename Scalar, typename ActivationType, bool kComputeJacobian>
static void MlpForwardAndJacobianHelper(benchmark::State& state) {
  using WeightType = typename MlpLayer<Scalar>::WeightType;
  using BiasType = typename MlpLayer<Scalar>::BiasType;
  auto const numLayers = static_cast<int>(state.range(0));
  auto const hiddenDim = static_cast<int>(state.range(1));
  auto const inputDim = static_cast<int>(state.range(2));
  auto const outputDim = static_cast<int>(state.range(3));
  auto const batchSize = static_cast<int>(state.range(4));
  auto const numThreads = static_cast<int>(state.range(5));
  MOCHI_ASSERT(numThreads > 0, "Invalid number of threads.");
  if (numThreads > TaskScheduler::GetNumSupportedLogicalProcessors()) {
    MOCHI_SKIP_BENCHMARK(state);
    return;
  }
  TaskScheduler scheduler(numThreads - 1); // Caller not included in TaskScheduler's pool.

  // MLP.
  DynamicArray<MlpLayer<Scalar>> layers;
  layers.reserve(numLayers);
  layers.emplace_back(WeightType(hiddenDim, inputDim), BiasType(hiddenDim), ActivationType());
  for (int l = 1; l < numLayers - 1; ++l) {
    layers.emplace_back(WeightType(hiddenDim, hiddenDim), BiasType(hiddenDim), ActivationType());
  }
  layers.emplace_back(WeightType(outputDim, hiddenDim), BiasType(outputDim), ActivationType());
  for (auto& l : layers) {
    l.Randomize();
  }
  Mlp<Scalar> mlp(std::move(layers));

  // Inputs, outputs and Jacobian.
  Matrix<Scalar> X(inputDim, batchSize), Y(outputDim, batchSize);
  RowMatrix<Scalar> dYdX(outputDim * batchSize, inputDim);
  X.SetRandom(1);

  // Run benchmark.
  for (auto x : state) {
    if constexpr (kComputeJacobian) {
      mlp.ForwardAndJacobian(X, Y, dYdX);
      MOCHI_NO_DISCARD_IN_LOOP(dYdX);
    } else {
      mlp.Forward(X, Y);
    }
    MOCHI_NO_DISCARD_IN_LOOP(Y);
  }
  benchmark::DoNotOptimize(Y);
  benchmark::DoNotOptimize(dYdX);

  // Report FLOPs.
  auto const flopsPerIter = kComputeJacobian
      ? (mlp.ForwardFlopsPerPoint() + mlp.JacobianFlopsPerPoint()) * batchSize
      : mlp.ForwardFlopsPerPoint() * batchSize;
  state.counters["FLOPs"] =
      benchmark::Counter(state.iterations() * flopsPerIter, benchmark::Counter::kIsRate);
}

template <typename Scalar, typename ActivationType>
static void MlpForward(benchmark::State& state) {
  MlpForwardAndJacobianHelper<Scalar, ActivationType, false>(state);
}

template <typename Scalar, typename ActivationType>
static void MlpForwardAndJacobian(benchmark::State& state) {
  MlpForwardAndJacobianHelper<Scalar, ActivationType, true>(state);
}

// Benchmark different activations (single-threaded) and strong parallel scaling with ELU
// activation. Use MLP architecture from MVP CROMs: 6 (total) layers, 60 units in hidden layers, 13
// input dimensions (position + 10 latent variables), 3 output dimensions and 2048 points.
BENCHMARK_TEMPLATE(MlpForward, float, IdentityActivation<float>)
    ->Name("AI/MlpForward/Identity")
    ->ArgNames({"layers", "hiddenDim", "inputDim", "outputDim", "batchSize", "threads"})
    ->Args({6, 60, 13, 3, 2048, 1})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();
BENCHMARK_TEMPLATE(MlpForward, float, ReLUActivation<float>)
    ->Name("AI/MlpForward/ReLU")
    ->ArgNames({"layers", "hiddenDim", "inputDim", "outputDim", "batchSize", "threads"})
    ->Args({6, 60, 13, 3, 2048, 1})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();
BENCHMARK_TEMPLATE(MlpForward, float, ELUActivation<float>)
    ->Name("AI/MlpForward/ELU")
    ->ArgNames({"layers", "hiddenDim", "inputDim", "outputDim", "batchSize", "threads"})
    ->ArgsProduct({{6}, {60}, {13}, {3}, {2048}, {1, 2, 4, 8, 16, 32}})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_TEMPLATE(MlpForwardAndJacobian, float, IdentityActivation<float>)
    ->Name("AI/MlpForwardAndJacobian/Identity")
    ->ArgNames({"layers", "hiddenDim", "inputDim", "outputDim", "batchSize", "threads"})
    ->Args({6, 60, 13, 3, 2048, 1})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();
BENCHMARK_TEMPLATE(MlpForwardAndJacobian, float, ReLUActivation<float>)
    ->Name("AI/MlpForwardAndJacobian/ReLU")
    ->ArgNames({"layers", "hiddenDim", "inputDim", "outputDim", "batchSize", "threads"})
    ->Args({6, 60, 13, 3, 2048, 1})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();
BENCHMARK_TEMPLATE(MlpForwardAndJacobian, float, ELUActivation<float>)
    ->Name("AI/MlpForwardAndJacobian/ELU")
    ->ArgNames({"layers", "hiddenDim", "inputDim", "outputDim", "batchSize", "threads"})
    ->ArgsProduct({{6}, {60}, {13}, {3}, {2048}, {1, 2, 4, 8, 16, 32}})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

} // namespace mochi_benchmark
