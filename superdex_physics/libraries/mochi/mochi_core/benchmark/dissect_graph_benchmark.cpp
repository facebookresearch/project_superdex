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

#include <mochi_core/linear_algebra/multi_frontal/elim_tree.h>
#include <mochi_core/numbering/dissect_graph.h>
#include <mochi_core/numbering/weighted_graph.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/graph.h>
#include <mochi_core/utils/graph_utils.h>

#if MOCHI_USE_HDF5
#include <mochi_core/geometry/tetrahedral_mesh.h>
#endif

#include <numeric>

using namespace mochi;

namespace mochi_benchmark {
namespace {

/// @brief Build an @p nx by @p ny grid graph with unit edge weights and explicit self-loops
/// of weight 0, matching the convention used elsewhere in the dissection code.
WeightedGraph MakeGrid2D(int nx, int ny) {
  int const n = nx * ny;
  // Each interior vertex has 4 neighbors + 1 self-loop; boundary vertices have fewer. We
  // upper-bound at 5 * n to size the storage once.
  GraphBuilder<WeightedEdge, int> builder(n, 5 * n);
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      int const v = i + j * nx;
      builder.StartSet();
      builder.InsertTarget({v, 0});
      if (i > 0) {
        builder.InsertTarget({v - 1, 1});
      }
      if (i + 1 < nx) {
        builder.InsertTarget({v + 1, 1});
      }
      if (j > 0) {
        builder.InsertTarget({v - nx, 1});
      }
      if (j + 1 < ny) {
        builder.InsertTarget({v + nx, 1});
      }
    }
  }
  return builder.Build();
}

/// @brief Build an @p nx by @p ny by @p nz grid graph with unit edge weights and explicit
/// self-loops of weight 0.
WeightedGraph MakeGrid3D(int nx, int ny, int nz) {
  int const n = nx * ny * nz;
  GraphBuilder<WeightedEdge, int> builder(n, 7 * n);
  int const stride = nx * ny;
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        int const v = i + j * nx + k * stride;
        builder.StartSet();
        builder.InsertTarget({v, 0});
        if (i > 0) {
          builder.InsertTarget({v - 1, 1});
        }
        if (i + 1 < nx) {
          builder.InsertTarget({v + 1, 1});
        }
        if (j > 0) {
          builder.InsertTarget({v - nx, 1});
        }
        if (j + 1 < ny) {
          builder.InsertTarget({v + nx, 1});
        }
        if (k > 0) {
          builder.InsertTarget({v - stride, 1});
        }
        if (k + 1 < nz) {
          builder.InsertTarget({v + stride, 1});
        }
      }
    }
  }
  return builder.Build();
}

} // namespace

/// @brief Build a connectivity-only @c Graph<int,int> mirroring the structure of @p w,
/// dropping edge weights. Used to feed @ref EliminationTree, which expects integer targets.
static Graph<int, int> ConnectivityGraph(WeightedGraph const& w) {
  GraphBuilder<int, int> builder(w.size(), static_cast<int>(w.NumTargets()));
  for (auto [v, edges] : w) {
    (void)v;
    builder.StartSet();
    for (auto const& e : edges) {
      builder.InsertTarget(e.vertex);
    }
  }
  return builder.Build();
}

/// @brief Build an @ref EliminationTree using @p order as the new→old vertex permutation, and
/// return its factorization operation count.
static double
FactorOpsForOrdering(Graph<int, int> const& graph, Span<int const> orderIn, int dofsPerNode) {
  DynamicArray<int> order(orderIn.begin(), orderIn.end());
  DynamicArray<int> position = ReverseMap(MakeConstSpan(order));
  Graph<int const, int const, Span> const view{graph.GetPointers(), graph.GetTargets()};
  EliminationTree const tree(view, MakeSpan(order), MakeSpan(position));
  return tree.ComputeFactorMetrics(dofsPerNode).opCount;
}

/// @brief Number of DOFs per node assumed for the operation-count user counters. The
/// counters scale as @c dofsPerNode^3, but the ratio between original and dissected
/// orderings is independent of this value.
static constexpr int kBenchmarkDofsPerNode = 1;

/// @brief Single-shot @ref Trisect on a 2D square grid. Provides a baseline for the
/// per-level cost of the recursive algorithm.
static void TrisectGrid2D(benchmark::State& state) {
  int const side = static_cast<int>(state.range(0));
  WeightedGraph const graph = MakeGrid2D(side, side);
  int const n = graph.size();
  DynamicArray<int> const mass(n, 1);
  int const maxUnbalance = n / 10;

  for (auto _ : state) {
    auto result = Trisect(graph, mass, maxUnbalance);
    MOCHI_NO_DISCARD_IN_LOOP(result);
  }
  state.counters["vertices"] = static_cast<double>(n);
}
BENCHMARK(TrisectGrid2D)->Arg(64)->Arg(128)->Arg(256)->Arg(512);

/// @brief Shared body for @ref RenumberByDissection benchmarks: timed loop plus untimed
/// factorization-op-count counters (@c opsOriginal, @c opsDissected, @c vertices).
static void RunRenumberByDissectionBenchmark(
    benchmark::State& state,
    WeightedGraph const& graph,
    int leafSize) {
  int const n = graph.size();
  DynamicArray<int> const mass(n, 1);
  int const maxUnbalance = n / 10;

  // Untimed: compute factorization op counts before vs. after dissection.
  Graph<int, int> const connectivity = ConnectivityGraph(graph);
  DynamicArray<int> identity(n);
  std::iota(identity.begin(), identity.end(), 0);
  double const opsOriginal =
      FactorOpsForOrdering(connectivity, MakeConstSpan(identity), kBenchmarkDofsPerNode);
  DynamicArray<int> const samplePerm = RenumberByDissection(graph, mass, maxUnbalance, leafSize);
  double const opsDissected =
      FactorOpsForOrdering(connectivity, MakeConstSpan(samplePerm), kBenchmarkDofsPerNode);

  for (auto _ : state) {
    auto perm = RenumberByDissection(graph, mass, maxUnbalance, leafSize);
    MOCHI_NO_DISCARD_IN_LOOP(perm);
  }
  state.counters["vertices"] = static_cast<double>(n);
  state.counters["opsOriginal"] = opsOriginal;
  state.counters["opsDissected"] = opsDissected;
}

/// @brief Recursive @ref RenumberByDissection on a 2D square grid. The two arguments are
/// (gridSide, leafSize). Reports the factorization operation count for the original
/// numbering (@c opsOriginal) and after dissection (@c opsDissected) as user counters.
static void RenumberByDissectionGrid2D(benchmark::State& state) {
  int const side = static_cast<int>(state.range(0));
  int const leafSize = static_cast<int>(state.range(1));
  RunRenumberByDissectionBenchmark(state, MakeGrid2D(side, side), leafSize);
  state.counters["side"] = static_cast<double>(side);
}
BENCHMARK(RenumberByDissectionGrid2D)->ArgsProduct({{64, 128, 256, 512}, {10, 25, 50, 100, 200}});

/// @brief Recursive @ref RenumberByDissection on a 3D cubic grid. The two arguments are
/// (gridSide, leafSize). 3D grids are the most realistic FEM-style input for nested
/// dissection. Reports the factorization operation count for the original numbering
/// (@c opsOriginal) and after dissection (@c opsDissected) as user counters.
static void RenumberByDissectionGrid3D(benchmark::State& state) {
  int const side = static_cast<int>(state.range(0));
  int const leafSize = static_cast<int>(state.range(1));
  RunRenumberByDissectionBenchmark(state, MakeGrid3D(side, side, side), leafSize);
  state.counters["side"] = static_cast<double>(side);
}
BENCHMARK(RenumberByDissectionGrid3D)->ArgsProduct({{16, 24, 32, 64, 128}, {10, 25, 50, 100, 200}});

// The Duck mesh is not shipped externally.
#if MOCHI_USE_HDF5 && MOCHI_INTERNAL
/// @brief Build a nodal graph from the duck tet mesh (13051 nodes, 67889 elements).
static Graph<int, int> NodalGraphFromDuckTetMesh() {
  auto mesh = LoadTetrahedralMesh(GetAssetPath("duck/duck_13051.mochi.h5"), ErrorAssert{});
  auto const elems = mesh->GetElementConnectivity();
  int const numElems = isize(elems);
  int const numNodes = mesh->GetNumNodes();
  GraphBuilder<int, int> gb(numElems, 4 * numElems);
  for (int e = 0; e < numElems; ++e) {
    gb.append(elems[e]);
  }
  auto eToN = gb.Build();
  return Traverse(Reverse<int, int>(eToN, numNodes), eToN);
}

/// @brief Recursive @ref RenumberByDissection on the duck tet mesh (13051 nodes). The
/// argument is @c leafSize. Provides an unstructured-mesh data point complementing the
/// synthetic grid benchmarks.
static void RenumberByDissectionDuck13051(benchmark::State& state) {
  int const leafSize = static_cast<int>(state.range(0));
  Graph<int, int> const nodalGraph = NodalGraphFromDuckTetMesh();
  WeightedGraph const graph = MakeWeightedGraph(nodalGraph, 1);
  RunRenumberByDissectionBenchmark(state, graph, leafSize);
}
BENCHMARK(RenumberByDissectionDuck13051)->Arg(10)->Arg(25)->Arg(50)->Arg(100)->Arg(200);
#endif // MOCHI_USE_HDF5 && MOCHI_INTERNAL

} // namespace mochi_benchmark
