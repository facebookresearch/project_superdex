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

#pragma once

#include <mochi_core/element_operations/batched_element_utils.h>
#include <mochi_core/linear_algebra/block_sparse_matrix.h>
#include <mochi_core/linear_algebra/utils/assembly.h>
#include <mochi_core/memory/filo_allocator.h>
#include <mochi_core/mochi_platform.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/local_to_global_map.h>
#include <mochi_core/utils/prefetch.h>
#include <mochi_core/utils/profile.h>

#include <array>
#include <cstdint>
#include <type_traits>
#include <variant>

namespace mochi {

/**
 * @page dyn_assembly Dynamic Load-balancing Assembly
 * @section dyn_intro Principle
 * The dynamic automatically load-balancing assembly of BlockSparseMatrix from element matrices
 * is based on a decomposition of elements into set named aggregates. The work is parallelized
 * in a way that no two threads can simultaneously work on two aggregates that share any node.
 * Because of this property of the work organization, there is no need for any extraneous
 * storage for partial assemblies, nor for a second pass to sum partial assemblies.
 * The underlying technique for organizing such independent work is based on a coloring
 * of the assembly node-based connectivity graph. All the aggregates of a given color can be
 * worked on simultaneously without any conflict of simultaneous memory update.
 * Thus, it is possible to have the threads progress from color to color in a synchronous
 * manner.
 * However, such an approach shows a limited scalability. Instead, a more complex dependency
 * graph is derived from the coloring, and work for threads is based on this dependency graph.
 * This approach shows great scalability performance for a large number of threads.
 *
 * @section dyn_details Algorithm Details
 *
 * @subsection dyn_agg Element Aggregation
 *
 * To organize the parallel assembly, a first preprocessing phase is performed. It involves
 * the following steps:
 *    - Construct the graph of the matrix, i.e. node to node connectivity via elements.
 *    - Decompose the nodes, based on the connectivity, into small groups of nodes using a simple
 *    greedy decomposition method.
 *    - Aggregate elements connected to the node groups, aggregating around
 *    the smallest groups last.
 *
 * Note that one could perform an element based decomposition, but the greedy decomposition
 * is known to lead to irregular subdomains, some of which could have very few elements.
 * By using a nodal decomposition and aggregating first around all node groups larger than
 * a threshold, this problem is avoided. The result is a more evenly distributed aggregation
 * of elements.
 *
 * @subsection dyn_color Aggregate Coloring and Dependencies
 *
 * Once the aggregates are obtained, the graph of aggregate to aggregate connectivity via nodes
 * is constructed and the aggregates are colored such that no two aggregates of the same color
 * share a node. It is safe to process all the aggregates of a given color in parallel. As they
 * do not share any node, no two of them can be updating the same entry in the matrix.
 *
 * The simplest strategy of assembly is to proceed color-by-color in increasing color index
 * and having the thread synchronize at a barrier between colors. However this approach does not
 * scale very well. Waiting for a full color to be completed before starting any aggregate of the
 * next color is too strong a constraint. In fact, one can start processing
 * a given aggregate \f$ a \f$ of color \f$ c \f$  as soon as all the neighboring aggregates
 * \f$ d_n \f$ of color lower than \f$ c\f$ have completed. From this remark, one can base
 * the dispatching of work on a dependency graph, keeping track of the state of each aggregate
 * based on its dependencies:
 *
 * Let \f$ N_a \f$ be the set of neighboring aggregates of aggregate \f$ a \f$. The subset
 * of those aggregates whose color is lower will be noted by \f$ \bar N_a \f$. This reduced
 * subset can be used to build a directed graph from higher color to lower colors, indicating
 * the full set of aggregates that have to be completed before \f$ a \f$ can be processed.
 * However, even this graph is larger than necessary. Consider the following example:
 * Let \f$ \bar N_D = \{
  A, B, C
} \f$ be neighboring aggregates of D with A and B of color 0,
 * and \f$ C \f$ is of color 1 while \f$ D \f$ is of color 2.
 * Furthermore, let's assume that \f$ \bar N_C = \{ A, B \} \f$. It is thus not necessary for
 * \f$ D \f$ to check on \f$ A \f$ or \f$ B \f$, since when \f$ C \f$ has completed, the other
 * two dependencies are guaranteed to have been completed.
 *
 * Thus, the graph formed by the \f$ \bar N_a \f$ can be pruned into a much smaller graph
 * \f$ N^*_a \f$ from which the implicitly satisfied dependencies have been removed. In practice,
 * this pruned graph has on average around 3 dependencies for each aggregate.
 *
 * @subsection dyn_dynamic Dynamic Natural Load Balancing of Work
 *
 * To dynamically dispatch the assembly work to threads, three state bits are maintained for
 * each aggregate:
 *    - ready: if true, all the dependencies of the aggregate have completed
 *    - acquired: indicates if a thread has acquired the aggregate to assemble its contribution
 *    - finished: indicates when the assembly of the aggregate has been completed
 *
 * The flags are stored in three arrays of std::atomic<uint64_t>. At the start of a new assembly,
 * all the flags are cleared and the ready flags for all the aggregates of color 0 are set to true.

 * Each thread proceeds in a loop:
 *   - request to acquire an aggregate that is ready and that has not yet been acquired.
 *   - assembles the contribution of all the elements of the aggregate.
 *   - mark the work on the aggregate as completed
 *
 * The last step is the more complex:
 *   - Set the finished flag of the aggregate to true.
 *   - For each of the dependent aggregates, check their dependencies and update the
 *     ready flag if they have all finished.
 *
 * Threads exit the loop when all aggregates have been acquired.
 *
 * @subsection dyn_atomic Atomic Flag Update Care
 *
 * As much as possible, the minimal constraints of memory model is employed for atomic operations.
 * Note however that there may be more than 64 aggregates and consequently there may be more than
 * one uint64_t holding the `ready`, `acquired` or `finished` state.
 * Though most C++ programmer who have attempted to take advantage of the full C++ memory model
 * may be familiar with the `release` and `acquire` concepts of the model, they may not be aware
 * of the details regarding the writing and reading of several atomic variables. For x86 machines,
 * this does not cause any problem, as the hardware implements a sequentially consistent
 * model always. However for ARM and RISC-V, this is not the case. The following is an explanation
 * of a surprising behavior that the dynamic dispatch has to protect itself against:
 *
 *    Let's assume thread A and thread B complete work of aggregates a and b at the same time and
 *    that aggregate c depends on both a and b. A will mark finished_a = true atomically and
 *    B will mark finished_b = true. Such marking has to be done with a release semantic, since
 *    the modifications made by the threads must become visible to other threads that will check
 *    for the completion of that work. After having marked that their work is finished, both threads
 *    will examine the dependencies of `c`. Surprisingly, with a `release` or `release and acquire`
 *    semantic on the atomic operations of A and B, it is possible that both A sees
 *    finished_b == false  and B sees finished_a == false thus leading both to the conclusion that
 *    c is not ready. This is possible because in such case, these architectures do not guarantee
 *    any visible order of operations on different atomic variables. A is allowed to see
 *    a stale value of finished_b that B updates if it is in a different atomic variables than
 *    finished_a and similarly for B.
 *    The only update mode that guarantees that will not happen is a sequentially consistent
 *    update.
 *
 *  The `finished` atomics are the only place where the strictest and costliest sequentially
 *  consistent mode of operation is being used.
 *
 */

namespace details {

/**
 * @brief Gather batch element solution from the global solution vector.
 *
 * @tparam ElementT    Element type
 * @tparam kNumFields  Number of fields per node
 * @tparam kBatchSize  Number of elements per batch
 * @param[in]  globalSol         Global solution vector
 * @param[in]  indicesFlat       Flat local-to-global DoF indices
 * @param[in]  globalElemIndices Per-batch-lane element indices (padded to kBatchSize)
 * @param[out] outBatchElemSol   Batch element displacement vector
 */
template <class ElementT, int kNumFields, int kBatchSize>
MOCHI_FORCE_INLINE void GatherBatchElementSolution(
    ColumnVectorView<real const> globalSol,
    Span<int const> indicesFlat,
    NdArray<int, kBatchSize> const& globalElemIndices,
    fem::BatchElementVector<kBatchSize, ElementT, kNumFields>& outBatchElemSol) {
  using V = BatchReal<kBatchSize>;
  static_assert(ElementT::kSpaceDim == 3, "GatherBatchElementSolution requires kSpaceDim == 3");
  static_assert(kNumFields <= 4, "GatherBatchElementSolution requires kNumFields <= 4");
  constexpr int kNumNodes = ElementT::kNumDofs;
  constexpr int kNumEleDofs = kNumNodes * kNumFields;

  for (int d = 0; d < kNumEleDofs; ++d) {
    if constexpr (V::kSize == 2) {
      outBatchElemSol[d] =
          V{globalSol[indicesFlat[globalElemIndices[Min(0, kBatchSize - 1)] * kNumEleDofs + d]],
            globalSol[indicesFlat[globalElemIndices[Min(1, kBatchSize - 1)] * kNumEleDofs + d]]};
    } else if constexpr (V::kSize == 4) {
      outBatchElemSol[d] =
          V{globalSol[indicesFlat[globalElemIndices[Min(0, kBatchSize - 1)] * kNumEleDofs + d]],
            globalSol[indicesFlat[globalElemIndices[Min(1, kBatchSize - 1)] * kNumEleDofs + d]],
            globalSol[indicesFlat[globalElemIndices[Min(2, kBatchSize - 1)] * kNumEleDofs + d]],
            globalSol[indicesFlat[globalElemIndices[Min(3, kBatchSize - 1)] * kNumEleDofs + d]]};
    } else {
      static_assert(V::kSize == 8, "Unsupported SIMD size");
      outBatchElemSol[d] =
          V{globalSol[indicesFlat[globalElemIndices[Min(0, kBatchSize - 1)] * kNumEleDofs + d]],
            globalSol[indicesFlat[globalElemIndices[Min(1, kBatchSize - 1)] * kNumEleDofs + d]],
            globalSol[indicesFlat[globalElemIndices[Min(2, kBatchSize - 1)] * kNumEleDofs + d]],
            globalSol[indicesFlat[globalElemIndices[Min(3, kBatchSize - 1)] * kNumEleDofs + d]],
            globalSol[indicesFlat[globalElemIndices[Min(4, kBatchSize - 1)] * kNumEleDofs + d]],
            globalSol[indicesFlat[globalElemIndices[Min(5, kBatchSize - 1)] * kNumEleDofs + d]],
            globalSol[indicesFlat[globalElemIndices[Min(6, kBatchSize - 1)] * kNumEleDofs + d]],
            globalSol[indicesFlat[globalElemIndices[Min(7, kBatchSize - 1)] * kNumEleDofs + d]]};
    }
  }
}

template <class ElementT, int kNumFields, int kBatchSize, bool kGatherSolution>
using MaybeNoDispElOpFnType = std::conditional_t<
    kGatherSolution,
    ElOpFnType<ElementT, kNumFields, kBatchSize>,
    NoDispElOpFnType<ElementT, kNumFields, kBatchSize>>;

/** @brief Fused scatter-add of residual and/or dresidual into global outputs. */
template <class ElementT, int kNumFields, int kBatchSize, bool kScatterRes, bool kScatterDRes>
MOCHI_FORCE_INLINE void ScatterBatchElementResults(
    Span<int const> indicesFlat,
    NdArray<int, kBatchSize> const& batchElemIndices,
    int actualBatchSize,
    NodalBasedStructure const& nbs,
    fem::BatchElementVector<kBatchSize, ElementT, kNumFields> const& batchRes,
    fem::BatchElementMatrix<kBatchSize, ElementT, kNumFields> const& batchDRes,
    ColumnVectorView<real> outRes,
    BlockSparseMatrixView<real, kNumFields> outDRes) {
  static_assert(kNumFields > 0 && kNumFields <= 4, "Unsupported number of fields");
  static_assert(kScatterRes || kScatterDRes, "Must scatter something");
  MOCHI_ASSERT_VERBOSE(
      actualBatchSize >= 1 && actualBatchSize <= kBatchSize, "Invalid batch size.");
  constexpr int kNumNodes = ElementT::kNumDofs;
  constexpr int kNumDofs = kNumNodes * kNumFields;

  [[maybe_unused]] real* aiBase[2][kNumNodes] MOCHI_NO_INIT;
  [[maybe_unused]] int aiStride[2][kNumNodes] MOCHI_NO_INIT;
  [[maybe_unused]] int const* siPtr[2] = {};

  [[maybe_unused]] auto precomputeNbs = [&](int b, int slot) MOCHI_FORCE_INLINE_LAMBDA {
    if constexpr (kScatterDRes) {
      auto nodes = nbs.GetEleNodes(batchElemIndices[b]);
      MOCHI_ASSERT_VERBOSE(
          isize(nodes) == kNumNodes,
          "NBS per-element node count must equal kNumNodes. For variable node count per element, the NBS must be padded to kNumNodes.");
      siPtr[slot] = nbs.GetNodeSparseIndices(batchElemIndices[b]).Data();
      for (int nI = 0; nI < kNumNodes; ++nI) {
        auto Ai = outDRes.Values(nodes[nI]);
        aiBase[slot][nI] = Ai.Data();
        aiStride[slot][nI] = Ai.LeadDim();
      }
    }
  };

  // Precompute first element.
  if constexpr (kScatterDRes) {
    precomputeNbs(0, 0);
  }

  // NOTE: AssemblerOverhead benchmarks showed ~5-10% speedup on x86 by reading batchRes/batchDRes
  // as scalar lane storage instead of using repeated SIMD lane extraction. We keep the explicit
  // BatchReal::operator[] access here because that optimization relies on
  // reinterpret_cast<real const*> and unsupported SIMD storage-layout/aliasing assumptions.
  for (int b = 0; b < actualBatchSize; ++b) {
    int const cur = b & 1;
    int const nxt = 1 - cur;
    int const elemBase = batchElemIndices[b] * kNumDofs;

    // Precompute next element's NBS data and issue prefetches for its block row pointers while
    // scattering the current element.
    if constexpr (kScatterDRes) {
      if (b + 1 < actualBatchSize) {
        precomputeNbs(b + 1, nxt);
        for (int nI = 0; nI < kNumNodes; ++nI) {
          PrefetchWrite(aiBase[nxt][nI]);
        }
      }
    }

    for (int nI = 0; nI < kNumNodes; ++nI) {
      if constexpr (kScatterRes) {
        int const globalDof = indicesFlat[elemBase + nI * kNumFields];
#if MOCHI_ASSERT_VERBOSE_ENABLED
        for (int k = 1; k < kNumFields; ++k) {
          MOCHI_ASSERT_VERBOSE(
              indicesFlat[elemBase + nI * kNumFields + k] == globalDof + k,
              "Per-node DoFs must be contiguous in the global vector.");
        }
#endif // MOCHI_ASSERT_VERBOSE_ENABLED
        real* MOCHI_RESTRICT r = &outRes[globalDof];
        Store<kNumFields>(
            r,
            Load<kNumFields, Vec4r>(r) +
                Vec4r{
                    batchRes[nI * kNumFields + 0][b],
                    kNumFields >= 2 ? batchRes[nI * kNumFields + 1][b] : 0_r,
                    kNumFields >= 3 ? batchRes[nI * kNumFields + 2][b] : 0_r,
                    kNumFields >= 4 ? batchRes[nI * kNumFields + 3][b] : 0_r});
      }

      if constexpr (kScatterDRes) {
        real* const MOCHI_RESTRICT base = aiBase[cur][nI];
        int const stride = aiStride[cur][nI];

        // Some compilers (e.g. GCC on x86) do not unroll the loops below, causing a ~4x slowdown.
        // Request the compiler to unroll them to avoid pathologically slow generated code.
        static_assert(
            kNumNodes <= 6 && kNumFields <= 4,
            "Loop unrolling has only been benchmarked for the current size limits. Benchmark performance before increasing them.");
        MOCHI_UNROLL_LOOP_N(6)
        for (int nJ = 0; nJ < kNumNodes; ++nJ) {
          int const jPos = siPtr[cur][nI * kNumNodes + nJ] * kNumFields;
          MOCHI_UNROLL_LOOP_N(4)
          for (int k = 0; k < kNumFields; ++k) {
            int const batchBase = (nI * kNumFields + k) * kNumDofs + nJ * kNumFields;
            real* MOCHI_RESTRICT d = base + k * stride + jPos;
            Store<kNumFields>(
                d,
                Load<kNumFields, Vec4r>(d) +
                    Vec4r{
                        batchDRes[batchBase + 0][b],
                        kNumFields >= 2 ? batchDRes[batchBase + 1][b] : 0_r,
                        kNumFields >= 3 ? batchDRes[batchBase + 2][b] : 0_r,
                        kNumFields >= 4 ? batchDRes[batchBase + 3][b] : 0_r});
          }
        }
      }
    }
  }
}

/** @brief Gather the batch element solution (only when @p kGatherSolution is true) and invoke the
 * batched element operation. */
template <bool kGatherSolution, class ElementT, int kNumFields, int kBatchSize>
MOCHI_FORCE_INLINE bool InvokeBatchedElOp(
    MaybeNoDispElOpFnType<ElementT, kNumFields, kBatchSize, kGatherSolution> const& batchedElOp,
    NdArray<int, kBatchSize> const& batchElemIndices,
    Span<int const> indicesFlat,
    [[maybe_unused]] ColumnVectorView<real const> globalSol,
    BatchDouble<kBatchSize>* outEnergy,
    fem::BatchElementVector<kBatchSize, ElementT, kNumFields>* outRes,
    fem::BatchElementMatrix<kBatchSize, ElementT, kNumFields>* outDRes,
    bool psdDRes) {
  if constexpr (kGatherSolution) {
    fem::BatchElementVector<kBatchSize, ElementT, kNumFields> batchDisp MOCHI_NO_INIT;
    GatherBatchElementSolution<ElementT, kNumFields, kBatchSize>(
        globalSol, indicesFlat, batchElemIndices, batchDisp);
    return batchedElOp(
        batchElemIndices, indicesFlat, batchDisp, outEnergy, outRes, outDRes, psdDRes);
  } else {
    return batchedElOp(batchElemIndices, indicesFlat, outEnergy, outRes, outDRes, psdDRes);
  }
}

/// @brief Process one batch of elements.
///
/// @note Assembly flags are split between compile-time (kAssemRes, kAssemDRes) and runtime
/// (assemObj, psdDRes) based on benchmarked performance impact.
template <
    class ElementT,
    int kNumFields,
    int kBatchSize,
    bool kAssemRes,
    bool kAssemDRes,
    bool kGatherSolution>
MOCHI_FORCE_INLINE void ProcessBatch(
    Span<int const> indicesFlat,
    NdArray<int, kBatchSize> const& batchElemIndices,
    int actualBatchSize,
    MaybeNoDispElOpFnType<ElementT, kNumFields, kBatchSize, kGatherSolution> const& batchedElOp,
    bool assemObj,
    bool psdDRes,
    ColumnVectorView<real const> globalSol,
    NodalBasedStructure const& nbs,
    double* outObj,
    ColumnVectorView<real> outRes,
    BlockSparseMatrixView<real, kNumFields> outDRes,
    BatchDouble<kBatchSize>& batchEnergy,
    fem::BatchElementVector<kBatchSize, ElementT, kNumFields>& batchRes,
    fem::BatchElementMatrix<kBatchSize, ElementT, kNumFields>& batchDRes) {
  if (assemObj) {
    batchEnergy = {};
  }
  if constexpr (kAssemRes) {
    batchRes = {};
  }
  if constexpr (kAssemDRes) {
    batchDRes = {};
  }

  bool const hasOutput = InvokeBatchedElOp<kGatherSolution, ElementT, kNumFields, kBatchSize>(
      batchedElOp,
      batchElemIndices,
      indicesFlat,
      globalSol,
      assemObj ? &batchEnergy : nullptr,
      kAssemRes ? &batchRes : nullptr,
      kAssemDRes ? &batchDRes : nullptr,
      psdDRes);

  if (!hasOutput) {
    return;
  }

  // Validate the padded-position contract: when the padded L2G is used, it duplicates each
  // element's node-0's per-field DoF pattern into padded stencil positions (see
  // Local2GlobalMap::InitializePaddedIndices). The element operations MUST produce zero residual
  // and dresidual for these positions, otherwise the scatter corrupts node-0's DoFs.
  //
  // Detection: a node is padded if ALL kNumFields DoF indices for that node equal node-0's
  // corresponding DoF indices. This distinguishes true padding from coincidental index sharing.
  // For raw (non-padded) L2G, all nodes have distinct DoF indices, so no padded positions are
  // detected and the validation is a no-op.
#if MOCHI_ASSERT_VERBOSE_ENABLED
  constexpr int kNumDofs = ElementT::kNumDofs * kNumFields;
  for (int b = 0; b < actualBatchSize; ++b) {
    int const elemBase = batchElemIndices[b] * kNumDofs;
    for (int n = 1; n < ElementT::kNumDofs; ++n) {
      // Check if all kNumFields DoFs of node n match node 0's DoFs.
      bool isPadded = true;
      for (int f = 0; f < kNumFields; ++f) {
        if (indicesFlat[elemBase + n * kNumFields + f] != indicesFlat[elemBase + f]) {
          isPadded = false;
          break;
        }
      }
      if (!isPadded) {
        continue;
      }
      int const d = n * kNumFields;
      if constexpr (kAssemRes) {
        for (int f = 0; f < kNumFields; ++f) {
          MOCHI_ASSERT_VERBOSE(
              batchRes[d + f][b] == 0_r,
              "Element operations produced non-zero residual at padded stencil position.");
        }
      }
      if constexpr (kAssemDRes) {
        for (int f = 0; f < kNumFields; ++f) {
          for (int j = 0; j < kNumDofs; ++j) {
            MOCHI_ASSERT_VERBOSE(
                batchDRes[(d + f) * kNumDofs + j][b] == 0_r,
                "Element operations produced non-zero dresidual row at padded stencil position.");
            MOCHI_ASSERT_VERBOSE(
                batchDRes[j * kNumDofs + (d + f)][b] == 0_r,
                "Element operations produced non-zero dresidual column at padded stencil position.");
          }
        }
      }
    }
  }
#endif

  if (assemObj) {
    double sum = 0.0;
    for (int b = 0; b < actualBatchSize; ++b) {
      sum += batchEnergy[b];
    }
    *outObj += sum;
  }

  if constexpr (kAssemRes || kAssemDRes) {
    ScatterBatchElementResults<ElementT, kNumFields, kBatchSize, kAssemRes, kAssemDRes>(
        indicesFlat, batchElemIndices, actualBatchSize, nbs, batchRes, batchDRes, outRes, outDRes);
  }
}

template <
    class ElementT,
    int kNumFields,
    int kBatchSize,
    bool kAssemRes,
    bool kAssemDRes,
    bool kGatherSolution>
static void DynamicLoadBalancingAssembly(
    Span<int const> indicesFlat,
    NodalBasedStructure const& nbs,
    MaybeNoDispElOpFnType<ElementT, kNumFields, kBatchSize, kGatherSolution> const& batchedElOp,
    ColumnVectorView<real const> globalSol,
    Span<int const> activeElementIndices,
    Span<bool const> isElementActive,
    bool assemObj,
    bool psdDRes,
    double* outObj,
    ColumnVectorView<real> outRes,
    BlockSparseMatrixView<real, kNumFields> outDRes) {
  int const numGroups = nbs.NumGroups();
  int const numColors = Max(1, nbs.NumColors());
  int const numActiveElements =
      activeElementIndices.empty() ? nbs.NumElements() : isize(activeElementIndices);

  // Select the number of threads by applying three caps: available scheduler parallelism, average
  // NBS groups per color, and minimum estimated work per thread. NBS groups execute color by color,
  // so useful parallelism is bounded by the approximate color width rather than the total number of
  // groups. The work cap avoids over-parallelizing small assemblies where task scheduling and group
  // dispatch can dominate the assembly work.
  //
  // The per-element timing estimates below are intentionally conservative: the empirically optimal
  // cap is likely more aggressive, i.e. fewer elements per thread, but these values preserve more
  // parallelism for larger body assemblies while removing the worst small-assembly
  // over-parallelization cases. If this heuristic needs to be generalized, the timing estimates
  // could be optionally provided by the caller.
  int const schedulerThreads = TaskScheduler::StaticGetNumOtherThreads() + 1;
  int const maxThreadsByGroupsPerColor = (numGroups + numColors - 1) / numColors;
  constexpr int kMinWorkPerThreadNs = 12000;
  constexpr int kNsPerElementNoDRes = 100;
  constexpr int kNsPerElementWithDRes = 200;
  int const nsPerElement = kAssemDRes ? kNsPerElementWithDRes : kNsPerElementNoDRes;
  int const maxThreadsByWork =
      Max(1, static_cast<int>((int64_t{numActiveElements} * nsPerElement) / kMinWorkPerThreadNs));
  int const numThreads =
      Max(1, Min(schedulerThreads, maxThreadsByGroupsPerColor, maxThreadsByWork));

  WorkState ws(nbs);

  // Per-group objective for deterministic accumulation.
  constexpr int kGroupObjsAllocBytes = 10240; // Holds ~1k groups. Falls back to heap if exceeded.
  MOCHI_FILO_STACK_ALLOCATOR(objAlloc, kGroupObjsAllocBytes);
  DynamicArray<double> groupObjs(&objAlloc);
  groupObjs.resize_noinit(assemObj && numThreads > 1 ? numGroups : 0);

  auto dynamicWork = [&](int numThreads) {
    MOCHI_PROFILE_SCOPE_N("FemAssemblyWorkerTask");

    // Per-thread buffers.
    BatchDouble<kBatchSize> batchEnergy{};
    fem::BatchElementVector<kBatchSize, ElementT, kNumFields> batchRes{};
    fem::BatchElementMatrix<kBatchSize, ElementT, kNumFields> batchDRes{};

    if (numThreads == 1) {
      // Single-threaded: linear pass over all (active) elements to improve cache locality.
      double* obj = assemObj ? outObj : nullptr;
      NdArray<int, kBatchSize> batchElemIndices MOCHI_NO_INIT;
      bool const useSubset = !activeElementIndices.empty();
      for (int batchStart = 0; batchStart < numActiveElements; batchStart += kBatchSize) {
        int const actualBatchSize = Min(kBatchSize, numActiveElements - batchStart);
        for (int b = 0; b < kBatchSize; ++b) {
          int const i = batchStart + Min(b, actualBatchSize - 1);
          batchElemIndices[b] = useSubset ? activeElementIndices[i] : i;
        }
        ProcessBatch<ElementT, kNumFields, kBatchSize, kAssemRes, kAssemDRes, kGatherSolution>(
            indicesFlat,
            batchElemIndices,
            actualBatchSize,
            batchedElOp,
            assemObj,
            psdDRes,
            globalSol,
            nbs,
            obj,
            outRes,
            outDRes,
            batchEnergy,
            batchRes,
            batchDRes);
      }
    } else {
      // Multi-threaded: Dynamic load balancing aggregate dispatch.
      double localObj = 0.0;
      double* obj = assemObj ? &localObj : nullptr;

      DynamicArray<int> activeElems;
      activeElems.reserve(2 * nbs.NumElements() / numGroups);

      auto processGroup = [&](Span<int const> groupElements) {
        Span<int const> elemsToProcess = groupElements;
        if (!isElementActive.empty()) {
          activeElems.clear();
          for (auto e : groupElements) {
            if (isElementActive[e]) {
              activeElems.push_back(e);
            }
          }
          if (activeElems.empty()) {
            return;
          }
          elemsToProcess = MakeConstSpan(activeElems);
        }

        NdArray<int, kBatchSize> batchElemIndices MOCHI_NO_INIT;
        int const numElements = isize(elemsToProcess);
        for (int batchStart = 0; batchStart < numElements; batchStart += kBatchSize) {
          int const actualBatchSize = Min(kBatchSize, numElements - batchStart);
          for (int b = 0; b < kBatchSize; ++b) {
            batchElemIndices[b] = elemsToProcess[batchStart + Min(b, actualBatchSize - 1)];
          }
          ProcessBatch<ElementT, kNumFields, kBatchSize, kAssemRes, kAssemDRes, kGatherSolution>(
              indicesFlat,
              batchElemIndices,
              actualBatchSize,
              batchedElOp,
              assemObj,
              psdDRes,
              globalSol,
              nbs,
              obj,
              outRes,
              outDRes,
              batchEnergy,
              batchRes,
              batchDRes);
        }
      };

      constexpr auto kMaskBits = WorkState::kMaskBits;
      auto maskSize = (numGroups + kMaskBits - 1) / kMaskBits;
      int g = -1;
      DynamicArray<uint64_t> readyMask(maskSize);
      while (true) {
        if (g != -1) {
          ws.Complete(g, readyMask);
          if (assemObj) {
            groupObjs[g] = localObj;
            localObj = 0.0;
          }
        }
        g = ws.Acquire(readyMask);
        if (g == -1) {
          break;
        }
        processGroup(nbs.GetElemsInGroup(g));
      }
    }
  };

  // Enqueue tasks.
  TaskSemaphore sem(numThreads - 1);
  if (numThreads > 1) {
    auto* scheduler = TaskScheduler::TryGet();
    MOCHI_ASSERT(scheduler); // Must have succeeded if numThreads > 1.
    for (int threadId = 1; threadId < numThreads; ++threadId) {
      scheduler->AddTaskNoProfile(
          [&dynamicWork, sem, numThreads]() {
            dynamicWork(numThreads);
            sem.Done();
          },
          false);
    }
  }
  dynamicWork(numThreads);
  sem.Wait();

  // Combine per-group objectives (multi-threaded only; single-threaded combined above).
  if (assemObj && numThreads > 1) {
    MOCHI_ASSERT(outObj, "Invalid objective pointer.");
    *outObj += HSum(MakeConstSpan(groupObjs));
  }
}

template <class ElementT, int kNumFields, int kBatchSize, bool kGatherSolution>
void AssembleObjResDResImpl(
    Local2GlobalMap const& l2g,
    NodalBasedStructure const& nbs,
    MaybeNoDispElOpFnType<ElementT, kNumFields, kBatchSize, kGatherSolution> const& batchedElOp,
    ColumnVectorView<real const> globalSol,
    AssemblyResults<real> results,
    AssemblyActiveSubset const& activeSubset) {
  MOCHI_PROFILE_SCOPE();
  static_assert(ElementT::kSpaceDim == 3, "FEM assembler requires kSpaceDim == 3");
  static_assert(kNumFields <= 4, "FEM assembler requires kNumFields <= 4");
  MOCHI_ASSERT_VERBOSE(
      results.params.assemObj || results.params.assemRes || results.params.assemDRes,
      "Must assemble something.");
  MOCHI_ASSERT_VERBOSE(l2g.GetNumElements() > 0, "L2G map must not be empty.");
  MOCHI_ASSERT_VERBOSE(
      nbs.NumElements() == l2g.GetNumElements(), "NBS and L2G element count mismatch.");
  MOCHI_ASSERT_VERBOSE(
      IsUnique(activeSubset.activeElementIndices), "Active elements must be unique.");
  MOCHI_ASSERT_VERBOSE(
      activeSubset.activeElementIndices.empty() || !activeSubset.isElementActive.empty(),
      "isElementActive must be provided when activeElementIndices is non-empty.");
  MOCHI_ASSERT_VERBOSE(
      activeSubset.isElementActive.empty() ||
          isize(activeSubset.isElementActive) == l2g.GetNumElements(),
      "isElementActive size must match the number of elements.");
  if constexpr (kGatherSolution) {
    MOCHI_ASSERT_VERBOSE(
        !globalSol.empty(), "Solution-gathering assembly requires a non-empty global solution.");
  }

  // Select the L2G index array.
  [[maybe_unused]] constexpr int kExpectedStride = ElementT::kNumDofs * kNumFields;
  Span<int const> indicesFlat;
  if (l2g.HasPaddedIndices()) {
    MOCHI_ASSERT_VERBOSE(
        l2g.GetPaddedStride() == kExpectedStride,
        "Padded L2G stride must equal kNumDofs * kNumFields.");
    indicesFlat = l2g.GetPaddedGlobalIndices();
  } else {
    MOCHI_ASSERT_VERBOSE(
        l2g.GetElementSize(0) == kExpectedStride,
        "L2G requires uniform stride kNumDofs * kNumFields.");
    indicesFlat = l2g.GetGlobalIndices();
  }

  using BSpMatView = BlockSparseMatrixView<real, kNumFields>;
  BSpMatView outDRes{};
  if (results.params.assemDRes) {
    MOCHI_ASSERT_VERBOSE(
        std::holds_alternative<BSpMatView>(results.outDRes),
        "FEM assembler requires BlockSparseMatrix for dresidual.");
    outDRes = std::get<BSpMatView>(results.outDRes);
  }

  auto dispatcher = [&](auto kResTag, auto kDResTag) {
    constexpr bool kRes = decltype(kResTag)::value;
    constexpr bool kDRes = decltype(kDResTag)::value;
    details::DynamicLoadBalancingAssembly<
        ElementT,
        kNumFields,
        kBatchSize,
        kRes,
        kDRes,
        kGatherSolution>(
        indicesFlat,
        nbs,
        batchedElOp,
        globalSol,
        activeSubset.activeElementIndices,
        activeSubset.isElementActive,
        results.params.assemObj,
        results.params.psdDRes,
        results.outObj,
        results.outRes,
        outDRes);
  };

  using T = std::true_type;
  using F = std::false_type;
  auto const& p = results.params;
  if (p.assemRes && p.assemDRes) {
    dispatcher(T{}, T{});
  } else if (p.assemRes) {
    dispatcher(T{}, F{});
  } else if (p.assemDRes) {
    dispatcher(F{}, T{});
  } else {
    dispatcher(F{}, F{});
  }
}

} // namespace details

template <class ElementT, int kNumFields, int kBatchSize>
void AssembleObjResDRes(
    Local2GlobalMap const& l2g,
    NodalBasedStructure const& nbs,
    ElOpFnType<ElementT, kNumFields, kBatchSize> const& batchedElOp,
    ColumnVectorView<real const> globalSol,
    AssemblyResults<real> results,
    AssemblyActiveSubset const& activeSubset) {
  details::AssembleObjResDResImpl<ElementT, kNumFields, kBatchSize, /*kGatherSolution*/ true>(
      l2g, nbs, batchedElOp, globalSol, results, activeSubset);
}

template <class ElementT, int kNumFields, int kBatchSize>
void AssembleObjResDRes(
    Local2GlobalMap const& l2g,
    NodalBasedStructure const& nbs,
    NoDispElOpFnType<ElementT, kNumFields, kBatchSize> const& batchedElOp,
    AssemblyResults<real> results,
    AssemblyActiveSubset const& activeSubset) {
  details::AssembleObjResDResImpl<ElementT, kNumFields, kBatchSize, /*kGatherSolution*/ false>(
      l2g, nbs, batchedElOp, /*globalSol*/ {}, results, activeSubset);
}

namespace details {
template <
    class ElementT,
    int kNumFields,
    int kBatchSize,
    bool kGatherSolution,
    typename InitEleDResFn>
void AssembleAndProjectObjResDResImpl(
    Local2GlobalMap const& l2g,
    MaybeNoDispElOpFnType<ElementT, kNumFields, kBatchSize, kGatherSolution> const& batchedElOp,
    AssemblyActiveSubset const& activeSubset,
    InitEleDResFn const& initEleDResFn,
    ColumnVectorView<real const> globalSol,
    RowMatrixView<real const> J,
    double& outObj,
    ColumnVectorView<real> outRes,
    MatrixView<real> outDRes,
    AssemblyParams const& params) {
  MOCHI_PROFILE_SCOPE();
  static_assert(ElementT::kSpaceDim == 3, "Projected FEM assembler requires kSpaceDim == 3.");
  static_assert(kNumFields == 3, "Projected FEM assembler requires kNumFields == 3.");

  auto const& activeElementIndices = activeSubset.activeElementIndices;
#if MOCHI_ASSERT_VERBOSE_ENABLED
  MOCHI_ASSERT_VERBOSE(
      params.assemObj || params.assemRes || params.assemDRes, "Must assemble something.");
  MOCHI_ASSERT_VERBOSE(l2g.GetNumElements() > 0, "L2G map must not be empty.");
  for (int e = 0; e < l2g.GetNumElements(); ++e) {
    MOCHI_ASSERT_VERBOSE(
        l2g.GetElementSize(e) == ElementT::kNumDofs * kNumFields,
        "L2G must have uniform stride kNumDofs * kNumFields for all elements.");
  }
  MOCHI_ASSERT_VERBOSE(IsUnique(activeElementIndices), "Active elements must be unique.");
  MOCHI_ASSERT_VERBOSE(
      activeElementIndices.empty() || !activeSubset.isElementActive.empty(),
      "isElementActive must be provided when activeElementIndices is non-empty.");
  MOCHI_ASSERT_VERBOSE(
      activeSubset.isElementActive.empty() ||
          isize(activeSubset.isElementActive) == l2g.GetNumElements(),
      "isElementActive size must match the number of elements.");
  if constexpr (kGatherSolution) {
    MOCHI_ASSERT_VERBOSE(
        !globalSol.empty(), "Solution-gathering assembly requires a non-empty global solution.");
  }
#endif // MOCHI_ASSERT_VERBOSE_ENABLED

  static constexpr int kNumNodes = ElementT::kNumDofs;
  static constexpr int kNumEleDofs = kNumNodes * kNumFields;

  bool const useSubset = !activeElementIndices.empty();
  int const numElems = useSubset ? isize(activeElementIndices) : l2g.GetNumElements();
  Span<int const> indicesFlat = l2g.GetGlobalIndices();

  // TODO: kMinElemsPerWorker hasn't been tuned. The current value may be suboptimal.
  constexpr int kMinElemsPerWorker = 20; // At least ~20 µs per worker, assuming ~1 µs per element.
  int const numTasks =
      Max(1,
          Min((numElems + kMinElemsPerWorker - 1) / kMinElemsPerWorker,
              // 2x to improve load balancing.
              2 * (TaskScheduler::StaticGetNumOtherThreads() + 1)));

  // Per-worker objective, residual and dresidual. One worker directly assembles to the output
  // data. TODO: Cache-align per-thread accumulation buffers to reduce false sharing.
  DynamicArray<double> workerObj(params.assemObj ? numTasks - 1 : 0, 0.0);
  DynamicArray<ColumnVector<real>> workerRes(params.assemRes ? numTasks - 1 : 0);
  DynamicArray<Matrix<real>> workerDRes(params.assemDRes ? numTasks - 1 : 0);

  ParallelForN("AssembleAndProjectObjResDResWorkerTask", numTasks, 1, [&](int iTask) {
    int const loopBegin = (numElems * iTask) / numTasks;
    int const loopEnd = (numElems * (iTask + 1)) / numTasks;

    // The last task directly assembles to the output data. The rationale to select the last task
    // is that it's run by the same thread that invoked ParallelForN and thus only one thread
    // accesses the output data.
    bool const assembleToOutput = (iTask == numTasks - 1);

    // Initialize per-worker residual, dresidual and auxiliary data structures. 'Je' could be
    // precomputed at the cost of increasing memory footprint.
    RowMatrix<real, kNumEleDofs, krylov::kDynamic> Je;
    Matrix<real, kNumEleDofs, krylov::kDynamic> DeJe;
    if (params.assemRes && !assembleToOutput) {
      workerRes[iTask].Resize(outRes.Rows(), 1);
      workerRes[iTask].SetZero();
    }
    if (params.assemDRes) {
      if (!assembleToOutput) {
        workerDRes[iTask].Resize(outDRes.Rows(), outDRes.Cols());
        workerDRes[iTask].SetZero();
      }
      Je.Resize(kNumEleDofs, outDRes.Cols());
      DeJe.Resize(kNumEleDofs, outDRes.Cols());
    }

    // Per-batch objective, residual and dresidual.
    BatchDouble<kBatchSize> batchEnergy{};
    fem::BatchElementVector<kBatchSize, ElementT, kNumFields> batchRes{};
    fem::BatchElementMatrix<kBatchSize, ElementT, kNumFields> batchDRes{};

    // Per-element residual and dresidual.
    alignas(alignof(Simd<real>)) ColumnVector<real, kNumEleDofs> eleRes MOCHI_NO_INIT;
    alignas(alignof(Simd<real>)) RowMatrix<real, kNumEleDofs, kNumEleDofs> eleDRes MOCHI_NO_INIT;
    std::array<int, kNumNodes> nodes MOCHI_NO_INIT;

    for (int batchStart = loopBegin; batchStart < loopEnd; batchStart += kBatchSize) {
      // TODO: Potential optimizations:
      // 1. Within a batch, elements may share nodes. Instead of projecting each element
      // independently, identify the unique node set across the batch, scatter element dresiduals
      // into a shared matrix in the unique-node basis, and contract once with the deduplicated J
      // rows.
      // 2. Perform dresidual projection using batch matrix-matrix products. Early implementations
      // of a batch matrix-matrix kernel did not outperform the current per-element projection using
      // Mochi's GEMM.
      int const actualBatchSize = Min(kBatchSize, loopEnd - batchStart);

      NdArray<int, kBatchSize> batchElemIndices MOCHI_NO_INIT;
      for (int b = 0; b < kBatchSize; ++b) {
        int const i = batchStart + Min(b, actualBatchSize - 1);
        batchElemIndices[b] = useSubset ? activeElementIndices[i] : i;
      }

      if (params.assemObj) {
        batchEnergy = {};
      }
      if (params.assemRes) {
        batchRes = {};
      }
      if (params.assemDRes) {
        batchDRes = {};
      }

      bool const hasOutput = InvokeBatchedElOp<kGatherSolution, ElementT, kNumFields, kBatchSize>(
          batchedElOp,
          batchElemIndices,
          indicesFlat,
          globalSol,
          params.assemObj ? &batchEnergy : nullptr,
          params.assemRes ? &batchRes : nullptr,
          params.assemDRes ? &batchDRes : nullptr,
          params.psdDRes);

      // Determine whether to add the element contribution to the output.
      bool const addElemObj = params.assemObj && hasOutput;
      bool const addElemRes = params.assemRes && hasOutput;

      // Add objective.
      if (addElemObj) {
        auto* dstObj = assembleToOutput ? &outObj : &workerObj[iTask];
        for (int b = 0; b < actualBatchSize; ++b) {
          *dstObj += batchEnergy[b];
        }
      }

      // Project residual and dresidual.
      for (int b = 0; b < actualBatchSize; ++b) {
        int const eleIdx = batchElemIndices[b];

        // Set eleDRes via initEleDResFn, then add batch element op contribution.
        bool nonZeroInitDRes = false;
        if (params.assemDRes) {
          // Initialization of the element dresidual may be different for each element, e.g. if
          // initializing to the mass matrix.
          nonZeroInitDRes = initEleDResFn(eleDRes, eleIdx);
          MOCHI_ASSERT_VERBOSE(
              nonZeroInitDRes || eleDRes.NormSqr() == 0, "Inconsistent initialization.");
          if (hasOutput) {
            for (int i = 0; i < kNumEleDofs; ++i) {
              for (int j = 0; j < kNumEleDofs; ++j) {
                eleDRes(i, j) += batchDRes[i * kNumEleDofs + j][b];
              }
            }
          }
        }

        // NOTE: If all the element operations have zero dresidual, the dresidual projection could
        // be skipped. This is not an important use-case as of today but could be optimized in the
        // future.
        bool const addElemDRes = params.assemDRes && (nonZeroInitDRes || hasOutput);

        if (addElemDRes) {
          // If the dresidual needs to be assembled, it pays off to copy the element Jacobian to
          // consecutive memory and then project in a single & larger matrix-matrix product.
          l2g.GetElementNodes(eleIdx, nodes);
          for (int i = 0; i < kNumNodes; ++i) {
            Je.template MiddleRows<kNumFields>(kNumFields * i, kNumFields) =
                J.template MiddleRows<kNumFields>(kNumFields * nodes[i], kNumFields);
          }

          if (addElemRes) {
            for (int d = 0; d < kNumEleDofs; ++d) {
              eleRes[d] = batchRes[d][b];
            }
            auto dstRes = assembleToOutput ? outRes : AsView(workerRes[iTask]);
            dstRes += Je.Transpose() * eleRes;
          }

          auto dstDRes = assembleToOutput ? outDRes : AsView(workerDRes[iTask]);
          DeJe = eleDRes * Je; // Col-major = Row-major x Row-major
          dstDRes += Je.Transpose() * DeJe; // Col-major += Col-major x Col-major
        } else if (addElemRes) {
          // If the dresidual doesn't need to be assembled, it's faster to perform the residual
          // projection through a series of per-node matrix-vector products than to copy the element
          // Jacobian to consecutive memory and project in a single & larger matrix-vector product.
          for (int d = 0; d < kNumEleDofs; ++d) {
            eleRes[d] = batchRes[d][b];
          }
          l2g.GetElementNodes(eleIdx, nodes);
          auto dstRes = assembleToOutput ? outRes : AsView(workerRes[iTask]);
          for (int i = 0; i < kNumNodes; ++i) {
            dstRes +=
                J.Transpose().template MiddleCols<kNumFields>(kNumFields * nodes[i], kNumFields) *
                eleRes.template MiddleRows<kNumFields>(kNumFields * i, kNumFields);
          }
        }
      }
    }
  });

  // Add per-worker contributions. TODO: Could be parallelized with a tree approach.
  for (int iTask = 0; iTask < numTasks - 1; ++iTask) {
    if (params.assemObj) {
      outObj += workerObj[iTask];
    }
    if (params.assemRes) {
      outRes += workerRes[iTask];
    }
    if (params.assemDRes) {
      outDRes += workerDRes[iTask];
    }
  }
}

} // namespace details

template <class ElementT, int kNumFields, int kBatchSize, typename InitEleDResFn>
void AssembleAndProjectObjResDRes(
    Local2GlobalMap const& l2g,
    ElOpFnType<ElementT, kNumFields, kBatchSize> const& batchedElOp,
    AssemblyActiveSubset const& activeSubset,
    InitEleDResFn const& initEleDResFn,
    ColumnVectorView<real const> globalSol,
    RowMatrixView<real const> J,
    double& outObj,
    ColumnVectorView<real> outRes,
    MatrixView<real> outDRes,
    AssemblyParams const& params) {
  details::AssembleAndProjectObjResDResImpl<
      ElementT,
      kNumFields,
      kBatchSize,
      /*kGatherSolution*/ true>(
      l2g, batchedElOp, activeSubset, initEleDResFn, globalSol, J, outObj, outRes, outDRes, params);
}

template <class ElementT, int kNumFields, int kBatchSize, typename InitEleDResFn>
void AssembleAndProjectObjResDRes(
    Local2GlobalMap const& l2g,
    NoDispElOpFnType<ElementT, kNumFields, kBatchSize> const& batchedElOp,
    AssemblyActiveSubset const& activeSubset,
    InitEleDResFn const& initEleDResFn,
    RowMatrixView<real const> J,
    double& outObj,
    ColumnVectorView<real> outRes,
    MatrixView<real> outDRes,
    AssemblyParams const& params) {
  details::AssembleAndProjectObjResDResImpl<
      ElementT,
      kNumFields,
      kBatchSize,
      /*kGatherSolution*/ false>(
      l2g,
      batchedElOp,
      activeSubset,
      initEleDResFn,
      /*globalSol*/ {},
      J,
      outObj,
      outRes,
      outDRes,
      params);
}

} // namespace mochi
