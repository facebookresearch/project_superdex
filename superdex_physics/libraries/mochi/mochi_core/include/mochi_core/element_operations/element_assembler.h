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
#include <mochi_core/linear_algebra/any_matrix.h>
#include <mochi_core/linear_algebra/block_sparse_matrix.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/sparse_matrix.h>
#include <mochi_core/linear_algebra/utils/assembly.h>
#include <mochi_core/mochi_config.h>
#include <mochi_core/utils/array_utils.h>
#include <mochi_core/utils/assembly_params.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/batch_config.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/local_to_global_map.h>
#include <mochi_core/utils/simd.h>
#include <mochi_core/utils/span.h>
#include <mochi_core/utils/time.h>

#include <functional>
#include <optional>
#include <vector>

namespace mochi {

/**
 * @brief Boundary-face connectivity, Local2GlobalMap, and NodalBasedStructure for boundary
 * assembly. Each boundary face maps to its parent tet's 4 nodes with volume node numbering,
 * enabling direct assembly into the volume system matrix.
 */
struct BoundaryAssemblyData {
  DynamicArray<Int4> connectivity;
  Local2GlobalMap l2g;
  NodalBasedStructure nbs;

  /**
   * @brief Construct from boundary traces and volume mesh connectivity.
   * @param traces Span of boundary trace elements. Must expose GetElementIndex().
   * @param volumeConnectivity Volume mesh element connectivity.
   * @param volumeNToN Node-to-node connectivity of the volume mesh.
   * @param numFields Number of fields per node.
   */
  template <typename TraceT>
  BoundaryAssemblyData(
      Span<TraceT const> traces,
      Span<Int4 const> volumeConnectivity,
      Graph<int, int> const& volumeNToN,
      int numFields)
      : connectivity(BuildConnectivity(traces, volumeConnectivity)),
        nbs(GraphFromRangeOfRanges<int, int>(connectivity), volumeNToN) {
    l2g.InitializeFromElementNodeConnectivity(connectivity, numFields);
  }

 private:
  template <typename TraceT>
  static DynamicArray<Int4> BuildConnectivity(
      Span<TraceT const> traces,
      Span<Int4 const> volumeConnectivity) {
    DynamicArray<Int4> conn;
    conn.reserve(traces.size());
    for (auto const& trace : traces) {
      conn.push_back(volumeConnectivity[trace.GetElementIndex()]);
    }
    return conn;
  }
};

/** @brief Helper structure to specify a subset of elements to assemble. */
struct AssemblyActiveSubset {
  /// Indices of the elements to assemble. If empty, all elements are assembled. Indices must be
  /// unique but not necessarily sorted.
  Span<int const> activeElementIndices = {};

  /// Element membership bitmap. If non-empty, isElementActive[i] is nonzero iff element i is in the
  /// active set. Must be consistent with activeElementIndices.
  Span<bool const> isElementActive = {};

  AssemblyActiveSubset() = default;
  AssemblyActiveSubset(Span<int const> activeElementIndicesIn, Span<bool const> isElementActiveIn)
      : activeElementIndices(activeElementIndicesIn), isElementActive(isElementActiveIn) {
#if MOCHI_ASSERT_VERBOSE_ENABLED
    MOCHI_ASSERT_VERBOSE(
        IsUnique(activeElementIndices), "activeElementIndices must not contain duplicates.");
    for (int e : activeElementIndices) {
      MOCHI_ASSERT_VERBOSE(
          e >= 0 && e < isize(isElementActive),
          "activeElementIndices contains index out of range of isElementActive.");
      MOCHI_ASSERT_VERBOSE(
          isElementActive[e],
          "activeElementIndices contains index not marked active in isElementActive.");
    }

    int activeCount = 0;
    for (bool c : isElementActive) {
      activeCount += c;
    }
    MOCHI_ASSERT_VERBOSE(
        activeCount == isize(activeElementIndices),
        "isElementActive has more active entries than activeElementIndices.");
#endif // MOCHI_ASSERT_VERBOSE_ENABLED
  }
};

/**
 * Helper structure to store where the assembly overwrites the results
 * and flags to turn on/off what the assembly needs to compute (merit, residual, dresidual).
 */
template <typename Scalar>
struct AssemblyResults {
  double* outObj = nullptr;
  ColumnVectorView<Scalar> outRes = {};
  AnyMatrixView<Scalar> outDRes = {};

  AssemblyParams const params; // this contains just bools, just copy it
};

namespace details {
/**
 * @brief Callback signature for batched element operations used by @ref AssembleObjResDRes and @ref
 * AssembleAndProjectObjResDRes.
 *
 * The caller composes one or more batched operations (e.g. @ref StressWork, @ref GravityWork) into
 * this functor. The assembler invokes it once per batch of elements.
 *
 * @param[in]  elementIndices  Element indices for each batch lane.
 * @param[in]  l2gFlat         Flat local-to-global DoF map.
 * @param[in]  disp            Batched displacement DoF vector (gathered from global solution).
 * @param[out] outEnergy       If non-null, accumulates per-element energy.
 * @param[out] outRes          If non-null, accumulates per-element residual.
 * @param[out] outDRes         If non-null, accumulates per-element dresidual.
 * @param[in]  projectPsd      Whether to project the dresidual to PSD.
 * @return true if any output was written.
 *
 * @see NoDispElOpFn
 */
// TODO(T264957520): Consider exploiting symmetry end-to-end by introducing
// BatchElementUpperTriangularMatrix and using it instead of BatchElementMatrix.
template <class ElementT, int kNumFields, int kBatchSize>
struct ElOpFn {
  using Type = std::function<bool(
      NdArray<int, kBatchSize> const&,
      Span<int const>,
      fem::BatchElementVector<kBatchSize, ElementT, kNumFields> const&,
      BatchDouble<kBatchSize>*,
      fem::BatchElementVector<kBatchSize, ElementT, kNumFields>*,
      fem::BatchElementMatrix<kBatchSize, ElementT, kNumFields>*,
      bool)>;
};

/**
 * @brief Callback signature for batched element operations that do not read displacement. Used by
 * @ref AssembleObjResDRes and @ref AssembleAndProjectObjResDRes.
 *
 * @see ElOpFn
 */
template <class ElementT, int kNumFields, int kBatchSize>
struct NoDispElOpFn {
  using Type = std::function<bool(
      NdArray<int, kBatchSize> const&,
      Span<int const>,
      BatchDouble<kBatchSize>*,
      fem::BatchElementVector<kBatchSize, ElementT, kNumFields>*,
      fem::BatchElementMatrix<kBatchSize, ElementT, kNumFields>*,
      bool)>;
};
} // namespace details

template <
    class ElementT,
    int kNumFields = ElementT::kSpaceDim,
    int kBatchSize = kDefaultFemBatchSize>
using ElOpFnType = typename details::ElOpFn<ElementT, kNumFields, kBatchSize>::Type;

template <
    class ElementT,
    int kNumFields = ElementT::kSpaceDim,
    int kBatchSize = kDefaultFemBatchSize>
using NoDispElOpFnType = typename details::NoDispElOpFn<ElementT, kNumFields, kBatchSize>::Type;

/**
 * @brief Assemble objective, residual, and/or dresidual using batched element operations.
 *
 * The caller provides a functor that composes one or more batched element operations (e.g. @ref
 * StressWork, @ref GravityWork).
 *
 * @tparam ElementT         Element type
 * @tparam kNumFields       Number of DoF fields per node (default: ElementT::kSpaceDim)
 * @tparam kBatchSize       Number of elements per batch (default: @ref kDefaultFemBatchSize)
 * @param[in]  l2g          Local-to-global DoF map. Must provide uniform-stride access for all
 *                          elements: either a raw L2G where all elements have exactly
 *                          ElementT::kNumDofs × kNumFields DoFs, or a padded L2G (via @ref
 *                          Local2GlobalMap::InitializePaddedIndices) with stride ElementT::kNumDofs
 *                          × kNumFields. Padded L2G is auto-detected and used when available.
 * @param[in]  nbs          Nodal-based structure. Must report @ref ElementT::kNumDofs nodes per
 *                          element. When the underlying raw connectivity has variable node counts
 *                          (e.g. shell bending stencils), the NBS must be padded to @ref
 *                          ElementT::kNumDofs nodes per element. Use @ref
 *                          BuildPaddedNodalBasedStructure to build the matching padded structure
 *                          from variable-width connectivity and stencil positions.
 * @param[in]  batchedElOp  User functor composing batched element operations
 * @param[in]  globalSol    Global solution vector
 * @param[in,out] results   Assembly outputs (objective, residual, dresidual) and request flags.
 *                          DRes must be BlockSparseMatrixView<real, kNumFields>.
 * @param[in]  activeSubset Active element subset (empty = all elements)
 *
 * @note Template parameter order is @p ElementT, @p kNumFields, @p kBatchSize so production callers
 * can omit both @p kNumFields and @p kBatchSize. (This differs from @ref fem::BatchElementVector /
 * @ref fem::BatchElementMatrix, which place @p kBatchSize first.)
 */
template <
    class ElementT,
    int kNumFields = ElementT::kSpaceDim,
    int kBatchSize = kDefaultFemBatchSize>
void AssembleObjResDRes(
    Local2GlobalMap const& l2g,
    NodalBasedStructure const& nbs,
    ElOpFnType<ElementT, kNumFields, kBatchSize> const& batchedElOp,
    ColumnVectorView<real const> globalSol,
    AssemblyResults<real> results,
    AssemblyActiveSubset const& activeSubset = {});

/**
 * @brief No-displacement overload of @ref AssembleObjResDRes for element ops that do not read the
 * displacement.
 */
template <
    class ElementT,
    int kNumFields = ElementT::kSpaceDim,
    int kBatchSize = kDefaultFemBatchSize>
void AssembleObjResDRes(
    Local2GlobalMap const& l2g,
    NodalBasedStructure const& nbs,
    NoDispElOpFnType<ElementT, kNumFields, kBatchSize> const& batchedElOp,
    AssemblyResults<real> results,
    AssemblyActiveSubset const& activeSubset = {});

/**
 * @brief Assemble objective, residual, and/or dresidual using batched element operations, and
 * project them onto a low-dimensional subspace. The projection is performed element-by-element.
 *
 * @tparam ElementT         Element type.
 * @tparam kNumFields       Number of DoF fields per node (default: ElementT::kSpaceDim).
 * @tparam kBatchSize       Batch size (default: @ref kDefaultFemBatchSize).
 * @tparam InitEleDResFn    Functor type for element dresidual initialization.
 *
 * @param[in]  l2g          Local-to-global DoF map. Must have uniform stride (ElementT::kNumDofs ×
 *                          kNumFields) for all elements.
 * @param[in]  batchedElOp  User functor composing batched element operations.
 * @param[in]  activeSubset Active element subset (empty = all elements).
 * @param[in]  initEleDResFn Functor to initialize the element dresidual matrix. Signature:
 *   (RowMatrix<real, kNumEleDofs, kNumEleDofs>&, int eleIdx) -> bool. Must return true if
 *   initialized to non-zero, false if initialized to zero.
 * @param[in]  globalSol    Global solution vector.
 * @param[in]  J            Jacobian matrix whose columns span the low-dimensional subspace.
 * @param[in,out] outObj    Objective (merit) value. Only accumulated if requested.
 * @param[in,out] outRes    Projected residual vector (reduced dimension). Only accumulated if
 *                          requested.
 * @param[in,out] outDRes   Projected dresidual matrix (reduced x reduced). Only accumulated if
 *                          requested.
 * @param[in]  params       Assembly parameters (which of obj/res/dres to compute).
 */
template <
    class ElementT,
    int kNumFields = ElementT::kSpaceDim,
    int kBatchSize = kDefaultFemBatchSize,
    typename InitEleDResFn>
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
    AssemblyParams const& params);

/**
 * @brief No-displacement overload of @ref AssembleAndProjectObjResDRes for element ops that do not
 * read the displacement.
 */
template <
    class ElementT,
    int kNumFields = ElementT::kSpaceDim,
    int kBatchSize = kDefaultFemBatchSize,
    typename InitEleDResFn>
void AssembleAndProjectObjResDRes(
    Local2GlobalMap const& l2g,
    NoDispElOpFnType<ElementT, kNumFields, kBatchSize> const& batchedElOp,
    AssemblyActiveSubset const& activeSubset,
    InitEleDResFn const& initEleDResFn,
    RowMatrixView<real const> J,
    double& outObj,
    ColumnVectorView<real> outRes,
    MatrixView<real> outDRes,
    AssemblyParams const& params);

} // namespace mochi

#include "element_assembler_inl.h"
