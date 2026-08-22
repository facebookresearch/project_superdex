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

#include <mochi_core/linear_algebra/block_sparse_matrix.h>
#include <mochi_core/linear_algebra/krylov/amg/coarsening.h>
#include <mochi_core/linear_algebra/sparse_matrix.h>
#include <mochi_core/linear_algebra/utils/matrix_concepts.h>
#include <mochi_core/utils/graph.h>
#include <mochi_core/utils/graph_views.h>
#include <mochi_core/utils/simd.h>

#include <limits>
#include <type_traits>
#include <utility>

namespace mochi::krylov {

template <typename Scalar, int kDofsPerNode>
class TransferOperator {
 protected:
  template <typename VectorIn, typename VectorOut>
  void ApplyOperatorToNodeRange(
      SparseMatrix<Scalar, int, int> const& Op,
      VectorIn const& X,
      VectorOut& Y,
      int nodeBegin,
      int nodeEnd) const {
    MOCHI_ASSERT_VERBOSE(
        (Y.Rows() == Op.Rows() * kDofsPerNode) && (X.Rows() == Op.Cols() * kDofsPerNode) &&
            (X.Cols() == Y.Cols()),
        "Inconsistent sizes.");
    MOCHI_ASSERT_VERBOSE(
        nodeBegin >= 0 && nodeEnd <= Op.Rows() && nodeBegin <= nodeEnd, "Invalid row range.");
    if (nodeBegin == nodeEnd) {
      return;
    }

    // At least ~200k FLOPs per task. Empirically chosen value.
    auto const numNodes = nodeEnd - nodeBegin;
    auto const minNodesPerTask = Clamp(
        static_cast<int>(
            static_cast<int64_t>(200000) * numNodes /
            Max(1,
                2 * (Op.Pointers()[nodeEnd] - Op.Pointers()[nodeBegin]) * kDofsPerNode * X.Cols())),
        1,
        numNodes);
    ParallelForRange(
        "TransferOperator",
        nodeBegin,
        nodeEnd,
        minNodesPerTask,
        numNodes,
        [&](int nodeBeginTask, int nodeEndTask) {
          using V4 = Simd<Scalar, 4>;
          if constexpr (
              kDofsPerNode == 3 && V4::kIsSupported && VectorIn::kIsColMajor &&
              VectorOut::kIsColMajor) {
            for (int c = 0; c < X.Cols(); ++c) {
              for (int r = nodeBeginTask; r < nodeEndTask; ++r) {
                auto const rowIndices = Op.Indices(r);
                auto const rowValues = Op.Values(r);
                int const numValues = isize(rowValues);
                int j = 0;
                V4 result[2] = {}; // Initializes to zero.
                // Batches of 2 blocks to reduce overhead due to latency of FMA instructions.
                for (; j + 2 <= numValues; j += 2) {
                  result[0] += rowValues[j + 0] *
                      Load<kDofsPerNode, V4>(&X(rowIndices[j + 0] * kDofsPerNode, c));
                  result[1] += rowValues[j + 1] *
                      Load<kDofsPerNode, V4>(&X(rowIndices[j + 1] * kDofsPerNode, c));
                }
                if (j < numValues) {
                  result[0] +=
                      rowValues[j] * Load<kDofsPerNode, V4>(&X(rowIndices[j] * kDofsPerNode, c));
                }
                Store<kDofsPerNode>(&Y(r * kDofsPerNode, c), result[0] + result[1]);
              }
            }
          } else {
            //--- Implementation independent of storage direction in X and Y.
            Y.MiddleRows(nodeBeginTask * kDofsPerNode, (nodeEndTask - nodeBeginTask) * kDofsPerNode)
                .SetZero();
            for (int r = nodeBeginTask; r < nodeEndTask; ++r) {
              auto const rowIndices = Op.Indices(r);
              auto const rowValues = Op.Values(r);
              auto Yblock = Y.template MiddleRows<kDofsPerNode>(r * kDofsPerNode, kDofsPerNode);
              for (int j = 0; j < rowIndices.size(); ++j) {
                Yblock += rowValues[j] *
                    X.template MiddleRows<kDofsPerNode>(rowIndices[j] * kDofsPerNode, kDofsPerNode);
              }
            }
          }
        });
  }

 public:
  /// @brief Restrict a vector X to the next coarser level
  template <typename VectorIn, typename VectorOut>
  void Restrict(VectorIn const& X, VectorOut& PtX) const {
    if constexpr (kDofsPerNode == 1) {
      Pt.Apply(X, PtX);
    } else {
      ApplyOperatorToNodeRange(Pt, X, PtX, 0, Pt.Rows());
    }
  }

  /// @brief Partial application of the restriction operator to the next coarser level.
  /// @note Only the nodes [nodeBegin, nodeEnd) of the output are computed.
  template <typename VectorIn, typename VectorOut>
  void RestrictToNodeRange(VectorIn const& X, VectorOut& PtX, int nodeBegin, int nodeEnd) const {
    if constexpr (kDofsPerNode == 1) {
      Pt.ApplyToRange(X, PtX, nodeBegin, nodeEnd);
    } else {
      ApplyOperatorToNodeRange(Pt, X, PtX, nodeBegin, nodeEnd);
    }
  }

  /// @brief Interpolate a vector X to the next finer level
  template <typename VectorIn, typename VectorOut>
  void Interpolate(VectorIn const& X, VectorOut& PX) const {
    if constexpr (kDofsPerNode == 1) {
      P.Apply(X, PX);
    } else {
      ApplyOperatorToNodeRange(P, X, PX, 0, P.Rows());
    }
  }

  /// @brief Partial application of the interpolation operator to the next finer level.
  /// @note Only the nodes [nodeBegin, nodeEnd) of the output are computed.
  template <typename VectorIn, typename VectorOut>
  void InterpolateToNodeRange(VectorIn const& X, VectorOut& PX, int nodeBegin, int nodeEnd) const {
    if constexpr (kDofsPerNode == 1) {
      P.ApplyToRange(X, PX, nodeBegin, nodeEnd);
    } else {
      ApplyOperatorToNodeRange(P, X, PX, nodeBegin, nodeEnd);
    }
  }

  /// @brief Sparse matrix representation of the interpolation among "nodes"
  SparseMatrix<Scalar, int, int> P;

  /// @brief Sparse matrix representation of the restriction among "nodes"
  SparseMatrix<Scalar, int, int> Pt;
};

template <typename Scalar, int kDofsPerNode = 3>
struct AMGLevel {
  TransferOperator<Scalar, kDofsPerNode> T;
  BlockSparseMatrix<Scalar, kDofsPerNode, int, int> PtA;
  BlockSparseMatrix<Scalar, kDofsPerNode, int, int> PtAP;
};

} // namespace mochi::krylov

namespace mochi::krylov::details {

/// @brief Routine to compute the numerical values of a sparse matrix-matrix product
///
/// @param[in] A Input sparse or block-sparse matrix
/// @param[in] B Input sparse or block-sparse matrix
/// @param[in,out] AB Output block sparse matrix whose sparsity pattern has been defined outside the
/// function
///
/// @note MatA and MatB are either `SparseMatrix` or `BlockSparseMatrix`, with at most one of them
/// being `SparseMatrix`
///
/// @note If A is `SparseMatrix`, the product is actually AB = kron(A, I_{kBlockSize}) * B
/// (i.e. number of rows of A = ( number of rows of AB ) / kBlockSize
/// and number of columns of A = ( number of rows of B ) / kBlockSize )
/// If B is `SparseMatrix`, the product is actually AB = A * kron(B, I_{kBlockSize})
/// (i.e. number of rows of B = ( number of columns of A ) / kBlockSize
/// and number of columns of B = ( number of columns of AB ) / kBlockSize )
/// When A and B are both `BlockSparseMatrix` objects, the product is actually AB = A * B
///
/// @note The routine assumes that the resulting matrix AB has already the correct sparsity pattern
template <typename MatA, typename MatB, typename ScalarAB, int kBlockSize>
void SparseMatProduct(
    MatA const& A,
    MatB const& B,
    BlockSparseMatrix<ScalarAB, kBlockSize, int, int>& AB) {
  //
  if constexpr ((IsBlockSparseMatrix<MatA>) && (IsBlockSparseMatrix<MatB>)) {
    static_assert(
        MatA::kBlockSize == kBlockSize && MatB::kBlockSize == kBlockSize, "Mismatched block sizes");
    MOCHI_ASSERT_VERBOSE(A.BlockCols() == B.BlockRows(), "Mismatched inner dimensions");
    MOCHI_ASSERT_VERBOSE(A.Rows() == AB.Rows(), "Mismatched number of rows");
    MOCHI_ASSERT_VERBOSE(B.Cols() == AB.Cols(), "Mismatched number of columns");
  } else if constexpr ((IsSparseMatrix<MatA>) && (IsBlockSparseMatrix<MatB>)) {
    static_assert(MatB::kBlockSize == kBlockSize, "Mismatched block sizes");
    MOCHI_ASSERT_VERBOSE(A.Cols() == B.BlockRows(), "Mismatched inner dimensions");
    MOCHI_ASSERT_VERBOSE(A.Rows() == AB.BlockRows(), "Mismatched number of rows");
    MOCHI_ASSERT_VERBOSE(B.Cols() == AB.Cols(), "Mismatched number of columns");
  } else if constexpr ((IsBlockSparseMatrix<MatA>) && (IsSparseMatrix<MatB>)) {
    static_assert(MatA::kBlockSize == kBlockSize, "Mismatched block sizes");
    MOCHI_ASSERT_VERBOSE(A.BlockCols() == B.Rows(), "Mismatched inner dimensions");
    MOCHI_ASSERT_VERBOSE(A.BlockRows() == AB.BlockRows(), "Mismatched number of rows");
    MOCHI_ASSERT_VERBOSE(B.Cols() == AB.BlockCols(), "Mismatched number of columns");
  }
  //
  using Idx = typename MatA::NonConstIdx;
  auto nBlockRows = AB.BlockRows();
  auto nItems = AB.NumNonZeroBlocks();
  [[maybe_unused]] Idx const dummyFlag = std::numeric_limits<Idx>::max();
  MOCHI_ASSERT_VERBOSE(AB.BlockCols() < dummyFlag, "Incompatible flag");
  // TODO Explore whether the 'minPerTask' formula remains appropriate
  ParallelForRange(
      "SparseMatProduct",
      /* rangeBegin */ 0,
      /* rangeEnd */ nBlockRows,
      // At least 250 non-zero blocks in AB per task.
      /* minPerTask */ Clamp<Idx>((250 * nBlockRows) / nItems, 1, nBlockRows),
      /* maxPerTask */ nBlockRows,
      [&](Idx rowBegin, Idx rowEnd) {
        // Offset of nodes in the current row of AB being formed.
        DynamicArray<Idx> ndOffset;
#if MOCHI_ASSERT_VERBOSE_ENABLED
        ndOffset.resize(AB.BlockCols(), dummyFlag);
#else
        ndOffset.resize_noinit(AB.BlockCols());
#endif
        for (Idx i = rowBegin; i < rowEnd; ++i) {
          auto rowNodes = AB.Indices(i);
          auto ABrowValues = AB.Values(i);
          ABrowValues.SetZero();
          for (Idx k = 0; k < rowNodes.size(); ++k) {
            ndOffset[rowNodes[k]] = k;
          }
          auto Arow = A.Values(i);
          auto ArowIndices = A.Indices(i);
          for (Idx k = 0; k < ArowIndices.size(); ++k) {
            auto const& Aval = Arow[k];
            auto const aCol = ArowIndices[k];
            auto const Brow = B.Values(aCol);
            auto const Bindices = B.Indices(aCol);
            for (Idx jj = 0; jj < Bindices.size(); ++jj) {
              MOCHI_ASSERT_VERBOSE(Bindices[jj] >= 0, "Out of bounds");
              MOCHI_ASSERT_VERBOSE(Bindices[jj] < isize(ndOffset), "Out of bounds");
              MOCHI_ASSERT_VERBOSE(ndOffset[Bindices[jj]] != dummyFlag, "Out of bounds");
              MOCHI_ASSERT_VERBOSE(ndOffset[Bindices[jj]] < rowNodes.size(), "Out of bounds");
              ABrowValues[ndOffset[Bindices[jj]]] += Aval * Brow[jj];
            }
          }
#if MOCHI_ASSERT_VERBOSE_ENABLED
          for (Idx k = 0; k < rowNodes.size(); ++k) {
            // Only for the above check.
            ndOffset[rowNodes[k]] = dummyFlag;
          }
#endif
        }
      });
}

/** @brief Perform one level of AMG coarsening.
 *
 * @param A The system matrix to coarsen.
 * @return
 */
template <
    typename Scalar,
    int kDofsPerNode,
    typename InputIdx,
    template <typename, typename...> typename Storage>
auto Coarsen(
    BlockSparseMatrix<Scalar, kDofsPerNode, InputIdx, InputIdx, Storage> const& A,
    std::remove_const_t<Scalar> prolongOmega) {
  using NonConstScalar = std::remove_const_t<Scalar>;
  using NonConstIdx = std::remove_const_t<InputIdx>;
  // Extract the node-to-node graph
  auto nToN = AsGraphView(A);
  // Create the tentative prolongation
  auto [partition, numAgg] = details::Aggregate(nToN);
  // Smoothen the prolongation
  auto P = Smoothing(A, numAgg, partition, prolongOmega);
  //
  auto Pt = Transpose(P);
  auto gPtA = Traverse(AsGraphView(Pt), nToN).SortTargets();
  BlockSparseMatrix<NonConstScalar, kDofsPerNode, NonConstIdx, NonConstIdx> PtA(
      A.BlockCols(), gPtA);
  SparseMatProduct(Pt, A, PtA);
  //
  auto gPtAP = Traverse(gPtA, AsGraphView(P)).SortTargets();
  BlockSparseMatrix<NonConstScalar, kDofsPerNode, NonConstIdx, NonConstIdx> PtAP(numAgg, gPtAP);
  SparseMatProduct(PtA, P, PtAP);
  //
  return AMGLevel<NonConstScalar, kDofsPerNode>{
      {std::move(P), std::move(Pt)}, std::move(PtA), std::move(PtAP)};
}

} // namespace mochi::krylov::details
