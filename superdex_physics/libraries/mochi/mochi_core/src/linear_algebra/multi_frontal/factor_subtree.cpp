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

#include <mochi_core/linear_algebra/block_sparse_matrix.h>
#include <mochi_core/linear_algebra/multi_frontal/assembly_helper.h>
#include <mochi_core/linear_algebra/multi_frontal/elim_tree.h>
#include <mochi_core/linear_algebra/multi_frontal/factor_subtree.h>
#include <mochi_core/linear_algebra/multi_frontal/front_assembly.h>
#include <mochi_core/linear_algebra/multi_frontal/front_operations.h>
#include <mochi_core/linear_algebra/multi_frontal/front_stack.h>
#include <mochi_core/linear_algebra/multi_frontal/l_assembly.h>
#include <mochi_core/linear_algebra/multi_frontal/l_matrix.h>

namespace mochi {
template <int kDofsPerNode, typename Scalar, size_t kBlockSize>
void FactorSubtree(
    SymbolicEliminationTree const& tree,
    FrontalOrganizer const& organizer,
    LMatrix<Scalar, kBlockSize>& lMatrix,
    BlockSparseMatrixView<Scalar const, kDofsPerNode> const& A,
    AssemblyHelper& helper,
    int root,
    Span<Scalar> rootFrontSpace) {
  FrontStack<Scalar, kBlockSize, true> stack(tree, organizer, root, rootFrontSpace);
  // This number has to be larger or equal to the largest of the block sizes of Front and LMatrix.
  // Here they are equal.
  size_t rows = kBlockSize;
  FrontManipulator<Scalar, kBlockSize> fm(rows);
  for (auto supNd : tree.SubtreeRange(root)) {
    size_t snSize = kDofsPerNode * tree.SuperSize(supNd);
    auto snHeight = kDofsPerNode * tree.SuperColSize(supNd);
    // Leaf nodes won't be on the stack yet, but parents will be.
    auto space = stack.IsOnStack(supNd) ? stack.GetTopFront() : stack.PushFront(supNd);
    auto snFront = Front<kBlockSize, true>(space.end(), snHeight - snSize, false);

    auto snL = lMatrix.LforSN(supNd);
    bool isLeaf = tree.IsLeaf(supNd);
    AssembleSupernodeL(lMatrix, helper, A, supNd, isLeaf);
    // Factor the supernode's L
    fm.ToLD(snL);
    fm.RankNUpdate(snL, snFront, isLeaf);
    if (supNd == root) {
      break;
    }
    stack.PopFront(supNd);
    auto parent = tree.SuperParent(supNd);
    bool parentOnStack = stack.IsParentOnStack(supNd);
    auto parentSpace = parentOnStack ? stack.GetTopFront() : stack.PushFront(parent);
    auto parentL = lMatrix.LforSN(parent);
    auto parentFront = Front<kBlockSize>(parentSpace.end(), parentL.Rows() - parentL.Cols(), false);
    auto ranges = organizer.GetRangesInParent(supNd);
    if (!parentOnStack) {
      ExpandIntoParent<kDofsPerNode>(snFront, parentL, parentFront, ranges);
    } else {
      AssembleIntoParent<kDofsPerNode>(snFront, parentL, parentFront, ranges);
    }
  }
}

/** @brief Perform the forward elimination step in the multifrontal factorization.
 * @details Compute in place y <- D^-1 L^-1 y
 *
 * @tparam kDofsPerNode Number of DOFs per node
 * @tparam Scalar Scalar type
 * @tparam kBlockSize Block size for matrix operations
 * @param tree Elimination tree
 * @param lMatrix Lower triangular matrix
 * @param y [inout] Right-hand side vector
 * @param [in] z Temporary space vector holding part the lower part of super-nodal L*y
 */
template <int kDofsPerNode, typename Scalar, size_t kBlockSize>
void ForwardElimination(
    SymbolicEliminationTree const& tree,
    LMatrix<Scalar, kBlockSize>& lMatrix,
    ColumnVector<Scalar>& y,
    ColumnVector<Scalar>& z) {
  // The blocking size for working on L's diagonal block. Preferably a multiple of
  // SIMD size and of the GEMM kernel's number of columns.
  constexpr int kSubBlockSize = 24;
  // Forward elimination (L^-1 .) and division by D
  auto numSuper = tree.NumSuperNodes();
  int dofOffset = 0;
  for (int super = 0; super < numSuper; super++) {
    auto superL = lMatrix.LforSN(super);
    int subOffset = 0;
    for (auto block : superL.Blocks()) {
      // Separate diagonal block.
      auto diagBlock = block.TopRows(block.Cols());
      auto lowerBlock = block.BottomRows(block.Rows() - block.Cols());
      auto diagX = y.MiddleRows(dofOffset, block.Cols());
      // Account for the previous blocks' contributions.
      if (subOffset > 0) {
        diagX -= z.MiddleRows(subOffset, diagX.Rows());
      }
      // Apply diagonal L^-1
      using namespace blocking;
      PartDown<kSubBlockSize>(
          block.Cols(),
          [](auto&& X, auto&& L) {
            auto currentX = X(DiagRows);
            auto prevX = X(Above);
            if (prevX.Rows() > 0) {
              currentX -= L(DiagRows, Left) * prevX;
            }
            kernel::ApplyLm1OnLeft(L(DiagBlock), currentX);
          },
          diagX,
          diagBlock);
      bool isFirst = subOffset == 0;
      subOffset += block.Cols();
      dofOffset += block.Cols();
      MOCHI_ASSERT_VERBOSE(subOffset + lowerBlock.Rows() == superL.Rows());
      // Compute the forward propagation contribution.
      if (isFirst) {
        z.MiddleRows(subOffset, lowerBlock.Rows()) = lowerBlock * diagX;
      } else {
        z.MiddleRows(subOffset, lowerBlock.Rows()) += lowerBlock * diagX;
      }
      // Apply diagonal D^-1
      for (int i = 0; i < diagX.Rows(); i++) {
        diagX[i] *= diagBlock(i, i);
      }
    }
    // Scatter z to the rhs.
    auto lowerIndices = tree.LowerIndices(super);
    for (int iNd = 0; iNd < lowerIndices.size(); iNd++) {
      auto node = lowerIndices[iNd];
      y.template MiddleRows<kDofsPerNode>(kDofsPerNode * node, kDofsPerNode) -=
          z.template MiddleRows<kDofsPerNode>(kDofsPerNode * iNd + subOffset, kDofsPerNode);
    }
  }
}

/**
 * @brief Backward substitution computing y <- L^{-T} y
 *
 * @tparam kDofsPerNode Number of degrees of freedom per node.
 * @tparam Scalar Scalar type of the matrix and vectors.
 * @tparam kBlockSize Block size of the L matrix.
 * @param tree Elimination tree.
 * @param lMatrix L matrix also storing D on the diagonal.
 * @param y [inout] RHS on input, solution on output.
 * @param z Temporary vector to gather already computed results.
 */
template <int kDofsPerNode, typename Scalar, size_t kBlockSize>
void BackSubstitution(
    SymbolicEliminationTree const& tree,
    LMatrix<Scalar, kBlockSize>& lMatrix,
    ColumnVector<Scalar>& y,
    ColumnVector<Scalar>& z) {
  // The blocking size for working on L's diagonal block. Preferably a multiple of
  // SIMD size and of the GEMM kernel's number of columns.
  constexpr int kSubBlockSize = 24;
  // Forward elimination (L^-1 .) and division by D
  auto numSuper = tree.NumSuperNodes();
  int dofOffset = kDofsPerNode * tree.NumNodes();
  for (int super = numSuper; --super >= 0;) {
    auto superL = lMatrix.LforSN(super);
    // Gather z from rhs
    auto lowerIndices = tree.LowerIndices(super);
    int subOffset = kDofsPerNode * tree.SuperSize(super);

    for (int iNd = 0; iNd < lowerIndices.size(); iNd++) {
      auto node = lowerIndices[iNd];
      z.template MiddleRows<kDofsPerNode>(kDofsPerNode * iNd + subOffset, kDofsPerNode) =
          y.template MiddleRows<kDofsPerNode>(kDofsPerNode * node, kDofsPerNode);
    }
    for (int iBl = superL.NumBlocks(); iBl-- > 0;) {
      auto block = superL.Block(iBl);
      dofOffset -= block.Cols();
      // Separate diagonal block.
      auto diagBlock = block.TopRows(block.Cols());
      auto lowerBlock = block.BottomRows(block.Rows() - block.Cols());
      auto diagX = y.MiddleRows(dofOffset, block.Cols());
      using namespace blocking;
      // Compute the substitution from below the block.
      diagX -= lowerBlock.Transpose() * z.MiddleRows(subOffset, lowerBlock.Rows());

      // Apply diagonal L^-T
      PartUp<kSubBlockSize>(
          block.Cols(),
          [](auto&& X, auto&& L) {
            auto currentX = X(DiagRows);
            auto prevX = X(Below);
            if (prevX.Rows() > 0) {
              currentX -= L(Below, DiagCols).Transpose() * prevX;
            }
            kernel::ApplyLmtOnLeft(L(DiagBlock), currentX);
          },
          diagX,
          diagBlock);
      subOffset -= block.Cols();
      if (subOffset > 0) {
        z.MiddleRows(subOffset, diagX.Rows()) = diagX;
      }
    }
  }
}

template <int kDofsPerNode, typename Scalar, size_t kBlockSize>
void MultiFrontalSolveInPlace(
    SymbolicEliminationTree const& tree,
    [[maybe_unused]] FrontalOrganizer const& organizer,
    LMatrix<Scalar, kBlockSize>& lMatrix,
    Span<int const> order,
    ColumnVectorView<Scalar> x) {
  MOCHI_ASSERT_VERBOSE(x.Rows() % kDofsPerNode == 0);
  // Values in the solver's order
  ColumnVector<Scalar> y(x.Rows());
  // Apply renumbering to the RHS.
  for (int dof = 0, nd = 0; dof < x.Rows(); dof += kDofsPerNode) {
    auto origDof = kDofsPerNode * order[nd];
    y.template Block<kDofsPerNode, 1>(dof, 0, kDofsPerNode, 1) =
        x.template Block<kDofsPerNode, 1>(origDof, 0, kDofsPerNode, 1);
    ++nd;
  }
  ColumnVector<Scalar> tmp(x.Rows());
  // Apply D^-1 L^-1
  ForwardElimination<kDofsPerNode, Scalar, kBlockSize>(tree, lMatrix, y, tmp);
  // Backward subsitution. L^-T .
  BackSubstitution<kDofsPerNode, Scalar, kBlockSize>(tree, lMatrix, y, tmp);
  // Put the result in the original order.
  for (int dof = 0, nd = 0; dof < x.Rows(); dof += kDofsPerNode) {
    auto origDof = kDofsPerNode * order[nd];
    x.template Block<kDofsPerNode, 1>(origDof, 0, kDofsPerNode, 1) =
        y.template Block<kDofsPerNode, 1>(dof, 0, kDofsPerNode, 1);
    ++nd;
  }
}

// Instantiations for block size 1
template void FactorSubtree<1, float, 96>(
    SymbolicEliminationTree const& tree,
    FrontalOrganizer const& organizer,
    LMatrix<float, 96>& lMatrix,
    BlockSparseMatrixView<float const, 1> const& A,
    AssemblyHelper& helper,
    int root,
    Span<float> rootFrontSpace);

template void FactorSubtree<1, double, 96>(
    SymbolicEliminationTree const& tree,
    FrontalOrganizer const& organizer,
    LMatrix<double, 96>& lMatrix,
    BlockSparseMatrixView<double const, 1> const& A,
    AssemblyHelper& helper,
    int root,
    Span<double> rootFrontSpace);

template void MultiFrontalSolveInPlace<1, float, 96>(
    SymbolicEliminationTree const& tree,
    FrontalOrganizer const& organizer,
    LMatrix<float, 96>& lMatrix,
    Span<int const> order,
    ColumnVectorView<float> x);

template void MultiFrontalSolveInPlace<1, double, 96>(
    SymbolicEliminationTree const& tree,
    FrontalOrganizer const& organizer,
    LMatrix<double, 96>& lMatrix,
    Span<int const> order,
    ColumnVectorView<double> x);

// Instantiations for block size 3
template void FactorSubtree<3, double, 6 * 16>(
    SymbolicEliminationTree const& tree,
    FrontalOrganizer const& organizer,
    LMatrix<double, 6 * 16>& lMatrix,
    BlockSparseMatrixView<double const, 3> const& A,
    AssemblyHelper& helper,
    int root,
    Span<double> rootFrontSpace);

template void FactorSubtree<2, double, 6 * 16>(
    SymbolicEliminationTree const& tree,
    FrontalOrganizer const& organizer,
    LMatrix<double, 6 * 16>& lMatrix,
    BlockSparseMatrixView<double const, 2> const& A,
    AssemblyHelper& helper,
    int root,
    Span<double> rootFrontSpace);

template void FactorSubtree<3, float, 6 * 16>(
    SymbolicEliminationTree const& tree,
    FrontalOrganizer const& organizer,
    LMatrix<float, 6 * 16>& lMatrix,
    BlockSparseMatrixView<float const, 3> const& A,
    AssemblyHelper& helper,
    int root,
    Span<float> rootFrontSpace);

template void MultiFrontalSolveInPlace<3, double, 96>(
    SymbolicEliminationTree const& tree,
    FrontalOrganizer const& organizer,
    LMatrix<double, 96>& lMatrix,
    Span<int const> order,
    ColumnVectorView<double> x);

template void MultiFrontalSolveInPlace<3, float, 96>(
    SymbolicEliminationTree const& tree,
    FrontalOrganizer const& organizer,
    LMatrix<float, 96>& lMatrix,
    Span<int const> order,
    ColumnVectorView<float> x);

template void MultiFrontalSolveInPlace<3, double, 6>(
    SymbolicEliminationTree const& tree,
    FrontalOrganizer const& organizer,
    LMatrix<double, 6>& lMatrix,
    Span<int const> order,
    ColumnVectorView<double> x);

template void MultiFrontalSolveInPlace<3, float, 6>(
    SymbolicEliminationTree const& tree,
    FrontalOrganizer const& organizer,
    LMatrix<float, 6>& lMatrix,
    Span<int const> order,
    ColumnVectorView<float> x);

// Instantiations for block size 4
template void FactorSubtree<4, float, 96>(
    SymbolicEliminationTree const& tree,
    FrontalOrganizer const& organizer,
    LMatrix<float, 96>& lMatrix,
    BlockSparseMatrixView<float const, 4> const& A,
    AssemblyHelper& helper,
    int root,
    Span<float> rootFrontSpace);

template void FactorSubtree<4, double, 96>(
    SymbolicEliminationTree const& tree,
    FrontalOrganizer const& organizer,
    LMatrix<double, 96>& lMatrix,
    BlockSparseMatrixView<double const, 4> const& A,
    AssemblyHelper& helper,
    int root,
    Span<double> rootFrontSpace);

template void MultiFrontalSolveInPlace<4, float, 96>(
    SymbolicEliminationTree const& tree,
    FrontalOrganizer const& organizer,
    LMatrix<float, 96>& lMatrix,
    Span<int const> order,
    ColumnVectorView<float> x);

template void MultiFrontalSolveInPlace<4, double, 96>(
    SymbolicEliminationTree const& tree,
    FrontalOrganizer const& organizer,
    LMatrix<double, 96>& lMatrix,
    Span<int const> order,
    ColumnVectorView<double> x);

// The instantiations below are for testing.
template void FactorSubtree<3, double, 6>(
    SymbolicEliminationTree const& tree,
    FrontalOrganizer const& organizer,
    LMatrix<double, 6>& lMatrix,
    BlockSparseMatrixView<double const, 3> const& A,
    AssemblyHelper& helper,
    int root,
    Span<double> rootFrontSpace);

template void FactorSubtree<1, double, 6>(
    SymbolicEliminationTree const& tree,
    FrontalOrganizer const& organizer,
    LMatrix<double, 6>& lMatrix,
    BlockSparseMatrixView<double const, 1> const& A,
    AssemblyHelper& helper,
    int root,
    Span<double> rootFrontSpace);

template void FactorSubtree<3, float, 6>(
    SymbolicEliminationTree const& tree,
    FrontalOrganizer const& organizer,
    LMatrix<float, 6>& lMatrix,
    BlockSparseMatrixView<float const, 3> const& A,
    AssemblyHelper& helper,
    int root,
    Span<float> rootFrontSpace);

template void MultiFrontalSolveInPlace<1, double, 6>(
    SymbolicEliminationTree const& tree,
    FrontalOrganizer const& organizer,
    LMatrix<double, 6>& lMatrix,
    Span<int const> order,
    ColumnVectorView<double> x);

template void MultiFrontalSolveInPlace<2, double, 96>(
    SymbolicEliminationTree const& tree,
    FrontalOrganizer const& organizer,
    LMatrix<double, 96>& lMatrix,
    Span<int const> order,
    ColumnVectorView<double> x);

} // namespace mochi
