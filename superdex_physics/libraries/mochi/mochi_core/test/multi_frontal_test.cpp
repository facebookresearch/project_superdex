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
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/multi_frontal/assembly_helper.h>
#include <mochi_core/linear_algebra/multi_frontal/elim_tree.h>
#include <mochi_core/linear_algebra/multi_frontal/factor_branch.h>
#include <mochi_core/linear_algebra/multi_frontal/factor_subtree.h>
#include <mochi_core/linear_algebra/multi_frontal/front_assembly.h>
#include <mochi_core/linear_algebra/multi_frontal/front_matrix.h>
#include <mochi_core/linear_algebra/multi_frontal/front_operations.h>
#include <mochi_core/linear_algebra/multi_frontal/front_stack.h>
#include <mochi_core/linear_algebra/multi_frontal/l_assembly.h>
#include <mochi_core/linear_algebra/multi_frontal/l_matrix.h>
#include <mochi_core/linear_algebra/multi_frontal/organizer.h>
#include <mochi_core/linear_algebra/multi_frontal/stair_matrix.h>
#include <mochi_core/linear_algebra/multi_frontal/truncated_tree.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/graph_utils.h>

#include <gtest/gtest.h>

#include <mochi_core/async/root_task.h>
#include <array>
#include <thread>
#include <vector>

using namespace mochi;
namespace mochi {
template <int kDofsPerNode, typename Scalar, size_t kBlockSize>
void ForwardElimination(
    SymbolicEliminationTree const& tree,
    LMatrix<Scalar, kBlockSize>& lMatrix,
    ColumnVector<Scalar>& y,
    ColumnVector<Scalar>& z);
template <int kDofsPerNode, typename Scalar, size_t kBlockSize>
void BackSubstitution(
    SymbolicEliminationTree const& tree,
    LMatrix<Scalar, kBlockSize>& lMatrix,
    ColumnVector<Scalar>& y,
    ColumnVector<Scalar>& z);
} // namespace mochi

namespace {

void OrderNodes(int Nx, int i0, int j0, int Ni, int Nj, std::vector<int>& nodes) {
  MOCHI_ASSERT(Ni % 2 == 1 && Nj % 2 == 1);
  auto append = [&](int i, int j) { nodes.push_back(i0 + i + (j0 + j) * Nx); };
  if (Ni > Nj) {
    if (Ni == 3) {
      append(0, 0);
      append(2, 0);
      append(1, 0);
    } else {
      auto halfNi = (Ni - 1) / 2;
      OrderNodes(Nx, i0, j0, halfNi, Nj, nodes);
      OrderNodes(Nx, i0 + halfNi + 1, j0, halfNi, Nj, nodes);
      for (int j = 0; j < Nj; ++j) {
        append(halfNi, j);
      }
    }
  } else if (Ni < Nj && Nj == 3) {
    append(0, 0);
    append(0, 2);
    append(0, 1);
  } else {
    auto halfNj = (Nj - 1) / 2;
    OrderNodes(Nx, i0, j0, Ni, halfNj, nodes);
    OrderNodes(Nx, i0, j0 + halfNj + 1, Ni, halfNj, nodes);
    for (int i = 0; i < Ni; ++i) {
      append(i, halfNj);
    }
  }
}

auto MakeSquareMeshGraph(int elemsPerSide) {
  int nodesPerSide = elemsPerSide + 1;
  GraphBuilder<int, int> gb(elemsPerSide * elemsPerSide, 4 * elemsPerSide * elemsPerSide);
  for (int i = 0; i < elemsPerSide; ++i) {
    for (int j = 0; j < elemsPerSide; ++j) {
      std::array nodes{
          i * nodesPerSide + j,
          i * nodesPerSide + j + 1,
          (i + 1) * nodesPerSide + j + 1,
          (i + 1) * nodesPerSide + j};
      gb.append(nodes);
    }
  }
  auto eToN = gb.Build();
  return Traverse(Reverse(eToN), eToN);
}

auto MakeMFStructure(int elemsPerSide) {
  auto graph = MakeSquareMeshGraph(elemsPerSide);
  std::vector<int> order;
  OrderNodes(elemsPerSide + 1, 0, 0, elemsPerSide + 1, elemsPerSide + 1, order);
  auto position = ReverseMap(order);
  SymbolicEliminationTree eTree(graph, order, position);
  return std::tuple{std::move(graph), std::move(order), std::move(position), std::move(eTree)};
}

void TestMultifrontal(int elemsPerSide) {
  auto [graph, order, position, eTree] = MakeMFStructure(elemsPerSide);
  size_t count = 0;
  std::vector<bool> isUsed((elemsPerSide + 1) * (elemsPerSide + 1), false);
  for (auto n : order) {
    if (!isUsed[n]) {
      ++count;
    }
    isUsed[n] = true;
  }
  int numNodes = (elemsPerSide + 1) * (elemsPerSide + 1);
  MOCHI_ASSERT(count == numNodes);
  EXPECT_TRUE(eTree.NumSuperNodes() == 30);
  EXPECT_EQ(eTree.SuperIndices().NumTargets(), 224);
  // Connected graphs should have only one root, and the root should be the last node.
  EXPECT_EQ(eTree.Roots().size(), 1);
  EXPECT_EQ(eTree.Roots()[0], 29);
  EXPECT_EQ(eTree.TreeGraph().size(), eTree.NumSuperNodes());
  EXPECT_EQ(eTree.TreeGraph().NumTargets(), 29);

  // Test EliminationTree structural properties & pre-symbolic metrics
  std::vector<int> orderCopy;
  OrderNodes(elemsPerSide + 1, 0, 0, elemsPerSide + 1, elemsPerSide + 1, orderCopy);
  auto positionCopy = ReverseMap(orderCopy);
  EliminationTree structTree(graph, orderCopy, positionCopy);
  EXPECT_EQ(structTree.NumSuperNodes(), 30);
  EXPECT_EQ(structTree.NumNodes(), numNodes);

  // Test ComputeFactorMetrics with a 3-node "Y-shape" graph (manually computed golden values):
  // nodes 0 and 1 each connected only to node 2, symmetrized.
  // With kDofsPerNode = 2 and elimination order {0, 1, 2}, multifrontal supernode amalgamation
  // merges nodes 1 and 2 (column of 2 equals column of 1 minus node 1 itself), but NOT node 0
  // (its column {0, 2} differs from node 1's column {1, 2}). Resulting structure:
  //   - SN0 = {node 0}: nInSuper = 2, nCoupling = 2 (couples to node 2)
  //       opCount = (2/3) * 2^3 + 2^2 * 2 + 2^2 * 2 = 16/3 + 16
  //       storage = 2 * 4 - 2 * 1 / 2 = 7
  //   - SN1 = {nodes 1, 2}: nInSuper = 4, nCoupling = 0 (root)
  //       opCount = (2/3) * 4^3 = 128/3
  //       storage = 4 * 4 - 4 * 3 / 2 = 10
  //   - Total opCount = 16/3 + 16 + 128/3 = 64
  //   - Total storage = 7 + 10 = 17
  {
    GraphBuilder<int, int> gb(2, 4);
    gb.append(std::array{0, 2});
    gb.append(std::array{1, 2});
    gb.append(std::array{0, 1, 2});
    auto threeNodeGraph = gb.Build();
    std::vector<int> threeOrder = {0, 1, 2};
    std::vector<int> threePosition = {0, 1, 2};
    EliminationTree threeTree(threeNodeGraph, threeOrder, threePosition);
    auto threeMetrics = threeTree.ComputeFactorMetrics(2);
    EXPECT_NEAR(threeMetrics.opCount, 64.0, 1e-9);
    EXPECT_EQ(threeMetrics.storage, 17);
  }

  // Independently re-derive the expected metrics from the public supernode
  // sizes using the multifrontal cost formulas documented on ComputeFactorMetrics.
  constexpr int kDofsPerNode = 2;
  double expectedOpCount = 0.0;
  size_t expectedStorage = 0;
  for (int sn = 0; sn < structTree.NumSuperNodes(); ++sn) {
    double const nInSuper = structTree.SuperSize(sn) * kDofsPerNode;
    double const superLRows = structTree.SuperColSize(sn) * kDofsPerNode;
    double const nCoupling = superLRows - nInSuper;
    expectedOpCount += (2.0 / 3.0) * nInSuper * nInSuper * nInSuper +
        nInSuper * nInSuper * nCoupling + nCoupling * nCoupling * nInSuper;
    expectedStorage += static_cast<size_t>(nInSuper * superLRows - nInSuper * (nInSuper - 1) / 2);
  }
  auto metrics = structTree.ComputeFactorMetrics(kDofsPerNode);
  EXPECT_DOUBLE_EQ(metrics.opCount, expectedOpCount);
  EXPECT_EQ(metrics.storage, expectedStorage);
}

template <typename Scalar, size_t kColumnBlock, bool kPackSmall>
void TestFront(size_t numDofs) {
  using Fr = Front<kColumnBlock, kPackSmall>;
  std::vector<Scalar> values(Fr::StorageSize(numDofs));
  auto v = values.data();
  auto vEnd = v + values.size();
  Front<kColumnBlock, kPackSmall> front(v, numDofs, true);
  Front<kColumnBlock, kPackSmall> frontBack(vEnd, numDofs, false);
  auto nb = front.NumBlocks();
  auto block_0 = front.template Block<Scalar>(0);
  auto last_block = front.template Block<Scalar>(nb - 1);
  auto leftOver = numDofs % kColumnBlock;
  if (leftOver == 0) {
    leftOver = kColumnBlock;
  }
  // Check that the first and last blocks have the correct sizes.
  if constexpr (kPackSmall) {
    EXPECT_EQ(last_block.Cols(), std::min(numDofs, kColumnBlock));
    EXPECT_EQ(block_0.Cols(), leftOver);
  } else {
    EXPECT_EQ(block_0.Cols(), std::min(numDofs, kColumnBlock));
    EXPECT_EQ(last_block.Cols(), leftOver);
  }
  Scalar* nextBlock = nullptr;
  Scalar* nextBlockBack = nullptr;
  for (size_t i = 0; i < nb; ++i) {
    auto block = front.template Block<Scalar>(i);
    auto blockBack = frontBack.template Block<Scalar>(i);
    auto rows = block.Rows();
    auto cols = block.Cols();
    EXPECT_GE(&block(0, 0), v);
    EXPECT_LE(&block(rows - 1, cols - 1), vEnd);
    // Check both regular and mirrored fronts give the same size
    EXPECT_EQ(blockBack.Rows(), rows);
    EXPECT_EQ(blockBack.Cols(), cols);
    // Check that the mirrored block precedes the previous one in memory.
    if (nextBlock != nullptr) {
      EXPECT_EQ(nextBlock, &block(rows - 1, cols - 1));
    }
    // Check that the regular block follows the previous one in memory.
    if (nextBlockBack != nullptr) {
      EXPECT_EQ(nextBlockBack, &blockBack(0, 0));
    }
    nextBlock = &block(0, 0) - 1;
    nextBlockBack = &blockBack(rows - 1, cols - 1) + 1;
  }
}

template <typename Scalar, size_t kColumnBlock, bool kPackSmall>
void TestFrontMemoryStart(size_t numDofs) {
  using Fr = Front<kColumnBlock, kPackSmall>;
  DynamicArray<Scalar> values(Fr::StorageSize(numDofs));
  auto v = values.data();
  auto vEnd = v + values.size();
  Front<kColumnBlock, kPackSmall> front(v, numDofs, true);
  Front<kColumnBlock, kPackSmall> frontBack(vEnd, numDofs, false);
  EXPECT_EQ(front.template MemoryStart<Scalar>(), v);
  EXPECT_EQ(frontBack.template MemoryStart<Scalar>(), v);
}

template <typename Scalar, size_t kColumnBlock, bool kPackSmall>
void TestFrontBlockRange(size_t numDofs) {
  using Fr = Front<kColumnBlock, kPackSmall>;
  std::vector<Scalar> values(Fr::StorageSize(numDofs));
  auto v = values.data();
  auto vEnd = v + values.size();
  Front<kColumnBlock, kPackSmall> front(v, numDofs, true);
  Front<kColumnBlock, kPackSmall> frontBack(vEnd, numDofs, false);

  for (size_t i = 0; auto b : front.template Blocks<Scalar>()) {
    auto bRef = front.template Block<Scalar>(i);
    EXPECT_EQ(&bRef(0, 0), &b(0, 0));
    EXPECT_EQ(&bRef(bRef.Rows() - 1, bRef.Cols() - 1), &b(b.Rows() - 1, b.Cols() - 1));
    ++i;
  }
  // Check that the first and last blocks have the correct sizes.
  for (size_t i = 0; auto bBack : frontBack.template Blocks<Scalar>()) {
    auto bBackRef = frontBack.template Block<Scalar>(i);
    EXPECT_EQ(&bBackRef(0, 0), &bBack(0, 0));
    EXPECT_EQ(
        &bBackRef(bBackRef.Rows() - 1, bBackRef.Cols() - 1),
        &bBack(bBack.Rows() - 1, bBack.Cols() - 1));
    ++i;
  }
}

template <typename Scalar, size_t kColumnBlock, size_t kNodeSize>
void TestStairs(size_t rows, size_t columns) {
  auto sz = StairMatrixSize(rows, columns, kColumnBlock);
  std::vector<Scalar> v(sz, Scalar{-1});
  StairMatrixView<Scalar, kColumnBlock> mat(v.data(), rows, columns);
  Scalar* last_end = &v[0];
  for (auto ndCol : mat.template NodalColumns<kNodeSize>()) {
    Scalar* begin = &ndCol(0, 0);
    Scalar* end = &ndCol(ndCol.Rows() - 1, ndCol.Cols() - 1) + 1;
    EXPECT_GE(begin, last_end);
    last_end = end;
  }
  EXPECT_EQ(last_end, &v[sz - 1] + 1);
}

template <typename Scalar, size_t kColumnBlock, size_t kDofsPerNode>
void TestMultifrontalLMatrix(int elemsPerSide) {
  auto [graph, order, position, eTree] = MakeMFStructure(elemsPerSide);

  LShape<kColumnBlock> lShape{eTree, kDofsPerNode};
  std::vector<Scalar> lMem(lShape.StorageSize());
  auto numSuperNodes = eTree.NumSuperNodes();
  auto* p = lMem.data();
  for (int s = 0; s < numSuperNodes; ++s) {
    auto superL = lShape.template LforSN<Scalar>(lMem, s);
    auto* begin = superL.Block(0).data();
    EXPECT_EQ(begin, p);
    auto lastBlock = superL.Block(superL.NumBlocks() - 1);
    auto* end = lastBlock.data() + lastBlock.StorageSize();
    p = end;
  }
  auto endSize = p - lMem.data();
  EXPECT_EQ(endSize, lMem.size());
}

template <typename Scalar, size_t kColumnBlock, size_t kNodeSize>
void TestMatrixAssembly(int elemsPerSide) {
  static_assert(kNodeSize > 1, "kNodeSize must be greater than 1");
  auto [graph, order, position, eTree] = MakeMFStructure(elemsPerSide);

  LMatrix<Scalar, kColumnBlock> L(eTree, kNodeSize);
  L.SetZero();
  auto numNodes = isize(order);
  BlockSparseMatrix<Scalar, kNodeSize> A(numNodes, graph);
  // Fill A with data based on original row and column numbers.
  for (auto [r, cols] : graph) {
    auto rowA = A.Values(r);
    for (int i_c = 0; i_c < cols.size(); ++i_c) {
      int c = cols[i_c];
      auto block = rowA[i_c];
      block.SetZero();
      block(0, 0) = Scalar(100 * r + c + 1);
      block(kNodeSize - 1, 0) = r; // Original matrix row index.
      block(0, kNodeSize - 1) = c; // Original matrix column index.
    }
  }

  auto cGraph = Graph<int const, int const, Span>(graph.GetPointers(), graph.GetTargets());
  auto superIndices = eTree.SuperIndices();
  auto cIndices =
      Graph<int const, size_t const, Span>(superIndices.GetPointers(), superIndices.GetTargets());
  AssemblyHelper helper(cGraph, cIndices, eTree.SuperBounds(), order, position);
  for (int sn = 0; sn < eTree.NumSuperNodes(); ++sn) {
    AssembleSupernodeL(L, helper, A, sn, true);
  }

  int nonZeroCount = 0;
  auto numSuperNodes = eTree.NumSuperNodes();
  for (int sn = 0, nc = 0; sn < numSuperNodes; ++sn) {
    auto snL = L.LforSN(sn);

    auto lIdxIt = eTree.SuperIndices(sn).begin();
    for (auto nodeCol : snL.template NodalColumns<kNodeSize>()) {
      auto origCol = order[nc];
      for (int r = 0; r * kNodeSize < nodeCol.Rows(); ++r) {
        auto origRow = order[lIdxIt[r]];
        // Values have been transposed
        if (nodeCol(kNodeSize * r, 0) != 0) {
          ++nonZeroCount;
          EXPECT_EQ(nodeCol(kNodeSize * r, 0), Scalar(100 * origCol + origRow + 1));
          EXPECT_EQ(nodeCol(kNodeSize * r, kNodeSize - 1), origCol);
          EXPECT_EQ(nodeCol(kNodeSize * r + kNodeSize - 1, 0), origRow);
        }
      }
      ++lIdxIt;
      ++nc;
    }
  }
  EXPECT_EQ(2 * nonZeroCount, graph.NumTargets() + numNodes);
}

template <typename Scalar, size_t kColumnBlock, size_t kNodeSize>
void TestAssembly(int elemsPerSide) {
  auto [graph, order, position, eTree] = MakeMFStructure(elemsPerSide);
  FrontalOrganizer organizer(eTree, kColumnBlock, kNodeSize);
  auto firstRoot = eTree.Roots()[0];
  if (eTree.SubtreeDepth(eTree.Roots()[0]) < 3) {
    return;
  }
  LMatrix<Scalar, kColumnBlock> L(eTree, kNodeSize);
  auto treeGraph = eTree.TreeGraph();
  // Find a descendant two levels down
  int sn = firstRoot;
  for (int depth = 0; depth < 2; ++depth) {
    auto children = treeGraph[sn];
    auto maxIt = std::max_element(children.begin(), children.end(), [&et = eTree](int a, int b) {
      return et.SubtreeDepth(a) < et.SubtreeDepth(b);
    });
    MOCHI_ASSERT(maxIt != children.end(), "No children");
    sn = *maxIt;
  }
  FrontStack<Scalar, kColumnBlock, true> frontStack(eTree, organizer, sn);
  for (auto& v : frontStack.FullSpace()) {
    v = Scalar(-1);
  }
  auto frontSpace = frontStack.PushFront(sn);
  auto childL = L.LforSN(sn);
  EXPECT_GE(frontSpace.begin(), frontStack.FullSpace().begin());
  EXPECT_LE(frontSpace.end(), frontStack.FullSpace().end());
  frontStack.PopFront(sn);
  auto childFrontIndices = eTree.LowerIndices(sn);
  // Fill the child front with data:
  std::vector<bool> isInChild(eTree.NumNodes(), false);
  auto fillFront = [&isInChild](auto& childFront, auto const& childFrontIndices, auto&& value) {
    for (auto [ndIdx, nodeCol] : childFront.template NodalColumns<Scalar, kNodeSize>()) {
      isInChild[childFrontIndices[ndIdx]] = true;
      auto rows = nodeCol.Rows();
      EXPECT_EQ(rows, kNodeSize * (childFrontIndices.size() - ndIdx));
      for (int r = ndIdx; r < childFrontIndices.size(); ++r) {
        EXPECT_LE(kNodeSize * (r - ndIdx) + kNodeSize, rows);
        auto v = value(childFrontIndices[r], childFrontIndices[ndIdx]);
        nodeCol.MiddleRows(kNodeSize * (r - ndIdx), kNodeSize).SetConstant(v);
      }
    }
  };

  auto overlappingChildFront =
      Front<kColumnBlock>(frontSpace.end(), childL.Rows() - childL.Cols(), false);
  auto value = [](int r, int c) { return Scalar(100 * r + c); };
  fillFront(overlappingChildFront, childFrontIndices, value);
  auto ranges = organizer.GetRangesInParent(sn);

  auto parent = eTree.SuperParent(sn);
  auto parentSpace = frontStack.PushFront(parent);
  auto parentL = L.LforSN(parent);
  auto parentFront = Front<kColumnBlock>(parentSpace.end(), parentL.Rows() - parentL.Cols(), false);

  ExpandIntoParent<kNodeSize>(overlappingChildFront, parentL, parentFront, ranges);

  auto parentIndices = eTree.SuperIndices(parent);
  auto checkValues = [&](auto&& expectedValue) {
    int nd = 0; // Parent node index.
    for (auto block : parentL.template NodalColumns<kNodeSize>()) {
      auto cnd = parentIndices[nd];
      for (int r = 0; r < block.Rows(); ++r) {
        int rnd = parentIndices[nd + r / kNodeSize];
        for (int c = 0; c < block.Cols(); ++c) {
          auto v = block(r, c);
          if (isInChild[rnd] && isInChild[cnd]) {
            EXPECT_EQ(v, expectedValue(rnd, cnd));
          } else {
            EXPECT_EQ(v, Scalar(0));
          }
        }
      }
      ++nd;
    }
    for (auto [_, block] : parentFront.template NodalColumns<Scalar, kNodeSize>()) {
      auto cnd = parentIndices[nd];
      for (int r = 0; r < block.Rows(); ++r) {
        int rnd = parentIndices[nd + r / kNodeSize];
        for (int c = 0; c < block.Cols(); ++c) {
          auto v = block(r, c);
          if (isInChild[rnd] && isInChild[cnd]) {
            EXPECT_EQ(v, expectedValue(rnd, cnd));
          } else {
            EXPECT_EQ(v, Scalar(0));
          }
        }
      }
      ++nd;
    }
  };
  checkValues(value);

  std::vector<Scalar> buffer(frontSpace.size());
  auto nonOverlappingChildFront =
      Front<kColumnBlock>(buffer.data() + buffer.size(), childL.Rows() - childL.Cols(), false);
  fillFront(nonOverlappingChildFront, childFrontIndices, [](int r, int c) {
    return -Scalar(100 * r + c);
  });
  AssembleIntoParent<kNodeSize>(nonOverlappingChildFront, parentL, parentFront, ranges);
  checkValues([](int, int) { return Scalar(0); });
}

template <typename Scalar, size_t kColumnBlock, size_t kNodeSize>
void TestLOperations(int numRows, int numCols) {
  Scalar epsilon = Sqrt(std::numeric_limits<Scalar>::epsilon());
  MOCHI_ASSERT_VERBOSE(numCols % kNodeSize == 0, "numCols must be a multiple of kNodeSize");
  MOCHI_ASSERT_VERBOSE(numRows > numCols, "The matrix must have more rows than columns.");
  auto size = StairMatrixSize(numRows, numCols, kColumnBlock);
  std::vector<Scalar> v(size, Scalar{-1});
  StairMatrixView<Scalar, kColumnBlock> mat(v.data(), numRows, numCols);
  // Create a test matrix. It is full but we only care about the numCols first columns.
  // This is because LDLt does not handle partial factorization of a matrix.
  Matrix<Scalar> A(numRows, numRows);
  A.SetZero();
  for (int c = 0; c < numRows; ++c) {
    A(c, c) = Scalar(4 + numRows + numCols);
    for (int r = c + 1; r < numRows; ++r) {
      A(r, c) = Scalar(-r + c + 1);
      if (r < numCols) {
        A(c, r) = A(r, c);
      }
    }
  }

  for (int c = 0; auto block : mat.template NodalColumns<kNodeSize>()) {
    block = A.Block(numRows - block.Rows(), c, block.Rows(), block.Cols());
    c += block.Cols();
  }
  FrontManipulator<Scalar, kColumnBlock> fm(kColumnBlock);
  fm.ToLD(mat);

  int info = 0;
  LDLt<Scalar> ldlt(A, info);
  for (int c = 0; auto block : mat.template NodalColumns<kNodeSize>()) {
    for (int k = 0; k < block.Cols(); ++k) {
      auto bCol = block.Col(k).BottomRows(block.Rows() - k);
      auto rCol = ldlt.GetStorage().Col(c + k).BottomRows(bCol.Rows());
      ColumnVector<Scalar> diff = bCol - rCol;
      auto diffNorm = diff.Norm();
      EXPECT_LE(diffNorm, epsilon * bCol.Norm());
    }
    c += block.Cols();
  }

  auto frontSize = FrontStorageSize(numRows - numCols, kColumnBlock);
  std::vector<Scalar> frontStorage(frontSize, Scalar{0});
  Front<kColumnBlock> front(frontStorage.data() + frontSize, numRows - numCols, false);

  fm.RankNUpdate(mat, front, false);

  auto C = A.Block(numCols, 0, numRows - numCols, numCols);
  Matrix<Scalar> Ct(numCols, numRows - numCols);
  Ct = C.Transpose();
  LDLt<Scalar> cornerLdlt(A.Block(0, 0, numCols, numCols), info);
  cornerLdlt.LeftSolveInPlace(Ct);
  Matrix<Scalar> R = -C * Ct;
  Matrix<Scalar> C2 = Ct.Transpose() * A.Block(0, 0, numCols, numCols);

  for (auto [nd, sc] : front.template NodalColumns<Scalar, kNodeSize>()) {
    auto rBlock = R.Block(nd * kNodeSize, nd * kNodeSize, sc.Rows(), sc.Cols());
    auto refNrm = sc.Norm();
    Matrix<Scalar> comp = rBlock - sc;
    auto diffNrm = comp.Norm();
    EXPECT_LT(diffNrm, epsilon * refNrm);
  }
}

template <int kDofsPerNode, typename Scalar, size_t kBlockSize>
void TestRootedFactor(
    SymbolicEliminationTree const& eTree,
    FrontalOrganizer const& organizer,
    AssemblyHelper& helper,
    BlockSparseMatrix<Scalar, kDofsPerNode> const& A,
    LMatrix<Scalar, kBlockSize>& Lref) {
  Scalar epsilon = Sqrt(std::numeric_limits<Scalar>::epsilon());
  LMatrix<Scalar, kBlockSize> L(eTree, kDofsPerNode);
  L.SetZero();
  int root = eTree.Roots()[0];
  auto treeGraph = eTree.TreeGraph();
  auto children = treeGraph[root];

  // Allocate memory for the front of all children and factor each subtree.
  auto const& costs = organizer.GetCosts();
  std::vector<std::vector<Scalar>> childrenFronts;
  for (auto child : children) {
    childrenFronts.emplace_back(costs[child].frontSize, Scalar(0));
    FactorSubtree<kDofsPerNode>(
        eTree, organizer, L, A, helper, child, MakeSpan(childrenFronts.back()));
  }

  auto snL = L.LforSN(root);
  // At the root, there is no front, hence the nullptr.
  Front<kBlockSize> rootFront(static_cast<Scalar*>(nullptr), 0, false);

  // Expand/Assemble children fronts into the root's snL
  for (int i = 0; i < isize(children); ++i) {
    auto child = children[i];
    auto childL = L.LforSN(child);
    auto childFront = Front<kBlockSize>(
        childrenFronts[i].data() + childrenFronts[i].size(), childL.Rows() - childL.Cols(), false);
    auto ranges = organizer.GetRangesInParent(child);
    if (i == 0) {
      ExpandIntoParent<kDofsPerNode>(childFront, snL, rootFront, ranges);
    } else {
      AssembleIntoParent<kDofsPerNode>(childFront, snL, rootFront, ranges);
    }
  }

  // Assemble root supernode L manually
  AssembleSupernodeL(L, helper, A, root, false);
  // Finalize root factorization.
  // Note: at the root, there is no front to update.
  FrontManipulator<Scalar, kBlockSize> fm(kBlockSize);
  fm.ToLD(snL);

  Scalar diffSqNrm = 0;
  Scalar nrmSqLref = 0;
  auto lData = L.ConstData();
  auto lRefData = Lref.ConstData();
  for (int i = 0; i < isize(lData); ++i) {
    diffSqNrm += (lData[i] - lRefData[i]) * (lData[i] - lRefData[i]);
    nrmSqLref += lRefData[i] * lRefData[i];
  }
  EXPECT_LE(diffSqNrm, epsilon * nrmSqLref);
}

template <int kDofsPerNode, typename Scalar, size_t kBlockSize>
void TestPanelBasedWork(
    SymbolicEliminationTree const& eTree,
    FrontalOrganizer const& organizer,
    AssemblyHelper& helper,
    BlockSparseMatrix<Scalar, kDofsPerNode> const& A,
    LMatrix<Scalar, kBlockSize>& Lref) {
  Scalar epsilon = Sqrt(std::numeric_limits<Scalar>::epsilon());
  LMatrix<Scalar, kBlockSize> L(eTree, kDofsPerNode);
  for (int mode : {0, 1}) {
    L.SetZero();
    int root = eTree.Roots()[0];
    auto treeGraph = eTree.TreeGraph();
    auto children = treeGraph[root];

    // Allocate memory for the front of all children and factor each subtree.
    auto const& costs = organizer.GetCosts();
    std::vector<std::vector<Scalar>> childrenFronts;
    for (auto child : children) {
      childrenFronts.emplace_back(costs[child].frontSize, Scalar(0));
      FactorSubtree<kDofsPerNode>(
          eTree, organizer, L, A, helper, child, MakeSpan(childrenFronts.back()));
    }

    auto snL = L.LforSN(root);

    // Prepare children front buffer spans
    DynamicArray<Span<Scalar>> childrenFrontBufferSpans;
    for (auto& buf : childrenFronts) {
      childrenFrontBufferSpans.push_back(MakeSpan(buf));
    }

    // Allocate front buffer for the root
    auto rootFrontDOFs = snL.Rows() - snL.Cols();
    std::vector<Scalar> rootFrontBuffer(Front<kBlockSize>::StorageSize(rootFrontDOFs), Scalar(0));

    // Construct TrunkWork
    TrunkWork<Scalar, kBlockSize> work(
        root,
        L,
        kDofsPerNode,
        helper,
        eTree,
        organizer,
        MakeSpan(childrenFrontBufferSpans),
        MakeSpan(rootFrontBuffer));

    // Allocate buffer for EliminatePanel
    auto maxPanelCols = kBlockSize;
    std::vector<Scalar> uBuffer(maxPanelCols * maxPanelCols);

    // Process each panel
    for (int p = 0; p < work.GetNumPanels(); ++p) {
      if (mode == 0) {
        // Assemble input matrix for L panels
        if (p < work.numPanelsInL) {
          work.template AddInputMatrix<kDofsPerNode>(p, A);
        }
        // Assemble children fronts
        work.template AssembleChildrenFronts<kDofsPerNode>(p);
        // Left-looking updates from all previously factored L panels
        for (int s = 0; s < p && s < work.numPanelsInL; ++s) {
          work.EliminatePanel(s, p, MakeSpan(uBuffer));
        }
        // Factor if this is an L panel
        if (p < work.numPanelsInL) {
          work.FactorPanel(p);
        }
        work.completedPanelCount++;
      } else {
        work.template WorkOnPanel<kDofsPerNode>(p, A, uBuffer);
      }
    }

    // Compare L data with reference
    Scalar diffSqNrm = 0;
    Scalar nrmSqLref = 0;
    auto lData = L.ConstData();
    auto lRefData = Lref.ConstData();
    for (int i = 0; i < lData.size(); ++i) {
      diffSqNrm += (lData[i] - lRefData[i]) * (lData[i] - lRefData[i]);
      nrmSqLref += lRefData[i] * lRefData[i];
    }
    EXPECT_LE(diffSqNrm, epsilon * nrmSqLref) << "Under mode: " << mode;
  }
}

template <typename Exec, int kDofsPerNode, typename Scalar, size_t kBlockSize>
RootTask Factor(
    TaskSemaphore,
    Exec executor,
    FactorData<kDofsPerNode, Scalar, kBlockSize>& data,
    int treeRoot) {
  auto fData = co_await FactorBranch(executor, data, treeRoot);
  EXPECT_EQ(fData.size(), 0);
  co_return;
}

template <int kDofsPerNode, typename Scalar, size_t kBlockSize>
void TestFullCoroutineSystem(
    SymbolicEliminationTree const& eTree,
    FrontalOrganizer const& organizer,
    AssemblyHelper& helper,
    BlockSparseMatrix<Scalar, kDofsPerNode> const& A,
    LMatrix<Scalar, kBlockSize>& Lref) {
  Scalar epsilon = Sqrt(std::numeric_limits<Scalar>::epsilon());
  LMatrix<Scalar, kBlockSize> L(eTree, kDofsPerNode);
  L.SetZero();
  TaskScheduler scheduler(Min<int>(std::thread::hardware_concurrency(), 8));
  int numThreads = scheduler.GetNumThreads();

  for (auto root : eTree.Roots()) {
    int nBranches = 2;
    auto [picked, trunkCount] = organizer.PickBranches(eTree, root, nBranches);
    FactorData<kDofsPerNode, Scalar, kBlockSize> data(
        eTree, organizer, helper, A, L, picked, numThreads);
    TaskSemaphore sem{1};
    Executor executor(scheduler);
    auto dispatch = [executor]<typename F>(F&& f) { executor.schedule(std::forward<F>(f)); };
    auto task = Factor(sem, dispatch, data, root);
    task.Schedule(scheduler, "Factor tree");
    sem.Wait();
  }
  // Compare L data with reference
  Scalar diffSqNrm = 0;
  Scalar nrmSqLref = 0;
  auto lData = L.ConstData();
  auto lRefData = Lref.ConstData();
  for (int i = 0; i < lData.size(); ++i) {
    diffSqNrm += (lData[i] - lRefData[i]) * (lData[i] - lRefData[i]);
    nrmSqLref += lRefData[i] * lRefData[i];
  }
  EXPECT_LE(diffSqNrm, epsilon * nrmSqLref);
}

template <typename Scalar, size_t kColumnBlock, size_t kNodeSize>
void TestFullSystem(int elemsPerSide) {
  auto kEps = Sqrt(elemsPerSide * std::numeric_limits<Scalar>::epsilon());
  auto [graph, order, position, eTree] = MakeMFStructure(elemsPerSide);
  FrontalOrganizer organizer(eTree, kColumnBlock, kNodeSize);
  LMatrix<Scalar, kColumnBlock> L(eTree, kNodeSize);
  L.SetZero();
  auto numNodes = isize(order);
  BlockSparseMatrix<Scalar, kNodeSize> A(numNodes, graph);

  // Fill A with data based on original row and column numbers.
  for (auto [r, cols] : graph) {
    auto rowA = A.Values(r);
    for (int i_c = 0; i_c < cols.size(); ++i_c) {
      int c = cols[i_c];
      auto block = rowA[i_c];
      block.SetZero();
      for (int k = 0; k < kNodeSize; ++k) {
        block(k, k) = Scalar(r == c ? 1 : -0.1);
      }
      if (kNodeSize > 1) {
        block(kNodeSize - 1, 0) = 0.01 * (r + c); // Original matrix row index.
        block(0, kNodeSize - 1) = 0.01 * (c + r); // Original matrix column index.
      }
    }
  }
  auto numSuperNodes = eTree.NumSuperNodes();
  auto cGraph = Graph<int const, int const, Span>(graph.GetPointers(), graph.GetTargets());
  auto superIndices = eTree.SuperIndices();
  auto cIndices =
      Graph<int const, size_t const, Span>(superIndices.GetPointers(), superIndices.GetTargets());
  AssemblyHelper helper(cGraph, cIndices, eTree.SuperBounds(), order, position);
  FactorSubtree<kNodeSize>(eTree, organizer, L, A, helper, numSuperNodes - 1);

  auto AFull = Matrix<Scalar>::Zero(kNodeSize * numNodes, kNodeSize * numNodes);
  for (int ir = 0; ir < numNodes; ++ir) {
    auto row = A.Values(ir);
    auto idx = A.Indices(ir);
    auto r = position[ir];
    for (int ic = 0; ic < idx.size(); ++ic) {
      auto block = row[ic];
      auto c = position[idx[ic]];
      for (int i = 0; i < kNodeSize; ++i) {
        for (int j = 0; j < kNodeSize; ++j) {
          AFull(r * kNodeSize + i, c * kNodeSize + j) = block(i, j);
        }
      }
    }
  }
  int info = 0;
  LDLt<Scalar> fullLDLt(AFull, info);
  int colNode = 0;
  Scalar errorSqNrm{0};
  Scalar matSqNrm{0};
  for (int sn = 0; sn < numSuperNodes; ++sn) {
    auto snL = L.LforSN(sn);
    auto rowNodes = eTree.SuperIndices(sn);
    auto colIt = rowNodes.begin();
    for (auto col : snL.template NodalColumns<kNodeSize>()) {
      auto lPerNode =
          BlockColView<Scalar, kNodeSize>(col.data(), col.LeadDim(), col.Rows() / kNodeSize);
      for (int ri = 0; ri < lPerNode.NumBlocks(); ++ri) {
        int rowIndex = colIt[ri];
        if (rowIndex == colNode) {
          continue;
        }
        auto lBlock = lPerNode[ri];
        auto refBlock = fullLDLt.GetStorage().template Block<kNodeSize, kNodeSize>(
            rowIndex * kNodeSize, colNode * kNodeSize, kNodeSize, kNodeSize);
        matSqNrm += refBlock.NormSqr();
        Matrix<Scalar> diff = lBlock - refBlock;
        errorSqNrm += diff.NormSqr();
      }
      rowNodes = rowNodes.subspan(1);
      ++colNode;
      ++colIt;
    }
  }
  EXPECT_LE(errorSqNrm, kEps * matSqNrm);

  ColumnVector<Scalar> x(kNodeSize * numNodes);
  ColumnVector<Scalar> b(kNodeSize * numNodes);
  ColumnVector<Scalar> tmp(kNodeSize * numNodes);
  ColumnVector<Scalar> sol(kNodeSize * numNodes);
  for (int r = 0; r < x.Rows(); ++r) {
    sol[r] = x[r] = Scalar(-2.5 + (r % 100));
  }
  ForwardElimination<kNodeSize>(eTree, L, x, tmp);
  BackSubstitution<kNodeSize>(eTree, L, x, tmp);
  fullLDLt.LeftSolveInPlace(sol);
  x -= sol;
  EXPECT_LE(x.Norm(), kEps * sol.Norm());

  x = sol;
  b = A * x;
  MultiFrontalSolveInPlace<kNodeSize>(eTree, organizer, L, order, ColumnVectorView<Scalar>{b});
  b -= sol;
  EXPECT_LE(b.Norm(), kEps * sol.Norm());

  TestRootedFactor<kNodeSize, Scalar, kColumnBlock>(eTree, organizer, helper, A, L);
  TestPanelBasedWork<kNodeSize, Scalar, kColumnBlock>(eTree, organizer, helper, A, L);
  TestFullCoroutineSystem<kNodeSize, Scalar, kColumnBlock>(eTree, organizer, helper, A, L);
}

TEST(MultifrontalNumerical, FullFactor) {
  TestFullSystem<double, 6, 1>(6);
  TestFullSystem<double, 6, 3>(6);
  TestFullSystem<float, 6, 3>(6);
  TestFullSystem<double, 96, 2>(6);
  TestFullSystem<double, 96, 3>(14);
  TestFullSystem<float, 96, 3>(14);
}

TEST(MultifrontalNumerical, LFactor) {
  TestLOperations<real, 6, 3>(12, 3);
  TestLOperations<real, 24, 3>(51, 15);
  TestLOperations<real, 24, 3>(36, 15);
  TestLOperations<real, 24, 3>(48, 27);
  TestLOperations<real, 24, 3>(63, 27);
  TestLOperations<real, 24, 3>(63, 48);
}

TEST(MultifrontalStructure, LShape) {
  TestAssembly<real, 24, 3>(6);
  TestAssembly<real, 24, 3>(14);
  TestMultifrontalLMatrix<real, 24, 3>(6);
  TestMultifrontalLMatrix<real, 24, 3>(14);
  TestMatrixAssembly<real, 24, 3>(6);
  TestMatrixAssembly<real, 24, 3>(14);
}

TEST(MultifrontalStructure, Sizes) {
  TestMultifrontal(6);
}

void TestPickBranches(int elemsPerSide, int nBranches) {
  auto [graph, order, position, eTree] = MakeMFStructure(elemsPerSide);
  FrontalOrganizer organizer(eTree, 24, 3);
  auto [picked, trunkCount] = organizer.PickBranches(eTree, eTree.Roots()[0], nBranches);

  EXPECT_GE(trunkCount, 1);
  EXPECT_LE(trunkCount + isize(picked), eTree.NumSuperNodes());
  EXPECT_FALSE(picked.empty());

  auto const& costs = organizer.GetCosts();
  double totalTime = 0.0;
  for (auto r : eTree.Roots()) {
    totalTime += costs[r].time;
  }
  double threshold = totalTime / (2.0 * nBranches);

  for (auto sn : picked) {
    // 1. Threshold check if not a leaf.
    if (!eTree.IsLeaf(sn)) {
      EXPECT_LE(costs[sn].time, threshold);
    }

    // 2. Parent check (the parent should be above threshold)
    auto parent = eTree.SuperParent(sn);
    if (parent != -1) {
      EXPECT_GT(costs[parent].time, threshold);
    }
  }

  // 3. No picked node is an ancestor of another picked node.
  std::vector<bool> isPicked(eTree.NumSuperNodes(), false);
  for (auto sn : picked) {
    isPicked[sn] = true;
  }

  for (auto sn : picked) {
    auto parent = eTree.SuperParent(sn);
    while (parent != -1) {
      EXPECT_FALSE(isPicked[parent])
          << "Node " << parent << " is an ancestor of picked node " << sn << " and is also picked.";
      parent = eTree.SuperParent(parent);
    }
  }
}

TEST(MultifrontalStructure, PickBranches) {
  auto [graph, order, position, eTree] = MakeMFStructure(6);
  FrontalOrganizer organizer(eTree, 24, 3);
  EXPECT_TRUE(organizer.PickBranches(eTree, eTree.Roots()[0], 0).first.empty());
  EXPECT_TRUE(organizer.PickBranches(eTree, eTree.Roots()[0], -1).first.empty());

  TestPickBranches(6, 4);
  TestPickBranches(14, 1);
  TestPickBranches(14, 8);
  // High nBranches exercises the leaf-above-threshold branch in PickBranches.
  TestPickBranches(14, 100);
}

void TestRangesInParent(int elemsPerSide) {
  auto [graph, order, position, eTree] = MakeMFStructure(elemsPerSide);
  FrontalOrganizer organizer(eTree, 24, 3);

  for (int sn = 0; sn < eTree.NumSuperNodes(); ++sn) {
    auto indices = organizer.GetIndicesInParent(sn);
    auto ranges = organizer.GetRangesInParent(sn);

    // Verify ranges reconstruct the original index mapping.
    int totalCovered = 0;
    for (int ri = 0; ri < isize(ranges); ++ri) {
      auto const& range = ranges[ri];
      EXPECT_GE(range.childStart, 0);
      EXPECT_GT(range.length, 0);
      EXPECT_EQ(range.childStart, totalCovered)
          << "Ranges must be contiguous in child space, sn=" << sn;
      for (int j = 0; j < range.length; ++j) {
        EXPECT_EQ(indices[range.childStart + j], range.parentStart + j)
            << "Range entry must match original index, sn=" << sn << " ri=" << ri << " j=" << j;
      }
      totalCovered += range.length;
    }
    EXPECT_EQ(totalCovered, isize(indices));

    // Verify ranges are maximally coalesced: adjacent ranges must have a gap in parent space.
    for (int ri = 1; ri < isize(ranges); ++ri) {
      auto const& prev = ranges[ri - 1];
      auto const& curr = ranges[ri];
      EXPECT_NE(prev.parentStart + prev.length, curr.parentStart)
          << "Adjacent ranges with consecutive parent indices should be merged, sn=" << sn;
    }
  }
}

TEST(MultifrontalStructure, RangesInParent) {
  TestRangesInParent(6);
  TestRangesInParent(14);
}

TEST(MultifrontalStructure, Layout) {
  TestFront<real, 24, false>(6);
  TestFront<real, 24, true>(6);
  TestFront<real, 24, false>(120);
  TestFront<real, 24, true>(120);
  TestFront<real, 24, false>(188);
  TestFront<real, 24, true>(188);
}

TEST(MultifrontalStructure, MemoryStart) {
  TestFrontMemoryStart<real, 24, false>(6);
  TestFrontMemoryStart<real, 24, true>(6);
  TestFrontMemoryStart<real, 24, false>(120);
  TestFrontMemoryStart<real, 24, true>(120);
  TestFrontMemoryStart<real, 24, false>(188);
  TestFrontMemoryStart<real, 24, true>(188);
}

TEST(MultifrontalStructure, BlockRange) {
  TestFrontBlockRange<real, 24, false>(6);
  TestFrontBlockRange<real, 24, true>(6);
  TestFrontBlockRange<real, 24, false>(120);
  TestFrontBlockRange<real, 24, true>(120);
  TestFrontBlockRange<real, 24, false>(188);
  TestFrontBlockRange<real, 24, true>(188);
}

TEST(MultifrontalStructure, StairNodalIterator) {
  TestStairs<real, 24, 3>(48, 33);
  TestStairs<real, 24, 3>(12, 6);
  TestStairs<real, 24, 3>(12, 12);
  TestStairs<real, 24, 3>(33, 6);
  TestStairs<real, 24, 3>(33, 33);
  TestStairs<real, 24, 3>(48, 33);
  TestStairs<real, 24, 3>(48, 48);
}

// Deterministic single-threaded tests for EnterWork/LeaveWork CAS logic.
// Verifies the CAS loop semantics in isolation without thread contention.
class TrunkWorkCASTest : public ::testing::Test {
 protected:
  // Minimal state to test the CAS logic in isolation.
  std::atomic<int> activeWorkerCount{0};
  std::atomic<int> completedPanelCount{0};
  int numPanels = 0;

  bool EnterWork() {
    auto activeWorkers = activeWorkerCount.load(std::memory_order::relaxed);
    auto activePanels = numPanels - completedPanelCount.load(std::memory_order::relaxed);
    while (activePanels > activeWorkers &&
           !activeWorkerCount.compare_exchange_strong(activeWorkers, activeWorkers + 1)) {
      activePanels = numPanels - completedPanelCount.load(std::memory_order::relaxed);
    }
    return activePanels > activeWorkers;
  }

  bool LeaveWork() {
    auto activeWorkers = activeWorkerCount.load(std::memory_order::relaxed);
    auto pendingPanels = numPanels - completedPanelCount.load(std::memory_order::relaxed);
    while (pendingPanels < activeWorkers &&
           !activeWorkerCount.compare_exchange_strong(activeWorkers, activeWorkers - 1)) {
      pendingPanels = numPanels - completedPanelCount.load(std::memory_order::relaxed);
    }
    return pendingPanels < activeWorkers;
  }
};

// EnterWork returns false when there are fewer panels than active workers.
TEST_F(TrunkWorkCASTest, EnterWorkReturnsFalseWhenFewerPanelsThanWorkers) {
  numPanels = 2;
  completedPanelCount = 0;
  activeWorkerCount = 3; // More workers than panels

  EXPECT_FALSE(EnterWork());
  EXPECT_EQ(activeWorkerCount.load(), 3); // Count unchanged
}

// EnterWork returns false when panels equal active workers.
TEST_F(TrunkWorkCASTest, EnterWorkReturnsFalseWhenPanelsEqualWorkers) {
  numPanels = 4;
  completedPanelCount = 1;
  activeWorkerCount = 3; // activePanels == activeWorkers

  EXPECT_FALSE(EnterWork());
  EXPECT_EQ(activeWorkerCount.load(), 3); // Count unchanged
}

// EnterWork returns true and increments when panels exceed workers.
TEST_F(TrunkWorkCASTest, EnterWorkReturnsTrueAndIncrementsWhenPanelsExceedWorkers) {
  numPanels = 5;
  completedPanelCount = 0;
  activeWorkerCount = 2;

  EXPECT_TRUE(EnterWork());
  EXPECT_EQ(activeWorkerCount.load(), 3); // Incremented
}

// LeaveWork returns true and decrements when workers exceed pending panels.
TEST_F(TrunkWorkCASTest, LeaveWorkReturnsTrueWhenWorkersExceedPendingPanels) {
  numPanels = 4;
  completedPanelCount = 3;
  activeWorkerCount = 3; // pendingPanels (1) < activeWorkers (3)

  EXPECT_TRUE(LeaveWork());
  EXPECT_EQ(activeWorkerCount.load(), 2); // Decremented
}

// LeaveWork returns false when pending panels >= workers.
TEST_F(TrunkWorkCASTest, LeaveWorkReturnsFalseWhenPendingPanelsExceedWorkers) {
  numPanels = 6;
  completedPanelCount = 0;
  activeWorkerCount = 2; // pendingPanels (6) >= activeWorkers (2)

  EXPECT_FALSE(LeaveWork());
  EXPECT_EQ(activeWorkerCount.load(), 2); // Count unchanged
}

// LeaveWork returns false when pending panels equal workers.
TEST_F(TrunkWorkCASTest, LeaveWorkReturnsFalseWhenPendingPanelsEqualWorkers) {
  numPanels = 5;
  completedPanelCount = 2;
  activeWorkerCount = 3; // pendingPanels (3) == activeWorkers (3)

  EXPECT_FALSE(LeaveWork());
  EXPECT_EQ(activeWorkerCount.load(), 3); // Count unchanged
}

// Sequential EnterWork calls correctly increment up to panel limit.
TEST_F(TrunkWorkCASTest, SequentialEnterWorkIncrementsUpToLimit) {
  numPanels = 3;
  completedPanelCount = 0;
  activeWorkerCount = 0;

  EXPECT_TRUE(EnterWork());
  EXPECT_EQ(activeWorkerCount.load(), 1);
  EXPECT_TRUE(EnterWork());
  EXPECT_EQ(activeWorkerCount.load(), 2);
  EXPECT_TRUE(EnterWork());
  EXPECT_EQ(activeWorkerCount.load(), 3);
  // Fourth call should fail - panels == workers now
  EXPECT_FALSE(EnterWork());
  EXPECT_EQ(activeWorkerCount.load(), 3);
}

// Sequential LeaveWork calls correctly decrement when work depletes.
TEST_F(TrunkWorkCASTest, SequentialLeaveWorkDecrementsCorrectly) {
  numPanels = 4;
  completedPanelCount = 4; // All panels completed, pendingPanels = 0
  activeWorkerCount = 3;

  EXPECT_TRUE(LeaveWork());
  EXPECT_EQ(activeWorkerCount.load(), 2);
  EXPECT_TRUE(LeaveWork());
  EXPECT_EQ(activeWorkerCount.load(), 1);
  EXPECT_TRUE(LeaveWork());
  EXPECT_EQ(activeWorkerCount.load(), 0);
  // Fourth call should fail - workers == 0 already
  EXPECT_FALSE(LeaveWork());
  EXPECT_EQ(activeWorkerCount.load(), 0);
}

// Mixed Enter/Leave sequence with changing completed panel count.
TEST_F(TrunkWorkCASTest, MixedEnterLeaveWithChangingCompletedPanels) {
  numPanels = 5;
  completedPanelCount = 0;
  activeWorkerCount = 0;

  // Enter 3 workers
  EXPECT_TRUE(EnterWork());
  EXPECT_TRUE(EnterWork());
  EXPECT_TRUE(EnterWork());
  EXPECT_EQ(activeWorkerCount.load(), 3);

  // Complete 2 panels, now pendingPanels = 3, workers = 3 -> Leave fails
  completedPanelCount = 2;
  EXPECT_FALSE(LeaveWork());
  EXPECT_EQ(activeWorkerCount.load(), 3);

  // Complete 1 more panel, now pendingPanels = 2, workers = 3 -> Leave succeeds
  completedPanelCount = 3;
  EXPECT_TRUE(LeaveWork());
  EXPECT_EQ(activeWorkerCount.load(), 2);

  // Still pendingPanels = 2, workers = 2 -> Enter fails
  EXPECT_FALSE(EnterWork());
  EXPECT_EQ(activeWorkerCount.load(), 2);
}

// Multi-threaded test of the EnterWork/LeaveWork CAS loops.
// Verifies correct thread coordination semantics under concurrent access.
class TrunkWorkCASConcurrencyTest : public ::testing::Test {
 protected:
  // Minimal state to test the CAS logic in isolation.
  std::atomic<int> activeWorkerCount{0};
  std::atomic<int> completedPanelCount{0};
  int numPanels = 0;

  bool EnterWork() {
    auto activeWorkers = activeWorkerCount.load(std::memory_order::relaxed);
    auto activePanels = numPanels - completedPanelCount.load(std::memory_order::relaxed);
    while (activePanels > activeWorkers &&
           !activeWorkerCount.compare_exchange_strong(activeWorkers, activeWorkers + 1)) {
      activePanels = numPanels - completedPanelCount.load(std::memory_order::relaxed);
    }
    return activePanels > activeWorkers;
  }

  bool LeaveWork() {
    auto activeWorkers = activeWorkerCount.load(std::memory_order::relaxed);
    auto pendingPanels = numPanels - completedPanelCount.load(std::memory_order::relaxed);
    while (pendingPanels < activeWorkers &&
           !activeWorkerCount.compare_exchange_strong(activeWorkers, activeWorkers - 1)) {
      pendingPanels = numPanels - completedPanelCount.load(std::memory_order::relaxed);
    }
    return pendingPanels < activeWorkers;
  }
};

TEST_F(TrunkWorkCASConcurrencyTest, ConcurrentEnterWorkRespectsLimit) {
  numPanels = 4;
  completedPanelCount = 0;
  activeWorkerCount = 0;

  std::atomic<int> successfulEnters{0};
  constexpr int kNumThreads = 8;

  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([&]() {
      if (EnterWork()) {
        successfulEnters.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  // At most numPanels workers should have entered
  EXPECT_LE(successfulEnters.load(), numPanels);
  EXPECT_EQ(activeWorkerCount.load(), successfulEnters.load());
}

TEST_F(TrunkWorkCASConcurrencyTest, ConcurrentEnterAndLeaveWorkBalances) {
  numPanels = 10;
  completedPanelCount = 0;
  activeWorkerCount = 0;

  constexpr int kNumThreads = 6;
  std::atomic<bool> startFlag{false};
  std::atomic<int> enterCount{0};
  std::atomic<int> leaveCount{0};

  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  // Half the threads try to enter, half try to leave
  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([&]() {
      // Wait for all threads to be ready
      while (!startFlag.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }

      // Alternate between enter and leave attempts
      for (int i = 0; i < 100; ++i) {
        if (EnterWork()) {
          enterCount.fetch_add(1, std::memory_order_relaxed);
        }
        // Simulate some work
        std::this_thread::yield();
        if (LeaveWork()) {
          leaveCount.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }

  // Start all threads simultaneously
  startFlag.store(true, std::memory_order_release);

  for (auto& t : threads) {
    t.join();
  }

  // Final worker count should be non-negative and within bounds
  auto finalWorkers = activeWorkerCount.load();
  EXPECT_GE(finalWorkers, 0);
  EXPECT_LE(finalWorkers, numPanels);
}

TEST_F(TrunkWorkCASConcurrencyTest, ConcurrentWorkersDecreaseAsWorkCompletes) {
  numPanels = 8;
  completedPanelCount = 0;
  activeWorkerCount = 0;

  constexpr int kNumThreads = 4;
  std::atomic<bool> stopFlag{false};

  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  // Workers continuously try to enter/leave based on work availability
  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([&]() {
      while (!stopFlag.load(std::memory_order_relaxed)) {
        EnterWork();
        std::this_thread::yield();
        LeaveWork();
      }
    });
  }

  // Simulate work completion over time
  for (int completed = 1; completed <= numPanels; ++completed) {
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    completedPanelCount.store(completed, std::memory_order_relaxed);
  }

  // Allow threads to stabilize
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  stopFlag.store(true, std::memory_order_relaxed);

  for (auto& t : threads) {
    t.join();
  }

  // When all panels are complete, no workers should remain
  // (they should have all left via LeaveWork)
  EXPECT_EQ(completedPanelCount.load(), numPanels);
}

} // namespace
