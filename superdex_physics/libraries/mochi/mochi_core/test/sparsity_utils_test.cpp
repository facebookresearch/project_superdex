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
#include <mochi_core/linear_algebra/block_sparse_matrix_utils.h>
#include <mochi_core/linear_algebra/sparse_matrix.h>
#include <mochi_core/linear_algebra/sparse_matrix_utils.h>
#include <mochi_core/linear_algebra/utils/matrix_conversions.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/sparsity_utils.h>
#include <mochi_core/utils/spmat_utils.h>

#include <algorithm>
#include <limits>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace mochi;

TEST(SparsityUtils, MakeSparsityGraphFromCoords) {
  // |xx |
  // |xxx|
  // |x x|

  // Coordinates out-of-order, with duplicates
  auto graph = MakeSparsityGraph(
      std::vector<Int2>{
          {0, 0}, {0, 1}, {2, 2}, {1, 1}, {2, 0}, {1, 2}, {0, 0}, {1, 0}, {2, 2}, {1, 1}, {0, 0}});

  EXPECT_EQ(3, graph.size());
  EXPECT_TRUE(test::EqualSpan(std::vector<int>{0, 2, 5, 7}, graph.GetPointers()));
  EXPECT_TRUE(test::EqualSpan(std::vector<int>{0, 1, 0, 1, 2, 0, 2}, graph.GetTargets()));
}

TEST(SparsityUtils, MakeSparsityGraphFromCoordsWithEmptyRows) {
  //   0123
  // 0|xx--|
  // 1|----| // intentionally empty
  // 2|x-x-|
  // 3|x---|
  // 4|x-xx|

  // Coordinates out-of-order, with duplicates
  auto graph = MakeSparsityGraph(
      std::vector<Int2>{{0, 0}, {0, 1}, {2, 2}, {2, 0}, {2, 2}, {3, 0}, {4, 0}, {4, 3}, {4, 2}});

  EXPECT_EQ(5, graph.size());
  EXPECT_TRUE(test::EqualSpan(std::vector<int>{0, 2, 2, 4, 5, 8}, graph.GetPointers()));
  EXPECT_TRUE(test::EqualSpan(std::vector<int>{0, 1, 0, 2, 0, 0, 2, 3}, graph.GetTargets()));
}

TEST(SparsityUtils, MakeSparsityGraphFromCoordsWithEmptyRowsAndSpecifiedNumRows) {
  //   0123
  // 0|xx--|
  // 1|----| // intentionally empty
  // 2|x-x-|
  // 3|x---|
  // 4|----| // intentionally empty
  // 5|----| // intentionally empty

  // Coordinates out-of-order, with duplicates
  int const forcedNumRows = 6;
  auto graph = MakeSparsityGraph(
      std::vector<Int2>{{0, 0}, {0, 1}, {2, 2}, {2, 0}, {3, 0}, {2, 2}}, forcedNumRows);

  EXPECT_EQ(forcedNumRows, graph.size());
  EXPECT_TRUE(test::EqualSpan(std::vector<int>{0, 2, 2, 4, 5, 5, 5}, graph.GetPointers()));
  EXPECT_TRUE(test::EqualSpan(std::vector<int>{0, 1, 0, 2, 0}, graph.GetTargets()));
}

TEST(SparsityUtils, MakeSparsityGraphFromCoordsMatrixFormat) {
  // this is the same test above, but testing the other API

  // Coordinates with duplicates
  using v_int_t = std::vector<int>;
  std::vector<v_int_t> coords = {v_int_t{0, 1, 0, 0}, v_int_t{1, 2, 0, 1}, v_int_t{2, 0, 2}};
  auto graph = MakeSparsityGraph(std::move(coords));

  EXPECT_EQ(3, graph.size());
  EXPECT_TRUE(test::EqualSpan(std::vector<int>{0, 2, 5, 7}, graph.GetPointers()));
  EXPECT_TRUE(test::EqualSpan(std::vector<int>{0, 1, 0, 1, 2, 0, 2}, graph.GetTargets()));
}

TEST(SparsityUtils, MakeSparsityGraphFromCoordsMatrixFormat2) {
  /*
     012345
   0|xx--x-|
   1|x-x---|
   2|x---x-|
   3|--x-xx|
   4|xx--x-|
   5|----x-|
  */

  // intentionally put this out-of-order and with duplicates
  using v_int_t = std::vector<int>;
  std::vector<v_int_t> coords = {
      v_int_t{0, 1, 0, 0, 4, 4, 1, 1, 1}, // row 0
      v_int_t{2, 0, 0, 2, 2, 2, 2}, // row 1
      v_int_t{4, 0, 4, 0}, // row 2
      v_int_t{5, 4, 2, 4, 2}, // row 3
      v_int_t{1, 0, 4}, // row 4
      v_int_t{4, 4, 4} // row 5
  };

  auto graph = MakeSparsityGraph(std::move(coords));

  EXPECT_EQ(6, graph.size());
  std::vector<int> const goldPtrs = {0, 3, 5, 7, 10, 13, 14};
  EXPECT_TRUE(test::EqualSpan(goldPtrs, graph.GetPointers()));

  std::vector<int> const goldTargets = {0, 1, 4, 0, 2, 0, 4, 2, 4, 5, 0, 1, 4, 4};
  EXPECT_TRUE(test::EqualSpan(goldTargets, graph.GetTargets()));
}

TEST(SparsityUtils, MakeSparsityGraphFromCoordsMatrixFormatWithSpecifiedNumRows) {
  // this is the same test above, but we also specify the num of rows

  // *within each row*, we intentionally put indices out-of-order and with duplicates
  // however, the rows must be given in order
  using v_int_t = std::vector<int>;
  std::vector<v_int_t> coords = {
      v_int_t{0, 1, 0, 0, 4, 4, 1, 1, 1}, // row 0
      v_int_t{2, 0, 0, 2, 2, 2, 2}, // row 1
      v_int_t{4, 0, 4, 0}, // row 2
      v_int_t{5, 4, 2, 4, 2}, // row 3
      v_int_t{1, 0, 4}, // row 4
      v_int_t{4, 4, 4} // row 5
  };

  int const forcedNumRows = 8;
  auto graph = MakeSparsityGraph(std::move(coords), forcedNumRows);

  EXPECT_EQ(forcedNumRows, graph.size());
  std::vector<int> const goldPtrs = {0, 3, 5, 7, 10, 13, 14, 14, 14};
  EXPECT_TRUE(test::EqualSpan(goldPtrs, graph.GetPointers()));

  std::vector<int> const goldTargets = {0, 1, 4, 0, 2, 0, 4, 2, 4, 5, 0, 1, 4, 4};
  EXPECT_TRUE(test::EqualSpan(goldTargets, graph.GetTargets()));
}

TEST(SparsityUtils, MakeDenseSparsityGraph) {
  constexpr int kSizes[] = {1, 2, 3, 8};
  for (int numRows : kSizes) {
    for (int numCols : kSizes) {
      auto denseMat = Matrix<real>{numRows, numCols};
      auto graph = MakeDenseSparsityGraph(numRows, numCols);
      auto sparseMat = SparseMatrix<real>{numCols, std::move(graph)};
      denseMat.SetConstant(1_r);
      sparseMat.SetConstant(1_r);
      EXPECT_TRUE(test::NearEqualMatrices(denseMat, sparseMat)); // Compare every entry
    }
  }
}

TEST(SparsityUtils, MatAddSubBlocks_BlockSparseMatrix_BlockSize3) {
  // Create a 9x9 block sparse matrix with block size 3
  DynamicArray<int> blockPointers = {0, 2, 4, 6};
  DynamicArray<int> blockIndices = {0, 2, 0, 1, 1, 2};
  DynamicArray<real> values(blockIndices.size() * 3 * 3);
  std::fill(values.begin(), values.end(), 0_r);

  BlockSparseMatrix<real, 3> bSp(3, blockPointers, blockIndices, values);

  // Create index groups for adding values
  // Block row 1 (rows 3-5) has blocks at columns 0-2 and 3-5
  // Add to rows [3,4,5] and columns [3,4,5]
  DynamicArray<IndexGroup> rows = {IndexGroup{.src = 0, .dst = 3, .count = 3}};
  DynamicArray<IndexGroup> cols = {IndexGroup{.src = 0, .dst = 3, .count = 3}};

  // Create a 3x3 block of values to add
  RowMatrix<real> valuesToAdd(3, 3);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      valuesToAdd(i, j) = static_cast<real>(i * 3 + j + 1);
    }
  }

  MatAddSubBlocks(AsView(bSp), MakeSpan(rows), MakeSpan(cols), valuesToAdd);

  // Verify the values were added correctly
  RowMatrix<real> expected = RowMatrix<real>::Zero(9, 9);
  for (int i = 3; i <= 5; ++i) {
    for (int j = 3; j <= 5; ++j) {
      expected(i, j) = static_cast<real>((i - 3) * 3 + (j - 3) + 1);
    }
  }
  EXPECT_TRUE(test::NearEqualMatrices(expected, bSp));

  // Test accumulation - add the same values again
  MatAddSubBlocks(AsView(bSp), MakeSpan(rows), MakeSpan(cols), valuesToAdd);

  for (int i = 3; i <= 5; ++i) {
    for (int j = 3; j <= 5; ++j) {
      expected(i, j) = static_cast<real>(2 * ((i - 3) * 3 + (j - 3) + 1));
    }
  }
  EXPECT_TRUE(test::NearEqualMatrices(expected, bSp));
}

TEST(SparsityUtils, MatAddSubBlocks_BlockSparseMatrix_BlockSize4) {
  // Create a 12x12 block sparse matrix with block size 4
  DynamicArray<int> blockPointers = {0, 2, 4, 6};
  DynamicArray<int> blockIndices = {0, 2, 0, 1, 1, 2};
  DynamicArray<real> values(blockIndices.size() * 4 * 4);
  std::fill(values.begin(), values.end(), 0_r);

  BlockSparseMatrix<real, 4> bSp(3, blockPointers, blockIndices, values);

  // Create index groups for adding values
  // Block row 1 (rows 4-7) has blocks at columns 0-3 and 4-7
  // Add to rows [4,5,6,7] and columns [4,5,6,7]
  // This tests a full 4x4 block addition
  DynamicArray<IndexGroup> rows = {IndexGroup{.src = 0, .dst = 4, .count = 4}};
  DynamicArray<IndexGroup> cols = {IndexGroup{.src = 0, .dst = 4, .count = 4}};

  // Create a 4x4 block of values to add
  RowMatrix<real> valuesToAdd(4, 4);
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      valuesToAdd(i, j) = static_cast<real>(i * 4 + j + 1);
    }
  }

  MatAddSubBlocks(AsView(bSp), MakeSpan(rows), MakeSpan(cols), valuesToAdd);

  // Verify the values were added correctly
  RowMatrix<real> expected = RowMatrix<real>::Zero(12, 12);
  for (int i = 4; i <= 7; ++i) {
    for (int j = 4; j <= 7; ++j) {
      expected(i, j) = static_cast<real>((i - 4) * 4 + (j - 4) + 1);
    }
  }
  EXPECT_TRUE(test::NearEqualMatrices(expected, bSp));

  // Test accumulation - add the same values again
  MatAddSubBlocks(AsView(bSp), MakeSpan(rows), MakeSpan(cols), valuesToAdd);

  for (int i = 4; i <= 7; ++i) {
    for (int j = 4; j <= 7; ++j) {
      expected(i, j) = static_cast<real>(2 * ((i - 4) * 4 + (j - 4) + 1));
    }
  }
  EXPECT_TRUE(test::NearEqualMatrices(expected, bSp));
}

TEST(SparsityUtils, MatAddSubBlocks_BlockSparseMatrix_MultipleGroups_BlockSize3) {
  // Create a 9x9 block sparse matrix with block size 3
  // Sparsity pattern:
  // Block row 0 (rows 0-2): block cols 0,2 -> cols 0-2, 6-8
  // Block row 1 (rows 3-5): block cols 0,1 -> cols 0-2, 3-5
  // Block row 2 (rows 6-8): block cols 1,2 -> cols 3-5, 6-8
  DynamicArray<int> blockPointers = {0, 2, 4, 6};
  DynamicArray<int> blockIndices = {0, 2, 0, 1, 1, 2};
  DynamicArray<real> values(blockIndices.size() * 3 * 3);
  std::fill(values.begin(), values.end(), 0_r);

  BlockSparseMatrix<real, 3> bSp(3, blockPointers, blockIndices, values);

  // Create multiple row groups that all use the same column group
  // Row group 1: rows [0,1] (block row 0)
  // Row group 2: rows [3,4] (block row 1)
  // Both add to columns [0,1,2] (block col 0) which exists in both block rows
  DynamicArray<IndexGroup> rows = {
      IndexGroup{.src = 0, .dst = 0, .count = 2}, IndexGroup{.src = 2, .dst = 3, .count = 2}};
  DynamicArray<IndexGroup> cols = {IndexGroup{.src = 0, .dst = 0, .count = 3}};

  // Create a 4x3 block of values to add
  RowMatrix<real> valuesToAdd(4, 3);
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 3; ++j) {
      valuesToAdd(i, j) = static_cast<real>(i * 10 + j);
    }
  }

  MatAddSubBlocks(AsView(bSp), MakeSpan(rows), MakeSpan(cols), valuesToAdd);

  // Verify the values
  RowMatrix<real> expected = RowMatrix<real>::Zero(9, 9);
  // Verify rows 0-1, columns 0-2
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      expected(i, j) = static_cast<real>(i * 10 + j);
    }
  }
  // Verify rows 3-4, columns 0-2 (using rows 2-3 from valuesToAdd)
  for (int i = 3; i < 5; ++i) {
    for (int j = 0; j < 3; ++j) {
      expected(i, j) = static_cast<real>((i - 1) * 10 + j);
    }
  }
  EXPECT_TRUE(test::NearEqualMatrices(expected, bSp));
}

TEST(SparsityUtils, MatAddSubBlocks_BlockSparseMatrix_MultipleGroups_BlockSize4) {
  // Create a 12x12 block sparse matrix with block size 4
  // Sparsity pattern:
  // Block row 0 (rows 0-3): block cols 0,2 -> cols 0-3, 8-11
  // Block row 1 (rows 4-7): block cols 0,1 -> cols 0-3, 4-7
  // Block row 2 (rows 8-11): block cols 1,2 -> cols 4-7, 8-11
  DynamicArray<int> blockPointers = {0, 2, 4, 6};
  DynamicArray<int> blockIndices = {0, 2, 0, 1, 1, 2};
  DynamicArray<real> values(blockIndices.size() * 4 * 4);
  std::fill(values.begin(), values.end(), 0_r);

  BlockSparseMatrix<real, 4> bSp(3, blockPointers, blockIndices, values);

  // Create multiple row groups that all use the same column group
  // Row group 1: rows [0,1,2] (block row 0)
  // Row group 2: rows [4,5] (block row 1)
  // Both add to columns [0,1,2,3] (block col 0) which exists in both block rows
  DynamicArray<IndexGroup> rows = {
      IndexGroup{.src = 0, .dst = 0, .count = 3}, IndexGroup{.src = 3, .dst = 4, .count = 2}};
  DynamicArray<IndexGroup> cols = {IndexGroup{.src = 0, .dst = 0, .count = 4}};

  // Create a 5x4 block of values to add
  RowMatrix<real> valuesToAdd(5, 4);
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 4; ++j) {
      valuesToAdd(i, j) = static_cast<real>(i * 10 + j);
    }
  }

  MatAddSubBlocks(AsView(bSp), MakeSpan(rows), MakeSpan(cols), valuesToAdd);

  // Verify the values
  RowMatrix<real> expected = RowMatrix<real>::Zero(12, 12);
  // Verify rows 0-2, columns 0-3
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 4; ++j) {
      expected(i, j) = static_cast<real>(i * 10 + j);
    }
  }
  // Verify rows 4-5, columns 0-3 (using rows 3-4 from valuesToAdd)
  for (int i = 4; i < 6; ++i) {
    for (int j = 0; j < 4; ++j) {
      expected(i, j) = static_cast<real>((i - 1) * 10 + j);
    }
  }
  EXPECT_TRUE(test::NearEqualMatrices(expected, bSp));
}

TEST(SparsityUtils, MakeSparsityGraphFromL2G) {
  // Create a tetrahedral mesh (arbitrary shape)
  auto mesh = TetrahedralMesh{test::CreateMinimalTetMeshUnitGrid()};

  // Use the node connectivity to build a dense matrix, such that denseConnectivityMatrix(i, j) is
  // 1 if the node with DOF i is connected to the node with DOF j.
  int numNodes = mesh.GetNumNodes();
  int numDofs = numNodes * 3;
  auto denseDofConnectivityMat = RowMatrix<real>::Zero(numDofs, numDofs);
  for (Int4 const& elem : mesh.GetElementConnectivity()) {
    for (int iNode : elem) {
      for (int jNode : elem) {
        for (int ii = 0; ii < 3; ++ii) {
          for (int jj = 0; jj < 3; ++jj) {
            denseDofConnectivityMat(3 * iNode + ii, 3 * jNode + jj) = 1_r;
          }
        }
      }
    }
  }

  // Use Local2GlobalMap to construct the sparsity pattern as a Graph.
  // Then use the Graph to construct a SparseMatrix
  Local2GlobalMap map;
  map.InitializeFromMesh(&mesh, 3);
  auto sparsityGraph = MakeSparsityGraph(map);
  auto pointers = sparsityGraph.GetPointers();
  auto indices = sparsityGraph.GetTargets();
  EXPECT_EQ(numDofs + 1, isize(pointers));

  // Verify that column indices are sorted along each row
  for (int r = 0; r < numDofs; ++r) {
    EXPECT_TRUE(std::is_sorted(indices.begin() + pointers[r], indices.begin() + pointers[r + 1]));
  }

  // Use the sparsity graph to initialize a SparseMatrix
  SparseMatrix<real> sparseMat(numDofs, std::move(sparsityGraph));
  EXPECT_EQ(numDofs, sparseMat.Rows());
  EXPECT_EQ(numDofs, sparseMat.Cols());

  // Set all values in the SparseMatrix to 1
  std::fill(sparseMat.Values().begin(), sparseMat.Values().end(), 1_r);

  // If the sparsity graph was correct, then the non-zero entries in the sparse matrix should
  // match the non-zero entries in denseDofConnectivityMat.
  EXPECT_TRUE(test::NearEqualMatrices(denseDofConnectivityMat, sparseMat));
}

TEST(SparsityUtils, MakeSparseMatrixFromL2G) {
  auto mesh = TetrahedralMesh{test::CreateMinimalTetMeshUnitGrid()}; // arbitrary mesh
  Local2GlobalMap map;
  map.InitializeFromMesh(&mesh, 3);
  Graph<int, int> sparsity = MakeSparsityGraph(map);
  SparseMatrix<real> spmat = MakeSparseMatrix(map);
  EXPECT_EQ(mesh.GetNumNodes() * 3, spmat.Rows());
  EXPECT_EQ(mesh.GetNumNodes() * 3, spmat.Cols());
  EXPECT_EQ(sparsity.GetPointers(), spmat.Pointers());
  EXPECT_EQ(sparsity.GetTargets(), spmat.Indices());
  std::vector<real> zeros(sparsity.NumTargets(), 0_r);
  EXPECT_EQ(MakeSpan(zeros), spmat.Values());
}

TEST(SparsityUtils, AppendNonZeroCoordinates) {
  // |xx |
  // |xxx|
  // |x x|
  auto graph = Graph<int, int>{{0, 2, 5, 7}, {0, 1, 0, 1, 2, 0, 2}};
  std::vector<Int2> coords;
  coords.emplace_back(123, 456);
  int dofOffset = 10;
  AppendNonZeroCoordinates(coords, graph, dofOffset);
  EXPECT_TRUE(
      test::EqualSpan(
          std::vector<Int2>{
              {123, 456}, {10, 10}, {10, 11}, {11, 10}, {11, 11}, {11, 12}, {12, 10}, {12, 12}},
          coords));
}

TEST(SparsityUtils, SetZeroOnRowsCols_SparseMatrix) {
  // |11 |
  // |111|
  // |1 1|
  std::vector<Int2> coords = {{0, 0}, {0, 1}, {1, 0}, {1, 1}, {1, 2}, {2, 0}, {2, 2}};

  // Zero two cols (and corresponding rows for comparison)
  {
    SparseMatrix<real, int, int> sp{MakeSparsityGraph(coords)};
    sp.SetConstant(1_r);
    SparseMatrix<real, int, int> spT = Transpose(sp);
    int constexpr kCols[] = {2, 0};
    // Use spT's own transpose (sp) as the symmetric pair for column lookup
    SetZeroOnCols(AsView(spT), MakeSpan(kCols), 0, AsConstView(sp));
    SetZeroOnRows(AsView(sp), MakeSpan(kCols), 0, 0_r);
    EXPECT_TRUE(
        test::NearEqualMatrices(
            sp, RowMatrix<real>{{0_r, 0_r, 0_r}, {1_r, 1_r, 1_r}, {0_r, 0_r, 0_r}}));
    EXPECT_TRUE(test::NearEqualMatrices(sp, Transpose(spT)));
  }

  // Zero two rows with a DOF offset and non-zero diagonal value.
  {
    SparseMatrix<real> sp{MakeSparsityGraph(coords)};
    sp.SetConstant(1_r);
    int constexpr kDofOffset = 1;
    int constexpr kRows[] = {0, 1}; // {1, 2} after offset
    SetZeroOnRows(AsView(sp), MakeSpan(kRows), kDofOffset, 2_r);
    EXPECT_TRUE(
        test::NearEqualMatrices(
            sp, RowMatrix<real>{{1_r, 1_r, 0_r}, {0_r, 2_r, 0_r}, {0_r, 0_r, 2_r}}));
  }
}

TEST(SparsityUtils, SetZeroOnRowsCols_BlockSparseMatrix_BlockSize3) {
  // |111   111|
  // |111   111|
  // |111   111|
  // |111111   |
  // |111111   |
  // |111111   |
  // |   111111|
  // |   111111|
  // |   111111|

  DynamicArray<int> blockPointers = {0, 2, 4, 6};
  DynamicArray<int> blockIndices = {0, 2, 0, 1, 1, 2};
  DynamicArray<real> values(blockIndices.size() * 3 * 3);
  std::fill(values.begin(), values.end(), 1_r);

  // Zero individual cols (and corresponding rows for comparison)
  {
    BlockSparseMatrix<real, 3> bSp(3, blockPointers, blockIndices, values);
    BlockSparseMatrix<real, 3> bSpT = Transpose(bSp);
    int constexpr kCols[] = {8, 0, 4};
    SetZeroOnRows(AsView(bSp), MakeSpan(kCols), 0, 0_r);
    SetZeroOnCols(AsView(bSpT), MakeSpan(kCols), 0, AsConstView(bSp));
    EXPECT_TRUE(
        test::NearEqualMatrices(
            bSp,
            RowMatrix<real>{
                {0, 0, 0, 0, 0, 0, 0, 0, 0}, // row 0
                {1, 1, 1, 0, 0, 0, 1, 1, 1},
                {1, 1, 1, 0, 0, 0, 1, 1, 1},
                {1, 1, 1, 1, 1, 1, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0}, // row 4
                {1, 1, 1, 1, 1, 1, 0, 0, 0},
                {0, 0, 0, 1, 1, 1, 1, 1, 1},
                {0, 0, 0, 1, 1, 1, 1, 1, 1},
                {0, 0, 0, 0, 0, 0, 0, 0, 0}, // row 8
            }));
    EXPECT_TRUE(test::NearEqualMatrices(bSp, Transpose(bSpT)));
  }

  // Zero individual rows with a DOF offset and non-zero diagonal value
  {
    BlockSparseMatrix<real, 3> bSp(3, blockPointers, blockIndices, values);
    int constexpr kDofOffset = 1;
    int constexpr kRows[] = {-1, 7, 3}; // {0, 8, 4} after offset
    SetZeroOnRows(AsView(bSp), MakeSpan(kRows), kDofOffset, 2_r);
    EXPECT_TRUE(
        test::NearEqualMatrices(
            bSp,
            RowMatrix<real>{
                {2, 0, 0, 0, 0, 0, 0, 0, 0}, // row 0
                {1, 1, 1, 0, 0, 0, 1, 1, 1},
                {1, 1, 1, 0, 0, 0, 1, 1, 1},
                {1, 1, 1, 1, 1, 1, 0, 0, 0},
                {0, 0, 0, 0, 2, 0, 0, 0, 0}, // row 4
                {1, 1, 1, 1, 1, 1, 0, 0, 0},
                {0, 0, 0, 1, 1, 1, 1, 1, 1},
                {0, 0, 0, 1, 1, 1, 1, 1, 1},
                {0, 0, 0, 0, 0, 0, 0, 0, 2}, // row 8
            }));
  }

  // Zero full row/col blocks.
  {
    BlockSparseMatrix<real, 3> bSp(3, blockPointers, blockIndices, values);
    BlockSparseMatrix<real, 3> bSpT = Transpose(bSp);
    int constexpr kCols[] = {0, 1, 2, 6, 7, 8};
    SetZeroOnCols(AsView(bSpT), MakeSpan(kCols), 0, AsConstView(bSp));
    SetZeroOnRows(AsView(bSp), MakeSpan(kCols), 0, 0_r);
    EXPECT_TRUE(
        test::NearEqualMatrices(
            bSp,
            RowMatrix<real>{
                {0, 0, 0, 0, 0, 0, 0, 0, 0}, // row 0
                {0, 0, 0, 0, 0, 0, 0, 0, 0}, // row 1
                {0, 0, 0, 0, 0, 0, 0, 0, 0}, // row 2
                {1, 1, 1, 1, 1, 1, 0, 0, 0},
                {1, 1, 1, 1, 1, 1, 0, 0, 0},
                {1, 1, 1, 1, 1, 1, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0}, // row 6
                {0, 0, 0, 0, 0, 0, 0, 0, 0}, // row 7
                {0, 0, 0, 0, 0, 0, 0, 0, 0}, // row 8
            }));
    EXPECT_TRUE(test::NearEqualMatrices(bSp, Transpose(bSpT)));
  }

  // Zero blocks of rows with a DOF offset and non-zero diagonal value
  {
    BlockSparseMatrix<real, 3> bSp(3, blockPointers, blockIndices, values);
    int constexpr kDofOffset = 3;
    int constexpr kRows[] = {3, 4, 5, -3, -2, -1}; // {6, 7, 8, 0, 1, 2} after offset
    SetZeroOnRows(AsView(bSp), MakeSpan(kRows), kDofOffset, 2_r);
    EXPECT_TRUE(
        test::NearEqualMatrices(
            bSp,
            RowMatrix<real>{
                {2, 0, 0, 0, 0, 0, 0, 0, 0}, // row 0
                {0, 2, 0, 0, 0, 0, 0, 0, 0}, // row 1
                {0, 0, 2, 0, 0, 0, 0, 0, 0}, // row 2
                {1, 1, 1, 1, 1, 1, 0, 0, 0},
                {1, 1, 1, 1, 1, 1, 0, 0, 0},
                {1, 1, 1, 1, 1, 1, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 2, 0, 0}, // row 6
                {0, 0, 0, 0, 0, 0, 0, 2, 0}, // row 7
                {0, 0, 0, 0, 0, 0, 0, 0, 2}, // row 8
            }));
  }
}

TEST(SparsityUtils, SetZeroOnRowsCols_BlockSparseMatrix_BlockSize4) {
  // |1111    1111|
  // |1111    1111|
  // |1111    1111|
  // |1111    1111|
  // |11111111    |
  // |11111111    |
  // |11111111    |
  // |11111111    |
  // |    11111111|
  // |    11111111|
  // |    11111111|
  // |    11111111|

  DynamicArray<int> blockPointers = {0, 2, 4, 6};
  DynamicArray<int> blockIndices = {0, 2, 0, 1, 1, 2};
  DynamicArray<real> values(blockIndices.size() * 4 * 4);
  std::fill(values.begin(), values.end(), 1_r);

  // Zero individual cols (and corresponding rows for comparison)
  {
    BlockSparseMatrix<real, 4> bSp(3, blockPointers, blockIndices, values);
    BlockSparseMatrix<real, 4> bSpT = Transpose(bSp);
    int constexpr kCols[] = {11, 0, 5};
    SetZeroOnCols(AsView(bSpT), MakeSpan(kCols), 0, AsConstView(bSp));
    SetZeroOnRows(AsView(bSp), MakeSpan(kCols), 0, 0_r);
    EXPECT_TRUE(
        test::NearEqualMatrices(
            bSp,
            RowMatrix<real>{
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // row 0
                {1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1},
                {1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1},
                {1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1},
                {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // row 5
                {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0},
                {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0},
                {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1},
                {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1},
                {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // row 11
            }));
    EXPECT_TRUE(test::NearEqualMatrices(bSp, Transpose(bSpT)));
  }

  // Zero individual rows with a DOF offset and non-zero diagonal value
  {
    BlockSparseMatrix<real, 4> bSp(3, blockPointers, blockIndices, values);
    int constexpr kDofOffset = 1;
    int constexpr kCols[] = {-1, 10, 4}; // {0, 11, 5} after offset
    SetZeroOnRows(AsView(bSp), MakeSpan(kCols), kDofOffset, 2_r);
    EXPECT_TRUE(
        test::NearEqualMatrices(
            bSp,
            RowMatrix<real>{
                {2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // row 0
                {1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1},
                {1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1},
                {1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1},
                {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0}, // row 5
                {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0},
                {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0},
                {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1},
                {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1},
                {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2}, // row 11
            }));
  }

  // Zero full row/col blocks.
  {
    BlockSparseMatrix<real, 4> bSp(3, blockPointers, blockIndices, values);
    BlockSparseMatrix<real, 4> bSpT = Transpose(bSp);
    int constexpr kCols[] = {0, 1, 2, 3, 8, 9, 10, 11};
    SetZeroOnCols(AsView(bSpT), MakeSpan(kCols), 0, AsConstView(bSp));
    SetZeroOnRows(AsView(bSp), MakeSpan(kCols), 0, 0_r);
    EXPECT_TRUE(
        test::NearEqualMatrices(
            bSp,
            RowMatrix<real>{
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // row 0
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // row 1
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // row 2
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // row 3
                {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0},
                {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0},
                {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0},
                {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // row 8
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // row 9
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // row 10
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // row 11
            }));
    EXPECT_TRUE(test::NearEqualMatrices(bSp, Transpose(bSpT)));
  }

  // Zero blocks of rows with a DOF offset and non-zero diagonal value
  {
    BlockSparseMatrix<real, 4> bSp(3, blockPointers, blockIndices, values);
    int constexpr kDofOffset = 4;
    int constexpr kRows[] = {4, 5, 6, 7, -4, -3, -2, -1}; // {8, 9, 10, 11, 0, 1, 2, 3} after offset
    SetZeroOnRows(AsView(bSp), MakeSpan(kRows), kDofOffset, 2_r);
    EXPECT_TRUE(
        test::NearEqualMatrices(
            bSp,
            RowMatrix<real>{
                {2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // row 0
                {0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // row 1
                {0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // row 2
                {0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0}, // row 3
                {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0},
                {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0},
                {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0},
                {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0}, // row 8
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0}, // row 9
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0}, // row 10
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2}, // row 11
            }));
  }
}

TEST(SparsityUtils, SetZeroOnRowsCols_Matrix) {
  // |111|
  // |111|
  // |111|

  // Zero two cols (and corresponding rows for comparison)
  {
    Matrix<real> mat(3, 3);
    mat.SetConstant(1_r);
    Matrix<real> matT = mat.Transpose();
    int constexpr kCols[] = {0, 2};
    SetZeroOnRows(AsView(mat), MakeSpan(kCols), 0, 0_r);
    SetZeroOnCols(AsView(matT), MakeSpan(kCols), 0, AsConstView(matT));
    EXPECT_TRUE(
        test::NearEqualMatrices(
            mat,
            RowMatrix<real>{
                {0, 0, 0}, // row 0
                {1, 1, 1},
                {0, 0, 0} // row 2
            }));
    EXPECT_TRUE(test::NearEqualMatrices(mat, matT.Transpose()));
  }

  // Zero two rows with a DOF offset and non-zero diagonal value
  {
    Matrix<real> mat(3, 3);
    mat.SetConstant(1_r);
    int constexpr kDofOffset = 1;
    int constexpr kRows[] = {-1, 1}; // {0, 2} after offset
    SetZeroOnRows(AsView(mat), MakeSpan(kRows), kDofOffset, 2_r);
    EXPECT_TRUE(
        test::NearEqualMatrices(
            mat,
            RowMatrix<real>{
                {2, 0, 0}, // row 0
                {1, 1, 1},
                {0, 0, 2} // row 2
            }));
  }
}

TEST(SparsityUtils, SetZeroOnRows_ColumnVector) {
  {
    ColumnVector<real> col(5);
    col.SetConstant(1_r);
    int constexpr kRows[] = {0, 2, 4};
    SetZeroOnRows(AsView(col), MakeSpan(kRows), 0);
    EXPECT_EQ(0_r, col[0]);
    EXPECT_EQ(1_r, col[1]);
    EXPECT_EQ(0_r, col[2]);
    EXPECT_EQ(1_r, col[3]);
    EXPECT_EQ(0_r, col[4]);
  }

  // With DOF offset
  {
    ColumnVector<real> col(5);
    col.SetConstant(1_r);
    int constexpr kRows[] = {-1, 1, 3}; // {0, 2, 4} after offset
    SetZeroOnRows(AsView(col), MakeSpan(kRows), 1);
    EXPECT_EQ(0_r, col[0]);
    EXPECT_EQ(1_r, col[1]);
    EXPECT_EQ(0_r, col[2]);
    EXPECT_EQ(1_r, col[3]);
    EXPECT_EQ(0_r, col[4]);
  }
}

TEST(SparsityUtils, SetZeroOnRowsCols_AnyMatrix) {
  // |11 |
  // |111|
  // |1 1|
  std::vector<Int2> coords = {{0, 0}, {0, 1}, {1, 0}, {1, 1}, {1, 2}, {2, 0}, {2, 2}};

  {
    // SetZeroOnRows
    SparseMatrix<real> sp1{MakeSparsityGraph(coords)};
    sp1.SetConstant(1_r);
    SparseMatrix<real> sp2 = sp1;
    AnyMatrixView<real> anyMat = AsView(sp2);
    int constexpr kRows[] = {0, 2};
    SetZeroOnRows(AsView(sp1), MakeSpan(kRows), 0, 1_r); // Specialized overload
    SetZeroOnRows(anyMat, MakeSpan(kRows), 0, 1_r); // AnyMatrixView overload
    EXPECT_TRUE(test::NearEqualMatrices(sp1, sp2));

    // SetZeroOnCols
    sp1.SetConstant(1_r);
    SparseMatrix<real> symSp1 =
        ToSparseMatrix(Matrix<real>{ToMatrix(sp1) + Transpose(ToMatrix(sp1))}, /*pruneZeros*/ true);
    SparseMatrix<real> symSp2 = symSp1;
    AnyMatrixView<real> anyMat2 = AsView(symSp2);
    int constexpr kCols[] = {0, 2};
    SetZeroOnCols(AsView(symSp1), MakeSpan(kCols), 0, AsConstView(symSp1)); // Specialized overload
    SetZeroOnCols(anyMat2, MakeSpan(kCols), 0, AsConstView(symSp2)); // AnyMatrixView overload
    EXPECT_TRUE(test::NearEqualMatrices(symSp1, symSp2));
  }
}

TEST(SparsityUtils, SetZeroOnCols_ValueIndicesCache) {
  // Test the valueIndicesCache parameter for SetZeroOnCols.

  // SparseMatrix
  {
    // Create a symmetric sparsity pattern for testing the cache:
    // |111|
    // |111|
    // |111|
    std::vector<Int2> coords = {
        {0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 1}, {1, 2}, {2, 0}, {2, 1}, {2, 2}};
    SparseMatrix<real> symSp{MakeSparsityGraph(coords)};
    symSp.SetConstant(1_r);

    // Create an owning graph from the sparse matrix for AppendColValueIndexCache
    auto graphView = AsGraphView(symSp);
    Graph<int, int> sparsityGraph{
        DynamicArray<int>(graphView.GetPointers().begin(), graphView.GetPointers().end()),
        DynamicArray<int>(graphView.GetTargets().begin(), graphView.GetTargets().end())};

    // Build the cache for columns {0, 2}
    int constexpr kCols[] = {0, 2};
    DynamicArray<int> colValueIndices;
    AppendColValueIndexCache(sparsityGraph, MakeSpan(kCols), colValueIndices);
    EXPECT_FALSE(colValueIndices.empty());

    // Test that using the cache produces the same result as without cache
    SparseMatrix<real> symSp1 = symSp;

    // Without cache: pass the matrix itself as symmetric pair (symmetric sparsity)
    SetZeroOnCols(AsView(symSp), MakeSpan(kCols), 0, AsConstView(symSp));
    // With cache: cache is used directly, symmetric pair is still required
    SetZeroOnCols(
        AsView(symSp1), MakeSpan(kCols), 0, AsConstView(symSp1), MakeConstSpan(colValueIndices));

    EXPECT_TRUE(test::NearEqualMatrices(symSp, symSp1));

    // Verify the specified columns are zeroed
    for (int c : kCols) {
      for (int r = 0; r < symSp.Rows(); ++r) {
        EXPECT_EQ(0_r, symSp(r, c));
      }
    }
  }

  // BlockSparseMatrix
  // For BlockSparseMatrix, the cache is built using AppendColValueIndexCache with a scalar
  // sparsity graph that matches the BlockSparseMatrix's value array layout. Each block of
  // size 3x3 stores 9 consecutive values in row-major order.
  {
    // Create a 9x9 diagonal BlockSparseMatrix with 3 blocks of 3x3:
    //   Block row 0: block at col 0
    //   Block row 1: block at col 1
    //   Block row 2: block at col 2
    // This is a symmetric block sparsity pattern.
    DynamicArray<int> blockPointers = {0, 1, 2, 3};
    DynamicArray<int> blockIndices = {0, 1, 2};
    DynamicArray<real> values(blockIndices.size() * 3 * 3);
    std::fill(values.begin(), values.end(), 1_r);

    BlockSparseMatrix<real, 3> bSp(3, blockPointers, blockIndices, values);

    // Construct the scalar sparsity graph that matches the BlockSparseMatrix's value layout.
    // For each scalar row r, the columns are in the block at block row r/3.
    // The value indices must match the BlockSparseMatrix's Values() array layout.
    //   Rows 0-2 have cols 0-2 (block 0, values 0-8)
    //   Rows 3-5 have cols 3-5 (block 1, values 9-17)
    //   Rows 6-8 have cols 6-8 (block 2, values 18-26)
    DynamicArray<int> scalarPointers = {0, 3, 6, 9, 12, 15, 18, 21, 24, 27};
    DynamicArray<int> scalarIndices = {0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4,
                                       5, 3, 4, 5, 6, 7, 8, 6, 7, 8, 6, 7, 8};
    Graph<int, int> scalarSparsityGraph{std::move(scalarPointers), std::move(scalarIndices)};

    // Build the cache for columns {0, 2} using AppendColValueIndexCache
    int constexpr kCols[] = {0, 2};
    DynamicArray<int> colValueIndices;
    AppendColValueIndexCache(scalarSparsityGraph, MakeSpan(kCols), colValueIndices);
    EXPECT_FALSE(colValueIndices.empty());

    // Test that using the cache produces the same result as without cache
    BlockSparseMatrix<real, 3> bSp1 = bSp;

    // Without cache: pass the matrix itself as symmetric pair (symmetric sparsity)
    SetZeroOnCols(AsView(bSp), MakeSpan(kCols), 0, AsConstView(bSp));
    // With cache: cache is used directly, symmetric pair is still required
    SetZeroOnCols(
        AsView(bSp1), MakeSpan(kCols), 0, AsConstView(bSp1), MakeConstSpan(colValueIndices));

    EXPECT_TRUE(test::NearEqualMatrices(bSp, bSp1));

    // Verify the specified columns are zeroed
    for (int c : kCols) {
      for (int r = 0; r < bSp.Rows(); ++r) {
        EXPECT_EQ(0_r, bSp(r, c));
      }
    }

    // Also verify that other columns are NOT zeroed
    EXPECT_EQ(1_r, bSp(0, 1));
    EXPECT_EQ(1_r, bSp(3, 4));
  }

  // Matrix
  {
    Matrix<real> mat(3, 3);
    mat.SetConstant(1_r);
    Matrix<real> mat2 = mat;
    int constexpr kCols[] = {0, 2};
    DynamicArray<int> dummyCache = {0, 1, 2}; // Must be ignored by dense implementation.
    SetZeroOnCols(AsView(mat), MakeSpan(kCols), 0, AsConstView(mat));
    SetZeroOnCols(AsView(mat2), MakeSpan(kCols), 0, AsConstView(mat2), MakeConstSpan(dummyCache));
    EXPECT_TRUE(test::NearEqualMatrices(mat, mat2));
    for (int c : kCols) {
      for (int r = 0; r < mat.Rows(); ++r) {
        EXPECT_EQ(0_r, mat(r, c));
      }
    }
  }
}

TEST(SparsityUtils, AppendColValueIndexCache) {
  //      0 1 2 3 4
  //    +-----------
  //  0 | X . X . .    row 0: cols {0, 2}       -> value indices 0, 1
  //  1 | . X X . .    row 1: cols {1, 2}       -> value indices 2, 3
  //  2 | X X X X .    row 2: cols {0, 1, 2, 3} -> value indices 4, 5, 6, 7
  //  3 | . . X X .    row 3: cols {2, 3}       -> value indices 8, 9
  //  4 | . . . . X    row 4: cols {4}          -> value index 10
  //
  // This pattern tests:
  // - Column with multiple non-zeros at different row positions (col 2 has 4 entries)
  // - Column with single non-zero (col 4 has 1 entry)
  // - Column at beginning of row's targets (col 0 in row 2)
  // - Column at end of row's targets (col 3 in row 2)
  // - Diagonal entries (0,0), (1,1), (2,2), (3,3), (4,4)

  std::vector<Int2> coords = {
      {0, 0},
      {0, 2}, // row 0
      {1, 1},
      {1, 2}, // row 1
      {2, 0},
      {2, 1},
      {2, 2},
      {2, 3}, // row 2
      {3, 2},
      {3, 3}, // row 3
      {4, 4} // row 4
  };

  auto sparsityGraph = MakeSparsityGraph(coords);

  // Verify the graph structure matches our expectations
  EXPECT_TRUE(test::EqualSpan(std::vector<int>{0, 2, 4, 8, 10, 11}, sparsityGraph.GetPointers()));
  EXPECT_TRUE(
      test::EqualSpan(
          std::vector<int>{0, 2, 1, 2, 0, 1, 2, 3, 2, 3, 4}, sparsityGraph.GetTargets()));

  // Test empty input: should not modify the cache
  {
    DynamicArray<int> cache;
    AppendColValueIndexCache(sparsityGraph, Span<int const>{}, cache);
    EXPECT_TRUE(cache.empty());
  }

  // Test single column with multiple non-zeros (col 2 has entries in rows 0, 1, 2, 3)
  // For col 2:
  //   - Row 0 has col 2 at value index 1 (targets[1] = 2)
  //   - Row 1 has col 2 at value index 3 (targets[3] = 2)
  //   - Row 2 has col 2 at value index 6 (targets[6] = 2)
  //   - Row 3 has col 2 at value index 8 (targets[8] = 2)
  {
    DynamicArray<int> cache;
    int constexpr kCols[] = {2};
    AppendColValueIndexCache(sparsityGraph, MakeSpan(kCols), cache);
    // Expected: value indices for (0,2), (1,2), (2,2), (3,2) = {1, 3, 6, 8}
    EXPECT_TRUE(test::EqualSpan(std::vector<int>{1, 3, 6, 8}, cache));
  }

  // Test single column with single non-zero (col 4 has entry only in row 4)
  // For col 4:
  //   - Row 4 has col 4 at value index 10 (targets[10] = 4)
  {
    DynamicArray<int> cache;
    int constexpr kCols[] = {4};
    AppendColValueIndexCache(sparsityGraph, MakeSpan(kCols), cache);
    EXPECT_TRUE(test::EqualSpan(std::vector<int>{10}, cache));
  }

  // Test multiple columns
  // For cols {0, 3}:
  //   Col 0: rows {0, 2} -> value indices 0, 4
  //   Col 3: rows {2, 3} -> value indices 7, 9
  {
    DynamicArray<int> cache;
    int constexpr kCols[] = {0, 3};
    AppendColValueIndexCache(sparsityGraph, MakeSpan(kCols), cache);
    // Expected: {0, 4, 7, 9} (col 0 indices first, then col 3 indices)
    EXPECT_TRUE(test::EqualSpan(std::vector<int>{0, 4, 7, 9}, cache));
  }

  // Test append behavior: calling twice should append to existing cache
  {
    DynamicArray<int> cache;
    int constexpr kCols1[] = {1};
    int constexpr kCols2[] = {4};
    AppendColValueIndexCache(sparsityGraph, MakeSpan(kCols1), cache);
    // Col 1: rows {1, 2} -> value indices 2, 5
    EXPECT_TRUE(test::EqualSpan(std::vector<int>{2, 5}, cache));
    AppendColValueIndexCache(sparsityGraph, MakeSpan(kCols2), cache);
    // After appending col 4: {2, 5, 10}
    EXPECT_TRUE(test::EqualSpan(std::vector<int>{2, 5, 10}, cache));
  }

  // Test all columns at once
  {
    DynamicArray<int> cache;
    int constexpr kCols[] = {0, 1, 2, 3, 4};
    AppendColValueIndexCache(sparsityGraph, MakeSpan(kCols), cache);
    // All value indices should be present (and sorted).
    EXPECT_TRUE(test::EqualSpan(std::vector{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, cache));
  }
}

TEST(SparsityUtils, DuplicateAndPrune) {
  // Empty
  {
    SparseMatrix<real> mat;
    auto copy = DuplicateAndPrune(mat, 1);
    EXPECT_TRUE(test::NearEqualMatrices(mat, copy));
    EXPECT_EQ(0, copy.NumNonZeros());
  }

  // Dense non-zeros -> same
  for (int blockSize = 1; blockSize <= 3; ++blockSize) {
    SparseMatrix<real> mat{6, MakeDenseSparsityGraph(6, 6)};
    mat.SetConstant(1_r); // anything but zero
    auto copy = DuplicateAndPrune(mat, blockSize);
    EXPECT_TRUE(test::NearEqualMatrices(mat, copy));
    EXPECT_EQ(6 * 6, copy.NumNonZeros());
  }

  // Dense all zeros -> same dims, but empty non-zero structure
  for (int blockSize = 1; blockSize <= 3; ++blockSize) {
    SparseMatrix<real> mat{6, MakeDenseSparsityGraph(6, 6)};
    mat.SetZero();
    auto copy = DuplicateAndPrune(mat, blockSize);
    EXPECT_TRUE(test::NearEqualMatrices(mat, copy));
    EXPECT_EQ(0, copy.NumNonZeros());
  }

  // Dense single non-zero -> sparse single non-zero
  for (int blockSize = 1; blockSize <= 3; ++blockSize) {
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        SparseMatrix<real> mat{6, MakeDenseSparsityGraph(6, 6)};
        mat.SetZero();
        mat.SetValue(i, j, 1_r); // single non-zero value
        auto copy = DuplicateAndPrune(mat, blockSize);
        EXPECT_TRUE(test::NearEqualMatrices(mat, copy));
        EXPECT_EQ(6 * 6, mat.NumNonZeros());
        EXPECT_EQ(blockSize * blockSize, copy.NumNonZeros());
      }
    }
  }

  // Sparse single non-zero -> same
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      Int2 coord{i, j};
      SparseMatrix<real> mat{6, MakeSparsityGraph(std::vector<Int2>{coord})};
      mat.SetZero();
      mat.SetValue(i, j, 1_r); // single non-zero value
      auto copy = DuplicateAndPrune(mat, 1);
      EXPECT_TRUE(test::NearEqualMatrices(mat, copy));
      EXPECT_EQ(1, mat.NumNonZeros());
    }
  }

  // Mixed
  //  |   1 1 0 |      |   1 1   |
  //  | 1 1 1 1 |      | 1 1 1 1 |
  //  | 0   0 0 | ---> |         |
  //  | 1 1   0 |      | 1 1     |
  {
    std::vector<Int2> coords = {
        {0, 1},
        {0, 2},
        {0, 3},
        {1, 0},
        {1, 1},
        {1, 2},
        {1, 3},
        {2, 0},
        {2, 2},
        {2, 3},
        {3, 0},
        {3, 1},
        {3, 3}};
    SparseMatrix<real> mat{4, MakeSparsityGraph(std::move(coords))};
    mat.SetZero();
    mat.SetValue(0, 1, 1_r);
    mat.SetValue(0, 2, 1_r);
    mat.SetValue(1, 0, 1_r);
    mat.SetValue(1, 1, 1_r);
    mat.SetValue(1, 2, 1_r);
    mat.SetValue(1, 3, 1_r);
    mat.SetValue(3, 0, 1_r);
    mat.SetValue(3, 1, 1_r);
    auto copy = DuplicateAndPrune(mat, 1);
    EXPECT_TRUE(test::NearEqualMatrices(mat, copy));
    EXPECT_EQ(13, mat.NumNonZeros()); // Only 8 of these 13 values are actually non-zero.
    EXPECT_EQ(8, copy.NumNonZeros()); // Just the true non-zeros
  }

  // Test block size inference and block-wise pruning.
  {
    // Define a blockable sparsity pattern with block size of 3.
    // | 1 2 3 x x x 0 0 0 |
    // | 0 4 5 x x x 0 0 0 |
    // | 0 0 6 x x x 0 0 0 |
    // | x x x 0 0 0 7 0 0 |
    // | x x x 0 8 0 0 0 0 |
    // | x x x 0 0 0 0 0 0 |
    DynamicArray<int> pointers = {0, 6, 12, 18, 24, 30, 36};
    DynamicArray<int> indices = {0, 1, 2, 6, 7, 8, //
                                 0, 1, 2, 6, 7, 8, //
                                 0, 1, 2, 6, 7, 8, //
                                 3, 4, 5, 6, 7, 8, //
                                 3, 4, 5, 6, 7, 8, //
                                 3, 4, 5, 6, 7, 8};
    DynamicArray<real> values = {1, 2, 3, 0, 0, 0, //
                                 0, 4, 5, 0, 0, 0, //
                                 0, 0, 6, 0, 0, 0, //
                                 0, 0, 0, 7, 0, 0, //
                                 0, 8, 0, 0, 0, 0, //
                                 0, 0, 0, 0, 0, 0};
    SparseMatrix<real> mat{9, std::move(pointers), std::move(indices), std::move(values)};
    EXPECT_EQ(36, mat.NumNonZeros());

    // Block size == 1
    auto copy = DuplicateAndPrune(mat, 1);
    EXPECT_TRUE(test::NearEqualMatrices(mat, copy));
    EXPECT_EQ(8, copy.NumNonZeros());

    // Block size == 3
    copy.Reset(DuplicateAndPrune(mat, 3));
    EXPECT_TRUE(test::NearEqualMatrices(mat, copy));
    EXPECT_EQ(3 * 9, copy.NumNonZeros()); // 3 non-zero blocks
  }
}

TEST(LevelFill, LevelFill) {
  int nx = 5, ny = 7;
  int n = nx * ny;
  auto Adense = Matrix<real>::Zero(n, n);
  Matrix<int> S(n, n);
  S.SetConstant(std::numeric_limits<int>::max());
  for (int iy = 0; iy < ny; ++iy) {
    for (int ix = 0; ix < nx; ++ix) {
      int row = ix + iy * nx;
      Adense(row, row) = 4_r;
      S(row, row) = 0;
      if (ix > 0) {
        Adense(row, row - 1) = -1_r;
        S(row, row - 1) = 0;
      }
      if (ix + 1 < nx) {
        Adense(row, row + 1) = -1_r;
        S(row, row + 1) = 0;
      }
      if (iy > 0) {
        Adense(row, row - nx) = -1_r;
        S(row, row - nx) = 0;
      }
      if (iy + 1 < ny) {
        Adense(row, row + nx) = -1_r;
        S(row, row + nx) = 0;
      }
    }
  }
  //
  for (int i = 0; i < n; ++i) {
    for (int k = 0; k <= i; ++k) {
      if (S(i, k) == std::numeric_limits<int>::max()) {
        continue;
      }
      for (int j = 0; j <= i; ++j) {
        if ((S(i, j) == std::numeric_limits<int>::max()) &&
            (S(k, j) == std::numeric_limits<int>::max())) {
          continue;
        }
        if (S(k, j) != std::numeric_limits<int>::max()) {
          S(i, j) = Min(S(i, j), S(i, k) + S(k, j) + 1);
        }
      }
    }
  }
  //
  constexpr int pmax = 4;
  {
    // Check whether the fill level routine gives the expected entries
    // for the fill-in up to level 'pmax'
    auto Asp = ToSparseMatrix(Adense, /*pruneZeros*/ true);
    int pCount = 0;
    for (int p = 0; p <= pmax; ++p) {
      SparseMatrix<real> Rfill;
      mochi::details::ConvertToFillLevel(p, Asp, Rfill);
      int sCount = 0;
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
          if (S(i, j) <= p) {
            sCount += 1;
          }
        }
      }
      // Get the total number by symmetry
      EXPECT_EQ(2 * sCount - n, Rfill.NumNonZeros());
      if (2 * sCount - n != Rfill.NumNonZeros()) {
        break;
      }
      //
      // Number of non-zero entries match for level p
      //
      pCount += 1;
      //
      // Check the positions of non-zero entries in R match the positions in S
      //
      for (int i = 0; i < n; ++i) {
        auto colIdx = Rfill.Indices(i);
        for (int k = 0; k < isize(colIdx); ++k) {
          auto const j = colIdx[k];
          if (j > i) {
            break;
          }
          //
          // The non-zero entry (i,j) in R must have a level less or equal to p in S
          //
          EXPECT_LE(S(i, j), p);
        }
      }
    }
    EXPECT_EQ(pmax + 1, pCount);
  }
}
