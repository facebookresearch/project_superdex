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

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/graph_utils.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <set>
#include <vector>

using namespace mochi;

TEST(Graph, Reverse) {
  int const nEleX = 4, nEleY = 3;
  int numEle = nEleX * nEleY;
  DynamicArray<int> pointer;
  pointer.reserve(numEle + 1);
  pointer.push_back(0);
  DynamicArray<int> targets;
  targets.reserve(4 * numEle);
  //
  for (int jy = 0; jy < nEleY; ++jy) {
    for (int ix = 0; ix < nEleX; ++ix) {
      int node = ix + jy * (nEleX + 1);
      targets.push_back(node);
      targets.push_back(node + 1);
      targets.push_back(node + 1 + nEleX + 1);
      targets.push_back(node + nEleX + 1);
      pointer.push_back(isize(targets));
    }
  }
  //
  mochi::Graph<int, int> eToN(pointer, targets);
  auto nToE = Reverse(eToN);
  //
  EXPECT_EQ((nEleX + 1) * (nEleY + 1), nToE.size());
  //
  EXPECT_EQ(1, nToE.EdgeCount(0));
  EXPECT_EQ(1, nToE.EdgeCount(4));
  EXPECT_EQ(1, nToE.EdgeCount(15));
  EXPECT_EQ(1, nToE.EdgeCount(19));
  //
  std::vector<int> border{1, 2, 3, 5, 9, 10, 14, 16, 17, 18};
  for (auto node : border) {
    EXPECT_EQ(2, nToE.EdgeCount(node));
  }
  //
  std::vector<int> interior{6, 7, 8, 11, 12, 13};
  for (auto node : interior) {
    EXPECT_EQ(4, nToE.EdgeCount(node));
  }
  //
  auto eleList = nToE[12];
  EXPECT_EQ(5, eleList[0]);
  EXPECT_EQ(6, eleList[1]);
  EXPECT_EQ(9, eleList[2]);
  EXPECT_EQ(10, eleList[3]);
}

TEST(Graph, Traverse) {
  int const nEleX = 4, nEleY = 3;
  int numEle = nEleX * nEleY;
  DynamicArray<int> pointer;
  pointer.reserve(numEle + 1);
  pointer.push_back(0);
  DynamicArray<int> targets;
  targets.reserve(4 * numEle);
  //
  for (int jy = 0; jy < nEleY; ++jy) {
    for (int ix = 0; ix < nEleX; ++ix) {
      int node = ix + jy * (nEleX + 1);
      targets.push_back(node);
      targets.push_back(node + 1);
      targets.push_back(node + 1 + nEleX + 1);
      targets.push_back(node + nEleX + 1);
      pointer.push_back(isize(targets));
    }
  }
  //
  mochi::Graph<int, int> eToN(pointer, targets);
  auto nToE = Reverse(eToN);
  //
  {
    auto nToN = Traverse(nToE, eToN);
    EXPECT_EQ((nEleX + 1) * (nEleY + 1), nToN.size());
    EXPECT_EQ((nEleX + 1) * (nEleY + 1) - 1, nToN.MaxTarget());
    //
    EXPECT_EQ(4, nToN.EdgeCount(0));
    EXPECT_EQ(4, nToN.EdgeCount(4));
    EXPECT_EQ(4, nToN.EdgeCount(15));
    EXPECT_EQ(4, nToN.EdgeCount(19));
    //
    std::vector<int> border{1, 2, 3, 5, 9, 10, 14, 16, 17, 18};
    for (auto node : border) {
      EXPECT_EQ(6, nToN.EdgeCount(node));
    }
    //
    std::vector<int> interior{6, 7, 8, 11, 12, 13};
    for (auto node : interior) {
      EXPECT_EQ(9, nToN.EdgeCount(node));
    }
    //
    auto nodeList = nToN[11];
    // nToN may not have sorted entries
    // Sort the list to simplify the comparison
    std::sort(nodeList.begin(), nodeList.end());
    std::vector<int> neighbors{5, 6, 7, 10, 11, 12, 15, 16, 17};
    EXPECT_EQ(isize(neighbors), isize(nodeList));
    for (int i = 0; i < isize(neighbors); ++i) {
      EXPECT_EQ(neighbors[i], nodeList[i]);
    }
  }
  //
  {
    auto eToE = Traverse(eToN, nToE);
    EXPECT_EQ(numEle, eToE.size());
    EXPECT_EQ(numEle - 1, eToE.MaxTarget());
    EXPECT_EQ(4, eToE.EdgeCount(0));
    EXPECT_EQ(4, eToE.EdgeCount(3));
    EXPECT_EQ(4, eToE.EdgeCount(8));
    EXPECT_EQ(4, eToE.EdgeCount(11));
    //
    std::vector<int> border{4, 7, 9, 10};
    for (auto node : border) {
      EXPECT_EQ(6, eToE.EdgeCount(node));
    }
    //
    std::vector<int> interior{5, 6};
    for (auto node : interior) {
      EXPECT_EQ(9, eToE.EdgeCount(node));
    }
    auto eleList = eToE[5];
    // eToE may not have sorted entries
    // Sort the list to simplify the comparison
    std::sort(eleList.begin(), eleList.end());
    std::vector<int> neighbors{0, 1, 2, 4, 5, 6, 8, 9, 10};
    EXPECT_EQ(isize(neighbors), isize(eleList));
    for (int i = 0; i < isize(neighbors); ++i) {
      EXPECT_EQ(neighbors[i], eleList[i]);
    }
  }
}

TEST(GraphUtils, RedBlack2D) {
  int nx = 5, ny = 5;
  int n = nx * ny;
  std::vector<int> pointer;
  pointer.reserve(n + 1);
  pointer.push_back(0);
  std::vector<int> targets;
  targets.reserve(5 * n);
  for (int jy = 0; jy < ny; ++jy) {
    for (int ix = 0; ix < nx; ++ix) {
      int node = ix + jy * nx;
      if (jy > 0) {
        targets.push_back(node - nx);
      }
      if (ix > 0) {
        targets.push_back(node - 1);
      }
      targets.push_back(node);
      if (ix + 1 < nx) {
        targets.push_back(node + 1);
      }
      if (jy + 1 < ny) {
        targets.push_back(node + nx);
      }
      pointer.push_back(isize(targets));
    }
  }
  Graph<int, int, std::vector> g(pointer, targets);
  auto color = mochi::GreedyColoring(g);
  auto colPtr = color.GetPointers();
  auto colTgt = color.GetTargets();
  EXPECT_EQ(3, colPtr.size());
  for (int i = 0; i + 1 < colPtr.size(); ++i) {
    for (int k = colPtr[i]; k < colPtr[i + 1]; ++k) {
      EXPECT_EQ(i, colTgt[k] % 2);
    }
  }
}

TEST(GraphUtils, Color9pt) {
  int nx = 5, ny = 5;
  int n = nx * ny;
  DynamicArray<int> pointer;
  pointer.reserve(n + 1);
  pointer.push_back(0);
  DynamicArray<int> targets;
  targets.reserve(9 * n);
  for (int jy = 0; jy < ny; ++jy) {
    for (int ix = 0; ix < nx; ++ix) {
      int node = ix + jy * nx;
      if (jy > 0) {
        if (ix > 0) {
          targets.push_back(node - nx - 1);
        }
        targets.push_back(node - nx);
        if (ix + 1 < nx) {
          targets.push_back(node - nx + 1);
        }
      }
      if (ix > 0) {
        targets.push_back(node - 1);
      }
      targets.push_back(node);
      if (ix + 1 < nx) {
        targets.push_back(node + 1);
      }
      if (jy + 1 < ny) {
        if (ix > 0) {
          targets.push_back(node + nx - 1);
        }
        targets.push_back(node + nx);
        if (ix + 1 < nx) {
          targets.push_back(node + nx + 1);
        }
      }
      pointer.push_back(isize(targets));
    }
  }
  Graph<int, int> g(pointer, targets);
  auto color = mochi::GreedyColoring(g);
  // Flatten color-graph
  std::vector<int> nodeToColor(n);
  auto colPtr = color.GetPointers();
  // Check that we use 4 colors
  EXPECT_EQ(4 + 1, isize(colPtr));
  //
  auto colTgt = color.GetTargets();
  for (int i = 0; i + 1 < isize(colPtr); ++i) {
    for (int k = colPtr[i]; k < colPtr[i + 1]; ++k) {
      nodeToColor[colTgt[k]] = i;
    }
  }
  //
  auto gPtr = g.GetPointers();
  auto gTgt = g.GetTargets();
  for (int i = 0; i + 1 < isize(gPtr); ++i) {
    std::set<int> colorList;
    for (int k = gPtr[i]; k < gPtr[i + 1]; ++k) {
      if (gTgt[k] == i) {
        continue;
      }
      colorList.insert(nodeToColor[gTgt[k]]);
    }
    // Check that the list of neighboring colors is not empty
    EXPECT_EQ(false, colorList.empty());
    // Check that there are 3 neighboring colors
    EXPECT_EQ(3, colorList.size());
    // Check that the list of neighboring colors does not contain the color for node 'i'
    EXPECT_EQ(colorList.end(), colorList.find(nodeToColor[i]));
  }
}
