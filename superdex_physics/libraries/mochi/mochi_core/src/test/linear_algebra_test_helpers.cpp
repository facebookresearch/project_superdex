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

#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/test/linear_algebra_test_helpers.h>
#include <mochi_core/utils/dynamic_array.h>

#include <set>

namespace mochi::test {

static void Connection3pt(int i, int n, int offset, std::set<int>& nodes) {
  MOCHI_ASSERT_VERBOSE(n >= 2, "Insufficient number of points");
  if (i > 0) {
    nodes.insert(offset - 1);
  }
  nodes.insert(offset);
  if (i + 1 < n) {
    nodes.insert(offset + 1);
  }
}

static void Connection9pt(int ix, int nx, int iy, int ny, int offset, std::set<int>& nodes) {
  MOCHI_ASSERT_VERBOSE(nx >= 2, "Insufficient number of points");
  MOCHI_ASSERT_VERBOSE(ny >= 2, "Insufficient number of points");
  if (iy > 0) {
    Connection3pt(ix, nx, offset - nx, nodes);
  }
  Connection3pt(ix, nx, offset, nodes);
  if (iy + 1 < ny) {
    Connection3pt(ix, nx, offset + nx, nodes);
  }
}

/// @brief Create the graph of node-to-node connectivity
/// for a brick [0, 1] x [0, 1] x [0, 1] discretized with
/// 8-noded hexahedra (forming an orthogonal grid)
///
/// @param[in] nx Number of elements in the x-direction
/// @param[in] ny Number of elements in the y-direction
/// @param[in] nz Number of elements in the z-direction
/// @param[out] rowPtr Array similar to the row pointer array in a CSR matrix
/// @param[out] nodeIdx Array similar to the column index pointer array in a CSR matrix
void MakeGraphBrick(int nx, int ny, int nz, DynamicArray<int>& rowPtr, DynamicArray<int>& nodeIdx) {
  int node_x = nx + 1;
  int node_y = ny + 1;
  int node_z = nz + 1;
  int totalNodes = node_x * node_y * node_z;
  int maxNnzPerNode = 27;

  rowPtr.resize(totalNodes + 1, 0);

  nodeIdx.clear();
  nodeIdx.reserve(maxNnzPerNode * totalNodes);

  for (int iz = 0; iz < node_z; ++iz) {
    for (int iy = 0; iy < node_y; ++iy) {
      for (int ix = 0; ix < node_x; ++ix) {
        int node = ix + iy * node_x + iz * node_x * node_y;
        std::set<int> currentIdx;
        if (iz > 0) {
          Connection9pt(ix, node_x, iy, node_y, node - node_x * node_y, currentIdx);
        }
        Connection9pt(ix, node_x, iy, node_y, node, currentIdx);
        if (iz + 1 < node_z) {
          Connection9pt(ix, node_x, iy, node_y, node + node_x * node_y, currentIdx);
        }
        for (auto const& idx : currentIdx) {
          nodeIdx.push_back(idx);
        }
        rowPtr[node + 1] = rowPtr[node] + isize(currentIdx);
      }
    }
  }
}

} // namespace mochi::test
