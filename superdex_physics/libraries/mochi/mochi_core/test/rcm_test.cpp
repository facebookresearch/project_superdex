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

#include <mochi_core/numbering/rcm.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/graph.h>
#include <mochi_core/utils/graph_utils.h>

#include <gtest/gtest.h>

#include <algorithm>

namespace mochi {

TEST(Rcm, NumbersComponentsFromPseudoPeripheralVertices) {
  EXPECT_TRUE(RenumberByRCM(Graph<int, int>{}).empty());

  // Triangles 0-4-8 and 1-5-9 joined by the paths 4-2-5 and 0-6-9, the path 1-11-3-9, an edge 7-12
  // and an isolated vertex 10. In the triangles' component, the breadth-first searches, each from
  // the vertex of minimum degree, then index, of the component or of the previous last level, go:
  // - from 2, of the component: 4 levels, the last one 6, 11, 3;
  // - from 3, of that level: 5 levels, the last one 8, 4;
  // - from 8: 6 levels, the last one 11;
  // - from 11: 6 levels again, so the component is numbered 11, 3, 1, 9, 5, 6, 2, 0, 4, 8.
  // The edge is numbered from 12, after a search from 7. The components are numbered in order of
  // their smallest vertex, and the whole order is reversed.
  int constexpr kNumVertices = 13;
  DynamicArray<DynamicArray<int>> const paths = {
      {0, 4, 8, 0}, {1, 5, 9, 1}, {4, 2, 5}, {0, 6, 9}, {1, 11, 3, 9}, {7, 12}};
  DynamicArray<DynamicArray<int>> rows(kNumVertices);
  for (DynamicArray<int> const& path : paths) {
    for (int i = 1; i < isize(path); ++i) {
      rows[path[i - 1]].push_back(path[i]);
      rows[path[i]].push_back(path[i - 1]);
    }
  }
  DynamicArray<int> const expected = {10, 7, 12, 8, 4, 0, 2, 6, 5, 9, 1, 3, 11};
  EXPECT_EQ(expected, RenumberByRCM(GraphFromRangeOfRanges<int, int>(rows)));

  // Reversing each row, then adding to each odd row its diagonal entry and a repeat of its first
  // entry and rotating it by its vertex, doesn't change the order, though these entries would
  // change the degree order if counted. The diagonal entry comes first in row 7, and last, after a
  // repeat, in rows 3 and 11.
  for (int v = 0; v < kNumVertices; ++v) {
    DynamicArray<int>& row = rows[v];
    std::ranges::reverse(row);
    if (v % 2 == 1) {
      row.push_back(v);
      row.push_back(int{row.front()});
      std::ranges::rotate(row, row.begin() + v % isize(row));
    }
  }
  EXPECT_EQ(expected, RenumberByRCM(GraphFromRangeOfRanges<int, int>(rows)));

  // Equal-degree neighbors in index order, a separated repeat and a row of diagonal entries only.
  rows = {{2, 1}, {2, 0, 2}, {0, 1}, {3, 3}};
  EXPECT_EQ((DynamicArray<int>{3, 2, 0, 1}), RenumberByRCM(GraphFromRangeOfRanges<int, int>(rows)));
}

} // namespace mochi
