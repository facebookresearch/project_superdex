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

#include <gtest/gtest.h>

#include <mochi_core/utils/graph_utils.h>

#include <algorithm>
#include <limits>
#include <vector>

namespace mochi::test {

static void TestPriority() {
  std::vector<int> priorities{2, 5, 3, 2, 3, 1};
  auto n = isize(priorities);
  PriorityQueue pq(n, [&priorities](int i) { return priorities[i]; });

  auto invalid = std::numeric_limits<int>::max();

  for (int i = 0; i < n; ++i) {
    auto min = *std::min_element(priorities.begin(), priorities.end());
    auto next = pq.RemoveMinElement();
    EXPECT_EQ(priorities[next], min);
    priorities[next] = invalid;
    auto up = (next + 2) % n;
    if (priorities[up] > 0 && priorities[up] != invalid) {
      pq.Decrement(up);
      --priorities[up];
    }
    auto down = (next + n - 2) % n;
    if (priorities[down] > 0 && priorities[down] != invalid) {
      pq.Decrement(down);
      --priorities[down];
    }
  }
}

} // namespace mochi::test

TEST(Algorithms, PriorityQueue) {
  mochi::test::TestPriority();
}
