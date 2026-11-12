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

#include <mochi_physics/src/mochi_contact_anchors.h>

#include <gtest/gtest.h>

#include <algorithm>

// Comparisons of friction anchors for tests. Unlike the anchor lookups, which match a sample and a
// collider, they also compare the anchor points.
namespace mochi::test {

// Whether the anchors are equal, field by field and in order.
inline bool AreEqualAnchors(
    DynamicArray<FrictionAnchor> const& a,
    DynamicArray<FrictionAnchor> const& b) {
  return std::ranges::equal(a, b, [](FrictionAnchor const& x, FrictionAnchor const& y) {
    return x.sample == y.sample && x.collider == y.collider && x.point == y.point;
  });
}

// Expects the anchors, field by field and in order.
inline void ExpectAnchors(
    DynamicArray<FrictionAnchor> const& expected,
    DynamicArray<FrictionAnchor> const& actual) {
  ASSERT_EQ(isize(expected), isize(actual));
  for (int i = 0; i < isize(expected); ++i) {
    EXPECT_EQ(expected[i].sample, actual[i].sample) << "anchor " << i;
    EXPECT_EQ(expected[i].collider, actual[i].collider) << "anchor " << i;
    EXPECT_EQ(expected[i].point, actual[i].point) << "anchor " << i;
  }
}

} // namespace mochi::test
