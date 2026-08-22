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
#include <mochi_core/utils/coordinate_space.h>

using namespace mochi;

TEST(CoordinateSpace, Default) {
  CoordinateSpace cs;
  EXPECT_EQ(CoordinateSpaceAxes::Default, cs.axes);
  EXPECT_EQ(1.0, cs.unitsPerMeter);
}

TEST(CoordinateSpace, ValidateAxes) {
  // Build a lookup table for the valid axis combinations.
  bool isValidAxes[1 << 9] = {};
  auto const& enumInfo = SReflect::GetTypeInfo<CoordinateSpaceAxes>();
  for (auto const& item : enumInfo._items) {
    ASSERT_LT(item._value, uint64_t(1) << 9); // Out-of-range?
    EXPECT_FALSE(isValidAxes[item._value]); // Redundant?
    isValidAxes[item._value] = true;
  }

  // Validate all combinations of the lower 9 bits (3 per axis) of CoordinateSpaceAxes.
  for (int x = 0; x < 8; ++x) {
    for (int y = 0; y < 8; ++y) {
      for (int z = 0; z < 8; ++z) {
        int const axes = (x << 0) | (y << 3) | (z << 6);
        CoordinateSpace space(static_cast<CoordinateSpaceAxes>(axes), 1.0);
        if (isValidAxes[axes]) {
          space.Validate(test::ExpectOK{});
        } else {
          space.Validate(test::ExpectNotOK{});
        }
      }
    }
  }
}

TEST(CoordinateSpace, ValidateScale) {
  CoordinateSpace cs;
  cs.Validate(test::ExpectOK{});
  cs.unitsPerMeter = 0.0;
  cs.Validate(test::ExpectNotOK{});
  cs.unitsPerMeter = -1.0;
  cs.Validate(test::ExpectNotOK{});
  cs.unitsPerMeter = kInf;
  cs.Validate(test::ExpectNotOK{});
  cs.unitsPerMeter = 100.0;
  cs.Validate(test::ExpectOK{});
}

namespace {

// The unit vector that an axis letter names, in a shared (right, up, forward) semantic basis.
Real3 SemanticVector(char letter) {
  switch (letter) {
    case 'R':
      return {1_r, 0_r, 0_r};
    case 'L':
      return {-1_r, 0_r, 0_r};
    case 'U':
      return {0_r, 1_r, 0_r};
    case 'D':
      return {0_r, -1_r, 0_r};
    case 'F':
      return {0_r, 0_r, 1_r};
    case 'B':
      return {0_r, 0_r, -1_r};
  }
  ADD_FAILURE() << "Unknown axis letter: " << letter;
  return {};
}

} // namespace

TEST(CoordinateSpace, SemanticAxesMatchTheConventionName) {
  auto const& enumInfo = SReflect::GetTypeInfo<CoordinateSpaceAxes>();
  for (auto const& item : enumInfo._items) {
    SCOPED_TRACE(item._name);
    CoordinateSpace const space{static_cast<CoordinateSpaceAxes>(item._value), 1.0};

    // The name's three letters give the semantic direction of +X, +Y and +Z in order. So the
    // component of "right" along axis i is the right-component of whatever direction letter i
    // names: 1 for R, -1 for L, and 0 for the rest. Likewise for up and forward.
    Real3 expectedRight{};
    Real3 expectedUp{};
    Real3 expectedForward{};
    for (int i = 0; i < 3; ++i) {
      Real3 const named = SemanticVector(item._name[i]);
      expectedRight[i] = named[0];
      expectedUp[i] = named[1];
      expectedForward[i] = named[2];
    }

    EXPECT_EQ(expectedRight, space.GetRight());
    EXPECT_EQ(expectedUp, space.GetUp());
    EXPECT_EQ(expectedForward, space.GetForward());
  }
}

TEST(CoordinateSpace, SemanticAxesOfCommonConventions) {
  // Mochi's own (FLU): X-forward, Y-left, Z-up.
  EXPECT_EQ(Real3(0_r, -1_r, 0_r), CoordinateSpace::Default().GetRight());
  EXPECT_EQ(Real3(0_r, 0_r, 1_r), CoordinateSpace::Default().GetUp());
  EXPECT_EQ(Real3(1_r, 0_r, 0_r), CoordinateSpace::Default().GetForward());

  // Filament and OpenGL (RUB): X-right, Y-up, Z-backward.
  EXPECT_EQ(Real3(1_r, 0_r, 0_r), CoordinateSpace::Filament().GetRight());
  EXPECT_EQ(Real3(0_r, 1_r, 0_r), CoordinateSpace::Filament().GetUp());
  EXPECT_EQ(Real3(0_r, 0_r, -1_r), CoordinateSpace::Filament().GetForward());

  // Unreal (FRU): X-forward, Y-right, Z-up. The unit scale must not leak into the axes.
  EXPECT_EQ(Real3(0_r, 1_r, 0_r), CoordinateSpace::Unreal().GetRight());
  EXPECT_EQ(Real3(0_r, 0_r, 1_r), CoordinateSpace::Unreal().GetUp());
  EXPECT_EQ(Real3(1_r, 0_r, 0_r), CoordinateSpace::Unreal().GetForward());
}
