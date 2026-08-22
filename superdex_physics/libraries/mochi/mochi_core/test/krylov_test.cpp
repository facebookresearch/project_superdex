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

#include <mochi_core/linear_algebra/base_tools.h>
#include <mochi_core/linear_algebra/krylov_interop.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/simd.h>
#include <mochi_core/utils/vmatrix.h>

#include <gtest/gtest.h>

#include <algorithm>

using namespace mochi;

static void TestIntOrEmpty() {
  mochi::details::IntOrEmpty<-1> v6{6};
  mochi::details::IntOrEmpty<5> v5;
  EXPECT_TRUE(4 < v6 && v6 < 7);

  EXPECT_FALSE(6 < v6);
  EXPECT_FALSE(v6 < 6);
  EXPECT_TRUE(v6 <= 6 && 6 <= v6);

  EXPECT_TRUE(4 < v5 && v5 < 6);
  EXPECT_FALSE(5 < v5 || 6 < v5);
  EXPECT_TRUE(5 <= v5 && v5 <= 5);
  EXPECT_FALSE(6 <= v5 || v5 <= 4);
}

TEST(Krylov, SizesMemory) {
  static_assert(sizeof(details::Sizes<3, 2, 1>) == 1);
  static_assert(sizeof(details::Sizes<0, 0, 0>) == 1);
  static_assert(sizeof(details::Sizes<krylov::kDynamic, 0, 1>) == sizeof(int));
  static_assert(sizeof(details::Sizes<1, krylov::kDynamic, 0>) == sizeof(int));
  static_assert(sizeof(details::Sizes<0, 1, krylov::kDynamic>) == sizeof(int));
  static_assert(sizeof(details::Sizes<3, krylov::kDynamic, krylov::kDynamic>) == 2 * sizeof(int));
  static_assert(sizeof(details::Sizes<krylov::kDynamic, 0, krylov::kDynamic>) == 2 * sizeof(int));
  static_assert(sizeof(details::Sizes<krylov::kDynamic, krylov::kDynamic, 3>) == 2 * sizeof(int));
  static_assert(
      sizeof(details::Sizes<krylov::kDynamic, krylov::kDynamic, krylov::kDynamic>) ==
      3 * sizeof(int));
}

TEST(Krylov, FixedSizeBaseMatrixMemory) {
  constexpr size_t kVariableStorageAlignment =
      alignof(krylov::details::BaseStorage<float, -1, krylov::Ownership::Owner>);
  static_assert(
      sizeof(krylov::BaseMatrix<float, -1, 5, krylov::Direction::ColMajor>) ==
          std::max(kVariableStorageAlignment, sizeof(int)) +
              sizeof(krylov::details::BaseStorage<float, -1, krylov::Ownership::Owner>),
      "Extra-memory introduced");
  static_assert(
      sizeof(krylov::BaseMatrix<float, 3, 5, krylov::Direction::ColMajor>) == sizeof(float) * 3 * 5,
      "Extra-memory introduced");
}

TEST(Krylov, IntOrEmpty) {
  TestIntOrEmpty();
}

TEST(Krylov, AsMatrixView) {
  auto checkResult = [](auto&& A, int nRows, int nCols) {
    static_assert(IsView(details::MatTraits<decltype(A)>::kOwner));
    EXPECT_EQ(nRows, A.Rows());
    EXPECT_EQ(nCols, A.Cols());
    for (int i = 0, counter = 0; i < nRows; ++i) {
      for (int j = 0; j < nCols; ++j, ++counter) {
        EXPECT_EQ(A(i, j), real(counter));
      }
    }
  };

  // 3x3 overload.
  VMatrix3x3r mat3x3 = {
      Vec4r{0_r, 1_r, 2_r, -1_r}, Vec4r{3_r, 4_r, 5_r, -2_r}, Vec4r{6_r, 7_r, 8_r, -3_r}};
  auto view3x3 = AsMatrixView(mat3x3);
  checkResult(view3x3, 3, 3);

  // 4x4 overload.
  VMatrix4x4r mat4x4 = {
      Vec4r{0_r, 1_r, 2_r, 3_r},
      Vec4r{4_r, 5_r, 6_r, 7_r},
      Vec4r{8_r, 9_r, 10_r, 11_r},
      Vec4r{12_r, 13_r, 14_r, 15_r}};
  auto view4x4 = AsMatrixView(mat4x4);
  checkResult(view4x4, 4, 4);
}

TEST(Krylov, AsColumnVectorView) {
  Vec8r vec{0_r, 1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r};
  auto view1 = AsColumnVectorView<1>(vec);
  auto view2 = AsColumnVectorView<3>(vec);
  auto view3 = AsColumnVectorView(vec);
  static_assert(IsView(details::MatTraits<decltype(view1)>::kOwner));
  EXPECT_EQ(1, view1.Rows());
  EXPECT_EQ(3, view2.Rows());
  EXPECT_EQ(8, view3.Rows());
  for (int i = 0; i < 8; ++i) {
    if (i < view1.Rows()) {
      EXPECT_EQ(view1(i), real(i));
    }
    if (i < view2.Rows()) {
      EXPECT_EQ(view2(i), real(i));
    }
    EXPECT_EQ(view3(i), real(i));
  }
}

TEST(Krylov, TransformToRawPose) {
  TransformRT rt{
      Quaternion::FromAxisAngle(Real3{0.5_r, 0.5_r, 0.5_r}, kPI / 4.0),
      Real3{0.382683456_r, 0_r, 0.923879504_r}};
  ColumnVector<real, RigidSize::kAll> raw;

  TransformToRawPose(rt, AsView(raw));
  auto rt2 = TransformFromRawPose(AsConstView(raw));

  EXPECT_NEAR_EQ(rt.GetTranslation(), rt2.GetTranslation());
  EXPECT_NEAR_EQ(rt.GetRotation(), rt2.GetRotation());
}
