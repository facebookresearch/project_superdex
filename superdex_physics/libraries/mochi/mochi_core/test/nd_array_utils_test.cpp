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
#include <mochi_core/utils/nd_array_utils.h>
#include <mochi_core/utils/vmatrix.h>

using namespace mochi;

TEST(NdArrayUtils, StaticCast) {
  // 1D NdArray real to int
  {
    constexpr Real3 r3 = {1.1_r, 2.2_r, 3.3_r};
    static_assert(Int3{1, 2, 3} == StaticCast<Int3>(r3));
    EXPECT_EQ(Int3(1, 2, 3), StaticCast<Int3>(r3));
  }

  // 2D NdArray real to int
  {
    using Matrix2x2i = NdArray<int, 2, 2>;
    constexpr Matrix2x2r m22{Real2{1.1_r, 2.2_r}, Real2{3.3_r, 4.4_r}};
    static_assert(Matrix2x2i{Int2{1, 2}, Int2{3, 4}} == StaticCast<Matrix2x2i>(m22));
    EXPECT_EQ((Matrix2x2i{Int2{1, 2}, Int2{3, 4}}), StaticCast<Matrix2x2i>(m22));
  }

  // VMatrix real to int
  {
    using VMatrix4x4i = NdArray<Simd<int, 4>, 4>;
    VMatrix4x4r const m22{
        Vec4r{1.1_r, 2.2_r, 3.3_r, 4.4_r},
        Vec4r{2.2_r, 3.3_r, 4.4_r, 5.5_r},
        Vec4r{3.3_r, 4.4_r, 5.5_r, 6.6_r},
        Vec4r{4.4_r, 5.5_r, 6.6_r, 7.7_r}};
    auto m22i = StaticCast<VMatrix4x4i>(m22);
    EXPECT_EQ(Vec4i(1, 2, 3, 4), m22i[0]);
    EXPECT_EQ(Vec4i(2, 3, 4, 5), m22i[1]);
    EXPECT_EQ(Vec4i(3, 4, 5, 6), m22i[2]);
    EXPECT_EQ(Vec4i(4, 5, 6, 7), m22i[3]);
  }
}

TEST(NdArrayUtils, Sign) {
  // Int
  EXPECT_EQ((Int3{1, 1, 1}), Sign(Int3{0, 0, 0}));
  EXPECT_EQ((Int3{1, -1, 1}), Sign(Int3{1, -2, 3}));
  EXPECT_EQ((Int3{-1, 1, -1}), Sign(Int3{-4, 5, -6}));

  // Real4
  EXPECT_EQ((Real4{1_r, 1_r, 1_r, 1_r}), Sign(Real4{0_r, 0_r, 0_r, 0_r}));
  EXPECT_EQ((Real4{1_r, -1_r, 1_r, -1_r}), Sign(Real4{0.1_r, -0.2_r, 0.3_r, -0.4_r}));
  EXPECT_EQ((Real4{-1_r, 1_r, -1_r, 1_r}), Sign(Real4{-0.5_r, 0.6_r, -0.7_r, 0.8_r}));

  // NdArray<real, 2, 2>
  EXPECT_EQ(
      (NdArray<real, 2, 2>{Real2{1_r, 1_r}, Real2{1_r, 1_r}}),
      Sign(NdArray<real, 2, 2>{Real2{0_r, 0_r}, Real2{0_r, 0_r}}));
  EXPECT_EQ(
      (NdArray<real, 2, 2>{Real2{1_r, -1_r}, Real2{-1_r, 1_r}}),
      Sign(NdArray<real, 2, 2>{Real2{1_r, -2_r}, Real2{-3_r, 4_r}}));
}
