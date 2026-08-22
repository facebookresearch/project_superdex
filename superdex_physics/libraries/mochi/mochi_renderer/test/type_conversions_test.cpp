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
#include <mochi_renderer/type_conversions.h>

#include <gtest/gtest.h>

#include <type_traits>

using namespace mochi;
using namespace mochi_renderer;

// Distinct entries in every slot, so any transposition or component swap is detectable.
static Matrix4x4r MakeAsymmetric4x4() {
  Matrix4x4r m{};
  for (size_t row = 0; row < 4; ++row) {
    for (size_t col = 0; col < 4; ++col) {
      m[row][col] = static_cast<real>(1 + row * 4 + col);
    }
  }
  return m;
}

TEST(TypeConversions, Vector3) {
  // The scalar type is preserved unless an output type is requested.
  static_assert(std::is_same_v<decltype(ToFilament(Float3{})), filament::math::float3>);
  static_assert(std::is_same_v<decltype(ToFilament<double>(Float3{})), filament::math::double3>);
  static_assert(std::is_same_v<decltype(ToMochi(filament::math::double3{})), Double3>);
  static_assert(std::is_same_v<decltype(ToMochi<real>(filament::math::double3{})), Real3>);

  // Float3 <--> filament::math::float3
  {
    Float3 const mochiValue{1.0f, 2.0f, 3.0f};
    filament::math::float3 const filamentValue = ToFilament(mochiValue);
    EXPECT_FLOAT_EQ(1.0f, filamentValue.x);
    EXPECT_FLOAT_EQ(2.0f, filamentValue.y);
    EXPECT_FLOAT_EQ(3.0f, filamentValue.z);
    EXPECT_EQ(mochiValue, ToMochi(filamentValue)); // exact
    EXPECT_EQ(mochiValue, StaticCast<Float3>(ToMochi<real>(filamentValue))); // exact
  }

  // Double3 <--> filament::math::float3
  {
    Double3 const mochiValue{1.0, 2.0, 3.0};
    filament::math::float3 const filamentValue = ToFilament<float>(mochiValue);
    EXPECT_FLOAT_EQ(1.0f, filamentValue.x);
    EXPECT_FLOAT_EQ(2.0f, filamentValue.y);
    EXPECT_FLOAT_EQ(3.0f, filamentValue.z);
    EXPECT_EQ(mochiValue, ToMochi<double>(filamentValue)); // exact
  }
}

TEST(TypeConversions, Vector4) {
  // The scalar type is preserved unless an output type is requested.
  static_assert(std::is_same_v<decltype(ToFilament(Float4{})), filament::math::float4>);
  static_assert(std::is_same_v<decltype(ToFilament<double>(Float4{})), filament::math::double4>);
  static_assert(std::is_same_v<decltype(ToMochi(filament::math::double4{})), NdArray<double, 4>>);
  static_assert(std::is_same_v<decltype(ToMochi<real>(filament::math::double4{})), Real4>);

  // Float4 <--> filament::math::float4
  {
    Float4 const mochiValue{1.0f, 2.0f, 3.0f, 4.0f};
    filament::math::float4 const filamentValue = ToFilament(mochiValue);
    EXPECT_FLOAT_EQ(1.0f, filamentValue.x);
    EXPECT_FLOAT_EQ(2.0f, filamentValue.y);
    EXPECT_FLOAT_EQ(3.0f, filamentValue.z);
    EXPECT_FLOAT_EQ(4.0f, filamentValue.w);
    EXPECT_EQ(mochiValue, ToMochi(filamentValue)); // exact
    EXPECT_EQ(mochiValue, StaticCast<Float4>(ToMochi<real>(filamentValue))); // exact
  }

  // Double4 <--> filament::math::float4
  {
    NdArray<double, 4> const mochiValue{1.0, 2.0, 3.0, 4.0};
    filament::math::float4 const filamentValue = ToFilament<float>(mochiValue);
    EXPECT_FLOAT_EQ(1.0f, filamentValue.x);
    EXPECT_FLOAT_EQ(2.0f, filamentValue.y);
    EXPECT_FLOAT_EQ(3.0f, filamentValue.z);
    EXPECT_FLOAT_EQ(4.0f, filamentValue.w);
    EXPECT_EQ(mochiValue, ToMochi<double>(filamentValue)); // exact
  }
}

TEST(TypeConversions, Quaternion) {
  // mochi::Quaternion is always real-valued, so only the Filament side is configurable.
  static_assert(std::is_same_v<
                decltype(ToFilament(mochi::Quaternion{})),
                filament::math::details::TQuaternion<real>>);
  static_assert(
      std::is_same_v<decltype(ToFilament<float>(mochi::Quaternion{})), filament::math::quatf>);
  static_assert(std::is_same_v<decltype(ToMochi(filament::math::quat{})), mochi::Quaternion>);

  // Four distinct components catch an XYZW/WXYZ mix-up in either direction.
  mochi::Quaternion const mochiValue{0.1_r, 0.2_r, 0.3_r, 0.4_r}; // (x, y, z, w)
  filament::math::quatf const filamentValue = ToFilament<float>(mochiValue);
  EXPECT_FLOAT_EQ(0.1f, filamentValue.x);
  EXPECT_FLOAT_EQ(0.2f, filamentValue.y);
  EXPECT_FLOAT_EQ(0.3f, filamentValue.z);
  EXPECT_FLOAT_EQ(0.4f, filamentValue.w);
  EXPECT_EQ(
      StaticCast<Float4>(mochiValue.ToReal4()),
      StaticCast<Float4>(ToMochi(filamentValue).ToReal4()));
}

TEST(TypeConversions, Matrix4x4) {
  // The scalar type is preserved unless an output type is requested.
  static_assert(
      std::is_same_v<decltype(ToFilament(Matrix4x4r{})), filament::math::details::TMat44<real>>);
  static_assert(std::is_same_v<decltype(ToFilament<float>(Matrix4x4r{})), filament::math::mat4f>);
  static_assert(std::is_same_v<decltype(ToMochi(filament::math::mat4f{})), Matrix4x4f>);
  static_assert(std::is_same_v<decltype(ToMochi<real>(filament::math::mat4f{})), Matrix4x4r>);

  mochi::Matrix4x4r const mochiValue = MakeAsymmetric4x4();
  filament::math::mat4f const filamentValue = ToFilament<float>(mochiValue);

  // Mochi is row-major and Filament is column-major, so filament[col][row] == mochi[row][col].
  for (size_t row = 0; row < 4; ++row) {
    for (size_t col = 0; col < 4; ++col) {
      EXPECT_FLOAT_EQ(mochiValue[row][col], filamentValue[col][row]);
    }
  }
  EXPECT_EQ(StaticCast<Matrix4x4f>(mochiValue), ToMochi(filamentValue));

  // The translation of a Mochi transform lives in the last column, which is Filament's last row.
  EXPECT_FLOAT_EQ(static_cast<float>(mochiValue[0][3]), filamentValue[3].x);
  EXPECT_FLOAT_EQ(static_cast<float>(mochiValue[1][3]), filamentValue[3].y);
  EXPECT_FLOAT_EQ(static_cast<float>(mochiValue[2][3]), filamentValue[3].z);
}
