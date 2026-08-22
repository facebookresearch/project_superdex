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
#include <mochi_core/utils/simd.h>

#include <picojson/picojson.h>

using namespace mochi;

TEST(Vec4r, Reflection) {
  using SimdT = Simd<real, 4>;
  SimdT vec{-1_r, 0_r, 0.5_r, 1_r};

  auto const& typeInfo = SReflect::GetTypeInfo<SimdT>();

  // Serialization
  EXPECT_STREQ("[-1,0,0.5,1]", SReflect::ToJsonString(vec, false).c_str());
  EXPECT_EQ(vec, SReflect::FromJsonString<SimdT>("[-1,0,0.5,1]"));

  // Type Introspection
  if constexpr (MOCHI_USE_DOUBLE_PRECISION) {
    EXPECT_STREQ("Simd<double,4>", typeInfo._name);
    EXPECT_STREQ("mochi::Simd<double,4>", typeInfo._nameWithNamespace);
  } else {
    EXPECT_STREQ("Simd<float,4>", typeInfo._name);
    EXPECT_STREQ("mochi::Simd<float,4>", typeInfo._nameWithNamespace);
  }
  EXPECT_EQ(SReflect::CoreType::CT_array, typeInfo._coreType);
  EXPECT_EQ(sizeof(SimdT), typeInfo._sizeInBytes);
  EXPECT_EQ(alignof(SimdT), typeInfo._alignment);
  EXPECT_EQ(&SReflect::GetTypeInfo<real>(), typeInfo._innerTypeInfo);

  // Factor Creation (does not require compile-time access to SimdT)
  void* newObj = typeInfo.New();
  picojson::value json = picojson::object();
  typeInfo.Serialize(newObj, json);
  EXPECT_STREQ("[0,0,0,0]", json.serialize(false).c_str());
  typeInfo.Delete(newObj);
}

TEST(Vec8i, Reflection) {
  using SimdT = Simd<int, 8>;
  SimdT vec{1, 2, 3, 4, 5, 6, 7, 8};

  auto const& typeInfo = SReflect::GetTypeInfo<SimdT>();

  // Serialization
  EXPECT_STREQ("[1,2,3,4,5,6,7,8]", SReflect::ToJsonString(vec, false).c_str());
  EXPECT_EQ(vec, SReflect::FromJsonString<SimdT>("[1,2,3,4,5,6,7,8]"));

  // Type Introspection
  EXPECT_STREQ("Simd<int32,8>", typeInfo._name);
  EXPECT_STREQ("mochi::Simd<int32,8>", typeInfo._nameWithNamespace);
  EXPECT_EQ(SReflect::CoreType::CT_array, typeInfo._coreType);
  EXPECT_EQ(sizeof(SimdT), typeInfo._sizeInBytes);
  EXPECT_EQ(alignof(SimdT), typeInfo._alignment);
  EXPECT_EQ(&SReflect::GetTypeInfo<int>(), typeInfo._innerTypeInfo);

  // Factor Creation (does not require compile-time access to SimdT)
  void* newObj = typeInfo.New();
  picojson::value json = picojson::object();
  typeInfo.Serialize(newObj, json);
  EXPECT_STREQ("[0,0,0,0,0,0,0,0]", json.serialize(false).c_str());
  typeInfo.Delete(newObj);
}
