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

#include <mochi_core/utils/monadic.h>

#include <gtest/gtest.h>
#include <optional>

using namespace mochi;

struct TestStruct {
  int a;
  int b;
};

TEST(MonadicTest, AndThenOptional) {
  auto transformer = [](TestStruct const& in) -> std::optional<int> {
    int result = in.a + in.b;
    if (result > 3) {
      return std::nullopt;
    } else {
      return in.a + in.b;
    }
  };

  auto opt1 = std::make_optional(TestStruct{1, 2});
  auto opt2 = std::make_optional(TestStruct{3, 4});
  auto opt3 = std::optional<TestStruct>{};

  auto result1 = AndThen(opt1, transformer);
  auto result2 = AndThen(opt2, transformer);
  auto result3 = AndThen(opt3, transformer);

  EXPECT_TRUE(result1.has_value());
  EXPECT_EQ(result1.value(), 3);
  EXPECT_FALSE(result2.has_value());
  EXPECT_FALSE(result3.has_value());
}

TEST(MonadicTest, AndThenPointer) {
  auto transformer = [](TestStruct const& in) -> int const* {
    int result = in.a + in.b;
    if (result > 3) {
      return nullptr;
    } else {
      return &in.b;
    }
  };

  auto obj = TestStruct{1, 2};
  auto* obj_ptr = &obj;
  auto obj2 = TestStruct{3, 4};
  auto* obj2_ptr = &obj2;
  TestStruct* obj3_ptr = nullptr;

  auto const* result1 = AndThen(obj_ptr, transformer);
  auto const* result2 = AndThen(obj2_ptr, transformer);
  auto const* result3 = AndThen(obj3_ptr, transformer);

  EXPECT_TRUE(result1);
  EXPECT_EQ(result1, &obj_ptr->b);
  EXPECT_FALSE(result2);
  EXPECT_FALSE(result3);
}

TEST(MonadicTest, TransformOptional) {
  auto transformer = [](TestStruct const& in) { return in.a + in.b; };

  auto opt1 = std::make_optional(TestStruct{1, 2});
  auto opt2 = std::optional<TestStruct>{};

  auto result1 = Transform(opt1, transformer);
  auto result2 = Transform(opt2, transformer);

  EXPECT_TRUE(result1.has_value());
  EXPECT_EQ(result1.value(), 3);

  EXPECT_FALSE(result2.has_value());
}

TEST(MonadicTest, TransformPointer) {
  auto transformer = [](TestStruct const& in) { return in.a + in.b; };

  auto obj1 = TestStruct{1, 2};
  auto* obj1_ptr = &obj1;
  TestStruct* obj2_ptr = nullptr;

  auto result1 = Transform(obj1_ptr, transformer);
  auto result2 = Transform(obj2_ptr, transformer);

  EXPECT_TRUE(result1);
  EXPECT_EQ(*result1, 3);

  EXPECT_FALSE(result2);
}
