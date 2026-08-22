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

#include <mochi_core/utils/defer.h>

#include <gtest/gtest.h>

#include <exception>

using namespace mochi;

TEST(Defer, Macro) {
  int value = 0;

  // Single Defer
  {
    MOCHI_DEFER(++value);
    EXPECT_EQ(0, value);
  }
  EXPECT_EQ(1, value);

  // Multiple Defer
  {
    value = 0;
    MOCHI_DEFER(value = 2);
    MOCHI_DEFER(value = 3);
    EXPECT_EQ(0, value);
  }
  EXPECT_EQ(2, value); // stack unwinds in reverse order

// WORKAROUND FOR COMPILER BUG:
// The try-catch block does not function correctly in some optimized builds.
#if !MOCHI_COMPILER_CLANG || !MOCHI_PLATFORM_MACOS || MOCHI_DEBUG
  // Exception thrown
  struct MyException : public std::exception {};
  value = 0;
  try {
    MOCHI_DEFER(value = 4);
    EXPECT_EQ(0, value);
    throw MyException{};
  } catch (MyException const&) {
    EXPECT_EQ(4, value);
  }
#endif
}
