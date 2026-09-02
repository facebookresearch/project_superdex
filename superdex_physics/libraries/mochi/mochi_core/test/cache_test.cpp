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

#include <mochi_core/memory/cache.h>
#include <mochi_core/utils/basic_utils.h>

#include <gtest/gtest.h>

#include <cstddef>

using namespace mochi;

TEST(CacheLineInfo, Default) {
  CacheLineInfo info;
  EXPECT_EQ(size_t{MOCHI_CONSERVATIVE_CACHE_LINE_SIZE}, info.size);
  EXPECT_FALSE(info.detected);
}

TEST(CacheLineInfo, GetCacheLineInfo) {
  auto const info = GetCacheLineInfo();
  if (info.detected) {
    // 64 bytes is the smallest cache line of any plausible CPU at the time of writing.
    EXPECT_LE(size_t{64}, info.size);
    EXPECT_GE(size_t{MOCHI_CONSERVATIVE_CACHE_LINE_SIZE}, info.size);
    EXPECT_TRUE(IsPowerOfTwo(info.size));
  } else {
    EXPECT_EQ(size_t{MOCHI_CONSERVATIVE_CACHE_LINE_SIZE}, info.size);
  }
}
