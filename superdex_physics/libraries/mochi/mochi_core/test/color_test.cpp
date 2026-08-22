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

#include <mochi_core/utils/color.h>

using namespace mochi;

TEST(Color, FromRGBA) {
  Color c = MakeColor(1, 2, 3, 4);
  EXPECT_EQ(1, c[0]);
  EXPECT_EQ(2, c[1]);
  EXPECT_EQ(3, c[2]);
  EXPECT_EQ(4, c[3]);

  c = MakeColor(252, 253, 254);
  EXPECT_EQ(252, c[0]);
  EXPECT_EQ(253, c[1]);
  EXPECT_EQ(254, c[2]);
  EXPECT_EQ(255, c[3]); // default param
}

TEST(Color, FromUint32) {
  EXPECT_EQ(MakeColor(0, 0, 0, 0), MakeColor(0x00000000));
  EXPECT_EQ(MakeColor(1, 2, 3, 4), MakeColor(0x01020304));
  EXPECT_EQ(MakeColor(252, 253, 254, 255), MakeColor(0xFCFDFEFF));
  EXPECT_EQ(MakeColor(255, 255, 255, 255), MakeColor(0xFFFFFFFF));
}
