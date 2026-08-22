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
#include <mochi_core/utils/debug.h>

using namespace mochi;

TEST(AssertCast, BasicUse) {
  // This test case verifies that mochi::assert_cast compiles and behaves like static_cast when
  // the cast is legitimate. If the cast was illegitimate, the MOCHI_ASSERT would fail, but we don't
  // currently have a way to test that (without also failing the unit test).
  //
  // TODO: Add the ability to test for assertion failures, similar to how we test for expected error
  // logging.

  struct MyBase1 {
    virtual ~MyBase1() = default;
    int x = 1;
  };

  struct MyBase2 {
    virtual ~MyBase2() = default;
    int y = 2;
  };

  struct MyDerived : public MyBase1, public MyBase2 {
    int z = 3;
  };

  MyDerived obj;

  // Pointers
  auto* pb1 = assert_cast<MyBase1*>(&obj);
  EXPECT_EQ(1, pb1->x);
  auto* pb2 = assert_cast<MyBase2*>(&obj);
  EXPECT_EQ(2, pb2->y);
  auto* pd = assert_cast<MyDerived*>(&obj);
  EXPECT_EQ(3, pd->z);
  auto* pd1 = assert_cast<MyDerived*>(pb1);
  EXPECT_EQ(pd, pd1);
  auto* pd2 = assert_cast<MyDerived*>(pb2);
  EXPECT_EQ(pd, pd2);
  auto const* pcd2 = assert_cast<MyDerived const*>(pb2);
  EXPECT_EQ(pd, pcd2);

  // References
  auto& rb1 = assert_cast<MyBase1&>(obj);
  EXPECT_EQ(1, rb1.x);
  auto& rb2 = assert_cast<MyBase2&>(obj);
  EXPECT_EQ(2, rb2.y);
  auto& rd = assert_cast<MyDerived&>(obj);
  EXPECT_EQ(3, rd.z);
  auto& rd1 = assert_cast<MyDerived&>(rb1);
  EXPECT_EQ(&rd, &rd1);
  auto& rd2 = assert_cast<MyDerived&>(rb2);
  EXPECT_EQ(&rd, &rd2);
  auto const& rcd2 = assert_cast<MyDerived const&>(rb2);
  EXPECT_EQ(&rd, &rcd2);

  // R-Value References
  auto&& rrd = assert_cast<MyDerived&&>(rb1);
  EXPECT_EQ(&rrd, &rd1);
}
