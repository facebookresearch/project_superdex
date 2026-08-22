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

#include <mochi_core/contact/contact_samples_bsh.h>

#include <array>

#include <gtest/gtest.h>

using namespace mochi;

/*
    Create a bsh:

             (0, Internal)
        0			1		   2
     3     4     5    6     7
*/
static ContactSamplesBsh CreateBsh() {
  // Root
  constexpr auto kRootIdx = ContactSamplesBsh::RootIdx();
  ContactSamplesBsh bsh(ContactSamplesBsh::NodeData{.sampleIdx = -1});

  // Layer 1
  auto idx0 = bsh.AddNode(ContactSamplesBsh::NodeData{.sampleIdx = 0}, kRootIdx);
  auto idx1 = bsh.AddNode(ContactSamplesBsh::NodeData{.sampleIdx = 1}, kRootIdx);
  auto idx2 = bsh.AddNode(ContactSamplesBsh::NodeData{.sampleIdx = 2}, kRootIdx);

  // Layer 2
  bsh.AddNode(ContactSamplesBsh::NodeData{.sampleIdx = 3}, idx0);
  bsh.AddNode(ContactSamplesBsh::NodeData{.sampleIdx = 4}, idx0);

  bsh.AddNode(ContactSamplesBsh::NodeData{.sampleIdx = 5}, idx1);
  bsh.AddNode(ContactSamplesBsh::NodeData{.sampleIdx = 6}, idx1);

  bsh.AddNode(ContactSamplesBsh::NodeData{.sampleIdx = 7}, idx2);

  return bsh;
}

constexpr size_t kPointCount = 8;
constexpr std::array<Real3, kPointCount> kPoints = {
    Real3{0_r, 0_r, 0_r}, // 0
    Real3{0_r, 0_r, 1_r}, // 1
    Real3{0_r, 1_r, 0_r}, // 2
    Real3{0_r, 1_r, 1_r}, // 3
    Real3{1_r, 0_r, 0_r}, // 4
    Real3{1_r, 0_r, 1_r}, // 5
    Real3{1_r, 1_r, 0_r}, // 6
    Real3{1_r, 1_r, 0_r}, // 7
};

TEST(ContactSamplesBsh, NumSamples) {
  auto bsh = CreateBsh();
  EXPECT_EQ(bsh.NumSamples(), kPointCount);
}

TEST(ContactSamplesBsh, SampleIdxToNodeIdx) {
  auto bsh = CreateBsh();
  for (int i = 0; i < kPointCount; ++i) {
    EXPECT_EQ(bsh.GetNodeData(bsh.SampleIdxToNodeIdx(i)).sampleIdx, i);
  }
}

TEST(ContactSamplesBsh, Positions) {
  auto bsh = CreateBsh();
  bsh.Update(kPoints);

  for (int i = 0; i < kPointCount; ++i) {
    EXPECT_EQ(bsh.GetNodeData(bsh.SampleIdxToNodeIdx(i)).position, kPoints[i]);
  }
}

TEST(ContactSamplesBsh, Radii) {
  auto bsh = CreateBsh();
  bsh.Update(kPoints);

  EXPECT_EQ(bsh.GetNodeData(bsh.SampleIdxToNodeIdx(3)).radius, 0_r);
  EXPECT_EQ(bsh.GetNodeData(bsh.SampleIdxToNodeIdx(4)).radius, 0_r);
  EXPECT_EQ(bsh.GetNodeData(bsh.SampleIdxToNodeIdx(5)).radius, 0_r);
  EXPECT_EQ(bsh.GetNodeData(bsh.SampleIdxToNodeIdx(6)).radius, 0_r);

  EXPECT_EQ(bsh.GetNodeData(bsh.SampleIdxToNodeIdx(0)).radius, Sqrt(2_r));
  EXPECT_EQ(bsh.GetNodeData(bsh.SampleIdxToNodeIdx(1)).radius, Sqrt(3_r));
  EXPECT_EQ(bsh.GetNodeData(bsh.SampleIdxToNodeIdx(2)).radius, 1_r);
}
