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

#include <superdex_robotics/ik/bot_ik.h>
#include <superdex_robotics/utils/bot_utils.h>

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/defer.h>
#include <mochi_physics/mochi_physics.h>

#include <gtest/gtest.h>

#include <atomic>
#include <cstdlib>
#include <new>
#include <type_traits>

namespace {

std::atomic<void*> watchedAllocation{nullptr};
std::atomic<bool> callerDeletedBotIK{false};

static_assert(!std::is_destructible_v<superdex::robotics::BotIK>);

void FreeCallerAllocation(void* pointer) {
  if (pointer != nullptr && pointer == watchedAllocation.load()) {
    callerDeletedBotIK = true;
    // Do not pass foreign storage to the caller's allocator when testing the broken path.
    return;
  }
  std::free(pointer);
}

} // namespace

// Windows DLLs bind their global allocation functions independently, as Unreal modules do.
// These replacements belong only to this test executable, never to the native DLL.
void* operator new(std::size_t size) {
  void* pointer = std::malloc(size == 0 ? 1 : size);
  if (pointer == nullptr) {
    std::abort();
  }
  return pointer;
}

void operator delete(void* pointer) noexcept {
  FreeCallerAllocation(pointer);
}

void operator delete(void* pointer, std::size_t) noexcept {
  FreeCallerAllocation(pointer);
}

TEST(BotIKAllocationTest, Destroy_ExplicitEntryPoint_DoesNotUseCallerAllocator) {
  using namespace mochi;
  using namespace mochi::test;
  using namespace superdex::robotics;

  Context* const context = CreateContext(0);
  ASSERT_NE(context, nullptr);
  MOCHI_DEFER(DestroyContext(context));

  BotPrefab prefab;
  prefab.name = "allocation_test";
  for (int i = 0; i < 2; ++i) {
    BotLinkPrefab link;
    link.name = i == 0 ? "root" : "tip";
    link.parentLink = i == 0 ? kIndexNone : 0;
    link.mass = 1_r;
    link.centerOfMass = Real3{};
    link.momentOfInertia = Real6{1_r, 0_r, 0_r, 1_r, 0_r, 1_r};
    prefab.links.push_back(link);
    BotJointPrefab joint;
    joint.name = i == 0 ? "root_joint" : "tip_joint";
    joint.type = i == 0 ? ArticulatedJointType::Hard : ArticulatedJointType::Revolute;
    joint.axis = Real3{0_r, 0_r, 1_r};
    prefab.joints.push_back(joint);
  }
  prefab.defaultPose = {0_r};
  RebuildBotData(prefab, ExpectOK{});

  BotIK* ik = BotIK::Create(context, prefab, {}, ExpectOK{});
  ASSERT_NE(ik, nullptr);
  MOCHI_DEFER(BotIK::Destroy(ik));
  ik->SetRotationGoal(1, {}, {}, 1000_r, ExpectOK{});
  EXPECT_TRUE(ik->SolveIK(ExpectOK{}));

  watchedAllocation = ik;
  callerDeletedBotIK = false;
  BotIK::Destroy(ik);
  watchedAllocation = nullptr;
  ik = nullptr;
  BotIK::Destroy(ik);

  EXPECT_FALSE(callerDeletedBotIK.load());
  Scene* const scene = context->CreateScene("StillUsable");
  ASSERT_NE(scene, nullptr);
  context->DestroyScene(scene);
}
