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

#include "mochi_physics_test_fixture.h"

#include <mochi_physics/mochi_physics.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <initializer_list>
#include <string>
#include <string_view>

using namespace mochi;

// TODO: Move existing soft-skinned actor coverage here.

class CreateSoftSkinnedActorNameTest : public test::MochiSceneTestBase {
 protected:
  SoftSkinnedActorParams MakeParams(
      std::initializer_list<std::string_view> softNames,
      std::string_view linkName = "bone",
      std::string_view skeletonName = "Skel") {
    SoftSkinnedActorParams params;
    auto& skeleton = params.skeletonParams;
    skeleton.name = DynamicString(skeletonName);
    skeleton.joints = {{.type = ArticulatedJointType::Free}};
    skeleton.links = {
        {.name = DynamicString(linkName),
         .parentLink = -1,
         .shape = test::CreateUnitCubeTetMeshShape(_mochiContext),
         .colliderType = ColliderType::None}};

    for (auto softName : softNames) {
      SoftActorParams softParams;
      softParams.name = DynamicString(softName);
      softParams.shape = test::CreateUnitCubeTetSoftShape(_mochiContext);
      softParams.hasGravity = false;
      params.softParams.push_back(softParams);
    }
    return params;
  }
};

TEST_F(CreateSoftSkinnedActorNameTest, RejectsInvalidNestedSoftLocalNamesBeforeMutation) {
  auto const expectRejected = [&](SoftSkinnedActorParams params) {
    int const numActorsBefore = _scene->GetNumActors();
    EXPECT_EQ(nullptr, _scene->CreateSoftSkinnedActor(params, test::ExpectNotOK{}));
    EXPECT_EQ(numActorsBefore, _scene->GetNumActors());
  };

  expectRejected(MakeParams({"soft/name"}));
  expectRejected(MakeParams({"soft\\name"}));

  std::string const softNameWithNull{"soft\0name", 9};
  expectRejected(MakeParams({std::string_view{softNameWithNull.data(), softNameWithNull.size()}}));

  expectRejected(MakeParams({"dup", "dup"}));
  expectRejected(MakeParams({"bone"}));
  expectRejected(MakeParams({"link_0"}, ""));

  auto wrongAttachmentCount = MakeParams({"soft"});
  wrongAttachmentCount.softAttachLinks.push_back("bone");
  wrongAttachmentCount.softAttachLinks.push_back("bone");
  expectRejected(wrongAttachmentCount);

  auto emptyAttachment = MakeParams({"soft"});
  emptyAttachment.softAttachLinks.push_back("");
  expectRejected(emptyAttachment);

  auto slashAttachment = MakeParams({"soft"});
  slashAttachment.softAttachLinks.push_back("bone/child");
  expectRejected(slashAttachment);

  auto backslashAttachment = MakeParams({"soft"});
  backslashAttachment.softAttachLinks.push_back("bone\\child");
  expectRejected(backslashAttachment);

  std::string const attachLinkWithNull{"bone\0child", 10};
  auto nullAttachment = MakeParams({"soft"});
  nullAttachment.softAttachLinks.push_back(
      DynamicString(std::string_view{attachLinkWithNull.data(), attachLinkWithNull.size()}));
  expectRejected(nullAttachment);

  auto unknownAttachment = MakeParams({"soft"});
  unknownAttachment.softAttachLinks.push_back("missing");
  expectRejected(unknownAttachment);
}

TEST_F(CreateSoftSkinnedActorNameTest, AcceptsAttachLinksMatchingAutocorrectedLinkNames) {
  auto params = MakeParams({"soft"}, "");
  params.softAttachLinks.push_back("link_0");

  Actor const* actor = _scene->CreateSoftSkinnedActor(params, test::ExpectOK{});
  ASSERT_NE(nullptr, actor);
}

TEST_F(CreateSoftSkinnedActorNameTest, AssignsCollisionFreeEmptyNestedSoftLocalNames) {
  auto params = MakeParams({"named", "", "", "soft_2"}, "soft_0");

  Actor const* actor = _scene->CreateSoftSkinnedActor(params, test::ExpectOK{});
  ASSERT_NE(nullptr, actor);

  auto const& softActors = actor->GetNestedSoftActors(test::ExpectOK{});
  ASSERT_EQ(size_t{4}, softActors.size());
  for (auto const soft : softActors) {
    ASSERT_EQ(actor->GetHandle(), _scene->GetActor(soft)->GetArticulatedActor(test::ExpectOK{}));
  }
  EXPECT_STREQ("Skel/named", _scene->GetActor(softActors[0])->GetName());
  EXPECT_STREQ("Skel/soft_1", _scene->GetActor(softActors[1])->GetName());
  EXPECT_STREQ("Skel/soft_3", _scene->GetActor(softActors[2])->GetName());
  EXPECT_STREQ("Skel/soft_2", _scene->GetActor(softActors[3])->GetName());
}

TEST_F(CreateSoftSkinnedActorNameTest, UsesFallbackParentNameForUnnamedNestedSoftActors) {
  auto params = MakeParams({""}, "bone", "");

  Actor const* actor = _scene->CreateSoftSkinnedActor(params, test::ExpectOK{});
  ASSERT_NE(nullptr, actor);

  auto const& softActors = actor->GetNestedSoftActors(test::ExpectOK{});
  ASSERT_EQ(size_t{1}, softActors.size());
  for (auto const soft : softActors) {
    ASSERT_EQ(actor->GetHandle(), _scene->GetActor(soft)->GetArticulatedActor(test::ExpectOK{}));
  }
  EXPECT_STREQ("unnamed_articulation/soft_0", _scene->GetActor(softActors[0])->GetName());
}
