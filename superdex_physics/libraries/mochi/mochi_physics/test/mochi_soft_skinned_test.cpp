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
#include <mochi_core/geometry/tetrahedral_mesh.h>
#include <mochi_physics/src/mochi_articulated_body.h>
#include <mochi_physics/src/mochi_common_components.h>
#include <mochi_physics/src/mochi_context.h>
#include <mochi_physics/src/mochi_ecs.h>
#include <mochi_physics/src/mochi_shape.h>

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

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

TEST_F(CreateSoftSkinnedActorNameTest, RejectsEmptySoftParamsWithoutMutation) {
  auto const params = MakeParams({});
  for (bool const useExperimental : {false, true}) {
    SCOPED_TRACE(useExperimental);
    int const numActorsBefore = _scene->GetNumActors();
    Error error;
    Actor const* const actor = useExperimental
        ? experimental::CreateSoftSkinnedActor(_scene, params, {}, error)
        : _scene->CreateSoftSkinnedActor(params, error);
    EXPECT_EQ(nullptr, actor);
    EXPECT_STREQ(
        "SoftSkinnedActorParams::softParams must contain at least one soft actor.",
        error.GetDescription());
    EXPECT_EQ(numActorsBefore, _scene->GetNumActors());
  }
}

TEST_F(CreateSoftSkinnedActorNameTest, RejectsInvalidNestedSoftParamsWithoutMutation) {
  auto const expectRejected =
      [&](SoftSkinnedActorParams params,
          experimental::ExperimentalSoftSkinnedActorParams const& experimentalParams = {}) {
        int const numActorsBefore = _scene->GetNumActors();
        EXPECT_EQ(
            nullptr,
            experimental::CreateSoftSkinnedActor(
                _scene, params, experimentalParams, test::ExpectNotOK{}));
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

  experimental::ExperimentalSoftSkinnedActorParams softCollider;
  softCollider.softParams.resize(2);
  softCollider.softParams[1].colliderType = ColliderType::Sdf;
  expectRejected(MakeParams({"soft", "collider"}), softCollider);
}

TEST_F(CreateSoftSkinnedActorNameTest, AcceptsAttachLinksMatchingAutocorrectedLinkNames) {
  auto params = MakeParams({"soft"}, "");
  params.softAttachLinks.push_back("link_0");

  Actor const* actor = _scene->CreateSoftSkinnedActor(params, test::ExpectOK{});
  ASSERT_NE(nullptr, actor);
}

TEST_F(CreateSoftSkinnedActorNameTest, RejectsContactSkinOnNestedSoftActor) {
  auto params = MakeParams({"soft"}, "");
  params.softParams[0].shape =
      _mochiContext->CreateModelShape(test::CreateUnitCubeContactSkinModel(), test::ExpectOK{});
  params.softParams[0].useContactSkin = true;
  params.softAttachLinks.push_back("link_0");
  Error error;
  EXPECT_EQ(nullptr, _scene->CreateSoftSkinnedActor(params, error));
  EXPECT_STREQ("useContactSkin is not supported for nested soft actors.", error.GetDescription());
}

TEST_F(CreateSoftSkinnedActorNameTest, PreservesContactSkinWhenCloningAttachedSoftShape) {
  auto params = MakeParams({"soft"}, "");
  ModelData model = test::CreateUnitCubeContactSkinModel();
  model.constrainedNodes = DynamicArray<int>{0};
  MeshData const expectedContactSkin = *model.contactSkinMesh;

  params.softParams[0].shape = _mochiContext->CreateModelShape(model, test::ExpectOK{});
  params.softAttachLinks.push_back("link_0");
  Actor const* actor = _scene->CreateSoftSkinnedActor(params, test::ExpectOK{});
  ASSERT_NE(nullptr, actor);

  auto const& softActors = actor->GetNestedSoftActors(test::ExpectOK{});
  ASSERT_EQ(size_t{1}, softActors.size());
  auto& registry = GetRegistry();
  auto const entity = GetEntity(softActors[0]);
  auto const& shape = registry.get<CShape const>(entity).shape;
  auto const tetShape = std::dynamic_pointer_cast<TetrahedralMeshShape const>(shape);
  ASSERT_NE(nullptr, tetShape);
  ModelData const clonedModel = tetShape->GetModelData(test::ExpectOK{});
  ASSERT_TRUE(clonedModel.contactSkinMesh.has_value());
  EXPECT_EQ(expectedContactSkin, *clonedModel.contactSkinMesh);

  Actor const* const softActor = _scene->GetActor(softActors[0]);
  ASSERT_NE(nullptr, softActor);
  EXPECT_EQ(expectedContactSkin.GetNumNodes(), softActor->GetContactSkinMesh().GetNumNodes());
  EXPECT_TRUE(softActor->IsQuerySupported(QueryType::ContactSkinNodePositions));
}

TEST_F(CreateSoftSkinnedActorNameTest, NestedSoftActorsDoNotGetNodeVelocities) {
  Actor const* actor = _scene->CreateSoftSkinnedActor(MakeParams({"soft"}), test::ExpectOK{});
  ASSERT_NE(nullptr, actor);
  Actor const* const soft = _scene->GetActor(actor->GetNestedSoftActors(test::ExpectOK{})[0]);
  ASSERT_NE(nullptr, soft);

  EXPECT_TRUE(soft->GetNodeVelocitiesLocal(test::ExpectNotOK{}).empty());
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

// Deformable node constraints model neither nested soft actor nodes, which are posed by skinning,
// nor ROM actor DoFs, which are reduced coordinates.
TEST_F(CreateSoftSkinnedActorNameTest, DeformableNodeConstraintsRejectNestedSoftAndRomActors) {
  Actor* parent = _scene->CreateSoftSkinnedActor(MakeParams({"soft"}), test::ExpectOK{});
  ActorHandle const link = parent->GetNestedLinkActors(test::ExpectOK{})[0];
  SoftActorParams standaloneParams;
  standaloneParams.shape = test::CreateUnitCubeTetMeshShape(_mochiContext);
  ActorHandle const standalone =
      _scene->CreateSoftActor(standaloneParams, test::ExpectOK{})->GetHandle();

  DynamicArray<ActorHandle> rejected = {parent->GetNestedSoftActors(test::ExpectOK{})[0]};
  if constexpr (MOCHI_ENABLE_ROM_ACTORS) {
    // A ROM actor, and a ROM/FOM switching one in its full-order phase, which ends at step 3.
    experimental::ExperimentalSoftActorParams experimentalParams;
    experimentalParams.rom = experimental::RomParams{.source = "polynomial_crom_order_1"};
    auto const createRomActor = [&] {
      return experimental::CreateSoftActor(
                 _scene, standaloneParams, experimentalParams, test::ExpectOK{})
          ->GetHandle();
    };

    rejected.push_back(createRomActor());
    experimentalParams.rom->romFomSwitching = experimental::RomFomSwitchingTestOnlyParams{
        .romToFomSwappingStep = 1, .fomToRomSwappingStep = 3};
    rejected.push_back(createRomActor());
    _scene->Step(0.01);
    ASSERT_FALSE(GetRegistry().all_of<TagRomActor>(GetEntity(rejected.back())));
  }

  // Each constraint is valid on the standalone soft actor and fails on the rejected ones.
  auto const expectRejects = [&](auto const& create) {
    int const numConstraints = _scene->GetNumConstraints();
    for (ActorHandle const actor : rejected) {
      EXPECT_EQ(nullptr, create(actor, test::ExpectNotOK{}));
    }
    EXPECT_EQ(numConstraints, _scene->GetNumConstraints());
    EXPECT_NE(nullptr, create(standalone, test::ExpectOK{}));
  };
  expectRejects([&](ActorHandle actor, Error& error) {
    return _scene->CreateDeformableNodePositionConstraint({.nodeIndex = 1, .actor = actor}, error);
  });
  expectRejects([&](ActorHandle actor, Error& error) {
    return _scene->CreateDeformableNodeToDeformableNodeConstraint(
        {.nodeIndexA = 1, .nodeIndexB = 2, .actorA = actor, .actorB = standalone}, error);
  });
  expectRejects([&](ActorHandle actor, Error& error) {
    return _scene->CreateDeformableNodeToDeformableNodeConstraint(
        {.nodeIndexA = 1, .nodeIndexB = 2, .actorA = standalone, .actorB = actor}, error);
  });
  expectRejects([&](ActorHandle actor, Error& error) {
    return _scene->CreateDeformableNodeToRigidConstraint(
        {.deformableNodeIndex = 1, .rigidActor = link, .deformableActor = actor}, error);
  });
}

// An articulated actor whose root link frame differs from its root frame. Skins are defined in the
// root frame.
class SkinReferenceFrameTest : public test::MochiSceneTestBase {
 protected:
  static inline TransformRT const kWorldFromRoot{
      Quaternion::FromRotationVector(Real3{0.2_r, -0.4_r, 0.3_r}),
      Real3{1_r, 2_r, -3_r}};

  // Unit cube skinned to the root link, with a constrained node for nested soft actors.
  ShapeHandle CreateSkinShape() {
    auto&& [coords, connectivity] = test::CreateMinimalTetMeshUnitCube();
    auto mesh = std::make_shared<TetrahedralMesh const>(coords, connectivity);
    SkinningData skinning;
    skinning.weightsPerNode = 1;
    skinning.indices.resize(mesh->GetNumNodes(), 0);
    skinning.weights.resize(mesh->GetNumNodes(), 1_r);
    auto shape = std::make_shared<TetrahedralMeshShape>(
        mesh,
        std::make_shared<SkinningData const>(std::move(skinning)),
        std::make_shared<ConstrainedNodesData const>(DynamicArray<int>{0}));
    return assert_cast<ContextImpl*>(_mochiContext)->RegisterShape(shape, test::ExpectOK{});
  }

  ArticulatedActorParams MakeSkeletonParams() {
    ArticulatedActorParams params;
    params.worldFromRoot = kWorldFromRoot;
    params.joints = {
        {.name = "root",
         .type = ArticulatedJointType::Free,
         .parentLinkFromJoint = TransformRT{
             Quaternion::FromRotationVector(Real3{0_r, 0_r, 0.5_r}), Real3{0.3_r, -0.2_r, 0.1_r}}}};
    params.links = {
        {.name = "root",
         .parentLink = -1,
         .shape = test::CreateUnitCubeTetMeshShape(_mochiContext),
         .colliderType = ColliderType::None}};
    return params;
  }

  // Expects the skin of entity e to be at its rest coordinates in the articulated root frame.
  void ExpectSkinAtRestInRootFrame(entt::entity e) {
    auto const& reg = GetRegistry();
    auto const rest = Unflatten<Real3 const>(
        MakeConstSpan(reg.get<CArticulatedSkinningData const>(e).restCoords));
    auto const displacements = Unflatten<Real3 const>(MakeConstSpan(
        reg.get<CDisplacementSlice<real, TimeStep::Current, DisplacementLayer::Skinned> const>(e)
            .value));
    TransformRT const& worldFromLocal = reg.get<CRootTransform const>(e).worldFromLocal;
    for (int node = 0; node < isize(rest); ++node) {
      EXPECT_NEAR_TOL(
          worldFromLocal.TransformPoint(rest[node] + displacements[node]),
          kWorldFromRoot.TransformPoint(rest[node]),
          1e-5_r);
    }
  }
};

TEST_F(SkinReferenceFrameTest, ArticulatedSkinIsDefinedInRootFrame) {
  ArticulatedActorParams params = MakeSkeletonParams();
  params.skin.emplace();
  params.skin->shape = CreateSkinShape();
  Actor* actor = _scene->CreateArticulatedActor(params, test::ExpectOK{});
  ExpectSkinAtRestInRootFrame(GetEntity(actor));
}

TEST_F(SkinReferenceFrameTest, NestedSoftSkinIsDefinedInRootFrame) {
  SoftSkinnedActorParams params;
  params.skeletonParams = MakeSkeletonParams();
  SoftActorParams soft;
  soft.shape = CreateSkinShape();
  soft.hasGravity = false;
  params.softParams.push_back(soft);
  Actor* actor = _scene->CreateSoftSkinnedActor(params, test::ExpectOK{});
  ExpectSkinAtRestInRootFrame(GetEntity(actor->GetNestedSoftActors(test::ExpectOK{})[0]));
}
