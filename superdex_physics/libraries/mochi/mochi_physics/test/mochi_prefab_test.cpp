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

#include <mochi_core/test/log_suppression.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/file_utils.h>
#include <mochi_physics/src/mochi_constraint.h>
#include <mochi_physics/src/mochi_contact_filter.h>
#include <mochi_physics/src/mochi_ecs_utils.h>
#include <mochi_physics/src/mochi_hdf5.h>
#include <mochi_physics/src/mochi_scene.h>
#include <mochi_physics/utils/mochi_prefab.h>

#include <limits>
#include <memory>
#include <string>

using namespace mochi;
using namespace mochi::test;

// The prefab assets used by these tests are not shipped externally.
#if MOCHI_USE_HDF5 && MOCHI_INTERNAL
#define MOCHI_HDF5_AND_INTERNAL 1
#else
#define MOCHI_HDF5_AND_INTERNAL 0
#endif

// Helper to find actor by name
static ActorHandle FindActorByName(Scene const* scene, std::string_view name) {
  ActorHandle handle;
  scene->ForEachActor([&](Actor const* a) {
    if (name == a->GetName()) {
      EXPECT_FALSE(handle.IsValid()) << "Duplicate actor name";
      handle = a->GetHandle();
    }
  });
  return handle;
}

// Inertia of two equal point masses at +r and -r with total mass totalMass:
// I = sum m * (|r|^2 Identity - r * r^T).
static Real6 PointPairInertia(real totalMass, Real3 const& r) {
  return {
      totalMass * (r[1] * r[1] + r[2] * r[2]),
      -totalMass * r[0] * r[1],
      -totalMass * r[0] * r[2],
      totalMass * (r[0] * r[0] + r[2] * r[2]),
      -totalMass * r[1] * r[2],
      totalMass * (r[0] * r[0] + r[1] * r[1])};
}

TEST(Prefab, ArticulatedActor_Serialization) {
  // One articulated actor with non-default settings. Does not attempt to cover every field of the
  // nested rigid actors nor contact params, since those have sufficient coverage in other tests.
  std::string json = R"({
    "actors": {
      "articulated": [
        {
          "_comment": "Good stuff",
          "name": "myArticulation",
          "scale": 3.14,
          "rotation": [
            1,
            2,
            3,
            4
          ],
          "translation": [
            0.5,
            0.6,
            0.7
          ],
          "joints": [
            {
              "name": "joint0",
              "type": "Revolute",
              "axis": [0, 0, 1],
              "parentLinkFromJoint": {
                "rotation": [0, 0, 0, 1],
                "translation": [0.1, 0.2, 0.3]
              },
              "friction": {
                "viscous": 0.1,
                "coulomb": 0.3,
                "falloffVel": 0.001,
                "stictionExtra": 0.0,
                "stribeckVel": 0.0
              },
              "inertia": 0.2,
              "limitStiffness": 0.4,
              "limitDamping": 0.5
            }
          ],
          "links": [
            {
              "name": "parentLink",
              "parentLink": -1,
              "shape": "some/dir/my_parent_link",
              "contact": {
                "penaltyCoefficient": 2e9
              },
              "density": 1.23
            },
            {
              "name": "childLink",
              "parentLink": 0,
              "parentJointFromLink": {
                "rotation": [0, 0, 0, 1],
                "translation": [0.4, 0.5, 0.6]
              },
              "shape": "some/dir/my_child_link",
              "contact": {
                "penaltyCoefficient": 3e9
              },
              "density": 2.34
            }
          ],
          "skin": {
            "shape": "some/dir/skin_shape",
            "layer": "mySkinLayer",
            "contact": {
              "penaltyCoefficient": 4e9
            },
            "boundaryElementType": "P1Q6"
          }
        }
      ]
    }
  })";

  // Expect deserialized values
  auto expectArticulatedActor = [](prefab::ArticulatedActorPrefab const& art) {
    EXPECT_STREQ("Good stuff", art.comment->c_str());
    EXPECT_STREQ("myArticulation", art.name.c_str());
    EXPECT_NEAR_EQ(3.14_r, art.scale);
    EXPECT_NEAR_EQ(Quaternion(1_r, 2_r, 3_r, 4_r), art.rotation);
    EXPECT_NEAR_EQ(Real3(0.5_r, 0.6_r, 0.7_r), art.translation);
    // Joints
    ASSERT_EQ(1, art.joints.size());
    EXPECT_STREQ("joint0", art.joints[0].name.c_str());
    EXPECT_EQ(ArticulatedJointType::Revolute, art.joints[0].type);
    EXPECT_NEAR_EQ(Real3(0_r, 0_r, 1_r), art.joints[0].axis);
    EXPECT_NEAR_EQ(Real3(0.1_r, 0.2_r, 0.3_r), art.joints[0].parentLinkFromJoint.GetTranslation());
    EXPECT_NEAR_EQ(0.1_r, art.joints[0].friction.viscous);
    EXPECT_NEAR_EQ(0.3_r, art.joints[0].friction.coulomb);
    ASSERT_TRUE(art.joints[0].inertia.has_value());
    EXPECT_NEAR_EQ(0.2_r, *art.joints[0].inertia);
    EXPECT_NEAR_EQ(0.4_r, art.joints[0].limitStiffness);
    EXPECT_NEAR_EQ(0.5_r, art.joints[0].limitDamping);

    // Links
    ASSERT_EQ(2, art.links.size());
    EXPECT_STREQ("parentLink", art.links[0].name.c_str());
    EXPECT_EQ(-1, art.links[0].parentLink);
    EXPECT_STREQ("some/dir/my_parent_link", art.links[0].shapeFile.c_str());
    ASSERT_TRUE(art.links[0].density.has_value());
    EXPECT_NEAR_EQ(1.23_r, *art.links[0].density);
    EXPECT_NEAR_EQ(2e9_r, art.links[0].contact.penaltyCoefficient);
    EXPECT_STREQ("childLink", art.links[1].name.c_str());
    EXPECT_EQ(0, art.links[1].parentLink);
    EXPECT_NEAR_EQ(Real3(0.4_r, 0.5_r, 0.6_r), art.links[1].parentJointFromLink.GetTranslation());
    EXPECT_STREQ("some/dir/my_child_link", art.links[1].shapeFile.c_str());
    ASSERT_TRUE(art.links[1].density.has_value());
    EXPECT_NEAR_EQ(2.34_r, *art.links[1].density);
    EXPECT_NEAR_EQ(3e9_r, art.links[1].contact.penaltyCoefficient);

    // Skin
    ASSERT_TRUE(art.skin.has_value());
    EXPECT_STREQ("some/dir/skin_shape", art.skin->shapeFile.c_str());
    EXPECT_STREQ("mySkinLayer", art.skin->layer.c_str());
    EXPECT_NEAR_EQ(4e9_r, art.skin->contact.penaltyCoefficient);
    EXPECT_EQ(ActorBoundaryElementType::P1Q6, art.skin->boundaryElementType);
  };

  // Load one articulated actor from JSON.
  prefab::ScenePrefab content = prefab::ShallowLoadFromJsonString(json, test::ExpectOK{});
  EXPECT_EQ(1, content.actors.articulated.size());

  // If this fails, then a code change may have broken existing prefabs
  expectArticulatedActor(content.actors.articulated[0]);

  // Test round-trip serialization
  auto json2 = prefab::SaveToJsonString(content, test::ExpectOK{});
  prefab::ScenePrefab content2 = prefab::ShallowLoadFromJsonString(json2, test::ExpectOK{});
  EXPECT_EQ(1, content2.actors.articulated.size());
  expectArticulatedActor(content2.actors.articulated[0]);
}

TEST_IF(MOCHI_HDF5_AND_INTERNAL, Prefab, ArticulatedActor_AddToScene) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("my scene");
  MOCHI_DEFER(context->DestroyScene(scene));

  // Use a prefab to add an articulated actor to a scene.
  // No need to re-test serialization for this part.
  prefab::ScenePrefab scenePrefab;
  auto& art = scenePrefab.actors.articulated.push_back();
  art.name = "myArticulation";
  art.scale = 1_r;
  art.rotation = Quaternion(1_r, 2_r, 3_r, 4_r);
  art.translation = Real3(0.5_r, 0.6_r, 0.7_r);
  // Define inline topology: 2 links with a free root joint and a revolute joint
  art.joints.resize(2);
  art.joints[0].type = ArticulatedJointType::Free;
  art.joints[1].type = ArticulatedJointType::Revolute;
  art.joints[1].axis = Real3{0_r, 0_r, 1_r};
  art.joints[1].parentLinkFromJoint = TransformRT(Real3{0.5_r, 0_r, 0_r});

  art.links.resize(2);
  art.links[0].name = "boneA";
  art.links[0].parentLink = -1;
  art.links[0].shapeFile = "articulated/two_links_revolute/bone_a.mochi.h5";
  art.links[0].colliderType = ColliderType::Box;
  art.links[0].density = 1.23_r;
  art.links[1].name = "boneB";
  art.links[1].parentLink = 0;
  art.links[1].shapeFile = "articulated/two_links_revolute/bone_b.mochi.h5";
  art.links[1].colliderType = ColliderType::Box;
  art.links[1].density = 2.34_r;

  // Compute expected rest transforms from inline topology. Joint and link
  // translations are scaled by `effectiveScale = art.scale * prefabParams.scale`
  // to match the scale baked into the actor by AddToScene.
  prefab::PrefabParams prefabParams;
  prefabParams.rotation = Quaternion::FromAxisAngle(Real3{0_r, 1_r, 0_r}, 0.5_r);
  prefabParams.translation = Real3{2.22_r, 3.33_r, 4.44_r};
  real const effectiveScale = art.scale * prefabParams.scale;
  auto scaleTranslation = [effectiveScale](TransformRT const& t) {
    return TransformRT(t.GetRotation(), t.GetTranslation() * effectiveScale);
  };
  TransformRT const parentRootFromLink = scaleTranslation(art.joints[0].parentLinkFromJoint) *
      scaleTranslation(art.links[0].parentJointFromLink);
  TransformRT const childRootFromLink = parentRootFromLink *
      scaleTranslation(art.joints[1].parentLinkFromJoint) *
      scaleTranslation(art.links[1].parentJointFromLink);

  // Load the referenced link shapes
  prefab::LoadShapes(scenePrefab, test::GetAssetsDir(), context, test::ExpectOK{});
  EXPECT_TRUE(art.links[0].shape.IsValid());
  EXPECT_TRUE(art.links[1].shape.IsValid());

  // Add the prefab to the scene
  prefab::AddToScene(scenePrefab, scene, prefabParams, test::ExpectOK{});
  EXPECT_EQ(3, scene->GetNumActors()); // 1 articulation and 2 rigid links

  TransformSRT worldFromPrefab(prefabParams.scale, prefabParams.rotation, prefabParams.translation);

  // Get the actor handles and check the actor properties.
  std::string expectedNamePrefix;
  ActorHandle articulatedActorHandle;
  DynamicArray<ActorHandle> linkActorHandles;
  auto checkActor = [&](Actor* actor) {
    std::string actorName = actor->GetName();
    TransformSRT prefabFromActor = TransformSRT{1_r, Normalize(art.rotation), art.translation};
    TransformRT worldFromActor = (worldFromPrefab * prefabFromActor).GetTransformRT();
    if (actorName == expectedNamePrefix + "myArticulation") {
      EXPECT_EQ(ActorType::Articulated, actor->GetType());
      EXPECT_NEAR_EQ(worldFromActor.GetRotation(), actor->GetRootTransform().GetRotation());
      EXPECT_NEAR_EQ(worldFromActor.GetTranslation(), actor->GetRootTransform().GetTranslation());
      articulatedActorHandle = actor->GetHandle();
    } else if (actorName == expectedNamePrefix + "myArticulation/boneA") {
      EXPECT_EQ(ActorType::Rigid, actor->GetType());
      worldFromActor = worldFromActor * parentRootFromLink; // NOTE: use worldRT because scale is
                                                            // already baked into parentRootFromLink
      EXPECT_NEAR_EQ(worldFromActor.GetRotation(), actor->GetRootTransform().GetRotation());
      EXPECT_NEAR_EQ(worldFromActor.GetTranslation(), actor->GetRootTransform().GetTranslation());
      EXPECT_NEAR_EQ(1.23_r, actor->GetDensity(test::ExpectOK{}));
      linkActorHandles.push_back(actor->GetHandle());
    } else if (actorName == expectedNamePrefix + "myArticulation/boneB") {
      EXPECT_EQ(ActorType::Rigid, actor->GetType());
      worldFromActor = worldFromActor * childRootFromLink; // NOTE: use worldRT because scale is
                                                           // already baked into childRootFromLink
      EXPECT_NEAR_EQ(worldFromActor.GetRotation(), actor->GetRootTransform().GetRotation());
      EXPECT_NEAR_EQ(worldFromActor.GetTranslation(), actor->GetRootTransform().GetTranslation());
      EXPECT_NEAR_EQ(2.34_r, actor->GetDensity(test::ExpectOK{}));
      linkActorHandles.push_back(actor->GetHandle());
    } else {
      FAIL() << "Unexpected actor name: " << actorName;
    }
  };
  EXPECT_EQ(3, scene->GetNumActors());
  scene->ForEachActor(checkActor);

  // Destroy the actors
  scene->DestroyActor(articulatedActorHandle);
  EXPECT_EQ(0, scene->GetNumActors());
  articulatedActorHandle = {};
  linkActorHandles.clear();

  // Repeat but give the prefab instance a name, and use AddToSceneResult to get the actors.
  prefabParams.name = "myPrefab";
  expectedNamePrefix = "myPrefab/";
  auto const result = prefab::AddToScene(scenePrefab, scene, prefabParams, test::ExpectOK{});
  EXPECT_EQ(3, scene->GetNumActors());
  scene->ForEachActor(checkActor);

  // The articulated actor (but not the link actors) should have been returned.
  ASSERT_EQ(1, isize(result.actors));
  EXPECT_EQ(ActorType::Articulated, result.actors[0]->GetType());
  EXPECT_EQ(articulatedActorHandle, result.actors[0]->GetHandle());
  EXPECT_EQ(result.actors, result.Filter(ActorType::Articulated));
}

// Verifies that actor.scale on an articulated actor scales both joint (parentLinkFromJoint)
// and link (parentJointFromLink) translations, and that it composes correctly with per-link
// shape transforms (shapeScale / shapeRotation / shapeTranslation).
TEST_IF(MOCHI_HDF5_AND_INTERNAL, Prefab, ArticulatedActor_UniformScale) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("test scene");
  MOCHI_DEFER(context->DestroyScene(scene));

  prefab::ScenePrefab scenePrefab;
  auto& art = scenePrefab.actors.articulated.push_back();
  art.name = "scaledArt";
  art.scale = 2_r;
  art.rotation = Quaternion::Identity();
  art.translation = Real3(1_r, 0_r, 0_r);

  // 2 links: free root + revolute child. Both joint[1].parentLinkFromJoint and
  // link[0].parentJointFromLink have non-zero translations so both sides of the chain scale.
  art.joints.resize(2);
  art.joints[0].type = ArticulatedJointType::Free;
  art.joints[1].type = ArticulatedJointType::Revolute;
  art.joints[1].axis = Real3{0_r, 0_r, 1_r};
  art.joints[1].parentLinkFromJoint = TransformRT(Real3{0.5_r, 0.3_r, 0_r});

  art.links.resize(2);
  art.links[0].name = "linkA";
  art.links[0].parentLink = -1;
  art.links[0].shapeFile = "articulated/two_links_revolute/bone_a.mochi.h5";
  art.links[0].colliderType = ColliderType::Box;
  art.links[0].parentJointFromLink = TransformRT(Real3{0_r, 0.1_r, 0_r});

  // linkB: also exercise shape-level non-uniform scale, rotation, and translation to confirm
  // they compose correctly with the actor scale.
  art.links[1].name = "linkB";
  art.links[1].parentLink = 0;
  art.links[1].shapeFile = "articulated/two_links_revolute/bone_b.mochi.h5";
  art.links[1].colliderType = ColliderType::Box;
  art.links[1].shapeScale = Real3{1_r, 2_r, 3_r};
  art.links[1].shapeRotation = Quaternion::FromAxisAngle(Real3{0_r, 1_r, 0_r}, 0.5_r);
  art.links[1].shapeTranslation = Real3{0.1_r, 0.2_r, 0_r};
  art.links[1].parentJointFromLink = TransformRT(Real3{0.2_r, 0_r, 0_r});

  // Expected rest transforms: every joint/link translation is multiplied by the effective scale
  // (actor.scale * scaleModifier, with scaleModifier = 1 at the top level).
  real const effectiveScale = art.scale;
  auto scaleTranslation = [effectiveScale](TransformRT const& t) {
    return TransformRT(t.GetRotation(), t.GetTranslation() * effectiveScale);
  };
  TransformRT const scaledRootFromLinkA = scaleTranslation(art.joints[0].parentLinkFromJoint) *
      scaleTranslation(art.links[0].parentJointFromLink);
  TransformRT const scaledRootFromLinkB = scaledRootFromLinkA *
      scaleTranslation(art.joints[1].parentLinkFromJoint) *
      scaleTranslation(art.links[1].parentJointFromLink);

  // Sanity: with scale != 1 and non-zero translations, the scaled chain must differ from the
  // unscaled chain.
  TransformRT const unscaledRootFromLinkB =
      (art.joints[0].parentLinkFromJoint * art.links[0].parentJointFromLink) *
      art.joints[1].parentLinkFromJoint * art.links[1].parentJointFromLink;
  EXPECT_FALSE(
      NearEqual(scaledRootFromLinkB.GetTranslation(), unscaledRootFromLinkB.GetTranslation()));

  // Load shapes
  prefab::LoadShapes(scenePrefab, test::GetAssetsDir(), context, test::ExpectOK{});
  EXPECT_TRUE(art.links[0].shape.IsValid());
  EXPECT_TRUE(art.links[1].shape.IsValid());

  // Add to scene
  prefab::PrefabParams prefabParams;
  prefab::AddToScene(scenePrefab, scene, prefabParams, test::ExpectOK{});
  EXPECT_EQ(3, scene->GetNumActors());

  TransformSRT worldFromPrefab(prefabParams.scale, prefabParams.rotation, prefabParams.translation);
  TransformSRT prefabFromActor(1_r, Normalize(art.rotation), art.translation);
  TransformRT worldFromActor = (worldFromPrefab * prefabFromActor).GetTransformRT();

  scene->ForEachActor([&](Actor const* actor) {
    std::string const actorName = actor->GetName();
    if (actorName == "scaledArt") {
      EXPECT_EQ(ActorType::Articulated, actor->GetType());
      EXPECT_NEAR_EQ(worldFromActor.GetTranslation(), actor->GetRootTransform().GetTranslation());
    } else if (actorName == "scaledArt/linkA") {
      TransformRT expected = worldFromActor * scaledRootFromLinkA;
      EXPECT_NEAR_EQ(expected.GetTranslation(), actor->GetRootTransform().GetTranslation());
    } else if (actorName == "scaledArt/linkB") {
      TransformRT expected = worldFromActor * scaledRootFromLinkB;
      EXPECT_NEAR_EQ(expected.GetTranslation(), actor->GetRootTransform().GetTranslation());
    } else {
      FAIL() << "Unexpected actor name: " << actorName;
    }
  });
}

// Verifies that uniform scale works end-to-end on a real soft-skinned asset (Allegro hand).
// Loads the allegro_soft prefab at 1x and 2x, checks structural sanity (1 articulated actor,
// 21 rigid links, 4 soft fingertips of the correct type), and checks that every link's offset
// from the articulated root scales by exactly 2x. Catches regressions that pure synthetic
// tests might miss.
TEST_IF(MOCHI_HDF5_AND_INTERNAL, Prefab, ArticulatedActor_UniformScale_SkinnedMesh) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));

  struct ActorPositions {
    Real3 rootPosition;
    DynamicArray<Real3> linkPositions;
  };

  auto loadWithScale = [&](real scale) {
    auto* scene = context->CreateScene("test");
    auto scenePrefab = prefab::ShallowLoadFromFile(
        test::GetAssetPath("allegro_soft/allegro_soft.mochi_prefab"), test::ExpectOK{});
    scenePrefab.actors.softSkinned[0].skeletonParams.scale = scale;
    prefab::LoadNestedPrefabs(scenePrefab, test::GetAssetPath(""), test::ExpectOK{});
    prefab::LoadShapes(scenePrefab, test::GetAssetPath(""), context, test::ExpectOK{});
    auto const result = prefab::AddToScene(scenePrefab, scene, {}, test::ExpectOK{});

    // Structural checks: exactly one articulated actor with 21 rigid links and 4 soft actors.
    EXPECT_EQ(1, isize(result.actors));
    Actor* articulatedActor = result.actors[0];
    EXPECT_EQ(ActorType::Articulated, articulatedActor->GetType());
    auto const& linkActors = articulatedActor->GetNestedLinkActors(test::ExpectOK{});
    auto const& softActors = articulatedActor->GetNestedSoftActors(test::ExpectOK{});
    EXPECT_EQ(21, isize(linkActors));
    EXPECT_EQ(4, isize(softActors));
    for (auto const& softHandle : softActors) {
      EXPECT_EQ(ActorType::Soft, scene->GetActor(softHandle)->GetType());
    }

    ActorPositions out;
    out.rootPosition = articulatedActor->GetRootTransform().GetTranslation();
    for (auto const& linkHandle : linkActors) {
      Actor* link = scene->GetActor(linkHandle);
      EXPECT_EQ(ActorType::Rigid, link->GetType());
      out.linkPositions.push_back(link->GetRootTransform().GetTranslation());
    }
    context->DestroyScene(scene);
    return out;
  };

  real const testScale = 2_r;
  auto const positions1x = loadWithScale(1_r);
  auto const positions2x = loadWithScale(testScale);

  // Every link's offset from the articulated root must scale by testScale.
  ASSERT_EQ(isize(positions1x.linkPositions), isize(positions2x.linkPositions));
  bool foundScaledLink = false;
  for (int i = 0; i < isize(positions1x.linkPositions); ++i) {
    Real3 const offset1x = positions1x.linkPositions[i] - positions1x.rootPosition;
    Real3 const offset2x = positions2x.linkPositions[i] - positions2x.rootPosition;
    if (Norm(offset1x) > 1e-6_r) {
      EXPECT_NEAR_TOL(offset2x, offset1x * testScale, 1e-4_r) << "link index " << i;
      foundScaledLink = true;
    }
  }
  EXPECT_TRUE(foundScaledLink) << "Expected at least one link with non-zero offset from root";
}

// Verifies that prismatic joint limits (minLimit, maxLimit), expressed in meters along the
// prismatic axis, are multiplied by the effective scale when the articulated actor is created.
TEST_IF(MOCHI_HDF5_AND_INTERNAL, Prefab, ArticulatedActor_UniformScale_PrismaticLimits) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("test scene");
  MOCHI_DEFER(context->DestroyScene(scene));

  prefab::ScenePrefab scenePrefab;
  auto& art = scenePrefab.actors.articulated.push_back();
  art.name = "prismArt";
  real const testScale = 2_r;
  art.scale = testScale;

  // 2 joints: free root + prismatic child along X with min/max limits in meters.
  Real3 const minLimit{-0.2_r, 0_r, 0_r};
  Real3 const maxLimit{0.5_r, 0_r, 0_r};
  art.joints.resize(2);
  art.joints[0].type = ArticulatedJointType::Free;
  art.joints[1].type = ArticulatedJointType::Prismatic;
  art.joints[1].axis = Real3{1_r, 0_r, 0_r};
  art.joints[1].parentLinkFromJoint = TransformRT(Real3{0.1_r, 0_r, 0_r});
  art.joints[1].minLimit = minLimit;
  art.joints[1].maxLimit = maxLimit;

  art.links.resize(2);
  art.links[0].name = "linkA";
  art.links[0].parentLink = -1;
  art.links[0].shapeFile = "articulated/two_links_revolute/bone_a.mochi.h5";
  art.links[0].colliderType = ColliderType::Box;
  art.links[1].name = "linkB";
  art.links[1].parentLink = 0;
  art.links[1].shapeFile = "articulated/two_links_revolute/bone_b.mochi.h5";
  art.links[1].colliderType = ColliderType::Box;

  prefab::LoadShapes(scenePrefab, test::GetAssetsDir(), context, test::ExpectOK{});
  prefab::AddToScene(scenePrefab, scene, {}, test::ExpectOK{});

  // Retrieve the articulated actor and read its joint limits from the shape info.
  ActorHandle const artHandle = FindActorByName(scene, "prismArt");
  ASSERT_TRUE(artHandle.IsValid());
  Actor* artActor = scene->GetActor(artHandle);
  ASSERT_EQ(ActorType::Articulated, artActor->GetType());

  auto const info = artActor->GetArticulatedShapeInfo(test::ExpectOK{});
  ASSERT_EQ(2, isize(info.jointTypes));
  ASSERT_EQ(ArticulatedJointType::Prismatic, info.jointTypes[1]);
  EXPECT_NEAR_EQ(minLimit * testScale, info.jointMinLimits[1]);
  EXPECT_NEAR_EQ(maxLimit * testScale, info.jointMaxLimits[1]);
}

// Verifies that a per-link centerOfMass override (in link-local coordinates) is multiplied by
// the effective scale when the rigid link actor is created.
TEST_IF(MOCHI_HDF5_AND_INTERNAL, Prefab, ArticulatedActor_UniformScale_CenterOfMass) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("test scene");
  MOCHI_DEFER(context->DestroyScene(scene));

  prefab::ScenePrefab scenePrefab;
  auto& art = scenePrefab.actors.articulated.push_back();
  art.name = "comArt";
  real const testScale = 2_r;
  art.scale = testScale;

  art.joints.resize(2);
  art.joints[0].type = ArticulatedJointType::Free;
  art.joints[1].type = ArticulatedJointType::Revolute;
  art.joints[1].axis = Real3{0_r, 0_r, 1_r};

  // Set a non-zero centerOfMass (in link-local coordinates) on the child link.
  Real3 const comLocal{0.3_r, -0.1_r, 0.05_r};
  art.links.resize(2);
  art.links[0].name = "linkA";
  art.links[0].parentLink = -1;
  art.links[0].shapeFile = "articulated/two_links_revolute/bone_a.mochi.h5";
  art.links[0].colliderType = ColliderType::Box;
  art.links[1].name = "linkB";
  art.links[1].parentLink = 0;
  art.links[1].shapeFile = "articulated/two_links_revolute/bone_b.mochi.h5";
  art.links[1].colliderType = ColliderType::Box;
  art.links[1].centerOfMass = comLocal;
  // Set momentOfInertia too — Mochi requires both or neither to be set.
  art.links[1].momentOfInertia = Real6{1_r, 1_r, 1_r, 0_r, 0_r, 0_r};

  prefab::LoadShapes(scenePrefab, test::GetAssetsDir(), context, test::ExpectOK{});
  {
    auto suppressWarning = test::SuppressLogWarning();
    prefab::AddToScene(scenePrefab, scene, {}, test::ExpectOK{});
  }

  ActorHandle const artHandle = FindActorByName(scene, "comArt");
  ASSERT_TRUE(artHandle.IsValid());
  Actor* artActor = scene->GetActor(artHandle);

  auto const& linkActors = artActor->GetNestedLinkActors(test::ExpectOK{});
  ASSERT_EQ(2, isize(linkActors));
  Actor* linkB = scene->GetActor(linkActors[1]);
  ASSERT_NE(nullptr, linkB);
  EXPECT_EQ(ActorType::Rigid, linkB->GetType());
  EXPECT_NEAR_EQ(comLocal * testScale, linkB->GetRigidCenterOfMassLocal(test::ExpectOK{}));
}

// Verifies that a cycle joint's jointFromChildLink translation is multiplied by the effective
// scale. Uses a closed kinematic loop to confirm the scaling applies to cycle entries.
TEST_IF(MOCHI_HDF5_AND_INTERNAL, Prefab, ArticulatedActor_UniformScale_CycleJoint) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("test scene");
  MOCHI_DEFER(context->DestroyScene(scene));

  prefab::ScenePrefab scenePrefab;
  auto& art = scenePrefab.actors.articulated.push_back();
  art.name = "cycleArt";
  real const testScale = 2_r;
  art.scale = testScale;

  // 3 links with a closing cycle from linkC back to linkA.
  art.joints.resize(3);
  art.joints[0].type = ArticulatedJointType::Free;
  art.joints[1].type = ArticulatedJointType::Revolute;
  art.joints[1].axis = Real3{0_r, 0_r, 1_r};
  art.joints[1].parentLinkFromJoint = TransformRT(Real3{0.4_r, 0_r, 0_r});
  art.joints[2].type = ArticulatedJointType::Revolute;
  art.joints[2].axis = Real3{0_r, 0_r, 1_r};
  art.joints[2].parentLinkFromJoint = TransformRT(Real3{0.4_r, 0_r, 0_r});

  art.links.resize(3);
  art.links[0].name = "linkA";
  art.links[0].parentLink = -1;
  art.links[0].shapeFile = "articulated/three_links_spherical/bone_a.mochi.h5";
  art.links[0].colliderType = ColliderType::Box;
  art.links[1].name = "linkB";
  art.links[1].parentLink = 0;
  art.links[1].shapeFile = "articulated/three_links_spherical/bone_b.mochi.h5";
  art.links[1].colliderType = ColliderType::Box;
  art.links[2].name = "linkC";
  art.links[2].parentLink = 1;
  art.links[2].shapeFile = "articulated/three_links_spherical/bone_c.mochi.h5";
  art.links[2].colliderType = ColliderType::Box;

  Real3 const cycleTranslation{0.7_r, 0.2_r, 0_r};
  art.cycles.resize(1);
  art.cycles[0].parentLink = 0;
  art.cycles[0].childLink = 2;
  art.cycles[0].jointFromChildLink = TransformRT(cycleTranslation);

  prefab::LoadShapes(scenePrefab, test::GetAssetsDir(), context, test::ExpectOK{});
  prefab::AddToScene(scenePrefab, scene, {}, test::ExpectOK{});

  ActorHandle const artHandle = FindActorByName(scene, "cycleArt");
  ASSERT_TRUE(artHandle.IsValid());
  Actor* artActor = scene->GetActor(artHandle);

  // Cycle entries appear at indices [numLinks, numLinks + numCycles) of the per-joint arrays.
  auto const info = artActor->GetArticulatedShapeInfo(test::ExpectOK{});
  int const numLinks = isize(art.links);
  ASSERT_EQ(1, isize(info.cycles));
  ASSERT_EQ(ArticulatedJointType::Cycle, info.jointTypes[numLinks]);
  EXPECT_NEAR_EQ(cycleTranslation * testScale, info.jointFromChildLink[numLinks].GetTranslation());
}

// Verifies that effectiveScale = actor.scale * nested.scale applies uniformly across the full
// kinematic chain when every scaled quantity is exercised at once. A single articulated actor
// populates joint/link translations, prismatic limits, per-link centerOfMass, and cycle
// jointFromChildLink, then is nested under an outer prefab that applies an additional scale.
TEST_IF(MOCHI_HDF5_AND_INTERNAL, Prefab, ArticulatedActor_UniformScale_Composite) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("test scene");
  MOCHI_DEFER(context->DestroyScene(scene));

  // Inner prefab contains an articulated actor with every scaled quantity populated.
  auto innerPrefab = std::make_shared<prefab::ScenePrefab>();
  auto& art = innerPrefab->actors.articulated.push_back();
  art.name = "robot";
  real const actorScale = 2_r;
  art.scale = actorScale;

  // 3 joints: free root, prismatic with limits, revolute.
  Real3 const prismMin{-0.15_r, 0_r, 0_r};
  Real3 const prismMax{0.3_r, 0_r, 0_r};
  Real3 const joint1Trans{0.4_r, 0_r, 0_r};
  Real3 const joint2Trans{0.5_r, 0.1_r, 0_r};
  art.joints.resize(3);
  art.joints[0].type = ArticulatedJointType::Free;
  art.joints[1].type = ArticulatedJointType::Prismatic;
  art.joints[1].axis = Real3{1_r, 0_r, 0_r};
  art.joints[1].parentLinkFromJoint = TransformRT(joint1Trans);
  art.joints[1].minLimit = prismMin;
  art.joints[1].maxLimit = prismMax;
  art.joints[2].type = ArticulatedJointType::Revolute;
  art.joints[2].axis = Real3{0_r, 0_r, 1_r};
  art.joints[2].parentLinkFromJoint = TransformRT(joint2Trans);

  // 3 links with a non-zero centerOfMass override on the tip.
  Real3 const linkBTrans{0.1_r, 0_r, 0_r};
  Real3 const linkCTrans{0.2_r, -0.05_r, 0_r};
  Real3 const tipCom{0.25_r, 0.1_r, 0_r};
  art.links.resize(3);
  art.links[0].name = "linkA";
  art.links[0].parentLink = -1;
  art.links[0].shapeFile = "articulated/three_links_spherical/bone_a.mochi.h5";
  art.links[0].colliderType = ColliderType::Box;
  art.links[1].name = "linkB";
  art.links[1].parentLink = 0;
  art.links[1].shapeFile = "articulated/three_links_spherical/bone_b.mochi.h5";
  art.links[1].colliderType = ColliderType::Box;
  art.links[1].parentJointFromLink = TransformRT(linkBTrans);
  art.links[2].name = "linkC";
  art.links[2].parentLink = 1;
  art.links[2].shapeFile = "articulated/three_links_spherical/bone_c.mochi.h5";
  art.links[2].colliderType = ColliderType::Box;
  art.links[2].parentJointFromLink = TransformRT(linkCTrans);
  art.links[2].centerOfMass = tipCom;
  // Set momentOfInertia too — Mochi requires both or neither to be set.
  art.links[2].momentOfInertia = Real6{1_r, 1_r, 1_r, 0_r, 0_r, 0_r};

  // Cycle that closes the loop linkA <- linkC.
  Real3 const cycleTrans{0.6_r, 0.2_r, 0_r};
  art.cycles.resize(1);
  art.cycles[0].parentLink = 0;
  art.cycles[0].childLink = 2;
  art.cycles[0].jointFromChildLink = TransformRT(cycleTrans);

  // Nest the inner prefab under an outer prefab applying an additional scale factor.
  real const nestedScale = 3_r;
  prefab::ScenePrefab outerPrefab;
  auto& ref = outerPrefab.prefabs.push_back();
  ref.name = "nested";
  ref.scale = nestedScale;
  ref.prefab = innerPrefab;

  prefab::LoadShapes(outerPrefab, test::GetAssetsDir(), context, test::ExpectOK{});
  {
    auto suppressWarning = test::SuppressLogWarning();
    prefab::AddToScene(outerPrefab, scene, {}, test::ExpectOK{});
  }

  // effectiveScale = actor.scale * nested.scale
  real const effectiveScale = actorScale * nestedScale;

  ActorHandle const artHandle = FindActorByName(scene, "nested/robot");
  ASSERT_TRUE(artHandle.IsValid());
  Actor* artActor = scene->GetActor(artHandle);
  ASSERT_EQ(ActorType::Articulated, artActor->GetType());

  // Verify prismatic joint limits scaled.
  // jointTypes contains numLinks (3) joint entries followed by numCycles (1) cycle entries.
  auto const info = artActor->GetArticulatedShapeInfo(test::ExpectOK{});
  ASSERT_EQ(4, isize(info.jointTypes));
  ASSERT_EQ(ArticulatedJointType::Prismatic, info.jointTypes[1]);
  EXPECT_NEAR_EQ(prismMin * effectiveScale, info.jointMinLimits[1]);
  EXPECT_NEAR_EQ(prismMax * effectiveScale, info.jointMaxLimits[1]);

  // Verify cycle jointFromChildLink scaled.
  int const numLinks = isize(art.links);
  ASSERT_EQ(1, isize(info.cycles));
  ASSERT_EQ(ArticulatedJointType::Cycle, info.jointTypes[numLinks]);
  EXPECT_NEAR_EQ(cycleTrans * effectiveScale, info.jointFromChildLink[numLinks].GetTranslation());

  // Verify link centerOfMass scaled (read on the rigid link actor).
  auto const& linkActors = artActor->GetNestedLinkActors(test::ExpectOK{});
  ASSERT_EQ(3, isize(linkActors));
  Actor* linkC = scene->GetActor(linkActors[2]);
  EXPECT_NEAR_EQ(tipCom * effectiveScale, linkC->GetRigidCenterOfMassLocal(test::ExpectOK{}));

  // Verify joint/link translations scaled (rest pose of linkC relative to root).
  auto scaleTranslation = [effectiveScale](TransformRT const& t) {
    return TransformRT(t.GetRotation(), t.GetTranslation() * effectiveScale);
  };
  TransformRT const rootFromLinkC = scaleTranslation(art.joints[0].parentLinkFromJoint) *
      scaleTranslation(art.links[0].parentJointFromLink) *
      scaleTranslation(art.joints[1].parentLinkFromJoint) *
      scaleTranslation(art.links[1].parentJointFromLink) *
      scaleTranslation(art.joints[2].parentLinkFromJoint) *
      scaleTranslation(art.links[2].parentJointFromLink);
  TransformRT const rootTransform = artActor->GetRootTransform();
  EXPECT_NEAR_EQ(
      rootTransform.GetTranslation() + rootFromLinkC.GetTranslation(),
      linkC->GetRootTransform().GetTranslation());
}

// Verifies that rotations propagate unchanged through the chain when translations are scaled
// under a nested prefab. The inner articulated actor has actor.scale and non-identity rotations
// on its revolute joints (parentLinkFromJoint); the outer prefab adds a uniform nested.scale.
// Checks (a) every chain translation scales by actor.scale * nested.scale, and (b) composed
// rotations are identical to the unscaled chain — scale must never leak into rotations.
TEST_IF(MOCHI_HDF5_AND_INTERNAL, Prefab, NestedPrefab_ArticulatedActor_ScaleWithRotations) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("test scene");
  MOCHI_DEFER(context->DestroyScene(scene));

  // Inner prefab: articulated actor with an individual (actor-level) scale.
  auto innerPrefab = std::make_shared<prefab::ScenePrefab>();
  auto& art = innerPrefab->actors.articulated.push_back();
  art.name = "robot";
  real const actorScale = 2_r;
  art.scale = actorScale;

  // Non-identity rotations on joint parentLinkFromJoint transforms. Link parentJointFromLink
  // rotations are intentionally left identity
  Quaternion const qJoint1 = Quaternion::FromAxisAngle(Real3{0_r, 0_r, 1_r}, 0.4_r);
  Quaternion const qJoint2 = Quaternion::FromAxisAngle(Real3{1_r, 0_r, 0_r}, -0.3_r);

  // Non-zero translations on every joint and link transform.
  Real3 const joint1Trans{0.5_r, 0.1_r, 0_r};
  Real3 const joint2Trans{0.3_r, -0.2_r, 0.1_r};
  Real3 const linkATrans{0_r, 0.15_r, 0_r};
  Real3 const linkBTrans{0.25_r, 0_r, -0.05_r};
  Real3 const linkCTrans{0.1_r, 0.1_r, 0.1_r};

  // Odd joint count (3): free root + two revolute.
  art.joints.resize(3);
  art.joints[0].type = ArticulatedJointType::Free;
  art.joints[1].type = ArticulatedJointType::Revolute;
  art.joints[1].axis = Real3{0_r, 0_r, 1_r};
  art.joints[1].parentLinkFromJoint = TransformRT(qJoint1, joint1Trans);
  art.joints[2].type = ArticulatedJointType::Revolute;
  art.joints[2].axis = Real3{1_r, 0_r, 0_r};
  art.joints[2].parentLinkFromJoint = TransformRT(qJoint2, joint2Trans);

  // Odd link count (3), each with a non-zero parentJointFromLink translation.
  art.links.resize(3);
  art.links[0].name = "linkA";
  art.links[0].parentLink = -1;
  art.links[0].shapeFile = "articulated/three_links_spherical/bone_a.mochi.h5";
  art.links[0].colliderType = ColliderType::Box;
  art.links[0].parentJointFromLink = TransformRT(linkATrans);
  art.links[1].name = "linkB";
  art.links[1].parentLink = 0;
  art.links[1].shapeFile = "articulated/three_links_spherical/bone_b.mochi.h5";
  art.links[1].colliderType = ColliderType::Box;
  art.links[1].parentJointFromLink = TransformRT(linkBTrans);
  art.links[2].name = "linkC";
  art.links[2].parentLink = 1;
  art.links[2].shapeFile = "articulated/three_links_spherical/bone_c.mochi.h5";
  art.links[2].colliderType = ColliderType::Box;
  art.links[2].parentJointFromLink = TransformRT(linkCTrans);

  // Outer prefab nests the inner at a uniform scale.
  real const nestedScale = 3_r;
  prefab::ScenePrefab outerPrefab;
  auto& ref = outerPrefab.prefabs.push_back();
  ref.name = "nested";
  ref.scale = nestedScale;
  ref.prefab = innerPrefab;

  prefab::LoadShapes(outerPrefab, test::GetAssetsDir(), context, test::ExpectOK{});
  prefab::AddToScene(outerPrefab, scene, {}, test::ExpectOK{});

  // Expected transforms: every translation is scaled by actor.scale * nested.scale; rotations
  // must remain unchanged.
  real const effectiveScale = actorScale * nestedScale;
  auto scaleTranslation = [effectiveScale](TransformRT const& t) {
    return TransformRT(t.GetRotation(), t.GetTranslation() * effectiveScale);
  };

  TransformRT const rootFromLinkA = scaleTranslation(art.joints[0].parentLinkFromJoint) *
      scaleTranslation(art.links[0].parentJointFromLink);
  TransformRT const rootFromLinkB = rootFromLinkA *
      scaleTranslation(art.joints[1].parentLinkFromJoint) *
      scaleTranslation(art.links[1].parentJointFromLink);
  TransformRT const rootFromLinkC = rootFromLinkB *
      scaleTranslation(art.joints[2].parentLinkFromJoint) *
      scaleTranslation(art.links[2].parentJointFromLink);

  // Sanity: scaling only translations must not change the composed rotation, but must change
  // the composed translation (given non-zero link/joint translations and scale != 1).
  TransformRT const unscaledRootFromLinkC =
      (art.joints[0].parentLinkFromJoint * art.links[0].parentJointFromLink) *
      art.joints[1].parentLinkFromJoint * art.links[1].parentJointFromLink *
      art.joints[2].parentLinkFromJoint * art.links[2].parentJointFromLink;
  EXPECT_NEAR_EQ(unscaledRootFromLinkC.GetRotation(), rootFromLinkC.GetRotation());
  EXPECT_FALSE(NearEqual(unscaledRootFromLinkC.GetTranslation(), rootFromLinkC.GetTranslation()));

  // The nested prefab sits at the world origin with identity rotation, so
  // worldFromActor = identity and each link's world transform equals rootFromLink.
  ActorHandle const artHandle = FindActorByName(scene, "nested/robot");
  ASSERT_TRUE(artHandle.IsValid());
  Actor* artActor = scene->GetActor(artHandle);
  ASSERT_EQ(ActorType::Articulated, artActor->GetType());

  auto const& linkActors = artActor->GetNestedLinkActors(test::ExpectOK{});
  ASSERT_EQ(3, isize(linkActors));

  Actor* linkA = scene->GetActor(linkActors[0]);
  Actor* linkB = scene->GetActor(linkActors[1]);
  Actor* linkC = scene->GetActor(linkActors[2]);
  EXPECT_NEAR_EQ(rootFromLinkA.GetRotation(), linkA->GetRootTransform().GetRotation());
  EXPECT_NEAR_EQ(rootFromLinkA.GetTranslation(), linkA->GetRootTransform().GetTranslation());
  EXPECT_NEAR_EQ(rootFromLinkB.GetRotation(), linkB->GetRootTransform().GetRotation());
  EXPECT_NEAR_EQ(rootFromLinkB.GetTranslation(), linkB->GetRootTransform().GetTranslation());
  EXPECT_NEAR_EQ(rootFromLinkC.GetRotation(), linkC->GetRootTransform().GetRotation());
  EXPECT_NEAR_EQ(rootFromLinkC.GetTranslation(), linkC->GetRootTransform().GetTranslation());
}

TEST_IF(MOCHI_HDF5_AND_INTERNAL, Prefab, ArticulatedActor_InitialJointVelocities) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("my scene");
  MOCHI_DEFER(context->DestroyScene(scene));

  prefab::PrefabParams params;
  params.name = "MyPrefab";

  // Create a prefab with an articulated actor with initial joint velocities.
  // Topology: 2 links (Free root + Revolute joint) = 7 DOFs.
  std::string json = R"({
    "actors": {
      "articulated": [
        {
          "name": "myArticulation",
          "joints": [
            {"type": "Free"},
            {"type": "Revolute", "axis": [0, 0, 1]}
          ],
          "links": [
            {
              "name": "boneA",
              "parentLink": -1,
              "colliderType": "Box",
              "shape": "articulated/two_links_revolute/bone_a.mochi.h5"
            },
            {
              "name": "boneB",
              "parentLink": 0,
              "colliderType": "Box",
              "shape": "articulated/two_links_revolute/bone_b.mochi.h5"
            }
          ],
          "jointVelocities": [1.3, -0.7, 0.6, -0.8, 0.5, 0.7, -0.3]
        }
      ]
    }
  })";

  prefab::ScenePrefab prefab = prefab::ShallowLoadFromJsonString(json, test::ExpectOK{});
  prefab::LoadShapes(prefab, test::GetAssetPath(""), context, test::ExpectOK{});
  auto const result = prefab::AddToScene(prefab, scene, params, test::ExpectOK{});

  // Verify the actor was created
  EXPECT_EQ(3, scene->GetNumActors()); // 1 articulated + 2 links
  auto const* actor = result.actors[0];
  EXPECT_EQ(ActorType::Articulated, actor->GetType());

  // Verify joint velocities were set correctly
  DynamicArray<real> jointVel(7);
  actor->GetArticulatedJointVelocities(MakeSpan(jointVel), test::ExpectOK{});

  EXPECT_NEAR(1.3_r, jointVel[0], 1e-6_r);
  EXPECT_NEAR(0.7_r, jointVel[5], 1e-6_r);

  scene->DestroyActor(actor->GetHandle());
}

TEST(Prefab, PoseController_Serialization) {
  std::string json = R"({
    "controllers": [
      {
        "_comment": "Good stuff",
        "articulatedActor": "myPrefab/myNested/myActor",
        "linkPosTracking": [
          {
            "damping": 0.1,
            "saturation": 0.2,
            "stiffness": 0.3
          }
        ],
        "linkRotTracking": [
          {
            "damping": 0.4,
            "saturation": 0.5,
            "stiffness": 0.6
          }
        ],
        "jointTracking": [
          {
            "damping": 0.7,
            "saturation": 0.8,
            "stiffness": 0.9
          }
        ]
      }
    ]
  })";

  // Expect deserialized values
  auto expectPoseController = [](prefab::PoseControllerPrefab const& controller) {
    EXPECT_STREQ("Good stuff", controller.comment->c_str());
    EXPECT_STREQ("myPrefab/myNested/myActor", controller.articulatedActor.c_str());
    EXPECT_EQ(1, controller.linkPosTracking.size());
    EXPECT_EQ(1, controller.linkRotTracking.size());
    EXPECT_EQ(1, controller.jointTracking.size());
    EXPECT_NEAR_EQ(0.1_r, controller.linkPosTracking[0].damping);
    EXPECT_NEAR_EQ(0.2_r, controller.linkPosTracking[0].saturation);
    EXPECT_NEAR_EQ(0.3_r, controller.linkPosTracking[0].stiffness);
    EXPECT_NEAR_EQ(0.4_r, controller.linkRotTracking[0].damping);
    EXPECT_NEAR_EQ(0.5_r, controller.linkRotTracking[0].saturation);
    EXPECT_NEAR_EQ(0.6_r, controller.linkRotTracking[0].stiffness);
    EXPECT_NEAR_EQ(0.7_r, controller.jointTracking[0].damping);
    EXPECT_NEAR_EQ(0.8_r, controller.jointTracking[0].saturation);
    EXPECT_NEAR_EQ(0.9_r, controller.jointTracking[0].stiffness);
  };

  // Load one articulated actor from JSON.
  prefab::ScenePrefab content = prefab::ShallowLoadFromJsonString(json, test::ExpectOK{});
  EXPECT_EQ(1, content.controllers.size());

  // If this fails, then a code change may have broken existing prefabs
  expectPoseController(content.controllers[0]);

  // Test round-trip serialization
  auto json2 = prefab::SaveToJsonString(content, test::ExpectOK{});
  prefab::ScenePrefab content2 = prefab::ShallowLoadFromJsonString(json2, test::ExpectOK{});
  EXPECT_EQ(1, content2.controllers.size());
  expectPoseController(content2.controllers[0]);
}

// A pose controller references an articulated actor by name and is attached to it during
// AddToScene. Size-1 tracking arrays broadcast to every link/joint, so one entry suffices for any
// topology.
TEST_IF(MOCHI_HDF5_AND_INTERNAL, Prefab, PoseController_AddToScene) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("test");
  MOCHI_DEFER(context->DestroyScene(scene));

  // Articulated actor with real link shapes (mirrors ArticulatedActor_AddToScene).
  prefab::ScenePrefab scenePrefab;
  auto& art = scenePrefab.actors.articulated.push_back();
  art.name = "myArticulation";
  art.joints.resize(2);
  art.joints[0].type = ArticulatedJointType::Free;
  art.joints[1].type = ArticulatedJointType::Revolute;
  art.joints[1].axis = Real3{0_r, 0_r, 1_r};
  art.links.resize(2);
  art.links[0].name = "boneA";
  art.links[0].parentLink = -1;
  art.links[0].shapeFile = "articulated/two_links_revolute/bone_a.mochi.h5";
  art.links[0].colliderType = ColliderType::Box;
  art.links[1].name = "boneB";
  art.links[1].parentLink = 0;
  art.links[1].shapeFile = "articulated/two_links_revolute/bone_b.mochi.h5";
  art.links[1].colliderType = ColliderType::Box;

  // Controller with one link-position and one joint tracking entry (both broadcast to all links).
  auto& controller = scenePrefab.controllers.push_back();
  controller.articulatedActor = "myArticulation";
  auto& linkPos = controller.linkPosTracking.push_back();
  linkPos.stiffness = 101_r;
  linkPos.damping = 11_r;
  auto& jointTrack = controller.jointTracking.push_back();
  jointTrack.stiffness = 202_r;
  jointTrack.damping = 22_r;

  prefab::LoadShapes(scenePrefab, test::GetAssetsDir(), context, test::ExpectOK{});
  auto const result = prefab::AddToScene(scenePrefab, scene, {}, test::ExpectOK{});

  // The articulated actor was created and now owns a pose controller.
  ASSERT_EQ(1, isize(result.actors));
  Actor* articulatedActor = result.actors[0];
  EXPECT_EQ(ActorType::Articulated, articulatedActor->GetType());
  EXPECT_TRUE(articulatedActor->HasArticulatedPoseController(test::ExpectOK{}));

  // GetArticulatedPoseControllerParams succeeds only when a controller exists; it fills numLinks
  // entries, and the authored gains were broadcast into each.
  int const numLinks = isize(articulatedActor->GetNestedLinkActors(test::ExpectOK{}));
  PoseControllerParams outParams(numLinks);
  articulatedActor->GetArticulatedPoseControllerParams(outParams, test::ExpectOK{});
  ASSERT_EQ(numLinks, isize(outParams.linkPosTracking));
  ASSERT_EQ(numLinks, isize(outParams.jointTracking));
  for (int i = 0; i < numLinks; ++i) {
    EXPECT_NEAR_EQ(101_r, outParams.linkPosTracking[i].stiffness);
    EXPECT_NEAR_EQ(11_r, outParams.linkPosTracking[i].damping);
    EXPECT_NEAR_EQ(202_r, outParams.jointTracking[i].stiffness);
    EXPECT_NEAR_EQ(22_r, outParams.jointTracking[i].damping);
  }
}

TEST(Prefab, PoseController_AddToScene_EmptyActorReturnsError) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("test");
  MOCHI_DEFER(context->DestroyScene(scene));

  // A pose controller with no articulated actor name cannot be attached.
  prefab::ScenePrefab scenePrefab;
  scenePrefab.controllers.push_back().articulatedActor = "";

  auto suppress = test::SuppressLogError();
  prefab::AddToScene(scenePrefab, scene, {}, test::ExpectNotOK{});
}

TEST(Prefab, PoseController_AddToScene_MissingActorReturnsError) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("test");
  MOCHI_DEFER(context->DestroyScene(scene));

  // The referenced articulated actor does not exist.
  prefab::ScenePrefab scenePrefab;
  scenePrefab.controllers.push_back().articulatedActor = "DoesNotExist";

  auto suppress = test::SuppressLogError();
  prefab::AddToScene(scenePrefab, scene, {}, test::ExpectNotOK{});
}

// The required prefab asset is not shipped externally.
TEST_IF(MOCHI_INTERNAL, Prefab, PoseController_AddToScene_NonArticulatedActorReturnsError) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("test");
  MOCHI_DEFER(context->DestroyScene(scene));

  // A pose controller may only reference an articulated actor, not a rigid one.
  prefab::ScenePrefab scenePrefab;
  auto& rigid = scenePrefab.actors.rigid.push_back();
  rigid.name = "Rigid";
  rigid.shapeFile = "cube/cube_minimal.mochi.json";
  rigid.colliderType = ColliderType::Box;
  scenePrefab.controllers.push_back().articulatedActor = "Rigid";

  prefab::LoadShapes(scenePrefab, test::GetAssetsDir(), context, test::ExpectOK{});
  auto suppress = test::SuppressLogError();
  prefab::AddToScene(scenePrefab, scene, {}, test::ExpectNotOK{});
}

TEST(Prefab, RigidActor_Serialization) {
  // One rigid actor with all non-default settings
  std::string json = R"({
    "actors": {
      "rigid": [
        {
          "_comment": "Good stuff",
          "name": "bob",
          "layer": "bob's layer",
          "shape": "some/dir/path",
          "scale": [0.1, 0.2, 0.3],
          "shapeRotation": [9.0, 8.0, 7.0, 6.0],
          "shapeTranslation": [5.0, 4.0, 3.0],
          "rotation": [1.0, 2.0, 3.0, 4.0],
          "translation": [5.0, 6.0, 7.0],
          "colliderType": "Box",
          "isStatic": true,
          "contact": {
            "penaltyCoefficient": 2e9,
            "penaltySmoothingHalfDistance": 3e-3,
            "frictionWithColliderNormal": false,
            "maxAlignmentNormals": -1,
            "viscousFrictionCoefficient": 0.123,
            "coulombFrictionCoefficient": 0.456,
            "frictionFalloffVel": 0.0234,
            "distanceErrorBound": 0.0567,
            "objScale": 1.111,
            "penaltyThresholdDefault": 4e-3,
            "penaltyThresholdExtraPadding": 0.00911
          },
          "sdf": {
            "resolutionMode": "Explicit",
            "resolutionDelta": [0.01, 0.02, 0.03],
            "boundaryPaddingDist": 0.0022
          },
          "hasGravity": false,
          "density": 1.23,
          "mass": 4.56,
          "centerOfMass": [0.1, 0.2, 0.3],
          "momentOfInertia": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
          "boundaryElementType": "P1Q6"
        }
      ]
    }
  })";

  // Expect deserialized values
  auto expectRigidActor = [](prefab::RigidActorPrefab const& rigid) {
    EXPECT_STREQ("Good stuff", rigid.comment->c_str());
    EXPECT_STREQ("bob", rigid.name.c_str());
    EXPECT_STREQ("bob's layer", rigid.layer.c_str());
    EXPECT_STREQ("some/dir/path", rigid.shapeFile.c_str());
    EXPECT_NEAR_EQ(Real3(0.1_r, 0.2_r, 0.3_r), rigid.scale);
    EXPECT_NEAR_EQ(Quaternion(9_r, 8_r, 7_r, 6_r), rigid.shapeRotation);
    EXPECT_NEAR_EQ(Real3(5_r, 4_r, 3_r), rigid.shapeTranslation);
    EXPECT_NEAR_EQ(Quaternion(1_r, 2_r, 3_r, 4_r), rigid.rotation);
    EXPECT_NEAR_EQ(Real3(5_r, 6_r, 7_r), rigid.translation);
    EXPECT_EQ(ColliderType::Box, rigid.colliderType);
    EXPECT_TRUE(rigid.isStatic);
    EXPECT_NEAR_EQ(2e9_r, rigid.contact.penaltyCoefficient);
    EXPECT_NEAR_EQ(3e-3_r, rigid.contact.penaltySmoothingHalfDistance);
    EXPECT_FALSE(rigid.contact.frictionWithColliderNormal);
    EXPECT_NEAR_EQ(-1_r, rigid.contact.maxAlignmentNormals);
    EXPECT_NEAR_EQ(0.123_r, rigid.contact.viscousFrictionCoefficient);
    EXPECT_NEAR_EQ(0.456_r, rigid.contact.coulombFrictionCoefficient);
    EXPECT_NEAR_EQ(0.0234_r, rigid.contact.frictionFalloffVel);
    EXPECT_NEAR_EQ(0.0567_r, rigid.contact.distanceErrorBound);
    EXPECT_NEAR_EQ(1.111_r, rigid.contact.objScale);
    EXPECT_EQ(4e-3_r, rigid.contact.penaltyThresholdDefault);
    EXPECT_NEAR_EQ(0.00911_r, rigid.contact.penaltyThresholdExtraPadding);
    EXPECT_NEAR_EQ(Real3(0.01_r, 0.02_r, 0.03_r), rigid.sdf.resolutionDelta);
    EXPECT_NEAR_EQ(0.0022_r, rigid.sdf.boundaryPaddingDist);
    EXPECT_FALSE(rigid.hasGravity);
    EXPECT_NEAR_EQ(1.23_r, *rigid.density);
    EXPECT_NEAR_EQ(4.56_r, *rigid.mass);
    EXPECT_NEAR_EQ(Real3(0.1_r, 0.2_r, 0.3_r), *rigid.centerOfMass);
    EXPECT_NEAR_EQ(Real6(0.1_r, 0.2_r, 0.3_r, 0.4_r, 0.5_r, 0.6_r), *rigid.momentOfInertia);
    EXPECT_EQ(ActorBoundaryElementType::P1Q6, rigid.boundaryElementType);
  };

  // Load one rigid actor from JSON.
  prefab::ScenePrefab content = prefab::ShallowLoadFromJsonString(json, test::ExpectOK{});
  EXPECT_EQ(1, content.actors.rigid.size());

  // If this fails, then a code change may have broken existing prefabs
  expectRigidActor(content.actors.rigid[0]);

  // Test round-trip serialization
  auto json2 = prefab::SaveToJsonString(content, test::ExpectOK{});
  prefab::ScenePrefab content2 = prefab::ShallowLoadFromJsonString(json2, test::ExpectOK{});
  EXPECT_EQ(1, content2.actors.rigid.size());
  expectRigidActor(content2.actors.rigid[0]);
}

// The prefab assets used by the tests below are not shipped externally.
TEST_IF(MOCHI_INTERNAL, Prefab, RigidActor_AddToScene) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("my scene");
  MOCHI_DEFER(context->DestroyScene(scene));

  // Use a prefab to add a rigid actor to a scene.
  // No need to re-test serialization for this part.
  prefab::ScenePrefab scenePrefab;
  auto& rigid = scenePrefab.actors.rigid.push_back();
  rigid.name = "bob";
  rigid.shapeFile = "cube/cube_minimal.mochi.json";
  rigid.isStatic = true;
  rigid.colliderType = ColliderType::Box;
  rigid.scale = Real3(0.1_r, 0.2_r, 0.3_r);
  rigid.rotation = Quaternion(1_r, 2_r, 3_r, 4_r);
  rigid.translation = Real3(5_r, 6_r, 7_r);

  // First load the referenced shape
  prefab::LoadShapes(scenePrefab, test::GetAssetsDir(), context, test::ExpectOK{});
  EXPECT_TRUE(scenePrefab.actors.rigid[0].shape.IsValid());

  // Add the prefab to the scene
  prefab::PrefabParams prefabParams;
  prefabParams.rotation = Quaternion::FromAxisAngle(Real3{0_r, 1_r, 0_r}, 0.5_r);
  prefabParams.translation = Real3{2.22_r, 3.33_r, 4.44_r};
  prefab::AddToScene(scenePrefab, scene, prefabParams, test::ExpectOK{});
  EXPECT_EQ(1, scene->GetNumActors());

  TransformSRT worldFromPrefab(prefabParams.scale, prefabParams.rotation, prefabParams.translation);

  // Get the ActorHandle and check the actor properties.
  ActorHandle actorHandle;
  std::string expectedActorName = "bob";
  auto checkActor = [&](Actor* actor) {
    EXPECT_STREQ(expectedActorName.c_str(), actor->GetName());
    EXPECT_EQ(ActorType::Rigid, actor->GetType());
    EXPECT_EQ(rigid.isStatic, actor->IsStatic());
    TransformSRT prefabFromActor{1_r, Normalize(rigid.rotation), rigid.translation};
    TransformSRT worldFromActor = worldFromPrefab * prefabFromActor;
    EXPECT_NEAR_EQ(worldFromActor.GetRotation(), actor->GetRootTransform().GetRotation());
    EXPECT_NEAR_EQ(worldFromActor.GetTranslation(), actor->GetRootTransform().GetTranslation());
    actorHandle = actor->GetHandle();
  };
  scene->ForEachActor(checkActor);

  // Destroy the actor
  scene->DestroyActor(actorHandle);

  // Repeat but give the prefab instance a name, and use AddToSceneResult to get the actors.
  rigid.name = "fred";
  rigid.isStatic = false;
  prefabParams.name = "myPrefab";
  expectedActorName = "myPrefab/fred";
  auto const result = prefab::AddToScene(scenePrefab, scene, prefabParams, test::ExpectOK{});
  EXPECT_EQ(1, scene->GetNumActors());
  scene->ForEachActor(checkActor);
  ASSERT_EQ(1, isize(result.actors));
  EXPECT_EQ(ActorType::Rigid, result.actors[0]->GetType());
  EXPECT_EQ(actorHandle, result.actors[0]->GetHandle());
  EXPECT_EQ(result.actors, result.Filter(ActorType::Rigid));
  scene->DestroyActor(actorHandle);
}

TEST_IF(MOCHI_INTERNAL, Prefab, RigidActor_ShapeTransform) {
  // Load a prefab with a rigid actor that has non-default shapeRotation and shapeTranslation
  std::string json = R"({
    "actors": {
      "rigid": [
        {
          "name": "bob",
          "shape": "cube/cube_minimal.mochi.json",
          "scale": [0.1, 0.2, 0.3],
          "shapeRotation": [0.7071, 0, 0, 0.7071],
          "shapeTranslation": [1, 2, 3]
        }
      ]
    }
  })";

  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));

  // Deep load
  auto loadedPrefab =
      prefab::LoadFromJsonString(json, test::GetAssetPath(""), context, test::ExpectOK{});
  EXPECT_EQ(1, isize(loadedPrefab.actors.rigid));
  auto shape = loadedPrefab.actors.rigid[0].shape;
  EXPECT_TRUE(shape.IsValid());

  // Check the bounds of the loaded shape, with baked-in transform.
  // Expect scale was applied, then rotation (+90 deg about X), then translation.
  auto shapeBounds = context->GetShapeAabb(shape, test::ExpectOK{});
  auto constexpr kTol = kDefaultNearEqualEpsilon<real> * 10_r;
  EXPECT_NEAR_TOL(Real3(1_r, 1.7_r, 3_r), shapeBounds.GetMin(), kTol);
  EXPECT_NEAR_TOL(Real3(1.1_r, 2_r, 3.2_r), shapeBounds.GetMax(), kTol);
}

TEST_IF(MOCHI_INTERNAL, Prefab, RigidActor_InitialVelocity) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("my scene");
  MOCHI_DEFER(context->DestroyScene(scene));

  prefab::PrefabParams params;
  params.name = "MyPrefab";

  // Create a prefab with a rigid actor with initial velocities
  std::string json = R"({
    "actors": {
      "rigid": [
        {
          "name": "Cube",
          "layer": "Object",
          "colliderType": "Box",
          "scale": [0.2, 0.2, 0.2],
          "shape": "cube/cube_minimal.mochi.json",
          "translation": [0, 0.5, 0],
          "linearVelocity": [1.0, 2.0, 3.0],
          "angularVelocity": [0.1, 0.2, 0.3]
        }
      ]
    }
  })";

  prefab::ScenePrefab prefab = prefab::ShallowLoadFromJsonString(json, test::ExpectOK{});
  prefab::LoadShapes(prefab, test::GetAssetPath(""), context, test::ExpectOK{});
  auto const result = prefab::AddToScene(prefab, scene, params, test::ExpectOK{});

  // Verify the actor was created with correct velocities
  EXPECT_EQ(1, scene->GetNumActors());
  ASSERT_EQ(1, isize(result.actors));
  auto const* actor = result.actors[0];
  EXPECT_EQ(ActorType::Rigid, actor->GetType());

  Real3 linearVel = actor->GetLinearVelocity(test::ExpectOK{});
  Real3 angularVel = actor->GetAngularVelocity(test::ExpectOK{});

  EXPECT_NEAR(1.0_r, linearVel[0], 1e-6_r);
  EXPECT_NEAR(2.0_r, linearVel[1], 1e-6_r);
  EXPECT_NEAR(3.0_r, linearVel[2], 1e-6_r);

  EXPECT_NEAR(0.1_r, angularVel[0], 1e-6_r);
  EXPECT_NEAR(0.2_r, angularVel[1], 1e-6_r);
  EXPECT_NEAR(0.3_r, angularVel[2], 1e-6_r);

  scene->DestroyActor(actor->GetHandle());
}

TEST(Prefab, SoftActor_Serialization) {
  // One soft actor with all non-default settings
  std::string json = R"({
    "actors": {
      "soft": [
        {
          "_comment": "Good stuff",
          "name": "bob",
          "layer": "bob's layer",
          "shape": "some/dir/path",
          "scale": [0.1, 0.2, 0.3],
          "shapeRotation": [9.0, 8.0, 7.0, 6.0],
          "shapeTranslation": [5.0, 4.0, 3.0],
          "rotation": [1.0, 2.0, 3.0, 4.0],
          "translation": [5.0, 6.0, 7.0],
          "material": {
            "type": "LinearElastic",
            "linearElastic": {
              "youngsModulus": 211111,
              "poissonRatio": 0.325
            },
            "density": 4321
          },
          "colliderType": "Box",
          "contact": {
            "penaltyCoefficient": 2e9,
            "penaltySmoothingHalfDistance": 3e-3,
            "frictionWithColliderNormal": false,
            "maxAlignmentNormals": -1,
            "viscousFrictionCoefficient": 0.123,
            "coulombFrictionCoefficient": 0.456,
            "frictionFalloffVel": 0.0234,
            "distanceErrorBound": 0.0567,
            "objScale": 1.111,
            "penaltyThresholdDefault": 4e-3,
            "penaltyThresholdExtraPadding": 0.00911
          },
          "sdf": {
            "resolutionMode": "Explicit",
            "resolutionDelta": [0.01, 0.02, 0.03],
            "boundaryPaddingDist": 0.0022
          },
          "flow": "some/other/path",
          "useRecentering": false,
          "hasGravity": false,
          "hasInertia": false,
          "hasStress": false,
          "boundaryElementType": "P1Q6"
        }
      ]
    }
  })";

  // Expect deserialized values
  auto expectSoftActor = [](prefab::SoftActorPrefab const& soft) {
    EXPECT_STREQ("Good stuff", soft.comment->c_str());
    EXPECT_STREQ("bob", soft.name.c_str());
    EXPECT_STREQ("bob's layer", soft.layer.c_str());
    EXPECT_STREQ("some/dir/path", soft.shapeFile.c_str());
    EXPECT_NEAR_EQ(Real3(0.1_r, 0.2_r, 0.3_r), soft.scale);
    EXPECT_NEAR_EQ(Quaternion(9_r, 8_r, 7_r, 6_r), soft.shapeRotation);
    EXPECT_NEAR_EQ(Real3(5_r, 4_r, 3_r), soft.shapeTranslation);
    EXPECT_NEAR_EQ(Quaternion(1_r, 2_r, 3_r, 4_r), soft.rotation);
    EXPECT_NEAR_EQ(Real3(5_r, 6_r, 7_r), soft.translation);
    EXPECT_EQ(SoftMaterialType::LinearElastic, soft.material.type);
    EXPECT_NEAR_EQ(211111_r, soft.material.linearElastic.youngsModulus);
    EXPECT_NEAR_EQ(0.325_r, soft.material.linearElastic.poissonRatio);
    EXPECT_NEAR_EQ(4321_r, soft.material.density);
    EXPECT_EQ(NeoHookeanMaterialParams{}, soft.material.neoHookean);
    EXPECT_EQ(StVenantKirchhoffMaterialParams{}, soft.material.stVenantKirchhoff);
    EXPECT_EQ(ArapMaterialParams{}, soft.material.arap);
    EXPECT_EQ(ActiveShapeTargetingArapMaterialParams{}, soft.material.activeShapeTargetingArap);
    EXPECT_EQ(ActiveNeoHookeanMaterialParams{}, soft.material.activeNeoHookean);
    EXPECT_EQ(ColliderType::Box, soft.colliderType);
    EXPECT_NEAR_EQ(2e9_r, soft.contact.penaltyCoefficient);
    EXPECT_NEAR_EQ(3e-3_r, soft.contact.penaltySmoothingHalfDistance);
    EXPECT_FALSE(soft.contact.frictionWithColliderNormal);
    EXPECT_NEAR_EQ(-1_r, soft.contact.maxAlignmentNormals);
    EXPECT_NEAR_EQ(0.123_r, soft.contact.viscousFrictionCoefficient);
    EXPECT_NEAR_EQ(0.456_r, soft.contact.coulombFrictionCoefficient);
    EXPECT_NEAR_EQ(0.0234_r, soft.contact.frictionFalloffVel);
    EXPECT_NEAR_EQ(0.0567_r, soft.contact.distanceErrorBound);
    EXPECT_NEAR_EQ(1.111_r, soft.contact.objScale);
    EXPECT_EQ(4e-3_r, soft.contact.penaltyThresholdDefault);
    EXPECT_NEAR_EQ(0.00911_r, soft.contact.penaltyThresholdExtraPadding);
    EXPECT_NEAR_EQ(Real3(0.01_r, 0.02_r, 0.03_r), soft.sdf.resolutionDelta);
    EXPECT_NEAR_EQ(0.0022_r, soft.sdf.boundaryPaddingDist);
    EXPECT_STREQ("some/other/path", soft.flowFile.c_str());
    EXPECT_FALSE(soft.useRecentering);
    EXPECT_FALSE(soft.hasGravity);
    EXPECT_FALSE(soft.hasInertia);
    EXPECT_FALSE(soft.hasStress);
    EXPECT_EQ(ActorBoundaryElementType::P1Q6, soft.boundaryElementType);
  };

  // Load one soft actor from JSON.
  prefab::ScenePrefab content = prefab::ShallowLoadFromJsonString(json, test::ExpectOK{});
  EXPECT_EQ(1, content.actors.soft.size());

  // If this fails, then a code change may break existing prefab files.
  expectSoftActor(content.actors.soft[0]);

  // Test round-trip serialization
  auto json2 = prefab::SaveToJsonString(content, test::ExpectOK{});
  prefab::ScenePrefab content2 = prefab::ShallowLoadFromJsonString(json2, test::ExpectOK{});
  EXPECT_EQ(1, content2.actors.soft.size());
  expectSoftActor(content2.actors.soft[0]);
}

// The prefab assets used by the tests below are not shipped externally.
TEST_IF(MOCHI_INTERNAL, Prefab, SoftActor_AddToScene) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("my scene");
  MOCHI_DEFER(context->DestroyScene(scene));

  // Use a prefab to add a soft actor to a scene.
  // No need to re-test serialization for this part.
  prefab::ScenePrefab scenePrefab;
  auto& soft = scenePrefab.actors.soft.push_back();
  soft.name = "bob";
  soft.shapeFile = "cube/cube_minimal.mochi.json";
  soft.scale = Real3(0.1_r, 0.2_r, 0.3_r);
  soft.rotation = Quaternion(1_r, 2_r, 3_r, 4_r);
  soft.translation = Real3(5_r, 6_r, 7_r);

  // First load the referenced shape
  prefab::LoadShapes(scenePrefab, test::GetAssetsDir(), context, test::ExpectOK{});
  EXPECT_TRUE(scenePrefab.actors.soft[0].shape.IsValid());

  // Parameters
  prefab::PrefabParams prefabParams;
  prefabParams.rotation = Quaternion::FromAxisAngle(Real3{0_r, 1_r, 0_r}, 0.5_r);
  prefabParams.translation = Real3{2.22_r, 3.33_r, 4.44_r};

  // Add an instance of the loaded ScenePrefab to the scene
  prefab::AddToScene(scenePrefab, scene, prefabParams, test::ExpectOK{});
  EXPECT_EQ(1, scene->GetNumActors());

  TransformSRT worldFromPrefab(prefabParams.scale, prefabParams.rotation, prefabParams.translation);

  // Get the ActorHandle and check the actor properties.
  ActorHandle actorHandle;
  std::string expectedActorName = "bob";
  auto checkActor = [&](Actor* actor) {
    EXPECT_STREQ(expectedActorName.c_str(), actor->GetName());
    EXPECT_EQ(ActorType::Soft, actor->GetType());
    TransformSRT prefabFromActor{1_r, Normalize(soft.rotation), soft.translation};
    TransformSRT worldFromActor = worldFromPrefab * prefabFromActor;
    EXPECT_NEAR_EQ(worldFromActor.GetRotation(), actor->GetRootTransform().GetRotation());
    EXPECT_NEAR_EQ(worldFromActor.GetTranslation(), actor->GetRootTransform().GetTranslation());
    actorHandle = actor->GetHandle();
  };
  scene->ForEachActor(checkActor);

  // Destroy the actor
  scene->DestroyActor(actorHandle);

  // Repeat, but this time give the prefab instance a name, and use AddToSceneResult to get the
  // actors.
  prefabParams.name = "myPrefab";
  soft.name = "fred";
  expectedActorName = "myPrefab/fred";
  auto const result = prefab::AddToScene(scenePrefab, scene, prefabParams, test::ExpectOK{});
  EXPECT_EQ(1, scene->GetNumActors());
  scene->ForEachActor(checkActor);
  ASSERT_EQ(1, isize(result.actors));
  EXPECT_EQ(actorHandle, result.actors[0]->GetHandle());
}

TEST_IF(MOCHI_HDF5_AND_INTERNAL, Prefab, SoftSkinnedActor_AddToScene) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("my scene");
  MOCHI_DEFER(context->DestroyScene(scene));

  prefab::PrefabParams params;
  params.name = "MyPrefab";
  // TODO: Test scale, rotation, and translation at the prefab level

  // Load a prefab with a soft skinned actor
  auto const result = prefab::AddToScene(
      test::GetAssetPath("allegro_soft/allegro_soft.mochi_prefab"),
      test::GetAssetPath(""),
      scene,
      params,
      test::ExpectOK{});

  // The prefab specified one (soft skinned) actor, so the function should return just one
  // Actor*. It should be the articulated actor.
  ASSERT_EQ(1, isize(result.actors));
  auto const* articulatedActor = result.actors[0];
  ASSERT_NE(nullptr, articulatedActor);
  EXPECT_EQ(ActorType::Articulated, articulatedActor->GetType());
  EXPECT_STREQ("MyPrefab/allegro_soft_hand", articulatedActor->GetName());

  // Nested Link Actors
  auto const& linkActors = articulatedActor->GetNestedLinkActors(test::ExpectOK{});
  EXPECT_EQ(21, isize(linkActors));
  for (int i = 0; i < 21; ++i) {
    auto const* actor = scene->GetActor(linkActors[i]);
    ASSERT_NE(nullptr, actor);
    EXPECT_EQ(ActorType::Rigid, actor->GetType());
  }

  // Nested Soft Actors
  std::array<char const*, 4> softActorNames = {
      "MyPrefab/allegro_soft_hand/link_15.0_tip_soft",
      "MyPrefab/allegro_soft_hand/link_11.0_tip_soft",
      "MyPrefab/allegro_soft_hand/link_7.0_tip_soft",
      "MyPrefab/allegro_soft_hand/link_3.0_tip_soft"};
  auto const& softActors = articulatedActor->GetNestedSoftActors(test::ExpectOK{});
  EXPECT_EQ(isize(softActorNames), isize(softActors));
  for (int i = 0; i < isize(softActors); ++i) {
    auto const* actor = scene->GetActor(softActors[i]);
    ASSERT_NE(nullptr, actor);
    EXPECT_EQ(ActorType::Soft, actor->GetType());
    EXPECT_STREQ(softActorNames[i], actor->GetName());
  }
}

TEST(Prefab, Serialization_SceneParams) {
  // Scene settings with all non-default settings
  std::string json = R"({
    "scene": {
      "_comment": "Good stuff",
      "description": "My Scene",
      "gravity": [
        1.0,
        2.0,
        3.0
      ],
      "solver": {
        "nonLinearSolver": {
          "solverType": "BFGS",
          "dResidualAssemblyPeriod": 2,
          "maxIter": 6,
          "maxElapsedTimeSeconds": 3.14,
          "absTol": 6.78e-05,
          "relTol": 6.66e-05,
          "relStepTol": 0.555,
          "stopIfNoImprovement": true,
          "psdProjMode": "IfFailRetry",
          "gradientDescentFallback": true,
          "explosionControl": false,
          "absDivTol": 3450000000,
          "relDivTol": 44400,
          "lineSearchMaxIter": 8,
          "lineSearchAlpha": 0.5123,
          "lineSearchWolfe1": 2.22e-05,
          "lineSearchWolfe2": 3.33,
          "lineSearchMaxRelIncrease": 0.111,
          "lineSearchType": "Simple",
          "linearToleranceStrategy": "EisenstatWalker1",
          "verbosity": "Verbose"
        },
        "linearSolver": {
          "solverType": "ParallelCG",
          "preconditionerType": "BlockJacobi",
          "normType": "ResidualL2",
          "absTol": 1.23e-06,
          "relTol": 3.32e-06,
          "relDivTol": 12300000000,
          "maxIter": 911,
          "restartSize": 2345,
          "abortIfNotSpd": true
        },
        "integrationMethod": "BDF2",
        "experimentalEval": {
          "explicitNormals": true,
          "fadeFriction": false,
          "implicitNormalForceForDissipation": true,
          "fittedSaturationHessian": {
            "contactFriction": false,
            "jointFriction": false,
            "constraintSaturation": false
          }
        }
      }
    },
    "contactFilter": {
        "actorContactSymmetric": [
          {
            "enable": true,
            "includeNestedActors": false,
            "actors": ["ActorA", "ActorB"]
          }
        ],
      "actorContactAsymmetric": [
        {
          "enable": false,
          "actors": ["ActorC", "ActorD"]
        }
      ],
      "layerContactSymmetric": [
        {
          "enable": true,
          "layers": ["LayerA", "LayerA"]
        },
        {
          "enable": true,
          "layers": ["LayerA", "LayerB"]
        }
      ],
      "layerContactAsymmetric": [
        {
          "enable": false,
          "layers": ["LayerB", "LayerA"]
        }
      ]
    }
  })";

  // Expect deserialized values for SceneParams
  auto expectSceneParams = [](prefab::SceneParams const& scene) {
    EXPECT_STREQ("Good stuff", scene.comment->c_str());

    // Description
    EXPECT_STREQ("My Scene", scene.description.c_str());

    // Gravity
    ASSERT_TRUE(scene.gravity.has_value());
    EXPECT_EQ(Real3(1_r, 2_r, 3_r), *scene.gravity);

    // Solver Params:
    ASSERT_TRUE(scene.solver.has_value());
    auto const& solver = *scene.solver;
    EXPECT_EQ(SaturationHessianParams::All(false), solver.experimentalEval.fittedSaturationHessian);
    EXPECT_EQ(true, solver.experimentalEval.implicitNormalForceForDissipation);
    EXPECT_EQ(false, solver.experimentalEval.fadeFriction);
    EXPECT_EQ(true, solver.experimentalEval.explicitNormals);
    EXPECT_EQ(IntegrationMethod::BDF2, solver.integrationMethod);
    EXPECT_EQ(LinearSolverType::ParallelCG, solver.linearSolver.solverType);
    EXPECT_EQ(PreconditionerType::BlockJacobi, solver.linearSolver.preconditionerType);
    EXPECT_EQ(LinearSolverConvergenceNorm::ResidualL2, solver.linearSolver.normType);
    EXPECT_EQ(911, solver.linearSolver.maxIter);
    EXPECT_NEAR_EQ(1.23e-6_r, solver.linearSolver.absTol);
    EXPECT_NEAR_EQ(12300000000_r, solver.linearSolver.relDivTol);
    EXPECT_NEAR_EQ(3.32e-6_r, solver.linearSolver.relTol);
    EXPECT_EQ(2345, solver.linearSolver.restartSize);
    EXPECT_EQ(NonLinearSolverType::BFGS, solver.nonLinearSolver.solverType);
    EXPECT_EQ(
        LinearToleranceStrategy::EisenstatWalker1, solver.nonLinearSolver.linearToleranceStrategy);
    EXPECT_EQ(6, solver.nonLinearSolver.maxIter);
    EXPECT_NEAR_EQ(3.14, solver.nonLinearSolver.maxElapsedTimeSeconds);
    EXPECT_NEAR_EQ(6.66e-5_r, solver.nonLinearSolver.relTol);
    EXPECT_NEAR_EQ(44400_r, solver.nonLinearSolver.relDivTol);
    EXPECT_NEAR_EQ(0.555_r, solver.nonLinearSolver.relStepTol);
    EXPECT_NEAR_EQ(0.5123_r, solver.nonLinearSolver.lineSearchAlpha);
    EXPECT_EQ(8, solver.nonLinearSolver.lineSearchMaxIter);
    EXPECT_NEAR_EQ(0.111_r, solver.nonLinearSolver.lineSearchMaxRelIncrease);
    EXPECT_NEAR_EQ(2.22e-5_r, solver.nonLinearSolver.lineSearchWolfe1);
    EXPECT_NEAR_EQ(3.33_r, solver.nonLinearSolver.lineSearchWolfe2);
    EXPECT_NEAR_EQ(3450000000_r, solver.nonLinearSolver.absDivTol);
    EXPECT_NEAR_EQ(6.78e-5_r, solver.nonLinearSolver.absTol);
    EXPECT_EQ(LineSearchType::Simple, solver.nonLinearSolver.lineSearchType);
    EXPECT_EQ(2, solver.nonLinearSolver.dResidualAssemblyPeriod);
    EXPECT_FALSE(solver.nonLinearSolver.explosionControl);
    EXPECT_TRUE(solver.nonLinearSolver.gradientDescentFallback);
    EXPECT_TRUE(solver.nonLinearSolver.stopIfNoImprovement);
    EXPECT_EQ(VerbosityLevel::Verbose, solver.nonLinearSolver.verbosity);
    EXPECT_EQ(PsdProjectionMode::IfFailRetry, solver.nonLinearSolver.psdProjMode);
  };

  // Expect deserialized values for contactFilter
  auto expectContactFilter = [](prefab::ContactFilter const& contactFilter) {
    // Actor contact symmetric entries
    ASSERT_TRUE(contactFilter.actorContactSymmetric.has_value());
    EXPECT_EQ(1, contactFilter.actorContactSymmetric->size());
    EXPECT_TRUE((*contactFilter.actorContactSymmetric)[0].enable);
    EXPECT_FALSE((*contactFilter.actorContactSymmetric)[0].includeNestedActors);
    EXPECT_EQ(2, (*contactFilter.actorContactSymmetric)[0].actors.size());
    EXPECT_EQ("ActorA", (*contactFilter.actorContactSymmetric)[0].actors[0]);
    EXPECT_EQ("ActorB", (*contactFilter.actorContactSymmetric)[0].actors[1]);

    // Actor contact asymmetric entries
    ASSERT_TRUE(contactFilter.actorContactAsymmetric.has_value());
    EXPECT_EQ(1, contactFilter.actorContactAsymmetric->size());
    EXPECT_FALSE((*contactFilter.actorContactAsymmetric)[0].enable);
    EXPECT_TRUE((*contactFilter.actorContactAsymmetric)[0].includeNestedActors);
    EXPECT_EQ(2, (*contactFilter.actorContactAsymmetric)[0].actors.size());
    EXPECT_EQ("ActorC", (*contactFilter.actorContactAsymmetric)[0].actors[0]);
    EXPECT_EQ("ActorD", (*contactFilter.actorContactAsymmetric)[0].actors[1]);

    // Layer contact symmetric entries
    ASSERT_TRUE(contactFilter.layerContactSymmetric.has_value());
    EXPECT_EQ(2, contactFilter.layerContactSymmetric->size());
    EXPECT_TRUE((*contactFilter.layerContactSymmetric)[0].enable);
    EXPECT_EQ(2, (*contactFilter.layerContactSymmetric)[0].layers.size());
    EXPECT_EQ("LayerA", (*contactFilter.layerContactSymmetric)[0].layers[0]);
    EXPECT_EQ("LayerA", (*contactFilter.layerContactSymmetric)[0].layers[1]);
    EXPECT_TRUE((*contactFilter.layerContactSymmetric)[1].enable);
    EXPECT_EQ(2, (*contactFilter.layerContactSymmetric)[1].layers.size());
    EXPECT_EQ("LayerA", (*contactFilter.layerContactSymmetric)[1].layers[0]);
    EXPECT_EQ("LayerB", (*contactFilter.layerContactSymmetric)[1].layers[1]);

    // Layer contact asymmetric entries
    ASSERT_TRUE(contactFilter.layerContactAsymmetric.has_value());
    EXPECT_EQ(1, contactFilter.layerContactAsymmetric->size());
    EXPECT_FALSE((*contactFilter.layerContactAsymmetric)[0].enable);
    EXPECT_EQ(2, (*contactFilter.layerContactAsymmetric)[0].layers.size());
    EXPECT_EQ("LayerB", (*contactFilter.layerContactAsymmetric)[0].layers[0]);
    EXPECT_EQ("LayerA", (*contactFilter.layerContactAsymmetric)[0].layers[1]);
  };

  // Load scene params and contact filter from JSON
  prefab::ScenePrefab content = prefab::ShallowLoadFromJsonString(json, test::ExpectOK{});
  EXPECT_TRUE(content.scene.has_value());
  EXPECT_TRUE(content.contactFilter.has_value());

  // If this fails, then a code change may break existing prefab files.
  expectSceneParams(*content.scene);
  expectContactFilter(*content.contactFilter);

  // Test round-trip serialization
  auto json2 = prefab::SaveToJsonString(content, test::ExpectOK{});
  prefab::ScenePrefab content2 = prefab::ShallowLoadFromJsonString(json2, test::ExpectOK{});
  EXPECT_TRUE(content2.scene.has_value());
  EXPECT_TRUE(content2.contactFilter.has_value());
  expectSceneParams(*content2.scene);
  expectContactFilter(*content2.contactFilter);
}

TEST(Prefab, SceneParams_AddToScene) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("my scene");
  MOCHI_DEFER(context->DestroyScene(scene));
  scene->SetGravity(Real3{0_r, 0_r, 0_r});
  auto solverParams = scene->GetSolverParams();
  solverParams.linearSolver.maxIter = 123;
  scene->SetSolverParams(solverParams, test::ExpectOK{});

  auto prefabWithoutOverrides =
      prefab::ShallowLoadFromJsonString(R"({"scene": {}})", test::ExpectOK{});
  ASSERT_TRUE(prefabWithoutOverrides.scene.has_value());
  EXPECT_FALSE(prefabWithoutOverrides.scene->gravity.has_value());
  EXPECT_FALSE(prefabWithoutOverrides.scene->solver.has_value());

  prefab::AddToScene(prefabWithoutOverrides, scene, prefab::PrefabParams{}, test::ExpectOK{});
  EXPECT_EQ(Real3(0_r, 0_r, 0_r), scene->GetGravity());
  EXPECT_EQ(123, scene->GetSolverParams().linearSolver.maxIter);

  auto prefabWithOverrides = prefab::ShallowLoadFromJsonString(
      R"({"scene": {"gravity": [1, 2, 3], "solver": {}}})", test::ExpectOK{});
  prefab::PrefabParams prefabParams;
  prefabParams.applySceneSettings = false;
  prefab::AddToScene(prefabWithOverrides, scene, prefabParams, test::ExpectOK{});
  EXPECT_EQ(Real3(0_r, 0_r, 0_r), scene->GetGravity());
  EXPECT_EQ(123, scene->GetSolverParams().linearSolver.maxIter);

  prefabParams.applySceneSettings = true;

  auto prefabWithGravityOverride =
      prefab::ShallowLoadFromJsonString(R"({"scene": {"gravity": [1, 2, 3]}})", test::ExpectOK{});
  prefab::AddToScene(prefabWithGravityOverride, scene, prefabParams, test::ExpectOK{});
  EXPECT_NEAR_EQ(Real3(1_r, 2_r, 3_r), scene->GetGravity());
  EXPECT_EQ(123, scene->GetSolverParams().linearSolver.maxIter);

  auto prefabWithSolverOverride =
      prefab::ShallowLoadFromJsonString(R"({"scene": {"solver": {}}})", test::ExpectOK{});
  prefab::AddToScene(prefabWithSolverOverride, scene, prefabParams, test::ExpectOK{});
  EXPECT_NEAR_EQ(Real3(1_r, 2_r, 3_r), scene->GetGravity());
  EXPECT_EQ(SolverParams{}.linearSolver.maxIter, scene->GetSolverParams().linearSolver.maxIter);
}

TEST_IF(MOCHI_HDF5_AND_INTERNAL, Prefab, LoadAndAddToScene) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("my scene");
  MOCHI_DEFER(context->DestroyScene(scene));

  // This overload of prefab::AddToScene is your one-stop-shopping for loading a prefab from file
  // directly into a scene. It is equivalent to:
  //    1) prefab::ShallowLoadFromFile
  //    2) prefab::LoadNestedPrefabs
  //    3) prefab::LoadShapes
  //    4) prefab::AddToScene

  auto const result = prefab::AddToScene(
      test::GetAssetPath("samples/soft_letters_mochi.mochi_scene"),
      test::GetAssetsDir(),
      scene,
      prefab::PrefabParams{},
      test::ExpectOK{});

  // Actors for "Ground", "m", "o", "c", "h", "i"
  EXPECT_EQ(6, scene->GetNumActors());
  ASSERT_EQ(6, isize(result.actors));
  auto rigidActors = result.Filter(ActorType::Rigid);
  auto softActors = result.Filter(ActorType::Soft);
  ASSERT_EQ(1, isize(rigidActors));
  EXPECT_STREQ("Ground", rigidActors[0]->GetName());
  ASSERT_EQ(5, isize(softActors));
  EXPECT_STREQ("m", softActors[0]->GetName());
  EXPECT_STREQ("o", softActors[1]->GetName());
  EXPECT_STREQ("c", softActors[2]->GetName());
  EXPECT_STREQ("h", softActors[3]->GetName());
  EXPECT_STREQ("i", softActors[4]->GetName());
}

TEST_IF(MOCHI_HDF5_AND_INTERNAL, Prefab, NestedPrefabs) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("my scene");
  MOCHI_DEFER(context->DestroyScene(scene));

  // JSON with two copies of a nested prefab, at different transforms
  std::string json = R"({
    "prefabs": [
      {
        "_comment": "Something about nested1",
        "name": "nested1",
        "path": "samples/soft_letters_mochi.mochi_scene",
        "scale": 0.5,
        "rotation": [0.0, 0.0, 0.0, 1.0],
        "translation": [0.0, 0.0, 1.0]
      },
      {
        "name": "nested2",
        "path": "samples/soft_letters_mochi.mochi_scene",
        "scale": 2.0,
        "rotation": [0.0, 0.0, 0.0, 1.0],
        "translation": [0.0, 0.0, -1.0]
      }
    ]
  })";

  auto scenePrefab = prefab::ShallowLoadFromJsonString(json, test::ExpectOK{});
  EXPECT_EQ(2, scenePrefab.prefabs.size());
  EXPECT_STREQ("Something about nested1", scenePrefab.prefabs[0].comment->c_str());
  EXPECT_STREQ("samples/soft_letters_mochi.mochi_scene", scenePrefab.prefabs[0].path.c_str());
  EXPECT_NEAR_EQ(0.5_r, scenePrefab.prefabs[0].scale);
  EXPECT_NEAR_EQ(Quaternion{}, scenePrefab.prefabs[0].rotation);
  EXPECT_NEAR_EQ(Real3(0_r, 0_r, 1_r), scenePrefab.prefabs[0].translation);
  EXPECT_FALSE(scenePrefab.prefabs[1].comment.has_value());
  EXPECT_STREQ("samples/soft_letters_mochi.mochi_scene", scenePrefab.prefabs[1].path.c_str());
  EXPECT_NEAR_EQ(2_r, scenePrefab.prefabs[1].scale);
  EXPECT_NEAR_EQ(Quaternion{}, scenePrefab.prefabs[1].rotation);
  EXPECT_NEAR_EQ(Real3(0_r, 0_r, -1_r), scenePrefab.prefabs[1].translation);

  // Load nested contents
  prefab::LoadNestedPrefabs(scenePrefab, test::GetAssetsDir(), test::ExpectOK{});
  prefab::LoadShapes(scenePrefab, test::GetAssetsDir(), context, test::ExpectOK{});
  // Add to scene — verify result includes actors from all nested prefabs
  auto const result = prefab::AddToScene(scenePrefab, scene, {}, test::ExpectOK{});

  // Each nested prefab has 6 actors
  EXPECT_EQ(scene->GetNumActors(), isize(result.actors));
  ASSERT_EQ(12, isize(result.actors));
  auto rigidActors = result.Filter(ActorType::Rigid);
  ASSERT_EQ(2, isize(rigidActors));
  EXPECT_STREQ("nested1/Ground", rigidActors[0]->GetName());
  EXPECT_STREQ("nested2/Ground", rigidActors[1]->GetName());
  auto softActors = result.Filter(ActorType::Soft);
  ASSERT_EQ(10, isize(softActors));
  EXPECT_STREQ("nested1/m", softActors[0]->GetName());
  EXPECT_STREQ("nested1/o", softActors[1]->GetName());
  EXPECT_STREQ("nested1/c", softActors[2]->GetName());
  EXPECT_STREQ("nested1/h", softActors[3]->GetName());
  EXPECT_STREQ("nested1/i", softActors[4]->GetName());
  EXPECT_STREQ("nested2/m", softActors[5]->GetName());
  EXPECT_STREQ("nested2/o", softActors[6]->GetName());
  EXPECT_STREQ("nested2/c", softActors[7]->GetName());
  EXPECT_STREQ("nested2/h", softActors[8]->GetName());
  EXPECT_STREQ("nested2/i", softActors[9]->GetName());

  // Expect the following actor names to exist
  EXPECT_NE(ActorHandle{}, FindActorByName(scene, "nested1/m"));
  EXPECT_NE(ActorHandle{}, FindActorByName(scene, "nested1/o"));
  EXPECT_NE(ActorHandle{}, FindActorByName(scene, "nested1/c"));
  EXPECT_NE(ActorHandle{}, FindActorByName(scene, "nested1/h"));
  EXPECT_NE(ActorHandle{}, FindActorByName(scene, "nested1/i"));
  EXPECT_NE(ActorHandle{}, FindActorByName(scene, "nested1/Ground"));
  EXPECT_NE(ActorHandle{}, FindActorByName(scene, "nested2/m"));
  EXPECT_NE(ActorHandle{}, FindActorByName(scene, "nested2/o"));
  EXPECT_NE(ActorHandle{}, FindActorByName(scene, "nested2/c"));
  EXPECT_NE(ActorHandle{}, FindActorByName(scene, "nested2/h"));
  EXPECT_NE(ActorHandle{}, FindActorByName(scene, "nested2/i"));
  EXPECT_NE(ActorHandle{}, FindActorByName(scene, "nested2/Ground"));
}

TEST(Prefab, LoadNestedPrefabs_ReloadsOnPathChange) {
  // Create two nested prefab files with different actor names
  auto tempDir = CreateTempDirectory("reload_nested_test", ExpectOK{});

  std::string nestedJsonA = R"({
    "actors": { "rigid": [{ "name": "ActorFromA" }] }
  })";
  std::string nestedJsonB = R"({
    "actors": { "rigid": [{ "name": "ActorFromB" }] }
  })";
  auto pathA = tempDir.Path() / "nestedA.mochi_scene";
  auto pathB = tempDir.Path() / "nestedB.mochi_scene";
  WriteFile(pathA, nestedJsonA, ExpectOK{});
  WriteFile(pathB, nestedJsonB, ExpectOK{});

  // Top-level prefab referencing nestedA
  std::string topJson = R"({
    "prefabs": [{ "name": "child", "path": "nestedA.mochi_scene" }]
  })";
  auto topPath = tempDir.Path() / "top.mochi_scene";
  WriteFile(topPath, topJson, ExpectOK{});

  // Load the top-level prefab and its nested prefab
  auto scenePrefab = prefab::ShallowLoadFromFile(topPath.string(), ExpectOK{});
  prefab::LoadNestedPrefabs(scenePrefab, tempDir.Path().string(), ExpectOK{});

  // Verify nested prefab A was loaded
  ASSERT_NE(nullptr, scenePrefab.prefabs[0].prefab);
  ASSERT_EQ(1, scenePrefab.prefabs[0].prefab->actors.rigid.size());
  EXPECT_STREQ("ActorFromA", scenePrefab.prefabs[0].prefab->actors.rigid[0].name.c_str());

  // Change the nested prefab path to nestedB
  scenePrefab.prefabs[0].path = "nestedB.mochi_scene";

  // Reload nested prefabs — should load from the new path
  prefab::LoadNestedPrefabs(scenePrefab, tempDir.Path().string(), ExpectOK{});

  // Verify nested prefab B was loaded
  ASSERT_NE(nullptr, scenePrefab.prefabs[0].prefab);
  ASSERT_EQ(1, scenePrefab.prefabs[0].prefab->actors.rigid.size());
  EXPECT_STREQ("ActorFromB", scenePrefab.prefabs[0].prefab->actors.rigid[0].name.c_str());
}

TEST(Prefab, LoadNestedPrefabs_PreservesPopulatedPathlessReference) {
  auto tempDir = CreateTempDirectory("pathless_nested_test", ExpectOK{});
  WriteFile(
      tempDir.Path() / "grandchild.mochi_scene",
      R"({"actors": {"rigid": [{"name": "Grandchild"}]}})",
      ExpectOK{});

  auto child = std::make_shared<prefab::ScenePrefab>();
  child->prefabs.push_back().path = "grandchild.mochi_scene";
  prefab::ScenePrefab root;
  root.prefabs.push_back().prefab = child;

  prefab::LoadNestedPrefabs(root, tempDir.Path().string(), ExpectOK{});

  EXPECT_EQ(child, root.prefabs[0].prefab);
  ASSERT_NE(nullptr, child->prefabs[0].prefab);
  ASSERT_EQ(1, child->prefabs[0].prefab->actors.rigid.size());
  EXPECT_STREQ("Grandchild", child->prefabs[0].prefab->actors.rigid[0].name.c_str());
}

TEST(Prefab, LoadNestedPrefabs_RejectsUnpopulatedPathlessReference) {
  prefab::ScenePrefab root;
  auto& reference = root.prefabs.push_back();

  Error error;
  {
    auto suppressError = test::SuppressLogError();
    prefab::LoadNestedPrefabs(root, test::GetAssetsDir(), error);
  }

  EXPECT_NOT_OK(error);
  EXPECT_NE(error.ToString().find("nested prefab reference"), std::string::npos);
  EXPECT_EQ(nullptr, reference.prefab);
}

TEST(Prefab, LoadNestedPrefabs_FailedReloadPreservesExistingReference) {
  auto tempDir = CreateTempDirectory("failed_nested_reload_test", ExpectOK{});
  auto existing = std::make_shared<prefab::ScenePrefab>();
  existing->actors.rigid.push_back().name = "Existing";

  prefab::ScenePrefab root;
  auto& reference = root.prefabs.push_back();
  reference.path = "missing.mochi_scene";
  reference.prefab = existing;

  Error missingReplacementError;
  {
    auto suppressError = test::SuppressLogError();
    prefab::LoadNestedPrefabs(root, tempDir.Path().string(), missingReplacementError);
  }
  EXPECT_NOT_OK(missingReplacementError);
  EXPECT_EQ(existing, reference.prefab);

  // The replacement itself loads, but its missing child prevents publication of the replacement.
  WriteFile(
      tempDir.Path() / "replacement.mochi_scene",
      R"({"prefabs": [{"path": "missing.mochi_scene"}]})",
      ExpectOK{});
  reference.path = "replacement.mochi_scene";
  Error missingDescendantError;
  {
    auto suppressError = test::SuppressLogError();
    prefab::LoadNestedPrefabs(root, tempDir.Path().string(), missingDescendantError);
  }
  EXPECT_NOT_OK(missingDescendantError);
  EXPECT_EQ(existing, reference.prefab);
  ASSERT_EQ(1, reference.prefab->actors.rigid.size());
  EXPECT_STREQ("Existing", reference.prefab->actors.rigid[0].name.c_str());
}

static void ExpectPrefabCycleError(Error const& error) {
  EXPECT_NOT_OK(error);
  EXPECT_NE(
      error.ToString().find("A prefab references itself, directly or indirectly."),
      std::string::npos);
}

static void ExpectNestedPrefabCycle(prefab::ScenePrefab& scenePrefab, std::string_view rootPath) {
  Error error;
  {
    auto suppressError = test::SuppressLogError();
    prefab::LoadNestedPrefabs(scenePrefab, rootPath, error);
  }
  ExpectPrefabCycleError(error);
}

static std::shared_ptr<prefab::ScenePrefab> MakePathlessSelfCyclePrefab() {
  auto scenePrefab = std::make_shared<prefab::ScenePrefab>();
  scenePrefab->prefabs.push_back().prefab = scenePrefab;
  return scenePrefab;
}

TEST(Prefab, LoadNestedPrefabs_DetectsCanonicalPathCycle) {
  auto tempDir = CreateTempDirectory("canonical_nested_cycle_test", ExpectOK{});
  std::filesystem::create_directories(tempDir.Path() / "sub");
  auto const prefabPath = tempDir.Path() / "self.mochi_scene";
  WriteFile(
      prefabPath,
      R"({"prefabs": [{"name": "self", "path": "./sub/../self.mochi_scene"}]})",
      ExpectOK{});

  auto scenePrefab = prefab::ShallowLoadFromFile(prefabPath.string(), ExpectOK{});
  ExpectNestedPrefabCycle(scenePrefab, tempDir.Path().string());
}

TEST(Prefab, LoadNestedPrefabs_SharedNestedPrefabIsNotACycle) {
  // Diamond reuse: top -> {childB, childC}, and both children reference the same leaf prefab. The
  // shared leaf appears on two independent branches, never twice on a single root-to-leaf path, so
  // cycle detection (a DFS stack of active paths) must NOT reject it.
  auto tempDir = CreateTempDirectory("diamond_nested_test", ExpectOK{});
  WriteFile(
      tempDir.Path() / "shared.mochi_scene",
      R"({"actors": {"rigid": [{"name": "Shared"}]}})",
      ExpectOK{});
  WriteFile(
      tempDir.Path() / "childB.mochi_scene",
      R"({"prefabs": [{"name": "b", "path": "shared.mochi_scene"}]})",
      ExpectOK{});
  WriteFile(
      tempDir.Path() / "childC.mochi_scene",
      R"({"prefabs": [{"name": "c", "path": "shared.mochi_scene"}]})",
      ExpectOK{});
  WriteFile(
      tempDir.Path() / "top.mochi_scene",
      R"({"prefabs": [{"name": "cb", "path": "childB.mochi_scene"}, {"name": "cc", "path": "childC.mochi_scene"}]})",
      ExpectOK{});

  auto scenePrefab =
      prefab::ShallowLoadFromFile((tempDir.Path() / "top.mochi_scene").string(), ExpectOK{});
  // Must load without a spurious "cycle" error (ExpectOK asserts success).
  prefab::LoadNestedPrefabs(scenePrefab, tempDir.Path().string(), ExpectOK{});

  // ...and the shared leaf must actually be loaded via BOTH branches, not silently skipped.
  for (auto const& child : scenePrefab.prefabs) {
    ASSERT_NE(nullptr, child.prefab);
    ASSERT_NE(nullptr, child.prefab->prefabs[0].prefab);
    EXPECT_STREQ("Shared", child.prefab->prefabs[0].prefab->actors.rigid[0].name.c_str());
  }
}

TEST(Prefab, LoadNestedPrefabs_DetectsMutualCycle) {
  // a -> b -> a: a cycle spanning two distinct files must be detected (and terminate), not just a
  // direct self-reference.
  auto tempDir = CreateTempDirectory("mutual_nested_cycle_test", ExpectOK{});
  WriteFile(
      tempDir.Path() / "a.mochi_scene",
      R"({"prefabs": [{"name": "b", "path": "b.mochi_scene"}]})",
      ExpectOK{});
  WriteFile(
      tempDir.Path() / "b.mochi_scene",
      R"({"prefabs": [{"name": "a", "path": "a.mochi_scene"}]})",
      ExpectOK{});

  auto scenePrefab =
      prefab::ShallowLoadFromFile((tempDir.Path() / "a.mochi_scene").string(), ExpectOK{});
  ExpectNestedPrefabCycle(scenePrefab, tempDir.Path().string());
}

TEST(Prefab, EnsureFullyLoaded_InMemoryNestedTreeIsNotACycle) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto tempDir = CreateTempDirectory("in_memory_nested_tree_test", ExpectOK{});
  WriteFile(tempDir.Path() / "loaded.mochi_scene", R"({})", ExpectOK{});

  // Depth-2 tree of in-memory (pathless) prefabs. Their resolved paths all collapse to rootPath, so
  // a cycle guard keyed on the resolved path would falsely reject the inner reference; gating on
  // the reference's own (empty) path treats these as non-file references and skips path-based cycle
  // tracking.
  // Runs through EnsureFullyLoaded (skipLoaded=true), the path where that false positive would
  // surface. It also verifies that both pathless handles are retained while their file-backed
  // descendant is loaded.
  auto grandchild = std::make_shared<prefab::ScenePrefab>();
  grandchild->prefabs.push_back().path = "loaded.mochi_scene";
  auto child = std::make_shared<prefab::ScenePrefab>();
  child->prefabs.push_back().prefab = grandchild;
  prefab::ScenePrefab root;
  root.prefabs.push_back().prefab = child;

  prefab::EnsureFullyLoaded(root, tempDir.Path().string(), context, test::ExpectOK{});

  EXPECT_EQ(child, root.prefabs[0].prefab);
  EXPECT_EQ(grandchild, child->prefabs[0].prefab);
  EXPECT_NE(nullptr, grandchild->prefabs[0].prefab);
}

TEST(Prefab, SharedInMemoryNestedPrefabIsNotACycle) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("my scene");
  MOCHI_DEFER(context->DestroyScene(scene));

  auto shared = std::make_shared<prefab::ScenePrefab>();
  prefab::ScenePrefab root;
  root.prefabs.push_back().prefab = shared;
  root.prefabs.push_back().prefab = shared;

  prefab::EnsureFullyLoaded(root, test::GetAssetsDir(), context, test::ExpectOK{});
  prefab::LoadShapes(root, test::GetAssetsDir(), context, test::ExpectOK{});
  auto const result = prefab::AddToScene(root, scene, {}, test::ExpectOK{});
  EXPECT_TRUE(result.actors.empty());
  EXPECT_TRUE(result.constraints.empty());
}

TEST(Prefab, PathlessInMemoryNestedCycle_DetectedByPublicTraversals) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("my scene");
  MOCHI_DEFER(context->DestroyScene(scene));

  {
    auto scenePrefab = MakePathlessSelfCyclePrefab();
    MOCHI_DEFER(scenePrefab->prefabs.clear());
    Error error;
    auto suppressError = test::SuppressLogError();
    prefab::LoadNestedPrefabs(*scenePrefab, test::GetAssetsDir(), error);
    ExpectPrefabCycleError(error);
  }

  {
    auto scenePrefab = MakePathlessSelfCyclePrefab();
    MOCHI_DEFER(scenePrefab->prefabs.clear());
    Error error;
    auto suppressError = test::SuppressLogError();
    prefab::EnsureFullyLoaded(*scenePrefab, test::GetAssetsDir(), context, error);
    ExpectPrefabCycleError(error);
  }
  {
    auto scenePrefab = MakePathlessSelfCyclePrefab();
    MOCHI_DEFER(scenePrefab->prefabs.clear());
    Error error;
    auto suppressError = test::SuppressLogError();
    prefab::LoadShapes(*scenePrefab, test::GetAssetsDir(), context, error);
    ExpectPrefabCycleError(error);
  }
  {
    auto scenePrefab = MakePathlessSelfCyclePrefab();
    MOCHI_DEFER(scenePrefab->prefabs.clear());
    Error error;
    auto suppressError = test::SuppressLogError();
    prefab::AddToScene(*scenePrefab, scene, {}, error);
    ExpectPrefabCycleError(error);
  }
}

TEST(Prefab, EnsureFullyLoaded_DetectsFileCycle) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));

  // Cycle detection runs in the shared LoadNestedPrefabsImpl before the skipLoaded guard, so a
  // file-backed a -> b -> a cycle must also be rejected through EnsureFullyLoaded
  // (skipLoaded=true), not just through LoadNestedPrefabs (skipLoaded=false).
  auto tempDir = CreateTempDirectory("ensure_fully_loaded_cycle_test", ExpectOK{});
  WriteFile(
      tempDir.Path() / "a.mochi_scene",
      R"({"prefabs": [{"name": "b", "path": "b.mochi_scene"}]})",
      ExpectOK{});
  WriteFile(
      tempDir.Path() / "b.mochi_scene",
      R"({"prefabs": [{"name": "a", "path": "a.mochi_scene"}]})",
      ExpectOK{});

  auto scenePrefab =
      prefab::ShallowLoadFromFile((tempDir.Path() / "a.mochi_scene").string(), ExpectOK{});
  Error error;
  {
    auto suppressError = test::SuppressLogError();
    prefab::EnsureFullyLoaded(scenePrefab, tempDir.Path().string(), context, error);
  }
  ExpectPrefabCycleError(error);
}

// The prefab assets used by the tests below are not shipped externally.
TEST_IF(MOCHI_INTERNAL, Prefab, LoadShapes_ReloadsOnPathChange) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  EXPECT_EQ(0, context->GetNumShapes());

  // Load a prefab with a rigid actor shape
  std::string json = R"({
    "actors": {
      "rigid": [{
        "name": "box",
        "shape": "cube/cube_minimal.mochi.json"
      }]
    }
  })";
  auto scenePrefab =
      prefab::LoadFromJsonString(json, test::GetAssetPath(""), context, test::ExpectOK{});
  EXPECT_EQ(1, context->GetNumShapes());
  auto originalShape = scenePrefab.actors.rigid[0].shape;
  EXPECT_TRUE(originalShape.IsValid());

  // Make a copy of the prefab (holds a reference to the original shape handle)
  auto prefabCopy = scenePrefab;

  // Change the shape file path in the first prefab to a different asset
  scenePrefab.actors.rigid[0].shapeFile = "duck/duck_coarse_mesh.mochi.json";

  // Reload shapes — should load a new shape from the new path
  prefab::LoadShapes(scenePrefab, test::GetAssetPath(""), context, test::ExpectOK{});

  // The new shape should be different from the original
  auto newShape = scenePrefab.actors.rigid[0].shape;
  EXPECT_TRUE(newShape.IsValid());
  EXPECT_NE(originalShape, newShape);

  // Both the old shape (held by the copy) and the new shape should exist
  EXPECT_EQ(2, context->GetNumShapes());

  // Release the copy's shape and verify the count drops
  context->ReleaseShape(prefabCopy.actors.rigid[0].shape);
  EXPECT_EQ(1, context->GetNumShapes());

  // Release the original prefab's shape and verify the count drops to zero
  context->ReleaseShape(scenePrefab.actors.rigid[0].shape);
  EXPECT_EQ(0, context->GetNumShapes());
}

TEST_IF(MOCHI_INTERNAL, Prefab, LoadShapes_ReloadsOnScaleChange) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));

  // Load a prefab with a rigid actor and unit scale
  std::string json = R"({
    "actors": {
      "rigid": [{
        "name": "box",
        "shape": "cube/cube_minimal.mochi.json",
        "scale": [1, 1, 1]
      }]
    }
  })";
  auto scenePrefab =
      prefab::LoadFromJsonString(json, test::GetAssetPath(""), context, test::ExpectOK{});
  EXPECT_EQ(1, context->GetNumShapes());

  // Verify the AABB of the original shape (unit cube: [0,1]^3)
  auto originalShape = scenePrefab.actors.rigid[0].shape;
  auto originalBounds = context->GetShapeAabb(originalShape, test::ExpectOK{});
  EXPECT_NEAR_EQ(Real3(0_r, 0_r, 0_r), originalBounds.GetMin());
  EXPECT_NEAR_EQ(Real3(1_r, 1_r, 1_r), originalBounds.GetMax());

  // Change the baked-in scale and reload shapes
  scenePrefab.actors.rigid[0].scale = Real3{2_r, 3_r, 4_r};
  prefab::LoadShapes(scenePrefab, test::GetAssetPath(""), context, test::ExpectOK{});

  // The reloaded shape should be different (new baked-in scale)
  auto scaledShape = scenePrefab.actors.rigid[0].shape;
  EXPECT_NE(originalShape, scaledShape);
  EXPECT_EQ(2, context->GetNumShapes());

  // Verify the scaled shape AABB reflects the new scale
  auto scaledBounds = context->GetShapeAabb(scaledShape, test::ExpectOK{});
  EXPECT_NEAR_EQ(Real3(0_r, 0_r, 0_r), scaledBounds.GetMin());
  EXPECT_NEAR_EQ(Real3(2_r, 3_r, 4_r), scaledBounds.GetMax());

  // Cleanup
  context->ReleaseShape(originalShape);
  context->ReleaseShape(scaledShape);
  EXPECT_EQ(0, context->GetNumShapes());
}

TEST_IF(MOCHI_INTERNAL, Prefab, EnsureFullyLoaded) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));

  // Disable file caching so that each load creates a distinct shape handle
  bool const wasCacheEnabled = context->IsFileCacheEnabled();
  context->EnableFileCache(false);
  MOCHI_DEFER(context->EnableFileCache(wasCacheEnabled));

  EXPECT_EQ(0, context->GetNumShapes());

  // Create a nested prefab file with one rigid actor (duck shape)
  auto tempDir = CreateTempDirectory("ensure_fully_loaded_test", ExpectOK{});
  std::string childJson = R"({
    "actors": {
      "rigid": [{
        "name": "ChildActor",
        "shape": "duck/duck_coarse_mesh.mochi.json"
      }]
    }
  })";
  auto childPath = tempDir.Path() / "child.mochi_scene";
  WriteFile(childPath, childJson, ExpectOK{});

  // Create a parent prefab with one rigid actor (cube shape) and a nested reference to the child
  std::string parentJson = R"({
    "actors": {
      "rigid": [{
        "name": "ParentActor",
        "shape": "cube/cube_minimal.mochi.json"
      }]
    },
    "prefabs": [{
      "name": "child",
      "path": "./child.mochi_scene"
    }]
  })";
  auto parentPath = tempDir.Path() / "parent.mochi_scene";
  WriteFile(parentPath, parentJson, ExpectOK{});

  // Shallow load — no nested prefabs or shapes loaded yet
  auto scenePrefab = prefab::ShallowLoadFromFile(parentPath.string(), ExpectOK{});
  EXPECT_EQ(1, scenePrefab.actors.rigid.size());
  EXPECT_EQ(1, scenePrefab.prefabs.size());
  EXPECT_EQ(nullptr, scenePrefab.prefabs[0].prefab);
  EXPECT_FALSE(scenePrefab.actors.rigid[0].shape.IsValid());
  EXPECT_EQ(0, context->GetNumShapes());

  // EnsureFullyLoaded should load the nested prefab and both shapes.
  // Shape paths are relative to the assets directory.
  prefab::EnsureFullyLoaded(scenePrefab, test::GetAssetsDir(), context, ExpectOK{});

  // Verify nested prefab was loaded
  ASSERT_NE(nullptr, scenePrefab.prefabs[0].prefab);
  ASSERT_EQ(1, scenePrefab.prefabs[0].prefab->actors.rigid.size());
  EXPECT_STREQ("ChildActor", scenePrefab.prefabs[0].prefab->actors.rigid[0].name.c_str());

  // Verify both shapes were loaded (parent + child)
  EXPECT_TRUE(scenePrefab.actors.rigid[0].shape.IsValid());
  EXPECT_TRUE(scenePrefab.prefabs[0].prefab->actors.rigid[0].shape.IsValid());
  EXPECT_EQ(2, context->GetNumShapes());

  // Make a copy and call EnsureFullyLoaded again — nothing should change
  auto prefabCopy = scenePrefab;
  prefab::EnsureFullyLoaded(scenePrefab, test::GetAssetsDir(), context, ExpectOK{});

  // The shape count should not have increased (no redundant loads)
  EXPECT_EQ(2, context->GetNumShapes());

  // Verify the handles are unchanged
  EXPECT_EQ(prefabCopy.actors.rigid[0].shape, scenePrefab.actors.rigid[0].shape);
  EXPECT_EQ(
      prefabCopy.prefabs[0].prefab->actors.rigid[0].shape,
      scenePrefab.prefabs[0].prefab->actors.rigid[0].shape);
}

TEST(Prefab, SaveToJsonFile) {
  // A prefab with some non-default data.
  prefab::ScenePrefab srcPrefab;
  srcPrefab.actors.rigid.push_back().name = "Actor A";
  srcPrefab.actors.rigid.push_back().name = "Actor B";

  // Serialize as a string (a feature covered more thoroughly in other test cases).
  auto srcPrefabAsString = prefab::SaveToJsonString(srcPrefab, test::ExpectOK{});

  // Now use prefab::SaveToJsonFile to write a file in a temporary directory. Any missing
  // directories in the path should be created automatically.
  auto tempDir = CreateTempDirectory("prefab_save_load_test", test::ExpectOK{});
  auto tempFilePath = tempDir.Path() / "new_subdir" / "my_prefab.any_extension";
  EXPECT_FALSE(std::filesystem::exists(tempFilePath));
  prefab::SaveToJsonFile(srcPrefab, tempFilePath.string(), test::ExpectOK{});
  EXPECT_TRUE(std::filesystem::exists(tempFilePath));

  // Read the temp file back as a string. Expect the same results as prefab::SaveToJsonString.
  auto reloadedPrefabAsString = ReadFileString(tempFilePath, test::ExpectOK{});
  EXPECT_STREQ(srcPrefabAsString.c_str(), reloadedPrefabAsString.c_str());

  // Read the temp file back as a ScenePrefab
  auto reloadedPrefab = prefab::ShallowLoadFromFile(tempFilePath.string(), test::ExpectOK{});

  // Check it
  EXPECT_EQ(2, reloadedPrefab.actors.rigid.size());
  EXPECT_STREQ("Actor A", reloadedPrefab.actors.rigid[0].name.c_str());
  EXPECT_STREQ("Actor B", reloadedPrefab.actors.rigid[1].name.c_str());
}

// These tests are for loading nested prefabs with mixed asset path resolution styles in even and
// odd ordering. Test 1 Prefab hierarchy: A -> B -> C (all prefabs in same directory) Shape paths:
//   A.shape: "cube/cube_minimal.mochi.json" (asset-dir-relative)
//   B.shape: "./shapes/B.mochi.json" (prefab-relative)
//   C.shape: "cube/cube_minimal.mochi.json" (asset-dir-relative)
TEST_IF(MOCHI_HDF5_AND_INTERNAL, Prefab, NestedPrefabs_MixedPathStyles1) {
  auto tempDir = CreateTempDirectory("nested_prefab_mixed_paths_test_1", ExpectOK{});
  auto prefabsDir = tempDir.Path() / "prefabs";

  // Setup: Copy cube mesh for prefab-relative path (B's shape)
  auto shapesDir = prefabsDir / "shapes";
  std::filesystem::create_directories(shapesDir);
  std::filesystem::copy_file(
      GetAssetPath("cube/cube_minimal.mochi.json"), shapesDir / "B.mochi.json");

  // Create test prefabs with mixed path styles
  WriteFile(
      prefabsDir / "C.mochi_scene",
      R"({
    "actors": { "rigid": [{ "name": "C", "shape": "cube/cube_minimal.mochi.json", "colliderType": "Box", "isStatic": true }] }
  })",
      ExpectOK{});

  WriteFile(
      prefabsDir / "B.mochi_scene",
      R"({
    "actors": { "rigid": [{ "name": "B", "shape": "./shapes/B.mochi.json", "colliderType": "Box", "isStatic": true }] },
    "prefabs": [{ "name": "c", "path": "./C.mochi_scene" }]
  })",
      ExpectOK{});

  WriteFile(
      prefabsDir / "A.mochi_scene",
      R"({
    "actors": { "rigid": [{ "name": "A", "shape": "cube/cube_minimal.mochi.json", "colliderType": "Box", "isStatic": true }] },
    "prefabs": [{ "name": "b", "path": "./B.mochi_scene" }]
  })",
      ExpectOK{});

  // Load and verify all actors are created correctly
  auto* context = CreateContext(0);
  MOCHI_DEFER(DestroyContext(context));
  auto* scene = context->CreateScene("test");
  MOCHI_DEFER(context->DestroyScene(scene));

  prefab::AddToScene(
      (prefabsDir / "A.mochi_scene").string(), GetAssetsDir(), scene, {}, ExpectOK{});

  EXPECT_EQ(3, scene->GetNumActors());

  EXPECT_NE(ActorHandle{}, FindActorByName(scene, "A"));
  EXPECT_NE(ActorHandle{}, FindActorByName(scene, "b/B"));
  EXPECT_NE(ActorHandle{}, FindActorByName(scene, "b/c/C"));
}

// Test 2
// Prefab hierarchy: A -> B -> C
// Shape paths:
//   A.shape: "./shapes/A.mochi.json" (prefab-relative)
//   B.shape: "cube/cube_minimal.mochi.json" (asset-dir-relative)
//   C.shape: "./shapes/C.mochi.json" (prefab-relative)
TEST_IF(MOCHI_HDF5_AND_INTERNAL, Prefab, NestedPrefabs_MixedPathStyles2) {
  auto tempDir = CreateTempDirectory("nested_prefab_mixed_path_styles_2", ExpectOK{});
  auto prefabsDir = tempDir.Path() / "prefabs";
  auto shapesDir = prefabsDir / "shapes";

  // Setup: Copy cube mesh for prefab-relative paths (A's and C's shapes)
  std::filesystem::create_directories(shapesDir);
  auto cubeMesh = GetAssetPath("cube/cube_minimal.mochi.json");
  std::filesystem::copy_file(cubeMesh, shapesDir / "A.mochi.json");
  std::filesystem::copy_file(cubeMesh, shapesDir / "C.mochi.json");

  // Create test prefabs with mixed path styles
  WriteFile(
      prefabsDir / "C.mochi_scene",
      R"({
    "actors": { "rigid": [{ "name": "C", "shape": "./shapes/C.mochi.json", "colliderType": "Box", "isStatic": true }] }
  })",
      ExpectOK{});

  WriteFile(
      prefabsDir / "B.mochi_scene",
      R"({
    "actors": { "rigid": [{ "name": "B", "shape": "cube/cube_minimal.mochi.json", "colliderType": "Box", "isStatic": true }] },
    "prefabs": [{ "name": "c", "path": "./C.mochi_scene" }]
  })",
      ExpectOK{});

  WriteFile(
      prefabsDir / "A.mochi_scene",
      R"({
    "actors": { "rigid": [{ "name": "A", "shape": "./shapes/A.mochi.json", "colliderType": "Box", "isStatic": true }] },
    "prefabs": [{ "name": "b", "path": "./B.mochi_scene" }]
  })",
      ExpectOK{});

  // Load and verify all actors are created correctly
  auto* context = CreateContext(0);
  MOCHI_DEFER(DestroyContext(context));
  auto* scene = context->CreateScene("test");
  MOCHI_DEFER(context->DestroyScene(scene));

  prefab::AddToScene(
      (prefabsDir / "A.mochi_scene").string(), GetAssetsDir(), scene, {}, ExpectOK{});

  EXPECT_EQ(3, scene->GetNumActors());

  // Verify actor hierarchy
  EXPECT_NE(ActorHandle{}, FindActorByName(scene, "A"));
  EXPECT_NE(ActorHandle{}, FindActorByName(scene, "b/B"));
  EXPECT_NE(ActorHandle{}, FindActorByName(scene, "b/c/C"));
}

TEST(Prefab, ContactFilter_Serialization) {
  prefab::ActorContactEntry const positionalEntry{false, {"A", "B"}};
  EXPECT_TRUE(positionalEntry.includeNestedActors);

  std::string json = R"({
      "contactFilter": {
        "actorContactSymmetric": [
          {"enable": true, "includeNestedActors": true, "actors": ["A", "B"]}
        ],
      "layerContactSymmetric": [
        {"enable": false, "layers": ["LayerA", "LayerB"]}
      ],
      "actorContactAsymmetric": [
        {"enable": false, "includeNestedActors": false, "actors": ["ActorA", "ActorB"]}
      ],
      "layerContactAsymmetric": [
        {"enable": false, "layers": ["LayerC", "LayerD"]}
      ]
    }
  })";

  auto expectContactFilter = [](prefab::ContactFilter const& filter) {
    ASSERT_TRUE(filter.actorContactSymmetric.has_value());
    ASSERT_EQ(1, filter.actorContactSymmetric->size());
    EXPECT_TRUE((*filter.actorContactSymmetric)[0].enable);
    EXPECT_TRUE((*filter.actorContactSymmetric)[0].includeNestedActors);
    ASSERT_EQ(2, (*filter.actorContactSymmetric)[0].actors.size());
    EXPECT_STREQ("A", (*filter.actorContactSymmetric)[0].actors[0].c_str());
    EXPECT_STREQ("B", (*filter.actorContactSymmetric)[0].actors[1].c_str());

    ASSERT_TRUE(filter.layerContactSymmetric.has_value());
    ASSERT_EQ(1, filter.layerContactSymmetric->size());
    EXPECT_FALSE((*filter.layerContactSymmetric)[0].enable);
    ASSERT_EQ(2, (*filter.layerContactSymmetric)[0].layers.size());
    EXPECT_STREQ("LayerA", (*filter.layerContactSymmetric)[0].layers[0].c_str());
    EXPECT_STREQ("LayerB", (*filter.layerContactSymmetric)[0].layers[1].c_str());

    ASSERT_TRUE(filter.actorContactAsymmetric.has_value());
    ASSERT_EQ(1, filter.actorContactAsymmetric->size());
    EXPECT_FALSE((*filter.actorContactAsymmetric)[0].enable);
    EXPECT_FALSE((*filter.actorContactAsymmetric)[0].includeNestedActors);
    ASSERT_EQ(2, (*filter.actorContactAsymmetric)[0].actors.size());
    EXPECT_STREQ("ActorA", (*filter.actorContactAsymmetric)[0].actors[0].c_str());
    EXPECT_STREQ("ActorB", (*filter.actorContactAsymmetric)[0].actors[1].c_str());

    ASSERT_TRUE(filter.layerContactAsymmetric.has_value());
    ASSERT_EQ(1, filter.layerContactAsymmetric->size());
    EXPECT_FALSE((*filter.layerContactAsymmetric)[0].enable);
    ASSERT_EQ(2, (*filter.layerContactAsymmetric)[0].layers.size());
    EXPECT_STREQ("LayerC", (*filter.layerContactAsymmetric)[0].layers[0].c_str());
    EXPECT_STREQ("LayerD", (*filter.layerContactAsymmetric)[0].layers[1].c_str());
  };

  prefab::ScenePrefab content = prefab::ShallowLoadFromJsonString(json, test::ExpectOK{});
  EXPECT_TRUE(content.contactFilter.has_value());
  expectContactFilter(*content.contactFilter);

  // Test round-trip serialization
  auto json2 = prefab::SaveToJsonString(content, test::ExpectOK{});
  auto const includeNestedActorsPos = json2.find("\"includeNestedActors\"");
  ASSERT_NE(std::string::npos, includeNestedActorsPos);
  EXPECT_EQ(std::string::npos, json2.find("\"includeNestedActors\"", includeNestedActorsPos + 1));
  EXPECT_NE(std::string::npos, json2.find("\"includeNestedActors\": false"));
  EXPECT_NE(std::string::npos, json2.find("\"enable\": true"));
  prefab::ScenePrefab content2 = prefab::ShallowLoadFromJsonString(json2, test::ExpectOK{});
  EXPECT_TRUE(content2.contactFilter.has_value());
  expectContactFilter(*content2.contactFilter);
}

// The prefab assets used by the tests below are not shipped externally.
TEST_IF(MOCHI_INTERNAL, Prefab, ContactFilter_SymmetricSelfPairAppliesToResolvedActor) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("test");
  MOCHI_DEFER(context->DestroyScene(scene));

  auto scenePrefab = prefab::ShallowLoadFromJsonString(
      R"({
        "actors": {
          "rigid": [
            {"name": "Cube", "colliderType": "Box", "shape": "cube/cube_minimal.mochi.json"}
          ]
        },
        "contactFilter": {
          "actorContactSymmetric": [
            {"enable": false, "actors": ["Cube", "Cube"]}
          ]
        }
      })",
      test::ExpectOK{});
  prefab::LoadShapes(scenePrefab, test::GetAssetsDir(), context, test::ExpectOK{});
  prefab::AddToScene(scenePrefab, scene, {}, test::ExpectOK{});

  auto const cube = FindActorByName(scene, "Cube");
  auto const* sceneImpl = assert_cast<SceneImpl const*>(scene);
  EXPECT_FALSE(sceneImpl->IsActorContactEnabled(cube, cube, test::ExpectOK{}));
}

TEST(Prefab, ContactFilter_InvalidActorArraySize) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("test");
  MOCHI_DEFER(context->DestroyScene(scene));

  // Test with 1 actor (too few)
  {
    prefab::ScenePrefab scenePrefab;
    auto& entry = scenePrefab.contactFilter.emplace().actorContactSymmetric.emplace().push_back();
    entry.actors.push_back("A");
    prefab::AddToScene(scenePrefab, scene, {}, test::ExpectNotOK{});
  }

  // Test with 3 actors (too many)
  {
    prefab::ScenePrefab scenePrefab;
    auto& entry = scenePrefab.contactFilter.emplace().actorContactAsymmetric.emplace().push_back();
    entry.actors.push_back("A");
    entry.actors.push_back("B");
    entry.actors.push_back("C");
    prefab::AddToScene(scenePrefab, scene, {}, test::ExpectNotOK{});
  }
}

TEST_IF(MOCHI_INTERNAL, Prefab, ContactFilter_InvalidLayerArraySize) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));

  auto tempDir = CreateTempDirectory("prefab_contact_filter_invalid_layer_test", test::ExpectOK{});
  std::string meshJson;
  ReadFile(test::GetAssetPath("cube/cube_minimal.mochi.json"), meshJson, test::ExpectOK{});
  WriteFile(tempDir.Path() / "cube.mochi.json", meshJson, test::ExpectOK{});

  // Test with 0 layers (empty)
  {
    std::string json = R"({
      "actors": { "rigid": [{"name": "A", "layer": "L1", "colliderType": "Box", "shape": "cube.mochi.json"}] },
      "contactFilter": { "layerContactSymmetric": [{"enable": false, "layers": []}] }
    })";
    auto prefabPath = tempDir.Path() / "invalid_layer1.prefab";
    WriteFile(prefabPath, json, test::ExpectOK{});

    Scene* scene = context->CreateScene("test");
    MOCHI_DEFER(context->DestroyScene(scene));
    prefab::AddToScene(
        prefabPath.string(), tempDir.Path().string(), scene, {}, test::ExpectNotOK{});
  }

  // Test with 1 layer (too few)
  {
    std::string json = R"({
      "actors": { "rigid": [{"name": "A", "layer": "L1", "colliderType": "Box", "shape": "cube.mochi.json"}] },
      "contactFilter": { "layerContactAsymmetric": [{"enable": false, "layers": ["L1"]}] }
    })";
    auto prefabPath = tempDir.Path() / "invalid_layer2.prefab";
    WriteFile(prefabPath, json, test::ExpectOK{});

    Scene* scene = context->CreateScene("test");
    MOCHI_DEFER(context->DestroyScene(scene));
    prefab::AddToScene(
        prefabPath.string(), tempDir.Path().string(), scene, {}, test::ExpectNotOK{});
  }
}

TEST_IF(MOCHI_INTERNAL, Prefab, ContactFilter_AddToScene) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));

  // Create a temporary directory and copy one mesh file into it.
  // This will serve as the assets directory for this test
  auto tempDir = CreateTempDirectory("prefab_contact_filter_test", test::ExpectOK{});
  std::string meshJson;
  ReadFile(test::GetAssetPath("cube/cube_minimal.mochi.json"), meshJson, test::ExpectOK{});
  WriteFile(tempDir.Path() / "cube.mochi.json", meshJson, test::ExpectOK{});

  // Create a prefab with actors in different layers and contact filters
  std::string json1 = R"({
    "actors": {
      "rigid": [
        {
          "name": "Floor",
          "layer": "LayerA",
          "isStatic": true,
          "colliderType": "Box",
          "shape": "cube.mochi.json"
        },
        {
          "name": "CubeA",
          "layer": "LayerB",
          "colliderType": "Box",
          "shape": "cube.mochi.json"
        },
        {
          "name": "CubeB",
          "layer": "LayerC",
          "colliderType": "Box",
          "shape": "cube.mochi.json"
        },
        {
          "name": "CubeC",
          "layer": "LayerD",
          "colliderType": "Box",
          "shape": "cube.mochi.json"
        }
      ]
    },
    "contactFilter": {
      "actorContactSymmetric": [
        {
          "enable": false,
          "actors": ["CubeA", "CubeB"]
        }
      ],
      "actorContactAsymmetric": [
        {
          "enable": false,
          "actors": ["CubeC", "Floor"]
        }
      ],
      "layerContactSymmetric": [
        {
          "enable": false,
          "layers": ["LayerB", "LayerC"]
        }
      ],
      "layerContactAsymmetric": [
        {
          "enable": false,
          "layers": ["LayerD", "LayerA"]
        }
      ]
    }
  })";
  auto prefabPath1 = tempDir.Path() / "test1.prefab"; // extension doesn't matter here
  WriteFile(prefabPath1, json1, test::ExpectOK{});

  // Now, create a second prefab file. It will also have an actors called "CubeA" and "CubeB". The
  // first prefab will be nested inside of it, as well.
  std::string json2 = R"({
    "actors": {
      "rigid": [
        {
          "name": "CubeA",
          "colliderType": "Box",
          "shape": "cube.mochi.json"
        },
        {
          "name": "CubeB",
          "colliderType": "Box",
          "shape": "cube.mochi.json"
        }
      ]
    },
    "prefabs": [
      {
        "name": "MyNestedPrefab",
        "path": "test1.prefab"
      }
    ]
  })";
  auto prefabPath2 = tempDir.Path() / "test2.prefab";
  WriteFile(prefabPath2, json2, test::ExpectOK{});

  // Helper
  auto checkContactFiltering = [](SceneImpl const* scene, std::string const& namePrefix) {
    // Find our actors
    ActorHandle cubeA = FindActorByName(scene, namePrefix + "CubeA");
    ActorHandle cubeB = FindActorByName(scene, namePrefix + "CubeB");
    ActorHandle cubeC = FindActorByName(scene, namePrefix + "CubeC");
    ActorHandle floor = FindActorByName(scene, namePrefix + "Floor");

    // Check actor-vs-actor contact filters
    EXPECT_FALSE(scene->IsActorContactEnabled(cubeA, cubeB, test::ExpectOK{}));
    EXPECT_TRUE(scene->IsActorContactEnabled(cubeA, cubeC, test::ExpectOK{}));
    EXPECT_TRUE(scene->IsActorContactEnabled(cubeA, floor, test::ExpectOK{}));
    EXPECT_FALSE(scene->IsActorContactEnabled(cubeB, cubeA, test::ExpectOK{}));
    EXPECT_TRUE(scene->IsActorContactEnabled(cubeB, cubeB, test::ExpectOK{}));
    EXPECT_TRUE(scene->IsActorContactEnabled(cubeB, cubeC, test::ExpectOK{}));
    EXPECT_TRUE(scene->IsActorContactEnabled(cubeB, floor, test::ExpectOK{}));
    EXPECT_TRUE(scene->IsActorContactEnabled(cubeC, cubeA, test::ExpectOK{}));
    EXPECT_TRUE(scene->IsActorContactEnabled(cubeC, cubeB, test::ExpectOK{}));
    EXPECT_TRUE(scene->IsActorContactEnabled(cubeC, cubeC, test::ExpectOK{}));
    EXPECT_FALSE(scene->IsActorContactEnabled(cubeC, floor, test::ExpectOK{}));
    EXPECT_TRUE(scene->IsActorContactEnabled(floor, cubeA, test::ExpectOK{}));
    EXPECT_TRUE(scene->IsActorContactEnabled(floor, cubeB, test::ExpectOK{}));
    EXPECT_TRUE(scene->IsActorContactEnabled(floor, cubeC, test::ExpectOK{}));
    EXPECT_TRUE(scene->IsActorContactEnabled(floor, floor, test::ExpectOK{}));

    // Verify layer contact filters were applied correctly
    // Symmetric: LayerB-LayerC should be disabled in both directions
    EXPECT_FALSE(scene->IsLayerContactEnabled("LayerB", "LayerC"));
    EXPECT_FALSE(scene->IsLayerContactEnabled("LayerC", "LayerB"));

    // Asymmetric: LayerD->LayerA should be disabled, but LayerA->LayerD should still be enabled
    EXPECT_FALSE(scene->IsLayerContactEnabled("LayerD", "LayerA"));
    EXPECT_TRUE(scene->IsLayerContactEnabled("LayerA", "LayerD"));

    // Other layer pairs should have default contact enabled
    EXPECT_TRUE(scene->IsLayerContactEnabled("LayerA", "LayerB"));
    EXPECT_TRUE(scene->IsLayerContactEnabled("LayerA", "LayerC"));
  };

  // Add the first prefab to an empty scene. Preserve actor names exactly as they appear in the
  // prefab.
  {
    Scene* scene = context->CreateScene("test");
    MOCHI_DEFER(context->DestroyScene(scene));
    prefab::AddToScene(prefabPath1.string(), tempDir.Path().string(), scene, {}, test::ExpectOK{});
    checkContactFiltering(assert_cast<SceneImpl const*>(scene), "");
  }

  // Add the first prefab to an empty scene. This time, give the prefab instance a name. This name
  // will be prefixed on all actor names. That should not affect references within the prefab.
  {
    Scene* scene = context->CreateScene("test");
    MOCHI_DEFER(context->DestroyScene(scene));
    prefab::AddToScene(
        prefabPath1.string(),
        tempDir.Path().string(),
        scene,
        prefab::PrefabParams{.name = "MyPrefab"},
        test::ExpectOK{});
    checkContactFiltering(assert_cast<SceneImpl const*>(scene), "MyPrefab/");
  }

  // Add the second prefab to an empty scene. This time, the prefab with contact filtering will be
  // nested within another prefab. That should not affect references within the prefab.
  {
    Scene* scene = context->CreateScene("test");
    auto const* sceneImpl = assert_cast<SceneImpl const*>(scene);
    MOCHI_DEFER(context->DestroyScene(scene));
    prefab::AddToScene(
        prefabPath2.string(),
        tempDir.Path().string(),
        scene,
        prefab::PrefabParams{.name = "MyNewPrefab"},
        test::ExpectOK{});
    checkContactFiltering(sceneImpl, "MyNewPrefab/MyNestedPrefab/");

    // Actor-vs-actor filtering in the nested prefab should not affect actors in the outer prefab,
    // even though they have similar names.
    EXPECT_TRUE(sceneImpl->IsActorContactEnabled(
        FindActorByName(scene, "MyNewPrefab/CubeA"),
        FindActorByName(scene, "MyNewPrefab/CubeB"),
        test::ExpectOK{}));
    EXPECT_TRUE(sceneImpl->IsActorContactEnabled(
        FindActorByName(scene, "MyNewPrefab/CubeA"),
        FindActorByName(scene, "MyNewPrefab/MyNestedPrefab/CubeB"),
        test::ExpectOK{}));
    EXPECT_FALSE(sceneImpl->IsActorContactEnabled(
        FindActorByName(scene, "MyNewPrefab/MyNestedPrefab/CubeA"),
        FindActorByName(scene, "MyNewPrefab/MyNestedPrefab/CubeB"),
        test::ExpectOK{}));
  }
}

TEST_IF(MOCHI_HDF5_AND_INTERNAL, Prefab, ContactFilter_ParentExpansion) {
  // Test that contact filters expand to nested links by default when the parent articulated actor
  // is named, but do NOT expand when a specific link is named.
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("test");
  auto const* sceneImpl = assert_cast<SceneImpl const*>(scene);
  MOCHI_DEFER(context->DestroyScene(scene));

  std::string json = R"({
    "actors": {
      "articulated": [{
        "name": "Robot",
        "joints": [
          {"type": "Free"},
          {"type": "Revolute", "axis": [0, 0, 1]}
        ],
        "links": [
          {
            "name": "boneA",
            "parentLink": -1,
            "colliderType": "Box",
            "shape": "articulated/two_links_revolute/bone_a.mochi.h5"
          },
          {
            "name": "boneB",
            "parentLink": 0,
            "colliderType": "Box",
            "shape": "articulated/two_links_revolute/bone_b.mochi.h5"
          }
        ]
      }],
      "rigid": [{
        "name": "Cube",
        "colliderType": "Box",
        "shape": "cube/cube_minimal.mochi.json",
        "scale": [0.1, 0.1, 0.1]
      }]
    },
      "contactFilter": {
        "actorContactSymmetric": [
          {"enable": false, "actors": ["Cube", "Robot"]},
          {"enable": true, "actors": ["Cube", "Robot/boneA"]}
        ]
      }
  })";

  auto scenePrefab = prefab::ShallowLoadFromJsonString(json, test::ExpectOK{});
  prefab::LoadShapes(scenePrefab, test::GetAssetsDir(), context, test::ExpectOK{});
  prefab::AddToScene(scenePrefab, scene, {}, test::ExpectOK{});

  // Access contact filter table
  auto cube = FindActorByName(scene, "Cube");
  auto robot = FindActorByName(scene, "Robot");
  auto boneA = FindActorByName(scene, "Robot/boneA");
  auto boneB = FindActorByName(scene, "Robot/boneB");

  // Cube <-> boneA: ENABLED (re-enabled by second filter)
  EXPECT_TRUE(sceneImpl->IsActorContactEnabled(cube, boneA, test::ExpectOK{}));
  EXPECT_TRUE(sceneImpl->IsActorContactEnabled(boneA, cube, test::ExpectOK{}));

  // Cube <-> Robot*: DISABLED (disabled by first filter, not re-enabled)
  EXPECT_FALSE(sceneImpl->IsActorContactEnabled(cube, robot, test::ExpectOK{}));
  EXPECT_FALSE(sceneImpl->IsActorContactEnabled(robot, cube, test::ExpectOK{}));
  EXPECT_FALSE(sceneImpl->IsActorContactEnabled(cube, boneB, test::ExpectOK{}));
  EXPECT_FALSE(sceneImpl->IsActorContactEnabled(boneB, cube, test::ExpectOK{}));
}

TEST_IF(MOCHI_HDF5_AND_INTERNAL, Prefab, ContactFilter_ExplicitFalseUsesExactParent) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("test");
  MOCHI_DEFER(context->DestroyScene(scene));

  auto scenePrefab = prefab::ShallowLoadFromJsonString(
      R"({
        "actors": {
          "articulated": [{
            "name": "Robot",
            "joints": [
              {"type": "Free"},
              {"type": "Revolute", "axis": [0, 0, 1]}
            ],
            "links": [
              {
                "name": "boneA",
                "parentLink": -1,
                "colliderType": "Box",
                "shape": "articulated/two_links_revolute/bone_a.mochi.h5"
              },
              {
                "name": "boneB",
                "parentLink": 0,
                "colliderType": "Box",
                "shape": "articulated/two_links_revolute/bone_b.mochi.h5"
              }
            ]
          }],
          "rigid": [
            {"name": "Cube", "colliderType": "Box", "shape": "cube/cube_minimal.mochi.json"}
          ]
        },
        "contactFilter": {
          "actorContactSymmetric": [
            {"enable": false, "includeNestedActors": false, "actors": ["Cube", "Robot"]}
          ]
        }
      })",
      test::ExpectOK{});
  prefab::LoadShapes(scenePrefab, test::GetAssetsDir(), context, test::ExpectOK{});
  prefab::AddToScene(scenePrefab, scene, {}, test::ExpectOK{});

  auto const* sceneImpl = assert_cast<SceneImpl const*>(scene);
  auto cube = FindActorByName(scene, "Cube");
  auto robot = FindActorByName(scene, "Robot");
  auto boneA = FindActorByName(scene, "Robot/boneA");
  auto boneB = FindActorByName(scene, "Robot/boneB");

  EXPECT_FALSE(sceneImpl->IsActorContactEnabled(cube, robot, test::ExpectOK{}));
  EXPECT_FALSE(sceneImpl->IsActorContactEnabled(robot, cube, test::ExpectOK{}));
  EXPECT_TRUE(sceneImpl->IsActorContactEnabled(cube, boneA, test::ExpectOK{}));
  EXPECT_TRUE(sceneImpl->IsActorContactEnabled(boneB, cube, test::ExpectOK{}));
}

TEST_IF(MOCHI_HDF5_AND_INTERNAL, Prefab, ContactFilter_SymmetricSelfPairIncludesNestedActors) {
  // A self-pair on an articulated parent applies to the cross product of its nested actors: both
  // self-contact of each nested link (e.g. boneA vs itself -- the pair the removed self-pair skip
  // used to leave enabled) and cross-contact between distinct nested links (boneA vs boneB).
  // A jointed chain defaults to self-contact ENABLED but adjacent cross-contact DISABLED, so we
  // probe each direction with the value that actually flips it (non-vacuous).
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));

  auto loadRobotWithSelfPair = [&](bool enable) -> Scene* {
    auto* scene = context->CreateScene("test");
    std::string json = R"({
      "actors": {
        "articulated": [{
          "name": "Robot",
          "joints": [
            {"type": "Free"},
            {"type": "Revolute", "axis": [0, 0, 1]}
          ],
          "links": [
            {
              "name": "boneA",
              "parentLink": -1,
              "colliderType": "Box",
              "shape": "articulated/two_links_revolute/bone_a.mochi.h5"
            },
            {
              "name": "boneB",
              "parentLink": 0,
              "colliderType": "Box",
              "shape": "articulated/two_links_revolute/bone_b.mochi.h5"
            }
          ]
        }]
      },
      "contactFilter": {
          "actorContactSymmetric": [
            {"enable": )" +
        std::string(enable ? "true" : "false") +
        R"(, "actors": ["Robot", "Robot"]}
          ]
        }
      })";
    auto scenePrefab = prefab::ShallowLoadFromJsonString(json, test::ExpectOK{});
    prefab::LoadShapes(scenePrefab, test::GetAssetsDir(), context, test::ExpectOK{});
    prefab::AddToScene(scenePrefab, scene, {}, test::ExpectOK{});
    return scene;
  };

  // enable:false -> self-contact (enabled by default) is disabled on the diagonal.
  {
    auto* scene = loadRobotWithSelfPair(false);
    MOCHI_DEFER(context->DestroyScene(scene));
    auto const* sceneImpl = assert_cast<SceneImpl const*>(scene);
    auto boneA = FindActorByName(scene, "Robot/boneA");
    auto boneB = FindActorByName(scene, "Robot/boneB");
    EXPECT_FALSE(sceneImpl->IsActorContactEnabled(boneA, boneA, test::ExpectOK{}));
    EXPECT_FALSE(sceneImpl->IsActorContactEnabled(boneB, boneB, test::ExpectOK{}));
  }

  // enable:true -> cross-contact between adjacent links (disabled by default) is enabled
  // off-diagonal.
  {
    auto* scene = loadRobotWithSelfPair(true);
    MOCHI_DEFER(context->DestroyScene(scene));
    auto const* sceneImpl = assert_cast<SceneImpl const*>(scene);
    auto boneA = FindActorByName(scene, "Robot/boneA");
    auto boneB = FindActorByName(scene, "Robot/boneB");
    EXPECT_TRUE(sceneImpl->IsActorContactEnabled(boneA, boneB, test::ExpectOK{}));
  }
}

TEST_IF(MOCHI_INTERNAL, Prefab, AddToSceneResult_ConstraintsAndActorIdentity) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("test");
  MOCHI_DEFER(context->DestroyScene(scene));

  // Build a prefab with two dynamic rigid actors and two constraints (of different types).
  // Both actors must be dynamic — rigid joint constraints do not support static actors.
  prefab::ScenePrefab scenePrefab;

  auto& boxA = scenePrefab.actors.rigid.push_back();
  boxA.name = "BoxA";
  boxA.shapeFile = "cube/cube_minimal.mochi.json";
  boxA.colliderType = ColliderType::Box;

  auto& boxB = scenePrefab.actors.rigid.push_back();
  boxB.name = "BoxB";
  boxB.shapeFile = "cube/cube_minimal.mochi.json";
  boxB.colliderType = ColliderType::Box;
  boxB.translation = Real3{0_r, 1_r, 0_r};

  // Spherical joint constraint between the two actors
  auto& spherical = scenePrefab.constraints.rigidSphericalJoint.push_back();
  spherical.actorNameA = "BoxA";
  spherical.actorNameB = "BoxB";
  spherical.localPosA = Real3{0_r, 0.5_r, 0_r};
  spherical.localPosB = Real3{0_r, -0.5_r, 0_r};
  spherical.stiffness = 1e5_r;

  // Pivot position constraint on BoxB
  auto& pivot = scenePrefab.constraints.rigidPivotPosition.push_back();
  pivot.actorName = "BoxB";
  pivot.targetPosition = Real3{0_r, 1_r, 0_r};
  pivot.localPosition = Real3{0_r, 0_r, 0_r};
  pivot.stiffness = 1e4_r;

  prefab::LoadShapes(scenePrefab, test::GetAssetsDir(), context, test::ExpectOK{});

  auto const result = prefab::AddToScene(scenePrefab, scene, {}, test::ExpectOK{});

  // Verify actors: correct count and names. Only one actor type (rigid) is present, so the
  // returned actors must be in the same order as the prefab's rigid actor list.
  EXPECT_EQ(2, scene->GetNumActors());
  ASSERT_EQ(2, isize(result.actors));
  auto rigidActors = result.Filter(ActorType::Rigid);
  ASSERT_EQ(2, isize(rigidActors));
  EXPECT_STREQ("BoxA", rigidActors[0]->GetName());
  EXPECT_STREQ("BoxB", rigidActors[1]->GetName());

  // Each returned Actor* should match the scene lookup by name
  for (auto const* actor : result.actors) {
    EXPECT_NE(nullptr, actor);
    ActorHandle const handle = FindActorByName(scene, actor->GetName());
    EXPECT_TRUE(handle.IsValid());
    EXPECT_EQ(actor, scene->GetActor(handle));
  }

  // Verify constraints: correct count and identity. The order in result.constraints is NOT
  // guaranteed when multiple constraint types are present, so use Filter() to look them up.
  EXPECT_EQ(2, scene->GetNumConstraints());
  ASSERT_EQ(2, isize(result.constraints));
  for (auto const* constraint : result.constraints) {
    EXPECT_NE(nullptr, constraint);
    EXPECT_EQ(constraint, scene->GetConstraint(constraint->GetHandle()));
  }

  auto rigidSphericalJoints = result.Filter(ConstraintType::RigidSphericalJoint);
  ASSERT_EQ(1, isize(rigidSphericalJoints));
  ASSERT_EQ(ConstraintType::RigidSphericalJoint, rigidSphericalJoints[0]->GetType());
  EXPECT_NEAR_EQ(1e5_r, rigidSphericalJoints[0]->GetStiffness());

  auto rigidPivotPositions = result.Filter(ConstraintType::RigidPivotPosition);
  ASSERT_EQ(1, isize(rigidPivotPositions));
  ASSERT_EQ(ConstraintType::RigidPivotPosition, rigidPivotPositions[0]->GetType());
  EXPECT_NEAR_EQ(1e4_r, rigidPivotPositions[0]->GetStiffness());
}

static prefab::RigidActorPrefab MakeRigidBox(std::string_view name) {
  prefab::RigidActorPrefab actor;
  actor.name = std::string(name);
  actor.shapeFile = "cube/cube_minimal.mochi.json";
  actor.colliderType = ColliderType::Box;
  return actor;
}

static prefab::SoftActorPrefab MakeSoftBox(std::string_view name) {
  prefab::SoftActorPrefab actor;
  actor.name = std::string(name);
  actor.shapeFile = "cube/cube_minimal.mochi.json";
  return actor;
}

static prefab::RigidPivotPositionConstraintPrefab MakePivotPositionConstraint(
    std::string_view actorName,
    real stiffness) {
  prefab::RigidPivotPositionConstraintPrefab c;
  c.actorName = std::string(actorName);
  c.targetPosition = Real3{0_r, 1_r, 0_r};
  c.localPosition = Real3{0_r, 0_r, 0_r};
  c.stiffness = stiffness;
  return c;
}

static prefab::RigidPivotRotationConstraintPrefab MakePivotRotationConstraint(
    std::string_view actorName,
    real stiffness) {
  prefab::RigidPivotRotationConstraintPrefab c;
  c.actorName = std::string(actorName);
  c.targetRotation = Real3{0_r, 0_r, 0_r};
  c.localRotation = Real3{0_r, 0_r, 0_r};
  c.stiffness = stiffness;
  return c;
}

TEST_IF(MOCHI_INTERNAL, Prefab, AddToSceneResult_MultipleActorTypes) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("test");
  MOCHI_DEFER(context->DestroyScene(scene));

  // Mix of rigid and soft actors. The relative order of "rigid vs soft" in result.actors is
  // unspecified. We verify only the total set of actors and per-type ordering via Filter().
  prefab::ScenePrefab scenePrefab;
  scenePrefab.actors.rigid.push_back(MakeRigidBox("RigidA"));
  scenePrefab.actors.rigid.push_back(MakeRigidBox("RigidB"));
  scenePrefab.actors.soft.push_back(MakeSoftBox("SoftA"));
  scenePrefab.actors.soft.push_back(MakeSoftBox("SoftB"));
  scenePrefab.actors.soft.push_back(MakeSoftBox("SoftC"));

  prefab::LoadShapes(scenePrefab, test::GetAssetsDir(), context, test::ExpectOK{});
  auto const result = prefab::AddToScene(scenePrefab, scene, {}, test::ExpectOK{});

  // Total counts match, but order across types is unspecified.
  ASSERT_EQ(5, isize(result.actors));
  EXPECT_EQ(5, scene->GetNumActors());

  // Every result actor must exist in the scene.
  for (auto const* actor : result.actors) {
    ASSERT_NE(nullptr, actor);
    EXPECT_EQ(actor, scene->GetActor(actor->GetHandle()));
  }

  // Filter() preserves declaration order within each actor type.
  auto rigids = result.Filter(ActorType::Rigid);
  ASSERT_EQ(2, isize(rigids));
  EXPECT_STREQ("RigidA", rigids[0]->GetName());
  EXPECT_STREQ("RigidB", rigids[1]->GetName());
  EXPECT_EQ(ActorType::Rigid, rigids[0]->GetType());
  EXPECT_EQ(ActorType::Rigid, rigids[1]->GetType());

  auto softs = result.Filter(ActorType::Soft);
  ASSERT_EQ(3, isize(softs));
  EXPECT_STREQ("SoftA", softs[0]->GetName());
  EXPECT_STREQ("SoftB", softs[1]->GetName());
  EXPECT_STREQ("SoftC", softs[2]->GetName());
  EXPECT_EQ(ActorType::Soft, softs[0]->GetType());
  EXPECT_EQ(ActorType::Soft, softs[1]->GetType());
  EXPECT_EQ(ActorType::Soft, softs[2]->GetType());

  // Filter() with a type that doesn't appear returns empty.
  EXPECT_TRUE(result.Filter(ActorType::Articulated).empty());
}

TEST_IF(MOCHI_INTERNAL, Prefab, AddToSceneResult_MultipleConstraintTypes) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("test");
  MOCHI_DEFER(context->DestroyScene(scene));

  // One rigid actor, multiple constraints of two distinct types. Stiffness encodes declaration
  // order so we can verify per-type ordering through Filter().
  prefab::ScenePrefab scenePrefab;
  scenePrefab.actors.rigid.push_back(MakeRigidBox("Box"));

  scenePrefab.constraints.rigidPivotPosition.push_back(MakePivotPositionConstraint("Box", 11_r));
  scenePrefab.constraints.rigidPivotPosition.push_back(MakePivotPositionConstraint("Box", 12_r));

  scenePrefab.constraints.rigidPivotRotation.push_back(MakePivotRotationConstraint("Box", 21_r));
  scenePrefab.constraints.rigidPivotRotation.push_back(MakePivotRotationConstraint("Box", 22_r));
  scenePrefab.constraints.rigidPivotRotation.push_back(MakePivotRotationConstraint("Box", 23_r));

  prefab::LoadShapes(scenePrefab, test::GetAssetsDir(), context, test::ExpectOK{});
  auto const result = prefab::AddToScene(scenePrefab, scene, {}, test::ExpectOK{});

  // Total count matches, but ordering across constraint types is unspecified.
  ASSERT_EQ(5, isize(result.constraints));
  EXPECT_EQ(5, scene->GetNumConstraints());

  // Filter() preserves declaration order within each constraint type.
  auto positions = result.Filter(ConstraintType::RigidPivotPosition);
  ASSERT_EQ(2, isize(positions));
  EXPECT_NEAR_EQ(11_r, positions[0]->GetStiffness());
  EXPECT_NEAR_EQ(12_r, positions[1]->GetStiffness());
  EXPECT_EQ(ConstraintType::RigidPivotPosition, positions[0]->GetType());
  EXPECT_EQ(ConstraintType::RigidPivotPosition, positions[1]->GetType());

  auto rotations = result.Filter(ConstraintType::RigidPivotRotation);
  ASSERT_EQ(3, isize(rotations));
  EXPECT_NEAR_EQ(21_r, rotations[0]->GetStiffness());
  EXPECT_NEAR_EQ(22_r, rotations[1]->GetStiffness());
  EXPECT_NEAR_EQ(23_r, rotations[2]->GetStiffness());
  EXPECT_EQ(ConstraintType::RigidPivotRotation, rotations[0]->GetType());
  EXPECT_EQ(ConstraintType::RigidPivotRotation, rotations[1]->GetType());
  EXPECT_EQ(ConstraintType::RigidPivotRotation, rotations[2]->GetType());

  // Filter() with a constraint type that doesn't appear returns empty.
  EXPECT_TRUE(result.Filter(ConstraintType::RigidSphericalJoint).empty());
}

TEST(Prefab, AddToScene_MissingActorInActorLocalConstraintReturnsError) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("test");
  MOCHI_DEFER(context->DestroyScene(scene));

  prefab::ScenePrefab scenePrefab;
  scenePrefab.constraints.rigidPivotPosition.push_back(MakePivotPositionConstraint("Missing", 1_r));

  auto suppressError = test::SuppressLogError();
  prefab::AddToScene(scenePrefab, scene, {}, test::ExpectNotOK{});
}

// Prefab-transform coverage: the ApplyPrefabTransformToConstraint overloads reuse a few field
// transforms, so one test per transform stands in for the rest. RigidPivotRotation covers
// TransformRotationVector -- shared by JointRotationRange and JointRotationTracking
// (refFrameRotVec). RigidPivotPosition covers worldFromPrefab.TransformPoint -- shared by
// DeformableNodePosition and the RigidPivotToRigidTarget target translation.
// RigidPivotToRigidTarget has its own test for the unique targetTransform rotation compose.
// RigidPrismaticJoint's deviation reads zero regardless of the transform (it captures its reference
// pose at construction), so RigidPrismaticJointConstraint_PrefabScaleScalesLimits reads the stored
// constraint data directly to verify its scaled min/max limits instead.
// Note: The prefab assets used by the tests below are not shipped externally.
TEST_IF(MOCHI_INTERNAL, Prefab, RigidPivotToRigidTargetConstraint_PrefabTransformFollowsActor) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));

  // This constraint's deviation is position-only and center-of-mass-dependent, so a zero-deviation
  // setup would be fragile. Instead assert the invariant the fix guarantees: instantiating with a
  // prefab transform moves the actor and its target frame together, so the world-space deviation
  // just rotates by the prefab rotation (the translation cancels in the difference). The target's
  // non-identity rotation (a different axis than the prefab) exercises the targetTransform rotation
  // compose, not just the translation.
  auto devFor = [&](prefab::PrefabParams const& params) {
    auto* scene = context->CreateScene("s");
    MOCHI_DEFER(context->DestroyScene(scene));
    prefab::ScenePrefab scenePrefab;
    scenePrefab.actors.rigid.push_back(MakeRigidBox("Box"));
    prefab::RigidPivotToRigidTargetConstraintPrefab constraint;
    constraint.actorName = "Box";
    constraint.localPosition = Real3{0.5_r, 0.3_r, -0.2_r};
    constraint.targetTransform = TransformRT(
        Quaternion::FromAxisAngle(Real3{1_r, 0_r, 0_r}, 0.4_r), Real3{0.1_r, -0.4_r, 0.2_r});
    scenePrefab.constraints.rigidPivotToRigidTarget.push_back(constraint);
    prefab::LoadShapes(scenePrefab, test::GetAssetsDir(), context, test::ExpectOK{});
    auto const result = prefab::AddToScene(scenePrefab, scene, params, test::ExpectOK{});
    return Unflatten<Real3>(
        result.Filter(ConstraintType::RigidPivotToRigidTarget)[0]->GetDeviation())[0];
  };

  auto const devId = devFor(prefab::PrefabParams{});
  prefab::PrefabParams params;
  params.rotation = Quaternion::FromAxisAngle(Real3{0_r, 1_r, 0_r}, 0.5_r);
  params.translation = Real3{3_r, -1_r, 2_r};
  auto const devTf = devFor(params);

  Real3 const expected = TransformRT(params.rotation).TransformPoint(devId);
  EXPECT_FALSE(NearEqual(Real3{}, devId, 0.1_r));
  EXPECT_NEAR_TOL(devTf, expected, 1e-5_r);
}

TEST_IF(MOCHI_INTERNAL, Prefab, RigidPivotRotationConstraint_PrefabRotationTransformsTarget) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));

  // Covers TransformRotationVector (shared with JointRotationRange/Tracking refFrameRotVec). The
  // target rotation is about a different axis than the prefab rotation, and non-zero, so the
  // compose order (R_prefab applied in the parent frame) is actually exercised -- a zero
  // targetRotation would make left-vs-right compose order indistinguishable. The target rotates
  // with the prefab while the actor's authored rotation is identity, so the world-space rotational
  // deviation just rotates by the prefab rotation.
  auto devFor = [&](prefab::PrefabParams const& params) {
    auto* scene = context->CreateScene("s");
    MOCHI_DEFER(context->DestroyScene(scene));
    prefab::ScenePrefab scenePrefab;
    scenePrefab.actors.rigid.push_back(MakeRigidBox("Box"));
    prefab::RigidPivotRotationConstraintPrefab c;
    c.actorName = "Box";
    c.targetRotation = Real3{0.6_r, 0_r, 0_r};
    c.localRotation = Real3{0_r, 0_r, 0_r};
    c.stiffness = 1_r;
    scenePrefab.constraints.rigidPivotRotation.push_back(c);
    prefab::LoadShapes(scenePrefab, test::GetAssetsDir(), context, test::ExpectOK{});
    auto const result = prefab::AddToScene(scenePrefab, scene, params, test::ExpectOK{});
    return Unflatten<Real3>(
        result.Filter(ConstraintType::RigidPivotRotation)[0]->GetDeviation())[0];
  };

  auto const devId = devFor(prefab::PrefabParams{});
  prefab::PrefabParams params;
  params.rotation = Quaternion::FromAxisAngle(Real3{0_r, 1_r, 0_r}, 0.5_r);
  auto const devTf = devFor(params);

  Real3 const expected = TransformRT(params.rotation).TransformPoint(devId);
  EXPECT_FALSE(NearEqual(Real3{}, devId, 0.1_r));
  EXPECT_NEAR_TOL(devTf, expected, 1e-5_r);
}

TEST_IF(MOCHI_INTERNAL, Prefab, RigidPivotPositionConstraint_PrefabTransformMovesTarget) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("test");
  MOCHI_DEFER(context->DestroyScene(scene));

  // The actor-local pivot and its world-space target coincide at authoring time, so the constraint
  // has zero deviation. Instantiating the prefab with a non-identity rotation + translation must
  // move the world-space targetPosition by the same transform as the actor, keeping the deviation
  // ~zero. (Rigid actors store an RT pose with shape scale applied outside the pose, so uniform
  // scale exercises a separate code path and is intentionally not varied here.)
  prefab::ScenePrefab scenePrefab;
  scenePrefab.actors.rigid.push_back(MakeRigidBox("Box"));
  prefab::RigidPivotPositionConstraintPrefab constraint;
  constraint.actorName = "Box";
  constraint.localPosition = Real3{0.5_r, 0_r, 0_r};
  constraint.targetPosition = Real3{0.5_r, 0_r, 0_r};
  constraint.stiffness = 1_r;
  scenePrefab.constraints.rigidPivotPosition.push_back(constraint);
  prefab::LoadShapes(scenePrefab, test::GetAssetsDir(), context, test::ExpectOK{});

  prefab::PrefabParams params;
  params.rotation = Quaternion::FromAxisAngle(Real3{0_r, 1_r, 0_r}, 0.5_r);
  params.translation = Real3{3_r, -1_r, 2_r};
  auto const result = prefab::AddToScene(scenePrefab, scene, params, test::ExpectOK{});

  auto constraints = result.Filter(ConstraintType::RigidPivotPosition);
  ASSERT_EQ(1, isize(constraints));
  Real3 const deviation = Unflatten<Real3>(constraints[0]->GetDeviation())[0];
  EXPECT_NEAR_TOL(Real3{}, deviation, 1e-5_r);
}

TEST_IF(
    MOCHI_INTERNAL,
    Prefab,
    RigidPrismaticJointConstraint_PrefabTransformRotatesAxisAndScalesLimits) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("test");
  MOCHI_DEFER(context->DestroyScene(scene));

  // A prismatic joint's min/max travel limits are distances (meters) and must scale with the prefab
  // scale, while its free axis must rotate with the prefab. No public accessor exposes them, so
  // read the stored CConstraintData directly. The transform is applied via a nested PrefabReference
  // (the ScenePrefab AddToScene overload requires PrefabParams::scale == 1). Also verifies that
  // stiffness, damping, and saturation are preserved as authored under non-unit prefab scale.
  real const nestedScale = 2_r;
  real const authoredMin = -0.5_r;
  real const authoredMax = 1.5_r;
  real const authoredStiffness = 3_r;
  real const authoredDamping = 4_r;
  real const authoredSaturation = 0.25_r;
  Real3 const authoredFreeAxis = Real3{1_r, 0_r, 0_r};
  Real3 const expectedLocalFreeAxis = Real3{0_r, 0_r, 1_r};
  auto inner = std::make_shared<prefab::ScenePrefab>();
  inner->actors.rigid.push_back(MakeRigidBox("A"));
  auto boxB = MakeRigidBox("B");
  boxB.translation = Real3{1_r, 0_r, 0_r};
  inner->actors.rigid.push_back(boxB);
  prefab::RigidPrismaticJointConstraintPrefab c;
  c.actorNameA = "A";
  c.actorNameB = "B";
  c.freeAxis = authoredFreeAxis;
  c.min = authoredMin;
  c.max = authoredMax;
  c.stiffness = authoredStiffness;
  c.damping = authoredDamping;
  c.saturation = authoredSaturation;
  inner->constraints.rigidPrismaticJoint.push_back(c);
  prefab::ScenePrefab scenePrefab;
  auto& nestedRef = scenePrefab.prefabs.push_back();
  nestedRef.scale = nestedScale;
  nestedRef.rotation = Quaternion::FromAxisAngle(Real3{0_r, 0_r, 1_r}, 0.7_r);
  nestedRef.prefab = inner;
  prefab::LoadShapes(scenePrefab, test::GetAssetsDir(), context, test::ExpectOK{});
  auto const result = prefab::AddToScene(scenePrefab, scene, {}, test::ExpectOK{});

  auto constraints = result.Filter(ConstraintType::RigidPrismaticJoint);
  ASSERT_EQ(1, isize(constraints));
  auto const& reg = static_cast<SceneImpl*>(scene)->GetRegistry();
  auto const entity = GetEntityUnchecked(constraints[0]->GetHandle());
  auto const& data = reg.get<CConstraintData<ConstraintType::RigidPrismaticJoint>>(entity);
  ASSERT_TRUE(data.min.has_value());
  ASSERT_TRUE(data.max.has_value());
  EXPECT_NEAR_TOL(data.localFrame * authoredFreeAxis, expectedLocalFreeAxis, 1e-5_r);
  EXPECT_NEAR_EQ(authoredMin * nestedScale, *data.min);
  EXPECT_NEAR_EQ(authoredMax * nestedScale, *data.max);
  EXPECT_NEAR_EQ(authoredStiffness, constraints[0]->GetStiffness());
  EXPECT_NEAR_EQ(authoredDamping, constraints[0]->GetDamping());
  EXPECT_NEAR_EQ(authoredSaturation, constraints[0]->GetSaturation());
}

TEST_IF(MOCHI_INTERNAL, Prefab, RigidPivotPositionConstraint_PrefabScaleScalesLocalPivot) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("test");
  MOCHI_DEFER(context->DestroyScene(scene));

  // Pivot (actor-local) and target (world) coincide at authoring, so the deviation is zero. Under a
  // non-unit prefab scale the world target scales; the actor-local pivot offset must scale by the
  // same factor or the constraint acquires a spurious deviation. The scale is applied via a nested
  // PrefabReference (the ScenePrefab AddToScene overload used here requires PrefabParams::scale ==
  // 1); combined with rotation+translation to confirm the full prefab transform composes.
  auto inner = std::make_shared<prefab::ScenePrefab>();
  inner->actors.rigid.push_back(MakeRigidBox("Box"));
  prefab::RigidPivotPositionConstraintPrefab c;
  c.actorName = "Box";
  c.localPosition = Real3{0.5_r, 0_r, 0_r};
  c.targetPosition = Real3{0.5_r, 0_r, 0_r};
  c.stiffness = 1_r;
  inner->constraints.rigidPivotPosition.push_back(c);
  prefab::ScenePrefab scenePrefab;
  auto& nestedRef = scenePrefab.prefabs.push_back();
  nestedRef.scale = 2_r;
  nestedRef.prefab = inner;
  prefab::LoadShapes(scenePrefab, test::GetAssetsDir(), context, test::ExpectOK{});

  prefab::PrefabParams params;
  params.rotation = Quaternion::FromAxisAngle(Real3{0_r, 1_r, 0_r}, 0.5_r);
  params.translation = Real3{3_r, -1_r, 2_r};
  auto const result = prefab::AddToScene(scenePrefab, scene, params, test::ExpectOK{});

  auto constraints = result.Filter(ConstraintType::RigidPivotPosition);
  ASSERT_EQ(1, isize(constraints));
  auto const deviation = constraints[0]->GetDeviation();
  EXPECT_TRUE(test::NearEqualSpan(deviation, Real3{0_r, 0_r, 0_r}, 1e-5_r));
}

TEST_IF(MOCHI_INTERNAL, Prefab, RigidPivotToRigidTargetConstraint_PrefabScaleScalesLocalPosition) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));

  // This constraint's deviation is position-only and center-of-mass-dependent, so a zero-deviation
  // setup is fragile. Instead assert the invariant the localPosition scaling guarantees:
  // instantiating at a non-unit scale (via a nested PrefabReference) scales the actor, its target
  // frame, and the actor-local localPosition together, so the world-space deviation just scales by
  // that factor and rotates by the prefab rotation. Without scaling localPosition it would not.
  auto devFor = [&](real nestedScale, prefab::PrefabParams const& params) {
    auto* scene = context->CreateScene("s");
    MOCHI_DEFER(context->DestroyScene(scene));
    auto inner = std::make_shared<prefab::ScenePrefab>();
    inner->actors.rigid.push_back(MakeRigidBox("Box"));
    prefab::RigidPivotToRigidTargetConstraintPrefab c;
    c.actorName = "Box";
    c.localPosition = Real3{0.5_r, 0.3_r, -0.2_r};
    c.targetTransform = TransformRT(
        Quaternion::FromAxisAngle(Real3{1_r, 0_r, 0_r}, 0.4_r), Real3{0.1_r, -0.4_r, 0.2_r});
    c.stiffness = 1_r;
    inner->constraints.rigidPivotToRigidTarget.push_back(c);
    prefab::ScenePrefab scenePrefab;
    auto& nestedRef = scenePrefab.prefabs.push_back();
    nestedRef.scale = nestedScale;
    nestedRef.prefab = inner;
    prefab::LoadShapes(scenePrefab, test::GetAssetsDir(), context, test::ExpectOK{});
    auto const result = prefab::AddToScene(scenePrefab, scene, params, test::ExpectOK{});
    return result.Filter(ConstraintType::RigidPivotToRigidTarget)[0]->GetDeviation();
  };

  auto const devId = devFor(1_r, prefab::PrefabParams{});
  prefab::PrefabParams params;
  params.rotation = Quaternion::FromAxisAngle(Real3{0_r, 1_r, 0_r}, 0.5_r);
  params.translation = Real3{3_r, -1_r, 2_r};
  auto const devScaled = devFor(2_r, params);

  // devScaled = 2 * R * devId: scaled by the nested factor, rotated by the prefab rotation.
  Real3 const expected =
      TransformRT(params.rotation).TransformPoint(Real3{devId[0], devId[1], devId[2]}) * 2_r;
  EXPECT_FALSE(test::NearEqualSpan(devId, Real3{0_r, 0_r, 0_r}, 0.1_r));
  EXPECT_TRUE(test::NearEqualSpan(devScaled, expected, 1e-5_r));
}

TEST_IF(MOCHI_INTERNAL, Prefab, RigidSphericalJointConstraint_PrefabScaleScalesLocalPivots) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("test");
  MOCHI_DEFER(context->DestroyScene(scene));

  // Two rigid actors whose spherical-joint pivots coincide at authoring (deviation 0). Under a
  // non-unit prefab scale the actor positions scale, so both actor-local pivot offsets must scale
  // by the same factor or the joint's attachment points separate. The scale is applied via a nested
  // PrefabReference (the ScenePrefab AddToScene overload used here requires PrefabParams::scale ==
  // 1); combined with rotation+translation to confirm the full prefab transform composes.
  auto inner = std::make_shared<prefab::ScenePrefab>();
  inner->actors.rigid.push_back(MakeRigidBox("BoxA"));
  auto boxB = MakeRigidBox("BoxB");
  boxB.translation = Real3{1_r, 0_r, 0_r};
  inner->actors.rigid.push_back(boxB);
  prefab::RigidSphericalJointConstraintPrefab c;
  c.actorNameA = "BoxA";
  c.actorNameB = "BoxB";
  c.localPosA = Real3{0.5_r, 0_r, 0_r};
  c.localPosB = Real3{-0.5_r, 0_r, 0_r};
  c.stiffness = 1_r;
  inner->constraints.rigidSphericalJoint.push_back(c);
  prefab::ScenePrefab scenePrefab;
  auto& nestedRef = scenePrefab.prefabs.push_back();
  nestedRef.scale = 2_r;
  nestedRef.prefab = inner;
  prefab::LoadShapes(scenePrefab, test::GetAssetsDir(), context, test::ExpectOK{});

  prefab::PrefabParams params;
  params.rotation = Quaternion::FromAxisAngle(Real3{0_r, 1_r, 0_r}, 0.5_r);
  params.translation = Real3{3_r, -1_r, 2_r};
  auto const result = prefab::AddToScene(scenePrefab, scene, params, test::ExpectOK{});

  auto constraints = result.Filter(ConstraintType::RigidSphericalJoint);
  ASSERT_EQ(1, isize(constraints));
  auto const deviation = constraints[0]->GetDeviation();
  EXPECT_TRUE(test::NearEqualSpan(deviation, Real3{0_r, 0_r, 0_r}, 1e-5_r));
}

TEST_IF(
    MOCHI_INTERNAL,
    Prefab,
    RigidSphericalJointConstraint_ParentOwnedNestedScaleScalesLocalPivots) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("test");
  MOCHI_DEFER(context->DestroyScene(scene));

  auto child = std::make_shared<prefab::ScenePrefab>();
  child->actors.rigid.push_back(MakeRigidBox("BoxA"));
  auto boxB = MakeRigidBox("BoxB");
  boxB.translation = Real3{1_r, 0_r, 0_r};
  child->actors.rigid.push_back(boxB);

  prefab::ScenePrefab scenePrefab;
  auto& nestedRef = scenePrefab.prefabs.push_back();
  nestedRef.name = "child";
  nestedRef.scale = 2_r;
  nestedRef.prefab = child;

  prefab::RigidSphericalJointConstraintPrefab c;
  c.actorNameA = "child/BoxA";
  c.actorNameB = "child/BoxB";
  c.localPosA = Real3{0.5_r, 0_r, 0_r};
  c.localPosB = Real3{-0.5_r, 0_r, 0_r};
  c.stiffness = 1_r;
  scenePrefab.constraints.rigidSphericalJoint.push_back(c);
  prefab::LoadShapes(scenePrefab, test::GetAssetsDir(), context, test::ExpectOK{});
  auto const result = prefab::AddToScene(scenePrefab, scene, {}, test::ExpectOK{});

  auto constraints = result.Filter(ConstraintType::RigidSphericalJoint);
  ASSERT_EQ(1, isize(constraints));
  auto const deviation = constraints[0]->GetDeviation();
  EXPECT_TRUE(test::NearEqualSpan(deviation, Real3{0_r, 0_r, 0_r}, 1e-5_r));
}

TEST_IF(MOCHI_INTERNAL, Prefab, DeformableNodeToRigidConstraint_PrefabScaleScalesRigidLocalPos) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));

  // With fixToDeformablePos=false the stored rigid attachment point is rigidLocalPos - rigidCOM.
  // Both the authored offset and the geometry-baked COM scale with the prefab scale, so the stored
  // point must scale by the same factor. Read it at scale 1 and 2 (applied via a nested
  // PrefabReference, as the ScenePrefab overload requires PrefabParams::scale == 1) and compare.
  auto posLocalRigidFor = [&](real nestedScale) {
    auto* scene = context->CreateScene("s");
    MOCHI_DEFER(context->DestroyScene(scene));
    auto inner = std::make_shared<prefab::ScenePrefab>();
    inner->actors.rigid.push_back(MakeRigidBox("Rigid"));
    inner->actors.soft.push_back(MakeSoftBox("Soft"));
    prefab::DeformableNodeToRigidConstraintPrefab c;
    c.rigidActorName = "Rigid";
    c.deformableActorName = "Soft";
    c.rigidLocalPos = Real3{0.5_r, 0.3_r, -0.2_r};
    c.deformableNodeIndex = 0;
    c.findClosest = false;
    c.fixToDeformablePos = false;
    inner->constraints.deformableNodeToRigid.push_back(c);
    prefab::ScenePrefab scenePrefab;
    auto& nestedRef = scenePrefab.prefabs.push_back();
    nestedRef.scale = nestedScale;
    nestedRef.prefab = inner;
    prefab::LoadShapes(scenePrefab, test::GetAssetsDir(), context, test::ExpectOK{});
    auto const result = prefab::AddToScene(scenePrefab, scene, {}, test::ExpectOK{});
    auto const& reg = static_cast<SceneImpl*>(scene)->GetRegistry();
    auto const entity =
        GetEntityUnchecked(result.Filter(ConstraintType::DeformableNodeToRigid)[0]->GetHandle());
    return reg.get<CConstraintData<ConstraintType::DeformableNodeToRigid>>(entity).posLocalRigid;
  };

  Real3 const posId = posLocalRigidFor(1_r);
  EXPECT_FALSE(test::NearEqualSpan(posId, Real3{0_r, 0_r, 0_r}, 0.1_r));
  EXPECT_TRUE(test::NearEqualSpan(posLocalRigidFor(2_r), posId * 2_r, 1e-5_r));
}

TEST(Prefab, ArticulatedSingleDofTargetConstraint_PrefabScaleScalesTranslationalTargetOnly) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));

  // A single-DoF target value is a distance [m] for a translational DoF (prismatic/free) and an
  // angle [rad] for a rotational DoF (revolute). Only the translational value must scale with the
  // prefab, so it is scaled by the referenced actor's effective scale while the rotational value is
  // left unchanged. Build a shape-less single-joint actor with one target constraint on its DoF 0,
  // instantiate it at a nested scale, and read back the stored target value. (Nested
  // PrefabReference scale is used because the ScenePrefab AddToScene overload requires
  // PrefabParams::scale == 1.)
  real const authoredTarget = 0.25_r;
  real const authoredSaturation = 0.1_r;
  struct TargetValues {
    real targetValue;
    real saturation;
  };
  enum class ConstraintOwner { NestedPrefab, ParentPrefab };
  auto targetValuesFor = [&](ArticulatedJointType jointType,
                             real artScale,
                             real nestedScale,
                             ConstraintOwner owner =
                                 ConstraintOwner::NestedPrefab) -> TargetValues {
    auto* scene = context->CreateScene("s");
    MOCHI_DEFER(context->DestroyScene(scene));
    auto inner = std::make_shared<prefab::ScenePrefab>();
    auto& art = inner->actors.articulated.push_back();
    art.name = "Art";
    art.scale = artScale;
    art.joints.resize(1);
    art.joints[0].type = jointType;
    art.joints[0].axis = Real3{1_r, 0_r, 0_r};
    art.links.resize(1);
    art.links[0].name = "Root";
    art.links[0].parentLink = -1;
    prefab::ArticulatedSingleDofTargetConstraintPrefab c;
    c.actorName = "Art";
    c.jointIndex = 0;
    c.dofIndex = 0;
    c.targetValue = authoredTarget;
    c.stiffness = 1_r;
    c.saturation = authoredSaturation;
    if (owner == ConstraintOwner::NestedPrefab) {
      inner->constraints.articulatedSingleDofTarget.push_back(c);
    }
    prefab::ScenePrefab scenePrefab;
    auto& nestedRef = scenePrefab.prefabs.push_back();
    nestedRef.name = "Nested";
    nestedRef.scale = nestedScale;
    nestedRef.prefab = inner;
    if (owner == ConstraintOwner::ParentPrefab) {
      c.actorName = "Nested/Art";
      scenePrefab.constraints.articulatedSingleDofTarget.push_back(c);
    }
    auto const result = prefab::AddToScene(scenePrefab, scene, {}, test::ExpectOK{});
    auto const constraints = result.Filter(ConstraintType::ArticulatedSingleDofTarget);
    auto const& reg = static_cast<SceneImpl*>(scene)->GetRegistry();
    auto const entity = GetEntityUnchecked(constraints[0]->GetHandle());
    return TargetValues{
        reg.get<CConstraintTarget<real, TimeStep::Current>>(entity).value,
        constraints[0]->GetSaturation()};
  };

  // Prismatic DoF (translational): the target distance scales with the prefab.
  {
    auto const values = targetValuesFor(ArticulatedJointType::Prismatic, 1_r, 1_r);
    EXPECT_NEAR_EQ(authoredTarget, values.targetValue);
    EXPECT_NEAR_EQ(authoredSaturation, values.saturation);
  }
  {
    auto const values = targetValuesFor(ArticulatedJointType::Prismatic, 1_r, 2_r);
    EXPECT_NEAR_EQ(authoredTarget * 2_r, values.targetValue);
    EXPECT_NEAR_EQ(authoredSaturation, values.saturation);
  }
  // The actor's own scale scales the target too (effective scale = actor.scale * prefab scale), so
  // a fix keyed only on the prefab-instance scale would miss this case.
  {
    auto const values = targetValuesFor(ArticulatedJointType::Prismatic, 2_r, 1_r);
    EXPECT_NEAR_EQ(authoredTarget * 2_r, values.targetValue);
    EXPECT_NEAR_EQ(authoredSaturation, values.saturation);
  }
  // Parent-owned constraints can reference nested actors by hierarchy path and still use the nested
  // actor's effective scale.
  {
    auto const values =
        targetValuesFor(ArticulatedJointType::Prismatic, 1_r, 2_r, ConstraintOwner::ParentPrefab);
    EXPECT_NEAR_EQ(authoredTarget * 2_r, values.targetValue);
    EXPECT_NEAR_EQ(authoredSaturation, values.saturation);
  }
  // Revolute DoF (rotational): the target angle is scale-invariant.
  {
    auto const values = targetValuesFor(ArticulatedJointType::Revolute, 2_r, 2_r);
    EXPECT_NEAR_EQ(authoredTarget, values.targetValue);
    EXPECT_NEAR_EQ(authoredSaturation, values.saturation);
  }
}

TEST(Prefab, ArticulatedSingleDofRangeConstraint_PrefabScaleScalesTranslationalRangeOnly) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));

  // Mirrors ArticulatedSingleDofTargetConstraint_PrefabScaleScalesTranslationalTargetOnly: a
  // single-DoF range [minValue, maxValue] is a distance [m] for a translational (prismatic) DoF and
  // an angle [rad] for a rotational (revolute) DoF. Only the translational range scales with the
  // referenced actor's effective scale; the rotational range is unchanged. Nested PrefabReference
  // scale is used because the ScenePrefab AddToScene overload requires PrefabParams::scale == 1.
  real const authoredMin = -0.25_r;
  real const authoredMax = 0.5_r;
  // CConstraintData is non-copyable, so extract the stored range into a small value type.
  struct RangeValues {
    real minValue;
    real maxValue;
  };
  enum class ConstraintOwner { NestedPrefab, ParentPrefab };
  auto rangeDataFor = [&](ArticulatedJointType jointType,
                          real artScale,
                          real nestedScale,
                          ConstraintOwner owner = ConstraintOwner::NestedPrefab) -> RangeValues {
    auto* scene = context->CreateScene("s");
    MOCHI_DEFER(context->DestroyScene(scene));
    auto inner = std::make_shared<prefab::ScenePrefab>();
    auto& art = inner->actors.articulated.push_back();
    art.name = "Art";
    art.scale = artScale;
    art.joints.resize(1);
    art.joints[0].type = jointType;
    art.joints[0].axis = Real3{1_r, 0_r, 0_r};
    art.links.resize(1);
    art.links[0].name = "Root";
    art.links[0].parentLink = -1;
    prefab::ArticulatedSingleDofRangeConstraintPrefab c;
    c.actorName = "Art";
    c.jointIndex = 0;
    c.dofIndex = 0;
    c.minValue = authoredMin;
    c.maxValue = authoredMax;
    c.stiffness = 1_r;
    if (owner == ConstraintOwner::NestedPrefab) {
      inner->constraints.articulatedSingleDofRange.push_back(c);
    }
    prefab::ScenePrefab scenePrefab;
    auto& nestedRef = scenePrefab.prefabs.push_back();
    nestedRef.name = "Nested";
    nestedRef.scale = nestedScale;
    nestedRef.prefab = inner;
    if (owner == ConstraintOwner::ParentPrefab) {
      c.actorName = "Nested/Art";
      scenePrefab.constraints.articulatedSingleDofRange.push_back(c);
    }
    auto const result = prefab::AddToScene(scenePrefab, scene, {}, test::ExpectOK{});
    auto const constraints = result.Filter(ConstraintType::ArticulatedSingleDofRange);
    EXPECT_EQ(1, isize(constraints)); // exactly one constraint of this type was created
    if (isize(constraints) != 1) {
      return RangeValues{}; // guard the constraints[0] access below against a dispatch regression
    }
    auto const& reg = static_cast<SceneImpl*>(scene)->GetRegistry();
    auto const entity = GetEntityUnchecked(constraints[0]->GetHandle());
    auto const& stored =
        reg.get<CConstraintData<ConstraintType::ArticulatedSingleDofRange>>(entity);
    return RangeValues{stored.minValue, stored.maxValue};
  };

  // Prismatic DoF (translational): the range scales with the effective scale (actor.scale *
  // nested).
  {
    auto const data = rangeDataFor(ArticulatedJointType::Prismatic, 1_r, 1_r);
    EXPECT_NEAR_EQ(authoredMin, data.minValue);
    EXPECT_NEAR_EQ(authoredMax, data.maxValue);
  }
  {
    auto const data = rangeDataFor(ArticulatedJointType::Prismatic, 1_r, 2_r);
    EXPECT_NEAR_EQ(authoredMin * 2_r, data.minValue);
    EXPECT_NEAR_EQ(authoredMax * 2_r, data.maxValue);
  }
  {
    auto const data = rangeDataFor(ArticulatedJointType::Prismatic, 2_r, 1_r);
    EXPECT_NEAR_EQ(authoredMin * 2_r, data.minValue);
    EXPECT_NEAR_EQ(authoredMax * 2_r, data.maxValue);
  }
  // Parent-owned constraints can reference nested actors by hierarchy path and still use the
  // nested actor's effective scale.
  {
    auto const data =
        rangeDataFor(ArticulatedJointType::Prismatic, 1_r, 2_r, ConstraintOwner::ParentPrefab);
    EXPECT_NEAR_EQ(authoredMin * 2_r, data.minValue);
    EXPECT_NEAR_EQ(authoredMax * 2_r, data.maxValue);
  }
  // Revolute DoF (rotational): the range is an angle and is scale-invariant.
  {
    auto const data = rangeDataFor(ArticulatedJointType::Revolute, 2_r, 2_r);
    EXPECT_NEAR_EQ(authoredMin, data.minValue);
    EXPECT_NEAR_EQ(authoredMax, data.maxValue);
  }
}

// Positive AddToScene coverage for the constraint types not already exercised by the AddToScene
// tests above. Each builds the minimal actor(s) the constraint references, adds one constraint, and
// verifies representative stored fields on the created constraint.

TEST(Prefab, Articulated3dRotationRangeConstraint_AddToScene) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("test");
  MOCHI_DEFER(context->DestroyScene(scene));

  // Shape-less single-link articulation with a Free root joint (rotSize == 3, required by the
  // 3D-rotation-range constraint).
  prefab::ScenePrefab scenePrefab;
  auto& art = scenePrefab.actors.articulated.push_back();
  art.name = "Art";
  art.joints.resize(1);
  art.joints[0].type = ArticulatedJointType::Free;
  art.links.resize(1);
  art.links[0].name = "root";
  art.links[0].parentLink = -1;

  prefab::Articulated3dRotationRangeConstraintPrefab c;
  c.actorName = "Art";
  c.jointIndex = 0;
  c.minValues = Real3{-1_r, -1_r, -1_r};
  c.maxValues = Real3{1_r, 1_r, 1_r};
  scenePrefab.constraints.articulated3dRotationRange.push_back(c);

  auto const result = prefab::AddToScene(scenePrefab, scene, {}, test::ExpectOK{});
  auto const constraints = result.Filter(ConstraintType::Articulated3dRotationRange);
  ASSERT_EQ(1, isize(constraints));
  auto const& reg = static_cast<SceneImpl*>(scene)->GetRegistry();
  auto const entity = GetEntityUnchecked(constraints[0]->GetHandle());
  auto const& data = reg.get<CConstraintData<ConstraintType::Articulated3dRotationRange>>(entity);
  EXPECT_EQ(c.jointIndex, data.jointIdx);
  EXPECT_NEAR_EQ(c.minValues, data.minValues);
  EXPECT_NEAR_EQ(c.maxValues, data.maxValues);
}

TEST(Prefab, Articulated3dRotationTargetConstraint_AddToScene) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("test");
  MOCHI_DEFER(context->DestroyScene(scene));

  prefab::ScenePrefab scenePrefab;
  auto& art = scenePrefab.actors.articulated.push_back();
  art.name = "Art";
  art.joints.resize(1);
  art.joints[0].type = ArticulatedJointType::Free;
  art.links.resize(1);
  art.links[0].name = "root";
  art.links[0].parentLink = -1;

  prefab::Articulated3dRotationTargetConstraintPrefab c;
  c.actorName = "Art";
  c.jointIndex = 0;
  c.target = Quaternion::FromAxisAngle(Real3{1_r, 0_r, 0_r}, 0.5_r); // must be non-zero
  scenePrefab.constraints.articulated3dRotationTarget.push_back(c);

  auto const result = prefab::AddToScene(scenePrefab, scene, {}, test::ExpectOK{});
  auto const constraints = result.Filter(ConstraintType::Articulated3dRotationTarget);
  ASSERT_EQ(1, isize(constraints));
  auto const& reg = static_cast<SceneImpl*>(scene)->GetRegistry();
  auto const entity = GetEntityUnchecked(constraints[0]->GetHandle());
  auto const& data = reg.get<CConstraintData<ConstraintType::Articulated3dRotationTarget>>(entity);
  auto const& target = reg.get<CConstraintTarget<Quaternion, TimeStep::Current>>(entity);
  EXPECT_EQ(c.jointIndex, data.jointIdx);
  EXPECT_NEAR_EQ(Normalize(c.target), target.value);
}

// The prefab assets used by the tests below are not shipped externally.
TEST_IF(MOCHI_INTERNAL, Prefab, JointRotationRangeConstraint_AddToScene) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("test");
  MOCHI_DEFER(context->DestroyScene(scene));

  // JointRotationRange constrains the relative rotation between two rigid actors.
  prefab::ScenePrefab scenePrefab;
  scenePrefab.actors.rigid.push_back(MakeRigidBox("BoxA"));
  auto boxB = MakeRigidBox("BoxB");
  boxB.translation = Real3{1_r, 0_r, 0_r};
  scenePrefab.actors.rigid.push_back(boxB);

  prefab::JointRotationRangeConstraintPrefab c;
  c.actorNameA = "BoxA";
  c.actorNameB = "BoxB";
  c.refFrameRotVec = Real3{0.1_r, -0.2_r, 0.3_r};
  c.angleRangeX = Real2{-0.5_r, 0.25_r};
  c.angleRangeY = Real2{-0.4_r, 0.35_r};
  c.angleRangeZ = Real2{-0.3_r, 0.45_r};
  c.rangeAroundRest = false;
  scenePrefab.constraints.jointRotationRange.push_back(c);

  prefab::LoadShapes(scenePrefab, test::GetAssetsDir(), context, test::ExpectOK{});
  auto const result = prefab::AddToScene(scenePrefab, scene, {}, test::ExpectOK{});
  auto const constraints = result.Filter(ConstraintType::JointRotationRange);
  ASSERT_EQ(1, isize(constraints));
  auto const& reg = static_cast<SceneImpl*>(scene)->GetRegistry();
  auto const entity = GetEntityUnchecked(constraints[0]->GetHandle());
  auto const& data = reg.get<CConstraintData<ConstraintType::JointRotationRange>>(entity);
  EXPECT_NEAR_EQ(Quaternion::FromRotationVector(c.refFrameRotVec), data.q0);
  EXPECT_NEAR_EQ(Quaternion::Identity(), data.qr);
  EXPECT_NEAR_EQ(Real3(c.angleRangeX[0], c.angleRangeY[0], c.angleRangeZ[0]), data.minRotVec);
  EXPECT_NEAR_EQ(Real3(c.angleRangeX[1], c.angleRangeY[1], c.angleRangeZ[1]), data.maxRotVec);
}

TEST_IF(MOCHI_INTERNAL, Prefab, JointRotationTrackingConstraint_AddToScene) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("test");
  MOCHI_DEFER(context->DestroyScene(scene));

  // JointRotationTracking tracks the relative rotation between two rigid actors.
  prefab::ScenePrefab scenePrefab;
  scenePrefab.actors.rigid.push_back(MakeRigidBox("BoxA"));
  auto boxB = MakeRigidBox("BoxB");
  boxB.translation = Real3{1_r, 0_r, 0_r};
  scenePrefab.actors.rigid.push_back(boxB);

  prefab::JointRotationTrackingConstraintPrefab c;
  c.actorNameA = "BoxA";
  c.actorNameB = "BoxB";
  c.refFrameRotVec = Real3{0.1_r, -0.2_r, 0.3_r};
  scenePrefab.constraints.jointRotationTracking.push_back(c);

  prefab::LoadShapes(scenePrefab, test::GetAssetsDir(), context, test::ExpectOK{});
  auto const result = prefab::AddToScene(scenePrefab, scene, {}, test::ExpectOK{});
  auto const constraints = result.Filter(ConstraintType::JointRotationTracking);
  ASSERT_EQ(1, isize(constraints));
  auto const& reg = static_cast<SceneImpl*>(scene)->GetRegistry();
  auto const entity = GetEntityUnchecked(constraints[0]->GetHandle());
  auto const& data = reg.get<CConstraintData<ConstraintType::JointRotationTracking>>(entity);
  EXPECT_NEAR_EQ(Quaternion::FromRotationVector(c.refFrameRotVec), data.q0);
}

TEST_IF(MOCHI_INTERNAL, Prefab, DeformableNodePositionConstraint_AddToScene) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("test");
  MOCHI_DEFER(context->DestroyScene(scene));

  // DeformableNodePosition pins one node of a soft actor to a world position.
  prefab::ScenePrefab scenePrefab;
  scenePrefab.actors.soft.push_back(MakeSoftBox("Soft"));

  prefab::DeformableNodePositionConstraintPrefab c;
  c.actorName = "Soft";
  c.nodeIndex = 1;
  c.position = Real3{0_r, 1_r, 0_r};
  scenePrefab.constraints.deformableNodePosition.push_back(c);

  prefab::LoadShapes(scenePrefab, test::GetAssetsDir(), context, test::ExpectOK{});
  auto const result = prefab::AddToScene(scenePrefab, scene, {}, test::ExpectOK{});
  auto const constraints = result.Filter(ConstraintType::DeformableNodePosition);
  ASSERT_EQ(1, isize(constraints));
  auto const& reg = static_cast<SceneImpl*>(scene)->GetRegistry();
  auto const entity = GetEntityUnchecked(constraints[0]->GetHandle());
  auto const& info = reg.get<CConstraintInfo>(entity);
  auto const& target = reg.get<CConstraintTarget<Real3, TimeStep::Current>>(entity);
  ASSERT_EQ(1, isize(info.actorDofs));
  ASSERT_EQ(3, isize(info.actorDofs[0]));
  EXPECT_EQ(3 * c.nodeIndex + 0, info.actorDofs[0][0]);
  EXPECT_EQ(3 * c.nodeIndex + 1, info.actorDofs[0][1]);
  EXPECT_EQ(3 * c.nodeIndex + 2, info.actorDofs[0][2]);
  EXPECT_NEAR_EQ(c.position, target.value);
}

TEST_IF(MOCHI_INTERNAL, Prefab, DeformableNodeToDeformableNodeConstraint_AddToScene) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("test");
  MOCHI_DEFER(context->DestroyScene(scene));

  // DeformableNodeToDeformableNode ties a node of one soft actor to a node of another.
  prefab::ScenePrefab scenePrefab;
  scenePrefab.actors.soft.push_back(MakeSoftBox("SoftA"));
  auto softB = MakeSoftBox("SoftB");
  softB.translation = Real3{1_r, 0_r, 0_r};
  scenePrefab.actors.soft.push_back(softB);

  prefab::DeformableNodeToDeformableNodeConstraintPrefab c;
  c.actorNameA = "SoftA";
  c.actorNameB = "SoftB";
  c.nodeIndexA = 1;
  c.nodeIndexB = 2;
  c.findClosest = false;
  scenePrefab.constraints.deformableNodeToDeformableNode.push_back(c);

  prefab::LoadShapes(scenePrefab, test::GetAssetsDir(), context, test::ExpectOK{});
  auto const result = prefab::AddToScene(scenePrefab, scene, {}, test::ExpectOK{});
  auto const constraints = result.Filter(ConstraintType::DeformableNodeToDeformableNode);
  ASSERT_EQ(1, isize(constraints));
  auto const& reg = static_cast<SceneImpl*>(scene)->GetRegistry();
  auto const entity = GetEntityUnchecked(constraints[0]->GetHandle());
  auto const& info = reg.get<CConstraintInfo>(entity);
  ASSERT_EQ(2, isize(info.actorDofs));
  ASSERT_EQ(3, isize(info.actorDofs[0]));
  ASSERT_EQ(3, isize(info.actorDofs[1]));
  EXPECT_EQ(3 * c.nodeIndexA + 0, info.actorDofs[0][0]);
  EXPECT_EQ(3 * c.nodeIndexA + 1, info.actorDofs[0][1]);
  EXPECT_EQ(3 * c.nodeIndexA + 2, info.actorDofs[0][2]);
  EXPECT_EQ(3 * c.nodeIndexB + 0, info.actorDofs[1][0]);
  EXPECT_EQ(3 * c.nodeIndexB + 1, info.actorDofs[1][1]);
  EXPECT_EQ(3 * c.nodeIndexB + 2, info.actorDofs[1][2]);
}

TEST_IF(MOCHI_INTERNAL, Prefab, AddToSceneResult_NestedPrefabs) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("test");
  MOCHI_DEFER(context->DestroyScene(scene));

  // Multi-type setup with nested prefabs to confirm Filter() preserves the depth-first
  // ordering in the source list. Each nested prefab and the top-level prefab contribute
  // both rigid and soft actors so the per-type order is well-defined and predictable.
  auto childPrefab = std::make_shared<prefab::ScenePrefab>();
  childPrefab->actors.rigid.push_back(MakeRigidBox("ChildRigid"));
  childPrefab->actors.soft.push_back(MakeSoftBox("ChildSoft"));

  prefab::ScenePrefab topPrefab;
  topPrefab.actors.rigid.push_back(MakeRigidBox("TopRigid1"));
  topPrefab.actors.rigid.push_back(MakeRigidBox("TopRigid2"));
  topPrefab.actors.soft.push_back(MakeSoftBox("TopSoft"));
  {
    auto& nested = topPrefab.prefabs.push_back();
    nested.name = "child";
    // Procedural child with no file backing; leave nested.path empty.
    nested.prefab = childPrefab;
  }

  prefab::LoadShapes(topPrefab, test::GetAssetsDir(), context, test::ExpectOK{});
  prefab::LoadShapes(*childPrefab, test::GetAssetsDir(), context, test::ExpectOK{});

  auto const result = prefab::AddToScene(topPrefab, scene, {}, test::ExpectOK{});
  EXPECT_EQ(5, isize(result.actors));
  EXPECT_EQ(5, scene->GetNumActors());

  // Filter(Rigid): nested child's rigid actor comes first (depth-first), then top-level
  // rigid actors in declaration order.
  auto rigids = result.Filter(ActorType::Rigid);
  ASSERT_EQ(3, isize(rigids));
  EXPECT_STREQ("child/ChildRigid", rigids[0]->GetName());
  EXPECT_STREQ("TopRigid1", rigids[1]->GetName());
  EXPECT_STREQ("TopRigid2", rigids[2]->GetName());

  // Filter(Soft): nested child's soft actor comes first, then top-level soft actor.
  auto softs = result.Filter(ActorType::Soft);
  ASSERT_EQ(2, isize(softs));
  EXPECT_STREQ("child/ChildSoft", softs[0]->GetName());
  EXPECT_STREQ("TopSoft", softs[1]->GetName());
}

// Helper: create a parent prefab that nests childPrefab at the given scale
static prefab::ScenePrefab CreateNestedParentPrefab(
    std::shared_ptr<prefab::ScenePrefab> const& childPrefab,
    real nestedScale) {
  prefab::ScenePrefab parentPrefab;
  auto& nested = parentPrefab.prefabs.push_back();
  nested.name = "child";
  nested.scale = nestedScale;
  nested.prefab = childPrefab;
  return parentPrefab;
}

TEST_IF(MOCHI_INTERNAL, Prefab, RigidActor_NestedPrefabScaleBakesTranslation) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));

  // Child prefab: rigid actor with shapeTranslation and the unit cube shape [0,1]^3
  auto childPrefab = std::make_shared<prefab::ScenePrefab>();
  auto& actor = childPrefab->actors.rigid.push_back();
  actor.name = "box";
  actor.shapeFile = "cube/cube_minimal.mochi.json";
  actor.scale = Real3{1_r, 1_r, 1_r};
  actor.shapeTranslation = Real3{1_r, 2_r, 3_r};

  // Nest the child at scale 2 and load shapes
  auto parentPrefab = CreateNestedParentPrefab(childPrefab, 2_r);
  prefab::LoadShapes(parentPrefab, test::GetAssetPath(""), context, test::ExpectOK{});

  // baked scale = [1,1,1] * 2 = [2,2,2], baked translation = [1,2,3] * 2 = [2,4,6]
  // Unit cube [0,1]^3 scaled by [2,2,2] = [0,2]^3, translated by [2,4,6] = [2,4]x[4,6]x[6,8]
  auto const& shape = childPrefab->actors.rigid[0].shape;
  EXPECT_TRUE(shape.IsValid());
  auto bounds = context->GetShapeAabb(shape, test::ExpectOK{});
  EXPECT_NEAR_EQ(Real3(2_r, 4_r, 6_r), bounds.GetMin());
  EXPECT_NEAR_EQ(Real3(4_r, 6_r, 8_r), bounds.GetMax());
}

TEST_IF(MOCHI_INTERNAL, Prefab, SoftActor_NestedPrefabScaleBakesTranslation) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));

  // Child prefab: soft actor with shapeTranslation and the unit cube shape [0,1]^3
  auto childPrefab = std::make_shared<prefab::ScenePrefab>();
  auto& actor = childPrefab->actors.soft.push_back();
  actor.name = "softBox";
  actor.shapeFile = "cube/cube_minimal.mochi.json";
  actor.scale = Real3{1_r, 1_r, 1_r};
  actor.shapeTranslation = Real3{1_r, 2_r, 3_r};

  // Nest the child at scale 3 and load shapes
  auto parentPrefab = CreateNestedParentPrefab(childPrefab, 3_r);
  prefab::LoadShapes(parentPrefab, test::GetAssetPath(""), context, test::ExpectOK{});

  // baked scale = [1,1,1] * 3 = [3,3,3], baked translation = [1,2,3] * 3 = [3,6,9]
  // Unit cube [0,1]^3 scaled by [3,3,3] = [0,3]^3, translated by [3,6,9] = [3,6]x[6,9]x[9,12]
  auto const& shape = childPrefab->actors.soft[0].shape;
  EXPECT_TRUE(shape.IsValid());
  auto bounds = context->GetShapeAabb(shape, test::ExpectOK{});
  EXPECT_NEAR_EQ(Real3(3_r, 6_r, 9_r), bounds.GetMin());
  EXPECT_NEAR_EQ(Real3(6_r, 9_r, 12_r), bounds.GetMax());
}

TEST_IF(MOCHI_INTERNAL, Prefab, RigidActor_NestedPrefabScaleWithNonUniformActorScale) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));

  // Child prefab: rigid actor with non-uniform actor scale and shapeTranslation
  auto childPrefab = std::make_shared<prefab::ScenePrefab>();
  auto& actor = childPrefab->actors.rigid.push_back();
  actor.name = "box";
  actor.shapeFile = "cube/cube_minimal.mochi.json";
  actor.scale = Real3{1_r, 2_r, 3_r};
  actor.shapeTranslation = Real3{10_r, 10_r, 10_r};

  // Nest the child at scale 2 and load shapes
  auto parentPrefab = CreateNestedParentPrefab(childPrefab, 2_r);
  prefab::LoadShapes(parentPrefab, test::GetAssetPath(""), context, test::ExpectOK{});

  // combined scale = [1,2,3] * 2 = [2,4,6], baked translation = [10,10,10] * 2 = [20,20,20]
  // Unit cube [0,1]^3 scaled by [2,4,6] = [0,2]x[0,4]x[0,6], translated by [20,20,20]
  auto const& shape = childPrefab->actors.rigid[0].shape;
  EXPECT_TRUE(shape.IsValid());
  auto bounds = context->GetShapeAabb(shape, test::ExpectOK{});
  EXPECT_NEAR_EQ(Real3(20_r, 20_r, 20_r), bounds.GetMin());
  EXPECT_NEAR_EQ(Real3(22_r, 24_r, 26_r), bounds.GetMax());
}

TEST_IF(MOCHI_INTERNAL, Prefab, RigidActor_SinglePrefab_UnitScaleBakesTranslation) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));

  // Without nested scale (scaleModifier = 1), translation bakes as-is
  std::string json = R"({
    "actors": {
      "rigid": [{
        "name": "box",
        "shape": "cube/cube_minimal.mochi.json",
        "scale": [1, 1, 1],
        "shapeTranslation": [5, 10, 15]
      }]
    }
  })";

  auto scenePrefab =
      prefab::LoadFromJsonString(json, test::GetAssetPath(""), context, test::ExpectOK{});
  auto const& shape = scenePrefab.actors.rigid[0].shape;
  EXPECT_TRUE(shape.IsValid());

  // scaleModifier=1, scale=[1,1,1] => combined=[1,1,1]
  // translation = [5,10,15] * [1,1,1] = [5,10,15]
  // Cube [0,1]^3 + [5,10,15] = [5,6]x[10,11]x[15,16]
  auto bounds = context->GetShapeAabb(shape, test::ExpectOK{});
  EXPECT_NEAR_EQ(Real3(5_r, 10_r, 15_r), bounds.GetMin());
  EXPECT_NEAR_EQ(Real3(6_r, 11_r, 16_r), bounds.GetMax());
}

TEST_IF(MOCHI_INTERNAL, Prefab, RigidActor_NestedPrefabScaleBakesInertia) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("scene");
  MOCHI_DEFER(context->DestroyScene(scene));

  // Child prefab: rigid actor with signed non-uniform shape scale and explicit mass / COM / MOI /
  // linearVelocity / angularVelocity.
  auto childPrefab = std::make_shared<prefab::ScenePrefab>();
  auto& actor = childPrefab->actors.rigid.push_back();
  actor.name = "box";
  actor.shapeFile = "cube/cube_minimal.mochi.json";
  actor.scale = Real3{-0.5_r, 1_r, 1.5_r};
  actor.mass = 10_r;
  actor.centerOfMass = Real3{1_r, 2_r, 3_r};
  Real3 const inertiaPoint = Real3{1_r, 2_r, 3_r};
  actor.momentOfInertia = PointPairInertia(*actor.mass, inertiaPoint);
  actor.linearVelocity = Real3{0.5_r, -1_r, 2_r};
  actor.angularVelocity = Real3{0.7_r, -0.2_r, 1.1_r};

  // Nest the child at scale 2 and add to scene. The effective scale baked
  // into the shape and explicit dynamics is [-1, 2, 3].
  real const nestedScale = 2_r;
  Real3 const effectiveScale = actor.scale * nestedScale;
  real const volumeScale = Abs(effectiveScale[0] * effectiveScale[1] * effectiveScale[2]);
  auto parentPrefab = CreateNestedParentPrefab(childPrefab, nestedScale);
  prefab::LoadShapes(parentPrefab, test::GetAssetPath(""), context, test::ExpectOK{});
  auto const result = [&] {
    auto suppressWarning = test::SuppressLogWarning();
    return prefab::AddToScene(parentPrefab, scene, {}, test::ExpectOK{});
  }();

  // Mass scales by absolute volume, COM scales by signed length, and MOI uses
  // the full diagonal inertia transform. linearVelocity / angularVelocity are
  // absolute initial conditions and must NOT scale.
  ASSERT_EQ(1, isize(result.actors));
  Actor* a = result.actors[0];
  EXPECT_NEAR_EQ(10_r * volumeScale, a->GetMass(test::ExpectOK{}));
  EXPECT_NEAR_EQ(
      Real3(1_r, 2_r, 3_r) * effectiveScale, a->GetRigidCenterOfMassLocal(test::ExpectOK{}));
  EXPECT_NEAR_EQ(
      PointPairInertia(*actor.mass * volumeScale, inertiaPoint * effectiveScale),
      a->GetRigidMomentOfInertiaLocal(test::ExpectOK{}));
  EXPECT_NEAR_EQ(Real3(0.5_r, -1_r, 2_r), a->GetLinearVelocity(test::ExpectOK{}));
  EXPECT_NEAR_EQ(Real3(0.7_r, -0.2_r, 1.1_r), a->GetAngularVelocity(test::ExpectOK{}));
  // Density (m / V) is preserved: authored mass = 10 on a unit cube implies
  // density = 10; after signed scale the mass and volume both use the same
  // absolute volume factor, so density should still be 10.
  EXPECT_NEAR_EQ(10_r, a->GetDensity(test::ExpectOK{}));
}

// Authors a rigid actor with a density override (no mass override). After
// nested-prefab scaling, density must be unchanged (it is intrinsic) and the
// engine-derived mass must equal density * scaled volume. If we ever
// accidentally scaled density itself, this test would fail.
TEST_IF(MOCHI_INTERNAL, Prefab, RigidActor_NestedPrefabScalePreservesDensity) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("scene");
  MOCHI_DEFER(context->DestroyScene(scene));

  // Unit cube (V_0 = 1) with signed non-uniform scale and density override;
  // no mass override.
  auto childPrefab = std::make_shared<prefab::ScenePrefab>();
  auto& actor = childPrefab->actors.rigid.push_back();
  actor.name = "box";
  actor.shapeFile = "cube/cube_minimal.mochi.json";
  actor.scale = Real3{-0.5_r, 1_r, 1.5_r};
  actor.density = 5_r;

  real const nestedScale = 2_r;
  Real3 const effectiveScale = actor.scale * nestedScale;
  real const volumeScale = Abs(effectiveScale[0] * effectiveScale[1] * effectiveScale[2]);
  auto parentPrefab = CreateNestedParentPrefab(childPrefab, nestedScale);
  prefab::LoadShapes(parentPrefab, test::GetAssetPath(""), context, test::ExpectOK{});
  auto const result = prefab::AddToScene(parentPrefab, scene, {}, test::ExpectOK{});

  ASSERT_EQ(1, isize(result.actors));
  Actor* a = result.actors[0];
  // density is invariant under spatial scale.
  EXPECT_NEAR_EQ(5_r, a->GetDensity(test::ExpectOK{}));
  // mass = density * volume(scaled mesh).
  EXPECT_NEAR_EQ(5_r * volumeScale, a->GetMass(test::ExpectOK{}));
  EXPECT_NEAR_EQ(Real3(-0.5_r, 1_r, 1.5_r), a->GetRigidCenterOfMassLocal(test::ExpectOK{}));
  EXPECT_NEAR_EQ(
      Real6(32.5_r, 0_r, 0_r, 25_r, 0_r, 12.5_r),
      a->GetRigidMomentOfInertiaLocal(test::ExpectOK{}));
}

TEST_IF(MOCHI_INTERNAL, Prefab, RigidActor_PathAddToSceneScalesGeometry) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("scene");
  MOCHI_DEFER(context->DestroyScene(scene));

  // A density-only unit cube: engine-derived mass = density * mesh volume, so mass scaling by the
  // cubed scale proves the mesh (not just the poses) was scaled. The prefabPath-based AddToScene
  // owns the load, so unlike the ScenePrefab overload it honors a non-identity PrefabParams::scale
  // by baking it into geometry. (The ScenePrefab overload rejects scale != 1; see the
  // PrefabParams::scale validation test.)
  prefab::ScenePrefab srcPrefab;
  auto& actor = srcPrefab.actors.rigid.push_back();
  actor.name = "box";
  actor.shapeFile = "cube/cube_minimal.mochi.json";
  actor.density = 5_r;
  actor.translation = Real3{1_r, 0_r, 0_r};
  auto tempDir = CreateTempDirectory("path_addtoscene_scale_test", test::ExpectOK{});
  auto prefabPath = (tempDir.Path() / "box.mochi_scene").string();
  prefab::SaveToJsonFile(srcPrefab, prefabPath, test::ExpectOK{});

  prefab::PrefabParams params;
  params.scale = 2_r;
  real const volumeScale = params.scale * params.scale * params.scale;
  auto const result =
      prefab::AddToScene(prefabPath, test::GetAssetPath(""), scene, params, test::ExpectOK{});

  ASSERT_EQ(1, isize(result.actors));
  Actor* a = result.actors[0];
  EXPECT_NEAR_EQ(5_r, a->GetDensity(test::ExpectOK{})); // density is scale-invariant
  EXPECT_NEAR_EQ(5_r * volumeScale, a->GetMass(test::ExpectOK{})); // mass = density * scaled volume
  // Poses scale too: this overload is now the only path where non-unit pose scaling is reachable.
  EXPECT_NEAR_EQ(Real3(1_r, 0_r, 0_r) * params.scale, a->GetRootTransform().GetTranslation());
}

TEST_IF(
    MOCHI_INTERNAL,
    Prefab,
    RigidPivotPositionConstraint_PathAddToSceneScalesTargetAndLocalPivot) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("scene");
  MOCHI_DEFER(context->DestroyScene(scene));

  prefab::ScenePrefab srcPrefab;
  srcPrefab.actors.rigid.push_back(MakeRigidBox("Box"));
  prefab::RigidPivotPositionConstraintPrefab c;
  c.actorName = "Box";
  c.localPosition = Real3{0.5_r, -0.25_r, 0.125_r};
  c.targetPosition = Real3{-0.2_r, 0.75_r, 0.4_r};
  c.stiffness = 1_r;
  srcPrefab.constraints.rigidPivotPosition.push_back(c);

  auto tempDir = CreateTempDirectory("path_addtoscene_constraint_scale_test", test::ExpectOK{});
  auto prefabPath = (tempDir.Path() / "box.mochi_scene").string();
  prefab::SaveToJsonFile(srcPrefab, prefabPath, test::ExpectOK{});

  prefab::PrefabParams params;
  params.scale = 2_r;
  auto const result =
      prefab::AddToScene(prefabPath, test::GetAssetPath(""), scene, params, test::ExpectOK{});

  auto constraints = result.Filter(ConstraintType::RigidPivotPosition);
  ASSERT_EQ(1, isize(constraints));
  auto const& reg = static_cast<SceneImpl*>(scene)->GetRegistry();
  auto const entity = GetEntityUnchecked(constraints[0]->GetHandle());
  auto const& data = reg.get<CConstraintData<ConstraintType::RigidPivotPosition>>(entity);
  auto const& target = reg.get<CConstraintTarget<Real3, TimeStep::Current>>(entity);
  EXPECT_NEAR_EQ(c.localPosition * params.scale, data.posLocal);
  EXPECT_NEAR_EQ(c.targetPosition * params.scale, target.value);
}

TEST_IF(MOCHI_INTERNAL, Prefab, RigidActor_SinglePrefab_UnitScaleLeavesInertiaUnchanged) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("scene");
  MOCHI_DEFER(context->DestroyScene(scene));

  // No nested scaling; explicit mass / COM / MOI / linearVelocity / angularVelocity
  // should pass through unchanged.
  prefab::ScenePrefab scenePrefab;
  auto& actor = scenePrefab.actors.rigid.push_back();
  actor.name = "box";
  actor.shapeFile = "cube/cube_minimal.mochi.json";
  actor.scale = Real3{1_r, 1_r, 1_r};
  actor.mass = 7_r;
  actor.centerOfMass = Real3{1_r, 2_r, 3_r};
  actor.momentOfInertia = Real6{1_r, 2_r, 3_r, 0_r, 0_r, 0_r};
  actor.linearVelocity = Real3{0.5_r, -1_r, 2_r};
  actor.angularVelocity = Real3{0.7_r, -0.2_r, 1.1_r};

  prefab::LoadShapes(scenePrefab, test::GetAssetPath(""), context, test::ExpectOK{});
  auto const result = [&] {
    auto suppressWarning = test::SuppressLogWarning();
    return prefab::AddToScene(scenePrefab, scene, {}, test::ExpectOK{});
  }();

  ASSERT_EQ(1, isize(result.actors));
  Actor* a = result.actors[0];
  EXPECT_NEAR_EQ(7_r, a->GetMass(test::ExpectOK{}));
  EXPECT_NEAR_EQ(Real3(1_r, 2_r, 3_r), a->GetRigidCenterOfMassLocal(test::ExpectOK{}));
  EXPECT_NEAR_EQ(
      Real6(1_r, 2_r, 3_r, 0_r, 0_r, 0_r), a->GetRigidMomentOfInertiaLocal(test::ExpectOK{}));
  EXPECT_NEAR_EQ(Real3(0.5_r, -1_r, 2_r), a->GetLinearVelocity(test::ExpectOK{}));
  EXPECT_NEAR_EQ(Real3(0.7_r, -0.2_r, 1.1_r), a->GetAngularVelocity(test::ExpectOK{}));
}

TEST_IF(MOCHI_INTERNAL, Prefab, ArticulatedLink_SignedShapeScaleBakesExplicitInertia) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("scene");
  MOCHI_DEFER(context->DestroyScene(scene));

  prefab::ScenePrefab scenePrefab;
  auto& art = scenePrefab.actors.articulated.push_back();
  art.name = "myArt";
  art.joints.resize(1);
  art.joints[0].type = ArticulatedJointType::Free;

  art.links.resize(1);
  auto& link = art.links[0];
  link.name = "boneA";
  link.parentLink = -1;
  link.shapeFile = "cube/cube_minimal.mochi.json";
  link.colliderType = ColliderType::Box;
  link.shapeScale = Real3{-2_r, 3_r, 4_r};
  Real3 const inertiaPoint = Real3{1_r, 2_r, 3_r};
  Real3 const effectiveScale = link.shapeScale;
  real const volumeScale = Abs(effectiveScale[0] * effectiveScale[1] * effectiveScale[2]);
  link.mass = 1_r;
  link.centerOfMass = inertiaPoint;
  link.momentOfInertia = PointPairInertia(*link.mass, inertiaPoint);

  prefab::LoadShapes(scenePrefab, test::GetAssetPath(""), context, test::ExpectOK{});
  {
    auto suppressWarning = test::SuppressLogWarning();
    prefab::AddToScene(scenePrefab, scene, {}, test::ExpectOK{});
  }

  Actor* boneA = scene->GetActor(FindActorByName(scene, "myArt/boneA"));
  ASSERT_NE(nullptr, boneA);
  EXPECT_NEAR_EQ(*link.mass * volumeScale, boneA->GetMass(test::ExpectOK{}));
  EXPECT_NEAR_EQ(inertiaPoint * effectiveScale, boneA->GetRigidCenterOfMassLocal(test::ExpectOK{}));
  EXPECT_NEAR_EQ(
      PointPairInertia(*link.mass * volumeScale, inertiaPoint * effectiveScale),
      boneA->GetRigidMomentOfInertiaLocal(test::ExpectOK{}));
}

TEST_IF(MOCHI_HDF5_AND_INTERNAL, Prefab, ArticulatedLinks_NestedPrefabScaleBakesInertia) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("scene");
  MOCHI_DEFER(context->DestroyScene(scene));

  // Child prefab: 2-link articulation with explicit per-link mass / COM / MOI.
  // Use a non-unit art.scale so that effectiveScale = art.scale * nested.scale
  // exercises both sources of scale (regression test against the original bug
  // of using only the top-level prefab scale). A non-zero joint motor inertia
  // is also set on the revolute joint to pin down the design contract that
  // `joint.inertia` (an actuator characterization) is intentionally NOT
  // scaled by the prefab system — see the comment in mochi_prefab.cpp.
  auto childPrefab = std::make_shared<prefab::ScenePrefab>();
  auto& art = childPrefab->actors.articulated.push_back();
  art.name = "myArt";
  art.scale = 3_r;
  art.joints.resize(2);
  art.joints[0].type = ArticulatedJointType::Free;
  art.joints[1].type = ArticulatedJointType::Revolute;
  art.joints[1].axis = Real3{0_r, 0_r, 1_r};
  art.joints[1].parentLinkFromJoint = TransformRT(Real3{0.5_r, 0_r, 0_r});
  art.joints[1].inertia = 0.25_r;

  art.links.resize(2);
  art.links[0].name = "boneA";
  art.links[0].parentLink = -1;
  art.links[0].shapeFile = "articulated/two_links_revolute/bone_a.mochi.h5";
  art.links[0].colliderType = ColliderType::Box;
  art.links[0].mass = 5_r;
  art.links[0].centerOfMass = Real3{0.1_r, 0.2_r, 0.3_r};
  art.links[0].momentOfInertia = Real6{1_r, 2_r, 3_r, 0_r, 0_r, 0_r};
  art.links[1].name = "boneB";
  art.links[1].parentLink = 0;
  art.links[1].shapeFile = "articulated/two_links_revolute/bone_b.mochi.h5";
  art.links[1].colliderType = ColliderType::Box;
  art.links[1].mass = 8_r;
  art.links[1].centerOfMass = Real3{0.4_r, 0.5_r, 0.6_r};
  art.links[1].momentOfInertia = Real6{4_r, 5_r, 6_r, 0_r, 0_r, 0_r};

  // Nest the child at scale 2 (top-level PrefabParams::scale must be 1). Effective scale baked into
  // explicit link dynamics is es = art.scale * nestedScale.
  real const nestedScale = 2_r;
  prefab::PrefabParams prefabParams;
  real const es = art.scale * nestedScale;
  real const es3 = es * es * es;
  real const es5 = es3 * es * es;
  auto parentPrefab = CreateNestedParentPrefab(childPrefab, nestedScale);
  prefab::LoadShapes(parentPrefab, test::GetAssetPath(""), context, test::ExpectOK{});
  {
    auto suppressWarning = test::SuppressLogWarning();
    prefab::AddToScene(parentPrefab, scene, prefabParams, test::ExpectOK{});
  }

  // Per-link dynamics scale the same way as for top-level rigid actors.
  Actor* boneA = scene->GetActor(FindActorByName(scene, "child/myArt/boneA"));
  Actor* boneB = scene->GetActor(FindActorByName(scene, "child/myArt/boneB"));
  ASSERT_NE(nullptr, boneA);
  ASSERT_NE(nullptr, boneB);

  EXPECT_NEAR_EQ(5_r * es3, boneA->GetMass(test::ExpectOK{}));
  EXPECT_NEAR_EQ(
      Real3(0.1_r, 0.2_r, 0.3_r) * es, boneA->GetRigidCenterOfMassLocal(test::ExpectOK{}));
  EXPECT_NEAR_EQ(
      Real6(es5, 2_r * es5, 3_r * es5, 0_r, 0_r, 0_r),
      boneA->GetRigidMomentOfInertiaLocal(test::ExpectOK{}));

  EXPECT_NEAR_EQ(8_r * es3, boneB->GetMass(test::ExpectOK{}));
  EXPECT_NEAR_EQ(
      Real3(0.4_r, 0.5_r, 0.6_r) * es, boneB->GetRigidCenterOfMassLocal(test::ExpectOK{}));
  EXPECT_NEAR_EQ(
      Real6(4_r * es5, 5_r * es5, 6_r * es5, 0_r, 0_r, 0_r),
      boneB->GetRigidMomentOfInertiaLocal(test::ExpectOK{}));

  // joint.inertia is intentionally not scaled by the prefab system — there is
  // no public getter that exposes it, so this is a smoke check that authoring
  // a non-zero joint.inertia under nested scale does not crash.
}

// Articulated counterpart to RigidActor_NestedPrefabScalePreservesDensity:
// authoring with a per-link density override (no mass override) must preserve
// density under nested-prefab scaling, since the helper never touches density
// and the engine derives mass from density * volume(scaled link shape).
TEST_IF(MOCHI_HDF5_AND_INTERNAL, Prefab, ArticulatedLinks_NestedPrefabScalePreservesDensity) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("scene");
  MOCHI_DEFER(context->DestroyScene(scene));

  auto childPrefab = std::make_shared<prefab::ScenePrefab>();
  auto& art = childPrefab->actors.articulated.push_back();
  art.name = "myArt";
  art.scale = 3_r;
  art.joints.resize(1);
  art.joints[0].type = ArticulatedJointType::Free;
  art.links.resize(1);
  art.links[0].name = "boneA";
  art.links[0].parentLink = -1;
  art.links[0].shapeFile = "articulated/two_links_revolute/bone_a.mochi.h5";
  art.links[0].colliderType = ColliderType::Box;
  art.links[0].density = 5_r; // density-only path; no mass override

  auto parentPrefab = CreateNestedParentPrefab(childPrefab, 2_r);
  prefab::LoadShapes(parentPrefab, test::GetAssetPath(""), context, test::ExpectOK{});
  prefab::AddToScene(parentPrefab, scene, {}, test::ExpectOK{});

  Actor* boneA = scene->GetActor(FindActorByName(scene, "child/myArt/boneA"));
  ASSERT_NE(nullptr, boneA);
  // Density is invariant under spatial scale: any double-application would
  // surface here as a non-5 value.
  EXPECT_NEAR_EQ(5_r, boneA->GetDensity(test::ExpectOK{}));
}

TEST(Prefab, ValidateScale) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("scene");
  MOCHI_DEFER(context->DestroyScene(scene));

  // A minimal rigid actor prefab. Validation runs before any shape resolution or actor creation, so
  // the test does not need to load shapes.
  auto makeMinimalRigidPrefab = []() {
    prefab::ScenePrefab scenePrefab;
    auto& rigid = scenePrefab.actors.rigid.push_back();
    rigid.name = "r";
    rigid.isStatic = true;
    return scenePrefab;
  };

  // PrefabParams::scale on the ScenePrefab AddToScene overload: already-loaded geometry cannot be
  // rescaled here, so this overload rejects otherwise-valid positive finite non-identity scales.
  // The prefabPath overload does bake non-identity scale into geometry; see
  // RigidActor_PathAddToSceneScalesGeometry.
  {
    auto scenePrefab = makeMinimalRigidPrefab();
    prefab::PrefabParams params;
    params.scale = 2_r;
    prefab::AddToScene(scenePrefab, scene, params, test::ExpectNotOK{});

    params.scale = 0.5_r;
    prefab::AddToScene(scenePrefab, scene, params, test::ExpectNotOK{});
  }

  // Near-identity scale values are accepted for round-trip tolerance, but they must instantiate as
  // exact identity because this overload cannot rescale already-loaded geometry.
#if MOCHI_INTERNAL // The prefab asset used in the block below is not shipped externally.
  {
    auto* nearIdentityScene = context->CreateScene("near identity scene");
    MOCHI_DEFER(context->DestroyScene(nearIdentityScene));

    prefab::ScenePrefab scenePrefab;
    auto& rigid = scenePrefab.actors.rigid.push_back();
    rigid.name = "r";
    rigid.shapeFile = "cube/cube_minimal.mochi.json";
    rigid.translation = Real3{1000_r, 0_r, 0_r};
    prefab::LoadShapes(scenePrefab, test::GetAssetPath(""), context, test::ExpectOK{});

    prefab::PrefabParams params;
    params.scale = 1_r + 0.5_r * kDefaultNearEqualEpsilon<real>;
    auto const result =
        prefab::AddToScene(scenePrefab, nearIdentityScene, params, test::ExpectOK{});

    ASSERT_EQ(1, isize(result.actors));
    EXPECT_NEAR_EQ(Real3(1000_r, 0_r, 0_r), result.actors[0]->GetRootTransform().GetTranslation());
  }
#endif // MOCHI_INTERNAL

  // PrefabParams::scale on the prefabPath AddToScene overload: invalid scales are rejected before
  // loading the prefab file.
  {
    auto tempDir = CreateTempDirectory("path_addtoscene_invalid_scale_test", test::ExpectOK{});
    auto const missingPrefabPath = (tempDir.Path() / "missing.mochi_scene").string();
    char const* const expectedError =
        "PrefabParams::scale must be strictly positive and finite. Negative or zero scale is not "
        "supported.";
    real const invalidScales[] = {
        -1_r,
        std::numeric_limits<real>::min(),
        0_r,
        std::numeric_limits<real>::quiet_NaN(),
        std::numeric_limits<real>::infinity(),
    };

    for (real const invalidScale : invalidScales) {
      prefab::PrefabParams params;
      params.scale = invalidScale;
      Error error;
      auto const result =
          prefab::AddToScene(missingPrefabPath, test::GetAssetPath(""), scene, params, error);

      ASSERT_FALSE(error.IsOK());
      EXPECT_STREQ(expectedError, error.GetDescription());
      EXPECT_TRUE(result.actors.empty());
      EXPECT_EQ(0, scene->GetNumActors());
    }
  }

  // RigidActorPrefab::scale: Components must be non-zero and finite. Negative scale is legal
  // mirroring.
  {
    auto scenePrefab = makeMinimalRigidPrefab();
    scenePrefab.actors.rigid[0].scale = Real3{1_r, 0_r, 1_r};
    prefab::AddToScene(scenePrefab, scene, {}, test::ExpectNotOK{});

    scenePrefab.actors.rigid[0].scale = Real3{std::numeric_limits<real>::min(), 1_r, 1_r};
    prefab::AddToScene(scenePrefab, scene, {}, test::ExpectNotOK{});

    scenePrefab.actors.rigid[0].scale = Real3{1_r, std::numeric_limits<real>::quiet_NaN(), 1_r};
    prefab::AddToScene(scenePrefab, scene, {}, test::ExpectNotOK{});

    scenePrefab.actors.rigid[0].scale = Real3{1_r, std::numeric_limits<real>::infinity(), 1_r};
    prefab::AddToScene(scenePrefab, scene, {}, test::ExpectNotOK{});
  }

  // SoftActorPrefab::scale: Components must be strictly positive and finite.
  {
    prefab::ScenePrefab scenePrefab;
    auto& soft = scenePrefab.actors.soft.push_back();
    soft.name = "s";
    soft.scale = Real3{-1_r, -1_r, -1_r};
    prefab::AddToScene(scenePrefab, scene, {}, test::ExpectNotOK{});

    soft.scale = Real3{std::numeric_limits<real>::min(), 1_r, 1_r};
    prefab::AddToScene(scenePrefab, scene, {}, test::ExpectNotOK{});
  }

  // SoftActorPrefab::scale with flowFile: Must be identity.
  {
    prefab::ScenePrefab scenePrefab;
    auto& soft = scenePrefab.actors.soft.push_back();
    soft.name = "s";
    soft.flowFile = "flow.mochi.h5";
    soft.scale = Real3{2_r, 1_r, 1_r};
    prefab::AddToScene(scenePrefab, scene, {}, test::ExpectNotOK{});
  }

  // SoftSkinnedActorPrefab::softParams[].scale: Must be strictly positive, finite, and uniform.
  {
    prefab::ScenePrefab scenePrefab;
    auto& skinned = scenePrefab.actors.softSkinned.push_back();
    auto& soft = skinned.softParams.push_back();
    soft.name = "s";
    soft.scale = Real3{0_r, 1_r, 1_r};
    prefab::AddToScene(scenePrefab, scene, {}, test::ExpectNotOK{});

    soft.scale = Real3{1_r, 2_r, 1_r};
    prefab::AddToScene(scenePrefab, scene, {}, test::ExpectNotOK{});
  }

  // SoftSkinnedActorPrefab::softParams[].scale with flowFile: Must be identity.
  {
    prefab::ScenePrefab scenePrefab;
    auto& skinned = scenePrefab.actors.softSkinned.push_back();
    auto& soft = skinned.softParams.push_back();
    soft.name = "s";
    soft.flowFile = "flow.mochi.h5";
    soft.scale = Real3{2_r, 2_r, 2_r};
    prefab::AddToScene(scenePrefab, scene, {}, test::ExpectNotOK{});
  }

  // ArticulatedActorPrefab::scale: Must be strictly positive and finite.
  {
    prefab::ScenePrefab scenePrefab;
    auto& art = scenePrefab.actors.articulated.push_back();
    art.name = "a";
    art.scale = -1_r;
    art.joints.resize(1);
    art.joints[0].type = ArticulatedJointType::Free;
    art.links.resize(1);
    art.links[0].name = "root";
    art.links[0].parentLink = -1;
    prefab::AddToScene(scenePrefab, scene, {}, test::ExpectNotOK{});

    art.scale = std::numeric_limits<real>::min();
    prefab::AddToScene(scenePrefab, scene, {}, test::ExpectNotOK{});
  }

  // ArticulatedLinkPrefab::shapeScale: Components must be non-zero and finite when shapeFile is
  // set. Negative scale is legal mirroring.
  {
    prefab::ScenePrefab scenePrefab;
    auto& art = scenePrefab.actors.articulated.push_back();
    art.name = "a";
    art.joints.resize(1);
    art.joints[0].type = ArticulatedJointType::Free;
    art.links.resize(1);
    art.links[0].name = "root";
    art.links[0].parentLink = -1;
    art.links[0].shapeFile = "cube/cube_minimal.mochi.json"; // any non-empty
    art.links[0].shapeScale = Real3{0_r, 1_r, 1_r};
    prefab::AddToScene(scenePrefab, scene, {}, test::ExpectNotOK{});

    art.links[0].shapeScale = Real3{std::numeric_limits<real>::min(), 1_r, 1_r};
    prefab::AddToScene(scenePrefab, scene, {}, test::ExpectNotOK{});

#if MOCHI_INTERNAL // The prefab asset used in the block below is not shipped externally.
    art.links[0].shapeScale = Real3{-1_r, 1_r, 1_r};
    prefab::LoadShapes(scenePrefab, test::GetAssetPath(""), context, test::ExpectOK{});
    prefab::AddToScene(scenePrefab, scene, {}, test::ExpectOK{});

    art.links[0].shapeScale = Real3{1_r, 2_r, 3_r};
    prefab::LoadShapes(scenePrefab, test::GetAssetPath(""), context, test::ExpectOK{});
    prefab::AddToScene(scenePrefab, scene, {}, test::ExpectOK{});
#endif // MOCHI_INTERNAL
  }

  // PrefabReference::scale: Must be strictly positive and finite.
  {
    auto nested = std::make_shared<prefab::ScenePrefab>();
    *nested = makeMinimalRigidPrefab();

    prefab::ScenePrefab scenePrefab;
    auto& ref = scenePrefab.prefabs.push_back();
    ref.name = "child";
    ref.scale = -1_r;
    ref.prefab = nested;
    prefab::AddToScene(scenePrefab, scene, {}, test::ExpectNotOK{});
  }
  {
    auto nested = std::make_shared<prefab::ScenePrefab>();
    *nested = makeMinimalRigidPrefab();

    prefab::ScenePrefab scenePrefab;
    auto& ref = scenePrefab.prefabs.push_back();
    ref.name = "child";
    ref.scale = std::numeric_limits<real>::min();
    ref.prefab = nested;
    prefab::AddToScene(scenePrefab, scene, {}, test::ExpectNotOK{});
  }
}

// Duplicate actor names are allowed, but a name claimed by more than one actor is ambiguous and
// cannot be referenced by name. Referencing an ambiguous name from a constraint, pose controller,
// or contact filter fails AddToScene. A duplicated name that is never referenced loads fine.

// Asserts an error is the ambiguous-actor-name failure.
static void ExpectAmbiguousActorNameError(Error const& error) {
  EXPECT_NOT_OK(error);
  EXPECT_NE(error.ToString().find("Ambiguous prefab actor name"), std::string::npos);
}

// Adds the prefab to the scene and asserts it fails with the ambiguous-actor-name error. The
// ambiguity check logs on the Error channel and sets the error, so the log is suppressed around the
// AddToScene call (an unsuppressed error-channel log fails the test).
static void ExpectAmbiguousActorNameError(prefab::ScenePrefab const& scenePrefab, Scene* scene) {
  Error error;
  {
    auto suppressError = test::SuppressLogError();
    prefab::AddToScene(scenePrefab, scene, {}, error);
  }
  ExpectAmbiguousActorNameError(error);
}

// The prefab assets used by the tests below are not shipped externally.
TEST_IF(MOCHI_INTERNAL, Prefab, DuplicateActorName_UnreferencedLoadsSuccessfully) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("test");
  MOCHI_DEFER(context->DestroyScene(scene));

  // Two actors share the name "Cube", but nothing references the name, so the duplicate is benign.
  prefab::ScenePrefab scenePrefab;
  scenePrefab.actors.rigid.push_back(MakeRigidBox("Cube"));
  scenePrefab.actors.rigid.push_back(MakeRigidBox("Cube"));
  prefab::LoadShapes(scenePrefab, test::GetAssetsDir(), context, test::ExpectOK{});

  prefab::AddToScene(scenePrefab, scene, {}, test::ExpectOK{});
  EXPECT_EQ(2, scene->GetNumActors());
}

TEST_IF(MOCHI_INTERNAL, Prefab, DuplicateActorName_ReferencedByContactFilterReturnsError) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));

  auto expectAmbiguousContactFilterReference = [&](bool symmetric) {
    auto* scene = context->CreateScene(symmetric ? "symmetric" : "asymmetric");
    MOCHI_DEFER(context->DestroyScene(scene));

    // Two actors named "Cube"; a contact filter that references "Cube" cannot resolve to one actor.
    prefab::ScenePrefab scenePrefab;
    scenePrefab.actors.rigid.push_back(MakeRigidBox("Cube"));
    scenePrefab.actors.rigid.push_back(MakeRigidBox("Cube"));
    auto& filter = scenePrefab.contactFilter.emplace();
    auto addEntry = [](auto& entries) {
      auto& entry = entries.push_back();
      entry.actors.push_back(DynamicString("Cube"));
      entry.actors.push_back(DynamicString("Cube"));
    };
    if (symmetric) {
      addEntry(filter.actorContactSymmetric.emplace());
    } else {
      addEntry(filter.actorContactAsymmetric.emplace());
    }
    prefab::LoadShapes(scenePrefab, test::GetAssetsDir(), context, test::ExpectOK{});

    ExpectAmbiguousActorNameError(scenePrefab, scene);
  };

  expectAmbiguousContactFilterReference(false);
  expectAmbiguousContactFilterReference(true);
}

TEST_IF(MOCHI_INTERNAL, Prefab, AddToScene_MissingActorNameInContactFilterReturnsError) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));

  // A contact filter that references an actor which does not exist is rejected at AddToScene,
  // matching constraints and pose controllers (and the actor-contact API's own rejection of unknown
  // actors).
  auto expectMissingContactActor = [&](char const* actorA, char const* actorB) {
    auto* scene = context->CreateScene("test");
    MOCHI_DEFER(context->DestroyScene(scene));

    prefab::ScenePrefab scenePrefab;
    scenePrefab.actors.rigid.push_back(MakeRigidBox("Cube"));
    auto& filter = scenePrefab.contactFilter.emplace();
    auto& entries = filter.actorContactSymmetric.emplace();
    auto& entry = entries.push_back();
    entry.actors.push_back(DynamicString(actorA));
    entry.actors.push_back(DynamicString(actorB));
    prefab::LoadShapes(scenePrefab, test::GetAssetsDir(), context, test::ExpectOK{});

    Error error;
    {
      auto suppress = test::SuppressLogError();
      prefab::AddToScene(scenePrefab, scene, {}, error);
    }
    EXPECT_NOT_OK(error);
    EXPECT_NE(
        error.ToString().find("Failed to find actor referenced by contact filter"),
        std::string::npos);
  };

  expectMissingContactActor("Cube", "Missing");
  expectMissingContactActor("Missing", "Cube");
}

TEST_IF(MOCHI_INTERNAL, Prefab, DuplicateActorName_ReferencedByConstraintReturnsError) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("test");
  MOCHI_DEFER(context->DestroyScene(scene));

  // Two actors named "Cube"; a constraint that references "Cube" by name cannot resolve to one
  // actor.
  prefab::ScenePrefab scenePrefab;
  scenePrefab.actors.rigid.push_back(MakeRigidBox("Cube"));
  scenePrefab.actors.rigid.push_back(MakeRigidBox("Cube"));
  scenePrefab.constraints.rigidPivotPosition.push_back(MakePivotPositionConstraint("Cube", 1_r));
  prefab::LoadShapes(scenePrefab, test::GetAssetsDir(), context, test::ExpectOK{});

  ExpectAmbiguousActorNameError(scenePrefab, scene);
}

TEST(Prefab, DuplicateActorName_ReferencedByPoseControllerReturnsError) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("test");
  MOCHI_DEFER(context->DestroyScene(scene));

  // Two articulated actors share the name "Art"; a pose controller that references "Art" cannot
  // resolve to one actor. Shape-less single-link articulations need no shape files.
  auto addArt = [](prefab::ScenePrefab& scenePrefab) {
    auto& art = scenePrefab.actors.articulated.push_back();
    art.name = "Art";
    art.joints.resize(1);
    art.joints[0].type = ArticulatedJointType::Free;
    art.links.resize(1);
    art.links[0].name = "root";
    art.links[0].parentLink = -1;
  };
  prefab::ScenePrefab scenePrefab;
  addArt(scenePrefab);
  addArt(scenePrefab);
  auto& controller = scenePrefab.controllers.push_back();
  controller.articulatedActor = "Art";

  ExpectAmbiguousActorNameError(scenePrefab, scene);
}

// The prefab assets used by the tests below are not shipped externally.
TEST_IF(MOCHI_INTERNAL, Prefab, DuplicateActorName_AmbiguousAcrossNestedPrefabsReturnsError) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("test");
  MOCHI_DEFER(context->DestroyScene(scene));

  // Two nested prefabs are each instantiated as "a" and each contribute a rigid "b", so the
  // combined name "a/b" is claimed by two actors. A constraint referencing "a/b" is therefore
  // ambiguous.
  auto makeInner = [] {
    auto inner = std::make_shared<prefab::ScenePrefab>();
    inner->actors.rigid.push_back(MakeRigidBox("b"));
    return inner;
  };
  prefab::ScenePrefab scenePrefab;
  {
    auto& ref = scenePrefab.prefabs.push_back();
    ref.name = "a";
    ref.prefab = makeInner();
  }
  {
    auto& ref = scenePrefab.prefabs.push_back();
    ref.name = "a";
    ref.prefab = makeInner();
  }
  scenePrefab.constraints.rigidPivotPosition.push_back(MakePivotPositionConstraint("a/b", 1_r));
  prefab::LoadShapes(scenePrefab, test::GetAssetsDir(), context, test::ExpectOK{});

  ExpectAmbiguousActorNameError(scenePrefab, scene);
}

TEST_IF(MOCHI_INTERNAL, Prefab, DuplicateActorName_AmbiguousWithinNestedPrefabReturnsError) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("test");
  MOCHI_DEFER(context->DestroyScene(scene));

  // A single nested prefab "n" internally reuses the name "dup" for two rigid actors. That is
  // benign inside "n" (nothing there references it), but it makes "n/dup" ambiguous within the
  // nested subtree. A parent constraint referencing "n/dup" must fail: MergeActorNameRegistry
  // propagates the nested subtree's ambiguity up to this level. (Contrast
  // DuplicateActorName_AmbiguousAcrossNestedPrefabsReturnsError, where two sibling instances
  // collide at the same combined name.)
  auto inner = std::make_shared<prefab::ScenePrefab>();
  inner->actors.rigid.push_back(MakeRigidBox("dup"));
  inner->actors.rigid.push_back(MakeRigidBox("dup"));

  prefab::ScenePrefab scenePrefab;
  auto& ref = scenePrefab.prefabs.push_back();
  ref.name = "n";
  ref.prefab = inner;
  scenePrefab.constraints.rigidPivotPosition.push_back(MakePivotPositionConstraint("n/dup", 1_r));
  prefab::LoadShapes(scenePrefab, test::GetAssetsDir(), context, test::ExpectOK{});

  ExpectAmbiguousActorNameError(scenePrefab, scene);
}

TEST_IF(MOCHI_INTERNAL, Prefab, DistinctActorNames_ReferencedByContactFilterResolves) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("test");
  MOCHI_DEFER(context->DestroyScene(scene));

  // Distinct names stay unambiguous, so a contact filter that references them resolves and applies.
  prefab::ScenePrefab scenePrefab;
  scenePrefab.actors.rigid.push_back(MakeRigidBox("CubeA"));
  scenePrefab.actors.rigid.push_back(MakeRigidBox("CubeB"));
  auto& filter = scenePrefab.contactFilter.emplace();
  auto& entries = filter.actorContactSymmetric.emplace();
  auto& entry = entries.push_back();
  entry.enable = false;
  entry.actors.push_back(DynamicString("CubeA"));
  entry.actors.push_back(DynamicString("CubeB"));
  prefab::LoadShapes(scenePrefab, test::GetAssetsDir(), context, test::ExpectOK{});

  prefab::AddToScene(scenePrefab, scene, {}, test::ExpectOK{});
  EXPECT_EQ(2, scene->GetNumActors());

  // The reference resolved to the two distinct actors and disabled their mutual contact.
  auto const* sceneImpl = assert_cast<SceneImpl const*>(scene);
  EXPECT_FALSE(sceneImpl->IsActorContactEnabled(
      FindActorByName(scene, "CubeA"), FindActorByName(scene, "CubeB"), test::ExpectOK{}));
}

TEST_IF(MOCHI_INTERNAL, Prefab, DuplicateActorName_SelfContainedNestedInstancesLoad) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("test");
  MOCHI_DEFER(context->DestroyScene(scene));

  // Two nested prefabs are each instantiated as "a" and each contain a rigid "b" plus an internal
  // constraint referencing "b". Each internal reference resolves within its own subtree, where "b"
  // is unambiguous, so both instances load even though the combined name "a/b" is claimed twice at
  // the parent level -- nothing at the parent references it. (Under a single shared registry the
  // second instance's internal reference would see "a/b" as ambiguous and fail.)
  auto makeInner = [] {
    auto inner = std::make_shared<prefab::ScenePrefab>();
    inner->actors.rigid.push_back(MakeRigidBox("b"));
    inner->constraints.rigidPivotPosition.push_back(MakePivotPositionConstraint("b", 1_r));
    return inner;
  };
  prefab::ScenePrefab scenePrefab;
  {
    auto& ref = scenePrefab.prefabs.push_back();
    ref.name = "a";
    ref.prefab = makeInner();
  }
  {
    auto& ref = scenePrefab.prefabs.push_back();
    ref.name = "a";
    ref.prefab = makeInner();
  }
  prefab::LoadShapes(scenePrefab, test::GetAssetsDir(), context, test::ExpectOK{});

  auto const result = prefab::AddToScene(scenePrefab, scene, {}, test::ExpectOK{});

  // Both rigid "b" actors were created and each instance's internal constraint bound to its own
  // "a/b".
  EXPECT_EQ(2, scene->GetNumActors());
  EXPECT_EQ(2, isize(result.constraints));
}

TEST(Prefab, DuplicateActorName_AmbiguousArticulatedLinkNameReturnsError) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));

  // Two articulated actors share the name "art" and each has a link named "link", so the nested
  // link name "art/link" is registered by two different link actors and is ambiguous. A constraint
  // referencing "art/link" therefore fails. (Mochi rejects two links sharing a name within a single
  // articulated actor, so the collision is produced across two actors instead.) Shape-less
  // single-link articulations need no shape files.
  auto* scene = context->CreateScene("articulated_link_name_ambiguity");
  MOCHI_DEFER(context->DestroyScene(scene));
  auto addArt = [](prefab::ScenePrefab& scenePrefab) {
    auto& art = scenePrefab.actors.articulated.push_back();
    art.name = "art";
    art.joints.resize(1);
    art.joints[0].type = ArticulatedJointType::Free;
    art.links.resize(1);
    art.links[0].name = "link";
    art.links[0].parentLink = -1;
  };
  prefab::ScenePrefab scenePrefab;
  addArt(scenePrefab);
  addArt(scenePrefab);
  scenePrefab.constraints.rigidPivotPosition.push_back(
      MakePivotPositionConstraint("art/link", 1_r));

  ExpectAmbiguousActorNameError(scenePrefab, scene);
}

TEST_IF(MOCHI_HDF5_AND_INTERNAL, Prefab, DuplicateActorName_DefaultedNestedSoftNameReturnsError) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));

  // A defaulted nested soft local name is just as referenceable as an explicit local name. If a
  // top-level actor already claims the resulting path, resolving that path must fail as ambiguous
  // instead of silently binding to whichever actor was registered last.
  auto* scene = context->CreateScene("nested_soft_name_ambiguity");
  MOCHI_DEFER(context->DestroyScene(scene));
  auto scenePrefab = prefab::ShallowLoadFromFile(
      test::GetAssetPath("allegro_soft/allegro_soft.mochi_prefab"), test::ExpectOK{});
  ASSERT_EQ(1, isize(scenePrefab.actors.softSkinned));

  auto& skel = scenePrefab.actors.softSkinned[0];
  skel.skeletonParams.name = "Skel";
  ASSERT_FALSE(skel.softParams.empty());
  skel.softParams[0].name = "";
  scenePrefab.actors.rigid.push_back(MakeRigidBox("Skel/soft_0"));
  scenePrefab.constraints.rigidPivotPosition.push_back(
      MakePivotPositionConstraint("Skel/soft_0", 1_r));

  prefab::LoadNestedPrefabs(scenePrefab, test::GetAssetPath(""), test::ExpectOK{});
  prefab::LoadShapes(scenePrefab, test::GetAssetPath(""), context, test::ExpectOK{});

  ExpectAmbiguousActorNameError(scenePrefab, scene);
}

// Malformed-input and error paths for the load/shape-resolution entry points. Deserialization and
// shape-load failures log on the Error channel, which the test fixture would otherwise treat as a
// test failure, so each failing call is wrapped in SuppressLogError.

TEST(Prefab, ShallowLoadFromJsonString_MalformedJsonReturnsError) {
  auto suppress = test::SuppressLogError();
  auto const prefab =
      prefab::ShallowLoadFromJsonString("{ this is not valid json", test::ExpectNotOK{});

  // A parse failure returns a default-constructed (empty) prefab.
  EXPECT_EQ(prefab::ScenePrefab{}, prefab);
}

TEST(Prefab, ShallowLoadFromFile_NonexistentPathReturnsError) {
  auto suppress = test::SuppressLogError();
  auto const prefab =
      prefab::ShallowLoadFromFile("this/path/does/not/exist.mochi_scene", test::ExpectNotOK{});

  // A failed open returns a default-constructed (empty) prefab.
  EXPECT_EQ(prefab::ScenePrefab{}, prefab);
}

TEST(Prefab, LoadShapes_NonexistentShapeFileReturnsError) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));

  // A rigid actor whose shapeFile points at a file that does not exist.
  prefab::ScenePrefab scenePrefab;
  auto& rigid = scenePrefab.actors.rigid.push_back();
  rigid.name = "Box";
  rigid.shapeFile = "does/not/exist.mochi.json";
  rigid.colliderType = ColliderType::Box;

  auto suppress = test::SuppressLogError();
  prefab::LoadShapes(scenePrefab, test::GetAssetsDir(), context, test::ExpectNotOK{});
  EXPECT_FALSE(scenePrefab.actors.rigid[0].shape.IsValid());
}

TEST(Prefab, LoadShapes_EmptyShapeFileReturnsError) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));

  // A rigid actor is not a dummy link, so an empty shapeFile is an error at shape-load time.
  prefab::ScenePrefab scenePrefab;
  auto& rigid = scenePrefab.actors.rigid.push_back();
  rigid.name = "Box";
  rigid.shapeFile = "";
  rigid.colliderType = ColliderType::Box;

  auto suppress = test::SuppressLogError();
  prefab::LoadShapes(scenePrefab, test::GetAssetsDir(), context, test::ExpectNotOK{});
  EXPECT_FALSE(scenePrefab.actors.rigid[0].shape.IsValid());
}

TEST(Prefab, ContactPairParamsOverride_SerializationPreservesSparseFields) {
  auto const verify = [](prefab::ScenePrefab const& scenePrefab) {
    ASSERT_TRUE(scenePrefab.contactPairParamsOverrides.has_value());
    ASSERT_EQ(2, isize(*scenePrefab.contactPairParamsOverrides));

    auto const& pair = (*scenePrefab.contactPairParamsOverrides)[0];
    ASSERT_EQ(2, isize(pair.actors));
    EXPECT_STREQ("ActorA", pair.actors[0].c_str());
    EXPECT_STREQ("ActorB", pair.actors[1].c_str());
    EXPECT_EQ(std::optional<real>{12_r}, pair.paramsOverride.penaltyCoefficient);
    EXPECT_EQ(std::optional<real>{0_r}, pair.paramsOverride.coulombFrictionCoefficient);
    EXPECT_FALSE(pair.paramsOverride.frictionFalloffVel.has_value());
    EXPECT_FALSE(pair.paramsOverride.viscousFrictionCoefficient.has_value());
    EXPECT_FALSE(pair.paramsOverride.normalViscousDampingCoefficient.has_value());

    auto const& self = (*scenePrefab.contactPairParamsOverrides)[1];
    ASSERT_EQ(2, isize(self.actors));
    EXPECT_STREQ("Self", self.actors[0].c_str());
    EXPECT_STREQ("Self", self.actors[1].c_str());
    EXPECT_EQ(std::optional<real>{0_r}, self.paramsOverride.frictionFalloffVel);
    EXPECT_FALSE(self.paramsOverride.penaltyCoefficient.has_value());
  };

  auto scenePrefab = prefab::ShallowLoadFromJsonString(
      R"({
        "contactPairParamsOverrides": [
          {
            "actors": ["ActorA", "ActorB"],
            "paramsOverride": {
              "penaltyCoefficient": 12,
              "coulombFrictionCoefficient": 0
            }
          },
          {
            "actors": ["Self", "Self"],
            "paramsOverride": {"frictionFalloffVel": 0}
          }
        ]
      })",
      test::ExpectOK{});
  verify(scenePrefab);

  auto const json = prefab::SaveToJsonString(scenePrefab, test::ExpectOK{});
  EXPECT_NE(std::string::npos, json.find("\"coulombFrictionCoefficient\": 0"));
  verify(prefab::ShallowLoadFromJsonString(json, test::ExpectOK{}));
}

TEST_IF(
    MOCHI_INTERNAL,
    Prefab,
    ContactPairParamsOverride_AppliesNestedPrefabPathsAndLastEntryWins) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("test");
  MOCHI_DEFER(context->DestroyScene(scene));

  auto child = std::make_shared<prefab::ScenePrefab>();
  child->actors.rigid.push_back(MakeRigidBox("A"));
  child->actors.rigid.push_back(MakeRigidBox("B"));
  auto& childOverrides = child->contactPairParamsOverrides.emplace();
  auto& childEntry = childOverrides.push_back();
  childEntry.actors = {"A", "B"};
  childEntry.paramsOverride.penaltyCoefficient = 3_r;
  auto& childSelf = childOverrides.push_back();
  childSelf.actors = {"A", "A"};
  childSelf.paramsOverride.normalViscousDampingCoefficient = 4_r;

  prefab::ScenePrefab scenePrefab;
  scenePrefab.actors.rigid.push_back(MakeRigidBox("Self"));
  auto& childRef = scenePrefab.prefabs.push_back();
  childRef.name = "child";
  childRef.prefab = child;
  childRef.scale = 2_r;

  auto& overrides = scenePrefab.contactPairParamsOverrides.emplace();
  auto& firstReplacement = overrides.push_back();
  firstReplacement.actors = {"child/A", "child/B"};
  firstReplacement.paramsOverride.penaltyCoefficient = 6_r;
  auto& lastReplacement = overrides.push_back();
  lastReplacement.actors = {"child/B", "child/A"};
  lastReplacement.paramsOverride.viscousFrictionCoefficient = 0.25_r;
  auto& self = overrides.push_back();
  self.actors = {"Self", "Self"};
  self.paramsOverride.frictionFalloffVel = 0_r;

  prefab::LoadShapes(scenePrefab, test::GetAssetsDir(), context, test::ExpectOK{});
  prefab::AddToScene(scenePrefab, scene, {}, test::ExpectOK{});

  auto const actorA = FindActorByName(scene, "child/A");
  auto const actorB = FindActorByName(scene, "child/B");
  auto const selfActor = FindActorByName(scene, "Self");
  auto const pairOverride = scene->GetContactPairParamsOverride(actorB, actorA, test::ExpectOK{});
  EXPECT_FALSE(pairOverride.penaltyCoefficient.has_value());
  EXPECT_EQ(std::optional<real>{0.25_r}, pairOverride.viscousFrictionCoefficient);
  auto const childSelfOverride =
      scene->GetContactPairParamsOverride(actorA, actorA, test::ExpectOK{});
  EXPECT_EQ(std::optional<real>{4_r}, childSelfOverride.normalViscousDampingCoefficient);
  auto const selfOverride =
      scene->GetContactPairParamsOverride(selfActor, selfActor, test::ExpectOK{});
  EXPECT_EQ(std::optional<real>{0_r}, selfOverride.frictionFalloffVel);
}

TEST_IF(MOCHI_INTERNAL, Prefab, ContactPairParamsOverride_RejectsInvalidActorNames) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));

  auto makeEntry = [](prefab::ScenePrefab& scenePrefab, char const* actorA, char const* actorB) {
    auto& entry = scenePrefab.contactPairParamsOverrides.emplace().push_back();
    entry.actors = {actorA, actorB};
    entry.paramsOverride.penaltyCoefficient = 1_r;
  };

  {
    auto* scene = context->CreateScene("missing");
    MOCHI_DEFER(context->DestroyScene(scene));
    prefab::ScenePrefab scenePrefab;
    scenePrefab.actors.rigid.push_back(MakeRigidBox("Cube"));
    makeEntry(scenePrefab, "Cube", "Missing");
    prefab::LoadShapes(scenePrefab, test::GetAssetsDir(), context, test::ExpectOK{});
    auto suppress = test::SuppressLogError();
    prefab::AddToScene(scenePrefab, scene, {}, test::ExpectNotOK{});
  }

  {
    auto* scene = context->CreateScene("ambiguous");
    MOCHI_DEFER(context->DestroyScene(scene));
    prefab::ScenePrefab scenePrefab;
    scenePrefab.actors.rigid.push_back(MakeRigidBox("Cube"));
    scenePrefab.actors.rigid.push_back(MakeRigidBox("Cube"));
    makeEntry(scenePrefab, "Cube", "Cube");
    prefab::LoadShapes(scenePrefab, test::GetAssetsDir(), context, test::ExpectOK{});
    ExpectAmbiguousActorNameError(scenePrefab, scene);
  }

  for (int numActors : {0, 1, 3}) {
    auto* scene = context->CreateScene("wrong-size");
    MOCHI_DEFER(context->DestroyScene(scene));
    prefab::ScenePrefab scenePrefab;
    auto& entry = scenePrefab.contactPairParamsOverrides.emplace().push_back();
    entry.paramsOverride.penaltyCoefficient = 1_r;
    for (int i = 0; i < numActors; ++i) {
      auto const actorName = "Actor" + std::to_string(i);
      scenePrefab.actors.rigid.push_back(MakeRigidBox(actorName));
      entry.actors.push_back(DynamicString(actorName));
    }
    prefab::LoadShapes(scenePrefab, test::GetAssetsDir(), context, test::ExpectOK{});

    Error error;
    prefab::AddToScene(scenePrefab, scene, {}, error);
    EXPECT_STREQ(
        "ContactPairParamsOverrideEntry must have exactly 2 actors.", error.GetDescription());
  }
}
