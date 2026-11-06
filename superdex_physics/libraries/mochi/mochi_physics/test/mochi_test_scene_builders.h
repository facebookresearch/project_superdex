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

#pragma once

#include "mochi_physics_test_fixture.h"

#include <mochi_core/utils/constants.h>
#include <mochi_physics/mochi_physics_experimental.h>

#include <array>
#include <string>
#include <string_view>
#include <tuple>

namespace mochi::test {

inline void AdvanceScene(Scene* scene, int nSteps, double dt = 0.01) {
  for (int i = 0; i < nSteps; i++) {
    scene->Step(dt);
  }
}

// Static rigid ground plane at y = 0, slightly tilted
inline Actor* AddGroundPlane(Scene* scene) {
  auto* context = scene->GetContext();
  RigidActorParams groundParams;
  groundParams.name = "Ground";
  Real3 normal = Real3{0.1_r, 0.9_r, 0_r};
  normal = Normalize(normal);
  groundParams.shape = context->CreatePlaneShape(normal, 0_r, ErrorAssert{});
  groundParams.isStatic = true;
  groundParams.colliderType = ColliderType::Plane;
  return scene->CreateRigidActor(groundParams, test::ExpectOK{});
}

// Dynamic rigid cube just above the ground plane.
inline Actor* AddRigidCube(Scene* scene, real* offsetX = nullptr) {
  auto* context = scene->GetContext();
  real constexpr kBoxScale = 0.2_r;
  // At the default offset, the tilted ground is 11.1 mm high under the cube's -x edge.
  real constexpr kBoxLift = 0.0115_r;
  real const x = offsetX ? *offsetX : 0_r;

  RigidActorParams params;
  params.name = "Box";
  params.layer = "Object";
  params.shape = context->LoadShapeFromFile(
      test::GetAssetPath("cube/cube_fine_mesh.mochi.json"),
      Real3{kBoxScale, kBoxScale, kBoxScale},
      TransformRT::Identity(),
      test::ExpectOK{});
  params.colliderType = ColliderType::Box;
  params.density = 1000_r;
  params.worldFromLocal.SetTranslation(Real3{x - 0.5_r * kBoxScale, kBoxLift, -0.5_r * kBoxScale});
  auto* actor = scene->CreateRigidActor(params, test::ExpectOK{});

  // Add an initial velocity to test that the full velocity is correctly restored.
  actor->SetVelocity({0.06_r, -0.04_r, 0.07_r}, {0.05_r, 0.02_r, 0.03_r}, test::ExpectOK{});

  if (offsetX) {
    *offsetX += 1.1_r * kBoxScale;
  }

  return actor;
}

// Dynamic soft duck just above the ground plane, or just above the top of a support actor and
// moving with it.
inline Actor* AddSoftDuck(Scene* scene, real* offsetX = nullptr, Actor const* support = nullptr) {
  auto* context = scene->GetContext();
  real constexpr kDuckScale = 0.1_r;
  real constexpr kDuckLift = 0.1_r;
  real constexpr kSupportGap = 0.0001_r;
  real const x = offsetX ? *offsetX : 0_r;

  SoftActorParams params;
  params.name = "Duck";
  params.layer = "Object";
  params.shape = context->LoadShapeFromFile(
      test::GetAssetPath("duck/duck_coarse_mesh.mochi.json"),
      Real3{kDuckScale, kDuckScale, kDuckScale},
      TransformRT::Identity(),
      test::ExpectOK{});
  params.material.density = 1000_r;
  params.material.type = SoftMaterialType::NeoHookean;
  params.material.neoHookean.poissonRatio = 0.3_r;
  params.material.neoHookean.youngsModulus = 10000_r;
  Real3 translation{x, kDuckScale + kDuckLift, 0_r};
  if (support) {
    // Centers the duck, about kDuckScale wide from its origin, on the support's top.
    Aabb const bounds = support->GetAabbWorld(test::ExpectOK{});
    translation = bounds.GetCenter() +
        Real3{-0.5_r * kDuckScale, 0.5_r * bounds.GetSize()[1] + kSupportGap, -0.5_r * kDuckScale};
  }
  params.worldFromLocal.SetTranslation(translation);
  auto* actor = scene->CreateSoftActor(params, test::ExpectOK{});
  if (support) {
    actor->SetVelocity(support->GetLinearVelocity(test::ExpectOK{}), Real3{}, test::ExpectOK{});
  }

  if (offsetX) {
    *offsetX += 2.2_r * kDuckScale;
  }

  return actor;
}

// Dynamic soft cube with a contact skin, just above the ground plane.
inline Actor* AddSoftContactSkinCube(Scene* scene, real* offsetX = nullptr) {
  auto* context = scene->GetContext();
  real constexpr kCubeScale = 0.1_r;
  real constexpr kCubeLift = 0.1_r;
  real const x = offsetX ? *offsetX : 0_r;

  SoftActorParams params;
  params.name = "ContactSkinCube";
  params.layer = "Object";
  params.shape = context->LoadShapeFromFile(
      test::GetAssetPath("cube/cube_minimal_contact_skin.mochi.json"),
      Real3{kCubeScale, kCubeScale, kCubeScale},
      TransformRT::Identity(),
      test::ExpectOK{});
  params.useContactSkin = true;
  params.material.density = 1000_r;
  params.material.type = SoftMaterialType::NeoHookean;
  params.material.neoHookean.poissonRatio = 0.3_r;
  params.material.neoHookean.youngsModulus = 10000_r;
  params.worldFromLocal.SetTranslation(Real3{x, kCubeLift, -0.5_r * kCubeScale});
  auto* actor = scene->CreateSoftActor(params, test::ExpectOK{});

  if (offsetX) {
    *offsetX += 1.5_r * kCubeScale;
  }

  return actor;
}

// Dynamic shell duck just above the ground plane.
inline Actor* AddShellDuck(Scene* scene, real* offsetX = nullptr) {
  MOCHI_ASSERT(MOCHI_USE_HDF5, "Adding a shell duck requires HDF5 support");
  auto* context = scene->GetContext();
  real constexpr kDuckScale = 0.1_r;
  real constexpr kDuckLift = 0.1_r;
  real const x = offsetX ? *offsetX : 0_r;

  experimental::ShellActorParams params;
  params.name = "ShellDuck";
  params.layer = "Object";
  params.shape = context->LoadShapeFromFile(
      test::GetAssetPath("duck/duck_surface_mesh_301.mochi.h5"),
      Real3{kDuckScale, kDuckScale, kDuckScale},
      TransformRT::Identity(),
      test::ExpectOK{});
  params.material = experimental::ShellMaterialParamsFrom3dIsotropic(
      10000_r, // youngsModulus3d
      0.3_r, // poissonRatio
      1000_r, // density3d
      0.01_r, // thickness
      test::ExpectOK{});
  params.worldFromLocal.SetTranslation(Real3{x, kDuckScale + kDuckLift, 0_r});
  auto* actor = experimental::CreateShellActor(scene, params, test::ExpectOK{});

  if (offsetX) {
    *offsetX += 2.2_r * kDuckScale;
  }

  return actor;
}

// Dynamic shell box with a contact skin, just above the ground plane.
inline Actor* AddShellContactSkinBox(Scene* scene, real* offsetX = nullptr) {
  MOCHI_ASSERT(MOCHI_USE_HDF5, "Adding a shell box requires HDF5 support");
  auto* context = scene->GetContext();
  // The asset spans about [-0.09, 0.1] x [-0.03, 0.03] x [0, 0.29].
  real constexpr kBoxSizeX = 0.2_r;
  real constexpr kBoxSizeZ = 0.3_r;
  real constexpr kBoxLift = 0.1_r;
  real const x = offsetX ? *offsetX : 0_r;

  experimental::ShellActorParams params;
  params.name = "ContactSkinBox";
  params.layer = "Object";
  params.shape = context->LoadShapeFromFile(
      test::GetAssetPath("shell_contact_skin/cereal_box_subdivided.mochi.h5"),
      Real3{1_r, 1_r, 1_r},
      TransformRT::Identity(),
      test::ExpectOK{});
  params.useContactSkin = true;
  params.material = experimental::ShellMaterialParamsFrom3dIsotropic(
      10000_r, // youngsModulus3d
      0.3_r, // poissonRatio
      1000_r, // density3d
      0.01_r, // thickness
      test::ExpectOK{});
  params.worldFromLocal.SetTranslation(Real3{x + 0.5_r * kBoxSizeX, kBoxLift, -0.5_r * kBoxSizeZ});
  auto* actor = experimental::CreateShellActor(scene, params, test::ExpectOK{});

  if (offsetX) {
    *offsetX += 1.2_r * kBoxSizeX;
  }

  return actor;
}

// Dynamic rod actor: a free rod above the ground plane, optionally with a tubular contact skin.
inline Actor* AddRod(Scene* scene, real* offsetX = nullptr, bool useContactSkin = false) {
  int constexpr kNumNodes = 8;
  real constexpr kRodLength = 0.2_r;
  real constexpr kRodRadius = 0.005_r;
  real constexpr kRodLift = 0.15_r;
  real constexpr kYoungsModulus = 2e6_r;
  real constexpr kShearModulus = 1e6_r;
  real constexpr kDensity = 1000_r;
  real const x = offsetX ? *offsetX : 0_r;

  // Generate nodes and frame axes
  DynamicArray<Real3> nodes;
  DynamicArray<Real3> frameAxes;
  nodes.reserve(kNumNodes);
  for (int i = 0; i < kNumNodes; ++i) {
    real const t = static_cast<real>(i) / static_cast<real>(kNumNodes - 1);
    nodes.emplace_back(0_r, 0_r, t * kRodLength);
    if (i > 0) {
      frameAxes.emplace_back(0_r, 1_r, 0_r);
    }
  }
  // Define parameters for the rod actor
  experimental::RodActorParams params;
  params.name = useContactSkin ? "ContactSkinRod" : "Rod";
  params.layer = "Object";
  if (useContactSkin) {
    int constexpr kNumCrossSectionSegments = 8;
    ModelData const model = experimental::GenerateTubularRodModelData(
        nodes,
        frameAxes,
        kRodRadius,
        kNumCrossSectionSegments,
        /*isClosedLoop=*/false,
        test::ExpectOK{});
    params.shape = scene->GetContext()->CreateModelShape(model, test::ExpectOK{});
  } else {
    params.shape = experimental::CreatePolylineShape(
        scene->GetContext(), nodes, frameAxes, /*isClosedLoop=*/false, test::ExpectOK{});
  }
  params.useContactSkin = useContactSkin;
  params.worldFromLocal.SetTranslation(Real3{x, kRodLift, 0_r});

  // Material properties for a circular cross-section rod
  real const area = kPI * Sqr(kRodRadius);
  params.material.linearDensity = kDensity * area;
  params.material.linearRotationalInertia = kDensity * 0.5_r * kPI * Pow(kRodRadius, 4);
  params.material.axialStiffness = kYoungsModulus * area;
  params.material.flexuralStiffness =
      kYoungsModulus * Real2{0.25_r * kPI * Pow(kRodRadius, 4), 0.25_r * kPI * Pow(kRodRadius, 4)};
  real const torsionConstant = 0.5_r * kPI * Pow(kRodRadius, 4);
  params.material.torsionalStiffness = kShearModulus * torsionConstant;

  auto* actor = experimental::CreateRodActor(scene, params, test::ExpectOK{});

  if (offsetX) {
    *offsetX += 2_r * kRodLength;
  }

  return actor;
}

// Articulated chain
inline Actor* AddNChainBody(
    Context* context,
    Scene* scene,
    std::string_view name,
    real rotationX,
    real posY,
    bool fixedRoot,
    real& offsetX) {
  int constexpr kNumChainLinks = 8;
  real constexpr kChainSize = 0.03_r;
  real constexpr kChainLength = 0.07_r;

  constexpr char const* kBoxMesh = "cube/cube_mesh.mochi.json";
  auto shape = context->LoadShapeFromFile(
      test::GetAssetPath(kBoxMesh),
      Real3{kChainSize, kChainSize, kChainLength},
      TransformRT::Identity(),
      test::ExpectOK{});

  Real3 const jointOffset{0.5_r * kChainSize, 0.5_r * kChainSize, 0_r};
  TransformRT const parentJointFromLink{jointOffset};

  ArticulatedActorParams actorParams;
  actorParams.name = name;
  actorParams.worldFromRoot = TransformRT{
      Quaternion::FromRotationVector(Real3{rotationX, 0_r, 0_r}), Real3{offsetX, posY, 0_r}};
  actorParams.joints.resize(kNumChainLinks);
  actorParams.links.resize(kNumChainLinks);

  for (int i = 0; i < kNumChainLinks; ++i) {
    bool const isRoot = (i == 0);

    auto& joint = actorParams.joints[i];
    joint.type = isRoot ? (fixedRoot ? ArticulatedJointType::Hard : ArticulatedJointType::Free)
                        : ArticulatedJointType::Spherical;
    joint.axis = Real3{-1_r, 0_r, 0_r};
    joint.parentLinkFromJoint = isRoot
        ? TransformRT{-jointOffset}
        : TransformRT{Real3{-jointOffset[0], -jointOffset[1], kChainLength}};
    if (joint.type != ArticulatedJointType::Free) {
      joint.friction = ArticulatedJointFrictionParams{.viscous = 0.1_r};
      joint.inertia = 0.1_r;
    }

    auto& link = actorParams.links[i];
    link.parentLink = i - 1;
    link.parentJointFromLink = parentJointFromLink;
    link.shape = shape;
    link.colliderType = ColliderType::None;
    link.density = 1000_r;
  }

  auto* actor = scene->CreateArticulatedActor(actorParams, test::ExpectOK{});

  offsetX += 1.1_r * kChainSize;

  return actor;
}

// Pose controller for the articulated chain
inline void AddPoseController(Actor* agent) {
  auto const shapeInfo = agent->GetArticulatedShapeInfo(test::ExpectOK{});
  auto const& jointTypes = shapeInfo.jointTypes;

  PoseControllerParams controllerParams;
  controllerParams.jointTracking.resize(
      jointTypes.size(), PoseTrackingParams{.stiffness = 1e3_r, .damping = 1e2_r});
  PoseTrackingParams const kDefault{};
  for (int i = 0; i < isize(jointTypes); ++i) {
    if (jointTypes[i] == ArticulatedJointType::Hard) {
      controllerParams.jointTracking[i] = kDefault;
    }
  }
  agent->AddArticulatedPoseController(controllerParams, test::ExpectOK{});
}

// Complex soft skinned actor
inline Actor* AddSoftAllegro(Context* context, Scene* scene, real* offsetX = nullptr) {
  // Extent of the hand along x around its root, fingertips included.
  real constexpr kHandMinX = -0.075_r;
  real constexpr kHandMaxX = 0.215_r;
  real const x = offsetX ? *offsetX : 0_r;

  int constexpr kNumBonesAllegroSoft = 21;
  std::array<std::string, kNumBonesAllegroSoft> const kObjectTetMeshFilePaths = {
      "allegro_soft/mesh_000.mochi.h5", "allegro_soft/mesh_001.mochi.h5",
      "allegro_soft/mesh_002.mochi.h5", "allegro_soft/mesh_002.mochi.h5",
      "allegro_soft/mesh_002.mochi.h5", "allegro_soft/mesh_005.mochi.h5",
      "allegro_soft/mesh_006.mochi.h5", "allegro_soft/mesh_006.mochi.h5",
      "allegro_soft/mesh_006.mochi.h5", "allegro_soft/mesh_009.mochi.h5",
      "allegro_soft/mesh_010.mochi.h5", "allegro_soft/mesh_010.mochi.h5",
      "allegro_soft/mesh_010.mochi.h5", "allegro_soft/mesh_013.mochi.h5",
      "allegro_soft/mesh_014.mochi.h5", "allegro_soft/mesh_014.mochi.h5",
      "allegro_soft/mesh_014.mochi.h5", "allegro_soft/mesh_017.mochi.h5",
      "allegro_soft/mesh_017.mochi.h5", "allegro_soft/mesh_017.mochi.h5",
      "allegro_soft/mesh_017.mochi.h5"};
  std::array<ShapeHandle, kNumBonesAllegroSoft> linkShapes;
  for (int i = 0; i < kNumBonesAllegroSoft; ++i) {
    linkShapes[i] =
        context->LoadShapeFromFile(GetAssetPath(kObjectTetMeshFilePaths[i]), test::ExpectOK{});
  }

  // Allegro soft hand: 21 links (data from allegro_soft/mesh_transforms.mochi.h5)
  // Parent-relative transforms pre-computed from root-relative rotations and translations.
  Quaternion const kThumbBase{
      0.5213338040146993_r, 0.5213338044812392_r, 0.4777144171407377_r, -0.47771441756824273_r};
  real constexpr kS5 = 0.043619387340501414_r; // sin(2.5°), for ±5° finger splay
  real constexpr kC5 = 0.999048221582942_r; // cos(2.5°)

  ArticulatedActorParams skeletonParams;
  skeletonParams.name = "allegro_hand";
  skeletonParams.worldFromRoot = TransformRT{
      Quaternion(0_r, 0_r, -1_r / Sqrt(2.0_r), 1_r / Sqrt(2.0_r)),
      Real3{x - kHandMinX, 0.5_r, 0_r}};
  skeletonParams.joints = {
      {
          .name = "", //
          .type = ArticulatedJointType::Free, //
          .axis = {0_r, 0_r, 0_r} //
      },
      {
          .name = "joint_12.0",
          .type = ArticulatedJointType::Revolute,
          .parentLinkFromJoint = TransformRT{kThumbBase, Real3{-0.0182_r, 0.019333_r, -0.045987_r}},
          .axis = {-1_r, 0_r, 0_r},
          .minLimit = Real3{-0.263_r, 0_r, 0_r},
          .maxLimit = Real3{-1.396_r, 0_r, 0_r} //
      },
      {
          .name = "joint_8.0",
          .type = ArticulatedJointType::Revolute,
          .parentLinkFromJoint =
              TransformRT{Quaternion{kS5, 0_r, 0_r, kC5}, Real3{0_r, -0.0435_r, -0.001542_r}},
          .axis = {0_r, 0_r, 1_r},
          .minLimit = Real3{0_r, 0_r, -0.47_r},
          .maxLimit = Real3{0_r, 0_r, 0.47_r} //
      },
      {
          .name = "joint_4.0",
          .type = ArticulatedJointType::Revolute,
          .parentLinkFromJoint = TransformRT{Real3{0_r, 0_r, 0.0007_r}},
          .axis = {0_r, 0_r, 1_r},
          .minLimit = Real3{0_r, 0_r, -0.47_r},
          .maxLimit = Real3{0_r, 0_r, 0.47_r} //
      },
      {
          .name = "joint_0.0",
          .type = ArticulatedJointType::Revolute,
          .parentLinkFromJoint =
              TransformRT{Quaternion{-kS5, 0_r, 0_r, kC5}, Real3{0_r, 0.0435_r, -0.001542_r}},
          .axis = {0_r, 0_r, 1_r},
          .minLimit = Real3{0_r, 0_r, -0.47_r},
          .maxLimit = Real3{0_r, 0_r, 0.47_r} //
      },
      {
          .name = "joint_13.0",
          .type = ArticulatedJointType::Revolute,
          .parentLinkFromJoint = TransformRT{Real3{-0.027_r, 0.005_r, 0.0399_r}},
          .axis = {0_r, 0_r, 1_r},
          .minLimit = Real3{0_r, 0_r, -0.105_r},
          .maxLimit = Real3{0_r, 0_r, 1.163_r} //
      },
      {
          .name = "joint_9.0",
          .type = ArticulatedJointType::Revolute,
          .parentLinkFromJoint = TransformRT{Real3{0_r, 0_r, 0.0164_r}},
          .axis = {0_r, 1_r, 0_r},
          .minLimit = Real3{0_r, -0.196_r, 0_r},
          .maxLimit = Real3{0_r, 1.61_r, 0_r} //
      },
      {
          .name = "joint_5.0",
          .type = ArticulatedJointType::Revolute,
          .parentLinkFromJoint = TransformRT{Real3{0_r, 0_r, 0.0164_r}},
          .axis = {0_r, 1_r, 0_r},
          .minLimit = Real3{0_r, -0.196_r, 0_r},
          .maxLimit = Real3{0_r, 1.61_r, 0_r} //
      },
      {
          .name = "joint_1.0",
          .type = ArticulatedJointType::Revolute,
          .parentLinkFromJoint = TransformRT{Real3{0_r, 0_r, 0.0164_r}},
          .axis = {0_r, 1_r, 0_r},
          .minLimit = Real3{0_r, -0.196_r, 0_r},
          .maxLimit = Real3{0_r, 1.61_r, 0_r} //
      },
      {
          .name = "joint_14.0",
          .type = ArticulatedJointType::Revolute,
          .parentLinkFromJoint = TransformRT{Real3{0_r, 0_r, 0.0177_r}},
          .axis = {0_r, 1_r, 0_r},
          .minLimit = Real3{0_r, -0.189_r, 0_r},
          .maxLimit = Real3{0_r, 1.644_r, 0_r} //
      },
      {
          .name = "joint_10.0",
          .type = ArticulatedJointType::Revolute,
          .parentLinkFromJoint = TransformRT{Real3{0_r, 0_r, 0.054_r}},
          .axis = {0_r, 1_r, 0_r},
          .minLimit = Real3{0_r, -0.174_r, 0_r},
          .maxLimit = Real3{0_r, 1.709_r, 0_r} //
      },
      {
          .name = "joint_6.0",
          .type = ArticulatedJointType::Revolute,
          .parentLinkFromJoint = TransformRT{Real3{0_r, 0_r, 0.054_r}},
          .axis = {0_r, 1_r, 0_r},
          .minLimit = Real3{0_r, -0.174_r, 0_r},
          .maxLimit = Real3{0_r, 1.709_r, 0_r} //
      },
      {
          .name = "joint_2.0",
          .type = ArticulatedJointType::Revolute,
          .parentLinkFromJoint = TransformRT{Real3{0_r, 0_r, 0.054_r}},
          .axis = {0_r, 1_r, 0_r},
          .minLimit = Real3{0_r, -0.174_r, 0_r},
          .maxLimit = Real3{0_r, 1.709_r, 0_r} //
      },
      {
          .name = "joint_15.0",
          .type = ArticulatedJointType::Revolute,
          .parentLinkFromJoint = TransformRT{Real3{0_r, 0_r, 0.0514_r}},
          .axis = {0_r, 1_r, 0_r},
          .minLimit = Real3{0_r, -0.162_r, 0_r},
          .maxLimit = Real3{0_r, 1.719_r, 0_r} //
      },
      {
          .name = "joint_11.0",
          .type = ArticulatedJointType::Revolute,
          .parentLinkFromJoint = TransformRT{Real3{0_r, 0_r, 0.0384_r}},
          .axis = {0_r, 1_r, 0_r},
          .minLimit = Real3{0_r, -0.227_r, 0_r},
          .maxLimit = Real3{0_r, 1.618_r, 0_r} //
      },
      {
          .name = "joint_7.0",
          .type = ArticulatedJointType::Revolute,
          .parentLinkFromJoint = TransformRT{Real3{0_r, 0_r, 0.0384_r}},
          .axis = {0_r, 1_r, 0_r},
          .minLimit = Real3{0_r, -0.227_r, 0_r},
          .maxLimit = Real3{0_r, 1.618_r, 0_r} //
      },
      {
          .name = "joint_3.0",
          .type = ArticulatedJointType::Revolute,
          .parentLinkFromJoint = TransformRT{Real3{0_r, 0_r, 0.0384_r}},
          .axis = {0_r, 1_r, 0_r},
          .minLimit = Real3{0_r, -0.227_r, 0_r},
          .maxLimit = Real3{0_r, 1.618_r, 0_r} //
      },
      {
          .name = "joint_15.0_digit2_sensor_base",
          .type = ArticulatedJointType::Hard,
          .parentLinkFromJoint = TransformRT{Real3{0_r, 0_r, 0.0353_r}},
          .axis = {1_r, 0_r, 0_r} //
      },
      {
          .name = "joint_11.0_digit2_sensor_base",
          .type = ArticulatedJointType::Hard,
          .parentLinkFromJoint = TransformRT{Real3{0_r, 0_r, 0.02_r}},
          .axis = {1_r, 0_r, 0_r} //
      },
      {
          .name = "joint_7.0_digit2_sensor_base",
          .type = ArticulatedJointType::Hard,
          .parentLinkFromJoint = TransformRT{Real3{0_r, 0_r, 0.02_r}},
          .axis = {1_r, 0_r, 0_r} //
      },
      {
          .name = "joint_3.0_digit2_sensor_base",
          .type = ArticulatedJointType::Hard,
          .parentLinkFromJoint = TransformRT{Real3{0_r, 0_r, 0.02_r}},
          .axis = {1_r, 0_r, 0_r} //
      },
  };
  skeletonParams.links = {
      {
          .name = "",
          .parentLink = -1,
          .shape = linkShapes[0],
          .layer = "Links",
          .colliderType = ColliderType::None,
          .density = 2000_r,
          .boundaryElementType = ActorBoundaryElementType::P1Q1 //
      },
      {
          .name = "link_12.0",
          .parentLink = 0,
          .shape = linkShapes[1],
          .layer = "Links",
          .colliderType = ColliderType::None,
          .density = 2000_r,
          .boundaryElementType = ActorBoundaryElementType::P1Q1 //
      },
      {
          .name = "link_8.0",
          .parentLink = 0,
          .shape = linkShapes[2],
          .layer = "Links",
          .colliderType = ColliderType::None,
          .density = 2000_r,
          .boundaryElementType = ActorBoundaryElementType::P1Q1 //
      },
      {
          .name = "link_4.0",
          .parentLink = 0,
          .shape = linkShapes[3],
          .layer = "Links",
          .colliderType = ColliderType::None,
          .density = 2000_r,
          .boundaryElementType = ActorBoundaryElementType::P1Q1 //
      },
      {
          .name = "link_0.0",
          .parentLink = 0,
          .shape = linkShapes[4],
          .layer = "Links",
          .colliderType = ColliderType::None,
          .density = 2000_r,
          .boundaryElementType = ActorBoundaryElementType::P1Q1 //
      },
      {
          .name = "link_13.0",
          .parentLink = 1,
          .shape = linkShapes[5],
          .layer = "Links",
          .colliderType = ColliderType::None,
          .density = 2000_r,
          .boundaryElementType = ActorBoundaryElementType::P1Q1 //
      },
      {
          .name = "link_9.0",
          .parentLink = 2,
          .shape = linkShapes[6],
          .layer = "Links",
          .colliderType = ColliderType::None,
          .density = 2000_r,
          .boundaryElementType = ActorBoundaryElementType::P1Q1 //
      },
      {
          .name = "link_5.0",
          .parentLink = 3,
          .shape = linkShapes[7],
          .layer = "Links",
          .colliderType = ColliderType::None,
          .density = 2000_r,
          .boundaryElementType = ActorBoundaryElementType::P1Q1 //
      },
      {
          .name = "link_1.0",
          .parentLink = 4,
          .shape = linkShapes[8],
          .layer = "Links",
          .colliderType = ColliderType::None,
          .density = 2000_r,
          .boundaryElementType = ActorBoundaryElementType::P1Q1 //
      },
      {
          .name = "link_14.0",
          .parentLink = 5,
          .shape = linkShapes[9],
          .layer = "Links",
          .colliderType = ColliderType::None,
          .density = 2000_r,
          .boundaryElementType = ActorBoundaryElementType::P1Q1 //
      },
      {
          .name = "link_10.0",
          .parentLink = 6,
          .shape = linkShapes[10],
          .layer = "Links",
          .colliderType = ColliderType::None,
          .density = 2000_r,
          .boundaryElementType = ActorBoundaryElementType::P1Q1 //
      },
      {
          .name = "link_6.0",
          .parentLink = 7,
          .shape = linkShapes[11],
          .layer = "Links",
          .colliderType = ColliderType::None,
          .density = 2000_r,
          .boundaryElementType = ActorBoundaryElementType::P1Q1 //
      },
      {
          .name = "link_2.0",
          .parentLink = 8,
          .shape = linkShapes[12],
          .layer = "Links",
          .colliderType = ColliderType::None,
          .density = 2000_r,
          .boundaryElementType = ActorBoundaryElementType::P1Q1 //
      },
      {
          .name = "link_15.0",
          .parentLink = 9,
          .shape = linkShapes[13],
          .layer = "Links",
          .colliderType = ColliderType::None,
          .density = 2000_r,
          .boundaryElementType = ActorBoundaryElementType::P1Q1 //
      },
      {
          .name = "link_11.0",
          .parentLink = 10,
          .shape = linkShapes[14],
          .layer = "Links",
          .colliderType = ColliderType::None,
          .density = 2000_r,
          .boundaryElementType = ActorBoundaryElementType::P1Q1 //
      },
      {
          .name = "link_7.0",
          .parentLink = 11,
          .shape = linkShapes[15],
          .layer = "Links",
          .colliderType = ColliderType::None,
          .density = 2000_r,
          .boundaryElementType = ActorBoundaryElementType::P1Q1 //
      },
      {
          .name = "link_3.0",
          .parentLink = 12,
          .shape = linkShapes[16],
          .layer = "Links",
          .colliderType = ColliderType::None,
          .density = 2000_r,
          .boundaryElementType = ActorBoundaryElementType::P1Q1 //
      },
      {
          .name = "link_15.0_digit2_sensor_base",
          .parentLink = 13,
          .shape = linkShapes[17],
          .layer = "Links",
          .colliderType = ColliderType::None,
          .density = 2000_r,
          .boundaryElementType = ActorBoundaryElementType::P1Q1 //
      },
      {
          .name = "link_11.0_digit2_sensor_base",
          .parentLink = 14,
          .shape = linkShapes[18],
          .layer = "Links",
          .colliderType = ColliderType::None,
          .density = 2000_r,
          .boundaryElementType = ActorBoundaryElementType::P1Q1 //
      },
      {
          .name = "link_7.0_digit2_sensor_base",
          .parentLink = 15,
          .shape = linkShapes[19],
          .layer = "Links",
          .colliderType = ColliderType::None,
          .density = 2000_r,
          .boundaryElementType = ActorBoundaryElementType::P1Q1 //
      },
      {
          .name = "link_3.0_digit2_sensor_base",
          .parentLink = 16,
          .shape = linkShapes[20],
          .layer = "Links",
          .colliderType = ColliderType::None,
          .density = 2000_r,
          .boundaryElementType = ActorBoundaryElementType::P1Q1 //
      },
  };

  // Create fingertip shapes.
  // Translations and rotations are copied from the transforms file of the Allegro hand
  int constexpr kNumFingertips = 4;
  std::array<Real3, kNumFingertips> fingertipTranslations = {
      Real3(-0.013200000126536321_r, 0.1607306899222141_r, -0.08546083039339811_r),
      Real3(0.0_r, -0.054725659659519255_r, 0.12676787711477494_r),
      Real3(0.0_r, 0.0_r, 0.12949999999999998_r),
      Real3(0.0_r, 0.054725659659519255_r, 0.12676787711477494_r)};
  std::array<Quaternion, kNumFingertips> fingertipRotations = {
      Quaternion(
          0.5213338040146993_r, 0.5213338044812392_r, 0.4777144171407377_r, -0.47771441756824273_r),
      Quaternion(0.043619387340501414_r, 0.0_r, 0.0_r, 0.999048221582942_r),
      Quaternion(0.0_r, 0.0_r, 0.0_r, 1.0_r),
      Quaternion(-0.043619387340501414_r, 0.0_r, 0.0_r, 0.999048221582942_r)};
  // A small grid exercises constrained and free soft nodes, including one interior node.
  // The bounds approximate a Digit360 sensor in its local frame.
  real constexpr kFingertipSize = 0.025_r;
  Real3 const fingertipOffset{-0.5_r * kFingertipSize, -0.5_r * kFingertipSize, 0.024_r};
  auto [fingertipCoordinates, fingertipConnectivity] = CreateMinimalTetMeshUnitGrid(
      Real3{kFingertipSize, kFingertipSize, kFingertipSize}, Int3{2, 2, 2});
  ModelData fingertipModel;
  auto& fingertipMesh = fingertipModel.mesh.emplace();
  fingertipMesh.nodesPerElement = 4;
  fingertipMesh.connectivity = Flatten(MakeSpan(fingertipConnectivity));
  auto& constrainedNodes = fingertipModel.constrainedNodes.emplace();
  for (int n = 0; n < isize(fingertipCoordinates); ++n) {
    if (fingertipCoordinates[n][2] == 0_r) {
      constrainedNodes.push_back(n);
    }
  }
  std::array<ShapeHandle, kNumFingertips> fingertipShapes;
  for (int i = 0; i < kNumFingertips; ++i) {
    fingertipMesh.coordinates = Flatten(MakeSpan(fingertipCoordinates));
    TransformRT const transform{fingertipRotations[i], fingertipTranslations[i]};
    for (auto& point : Unflatten<Real3>(MakeSpan(fingertipMesh.coordinates))) {
      point = transform.TransformPoint(point + fingertipOffset);
    }
    fingertipShapes[i] = context->CreateModelShape(fingertipModel, test::ExpectOK{});
  }
  // Configure soft-actor params for the 4 fingertips
  std::array<std::string, kNumFingertips> const kFingertipNames = {
      "link_15.0_tip_soft", "link_11.0_tip_soft", "link_7.0_tip_soft", "link_3.0_tip_soft"};
  std::array<SoftActorParams, kNumFingertips> fingertipSoftParams;
  for (int i = 0; i < kNumFingertips; ++i) {
    fingertipSoftParams[i].name = kFingertipNames[i];
    fingertipSoftParams[i].shape = fingertipShapes[i];
    fingertipSoftParams[i].hasGravity = false;
    fingertipSoftParams[i].hasInertia = false;
    fingertipSoftParams[i].hasStress = false;
    fingertipSoftParams[i].layer = "Links";
    fingertipSoftParams[i].boundaryElementType = ActorBoundaryElementType::P1Q1;
  }

  // Configure the soft skinned actor params
  SoftSkinnedActorParams softSkinnedActorParams;
  softSkinnedActorParams.skeletonParams = skeletonParams;
  softSkinnedActorParams.softParams = fingertipSoftParams;
  softSkinnedActorParams.softAttachLinks = {
      "link_15.0_digit2_sensor_base",
      "link_11.0_digit2_sensor_base",
      "link_7.0_digit2_sensor_base",
      "link_3.0_digit2_sensor_base"};
  softSkinnedActorParams.enableCollidingLinks = true;
  softSkinnedActorParams.hasGravity = true;
  softSkinnedActorParams.hasInertia = true;
  softSkinnedActorParams.hasStress = true;

  // Create the soft skinned actor
  auto* actor = scene->CreateSoftSkinnedActor(softSkinnedActorParams, test::ExpectOK{});

  if (offsetX) {
    *offsetX += 1.1_r * (kHandMaxX - kHandMinX);
  }

  return actor;
}

// Creates a scene with various actor types and configures the integration method.
// Returns the created scene and the agent.
inline std::tuple<Scene*, Actor*> CreateSceneActorTypes(
    Context* mochiContext,
    char const* sceneName,
    IntegrationMethod integrationMethod) {
  static_assert(
      static_cast<int>(ActorType::Count) == 6, "Please update this test if adding new actor types");

  auto* scene = mochiContext->CreateScene(sceneName);

  // Configure the integration method.
  SolverParams solverParams = scene->GetSolverParams();
  solverParams.integrationMethod = integrationMethod;
  scene->SetSolverParams(solverParams, test::ExpectOK{});

  real offsetX = 0_r;
  AddGroundPlane(scene);
  Actor* cube = AddRigidCube(scene, &offsetX);
  // The duck rests on the cube, so that two dynamic actors are in contact.
  AddSoftDuck(scene, /*offsetX=*/nullptr, cube);
  AddShellDuck(scene, &offsetX);
  AddRod(scene, &offsetX);
  AddNChainBody(
      mochiContext, scene, "Articulated Chain", 45_r * kRadiansPerDegree, 0.5_r, false, offsetX);
  auto* agent = AddNChainBody(
      mochiContext, scene, "Articulated Agent", -90_r * kRadiansPerDegree, 0_r, true, offsetX);
  AddPoseController(agent);
  AddSoftContactSkinCube(scene, &offsetX);
  AddShellContactSkinBox(scene, &offsetX);
  AddRod(scene, &offsetX, /*useContactSkin=*/true);
  AddSoftAllegro(mochiContext, scene, &offsetX);
  // TODO: Add a ROM actor to the scene.

  return {scene, agent};
}

} // namespace mochi::test
