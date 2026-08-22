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

#include <mochi_core/utils/constants.h>
#include <mochi_physics/mochi_physics_experimental.h>
#include <mochi_physics/src/mochi_capture.h>
#include <mochi_physics/src/mochi_context.h>
#include <mochi_physics/src/mochi_island.h>

#include <algorithm>
#include <array>
#include <concepts>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

using namespace mochi;

// The recorded-scene assets used by these tests are not shipped externally.
#if MOCHI_USE_HDF5 && MOCHI_INTERNAL
#define MOCHI_HDF5_AND_INTERNAL 1
#else
#define MOCHI_HDF5_AND_INTERNAL 0

#endif
using namespace mochi::test;

static void AdvanceScene(Scene* scene, int nSteps) {
  real constexpr kDt = 0.01_r;
  for (int i = 0; i < nSteps; i++) {
    scene->Step(kDt);
  }
};

// Representation of a scene's island partition that ignores island ordering, but preserves actor
// ordering within each island.
using IslandSignature = std::set<std::vector<entt::entity>>;

static IslandSignature GetIslandSignature(Scene* scene) {
  auto const& reg = test::GetRegistry(scene);
  IslandSignature signature;

  reg.view<CIslandMembers const>().each([&](CIslandMembers const& members) {
    signature.emplace(members.actors.begin(), members.actors.end());
  });

  return signature;
}

// Static rigid ground plane at y = 0, slightly tilted
static Actor* AddGroundPlane(Scene* scene) {
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
static Actor* AddRigidCube(Scene* scene, real* offsetX = nullptr) {
  auto* context = scene->GetContext();
  real constexpr kBoxScale = 0.2_r;
  real constexpr kBoxLift = 0.3_r;
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
  actor->SetVelocity({0.6_r, -0.4_r, 0.7_r}, {0.5_r, 0.2_r, 0.3_r}, test::ExpectOK{});

  if (offsetX) {
    *offsetX += 1.1_r * kBoxScale;
  }

  return actor;
}

// Dynamic soft duck just above the ground plane.
static Actor* AddSoftDuck(Scene* scene, real* offsetX = nullptr) {
  auto* context = scene->GetContext();
  real constexpr kDuckScale = 0.1_r;
  real constexpr kDuckLift = 0.1_r;
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
  params.worldFromLocal.SetTranslation(Real3{x, kDuckScale + kDuckLift, 0_r});
  auto* actor = scene->CreateSoftActor(params, test::ExpectOK{});

  if (offsetX) {
    *offsetX += 2.2_r * kDuckScale;
  }

  return actor;
}

// Dynamic shell duck just above the ground plane.
static Actor* AddShellDuck(Scene* scene, real* offsetX = nullptr) {
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

// Dynamic rod actor: a simple cantilever rod above the ground plane.
static Actor* AddRod(Scene* scene, real* offsetX = nullptr) {
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
  params.name = "Rod";
  params.layer = "Object";
  params.shape = experimental::CreatePolylineShape(
      scene->GetContext(), nodes, frameAxes, /*isClosedLoop=*/false, test::ExpectOK{});
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
static Actor* AddNChainBody(
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
static void AddPoseController(Actor* agent) {
  auto const shapeInfo = agent->GetArticulatedShapeInfo(test::ExpectOK{});

  std::vector<PoseTrackingParams> jointTracking(
      shapeInfo.jointTypes.size(), PoseTrackingParams{.stiffness = 1e3_r, .damping = 1e2_r});

  PoseControllerParams controllerParams;
  controllerParams.jointTracking = jointTracking;
  PoseTrackingParams const kDefault{};
  for (int i = 0; i < isize(shapeInfo.jointTypes); ++i) {
    if (shapeInfo.jointTypes[i] == ArticulatedJointType::Hard) {
      controllerParams.jointTracking[i] = kDefault;
    }
  }
  agent->AddArticulatedPoseController(controllerParams, test::ExpectOK{});
}

// Complex soft skinned actor
static Actor* AddSoftAllegro(Context* context, Scene* scene) {
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
      Quaternion(0_r, 0_r, -1_r / Sqrt(2.0_r), 1_r / Sqrt(2.0_r)), Real3{0_r, 0.5_r, 0_r}};
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
  std::string const fingertipFilePath =
      "robots/digit360/Digit360_Sensor_Simplified_Constrained.mochi.h5";
  std::array<ShapeHandle, kNumFingertips> fingertipShapes;
  for (int i = 0; i < kNumFingertips; ++i) {
    fingertipShapes[i] = context->LoadShapeFromFile(
        GetAssetPath(fingertipFilePath),
        Real3{1_r, 1_r, 1_r},
        TransformRT(fingertipRotations[i], fingertipTranslations[i]),
        test::ExpectOK{});
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
  return scene->CreateSoftSkinnedActor(softSkinnedActorParams, test::ExpectOK{});
}

namespace {
using CreateSceneFunc = std::tuple<Scene*, Actor*> (*)(Context*, char const*, IntegrationMethod);

template <typename HandleT>
class RestoreType {
 public:
  using HandleType = HandleT;
  virtual ~RestoreType() = default;
  virtual void
  Initialize(CreateSceneFunc func, Context* mochiContext, IntegrationMethod integrationMethod) = 0;
  virtual void RestoreFromSource(HandleT handle, Error& error) const = 0;
  virtual bool IsEqualState(HandleT handleFrom, HandleT handleTo) const = 0;
  virtual void ReleaseAllStates() = 0;
  virtual void Cleanup(Context* mochiContext) = 0;

  Scene* sceneFrom = nullptr;
  Scene* sceneTo = nullptr;
  Actor* actorFrom = nullptr;
  Actor* actorTo = nullptr;
};

class RestoreTypeStateHandle : public RestoreType<StateHandle> {
 public:
  static bool IsValidState(StateHandle handle) {
    return handle.IsValid();
  }

  static bool IsEqualHandle(StateHandle handleFrom, StateHandle handleTo) {
    return handleFrom == handleTo;
  }

  static bool IsEqualStateSameScene(Scene const* scene, StateHandle handleA, StateHandle handleB) {
    return scene->IsEqualState(handleA, handleB);
  }

  static void
  RestoreState(Scene* scene, StateHandle handle, bool releaseImmediately, Error& error) {
    scene->RestoreState(handle, releaseImmediately, error);
  }

  static StateHandle CaptureState(Scene* scene, Error& error) {
    return scene->CaptureState(error);
  }

  static void ReleaseState(Scene* scene, StateHandle handle) {
    scene->ReleaseState(handle);
  }
};

class RestoreSame : public RestoreTypeStateHandle {
 public:
  void Initialize(CreateSceneFunc func, Context* mochiContext, IntegrationMethod integrationMethod)
      override {
    std::tie(sceneFrom, actorFrom) = func(mochiContext, "Scene", integrationMethod);
    sceneTo = sceneFrom;
    actorTo = actorFrom;
  }

  void RestoreFromSource(StateHandle handle, Error& error) const override {
    sceneFrom->RestoreState(handle, false, error);
  }

  bool IsEqualState(StateHandle handleFrom, StateHandle handleTo) const override {
    return sceneFrom->IsEqualState(handleFrom, handleTo);
  }

  void ReleaseAllStates() override {
    sceneFrom->ReleaseAllStates();
  }

  void Cleanup(Context* mochiContext) override {
    mochiContext->DestroyScene(sceneFrom);
  }
};

class RestoreDifferent : public RestoreTypeStateHandle {
 public:
  void Initialize(CreateSceneFunc func, Context* mochiContext, IntegrationMethod integrationMethod)
      override {
    std::tie(sceneFrom, actorFrom) = func(mochiContext, "SceneFrom", integrationMethod);
    std::tie(sceneTo, actorTo) = func(mochiContext, "SceneTo", integrationMethod);
  }

  void RestoreFromSource(StateHandle handle, Error& error) const override {
    experimental::RestoreStateFromScene(sceneTo, sceneFrom, handle, error);
  }

  bool IsEqualState(StateHandle handleFrom, StateHandle handleTo) const override {
    Error error;
    Span<uint8_t const> bytesFrom =
        assert_cast<SceneImpl const*>(sceneFrom)->FindState(handleFrom, error);
    Span<uint8_t const> bytesTo =
        assert_cast<SceneImpl const*>(sceneTo)->FindState(handleTo, error);
    return error.IsOK() &&
        capture::IsEqualState(
               assert_cast<SceneImpl const*>(sceneFrom)->GetRegistry(), bytesFrom, bytesTo);
  }

  void ReleaseAllStates() override {
    sceneFrom->ReleaseAllStates();
    sceneTo->ReleaseAllStates();
  }

  void Cleanup(Context* mochiContext) override {
    mochiContext->DestroyScene(sceneFrom);
    mochiContext->DestroyScene(sceneTo);
  }
};

struct ByteHandle {
  ByteHandle() {
    value = std::make_shared<DynamicArray<uint8_t>>(DynamicArray<uint8_t>{});
  }
  std::shared_ptr<DynamicArray<uint8_t>> value;
};

class RestoreBytes : public RestoreType<ByteHandle> {
 public:
  static bool IsValidState(ByteHandle handle) {
    return !handle.value->empty();
  }

  static bool IsEqualHandle(ByteHandle handleFrom, ByteHandle handleTo) {
    return handleFrom.value.get() == handleTo.value.get();
  }

  static bool IsEqualStateSameScene(Scene const* scene, ByteHandle handleA, ByteHandle handleB) {
    return capture::IsEqualState(
        assert_cast<SceneImpl const*>(scene)->GetRegistry(), *handleA.value, *handleB.value);
  }

  static void RestoreState(Scene* scene, ByteHandle handle, bool releaseImmediately, Error& error) {
    scene->RestoreStateFromBytes(*handle.value, error);
    if (releaseImmediately) {
      handle.value->clear();
    }
  }

  ByteHandle CaptureState(Scene* scene, Error& error) {
    ByteHandle result{};
    scene->CaptureStateToBytes(*result.value, error);
    _states.push_back(result);
    return result;
  }

  static void ReleaseState(Scene* /*scene*/, ByteHandle handle) {
    handle.value->clear();
  }

  void Initialize(CreateSceneFunc func, Context* mochiContext, IntegrationMethod integrationMethod)
      override {
    std::tie(sceneFrom, actorFrom) = func(mochiContext, "SceneFrom", integrationMethod);
    std::tie(sceneTo, actorTo) = func(mochiContext, "SceneTo", integrationMethod);
  }

  void RestoreFromSource(ByteHandle handle, Error& error) const override {
    sceneTo->RestoreStateFromBytes(*handle.value, error);
  }

  bool IsEqualState(ByteHandle handleFrom, ByteHandle handleTo) const override {
    return capture::IsEqualState(
        assert_cast<SceneImpl const*>(sceneFrom)->GetRegistry(),
        *handleFrom.value,
        *handleTo.value);
  }

  void ReleaseAllStates() override {
    for (auto& state : _states) {
      state.value->clear();
    }
  }

  void Cleanup(Context* mochiContext) override {
    mochiContext->DestroyScene(sceneFrom);
    mochiContext->DestroyScene(sceneTo);
    _states.clear();
  }

 protected:
  DynamicArray<ByteHandle> _states;
};

template <typename RestoreT>
concept RestoreTypeConcept =
    std::derived_from<RestoreT, RestoreType<typename RestoreT::HandleType>>;

enum class RestoreTypeEnum { Same, Different, Bytes };

struct TestParams {
  RestoreTypeEnum restore;
  IntegrationMethod integration;

  // For test naming
  friend std::ostream& operator<<(std::ostream& os, TestParams const& p) {
    char const* restoreStr = p.restore == RestoreTypeEnum::Same ? "Same"
        : p.restore == RestoreTypeEnum::Different               ? "Different"
                                                                : "Bytes";
    char const* integrationStr = p.integration == IntegrationMethod::BackwardEuler ? "BackwardEuler"
        : p.integration == IntegrationMethod::DIRK33                               ? "DIRK33"
                                                                                   : "BDF3";
    return os << restoreStr << "_" << integrationStr;
  }
};

class CaptureRestoreTest : public MochiContextTestBase,
                           public ::testing::WithParamInterface<TestParams> {};

} // namespace

INSTANTIATE_TEST_SUITE_P(
    RestoreAndIntegrationVariations,
    CaptureRestoreTest,
    ::testing::Values(
        TestParams{RestoreTypeEnum::Same, IntegrationMethod::BackwardEuler},
        TestParams{RestoreTypeEnum::Same, IntegrationMethod::DIRK33},
        TestParams{RestoreTypeEnum::Same, IntegrationMethod::BDF3},
        TestParams{RestoreTypeEnum::Different, IntegrationMethod::BackwardEuler},
        TestParams{RestoreTypeEnum::Different, IntegrationMethod::DIRK33},
        TestParams{RestoreTypeEnum::Different, IntegrationMethod::BDF3},
        TestParams{RestoreTypeEnum::Bytes, IntegrationMethod::BackwardEuler},
        TestParams{RestoreTypeEnum::Bytes, IntegrationMethod::DIRK33},
        TestParams{RestoreTypeEnum::Bytes, IntegrationMethod::BDF3}),
    [](::testing::TestParamInfo<TestParams> const& info) {
      std::ostringstream oss;
      oss << info.param;
      return oss.str();
    });

// Creates a scene with a rigid cube actor and configures the integration method.
// Returns the created scene and actor.
static std::tuple<Scene*, Actor*>
CreateCubeScene(Context* mochiContext, char const* sceneName, IntegrationMethod integrationMethod) {
  auto* scene = mochiContext->CreateScene(sceneName);

  // Configure the integration method.
  SolverParams solverParams = scene->GetSolverParams();
  solverParams.integrationMethod = integrationMethod;
  scene->SetSolverParams(solverParams, test::ExpectOK{});

  Actor* actor = AddRigidCube(scene);

  return {scene, actor};
}

// Common test logic for capture-restore-release tests, parameterized by restore type.
template <RestoreTypeConcept RestoreT>
static void TestCaptureRestoreRelease(Context* mochiContext, IntegrationMethod integrationMethod) {
  // Test with 0, ..., numSteps - 1 warm-up steps to test state capturing and restoring during cold
  // start.
  int const numSteps = GetNumSteps(integrationMethod);
  for (int warmUpSteps = 0; warmUpSteps < numSteps; ++warmUpSteps) {
    RestoreT restoreT;
    restoreT.Initialize(CreateCubeScene, mochiContext, integrationMethod);
    Scene* sceneFrom = restoreT.sceneFrom;
    Scene* sceneTo = restoreT.sceneTo;
    Actor* actorFrom = restoreT.actorFrom;
    Actor* actorTo = restoreT.actorTo;

    // Perform warm-up steps on the source scene.
    AdvanceScene(sceneFrom, warmUpSteps);

    // Initial state from source scene
    real height0 = actorFrom->GetRootTransform().GetTranslation()[1];
    auto state0 = restoreT.CaptureState(sceneFrom, test::ExpectOK{});
    EXPECT_TRUE(RestoreT::IsValidState(state0));

    // Advance the source scene so the actor falls some distance
    AdvanceScene(sceneFrom, 10);
    real height1 = actorFrom->GetRootTransform().GetTranslation()[1];
    auto state1 = restoreT.CaptureState(sceneFrom, test::ExpectOK{});
    EXPECT_TRUE(RestoreT::IsValidState(state1));
    EXPECT_FALSE(RestoreT::IsEqualHandle(state1, state0)); // Different handles
    EXPECT_FALSE(RestoreT::IsEqualStateSameScene(sceneFrom, state0, state1)); // Different state
    EXPECT_LT(height1, height0); // Actor fell in -Y direction

    // Restore state0 from sceneFrom to sceneTo
    restoreT.RestoreFromSource(state0, test::ExpectOK{});
    real height0B = actorTo->GetRootTransform().GetTranslation()[1];
    auto state0B = restoreT.CaptureState(sceneTo, test::ExpectOK{});
    EXPECT_TRUE(RestoreT::IsValidState(state0B));
    EXPECT_FALSE(RestoreT::IsEqualHandle(state0, state0B)); // Different handles
    EXPECT_TRUE(restoreT.IsEqualState(state0, state0B)); // Same state (full precision)
    EXPECT_EQ(height0, height0B); // Same height (full precision)

    // Try to restore an invalid state. Should have no effect on the scene.
    auto stateInvalid = typename RestoreT::HandleType{};
    RestoreT::RestoreState(sceneTo, stateInvalid, false, test::ExpectNotOK{});

    // Simulate forward the destination scene
    AdvanceScene(sceneTo, 10);
    real height1B = actorTo->GetRootTransform().GetTranslation()[1];
    auto state1B = restoreT.CaptureState(sceneTo, test::ExpectOK{});
    EXPECT_TRUE(RestoreT::IsValidState(state1B));
    EXPECT_FALSE(RestoreT::IsEqualHandle(state1, state1B)); // Different handles
    EXPECT_TRUE(restoreT.IsEqualState(state1, state1B)); // Same state (full precision)
    EXPECT_EQ(height1, height1B); // Same height (full precision)

    // Now release state0. Further attempts to use it should fail.
    RestoreT::ReleaseState(sceneFrom, state0);
    restoreT.RestoreFromSource(state0, test::ExpectNotOK{}); // stateFrom0 no longer valid
    RestoreT::ReleaseState(sceneFrom, state0); // Redundant call ignored
    EXPECT_EQ(
        height1,
        actorTo->GetRootTransform().GetTranslation()[1]); // No change. Still at state1.

    // We can still restore state0B. This time relase it as well.
    RestoreT::RestoreState(sceneTo, state0B, true, test::ExpectOK{});
    EXPECT_EQ(
        height0, actorTo->GetRootTransform().GetTranslation()[1]); // Back to the initial height

    // We can't restore state0B again because the call to RestoreState also released it.
    AdvanceScene(sceneTo, 10);
    RestoreT::RestoreState(sceneTo, state0B, false, test::ExpectNotOK{});
    EXPECT_EQ(height1, actorTo->GetRootTransform().GetTranslation()[1]); // state0B not restored

    // Release all remaining state handles
    restoreT.ReleaseAllStates();

    // All should now fail
    restoreT.RestoreFromSource(state0, test::ExpectNotOK{});
    RestoreT::RestoreState(sceneTo, state0B, false, test::ExpectNotOK{});
    restoreT.RestoreFromSource(state1, test::ExpectNotOK{});
    RestoreT::RestoreState(sceneTo, state1B, false, test::ExpectNotOK{});
    EXPECT_EQ(height1, actorTo->GetRootTransform().GetTranslation()[1]); // no change

    // Cleanup
    restoreT.Cleanup(mochiContext);
  }
}

TEST_P(CaptureRestoreTest, CaptureRestoreRelease) {
  auto const& params = GetParam();
  switch (params.restore) {
    case RestoreTypeEnum::Same:
      TestCaptureRestoreRelease<RestoreSame>(_mochiContext, params.integration);
      break;
    case RestoreTypeEnum::Different:
      TestCaptureRestoreRelease<RestoreDifferent>(_mochiContext, params.integration);
      break;
    case RestoreTypeEnum::Bytes:
      TestCaptureRestoreRelease<RestoreBytes>(_mochiContext, params.integration);
      break;
  }
}

static void TestWrongScene(Context* mochiContext, IntegrationMethod integrationMethod) {
  // Test with 0, ..., numSteps - 1 warm-up steps to test state capturing and restoring during cold
  // start.
  int const numSteps = GetNumSteps(integrationMethod);
  for (int warmUpSteps = 0; warmUpSteps < numSteps; ++warmUpSteps) {
    // Create two identical scenes
    Scene* sceneA = mochiContext->CreateScene("A");
    Scene* sceneB = mochiContext->CreateScene("B");

    // Configure the integration method for both scenes.
    SolverParams solverParamsA = sceneA->GetSolverParams();
    SolverParams solverParamsB = sceneB->GetSolverParams();
    solverParamsA.integrationMethod = integrationMethod;
    solverParamsB.integrationMethod = integrationMethod;
    sceneA->SetSolverParams(solverParamsA, test::ExpectOK{});
    sceneB->SetSolverParams(solverParamsB, test::ExpectOK{});

    Actor* actorA = AddRigidCube(sceneA);
    Actor* actorB = AddRigidCube(sceneB);

    // Perform warm-up steps.
    AdvanceScene(sceneA, warmUpSteps);
    AdvanceScene(sceneB, warmUpSteps);

    // Capture state from both scenes
    StateHandle stateA0 = sceneA->CaptureState(test::ExpectOK{});
    StateHandle stateB0 = sceneB->CaptureState(test::ExpectOK{});
    real heightA0 = actorA->GetRootTransform().GetTranslation()[1];
    real heightB0 = actorB->GetRootTransform().GetTranslation()[1];
    EXPECT_EQ(heightA0, heightB0);

    // Advance both scenes
    AdvanceScene(sceneA, 1);
    AdvanceScene(sceneB, 1);
    real heightA1 = actorA->GetRootTransform().GetTranslation()[1];
    real heightB1 = actorB->GetRootTransform().GetTranslation()[1];
    EXPECT_EQ(heightA1, heightB1); // Same to full precision
    EXPECT_LT(heightA1, heightA0); // Fell in -Y direction

    // Try to release handles for the wrong scenes. Should do nothing.
    sceneA->ReleaseState(stateB0);
    sceneB->ReleaseState(stateA0);

    // Try to restore state for the wrong scenes. Should fail.
    sceneA->RestoreState(stateB0, false, test::ExpectNotOK{});
    sceneB->RestoreState(stateA0, false, test::ExpectNotOK{});
    EXPECT_EQ(heightA1, actorA->GetRootTransform().GetTranslation()[1]); // no change
    EXPECT_EQ(heightB1, actorB->GetRootTransform().GetTranslation()[1]); // no change

    // Try to compare state for the wrong scenes. Should fail because of unknown handle values.
    EXPECT_FALSE(sceneA->IsEqualState(stateA0, stateB0));
    EXPECT_FALSE(sceneB->IsEqualState(stateA0, stateB0));

    // Restore state to prove we still can
    sceneA->RestoreState(stateA0, false, test::ExpectOK{});
    sceneB->RestoreState(stateB0, false, test::ExpectOK{});
    EXPECT_EQ(heightA0, actorA->GetRootTransform().GetTranslation()[1]);
    EXPECT_EQ(heightB0, actorB->GetRootTransform().GetTranslation()[1]);

    // Cleanup
    mochiContext->DestroyScene(sceneA);
    mochiContext->DestroyScene(sceneB);
  }
}

TEST_F(CaptureRestoreTest, WrongScene) {
  TestWrongScene(_mochiContext, IntegrationMethod::BackwardEuler);
}

TEST_F(CaptureRestoreTest, WrongSceneMultiStageIntegration) {
  TestWrongScene(_mochiContext, IntegrationMethod::DIRK33);
}

TEST_F(CaptureRestoreTest, WrongSceneMultiStepIntegration) {
  TestWrongScene(_mochiContext, IntegrationMethod::BDF3);
}

// Creates a scene with various actor types and configures the integration method.
// Returns the created scene and the agent.
static std::tuple<Scene*, Actor*> CreateSceneActorTypes(
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
  AddRigidCube(scene, &offsetX);
  AddSoftDuck(scene, &offsetX);
  AddShellDuck(scene, &offsetX);
  AddRod(scene, &offsetX);
  AddNChainBody(
      mochiContext, scene, "Articulated Chain", 45_r * kDegreesPerRadian, 0.5_r, false, offsetX);
  auto* agent = AddNChainBody(
      mochiContext, scene, "Articulated Agent", 90_r * kDegreesPerRadian, 0_r, true, offsetX);
  AddPoseController(agent);
  AddSoftAllegro(mochiContext, scene);
  // TODO: Add a ROM actor to the scene.

  return {scene, agent};
}

// Common test logic for ActorTypes tests, parameterized by restore type.
template <RestoreTypeConcept RestoreT>
static void TestActorTypes(Context* mochiContext, IntegrationMethod integrationMethod) {
  // Test with 0, ..., numSteps - 1 warm-up steps to test state capturing and restoring during cold
  // start.
  int const numSteps = GetNumSteps(integrationMethod);
  for (int warmUpSteps = 0; warmUpSteps < numSteps; ++warmUpSteps) {
    RestoreT restoreT;
    restoreT.Initialize(CreateSceneActorTypes, mochiContext, integrationMethod);
    Scene* sceneFrom = restoreT.sceneFrom;
    Scene* sceneTo = restoreT.sceneTo;

    // Perform warm-up steps on the source scene.
    AdvanceScene(sceneFrom, warmUpSteps);

    // Advance the source scene and get the state
    AdvanceScene(sceneFrom, 2);
    auto state1 = restoreT.CaptureState(sceneFrom, test::ExpectOK{});
    EXPECT_TRUE(RestoreT::IsValidState(state1));

    // Advance the source scene and get the state again
    AdvanceScene(sceneFrom, 2);
    auto state2 = restoreT.CaptureState(sceneFrom, test::ExpectOK{});
    EXPECT_TRUE(RestoreT::IsValidState(state2));
    EXPECT_FALSE(RestoreT::IsEqualHandle(state1, state2)); // Different handles

    // Confirm the state is different from state1
    EXPECT_FALSE(RestoreT::IsEqualStateSameScene(sceneFrom, state1, state2));

    // Restore state1 to sceneTo and verify
    restoreT.RestoreFromSource(state1, test::ExpectOK{});
    auto state1B = restoreT.CaptureState(sceneTo, test::ExpectOK{});
    EXPECT_TRUE(RestoreT::IsValidState(state1B));
    EXPECT_FALSE(RestoreT::IsEqualHandle(state1, state1B)); // Different handles
    EXPECT_TRUE(restoreT.IsEqualState(state1, state1B));

    // Advance sceneTo and confirm the state matches state2
    AdvanceScene(sceneTo, 2);
    auto state2B = restoreT.CaptureState(sceneTo, test::ExpectOK{});
    EXPECT_TRUE(RestoreT::IsValidState(state2B));
    EXPECT_FALSE(RestoreT::IsEqualHandle(state2, state2B)); // Different handles
    EXPECT_TRUE(restoreT.IsEqualState(state2, state2B));

    restoreT.Cleanup(mochiContext);
  }
}

TEST_IF_P(MOCHI_HDF5_AND_INTERNAL, CaptureRestoreTest, ActorTypes) {
  auto const& params = GetParam();

// Test times out in debug builds with multi-stage/multi-step integration methods.
#if MOCHI_DEBUG
  if (params.integration != IntegrationMethod::BackwardEuler) {
    return;
  }
#endif
  // Disable warnings about solution explosion. DIRK33 is unstable for this problem, but that's OK
  // for the purpose of this test.
  bool const wasWarningEnabled = IsLogChannelEnabled(LogChannel::Warning);
  if (params.integration == IntegrationMethod::DIRK33) {
    EnableLogChannel(LogChannel::Warning, false);
  }

  switch (params.restore) {
    case RestoreTypeEnum::Same:
      TestActorTypes<RestoreSame>(_mochiContext, params.integration);
      break;
    case RestoreTypeEnum::Different:
      TestActorTypes<RestoreDifferent>(_mochiContext, params.integration);
      break;
    case RestoreTypeEnum::Bytes:
      TestActorTypes<RestoreBytes>(_mochiContext, params.integration);
      break;
  }

  if (params.integration == IntegrationMethod::DIRK33) {
    EnableLogChannel(LogChannel::Warning, wasWarningEnabled);
  }
}

// Creates a scene with actors for island preservation testing and configures the integration
// method. Returns the created scene and the rigid cube actor.
static std::tuple<Scene*, Actor*> CreateSceneIslandPreservation(
    Context* mochiContext,
    char const* sceneName,
    IntegrationMethod integrationMethod) {
  auto* scene = mochiContext->CreateScene(sceneName);

  // Configure the integration method.
  SolverParams solverParams = scene->GetSolverParams();
  solverParams.integrationMethod = integrationMethod;
  scene->SetSolverParams(solverParams, test::ExpectOK{});

  real offsetX = 0_r;
  AddGroundPlane(scene);
  Actor* rigidCube = AddRigidCube(scene, &offsetX);
  offsetX += 50_r; // Far apart so that they are on different islands.
  AddSoftDuck(scene, &offsetX);
#if MOCHI_USE_HDF5
  offsetX += 50_r;
  AddShellDuck(scene, &offsetX);
#endif

  return {scene, rigidCube};
}

// Common test logic for IslandPreservation tests, parameterized by restore type.
template <RestoreTypeConcept RestoreT>
static void TestIslandPreservation(Context* mochiContext, IntegrationMethod integrationMethod) {
  RestoreT restoreT;
  restoreT.Initialize(CreateSceneIslandPreservation, mochiContext, integrationMethod);
  Scene* sceneFrom = restoreT.sceneFrom;
  Scene* sceneTo = restoreT.sceneTo;

  // Get the initial state
  auto state0 = restoreT.CaptureState(sceneFrom, test::ExpectOK{});
  EXPECT_TRUE(RestoreT::IsValidState(state0));

  // Advance the source scene and record its island partition.
  AdvanceScene(sceneFrom, 1);
  auto const expectedSignature = GetIslandSignature(sceneFrom);
  EXPECT_GT(isize(expectedSignature), 1);

  // Deliberately make the target scene's island state different before restore. This keeps the test
  // non-vacuous without depending on conservative-bound heuristics changing the scene's island
  // count naturally over time.
  sceneTo->SetForceSingleIsland(true);
  AdvanceScene(sceneTo, 10);
  auto const staleSignature = GetIslandSignature(sceneTo);
  EXPECT_EQ(1, isize(staleSignature));
  EXPECT_NE(expectedSignature, staleSignature);
  sceneTo->SetForceSingleIsland(false);

  // Restore state0 to sceneTo and confirm that the island partition is restored.
  restoreT.RestoreFromSource(state0, test::ExpectOK{});
  AdvanceScene(sceneTo, 1);
  EXPECT_EQ(expectedSignature, GetIslandSignature(sceneTo));

  restoreT.Cleanup(mochiContext);
}

// The Duck meshes used by this test are not shipped externally.
TEST_IF_P(MOCHI_INTERNAL, CaptureRestoreTest, IslandPreservation) {
  auto const& params = GetParam();

  switch (params.restore) {
    case RestoreTypeEnum::Same:
      TestIslandPreservation<RestoreSame>(_mochiContext, params.integration);
      break;
    case RestoreTypeEnum::Different:
      TestIslandPreservation<RestoreDifferent>(_mochiContext, params.integration);
      break;
    case RestoreTypeEnum::Bytes:
      TestIslandPreservation<RestoreBytes>(_mochiContext, params.integration);
      break;
  }
}

static void TestSceneSharing(Context* mochiContext, IntegrationMethod integrationMethod) {
  // Run three simulations on a scene, and confirm that they produce the same results.
  // 1) 10 steps from scratch.
  // 2) 2 batches of 5 steps, alternating two scenes.
  // 3) 5 batches of 2 steps, alternating two scenes.

  // Test with 0, ..., numSteps - 1 warm-up steps to test state capturing and restoring during
  // cold start.
  int const numSteps = GetNumSteps(integrationMethod);
  for (int warmUpSteps = 0; warmUpSteps < numSteps; ++warmUpSteps) {
    auto* scene = mochiContext->CreateScene("Scene");

    // Configure the integration method.
    SolverParams solverParams = scene->GetSolverParams();
    solverParams.integrationMethod = integrationMethod;
    scene->SetSolverParams(solverParams, test::ExpectOK{});

    real offsetX = 0_r;
    AddGroundPlane(scene);
    AddRigidCube(scene, &offsetX);
    AddSoftDuck(scene, &offsetX);
#if MOCHI_USE_HDF5
    AddShellDuck(scene, &offsetX);
#endif

    // Perform warm-up steps.
    AdvanceScene(scene, warmUpSteps);

    // Get the initial state
    StateHandle state0 = scene->CaptureState(test::ExpectOK{});
    EXPECT_TRUE(state0.IsValid());

    // Run 1
    AdvanceScene(scene, 10);
    StateHandle state1 = scene->CaptureState(test::ExpectOK{});
    auto time1 = scene->GetTotalSimulationTime();

    // Run 2
    StateHandle stateA = state0;
    StateHandle stateB = state0;
    for (int i = 0; i < 2; ++i) {
      scene->RestoreState(stateA, false, test::ExpectOK{});
      AdvanceScene(scene, 5);
      stateA = scene->CaptureState(test::ExpectOK{});
      scene->RestoreState(stateB, false, test::ExpectOK{});
      AdvanceScene(scene, 5);
      stateB = scene->CaptureState(test::ExpectOK{});
    }
    StateHandle state2 = scene->CaptureState(test::ExpectOK{});
    auto time2 = scene->GetTotalSimulationTime();

    // Run 3
    stateA = state0;
    stateB = state0;
    for (int i = 0; i < 5; ++i) {
      scene->RestoreState(stateA, false, test::ExpectOK{});
      AdvanceScene(scene, 2);
      stateA = scene->CaptureState(test::ExpectOK{});
      scene->RestoreState(stateB, false, test::ExpectOK{});
      AdvanceScene(scene, 2);
      stateB = scene->CaptureState(test::ExpectOK{});
    }
    StateHandle state3 = scene->CaptureState(test::ExpectOK{});
    auto time3 = scene->GetTotalSimulationTime();

    // Compare
    EXPECT_TRUE(scene->IsEqualState(state2, state1));
    EXPECT_TRUE(scene->IsEqualState(state3, state1));
    EXPECT_EQ(time2, time1);
    EXPECT_EQ(time3, time1);

    mochiContext->DestroyScene(scene);
  }
}

// The Duck meshes used by TestSceneSharing are not shipped externally.
TEST_IF_F(MOCHI_INTERNAL, CaptureRestoreTest, SceneSharing) {
  TestSceneSharing(_mochiContext, IntegrationMethod::BackwardEuler);
}

TEST_IF_F(MOCHI_INTERNAL, CaptureRestoreTest, SceneSharingMultiStageIntegration) {
  TestSceneSharing(_mochiContext, IntegrationMethod::DIRK33);
}

TEST_IF_F(MOCHI_INTERNAL, CaptureRestoreTest, SceneSharingMultiStepIntegration) {
  TestSceneSharing(_mochiContext, IntegrationMethod::BDF3);
}

// Creates a scene with an articulated agent with pose controller and configures the integration
// method. Returns the created scene and agent.
static std::tuple<Scene*, Actor*> CreateScenePoseController(
    Context* mochiContext,
    char const* sceneName,
    IntegrationMethod integrationMethod) {
  auto* scene = mochiContext->CreateScene(sceneName);

  // Configure the integration method.
  SolverParams solverParams = scene->GetSolverParams();
  solverParams.integrationMethod = integrationMethod;
  scene->SetSolverParams(solverParams, test::ExpectOK{});

  real offsetX = 0_r;
  auto* agent = AddNChainBody(
      mochiContext, scene, "Articulated Agent", 90_r * kDegreesPerRadian, 0_r, true, offsetX);
  AddPoseController(agent);

  return {scene, agent};
}

// Common test logic for PoseController tests, parameterized by restore type.
template <RestoreTypeConcept RestoreT>
static void TestPoseController(Context* mochiContext, IntegrationMethod integrationMethod) {
  // Test with 0, ..., numSteps - 1 warm-up steps to test state capturing and restoring during
  // cold start.
  int const numSteps = GetNumSteps(integrationMethod);
  for (int warmUpSteps = 0; warmUpSteps < numSteps; ++warmUpSteps) {
    RestoreT restoreT;
    restoreT.Initialize(CreateScenePoseController, mochiContext, integrationMethod);
    Scene* sceneFrom = restoreT.sceneFrom;
    Scene* sceneTo = restoreT.sceneTo;
    Actor* agentFrom = restoreT.actorFrom;
    Actor* agentTo = restoreT.actorTo;

    std::vector<real> target(agentFrom->GetNumDofs());

    auto applyTargetAndAdvance = [&](Scene* scene, Actor* agent, int stepBegin, int stepEnd) {
      for (int i = stepBegin; i < stepEnd; ++i) {
        std::fill(target.begin(), target.end(), 0.01_r * i);
        agent->SetArticulatedTargetPose(target, test::ExpectOK{});
        AdvanceScene(scene, 1);
      }
    };

    // Perform warm-up steps on sceneFrom.
    applyTargetAndAdvance(sceneFrom, agentFrom, 0, warmUpSteps);

    // Advance sceneFrom and get the state
    applyTargetAndAdvance(sceneFrom, agentFrom, warmUpSteps, 10 + warmUpSteps);
    auto state1 = restoreT.CaptureState(sceneFrom, test::ExpectOK{});
    EXPECT_TRUE(RestoreT::IsValidState(state1));

    // Advance sceneFrom and get the state again
    applyTargetAndAdvance(sceneFrom, agentFrom, 10 + warmUpSteps, 20 + warmUpSteps);
    auto state2 = restoreT.CaptureState(sceneFrom, test::ExpectOK{});
    EXPECT_TRUE(RestoreT::IsValidState(state2));

    // Restore state1 to sceneTo and confirm it works
    restoreT.RestoreFromSource(state1, test::ExpectOK{});
    auto stateTest = restoreT.CaptureState(sceneTo, test::ExpectOK{});
    EXPECT_TRUE(RestoreT::IsValidState(stateTest));
    EXPECT_FALSE(RestoreT::IsEqualHandle(state1, stateTest)); // Different handles
    EXPECT_TRUE(restoreT.IsEqualState(state1, stateTest)); // Same state data

    // Advance sceneTo and confirm the state matches state2
    applyTargetAndAdvance(sceneTo, agentTo, 10 + warmUpSteps, 20 + warmUpSteps);
    restoreT.ReleaseState(sceneTo, stateTest); // Release before overwriting
    stateTest = restoreT.CaptureState(sceneTo, test::ExpectOK{});
    EXPECT_TRUE(RestoreT::IsValidState(stateTest));
    EXPECT_FALSE(RestoreT::IsEqualHandle(state2, stateTest)); // Different handles
    EXPECT_TRUE(restoreT.IsEqualState(state2, stateTest)); // Same state data

    restoreT.Cleanup(mochiContext);
  }
}

TEST_IF_P(MOCHI_USE_HDF5, CaptureRestoreTest, PoseController) {
  auto const& params = GetParam();

  switch (params.restore) {
    case RestoreTypeEnum::Same:
      TestPoseController<RestoreSame>(_mochiContext, params.integration);
      break;
    case RestoreTypeEnum::Different:
      TestPoseController<RestoreDifferent>(_mochiContext, params.integration);
      break;
    case RestoreTypeEnum::Bytes:
      TestPoseController<RestoreBytes>(_mochiContext, params.integration);
      break;
  }
}
