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

#include "mochi_articulated_test_scenes.h"
#include "mochi_physics_test_fixture.h"

#include <mochi_core/geometry/tetrahedral_mesh.h>
#include <mochi_core/geometry/triangular_mesh.h>
#include <mochi_core/linear_algebra/utils/matrix_conversions.h>
#include <mochi_core/utils/quaternion_utils.h>
#include <mochi_core/utils/rand_utils.h>
#include <mochi_core/utils/rigid_body_utils.h>
#include <mochi_physics/mochi_physics.h>
#include <mochi_physics/mochi_physics_experimental.h>

// Private src headers
#include <mochi_physics/src/mochi_articulated_body.h>
#include <mochi_physics/src/mochi_contact.h>
#include <mochi_physics/src/mochi_context.h>
#include <mochi_physics/src/mochi_deformable.h>
#include <mochi_physics/src/mochi_differentiable.h>
#include <mochi_physics/src/mochi_group.h>
#include <mochi_physics/src/mochi_island.h>
#include <mochi_physics/src/mochi_linear_contact_skin.h>
#include <mochi_physics/src/mochi_rigid.h>
#include <mochi_physics/src/mochi_shape.h>
#include <mochi_physics/src/mochi_snle.h>
#include <mochi_physics/src/mochi_soft.h>
#include <mochi_physics/src/mochi_soft_rom_linear_systems.h>
#include <mochi_physics/src/mochi_soft_rom_systems.h>
#include <mochi_physics/src/mochi_solve.h>
#include <mochi_physics/src/mochi_step.h>

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

using namespace mochi;
using namespace mochi::experimental;
using namespace mochi::test;
namespace {
void UpdateArticulatedFullPose(entt::registry& reg, entt::entity entity) {
  auto const* joints = reg.get<CArticulatedBodyShape const>(entity).shape->GetJointsData();
  auto const& poseInfo = reg.get<CArticulatedJointPoseInfo const>(entity);
  auto const& parents = reg.get<CArticulatedParents const>(entity);
  auto const& restTransforms = reg.get<CArticulatedRestTransforms const>(entity);
  auto const& reducedPose = reg.get<CArticulatedReducedPose<TimeStep::Current>>(entity).value;
  auto const& worldFromRoot = reg.get<CRootTransform const>(entity).worldFromLocal;
  auto& jointTransforms = reg.get<CArticulatedJointTransforms<TimeStep::Current>>(entity);
  auto& linkTransforms = reg.get<CArticulatedLinkTransforms<TimeStep::Current>>(entity);
  auto& fullDofs = reg.get<CArticulatedFullPose>(entity).value;
  articulated::ComputeFullPose(
      joints->jointTypes,
      joints->jointAxes,
      poseInfo,
      parents,
      restTransforms,
      worldFromRoot,
      reducedPose,
      jointTransforms,
      linkTransforms,
      fullDofs);
}

void AddToArticulatedPose(entt::registry& reg, entt::entity entity, int localDof, real eps) {
  auto& reducedPose = reg.get<CArticulatedReducedPose<TimeStep::Current>>(entity).value;
  ColumnVector<real> delta(reg.get<CArticulatedProps const>(entity).reducedDofsDim);
  delta.SetZero();
  delta[localDof] = eps;
  auto const* joints = reg.get<CArticulatedBodyShape const>(entity).shape->GetJointsData();
  auto const& poseInfo = reg.get<CArticulatedJointPoseInfo const>(entity);
  articulated::AddLieDeltaToReducedPose(
      joints->jointTypes, joints->dofInfo, poseInfo, reducedPose, delta, reducedPose);
  UpdateArticulatedFullPose(reg, entity);
}

} // namespace

// Test collision response on multiple contacts by comparing analytical force and dforce to
// finite-difference approximations, and rigid contact sums to sums of the per-point response.
TEST(MochiContact, ComputeCollisionResponse) {
  auto runTest = [](bool explicitNormals, real coulombCoefficient, real viscousCoefficient) {
    // Contact eval config for derivative consistency tests
    ContactEvalConfig config{
        .explicitNormals = explicitNormals,
        .fadeFriction = explicitNormals,
        .useFittedHessian = false};

    ContactParams params;
    params.viscousFrictionCoefficient = viscousCoefficient;
    params.coulombFrictionCoefficient = coulombCoefficient;
    params.penaltyThresholdDefault = 0.005_r; // Large enough to cover the tested distances
    real dtStage = 0.01_r;

    // Pool of test data
    std::vector<real> distances = {0.003_r, -0.001_r, 0.002_r, -0.005_r};
    size_t nDistances = distances.size();
    std::vector<Real3> normals = {
        Real3{1_r, 2_r, 3_r}, Real3{-2_r, 3_r, -1_r}, Real3{3_r, -1_r, -2_r}};
    size_t nNormals = normals.size();
    for (size_t i = 0; i < nNormals; i++) {
      normals[i] = Normalize(normals[i]);
    }
    std::vector<Real3> positions = {Real3{4_r, 2_r, 3_r}, Real3{-2_r, 3_r, -4_r}};
    size_t nPositions = positions.size();
    std::vector<Real3> velocities = {
        Real3{-4_r, 2_r, -5_r}, Real3{-6_r, 3_r, 4_r}, Real3{5_r, 3_r, 4_r}};
    size_t nVelocities = velocities.size();

    // Fill the contact data
    ContactDetectionResult contactQuery;
    size_t nContacts = nDistances * nNormals * nPositions * nVelocities;
    contactQuery.sampleIndices.resize(nContacts);
    contactQuery.posColliding.resize(nContacts);
    contactQuery.posCollidingStageStart.resize(nContacts);
    contactQuery.sdfInfo.resize(nContacts);
    contactQuery.sdfInfoStageStart.resize(nContacts);
    contactQuery.normalColliding.resize(nContacts);
    size_t nContact = 0;
    for (auto const& distance : distances) {
      for (auto const& normal : normals) {
        for (auto const& position : positions) {
          for (auto const& velocity : velocities) {
            // Not the identity, so that sample weights indexed by contact point would be caught.
            contactQuery.sampleIndices[nContact] = static_cast<int>(nContacts - 1 - nContact);
            contactQuery.sdfInfo.val[nContact] = distance;
            contactQuery.sdfInfo.grad[nContact] = normal;
            contactQuery.posColliding[nContact] = position;
            contactQuery.sdfInfoStageStart.val[nContact] =
                distance + Dot(normal, velocity * dtStage);
            contactQuery.sdfInfoStageStart.grad[nContact] = normal;
            contactQuery.posCollidingStageStart[nContact] = position + velocity * dtStage;
            contactQuery.normalColliding[nContact] = -normal;
            nContact++;
          }
        }
      }
    }

    CollisionResponseResult res;
    res.ResizeNoInit(isize(contactQuery.sampleIndices), true, true, true);
    mochi::ComputeCollisionResponse<GradTarget::Current>(
        contactQuery, params, config, dtStage, true, true, true, res);

    // Random bone points (rho, s) and bone rotations of the skinned contact sums.
    constexpr int kNumBones = 3;
    auto generator = RandomGenerator(42);
    DynamicArray<Vec4r> bonePoints(nContacts * kNumBones);
    for (Vec4r& point : bonePoints) {
      Real3 rho;
      SetRandom(generator, -1_r, 1_r, rho);
      point = Vec4r{rho[0], rho[1], rho[2], RandomUniformValue(generator, 0_r, 1_r)};
    }
    DynamicArray<int> bonePointOffsets(nContacts + 1);
    for (size_t i = 0; i <= nContacts; ++i) {
      bonePointOffsets[i] = kNumBones * static_cast<int>(i);
    }
    DynamicArray<VMatrix3x3r> boneRotations;
    for (int bone = 0; bone < kNumBones; ++bone) {
      Real3 rotationVector;
      SetRandom(generator, -1_r, 1_r, rotationVector);
      boneRotations.push_back(ToVMatrix3x3(Quaternion::FromRotationVector(rotationVector)));
    }

    // The rigid and skinned contact sums match the sums of the per-point response for ranges of
    // every length. The requested quantities cycle with period 7, coprime with the batch size, so
    // every combination meets every size of the last batch.
    auto const testContactSums = [&](auto gradTarget) {
      constexpr GradTarget kGradTarget = decltype(gradTarget)::value;
      Vec4r const origin{0.3_r, -0.2_r, 0.1_r, 0_r};
      DynamicArray<real> weights(nContacts);
      for (size_t i = 0; i < nContacts; ++i) {
        weights[i] = 0.5_r + 0.01_r * static_cast<real>(i);
      }
      constexpr real kTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-12_r : 1e-5_r;
      CollisionResponseResult ref;
      ref.ResizeNoInit(isize(contactQuery.sampleIndices), true, true, true);
      for (int numPoints = 1; numPoints < isize(contactQuery.sampleIndices); ++numPoints) {
        Interval<int> const range{1, 1 + numPoints};
        int const flags = 1 + numPoints % 7;
        bool const assemEnergy = (flags & 1) != 0;
        bool const assemForce = (flags & 2) != 0;
        bool const assemDForce = kGradTarget == GradTarget::Current && (flags & 4) != 0;
        mochi::ComputeCollisionResponseRange<kGradTarget>(
            range, contactQuery, params, config, dtStage, true, true, true, ref);
        RigidContactSums expected;
        // Upper bounds of the sums of the term magnitudes, to scale the tolerances
        real energyScale = 0_r;
        real forceScale = 0_r;
        real dForceScale = 0_r;
        for (int s : range) {
          real const w = weights[contactQuery.sampleIndices[s]];
          Vec4r const leverArm =
              ToSimd(GetCollidingPosition<GetTimeStep<kGradTarget>()>(contactQuery, s)) - origin;
          VMatrix3x3r const skew = Skew3(leverArm);
          // Quantities not requested are zero.
          double const we = (assemEnergy ? w : 0_r) * ref.energy[s];
          Vec4r const wf = (assemForce ? w : 0_r) * ToSimd(ref.force[s]);
          VMatrix3x3r const wd = (assemDForce ? w : 0_r) * ref.dforce[s];
          expected.energy += we;
          expected.force += wf;
          expected.torque += Cross3(leverArm, wf);
          expected.dForce += wd;
          expected.skewDForce += Dot3x3(skew, wd);
          expected.skewDForceSkew += Dot3x3(Dot3x3(skew, wd), skew);
          real const lever = 1_r + Norm(ToReal3(leverArm));
          energyScale += Abs(static_cast<real>(we));
          forceScale += lever * Norm(ToReal3(wf));
          dForceScale += Sqr(lever) * Norm3x3(wd);
        }

        DynamicArray<Real3> outForce(nContacts, Real3{});
        RigidContactSums const actual = ComputeRigidContactSums<kGradTarget>(
            range,
            contactQuery,
            params,
            config,
            dtStage,
            MakeConstSpan(weights),
            origin,
            assemEnergy,
            assemForce,
            assemDForce,
            assemForce ? MakeSpan(outForce) : Span<Real3>{});

        auto const expectOutForce = [&](Span<Real3 const> forces) {
          for (int s = 0; s < isize(forces); ++s) {
            Real3 const expectedForce = (assemForce && range.Within(s)) ? ref.force[s] : Real3{};
            EXPECT_LE(Norm(forces[s] - expectedForce), kTol * Max(1_r, Norm(expectedForce)));
          }
        };

        EXPECT_NEAR(actual.energy, expected.energy, kTol * energyScale);
        EXPECT_LE(Norm(ToReal3(actual.force - expected.force)), kTol * forceScale);
        EXPECT_LE(Norm(ToReal3(actual.torque - expected.torque)), kTol * forceScale);
        EXPECT_LE(Norm3x3(actual.dForce - expected.dForce), kTol * dForceScale);
        EXPECT_LE(Norm3x3(actual.skewDForce - expected.skewDForce), kTol * dForceScale);
        EXPECT_LE(Norm3x3(actual.skewDForceSkew - expected.skewDForceSkew), kTol * dForceScale);
        expectOutForce(MakeConstSpan(outForce));

        if constexpr (kGradTarget == GradTarget::Current) {
          // The skinned sums of the bone slices followed by a rigid-collider slice about origin,
          // from their definitions in SkinnedContactSums. The pair (a, b) is at a * kNumSlices + b.
          constexpr int kNumSlices = kNumBones + 1;
          std::array<Vec4r, kNumSlices> skForce{};
          std::array<Vec4r, kNumSlices> skTorque{};
          std::array<VMatrix3x3r, kNumSlices * kNumSlices> skDForce{};
          std::array<VMatrix3x3r, kNumSlices * kNumSlices> skSkewBDForce{};
          std::array<VMatrix3x3r, kNumSlices * kNumSlices> skSkewADForce{};
          std::array<VMatrix3x3r, kNumSlices * kNumSlices> skSkewDForceSkew{};
          real skForceScale = 0_r;
          real skDForceScale = 0_r;
          for (int s : range) {
            int const sample = contactQuery.sampleIndices[s];
            real const w = weights[sample];
            Vec4r const wf = (assemForce ? w : 0_r) * ToSimd(ref.force[s]);
            VMatrix3x3r const wd = (assemDForce ? w : 0_r) * ref.dforce[s];
            std::array<real, kNumSlices> scales{};
            std::array<Vec4r, kNumSlices> leverArms{};
            for (int a = 0; a < kNumBones; ++a) {
              Vec4r const& point = bonePoints[sample * kNumBones + a];
              scales[a] = point[3];
              leverArms[a] = ToSimdDirection(DotMatVec3x3(boneRotations[a], point));
            }
            scales[kNumBones] = 1_r;
            leverArms[kNumBones] = ToSimd(contactQuery.posColliding[s]) - origin;
            real maxLever = 0_r;
            for (int a = 0; a < kNumSlices; ++a) {
              maxLever = Max(maxLever, Norm(ToReal3(leverArms[a])));
              skForce[a] += scales[a] * wf;
              skTorque[a] += Cross3(leverArms[a], wf);
              for (int b = a; b < kNumSlices; ++b) {
                int const p = a * kNumSlices + b;
                VMatrix3x3r const skewAWD = Dot3x3(Skew3(leverArms[a]), wd);
                skDForce[p] += (scales[a] * scales[b]) * wd;
                skSkewBDForce[p] += scales[a] * Dot3x3(Skew3(leverArms[b]), wd);
                skSkewADForce[p] += scales[b] * skewAWD;
                skSkewDForceSkew[p] += Dot3x3(skewAWD, Skew3(leverArms[b]));
              }
            }
            skForceScale += (1_r + maxLever) * Norm(ToReal3(wf));
            skDForceScale += Sqr(1_r + maxLever) * Norm3x3(wd);
          }

          for (bool hasRigidColliderSlice : {false, true}) {
            SCOPED_TRACE("rigid-collider slice " + std::to_string(hasRigidColliderSlice));
            SkinnedContactSlices const slices{
                .pointOffsets = MakeConstSpan(bonePointOffsets),
                .points = MakeConstSpan(bonePoints),
                .boneRotations = MakeConstSpan(boneRotations),
                .rigidColliderOrigin =
                    hasRigidColliderSlice ? std::optional{origin} : std::nullopt};
            DynamicArray<Real3> skOutForce(nContacts, Real3{});
            SkinnedContactSums const skinned = ComputeSkinnedContactSums(
                range,
                contactQuery,
                params,
                config,
                dtStage,
                MakeConstSpan(weights),
                slices,
                assemEnergy,
                assemForce,
                assemDForce,
                GetDefaultAllocator(),
                assemForce ? MakeSpan(skOutForce) : Span<Real3>{});

            EXPECT_NEAR(skinned.energy, expected.energy, kTol * energyScale);
            for (int a = 0; a < skinned.numSlices; ++a) {
              if (assemForce) {
                EXPECT_LE(Norm(ToReal3(skinned.force[a] - skForce[a])), kTol * skForceScale);
                EXPECT_LE(Norm(ToReal3(skinned.torque[a] - skTorque[a])), kTol * skForceScale);
              }
              if (assemDForce) {
                for (int b = a; b < skinned.numSlices; ++b) {
                  int const p = skinned.PairIndex(a, b);
                  int const q = a * kNumSlices + b;
                  real const tol = kTol * skDForceScale;
                  EXPECT_LE(Norm3x3(skinned.dForce[p] - skDForce[q]), tol);
                  EXPECT_LE(Norm3x3(skinned.skewBDForce[p] - skSkewBDForce[q]), tol);
                  EXPECT_LE(Norm3x3(skinned.skewADForce[p] - skSkewADForce[q]), tol);
                  EXPECT_LE(Norm3x3(skinned.skewDForceSkew[p] - skSkewDForceSkew[q]), tol);
                }
              }
            }
            expectOutForce(MakeConstSpan(skOutForce));
          }
        }
      }
    };
    testContactSums(std::integral_constant<GradTarget, GradTarget::Current>{});
    if (explicitNormals) {
      testContactSums(std::integral_constant<GradTarget, GradTarget::Previous>{});
    }

    // Compute through finite differences
    CollisionResponseResult resTest;
    resTest.force.resize(nContacts, {});
    resTest.dforce.resize(nContacts, {});
    constexpr real kEps = 1e-4_r;
    for (int j = 0; j < 3; j++) {
      Real3 delta = kEps * BasisVector<real, 3>(j);
      for (size_t i = 0; i < nContacts; i++) {
        contactQuery.posColliding[i] = delta + contactQuery.posColliding[i];
        contactQuery.sdfInfo.val[i] += Dot(contactQuery.sdfInfo.grad[i], delta);
      }
      CollisionResponseResult resF;
      resF.ResizeNoInit(isize(contactQuery.sampleIndices), true, true, false);
      mochi::ComputeCollisionResponse<GradTarget::Current>(
          contactQuery, params, config, dtStage, true, true, false, resF);

      delta = -2_r * kEps * BasisVector<real, 3>(j);
      for (size_t i = 0; i < nContacts; i++) {
        contactQuery.posColliding[i] = delta + contactQuery.posColliding[i];
        contactQuery.sdfInfo.val[i] += Dot(contactQuery.sdfInfo.grad[i], delta);
      }
      CollisionResponseResult resB;
      resB.ResizeNoInit(isize(contactQuery.sampleIndices), true, true, false);
      mochi::ComputeCollisionResponse<GradTarget::Current>(
          contactQuery, params, config, dtStage, true, true, false, resB);

      delta = kEps * BasisVector<real, 3>(j);
      for (size_t i = 0; i < nContacts; i++) {
        contactQuery.posColliding[i] = delta + contactQuery.posColliding[i];
        contactQuery.sdfInfo.val[i] += Dot(contactQuery.sdfInfo.grad[i], delta);
      }

      for (size_t i = 0; i < nContacts; i++) {
        resTest.force[i][j] = static_cast<real>(resF.energy[i] - resB.energy[i]) / (-2_r * kEps);
        resTest.dforce[i][j] = ToSimd((resF.force[i] - resB.force[i]) / (2_r * kEps));
      }
    }

    // Test
    for (size_t i = 0; i < nContacts; i++) {
      EXPECT_NEAR(
          Norm(res.force[i] - resTest.force[i]) / Max(Norm(res.force[i]), Norm(resTest.force[i])),
          0_r,
          1e-2_r);
      EXPECT_NEAR(
          Norm3x3(res.dforce[i] - resTest.dforce[i]) /
              Max(Norm3x3(res.dforce[i]), Norm3x3(resTest.dforce[i])),
          0_r,
          1e-2_r);
    }
  };

  runTest(/* explicitNormals */ false, /*coulombCoefficient*/ 0_r, /*viscousCoefficient*/ 100.0_r);
  runTest(/* explicitNormals */ false, /*coulombCoefficient*/ 0.5_r, /*viscousCoefficient*/ 0_r);
  runTest(/* explicitNormals */ true, /*coulombCoefficient*/ 0_r, /*viscousCoefficient*/ 100.0_r);
  runTest(/* explicitNormals */ true, /*coulombCoefficient*/ 0.5_r, /*viscousCoefficient*/ 0_r);
}

TEST(MochiContact, SetupActiveCollisionNormalsCachesLinearElementNormal) {
  CSimulationParams simulationParams;
  TetrahedralMesh const tetMesh{CreateMinimalTetMeshSingleTet()};
  auto const displacements = ColumnVector<real>::Zero(kSpaceDim3 * tetMesh.GetNumNodes());
  CRootTransform const transform{
      TransformRT{Quaternion::FromRotationVector(Real3{0.2_r, -0.3_r, 0.4_r})}};
  CContactSamples<TimeStep::Current> const contactPositions{0};
  auto const surfaceMesh = CreateBoundaryMesh(tetMesh);
  Real3 const expectedNormal =
      transform.worldFromLocal.GetRotation() * surfaceMesh.GetElementNormals()[0];
  auto runTest = [&](auto const& discretization) {
    CActiveCollisions</*kIsSync*/ true, TimeStep::Current> collisions(/*numPartitions*/ 1);
    entt::entity const collider{};
    collisions.SetUp(MakeSingletonConstSpan(collider));
    auto& result = collisions.front().collisionResult;
    result.sampleIndices = {0, 1, 3};
    result.jacColliderFromWorld.resize(3, VEye<3>());
    deformable::SetupActiveCollisionNormals</*kIsSync*/ true>(
        {},
        ecs::CtxGlobal<CSimulationParams const>{simulationParams},
        discretization,
        CFinalDisplacementRef<TimeStep::Current>{AsConstView(displacements)},
        CFinalDisplacementRef<TimeStep::StageStart>{AsConstView(displacements)},
        transform,
        contactPositions,
        collisions);
    EXPECT_NEAR_EQ(expectedNormal, result.normalColliding[0]);
    EXPECT_NEAR_EQ(result.normalColliding[0], result.normalColliding[1]);
    EXPECT_NEAR_EQ(
        transform.worldFromLocal.GetRotation() * surfaceMesh.GetElementNormals()[1],
        result.normalColliding[2]);
  };

  CFemVolumeDiscretizationP1Q1 volumeDiscretization;
  volumeDiscretization.femElements.emplace_back(
      0, tetMesh.GetNodeCoordinates(), tetMesh.GetElementConnectivity());
  runTest(
      CFemBoundaryDiscretization::Create(
          tetMesh, volumeDiscretization, ActorBoundaryElementType::P1Q3));
  runTest(CFemSurfaceDiscretization::Create(ActorBoundaryElementType::P1Q3, surfaceMesh));
}

// Rigid colliders whose actor frame is at the world origin, while their contact surface is near
// z = 32 m, where single-precision rounding is large. The flat colliders' top face is at y = 0.
enum class FarCollider { Plane, Box, Mesh, GridSdf, Sphere };

class FarColliderContactTest : public test::MochiSceneTestBase,
                               public ::testing::WithParamInterface<FarCollider> {
 protected:
  void CreateFarCollider() {
    RigidActorParams params;
    params.isStatic = true;
    switch (GetParam()) {
      case FarCollider::Plane:
        params.shape = _mochiContext->CreatePlaneShape(Real3{0_r, 1_r, 0_r}, 0_r, ExpectOK{});
        break;
      case FarCollider::Sphere:
        params.shape = _mochiContext->CreateSphereShape(Real3{0_r, -1_r, 32_r}, 1_r, ExpectOK{});
        break;
      case FarCollider::Box:
      case FarCollider::Mesh:
      case FarCollider::GridSdf: {
        auto [positions, triangles] = test::CreateMinimalTriMeshUnitCube(Real3{2_r, 1_r, 2_r});
        for (auto& position : positions) {
          position += Real3{-1_r, -1_r, 31_r};
        }
        params.shape = _mochiContext->CreateTriMeshShape(
            Flatten(MakeSpan(positions)), Flatten(MakeSpan(triangles)), ExpectOK{});
        params.colliderType = GetParam() == FarCollider::Box ? ColliderType::Box
            : GetParam() == FarCollider::Mesh                ? ColliderType::Mesh
                                                             : ColliderType::Sdf;
        params.sdf.resolutionMode = GridSdfResolutionMode::Explicit;
        params.sdf.resolutionDelta = Real3{0.05_r, 0.05_r, 0.05_r};
      } break;
    }
    ASSERT_NE(_scene->CreateRigidActor(params, ExpectOK{}), nullptr);
  }

  // Creates a dynamic cube with 0.5 m sides, centered at the origin of its actor frame.
  Actor* CreateCube(TransformRT const& worldFromCube) {
    auto [coordinates, connectivity] =
        test::CreateMinimalTetMeshUnitCube(Real3{0.5_r, 0.5_r, 0.5_r});
    for (auto& coordinate : coordinates) {
      coordinate -= Real3{0.25_r, 0.25_r, 0.25_r};
    }
    RigidActorParams params;
    params.shape = _mochiContext->CreateTetMeshShape(
        Flatten(MakeSpan(coordinates)), Flatten(MakeSpan(connectivity)), ExpectOK{});
    params.worldFromLocal = worldFromCube;
    return _scene->CreateRigidActor(params, ExpectOK{});
  }
};

// Friction treats any difference between current and stage-start contact positions as slip, so a
// stationary pair must produce identical positions on both sides. With implicit normals, stage
// start subtracts each contact's displacement; with explicit normals, it repeats the query.
TEST_P(FarColliderContactTest, StationaryContactPositionsMatchStageStart) {
  CreateFarCollider();
  Actor* cube = CreateCube(
      TransformRT{
          Quaternion::FromRotationVector(Real3{0.31_r, -0.58_r, 0.24_r}),
          Real3{0.1_r, 0.2_r, 32.1_r}});
  ASSERT_NE(cube, nullptr);

  for (bool explicitNormals : {false, true}) {
    SCOPED_TRACE(explicitNormals ? "explicit normals" : "implicit normals");
    SolverParams solverParams = _scene->GetSolverParams();
    solverParams.experimentalEval.explicitNormals = explicitNormals;
    _scene->SetSolverParams(solverParams, ExpectOK{});

    auto& reg = GetRegistry();
    reg.ctx<CSceneTime>().Advance(0.001);
    PreStepEcs(reg);
    for (auto [island] : reg.view<TagIsland>().each()) {
      CollisionDetectionPipeline<TimeStep::Current>(reg, reg.get<CIslandDescendants const>(island));
    }

    int numContacts = 0;
    int numMismatches = 0;
    for (auto const& collision :
         reg.get<CActiveCollisions</*kIsSync*/ false, TimeStep::Current> const>(GetEntity(cube))) {
      auto const& result = collision.collisionResult;
      ASSERT_EQ(result.posColliding.size(), result.posCollidingStageStart.size());
      for (int i = 0; i < isize(result.posColliding); ++i) {
        ++numContacts;
        numMismatches +=
            std::memcmp(
                &result.posColliding[i], &result.posCollidingStageStart[i], sizeof(Real3)) != 0;
      }
    }
    EXPECT_GT(numContacts, 0);
    EXPECT_EQ(numMismatches, 0) << numMismatches << " of " << numContacts << " contacts differ";
  }
}

static std::string FarColliderName(::testing::TestParamInfo<FarCollider> const& info) {
  constexpr std::array kNames{"Plane", "Box", "Mesh", "GridSdf", "Sphere"};
  return kNames[static_cast<int>(info.param)];
}

INSTANTIATE_TEST_SUITE_P(
    AllColliders,
    FarColliderContactTest,
    ::testing::Values(
        FarCollider::Plane,
        FarCollider::Box,
        FarCollider::Mesh,
        FarCollider::GridSdf,
        FarCollider::Sphere),
    FarColliderName);

class FarGroundRestingTest : public FarColliderContactTest {};

// Cubes resting on stationary ground must not slide. Friction would read any difference between
// current and stage-start contact positions as slip.
TEST_P(FarGroundRestingTest, CubesDoNotSlide) {
  CreateFarCollider();
  // Separated cubes on the top face, with orientations that slide when positions differ.
  struct Placement {
    real yaw;
    real x;
    real z;
  };
  Placement const placements[] = {
      {0.37_r, -0.5_r, 31.5_r},
      {1.13_r, 0.5_r, 31.5_r},
      {2.29_r, -0.5_r, 32.5_r},
      {-0.71_r, 0.5_r, 32.5_r}};
  DynamicArray<Actor*> cubes;
  DynamicArray<Real3> starts;
  for (auto const& [yaw, x, z] : placements) {
    Actor* cube = CreateCube(TransformRT{Quaternion::RotationY(yaw), Real3{x, 0.25_r, z}});
    ASSERT_NE(cube, nullptr);
    cubes.push_back(cube);
    starts.push_back(cube->GetRootTransform().GetTranslation());
  }
  for (int step = 0; step < 500; ++step) {
    _scene->Step(0.001_r);
  }
  for (int i = 0; i < isize(cubes); ++i) {
    Real3 const displacement = cubes[i]->GetRootTransform().GetTranslation() - starts[i];
    EXPECT_LT(Norm(Real3{displacement[0], 0_r, displacement[2]}), 1e-4_r) << "cube " << i;
  }
}

INSTANTIATE_TEST_SUITE_P(
    FlatColliders,
    FarGroundRestingTest,
    ::testing::Values(
        FarCollider::Plane,
        FarCollider::Box,
        FarCollider::Mesh,
        FarCollider::GridSdf),
    FarColliderName);

namespace {
struct TestParams {
  std::string name = {};
  GradTarget gradTarget = GradTarget::Current;
  real coulombCoefficient = 0_r;
  real viscousCoefficient = 0_r;
  real dampingCoefficient = 0_r;
  bool explicitNormals = false;
  real resTol = 1e-3_r;
  real dresTol = 1e-3_r;
  bool ignoreColliderRotDres = false;
  real eps = 1e-4_r;
  bool frictionWithColliderNormal = true;
  // Stretch of the soft collider along its local x axis at stage start.
  real colliderStretch = 0_r;
  real colliderShear = 0_r;
  std::optional<ActorBoundaryElementType> collidingBoundaryElementType = {};
  bool ignoreColliderDres = false;
};
} // namespace

TEST(MochiContact, MakeContactGraphWithFiveContactJacobians) {
  entt::registry reg;
  auto const colliding = reg.create();
  auto const collider = reg.create();

  auto& activeCollisions =
      reg.emplace<CActiveCollisions</*kIsSync*/ true, TimeStep::Current>>(colliding, 1);
  activeCollisions.SetUp(MakeSingletonConstSpan(collider));
  auto& collision = activeCollisions.front();
  collision.collisionResult.sampleIndices.push_back(0);

  auto& collidingJacs = reg.emplace<CCollJacs<CollRole::Colliding>>(colliding);
  collidingJacs.emplace_back(
      /*isSync*/ true, &collision.collisionResult, /*useContactSums*/ false, collider, 0);
  auto collidingSlices = collidingJacs[0].PrepareJacs(3);
  auto& colliderJacs = reg.emplace<CCollJacs<CollRole::Collider>>(collider);
  colliderJacs.emplace_back(
      /*isSync*/ true, &collision.collisionResult, /*useContactSums*/ false, colliding, 0);
  auto colliderSlices = colliderJacs[0].PrepareJacs(2);

  std::array<ContactJac*, 5> const slices{
      &collidingSlices[0],
      &collidingSlices[1],
      &collidingSlices[2],
      &colliderSlices[0],
      &colliderSlices[1]};
  std::array<int, 5> const dofIndices{0, 1, 2, 0, 3};
  for (int i = 0; i < isize(slices); ++i) {
    slices[i]->Resize(/*sharedDoFs*/ true, /*sharedJacs*/ true, 1, 1);
    slices[i]->Inds(0)[0] = dofIndices[i];
  }

  auto graph = MakeContactGraph<1, /*kIsSync*/ true>(reg, MakeSingletonConstSpan(colliding));
  std::array<int, 5> const expectedPointers{0, 4, 8, 12, 16};
  std::array<int, 16> const expectedTargets{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
  EXPECT_SPAN_EQ(graph.GetPointers(), MakeConstSpan(expectedPointers));
  EXPECT_SPAN_EQ(graph.GetTargets(), MakeConstSpan(expectedTargets));
}

// Assemble the contact residual and dresidual of slices with shared DoFs (with and without
// auxiliary Jacobians, and with overlapping DoFs) and slices with per-point DoFs, and compare them
// to the sums over points of the products of their full Jacobians. The residuals of the slices with
// auxiliary Jacobians are written to link residual slots, and are projected before the comparison.
TEST(MochiContact, AssembleResAndDResWithSharedAndPerPointSlices) {
  constexpr int kNumDofs = 32;
  constexpr int kNumContacts = 3;
  unsigned int seed = 7;

  // DoF indices are given per point, or in a single row for shared DoFs. Points 0 and 1 share all
  // indices, so they are assembled in the same batch, and point 2 in another batch.
  auto initJac = [&](ContactJac& jac,
                     std::vector<std::vector<int>> const& inds,
                     ContactJac::AuxView jacAux = {},
                     int internalDofOffset = kSentinelIndex) {
    bool const sharedDoFs = isize(inds) == 1;
    int const nDoFs = jacAux.empty() ? isize(inds[0]) : RigidSize::kDAll;
    jac.Resize(sharedDoFs, /*sharedJacs*/ false, nDoFs, kNumContacts);
    for (int s = 0; s < kNumContacts; ++s) {
      jac.Jac(s).SetRandom(++seed, -1_r, 1_r);
    }
    if (jacAux.empty()) {
      for (int row = 0; row < isize(inds); ++row) {
        std::copy(inds[row].begin(), inds[row].end(), jac.Inds(row).begin());
      }
    } else {
      jac.SetJacAux(jacAux, 0, inds[0], internalDofOffset);
    }
    jac.CompressIndices();
  };
  std::array<ContactJac, 6> slices;
  // Per-point DoFs
  initJac(
      slices[0], {{19, 20, 21, 25, 26, 27}, {19, 20, 21, 25, 26, 27}, {22, 23, 24, 19, 20, 21}});
  // Shared DoFs with auxiliary Jacobians, with fewer and more state DoFs than internal DoFs,
  // overlapping in DoF 2
  RowMatrix<real, RigidSize::kDAll> jacAux1(RigidSize::kDAll, 3);
  RowMatrix<real, RigidSize::kDAll> jacAux2(RigidSize::kDAll, 8);
  jacAux1.SetRandom(++seed, -1_r, 1_r);
  jacAux2.SetRandom(++seed, -1_r, 1_r);
  initJac(slices[1], {{0, 1, 2}}, jacAux1, /*internalDofOffset*/ 0);
  initJac(slices[2], {{2, 3, 4, 5, 6, 7, 8, 9}}, jacAux2, /*internalDofOffset*/ RigidSize::kDAll);
  // Shared DoFs without auxiliary Jacobian
  initJac(slices[3], {{10, 11, 12}});
  // Per-point DoFs
  initJac(slices[4], {{28, 29, 30}, {28, 29, 30}, {29, 30, 31}});
  // Shared rigid DoFs without auxiliary Jacobian
  initJac(slices[5], {{13, 14, 15, 16, 17, 18}});

  ContactDetectionResult query;
  query.sampleIndices = {0, 1, 2};
  std::array<real, kNumContacts> const weights{0.5_r, 1_r, 2_r};
  CollisionResponseResult response;
  response.energy.resize(kNumContacts, 0.0);
  response.force.resize(kNumContacts, Real3{});
  for (auto& force : response.force) {
    ColumnVector<real, 3> randomForce;
    randomForce.SetRandom(++seed, -1_r, 1_r);
    force = Real3{randomForce[0], randomForce[1], randomForce[2]};
  }
  response.dforce.resize(kNumContacts);
  for (auto& dforce : response.dforce) {
    // Symmetric, since off-diagonal blocks are assembled as transposes of each other.
    Matrix<real, 3, 3> mat;
    mat.SetRandom(++seed, -1_r, 1_r);
    AsMatrixView(dforce) = mat + mat.Transpose();
  }

  auto runTest = [&](std::vector<int> const& collidingSlices,
                     std::vector<int> const& colliderSlices = {}) {
    DynamicArray<ContactJac const*> jacs;
    for (int i : collidingSlices) {
      jacs.push_back(&slices[i]);
    }
    for (int i : colliderSlices) {
      jacs.push_back(&slices[i]);
    }

    // The sparsity only couples the DoFs of each point, so that writes outside it are caught. The
    // dresidual starts at 1 within it, so that overwriting instead of accumulating is caught.
    std::vector<std::vector<int>> pattern(kNumDofs);
    auto expected = Matrix<real>::Zero(kNumDofs, kNumDofs);
    for (int s = 0; s < kNumContacts; ++s) {
      for (auto const* jacRow : jacs) {
        for (int row : jacRow->Inds(s)) {
          for (auto const* jacCol : jacs) {
            for (int col : jacCol->Inds(s)) {
              pattern[row].push_back(col);
              expected(row, col) = 1_r;
            }
          }
        }
      }
    }
    SparseMatrix<real> dres{MakeSparsityGraph(std::move(pattern), kNumDofs)};
    dres.SetConstant(1_r);
    auto res = ColumnVector<real>::Zero(kNumDofs);
    // The slots start with stale values, which must be overwritten.
    ColumnVector<real> linkRes(RigidSize::kDAll * isize(jacs));
    linkRes.SetConstant(100_r);
    MOCHI_FILO_STACK_ALLOCATOR(allocator, 64 * 1024);
    AssembleCollisionResponse(
        query,
        response,
        MakeConstSpan(weights),
        MakeSpan(jacs),
        isize(collidingSlices),
        &allocator,
        nullptr,
        AsView(res),
        AsView(dres),
        AsView(linkRes));

    // Project the link residuals of the slices with auxiliary Jacobians, from their slots, to their
    // state DoFs.
    for (int a = 0; a < isize(jacs); ++a) {
      ContactJac const& jac = *jacs[a];
      if (!jac.JacAux()) {
        continue;
      }
      ColumnVector<real> const stateRes =
          jac.JacAux().Transpose() * linkRes.MiddleRows(RigidSize::kDAll * a, RigidSize::kDAll);
      Span<int const> inds = jac.Inds(0);
      for (int c = 0; c < isize(inds); ++c) {
        res[inds[c]] += stateRes[c];
      }
    }

    auto expectedRes = ColumnVector<real>::Zero(kNumDofs);
    for (int s = 0; s < kNumContacts; ++s) {
      auto jacFull = Matrix<real>::Zero(3, kNumDofs);
      for (auto const* jac : jacs) {
        Matrix<real> const jacState =
            jac->JacAux() ? Matrix<real>{jac->Jac(s) * jac->JacAux()} : Matrix<real>{jac->Jac(s)};
        Span<int const> inds = jac->Inds(s);
        for (int c = 0; c < isize(inds); ++c) {
          jacFull.Col(inds[c]) += jacState.Col(c);
        }
      }
      Matrix<real, 3, 3> const dresContact{-weights[s] * AsMatrixView(response.dforce[s])};
      expected += jacFull.Transpose() * Matrix<real>{dresContact * jacFull};
      Real3 const resContact = -weights[s] * response.force[s];
      expectedRes += jacFull.Transpose() * ColumnVectorView<real const, 3>(&resContact[0]);
    }
    constexpr real kTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-12_r : 1e-5_r;
    EXPECT_TRUE(NearEqualMatrices(res, expectedRes, kTol * expectedRes.Norm()));
    EXPECT_TRUE(NearEqualMatrices(dres, expected, kTol * expected.Norm()));
  };
  // Grouped Jacobians with shared DoFs, which are both first and second in pairs with per-point
  // Jacobians
  runTest({0, 1, 2, 3}, {4});
  runTest({0, 1, 2, 3});
  // Two Jacobians with shared DoFs, the minimum for grouping
  runTest({0, 1, 3});
  // No Jacobian with per-point DoFs
  runTest({1, 2, 3});
  // Jacobians with auxiliary Jacobians after shared Jacobians without them
  runTest({3, 5, 1, 2});
  // The shared Jacobian of the collider is grouped with those of the colliding actor
  runTest({0, 1, 2}, {5});
  // Fewer than two shared Jacobians in the colliding actor, so none are grouped
  runTest({3}, {5, 1, 2});
  runTest({0, 1}, {2, 4});
  // A single Jacobian with shared DoFs
  runTest({1, 0});
  runTest({4, 2});
  runTest({2});
  // No Jacobian with shared DoFs
  runTest({0, 4});
}

// Objective, residual and dense dresidual of the only island of a scene.
struct IslandAssembly {
  double objective = 0.0;
  ColumnVector<real> residual;
  Matrix<real> dresidual;
};

static IslandAssembly AssembleSingleIsland(
    entt::registry& reg,
    GradTarget gradTarget,
    bool assemRes = true,
    bool assemDRes = true) {
  // There should be exactly one island.
  entt::entity island = entt::null;
  for (auto [e] : reg.view<TagIsland>().each()) {
    EXPECT_FALSE(reg.valid(island));
    island = e;
  }
  EXPECT_TRUE(reg.valid(island));

  // TODO: Add this collision detection pass. Currently it fails because stage-start contacts of
  // the mapped collider produce no friction. Fix the mapped collider before adding this code.
  // T259662824
  // Run stage-start collision detection
  // auto const& descendants = reg.get<CIslandDescendants const>(island);
  // CollisionDetectionPipeline<TimeStep::StageStart>(reg, descendants);

  // Get island DOF info
  auto const& islandDofInfo = reg.get<CIslandDofInfo const>(island);
  int const solutionSize = islandDofInfo.poseSize;
  int const dofsSize = islandDofInfo.dofsSize;

  // Set up the SNLE problem with an assembly function
  SnleProblemFunctions<real> functions;
  functions.assemble = [&](SnleProblem<real>& problem, AssemblyParams const& params) {
    solver::AssembleIslandPipeline(reg, island, params, problem);
  };
  SnleProblem<real> problem(dofsSize, solutionSize, std::move(functions));

  // Initialize solution to zero
  problem.solution.SetZero();

  // Request the objective, and the residual and dresidual if requested. The dresidual is assembled
  // only for GradTarget::Current.
  AssemblyParams params = {
      .assemObj = true,
      .assemRes = assemRes,
      .assemDRes = assemDRes && gradTarget == GradTarget::Current,
      .psdDRes = false,
      .fittedSaturationHessian = SaturationHessianParams::All(false),
      .gradTarget = gradTarget};
  problem.UpdateObjResDRes(params);

  IslandAssembly assembly;
  assembly.objective = problem.GetObjective();
  if (params.assemRes) {
    assembly.residual = problem.GetResidual();
  }
  if (params.assemDRes) {
    assembly.dresidual = ToMatrix(problem.GetDResidual());
  }
  return assembly;
}

// Base class to implement contact tests between a colliding actor and a collider actor. The tests
// create a Mochi scene with both actors, compute objective, residual and dresidual at the initial
// state, and validate the residual and dresidual through finite differences.
class MochiContactTestBase : public test::MochiSceneTestBase,
                             public ::testing::WithParamInterface<TestParams> {
 protected:
  static constexpr real kDtStage = 1e-2_r;

  entt::entity _colliding = entt::null;
  entt::entity _collider = entt::null;

  int _numDofsA = 0;
  int _numDofsB = 0;
  int _numDofs = 0;

  double _objective = 0.0;
  ColumnVector<real> _residual;
  SparseMatrix<real> _dresidual;

  // Whether the colliding soft cube contacts through a linear contact skin.
  bool _useContactSkin = false;

  // Boundary element type of the colliding contact skin or articulated skin.
  ActorBoundaryElementType _collidingBoundaryElementType = ActorBoundaryElementType::P1Q3;

  // Whether colliders are enlarged about the origin of their transform, so that they contain the
  // colliding unit cube and all its samples are in contact.
  bool _largeColliders = false;
  static constexpr real kLargeColliderScale = 3_r;

  // A collider cube of scale kShallowColliderScale with ShallowColliderTransform() lies under the
  // colliding unit cube and overlaps its bottom face by less than the penalty threshold.
  static constexpr real kShallowColliderScale = 3_r;
  static TransformRT ShallowColliderTransform() {
    real constexpr kOverlap = 1e-3_r;
    return TransformRT{Real3{-1_r, -1_r, kOverlap - kShallowColliderScale}};
  }

 public:
  virtual void InitState() = 0;

  virtual void AddToState(int i, real eps) = 0;

  virtual void AddToOldState(int /*i*/, real /*eps*/) {
    MOCHI_ASSERT(false, "Not supported for this test configuration");
  }

  virtual void
  InitializeScene(real coulombCoefficient, real viscousCoefficient, real dampingCoefficient) = 0;

  void SetUp() override { // Called just before each test case
    test::MochiSceneTestBase::SetUp();

    // Initialize result
    _objective = 0.0;
    _residual.Reset(_numDofs);
    _residual.SetZero();
    _dresidual = ToSparseMatrix(Matrix<real>::Zero(_numDofs, _numDofs));
  }

  void InitActors(Actor* collidingActor, Actor* colliderActor) {
    auto& reg = GetRegistry();

    // Store colliding & collider entities
    _colliding = GetEntity(collidingActor->GetHandle());
    _collider = GetEntity(colliderActor->GetHandle());
    EXPECT_EQ(_numDofsA, reg.get<CActorDofInfo>(_colliding).dofsSize);
    if (_numDofsB != 0) {
      EXPECT_EQ(_numDofsB, reg.get<CActorDofInfo>(_collider).dofsSize);
    }

    // Contact filtering
    _scene->EnableLayerContactSymmetric("Colliding", "Colliding", false, test::ExpectOK{});
    _scene->EnableLayerContactSymmetric("Collider", "Collider", false, test::ExpectOK{});
    _scene->EnableLayerContactSymmetric("Colliding", "Collider", true, test::ExpectOK{});

    // This test does not step the full simulation, but calling PreStep ensures that certain things
    // are initialized, including islands and DOF offsets.
    PreStep();
  }

  void SetTransformFromState(entt::entity entity, TransformRT const& rigidState) {
    auto& reg = GetRegistry();
    auto const& rigidInertia = reg.get<CRigidBodyInertia const>(entity);
    auto& rootTransform = reg.get<CRootTransform>(entity);
    mochi::rigid::RigidStateToRootTransform(
        rigidInertia.GetCenterOfMassLocal(), rigidState, rootTransform.worldFromLocal);
  }

  void SetStageStartTransformFromState(entt::entity entity, TransformRT const& rigidState) {
    auto& reg = GetRegistry();
    auto const& rigidInertia = reg.get<CRigidBodyInertia const>(entity);
    auto& rootTransform = reg.get<CRootTransform>(entity);
    mochi::rigid::RigidStateToRootTransform(
        rigidInertia.GetCenterOfMassLocal(), rigidState, rootTransform.worldFromLocalStageStart);
  }

  // Find out if a global DOF index is associated with the colliding or the collider entity.
  // Return the entity and the local DOF index for that entity.
  virtual std::pair<entt::entity, int> GetEntityAndLocalDofFromGlobalDof(int globalDof) {
    auto const& reg = GetRegistry();
    auto dofOffsetA = reg.get<CDofOffset>(_colliding).dofsOffset;
    auto dofOffsetB = (_numDofsB != 0) ? reg.get<CDofOffset>(_collider).dofsOffset : 0;
    if ((globalDof >= dofOffsetA) && (globalDof < dofOffsetA + _numDofsA)) {
      return std::make_pair(_colliding, globalDof - dofOffsetA);
    } else {
      EXPECT_TRUE(
          (globalDof >= dofOffsetB) &&
          (globalDof < dofOffsetB + _numDofsB)); // DOF index out of range
      return std::make_pair(_collider, globalDof - dofOffsetB);
    }
  }

  void PreStep() {
    // Prepare for contact by doing things that Scene::Step would normally take care of.
    // This includes: advancing time, which is required by UpdateConservativeStepBounds, which is
    // required by UpdateConservativePotentialColliders, which is required by
    // UpdatePotentialColliders, which happens during CollisionDetectionPipeline (see
    // ComputeResponse()).
    auto& reg = GetRegistry();
    double constexpr kTimeStep = 0.01; // Arbitrary
    reg.ctx<CSceneTime>().Advance(kTimeStep);
    PreStepEcs(reg);
  }

  void ComputeResponse(GradTarget gradTarget) {
    IslandAssembly assembly = AssembleSingleIsland(GetRegistry(), gradTarget);
    _objective = assembly.objective;
    _residual = std::move(assembly.residual);
    _dresidual.SetZero();
    if (gradTarget == GradTarget::Current) {
      for (int row = 0; row < _numDofs; ++row) {
        for (int col = 0; col < _numDofs; ++col) {
          _dresidual.SetValue(row, col, assembly.dresidual(row, col));
        }
      }
    }
  }

  void InitializeSceneFromParams() {
    auto const& params = GetParam();
    _collidingBoundaryElementType =
        params.collidingBoundaryElementType.value_or(_collidingBoundaryElementType);

    auto solverParams = _scene->GetSolverParams();
    solverParams.experimentalEval.fittedSaturationHessian = SaturationHessianParams::All(false);
    solverParams.experimentalEval.explicitNormals = params.explicitNormals;
    // Disable fade friction if testing with GradTarget::Previous, or if testing with
    // GradTarget::Current and implicit normals.
    bool const isCurrentTarget = params.gradTarget == GradTarget::Current;
    solverParams.experimentalEval.fadeFriction = isCurrentTarget && params.explicitNormals;
    _scene->SetSolverParams(solverParams, test::ExpectOK{});

    InitializeScene(
        params.coulombCoefficient, params.viscousCoefficient, params.dampingCoefficient);
  }

  // Test with finite differences. For GradTarget::Current, test both residual and dresidual. For
  // GradTarget::Previous, test only the residual. The results for the dresidual are not perfect, as
  // the derivatives discard some nonlinearities (e.g., derivative of the SDF normal for some SDFs,
  // 2nd derivative of the mapping from contact sample to DoFs). In particular, error grows for the
  // rotation of the _collider, and its corresponding block in the Hessian is sometimes ignored
  void RunTest() {
    auto const& params = GetParam();
    InitializeSceneFromParams();

    InitState();
    PreStep();
    ComputeResponse(params.gradTarget);
    ColumnVector<real> res0 = _residual.Duplicate();
    SparseMatrix<real> dres = _dresidual.Duplicate();
    ColumnVector<real> resFD = _residual.Duplicate();
    SparseMatrix<real> dresFD = _dresidual.Duplicate();

    // Test through finite differences
    real const eps = params.eps;
    for (int i = 0; i < _numDofs; i++) {
      InitState();
      if (params.gradTarget == GradTarget::Current) {
        AddToState(i, eps);
      } else {
        AddToOldState(i, eps);
      }
      ComputeResponse(params.gradTarget);
      double objF = _objective;
      ColumnVector<real> resF = _residual.Duplicate();

      InitState();
      if (params.gradTarget == GradTarget::Current) {
        AddToState(i, -eps);
      } else {
        AddToOldState(i, -eps);
      }
      ComputeResponse(params.gradTarget);
      double objB = _objective;
      ColumnVector<real> resB = _residual.Duplicate();

      resFD[i] = (real)(objF - objB) / (2_r * eps);

      if (params.gradTarget == GradTarget::Current) {
        // Computing the dresidual by finite-differencing the residual is not accurate for Lie
        // rotation derivatives, but it seems sufficient in this case.
        ColumnVector<real> dresFDi = (resF - resB) * (1_r / (2_r * eps));
        for (int j = 0; j < _numDofs; ++j) {
          int row = j;
          int col = i;
          dresFD.SetValue(row, col, dresFDi[j]);
        }
      }
    }

    if (params.gradTarget == GradTarget::Current &&
        (params.ignoreColliderRotDres || params.ignoreColliderDres)) {
      auto const colliderDofOffset = GetRegistry().get<CDofOffset const>(_collider).dofsOffset;
      std::vector<int> ignoredDofs;
      if (params.ignoreColliderDres) {
        ignoredDofs.resize(_numDofsB);
        std::iota(ignoredDofs.begin(), ignoredDofs.end(), colliderDofOffset);
      } else {
        ignoredDofs.resize(RigidSize::kDRot);
        std::iota(ignoredDofs.begin(), ignoredDofs.end(), colliderDofOffset + RigidSize::kDTrans);
      }
      SetZeroOnCols(AsView(dres), ignoredDofs, 0, AsConstView(dres));
      SetZeroOnCols(AsView(dresFD), ignoredDofs, 0, AsConstView(dresFD));
      SetZeroOnRows(AsView(dres), ignoredDofs, 0, 1.0);
      SetZeroOnRows(AsView(dresFD), ignoredDofs, 0, 1.0);
    }

    auto const resTol = params.resTol;
    ColumnVector<real> resDelta = res0 - resFD;
    auto const resNorm = resDelta.Norm();
    if (resNorm > 1e-6_r) {
      EXPECT_NEAR(resNorm / Max(res0.Norm(), resFD.Norm()), 0_r, resTol);
    }
    if (params.gradTarget == GradTarget::Current) {
      auto const dresTol = params.dresTol;
      EXPECT_EQ(dres.Pointers(), dresFD.Pointers()); // Expect same sparsity
      EXPECT_EQ(dres.Indices(), dresFD.Indices()); // Expect same sparsity
      ColumnVector<real> dresValueDelta = AsConstView(dres.Values()) - AsConstView(dresFD.Values());
      EXPECT_NEAR(dresValueDelta.Norm() / Max(dres.Norm(), dresFD.Norm()), 0_r, dresTol);
    }
  }

  template <bool kIsSync>
  bool CollidingHasContact() const {
    for (auto const& collision :
         GetRegistry().get<CActiveCollisions<kIsSync, TimeStep::Current> const>(_colliding)) {
      if (!collision.collisionResult.Empty()) {
        return true;
      }
    }
    return false;
  }

  // Expects the colliding contact skin to carry a linear element map, and its contacts to be
  // detected only as sync or only as async, so that the corresponding mapped assembly runs.
  template <bool kIsSync>
  void ExpectMappedContactSkinCollisions() const {
    EXPECT_TRUE(GetRegistry().all_of<CLinearContactSkinElementMap>(_colliding));
    EXPECT_TRUE(CollidingHasContact<kIsSync>());
    EXPECT_FALSE(CollidingHasContact<!kIsSync>());
  }

  // Removes the linear element map of the colliding contact skin, so that its contact is assembled
  // through colliding contact Jacobians.
  void RemoveLinearContactSkinMap() {
    auto& reg = GetRegistry();
    InitializeContactSkinningColumnCoalescingMap(
        reg.get<CFemSurfaceDiscretization const>(_colliding),
        reg.get<CContactSkinningData>(_colliding));
    reg.remove<CLinearContactSkinElementMap>(_colliding);
  }

  // Compares the mapped contact-skin assembly, with contacts detected only as sync or only as
  // async, with the legacy one through colliding contact Jacobians, which runs once the linear
  // element map is removed.
  template <bool kIsSync>
  void RunMappedVsLegacyContactSkinTest() {
    auto const gradTarget = GetParam().gradTarget;
    InitializeSceneFromParams();
    InitState();
    PreStep();
    ComputeResponse(gradTarget);
    ExpectMappedContactSkinCollisions<kIsSync>();
    double const mappedObjective = _objective;
    ColumnVector<real> const mappedResidual = _residual.Duplicate();
    SparseMatrix<real> const mappedDResidual = _dresidual.Duplicate();

    // Assemblies without the dresidual, as in line searches, or without the residual, must match
    // the full one.
    real constexpr kRelTol = 1e-5_r;
    IslandAssembly const objRes =
        AssembleSingleIsland(GetRegistry(), gradTarget, /*assemRes*/ true, /*assemDRes*/ false);
    IslandAssembly const objOnly =
        AssembleSingleIsland(GetRegistry(), gradTarget, /*assemRes*/ false, /*assemDRes*/ false);
    EXPECT_NEAR(objRes.objective, mappedObjective, kRelTol * Abs(mappedObjective));
    EXPECT_NEAR(objOnly.objective, mappedObjective, kRelTol * Abs(mappedObjective));
    ColumnVector<real> const objResDelta = objRes.residual - mappedResidual;
    EXPECT_LE(objResDelta.Norm(), kRelTol * mappedResidual.Norm());

    RemoveLinearContactSkinMap();
    InitState();
    PreStep();
    ComputeResponse(gradTarget);

    // Both paths evaluate the same contacts and differ only in summation order.
    EXPECT_NEAR(mappedObjective, _objective, kRelTol * Abs(mappedObjective));
    ColumnVector<real> const resDelta = mappedResidual - _residual;
    EXPECT_LE(resDelta.Norm(), kRelTol * mappedResidual.Norm());
    // Values are compared positionally, so the sparsity patterns must match.
    ASSERT_TRUE(test::EqualSpan(mappedDResidual.Pointers(), _dresidual.Pointers()));
    ASSERT_TRUE(test::EqualSpan(mappedDResidual.Indices(), _dresidual.Indices()));
    ColumnVector<real> const dresDelta =
        AsConstView(mappedDResidual.Values()) - AsConstView(_dresidual.Values());
    EXPECT_LE(dresDelta.Norm(), kRelTol * mappedDResidual.Norm());
  }

  // Adds worker threads, densifies the colliding samples and enlarges the colliders, so that
  // contact pairs are expensive enough to be split across tasks. Must be called before SetUp.
  void UseParallelDenseContact() {
    _numWorkerThreads = 4;
    _collidingBoundaryElementType = ActorBoundaryElementType::ExperimentalP1Q16;
    _largeColliders = true;
  }

  // Compares contact assembly split across tasks with serial assembly, including the contact forces
  // retained for queries. If removeLinearContactSkinMap, the colliding contact skin is assembled
  // through colliding contact Jacobians.
  void RunSerialVsParallelTest(bool removeLinearContactSkinMap = false) {
    InitializeSceneFromParams();
    if (removeLinearContactSkinMap) {
      RemoveLinearContactSkinMap();
    }
    GetRegistry().emplace_or_replace<TagQueryActiveContacts>(_colliding);
    ThreadedAssembly const serial = AssembleWithThreading(/*singleThreaded*/ true);
    ThreadedAssembly const parallel = AssembleWithThreading(/*singleThreaded*/ false);

    // Both evaluate the same contacts and differ only in summation order.
    real constexpr kRelTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-12_r : 1e-5_r;
    EXPECT_NEAR(
        serial.island.objective, parallel.island.objective, kRelTol * Abs(serial.island.objective));
    ColumnVector<real> const resDelta = serial.island.residual - parallel.island.residual;
    EXPECT_LE(resDelta.Norm(), kRelTol * serial.island.residual.Norm());
    Matrix<real> const dresDelta{serial.island.dresidual - parallel.island.dresidual};
    EXPECT_LE(dresDelta.Norm(), kRelTol * serial.island.dresidual.Norm());

    ASSERT_EQ(serial.forces.size(), parallel.forces.size());
    real forceScale = 0_r;
    for (auto const& forces : serial.forces) {
      for (Real3 const& force : forces) {
        for (int i = 0; i < kSpaceDim3; ++i) {
          forceScale = Max(forceScale, Abs(force[i]));
        }
      }
    }
    EXPECT_GT(forceScale, 0_r);
    for (int c = 0; c < isize(serial.forces); ++c) {
      ASSERT_EQ(serial.forces[c].size(), parallel.forces[c].size());
      for (int p = 0; p < isize(serial.forces[c]); ++p) {
        for (int i = 0; i < kSpaceDim3; ++i) {
          EXPECT_NEAR(serial.forces[c][p][i], parallel.forces[c][p][i], kRelTol * forceScale);
        }
      }
    }
  }

  Actor* InitStaticCube(
      real scale,
      TransformRT transform,
      real coulombCoefficient,
      real viscousCoefficient,
      real dampingCoefficient) {
    EnlargeColliderIfRequested(scale, transform);
    RigidActorParams params;
    params.isStatic = true;
    params.layer = "Collider";
    params.shape = _mochiContext->LoadShapeFromFile(
        test::GetAssetPath("cube/cube_mesh.mochi.json"),
        Real3{scale, scale, scale},
        TransformRT::Identity(),
        test::ExpectOK{});
    params.worldFromLocal = transform;

    params.contact.frictionWithColliderNormal = GetParam().frictionWithColliderNormal;
    params.contact.viscousFrictionCoefficient = viscousCoefficient;
    params.contact.coulombFrictionCoefficient = coulombCoefficient;
    params.contact.normalViscousDampingCoefficient = dampingCoefficient;
    params.contact.penaltyThresholdDefault = 5e-3_r;
    params.colliderType = ColliderType::Box;

    return _scene->CreateRigidActor(params, test::ExpectOK{});
  }

  Actor* InitRigidCube(
      real scale,
      TransformRT transform,
      bool isCollider,
      real coulombCoefficient,
      real viscousCoefficient,
      real dampingCoefficient) {
    auto& reg = GetRegistry();
    if (isCollider) {
      EnlargeColliderIfRequested(scale, transform);
    }

    RigidActorParams params;
    params.density = 1e-10_r; // Negligible density to isolate contact forces from inertia.
    params.layer = isCollider ? "Collider" : "Colliding";
    params.shape = _mochiContext->LoadShapeFromFile(
        test::GetAssetPath("cube/cube_mesh.mochi.json"),
        Real3{scale, scale, scale},
        TransformRT::Identity(),
        test::ExpectOK{});
    params.worldFromLocal = transform;

    params.contact.frictionWithColliderNormal = GetParam().frictionWithColliderNormal;
    params.contact.viscousFrictionCoefficient = viscousCoefficient;
    params.contact.coulombFrictionCoefficient = coulombCoefficient;
    params.contact.normalViscousDampingCoefficient = dampingCoefficient;
    params.contact.penaltyThresholdDefault = 5e-3_r;
    params.colliderType = isCollider ? ColliderType::Box : ColliderType::None;

    auto* actor = _scene->CreateRigidActor(params, test::ExpectOK{});
    auto entity = mochi::GetEntity(reg, actor->GetHandle(), test::ExpectOK{});
    reg.get<CTimeIntegratorState>(entity).dtStage = kDtStage;
    return actor;
  }

  Actor* InitSoftCube(
      TransformRT const& transform,
      bool isCollider,
      bool useDeepFlow,
      real coulombCoefficient,
      real viscousCoefficient,
      real dampingCoefficient) {
    auto& reg = GetRegistry();

    // Create soft actor. This is a cube of size 1, centered at 0.5
    SoftActorParams caparams;
    caparams.material.density =
        1e-10_r; // Negligible density to isolate contact forces from inertia.
    caparams.material.type = SoftMaterialType::NeoHookean;
    caparams.material.neoHookean.youngsModulus = 1e-6_r;
    caparams.layer = isCollider ? "Collider" : "Colliding";
    if (_useContactSkin && !isCollider) {
      caparams.shape =
          _scene->GetContext()->CreateModelShape(CreateMixedContactSkinModel(), test::ExpectOK{});
      caparams.useContactSkin = true;
      // Several samples per element, so element blocks accumulate more than one sample.
      caparams.boundaryElementType = _collidingBoundaryElementType;
    } else {
      auto&& [unitCubeCoordinates, unitCubeConnectivity] = test::CreateMinimalTetMeshUnitCube();
      caparams.shape = _scene->GetContext()->CreateTetMeshShape(
          Flatten(MakeSpan(unitCubeCoordinates)),
          Flatten(MakeSpan(unitCubeConnectivity)),
          test::ExpectOK{});
    }
    caparams.worldFromLocal = transform;

    caparams.contact.frictionWithColliderNormal = GetParam().frictionWithColliderNormal;
    caparams.contact.viscousFrictionCoefficient = viscousCoefficient;
    caparams.contact.coulombFrictionCoefficient = coulombCoefficient;
    caparams.contact.normalViscousDampingCoefficient = dampingCoefficient;

    ExperimentalSoftActorParams experimentalParams;
    if (isCollider) {
      experimentalParams.colliderType = ColliderType::Sdf;
      experimentalParams.sdf.resolutionDelta = {0.04_r, 0.04_r, 0.04_r};
      experimentalParams.sdf.resolutionMode = GridSdfResolutionMode::Explicit;
      caparams.contact.penaltyThresholdDefault = 0_r;

      if (useDeepFlow) {
        DeepModelParams flowParams;
        flowParams.shiftX = 0.49734753_r;
        flowParams.shiftY = 0.51221045_r;
        flowParams.shiftZ = 0.49657665_r;
        flowParams.scale = 1.3309090553162923_r;
        flowParams.deepModelPath = test::GetAssetPath("cube/cube_minimal_flow_revised.mochi.h5");
        flowParams.numDof = 24;
        mochi::ShapeHandle flow =
            CreateDeepFlowShape(_scene->GetContext(), flowParams, test::ExpectOK());
        experimentalParams.flow = flow;
        caparams.contact.objScale = 1_r;
        caparams.contact.distanceErrorBound = flowParams.errorBound;
      }
    }
    auto* actor = CreateSoftActor(_scene, caparams, experimentalParams, test::ExpectOK{});
    auto entity = mochi::GetEntity(reg, actor->GetHandle(), test::ExpectOK{});
    reg.get<CTimeIntegratorState>(entity).dtStage = kDtStage;

    return actor;
  }

  // Rotation applied to a whole scene, so that the actors' root rotations are not the identity.
  static Quaternion SceneRotation() {
    return Quaternion::FromRotationVector(Real3(0.3_r, -0.4_r, 0.5_r));
  }

  // Unit-cube contact skin whose node i is embedded at tet node i plus a zero-sum combination of
  // the four nodes of the opposite z face, so that the skin keeps the unit-cube geometry while
  // each skin node depends on five physics nodes.
  static ModelData CreateMixedContactSkinModel() {
    ModelData model = test::CreateUnitCubeContactSkinModel();
    auto& skinning = *model.contactSkinMesh->skinning;
    // Unit-cube node n is at (n & 1, (n >> 1) & 1, n >> 2), so x(f) - x(f+1) - x(f+2) + x(f+3) = 0
    // for the first node f of either z face.
    std::array<real, 4> constexpr kFaceWeights = {0.1_r, -0.1_r, -0.1_r, 0.1_r};
    SkinningData mixed;
    mixed.weightsPerNode = 1 + isize(kFaceWeights);
    for (int i = 0; i < isize(skinning.indices); ++i) {
      mixed.indices.push_back(skinning.indices[i]);
      mixed.weights.push_back(skinning.weights[i]);
      int const oppositeFace = skinning.indices[i] < 4 ? 4 : 0;
      for (int k = 0; k < isize(kFaceWeights); ++k) {
        mixed.indices.push_back(oppositeFace + k);
        mixed.weights.push_back(kFaceWeights[k]);
      }
    }
    skinning = std::move(mixed);
    return model;
  }

  void InitSoftCubeStageStart(entt::entity e) {
    auto& reg = GetRegistry();

    // Set up contact samples at stage start
    if (reg.all_of<TagUseDeformableContactSkin>(e)) {
      ecs::InvokeOnEntity(
          linear_contact_skin::UpdatePositionsAndSamples<TimeStep::StageStart>, reg, e);
    } else {
      ecs::InvokeOnEntity(
          UpdateCollisionSamplePositions<
              CFemBoundaryDiscretization,
              TimeStep::StageStart,
              kSpaceDim3>,
          reg,
          e);
    }

    // Set up the tetrahedral map at stage start
    ecs::TryInvokeOnEntity(soft::UpdateMap<TimeStep::StageStart>, reg, e);
  }

 protected:
  void EnlargeColliderIfRequested(real& scale, TransformRT& transform) const {
    if (_largeColliders) {
      scale *= kLargeColliderScale;
      real const halfScale = 0.5_r * scale;
      transform = transform * TransformRT{Real3{-halfScale, -halfScale, -halfScale}};
    }
  }

 private:
  struct ThreadedAssembly {
    IslandAssembly island;
    // Contact forces of each collision of the colliding actor, sync then async.
    DynamicArray<DynamicArray<Real3>> forces;
  };

  ThreadedAssembly AssembleWithThreading(bool singleThreaded) {
    auto& reg = GetRegistry();
    auto const forEachCollision = [&](auto&& fn) {
      auto* const syncCollisions =
          reg.try_get<CActiveCollisions</*kIsSync*/ true, TimeStep::Current>>(_colliding);
      auto* const asyncCollisions =
          reg.try_get<CActiveCollisions</*kIsSync*/ false, TimeStep::Current>>(_colliding);
      if (syncCollisions) {
        std::ranges::for_each(*syncCollisions, fn);
      }
      if (asyncCollisions) {
        std::ranges::for_each(*asyncCollisions, fn);
      }
    };
    forEachCollision([](auto& collision) { collision.collisionResult.forcePerUnitArea.clear(); });

    _mochiContext->SetIsSingleThreaded(singleThreaded);
    InitState();
    PreStep();
    ThreadedAssembly result{.island = AssembleSingleIsland(reg, GradTarget::Current), .forces = {}};
    forEachCollision([&](auto const& collision) {
      result.forces.push_back(collision.collisionResult.forcePerUnitArea);
    });
    return result;
  }
};

class MochiRigidStaticContact : public MochiContactTestBase {
  TransformRT _pos{};
  TransformRT _posOld{};

 public:
  MochiRigidStaticContact() {
    _numDofsA = 6; // Rigid
    _numDofsB = 0; // Static
    _numDofs = _numDofsA + _numDofsB;
  }

  void InitializeScene(real coulombCoefficient, real viscousCoefficient, real dampingCoefficient)
      override {
    auto& reg = GetRegistry();

    auto transform =
        TransformRT(Quaternion::FromRotationVector(Real3(-1_r, -0.5_r, -0.4_r)), Real3{});
    auto* actorA =
        InitRigidCube(1_r, {}, false, coulombCoefficient, viscousCoefficient, dampingCoefficient);
    transform = TransformRT(
        Quaternion::FromRotationVector(Real3(1_r, -0.1_r, 0.2_r)), Real3(0.8_r, 0.8_r, 0.8_r));
    auto* actorB =
        InitStaticCube(1_r, transform, coulombCoefficient, viscousCoefficient, dampingCoefficient);
    InitActors(actorA, actorB);

    // Add velocity to the actor
    _pos = reg.get<CRigidState<TimeStep::Current>>(_colliding).value;
    auto& posOld = reg.get<CRigidState<TimeStep::StageStart>>(_colliding).value;
    posOld.SetTranslation(_pos.GetTranslation() + Real3(0.01_r, -0.02_r, 0.03_r));
    posOld.SetRotation(
        Quaternion::FromRotationVector(Vec4r(-0.2_r, 0.1_r, -0.1_r)) * _pos.GetRotation());
    SetStageStartTransformFromState(_colliding, posOld);
    _posOld = posOld;
  }

  void InitState() override {
    auto& reg = GetRegistry();
    reg.get<CRigidState<TimeStep::Current>>(_colliding).value = _pos;
    SetTransformFromState(_colliding, _pos);
    reg.get<CRigidState<TimeStep::StageStart>>(_colliding).value = _posOld;
    SetStageStartTransformFromState(_colliding, _posOld);
  }

  void AddToState(int globalDof, real eps) override {
    EXPECT_GT(_numDofs, globalDof);
    auto [entity, localDof] = GetEntityAndLocalDofFromGlobalDof(globalDof);
    Vec4r delta = SimdBasisVector(localDof % 3) * eps;
    auto& reg = GetRegistry();
    auto& rigidState = reg.get<CRigidState<TimeStep::Current>>(entity).value;
    if (localDof < 3) {
      // Change state of the rigid translation
      rigidState.SetTranslation(rigidState.GetTranslation() + ToReal3(delta));
    } else {
      // Change state of the rigid rotation
      rigidState.SetRotation(Quaternion::FromRotationVector(delta) * rigidState.GetRotation());
    }
    SetTransformFromState(entity, rigidState);
  }

  void AddToOldState(int globalDof, real eps) override {
    EXPECT_GT(_numDofs, globalDof);
    auto [entity, localDof] = GetEntityAndLocalDofFromGlobalDof(globalDof);
    Vec4r delta = SimdBasisVector(localDof % 3) * eps;
    auto& reg = GetRegistry();
    auto& rigidState = reg.get<CRigidState<TimeStep::StageStart>>(entity).value;
    if (localDof < 3) {
      // Change state of the rigid translation
      rigidState.SetTranslation(rigidState.GetTranslation() + ToReal3(delta));
    } else {
      // Change state of the rigid rotation
      rigidState.SetRotation(Quaternion::FromRotationVector(delta) * rigidState.GetRotation());
    }
    SetStageStartTransformFromState(entity, rigidState);
  }
};

TEST_P(MochiRigidStaticContact, ContactAssembly) {
  RunTest();
}

INSTANTIATE_TEST_SUITE_P(
    FrictionVariations,
    MochiRigidStaticContact,
    ::testing::Values(
        TestParams{"NoFriction", GradTarget::Current, 0_r, 0_r, 0_r, false, 2e-3_r, 3e-2_r},
        TestParams{"Viscous", GradTarget::Current, 0_r, 1_r, 0_r, false, 2e-3_r, 3e-2_r},
        TestParams{"Coulomb", GradTarget::Current, 0.5_r, 0_r, 0_r, false, 2e-3_r, 3e-2_r},
        TestParams{"Damping", GradTarget::Current, 0_r, 0_r, 1_r, false, 2e-3_r, 4e-2_r},
        TestParams{"NoFrictionExplicit", GradTarget::Current, 0_r, 0_r, 0_r, true, 2e-3_r, 3e-2_r},
        TestParams{"ViscousExplicit", GradTarget::Current, 0_r, 1_r, 0_r, true, 1e-3_r, 3e-2_r},
        TestParams{"CoulombExplicit", GradTarget::Current, 0.5_r, 0_r, 0_r, true, 2e-3_r, 3e-2_r},
        TestParams{"DampingExplicit", GradTarget::Current, 0_r, 0_r, 1_r, true, 2e-3_r, 4e-2_r},
        TestParams{"NoFrictionPrevious", GradTarget::Previous, 0_r, 0_r, 0_r, true, {}, {}},
        TestParams{"ViscousPrevious", GradTarget::Previous, 0_r, 1_r, 0_r, true, 1e-3_r, {}},
        TestParams{"CoulombPrevious", GradTarget::Previous, 0.5_r, 0_r, 0_r, true, 1e-3_r, {}},
        TestParams{"DampingPrevious", GradTarget::Previous, 0_r, 0_r, 1_r, true, 3e-3_r, {}}),
    [](::testing::TestParamInfo<TestParams> const& info) { return info.param.name; });

class MochiRigidMovingStaticContact : public MochiRigidStaticContact {
 public:
  void InitializeScene(real coulombCoefficient, real viscousCoefficient, real dampingCoefficient)
      override {
    MochiRigidStaticContact::InitializeScene(
        coulombCoefficient, viscousCoefficient, dampingCoefficient);
    auto& rootTransform = GetRegistry().get<CRootTransform>(_collider);
    rootTransform.worldFromLocalPrev = TransformRT(
        Quaternion::FromRotationVector(Real3(0.05_r, -0.03_r, 0.04_r)) *
            rootTransform.worldFromLocal.GetRotation(),
        rootTransform.worldFromLocal.GetTranslation() + Real3(0.02_r, -0.01_r, 0.03_r));
  }

  void TestContactForceAdjoints() {
    auto& reg = GetRegistry();
    reg.emplace<TagQueryActiveContacts>(_colliding);
    reg.emplace<CQueryActorContactForces>(_colliding);
    EmplaceDifferentiableContactComponents(
        reg, _colliding, reg.get<CActorDofInfo const>(_colliding));

    InitState();
    ComputeResponse(GradTarget::Current);
    auto& collisions = reg.get<CActiveCollisions</*kIsSync*/ false, TimeStep::Current>>(_colliding);
    Real3 const forceAdjoint{0.7_r, -0.4_r, 0.2_r};
    int numContacts = 0;
    for (auto& collision : collisions) {
      numContacts += isize(collision.collisionResult.forcePerUnitArea);
      for (auto& force : collision.collisionResult.forcePerUnitArea) {
        force = forceAdjoint;
      }
    }
    ASSERT_GT(numContacts, 0);
    AccumulateContactForceAdjoints(reg);
    ColumnVector<real> const analytical =
        reg.get<CDiffContactGrad<GradTarget::Current> const>(_colliding);

    auto const evaluateLoss = [&]() {
      ComputeResponse(GradTarget::Current);
      real loss = 0_r;
      int count = 0;
      for (auto const& collision : collisions) {
        count += isize(collision.collisionResult.forcePerUnitArea);
        for (auto const& force : collision.collisionResult.forcePerUnitArea) {
          loss += Dot(forceAdjoint, force);
        }
      }
      EXPECT_EQ(count, numContacts);
      return loss;
    };
    ColumnVector<real> finiteDiff(_numDofs);
    real const eps = GetParam().eps;
    for (int i = 0; i < _numDofs; ++i) {
      InitState();
      AddToState(i, eps);
      real const lossPlus = evaluateLoss();
      InitState();
      AddToState(i, -eps);
      finiteDiff[i] = (lossPlus - evaluateLoss()) / (2_r * eps);
    }
    ColumnVector<real> const delta = analytical - finiteDiff;
    EXPECT_NEAR(delta.Norm() / Max(analytical.Norm(), finiteDiff.Norm()), 0_r, GetParam().resTol);
  }
};

TEST_P(MochiRigidMovingStaticContact, ContactAssembly) {
  RunTest();
  if (GetParam().gradTarget == GradTarget::Current && GetParam().explicitNormals) {
    TestContactForceAdjoints();
  }
}

INSTANTIATE_TEST_SUITE_P(
    FrictionVariations,
    MochiRigidMovingStaticContact,
    ::testing::Values(
        TestParams{"NoFriction", GradTarget::Current, 0_r, 0_r, 0_r, false, 2e-3_r, 3e-2_r},
        TestParams{"NoFrictionExplicit", GradTarget::Current, 0_r, 0_r, 0_r, true, 2e-3_r, 3e-2_r},
        TestParams{"ViscousExplicit", GradTarget::Current, 0_r, 1_r, 0_r, true, 2e-3_r, 3e-2_r},
        // With GradTarget::Previous only dissipation responds, so friction is needed to check the
        // stage-start collider frame.
        TestParams{"ViscousPrevious", GradTarget::Previous, 0_r, 1_r, 0_r, true, 2e-3_r, {}}),
    [](::testing::TestParamInfo<TestParams> const& info) { return info.param.name; });

class MochiRigidRigidContact : public MochiContactTestBase {
  TransformRT _posA{};
  TransformRT _posB{};
  TransformRT _posAOld{};
  TransformRT _posBOld{};

 public:
  MochiRigidRigidContact() {
    _numDofsA = 6; // Rigid
    _numDofsB = 6; // Rigid
    _numDofs = _numDofsA + _numDofsB;
  }

  void InitializeScene(real coulombCoefficient, real viscousCoefficient, real dampingCoefficient)
      override {
    auto& reg = GetRegistry();

    auto transform =
        TransformRT(Quaternion::FromRotationVector(Real3(-1_r, -0.5_r, -0.4_r)), Real3{});
    auto* actorA =
        InitRigidCube(1_r, {}, false, coulombCoefficient, viscousCoefficient, dampingCoefficient);
    transform = TransformRT(
        Quaternion::FromRotationVector(Real3(1_r, -0.1_r, 0.2_r)), Real3(0.8_r, 0.8_r, 0.8_r));
    auto* actorB = InitRigidCube(
        1_r, transform, true, coulombCoefficient, viscousCoefficient, dampingCoefficient);
    InitActors(actorA, actorB);

    // Add velocity to the actors
    _posA = reg.get<CRigidState<TimeStep::Current>>(_colliding).value;
    auto& posAold = reg.get<CRigidState<TimeStep::StageStart>>(_colliding).value;
    posAold.SetTranslation(_posA.GetTranslation() + Real3(0.01_r, -0.02_r, 0.03_r));
    posAold.SetRotation(
        Quaternion::FromRotationVector(Vec4r(-0.2_r, 0.1_r, -0.1_r)) * _posA.GetRotation());
    SetStageStartTransformFromState(_colliding, posAold);
    _posAOld = posAold;
    _posB = reg.get<CRigidState<TimeStep::Current>>(_collider).value;
    auto& posBold = reg.get<CRigidState<TimeStep::StageStart>>(_collider).value;
    posBold.SetTranslation(_posB.GetTranslation() + Real3(0.02_r, 0.01_r, -0.01_r));
    posBold.SetRotation(
        Quaternion::FromRotationVector(Vec4r(-0.1_r, -0.1_r, -0.2_r)) * _posB.GetRotation());
    SetStageStartTransformFromState(_collider, posBold);
    _posBOld = posBold;
  }

  void InitState() override {
    auto& reg = GetRegistry();
    reg.get<CRigidState<TimeStep::Current>>(_colliding).value = _posA;
    reg.get<CRigidState<TimeStep::Current>>(_collider).value = _posB;
    SetTransformFromState(_colliding, _posA);
    SetTransformFromState(_collider, _posB);
    reg.get<CRigidState<TimeStep::StageStart>>(_colliding).value = _posAOld;
    reg.get<CRigidState<TimeStep::StageStart>>(_collider).value = _posBOld;
    SetStageStartTransformFromState(_colliding, _posAOld);
    SetStageStartTransformFromState(_collider, _posBOld);
  }

  void AddToState(int globalDof, real eps) override {
    EXPECT_GT(_numDofs, globalDof);
    auto [entity, localDof] = GetEntityAndLocalDofFromGlobalDof(globalDof);
    Vec4r delta = SimdBasisVector(localDof % 3) * eps;
    auto& reg = GetRegistry();
    auto& rigidState = reg.get<CRigidState<TimeStep::Current>>(entity).value;
    if (localDof < 3) {
      // Change state of the rigid translation
      rigidState.SetTranslation(rigidState.GetTranslation() + ToReal3(delta));
    } else {
      // Change state of the rigid rotation
      rigidState.SetRotation(Quaternion::FromRotationVector(delta) * rigidState.GetRotation());
    }
    SetTransformFromState(entity, rigidState);
  }

  void AddToOldState(int globalDof, real eps) override {
    EXPECT_GT(_numDofs, globalDof);
    auto [entity, localDof] = GetEntityAndLocalDofFromGlobalDof(globalDof);
    Vec4r delta = SimdBasisVector(localDof % 3) * eps;
    auto& reg = GetRegistry();
    auto& rigidState = reg.get<CRigidState<TimeStep::StageStart>>(entity).value;
    if (localDof < 3) {
      // Change state of the rigid translation
      rigidState.SetTranslation(rigidState.GetTranslation() + ToReal3(delta));
    } else {
      // Change state of the rigid rotation
      rigidState.SetRotation(Quaternion::FromRotationVector(delta) * rigidState.GetRotation());
    }
    SetStageStartTransformFromState(entity, rigidState);
  }
};

TEST_P(MochiRigidRigidContact, ContactAssembly) {
  RunTest();
}

INSTANTIATE_TEST_SUITE_P(
    FrictionVariations,
    MochiRigidRigidContact,
    ::testing::Values(
        TestParams{"NoFriction", GradTarget::Current, 0_r, 0_r, 0_r, false, 2e-3_r, 6e-2_r},
        TestParams{"Viscous", GradTarget::Current, 0_r, 1_r, 0_r, false, 2e-3_r, 4e-2_r},
        TestParams{"Coulomb", GradTarget::Current, 0.5_r, 0_r, 0_r, false, 4e-3_r, 6e-2_r},
        TestParams{"Damping", GradTarget::Current, 0_r, 0_r, 1_r, false, 2e-3_r, 3e-2_r},
        TestParams{"NoFrictionExplicit", GradTarget::Current, 0_r, 0_r, 0_r, true, 2e-3_r, 7e-2_r},
        TestParams{"ViscousExplicit", GradTarget::Current, 0_r, 1_r, 0_r, true, 2e-3_r, 4e-2_r},
        TestParams{"CoulombExplicit", GradTarget::Current, 0.5_r, 0_r, 0_r, true, 2e-3_r, 7e-2_r},
        TestParams{"DampingExplicit", GradTarget::Current, 0_r, 0_r, 1_r, true, 2e-3_r, 3e-2_r},
        TestParams{"NoFrictionPrevious", GradTarget::Previous, 0_r, 0_r, 0_r, true, {}, {}},
        TestParams{"ViscousPrevious", GradTarget::Previous, 0_r, 1_r, 0_r, true, 2e-3_r, {}},
        TestParams{"CoulombPrevious", GradTarget::Previous, 0.5_r, 0_r, 0_r, true, 1e-3_r, {}},
        TestParams{"DampingPrevious", GradTarget::Previous, 0_r, 0_r, 1_r, true, 4e-3_r, {}}),
    [](::testing::TestParamInfo<TestParams> const& info) { return info.param.name; });

class MochiSkinnedArticulatedStaticContact : public MochiContactTestBase {
  ColumnVector<real> _pose;

  void UpdateFullState() {
    UpdateArticulatedFullPose(GetRegistry(), _colliding);
  }

 public:
  MochiSkinnedArticulatedStaticContact() {
    _numDofsA = 6;
    _numDofsB = 0;
    _numDofs = _numDofsA;
  }

  void InitializeScene(real coulombCoefficient, real viscousCoefficient, real dampingCoefficient)
      override {
    Actor* const articulatedActor = CreateThreeBoneSkinnedActor(
        _mochiContext,
        _scene,
        coulombCoefficient,
        viscousCoefficient,
        dampingCoefficient,
        _collidingBoundaryElementType);

    TransformRT const colliderTransform{
        Quaternion::FromRotationVector(Real3{0.1_r, -0.1_r, 0.2_r}), Real3{0.6_r, 0.6_r, 0.6_r}};
    Actor* const collider = InitStaticCube(
        1_r, colliderTransform, coulombCoefficient, viscousCoefficient, dampingCoefficient);
    InitActors(articulatedActor, collider);

    auto& pose = GetRegistry().get<CArticulatedReducedPose<TimeStep::Current>>(_colliding).value;
    UpdateFullState();
    _pose = pose.Duplicate();
    island::PreStep(GetRegistry(), /*updateSleepState*/ true);
  }

  void InitState() override {
    GetRegistry().get<CArticulatedReducedPose<TimeStep::Current>>(_colliding).value = _pose;
    UpdateFullState();
  }

  void AddToState(int globalDof, real eps) override {
    auto [entity, localDof] = GetEntityAndLocalDofFromGlobalDof(globalDof);
    MOCHI_ASSERT_VERBOSE(entity == _colliding);
    AddToArticulatedPose(GetRegistry(), _colliding, localDof, eps);
  }
};

TEST_P(MochiSkinnedArticulatedStaticContact, ContactAssembly) {
  RunTest();
}

INSTANTIATE_TEST_SUITE_P(
    MultipleBones,
    MochiSkinnedArticulatedStaticContact,
    ::testing::Values(
        TestParams{"NoFriction", GradTarget::Current, 0_r, 0_r, 0_r, false, 2e-3_r, 2.6e-1_r}),
    [](::testing::TestParamInfo<TestParams> const& info) { return info.param.name; });

// Async contact sums of the skin, split across tasks.
class MochiParallelSkinnedArticulatedStaticContact : public MochiSkinnedArticulatedStaticContact {
 public:
  MochiParallelSkinnedArticulatedStaticContact() {
    UseParallelDenseContact();
  }
};

TEST_P(MochiParallelSkinnedArticulatedStaticContact, MatchesSerialAssembly) {
  RunSerialVsParallelTest();
}

INSTANTIATE_TEST_SUITE_P(
    MultipleBones,
    MochiParallelSkinnedArticulatedStaticContact,
    ::testing::Values(TestParams{.name = "NoFriction"}),
    [](::testing::TestParamInfo<TestParams> const& info) { return info.param.name; });

// A skinned articulated actor against a dynamic rigid box, in sync contact.
class MochiSkinnedArticulatedRigidContact : public MochiContactTestBase {
  ColumnVector<real> _pose;
  TransformRT _boxState{};

 public:
  MochiSkinnedArticulatedRigidContact() {
    _numDofsA = 6;
    _numDofsB = 6;
    _numDofs = _numDofsA + _numDofsB;
  }

  void InitializeScene(real coulombCoefficient, real viscousCoefficient, real dampingCoefficient)
      override {
    Actor* const articulatedActor = CreateThreeBoneSkinnedActor(
        _mochiContext,
        _scene,
        coulombCoefficient,
        viscousCoefficient,
        dampingCoefficient,
        _collidingBoundaryElementType);

    TransformRT const boxTransform{
        Quaternion::FromRotationVector(Real3{0.1_r, -0.1_r, 0.2_r}), Real3{0.6_r, 0.6_r, 0.6_r}};
    Actor* const box = InitRigidCube(
        1_r, boxTransform, true, coulombCoefficient, viscousCoefficient, dampingCoefficient);
    InitActors(articulatedActor, box);

    auto& reg = GetRegistry();
    UpdateArticulatedFullPose(reg, _colliding);
    _pose = reg.get<CArticulatedReducedPose<TimeStep::Current>>(_colliding).value.Duplicate();
    _boxState = reg.get<CRigidState<TimeStep::Current>>(_collider).value;
    island::PreStep(reg, /*updateSleepState*/ true);
  }

  void InitState() override {
    auto& reg = GetRegistry();
    reg.get<CArticulatedReducedPose<TimeStep::Current>>(_colliding).value = _pose;
    UpdateArticulatedFullPose(reg, _colliding);
    reg.get<CRigidState<TimeStep::Current>>(_collider).value = _boxState;
    SetTransformFromState(_collider, _boxState);
  }

  void AddToState(int globalDof, real eps) override {
    auto [entity, localDof] = GetEntityAndLocalDofFromGlobalDof(globalDof);
    auto& reg = GetRegistry();
    if (entity == _colliding) {
      AddToArticulatedPose(reg, _colliding, localDof, eps);
      return;
    }
    Vec4r const delta = SimdBasisVector(localDof % 3) * eps;
    auto& rigidState = reg.get<CRigidState<TimeStep::Current>>(entity).value;
    if (localDof < 3) {
      rigidState.SetTranslation(rigidState.GetTranslation() + ToReal3(delta));
    } else {
      rigidState.SetRotation(Quaternion::FromRotationVector(delta) * rigidState.GetRotation());
    }
    SetTransformFromState(entity, rigidState);
  }
};

TEST_P(MochiSkinnedArticulatedRigidContact, ContactAssembly) {
  RunTest();
}

INSTANTIATE_TEST_SUITE_P(
    MultipleBones,
    MochiSkinnedArticulatedRigidContact,
    ::testing::Values(
        TestParams{
            .name = "NoFriction",
            .resTol = 2e-3_r,
            .dresTol = 2.6e-1_r,
            .ignoreColliderRotDres = true}),
    [](::testing::TestParamInfo<TestParams> const& info) { return info.param.name; });

// Sync contact sums of the skin and the rigid collider, split across tasks.
class MochiParallelSkinnedArticulatedRigidContact : public MochiSkinnedArticulatedRigidContact {
 public:
  MochiParallelSkinnedArticulatedRigidContact() {
    UseParallelDenseContact();
  }
};

TEST_P(MochiParallelSkinnedArticulatedRigidContact, MatchesSerialAssembly) {
  RunSerialVsParallelTest();
}

INSTANTIATE_TEST_SUITE_P(
    MultipleBones,
    MochiParallelSkinnedArticulatedRigidContact,
    ::testing::Values(TestParams{.name = "NoFriction"}),
    [](::testing::TestParamInfo<TestParams> const& info) { return info.param.name; });

// Contact with an articulated collider of two links: a free root link away from the colliding
// actor, and a child link on a spherical joint that overlaps it. The collider slice maps the rigid
// DoFs of the child link to the articulated DoFs through its auxiliary Jacobian.
class MochiLinkColliderContactTestBase : public MochiContactTestBase {
  ColumnVector<real> _colliderPose;
  DynamicArray<entt::entity> _colliderLinks;

  // Update the rigid states of the collider links from the articulated pose.
  void UpdateColliderLinks() {
    auto& reg = GetRegistry();
    UpdateArticulatedFullPose(reg, _collider);
    auto const& fullPose = reg.get<CArticulatedFullPose const>(_collider).value;
    for (entt::entity const link : _colliderLinks) {
      int const offset = reg.get<CDofOffset const>(link).poseOffset;
      auto& rigidState = reg.get<CRigidState<TimeStep::Current>>(link);
      rigid::EntitySetSolution(
          fullPose.MiddleRows(offset, RigidSize::kAll),
          ecs::Included<TagRigidActor>{},
          {},
          rigidState);
      SetTransformFromState(link, rigidState.value);
    }
  }

 protected:
  static constexpr int kNumColliderDofs = 9; // A free root and a spherical joint.

  // Scale and transform of the child link, before any enlargement requested by _largeColliders.
  real _childLinkScale = 1_r;
  TransformRT _childLinkTransform{
      Quaternion::FromRotationVector(Real3{0.1_r, -0.1_r, 0.2_r}),
      Real3{0.6_r, 0.6_r, 0.6_r}};

  Actor*
  InitLinkCollider(real coulombCoefficient, real viscousCoefficient, real dampingCoefficient) {
    TransformRT const rootTransform{Real3{1.8_r, 0.6_r, 0.6_r}};
    TransformRT childTransform = _childLinkTransform;
    real childScale = _childLinkScale;
    EnlargeColliderIfRequested(childScale, childTransform);
    ArticulatedActorParams colliderParams;
    colliderParams.joints.resize(2);
    colliderParams.joints[0].type = ArticulatedJointType::Free;
    colliderParams.joints[0].parentLinkFromJoint = rootTransform;
    colliderParams.joints[1].type = ArticulatedJointType::Spherical;
    colliderParams.joints[1].parentLinkFromJoint = Invert(rootTransform) * childTransform;
    auto const loadLinkShape = [&](real scale) {
      return _mochiContext->LoadShapeFromFile(
          test::GetAssetPath("cube/cube_mesh.mochi.json"),
          Real3{scale, scale, scale},
          TransformRT::Identity(),
          test::ExpectOK{});
    };
    std::array<ShapeHandle, 2> const linkShapes{loadLinkShape(1_r), loadLinkShape(childScale)};
    colliderParams.links.resize(2);
    for (int i = 0; i < 2; ++i) {
      auto& link = colliderParams.links[i];
      link.parentLink = i - 1;
      link.shape = linkShapes[i];
      link.layer = "Collider";
      link.colliderType = ColliderType::Box;
      link.hasGravity = false;
      link.density = 1e-10_r;
      link.contact.coulombFrictionCoefficient = coulombCoefficient;
      link.contact.viscousFrictionCoefficient = viscousCoefficient;
      link.contact.normalViscousDampingCoefficient = dampingCoefficient;
      link.contact.penaltyThresholdDefault = 5e-3_r;
    }
    Actor* const colliderActor = _scene->CreateArticulatedActor(colliderParams, test::ExpectOK{});

    auto& reg = GetRegistry();
    for (ActorHandle const link : colliderActor->GetNestedLinkActors(test::ExpectOK{})) {
      entt::entity const entity = GetEntity(link);
      reg.get<CTimeIntegratorState>(entity).dtStage = kDtStage;
      _colliderLinks.push_back(entity);
    }
    return colliderActor;
  }

  // Updates the collider links from the current articulated pose and saves it for
  // ResetColliderPose. Must be called after InitActors.
  void SaveColliderPose() {
    UpdateColliderLinks();
    _colliderPose =
        GetRegistry().get<CArticulatedReducedPose<TimeStep::Current>>(_collider).value.Duplicate();
  }

  void ResetColliderPose() {
    GetRegistry().get<CArticulatedReducedPose<TimeStep::Current>>(_collider).value = _colliderPose;
    UpdateColliderLinks();
  }

  void AddToColliderPose(int localDof, real eps) {
    AddToArticulatedPose(GetRegistry(), _collider, localDof, eps);
    UpdateColliderLinks();
  }
};

// A skinned articulated actor against a link of another articulated actor, in sync contact.
class MochiSkinnedArticulatedLinkContact : public MochiLinkColliderContactTestBase {
  ColumnVector<real> _skinPose;

 public:
  MochiSkinnedArticulatedLinkContact() {
    _numDofsA = 6;
    _numDofsB = kNumColliderDofs;
    _numDofs = _numDofsA + _numDofsB;
  }

  void InitializeScene(real coulombCoefficient, real viscousCoefficient, real dampingCoefficient)
      override {
    Actor* const skinnedActor = CreateThreeBoneSkinnedActor(
        _mochiContext,
        _scene,
        coulombCoefficient,
        viscousCoefficient,
        dampingCoefficient,
        _collidingBoundaryElementType);
    Actor* const colliderActor =
        InitLinkCollider(coulombCoefficient, viscousCoefficient, dampingCoefficient);
    InitActors(skinnedActor, colliderActor);

    auto& reg = GetRegistry();
    UpdateArticulatedFullPose(reg, _colliding);
    SaveColliderPose();
    _skinPose = reg.get<CArticulatedReducedPose<TimeStep::Current>>(_colliding).value.Duplicate();
    island::PreStep(reg, /*updateSleepState*/ true);
  }

  void InitState() override {
    auto& reg = GetRegistry();
    reg.get<CArticulatedReducedPose<TimeStep::Current>>(_colliding).value = _skinPose;
    UpdateArticulatedFullPose(reg, _colliding);
    ResetColliderPose();
  }

  void AddToState(int globalDof, real eps) override {
    auto [entity, localDof] = GetEntityAndLocalDofFromGlobalDof(globalDof);
    if (entity == _colliding) {
      AddToArticulatedPose(GetRegistry(), _colliding, localDof, eps);
    } else {
      AddToColliderPose(localDof, eps);
    }
  }
};

TEST_P(MochiSkinnedArticulatedLinkContact, ContactAssembly) {
  RunTest();

  // The skin must be assembled from contact sums against the link.
  bool hasSumsContact = false;
  for (auto const& jac : GetRegistry().get<CCollJacs<CollRole::Colliding> const>(_colliding)) {
    hasSumsContact |= jac.query && !jac.query->sampleIndices.empty() && jac.useContactSums;
  }
  EXPECT_TRUE(hasSumsContact);
}

INSTANTIATE_TEST_SUITE_P(
    MultipleBones,
    MochiSkinnedArticulatedLinkContact,
    ::testing::Values(
        TestParams{"NoFriction", GradTarget::Current, 0_r, 0_r, 0_r, false, 2e-3_r, 2.6e-1_r}),
    [](::testing::TestParamInfo<TestParams> const& info) { return info.param.name; });

// Sync contact sums of the skin and the link slice, split across tasks, so that both sides of the
// pair write link-space residuals.
class MochiParallelSkinnedArticulatedLinkContact : public MochiSkinnedArticulatedLinkContact {
 public:
  MochiParallelSkinnedArticulatedLinkContact() {
    UseParallelDenseContact();
  }
};

TEST_P(MochiParallelSkinnedArticulatedLinkContact, MatchesSerialAssembly) {
  RunSerialVsParallelTest();
}

INSTANTIATE_TEST_SUITE_P(
    MultipleBones,
    MochiParallelSkinnedArticulatedLinkContact,
    ::testing::Values(TestParams{.name = "NoFriction"}),
    [](::testing::TestParamInfo<TestParams> const& info) { return info.param.name; });

// The contact of a soft-skinned actor with zero soft displacement matches the contact of an
// articulated skin with the same mesh and skinning: their bone slices are the same, and the terms
// of the soft slice involve only the soft DoFs. The articulated skin is assembled from contact sums
// and the soft-skinned actor with the generic assembly, so their objective, and the residual and
// dresidual of the skeleton and collider DoFs must match. A dynamic box is in sync contact, and a
// static box in async contact.
class MochiSkinnedContactEquivalence : public test::MochiSceneTestBase {
 protected:
  struct SkinnedScene {
    IslandAssembly assembly;
    DynamicArray<int> dofs; // Skeleton DoFs, then collider DoFs if dynamic.
    int numContacts = 0;
    bool hasAsyncContacts = false;
    bool usesContactSums = false;
    // Whether contact sums are assembled for partitions with 1 bone slice (direct scatters) and
    // with 2+ bone slices (grouped into a single scatter).
    bool hasSingleBoneSums = false;
    bool hasMultiBoneSums = false;
  };

  // Create a three-bone skeleton with a skin, either articulated or soft-skinned, against a rigid
  // box, and assemble the island.
  SkinnedScene AssembleScene(bool isSoftSkinned, bool isBoxStatic) {
    _mochiContext->DestroyScene(_scene);
    _scene = _mochiContext->CreateScene("MochiPhysicsTest");
    auto& reg = GetRegistry();
    ArticulatedActorParams skeletonParams = MakeThreeBoneSkeletonParams(_mochiContext);
    Actor* articulatedActor = nullptr;
    Actor* collidingActor = nullptr;
    if (isSoftSkinned) {
      SoftSkinnedActorParams params;
      params.skeletonParams = std::move(skeletonParams);
      SoftActorParams softParams;
      softParams.name = "soft";
      // A nested soft actor needs Dirichlet conditions. They remove soft DoFs only.
      softParams.shape = CreateThreeBoneShape(_mochiContext, DynamicArray<int>{0});
      softParams.layer = "Colliding";
      softParams.hasGravity = false;
      softParams.hasInertia = false;
      softParams.hasStress = true;
      softParams.material.neoHookean.youngsModulus = 1e-10_r;
      params.softParams.push_back(std::move(softParams));
      articulatedActor = _scene->CreateSoftSkinnedActor(params, test::ExpectOK{});
      collidingActor = _scene->GetActor(articulatedActor->GetNestedSoftActors(test::ExpectOK{})[0]);
    } else {
      ArticulatedSkinParams skinParams;
      skinParams.shape = CreateThreeBoneShape(_mochiContext);
      skinParams.layer = "Colliding";
      skeletonParams.skin = std::move(skinParams);
      articulatedActor = _scene->CreateArticulatedActor(skeletonParams, test::ExpectOK{});
      collidingActor = articulatedActor;
    }

    RigidActorParams boxParams;
    boxParams.density = 1e-10_r;
    boxParams.layer = "Collider";
    boxParams.shape = _mochiContext->LoadShapeFromFile(
        test::GetAssetPath("cube/cube_mesh.mochi.json"), test::ExpectOK{});
    boxParams.worldFromLocal = TransformRT{
        Quaternion::FromRotationVector(Real3{0.1_r, -0.1_r, 0.2_r}), Real3{0.6_r, 0.6_r, 0.6_r}};
    boxParams.contact.penaltyThresholdDefault = 5e-3_r;
    boxParams.colliderType = ColliderType::Box;
    boxParams.isStatic = isBoxStatic;
    Actor* const box = _scene->CreateRigidActor(boxParams, test::ExpectOK{});

    _scene->EnableLayerContactSymmetric("Colliding", "Colliding", false, test::ExpectOK{});
    _scene->EnableLayerContactSymmetric("Collider", "Collider", false, test::ExpectOK{});
    _scene->EnableLayerContactSymmetric("Colliding", "Collider", true, test::ExpectOK{});

    entt::entity const articulated = GetEntity(articulatedActor);
    entt::entity const colliding = GetEntity(collidingActor);
    entt::entity const boxEntity = GetEntity(box);
    for (entt::entity const entity : {colliding, boxEntity}) {
      if (auto* const intState = reg.try_get<CTimeIntegratorState>(entity)) {
        intState->dtStage = 1e-2_r;
      }
    }
    reg.ctx<CSceneTime>().Advance(0.01);
    PreStepEcs(reg);
    UpdateArticulatedFullPose(reg, articulated);
    island::PreStep(reg, /*updateSleepState*/ true);

    SkinnedScene scene;
    scene.assembly = AssembleSingleIsland(reg, GradTarget::Current);
    DynamicArray<entt::entity> dofActors{articulated};
    if (!isBoxStatic) {
      dofActors.push_back(boxEntity);
    }
    for (entt::entity const actor : dofActors) {
      int const offset = reg.get<CDofOffset const>(actor).dofsOffset;
      for (int dof = 0; dof < reg.get<CActorDofInfo const>(actor).dofsSize; ++dof) {
        scene.dofs.push_back(offset + dof);
      }
    }
    for (auto const& coll :
         reg.get<CActiveCollisions</*kIsSync*/ true, TimeStep::Current> const>(colliding)) {
      scene.numContacts += isize(coll.collisionResult.sampleIndices);
    }
    for (auto const& coll :
         reg.get<CActiveCollisions</*kIsSync*/ false, TimeStep::Current> const>(colliding)) {
      scene.numContacts += isize(coll.collisionResult.sampleIndices);
      scene.hasAsyncContacts |= !coll.collisionResult.sampleIndices.empty();
    }
    for (auto const& jac : reg.get<CCollJacs<CollRole::Colliding> const>(colliding)) {
      if (jac.query && jac.useContactSums) {
        scene.usesContactSums = true;
        int numBoneSlices = 0;
        jac.ForEachActiveJac([&](ContactJac const&) { ++numBoneSlices; });
        scene.hasSingleBoneSums |= numBoneSlices == 1;
        scene.hasMultiBoneSums |= numBoneSlices >= 2;
      }
    }
    return scene;
  }
};

TEST_F(MochiSkinnedContactEquivalence, SoftSkinnedMatchesArticulatedSkin) {
  for (bool const isBoxStatic : {false, true}) {
    SCOPED_TRACE(isBoxStatic ? "static box" : "dynamic box");
    SkinnedScene const articulated = AssembleScene(/*isSoftSkinned*/ false, isBoxStatic);
    SkinnedScene const softSkinned = AssembleScene(/*isSoftSkinned*/ true, isBoxStatic);

    EXPECT_TRUE(articulated.usesContactSums);
    EXPECT_TRUE(articulated.hasSingleBoneSums);
    EXPECT_TRUE(articulated.hasMultiBoneSums);
    EXPECT_FALSE(softSkinned.usesContactSums);
    EXPECT_EQ(isBoxStatic, articulated.hasAsyncContacts);
    EXPECT_GT(articulated.numContacts, 0);
    EXPECT_EQ(articulated.numContacts, softSkinned.numContacts);
    ASSERT_EQ(isize(articulated.dofs), isize(softSkinned.dofs));

    // The paths sum in different orders, so they match up to rounding.
    real constexpr kTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-10_r : 1e-4_r;
    EXPECT_NEAR(
        articulated.assembly.objective,
        softSkinned.assembly.objective,
        kTol * Max(1.0, Abs(articulated.assembly.objective)));
    int const numDofs = isize(articulated.dofs);
    real resScale = 1_r;
    real dresScale = 1_r;
    for (int i = 0; i < numDofs; ++i) {
      resScale = Max(resScale, Abs(articulated.assembly.residual[articulated.dofs[i]]));
      for (int j = 0; j < numDofs; ++j) {
        dresScale =
            Max(dresScale,
                Abs(articulated.assembly.dresidual(articulated.dofs[i], articulated.dofs[j])));
      }
    }
    for (int i = 0; i < numDofs; ++i) {
      EXPECT_NEAR(
          articulated.assembly.residual[articulated.dofs[i]],
          softSkinned.assembly.residual[softSkinned.dofs[i]],
          kTol * resScale)
          << "dof " << i;
      for (int j = 0; j < numDofs; ++j) {
        EXPECT_NEAR(
            articulated.assembly.dresidual(articulated.dofs[i], articulated.dofs[j]),
            softSkinned.assembly.dresidual(softSkinned.dofs[i], softSkinned.dofs[j]),
            kTol * dresScale)
            << "dofs " << i << ", " << j;
      }
    }
  }
}

class MochiSoftSkinnedStaticContact : public MochiContactTestBase {
 protected:
  static constexpr int kSkeletonDofs = 6;
  static constexpr int kFomSoftDofs = 24;
  static constexpr int kRomSoftDofs = 1;

  // PartiallyBlendedFom blends only the nodes with x < 0.5, so that the partitions of the samples
  // on the face x = 1 have no soft actor. PosedFom assembles posed gravity and stress for the
  // nested soft actor, through skinned::EntityAssembleBody.
  using Mode = ThreeBoneSoftSkinnedMode;

  entt::entity _articulated = entt::null;
  ColumnVector<real> _pose;
  ColumnVector<real> _softState;
  Mode _mode;

 public:
  explicit MochiSoftSkinnedStaticContact(Mode mode = Mode::Fom) : _mode(mode) {
    int const softDofs = UsesRom() ? kRomSoftDofs : kFomSoftDofs;
    _numDofsA = UsesBlendedSurface() ? kSkeletonDofs : softDofs;
    _numDofsB = 0;
    _numDofs = kSkeletonDofs + softDofs;
  }

  void InitializeScene(real coulombCoefficient, real viscousCoefficient, real dampingCoefficient)
      override {
    TransformRT const worldFromRoot = UsesBlendedSurface()
        ? TransformRT{Quaternion::FromRotationVector(Real3{0.2_r, -0.3_r, 0.1_r}), Real3{0.4_r, -0.2_r, 0.3_r}}
        : TransformRT::Identity();
    Actor* const articulatedActor = CreateThreeBoneSoftSkinnedActor(
        _mochiContext,
        _scene,
        _mode,
        worldFromRoot,
        coulombCoefficient,
        viscousCoefficient,
        dampingCoefficient,
        _collidingBoundaryElementType);
    _articulated = GetEntity(articulatedActor->GetHandle());
    auto const& softHandles = articulatedActor->GetNestedSoftActors(test::ExpectOK{});
    ASSERT_EQ(1, isize(softHandles));
    Actor* const softActor = _scene->GetActor(softHandles[0]);
    ASSERT_NE(nullptr, softActor);
    _soft = GetEntity(softActor->GetHandle());

    TransformRT const colliderTransformLocal{
        Quaternion::FromRotationVector(Real3{0.1_r, -0.1_r, 0.2_r}), Real3{0.6_r, 0.6_r, 0.6_r}};
    Actor* const collider = InitStaticCube(
        1_r,
        worldFromRoot * colliderTransformLocal,
        coulombCoefficient,
        viscousCoefficient,
        dampingCoefficient);
    InitActors(UsesBlendedSurface() ? articulatedActor : softActor, collider);

    auto& reg = GetRegistry();
    _pose = reg.get<CArticulatedReducedPose<TimeStep::Current>>(_articulated).value.Duplicate();
    if (UsesBlendedSurface()) {
      if (UsesRom()) {
        reg.get<CRomModeAmplitudes>(_soft).value[0] = 0.1_r;
        UpdateRomState();
      } else {
        auto& displacement = reg.get<CDisplacementSlice<real, TimeStep::Current>>(_soft).value;
        for (int node = 1; node < kFomSoftDofs / 3; ++node) {
          displacement[3 * node] = 0.1_r;
        }
      }
    }
    _softState = UsesRom()
        ? reg.get<CRomModeAmplitudes const>(_soft).value.Duplicate()
        : reg.get<CDisplacementSlice<real, TimeStep::Current> const>(_soft).value.Duplicate();
  }

  std::pair<entt::entity, int> GetEntityAndLocalDofFromGlobalDof(int globalDof) override {
    auto const& reg = GetRegistry();
    int const articulatedOffset = reg.get<CDofOffset const>(_articulated).dofsOffset;
    if (globalDof >= articulatedOffset && globalDof < articulatedOffset + kSkeletonDofs) {
      return {_articulated, globalDof - articulatedOffset};
    }
    int const softOffset = reg.get<CDofOffset const>(_soft).dofsOffset;
    EXPECT_GE(globalDof, softOffset);
    int const softDofs = UsesRom() ? kRomSoftDofs : kFomSoftDofs;
    EXPECT_LT(globalDof, softOffset + softDofs);
    return {_soft, globalDof - softOffset};
  }

  void InitState() override {
    auto& reg = GetRegistry();
    reg.get<CArticulatedReducedPose<TimeStep::Current>>(_articulated).value = _pose;
    if (UsesRom()) {
      reg.get<CRomModeAmplitudes>(_soft).value = _softState;
      UpdateRomState();
    } else {
      reg.get<CDisplacementSlice<real, TimeStep::Current>>(_soft).value = _softState;
    }
    UpdateArticulatedFullPose(reg, _articulated);
  }

  void AddToState(int globalDof, real eps) override {
    auto const [entity, localDof] = GetEntityAndLocalDofFromGlobalDof(globalDof);
    auto& reg = GetRegistry();
    if (entity == _soft) {
      if (UsesRom()) {
        reg.get<CRomModeAmplitudes>(_soft).value[localDof] += eps;
        UpdateRomState();
      } else {
        reg.get<CDisplacementSlice<real, TimeStep::Current>>(_soft).value[localDof] += eps;
      }
      return;
    }
    AddToArticulatedPose(reg, _articulated, localDof, eps);
  }

 private:
  bool UsesRom() const {
    return test::UsesRom(_mode);
  }

  bool UsesBlendedSurface() const {
    return test::UsesBlendedSurface(_mode);
  }

  void UpdateRomState() {
    auto& reg = GetRegistry();
    ecs::TryInvokeOnEntity(
        &rom::linear::ResolveDisplacementAndJacobian</*kForceUseAllNodes = */ true>, reg, _soft);
    ecs::TryInvokeOnEntity(&rom::UpdateCurrentRomVelocity, reg, _soft);
  }

  entt::entity _soft = entt::null;
};

TEST_P(MochiSoftSkinnedStaticContact, ContactAssembly) {
  RunTest();
}

INSTANTIATE_TEST_SUITE_P(
    MultipleBones,
    MochiSoftSkinnedStaticContact,
    ::testing::Values(
        TestParams{"NoFriction", GradTarget::Current, 0_r, 0_r, 0_r, false, 3e-3_r, 3e-1_r}),
    [](::testing::TestParamInfo<TestParams> const& info) { return info.param.name; });

class MochiPosedSoftSkinnedStaticContact : public MochiSoftSkinnedStaticContact {
 public:
  MochiPosedSoftSkinnedStaticContact() : MochiSoftSkinnedStaticContact(Mode::PosedFom) {}
};

TEST_P(MochiPosedSoftSkinnedStaticContact, ContactAssembly) {
  RunTest();
}

INSTANTIATE_TEST_SUITE_P(
    MultipleBones,
    MochiPosedSoftSkinnedStaticContact,
    ::testing::Values(
        TestParams{"NoFriction", GradTarget::Current, 0_r, 0_r, 0_r, false, 3e-3_r, 3e-1_r}),
    [](::testing::TestParamInfo<TestParams> const& info) { return info.param.name; });

class MochiRomSoftSkinnedStaticContact : public MochiSoftSkinnedStaticContact {
 public:
  MochiRomSoftSkinnedStaticContact() : MochiSoftSkinnedStaticContact(Mode::Rom) {}
};

TEST_IF_P(MOCHI_ENABLE_ROM_ACTORS, MochiRomSoftSkinnedStaticContact, ContactAssembly) {
  RunTest();
}

INSTANTIATE_TEST_SUITE_P(
    MultipleBones,
    MochiRomSoftSkinnedStaticContact,
    ::testing::Values(
        TestParams{"NoFriction", GradTarget::Current, 0_r, 0_r, 0_r, false, 3e-3_r, 3e-1_r}),
    [](::testing::TestParamInfo<TestParams> const& info) { return info.param.name; });

class MochiBlendedStaticContact : public MochiSoftSkinnedStaticContact {
 public:
  MochiBlendedStaticContact() : MochiSoftSkinnedStaticContact(Mode::BlendedFom) {}
};

TEST_P(MochiBlendedStaticContact, ContactAssembly) {
  RunTest();
}

INSTANTIATE_TEST_SUITE_P(
    MultipleBones,
    MochiBlendedStaticContact,
    ::testing::Values(
        TestParams{"NoFriction", GradTarget::Current, 0_r, 0_r, 0_r, false, 3e-3_r, 3e-1_r}),
    [](::testing::TestParamInfo<TestParams> const& info) { return info.param.name; });

class MochiRomBlendedStaticContact : public MochiSoftSkinnedStaticContact {
 public:
  MochiRomBlendedStaticContact() : MochiSoftSkinnedStaticContact(Mode::BlendedRom) {}
};

TEST_IF_P(MOCHI_ENABLE_ROM_ACTORS, MochiRomBlendedStaticContact, ContactAssembly) {
  RunTest();
}

INSTANTIATE_TEST_SUITE_P(
    MultipleBones,
    MochiRomBlendedStaticContact,
    ::testing::Values(
        TestParams{"NoFriction", GradTarget::Current, 0_r, 0_r, 0_r, false, 3e-3_r, 3e-1_r}),
    [](::testing::TestParamInfo<TestParams> const& info) { return info.param.name; });

// A blended skin whose partitions without a soft actor are assembled from contact sums, while the
// partitions blended with the soft actor use the generic assembly.
class MochiPartiallyBlendedStaticContact : public MochiSoftSkinnedStaticContact {
 public:
  MochiPartiallyBlendedStaticContact() : MochiSoftSkinnedStaticContact(Mode::PartiallyBlendedFom) {}

  void ExpectSumsAndGenericPartitionsInContact() const {
    int numSumsJacs = 0;
    int numGenericJacs = 0;
    for (auto const& jac : GetRegistry().get<CCollJacs<CollRole::Colliding> const>(_colliding)) {
      if (jac.query && !jac.query->sampleIndices.empty()) {
        ++(jac.useContactSums ? numSumsJacs : numGenericJacs);
      }
    }
    EXPECT_GT(numSumsJacs, 0);
    EXPECT_GT(numGenericJacs, 0);
  }
};

TEST_P(MochiPartiallyBlendedStaticContact, ContactAssembly) {
  RunTest();
  ExpectSumsAndGenericPartitionsInContact();
}

INSTANTIATE_TEST_SUITE_P(
    MultipleBones,
    MochiPartiallyBlendedStaticContact,
    ::testing::Values(
        TestParams{"NoFriction", GradTarget::Current, 0_r, 0_r, 0_r, false, 3e-3_r, 3e-1_r}),
    [](::testing::TestParamInfo<TestParams> const& info) { return info.param.name; });

// Async contact sums and generic assembly with grouped shared DoFs, split across tasks.
class MochiParallelPartiallyBlendedStaticContact : public MochiPartiallyBlendedStaticContact {
 public:
  MochiParallelPartiallyBlendedStaticContact() {
    UseParallelDenseContact();
  }
};

TEST_P(MochiParallelPartiallyBlendedStaticContact, MatchesSerialAssembly) {
  RunSerialVsParallelTest();
  ExpectSumsAndGenericPartitionsInContact();
}

INSTANTIATE_TEST_SUITE_P(
    MultipleBones,
    MochiParallelPartiallyBlendedStaticContact,
    ::testing::Values(TestParams{.name = "NoFriction"}),
    [](::testing::TestParamInfo<TestParams> const& info) { return info.param.name; });

class MochiSoftStaticContact : public MochiContactTestBase {
  ColumnVector<real> _posA;

 public:
  MochiSoftStaticContact() {
    _numDofsA = 24; // a soft with 8 nodes
    _numDofsB = 0; // Static
    _numDofs = _numDofsA + _numDofsB;
  }

  void InitializeScene(real coulombCoefficient, real viscousCoefficient, real dampingCoefficient)
      override {
    auto& reg = GetRegistry();

    auto* actorA =
        InitSoftCube({}, false, {}, coulombCoefficient, viscousCoefficient, dampingCoefficient);
    auto transform = TransformRT(
        Quaternion::FromRotationVector(Real3(0.1_r, -0.1_r, 0.2_r)), Real3(0.6_r, 0.6_r, 0.6_r));
    auto* actorB =
        InitStaticCube(1_r, transform, coulombCoefficient, viscousCoefficient, dampingCoefficient);
    InitActors(actorA, actorB);

    _posA =
        reg.get<CDisplacementSlice<real, TimeStep::Current> const>(_colliding).value.Duplicate();

    // Add velocity to the actor
    auto& posAold = reg.get<CDisplacementSlice<real, TimeStep::StageStart>>(_colliding);
    for (int i = 0; i < _numDofsA; i++) {
      posAold.value[i] = (i % 3 == 0) ? -0.01_r : 0.01_r;
    }

    InitSoftCubeStageStart(_colliding);
  }

  void InitState() override {
    auto& reg = GetRegistry();
    reg.get<CDisplacementSlice<real, TimeStep::Current>>(_colliding).value = _posA;
  }

  void AddToState(int globalDof, real eps) override {
    auto [entity, localDof] = GetEntityAndLocalDofFromGlobalDof(globalDof);
    auto& reg = GetRegistry();
    MOCHI_ASSERT_VERBOSE(entity == _colliding);

    // Change state of a node of object A
    auto& posA = reg.get<CDisplacementSlice<real, TimeStep::Current>>(_colliding);
    posA.value[localDof] += eps;
  }
};

TEST_P(MochiSoftStaticContact, ContactAssembly) {
  RunTest();
}

INSTANTIATE_TEST_SUITE_P(
    FrictionVariations,
    MochiSoftStaticContact,
    ::testing::Values(
        TestParams{"NoFriction", GradTarget::Current, 0_r, 0_r, 0_r, false, 1e-3_r, 1e-3_r},
        TestParams{"Viscous", GradTarget::Current, 0_r, 1_r, 0_r, false, 1e-3_r, 1e-3_r},
        TestParams{"Coulomb", GradTarget::Current, 0.5_r, 0_r, 0_r, false, 1e-3_r, 1e-3_r},
        TestParams{"Damping", GradTarget::Current, 0_r, 0_r, 1_r, false, 1e-3_r, 1e-3_r},
        TestParams{"NoFrictionExplicit", GradTarget::Current, 0_r, 0_r, 0_r, true, 1e-3_r, 1e-3_r},
        TestParams{"ViscousExplicit", GradTarget::Current, 0_r, 1_r, 0_r, true, 1e-3_r, 1e-3_r},
        TestParams{"CoulombExplicit", GradTarget::Current, 0.5_r, 0_r, 0_r, true, 1e-3_r, 1e-3_r},
        TestParams{"DampingExplicit", GradTarget::Current, 0_r, 0_r, 1_r, true, 1e-3_r, 1e-3_r}),
    [](::testing::TestParamInfo<TestParams> const& info) { return info.param.name; });

// The static cube was moved by a prescribed transform, so its stage-start pose differs from its
// current pose. Explicit normals take the friction plane from the stage-start pose, but the
// response is the gradient wrt the current contact points, so it must be mapped back through the
// current collider frame.
class MochiSoftMovingStaticContact : public MochiSoftStaticContact {
 public:
  void InitializeScene(real coulombCoefficient, real viscousCoefficient, real dampingCoefficient)
      override {
    MochiSoftStaticContact::InitializeScene(
        coulombCoefficient, viscousCoefficient, dampingCoefficient);
    auto& rootTransform = GetRegistry().get<CRootTransform>(_collider);
    rootTransform.worldFromLocalPrev = TransformRT(
        Quaternion::FromRotationVector(Real3(0.05_r, -0.03_r, 0.04_r)) *
            rootTransform.worldFromLocal.GetRotation(),
        rootTransform.worldFromLocal.GetTranslation());
  }
};

TEST_P(MochiSoftMovingStaticContact, ContactAssembly) {
  RunTest();
}

INSTANTIATE_TEST_SUITE_P(
    FrictionVariations,
    MochiSoftMovingStaticContact,
    ::testing::Values(
        TestParams{"NoFriction", GradTarget::Current, 0_r, 0_r, 0_r, false, 1e-3_r, 1e-3_r},
        TestParams{"Viscous", GradTarget::Current, 0_r, 1_r, 0_r, false, 1e-3_r, 1e-3_r},
        TestParams{"NoFrictionExplicit", GradTarget::Current, 0_r, 0_r, 0_r, true, 1e-3_r, 1e-3_r},
        TestParams{"ViscousExplicit", GradTarget::Current, 0_r, 1_r, 0_r, true, 1e-3_r, 1e-3_r}),
    [](::testing::TestParamInfo<TestParams> const& info) { return info.param.name; });

// The soft cube contacts the static cube through a linear contact skin, which takes the async
// mapped assembly.
class MochiSkinnedSoftStaticContact : public MochiSoftStaticContact {
 public:
  MochiSkinnedSoftStaticContact() {
    _useContactSkin = true;
  }
};

TEST_P(MochiSkinnedSoftStaticContact, ContactAssembly) {
  RunTest();
  ExpectMappedContactSkinCollisions</*kIsSync*/ false>();
}

TEST_P(MochiSkinnedSoftStaticContact, MappedMatchesLegacyAssembly) {
  RunMappedVsLegacyContactSkinTest</*kIsSync*/ false>();
}

// Finite differences of the dresidual through the contact skin have float rounding error near 1e-3
// that varies across build configurations.
INSTANTIATE_TEST_SUITE_P(
    FrictionVariations,
    MochiSkinnedSoftStaticContact,
    ::testing::Values(
        TestParams{"NoFriction", GradTarget::Current, 0_r, 0_r, 0_r, false, 1e-3_r, 2e-3_r},
        TestParams{"Viscous", GradTarget::Current, 0_r, 1_r, 0_r, false, 1e-3_r, 2e-3_r},
        TestParams{"Coulomb", GradTarget::Current, 0.5_r, 0_r, 0_r, false, 1e-3_r, 2e-3_r},
        TestParams{"Damping", GradTarget::Current, 0_r, 0_r, 1_r, false, 1e-3_r, 2e-3_r},
        TestParams{"NoFrictionExplicit", GradTarget::Current, 0_r, 0_r, 0_r, true, 1e-3_r, 2e-3_r},
        TestParams{"ViscousExplicit", GradTarget::Current, 0_r, 1_r, 0_r, true, 1e-3_r, 2e-3_r},
        TestParams{"CoulombExplicit", GradTarget::Current, 0.5_r, 0_r, 0_r, true, 1e-3_r, 2e-3_r},
        TestParams{"DampingExplicit", GradTarget::Current, 0_r, 0_r, 1_r, true, 1e-3_r, 2e-3_r}),
    [](::testing::TestParamInfo<TestParams> const& info) { return info.param.name; });

// Async contact of the soft contact skin, split across tasks.
class MochiParallelSkinnedSoftStaticContact : public MochiSkinnedSoftStaticContact {
 public:
  MochiParallelSkinnedSoftStaticContact() {
    UseParallelDenseContact();
  }
};

TEST_P(MochiParallelSkinnedSoftStaticContact, MappedMatchesSerialAssembly) {
  RunSerialVsParallelTest();
  ExpectMappedContactSkinCollisions</*kIsSync*/ false>();
}

TEST_P(MochiParallelSkinnedSoftStaticContact, LegacyMatchesSerialAssembly) {
  RunSerialVsParallelTest(/*removeLinearContactSkinMap*/ true);
}

INSTANTIATE_TEST_SUITE_P(
    FrictionVariations,
    MochiParallelSkinnedSoftStaticContact,
    ::testing::Values(TestParams{.name = "Coulomb", .coulombCoefficient = 0.5_r}),
    [](::testing::TestParamInfo<TestParams> const& info) { return info.param.name; });

// A skinned soft cube against the moving static cube takes the async mapped assembly, which must
// also map the response back through the current collider frame.
class MochiSkinnedSoftMovingStaticContact : public MochiSoftMovingStaticContact {
 public:
  MochiSkinnedSoftMovingStaticContact() {
    _useContactSkin = true;
  }
};

TEST_P(MochiSkinnedSoftMovingStaticContact, ContactAssembly) {
  RunTest();
  ExpectMappedContactSkinCollisions</*kIsSync*/ false>();
}

// Only explicit normals read the stage-start collider frame.
INSTANTIATE_TEST_SUITE_P(
    FrictionVariations,
    MochiSkinnedSoftMovingStaticContact,
    ::testing::Values(
        TestParams{"NoFrictionExplicit", GradTarget::Current, 0_r, 0_r, 0_r, true, 1e-3_r, 2e-3_r},
        TestParams{"ViscousExplicit", GradTarget::Current, 0_r, 1_r, 0_r, true, 1e-3_r, 2e-3_r}),
    [](::testing::TestParamInfo<TestParams> const& info) { return info.param.name; });

class MochiSoftRigidContact : public MochiContactTestBase {
  ColumnVector<real> _posA;
  TransformRT _posB;

 protected:
  real _colliderScale = 1_r;
  TransformRT _colliderTransform{
      Quaternion::FromRotationVector(Real3(0.1_r, -0.1_r, 0.2_r)),
      Real3(0.6_r, 0.6_r, 0.6_r)};

 public:
  MochiSoftRigidContact() {
    _numDofsA = 24; // a soft with 8 nodes
    _numDofsB = 6; // Rigid
    _numDofs = _numDofsA + _numDofsB;
  }

  void InitializeScene(real coulombCoefficient, real viscousCoefficient, real dampingCoefficient)
      override {
    auto& reg = GetRegistry();

    // Create the rigid cube first so that the soft cube's DoFs are offset.
    auto* actorB = InitRigidCube(
        _colliderScale,
        _colliderTransform,
        true,
        coulombCoefficient,
        viscousCoefficient,
        dampingCoefficient);
    auto* actorA =
        InitSoftCube({}, false, {}, coulombCoefficient, viscousCoefficient, dampingCoefficient);
    InitActors(actorA, actorB);

    _posA =
        reg.get<CDisplacementSlice<real, TimeStep::Current> const>(_colliding).value.Duplicate();

    // Add velocity to the actors
    auto& posAold = reg.get<CDisplacementSlice<real, TimeStep::StageStart>>(_colliding);
    for (int i = 0; i < _numDofsA; i++) {
      posAold.value[i] = (i % 3 == 0) ? -0.01_r : 0.01_r;
    }
    _posB = reg.get<CRigidState<TimeStep::Current>>(_collider).value;
    auto& posBold = reg.get<CRigidState<TimeStep::StageStart>>(_collider).value;
    posBold.SetTranslation(_posB.GetTranslation() + Real3(0.02_r, 0.01_r, -0.01_r));
    posBold.SetRotation(
        Quaternion::FromRotationVector(Vec4r(-0.1_r, -0.1_r, -0.2_r)) * _posB.GetRotation());

    InitSoftCubeStageStart(_colliding);
  }

  void InitState() override {
    auto& reg = GetRegistry();
    reg.get<CDisplacementSlice<real, TimeStep::Current>>(_colliding).value = _posA;
    reg.get<CRigidState<TimeStep::Current>>(_collider).value = _posB;
    SetTransformFromState(_collider, _posB);
  }

  void AddToState(int globalDof, real eps) override {
    auto [entity, localDof] = GetEntityAndLocalDofFromGlobalDof(globalDof);
    auto& reg = GetRegistry();
    if (entity == _colliding) {
      // Change state of a node of object A
      auto& posA = reg.get<CDisplacementSlice<real, TimeStep::Current>>(_colliding);
      posA.value[localDof] += eps;
    } else {
      Vec4r delta = SimdBasisVector(localDof % 3) * eps;
      auto& rigidState = reg.get<CRigidState<TimeStep::Current>>(_collider).value;
      if (localDof < 3) {
        // Change state of the rigid translation
        rigidState.SetTranslation(rigidState.GetTranslation() + ToReal3(delta));
      } else {
        // Change state of the rigid rotation
        rigidState.SetRotation(Quaternion::FromRotationVector(delta) * rigidState.GetRotation());
      }
      SetTransformFromState(_collider, rigidState);
    }
  }
};

TEST_P(MochiSoftRigidContact, ContactAssembly) {
  RunTest();
}

INSTANTIATE_TEST_SUITE_P(
    FrictionVariations,
    MochiSoftRigidContact,
    ::testing::Values(
        TestParams{"NoFriction", GradTarget::Current, 0_r, 0_r, 0_r, false, 1e-3_r, 1e-3_r, true},
        TestParams{"Viscous", GradTarget::Current, 0_r, 1_r, 0_r, false, 1e-3_r, 1e-3_r, true},
        TestParams{"Coulomb", GradTarget::Current, 0.5_r, 0_r, 0_r, false, 3e-3_r, 1e-3_r, true},
        TestParams{
            "NoFrictionExplicit",
            GradTarget::Current,
            0_r,
            0_r,
            0_r,
            true,
            1e-3_r,
            1e-3_r,
            true},
        TestParams{
            "ViscousExplicit",
            GradTarget::Current,
            0_r,
            1_r,
            0_r,
            true,
            1e-3_r,
            1e-3_r,
            true},
        TestParams{
            "CoulombExplicit",
            GradTarget::Current,
            0.5_r,
            0_r,
            0_r,
            true,
            1e-3_r,
            1e-3_r,
            true},
        TestParams{"Damping", GradTarget::Current, 0_r, 0_r, 1_r, false, 1e-3_r, 1e-3_r, true},
        TestParams{
            "DampingExplicit",
            GradTarget::Current,
            0_r,
            0_r,
            1_r,
            true,
            1e-3_r,
            1e-3_r,
            true}),
    [](::testing::TestParamInfo<TestParams> const& info) { return info.param.name; });

// The soft cube contacts the dynamic rigid cube through a linear contact skin, which takes the sync
// mapped assembly with skin-collider coupling blocks.
class MochiSkinnedSoftRigidContact : public MochiSoftRigidContact {
 public:
  MochiSkinnedSoftRigidContact() {
    _useContactSkin = true;
  }
};

TEST_P(MochiSkinnedSoftRigidContact, ContactAssembly) {
  RunTest();
  ExpectMappedContactSkinCollisions</*kIsSync*/ true>();
}

TEST_P(MochiSkinnedSoftRigidContact, MappedMatchesLegacyAssembly) {
  RunMappedVsLegacyContactSkinTest</*kIsSync*/ true>();
}

INSTANTIATE_TEST_SUITE_P(
    FrictionVariations,
    MochiSkinnedSoftRigidContact,
    ::testing::Values(
        TestParams{"NoFriction", GradTarget::Current, 0_r, 0_r, 0_r, false, 1e-3_r, 1e-3_r, true},
        TestParams{"Viscous", GradTarget::Current, 0_r, 1_r, 0_r, false, 1e-3_r, 1e-3_r, true},
        TestParams{"Coulomb", GradTarget::Current, 0.5_r, 0_r, 0_r, false, 3e-3_r, 1e-3_r, true},
        TestParams{
            "NoFrictionExplicit",
            GradTarget::Current,
            0_r,
            0_r,
            0_r,
            true,
            1e-3_r,
            1e-3_r,
            true},
        TestParams{
            "ViscousExplicit",
            GradTarget::Current,
            0_r,
            1_r,
            0_r,
            true,
            1e-3_r,
            1e-3_r,
            true},
        TestParams{
            "CoulombExplicit",
            GradTarget::Current,
            0.5_r,
            0_r,
            0_r,
            true,
            1e-3_r,
            1e-3_r,
            true},
        TestParams{"Damping", GradTarget::Current, 0_r, 0_r, 1_r, false, 1e-3_r, 1e-3_r, true},
        TestParams{
            "DampingExplicit",
            GradTarget::Current,
            0_r,
            0_r,
            1_r,
            true,
            1e-3_r,
            1e-3_r,
            true}),
    [](::testing::TestParamInfo<TestParams> const& info) { return info.param.name; });

// Sync contact of the soft contact skin, split across tasks.
class MochiParallelSkinnedSoftRigidContact : public MochiSkinnedSoftRigidContact {
 public:
  MochiParallelSkinnedSoftRigidContact() {
    UseParallelDenseContact();
  }
};

TEST_P(MochiParallelSkinnedSoftRigidContact, MappedMatchesSerialAssembly) {
  RunSerialVsParallelTest();
  ExpectMappedContactSkinCollisions</*kIsSync*/ true>();
}

INSTANTIATE_TEST_SUITE_P(
    FrictionVariations,
    MochiParallelSkinnedSoftRigidContact,
    ::testing::Values(TestParams{.name = "Coulomb", .coulombCoefficient = 0.5_r}),
    [](::testing::TestParamInfo<TestParams> const& info) { return info.param.name; });

// An enlarged rigid cube under the soft cube overlaps it by less than the penalty threshold, so
// that all samples of its bottom face have a nonzero response and the coupling blocks of each
// element sum several contacts.
class MochiDenseSkinnedSoftRigidContact : public MochiSkinnedSoftRigidContact {
 public:
  MochiDenseSkinnedSoftRigidContact() {
    _colliderScale = kShallowColliderScale;
    _colliderTransform = ShallowColliderTransform();
  }
};

TEST_P(MochiDenseSkinnedSoftRigidContact, MappedMatchesLegacyAssembly) {
  RunMappedVsLegacyContactSkinTest</*kIsSync*/ true>();
}

INSTANTIATE_TEST_SUITE_P(
    FrictionVariations,
    MochiDenseSkinnedSoftRigidContact,
    ::testing::Values(
        TestParams{.name = "Coulomb", .coulombCoefficient = 0.5_r},
        TestParams{
            .name = "CoulombP1Q6",
            .coulombCoefficient = 0.5_r,
            .collidingBoundaryElementType = ActorBoundaryElementType::P1Q6}),
    [](::testing::TestParamInfo<TestParams> const& info) { return info.param.name; });

// The soft cube contacts a link of an articulated actor through a linear contact skin, in sync
// contact.
class MochiSkinnedSoftLinkContact : public MochiLinkColliderContactTestBase {
  ColumnVector<real> _softDispl;

 public:
  MochiSkinnedSoftLinkContact() {
    _numDofsA = 24; // A soft cube with 8 nodes.
    _numDofsB = kNumColliderDofs;
    _numDofs = _numDofsA + _numDofsB;
    _useContactSkin = true;
  }

  void InitializeScene(real coulombCoefficient, real viscousCoefficient, real dampingCoefficient)
      override {
    Actor* const softActor =
        InitSoftCube({}, false, {}, coulombCoefficient, viscousCoefficient, dampingCoefficient);
    Actor* const colliderActor =
        InitLinkCollider(coulombCoefficient, viscousCoefficient, dampingCoefficient);
    InitActors(softActor, colliderActor);

    auto& reg = GetRegistry();
    SaveColliderPose();
    _softDispl =
        reg.get<CDisplacementSlice<real, TimeStep::Current> const>(_colliding).value.Duplicate();
    InitSoftCubeStageStart(_colliding);
    island::PreStep(reg, /*updateSleepState*/ true);
  }

  void InitState() override {
    GetRegistry().get<CDisplacementSlice<real, TimeStep::Current>>(_colliding).value = _softDispl;
    ResetColliderPose();
  }

  void AddToState(int globalDof, real eps) override {
    auto [entity, localDof] = GetEntityAndLocalDofFromGlobalDof(globalDof);
    if (entity == _colliding) {
      GetRegistry().get<CDisplacementSlice<real, TimeStep::Current>>(_colliding).value[localDof] +=
          eps;
    } else {
      AddToColliderPose(localDof, eps);
    }
  }
};

TEST_P(MochiSkinnedSoftLinkContact, ContactAssembly) {
  RunTest();
  ExpectMappedContactSkinCollisions</*kIsSync*/ true>();
}

TEST_P(MochiSkinnedSoftLinkContact, MappedMatchesLegacyAssembly) {
  RunMappedVsLegacyContactSkinTest</*kIsSync*/ true>();
}

// The dresidual omits the second derivative of the map from the link DoFs to the contact points,
// whose lever arm depends on both the link pose and the skin samples, so it differs from the finite
// differences in every row and column of the link DoFs. Only the soft block, whose skin map is
// linear, is compared. Its finite differences have float rounding error near 1e-3 that varies
// across build configurations.
INSTANTIATE_TEST_SUITE_P(
    FrictionVariations,
    MochiSkinnedSoftLinkContact,
    ::testing::Values(
        TestParams{.name = "NoFriction", .dresTol = 2e-3_r, .ignoreColliderDres = true},
        TestParams{
            .name = "Viscous",
            .viscousCoefficient = 1_r,
            .dresTol = 2e-3_r,
            .ignoreColliderDres = true}),
    [](::testing::TestParamInfo<TestParams> const& info) { return info.param.name; });

// An enlarged child link under the soft cube overlaps it by less than the penalty threshold, so
// that all samples of its bottom face have a nonzero response and the coupling blocks of each
// element sum several contacts.
class MochiDenseSkinnedSoftLinkContact : public MochiSkinnedSoftLinkContact {
 public:
  MochiDenseSkinnedSoftLinkContact() {
    _childLinkScale = kShallowColliderScale;
    _childLinkTransform = ShallowColliderTransform();
  }
};

TEST_P(MochiDenseSkinnedSoftLinkContact, MappedMatchesLegacyAssembly) {
  RunMappedVsLegacyContactSkinTest</*kIsSync*/ true>();
}

INSTANTIATE_TEST_SUITE_P(
    FrictionVariations,
    MochiDenseSkinnedSoftLinkContact,
    ::testing::Values(
        TestParams{.name = "Coulomb", .coulombCoefficient = 0.5_r},
        TestParams{
            .name = "CoulombP1Q6",
            .coulombCoefficient = 0.5_r,
            .collidingBoundaryElementType = ActorBoundaryElementType::P1Q6}),
    [](::testing::TestParamInfo<TestParams> const& info) { return info.param.name; });

class MochiRigidSoftContact : public MochiContactTestBase {
  TransformRT _posA;
  ColumnVector<real> _posB;

 public:
  MochiRigidSoftContact() {
    _numDofsA = 6; // Rigid
    _numDofsB = 24; // a soft with 8 nodes
    _numDofs = _numDofsA + _numDofsB;
  }

  void InitializeScene(real coulombCoefficient, real viscousCoefficient, real dampingCoefficient)
      override {
    auto& reg = GetRegistry();

    // Rotate the whole scene, including the stage-start offsets below.
    Quaternion const sceneRotation = SceneRotation();
    TransformRT const transform = TransformRT(sceneRotation) *
        TransformRT(Quaternion::FromRotationVector(Real3(0.1_r, -0.1_r, 0.2_r)),
                    Real3(0.6_r, 0.6_r, 0.6_r));
    auto* actorA = InitRigidCube(
        1_r, transform, false, coulombCoefficient, viscousCoefficient, dampingCoefficient);
    auto* actorB = InitSoftCube(
        TransformRT(sceneRotation),
        true,
        {},
        coulombCoefficient,
        viscousCoefficient,
        dampingCoefficient);
    InitActors(actorA, actorB);

    // Add velocity to the actors
    _posA = reg.get<CRigidState<TimeStep::Current>>(_colliding).value;
    auto& posAold = reg.get<CRigidState<TimeStep::StageStart>>(_colliding).value;
    posAold.SetTranslation(_posA.GetTranslation() + sceneRotation * Real3(0.02_r, 0.01_r, -0.01_r));
    posAold.SetRotation(
        Quaternion::FromRotationVector(sceneRotation * Vec4r(-0.1_r, -0.1_r, -0.2_r)) *
        _posA.GetRotation());
    _posB = reg.get<CDisplacementSlice<real, TimeStep::Current> const>(_collider).value.Duplicate();
    auto& posBold = reg.get<CDisplacementSlice<real, TimeStep::StageStart>>(_collider);
    for (int i = 0; i < _numDofsB; i++) {
      posBold.value[i] = (i % 3 == 0) ? -0.01_r : 0.01_r;
    }

    InitSoftCubeStageStart(_collider);
  }

  void InitState() override {
    auto& reg = GetRegistry();
    reg.get<CRigidState<TimeStep::Current>>(_colliding).value = _posA;
    SetTransformFromState(_colliding, _posA);
    reg.get<CDisplacementSlice<real, TimeStep::Current>>(_collider).value = _posB;
  }

  void AddToState(int globalDof, real eps) override {
    auto [entity, localDof] = GetEntityAndLocalDofFromGlobalDof(globalDof);
    auto& reg = GetRegistry();
    if (entity == _colliding) {
      Vec4r delta = SimdBasisVector(localDof % 3) * eps;
      auto& rigidState = reg.get<CRigidState<TimeStep::Current>>(_colliding).value;
      if (localDof < 3) {
        // Change state of the rigid translation
        rigidState.SetTranslation(rigidState.GetTranslation() + ToReal3(delta));
      } else {
        // Change state of the rigid rotation
        rigidState.SetRotation(Quaternion::FromRotationVector(delta) * rigidState.GetRotation());
      }
      SetTransformFromState(_colliding, rigidState);
    } else {
      // Change state of a node of object B
      auto& posB = reg.get<CDisplacementSlice<real, TimeStep::Current>>(_collider);
      posB.value[localDof] += eps;
    }
  }
};

TEST_P(MochiRigidSoftContact, ContactAssembly) {
  RunTest();
}

INSTANTIATE_TEST_SUITE_P(
    FrictionVariations,
    MochiRigidSoftContact,
    ::testing::Values(
        TestParams{"NoFriction", GradTarget::Current, 0_r, 0_r, 0_r, false, 1e-3_r, 3e-1_r},
        TestParams{"Viscous", GradTarget::Current, 0_r, 10_r, 0_r, false, 1e-3_r, 2e-2_r},
        TestParams{"Coulomb", GradTarget::Current, 0.5_r, 0_r, 0_r, false, 1e-3_r, 6e-2_r},
        TestParams{"Damping", GradTarget::Current, 0_r, 0_r, 10_r, false, 1e-3_r, 2e-2_r},
        TestParams{"NoFrictionExplicit", GradTarget::Current, 0_r, 0_r, 0_r, true, 1e-3_r, 3e-1_r},
        TestParams{"ViscousExplicit", GradTarget::Current, 0_r, 10_r, 0_r, true, 1e-3_r, 2e-2_r},
        TestParams{"CoulombExplicit", GradTarget::Current, 0.5_r, 0_r, 0_r, true, 1e-3_r, 3e-1_r},
        TestParams{"DampingExplicit", GradTarget::Current, 0_r, 0_r, 10_r, true, 2e-3_r, 1e-2_r}),
    [](::testing::TestParamInfo<TestParams> const& info) { return info.param.name; });

// Scene with 2 soft actors, rotated as a whole. Actor A is a translated soft cube. Actor B is a
// soft cube at the scene's origin, and it has a mapped SDF _collider.
class MochiSoftSoftContact : public MochiContactTestBase {
 protected:
  ColumnVector<real> _posA;
  ColumnVector<real> _posB;
  bool _useDeepFlow = false;

 public:
  MochiSoftSoftContact() {
    _numDofsA = 24; // a soft with 8 nodes
    _numDofsB = 24; // a soft with 8 nodes
    _numDofs = _numDofsA + _numDofsB;
  }

  void InitializeScene(real coulombCoefficient, real viscousCoefficient, real dampingCoefficient)
      override {
    auto& reg = GetRegistry();
    Quaternion const sceneRotation = SceneRotation();

    // Translate A to avoid contacts on edges
    TransformRT const transformA =
        TransformRT(sceneRotation) * TransformRT(Vec4r(0.98_r, 0.24_r, 0.16_r));
    auto* actorA = InitSoftCube(
        transformA, false, {}, coulombCoefficient, viscousCoefficient, dampingCoefficient);
    auto* actorB = InitSoftCube(
        TransformRT(sceneRotation),
        true,
        _useDeepFlow,
        coulombCoefficient,
        viscousCoefficient,
        dampingCoefficient);
    InitActors(actorA, actorB);

    // Set initial positions
    auto& posA = reg.get<CDisplacementSlice<real, TimeStep::Current>>(_colliding);
    auto& posB = reg.get<CDisplacementSlice<real, TimeStep::Current>>(_collider);
    auto const coordsB = reg.get<CTetrahedralMesh const>(_collider).mesh->GetNodeCoordinates();
    for (int i = 0; i < _numDofsA; i++) {
      posA.value[i] = (i % 3 == 0) ? 0_r : 0.01_r;
      posB.value[i] = (i % 3 == 0) ? GetParam().colliderShear * coordsB[i / 3][1] : -0.01_r;
    }
    _posA = posA.value.Duplicate();
    _posB = posB.value.Duplicate();

    // Add velocity to the actors
    auto& posAold = reg.get<CDisplacementSlice<real, TimeStep::StageStart>>(_colliding);
    for (int i = 0; i < _numDofsA; i++) {
      posAold.value[i] = (i % 3 == 0) ? -0.01_r : 0.01_r;
    }
    auto& posBold = reg.get<CDisplacementSlice<real, TimeStep::StageStart>>(_collider);
    for (int i = 0; i < _numDofsB; i++) {
      posBold.value[i] =
          (i % 3 == 0) ? GetParam().colliderStretch * coordsB[i / 3][0] - 0.01_r : 0.01_r;
    }

    InitSoftCubeStageStart(_colliding);
    InitSoftCubeStageStart(_collider);
  }

  void InitState() override {
    auto& reg = GetRegistry();
    reg.get<CDisplacementSlice<real, TimeStep::Current>>(_colliding).value = _posA;
    reg.get<CDisplacementSlice<real, TimeStep::Current>>(_collider).value = _posB;
  }

  void AddToState(int globalDof, real eps) override {
    auto [entity, localDof] = GetEntityAndLocalDofFromGlobalDof(globalDof);
    auto& reg = GetRegistry();
    auto& pos = reg.get<CDisplacementSlice<real, TimeStep::Current>>(entity);
    pos.value[localDof] += eps;
  }
};

class MochiMappedColliderStageStartTest : public MochiSoftSoftContact {};

TEST_P(MochiMappedColliderStageStartTest, ImplicitPositionsMatchFullQuery) {
  InitializeScene(
      /*coulombCoefficient=*/0_r,
      /*viscousCoefficient=*/0_r,
      /*dampingCoefficient=*/0_r);
  InitState();
  PreStep();

  auto getStageStartContacts = [&](bool explicitNormals) {
    auto solverParams = _scene->GetSolverParams();
    solverParams.experimentalEval.explicitNormals = explicitNormals;
    _scene->SetSolverParams(solverParams, test::ExpectOK{});

    auto& reg = GetRegistry();
    entt::entity const island = reg.get<CIslandMemberInfo const>(_colliding).island;
    CollisionDetectionPipeline<TimeStep::Current>(reg, reg.get<CIslandDescendants const>(island));

    DynamicArray<int> sampleIndices;
    DynamicArray<Real3> positions;
    auto appendContacts = [&](auto const& collisions) {
      for (auto const& collision : collisions) {
        if (collision.colliderEntity != _collider) {
          continue;
        }
        auto const& result = collision.collisionResult;
        Append(sampleIndices, MakeConstSpan(result.sampleIndices));
        Append(positions, MakeConstSpan(result.posCollidingStageStart));
      }
    };
    if (auto const* collisions =
            reg.try_get<CActiveCollisions</*kIsSync*/ false, TimeStep::Current> const>(
                _colliding)) {
      appendContacts(*collisions);
    }
    if (auto const* collisions =
            reg.try_get<CActiveCollisions</*kIsSync*/ true, TimeStep::Current> const>(_colliding)) {
      appendContacts(*collisions);
    }
    EXPECT_FALSE(sampleIndices.empty());
    EXPECT_EQ(sampleIndices.size(), positions.size());
    return std::pair{std::move(sampleIndices), std::move(positions)};
  };

  auto const [expectedIndices, expectedPositions] = getStageStartContacts(true);
  auto const [actualIndices, actualPositions] = getStageStartContacts(false);
  EXPECT_SPAN_EQ(expectedIndices, actualIndices);
  EXPECT_SPAN_EQ(expectedPositions, actualPositions);
}

INSTANTIATE_TEST_SUITE_P(
    NoFriction,
    MochiMappedColliderStageStartTest,
    ::testing::Values(TestParams{}));

TEST_P(MochiSoftSoftContact, ContactAssembly) {
  RunTest();
}

INSTANTIATE_TEST_SUITE_P(
    FrictionVariations,
    MochiSoftSoftContact,
    ::testing::Values(
        TestParams{"NoFriction", GradTarget::Current, 0_r, 0_r, 0_r, false, 1e-3_r, 1e-1_r},
        TestParams{.name = "NoFrictionSheared", .dresTol = 1e-1_r, .colliderShear = 0.01_r},
        TestParams{"Viscous", GradTarget::Current, 0_r, 10_r, 0_r, false, 1e-3_r, 1e-1_r},
        TestParams{"Coulomb", GradTarget::Current, 0.5_r, 0_r, 0_r, false, 1e-3_r, 1e-1_r},
        TestParams{"Damping", GradTarget::Current, 0_r, 0_r, 10_r, false, 1e-3_r, 5e-3_r},
        TestParams{"NoFrictionExplicit", GradTarget::Current, 0_r, 0_r, 0_r, true, 1e-3_r, 1e-1_r},
        TestParams{"ViscousExplicit", GradTarget::Current, 0_r, 10_r, 0_r, true, 1e-3_r, 1e-1_r},
        TestParams{"CoulombExplicit", GradTarget::Current, 0.5_r, 0_r, 0_r, true, 1e-3_r, 1e-1_r},
        TestParams{"DampingExplicit", GradTarget::Current, 0_r, 0_r, 10_r, true, 1e-3_r, 5e-3_r},
        // The stretched stage-start map makes the colliding normal in collider space non-unit.
        TestParams{
            .name = "ViscousExplicitCollidingNormal",
            .viscousCoefficient = 10_r,
            .explicitNormals = true,
            .dresTol = 1e-1_r,
            .frictionWithColliderNormal = false,
            .colliderStretch = 0.05_r}),
    [](::testing::TestParamInfo<TestParams> const& info) { return info.param.name; });

class MochiSoftSoftContactDeepFlow : public MochiSoftSoftContact {
 public:
  MochiSoftSoftContactDeepFlow() {
    _useDeepFlow = true;
  }
};

// Test of the integration of deep flow and biharmonic ROMs
#if MOCHI_ENABLE_DEEP_FLOW_ACTORS && MOCHI_USE_HDF5
#define MOCHI_CAN_TEST_DEEP_FLOW 0 // TODO(T228959906): Re-enable deep flow tests
#else
#define MOCHI_CAN_TEST_DEEP_FLOW 0
#endif

// The Deep Flow model data is not shipped externally.
#if MOCHI_CAN_TEST_DEEP_FLOW && MOCHI_INTERNAL
#define MOCHI_CAN_TEST_DEEP_FLOW_INTERNAL 1
#else
#define MOCHI_CAN_TEST_DEEP_FLOW_INTERNAL 0
#endif

TEST_IF_P(MOCHI_CAN_TEST_DEEP_FLOW_INTERNAL, MochiSoftSoftContactDeepFlow, ContactAssembly) {
  RunTest();
}

INSTANTIATE_TEST_SUITE_P(
    FrictionVariations,
    MochiSoftSoftContactDeepFlow,
    ::testing::Values(
        TestParams{"Viscous", GradTarget::Current, 0_r, 10_r, 0_r, false, 6e-2_r, 1e-1_r}),
    [](::testing::TestParamInfo<TestParams> const& info) { return info.param.name; });

class DeepFlowBiharmonicRom : public MochiContactTestBase {
 protected:
  static constexpr real kCubeScale = 0.1_r;
  static constexpr real kDuckScale = 0.1_r;
  TransformRT _posA;
  ColumnVector<real> _posB;

 public:
  DeepFlowBiharmonicRom() {
    _numDofsA = 6; // Rigid
    _numDofsB = 30; // a biharmonic ROM with 30 DoFs
    _numDofs = _numDofsA + _numDofsB;
  }

  void InitializeScene(real coulombCoefficient, real viscousCoefficient, real dampingCoefficient)
      override {
    auto* actorA = InitRigidCube(
        kCubeScale, {}, false, coulombCoefficient, viscousCoefficient, dampingCoefficient);
    auto* actorB = InitRomDuck(coulombCoefficient, viscousCoefficient, dampingCoefficient);
    InitActors(actorA, actorB);
    auto& reg = GetRegistry();
    _posA = reg.get<CRigidState<TimeStep::Current>>(_colliding).value;
    _posB = reg.get<CRomModeAmplitudes const>(_collider).value.Duplicate();
  }

  Actor* InitRomDuck(real coulombCoefficient, real viscousCoefficient, real dampingCoefficient) {
    ShapeHandle duck = _mochiContext->LoadShapeFromFile(
        test::GetAssetPath("duck/duck_coarse.mochi.h5"),
        Real3{kDuckScale, kDuckScale, kDuckScale},
        TransformRT::Identity(),
        test::ExpectOK());
    DeepModelParams dflowparams;
    dflowparams.deepModelPath =
        test::GetAssetPath("duck/duck_coarse_biharmonic_rom/duck_10handles_flow.mochi.h5");
    dflowparams.shiftX = 0.43273339_r;
    dflowparams.shiftY = 0.290418_r;
    dflowparams.shiftZ = 0.49103763_r;
    dflowparams.scale = 0.6447050910181614_r;
    dflowparams.numDof = 30;
    ShapeHandle flow = CreateDeepFlowShape(_mochiContext, dflowparams, test::ExpectOK());
    SoftActorParams params;
    params.layer = "Collider";
    params.shape = duck;
    params.worldFromLocal.SetTranslation(Real3(0_r, 0.96_r * kDuckScale, 0_r));
    params.material.density = 1e-10_r; // Negligible density to isolate contact forces from inertia.
    params.material.type = SoftMaterialType::NeoHookean;
    params.material.neoHookean.youngsModulus = 1e-6_r;
    params.contact.objScale = kDuckScale;
    params.contact.penaltyThresholdDefault = 0_r;
    params.contact.penaltySmoothingHalfDistance = 0.001_r;
    params.contact.viscousFrictionCoefficient = viscousCoefficient;
    params.contact.coulombFrictionCoefficient = coulombCoefficient;
    params.contact.normalViscousDampingCoefficient = dampingCoefficient;
    params.contact.distanceErrorBound = kDuckScale * dflowparams.errorBound;

    ExperimentalSoftActorParams experimentalParams;
    experimentalParams.colliderType = ColliderType::Sdf;
    experimentalParams.flow = flow;
    experimentalParams.rom = RomParams{.source = "biharmonic_10"};

    return CreateSoftActor(_scene, params, experimentalParams, test::ExpectOK());
  }

  void PostResetStateRom(entt::entity entity) {
    auto& reg = GetRegistry();

    // Update FOM state
    // ResolveDisplacement requires setting the template kForceUseAllNodes.
    // This test does not use hyper-reduction on the ROM actor, so we could
    // set it true or false and should have the same result.
    ecs::TryInvokeOnEntity(
        &rom::linear::ResolveDisplacement</*kForceUseAllNodes = */ true>, reg, entity);

    // Update the current velocity of the ROM in reduced coordinates
    ecs::TryInvokeOnEntity(&rom::UpdateCurrentRomVelocity, reg, entity);
  }

  void InitState() override {
    auto& reg = GetRegistry();
    reg.get<CRigidState<TimeStep::Current>>(_colliding).value = _posA;
    SetTransformFromState(_colliding, _posA);
    reg.get<CRomModeAmplitudes>(_collider).value = _posB;
    PostResetStateRom(_collider);
  }

  void AddToState(int globalDof, real eps) override {
    auto [entity, localDof] = GetEntityAndLocalDofFromGlobalDof(globalDof);
    auto& reg = GetRegistry();
    if (entity == _colliding) {
      Vec4r delta = SimdBasisVector(localDof % 3) * eps;
      auto& rigidState = reg.get<CRigidState<TimeStep::Current>>(_colliding).value;
      if (localDof < 3) {
        // Change state of the rigid translation
        rigidState.SetTranslation(rigidState.GetTranslation() + ToReal3(delta));
      } else {
        // Change state of the rigid rotation
        rigidState.SetRotation(Quaternion::FromRotationVector(delta) * rigidState.GetRotation());
      }
      SetTransformFromState(_colliding, rigidState);
    } else {
      // Change ROM state
      auto& pos = reg.get<CRomModeAmplitudes>(_collider).value;
      pos(localDof) += eps;
      PostResetStateRom(_collider);
    }
  }
};

TEST_IF_P(MOCHI_CAN_TEST_DEEP_FLOW_INTERNAL, DeepFlowBiharmonicRom, ContactAssembly) {
  RunTest();
}

INSTANTIATE_TEST_SUITE_P(
    FrictionVariations,
    DeepFlowBiharmonicRom,
    ::testing::Values(
        TestParams{"Viscous", GradTarget::Current, 0_r, 1_r, 0_r, false, 1e-2_r, 2e-1_r, false},
        TestParams{
            "Coulomb",
            GradTarget::Current,
            0.5_r,
            0_r,
            0_r,
            false,
            1e-2_r,
            2e-1_r,
            false,
            1e-5_r}),
    [](::testing::TestParamInfo<TestParams> const& info) { return info.param.name; });

// Scene with 2 articulated actors. Each articulated actor consists of 2 rigid actors connected
// through a revolute joint.
using namespace mochi::articulated;
using namespace mochi::articulated::compound;
class MochiArticulatedArticulatedContact : public MochiContactTestBase {
 protected:
  // This class needs to overwrite the 'colliding' and 'collider' actors to use rigid bodies.
  // '_artA' and '_artB' store the articulated bodies for state updates.
  entt::entity _artA = entt::null;
  entt::entity _artB = entt::null;
  ColumnVector<real> _posA;
  ColumnVector<real> _posB;
  ColumnVector<real> _posAOld;
  ColumnVector<real> _posBOld;

 public:
  MochiArticulatedArticulatedContact() {
    _numDofsA = 9; // two bones with one spherical joint
    _numDofsB = 9; // two bones with one spherical joint
    _numDofs = _numDofsA + _numDofsB;
  }

  Actor* CreateArticulated(
      ShapeHandle const& boneShape,
      RigidActorParams const& rigidParams,
      std::vector<TransformRT> const& worldFromLocalTxs) {
    ArticulatedActorParams params;

    params.joints.resize(2);
    params.joints[0].type = ArticulatedJointType::Free;
    params.joints[0].parentLinkFromJoint = worldFromLocalTxs[0];
    params.joints[1].type = ArticulatedJointType::Spherical;
    params.joints[1].parentLinkFromJoint = Invert(worldFromLocalTxs[0]) * worldFromLocalTxs[1];

    params.links.resize(2);
    for (int i = 0; i < 2; ++i) {
      params.links[i].parentLink = i == 0 ? -1 : 0;
      params.links[i].shape = boneShape;
      params.links[i].layer = rigidParams.layer;
      params.links[i].colliderType = rigidParams.colliderType;
      params.links[i].contact = rigidParams.contact;
    }

    Actor* articulatedActor = _scene->CreateArticulatedActor(params, test::ExpectOK{});

    // Set time step size.
    auto actors = articulatedActor->GetNestedLinkActors(test::ExpectOK{});
    auto& reg = GetRegistry();
    for (auto actor : actors) {
      auto entity = mochi::GetEntity(reg, actor, test::ExpectOK{});
      reg.get<CTimeIntegratorState>(entity).dtStage = kDtStage;
    }

    return articulatedActor;
  }

  void InitializeScene(real coulombCoefficient, real viscousCoefficient, real dampingCoefficient)
      override {
    auto& reg = GetRegistry();

    // Create bone shape
    ShapeHandle boneShape = _mochiContext->LoadShapeFromFile(
        test::GetAssetPath("cube/cube_mesh.mochi.json"), test::ExpectOK{});

    // Create colliding
    Actor* actorA = nullptr;
    {
      std::vector<TransformRT> worldFromLocalTxs = {{}, TransformRT(Real3(1_r, 0_r, 0_r))};

      RigidActorParams rigidParams;
      rigidParams.contact.viscousFrictionCoefficient = viscousCoefficient;
      rigidParams.contact.coulombFrictionCoefficient = coulombCoefficient;
      rigidParams.contact.normalViscousDampingCoefficient = dampingCoefficient;
      rigidParams.layer = "Colliding";

      actorA = CreateArticulated(boneShape, rigidParams, worldFromLocalTxs);
    }

    Actor* actorB = nullptr;
    {
      std::vector<TransformRT> worldFromLocalTxs = {
          TransformRT(Real3(2.5_r, 1_r, 0_r)), TransformRT(Real3(1.5_r, 1_r, 0_r))};

      RigidActorParams rigidParams;
      rigidParams.layer = "Collider";
      rigidParams.colliderType = ColliderType::Box;
      rigidParams.contact.penaltyThresholdDefault = 0_r;
      rigidParams.contact.viscousFrictionCoefficient = viscousCoefficient;
      rigidParams.contact.coulombFrictionCoefficient = coulombCoefficient;
      rigidParams.contact.normalViscousDampingCoefficient = dampingCoefficient;

      actorB = CreateArticulated(boneShape, rigidParams, worldFromLocalTxs);
    }

    // Create the compound and overwrite 'colliding' and 'collider'
    InitActors(actorA, actorB);
    _artA = _colliding;
    _artB = _collider;
    auto actorA2 = actorA->GetNestedLinkActors(test::ExpectOK{})[1];
    auto actorB2 = actorB->GetNestedLinkActors(test::ExpectOK{})[1];
    _colliding = mochi::GetEntity(reg, actorA2, test::ExpectOK{});
    _collider = mochi::GetEntity(reg, actorB2, test::ExpectOK{});

    // Initialize previous state
    CopyToPreviousState(_colliding);
    CopyToPreviousState(_collider);

    // Add some initial rotations
    auto tinyRot = Real3{0_r, 0_r, 0.01_r};
    auto& posA = reg.get<CArticulatedReducedPose<TimeStep::Current>>(_artA).value;
    posA.BottomRows<RigidSize::kRot>(RigidSize::kRot) =
        AsColumnVectorView(Quaternion::FromRotationVector(tinyRot).data);
    auto& posB = reg.get<CArticulatedReducedPose<TimeStep::Current>>(_artB).value;
    posB.BottomRows<RigidSize::kRot>(RigidSize::kRot) =
        AsColumnVectorView(Quaternion::FromRotationVector(-tinyRot).data);
    UpdateFullState(_artA);
    UpdateFullState(_artB);
    _posA = posA.Duplicate();
    _posB = posB.Duplicate();

    // The stage-start poses differ from the current ones, so that the links have velocity, and the
    // Jacobians of GradTarget::Current and GradTarget::Previous differ.
    constexpr std::array<real, 9> kStageStartDelta = {
        0.02_r, -0.01_r, 0.01_r, 0.03_r, -0.02_r, 0.01_r, -0.02_r, 0.03_r, 0.02_r};
    _posAOld = _posA.Duplicate();
    _posBOld = _posB.Duplicate();
    for (int i = 0; i < isize(kStageStartDelta); ++i) {
      AddToPose(_artA, i, kStageStartDelta[i], _posAOld);
      AddToPose(_artB, i, -kStageStartDelta[i], _posBOld);
    }

    // Update islands manually since this test does not step the scene.
    // This assigns DOF offsets, among other things.
    island::PreStep(reg, /*updateSleepState*/ true);
  }

  std::pair<entt::entity, int> GetEntityAndLocalDofFromGlobalDof(int globalDof) override {
    // In this case the global DOFs belong to _artA or _artB, not the colliding or collider
    // entities, which are full DOF rigid bodies. See InitializeScene().
    auto const& reg = GetRegistry();
    auto dofOffsetA = reg.get<CDofOffset>(_artA).dofsOffset;
    auto dofOffsetB = reg.get<CDofOffset>(_artB).dofsOffset;
    if ((globalDof >= dofOffsetA) && (globalDof < dofOffsetA + _numDofsA)) {
      return std::make_pair(_artA, globalDof - dofOffsetA);
    } else {
      EXPECT_TRUE(
          (globalDof >= dofOffsetB) &&
          (globalDof < dofOffsetB + _numDofsB)); // DOF index out of range
      return std::make_pair(_artB, globalDof - dofOffsetB);
    }
  }

  void InitState() override {
    auto& reg = GetRegistry();
    reg.get<CArticulatedReducedPose<TimeStep::Current>>(_artA).value = _posA;
    reg.get<CArticulatedReducedPose<TimeStep::Current>>(_artB).value = _posB;
    UpdateFullState(_artA);
    UpdateFullState(_artB);
    reg.get<CArticulatedReducedPose<TimeStep::StageStart>>(_artA).value = _posAOld;
    reg.get<CArticulatedReducedPose<TimeStep::StageStart>>(_artB).value = _posBOld;
    UpdateStageStartState(_artA);
    UpdateStageStartState(_artB);
  }

  void AddToState(int globalDof, real eps) override {
    auto [art, localDof] = GetEntityAndLocalDofFromGlobalDof(globalDof);
    AddToPose(
        art,
        localDof,
        eps,
        GetRegistry().get<CArticulatedReducedPose<TimeStep::Current>>(art).value);
    UpdateFullState(art);
  }

  void AddToOldState(int globalDof, real eps) override {
    auto [art, localDof] = GetEntityAndLocalDofFromGlobalDof(globalDof);
    AddToPose(
        art,
        localDof,
        eps,
        GetRegistry().get<CArticulatedReducedPose<TimeStep::StageStart>>(art).value);
    UpdateStageStartState(art);
  }

  void AddToPose(entt::entity art, int localDof, real eps, ColumnVector<real>& pose) {
    auto const& reg = GetRegistry();
    ColumnVector<real> delta(reg.get<CArticulatedProps const>(art).reducedDofsDim);
    delta.SetZero();
    delta[localDof] = eps;
    auto const* joints = reg.get<CArticulatedBodyShape const>(art).shape->GetJointsData();
    auto const& poseInfo = reg.get<CArticulatedJointPoseInfo const>(art);
    articulated::AddLieDeltaToReducedPose(
        joints->jointTypes, joints->dofInfo, poseInfo, pose, delta, pose);
  }

  // Updates the stage-start transforms of the articulated actor, and the stage-start states of all
  // its links, from its stage-start pose, as articulated::compound::PreStagePipeline does.
  void UpdateStageStartState(entt::entity art) {
    auto& reg = GetRegistry();
    auto const* joints = reg.get<CArticulatedBodyShape const>(art).shape->GetJointsData();
    ColumnVector<real> fullPose(reg.get<CArticulatedFullPose const>(art).value.Rows());
    articulated::ComputeFullPose(
        joints->jointTypes,
        joints->jointAxes,
        reg.get<CArticulatedJointPoseInfo const>(art),
        reg.get<CArticulatedParents const>(art),
        reg.get<CArticulatedRestTransforms const>(art),
        reg.get<CRootTransform const>(art).worldFromLocal,
        reg.get<CArticulatedReducedPose<TimeStep::StageStart> const>(art).value,
        reg.get<CArticulatedJointTransforms<TimeStep::StageStart>>(art),
        reg.get<CArticulatedLinkTransforms<TimeStep::StageStart>>(art),
        fullPose);
    for (entt::entity const link : reg.get<CGroupMembers const>(art).actors) {
      int const offset = reg.get<CDofOffset const>(link).poseOffset;
      auto& rigidState = reg.get<CRigidState<TimeStep::StageStart>>(link);
      mochi::rigid::EntitySetSolution(
          AsConstView(fullPose).MiddleRows(offset, RigidSize::kAll),
          ecs::Included<TagRigidActor>{},
          {},
          rigidState);
      SetStageStartTransformFromState(link, rigidState.value);
    }
  }

  // Updates the full pose of the articulated actor, and the current states of all its links, from
  // its current pose, as UpdateDerivedStateFromPose does.
  void UpdateFullState(entt::entity art) {
    auto& reg = GetRegistry();
    UpdateArticulatedFullPose(reg, art);

    auto const& posFull = reg.get<CArticulatedFullPose const>(art).value;
    for (entt::entity const link : reg.get<CGroupMembers const>(art).actors) {
      int const offset = reg.get<CDofOffset const>(link).poseOffset;
      auto& rigidState = reg.get<CRigidState<TimeStep::Current>>(link);
      mochi::rigid::EntitySetSolution(
          posFull.MiddleRows(offset, RigidSize::kAll),
          ecs::Included<TagRigidActor>{},
          {},
          rigidState);
      SetTransformFromState(link, rigidState.value);
    }
  }

  void CopyToPreviousState(entt::entity body) {
    auto& reg = GetRegistry();

    auto const& rigidState = reg.get<CRigidState<TimeStep::Current> const>(body).value;
    auto& prevRigidState = reg.get<CRigidState<TimeStep::Previous>>(body).value;
    prevRigidState = rigidState;
  }
};

TEST_P(MochiArticulatedArticulatedContact, ContactAssembly) {
  RunTest();
}

// The residual matches the FD of the objective to round-off where it is its true gradient: without
// friction and damping, or with GradTarget::Current and explicit normals. Otherwise, friction and
// damping make the residual differ from that gradient, more so for larger stage velocities.
constexpr real kResExactTol = 2e-3_r;
constexpr real kResApproxTol = 5e-2_r;
// The dresidual of friction and damping discards some of their nonlinearities.
constexpr real kDResSmallTol = 1e-2_r;
constexpr real kDResLargeTol = 5e-2_r;

INSTANTIATE_TEST_SUITE_P(
    FrictionVariations,
    MochiArticulatedArticulatedContact,
    ::testing::Values(
        TestParams{.name = "NoFriction", .resTol = kResExactTol, .dresTol = kDResSmallTol},
        TestParams{
            .name = "Viscous",
            .viscousCoefficient = 10_r,
            .resTol = kResApproxTol,
            .dresTol = kDResLargeTol},
        TestParams{
            .name = "Coulomb",
            .coulombCoefficient = 0.5_r,
            .resTol = kResApproxTol,
            .dresTol = kDResLargeTol},
        TestParams{
            .name = "Damping",
            .dampingCoefficient = 10_r,
            .resTol = kResApproxTol,
            .dresTol = kDResLargeTol},
        TestParams{
            .name = "NoFrictionExplicit",
            .explicitNormals = true,
            .resTol = kResExactTol,
            .dresTol = kDResSmallTol},
        TestParams{
            .name = "ViscousExplicit",
            .viscousCoefficient = 10_r,
            .explicitNormals = true,
            .resTol = kResExactTol,
            .dresTol = kDResLargeTol},
        TestParams{
            .name = "CoulombExplicit",
            .coulombCoefficient = 0.5_r,
            .explicitNormals = true,
            .resTol = kResExactTol,
            .dresTol = kDResLargeTol},
        TestParams{
            .name = "DampingExplicit",
            .dampingCoefficient = 10_r,
            .explicitNormals = true,
            .resTol = kResExactTol,
            .dresTol = kDResLargeTol},
        // The residual of GradTarget::Previous is projected with the stage-start Jacobian. The
        // dresidual is not tested.
        TestParams{
            .name = "NoFrictionPrevious",
            .gradTarget = GradTarget::Previous,
            .explicitNormals = true,
            .resTol = kResExactTol},
        TestParams{
            .name = "ViscousPrevious",
            .gradTarget = GradTarget::Previous,
            .viscousCoefficient = 10_r,
            .explicitNormals = true,
            .resTol = kResApproxTol},
        TestParams{
            .name = "CoulombPrevious",
            .gradTarget = GradTarget::Previous,
            .coulombCoefficient = 0.5_r,
            .explicitNormals = true,
            .resTol = kResApproxTol},
        TestParams{
            .name = "DampingPrevious",
            .gradTarget = GradTarget::Previous,
            .dampingCoefficient = 10_r,
            .explicitNormals = true,
            .resTol = kResApproxTol}),
    [](::testing::TestParamInfo<TestParams> const& info) { return info.param.name; });

class SkinnedContactPairCapacityTest : public test::MochiSceneTestBase {
 protected:
  static constexpr int kNumActiveBones = 5;
  static constexpr int kNumPartitions = (1 << kNumActiveBones) - 1;

  ShapeHandle CreatePartitionedSkinShape() {
    DynamicArray<Real3> coordinates;
    DynamicArray<Int3> connectivity;
    SkinningData skinning;
    skinning.weightsPerNode = kNumActiveBones;
    coordinates.reserve(3 * kNumPartitions);
    connectivity.reserve(kNumPartitions);
    skinning.indices.reserve(3 * kNumPartitions * kNumActiveBones);
    skinning.weights.reserve(3 * kNumPartitions * kNumActiveBones);

    for (int partition = 0; partition < kNumPartitions; ++partition) {
      auto const boneMask = static_cast<unsigned int>(partition + 1);
      real const boneWeight = 1_r / static_cast<real>(std::popcount(boneMask));

      real const y = 0.08_r + 0.14_r * static_cast<real>(partition % 6);
      real const z = 0.08_r + 0.14_r * static_cast<real>(partition / 6);
      int const firstNode = isize(coordinates);
      coordinates.emplace_back(Real3{0.01_r, y, z});
      coordinates.emplace_back(Real3{0.01_r, y + 0.04_r, z});
      coordinates.emplace_back(Real3{0.01_r, y, z + 0.04_r});
      connectivity.emplace_back(Int3{firstNode, firstNode + 1, firstNode + 2});

      for (int vertex = 0; vertex < 3; ++vertex) {
        for (int bone = 0; bone < kNumActiveBones; ++bone) {
          skinning.indices.emplace_back(bone + 1);
          skinning.weights.emplace_back(((boneMask >> bone) & 1U) != 0 ? boneWeight : 0_r);
        }
      }
    }

    auto mesh = std::make_unique<TriangularMesh>(coordinates, connectivity);
    auto shape = std::make_shared<TriangularMeshShape>(
        std::move(mesh), std::make_shared<SkinningData const>(std::move(skinning)));
    return assert_cast<ContextImpl*>(_mochiContext)->RegisterShape(shape, test::ExpectOK{});
  }

  ArticulatedActorParams MakeArticulationParams(
      ShapeHandle skinShape,
      std::string_view skinLayer,
      std::string_view linkLayer) {
    ArticulatedActorParams params;
    params.joints.resize(kNumActiveBones + 1);
    params.links.resize(kNumActiveBones + 1);
    params.joints[0].type = ArticulatedJointType::Hard;

    ShapeHandle const linkShape = test::CreateUnitCubeTetMeshShape(_mochiContext);
    for (int bone = 0; bone < kNumActiveBones; ++bone) {
      int const link = bone + 1;
      params.joints[link].type = ArticulatedJointType::Prismatic;
      params.joints[link].axis = Real3{1_r, 0_r, 0_r};
      params.links[link].parentLink = 0;
      params.links[link].shape = linkShape;
      params.links[link].layer = linkLayer;
      params.links[link].colliderType = ColliderType::Box;
      params.links[link].hasGravity = false;
      params.links[link].contact.penaltyThresholdDefault = 0_r;
    }

    ArticulatedSkinParams skin;
    skin.shape = skinShape;
    skin.layer = skinLayer;
    skin.nonCollidingLinks = DynamicArray<DynamicString>{};
    skin.contact.penaltyThresholdDefault = 0_r;
    params.skin = std::move(skin);
    return params;
  }
};

TEST_F(SkinnedContactPairCapacityTest, PartitionedPairsCanExceedActorPairs) {
  _scene->SetGravity(Real3{});
  ShapeHandle const skinShape = CreatePartitionedSkinShape();
  Actor* const actorA = _scene->CreateArticulatedActor(
      MakeArticulationParams(skinShape, "SkinA", "LinksA"), test::ExpectOK{});
  Actor* const actorB = _scene->CreateArticulatedActor(
      MakeArticulationParams(skinShape, "SkinB", "LinksB"), test::ExpectOK{});
  ASSERT_NE(nullptr, actorA);
  ASSERT_NE(nullptr, actorB);

  std::array<std::string_view, 4> const layers{"SkinA", "LinksA", "SkinB", "LinksB"};
  for (auto const collidingLayer : layers) {
    for (auto const colliderLayer : layers) {
      _scene->EnableLayerContactAsymmetric(collidingLayer, colliderLayer, false, test::ExpectOK{});
    }
  }
  _scene->EnableLayerContactAsymmetric("SkinA", "LinksB", true, test::ExpectOK{});
  _scene->EnableLayerContactAsymmetric("SkinB", "LinksA", true, test::ExpectOK{});

  auto& reg = GetRegistry();
  entt::entity const entityA = GetEntity(actorA->GetHandle());
  entt::entity const entityB = GetEntity(actorB->GetHandle());
  EXPECT_EQ(kNumPartitions, isize(reg.get<CContactPartitions const>(entityA)));
  EXPECT_EQ(kNumPartitions, isize(reg.get<CContactPartitions const>(entityB)));

  _scene->Step(1e-3_r);

  entt::entity const island = reg.get<CIslandMemberInfo const>(entityA).island;
  ASSERT_EQ(island, reg.get<CIslandMemberInfo const>(entityB).island);
  auto const& actors = reg.get<CIslandDescendants const>(island).actors;
  int numActiveCollisions = 0;
  for (auto actor : actors) {
    if (auto const* activeCollisions =
            reg.try_get<CActiveCollisions</*kIsSync*/ true, TimeStep::Current> const>(actor)) {
      numActiveCollisions += isize(*activeCollisions);
    }
  }
  EXPECT_GT(numActiveCollisions, Sqr(isize(actors)));
}

class ZeroDofSkinContact : public test::MochiSceneTestBase {
 protected:
  static constexpr double kTimeStep = 1e-2;
  // Box extents along x that overlap only zero-DoF partitions, or both zero-DoF and DoF-bearing
  // ones.
  static constexpr real kZeroDofBoxMaxX = 0.05_r;
  static constexpr real kMixedBoxMaxX = 0.3_r;

  ShapeHandle CreateSkinShape() {
    auto&& [coordinates, connectivity] = test::CreateMinimalTriMeshUnitCube();
    auto mesh = std::make_unique<TriangularMesh>(coordinates, connectivity);
    SkinningData skinning;
    skinning.weightsPerNode = 1;
    for (Real3 const& coordinate : coordinates) {
      skinning.indices.push_back(coordinate[0] == 0_r ? 0 : 1);
      skinning.weights.push_back(1_r);
    }
    auto shape = std::make_shared<TriangularMeshShape>(
        std::move(mesh), std::make_shared<SkinningData const>(std::move(skinning)));
    return assert_cast<ContextImpl*>(_mochiContext)->RegisterShape(shape, test::ExpectOK{});
  }

  ShapeHandle CreatePartiallyBlendedSkinShape() {
    auto&& [coordinates, connectivity] = test::CreateMinimalTriMeshUnitCube();
    auto mesh = std::make_unique<TriangularMesh>(coordinates, connectivity);
    auto skinning = std::make_shared<SkinningData const>(
        test::MakeSingleBoneSkinning(isize(coordinates), /*boneIndex=*/0));
    BlendingDataTargetMesh target;
    target.indices.resize(coordinates.size());
    target.weights.resize(coordinates.size());
    for (int node = 0; node < isize(coordinates); ++node) {
      target.indices[node] = node;
      target.weights[node] = coordinates[node][0] == 0_r ? 0_r : 0.5_r;
    }
    auto blending = std::make_shared<BlendingDataMap>();
    blending->perSourceShapeData.emplace(DynamicString{"soft"}, std::move(target));
    auto shape = std::make_shared<TriangularMeshShape>(
        std::move(mesh), skinning, /*constrainedNodesData=*/nullptr, std::move(blending));
    return assert_cast<ContextImpl*>(_mochiContext)->RegisterShape(shape, test::ExpectOK{});
  }

  ArticulatedActorParams MakeSkinParams(ShapeHandle skinShape) {
    ArticulatedActorParams params;
    params.joints = {
        {.type = ArticulatedJointType::Hard},
        {.type = ArticulatedJointType::Prismatic, .axis = Real3{1_r, 0_r, 0_r}}};
    ShapeHandle const linkShape = test::CreateUnitCubeTetMeshShape(_mochiContext);
    params.links = {
        {.parentLink = -1,
         .shape = linkShape,
         .colliderType = ColliderType::None,
         .hasGravity = false},
        {.parentLink = 0,
         .shape = linkShape,
         .colliderType = ColliderType::None,
         .hasGravity = false}};
    params.skin = ArticulatedSkinParams{.shape = skinShape, .layer = "Skin"};
    params.skin->contact.penaltyThresholdDefault = 0_r;
    return params;
  }

  Actor* CreateBox(bool isStatic, real boxMaxX) {
    RigidActorParams boxParams;
    boxParams.layer = "Box";
    auto&& [boxCoordinates, boxConnectivity] =
        test::CreateMinimalTetMeshUnitCube(Real3{boxMaxX + 0.95_r, 2_r, 2_r});
    boxParams.shape = _mochiContext->CreateTetMeshShape(
        Flatten(MakeSpan(boxCoordinates)), Flatten(MakeSpan(boxConnectivity)), test::ExpectOK{});
    boxParams.worldFromLocal = TransformRT{Real3{-0.95_r, -0.5_r, -0.5_r}};
    boxParams.colliderType = ColliderType::Box;
    boxParams.isStatic = isStatic;
    boxParams.hasGravity = false;
    boxParams.mass = 1_r;
    boxParams.contact.penaltyThresholdDefault = 0_r;
    return _scene->CreateRigidActor(boxParams, test::ExpectOK{});
  }

  std::pair<Actor*, Actor*> CreateActors(bool isBoxStatic, real boxMaxX = kZeroDofBoxMaxX) {
    Actor* skin =
        _scene->CreateArticulatedActor(MakeSkinParams(CreateSkinShape()), test::ExpectOK{});
    Actor* box = CreateBox(isBoxStatic, boxMaxX);

    _scene->EnableLayerContactSymmetric("Skin", "Box", true, test::ExpectOK{});
    return {skin, box};
  }

  std::pair<Actor*, Actor*> CreateBlendedActors(real boxMaxX) {
    SoftSkinnedActorParams params;
    params.skeletonParams = MakeSkinParams(CreatePartiallyBlendedSkinShape());
    SoftActorParams softParams;
    softParams.name = "soft";
    softParams.shape = test::CreateUnitCubeTetSoftShape(_mochiContext);
    softParams.hasGravity = false;
    softParams.hasStress = false;
    params.softParams = {softParams};
    Actor* skin = _scene->CreateSoftSkinnedActor(params, test::ExpectOK{});
    Actor* box = CreateBox(/*isStatic=*/true, boxMaxX);
    _scene->EnableLayerContactSymmetric("Skin", "Box", true, test::ExpectOK{});
    return {skin, box};
  }

  static void ShiftBoxX(Actor* box, real dx) {
    TransformRT transform = box->GetRootTransform();
    transform.SetTranslation(transform.GetTranslation() + Real3{dx, 0_r, 0_r});
    box->SetRootTransform(transform, test::ExpectOK{});
  }

  bool UsesSkinnedContactInSolver(entt::entity skinEntity) {
    return GetRegistry().get<CSkinnedContactSnle const>(skinEntity).useInSolver;
  }

  real GetPrismaticPose(entt::entity skinEntity) {
    return GetRegistry().get<CArticulatedReducedPose<TimeStep::Current> const>(skinEntity).value[0];
  }

  static void ExpectJacDataMatches(
      std::vector<JacData> const& jacs,
      int jacId,
      entt::entity otherEntity,
      ContactDetectionResult const& result) {
    ASSERT_GE(jacId, 0);
    ASSERT_LT(jacId, isize(jacs));
    JacData const& jac = jacs[jacId];
    EXPECT_EQ(otherEntity, jac.otherEntity);
    EXPECT_EQ(result.collidingPartitionId, jac.collidingPartitionId);
    EXPECT_EQ(&result, jac.query);
  }

  // Returns the number of active collisions checked.
  template <bool kIsSync>
  int ExpectActiveCollisionsMatchJacData(entt::entity e) const {
    auto const& reg = GetRegistry();
    auto const& collisions = reg.get<CActiveCollisions<kIsSync, TimeStep::Current> const>(e);
    auto const& collidingJacs = reg.get<CCollJacs<CollRole::Colliding> const>(e);
    for (auto const& collision : collisions) {
      ExpectJacDataMatches(
          collidingJacs,
          collision.collidingJacId,
          collision.colliderEntity,
          collision.collisionResult);
      if constexpr (kIsSync) {
        ExpectJacDataMatches(
            reg.get<CCollJacs<CollRole::Collider> const>(collision.colliderEntity),
            collision.colliderJacId,
            e,
            collision.collisionResult);
      }
    }
    return isize(collisions);
  }

  // Sparse partitions drop and recreate JacData as contact moves across partitions. Every active
  // collision must index the JacData of its pair, and no stale JacData may remain.
  void ExpectJacDataMatchesActiveCollisions() const {
    auto const& reg = GetRegistry();
    reg.view<CCollJacs<CollRole::Colliding> const>().each(
        [&](entt::entity e, auto const& collidingJacs) {
          int const numCollisions = ExpectActiveCollisionsMatchJacData</*kIsSync*/ false>(e) +
              ExpectActiveCollisionsMatchJacData</*kIsSync*/ true>(e);
          EXPECT_EQ(numCollisions, isize(collidingJacs));
        });
    reg.view<CCollJacs<CollRole::Collider> const>().each(
        [&](entt::entity collider, auto const& colliderJacs) {
          int numSyncCollisions = 0;
          reg.view<CActiveCollisions</*kIsSync*/ true, TimeStep::Current> const>().each(
              [&](auto const& collisions) {
                numSyncCollisions += StaticCast<int>(std::count_if(
                    collisions.begin(), collisions.end(), [&](ActiveCollision const& collision) {
                      return collision.colliderEntity == collider;
                    }));
              });
          EXPECT_EQ(numSyncCollisions, isize(colliderJacs));
        });
  }

  void Step() {
    _scene->Step(kTimeStep);
    ExpectJacDataMatchesActiveCollisions();
  }

  void ExpectEmptyAndNonemptyPartitions(entt::entity skinEntity) {
    auto const& partitions = GetRegistry().get<CContactPartitions const>(skinEntity);
    bool hasEmpty = false;
    bool hasNonempty = false;
    for (int partitionId = 0; partitionId < isize(partitions); ++partitionId) {
      auto const& descriptors = partitions[partitionId].GetDofDescriptors();
      ASSERT_EQ(1, isize(descriptors));
      auto const& boneIds = std::get<DynamicArray<int>>(descriptors[0]);
      hasEmpty |= boneIds.empty();
      hasNonempty |= !boneIds.empty();
    }
    EXPECT_TRUE(hasEmpty);
    EXPECT_TRUE(hasNonempty);
  }

  template <bool kIsSync>
  std::pair<int, int> CountContactsByDofType(entt::entity skinEntity, entt::entity boxEntity) {
    auto const& reg = GetRegistry();
    auto const& partitions = reg.get<CContactPartitions const>(skinEntity);
    auto const& collisions =
        reg.get<CActiveCollisions<kIsSync, TimeStep::Current> const>(skinEntity);
    int zeroDofContacts = 0;
    int nonzeroDofContacts = 0;
    for (auto const& collision : collisions) {
      if (collision.collisionResult.Empty()) {
        continue;
      }
      EXPECT_EQ(boxEntity, collision.colliderEntity);
      auto const& descriptors =
          partitions[collision.collisionResult.collidingPartitionId].GetDofDescriptors();
      EXPECT_EQ(1, isize(descriptors));
      if (isize(descriptors) != 1) {
        continue;
      }
      int& count = std::get<DynamicArray<int>>(descriptors[0]).empty() ? zeroDofContacts
                                                                       : nonzeroDofContacts;
      count += isize(collision.collisionResult.sampleIndices);
    }
    return {zeroDofContacts, nonzeroDofContacts};
  }

  struct BlendedContactStats {
    int contactsWithoutSoftDofs = 0;
    int contactsWithSoftDofs = 0;
    real maxForceWithoutSoftDofs = 0_r;
  };

  BlendedContactStats GetBlendedContactStats(entt::entity skinEntity, entt::entity boxEntity) {
    auto const& reg = GetRegistry();
    auto const& collisions =
        reg.get<CActiveCollisions</*kIsSync*/ false, TimeStep::Current> const>(skinEntity);
    auto const& partitions = reg.get<CContactPartitions const>(skinEntity);
    BlendedContactStats stats;
    for (auto const& collision : collisions) {
      auto const& result = collision.collisionResult;
      if (result.Empty()) {
        continue;
      }
      EXPECT_EQ(boxEntity, collision.colliderEntity);
      auto const& descriptors = partitions[result.collidingPartitionId].GetDofDescriptors();
      EXPECT_EQ(2, isize(descriptors));
      if (isize(descriptors) != 2) {
        continue;
      }
      EXPECT_TRUE(std::get<DynamicArray<int>>(descriptors[0]).empty());
      int const softId = std::get<int>(descriptors[1]);
      if (softId < 0) {
        stats.contactsWithoutSoftDofs += isize(result.sampleIndices);
        EXPECT_EQ(result.sampleIndices.size(), result.forcePerUnitArea.size());
        for (Real3 const& force : result.forcePerUnitArea) {
          stats.maxForceWithoutSoftDofs = Max(stats.maxForceWithoutSoftDofs, Norm(force));
        }
      } else {
        EXPECT_EQ(0, softId);
        stats.contactsWithSoftDofs += isize(result.sampleIndices);
      }
    }
    return stats;
  }

  // Steps once with only zero-DoF contact, then once with mixed contact, which must set up the
  // skinned contact SNLE for the first time.
  void ExpectSnleSetUpAfterZeroDofStep(bool registerForceQuery) {
    auto [skin, box] = CreateActors(/*isBoxStatic=*/true, kZeroDofBoxMaxX);
    entt::entity const skinEntity = GetEntity(skin->GetHandle());
    entt::entity const boxEntity = GetEntity(box->GetHandle());
    if (registerForceQuery) {
      skin->RegisterQuery(QueryType::TotalContactForce, test::ExpectOK{});
    }

    Step();
    auto const [zeroDofContacts, nonzeroDofContacts] =
        CountContactsByDofType</*kIsSync*/ false>(skinEntity, boxEntity);
    EXPECT_GT(zeroDofContacts, 0);
    EXPECT_EQ(0, nonzeroDofContacts);
    EXPECT_FALSE(UsesSkinnedContactInSolver(skinEntity));
    EXPECT_TRUE(GetRegistry().get<CSkinnedContactSnle const>(skinEntity).residuals.empty());

    ShiftBoxX(box, kMixedBoxMaxX - kZeroDofBoxMaxX);
    real const poseBefore = GetPrismaticPose(skinEntity);
    Step();
    EXPECT_GT(CountContactsByDofType</*kIsSync*/ false>(skinEntity, boxEntity).second, 0);
    EXPECT_TRUE(UsesSkinnedContactInSolver(skinEntity));
    EXPECT_NE(poseBefore, GetPrismaticPose(skinEntity));
  }
};

TEST_F(ZeroDofSkinContact, StaticBoxReportsForceFromAsyncContact) {
  auto [skin, box] = CreateActors(/*isBoxStatic=*/true);
  entt::entity const skinEntity = GetEntity(skin->GetHandle());
  entt::entity const boxEntity = GetEntity(box->GetHandle());
  ExpectEmptyAndNonemptyPartitions(skinEntity);

  skin->RegisterQuery(QueryType::TotalContactForce, test::ExpectOK{});
  skin->RegisterQuery(QueryType::ContactPoints, test::ExpectOK{});
  Step();

  auto const [zeroDofContacts, nonzeroDofContacts] =
      CountContactsByDofType</*kIsSync*/ false>(skinEntity, boxEntity);
  EXPECT_GT(zeroDofContacts, 0);
  EXPECT_EQ(0, nonzeroDofContacts);
  EXPECT_EQ((std::pair{0, 0}), CountContactsByDofType</*kIsSync*/ true>(skinEntity, boxEntity));
  EXPECT_FALSE(GetRegistry().get<CSkinnedContactSnle const>(skinEntity).useInSolver);
  EXPECT_FALSE(skin->GetContactPointsWorld(test::ExpectOK{}).empty());
  EXPECT_GT(Norm(skin->GetContactForceWorld(test::ExpectOK{})), 0_r);
}

TEST_F(ZeroDofSkinContact, DynamicBoxMovesFromSyncContact) {
  auto [skin, box] = CreateActors(/*isBoxStatic=*/false);
  entt::entity const skinEntity = GetEntity(skin->GetHandle());
  entt::entity const boxEntity = GetEntity(box->GetHandle());
  ExpectEmptyAndNonemptyPartitions(skinEntity);

  skin->RegisterQuery(QueryType::TotalContactForce, test::ExpectOK{});
  box->RegisterQuery(QueryType::TotalContactForce, test::ExpectOK{});
  real const initialBoxX = box->GetRootTransform().GetTranslation()[0];
  Step();

  auto const [zeroDofContacts, nonzeroDofContacts] =
      CountContactsByDofType</*kIsSync*/ true>(skinEntity, boxEntity);
  EXPECT_GT(zeroDofContacts, 0);
  EXPECT_EQ(0, nonzeroDofContacts);
  EXPECT_EQ((std::pair{0, 0}), CountContactsByDofType</*kIsSync*/ false>(skinEntity, boxEntity));
  EXPECT_LT(box->GetRootTransform().GetTranslation()[0], initialBoxX);
  Real3 const skinForce = skin->GetContactForceWorld(test::ExpectOK{});
  Real3 const boxForce = box->GetContactForceWorld(test::ExpectOK{});
  real const forceScale = Max(Norm(skinForce), Norm(boxForce));
  ASSERT_GT(forceScale, 0_r);
  EXPECT_NEAR(Norm(skinForce + boxForce), 0_r, 1e-5_r * forceScale);
}

TEST_F(ZeroDofSkinContact, MixedPartitionsPreserveAsyncSolverContactAndQueries) {
  auto [skin, box] = CreateActors(/*isBoxStatic=*/true, kMixedBoxMaxX);
  entt::entity const skinEntity = GetEntity(skin->GetHandle());
  entt::entity const boxEntity = GetEntity(box->GetHandle());
  ExpectEmptyAndNonemptyPartitions(skinEntity);

  skin->RegisterQuery(QueryType::TotalContactForce, test::ExpectOK{});
  real const initialPose =
      GetRegistry().get<CArticulatedReducedPose<TimeStep::Current> const>(skinEntity).value[0];
  Step();

  auto const [zeroDofContacts, nonzeroDofContacts] =
      CountContactsByDofType</*kIsSync*/ false>(skinEntity, boxEntity);
  EXPECT_GT(zeroDofContacts, 0);
  EXPECT_GT(nonzeroDofContacts, 0);
  EXPECT_TRUE(GetRegistry().get<CSkinnedContactSnle const>(skinEntity).useInSolver);
  EXPECT_NE(
      initialPose,
      GetRegistry().get<CArticulatedReducedPose<TimeStep::Current> const>(skinEntity).value[0]);

  auto const& collisions =
      GetRegistry().get<CActiveCollisions</*kIsSync*/ false, TimeStep::Current> const>(skinEntity);
  auto const& partitions = GetRegistry().get<CContactPartitions const>(skinEntity);
  real maxZeroDofForce = 0_r;
  for (auto const& collision : collisions) {
    auto const& result = collision.collisionResult;
    if (result.Empty()) {
      continue;
    }
    auto const& boneIds =
        std::get<DynamicArray<int>>(partitions[result.collidingPartitionId].GetDofDescriptors()[0]);
    if (boneIds.empty()) {
      EXPECT_EQ(result.sampleIndices.size(), result.forcePerUnitArea.size());
      for (Real3 const& force : result.forcePerUnitArea) {
        maxZeroDofForce = Max(maxZeroDofForce, Norm(force));
      }
    }
  }
  EXPECT_GT(maxZeroDofForce, 0_r);
}

TEST_F(ZeroDofSkinContact, BlendedContactSkipsOnlyPartitionsWithoutSoftDofs) {
  auto [skin, box] = CreateBlendedActors(kMixedBoxMaxX);
  entt::entity const skinEntity = GetEntity(skin->GetHandle());
  entt::entity const boxEntity = GetEntity(box->GetHandle());
  skin->RegisterQuery(QueryType::TotalContactForce, test::ExpectOK{});
  Step();

  BlendedContactStats const stats = GetBlendedContactStats(skinEntity, boxEntity);
  EXPECT_GT(stats.contactsWithoutSoftDofs, 0);
  EXPECT_GT(stats.contactsWithSoftDofs, 0);
  EXPECT_GT(stats.maxForceWithoutSoftDofs, 0_r);
  EXPECT_TRUE(UsesSkinnedContactInSolver(skinEntity));
}

TEST_F(ZeroDofSkinContact, BlendedContactWithoutSoftDofsReportsForceOutsideSolver) {
  auto [skin, box] = CreateBlendedActors(kZeroDofBoxMaxX);
  entt::entity const skinEntity = GetEntity(skin->GetHandle());
  entt::entity const boxEntity = GetEntity(box->GetHandle());
  skin->RegisterQuery(QueryType::TotalContactForce, test::ExpectOK{});
  Step();

  BlendedContactStats const stats = GetBlendedContactStats(skinEntity, boxEntity);
  EXPECT_GT(stats.contactsWithoutSoftDofs, 0);
  EXPECT_EQ(0, stats.contactsWithSoftDofs);
  EXPECT_GT(stats.maxForceWithoutSoftDofs, 0_r);
  EXPECT_FALSE(UsesSkinnedContactInSolver(skinEntity));
  EXPECT_GT(Norm(skin->GetContactForceWorld(test::ExpectOK{})), 0_r);
}

TEST_F(ZeroDofSkinContact, StaticBoxWithoutQuerySkipsAsyncContact) {
  auto [skin, box] = CreateActors(/*isBoxStatic=*/true);
  entt::entity const skinEntity = GetEntity(skin->GetHandle());
  entt::entity const boxEntity = GetEntity(box->GetHandle());
  Step();

  auto const [zeroDofContacts, nonzeroDofContacts] =
      CountContactsByDofType</*kIsSync*/ false>(skinEntity, boxEntity);
  EXPECT_GT(zeroDofContacts, 0);
  EXPECT_EQ(0, nonzeroDofContacts);
  EXPECT_FALSE(UsesSkinnedContactInSolver(skinEntity));
}

TEST_F(ZeroDofSkinContact, MixedPartitionsWithoutQueryAssembleDofContacts) {
  auto [skin, box] = CreateActors(/*isBoxStatic=*/true, kMixedBoxMaxX);
  entt::entity const skinEntity = GetEntity(skin->GetHandle());
  entt::entity const boxEntity = GetEntity(box->GetHandle());
  real const initialPose = GetPrismaticPose(skinEntity);
  Step();

  auto const [zeroDofContacts, nonzeroDofContacts] =
      CountContactsByDofType</*kIsSync*/ false>(skinEntity, boxEntity);
  EXPECT_GT(zeroDofContacts, 0);
  EXPECT_GT(nonzeroDofContacts, 0);
  EXPECT_TRUE(UsesSkinnedContactInSolver(skinEntity));
  EXPECT_NE(initialPose, GetPrismaticPose(skinEntity));
}

TEST_F(ZeroDofSkinContact, SolverUseStopsWhenOnlyZeroDofContactRemains) {
  auto [skin, box] = CreateActors(/*isBoxStatic=*/true, kMixedBoxMaxX);
  entt::entity const skinEntity = GetEntity(skin->GetHandle());
  entt::entity const boxEntity = GetEntity(box->GetHandle());

  Step();
  EXPECT_GT(CountContactsByDofType</*kIsSync*/ false>(skinEntity, boxEntity).second, 0);
  EXPECT_TRUE(UsesSkinnedContactInSolver(skinEntity));

  ShiftBoxX(box, kZeroDofBoxMaxX - kMixedBoxMaxX);
  Step();
  auto const [zeroDofContacts, nonzeroDofContacts] =
      CountContactsByDofType</*kIsSync*/ false>(skinEntity, boxEntity);
  EXPECT_GT(zeroDofContacts, 0);
  EXPECT_EQ(0, nonzeroDofContacts);
  EXPECT_FALSE(UsesSkinnedContactInSolver(skinEntity));
}

TEST_F(ZeroDofSkinContact, SnleSetUpAfterZeroDofStepWithoutQuery) {
  ExpectSnleSetUpAfterZeroDofStep(/*registerForceQuery=*/false);
}

TEST_F(ZeroDofSkinContact, SnleSetUpAfterZeroDofStepWithQuery) {
  ExpectSnleSetUpAfterZeroDofStep(/*registerForceQuery=*/true);
}

// Verify consistency of contact torque queries on a rigid cube constrained at its center of mass,
// contacted by another orbiting rigid cube. Checks:
//   1) Body torque from GetContactTorqueWorld matches body torque summed from GetContactPointsWorld
//   2) dL/dt (finite difference of angular momentum) matches contact torque
class ContactTorqueScene : public test::MochiSceneTestBase {
 protected:
  static constexpr real kCubeSize = 0.1_r;
  static constexpr Real3 kTargetPosition = {0.3_r, 0.2_r, -0.1_r};
  static constexpr real kOrbitRadius = kCubeSize;

  Actor* _targetActor = nullptr;
  Constraint* _orbiterConstraint = nullptr;

  void SetUp() override {
    test::MochiSceneTestBase::SetUp();
    _scene->SetGravity(Real3{});

    auto&& [coords, connectivity] =
        test::CreateMinimalTetMeshUnitCube(Real3{kCubeSize, kCubeSize, kCubeSize});
    auto cubeShape = _mochiContext->CreateTetMeshShape(
        Flatten(MakeSpan(coords)), Flatten(MakeSpan(connectivity)), test::ExpectOK{});

    // Target cube: constrained at center of mass
    {
      RigidActorParams params;
      params.name = "Target";
      params.shape = cubeShape;
      params.worldFromLocal.SetTranslation(kTargetPosition);
      params.colliderType = ColliderType::Box;
      // Disable normal damping (nonzero by default) to preserve the calibrated contact dynamics.
      params.contact.normalViscousDampingCoefficient = 0_r;
      _targetActor = _scene->CreateRigidActor(params, test::ExpectOK{});

      Real3 const com = _targetActor->GetCenterOfMassTransform(test::ExpectOK{}).GetTranslation();
      std::vector<int> bcDofIndices = {0, 1, 2};
      std::vector<real> bcDofValues = {com[0], com[1], com[2]};
      _targetActor->AddBoundaryConditionDofsWorld(bcDofIndices, bcDofValues, test::ExpectOK{});

      _targetActor->RegisterQuery(QueryType::TotalContactForce, test::ExpectOK{});
      _targetActor->RegisterQuery(QueryType::ContactPoints, test::ExpectOK{});
    }

    // Orbiter cube: soft-constrained via pivot constraint
    {
      Real3 const initialPos = kTargetPosition + Real3{kOrbitRadius, 0_r, 0_r};

      RigidActorParams params;
      params.name = "Orbiter";
      params.shape = cubeShape;
      params.worldFromLocal.SetTranslation(initialPos);
      params.colliderType = ColliderType::Box;
      // Disable normal damping (nonzero by default) to preserve the calibrated contact dynamics.
      params.contact.normalViscousDampingCoefficient = 0_r;
      auto* orbiter = _scene->CreateRigidActor(params, test::ExpectOK{});

      RigidPivotPositionConstraintParams conParams;
      conParams.localPosition = {0_r, 0_r, 0_r};
      conParams.targetPosition = initialPos;
      conParams.actor = orbiter->GetHandle();
      conParams.stiffness = 1e2_r;
      conParams.damping = 1e2_r;
      _orbiterConstraint = _scene->CreateRigidPivotPositionConstraint(conParams, test::ExpectOK{});
    }
  }
};

TEST_F(ContactTorqueScene, Consistency) {
  // NOTE: Time step size has a critical influence on the momentum difference error, because the
  // error is dominated by the integration method. Reducing time step by 10x reduces the error
  // roughly by 10x too. The inertia model (default merit-based vs. Newton-Euler) has no impact.
  real constexpr kDt = 1e-2_r;
  int constexpr kTotalSteps = 200;

  Real3 angularMomentumPrev = {};
  int numStepsWithContact = 0;
  real momentumDifferenceErrorRms = 0;
  real torqueRms = 0;

  // Constant body inertia
  Real6 const m = _targetActor->GetRigidMomentOfInertiaLocal(test::ExpectOK{});
  VMatrix3x3r const inertiaBody = {
      Vec4r{m[0], m[1], m[2]}, Vec4r{m[1], m[3], m[4]}, Vec4r{m[2], m[4], m[5]}};

  // Step the simulation and verify torque consistency whenever contact occurs
  for (int step = 0; step < kTotalSteps; ++step) {
    real const t = static_cast<real>(_scene->GetTotalSimulationTime());

    // Set new orbit position on a pseudo-random spherical trajectory
    real const theta = 0.5_r * (kPI + t) + 0.3_r * Sin(1.7_r * t);
    real const phi = 0.7_r * t + 0.4_r * Sin(2.3_r * t) + 0.2_r * Sin(3.7_r * t);
    Real3 const orbitPosition = kTargetPosition +
        kOrbitRadius * Real3{Sin(theta) * Cos(phi), Sin(theta) * Sin(phi), Cos(theta)};
    _orbiterConstraint->SetTargetPosition(orbitPosition, test::ExpectOK{});

    // Step the scene
    _scene->Step(kDt);

    // Get queries. Skip if there's no contact
    auto const points = _targetActor->GetContactPointsWorld(test::ExpectOK{});
    if (points.empty()) {
      continue;
    }
    Real3 const torque = _targetActor->GetContactTorqueWorld(test::ExpectOK{});

    numStepsWithContact++;

    // Compute torque from contact points
    Real3 const com = _targetActor->GetCenterOfMassTransform(test::ExpectOK{}).GetTranslation();
    Real3 torqueFromContacts = {};
    for (auto const& cp : points) {
      Real3 const force = (cp.actorA == _targetActor->GetHandle()) ? cp.force : -cp.force;
      Real3 const r = cp.posA - com;
      torqueFromContacts += Cross(r, force);
    }

    // Test relative error
    real const torqueNormSqr = NormSqr(torqueFromContacts);
    real const torqueError = Norm(torque - torqueFromContacts) / Sqrt(torqueNormSqr);
    EXPECT_LE(torqueError, 1e-5_r); // Torque query error should be tiny

    // Compute angular momentum
    Real3 const omega = _targetActor->GetAngularVelocity(test::ExpectOK{});
    Quaternion const q = _targetActor->GetRootTransform().GetRotation();
    VMatrix3x3r const inertia = RotateInertia(inertiaBody, q);
    Real3 const angularMomentum = ToReal3(DotMatVec3x3(inertia, ToSimd(omega)));

    // Estimate torque from angular momentum difference
    Real3 const torqueFromAngularMomentum = (angularMomentum - angularMomentumPrev) / kDt;

    // Accumulate torque and error for RMS evaluation
    torqueRms += torqueNormSqr;
    momentumDifferenceErrorRms += NormSqr(torqueFromAngularMomentum - torqueFromContacts);

    // Shift values
    angularMomentumPrev = angularMomentum;
  }

  // Test aggregate values
  EXPECT_GE(numStepsWithContact, kTotalSteps / 10); // Expect at least 10% of the steps in contact
  real const momentumDifferenceError = Sqrt(momentumDifferenceErrorRms / torqueRms);
  EXPECT_LE(
      momentumDifferenceError, 0.1_r); // Momentum difference should be same order of magnitude
}
