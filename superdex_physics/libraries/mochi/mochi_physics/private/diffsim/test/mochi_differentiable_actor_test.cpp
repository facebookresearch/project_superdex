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

#include <mochi_core/articulated_body/articulated_body.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/rand_utils.h>
#include <mochi_physics/diffsim/mochi_diffsim.h>
#include <mochi_physics/mochi_physics.h>
#include <mochi_physics/src/mochi_articulated_body.h>
#include <mochi_physics/src/mochi_rigid.h>

#include <gtest/gtest.h>

#include <array>
#include <vector>

// These tests exercise the differentiable-simulation gradient-conversion utilities declared in
// <mochi_physics/diffsim/mochi_diffsim.h>. They live in the dedicated differentiable test
// executable so the main physics test target does not depend on the access-restricted diffsim
// header.

using namespace mochi;

// Test that diffsim::ConvertRigidGradientLieToRotationVector correctly transports a
// Lie-parameterized gradient to a full rotation-vector-parameterized gradient. We compare: (a)
// Gradient computed using regular finite differences on rotation vector (b) Gradient computed using
// Lie finite differences and then transported
// Test that diffsim::ConvertRigidGradientQuaternionToLie correctly transports a
// quaternion-parameterized gradient to a Lie-parameterized gradient. We compare: (a) Gradient
// computed using Lie finite differences (b) Gradient computed using quaternion finite differences
// and then transported
TEST(MochiRigid, ConvertRigidGradient) {
  // Disable warnings on this test because it uses non-unit quaternions
  bool const wasWarningEnabled = IsLogChannelEnabled(LogChannel::Warning);
  EnableLogChannel(LogChannel::Warning, false);

  // Set up a test state with non-trivial rotation
  Real3 const translation = {1.5_r, -0.7_r, 2.3_r};
  Real3 const rotVec = {0.3_r, -0.5_r, 0.2_r}; // Rotation vector parameterization
  Quaternion const rotation = Quaternion::FromRotationVector(rotVec);
  TransformRT state(rotation, translation);

  real constexpr kEps = 1e-3_r;
  real constexpr kOneOverTwoEps = 0.5_r / kEps;

  enum class RotCase { Lie = 0, Quat = 1, RotVec = 2 };

  // Compute gradients using finite differences
  ColumnVector<real, RigidSize::kDAll> gradRotVecFD;
  ColumnVector<real, RigidSize::kDAll> gradLieFD;
  ColumnVector<real, RigidSize::kAll> gradQuatFD;
  for (int i = 0; i < RigidSize::kAll; ++i) {
    auto evalMerit = [&](real eps, RotCase rotCase) {
      TransformRT stateNew = state;
      if (i < RigidSize::kDTrans) {
        Real3 deltaVec{};
        deltaVec[i] = eps;
        stateNew.SetTranslation(state.GetTranslation() + deltaVec);
      } else if (rotCase == RotCase::Quat) {
        // Perturb quaternion
        Real4 deltaVec{};
        deltaVec[i - 3] = eps;
        stateNew.SetRotation(Quaternion{rotation.ToReal4() + deltaVec});
      } else if (rotCase == RotCase::Lie) {
        // Perturb rotation in Lie manner: exp(delta) * R
        Real3 deltaVec{};
        deltaVec[i - 3] = eps;
        stateNew.SetRotation(Quaternion::FromRotationVector(deltaVec) * rotation);
      } else {
        // Perturb the rotation vector directly
        Real3 deltaVec{};
        deltaVec[i - 3] = eps;
        stateNew.SetRotation(Quaternion::FromRotationVector(rotVec + deltaVec));
      }
      // Merit function: L2 norm of rigid state = ||translation||^2 + ||rotVec||^2
      Real3 t = stateNew.GetTranslation();
      Real3 r = stateNew.GetRotation().ToRotationVector();
      return Dot(t, t) + Dot(r, r);
    };

    if (i < RigidSize::kDAll) {
      gradRotVecFD(i) =
          (evalMerit(kEps, RotCase::RotVec) - evalMerit(-kEps, RotCase::RotVec)) * kOneOverTwoEps;
      gradLieFD(i) =
          (evalMerit(kEps, RotCase::Lie) - evalMerit(-kEps, RotCase::Lie)) * kOneOverTwoEps;
    }
    gradQuatFD(i) =
        (evalMerit(kEps, RotCase::Quat) - evalMerit(-kEps, RotCase::Quat)) * kOneOverTwoEps;
  }

  // Transport the Lie gradient to the full rotation vector parameterization
  ColumnVector<real, RigidSize::kDAll> gradLieTransported = gradLieFD.Duplicate();
  diffsim::ConvertRigidGradientLieToRotationVector(
      state, MakeSpan(gradLieTransported), test::ExpectOK{});

  // Compare the gradients
  real normMax = Max(gradRotVecFD.Norm(), gradLieTransported.Norm());
  real normDiff = ColumnVector<real>(gradRotVecFD - gradLieTransported).Norm();
  EXPECT_NEAR(normDiff / normMax, 0_r, 2e-4_r);

  // Convert the quaternion gradient to the Lie parameterization
  ColumnVector<real, RigidSize::kDAll> gradQuatTransported;
  diffsim::ConvertRigidGradientQuaternionToLie(
      state, MakeConstSpan(gradQuatFD), MakeSpan(gradQuatTransported), test::ExpectOK{});

  // Compare the gradients
  normMax = Max(gradLieFD.Norm(), gradQuatTransported.Norm());
  normDiff = ColumnVector<real>(gradLieFD - gradQuatTransported).Norm();
  EXPECT_NEAR(normDiff / normMax, 0_r, 2e-4_r);

  // Convert back to quaternion gradient and validate round trip
  ColumnVector<real, RigidSize::kAll> gradQuatRoundTrip;
  diffsim::ConvertRigidGradientLieToQuaternion(
      state, MakeConstSpan(gradQuatTransported), MakeSpan(gradQuatRoundTrip), test::ExpectOK{});

  // Compare the gradients
  normMax = Max(gradQuatRoundTrip.Norm(), gradQuatFD.Norm());
  normDiff = ColumnVector<real>(gradQuatRoundTrip - gradQuatFD).Norm();
  EXPECT_NEAR(normDiff / normMax, 0_r, 2e-4_r);

  // Re-enable warnings if necessary
  EnableLogChannel(LogChannel::Warning, wasWarningEnabled);
}

static void Compare(MatrixView<real const> a, MatrixView<real const> b, real tol = 1e-2_r) {
  real normMax = std::max(a.Norm(), b.Norm());
  if (normMax > 1e-6_r) {
    real normDiff = Matrix<real>(a - b).Norm();
    EXPECT_NEAR(normDiff / normMax, 0_r, tol);
  }
}

namespace {
std::array<std::vector<ArticulatedJointType>, 6> const kBodies = {
    {{ArticulatedJointType::Spherical},
     {ArticulatedJointType::Free, ArticulatedJointType::Revolute},
     {ArticulatedJointType::Revolute, ArticulatedJointType::Spherical},
     {ArticulatedJointType::Free, ArticulatedJointType::Spherical, ArticulatedJointType::Revolute},
     {ArticulatedJointType::Free, ArticulatedJointType::Prismatic, ArticulatedJointType::Prismatic},
     {ArticulatedJointType::Free,
      ArticulatedJointType::Spherical,
      ArticulatedJointType::Revolute,
      ArticulatedJointType::Spherical}}};
}

// Test fixture replicating the minimal articulated-actor setup needed to exercise the
// gradient-conversion utilities.
class ConvertArticulatedGradientTest : public test::MochiSceneTestBase {
 protected:
  ShapeHandle _cubeShape;

  void SetUp() override {
    test::MochiSceneTestBase::SetUp();
    auto [coordinates, connectivity] = test::CreateMinimalTetMeshUnitCube();
    _cubeShape = _mochiContext->CreateTetMeshShape(
        Flatten(MakeSpan(coordinates)), Flatten(MakeSpan(connectivity)), test::ExpectOK{});
  }

  Actor* CreateActor(Span<ArticulatedJointType const> joints) {
    auto const numBones = isize(joints);
    TransformRT const offset{Real3{1_r, 1_r, 1_r}};

    ArticulatedActorParams params;
    params.joints.resize(numBones);
    params.links.resize(numBones);
    for (int i = 0; i < numBones; ++i) {
      auto& joint = params.joints[i];
      joint.type = joints[i];
      joint.axis = Real3{1_r, 0_r, 0_r};
      // The first link is offset from the root; subsequent links are co-located.
      if (i == 0) {
        joint.parentLinkFromJoint = offset;
      }

      auto& link = params.links[i];
      link.parentLink = i - 1;
      link.shape = _cubeShape;
      link.layer = "Articulated";
      link.colliderType = ColliderType::Box;
    }
    return _scene->CreateArticulatedActor(params, test::ExpectOK{});
  }

  void TestConvertGradient(Span<ArticulatedJointType const> joints) {
    // Get joint infos from the actor
    Actor* actor = CreateActor(joints);
    auto const numDofs = actor->GetNumDofs();

    // Generate a pseudo-random pose
    auto rng = RandomGenerator(42);
    std::vector<real> pose(numDofs);
    SetRandom(rng, -0.5_r, 0.5_r, MakeSpan(pose));

    real constexpr kEps = 1e-3_r;
    real constexpr kOneOverTwoEps = 0.5_r / kEps;

    // Compute gradients using finite differences
    ColumnVector<real> gradRegularFD(numDofs);
    ColumnVector<real> gradLieFD(numDofs);
    std::vector<real> poseNew(numDofs);
    std::vector<real> delta(numDofs, 0_r);
    for (int i = 0; i < numDofs; ++i) {
      auto evalMerit = [&](real eps, bool useLieFD) {
        if (useLieFD) {
          // Perturb pose in Lie manner
          delta[i] = eps;
          actor->AddArticulatedDeltaToPose(pose, delta, poseNew, test::ExpectOK{});
          delta[i] = 0_r;
        } else {
          // Perturb the pose directly
          poseNew = pose;
          poseNew[i] += eps;
        }
        // Merit function: L2 norm of pose
        return AsConstView(poseNew).NormSqr();
      };

      gradRegularFD(i) = (evalMerit(kEps, false) - evalMerit(-kEps, false)) * kOneOverTwoEps;
      gradLieFD(i) = (evalMerit(kEps, true) - evalMerit(-kEps, true)) * kOneOverTwoEps;
    }

    // Transport the Lie gradients
    ColumnVector<real> gradLieFDTransported = gradLieFD.Duplicate();
    diffsim::ConvertArticulatedGradientLieToRotationVector(
        actor, pose, MakeSpan(gradLieFDTransported), test::ExpectOK{});
    ColumnVector<real> gradRegularFDTransported = gradRegularFD.Duplicate();
    diffsim::ConvertArticulatedGradientRotationVectorToLie(
        actor, pose, MakeSpan(gradRegularFDTransported), test::ExpectOK{});

    // Compare the gradients
    Compare(gradRegularFD, gradLieFDTransported, 2e-4_r);
    Compare(gradLieFD, gradRegularFDTransported, 2e-4_r);
  }
};

// Tests that diffsim::ConvertArticulatedGradientLieToRotationVector correctly transports a
// Lie-parameterized gradient to a full rotation-vector-parameterized gradient. We compare:
// (a) Gradient computed using regular finite differences on pose
// (b) Gradient computed using Lie finite differences and then transported
// Test that diffsim::ConvertArticulatedGradientRotationVectorToLie correctly transports a
// full rotation-vector-parameterized gradient to a Lie-parameterized gradient. We compare:
// (a) Gradient computed using Lie finite differences on pose
// (b) Gradient computed using regular finite differences and then transported
TEST_F(ConvertArticulatedGradientTest, ConvertArticulatedGradient) {
  for (auto const& body : kBodies) {
    TestConvertGradient(body);
  }
}
