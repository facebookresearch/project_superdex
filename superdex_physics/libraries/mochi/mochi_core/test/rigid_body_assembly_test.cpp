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

#include <mochi_core/test/mochi_test_helpers.h>

#include <mochi_core/utils/matrix_utils.h>
#include <mochi_core/utils/nd_array_utils.h>
#include <mochi_core/utils/rigid_body_assembly.h>

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <limits>
#include <utility>

using namespace mochi;

namespace {
int constexpr kNumDofs = RigidSize::kDAll;
std::array<real, kNumDofs> kPoseValsRef = {1.2_r, 0.3_r, -0.6_r, 0.5_r, -0.3_r, 0.4_r};
} // namespace

static void SetPose(Span<real const> poseVals, TransformRT& outPose) {
  outPose.SetTranslation(Unflatten<Real3 const>(poseVals)[0]);
  outPose.SetRotation(Quaternion::FromRotationVector(Unflatten<Real3 const>(poseVals)[1]));
}

static void AddToPose(int coordi, real vali, int coordj, real valj, TransformRT& outPose) {
  MOCHI_ASSERT(coordi >= 0 && coordi < kNumDofs);
  MOCHI_ASSERT(coordj >= 0 && coordj < kNumDofs);
  MOCHI_ASSERT(coordi != coordj || valj == 0_r);
  Real3 deltaPos{};
  Real3 deltaRot{};
  auto add = [&](int coord, real val) {
    if (coord < 3) {
      deltaPos[coord] += val;
    } else {
      deltaRot[coord - 3] += val;
    }
  };
  add(coordi, vali);
  add(coordj, valj);
  if (deltaPos != Real3{}) {
    outPose.SetTranslation(outPose.GetTranslation() + deltaPos);
  }
  if (deltaRot != Real3{}) {
    outPose.SetRotation(Quaternion::FromRotationVector(deltaRot) * outPose.GetRotation());
  }
}

static void AddToPose(int coordi, real vali, TransformRT& outPose) {
  AddToPose(coordi, vali, 0, 0_r, outPose);
}

static void Compare(Matrix<real> m1, Matrix<real> m2, real tol) {
  real normDiff = Matrix<real>(m1 - m2).Norm();
  real maxNorm = std::max(m1.Norm(), m2.Norm());
  EXPECT_NEAR(normDiff / maxNorm, 0_r, tol);
}

static void TestConsistency(
    std::function<void(TransformRT const&, double*, RigidGradient*)> assembler,
    real tolGrad) {
  real constexpr kEps = 1e-2_r;
  real constexpr kOneOverTwoEps = 1_r / (2_r * kEps);

  // Initialize test pose
  TransformRT pose;
  SetPose(kPoseValsRef, pose);

  // Evaluate analytical values
  double energy{};
  RigidGradient gradient{};
  assembler(pose, &energy, &gradient);
  auto gradientBase = gradient;

  // Compute test values using finite differences
  auto gradientTest = gradientBase;
  for (int i = 0; i < kNumDofs; ++i) {
    auto evalDelta = [&](real deltai) -> real {
      SetPose(kPoseValsRef, pose);
      AddToPose(i, deltai, i, 0_r, pose);
      energy = {};
      assembler(pose, &energy, nullptr);
      return energy;
    };

    // Evaluate gradient term and Hessian diagonal term
    auto energyFwd = evalDelta(kEps);
    auto energyBwd = evalDelta(-kEps);
    gradientTest[i] = kOneOverTwoEps * (energyFwd - energyBwd);
  }

  // Compare analytical and test values
  Compare(AsConstView(gradientBase), AsConstView(gradientTest), tolGrad);
}

static void TestConsistency(
    std::function<void(TransformRT const&, double*, RigidGradient*, RigidHessian*)> assembler,
    real tolGrad,
    real tolHess) {
  real constexpr kEps = 1e-2_r;
  real constexpr kOneOverTwoEps = 1_r / (2_r * kEps);
  real constexpr kOneOverEpsSqr = 1_r / (kEps * kEps);
  real constexpr kOneOverFourEpsSqr = kOneOverEpsSqr / 4_r;

  // Initialize test pose
  TransformRT pose;
  SetPose(kPoseValsRef, pose);

  // Evaluate analytical values
  double energy{};
  RigidGradient gradient{};
  RigidHessian hessian{};
  assembler(pose, &energy, &gradient, &hessian);
  auto energyBase = energy;
  auto gradientBase = gradient;
  auto hessianBase = hessian;

  // Compute test values using finite differences
  auto gradientTest = gradientBase;
  auto hessianTest = hessianBase;
  for (int i = 0; i < kNumDofs; ++i) {
    for (int j = 0; j < kNumDofs; ++j) {
      auto evalDelta = [&](real deltai, real deltaj) -> real {
        SetPose(kPoseValsRef, pose);
        AddToPose(i, deltai, j, deltaj, pose);
        energy = {};
        assembler(pose, &energy, nullptr, nullptr);
        return energy;
      };

      if (i == j) {
        // Evaluate gradient term and Hessian diagonal term
        auto energyFwd = evalDelta(kEps, 0_r);
        auto energyBwd = evalDelta(-kEps, 0_r);
        gradientTest[i] = kOneOverTwoEps * (energyFwd - energyBwd);
        hessianTest[i][i] = kOneOverEpsSqr * (energyFwd - 2_r * energyBase + energyBwd);
      } else {
        // Evaluate Hessian off-diagonal term
        auto energyFwdFwd = evalDelta(kEps, kEps);
        auto energyFwdBwd = evalDelta(kEps, -kEps);
        auto energyBwdFwd = evalDelta(-kEps, kEps);
        auto energyBwdBwd = evalDelta(-kEps, -kEps);
        auto val = kOneOverFourEpsSqr * (energyFwdFwd - energyFwdBwd - energyBwdFwd + energyBwdBwd);
        hessianTest[i][j] = val;
      }
    }
  }

  // Compare analytical and test values
  Compare(AsConstView(gradientBase), AsConstView(gradientTest), tolGrad);
  Compare(
      AsConstView<real, kNumDofs, kNumDofs>(hessianBase),
      AsConstView<real, kNumDofs, kNumDofs>(hessianTest),
      tolHess);
}

static void TestConsistencyMixed(
    std::function<void(TransformRT const&, RigidGradient*, RigidHessian*)> assembler,
    real tolHess) {
  real constexpr kEps = 1e-2_r;
  real constexpr kOneOverTwoEps = 1_r / (2_r * kEps);

  // Initialize test pose
  TransformRT pose;
  SetPose(kPoseValsRef, pose);

  // Evaluate analytical values
  RigidGradient gradient{};
  RigidHessian hessianMixed{};
  assembler(pose, &gradient, &hessianMixed);

  // Compute test values using finite differences
  RigidHessian hessianMixedTest{};
  for (int i = 0; i < kNumDofs; ++i) {
    for (int j = 0; j < kNumDofs; j++) {
      SetPose(kPoseValsRef, pose);
      AddToPose(j, kEps, pose);
      RigidGradient gradientTestP{};
      assembler(pose, &gradientTestP, nullptr);
      SetPose(kPoseValsRef, pose);
      AddToPose(j, -kEps, pose);
      RigidGradient gradientTestM{};
      assembler(pose, &gradientTestM, nullptr);
      auto val = (gradientTestP[i] - gradientTestM[i]) * kOneOverTwoEps;
      hessianMixedTest[i][j] = val;
    }
  }
  Compare(
      AsConstView<real, kNumDofs, kNumDofs>(hessianMixed),
      AsConstView<real, kNumDofs, kNumDofs>(hessianMixedTest),
      tolHess);
}

TEST(RigidBodyAssembly, RigidBodyConstantForces) {
  // Initialize other data needed by the assembler
  TransformRT stageStartPos;
  SetPose(
      kPoseValsRef, stageStartPos); // Must be the same as the test pose for the consistency test.
  RigidBodyInertia const inertia(
      Vec4r{},
      1_r,
      VMatrix3x3r{Vec4r{1_r, 0_r, 0_r}, Vec4r{0_r, 1_r, 0_r}, Vec4r{0_r, 0_r, 1_r}},
      2_r);
  std::array<int, kNumDofs> dofs = {0, 1, 2, 3, 4, 5};
  Span<int const> dofsSpan = MakeConstSpan(dofs);
  std::array<real, kNumDofs> forces = {-0.7_r, 1.3_r, 0.5_r, 1.0_r, -0.6_r, -0.4_r};
  Span<real const> forcesSpan = MakeConstSpan(forces);

  // Create assembly lambda and test. Allow 100% error on the Hessian.
  auto assembler = [&](TransformRT const& pose,
                       double* energy,
                       RigidGradient* gradient,
                       RigidHessian* /* hessian */) {
    AddRigidBodyGravity(
        pose, stageStartPos, inertia, Real3{0.3_r, -9.8_r, 1.7_r}, energy, gradient);
    AddRigidBodyExternalForces(pose, stageStartPos, dofsSpan, forcesSpan, energy, gradient);
  };
  TestConsistency(assembler, 1e-3_r, 1_r);

  // The energies are relative to the stage start, up to rounding in the torque term.
  double energy = 0.0;
  assembler(stageStartPos, &energy, nullptr, nullptr);
  EXPECT_NEAR(0.0, energy, 1e-6);
}

TEST(MultiBody, RigidBodyInertiaFromMerit) {
  // Initialize data needed by the assembler
  real mass = 1.3_r;
  VMatrix3x3r rot = Rodrigues(Vec4r{-0.4_r, 0.2_r, 0.1_r});
  VMatrix3x3r inertia = VDiagonalMatrix<3>(Vec4r{1.2_r, 0.7_r, 0.1_r});
  inertia = Dot3x3(rot, Dot3x3(inertia, Transpose3x3(rot)));

  TransformRT stageStartPos;
  SetPose(std::array<real, kNumDofs>{-0.2_r, 0.2_r, 0.4_r, -0.5_r, 0.2_r, -0.6_r}, stageStartPos);
  RigidBodyVel stageStartVel;
  stageStartVel.SetVCom({-0.9_r, 0.9_r, -0.2_r});
  stageStartVel.SetOmega({-0.8_r, 0.6_r, -0.4_r});
  real dtStage = 1e-2_r;
  stageStartVel.UpdateVSymIfDirty(dtStage);

  // Create assembly lambda and test
  auto assembler =
      [&](TransformRT const& pose, double* energy, RigidGradient* gradient, RigidHessian* hessian) {
        AddRigidBodyInertiaFromMerit<GradTarget::Current, /*kPsdDRes*/ false>(
            mass, inertia, dtStage, pose, stageStartPos, stageStartVel, energy, gradient, hessian);
      };
  TestConsistency(assembler, 1e-3_r, 4e-3_r);
}

TEST(MultiBody, RigidBodyInertiaFromMerit_GradPrevPose) {
  // Initialize data needed by the assembler
  real mass = 1.3_r;
  VMatrix3x3r rot = Rodrigues(Vec4r{-0.4_r, 0.2_r, 0.1_r});
  VMatrix3x3r inertia = VDiagonalMatrix<3>(Vec4r{1.2_r, 0.7_r, 0.1_r});
  inertia = Dot3x3(rot, Dot3x3(inertia, Transpose3x3(rot)));

  TransformRT pose;
  SetPose(std::array<real, kNumDofs>{-0.2_r, 0.2_r, 0.4_r, -0.5_r, 0.2_r, -0.6_r}, pose);
  RigidBodyVel stageStartVel;
  stageStartVel.SetVCom({-0.9_r, 0.9_r, -0.2_r});
  stageStartVel.SetOmega({-0.8_r, 0.6_r, -0.4_r});
  real dtStage = 1e-2_r;
  stageStartVel.UpdateVSymIfDirty(dtStage);

  // Create assembly lambda and test
  auto assembler = [&](TransformRT const& stageStartPos, double* energy, RigidGradient* gradient) {
    AddRigidBodyInertiaFromMerit<GradTarget::Previous, /*kPsdDRes*/ false>(
        mass, inertia, dtStage, pose, stageStartPos, stageStartVel, energy, gradient, nullptr);
  };
  TestConsistency(assembler, 1e-3_r);

  // Validate that the energy is the same as in AddRigidBodyInertiaFromMerit()
  TransformRT stageStartPos;
  SetPose(kPoseValsRef, stageStartPos);
  double energyOriginal{};
  double energyTest{};
  AddRigidBodyInertiaFromMerit<GradTarget::Current, /*kPsdDRes*/ false>(
      mass,
      inertia,
      dtStage,
      pose,
      stageStartPos,
      stageStartVel,
      &energyOriginal,
      nullptr,
      nullptr);
  AddRigidBodyInertiaFromMerit<GradTarget::Previous, /*kPsdDRes*/ false>(
      mass, inertia, dtStage, pose, stageStartPos, stageStartVel, &energyTest, nullptr, nullptr);

  EXPECT_NEAR(energyTest / energyOriginal, 1_r, 1e-6_r);
}

TEST(MultiBody, RigidBodyInertiaFromMerit_GradPrevPoseIncrement) {
  // Initialize data needed by the assembler
  real mass = 1.3_r;
  VMatrix3x3r rot = Rodrigues(Vec4r{-0.4_r, 0.2_r, 0.1_r});
  VMatrix3x3r inertia = VDiagonalMatrix<3>(Vec4r{1.2_r, 0.7_r, 0.1_r});
  inertia = Dot3x3(rot, Dot3x3(inertia, Transpose3x3(rot)));

  TransformRT pose;
  SetPose(std::array<real, kNumDofs>{-0.2_r, 0.2_r, 0.4_r, -0.5_r, 0.2_r, -0.6_r}, pose);
  TransformRT stageStartPos;
  SetPose(std::array<real, kNumDofs>{-0.1_r, 0.3_r, 0.1_r, -0.3_r, -0.2_r, 0.1_r}, stageStartPos);
  real dtStage = 1e-2_r;

  // Create assembly lambda and test
  auto assembler =
      [&](TransformRT const& stageStartDeltaPos, double* energy, RigidGradient* gradient) {
        RigidBodyVel stageStartVel;
        TransformRT oldPose{
            stageStartDeltaPos.GetRotation().GetConjugate() * stageStartPos.GetRotation(),
            stageStartPos.VGetTranslation() - stageStartDeltaPos.VGetTranslation()};
        stageStartVel.SetFromFiniteDifferencePose(oldPose, stageStartPos, dtStage);
        AddRigidBodyInertiaFromMerit<GradTarget::PreviousDelta, /*kPsdDRes*/ false>(
            mass, inertia, dtStage, pose, stageStartPos, stageStartVel, energy, gradient, nullptr);
      };
  TestConsistency(assembler, 1e-3_r);

  // Validate that the energy is the same as in AddRigidBodyInertiaFromMerit()
  RigidBodyVel stageStartVel;
  stageStartVel.SetVCom({-0.9_r, 0.9_r, -0.2_r});
  stageStartVel.SetOmega({-0.8_r, 0.6_r, -0.4_r});
  stageStartVel.UpdateVSymIfDirty(dtStage);
  double energyOriginal{};
  double energyTest{};
  AddRigidBodyInertiaFromMerit<GradTarget::Current, /*kPsdDRes*/ false>(
      mass,
      inertia,
      dtStage,
      pose,
      stageStartPos,
      stageStartVel,
      &energyOriginal,
      nullptr,
      nullptr);
  AddRigidBodyInertiaFromMerit<GradTarget::PreviousDelta, /*kPsdDRes*/ false>(
      mass, inertia, dtStage, pose, stageStartPos, stageStartVel, &energyTest, nullptr, nullptr);

  EXPECT_NEAR(energyTest / energyOriginal, 1_r, 1e-6_r);
}

TEST(MultiBody, RigidBodyInertiaFromMerit_HessMixedPrevPose) {
  // Initialize data needed by the assembler
  real mass = 1.3_r;
  VMatrix3x3r rot = Rodrigues(Vec4r{-0.4_r, 0.2_r, 0.1_r});
  VMatrix3x3r inertia = VDiagonalMatrix<3>(Vec4r{1.2_r, 0.7_r, 0.1_r});
  inertia = Dot3x3(rot, Dot3x3(inertia, Transpose3x3(rot)));

  TransformRT pose;
  SetPose(std::array<real, kNumDofs>{-0.2_r, 0.2_r, 0.4_r, -0.5_r, 0.2_r, -0.6_r}, pose);
  RigidBodyVel stageStartVel;
  stageStartVel.SetVCom({-0.9_r, 0.9_r, -0.2_r});
  stageStartVel.SetOmega({-0.8_r, 0.6_r, -0.4_r});
  real dtStage = 1e-2_r;
  stageStartVel.UpdateVSymIfDirty(dtStage);

  // Create assembly lambda and test
  auto assembler = [&](TransformRT const& stageStartPos,
                       RigidGradient* gradient,
                       RigidHessian* hessianMixed) {
    AddRigidBodyInertiaFromMerit<GradTarget::Current, /*kPsdDRes*/ false>(
        mass, inertia, dtStage, pose, stageStartPos, stageStartVel, nullptr, gradient, nullptr);
    AddRigidBodyInertiaFromMerit<GradTarget::Previous, /*kPsdDRes*/ false>(
        mass, inertia, dtStage, pose, stageStartPos, stageStartVel, nullptr, nullptr, hessianMixed);
  };
  TestConsistencyMixed(assembler, 4e-3_r);
}

TEST(MultiBody, RigidBodyInertiaFromMerit_HessMixedPrevPoseIncrement) {
  // Initialize data needed by the assembler
  real mass = 1.3_r;
  VMatrix3x3r rot = Rodrigues(Vec4r{-0.4_r, 0.2_r, 0.1_r});
  VMatrix3x3r inertia = VDiagonalMatrix<3>(Vec4r{1.2_r, 0.7_r, 0.1_r});
  inertia = Dot3x3(rot, Dot3x3(inertia, Transpose3x3(rot)));

  TransformRT pose;
  SetPose(std::array<real, kNumDofs>{-0.2_r, 0.2_r, 0.4_r, -0.5_r, 0.2_r, -0.6_r}, pose);
  TransformRT stageStartPos;
  SetPose(std::array<real, kNumDofs>{-0.1_r, 0.3_r, 0.1_r, -0.3_r, -0.2_r, 0.1_r}, stageStartPos);
  real dtStage = 1e-2_r;

  // Create assembly lambda and test
  auto assembler = [&](TransformRT const& stageStartDeltaPos,
                       RigidGradient* gradient,
                       RigidHessian* hessianMixed) {
    RigidBodyVel stageStartVel;
    TransformRT oldPose{
        stageStartDeltaPos.GetRotation().GetConjugate() * stageStartPos.GetRotation(),
        stageStartPos.VGetTranslation() - stageStartDeltaPos.VGetTranslation()};
    stageStartVel.SetFromFiniteDifferencePose(oldPose, stageStartPos, dtStage);
    AddRigidBodyInertiaFromMerit<GradTarget::Current, /*kPsdDRes*/ false>(
        mass, inertia, dtStage, pose, stageStartPos, stageStartVel, nullptr, gradient, nullptr);
    AddRigidBodyInertiaFromMerit<GradTarget::PreviousDelta, /*kPsdDRes*/ false>(
        mass, inertia, dtStage, pose, stageStartPos, stageStartVel, nullptr, nullptr, hessianMixed);
  };
  TestConsistencyMixed(assembler, 4e-3_r);
}

TEST(RigidBodyAssembly, RigidBodyInertiaNewtonEuler) {
  // In this inertia model, merit and dresidual are consistent with the residual only if the inertia
  // is constant and the current rotation is equal to the stage-start rotation.

  // Initialize data needed by the assembler
  real mass = 1.3_r;
  // Use close-to homogeneous inertia, hence close to constant
  VMatrix3x3r inertia = VDiagonalMatrix<3>(Vec4r{1.3_r, 0.7_r, 1.1_r});
  inertia = SecondMomentToMomentOfInertia(inertia);

  // Make the stage-start rotation the same as the current rotation
  TransformRT stageStartPos;
  SetPose(
      std::array<real, kNumDofs>{
          -0.2_r, 0.2_r, 0.4_r, kPoseValsRef[3], kPoseValsRef[4], kPoseValsRef[5]},
      stageStartPos);
  RigidBodyVel stageStartVel;
  stageStartVel.SetVCom({-0.9_r, 0.9_r, -0.2_r});
  stageStartVel.SetOmega({-0.8_r, 0.6_r, -0.4_r});
  real dtStage = 1e-2_r;
  stageStartVel.UpdateVSymIfDirty(dtStage);

  // Create assembly lambda and test
  auto assembler =
      [&](TransformRT const& pose, double* energy, RigidGradient* gradient, RigidHessian* hessian) {
        AddRigidBodyInertiaNewtonEuler</*kPsdDRes*/ false>(
            mass, inertia, dtStage, pose, stageStartPos, stageStartVel, energy, gradient, hessian);
      };
  TestConsistency(assembler, 1e-3_r, 3e-3_r);
}

TEST(RigidBodyAssembly, RigidInertiaFnPsdDRes) {
  real const mass = 1.3_r;
  VMatrix3x3r const inertia =
      RotateInertia(Vec4r{1.2_r, 0.7_r, 0.1_r}, Rodrigues(Vec4r{-0.4_r, 0.2_r, 0.1_r}));
  real const dtStage = 1e-2_r;
  TransformRT stageStartPos;
  SetPose(std::array<real, kNumDofs>{-0.2_r, 0.2_r, 0.4_r, -0.5_r, 0.2_r, -0.6_r}, stageStartPos);
  RigidBodyVel stageStartVel;
  stageStartVel.SetOmega({-0.8_r, 0.6_r, -0.4_r});
  stageStartVel.UpdateVSymIfDirty(dtStage);
  // Far from the extrapolated rotation, the exact Hessian is below the floor and is raised. At that
  // rotation, it equals the Gauss-Newton Hessian G, which the floor changes only within its
  // rounding margins.
  TransformRT referencePose;
  SetPose(kPoseValsRef, referencePose);
  TransformRT const extrapolatedPose{
      Quaternion::FromRotationVector(stageStartVel.GetFiniteStepRotationVector(dtStage)) *
      stageStartPos.GetRotation()};

  // The rotational block of the Hessian, accumulated onto `seed`.
  auto hessianRot = [&]<typename InertiaT>(
                        bool useNewtonEuler,
                        bool psdDRes,
                        InertiaT const& in,
                        TransformRT const& pose,
                        VMatrix3x3r const& seed = {}) {
    RigidHessian hessian{};
    StoreSubmatrix<3, 3, 6, 6>(hessian, Int2{3, 3}, seed);
    GetRigidInertiaFn<InertiaT>(GradTarget::Current, useNewtonEuler, psdDRes)(
        mass, in, dtStage, pose, stageStartPos, stageStartVel, nullptr, nullptr, &hessian, nullptr);
    VMatrix3x3r rotational;
    LoadSubmatrix<3, 3, 6, 6>(rotational, Int2{3, 3}, hessian);
    return rotational;
  };

  // Merit-based inertia: ProjectSymAbove's projection of the exact Hessian above the floor
  // kMinInertiaCurvatureRatio * (G + 16 eps tr(W) I), with G from Newton-Euler inertia and
  // tr(W) = tr(G) / 2, applied to the inertia's Hessian alone.
  auto expectFloored = [&]<typename InertiaT>(
                           InertiaT const& secondMoment, InertiaT const& momentOfInertia) {
    for (auto const& [pose, raised] :
         {std::pair{referencePose, true}, std::pair{extrapolatedPose, false}}) {
      VMatrix3x3r const exact =
          hessianRot(/*useNewtonEuler*/ false, /*psdDRes*/ false, secondMoment, pose);
      VMatrix3x3r const gaussNewton =
          hessianRot(/*useNewtonEuler*/ true, /*psdDRes*/ false, momentOfInertia, pose);
      VMatrix3x3r const floor = kMinInertiaCurvatureRatio *
          (gaussNewton +
           VDiagonalMatrix<3>(8_r * std::numeric_limits<real>::epsilon() * Trace3x3(gaussNewton)));
      VMatrix3x3r const floored =
          gaussNewton +
          hessianRot(
              /*useNewtonEuler*/ false, /*psdDRes*/ true, secondMoment, pose, -gaussNewton);
      EXPECT_NEAR_TOL(
          ToNdArray3x3(floored),
          ToNdArray3x3(ProjectSymAbove(exact, floor)),
          1e-5_r * Norm3x3(gaussNewton));
      EXPECT_EQ(Norm3x3(floored - exact) > 1e-3_r * Norm3x3(gaussNewton), raised);
    }
  };
  // A line-like body's second moment n n^T has rank 1, so its G is singular about n without the
  // 16 eps tr(W) shift.
  Vec4r const axis{2_r / 7_r, -3_r / 7_r, 6_r / 7_r};
  VMatrix3x3r const line = Outer3(axis, axis);
  // The floor has no absolute tolerances, so it holds at any scale.
  for (real scale : {1_r, 1e-12_r}) {
    expectFloored(scale * inertia, scale * SecondMomentToMomentOfInertia(inertia));
    // Isotropic, and anisotropic unrotated
    expectFloored(scale * 0.6_r, scale * 1.2_r);
    expectFloored(scale * Vec4r{0.2_r, 0.5_r, 0.3_r}, scale * Vec4r{0.8_r, 0.5_r, 0.7_r});
    expectFloored(scale * line, scale * SecondMomentToMomentOfInertia(line));
  }
  // A block above the floor is kept bitwise.
  EXPECT_EQ(
      ToNdArray3x3(
          hessianRot(/*useNewtonEuler*/ false, /*psdDRes*/ true, inertia, extrapolatedPose)),
      ToNdArray3x3(
          hessianRot(/*useNewtonEuler*/ false, /*psdDRes*/ false, inertia, extrapolatedPose)));

  // Zero inertia, or invalid inertia with a negative principal moment, has no positive definite
  // floor, and Newton-Euler inertia's Hessian is its own Gauss-Newton Hessian. Both get a positive
  // semi-definite projection of the inertia's Hessian alone.
  auto expectProjected = [&]<typename InertiaT>(bool useNewtonEuler, InertiaT const& in) {
    for (TransformRT const& pose : {referencePose, extrapolatedPose}) {
      VMatrix3x3r projected = hessianRot(useNewtonEuler, /*psdDRes*/ false, in, pose);
      ProjectSymPsd(projected);
      VMatrix3x3r const seed = -2_r * projected;
      EXPECT_NEAR_TOL(
          ToNdArray3x3(hessianRot(useNewtonEuler, /*psdDRes*/ true, in, pose, seed) - seed),
          ToNdArray3x3(projected),
          1e-6_r * Norm3x3(projected));
    }
  };
  expectProjected(/*useNewtonEuler*/ false, VMatrix3x3r{});
  expectProjected(/*useNewtonEuler*/ false, VDiagonalMatrix<3>(Vec4r{-0.25_r, -0.25_r, 1.25_r}));
  expectProjected(/*useNewtonEuler*/ false, 0_r);
  expectProjected(/*useNewtonEuler*/ false, Vec4r{-0.25_r, -0.25_r, 1.25_r});
  expectProjected(/*useNewtonEuler*/ true, SecondMomentToMomentOfInertia(inertia));
  expectProjected(/*useNewtonEuler*/ true, -0.1_r);
  expectProjected(/*useNewtonEuler*/ true, Vec4r{1.3_r, 0.7_r, -0.1_r});
}

TEST(RigidBodyAssembly, RigidBodyDampingCurrent) {
  // Initialize data needed by the assembler
  real damping = 1.6_r;
  TransformRT stageStartPos;
  SetPose(std::array<real, kNumDofs>{-0.2_r, 0.2_r, 0.4_r, -0.5_r, 0.2_r, -0.6_r}, stageStartPos);
  real dtStage = 1e-2_r;

  // Create assembly lambda and test
  auto assembler =
      [&](TransformRT const& pose, double* energy, RigidGradient* gradient, RigidHessian* hessian) {
        AddRigidBodyDamping<GradTarget::Current>(
            pose, damping, stageStartPos, dtStage, energy, gradient, hessian);
      };
  TestConsistency(assembler, 1e-3_r, 4e-3_r);

  // Test the force and torque against expected values
  TransformRT pose;
  SetPose(kPoseValsRef, pose);
  RigidBodyVel vel;
  vel.SetFromFiniteDifferencePose(stageStartPos, pose, dtStage);
  Real3 expectedForce = ToReal3(damping * vel.GetVCom());
  Real3 expectedTorque = ToReal3(damping * vel.GetOmegaAndVSym().first);

  RigidGradient gradient{};
  AddRigidBodyDamping<GradTarget::Current>(
      pose, damping, stageStartPos, dtStage, nullptr, &gradient, nullptr);
  Real3 force = Unflatten<Real3 const>(gradient)[0];
  Real3 torque = Unflatten<Real3 const>(gradient)[1];

  EXPECT_NEAR_TOL(force, expectedForce, 1e-3_r);
  EXPECT_NEAR_TOL(torque, expectedTorque, 1e-3_r);
}

TEST(RigidBodyAssembly, RigidBodyDampingPrevious) {
  // Initialize data needed by the assembler
  real damping = 1.6_r;
  TransformRT pose;
  SetPose(std::array<real, kNumDofs>{-0.5_r, -0.2_r, 0.1_r, -0.8_r, 0.7_r, 0.3_r}, pose);
  real dtStage = 1e-2_r;

  // Create assembly lambda and test
  auto assembler = [&](TransformRT const& stageStartPos,
                       double* energy,
                       RigidGradient* gradient,
                       RigidHessian* hessian) {
    AddRigidBodyDamping<GradTarget::Previous>(
        pose, damping, stageStartPos, dtStage, energy, gradient, hessian);
  };
  TestConsistency(assembler, 1e-3_r, 7e-3_r);
}

TEST(RigidBodyAssembly, RigidBodyFrictionCurrent) {
  // Initialize data needed by the assembler
  real friction = 3.2_r;
  ArticulatedJointFrictionParams frictionParams{.coulomb = friction};
  TransformRT stageStartPos;
  SetPose(kPoseValsRef, stageStartPos);
  real dtStage = 1e-2_r;
  stageStartPos.SetTranslation(stageStartPos.GetTranslation() + dtStage * Real3{10_r, -10_r, 5_r});
  stageStartPos.SetRotation(
      Quaternion::FromRotationVector(dtStage * Real3{5_r, 10_r, -10_r}) *
      stageStartPos.GetRotation());

  // Create assembly lambda and test
  auto assembler =
      [&](TransformRT const& pose, double* energy, RigidGradient* gradient, RigidHessian* hessian) {
        AddRigidBodyFriction<GradTarget::Current>(
            false /*useFittedHessian*/,
            false /*psdDRes*/,
            frictionParams,
            dtStage,
            pose,
            stageStartPos,
            energy,
            gradient,
            hessian);
      };
  TestConsistency(assembler, 2e-3_r, 5e-3_r);

  // Test the force and torque against expected values
  TransformRT pose;
  SetPose(kPoseValsRef, pose);
  Real3 vtrans = (pose.GetTranslation() - stageStartPos.GetTranslation()) / dtStage;
  Real3 vrot = ToReal3(
      InvRodrigues(ToVMatrix3x3(pose.GetRotation() * stageStartPos.GetRotation().GetConjugate())) /
      dtStage);

  Real3 expectedForce = (friction / Norm(vtrans)) * vtrans;
  Real3 expectedTorque = (friction / Norm(vrot)) * vrot;

  RigidGradient gradient{};
  AddRigidBodyFriction<GradTarget::Current>(
      false /*useFittedHessian*/,
      false /*psdDRes*/,
      frictionParams,
      dtStage,
      pose,
      stageStartPos,
      nullptr,
      &gradient,
      nullptr);
  Real3 force = Unflatten<Real3 const>(gradient)[0];
  Real3 torque = Unflatten<Real3 const>(gradient)[1];

  EXPECT_NEAR_TOL(force, expectedForce, 1e-3_r);
  EXPECT_NEAR_TOL(torque, expectedTorque, 1e-3_r);
}

TEST(RigidBodyAssembly, RigidBodyFrictionCurrentWithStribeck) {
  // Initialize data needed by the assembler
  ArticulatedJointFrictionParams frictionParams{
      .coulomb = 3.2_r, .stictionExtra = 1.5_r, .stribeckVel = 0.1_r};
  TransformRT stageStartPos;
  SetPose(kPoseValsRef, stageStartPos);
  real dtStage = 1e-2_r;
  stageStartPos.SetTranslation(stageStartPos.GetTranslation() + dtStage * Real3{10_r, -10_r, 5_r});
  stageStartPos.SetRotation(
      Quaternion::FromRotationVector(dtStage * Real3{5_r, 10_r, -10_r}) *
      stageStartPos.GetRotation());

  // Create assembly lambda and test consistency only (expected force formula differs with Stribeck)
  auto assembler =
      [&](TransformRT const& pose, double* energy, RigidGradient* gradient, RigidHessian* hessian) {
        AddRigidBodyFriction<GradTarget::Current>(
            false /*useFittedHessian*/,
            false /*psdDRes*/,
            frictionParams,
            dtStage,
            pose,
            stageStartPos,
            energy,
            gradient,
            hessian);
      };
  TestConsistency(assembler, 2e-3_r, 5e-3_r);
}

TEST(RigidBodyAssembly, RigidBodyFrictionPrevious) {
  // Initialize data needed by the assembler
  real friction = 3.2_r;
  ArticulatedJointFrictionParams frictionParams{.coulomb = friction};
  TransformRT pose;
  SetPose(kPoseValsRef, pose);
  real dtStage = 1e-2_r;
  pose.SetTranslation(pose.GetTranslation() + dtStage * Real3{10_r, -10_r, 5_r});
  pose.SetRotation(
      Quaternion::FromRotationVector(dtStage * Real3{5_r, 10_r, -10_r}) * pose.GetRotation());

  // Create assembly lambda and test
  auto assembler = [&](TransformRT const& stageStartPos,
                       double* energy,
                       RigidGradient* gradient,
                       RigidHessian* hessian) {
    AddRigidBodyFriction<GradTarget::Previous>(
        false /*useFittedHessian*/,
        false /*psdDRes*/,
        frictionParams,
        dtStage,
        pose,
        stageStartPos,
        energy,
        gradient,
        hessian);
  };
  TestConsistency(assembler, 2e-3_r, 5e-3_r);
}

TEST(RigidBodyAssembly, RigidBodyFrictionPreviousWithStribeck) {
  // Initialize data needed by the assembler
  ArticulatedJointFrictionParams frictionParams{
      .coulomb = 3.2_r, .stictionExtra = 1.5_r, .stribeckVel = 0.1_r};
  TransformRT pose;
  SetPose(kPoseValsRef, pose);
  real dtStage = 1e-2_r;
  pose.SetTranslation(pose.GetTranslation() + dtStage * Real3{10_r, -10_r, 5_r});
  pose.SetRotation(
      Quaternion::FromRotationVector(dtStage * Real3{5_r, 10_r, -10_r}) * pose.GetRotation());

  // Create assembly lambda and test consistency only (expected force formula differs with Stribeck)
  auto assembler = [&](TransformRT const& stageStartPos,
                       double* energy,
                       RigidGradient* gradient,
                       RigidHessian* hessian) {
    AddRigidBodyFriction<GradTarget::Previous>(
        false /*useFittedHessian*/,
        false /*psdDRes*/,
        frictionParams,
        dtStage,
        pose,
        stageStartPos,
        energy,
        gradient,
        hessian);
  };
  TestConsistency(assembler, 2e-3_r, 5e-3_r);
}
