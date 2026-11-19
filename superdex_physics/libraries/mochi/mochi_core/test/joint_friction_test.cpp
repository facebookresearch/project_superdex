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

#include <mochi_core/articulated_body/joint_friction.h>
#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/reflection.h>

#include <gtest/gtest.h>
#include <array>
#include <cmath>
#include <string>

using namespace mochi;

namespace {

real constexpr kDt = 1e-2_r;

// Static peak 4, Coulomb 3, presliding displacement 0.1, and a Stribeck velocity whose Stribeck
// displacement (1 * kDt = 0.01) is comparable to the slips of the tests.
ArticulatedJointFrictionParams const kParams{
    .coulomb = 3_r,
    .preslidingDisplacement = 0.1_r,
    .stictionExtra = 1_r,
    .stribeckVel = 1_r};
ArticulatedJointFrictionParams const kParamsNoStribeck{
    .coulomb = 3_r,
    .preslidingDisplacement = 0.1_r};

// Displacements from the anchor: sticking, close to the presliding displacement, and sliding in the
// Stribeck transition.
std::array<real, 3> constexpr kDisplacements = {0.03_r, 0.095_r, 0.104_r};

real SingleDofEnergy(ArticulatedJointFrictionParams const& params, real pos, real anchor) {
  double energy = 0.0;
  AddSingleDofFriction(false, false, params, kDt, pos, anchor, &energy, nullptr, nullptr);
  return static_cast<real>(energy);
}

real SingleDofGradient(ArticulatedJointFrictionParams const& params, real pos, real anchor) {
  real gradient = 0_r;
  AddSingleDofFriction(false, false, params, kDt, pos, anchor, nullptr, &gradient, nullptr);
  return gradient;
}

Quaternion Rotate(Quaternion const& rot, Vec4r leftIncrement) {
  return Quaternion::FromRotationVector(leftIncrement) * rot;
}

Vec4r Axis(int i) {
  Real3 e{};
  e[i] = 1_r;
  return ToSimd(e);
}

Vec4r RotationGradient(ArticulatedJointFrictionParams const& params, Quaternion rot, Quaternion a) {
  Vec4r gradient{};
  AddRotationFriction(false, false, params, kDt, rot, a, nullptr, &gradient, nullptr);
  return gradient;
}

// A rotation at a displacement d from the anchor, about a generic axis.
Quaternion RotationAtDisplacement(Quaternion const& anchor, real d) {
  return Rotate(anchor, d * Normalize<3>(Vec4r{0.3_r, -0.5_r, 0.8_r, 0_r}));
}

Quaternion const kAnchorRot = Quaternion::FromRotationVector(Real3{0.4_r, -0.2_r, 0.7_r});

} // namespace

TEST(JointFriction, SingleDofDerivatives) {
  real constexpr kEps = 2e-4_r;
  real const anchor = 0.6_r;
  for (auto const& params : {kParams, kParamsNoStribeck}) {
    for (real d : kDisplacements) {
      for (real sign : {1_r, -1_r}) {
        real const pos = anchor + sign * d;
        real gradient = 0_r;
        real hessian = 0_r;
        AddSingleDofFriction(false, false, params, kDt, pos, anchor, nullptr, &gradient, &hessian);
        real const gradientFd = (SingleDofEnergy(params, pos + kEps, anchor) -
                                 SingleDofEnergy(params, pos - kEps, anchor)) /
            (2_r * kEps);
        real const hessianFd = (SingleDofGradient(params, pos + kEps, anchor) -
                                SingleDofGradient(params, pos - kEps, anchor)) /
            (2_r * kEps);
        EXPECT_NEAR(gradient, gradientFd, 2e-2_r * Abs(gradient));
        EXPECT_NEAR(hessian, hessianFd, 2e-2_r * Max(Abs(hessian), 1_r));

        // The derivative with respect to the anchor is the negated exact Hessian.
        real const dGradientDAnchorFd = (SingleDofGradient(params, pos, anchor + kEps) -
                                         SingleDofGradient(params, pos, anchor - kEps)) /
            (2_r * kEps);
        EXPECT_NEAR(-hessian, dGradientDAnchorFd, 2e-2_r * Max(Abs(hessian), 1_r));
      }
    }
  }
}

TEST(JointFriction, SingleDofApproximateHessians) {
  // Sliding in the Stribeck transition, where the force decreases with the displacement.
  real const anchor = 0.6_r;
  real const pos = anchor + kDisplacements[2];
  real exact = 0_r;
  real clamped = 0_r;
  real fitted = 0_r;
  AddSingleDofFriction(false, false, kParams, kDt, pos, anchor, nullptr, nullptr, &exact);
  AddSingleDofFriction(false, true, kParams, kDt, pos, anchor, nullptr, nullptr, &clamped);
  AddSingleDofFriction(true, false, kParams, kDt, pos, anchor, nullptr, nullptr, &fitted);
  EXPECT_LT(exact, 0_r);
  EXPECT_EQ(clamped, 0_r);
  // The fitted Hessian is the secant of the force.
  EXPECT_NEAR(fitted, SingleDofGradient(kParams, pos, anchor) / (pos - anchor), 1e-4_r);
}

TEST(JointFriction, SingleDofForce) {
  real const anchor = -0.2_r;
  real const delta = kParams.preslidingDisplacement;

  // Below the presliding displacement, the force grows from zero; at it, it reaches the static
  // peak.
  EXPECT_EQ(SingleDofGradient(kParams, anchor, anchor), 0_r);
  real const stickForce = SingleDofGradient(kParams, anchor + 0.5_r * delta, anchor);
  EXPECT_GT(stickForce, 0_r);
  EXPECT_LT(stickForce, 4_r);
  EXPECT_NEAR(SingleDofGradient(kParams, anchor + delta, anchor), 4_r, 1e-4_r);
  EXPECT_NEAR(SingleDofGradient(kParams, anchor - delta, anchor), -4_r, 1e-4_r);

  // Sliding, the force follows the Stribeck curve of the slip velocity (slip beyond the presliding
  // displacement over the time since the anchor state), or is the Coulomb force without it.
  real const slip = 0.004_r;
  real const slipVel = slip / kDt;
  real const expected = 3_r + std::exp(-Sqr(slipVel / kParams.stribeckVel));
  EXPECT_NEAR(SingleDofGradient(kParams, anchor + delta + slip, anchor), expected, 1e-4_r);
  EXPECT_NEAR(SingleDofGradient(kParamsNoStribeck, anchor + delta + slip, anchor), 3_r, 1e-4_r);
}

TEST(JointFriction, QuasistaticStepKeepsStaticPeak) {
  // With an infinite time since the anchor state, the slip velocity is zero: friction keeps its
  // static peak, and the energy is finite.
  real const anchor = 0.1_r;
  real const pos = anchor + 2_r * kParams.preslidingDisplacement;
  double energy = 0.0;
  real gradient = 0_r;
  real hessian = 0_r;
  AddSingleDofFriction(false, false, kParams, kInf, pos, anchor, &energy, &gradient, &hessian);
  EXPECT_TRUE(std::isfinite(energy));
  EXPECT_NEAR(gradient, 4_r, 1e-4_r);
  EXPECT_EQ(hessian, 0_r);
}

TEST(JointFriction, AdvanceSingleDofAnchor) {
  real const delta = kParams.preslidingDisplacement;
  real const anchor = 1_r;
  // Sticking keeps the anchor; sliding drags it at the presliding displacement.
  EXPECT_EQ(AdvanceSingleDofFrictionAnchor(anchor + 0.5_r * delta, anchor, delta), anchor);
  EXPECT_NEAR(AdvanceSingleDofFrictionAnchor(anchor + 3_r * delta, anchor, delta), 1.2_r, 1e-6_r);
  EXPECT_NEAR(AdvanceSingleDofFrictionAnchor(anchor - 3_r * delta, anchor, delta), 0.8_r, 1e-6_r);

  for (real d : {0.5_r * delta, 3_r * delta}) {
    real posAdjoint = 0_r;
    real anchorAdjoint = 0_r;
    AdvanceSingleDofFrictionAnchorVjp(anchor + d, anchor, delta, 2_r, posAdjoint, anchorAdjoint);
    real constexpr kEps = 1e-3_r;
    real const dPos = (AdvanceSingleDofFrictionAnchor(anchor + d + kEps, anchor, delta) -
                       AdvanceSingleDofFrictionAnchor(anchor + d - kEps, anchor, delta)) /
        (2_r * kEps);
    real const dAnchor = (AdvanceSingleDofFrictionAnchor(anchor + d, anchor + kEps, delta) -
                          AdvanceSingleDofFrictionAnchor(anchor + d, anchor - kEps, delta)) /
        (2_r * kEps);
    EXPECT_NEAR(posAdjoint, 2_r * dPos, 1e-2_r);
    EXPECT_NEAR(anchorAdjoint, 2_r * dAnchor, 1e-2_r);
  }
}

TEST(JointFriction, RotationDerivatives) {
  real constexpr kEps = 5e-4_r;
  for (auto const& params : {kParams, kParamsNoStribeck}) {
    for (real d : kDisplacements) {
      Quaternion const rot = RotationAtDisplacement(kAnchorRot, d);
      Vec4r gradient{};
      AddRotationFriction(false, false, params, kDt, rot, kAnchorRot, nullptr, &gradient, nullptr);
      for (int i = 0; i < 3; ++i) {
        auto energyAt = [&](real h) {
          double energy = 0.0;
          AddRotationFriction(
              false,
              false,
              params,
              kDt,
              Rotate(rot, h * Axis(i)),
              kAnchorRot,
              &energy,
              nullptr,
              nullptr);
          return static_cast<real>(energy);
        };
        real const gradientFd = (energyAt(kEps) - energyAt(-kEps)) / (2_r * kEps);
        EXPECT_NEAR(gradient[i], gradientFd, 2e-2_r * Norm<3>(gradient));
        real const hessianScale = 4_r / params.preslidingDisplacement;

        // Derivative of the gradient with respect to the anchor, along a test vector.
        Vec4r const v{0.2_r, 0.7_r, -0.4_r, 0_r};
        real const dGradientDAnchorFd =
            Dot<3>(
                v,
                RotationGradient(params, rot, Rotate(kAnchorRot, kEps * Axis(i))) -
                    RotationGradient(params, rot, Rotate(kAnchorRot, -kEps * Axis(i)))) /
            (2_r * kEps);
        Vec4r const product =
            RotationFrictionDGradientDAnchorTransposeProduct(params, kDt, rot, kAnchorRot, v);
        EXPECT_NEAR(product[i], dGradientDAnchorFd, 1e-2_r * hessianScale);
      }
    }
  }
}

// The rotation Hessians are C^T H C, with C = dz/domega the Jacobian of the rotation vector z with
// respect to the left increment omega, and H = d2E/dz2: k_r u u^T + k_t (I - u u^T), with
// theta = |z|, u = z / theta and F the scalar force (single-DoF gradient). C keeps u, and C^T C
// scales the plane orthogonal to u by [theta / (2 sin(theta / 2))]^2, so
// k_t = (F / theta) [theta / (2 sin(theta / 2))]^2. k_r is F' (single-DoF exact Hessian) for the
// exact Hessian, its positive part for the clamped one, and F / theta for the fitted one.
TEST(JointFriction, RotationApproximateHessians) {
  real const hessianScale = 4_r / kParams.preslidingDisplacement;
  for (real d : kDisplacements) {
    SCOPED_TRACE(d);
    Quaternion const rot = RotationAtDisplacement(kAnchorRot, d);
    Vec4r const z = (rot * kAnchorRot.GetConjugate()).VToRotationVector();
    real const theta = Norm<3>(z);
    Vec4r const u = z / theta;
    real const force = SingleDofGradient(kParams, theta, 0_r);
    real stiffness = 0_r;
    AddSingleDofFriction(false, false, kParams, kDt, theta, 0_r, nullptr, nullptr, &stiffness);
    real const secant = force / theta;
    real const kt = secant * Sqr(theta / (2_r * Sin(0.5_r * theta)));
    struct Case {
      bool useFittedHessian;
      bool psdDRes;
      real kr;
    };
    for (Case const& c :
         {Case{.useFittedHessian = false, .psdDRes = false, .kr = stiffness},
          Case{.useFittedHessian = false, .psdDRes = true, .kr = Max(stiffness, 0_r)},
          Case{.useFittedHessian = true, .psdDRes = false, .kr = secant}}) {
      VMatrix3x3r hessian{};
      AddRotationFriction(
          c.useFittedHessian, c.psdDRes, kParams, kDt, rot, kAnchorRot, nullptr, nullptr, &hessian);
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          real const identity = i == j ? 1_r : 0_r;
          real const expected = c.kr * u[i] * u[j] + kt * (identity - u[i] * u[j]);
          EXPECT_NEAR(hessian[i][j], expected, 1e-4_r * hessianScale) << i << " " << j;
          EXPECT_NEAR(hessian[i][j], hessian[j][i], 1e-5_r * hessianScale);
        }
      }
    }
  }
  // In the Stribeck transition, the exact stiffness is negative, so the three Hessians differ.
  real stiffness = 0_r;
  AddSingleDofFriction(
      false, false, kParams, kDt, kDisplacements[2], 0_r, nullptr, nullptr, &stiffness);
  EXPECT_LT(stiffness, 0_r);
}

TEST(JointFriction, RotationForce) {
  // Sliding without Stribeck, the torque has the Coulomb magnitude, along the displacement.
  Quaternion const rot = RotationAtDisplacement(kAnchorRot, 0.15_r);
  Vec4r const gradient = RotationGradient(kParamsNoStribeck, rot, kAnchorRot);
  Vec4r const z = (rot * kAnchorRot.GetConjugate()).VToRotationVector();
  EXPECT_NEAR(Norm<3>(gradient), 3_r, 1e-3_r);
  EXPECT_NEAR(Dot<3>(gradient, z) / (Norm<3>(gradient) * Norm<3>(z)), 1_r, 1e-4_r);
}

TEST(JointFriction, AdvanceRotationAnchor) {
  real const delta = kParams.preslidingDisplacement;

  // Sticking keeps the anchor; sliding drags it at the presliding displacement.
  Quaternion const stickRot = RotationAtDisplacement(kAnchorRot, 0.5_r * delta);
  EXPECT_EQ(AdvanceRotationFrictionAnchor(stickRot, kAnchorRot, delta), kAnchorRot);
  Quaternion const slideRot = RotationAtDisplacement(kAnchorRot, 3_r * delta);
  Quaternion const newAnchor = AdvanceRotationFrictionAnchor(slideRot, kAnchorRot, delta);
  EXPECT_NEAR((slideRot * newAnchor.GetConjugate()).GetAngle(), delta, 1e-5_r);
  EXPECT_NEAR(
      (newAnchor * kAnchorRot.GetConjugate()).GetAngle(), 2_r * delta, 1e-5_r); // Along the slip

  // Vector-Jacobian product against finite differences of the new anchor, in left increments.
  real constexpr kEps = 1e-3_r;
  Vec4r const newAnchorAdjoint{0.3_r, -0.6_r, 0.5_r, 0_r};
  for (Quaternion const& rot : {stickRot, slideRot}) {
    Quaternion const anchorBase = AdvanceRotationFrictionAnchor(rot, kAnchorRot, delta);
    Vec4r rotAdjoint{};
    Vec4r anchorAdjoint{};
    AdvanceRotationFrictionAnchorVjp(
        rot, kAnchorRot, delta, newAnchorAdjoint, rotAdjoint, anchorAdjoint);
    for (int i = 0; i < 3; ++i) {
      auto directional = [&](Quaternion const& anchorP, Quaternion const& anchorM) {
        Vec4r const dP = (anchorP * anchorBase.GetConjugate()).VToRotationVector();
        Vec4r const dM = (anchorM * anchorBase.GetConjugate()).VToRotationVector();
        return Dot<3>(newAnchorAdjoint, dP - dM) / (2_r * kEps);
      };
      real const rotFd = directional(
          AdvanceRotationFrictionAnchor(Rotate(rot, kEps * Axis(i)), kAnchorRot, delta),
          AdvanceRotationFrictionAnchor(Rotate(rot, -kEps * Axis(i)), kAnchorRot, delta));
      real const anchorFd = directional(
          AdvanceRotationFrictionAnchor(rot, Rotate(kAnchorRot, kEps * Axis(i)), delta),
          AdvanceRotationFrictionAnchor(rot, Rotate(kAnchorRot, -kEps * Axis(i)), delta));
      EXPECT_NEAR(rotAdjoint[i], rotFd, 5e-3_r);
      EXPECT_NEAR(anchorAdjoint[i], anchorFd, 5e-3_r);
    }
  }
}

TEST(JointFriction, SingleDofTrustMargin) {
  real const delta = kParams.preslidingDisplacement;
  real const anchor = 1_r;
  real constexpr kTol = 1e-5_r;
  for (real sign : {1_r, -1_r}) {
    // Measured from a stage start half a presliding displacement from the anchor, on either side:
    // ahead of it, at the anchor, on the reverse yield point, and past it.
    real const stageStart = anchor + sign * 0.5_r * delta;
    auto marginAt = [&](real d) {
      return SingleDofFrictionTrustMargin(stageStart, anchor + sign * d, anchor, delta);
    };
    EXPECT_NEAR(marginAt(2_r * delta), 3_r, kTol);
    EXPECT_NEAR(marginAt(0_r), 1_r, kTol);
    EXPECT_NEAR(marginAt(-delta), 0_r, kTol);
    EXPECT_NEAR(marginAt(-2_r * delta), -1_r, kTol);
  }
  // Without a reliable reference, less than 0.1 presliding displacements from the anchor.
  EXPECT_EQ(SingleDofFrictionTrustMargin(anchor, anchor - 2_r * delta, anchor, delta), kInf);
  EXPECT_EQ(
      SingleDofFrictionTrustMargin(anchor + 0.09_r * delta, anchor - 2_r * delta, anchor, delta),
      kInf);
}

TEST(JointFriction, RotationTrustMargin) {
  real const delta = kParams.preslidingDisplacement;
  real constexpr kTol = 1e-4_r;
  Quaternion const stageStart = RotationAtDisplacement(kAnchorRot, 0.5_r * delta);
  // Along the reference: ahead of the anchor, on the reverse yield point, and past it.
  EXPECT_NEAR(
      RotationFrictionTrustMargin(
          stageStart, RotationAtDisplacement(kAnchorRot, 2_r * delta), kAnchorRot, delta),
      3_r,
      kTol);
  EXPECT_NEAR(
      RotationFrictionTrustMargin(
          stageStart, RotationAtDisplacement(kAnchorRot, -delta), kAnchorRot, delta),
      0_r,
      kTol);
  EXPECT_NEAR(
      RotationFrictionTrustMargin(
          stageStart, RotationAtDisplacement(kAnchorRot, -2_r * delta), kAnchorRot, delta),
      -1_r,
      kTol);
  // Sideways, orthogonal to the reference.
  Vec4r const reference = (stageStart * kAnchorRot.GetConjugate()).VToRotationVector();
  Vec4r const sideways = 3_r * delta * Normalize<3>(Cross3(reference, Axis(0)));
  EXPECT_NEAR(
      RotationFrictionTrustMargin(stageStart, Rotate(kAnchorRot, sideways), kAnchorRot, delta),
      1_r,
      kTol);
  // Without a reliable reference.
  EXPECT_EQ(
      RotationFrictionTrustMargin(
          RotationAtDisplacement(kAnchorRot, 0.09_r * delta),
          RotationAtDisplacement(kAnchorRot, -2_r * delta),
          kAnchorRot,
          delta),
      kInf);
}

// The deprecated falloffVel is serialized only when set, so exported prefabs omit it.
TEST(JointFriction, SerializesFalloffVelOnlyWhenSet) {
  ArticulatedJointFrictionParams params = kParams;
  EXPECT_EQ(SReflect::ToJsonString(params, /*pretty*/ false).find("falloffVel"), std::string::npos);
  params.falloffVel = 1e-3_r;
  EXPECT_NE(SReflect::ToJsonString(params, /*pretty*/ false).find("falloffVel"), std::string::npos);
}
