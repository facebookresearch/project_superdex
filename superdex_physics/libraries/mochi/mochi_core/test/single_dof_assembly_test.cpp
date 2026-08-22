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

#include <mochi_core/utils/single_dof_assembly.h>

#include <gtest/gtest.h>

using namespace mochi;

namespace {
real constexpr kPosValRef = 1.2_r;
} // namespace

static void Compare(real m1, real m2, real tol) {
  real normDiff = Abs(m1 - m2);
  real maxNorm = std::max(Abs(m1), Abs(m2));
  if (maxNorm > 1e-9_r) {
    EXPECT_NEAR(normDiff / maxNorm, 0_r, tol);
  }
}

static void TestAssembler(
    std::function<void(real, double*, real*, real*)> assembler,
    real tolGrad,
    real tolHess) {
  real constexpr kEps = 2.5e-3_r;
  real constexpr kOneOverTwoEps = 1_r / (2_r * kEps);

  // Initialize test position
  real pos = kPosValRef;

  // Evaluate analytical values
  double energy{};
  real gradient{};
  real hessian{};
  assembler(pos, &energy, &gradient, &hessian);
  auto gradientBase = gradient;
  auto hessianBase = hessian;

  // Compute test values using finite differences
  auto evalDelta = [&](real delta) {
    pos = kPosValRef + delta;
    energy = {};
    gradient = {};
    assembler(pos, &energy, &gradient, nullptr);
  };
  evalDelta(kEps);
  auto energyFwd = energy;
  auto gradientFwd = gradient;
  evalDelta(-kEps);
  auto energyBwd = energy;
  auto gradientBwd = gradient;
  real gradientTest = kOneOverTwoEps * (energyFwd - energyBwd);
  real hessianTest = kOneOverTwoEps * (gradientFwd - gradientBwd);

  // Compare analytical and test values
  Compare(gradientBase, gradientTest, tolGrad);
  Compare(hessianBase, hessianTest, tolHess);
}

TEST(SingleDofAssembly, SingleDofExternal) {
  // Initialize data needed by the assembler
  real force = 1.3_r;

  // Create assembly lambda and test
  auto assembler = [&](real pos, double* energy, real* gradient, real* /* hessian */) {
    AddSingleDofExternalForce(pos, force, energy, gradient);
  };
  TestAssembler(assembler, 1e-3_r, 1e-3_r);
}

TEST(SingleDofAssembly, SingleDofInertiaCurrent) {
  // Initialize data needed by the assembler
  real inertia = 1.3_r;
  real stageStartPos = -0.2_r;
  real stageStartVel = -0.9_r;
  real dtStage = 1e-2_r;

  // Create assembly lambda and test
  auto assembler = [&](real pos, double* energy, real* gradient, real* hessian) {
    AddSingleDofInertia<GradTarget::Current>(
        pos, inertia, stageStartPos, stageStartVel, dtStage, energy, gradient, hessian);
  };
  TestAssembler(assembler, 1e-3_r, 1e-3_r);
}

TEST(SingleDofAssembly, SingleDofInertiaPrevious) {
  // Initialize data needed by the assembler
  real inertia = 1.3_r;
  real pos = 0.7_r;
  real stageStartVel = -0.9_r;
  real dtStage = 1e-2_r;

  // Create assembly lambda and test
  auto assembler = [&](real stageStartPos, double* energy, real* gradient, real* hessian) {
    AddSingleDofInertia<GradTarget::Previous>(
        pos, inertia, stageStartPos, stageStartVel, dtStage, energy, gradient, hessian);
  };
  TestAssembler(assembler, 1e-3_r, 1e-3_r);
}

TEST(SingleDofAssembly, SingleDofInertiaPreviousDelta) {
  // Initialize data needed by the assembler
  real inertia = 1.3_r;
  real pos = 0.7_r;
  real stageStartPos = -0.2_r;
  real dtStage = 1e-2_r;

  // Create assembly lambda and test
  auto assembler = [&](real stageStartDeltaPos, double* energy, real* gradient, real* hessian) {
    real stageStartVel = stageStartDeltaPos / dtStage;
    AddSingleDofInertia<GradTarget::PreviousDelta>(
        pos, inertia, stageStartPos, stageStartVel, dtStage, energy, gradient, hessian);
  };
  TestAssembler(assembler, 1e-3_r, 1e-3_r);
}

TEST(SingleDofAssembly, SingleDofDampingCurrent) {
  // Initialize data needed by the assembler
  real damping = 1.6_r;
  real stageStartPos = -0.2_r;
  real dtStage = 1e-2_r;

  // Create assembly lambda and test
  auto assembler = [&](real pos, double* energy, real* gradient, real* hessian) {
    AddSingleDofDamping<GradTarget::Current>(
        pos, damping, stageStartPos, dtStage, energy, gradient, hessian);
  };
  TestAssembler(assembler, 1e-3_r, 1e-3_r);

  // Test the force against the expected value
  real pos = kPosValRef;
  real vel = (pos - stageStartPos) / dtStage;
  real expectedForce = damping * vel;

  real force{};
  AddSingleDofDamping<GradTarget::Current>(
      pos, damping, stageStartPos, dtStage, nullptr, &force, nullptr);

  EXPECT_NEAR_TOL(force, expectedForce, 1e-3_r);
}

TEST(SingleDofAssembly, SingleDofDampingPrevious) {
  // Initialize data needed by the assembler
  real damping = 1.6_r;
  real pos = 0.7_r;
  real dtStage = 1e-2_r;

  // Create assembly lambda and test
  auto assembler = [&](real stageStartPos, double* energy, real* gradient, real* hessian) {
    AddSingleDofDamping<GradTarget::Previous>(
        pos, damping, stageStartPos, dtStage, energy, gradient, hessian);
  };
  TestAssembler(assembler, 1e-3_r, 1e-3_r);
}

TEST(SingleDofAssembly, SingleDofFrictionCurrent) {
  // Initialize data needed by the assembler
  real friction = 3.1_r;
  ArticulatedJointFrictionParams frictionParams{.coulomb = friction};
  real stageStartPos = -0.2_r;
  real dtStage = 1e-2_r;

  // Create assembly lambda and test
  auto assembler = [&](real pos, double* energy, real* gradient, real* hessian) {
    AddSingleDofFriction<GradTarget::Current>(
        false /*useFittedHessian*/,
        false /*psdDRes*/,
        frictionParams,
        dtStage,
        pos,
        stageStartPos,
        energy,
        gradient,
        hessian);
  };
  TestAssembler(assembler, 1e-3_r, 1e-3_r);

  // Test the force against the expected value
  real pos = kPosValRef;
  real vel = (pos - stageStartPos) / dtStage;
  real expectedForce = friction * vel / Abs(vel);

  real force{};
  AddSingleDofFriction<GradTarget::Current>(
      false /*useFittedHessian*/,
      false /*psdDRes*/,
      frictionParams,
      dtStage,
      pos,
      stageStartPos,
      nullptr,
      &force,
      nullptr);

  EXPECT_NEAR_TOL(force, expectedForce, 1e-3_r);
}

TEST(SingleDofAssembly, SingleDofFrictionCurrentWithStribeck) {
  // Initialize data needed by the assembler; tuned to be in the Stribeck transition region for
  // nontrivial Hessian.
  ArticulatedJointFrictionParams frictionParams{
      .coulomb = 3.1_r, .falloffVel = 0.5_r, .stictionExtra = 1.5_r, .stribeckVel = 5.0_r};
  real dtStage = 1e-2_r;
  real stageStartPos = kPosValRef - dtStage * 5.5_r;

  auto assembler = [&](real pos, double* energy, real* gradient, real* hessian) {
    AddSingleDofFriction<GradTarget::Current>(
        false /*useFittedHessian*/,
        false /*psdDRes*/,
        frictionParams,
        dtStage,
        pos,
        stageStartPos,
        energy,
        gradient,
        hessian);
  };
  TestAssembler(assembler, 1e-3_r, 1e-3_r);
}

TEST(SingleDofAssembly, SingleDofFrictionPrevious) {
  // Initialize data needed by the assembler
  real friction = 3.1_r;
  ArticulatedJointFrictionParams frictionParams{.coulomb = friction};
  real pos = 0.7_r;
  real dtStage = 1e-2_r;

  // Create assembly lambda and test
  auto assembler = [&](real stageStartPos, double* energy, real* gradient, real* hessian) {
    AddSingleDofFriction<GradTarget::Previous>(
        false /*useFittedHessian*/,
        false /*psdDRes*/,
        frictionParams,
        dtStage,
        pos,
        stageStartPos,
        energy,
        gradient,
        hessian);
  };
  TestAssembler(assembler, 1e-3_r, 1e-3_r);
}

TEST(SingleDofAssembly, SingleDofFrictionPreviousWithStribeck) {
  // Initialize data needed by the assembler
  ArticulatedJointFrictionParams frictionParams{
      .coulomb = 3.1_r, .falloffVel = 0.5_r, .stictionExtra = 1.5_r, .stribeckVel = 5.0_r};
  real dtStage = 1e-2_r;
  real pos = kPosValRef + dtStage * 5.5_r;

  auto assembler = [&](real stageStartPos, double* energy, real* gradient, real* hessian) {
    AddSingleDofFriction<GradTarget::Previous>(
        false /*useFittedHessian*/,
        false /*psdDRes*/,
        frictionParams,
        dtStage,
        pos,
        stageStartPos,
        energy,
        gradient,
        hessian);
  };
  TestAssembler(assembler, 1e-3_r, 1e-3_r);
}
