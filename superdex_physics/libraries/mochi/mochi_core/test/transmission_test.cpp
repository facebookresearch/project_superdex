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

#include <gtest/gtest.h>

#include <mochi_core/articulated_body/articulated_body.h>
#include <mochi_core/articulated_body/transmission.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/rand_utils.h>
#include <mochi_core/utils/rigid_body_size.h>

#include <cmath>

using namespace mochi;
using namespace mochi::articulated;

namespace {

// Finite difference step size for derivative tests
constexpr real kEpsilon = 1e-5_r;

// Tolerance for comparing finite differences to analytical derivatives
constexpr real kRelativeTolerance = 2e-3_r;

// Helper function to verify that Force is the derivative of Energy w.r.t. displacement
// and that Stiffness is the derivative of Force w.r.t. displacement
void VerifyDerivatives(
    TransmissionActuator const* actuator,
    real displacement,
    real prevDisplacement,
    real timeStep) {
  // Get the analytical values using the new interface
  real energy{}, force{}, stiffness{};
  actuator->EnergyGradientHessian(
      displacement, prevDisplacement, timeStep, &energy, &force, &stiffness);

  // Compute dEnergy/dDisplacement using finite differences
  real energyPlus{}, energyMinus{};
  actuator->EnergyGradientHessian(
      displacement + kEpsilon, prevDisplacement, timeStep, &energyPlus, nullptr, nullptr);
  actuator->EnergyGradientHessian(
      displacement - kEpsilon, prevDisplacement, timeStep, &energyMinus, nullptr, nullptr);
  real const dEnergyDDisplacementFD = (energyPlus - energyMinus) / (2_r * kEpsilon);

  // Verify that Force is the derivative of Energy w.r.t. displacement
  if (Abs(force) > 1e-10_r) {
    EXPECT_NEAR_RTOL(dEnergyDDisplacementFD, force, kRelativeTolerance);
  } else {
    // Use absolute tolerance for near-zero values
    EXPECT_NEAR(dEnergyDDisplacementFD, force, 1e-6_r);
  }

  // Compute dForce/dDisplacement using finite differences
  real forcePlus{}, forceMinus{};
  actuator->EnergyGradientHessian(
      displacement + kEpsilon, prevDisplacement, timeStep, nullptr, &forcePlus, nullptr);
  actuator->EnergyGradientHessian(
      displacement - kEpsilon, prevDisplacement, timeStep, nullptr, &forceMinus, nullptr);
  real const dForceDDisplacementFD = (forcePlus - forceMinus) / (2_r * kEpsilon);

  // Verify that Stiffness is the derivative of Force w.r.t. displacement
  if (Abs(stiffness) > 1e-10_r) {
    EXPECT_NEAR_RTOL(dForceDDisplacementFD, stiffness, kRelativeTolerance);
  } else {
    // Use absolute tolerance for near-zero values
    EXPECT_NEAR(dForceDDisplacementFD, stiffness, 1e-6_r);
  }
}

} // namespace

/*************************************************************************************************/
// DisplacementControlActuator Tests
/*************************************************************************************************/

TEST(TransmissionActuator, DisplacementControlActuator_PositiveDisplacement) {
  real const targetDisplacement = 0.01_r;
  real const stiffness = 1e5_r;
  real const damping = 1e3_r;
  DisplacementControlActuator actuator(targetDisplacement, stiffness, damping);

  real const displacement = 0.02_r;
  real const prevDisplacement = 0.01_r;
  real const timeStep = 0.01_r;

  VerifyDerivatives(&actuator, displacement, prevDisplacement, timeStep);
}

TEST(TransmissionActuator, DisplacementControlActuator_NegativeDisplacement) {
  real const targetDisplacement = 0.01_r;
  real const stiffness = 1e5_r;
  real const damping = 1e3_r;
  DisplacementControlActuator actuator(targetDisplacement, stiffness, damping);

  real const displacement = -0.01_r;
  real const prevDisplacement = 0_r;
  real const timeStep = 0.01_r;

  VerifyDerivatives(&actuator, displacement, prevDisplacement, timeStep);
}

TEST(TransmissionActuator, DisplacementControlActuator_CompressionState) {
  // Test when transmission is in compression (displacement < targetDisplacement).
  real const targetDisplacement = 0.02_r;
  real const stiffness = 1e5_r;
  // Note that damping would complicate this, since we could be below the target but still have
  // force, depending on the sign of the velocity. Using zero damping in this test.
  real const damping = 0_r;
  DisplacementControlActuator actuator(targetDisplacement, stiffness, damping);

  real const displacement = 0.01_r; // Below target, in compression
  real const prevDisplacement = 0_r;
  real const timeStep = 0.01_r;

  // In compression with the default `allowCompressiveForce = false`, energy, force, and stiffness
  // should all be zero.
  real energy{}, force{}, computedStiffness{};
  actuator.EnergyGradientHessian(
      displacement, prevDisplacement, timeStep, &energy, &force, &computedStiffness);
  EXPECT_NEAR(energy, 0_r, 1e-10_r);
  EXPECT_NEAR(force, 0_r, 1e-10_r);
  EXPECT_NEAR(computedStiffness, 0_r, 1e-10_r);

  // Derivatives should still be consistent
  VerifyDerivatives(&actuator, displacement, prevDisplacement, timeStep);
}

TEST(TransmissionActuator, DisplacementControlActuator_MultipleDisplacements) {
  real const targetDisplacement = 0.01_r;
  real const stiffness = 1e5_r;
  real const damping = 1e3_r;
  DisplacementControlActuator actuator(targetDisplacement, stiffness, damping);

  real const timeStep = 0.01_r;
  real const prevDisplacement = 0.005_r;

  // Test a range of displacements
  DynamicArray<real> displacements = {-0.01_r, -0.005_r, 0.005_r, 0.015_r, 0.02_r, 0.03_r};

  for (real displacement : displacements) {
    VerifyDerivatives(&actuator, displacement, prevDisplacement, timeStep);
  }
}

TEST(TransmissionActuator, DisplacementControlActuator_StateVariables) {
  real const initialTarget = 0.015_r;
  real const stiffness = 1e5_r;
  real const damping = 0_r;
  DisplacementControlActuator actuator(initialTarget, stiffness, damping);

  EXPECT_EQ(actuator.GetNumStateVariables(), 1);

  // Verify initial state variable can be retrieved (should be bitwise identical)
  DynamicArray<real> outStateVars(1);
  actuator.GetStateVariables(MakeSpan(outStateVars));
  EXPECT_EQ(outStateVars[0], initialTarget);

  // Set new state variable and verify it can be retrieved
  real const newTarget = -0.03_r;
  DynamicArray<real> newStateVars = {newTarget};
  actuator.SetStateVariables(MakeConstSpan(newStateVars), test::ExpectOK{});

  actuator.GetStateVariables(MakeSpan(outStateVars));
  EXPECT_EQ(outStateVars[0], newTarget);
}

TEST(TransmissionActuator, DisplacementControlActuator_AllowCompressiveForce) {
  // Same parameters as DisplacementControlActuator_CompressionState but with compression allowed,
  // verifying that a compressive force is now produced rather than clamped.
  real const targetDisplacement = 0.02_r;
  real const stiffness = 1e5_r;
  real const damping = 0_r;
  bool const allowCompressiveForce = true;
  DisplacementControlActuator actuator(
      targetDisplacement, stiffness, damping, allowCompressiveForce);

  real const displacement = 0.01_r; // Below target, in compression
  real const prevDisplacement = 0_r;
  real const timeStep = 0.01_r;

  real energy{}, force{}, computedStiffness{};
  actuator.EnergyGradientHessian(
      displacement, prevDisplacement, timeStep, &energy, &force, &computedStiffness);

  // With compressive force allowed, expect negative force and non-zero energy/stiffness.
  real const dDisplacement = displacement - targetDisplacement;
  real const expectedForce = stiffness * dDisplacement;
  EXPECT_NEAR_RTOL(force, expectedForce, kRelativeTolerance);
  EXPECT_GT(energy, 0_r);
  EXPECT_NEAR_RTOL(computedStiffness, stiffness, kRelativeTolerance);

  // Derivatives should still be consistent through the (formerly clamped) region.
  VerifyDerivatives(&actuator, displacement, prevDisplacement, timeStep);
}

/*************************************************************************************************/
// ForceControlActuator Tests
/*************************************************************************************************/

TEST(TransmissionActuator, ForceControlActuator_PositiveForce) {
  real const force = 100_r;
  ForceControlActuator actuator(force, test::ExpectOK{});

  real const displacement = 0.01_r;
  real const prevDisplacement = 0_r;
  real const timeStep = 0.01_r;

  VerifyDerivatives(&actuator, displacement, prevDisplacement, timeStep);
}

TEST(TransmissionActuator, ForceControlActuator_ZeroForce) {
  real const force = 0_r;
  ForceControlActuator actuator(force, test::ExpectOK{});

  real const displacement = 0.01_r;
  real const prevDisplacement = 0_r;
  real const timeStep = 0.01_r;

  real computedForce{}, computedStiffness{};
  actuator.EnergyGradientHessian(
      displacement, prevDisplacement, timeStep, nullptr, &computedForce, &computedStiffness);
  EXPECT_NEAR(computedForce, 0_r, 1e-10_r);
  EXPECT_NEAR(computedStiffness, 0_r, 1e-10_r);

  VerifyDerivatives(&actuator, displacement, prevDisplacement, timeStep);
}

TEST(TransmissionActuator, ForceControlActuator_MultipleDisplacements) {
  real const force = 100_r;
  ForceControlActuator actuator(force, test::ExpectOK{});

  real const timeStep = 0.01_r;
  real const prevDisplacement = 0_r;

  // Test various displacements - force should be constant
  DynamicArray<real> displacements = {-0.01_r, 0.01_r, 0.02_r, 0.05_r};

  for (real displacement : displacements) {
    real computedForce{}, computedStiffness{};
    actuator.EnergyGradientHessian(
        displacement, prevDisplacement, timeStep, nullptr, &computedForce, &computedStiffness);
    EXPECT_NEAR(computedForce, force, 1e-10_r);
    EXPECT_NEAR(computedStiffness, 0_r, 1e-10_r);
    VerifyDerivatives(&actuator, displacement, prevDisplacement, timeStep);
  }
}

TEST(TransmissionActuator, ForceControlActuator_StateVariables) {
  real const initialForce = 150_r;
  ForceControlActuator actuator(initialForce, test::ExpectOK{});

  EXPECT_EQ(actuator.GetNumStateVariables(), 1);

  // Verify initial state variable can be retrieved (should be bitwise identical)
  DynamicArray<real> outStateVars(1);
  actuator.GetStateVariables(MakeSpan(outStateVars));
  EXPECT_EQ(outStateVars[0], initialForce);

  // Set new state variable and verify it can be retrieved
  real const newForce = 300_r;
  DynamicArray<real> newStateVars = {newForce};
  actuator.SetStateVariables(MakeConstSpan(newStateVars), test::ExpectOK{});

  actuator.GetStateVariables(MakeSpan(outStateVars));
  EXPECT_EQ(outStateVars[0], newForce);

  DynamicArray<real> compressiveStateVars = {-200_r};
  actuator.SetStateVariables(MakeConstSpan(compressiveStateVars), test::ExpectNotOK{});
  actuator.GetStateVariables(MakeSpan(outStateVars));
  EXPECT_EQ(outStateVars[0], newForce);
}

TEST(TransmissionActuator, ForceControlActuator_AllowCompressiveForce) {
  // With the default flag, negative forces are rejected with an error.
  ForceControlActuator(-50_r, test::ExpectNotOK{});

  // With allowCompressiveForce = true, negative forces are accepted and round-tripped through
  // SetStateVariables / GetStateVariables.
  bool const allowCompressiveForce = true;
  real const initialForce = -75_r;
  ForceControlActuator actuator(initialForce, test::ExpectOK{}, allowCompressiveForce);

  DynamicArray<real> outStateVars(1);
  actuator.GetStateVariables(MakeSpan(outStateVars));
  EXPECT_EQ(outStateVars[0], initialForce);

  // Setting a negative state variable should not produce an error either.
  real const newForce = -200_r;
  DynamicArray<real> newStateVars = {newForce};
  actuator.SetStateVariables(MakeConstSpan(newStateVars), test::ExpectOK{});
  actuator.GetStateVariables(MakeSpan(outStateVars));
  EXPECT_EQ(outStateVars[0], newForce);

  // The reported gradient should be the (negative) force itself, unclamped.
  real const displacement = 0.01_r;
  real computedForce{};
  actuator.EnergyGradientHessian(displacement, 0_r, 0.01_r, nullptr, &computedForce, nullptr);
  EXPECT_EQ(computedForce, newForce);
}

/*************************************************************************************************/
// McKibbenActuator Tests
/*************************************************************************************************/

TEST(TransmissionActuator, McKibbenActuator_PositivePressure) {
  real const pressure = 1e5_r;
  real const minimumPressure = 0_r;
  real const threadLength = 0.1_r;
  real const numberOfWraps = 10_r;
  real const deflatedStiffness = 1e3_r;
  real const deflatedEquilibriumLength = 0.05_r;

  McKibbenActuator actuator(
      pressure,
      minimumPressure,
      threadLength,
      numberOfWraps,
      deflatedStiffness,
      deflatedEquilibriumLength);

  real const displacement = 0.01_r;
  real const prevDisplacement = 0_r;
  real const timeStep = 0.01_r;

  VerifyDerivatives(&actuator, displacement, prevDisplacement, timeStep);
}

TEST(TransmissionActuator, McKibbenActuator_BelowMinimumPressure) {
  real const pressure = 50_r; // Below minimum
  real const minimumPressure = 100_r;
  real const threadLength = 0.1_r;
  real const numberOfWraps = 10_r;
  real const deflatedStiffness = 1e3_r;
  real const deflatedEquilibriumLength = 0.05_r;

  McKibbenActuator actuator(
      pressure,
      minimumPressure,
      threadLength,
      numberOfWraps,
      deflatedStiffness,
      deflatedEquilibriumLength);

  real const displacement = 0.01_r;
  real const prevDisplacement = 0_r;
  real const timeStep = 0.01_r;

  // When pressure is below minimum, only deflated stiffness contributes
  real const expectedForce = deflatedStiffness * displacement;
  real computedForce{};
  actuator.EnergyGradientHessian(
      displacement, prevDisplacement, timeStep, nullptr, &computedForce, nullptr);
  EXPECT_NEAR(computedForce, expectedForce, 1e-6_r);

  VerifyDerivatives(&actuator, displacement, prevDisplacement, timeStep);
}

TEST(TransmissionActuator, McKibbenActuator_NegativeDisplacement) {
  real const pressure = 1e5_r;
  real const minimumPressure = 0_r;
  real const threadLength = 0.1_r;
  real const numberOfWraps = 10_r;
  real const deflatedStiffness = 1e3_r;
  real const deflatedEquilibriumLength = 0.05_r;

  McKibbenActuator actuator(
      pressure,
      minimumPressure,
      threadLength,
      numberOfWraps,
      deflatedStiffness,
      deflatedEquilibriumLength);

  real const displacement = -0.01_r; // Compression
  real const prevDisplacement = 0_r;
  real const timeStep = 0.01_r;

  // In compression, energy and force should be zero
  real energy{}, force{}, stiffness{};
  actuator.EnergyGradientHessian(
      displacement, prevDisplacement, timeStep, &energy, &force, &stiffness);
  EXPECT_NEAR(energy, 0_r, 1e-10_r);
  EXPECT_NEAR(force, 0_r, 1e-10_r);
  EXPECT_NEAR(stiffness, 0_r, 1e-10_r);

  VerifyDerivatives(&actuator, displacement, prevDisplacement, timeStep);
}

TEST(TransmissionActuator, McKibbenActuator_MultipleDisplacements) {
  real const pressure = 1e5_r;
  real const minimumPressure = 0_r;
  real const threadLength = 0.1_r;
  real const numberOfWraps = 10_r;
  real const deflatedStiffness = 1e3_r;
  real const deflatedEquilibriumLength = 0.05_r;

  McKibbenActuator actuator(
      pressure,
      minimumPressure,
      threadLength,
      numberOfWraps,
      deflatedStiffness,
      deflatedEquilibriumLength);

  real const timeStep = 0.01_r;
  real const prevDisplacement = 0_r;

  // Test a range of displacements
  DynamicArray<real> displacements = {0.001_r, 0.005_r, 0.01_r, 0.02_r, 0.03_r};

  for (real displacement : displacements) {
    VerifyDerivatives(&actuator, displacement, prevDisplacement, timeStep);
  }
}

TEST(TransmissionActuator, McKibbenActuator_StateVariables) {
  real const initialPressure = 1.5e5_r;
  real const minimumPressure = 0_r;
  real const threadLength = 0.1_r;
  real const numberOfWraps = 10_r;
  real const deflatedStiffness = 1e3_r;
  real const deflatedEquilibriumLength = 0.05_r;

  McKibbenActuator actuator(
      initialPressure,
      minimumPressure,
      threadLength,
      numberOfWraps,
      deflatedStiffness,
      deflatedEquilibriumLength);

  EXPECT_EQ(actuator.GetNumStateVariables(), 1);

  // Verify initial state variable can be retrieved (should be bitwise identical)
  DynamicArray<real> outStateVars(1);
  actuator.GetStateVariables(MakeSpan(outStateVars));
  EXPECT_EQ(outStateVars[0], initialPressure);

  // Set new state variable and verify it can be retrieved
  real const newPressure = 3e5_r;
  DynamicArray<real> newStateVars = {newPressure};
  actuator.SetStateVariables(MakeConstSpan(newStateVars), test::ExpectOK{});

  actuator.GetStateVariables(MakeSpan(outStateVars));
  EXPECT_EQ(outStateVars[0], newPressure);
}

/*************************************************************************************************/
// Transmission Displacement Jacobian Tests
/*************************************************************************************************/

TEST(Transmission, LinearTransmission_DisplacementJacobianOverwritesAndAccumulatesTerms) {
  DynamicArray<ArticulatedJointType> const jointTypes(3, ArticulatedJointType::Revolute);
  auto const dofInfo = SetupJointDofs(MakeConstSpan(jointTypes));
  auto const poseInfo = SetupJointPose(MakeConstSpan(jointTypes));
  DynamicArray<int> const jointIndices = {2, 0, 2};
  DynamicArray<real> const jointCoefficients = {0.4_r, -0.3_r, 0.6_r};
  LinearTransmission const transmission(
      MakeConstSpan(jointIndices),
      MakeConstSpan(jointCoefficients),
      MakeConstSpan(dofInfo),
      MakeConstSpan(poseInfo),
      {});

  RowMatrix<real> const bodyJacobian = RowMatrix<real>::Zero(0, 3);
  DynamicArray<real> jacobian(3, 911_r);
  transmission.DisplacementJacobian({}, AsConstView(bodyJacobian), MakeSpan(jacobian));

  DynamicArray<real> const expected = {-0.3_r, 0_r, 1_r};
  EXPECT_SPAN_EQ(MakeConstSpan(jacobian), MakeConstSpan(expected));
}

/*************************************************************************************************/
// Transmission Residual Derivative Tests
/*************************************************************************************************/

namespace {

// Helper function to verify that Residual is the gradient of Energy w.r.t. currentPose
void VerifyResidualIsEnergyGradient(
    Transmission const* transmission,
    Span<real const> currentPose,
    Span<real const> stageStartPose,
    real timeStep,
    Span<TransformRT const> currentLinkTransforms = {},
    Span<TransformRT const> stageStartLinkTransforms = {},
    RowMatrixView<real const> bodyJacobian = {},
    int numDirections = 5) {
  int const numDofs = isize(currentPose);

  // Compute energy and residual at current pose
  AssemblyParams assemblyParams{.assemObj = true, .assemRes = true, .assemDRes = false};
  double energy = 0.0;
  DynamicArray<real> residual(numDofs, 0_r);
  Matrix<real> dresidual = Matrix<real>::Zero(numDofs, numDofs);
  transmission->AddObjResDRes(
      currentLinkTransforms,
      stageStartLinkTransforms,
      bodyJacobian,
      currentPose,
      stageStartPose,
      timeStep,
      assemblyParams,
      energy,
      MakeSpan(residual),
      dresidual);

  // Random number generator for perturbation directions
  int constexpr kSeed = 54321;
  mochi_default_random_engine randomGenerator = RandomGenerator(kSeed);

  // Test with multiple random perturbation directions
  for (int testIter = 0; testIter < numDirections; ++testIter) {
    // Create a random perturbation direction
    DynamicArray<real> direction(numDofs);
    for (int i = 0; i < numDofs; ++i) {
      direction[i] = RandomUniformValue(randomGenerator, -1_r, 1_r);
    }

    // Normalize the direction for better numerical stability
    real directionNorm = 0_r;
    for (int i = 0; i < numDofs; ++i) {
      directionNorm += direction[i] * direction[i];
    }
    directionNorm = Sqrt(directionNorm);
    if (directionNorm > 1e-10_r) {
      for (int i = 0; i < numDofs; ++i) {
        direction[i] /= directionNorm;
      }
    }

    // Compute directional derivative using residual: residual · direction
    real directionalDerivative = 0_r;
    for (int i = 0; i < numDofs; ++i) {
      directionalDerivative += residual[i] * direction[i];
    }

    // Compute directional derivative using finite differences
    DynamicArray<real> perturbedPose(currentPose.begin(), currentPose.end());
    for (int i = 0; i < numDofs; ++i) {
      perturbedPose[i] += kEpsilon * direction[i];
    }

    double perturbedEnergy = 0.0;
    DynamicArray<real> dummyResidual(numDofs, 0_r);
    Matrix<real> dummyDResidual = Matrix<real>::Zero(numDofs, numDofs);
    transmission->AddObjResDRes(
        currentLinkTransforms,
        stageStartLinkTransforms,
        bodyJacobian,
        MakeConstSpan(perturbedPose),
        stageStartPose,
        timeStep,
        assemblyParams,
        perturbedEnergy,
        MakeSpan(dummyResidual),
        dummyDResidual);

    real const directionalDerivativeFD = (perturbedEnergy - energy) / kEpsilon;

    // Compare directional derivatives
    if (Abs(directionalDerivative) > 1e-8_r) {
      EXPECT_NEAR_RTOL(directionalDerivativeFD, directionalDerivative, kRelativeTolerance);
    } else {
      // Use absolute tolerance for near-zero values
      EXPECT_NEAR(directionalDerivativeFD, directionalDerivative, 1e-5_r);
    }
  }
}

// Helper function to verify that DResidual is the directional derivative of Residual w.r.t.
// currentPose
void VerifyDResidualDerivative(
    Transmission const* transmission,
    Span<real const> currentPose,
    Span<real const> stageStartPose,
    real timeStep,
    Span<TransformRT const> currentLinkTransforms = {},
    Span<TransformRT const> stageStartLinkTransforms = {},
    RowMatrixView<real const> bodyJacobian = {},
    int numDirections = 5) {
  int const numDofs = isize(currentPose);

  // Compute residual and dresidual at current pose
  AssemblyParams assemblyParams{.assemObj = false, .assemRes = true, .assemDRes = true};
  double energy = 0.0;
  DynamicArray<real> residual(numDofs, 0_r);
  Matrix<real> dresidual = Matrix<real>::Zero(numDofs, numDofs);
  transmission->AddObjResDRes(
      currentLinkTransforms,
      stageStartLinkTransforms,
      bodyJacobian,
      currentPose,
      stageStartPose,
      timeStep,
      assemblyParams,
      energy,
      MakeSpan(residual),
      dresidual);

  // Random number generator for perturbation directions
  int constexpr kSeed = 12345;
  mochi_default_random_engine randomGenerator = RandomGenerator(kSeed);

  // Test with multiple random perturbation directions
  for (int testIter = 0; testIter < numDirections; ++testIter) {
    // Create a random perturbation direction
    DynamicArray<real> direction(numDofs);
    for (int i = 0; i < numDofs; ++i) {
      direction[i] = RandomUniformValue(randomGenerator, -1_r, 1_r);
    }

    // Normalize the direction for better numerical stability
    real directionNorm = 0_r;
    for (int i = 0; i < numDofs; ++i) {
      directionNorm += direction[i] * direction[i];
    }
    directionNorm = Sqrt(directionNorm);
    if (directionNorm > 1e-10_r) {
      for (int i = 0; i < numDofs; ++i) {
        direction[i] /= directionNorm;
      }
    }

    // Compute directional derivative using dresidual: dresidual * direction
    DynamicArray<real> directionalDerivative(numDofs, 0_r);
    for (int i = 0; i < numDofs; ++i) {
      for (int j = 0; j < numDofs; ++j) {
        directionalDerivative[i] += dresidual(i, j) * direction[j];
      }
    }

    // Compute directional derivative using finite differences
    DynamicArray<real> perturbedPose(currentPose.begin(), currentPose.end());
    for (int i = 0; i < numDofs; ++i) {
      perturbedPose[i] += kEpsilon * direction[i];
    }

    double perturbedEnergy = 0.0;
    DynamicArray<real> perturbedResidual(numDofs, 0_r);
    Matrix<real> dummyDResidual = Matrix<real>::Zero(numDofs, numDofs);
    transmission->AddObjResDRes(
        currentLinkTransforms,
        stageStartLinkTransforms,
        bodyJacobian,
        MakeConstSpan(perturbedPose),
        stageStartPose,
        timeStep,
        assemblyParams,
        perturbedEnergy,
        MakeSpan(perturbedResidual),
        dummyDResidual);

    DynamicArray<real> directionalDerivativeFD(numDofs);
    for (int i = 0; i < numDofs; ++i) {
      directionalDerivativeFD[i] = (perturbedResidual[i] - residual[i]) / kEpsilon;
    }

    // Compare directional derivatives
    for (int i = 0; i < numDofs; ++i) {
      if (Abs(directionalDerivative[i]) > 1e-8_r) {
        EXPECT_NEAR_RTOL(directionalDerivativeFD[i], directionalDerivative[i], kRelativeTolerance);
      } else {
        // Use absolute tolerance for near-zero values
        EXPECT_NEAR(directionalDerivativeFD[i], directionalDerivative[i], 1e-5_r);
      }
    }
  }
}

} // namespace

TEST(Transmission, LinearTransmission_DResidualDerivative_SingleRevoluteJoint) {
  // Create a simple articulated body with a single revolute joint
  DynamicArray<ArticulatedJointType> jointTypes = {ArticulatedJointType::Revolute};
  auto dofInfo = SetupJointDofs(MakeConstSpan(jointTypes));
  auto poseInfo = SetupJointPose(MakeConstSpan(jointTypes));

  // Create a LinearTransmission that wraps around the revolute joint
  DynamicArray<int> jointIndices = {0};
  DynamicArray<real> jointCoefficients = {0.05_r};

  // Create a DisplacementControlActuator with target displacement
  real const targetDisplacement = 0.01_r;
  real const stiffness = 1e1_r;
  real const damping = 1_r;
  auto actuator =
      std::make_unique<DisplacementControlActuator>(targetDisplacement, stiffness, damping);

  LinearTransmission transmission(
      MakeConstSpan(jointIndices),
      MakeConstSpan(jointCoefficients),
      MakeConstSpan(dofInfo),
      MakeConstSpan(poseInfo),
      std::move(actuator));

  // Set up poses (single revolute joint has 1 DOF - rotation angle)
  real const currentAngle = 0.3_r;
  real const previousAngle = 0.2_r;
  real const timeStep = 0.01_r;

  DynamicArray<real> currentPose = {currentAngle};
  DynamicArray<real> previousPose = {previousAngle};

  // Verify that AddResidualContribution is the gradient of Energy
  VerifyResidualIsEnergyGradient(
      &transmission, MakeConstSpan(currentPose), MakeConstSpan(previousPose), timeStep);

  // Verify that AddDResidualContribution correctly computes the derivative
  VerifyDResidualDerivative(
      &transmission, MakeConstSpan(currentPose), MakeConstSpan(previousPose), timeStep);
}

TEST(Transmission, LinearTransmission_DResidualDerivative_TwoRevoluteJoints) {
  // Create an articulated body with two revolute joints
  DynamicArray<ArticulatedJointType> jointTypes = {
      ArticulatedJointType::Revolute, // Joint 0: rotate around z-axis
      ArticulatedJointType::Revolute // Joint 1: rotate around z-axis
  };
  auto dofInfo = SetupJointDofs(MakeConstSpan(jointTypes));
  auto poseInfo = SetupJointPose(MakeConstSpan(jointTypes));

  // Create a LinearTransmission that wraps around both joints
  DynamicArray<int> jointIndices = {0, 1};
  DynamicArray<real> jointCoefficients = {0.05_r, 0.04_r}; // Different coefficients

  // Create a DisplacementControlActuator
  real const targetDisplacement = 0.02_r;
  real const stiffness = 5e1_r;
  real const damping = 1e-1_r;
  auto actuator =
      std::make_unique<DisplacementControlActuator>(targetDisplacement, stiffness, damping);

  LinearTransmission transmission(
      MakeConstSpan(jointIndices),
      MakeConstSpan(jointCoefficients),
      MakeConstSpan(dofInfo),
      MakeConstSpan(poseInfo),
      std::move(actuator));

  // Set up poses (two joints = 2 DOFs)
  DynamicArray<real> currentPose = {0.4_r, 0.6_r}; // Angles for joints 0 and 1
  DynamicArray<real> previousPose = {0.3_r, 0.5_r};
  real const timeStep = 0.01_r;

  // Verify that AddResidualContribution is the gradient of Energy
  VerifyResidualIsEnergyGradient(
      &transmission, MakeConstSpan(currentPose), MakeConstSpan(previousPose), timeStep);

  // Verify derivatives
  VerifyDResidualDerivative(
      &transmission, MakeConstSpan(currentPose), MakeConstSpan(previousPose), timeStep);
}

TEST(Transmission, LinearTransmission_GetTerms) {
  // Three joints so the transmission's joint indices differ from term positions; this verifies
  // GetTerms reports the original joint indices passed to the constructor (not 0..N-1), each
  // paired with its coefficient.
  DynamicArray<ArticulatedJointType> jointTypes = {
      ArticulatedJointType::Revolute,
      ArticulatedJointType::Revolute,
      ArticulatedJointType::Revolute};
  auto const dofInfo = SetupJointDofs(MakeConstSpan(jointTypes));
  auto const poseInfo = SetupJointPose(MakeConstSpan(jointTypes));

  DynamicArray<int> const jointIndices = {2, 0};
  DynamicArray<real> const jointCoefficients = {0.05_r, -0.03_r};

  LinearTransmission const transmission(
      MakeConstSpan(jointIndices),
      MakeConstSpan(jointCoefficients),
      MakeConstSpan(dofInfo),
      MakeConstSpan(poseInfo));

  DynamicArray<int> gotIndices;
  DynamicArray<real> gotCoefficients;
  for (auto const& term : transmission.GetTerms()) {
    gotIndices.push_back(term.jointIndex);
    gotCoefficients.push_back(term.coefficient);
  }

  EXPECT_SPAN_EQ(MakeConstSpan(gotIndices), MakeConstSpan(jointIndices));
  EXPECT_SPAN_EQ(MakeConstSpan(gotCoefficients), MakeConstSpan(jointCoefficients));
}

TEST(Transmission, LinearTransmission_DResidualDerivative_MultiplePoses) {
  // Test with various poses to ensure robustness
  DynamicArray<ArticulatedJointType> jointTypes = {ArticulatedJointType::Revolute};
  auto dofInfo = SetupJointDofs(MakeConstSpan(jointTypes));
  auto poseInfo = SetupJointPose(MakeConstSpan(jointTypes));

  DynamicArray<int> jointIndices = {0};
  DynamicArray<real> jointCoefficients = {0.05_r};

  real const targetDisplacement = 0.01_r;
  real const stiffness = 1e1_r;
  real const damping = 1e-1_r;
  auto actuator =
      std::make_unique<DisplacementControlActuator>(targetDisplacement, stiffness, damping);

  LinearTransmission transmission(
      MakeConstSpan(jointIndices),
      MakeConstSpan(jointCoefficients),
      MakeConstSpan(dofInfo),
      MakeConstSpan(poseInfo),
      std::move(actuator));

  real const timeStep = 0.01_r;
  real const previousAngle = 0.1_r;
  DynamicArray<real> previousPose = {previousAngle};

  // Test at multiple current angles; avoid non-differentiable behavior at 0.2, where the
  // transmission crosses the target displacement and switches between tension and compression.
  DynamicArray<real> testAngles = {-0.5_r, -0.2_r, 0_r, 0.5_r, 1.0_r};

  for (real angle : testAngles) {
    DynamicArray<real> currentPose = {angle};

    // Verify that AddResidualContribution is the gradient of Energy
    VerifyResidualIsEnergyGradient(
        &transmission, MakeConstSpan(currentPose), MakeConstSpan(previousPose), timeStep);

    // Verify that AddDResidualContribution correctly computes the derivative
    VerifyDResidualDerivative(
        &transmission, MakeConstSpan(currentPose), MakeConstSpan(previousPose), timeStep);
  }
}

/*************************************************************************************************/
// SpatialTendon Tests
/*************************************************************************************************/

namespace {

// SpatialTendon::Displacement is nonlinear (sqrt-based), so finite differences need a larger step
// than the linear-transmission tests to balance truncation and roundoff. Step and tolerances are
// precision-aware: tight in double, looser in single.
constexpr real kSpatialFdEpsilon = MOCHI_USE_DOUBLE_PRECISION ? 1e-6_r : 2e-3_r;
constexpr real kSpatialRelTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-5_r : 1.5e-2_r;
constexpr real kSpatialAbsTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-8_r : 1e-3_r;

// A 2-link planar chain, both joints revolute about +z. Link 0's joint is at the origin; link 1's
// joint origin is offset by `linkOffset` along x in link 0's frame. At the rest pose both link
// frames are axis-aligned, with link 1 translated by (linkOffset, 0, 0).
struct TwoLinkChain {
  ParentIndexArray parents;
  DynamicArray<ArticulatedJointType> jointTypes;
  DynamicArray<Real3> jointAxes;
  DynamicArray<ArticulatedDofInfo> dofInfo;
  DynamicArray<ArticulatedPoseInfo> poseInfo;
  RestTransformArray restTransforms;
};

TwoLinkChain MakeTwoLinkRevoluteChain(real linkOffset) {
  TwoLinkChain chain;
  chain.parents = {-1, 0};
  chain.jointTypes = {ArticulatedJointType::Revolute, ArticulatedJointType::Revolute};
  chain.jointAxes = {Real3{0_r, 0_r, 1_r}, Real3{0_r, 0_r, 1_r}};
  chain.dofInfo = SetupJointDofs(MakeConstSpan(chain.jointTypes));
  chain.poseInfo = SetupJointPose(MakeConstSpan(chain.jointTypes));
  chain.restTransforms = {
      {TransformRT::Identity(), TransformRT::Identity()},
      {TransformRT::Identity(), TransformRT(Quaternion::Identity(), Real3{linkOffset, 0_r, 0_r})}};
  return chain;
}

// Computes link transforms at the given reduced pose via forward kinematics on the test chain.
// Matches what the engine supplies to Displacement / AddObjResDRes.
DynamicArray<TransformRT> ComputeLinkTransforms(TwoLinkChain const& chain, Span<real const> pose) {
  int const numLinks = isize(chain.parents);
  DynamicArray<TransformRT> jointTransforms(numLinks);
  DynamicArray<TransformRT> linkTransforms(numLinks);
  ComputeTransformsFromReducedPose(
      MakeConstSpan(chain.jointTypes),
      MakeConstSpan(chain.jointAxes),
      MakeConstSpan(chain.poseInfo),
      MakeConstSpan(chain.parents),
      MakeConstSpan(chain.restTransforms),
      TransformRT::Identity(),
      AsConstView(pose),
      MakeSpan(jointTransforms),
      MakeSpan(linkTransforms));
  return linkTransforms;
}

// Computes the body Jacobian (6 * numLinks rows, reducedDofs cols, Lie/angular convention) at the
// given reduced pose, matching what the physics engine supplies during assembly.
RowMatrix<real> ComputeBodyJacobian(TwoLinkChain const& chain, Span<real const> pose) {
  int const numLinks = isize(chain.parents);
  int const reducedDofs = GetReducedDofsSize(MakeConstSpan(chain.dofInfo));
  DynamicArray<TransformRT> jointTransforms(numLinks);
  DynamicArray<TransformRT> linkTransforms(numLinks);
  ComputeTransformsFromReducedPose(
      MakeConstSpan(chain.jointTypes),
      MakeConstSpan(chain.jointAxes),
      MakeConstSpan(chain.poseInfo),
      MakeConstSpan(chain.parents),
      MakeConstSpan(chain.restTransforms),
      TransformRT::Identity(),
      AsConstView(pose),
      MakeSpan(jointTransforms),
      MakeSpan(linkTransforms));
  RowMatrix<real> jacobian = CreateJacobianStorage(numLinks * RigidSize::kDAll, reducedDofs);
  Jacobian(
      MakeConstSpan(chain.jointTypes),
      MakeConstSpan(chain.parents),
      MakeConstSpan(chain.jointAxes),
      MakeConstSpan(chain.dofInfo),
      MakeConstSpan(chain.restTransforms),
      TransformRT::Identity(),
      MakeConstSpan(jointTransforms),
      MakeConstSpan(linkTransforms),
      jacobian);
  return jacobian;
}

// Central finite-difference gradient of SpatialTendon::Displacement w.r.t. the reduced pose. Valid
// for the test chain because revolute joints have pose == dofs. Re-runs FK at every perturbed pose
// since Displacement now consumes link transforms directly.
DynamicArray<real> FiniteDiffDisplacementGradient(
    SpatialTendon const& tendon,
    TwoLinkChain const& chain,
    Span<real const> pose) {
  int const numDofs = isize(pose);
  DynamicArray<real> gradient(numDofs, 0_r);
  DynamicArray<real> perturbed(pose.begin(), pose.end());
  for (int i = 0; i < numDofs; ++i) {
    perturbed[i] = pose[i] + kSpatialFdEpsilon;
    DynamicArray<TransformRT> const linkTransformsPlus =
        ComputeLinkTransforms(chain, MakeConstSpan(perturbed));
    real const lengthPlus =
        tendon.Displacement(MakeConstSpan(linkTransformsPlus), MakeConstSpan(perturbed));
    perturbed[i] = pose[i] - kSpatialFdEpsilon;
    DynamicArray<TransformRT> const linkTransformsMinus =
        ComputeLinkTransforms(chain, MakeConstSpan(perturbed));
    real const lengthMinus =
        tendon.Displacement(MakeConstSpan(linkTransformsMinus), MakeConstSpan(perturbed));
    perturbed[i] = pose[i];
    gradient[i] = (lengthPlus - lengthMinus) / (2_r * kSpatialFdEpsilon);
  }
  return gradient;
}

// Builds a waypoint-only element list from parallel link-index / local-position arrays.
DynamicArray<RoutingElement> MakeWaypointElements(
    std::initializer_list<int> const links,
    std::initializer_list<Real3> const locals) {
  MOCHI_ASSERT(links.size() == locals.size());
  DynamicArray<RoutingElement> routingElements;
  routingElements.reserve(links.size());
  auto const* localIt = locals.begin();
  for (int link : links) {
    routingElements.push_back(
        {.type = RoutingElementType::Waypoint, .index = link, .localPosition = *localIt});
    ++localIt;
  }
  return routingElements;
}

// Builds a linear-joint element with the given joint index and signed coefficient.
RoutingElement LinearJointElement(int jointIndex, real coefficient) {
  return {.type = RoutingElementType::LinearJoint, .index = jointIndex, .coefficient = coefficient};
}

// Closed-form Displacement (routed length minus rest length) for a two-waypoint tendon on the test
// chain, with waypoint 0 at (a, 0, 0) on link 0 and waypoint 1 at (b, 0, 0) on link 1, link offset
// d, joint 0 held at 0, and joint 1 rotated by phi. At phi == 0 this is exactly zero (rest pose).
real AnalyticDisplacementSingleRevolute(real d, real a, real b, real phi) {
  real const restLength = d + b - a;
  real const px = (d - a) + b * static_cast<real>(std::cos(static_cast<double>(phi)));
  real const py = b * static_cast<real>(std::sin(static_cast<double>(phi)));
  return Sqrt(px * px + py * py) - restLength;
}

// Builds a SpatialTendon on the given chain from a routing list and optional actuator. The test
// chain has all-zero link CoMs, so the link's local (mesh-authoring) frame coincides with its
// CoM frame for every link; we pass a zero-filled `comLocals` array so the constructor's
// local->CoM conversion is a no-op and existing waypoint coordinates need no shift.
SpatialTendon MakeTendon(
    TwoLinkChain const& chain,
    Span<RoutingElement const> routingElements,
    std::unique_ptr<TransmissionActuator>&& actuator = {}) {
  DynamicArray<Real3> const comLocals(chain.parents.size(), Real3{});
  return {
      routingElements,
      MakeConstSpan(chain.jointTypes),
      MakeConstSpan(chain.jointAxes),
      MakeConstSpan(chain.dofInfo),
      MakeConstSpan(chain.poseInfo),
      MakeConstSpan(chain.parents),
      MakeConstSpan(chain.restTransforms),
      MakeConstSpan(comLocals),
      std::move(actuator)};
}

// Evaluates Transmission::Displacement on the chain at the given pose. Recomputes link transforms
// via FK so callers don't have to.
real DisplacementOf(
    Transmission const& transmission,
    TwoLinkChain const& chain,
    Span<real const> pose) {
  DynamicArray<TransformRT> const linkTransforms = ComputeLinkTransforms(chain, pose);
  return transmission.Displacement(MakeConstSpan(linkTransforms), pose);
}

// Objective / residual / dresidual produced by a single AddObjResDRes assembly.
struct AssembledTransmission {
  double objective = 0.0;
  DynamicArray<real> residual;
  Matrix<real> dresidual;
};

// Assembles a transmission's contribution on the chain, sizing the residual and dresidual outputs
// from the pose dimension. Computes link transforms at both the current and stage-start poses via
// FK so callers don't have to.
AssembledTransmission AssembleResDRes(
    Transmission const& transmission,
    TwoLinkChain const& chain,
    Span<real const> currentPose,
    Span<real const> stageStartPose,
    real timeStep,
    AssemblyParams const& params,
    RowMatrixView<real const> bodyJacobian) {
  int const numDofs = isize(currentPose);
  AssembledTransmission out;
  out.residual = DynamicArray<real>(numDofs, 0_r);
  out.dresidual = Matrix<real>::Zero(numDofs, numDofs);
  DynamicArray<TransformRT> const currentLinkTransforms = ComputeLinkTransforms(chain, currentPose);
  DynamicArray<TransformRT> const stageStartLinkTransforms =
      ComputeLinkTransforms(chain, stageStartPose);
  transmission.AddObjResDRes(
      MakeConstSpan(currentLinkTransforms),
      MakeConstSpan(stageStartLinkTransforms),
      bodyJacobian,
      currentPose,
      stageStartPose,
      timeStep,
      params,
      out.objective,
      MakeSpan(out.residual),
      out.dresidual);
  return out;
}

} // namespace

TEST(SpatialTendon, Displacement_MatchesAnalyticLength) {
  // The routed length is known in closed form: waypoint 0 sits at (a, 0, 0); link 1's origin is at
  // (d, 0, 0) and, with joint 1 rotated by phi, waypoint 1 sits at (d + b cos(phi), b sin(phi), 0).
  // Looping phi == 0 folds in the rest-pose-zero check (the closed form is exactly zero there).
  real const d = 1.0_r, a = 0.3_r, b = 0.5_r;
  TwoLinkChain const chain = MakeTwoLinkRevoluteChain(d);
  DynamicArray<RoutingElement> const routingElements =
      MakeWaypointElements({0, 1}, {Real3{a, 0_r, 0_r}, Real3{b, 0_r, 0_r}});
  SpatialTendon const tendon = MakeTendon(chain, MakeConstSpan(routingElements));

  for (real phi : {0_r, 0.5_r}) {
    DynamicArray<real> const pose = {0_r, phi};
    real const displacement = DisplacementOf(tendon, chain, MakeConstSpan(pose));
    EXPECT_NEAR(displacement, AnalyticDisplacementSingleRevolute(d, a, b, phi), 1e-4_r);
  }
}

TEST(SpatialTendon, Displacement_MixedElements_MatchesHandComputed) {
  // A waypoint segment plus a linear-joint element: Displacement = (routed length change) + coef *
  // angle.
  real const d = 1.0_r, a = 0.3_r, b = 0.5_r, coef = 0.7_r;
  TwoLinkChain const chain = MakeTwoLinkRevoluteChain(d);
  DynamicArray<RoutingElement> routingElements =
      MakeWaypointElements({0, 1}, {Real3{a, 0_r, 0_r}, Real3{b, 0_r, 0_r}});
  routingElements.push_back(
      LinearJointElement(1, coef)); // Append a constant moment arm on joint 1.
  SpatialTendon const tendon = MakeTendon(chain, MakeConstSpan(routingElements));

  real const phi = 0.5_r;
  DynamicArray<real> const pose = {0_r, phi};
  real const displacement = DisplacementOf(tendon, chain, MakeConstSpan(pose));

  // Linear-joint elements contribute 0 at rest, so the rest length is segment-only.
  EXPECT_NEAR(displacement, AnalyticDisplacementSingleRevolute(d, a, b, phi) + coef * phi, 1e-4_r);
}

TEST(SpatialTendon, DisplacementJacobian_MatchesFiniteDifferencesAndOverwrites) {
  TwoLinkChain const chain = MakeTwoLinkRevoluteChain(1.0_r);
  DynamicArray<RoutingElement> const waypointOnly =
      MakeWaypointElements({0, 1}, {Real3{0.3_r, 0_r, 0_r}, Real3{0.5_r, 0_r, 0_r}});
  DynamicArray<RoutingElement> mixed = waypointOnly;
  mixed.push_back(LinearJointElement(0, -0.6_r));
  DynamicArray<RoutingElement> const allLinear = {
      LinearJointElement(0, 0.5_r), LinearJointElement(1, -0.3_r)};
  DynamicArray<real> const pose = {0.2_r, 0.4_r};
  DynamicArray<TransformRT> const linkTransforms =
      ComputeLinkTransforms(chain, MakeConstSpan(pose));
  RowMatrix<real> const bodyJacobian = ComputeBodyJacobian(chain, MakeConstSpan(pose));

  for (DynamicArray<RoutingElement> const& routingElements : {waypointOnly, mixed, allLinear}) {
    SpatialTendon const tendon = MakeTendon(chain, MakeConstSpan(routingElements));
    DynamicArray<real> jacobian(isize(pose), 911_r);
    tendon.DisplacementJacobian(
        MakeConstSpan(linkTransforms), AsConstView(bodyJacobian), MakeSpan(jacobian));
    DynamicArray<real> const finiteDifference =
        FiniteDiffDisplacementGradient(tendon, chain, MakeConstSpan(pose));
    for (int i = 0; i < isize(pose); ++i) {
      EXPECT_NEAR(
          jacobian[i],
          finiteDifference[i],
          kSpatialRelTol * Abs(finiteDifference[i]) + kSpatialAbsTol);
    }
  }
}

TEST(SpatialTendon, Residual_MatchesLengthGradient) {
  // With a unit-force actuator the assembled residual equals force * d(Displacement)/dq, verified
  // against an independent finite-difference of Displacement. Checked for a waypoint-only routing,
  // a routing that mixes in a linear-joint element (so the gradient combines the waypoint lever-arm
  // Jacobian and the linear element's constant coefficient), and an all-linear routing with no
  // waypoint segments. The same assembly also exercises the objective branch: with a
  // ForceControlActuator the energy is the closed form force * displacement, so the accumulated
  // objective must equal force * Displacement (Displacement taken from the independently validated
  // helper).
  TwoLinkChain const chain = MakeTwoLinkRevoluteChain(1.0_r);
  DynamicArray<RoutingElement> const waypointOnly =
      MakeWaypointElements({0, 1}, {Real3{0.3_r, 0_r, 0_r}, Real3{0.5_r, 0_r, 0_r}});
  DynamicArray<RoutingElement> mixed = waypointOnly;
  mixed.push_back(LinearJointElement(0, -0.6_r));
  // An all-linear routing (no waypoint segments) exercises the linear-joint-only gradient path.
  DynamicArray<RoutingElement> const allLinear = {
      LinearJointElement(0, 0.5_r), LinearJointElement(1, -0.3_r)};

  DynamicArray<real> const currentPose = {0.2_r, 0.4_r};
  DynamicArray<real> const previousPose = {0.1_r, 0.3_r};
  real const timeStep = 0.01_r;
  RowMatrix<real> const jacobian = ComputeBodyJacobian(chain, MakeConstSpan(currentPose));
  AssemblyParams const params{.assemObj = true, .assemRes = true, .assemDRes = false};
  real const force = 1_r;
  int const numDofs = isize(currentPose);

  for (DynamicArray<RoutingElement> const& routingElements : {waypointOnly, mixed, allLinear}) {
    SpatialTendon const tendon = MakeTendon(
        chain,
        MakeConstSpan(routingElements),
        std::make_unique<ForceControlActuator>(force, test::ExpectOK{}));
    AssembledTransmission const assembled = AssembleResDRes(
        tendon,
        chain,
        MakeConstSpan(currentPose),
        MakeConstSpan(previousPose),
        timeStep,
        params,
        AsConstView(jacobian));
    DynamicArray<real> const g =
        FiniteDiffDisplacementGradient(tendon, chain, MakeConstSpan(currentPose));
    for (int i = 0; i < numDofs; ++i) {
      EXPECT_NEAR(
          assembled.residual[i], force * g[i], kSpatialRelTol * Abs(force * g[i]) + kSpatialAbsTol);
    }
    auto const expectedObjective =
        static_cast<double>(force * DisplacementOf(tendon, chain, MakeConstSpan(currentPose)));
    EXPECT_NEAR(
        assembled.objective,
        expectedObjective,
        static_cast<double>(kSpatialRelTol * Abs(expectedObjective) + kSpatialAbsTol));
  }

  // A tendon with no actuator contributes nothing: AddObjResDRes early-returns via the
  // HasActuator() == false path and leaves the residual untouched.
  SpatialTendon const passiveTendon = MakeTendon(chain, MakeConstSpan(waypointOnly));
  AssembledTransmission const passive = AssembleResDRes(
      passiveTendon,
      chain,
      MakeConstSpan(currentPose),
      MakeConstSpan(previousPose),
      timeStep,
      params,
      AsConstView(jacobian));
  for (int i = 0; i < numDofs; ++i) {
    EXPECT_EQ(passive.residual[i], 0_r);
  }
}

TEST(SpatialTendon, DResidual_MatchesQuadraticApproximation) {
  // The assembled dresidual must equal stiffness * g (outer) g, where g is the length gradient
  // (Gauss-Newton approximation that drops the second-derivative term). We obtain g independently
  // via finite differences of Displacement. A linear-joint element is included so g mixes both
  // gradient contributions.
  TwoLinkChain const chain = MakeTwoLinkRevoluteChain(1.0_r);
  DynamicArray<RoutingElement> routingElements =
      MakeWaypointElements({0, 1}, {Real3{0.3_r, 0_r, 0_r}, Real3{0.5_r, 0_r, 0_r}});
  routingElements.push_back(LinearJointElement(0, 0.4_r));

  real const stiffness = 1_r;
  bool const allowCompressiveForce = true; // Avoid clamping so the force/hessian stay smooth.
  SpatialTendon const tendon = MakeTendon(
      chain,
      MakeConstSpan(routingElements),
      std::make_unique<DisplacementControlActuator>(0_r, stiffness, 0_r, allowCompressiveForce));

  DynamicArray<real> const currentPose = {0.2_r, 0.4_r};
  // Zero velocity (previous == current) so the force is purely the spring term.
  DynamicArray<real> const& previousPose = currentPose;
  real const timeStep = 0.01_r;
  RowMatrix<real> const jacobian = ComputeBodyJacobian(chain, MakeConstSpan(currentPose));
  AssemblyParams const params{
      .assemObj = false, .assemRes = true, .assemDRes = true, .psdDRes = false};
  AssembledTransmission const assembled = AssembleResDRes(
      tendon,
      chain,
      MakeConstSpan(currentPose),
      MakeConstSpan(previousPose),
      timeStep,
      params,
      AsConstView(jacobian));

  DynamicArray<real> const g =
      FiniteDiffDisplacementGradient(tendon, chain, MakeConstSpan(currentPose));
  int const numDofs = isize(currentPose);
  for (int i = 0; i < numDofs; ++i) {
    for (int j = 0; j < numDofs; ++j) {
      real const expected = stiffness * g[i] * g[j];
      EXPECT_NEAR(
          assembled.dresidual(i, j), expected, kSpatialRelTol * Abs(expected) + kSpatialAbsTol);
      // The quadratic approximation is symmetric.
      EXPECT_NEAR(assembled.dresidual(i, j), assembled.dresidual(j, i), kSpatialAbsTol);
    }
  }
}
