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
#include <mochi_core/geometry/model_data.h>
#include <mochi_core/utils/array_utils.h>
#include <mochi_core/utils/rand_utils.h>
#include <mochi_physics/src/mochi_context.h>
#include <mochi_physics/src/mochi_integration.h>
#include <mochi_physics/src/mochi_rod.h>
#include <mochi_physics/src/mochi_rod_pose.h>
#include "mochi_core/test/mochi_test_helpers.h"
#include "mochi_physics_test_fixture.h"

#include <limits>

using namespace mochi;
using namespace mochi::experimental;

// Parameter struct for world-from-local transformation of the rod actor.
// Used to test that the rod physics is invariant under rigid body transformations.
struct RodTransformParam {
  TransformRT worldFromLocal;
  std::string name;

  // Factory methods for common test cases
  static RodTransformParam Identity() {
    return {TransformRT::Identity(), "Identity"};
  }

  static RodTransformParam Translation(Real3 const& t, std::string const& nameStr) {
    return {TransformRT{t}, nameStr};
  }

  static RodTransformParam Rotation(Quaternion const& q, std::string const& nameStr) {
    return {TransformRT{q, Real3{0_r, 0_r, 0_r}}, nameStr};
  }

  static RodTransformParam Full(Quaternion const& q, Real3 const& t, std::string const& nameStr) {
    return {TransformRT{q, t}, nameStr};
  }
};

// Returns a set of test transformations covering identity, translations, and various rotations
inline DynamicArray<RodTransformParam> GetRodTransformTestCases() {
  return {
      RodTransformParam::Identity(),
      RodTransformParam::Translation(Real3{1.5_r, -2.3_r, 0.7_r}, "TranslationOnly"),
      RodTransformParam::Rotation(
          Quaternion::FromAxisAngle(Real3{1_r, 0_r, 0_r}, kPI / 4_r), "RotateX45"),
      RodTransformParam::Rotation(
          Quaternion::FromAxisAngle(Real3{0_r, 1_r, 0_r}, kPI / 3_r), "RotateY60"),
      RodTransformParam::Rotation(
          Quaternion::FromAxisAngle(Real3{0_r, 0_r, 1_r}, kPI / 2_r), "RotateZ90"),
      RodTransformParam::Rotation(
          Quaternion::FromAxisAngle(Normalize(Real3{1_r, 1_r, 1_r}), 2_r * kPI / 3_r),
          "RotateDiag120"),
      RodTransformParam::Full(
          Quaternion::FromAxisAngle(Normalize(Real3{1_r, -2_r, 0.5_r}), kPI / 5_r),
          Real3{-0.5_r, 1.2_r, 2.1_r},
          "FullTransform"),
  };
}

class MochiRodActorStraightRodScene : public test::MochiSceneTestBase,
                                      public ::testing::WithParamInterface<RodTransformParam> {
 protected:
  Actor* _actor = nullptr;
  TransformRT _worldFromLocal;
  DynamicArray<Real3> _visNodePositions;

 public:
  void SetUp() override {
    test::MochiSceneTestBase::SetUp();

    _worldFromLocal = GetParam().worldFromLocal;

    // Create rod actor
    // This is a straight rod along the x-axis from 0 to 1
    auto& reg = GetRegistry();

    // Define nodes for a straight rod
    DynamicArray<Real3> nodes;
    nodes.push_back(Real3{0_r, 0_r, 0_r});
    nodes.push_back(Real3{0.25_r, 0_r, 0_r});
    nodes.push_back(Real3{0.5_r, 0_r, 0_r});
    nodes.push_back(Real3{0.75_r, 0_r, 0_r});
    nodes.push_back(Real3{1_r, 0_r, 0_r});

    // Define element frame axes (using y-axis as the frame axis)
    int const numNodes = isize(nodes);
    int const numElements = numNodes - 1;
    DynamicArray<Real3> elementFrameAxes;
    for (int i = 0; i < numElements; ++i) {
      elementFrameAxes.push_back(Real3{0_r, 1_r, 0_r});
    }

    // Visual mesh: 3 nodes forming a triangle around element 2's midpoint
    _visNodePositions = {
        Real3{0.625_r, 0_r, 0_r}, Real3{0.625_r, 0.1_r, 0_r}, Real3{0.625_r, 0_r, 0.1_r}};
    DynamicArray<Int3> visTriangles = {Int3{0, 1, 2}};
    DynamicArray<int> visElementIndices = {2, 2, 2};
    DynamicArray<real> visWeights = {1_r, 1_r, 1_r};

    ModelDataView modelView;
    modelView.mesh.emplace();
    modelView.mesh->nodesPerElement = 2;
    modelView.mesh->coordinates = Flatten(MakeConstSpan(nodes));
    modelView.elementFrameAxes = Flatten(MakeConstSpan(elementFrameAxes));

    modelView.visualMesh.emplace();
    modelView.visualMesh->nodesPerElement = 3;
    modelView.visualMesh->coordinates = Flatten(MakeConstSpan(_visNodePositions));
    modelView.visualMesh->connectivity = Flatten(MakeConstSpan(visTriangles));
    modelView.visualMesh->skinning.emplace();
    modelView.visualMesh->skinning->weightsPerNode = 1;
    modelView.visualMesh->skinning->indices = MakeConstSpan(visElementIndices);
    modelView.visualMesh->skinning->weights = MakeConstSpan(visWeights);

    ShapeHandle shape = _scene->GetContext()->CreateModelShape(modelView, ErrorAssert{});

    RodActorParams params;
    params.worldFromLocal = _worldFromLocal;
    params.shape = shape;
    params.material.linearDensity = 1_r;
    // Rotational inertia about the rod's axis must be nonzero for the problem to remain well-posed
    // with no constraints on the twisting DoFs.
    params.material.linearRotationalInertia = 1_r;
    params.material.axialStiffness = 1e3_r;
    params.material.torsionalStiffness = 1e1_r;
    params.material.flexuralStiffness = {1e1_r, 1e1_r};

    _actor = CreateRodActor(_scene, params, ErrorAssert{});
    EXPECT_NE(entt::entity{}, mochi::GetEntity(reg, _actor->GetHandle(), test::ExpectOK{}));
  }
};

TEST_P(MochiRodActorStraightRodScene, RodActorFreefall) {
  // Simulate with a gravitational acceleration in all three directions (in world space).
  Real3 constexpr kGravityWorld = {1_r, 2_r, 3_r};
  _scene->SetGravity(kGravityWorld);

  // Need implicit midpoint for exact free-falling trajectory.
  auto solverParams = _scene->GetSolverParams();
  solverParams.integrationMethod = IntegrationMethod::SymplecticDIRK12;
  if (MOCHI_USE_DOUBLE_PRECISION) {
    // Prove rod preconditioner is exact.
    solverParams.linearSolver.maxIter = 1;
  }
  _scene->SetSolverParams(solverParams, ErrorAssert{});

  _actor->RegisterQuery(QueryType::VisualNodePositions, ErrorAssert{});

  real constexpr kTimeInterval = 4_r;
  int constexpr kNumSteps = 16;
  real constexpr kTimeStep = kTimeInterval / (real)kNumSteps;
  for (int i = 0; i < kNumSteps; ++i) {
    _scene->Step(kTimeStep);
  }

  // Expected displacement in world space
  Real3 const expectedDisplacementWorld = 0.5_r * kGravityWorld * kTimeInterval * kTimeInterval;
  // Transform to local space for comparison with DoF values (which are in local coords)
  Real3 const expectedDisplacementLocal =
      _worldFromLocal.TransformDirectionInverse(expectedDisplacementWorld);

  int const numDofs = _actor->GetNumDofs();
  EXPECT_TRUE(numDofs % 4 == 0);
  DynamicArray<real> dofValues(numDofs);
  _actor->GetDofValues({}, dofValues, test::ExpectOK{});

  real constexpr kTolerance = MOCHI_USE_DOUBLE_PRECISION ? 1e-8_r : 1e-3_r;

  // Check the first 3 DoFs (displacement) for each node
  for (int i = 0; i < numDofs; i += 4) {
    // The result should be exact up to algebraic solver tolerances (and floating point rounding
    // errors). The truncation error from time integration should be zero in freefall with implicit
    // midpoint.
    for (int j = 0; j < 3; ++j) {
      EXPECT_NEAR_RTOL(expectedDisplacementLocal[j], dofValues[i + j], kTolerance);
    }
  }

  // Visual mesh positions should translate by the same freefall displacement
  auto visPositions = _actor->GetVisualMeshNodePositionsLocal(ErrorAssert{});
  ASSERT_EQ(isize(visPositions), 3 * isize(_visNodePositions));
  for (int i = 0; i < isize(_visNodePositions); ++i) {
    Real3 const expected = _visNodePositions[i] + expectedDisplacementLocal;
    for (int d = 0; d < 3; ++d) {
      EXPECT_NEAR_RTOL(expected[d], visPositions[3 * i + d], kTolerance)
          << "Visual node " << i << " component " << d;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    WorldFromLocalTransform,
    MochiRodActorStraightRodScene,
    ::testing::ValuesIn(GetRodTransformTestCases()),
    [](::testing::TestParamInfo<RodTransformParam> const& info) { return info.param.name; });

class MochiRodActorAxialStretchScene : public test::MochiSceneTestBase,
                                       public ::testing::WithParamInterface<RodTransformParam> {
 protected:
  Actor* _actor = nullptr;
  TransformRT _worldFromLocal = {};

 public:
  static real constexpr kLength = 2_r;
  static int constexpr kNumNodes = 9;
  static real constexpr kAxialStiffness = 1e3_r;
  static real constexpr kLinearDensity = 1_r;
  static real constexpr kForce = 1_r;
  // Approximate asserts
  static real constexpr kLooseTolerance = 1e-2_r;
  // Exact asserts
  static real constexpr kTightTolerance = MOCHI_USE_DOUBLE_PRECISION ? 1e-6_r : 1.2e-4_r;

  void SetUp() override {
    test::MochiSceneTestBase::SetUp();

    _worldFromLocal = GetParam().worldFromLocal;
    auto& reg = GetRegistry();

    // Create a straight rod along the x-axis
    DynamicArray<Real3> nodes;
    nodes.reserve(kNumNodes);
    for (int i = 0; i < kNumNodes; ++i) {
      real t = static_cast<real>(i) / static_cast<real>(kNumNodes - 1);
      nodes.push_back(Real3{t * kLength, 0_r, 0_r});
    }

    // Define element frame axes
    DynamicArray<Real3> elementFrameAxes;
    elementFrameAxes.reserve(kNumNodes - 1);
    for (int i = 0; i < kNumNodes - 1; ++i) {
      elementFrameAxes.push_back(Real3{0_r, 1_r, 0_r});
    }

    ShapeHandle shape = CreatePolylineShape(
        _scene->GetContext(), nodes, elementFrameAxes, /*isClosedLoop=*/false, ErrorAssert{});

    RodActorParams params;
    params.worldFromLocal = _worldFromLocal;
    params.shape = shape;
    params.material.linearDensity = kLinearDensity;
    params.material.axialStiffness = kAxialStiffness;
    params.material.torsionalStiffness = 1e3_r;
    params.material.flexuralStiffness = {1e3_r, 1e3_r};

    _actor = CreateRodActor(_scene, params, ErrorAssert{});
    EXPECT_NE(entt::entity{}, mochi::GetEntity(reg, _actor->GetHandle(), test::ExpectOK{}));
  }

  void TearDown() override {
    _scene->DestroyActor(_actor);
    test::MochiSceneTestBase::TearDown();
  }
};

TEST_P(MochiRodActorAxialStretchScene, RodActorAxialStretch) {
  // No gravity for this test
  _scene->SetGravity({0_r, 0_r, 0_r});

  // Fix the position of the first node - BCs are in world space
  DynamicArray<int> bcDofIndices;
  DynamicArray<real> bcDofValues;

  // The first node is at local position (0, 0, 0), transform to world space
  Real3 const firstNodeLocal{0_r, 0_r, 0_r};
  Real3 const firstNodeWorld = _worldFromLocal.TransformPoint(firstNodeLocal);
  for (int dof = 0; dof < 3; ++dof) {
    bcDofIndices.push_back(dof);
    bcDofValues.push_back(firstNodeWorld[dof]);
  }
  // Fix twist of first node to remove rigid spinning mode
  bcDofIndices.push_back(3);
  bcDofValues.push_back(0_r);

  _actor->AddBoundaryConditionDofsWorld(
      MakeConstSpan(bcDofIndices), MakeConstSpan(bcDofValues), ErrorAssert{});

  // Apply axial force on the last node in the local +x direction (transformed to world)
  // The force DoF API uses local coordinates, so we need to apply force in all 3 world directions
  // proportionally to match the local x-direction force
  Real3 const forceLocalDir = Normalize(Real3{1_r, 0_r, 0_r});
  Real3 const forceWorldDir = _worldFromLocal.TransformDirection(forceLocalDir);

  DynamicArray<int> forceDofs;
  DynamicArray<real> forceValues;
  int const lastNodeIndex = kNumNodes - 1;
  for (int dof = 0; dof < 3; ++dof) {
    forceDofs.push_back(4 * lastNodeIndex + dof);
    forceValues.push_back(kForce * forceWorldDir[dof]);
  }
  _actor->SetExternalForcesOnDofs(forceDofs, forceValues, ErrorAssert{});

  if (MOCHI_USE_DOUBLE_PRECISION) {
    // Prove rod preconditioner is exact.
    auto solverParams = _scene->GetSolverParams();
    solverParams.linearSolver.maxIter = 1;
    _scene->SetSolverParams(solverParams, ErrorAssert{});
  }

  // Take a small step first to avoid a singular DResidual to machine precision if using a large
  // step first.
  real constexpr kDummyTimeStep = 1e-2_r;
  _scene->Step(kDummyTimeStep);

  // Take large backward Euler steps to reach static equilibrium quickly
  test::SetSceneIntegrationMethod(_scene, IntegrationMethod::BackwardEuler);
  real constexpr kTimeInterval = 1e8_r;
  int constexpr kNumSteps = 10;
  real constexpr kTimeStep = kTimeInterval / (real)kNumSteps;
  for (int i = 0; i < kNumSteps; ++i) {
    _scene->Step(kTimeStep);
  }

  // Check the axial displacement at the end
  int const numDofs = _actor->GetNumDofs();
  DynamicArray<real> dofValues(numDofs);
  _actor->GetDofValues({}, dofValues, ErrorAssert{});

  // Get the displacement of the last node in local coordinates
  // DoF values are displacements in local coordinates
  Real3 const lastNodeDisplacementLocal{
      dofValues[4 * lastNodeIndex + 0],
      dofValues[4 * lastNodeIndex + 1],
      dofValues[4 * lastNodeIndex + 2]};

  // The displacement direction should match the local x-axis (rod axis)
  // Project the displacement onto the local x-axis to get the axial displacement
  real const actualDisplacement = lastNodeDisplacementLocal[0];

  // The transverse displacement should be negligible (rod only stretches axially)
  real const transverseDisplacement =
      Sqrt(Sqr(lastNodeDisplacementLocal[1]) + Sqr(lastNodeDisplacementLocal[2]));
  EXPECT_NEAR(0_r, transverseDisplacement, kLooseTolerance * Abs(actualDisplacement));

  // Expected axial displacement in the small-strain limit: delta = F * L / (E * A) where E*A is the
  // axial stiffness Note: This formula is not exact for large deformations, due to the use of the
  // Green--Lagrange strain, but it accurate in the small-strain limit. The approximate nature of
  // this assert is why the "loose" tolerance is used.
  real constexpr kExpectedDisplacement = kForce * kLength / kAxialStiffness;

  EXPECT_NEAR(
      kExpectedDisplacement, actualDisplacement, kLooseTolerance * Abs(kExpectedDisplacement));

  // Stretch should satisfy this exactly for the Green--Lagrange-strain-based St. Venant--Kirchhoff
  // constitutive model, so it uses the "tight" tolerance.
  real const actualLength = kLength + actualDisplacement;
  real const stretchRatio = actualLength / kLength;
  real const glStrain = 0.5_r * (Sqr(stretchRatio) - 1_r);
  EXPECT_NEAR(kForce, kAxialStiffness * glStrain * stretchRatio, kTightTolerance * Abs(kForce));
}

INSTANTIATE_TEST_SUITE_P(
    WorldFromLocalTransform,
    MochiRodActorAxialStretchScene,
    ::testing::ValuesIn(GetRodTransformTestCases()),
    [](::testing::TestParamInfo<RodTransformParam> const& info) { return info.param.name; });

class MochiRodActorTorsionScene : public test::MochiSceneTestBase,
                                  public ::testing::WithParamInterface<RodTransformParam> {
 protected:
  Actor* _actor = nullptr;
  TransformRT _worldFromLocal = {};

 public:
  static real constexpr kLength = 2_r;
  static int constexpr kNumNodes = 128;
  static real constexpr kTorsionalStiffness = 1e3_r;
  static real constexpr kLinearDensity = 1_r;
  static real constexpr kTorque = 10_r;
  // Note: There is some discretization error here, due to the nonlinear preasymptotic torsional
  // stiffness penalizing large jumps in twist between elements, so the tolerance should be well
  // above what would be used for a patch test.
  static real constexpr kTolerance = 1e-3_r;

  static int constexpr kNumElements = kNumNodes - 1;
  static real constexpr kElementLength = kLength / kNumElements;

  void SetUp() override {
    test::MochiSceneTestBase::SetUp();

    _worldFromLocal = GetParam().worldFromLocal;
    auto& reg = GetRegistry();

    // Create a straight rod along the x-axis
    DynamicArray<Real3> nodes;
    nodes.reserve(kNumNodes);
    for (int i = 0; i < kNumNodes; ++i) {
      real t = static_cast<real>(i) / static_cast<real>(kNumElements);
      nodes.push_back(Real3{t * kLength, 0_r, 0_r});
    }

    // Define element frame axes
    DynamicArray<Real3> elementFrameAxes;
    elementFrameAxes.reserve(kNumElements);
    for (int i = 0; i < kNumElements; ++i) {
      elementFrameAxes.push_back(Real3{0_r, 1_r, 0_r});
    }

    ShapeHandle shape = CreatePolylineShape(
        _scene->GetContext(), nodes, elementFrameAxes, /*isClosedLoop=*/false, ErrorAssert{});

    RodActorParams params;
    params.worldFromLocal = _worldFromLocal;
    params.shape = shape;
    params.material.linearDensity = kLinearDensity;
    params.material.axialStiffness = 1e3_r;
    params.material.torsionalStiffness = kTorsionalStiffness;
    params.material.flexuralStiffness = {1e3_r, 1e3_r};

    _actor = CreateRodActor(_scene, params, ErrorAssert{});
    EXPECT_NE(entt::entity{}, mochi::GetEntity(reg, _actor->GetHandle(), test::ExpectOK{}));
  }

  void TearDown() override {
    _scene->DestroyActor(_actor);
    test::MochiSceneTestBase::TearDown();
  }
};

TEST_P(MochiRodActorTorsionScene, RodActorTorsion) {
  // No gravity for this test
  _scene->SetGravity({0_r, 0_r, 0_r});

  // Fix the position of the first node - BCs are in world space
  DynamicArray<int> bcDofIndices;
  DynamicArray<real> bcDofValues;

  // The first node is at local position (0, 0, 0), transform to world space
  Real3 const firstNodeLocal{0_r, 0_r, 0_r};
  Real3 const firstNodeWorld = _worldFromLocal.TransformPoint(firstNodeLocal);
  for (int dof = 0; dof < 3; ++dof) {
    bcDofIndices.push_back(dof);
    bcDofValues.push_back(firstNodeWorld[dof]);
  }
  // Fix twist of first node (this is the BC for the "first element" which is grouped with node 0)
  bcDofIndices.push_back(3);
  bcDofValues.push_back(0_r);

  // Fix the end positions of the rod to remove rotational rigid body modes.
  // For pure torsion, the centerline doesn't deform, so these constraints don't affect the physics.
  int const lastNodeIndex = kNumNodes - 1;
  Real3 const lastNodeLocal{kLength, 0_r, 0_r};
  Real3 const lastNodeWorld = _worldFromLocal.TransformPoint(lastNodeLocal);
  for (int i = 0; i < 3; ++i) {
    bcDofIndices.push_back(4 * lastNodeIndex + i);
    bcDofValues.push_back(lastNodeWorld[i]);
  }

  _actor->AddBoundaryConditionDofsWorld(
      MakeConstSpan(bcDofIndices), MakeConstSpan(bcDofValues), ErrorAssert{});

  // Apply torque to the last element's twist DoF (grouped with the second-to-last node)
  DynamicArray<int> forceDofs;
  DynamicArray<real> forceValues;
  int const secondToLastNodeIndex = kNumNodes - 2;
  forceDofs.push_back(4 * secondToLastNodeIndex + 3); // twist DoF
  forceValues.push_back(kTorque);
  _actor->SetExternalForcesOnDofs(forceDofs, forceValues, ErrorAssert{});

  if (MOCHI_USE_DOUBLE_PRECISION) {
    // Prove rod preconditioner is exact.
    auto solverParams = _scene->GetSolverParams();
    solverParams.linearSolver.maxIter = 1;
    _scene->SetSolverParams(solverParams, ErrorAssert{});
  }

  // Take large backward Euler steps to reach static equilibrium quickly
  test::SetSceneIntegrationMethod(_scene, IntegrationMethod::BackwardEuler);
  real constexpr kTimeInterval = 1e8_r;
  int constexpr kNumSteps = 10;
  real constexpr kTimeStep = kTimeInterval / (real)kNumSteps;
  for (int i = 0; i < kNumSteps; ++i) {
    _scene->Step(kTimeStep);
  }

  // Get the element frame axes to check the twist distribution
  auto& reg = GetRegistry();
  auto entity = mochi::GetEntity(reg, _actor->GetHandle(), test::ExpectOK{});
  auto const& currPose = reg.get<CRodPose<TimeStep::Current>>(entity);

  // Expected total twist angle: theta = M * L / (G * J) where G*J is the torsional stiffness
  real constexpr kExpectedTotalTwist = kTorque * kLength / kTorsionalStiffness;

  // The twist should be distributed linearly over the rod length
  // For each element, compute the angle between the element frame axis and the reference frame
  // axis
  for (int i = 0; i < kNumElements; ++i) {
    // Position along the rod (at element center)
    real const positionAlongRod = (static_cast<real>(i) + 0.5_r) * kElementLength;

    // Expected twist angle at this position (linear distribution)
    real const expectedTwistAtPosition = kExpectedTotalTwist * positionAlongRod / kLength;

    // The frame axis is rotated from the reference axis by the twist angle
    // Reference axis is [0, 1, 0], rotated about the x-axis (rod axis)
    Real3 const expectedAxis = {0_r, Cos(expectedTwistAtPosition), Sin(expectedTwistAtPosition)};

    // Check that the element frame axis matches the expected axis
    Real3 const actualAxis = currPose.value.frameAxes[i];
    for (int j = 0; j < 3; ++j) {
      EXPECT_NEAR_RTOL(expectedAxis[j], actualAxis[j], kTolerance);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    WorldFromLocalTransform,
    MochiRodActorTorsionScene,
    ::testing::ValuesIn(GetRodTransformTestCases()),
    [](::testing::TestParamInfo<RodTransformParam> const& info) { return info.param.name; });

// Combined parameter struct for cantilever tests: gravity direction + world-from-local transform
struct CantileverTestParam {
  int gravityAxis; // 1 for y-axis, 2 for z-axis
  TransformRT worldFromLocal;
  std::string name;
};

// Returns a set of test parameters combining gravity directions and transforms
inline DynamicArray<CantileverTestParam> GetCantileverTestCases() {
  DynamicArray<CantileverTestParam> params;

  DynamicArray<std::pair<int, std::string>> gravityDirs = {{1, "GravY"}, {2, "GravZ"}};

  DynamicArray<RodTransformParam> transforms = GetRodTransformTestCases();

  for (auto const& [axis, axisName] : gravityDirs) {
    for (auto const& tx : transforms) {
      params.push_back({axis, tx.worldFromLocal, axisName + "_" + tx.name});
    }
  }

  return params;
}

class MochiRodActorCantileverScene : public test::MochiSceneTestBase,
                                     public ::testing::WithParamInterface<CantileverTestParam> {
 protected:
  Actor* _actor = nullptr;
  TransformRT _worldFromLocal = {};

 public:
  static real constexpr kLength = 1_r; // Axial length in x-direction
  static real constexpr kWidth = 2e-2_r; // Cross-section width in y-direction
  static real constexpr kHeight = 1e-2_r; // Cross-section height in z-direction
  static int constexpr kNumElements = 64;
  static int constexpr kNumNodes = kNumElements + 1;
  static real constexpr kElementLength = kLength / kNumElements;
  static real constexpr kShearModulus = 1e5_r;
  static real constexpr kYoungsModulus = 1e5_r;
  static real constexpr kDensity = 1e0_r;
  static real constexpr kGravity = 1e0_r;
  // Asserts are based on a solution from linearized beam theory, so they are not expected to be
  // exact for the nonlinear formulation, but this test can still detect significant errors in the
  // bending formulation.
  static real constexpr kTolerance = 5e-2_r;

  void SetUp() override {
    test::MochiSceneTestBase::SetUp();

    _worldFromLocal = GetParam().worldFromLocal;
    auto& reg = GetRegistry();

    // Create a straight rod along the x-axis
    DynamicArray<Real3> nodes;
    DynamicArray<Real3> frameAxes;
    nodes.reserve(kNumNodes);
    frameAxes.reserve(kNumElements);

    for (int i = 0; i < kNumNodes; ++i) {
      real t = static_cast<real>(i) / static_cast<real>(kNumElements);
      nodes.push_back(Real3{t * kLength, 0_r, 0_r});
      if (i < kNumNodes - 1) {
        frameAxes.push_back(Real3{0_r, 1_r, 0_r});
      }
    }

    ShapeHandle shape = CreatePolylineShape(
        _scene->GetContext(), nodes, frameAxes, /*isClosedLoop=*/false, ErrorAssert{});

    RodActorParams params;
    params.worldFromLocal = _worldFromLocal;
    params.shape = shape;

    real const area = kWidth * kHeight;
    params.material.linearDensity = kDensity * area;
    params.material.axialStiffness = area * kYoungsModulus;

    // For a rectangular cross section, the second moment of area is:
    // I_y = (width * height^3) / 12 (bending about y-axis, deflection in z-direction)
    // I_z = (height * width^3) / 12 (bending about z-axis, deflection in y-direction)
    real const Iy = (kWidth * Pow(kHeight, 3)) / 12_r;
    real const Iz = (kHeight * Pow(kWidth, 3)) / 12_r;
    params.material.flexuralStiffness = kYoungsModulus * Real2{Iy, Iz};

    // This is not correct for a rectangular cross section (which in general has no closed-form
    // expression, requiring a solution of the Prandtl stress function), but the torsion constant is
    // not important for verifying pure bending.
    real const torsionConstant = 1e-8_r;
    params.material.torsionalStiffness = kShearModulus * torsionConstant;

    _actor = CreateRodActor(_scene, params, ErrorAssert{});
    EXPECT_NE(entt::entity{}, mochi::GetEntity(reg, _actor->GetHandle(), test::ExpectOK{}));
  }

  void TearDown() override {
    _scene->DestroyActor(_actor);
    test::MochiSceneTestBase::TearDown();
  }
};

TEST_P(MochiRodActorCantileverScene, RodActorCantileverBending) {
  CantileverTestParam const param = GetParam();

  // Define the gravity direction in local coordinates
  Real3 gravityDirLocal{0_r, 0_r, 0_r};
  gravityDirLocal[param.gravityAxis] = -1_r;

  // Transform to world coordinates for setting the scene gravity
  Real3 const gravityWorld = _worldFromLocal.TransformDirection(gravityDirLocal) * kGravity;
  _scene->SetGravity(gravityWorld);

  // Apply cantilever boundary conditions: fix first two nodes in world space
  DynamicArray<int> bcDofIndices;
  DynamicArray<real> bcDofValues;
  for (int i = 0; i < 2; ++i) {
    for (int dof = 0; dof < 3; ++dof) {
      bcDofIndices.push_back(i * 4 + dof);
      // Transform local position to world for position BCs
      Real3 localPos{(real)i * kElementLength, 0_r, 0_r};
      Real3 worldPos = _worldFromLocal.TransformPoint(localPos);
      bcDofValues.push_back(worldPos[dof]);
    }
  }
  // Fix twist of first element
  bcDofIndices.push_back(3);
  bcDofValues.push_back(0_r);

  _actor->AddBoundaryConditionDofsWorld(
      MakeConstSpan(bcDofIndices), MakeConstSpan(bcDofValues), ErrorAssert{});

  // Increase number of non-linear iterations for accurate static solution.
  auto solverParams = _scene->GetSolverParams();
  solverParams.nonLinearSolver.maxIter = 8;
  if (MOCHI_USE_DOUBLE_PRECISION) {
    // Prove rod preconditioner is exact.
    solverParams.linearSolver.maxIter = 1;
  }
  _scene->SetSolverParams(solverParams, ErrorAssert{});

  // Take large backward Euler steps to reach static equilibrium quickly
  test::SetSceneIntegrationMethod(_scene, IntegrationMethod::BackwardEuler);
  real constexpr kTimeInterval = 1e6_r;
  int constexpr kNumSteps = 16;
  real constexpr kTimeStep = kTimeInterval / (real)kNumSteps;
  for (int i = 0; i < kNumSteps; ++i) {
    _scene->Step(kTimeStep);
  }

  // Get the DoF values to check the tip deflection
  int const numDofs = _actor->GetNumDofs();
  DynamicArray<real> dofValues(numDofs);
  _actor->GetDofValues({}, dofValues, ErrorAssert{});

  // For a cantilever beam under uniform load (self-weight), the tip deflection from linearized beam
  // theory is: delta = (w * L^4) / (8 * E * I) where w is the load per unit length, L is the
  // length, E is Young's modulus, and I is the second moment of area.
  real const w = kDensity * kWidth * kHeight * kGravity;

  // Choose correct area moment of inertia for analytical solution based on gravity axis.
  real const Iz = (kHeight * Pow(kWidth, 3)) / 12_r;
  real const Iy = (kWidth * Pow(kHeight, 3)) / 12_r;
  real const IAxis = param.gravityAxis == 1 ? Iz : Iy;
  real const expectedTipDeflection = (w * Pow(kLength, 4)) / (8_r * kYoungsModulus * IAxis);

  // Get tip deflection in local coordinates
  // DoF values are in local coordinates, so the deflection direction corresponds to the gravity
  // axis
  int const lastNodeIndex = kNumNodes - 1;
  real const actualTipDeflection = dofValues[4 * lastNodeIndex + param.gravityAxis];

  EXPECT_NEAR(-expectedTipDeflection, actualTipDeflection, kTolerance * Abs(expectedTipDeflection));
}

INSTANTIATE_TEST_SUITE_P(
    GravityAndTransform,
    MochiRodActorCantileverScene,
    ::testing::ValuesIn(GetCantileverTestCases()),
    [](::testing::TestParamInfo<CantileverTestParam> const& info) { return info.param.name; });

// Test for the GenerateDiscreteBishopFrame free function
class GenerateDiscreteBishopFrameTest : public ::testing::Test {
 protected:
  static real constexpr kTolerance = MOCHI_USE_DOUBLE_PRECISION ? 1e-8_r : 1e-5_r;

  // Generate a random polyline with the given number of nodes using the provided RNG
  static DynamicArray<Real3> GenerateRandomPolyline(
      int numNodes,
      mochi_default_random_engine& rng,
      real minEdgeLength = 0.1_r,
      real maxEdgeLength = 1.0_r,
      real maxBendAngle = 0.9_r * kPI) {
    MOCHI_ASSERT(numNodes >= 2, "Need at least 2 nodes");

    DynamicArray<Real3> nodes;
    nodes.reserve(numNodes);

    // Start at the origin
    nodes.push_back(Real3{0_r, 0_r, 0_r});

    // Initial direction (along x-axis)
    Real3 direction = {1_r, 0_r, 0_r};

    for (int i = 1; i < numNodes; ++i) {
      // Random edge length
      real const edgeLength = RandomUniformValue(rng, minEdgeLength, maxEdgeLength);

      // Add node in current direction
      nodes.push_back(nodes[i - 1] + direction * edgeLength);

      // Rotate direction for next edge (if not the last node)
      if (i < numNodes - 1) {
        // Generate random rotation axis perpendicular to current direction
        Real3 randomVec;
        SetRandom(rng, -1_r, 1_r, randomVec);
        Real3 rotAxis = Cross(direction, randomVec);

        // Handle case where randomVec is parallel to direction
        if (Norm(rotAxis) < 1e-6_r) {
          rotAxis = {0_r, 1_r, 0_r};
          if (Abs(Dot(direction, rotAxis)) > 0.9_r) {
            rotAxis = {0_r, 0_r, 1_r};
          }
          rotAxis = Cross(direction, rotAxis);
        }
        rotAxis = Normalize(rotAxis);

        // Random bend angle
        real const bendAngle = RandomUniformValue(rng, -maxBendAngle, maxBendAngle);

        // Apply rotation using quaternion multiplication
        Quaternion const rotation = Quaternion::FromAxisAngle(rotAxis, bendAngle);
        direction = rotation * direction;
        direction = Normalize(direction);
      }
    }

    return nodes;
  }

  // Compute the angle between two vectors (in radians)
  static real AngleBetween(Real3 const& a, Real3 const& b) {
    real const cosAngle = std::clamp(Dot(Normalize(a), Normalize(b)), -1_r, 1_r);
    return std::acos(cosAngle);
  }
};

TEST(RodApplyLieDeltaToPose, ZeroDeltaPreservesPose) {
  // A simple 3-node rod along the x-axis: nodes at (0,0,0), (1,0,0), (2,0,0).
  DynamicArray<Real3> meshNodes = {{0_r, 0_r, 0_r}, {1_r, 0_r, 0_r}, {2_r, 0_r, 0_r}};
  int const numNodes = isize(meshNodes);
  int const numElements = numNodes - 1;
  int const numDofs = numNodes * fem::kNumRodFields;

  // Reference displacements: zero (reference configuration).
  ColumnVector<real> refDisp = ColumnVector<real>::Zero(numDofs);
  // Reference axes: y-axis for both elements.
  DynamicArray<Real3> refAxes(numElements, Real3{0_r, 1_r, 0_r});

  // Zero delta.
  ColumnVector<real> dofDelta = ColumnVector<real>::Zero(numDofs);

  // Output buffers.
  ColumnVector<real> outDisp(numDofs);
  DynamicArray<Real3> outAxes(numElements);

  rod::ApplyLieDeltaToPose(
      MakeConstSpan(meshNodes),
      refDisp,
      MakeConstSpan(refAxes),
      dofDelta,
      outDisp,
      MakeSpan(outAxes));

  // Displacements should remain zero.
  for (int i = 0; i < numDofs; ++i) {
    EXPECT_NEAR(0_r, outDisp[i], kDefaultNearEqualEpsilon<real>);
  }
  // Axes should remain the y-axis.
  for (int i = 0; i < numElements; ++i) {
    EXPECT_NEAR(0_r, outAxes[i][0], kDefaultNearEqualEpsilon<real>);
    EXPECT_NEAR(1_r, outAxes[i][1], kDefaultNearEqualEpsilon<real>);
    EXPECT_NEAR(0_r, outAxes[i][2], kDefaultNearEqualEpsilon<real>);
  }
}

TEST(RodApplyLieDeltaToPose, PureTwistRotatesAxes) {
  // A 2-node rod along the x-axis: nodes at (0,0,0), (1,0,0).
  DynamicArray<Real3> meshNodes = {{0_r, 0_r, 0_r}, {1_r, 0_r, 0_r}};
  int const numNodes = isize(meshNodes);
  int const numElements = numNodes - 1;
  int const numDofs = numNodes * fem::kNumRodFields;

  ColumnVector<real> refDisp = ColumnVector<real>::Zero(numDofs);
  DynamicArray<Real3> refAxes(numElements, Real3{0_r, 1_r, 0_r});

  // Apply a pi/2 twist at the first node (twist DOF index = 3).
  ColumnVector<real> dofDelta = ColumnVector<real>::Zero(numDofs);
  real const twist = kPI / 2_r;
  dofDelta[3] = twist;

  ColumnVector<real> outDisp(numDofs);
  DynamicArray<Real3> outAxes(numElements);

  rod::ApplyLieDeltaToPose(
      MakeConstSpan(meshNodes),
      refDisp,
      MakeConstSpan(refAxes),
      dofDelta,
      outDisp,
      MakeSpan(outAxes));

  // Displacements should equal the delta (pure additive for displacement DOFs).
  for (int i = 0; i < numDofs; ++i) {
    EXPECT_NEAR(dofDelta[i], outDisp[i], kDefaultNearEqualEpsilon<real>);
  }

  // The tangent is along x-axis and doesn't change (no translational displacement).
  // The twist rotates the frame axis (y-axis) by pi/2 about the x-axis → z-axis.
  real const tol = 1e-5_r;
  EXPECT_NEAR(0_r, outAxes[0][0], tol);
  EXPECT_NEAR(0_r, outAxes[0][1], tol);
  EXPECT_NEAR(1_r, outAxes[0][2], tol);
}

TEST(RodApplyLieDeltaToPose, TranslationalDisplacementTransportsAxes) {
  // A 2-node rod along the x-axis: nodes at (0,0,0), (1,0,0).
  DynamicArray<Real3> meshNodes = {{0_r, 0_r, 0_r}, {1_r, 0_r, 0_r}};
  int const numNodes = isize(meshNodes);
  int const numElements = numNodes - 1;
  int const numDofs = numNodes * fem::kNumRodFields;

  ColumnVector<real> refDisp = ColumnVector<real>::Zero(numDofs);
  DynamicArray<Real3> refAxes(numElements, Real3{0_r, 1_r, 0_r});

  // Apply a translational displacement that rotates the tangent into the x-z plane.
  // Move the second node upward in z, so the new tangent is in the x-z plane.
  ColumnVector<real> dofDelta = ColumnVector<real>::Zero(numDofs);
  dofDelta[fem::kNumRodFields + 2] = 1_r; // z-displacement of node 1

  ColumnVector<real> outDisp(numDofs);
  DynamicArray<Real3> outAxes(numElements);

  rod::ApplyLieDeltaToPose(
      MakeConstSpan(meshNodes),
      refDisp,
      MakeConstSpan(refAxes),
      dofDelta,
      outDisp,
      MakeSpan(outAxes));

  // The frame axis should be parallel-transported to the new tangent direction.
  // Since the tangent rotated within the x-z plane, and the frame axis was along y,
  // the transported axis should remain along y (orthogonal to the rotation plane).
  real const tol = 1e-5_r;
  EXPECT_NEAR(0_r, outAxes[0][0], tol);
  EXPECT_NEAR(1_r, outAxes[0][1], tol);
  EXPECT_NEAR(0_r, outAxes[0][2], tol);

  // The output axis should be orthogonal to the new tangent.
  Real3 const newTangent =
      Normalize(Real3{1_r, 0_r, 1_r}); // (1,0,0) + delta(0,0,1) for node1, node0 unchanged
  real const dotProduct = Dot(outAxes[0], newTangent);
  EXPECT_NEAR(0_r, dotProduct, tol);
}

// ============================================================================
// Rotational inertia test
// ============================================================================

// Tests rod rotational inertia with a multi-step integrator (BDF3).
// A single-element rod spins freely about its axis. The test verifies that angular momentum is
// conserved and the frame axis rotates as expected. Multi-step integrators are non-trivial for rods
// because twist recentering must preserve differences between previous step values; if done
// incorrectly, the rotational inertia term would be scrambled and this test would fail.
class MochiRodRotationalInertiaTest : public test::MochiSceneTestBase {
 protected:
  Actor* _actor = nullptr;

 public:
  static real constexpr kLength = 1_r;
  static int constexpr kNumNodes = 2;
  static int constexpr kNumElements = kNumNodes - 1;

  void SetUp() override {
    test::MochiSceneTestBase::SetUp();

    auto& reg = GetRegistry();

    // Create a single-element rod along the x-axis.
    DynamicArray<Real3> nodes;
    nodes.push_back(Real3{0_r, 0_r, 0_r});
    nodes.push_back(Real3{kLength, 0_r, 0_r});

    DynamicArray<Real3> frameAxes;
    frameAxes.push_back(Real3{0_r, 1_r, 0_r});

    ShapeHandle shape = CreatePolylineShape(
        _scene->GetContext(), nodes, frameAxes, /*isClosedLoop=*/false, ErrorAssert{});

    RodActorParams params;
    params.shape = shape;
    // Use default material parameters, just ensure nonzero rotational inertia.
    params.material.linearRotationalInertia = 1_r;

    _actor = CreateRodActor(_scene, params, ErrorAssert{});
    EXPECT_NE(entt::entity{}, mochi::GetEntity(reg, _actor->GetHandle(), test::ExpectOK{}));
  }

  void TearDown() override {
    _scene->DestroyActor(_actor);
    test::MochiSceneTestBase::TearDown();
  }
};

TEST_F(MochiRodRotationalInertiaTest, FreeSpinningRodWithBDF3) {
  // No gravity.
  _scene->SetGravity({0_r, 0_r, 0_r});

  // No boundary conditions - let the rod spin freely.

  // Use BDF3 integrator (multi-step) - this is the non-trivial case for recentering.
  SolverParams solverParams = _scene->GetSolverParams();
  solverParams.integrationMethod = IntegrationMethod::BDF3;
  _scene->SetSolverParams(solverParams, ErrorAssert{});

  // Apply initial twist rate using SetNodeVelocitiesLocal.
  // For rod actors: 4 values per node — (vx, vy, vz) in [m/s] plus a twist rate [rad/s].
  real constexpr kInitialTwistRate = 2_r; // rad/s
  DynamicArray<real> initialVelocities(kNumNodes * 4, 0_r);
  for (int node = 0; node < kNumNodes; ++node) {
    initialVelocities[node * 4 + 3] = kInitialTwistRate;
  }
  _actor->SetNodeVelocitiesLocal(MakeConstSpan(initialVelocities), ErrorAssert{});

  // Simulate for multiple time steps.
  // BDF3 needs 3 previous steps, so we need at least 4 steps to exercise the full BDF3 logic.
  real constexpr kTimeStep = 0.1_r;
  int constexpr kNumSteps = 10;
  real constexpr kTotalTime = kTimeStep * kNumSteps;

  // Expected total twist angle from conservation of angular momentum.
  real constexpr kExpectedTotalTwist = kInitialTwistRate * kTotalTime;

  auto& reg = GetRegistry();
  auto entity = mochi::GetEntity(reg, _actor->GetHandle(), test::ExpectOK{});

  real maxTwistDoF = 0_r;
  for (int step = 0; step < kNumSteps; ++step) {
    _scene->Step(kTimeStep);

    // After each step, verify that the current twist DoFs stay bounded (recentered).
    // Without recentering, the twist DoFs would grow unboundedly as the rod spins.
    // With recentering, they should stay near one time step's worth of rotation at most.
    auto const& currPose = reg.get<CRodPose<TimeStep::Current>>(entity);
    for (int node = 0; node < kNumNodes; ++node) {
      real const currentTwist = Abs(currPose.value.displacements[node * 4 + 3]);
      maxTwistDoF = Max(maxTwistDoF, currentTwist);

      // The twist DoF should be bounded (not accumulating indefinitely).
      real const twistBound = 3_r * kInitialTwistRate * kTimeStep;
      EXPECT_LT(currentTwist, twistBound)
          << "At step " << step + 1 << ", node " << node
          << ": twist DoF should be recentered, but has large value: " << currentTwist
          << " (bound: " << twistBound << ")";
    }
  }

  // Verify the frame axis has rotated by approximately the expected total angle.
  auto const& currPose = reg.get<CRodPose<TimeStep::Current>>(entity);
  Real3 const actualAxis = currPose.value.frameAxes[0];

  // The frame axis started as [0, 1, 0] and should rotate about the x-axis by expectedTotalTwist.
  Real3 const expectedAxis = {0_r, Cos(kExpectedTotalTwist), Sin(kExpectedTotalTwist)};

  // Use looser tolerance due to time integration errors and torsional stiffness.
  real constexpr kAxisTolerance = MOCHI_USE_DOUBLE_PRECISION ? 5e-2_r : 1e-1_r;
  for (int j = 0; j < 3; ++j) {
    EXPECT_NEAR(expectedAxis[j], actualAxis[j], kAxisTolerance)
        << "Frame axis component " << j << " mismatch after " << kNumSteps << " steps. "
        << "Expected total twist: " << kExpectedTotalTwist << " rad";
  }

  // Verify twist DoFs stayed small throughout (implicitly tests recentering).
  EXPECT_LT(maxTwistDoF, kExpectedTotalTwist / 2_r)
      << "Twist DoFs should stay bounded (recentered), not accumulate to total rotation. "
      << "Max twist DoF: " << maxTwistDoF << ", Expected total twist: " << kExpectedTotalTwist;
}

// ============================================================================
// AddWeightedDifferences tests for RodPose
// ============================================================================

class RodAddWeightedDifferencesTest : public ::testing::Test {
 protected:
  // Default: 2-node rod along x-axis.
  void SetUp() override {
    SetUpRod(2);
  }

  void SetUpRod(int numNodes) {
    meshNodes_.clear();
    for (int i = 0; i < numNodes; ++i) {
      meshNodes_.push_back(Real3{real(i), 0_r, 0_r});
    }
    numNodes_ = numNodes;
    numElements_ = numNodes - 1;
    numDofs_ = numNodes * fem::kNumRodFields;

    base_.displacements = ColumnVector<real>::Zero(numDofs_);
    base_.frameAxes.clear();
    for (int i = 0; i < numElements_; ++i) {
      base_.frameAxes.push_back(Real3{0_r, 1_r, 0_r});
    }
  }

  RodPose MakePose() const {
    RodPose pose;
    pose.displacements = ColumnVector<real>::Zero(numDofs_);
    pose.frameAxes.clear();
    for (int i = 0; i < numElements_; ++i) {
      pose.frameAxes.push_back(Real3{0_r, 1_r, 0_r});
    }
    return pose;
  }

  void SetFrameAxes(RodPose& pose, Real3 const& axis) const {
    pose.frameAxes.clear();
    for (int i = 0; i < numElements_; ++i) {
      pose.frameAxes.push_back(axis);
    }
  }

  void RunAddWeightedDifferences(
      std::initializer_list<RodPose> poses,
      std::initializer_list<real> coefficients) {
    // Pre-initialize output with correct dimensions (required by AddWeightedDifferences).
    out_.displacements = ColumnVector<real>::Zero(numDofs_);
    out_.frameAxes.clear();
    for (int i = 0; i < numElements_; ++i) {
      out_.frameAxes.push_back(Real3{0_r, 0_r, 0_r});
    }
    DynamicArray<RodPoseContainer> vals;
    for (auto const& pose : poses) {
      vals.push_back(RodPoseContainer(pose));
    }
    DynamicArray<real> coeffs;
    for (auto c : coefficients) {
      coeffs.push_back(c);
    }
    integration::details::AddWeightedDifferences(
        MakeConstSpan(meshNodes_),
        base_,
        MakeConstSpan(vals),
        MakeConstSpan(coeffs),
        Int2{0, isize(coeffs)},
        out_);
  }

  DynamicArray<Real3> meshNodes_;
  int numNodes_ = 0;
  int numElements_ = 0;
  int numDofs_ = 0;
  RodPose base_;
  RodPose out_;
  static constexpr real kTol = 1e-5_r;
};

TEST_F(RodAddWeightedDifferencesTest, ZeroCoefficientsPreservesBase) {
  RodPose pose1 = MakePose();
  pose1.displacements[0] = 0.5_r;
  pose1.displacements[3] = kPI / 4_r;
  SetFrameAxes(pose1, Real3{0_r, 0_r, 1_r});

  RunAddWeightedDifferences({pose1}, {0_r});

  for (int i = 0; i < numDofs_; ++i) {
    EXPECT_NEAR(base_.displacements[i], out_.displacements[i], kTol);
  }
  for (int i = 0; i < numElements_; ++i) {
    EXPECT_NEAR(base_.frameAxes[i][1], out_.frameAxes[i][1], kTol);
  }
}

TEST_F(RodAddWeightedDifferencesTest, UnitCoefficientCopiesPose) {
  RodPose pose1 = MakePose();
  pose1.displacements[3] = kPI / 2_r;
  SetFrameAxes(pose1, Real3{0_r, 0_r, 1_r});

  RunAddWeightedDifferences({pose1}, {1_r});

  EXPECT_NEAR(kPI / 2_r, out_.displacements[3], kTol);
  EXPECT_NEAR(0_r, out_.frameAxes[0][1], kTol);
  EXPECT_NEAR(1_r, out_.frameAxes[0][2], kTol);
}

TEST_F(RodAddWeightedDifferencesTest, HalfCoefficientInterpolatesTwist) {
  RodPose pose1 = MakePose();
  pose1.displacements[3] = kPI / 2_r;
  SetFrameAxes(pose1, Real3{0_r, 0_r, 1_r});

  RunAddWeightedDifferences({pose1}, {0.5_r});

  // Expected twist: 0.5 * (pi/2) = pi/4.
  EXPECT_NEAR(kPI / 4_r, out_.displacements[3], kTol);

  // Expected frame axis: rotated by pi/4 about x-axis from y toward z.
  real const sqrt2Over2 = Sqrt(2_r) / 2_r;
  EXPECT_NEAR(sqrt2Over2, out_.frameAxes[0][1], kTol);
  EXPECT_NEAR(sqrt2Over2, out_.frameAxes[0][2], kTol);
}

TEST_F(RodAddWeightedDifferencesTest, MultipleStagesWeightedCombination) {
  RodPose stage1 = MakePose();
  stage1.displacements[0] = 1_r;

  RodPose stage2 = MakePose();
  stage2.displacements[0] = 2_r;

  RunAddWeightedDifferences({stage1, stage2}, {0.3_r, 0.7_r});

  // Expected: 0 + 0.3*(1-0) + 0.7*(2-0) = 1.7.
  EXPECT_NEAR(1.7_r, out_.displacements[0], kTol);
}

TEST_F(RodAddWeightedDifferencesTest, TranslationalDisplacementTransportsAxes) {
  // Move node 1 up in z, rotating tangent into x-z plane.
  RodPose pose1 = MakePose();
  pose1.displacements[fem::kNumRodFields + 2] = 1_r;

  RunAddWeightedDifferences({pose1}, {1_r});

  // Frame axis remains along y (parallel transport in x-z plane).
  EXPECT_NEAR(0_r, out_.frameAxes[0][0], kTol);
  EXPECT_NEAR(1_r, out_.frameAxes[0][1], kTol);
  EXPECT_NEAR(0_r, out_.frameAxes[0][2], kTol);

  // Verify orthogonality to new tangent.
  Real3 const newTangent = Normalize(Real3{1_r, 0_r, 1_r});
  EXPECT_NEAR(0_r, Dot(out_.frameAxes[0], newTangent), kTol);
}

TEST_F(RodAddWeightedDifferencesTest, CombinedTwistAndTranslation) {
  RodPose pose1 = MakePose();
  pose1.displacements[0] = 0.5_r;
  pose1.displacements[3] = kPI / 2_r;
  SetFrameAxes(pose1, Real3{0_r, 0_r, 1_r});

  RunAddWeightedDifferences({pose1}, {1_r});

  EXPECT_NEAR(0.5_r, out_.displacements[0], kTol);
  EXPECT_NEAR(kPI / 2_r, out_.displacements[3], kTol);
  EXPECT_NEAR(1_r, out_.frameAxes[0][2], kTol);
}

TEST_F(GenerateDiscreteBishopFrameTest, RandomPolylinesValidityAndMinimalRotation) {
  // Use a fixed seed for reproducibility
  mochi_default_random_engine rng = RandomGenerator(42);

  // Test various polyline sizes
  DynamicArray<int> nodeCounts = {2, 3, 5, 10, 20, 50};
  int constexpr kNumPolylinesPerSize = 10;

  for (int numNodes : nodeCounts) {
    for (int polylineIdx = 0; polylineIdx < kNumPolylinesPerSize; ++polylineIdx) {
      // Generate a random polyline
      DynamicArray<Real3> nodes = GenerateRandomPolyline(numNodes, rng);
      int const numElements = numNodes - 1;

      // Generate the Bishop frame
      DynamicArray<Real3> axes = GenerateDiscreteBishopFrame(nodes, /*isClosedLoop=*/false);

      // Check that we got the expected number of axes
      ASSERT_EQ(isize(axes), numElements)
          << "Wrong number of axes for polyline with " << numNodes << " nodes";

      // Compute tangent vectors for each element
      DynamicArray<Real3> tangents;
      tangents.reserve(numElements);
      for (int i = 0; i < numElements; ++i) {
        tangents.push_back(Normalize(nodes[i + 1] - nodes[i]));
      }

      // Validity check 1: Each axis must be a unit vector
      for (int i = 0; i < numElements; ++i) {
        real const axisNorm = Norm(axes[i]);
        EXPECT_NEAR(1_r, axisNorm, kTolerance)
            << "Axis " << i << " is not a unit vector (norm = " << axisNorm << ")";
      }

      // Validity check 2: Each axis must be orthogonal to its element's tangent
      for (int i = 0; i < numElements; ++i) {
        real const dotProduct = Abs(Dot(axes[i], tangents[i]));
        EXPECT_LT(dotProduct, kTolerance)
            << "Axis " << i << " is not orthogonal to tangent (dot = " << dotProduct << ")";
      }

      // Validity check 3: No 180-degree jumps between consecutive axes
      for (int i = 0; i < numElements - 1; ++i) {
        real const dotProduct = Dot(axes[i], axes[i + 1]);
        EXPECT_GT(dotProduct, kTolerance - 1_r)
            << "Axes " << i << " and " << i + 1 << " have a 180-degree jump";
      }

      // Bishop frame property: angle between consecutive axes <= angle between consecutive
      // tangents, which should always be true for minimal rotation transport.
      for (int i = 0; i < numElements - 1; ++i) {
        real const axisAngle = AngleBetween(axes[i], axes[i + 1]);
        real const tangentAngle = AngleBetween(tangents[i], tangents[i + 1]);

        // Allow small tolerance for numerical errors
        EXPECT_LE(axisAngle, tangentAngle + kTolerance)
            << "Axis rotation (" << axisAngle << " rad) exceeds tangent rotation (" << tangentAngle
            << " rad) between elements " << i << " and " << i + 1;
      }
    }
  }
}

// ============================================================================
// Rod Visual Mesh Tests
// ============================================================================

class MochiRodVisualMesh : public test::MochiSceneTestBase {
 protected:
  // These members are populated by CreateRodShapeWithVisualMesh for use by tests that need to
  // reference the input geometry (e.g., ReferenceConfigQueriesMatchInput, Hdf5RoundTrip).
  static constexpr int kNumNodes = 4;
  DynamicArray<Real3> _rodNodes;
  DynamicArray<Real3> _visNodePositions;
  DynamicArray<Int3> _visTriangles;
  DynamicArray<int> _elementIndices;
  DynamicArray<real> _weights;
  int _weightsPerNode = 1;

  void SetUp() override {
    test::MochiSceneTestBase::SetUp();
  }

  // Create a rod shape with a visual mesh triangle. When periodic, the visual mesh is placed on
  // the closing element to exercise wrapping; otherwise it is placed on an interior element.
  ShapeHandle CreateRodShapeWithVisualMesh(bool isClosedLoop = false, bool includeSkinning = true) {
    _rodNodes = isClosedLoop
        ? DynamicArray<
              Real3>{Real3{0_r, 0_r, 0_r}, Real3{0.75_r, 0_r, 0_r}, Real3{0.75_r, 0_r, 0.75_r}, Real3{0_r, 0_r, 0.75_r}}
        : DynamicArray<Real3>{
              Real3{0_r, 0_r, 0_r},
              Real3{0.25_r, 0_r, 0_r},
              Real3{0.5_r, 0_r, 0_r},
              Real3{0.75_r, 0_r, 0_r}};
    int const numNodes = isize(_rodNodes);
    int const numElements = isClosedLoop ? numNodes : numNodes - 1;
    DynamicArray<Real3> frameAxes(numElements, Real3{0_r, 1_r, 0_r});

    // Use default visual mesh if not already set by the test
    if (_visNodePositions.empty()) {
      int const visElement = isClosedLoop ? numElements - 1 : 1;
      Int2 const en = {visElement, (visElement + 1) % numNodes};
      Real3 const midpoint = 0.5_r * (_rodNodes[en[0]] + _rodNodes[en[1]]);
      _visNodePositions = {
          midpoint, midpoint + Real3{0_r, 0.1_r, 0_r}, midpoint + Real3{0_r, 0_r, 0.1_r}};
      _visTriangles = {Int3{0, 1, 2}};
      _weightsPerNode = 1;
      _elementIndices = {visElement, visElement, visElement};
      _weights = {1_r, 1_r, 1_r};
    }

    ModelDataView modelView;
    modelView.mesh.emplace();
    modelView.mesh->nodesPerElement = 2;
    modelView.mesh->coordinates = Flatten(MakeConstSpan(_rodNodes));
    modelView.elementFrameAxes = Flatten(MakeConstSpan(frameAxes));
    // Periodicity is now encoded by the mesh connectivity array. Build the appropriate
    // sequential connectivity for either an open or closed polyline.
    DynamicArray<int> connectivity;
    {
      int const numSegments = isClosedLoop ? numNodes : numNodes - 1;
      connectivity.reserve(2 * numSegments);
      for (int i = 0; i < numSegments; ++i) {
        connectivity.push_back(i);
        connectivity.push_back((i + 1) % numNodes);
      }
    }
    modelView.mesh->connectivity = MakeConstSpan(connectivity);

    modelView.visualMesh.emplace();
    modelView.visualMesh->nodesPerElement = 3;
    modelView.visualMesh->coordinates = Flatten(MakeConstSpan(_visNodePositions));
    modelView.visualMesh->connectivity = Flatten(MakeConstSpan(_visTriangles));
    if (includeSkinning) {
      modelView.visualMesh->skinning.emplace();
      modelView.visualMesh->skinning->weightsPerNode = _weightsPerNode;
      modelView.visualMesh->skinning->indices = MakeConstSpan(_elementIndices);
      modelView.visualMesh->skinning->weights = MakeConstSpan(_weights);
    }

    return _scene->GetContext()->CreateModelShape(modelView, ErrorAssert{});
  }

  static RodActorParams GetRodActorParams(ShapeHandle shape, bool useVisualMeshContact = false) {
    RodActorParams params;
    params.shape = shape;
    params.useVisualMeshContact = useVisualMeshContact;
    params.material.linearDensity = 1_r;
    params.material.linearRotationalInertia = 1_r;
    params.material.axialStiffness = 1e3_r;
    params.material.torsionalStiffness = 1e1_r;
    params.material.flexuralStiffness = {1e1_r, 1e1_r};
    return params;
  }

  Actor* CreateRodActorWithVisualMesh(ShapeHandle shape, bool useVisualMeshContact = false) {
    return CreateRodActor(_scene, GetRodActorParams(shape, useVisualMeshContact), ErrorAssert{});
  }

  // Verify skinning Jacobian FD consistency at reference config and after a deformation step.
  void VerifySkinningJacobianFDAtRefAndDeformed(Actor* actor) {
    _scene->Step(0_r);
    VerifySkinningJacobianFD(actor);

    // Pin the first node to ensure a non-trivial deformation (not just rigid-body translation)
    DynamicArray<int> bcDofs = {0, 1, 2, 3};
    DynamicArray<real> bcVals = {0_r, 0_r, 0_r, 0_r};
    actor->AddBoundaryConditionDofsWorld(
        MakeConstSpan(bcDofs), MakeConstSpan(bcVals), ErrorAssert{});
    _scene->SetGravity({1e2_r, 2e2_r, 3e2_r});
    _scene->Step(0.1_r);

    VerifySkinningJacobianFD(actor);
  }
  void VerifySkinningJacobianFD(Actor* actor) {
    static constexpr real kFdEps = MOCHI_USE_DOUBLE_PRECISION ? 1e-6_r : 1e-4_r;
    static constexpr real kFdTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-4_r : 1e-2_r;

    auto& reg = GetRegistry();
    auto entity = mochi::GetEntity(reg, actor->GetHandle(), test::ExpectOK{});

    auto const& polylineMesh = reg.get<CPolylineMesh const>(entity);
    auto const& basePose = reg.get<CRodPose<TimeStep::Current> const>(entity);
    int const numElements = polylineMesh.NumElements();

    auto const& visualMesh = reg.get<CVisualMesh const>(entity);
    auto const& rodEmbedding = reg.get<CRodVisualMeshEmbedding const>(entity);
    CRodContactSkin contactSkin(visualMesh.mesh, rodEmbedding.data);
    CRodContactSkinningData skinningData;
    rod::InitializeContactSkinningJacobian(contactSkin, polylineMesh, skinningData);
    rod::ResolveContactSkinningJacobian(contactSkin, polylineMesh, basePose, skinningData);

    auto const& jac = skinningData.jacobian;
    int const numDofs = jac.Cols();
    int const numVisualNodes = jac.Rows();

    auto computeVisualPositions = [&](CRodPose<TimeStep::Current> const& pose) {
      CQueryVisualNodePositions visPosQuery;
      rod::UpdateQueryVisualNodePositionsAndNormals(
          visualMesh, rodEmbedding, polylineMesh, pose, visPosQuery, nullptr);
      return visPosQuery.nodePositions;
    };

    for (int dofIdx = 0; dofIdx < numDofs; ++dofIdx) {
      ColumnVector<real> dofDeltaPlus = ColumnVector<real>::Zero(numDofs);
      dofDeltaPlus(dofIdx) = kFdEps;

      CRodPose<TimeStep::Current> perturbedPosePlus;
      perturbedPosePlus.value.displacements = ColumnVector<real>::Zero(numDofs);
      perturbedPosePlus.value.frameAxes.resize(numElements, Real3{});
      rod::ApplyLieDeltaToPose(
          polylineMesh.nodes,
          basePose.value.displacements,
          MakeConstSpan(basePose.value.frameAxes),
          dofDeltaPlus,
          perturbedPosePlus.value.displacements,
          MakeSpan(perturbedPosePlus.value.frameAxes));

      auto positionsPlus = computeVisualPositions(perturbedPosePlus);

      ColumnVector<real> dofDeltaMinus = ColumnVector<real>::Zero(numDofs);
      dofDeltaMinus(dofIdx) = -kFdEps;

      CRodPose<TimeStep::Current> perturbedPoseMinus;
      perturbedPoseMinus.value.displacements = ColumnVector<real>::Zero(numDofs);
      perturbedPoseMinus.value.frameAxes.resize(numElements, Real3{});
      rod::ApplyLieDeltaToPose(
          polylineMesh.nodes,
          basePose.value.displacements,
          MakeConstSpan(basePose.value.frameAxes),
          dofDeltaMinus,
          perturbedPoseMinus.value.displacements,
          MakeSpan(perturbedPoseMinus.value.frameAxes));

      auto positionsMinus = computeVisualPositions(perturbedPoseMinus);

      for (int visNodeIdx = 0; visNodeIdx < numVisualNodes; ++visNodeIdx) {
        auto const colIndices = jac.Indices(visNodeIdx);

        auto const* it = std::find(colIndices.begin(), colIndices.end(), dofIdx);

        if (it == colIndices.end()) {
          for (int d = 0; d < 3; ++d) {
            real const fdDeriv =
                (positionsPlus[3 * visNodeIdx + d] - positionsMinus[3 * visNodeIdx + d]) /
                (2_r * kFdEps);
            EXPECT_NEAR(0_r, fdDeriv, kFdTol)
                << "DoF " << dofIdx << " should not affect visual node " << visNodeIdx
                << " component " << d;
          }
          continue;
        }

        int const localCol = static_cast<int>(it - colIndices.begin());

        for (int d = 0; d < 3; ++d) {
          real const fdDeriv =
              (positionsPlus[3 * visNodeIdx + d] - positionsMinus[3 * visNodeIdx + d]) /
              (2_r * kFdEps);

          real const analyticDeriv = jac.Values(visNodeIdx)[localCol][d];

          EXPECT_NEAR(analyticDeriv, fdDeriv, kFdTol)
              << "Mismatch at dPosition_visual[" << visNodeIdx << "][" << d << "] / dDoF[" << dofIdx
              << "]";
        }
      }
    }
  }
};

TEST_F(MochiRodVisualMesh, ReferenceConfigQueriesMatchInput) {
  ShapeHandle shape = CreateRodShapeWithVisualMesh();
  Actor* actor = CreateRodActorWithVisualMesh(shape);

  auto const visualMesh = actor->GetVisualMesh();

  // Reference positions should match input
  auto refPositions = visualMesh.coordinates;
  ASSERT_EQ(isize(refPositions), 3 * isize(_visNodePositions));
  for (int i = 0; i < isize(_visNodePositions); ++i) {
    for (int d = 0; d < 3; ++d) {
      EXPECT_NEAR(_visNodePositions[i][d], refPositions[3 * i + d], kTolerance);
    }
  }

  // Connectivity should match input
  auto connectivity = visualMesh.connectivity;
  ASSERT_EQ(isize(connectivity), 3);
  EXPECT_EQ(connectivity[0], 0);
  EXPECT_EQ(connectivity[1], 1);
  EXPECT_EQ(connectivity[2], 2);

  // Register both queries and step at dt=0
  actor->RegisterQuery(QueryType::VisualNodePositions, ErrorAssert{});
  actor->RegisterQuery(QueryType::VisualNodeNormals, ErrorAssert{});
  _scene->Step(0_r);

  // At reference configuration, queried positions should match input
  auto positions = actor->GetVisualMeshNodePositionsLocal(ErrorAssert{});
  ASSERT_EQ(isize(positions), 3 * isize(_visNodePositions));
  for (int i = 0; i < isize(_visNodePositions); ++i) {
    for (int d = 0; d < 3; ++d) {
      EXPECT_NEAR(_visNodePositions[i][d], positions[3 * i + d], kTolerance)
          << "Node " << i << " component " << d;
    }
  }

  // Normals should be along ±x for this triangle lying in the y-z plane
  auto normals = actor->GetVisualMeshNodeNormalsLocal(ErrorAssert{});
  ASSERT_EQ(isize(normals), 3 * isize(_visNodePositions));
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(Abs(normals[3 * i + 0]), 1_r, kTolerance) << "Node " << i;
    EXPECT_NEAR(normals[3 * i + 1], 0_r, kTolerance) << "Node " << i;
    EXPECT_NEAR(normals[3 * i + 2], 0_r, kTolerance) << "Node " << i;
  }
}

TEST_F(MochiRodVisualMesh, QueryWithoutVisualMeshFails) {
  // Rod without visual mesh should fail visual query registration
  CreateRodShapeWithVisualMesh(); // populate _rodNodes
  // NOTE: DynamicArray fill constructor for Real3 is miscompiled by MSVC in double precision,
  // corrupting the last element. Filling with an explicit for loop to work around it.
  DynamicArray<Real3> frameAxes(kNumNodes - 1);
  for (int i = 0; i < kNumNodes - 1; ++i) {
    frameAxes[i] = Real3{0_r, 1_r, 0_r};
  }
  ShapeHandle shape = CreatePolylineShape(
      _scene->GetContext(), _rodNodes, frameAxes, /*isClosedLoop=*/false, ErrorAssert{});
  RodActorParams params;
  params.shape = shape;
  Actor* actor = CreateRodActor(_scene, params, ErrorAssert{});

  actor->RegisterQuery(QueryType::VisualNodePositions, test::ExpectNotOK{});
}

TEST_F(MochiRodVisualMesh, UnskinnedVisualMeshIsIgnored) {
  ShapeHandle shape =
      CreateRodShapeWithVisualMesh(/*isClosedLoop=*/false, /*includeSkinning=*/false);

  test::ExpectLoggingInScope expectWarning(_scene->GetContext(), LogChannel::Warning);
  Actor* actor = CreateRodActorWithVisualMesh(shape);
  ASSERT_NE(nullptr, actor);

  EXPECT_EQ(MeshDataView{}, actor->GetVisualMesh());
  actor->RegisterQuery(QueryType::VisualNodePositions, test::ExpectNotOK{});

  auto& reg = GetRegistry();
  auto entity = mochi::GetEntity(reg, actor->GetHandle(), test::ExpectOK{});
  EXPECT_TRUE(reg.all_of<CFemSegmentDiscretization>(entity));
  EXPECT_FALSE(reg.all_of<CVisualMesh>(entity));
  EXPECT_FALSE(reg.all_of<CRodVisualMeshEmbedding>(entity));
  EXPECT_FALSE(reg.all_of<CRodContactSkin>(entity));
  EXPECT_FALSE(reg.all_of<CRodContactSkinningData>(entity));
  EXPECT_FALSE(reg.all_of<CFemSurfaceDiscretization>(entity));
  EXPECT_FALSE(reg.all_of<TagRodSurfaceContact>(entity));
}

TEST_F(MochiRodVisualMesh, UnskinnedVisualMeshCannotBeUsedForContact) {
  ShapeHandle shape =
      CreateRodShapeWithVisualMesh(/*isClosedLoop=*/false, /*includeSkinning=*/false);
  RodActorParams params = GetRodActorParams(shape, /*useVisualMeshContact=*/true);
  params.layer = "RejectedUnskinnedVisualMesh";
  int const numActors = _scene->GetNumActors();
  int const numContactLayers = _scene->GetNumContactLayers();

  Actor* actor = CreateRodActor(_scene, params, test::ExpectNotOK{});
  EXPECT_EQ(nullptr, actor);
  EXPECT_EQ(numActors, _scene->GetNumActors());
  EXPECT_EQ(numContactLayers, _scene->GetNumContactLayers());
}

TEST_F(MochiRodVisualMesh, VisualMeshContact_InitializesSkinningJacobianSparsity) {
  ShapeHandle shape = CreateRodShapeWithVisualMesh();
  Actor* actor = CreateRodActorWithVisualMesh(shape, /*useVisualMeshContact=*/true);

  auto& reg = GetRegistry();
  auto entity = mochi::GetEntity(reg, actor->GetHandle(), test::ExpectOK{});

  ASSERT_TRUE((reg.all_of<
               CVisualMesh,
               CRodVisualMeshEmbedding,
               CRodContactSkin,
               CRodContactSkinningData,
               CFemSurfaceDiscretization,
               TagRodSurfaceContact>(entity)));
  EXPECT_FALSE(reg.all_of<CFemSegmentDiscretization>(entity));

  auto const& visualMesh = reg.get<CVisualMesh const>(entity);
  auto const& rodEmbedding = reg.get<CRodVisualMeshEmbedding const>(entity);
  auto const& contactSkin = reg.get<CRodContactSkin const>(entity);
  auto const& skinningData = reg.get<CRodContactSkinningData const>(entity);

  EXPECT_EQ(contactSkin.mesh, visualMesh.mesh);
  EXPECT_EQ(contactSkin.embedding, rodEmbedding.data);

  auto const& jac = skinningData.jacobian;

  int const numVisualNodes = jac.Rows();
  EXPECT_EQ(numVisualNodes, 3);
  EXPECT_EQ(jac.Cols(), kNumNodes * fem::kNumRodFields);

  for (int i = 0; i < numVisualNodes; ++i) {
    int const nnz = isize(jac.Indices(i));
    EXPECT_EQ(nnz, 8) << "Visual node " << i << " should have 8 DoFs (2 nodes × 4 DoFs)";
  }

  for (int i = 0; i < numVisualNodes; ++i) {
    auto const colIndices = jac.Indices(i);

    int const elemIdx = 1;
    int const node0Start = elemIdx * fem::kNumRodFields;
    int const node1Start = (elemIdx + 1) * fem::kNumRodFields;

    DynamicArray<int> expectedDofs;
    for (int d = 0; d < fem::kNumRodFields; ++d) {
      expectedDofs.push_back(node0Start + d);
      expectedDofs.push_back(node1Start + d);
    }
    std::sort(expectedDofs.begin(), expectedDofs.end());

    ASSERT_EQ(isize(colIndices), isize(expectedDofs));
    for (int j = 0; j < isize(expectedDofs); ++j) {
      EXPECT_EQ(colIndices[j], expectedDofs[j]) << "Visual node " << i << " DoF index " << j;
    }
  }

  EXPECT_EQ(jac.NumNonZeros(), isize(jac.Indices(0)) * numVisualNodes);
}

TEST_F(MochiRodVisualMesh, ResolveContactSkinningJacobian_FiniteDifferenceConsistency) {
  ShapeHandle shape = CreateRodShapeWithVisualMesh();
  Actor* actor = CreateRodActorWithVisualMesh(shape);
  VerifySkinningJacobianFDAtRefAndDeformed(actor);
}

TEST_F(MochiRodVisualMesh, ResolveContactSkinningJacobian_FiniteDifferenceConsistency_ClosedLoop) {
  ShapeHandle shape = CreateRodShapeWithVisualMesh(/*isClosedLoop=*/true);
  Actor* actor = CreateRodActorWithVisualMesh(shape);
  VerifySkinningJacobianFDAtRefAndDeformed(actor);
}

// ============================================================================
// GenerateTubularRodModelData Tests
// ============================================================================

namespace {

class MochiRodTubularCrossSection : public test::MochiSceneTestBase {
 protected:
  static constexpr int kNumNodes = 5;
  static constexpr int kNumElements = kNumNodes - 1;
  static constexpr int kNumCrossSectionSegments = 8;

  DynamicArray<Real3> _nodes;
  DynamicArray<Real3> _frameAxes;

  void SetUp() override {
    test::MochiSceneTestBase::SetUp();

    // Straight rod along x-axis
    _nodes.reserve(kNumNodes);
    for (int i = 0; i < kNumNodes; ++i) {
      real const t = static_cast<real>(i) / static_cast<real>(kNumElements);
      _nodes.push_back(Real3{t, 0_r, 0_r});
    }
    _frameAxes.resize(kNumElements);
    Fill(MakeSpan(_frameAxes), Real3{0_r, 1_r, 0_r});
  }

  Actor* CreateRodActorFromShape(ShapeHandle shape) {
    RodActorParams params;
    params.shape = shape;
    params.material.linearDensity = 1_r;
    params.material.linearRotationalInertia = 1_r;
    params.material.axialStiffness = 1e3_r;
    params.material.torsionalStiffness = 1e1_r;
    params.material.flexuralStiffness = {1e1_r, 1e1_r};
    return CreateRodActor(_scene, params, ErrorAssert{});
  }
};

TEST_F(MochiRodTubularCrossSection, TopologyMatchesExpected) {
  real constexpr kRadius = 0.01_r;
  ModelData const model = GenerateTubularRodModelData(
      _nodes, _frameAxes, kRadius, kNumCrossSectionSegments, /*isClosedLoop=*/false, ErrorAssert{});
  ShapeHandle shape = _scene->GetContext()->CreateModelShape(model, ErrorAssert{});
  ASSERT_TRUE(shape.IsValid());

  Actor* actor = CreateRodActorFromShape(shape);
  ASSERT_NE(actor, nullptr);

  actor->RegisterQuery(QueryType::VisualNodePositions, ErrorAssert{});
  _scene->Step(0_r);

  int const expectedVisualNodes = kNumCrossSectionSegments * (kNumElements + 2) + 2;
  auto positions = actor->GetVisualMeshNodePositionsLocal(ErrorAssert{});
  EXPECT_EQ(isize(positions), 3 * expectedVisualNodes);

  int const expectedTriangles =
      2 * kNumCrossSectionSegments * (kNumElements + 1) + 2 * kNumCrossSectionSegments;
  auto connectivity = actor->GetVisualMesh().connectivity;
  EXPECT_EQ(isize(connectivity) / 3, expectedTriangles);

  // Verify ring 0 vertices are at expected radius from centerline.
  // For a straight rod along the x-axis, the first ring is at node 0 and consists of
  // kNumCrossSectionSegments vertices. Each should be at distance kRadius from the
  // x-axis (i.e., sqrt(y² + z²) ≈ kRadius).
  for (int i = 0; i < kNumCrossSectionSegments; ++i) {
    int const vertexIndex = i;
    real const y = positions[3 * vertexIndex + 1];
    real const z = positions[3 * vertexIndex + 2];
    real const distFromAxis = Sqrt(Sqr(y) + Sqr(z));
    EXPECT_NEAR(kRadius, distFromAxis, kTolerance)
        << "Ring vertex " << i << " is not at expected radius from centerline";
  }
}

TEST_F(MochiRodTubularCrossSection, AutoGeneratedFrameAxes) {
  real constexpr kRadius = 0.01_r;
  ModelData const model = GenerateTubularRodModelData(
      _nodes, {}, kRadius, kNumCrossSectionSegments, /*isClosedLoop=*/false, ErrorAssert{});
  ShapeHandle shape = _scene->GetContext()->CreateModelShape(model, ErrorAssert{});
  ASSERT_TRUE(shape.IsValid());

  Actor* actor = CreateRodActorFromShape(shape);
  ASSERT_NE(actor, nullptr);

  actor->RegisterQuery(QueryType::VisualNodePositions, ErrorAssert{});
  _scene->Step(0_r);

  int const expectedVisualNodes = kNumCrossSectionSegments * (kNumElements + 2) + 2;
  auto positions = actor->GetVisualMeshNodePositionsLocal(ErrorAssert{});
  EXPECT_EQ(isize(positions), 3 * expectedVisualNodes);
}

TEST_F(MochiRodTubularCrossSection, InvalidRadiusErrors) {
  {
    test::ExpectNotOK error;
    (void)GenerateTubularRodModelData(
        _nodes, _frameAxes, 0_r, kNumCrossSectionSegments, /*isClosedLoop=*/false, error);
  }
  {
    test::ExpectNotOK error;
    (void)GenerateTubularRodModelData(
        _nodes, _frameAxes, -1_r, kNumCrossSectionSegments, /*isClosedLoop=*/false, error);
  }
  {
    test::ExpectNotOK error;
    (void)GenerateTubularRodModelData(
        _nodes,
        _frameAxes,
        std::numeric_limits<real>::infinity(),
        kNumCrossSectionSegments,
        /*isClosedLoop=*/false,
        error);
  }
  {
    test::ExpectNotOK error;
    (void)GenerateTubularRodModelData(
        _nodes,
        _frameAxes,
        std::numeric_limits<real>::quiet_NaN(),
        kNumCrossSectionSegments,
        /*isClosedLoop=*/false,
        error);
  }
}

TEST_F(MochiRodTubularCrossSection, InsufficientSegmentsErrors) {
  test::ExpectNotOK error;
  (void)GenerateTubularRodModelData(_nodes, _frameAxes, 0.01_r, 2, /*isClosedLoop=*/false, error);
}

TEST_F(MochiRodTubularCrossSection, InsufficientNodesErrors) {
  real constexpr kRadius = 0.01_r;
  {
    test::ExpectNotOK error;
    DynamicArray<Real3> emptyNodes;
    (void)GenerateTubularRodModelData(
        emptyNodes, {}, kRadius, kNumCrossSectionSegments, /*isClosedLoop=*/false, error);
  }
  {
    test::ExpectNotOK error;
    DynamicArray<Real3> singleNode = {Real3{0_r, 0_r, 0_r}};
    (void)GenerateTubularRodModelData(
        singleNode, {}, kRadius, kNumCrossSectionSegments, /*isClosedLoop=*/false, error);
  }
}

} // namespace

TEST_F(MochiRodVisualMesh, UnskinnedVisualMeshModelDataRoundTrip) {
  ShapeHandle shape =
      CreateRodShapeWithVisualMesh(/*isClosedLoop=*/false, /*includeSkinning=*/false);

  auto* contextImpl = assert_cast<ContextImpl*>(_scene->GetContext());
  auto shapePtr = contextImpl->GetShapeSharedPtr(shape);
  ASSERT_NE(nullptr, shapePtr);
  ModelData modelData = shapePtr->GetModelData(ErrorAssert{});

  ASSERT_TRUE(modelData.visualMesh.has_value());
  EXPECT_FALSE(modelData.visualMesh->skinning.has_value());
  EXPECT_SPAN_EQ(Flatten(MakeConstSpan(_visNodePositions)), modelData.visualMesh->coordinates);
  EXPECT_SPAN_EQ(Flatten(MakeConstSpan(_visTriangles)), modelData.visualMesh->connectivity);

  ModelData expectedModelData = modelData;
  auto newShapePtr = ContextImpl::CreateShapeFromModelData(std::move(modelData), ErrorAssert{});
  ASSERT_NE(nullptr, newShapePtr);
  EXPECT_EQ(expectedModelData, newShapePtr->GetModelData(ErrorAssert{}));
}

TEST_IF_F(MOCHI_USE_HDF5, MochiRodVisualMesh, Hdf5RoundTrip) {
  ShapeHandle shape = CreateRodShapeWithVisualMesh();

  // Get model data from shape via internal API
  auto* contextImpl = assert_cast<ContextImpl*>(_scene->GetContext());
  auto shapePtr = contextImpl->GetShapeSharedPtr(shape);
  ASSERT_NE(shapePtr, nullptr);
  ModelData modelData = shapePtr->GetModelData(ErrorAssert{});

  // Verify visual mesh is present in model data
  ASSERT_TRUE(modelData.visualMesh.has_value());
  ASSERT_TRUE(modelData.visualMesh->skinning.has_value());
  EXPECT_EQ(modelData.visualMesh->skinning->weightsPerNode, _weightsPerNode);
  EXPECT_EQ(isize(modelData.visualMesh->skinning->indices), isize(_elementIndices));

  // Create a new shape from the model data (round-trip)
  auto newShapePtr = ContextImpl::CreateShapeFromModelData(std::move(modelData), ErrorAssert{});
  auto newShape = contextImpl->RegisterShape(newShapePtr, ErrorAssert{});

  // Create an actor from the round-tripped shape and verify visual mesh queries work
  RodActorParams params;
  params.shape = newShape;
  params.material.linearDensity = 1_r;
  params.material.linearRotationalInertia = 1_r;
  params.material.axialStiffness = 1e3_r;
  params.material.torsionalStiffness = 1e1_r;
  params.material.flexuralStiffness = {1e1_r, 1e1_r};
  Actor* actor = CreateRodActor(_scene, params, ErrorAssert{});

  actor->RegisterQuery(QueryType::VisualNodePositions, ErrorAssert{});
  _scene->Step(0_r);

  auto positions = actor->GetVisualMeshNodePositionsLocal(ErrorAssert{});
  ASSERT_EQ(isize(positions), 3 * isize(_visNodePositions));
  for (int i = 0; i < isize(_visNodePositions); ++i) {
    for (int d = 0; d < 3; ++d) {
      EXPECT_NEAR(_visNodePositions[i][d], positions[3 * i + d], kTolerance)
          << "Node " << i << " component " << d;
    }
  }
}

TEST_F(MochiRodVisualMesh, SkinningUnderRotation) {
  // Create a rod shape with a visual mesh
  ShapeHandle shape = CreateRodShapeWithVisualMesh();
  Actor* actor = CreateRodActorWithVisualMesh(shape);
  actor->RegisterQuery(QueryType::VisualNodePositions, ErrorAssert{});

  // Fix the first node and its twist to zero
  DynamicArray<int> bcDofIndices = {0, 1, 2, 3};
  DynamicArray<real> bcDofValues = {0_r, 0_r, 0_r, 0_r};
  actor->AddBoundaryConditionDofsWorld(
      MakeConstSpan(bcDofIndices), MakeConstSpan(bcDofValues), ErrorAssert{});

  // Fix the last node position at its reference position with twist = 0
  int const lastNodeDofOffset = 4 * (kNumNodes - 1);
  DynamicArray<int> lastNodeBcDofIndices = {
      lastNodeDofOffset, lastNodeDofOffset + 1, lastNodeDofOffset + 2, lastNodeDofOffset + 3};
  DynamicArray<real> lastNodeBcDofValues = {
      _rodNodes[kNumNodes - 1][0], _rodNodes[kNumNodes - 1][1], _rodNodes[kNumNodes - 1][2], 0_r};
  actor->AddBoundaryConditionDofsWorld(
      MakeConstSpan(lastNodeBcDofIndices), MakeConstSpan(lastNodeBcDofValues), ErrorAssert{});

  // Apply a torque to the middle nodes to induce twist
  real constexpr kTorque = 5_r;
  DynamicArray<int> torqueDofIndices;
  DynamicArray<real> torqueValues;
  for (int node = 1; node < kNumNodes - 1; ++node) {
    torqueDofIndices.push_back(4 * node + 3);
    torqueValues.push_back(kTorque);
  }
  actor->SetExternalForcesOnDofs(torqueDofIndices, torqueValues, ErrorAssert{});

  // Step to static equilibrium (quasi-static)
  _scene->SetGravity({0_r, 0_r, 0_r});

  // The huge quasi-static steps below reach static equilibrium via backward Euler's damping.
  test::SetSceneIntegrationMethod(_scene, IntegrationMethod::BackwardEuler);

  real constexpr kLargeStep = 1e6_r;
  for (int i = 0; i < 10; ++i) {
    _scene->Step(kLargeStep);
  }

  // Query visual mesh positions
  auto positions = actor->GetVisualMeshNodePositionsLocal(ErrorAssert{});
  ASSERT_EQ(isize(positions), 3 * isize(_visNodePositions));

  // The visual mesh nodes are at element 1's midpoint.
  // After applying torque, the rod twists and frame axes rotate.
  // Visual nodes that were offset along the y-axis (frame axis direction) should now
  // have an offset that rotates in the y-z plane.

  // Node 0 was at midpoint (0.375, 0, 0) -- should remain approximately there
  EXPECT_NEAR(0.375_r, positions[0], kTolerance);
  EXPECT_NEAR(0_r, positions[1], kTolerance);
  EXPECT_NEAR(0_r, positions[2], kTolerance);

  // Nodes 1 and 2 were offset from the midpoint along frame axis and binormal.
  // After twist, the offset should have rotated around the x-axis.
  // Node 1 was at (0.375, 0.1, 0) -- the offset (0, 0.1, 0) should rotate.
  // Node 2 was at (0.375, 0, 0.1) -- the offset (0, 0, 0.1) should rotate.

  // Verify the x-coordinate remains at the midpoint
  EXPECT_NEAR(0.375_r, positions[3], kTolerance);
  EXPECT_NEAR(0.375_r, positions[6], kTolerance);

  // Compute offsets from the midpoint (node 0) for nodes 1 and 2
  Real3 const node0Pos{positions[0], positions[1], positions[2]};
  Real3 const node1Pos{positions[3], positions[4], positions[5]};
  Real3 const node2Pos{positions[6], positions[7], positions[8]};
  Real3 const offset1 = node1Pos - node0Pos;
  Real3 const offset2 = node2Pos - node0Pos;

  // Verify that the offset magnitudes are preserved (rotation doesn't change length)
  real const node1OffsetMag = Norm(offset1);
  real const node2OffsetMag = Norm(offset2);
  EXPECT_NEAR(0.1_r, node1OffsetMag, kTolerance) << "Node 1 offset magnitude should be preserved";
  EXPECT_NEAR(0.1_r, node2OffsetMag, kTolerance) << "Node 2 offset magnitude should be preserved";

  // Verify that nodes 1 and 2 have rotated: after twist, node 1 should have non-zero z offset
  // and node 2 should have non-zero y offset (both were zero initially).
  // With non-zero torque, there should be measurable rotation.
  real constexpr kRotationThreshold = 1e-4_r;
  EXPECT_GT(Abs(offset1[2]), kRotationThreshold)
      << "Node 1 z-offset should become non-zero after twist";
  EXPECT_GT(Abs(offset2[1]), kRotationThreshold)
      << "Node 2 y-offset should become non-zero after twist";
}

TEST_F(MochiRodVisualMesh, MultiElementBlending) {
  // Test with weightsPerNode = 2, where a visual node is blended across two elements.
  // Also validates skinning Jacobian sparsity with K>1 (exercises deduplication logic).

  // Create a simple visual mesh with one node at the junction between elements 0 and 1.
  // The node at x = 0.25 is exactly at the boundary between element 0 and element 1.
  // We'll blend it 50% from each element.
  // We need at least 3 nodes to form a valid (non-degenerate) triangle for the visual mesh.
  _visNodePositions = {
      Real3{0.25_r, 0.05_r, 0_r}, Real3{0.25_r, 0.06_r, 0_r}, Real3{0.25_r, 0.05_r, 0.01_r}};
  _visTriangles = {Int3{0, 1, 2}};
  _weightsPerNode = 2;
  // All 3 nodes blend 50% from element 0 and 50% from element 1.
  _elementIndices = {0, 1, 0, 1, 0, 1};
  _weights = {0.5_r, 0.5_r, 0.5_r, 0.5_r, 0.5_r, 0.5_r};

  ShapeHandle shape = CreateRodShapeWithVisualMesh();
  Actor* actor = CreateRodActorWithVisualMesh(shape);
  actor->RegisterQuery(QueryType::VisualNodePositions, ErrorAssert{});

  // At reference configuration, position should match the input
  _scene->Step(0_r);
  auto positions = actor->GetVisualMeshNodePositionsLocal(ErrorAssert{});
  ASSERT_EQ(isize(positions), 9); // 3 nodes × 3 components
  for (int n = 0; n < 3; ++n) {
    for (int d = 0; d < 3; ++d) {
      EXPECT_NEAR(_visNodePositions[n][d], positions[3 * n + d], kTolerance)
          << "Node " << n << " component " << d << " at reference config";
    }
  }

  // Validate skinning Jacobian sparsity for K=2 multi-element blending.
  // With K=2, each visual node depends on DoFs from 2 elements.
  // Elements 0 and 1 share rod node 1 (rod has 4 nodes: 0, 1, 2, 3).
  // Element 0 uses nodes 0, 1 and element 1 uses nodes 1, 2.
  // After deduplication, each visual node depends on nodes 0, 1, 2 → 12 DoFs (3 nodes × 4 DoFs).
  {
    auto& reg = GetRegistry();
    auto entity = mochi::GetEntity(reg, actor->GetHandle(), test::ExpectOK{});

    auto const& visualMesh = reg.get<CVisualMesh const>(entity);
    auto const& rodEmbedding = reg.get<CRodVisualMeshEmbedding const>(entity);
    auto const& polylineMesh = reg.get<CPolylineMesh const>(entity);
    auto const& basePose = reg.get<CRodPose<TimeStep::Current> const>(entity);

    CRodContactSkin contactSkin(visualMesh.mesh, rodEmbedding.data);
    CRodContactSkinningData skinningData;
    rod::InitializeContactSkinningJacobian(contactSkin, polylineMesh, skinningData);
    rod::ResolveContactSkinningJacobian(contactSkin, polylineMesh, basePose, skinningData);

    auto const& jac = skinningData.jacobian;
    int const numVisualNodes = jac.Rows();
    EXPECT_EQ(numVisualNodes, 3);

    for (int i = 0; i < numVisualNodes; ++i) {
      int const nnz = isize(jac.Indices(i));
      EXPECT_EQ(nnz, 12) << "Visual node " << i << " should have 12 DoFs (3 nodes × 4 DoFs)";
    }

    VerifySkinningJacobianFD(actor);
  }

  // Apply a small displacement to the middle node (node 1) to create different deformations
  // in elements 0 and 1.
  _scene->SetGravity({0_r, 0_r, 0_r});
  DynamicArray<int> bcDofIndices;
  DynamicArray<real> bcDofValues;
  // Fix nodes 0, 2, 3 at reference position
  for (int node : {0, 2, 3}) {
    for (int d = 0; d < 3; ++d) {
      bcDofIndices.push_back(4 * node + d);
      bcDofValues.push_back(_rodNodes[node][d]);
    }
    bcDofIndices.push_back(4 * node + 3);
    bcDofValues.push_back(0_r);
  }
  // Move node 1 slightly in the y direction
  real constexpr kDisplacement = 0.02_r;
  bcDofIndices.push_back(4);
  bcDofValues.push_back(_rodNodes[1][0]);
  bcDofIndices.push_back(5);
  bcDofValues.push_back(_rodNodes[1][1] + kDisplacement);
  bcDofIndices.push_back(6);
  bcDofValues.push_back(_rodNodes[1][2]);
  bcDofIndices.push_back(7);
  bcDofValues.push_back(0_r);

  actor->AddBoundaryConditionDofsWorld(
      MakeConstSpan(bcDofIndices), MakeConstSpan(bcDofValues), ErrorAssert{});

  _scene->Step(1e6_r);

  positions = actor->GetVisualMeshNodePositionsLocal(ErrorAssert{});
  ASSERT_EQ(isize(positions), 9); // 3 nodes × 3 components

  // The visual nodes at the junction should be the weighted average of the two element
  // reconstructions. Since node 1 moved in y, the visual nodes should also move in y,
  // with the amount depending on the local coordinate within each element's frame.
  // The key assertion is that the blending produces a reasonable result that differs
  // from the single-element case.
  EXPECT_NEAR(0.25_r, positions[0], kTolerance) << "x-position should be preserved";
  EXPECT_GT(positions[1], _visNodePositions[0][1])
      << "y-position should increase due to node 1 displacement";
}

// ============================================================================
// Rod Visual Mesh Contact Tests
// ============================================================================

// Parameterized by whether the box the rod settles on is static (true) or dynamic (false).
// Static box → rod–box contact is async; dynamic box → rod–box contact is sync (same island).
class MochiRodVisualMeshContactOnBox : public test::MochiSceneTestBase,
                                       public ::testing::WithParamInterface<bool> {
 protected:
  static constexpr int kNumNodes = 17;
  static constexpr int kNumElements = kNumNodes - 1;
  static constexpr real kLength = 1_r;
  static constexpr real kRadius = 0.02_r;
  static constexpr int kNumCrossSectionSegments = 16;
  static constexpr real kDensity = 1e3_r;
  static constexpr real kYoungsModulus = 1e5_r;
  static constexpr real kShearModulus = 1e5_r;
  static constexpr real kGravity = -10_r;
  static constexpr real kTimeStep = 1e-2_r;
  // Need stiffer-than-default contact to precisely assert on expected effect of offsetting contact
  // by skin mesh radius.
  static constexpr real kPenaltyCoefficient = 1e11_r;
  static constexpr real kBoxHeight = 0.5_r;
  static constexpr Real3 kBoxSize{2_r, kBoxHeight, 2_r};
  static constexpr real kBoxDensity = 1e4_r;
  static constexpr real kRodInitialHeight = kBoxHeight + 0.1_r;

  Actor* _rodActor = nullptr;

  void SetUp() override {
    test::MochiSceneTestBase::SetUp();

    // The at-rest force-balance check (to 0.1%) relies on backward Euler's damping.
    test::SetSceneIntegrationMethod(_scene, IntegrationMethod::BackwardEuler);

    _scene->SetGravity(Real3{0_r, kGravity, 0_r});

    // Create the rod above the box
    DynamicArray<Real3> nodes;
    DynamicArray<Real3> frameAxes;
    nodes.reserve(kNumNodes);
    frameAxes.reserve(kNumElements);

    for (int i = 0; i < kNumNodes; ++i) {
      real const t = static_cast<real>(i) / static_cast<real>(kNumNodes - 1);
      nodes.push_back(Real3{t * kLength, kRodInitialHeight, 0_r});
      if (i < kNumNodes - 1) {
        frameAxes.push_back(Real3{0_r, 1_r, 0_r});
      }
    }

    RodActorParams rodParams;
    rodParams.name = "TestRod";
    ModelData const model = GenerateTubularRodModelData(
        nodes, frameAxes, kRadius, kNumCrossSectionSegments, /*isClosedLoop=*/false, ErrorAssert{});
    rodParams.shape = _scene->GetContext()->CreateModelShape(model, ErrorAssert{});
    rodParams.useVisualMeshContact = true;
    rodParams.contact.penaltyCoefficient = kPenaltyCoefficient;

    real const area = kPI * Sqr(kRadius);
    real const polarMomentOfInertia = 0.5_r * kPI * Pow(kRadius, 4);
    rodParams.material.linearDensity = kDensity * area;
    rodParams.material.linearRotationalInertia = kDensity * polarMomentOfInertia;
    rodParams.material.axialStiffness = area * kYoungsModulus;
    rodParams.material.flexuralStiffness =
        kYoungsModulus * Real2{0.25_r * kPI * Pow(kRadius, 4), 0.25_r * kPI * Pow(kRadius, 4)};
    real const torsionConstant = 0.5_r * kPI * Pow(kRadius, 4);
    rodParams.material.torsionalStiffness = kShearModulus * torsionConstant;

    _rodActor = CreateRodActor(_scene, rodParams, ErrorAssert{});
    ASSERT_NE(_rodActor, nullptr);
    _rodActor->RegisterQuery(QueryType::TotalContactForce, ErrorAssert{});

    // Create ground plane
    Real3 const planeNormal = Real3{0_r, 1_r, 0_r};
    auto planeShape = _scene->GetContext()->CreatePlaneShape(planeNormal, 0_r, ErrorAssert{});
    RigidActorParams planeParams;
    planeParams.name = "Ground";
    planeParams.isStatic = true;
    planeParams.shape = planeShape;
    planeParams.colliderType = ColliderType::Plane;
    planeParams.contact.penaltyCoefficient = kPenaltyCoefficient;
    _scene->CreateRigidActor(planeParams, ErrorAssert{});

    // Create box on top of the ground plane (static or dynamic based on test parameter)
    auto [coordinates, connectivity] = test::CreateMinimalTetMeshUnitCube(kBoxSize);
    auto boxShape = _mochiContext->CreateTetMeshShape(
        Flatten(MakeSpan(coordinates)), Flatten(MakeSpan(connectivity)), ErrorAssert{});
    RigidActorParams boxParams;
    boxParams.name = "Box";
    boxParams.shape = boxShape;
    boxParams.colliderType = ColliderType::Box;
    boxParams.isStatic = GetParam();
    boxParams.worldFromLocal.SetTranslation(Real3{-0.5_r, 0_r, -1_r});
    boxParams.contact.penaltyCoefficient = kPenaltyCoefficient;
    if (!GetParam()) {
      boxParams.density = kBoxDensity;
    }
    _scene->CreateRigidActor(boxParams, ErrorAssert{});
  }

  void TearDown() override {
    _scene->DestroyActor(_rodActor);
    test::MochiSceneTestBase::TearDown();
  }
};

TEST_P(MochiRodVisualMeshContactOnBox, RodSettlesOnBox) {
  int constexpr kNumSteps = 250;
  for (int step = 0; step < kNumSteps; ++step) {
    _scene->Step(kTimeStep);
  }

  // Read rod DOFs and compute the minimum y-coordinate of the centerline nodes
  int const numDofs = _rodActor->GetNumDofs();
  DynamicArray<int> dofIndices(numDofs);
  std::iota(dofIndices.begin(), dofIndices.end(), 0);
  DynamicArray<real> dofs(numDofs);
  _rodActor->GetDofValues(dofIndices, dofs, ErrorAssert{});

  real minY = std::numeric_limits<real>::max();
  for (int i = 0; i < kNumNodes; ++i) {
    // Rod DoFs: [dx0, dy0, dz0, twist0, dx1, dy1, dz1, twist1, ...]
    real const defY = kRodInitialHeight + dofs[4 * i + 1];
    minY = Min(minY, defY);
  }

  // Verify the total contact force balances the rod's weight. The only source of error
  // is solver convergence, so a tight tolerance is appropriate.
  Real3 const contactForce = _rodActor->GetContactForceWorld(ErrorAssert{});
  real const expectedWeight = kDensity * kPI * Sqr(kRadius) * kLength * (-kGravity);
  real constexpr kForceRelativeTolerance = 1e-3_r;
  EXPECT_NEAR(contactForce[1], expectedWeight, kForceRelativeTolerance * expectedWeight)
      << "Contact force y-component should balance the rod weight";

  // The rod should settle on the box. With visual mesh contact and radius kRadius,
  // the centerline should be approximately kRadius (plus contact threshold) above the
  // box top surface. A loose relative tolerance accounts for finite penalty stiffness
  // and geometric discretization of the tubular visual mesh.
  real constexpr kHeightRelativeTolerance = 0.1_r;
  real constexpr kExpectedHeight = kBoxHeight + kRadius + ContactParams{}.penaltyThresholdDefault;
  EXPECT_NEAR(minY, kExpectedHeight, kHeightRelativeTolerance * (kExpectedHeight - kBoxHeight))
      << "Rod centerline should settle at approximately kRadius above the box top";
  EXPECT_GT(minY, kBoxHeight) << "Rod centerline should be above the box top";
}

INSTANTIATE_TEST_SUITE_P(
    BoxIsStatic,
    MochiRodVisualMeshContactOnBox,
    ::testing::Bool(),
    [](::testing::TestParamInfo<bool> const& info) {
      return info.param ? "StaticBox" : "DynamicBox";
    });

// =============================================================================
// Periodic (closed-loop) rod tests
// =============================================================================

TEST(MochiClosedLoopRod, StencilGeneration) {
  // Test that periodic stencil generation produces uniform 3-node stencils with modular wrapping
  int constexpr kNumNodes = 5;
  auto const [connectivity, stencil] =
      GenerateRodConnectivityAndStencil(kNumNodes, /*isClosedLoop=*/true);

  // All nodes should have exactly 3-node stencils
  for (int i = 0; i < kNumNodes; ++i) {
    auto const row = connectivity[i];
    EXPECT_EQ(isize(row), 3) << "Closed-loop rod: all stencils need 3 nodes (node " << i << ")";
    EXPECT_EQ(row[0], i);
    EXPECT_EQ(row[1], (i + 1) % kNumNodes);
    EXPECT_EQ(row[2], (i + 2) % kNumNodes);
  }
}

TEST(MochiClosedLoopRod, StencilGenerationOpen) {
  // Verify that open-polyline stencils are unchanged
  int constexpr kNumNodes = 5;
  auto const [connectivity, stencil] =
      GenerateRodConnectivityAndStencil(kNumNodes, /*isClosedLoop=*/false);

  // First node: 3 entries [0, 1, 2]
  EXPECT_EQ(isize(connectivity[0]), 3);
  // Second-to-last node: 2 entries [3, 4]
  EXPECT_EQ(isize(connectivity[kNumNodes - 2]), 2);
  // Last node: 1 entry [4]
  EXPECT_EQ(isize(connectivity[kNumNodes - 1]), 1);
}

TEST(MochiClosedLoopRod, BishopFrameNonzeroHolonomy) {
  // Test discrete Bishop frame for a closed-loop curve with nonzero holonomy. The resulting frame
  // axes won't strictly be a Bishop frame (parallel transport), because the holonomy correction
  // deviates from parallel transport. However, the axes should vary continuously around the loop,
  // with adjacent elements having similar orientations that converge under refinement.

  // Helper to generate a torus knot curve
  auto generateTorusKnot = [](int numNodes, real scale) {
    DynamicArray<Real3> nodes;
    nodes.reserve(numNodes);
    for (int i = 0; i < numNodes; ++i) {
      real const theta = 2_r * kPI * static_cast<real>(i) / static_cast<real>(numNodes);
      real const pTheta = 2_r * theta;
      real const qTheta = 3_r * theta;
      real const twoPlusCosPTheta = (2_r + Cos(pTheta));
      nodes.push_back(
          scale *
          Real3{twoPlusCosPTheta * Cos(qTheta), twoPlusCosPTheta * Sin(qTheta), Sin(pTheta)});
    }
    return nodes;
  };

  real constexpr kCurveScale = 0.1_r;
  real constexpr kTolerance = 1e-4_r;

  // Test with multiple resolutions (powers of two) to verify convergence under refinement
  int constexpr kMinPower = 5; // 2^5 = 32 nodes
  int constexpr kMaxPower = 7; // 2^7 = 128 nodes

  real prevMaxAngle = 0_r;
  for (int power = kMinPower; power <= kMaxPower; ++power) {
    int const numNodes = 1 << power;
    DynamicArray<Real3> nodes = generateTorusKnot(numNodes, kCurveScale);
    auto const axes = GenerateDiscreteBishopFrame(MakeConstSpan(nodes), /*isClosedLoop=*/true);

    EXPECT_EQ(isize(axes), numNodes) << "Wrong number of axes for " << numNodes << " nodes";

    // Basic validity checks: unit length and orthogonality to tangents
    for (int i = 0; i < numNodes; ++i) {
      // Unit length
      EXPECT_NEAR(Norm(axes[i]), 1_r, kTolerance)
          << "Frame axis " << i << " should be unit-length for " << numNodes << " nodes";

      // Perpendicular to element tangent
      Real3 const tangent = Normalize(nodes[(i + 1) % numNodes] - nodes[i]);
      EXPECT_NEAR(Abs(Dot(axes[i], tangent)), 0_r, kTolerance)
          << "Frame axis " << i << " should be perpendicular to tangent for " << numNodes
          << " nodes";
    }

    // Check continuity: adjacent axes should be close (no large jumps)
    // The maximum angle between consecutive axes should decrease with refinement
    real maxAngleBetweenConsecutive = 0_r;
    for (int i = 0; i < numNodes; ++i) {
      int const j = (i + 1) % numNodes;
      real const dot = Clamp(Dot(axes[i], axes[j]), -1_r, 1_r);
      real const angle = ACos(dot);
      maxAngleBetweenConsecutive = Max(maxAngleBetweenConsecutive, angle);

      // No 180-degree flips between consecutive axes
      EXPECT_GT(dot, kTolerance - 1_r)
          << "Axes " << i << " and " << j << " have a 180-degree jump for " << numNodes << " nodes";
    }

    // With holonomy correction, the frame should be continuous across the closing edge.
    // The maximum angle between consecutive elements should scale roughly with 1/numNodes
    // (since the curve is smooth and we're sampling more densely).
    // For this torus knot, the maximum angle should be reasonable even at coarse resolution.
    // Use a generous bound since the torus knot has regions of higher curvature.
    real const expectedMaxAngle = 2_r * kPI / static_cast<real>(numNodes) * 5_r; // generous bound
    EXPECT_LT(maxAngleBetweenConsecutive, expectedMaxAngle)
        << "Max angle between consecutive axes (" << maxAngleBetweenConsecutive
        << " rad) exceeds expected bound for " << numNodes << " nodes";

    // Verify that refinement reduces the maximum angle between consecutive axes
    // (axes of adjacent elements should approach each other under refinement)
    if (power > kMinPower) {
      // The angle should roughly halve when doubling resolution (for smooth curves)
      // We use a generous factor of 0.55 to account for preasymptotic behavior.
      EXPECT_LT(maxAngleBetweenConsecutive, prevMaxAngle * 0.55_r)
          << "Max angle should decrease with refinement: " << prevMaxAngle << " (2^" << (power - 1)
          << " nodes) vs " << maxAngleBetweenConsecutive << " (2^" << power << " nodes)";
    }
    prevMaxAngle = maxAngleBetweenConsecutive;

    // Check continuity across the closing edge: the frame should wrap around smoothly
    // (no discontinuity from last element back to first)
    Vec4r const lastTangent = ToSimd(Normalize(nodes[0] - nodes[numNodes - 1]));
    Vec4r const firstTangent = ToSimd(Normalize(nodes[1] - nodes[0]));
    VMatrix3x3r const P = fem::ParallelTransportOperator(lastTangent, firstTangent);
    Vec4r const transported = DotMatVec3x3(P, ToSimd(axes[numNodes - 1]));
    Real3 const diff = axes[0] - ToReal3(transported);
    // With holonomy correction, the transported last axis should be close to the first axis.
    // The error should decrease with refinement as the discretization improves.
    // Scale threshold by 1/sqrt(numNodes) since error should decrease with refinement.
    real const continuityThreshold = 0.1_r / static_cast<real>(numNodes);
    EXPECT_LT(Norm(diff), continuityThreshold)
        << "Frame should be continuous across closing edge for " << numNodes
        << " nodes (diff norm: " << Norm(diff) << ")";
  }
}

// ============================================================================
// Rod actor creation validation
// ============================================================================

class MochiRodActorValidationTest : public test::MochiSceneTestBase {};

TEST_F(MochiRodActorValidationTest, CreationRejectsInvalidContactParams) {
  DynamicArray<Real3> const nodes{{0_r, 0_r, 0_r}, {1_r, 0_r, 0_r}};
  DynamicArray<Real3> const frameAxes{Real3{0_r, 1_r, 0_r}};
  ShapeHandle const shape = CreatePolylineShape(
      _scene->GetContext(), nodes, frameAxes, /*isClosedLoop=*/false, test::ExpectOK{});
  RodActorParams params;
  params.shape = shape;
  params.contact.penaltyThresholdDefault = std::numeric_limits<real>::infinity();

  EXPECT_EQ(nullptr, CreateRodActor(_scene, params, test::ExpectNotOK{}));
}

class MochiClosedLoopRodActorTest : public test::MochiSceneTestBase {};

TEST_F(MochiClosedLoopRodActorTest, CreateAndStep) {
  // Create a closed-loop rod actor and verify it can be stepped without errors
  int constexpr kNumNodes = 8;
  real constexpr kRadius = 0.5_r;

  DynamicArray<Real3> nodes;
  nodes.reserve(kNumNodes);
  for (int i = 0; i < kNumNodes; ++i) {
    real const theta = 2_r * kPI * static_cast<real>(i) / static_cast<real>(kNumNodes);
    nodes.push_back(Real3{kRadius * Cos(theta), 0_r, kRadius * Sin(theta)});
  }

  RodActorParams actorParams;
  actorParams.name = "ClosedLoopRod";
  actorParams.shape =
      CreatePolylineShape(_scene->GetContext(), nodes, {}, /*isClosedLoop=*/true, ErrorAssert{});

  real constexpr kCrossSectionRadius = 0.01_r;
  real constexpr kArea = kPI * Sqr(kCrossSectionRadius);
  actorParams.material.linearDensity = 1000_r * kArea;
  actorParams.material.linearRotationalInertia = 1000_r * 0.5_r * kPI * Pow(kCrossSectionRadius, 4);
  actorParams.material.axialStiffness = 1e6_r * kArea;
  actorParams.material.flexuralStiffness = 1e6_r *
      Real2{0.25_r * kPI * Pow(kCrossSectionRadius, 4), 0.25_r * kPI * Pow(kCrossSectionRadius, 4)};
  actorParams.material.torsionalStiffness = 1e6_r * 0.5_r * kPI * Pow(kCrossSectionRadius, 4);

  Actor* actor = CreateRodActor(_scene, actorParams, ErrorAssert{});
  ASSERT_NE(actor, nullptr);

  // Verify element count = node count for closed-loop rods
  EXPECT_EQ(actor->GetMesh().GetNumElements(), kNumNodes);

  // Step the simulation
  _scene->SetGravity(Real3{0_r, -10_r, 0_r});
  for (int i = 0; i < 5; ++i) {
    _scene->Step(0.01_r);
  }
}

TEST_F(MochiClosedLoopRodActorTest, MassDistribution) {
  // Create a closed-loop rod actor (circular ring) and verify total mass and nodal mass
  // distribution
  int constexpr kNumNodes = 8;
  real constexpr kRadius = 0.5_r;
  real constexpr kLinearDensity = 100_r;

  DynamicArray<Real3> nodes;
  nodes.reserve(kNumNodes);
  for (int i = 0; i < kNumNodes; ++i) {
    real const theta = 2_r * kPI * static_cast<real>(i) / static_cast<real>(kNumNodes);
    nodes.push_back(Real3{kRadius * Cos(theta), 0_r, kRadius * Sin(theta)});
  }

  RodActorParams actorParams;
  actorParams.shape =
      CreatePolylineShape(_scene->GetContext(), nodes, {}, /*isClosedLoop=*/true, ErrorAssert{});
  actorParams.material.linearDensity = kLinearDensity;

  Actor* actor = CreateRodActor(_scene, actorParams, ErrorAssert{});
  ASSERT_NE(actor, nullptr);

  // Compute expected total length (circumference of the regular polygon)
  auto& reg = GetRegistry();
  auto entity = mochi::GetEntity(reg, actor->GetHandle(), test::ExpectOK{});
  auto const& mesh = reg.get<CPolylineMesh const>(entity);
  real expectedTotalLength = 0_r;
  for (int i = 0; i < kNumNodes; ++i) {
    Int2 const en = mesh.ElementNodes(i);
    expectedTotalLength += Norm(mesh.nodes[en[1]] - mesh.nodes[en[0]]);
  }

  // Verify total mass
  real const expectedTotalMass = kLinearDensity * expectedTotalLength;
  EXPECT_NEAR(actor->GetMass(test::ExpectOK{}), expectedTotalMass, 1e-4_r * expectedTotalMass);

  // Verify all nodal masses are equal (circular ring with equal-sized elements)
  auto const& nodalMasses = reg.get<CNodalMasses const>(entity);
  ASSERT_EQ(isize(nodalMasses.values), kNumNodes);
  real const expectedNodalMass = expectedTotalMass / static_cast<real>(kNumNodes);
  for (int i = 0; i < kNumNodes; ++i) {
    EXPECT_NEAR(nodalMasses.values[i], expectedNodalMass, 1e-4_r * expectedNodalMass)
        << "Nodal mass mismatch at node " << i;
  }
}

TEST_F(MochiClosedLoopRodActorTest, GetShapeMeshEncodesPeriodicityViaConnectivity) {
  // Periodicity is now exposed through the connectivity array returned by
  // Context::GetShapeMesh. Closed-loop polylines: 2 * numNodes entries; open: 2 * (numNodes - 1).
  int constexpr kNumNodes = 6;
  real constexpr kRadius = 0.5_r;

  DynamicArray<Real3> nodes;
  nodes.reserve(kNumNodes);
  for (int i = 0; i < kNumNodes; ++i) {
    real const theta = 2_r * kPI * static_cast<real>(i) / static_cast<real>(kNumNodes);
    nodes.push_back(Real3{kRadius * Cos(theta), 0_r, kRadius * Sin(theta)});
  }

  ShapeHandle const periodicShape =
      CreatePolylineShape(_scene->GetContext(), nodes, {}, /*isClosedLoop=*/true, ErrorAssert{});
  MeshDataView const periodicView =
      _scene->GetContext()->GetShapeMesh(periodicShape, ErrorAssert{});
  ASSERT_EQ(periodicView.nodesPerElement, 2);
  EXPECT_EQ(isize(periodicView.connectivity), 2 * kNumNodes);
  EXPECT_EQ(periodicView.connectivity[2 * (kNumNodes - 1) + 1], 0);

  ShapeHandle const openShape =
      CreatePolylineShape(_scene->GetContext(), nodes, {}, /*isClosedLoop=*/false, ErrorAssert{});
  MeshDataView const openView = _scene->GetContext()->GetShapeMesh(openShape, ErrorAssert{});
  ASSERT_EQ(openView.nodesPerElement, 2);
  EXPECT_EQ(isize(openView.connectivity), 2 * (kNumNodes - 1));
}

TEST_P(MochiRodActorStraightRodScene, ExternalForceEnergyResidualConsistency) {
  int constexpr kNumNodes = 5;
  int const lastNode = kNumNodes - 1;

  // Apply external forces on displacement DOFs (world coords) and a twist DOF.
  Real3 const forceWorldDir = _worldFromLocal.TransformDirection(Real3{0.3_r, -0.5_r, 0.7_r});
  DynamicArray<int> forceDofs;
  DynamicArray<real> forceValues;
  for (int d = 0; d < fem::kRodThetaDofOffset; ++d) {
    forceDofs.push_back(fem::kNumRodFields * lastNode + d);
    forceValues.push_back(10_r * forceWorldDir[d]);
  }
  forceDofs.push_back(fem::kNumRodFields * 1 + fem::kRodThetaDofOffset);
  forceValues.push_back(5_r);
  _actor->SetExternalForcesOnDofs(forceDofs, forceValues, ErrorAssert{});

  _scene->Step(0.01_r);

  auto& reg = GetRegistry();
  auto entity = mochi::GetEntity(reg, _actor->GetHandle(), test::ExpectOK{});

  // Assemble baseline objective and residual.
  AssemblyParams const params{.assemObj = true, .assemRes = true, .assemDRes = false};
  ecs::InvokeOnEntity(rod::AssembleBody, reg, entity, std::cref(params));

  auto const& snle = reg.get<CActorSnle const>(entity);
  ColumnVector<real> const r0(snle.fullResidual);

  // Finite-difference verification: dE/du_i ≈ R_i for each DOF.
  // Perturbations use ApplyLieDeltaToPose to keep displacements and frame axes consistent.
  AssemblyParams const objOnly{.assemObj = true, .assemRes = false, .assemDRes = false};
  auto& currPose = reg.get<CRodPose<TimeStep::Current>>(entity);
  auto const& polylineMesh = reg.get<CPolylineMesh const>(entity);
  int const numDofs = r0.Rows();
  int const numElements = polylineMesh.NumElements();

  // Save baseline pose.
  ColumnVector<real> const baseDisp(currPose.value.displacements);
  DynamicArray<Real3> const baseAxes(currPose.value.frameAxes);

  real constexpr kEps = MOCHI_USE_DOUBLE_PRECISION ? 1e-7_r : 1e-4_r;
  real constexpr kRelTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-5_r : 2e-2_r;

  ColumnVector<real> dofDelta = ColumnVector<real>::Zero(numDofs);
  ColumnVector<real> pertDisp(numDofs);
  DynamicArray<Real3> pertAxes(numElements);

  auto const evalObjAt = [&](real eps, int dof) {
    dofDelta(dof) = eps;
    rod::ApplyLieDeltaToPose(
        polylineMesh.nodes,
        baseDisp,
        MakeConstSpan(baseAxes),
        dofDelta,
        pertDisp,
        MakeSpan(pertAxes));
    currPose.value.displacements = pertDisp;
    currPose.value.frameAxes = pertAxes;
    ecs::InvokeOnEntity(rod::AssembleBody, reg, entity, std::cref(objOnly));
    dofDelta(dof) = 0_r;
    return snle.objective;
  };

  for (int i = 0; i < numDofs; ++i) {
    double const ePlus = evalObjAt(kEps, i);
    double const eMinus = evalObjAt(-kEps, i);

    double const fdGrad = (ePlus - eMinus) / (2.0 * static_cast<double>(kEps));
    double const scale = Max(1.0, Max(Abs(fdGrad), Abs(double(r0[i]))));
    EXPECT_NEAR(fdGrad, double(r0[i]), double(kRelTol) * scale) << "DOF " << i;
  }
}

// Builds `numNodes` points equally spaced by `angleStep` (signed) on a circle of `radius` in the XY
// plane. Pass angleStep = 2*pi/numNodes for a closed regular polygon; negative for clockwise.
static DynamicArray<Real3> CircularPolyline(int numNodes, real radius, real angleStep) {
  DynamicArray<Real3> nodes(numNodes);
  for (int i = 0; i < numNodes; ++i) {
    real const theta = angleStep * static_cast<real>(i);
    nodes[i] = Real3{radius * Cos(theta), radius * Sin(theta), 0_r};
  }
  return nodes;
}

// Verifies curvature binormals for equally-spaced circular samples against the analytical value:
// every node (interior, or all nodes for a closed loop) equals (0, 0, sign/(radius*cos(angle/2)));
// open-loop endpoints are zero. The cross product of consecutive CCW edges points along +Z.
static void
ExpectCircularCurvature(Span<Real3 const> nodes, bool isClosedLoop, real radius, real angleStep) {
  int const numNodes = isize(nodes);
  DynamicArray<Real3> out(numNodes);
  rod::ComputeRodNodeCurvatureBinormals(nodes, isClosedLoop, MakeSpan(out));

  Real3 const expected{0_r, 0_r, Sign(angleStep) / (radius * Cos(0.5_r * angleStep))};
  for (int i = 0; i < numNodes; ++i) {
    real constexpr kTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-12_r : 1e-4_r;
    bool const isOpenEndpoint = !isClosedLoop && (i == 0 || i == numNodes - 1);
    EXPECT_NEAR_TOL(out[i], isOpenEndpoint ? Real3{} : expected, kTol);
  }
}

// Closed regular polygon: every node, including node 0 and the last node (which rely on modular
// wraparound indexing), matches the analytical curvature 1/(R*cos(pi/N)).
TEST(RodCurvatureBinormal, ClosedRegularPolygonMatchesAnalyticalCurvature) {
  int constexpr kNumNodes = 6;
  real constexpr kRadius = 2_r;
  real constexpr kAngleStep = 2_r * kPI / static_cast<real>(kNumNodes);
  auto const nodes = CircularPolyline(kNumNodes, kRadius, kAngleStep);
  ExpectCircularCurvature(MakeConstSpan(nodes), /*isClosedLoop*/ true, kRadius, kAngleStep);
}

// Open circular arc: interior nodes match the analytical curvature; both endpoints are zero.
TEST(RodCurvatureBinormal, OpenArcZeroesEndpointsAndMatchesInterior) {
  int constexpr kNumNodes = 5;
  real constexpr kRadius = 1.5_r;
  real constexpr kAngleStep = kPI / 12_r;
  auto const nodes = CircularPolyline(kNumNodes, kRadius, kAngleStep);
  ExpectCircularCurvature(MakeConstSpan(nodes), /*isClosedLoop*/ false, kRadius, kAngleStep);
}

// Reversing the winding (clockwise) flips the binormal direction.
TEST(RodCurvatureBinormal, ReversedWindingFlipsSign) {
  int constexpr kNumNodes = 6;
  real constexpr kRadius = 2_r;
  real constexpr kAngleStep = -2_r * kPI / static_cast<real>(kNumNodes);
  auto const nodes = CircularPolyline(kNumNodes, kRadius, kAngleStep);
  ExpectCircularCurvature(MakeConstSpan(nodes), /*isClosedLoop*/ true, kRadius, kAngleStep);
}

// Single 90-degree corner (open, 3 nodes): hand-computed binormal, zero at the two endpoints.
// e0=(1,0,0), e1=(0,1,0): kappa_b = 2*(e0 x e1)/(1 + e0.e1) = (0,0,2); averaged length L = 1.
TEST(RodCurvatureBinormal, SingleCornerMatchesHandComputedValue) {
  DynamicArray<Real3> const nodes{{-1_r, 0_r, 0_r}, {0_r, 0_r, 0_r}, {0_r, 1_r, 0_r}};
  DynamicArray<Real3> out(3);
  rod::ComputeRodNodeCurvatureBinormals(
      MakeConstSpan(nodes), /*isClosedLoop*/ false, MakeSpan(out));
  real constexpr kTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-12_r : 1e-6_r;
  EXPECT_NEAR_TOL(out[0], Real3{}, kTol);
  EXPECT_NEAR_TOL(out[1], (Real3{0_r, 0_r, 2_r}), kTol);
  EXPECT_NEAR_TOL(out[2], Real3{}, kTol);
}

// A straight (collinear) polyline has zero curvature at every node.
TEST(RodCurvatureBinormal, StraightLineHasZeroCurvature) {
  DynamicArray<Real3> const nodes{
      {0_r, 0_r, 0_r}, {1_r, 0_r, 0_r}, {2_r, 0_r, 0_r}, {3_r, 0_r, 0_r}};
  DynamicArray<Real3> out(4);
  rod::ComputeRodNodeCurvatureBinormals(
      MakeConstSpan(nodes), /*isClosedLoop*/ false, MakeSpan(out));
  real constexpr kTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-12_r : 1e-6_r;
  for (auto const& k : out) {
    EXPECT_NEAR_TOL(k, Real3{}, kTol);
  }
}

// ============================================================================
// Rod damping (actor-level)
// ============================================================================

namespace {

// Fixture providing helpers to build straight rods and run damped/undamped single steps.
class MochiRodDampingTest : public test::MochiSceneTestBase {
 protected:
  // Straight rod of `numNodes` nodes spaced 1 m along +x, with the given material.
  Actor* CreateStraightRod(int numNodes, RodMaterialParams const& material) {
    DynamicArray<Real3> nodes;
    for (int i = 0; i < numNodes; ++i) {
      nodes.push_back(Real3{static_cast<real>(i), 0_r, 0_r});
    }
    DynamicArray<Real3> const frameAxes(numNodes - 1, Real3{0_r, 1_r, 0_r});
    ShapeHandle const shape = CreatePolylineShape(
        _scene->GetContext(), nodes, frameAxes, /*isClosedLoop=*/false, ErrorAssert{});
    RodActorParams params;
    params.shape = shape;
    params.material = material;
    return CreateRodActor(_scene, params, ErrorAssert{});
  }

  // Backward Euler with an exact direct linear solve (LU, no preconditioner) and a fully-converged
  // nonlinear solve. The rod's element frame axes (which carry the twist/bend strain) are refreshed
  // between Newton iterations, so a pure-twist mode is only resisted from the second iteration
  // onward; forcing several tightly-converged iterations (and disabling early stopping) is required
  // for the implicit step to capture the torsional response.
  void UseBackwardEulerDirectSolver() {
    SolverParams sp = _scene->GetSolverParams();
    sp.integrationMethod = IntegrationMethod::BackwardEuler;
    sp.linearSolver.solverType = LinearSolverType::LU;
    sp.linearSolver.preconditionerType = PreconditionerType::None;
    _scene->SetSolverParams(sp, ErrorAssert{});
  }

  static DynamicArray<real> GetDofs(Actor* actor) {
    DynamicArray<real> dofs(actor->GetNumDofs());
    actor->GetDofValues({}, dofs, ErrorAssert{});
    return dofs;
  }

  // Material with positive everything; callers override stiffness as needed.
  static RodMaterialParams BaseMaterial() {
    RodMaterialParams m;
    m.linearDensity = 1_r;
    m.linearRotationalInertia = 1_r;
    m.axialStiffness = 1e3_r;
    m.torsionalStiffness = 1e1_r;
    m.flexuralStiffness = {1e1_r, 1e1_r};
    return m;
  }
};

} // namespace

// CreateRodActor rejects non-finite / negative damping coefficients and accepts a small positive
// pair.
TEST_F(MochiRodDampingTest, ValidationRejectsInvalidDampingCoefficients) {
  auto const rejected = [&](auto mutate) {
    RodMaterialParams m = BaseMaterial();
    mutate(m);
    DynamicArray<Real3> const nodes{{0_r, 0_r, 0_r}, {1_r, 0_r, 0_r}};
    DynamicArray<Real3> const frameAxes{Real3{0_r, 1_r, 0_r}};
    ShapeHandle const shape = CreatePolylineShape(
        _scene->GetContext(), nodes, frameAxes, /*isClosedLoop=*/false, ErrorAssert{});
    RodActorParams params;
    params.shape = shape;
    params.material = m;
    test::ExpectNotOK error;
    EXPECT_EQ(nullptr, CreateRodActor(_scene, params, error));
  };
  real const kNan = std::numeric_limits<real>::quiet_NaN();
  rejected([](RodMaterialParams& m) { m.massDampingCoefficient = -1_r; });
  rejected([&](RodMaterialParams& m) { m.massDampingCoefficient = kNan; });
  rejected([](RodMaterialParams& m) { m.stiffnessDampingCoefficient = -1_r; });
  rejected([&](RodMaterialParams& m) { m.stiffnessDampingCoefficient = kNan; });

  // A valid small positive pair succeeds.
  RodMaterialParams valid = BaseMaterial();
  valid.massDampingCoefficient = 0.1_r;
  valid.stiffnessDampingCoefficient = 0.01_r;
  EXPECT_NE(nullptr, CreateStraightRod(2, valid));
}

// Mass-proportional damping decays a uniform translational velocity by the analytic backward-Euler
// factor v_n = v0 / (1 + alpha*dt)^n. The undamped (alpha = 0) case conserves velocity. Stiffness
// damping is parameterized too: under rigid translation the strain stays zero so it is inert, but a
// nonzero coefficient switches assembly to the full-stencil stage-start gather (with mass damping
// reusing its leading sub-span), guarding that shared gather against corrupting the mass target.
TEST_F(MochiRodDampingTest, MassDampingVelocityDecay) {
  _scene->SetGravity({0_r, 0_r, 0_r});
  UseBackwardEulerDirectSolver();

  int constexpr kNumNodes = 3;
  real constexpr kDt = 1e-2_r;
  int constexpr kNumSteps = 5;
  real constexpr kV0 = 2_r;

  auto const run = [&](real alpha, real beta) {
    RodMaterialParams m = BaseMaterial();
    m.massDampingCoefficient = alpha;
    m.stiffnessDampingCoefficient = beta;
    Actor* actor = CreateStraightRod(kNumNodes, m);
    MOCHI_DEFER(_scene->DestroyActor(actor));

    // Uniform initial translational velocity in +x for every node (rigid translation => no strain).
    DynamicArray<real> vel(kNumNodes * 4, 0_r);
    for (int n = 0; n < kNumNodes; ++n) {
      vel[n * 4 + 0] = kV0;
    }
    actor->SetNodeVelocitiesLocal(MakeConstSpan(vel), ErrorAssert{});

    real const kRelTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-5_r : 5e-3_r;
    DynamicArray<real> prev = GetDofs(actor);
    real expectedVel = kV0;
    for (int step = 0; step < kNumSteps; ++step) {
      _scene->Step(kDt);
      DynamicArray<real> const curr = GetDofs(actor);
      // Recover velocity of node 0's x-DoF from the displacement increment.
      real const measuredVel = (curr[0] - prev[0]) / kDt;
      expectedVel = kV0 / Pow(1_r + alpha * kDt, static_cast<real>(step + 1));
      EXPECT_NEAR_RTOL(expectedVel, measuredVel, kRelTol)
          << "alpha=" << alpha << " beta=" << beta << " step=" << step;
      prev = curr;
    }
  };

  run(3_r, 0_r); // mass damping only
  run(3_r, 0.5_r); // mass + stiffness damping; stiffness inert under rigid translation
  run(0_r, 0_r); // undamped control: velocity conserved (expectedVel stays kV0)
}

// Rigid translation produces zero strain, so stiffness damping has no effect: velocity is conserved
// even with a large stiffness-damping coefficient.
TEST_F(MochiRodDampingTest, RigidTranslationUnaffectedByStiffnessDamping) {
  _scene->SetGravity({0_r, 0_r, 0_r});
  UseBackwardEulerDirectSolver();

  int constexpr kNumNodes = 3;
  real constexpr kDt = 1e-2_r;
  real constexpr kV0 = 2_r;

  RodMaterialParams m = BaseMaterial();
  m.stiffnessDampingCoefficient = 0.5_r; // large, but inactive for rigid translation
  Actor* actor = CreateStraightRod(kNumNodes, m);
  MOCHI_DEFER(_scene->DestroyActor(actor));

  DynamicArray<real> vel(kNumNodes * 4, 0_r);
  for (int n = 0; n < kNumNodes; ++n) {
    vel[n * 4 + 0] = kV0;
  }
  actor->SetNodeVelocitiesLocal(MakeConstSpan(vel), ErrorAssert{});

  real const kRelTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-5_r : 5e-3_r;
  DynamicArray<real> prev = GetDofs(actor);
  for (int step = 0; step < 4; ++step) {
    _scene->Step(kDt);
    DynamicArray<real> const curr = GetDofs(actor);
    real const measuredVel = (curr[0] - prev[0]) / kDt;
    EXPECT_NEAR_RTOL(kV0, measuredVel, kRelTol) << "step=" << step;
    prev = curr;
  }
}

namespace {

// A single backward-Euler step from an initial velocity, at a realistic (small, dynamic) timestep
// where the stiff elastic mode is under-resolved — exactly the regime where stiffness damping is
// useful. Stiffness damping contributes an effective stiffness K*(1 + beta/dt), so the damped
// single-step displacement is (m*v0/dt)/(m/dt^2 + K*(1+beta/dt)) versus (m*v0/dt)/(m/dt^2 + K)
// undamped. With the mode in the stiff regime (K >> m/dt^2) the ratio approaches dt/(dt+beta). The
// velocity is kept small so the predictor (dt*v0) stays in the linear-kinematics regime.
real constexpr kRatioDt = 0.1_r;
real constexpr kRatioBeta = 0.1_r; // beta/dt = 1 => ratio dt/(dt+beta) = 0.5
real constexpr kRatioStiffness =
    1e4_r; // stiff vs the inertia scale m/dt^2 (~1e2), well-conditioned
real constexpr kRatioV0 = 1_r; // small initial velocity => predictor dt*v0 stays linear
real constexpr kRatioExpected = kRatioDt / (kRatioDt + kRatioBeta);
real constexpr kRatioTol = 5e-2_r;

} // namespace

// Axial stiffness damping: a 2-node rod given an axial velocity at its free node displaces over one
// step by ~dt/(dt+beta) of the undamped amount.
TEST_F(MochiRodDampingTest, AxialStiffnessDampingDecayRatio) {
  _scene->SetGravity({0_r, 0_r, 0_r});
  UseBackwardEulerDirectSolver();

  auto const runFreeDisp = [&](real beta) {
    RodMaterialParams m = BaseMaterial();
    m.axialStiffness = kRatioStiffness;
    m.flexuralStiffness = {0_r, 0_r};
    m.torsionalStiffness = 0_r; // ok: rotational inertia is nonzero
    m.stiffnessDampingCoefficient = beta;
    Actor* actor = CreateStraightRod(2, m);
    MOCHI_DEFER(_scene->DestroyActor(actor));

    DynamicArray<int> const bcDofs{0, 1, 2, 3}; // fix node 0
    DynamicArray<real> const bcVals{0_r, 0_r, 0_r, 0_r};
    actor->AddBoundaryConditionDofsWorld(
        MakeConstSpan(bcDofs), MakeConstSpan(bcVals), ErrorAssert{});

    DynamicArray<real> vel(2 * 4, 0_r);
    vel[1 * 4 + 0] = kRatioV0; // axial velocity on the free node
    actor->SetNodeVelocitiesLocal(MakeConstSpan(vel), ErrorAssert{});

    _scene->Step(kRatioDt);
    return GetDofs(actor)[1 * 4 + 0];
  };

  real const damped = runFreeDisp(kRatioBeta);
  real const undamped = runFreeDisp(0_r);
  EXPECT_NEAR_RTOL(kRatioExpected, damped / undamped, kRatioTol);
}

// Bending stiffness damping: a cantilever (first two nodes constrained, isolating bending from
// membrane coupling) given a transverse tip velocity deflects over one step by ~dt/(dt+beta) of the
// undamped amount.
TEST_F(MochiRodDampingTest, BendingStiffnessDampingDecayRatio) {
  _scene->SetGravity({0_r, 0_r, 0_r});
  UseBackwardEulerDirectSolver();
  int constexpr kNumNodes = 4;
  int constexpr kTip = kNumNodes - 1;

  auto const runTipDisp = [&](real beta) {
    RodMaterialParams m = BaseMaterial();
    m.linearDensity = 1e-3_r; // light beam: puts the bending mode in the stiff regime vs inertia
    m.flexuralStiffness = {kRatioStiffness, kRatioStiffness};
    m.torsionalStiffness = 0_r;
    m.stiffnessDampingCoefficient = beta;
    Actor* actor = CreateStraightRod(kNumNodes, m);
    MOCHI_DEFER(_scene->DestroyActor(actor));

    // Cantilever clamp: constrain the translations of the first two nodes (fixes the root position
    // and tangent) and leave the rest free. Translation BCs take world positions, so pin each node
    // at its reference position.
    DynamicArray<int> bcDofs;
    DynamicArray<real> bcVals;
    for (int node : {0, 1}) {
      for (int d = 0; d < 3; ++d) {
        bcDofs.push_back(node * 4 + d);
        bcVals.push_back(
            d == 0 ? static_cast<real>(node) : 0_r); // node reference position (x=node)
      }
    }
    actor->AddBoundaryConditionDofsWorld(
        MakeConstSpan(bcDofs), MakeConstSpan(bcVals), ErrorAssert{});

    DynamicArray<real> vel(kNumNodes * 4, 0_r);
    vel[kTip * 4 + 1] = kRatioV0; // transverse (y) velocity at the tip
    actor->SetNodeVelocitiesLocal(MakeConstSpan(vel), ErrorAssert{});

    _scene->Step(kRatioDt);
    return GetDofs(actor)[kTip * 4 + 1];
  };

  real const damped = runTipDisp(kRatioBeta);
  real const undamped = runTipDisp(0_r);
  EXPECT_NEAR_RTOL(kRatioExpected, damped / undamped, kRatioTol);
}

// Twist stiffness damping: a 3-node rod with all translations fixed and the first element's twist
// anchored, given a twist rate on the second element, twists over one step by ~dt/(dt+beta) of the
// undamped amount. Twists are element quantities grouped with each element's first node. The twist
// DoF is recentered at the start of the next step, so after this single step it still holds the
// element's twist response and can be read directly.
TEST_F(MochiRodDampingTest, TwistStiffnessDampingDecayRatio) {
  _scene->SetGravity({0_r, 0_r, 0_r});
  UseBackwardEulerDirectSolver();

  auto const runTwist = [&](real beta) {
    RodMaterialParams m = BaseMaterial();
    m.torsionalStiffness = kRatioStiffness;
    m.flexuralStiffness = {0_r, 0_r};
    m.stiffnessDampingCoefficient = beta;
    Actor* actor = CreateStraightRod(3, m);
    MOCHI_DEFER(_scene->DestroyActor(actor));

    // Fix all translations (no bending/axial motion) and anchor the first element's twist (node 0).
    // Translation BCs take world positions, so pin each node at its reference position. The second
    // element's twist (node 1) is free and feels the torsional restoring force.
    DynamicArray<int> bcDofs;
    DynamicArray<real> bcVals;
    for (int node = 0; node < 3; ++node) {
      for (int d = 0; d < 3; ++d) {
        bcDofs.push_back(node * 4 + d);
        bcVals.push_back(
            d == 0 ? static_cast<real>(node) : 0_r); // node reference position (x=node)
      }
    }
    bcDofs.push_back(0 * 4 + 3); // first element's twist
    bcVals.push_back(0_r);
    actor->AddBoundaryConditionDofsWorld(
        MakeConstSpan(bcDofs), MakeConstSpan(bcVals), ErrorAssert{});

    DynamicArray<real> vel(3 * 4, 0_r);
    vel[1 * 4 + 3] = kRatioV0; // twist rate on the second element (grouped with node 1)
    actor->SetNodeVelocitiesLocal(MakeConstSpan(vel), ErrorAssert{});

    _scene->Step(kRatioDt);
    // The twist DoF is recentered at the start of the next step, so after this single step it still
    // holds the element's twist response.
    return GetDofs(actor)[1 * 4 + 3];
  };

  real const damped = runTwist(kRatioBeta);
  real const undamped = runTwist(0_r);
  EXPECT_NEAR_RTOL(kRatioExpected, damped / undamped, kRatioTol);
}
