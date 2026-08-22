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

#include "mochi_differentiable_test_utils.h"
#include "mochi_physics_test_fixture.h"

#include <mochi_core/utils/defer.h>
#include <mochi_physics/diffsim/mochi_diffsim.h>
#include <mochi_physics/src/mochi_differentiable.h>

#include <numeric>
#include <string>

using namespace mochi;
using namespace mochi::diffsim;
using namespace mochi::test;

// The scene assets used by these tests are not shipped externally.
#if MOCHI_USE_DOUBLE_PRECISION && MOCHI_INTERNAL
#define MOCHI_USE_DOUBLE_AND_INTERNAL 1
#else
#define MOCHI_USE_DOUBLE_AND_INTERNAL 0
#endif

namespace {
struct ActorData {
  Actor* ptr{};
  ActorType type{};
  int dofsOffset{};
  int dofsSize{};
  int poseOffset{};
  int poseSize{};
  int inputOffset{};
  int inputSize{};
  DynamicArray<int> forceDofs;
};
} // namespace

using LossEvalFn = real (*)(Actor const*);
using GradientEvalFn = void (*)(Actor*);
using LossAndGradientEvalPair = std::pair<LossEvalFn, GradientEvalFn>;

namespace {
real constexpr kDt = 0.01_r;

Real3 constexpr kPosRef{1.4_r, -0.7_r, 0.5_r};

Quaternion const kRotRef = Quaternion::FromRotationVector(Real3{-0.8_r, -0.2_r, 0.3_r});

struct TestParams {
  // Test params
  real tol = 1e-2_r;
  real gradEpsFiniteDiff = kDefaultBackPropagationEpsFiniteDiff;
  real controlSpeed = 0_r;
  real forceSpeed = 1_r;
  // Forward simulation solver params
  VerbosityLevel simVerbosity = NonLinearSolverParams{}.verbosity;
  // Back propagation solver params
  VerbosityLevel diffVerbosity = BackPropagationSolverParams{}.verbosity;
};
} // namespace

static Real3 ActorComDisplacementToRef(Actor const* actor) {
  auto pos = actor->GetCenterOfMassTransform(test::ExpectOK{}).GetTranslation();
  return pos - kPosRef;
}

static real LastStepTranslationErrorLoss(Actor const* target) {
  auto disp = ActorComDisplacementToRef(target);
  return 0.5_r * NormSqr(disp);
}

static void LastStepTranslationErrorGradient(Actor* target) {
  auto disp = ActorComDisplacementToRef(target);
  // Gradient in output space: [disp_x, disp_y, disp_z, 0, 0, 0, 0]
  ColumnVector<real, RigidSize::kAll> gradOutput = ColumnVector<real, RigidSize::kAll>::Zero();
  gradOutput.TopRows<RigidSize::kDTrans>(RigidSize::kDTrans) = AsConstView(disp);
  GetCenterOfMassTransformBackward(target, gradOutput, test::ExpectOK{});
}

static Quaternion ActorRotationToRef(Actor const* actor) {
  auto rot = actor->GetCenterOfMassTransform(test::ExpectOK{}).GetRotation();
  return rot * kRotRef.GetConjugate();
}

static real LastStepRotationErrorLoss(Actor const* target) {
  auto rot = ActorRotationToRef(target);
  return -Trace3x3(ToVMatrix3x3(rot));
}

static void LastStepRotationErrorGradient(Actor* target) {
  auto rot = ActorRotationToRef(target);
  auto rotGrad = -lie::DTrMultRotMatDRot(ToVMatrix3x3(rot));
  // Build 6D Lie gradient: [0, 0, 0, rotGrad]
  ColumnVector<real, RigidSize::kDAll> gradLie = ColumnVector<real, RigidSize::kDAll>::Zero();
  gradLie.BottomRows<RigidSize::kDRot>(RigidSize::kDRot) =
      AsColumnVectorView<RigidSize::kDRot>(rotGrad);
  // Convert Lie gradient to 7D output-space gradient (trans + quaternion)
  ColumnVector<real, RigidSize::kAll> gradOutput;
  auto const transform = target->GetCenterOfMassTransform(test::ExpectOK{});
  ConvertRigidGradientLieToQuaternion(transform, gradLie, gradOutput, test::ExpectOK{});
  GetCenterOfMassTransformBackward(target, gradOutput, test::ExpectOK{});
}

static void
ControlAddEps(ActorData const& actor, int dof, real eps, ColumnVectorView<real> outControl) {
  auto actorControl = outControl.MiddleRows(actor.inputOffset, actor.inputSize);
  // Directly perturb rotation vector representation (not Lie delta)
  actorControl[dof] += eps;
}

static DynamicArray<int> GetForceDofs(Actor const* actor, ActorType type, int /* numDofs */) {
  DynamicArray<int> result;
  if (type == ActorType::Articulated) {
    auto const shapeInfo = actor->GetArticulatedShapeInfo(test::ExpectOK{});
    for (auto const& dofs : shapeInfo.dofInfo) {
      if (dofs.GetSize() == 1) {
        result.push_back(dofs.offset);
      }
    }
  }
  return result;
}

namespace {
LossAndGradientEvalPair lastStepTranslationError =
    std::make_pair(LastStepTranslationErrorLoss, LastStepTranslationErrorGradient);

LossAndGradientEvalPair lastStepRotationError =
    std::make_pair(LastStepRotationErrorLoss, LastStepRotationErrorGradient);

// Test fixture for differentiable tests
struct MochiDifferentiableTestParam {
  int numWorkerThreads = 0;
};

// Various parameter combinations
constexpr MochiDifferentiableTestParam kThreadingParams[] = {
    {.numWorkerThreads = 0}, // Single-threaded
    {.numWorkerThreads = -1} // Multi-threaded with all available cores.
};

// Class to handle the differentiable Mochi scene
class MochiDifferentiable : public MochiSceneTestBase,
                            public ::testing::WithParamInterface<MochiDifferentiableTestParam> {
 public:
  void SetUp() override {
    // Setup number of worker threads.
    _numWorkerThreads = GetParam().numWorkerThreads;

    // Call down
    MochiSceneTestBase::SetUp();

    // Back-propagation requires backward Euler.
    test::SetSceneIntegrationMethod(_scene, IntegrationMethod::BackwardEuler);
  }

 protected:
  // Helper function for debugging
  void SetVerbosity(VerbosityLevel verbosity) {
    auto solver = _scene->GetSolverParams();
    solver.nonLinearSolver.verbosity = verbosity;
    _scene->SetSolverParams(solver, test::ExpectOK{});
  }

  // Run a simulation and collect pre- and post-step states.
  void RunSimulation(
      Span<ActorData const> actors,
      int numSteps,
      MatrixView<real const> control,
      MatrixView<real const> externalForces,
      DynamicArray<StateHandle>* statesPre,
      DynamicArray<StateHandle>* statesPost) {
    // Initialize the control at step 0
    for (auto const& actor : actors) {
      if (actor.inputSize > 0) {
        ColumnVector<real> targetVel = ColumnVector<real>::Zero(actor.inputSize);
        actor.ptr->SetArticulatedTargetVelocity(targetVel, test::ExpectOK{});
      }
    }

    // Run the simulation, applying control and external forces on each step
    for (int i = 0; i < numSteps; ++i) {
      // Apply control and forces for step i
      for (auto const& actor : actors) {
        if (actor.inputSize > 0) {
          actor.ptr->SetArticulatedTargetPose(
              control.Col(i).MiddleRows(actor.inputOffset, actor.inputSize), test::ExpectOK{});
        }
        if (!actor.forceDofs.empty() && isize(externalForces) > 0) {
          auto forceCol = externalForces.Col(i);
          DynamicArray<real> forceValues(isize(actor.forceDofs));
          for (int d = 0; d < isize(actor.forceDofs); ++d) {
            forceValues[d] = forceCol[actor.dofsOffset + actor.forceDofs[d]];
          }
          actor.ptr->SetExternalForcesOnDofs(actor.forceDofs, forceValues, test::ExpectOK{});
        }
      }
      // If required, capture state AFTER applying control, BEFORE Step()
      if (statesPre) {
        statesPre->push_back(_scene->CaptureState(test::ExpectOK{}));
      }
      _scene->Step(kDt);
      // If required, capture state AFTER Step()
      if (statesPost) {
        statesPost->push_back(_scene->CaptureState(test::ExpectOK{}));
      }
    }
  }

  // Run a gradient consistency test
  void RunGradientConsistencyTest(
      Span<LossAndGradientEvalPair const> lossAndGradientEvalPair,
      Actor* target,
      int numSteps,
      TestParams const& params = {}) {
    // Make the scene differentiable (in case it's not already)
    MakeSceneDifferentiable(_scene, test::ExpectOK{});

    // Set the forward simulation solver settings.
    auto simParams = _scene->GetSolverParams();
    // Special settings for differentiability
    simParams.nonLinearSolver.maxIter = 15;
    simParams.nonLinearSolver.absTol = BackPropagationSolverParams{}.outerSolverAbsTol;
    simParams.nonLinearSolver.relTol = BackPropagationSolverParams{}.outerSolverRelTol;
    simParams.nonLinearSolver.convergenceMode =
        BackPropagationSolverParams{}.outerSolverConvergenceMode;
    simParams.linearSolver.absTol = BackPropagationSolverParams{}.innerSolverAbsTol;
    simParams.experimentalEval.fittedSaturationHessian = SaturationHessianParams::All(false);
    _scene->SetSolverParams(simParams, test::ExpectOK{});
    // Set the back-propagation solver settings.
    auto diffParams = GetBackPropagationSolverParams(_scene, test::ExpectOK{});
    diffParams.verbosity = params.diffVerbosity;
    diffParams.validateFiniteDiff = true; // Validate finite-difference robustness
    SetBackPropagationSolverParams(_scene, diffParams, test::ExpectOK{});

    // Initialize the actor data
    DynamicArray<ActorData> actors;
    actors.reserve(_scene->GetNumActors());
    int totalDofs = 0;
    int totalPose = 0;
    int totalInput = 0;
    _scene->ForEachActor([&](Actor* actor) {
      if (!actor->IsStatic() && !actor->IsNestedLinkActor()) {
        int actorDofs = actor->GetNumDofs();
        int actorPose = actor->GetType() == ActorType::Rigid ? RigidSize::kAll : actorDofs;
        bool const hasController = actor->GetType() == ActorType::Articulated &&
            actor->HasArticulatedPoseController(test::ExpectOK{});
        int actorInput = hasController ? actorDofs : 0;
        auto forceDofs = GetForceDofs(actor, actor->GetType(), actorDofs);
        actors.push_back(
            ActorData{
                actor,
                actor->GetType(),
                totalDofs,
                actorDofs,
                totalPose,
                actorPose,
                totalInput,
                actorInput,
                std::move(forceDofs)});
        totalDofs += actorDofs;
        totalPose += actorPose;
        totalInput += actorInput;
      }
    });

    // Define the control trajectory. Take the initial pose and then perturb.
    Matrix<real> control(totalInput, numSteps);
    for (auto const& actor : actors) {
      if (actor.inputSize > 0) {
        actor.ptr->GetArticulatedPose(
            control.Col(0).MiddleRows(actor.inputOffset, actor.inputSize), test::ExpectOK{});
      }
    }
    for (int j = 0; j < numSteps; ++j) {
      auto amplitude = params.controlSpeed * kDt * static_cast<real>(j);
      control.Col(j) = control.Col(0) + amplitude * ColumnVector<real>::Ones(totalInput);
    }

    // Define the external force trajectory (zero-initialized, then perturb valid force DOFs).
    Matrix<real> externalForces;
    externalForces = Matrix<real>::Zero(totalDofs, numSteps);
    for (int j = 0; j < numSteps; ++j) {
      auto amplitude = params.forceSpeed * kDt * static_cast<real>(j);
      for (auto const& actor : actors) {
        for (int const d : actor.forceDofs) {
          externalForces(actor.dofsOffset + d, j) = amplitude;
        }
      }
    }

    // Capture initial state before any control is applied.
    StateHandle stateInit = _scene->CaptureState(test::ExpectOK{});

    // Run the simulation and collect the states (pre- and post-step)
    DynamicArray<StateHandle> statesPre;
    statesPre.reserve(numSteps);
    DynamicArray<StateHandle> statesPost;
    statesPost.reserve(numSteps);
    auto oldVerbosity = _scene->GetSolverParams().nonLinearSolver.verbosity;
    SetVerbosity(params.simVerbosity);
    RunSimulation(actors, numSteps, control, externalForces, &statesPre, &statesPost);
    SetVerbosity(oldVerbosity);

    // Prepare for back-propagation.
    ResetBackPropagation(_scene, test::ExpectOK{});

    // Prepare and evaluate the loss gradient at the last step via output backward functions.
    PrepareBackPropagate(
        _scene, statesPost[numSteps - 1], statesPre[numSteps - 1], test::ExpectOK{});
    for (auto const& lossTerm : lossAndGradientEvalPair) {
      lossTerm.second(target);
    }

    // Run back-propagation.
    Matrix<real> gradControl = Matrix<real>::Zero(totalInput, numSteps);
    Matrix<real> gradForce = Matrix<real>::Zero(totalDofs, numSteps);
    for (int i = numSteps; i > 0; --i) {
      if (i != numSteps) {
        PrepareBackPropagate(_scene, statesPost[i - 1], statesPre[i - 1], test::ExpectOK{});
      }
      BackPropagate(_scene, test::ExpectOK{});

      // Extract control gradient via SetArticulatedTargetPoseBackward.
      for (auto const& actor : actors) {
        if (actor.inputSize > 0) {
          auto controlGrad = gradControl.Col(i - 1).MiddleRows(actor.inputOffset, actor.inputSize);
          SetArticulatedTargetPoseBackward(actor.ptr, controlGrad, test::ExpectOK{});
        }
      }

      // Extract force gradient via SetExternalForcesOnDofsBackward.
      for (auto const& actor : actors) {
        if (!actor.forceDofs.empty()) {
          DynamicArray<real> forceGradValues(isize(actor.forceDofs));
          SetExternalForcesOnDofsBackward(
              actor.ptr, actor.forceDofs, forceGradValues, test::ExpectOK{});
          for (int d = 0; d < isize(actor.forceDofs); ++d) {
            gradForce(actor.dofsOffset + actor.forceDofs[d], i - 1) = forceGradValues[d];
          }
        }
      }
    }

    // Extract gradients wrt initial state and velocity via backward functions.
    ColumnVector<real> gradInitState(totalPose);
    ColumnVector<real> gradInitVel(totalDofs);
    for (auto const& actor : actors) {
      auto stateGrad = gradInitState.MiddleRows(actor.poseOffset, actor.poseSize);
      if (actor.type == ActorType::Articulated) {
        auto velGrad = gradInitVel.MiddleRows(actor.dofsOffset, actor.dofsSize);
        SetArticulatedPoseFromJointsBackward(actor.ptr, stateGrad, test::ExpectOK{});
        SetArticulatedJointVelocitiesBackward(actor.ptr, velGrad, test::ExpectOK{});
      } else {
        auto gradLin = gradInitVel.MiddleRows(actor.dofsOffset, RigidSize::kDTrans);
        auto gradAng =
            gradInitVel.MiddleRows(actor.dofsOffset + RigidSize::kDTrans, RigidSize::kDRot);
        SetCenterOfMassTransformBackward(actor.ptr, stateGrad, test::ExpectOK{});
        SetVelocityBackward(actor.ptr, gradLin, gradAng, test::ExpectOK{});
      }
    }

    auto evalObjective = [&]() -> real {
      real objective = 0_r;
      for (auto const& lossTerm : lossAndGradientEvalPair) {
        objective += lossTerm.first(target);
      }
      return objective;
    };

    auto compare = [&](ColumnVectorView<real const> a, ColumnVectorView<real const> b) {
      if (a.empty()) {
        return;
      }
      auto normA = a.Norm();
      auto normB = b.Norm();
      ColumnVector<real> diff = a - b;
      auto normDiff = diff.Norm();
      EXPECT_NEAR(normDiff / Max(normA, normB), 0_r, params.tol);
    };

    // Obtain gradient wrt initial state by finite differences
    [[maybe_unused]] real objective = evalObjective();
    ColumnVector<real> testGradientInitialState(totalPose);
    for (auto const& actor : actors) {
      auto actorAddEpsFn = GetActorAddEpsFn(actor.type);
      for (int j = 0; j < actor.poseSize; ++j) {
        _scene->RestoreState(stateInit, false, test::ExpectOK{});
        actorAddEpsFn(actor.ptr, j, params.gradEpsFiniteDiff);
        RunSimulation(actors, numSteps, control, externalForces, nullptr, nullptr);
        real objectiveP = evalObjective();
        _scene->RestoreState(stateInit, false, test::ExpectOK{});
        actorAddEpsFn(actor.ptr, j, -params.gradEpsFiniteDiff);
        RunSimulation(actors, numSteps, control, externalForces, nullptr, nullptr);
        real objectiveM = evalObjective();
        testGradientInitialState[actor.poseOffset + j] =
            (objectiveP - objectiveM) / (2_r * params.gradEpsFiniteDiff);
      }
    }
    compare(gradInitState, testGradientInitialState);

    // Obtain gradient wrt initial velocity by finite differences
    ColumnVector<real> testGradientInitialVelocity(totalDofs);
    for (auto const& actor : actors) {
      auto actorAddVelEpsFn = GetActorAddVelEpsFn(actor.type);
      for (int j = 0; j < actor.dofsSize; ++j) {
        _scene->RestoreState(stateInit, false, test::ExpectOK{});
        actorAddVelEpsFn(actor.ptr, j, params.gradEpsFiniteDiff);
        RunSimulation(actors, numSteps, control, externalForces, nullptr, nullptr);
        real objectiveP = evalObjective();
        _scene->RestoreState(stateInit, false, test::ExpectOK{});
        actorAddVelEpsFn(actor.ptr, j, -params.gradEpsFiniteDiff);
        RunSimulation(actors, numSteps, control, externalForces, nullptr, nullptr);
        real objectiveM = evalObjective();
        testGradientInitialVelocity[actor.dofsOffset + j] =
            (objectiveP - objectiveM) / (2_r * params.gradEpsFiniteDiff);
      }
    }
    compare(gradInitVel, testGradientInitialVelocity);

    // Obtain gradient wrt control by finite differences
    int const numStepsGradControl = MOCHI_DEBUG ? 1 : numSteps;
    ColumnVector<real> testGradientControl(totalInput);
    for (int i = 0; i < numStepsGradControl; ++i) {
      testGradientControl.SetZero();
      auto controlRef = control.Col(i).Duplicate();
      for (auto const& actor : actors) {
        if (actor.inputSize <= 0) {
          continue;
        }
        for (int j = 0; j < actor.inputSize; ++j) {
          _scene->RestoreState(stateInit, false, test::ExpectOK{});
          ControlAddEps(actor, j, params.gradEpsFiniteDiff, control.Col(i));
          RunSimulation(actors, numSteps, control, externalForces, nullptr, nullptr);
          real objectiveP = evalObjective();
          _scene->RestoreState(stateInit, false, test::ExpectOK{});
          control.Col(i) = controlRef;
          ControlAddEps(actor, j, -params.gradEpsFiniteDiff, control.Col(i));
          RunSimulation(actors, numSteps, control, externalForces, nullptr, nullptr);
          real objectiveM = evalObjective();
          testGradientControl(actor.inputOffset + j) =
              (objectiveP - objectiveM) / (2_r * params.gradEpsFiniteDiff);
          control.Col(i) = controlRef;
        }
      }
      compare(gradControl.Col(i), testGradientControl);
    }

    // Obtain gradient wrt external force by finite differences (only for valid force DOFs)
    int const numStepsGradForce = MOCHI_DEBUG ? 1 : numSteps;
    for (int i = 0; i < numStepsGradForce; ++i) {
      // Collect all valid force DOF global indices and their finite-difference gradients
      DynamicArray<int> allForceDofIndices;
      for (auto const& actor : actors) {
        for (int const d : actor.forceDofs) {
          allForceDofIndices.push_back(actor.dofsOffset + d);
        }
      }
      ColumnVector<real> testGradientForceFiltered(isize(allForceDofIndices));
      auto forceRef = externalForces.Col(i).Duplicate();
      for (int fi = 0; fi < isize(allForceDofIndices); ++fi) {
        int const globalDof = allForceDofIndices[fi];
        _scene->RestoreState(stateInit, false, test::ExpectOK{});
        externalForces.Col(i)[globalDof] += params.gradEpsFiniteDiff;
        RunSimulation(actors, numSteps, control, externalForces, nullptr, nullptr);
        real objectiveP = evalObjective();
        _scene->RestoreState(stateInit, false, test::ExpectOK{});
        externalForces.Col(i) = forceRef;
        externalForces.Col(i)[globalDof] -= params.gradEpsFiniteDiff;
        RunSimulation(actors, numSteps, control, externalForces, nullptr, nullptr);
        real objectiveM = evalObjective();
        testGradientForceFiltered[fi] =
            (objectiveP - objectiveM) / (2_r * params.gradEpsFiniteDiff);
        externalForces.Col(i) = forceRef;
      }
      // Extract corresponding entries from analytical gradient
      ColumnVector<real> gradForceFiltered(isize(allForceDofIndices));
      for (int fi = 0; fi < isize(allForceDofIndices); ++fi) {
        gradForceFiltered[fi] = gradForce.Col(i)[allForceDofIndices[fi]];
      }
      compare(gradForceFiltered, testGradientForceFiltered);
    }

    // Release states
    _scene->ReleaseAllStates();
  }
};
} // namespace

// The scene assets used by the tests below are not shipped externally.
TEST_IF_P(MOCHI_INTERNAL, MochiDifferentiable, SingleRigidActor_LastStepTranslationError) {
  auto* target = LoadScenePrefab(_scene, "rigid_cube_free.mochi_scene", "Cube");
  RunGradientConsistencyTest(MakeSingletonConstSpan(lastStepTranslationError), target, 10);
}

TEST_IF_P(MOCHI_INTERNAL, MochiDifferentiable, SingleRigidActor_LastStepRotationError) {
  auto* target = LoadScenePrefab(_scene, "rigid_cube_free.mochi_scene", "Cube");
  RunGradientConsistencyTest(MakeSingletonConstSpan(lastStepRotationError), target, 10);
}

// The cube is penetrating the incline at initialization, with no friction.
TEST_IF_P(
    MOCHI_USE_DOUBLE_AND_INTERNAL,
    MochiDifferentiable,
    SingleRigidActorContactFrictionless_LastStepError) {
  auto* target = LoadScenePrefab(_scene, "rigid_cube_on_plane_frictionless.mochi_scene", "Cube");
  std::array<LossAndGradientEvalPair, 2> lossTerms = {
      lastStepTranslationError, lastStepRotationError};
  RunGradientConsistencyTest(lossTerms, target, 10, TestParams{.gradEpsFiniteDiff = 1e-5_r});
}

// The cube is penetrating the incline at initialization, with viscous friction.
TEST_IF_P(
    MOCHI_USE_DOUBLE_AND_INTERNAL,
    MochiDifferentiable,
    SingleRigidActorContactViscousFriction_LastStepError) {
  auto* target =
      LoadScenePrefab(_scene, "rigid_cube_on_plane_viscous_friction.mochi_scene", "Cube");
  std::array<LossAndGradientEvalPair, 2> lossTerms = {
      lastStepTranslationError, lastStepRotationError};
  TestParams params{.gradEpsFiniteDiff = 1e-4_r};
  RunGradientConsistencyTest(lossTerms, target, 10, params);
}

// The cube is penetrating the incline at initialization, with Coulomb friction.
TEST_IF_P(
    MOCHI_USE_DOUBLE_AND_INTERNAL,
    MochiDifferentiable,
    SingleRigidActorContactCoulombFriction_LastStepError) {
  auto* target =
      LoadScenePrefab(_scene, "rigid_cube_on_plane_coulomb_friction.mochi_scene", "Cube");
  std::array<LossAndGradientEvalPair, 2> lossTerms = {
      lastStepTranslationError, lastStepRotationError};
  TestParams params{.tol = 2e-2_r, .gradEpsFiniteDiff = 1e-4_r, .forceSpeed = 0.1_r};
  RunGradientConsistencyTest(lossTerms, target, 10, params);
}

// The cube falls on the incline on step 10, then it slides for 10 more steps.
TEST_IF_P(
    MOCHI_USE_DOUBLE_AND_INTERNAL,
    MochiDifferentiable,
    SingleRigidActorContactAfterNSteps_LastStepError) {
  auto* target = LoadScenePrefab(_scene, "rigid_cube_falling_on_plane.mochi_scene", "Cube");
  target->RegisterQuery(QueryType::TotalContactForce, test::ExpectOK{});
  std::array<LossAndGradientEvalPair, 2> lossTerms = {
      lastStepTranslationError, lastStepRotationError};
  TestParams params{.tol = 2e-2_r, .gradEpsFiniteDiff = 1e-8_r, .forceSpeed = 0.1_r};
  RunGradientConsistencyTest(lossTerms, target, 20, params);
  // Confirm that there are contact forces at the end of the simulation
  Real3 force = target->GetContactForceWorld(test::ExpectOK{});
  EXPECT_NE(force, Real3{});
}

// One cube is penetrating the incline at initialization, the other one is on top. There is no
// friction.
TEST_IF_P(
    MOCHI_USE_DOUBLE_AND_INTERNAL,
    MochiDifferentiable,
    TwoRigidActorsContactFrictionless_LastStepError) {
  auto* target =
      LoadScenePrefab(_scene, "two_rigid_cubes_on_plane_frictionless.mochi_scene", "BottomCube");
  std::array<LossAndGradientEvalPair, 2> lossTerms = {
      lastStepTranslationError, lastStepRotationError};
  RunGradientConsistencyTest(lossTerms, target, 10, TestParams{.tol = 3e-2_r});
}

// One cube is penetrating the incline at initialization, the other one is on top. Coulomb friction.
TEST_IF_P(
    MOCHI_USE_DOUBLE_AND_INTERNAL,
    MochiDifferentiable,
    TwoRigidActorsContactCoulombFriction_LastStepError) {
  auto* target = LoadScenePrefab(
      _scene, "two_rigid_cubes_on_plane_coulomb_friction.mochi_scene", "BottomCube");
  std::array<LossAndGradientEvalPair, 2> lossTerms = {
      lastStepTranslationError, lastStepRotationError};
  TestParams params{.tol = 3e-2_r};
  RunGradientConsistencyTest(lossTerms, target, 10, params);
}

// These tests are too slow for CI in non-optimized builds.
// Modify the following line if you ever need to debug them locally.
// The articulated scene assets used below are not shipped externally.
#if MOCHI_USE_DOUBLE_PRECISION && MOCHI_USE_HDF5 && MOCHI_OPTIMIZED && MOCHI_INTERNAL
#define MOCHI_TEST_ARTICULATED_DOUBLE 1
#else
#define MOCHI_TEST_ARTICULATED_DOUBLE 0
#endif

// An articulated body in free motion.
TEST_IF_P(
    MOCHI_TEST_ARTICULATED_DOUBLE,
    MochiDifferentiable,
    SingleArticulatedActor_LastStepError) {
  auto* target = LoadScenePrefab(
      _scene,
      "articulated_actor_free_motion.mochi_scene",
      "ArticulatedActor/ArticulatedActor/Link2_Horizontal");
  std::array<LossAndGradientEvalPair, 2> lossTerms = {
      lastStepTranslationError, lastStepRotationError};
  RunGradientConsistencyTest(lossTerms, target, 10, TestParams{.tol = 2e-2_r});
}

// An articulated body in contact with the ground.
TEST_IF_P(
    MOCHI_TEST_ARTICULATED_DOUBLE,
    MochiDifferentiable,
    SingleArticulatedActorContact_LastStepError) {
  auto* target = LoadScenePrefab(
      _scene, "articulated_actor_on_plane.mochi_scene", "ArticulatedActor/Link2_Horizontal");
  TestParams params{.tol = 2e-2_r, .gradEpsFiniteDiff = 1e-6_r, .forceSpeed = 0.1_r};
  std::array<LossAndGradientEvalPair, 2> lossTerms = {
      lastStepTranslationError, lastStepRotationError};
  RunGradientConsistencyTest(lossTerms, target, 10, params);
}

// Contact between a rigid body and an articulated body.
TEST_IF_P(
    MOCHI_TEST_ARTICULATED_DOUBLE,
    MochiDifferentiable,
    RigidAndArticulatedActorContact_LastStepError) {
  auto* target = LoadScenePrefab(
      _scene,
      "articulated_and_rigid_actor_contact.mochi_scene",
      "ArticulatedActor/Link2_Horizontal");

  TestParams params{.tol = 2e-2_r, .gradEpsFiniteDiff = 1e-8_r, .forceSpeed = 0.1_r};
  std::array<LossAndGradientEvalPair, 2> lossTerms = {
      lastStepTranslationError, lastStepRotationError};
  RunGradientConsistencyTest(lossTerms, target, 10, params);
}

// An articulated body with joint friction.
TEST_IF_P(
    MOCHI_TEST_ARTICULATED_DOUBLE,
    MochiDifferentiable,
    SingleArticulatedActorJointFriction_LastStepError) {
  auto* target = LoadScenePrefab(
      _scene, "articulated_actor_joint_friction.mochi_scene", "ArticulatedActor/Link2_Horizontal");
  std::array<LossAndGradientEvalPair, 2> lossTerms = {
      lastStepTranslationError, lastStepRotationError};
  RunGradientConsistencyTest(lossTerms, target, 10, TestParams{.tol = 2e-2_r});
}

// An articulated body with joint limits.
TEST_IF_P(
    MOCHI_TEST_ARTICULATED_DOUBLE,
    MochiDifferentiable,
    ArticulatedActorJointLimits_LastStepError) {
  auto* target = LoadScenePrefab(
      _scene, "articulated_actor_joint_limits.mochi_scene", "ArticulatedActor/Link2_Horizontal");
  _scene->ForEachConstraint([](Constraint* constraint) {
    constraint->RegisterQuery(QueryType::ConstraintForce, test::ExpectOK{});
  });
  std::array<LossAndGradientEvalPair, 2> lossTerms = {
      lastStepTranslationError, lastStepRotationError};
  RunGradientConsistencyTest(lossTerms, target, 10);
  // Confirm that some constraint (joint limit) is active
  real forceTotal{};
  _scene->ForEachConstraint([&forceTotal](Constraint* constraint) {
    auto force = constraint->GetForce(test::ExpectOK{});
    forceTotal += AsConstView(force).Norm();
  });
  EXPECT_NE(forceTotal, 0_r);
}

// An articulated body with pose controller and fixed control.
TEST_IF_P(
    MOCHI_TEST_ARTICULATED_DOUBLE,
    MochiDifferentiable,
    ArticulatedActorPoseController_LastStepError) {
  auto* target = LoadScenePrefab(
      _scene,
      "articulated_actor_controller.mochi_scene",
      "ArticulatedActor/ArticulatedActor/Link2_Horizontal");
  std::array<LossAndGradientEvalPair, 2> lossTerms = {
      lastStepTranslationError, lastStepRotationError};
  RunGradientConsistencyTest(lossTerms, target, 10);
}

// An articulated body with pose controller and a control trajectory.
TEST_IF_P(
    MOCHI_TEST_ARTICULATED_DOUBLE,
    MochiDifferentiable,
    ArticulatedActorPoseControllerTrajectory_LastStepError) {
  auto* target = LoadScenePrefab(
      _scene,
      "articulated_actor_controller.mochi_scene",
      "ArticulatedActor/ArticulatedActor/Link2_Horizontal");
  TestParams params{.tol = 2e-2_r, .gradEpsFiniteDiff = 1e-6_r, .controlSpeed = 10_r};
  std::array<LossAndGradientEvalPair, 2> lossTerms = {
      lastStepTranslationError, lastStepRotationError};
  RunGradientConsistencyTest(lossTerms, target, 10, params);
}

// Mixed scene with 2 rigid actors and 3 articulated actors (2 with pose controllers), all in
// contact.
TEST_IF_P(
    MOCHI_TEST_ARTICULATED_DOUBLE,
    MochiDifferentiable,
    MixedRigidAndArticulatedContact_LastStepError) {
#if MOCHI_DEBUG
  if (GetParam().numWorkerThreads == 0) {
    // Skip single-threaded variant in debug builds due to timeout.
    return;
  }
#endif
  auto* target = LoadScenePrefab(
      _scene,
      "mixed_rigid_and_articulated_contact.mochi_scene",
      "ArticulatedActor1/ArticulatedActor/Link2_Horizontal");
  TestParams params{.tol = 3e-2_r, .controlSpeed = 5_r};
  std::array<LossAndGradientEvalPair, 2> lossTerms = {
      lastStepTranslationError, lastStepRotationError};
  RunGradientConsistencyTest(lossTerms, target, 10, params);
}

/***************************************************************************************************
  BackPropagationSceneStats Tests
*/

// Test that GetBackPropagationSceneStats returns valid initial values before any back-propagation
TEST_P(MochiDifferentiable, BackPropagationSceneStats_InitialStatsAreZero) {
  // Make the scene differentiable
  MakeSceneDifferentiable(_scene, test::ExpectOK{});

  auto stats = GetBackPropagationSceneStats(_scene, test::ExpectOK{});

  EXPECT_EQ(stats.totalDurationSec, 0.0);
  EXPECT_EQ(stats.solveDurationSec, 0.0);
  EXPECT_EQ(stats.maxOuterIters, 0);
  EXPECT_EQ(stats.residualNorm, 0.0);
}

// Test that GetBackPropagationSceneStats returns correct values after back-propagation
// The scene asset used by this test is not shipped externally.
TEST_IF_P(MOCHI_INTERNAL, MochiDifferentiable, BackPropagationSceneStats_StatsAfterBackPropagate) {
  // Load a simple rigid actor scene
  auto* target = LoadScenePrefab(_scene, "rigid_cube_free.mochi_scene", "Cube");
  ASSERT_NE(target, nullptr);

  // Make the scene differentiable
  MakeSceneDifferentiable(_scene, test::ExpectOK{});

  // Disable gravity for deterministic behavior
  _scene->SetGravity({});

  // Run forward simulation
  auto state0 = _scene->CaptureState(test::ExpectOK{});
  _scene->Step(kDt);
  auto state1 = _scene->CaptureState(test::ExpectOK{});

  // Run back-propagation with a random output gradient
  ResetBackPropagation(_scene, test::ExpectOK{});
  ColumnVector<real, RigidSize::kAll> gradOutput;
  gradOutput.SetRandom(1, -1_r, 1_r);
  GetCenterOfMassTransformBackward(target, gradOutput, test::ExpectOK{});
  PrepareBackPropagate(_scene, state1, state0, test::ExpectOK{});
  BackPropagate(_scene, test::ExpectOK{});

  // Get the stats
  auto stats = GetBackPropagationSceneStats(_scene, test::ExpectOK{});

  // Verify stats are populated and valid
  EXPECT_GE(stats.totalDurationSec, 0.0);
  EXPECT_GE(stats.solveDurationSec, 0.0);
  EXPECT_GE(stats.maxOuterIters, 1);
  EXPECT_GE(stats.residualNorm, 0.0);
  EXPECT_TRUE(std::isfinite(stats.residualNorm));

  // Total duration should be at least as long as solve duration
  EXPECT_GE(stats.totalDurationSec, stats.solveDurationSec);

  // Clean up states
  _scene->ReleaseState(state0);
  _scene->ReleaseState(state1);
}

// Test that GetBackPropagationSceneStats fails for non-differentiable scenes
TEST_P(MochiDifferentiable, BackPropagationSceneStats_FailsForNonDifferentiableScene) {
  // Do NOT make the scene differentiable
  Error error;
  [[maybe_unused]] auto stats = GetBackPropagationSceneStats(_scene, error);

  // Should return an error for non-differentiable scene
  EXPECT_FALSE(error.IsOK());
}

/***************************************************************************************************
  Mid-back-propagation Capture / Restore Test
*/

// Verify that bisecting a backward pass with @ref Scene::CaptureState / @ref Scene::RestoreState
// yields gradients that are bit-identical to a single-shot reference back-propagation. This
// confirms that:
//   (a) the capture buffer carries the persistent adjoint state alongside forward state on
//       differentiable scenes, and
//   (b) @ref BackPropagate's per-step internal restores do not clobber that live adjoint state.
TEST_IF_P(MOCHI_TEST_ARTICULATED_DOUBLE, MochiDifferentiable, CaptureRestoreAdjoints) {
  Actor* target = LoadScenePrefab(
      _scene,
      "articulated_actor_controller.mochi_scene",
      "ArticulatedActor/ArticulatedActor/Link2_Horizontal");
  MakeSceneDifferentiable(_scene, test::ExpectOK{});

  Actor* articulated = _scene->GetActor(target->GetArticulatedActor(test::ExpectOK{}));
  ASSERT_NE(articulated, nullptr);
  EXPECT_TRUE(articulated->HasArticulatedPoseController(test::ExpectOK{}));
  int const numDofs = articulated->GetNumDofs();

  constexpr int kNumSteps = 6;
  constexpr int kCheckpoint = 3;

  // Initialize controller (zero target velocity).
  ColumnVector<real> initialTargetVel = ColumnVector<real>::Zero(numDofs);
  articulated->SetArticulatedTargetVelocity(initialTargetVel, test::ExpectOK{});

  // Build a control trajectory: starting from the initial pose, drift the target by
  // `controlSpeed * dt * i` on every step.
  DynamicArray<real> initPose(numDofs);
  articulated->GetArticulatedPose(initPose, test::ExpectOK{});
  Matrix<real> control(numDofs, kNumSteps);
  real constexpr kControlSpeed = 5_r;
  for (int j = 0; j < kNumSteps; ++j) {
    real const amplitude = kControlSpeed * kDt * static_cast<real>(j);
    for (int d = 0; d < numDofs; ++d) {
      control(d, j) = initPose[d] + amplitude;
    }
  }

  // Run forward simulation: apply control each step, capture pre/post states.
  DynamicArray<StateHandle> statesPre;
  DynamicArray<StateHandle> statesPost;
  statesPre.reserve(kNumSteps);
  statesPost.reserve(kNumSteps);
  for (int i = 0; i < kNumSteps; ++i) {
    articulated->SetArticulatedTargetPose(control.Col(i), test::ExpectOK{});
    statesPre.push_back(_scene->CaptureState(test::ExpectOK{}));
    _scene->Step(kDt);
    statesPost.push_back(_scene->CaptureState(test::ExpectOK{}));
  }
  MOCHI_DEFER({ _scene->ReleaseAllStates(); });

  // Initial-state gradient extractors.
  auto extractInitialGradients = [&]() {
    ColumnVector<real> gradPose(numDofs);
    ColumnVector<real> gradVel(numDofs);
    SetArticulatedPoseFromJointsBackward(articulated, gradPose, test::ExpectOK{});
    SetArticulatedJointVelocitiesBackward(articulated, gradVel, test::ExpectOK{});
    return std::make_tuple(gradPose, gradVel);
  };

  // Loss: combined translation + rotation error of the link target at the last step.
  auto applyLossGradient = [&]() {
    LastStepTranslationErrorGradient(target);
    LastStepRotationErrorGradient(target);
  };

  // Backward step plus per-step control-gradient extraction.
  auto backPropStep = [&](int i, MatrixView<real> outControlGrads) {
    PrepareBackPropagate(_scene, statesPost[i - 1], statesPre[i - 1], test::ExpectOK{});
    BackPropagate(_scene, test::ExpectOK{});
    SetArticulatedTargetPoseBackward(articulated, outControlGrads.Col(i - 1), test::ExpectOK{});
  };

  // -- Reference run: full backward pass, with a state checkpoint captured mid-way. --
  ResetBackPropagation(_scene, test::ExpectOK{});
  applyLossGradient();
  Matrix<real> refControlGrads = Matrix<real>::Zero(numDofs, kNumSteps);
  // Walk back from the last step down to the checkpoint.
  for (int i = kNumSteps; i > kCheckpoint; --i) {
    backPropStep(i, refControlGrads);
  }
  // Capture the full state (forward + adjoints) at the checkpoint.
  StateHandle checkpoint = _scene->CaptureState(test::ExpectOK{});
  // Continue back-propagation through to step 0 to obtain reference gradients.
  for (int i = kCheckpoint; i > 0; --i) {
    backPropStep(i, refControlGrads);
  }
  // Snapshot the post-backward state (forward + accumulated adjoints) for comparison.
  StateHandle refFinalState = _scene->CaptureState(test::ExpectOK{});
  auto [refPose, refVel] = extractInitialGradients();

  // -- Replay run: restore the checkpoint and re-run only the suffix back-propagation. --
  _scene->RestoreState(checkpoint, /*releaseImmediately=*/false, test::ExpectOK{});
  Matrix<real> replayControlGrads = Matrix<real>::Zero(numDofs, kNumSteps);
  for (int i = kCheckpoint; i > 0; --i) {
    backPropStep(i, replayControlGrads);
  }
  // Snapshot the post-backward state for comparison with the reference.
  StateHandle replayFinalState = _scene->CaptureState(test::ExpectOK{});
  auto [replayPose, replayVel] = extractInitialGradients();

  // Replay must reproduce the reference gradients exactly.
  EXPECT_SPAN_EQ(refPose, replayPose);
  EXPECT_SPAN_EQ(refVel, replayVel);
  // Per-step control gradients: the replay only re-ran steps [0, kCheckpoint), so compare those.
  for (int i = 0; i < kCheckpoint; ++i) {
    EXPECT_SPAN_EQ(refControlGrads.Col(i), replayControlGrads.Col(i));
  }
  // The full post-backward snapshots (forward state + all live adjoints) must also match.
  EXPECT_TRUE(_scene->IsEqualState(refFinalState, replayFinalState));
}

INSTANTIATE_TEST_SUITE_P(
    ThreadingVariations,
    MochiDifferentiable,
    testing::ValuesIn(kThreadingParams));

#undef MOCHI_TEST_ARTICULATED_DOUBLE
