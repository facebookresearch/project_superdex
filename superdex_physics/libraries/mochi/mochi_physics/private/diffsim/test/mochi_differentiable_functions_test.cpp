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

#include <mochi_physics/diffsim/mochi_diffsim.h>
#include <mochi_physics/src/mochi_differentiable.h>
#include <mochi_physics/utils/mochi_prefab.h>

using namespace mochi;
using namespace mochi::diffsim;
using namespace mochi::test;

namespace {

real constexpr kDt = 0.01_r;
int constexpr kNumSteps = 3;
int constexpr kNumRuns = 2;

// These tests are too slow for CI in non-optimized builds.
// Modify the following line if you ever need to debug them locally.
// The model assets used by these tests are not shipped externally.
#if MOCHI_USE_DOUBLE_PRECISION && MOCHI_USE_HDF5 && MOCHI_OPTIMIZED && MOCHI_INTERNAL
#define MOCHI_TEST_DIFFERENTIABLE_FUNCTIONS 1
#else
#define MOCHI_TEST_DIFFERENTIABLE_FUNCTIONS 0
#endif

struct TestParams {
  real tol = 1e-2_r;
  real epsFiniteDiff = 1e-7_r;
};

/// @brief Base fixture for gradient consistency tests of differentiable I/O backward functions.
///
/// Loads the shared mixed scene (rigid + articulated + contact), makes it differentiable,
/// and provides RunFunctionTest() for finite-difference comparison.
/// Always runs kNumSteps forward steps to exercise velocity-dependent gradient paths.
class DifferentiableFunctionsTest : public MochiSceneTestBase {
 protected:
  int _totalDofs = 0;
  int _totalInput = 0;
  int _defaultOutputSize = 0;
  Matrix<real, krylov::kDynamic, kNumSteps> _defaultControl;

  void SetUp() override {
    MochiSceneTestBase::SetUp();

    // Back-propagation requires backward Euler.
    test::SetSceneIntegrationMethod(_scene, IntegrationMethod::BackwardEuler);

    // Load test scene and make it differentiable
    auto const* const kAssetPath =
        "differentiability_test/mixed_rigid_and_articulated_contact_high_damping_control.mochi_scene";
    prefab::PrefabParams params{.applySceneSettings = false};
    prefab::AddToScene(GetAssetPath(kAssetPath), GetAssetPath(""), _scene, params, ExpectOK{});
    MakeSceneDifferentiable(_scene, ExpectOK{});

    // Configure solver for differentiability
    auto simParams = _scene->GetSolverParams();
    simParams.experimentalEval.fittedSaturationHessian = {
        .contactFriction = false, .jointFriction = false, .constraintSaturation = false};
    simParams.nonLinearSolver.maxIter = 15;
    // TODO(T273020090): Enable NonLinearSolverConvergenceMode::PerActorWeighted.
    simParams.nonLinearSolver.convergenceMode = NonLinearSolverConvergenceMode::Global;
    simParams.nonLinearSolver.absTol = 1e-3_r;
    simParams.nonLinearSolver.relTol = 1e-10_r;
    _scene->SetSolverParams(simParams, ExpectOK{});

    CollectActorData();
    BuildDefaultControl();
  }

  /// Called at each forward step to apply input to the scene.
  using InputFn = std::function<void(int step)>;
  /// Called after each BackPropagate to accumulate gradients for the given step.
  using InputBackwardFn = std::function<void(int step, ColumnVectorView<real> outGrad)>;
  /// Reads the output function values into a pre-allocated buffer (e.g., calls GetFoo for each
  /// actor).
  using OutputFn = std::function<void(Span<real> output)>;
  /// Called at each step (reverse) to invoke the output dual with the given gradOutput.
  using OutputBackwardFn = std::function<void(Span<real const> gradOutput)>;

  /// Default input: SetArticulatedTargetPose using _defaultControl.
  void DefaultInputFn(int step) {
    int inputOffset = 0;
    _scene->ForEachActor([&](Actor* actor) {
      if (actor->GetType() == ActorType::Articulated &&
          actor->HasArticulatedPoseController(ExpectOK{})) {
        int numDofs = actor->GetNumDofs();
        actor->SetArticulatedTargetPose(
            _defaultControl.Col(step).MiddleRows(inputOffset, numDofs), ExpectOK{});
        inputOffset += numDofs;
      }
    });
  }

  /// Default input backward: SetArticulatedTargetPoseBackward.
  void DefaultInputBackwardFn(int step, ColumnVectorView<real> outGrad) {
    int offset = step * _totalInput;
    int inputOffset = 0;
    _scene->ForEachActor([&](Actor const* actor) {
      if (actor->GetType() == ActorType::Articulated &&
          actor->HasArticulatedPoseController(ExpectOK{})) {
        int numDofs = actor->GetNumDofs();
        DynamicArray<real> grad(numDofs, 0_r);
        SetArticulatedTargetPoseBackward(actor, grad, ExpectOK{});
        for (int j = 0; j < numDofs; ++j, ++offset) {
          outGrad[offset] += grad[j];
        }
        inputOffset += numDofs;
      }
    });
  }

  /// Default output: GetCenterOfMassTransform for all rigid actors.
  void DefaultOutputFn(Span<real> out) {
    int offset = 0;
    _scene->ForEachActor([&](Actor* actor) {
      if (!actor->IsStatic() && actor->GetType() == ActorType::Rigid) {
        auto transform = actor->GetCenterOfMassTransform(ExpectOK{});
        auto pos = transform.GetTranslation();
        for (int j = 0; j < 3; ++j) {
          out[offset++] = pos[j];
        }
        auto rot = transform.GetRotation().ToReal4();
        for (int j = 0; j < 4; ++j) {
          out[offset++] = rot[j];
        }
      }
    });
  }

  /// Default output backward: GetCenterOfMassTransformBackward for all rigid actors.
  void DefaultOutputBackwardFn(Span<real const> gradOutput) {
    int offset = 0;
    _scene->ForEachActor([&](Actor* actor) {
      if (!actor->IsStatic() && actor->GetType() == ActorType::Rigid) {
        GetCenterOfMassTransformBackward(
            actor, gradOutput.subspan(offset, RigidSize::kAll), ExpectOK{});
        offset += RigidSize::kAll;
      }
    });
  }

  /// Finite-difference gradient consistency test for backward functions.
  ///
  /// Tests the full chain: input → Step → output → sum loss → BackPropagate → input gradient.
  /// The loss is the sum of output values across all simulation steps. Each callback has a
  /// default:
  /// - Input default: SetArticulatedTargetPose with an internally built control matrix.
  /// - Input backward default: SetArticulatedTargetPoseBackward.
  /// - Output default: GetCenterOfMassTransform for all rigid actors.
  /// - Output backward default: GetCenterOfMassTransformBackward for all rigid actors.
  ///
  /// To test an **input** function, override inputVals/inputFn/inputBackwardFn and leave
  /// output defaults. To test an **output** function, override outputFn/outputBackwardFn and
  /// leave input defaults.
  ///
  /// @param input Values wrt gradient must be compared.
  /// @param inputFn Applies the input function at each step.
  /// @param inputBackwardFn Accumulates input gradients after each BackPropagate.
  /// @param outputSize Number of output values produced by outputFn.
  /// @param outputFn Reads output values into a pre-allocated buffer.
  /// @param outputBackwardFn Invokes the output dual with gradOutput.
  void RunFunctionTest(
      Span<real> input,
      InputFn const& inputFn,
      InputBackwardFn const& inputBackwardFn,
      int outputSize,
      OutputFn const& outputFn,
      OutputBackwardFn const& outputBackwardFn,
      TestParams const& params) {
    int const inputSize = isize(input);

    // Capture initial scene state before any runs.
    StateHandle initialSceneState = _scene->CaptureState(ExpectOK{});
    MOCHI_DEFER(_scene->ReleaseState(initialSceneState));

    // Run twice to ensure no debris data remains in backprop containers from the first run.
    for (int run = 0; run < kNumRuns; ++run) {
      // Restore scene to initial state before each run.
      _scene->RestoreState(initialSceneState, false, ExpectOK{});

      // Reset backprop containers before each run.
      diffsim::ResetBackPropagation(_scene, ExpectOK{});

      // --- Forward simulation, capturing states ---
      DynamicArray<StateHandle> statesPre(kNumSteps);
      DynamicArray<StateHandle> statesPost(kNumSteps);
      for (int i = 0; i < kNumSteps; ++i) {
        inputFn(i);
        statesPre[i] = _scene->CaptureState(ExpectOK{});
        _scene->Step(kDt);
        statesPost[i] = _scene->CaptureState(ExpectOK{});
      }

      // --- Construct gradOutput = ones ---
      DynamicArray<real> gradOutput(outputSize, 1_r);

      // --- Backward: prepare each step, apply output backward, then BackPropagate ---
      ColumnVector<real> analyticalGrad = ColumnVector<real>::Zero(inputSize);

      for (int i = kNumSteps - 1; i >= 0; --i) {
        PrepareBackPropagate(_scene, statesPost[i], statesPre[i], ExpectOK{});
        outputBackwardFn(MakeConstSpan(gradOutput));

        BackPropagate(_scene, ExpectOK{});

        inputBackwardFn(i, analyticalGrad);
      }

      // --- Finite differences ---
      ColumnVector<real> fdGrad = ColumnVector<real>::Zero(inputSize);
      for (int j = 0; j < inputSize; ++j) {
        auto evalLoss = [&]() {
          real loss = 0_r;
          DynamicArray<real> out(outputSize, 0_r);
          _scene->RestoreState(initialSceneState, false, ExpectOK{});
          for (int s = 0; s < kNumSteps; ++s) {
            inputFn(s);
            _scene->Step(kDt);
            outputFn(out);
            for (auto v : out) {
              loss += v;
            }
          }
          return loss;
        };

        // +eps
        input[j] += params.epsFiniteDiff;
        real lossP = evalLoss();

        // -eps
        input[j] -= 2_r * params.epsFiniteDiff;
        real lossM = evalLoss();

        fdGrad[j] = (lossP - lossM) / (2_r * params.epsFiniteDiff);

        // restore
        input[j] += params.epsFiniteDiff;
      }

      // Compare
      auto normDiff = ColumnVector<real>(analyticalGrad - fdGrad).Norm();
      auto maxNorm = Max(analyticalGrad.Norm(), fdGrad.Norm());
      EXPECT_NE(maxNorm, 0_r);
      EXPECT_NEAR(normDiff / maxNorm, 0_r, params.tol);

      for (auto& state : statesPre) {
        _scene->ReleaseState(state);
      }
      for (auto& state : statesPost) {
        _scene->ReleaseState(state);
      }
    }
  }

  void RunInputFunctionTest(
      Span<real> input,
      InputFn const& inputFn,
      InputBackwardFn const& inputBackwardFn,
      TestParams const& params = {}) {
    RunFunctionTest(
        input,
        inputFn,
        inputBackwardFn,
        _defaultOutputSize,
        [&](Span<real> out) { DefaultOutputFn(out); },
        [&](Span<real const> g) { DefaultOutputBackwardFn(g); },
        params);
  }

  void RunOutputFunctionTest(
      int outputSize,
      OutputFn const& outputFn,
      OutputBackwardFn const& outputBackwardFn,
      TestParams const& params = {}) {
    RunFunctionTest(
        _defaultControl,
        [&](int step) { DefaultInputFn(step); },
        [&](int step, ColumnVectorView<real> outGrad) { DefaultInputBackwardFn(step, outGrad); },
        outputSize,
        outputFn,
        outputBackwardFn,
        params);
  }

  void ExpectStep0TargetVelocityDiscarded(ColumnVector<real>& velocity, InputFn const& inputFn) {
    StateHandle initialSceneState = _scene->CaptureState(ExpectOK{});
    MOCHI_DEFER(_scene->ReleaseState(initialSceneState));

    ResetBackPropagation(_scene, ExpectOK{});

    DynamicArray<StateHandle> statesPre(kNumSteps);
    DynamicArray<StateHandle> statesPost(kNumSteps);
    for (int i = 0; i < kNumSteps; ++i) {
      inputFn(i);
      statesPre[i] = _scene->CaptureState(ExpectOK{});
      _scene->Step(kDt);
      statesPost[i] = _scene->CaptureState(ExpectOK{});
    }

    DynamicArray<real> gradOutput(_defaultOutputSize, 1_r);
    for (int i = kNumSteps - 1; i >= 0; --i) {
      PrepareBackPropagate(_scene, statesPost[i], statesPre[i], ExpectOK{});
      DefaultOutputBackwardFn(MakeConstSpan(gradOutput));
      BackPropagate(_scene, ExpectOK{});

      if (i == 0) {
        _scene->ForEachActor([&](Actor const* actor) {
          if (actor->GetType() == ActorType::Articulated &&
              actor->HasArticulatedPoseController(ExpectOK{})) {
            int const numDofs = actor->GetNumDofs();
            DynamicArray<real> grad(numDofs, 1_r);
            SetArticulatedTargetVelocityBackward(actor, grad, ExpectOK{});
            for (int j = 0; j < numDofs; ++j) {
              EXPECT_EQ(grad[j], 0_r);
            }
          }
        });
      }
    }

    for (auto& state : statesPre) {
      _scene->ReleaseState(state);
    }
    for (auto& state : statesPost) {
      _scene->ReleaseState(state);
    }

    auto evalLoss = [&]() {
      real loss = 0_r;
      DynamicArray<real> out(_defaultOutputSize, 0_r);
      _scene->RestoreState(initialSceneState, false, ExpectOK{});
      for (int s = 0; s < kNumSteps; ++s) {
        inputFn(s);
        _scene->Step(kDt);
        DefaultOutputFn(out);
        for (auto v : out) {
          loss += v;
        }
      }
      return loss;
    };

    real const lossBaseline = evalLoss();
    for (int j = 0; j < _totalInput; ++j) {
      velocity(j) += 10_r;
    }
    real const lossPerturbed = evalLoss();
    EXPECT_EQ(lossBaseline, lossPerturbed);
  }

 private:
  void BuildDefaultControl() {
    _defaultControl.Resize(_totalInput, kNumSteps);
    int inputOffset = 0;
    _scene->ForEachActor([&](Actor* actor) {
      if (actor->GetType() == ActorType::Articulated &&
          actor->HasArticulatedPoseController(ExpectOK{})) {
        int numDofs = actor->GetNumDofs();
        actor->GetArticulatedPose(
            _defaultControl.Col(0).MiddleRows(inputOffset, numDofs), ExpectOK{});
        inputOffset += numDofs;
      }
    });
    for (int i = 1; i < kNumSteps; ++i) {
      _defaultControl.Col(i) = _defaultControl.Col(0);
    }
    for (int j = 0; j < _totalInput; ++j) {
      for (int k = 0; k < kNumSteps; ++k) {
        _defaultControl(j, k) += 0.01_r * static_cast<real>(j + k + 1);
      }
    }
  }

  void CollectActorData() {
    _scene->ForEachActor([&](Actor* actor) {
      if (!actor->IsStatic() && !actor->IsNestedLinkActor()) {
        int numDofs = actor->GetNumDofs();
        bool hasController = actor->GetType() == ActorType::Articulated &&
            actor->HasArticulatedPoseController(ExpectOK{});
        _totalDofs += numDofs;
        _totalInput += hasController ? numDofs : 0;
      }
      if (!actor->IsStatic() && actor->GetType() == ActorType::Rigid) {
        _defaultOutputSize += RigidSize::kAll;
      }
    });
  }
};

// Test with inherited targets: set control target only at the first and last step.
TEST_IF_F(
    MOCHI_TEST_DIFFERENTIABLE_FUNCTIONS,
    DifferentiableFunctionsTest,
    SetArticulatedTargetPoseBackwardInheritedTarget) {
  int constexpr kLastStep = kNumSteps - 1;

  Matrix<real, krylov::kDynamic, 2> control(_totalInput, 2);
  control.Col(0) = _defaultControl.Col(0);
  control.Col(1) = _defaultControl.Col(kLastStep);

  // Set target pose only at the first and last step.
  auto inputFn = [&](int step) {
    if (step == 0 || step == kLastStep) {
      int const trueStep = step == kLastStep;
      int colOffset = 0; // Offset within the column (each column has _totalInput rows)
      _scene->ForEachActor([&](Actor* actor) {
        if (actor->GetType() == ActorType::Articulated &&
            actor->HasArticulatedPoseController(ExpectOK{})) {
          int numDofs = actor->GetNumDofs();
          auto targetPose = control.Col(trueStep).MiddleRows(colOffset, numDofs);
          actor->SetArticulatedTargetPose(targetPose, ExpectOK{});
          colOffset += numDofs;
        }
      });
    }
    // else: do nothing — target is inherited from previous step
  };

  // Extract gradient only at the first and last step, where the target was explicitly set.
  auto inputBackwardFn = [&](int step, ColumnVectorView<real> outGrad) {
    if (step == 0 || step == kLastStep) {
      int const trueStep = step == kLastStep;
      int colOffset = 0; // Offset within the column
      int gradOffset = trueStep * _totalInput; // Offset in flattened outGrad
      _scene->ForEachActor([&](Actor const* actor) {
        if (actor->GetType() == ActorType::Articulated &&
            actor->HasArticulatedPoseController(ExpectOK{})) {
          int numDofs = actor->GetNumDofs();
          DynamicArray<real> grad(numDofs, 0_r);
          SetArticulatedTargetPoseBackward(actor, grad, ExpectOK{});
          for (int j = 0; j < numDofs; ++j) {
            outGrad[gradOffset + colOffset + j] += grad[j];
          }
          colOffset += numDofs;
        }
      });
    }
  };

  RunInputFunctionTest(control, inputFn, inputBackwardFn, TestParams{.tol = 2e-2_r});
}

// Test SetArticulatedPoseFromJointsBackward with control applied at steps 1+.
// At step 0, set initial pose via SetArticulatedPoseFromJoints; at steps 1+, apply default control.
TEST_IF_F(
    MOCHI_TEST_DIFFERENTIABLE_FUNCTIONS,
    DifferentiableFunctionsTest,
    SetArticulatedPoseFromJointsBackwardWithControl) {
  // Collect initial poses for all articulated actors.
  int totalArticulatedDofs = 0;
  _scene->ForEachActor([&](Actor* actor) {
    if (actor->GetType() == ActorType::Articulated) {
      totalArticulatedDofs += actor->GetNumDofs();
    }
  });

  DynamicArray<real> initialPose(totalArticulatedDofs);
  int offset = 0;
  _scene->ForEachActor([&](Actor* actor) {
    if (actor->GetType() == ActorType::Articulated) {
      int numDofs = actor->GetNumDofs();
      actor->GetArticulatedPose(MakeSpan(initialPose).subspan(offset, numDofs), ExpectOK{});
      offset += numDofs;
    }
  });

  // Apply SetArticulatedPoseFromJoints at step 0 (initial pose, before simulation).
  // Apply default control otherwise, so the initial pose is not propagated to other steps.
  auto inputFn = [&](int step) {
    if (step == 0) {
      int off = 0;
      _scene->ForEachActor([&](Actor* actor) {
        if (actor->GetType() == ActorType::Articulated) {
          int numDofs = actor->GetNumDofs();
          actor->SetArticulatedPoseFromJoints(
              MakeConstSpan(initialPose).subspan(off, numDofs), ExpectOK{});
          off += numDofs;
        }
      });
    } else {
      DefaultInputFn(step);
    }
  };

  // Accumulate initial pose gradient only at step 0.
  auto inputBackwardFn = [&](int step, ColumnVectorView<real> outGrad) {
    if (step == 0) {
      int off = 0;
      _scene->ForEachActor([&](Actor const* actor) {
        if (actor->GetType() == ActorType::Articulated) {
          int numDofs = actor->GetNumDofs();
          DynamicArray<real> grad(numDofs, 0_r);
          SetArticulatedPoseFromJointsBackward(actor, grad, ExpectOK{});
          for (int j = 0; j < numDofs; ++j) {
            outGrad[off + j] += grad[j];
          }
          off += numDofs;
        }
      });
    }
  };

  RunInputFunctionTest(initialPose, inputFn, inputBackwardFn);
}

// Test SetArticulatedPoseFromJointsBackward with no control applied at any step.
// At step 0, set initial pose via SetArticulatedPoseFromJoints; at all steps, no control is
// applied. The controller targets are inherited from the initial pose.
TEST_IF_F(
    MOCHI_TEST_DIFFERENTIABLE_FUNCTIONS,
    DifferentiableFunctionsTest,
    SetArticulatedPoseFromJointsBackwardNoControl) {
  // Collect initial poses for all articulated actors.
  int totalArticulatedDofs = 0;
  _scene->ForEachActor([&](Actor* actor) {
    if (actor->GetType() == ActorType::Articulated) {
      totalArticulatedDofs += actor->GetNumDofs();
    }
  });

  DynamicArray<real> initialPose(totalArticulatedDofs);
  int offset = 0;
  _scene->ForEachActor([&](Actor* actor) {
    if (actor->GetType() == ActorType::Articulated) {
      int numDofs = actor->GetNumDofs();
      actor->GetArticulatedPose(MakeSpan(initialPose).subspan(offset, numDofs), ExpectOK{});
      offset += numDofs;
    }
  });

  // Apply SetArticulatedPoseFromJoints at step 0 only. No control at any step.
  auto inputFn = [&](int step) {
    if (step == 0) {
      int off = 0;
      _scene->ForEachActor([&](Actor* actor) {
        if (actor->GetType() == ActorType::Articulated) {
          int numDofs = actor->GetNumDofs();
          actor->SetArticulatedPoseFromJoints(
              MakeConstSpan(initialPose).subspan(off, numDofs), ExpectOK{});
          off += numDofs;
        }
      });
    }
    // No control applied at any step — targets inherited from initial pose.
  };

  // Accumulate initial pose gradient only at step 0.
  auto inputBackwardFn = [&](int step, ColumnVectorView<real> outGrad) {
    if (step == 0) {
      int off = 0;
      _scene->ForEachActor([&](Actor const* actor) {
        if (actor->GetType() == ActorType::Articulated) {
          int numDofs = actor->GetNumDofs();
          DynamicArray<real> grad(numDofs, 0_r);
          SetArticulatedPoseFromJointsBackward(actor, grad, ExpectOK{});
          for (int j = 0; j < numDofs; ++j) {
            outGrad[off + j] += grad[j];
          }
          off += numDofs;
        }
      });
    }
  };

  RunInputFunctionTest(initialPose, inputFn, inputBackwardFn);
}

TEST_IF_F(MOCHI_TEST_DIFFERENTIABLE_FUNCTIONS, DifferentiableFunctionsTest, SetVelocityBackward) {
  // Count standalone rigid actors and total velocity DOFs (6 per actor).
  int totalVelDofs = 0;
  _scene->ForEachActor([&](Actor* actor) {
    if (!actor->IsStatic() && actor->GetType() == ActorType::Rigid && !actor->IsNestedLinkActor()) {
      totalVelDofs += RigidSize::kDAll;
    }
  });

  // Build velocity for step 0
  ColumnVector<real> velocity(totalVelDofs);
  for (int j = 0; j < totalVelDofs; ++j) {
    velocity(j) = 1e-1_r * static_cast<real>(j + 1);
  }

  // Apply SetVelocity at step 0 only (initial velocity).
  // Apply control to the articulated actors at all steps.
  auto inputFn = [&](int step) {
    if (step == 0) {
      int offset = 0;
      _scene->ForEachActor([&](Actor* actor) {
        if (!actor->IsStatic() && actor->GetType() == ActorType::Rigid &&
            !actor->IsNestedLinkActor()) {
          auto vel = velocity.MiddleRows(offset, RigidSize::kDAll);
          Real3 linVel{vel[0], vel[1], vel[2]};
          Real3 angVel{vel[3], vel[4], vel[5]};
          actor->SetVelocity(linVel, angVel, ExpectOK{});
          offset += RigidSize::kDAll;
        }
      });
    }
    DefaultInputFn(step);
  };

  auto inputBackwardFn = [&](int step, ColumnVectorView<real> outGrad) {
    if (step == 0) {
      int offset = 0;
      _scene->ForEachActor([&](Actor const* actor) {
        if (!actor->IsStatic() && actor->GetType() == ActorType::Rigid &&
            !actor->IsNestedLinkActor()) {
          Real3 gradLin{};
          Real3 gradAng{};
          SetVelocityBackward(actor, gradLin, gradAng, ExpectOK{});
          for (int j = 0; j < 3; ++j) {
            outGrad[offset + j] += gradLin[j];
          }
          for (int j = 0; j < 3; ++j) {
            outGrad[offset + 3 + j] += gradAng[j];
          }
          offset += RigidSize::kDAll;
        }
      });
    }
  };

  RunInputFunctionTest(velocity, inputFn, inputBackwardFn);
}

TEST_IF_F(
    MOCHI_TEST_DIFFERENTIABLE_FUNCTIONS,
    DifferentiableFunctionsTest,
    SetCenterOfMassTransformBackward) {
  // Count standalone rigid actors and total transform DOFs (7 per actor: trans + quaternion).
  int totalTransformDofs = 0;
  _scene->ForEachActor([&](Actor* actor) {
    if (!actor->IsStatic() && actor->GetType() == ActorType::Rigid && !actor->IsNestedLinkActor()) {
      totalTransformDofs += RigidSize::kAll;
    }
  });

  // Build initial transforms as [trans(3), quaternion(4)] by reading current CoM transforms.
  ColumnVector<real> transforms(totalTransformDofs);
  int offset = 0;
  _scene->ForEachActor([&](Actor* actor) {
    if (!actor->IsStatic() && actor->GetType() == ActorType::Rigid && !actor->IsNestedLinkActor()) {
      ColumnVectorView<real, RigidSize::kAll> transformView(&transforms[offset], RigidSize::kAll);
      TransformToRawPose(actor->GetCenterOfMassTransform(ExpectOK{}), transformView);
      offset += RigidSize::kAll;
    }
  });

  // Apply SetCenterOfMassTransform at step 0 only (initial pose).
  // Apply default control at all steps.
  auto inputFn = [&](int step) {
    if (step == 0) {
      int off = 0;
      _scene->ForEachActor([&](Actor* actor) {
        if (!actor->IsStatic() && actor->GetType() == ActorType::Rigid &&
            !actor->IsNestedLinkActor()) {
          ColumnVectorView<real const, RigidSize::kAll> transformView(
              &transforms[off], RigidSize::kAll);
          actor->SetCenterOfMassTransform(TransformFromRawPose(transformView), ExpectOK{});
          off += RigidSize::kAll;
        }
      });
    }
    DefaultInputFn(step);
  };

  auto inputBackwardFn = [&](int step, ColumnVectorView<real> outGrad) {
    if (step == 0) {
      int off = 0;
      _scene->ForEachActor([&](Actor const* actor) {
        if (!actor->IsStatic() && actor->GetType() == ActorType::Rigid &&
            !actor->IsNestedLinkActor()) {
          NdArray<real, RigidSize::kAll> grad = {};
          SetCenterOfMassTransformBackward(actor, grad, ExpectOK{});
          for (int j = 0; j < RigidSize::kAll; ++j, ++off) {
            outGrad[off] += grad[j];
          }
        }
      });
    }
  };

  RunInputFunctionTest(transforms, inputFn, inputBackwardFn);
}

TEST_IF_F(
    MOCHI_TEST_DIFFERENTIABLE_FUNCTIONS,
    DifferentiableFunctionsTest,
    SetArticulatedJointVelocitiesBackward) {
  // Count total articulated DOFs.
  int totalArticulatedDofs = 0;
  _scene->ForEachActor([&](Actor* actor) {
    if (actor->GetType() == ActorType::Articulated) {
      totalArticulatedDofs += actor->GetNumDofs();
    }
  });

  // Build joint velocities for step 0.
  ColumnVector<real> jointVelocity(totalArticulatedDofs);
  for (int j = 0; j < totalArticulatedDofs; ++j) {
    jointVelocity(j) = 1e-1_r * static_cast<real>(j + 1);
  }

  // Apply SetArticulatedJointVelocities at step 0 only.
  // Apply default control at all steps.
  auto inputFn = [&](int step) {
    if (step == 0) {
      int offset = 0;
      _scene->ForEachActor([&](Actor* actor) {
        if (actor->GetType() == ActorType::Articulated) {
          int numDofs = actor->GetNumDofs();
          actor->SetArticulatedJointVelocities(
              jointVelocity.MiddleRows(offset, numDofs), ExpectOK{});
          offset += numDofs;
        }
      });
    }
    DefaultInputFn(step);
  };

  auto inputBackwardFn = [&](int step, ColumnVectorView<real> outGrad) {
    if (step == 0) {
      int offset = 0;
      _scene->ForEachActor([&](Actor const* actor) {
        if (actor->GetType() == ActorType::Articulated) {
          int numDofs = actor->GetNumDofs();
          DynamicArray<real> grad(numDofs, 0_r);
          SetArticulatedJointVelocitiesBackward(actor, grad, ExpectOK{});
          for (int j = 0; j < numDofs; ++j) {
            outGrad[offset + j] += grad[j];
          }
          offset += numDofs;
        }
      });
    }
  };

  RunInputFunctionTest(jointVelocity, inputFn, inputBackwardFn);
}

TEST_IF_F(
    MOCHI_TEST_DIFFERENTIABLE_FUNCTIONS,
    DifferentiableFunctionsTest,
    SetArticulatedTargetPoseBackward) {
  RunInputFunctionTest(
      _defaultControl,
      [&](int step) { DefaultInputFn(step); },
      [&](int step, ColumnVectorView<real> outGrad) { DefaultInputBackwardFn(step, outGrad); });
}

/// Test mixed usage: SetArticulatedTargetPose + SetArticulatedTargetVelocity at all steps.
/// Differentiability is not supported if the target velocity is set alone.
TEST_IF_F(
    MOCHI_TEST_DIFFERENTIABLE_FUNCTIONS,
    DifferentiableFunctionsTest,
    SetArticulatedTargetVelocityBackward) {
  // Build combined control input: target poses + velocities for all steps.
  // Layout: rows = DOFs, columns = [pose_step0, ..., pose_stepN-1, vel_step0, ..., vel_stepN-1].
  Matrix<real, krylov::kDynamic, 2 * kNumSteps> combinedControl(_totalInput, 2 * kNumSteps);

  // Fill poses from _defaultControl
  auto poseControl = combinedControl.LeftCols(kNumSteps);
  poseControl = _defaultControl;

  // Fill velocity columns with non-zero values
  auto velocityControl = combinedControl.RightCols(kNumSteps);
  for (int j = 0; j < _totalInput; ++j) {
    for (int k = 0; k < kNumSteps; ++k) {
      velocityControl(j, k) = 1e-1_r * static_cast<real>(j + k + 1);
    }
  }

  auto inputFn = [&](int step) {
    int inputOffset = 0;
    _scene->ForEachActor([&](Actor* actor) {
      if (actor->GetType() == ActorType::Articulated &&
          actor->HasArticulatedPoseController(ExpectOK{})) {
        int numDofs = actor->GetNumDofs();
        // Set target pose at every step.
        actor->SetArticulatedTargetPose(
            poseControl.Col(step).MiddleRows(inputOffset, numDofs), ExpectOK{});
        // Set target velocity at every step.
        actor->SetArticulatedTargetVelocity(
            velocityControl.Col(step).MiddleRows(inputOffset, numDofs), ExpectOK{});
        inputOffset += numDofs;
      }
    });
  };

  auto inputBackwardFn = [&](int step, ColumnVectorView<real> outGrad) {
    int inputOffset = 0;
    _scene->ForEachActor([&](Actor const* actor) {
      if (actor->GetType() == ActorType::Articulated &&
          actor->HasArticulatedPoseController(ExpectOK{})) {
        int numDofs = actor->GetNumDofs();
        // Read target pose gradient.
        {
          int const offset = step * _totalInput + inputOffset;
          DynamicArray<real> grad(numDofs, 0_r);
          SetArticulatedTargetPoseBackward(actor, grad, ExpectOK{});
          for (int j = 0; j < numDofs; ++j) {
            outGrad[offset + j] += grad[j];
          }
        }
        // Read target velocity gradient.
        {
          int const offset = (kNumSteps + step) * _totalInput + inputOffset;
          DynamicArray<real> grad(numDofs, 0_r);
          SetArticulatedTargetVelocityBackward(actor, grad, ExpectOK{});
          for (int j = 0; j < numDofs; ++j) {
            outGrad[offset + j] += grad[j];
          }
        }
        inputOffset += numDofs;
      }
    });
  };

  RunInputFunctionTest(combinedControl, inputFn, inputBackwardFn, TestParams{.tol = 2e-2_r});
}

/// Test mixed usage: SetArticulatedTargetPose + SetArticulatedTargetVelocity at step 0,
/// then only SetArticulatedTargetPose at remaining steps.
TEST_IF_F(
    MOCHI_TEST_DIFFERENTIABLE_FUNCTIONS,
    DifferentiableFunctionsTest,
    SetArticulatedTargetVelocityBackwardFirstStep) {
  // Build combined control input: target poses for all steps + velocity for step 0.
  // Layout: rows = DOFs, columns = [pose_step0, ..., pose_stepN-1, velocity].
  Matrix<real, krylov::kDynamic, kNumSteps + 1> combinedControl(_totalInput, kNumSteps + 1);

  // Fill poses from _defaultControl
  auto poseControl = combinedControl.LeftCols(kNumSteps);
  poseControl = _defaultControl;

  // Fill velocity column with non-zero values
  auto velocityControl = combinedControl.Col(kNumSteps);
  for (int j = 0; j < _totalInput; ++j) {
    velocityControl(j) = 1e-1_r * static_cast<real>(j + 1);
  }

  auto inputFn = [&](int step) {
    int inputOffset = 0;
    _scene->ForEachActor([&](Actor* actor) {
      if (actor->GetType() == ActorType::Articulated &&
          actor->HasArticulatedPoseController(ExpectOK{})) {
        int numDofs = actor->GetNumDofs();
        // Set target pose at every step.
        actor->SetArticulatedTargetPose(
            poseControl.Col(step).MiddleRows(inputOffset, numDofs), ExpectOK{});
        // At step 0, also set target velocity.
        if (step == 0) {
          actor->SetArticulatedTargetVelocity(
              velocityControl.MiddleRows(inputOffset, numDofs), ExpectOK{});
        }
        inputOffset += numDofs;
      }
    });
  };

  auto inputBackwardFn = [&](int step, ColumnVectorView<real> outGrad) {
    int inputOffset = 0;
    _scene->ForEachActor([&](Actor const* actor) {
      if (actor->GetType() == ActorType::Articulated &&
          actor->HasArticulatedPoseController(ExpectOK{})) {
        int numDofs = actor->GetNumDofs();
        // Read target pose gradient.
        {
          int const offset = step * _totalInput + inputOffset;
          DynamicArray<real> grad(numDofs, 0_r);
          SetArticulatedTargetPoseBackward(actor, grad, ExpectOK{});
          for (int j = 0; j < numDofs; ++j) {
            outGrad[offset + j] += grad[j];
          }
        }
        // At step 0, also read target velocity gradient.
        if (step == 0) {
          int const offset = kNumSteps * _totalInput + inputOffset;
          DynamicArray<real> grad(numDofs, 0_r);
          SetArticulatedTargetVelocityBackward(actor, grad, ExpectOK{});
          for (int j = 0; j < numDofs; ++j) {
            outGrad[offset + j] += grad[j];
          }
        }
        inputOffset += numDofs;
      }
    });
  };

  RunInputFunctionTest(combinedControl, inputFn, inputBackwardFn);
}

/// Test mixed usage: SetArticulatedPoseFromJoints at step 0, then SetArticulatedTargetPose
/// at all steps. Tests gradients for both the initial pose and target poses.
TEST_IF_F(
    MOCHI_TEST_DIFFERENTIABLE_FUNCTIONS,
    DifferentiableFunctionsTest,
    SetArticulatedPoseFromJointsAndTargetPoseBackward) {
  // Build combined control: [initialPose, pose_step0, ..., pose_stepN-1]
  // Layout: rows = DOFs, columns = [initialPose, targetPose_step0, ..., targetPose_stepN-1].
  Matrix<real, krylov::kDynamic, 1 + kNumSteps> combinedControl(_totalInput, 1 + kNumSteps);

  // Fill initial pose (column 0) from current articulated poses, perturbed slightly.
  auto initialPoseControl = combinedControl.Col(0);
  {
    int inputOffset = 0;
    _scene->ForEachActor([&](Actor* actor) {
      if (actor->GetType() == ActorType::Articulated &&
          actor->HasArticulatedPoseController(ExpectOK{})) {
        int numDofs = actor->GetNumDofs();
        actor->GetArticulatedPose(initialPoseControl.MiddleRows(inputOffset, numDofs), ExpectOK{});
        inputOffset += numDofs;
      }
    });
  }
  for (int j = 0; j < _totalInput; ++j) {
    initialPoseControl(j) += 0.01_r * static_cast<real>(j + 1);
  }

  // Fill target poses (columns 1 to kNumSteps) from _defaultControl.
  auto poseControl = combinedControl.RightCols(kNumSteps);
  poseControl = _defaultControl;

  // Step 0: SetArticulatedPoseFromJoints (sets state + both targets), then SetArticulatedTargetPose
  // (overwrites target_current).
  // Other steps: only SetArticulatedTargetPose.
  auto inputFn = [&](int step) {
    int inputOffset = 0;
    _scene->ForEachActor([&](Actor* actor) {
      if (actor->GetType() == ActorType::Articulated &&
          actor->HasArticulatedPoseController(ExpectOK{})) {
        int numDofs = actor->GetNumDofs();
        if (step == 0) {
          actor->SetArticulatedPoseFromJoints(
              initialPoseControl.MiddleRows(inputOffset, numDofs), ExpectOK{});
        }
        actor->SetArticulatedTargetPose(
            poseControl.Col(step).MiddleRows(inputOffset, numDofs), ExpectOK{});
        inputOffset += numDofs;
      }
    });
  };

  auto inputBackwardFn = [&](int step, ColumnVectorView<real> outGrad) {
    int inputOffset = 0;
    _scene->ForEachActor([&](Actor const* actor) {
      if (actor->GetType() == ActorType::Articulated &&
          actor->HasArticulatedPoseController(ExpectOK{})) {
        int numDofs = actor->GetNumDofs();
        DynamicArray<real> grad(numDofs, 0_r);
        // At step 0, read initial-pose gradient (column 0).
        if (step == 0) {
          SetArticulatedPoseFromJointsBackward(actor, grad, ExpectOK{});
          for (int j = 0; j < numDofs; ++j) {
            outGrad[inputOffset + j] += grad[j];
          }
        }
        // Read target-pose gradient (columns 1 to kNumSteps).
        int const offset = (1 + step) * _totalInput + inputOffset;
        SetArticulatedTargetPoseBackward(actor, grad, ExpectOK{});
        for (int j = 0; j < numDofs; ++j) {
          outGrad[offset + j] += grad[j];
        }
        inputOffset += numDofs;
      }
    });
  };

  TestParams params;

  // Rounding error is significantly affected by the order in which Scene::ForEachActor iterates
  // over actors. This test does not dictate the order. Instead the tolerance is increased to
  // account for orders differences.
  params.tol = 0.13_r;
  RunInputFunctionTest(combinedControl, inputFn, inputBackwardFn, params);
}

/// Test mixed usage: ResetArticulatedTargetPose followed by SetArticulatedTargetVelocity at
/// step 0. Reset assigns the velocity target to the controller (zeroing any pending velocity), so
/// the subsequent SetArticulatedTargetVelocity must reclaim velocity ownership for its gradient to
/// flow.
TEST_IF_F(
    MOCHI_TEST_DIFFERENTIABLE_FUNCTIONS,
    DifferentiableFunctionsTest,
    ResetArticulatedTargetPoseThenTargetVelocityBackward) {
  // Differentiable input: target velocity at step 0.
  ColumnVector<real> velocity(_totalInput);
  for (int j = 0; j < _totalInput; ++j) {
    velocity(j) = 1e-1_r * static_cast<real>(j + 1);
  }

  // Reset pose: the current articulated pose (constant, not differentiated).
  ColumnVector<real> resetPose(_totalInput);
  {
    int inputOffset = 0;
    _scene->ForEachActor([&](Actor* actor) {
      if (actor->GetType() == ActorType::Articulated &&
          actor->HasArticulatedPoseController(ExpectOK{})) {
        int numDofs = actor->GetNumDofs();
        actor->GetArticulatedPose(resetPose.MiddleRows(inputOffset, numDofs), ExpectOK{});
        inputOffset += numDofs;
      }
    });
  }

  // Step 0: reset the target pose, then set the target velocity (which must reclaim velocity
  // ownership from the reset). Remaining steps: default target pose.
  auto inputFn = [&](int step) {
    if (step != 0) {
      DefaultInputFn(step);
      return;
    }
    int inputOffset = 0;
    _scene->ForEachActor([&](Actor* actor) {
      if (actor->GetType() == ActorType::Articulated &&
          actor->HasArticulatedPoseController(ExpectOK{})) {
        int numDofs = actor->GetNumDofs();
        actor->ResetArticulatedTargetPose(resetPose.MiddleRows(inputOffset, numDofs), ExpectOK{});
        actor->SetArticulatedTargetVelocity(velocity.MiddleRows(inputOffset, numDofs), ExpectOK{});
        inputOffset += numDofs;
      }
    });
  };

  auto inputBackwardFn = [&](int step, ColumnVectorView<real> outGrad) {
    if (step != 0) {
      return;
    }
    int inputOffset = 0;
    _scene->ForEachActor([&](Actor const* actor) {
      if (actor->GetType() == ActorType::Articulated &&
          actor->HasArticulatedPoseController(ExpectOK{})) {
        int numDofs = actor->GetNumDofs();
        DynamicArray<real> grad(numDofs, 0_r);
        SetArticulatedTargetVelocityBackward(actor, grad, ExpectOK{});
        for (int j = 0; j < numDofs; ++j) {
          outGrad[inputOffset + j] += grad[j];
        }
        inputOffset += numDofs;
      }
    });
  };

  RunInputFunctionTest(velocity, inputFn, inputBackwardFn);
}

/// Test mixed usage: SetArticulatedPoseFromJoints followed by SetArticulatedTargetVelocity at
/// step 0. SetArticulatedPoseFromJoints resets the controller targets and zeroes the target
/// velocity, so the subsequent SetArticulatedTargetVelocity must reclaim velocity ownership. Tests
/// gradients for both the initial pose and the target velocity.
TEST_IF_F(
    MOCHI_TEST_DIFFERENTIABLE_FUNCTIONS,
    DifferentiableFunctionsTest,
    SetArticulatedPoseFromJointsThenTargetVelocityBackward) {
  // Build combined control, both columns applied at step 0.
  // Layout: rows = DOFs, columns = [pose, velocity].
  Matrix<real, krylov::kDynamic, 2> combinedControl(_totalInput, 2);

  // Fill pose (column 0) from the current articulated poses, perturbed slightly.
  auto poseControl = combinedControl.Col(0);
  {
    int inputOffset = 0;
    _scene->ForEachActor([&](Actor* actor) {
      if (actor->GetType() == ActorType::Articulated &&
          actor->HasArticulatedPoseController(ExpectOK{})) {
        int numDofs = actor->GetNumDofs();
        actor->GetArticulatedPose(poseControl.MiddleRows(inputOffset, numDofs), ExpectOK{});
        inputOffset += numDofs;
      }
    });
  }
  for (int j = 0; j < _totalInput; ++j) {
    poseControl(j) += 0.01_r * static_cast<real>(j + 1);
  }

  // Fill velocity (column 1) with non-zero values.
  auto velocityControl = combinedControl.Col(1);
  for (int j = 0; j < _totalInput; ++j) {
    velocityControl(j) = 1e-1_r * static_cast<real>(j + 1);
  }

  // Step 0: SetArticulatedPoseFromJoints (sets state + both targets, zeroes target velocity), then
  // SetArticulatedTargetVelocity (reclaims velocity ownership). Remaining steps: default target
  // pose.
  auto inputFn = [&](int step) {
    if (step != 0) {
      DefaultInputFn(step);
      return;
    }
    int inputOffset = 0;
    _scene->ForEachActor([&](Actor* actor) {
      if (actor->GetType() == ActorType::Articulated &&
          actor->HasArticulatedPoseController(ExpectOK{})) {
        int numDofs = actor->GetNumDofs();
        actor->SetArticulatedPoseFromJoints(
            poseControl.MiddleRows(inputOffset, numDofs), ExpectOK{});
        actor->SetArticulatedTargetVelocity(
            velocityControl.MiddleRows(inputOffset, numDofs), ExpectOK{});
        inputOffset += numDofs;
      }
    });
  };

  auto inputBackwardFn = [&](int step, ColumnVectorView<real> outGrad) {
    if (step != 0) {
      return;
    }
    int inputOffset = 0;
    _scene->ForEachActor([&](Actor const* actor) {
      if (actor->GetType() == ActorType::Articulated &&
          actor->HasArticulatedPoseController(ExpectOK{})) {
        int numDofs = actor->GetNumDofs();
        DynamicArray<real> grad(numDofs, 0_r);
        // Pose gradient (column 0).
        SetArticulatedPoseFromJointsBackward(actor, grad, ExpectOK{});
        for (int j = 0; j < numDofs; ++j) {
          outGrad[inputOffset + j] += grad[j];
        }
        // Target velocity gradient (column 1).
        int const velOffset = _totalInput + inputOffset;
        SetArticulatedTargetVelocityBackward(actor, grad, ExpectOK{});
        for (int j = 0; j < numDofs; ++j) {
          outGrad[velOffset + j] += grad[j];
        }
        inputOffset += numDofs;
      }
    });
  };

  TestParams params;
  // Rounding error is significantly affected by the order in which Scene::ForEachActor iterates
  // over actors. This test does not dictate the order. Instead the tolerance is increased to
  // account for orders differences.
  params.tol = 0.17_r;
  RunInputFunctionTest(combinedControl, inputFn, inputBackwardFn, params);
}

/// Test the reverse ordering: set a target velocity, then reset the target pose in the same step.
/// The reset overwrites the pending velocity with zero, so the velocity input is fully discarded:
/// no gradient may flow to it (the reset, not the velocity setter, owns the velocity target), and
/// perturbing it must not change the loss.
TEST_IF_F(
    MOCHI_TEST_DIFFERENTIABLE_FUNCTIONS,
    DifferentiableFunctionsTest,
    SetTargetVelocityThenResetArticulatedTargetPoseBackward) {
  // Target velocity applied (then discarded by the reset) at step 0.
  ColumnVector<real> velocity(_totalInput);
  for (int j = 0; j < _totalInput; ++j) {
    velocity(j) = 1e-1_r * static_cast<real>(j + 1);
  }

  // Reset pose: the current articulated pose (constant, not differentiated).
  ColumnVector<real> resetPose(_totalInput);
  {
    int inputOffset = 0;
    _scene->ForEachActor([&](Actor* actor) {
      if (actor->GetType() == ActorType::Articulated &&
          actor->HasArticulatedPoseController(ExpectOK{})) {
        int numDofs = actor->GetNumDofs();
        actor->GetArticulatedPose(resetPose.MiddleRows(inputOffset, numDofs), ExpectOK{});
        inputOffset += numDofs;
      }
    });
  }

  // Step 0: set the target velocity, THEN reset the target pose (which overwrites the pending
  // velocity with zero). Remaining steps: default target pose.
  auto inputFn = [&](int step) {
    if (step != 0) {
      DefaultInputFn(step);
      return;
    }
    int inputOffset = 0;
    _scene->ForEachActor([&](Actor* actor) {
      if (actor->GetType() == ActorType::Articulated &&
          actor->HasArticulatedPoseController(ExpectOK{})) {
        int numDofs = actor->GetNumDofs();
        actor->SetArticulatedTargetVelocity(velocity.MiddleRows(inputOffset, numDofs), ExpectOK{});
        actor->ResetArticulatedTargetPose(resetPose.MiddleRows(inputOffset, numDofs), ExpectOK{});
        inputOffset += numDofs;
      }
    });
  };

  ExpectStep0TargetVelocityDiscarded(velocity, inputFn);
}

/// Test the reverse ordering: set a target velocity, then reset target link transforms in the same
/// step. The reset overwrites the pending velocity with zero, so the velocity input is fully
/// discarded.
TEST_IF_F(
    MOCHI_TEST_DIFFERENTIABLE_FUNCTIONS,
    DifferentiableFunctionsTest,
    SetTargetVelocityThenResetArticulatedTargetLinkTransformsBackward) {
  ColumnVector<real> velocity(_totalInput);
  for (int j = 0; j < _totalInput; ++j) {
    velocity(j) = 1e-1_r * static_cast<real>(j + 1);
  }

  auto inputFn = [&](int step) {
    if (step != 0) {
      DefaultInputFn(step);
      return;
    }
    int inputOffset = 0;
    _scene->ForEachActor([&](Actor* actor) {
      if (actor->GetType() == ActorType::Articulated &&
          actor->HasArticulatedPoseController(ExpectOK{})) {
        int const numDofs = actor->GetNumDofs();
        auto const numLinks = actor->GetNestedLinkActors(ExpectOK{}).size();
        DynamicArray<TransformRT> resetLinkTransforms(numLinks);
        actor->GetArticulatedLinkTransforms(resetLinkTransforms, ExpectOK{});
        actor->SetArticulatedTargetVelocity(velocity.MiddleRows(inputOffset, numDofs), ExpectOK{});
        actor->ResetArticulatedTargetLinkTransforms(resetLinkTransforms, ExpectOK{});
        inputOffset += numDofs;
      }
    });
  };

  ExpectStep0TargetVelocityDiscarded(velocity, inputFn);
}

TEST_IF_F(
    MOCHI_TEST_DIFFERENTIABLE_FUNCTIONS,
    DifferentiableFunctionsTest,
    GetCenterOfMassTransformBackward) {
  RunOutputFunctionTest(
      _defaultOutputSize,
      [&](Span<real> out) { DefaultOutputFn(out); },
      [&](Span<real const> g) { DefaultOutputBackwardFn(g); });
}

TEST_IF_F(
    MOCHI_TEST_DIFFERENTIABLE_FUNCTIONS,
    DifferentiableFunctionsTest,
    GetContactForceWorldBackward) {
  // Select only every other actor, to prevent a trivial zero loss (i.e. forces add up to zero)
  int outputSize = 0;
  bool selectSize = true;
  _scene->ForEachActor([&](Actor* actor) {
    if (selectSize && !actor->IsStatic() && actor->GetType() == ActorType::Rigid) {
      actor->RegisterQuery(QueryType::TotalContactForce, ExpectOK{});
      outputSize += RigidSize::kDTrans;
    }
    selectSize = !selectSize;
  });

  auto outputFn = [&](Span<real> out) {
    int offset = 0;
    bool select = true;
    _scene->ForEachActor([&](Actor* actor) {
      if (select && !actor->IsStatic() && actor->GetType() == ActorType::Rigid) {
        auto const force = actor->GetContactForceWorld(ExpectOK{});
        for (int j = 0; j < RigidSize::kDTrans; ++j) {
          out[offset++] = force[j];
        }
      }
      select = !select;
    });
  };

  auto outputBackwardFn = [&](Span<real const> gradOutput) {
    int offset = 0;
    bool select = true;
    _scene->ForEachActor([&](Actor* actor) {
      if (select && !actor->IsStatic() && actor->GetType() == ActorType::Rigid) {
        GetContactForceWorldBackward(
            actor, gradOutput.subspan(offset, RigidSize::kDTrans), ExpectOK{});
        offset += RigidSize::kDTrans;
      }
      select = !select;
    });
  };

  RunOutputFunctionTest(outputSize, outputFn, outputBackwardFn, TestParams{.tol = 2e-2_r});
}

TEST_IF_F(
    MOCHI_TEST_DIFFERENTIABLE_FUNCTIONS,
    DifferentiableFunctionsTest,
    GetContactForceFromActorWorldBackward) {
  struct ActorPair {
    Actor* actor = nullptr;
    Actor* other = nullptr;
  };

  DynamicArray<ActorPair> pairs;
  bool selectSize = true;
  _scene->ForEachActor([&](Actor* actor) {
    if (selectSize && !actor->IsStatic() && actor->GetType() == ActorType::Rigid) {
      actor->RegisterQuery(QueryType::TotalContactForce, ExpectOK{});
      _scene->ForEachActor([&](Actor* other) {
        if (actor != other) {
          pairs.push_back({actor, other});
        }
      });
    }
    selectSize = !selectSize;
  });

  int const outputSize = RigidSize::kDTrans * isize(pairs);

  auto outputFn = [&](Span<real> out) {
    int offset = 0;
    bool hasNonZeroPairForce = false;
    for (auto const& pair : pairs) {
      auto const force = pair.actor->GetContactForceFromActorWorld(pair.other, ExpectOK{});
      hasNonZeroPairForce |= Norm(force) > 0_r;
      for (int j = 0; j < RigidSize::kDTrans; ++j) {
        out[offset++] = force[j];
      }
    }
    EXPECT_TRUE(hasNonZeroPairForce);
  };

  auto outputBackwardFn = [&](Span<real const> gradOutput) {
    int offset = 0;
    for (auto const& pair : pairs) {
      GetContactForceFromActorWorldBackward(
          pair.actor, pair.other, gradOutput.subspan(offset, RigidSize::kDTrans), ExpectOK{});
      offset += RigidSize::kDTrans;
    }
  };

  RunOutputFunctionTest(outputSize, outputFn, outputBackwardFn, TestParams{.tol = 2e-2_r});
}

TEST_IF_F(
    MOCHI_TEST_DIFFERENTIABLE_FUNCTIONS,
    DifferentiableFunctionsTest,
    GetRootTransformBackward) {
  int outputSize = 0;
  _scene->ForEachActor([&](Actor* actor) {
    if (!actor->IsStatic() && actor->GetType() == ActorType::Rigid) {
      outputSize += RigidSize::kAll;
    }
  });

  auto outputFn = [&](Span<real> out) {
    int offset = 0;
    _scene->ForEachActor([&](Actor* actor) {
      if (!actor->IsStatic() && actor->GetType() == ActorType::Rigid) {
        auto transform = actor->GetRootTransform();
        auto pos = transform.GetTranslation();
        for (int j = 0; j < 3; ++j) {
          // Add a non-uniform weight to the loss, which makes the translation gradient
          // non-parallel to the CoM offset. Otherwise, the test is not properly exercised.
          out[offset++] = static_cast<real>(2 * j + 1) * pos[j];
        }
        auto rot = transform.GetRotation().ToReal4();
        for (int j = 0; j < 4; ++j) {
          out[offset++] = rot[j];
        }
      }
    });
  };

  auto outputBackwardFn = [&](Span<real const> gradOutput) {
    int offset = 0;
    _scene->ForEachActor([&](Actor* actor) {
      if (!actor->IsStatic() && actor->GetType() == ActorType::Rigid) {
        DynamicArray<real> weightedGradOutput(
            gradOutput.begin() + offset, gradOutput.begin() + offset + RigidSize::kAll);
        for (int j = 0; j < 3; ++j) {
          // Add the same non-uniform weight to the gradient.
          weightedGradOutput[j] *= static_cast<real>(2 * j + 1);
        }
        GetRootTransformBackward(actor, MakeConstSpan(weightedGradOutput), ExpectOK{});
        offset += RigidSize::kAll;
      }
    });
  };

  RunOutputFunctionTest(outputSize, outputFn, outputBackwardFn);
}

TEST_IF_F(
    MOCHI_TEST_DIFFERENTIABLE_FUNCTIONS,
    DifferentiableFunctionsTest,
    GetArticulatedPoseBackward) {
  // Compute total output size for GetArticulatedPose across all articulated actors.
  int outputSize = 0;
  _scene->ForEachActor([&](Actor* actor) {
    if (actor->GetType() == ActorType::Articulated) {
      outputSize += actor->GetNumDofs();
    }
  });

  auto outputFn = [&](Span<real> out) {
    int offset = 0;
    _scene->ForEachActor([&](Actor* actor) {
      if (actor->GetType() == ActorType::Articulated) {
        int numDofs = actor->GetNumDofs();
        actor->GetArticulatedPose(out.subspan(offset, numDofs), ExpectOK{});
        offset += numDofs;
      }
    });
  };

  auto outputBackwardFn = [&](Span<real const> gradOutput) {
    int offset = 0;
    _scene->ForEachActor([&](Actor* actor) {
      if (actor->GetType() == ActorType::Articulated) {
        int numDofs = actor->GetNumDofs();
        GetArticulatedPoseBackward(actor, gradOutput.subspan(offset, numDofs), ExpectOK{});
        offset += numDofs;
      }
    });
  };

  RunOutputFunctionTest(outputSize, outputFn, outputBackwardFn, TestParams{.tol = 2e-2_r});
}

TEST_IF_F(
    MOCHI_TEST_DIFFERENTIABLE_FUNCTIONS,
    DifferentiableFunctionsTest,
    SetExternalForcesOnDofsBackward) {
  // Build force control: for each eligible actor (articulated or standalone rigid),
  // apply forces on all DOFs.
  int totalForceInput = 0;
  _scene->ForEachActor([&](Actor* actor) {
    if (!actor->IsStatic() && !actor->IsNestedLinkActor()) {
      totalForceInput += actor->GetNumDofs();
    }
  });

  Matrix<real, krylov::kDynamic, kNumSteps> forceControl(totalForceInput, kNumSteps);
  for (int j = 0; j < totalForceInput; ++j) {
    for (int k = 0; k < kNumSteps; ++k) {
      forceControl(j, k) = 1e-1_r * static_cast<real>(j + k + 1);
    }
  }

  auto inputFn = [&](int step) {
    int offset = 0;
    _scene->ForEachActor([&](Actor* actor) {
      if (!actor->IsStatic() && !actor->IsNestedLinkActor()) {
        int numDofs = actor->GetNumDofs();
        DynamicArray<int> dofs;
        dofs.resize_noinit(numDofs);
        std::iota(dofs.begin(), dofs.end(), 0);
        actor->SetExternalForcesOnDofs(
            dofs, forceControl.Col(step).MiddleRows(offset, numDofs), ExpectOK{});
        offset += numDofs;
      }
    });
  };

  auto inputBackwardFn = [&](int step, ColumnVectorView<real> outGrad) {
    int offset = step * totalForceInput;
    _scene->ForEachActor([&](Actor const* actor) {
      if (!actor->IsStatic() && !actor->IsNestedLinkActor()) {
        int numDofs = actor->GetNumDofs();
        DynamicArray<int> dofs;
        dofs.resize_noinit(numDofs);
        std::iota(dofs.begin(), dofs.end(), 0);
        DynamicArray<real> grad(numDofs, 0_r);
        SetExternalForcesOnDofsBackward(actor, dofs, grad, ExpectOK{});
        for (int j = 0; j < numDofs; ++j, ++offset) {
          outGrad[offset] += grad[j];
        }
      }
    });
  };

  RunInputFunctionTest(forceControl, inputFn, inputBackwardFn);
}

#undef MOCHI_TEST_DIFFERENTIABLE_FUNCTIONS

} // namespace
