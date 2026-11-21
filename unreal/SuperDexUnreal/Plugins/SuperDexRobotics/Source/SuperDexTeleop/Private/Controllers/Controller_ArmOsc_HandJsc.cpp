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

#include "Controllers/Controller_ArmOsc_HandJsc.h"

#include "ISuperDexRoboticsModule.h"
#include "MochiConversions.h"
#include "MochiPhysicsModule.h"
#include "SuperDexBot.h"
#include "SuperDexBotComponent.h"
#include "SuperDexTeleopModule.h"
#include "TeleopNode.h"

namespace {
// Values[Dof] for each of Dofs.
TArray<double> Gather(TConstArrayView<mochi::real> Values, TConstArrayView<int32> Dofs) {
  TArray<double> Out;
  Out.Reserve(Dofs.Num());
  for (const int32 Dof : Dofs) {
    Out.Add(static_cast<double>(Values[Dof]));
  }
  return Out;
}
} // namespace

superdex::robotics::ControllerBasicOscPd::Params UController_ArmOsc_HandJsc::MakeArmParams() const {
  superdex::robotics::ControllerBasicOscPd::Params Params;
  Params.Kp_p = static_cast<mochi::real>(ArmPositionStiffness);
  Params.Kd_p = static_cast<mochi::real>(ArmPositionDamping);
  Params.Kp_r = static_cast<mochi::real>(ArmRotationStiffness);
  Params.Kd_r = static_cast<mochi::real>(ArmRotationDamping);
  Params.bApplyMaxErrorMagnitudeNormalization = true;
  Params.maxTranslationError = static_cast<mochi::real>(ArmMaxPositionError);
  Params.maxRotationError = static_cast<mochi::real>(ArmMaxRotationError);
  // Scale the arm's efforts down, in proportion, to its joints' effort limits.
  Params.bApplyMaxOSCTorqueNormalization = true;
  return Params;
}

TArray<double> UController_ArmOsc_HandJsc::ComputeNullSpaceBias(
    TConstArrayView<double> Jacobian,
    TConstArrayView<double> Q,
    TConstArrayView<double> QDot,
    TConstArrayView<double> TargetQ,
    double Gain,
    double Damping) {
  constexpr int32 Rows = 6;
  const int32 N = Q.Num();
  if (N == 0 || QDot.Num() != N || TargetQ.Num() != N || Jacobian.Num() != Rows * N) {
    return {};
  }
  // The joint-space task: a spring toward TargetQ, damped.
  TArray<double> Tau;
  Tau.SetNum(N);
  for (int32 I = 0; I < N; ++I) {
    Tau[I] = Gain * (TargetQ[I] - Q[I]) - Damping * QDot[I];
  }
  // N Tau = Tau - J^T Y, with (J J^T) Y = J Tau. J J^T is symmetric positive definite for a J of
  // full rank; a tiny regularization keeps it so near singular configurations, where J loses rank.
  double A[Rows][Rows];
  double Y[Rows];
  for (int32 R = 0; R < Rows; ++R) {
    Y[R] = 0;
    for (int32 I = 0; I < N; ++I) {
      Y[R] += Jacobian[R * N + I] * Tau[I];
    }
    for (int32 C = 0; C < Rows; ++C) {
      A[R][C] = 0;
      for (int32 I = 0; I < N; ++I) {
        A[R][C] += Jacobian[R * N + I] * Jacobian[C * N + I];
      }
    }
    A[R][R] += 1e-9;
  }
  // Solve A Y = Y in place by Cholesky (A = L L^T, L in A's lower triangle).
  for (int32 C = 0; C < Rows; ++C) {
    for (int32 K = 0; K < C; ++K) {
      A[C][C] -= A[C][K] * A[C][K];
    }
    if (A[C][C] <= 0) {
      return {};
    }
    A[C][C] = FMath::Sqrt(A[C][C]);
    for (int32 R = C + 1; R < Rows; ++R) {
      for (int32 K = 0; K < C; ++K) {
        A[R][C] -= A[R][K] * A[C][K];
      }
      A[R][C] /= A[C][C];
    }
  }
  for (int32 R = 0; R < Rows; ++R) {
    for (int32 K = 0; K < R; ++K) {
      Y[R] -= A[R][K] * Y[K];
    }
    Y[R] /= A[R][R];
  }
  for (int32 R = Rows - 1; R >= 0; --R) {
    for (int32 K = R + 1; K < Rows; ++K) {
      Y[R] -= A[K][R] * Y[K];
    }
    Y[R] /= A[R][R];
  }
  for (int32 I = 0; I < N; ++I) {
    for (int32 R = 0; R < Rows; ++R) {
      Tau[I] -= Jacobian[R * N + I] * Y[R];
    }
  }
  return Tau;
}

superdex::robotics::ControllerBasicJscPd::Params UController_ArmOsc_HandJsc::MakeHandParams(
    int32 NumDofs,
    TConstArrayView<int32> InHandDofs) const {
  superdex::robotics::ControllerBasicJscPd::Params Params;
  // No gains and no effort on the arm's DOFs: the arm controller drives those.
  for (auto* Values : {&Params.Kp, &Params.Kd, &Params.saturation, &Params.deadband}) {
    *Values = mochi::DynamicArray<mochi::real>(NumDofs);
    for (mochi::real& Value : *Values) {
      Value = 0;
    }
  }
  for (const int32 Dof : InHandDofs) {
    Params.Kp[Dof] = static_cast<mochi::real>(HandJointStiffness);
    Params.Kd[Dof] = static_cast<mochi::real>(HandJointDamping);
    Params.saturation[Dof] = static_cast<mochi::real>(HandJointMaxEffort);
  }
  return Params;
}

bool UController_ArmOsc_HandJsc::Initialize(const USuperDexBotComponent* Component) {
  TeleopNode = Component->GetOwner()->GetComponentByClass<UTeleopNodeRobotComponent>();
  if (TeleopNode == nullptr) {
    UE_LOG(
        LogSuperDexTeleop,
        Error,
        TEXT("[UController_ArmOsc_HandJsc] No UTeleopNodeRobotComponent on actor [%s]."),
        *GetNameSafe(Component->GetOwner()));
    return false;
  }
  return true;
}

void UController_ArmOsc_HandJsc::OnBotSpawned(USuperDexBotComponent* Component) {
  UMochiArticulatedActorObject* ArticulatedActor = Component->GetMochiArticulatedActor();
  const USuperDexBot* BotAsset = Component->GetBotAsset();
  if (TeleopNode == nullptr || ArticulatedActor == nullptr || BotAsset == nullptr) {
    return;
  }
  RobotActorHandle = ArticulatedActor->GetArticulatedActorHandle();
  // The arm controller reads its joints' effort limits from the bot's description.
  const superdex::robotics::BotPrefab* Prefab = &BotAsset->GetBotPrefab();

  mochi::Error Error;
  UMochiSceneObject* MochiScene = ArticulatedActor->GetMochiScene();
  MochiScene->QueueOrExecuteCommand([this, Prefab, &Error](mochi::Scene* Scene) {
    mochi::Actor* Robot = Scene->GetActor(RobotActorHandle);
    auto* Robotics = ISuperDexRoboticsModule::Get()->GetRoboticsContext();
    ArmControllerHandle = Robotics->CreateController(
        superdex::robotics::ControllerBasicOscPd::TypeName(), Prefab, Robot, "", Error);
    if (!Error.IsOK()) {
      return;
    }
    ArmController = static_cast<superdex::robotics::ControllerBasicOscPd*>(
        Robotics->GetController(ArmControllerHandle));
    const FString RobotName = Robot->GetName();
    ArmController->Initialize(
        TCHAR_TO_UTF8(*(RobotName + TEXT("/") + BaseLinkName)),
        TCHAR_TO_UTF8(*(RobotName + TEXT("/") + EELinkName)),
        Error);
    if (!Error.IsOK()) {
      return;
    }
    HandControllerHandle = Robotics->CreateController(
        superdex::robotics::ControllerBasicJscPd::TypeName(), Prefab, Robot, "", Error);
    if (!Error.IsOK()) {
      return;
    }
    HandController = static_cast<superdex::robotics::ControllerBasicJscPd*>(
        Robotics->GetController(HandControllerHandle));

    // The arm is the arm controller's DOFs; the hand is all the others.
    ArmDofs.Reset();
    HandDofs.Reset();
    for (const int Dof : ArmController->GetControlledDofs()) {
      ArmDofs.Add(Dof);
    }
    for (int32 Dof = 0; Dof < Robot->GetNumDofs(); ++Dof) {
      if (!ArmDofs.Contains(Dof)) {
        HandDofs.Add(Dof);
      }
    }
  });
  MochiScene->WaitForQueuedCommands();
  if (!Error.IsOK()) {
    UE_LOG(
        LogSuperDexTeleop,
        Error,
        TEXT("[UController_ArmOsc_HandJsc] Creating the arm and hand controllers failed: %hs"),
        Error.GetDescription());
    return;
  }
  if (ArmDofs.Num() != TeleopNode->NumArmJoints || HandDofs.Num() != TeleopNode->NumHandJoints) {
    UE_LOG(
        LogSuperDexTeleop,
        Error,
        TEXT(
            "[UController_ArmOsc_HandJsc] The bot has %d arm joints (%s to %s) and %d hand joints,"
            " but the TeleopNode expects %d and %d."),
        ArmDofs.Num(),
        *BaseLinkName,
        *EELinkName,
        HandDofs.Num(),
        TeleopNode->NumArmJoints,
        TeleopNode->NumHandJoints);
    return;
  }
  bHasHeldTargets = false;
  bWarnedJacobianSize = false;
  bInitialized = true;
}

void UController_ArmOsc_HandJsc::Shutdown(const USuperDexBotComponent* Component) {
  bInitialized = false;
  UMochiArticulatedActorObject* ArticulatedActor = Component->GetMochiArticulatedActor();
  UMochiSceneObject* MochiScene = ArticulatedActor ? ArticulatedActor->GetMochiScene() : nullptr;
  // Without a scene, the controllers went with it.
  if (MochiScene != nullptr && (ArmController != nullptr || HandController != nullptr)) {
    MochiScene->QueueOrExecuteCommand([this](mochi::Scene* Scene) {
      auto* Robotics = ISuperDexRoboticsModule::Get()->GetRoboticsContext();
      Robotics->DestroyController(ArmControllerHandle);
      Robotics->DestroyController(HandControllerHandle);
    });
    MochiScene->WaitForQueuedCommands();
  }
  ArmController = nullptr;
  HandController = nullptr;
}

bool UController_ArmOsc_HandJsc::StepController(
    const USuperDexBotComponent* Component,
    const FMochiArticulationState& State,
    double MochiDeltaTime,
    mochi::Actor* ArticulatedActor,
    mochi::Scene* Scene) {
  if (!bInitialized) {
    return false;
  }
  mochi::Error Error;
  auto Failed = [&Error](const TCHAR* Step) {
    UE_LOG(
        LogSuperDexTeleop,
        Error,
        TEXT("[UController_ArmOsc_HandJsc] %s failed: %hs"),
        Step,
        Error.GetDescription());
    return false;
  };

  // The robot's current state, as each controller observes it.
  auto ArmObsv = ArmController->GetCurrentObservationsFromMochi(Error);
  if (!Error.IsOK()) {
    return Failed(TEXT("Observing the arm"));
  }
  auto HandObsv = HandController->GetCurrentObservationsFromMochi(Error);
  if (!Error.IsOK()) {
    return Failed(TEXT("Observing the hand"));
  }
  HandObsv.dt = static_cast<mochi::real>(MochiDeltaTime);

  // The targets: the operator's latest, or else the last ones held (at first, the current pose).
  if (!bHasHeldTargets) {
    HeldArmTarget = mochi::Invert(ArmObsv.worldFromRoot) * ArmObsv.worldFromEELink;
    HeldHandTargets.SetNum(HandDofs.Num());
    for (int32 Joint = 0; Joint < HandDofs.Num(); ++Joint) {
      HeldHandTargets[Joint] = State.JointPositions[HandDofs[Joint]];
    }
    ArmBiasTarget = Gather(State.JointPositions, ArmDofs);
    bHasHeldTargets = true;
  }
  FTransform ArmPoseTarget;
  if (TeleopNode->GetArmPoseTarget(ArmPoseTarget)) {
    HeldArmTarget = FUnrealToMochiConverter().TransformToMochi_RT(ArmPoseTarget);
  }
  TArray<double> HandJointTargets;
  HandJointTargets.SetNum(HandDofs.Num());
  if (TeleopNode->GetHandJointTargets(HandJointTargets) &&
      HandJointTargets.Num() == HandDofs.Num()) {
    for (int32 Joint = 0; Joint < HandDofs.Num(); ++Joint) {
      HeldHandTargets[Joint] = static_cast<mochi::real>(HandJointTargets[Joint]);
    }
  }

  // The arm: an operational-space PD on its end effector.
  ArmController->SetParams(MakeArmParams(), Error);
  if (!Error.IsOK()) {
    return Failed(TEXT("Setting the arm parameters"));
  }
  superdex::robotics::ControllerBasicOscPd::Target ArmTarget;
  ArmTarget.rootFromTargetEE = HeldArmTarget;
  const mochi::Span<const mochi::real> ArmEfforts =
      ArmController->ComputeOutput(ArmObsv, ArmTarget, Error);
  if (!Error.IsOK()) {
    return Failed(TEXT("Computing the arm efforts"));
  }

  // The hand: a joint-space PD on its joints. Its arm targets are the arm's current positions, and
  // carry no gain anyway.
  const int32 NumDofs = State.JointPositions.Num();
  HandController->SetParams(MakeHandParams(NumDofs, HandDofs), Error);
  if (!Error.IsOK()) {
    return Failed(TEXT("Setting the hand parameters"));
  }
  superdex::robotics::ControllerBasicJscPd::Target HandTarget;
  HandTarget.targetPose = HandObsv.dofPositions;
  for (int32 Joint = 0; Joint < HandDofs.Num(); ++Joint) {
    HandTarget.targetPose[HandDofs[Joint]] = HeldHandTargets[Joint];
  }
  const mochi::Span<const mochi::real> HandEfforts =
      HandController->ComputeOutput(HandObsv, HandTarget, Error);
  if (!Error.IsOK()) {
    return Failed(TEXT("Computing the hand efforts"));
  }

  // Each controller leaves the other's DOFs at zero effort, so their sum drives the whole robot.
  TArray<mochi::real> Efforts;
  Efforts.SetNumZeroed(NumDofs);
  for (int32 Dof = 0; Dof < NumDofs; ++Dof) {
    Efforts[Dof] = ArmEfforts[Dof] + HandEfforts[Dof];
  }

  // The arm's null space bias, from the end effector's Jacobian in the arm's joints (its columns of
  // the Jacobian in all the robot's DOFs).
  if (NullSpaceBiasGain > 0 || NullSpaceBiasDamping > 0) {
    if (static_cast<int32>(ArmObsv.eeJacobian.size()) == 6 * NumDofs) {
      TArray<double> ArmJacobian;
      ArmJacobian.SetNum(6 * ArmDofs.Num());
      for (int32 Row = 0; Row < 6; ++Row) {
        for (int32 Joint = 0; Joint < ArmDofs.Num(); ++Joint) {
          ArmJacobian[Row * ArmDofs.Num() + Joint] =
              static_cast<double>(ArmObsv.eeJacobian[Row * NumDofs + ArmDofs[Joint]]);
        }
      }
      const TArray<double> Bias = ComputeNullSpaceBias(
          ArmJacobian,
          Gather(State.JointPositions, ArmDofs),
          Gather(State.JointVelocities, ArmDofs),
          ArmBiasTarget,
          NullSpaceBiasGain,
          NullSpaceBiasDamping);
      for (int32 Joint = 0; Joint < Bias.Num(); ++Joint) {
        Efforts[ArmDofs[Joint]] += static_cast<mochi::real>(Bias[Joint]);
      }
    } else if (!bWarnedJacobianSize) {
      bWarnedJacobianSize = true;
      UE_LOG(
          LogSuperDexTeleop,
          Warning,
          TEXT(
              "[UController_ArmOsc_HandJsc] The end effector's Jacobian has %d entries, not 6 x %d;"
              " no null space bias."),
          static_cast<int32>(ArmObsv.eeJacobian.size()),
          NumDofs);
    }
  }
  SetArticulatedActorJointEfforts(ArticulatedActor, Efforts);

  // Report the robot's state to the operator. Without feedforward terms, the control policy's
  // efforts are the applied ones.
  TeleopNode->SetArmJointStates(Gather(State.JointPositions, ArmDofs));
  TeleopNode->SetArmJointVelocities(Gather(State.JointVelocities, ArmDofs));
  TeleopNode->SetArmJointEfforts(Gather(Efforts, ArmDofs));
  TeleopNode->SetArmJointControlPolicyEfforts(Gather(Efforts, ArmDofs));
  TeleopNode->SetHandJointStates(Gather(State.JointPositions, HandDofs));
  TeleopNode->SetHandJointVelocities(Gather(State.JointVelocities, HandDofs));
  TeleopNode->SetHandJointEfforts(Gather(Efforts, HandDofs));
  TeleopNode->SetHandJointControlPolicyEfforts(Gather(Efforts, HandDofs));
  return true;
}
