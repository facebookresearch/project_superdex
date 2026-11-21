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

#pragma once

#include "SuperDexBotController.h"

#include <superdex_robotics/controllers/controller_basic_jsc_pd.h>
#include <superdex_robotics/controllers/controller_basic_osc_pd.h>
#include <superdex_robotics/core/context.h>

#include <atomic>

#include "Controller_ArmOsc_HandJsc.generated.h"

class UTeleopNodeRobotComponent;

/**
 * Drives an arm + hand robot from a Roboperator's TeleopNode: an operational-space PD controller
 * (superdex::robotics::ControllerBasicOscPd) moves the arm's end effector to the arm pose target,
 * and a joint-space PD controller (superdex::robotics::ControllerBasicJscPd) moves the hand's joints
 * to the hand joint targets. Their joint efforts are summed and applied every simulation step, and
 * the robot's joint states are published back to the operator.
 *
 * The arm is the chain of joints from BaseLinkName to EELinkName; the hand is every other joint.
 * Their joint counts must match the robot-side TeleopNode's NumArmJoints and NumHandJoints. Until
 * the operator sends targets, and whenever they stop, the robot holds its last targets, starting
 * from the pose it spawned in.
 *
 * A redundant arm (more joints than the 6 its end-effector pose takes) is also pulled toward the
 * joint configuration it spawned in, within the null space of its end-effector task (see
 * ComputeNullSpaceBias), so it does not drift through its redundant configurations.
 *
 * Kept minimal as an example: no gravity compensation (simulate the bot without gravity, or add a
 * feedforward term).
 */
UCLASS(Blueprintable)
class SUPERDEXTELEOP_API UController_ArmOsc_HandJsc : public USuperDexBotController {
  GENERATED_BODY()

 public:
  // The first link of the arm's joint chain.
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Arm")
  FString BaseLinkName = TEXT("fr3_link0");

  // The arm's end-effector link, which tracks the arm pose target.
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Arm")
  FString EELinkName = TEXT("fr3_link8");

  // End-effector position stiffness [N/m].
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Arm", meta = (ClampMin = "0"))
  double ArmPositionStiffness = 2000.0;

  // End-effector position damping [N s/m].
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Arm", meta = (ClampMin = "0"))
  double ArmPositionDamping = 100.0;

  // End-effector rotation stiffness [N m/rad].
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Arm", meta = (ClampMin = "0"))
  double ArmRotationStiffness = 50.0;

  // End-effector rotation damping [N m s/rad].
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Arm", meta = (ClampMin = "0"))
  double ArmRotationDamping = 4.0;

  // The largest position error the arm acts on [m]; farther targets pull no harder.
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Arm", meta = (ClampMin = "0.001"))
  double ArmMaxPositionError = 0.05;

  // The largest rotation error the arm acts on [rad].
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Arm", meta = (ClampMin = "0.001"))
  double ArmMaxRotationError = 0.4;

  // Stiffness of the pull of every arm joint toward the spawn configuration, within the null space
  // of the end-effector task [N m/rad]. 0 disables it.
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Arm", meta = (ClampMin = "0"))
  double NullSpaceBiasGain = 2.0;

  // Damping of every arm joint's velocity within the null space of the end-effector task
  // [N m s/rad].
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Arm", meta = (ClampMin = "0"))
  double NullSpaceBiasDamping = 4.0;

  // Stiffness of every hand joint [N m/rad, or N/m for a prismatic joint].
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Hand", meta = (ClampMin = "0"))
  double HandJointStiffness = 1.0;

  // Damping of every hand joint [N m s/rad, or N s/m for a prismatic joint].
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Hand", meta = (ClampMin = "0"))
  double HandJointDamping = 0.0;

  // The largest effort of any hand joint [N m or N]; negative for no limit.
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Hand")
  double HandJointMaxEffort = 1.5;

  // The arm controller's parameters.
  superdex::robotics::ControllerBasicOscPd::Params MakeArmParams() const;

  /**
   * The arm's joint efforts of a joint-space PD bias toward a configuration, projected into the
   * null space of the end-effector task with the kinematic null space projector
   * N = I - J^+ J, J^+ = J^T (J J^T)^-1, so that they move the arm's joints without moving its end
   * effector:
   *
   *   tau = N (Gain (TargetQ - Q) - Damping QDot)
   *
   * J's rows may be in any units and frame: N projects onto J's null space either way.
   *
   * @param Jacobian The end effector's Jacobian in the arm's joints: 6 rows of Q.Num() columns,
   *     row-major.
   * @param Q, QDot, TargetQ The arm's joint positions [rad or m] and velocities, and the
   *     configuration to pull toward.
   * @return One effort per arm joint [N m or N], or none if the inputs' sizes disagree.
   */
  static TArray<double> ComputeNullSpaceBias(
      TConstArrayView<double> Jacobian,
      TConstArrayView<double> Q,
      TConstArrayView<double> QDot,
      TConstArrayView<double> TargetQ,
      double Gain,
      double Damping);

  // The hand controller's parameters, one entry per robot DOF (the joint-space controller acts on
  // all of them): the hand gains on HandDofs, and none on the others.
  superdex::robotics::ControllerBasicJscPd::Params MakeHandParams(
      int32 NumDofs,
      TConstArrayView<int32> HandDofs) const;

 protected:
  virtual bool Initialize(const USuperDexBotComponent* Component) override;
  virtual void OnBotSpawned(USuperDexBotComponent* Component) override;
  virtual void Shutdown(const USuperDexBotComponent* Component) override;
  virtual bool StepController(
      const USuperDexBotComponent* Component,
      const FMochiArticulationState& State,
      double MochiDeltaTime,
      mochi::Actor* ArticulatedActor,
      mochi::Scene* Scene) override;

  UPROPERTY(Transient)
  TObjectPtr<UTeleopNodeRobotComponent> TeleopNode = nullptr;

  // Set on the game thread, read on the simulation thread.
  std::atomic<bool> bInitialized = false;
  mochi::ActorHandle RobotActorHandle{};

  superdex::robotics::ControllerHandle ArmControllerHandle{};
  superdex::robotics::ControllerBasicOscPd* ArmController = nullptr;
  superdex::robotics::ControllerHandle HandControllerHandle{};
  superdex::robotics::ControllerBasicJscPd* HandController = nullptr;

  // The robot DOFs of the arm (the arm controller's) and of the hand (all others), ascending.
  TArray<int32> ArmDofs;
  TArray<int32> HandDofs;

  // The targets the robot holds: the end effector in the robot's root frame (as the TeleopNode's
  // arm pose target), and the hand's joint positions in HandDofs order. Set from the robot's own
  // state on the first step.
  mochi::TransformRT HeldArmTarget;
  TArray<mochi::real> HeldHandTargets;
  // The arm's joint positions the null space bias pulls toward, in ArmDofs order: those it spawned
  // in, taken with the held targets.
  TArray<double> ArmBiasTarget;
  bool bHasHeldTargets = false;
  bool bWarnedJacobianSize = false;
};
