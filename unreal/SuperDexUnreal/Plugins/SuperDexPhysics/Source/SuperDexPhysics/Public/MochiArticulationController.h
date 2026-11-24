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

#include "MochiArticulationUtil.h"

#include "MochiArticulationController.generated.h"

class UMochiArticulationComponent;

/** Pose targets and joint efforts supplied by an articulation controller. */
USTRUCT(BlueprintType)
struct SUPERDEXPHYSICS_API FMochiArticulationControllerCommands {
  GENERATED_BODY()
  /**
   * Desired transform target of the articulation base in world space. Only relevant for
   * free-base articulations and only has effect if BasePositionGains or BaseRotationGains
   * have non-zero stiffness.
   */
  FTransform BaseTransformTarget;

  /**
   * Desired joint position targets for the built-in PD pose controller [rad or m].
   * Setting a target only has effect if the corresponding JointControlGains stiffness > 0.
   */
  TArray<mochi::real> JointPositionTargets;

  /**
   * Joint efforts to be commanded [Nm or N]. These stack on the effort applied by the
   * built-in PD pose controller if it is being used.
   */
  TArray<mochi::real> JointEfforts;
};

/** Base class for articulation initialization, game-thread updates and native control callbacks. */
UCLASS(Abstract, Blueprintable, BlueprintType, DefaultToInstanced)
class SUPERDEXPHYSICS_API UMochiArticulationController : public UObject {
  GENERATED_BODY()

 protected:
  friend class UMochiArticulationComponent;

  /**
   * Initialize the controller. Called on the game thread in BeginPlay.
   * Return true on success, false to cancel controller deployment.
   *
   * @param Component Component supplied to the controller or editor operation.
   * @return True to permit controller deployment, false to cancel deployment.
   */
  UFUNCTION(BlueprintNativeEvent, BlueprintCallable, Category = "SuperDex Physics|Controller")
  bool Initialize(const UMochiArticulationComponent* Component);
  virtual bool Initialize_Implementation(const UMochiArticulationComponent* Component);

  /** Called when the articulation has been spawned. Game thread. */
  virtual void OnArticulationSpawned(UMochiArticulationComponent* Component);

  /**
   * Shutdown the controller. Called on the game thread in EndPlay or when swapped.
   *
   * @param Component Component supplied to the controller or editor operation.
   */
  UFUNCTION(BlueprintNativeEvent, BlueprintCallable, Category = "SuperDex Physics|Controller")
  void Shutdown(const UMochiArticulationComponent* Component);
  virtual void Shutdown_Implementation(const UMochiArticulationComponent* Component);

  /**
   * Implement control behavior. Given the current state, set desired pose targets
   * and/or joint efforts. Called on the Mochi thread in a pre-step callback.
   * Return true to apply commands.
   */
  virtual bool Control(
      const UMochiArticulationComponent* Component,
      const FMochiArticulationState& State,
      double MochiDeltaTime,
      FMochiArticulationControllerCommands& OutCommands,
      mochi::Scene* MochiScene);

  /**
   * Tick the controller on the game thread. Called in TickComponent and is
   * thread safe with Control.
   *
   * @param Component Component supplied to the controller or editor operation.
   * @param GameDeltaTime Game-thread update interval.
   */
  UFUNCTION(BlueprintNativeEvent, BlueprintCallable, Category = "SuperDex Physics|Controller")
  void TickController(const UMochiArticulationComponent* Component, double GameDeltaTime);
  virtual void TickController_Implementation(
      const UMochiArticulationComponent* Component,
      double GameDeltaTime);
};
