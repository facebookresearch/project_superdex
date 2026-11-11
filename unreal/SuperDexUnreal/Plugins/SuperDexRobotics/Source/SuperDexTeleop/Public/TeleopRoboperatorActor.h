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

#include "CoreMinimal.h"
#include "Retargeters/SuperDexClutchableInterface.h"
#include "SuperDexPoseableActor.h"

#include "TeleopRoboperatorActor.generated.h"

class USuperDexPoseableComponent;
class USuperDexBotComponent;
class UTeleopNodeOperatorComponent;
class UTeleopNodeRobotComponent;

/**
 * Abstract base actor for roboperator blueprints. Pairs an operator-side and robot-side teleop node
 * and implements clutch toggling for all attached ClutchSkeletonRootToVrRoot retargeters.
 */
UCLASS(Abstract)
class SUPERDEXTELEOP_API ATeleopRoboperatorActor : public ASuperDexPoseableActor,
                                                   public ISuperDexClutchableInterface {
  GENERATED_BODY()

 public:
  // Operator-side teleop communication node
  UPROPERTY(EditDefaultsOnly, BlueprintReadOnly, Category = "Teleop")
  TObjectPtr<UTeleopNodeOperatorComponent> TeleopNodeOperator;

  // Robot-side teleop communication node
  UPROPERTY(EditDefaultsOnly, BlueprintReadOnly, Category = "Teleop")
  TObjectPtr<UTeleopNodeRobotComponent> TeleopNodeRobot;

 public:
  // Create each bot when the tracked hand driving it is first seen, at that hand's pose and size,
  // instead of at this actor on Begin Play. Applies to every SuperDexBotComponent on the actor; a
  // bot's tracked hand is its controller's source poseable (the first poseable attached to it).
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Teleop|Bot Spawn")
  bool bSpawnBotsAtTrackedHand = false;

  // Scale each bot to the tracked hand when it spawns (BotPrefab::scale). Off: nominal size.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      Category = "Teleop|Bot Spawn",
      meta = (EditCondition = "bSpawnBotsAtTrackedHand"))
  bool bScaleBotsToTrackedHand = true;

  // Seconds the hand must be tracked with a steady scale before its bot spawns. The runtime refines
  // its hand-size estimate during the first moments of tracking.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      Category = "Teleop|Bot Spawn",
      meta = (EditCondition = "bSpawnBotsAtTrackedHand", ClampMin = "0"))
  float SpawnSettleSeconds = 0.5f;

  // Largest relative change of the tracked hand scale that still counts as steady.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      Category = "Teleop|Bot Spawn",
      meta = (EditCondition = "bSpawnBotsAtTrackedHand", ClampMin = "0"))
  float SpawnScaleTolerance = 0.01f;

  // Spawn with the current scale once the hand has been tracked this long, steady or not.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      Category = "Teleop|Bot Spawn",
      meta = (EditCondition = "bSpawnBotsAtTrackedHand", ClampMin = "0"))
  float SpawnMaxWaitSeconds = 3.f;

 public:
  ATeleopRoboperatorActor();

  /** ISuperDexClutchableInterface implementation */
  void SetClutchEnabled_Implementation(bool bClutchEnabled) override;

  virtual void PostInitializeComponents() override;
  virtual void Tick(float DeltaSeconds) override;

  // How long a pending bot spawn has watched its tracked hand's scale settle.
  struct FSpawnSettle {
    double TrackedSeconds = 0.0;
    double SteadySeconds = 0.0;
    double ReferenceScale = 0.0;
  };

  // Advances Settle by a tick of DeltaSeconds in which the hand was tracked with scale HandScale.
  // Returns the scale to spawn the bot at -- HandScale once it has been steady for SettleSeconds or
  // the hand has been tracked for MaxWaitSeconds, or 1 if HandScale is still not positive and
  // finite then -- or 0 to keep waiting.
  static double AdvanceSpawnSettle(
      FSpawnSettle& Settle,
      double HandScale,
      double DeltaSeconds,
      double SettleSeconds,
      double ScaleTolerance,
      double MaxWaitSeconds);

 private:
  struct FPendingBotSpawn {
    TWeakObjectPtr<USuperDexBotComponent> Bot;
    TWeakObjectPtr<USuperDexPoseableComponent> Source;
    FSpawnSettle Settle;
    // Whether a failed spawn has been reported; failed spawns are retried every tick.
    bool bReportedFailure = false;
  };

  // Advances one pending spawn; returns true once it is resolved (spawned, or handed back to the
  // bot's own auto-spawn).
  bool UpdatePendingBotSpawn(FPendingBotSpawn& Pending, float DeltaSeconds);

  TArray<FPendingBotSpawn> PendingBotSpawns;
};
