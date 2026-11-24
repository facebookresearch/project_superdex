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

#include <superdex_robotics/ik/bot_ik_config.h>
#include "CoreMinimal.h"
#include "MochiTypes.h"
#include "UObject/Object.h"

#include "SuperDexBotIKConfig.generated.h"

class USuperDexBot;
class UAssetImportData;

/** Unreal copy of the native bot IK solver parameters (mochi::experimental::IKSolverParams). */
USTRUCT(BlueprintType)
struct SUPERDEXROBOTICS_API FSuperDexBotIKSolverParams {
  GENERATED_BODY()

  /** Initializes the parameters to those of a new native bot IK config. */
  FSuperDexBotIKSolverParams();
  /** Converts to the native solver parameters. */
  mochi::experimental::IKSolverParams ToNative() const;
  /** Sets these parameters from native ones. */
  void FromNative(const mochi::experimental::IKSolverParams& Params);

  /** Maximum solver iterations. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IK Solver")
  int32 MaxIter;

  /** Solver log verbosity. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IK Solver")
  EMochiVerbosityLevel Verbosity;

  /** Tolerance in native solver units. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IK Solver")
  double AbsTol;

  /** Tolerance in native solver units. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IK Solver")
  double RelTol;

  /** Position tolerance in metres. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IK Solver")
  double PositionErrorThreshold;

  /** Rotation tolerance in radians. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IK Solver")
  double RotationErrorThreshold;

  /** Maximum line-search iterations. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IK Solver")
  int32 LineSearchMaxIter;

  /** Soft time budget in seconds; zero is unlimited. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IK Solver")
  double MaxElapsedTimeSeconds;
};

/** Options for creating a bot IK solver (superdex::robotics::BotIKCreateParams). */
USTRUCT(BlueprintType)
struct SUPERDEXROBOTICS_API FSuperDexBotIKCreateParams {
  GENERATED_BODY()

  FSuperDexBotIKCreateParams();

  /** Weld the bot root to its world-from-root transform; otherwise keep the root joint type. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IK Creation")
  bool bFixedBase;

  /** Omit collision geometry and skin from the solver's copy of the bot. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IK Creation")
  bool bExcludeCollisionShapes;

  /** Numerical termination and goal-reached thresholds of the solver. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IK Creation")
  FSuperDexBotIKSolverParams SolverParams;

  /** Converts to the native creation options. */
  superdex::robotics::BotIKCreateParams ToNative() const;
  /** Sets these options from native ones. */
  void FromNative(const superdex::robotics::BotIKCreateParams& Params);
};

/**
 * One bot IK goal: a constraint frame on an output bot link, pulled toward a frame on an input bot
 * link or on the output base.
 */
USTRUCT(BlueprintType)
struct SUPERDEXROBOTICS_API FSuperDexBotIKGoalConfig {
  GENERATED_BODY()

  FSuperDexBotIKGoalConfig();

  /** Goal name, used to address the goal at runtime (case-sensitive). */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IK Goal")
  FString Name;

  /** Output bot link the goal acts on. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IK Goal")
  FString OutputBotLink;

  /** Whether the goal is active. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IK Goal")
  bool bEnabled;

  /**
   * Constraint frame expressed in InputBotLink when bound, otherwise the output base frame.
   * Translation is in centimetres; scale is ignored by rigid conversion.
   */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IK Goal")
  FTransform BaseOrInputLinkFrameFromConstraintFrame;

  /**
   * Constraint frame expressed in the output bot link. Translation is in centimetres; scale is
   * ignored.
   */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IK Goal")
  FTransform LinkFrameFromConstraintFrame;

  /** Exact-case input-bot link binding; empty selects the output base frame. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IK Goal")
  FString InputBotLink;

  /** Native translational stiffness in N/m. Zero disables the position constraint. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IK Goal")
  double PositionStiffness;

  /** Native rotational stiffness in N m/rad. Zero disables the rotation constraint. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IK Goal")
  double RotationStiffness;

  /** Converts to the native goal, converting its frames from Unreal to SuperDex space. */
  superdex::robotics::BotIKGoalConfig ToNative() const;
  /** Sets this goal from a native one. */
  void FromNative(const superdex::robotics::BotIKGoalConfig& Goal);
};

/** Soft coupling between two joints of the solved bot (a penalty, not a mechanical linkage). */
USTRUCT(BlueprintType)
struct SUPERDEXROBOTICS_API FSuperDexBotIKJointCouplingConfig {
  GENERATED_BODY()

  FSuperDexBotIKJointCouplingConfig();

  /** First joint of the pair (exact-case name). */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IK Joint Coupling")
  FString JointA;

  /** Second joint of the pair (exact-case name). */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IK Joint Coupling")
  FString JointB;

  /** Soft bilateral relationship: qB = Ratio * qA, measured from authored joint zero. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IK Joint Coupling")
  double Ratio;

  /** Native N*m/rad for revolute pairs, N/m for prismatic pairs. Zero disables the penalty. */
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      Category = "IK Joint Coupling",
      meta = (ClampMin = "0"))
  double Stiffness;

  /** Whether the coupling is applied. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IK Joint Coupling")
  bool bEnabled;

  /** Converts to the native coupling. */
  superdex::robotics::BotIKJointCouplingConfig ToNative() const;
  /** Sets this coupling from a native one. */
  void FromNative(const superdex::robotics::BotIKJointCouplingConfig& Coupling);
};

/**
 * Bot IK configuration asset: the bot to solve, an optional input bot that goals can bind to, the
 * goals, joint couplings and solver options.
 */
UCLASS(BlueprintType, DisplayName = "SuperDex Bot IK Config")
class SUPERDEXROBOTICS_API USuperDexBotIKConfig : public UObject {
  GENERATED_BODY()

 public:
  USuperDexBotIKConfig();

  /** The bot the IK solves. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Output Bot")
  TObjectPtr<USuperDexBot> OutputBot;

  /** Optional input bot whose links goals can bind to (FSuperDexBotIKGoalConfig::InputBotLink). */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Input Bot")
  TObjectPtr<USuperDexBot> InputBot;

  /** Joint-only output seed in native radians/metres; empty uses the bot's default pose. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Output Bot")
  TArray<double> OutputBotConfiguration;

  /** Authored input-bot coordinates in native radians/metres; live retargeting may override. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Input Bot")
  TArray<double> InputBotConfiguration;

  /**
   * Rigid input-bot root placement in the IK base frame. Translation is in centimetres.
   * Scale must be identity; use InputBotScale for retarget geometry scaling.
   */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Input Bot")
  FTransform OutputBaseFrameFromInputBaseFrame;

  /** Positive uniform retarget scale, independent of either bot prefab's scale. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Input Bot")
  double InputBotScale;

  /**
   * Original document references, retained even when editor resolution fails. Never opened at
   * runtime.
   */
  UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Output Bot")
  FString OutputBotReference;

  /** The input bot reference from the original document. */
  UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Input Bot")
  FString InputBotReference;

  /** Solver creation options. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex Bot IK")
  FSuperDexBotIKCreateParams CreateParams;

  /** The IK goals. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IK Goal")
  TArray<FSuperDexBotIKGoalConfig> Goals;

  /** Couplings act on the solved bot, not on the reference hand. Applied at solver creation. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex Bot IK")
  TArray<FSuperDexBotIKJointCouplingConfig> JointCouplings;

  /**
   * Converts to the native IK config. Bot references are the assigned bots' paths, falling back to
   * OutputBotReference and InputBotReference.
   */
  superdex::robotics::BotIKConfig ToNative() const;
  /**
   * Sets this config from a native one. Bots are not resolved: the document's bot references are
   * stored in OutputBotReference and InputBotReference.
   */
  void FromNative(const superdex::robotics::BotIKConfig& Config);
  /**
   * Checks that the config's values are finite, representable in native precision and in range,
   * then validates the native config.
   *
   * @param OutDiagnostic Set to the reason the config is invalid, or the native validation
   * diagnostic.
   * @return True if the config is valid.
   */
  bool Validate(FString& OutDiagnostic) const;

#if WITH_EDITOR
  virtual void PostInitProperties() override;
  virtual void GetAssetRegistryTags(FAssetRegistryTagsContext Context) const override;
#endif

#if WITH_EDITORONLY_DATA
  /** Source file tracking for reimport. */
  UPROPERTY(VisibleAnywhere, Instanced, Category = "Import Settings")
  TObjectPtr<UAssetImportData> AssetImportData;

  /**
   * Output bot source path stored at the last import; reimport keeps OutputBot while it still
   * matches.
   */
  UPROPERTY(VisibleAnywhere, Category = "Import Settings")
  FString SourceOutputBotPath;

  /**
   * Input bot source path stored at the last import; reimport keeps InputBot while it still
   * matches.
   */
  UPROPERTY(VisibleAnywhere, Category = "Import Settings")
  FString SourceInputBotPath;
#endif
};
