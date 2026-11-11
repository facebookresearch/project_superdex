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

USTRUCT(BlueprintType)
struct SUPERDEXROBOTICS_API FSuperDexBotIKSolverParams {
  GENERATED_BODY()

  FSuperDexBotIKSolverParams();
  mochi::experimental::IKSolverParams ToNative() const;
  void FromNative(const mochi::experimental::IKSolverParams& Params);

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IK Solver")
  int32 MaxIter;

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

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IK Solver")
  int32 LineSearchMaxIter;

  /** Soft time budget in seconds; zero is unlimited. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IK Solver")
  double MaxElapsedTimeSeconds;
};

USTRUCT(BlueprintType)
struct SUPERDEXROBOTICS_API FSuperDexBotIKCreateParams {
  GENERATED_BODY()

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IK Creation")
  bool bFixedBase = true;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IK Creation")
  bool bExcludeCollisionShapes = true;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IK Creation")
  FSuperDexBotIKSolverParams SolverParams;

  superdex::robotics::BotIKCreateParams ToNative() const;
  void FromNative(const superdex::robotics::BotIKCreateParams& Params);
};

USTRUCT(BlueprintType)
struct SUPERDEXROBOTICS_API FSuperDexBotIKGoalConfig {
  GENERATED_BODY()

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IK Goal")
  FString Name;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IK Goal")
  FString OutputBotLink;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IK Goal")
  bool bEnabled = true;

  /** Constraint frame expressed in InputBotLink when bound, otherwise the output base frame.
   * Translation is in centimetres; scale is ignored by rigid conversion.
   */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IK Goal")
  FTransform BaseOrInputLinkFrameFromConstraintFrame = FTransform::Identity;

  /** Constraint frame expressed in the output bot link. Translation is in centimetres; scale is
   * ignored. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IK Goal")
  FTransform LinkFrameFromConstraintFrame = FTransform::Identity;

  /** Exact-case input-bot link binding; empty selects the output base frame. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IK Goal")
  FString InputBotLink;

  /** Native translational stiffness in N/m. Zero disables the position constraint. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IK Goal")
  double PositionStiffness = 100.0;

  /** Native rotational stiffness in N m/rad. Zero disables the rotation constraint. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IK Goal")
  double RotationStiffness = 100.0;

  superdex::robotics::BotIKGoalConfig ToNative() const;
  void FromNative(const superdex::robotics::BotIKGoalConfig& Goal);
};

USTRUCT(BlueprintType)
struct SUPERDEXROBOTICS_API FSuperDexBotIKJointCouplingConfig {
  GENERATED_BODY()

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IK Joint Coupling")
  FString JointA;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IK Joint Coupling")
  FString JointB;

  /** Soft bilateral relationship: qB = Ratio * qA, measured from authored joint zero. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IK Joint Coupling")
  double Ratio = 1.0;

  /** Native N*m/rad for revolute pairs, N/m for prismatic pairs. Zero disables the penalty. */
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      Category = "IK Joint Coupling",
      meta = (ClampMin = "0"))
  double Stiffness = 10.0;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IK Joint Coupling")
  bool bEnabled = true;

  superdex::robotics::BotIKJointCouplingConfig ToNative() const;
  void FromNative(const superdex::robotics::BotIKJointCouplingConfig& Coupling);
};

UCLASS(BlueprintType, DisplayName = "SuperDex Bot IK Config")
class SUPERDEXROBOTICS_API USuperDexBotIKConfig : public UObject {
  GENERATED_BODY()

 public:
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Output Bot")
  TObjectPtr<USuperDexBot> OutputBot;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Input Bot")
  TObjectPtr<USuperDexBot> InputBot;

  /** Joint-only output seed in native radians/metres; empty uses the bot's default pose. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Output Bot")
  TArray<double> OutputBotConfiguration;

  /** Authored input-bot coordinates in native radians/metres; live retargeting may override. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Input Bot")
  TArray<double> InputBotConfiguration;

  /** Rigid input-bot root placement in the IK base frame. Translation is in centimetres.
   * Scale must be identity; use InputBotScale for retarget geometry scaling.
   */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Input Bot")
  FTransform OutputBaseFrameFromInputBaseFrame = FTransform::Identity;

  /** Positive uniform retarget scale, independent of either bot prefab's scale. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Input Bot")
  double InputBotScale = 1.0;

  /** Original document references, retained even when editor resolution fails. Never opened at
   * runtime. */
  UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Output Bot")
  FString OutputBotReference;

  UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Input Bot")
  FString InputBotReference;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex Bot IK")
  FSuperDexBotIKCreateParams CreateParams;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IK Goal")
  TArray<FSuperDexBotIKGoalConfig> Goals;

  /** Couplings act on the solved bot, not on the reference hand. Applied at solver creation. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex Bot IK")
  TArray<FSuperDexBotIKJointCouplingConfig> JointCouplings;

  superdex::robotics::BotIKConfig ToNative() const;
  void FromNative(const superdex::robotics::BotIKConfig& Config);
  bool Validate(FString& OutDiagnostic) const;

#if WITH_EDITOR
  virtual void PostInitProperties() override;
  virtual void GetAssetRegistryTags(FAssetRegistryTagsContext Context) const override;
#endif

#if WITH_EDITORONLY_DATA
  UPROPERTY(VisibleAnywhere, Instanced, Category = "Import Settings")
  TObjectPtr<UAssetImportData> AssetImportData;

  UPROPERTY(VisibleAnywhere, Category = "Import Settings")
  FString SourceOutputBotPath;

  UPROPERTY(VisibleAnywhere, Category = "Import Settings")
  FString SourceInputBotPath;
#endif
};
