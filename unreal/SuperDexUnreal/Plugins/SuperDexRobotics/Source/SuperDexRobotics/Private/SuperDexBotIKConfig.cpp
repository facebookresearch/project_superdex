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

#include "SuperDexBotIKConfig.h"

#include <limits>
#include "MochiConversions.h"
#include "SuperDexBot.h"
#if WITH_EDITOR
#include "EditorFramework/AssetImportData.h"
#include "UObject/AssetRegistryTagsContext.h"
#endif

namespace {

// Reject narrowing overflow and nonzero values that would silently become zero.
bool NativeRealRepresentable(double Value) {
  return FMath::IsFinite(Value) &&
      FMath::Abs(Value) <= static_cast<double>(std::numeric_limits<mochi::real>::max()) &&
      (Value == 0.0 || static_cast<mochi::real>(Value) != 0);
}

bool NativeTranslationRepresentable(double Centimetres) {
  const double Metres = Centimetres / 100.0;
  return NativeRealRepresentable(Metres) && (Centimetres == 0.0 || Metres != 0.0);
}

bool ValidRigidFrame(const FTransform& Frame) {
  const auto& Translation = Frame.GetTranslation();
  const auto& Rotation = Frame.GetRotation();
  return !Translation.ContainsNaN() && !Rotation.ContainsNaN() &&
      !Frame.GetScale3D().ContainsNaN() && Rotation.IsNormalized() &&
      NativeTranslationRepresentable(Translation.X) &&
      NativeTranslationRepresentable(Translation.Y) &&
      NativeTranslationRepresentable(Translation.Z) && NativeRealRepresentable(Rotation.X) &&
      NativeRealRepresentable(Rotation.Y) && NativeRealRepresentable(Rotation.Z) &&
      NativeRealRepresentable(Rotation.W);
}

mochi::TransformRT ToNativeRigidFrame(const FTransform& Frame) {
  // Normalize valid rotations before narrowing. Do not repair malformed input; Validate rejects it.
  FQuat Rotation = Frame.GetRotation();
  if (!Rotation.ContainsNaN() && Rotation.IsNormalized()) {
    Rotation.Normalize();
  }
  return FUnrealToMochiConverter(kSuperDexSpace)
      .TransformToMochi_RT(FTransform(Rotation, Frame.GetTranslation()));
}

FTransform FromNativeRigidFrame(const mochi::TransformRT& Frame) {
  auto Result = FMochiToUnrealConverter(kSuperDexSpace).TransformToUnreal(Frame);
  // Native validation allows 1e-4 norm roundoff; use unit UE quaternions at the API boundary.
  if (!Result.GetRotation().ContainsNaN() &&
      FMath::IsNearlyEqual(Result.GetRotation().SizeSquared(), 1.0, 2.1e-4)) {
    Result.NormalizeRotation();
  }
  return Result;
}

} // namespace

FSuperDexBotIKSolverParams::FSuperDexBotIKSolverParams() {
  FromNative(superdex::robotics::BotIKConfig{}.createParams.solverParams);
}

mochi::experimental::IKSolverParams FSuperDexBotIKSolverParams::ToNative() const {
  mochi::experimental::IKSolverParams Params;
  Params.maxIter = static_cast<int32>(MaxIter);
  Params.verbosity = static_cast<mochi::VerbosityLevel>(Verbosity);
  Params.absTol = static_cast<mochi::real>(AbsTol);
  Params.relTol = static_cast<mochi::real>(RelTol);
  Params.positionErrorThres = static_cast<mochi::real>(PositionErrorThreshold);
  Params.rotationErrorThres = static_cast<mochi::real>(RotationErrorThreshold);
  Params.lineSearchMaxIter = static_cast<int32>(LineSearchMaxIter);
  Params.maxElapsedTimeSeconds = static_cast<double>(MaxElapsedTimeSeconds);
  return Params;
}

void FSuperDexBotIKSolverParams::FromNative(const mochi::experimental::IKSolverParams& Params) {
  MaxIter = static_cast<int32>(Params.maxIter);
  Verbosity = static_cast<EMochiVerbosityLevel>(Params.verbosity);
  AbsTol = static_cast<double>(Params.absTol);
  RelTol = static_cast<double>(Params.relTol);
  PositionErrorThreshold = static_cast<double>(Params.positionErrorThres);
  RotationErrorThreshold = static_cast<double>(Params.rotationErrorThres);
  LineSearchMaxIter = static_cast<int32>(Params.lineSearchMaxIter);
  MaxElapsedTimeSeconds = static_cast<double>(Params.maxElapsedTimeSeconds);
}

superdex::robotics::BotIKCreateParams FSuperDexBotIKCreateParams::ToNative() const {
  superdex::robotics::BotIKCreateParams Params;
  Params.fixedBase = bFixedBase;
  Params.excludeCollisionShapes = bExcludeCollisionShapes;
  Params.solverParams = SolverParams.ToNative();
  return Params;
}

void FSuperDexBotIKCreateParams::FromNative(const superdex::robotics::BotIKCreateParams& Params) {
  bFixedBase = Params.fixedBase;
  bExcludeCollisionShapes = Params.excludeCollisionShapes;
  SolverParams.FromNative(Params.solverParams);
}

superdex::robotics::BotIKGoalConfig FSuperDexBotIKGoalConfig::ToNative() const {
  superdex::robotics::BotIKGoalConfig Goal;
  Goal.name = TCHAR_TO_UTF8(*Name);
  Goal.outputBotLink = TCHAR_TO_UTF8(*OutputBotLink);
  Goal.enabled = bEnabled;
  Goal.baseOrInputLinkFrameFromConstraintFrame =
      ToNativeRigidFrame(BaseOrInputLinkFrameFromConstraintFrame);
  Goal.linkFrameFromConstraintFrame = ToNativeRigidFrame(LinkFrameFromConstraintFrame);
  Goal.inputBotLink = TCHAR_TO_UTF8(*InputBotLink);
  Goal.positionStiffness = static_cast<mochi::real>(PositionStiffness);
  Goal.rotationStiffness = static_cast<mochi::real>(RotationStiffness);
  return Goal;
}

void FSuperDexBotIKGoalConfig::FromNative(const superdex::robotics::BotIKGoalConfig& Goal) {
  Name = UTF8_TO_TCHAR(Goal.name.c_str());
  OutputBotLink = UTF8_TO_TCHAR(Goal.outputBotLink.c_str());
  bEnabled = Goal.enabled;
  BaseOrInputLinkFrameFromConstraintFrame =
      FromNativeRigidFrame(Goal.baseOrInputLinkFrameFromConstraintFrame);
  LinkFrameFromConstraintFrame = FromNativeRigidFrame(Goal.linkFrameFromConstraintFrame);
  InputBotLink = UTF8_TO_TCHAR(Goal.inputBotLink.c_str());
  PositionStiffness = Goal.positionStiffness;
  RotationStiffness = Goal.rotationStiffness;
}

superdex::robotics::BotIKJointCouplingConfig FSuperDexBotIKJointCouplingConfig::ToNative() const {
  superdex::robotics::BotIKJointCouplingConfig Coupling;
  Coupling.jointA = TCHAR_TO_UTF8(*JointA);
  Coupling.jointB = TCHAR_TO_UTF8(*JointB);
  Coupling.ratio = static_cast<mochi::real>(Ratio);
  Coupling.stiffness = static_cast<mochi::real>(Stiffness);
  Coupling.enabled = bEnabled;
  return Coupling;
}

void FSuperDexBotIKJointCouplingConfig::FromNative(
    const superdex::robotics::BotIKJointCouplingConfig& Coupling) {
  JointA = UTF8_TO_TCHAR(Coupling.jointA.c_str());
  JointB = UTF8_TO_TCHAR(Coupling.jointB.c_str());
  Ratio = Coupling.ratio;
  Stiffness = Coupling.stiffness;
  bEnabled = Coupling.enabled;
}

superdex::robotics::BotIKConfig USuperDexBotIKConfig::ToNative() const {
  superdex::robotics::BotIKConfig Config;
  Config.outputBot = TCHAR_TO_UTF8(*(OutputBot ? OutputBot->GetPathName() : OutputBotReference));
  Config.inputBot = TCHAR_TO_UTF8(*(InputBot ? InputBot->GetPathName() : InputBotReference));
  Config.outputBaseFrameFromInputBaseFrame = ToNativeRigidFrame(OutputBaseFrameFromInputBaseFrame);
  Config.inputBotScale = static_cast<mochi::real>(InputBotScale);
  for (double Position : OutputBotConfiguration) {
    Config.outputBotConfiguration.push_back(static_cast<mochi::real>(Position));
  }
  for (double Position : InputBotConfiguration) {
    Config.inputBotConfiguration.push_back(static_cast<mochi::real>(Position));
  }
  Config.createParams = CreateParams.ToNative();
  for (const auto& Goal : Goals) {
    Config.goals.push_back(Goal.ToNative());
  }
  for (const auto& Coupling : JointCouplings) {
    Config.jointCouplings.push_back(Coupling.ToNative());
  }
  return Config;
}

void USuperDexBotIKConfig::FromNative(const superdex::robotics::BotIKConfig& Config) {
  // UObject resolution belongs to the editor importer, never to this runtime conversion.
  OutputBotReference = UTF8_TO_TCHAR(Config.outputBot.c_str());
  InputBotReference = UTF8_TO_TCHAR(Config.inputBot.c_str());
  OutputBaseFrameFromInputBaseFrame =
      FromNativeRigidFrame(Config.outputBaseFrameFromInputBaseFrame);
  InputBotScale = Config.inputBotScale;
  OutputBotConfiguration.SetNum(static_cast<int32>(Config.outputBotConfiguration.size()));
  for (int32 Index = 0; Index < OutputBotConfiguration.Num(); ++Index) {
    OutputBotConfiguration[Index] = Config.outputBotConfiguration[Index];
  }
  InputBotConfiguration.SetNum(static_cast<int32>(Config.inputBotConfiguration.size()));
  for (int32 Index = 0; Index < InputBotConfiguration.Num(); ++Index) {
    InputBotConfiguration[Index] = Config.inputBotConfiguration[Index];
  }
  CreateParams.FromNative(Config.createParams);
  Goals.SetNum(static_cast<int32>(Config.goals.size()));
  for (int32 Index = 0; Index < Goals.Num(); ++Index) {
    Goals[Index].FromNative(Config.goals[Index]);
  }
  JointCouplings.SetNum(static_cast<int32>(Config.jointCouplings.size()));
  for (int32 Index = 0; Index < JointCouplings.Num(); ++Index) {
    JointCouplings[Index].FromNative(Config.jointCouplings[Index]);
  }
}

bool USuperDexBotIKConfig::Validate(FString& OutDiagnostic) const {
  OutDiagnostic.Reset();
  if (!ValidRigidFrame(OutputBaseFrameFromInputBaseFrame) ||
      !OutputBaseFrameFromInputBaseFrame.GetScale3D().Equals(FVector::OneVector)) {
    OutDiagnostic = TEXT(
        "OutputBaseFrameFromInputBaseFrame must be a finite, native-representable rigid frame with unit rotation and identity scale; use InputBotScale for geometry scaling.");
    return false;
  }
  if (!NativeRealRepresentable(InputBotScale) || InputBotScale <= 0) {
    OutDiagnostic =
        TEXT("InputBotScale must be positive, finite and representable in native precision.");
    return false;
  }
  for (const auto* Positions : {&OutputBotConfiguration, &InputBotConfiguration}) {
    for (double Position : *Positions) {
      if (!NativeRealRepresentable(Position)) {
        OutDiagnostic = TEXT(
            "OutputBotConfiguration and InputBotConfiguration must contain finite, native-representable joint coordinates (radians/metres, without root DOFs).");
        return false;
      }
    }
  }
  const auto& Params = CreateParams.SolverParams;
  for (double Value :
       {Params.AbsTol,
        Params.RelTol,
        Params.PositionErrorThreshold,
        Params.RotationErrorThreshold}) {
    if (!NativeRealRepresentable(Value) || Value < 0) {
      OutDiagnostic = TEXT(
          "Solver tolerances must be nonnegative, finite and representable in native precision.");
      return false;
    }
  }
  if (!FMath::IsFinite(Params.MaxElapsedTimeSeconds) || Params.MaxElapsedTimeSeconds < 0) {
    OutDiagnostic = TEXT("Solver time budget must be nonnegative and finite.");
    return false;
  }
  for (int32 Index = 0; Index < Goals.Num(); ++Index) {
    const auto& Goal = Goals[Index];
    if (!ValidRigidFrame(Goal.BaseOrInputLinkFrameFromConstraintFrame) ||
        !ValidRigidFrame(Goal.LinkFrameFromConstraintFrame)) {
      OutDiagnostic = FString::Printf(
          TEXT(
              "Goals[%d] goal and attachment frames must have finite native-representable translation and normalized rotation."),
          Index);
      return false;
    }
    if (!NativeRealRepresentable(Goal.PositionStiffness) || Goal.PositionStiffness < 0 ||
        !NativeRealRepresentable(Goal.RotationStiffness) || Goal.RotationStiffness < 0) {
      OutDiagnostic = FString::Printf(
          TEXT(
              "Goals[%d] stiffness must be nonnegative, finite and representable in native precision; nonzero stiffness must not underflow."),
          Index);
      return false;
    }
  }
  for (const auto& Coupling : JointCouplings) {
    if (!NativeRealRepresentable(Coupling.Ratio) || Coupling.Ratio == 0 ||
        !NativeRealRepresentable(Coupling.Stiffness) || Coupling.Stiffness < 0) {
      OutDiagnostic = TEXT(
          "Joint coupling ratio/stiffness must be finite and representable in native precision; ratio is nonzero and stiffness nonnegative.");
      return false;
    }
  }
  mochi::DynamicString Diagnostic;
  const bool bValid = superdex::robotics::ValidateIKConfig(ToNative(), Diagnostic);
  OutDiagnostic = UTF8_TO_TCHAR(Diagnostic.c_str());
  return bValid;
}

#if WITH_EDITOR
void USuperDexBotIKConfig::PostInitProperties() {
  Super::PostInitProperties();
  if (!HasAnyFlags(RF_ClassDefaultObject) && !AssetImportData) {
    AssetImportData = NewObject<UAssetImportData>(this, TEXT("AssetImportData"));
  }
}

void USuperDexBotIKConfig::GetAssetRegistryTags(FAssetRegistryTagsContext Context) const {
  if (AssetImportData) {
    Context.AddTag(FAssetRegistryTag(
        SourceFileTagName(),
        AssetImportData->GetSourceData().ToJson(),
        FAssetRegistryTag::TT_Hidden));
  }
  Super::GetAssetRegistryTags(Context);
}
#endif
