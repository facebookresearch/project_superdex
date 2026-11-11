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

#include "SuperDexBotIKComponent.h"

#include <mochi_core/utils/math_utils.h>
#include <limits>
#include "MochiConversions.h"
#include "MochiPhysicsModule.h"
#include "SuperDexBot.h"
#include "SuperDexIKSolverObject.h"

namespace {

bool ValidRigidTransform(const FTransform& Transform) {
  return !Transform.GetTranslation().ContainsNaN() && !Transform.GetRotation().ContainsNaN() &&
      Transform.GetRotation().IsNormalized();
}

FTransform RigidTransform(const FTransform& Transform) {
  return FTransform(Transform.GetRotation(), Transform.GetTranslation());
}

bool ValidStiffness(double Stiffness) {
  return FMath::IsFinite(Stiffness) && Stiffness >= 0 &&
      Stiffness <= static_cast<double>(std::numeric_limits<mochi::real>::max()) &&
      (Stiffness == 0 || static_cast<mochi::real>(Stiffness) != 0);
}

} // namespace

USuperDexBotIKComponent::USuperDexBotIKComponent() {
  PrimaryComponentTick.bCanEverTick = false;
}

void USuperDexBotIKComponent::BeginPlay() {
  Super::BeginPlay();
  InitializeSolver();
}

bool USuperDexBotIKComponent::InitializeSolver() {
  if (!IsInGameThread()) {
    return false;
  }
  ReleaseSolver();
  FString Diagnostic;
  if (!Config || !Config->OutputBot || !Config->Validate(Diagnostic)) {
    UE_LOG(LogMochiPhysics, Warning, TEXT("Bot IK initialization failed: %s"), *Diagnostic);
    return false;
  }
  const auto& Prefab = Config->OutputBot->GetBotPrefab();
  TMap<FString, int32> Links;
  for (int32 Index = 0; Index < static_cast<int32>(Prefab.links.size()); ++Index) {
    FString Name = UTF8_TO_TCHAR(Prefab.links[Index].name.c_str());
    if (Name.IsEmpty() || Links.Contains(Name)) {
      UE_LOG(
          LogMochiPhysics,
          Warning,
          TEXT("Bot IK initialization failed: %s link name '%s' at index %d."),
          Name.IsEmpty() ? TEXT("empty") : TEXT("duplicate"),
          *Name,
          Index);
      return false;
    }
    Links.Add(Name, Index);
  }
  TArray<int32> Indices;
  for (const auto& Goal : Config->Goals) {
    if (!Goal.InputBotLink.IsEmpty()) {
      UE_LOG(
          LogMochiPhysics,
          Warning,
          TEXT(
              "Bot IK manual component cannot evaluate input-bot binding '%s' on goal '%s'. Use the BotIK retargeter for bound configs; goal/attachment frames are not flattened."),
          *Goal.InputBotLink,
          *Goal.Name);
      return false;
    }
    const int32* Index = Links.Find(Goal.OutputBotLink);
    if (!Index) {
      UE_LOG(
          LogMochiPhysics, Warning, TEXT("Bot IK goal link is missing: %s"), *Goal.OutputBotLink);
      return false;
    }
    Indices.Add(*Index);
  }
  const auto NativeConfig = Config->ToNative();
  mochi::DynamicString CouplingDiagnostic;
  if (!superdex::robotics::ValidateIKJointCouplings(
          Prefab, NativeConfig.jointCouplings, CouplingDiagnostic)) {
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT("Bot IK joint couplings invalid: %hs"),
        CouplingDiagnostic.c_str());
    return false;
  }
  Solver = USuperDexIKSolverObject::CreateBotIKSolver(this, Config->OutputBot, NativeConfig);
  if (!Solver) {
    return false;
  }
  AuthoredGoals = Config->Goals;
  RuntimeGoals = AuthoredGoals;
  GoalLinks = MoveTemp(Indices);
  GoalSpaces.Init(ESuperDexBotIKGoalSpace::Component, RuntimeGoals.Num());
  UpdateResults();
  return true;
}

int32 USuperDexBotIKComponent::FindGoal(const FString& Name) const {
  if (!IsInGameThread() || !Solver) {
    return INDEX_NONE;
  }
  return RuntimeGoals.IndexOfByPredicate(
      [&Name](const auto& Goal) { return Goal.Name.Equals(Name, ESearchCase::CaseSensitive); });
}

bool USuperDexBotIKComponent::ResetGoalsToConfig() {
  if (!IsInGameThread() || !Solver) {
    return false;
  }
  RuntimeGoals = AuthoredGoals;
  GoalSpaces.Init(ESuperDexBotIKGoalSpace::Component, RuntimeGoals.Num());
  return true;
}

bool USuperDexBotIKComponent::SetGoalTransform(
    const FString& Name,
    const FTransform& Transform,
    ESuperDexBotIKGoalSpace Space) {
  const int32 Index = FindGoal(Name);
  if (Index == INDEX_NONE || !ValidRigidTransform(Transform) ||
      (Space != ESuperDexBotIKGoalSpace::Component && Space != ESuperDexBotIKGoalSpace::World)) {
    return false;
  }
  FUnrealToMochiConverter Converter(kSuperDexSpace);
  if (!mochi::IsFinite(Converter.TransformToMochi_RT(Transform))) {
    return false;
  }
  RuntimeGoals[Index].BaseOrInputLinkFrameFromConstraintFrame = RigidTransform(Transform);
  GoalSpaces[Index] = Space;
  return true;
}

bool USuperDexBotIKComponent::SetGoalEnabled(const FString& Name, bool bEnabled) {
  const int32 Index = FindGoal(Name);
  if (Index == INDEX_NONE) {
    return false;
  }
  RuntimeGoals[Index].bEnabled = bEnabled;
  return true;
}

bool USuperDexBotIKComponent::SetGoalStiffness(
    const FString& Name,
    double Position,
    double Rotation) {
  const int32 Index = FindGoal(Name);
  if (Index == INDEX_NONE || !ValidStiffness(Position) || !ValidStiffness(Rotation)) {
    return false;
  }
  RuntimeGoals[Index].PositionStiffness = Position;
  RuntimeGoals[Index].RotationStiffness = Rotation;
  return true;
}

bool USuperDexBotIKComponent::SolveIK(bool& OutGoalsReached) {
  OutGoalsReached = false;
  if (!IsInGameThread() || !Solver || !ValidRigidTransform(GetComponentTransform())) {
    return false;
  }
  bool bActive = false;
  const FTransform WorldFromComponent = RigidTransform(GetComponentTransform());
  FUnrealToMochiConverter Converter(kSuperDexSpace);
  for (int32 Index = 0; Index < RuntimeGoals.Num(); ++Index) {
    const auto& Goal = RuntimeGoals[Index];
    FTransform LocalGoal = RigidTransform(Goal.BaseOrInputLinkFrameFromConstraintFrame);
    if (GoalSpaces[Index] == ESuperDexBotIKGoalSpace::World) {
      LocalGoal = LocalGoal.GetRelativeTransform(WorldFromComponent);
    }
    const mochi::TransformRT NativeTransform = Converter.TransformToMochi_RT(LocalGoal);
    if (!mochi::IsFinite(NativeTransform)) {
      return false;
    }
    const auto Position = static_cast<mochi::real>(Goal.bEnabled ? Goal.PositionStiffness : 0.0);
    const auto Rotation = static_cast<mochi::real>(Goal.bEnabled ? Goal.RotationStiffness : 0.0);
    const auto AttachmentFrame = Goal.ToNative().linkFrameFromConstraintFrame;
    if (!Solver->SetBotGoal(
            GoalLinks[Index], AttachmentFrame, NativeTransform, Position, Rotation)) {
      return false;
    }
    bActive |= Position > 0 || Rotation > 0;
  }
  if (!bActive || !Solver->SolveBotIK(OutGoalsReached)) {
    return false;
  }
  UpdateResults();
  return true;
}

void USuperDexBotIKComponent::UpdateResults() {
  JointPositions = Solver->GetSolvedDofs();
  const FTransform WorldFromComponent = RigidTransform(GetComponentTransform());
  WorldFromLinks = Solver->GetSolvedLinkToWorldTransforms();
  for (auto& Transform : WorldFromLinks) {
    Transform = Transform * WorldFromComponent;
  }
}

TArray<double> USuperDexBotIKComponent::GetSolvedJointPositions() const {
  return IsInGameThread() ? JointPositions : TArray<double>{};
}

TArray<FTransform> USuperDexBotIKComponent::GetSolvedLinkTransforms() const {
  return IsInGameThread() ? WorldFromLinks : TArray<FTransform>{};
}

void USuperDexBotIKComponent::ReleaseSolver() {
  if (Solver) {
    Solver->DestroyMochiResources();
    Solver = nullptr;
  }
  RuntimeGoals.Reset();
  AuthoredGoals.Reset();
  GoalSpaces.Reset();
  GoalLinks.Reset();
  JointPositions.Reset();
  WorldFromLinks.Reset();
}

void USuperDexBotIKComponent::EndPlay(const EEndPlayReason::Type EndPlayReason) {
  check(IsInGameThread());
  ReleaseSolver();
  Super::EndPlay(EndPlayReason);
}

void USuperDexBotIKComponent::BeginDestroy() {
  check(IsInGameThread());
  ReleaseSolver();
  Super::BeginDestroy();
}
