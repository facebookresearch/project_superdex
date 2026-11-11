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

#include "SuperDexBotIKRetargeterTestTypes.h"
#if WITH_EDITOR && WITH_DEV_AUTOMATION_TESTS
#include <superdex_robotics/ik/bot_ik.h>
#include <superdex_robotics/utils/bot_utils.h>
#include <superdex_robotics/utils/file_utils.h>
#include "Engine/SkeletalMesh.h"
#include "Misc/AutomationTest.h"
#include "Misc/Guid.h"
#include "MochiConversions.h"
#include "ReferenceSkeleton.h"
#include "SuperDexBot.h"
#include "SuperDexIKSolverObject.h"
#include "SuperDexPoseSource.h"
#include "UObject/StrongObjectPtr.h"
#include "UObject/UnrealType.h"

struct FSuperDexBotIKRetargeterTestAccess {
  static superdex::robotics::BotIKConfig& Config(USuperDexBotIKRetargeter* Ret) {
    return Ret->NativeConfig;
  }
  static USuperDexIKSolverObject* Solver(USuperDexBotIKRetargeter* Ret) {
    return Ret->Solver;
  }
};

namespace {
using mochi::operator""_r;
using FGenericNativeIK = TStrongObjectPtr<USuperDexIKSolverObject>;

superdex::robotics::BotPrefab GenericModel(bool FreeRoot, bool Prismatic, mochi::real Scale) {
  using namespace superdex::robotics;
  BotPrefab Model;
  Model.name = "generic_mixed_tree";
  Model.scale = Scale;
  for (int I = 0; I < 6; ++I) {
    BotLinkPrefab Link;
    Link.name = TCHAR_TO_UTF8(*FString::Printf(TEXT("generic_link_%d"), I));
    Link.parentLink = I - 1;
    Link.mass = 1_r;
    Link.centerOfMass = {};
    Link.momentOfInertia = {1_r, 0_r, 0_r, 1_r, 0_r, 1_r};
    if (I > 0) {
      Link.parentJointFromLink = mochi::TransformRT{
          mochi::Quaternion::RotationX(0.6_r), mochi::Real3{0.01_r, -0.02_r, 0.03_r}};
    }
    Model.links.push_back(Link);
    BotJointPrefab Joint;
    Joint.name = TCHAR_TO_UTF8(*FString::Printf(TEXT("generic_joint_%d"), I));
    Joint.type = I == 0
        ? (FreeRoot ? mochi::ArticulatedJointType::Free : mochi::ArticulatedJointType::Hard)
        : I == 2              ? mochi::ArticulatedJointType::Revolute
        : I == 3 && Prismatic ? mochi::ArticulatedJointType::Prismatic
        : I == 4              ? mochi::ArticulatedJointType::Spherical
                              : mochi::ArticulatedJointType::Hard;
    Joint.axis = I == 3 ? mochi::Real3{1_r, 0_r, 0_r} : mochi::Real3{0_r, 0_r, 1_r};
    Joint.parentLinkFromJoint = I == 0
        ? mochi::TransformRT{}
        : mochi::TransformRT{
              mochi::Quaternion::RotationY(0.2_r), mochi::Real3{0.15_r, 0.02_r, -0.01_r}};
    Model.joints.push_back(Joint);
  }
  Model.defaultPose = Prismatic ? mochi::DynamicArray<mochi::real>{0_r, 0_r, 0_r, 0_r, 0_r}
                                : mochi::DynamicArray<mochi::real>{0_r, 0_r, 0_r, 0_r};
  mochi::Error Error;
  RebuildBotData(Model, Error);
  check(Error.IsOK());
  return Model;
}

USuperDexBot* GenericAsset(const superdex::robotics::BotPrefab& Model, const TCHAR* Name) {
  auto* Package = CreatePackage(*(TEXT("/Game/__GenericBotIK/") + FGuid::NewGuid().ToString()));
  auto* Asset = NewObject<USuperDexBot>(Package, FName(Name));
  mochi::Error Error;
  const auto Json = superdex::robotics::WriteToJsonString(Model, Error);
  check(Error.IsOK());
  FindFProperty<FStrProperty>(USuperDexBot::StaticClass(), TEXT("BotPrefabJson"))
      ->SetPropertyValue_InContainer(Asset, FString(UTF8_TO_TCHAR(Json.c_str())));
  auto* Mesh = NewObject<USkeletalMesh>(Asset);
  {
    FReferenceSkeletonModifier Modify(Mesh->GetRefSkeleton(), nullptr);
    FMochiToUnrealConverter Converter(kSuperDexSpace);
    for (int I = 0; I < static_cast<int>(Model.links.size()); ++I) {
      auto A = Model.joints[I].parentLinkFromJoint;
      auto B = Model.links[I].parentJointFromLink;
      A.SetTranslation(A.GetTranslation() * Model.scale);
      B.SetTranslation(B.GetTranslation() * Model.scale);
      const FName Bone(UTF8_TO_TCHAR(Model.links[I].name.c_str()));
      Modify.Add(
          FMeshBoneInfo(Bone, Bone.ToString(), I - 1),
          I == 0 ? FTransform::Identity : Converter.TransformToUnreal(A * B));
    }
  }
  FindFProperty<FObjectPropertyBase>(USuperDexBot::StaticClass(), TEXT("SkeletalMesh"))
      ->SetObjectPropertyValue_InContainer(Asset, Mesh);
  return Asset;
}

USuperDexBotIKConfig* GenericConfig(USuperDexBot* Input, USuperDexBot* Output) {
  auto* Config = NewObject<USuperDexBotIKConfig>();
  Config->InputBot = Input;
  Config->OutputBot = Output;
  Config->CreateParams.SolverParams.MaxIter = 200;
  Config->CreateParams.SolverParams.MaxElapsedTimeSeconds = 0;
  Config->CreateParams.SolverParams.AbsTol = 1e-6;
  Config->CreateParams.SolverParams.RelTol = 1e-8;
  for (int I : {2, 3, 4, 5}) {
    FSuperDexBotIKGoalConfig Goal;
    Goal.Name = FString::Printf(TEXT("generic_goal_%d"), I);
    Goal.OutputBotLink = UTF8_TO_TCHAR(Output->GetBotPrefab().links[I].name.c_str());
    Goal.InputBotLink = UTF8_TO_TCHAR(Input->GetBotPrefab().links[I].name.c_str());
    Goal.PositionStiffness = Goal.RotationStiffness = 1000;
    Config->Goals.Add(Goal);
  }
  return Config;
}

FSuperDexPose GenericNativePose(USuperDexBot* Input, const TArray<double>& Q) {
  superdex::robotics::BotIKCreateParams Params;
  Params.fixedBase = true;
  FGenericNativeIK FK(
      USuperDexIKSolverObject::CreateBotIKSolver(GetTransientPackage(), Input, Params));
  check(FK.Get());
  mochi::DynamicArray<mochi::TransformRT> Links;
  check(FK->CopyOutputBotFKAtJointConfiguration(Q, Links));
  const auto& Model = Input->GetBotPrefab();
  FMochiToUnrealConverter Converter(kSuperDexSpace);
  FSuperDexPose Pose;
  Pose.SourceName = TEXT("NativeBotFK");
  Pose.SkeletonName = FName(*Input->GetName());
  // This real native FK producer result deliberately keeps default false tracking flags.
  for (int I = 0; I < static_cast<int>(Links.size()); ++I) {
    const auto Local =
        I == 0 ? mochi::TransformRT{} : mochi::Invert(Links[Model.links[I].parentLink]) * Links[I];
    const auto Frame = Converter.TransformToUnreal(Local);
    Pose.BoneRotations.Add(Frame.GetRotation());
    Pose.BoneLocations.Add(Frame.GetTranslation());
  }
  FK->DestroyMochiResources();
  return Pose;
}

bool MatchGenericQ(
    FAutomationTestBase& Test,
    USuperDexBotIKRetargeter* Ret,
    const TArray<double>& Q) {
  TArray<double> Actual;
  bool Valid = false;
  if (!Test.TestTrue(
          TEXT("Generic output cache is present and valid"),
          Ret->GetCachedJointConfiguration(Actual, Valid) && Valid) ||
      !Test.TestEqual(TEXT("Generic output DOF width"), Actual.Num(), Q.Num())) {
    return false;
  }
  for (int I = 0; I < Q.Num(); ++I) {
    Test.TestTrue(
        TEXT("Generic projected native q matches output solve"),
        FMath::Abs(Actual[I] - Q[I]) < 3e-4);
  }
  return true;
}
} // namespace

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FGenericBotIKNativeChainTest,
    "Teleop.SuperDexBotIKRetargeter.GenericNativeTreeAndChaining",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FGenericBotIKNativeChainTest::RunTest(const FString&) {
  for (bool Free : {false, true}) {
    const auto Model = GenericModel(Free, true, 2_r);
    TStrongObjectPtr<USuperDexBot> Input(GenericAsset(Model, TEXT("GenericInput")));
    TStrongObjectPtr<USuperDexBot> Output(GenericAsset(Model, TEXT("GenericOutput")));
    TStrongObjectPtr<USuperDexBotIKConfig> Config(GenericConfig(Input.Get(), Output.Get()));
    TStrongObjectPtr<USuperDexBotIKRetargeter> Ret(NewObject<USuperDexBotIKRetargeter>());
    Ret->Config = Config.Get();
    if (!TestTrue(TEXT("Real mixed-family input initializes"), Ret->InitializeRetargeter()))
      return false;
    const TArray<double> Q{0.3, 0.025, 0.1, -0.2, 0.05};
    auto Pose = GenericNativePose(Input.Get(), Q);
    Pose.SkeletonRootToVrRoot =
        FTransform(FQuat(FVector::UpVector, 0.4), FVector(30, -20, 5), FVector(0.74));
    FSuperDexPose First;
    if (!TestTrue(
            TEXT("Native full pose with default nontracking flags retargets"),
            Ret->GetTransformedPose_Implementation(Pose, true, First)))
      return false;
    MatchGenericQ(*this, Ret.Get(), Q);
    auto Negated = Pose;
    for (int I = 1; I < Negated.BoneRotations.Num(); ++I) {
      const auto R = Negated.BoneRotations[I];
      Negated.BoneRotations[I] = FQuat(-R.X, -R.Y, -R.Z, -R.W);
    }
    FSuperDexPose NegatedOutput;
    TestTrue(
        TEXT("Full native antipodal quaternion pose accepts hard/rev/prism/spherical rotations"),
        Ret->GetTransformedPose_Implementation(Negated, true, NegatedOutput));
    MatchGenericQ(*this, Ret.Get(), Q);
    TestTrue(
        TEXT("Tracked/input root scale does not rescale prismatic q"),
        First.SkeletonRootToVrRoot.Equals(Pose.SkeletonRootToVrRoot, 1e-5));
    TestFalse(TEXT("Nontracking flags are not fabricated"), First.bIsTrackingEnabled);
    TStrongObjectPtr<USuperDexBot> ChainedOutput(GenericAsset(Model, TEXT("GenericChainedOutput")));
    TStrongObjectPtr<USuperDexBotIKConfig> ChainedConfig(
        GenericConfig(Output.Get(), ChainedOutput.Get()));
    TStrongObjectPtr<USuperDexBotIKRetargeter> Chained(NewObject<USuperDexBotIKRetargeter>());
    Chained->Config = ChainedConfig.Get();
    TestTrue(
        TEXT("Native-output to generic-input chain initializes"), Chained->InitializeRetargeter());
    FSuperDexPose Second;
    TestTrue(
        TEXT("Native scale/child-rest/prismatic output pose chains"),
        Chained->GetTransformedPose_Implementation(First, true, Second));
    MatchGenericQ(*this, Chained.Get(), Q);
    TestTrue(
        TEXT("Chain root placement remains separate and composed once"),
        Second.SkeletonRootToVrRoot.Equals(First.SkeletonRootToVrRoot, 1e-5));
    Pose.SkeletonName = FName(*Input->GetPathName());
    TestTrue(
        TEXT("Full asset identity is accepted"),
        Ret->GetTransformedPose_Implementation(Pose, true, First));
    for (double Angle : {3.13, -3.13}) {
      const auto NearPi = GenericNativePose(Input.Get(), {Angle, 0.025, Angle, 0.0, 0.0});
      auto Antipodal = NearPi;
      for (int I = 1; I < Antipodal.BoneRotations.Num(); ++I) {
        const auto R = Antipodal.BoneRotations[I];
        Antipodal.BoneRotations[I] = FQuat(-R.X, -R.Y, -R.Z, -R.W);
      }
      TestTrue(
          TEXT("Near-pi full pose remains projectable on either quaternion hemisphere"),
          Ret->GetTransformedPose_Implementation(Antipodal, true, First));
      for (int I = 1; I < First.BoneRotations.Num(); ++I) {
        TestTrue(
            TEXT("Near-pi solve preserves physical local frame rather than quaternion sign"),
            FTransform(First.BoneRotations[I], First.BoneLocations[I])
                .Equals(FTransform(NearPi.BoneRotations[I], NearPi.BoneLocations[I]), 0.05));
      }
    }
  }
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FGenericBotIKInvalidInputTest,
    "Teleop.SuperDexBotIKRetargeter.GenericIdentityAndInvalidInputs",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FGenericBotIKInvalidInputTest::RunTest(const FString&) {
  const auto Model = GenericModel(false, true, 2_r);
  TStrongObjectPtr<USuperDexBot> Input(GenericAsset(Model, TEXT("CollisionName")));
  TStrongObjectPtr<USuperDexBot> Other(GenericAsset(Model, TEXT("CollisionName")));
  TStrongObjectPtr<USuperDexBot> Output(GenericAsset(Model, TEXT("GenericOutput")));
  TStrongObjectPtr<USuperDexBotIKConfig> Config(GenericConfig(Input.Get(), Output.Get()));
  TStrongObjectPtr<USuperDexBotIKRetargeter> Ret(NewObject<USuperDexBotIKRetargeter>());
  Ret->Config = Config.Get();
  TestTrue(TEXT("Generic configuration initializes"), Ret->InitializeRetargeter());
  const auto Good = GenericNativePose(Input.Get(), {0.3, 0.025, 0.1, -0.2, 0.05});
  auto Reject = [&](FSuperDexPose Bad, const TCHAR* Message, bool SourceValid = true) {
    FSuperDexPose Recovery;
    TestTrue(
        TEXT("Valid native input resets diagnostic freshness before each rejection"),
        Ret->GetTransformedPose_Implementation(Good, true, Recovery));
    FSuperDexPose Out;
    Out.SourceName = TEXT("UnchangedOutput");
    AddExpectedError(Message, EAutomationExpectedErrorFlags::Contains, 1);
    TestFalse(
        TEXT("Invalid generic pose rejected"),
        Ret->GetTransformedPose_Implementation(Bad, SourceValid, Out));
    TestEqual(
        TEXT("Rejected input preserves output"), Out.SourceName, FName(TEXT("UnchangedOutput")));
  };
  auto Bad = Good;
  Bad.BoneLocations.Reset();
  Reject(Bad, TEXT("prismatic joints need input pose BoneLocations"));
  Bad = Good;
  Bad.BoneRotations.Pop();
  Reject(Bad, TEXT("bone count must match"));
  FSuperDexPose Out;
  for (const FName Name : {FName(TEXT("WrongSameWidth")), FName(*Other->GetPathName())}) {
    auto Renamed = Good;
    Renamed.SkeletonName = Name;
    TestTrue(
        TEXT("Any skeleton name with the input bot's bone count is accepted"),
        Ret->GetTransformedPose_Implementation(Renamed, true, Out));
  }
  TStrongObjectPtr<USuperDexBotIKTestPoseSource> Provider(
      NewObject<USuperDexBotIKTestPoseSource>());
  Provider->SetSkinnedAssetAndUpdate(Other->GetSkeletalMesh());
  Ret->SourcePoseableOverride = Provider.Get();
  FString Diagnostic;
  TestTrue(
      TEXT("A source with another mesh asset of the same layout is accepted"),
      Ret->InitializeRetargeter());
  TestTrue(
      TEXT("Generic recovers after rejected input"),
      Ret->GetTransformedPose_Implementation(Good, true, Out));
  TestFalse(
      TEXT("False provider validity rejects held native pose"),
      Ret->GetTransformedPose_Implementation(Good, false, Out));
  auto Tracking = Good;
  Tracking.bIsTrackingEnabled = true;
  Tracking.bIsTrackingConfident = false;
  TestFalse(
      TEXT("Enabled unconfident tracking rejects"),
      Ret->GetTransformedPose_Implementation(Tracking, true, Out));
  Config->CreateParams.bFixedBase = false;
  TestFalse(TEXT("Floating output limitation is explicit"), Ret->ValidateConfiguration(Diagnostic));
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FGenericBotIKRotationProjectionTest,
    "Teleop.SuperDexBotIKRetargeter.GenericRotationVectorProjection",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FGenericBotIKRotationProjectionTest::RunTest(const FString&) {
  const auto Model = GenericModel(false, false, 2_r);
  TStrongObjectPtr<USuperDexBot> Input(GenericAsset(Model, TEXT("GenericRotations")));
  TStrongObjectPtr<USuperDexBot> Output(GenericAsset(Model, TEXT("GenericOutput")));
  TStrongObjectPtr<USuperDexBotIKConfig> Config(GenericConfig(Input.Get(), Output.Get()));
  TStrongObjectPtr<USuperDexBotIKRetargeter> Ret(NewObject<USuperDexBotIKRetargeter>());
  Ret->Config = Config.Get();
  TestTrue(TEXT("Hard-root rotational tree initializes"), Ret->InitializeRetargeter());
  const TArray<double> Q{0.2, 0.1, -0.2, 0.05};
  auto Pose = GenericNativePose(Input.Get(), Q);
  Pose.BoneLocations.Reset();
  const auto A = Model.joints[2].parentLinkFromJoint.GetRotation();
  const auto B = Model.links[2].parentJointFromLink.GetRotation();
  // Q[0] about the revolute (Z) axis, then a swing about a perpendicular axis, which is discarded.
  Pose.BoneRotations[2] = FMochiToUnrealConverter(kSuperDexSpace)
                              .RotationToUnreal(
                                  A * mochi::Quaternion::RotationZ(static_cast<mochi::real>(Q[0])) *
                                  mochi::Quaternion::RotationX(0.6_r) * B);
  // The discarded swing is logged on the first such frame, then every 200th: 3 times in 401.
  AddExpectedMessagePlain(
      TEXT("the input pose doesn't fit the InputBot's joints"),
      ELogVerbosity::Warning,
      EAutomationExpectedMessageFlags::Contains,
      3);
  FSuperDexPose Out;
  TestTrue(
      TEXT("Rotation-only off-axis revolute projects to its twist about the axis"),
      Ret->GetTransformedPose_Implementation(Pose, true, Out));
  MatchGenericQ(*this, Ret.Get(), Q);
  for (int I = 1; I < 401; ++I) {
    Ret->GetTransformedPose_Implementation(Pose, true, Out);
  }
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FGenericBotIKGoalsAndMetadataTest,
    "Teleop.SuperDexBotIKRetargeter.GenericConstraintScaleAndMetadata",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FGenericBotIKGoalsAndMetadataTest::RunTest(const FString&) {
  const auto Model = GenericModel(true, true, 2_r);
  TStrongObjectPtr<USuperDexBot> Input(GenericAsset(Model, TEXT("ScaleInput")));
  TStrongObjectPtr<USuperDexBot> Output(GenericAsset(Model, TEXT("ScaleOutput")));
  TStrongObjectPtr<USuperDexBotIKConfig> Config(GenericConfig(Input.Get(), Output.Get()));
  FMochiToUnrealConverter ToUE(kSuperDexSpace);
  Config->OutputBaseFrameFromInputBaseFrame = ToUE.TransformToUnreal(
      mochi::TransformRT{
          mochi::Quaternion::RotationZ(0.2_r), mochi::Real3{0.01_r, 0.02_r, -0.03_r}});
  Config->Goals[0].BaseOrInputLinkFrameFromConstraintFrame =
      ToUE.TransformToUnreal(mochi::TransformRT{mochi::Real3{0.01_r, -0.02_r, 0.03_r}});
  Config->Goals[0].LinkFrameFromConstraintFrame = ToUE.TransformToUnreal(
      mochi::TransformRT{
          mochi::Quaternion::RotationY(0.1_r), mochi::Real3{0.007_r, -0.003_r, 0.011_r}});
  Config->Goals[1].InputBotLink.Reset();
  Config->Goals[1].BaseOrInputLinkFrameFromConstraintFrame =
      ToUE.TransformToUnreal(mochi::TransformRT{mochi::Real3{0.4_r, -0.3_r, 0.2_r}});
  Config->Goals[2].bEnabled = false;
  TStrongObjectPtr<USuperDexBotIKRetargeter> Ret(NewObject<USuperDexBotIKRetargeter>());
  Ret->Config = Config.Get();
  const TArray<double> Q{0.3, 0.025, 0.1, -0.2, 0.05};
  const auto Pose = GenericNativePose(Input.Get(), Q);
  FGenericNativeIK InputFK(
      USuperDexIKSolverObject::CreateBotIKSolver(
          GetTransientPackage(), Input.Get(), Config->CreateParams.ToNative()));
  if (!TestNotNull(TEXT("Independent physical input FK"), InputFK.Get()))
    return false;
  mochi::DynamicArray<mochi::TransformRT> NativeLinks;
  TestTrue(
      TEXT("Actual physical input FK generated"),
      InputFK->CopyOutputBotFKAtJointConfiguration(Q, NativeLinks));
  FSuperDexPose Out;
  for (double Scale : {0.5, 1.25, 2.0}) {
    Config->InputBotScale = Scale;
    TestTrue(TEXT("Scaled generic goals initialize"), Ret->InitializeRetargeter());
    const auto NativeConfig = Config->ToNative();
    FGenericNativeIK Direct(
        USuperDexIKSolverObject::CreateBotIKSolver(
            GetTransientPackage(), Output.Get(), NativeConfig));
    if (!TestNotNull(TEXT("Independent native output solver"), Direct.Get()))
      return false;
    for (const auto& Goal : NativeConfig.goals) {
      mochi::TransformRT Frame = Goal.baseOrInputLinkFrameFromConstraintFrame;
      if (!Goal.inputBotLink.empty()) {
        const int Index = superdex::robotics::FindLinkIndexByName(Model, Goal.inputBotLink);
        auto Link = NativeLinks[Index];
        Link.SetTranslation(Link.GetTranslation() * NativeConfig.inputBotScale);
        Frame.SetTranslation(Frame.GetTranslation() * NativeConfig.inputBotScale);
        Frame = NativeConfig.outputBaseFrameFromInputBaseFrame * Link * Frame;
      }
      TestTrue(
          TEXT("Expected bound/unbound scale rules preserve nonidentity output attachment"),
          Direct->SetBotGoal(
              superdex::robotics::FindLinkIndexByName(Model, Goal.outputBotLink),
              Goal.linkFrameFromConstraintFrame,
              Frame,
              Goal.enabled ? Goal.positionStiffness : 0_r,
              Goal.enabled ? Goal.rotationStiffness : 0_r));
    }
    bool Reached = false;
    TestTrue(
        TEXT("Independent scaled goal batch solves without assuming convergence"),
        Direct->SolveBotIK(Reached));
    TestTrue(
        TEXT("Generic scaled input goal route solves"),
        Ret->GetTransformedPose_Implementation(Pose, true, Out));
    TArray<double> Actual;
    bool Valid = false;
    Ret->GetCachedJointConfiguration(Actual, Valid);
    TestTrue(TEXT("Generic scaled result valid"), Valid);
    for (int I = 0; I < Actual.Num(); ++I) {
      TestTrue(
          TEXT("Generic direct goal path follows physical FK/offset scale rules"),
          FMath::Abs(Actual[I] - Direct->GetSolvedDofs()[I]) < 3e-4);
    }
    Direct->DestroyMochiResources();
  }
  FString Diagnostic;
  auto& Cached = const_cast<superdex::robotics::BotPrefab&>(Input->GetBotPrefab());
  const auto Axis = Cached.joints[2].axis;
  Cached.joints[2].axis = {};
  TestFalse(TEXT("Malformed native input axis rejects"), Ret->ValidateConfiguration(Diagnostic));
  Cached.joints[2].axis = Axis;
  Cached._dofIndices[0] = 3;
  TestFalse(TEXT("Wrong prepared DOF ordering rejects"), Ret->ValidateConfiguration(Diagnostic));
  Cached._dofIndices[0] = 2;
  const auto Name = Cached.links[2].name;
  Cached.links[2].name = Cached.links[1].name;
  TestFalse(TEXT("Colliding link/bone names reject"), Ret->ValidateConfiguration(Diagnostic));
  Cached.links[2].name = Name;
  const auto Type = Cached.joints[2].type;
  Cached.joints[2].type = mochi::ArticulatedJointType::Free;
  TestFalse(
      TEXT("Unsupported non-root free joint rejects"), Ret->ValidateConfiguration(Diagnostic));
  Cached.joints[2].type = Type;
  Cached.cycles.resize(1);
  TestTrue(
      TEXT("Cycle input validates (the input kinematics follow its tree)"),
      Ret->ValidateConfiguration(Diagnostic));
  Cached.cycles.clear();
  TestTrue(TEXT("Restored prepared input validates"), Ret->ValidateConfiguration(Diagnostic));
  InputFK->DestroyMochiResources();
  return true;
}

#endif
