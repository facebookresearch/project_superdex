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
#include "SuperDexPoseFunctionLibrary.h"
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
  UPackage* Package = CreatePackage(*(TEXT("/Game/__GenericBotIK/") + FGuid::NewGuid().ToString()));
  USuperDexBot* Asset = NewObject<USuperDexBot>(Package, FName(Name));
  mochi::Error Error;
  const mochi::DynamicString Json = superdex::robotics::WriteToJsonString(Model, Error);
  check(Error.IsOK());
  FindFProperty<FStrProperty>(USuperDexBot::StaticClass(), TEXT("BotPrefabJson"))
      ->SetPropertyValue_InContainer(Asset, FString(UTF8_TO_TCHAR(Json.c_str())));
  USkeletalMesh* Mesh = NewObject<USkeletalMesh>(Asset);
  {
    FReferenceSkeletonModifier Modify(Mesh->GetRefSkeleton(), nullptr);
    FMochiToUnrealConverter Converter(SuperDexSpace);
    for (int I = 0; I < static_cast<int>(Model.links.size()); ++I) {
      mochi::TransformRT A = Model.joints[I].parentLinkFromJoint;
      mochi::TransformRT B = Model.links[I].parentJointFromLink;
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
  USuperDexBotIKConfig* Config = NewObject<USuperDexBotIKConfig>();
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
  FSuperDexPose Pose;
  Pose.SourceName = TEXT("NativeBotFK");
  // This pose deliberately keeps default false tracking flags.
  check(USuperDexPoseFunctionLibrary::ComputeSuperDexPoseFromJointPositions(Input, Q, Pose));
  return Pose;
}

// A pose's links relative to their parents, at its joint positions.
TArray<FTransform> PoseLinkToParent(const FSuperDexPose& Pose) {
  TArray<FTransform> LinkToParent;
  check(
      USuperDexPoseFunctionLibrary::ComputeSuperDexPoseLinkToParentTransforms(Pose, LinkToParent));
  return LinkToParent;
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
    const superdex::robotics::BotPrefab Model = GenericModel(Free, true, 2_r);
    TStrongObjectPtr<USuperDexBot> Input(GenericAsset(Model, TEXT("GenericInput")));
    TStrongObjectPtr<USuperDexBot> Output(GenericAsset(Model, TEXT("GenericOutput")));
    TStrongObjectPtr<USuperDexBotIKConfig> Config(GenericConfig(Input.Get(), Output.Get()));
    TStrongObjectPtr<USuperDexBotIKRetargeter> Ret(NewObject<USuperDexBotIKRetargeter>());
    Ret->Config = Config.Get();
    if (!TestTrue(TEXT("Real mixed-family input initializes"), Ret->InitializeRetargeter())) {
      return false;
    }
    const TArray<double> Q{0.3, 0.025, 0.1, -0.2, 0.05};
    FSuperDexPose Pose = GenericNativePose(Input.Get(), Q);
    Pose.SkeletonRootToVrRoot =
        FTransform(FQuat(FVector::UpVector, 0.4), FVector(30, -20, 5), FVector(0.74));
    FSuperDexPose First;
    if (!TestTrue(
            TEXT("Native full pose with default nontracking flags retargets"),
            Ret->GetTransformedPose_Implementation(Pose, true, First))) {
      return false;
    }
    MatchGenericQ(*this, Ret.Get(), Q);
    TestTrue(TEXT("Output pose is on the output bot"), First.Bot.Get() == Output.Get());
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
    for (double Angle : {3.13, -3.13}) {
      const FSuperDexPose NearPi = GenericNativePose(Input.Get(), {Angle, 0.025, Angle, 0.0, 0.0});
      TestTrue(
          TEXT("Near-pi full pose remains projectable"),
          Ret->GetTransformedPose_Implementation(NearPi, true, First));
      const TArray<FTransform> Expected = PoseLinkToParent(NearPi);
      const TArray<FTransform> Actual = PoseLinkToParent(First);
      for (int I = 1; I < Actual.Num(); ++I) {
        TestTrue(
            TEXT("Near-pi solve preserves physical local frame"),
            Actual[I].Equals(Expected[I], 0.05));
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
  const superdex::robotics::BotPrefab Model = GenericModel(false, true, 2_r);
  TStrongObjectPtr<USuperDexBot> Input(GenericAsset(Model, TEXT("CollisionName")));
  TStrongObjectPtr<USuperDexBot> Other(GenericAsset(Model, TEXT("CollisionName")));
  TStrongObjectPtr<USuperDexBot> Output(GenericAsset(Model, TEXT("GenericOutput")));
  TStrongObjectPtr<USuperDexBotIKConfig> Config(GenericConfig(Input.Get(), Output.Get()));
  TStrongObjectPtr<USuperDexBotIKRetargeter> Ret(NewObject<USuperDexBotIKRetargeter>());
  Ret->Config = Config.Get();
  TestTrue(TEXT("Generic configuration initializes"), Ret->InitializeRetargeter());
  const FSuperDexPose Good = GenericNativePose(Input.Get(), {0.3, 0.025, 0.1, -0.2, 0.05});
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
  FSuperDexPose Bad = Good;
  Bad.DofPositions.Pop();
  Reject(Bad, TEXT("must hold joint positions of a bot"));
  Bad = Good;
  Bad.Bot = nullptr;
  Reject(Bad, TEXT("must hold joint positions of a bot"));
  FSuperDexPose Out;
  TestTrue(
      TEXT("A pose on another bot with the input bot's link count is accepted"),
      Ret->GetTransformedPose_Implementation(
          GenericNativePose(Other.Get(), {0.3, 0.025, 0.1, -0.2, 0.05}), true, Out));
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
  FSuperDexPose Tracking = Good;
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
  const superdex::robotics::BotPrefab Model = GenericModel(false, false, 2_r);
  TStrongObjectPtr<USuperDexBot> Input(GenericAsset(Model, TEXT("GenericRotations")));
  TStrongObjectPtr<USuperDexBot> Output(GenericAsset(Model, TEXT("GenericOutput")));
  TStrongObjectPtr<USuperDexBotIKConfig> Config(GenericConfig(Input.Get(), Output.Get()));
  TStrongObjectPtr<USuperDexBotIKRetargeter> Ret(NewObject<USuperDexBotIKRetargeter>());
  Ret->Config = Config.Get();
  TestTrue(TEXT("Hard-root rotational tree initializes"), Ret->InitializeRetargeter());
  // A bot of the same skeleton whose revolute link frame is swung about an axis perpendicular to
  // the revolute (Z) axis: its link 2 turns Q[0] about the axis, then swings, which is discarded.
  auto Swung = Model;
  Swung.links[2].parentJointFromLink =
      mochi::TransformRT{mochi::Quaternion::RotationX(0.6_r), mochi::Real3{}} *
      Swung.links[2].parentJointFromLink;
  mochi::Error Error;
  superdex::robotics::RebuildBotData(Swung, Error);
  TestTrue(TEXT("Swung bot rebuilds"), Error.IsOK());
  TStrongObjectPtr<USuperDexBot> SwungBot(GenericAsset(Swung, TEXT("GenericSwungRotations")));
  const TArray<double> Q{0.2, 0.1, -0.2, 0.05};
  const FSuperDexPose Pose = GenericNativePose(SwungBot.Get(), Q);
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
  const superdex::robotics::BotPrefab Model = GenericModel(true, true, 2_r);
  TStrongObjectPtr<USuperDexBot> Input(GenericAsset(Model, TEXT("ScaleInput")));
  TStrongObjectPtr<USuperDexBot> Output(GenericAsset(Model, TEXT("ScaleOutput")));
  TStrongObjectPtr<USuperDexBotIKConfig> Config(GenericConfig(Input.Get(), Output.Get()));
  FMochiToUnrealConverter ToUE(SuperDexSpace);
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
  const FSuperDexPose Pose = GenericNativePose(Input.Get(), Q);
  FGenericNativeIK InputFK(
      USuperDexIKSolverObject::CreateBotIKSolver(
          GetTransientPackage(), Input.Get(), Config->CreateParams.ToNative()));
  if (!TestNotNull(TEXT("Independent physical input FK"), InputFK.Get())) {
    return false;
  }
  mochi::DynamicArray<mochi::TransformRT> NativeLinks;
  TestTrue(
      TEXT("Actual physical input FK generated"),
      InputFK->CopyOutputBotFKAtJointConfiguration(Q, NativeLinks));
  FSuperDexPose Out;
  for (double Scale : {0.5, 1.25, 2.0}) {
    Config->InputBotScale = Scale;
    TestTrue(TEXT("Scaled generic goals initialize"), Ret->InitializeRetargeter());
    const superdex::robotics::BotIKConfig NativeConfig = Config->ToNative();
    FGenericNativeIK Direct(
        USuperDexIKSolverObject::CreateBotIKSolver(
            GetTransientPackage(), Output.Get(), NativeConfig));
    if (!TestNotNull(TEXT("Independent native output solver"), Direct.Get())) {
      return false;
    }
    for (const superdex::robotics::BotIKGoalConfig& Goal : NativeConfig.goals) {
      mochi::TransformRT Frame = Goal.baseOrInputLinkFrameFromConstraintFrame;
      if (!Goal.inputBotLink.empty()) {
        const int Index = superdex::robotics::FindLinkIndexByName(Model, Goal.inputBotLink);
        mochi::TransformRT Link = NativeLinks[Index];
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
  superdex::robotics::BotPrefab& Cached =
      const_cast<superdex::robotics::BotPrefab&>(Input->GetBotPrefab());
  const mochi::Real3 Axis = Cached.joints[2].axis;
  Cached.joints[2].axis = {};
  TestFalse(TEXT("Malformed native input axis rejects"), Ret->ValidateConfiguration(Diagnostic));
  Cached.joints[2].axis = Axis;
  Cached._dofIndices[0] = 3;
  TestFalse(TEXT("Wrong prepared DOF ordering rejects"), Ret->ValidateConfiguration(Diagnostic));
  Cached._dofIndices[0] = 2;
  const mochi::DynamicString Name = Cached.links[2].name;
  Cached.links[2].name = Cached.links[1].name;
  TestFalse(TEXT("Colliding link/bone names reject"), Ret->ValidateConfiguration(Diagnostic));
  Cached.links[2].name = Name;
  const mochi::ArticulatedJointType Type = Cached.joints[2].type;
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
