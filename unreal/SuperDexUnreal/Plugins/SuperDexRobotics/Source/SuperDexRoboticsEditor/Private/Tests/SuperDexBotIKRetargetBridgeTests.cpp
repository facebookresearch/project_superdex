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

#if WITH_DEV_AUTOMATION_TESTS

#include "Async/Async.h"
#include "Misc/AutomationTest.h"
#include "MochiConversions.h"
#include "MochiPhysicsModule.h"
#include "SuperDexBot.h"
#include "SuperDexBotLoader.h"
#include "SuperDexIKSolverObject.h"
#include "UObject/StrongObjectPtr.h"
#include "UObject/UnrealType.h"

#include <superdex_robotics/utils/bot_utils.h>
#include <superdex_robotics/utils/file_utils.h>
#include <limits>
#include <memory>

namespace {
using namespace superdex::robotics;
using mochi::operator""_r;
using FNativeIK = std::unique_ptr<BotIK, decltype(&BotIK::Destroy)>;

BotPrefab MakeBridgeRobot() {
  BotPrefab Bot;
  Bot.name = "retarget_bridge";
  Bot.worldFromRoot =
      mochi::TransformRT{mochi::Quaternion::RotationZ(0.6_r), mochi::Real3{3_r, -2_r, 1_r}};
  for (int Index = 0; Index < 4; ++Index) {
    BotLinkPrefab Link;
    Link.name = Index == 0 ? "root" : Index == 1 ? "arm" : Index == 2 ? "finger" : "tip";
    Link.parentLink = Index - 1;
    Link.mass = 1_r;
    Link.centerOfMass = mochi::Real3{0.01_r, -0.02_r, 0.03_r};
    Link.momentOfInertia = {1_r, 0_r, 0_r, 1_r, 0_r, 1_r};
    Bot.links.push_back(Link);
    BotJointPrefab Joint;
    Joint.name = Index == 0 ? "root_joint"
        : Index == 1        ? "arm_joint"
        : Index == 2        ? "finger_joint"
                            : "tip_joint";
    Joint.type = Index == 0 ? mochi::ArticulatedJointType::Free
        : Index == 3        ? mochi::ArticulatedJointType::Hard
                            : mochi::ArticulatedJointType::Revolute;
    Joint.axis = {0_r, 0_r, 1_r};
    Joint.parentLinkFromJoint = mochi::TransformRT{
        mochi::Quaternion::RotationZ(Index == 0 ? 0_r : 0.1_r),
        mochi::Real3{Index > 1 ? 0.3_r : 0_r, 0_r, 0_r}};
    Bot.joints.push_back(Joint);
  }
  Bot.defaultPose = {0.2_r, -0.1_r};
  mochi::Error Error;
  RebuildBotData(Bot, Error);
  Validate(Bot, nullptr, Error);
  check(Error.IsOK());
  return Bot;
}

USuperDexBot* MakeBridgeAsset(BotPrefab const& Prefab) {
  auto* Asset = NewObject<USuperDexBot>();
  mochi::Error Error;
  auto Json = WriteToJsonString(Prefab, Error);
  check(Error.IsOK());
  FindFProperty<FStrProperty>(USuperDexBot::StaticClass(), TEXT("BotPrefabJson"))
      ->SetPropertyValue_InContainer(Asset, FString(UTF8_TO_TCHAR(Json.c_str())));
  return Asset;
}

BotIKConfig MakeBridgeConfig() {
  BotIKConfig Config;
  Config.outputBot = "robot.superdex_bot";
  Config.inputBot = "goal.superdex_bot";
  Config.outputBotConfiguration = {0.35_r, -0.25_r};
  Config.inputBotConfiguration = {0.4_r, 0.2_r};
  Config.outputBaseFrameFromInputBaseFrame = mochi::TransformRT{
      mochi::Quaternion::RotationZ(-0.25_r), mochi::Real3{0.03_r, -0.04_r, 0.02_r}};
  Config.inputBotScale = 1.25_r;
  Config.createParams.solverParams.maxIter = 100;
  Config.createParams.solverParams.maxElapsedTimeSeconds = 0;
  Config.createParams.solverParams.absTol = 1e-6_r;
  Config.createParams.solverParams.relTol = 1e-8_r;
  BotIKGoalConfig Goal;
  Goal.name = "tip_goal";
  Goal.outputBotLink = "tip";
  Goal.inputBotLink = "tip";
  Goal.linkFrameFromConstraintFrame = mochi::TransformRT{
      mochi::Quaternion::RotationZ(0.2_r), mochi::Real3{0.05_r, -0.02_r, 0.01_r}};
  Goal.baseOrInputLinkFrameFromConstraintFrame = mochi::TransformRT{
      mochi::Quaternion::RotationZ(-0.15_r), mochi::Real3{-0.03_r, 0.04_r, 0.01_r}};
  Goal.positionStiffness = 100_r;
  Goal.rotationStiffness = 75_r;
  Config.goals.push_back(Goal);
  BotIKJointCouplingConfig Coupling;
  Coupling.jointA = "arm_joint";
  Coupling.jointB = "finger_joint";
  Coupling.ratio = 0.65_r;
  Coupling.stiffness = 200_r;
  Config.jointCouplings.push_back(Coupling);
  return Config;
}

FNativeIK CreateDirect(BotPrefab Prefab, BotIKConfig const& Config, mochi::Error& Error) {
  Prefab.worldFromRoot = {};
  if (!Config.outputBotConfiguration.empty()) {
    Prefab.defaultPose = Config.outputBotConfiguration;
  }
  auto Couplings = ToBotIKJointCouplingParams(Config.jointCouplings);
  FSuperDexBotLoader Loader;
  return FNativeIK(
      BotIK::Create(
          IMochiPhysicsModule::Get()->GetMochiContext(),
          Prefab,
          Config.createParams,
          Couplings,
          Loader,
          Error),
      &BotIK::Destroy);
}

bool CompareBridge(
    FAutomationTestBase& Test,
    USuperDexIKSolverObject* Solver,
    BotIK const& Direct) {
  auto const NativeDofs = Direct.GetJointPositions();
  auto const& Dofs = Solver->GetSolvedDofs();
  if (!Test.TestEqual(
          TEXT("Joint-only snapshot width"), Dofs.Num(), static_cast<int32>(NativeDofs.size()))) {
    return false;
  }
  for (int Index = 0; Index < Dofs.Num(); ++Index) {
    Test.TestTrue(
        TEXT("Bridge matches direct native joint positions"),
        FMath::IsNearlyEqual(Dofs[Index], static_cast<double>(NativeDofs[Index]), 1e-5));
  }
  FMochiToUnrealConverter Converter(kSuperDexSpace);
  auto const Links = Direct.GetWorldFromLinks();
  auto const RawLinks = Solver->GetOutputBotWorldFromLinks();
  if (!Test.TestEqual(
          TEXT("Native borrowed link snapshot width"),
          static_cast<int32>(RawLinks.size()),
          static_cast<int32>(Links.size()))) {
    return false;
  }
  for (int Index = 0; Index < static_cast<int>(Links.size()); ++Index) {
    Test.TestTrue(
        TEXT("Native borrowed snapshot preserves exact rest frames and scale"),
        mochi::NearEqual(RawLinks[Index], Links[Index], 1e-5_r));
  }
  auto const& UE = Solver->GetSolvedLinkToWorldTransforms();
  if (!Test.TestEqual(TEXT("Link snapshot width"), UE.Num(), static_cast<int32>(Links.size()))) {
    return false;
  }
  for (int Index = 0; Index < UE.Num(); ++Index) {
    Test.TestTrue(
        TEXT("Bridge matches FLU/metres to Unreal link snapshots"),
        UE[Index].Equals(Converter.TransformToUnreal(Links[Index]), 1e-3));
  }
  return true;
}
} // namespace

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexBotIKRetargetBridgeTest,
    "SuperDex.BotIK.RetargetBridge",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FSuperDexBotIKRetargetBridgeTest::RunTest(const FString&) {
  auto const Prefab = MakeBridgeRobot();
  TStrongObjectPtr<USuperDexBot> Asset(MakeBridgeAsset(Prefab));
  auto const Before = Asset->GetBotPrefabJson();
  auto Config = MakeBridgeConfig();
  TStrongObjectPtr<USuperDexIKSolverObject> Solver(
      USuperDexIKSolverObject::CreateBotIKSolver(GetTransientPackage(), Asset.Get(), Config));
  mochi::Error Error;
  auto Direct = CreateDirect(Prefab, Config, Error);
  if (!TestNotNull(TEXT("Full config creates the bridge"), Solver.Get()) ||
      !TestTrue(TEXT("Direct native solver created"), Error.IsOK() && Direct != nullptr)) {
    return false;
  }
  CompareBridge(*this, Solver.Get(), *Direct);
  TestEqual(
      TEXT("Source seed is copied once"), Solver->GetSolvedDofs()[0], static_cast<double>(0.35_r));
  TestTrue(
      TEXT("Source world pose is private identity"),
      Solver->GetSolvedLinkToWorldTransforms()[0].Equals(FTransform::Identity));
  auto InputBot = Prefab;
  InputBot.scale = 0.8_r;
  for (int Sample = 0; Sample < 3; ++Sample) {
    mochi::DynamicArray<mochi::real> ReferenceQ{0.4_r + 0.1_r * Sample, 0.2_r - 0.05_r * Sample};
    TestTrue(
        TEXT("Complete input-bot update accepted"),
        Solver->SetBotInputConfiguration(
            InputBot,
            ReferenceQ,
            Config.outputBaseFrameFromInputBaseFrame,
            Config.inputBotScale,
            Config.goals));
    Direct->SetInputBotConfiguration(
        InputBot,
        ReferenceQ,
        Config.outputBaseFrameFromInputBaseFrame,
        Config.inputBotScale,
        Config.goals,
        Error);
    bool Reached = false;
    TestTrue(
        TEXT("Bridge returns error-free best effort independently of convergence"),
        Solver->SolveBotIK(Reached));
    const bool DirectReached = Direct->SolveIK(Error);
    TestTrue(TEXT("Direct native solve remains error-free"), Error.IsOK());
    TestEqual(TEXT("Goals-reached status is forwarded separately"), Reached, DirectReached);
    CompareBridge(*this, Solver.Get(), *Direct);
    const auto Cached = Solver->GetSolvedDofs();
    TestTrue(
        TEXT("Repeated cache reads do not re-solve or reset source seed"),
        Cached == Solver->GetSolvedDofs());
  }
  auto UncoupledConfig = Config;
  UncoupledConfig.jointCouplings.clear();
  auto Uncoupled = CreateDirect(Prefab, UncoupledConfig, Error);
  mochi::DynamicArray<mochi::real> LastQ{0.6_r, 0.1_r};
  Uncoupled->SetInputBotConfiguration(
      InputBot,
      LastQ,
      Config.outputBaseFrameFromInputBaseFrame,
      Config.inputBotScale,
      Config.goals,
      Error);
  const bool UncoupledReached = Uncoupled->SolveIK(Error);
  (void)UncoupledReached;
  TestTrue(
      TEXT("Couplings affect the source solve"),
      mochi::Abs(Uncoupled->GetJointPositions()[1] - Direct->GetJointPositions()[1]) > 1e-3_r);
  TestTrue(
      TEXT("Solver creation and updates never mutate the asset"),
      Asset->GetBotPrefabJson() == Before);

  // The unbound component API must retain the same nonidentity local attachment.
  Direct->ClearAllGoals(Error);
  Solver->ClearAllTargets();
  auto const& Goal = Config.goals[0];
  TestTrue(
      TEXT("Full local attachment update accepted"),
      Solver->SetBotGoal(
          3,
          Goal.linkFrameFromConstraintFrame,
          Goal.baseOrInputLinkFrameFromConstraintFrame,
          Goal.positionStiffness,
          Goal.rotationStiffness));
  Direct->SetPositionGoal(
      3,
      Goal.linkFrameFromConstraintFrame.GetTranslation(),
      Goal.baseOrInputLinkFrameFromConstraintFrame.GetTranslation(),
      Goal.positionStiffness,
      Error);
  Direct->SetRotationGoal(
      3,
      Goal.linkFrameFromConstraintFrame.GetRotation(),
      Goal.baseOrInputLinkFrameFromConstraintFrame.GetRotation(),
      Goal.rotationStiffness,
      Error);
  bool Reached = false;
  TestTrue(TEXT("Attachment bridge solves"), Solver->SolveBotIK(Reached));
  const bool DirectReached = Direct->SolveIK(Error);
  TestEqual(TEXT("Attachment convergence parity"), Reached, DirectReached);
  TestTrue(TEXT("Attachment direct solve error-free"), Error.IsOK());
  CompareBridge(*this, Solver.Get(), *Direct);
  Solver->DestroyMochiResources();
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexBotIKRetargetBridgeValidationTest,
    "SuperDex.BotIK.RetargetBridgeValidation",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FSuperDexBotIKRetargetBridgeValidationTest::RunTest(const FString&) {
  auto const Prefab = MakeBridgeRobot();
  TStrongObjectPtr<USuperDexBot> Asset(MakeBridgeAsset(Prefab));
  auto const Before = Asset->GetBotPrefabJson();
  auto const Config = MakeBridgeConfig();
  auto Reject = [&](BotIKConfig const& Invalid) {
    auto* Rejected =
        USuperDexIKSolverObject::CreateBotIKSolver(GetTransientPackage(), Asset.Get(), Invalid);
    TestNull(TEXT("Invalid complete config rejects creation"), Rejected);
    if (Rejected) {
      Rejected->DestroyMochiResources();
    }
  };
  auto Invalid = Config;
  Invalid.outputBotConfiguration = {0.1_r};
  Reject(Invalid);
  Invalid = Config;
  Invalid.outputBotConfiguration[0] = std::numeric_limits<mochi::real>::quiet_NaN();
  Reject(Invalid);
  Invalid = Config;
  Invalid.jointCouplings[0].jointB = "missing";
  Reject(Invalid);
  Invalid = Config;
  Invalid.goals[0].outputBotLink = "missing";
  Reject(Invalid);
  Invalid = Config;
  Invalid.goals[0].linkFrameFromConstraintFrame.SetRotation(mochi::Quaternion(0_r, 0_r, 0_r, 0_r));
  Reject(Invalid);
  TestTrue(
      TEXT("Rejected config leaves source asset untouched"), Asset->GetBotPrefabJson() == Before);
  for (int Mode = 0; Mode < 3; ++Mode) {
    TStrongObjectPtr<USuperDexIKSolverObject> Solver(
        Mode == 0 ? USuperDexIKSolverObject::CreateBotIKSolver(
                        GetTransientPackage(), Asset.Get(), Config.createParams)
            : Mode == 1 ? USuperDexIKSolverObject::CreateBotIKSolver(
                              GetTransientPackage(), Asset.Get(), Config.createParams, {})
                        : [&]() {
                            auto Empty = Config;
                            Empty.outputBotConfiguration.clear();
                            return USuperDexIKSolverObject::CreateBotIKSolver(
                                GetTransientPackage(), Asset.Get(), Empty);
                          }());
    if (!TestNotNull(TEXT("Existing and empty-seed overloads create"), Solver.Get())) {
      return false;
    }
    TestEqual(
        TEXT("Empty seed retains authored default"),
        Solver->GetSolvedDofs()[0],
        static_cast<double>(Prefab.defaultPose[0]));
    auto const Cached = Solver->GetSolvedDofs();
    TestTrue(
        TEXT("Valid goal batch accepted"),
        Solver->SetBotInputConfiguration(
            Prefab,
            Config.inputBotConfiguration,
            Config.outputBaseFrameFromInputBaseFrame,
            Config.inputBotScale,
            Config.goals));
    auto BadGoals = Config.goals;
    BadGoals[0].inputBotLink = "missing";
    TestFalse(
        TEXT("Invalid goal binding rejected"),
        Solver->SetBotInputConfiguration(
            Prefab,
            Config.inputBotConfiguration,
            Config.outputBaseFrameFromInputBaseFrame,
            Config.inputBotScale,
            BadGoals));
    TestFalse(
        TEXT("Invalid retarget scale rejected"),
        Solver->SetBotInputConfiguration(
            Prefab,
            Config.inputBotConfiguration,
            Config.outputBaseFrameFromInputBaseFrame,
            0_r,
            Config.goals));
    TestFalse(
        TEXT("Invalid configuration width rejected"),
        Solver->SetBotInputConfiguration(
            Prefab,
            {Config.inputBotConfiguration.data(), 1},
            Config.outputBaseFrameFromInputBaseFrame,
            Config.inputBotScale,
            Config.goals));
    TestTrue(
        TEXT("Rejected updates do not change result cache"), Solver->GetSolvedDofs() == Cached);
    TestFalse(
        TEXT("Attachment invalid values rejected before mutation"),
        Solver->SetBotGoal(
            3,
            Config.goals[0].linkFrameFromConstraintFrame,
            Config.goals[0].baseOrInputLinkFrameFromConstraintFrame,
            -1_r,
            75_r));
    auto NonunitGoal = Config.goals[0].baseOrInputLinkFrameFromConstraintFrame;
    NonunitGoal.SetRotation(mochi::Quaternion(0_r, 0_r, 0_r, 2_r));
    TestFalse(
        TEXT("Nonunit attachment goal rejected before touching either goal component"),
        Solver->SetBotGoal(
            3, Config.goals[0].linkFrameFromConstraintFrame, NonunitGoal, 100_r, 75_r));
    TestTrue(
        TEXT("Game-thread guards reject worker calls without solving"),
        Async(EAsyncExecution::Thread, [&]() {
          bool Reached = true;
          return !Solver->SetBotInputConfiguration(
                     Prefab,
                     Config.inputBotConfiguration,
                     Config.outputBaseFrameFromInputBaseFrame,
                     Config.inputBotScale,
                     Config.goals) &&
              !Solver->SolveBotIK(Reached) && !Reached &&
              Solver->GetOutputBotWorldFromLinks().empty() &&
              !USuperDexIKSolverObject::CreateBotIKSolver(
                  GetTransientPackage(), Asset.Get(), Config);
        }).Get());
    bool Reached = false;
    TestTrue(TEXT("Valid goals survive rejected updates"), Solver->SolveBotIK(Reached));
    Solver->DestroyMochiResources();
    TestTrue(
        TEXT("Destroyed native borrowed snapshot is empty"),
        Solver->GetOutputBotWorldFromLinks().empty());
    TestFalse(
        TEXT("Destroyed native bridge rejects updates"),
        Solver->SetBotInputConfiguration(
            Prefab,
            Config.inputBotConfiguration,
            Config.outputBaseFrameFromInputBaseFrame,
            Config.inputBotScale,
            Config.goals));
  }
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexBotIKSingleFrameGoalValidationTest,
    "SuperDex.BotIK.SingleFrameGoalValidation",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FSuperDexBotIKSingleFrameGoalValidationTest::RunTest(const FString&) {
  auto const Prefab = MakeBridgeRobot();
  TStrongObjectPtr<USuperDexBot> Asset(MakeBridgeAsset(Prefab));
  auto const Config = MakeBridgeConfig();
  const mochi::TransformRT Goal{
      mochi::Quaternion::RotationZ(0.45_r), mochi::Real3{0.5_r, 0.4_r, 0_r}};
  const mochi::TransformRT ChangedGoal{
      mochi::Quaternion::RotationZ(-0.7_r), mochi::Real3{-0.4_r, 0.7_r, 0_r}};
  auto Reject = [&](int32 LinkIndex,
                    mochi::TransformRT const& InvalidGoal,
                    mochi::real PositionStiffness,
                    mochi::real RotationStiffness) {
    TStrongObjectPtr<USuperDexIKSolverObject> Solver(
        USuperDexIKSolverObject::CreateBotIKSolver(GetTransientPackage(), Asset.Get(), Config));
    mochi::Error Error;
    auto Direct = CreateDirect(Prefab, Config, Error);
    if (!TestNotNull(TEXT("Single-frame fixture creates"), Solver.Get()) ||
        !TestTrue(TEXT("Direct fixture creates"), Error.IsOK() && Direct != nullptr)) {
      return;
    }
    TestTrue(TEXT("Valid single-frame goal accepted"), Solver->SetBotGoal(3, Goal, 100_r, 75_r));
    Direct->SetPositionGoal(3, {}, Goal.GetTranslation(), 100_r, Error);
    Direct->SetRotationGoal(3, {}, Goal.GetRotation(), 75_r, Error);
    TestFalse(
        TEXT("Invalid single-frame update rejected"),
        Solver->SetBotGoal(LinkIndex, InvalidGoal, PositionStiffness, RotationStiffness));
    bool Reached = false;
    TestTrue(TEXT("Previous single-frame goals remain solvable"), Solver->SolveBotIK(Reached));
    const bool DirectReached = Direct->SolveIK(Error);
    TestTrue(TEXT("Reference goals solve without error"), Error.IsOK());
    TestEqual(TEXT("Rejected update preserves convergence result"), Reached, DirectReached);
    CompareBridge(*this, Solver.Get(), *Direct);
    TestTrue(TEXT("Zero stiffness still clears goals"), Solver->SetBotGoal(3, Goal, 0_r, 0_r));
    Direct->ClearAllGoals(Error);
    TestFalse(TEXT("Cleared goals have no IK objective"), Solver->SolveBotIK(Reached));
    const bool ClearedDirectReached = Direct->SolveIK(Error);
    TestFalse(TEXT("Direct solver also rejects an empty objective"), Error.IsOK());
    TestEqual(TEXT("Cleared goal convergence parity"), Reached, ClearedDirectReached);
    CompareBridge(*this, Solver.Get(), *Direct);
    Solver->DestroyMochiResources();
  };
  Reject(-1, ChangedGoal, 100_r, 75_r);
  Reject(4, ChangedGoal, 100_r, 75_r);
  for (mochi::real InvalidStiffness :
       {-1_r,
        std::numeric_limits<mochi::real>::quiet_NaN(),
        std::numeric_limits<mochi::real>::infinity()}) {
    Reject(3, ChangedGoal, InvalidStiffness, 75_r);
    Reject(3, ChangedGoal, 100_r, InvalidStiffness);
  }
  auto NonunitGoal = ChangedGoal;
  NonunitGoal.SetRotation(mochi::Quaternion(0_r, 0_r, 0_r, 2_r));
  Reject(3, NonunitGoal, 100_r, 75_r);
  return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
