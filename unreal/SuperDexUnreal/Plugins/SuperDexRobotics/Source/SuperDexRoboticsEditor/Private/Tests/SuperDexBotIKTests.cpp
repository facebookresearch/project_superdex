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

#include <mochi_core/utils/math_utils.h>
#include <superdex_robotics/utils/bot_utils.h>
#include <superdex_robotics/utils/file_utils.h>
#include <limits>
#include "AssetRegistry/AssetRegistryModule.h"
#include "Dom/JsonObject.h"
#include "EditorFramework/AssetImportData.h"
#include "HAL/FileManager.h"
#include "Misc/AutomationTest.h"
#include "Misc/FileHelper.h"
#include "Misc/Guid.h"
#include "Misc/Paths.h"
#include "MochiConversions.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "SuperDexBot.h"
#include "SuperDexBotIKComponent.h"
#include "SuperDexBotIKConfig.h"
#include "SuperDexBotIKConfigFactory.h"
#include "SuperDexIKSolverObject.h"
#include "UObject/CoreRedirects.h"
#include "UObject/StrongObjectPtr.h"
#include "UObject/UnrealType.h"

namespace {

using mochi::operator""_r;

superdex::robotics::BotPrefab MakeRobot(bool bFreeRoot) {
  using namespace superdex::robotics;
  BotPrefab Prefab;
  Prefab.name = "ik_test";
  Prefab.worldFromRoot = mochi::TransformRT{mochi::Real3{3_r, 2_r, 1_r}};
  for (int32 Index = 0; Index < 3; ++Index) {
    BotLinkPrefab Link;
    Link.name = Index == 0 ? "root" : Index == 1 ? "arm" : "tip";
    Link.parentLink = Index - 1;
    Link.mass = 1_r;
    Link.centerOfMass = mochi::Real3{};
    Link.momentOfInertia = {1_r, 0_r, 0_r, 1_r, 0_r, 1_r};
    Prefab.links.push_back(Link);
    BotJointPrefab Joint;
    Joint.name = Index == 0 ? "root_joint" : Index == 1 ? "arm_joint" : "tip_joint";
    Joint.type = Index == 0
        ? (bFreeRoot ? mochi::ArticulatedJointType::Free : mochi::ArticulatedJointType::Hard)
        : (Index == 1 ? mochi::ArticulatedJointType::Revolute : mochi::ArticulatedJointType::Hard);
    Joint.axis = {0_r, 0_r, 1_r};
    Joint.parentLinkFromJoint =
        Index == 2 ? mochi::TransformRT{mochi::Real3{1_r, 0_r, 0_r}} : mochi::TransformRT{};
    Prefab.joints.push_back(Joint);
  }
  Prefab.defaultPose = {0.25_r};
  mochi::Error Error;
  RebuildBotData(Prefab, Error);
  Validate(Prefab, nullptr, Error);
  check(Error.IsOK());
  return Prefab;
}

struct FIKFixture {
  FString Directory =
      FPaths::Combine(FPaths::ProjectSavedDir(), TEXT("IKTests"), FGuid::NewGuid().ToString());
  FString PackagePath = TEXT("/Game/__SuperDexIKTests/") + FGuid::NewGuid().ToString();
  TArray<TStrongObjectPtr<USuperDexBot>> Bots;

  FIKFixture() {
    IFileManager::Get().MakeDirectory(*Directory, true);
  }

  ~FIKFixture() {
    for (auto& Bot : Bots) {
      FAssetRegistryModule::AssetDeleted(Bot.Get());
      Bot->MarkAsGarbage();
    }
    IFileManager::Get().DeleteDirectory(*Directory, false, true);
  }

  USuperDexBot* MakeBot(const FString& Name, const FString& SourcePath, bool bFreeRoot = false) {
    auto* Bot = NewObject<USuperDexBot>(
        CreatePackage(*(PackagePath + TEXT("/") + Name)), *Name, RF_Public | RF_Standalone);
    auto* JsonProperty =
        FindFProperty<FStrProperty>(USuperDexBot::StaticClass(), TEXT("BotPrefabJson"));
    mochi::Error Error;
    const auto Json = superdex::robotics::WriteToJsonString(MakeRobot(bFreeRoot), Error);
    check(Error.IsOK());
    if (!SourcePath.IsEmpty()) {
      check(FFileHelper::SaveStringToFile(UTF8_TO_TCHAR(Json.c_str()), *SourcePath));
    }
    JsonProperty->SetPropertyValue_InContainer(Bot, FString(UTF8_TO_TCHAR(Json.c_str())));
    auto* ImportProperty =
        FindFProperty<FObjectPropertyBase>(USuperDexBot::StaticClass(), TEXT("AssetImportData"));
    auto* ImportData =
        CastChecked<UAssetImportData>(ImportProperty->GetObjectPropertyValue_InContainer(Bot));
    ImportData->UpdateFilenameOnly(SourcePath);
    Bots.Emplace(Bot);
    FAssetRegistryModule::AssetCreated(Bot);
    return Bot;
  }

  FString WriteConfig(
      const FString& Reference,
      double Stiffness = 100.0,
      const FString& InputBotReference = TEXT("")) {
    superdex::robotics::BotIKConfig Config;
    Config.outputBot = TCHAR_TO_UTF8(*Reference);
    Config.inputBot = TCHAR_TO_UTF8(*InputBotReference);
    if (!InputBotReference.IsEmpty()) {
      Config.outputBotConfiguration = {0.4_r};
      Config.inputBotConfiguration = {0.6_r};
      Config.inputBotScale = 1.25_r;
      Config.outputBaseFrameFromInputBaseFrame = mochi::TransformRT{
          mochi::Quaternion::FromRotationVector(mochi::Real3{-0.2_r, 0.1_r, 0.5_r}),
          mochi::Real3{0.07_r, -0.08_r, 0.09_r}};
    }
    superdex::robotics::BotIKGoalConfig Goal;
    Goal.name = "Goal";
    Goal.outputBotLink = "tip";
    if (!InputBotReference.IsEmpty()) {
      Goal.inputBotLink = "tip";
      Goal.linkFrameFromConstraintFrame = mochi::TransformRT{
          mochi::Quaternion::FromRotationVector(mochi::Real3{0.3_r, 0.2_r, -0.1_r}),
          mochi::Real3{0.01_r, -0.02_r, 0.03_r}};
    }
    Goal.baseOrInputLinkFrameFromConstraintFrame = mochi::TransformRT{
        mochi::Quaternion::FromRotationVector(mochi::Real3{0.2_r, -0.3_r, 0.4_r}),
        mochi::Real3{0.8_r, 0.6_r, 0_r}};
    Goal.positionStiffness = static_cast<mochi::real>(Stiffness);
    Config.goals.push_back(Goal);
    // Disabled entries need not bind to this one-DOF import fixture, but must be preserved.
    superdex::robotics::BotIKJointCouplingConfig Coupling;
    Coupling.jointA = "pip_joint";
    Coupling.jointB = "dip_joint";
    Coupling.ratio = 0.67_r;
    Coupling.stiffness = static_cast<mochi::real>(Stiffness);
    Coupling.enabled = false;
    Config.jointCouplings.push_back(Coupling);
    FString Path = FPaths::Combine(Directory, TEXT("goals.superdex_ik"));
    const auto Json = superdex::robotics::SerializeIKConfig(Config);
    check(FFileHelper::SaveStringToFile(UTF8_TO_TCHAR(Json.c_str()), *Path));
    return Path;
  }
};

bool VerifyLegacyV8Fixture(FAutomationTestBase& Test) {
  const char* LegacyV8 = R"json({
    "version":8,"bot":"v8_output.superdex_bot","botConfiguration":[0.25,-0.5,0.75],
    "referenceBot":"../v8_input.superdex_bot","referenceBotConfiguration":[-0.125,0.375],
    "referenceBotScale":1.25,
    "baseFromReferenceBot":{"rotation":[0,0,0.6,0.8],"translation":[0.05,-0.06,0.07]},
    "createParams":{"fixedBase":false,"excludeCollisionShapes":false,"solverParams":{
      "maxIter":37,"verbosity":"Verbose","absTol":0.003,"relTol":0.0002,
      "positionErrorThres":0.004,"rotationErrorThres":0.005,"lineSearchMaxIter":13,"maxElapsedTimeSeconds":0.075}},
    "goals":[{"name":"constraint","enabled":true,"link":"tip","referenceBotLink":"input_tip",
      "linkRelativeFrame":{"rotation":[0.6,0,0,0.8],"translation":[0.01,0.02,-0.03]},
      "goalFrame":{"rotation":[0,0.6,0,0.8],"translation":[0.2,-0.3,0.4]},
      "positionStiffness":123.5,"rotationStiffness":57.25}],
    "jointCouplings":[{"jointA":"joint_a","jointB":"joint_b","ratio":-0.65,"stiffness":32,"enabled":true}]
  })json";
  const char* CurrentV9 = R"json({
    "version":9,"outputBot":"v8_output.superdex_bot","outputBotConfiguration":[0.25,-0.5,0.75],
    "inputBot":"../v8_input.superdex_bot","inputBotConfiguration":[-0.125,0.375],
    "inputBotScale":1.25,
    "outputBaseFrameFromInputBaseFrame":{"rotation":[0,0,0.6,0.8],"translation":[0.05,-0.06,0.07]},
    "createParams":{"fixedBase":false,"excludeCollisionShapes":false,"solverParams":{
      "maxIter":37,"verbosity":"Verbose","absTol":0.003,"relTol":0.0002,
      "positionErrorThres":0.004,"rotationErrorThres":0.005,"lineSearchMaxIter":13,"maxElapsedTimeSeconds":0.075}},
    "goals":[{"name":"constraint","enabled":true,"outputBotLink":"tip","inputBotLink":"input_tip",
      "linkFrameFromConstraintFrame":{"rotation":[0.6,0,0,0.8],"translation":[0.01,0.02,-0.03]},
      "baseOrInputLinkFrameFromConstraintFrame":{"rotation":[0,0.6,0,0.8],"translation":[0.2,-0.3,0.4]},
      "positionStiffness":123.5,"rotationStiffness":57.25}],
    "jointCouplings":[{"jointA":"joint_a","jointB":"joint_b","ratio":-0.65,"stiffness":32,"enabled":true}]
  })json";
  superdex::robotics::BotIKConfig Migrated, Expected;
  mochi::DynamicString Diagnostic;
  if (!Test.TestTrue(
          TEXT("Complete explicit v8 legacy fixture parses"),
          superdex::robotics::ParseIKConfig(LegacyV8, Migrated, Diagnostic)) ||
      !Test.TestTrue(
          TEXT("Independent complete v9 fixture parses"),
          superdex::robotics::ParseIKConfig(CurrentV9, Expected, Diagnostic))) {
    return false;
  }
  Test.TestEqual(TEXT("Legacy v8 migrates to canonical v9"), Migrated.version, 9);
  Test.TestTrue(
      TEXT("Every legacy v8 number/reference/frame migrates without inversion"),
      superdex::robotics::SerializeIKConfig(Migrated) ==
          superdex::robotics::SerializeIKConfig(Expected));
  TStrongObjectPtr<USuperDexBotIKConfig> Asset(NewObject<USuperDexBotIKConfig>());
  Asset->FromNative(Migrated);
  auto RoundTrip = Asset->ToNative();
  Test.TestTrue(
      TEXT("Legacy v8 output/input base frame retains direction"),
      mochi::NearEqual(
          RoundTrip.outputBaseFrameFromInputBaseFrame,
          Expected.outputBaseFrameFromInputBaseFrame,
          1e-5_r));
  Test.TestTrue(
      TEXT("Legacy v8 output constraint attachment retains direction"),
      mochi::NearEqual(
          RoundTrip.goals[0].linkFrameFromConstraintFrame,
          Expected.goals[0].linkFrameFromConstraintFrame,
          1e-5_r));
  Test.TestTrue(
      TEXT("Legacy v8 input constraint frame retains direction"),
      mochi::NearEqual(
          RoundTrip.goals[0].baseOrInputLinkFrameFromConstraintFrame,
          Expected.goals[0].baseOrInputLinkFrameFromConstraintFrame,
          1e-5_r));
  RoundTrip.outputBaseFrameFromInputBaseFrame = Expected.outputBaseFrameFromInputBaseFrame;
  RoundTrip.goals[0].linkFrameFromConstraintFrame = Expected.goals[0].linkFrameFromConstraintFrame;
  RoundTrip.goals[0].baseOrInputLinkFrameFromConstraintFrame =
      Expected.goals[0].baseOrInputLinkFrameFromConstraintFrame;
  Test.TestTrue(
      TEXT("All v8 scalar/ref/settings fields survive full UE round trip exactly"),
      superdex::robotics::SerializeIKConfig(RoundTrip) ==
          superdex::robotics::SerializeIKConfig(Expected));
  return !Test.HasAnyErrors();
}

bool RecordColdLoadedConfigs(FAutomationTestBase& Test) {
  const FString ManifestPath =
      FPlatformMisc::GetEnvironmentVariable(TEXT("SUPERDEX_IK_COLDLOAD_MANIFEST"));
  const FString ReportPath =
      FPlatformMisc::GetEnvironmentVariable(TEXT("SUPERDEX_IK_COLDLOAD_REPORT"));
  if (ManifestPath.IsEmpty() && ReportPath.IsEmpty()) {
    return true;
  }
  if (!Test.TestTrue(
          TEXT("Cold-load probe requires both explicit manifest and report paths"),
          !ManifestPath.IsEmpty() && !ReportPath.IsEmpty())) {
    return false;
  }
  FString Text;
  TSharedPtr<FJsonObject> Manifest;
  if (!Test.TestTrue(
          TEXT("Read explicit cold-load manifest"),
          FFileHelper::LoadFileToString(Text, *ManifestPath) &&
              FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Text), Manifest))) {
    return false;
  }
  const TArray<TSharedPtr<FJsonValue>>* Entries = nullptr;
  if (!Test.TestTrue(
          TEXT("Cold-load manifest enumerates config assets"),
          Manifest->TryGetArrayField(TEXT("configs"), Entries) && !Entries->IsEmpty())) {
    return false;
  }
  TArray<TSharedPtr<FJsonValue>> Snapshots;
  for (const auto& Entry : *Entries) {
    const FString AssetPath = Entry->AsObject()->GetStringField(TEXT("asset"));
    TStrongObjectPtr<USuperDexBotIKConfig> Asset(
        LoadObject<USuperDexBotIKConfig>(nullptr, *AssetPath));
    if (!Test.TestNotNull(TEXT("Cold-load old serialized config without reimport"), Asset.Get())) {
      return false;
    }
    FString Diagnostic;
    if (!Test.TestTrue(TEXT("Cold-loaded config validates"), Asset->Validate(Diagnostic)) ||
        !Test.TestNotNull(TEXT("Cold-loaded Bot reference survives"), Asset->OutputBot.Get()) ||
        !Test.TestNotNull(
            TEXT("Cold-loaded import provenance survives"), Asset->AssetImportData.Get())) {
      return false;
    }
    const auto Native = Asset->ToNative();
    if (!Native.inputBot.empty() &&
        !Test.TestNotNull(TEXT("Cold-loaded InputBot survives"), Asset->InputBot.Get())) {
      return false;
    }
    const auto Json = superdex::robotics::SerializeIKConfig(Native);
    TSharedPtr<FJsonObject> NativeJson;
    if (!Test.TestTrue(
            TEXT("Cold-loaded native snapshot is canonical JSON"),
            FJsonSerializer::Deserialize(
                TJsonReaderFactory<>::Create(FString(UTF8_TO_TCHAR(Json.c_str()))), NativeJson))) {
      return false;
    }
    Test.TestEqual(TEXT("Cold-loaded native schema is v9"), Native.version, 9);
    auto Snapshot = MakeShared<FJsonObject>();
    Snapshot->SetStringField(TEXT("asset"), Asset->GetPathName());
    Snapshot->SetObjectField(TEXT("native"), NativeJson);
    Snapshot->SetStringField(TEXT("sourceBotPath"), Asset->SourceOutputBotPath);
    Snapshot->SetStringField(TEXT("sourceReferenceBotPath"), Asset->SourceInputBotPath);
    Snapshot->SetStringField(TEXT("importFilename"), Asset->AssetImportData->GetFirstFilename());
    Snapshots.Add(MakeShared<FJsonValueObject>(Snapshot));
  }
  auto Report = MakeShared<FJsonObject>();
  Report->SetArrayField(TEXT("configs"), Snapshots);
  Report->SetBoolField(TEXT("complete"), !Test.HasAnyErrors());
  FString ReportText;
  if (!FJsonSerializer::Serialize(Report, TJsonWriterFactory<>::Create(&ReportText)) ||
      !Test.TestTrue(
          TEXT("Save cold-load numeric/native proof"),
          FFileHelper::SaveStringToFile(ReportText, *ReportPath))) {
    return false;
  }
  UE_LOG(LogTemp, Display, TEXT("BOT_IK_COLDLOAD_NATIVE_COMPLETE %s"), *ReportPath);
  return !Test.HasAnyErrors();
}

} // namespace

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexBotIKConversionTest,
    "SuperDex.BotIK.Conversion",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FSuperDexBotIKConversionTest::RunTest(const FString& Parameters) {
  TestEqual(TEXT("Current native schema is v9"), superdex::robotics::kIKConfigVersion, 9);
  TestTrue(
      TEXT("Direct historical serialized redirect"),
      FCoreRedirects::GetRedirectedName(
          ECoreRedirectFlags::Type_Struct,
          FCoreRedirectObjectName(
              FString(TEXT("/Script/SuperDexRobotics.SuperDexBotIKTargetConfig")))) ==
          FCoreRedirectObjectName(
              FString(TEXT("/Script/SuperDexRobotics.SuperDexBotIKGoalConfig"))));
  TestTrue(
      TEXT("Direct historical serialized redirect"),
      FCoreRedirects::GetRedirectedName(
          ECoreRedirectFlags::Type_Enum,
          FCoreRedirectObjectName(
              FString(TEXT("/Script/SuperDexRobotics.ESuperDexBotIKTargetSpace")))) ==
          FCoreRedirectObjectName(
              FString(TEXT("/Script/SuperDexRobotics.ESuperDexBotIKGoalSpace"))));
  TestTrue(
      TEXT("Direct historical serialized redirect"),
      FCoreRedirects::GetRedirectedName(
          ECoreRedirectFlags::Type_Property,
          FCoreRedirectObjectName(
              FString(TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.Targets")))) ==
          FCoreRedirectObjectName(
              FString(TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.Goals"))));
  TestNull(
      TEXT("No old-name config alias"),
      FindFProperty<FProperty>(USuperDexBotIKConfig::StaticClass(), TEXT("Targets")));
  TestNotNull(
      TEXT("Final config property exists"),
      FindFProperty<FProperty>(USuperDexBotIKConfig::StaticClass(), TEXT("Goals")));
  TestTrue(
      TEXT("Direct historical serialized redirect"),
      FCoreRedirects::GetRedirectedName(
          ECoreRedirectFlags::Type_Property,
          FCoreRedirectObjectName(
              FString(TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.Bot")))) ==
          FCoreRedirectObjectName(
              FString(TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.OutputBot"))));
  TestNull(
      TEXT("No old-name config alias"),
      FindFProperty<FProperty>(USuperDexBotIKConfig::StaticClass(), TEXT("Bot")));
  TestNotNull(
      TEXT("Final config property exists"),
      FindFProperty<FProperty>(USuperDexBotIKConfig::StaticClass(), TEXT("OutputBot")));
  TestTrue(
      TEXT("Direct historical serialized redirect"),
      FCoreRedirects::GetRedirectedName(
          ECoreRedirectFlags::Type_Property,
          FCoreRedirectObjectName(
              FString(TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.ReferenceBot")))) ==
          FCoreRedirectObjectName(
              FString(TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.InputBot"))));
  TestNull(
      TEXT("No old-name config alias"),
      FindFProperty<FProperty>(USuperDexBotIKConfig::StaticClass(), TEXT("ReferenceBot")));
  TestNotNull(
      TEXT("Final config property exists"),
      FindFProperty<FProperty>(USuperDexBotIKConfig::StaticClass(), TEXT("InputBot")));
  TestTrue(
      TEXT("Direct historical serialized redirect"),
      FCoreRedirects::GetRedirectedName(
          ECoreRedirectFlags::Type_Property,
          FCoreRedirectObjectName(
              FString(TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.TargetBot")))) ==
          FCoreRedirectObjectName(
              FString(TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.InputBot"))));
  TestNull(
      TEXT("No old-name config alias"),
      FindFProperty<FProperty>(USuperDexBotIKConfig::StaticClass(), TEXT("TargetBot")));
  TestNotNull(
      TEXT("Final config property exists"),
      FindFProperty<FProperty>(USuperDexBotIKConfig::StaticClass(), TEXT("InputBot")));
  TestTrue(
      TEXT("Direct historical serialized redirect"),
      FCoreRedirects::GetRedirectedName(
          ECoreRedirectFlags::Type_Property,
          FCoreRedirectObjectName(
              FString(TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.BotConfiguration")))) ==
          FCoreRedirectObjectName(FString(
              TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.OutputBotConfiguration"))));
  TestNull(
      TEXT("No old-name config alias"),
      FindFProperty<FProperty>(USuperDexBotIKConfig::StaticClass(), TEXT("BotConfiguration")));
  TestNotNull(
      TEXT("Final config property exists"),
      FindFProperty<FProperty>(
          USuperDexBotIKConfig::StaticClass(), TEXT("OutputBotConfiguration")));
  TestTrue(
      TEXT("Direct historical serialized redirect"),
      FCoreRedirects::GetRedirectedName(
          ECoreRedirectFlags::Type_Property,
          FCoreRedirectObjectName(FString(
              TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.ReferenceBotConfiguration")))) ==
          FCoreRedirectObjectName(
              FString(TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.InputBotConfiguration"))));
  TestNull(
      TEXT("No old-name config alias"),
      FindFProperty<FProperty>(
          USuperDexBotIKConfig::StaticClass(), TEXT("ReferenceBotConfiguration")));
  TestNotNull(
      TEXT("Final config property exists"),
      FindFProperty<FProperty>(USuperDexBotIKConfig::StaticClass(), TEXT("InputBotConfiguration")));
  TestTrue(
      TEXT("Direct historical serialized redirect"),
      FCoreRedirects::GetRedirectedName(
          ECoreRedirectFlags::Type_Property,
          FCoreRedirectObjectName(FString(
              TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.TargetBotConfiguration")))) ==
          FCoreRedirectObjectName(
              FString(TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.InputBotConfiguration"))));
  TestNull(
      TEXT("No old-name config alias"),
      FindFProperty<FProperty>(
          USuperDexBotIKConfig::StaticClass(), TEXT("TargetBotConfiguration")));
  TestNotNull(
      TEXT("Final config property exists"),
      FindFProperty<FProperty>(USuperDexBotIKConfig::StaticClass(), TEXT("InputBotConfiguration")));
  TestTrue(
      TEXT("Direct historical serialized redirect"),
      FCoreRedirects::GetRedirectedName(
          ECoreRedirectFlags::Type_Property,
          FCoreRedirectObjectName(FString(
              TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.BaseFromReferenceBot")))) ==
          FCoreRedirectObjectName(FString(TEXT(
              "/Script/SuperDexRobotics.SuperDexBotIKConfig.OutputBaseFrameFromInputBaseFrame"))));
  TestNull(
      TEXT("No old-name config alias"),
      FindFProperty<FProperty>(USuperDexBotIKConfig::StaticClass(), TEXT("BaseFromReferenceBot")));
  TestNotNull(
      TEXT("Final config property exists"),
      FindFProperty<FProperty>(
          USuperDexBotIKConfig::StaticClass(), TEXT("OutputBaseFrameFromInputBaseFrame")));
  TestTrue(
      TEXT("Direct historical serialized redirect"),
      FCoreRedirects::GetRedirectedName(
          ECoreRedirectFlags::Type_Property,
          FCoreRedirectObjectName(
              FString(TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.BaseFromTargetBot")))) ==
          FCoreRedirectObjectName(FString(TEXT(
              "/Script/SuperDexRobotics.SuperDexBotIKConfig.OutputBaseFrameFromInputBaseFrame"))));
  TestNull(
      TEXT("No old-name config alias"),
      FindFProperty<FProperty>(USuperDexBotIKConfig::StaticClass(), TEXT("BaseFromTargetBot")));
  TestNotNull(
      TEXT("Final config property exists"),
      FindFProperty<FProperty>(
          USuperDexBotIKConfig::StaticClass(), TEXT("OutputBaseFrameFromInputBaseFrame")));
  TestTrue(
      TEXT("Direct historical serialized redirect"),
      FCoreRedirects::GetRedirectedName(
          ECoreRedirectFlags::Type_Property,
          FCoreRedirectObjectName(
              FString(TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.ReferenceBotScale")))) ==
          FCoreRedirectObjectName(
              FString(TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.InputBotScale"))));
  TestNull(
      TEXT("No old-name config alias"),
      FindFProperty<FProperty>(USuperDexBotIKConfig::StaticClass(), TEXT("ReferenceBotScale")));
  TestNotNull(
      TEXT("Final config property exists"),
      FindFProperty<FProperty>(USuperDexBotIKConfig::StaticClass(), TEXT("InputBotScale")));
  TestTrue(
      TEXT("Direct historical serialized redirect"),
      FCoreRedirects::GetRedirectedName(
          ECoreRedirectFlags::Type_Property,
          FCoreRedirectObjectName(
              FString(TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.TargetBotScale")))) ==
          FCoreRedirectObjectName(
              FString(TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.InputBotScale"))));
  TestNull(
      TEXT("No old-name config alias"),
      FindFProperty<FProperty>(USuperDexBotIKConfig::StaticClass(), TEXT("TargetBotScale")));
  TestNotNull(
      TEXT("Final config property exists"),
      FindFProperty<FProperty>(USuperDexBotIKConfig::StaticClass(), TEXT("InputBotScale")));
  TestTrue(
      TEXT("Direct historical serialized redirect"),
      FCoreRedirects::GetRedirectedName(
          ECoreRedirectFlags::Type_Property,
          FCoreRedirectObjectName(
              FString(TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.BotReference")))) ==
          FCoreRedirectObjectName(
              FString(TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.OutputBotReference"))));
  TestNull(
      TEXT("No old-name config alias"),
      FindFProperty<FProperty>(USuperDexBotIKConfig::StaticClass(), TEXT("BotReference")));
  TestNotNull(
      TEXT("Final config property exists"),
      FindFProperty<FProperty>(USuperDexBotIKConfig::StaticClass(), TEXT("OutputBotReference")));
  TestTrue(
      TEXT("Direct historical serialized redirect"),
      FCoreRedirects::GetRedirectedName(
          ECoreRedirectFlags::Type_Property,
          FCoreRedirectObjectName(FString(
              TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.ReferenceBotReference")))) ==
          FCoreRedirectObjectName(
              FString(TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.InputBotReference"))));
  TestNull(
      TEXT("No old-name config alias"),
      FindFProperty<FProperty>(USuperDexBotIKConfig::StaticClass(), TEXT("ReferenceBotReference")));
  TestNotNull(
      TEXT("Final config property exists"),
      FindFProperty<FProperty>(USuperDexBotIKConfig::StaticClass(), TEXT("InputBotReference")));
  TestTrue(
      TEXT("Direct historical serialized redirect"),
      FCoreRedirects::GetRedirectedName(
          ECoreRedirectFlags::Type_Property,
          FCoreRedirectObjectName(
              FString(TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.TargetBotReference")))) ==
          FCoreRedirectObjectName(
              FString(TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.InputBotReference"))));
  TestNull(
      TEXT("No old-name config alias"),
      FindFProperty<FProperty>(USuperDexBotIKConfig::StaticClass(), TEXT("TargetBotReference")));
  TestNotNull(
      TEXT("Final config property exists"),
      FindFProperty<FProperty>(USuperDexBotIKConfig::StaticClass(), TEXT("InputBotReference")));
  TestTrue(
      TEXT("Direct historical serialized redirect"),
      FCoreRedirects::GetRedirectedName(
          ECoreRedirectFlags::Type_Property,
          FCoreRedirectObjectName(
              FString(TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.SourceBotPath")))) ==
          FCoreRedirectObjectName(
              FString(TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.SourceOutputBotPath"))));
  TestNull(
      TEXT("No old-name config alias"),
      FindFProperty<FProperty>(USuperDexBotIKConfig::StaticClass(), TEXT("SourceBotPath")));
  TestNotNull(
      TEXT("Final config property exists"),
      FindFProperty<FProperty>(USuperDexBotIKConfig::StaticClass(), TEXT("SourceOutputBotPath")));
  TestTrue(
      TEXT("Direct historical serialized redirect"),
      FCoreRedirects::GetRedirectedName(
          ECoreRedirectFlags::Type_Property,
          FCoreRedirectObjectName(FString(
              TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.SourceReferenceBotPath")))) ==
          FCoreRedirectObjectName(
              FString(TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.SourceInputBotPath"))));
  TestNull(
      TEXT("No old-name config alias"),
      FindFProperty<FProperty>(
          USuperDexBotIKConfig::StaticClass(), TEXT("SourceReferenceBotPath")));
  TestNotNull(
      TEXT("Final config property exists"),
      FindFProperty<FProperty>(USuperDexBotIKConfig::StaticClass(), TEXT("SourceInputBotPath")));
  TestTrue(
      TEXT("Direct historical serialized redirect"),
      FCoreRedirects::GetRedirectedName(
          ECoreRedirectFlags::Type_Property,
          FCoreRedirectObjectName(
              FString(TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.SourceTargetBotPath")))) ==
          FCoreRedirectObjectName(
              FString(TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.SourceInputBotPath"))));
  TestNull(
      TEXT("No old-name config alias"),
      FindFProperty<FProperty>(USuperDexBotIKConfig::StaticClass(), TEXT("SourceTargetBotPath")));
  TestNotNull(
      TEXT("Final config property exists"),
      FindFProperty<FProperty>(USuperDexBotIKConfig::StaticClass(), TEXT("SourceInputBotPath")));
  TestTrue(
      TEXT("Direct historical serialized redirect"),
      FCoreRedirects::GetRedirectedName(
          ECoreRedirectFlags::Type_Property,
          FCoreRedirectObjectName(
              FString(TEXT("/Script/SuperDexRobotics.SuperDexBotIKTargetConfig.Link")))) ==
          FCoreRedirectObjectName(
              FString(TEXT("/Script/SuperDexRobotics.SuperDexBotIKGoalConfig.OutputBotLink"))));
  TestTrue(
      TEXT("Direct historical serialized redirect"),
      FCoreRedirects::GetRedirectedName(
          ECoreRedirectFlags::Type_Property,
          FCoreRedirectObjectName(FString(
              TEXT("/Script/SuperDexRobotics.SuperDexBotIKTargetConfig.ReferenceBotLink")))) ==
          FCoreRedirectObjectName(
              FString(TEXT("/Script/SuperDexRobotics.SuperDexBotIKGoalConfig.InputBotLink"))));
  TestTrue(
      TEXT("Direct historical serialized redirect"),
      FCoreRedirects::GetRedirectedName(
          ECoreRedirectFlags::Type_Property,
          FCoreRedirectObjectName(
              FString(TEXT("/Script/SuperDexRobotics.SuperDexBotIKTargetConfig.TargetBotLink")))) ==
          FCoreRedirectObjectName(
              FString(TEXT("/Script/SuperDexRobotics.SuperDexBotIKGoalConfig.InputBotLink"))));
  TestTrue(
      TEXT("Direct historical serialized redirect"),
      FCoreRedirects::GetRedirectedName(
          ECoreRedirectFlags::Type_Property,
          FCoreRedirectObjectName(FString(
              TEXT("/Script/SuperDexRobotics.SuperDexBotIKTargetConfig.LinkRelativeFrame")))) ==
          FCoreRedirectObjectName(FString(TEXT(
              "/Script/SuperDexRobotics.SuperDexBotIKGoalConfig.LinkFrameFromConstraintFrame"))));
  TestTrue(
      TEXT("Direct historical serialized redirect"),
      FCoreRedirects::GetRedirectedName(
          ECoreRedirectFlags::Type_Property,
          FCoreRedirectObjectName(
              FString(TEXT("/Script/SuperDexRobotics.SuperDexBotIKTargetConfig.Transform")))) ==
          FCoreRedirectObjectName(FString(TEXT(
              "/Script/SuperDexRobotics.SuperDexBotIKGoalConfig.BaseOrInputLinkFrameFromConstraintFrame"))));
  TestTrue(
      TEXT("Direct historical serialized redirect"),
      FCoreRedirects::GetRedirectedName(
          ECoreRedirectFlags::Type_Property,
          FCoreRedirectObjectName(
              FString(TEXT("/Script/SuperDexRobotics.SuperDexBotIKGoalConfig.Link")))) ==
          FCoreRedirectObjectName(
              FString(TEXT("/Script/SuperDexRobotics.SuperDexBotIKGoalConfig.OutputBotLink"))));
  TestNull(
      TEXT("No old-name goal alias"),
      FindFProperty<FProperty>(FSuperDexBotIKGoalConfig::StaticStruct(), TEXT("Link")));
  TestNotNull(
      TEXT("Final goal property exists"),
      FindFProperty<FProperty>(FSuperDexBotIKGoalConfig::StaticStruct(), TEXT("OutputBotLink")));
  TestTrue(
      TEXT("Direct historical serialized redirect"),
      FCoreRedirects::GetRedirectedName(
          ECoreRedirectFlags::Type_Property,
          FCoreRedirectObjectName(FString(
              TEXT("/Script/SuperDexRobotics.SuperDexBotIKGoalConfig.ReferenceBotLink")))) ==
          FCoreRedirectObjectName(
              FString(TEXT("/Script/SuperDexRobotics.SuperDexBotIKGoalConfig.InputBotLink"))));
  TestNull(
      TEXT("No old-name goal alias"),
      FindFProperty<FProperty>(FSuperDexBotIKGoalConfig::StaticStruct(), TEXT("ReferenceBotLink")));
  TestNotNull(
      TEXT("Final goal property exists"),
      FindFProperty<FProperty>(FSuperDexBotIKGoalConfig::StaticStruct(), TEXT("InputBotLink")));
  TestTrue(
      TEXT("Direct historical serialized redirect"),
      FCoreRedirects::GetRedirectedName(
          ECoreRedirectFlags::Type_Property,
          FCoreRedirectObjectName(
              FString(TEXT("/Script/SuperDexRobotics.SuperDexBotIKGoalConfig.TargetBotLink")))) ==
          FCoreRedirectObjectName(
              FString(TEXT("/Script/SuperDexRobotics.SuperDexBotIKGoalConfig.InputBotLink"))));
  TestNull(
      TEXT("No old-name goal alias"),
      FindFProperty<FProperty>(FSuperDexBotIKGoalConfig::StaticStruct(), TEXT("TargetBotLink")));
  TestNotNull(
      TEXT("Final goal property exists"),
      FindFProperty<FProperty>(FSuperDexBotIKGoalConfig::StaticStruct(), TEXT("InputBotLink")));
  TestTrue(
      TEXT("Direct historical serialized redirect"),
      FCoreRedirects::GetRedirectedName(
          ECoreRedirectFlags::Type_Property,
          FCoreRedirectObjectName(FString(
              TEXT("/Script/SuperDexRobotics.SuperDexBotIKGoalConfig.LinkRelativeFrame")))) ==
          FCoreRedirectObjectName(FString(TEXT(
              "/Script/SuperDexRobotics.SuperDexBotIKGoalConfig.LinkFrameFromConstraintFrame"))));
  TestNull(
      TEXT("No old-name goal alias"),
      FindFProperty<FProperty>(
          FSuperDexBotIKGoalConfig::StaticStruct(), TEXT("LinkRelativeFrame")));
  TestNotNull(
      TEXT("Final goal property exists"),
      FindFProperty<FProperty>(
          FSuperDexBotIKGoalConfig::StaticStruct(), TEXT("LinkFrameFromConstraintFrame")));
  TestTrue(
      TEXT("Direct historical serialized redirect"),
      FCoreRedirects::GetRedirectedName(
          ECoreRedirectFlags::Type_Property,
          FCoreRedirectObjectName(
              FString(TEXT("/Script/SuperDexRobotics.SuperDexBotIKGoalConfig.Transform")))) ==
          FCoreRedirectObjectName(FString(TEXT(
              "/Script/SuperDexRobotics.SuperDexBotIKGoalConfig.BaseOrInputLinkFrameFromConstraintFrame"))));
  TestNull(
      TEXT("No old-name goal alias"),
      FindFProperty<FProperty>(FSuperDexBotIKGoalConfig::StaticStruct(), TEXT("Transform")));
  TestNotNull(
      TEXT("Final goal property exists"),
      FindFProperty<FProperty>(
          FSuperDexBotIKGoalConfig::StaticStruct(),
          TEXT("BaseOrInputLinkFrameFromConstraintFrame")));
  TestTrue(
      TEXT("Direct historical serialized redirect"),
      FCoreRedirects::GetRedirectedName(
          ECoreRedirectFlags::Type_Function,
          FCoreRedirectObjectName(FString(
              TEXT("/Script/SuperDexRobotics.SuperDexBotIKComponent.ResetTargetsToConfig")))) ==
          FCoreRedirectObjectName(
              FString(TEXT("/Script/SuperDexRobotics.SuperDexBotIKComponent.ResetGoalsToConfig"))));
  TestTrue(
      TEXT("Direct historical serialized redirect"),
      FCoreRedirects::GetRedirectedName(
          ECoreRedirectFlags::Type_Function,
          FCoreRedirectObjectName(FString(
              TEXT("/Script/SuperDexRobotics.SuperDexBotIKComponent.SetTargetTransform")))) ==
          FCoreRedirectObjectName(
              FString(TEXT("/Script/SuperDexRobotics.SuperDexBotIKComponent.SetGoalTransform"))));
  TestTrue(
      TEXT("Direct historical serialized redirect"),
      FCoreRedirects::GetRedirectedName(
          ECoreRedirectFlags::Type_Function,
          FCoreRedirectObjectName(
              FString(TEXT("/Script/SuperDexRobotics.SuperDexBotIKComponent.SetTargetEnabled")))) ==
          FCoreRedirectObjectName(
              FString(TEXT("/Script/SuperDexRobotics.SuperDexBotIKComponent.SetGoalEnabled"))));
  TestTrue(
      TEXT("Direct historical serialized redirect"),
      FCoreRedirects::GetRedirectedName(
          ECoreRedirectFlags::Type_Function,
          FCoreRedirectObjectName(FString(
              TEXT("/Script/SuperDexRobotics.SuperDexBotIKComponent.SetTargetStiffness")))) ==
          FCoreRedirectObjectName(
              FString(TEXT("/Script/SuperDexRobotics.SuperDexBotIKComponent.SetGoalStiffness"))));
  TestTrue(
      TEXT("Direct historical serialized redirect"),
      FCoreRedirects::GetRedirectedName(
          ECoreRedirectFlags::Type_Property,
          FCoreRedirectObjectName(FString(TEXT(
              "/Script/SuperDexRobotics.SuperDexBotIKComponent.SolveIK.OutTargetsReached")))) ==
          FCoreRedirectObjectName(FString(
              TEXT("/Script/SuperDexRobotics.SuperDexBotIKComponent.SolveIK.OutGoalsReached"))));

  superdex::robotics::BotIKConfig Native;
  Native.outputBot = "robot.superdex_bot";
  Native.inputBot = "../OpenXR/right.superdex_bot";
  Native.inputBotScale = 1.25_r;
  Native.outputBaseFrameFromInputBaseFrame = mochi::TransformRT{
      mochi::Quaternion::FromRotationVector(mochi::Real3{-0.3_r, 0.4_r, 0.2_r}),
      mochi::Real3{0.05_r, -0.06_r, 0.07_r}};
  for (int32 Index = 0; Index < 26; ++Index) {
    Native.inputBotConfiguration.push_back(static_cast<mochi::real>(0.02 * Index));
    if (Index < 20) {
      Native.outputBotConfiguration.push_back(static_cast<mochi::real>(0.03 * (Index + 1)));
    }
  }
  Native.createParams.fixedBase = false;
  Native.createParams.excludeCollisionShapes = false;
  auto& Params = Native.createParams.solverParams;
  Params.maxIter = 37;
  Params.verbosity = mochi::VerbosityLevel::Verbose;
  Params.absTol = 0.003_r;
  Params.relTol = 0.0002_r;
  Params.positionErrorThres = 0.004_r;
  Params.rotationErrorThres = 0.005_r;
  Params.lineSearchMaxIter = 13;
  Params.maxElapsedTimeSeconds = 0.075;
  for (int32 Index = 0; Index < 5; ++Index) {
    superdex::robotics::BotIKGoalConfig Goal;
    Goal.name = TCHAR_TO_UTF8(*FString::Printf(TEXT("Goal%d"), Index));
    Goal.outputBotLink = TCHAR_TO_UTF8(*FString::Printf(TEXT("tip%d"), Index));
    Goal.inputBotLink = TCHAR_TO_UTF8(*FString::Printf(TEXT("openxr_tip%d"), Index));
    Goal.enabled = Index != 0;
    Goal.baseOrInputLinkFrameFromConstraintFrame = mochi::TransformRT{
        mochi::Quaternion::FromRotationVector(mochi::Real3{0.1_r, 0.2_r, -0.3_r}),
        mochi::Real3{0.2_r, -0.3_r, 0.4_r}};
    Goal.linkFrameFromConstraintFrame = mochi::TransformRT{
        mochi::Quaternion::FromRotationVector(mochi::Real3{-0.2_r, 0.3_r, 0.1_r}),
        mochi::Real3{0.01_r, -0.02_r, 0.03_r}};
    Goal.positionStiffness = Index == 0 ? 0_r : 123_r;
    Goal.rotationStiffness = 57_r;
    Native.goals.push_back(Goal);
    superdex::robotics::BotIKJointCouplingConfig Coupling;
    Coupling.jointA = TCHAR_TO_UTF8(*FString::Printf(TEXT("pip_joint%d"), Index));
    Coupling.jointB = TCHAR_TO_UTF8(*FString::Printf(TEXT("dip_joint%d"), Index));
    Coupling.ratio = -0.65_r;
    Coupling.stiffness = 32_r;
    Coupling.enabled = Index != 0;
    Native.jointCouplings.push_back(Coupling);
  }
  TStrongObjectPtr<USuperDexBotIKConfig> Asset(NewObject<USuperDexBotIKConfig>());
  Asset->FromNative(Native);
  auto RoundTrip = Asset->ToNative();
  if (!TestEqual(TEXT("Five goals round trip"), static_cast<int32>(RoundTrip.goals.size()), 5)) {
    return false;
  }
  TestEqual(TEXT("Twenty source coordinates retained"), Asset->OutputBotConfiguration.Num(), 20);
  TestEqual(
      TEXT("Twenty-six reference coordinates retained"), Asset->InputBotConfiguration.Num(), 26);
  TestEqual(TEXT("Five couplings retained"), Asset->JointCouplings.Num(), 5);
  TestEqual(TEXT("Independent input-bot scale retained"), Asset->InputBotScale, 1.25);
  TestTrue(
      TEXT("Both original references retained"),
      RoundTrip.outputBot == Native.outputBot && RoundTrip.inputBot == Native.inputBot);
  TestTrue(
      TEXT("Nonidentity root alignment round trips"),
      mochi::NearEqual(
          RoundTrip.outputBaseFrameFromInputBaseFrame,
          Native.outputBaseFrameFromInputBaseFrame,
          1e-5_r));
  for (int32 Index = 0; Index < Asset->Goals.Num(); ++Index) {
    TestTrue(
        TEXT("Goal rigid frame round trips"),
        mochi::NearEqual(
            RoundTrip.goals[Index].baseOrInputLinkFrameFromConstraintFrame,
            Native.goals[Index].baseOrInputLinkFrameFromConstraintFrame,
            1e-5_r));
    TestTrue(
        TEXT("Attachment rigid frame round trips"),
        mochi::NearEqual(
            RoundTrip.goals[Index].linkFrameFromConstraintFrame,
            Native.goals[Index].linkFrameFromConstraintFrame,
            1e-5_r));
    // Compare only frame conversion roundoff approximately; every other authored field exactly.
    RoundTrip.goals[Index].baseOrInputLinkFrameFromConstraintFrame =
        Native.goals[Index].baseOrInputLinkFrameFromConstraintFrame;
    RoundTrip.goals[Index].linkFrameFromConstraintFrame =
        Native.goals[Index].linkFrameFromConstraintFrame;
  }
  RoundTrip.outputBaseFrameFromInputBaseFrame = Native.outputBaseFrameFromInputBaseFrame;
  TestTrue(
      TEXT("Every native v9 field round trips"),
      superdex::robotics::SerializeIKConfig(RoundTrip) ==
          superdex::robotics::SerializeIKConfig(Native));
  TestTrue(
      TEXT("Goal translation converts metres to centimetres once"),
      FMath::IsNearlyEqual(
          Asset->Goals[0].BaseOrInputLinkFrameFromConstraintFrame.GetTranslation().Size(),
          100.0 *
              mochi::Norm(Native.goals[0].baseOrInputLinkFrameFromConstraintFrame.GetTranslation()),
          1e-3));
  TestTrue(
      TEXT("Attachment translation converts metres to centimetres once"),
      FMath::IsNearlyEqual(
          Asset->Goals[0].LinkFrameFromConstraintFrame.GetTranslation().Size(),
          100.0 * mochi::Norm(Native.goals[0].linkFrameFromConstraintFrame.GetTranslation()),
          1e-3));
  TestFalse(
      TEXT("Nonidentity rotation retained"),
      Asset->Goals[0].BaseOrInputLinkFrameFromConstraintFrame.GetRotation().Equals(
          FQuat::Identity));
  Asset->Goals[0].BaseOrInputLinkFrameFromConstraintFrame.SetScale3D(FVector(2, 3, 4));
  Asset->Goals[0].LinkFrameFromConstraintFrame.SetScale3D(FVector(5, 6, 7));
  TestTrue(
      TEXT("Rigid goal conversion ignores Unreal scale"),
      mochi::NearEqual(
          Asset->ToNative().goals[0].baseOrInputLinkFrameFromConstraintFrame,
          Native.goals[0].baseOrInputLinkFrameFromConstraintFrame,
          1e-5_r));
  TestTrue(
      TEXT("Rigid attachment conversion ignores Unreal scale"),
      mochi::NearEqual(
          Asset->ToNative().goals[0].linkFrameFromConstraintFrame,
          Native.goals[0].linkFrameFromConstraintFrame,
          1e-5_r));
  FString Diagnostic;
  TestTrue(
      TEXT("Unresolved references still validate structurally without guessing model width"),
      Asset->Validate(Diagnostic));
  TestNull(
      TEXT("Native conversion does not resolve source files at runtime"), Asset->OutputBot.Get());
  TestNull(
      TEXT("Native conversion does not resolve input-bot files at runtime"), Asset->InputBot.Get());
  Asset->JointCouplings[0].Ratio = std::numeric_limits<double>::infinity();
  TestFalse(TEXT("Nonfinite coupling conversion rejected"), Asset->Validate(Diagnostic));
  Asset->FromNative(superdex::robotics::BotIKConfig{});
  TestEqual(
      TEXT("FromNative replaces rather than appends coordinates"),
      Asset->OutputBotConfiguration.Num(),
      0);
  TestTrue(
      TEXT("FromNative clears absent input-bot references"), Asset->InputBotReference.IsEmpty());
  FSuperDexBotIKCreateParams Defaults;
  TestTrue(TEXT("Config defaults fix the base"), Defaults.bFixedBase);
  TestEqual(
      TEXT("Config defaults retain the solve budget"),
      Defaults.SolverParams.MaxElapsedTimeSeconds,
      0.01);
  return VerifyLegacyV8Fixture(*this) && RecordColdLoadedConfigs(*this);
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexBotIKImportTest,
    "SuperDex.BotIK.ImportReimport",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FSuperDexBotIKImportTest::RunTest(const FString& Parameters) {
  FIKFixture Fixture;
  const FString RobotPath = FPaths::Combine(Fixture.Directory, TEXT("robot.superdex_bot"));
  auto* Bot = Fixture.MakeBot(TEXT("Bot"), RobotPath);
  TStrongObjectPtr<USuperDexBotIKConfigFactory> Factory(NewObject<USuperDexBotIKConfigFactory>());
  bool bCanceled = true;
  const FString ConfigPath = Fixture.WriteConfig(TEXT("./sub/../robot.superdex_bot"));
  TStrongObjectPtr<USuperDexBotIKConfig> Asset(
      Cast<USuperDexBotIKConfig>(Factory->FactoryCreateFile(
          USuperDexBotIKConfig::StaticClass(),
          GetTransientPackage(),
          NAME_None,
          RF_Transient,
          ConfigPath,
          TEXT(""),
          GWarn,
          bCanceled)));
  if (!TestNotNull(TEXT("File import creates the config"), Asset.Get())) {
    return false;
  }
  TestFalse(TEXT("Successful import is not canceled"), bCanceled);
  TestTrue(TEXT("Exactly one source match assigns the bot"), Asset->OutputBot == Bot);
  TestTrue(
      TEXT("Source path is normalized"), FPaths::IsSamePath(Asset->SourceOutputBotPath, RobotPath));
  if (!TestEqual(TEXT("Import retains one goal"), Asset->Goals.Num(), 1)) {
    return false;
  }
  TestEqual(TEXT("Import retains joint couplings"), Asset->JointCouplings.Num(), 1);
  TestFalse(TEXT("Import retains disabled coupling"), Asset->JointCouplings[0].bEnabled);
  TestEqual(
      TEXT("Import retains ratio direction"),
      Asset->JointCouplings[0].JointA,
      FString(TEXT("pip_joint")));
  const mochi::TransformRT ExpectedFrame{
      mochi::Quaternion::FromRotationVector(mochi::Real3{0.2_r, -0.3_r, 0.4_r}),
      mochi::Real3{0.8_r, 0.6_r, 0_r}};
  TestTrue(
      TEXT("Import retains the authored rigid frame"),
      mochi::NearEqual(
          Asset->Goals[0].ToNative().baseOrInputLinkFrameFromConstraintFrame,
          ExpectedFrame,
          1e-5_r));
  auto* ManualBot = Fixture.MakeBot(
      TEXT("Manual"), FPaths::Combine(Fixture.Directory, TEXT("manual.superdex_bot")));
  Asset->OutputBot = ManualBot;
  Fixture.WriteConfig(TEXT("robot.superdex_bot"), 321.0);
  TStrongObjectPtr<UReimportSuperDexBotIKConfigFactory> Reimport(
      NewObject<UReimportSuperDexBotIKConfigFactory>());
  TestEqual(TEXT("Reimport succeeds"), Reimport->Reimport(Asset.Get()), EReimportResult::Succeeded);
  TestTrue(TEXT("Manual bot is preserved for the same source"), Asset->OutputBot == ManualBot);
  TestEqual(TEXT("Reimport replaces authored settings"), Asset->Goals[0].PositionStiffness, 321.0);
  TestEqual(TEXT("Reimport replaces coupling tuning"), Asset->JointCouplings[0].Stiffness, 321.0);
  TestTrue(
      TEXT("Reimport retains the authored rigid frame"),
      mochi::NearEqual(
          Asset->Goals[0].ToNative().baseOrInputLinkFrameFromConstraintFrame,
          ExpectedFrame,
          1e-5_r));
  const auto BeforeFailedReimport = superdex::robotics::SerializeIKConfig(Asset->ToNative());
  FFileHelper::SaveStringToFile(TEXT("{broken"), *ConfigPath);
  TestEqual(
      TEXT("Malformed reimport fails"), Reimport->Reimport(Asset.Get()), EReimportResult::Failed);
  TestTrue(TEXT("Failed parsing preserves the manual bot"), Asset->OutputBot == ManualBot);
  TestTrue(
      TEXT("Failed parsing preserves all exposed settings and goals"),
      superdex::robotics::SerializeIKConfig(Asset->ToNative()) == BeforeFailedReimport);
  Fixture.WriteConfig(TEXT("manual.superdex_bot"));
  TestEqual(
      TEXT("Changed source reimport succeeds"),
      Reimport->Reimport(Asset.Get()),
      EReimportResult::Succeeded);
  TestTrue(TEXT("Changed source resolves its bot"), Asset->OutputBot == ManualBot);
  Fixture.MakeBot(TEXT("Duplicate"), RobotPath);
  Fixture.WriteConfig(TEXT("robot.superdex_bot"));
  TestTrue(
      TEXT("Ambiguous bot import succeeds"), Factory->ImportConfig(Asset.Get(), ConfigPath, false));
  TestNull(TEXT("Ambiguous bot stays unset"), Asset->OutputBot.Get());
  Fixture.WriteConfig(TEXT("missing.superdex_bot"));
  TestTrue(
      TEXT("Missing bot import succeeds"), Factory->ImportConfig(Asset.Get(), ConfigPath, false));
  TestNull(TEXT("Missing bot stays unset"), Asset->OutputBot.Get());
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexBotIKFullImportTest,
    "SuperDex.BotIK.FullConfigImportReimport",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FSuperDexBotIKFullImportTest::RunTest(const FString&) {
  FIKFixture Fixture;
  const FString RobotPath = FPaths::Combine(Fixture.Directory, TEXT("robot.superdex_bot"));
  const FString InputBotPath = FPaths::Combine(Fixture.Directory, TEXT("reference.superdex_bot"));
  auto* Bot = Fixture.MakeBot(TEXT("Source"), RobotPath);
  auto* InputBot = Fixture.MakeBot(TEXT("InputBot"), InputBotPath);
  TStrongObjectPtr<USuperDexBotIKConfigFactory> Factory(NewObject<USuperDexBotIKConfigFactory>());
  TStrongObjectPtr<USuperDexBotIKConfig> Asset(NewObject<USuperDexBotIKConfig>());
  const FString ConfigPath = Fixture.WriteConfig(
      TEXT("./sub/../robot.superdex_bot"), 100.0, TEXT("./sub/../reference.superdex_bot"));
  if (!TestTrue(
          TEXT("Full config imports"), Factory->ImportConfig(Asset.Get(), ConfigPath, false))) {
    return false;
  }
  TestTrue(
      TEXT("Both bots resolve by source-file tags"),
      Asset->OutputBot == Bot && Asset->InputBot == InputBot);
  TestTrue(
      TEXT("Both normalized source paths retained"),
      FPaths::IsSamePath(Asset->SourceOutputBotPath, RobotPath) &&
          FPaths::IsSamePath(Asset->SourceInputBotPath, InputBotPath));
  TestEqual(
      TEXT("Source seed retained"), Asset->OutputBotConfiguration[0], static_cast<double>(0.4_r));
  TestEqual(
      TEXT("Goal coordinates retained"),
      Asset->InputBotConfiguration[0],
      static_cast<double>(0.6_r));
  TestEqual(TEXT("Scale retained"), Asset->InputBotScale, 1.25);
  TestNull(
      TEXT("Legacy recording association is not a reflected asset property"),
      FindFProperty<FProperty>(USuperDexBotIKConfig::StaticClass(), TEXT("TargetBotRecording")));
  TestNull(
      TEXT("No new input-bot recording association is introduced"),
      FindFProperty<FProperty>(USuperDexBotIKConfig::StaticClass(), TEXT("ReferenceBotRecording")));
  TestEqual(TEXT("Goal link binding retained"), Asset->Goals[0].InputBotLink, FString(TEXT("tip")));
  TestFalse(
      TEXT("Attachment translation retained"),
      Asset->Goals[0].LinkFrameFromConstraintFrame.GetTranslation().IsNearlyZero());
  TestFalse(
      TEXT("Attachment rotation retained"),
      Asset->Goals[0].LinkFrameFromConstraintFrame.GetRotation().Equals(FQuat::Identity));
  TestFalse(
      TEXT("Root alignment retained"),
      Asset->OutputBaseFrameFromInputBaseFrame.Equals(FTransform::Identity));
  auto ImportedNative = Asset->ToNative();
  FString SourceText;
  FFileHelper::LoadFileToString(SourceText, *ConfigPath);
  superdex::robotics::BotIKConfig Expected;
  mochi::DynamicString NativeDiagnostic;
  if (!TestTrue(
          TEXT("Fixture native parses"),
          superdex::robotics::ParseIKConfig(
              TCHAR_TO_UTF8(*SourceText), Expected, NativeDiagnostic))) {
    return false;
  }
  TestTrue(
      TEXT("Root alignment import has no double scale/unit conversion"),
      mochi::NearEqual(
          ImportedNative.outputBaseFrameFromInputBaseFrame,
          Expected.outputBaseFrameFromInputBaseFrame,
          1e-5_r));
  TestTrue(
      TEXT("Attachment import has no double scale/unit conversion"),
      mochi::NearEqual(
          ImportedNative.goals[0].linkFrameFromConstraintFrame,
          Expected.goals[0].linkFrameFromConstraintFrame,
          1e-5_r));
  TestTrue(
      TEXT("Bound goal import has no double scale/unit conversion"),
      mochi::NearEqual(
          ImportedNative.goals[0].baseOrInputLinkFrameFromConstraintFrame,
          Expected.goals[0].baseOrInputLinkFrameFromConstraintFrame,
          1e-5_r));
  ImportedNative.outputBot = Expected.outputBot;
  ImportedNative.inputBot = Expected.inputBot;
  ImportedNative.outputBaseFrameFromInputBaseFrame = Expected.outputBaseFrameFromInputBaseFrame;
  ImportedNative.goals[0].baseOrInputLinkFrameFromConstraintFrame =
      Expected.goals[0].baseOrInputLinkFrameFromConstraintFrame;
  ImportedNative.goals[0].linkFrameFromConstraintFrame =
      Expected.goals[0].linkFrameFromConstraintFrame;
  TestTrue(
      TEXT("Complete imported v9 config round trips"),
      superdex::robotics::SerializeIKConfig(ImportedNative) ==
          superdex::robotics::SerializeIKConfig(Expected));

  auto* ManualSource = Fixture.MakeBot(
      TEXT("ManualSource"), FPaths::Combine(Fixture.Directory, TEXT("manual_source.superdex_bot")));
  auto* ManualReferenceBot = Fixture.MakeBot(
      TEXT("ManualReferenceBot"),
      FPaths::Combine(Fixture.Directory, TEXT("manual_reference.superdex_bot")));
  Asset->OutputBot = ManualSource;
  Asset->InputBot = ManualReferenceBot;
  Fixture.WriteConfig(TEXT("robot.superdex_bot"), 321.0, TEXT("reference.superdex_bot"));
  TestTrue(TEXT("Full reimport succeeds"), Factory->ImportConfig(Asset.Get(), ConfigPath, true));
  TestTrue(
      TEXT("Both manual assignments preserved independently for unchanged references"),
      Asset->OutputBot == ManualSource && Asset->InputBot == ManualReferenceBot);
  TestEqual(TEXT("Full reimport updates tuning"), Asset->Goals[0].PositionStiffness, 321.0);
  Fixture.WriteConfig(TEXT("robot.superdex_bot"), 100.0, TEXT("manual_reference.superdex_bot"));
  TestTrue(
      TEXT("Changed input-bot reference reimports"),
      Factory->ImportConfig(Asset.Get(), ConfigPath, true));
  TestTrue(
      TEXT("Changed input-bot resolves without clobbering manual source"),
      Asset->InputBot == ManualReferenceBot && Asset->OutputBot == ManualSource);
  Fixture.WriteConfig(
      TEXT("manual_source.superdex_bot"), 100.0, TEXT("manual_reference.superdex_bot"));
  TestTrue(
      TEXT("Changed source reference reimports"),
      Factory->ImportConfig(Asset.Get(), ConfigPath, true));
  TestTrue(
      TEXT("Changed source resolves without clobbering goal"),
      Asset->OutputBot == ManualSource && Asset->InputBot == ManualReferenceBot);

  const auto BeforeFailure = superdex::robotics::SerializeIKConfig(Asset->ToNative());
  const FString OldSourcePath = Asset->SourceOutputBotPath;
  const FString OldReferenceBotPath = Asset->SourceInputBotPath;
  const FString OldImportData = Asset->AssetImportData->GetSourceData().ToJson();
  for (
      const FString& Invalid :
      {FString(TEXT("{broken")),
       FString(
           TEXT("{\"version\":8,\"bot\":\"changed.superdex_bot\",\"targets\":[],\"goals\":[]}")),
       FString(TEXT(
           "{\"version\":8,\"bot\":\"changed.superdex_bot\",\"targetBot\":\"old.superdex_bot\",\"referenceBot\":\"new.superdex_bot\"}")),
       FString(TEXT(
           "{\"version\":8,\"bot\":\"changed.superdex_bot\",\"goals\":[{\"name\":\"goal\",\"link\":\"tip\",\"targetFrame\":{}}]}")),
       FString(TEXT(
           "{\"version\":7,\"bot\":\"changed.superdex_bot\",\"referenceBot\":\"new.superdex_bot\"}")),
       FString(TEXT(
           "{\"version\":8,\"bot\":\"changed.superdex_bot\",\"referenceBotRecording\":\"old.rrd\"}")),
       FString(TEXT(
           "{\"version\":7,\"bot\":\"changed.superdex_bot\",\"targetBotRecording\":\"old.rrd\"}")),
       FString(TEXT("{\"version\":6,\"bot\":\"changed.superdex_bot\",\"targetBotScale\":-1}")),
       FString(TEXT(
           "{\"version\":6,\"bot\":\"changed.superdex_bot\",\"targetBotConfiguration\":[1e100]}")),
       FString(TEXT(
           "{\"version\":6,\"bot\":\"changed.superdex_bot\",\"baseFromTargetBot\":{\"unknown\":1}}")),
       FString(TEXT(
           "{\"version\":6,\"bot\":\"changed.superdex_bot\",\"targets\":[{\"name\":\"goal\",\"link\":\"tip\",\"positionStiffness\":1e-100}]}"))}) {
    FFileHelper::SaveStringToFile(Invalid, *ConfigPath);
    TestFalse(
        TEXT("Malformed full document rejects transaction"),
        Factory->ImportConfig(Asset.Get(), ConfigPath, true));
    TestTrue(
        TEXT("Failed import preserves every native setting"),
        superdex::robotics::SerializeIKConfig(Asset->ToNative()) == BeforeFailure);
    TestTrue(
        TEXT("Failed import preserves both manual assets and source paths"),
        Asset->OutputBot == ManualSource && Asset->InputBot == ManualReferenceBot &&
            Asset->SourceOutputBotPath == OldSourcePath &&
            Asset->SourceInputBotPath == OldReferenceBotPath);
    TestEqual(
        TEXT("Failed import preserves source-file provenance"),
        Asset->AssetImportData->GetSourceData().ToJson(),
        OldImportData);
  }
  Fixture.MakeBot(TEXT("DuplicateReferenceBot"), InputBotPath);
  Fixture.WriteConfig(TEXT("robot.superdex_bot"), 100.0, TEXT("reference.superdex_bot"));
  TestTrue(
      TEXT("Ambiguous input-bot remains importable for repair"),
      Factory->ImportConfig(Asset.Get(), ConfigPath, false));
  TestTrue(
      TEXT("Ambiguous input-bot never auto-picks and does not drop source"),
      !Asset->InputBot && Asset->OutputBot == Bot);
  FString Diagnostic;
  TestTrue(
      TEXT("Unresolved input-bot reference preserves structurally valid bindings"),
      Asset->Validate(Diagnostic));
  TestFalse(
      TEXT("Native unresolved input-bot reference is nonempty"),
      Asset->ToNative().inputBot.empty());
  TestEqual(
      TEXT("Unresolved binding retained"), Asset->Goals[0].InputBotLink, FString(TEXT("tip")));
  Fixture.WriteConfig(
      TEXT("missing_source.superdex_bot"), 100.0, TEXT("missing_reference.superdex_bot"));
  TestTrue(
      TEXT("Both missing references import for repair"),
      Factory->ImportConfig(Asset.Get(), ConfigPath, false));
  TestTrue(TEXT("Missing assets stay unset"), !Asset->OutputBot && !Asset->InputBot);
  TestTrue(
      TEXT("Missing references remain structurally valid without filesystem/runtime resolution"),
      Asset->Validate(Diagnostic));
  auto* RecoveredSource = Fixture.MakeBot(
      TEXT("RecoveredSource"),
      FPaths::Combine(Fixture.Directory, TEXT("missing_source.superdex_bot")));
  auto* RecoveredReferenceBot = Fixture.MakeBot(
      TEXT("RecoveredReferenceBot"),
      FPaths::Combine(Fixture.Directory, TEXT("missing_reference.superdex_bot")));
  TestTrue(
      TEXT("Unchanged unresolved references retry resolution on reimport"),
      Factory->ImportConfig(Asset.Get(), ConfigPath, true));
  TestTrue(
      TEXT("Newly imported bots repair both null assignments"),
      Asset->OutputBot == RecoveredSource && Asset->InputBot == RecoveredReferenceBot);
  Fixture.WriteConfig(TEXT("robot.superdex_bot"));
  TestTrue(
      TEXT("Legacy unbound document reimports"),
      Factory->ImportConfig(Asset.Get(), ConfigPath, true));
  TestTrue(
      TEXT("Removing input bot clears all reference-only provenance/configuration"),
      !Asset->InputBot && Asset->InputBotReference.IsEmpty() &&
          Asset->SourceInputBotPath.IsEmpty() && Asset->InputBotConfiguration.IsEmpty() &&
          Asset->Goals[0].InputBotLink.IsEmpty());
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexBotIKValidationTest,
    "SuperDex.BotIK.Validation",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FSuperDexBotIKValidationTest::RunTest(const FString&) {
  TStrongObjectPtr<USuperDexBotIKConfig> Asset(NewObject<USuperDexBotIKConfig>());
  Asset->OutputBotReference = TEXT("robot.superdex_bot");
  Asset->InputBotReference = TEXT("reference.superdex_bot");
  FSuperDexBotIKGoalConfig Goal;
  Goal.Name = TEXT("Goal");
  Goal.OutputBotLink = TEXT("tip");
  Goal.InputBotLink = TEXT("tip");
  Asset->Goals.Add(Goal);
  FSuperDexBotIKJointCouplingConfig Coupling;
  Coupling.JointA = TEXT("jointA");
  Coupling.JointB = TEXT("jointB");
  Asset->JointCouplings.Add(Coupling);
  FString Diagnostic;
  TestTrue(
      TEXT("Valid unresolved generic config has no guessed model width"),
      Asset->Validate(Diagnostic));
  for (double Invalid :
       {std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::max(),
        std::numeric_limits<double>::denorm_min()}) {
    Asset->OutputBotConfiguration = {Invalid};
    TestFalse(TEXT("Invalid source coordinate rejects narrowing"), Asset->Validate(Diagnostic));
    Asset->OutputBotConfiguration.Reset();
    Asset->InputBotConfiguration = {Invalid};
    TestFalse(TEXT("Invalid goal coordinate rejects narrowing"), Asset->Validate(Diagnostic));
    Asset->InputBotConfiguration.Reset();
    Asset->InputBotScale = Invalid;
    TestFalse(TEXT("Invalid input-bot scale rejected"), Asset->Validate(Diagnostic));
    Asset->InputBotScale = 1.25;
    Asset->Goals[0].PositionStiffness = Invalid;
    TestFalse(
        TEXT("Invalid position stiffness rejected without disabling goal"),
        Asset->Validate(Diagnostic));
    Asset->Goals[0].PositionStiffness = 100;
    Asset->Goals[0].RotationStiffness = Invalid;
    TestFalse(TEXT("Invalid rotation stiffness rejected"), Asset->Validate(Diagnostic));
    Asset->Goals[0].RotationStiffness = 100;
    Asset->JointCouplings[0].Ratio = Invalid;
    TestFalse(TEXT("Invalid ratio rejected"), Asset->Validate(Diagnostic));
    Asset->JointCouplings[0].Ratio = 1;
    Asset->JointCouplings[0].Stiffness = Invalid;
    TestFalse(TEXT("Invalid coupling stiffness rejected"), Asset->Validate(Diagnostic));
    Asset->JointCouplings[0].Stiffness = 10;
    Asset->CreateParams.SolverParams.AbsTol = Invalid;
    TestFalse(TEXT("Invalid native tolerance rejected"), Asset->Validate(Diagnostic));
    Asset->CreateParams.SolverParams.AbsTol = 0.001;
    Asset->Goals[0].LinkFrameFromConstraintFrame.SetTranslation(FVector(Invalid, 0, 0));
    TestFalse(TEXT("Invalid attachment translation rejected"), Asset->Validate(Diagnostic));
    Asset->Goals[0].LinkFrameFromConstraintFrame = FTransform::Identity;
  }
  Asset->OutputBaseFrameFromInputBaseFrame.SetScale3D(FVector(1, 2, 3));
  TestFalse(TEXT("Nonuniform root scale is not silently flattened"), Asset->Validate(Diagnostic));
  Asset->OutputBaseFrameFromInputBaseFrame = FTransform::Identity;
  Asset->OutputBaseFrameFromInputBaseFrame.SetRotation(FQuat(0, 0, 0, 2));
  TestFalse(
      TEXT("Nonunit root rotation rejected rather than repaired"), Asset->Validate(Diagnostic));
  Asset->OutputBaseFrameFromInputBaseFrame = FTransform::Identity;
  Asset->Goals[0].LinkFrameFromConstraintFrame.SetRotation(FQuat(0, 0, 0, 2));
  TestFalse(
      TEXT("Nonunit attachment rotation rejected rather than repaired"),
      Asset->Validate(Diagnostic));
  Asset->Goals[0].LinkFrameFromConstraintFrame = FTransform::Identity;
  Asset->InputBotReference.Reset();
  TestFalse(TEXT("Binding without input-bot reference rejected"), Asset->Validate(Diagnostic));
  Asset->InputBotReference = TEXT("reference.superdex_bot");
  Asset->OutputBotConfiguration = {0.1, 0.2, 0.3};
  Asset->InputBotConfiguration = {0.4};
  TestTrue(
      TEXT("Generic validation accepts finite coordinates without a model-width guess"),
      Asset->Validate(Diagnostic));
  Asset->Goals[0].PositionStiffness = 0;
  Asset->Goals[0].RotationStiffness = 0;
  Asset->CreateParams.SolverParams.AbsTol = 0;
  Asset->CreateParams.SolverParams.MaxElapsedTimeSeconds = 0;
  Asset->JointCouplings[0].Stiffness = 0;
  TestTrue(
      TEXT("Explicit zero tolerances/stiffness/unlimited budget remain legal"),
      Asset->Validate(Diagnostic));
  TestTrue(TEXT("Successful validation clears stale diagnostic"), Diagnostic.IsEmpty());
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexBotIKComponentTest,
    "SuperDex.BotIK.Component",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FSuperDexBotIKComponentTest::RunTest(const FString& Parameters) {
  FIKFixture Fixture;
  for (bool bFreeRoot : {false, true}) {
    auto* Bot = Fixture.MakeBot(bFreeRoot ? TEXT("Free") : TEXT("Welded"), TEXT(""), bFreeRoot);
    for (bool bFixedBase : {false, true}) {
      TStrongObjectPtr<USuperDexBotIKConfig> Config(NewObject<USuperDexBotIKConfig>());
      Config->OutputBot = Bot;
      Config->CreateParams.bFixedBase = bFixedBase;
      Config->CreateParams.SolverParams.MaxIter = 100;
      Config->CreateParams.SolverParams.MaxElapsedTimeSeconds = 0;
      FSuperDexBotIKGoalConfig Goal;
      Goal.Name = TEXT("Goal");
      Goal.OutputBotLink = TEXT("tip");
      Goal.BaseOrInputLinkFrameFromConstraintFrame =
          FTransform(FQuat::Identity, FVector(100, 0, 0));
      Goal.RotationStiffness = 0;
      Config->Goals.Add(Goal);
      TStrongObjectPtr<USuperDexBotIKComponent> Component(NewObject<USuperDexBotIKComponent>());
      Component->Config = Config.Get();
      const FTransform Anchor(
          FQuat(FVector::UpVector, 0.4), FVector(200, -300, 100), FVector(2, 3, 4));
      Component->SetWorldTransform(Anchor);
      if (!TestTrue(
              TEXT("Initialize fixed/free and welded/free roots"), Component->InitializeSolver())) {
        return false;
      }
      TestEqual(
          TEXT("Joint-only seed never drops welded-root joints"),
          Component->GetSolvedJointPositions().Num(),
          1);
      TestEqual(
          TEXT("Seed preserves the default pose"),
          Component->GetSolvedJointPositions()[0],
          static_cast<double>(0.25_r));
      bool bReached = false;
      TestTrue(TEXT("Component-space solve succeeds"), Component->SolveIK(bReached));
      const auto LocalResult = Component->GetSolvedLinkTransforms();
      TestEqual(TEXT("All links are returned"), LocalResult.Num(), 3);
      TestTrue(
          TEXT("Result ignores scale"), LocalResult[0].GetScale3D().Equals(FVector::OneVector));
      FTransform RigidAnchor(Anchor.GetRotation(), Anchor.GetTranslation());
      TestTrue(
          TEXT("World-space update accepted"),
          Component->SetGoalTransform(
              TEXT("Goal"),
              Goal.BaseOrInputLinkFrameFromConstraintFrame * RigidAnchor,
              ESuperDexBotIKGoalSpace::World));
      TestTrue(TEXT("Equivalent world-space solve succeeds"), Component->SolveIK(bReached));
      TestTrue(
          TEXT("World/component goal equivalence"),
          Component->GetSolvedLinkTransforms()[2].Equals(LocalResult[2], 0.1));
      RigidAnchor.AddToTranslation(FVector(25, 10, 0));
      Component->SetWorldTransform(RigidAnchor);
      TestTrue(TEXT("Moving component still solves world goal"), Component->SolveIK(bReached));
      TestFalse(TEXT("Names are case-sensitive"), Component->SetGoalEnabled(TEXT("goal"), true));
      TestFalse(
          TEXT("Negative stiffness rejected"), Component->SetGoalStiffness(TEXT("Goal"), -1, 0));
      TestFalse(
          TEXT("Nonfinite transform rejected"),
          Component->SetGoalTransform(
              TEXT("Goal"),
              FTransform(FVector(std::numeric_limits<double>::infinity(), 0, 0)),
              ESuperDexBotIKGoalSpace::World));
      TestTrue(TEXT("Disable update accepted"), Component->SetGoalEnabled(TEXT("Goal"), false));
      const auto Previous = Component->GetSolvedJointPositions();
      TestFalse(TEXT("No active goals rejects the solve"), Component->SolveIK(bReached));
      TestTrue(
          TEXT("Rejected solve preserves results"),
          Component->GetSolvedJointPositions() == Previous);
      TestTrue(TEXT("Reset restores authored goals"), Component->ResetGoalsToConfig());
      TestTrue(TEXT("Reset goals can solve"), Component->SolveIK(bReached));
      TestTrue(
          TEXT("Unreachable update accepted"),
          Component->SetGoalTransform(
              TEXT("Goal"), FTransform(FVector(100000, 0, 0)), ESuperDexBotIKGoalSpace::Component));
      TestTrue(
          TEXT("Unreachable goal still gives an error-free solve"), Component->SolveIK(bReached));
      if (bFixedBase || !bFreeRoot) {
        TestFalse(TEXT("Unreachable fixed-root goal reports nonconvergence"), bReached);
      }
      TStrongObjectPtr<USuperDexBotIKComponent> Independent(NewObject<USuperDexBotIKComponent>());
      Independent->Config = Config.Get();
      TestTrue(TEXT("Second instance initializes independently"), Independent->InitializeSolver());
      TestEqual(
          TEXT("Second instance starts at default pose"),
          Independent->GetSolvedJointPositions()[0],
          static_cast<double>(0.25_r));
      for (int32 Iteration = 0; Iteration < 3; ++Iteration) {
        TestTrue(
            TEXT("Repeated reinitialization releases the old solver"),
            Component->InitializeSolver());
      }
      Independent->DestroyComponent();
      Component->DestroyComponent();
      Config->Goals.Add(Goal);
      TStrongObjectPtr<USuperDexBotIKComponent> Invalid(NewObject<USuperDexBotIKComponent>());
      Invalid->Config = Config.Get();
      TestFalse(TEXT("Duplicate goals reject initialization"), Invalid->InitializeSolver());
      Config->Goals.Pop();
      Config->Goals[0].OutputBotLink = TEXT("missing");
      TestFalse(TEXT("Missing link rejects initialization"), Invalid->InitializeSolver());
    }
  }
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexBotIKAttachmentTest,
    "SuperDex.BotIK.ComponentAttachments",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FSuperDexBotIKAttachmentTest::RunTest(const FString&) {
  FIKFixture Fixture;
  auto* Bot = Fixture.MakeBot(TEXT("Attached"), TEXT(""));
  TStrongObjectPtr<USuperDexBotIKConfig> Config(NewObject<USuperDexBotIKConfig>());
  Config->OutputBot = Bot;
  Config->OutputBotConfiguration = {0.1};
  Config->CreateParams.SolverParams.MaxIter = 100;
  Config->CreateParams.SolverParams.MaxElapsedTimeSeconds = 0;
  const mochi::TransformRT Attachment{
      mochi::Quaternion::RotationZ(0.2_r), mochi::Real3{0.2_r, 0.1_r, 0_r}};
  const auto Prefab = MakeRobot(false);
  const mochi::DynamicArray<mochi::real> ExpectedPositions{0.7_r};
  mochi::DynamicArray<mochi::TransformRT> ExpectedLinks;
  mochi::Error Error;
  superdex::robotics::ComputeLinkTransformsFromPose(
      Prefab,
      ExpectedPositions,
      ExpectedLinks,
      superdex::robotics::LinkTransformSpace::RootFromParent,
      Error);
  if (!TestTrue(TEXT("Prepare expected attachment FK"), Error.IsOK())) {
    return false;
  }
  FMochiToUnrealConverter Converter(kSuperDexSpace);
  FSuperDexBotIKGoalConfig Goal;
  Goal.Name = TEXT("AttachmentGoal");
  Goal.OutputBotLink = TEXT("tip");
  Goal.LinkFrameFromConstraintFrame = Converter.TransformToUnreal(Attachment);
  Goal.BaseOrInputLinkFrameFromConstraintFrame =
      Converter.TransformToUnreal(ExpectedLinks[2] * Attachment);
  Config->Goals.Add(Goal);
  TStrongObjectPtr<USuperDexBotIKComponent> Component(NewObject<USuperDexBotIKComponent>());
  Component->Config = Config.Get();
  const FTransform Anchor(FQuat(FVector::UpVector, 0.4), FVector(200, -300, 100));
  Component->SetWorldTransform(Anchor);
  if (!TestTrue(TEXT("Full attachment config initializes"), Component->InitializeSolver())) {
    return false;
  }
  TestEqual(
      TEXT("Full config seed honored once instead of prefab default"),
      Component->GetSolvedJointPositions()[0],
      static_cast<double>(static_cast<mochi::real>(0.1)));
  bool bReached = false;
  if (!TestTrue(
          TEXT("Attachment-frame component-space solve succeeds"), Component->SolveIK(bReached))) {
    return false;
  }
  TestTrue(
      TEXT("Attachment, not link origin, reaches the authored goal"),
      FMath::Abs(Component->GetSolvedJointPositions()[0] - 0.7) < 1e-3);
  TestTrue(
      TEXT("Solved link transform retains component/world contract"),
      Component->GetSolvedLinkTransforms()[2].Equals(
          Converter.TransformToUnreal(ExpectedLinks[2]) * Anchor, 0.1));
  TestTrue(
      TEXT("World-space attachment goal update accepted"),
      Component->SetGoalTransform(
          Goal.Name,
          Goal.BaseOrInputLinkFrameFromConstraintFrame * Anchor,
          ESuperDexBotIKGoalSpace::World));
  TestTrue(TEXT("World-space attachment solve succeeds"), Component->SolveIK(bReached));
  TestTrue(
      TEXT("World-space update preserves local attachment semantics"),
      FMath::Abs(Component->GetSolvedJointPositions()[0] - 0.7) < 1e-3);
  TestFalse(
      TEXT("Tiny nonzero stiffness update cannot silently disable a goal"),
      Component->SetGoalStiffness(Goal.Name, 1e-100, 100));
  Component->DestroyComponent();

  Config->InputBotReference = TEXT("reference.superdex_bot");
  Config->Goals[0].InputBotLink = TEXT("tip");
  TStrongObjectPtr<USuperDexBotIKComponent> Bound(NewObject<USuperDexBotIKComponent>());
  Bound->Config = Config.Get();
  TestFalse(
      TEXT("Manual component explicitly rejects bound goal rather than flattening it"),
      Bound->InitializeSolver());
  TestTrue(
      TEXT("Rejected bound config creates no fabricated output"),
      Bound->GetSolvedJointPositions().IsEmpty());
  Config->Goals[0].InputBotLink.Reset();
  Config->OutputBotConfiguration = {0.1, 0.2};
  TestFalse(
      TEXT("Full-config creation rejects wrong source seed width"), Bound->InitializeSolver());
  Bound->DestroyComponent();
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexBotIKJointCouplingTest,
    "SuperDex.BotIK.JointCouplings",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FSuperDexBotIKJointCouplingTest::RunTest(const FString&) {
  FIKFixture Fixture;
  auto* Bot = Fixture.MakeBot(TEXT("Finger"), TEXT(""));
  auto Prefab = MakeRobot(false);
  Prefab.joints[2].type = mochi::ArticulatedJointType::Revolute;
  Prefab.defaultPose = {0.2_r, -0.1_r};
  mochi::Error Error;
  superdex::robotics::RebuildBotData(Prefab, Error);
  const auto Json = superdex::robotics::WriteToJsonString(Prefab, Error);
  if (!TestTrue(TEXT("Prepare two-joint finger"), Error.IsOK())) {
    return false;
  }
  FindFProperty<FStrProperty>(USuperDexBot::StaticClass(), TEXT("BotPrefabJson"))
      ->SetPropertyValue_InContainer(Bot, FString(UTF8_TO_TCHAR(Json.c_str())));
  TStrongObjectPtr<USuperDexBotIKConfig> Config(NewObject<USuperDexBotIKConfig>());
  Config->OutputBot = Bot;
  Config->CreateParams.SolverParams.MaxIter = 100;
  Config->CreateParams.SolverParams.MaxElapsedTimeSeconds = 0;
  Config->CreateParams.SolverParams.AbsTol = 1e-6;
  Config->CreateParams.SolverParams.RelTol = 1e-8;
  FSuperDexBotIKGoalConfig Goal;
  Goal.Name = TEXT("rotation_goal");
  Goal.OutputBotLink = TEXT("tip");
  Goal.PositionStiffness = 0;
  Goal.RotationStiffness = 100;
  FMochiToUnrealConverter Converter(kSuperDexSpace);
  Goal.BaseOrInputLinkFrameFromConstraintFrame =
      Converter.TransformToUnreal(mochi::TransformRT{mochi::Quaternion::RotationZ(0.7_r)});
  Config->Goals.Add(Goal);
  FSuperDexBotIKJointCouplingConfig Coupling;
  Coupling.JointA = TEXT("arm_joint");
  Coupling.JointB = TEXT("tip_joint");
  Coupling.Ratio = 0.7;
  Coupling.Stiffness = 100;
  Config->JointCouplings.Add(Coupling);
  TStrongObjectPtr<USuperDexBotIKComponent> Component(NewObject<USuperDexBotIKComponent>());
  Component->Config = Config.Get();
  bool bReached = false;
  if (!TestTrue(
          TEXT("Document coupling creates shared native solver"), Component->InitializeSolver()) ||
      !TestTrue(
          TEXT("Underconstrained rotational goal solves with coupling"),
          Component->SolveIK(bReached))) {
    return false;
  }
  auto Positions = Component->GetSolvedJointPositions();
  if (!TestEqual(TEXT("Finger result has two joint coordinates"), Positions.Num(), 2)) {
    return false;
  }
  TestTrue(
      TEXT("PIP/DIP ratio resolves the redundant motion"),
      FMath::Abs(Positions[1] - Coupling.Ratio * Positions[0]) < 1e-3);
  TestTrue(
      TEXT("Joint rotation sum reaches goal"),
      FMath::Abs(Positions[0] + Positions[1] - 0.7) < 1e-3);
  Component->DestroyComponent();

  // Legacy callers keep offsets and FName matching, but now use native BotIK transmissions too.
  FMochiJointCoupling Legacy;
  Legacy.JointAName = TEXT("ARM_JOINT");
  Legacy.JointBName = TEXT("tip_joint");
  Legacy.Ratio = 0.7;
  Legacy.Offset = 0.1;
  Legacy.Stiffness = 100;
  auto Bad = Legacy;
  Bad.JointAName = TEXT("missing_joint");
  FMochiIKSolverParams SolverParams;
  SolverParams.FromMochi(Config->CreateParams.ToNative().solverParams);
  TStrongObjectPtr<USuperDexIKSolverObject> LegacySolver(
      USuperDexIKSolverObject::CreateIKSolver(
          GetTransientPackage(),
          TEXT("legacy_coupling"),
          Bot,
          FTransform::Identity,
          {},
          true,
          SolverParams,
          {Bad, Legacy},
          true));
  if (!TestNotNull(
          TEXT("Legacy skips bad row and preserves valid offset coupling"), LegacySolver.Get())) {
    return false;
  }
  LegacySolver->CreateRotationTargetForLink(
      2, FQuat::Identity, Goal.BaseOrInputLinkFrameFromConstraintFrame.GetRotation(), 100);
  LegacySolver->SolveIK();
  const auto LegacyPositions = LegacySolver->GetSolvedDofs();
  if (TestEqual(TEXT("Legacy coupling has two joint coordinates"), LegacyPositions.Num(), 2)) {
    TestTrue(
        TEXT("Legacy signed offset convention remains intact"),
        FMath::Abs(LegacyPositions[1] - Legacy.Ratio * LegacyPositions[0] - Legacy.Offset) < 1e-3);
  }
  LegacySolver->DestroyMochiResources();
  Config->JointCouplings[0].JointB = TEXT("missing_joint");
  TStrongObjectPtr<USuperDexBotIKComponent> Invalid(NewObject<USuperDexBotIKComponent>());
  Invalid->Config = Config.Get();
  TestFalse(TEXT("Document missing joint rejects the whole solver"), Invalid->InitializeSolver());
  Invalid->DestroyComponent();
  return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
