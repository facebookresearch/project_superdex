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

#include "MochiActorObject.h"
#include "MochiGameInstanceSubsystem.h"
#include "MochiPhysicsSettings.h"
#include "MochiSceneObject.h"
#include "SuperDexBotSceneSubsystem.h"

#include "Engine/Engine.h"
#include "Engine/GameInstance.h"
#include "Engine/World.h"
#include "HAL/FileManager.h"
#include "Misc/AutomationTest.h"
#include "Misc/CommandLine.h"
#include "Misc/Guid.h"
#include "Misc/Parse.h"
#include "Misc/Paths.h"
#include "Misc/ScopeExit.h"
#include "UObject/UObjectIterator.h"

#if WITH_DEV_AUTOMATION_TESTS

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexRuntimeSceneFailClosedTest,
    "SuperDexRobotics.RuntimeScene.FailClosedOnEarlyValidation",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FSuperDexRuntimeSceneFailClosedTest::RunTest(const FString& Parameters) {
  // A test game instance temporarily changes the process-wide simulation space.
  for (TObjectIterator<UMochiGameInstanceSubsystem> It; It; ++It) {
    if (!It->IsTemplate() && It->GetScene() != nullptr) {
      AddError(TEXT(
          "Run runtime scene validation in a fresh automation process without a live Mochi game instance"));
      return false;
    }
  }
  const bool bHadOverride = UMochiPhysicsSettings::HasActiveSimulationSpaceOverride();
  const FMochiSpaceDefinition OriginalSpace = UMochiPhysicsSettings::GetActiveSimulationSpace();
  ON_SCOPE_EXIT {
    if (bHadOverride) {
      UMochiPhysicsSettings::SetActiveSimulationSpaceOverride(OriginalSpace);
    } else {
      UMochiPhysicsSettings::ClearActiveSimulationSpaceOverride();
    }
  };
  const FString OriginalCommandLine = FCommandLine::Get();
  ON_SCOPE_EXIT {
    FCommandLine::Set(*OriginalCommandLine);
  };
  FString TestCommandLine;
  const TCHAR* Cursor = *OriginalCommandLine;
  FString Token;
  while (*Cursor != TEXT('\0')) {
    const TCHAR* TokenStart = Cursor;
    if (!FParse::Token(Cursor, Token, true)) {
      break;
    }
    FString SwitchName = Token;
    SwitchName.RemoveFromStart(TEXT("-"));
    FString UnusedValue;
    SwitchName.Split(TEXT("="), &SwitchName, &UnusedValue);
    const bool bSceneOrTaskSwitch =
        SwitchName.Equals(TEXT("SuperDexSceneFile"), ESearchCase::IgnoreCase) ||
        SwitchName.Equals(TEXT("MochiBotSceneFile"), ESearchCase::IgnoreCase) ||
        SwitchName.Equals(TEXT("SuperDexTaskFile"), ESearchCase::IgnoreCase) ||
        SwitchName.Equals(TEXT("MochiBotTaskFile"), ESearchCase::IgnoreCase) ||
        SwitchName.Equals(TEXT("SuperDexTaskFolder"), ESearchCase::IgnoreCase) ||
        SwitchName.Equals(TEXT("MochiBotTaskFolder"), ESearchCase::IgnoreCase);
    if (!bSceneOrTaskSwitch) {
      TestCommandLine += TEXT(" ");
      TestCommandLine.AppendChars(TokenStart, static_cast<int32>(Cursor - TokenStart));
    }
  }
  TestEqual(
      TEXT("Unattended launch flag is preserved"),
      FParse::Param(*TestCommandLine, TEXT("unattended")),
      FParse::Param(*OriginalCommandLine, TEXT("unattended")));
  FString OriginalExecCommands;
  FString TestExecCommands;
  FParse::Value(*OriginalCommandLine, TEXT("ExecCmds="), OriginalExecCommands);
  FParse::Value(*TestCommandLine, TEXT("ExecCmds="), TestExecCommands);
  TestEqual(TEXT("Quoted launch commands are preserved"), TestExecCommands, OriginalExecCommands);
  const FString EmptyFolder = FPaths::ProjectSavedDir() / TEXT("RuntimeSceneTests") /
      FGuid::NewGuid().ToString(EGuidFormats::Digits);
  if (!IFileManager::Get().MakeDirectory(*EmptyFolder, true)) {
    AddError(TEXT("Could not create the test's empty task directory"));
    return false;
  }
  ON_SCOPE_EXIT {
    IFileManager::Get().DeleteDirectory(*EmptyFolder, false, true);
  };
  const FString MissingScene = EmptyFolder / TEXT("missing.superdex_scene");
  struct FCase {
    const TCHAR* Name;
    FString Arguments;
    bool bUseFallback;
    bool bShouldFail;
  };
  const FCase Cases[] = {
      {TEXT("NoSceneRequested"), {}, false, false},
      {TEXT("MissingSceneFile"),
       FString::Printf(TEXT(" -SuperDexSceneFile=\"%s\""), *MissingScene),
       false,
       true},
      {TEXT("EmptyScenePath"), TEXT(" -SuperDexSceneFile=\"\""), false, true},
      {TEXT("EmptyTaskFolder"),
       FString::Printf(
           TEXT(" -SuperDexSceneFile=\"%s\" -SuperDexTaskFolder=\"%s\""),
           *MissingScene,
           *EmptyFolder),
       false,
       true},
      {TEXT("FallbackEmptyTaskFolder"),
       FString::Printf(TEXT(" -SuperDexTaskFolder=\"%s\""), *EmptyFolder),
       true,
       true},
  };
  AddExpectedError(
      TEXT("SuperDexBotSceneSubsystem: failed to load"),
      EAutomationExpectedErrorFlags::Contains,
      4);
  for (const FCase& Case : Cases) {
    FCommandLine::Set(*(TestCommandLine + Case.Arguments));
    UGameInstance* GameInstance = NewObject<UGameInstance>(GEngine);
    GameInstance->AddToRoot();
    GameInstance->InitializeStandalone();
    UWorld* World = GameInstance->GetWorld();
    ON_SCOPE_EXIT {
      GameInstance->Shutdown();
      if (World) {
        GEngine->DestroyWorldContext(World);
        World->DestroyWorld(false);
      }
      GameInstance->RemoveFromRoot();
    };
    USuperDexBotSceneSubsystem* Runtime = GameInstance->GetSubsystem<USuperDexBotSceneSubsystem>();
    UMochiGameInstanceSubsystem* Physics =
        GameInstance->GetSubsystem<UMochiGameInstanceSubsystem>();
    if (!TestNotNull(TEXT("Runtime scene subsystem"), Runtime) ||
        !TestNotNull(TEXT("Physics subsystem"), Physics) || !TestNotNull(TEXT("World"), World)) {
      return false;
    }
    if (Case.bUseFallback) {
      Runtime->EnsureLoaded(MissingScene, {});
    }
    UMochiSceneObject* Scene = Physics->GetScene();
    if (!TestNotNull(TEXT("Native scene"), Scene)) {
      return false;
    }
    TestEqual(
        FString(Case.Name) + TEXT(" failed state"),
        Runtime->GetState() == ESuperDexBotSceneState::Failed,
        Case.bShouldFail);
    TestEqual(
        FString(Case.Name) + TEXT(" external ownership"),
        Scene->IsContentsExternallyOwned(),
        Case.bShouldFail);
    UMochiActorObject* Actor = UMochiActorObject::CreateImplicitPlane(
        World,
        TEXT("RuntimeSceneTestPlane"),
        FTransform::Identity,
        FVector(0, 0, 1),
        0,
        nullptr,
        nullptr,
        FName(TEXT("GenericObject")),
        Scene);
    TestEqual(
        FString(Case.Name) + TEXT(" rejects level actor"), Actor == nullptr, Case.bShouldFail);
    if (Actor) {
      Actor->DestroyMochiResources();
    }
  }
  return true;
}
#endif
