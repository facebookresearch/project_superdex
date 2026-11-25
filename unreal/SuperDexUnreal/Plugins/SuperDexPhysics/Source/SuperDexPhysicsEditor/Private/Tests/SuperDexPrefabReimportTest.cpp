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

#include "SuperDexImportUtils.h"
#include "SuperDexPrefabImporter.h"

#include "Engine/Blueprint.h"
#include "Engine/SCS_Node.h"
#include "Engine/SimpleConstructionScript.h"
#include "HAL/FileManager.h"
#include "Misc/AutomationTest.h"
#include "Misc/Guid.h"
#include "Misc/Paths.h"
#include "Misc/ScopeExit.h"
#include "UObject/StrongObjectPtr.h"

#if WITH_DEV_AUTOMATION_TESTS

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexPrefabReimportNamesTest,
    "SuperDexPhysics.Prefab.ReimportKeepsComponentNames",
    EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::EngineFilter)

// Constraints find the components they connect by name, so a reimport must give the regenerated
// components the names the first import gave them.
bool FSuperDexPrefabReimportNamesTest::RunTest(const FString& Parameters) {
  const FString name = TEXT("Reimport_") + FGuid::NewGuid().ToString(EGuidFormats::Digits);
  const FString dir = FPaths::ConvertRelativePathToFull(FPaths::AutomationTransientDir());
  const FString innerPath = FPaths::Combine(dir, name + TEXT("_inner.superdex_prefab"));
  const FString outerPath = FPaths::Combine(dir, name + TEXT(".superdex_prefab"));
  ON_SCOPE_EXIT {
    IFileManager::Get().Delete(*innerPath);
    IFileManager::Get().Delete(*outerPath);
  };
  mochi::prefab::ScenePrefab outer;
  auto& inner = outer.prefabs.push_back();
  inner.name = "Inner";
  inner.path = TCHAR_TO_UTF8(*(TEXT("./") + FPaths::GetCleanFilename(innerPath)));
  mochi::ErrorLog error;
  mochi::prefab::SaveToJsonFile(mochi::prefab::ScenePrefab{}, TCHAR_TO_UTF8(*innerPath), error);
  mochi::prefab::SaveToJsonFile(outer, TCHAR_TO_UTF8(*outerPath), error);
  if (!TestTrue(TEXT("Write prefabs"), error.IsOK())) {
    return false;
  }
  TStrongObjectPtr<USuperDexPrefab> prefab(
      FSuperDexPrefabImporter::CreatePrefabAsset(TEXT("/Temp/") + name, name));
  for (const TCHAR* pass : {TEXT("Import"), TEXT("Reimport")}) {
    FSuperDexImportSession session;
    FSuperDexPrefabImporter importer(session);
    if (!TestTrue(
            pass, importer.Import(prefab.Get(), outerPath, ESuperDexPrefabImportMode::Blueprint))) {
      return false;
    }
  }
  UBlueprint* blueprint = prefab->GeneratedBlueprint.LoadSynchronous();
  if (!TestNotNull(TEXT("Blueprint"), blueprint)) {
    return false;
  }
  TArray<FName> names;
  for (const USCS_Node* node : blueprint->SimpleConstructionScript->GetAllNodes()) {
    names.Add(node->GetVariableName());
  }
  TestTrue(
      TEXT("The nested prefab's component keeps its name"), names.Contains(FName(TEXT("Inner"))));
  TestFalse(TEXT("No renamed copy"), names.Contains(FName(TEXT("Inner1"))));
  return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
