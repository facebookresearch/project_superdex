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
#include "Engine/SimpleConstructionScript.h"
#include "HAL/FileManager.h"
#include "Misc/AutomationTest.h"
#include "Misc/Guid.h"
#include "Misc/Paths.h"
#include "Misc/ScopeExit.h"

#if WITH_DEV_AUTOMATION_TESTS

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexPrefabRodScaleTest,
    "SuperDexPhysics.Prefab.RejectsNonUniformPointCloudRodScale",
    EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::EngineFilter)

bool FSuperDexPrefabRodScaleTest::RunTest(const FString& Parameters) {
  for (const mochi::ColliderType collider :
       {mochi::ColliderType::PointCloud, mochi::ColliderType::Auto}) {
    const FString name = TEXT("RodScale_") + FGuid::NewGuid().ToString(EGuidFormats::Digits);
    const FString path = FPaths::ConvertRelativePathToFull(
        FPaths::Combine(FPaths::AutomationTransientDir(), name + TEXT(".superdex_prefab")));
    ON_SCOPE_EXIT {
      IFileManager::Get().Delete(*path);
    };
    mochi::prefab::ScenePrefab source;
    auto& rod = source.actors.rod.push_back();
    rod.name = "InvalidRod";
    rod.shapeFile = "missing.mochi.h5";
    rod.scale = {1_r, 2_r, 1_r};
    rod.colliderType = collider;
    mochi::ErrorLog error;
    mochi::prefab::SaveToJsonFile(source, TCHAR_TO_UTF8(*path), error);
    if (!TestTrue(TEXT("Write prefab"), error.IsOK())) {
      return false;
    }
    TStrongObjectPtr<USuperDexPrefab> prefab(
        FSuperDexPrefabImporter::CreatePrefabAsset(TEXT("/Temp/") + name, name));
    FSuperDexImportSession session;
    FSuperDexPrefabImporter importer(session);
    AddExpectedError(
        TEXT("must be uniform by absolute value"), EAutomationExpectedErrorFlags::Contains, 1);
    if (!TestTrue(
            TEXT("Import skips the invalid rod before loading its missing shape"),
            importer.Import(prefab.Get(), path, ESuperDexPrefabImportMode::Blueprint))) {
      return false;
    }
    UBlueprint* blueprint = prefab->GeneratedBlueprint.LoadSynchronous();
    if (!TestNotNull(TEXT("Blueprint"), blueprint)) {
      return false;
    }
    TestEqual(
        TEXT("Only the default root remains"),
        blueprint->SimpleConstructionScript->GetAllNodes().Num(),
        1);
  }
  return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
