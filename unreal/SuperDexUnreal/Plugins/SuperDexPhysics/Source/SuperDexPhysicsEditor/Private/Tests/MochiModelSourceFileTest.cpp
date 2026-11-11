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

#include "MochiModel.h"

#include "EditorReimportHandler.h"
#include "HAL/FileManager.h"
#include "Misc/AutomationTest.h"
#include "Misc/Paths.h"
#include "UObject/Package.h"

#if WITH_DEV_AUTOMATION_TESTS

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FMochiModelSourceFileTest,
    "SuperDexPhysics.MochiModel.SourceFileAndReimport",
    EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::EngineFilter)

bool FMochiModelSourceFileTest::RunTest(const FString& Parameters) {
  // Write a real Mochi model file to import from.
  UMochiModel* Source =
      LoadObject<UMochiModel>(nullptr, TEXT("/SuperDexPhysics/Models/MM_Sphere.MM_Sphere"));
  if (!TestNotNull(TEXT("The source model loads"), Source)) {
    return false;
  }
  const FString File = FPaths::ConvertRelativePathToFull(FPaths::Combine(
      FPaths::AutomationTransientDir(), TEXT("MochiModelSourceFileTest/sphere.mochi.h5")));
  if (!TestTrue(TEXT("The source model exports"), Source->ExportToFilename(File))) {
    return false;
  }

  UMochiModel* Model = NewObject<UMochiModel>(GetTransientPackage(), NAME_None, RF_Transient);
  TestNotNull(TEXT("A new model has import data"), Model->AssetImportData.Get());
  TestTrue(TEXT("The model imports the file"), Model->ImportFromFilename(File));
  TestTrue(
      TEXT("The import records the file as the source"),
      FPaths::IsSamePath(Model->GetSourceFilePath(), File));

  TArray<FString> Filenames;
  TestTrue(
      TEXT("The editor's reimport handlers accept the model"),
      FReimportManager::Instance()->CanReimport(Model, &Filenames));
  TestTrue(
      TEXT("The reimport handler reports the source file"),
      Filenames.Num() == 1 && FPaths::IsSamePath(Filenames[0], File));
  TestTrue(
      TEXT("The model reimports from its source file"),
      FReimportManager::Instance()->Reimport(
          Model, false, false, FString(), nullptr, INDEX_NONE, false, true));

  IFileManager::Get().Delete(*File);
  const FString RecordedBefore = Model->AssetImportData->GetSourceData().SourceFiles[0].RelativeFilename;
  AddExpectedError(TEXT("Source file not found"), EAutomationExpectedErrorFlags::Contains, 1);
  TestFalse(
      TEXT("Reimport fails once the source file is gone"),
      FReimportManager::Instance()->Reimport(
          Model, false, false, FString(), nullptr, INDEX_NONE, false, true));
  TestEqual(
      TEXT("A failed reimport keeps the recorded source file"),
      Model->AssetImportData->GetSourceData().SourceFiles[0].RelativeFilename,
      RecordedBefore);
  return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
