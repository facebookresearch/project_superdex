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

#pragma once

#include "CoreMinimal.h"
#include "EditorReimportHandler.h"
#include "Factories/Factory.h"
#include "SuperDexImportUtils.h"

#include "SuperDexSceneFactory.generated.h"

// Imports .superdex_scene and .superdex_scene_archive files authored in SuperDex Studio as
// USuperDexScene assets.
UCLASS(HideCategories = Object)
class USuperDexSceneFactory : public UFactory {
  GENERATED_UCLASS_BODY()

  //~ Begin UFactory Interface
  virtual UObject* FactoryCreateFile(
      UClass* InClass,
      UObject* InParent,
      FName InName,
      EObjectFlags Flags,
      const FString& Filename,
      const TCHAR* Parms,
      FFeedbackContext* Warn,
      bool& bOutOperationCanceled) override;
  virtual void CleanUp() override;
  //~ End UFactory Interface

 protected:
  FSuperDexImportSession ImportSession;
};

UCLASS(HideCategories = Object)
class UReimportSuperDexSceneFactory : public USuperDexSceneFactory, public FReimportHandler {
  GENERATED_BODY()

 public:
  UReimportSuperDexSceneFactory();

  //~ Begin FReimportHandler Interface
  virtual bool CanReimport(UObject* Obj, TArray<FString>& OutFilenames) override;
  virtual void SetReimportPaths(UObject* Obj, const TArray<FString>& NewReimportPaths) override;
  virtual EReimportResult::Type Reimport(UObject* Obj) override;
  virtual int32 GetPriority() const override;
  //~ End FReimportHandler Interface
};
