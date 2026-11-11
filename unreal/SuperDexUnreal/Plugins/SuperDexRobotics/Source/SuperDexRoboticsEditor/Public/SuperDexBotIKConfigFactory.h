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
#include "SuperDexBotIKConfigFactory.generated.h"

class USuperDexBot;
class USuperDexBotIKConfig;

UCLASS()
class SUPERDEXROBOTICSEDITOR_API USuperDexBotIKConfigFactory : public UFactory {
  GENERATED_BODY()

 public:
  USuperDexBotIKConfigFactory();
  virtual UObject* FactoryCreateFile(
      UClass* InClass,
      UObject* InParent,
      FName InName,
      EObjectFlags Flags,
      const FString& Filename,
      const TCHAR* Parms,
      FFeedbackContext* Warn,
      bool& bOutOperationCanceled) override;
  virtual bool FactoryCanImport(const FString& Filename) override;

  /** Import is transactional: malformed documents leave settings, assignments and provenance
   * intact. Missing/ambiguous bot references are retained for manual repair, with no automatic
   * selection.
   */
  bool ImportConfig(USuperDexBotIKConfig* Asset, const FString& Filename, bool bReimport);
  /** Resolve either bot by its imported source-file tag, not by basename or package proximity. */
  static USuperDexBot* ResolveBotAsset(const FString& SourcePath);
  static FString NormalizeSourcePath(const FString& Path);
};

UCLASS()
class USuperDexBotIKConfigFactoryNew : public UFactory {
  GENERATED_BODY()

 public:
  USuperDexBotIKConfigFactoryNew();
  virtual UObject* FactoryCreateNew(
      UClass* InClass,
      UObject* InParent,
      FName InName,
      EObjectFlags Flags,
      UObject* Context,
      FFeedbackContext* Warn) override;
};

UCLASS()
class SUPERDEXROBOTICSEDITOR_API UReimportSuperDexBotIKConfigFactory
    : public USuperDexBotIKConfigFactory,
      public FReimportHandler {
  GENERATED_BODY()

 public:
  UReimportSuperDexBotIKConfigFactory();
  virtual bool CanReimport(UObject* Obj, TArray<FString>& OutFilenames) override;
  virtual void SetReimportPaths(UObject* Obj, const TArray<FString>& NewReimportPaths) override;
  virtual EReimportResult::Type Reimport(UObject* Obj) override;
  virtual int32 GetPriority() const override;
};
