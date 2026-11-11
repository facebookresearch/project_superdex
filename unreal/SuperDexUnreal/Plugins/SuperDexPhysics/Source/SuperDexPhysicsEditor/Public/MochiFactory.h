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

#include "ActorFactories/ActorFactory.h"
#include "CoreMinimal.h"
#include "EditorReimportHandler.h"
#include "Misc/EngineVersionComparison.h"
#include "UObject/Object.h"
#include "UObject/ObjectMacros.h"

#include "MochiFactory.generated.h"

//------------------------------------------------------------------------------
// MOCHI H5 (MOCHI MODEL)
//------------------------------------------------------------------------------

UCLASS(HideCategories = Object)
class UMochiH5Factory : public UFactory {
  GENERATED_UCLASS_BODY()
  virtual UObject* FactoryCreateFile(
      UClass* InClass,
      UObject* InParent,
      FName InName,
      EObjectFlags Flags,
      const FString& Filename,
      const TCHAR* Parms,
      FFeedbackContext* Warn,
      bool& bOutOperationCanceled) override;
};

// Reimports Mochi models from their source file through the editor's standard Reimport and
// source file actions.
UCLASS(HideCategories = Object)
class UReimportMochiH5Factory : public UMochiH5Factory, public FReimportHandler {
  GENERATED_UCLASS_BODY()

  //~ Begin FReimportHandler Interface
  virtual bool CanReimport(UObject* Obj, TArray<FString>& OutFilenames) override;
  virtual void SetReimportPaths(UObject* Obj, const TArray<FString>& NewReimportPaths) override;
  virtual EReimportResult::Type Reimport(UObject* Obj) override;
  virtual int32 GetPriority() const override;
  //~ End FReimportHandler Interface
};

//------------------------------------------------------------------------------
// MOCHI SOFT MESH ACTOR FACTORY
//------------------------------------------------------------------------------

UCLASS(MinimalAPI, Config = Editor, CollapseCategories, HideCategories = Object)
class UMochiSoftMeshActorFactory : public UActorFactory {
  GENERATED_UCLASS_BODY()
  //~ Begin UActorFactory Interface
  virtual bool CanCreateActorFrom(const FAssetData& AssetData, FText& OutErrorMsg) override;
  virtual void PostSpawnActor(UObject* Asset, AActor* NewActor) override;
#if UE_VERSION_OLDER_THAN(5, 4, 0)
  virtual void PostCreateBlueprint(UObject* Asset, AActor* CDO) override;
#endif
  virtual UObject* GetAssetFromActorInstance(AActor* ActorInstance) override;
  virtual FQuat AlignObjectToSurfaceNormal(
      const FVector& InSurfaceNormal,
      const FQuat& ActorRotation) const override;
  //~ End UActorFactory Interface
};

//------------------------------------------------------------------------------
// MOCHI ARTICULATION ACTOR FACTORY
//------------------------------------------------------------------------------

UCLASS(MinimalAPI, Config = Editor, CollapseCategories, HideCategories = Object)
class UMochiArticulationActorFactory : public UActorFactory {
  GENERATED_UCLASS_BODY()
  virtual bool CanCreateActorFrom(const FAssetData& AssetData, FText& OutErrorMsg) override;
  virtual void PostSpawnActor(UObject* Asset, AActor* NewActor) override;
  virtual UObject* GetAssetFromActorInstance(AActor* ActorInstance) override;
  virtual FQuat AlignObjectToSurfaceNormal(
      const FVector& InSurfaceNormal,
      const FQuat& ActorRotation) const override;
};
