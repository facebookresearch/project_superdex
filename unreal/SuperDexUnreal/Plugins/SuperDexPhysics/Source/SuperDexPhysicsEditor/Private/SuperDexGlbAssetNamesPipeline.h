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
#include "InterchangePipelineBase.h"

#include "SuperDexGlbAssetNamesPipeline.generated.h"

// Renames the material and texture assets of a GLB import. Runs after the generic assets pipeline,
// which names them after the GLB's own material and image names; FSuperDexImportSession uses it to
// give a GLB's materials and textures their own assets when another GLB's same-named ones differ.
UCLASS(BlueprintType)
class USuperDexGlbAssetNamesPipeline : public UInterchangePipelineBase {
  GENERATED_BODY()

 public:
  // New asset names, keyed by the asset name the generic pipeline gave. A property, so it survives
  // Interchange duplicating the pipeline and storing it with the imported assets for reimports.
  UPROPERTY(VisibleAnywhere, Category = "SuperDex")
  TMap<FString, FString> AssetRenames;

 protected:
  virtual void ExecutePipeline(
      UInterchangeBaseNodeContainer* BaseNodeContainer,
      const TArray<UInterchangeSourceData*>& SourceDatas,
      const FString& ContentBasePath) override;

  virtual bool CanExecuteOnAnyThread(EInterchangePipelineTask PipelineTask) override {
    return true;
  }
};
