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
#include "InterchangeEditorUtilitiesBase.h"
#include "InterchangePipelineBase.h"

#include "SuperDexRuntimeRenderInterchange.generated.h"

/** Prevents editor rebuilds of transient meshes already built by Interchange's runtime path. */
UCLASS()
class USuperDexRuntimeMeshPipeline final : public UInterchangePipelineBase {
  GENERATED_BODY()

 public:
  // Each file imports into its own folder to keep Interchange from reusing another file's
  // materials.
  static constexpr const TCHAR* ImportRoot = TEXT("/Temp/SuperDexRuntimeMeshes");

  virtual void ExecutePostFactoryPipeline(
      const UInterchangeBaseNodeContainer* NodeContainer,
      const FString& NodeKey,
      UObject* CreatedAsset,
      bool bIsReimport) override;
};

/**
 * Interchange editor utilities for an editor binary running as a game (-game, Standalone Game).
 *
 * The editor's own utilities live in an editor-only module that is not loaded there, and the base
 * class only reports runtime in non-editor binaries. Interchange then takes its editor paths, such
 * as creating editor-only UMaterialInstanceConstants, which assert outside the editor.
 */
UCLASS()
class USuperDexInterchangeGameUtilities : public UInterchangeEditorUtilitiesBase {
  GENERATED_BODY()

 public:
  // Became const in newer engines: define both signatures (no `override`), so whichever the
  // engine's UInterchangeEditorUtilitiesBase declares is overridden.
  virtual bool IsRuntimeOrPIE() {
    return true;
  }
  virtual bool IsRuntimeOrPIE() const {
    return true;
  }
};
