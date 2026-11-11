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

#include "SuperDexRuntimeRenderInterchange.h"

#include "Engine/StaticMesh.h"
#include "UObject/Package.h"

void USuperDexRuntimeMeshPipeline::ExecutePostFactoryPipeline(
    const UInterchangeBaseNodeContainer* NodeContainer,
    const FString& NodeKey,
    UObject* CreatedAsset,
    bool bIsReimport) {
#if WITH_EDITOR
  check(IsInGameThread());
  UStaticMesh* Mesh = Cast<UStaticMesh>(CreatedAsset);
  if (!Mesh ||
      !Mesh->GetOutermost()->GetName().StartsWith(
          FString(ImportRoot) + TEXT("/"), ESearchCase::CaseSensitive) ||
      Mesh->GetNumSourceModels() == 0) {
    return;
  }
  // Leave genuine editor imports alone, including a mesh with any valid source LOD.
  for (int32 LodIndex = 0; LodIndex < Mesh->GetNumSourceModels(); ++LodIndex) {
    if (Mesh->IsMeshDescriptionValid(LodIndex)) {
      return;
    }
  }

  // In PIE, Interchange fast-builds the render data without committing MeshDescriptions, but
  // still creates empty source models. Its completion tasks then call editor Build() and
  // PostEditChange(), which replace that render data and fail with "Bad MeshDescription".
  // Removing only the empty editor source models makes CanBuild() reject both rebuild attempts
  // before they release the render resources. Render LODs, sections and materials stay intact.
  // This must happen post-factory, before BuildObject_GameThread, not in the OnDone callback.
  // SetNumSourceModels also clears the editor section maps; preserve their material assignments.
  const FMeshSectionInfoMap SectionInfo = Mesh->GetSectionInfoMap();
  const FMeshSectionInfoMap OriginalSectionInfo = Mesh->GetOriginalSectionInfoMap();
  Mesh->SetNumSourceModels(0);
  Mesh->GetSectionInfoMap() = SectionInfo;
  Mesh->GetOriginalSectionInfoMap() = OriginalSectionInfo;
#endif
}
