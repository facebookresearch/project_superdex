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

#include "MochiModelThumbnailRenderer.h"

void UMochiModelThumbnailRenderer::Draw(
    UObject* Object,
    int32 X,
    int32 Y,
    uint32 Width,
    uint32 Height,
    FRenderTarget* RenderTarget,
    FCanvas* Canvas,
    bool bAdditionalViewFamily) {
  UMochiModel* MochiModel = Cast<UMochiModel>(Object);
  if (!IsValid(MochiModel)) {
    return;
  }
  FMochiModelThumbnailScene& Scene = GetOrCreateScene(ThumbnailScene);
  Scene.SetMochiModel(MochiModel);
  RenderScene(Scene, X, Y, Width, Height, RenderTarget, Canvas, bAdditionalViewFamily);
  Scene.SetMochiModel(nullptr);
}

void UMochiModelThumbnailRenderer::BeginDestroy() {
  ThumbnailScene.Reset();
  Super::BeginDestroy();
}

FMochiModelThumbnailScene::FMochiModelThumbnailScene()
    : PreviewActor(SpawnShownActor<AMochiSoftMeshActor>()) {}

void FMochiModelThumbnailScene::SetMochiModel(UMochiModel* MochiModel) {
  UMochiSoftMeshComponent* Component = PreviewActor->GetMochiSoftMeshComponent();
  Component->SetMochiModel(MochiModel);
  if (MochiModel) {
    Component->SetMaterial(0, MochiModel->PreviewRenderMaterial);
    SetOnFloor(*PreviewActor, [Component]() { return Component->Bounds; });
  }
  Component->RecreateRenderState_Concurrent();
}

void FMochiModelThumbnailScene::GetViewMatrixParameters(
    const float InFOVDegrees,
    FVector& OutOrigin,
    float& OutOrbitPitch,
    float& OutOrbitYaw,
    float& OutOrbitZoom) const {
  const UMochiSoftMeshComponent* Component = PreviewActor->GetMochiSoftMeshComponent();
  check(Component->GetMochiModel());
  // A 15% margin keeps the edges of the mesh inside the view despite the perspective.
  FrameBounds(
      Component->Bounds,
      1.15 * Component->Bounds.SphereRadius,
      InFOVDegrees,
      OutOrigin,
      OutOrbitPitch,
      OutOrbitYaw,
      OutOrbitZoom);
}
