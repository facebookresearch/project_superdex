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

#include "MochiArticulationThumbnail.h"

FMochiArticulationThumbnailScene::FMochiArticulationThumbnailScene()
    : PreviewActor(SpawnShownActor<AMochiArticulationPreviewActor>()) {}

FBoxSphereBounds FMochiArticulationThumbnailScene::GetShownBounds() const {
  FBoxSphereBounds Bounds;
  const UMochiArticulationBase* Articulation = PreviewActor->GetArticulation();
  if (Articulation &&
      Articulation->ComputeVisualBoundsFromJointPositions(
          Articulation->GetDefaultJointPositions(), Bounds)) {
    return Bounds;
  }
  UPoseableMeshComponent* Mesh = PreviewActor->GetPoseableMeshComponent();
  Mesh->UpdateBounds();
  return Mesh->Bounds;
}

void FMochiArticulationThumbnailScene::SetArticulation(UMochiArticulationBase* Articulation) const {
  PreviewActor->SetArticulation(Articulation);
  if (Articulation && Articulation->GetSkeletalMesh()) {
    SetOnFloor(*PreviewActor, [this]() { return GetShownBounds(); });
  }
  PreviewActor->GetPoseableMeshComponent()->RecreateRenderState_Concurrent();
}

void FMochiArticulationThumbnailScene::SetJointPositions(
    const TArray<double>& JointPositions) const {
  PreviewActor->SetJointPositions(JointPositions);
}

void FMochiArticulationThumbnailScene::SetDrawDebugSkeleton(
    bool bInDrawDebugSkeleton,
    const FLinearColor& InSkeletonColor) const {
  UPoseableMeshComponent* Mesh = PreviewActor->GetPoseableMeshComponent();
  Mesh->SetDrawDebugSkeleton(bInDrawDebugSkeleton);
  Mesh->SetDebugDrawColor(InSkeletonColor);
  Mesh->RecreateRenderState_Concurrent();
}

void FMochiArticulationThumbnailScene::GetViewMatrixParameters(
    const float InFOVDegrees,
    FVector& OutOrigin,
    float& OutOrbitPitch,
    float& OutOrbitYaw,
    float& OutOrbitZoom) const {
  const FBoxSphereBounds Bounds = GetShownBounds();
  FrameBounds(
      Bounds,
      Bounds.SphereRadius,
      InFOVDegrees,
      OutOrigin,
      OutOrbitPitch,
      OutOrbitYaw,
      OutOrbitZoom);
}

void UMochiArticulationThumbnailRenderer::Draw(
    UObject* Object,
    int32 X,
    int32 Y,
    uint32 Width,
    uint32 Height,
    FRenderTarget* RenderTarget,
    FCanvas* Canvas,
    bool bAdditionalViewFamily) {
  UMochiArticulationBase* Articulation = Cast<UMochiArticulationBase>(Object);
  if (!IsValid(Articulation)) {
    return;
  }
  FMochiArticulationThumbnailScene& Scene = GetOrCreateScene(ThumbnailScene);
  Scene.SetArticulation(Articulation);
  Scene.SetJointPositions(Articulation->GetDefaultJointPositions());
  RenderScene(Scene, X, Y, Width, Height, RenderTarget, Canvas, bAdditionalViewFamily);
  Scene.SetArticulation(nullptr);
}

void UMochiArticulationThumbnailRenderer::BeginDestroy() {
  ThumbnailScene.Reset();
  Super::BeginDestroy();
}
