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

#include "MochiThumbnailRendererBase.h"

#include "SceneView.h"
#include "ShowFlags.h"
#include "ThumbnailRendering/SceneThumbnailInfo.h"

FMochiThumbnailSceneBase::FMochiThumbnailSceneBase() {
  bForceAllUsedMipsResident = false;
}

void FMochiThumbnailSceneBase::SetOnFloor(AActor& Actor, TFunctionRef<FBoxSphereBounds()> GetBounds)
    const {
  Actor.SetActorLocation(FVector::ZeroVector, false);
  const FBoxSphereBounds Bounds = GetBounds();
  Actor.SetActorLocation(FVector(0.0, 0.0, GetBoundsZOffset(Bounds)) - Bounds.Origin, false);
}

void FMochiThumbnailSceneBase::FrameBounds(
    const FBoxSphereBounds& Bounds,
    double FramedRadius,
    float FOVDegrees,
    FVector& OutOrigin,
    float& OutOrbitPitch,
    float& OutOrbitYaw,
    float& OutOrbitZoom) const {
  // At this distance the framed sphere just touches the edges of the view.
  const double Distance = FramedRadius / FMath::Tan(FMath::DegreesToRadians(0.5 * FOVDegrees));
  const USceneThumbnailInfo* Defaults = GetDefault<USceneThumbnailInfo>();
  OutOrigin = FVector(0.0, 0.0, -GetBoundsZOffset(Bounds));
  OutOrbitPitch = Defaults->OrbitPitch;
  OutOrbitYaw = Defaults->OrbitYaw;
  OutOrbitZoom = static_cast<float>(Distance) + Defaults->OrbitZoom;
}

void UMochiThumbnailRendererBase::RenderScene(
    const FThumbnailPreviewScene& Scene,
    int32 X,
    int32 Y,
    uint32 Width,
    uint32 Height,
    FRenderTarget* RenderTarget,
    FCanvas* Canvas,
    bool bAdditionalViewFamily) const {
  FSceneViewFamilyContext ViewFamily(
      FSceneViewFamily::ConstructionValues(
          RenderTarget, Scene.GetScene(), FEngineShowFlags(ESFIM_Game))
          .SetTime(GetTime())
          .SetAdditionalViewFamily(bAdditionalViewFamily));
  // A thumbnail is a small still image: render the full-detail LOD without motion blur or the
  // costlier rendering features.
  ViewFamily.EngineShowFlags.DisableAdvancedFeatures();
  ViewFamily.EngineShowFlags.SetMotionBlur(false);
  ViewFamily.EngineShowFlags.SetLOD(false);
  RenderViewFamily(Canvas, &ViewFamily, Scene.CreateView(&ViewFamily, X, Y, Width, Height));
}
