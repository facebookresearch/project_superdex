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
#include "RenderingThread.h"
#include "ThumbnailHelpers.h"
#include "ThumbnailRendering/DefaultSizedThumbnailRenderer.h"

#include "MochiThumbnailRendererBase.generated.h"

class FCanvas;
class FRenderTarget;

// A thumbnail preview scene that shows one transient actor resting on the floor, framed by its
// bounds.
class SUPERDEXPHYSICSEDITOR_API FMochiThumbnailSceneBase : public FThumbnailPreviewScene {
 public:
  FMochiThumbnailSceneBase();

 protected:
  // Spawns the actor the scene shows.
  template <typename TActor>
  TActor* SpawnShownActor() {
    FActorSpawnParameters SpawnParams;
    SpawnParams.ObjectFlags = RF_Transient;
    SpawnParams.bNoFail = true;
    SpawnParams.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
    return GetWorld()->SpawnActor<TActor>(SpawnParams);
  }

  // Moves Actor so that the bounds GetBounds returns with the actor at the origin are centered
  // over the origin and rest on the floor.
  void SetOnFloor(AActor& Actor, TFunctionRef<FBoxSphereBounds()> GetBounds) const;

  // View parameters that fit a sphere of FramedRadius around floor-placed Bounds into a view of
  // FOVDegrees, orbiting at the default scene thumbnail angles.
  void FrameBounds(
      const FBoxSphereBounds& Bounds,
      double FramedRadius,
      float FOVDegrees,
      FVector& OutOrigin,
      float& OutOrbitPitch,
      float& OutOrbitYaw,
      float& OutOrbitZoom) const;
};

// Base class of the Mochi asset thumbnail renderers: renders a preview scene into the thumbnail.
UCLASS(Abstract, config = Editor)
class SUPERDEXPHYSICSEDITOR_API UMochiThumbnailRendererBase
    : public UDefaultSizedThumbnailRenderer {
  GENERATED_BODY()

 protected:
  // The scene in Scene, created on first use and again if its world has gone away.
  template <typename TScene>
  static TScene& GetOrCreateScene(TUniquePtr<TScene>& Scene) {
    if (!Scene || !ensure(Scene->GetWorld() != nullptr)) {
      if (Scene) {
        // The render thread may still reference the old scene.
        FlushRenderingCommands();
        Scene.Reset();
      }
      Scene = MakeUnique<TScene>();
    }
    return *Scene;
  }

  // Renders the view of Scene into the thumbnail rectangle.
  void RenderScene(
      const FThumbnailPreviewScene& Scene,
      int32 X,
      int32 Y,
      uint32 Width,
      uint32 Height,
      FRenderTarget* RenderTarget,
      FCanvas* Canvas,
      bool bAdditionalViewFamily) const;
};
