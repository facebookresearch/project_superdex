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

#include "MochiActors.h"
#include "MochiThumbnailRendererBase.h"

#include "CoreMinimal.h"

#include "MochiModelThumbnailRenderer.generated.h"

// Shows a Mochi model on a soft mesh actor.
class SUPERDEXPHYSICSEDITOR_API FMochiModelThumbnailScene : public FMochiThumbnailSceneBase {
 public:
  FMochiModelThumbnailScene();

  // The model the next CreateView shows, or null to clear it.
  void SetMochiModel(class UMochiModel* MochiModel);

 protected:
  virtual void GetViewMatrixParameters(
      const float InFOVDegrees,
      FVector& OutOrigin,
      float& OutOrbitPitch,
      float& OutOrbitYaw,
      float& OutOrbitZoom) const override;

 private:
  AMochiSoftMeshActor* PreviewActor = nullptr;
};

UCLASS(config = Editor)
class UMochiModelThumbnailRenderer : public UMochiThumbnailRendererBase {
  GENERATED_BODY()

 public:
  virtual void Draw(
      UObject* Object,
      int32 X,
      int32 Y,
      uint32 Width,
      uint32 Height,
      FRenderTarget* RenderTarget,
      FCanvas* Canvas,
      bool bAdditionalViewFamily) override;
  virtual void BeginDestroy() override;

 private:
  TUniquePtr<FMochiModelThumbnailScene> ThumbnailScene;
};
