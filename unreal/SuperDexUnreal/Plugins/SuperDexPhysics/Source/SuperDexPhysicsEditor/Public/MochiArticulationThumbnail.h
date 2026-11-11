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

#include "MochiArticulationThumbnail.generated.h"

// Shows an articulation (or bot) in its default pose on a preview actor.
class FMochiArticulationThumbnailScene : public FMochiThumbnailSceneBase {
 public:
  FMochiArticulationThumbnailScene();
  // The articulation the next CreateView shows, or null to clear it.
  void SetArticulation(class UMochiArticulationBase* Articulation) const;
  void SetJointPositions(const TArray<double>& JointPositions) const;
  void SetDrawDebugSkeleton(bool bInDrawDebugSkeleton, const FLinearColor& InSkeletonColor) const;

 protected:
  virtual void GetViewMatrixParameters(
      const float InFOVDegrees,
      FVector& OutOrigin,
      float& OutOrbitPitch,
      float& OutOrbitYaw,
      float& OutOrbitZoom) const override;

 private:
  // The visual bounds of the articulation in its default pose, or the preview mesh's bounds if
  // they cannot be computed.
  FBoxSphereBounds GetShownBounds() const;

  AMochiArticulationPreviewActor* PreviewActor = nullptr;
};

// Exported since the sibling SuperDexRoboticsEditor module reuses this renderer for USuperDexBot
// thumbnails (it is bot-agnostic, so no need for a SuperDexRobotics-side duplicate).
UCLASS(config = Editor)
class SUPERDEXPHYSICSEDITOR_API UMochiArticulationThumbnailRenderer
    : public UMochiThumbnailRendererBase {
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
  TUniquePtr<FMochiArticulationThumbnailScene> ThumbnailScene;
};
