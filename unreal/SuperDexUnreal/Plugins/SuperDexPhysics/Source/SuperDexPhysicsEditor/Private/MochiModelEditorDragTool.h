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
#include "EditorDragTools.h"
#include "MochiModelEditorViewportClient.h"
#include "SMochiModelEditorViewport.h"

class FCanvas;

/**
 * Selects or deselects the elements of a Mochi model whose barycenters fall inside a box or a
 * freehand lasso dragged in the model editor viewport.
 */
class FMochiModelEditorDragTool : public FDragTool {
 public:
  explicit FMochiModelEditorDragTool(
      FMochiModelEditorViewportClient* InViewportClient,
      SMochiModelEditorViewport* InViewport)
      : FDragTool(InViewportClient->GetModeTools()),
        MochiViewport(InViewport),
        MochiViewportClient(InViewportClient) {}

  virtual void StartDrag(
      FEditorViewportClient* InViewportClient,
      const FVector& InStartWorld,
      const FVector2D& InStartScreen) override;
  virtual void AddDelta(const FVector& InDelta) override;
  virtual void EndDrag() override;
  virtual void Render(const FSceneView* View, FCanvas* Canvas) override;
  void SetSelectLasso(bool bUseLasso);
  bool IsUsingLasso() const;

 private:
  struct MochiState {
    TArray<int> FoundNodeIndices;
    mochi::ActorHandle ActorHandle;
  };

 private:
  SMochiModelEditorViewport* MochiViewport;
  FMochiModelEditorViewportClient* MochiViewportClient;

  FVector PreviousPosition;
  float kMinSqrDistance = 1000.0f;

  TArray<FVector2D> SelectionScreenPath;

  bool bUseLassoSelect = true;

  // Applies the current selection mode to the elements at Positions that lie inside the dragged
  // box or lasso.
  void UpdateSelectedElementIndices(const TArray<FVector>& Positions, TSet<int>& InOutIndices);
  // The world-space volumes covered by the dragged box (one volume) or lasso (one per triangle).
  TArray<FConvexVolume> CalculateSelectionVolumes(const FSceneView& View) const;
};
