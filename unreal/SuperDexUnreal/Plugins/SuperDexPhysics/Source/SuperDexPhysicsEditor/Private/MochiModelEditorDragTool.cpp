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

#include "MochiModelEditorDragTool.h"

#include "Algo/AnyOf.h"
#include "CanvasItem.h"
#include "CanvasTypes.h"
#include "GeomTools.h"
#include "MochiPhysicsEditorModule.h"

#define LOCTEXT_NAMESPACE "FMochiPhysicsEditorModule"

namespace {
// The part of the view seen through the convex screen polygon Corners (in pixels, clockwise on
// screen with y down, like the box corners below; the reverse order gives an empty volume), between
// the view's near and far planes. Each edge of the polygon and the eye
// position span one side plane.
FConvexVolume MakeScreenPolygonVolume(
    const FSceneView& View,
    const FVector& EyePosition,
    TConstArrayView<FVector2D> Corners) {
  TArray<FVector, TInlineAllocator<4>> WorldCorners;
  for (const FVector2D& Corner : Corners) {
    FVector WorldOrigin;
    FVector WorldDirection;
    View.DeprojectFVector2D(Corner, WorldOrigin, WorldDirection);
    WorldCorners.Add(WorldOrigin);
  }
  FConvexVolume Volume;
  for (int32 i = 0; i < WorldCorners.Num(); ++i) {
    Volume.Planes.Emplace(WorldCorners[i], WorldCorners[(i + 1) % WorldCorners.Num()], EyePosition);
  }
  const FMatrix& ViewProjection = View.ViewMatrices.GetViewProjectionMatrix();
  FPlane ClipPlane;
  if (ViewProjection.GetFrustumNearPlane(ClipPlane)) {
    Volume.Planes.Add(ClipPlane);
  }
  if (ViewProjection.GetFrustumFarPlane(ClipPlane)) {
    Volume.Planes.Add(ClipPlane);
  }
  Volume.Init();
  return Volume;
}

void DrawScreenLine(
    FCanvas* Canvas,
    const FVector2D& From,
    const FVector2D& To,
    const FLinearColor& Color) {
  FCanvasLineItem Line(From / Canvas->GetDPIScale(), To / Canvas->GetDPIScale());
  Line.SetColor(Color);
  Canvas->DrawItem(Line);
}
} // namespace

void FMochiModelEditorDragTool::StartDrag(
    FEditorViewportClient* InViewportClient,
    const FVector& InStartWorld,
    const FVector2D& InStartScreen) {
  FDragTool::StartDrag(InViewportClient, InStartWorld, InStartScreen);

  Start = FVector(InStartScreen.X, InStartScreen.Y, 0);
  End = EndWk = Start;

  SelectionScreenPath.SetNum(0);
  SelectionScreenPath.Add(FVector2D(End));
  PreviousPosition = Start;
}

void FMochiModelEditorDragTool::AddDelta(const FVector& InDelta) {
  FIntPoint MousePos;
  MochiViewportClient->Viewport->GetMousePos(MousePos);

  EndWk = FVector(MousePos);
  End = EndWk;

  if ((End - PreviousPosition).SquaredLength() > kMinSqrDistance) {
    PreviousPosition = End;
    SelectionScreenPath.Add(FVector2D(End));
  }
}

void FMochiModelEditorDragTool::EndDrag() {
  SelectionScreenPath.Add(FVector2D(End));
  UpdateSelectedElementIndices(
      MochiViewport->ElementBarycenterPositions, MochiViewportClient->GetSelectedElements());
  // FDragTool::EndDrag resets the drag state, so it runs last.
  FDragTool::EndDrag();
}

void FMochiModelEditorDragTool::Render(const FSceneView* View, FCanvas* Canvas) {
  if (!bUseLassoSelect) {
    const FVector2D Corners[] = {
        FVector2D(Start.X, Start.Y),
        FVector2D(End.X, Start.Y),
        FVector2D(End.X, End.Y),
        FVector2D(Start.X, End.Y)};
    for (int32 i = 0; i < 4; ++i) {
      DrawScreenLine(Canvas, Corners[i], Corners[(i + 1) % 4], FLinearColor::White);
    }
    return;
  }

  // The lasso is drawn red while its path cannot be triangulated (e.g. it crosses itself), since
  // it would then select nothing. TriangulatePoly logs each failure.
  TArray<FVector2D> Triangles;
  const bool bSelectionValid = FGeomTools2D::TriangulatePoly(Triangles, SelectionScreenPath);
  const FLinearColor PathColor = bSelectionValid ? FLinearColor::White : FLinearColor::Red;
  if (SelectionScreenPath.Num() > 2) {
    for (int32 i = 0; i + 1 < SelectionScreenPath.Num(); ++i) {
      DrawScreenLine(Canvas, SelectionScreenPath[i], SelectionScreenPath[i + 1], PathColor);
    }
    // The segment to the cursor, then the implicit segment that closes the lasso.
    DrawScreenLine(Canvas, SelectionScreenPath.Last(), FVector2D(End), PathColor);
    DrawScreenLine(
        Canvas,
        FVector2D(End),
        FVector2D(Start),
        bSelectionValid ? FLinearColor::Gray : FLinearColor::Red);
  } else {
    // Too short to judge yet; a white line avoids flashing red at the start of every drag.
    DrawScreenLine(Canvas, FVector2D(Start), FVector2D(End), FLinearColor::White);
  }
}

void FMochiModelEditorDragTool::SetSelectLasso(bool bUseLasso) {
  bUseLassoSelect = bUseLasso;
}

bool FMochiModelEditorDragTool::IsUsingLasso() const {
  return bUseLassoSelect;
}

void FMochiModelEditorDragTool::UpdateSelectedElementIndices(
    const TArray<FVector>& Positions,
    TSet<int>& InOutIndices) {
  if (bUseLassoSelect && !MochiViewportClient->GetSelectedElementsContainer()) {
    return;
  }
  const EMochiModelEditorSelectionMode SelectionMode = MochiViewport->GetSelectionMode();
  if (SelectionMode == EMochiModelEditorSelectionMode::ClearFirst) {
    InOutIndices.Reset();
  }
  if (Positions.IsEmpty()) {
    return;
  }

  FSceneViewFamilyContext ViewFamily(
      FSceneViewFamily::ConstructionValues(
          MochiViewportClient->Viewport,
          MochiViewportClient->GetScene(),
          MochiViewportClient->EngineShowFlags));
  const FSceneView* SceneView = MochiViewportClient->CalcSceneView(&ViewFamily);
  const TArray<FConvexVolume> Volumes = CalculateSelectionVolumes(*SceneView);

  // TODO: Support undo/redo for the selection. Transactions started through the Kismet library
  // or GEngine did not work here; the selected elements are already held in a UObject.

  // A linear scan over all elements is fast enough for an editor tool.
  for (int32 iElement = 0; iElement < Positions.Num(); ++iElement) {
    const bool bInside = Algo::AnyOf(Volumes, [&](const FConvexVolume& Volume) {
      return Volume.IntersectPoint(Positions[iElement]);
    });
    if (!bInside) {
      continue;
    }
    if (SelectionMode == EMochiModelEditorSelectionMode::Subtractive) {
      InOutIndices.Remove(iElement);
    } else {
      InOutIndices.Add(iElement);
    }
  }
}

TArray<FConvexVolume> FMochiModelEditorDragTool::CalculateSelectionVolumes(
    const FSceneView& View) const {
  const FVector EyePosition = MochiViewportClient->GetViewLocation();
  TArray<FConvexVolume> Volumes;
  if (!bUseLassoSelect) {
    const double MinX = FMath::Min(Start.X, End.X);
    const double MaxX = FMath::Max(Start.X, End.X);
    const double MinY = FMath::Min(Start.Y, End.Y);
    const double MaxY = FMath::Max(Start.Y, End.Y);
    const FVector2D Corners[] = {
        FVector2D(MinX, MinY), FVector2D(MaxX, MinY), FVector2D(MaxX, MaxY), FVector2D(MinX, MaxY)};
    Volumes.Add(MakeScreenPolygonVolume(View, EyePosition, Corners));
    return Volumes;
  }

  // TriangulatePoly expects a simple polygon and returns its triangles three vertices at a time.
  // One convex volume per triangle is simple but slow for long paths; a convex partition of the
  // polygon would need far fewer.
  TArray<FVector2D> Triangles;
  if (!FGeomTools2D::TriangulatePoly(Triangles, SelectionScreenPath)) {
    UE_LOG(
        LogMochiPhysicsEditor,
        Error,
        TEXT("Lasso Selection failed: Could not triangulate selection polygon"));
  }
  for (int32 i = 0; i + 2 < Triangles.Num(); i += 3) {
    Volumes.Add(MakeScreenPolygonVolume(View, EyePosition, MakeArrayView(&Triangles[i], 3)));
  }
  return Volumes;
}

#undef LOCTEXT_NAMESPACE
