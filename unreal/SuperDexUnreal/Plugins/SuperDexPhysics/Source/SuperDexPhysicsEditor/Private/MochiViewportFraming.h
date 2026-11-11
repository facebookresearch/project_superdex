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
#include "EditorViewportClient.h"

namespace MochiViewportFraming {

// Points Client's camera at Bounds the way the Mochi asset editors frame an asset: back along the
// current view direction until the whole bounding sphere is in view.
//
// FEditorViewportClient::FocusViewportOnBox fits a box's half-diagonal into the view with the
// tangent of half the field of view, which brings the camera close enough for a bounding sphere of
// that radius to touch the edges of the view. Giving it a cube whose half-diagonal is the sphere's
// radius divided by the cosine of half the field of view moves the camera back to where the whole
// sphere fits. FocusViewportOnBox treats any box as at least 10 units in radius.
inline void
FocusOnBounds(FEditorViewportClient& Client, const FBoxSphereBounds& Bounds, bool bInstant) {
  const double HalfFOVRadians = FMath::DegreesToRadians(0.5 * Client.ViewFOV);
  const double HalfDiagonal = Bounds.SphereRadius / FMath::Cos(HalfFOVRadians);
  const FVector HalfExtent(HalfDiagonal / FMath::Sqrt(3.0));
  Client.FocusViewportOnBox(FBox(Bounds.Origin - HalfExtent, Bounds.Origin + HalfExtent), bInstant);
}

} // namespace MochiViewportFraming
