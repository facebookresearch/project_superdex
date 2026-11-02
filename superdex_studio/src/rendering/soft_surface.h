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

#include <mochi_core/geometry/model_data.h>

namespace superdex::studio {

// One of a deformable actor's triangle meshes. Mochi reports each one's deformed nodes through its
// own pair of position/normal queries.
enum class SoftSurface {
  Visual, // ModelData::visualMesh (VisualNodePositions / VisualNodeNormals)
  ContactSkin, // ModelData::contactSkinMesh (ContactSkinNodePositions / ContactSkinNodeNormals)
  Surface, // the boundary of the actor's mesh (SurfaceNodePositions / SurfaceNodeNormals)
};

// The meshes a staged deformable actor shows: `rendered` in the render slot and `collision` in the
// shape (Collision Only view) slot. The defaults suit an articulated skin, which has neither a
// visual mesh nor a contact skin.
struct SoftSurfaces {
  SoftSurface rendered = SoftSurface::Surface;
  SoftSurface collision = SoftSurface::Surface;

  bool operator==(SoftSurfaces const& other) const = default;
};

// A soft actor is drawn with its visual mesh, else its contact skin, else its boundary. The
// Collision Only view shows what it collides with: its contact skin when it has one and
// SoftActorParams::useContactSkin is on, else its boundary.
constexpr SoftSurfaces
ChooseSoftSurfaces(bool hasVisualMesh, bool hasContactSkin, bool useContactSkin) {
  SoftSurface const rendered = hasVisualMesh ? SoftSurface::Visual
      : hasContactSkin                       ? SoftSurface::ContactSkin
                                             : SoftSurface::Surface;
  return {
      .rendered = rendered,
      .collision =
          hasContactSkin && useContactSkin ? SoftSurface::ContactSkin : SoftSurface::Surface};
}

// Deformable actors can update a visual mesh only when it has an embedding into the simulation
// mesh, so a soft actor whose visual mesh has none falls back to its contact skin or boundary.
constexpr SoftSurfaces ChooseSoftSurfaces(mochi::ModelData const& model, bool useContactSkin) {
  bool const hasDeformableVisualMesh = model.visualMesh && model.visualMesh->skinning;
  return ChooseSoftSurfaces(
      hasDeformableVisualMesh, model.contactSkinMesh.has_value(), useContactSkin);
}

constexpr char const* SoftSurfaceLabel(SoftSurface surface) {
  switch (surface) {
    case SoftSurface::Visual:
      return "Visual mesh";
    case SoftSurface::ContactSkin:
      return "Contact skin";
    case SoftSurface::Surface:
      return "Surface";
  }
  return "";
}

} // namespace superdex::studio
