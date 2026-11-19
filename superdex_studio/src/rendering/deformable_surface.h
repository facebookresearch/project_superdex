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

// One of a deformable actor's meshes. Mochi reports each triangle mesh's deformed nodes through its
// own pair of position/normal queries; a rod's centerline comes from its pose instead.
enum class DeformableSurface {
  Visual, // ModelData::visualMesh (VisualNodePositions / VisualNodeNormals)
  ContactSkin, // ModelData::contactSkinMesh (ContactSkinNodePositions / ContactSkinNodeNormals)
  Surface, // the boundary of the actor's mesh (SurfaceNodePositions / SurfaceNodeNormals)
  Centerline, // a rod's polyline mesh (Actor::GetMesh plus Actor::GetRodPose), drawn as lines
};

// The meshes a staged deformable actor shows: `rendered` in the render slot and `collision` in the
// shape (Collision Only view) slot. The defaults suit an articulated skin, which has neither a
// visual mesh nor a contact skin.
struct DeformableSurfaces {
  DeformableSurface rendered = DeformableSurface::Surface;
  DeformableSurface collision = DeformableSurface::Surface;

  bool operator==(DeformableSurfaces const& other) const = default;
};

// A soft actor is drawn with its visual mesh, else its contact skin, else its boundary. The
// Collision Only view shows what it collides with: its contact skin when it has one and
// SoftActorParams::useContactSkin is on, else its boundary.
constexpr DeformableSurfaces
ChooseSoftSurfaces(bool hasVisualMesh, bool hasContactSkin, bool useContactSkin) {
  DeformableSurface const rendered = hasVisualMesh ? DeformableSurface::Visual
      : hasContactSkin                             ? DeformableSurface::ContactSkin
                                                   : DeformableSurface::Surface;
  return {
      .rendered = rendered,
      .collision = hasContactSkin && useContactSkin ? DeformableSurface::ContactSkin
                                                    : DeformableSurface::Surface};
}

// Deformable actors can update a visual mesh only when it has an embedding into the simulation
// mesh, so a soft actor whose visual mesh has none falls back to its contact skin or boundary.
constexpr DeformableSurfaces ChooseSoftSurfaces(
    mochi::ModelData const& model,
    bool useContactSkin) {
  bool const hasDeformableVisualMesh = model.visualMesh && model.visualMesh->skinning;
  return ChooseSoftSurfaces(
      hasDeformableVisualMesh, model.contactSkinMesh.has_value(), useContactSkin);
}

// A rod actor is drawn like a soft actor, with its centerline in place of the boundary: its visual
// mesh, else its contact skin, else its centerline. The Collision Only view shows its contact skin
// when it has one and RodActorParams::useContactSkin is on, else its centerline.
constexpr DeformableSurfaces
ChooseRodSurfaces(bool hasVisualMesh, bool hasContactSkin, bool useContactSkin) {
  DeformableSurfaces const soft = ChooseSoftSurfaces(hasVisualMesh, hasContactSkin, useContactSkin);
  auto const rod = [](DeformableSurface surface) {
    return surface == DeformableSurface::Surface ? DeformableSurface::Centerline : surface;
  };
  return {.rendered = rod(soft.rendered), .collision = rod(soft.collision)};
}

// Unlike a soft actor's, a rod's visual mesh needs no skinning: Mochi embeds it in the rod's
// elements when the actor is created.
constexpr DeformableSurfaces ChooseRodSurfaces(mochi::ModelData const& model, bool useContactSkin) {
  return ChooseRodSurfaces(
      model.visualMesh.has_value(), model.contactSkinMesh.has_value(), useContactSkin);
}

constexpr char const* DeformableSurfaceLabel(DeformableSurface surface) {
  switch (surface) {
    case DeformableSurface::Visual:
      return "Visual mesh";
    case DeformableSurface::ContactSkin:
      return "Contact skin";
    case DeformableSurface::Surface:
      return "Surface";
    case DeformableSurface::Centerline:
      return "Centerline";
  }
  return "";
}

} // namespace superdex::studio
