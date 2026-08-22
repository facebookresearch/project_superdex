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

#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/span.h>

#include <filament/Engine.h>
#include <utils/Entity.h>

#include "filament/Box.h"
#include "mochi_core/utils/transform_rt.h"

#include <optional>

namespace mochi {
class CoordinateSpaceConverter;
} // namespace mochi

namespace mochi_renderer {

class MaterialInstance;
class IInstanceable;
class Resource;

// Filament renderable layer bit reserved for the highlight overlay pass. A renderable tagged with
// this bit is additionally re-rendered into an isolated overlay target (its normal appearance in
// the main view is unaffected), which is then composited back translucently to draw a clean
// see-through highlight over occluders. Bit 0x01 is the existing visibility bit (see
// SceneObject::SetVisible), and the skybox excludes this bit (see Scene::SetSkyboxVisible) so it
// never leaks into the overlay.
constexpr uint8_t kHighlightOverlayLayer = 0x02;

class SceneObject {
 public:
  filament::math::float3 _scale = {1.0f, 1.0f, 1.0f};
  filament::math::quatf _rotation = {};
  filament::math::float3 _translation = {0.0f, 0.0f, 0.0f};
  bool _internal = false;
  bool _showAABB = false;
  SceneObject* _pickProxy = nullptr;

 public:
  virtual ~SceneObject() = default;
  virtual utils::Entity GetRootEntity() const = 0;
  virtual mochi::Span<utils::Entity const> GetEntities() const = 0;
  void SetLocalTransform(
      mochi::Quaternion const& r,
      mochi::Real3 const& t,
      mochi::Real3 const& s,
      mochi::CoordinateSpaceConverter const* converter);
  void SetLocalTransform(
      mochi::Quaternion const& r,
      mochi::Real3 const& t,
      mochi::CoordinateSpaceConverter const* converter);
  void SetLocalTransform(
      mochi::TransformRT const& rt,
      mochi::Real3 const& s,
      mochi::CoordinateSpaceConverter const* converter);
  void SetLocalTransform(
      mochi::TransformRT const& rt,
      mochi::CoordinateSpaceConverter const* converter);
  void ApplyLocalTransform(mochi::CoordinateSpaceConverter const* converter = nullptr);
  filament::math::mat4f GetLocalTransform() const;
  filament::math::mat4f GetWorldTransform() const;
  filament::Box GetAABB() const;
  // Center of the object's (object-aligned) bounding box expressed in the object's own local frame,
  // i.e. relative to its root entity and before its world transform. This is a point fixed to the
  // geometry, independent of how the object is currently posed -- unlike the center of @ref
  // GetAABB, which is a world axis-aligned box recomputed from the posed geometry and drifts as the
  // object rotates. Returns nullopt if the object has no renderable geometry.
  std::optional<filament::math::float3> GetLocalBoundsCenter() const;
  void SetName(std::string const& name);
  std::string const& GetName() const;
  virtual void SetMaterial(std::shared_ptr<MaterialInstance> material);
  virtual void SetShadows(bool castShadows, bool receiveShadows);
  bool GetCastShadows() const;
  bool GetReceiveShadows() const;
  void SetVisible(bool visible);
  bool GetVisible() const;
  // Adds/removes this object's renderables from the highlight overlay pass (the reserved
  // @ref kHighlightOverlayLayer bit). Independent of SetVisible, so an object can be both normally
  // visible and part of the overlay.
  void SetHighlightOverlay(bool enabled);
  void SetSortPriority(uint8_t priority);
  uint8_t GetSortPriority() const;
  void SetCulling(bool culling);
  bool GetCulling() const;
  void SetScreenSpaceContactShadows(bool enabled);
  bool GetScreenSpaceContactShadows() const;

  virtual IInstanceable* GetInstanceable() = 0;
  void SetParent(SceneObject* other) const;

 protected:
  SceneObject(filament::Engine* engine);

 protected:
  filament::Engine* _engine;
  std::string _name;
  std::shared_ptr<MaterialInstance> _material = nullptr;
};

} // namespace mochi_renderer
