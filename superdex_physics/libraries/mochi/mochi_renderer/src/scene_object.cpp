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

#include <mochi_renderer/windows_compat.h> // Must be first — cleans up Windows macros before Filament headers

#include <mochi_renderer/material.h>
#include <mochi_renderer/scene_object.h>
#include <mochi_renderer/type_conversions.h>

#include <mochi_core/utils/coordinate_space_converter.h>

#include <filament/RenderableManager.h>
#include <filament/TransformManager.h>
#include <math/mat4.h>

#include <limits>

namespace mochi_renderer {

utils::Entity SceneObject::GetRootEntity() const {
  return GetEntities()[0];
}

void SceneObject::SetLocalTransform(
    mochi::Quaternion const& r,
    mochi::Real3 const& t,
    mochi::Real3 const& s,
    mochi::CoordinateSpaceConverter const* converter) {
  SetLocalTransform(mochi::TransformRT(r, t), s, converter);
}

void SceneObject::SetLocalTransform(
    mochi::Quaternion const& r,
    mochi::Real3 const& t,
    mochi::CoordinateSpaceConverter const* converter) {
  SetLocalTransform(mochi::TransformRT(r, t), converter);
}

void SceneObject::SetLocalTransform(
    mochi::TransformRT const& rt,
    mochi::Real3 const& s,
    mochi::CoordinateSpaceConverter const* converter) {
  auto t = rt.GetTranslation();
  auto r = rt.GetRotation();
  _scale = {s[0], s[1], s[2]};
  _translation = {t[0], t[1], t[2]};
  _rotation = {r.data[3], r.data[0], r.data[1], r.data[2]};
  ApplyLocalTransform(converter);
}

void SceneObject::SetLocalTransform(
    mochi::TransformRT const& rt,
    mochi::CoordinateSpaceConverter const* converter) {
  auto t = rt.GetTranslation();
  auto r = rt.GetRotation();
  _translation = {t[0], t[1], t[2]};
  _rotation = {r.data[3], r.data[0], r.data[1], r.data[2]};
  ApplyLocalTransform(converter);
}

void SceneObject::ApplyLocalTransform(mochi::CoordinateSpaceConverter const* converter) {
  auto root = GetRootEntity();
  auto T = filament::math::mat4f::translation(
      filament::math::float3{_translation[0], _translation[1], _translation[2]});
  auto R = filament::math::mat4f(_rotation);
  auto S = filament::math::mat4f::scaling(filament::math::float3{_scale[0], _scale[1], _scale[2]});
  auto& tm = _engine->getTransformManager();
  filament::math::mat4f x(T * R * S);
  if (converter) {
    x = ToFilament(converter->TransformToOutput(ToMochi(x)));
  }
  tm.setTransform(tm.getInstance(root), x);
}

filament::math::mat4f SceneObject::GetLocalTransform() const {
  auto root = GetRootEntity();
  auto& tm = _engine->getTransformManager();
  return tm.getTransform(tm.getInstance(root));
}

filament::math::mat4f SceneObject::GetWorldTransform() const {
  auto root = GetRootEntity();
  auto& tm = _engine->getTransformManager();
  return tm.getWorldTransform(tm.getInstance(root));
}

filament::Box SceneObject::GetAABB() const {
  auto entities = GetEntities();
  auto& rcm = _engine->getRenderableManager();
  auto& tm = _engine->getTransformManager();
  filament::math::float3 minBound{std::numeric_limits<float>::max()};
  filament::math::float3 maxBound{std::numeric_limits<float>::lowest()};
  bool hasRenderable = false;
  for (auto entity : entities) {
    auto ri = rcm.getInstance(entity);
    if (!ri.isValid()) {
      continue;
    }
    auto box = rcm.getAxisAlignedBoundingBox(ri);
    auto bMin = box.center - box.halfExtent;
    auto bMax = box.center + box.halfExtent;
    auto ti = tm.getInstance(entity);
    auto worldTransform = tm.getWorldTransform(ti);
    for (int c = 0; c < 8; ++c) {
      filament::math::float3 corner{
          (c & 1) ? bMax.x : bMin.x, (c & 2) ? bMax.y : bMin.y, (c & 4) ? bMax.z : bMin.z};
      auto world = (worldTransform * filament::math::float4{corner, 1.0f}).xyz;
      minBound = min(minBound, world);
      maxBound = max(maxBound, world);
    }
    hasRenderable = true;
  }
  if (!hasRenderable) {
    return filament::Box{};
  }
  auto center = (minBound + maxBound) * 0.5f;
  auto halfExtent = (maxBound - minBound) * 0.5f;
  return filament::Box{center, halfExtent};
}

std::optional<filament::math::float3> SceneObject::GetLocalBoundsCenter() const {
  auto entities = GetEntities();
  auto& rcm = _engine->getRenderableManager();
  auto& tm = _engine->getTransformManager();
  // Union the renderables' local bounding boxes in the root's local frame and return that
  // (object-aligned) box's center. Because it is expressed in the object's own frame (before the
  // world transform), it is fixed to the geometry and independent of the current pose.
  filament::math::mat4f const rootWorldInverse =
      inverse(tm.getWorldTransform(tm.getInstance(GetRootEntity())));
  filament::math::float3 minBound{std::numeric_limits<float>::max()};
  filament::math::float3 maxBound{std::numeric_limits<float>::lowest()};
  bool hasRenderable = false;
  for (auto entity : entities) {
    auto ri = rcm.getInstance(entity);
    if (!ri.isValid()) {
      continue;
    }
    auto box = rcm.getAxisAlignedBoundingBox(ri); // in the entity's local frame
    auto bMin = box.center - box.halfExtent;
    auto bMax = box.center + box.halfExtent;
    // Express this entity's local box in the root's local frame so the union is object-aligned.
    filament::math::mat4f const entityToRoot =
        rootWorldInverse * tm.getWorldTransform(tm.getInstance(entity));
    for (int c = 0; c < 8; ++c) {
      filament::math::float3 corner{
          (c & 1) ? bMax.x : bMin.x, (c & 2) ? bMax.y : bMin.y, (c & 4) ? bMax.z : bMin.z};
      auto local = (entityToRoot * filament::math::float4{corner, 1.0f}).xyz;
      minBound = min(minBound, local);
      maxBound = max(maxBound, local);
    }
    hasRenderable = true;
  }
  if (!hasRenderable) {
    return std::nullopt;
  }
  return (minBound + maxBound) * 0.5f;
}

void SceneObject::SetName(std::string const& name) {
  _name = name;
}

std::string const& SceneObject::GetName() const {
  return _name;
}

void SceneObject::SetMaterial(std::shared_ptr<MaterialInstance> material) {
  auto& rcm = _engine->getRenderableManager();
  auto entities = GetEntities();
  for (auto entity : entities) {
    auto ri = rcm.getInstance(entity);
    if (!ri.isValid()) {
      continue;
    }
    size_t const primitiveCount = rcm.getPrimitiveCount(ri);
    for (size_t p = 0; p < primitiveCount; ++p) {
      rcm.setMaterialInstanceAt(ri, p, material->Get());
    }
  }
  _material = material;
}

void SceneObject::SetShadows(bool castShadows, bool receiveShadows) {
  auto& rcm = _engine->getRenderableManager();
  auto entities = GetEntities();
  for (auto entity : entities) {
    auto ri = rcm.getInstance(entity);
    if (!ri.isValid()) {
      continue;
    }
    rcm.setCastShadows(ri, castShadows);
    rcm.setReceiveShadows(ri, receiveShadows);
  }
}

void SceneObject::SetVisible(bool visible) {
  auto& rcm = _engine->getRenderableManager();
  auto entities = GetEntities();
  for (auto entity : entities) {
    auto ri = rcm.getInstance(entity);
    if (!ri.isValid()) {
      continue;
    }
    rcm.setLayerMask(ri, 0x01, visible ? 0x01 : 0x00);
  }
}

void SceneObject::SetHighlightOverlay(bool enabled) {
  auto& rcm = _engine->getRenderableManager();
  auto entities = GetEntities();
  for (auto entity : entities) {
    auto ri = rcm.getInstance(entity);
    if (!ri.isValid()) {
      continue;
    }
    rcm.setLayerMask(ri, kHighlightOverlayLayer, enabled ? kHighlightOverlayLayer : 0x00);
  }
}

void SceneObject::SetCulling(bool culling) {
  auto& rcm = _engine->getRenderableManager();
  for (auto entity : GetEntities()) {
    auto ri = rcm.getInstance(entity);
    if (!ri.isValid()) {
      continue;
    }
    rcm.setCulling(ri, culling);
  }
}

void SceneObject::SetSortPriority(uint8_t priority) {
  auto& rcm = _engine->getRenderableManager();
  for (auto entity : GetEntities()) {
    auto ri = rcm.getInstance(entity);
    if (!ri.isValid()) {
      continue;
    }
    rcm.setPriority(ri, priority);
  }
}

uint8_t SceneObject::GetSortPriority() const {
  auto& rcm = _engine->getRenderableManager();
  for (auto entity : GetEntities()) {
    auto ri = rcm.getInstance(entity);
    if (ri.isValid()) {
      return rcm.getPriority(ri);
    }
  }
  return 4; // Filament's default priority.
}

void SceneObject::SetScreenSpaceContactShadows(bool enabled) {
  auto& rcm = _engine->getRenderableManager();
  for (auto entity : GetEntities()) {
    auto ri = rcm.getInstance(entity);
    if (!ri.isValid()) {
      continue;
    }
    rcm.setScreenSpaceContactShadows(ri, enabled);
  }
}

bool SceneObject::GetCastShadows() const {
  auto& rcm = _engine->getRenderableManager();
  for (auto entity : GetEntities()) {
    auto ri = rcm.getInstance(entity);
    if (ri.isValid()) {
      return rcm.isShadowCaster(ri);
    }
  }
  return false;
}

bool SceneObject::GetReceiveShadows() const {
  auto& rcm = _engine->getRenderableManager();
  for (auto entity : GetEntities()) {
    auto ri = rcm.getInstance(entity);
    if (ri.isValid()) {
      return rcm.isShadowReceiver(ri);
    }
  }
  return false;
}

bool SceneObject::GetVisible() const {
  auto& rcm = _engine->getRenderableManager();
  for (auto entity : GetEntities()) {
    auto ri = rcm.getInstance(entity);
    if (ri.isValid()) {
      return (rcm.getLayerMask(ri) & 0x01) != 0;
    }
  }
  return false;
}

bool SceneObject::GetCulling() const {
  auto& rcm = _engine->getRenderableManager();
  for (auto entity : GetEntities()) {
    auto ri = rcm.getInstance(entity);
    if (ri.isValid()) {
      return rcm.isCullingEnabled(ri);
    }
  }
  return false;
}

bool SceneObject::GetScreenSpaceContactShadows() const {
  auto& rcm = _engine->getRenderableManager();
  for (auto entity : GetEntities()) {
    auto ri = rcm.getInstance(entity);
    if (ri.isValid()) {
      return rcm.isScreenSpaceContactShadowsEnabled(ri);
    }
  }
  return false;
}

void SceneObject::SetParent(SceneObject* other) const {
  auto& tm = _engine->getTransformManager();
  auto thisRoot = GetRootEntity();
  auto otherRoot = other->GetRootEntity();
  tm.setParent(tm.getInstance(thisRoot), tm.getInstance(otherRoot));
  tm.setTransform(tm.getInstance(thisRoot), filament::math::mat4f{});
}

SceneObject::SceneObject(filament::Engine* engine) : _engine(engine) {}

} // namespace mochi_renderer
