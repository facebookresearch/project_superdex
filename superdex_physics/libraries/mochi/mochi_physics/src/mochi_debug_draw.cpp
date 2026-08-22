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

#include "mochi_debug_draw.h"

#include "mochi_common_components.h"
#include "mochi_ecs_utils.h"

#include <mochi_core/geometry/geometry_utils.h>
#include <mochi_core/utils/array_utils.h>
#include <mochi_core/utils/container_utils.h>
#include <mochi_core/utils/matrix_utils.h>
#include <mochi_core/utils/profile.h>

#include <algorithm>
#include <memory>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace mochi {

namespace {
void TransformCoords(
    std::vector<Real3>& outCoords,
    Span<Real3 const> inCoords,
    TransformRT const& transform) {
  outCoords.resize(inCoords.size());
  ArrayTransformPoints(MakeSpan(outCoords), inCoords, transform);
}

NdArray<DebugDrawLineVertex, 6> MakeTriAxisVertices(TransformRT const& transform, real scale) {
  Vec4r origin = transform.VGetTranslation();
  VMatrix3x3r scaledBasis = ToVMatrix3x3Transpose(transform.GetRotation()) * scale;
  Color constexpr kColors[3] = {colors::kRed, colors::kGreen, colors::kBlue};
  return {
      DebugDrawLineVertex{ToReal3(origin), kColors[0]},
      DebugDrawLineVertex{ToReal3(origin + scaledBasis[0]), kColors[0]},
      DebugDrawLineVertex{ToReal3(origin), kColors[1]},
      DebugDrawLineVertex{ToReal3(origin + scaledBasis[1]), kColors[1]},
      DebugDrawLineVertex{ToReal3(origin), kColors[2]},
      DebugDrawLineVertex{ToReal3(origin + scaledBasis[2]), kColors[2]},
  };
}

NdArray<DebugDrawLineVertex, 24> MakeBoundingBoxVertices(
    NdArray<real, 8, 3> const& corners,
    Color color) {
  //    3 +--------+ 7
  //     /|       /|    y  z
  //    / |      / |    | /
  // 2 +--------+ 6|    |/
  //   |  |     |  |    *-- x
  //   |1 +-----|--+ 5
  //   | /      | /
  //   |/       |/
  // 0 +--------+ 4

  // clang-format off
  return {
      // bottom
      DebugDrawLineVertex{corners[0], color}, DebugDrawLineVertex{corners[4], color},
      DebugDrawLineVertex{corners[4], color}, DebugDrawLineVertex{corners[5], color},
      DebugDrawLineVertex{corners[5], color}, DebugDrawLineVertex{corners[1], color},
      DebugDrawLineVertex{corners[1], color}, DebugDrawLineVertex{corners[0], color},
      // top
      DebugDrawLineVertex{corners[2], color}, DebugDrawLineVertex{corners[6], color},
      DebugDrawLineVertex{corners[6], color}, DebugDrawLineVertex{corners[7], color},
      DebugDrawLineVertex{corners[7], color}, DebugDrawLineVertex{corners[3], color},
      DebugDrawLineVertex{corners[3], color}, DebugDrawLineVertex{corners[2], color},
      // sides
      DebugDrawLineVertex{corners[0], color}, DebugDrawLineVertex{corners[2], color},
      DebugDrawLineVertex{corners[4], color}, DebugDrawLineVertex{corners[6], color},
      DebugDrawLineVertex{corners[5], color}, DebugDrawLineVertex{corners[7], color},
      DebugDrawLineVertex{corners[1], color}, DebugDrawLineVertex{corners[3], color},
  };
  // clang-format on
}

/**************************************************************************************************
  DebugDrawCollectorImpl
    - Wraps a DebugDraw::Collector and provides extended features
    - Stores data in SoA (Structure of Arrays) format.
*/
class DebugDrawCollectorImpl final : public DebugDrawCollector {
 public:
  void AddLines(Span<DebugDrawLineVertex const> lineVerts) override {
    auto offset = allLinePositions.size();
    allLinePositions.resize(offset + lineVerts.size());
    allLineColors.resize(offset + lineVerts.size());
    if (_applyTransform) {
      for (size_t i = 0; i < lineVerts.size(); ++i, ++offset) {
        auto position = DotVecMat4x4(ToSimd(lineVerts[i].position, 1_r), _worldFromLocalMatT);
        MOCHI_ASSERT_VERBOSE(IsFinite(position));
        allLinePositions[offset] = ToReal3(position);
        allLineColors[offset] = lineVerts[i].color;
      }
    } else {
      for (size_t i = 0; i < lineVerts.size(); ++i, ++offset) {
        allLinePositions[offset] = lineVerts[i].position;
        allLineColors[offset] = lineVerts[i].color;
      }
    }
  }

  void AddTriAxis(TransformRT const& transformIn, real scale) override {
    TransformRT rt = _applyTransform ? (_worldFromLocal * transformIn) : transformIn;
    auto lineVerts = MakeTriAxisVertices(rt, scale);
    auto vi = allLinePositions.size();
    allLinePositions.resize_noinit(vi + 6);
    allLineColors.resize_noinit(vi + 6);
    for (size_t i = 0; i < 6; ++i, ++vi) {
      allLinePositions[vi] = lineVerts[i].position;
      allLineColors[vi] = lineVerts[i].color;
    }
  }

  void AddWireframeMesh(
      Span<Real3 const> coordsIn,
      Span<Int2 const> edges,
      Color uniformColor,
      TransformRT const& localTransform) override {
    if (coordsIn.empty() && edges.empty()) {
      return;
    }
    bool const hasLocalTransform = !NearEqual(localTransform, TransformRT::Identity());
    Real3 const* coords = coordsIn.data();
    if (_applyTransform || hasLocalTransform) {
      TransformRT fullTransform =
          _applyTransform ? (_worldFromLocal * localTransform) : localTransform;
      TransformCoords(_tempCoords, coordsIn, fullTransform);
      coords = _tempCoords.data();
    }

    size_t const numEdges = edges.size();
    size_t vi = allLinePositions.size();
    allLinePositions.resize_noinit(vi + 2 * numEdges);
    allLineColors.resize_noinit(vi + 2 * numEdges);

    for (size_t i = 0; i < numEdges; ++i) {
      auto position = coords[edges[i][0]];
      MOCHI_ASSERT_VERBOSE(IsFinite(ToSimd(position)));
      allLinePositions[vi] = position;
      allLineColors[vi] = uniformColor;
      ++vi;

      position = coords[edges[i][1]];
      MOCHI_ASSERT_VERBOSE(IsFinite(ToSimd(position)));
      allLinePositions[vi] = position;
      allLineColors[vi] = uniformColor;
      ++vi;
    }
  }

  void AddWireframeMesh(Span<Real3 const> nodesIn, Span<Int2 const> edges, Span<Color const> colors)
      override {
    Real3 const* coords = nodesIn.data();
    if (_applyTransform) {
      TransformCoords(_tempCoords, nodesIn, _worldFromLocal);
      coords = _tempCoords.data();
    }
    size_t const numEdges = edges.size();
    size_t vi = allLinePositions.size();
    allLinePositions.resize_noinit(vi + 2 * numEdges);
    allLineColors.resize_noinit(vi + 2 * numEdges);

    for (size_t i = 0; i < numEdges; ++i) {
      auto position = coords[edges[i][0]];
      MOCHI_ASSERT_VERBOSE(IsFinite(ToSimd(position)));
      allLinePositions[vi] = position;
      allLineColors[vi] = colors[edges[i][0]];
      ++vi;

      position = coords[edges[i][1]];
      MOCHI_ASSERT_VERBOSE(IsFinite(ToSimd(position)));
      allLinePositions[vi] = position;
      allLineColors[vi] = colors[edges[i][1]];
      ++vi;
    }
  }

  void AddWireframeAabb(Aabb const& aabb, Color c) override {
    Real3 min = aabb.GetMin();
    Real3 max = aabb.GetMax();
    NdArray<real, 8, 3> const corners = {
        Real3{min[0], min[1], min[2]},
        Real3{min[0], min[1], max[2]},
        Real3{min[0], max[1], min[2]},
        Real3{min[0], max[1], max[2]},
        Real3{max[0], min[1], min[2]},
        Real3{max[0], min[1], max[2]},
        Real3{max[0], max[1], min[2]},
        Real3{max[0], max[1], max[2]},
    };
    auto const localVerts = MakeBoundingBoxVertices(corners, c);
    AddLines(localVerts);
  }

  void AddWireframeAabb(Real3 const& center, real halfWidth, Color color) override {
    Vec4r cnt = ToSimd(center);
    AddWireframeAabb(Aabb{cnt - halfWidth, cnt + halfWidth}, color);
  }

  void AddWireframeObb(Obb const& oobb, Color color) override {
    auto const& box = _applyTransform ? TransformShape(_worldFromLocal, oobb) : oobb;
    auto const corners = box.GetCorners();
    auto const lineVerts = MakeBoundingBoxVertices(corners, color);
    auto vi = allLinePositions.size();
    allLinePositions.resize_noinit(vi + 24);
    allLineColors.resize_noinit(vi + 24);
    for (size_t i = 0; i < 24; ++i, ++vi) {
      allLinePositions[vi] = lineVerts[i].position;
      allLineColors[vi] = lineVerts[i].color;
    }
  }

  void AddSphere(DebugDrawSphere const& sphere) override {
    if (_applyTransform) {
      auto position = _worldFromLocal.TransformPoint(sphere.position);
      MOCHI_ASSERT_VERBOSE(IsFinite(ToSimd(position)));
      allSpherePositions.emplace_back(position);
    } else {
      allSpherePositions.emplace_back(sphere.position);
    }
    allSphereRadii.emplace_back(sphere.radius);
    allSphereColors.emplace_back(sphere.color);
  }

  void SetTransform(TransformRT const& rt) {
    _worldFromLocal = rt;
    _worldFromLocalMatT = ToVMatrix4x4Transpose(rt);
    _applyTransform = true;
  }

  void ClearTransform() {
    _applyTransform = false;
  }

  void ClearData() {
    allLinePositions.clear();
    allLineColors.clear();
    allSpherePositions.clear();
    allSphereRadii.clear();
    allSphereColors.clear();
  }

  // All collected data in SOA format:
  DynamicArray<Real3> allLinePositions;
  DynamicArray<Color> allLineColors;
  DynamicArray<Real3> allSpherePositions;
  DynamicArray<real> allSphereRadii;
  DynamicArray<Color> allSphereColors;

 private:
  std::vector<Real3> _tempCoords;
  TransformRT _worldFromLocal;
  VMatrix4x4r _worldFromLocalMatT;
  bool _applyTransform = false;
};

/**************************************************************************************************
  DebugDrawSystemImpl
*/
class DebugDrawSystemImpl final : public DebugDrawSystem {
 public:
  explicit DebugDrawSystemImpl(DebugDrawSystem&& system) : DebugDrawSystem(std::move(system)) {}

  bool HasAllRequiredComponents(entt::entity e) const {
    for (auto const* pool : this->requiredComponentTypes) {
      if (!pool->contains(e)) {
        // This entity is missing one of the required components
        return false;
      }
    }
    return true;
  }

  int GetNumExcludedComponents(entt::entity e) const {
    int count = 0;
    for (auto const* pool : this->excludedComponentTypes) {
      if (pool->contains(e)) {
        count++;
      }
    }
    return count;
  }

  // This function is called by the entt::registry just AFTER one of this system's REQUIRED
  // components was added to an entity.
  void OnRequiredComponentConstruct(entt::registry& reg, entt::entity e) {
    if (this->isEnabled && this->onEntityEnable) {
      if (HasAllRequiredComponents(e) && (GetNumExcludedComponents(e) == 0)) {
        // This entity meets all requirements, but it did not previously.
        this->onEntityEnable(reg, e, true);
      }
    }
  }

  // This function is called by the entt::registry just BEFORE one of this system's REQUIRED
  // components is removed from an entity.
  void OnRequiredComponentDestroy(entt::registry& reg, entt::entity e) {
    if (this->isEnabled && this->onEntityEnable) {
      if (HasAllRequiredComponents(e) && (GetNumExcludedComponents(e) == 0)) {
        // This entity previously met all requirements, but now a required component is being
        // removed.
        this->onEntityEnable(reg, e, false);
      }
    }
  }

  // This function is called by the entt::registry just AFTER one of this system's EXCLUDED
  // components was added to an entity.
  void OnExcludedComponentConstruct(entt::registry& reg, entt::entity e) {
    if (this->isEnabled && this->onEntityEnable) {
      if (HasAllRequiredComponents(e) && (GetNumExcludedComponents(e) == 1)) {
        // This entity previously met all requirements, but we just added the first excluded
        // component.
        this->onEntityEnable(reg, e, false);
      }
    }
  }

  // This function is called by the entt::registry just BEFORE one of this system's EXCLUDED
  // components is removed from an entity.
  void OnExcludedComponentDestroy(entt::registry& reg, entt::entity e) {
    if (isEnabled && this->onEntityEnable) {
      if (HasAllRequiredComponents(e) && (GetNumExcludedComponents(e) == 1)) {
        // This entity will meet all requirements, now that the last excluded component is being
        // removed.
        this->onEntityEnable(reg, e, true);
      }
    }
  }

  using EcsPool = entt::sparse_set const*;
  using EcsObserver = entt::sink<entt::sigh<void(entt::registry&, entt::entity)>>;

  // Required:
  std::vector<EcsPool> requiredComponentTypes;
  std::vector<EcsObserver> observeRequiredOnConstruct;
  std::vector<EcsObserver> observeRequiredOnDestroy;

  // Excluded:
  std::vector<EcsPool> excludedComponentTypes;
  std::vector<EcsObserver> observeExcludedOnConstruct;
  std::vector<EcsObserver> observeExcludedOnDestroy;

  bool isEnabled = false;
};

/**************************************************************************************************
  DebugDrawImpl
*/
class DebugDrawImpl final : public DebugDrawInternal {
 public:
  explicit DebugDrawImpl(entt::registry& registry)
      : _myThreadId(std::this_thread::get_id()), _registry(registry) {}

  ~DebugDrawImpl() override {
    // Set thread affinity to prevent assertion failures in case this is not the same thread that
    // created and used the object.
    SetThreadAffinity();

    // Disable all systems and disconnect registry callbacks
    Enable(false);
  }

  // DebugDraw Public API:
  bool IsEnabled() const override;
  void Enable(bool enable) override;
  int GetNumFeatures() const override;
  int FindFeature(std::string_view name) const override;
  std::string_view GetFeatureName(int index) const override;
  std::string_view GetFeatureDescription(int index) const override;
  bool IsFeatureEnabled(int index) const override;
  void EnableFeature(int index, bool enable) override;
  DebugDrawData GatherData() override;
  void EnableActor(ActorHandle actor, bool enable, Error& error) override;

  // DebugDrawInternal API:
  void SetThreadAffinity() override {
    _myThreadId = std::this_thread::get_id();
  }
  entt::registry& GetRegistry() override {
    return _registry;
  }
  void RegisterSystemImpl(
      DebugDrawSystem&& system,
      Span<EcsComponentSet const> requiredTypes = {},
      Span<EcsComponentObserver const> onRequiredConstruct = {},
      Span<EcsComponentObserver const> onRequiredDestroy = {},
      Span<EcsComponentSet const> excludedTypes = {},
      Span<EcsComponentObserver const> onExcludedConstruct = {},
      Span<EcsComponentObserver const> onExcludedDestroy = {}) override;
  void FinalizeSystems() override {
    MOCHI_ASSERT(!_hasFinalizedSystems, "Redundant call");
    _hasFinalizedSystems = true;
    std::sort(_systems.begin(), _systems.end(), [](auto const& a, auto const& b) {
      return a->name < b->name;
    });
  }

 private:
  void OnSystemEnable(DebugDrawSystemImpl* system, bool enable);

  std::thread::id _myThreadId;
  entt::registry& _registry;
  bool _isEnabled = false;
  bool _hasFinalizedSystems = false;
  std::vector<std::unique_ptr<DebugDrawSystemImpl>> _systems;
  mutable std::vector<DebugDrawSystemImpl*> _enabledSystems; // cached list of just the enabled ones
  mutable bool _enabledSystemsDirty = false; // When to update _enabledSystems
  DebugDrawCollectorImpl _collector;
};

} // namespace

bool DebugDrawImpl::IsEnabled() const {
  MOCHI_ASSERT(std::this_thread::get_id() == _myThreadId);
  return _isEnabled;
}

void DebugDrawImpl::Enable(bool enable) {
  MOCHI_ASSERT(std::this_thread::get_id() == _myThreadId);
  if (enable != _isEnabled) {
    _isEnabled = enable;

    // Notify systems
    if (enable) {
      for (auto& system : _systems) {
        if (system->isEnabled) {
          // The system wanted to be enabled, but now it really can be.
          OnSystemEnable(system.get(), true);
        }
      }
    } else {
      for (auto& system : _systems) {
        if (system->isEnabled) {
          // Tell the system that it is no longer enabled, even though we preserve
          // the DebugDrawSystemImpl::isEnabled flag.
          OnSystemEnable(system.get(), false);
        }
      }
    }
  }
}

int DebugDrawImpl::GetNumFeatures() const {
  MOCHI_ASSERT(std::this_thread::get_id() == _myThreadId);
  return isize(_systems);
}

int DebugDrawImpl::FindFeature(std::string_view name) const {
  MOCHI_ASSERT(std::this_thread::get_id() == _myThreadId);
  for (int i = 0; i < isize(_systems); ++i) {
    if (_systems[i]->name == name) {
      return i;
    }
  }
  return -1;
}

void DebugDrawImpl::EnableFeature(int index, bool enable) {
  MOCHI_ASSERT(std::this_thread::get_id() == _myThreadId);
  MOCHI_ASSERT(index < isize(_systems));

  DebugDrawSystemImpl* system = _systems[index].get();
  if (enable != system->isEnabled) {
    // Mark it enabled/disabled
    system->isEnabled = enable;

    // If DebugDraw is already enabled overall, then notify the system immediately.
    if (_isEnabled) {
      OnSystemEnable(system, enable);
    }

    // Rebuilt _enabledSystems next time it is needed
    _enabledSystemsDirty = true;
  }
}

void DebugDrawImpl::OnSystemEnable(DebugDrawSystemImpl* system, bool enable) {
  if (enable) {
    // Notify the system that it is enabled
    if (system->onSystemEnable) {
      system->onSystemEnable(_registry, true);
    }

    // Then notify it about entities that it cares about
    if (system->onEntityEnable) {
      entt::runtime_view view{system->requiredComponentTypes, system->excludedComponentTypes};
      for (entt::entity e : view) {
        system->onEntityEnable(_registry, e, true);
      }
    }

    // Hook up observers so we know when any of the required components
    // are added/removed from entities
    for (auto& sink : system->observeRequiredOnConstruct) {
      sink.connect<&DebugDrawSystemImpl::OnRequiredComponentConstruct>(system);
    }
    for (auto& sink : system->observeRequiredOnDestroy) {
      sink.connect<&DebugDrawSystemImpl::OnRequiredComponentDestroy>(system);
    }
    for (auto& sink : system->observeExcludedOnConstruct) {
      sink.connect<&DebugDrawSystemImpl::OnExcludedComponentConstruct>(system);
    }
    for (auto& sink : system->observeExcludedOnDestroy) {
      sink.connect<&DebugDrawSystemImpl::OnExcludedComponentDestroy>(system);
    }
  } else {
    // Same as above, but in reverse order
    for (auto& sink : system->observeRequiredOnConstruct) {
      sink.disconnect<&DebugDrawSystemImpl::OnRequiredComponentConstruct>(system);
    }
    for (auto& sink : system->observeRequiredOnDestroy) {
      sink.disconnect<&DebugDrawSystemImpl::OnRequiredComponentDestroy>(system);
    }
    for (auto& sink : system->observeExcludedOnConstruct) {
      sink.disconnect<&DebugDrawSystemImpl::OnExcludedComponentConstruct>(system);
    }
    for (auto& sink : system->observeExcludedOnDestroy) {
      sink.disconnect<&DebugDrawSystemImpl::OnExcludedComponentDestroy>(system);
    }
    if (system->onEntityEnable) {
      entt::runtime_view view{system->requiredComponentTypes, system->excludedComponentTypes};
      for (entt::entity e : view) {
        system->onEntityEnable(_registry, e, false);
      }
    }
    if (system->onSystemEnable) {
      system->onSystemEnable(_registry, false);
    }
  }
}

std::string_view DebugDrawImpl::GetFeatureName(int index) const {
  MOCHI_ASSERT(std::this_thread::get_id() == _myThreadId);
  MOCHI_ASSERT(index < isize(_systems));
  return _systems[index]->name;
}

std::string_view DebugDrawImpl::GetFeatureDescription(int index) const {
  MOCHI_ASSERT(std::this_thread::get_id() == _myThreadId);
  MOCHI_ASSERT(index < isize(_systems));
  return _systems[index]->description;
}

bool DebugDrawImpl::IsFeatureEnabled(int index) const {
  MOCHI_ASSERT(std::this_thread::get_id() == _myThreadId);
  MOCHI_ASSERT(index < isize(_systems));
  return _systems[index]->isEnabled;
}

DebugDrawData DebugDrawImpl::GatherData() {
  MOCHI_ASSERT(std::this_thread::get_id() == _myThreadId);
  MOCHI_PROFILE_SCOPE();
  _collector.ClearData();
  if (!_isEnabled) {
    return {};
  }

  // Repopulate _enabledSystems if systems were enabled/disabled.
  // We could update it incrementally but this seems more robust.
  if (_enabledSystemsDirty) {
    _enabledSystems.clear();
    for (auto& system : _systems) {
      if (system->isEnabled) {
        _enabledSystems.push_back(system.get());
      }
    }
    // Sort by decreasing depth so that low depth draws last
    std::sort(_enabledSystems.begin(), _enabledSystems.end(), [](auto const* a, auto const* b) {
      return a->sortingDepth > b->sortingDepth;
    });
    _enabledSystemsDirty = false;
  }

  for (DebugDrawSystemImpl* system : _enabledSystems) {
    MOCHI_PROFILE_SCOPE_N("DebugDrawSystem");
    MOCHI_PROFILE_LABEL(system->name);

    _collector.ClearTransform();

    if (system->onDrawScene) {
      // Let the system draw whatever it wants in world-space
      system->onDrawScene(_registry, _collector);
    }

    if (system->onDrawEntityWorldSpace) {
      // Iterate over the entities with the required components and call the system.
      // Data is in world-space and may simply pass through to the caller's Collector.
      entt::runtime_view view{system->requiredComponentTypes, system->excludedComponentTypes};
      for (entt::entity e : view) {
        system->onDrawEntityWorldSpace(_registry, e, _collector);
      }
    }

    if (system->onDrawEntityLocalSpace) {
      // Iterate over the entities with the required components and call the system. This time,
      // transform all the data using the entity's CRootTransform component. Note that
      // CRootTransform is always a member of requiredComponentTypes in this case.
      entt::runtime_view view{system->requiredComponentTypes, system->excludedComponentTypes};
      for (entt::entity e : view) {
        auto const& root = _registry.get<CRootTransform>(e);
        _collector.SetTransform(root.worldFromLocal);
        system->onDrawEntityLocalSpace(_registry, e, _collector);
      }
    }
  }

  return DebugDrawData{
      .lineVertices =
          {.positions = MakeConstSpan(_collector.allLinePositions),
           .colors = MakeConstSpan(_collector.allLineColors)},
      .spheres =
          {.positions = MakeConstSpan(_collector.allSpherePositions),
           .radii = MakeConstSpan(_collector.allSphereRadii),
           .colors = MakeConstSpan(_collector.allSphereColors)},
  };
}

void DebugDrawImpl::EnableActor(ActorHandle actor, bool enable, Error& error) {
  MOCHI_ERROR_RETURN(error);
  auto e = GetEntity(_registry, actor, error);
  MOCHI_ERROR_RETURN(error);
  if (enable) {
    _registry.remove<TagExcludedFromDebugDraw>(e);
  } else {
    _registry.emplace_or_replace<TagExcludedFromDebugDraw>(e);
  }
}

// If a component is not already required by the system, then add it and hook up the
// necessary callbacks.
template <class ComponentT>
void EnsureSystemRequiresComponent(entt::registry& reg, DebugDrawSystemImpl* systemImpl) {
  auto const* typeToAdd = &reg.storage<ComponentT>();
  auto typeIt = std::find(
      systemImpl->requiredComponentTypes.begin(),
      systemImpl->requiredComponentTypes.end(),
      typeToAdd);
  if (typeIt == systemImpl->requiredComponentTypes.end()) {
    systemImpl->requiredComponentTypes.push_back(typeToAdd);
    systemImpl->observeRequiredOnConstruct.emplace_back(reg.on_construct<ComponentT>());
    systemImpl->observeRequiredOnDestroy.emplace_back(reg.on_destroy<ComponentT>());
  }
}

void DebugDrawImpl::RegisterSystemImpl(
    DebugDrawSystem&& system,
    Span<EcsComponentSet const> requiredTypes,
    Span<EcsComponentObserver const> onRequiredConstruct,
    Span<EcsComponentObserver const> onRequiredDestroy,
    Span<EcsComponentSet const> excludedTypes,
    Span<EcsComponentObserver const> onExcludedConstruct,
    Span<EcsComponentObserver const> onExcludedDestroy) {
  MOCHI_ASSERT(
      !_hasFinalizedSystems, "Illegal to register more systems after FinalizeSystems was called");
  if (FindFeature(system.name) != -1) {
    MOCHI_ASSERT(false, "All DebugDrawSystem names must be unique.");
    return;
  }

  auto systemImpl = std::make_unique<DebugDrawSystemImpl>(std::move(system));

  // Reserve memory
  int reserveSize = isize(requiredTypes) + (systemImpl->onDrawEntityLocalSpace ? 2 : 1);
  systemImpl->requiredComponentTypes.reserve(reserveSize);
  systemImpl->observeRequiredOnConstruct.reserve(reserveSize);
  systemImpl->observeRequiredOnDestroy.reserve(reserveSize);

  // All systems implicitly require TagFullyInitialized. That way, we don't have to worry about
  // onEntityEnable notifications happening in the middle of actor initialization. Register this one
  // first so that we check it first, thus reducing overhead for partially constructed/destroyed
  // actors.
  EnsureSystemRequiresComponent<TagFullyInitialized>(_registry, systemImpl.get());

  // If the system wants to draw in local space, then it implicitly requires a CRootTransform.
  if (systemImpl->onDrawEntityLocalSpace) {
    EnsureSystemRequiresComponent<CRootTransform>(_registry, systemImpl.get());
  }

  // Required components
  Append(systemImpl->requiredComponentTypes, requiredTypes);
  Append(systemImpl->observeRequiredOnConstruct, onRequiredConstruct);
  Append(systemImpl->observeRequiredOnDestroy, onRequiredDestroy);

  // Excluded components
  Append(systemImpl->excludedComponentTypes, excludedTypes);
  Append(systemImpl->observeExcludedOnConstruct, onExcludedConstruct);
  Append(systemImpl->observeExcludedOnDestroy, onExcludedDestroy);

  _systems.emplace_back(std::move(systemImpl));
}

// Static Method
std::unique_ptr<DebugDrawInternal> DebugDrawInternal::Create(entt::registry& registry) {
  return std::make_unique<DebugDrawImpl>(registry);
}

} // namespace mochi
