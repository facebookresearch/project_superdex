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

#include "mochi_ecs.h"

#include <mochi_physics/mochi_physics.h>

#include <functional>
#include <memory>
#include <string>
#include <utility>

namespace mochi {

/**
 * Represents a sphere for debug visualization.
 * Used to render spherical shapes in debug drawing operations.
 */
struct DebugDrawSphere {
  Real3 position = {}; // World-space position of the sphere center.
  real radius = 0_r; // Radius of the sphere in world units.
  Color color = colors::kWhite; // RGBA color for rendering the sphere.
};

/**
 * Represents a single vertex for debug line rendering.
 * Two vertices are typically used together to define a line segment.
 */
struct DebugDrawLineVertex {
  Real3 position = {}; // World-space position of the vertex.
  Color color = colors::kWhite; // RGBA color at this vertex (allows for gradient lines).
};

/**************************************************************************************************
  DebugDrawCollector
    - Internal API used by DebugDrawSystems
    - Extends the DebugDraw::Collector API with helper functions
*/
class DebugDrawCollector {
 public:
  // Add a line segment for each pair of LineVertex (must be multiple of 2).
  virtual void AddLines(Span<DebugDrawLineVertex const> lines) = 0;

  // Add a single line segment
  void AddLine(DebugDrawLineVertex const& start, DebugDrawLineVertex const& end) {
    DebugDrawLineVertex verts[2] = {start, end};
    AddLines(MakeSpan(verts));
  }

  // Add RGB lines for the XYZ basis vectors of a transformed coordinate frame
  virtual void AddTriAxis(TransformRT const& transform, real scale) = 0;

  // Add a wireframe axis-aligned box with uniform color
  virtual void AddWireframeAabb(Aabb const& aabb, Color color) = 0;

  // Add a wireframe axis-aligned box with uniform color
  virtual void AddWireframeAabb(Real3 const& center, real halfWidth, Color c) = 0;

  // Add a wireframe box with uniform color
  virtual void AddWireframeObb(Obb const& oobb, Color color) = 0;

  // Add a wireframe triangle mesh with uniform color. Optionally transform the coordinates.
  virtual void AddWireframeMesh(
      Span<Real3 const> nodes,
      Span<Int2 const> edges,
      Color uniformColor,
      TransformRT const& localTransform = {}) = 0;

  // Add a wireframe triangle mesh with per-vertex color
  virtual void AddWireframeMesh(
      Span<Real3 const> nodes,
      Span<Int2 const> edges,
      Span<Color const> nodeColors) = 0;

  // Add a sphere of uniform color
  virtual void AddSphere(DebugDrawSphere const& sphere) = 0;

 protected:
  virtual ~DebugDrawCollector() = default;
};

/**************************************************************************************************
  DebugDrawSystem
    - Implements a "feature" that the user can enable/disable using the mochi::DebugDraw API.
    - Registered with DebugDrawImpl along with a list of required ECS components.
    - Called by DebugDrawImpl when and if data is needed.
    - May be called concurrently for many systems and entities.
*/
struct DebugDrawSystem {
  using DrawSceneFn = std::function<void(entt::registry const& reg, DebugDrawCollector& out)>;
  using DrawEntityFn =
      std::function<void(entt::registry const& reg, entt::entity e, DebugDrawCollector& out)>;
  using SystemEnableFn = std::function<void(entt::registry& reg, bool enabled)>;
  using EntityEnableFn = std::function<void(entt::registry& reg, entt::entity, bool enabled)>;

  // Feature name reported to the user. Must be unique.
  std::string name;

  // Descriptive string for tool tips
  std::string description;

  // When multiple systems are enabled, they will be drawn in reverse order of sortingDepth. Thus, a
  // system with lower sortingDepth will draw on top of a system with higher sorting depth. This
  // generally only matters if visuals overlap spatially. Can be negative.
  real sortingDepth = 0_r;

  // Optional:
  //  Implement this function if you want to output debug data one entity at a time using
  //  local-space coordinates. DebugDrawImpl will ensure that it is only called for entities with
  //  the required components, and it will automatically transform the results to world-space using
  //  the entity's CRootTransform component. This is generally the most convenient option.
  DrawEntityFn onDrawEntityLocalSpace;

  // Optional:
  //  Similar to onDrawEntityLocalSpace except that the output data will not be transformed. Use
  //  this version if the data is already in world-space.
  DrawEntityFn onDrawEntityWorldSpace;

  // Optional:
  //  Implement this function if you want to output all your debug data for the scene at one time.
  //  This gives you full control. All output data should be in world-space coordinates.
  DrawSceneFn onDrawScene;

  // Optional:
  //  Implement this function if you have some global setup/cleanup to do when the system is
  //  enabled/disabled. See also onEntityEnable.
  SystemEnableFn onSystemEnable;

  // Optional:
  //  Implement this function if you have some per-entity setup/cleanup to do when the system is
  //  enabled of a specific entity.
  EntityEnableFn onEntityEnable;
};

/**************************************************************************************************
  DebugDrawInternal
    - Extends the public DebugDraw API with features only used inside MochiPhysics
    - Implemented in MochiDebugDraw.cpp
*/
class DebugDrawInternal : public DebugDraw {
 public:
  // Register your DebugDrawSystem along with its list of required components/tags (if any).
  // System functions will only be invoked for entities that have ALL the required components/tags.
  //
  // Example:
  //    debugDraw.RegisterSystem<CBoundingVolume>(mySystemThatDrawsBoundingVolumes).
  template <class... RequiredComponentT>
  void RegisterSystem(DebugDrawSystem system);

  // Register your DebugDrawSystem with an additional list of excluded components/tags.
  template <class... RequiredComponentT, class... ExcludedComponentTypes>
  void RegisterSystem(DebugDrawSystem system, ecs::Excluded<ExcludedComponentTypes...>);

  // Call this once after all systems have been registered
  virtual void FinalizeSystems() = 0;

  // Allow the API to be used on the current thread.
  virtual void SetThreadAffinity() = 0;

  // Create an implementation of this interface
  static std::unique_ptr<DebugDrawInternal> Create(entt::registry& registry);

 protected:
  using EcsComponentSet = entt::sparse_set const*;
  using EcsComponentObserver = entt::sink<entt::sigh<void(entt::registry&, entt::entity)>>;
  virtual entt::registry& GetRegistry() = 0;
  virtual void RegisterSystemImpl(
      DebugDrawSystem&& system,
      Span<EcsComponentSet const> requiredTypes = {},
      Span<EcsComponentObserver const> onRequiredConstruct = {},
      Span<EcsComponentObserver const> onRequiredDestroy = {},
      Span<EcsComponentSet const> excludedTypes = {},
      Span<EcsComponentObserver const> onExcludedConstruct = {},
      Span<EcsComponentObserver const> onExcludedDestroy = {}) = 0;
};

// Template implementation details
template <class... RequiredComponentT>
inline void DebugDrawInternal::RegisterSystem(DebugDrawSystem system) {
  if constexpr (sizeof...(RequiredComponentT) != 0) {
    entt::registry& reg = GetRegistry();
    EcsComponentSet requiredTypes[] = {&reg.storage<RequiredComponentT>()...};
    EcsComponentObserver onRequiredConstruct[] = {reg.on_construct<RequiredComponentT>()...};
    EcsComponentObserver onRequiredDestroy[] = {reg.on_destroy<RequiredComponentT>()...};
    RegisterSystemImpl(
        std::move(system),
        MakeSpan(requiredTypes),
        MakeSpan(onRequiredConstruct),
        MakeSpan(onRequiredDestroy));
  } else {
    RegisterSystemImpl(std::move(system));
  }
}

// Template implementation details
template <class... RequiredComponentT, class... ExcludedComponentTypes>
inline void DebugDrawInternal::RegisterSystem(
    DebugDrawSystem system,
    ecs::Excluded<ExcludedComponentTypes...>) {
  static_assert(sizeof...(ExcludedComponentTypes) > 0);
  entt::registry& reg = GetRegistry();
  EcsComponentSet excludedTypes[] = {&reg.storage<ExcludedComponentTypes>()...};
  EcsComponentObserver onExcludedConstruct[] = {reg.on_construct<ExcludedComponentTypes>()...};
  EcsComponentObserver onExcludedObservers[] = {reg.on_destroy<ExcludedComponentTypes>()...};
  if constexpr ((sizeof...(RequiredComponentT) > 0)) {
    EcsComponentSet requiredTypes[] = {&reg.storage<RequiredComponentT>()...};
    EcsComponentObserver onRequiredConstruct[] = {reg.on_construct<RequiredComponentT>()...};
    EcsComponentObserver onRequiredDestroy[] = {reg.on_destroy<RequiredComponentT>()...};
    RegisterSystemImpl(
        std::move(system),
        MakeSpan(requiredTypes),
        MakeSpan(onRequiredConstruct),
        MakeSpan(onRequiredDestroy),
        MakeSpan(excludedTypes),
        MakeSpan(onExcludedConstruct),
        MakeSpan(onExcludedObservers));
  } else {
    RegisterSystemImpl(
        std::move(system),
        {},
        {},
        {},
        MakeSpan(excludedTypes),
        MakeSpan(onExcludedConstruct),
        MakeSpan(onExcludedObservers));
  }
}

} // namespace mochi
