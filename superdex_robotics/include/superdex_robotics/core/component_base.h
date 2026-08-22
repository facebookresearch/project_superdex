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

#include <mochi_physics/mochi_physics.h>
#include <superdex_physics.h>

#include <string>
#include <string_view>
#include <utility>

namespace superdex::robotics {

class RoboticsContext; /* forward declaration for friend access */
class Bot; /* forward declaration for GetOwningBot (raw pointer, not a handle) */

/* @brief Base class for all bots components (controllers, sensors, actuators).
 * Provides common state: the associated actor's stale-safe identity, a validity flag, and the
 * Destroy pattern. The validity flag is set to false by RoboticsContext before deletion, allowing
 * dangling-pointer detection alongside generational handles.
 *
 * The associated actor is NOT cached as a raw pointer. Its stale-safe identity (@ref ActorHandle /
 * @ref SceneHandle / owning @ref Context) is retained and @ref GetActor resolves the live
 * @ref Actor* on demand, yielding nullptr once the owning scene (or the actor) has been destroyed.
 * A component that outlives its mochi Scene (e.g. the Scene was destroyed before the
 * RoboticsContext) therefore never dereferences a dangling Actor* — callers check @ref GetActor for
 * null. This assumes the mochi @ref Context outlives the RoboticsContext — the outermost lifetime
 * in the mochi ownership graph. */
class MOCHI_API ComponentBase {
 public:
  ComponentBase(ComponentBase const&) = delete;
  ComponentBase& operator=(ComponentBase const&) = delete;
  ComponentBase(ComponentBase&&) = delete;
  ComponentBase& operator=(ComponentBase&&) = delete;

  /* @brief Whether this component remains usable. Returns false after its owning
   * @ref RoboticsContext destroys it.
   * @return True while the component is live. */
  [[nodiscard]] bool IsValid() const {
    return _isValid;
  }
  /* @brief Resolve the live @ref Actor* this component is bound to, or nullptr if it was
   * constructed without an actor or its owning scene/actor has since been destroyed. Resolved on
   * demand from the stale-safe handle each call (an O(1) scene lookup), so it is always current and
   * never returns a dangling pointer — dereference only after a null check.
   * @return Live associated actor, or nullptr if none is available. */
  [[nodiscard]] Actor* GetActor() const;

  /* @brief The bot this component was created on, or nullptr if it was created directly on an actor
   * (not via a bot) — e.g. a scene-level sensor or a controller built on a raw articulation. Used
   * to scope FindByName / FindByType to a single bot and to drive DestroyBot's component cascade.
   * Borrowed: the owning bot is owned by the RoboticsContext, which destroys a bot's components
   * before the bot itself, so this never dangles. Held as a raw pointer (not a handle) because a
   * bot is not scene-owned, so the stale-safety the actor handles provide is unnecessary here.
   * @return Owning bot, or nullptr if this component has no owning bot. */
  [[nodiscard]] Bot* GetOwningBot() const {
    return _owningBot;
  }

  /* @brief This component's instance name, or empty if it was created without one. Names are NOT
   * required to be unique, so name-based lookups (FindByName) may return multiple matches.
   * @return Instance name; empty if unnamed. */
  [[nodiscard]] std::string_view GetName() const {
    return {_name.c_str(), _name.size()};
  }

  /* @brief This component's registered type name (e.g. "BASIC_OSC_PD", "SENSOR_CAMERA"), reported
   * polymorphically. Declared here so controllers, sensors, and actuators can be searched by type
   * uniformly (FindByType); each component kind implements it.
   * @return Registered component type name. */
  [[nodiscard]] virtual std::string_view GetTypeName() const = 0;

  /* @brief Reset internal state without destroying the component.
   * Parameters are preserved. Use between episodes to clear stale state
   * (e.g., derivative tracking, timing diagnostics). */
  virtual void Reset() = 0;

  /* @brief Destroy a component created by a static Create function.
   * Sets the validity flag to false before deletion.
   * @param component Pointer to the component to destroy. May be nullptr (no-op).
   * Only callable by RoboticsContext — destroy via RoboticsContext methods. */
  static void Destroy(ComponentBase* component);

 protected:
  /* @brief Capture @p actor's stale-safe identity (handle, owning scene handle, and owning context)
   * so the live actor can be resolved on demand via @ref GetActor. A null @p actor is permitted
   * (actor-less components such as fusion sensors or real-robot controllers). */
  explicit ComponentBase(Actor* actor);

  /* @brief Mark this component as invalid. Called by RoboticsContext before deletion. */
  void SetInvalid() {
    _isValid = false;
  }

  /* @brief The owning mochi @ref Context, or nullptr if constructed without an actor. The context
   * is the outermost lifetime in the mochi ownership graph, so (under the standing contract that it
   * outlives the RoboticsContext) this raw pointer stays valid for the component's whole life.
   * Useful in destructors to release context-owned resources even after the actor/scene are gone.
   */
  [[nodiscard]] Context* GetContext() const {
    return _context;
  }

  /* @brief Resolve the owning mochi @ref Scene from its stale-safe handle, or nullptr if this
   * component was constructed without an actor or its owning scene has since been destroyed. Prefer
   * this over caching a raw @ref Scene*: a component that outlives its scene resolves to nullptr
   * (so callers can bail) instead of dereferencing a dangling pointer. */
  [[nodiscard]] Scene* GetScene() const;

  // NOLINTNEXTLINE(cppcoreguidelines-virtual-class-destructor)
  virtual ~ComponentBase();

 private:
  friend class RoboticsContext;

  /* @brief Record the bot this component was created on. Called by RoboticsContext at creation
   * time; left null for components created directly on an actor. */
  void SetOwningBot(Bot* bot) {
    _owningBot = bot;
  }

  /* @brief Set this component's instance name. Called by RoboticsContext at creation time. Copies
   * @p name into an owned, null-terminated string (the view need not outlive the call). */
  void SetName(std::string_view name) {
    _name = DynamicString(std::string(name).c_str());
  }

  /* Stale-safe identity of the associated actor and its owning scene, plus the owning context used
   * to resolve them. Captured from the actor at construction; the handles stay meaningful after the
   * actor/scene are destroyed (resolution then yields nullptr). */
  ActorHandle _actorHandle;
  SceneHandle _sceneHandle;
  Context* _context = nullptr;
  /* The bot this component was created on (see GetOwningBot); null for actor-scoped components. */
  Bot* _owningBot = nullptr;
  /* This component's instance name (see GetName); empty when created without one. */
  DynamicString _name;
  bool _isValid = true;
};

} // namespace superdex::robotics
