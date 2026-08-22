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

#include <superdex_robotics/core/component_base.h>

#include <string>
#include <string_view>

namespace superdex::robotics {

struct BotPrefab;

/* @brief Base class for all controllers.
 * Inherits common lifecycle from ComponentBase. Controllers are identified
 * by string names via RoboticsContext registration, not enums. */
// NOLINTNEXTLINE(cppcoreguidelines-virtual-class-destructor)
class MOCHI_API ControllerBase : public ComponentBase {
 public:
  /* @brief Reset internal controller state (e.g., between episodes). */
  void Reset() override = 0;

  /* @brief The robot-model description this controller was constructed with, or nullptr if none.
   * Borrowed (not owned); the caller guarantees it outlives the controller. Optional — controllers
   * that need no model data ignore it. */
  [[nodiscard]] BotPrefab const* GetBotPrefab() const {
    return _prefab;
  }

  /* @brief Configure this controller from serialized parameter and initialization arguments.
   * @p paramArgs is either a params file path or inline JSON; empty uses defaults. @p initArgs is a
   * controller-specific file path or inline JSON, for example carrying baseLinkName/eeLinkName for
   * link-resolved controllers. Implementations typically deserialize it into a small reflected
   * init-args struct and ignore unrecognized framework keys.
   *
   * Pure virtual: every controller must implement this uniform configuration entry point. A
   * controller that cannot support serialized configuration should set @p error with a clear
   * message rather than silently succeeding.
   * @param paramArgs Params file path or inline JSON; empty uses defaults.
   * @param initArgs Controller-specific init-argument file path or inline JSON.
   * @param error Error status. */
  virtual void ConfigureFromSceneEntry(
      std::string_view paramArgs,
      std::string_view initArgs,
      superdex::Error& error) = 0;

 protected:
  /* @brief Common controller constructor archetype: stores the (optionally null) @p actor via
   * ComponentBase together with the optional borrowed @p prefab (see GetBotPrefab). A null Actor is
   * permitted so controllers can be constructed without a live Mochi scene (e.g. on a real robot);
   * controllers that require an Actor enforce that themselves. A non-null Actor must be an
   * articulated body. Every controller delegates to this, then bails on error before performing its
   * own initialization. */
  ControllerBase(BotPrefab const* prefab, Actor* actor, superdex::Error& error)
      : ComponentBase(actor), _prefab(prefab) {
    if (actor != nullptr) {
      MOCHI_ERROR_IF(
          actor->GetType() != ActorType::Articulated, error, "Actor is not an articulated body");
    }
  }

  /* @brief Build an actor-scoped link name ("<botActorName>/<linkName>") for link resolution during
   * ConfigureFromSceneEntry. Requires a non-null Actor (link-resolved controllers need one). */
  [[nodiscard]] std::string QualifiedLinkName(std::string_view linkName) const {
    Actor* const actor = GetActor();
    char const* actorName = actor != nullptr ? actor->GetName() : nullptr;
    return std::string(actorName != nullptr ? actorName : "") + "/" + std::string(linkName);
  }

  /* Borrowed robot-model description (see GetBotPrefab); nullptr when constructed without a model.
   */
  BotPrefab const* _prefab = nullptr;
};

} // namespace superdex::robotics
