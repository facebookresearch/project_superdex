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

#include <superdex_robotics/superdex_robotics.h>

#include <string_view>
#include <vector>

namespace superdex::robotics {
using namespace mochi;

class BotImpl : public Bot {
 public:
  BotImpl(Actor* actor, Scene* scene, RoboticsContext* botsContext, BotPrefab botPrefab);
  ~BotImpl() override;

  BotImpl(BotImpl const&) = delete;
  BotImpl& operator=(BotImpl const&) = delete;
  BotImpl(BotImpl&&) = delete;
  BotImpl& operator=(BotImpl&&) = delete;

  void SetHandle(BotHandle handle) {
    _handle = handle;
  }

  [[nodiscard]] BotHandle GetHandle() const override;

  [[nodiscard]] Context* GetMochiContext() override;
  [[nodiscard]] Context const* GetMochiContext() const override;

  [[nodiscard]] RoboticsContext* GetBotContext() override;
  [[nodiscard]] RoboticsContext const* GetBotContext() const override;

  [[nodiscard]] Scene* GetScene() override;
  [[nodiscard]] Scene const* GetScene() const override;

  [[nodiscard]] char const* GetName() const override;

  [[nodiscard]] BotPrefab const& GetBotPrefab() const override;

  [[nodiscard]] Actor* GetArticulatedActor() const override;

  [[nodiscard]] ControllerBase*
  CreateController(std::string_view typeName, std::string_view name, Error& error) override;

  [[nodiscard]] SensorBase* CreateSensor(
      std::string_view typeName,
      std::string_view linkName,
      std::string_view name,
      std::string_view paramArgs,
      Error& error) override;

  [[nodiscard]] DynamicArray<ControllerHandle> GetControllerHandles() const override;

  [[nodiscard]] ControllerBase* GetController(ControllerHandle controllerHandle, Error& error)
      const override;

  [[nodiscard]] DynamicArray<SensorHandle> GetSensorHandles() const override;

  [[nodiscard]] SensorBase* GetSensor(SensorHandle sensorHandle, Error& error) const override;

  [[nodiscard]] char const* GetSensorLinkName(SensorHandle sensorHandle, Error& error)
      const override;

  void CreateSensors(Error& error);

  [[nodiscard]] ActuatorBase* CreateActuator(
      std::string_view typeName,
      std::string_view linkName,
      std::string_view name,
      std::string_view paramArgs,
      Error& error) override;

  [[nodiscard]] DynamicArray<ActuatorHandle> GetActuatorHandles() const override;

  [[nodiscard]] ActuatorBase* GetActuator(ActuatorHandle actuatorHandle, Error& error)
      const override;

  [[nodiscard]] char const* GetActuatorLinkName(ActuatorHandle actuatorHandle, Error& error)
      const override;

  void CreateActuators(Error& error);

  /* @brief Resolve and cache the actors backing this bot's prefab links. Must run before
   * CreateSensors / CreateActuators (which index into the cache) and before any component is
   * created against a link actor, since @ref OwnsActor answers from it. */
  void CacheLinkActors(Error& error);

  /* @brief Whether @p actorHandle is this bot's articulation actor or one of its link actors. Used
   * by RoboticsContext to infer the owning bot of a component created from a bare actor. */
  [[nodiscard]] bool OwnsActor(ActorHandle actorHandle) const;

  /* Per-bot Find forwarders: scope the RoboticsContext finders to this bot. */
  [[nodiscard]] DynamicArray<ControllerHandle> FindControllersByName(
      std::string_view name) const override {
    return _botsContext->FindControllersByName(name, this);
  }
  [[nodiscard]] DynamicArray<ControllerHandle> FindControllersByType(
      std::string_view typeName) const override {
    return _botsContext->FindControllersByType(typeName, this);
  }
  [[nodiscard]] DynamicArray<SensorHandle> FindSensorsByName(std::string_view name) const override {
    return _botsContext->FindSensorsByName(name, this);
  }
  [[nodiscard]] DynamicArray<SensorHandle> FindSensorsByType(
      std::string_view typeName) const override {
    return _botsContext->FindSensorsByType(typeName, this);
  }
  [[nodiscard]] DynamicArray<ActuatorHandle> FindActuatorsByName(
      std::string_view name) const override {
    return _botsContext->FindActuatorsByName(name, this);
  }
  [[nodiscard]] DynamicArray<ActuatorHandle> FindActuatorsByType(
      std::string_view typeName) const override {
    return _botsContext->FindActuatorsByType(typeName, this);
  }

 private:
  /* @brief Resolve the owning scene from its stale-safe handle. Returns nullptr once the scene has
   * been destroyed (e.g. the mochi Scene was torn down before this bot / the RoboticsContext). */
  [[nodiscard]] Scene* ResolveScene() const;

  /* @brief Index into _botPrefab.links (and the parallel _linkActors) of the link named
   * @p linkName, or -1 if this bot has no such link. */
  [[nodiscard]] int FindLinkIndex(std::string_view linkName) const;

  /* @brief Index into _botPrefab.links of the link backed by @p linkActor, or @ref kIndexNone if
   * it is not one of this bot's link actors (e.g. its articulation actor, or null). */
  [[nodiscard]] int FindLinkActorIndex(Actor const* linkActor) const;

  ActorHandle _actor;
  /* Owning scene held as a stale-safe handle plus its owning context (the outermost mochi lifetime,
   * safe to hold raw) rather than a raw Scene*, so this bot never dereferences a dangling scene. */
  SceneHandle _sceneHandle;
  Context* _context = nullptr;
  RoboticsContext* _botsContext = nullptr;
  BotHandle _handle;
  BotPrefab _botPrefab;
  /* Actors backing _botPrefab.links, in prefab order; populated once by CacheLinkActors. */
  DynamicArray<ActorHandle> _linkActors;
};

} // namespace superdex::robotics
