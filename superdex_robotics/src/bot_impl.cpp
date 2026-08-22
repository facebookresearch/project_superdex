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

#include "bot_impl.h"

#include <superdex_robotics/core/context.h>

#include <string>
#include <utility>

using namespace mochi;
using namespace superdex::robotics;

BotImpl::BotImpl(Actor* actor, Scene* scene, RoboticsContext* botsContext, BotPrefab botPrefab)
    : _actor(actor->GetHandle()),
      _sceneHandle(scene != nullptr ? scene->GetHandle() : SceneHandle{}),
      _context(scene != nullptr ? scene->GetContext() : nullptr),
      _botsContext(botsContext),
      _botPrefab(std::move(botPrefab)) {}

BotImpl::~BotImpl() = default;

Scene* BotImpl::ResolveScene() const {
  return _context != nullptr ? _context->GetScene(_sceneHandle) : nullptr;
}

BotHandle BotImpl::GetHandle() const {
  return _handle;
}

Context* BotImpl::GetMochiContext() {
  return _context;
}

Context const* BotImpl::GetMochiContext() const {
  return _context;
}

RoboticsContext* BotImpl::GetBotContext() {
  return _botsContext;
}

RoboticsContext const* BotImpl::GetBotContext() const {
  return _botsContext;
}

Scene* BotImpl::GetScene() {
  return ResolveScene();
}

Scene const* BotImpl::GetScene() const {
  return ResolveScene();
}

char const* BotImpl::GetName() const {
  return _botPrefab.name.c_str();
}

BotPrefab const& BotImpl::GetBotPrefab() const {
  return _botPrefab;
}

Actor* BotImpl::GetArticulatedActor() const {
  if (Scene* const scene = ResolveScene()) {
    return scene->GetActor(_actor);
  }
  return nullptr;
}

ControllerBase*
BotImpl::CreateController(std::string_view typeName, std::string_view name, Error& error) {
  MOCHI_ERROR_RETURN(error, nullptr);
  MOCHI_ERROR_IF(_botsContext == nullptr, error, "Bots context is null");
  MOCHI_ERROR_RETURN(error, nullptr);
  /* The context infers this bot as the owner from the articulation actor, so no separate identity
   * stamp is needed here. */
  ControllerHandle const handle =
      _botsContext->CreateController(typeName, &_botPrefab, GetArticulatedActor(), name, error);
  MOCHI_ERROR_RETURN(error, nullptr);
  return _botsContext->GetController(handle);
}

void BotImpl::CacheLinkActors(Error& error) {
  MOCHI_ERROR_RETURN(error);
  if (_botPrefab.links.empty()) {
    return;
  }
  Actor* const articulatedActor = GetArticulatedActor();
  MOCHI_ERROR_IF(articulatedActor == nullptr, error, "Bot has no articulated actor");
  MOCHI_ERROR_RETURN(error);

  Span<ActorHandle const> const nestedLinks = articulatedActor->GetNestedLinkActors(error);
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(
      isize(nestedLinks) != isize(_botPrefab.links),
      error,
      "Articulated actor link count does not match BotPrefab link count.");
  MOCHI_ERROR_RETURN(error);

  _linkActors.clear();
  for (ActorHandle const linkActor : nestedLinks) {
    _linkActors.push_back(linkActor);
  }
}

int BotImpl::FindLinkActorIndex(Actor const* linkActor) const {
  if (linkActor == nullptr) {
    return kIndexNone;
  }
  ActorHandle const actorHandle = linkActor->GetHandle();
  for (int i = 0; i < isize(_linkActors); ++i) {
    if (_linkActors[i] == actorHandle) {
      return i;
    }
  }
  return kIndexNone;
}

bool BotImpl::OwnsActor(ActorHandle actorHandle) const {
  if (actorHandle == _actor) {
    return true;
  }
  for (ActorHandle const linkActor : _linkActors) {
    if (linkActor == actorHandle) {
      return true;
    }
  }
  return false;
}

void BotImpl::CreateSensors(Error& error) {
  MOCHI_ERROR_RETURN(error);
  if (_botPrefab.links.empty()) {
    return;
  }
  MOCHI_ERROR_IF(_botsContext == nullptr, error, "Bots context is null");
  Scene* const scene = ResolveScene();
  MOCHI_ERROR_IF(scene == nullptr, error, "Bot has no scene");
  MOCHI_ERROR_IF(
      isize(_linkActors) != isize(_botPrefab.links),
      error,
      "Bot link actors are not cached; CacheLinkActors must run first.");
  MOCHI_ERROR_RETURN(error);

  for (int i = 0; i < isize(_botPrefab.links); ++i) {
    auto const& linkPrefab = _botPrefab.links[i];
    if (linkPrefab.sensors.empty()) {
      continue;
    }
    Actor* linkActor = scene->GetActor(_linkActors[i]);
    MOCHI_ERROR_IF(linkActor == nullptr, error, "Failed to resolve link actor for sensor.");
    MOCHI_ERROR_RETURN(error);
    for (auto const& sensor : linkPrefab.sensors) {
      std::string_view const type(sensor.type.c_str(), sensor.type.size());
      /* A type this build does not have is skipped with a warning rather than failing the whole
       * bot, matching how the bot scene loader already treats scene-level sensors and controllers.
       * A prefab is routinely authored against a build with more sensor types than the one loading
       * it -- the open-source build ships bot assets whose sensors are Meta-internal, and every
       * such bot would otherwise be unloadable. Construction failures of a type that IS registered
       * still fail the bot: those mean the asset is wrong, not merely richer than this build. */
      if (!_botsContext->IsSensorTypeRegistered(type)) {
        MOCHI_LOG_WARNING(
            "BotImpl::CreateSensors: sensor '%s' on link '%s' has unknown type '%s'; skipping it "
            "and loading the rest of the bot",
            std::string(sensor.name).c_str(),
            std::string(linkPrefab.name).c_str(),
            std::string(sensor.type).c_str());
        continue;
      }
      SensorHandle const handle = _botsContext->CreateSensor(
          type,
          linkActor,
          std::string_view(sensor.name.c_str(), sensor.name.size()),
          std::string_view(sensor.params.c_str(), sensor.params.size()),
          error);
      MOCHI_ERROR_RETURN(error);
      // Place the sensor relative to its link actor.
      if (auto* created = _botsContext->GetSensor(handle); created != nullptr) {
        created->SetParentFromSensor(sensor.parentFromSensor);
      }
    }
  }
}

int BotImpl::FindLinkIndex(std::string_view linkName) const {
  for (int i = 0; i < isize(_botPrefab.links); ++i) {
    if (std::string_view(_botPrefab.links[i].name.c_str(), _botPrefab.links[i].name.size()) ==
        linkName) {
      return i;
    }
  }
  return -1;
}

SensorBase* BotImpl::CreateSensor(
    std::string_view typeName,
    std::string_view linkName,
    std::string_view name,
    std::string_view paramArgs,
    Error& error) {
  MOCHI_ERROR_RETURN(error, nullptr);
  MOCHI_ERROR_IF(_botsContext == nullptr, error, "Bots context is null");
  Scene* const scene = ResolveScene();
  MOCHI_ERROR_IF(scene == nullptr, error, "Bot has no scene");
  MOCHI_ERROR_RETURN(error, nullptr);

  int const linkIndex = FindLinkIndex(linkName);
  MOCHI_ERROR_IF(linkIndex < 0, error, "Bot has no link with the given name.");
  MOCHI_ERROR_IF(linkIndex >= isize(_linkActors), error, "Bot link actors are not cached.");
  MOCHI_ERROR_RETURN(error, nullptr);

  Actor* const linkActor = scene->GetActor(_linkActors[linkIndex]);
  MOCHI_ERROR_IF(linkActor == nullptr, error, "Failed to resolve link actor for sensor.");
  MOCHI_ERROR_RETURN(error, nullptr);

  SensorHandle const handle =
      _botsContext->CreateSensor(typeName, linkActor, name, paramArgs, error);
  MOCHI_ERROR_RETURN(error, nullptr);
  return _botsContext->GetSensor(handle);
}

DynamicArray<ControllerHandle> BotImpl::GetControllerHandles() const {
  if (_botsContext == nullptr) {
    return {};
  }
  return _botsContext->FindControllerHandlesByOwner(this);
}

ControllerBase* BotImpl::GetController(ControllerHandle controllerHandle, Error& error) const {
  MOCHI_ERROR_RETURN(error, nullptr);
  MOCHI_ERROR_IF(_botsContext == nullptr, error, "Bots context is null");
  MOCHI_ERROR_RETURN(error, nullptr);

  ControllerBase* const controller = _botsContext->GetController(controllerHandle);
  if (controller == nullptr) {
    MOCHI_ERROR_SET(error, "No live controller is associated with this handle");
    return nullptr;
  }
  if (controller->GetOwningBot() != this) {
    MOCHI_ERROR_SET(error, "Controller handle does not belong to this bot");
    return nullptr;
  }
  return controller;
}

DynamicArray<SensorHandle> BotImpl::GetSensorHandles() const {
  if (_botsContext == nullptr) {
    return {};
  }
  return _botsContext->FindSensorHandlesByOwner(this);
}

SensorBase* BotImpl::GetSensor(SensorHandle sensorHandle, Error& error) const {
  MOCHI_ERROR_RETURN(error, nullptr);
  MOCHI_ERROR_IF(_botsContext == nullptr, error, "Bots context is null");
  MOCHI_ERROR_RETURN(error, nullptr);

  SensorBase* const sensor = _botsContext->GetSensor(sensorHandle);
  if (sensor == nullptr) {
    MOCHI_ERROR_SET(error, "No live sensor is associated with this handle");
    return nullptr;
  }
  if (sensor->GetOwningBot() != this) {
    MOCHI_ERROR_SET(error, "Sensor handle does not belong to this bot");
    return nullptr;
  }
  return sensor;
}

char const* BotImpl::GetSensorLinkName(SensorHandle sensorHandle, Error& error) const {
  MOCHI_ERROR_RETURN(error, nullptr);
  SensorBase* const sensor = GetSensor(sensorHandle, error);
  MOCHI_ERROR_RETURN(error, nullptr);

  /* The link is derived from the actor the component sits on rather than tracked alongside it,
   * so it stays correct however the component was created. */
  int const linkIndex = FindLinkActorIndex(sensor->GetActor());
  MOCHI_ERROR_IF(
      linkIndex == kIndexNone,
      error,
      "Sensor belongs to this bot but is not attached to one of its links");
  MOCHI_ERROR_RETURN(error, nullptr);
  return _botPrefab.links[linkIndex].name.c_str();
}

void BotImpl::CreateActuators(Error& error) {
  MOCHI_ERROR_RETURN(error);
  if (_botPrefab.links.empty()) {
    return;
  }
  MOCHI_ERROR_IF(_botsContext == nullptr, error, "Bots context is null");
  Scene* const scene = ResolveScene();
  MOCHI_ERROR_IF(scene == nullptr, error, "Bot has no scene");
  MOCHI_ERROR_IF(
      isize(_linkActors) != isize(_botPrefab.links),
      error,
      "Bot link actors are not cached; CacheLinkActors must run first.");
  MOCHI_ERROR_RETURN(error);

  for (int i = 0; i < isize(_botPrefab.links); ++i) {
    auto const& linkPrefab = _botPrefab.links[i];
    if (linkPrefab.actuators.empty()) {
      continue;
    }
    Actor* linkActor = scene->GetActor(_linkActors[i]);
    MOCHI_ERROR_IF(linkActor == nullptr, error, "Failed to resolve link actor for actuator.");
    MOCHI_ERROR_RETURN(error);
    for (auto const& actuator : linkPrefab.actuators) {
      std::string_view const type(actuator.type.c_str(), actuator.type.size());
      /* A type this build does not have is skipped with a warning rather than failing the whole
       * bot, matching how CreateSensors treats unknown sensor types. A prefab is routinely authored
       * against a build with more actuator types than the one loading it -- e.g. the open-source
       * build ships bot assets whose actuators are registered only by an example script.
       * Construction failures of a type that IS registered still fail the bot: those mean the asset
       * is wrong, not merely richer than this build. */
      if (!_botsContext->IsActuatorTypeRegistered(type)) {
        MOCHI_LOG_WARNING(
            "BotImpl::CreateActuators: actuator '%s' on link '%s' has unknown type '%s'; skipping "
            "it and loading the rest of the bot",
            std::string(actuator.name).c_str(),
            std::string(linkPrefab.name).c_str(),
            std::string(actuator.type).c_str());
        continue;
      }
      /* The handle is not needed here: the actuator is attributed to this bot by the context, and
       * GetActuatorHandles reads it back from there. */
      _botsContext->CreateActuator(
          type,
          linkActor,
          std::string_view(actuator.name.c_str(), actuator.name.size()),
          std::string_view(actuator.params.c_str(), actuator.params.size()),
          error);
      MOCHI_ERROR_RETURN(error);
    }
  }
}

ActuatorBase* BotImpl::CreateActuator(
    std::string_view typeName,
    std::string_view linkName,
    std::string_view name,
    std::string_view paramArgs,
    Error& error) {
  MOCHI_ERROR_RETURN(error, nullptr);
  MOCHI_ERROR_IF(_botsContext == nullptr, error, "Bots context is null");
  Scene* const scene = ResolveScene();
  MOCHI_ERROR_IF(scene == nullptr, error, "Bot has no scene");
  MOCHI_ERROR_RETURN(error, nullptr);

  int const linkIndex = FindLinkIndex(linkName);
  MOCHI_ERROR_IF(linkIndex < 0, error, "Bot has no link with the given name.");
  MOCHI_ERROR_IF(linkIndex >= isize(_linkActors), error, "Bot link actors are not cached.");
  MOCHI_ERROR_RETURN(error, nullptr);

  Actor* const linkActor = scene->GetActor(_linkActors[linkIndex]);
  MOCHI_ERROR_IF(linkActor == nullptr, error, "Failed to resolve link actor for actuator.");
  MOCHI_ERROR_RETURN(error, nullptr);

  ActuatorHandle const handle =
      _botsContext->CreateActuator(typeName, linkActor, name, paramArgs, error);
  MOCHI_ERROR_RETURN(error, nullptr);
  return _botsContext->GetActuator(handle);
}

DynamicArray<ActuatorHandle> BotImpl::GetActuatorHandles() const {
  if (_botsContext == nullptr) {
    return {};
  }
  return _botsContext->FindActuatorHandlesByOwner(this);
}

ActuatorBase* BotImpl::GetActuator(ActuatorHandle actuatorHandle, Error& error) const {
  MOCHI_ERROR_RETURN(error, nullptr);
  MOCHI_ERROR_IF(_botsContext == nullptr, error, "Bots context is null");
  MOCHI_ERROR_RETURN(error, nullptr);

  ActuatorBase* const actuator = _botsContext->GetActuator(actuatorHandle);
  if (actuator == nullptr) {
    MOCHI_ERROR_SET(error, "No live actuator is associated with this handle");
    return nullptr;
  }
  if (actuator->GetOwningBot() != this) {
    MOCHI_ERROR_SET(error, "Actuator handle does not belong to this bot");
    return nullptr;
  }
  return actuator;
}

char const* BotImpl::GetActuatorLinkName(ActuatorHandle actuatorHandle, Error& error) const {
  MOCHI_ERROR_RETURN(error, nullptr);
  ActuatorBase* const actuator = GetActuator(actuatorHandle, error);
  MOCHI_ERROR_RETURN(error, nullptr);

  /* The link is derived from the actor the component sits on rather than tracked alongside it,
   * so it stays correct however the component was created. */
  int const linkIndex = FindLinkActorIndex(actuator->GetActor());
  MOCHI_ERROR_IF(
      linkIndex == kIndexNone,
      error,
      "Actuator belongs to this bot but is not attached to one of its links");
  MOCHI_ERROR_RETURN(error, nullptr);
  return _botPrefab.links[linkIndex].name.c_str();
}
