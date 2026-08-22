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

#include <superdex_robotics/controllers/controller_basic_jsc_pd.h>
#include <superdex_robotics/controllers/controller_basic_osc_pd.h>
#include <superdex_robotics/controllers/controller_mochi_articulated_pose.h>
#include <superdex_robotics/core/context.h>
#if MOCHI_INTERNAL
#include <superdex_robotics/actuators/internal/actuators_internal.h>
#include <superdex_robotics/controllers/internal/controllers_internal.h>
#include <superdex_robotics/sensors/internal/sensors_internal.h>
#endif // MOCHI_INTERNAL
#include <superdex_robotics/core/loader.h>
#include <superdex_robotics/sensors/camera_sensor.h>
#include <superdex_robotics/superdex_robotics.h>
#include <superdex_robotics/utils/bot_utils.h>

#include "bot_impl.h"

#include <algorithm>
#include <type_traits>
#include <vector>

using namespace mochi;
using namespace superdex::robotics;

namespace {

/* Scan a handle->slot map and collect the handles of every live component satisfying @p match.
 * @p THandle is the typed handle rebuilt from each map key (the raw value).
 *
 * Results come back sorted by handle value, which is creation order: values are drawn from a
 * single monotonic counter. The slot maps are unordered, so without this the order would vary
 * between runs and shift as the maps rehash. */
template <typename THandle, typename SlotMap, typename Match>
DynamicArray<THandle> CollectHandles(SlotMap const& slots, Match&& match) {
  DynamicArray<THandle> result;
  for (auto const& [value, slot] : slots) {
    ComponentBase const* const comp = slot.Component();
    if (comp != nullptr && match(*comp)) {
      result.emplace_back(value);
    }
  }
  std::sort(result.begin(), result.end(), [](THandle const a, THandle const b) {
    return a.value < b.value;
  });
  return result;
}

/* A null @p scope means "anywhere in this context"; otherwise the component must belong to it. */
bool InScope(ComponentBase const& comp, Bot const* scope) {
  return scope == nullptr || comp.GetOwningBot() == scope;
}

} // namespace

RoboticsContext* superdex::robotics::CreateRoboticsContext() {
  return new RoboticsContext();
}

void superdex::robotics::DestroyRoboticsContext(RoboticsContext* ctx) {
  delete ctx;
}

RoboticsContextCallbackHandle RoboticsContext::RegisterOnDestroy(std::function<void()> callback) {
  if (!callback) {
    return {};
  }
  std::lock_guard lock(_slotsMutex);
  RoboticsHandle::ValueType const value = _nextHandle++;
  _onDestroyCallbacks.emplace(value, std::move(callback));
  return RoboticsContextCallbackHandle(value);
}

void RoboticsContext::CancelOnDestroy(RoboticsContextCallbackHandle handle) {
  std::lock_guard lock(_slotsMutex);
  _onDestroyCallbacks.erase(handle.value);
}

RoboticsContext::RoboticsContext() {
  RegisterBuiltinTypes();
}

RoboticsContext::~RoboticsContext() {
  /* Run any on-destroy callbacks still registered against this context so their owners' later
   * teardown does not dereference this (soon-to-be-freed) context or the bots destroyed below. A
   * BotScene registers one to enforce the RoboticsContext-outlives-BotScene contract, making the
   * reverse order safe (mirrors mochi's Context->Scene ownership). Move them out under the lock in
   * reverse registration order (like mochi's dependent teardowns), then fire outside it — a
   * callback only clears its owner's back-reference and does not re-enter this context. */
  std::vector<std::function<void()>> onDestroyCallbacks;
  {
    std::lock_guard lock(_slotsMutex);
    onDestroyCallbacks.reserve(_onDestroyCallbacks.size());
    for (auto it = _onDestroyCallbacks.rbegin(); it != _onDestroyCallbacks.rend(); ++it) {
      onDestroyCallbacks.push_back(std::move(it->second));
    }
    _onDestroyCallbacks.clear();
  }
  for (auto& callback : onDestroyCallbacks) {
    callback();
  }

  /* Destroy any remaining bots first. DestroyBot also tears down each bot's underlying
   * articulated actor in its scene and destroys the bot's controllers. We collect handles
   * up-front (under the lock) because DestroyBot itself acquires _slotsMutex. */
  std::vector<std::pair<Scene*, BotHandle>> botsToDestroy;
  {
    std::lock_guard lock(_slotsMutex);
    botsToDestroy.reserve(_botSlots.size());
    for (auto const& [handleValue, slot] : _botSlots) {
      if (slot.bot) {
        botsToDestroy.emplace_back(slot.bot->GetScene(), BotHandle(handleValue));
      }
    }
  }
  for (auto const& [scene, h] : botsToDestroy) {
    DestroyBot(scene, h);
  }
  /* Now clean up any remaining controllers/sensors that may have been created on raw articulations.
   * Collect them under the lock, then destroy outside it: ~PythonController acquires the GIL, and
   * every other entry point takes _slotsMutex while already holding the GIL, so destroying under
   * the lock would invert that order (mutex -> GIL) and can deadlock. Mirrors DestroyController. */
  std::vector<ControllerBase*> controllersToDestroy;
  std::vector<SensorBase*> sensorsToDestroy;
  std::vector<ActuatorBase*> actuatorsToDestroy;
  {
    std::lock_guard lock(_slotsMutex);
    for (auto& [handleValue, slot] : _controllerSlots) {
      if (slot.controller) {
        controllersToDestroy.push_back(slot.controller);
        slot.controller = nullptr;
      }
    }
    for (auto& [handleValue, slot] : _sensorSlots) {
      if (slot.sensor) {
        sensorsToDestroy.push_back(slot.sensor);
        slot.sensor = nullptr;
      }
    }
    for (auto& [handleValue, slot] : _actuatorSlots) {
      if (slot.actuator) {
        actuatorsToDestroy.push_back(slot.actuator);
        slot.actuator = nullptr;
      }
    }
  }
  for (ControllerBase* controller : controllersToDestroy) {
    ComponentBase::Destroy(controller);
  }
  for (SensorBase* sensor : sensorsToDestroy) {
    ComponentBase::Destroy(sensor);
  }
  for (ActuatorBase* actuator : actuatorsToDestroy) {
    ComponentBase::Destroy(actuator);
  }
}

/* --- Controllers --- */

void RoboticsContext::RegisterControllerType(std::string_view typeName, ControllerFactory factory) {
  auto key = std::string(typeName);
  if (_controllerFactories.count(key)) {
    /* Duplicate registration is recoverable: keep the existing factory and warn rather than
     * aborting, so a stray double-registration cannot crash the process. */
    MOCHI_LOG_WARNING(
        "RoboticsContext::RegisterControllerType: type '%s' is already registered; ignoring duplicate",
        key.c_str());
    return;
  }
  _controllerFactories[key] = std::move(factory);
}

ControllerHandle RoboticsContext::CreateControllerInternal(
    std::string_view typeName,
    BotPrefab const* prefab,
    Actor* robot,
    std::string_view name,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  /* A null @p robot is allowed: a controller may drive a real robot, with observations pushed in
   * from outside rather than read off a Mochi articulation. Controllers that genuinely need a live
   * actor reject one themselves (see ControllerMochiArticulatedPose), so the choice stays with the
   * controller instead of being imposed on every type here. */

  auto it = _controllerFactories.find(std::string(typeName));
  MOCHI_ERROR_IF(it == _controllerFactories.end(), error, "Unknown controller type.");
  MOCHI_ERROR_RETURN(error, {});

  ControllerBase* controller = it->second(prefab, robot, error);
  MOCHI_ERROR_RETURN(error, {});

  if (controller == nullptr) {
    MOCHI_ERROR_SET(error, "Factory returned null.");
    return {};
  }

  std::lock_guard lock(_slotsMutex);
  RoboticsHandle::ValueType index = _nextHandle++;
  controller->SetName(name);
  controller->SetOwningBot(GetBotContainingActorLocked(robot));
  _controllerSlots[index] = ControllerSlot{controller};
  return ControllerHandle(index);
}

ControllerHandle RoboticsContext::CreateController(
    std::string_view typeName,
    BotPrefab const* prefab,
    Actor* robot,
    std::string_view name,
    Error& error) {
  return CreateControllerInternal(typeName, prefab, robot, name, error);
}

DynamicArray<ControllerHandle> RoboticsContext::FindControllersByName(
    std::string_view name,
    Bot const* scope) const {
  std::lock_guard lock(_slotsMutex);
  return CollectHandles<ControllerHandle>(_controllerSlots, [&](ComponentBase const& comp) {
    return InScope(comp, scope) && comp.GetName() == name;
  });
}

DynamicArray<ControllerHandle> RoboticsContext::FindControllersByType(
    std::string_view typeName,
    Bot const* scope) const {
  std::lock_guard lock(_slotsMutex);
  return CollectHandles<ControllerHandle>(_controllerSlots, [&](ComponentBase const& comp) {
    return InScope(comp, scope) && comp.GetTypeName() == typeName;
  });
}

DynamicArray<SensorHandle> RoboticsContext::FindSensorsByName(
    std::string_view name,
    Bot const* scope) const {
  std::lock_guard lock(_slotsMutex);
  return CollectHandles<SensorHandle>(_sensorSlots, [&](ComponentBase const& comp) {
    return InScope(comp, scope) && comp.GetName() == name;
  });
}

DynamicArray<SensorHandle> RoboticsContext::FindSensorsByType(
    std::string_view typeName,
    Bot const* scope) const {
  std::lock_guard lock(_slotsMutex);
  return CollectHandles<SensorHandle>(_sensorSlots, [&](ComponentBase const& comp) {
    return InScope(comp, scope) && comp.GetTypeName() == typeName;
  });
}

void RoboticsContext::DestroyController(ControllerHandle handle) {
  if (!handle.IsValid()) {
    return;
  }

  ControllerBase* toDestroy = nullptr;
  {
    std::lock_guard lock(_slotsMutex);
    auto it = _controllerSlots.find(handle.value);
    if (it == _controllerSlots.end()) {
      return;
    }
    toDestroy = it->second.controller;
    _controllerSlots.erase(it);
  }

  if (toDestroy) {
    ComponentBase::Destroy(toDestroy);
  }
}

ControllerBase* RoboticsContext::GetController(ControllerHandle handle) const {
  if (!handle.IsValid()) {
    return nullptr;
  }

  std::lock_guard lock(_slotsMutex);
  auto it = _controllerSlots.find(handle.value);
  if (it == _controllerSlots.end()) {
    return nullptr;
  }
  return it->second.controller;
}

bool RoboticsContext::IsValidController(ControllerHandle handle) const {
  if (!handle.IsValid()) {
    return false;
  }

  std::lock_guard lock(_slotsMutex);
  auto it = _controllerSlots.find(handle.value);
  if (it == _controllerSlots.end()) {
    return false;
  }
  return it->second.controller != nullptr && it->second.controller->IsValid();
}

bool RoboticsContext::IsControllerTypeRegistered(std::string_view typeName) const {
  return _controllerFactories.find(std::string(typeName)) != _controllerFactories.end();
}

Bot* RoboticsContext::CreateBot(
    Scene* scene,
    BotPrefab const& botPrefab,
    IBotLoader const& loader,
    Error& error) {
  MOCHI_ERROR_RETURN(error, nullptr);
  MOCHI_ERROR_IF(scene == nullptr, error, "Scene is null.");
  MOCHI_ERROR_RETURN(error, nullptr);

  Actor* actor = AddToScene(botPrefab, scene, loader, error);
  MOCHI_ERROR_RETURN(error, nullptr);

  /* Materialize the prefab-described transmissions on the freshly-built actor. On
   * failure, destroy the actor (which cascades through EnTT to free any
   * transmissions we managed to attach in earlier iterations) and return null,
   * matching the all-or-nothing creation semantics of AddToScene above. */
  AttachLinearTransmissions(actor, botPrefab, error);
  if (!error.IsOK()) {
    scene->DestroyActor(actor->GetHandle());
    return nullptr;
  }
  AttachSpatialTendons(actor, botPrefab, error);
  if (!error.IsOK()) {
    scene->DestroyActor(actor->GetHandle());
    return nullptr;
  }

  auto bot = std::make_unique<BotImpl>(actor, scene, this, botPrefab);

  /* Resolve the bot's link actors before creating any components on them: this is what lets the
   * creation entry points below infer this bot as their owner.
   *
   * This must happen before the bot is published into _botSlots. Owner inference asks every
   * registered bot whether it owns an actor while holding _slotsMutex, and OwnsActor reads
   * _linkActors; publishing first would expose a bot whose _linkActors this thread is still
   * filling in, so a concurrent component creation could read the container mid-write. Caching
   * here keeps the write unreachable rather than widening the lock to cover it. On failure the
   * bot is not registered yet, so tear the actor down directly -- DestroyBot cannot reach it. */
  bot->CacheLinkActors(error);
  if (!error.IsOK()) {
    scene->DestroyActor(actor->GetHandle());
    return nullptr;
  }

  BotHandle handle;
  BotImpl* raw = nullptr;
  {
    std::lock_guard lock(_slotsMutex);
    RoboticsHandle::ValueType const index = _nextHandle++;
    handle = BotHandle(index);
    bot->SetHandle(handle);
    raw = bot.get();
    BotSlot& slot = _botSlots[index];
    slot.bot = std::move(bot);
  }

  /* Auto-instantiate any sensors declared in the bot's prefab. On failure, tear down the bot
   * (which also destroys any partially-created sensors and the underlying articulated actor)
   * before returning null. */
  raw->CreateSensors(error);
  if (!error.IsOK()) {
    DestroyBot(scene, handle);
    return nullptr;
  }

  /* Auto-instantiate any actuators declared per-link in the bot's prefab, mirroring sensors. On
   * failure, tear down the partially-created bot before returning null. */
  raw->CreateActuators(error);
  if (!error.IsOK()) {
    DestroyBot(scene, handle);
    return nullptr;
  }
  return raw;
}

void RoboticsContext::DestroyBot(Scene* scene, BotHandle handle) {
  if (!handle.IsValid()) {
    return;
  }

  std::unique_ptr<BotImpl> toDestroy;
  std::vector<ControllerHandle> controllersToDestroy;
  std::vector<ActuatorHandle> actuatorsToDestroy;
  std::vector<SensorHandle> sensorsToDestroy;
  {
    std::lock_guard lock(_slotsMutex);
    auto it = _botSlots.find(handle.value);
    if (it == _botSlots.end() || it->second.bot == nullptr) {
      return;
    }
    /* Collect every component owned by this bot (recorded at creation, inferred from the actor it
     * was built on) so they can be destroyed below. Captured before the bot's BotImpl is moved
     * out — the pointer stays valid because moving a unique_ptr does not destroy the pointee. */
    Bot const* const botPtr = it->second.bot.get();
    for (auto const& [value, slot] : _controllerSlots) {
      if (slot.controller != nullptr && slot.controller->GetOwningBot() == botPtr) {
        controllersToDestroy.emplace_back(value);
      }
    }
    for (auto const& [value, slot] : _actuatorSlots) {
      if (slot.actuator != nullptr && slot.actuator->GetOwningBot() == botPtr) {
        actuatorsToDestroy.emplace_back(value);
      }
    }
    for (auto const& [value, slot] : _sensorSlots) {
      if (slot.sensor != nullptr && slot.sensor->GetOwningBot() == botPtr) {
        sensorsToDestroy.emplace_back(value);
      }
    }
    /* Move the BotImpl out, leaving a null-bot tombstone in the slot; this null is what marks the
     * handle destroyed (values are never recycled, so we can leave the empty slot in place). */
    toDestroy = std::move(it->second.bot);
  }

  /* Destroy the bot's components before its actor: actuators and sensors hold link actors, and a
   * controller's destructor may touch the articulation. */
  for (ControllerHandle const controllerHandle : controllersToDestroy) {
    DestroyController(controllerHandle);
  }
  for (ActuatorHandle const actuatorHandle : actuatorsToDestroy) {
    DestroyActuator(actuatorHandle);
  }
  for (SensorHandle const sensorHandle : sensorsToDestroy) {
    DestroySensor(sensorHandle);
  }

  if (toDestroy) {
    /* Destroy the underlying scene actor in the bot's own scene. A mismatched @p scene will warn
     * and then fall back to the bot's owning scene rather than acting on the wrong scene (or
     * aborting). */
    Scene* const owningScene = toDestroy->GetScene();
    if (scene != owningScene) {
      MOCHI_LOG_WARNING(
          "RoboticsContext::DestroyBot: passed scene does not match the bot's owning scene; using the bot's owning scene");
    }
    Actor* actor = toDestroy->GetArticulatedActor();
    if (owningScene != nullptr && actor != nullptr) {
      owningScene->DestroyActor(actor->GetHandle());
    }
  }
}

Bot* RoboticsContext::GetBotContainingActor(Actor const* actor) const {
  std::lock_guard lock(_slotsMutex);
  return GetBotContainingActorLocked(actor);
}

Bot* RoboticsContext::GetBotContainingActorLocked(Actor const* actor) const {
  if (actor == nullptr) {
    return nullptr;
  }
  ActorHandle const actorHandle = actor->GetHandle();
  for (auto const& [value, slot] : _botSlots) {
    if (slot.bot != nullptr && slot.bot->OwnsActor(actorHandle)) {
      return slot.bot.get();
    }
  }
  return nullptr;
}

DynamicArray<ControllerHandle> RoboticsContext::FindControllerHandlesByOwner(
    Bot const* owner) const {
  std::lock_guard lock(_slotsMutex);
  return CollectHandles<ControllerHandle>(
      _controllerSlots, [&](ComponentBase const& comp) { return comp.GetOwningBot() == owner; });
}

DynamicArray<SensorHandle> RoboticsContext::FindSensorHandlesByOwner(Bot const* owner) const {
  std::lock_guard lock(_slotsMutex);
  return CollectHandles<SensorHandle>(
      _sensorSlots, [&](ComponentBase const& comp) { return comp.GetOwningBot() == owner; });
}

DynamicArray<ActuatorHandle> RoboticsContext::FindActuatorHandlesByOwner(Bot const* owner) const {
  std::lock_guard lock(_slotsMutex);
  return CollectHandles<ActuatorHandle>(
      _actuatorSlots, [&](ComponentBase const& comp) { return comp.GetOwningBot() == owner; });
}

Bot* RoboticsContext::GetBot(BotHandle handle) const {
  if (!handle.IsValid()) {
    return nullptr;
  }
  std::lock_guard lock(_slotsMutex);
  auto it = _botSlots.find(handle.value);
  if (it == _botSlots.end() || it->second.bot == nullptr) {
    return nullptr;
  }
  return it->second.bot.get();
}

bool RoboticsContext::IsValidBot(BotHandle handle) const {
  if (!handle.IsValid()) {
    return false;
  }
  std::lock_guard lock(_slotsMutex);
  auto it = _botSlots.find(handle.value);
  return it != _botSlots.end() && it->second.bot != nullptr;
}

DynamicArray<BotHandle> RoboticsContext::FindBotsByName(std::string_view name) const {
  /* Bots are not ComponentBase, so CollectHandles (whose predicates read a ComponentBase) does not
   * apply — scan _botSlots directly. A destroyed bot's slot holds a null unique_ptr (values are
   * never recycled), so skip those. */
  std::lock_guard lock(_slotsMutex);
  DynamicArray<BotHandle> result;
  for (auto const& [value, slot] : _botSlots) {
    if (slot.bot == nullptr) {
      continue;
    }
    char const* const botName = slot.bot->GetName();
    if (botName != nullptr && std::string_view(botName) == name) {
      result.push_back(BotHandle(value));
    }
  }
  return result;
}

void RoboticsContext::AttachLinearTransmissions(
    Actor* actor,
    BotPrefab const& botPrefab,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(actor == nullptr, error, "Actor is null.");
  MOCHI_ERROR_RETURN(error);

  int const numJoints = static_cast<int>(botPrefab.joints.size());
  for (auto const& transmission : botPrefab.linearTransmissions) {
    /* Default-constructed transmissions (no joints yet) are silently skipped so
     * empty editor entries don't blow up at simulate time. */
    if (transmission.jointIndices.empty()) {
      continue;
    }

    /* Validate that the parallel arrays have equal length. The editor
     * defensively resizes these arrays when loading malformed data, but
     * instantiation from JSON without going through the editor may bypass
     * that normalization. */
    MOCHI_ERROR_IF_NOT(
        transmission.jointIndices.size() == transmission.jointCoefficients.size(),
        error,
        "Linear transmission joint arrays must have equal length.");
    MOCHI_ERROR_RETURN(error);

    for (int const jointIndex : transmission.jointIndices) {
      MOCHI_ERROR_IF(
          jointIndex < 0 || jointIndex >= numJoints,
          error,
          "Linear transmission joint index out of range.");
      MOCHI_ERROR_RETURN(error);
    }

    int const transmissionIndex = mochi::experimental::AddLinearTransmission(
        actor, ToLinearTransmissionParams(transmission), error);
    MOCHI_ERROR_RETURN(error);

    mochi::experimental::AttachDisplacementControlActuator(
        actor, transmissionIndex, ToDisplacementControlActuatorParams(transmission), error);
    MOCHI_ERROR_RETURN(error);
  }
}

void RoboticsContext::AttachSpatialTendons(Actor* actor, BotPrefab const& botPrefab, Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(actor == nullptr, error, "Actor is null.");
  MOCHI_ERROR_RETURN(error);

  int const numLinks = static_cast<int>(botPrefab.links.size());
  int const numJoints = static_cast<int>(botPrefab.joints.size());
  for (auto const& tendon : botPrefab.spatialTendons) {
    /* Default-constructed tendons (no routing elements yet) are silently skipped so
     * empty editor entries don't blow up at simulate time. */
    if (tendon.routingElements.empty()) {
      continue;
    }

    /* Validate routing element indices. Waypoint elements reference links,
     * LinearJoint elements reference joints. */
    for (auto const& elem : tendon.routingElements) {
      if (elem.type == RoutingElementType::Waypoint) {
        MOCHI_ERROR_IF(
            elem.index < 0 || elem.index >= numLinks,
            error,
            "Spatial tendon waypoint link index out of range.");
        MOCHI_ERROR_RETURN(error);
      } else if (elem.type == RoutingElementType::LinearJoint) {
        MOCHI_ERROR_IF(
            elem.index < 0 || elem.index >= numJoints,
            error,
            "Spatial tendon linear-joint index out of range.");
        MOCHI_ERROR_RETURN(error);
      }
    }

    int const transmissionIndex =
        mochi::experimental::AddSpatialTendon(actor, ToSpatialTendonParams(tendon), error);
    MOCHI_ERROR_RETURN(error);

    mochi::experimental::AttachDisplacementControlActuator(
        actor, transmissionIndex, ToDisplacementControlActuatorParams(tendon), error);
    MOCHI_ERROR_RETURN(error);
  }
}

mochi::experimental::LinearTransmissionParams RoboticsContext::ToLinearTransmissionParams(
    BotLinearTransmissionPrefab const& src) {
  mochi::experimental::LinearTransmissionParams dst;
  dst.jointIndices = src.jointIndices;
  dst.jointCoefficients = src.jointCoefficients;
  return dst;
}

mochi::experimental::SpatialTendonParams RoboticsContext::ToSpatialTendonParams(
    BotSpatialTendonPrefab const& src) {
  mochi::experimental::SpatialTendonParams dst;
  dst.routingElements = src.routingElements;
  return dst;
}

mochi::experimental::DisplacementControlActuatorParams
RoboticsContext::ToDisplacementControlActuatorParams(BotTransmissionPrefab const& src) {
  // Enforce the invariant that BotTransmissionPrefab publicly and unambiguously
  // inherits from DisplacementControlActuatorParams so the base-subobject slice
  // via static_cast is valid.
  static_assert(
      std::is_base_of_v<
          mochi::experimental::DisplacementControlActuatorParams,
          BotTransmissionPrefab>,
      "BotTransmissionPrefab must inherit from DisplacementControlActuatorParams");
  static_assert(
      std::is_convertible_v<
          BotTransmissionPrefab const*,
          mochi::experimental::DisplacementControlActuatorParams const*>,
      "BotTransmissionPrefab must publicly inherit from DisplacementControlActuatorParams");
  return static_cast<mochi::experimental::DisplacementControlActuatorParams const&>(src);
}

void RoboticsContext::RegisterBuiltinTypes() {
#if MOCHI_INTERNAL
  RegisterInternalControllerTypes(*this);
  RegisterInternalSensorTypes(*this);
  RegisterInternalActuatorTypes(*this);
  /* Internal-only backwards-compat aliases for legacy scene-file/asset type strings. */
  RegisterLegacyControllerTypes(*this);
  RegisterLegacySensorTypes(*this);
  RegisterLegacyActuatorTypes(*this);
#endif // MOCHI_INTERNAL
  RegisterController<ControllerMochiArticulatedPose>();
  RegisterController<ControllerBasicJscPd>();
  RegisterController<ControllerBasicOscPd>();
  RegisterSensor<CameraSensor>();
}

/* --- Sensors --- */

void RoboticsContext::RegisterSensorType(std::string_view typeName, SensorFactory factory) {
  auto key = std::string(typeName);
  if (_sensorFactories.count(key)) {
    MOCHI_LOG_WARNING(
        "RoboticsContext::RegisterSensorType: type '%s' is already registered; ignoring duplicate",
        key.c_str());
    return;
  }
  _sensorFactories[key] = std::move(factory);
}

bool RoboticsContext::IsSensorTypeRegistered(std::string_view typeName) const {
  return _sensorFactories.find(std::string(typeName)) != _sensorFactories.end();
}

SensorHandle RoboticsContext::CreateSensor(
    std::string_view typeName,
    Actor* linkActor,
    std::string_view name,
    std::string_view paramArgs,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  // A null actor is allowed: actor-less sensors (e.g. fusion sensors, or scene sensors relative to
  // the scene root) are valid. Concrete sensors that require a live actor validate it themselves.

  auto it = _sensorFactories.find(std::string(typeName));
  MOCHI_ERROR_IF(
      it == _sensorFactories.end(), error, "RoboticsContext::CreateSensor: unknown sensor type");
  MOCHI_ERROR_RETURN(error, {});

  SensorBase* sensor = it->second(linkActor, paramArgs, error);
  MOCHI_ERROR_RETURN(error, {});

  if (sensor == nullptr) {
    MOCHI_ERROR_SET(error, "RoboticsContext::CreateSensor: factory returned null");
    return {};
  }

  std::lock_guard lock(_slotsMutex);
  RoboticsHandle::ValueType index = _nextHandle++;
  sensor->SetName(name);
  sensor->SetOwningBot(GetBotContainingActorLocked(linkActor));
  _sensorSlots[index] = SensorSlot{sensor};
  return SensorHandle(index);
}

void RoboticsContext::DestroySensor(SensorHandle handle) {
  if (!handle.IsValid()) {
    return;
  }

  SensorBase* toDestroy = nullptr;
  {
    std::lock_guard lock(_slotsMutex);
    auto it = _sensorSlots.find(handle.value);
    if (it == _sensorSlots.end()) {
      return;
    }
    toDestroy = it->second.sensor;
    _sensorSlots.erase(it);
  }

  if (toDestroy) {
    ComponentBase::Destroy(toDestroy);
  }
}

SensorBase* RoboticsContext::GetSensor(SensorHandle handle) const {
  if (!handle.IsValid()) {
    return nullptr;
  }

  std::lock_guard lock(_slotsMutex);
  auto it = _sensorSlots.find(handle.value);
  if (it == _sensorSlots.end()) {
    return nullptr;
  }
  return it->second.sensor;
}

bool RoboticsContext::IsValidSensor(SensorHandle handle) const {
  if (!handle.IsValid()) {
    return false;
  }

  std::lock_guard lock(_slotsMutex);
  auto it = _sensorSlots.find(handle.value);
  if (it == _sensorSlots.end()) {
    return false;
  }
  return it->second.sensor != nullptr && it->second.sensor->IsValid();
}

/* --- Actuators --- */

void RoboticsContext::RegisterActuatorType(std::string_view typeName, ActuatorFactory factory) {
  auto key = std::string(typeName);
  if (_actuatorFactories.count(key)) {
    MOCHI_LOG_WARNING(
        "RoboticsContext::RegisterActuatorType: type '%s' is already registered; ignoring duplicate",
        key.c_str());
    return;
  }
  _actuatorFactories[key] = std::move(factory);
}

ActuatorHandle RoboticsContext::CreateActuator(
    std::string_view typeName,
    Actor* linkActor,
    std::string_view name,
    std::string_view paramArgs,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  /* Unlike a sensor, an actuator always drives a body: there is no scene-level actuator, so a null
   * link actor is a caller error rather than a valid "not attached to a bot" case. */
  MOCHI_ERROR_IF(
      linkActor == nullptr, error, "RoboticsContext::CreateActuator: link actor is null");
  MOCHI_ERROR_RETURN(error, {});

  auto it = _actuatorFactories.find(std::string(typeName));
  MOCHI_ERROR_IF(
      it == _actuatorFactories.end(),
      error,
      "RoboticsContext::CreateActuator: unknown actuator type");
  MOCHI_ERROR_RETURN(error, {});

  ActuatorBase* actuator = it->second(linkActor, paramArgs, error);
  MOCHI_ERROR_RETURN(error, {});

  if (actuator == nullptr) {
    MOCHI_ERROR_SET(error, "RoboticsContext::CreateActuator: factory returned null");
    return {};
  }

  std::lock_guard lock(_slotsMutex);
  RoboticsHandle::ValueType index = _nextHandle++;
  actuator->SetName(name);
  actuator->SetOwningBot(GetBotContainingActorLocked(linkActor));
  _actuatorSlots[index] = ActuatorSlot{actuator};
  return ActuatorHandle(index);
}

void RoboticsContext::DestroyActuator(ActuatorHandle handle) {
  if (!handle.IsValid()) {
    return;
  }

  ActuatorBase* toDestroy = nullptr;
  {
    std::lock_guard lock(_slotsMutex);
    auto it = _actuatorSlots.find(handle.value);
    if (it == _actuatorSlots.end()) {
      return;
    }
    toDestroy = it->second.actuator;
    _actuatorSlots.erase(it);
  }

  if (toDestroy) {
    ComponentBase::Destroy(toDestroy);
  }
}

ActuatorBase* RoboticsContext::GetActuator(ActuatorHandle handle) const {
  if (!handle.IsValid()) {
    return nullptr;
  }

  std::lock_guard lock(_slotsMutex);
  auto it = _actuatorSlots.find(handle.value);
  if (it == _actuatorSlots.end()) {
    return nullptr;
  }
  return it->second.actuator;
}

bool RoboticsContext::IsValidActuator(ActuatorHandle handle) const {
  if (!handle.IsValid()) {
    return false;
  }

  std::lock_guard lock(_slotsMutex);
  auto it = _actuatorSlots.find(handle.value);
  if (it == _actuatorSlots.end()) {
    return false;
  }
  return it->second.actuator != nullptr && it->second.actuator->IsValid();
}

bool RoboticsContext::IsActuatorTypeRegistered(std::string_view typeName) const {
  return _actuatorFactories.find(std::string(typeName)) != _actuatorFactories.end();
}

DynamicArray<ActuatorHandle> RoboticsContext::FindActuatorsByName(
    std::string_view name,
    Bot const* scope) const {
  std::lock_guard lock(_slotsMutex);
  return CollectHandles<ActuatorHandle>(_actuatorSlots, [&](ComponentBase const& comp) {
    return InScope(comp, scope) && comp.GetName() == name;
  });
}

DynamicArray<ActuatorHandle> RoboticsContext::FindActuatorsByType(
    std::string_view typeName,
    Bot const* scope) const {
  std::lock_guard lock(_slotsMutex);
  return CollectHandles<ActuatorHandle>(_actuatorSlots, [&](ComponentBase const& comp) {
    return InScope(comp, scope) && comp.GetTypeName() == typeName;
  });
}
