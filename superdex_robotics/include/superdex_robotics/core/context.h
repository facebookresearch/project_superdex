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

#include <mochi_physics/mochi_physics_experimental.h>
#include <superdex_physics.h>
#include <superdex_robotics/actuators/actuator_base.h>
#include <superdex_robotics/controllers/controller_base.h>
#include <superdex_robotics/sensors/sensor_base.h>

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>

namespace superdex::robotics {

/*----------------------------------------------------------------------------------------------
 * Component handles — opaque handles for objects managed by RoboticsContext.
 *
 * RoboticsHandle is the shared base: a monotonic uint64 value (same pattern as mochi::Handle).
 * Values are allocated from a single monotonic counter and never recycled — this eliminates
 * use-after-free at the cost of theoretical exhaustion after 2^64 allocations, and makes every
 * value globally unique across bots, controllers, sensors, and actuators.
 *
 * BotHandle / ControllerHandle / SensorHandle / ActuatorHandle are distinct C++ types deriving
 * RoboticsHandle (mirroring mochi's ActorHandle / SceneHandle / ShapeHandle). The component kind is
 * encoded by the handle's TYPE, so passing e.g. a controller handle to GetSensor() is a
 * compile-time error rather than a runtime tag mismatch.
 *---------------------------------------------------------------------------------------------*/
/* @brief Shared base handle for bots, controllers, sensors, and actuators. The distinct derived
 * handle types below encode the component kind in the C++ type (compile-time safety); all share
 * this base's raw value and validity check. */
struct RoboticsHandle {
  using ValueType = uint64_t;
  static constexpr ValueType kInvalidValue = 0;

  RoboticsHandle() = default;
  explicit RoboticsHandle(ValueType raw) : value(raw) {}

  ValueType value = kInvalidValue;

  [[nodiscard]] bool IsValid() const {
    return value != kInvalidValue;
  }

  bool operator==(RoboticsHandle const& rhs) const {
    return value == rhs.value;
  }
  bool operator!=(RoboticsHandle const& rhs) const {
    return value != rhs.value;
  }
  bool operator<(RoboticsHandle const& rhs) const {
    return value < rhs.value;
  }
  [[nodiscard]] std::size_t GetHash() const {
    return std::hash<ValueType>{}(value);
  }
};

/* Distinct handle types (mochi-style): each is a separate C++ type so component kinds cannot be
 * confused at compile time. They carry no extra state — the kind lives in the type, the identity in
 * the inherited RoboticsHandle::value. */
/* @brief Distinct handle identifying a bot in its owning @ref RoboticsContext. */
struct BotHandle : public RoboticsHandle {
  using RoboticsHandle::RoboticsHandle;
};
/* @brief Distinct handle identifying a controller in its owning @ref RoboticsContext. */
struct ControllerHandle : public RoboticsHandle {
  using RoboticsHandle::RoboticsHandle;
};
/* @brief Distinct handle identifying a sensor in its owning @ref RoboticsContext. */
struct SensorHandle : public RoboticsHandle {
  using RoboticsHandle::RoboticsHandle;
};
/* @brief Distinct handle identifying an actuator in its owning @ref RoboticsContext. */
struct ActuatorHandle : public RoboticsHandle {
  using RoboticsHandle::RoboticsHandle;
};

/* Registration token returned by @ref RoboticsContext::RegisterOnDestroy and passed back to
 * @ref RoboticsContext::CancelOnDestroy. Unlike the handles above it does not name a
 * RoboticsContext-managed component; it identifies a teardown callback registration. Mirrors
 * mochi's CallbackHandle / QueryHandle idiom (a value handle plus a Cancel method) so any object
 * can subscribe to context teardown by registering a closure rather than inheriting an observer
 * interface. Its value is drawn from the same monotonic counter as the component handles, so it too
 * is never recycled. */
struct RoboticsContextCallbackHandle : public RoboticsHandle {
  using RoboticsHandle::RoboticsHandle;
};

} // namespace superdex::robotics

/* std::hash specializations, matching the mochi::Handle pattern (one per type, all delegating to
 * the shared RoboticsHandle::GetHash). */
template <>
struct std::hash<superdex::robotics::RoboticsHandle> {
  std::size_t operator()(superdex::robotics::RoboticsHandle const& h) const noexcept {
    return h.GetHash();
  }
};
template <>
struct std::hash<superdex::robotics::BotHandle> {
  std::size_t operator()(superdex::robotics::BotHandle const& h) const noexcept {
    return h.GetHash();
  }
};
template <>
struct std::hash<superdex::robotics::ControllerHandle> {
  std::size_t operator()(superdex::robotics::ControllerHandle const& h) const noexcept {
    return h.GetHash();
  }
};
template <>
struct std::hash<superdex::robotics::SensorHandle> {
  std::size_t operator()(superdex::robotics::SensorHandle const& h) const noexcept {
    return h.GetHash();
  }
};
template <>
struct std::hash<superdex::robotics::ActuatorHandle> {
  std::size_t operator()(superdex::robotics::ActuatorHandle const& h) const noexcept {
    return h.GetHash();
  }
};

namespace superdex::robotics {

/* Forward declarations for free functions (defined after the class). */
class Bot;
class BotImpl;
class RoboticsContext;
struct BotPrefab;
struct BotTransmissionPrefab;
struct BotLinearTransmissionPrefab;
struct BotSpatialTendonPrefab;
struct IBotLoader;

/*----------------------------------------------------------------------------------------------
 * RoboticsContext — resource owner for bots, controllers, sensors, and actuators.
 *
 * Manages bot, controller, sensor, and actuator lifetimes via RoboticsHandle. Provides a
 * string-keyed factory registry for creating components by type name. Thread-safe for slot
 * operations via _slotsMutex.
 *
 * Created via CreateRoboticsContext(), destroyed via DestroyRoboticsContext(). The consumer owns
 * the pointer — matching the mochi::CreateContext() / mochi::DestroyContext() pattern. Built-in
 * controller types are registered automatically in the constructor — callers never need to call
 * registration functions.
 *
 * Usage:
 *   auto* ctx = superdex::robotics::CreateRoboticsContext();
 *   auto handle = ctx->CreateController("BASIC_OSC_PD", nullptr, robot, "arm_osc", error);
 *   auto* osc = static_cast<ControllerBasicOscPd*>(ctx->GetController(handle));
 *   osc->Initialize("base_link", "ee_link", error);
 *   osc->SetParams(params, error);
 *   auto obsv = osc->GetCurrentObservationsFromMochi(error);
 *   ControllerBasicOscPd::Target target{rootFromTargetEE};
 *   auto efforts = osc->ComputeOutput(obsv, target, error);
 *   ctx->DestroyController(handle);
 *
 *   auto sHandle = ctx->CreateSensor("SENSOR_CAMERA", linkActor, "wrist_cam", "", error);
 *   auto* cam = static_cast<CameraSensor*>(ctx->GetSensor(sHandle));
 *   cam->SetParams(camParams, error);
 *   auto const& intrinsics = cam->GetParams();
 *   ctx->DestroySensor(sHandle);
 *
 *   // At shutdown:
 *   superdex::robotics::DestroyRoboticsContext(ctx);
 *---------------------------------------------------------------------------------------------*/
/* @brief Resource owner for bots, controllers, sensors, and actuators.
 *
 * Owns the memory for bots, controllers, sensors, and actuators created through it. */
class MOCHI_API RoboticsContext {
 public:
  /* Factory function signature: takes an optional borrowed BotPrefab model (may be null), the robot
   * Actor*, and superdex::Error&; returns a new controller. std::function (not a bare function
   * pointer) so it can capture a Python callable for Python-registered controller types. */
  using ControllerFactory =
      std::function<ControllerBase*(BotPrefab const*, Actor*, superdex::Error&)>;
  /* Sensor factory signature: takes the link Actor*, @p paramArgs (a params file path or an inline
   * JSON string; empty means "use default-constructed Params"), and superdex::Error&; returns a new
   * sensor. std::function (not a bare function pointer) so it can capture, as ControllerFactory
   * does for Python-registered types. */
  using SensorFactory =
      std::function<SensorBase*(Actor*, std::string_view paramArgs, superdex::Error&)>;
  /* Actuator factory signature: mirrors SensorFactory — takes the link Actor*, @p paramArgs (a
   * params file path or inline JSON; empty means "use default-constructed Params"), and
   * superdex::Error&; returns a new actuator. */
  using ActuatorFactory =
      std::function<ActuatorBase*(Actor*, std::string_view paramArgs, superdex::Error&)>;

  RoboticsContext(RoboticsContext const&) = delete;
  RoboticsContext& operator=(RoboticsContext const&) = delete;
  RoboticsContext(RoboticsContext&&) = delete;
  RoboticsContext& operator=(RoboticsContext&&) = delete;

  /* --- Bots --- */

  /* Create a Bot from @p botPrefab, build its underlying articulated actor in @p scene
   * (using @p loader for shape assets), and register it with this RoboticsContext. The
   * returned pointer is owned by this RoboticsContext — destroy via @ref DestroyBot. */
  Bot* CreateBot(
      Scene* scene,
      BotPrefab const& botPrefab,
      IBotLoader const& loader,
      superdex::Error& error);

  /* Destroy the bot associated with the given handle. Also destroys the bot's underlying
   * articulated actor in its owning scene and any controllers, sensors, and actuators the bot
   * created. */
  void DestroyBot(Scene* scene, BotHandle handle);

  /* @brief Get the bot associated with the given handle.
   * @param handle Bot handle to resolve.
   * @return Context-owned bot, or nullptr if the handle is invalid or the bot was destroyed. */
  Bot* GetBot(BotHandle handle) const;

  /* @brief Check whether a handle refers to a live bot.
   * @param handle Bot handle to check.
   * @return True if the handle resolves to a live bot. */
  bool IsValidBot(BotHandle handle) const;

  /* @brief Find all bots whose name (from @ref BotPrefab::name) equals @p name. Bot names are not
   * unique (multiple bots may share a prefab name), so any number of handles may be returned (empty
   * if none). There is no FindBotsByType: a bot has no registered type.
   * @param name Exact bot name to match.
   * @return Handles of every matching live bot; empty if none. */
  [[nodiscard]] DynamicArray<BotHandle> FindBotsByName(std::string_view name) const;

  /* @brief Find the bot containing @p actor. Accepts either a bot's articulation actor or any of
   * its link actors. Returns null when the actor is contained by no bot, which is a valid result:
   * a scene may hold standalone actors and articulations from plain mochi prefabs alongside its
   * bots. A null @p actor also returns null.
   * @param actor Actor to locate; may be nullptr.
   * @return Context-owned bot containing the actor, or nullptr if none. */
  [[nodiscard]] Bot* GetBotContainingActor(Actor const* actor) const;

  /* --- Controllers --- */

  /* Register a new controller type by name, making it creatable via CreateController. The factory
   * is invoked with the robot Actor*, an optional borrowed BotPrefab (may be null), and Error&,
   * and returns a heap-allocated ControllerBase* (or null on failure). Users can register their
   * own C++ controller types this way; built-in types are registered automatically at
   * construction. Logs a warning and ignores a duplicate type name. */
  void RegisterControllerType(std::string_view typeName, ControllerFactory factory);

  /* Register a controller type using its own @c TypeName() and the uniform
   * (BotPrefab const*, Actor*, Error&) constructor. Convenience wrapper over RegisterControllerType
   * for the common case: @c ControllerT must expose a static @c TypeName() and that constructor. */
  template <typename ControllerT>
  void RegisterController() {
    RegisterControllerAs<ControllerT>(ControllerT::TypeName());
  }

  /* Register a controller type under an explicit @p typeName (rather than its own @c TypeName()),
   * using the uniform (BotPrefab const*, Actor*, Error&) constructor. Mainly for backwards-compat
   * aliases that map a legacy scene string to the correct controller (see
   * RegisterLegacyControllerTypes); prefer RegisterController<T>() for the canonical name. */
  template <typename ControllerT>
  void RegisterControllerAs(std::string_view typeName) {
    RegisterControllerType(
        typeName,
        [](BotPrefab const* prefab, Actor* robot, superdex::Error& error) -> ControllerBase* {
          auto* controller = new ControllerT(prefab, robot, error);
          if (!error.IsOK()) {
            ComponentBase::Destroy(controller);
            return nullptr;
          }
          return controller;
        });
  }

  /* @brief Create a controller of the given type for @p robot. Harvests model-derived configuration
   * (e.g. per-joint effort limits) from the optional borrowed @p prefab: pass null to create on a
   * raw articulated actor with no model (the prefab must outlive the controller otherwise).
   *
   * @p robot may also be null, to drive a real robot with observations pushed in from outside
   * instead of read off a Mochi articulation. Anything that needs the articulation (e.g. reading a
   * Jacobian) then has to be supplied externally, and the controller reports an error if asked for
   * it. A controller that cannot work at all without a simulation rejects the null actor itself —
   * see @ref ControllerMochiArticulatedPose, which drives Mochi's built-in implicit PD controller.
   *
   * The owning bot is inferred from @p robot (see @ref GetBotContainingActor): if that actor is a
   * bot's articulation or one of its link actors, the controller is recorded against that bot,
   * found by the bot-scoped finders, and destroyed with it; otherwise it has no owning bot.
   * @ref Bot::CreateController is the more direct way to create one on a bot you already hold, and
   * also carries an instance name.
   * @param typeName Registered controller type name.
   * @param prefab Optional borrowed robot-model description; may be nullptr and must outlive the
   * created controller.
   * @param robot Articulated actor to control, or nullptr for externally supplied observations.
   * @param name Instance name used by the name-based finders; need not be unique.
   * @param error Error status.
   * @return Handle to the created controller, or an invalid handle on error. */
  ControllerHandle CreateController(
      std::string_view typeName,
      BotPrefab const* prefab,
      Actor* robot,
      std::string_view name,
      superdex::Error& error);

  /* @brief Find all controllers whose instance name equals @p name. When @p scope is non-null only
   * that bot's controllers are considered; otherwise every controller in this context. Instance
   * names are not unique, so any number of handles may be returned (empty if none match). */
  [[nodiscard]] DynamicArray<ControllerHandle> FindControllersByName(
      std::string_view name,
      Bot const* scope) const;

  /* @brief Find all controllers whose instance name equals @p name, across every controller in
   * this context. Instance names are not unique, so any number of handles may be returned (empty
   * if none match).
   * @param name Exact instance name to match.
   * @return Handles of every match in creation order; empty if none. */
  [[nodiscard]] DynamicArray<ControllerHandle> FindControllersByName(std::string_view name) const {
    return FindControllersByName(name, /*scope*/ nullptr);
  }

  /* @brief Find all controllers of the given registered @p typeName, optionally scoped to a single
   * bot via @p scope (null = whole context). Returns every match; empty if none. */
  [[nodiscard]] DynamicArray<ControllerHandle> FindControllersByType(
      std::string_view typeName,
      Bot const* scope) const;

  /* @brief Find all controllers of the given registered @p typeName. Returns every match; empty if
   * none.
   * @param typeName Exact registered type name to match.
   * @return Handles of every match in creation order; empty if none. */
  [[nodiscard]] DynamicArray<ControllerHandle> FindControllersByType(
      std::string_view typeName) const {
    return FindControllersByType(typeName, /*scope*/ nullptr);
  }

  /* @brief Destroy the controller associated with the given handle.
   * @param handle Controller handle to destroy; invalid or stale handles are ignored. */
  void DestroyController(ControllerHandle handle);

  /* @brief Get the controller associated with the given handle.
   * @param handle Controller handle to resolve.
   * @return Context-owned controller, or nullptr if the handle is invalid or stale. */
  ControllerBase* GetController(ControllerHandle handle) const;

  /* @brief Check whether a handle refers to a live controller.
   * @param handle Controller handle to check.
   * @return True if the handle resolves to a live controller. */
  bool IsValidController(ControllerHandle handle) const;

  /* @brief Check whether a controller type name is registered (creatable via CreateController).
   * Lets the scene loader skip unknown types gracefully (e.g. internal-only types in a public
   * build).
   * @param typeName Registered type name to check.
   * @return True if a controller factory is registered for the type name. */
  [[nodiscard]] bool IsControllerTypeRegistered(std::string_view typeName) const;

  /* --- Sensors --- */

  /* Register a sensor type by name. The factory function is called by CreateSensor. */
  void RegisterSensorType(std::string_view typeName, SensorFactory factory);

  /* Register a sensor type using its own @c TypeName() and the uniform
   * (Actor*, std::string_view paramArgs, Error&) constructor. Convenience wrapper over
   * RegisterSensorType for the common case: @c SensorT must expose a static @c TypeName() and that
   * constructor (which loads its own params from the path or inline JSON). */
  template <typename SensorT>
  void RegisterSensor() {
    RegisterSensorAs<SensorT>(SensorT::TypeName());
  }

  /* Register a sensor type under an explicit @p typeName (rather than its own @c TypeName()), using
   * the uniform (Actor*, std::string_view paramArgs, Error&) constructor. Mainly for
   * backwards-compat aliases that map a legacy scene/asset string to the correct sensor (see
   * RegisterLegacySensorTypes); prefer RegisterSensor<T>() for the canonical name. */
  template <typename SensorT>
  void RegisterSensorAs(std::string_view typeName) {
    RegisterSensorType(
        typeName,
        [](Actor* linkActor, std::string_view paramArgs, superdex::Error& error) -> SensorBase* {
          auto* sensor = new SensorT(linkActor, paramArgs, error);
          if (!error.IsOK()) {
            ComponentBase::Destroy(sensor);
            return nullptr;
          }
          return sensor;
        });
  }

  /* @brief Check whether a sensor type name is registered (creatable via CreateSensor). Mirrors
   * @ref IsControllerTypeRegistered / @ref IsActuatorTypeRegistered.
   * @param typeName Registered type name to check.
   * @return True if a sensor factory is registered for the type name. */
  [[nodiscard]] bool IsSensorTypeRegistered(std::string_view typeName) const;

  /* @brief Create a sensor of the given type on the given link actor. @p paramArgs is an optional
   * params file path or inline JSON string consumed by the factory; pass an empty view to use
   * default-constructed parameters. @p name is the instance name used by @ref FindSensorsByName
   * and need not be unique. The owning bot is inferred from @p linkActor. Returns a handle. For
   * bot-owned sensors, prefer declaring them in the @ref BotPrefab so they are auto-instantiated
   * and destroyed with the bot.
   * @param typeName Registered sensor type name.
   * @param linkActor Actor to attach the sensor to, or nullptr for an actor-less sensor.
   * @param name Instance name used by the name-based finders; need not be unique.
   * @param paramArgs Params file path or inline JSON; empty uses defaults.
   * @param error Error status.
   * @return Handle to the created sensor, or an invalid handle on error. */
  SensorHandle CreateSensor(
      std::string_view typeName,
      Actor* linkActor,
      std::string_view name,
      std::string_view paramArgs,
      superdex::Error& error);

  /* @brief Destroy the sensor associated with the given handle.
   * @param handle Sensor handle to destroy; invalid or stale handles are ignored. */
  void DestroySensor(SensorHandle handle);

  /* @brief Get the sensor associated with the given handle. Returns nullptr if invalid. Downcast to
   * the concrete sensor type as needed (e.g. static_cast<CameraSensor*>(GetSensor(handle))).
   * @param handle Sensor handle to resolve.
   * @return Context-owned sensor, or nullptr if the handle is invalid or stale. */
  SensorBase* GetSensor(SensorHandle handle) const;

  /* @brief Check whether a handle refers to a live sensor.
   * @param handle Sensor handle to check.
   * @return True if the handle resolves to a live sensor. */
  bool IsValidSensor(SensorHandle handle) const;

  /* @brief Find all sensors whose instance name equals @p name. When @p scope is non-null only that
   * bot's sensors are considered; otherwise every sensor in this context (including scene-level
   * sensors that have no owning bot). Returns every match; empty if none. */
  [[nodiscard]] DynamicArray<SensorHandle> FindSensorsByName(
      std::string_view name,
      Bot const* scope) const;

  /* @brief Find all sensors whose instance name equals @p name, across every sensor in this context
   * (including scene-level sensors that have no owning bot). Returns every match; empty if none.
   * @param name Exact instance name to match.
   * @return Handles of every match in creation order; empty if none. */
  [[nodiscard]] DynamicArray<SensorHandle> FindSensorsByName(std::string_view name) const {
    return FindSensorsByName(name, /*scope*/ nullptr);
  }

  /* @brief Find all sensors of the given registered @p typeName, optionally scoped to a single bot
   * via @p scope (null = whole context). Returns every match; empty if none. */
  [[nodiscard]] DynamicArray<SensorHandle> FindSensorsByType(
      std::string_view typeName,
      Bot const* scope) const;

  /* @brief Find all sensors of the given registered @p typeName. Returns every match; empty if
   * none.
   * @param typeName Exact registered type name to match.
   * @return Handles of every match in creation order; empty if none. */
  [[nodiscard]] DynamicArray<SensorHandle> FindSensorsByType(std::string_view typeName) const {
    return FindSensorsByType(typeName, /*scope*/ nullptr);
  }

  /* --- Actuators --- */

  /* Register an actuator type by name. The factory is invoked by CreateActuator. Logs a warning and
   * ignores a duplicate type name. */
  void RegisterActuatorType(std::string_view typeName, ActuatorFactory factory);

  /* Register an actuator type using its own @c TypeName() and the uniform
   * (Actor*, std::string_view paramArgs, Error&) constructor. */
  template <typename ActuatorT>
  void RegisterActuator() {
    RegisterActuatorAs<ActuatorT>(ActuatorT::TypeName());
  }

  /* Register an actuator type under an explicit @p typeName (rather than its own @c TypeName()),
   * using the uniform (Actor*, std::string_view paramArgs, Error&) constructor. */
  template <typename ActuatorT>
  void RegisterActuatorAs(std::string_view typeName) {
    RegisterActuatorType(
        typeName,
        [](Actor* actor, std::string_view paramArgs, superdex::Error& error) -> ActuatorBase* {
          auto* actuator = new ActuatorT(actor, paramArgs, error);
          if (!error.IsOK()) {
            ComponentBase::Destroy(actuator);
            return nullptr;
          }
          return actuator;
        });
  }

  /* @brief Create an actuator of the given type on the given link actor. Unlike a sensor, an
   * actuator always drives a body, so @p linkActor is required: passing null sets @p error and
   * returns an invalid handle. @p paramArgs is an optional params file path or inline JSON string;
   * empty uses default-constructed parameters. @p name is the instance name used by @ref
   * FindActuatorsByName and need not be unique. The owning bot is inferred from @p linkActor.
   * Returns a handle. For bot-owned actuators, prefer declaring them in the @ref BotPrefab so they
   * are auto-instantiated and destroyed with the bot.
   * @param typeName Registered actuator type name.
   * @param linkActor Actor the actuator drives; must not be nullptr.
   * @param name Instance name used by the name-based finders; need not be unique.
   * @param paramArgs Params file path or inline JSON; empty uses defaults.
   * @param error Error status.
   * @return Handle to the created actuator, or an invalid handle on error. */
  ActuatorHandle CreateActuator(
      std::string_view typeName,
      Actor* linkActor,
      std::string_view name,
      std::string_view paramArgs,
      superdex::Error& error);

  /* @brief Destroy the actuator associated with the given handle.
   * @param handle Actuator handle to destroy; invalid or stale handles are ignored. */
  void DestroyActuator(ActuatorHandle handle);

  /* @brief Get the actuator associated with the given handle. Returns nullptr if invalid. Downcast
   * to the concrete actuator type as needed.
   * @param handle Actuator handle to resolve.
   * @return Context-owned actuator, or nullptr if the handle is invalid or stale. */
  ActuatorBase* GetActuator(ActuatorHandle handle) const;

  /* @brief Check whether a handle refers to a live actuator.
   * @param handle Actuator handle to check.
   * @return True if the handle resolves to a live actuator. */
  bool IsValidActuator(ActuatorHandle handle) const;

  /* @brief Check whether an actuator type name is registered (creatable via CreateActuator).
   * @param typeName Registered type name to check.
   * @return True if an actuator factory is registered for the type name. */
  [[nodiscard]] bool IsActuatorTypeRegistered(std::string_view typeName) const;

  /* @brief Find all actuators whose instance name equals @p name. When @p scope is non-null only
   * that bot's actuators are considered; otherwise every actuator in this context. Returns every
   * match; empty if none. */
  [[nodiscard]] DynamicArray<ActuatorHandle> FindActuatorsByName(
      std::string_view name,
      Bot const* scope) const;

  /* @brief Find all actuators whose instance name equals @p name, across every actuator in this
   * context. Returns every match; empty if none.
   * @param name Exact instance name to match.
   * @return Handles of every match in creation order; empty if none. */
  [[nodiscard]] DynamicArray<ActuatorHandle> FindActuatorsByName(std::string_view name) const {
    return FindActuatorsByName(name, /*scope*/ nullptr);
  }

  /* @brief Find all actuators of the given registered @p typeName, optionally scoped to a single
   * bot via @p scope (null = whole context). Returns every match; empty if none. */
  [[nodiscard]] DynamicArray<ActuatorHandle> FindActuatorsByType(
      std::string_view typeName,
      Bot const* scope) const;

  /* @brief Find all actuators of the given registered @p typeName. Returns every match; empty if
   * none.
   * @param typeName Exact registered type name to match.
   * @return Handles of every match in creation order; empty if none. */
  [[nodiscard]] DynamicArray<ActuatorHandle> FindActuatorsByType(std::string_view typeName) const {
    return FindActuatorsByType(typeName, /*scope*/ nullptr);
  }

  /* @brief Register @p callback to run when this RoboticsContext is destroyed, returning a token to
   * pass to @ref CancelOnDestroy. A BotScene registers here so it can detach itself if the context
   * is destroyed before the scene, enforcing that a RoboticsContext outlives its BotScenes (the
   * same shape as mochi's Context/Scene ownership); any other object that must react to context
   * teardown can subscribe the same way. Callbacks fire in reverse registration order during
   * ~RoboticsContext (mirroring mochi's dependent-teardown ordering) and must not re-enter this
   * context. A null
   * @p callback registers nothing and returns an invalid handle. Not part of the Python-bound API.
   */
  [[nodiscard]] RoboticsContextCallbackHandle RegisterOnDestroy(std::function<void()> callback);

  /* @brief Remove a callback previously registered via @ref RegisterOnDestroy. Called by
   * ~BotScene::Impl when a BotScene is destroyed before its context. Cancelling an invalid or
   * already-removed handle is a no-op (idempotent). Not part of the Python-bound API. */
  void CancelOnDestroy(RoboticsContextCallbackHandle handle);

 private:
  /* BotImpl reads its own components back out of the slot maps via the ByOwner helpers above. */
  friend class BotImpl;
  friend MOCHI_API RoboticsContext* CreateRoboticsContext();
  friend MOCHI_API void DestroyRoboticsContext(RoboticsContext*);

  RoboticsContext();
  ~RoboticsContext();

  /* Register the built-in controller types (MOCHI_ARTICULATED_POSE, BASIC_JSC_PD, BASIC_OSC_PD;
   * plus OSC_V1/OSC_V2 in internal builds only). Called by constructor — callers never need to
   * call this. */
  void RegisterBuiltinTypes();

  /* @ref GetBotContainingActor for callers that already hold _slotsMutex. */
  [[nodiscard]] Bot* GetBotContainingActorLocked(Actor const* actor) const;

  /* Every live sensor / actuator owned by @p owner, in creation order. Backs
   * @ref Bot::GetControllerHandles, @ref Bot::GetSensorHandles and @ref Bot::GetActuatorHandles,
   * which are how callers ask; a bot
   * reads its components out of the same slots the finders and @ref DestroyBot scan, so there is
   * one answer to what a bot has. */
  [[nodiscard]] DynamicArray<ControllerHandle> FindControllerHandlesByOwner(Bot const* owner) const;
  [[nodiscard]] DynamicArray<SensorHandle> FindSensorHandlesByOwner(Bot const* owner) const;
  [[nodiscard]] DynamicArray<ActuatorHandle> FindActuatorHandlesByOwner(Bot const* owner) const;

  /* Shared controller-creation core: looks up the factory, invokes it with the (optional, borrowed)
   * @p prefab and @p robot, and records the resulting instance in a controller slot. */
  ControllerHandle CreateControllerInternal(
      std::string_view typeName,
      BotPrefab const* prefab,
      Actor* robot,
      std::string_view name,
      superdex::Error& error);

  /* Attach the @ref BotPrefab::linearTransmissions entries to @p actor as runtime
   * @ref mochi::experimental::LinearTransmission instances, each with an attached
   * @ref mochi::experimental::DisplacementControlActuator. No-op when the prefab
   * has no transmissions. On error, @p error is set; the caller is responsible for
   * destroying @p actor (and the cascade through EnTT will free any transmissions
   * that were attached before the failure). */
  static void
  AttachLinearTransmissions(Actor* actor, BotPrefab const& botPrefab, superdex::Error& error);

  /* Attach the @ref BotPrefab::spatialTendons entries to @p actor as runtime
   * @ref mochi::experimental::SpatialTendon instances, each with an attached
   * @ref mochi::experimental::DisplacementControlActuator. No-op when the prefab
   * has no tendons. On error, @p error is set; caller destroys @p actor. */
  static void
  AttachSpatialTendons(Actor* actor, BotPrefab const& botPrefab, superdex::Error& error);

  /* Convert a @ref BotLinearTransmissionPrefab to the runtime
   * @ref mochi::experimental::LinearTransmissionParams consumed by
   * @ref mochi::experimental::AddLinearTransmission. Drops the display name and
   * actuator-specific fields. */
  static mochi::experimental::LinearTransmissionParams ToLinearTransmissionParams(
      BotLinearTransmissionPrefab const& src);

  /* Convert a @ref BotSpatialTendonPrefab to the runtime
   * @ref mochi::experimental::SpatialTendonParams consumed by
   * @ref mochi::experimental::AddSpatialTendon. Drops the display name and
   * actuator-specific fields. */
  static mochi::experimental::SpatialTendonParams ToSpatialTendonParams(
      BotSpatialTendonPrefab const& src);

  /* Convert a @ref BotTransmissionPrefab to the runtime
   * @ref mochi::experimental::DisplacementControlActuatorParams consumed by
   * @ref mochi::experimental::AttachDisplacementControlActuator.
   * @ref targetDisplacement maps to
   * @ref mochi::experimental::DisplacementControlActuatorParams::targetDisplacement
   * (and likewise stiffness, damping, etc. via inheritance). The implementation
   * uses a @c static_cast through inheritance because @ref BotTransmissionPrefab
   * inherits from @ref mochi::experimental::DisplacementControlActuatorParams.
   * Drops the display name. */
  static mochi::experimental::DisplacementControlActuatorParams ToDisplacementControlActuatorParams(
      BotTransmissionPrefab const& src);

  /* A slot holds the live object for a handle. Liveness is encoded by presence: controller/sensor/
   * actuator slots are erased from their map on destroy, and a bot slot's unique_ptr is moved out
   * (left null) on destroy. Handle values are monotonic and never recycled, so "absent" and "null
   * bot" both resolve to invalid — no separate active flag is needed. */
  struct BotSlot {
    std::unique_ptr<BotImpl> bot;
  };

  struct ControllerSlot {
    ControllerBase* controller = nullptr;

    /* Uniform accessor so the slot maps can be scanned generically. */
    [[nodiscard]] ComponentBase const* Component() const {
      return controller;
    }
  };

  struct SensorSlot {
    SensorBase* sensor = nullptr;

    /* Uniform accessor so the slot maps can be scanned generically. */
    [[nodiscard]] ComponentBase const* Component() const {
      return sensor;
    }
  };

  struct ActuatorSlot {
    ActuatorBase* actuator = nullptr;

    /* Uniform accessor so the slot maps can be scanned generically. */
    [[nodiscard]] ComponentBase const* Component() const {
      return actuator;
    }
  };

  mutable std::mutex _slotsMutex;
  std::unordered_map<RoboticsHandle::ValueType, BotSlot> _botSlots;
  std::unordered_map<RoboticsHandle::ValueType, ControllerSlot> _controllerSlots;
  std::unordered_map<RoboticsHandle::ValueType, SensorSlot> _sensorSlots;
  std::unordered_map<RoboticsHandle::ValueType, ActuatorSlot> _actuatorSlots;
  RoboticsHandle::ValueType _nextHandle = 1;
  std::unordered_map<std::string, ControllerFactory> _controllerFactories;
  std::unordered_map<std::string, SensorFactory> _sensorFactories;
  std::unordered_map<std::string, ActuatorFactory> _actuatorFactories;
  /* Teardown callbacks registered against this context (see RegisterOnDestroy), keyed by the
   * monotonic handle value. Ordered so ~RoboticsContext can fire them in reverse registration order
   * — the dependent-teardown ordering mochi uses — before releasing the bots below. A BotScene
   * registers one so a later BotScene teardown does not touch the freed context. */
  std::map<RoboticsHandle::ValueType, std::function<void()>> _onDestroyCallbacks;
};

} // namespace superdex::robotics
