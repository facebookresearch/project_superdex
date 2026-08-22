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

#include <superdex_physics.h>
#include <superdex_robotics/core/context.h>

#include <mochi_physics/mochi_physics.h>
#include <mochi_physics/mochi_physics_experimental.h>
#include <mochi_physics/utils/mochi_prefab.h>

#include <variant>

namespace superdex::robotics {

// Forwards:
struct IBotLoader;

// Constants:
constexpr int kIndexNone =
    -1; ///< Sentinel value indicating no valid index (e.g., a root link with no parent).
constexpr real kEffortUnbounded = -1_r; ///< Sentinel @ref BotJointPrefab::effortLimit value: the
                                        ///< joint's actuation effort is unbounded.

/**
 * @brief Type of content stored in a .superdex_bot file.
 */
enum class BotFileType {
  BotPrefab, ///< Flat bot parameters.
  ModBotPrefab, ///< Mod bot recipe that produces @ref BotPrefab.
  Count ///< Number of enum values.
};

/**
 * @brief Reference space for computed link transforms.
 */
enum class LinkTransformSpace {
  ParentFromLink, ///< Transform from each link to its parent link.
  RootFromParent, ///< Transform from each link to the root link.
  Count ///< Number of enum values.
};

/**
 * @brief Used to explicitly enable or disable contact between links in a bot. By default, all bots
 * will spawn with contact disabled between links and their first ancestor with a physics body not
 * connected via a Hard joint. This structure can be used to either reenable contact on those
 * pairs, or disable contact between links that are not strictly child-parent. Overrides are applied
 * on bot link actors after spawn using @ref Scene::EnableActorContactSymmetric.
 *
 * @see BotPrefab::contactOverrides, Scene::EnableActorContactSymmetric
 */
struct BotContactOverride {
  DynamicString linkA; // The first link in the pair.
  DynamicString linkB; // The second link in the pair.
  bool enable = false; // Should contact between the pair be enabled?

  MOCHI_STRUCT_BEGIN(superdex::robotics::BotContactOverride)
  MOCHI_FIELD(linkA)
  MOCHI_FIELD(linkB)
  MOCHI_FIELD(enable)
  MOCHI_STRUCT_END()
};

/**
 * @brief Parameters describing a single joint in a bot.
 */
struct BotJointPrefab : mochi::prefab::ArticulatedJointPrefab {
  /* @brief Maximum effort magnitude that may be applied to actuate this joint
   * [N·m for revolute, N for prismatic]. Interpreted by regime: a negative value (default
   * @ref kEffortUnbounded) is unbounded; @c 0 is non-actuated (no effort can be applied, though the
   * joint still moves freely within its range under external forces); a positive value is a finite
   * limit. This is advisory metadata — it is not copied into the physics articulation, so the sim
   * enforces nothing on its own; controllers may read it (e.g. to clamp their output). */
  real effortLimit = kEffortUnbounded;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(BotJointPrefab const& rhs) const = default;
#endif

  MOCHI_STRUCT_BEGIN(superdex::robotics::BotJointPrefab)
  MOCHI_BASE_CLASS(mochi::prefab::ArticulatedJointPrefab)
  MOCHI_FIELD(effortLimit) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_STRUCT_END()
};

/**
 * @brief Parameters describing a single sensor in a bot.
 */
struct BotSensorPrefab {
  /* Registered sensor type (e.g., "SENSOR_CAMERA"). */
  DynamicString type;

  /* Human-readable, findable identifier for this sensor instance. */
  DynamicString name;

  /* Sensor pose relative to the associated link's actor (aFromB convention: maps sensor-frame
   * coordinates into the link frame). Chain to world via SensorBase::GetWorldTransform. */
  TransformRT parentFromSensor;

  /* Sensor parameters: a path to a .superdex_sensor JSON file or inline JSON. Empty uses
   * defaults. */
  DynamicString params;

  // --- LEGACY (remove once all .superdex_bot assets use `type`/`params`) ---
  // Older assets named these fields `typeName`/`paramsFile`. These capture the old keys on load;
  // ApplyLegacyBotSensorFields folds them into type/params and clears them (so re-saves omit them).
  // TODO(superdex): delete these fields, the MOCHI_FIELD_NAME lines, and
  // ApplyLegacyBotSensorFields.
  /* @brief Legacy key captured on load: older assets named this field @c typeName. Folded into
   * @c type by @ref ApplyLegacyBotSensorFields and cleared (so re-saves omit it). */
  DynamicString _legacyTypeName;
  /* @brief Legacy key captured on load: older assets named this field @c paramsFile. Folded into
   * @c params by @ref ApplyLegacyBotSensorFields and cleared (so re-saves omit it). */
  DynamicString _legacyParamsFile;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(BotSensorPrefab const& rhs) const = default;
#endif

  MOCHI_STRUCT_BEGIN(superdex::robotics::BotSensorPrefab)
  MOCHI_FIELD(type)
  MOCHI_FIELD(name)
  MOCHI_FIELD(parentFromSensor) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_FIELD(params) MOCHI_ATTRIBUTE(JsonString) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_FIELD_NAME(_legacyTypeName, "typeName") MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_FIELD_NAME(_legacyParamsFile, "paramsFile") MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_STRUCT_END()
};

/**
 * @brief Parameters describing a single actuator in a bot.
 */
struct BotActuatorPrefab {
  /* Registered actuator type. */
  DynamicString type;

  /* Human-readable, findable identifier for this actuator instance. */
  DynamicString name;

  /* Actuator parameters: a path to a params JSON file or inline JSON. Empty uses defaults. */
  DynamicString params;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(BotActuatorPrefab const& rhs) const = default;
#endif

  MOCHI_STRUCT_BEGIN(superdex::robotics::BotActuatorPrefab)
  MOCHI_FIELD(type)
  MOCHI_FIELD(name)
  MOCHI_FIELD(params) MOCHI_ATTRIBUTE(JsonString) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_STRUCT_END()
};

/**
 * @brief Parameters describing a single link (body) in a bot.
 */
struct BotLinkPrefab : mochi::prefab::ArticulatedLinkPrefab {
  /* Sensors attached to this link. Optional; may be empty. */
  DynamicArray<BotSensorPrefab> sensors;
  /* Actuators attached to this link. Optional; may be empty. */
  DynamicArray<BotActuatorPrefab> actuators;
  /// [Internal] Link index wrt owning bot; computed at runtime by @ref RebuildBotData; DO NOT
  /// TOUCH!
  int _index = 0;

  /// [Internal] Child link indices wrt owning bot; computed at runtime by @ref RebuildBotData; DO
  /// NOT TOUCH!
  DynamicArray<int> _childrenIndices;

  /// [Internal] Transform to parent link; computed at runtime by @ref RebuildBotData; DO NOT TOUCH!
  TransformRT _parentFromLink;

  /// [Internal] Transform to bot root; computed at runtime by @ref RebuildBotData; DO NOT TOUCH!
  TransformRT _rootFromLink;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(BotLinkPrefab const& rhs) const = default;
#endif

  MOCHI_STRUCT_BEGIN(superdex::robotics::BotLinkPrefab)
  MOCHI_BASE_CLASS(mochi::prefab::ArticulatedLinkPrefab)
  MOCHI_FIELD(sensors) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_FIELD(actuators) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_STRUCT_END()
};

/**
 * @brief Common parameters for bot transmissions (linear transmissions and spatial tendons).
 *
 * @details Holds fields shared by transmission-like actuators that couple joint DOFs
 * via a displacement-control actuator. Provides the display name and inherits actuator tuning
 * parameters from @ref mochi::experimental::DisplacementControlActuatorParams.
 */
struct BotTransmissionPrefab : mochi::experimental::DisplacementControlActuatorParams {
  /* @brief Human-readable label (used by the editor UI; not consumed by the runtime API). */
  DynamicString name;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(BotTransmissionPrefab const& rhs) const = default;
#endif

  MOCHI_STRUCT_BEGIN(superdex::robotics::BotTransmissionPrefab)
  MOCHI_BASE_CLASS(mochi::experimental::DisplacementControlActuatorParams)
  MOCHI_FIELD(name)
  MOCHI_STRUCT_END()
};

/**
 * @brief Parameters describing a single linear transmission attached to a bot.
 *
 * @details Mirrors @ref mochi::experimental::LinearTransmissionParams with an added
 * display @ref name field and SReflect markup so it can live inside @ref BotPrefab
 * and round-trip through the bot's JSON serialization. The three joint arrays
 * (@ref jointIndices, @ref jointCoefficients, @ref jointAxisDisps)
 * must always be the same length, with one entry
 * per joint the transmission traverses. The remaining fields are inherited from
 * @ref BotTransmissionPrefab which inherits
 * @ref mochi::experimental::DisplacementControlActuatorParams so that one
 * displacement-control actuator can be configured alongside each transmission.
 */
struct BotLinearTransmissionPrefab : BotTransmissionPrefab {
  /* @brief Indices into @ref BotPrefab::joints identifying which joints the
   * transmission traverses. */
  DynamicArray<int> jointIndices;

  /* @brief Transmission Jacobian entry at each joint [m / joint-DoF units for a
   * tendon, dimensionless for a gearbox]. Sign encodes direction: positive if
   * the transmission displacement increases with the joint DoF, negative if it
   * decreases. */
  DynamicArray<real> jointCoefficients;

  /* @brief Displacement along the joint axis [m] for the transmission attachment
   * point at each joint. This allows the transmission to be offset from the joint
   * origin along the axis. */
  DynamicArray<real> jointAxisDisps;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(BotLinearTransmissionPrefab const& rhs) const = default;
#endif

  MOCHI_STRUCT_BEGIN(superdex::robotics::BotLinearTransmissionPrefab)
  MOCHI_BASE_CLASS(superdex::robotics::BotTransmissionPrefab)
  MOCHI_FIELD(jointIndices)
  MOCHI_FIELD(jointCoefficients)
  MOCHI_FIELD(jointAxisDisps)
  MOCHI_STRUCT_END()
};

/**
 * @brief Parameters describing a single spatial tendon attached to a bot.
 *
 * @details Mirrors @ref mochi::experimental::SpatialTendonParams with an added
 * display @ref name field and SReflect markup so it can live inside @ref BotPrefab
 * and round-trip through the bot's JSON serialization. The ordered
 * @ref routingElements list defines waypoint and linear-joint elements traversed by the
 * tendon. The remaining fields are inherited from @ref BotTransmissionPrefab which inherits
 * @ref mochi::experimental::DisplacementControlActuatorParams so that one
 * displacement-control actuator can be configured alongside each tendon.
 */
struct BotSpatialTendonPrefab : BotTransmissionPrefab {
  /* @brief Ordered routing elements defining the tendon path. Must contain at least
   * one element, and every waypoint must be adjacent to another waypoint. */
  DynamicArray<RoutingElement> routingElements;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(BotSpatialTendonPrefab const& rhs) const = default;
#endif

  MOCHI_STRUCT_BEGIN(superdex::robotics::BotSpatialTendonPrefab)
  MOCHI_BASE_CLASS(superdex::robotics::BotTransmissionPrefab)
  MOCHI_FIELD(routingElements)
  MOCHI_STRUCT_END()
};

/**
 * @brief Complete parameter set for a bot.
 */
struct BotPrefab {
  /// Bot name.
  DynamicString name;

  /// Joint parameters for each joint in the articulation.
  DynamicArray<BotJointPrefab> joints;

  /// Link parameters for each link in the articulation.
  DynamicArray<BotLinkPrefab> links;

  /// Transform from the root link to the world frame.
  TransformRT worldFromRoot;

  /// Default pose (array of joint DOF angles/translations) used when spawning the bot.
  DynamicArray<real> defaultPose;

  /// Linear transmissions that couple multiple joint DOFs via per-joint coefficients. Each entry
  /// produces one @ref mochi::experimental::LinearTransmissionParams at instantiation time.
  DynamicArray<BotLinearTransmissionPrefab> linearTransmissions;

  /// Spatial tendons routed through waypoint and linear-joint elements. Each entry
  /// produces one @ref mochi::experimental::SpatialTendonParams at instantiation time.
  DynamicArray<BotSpatialTendonPrefab> spatialTendons;

  /// Array of link-link contact override pairs.
  DynamicArray<BotContactOverride> contactOverrides;

  /// Cycle-closing joints for closed-loop mechanisms (e.g. four-bar linkages, parallel grippers).
  /// Each entry becomes one @ref mochi::ArticulatedActorParams::cycles entry at instantiation time.
  /// Link references (@ref mochi::ArticulatedCycleJointParams::parentLink /
  /// @ref mochi::ArticulatedCycleJointParams::childLink) are int indices into @ref links.
  DynamicArray<ArticulatedCycleJointParams> cycles;

  // [Internal] The number of DOFs; computed at runtime by RebuildBotData; DO NOT TOUCH!
  int _numDofs = 0;

  // [Internal] Joint indices that are DOFS; computed at runtime by RebuildBotData; DO NOT TOUCH!
  DynamicArray<int> _dofIndices;

  MOCHI_STRUCT_BEGIN(superdex::robotics::BotPrefab)
  MOCHI_FIELD(name)
  MOCHI_FIELD(joints) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_FIELD(links) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_FIELD(worldFromRoot) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_FIELD(defaultPose) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_FIELD(linearTransmissions) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_FIELD(spatialTendons) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_FIELD(contactOverrides) MOCHI_ATTRIBUTE(NoSerializeDefaults(/*recursive*/ false));
  MOCHI_FIELD(cycles) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_STRUCT_END()
};

/**
 * @brief Base class for bot modifications.
 */
struct BotModBase {
  /// Name or description of the bot modification.
  DynamicString name;
  /// Whether to enable the modification or not.
  bool enabled = true;

  MOCHI_STRUCT_BEGIN(superdex::robotics::BotModBase)
  MOCHI_FIELD(name)
  MOCHI_FIELD(enabled)
  MOCHI_STRUCT_END()
};

/**
 * @brief Modification that attaches an external bot file to a parent link.
 */
struct AttachBot : BotModBase {
  /// Name of the parent link to attach to.
  DynamicString parentLinkName;

  /// Joint connecting the parent link to the attached bot's root.
  BotJointPrefab joint;

  /// Optional name prefix applied to the attached bot's links and joints.
  DynamicString prefix;

  /// File path to the .superdex_bot file to attach.
  DynamicString path;

  MOCHI_STRUCT_BEGIN(superdex::robotics::AttachBot)
  MOCHI_BASE_CLASS(superdex::robotics::BotModBase)
  MOCHI_FIELD(parentLinkName)
  MOCHI_FIELD(joint) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_FIELD(prefix) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_FIELD(path)
  MOCHI_STRUCT_END()
};

/**
 * @brief Modification that attaches a single link to a parent link.
 */
struct AttachLink : BotModBase {
  /// Name of the parent link to attach to.
  DynamicString parentLinkName;

  /// Joint connecting the parent link to the new link.
  BotJointPrefab joint;

  /// Parameters for the new link to attach.
  BotLinkPrefab link;

  MOCHI_STRUCT_BEGIN(superdex::robotics::AttachLink)
  MOCHI_BASE_CLASS(superdex::robotics::BotModBase)
  MOCHI_FIELD(parentLinkName)
  MOCHI_FIELD(joint) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_FIELD(link) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_STRUCT_END()
};

/**
 * @brief Modification that replaces a link and its descendants with a new link, preserving the
 * joint.
 */
struct ReplaceLink : BotModBase {
  /// Name of the link to replace.
  DynamicString linkToReplace;

  /// New link parameters.
  BotLinkPrefab link;

  MOCHI_STRUCT_BEGIN(superdex::robotics::ReplaceLink)
  MOCHI_BASE_CLASS(superdex::robotics::BotModBase)
  MOCHI_FIELD(linkToReplace)
  MOCHI_FIELD(link) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_STRUCT_END()
};

/**
 * @brief Modification that replaces a link and its descendants with an external bot, preserving the
 * joint.
 */
struct ReplaceLinkWithBot : BotModBase {
  /// Name of the link to replace.
  DynamicString linkToReplace;

  /// Name prefix applied to the replacement bot's links and joints.
  DynamicString prefix;

  /// File path to the .superdex_bot file used as the replacement.
  DynamicString path;

  MOCHI_STRUCT_BEGIN(superdex::robotics::ReplaceLinkWithBot)
  MOCHI_BASE_CLASS(superdex::robotics::BotModBase)
  MOCHI_FIELD(linkToReplace)
  MOCHI_FIELD(prefix) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_FIELD(path)
  MOCHI_STRUCT_END()
};

/**
 * @brief A bot modification action.
 */
using BotMod = std::variant<AttachBot, AttachLink, ReplaceLink, ReplaceLinkWithBot>;

/**
 * @brief Recipe for building a @ref BotPrefab by modifying a base bot.
 */
struct ModBotPrefab {
  /// Name for the resulting bot.
  DynamicString name;

  /// File path to the base .superdex_bot file.
  DynamicString base;

  /// Ordered list of modifications to apply to the base bot.
  DynamicArray<BotMod> modifications;

  MOCHI_STRUCT_BEGIN(superdex::robotics::ModBotPrefab)
  MOCHI_FIELD(name)
  MOCHI_FIELD(base)
  MOCHI_FIELD(modifications) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_STRUCT_END()
};

/**
 * @brief Runtime interface to a bot instance in a @ref Scene.
 *
 * @details A @ref Bot wraps a single articulated @ref Actor with the controllers, sensors, and
 * actuators created for it. Bots are created via @ref CreateBot and destroyed via @ref DestroyBot.
 * The @ref RoboticsContext owns the bot memory allocated and deallocated by these functions,
 * respectively. The raw @ref Bot* returned by @ref CreateBot remains valid until @ref DestroyBot
 * is called or the @ref RoboticsContext is destroyed.
 *
 * @note Not thread-safe -- like the underlying @ref Scene, a @ref Bot must only be accessed
 * from the thread that owns its scene (usually in an async callback if using async scenes.)
 */
class MOCHI_API Bot {
 public:
  /** @brief Handle identifying this bot in its owning @ref RoboticsContext.
   * @return Handle identifying this bot. */
  [[nodiscard]] virtual BotHandle GetHandle() const = 0;

  /** @brief The Mochi @ref Context that owns this bot's scene.
   * @return Mochi context that owns this bot's scene. */
  [[nodiscard]] virtual Context* GetMochiContext() = 0;
  /** @copydoc GetMochiContext */
  [[nodiscard]] virtual Context const* GetMochiContext() const = 0;

  /** @brief The @ref RoboticsContext that owns this bot.
   * @return Robotics context that owns this bot. */
  [[nodiscard]] virtual RoboticsContext* GetBotContext() = 0;
  /** @copydoc GetBotContext */
  [[nodiscard]] virtual RoboticsContext const* GetBotContext() const = 0;

  /** @brief The @ref Scene in which this bot was created.
   * @return Owning scene, or nullptr if it no longer exists. */
  [[nodiscard]] virtual Scene* GetScene() = 0;
  /** @copydoc GetScene */
  [[nodiscard]] virtual Scene const* GetScene() const = 0;

  /** @brief Bot name (forwarded from @ref BotPrefab::name).
   * @return Null-terminated bot name. */
  [[nodiscard]] virtual char const* GetName() const = 0;

  /** @brief The @ref BotPrefab from which this bot was created.
   * @return Prefab used to create this bot. */
  [[nodiscard]] virtual BotPrefab const& GetBotPrefab() const = 0;

  /** @brief The underlying articulated @ref Actor that backs this bot in the scene.
   * @return Underlying articulated actor, or nullptr if it no longer exists. */
  [[nodiscard]] virtual Actor* GetArticulatedActor() const = 0;

  /**
   * @brief Create a controller of the given type and attach it to this bot.
   *
   * @param[in] typeName Registered controller type (e.g., @c "BASIC_OSC_PD", @c "BASIC_JSC_PD").
   * @param[in] name Instance name for the controller, used by @ref FindControllersByName. Names
   * need not be unique.
   * @param[in,out] error Error status. Check @ref Error::IsOK for success.
   * @return Pointer to the new controller, or nullptr on error.
   */
  [[nodiscard]] virtual ControllerBase*
  CreateController(std::string_view typeName, std::string_view name, superdex::Error& error) = 0;

  /**
   * @brief Get the handles of every controller this bot owns.
   *
   * @details Covers controllers created through @ref CreateController and those created on one of
   * this bot's actors with @ref RoboticsContext::CreateController — the owning bot is inferred from
   * the actor (see @ref RoboticsContext::GetBotContainingActor), so how a controller was created
   * makes no difference to whether it appears here. Mirrors @ref GetSensorHandles
   * "GetSensorHandles": the same set the bot-scoped finders draw from, and the same set
   * @ref DestroyBot tears down. Resolve each handle with @ref RoboticsContext::GetController.
   *
   * @return A snapshot of the bot's controller handles, in creation order.
   */
  [[nodiscard]] virtual DynamicArray<ControllerHandle> GetControllerHandles() const = 0;

  /**
   * @brief Resolve a controller handle to its underlying @ref ControllerBase.
   *
   * @details Unlike @ref RoboticsContext::GetController, which resolves any handle in the context,
   * this checks the controller belongs to this bot first — so a handle from another bot, or from a
   * standalone actor, is reported as an error rather than silently resolved.
   *
   * @param[in] controllerHandle Handle returned by @ref GetControllerHandles.
   * @param[in,out] error Error status. Set if the handle is invalid or does not belong to this bot.
   * @return Pointer to the controller, or nullptr on error.
   */
  [[nodiscard]] virtual ControllerBase* GetController(
      ControllerHandle controllerHandle,
      superdex::Error& error) const = 0;

  /**
   * @brief Create a sensor of the given type on one of this bot's links.
   *
   * @details The bot-level counterpart to declaring a @ref BotSensorPrefab in the @ref BotPrefab,
   * for sensors that are only known at runtime. The sensor joins @ref GetSensorHandles and is
   * destroyed with the bot, exactly like an auto-instantiated one.
   *
   * @param[in] typeName Registered sensor type (e.g. @c "SENSOR_CAMERA").
   * @param[in] linkName Name of the link (from @ref BotPrefab::links) to attach the sensor to.
   * @param[in] name Instance name, used by @ref FindSensorsByName. Names need not be unique.
   * @param[in] paramArgs Optional params file path or inline JSON; empty uses defaults.
   * @param[in,out] error Error status. Check @ref Error::IsOK for success.
   * @return Pointer to the new sensor, or nullptr on error.
   */
  [[nodiscard]] virtual SensorBase* CreateSensor(
      std::string_view typeName,
      std::string_view linkName,
      std::string_view name,
      std::string_view paramArgs,
      superdex::Error& error) = 0;

  /**
   * @brief Get the handles of every sensor this bot owns.
   *
   * @details Covers sensors declared per link in the @ref BotPrefab via @ref BotSensorPrefab and
   * instantiated by @ref CreateBot, those added afterwards via @ref CreateSensor, and those
   * created straight off one of this bot's actors with @ref RoboticsContext::CreateSensor — the
   * owning bot is inferred from the actor (see @ref RoboticsContext::GetBotContainingActor), so
   * how a sensor was created makes no difference to whether it appears here. This is the same set
   * the bot-scoped finders draw from, and the same set @ref DestroyBot tears down. Use
   * @ref RoboticsContext::GetSensor on each handle to obtain the underlying @ref SensorBase
   * pointer.
   *
   * @return A snapshot of the bot's sensor handles in creation order: the @ref BotPrefab::links
   * entries first (by link, then by per-link sensor index), followed by any later additions.
   */
  [[nodiscard]] virtual DynamicArray<SensorHandle> GetSensorHandles() const = 0;

  /**
   * @brief Resolve a sensor handle to its underlying @ref SensorBase.
   * @param[in] sensorHandle Handle returned by @ref GetSensorHandles.
   * @param[in,out] error Error status. Set if the handle is invalid or does not belong to this bot.
   * @return Pointer to the sensor, or nullptr on error.
   */
  [[nodiscard]] virtual SensorBase* GetSensor(SensorHandle sensorHandle, superdex::Error& error)
      const = 0;

  /**
   * @brief Get the name of the link to which the sensor is attached.
   * @param[in] sensorHandle Handle returned by @ref GetSensorHandles.
   * @param[in,out] error Error status. Set if the handle does not belong to this bot.
   * @return Link name, or nullptr on error.
   */
  [[nodiscard]] virtual char const* GetSensorLinkName(
      SensorHandle sensorHandle,
      superdex::Error& error) const = 0;

  /**
   * @brief Create an actuator of the given type on one of this bot's links. The actuator
   * counterpart to @ref CreateSensor, for actuators only known at runtime rather than declared as a
   * @ref BotActuatorPrefab. It joins @ref GetActuatorHandles and is destroyed with the bot.
   *
   * @param[in] typeName Registered actuator type.
   * @param[in] linkName Name of the link (from @ref BotPrefab::links) to attach the actuator to.
   * @param[in] name Instance name, used by @ref FindActuatorsByName. Names need not be unique.
   * @param[in] paramArgs Optional params file path or inline JSON; empty uses defaults.
   * @param[in,out] error Error status. Check @ref Error::IsOK for success.
   * @return Pointer to the new actuator, or nullptr on error.
   */
  [[nodiscard]] virtual ActuatorBase* CreateActuator(
      std::string_view typeName,
      std::string_view linkName,
      std::string_view name,
      std::string_view paramArgs,
      superdex::Error& error) = 0;

  /**
   * @brief Get the handles of every actuator this bot owns — its @ref BotPrefab link
   * @ref BotActuatorPrefab entries, any added afterwards via @ref CreateActuator, and any created
   * on one of this bot's actors through @ref RoboticsContext::CreateActuator. Destroyed with the
   * bot. Mirrors @ref GetSensorHandles exactly, ordering included.
   *
   * @return A snapshot of the bot's actuator handles in creation order.
   */
  [[nodiscard]] virtual DynamicArray<ActuatorHandle> GetActuatorHandles() const = 0;

  /**
   * @brief Resolve an actuator handle to its underlying @ref ActuatorBase.
   * @param[in] actuatorHandle Handle returned by @ref GetActuatorHandles.
   * @param[in,out] error Set if the handle is invalid or does not belong to this bot.
   * @return Pointer to the actuator, or nullptr on error.
   */
  [[nodiscard]] virtual ActuatorBase* GetActuator(
      ActuatorHandle actuatorHandle,
      superdex::Error& error) const = 0;

  /**
   * @brief Get the name of the link to which the actuator is attached.
   * @param[in] actuatorHandle Handle returned by @ref GetActuatorHandles.
   * @param[in,out] error Set if the handle does not belong to this bot.
   * @return Link name, or nullptr on error.
   */
  [[nodiscard]] virtual char const* GetActuatorLinkName(
      ActuatorHandle actuatorHandle,
      superdex::Error& error) const = 0;

  /** @brief Find controllers owned by this bot whose instance name equals @p name. Instance names
   * are not unique.
   * @param name Exact instance name to match.
   * @return Handles of every match in creation order; empty if none. */
  [[nodiscard]] virtual DynamicArray<ControllerHandle> FindControllersByName(
      std::string_view name) const = 0;

  /** @brief Find controllers owned by this bot whose registered type name equals @p typeName.
   * @param typeName Exact registered type name to match.
   * @return Handles of every match in creation order; empty if none. */
  [[nodiscard]] virtual DynamicArray<ControllerHandle> FindControllersByType(
      std::string_view typeName) const = 0;

  /** @brief Find sensors owned by this bot whose instance name equals @p name. Instance names are
   * not unique.
   * @param name Exact instance name to match.
   * @return Handles of every match in creation order; empty if none. */
  [[nodiscard]] virtual DynamicArray<SensorHandle> FindSensorsByName(
      std::string_view name) const = 0;

  /** @brief Find sensors owned by this bot whose registered type name equals @p typeName.
   * @param typeName Exact registered type name to match.
   * @return Handles of every match in creation order; empty if none. */
  [[nodiscard]] virtual DynamicArray<SensorHandle> FindSensorsByType(
      std::string_view typeName) const = 0;

  /** @brief Find actuators owned by this bot whose instance name equals @p name. Instance names are
   * not unique.
   * @param name Exact instance name to match.
   * @return Handles of every match in creation order; empty if none. */
  [[nodiscard]] virtual DynamicArray<ActuatorHandle> FindActuatorsByName(
      std::string_view name) const = 0;

  /** @brief Find actuators owned by this bot whose registered type name equals @p typeName.
   * @param typeName Exact registered type name to match.
   * @return Handles of every match in creation order; empty if none. */
  [[nodiscard]] virtual DynamicArray<ActuatorHandle> FindActuatorsByType(
      std::string_view typeName) const = 0;

 protected:
  virtual ~Bot() = default;
};

/**
 * @brief Create a new @ref RoboticsContext.
 *
 * @details The returned @ref RoboticsContext owns all bots, controllers, sensors, and actuators
 * created through it. Destroy with @ref DestroyRoboticsContext.
 *
 * @return Pointer to the new @ref RoboticsContext.
 */
MOCHI_API RoboticsContext* CreateRoboticsContext(); // NOLINT(readability-redundant-declaration)

/**
 * @brief Destroy a @ref RoboticsContext previously created by @ref CreateRoboticsContext.
 *
 * @details Destroys all bots, controllers, sensors, and actuators owned by the context. No-op if
 * @p ctx is null.
 *
 * @param[in] ctx Bots context to destroy.
 */
MOCHI_API void DestroyRoboticsContext(
    RoboticsContext* ctx); // NOLINT(readability-redundant-declaration)

/**
 * @brief Load bot parameters from a .superdex_bot file using the default file-based loader.
 * Automatically resolves @ref ModBotPrefab files by building the final @ref BotPrefab.
 *
 * If @p path is a `.superdex_bot_archive` file (see @ref ArchiveBot), the archive is
 * transparently extracted and cached into a temp directory and the embedded target bot is loaded.
 *
 * @param[in] path File path to the `.superdex_bot` or `.superdex_bot_archive` file.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 * @return Loaded or built @ref BotPrefab, or default-constructed on failure.
 */
MOCHI_API BotPrefab LoadBotPrefabFromFile(std::string_view path, superdex::Error& error);

/**
 * @brief Create a @ref Bot runtime object from a @ref BotPrefab. Builds the underlying
 * articulated actor in @p scene and seeds its default pose. The returned @ref Bot owns no
 * scene resources beyond the actor — destroy it with @ref DestroyBot. This must be in the scene
 * thread (i.e. in an async callback if using async scenes.)
 *
 * @param[in] scene Target scene in which to create the bot and its components.
 * @param[in] botPrefab Bot parameters describing the articulation.
 * @param[in] botsContext Bots context used for components created on the bot. Must
 * outlive the returned @ref Bot.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 * @return Pointer to the created @ref Bot.
 */
MOCHI_API Bot* CreateBot(
    Scene* scene,
    BotPrefab const& botPrefab,
    RoboticsContext* botsContext,
    superdex::Error& error);

/**
 * @brief Destroy a @ref Bot previously created by @ref CreateBot. Destroys the underlying
 * articulated actor in its owning scene and any controllers, sensors, and actuators the bot
 * created. This must be in the scene thread (i.e. in an async callback if using async scenes.)
 *
 * @param[in] scene Scene to remove the bot from. Must be the same scene used to create the bot.
 * @param[in] bot Pointer to the bot to be destroyed.
 */
MOCHI_API void DestroyBot(Scene* scene, Bot* bot);

} // namespace superdex::robotics

#ifdef MOCHI_USE_REFLECTION
#if MOCHI_USE_REFLECTION
#include "utils/superdex_robotics_reflection.generated.h"
#endif
#endif
