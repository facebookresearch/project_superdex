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
#include <superdex_robotics/core/context.h>

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

namespace superdex::robotics {

/**
 * @brief Human-readable metadata for a .superdex_scene file. Not consumed by the runtime.
 */
struct BotSceneMetadata {
  /// Human-readable scene name, e.g. "T3 Shapefilter Station".
  DynamicString name;

  /// Longer description of the scene purpose.
  DynamicString description;

  /// Schema version, e.g. "1.0".
  DynamicString version;

  /// Author name or unixname.
  DynamicString author;

  MOCHI_STRUCT_BEGIN(superdex::robotics::BotSceneMetadata)
  MOCHI_FIELD(name)
  MOCHI_FIELD(description)
  MOCHI_FIELD(version)
  MOCHI_FIELD(author)
  MOCHI_STRUCT_END()
};

/**
 * @brief A named reference to a spawnable scene, prefab, bot, or bot archive.
 */
struct PrefabEntry {
  /// Unique name for this prefab instance (used for runtime lookup).
  DynamicString name;

  /** @brief Relative path to a .mochi_scene or .mochi_prefab file to load as a prefab, or a
   * .superdex_bot or .superdex_bot_archive file to instantiate as a bot. */
  DynamicString path;

  MOCHI_STRUCT_BEGIN(superdex::robotics::PrefabEntry)
  MOCHI_FIELD(name)
  MOCHI_FIELD(path)
  MOCHI_STRUCT_END()
};

/**
 * @brief Scene definition within a .superdex_scene.
 *
 * The base_scene is optional — a scene under construction may not have one yet. When present it is
 * the authoritative source for all simulation physics parameters (gravity, solver config, contact
 * filtering, collision layers, etc.); when absent the scene keeps its default parameters.
 */
struct BaseSceneParams {
  /// Path to a .mochi_scene file. Defines the scene foundation. Empty means no base scene.
  DynamicString baseScene;

  /// Prefabs that can be loaded/destroyed on demand at runtime (e.g. task objects spawned each
  /// episode).
  DynamicArray<PrefabEntry> spawnablePrefabs;

  /// Where this scene's task frame (the nominal table top) is in the world. Every task's root-level
  /// spawns are placed onto it, through @ref BotTaskPrefab::taskRootOffset when the task declares
  /// one; see @ref ComputeWorldFromTask. Unset leaves every task where its own coordinates put it.
  std::optional<TransformRT> taskRoot;

  MOCHI_STRUCT_BEGIN(superdex::robotics::BaseSceneParams)
  MOCHI_FIELD(baseScene)
  MOCHI_FIELD(spawnablePrefabs) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_FIELD(taskRoot) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_STRUCT_END()
};

/**
 * @brief A controller instance to create and attach to a bot.
 */
struct ControllerEntry {
  /// Controller type identifier (registered string), e.g. "BASIC_OSC_PD", "BASIC_JSC_PD".
  DynamicString type;

  /// Instance name for runtime lookup.
  DynamicString name;

  /// Controller-specific parameters as a raw JSON string. Omit to use defaults.
  DynamicString params;

  /* The controller's init args as a serialized JSON object -- the contents of the scene entry's
   * nested "initArgs" object (e.g. {"baseLinkName":..,"eeLinkName":..} for link-resolved
   * controllers, or a model path for a learned controller). Passed verbatim to
   * ControllerBase::ConfigureFromSceneEntry, which deserializes it into a controller-specific
   * struct, so the scene loader never needs to know a controller's fields. Reflected with the
   * JsonString attribute: on load the nested "initArgs" object is captured into this string, and on
   * save it is re-emitted as a nested object (round-trip identity), so no bespoke parsing is
   * needed. May also be set or modified directly (e.g. from Python) before LoadBotScene. Defaults
   * to an empty object ("{}"), which NoSerializeDefaults omits from the saved file. (Legacy scenes
   * that interleave flat controller fields instead of a nested "initArgs" object are folded into
   * this string at load time -- see LoadBotScenePrefabFromFile; new scenes and all saves use the
   * nested form.) */
  DynamicString initArgs = DynamicString("{}");

  // IgnoreExtraneousFields lets legacy scenes -- which interleave controller-specific keys (e.g.
  // baseLinkName/eeLinkName) as flat siblings -- load without tripping strict extraneous-field
  // validation; those keys are folded into initArgs at load time.
  MOCHI_STRUCT_BEGIN(superdex::robotics::ControllerEntry)
  MOCHI_ATTRIBUTE(IgnoreExtraneousFields);
  MOCHI_FIELD(type)
  MOCHI_FIELD(name)
  MOCHI_FIELD(params)
  MOCHI_FIELD(initArgs) MOCHI_ATTRIBUTE(JsonString) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_STRUCT_END()
};

/**
 * @brief A bot (robot) to instantiate in the scene.
 */
struct BotEntry {
  /// Unique instance name (used for runtime lookup via GetBot(name)).
  DynamicString name;

  /** @brief Path to a .superdex_bot or .superdex_bot_archive file (self-contained or mod bot). */
  DynamicString path;

  /// Bot placement relative to the scene root (aFromB convention: maps bot-frame coordinates into
  /// the scene frame). Default: identity.
  TransformRT parentFromBot;

  /// Initial joint angles (length = total DOFs of the bot).
  DynamicArray<real> initialPose;

  /// Controllers to create and attach to this bot.
  DynamicArray<ControllerEntry> controllers;

  /// Key naming the teleoperator rig that drives this bot; the consumer maps it to a concrete
  /// class. Empty means use the consumer's default for the slot this bot fills.
  DynamicString roboperatorClass;

  // --- LEGACY (remove once all .mochi_bot_scene assets carry `parentFromBot`) ---
  // Older assets stored placement as separate flat `translation`/`rotation` keys. These capture
  // them on load; LoadBotScenePrefabFromFile folds them into `parentFromBot` and clears them (so
  // re-saves omit them). TODO(superdex): delete these fields, the MOCHI_FIELD_NAME lines, and the
  // fold.
  Real3 _legacyTranslation = {};
  Quaternion _legacyRotation = {};

  MOCHI_STRUCT_BEGIN(superdex::robotics::BotEntry)
  MOCHI_FIELD(name)
  MOCHI_FIELD(path)
  MOCHI_FIELD(parentFromBot) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_FIELD(initialPose) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_FIELD(controllers) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_FIELD(roboperatorClass) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_FIELD_NAME(_legacyTranslation, "translation") MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_FIELD_NAME(_legacyRotation, "rotation") MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_STRUCT_END()
};

/**
 * @brief One slot a scene offers, and the bots eligible to fill it.
 */
struct SlotEntry {
  /* Unique slot name within the scene, e.g. "right". Names a role, not a bot. */
  DynamicString name;

  /* Names of @ref BotEntry entries that may fill this slot. A consumer picks at most one per
   * slot; superdex validates that each named bot exists and is listed once, and nothing else.
   * Which slots a scene has, and what they mean, is the scene's business, not superdex's. */
  DynamicArray<DynamicString> controlledBots;

  MOCHI_STRUCT_BEGIN(superdex::robotics::SlotEntry)
  MOCHI_FIELD(name)
  MOCHI_FIELD(controlledBots)
  MOCHI_STRUCT_END()
};

/**
 * @brief A scene-level sensor (not attached to any bot).
 *
 * Useful for environment-level observations such as overhead cameras, force plates,
 * or proximity sensors mounted on fixtures.
 */
struct SceneSensorEntry {
  /// Sensor type identifier (registered string), e.g. "SENSOR_CAMERA".
  DynamicString type;

  /// Unique instance name for runtime lookup.
  DynamicString name;

  /// Sensor pose relative to the scene root (aFromB convention: maps sensor-frame coordinates into
  /// the parent scene frame). Chain to world via SensorBase::GetWorldTransform. Default: identity.
  TransformRT parentFromSensor;

  /** @brief Sensor-specific parameters: a path to a .superdex_sensor JSON file, an inline JSON
   * object, or empty to use the sensor type defaults. */
  DynamicString params;

  // --- LEGACY (remove once all .mochi_bot_scene assets carry `parentFromSensor`) ---
  // Older assets stored the pose as separate flat `translation`/`rotation` keys plus a
  // now-vestigial `attachTo` string. These two fields capture the flat keys on load;
  // LoadBotScenePrefabFromFile folds them into `parentFromSensor` and clears them (so re-saves omit
  // them). `attachTo` is absorbed by IgnoreExtraneousFields and dropped. TODO(superdex): delete
  // these fields, the MOCHI_FIELD_NAME lines, IgnoreExtraneousFields, and the fold in
  // LoadBotScenePrefabFromFile.
  Real3 _legacyTranslation = {};
  Quaternion _legacyRotation = {};

  MOCHI_STRUCT_BEGIN(superdex::robotics::SceneSensorEntry)
  MOCHI_ATTRIBUTE(IgnoreExtraneousFields);
  MOCHI_FIELD(type)
  MOCHI_FIELD(name)
  MOCHI_FIELD(parentFromSensor) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_FIELD(params)
  MOCHI_FIELD_NAME(_legacyTranslation, "translation") MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_FIELD_NAME(_legacyRotation, "rotation") MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_STRUCT_END()
};

/**
 * @brief Complete parameter set for a .superdex_scene file.
 *
 * A .superdex_scene is a JSON config that serves as the single source of truth for a complete
 * simulation environment. It unifies scene geometry, robots, controllers, and sensors.
 */
struct BotScenePrefab {
  /// Human-readable metadata. Not consumed by the runtime.
  BotSceneMetadata metadata;

  /// Scene definition. The base_scene is optional.
  BaseSceneParams scene;

  /// Robots to instantiate.
  DynamicArray<BotEntry> bots;

  /// Scene-level sensors (not attached to any bot).
  DynamicArray<SceneSensorEntry> sensors;

  /* Slots this scene offers, and which bots may fill each. A map in meaning, an array of named
   * entries in the file, as every other named collection in this schema is. Empty means the
   * scene declares no slots and offers no choice: consumers instantiate its bots as they find
   * them. */
  DynamicArray<SlotEntry> slots;

  MOCHI_STRUCT_BEGIN(superdex::robotics::BotScenePrefab)
  MOCHI_FIELD(metadata) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_FIELD(scene)
  MOCHI_FIELD(bots) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_FIELD(sensors) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_FIELD(slots) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_STRUCT_END()
};

/**
 * @brief Load bot scene parameters from a .superdex_scene JSON file.
 *
 * Deserializes the file, validates that bot, controller, and spawnable-prefab names are non-empty
 * and unique, and resolves relative paths to absolute. A scene with no base scene and no bots is
 * valid — that is the state of a newly created scene.
 *
 * @param[in] path File path to the .superdex_scene file.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 * @return Loaded @ref BotScenePrefab, or default-constructed on failure.
 */
MOCHI_API BotScenePrefab LoadBotScenePrefabFromFile(std::string_view path, superdex::Error& error);

/**
 * @brief Returns an already-loaded @ref BotPrefab for a bot entry's path, or nullptr to have it
 * loaded from file. The returned prefab must stay valid until the call it was passed to returns.
 */
using BotPrefabLookup = std::function<BotPrefab const*(std::string_view botPath)>;

struct BotTaskPrefab;
struct TerminationOutcome;

/* @brief World pose of one task object, and its own extents, as termination criteria read it. */
struct TaskObjectPose {
  Real3 position = {};
  Quaternion rotation = Quaternion::Identity();
  /* @brief The object's bounding box in its own local frame, or nullopt for an object that
   * carries no bounding volume. Read only by criteria that opt into it. */
  std::optional<Aabb> localBounds = std::nullopt;
};

/**
 * @brief A render model (e.g. a .glb) drawn for one actor or articulated link, in place of its
 * collision shape.
 */
struct ActorRenderModel {
  /// Absolute path of the render model file.
  DynamicString file;
  /// Pose of the model in the actor's (or link's) frame.
  TransformRT actorFromModel;
  /// Scale of the model along its own axes, including any prefab and actor scale.
  Real3 scale = {1_r, 1_r, 1_r};
};

/**
 * @brief The render model of a bot's deformable skin, skinned over the bot's links.
 *
 * @details The model's vertices are posed by linear blend skinning: bone b carries a vertex from
 * its rest pose to rootFromBone_b(current) * inverse(rootFromBoneRest[b]). Rest poses are those of
 * the bot's links at the zero pose, the convention the skin is authored in; the model's own
 * inverse-bind matrices are not used.
 */
struct ActorSkinRenderModel {
  /// Absolute path of the skinned render model file.
  DynamicString file;
  /// Pose of the model in the bot's root frame at the rest pose.
  TransformRT rootFromModel;
  /// Scale of the model along its own axes.
  Real3 scale = {1_r, 1_r, 1_r};
  /// Per skin bone (joint of the model's skin, in order): the link actor it binds to, or an invalid
  /// handle if the bone names a link the bot does not have.
  DynamicArray<ActorHandle> boneLinks;
  /// Per skin bone: the rest pose of its link in the bot's root frame (identity for an invalid
  /// bone).
  DynamicArray<TransformRT> rootFromBoneRest;
  /// Link actors whose rigid render models the skin stands in for (see ComputeSkinHiddenLinks).
  DynamicArray<ActorHandle> hiddenLinks;
};

/**
 * @brief Runtime representation of a loaded .superdex_scene.
 *
 * Owns the physics scene and provides named lookup for bots, controllers, sensors, actuators, and
 * spawnable prefabs. This is NOT a prefab — it composes data from both mochi_physics and
 * mochi_bots.
 */
class MOCHI_API BotScene {
 public:
  ~BotScene();

  /* Non-copyable, movable. */
  BotScene(BotScene const&) = delete;
  BotScene& operator=(BotScene const&) = delete;
  BotScene(BotScene&&) noexcept;
  BotScene& operator=(BotScene&&) noexcept;

  /* @brief Get the physics scene.
   * @return Physics scene, or nullptr if it is unavailable. */
  [[nodiscard]] Scene* GetScene() const;

  /* @brief Get the Mochi context.
   * @return Mochi context, or nullptr if this BotScene is moved from. */
  [[nodiscard]] Context* GetMochiContext() const;

  /* @brief Get the robotics context (bot/controller/sensor/actuator resource owner).
   * @return Robotics context, or nullptr if unavailable. */
  [[nodiscard]] RoboticsContext* GetRoboticsContext() const;

  /* @brief Look up a bot by the name declared in the .superdex_scene file.
   * @param name Exact declared bot name.
   * @return Loaded bot, or nullptr if the name is not found or this BotScene is moved from. */
  [[nodiscard]] Bot* GetBot(std::string_view name) const;

  /* Get (loading on first use) a bot for a spawnable prefab whose path is a
   * .superdex_bot_archive / .superdex_bot, by the name declared in the
   * .superdex_scene file's spawnablePrefabs. Unlike SpawnPrefabInstance (which
   * places a named instance), this instantiates the spawnable once per name
   * as an articulated Bot (CreateBot) so callers can access its articulated
   * actor (e.g. for FK). The bot is created at identity root; callers that need
   * placement set the actor pose themselves. Cached per name and owned by this
   * BotScene. Returns nullptr and sets error if the name is unknown or its
   * prefab is not a bot archive.
   * @param name Spawnable-prefab name declared in this scene.
   * @param error Error status.
   * @return Cached or newly instantiated articulated bot, or nullptr on error. */
  Bot* GetOrLoadSpawnableBot(std::string_view name, superdex::Error& error);

  /* @brief Spawn a spawnable prefab under an instance name at a world pose. The same prefab may be
   * spawned any number of times under distinct instance names. A .superdex_bot /
   * .superdex_bot_archive spawnable is instantiated as an articulated Bot named after the instance
   * and rooted at worldFromInstance; the instance owns it.
   * @param instanceName Unique instance name; must not already be live.
   * @param prefabName Spawnable-prefab name declared in this scene.
   * @param worldFromInstance Instance root pose in world space.
   * @param error Error status.
   * @return Actor handles in prefab::AddToScene order (for a bot, its articulated actor), or empty
   *         on error. Valid until the instance is destroyed. */
  Span<ActorHandle const> SpawnPrefabInstance(
      std::string_view instanceName,
      std::string_view prefabName,
      TransformRT const& worldFromInstance,
      superdex::Error& error);

  /* @brief Destroy a prefab instance created by @ref SpawnPrefabInstance.
   * @param instanceName Name of a currently live instance.
   * @param error Error status. */
  void DestroyPrefabInstance(std::string_view instanceName, superdex::Error& error);

  /* Destroy every live prefab instance. Used for episode reset, where the caller
   * wants a clean slate without tracking which spawns succeeded. */
  void DestroyAllPrefabInstances();

  /* @brief Get the actor handles of a live prefab instance.
   * @param instanceName Instance name passed to @ref SpawnPrefabInstance.
   * @return Actor handles in prefab::AddToScene order, or empty if not live. */
  [[nodiscard]] Span<ActorHandle const> GetPrefabInstanceActors(
      std::string_view instanceName) const;

  /* @brief Get the names of every live prefab instance, in creation order. */
  [[nodiscard]] DynamicArray<DynamicString> GetPrefabInstanceNames() const;

  /* Retire a bot loaded from the .superdex_scene: destroy its controllers, actuators, sensors,
   * and articulated actor (actuators and sensors hold link-actor pointers, so they are destroyed
   * before the actor), then erase it from the bot registry so scene teardown does not
   * double-destroy it. Errors if no bot named `name` is loaded.
   * @param name Exact declared name of the loaded bot to remove.
   * @param error Error status. */
  void RemoveBot(std::string_view name, superdex::Error& error);

  /* Instantiate a bot the .superdex_scene declares but that is not loaded -- one left out of the
   * scene's botsToInitialize, or removed with @ref RemoveBot -- together with its controllers, at
   * its declared placement and initial pose. Its prefab is loaded from file. Errors, creating
   * nothing, if no bot named `name` is declared, it is already loaded, or it fails to instantiate.
   * @param name Exact declared name of the bot to initialize.
   * @param error Error status.
   * @return The loaded bot, or nullptr on error. */
  Bot* InitializeBot(std::string_view name, superdex::Error& error);

  /* @brief Get the original parameters this BotScene was loaded from.
   * @return Original scene parameters, or an empty prefab if this BotScene is moved from. */
  [[nodiscard]] BotScenePrefab const& GetBotScenePrefab() const;

  /* @brief Get a controller handle by key (borrowed reference).
   * The BotScene retains ownership — the handle remains valid as long as
   * the BotScene is alive. Access the concrete controller via
   * GetRoboticsContext()->GetController(handle) and downcast.
   * @param key Controller key in "botName/controllerName" format.
   * @return The controller handle, or an invalid handle if not found. */
  [[nodiscard]] ControllerHandle GetControllerHandle(std::string_view key) const;

  /* @brief Find controllers in the associated @ref RoboticsContext whose instance name equals
   * @p name. This includes components from any other BotScene sharing the context.
   * @param name Exact instance name to match.
   * @return Handles of every match in creation order; empty if none or the context is unavailable.
   */
  [[nodiscard]] DynamicArray<ControllerHandle> FindControllersByName(std::string_view name) const;

  /* @brief Find controllers in the associated @ref RoboticsContext whose registered type name
   * equals @p typeName. This includes components from any other BotScene sharing the context.
   * @param typeName Exact registered type name to match.
   * @return Handles of every match in creation order; empty if none or the context is unavailable.
   */
  [[nodiscard]] DynamicArray<ControllerHandle> FindControllersByType(
      std::string_view typeName) const;

  /* @brief Find sensors in the associated @ref RoboticsContext whose instance name equals
   * @p name. This includes components from any other BotScene sharing the context.
   * @param name Exact instance name to match.
   * @return Handles of every match in creation order; empty if none or the context is unavailable.
   */
  [[nodiscard]] DynamicArray<SensorHandle> FindSensorsByName(std::string_view name) const;

  /* @brief Find sensors in the associated @ref RoboticsContext whose registered type name equals
   * @p typeName. This includes components from any other BotScene sharing the context.
   * @param typeName Exact registered type name to match.
   * @return Handles of every match in creation order; empty if none or the context is unavailable.
   */
  [[nodiscard]] DynamicArray<SensorHandle> FindSensorsByType(std::string_view typeName) const;

  /* @brief Find actuators in the associated @ref RoboticsContext whose instance name equals
   * @p name. This includes components from any other BotScene sharing the context.
   * @param name Exact instance name to match.
   * @return Handles of every match in creation order; empty if none or the context is unavailable.
   */
  [[nodiscard]] DynamicArray<ActuatorHandle> FindActuatorsByName(std::string_view name) const;

  /* @brief Find actuators in the associated @ref RoboticsContext whose registered type name equals
   * @p typeName. This includes components from any other BotScene sharing the context.
   * @param typeName Exact registered type name to match.
   * @return Handles of every match in creation order; empty if none or the context is unavailable.
   */
  [[nodiscard]] DynamicArray<ActuatorHandle> FindActuatorsByType(std::string_view typeName) const;

  /* Get the base-scene actor handles created from the base ScenePrefab, in
   * prefab::AddToScene order. Does not include bot actors. Valid for the
   * lifetime of this BotScene. */
  [[nodiscard]] Span<ActorHandle const> GetBaseSceneActorHandles() const;

  /* @brief Resolve a termination criterion's task-object reference to its current world pose.
   *
   * Resolution keys off Mochi actor names, which every task spawner derives from the spawn name:
   * `<spawn>.<sub>` is the actor named exactly `<spawn>/<sub>` (a prefab companion, or a bot's
   * link); a bare `<spawn>` is the actor named `<spawn>` (a bot's articulated actor) or else the
   * spawn's first `<spawn>/...` actor. A spawn registered through @ref SpawnPrefabInstance is
   * looked up in that registry first. An articulated actor's pose is its root frame where link 0
   * has carried it, not where the actor was placed.
   * @param ref Task-object reference as written in a criterion.
   * @return World pose, or nullopt if the reference does not resolve. */
  [[nodiscard]] std::optional<TaskObjectPose> ResolveTaskObjectPose(std::string_view ref) const;

  /* @brief Begin deciding @p task's termination criteria from episode time zero, replacing any
   * already in progress.
   *
   * The criteria are evaluated after every physics step of this scene, on the stepping thread,
   * against episode time summed from the step sizes. Every consumer therefore gets the same
   * decisions at the same simulated instants, however often it reads them. Safe to call from any
   * thread.
   * @param task The task whose spawns were applied to this scene. Its criteria are copied. */
  void StartTermination(BotTaskPrefab const& task);

  /* @brief Stop deciding termination criteria. Safe to call from any thread; no evaluation is in
   * progress once it returns. */
  void StopTermination();

  /* @brief Clear progress toward the criteria — held results, latches, dwell — without restarting
   * the episode clock, for task objects that were just re-placed. Safe to call from any thread. */
  void RestartTerminationProgress();

  /* @brief The outcome of the criteria begun by @ref StartTermination.
   *
   * Sticky: the first step on which either outcome is reached is frozen until the next start or
   * stop, so a consumer reading less often than the scene steps cannot miss an outcome that held
   * only briefly. With no criteria in progress, neither outcome is authored. Safe to call from any
   * thread.
   * @return Both outcomes. */
  [[nodiscard]] TerminationOutcome GetTerminationOutcome() const;

  /* @brief Get the render model of an actor this BotScene created: a base-scene or prefab-instance
   * actor, an articulated link of one, or a bot link.
   * @param actor Actor handle; for articulated actors, the handle of a nested link actor.
   * @return The render model, or nullptr if the actor has none or was not created by this BotScene.
   *         Valid until the actor is destroyed. */
  [[nodiscard]] ActorRenderModel const* FindRenderModel(ActorHandle actor) const;

  /* @brief Get the skin render model of a bot this BotScene created.
   * @param actor Handle of the bot's articulated actor.
   * @return The skin render model, or nullptr if the bot has no skin with a render model, its skin
   *         cannot be mapped onto its links, or the actor is not a bot of this BotScene. Valid
   * until the bot is destroyed. */
  [[nodiscard]] ActorSkinRenderModel const* FindSkinRenderModel(ActorHandle actor) const;

  /* Opaque implementation; defined in the .cpp. The forward declaration is public so the internal
   * population helper in the .cpp can name the type. */
  struct Impl;

 private:
  friend MOCHI_API BotScene LoadBotScene(
      std::string_view path,
      Context* context,
      RoboticsContext* botsCtx,
      superdex::Error& error);
  friend MOCHI_API BotScene LoadBotScene(
      Scene* scene,
      BotScenePrefab const& prefab,
      std::string_view rootPath,
      Context* context,
      RoboticsContext* botsCtx,
      superdex::Error& error);
  friend MOCHI_API BotScene LoadBotScene(
      Scene* scene,
      BotScenePrefab const& prefab,
      std::string_view rootPath,
      Context* context,
      RoboticsContext* botsCtx,
      superdex::Error& error,
      BotPrefabLookup const& findBotPrefab,
      IBotLoader const* loader);
  friend MOCHI_API BotScene LoadBotScene(
      Scene* scene,
      std::string_view path,
      Context* context,
      RoboticsContext* botsCtx,
      superdex::Error& error);
  friend MOCHI_API BotScene LoadBotScene(
      Scene* scene,
      std::string_view path,
      Span<std::string_view const> botsToInitialize,
      Context* context,
      RoboticsContext* botsCtx,
      superdex::Error& error);
  BotScene();

  std::unique_ptr<Impl> impl_;
};

/**
 * @brief Load a .superdex_scene file and instantiate all scene objects.
 *
 * Creates the physics scene from the base_scene, instantiates all bots with placement and
 * initial pose, and creates controllers via the provided RoboticsContext.
 *
 * @param[in] path File path to the .superdex_scene file.
 * @param[in] context Mochi context.
 * @param[in] botsCtx RoboticsContext for controller creation and lifetime management.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 * @return The loaded BotScene.
 */
MOCHI_API BotScene LoadBotScene(
    std::string_view path,
    Context* context,
    RoboticsContext* botsCtx,
    superdex::Error& error);

/**
 * @brief Load a bot scene into a caller-provided scene from an already-loaded prefab.
 *
 * Like @ref LoadBotScene(std::string_view, Context*, RoboticsContext*, superdex::Error&) but does
 * not create or own the scene: bots and controllers are created in @p scene and the base @p prefab
 * (already loaded and path-resolved, e.g. by an asset) is added directly. The returned @ref
 * BotScene destroys the bots and controllers it created but leaves @p scene intact (the caller
 * retains ownership of the scene).
 *
 * @param[in] scene Caller-owned scene to populate.
 * @param[in] prefab Already-loaded, path-resolved bot scene parameters.
 * @param[in] rootPath Root directory used to resolve the base scene's relative shape/prefab paths.
 * @param[in] context Mochi context.
 * @param[in] botsCtx RoboticsContext for controller creation and lifetime management.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 * @return The loaded BotScene (non-owning of @p scene).
 */
MOCHI_API BotScene LoadBotScene(
    Scene* scene,
    BotScenePrefab const& prefab,
    std::string_view rootPath,
    Context* context,
    RoboticsContext* botsCtx,
    superdex::Error& error);

/**
 * @brief Like @ref LoadBotScene(Scene*, BotScenePrefab const&, std::string_view, Context*,
 * RoboticsContext*, superdex::Error&), but takes bot prefabs from @p findBotPrefab (e.g. in-memory
 * assets with unsaved edits) and loads from file only the bots it does not provide. Bot shapes are
 * loaded through @p loader, or from file when it is nullptr. C++ only.
 */
MOCHI_API BotScene LoadBotScene(
    Scene* scene,
    BotScenePrefab const& prefab,
    std::string_view rootPath,
    Context* context,
    RoboticsContext* botsCtx,
    superdex::Error& error,
    BotPrefabLookup const& findBotPrefab,
    IBotLoader const* loader);

/**
 * @brief Load a bot scene into a caller-provided scene from a file or archive path.
 *
 * Like the prefab overload but loads and validates the @ref BotScenePrefab
 * through @ref LoadBotScenePrefabFromFile so archive extraction remains transparent.
 * The base-scene-relative root is derived analogously to the path overload
 * (bots root with a resolved base-scene fallback), then delegates to the prefab
 * overload. The returned @ref BotScene owns the bots, controllers, and
 * base-scene actors it added but leaves the caller-owned @ref Scene intact.
 *
 * @param[in] scene Caller-owned scene to populate.
 * @param[in] path File path to the .superdex_scene file or .superdex_scene_archive.
 * @param[in] context Mochi context.
 * @param[in] botsCtx RoboticsContext for controller creation and lifetime management.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 * @return The loaded BotScene (non-owning of @p scene).
 */
MOCHI_API BotScene LoadBotScene(
    Scene* scene,
    std::string_view path,
    Context* context,
    RoboticsContext* botsCtx,
    superdex::Error& error);

/**
 * @brief Like @ref LoadBotScene(Scene*, std::string_view, Context*, RoboticsContext*,
 * superdex::Error&), but instantiates only the declared bots named in @p botsToInitialize. The
 * rest stay declared and can be instantiated later with @ref BotScene::InitializeBot, so a caller
 * choosing between alternative bots never simulates the ones it did not choose. Errors if
 * @p botsToInitialize names a bot the scene does not declare. C++ only.
 */
MOCHI_API BotScene LoadBotScene(
    Scene* scene,
    std::string_view path,
    Span<std::string_view const> botsToInitialize,
    Context* context,
    RoboticsContext* botsCtx,
    superdex::Error& error);

} // namespace superdex::robotics
