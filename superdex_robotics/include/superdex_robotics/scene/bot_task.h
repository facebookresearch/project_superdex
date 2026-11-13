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
#include <superdex_robotics/domain_randomization/domain_randomization_episode.h>
#include <superdex_robotics/scene/bot_scene.h>

#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace superdex::robotics {

/*
 * @brief Randomize an entry's pose in its parent's frame, by a DR class.
 *
 * The class names a pose class in the DR system, which owns the ranges and generates each episode's
 * offset under @ref SpawnPoseDRKey. Ranges are deliberately not in the task file: the DR system's
 * profiles, seeding and manifests are where randomization is configured and reproduced.
 */
struct PoseRandomization {
  DynamicString domainRandomizationClass;

  MOCHI_STRUCT_BEGIN(superdex::robotics::PoseRandomization)
  MOCHI_FIELD(domainRandomizationClass)
  MOCHI_STRUCT_END()
};

/*
 * @brief Spawn one of several prefabs for an entry, chosen per episode.
 *
 * The entry keeps its name, which is the object's identity in recordings and criteria; only the
 * prefab varies. The DR system draws the choice under @ref SpawnPrefabDRKey. With no draw the entry
 * uses its nominal @ref SpawnEntry::prefabName.
 */
struct PrefabSelection {
  DynamicArray<DynamicString> prefabNames;

  MOCHI_STRUCT_BEGIN(superdex::robotics::PrefabSelection)
  MOCHI_FIELD(prefabNames)
  MOCHI_STRUCT_END()
};

/*
 * @brief One way an entry's reset state is randomized.
 *
 * Discriminated on the wire by its type name, like @ref TerminationCriterion. An entry declares at
 * most one of each type.
 */
using SpawnRandomization = std::variant<PoseRandomization, PrefabSelection>;

/**
 * @brief A single named object instance to spawn on top of a .superdex_scene.
 *
 * Declares which spawnable prefab to instantiate (by name) and where. The
 * instance @ref name is the key used everywhere downstream — it becomes the
 * recorded object id (HDF5 group), the dataset/replay feature key, and the
 * Unreal actor label — so a task can spawn multiple instances of the same
 * @ref prefabName under distinct names.
 */
struct SpawnEntry {
  /// Unique instance name within the task (e.g. "Foam2", "m8_knob").
  DynamicString name;

  /// Name of a spawnable prefab declared in the .superdex_scene's
  /// scene.spawnablePrefabs[]. Multiple spawns may reference the same prefab.
  DynamicString prefabName;

  /// Name of another spawn entry whose transform this spawn is relative to,
  /// or the reserved literal "task_root" (also the implicit default when
  /// empty) to place this spawn in the task actor's own frame. @ref
  /// parentFromSpawn, and any @ref PoseRandomization offset, are interpreted in
  /// the parent's local frame; the parent's resolved (post-DR) world transform
  /// is composed in after randomization. Parent chains may be arbitrarily deep.
  /// Cycles and references to undefined spawn names are rejected by
  /// @ref ValidateBotTaskPrefab.
  DynamicString parent;

  /// How this entry's reset state is randomized each episode. Empty = placed as
  /// declared every reset.
  DynamicArray<SpawnRandomization> randomization;

  /// Local pose relative to @ref parent (aFromB convention: maps spawn-frame coordinates into the
  /// parent frame). Default: identity.
  TransformRT parentFromSpawn;

  // --- LEGACY (remove once all .mochi_bot_task assets carry `parentFromSpawn`) ---
  // Older tasks stored placement as separate flat `translation`/`rotation` keys. These capture them
  // on load; LoadBotTaskPrefabFromFile folds them into `parentFromSpawn` and clears them (so
  // re-saves omit them). TODO(superdex): delete these fields, the MOCHI_FIELD_NAME lines, and the
  // fold.
  Real3 _legacyTranslation = {};
  Quaternion _legacyRotation = {};

  // --- LEGACY (remove once all .mochi_bot_task assets carry `randomization`) ---
  // Older tasks named a pose DR class in a flat `domainRandomizationClass` key.
  // LoadBotTaskPrefabFromFile folds it into a PoseRandomization entry and clears it.
  // TODO(superdex): delete this field, its MOCHI_FIELD_NAME line, and the fold.
  DynamicString _legacyDomainRandomizationClass;

  MOCHI_STRUCT_BEGIN(superdex::robotics::SpawnEntry)
  MOCHI_FIELD(name)
  MOCHI_FIELD(prefabName)
  MOCHI_FIELD(parent) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_FIELD(randomization) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_FIELD(parentFromSpawn) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_FIELD_NAME(_legacyTranslation, "translation") MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_FIELD_NAME(_legacyRotation, "rotation") MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_FIELD_NAME(_legacyDomainRandomizationClass, "domainRandomizationClass")
  MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_STRUCT_END()
};

/* @brief How the criteria in one outcome group combine. */
enum class CriteriaCombinator {
  /* @brief Every criterion must hold. */
  All,
  /* @brief Any single criterion holding is enough. */
  Any,
};

/* @brief Distance comparison for @ref SphericalProximity. */
enum class ProximityComparator {
  LessThan,
  GreaterThan,
  LessEqual,
  GreaterEqual,
};

/* @brief Containment comparison for @ref RelativeBounds. */
enum class BoundsComparator {
  Inside,
  Outside,
};

} // namespace superdex::robotics

/* Hand-written rather than generated, for the reason given on BotaStencil: the criterion structs
 * below reflect fields of these enum types, which instantiates the trait, so the specializations
 * have to precede them. */
MOCHI_ENUM_BEGIN(superdex::robotics::CriteriaCombinator)
MOCHI_ENUM_ITEM(All)
MOCHI_ENUM_ITEM(Any)
MOCHI_ENUM_END()

MOCHI_ENUM_BEGIN(superdex::robotics::ProximityComparator)
MOCHI_ENUM_ITEM(LessThan)
MOCHI_ENUM_ITEM(GreaterThan)
MOCHI_ENUM_ITEM(LessEqual)
MOCHI_ENUM_ITEM(GreaterEqual)
MOCHI_ENUM_END()

MOCHI_ENUM_BEGIN(superdex::robotics::BoundsComparator)
MOCHI_ENUM_ITEM(Inside)
MOCHI_ENUM_ITEM(Outside)
MOCHI_ENUM_END()

namespace superdex::robotics {

/*
 * @brief Criterion: world-space center-to-center distance between two task objects, compared
 * against a threshold.
 */
struct SphericalProximity {
  DynamicString taskObjectA;
  DynamicString taskObjectB;
  ProximityComparator comparator = ProximityComparator::LessThan;
  real thresholdM = 0.0_r;
  /* @brief See @ref TerminationCriterion. */
  bool latch = false;
  /* @brief See @ref TerminationCriterion. */
  real evaluationRateS = 0.05_r;

  MOCHI_STRUCT_BEGIN(superdex::robotics::SphericalProximity)
  MOCHI_FIELD(taskObjectA)
  MOCHI_FIELD(taskObjectB)
  MOCHI_FIELD(comparator)
  MOCHI_FIELD(thresholdM) MOCHI_ATTRIBUTE(Units("m"));
  MOCHI_FIELD(latch) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_FIELD(evaluationRateS) MOCHI_ATTRIBUTE(NoSerializeDefaults) MOCHI_ATTRIBUTE(Units("s"));
  MOCHI_STRUCT_END()
};

/*
 * @brief Criterion: whether one task object's origin lies inside (or outside) an axis-aligned box
 * expressed in another task object's local frame.
 */
struct RelativeBounds {
  DynamicString targetTaskObject;
  DynamicString taskObject;
  BoundsComparator comparator = BoundsComparator::Inside;
  /* @brief Take the box from the target's own bounding box instead of the ranges below, which
   * then become per-face offsets onto it: min range offsets the box min, max range offsets the
   * box max. All zero is the box itself; widen a face with a negative min or positive max, inset
   * it with the opposite sign. Offsets that inset past the far face leave a box containing
   * nothing. Evaluates false for a target that carries no bounding volume. */
  bool useTargetBounds = false;
  Real2 xRangeM = {};
  Real2 yRangeM = {};
  Real2 zRangeM = {};
  /* @brief See @ref TerminationCriterion. */
  bool latch = false;
  /* @brief See @ref TerminationCriterion. */
  real evaluationRateS = 0.05_r;

  MOCHI_STRUCT_BEGIN(superdex::robotics::RelativeBounds)
  MOCHI_FIELD(targetTaskObject)
  MOCHI_FIELD(taskObject)
  MOCHI_FIELD(comparator)
  MOCHI_FIELD(useTargetBounds) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_FIELD(xRangeM) MOCHI_ATTRIBUTE(Units("m"));
  MOCHI_FIELD(yRangeM) MOCHI_ATTRIBUTE(Units("m"));
  MOCHI_FIELD(zRangeM) MOCHI_ATTRIBUTE(Units("m"));
  MOCHI_FIELD(latch) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_FIELD(evaluationRateS) MOCHI_ATTRIBUTE(NoSerializeDefaults) MOCHI_ATTRIBUTE(Units("s"));
  MOCHI_STRUCT_END()
};

/*
 * @brief Criterion: whether a body-local axis on one task object stays nearly parallel to a
 * body-local axis on another, compared in the world frame.
 */
struct AxisAlignment {
  DynamicString taskObjectA;
  Real3 axisALocal = {};
  DynamicString taskObjectB;
  Real3 axisBLocal = {};
  real maxErrorThreshold = 0.0_r;
  /* @brief See @ref TerminationCriterion. */
  bool latch = false;
  /* @brief See @ref TerminationCriterion. */
  real evaluationRateS = 0.05_r;

  MOCHI_STRUCT_BEGIN(superdex::robotics::AxisAlignment)
  MOCHI_FIELD(taskObjectA)
  MOCHI_FIELD(axisALocal)
  MOCHI_FIELD(taskObjectB)
  MOCHI_FIELD(axisBLocal)
  MOCHI_FIELD(maxErrorThreshold)
  MOCHI_FIELD(latch) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_FIELD(evaluationRateS) MOCHI_ATTRIBUTE(NoSerializeDefaults) MOCHI_ATTRIBUTE(Units("s"));
  MOCHI_STRUCT_END()
};

/*
 * @brief One termination criterion.
 *
 * Discriminated on the wire by the wrapping type-name key — `{"RelativeBounds": {...}}`. That is
 * simple_reflection's own variant encoding (see VariantTypeInfo::SerializeInner), which is why a
 * plain MOCHI_FIELD round-trips it; no bespoke parsing or saving is needed. Inner type names must
 * stay unique WITHOUT their namespace, which the library asserts.
 *
 * Every criterion type carries `latch`: once a latched criterion is satisfied it stays satisfied
 * until the @ref TerminationMonitor resets, so a group can require stages that are passed through
 * one after another rather than held together. Latching does not order the stages, and it fires on
 * the first satisfied evaluation — @ref CriteriaGroup::validationTimeS debounces the group, not
 * individual criteria.
 *
 * Every criterion type also carries `evaluationRateS`: seconds of sim time between evaluations,
 * defaulting to 0.05 (20 Hz); 0 evaluates on every step. Evaluations fall on the first step at or
 * after each multiple of the rate since the episode began, so criteria sharing a rate are sampled
 * on the same step and a group never combines readings taken at different moments. Between
 * evaluations a criterion holds its last result, which means a condition that holds only between
 * two evaluations is never seen — set the rate to 0 for a criterion that must not miss one.
 */
using TerminationCriterion = std::variant<SphericalProximity, RelativeBounds, AxisAlignment>;

/*
 * @brief One outcome's criteria, how they combine, and how long they must hold.
 *
 * An empty @ref criteria means the outcome is not authored — distinct from authored criteria that
 * currently evaluate false. Consumers fall back to their own predicate in the former case.
 */
struct CriteriaGroup {
  CriteriaCombinator combinator = CriteriaCombinator::All;
  DynamicArray<TerminationCriterion> criteria;

  /* @brief Seconds the combined result must hold CONTINUOUSLY before the outcome is reported.
   * 0 reports on the first satisfied frame. Debounces a gate that a single frame of contact
   * jitter, or an object bouncing through the goal region, would otherwise trip. */
  real validationTimeS = 0.0_r;

  MOCHI_STRUCT_BEGIN(superdex::robotics::CriteriaGroup)
  MOCHI_FIELD(combinator)
  MOCHI_FIELD(criteria) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_FIELD(validationTimeS) MOCHI_ATTRIBUTE(NoSerializeDefaults) MOCHI_ATTRIBUTE(Units("s"));
  MOCHI_STRUCT_END()
};

/*
 * @brief When an episode ends and with what outcome.
 *
 * Success and failure are separate groups so each carries its own combinator: success is normally
 * `All` of several gates, failure `Any` of several disqualifying conditions, which one shared
 * combinator could not express. Either may be absent (empty criteria).
 */
struct TerminationCriteria {
  CriteriaGroup success;

  /* @brief Defaults its combinator to @ref CriteriaCombinator::Any, unlike @ref success.
   *
   * The useful default differs by outcome: every success gate must hold, but any one
   * disqualifying condition ends the episode. */
  CriteriaGroup failure{.combinator = CriteriaCombinator::Any};

  MOCHI_STRUCT_BEGIN(superdex::robotics::TerminationCriteria)
  MOCHI_FIELD(success) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_FIELD(failure) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_STRUCT_END()
};

/*
 * @brief Resolve a criterion's task-object reference to its world pose.
 *
 * Returns nullopt when the reference does not resolve — a spawn that was not created, or a
 * `"<spawn>.<sub>"` address naming a sub-actor this prefab does not produce. @ref BotScene supplies
 * one over its own contents; a callable keeps the evaluator testable without a scene.
 */
using TaskObjectPoseLookup = std::function<std::optional<TaskObjectPose>(std::string_view)>;

/*
 * @brief Evaluate one criterion against the current world state.
 *
 * A criterion whose references do not resolve evaluates FALSE rather than raising, so an
 * unresolved dependency can never spuriously report success. Stateless, so it ignores `latch`;
 * @ref TerminationMonitor applies that.
 */
MOCHI_API bool EvaluateTerminationCriterion(
    TerminationCriterion const& criterion,
    TaskObjectPoseLookup const& poseLookup);

/**
 * @brief Complete parameter set for a .superdex_task file.
 *
 * A .superdex_task is a JSON config that declares which spawnable-prefab
 * instances to place (and where) on top of a .superdex_scene. It is a
 * declarative, swappable task definition: one base scene can host many tasks.
 */
struct BotTaskPrefab {
  /// Human-readable metadata. Not consumed by the runtime.
  BotSceneMetadata metadata;

  /// The objects an episode starts with, and where: each entry is placed at every reset.
  DynamicArray<SpawnEntry> resetState;

  /// When the episode ends and with what outcome. Optional: a task declaring neither outcome
  /// leaves every consumer on whatever predicate it already implements.
  TerminationCriteria terminationCriteria;

  /// The transform from this task's own coordinates into the task frame (the nominal table top):
  /// for a table spawned at T, it is inverse(T). On a scene that declares
  /// @ref BaseSceneParams::taskRoot, root-level spawns are relocated by taskRoot * taskRootOffset;
  /// see @ref ComputeWorldFromTask. Unset, the task is authored in the task frame itself.
  std::optional<TransformRT> taskRootOffset;

  /* @brief The rate the task's physics steps at [Hz]; 0: the consumer keeps its own. */
  int physicsHz = 0;

  /* @brief How long an episode may run before it ends with neither outcome [s of sim time]; 0: no
   * limit. */
  real timeLimitS = 0.0_r;

  // --- LEGACY (remove once all .mochi_bot_task assets carry `resetState`) ---
  // Older tasks named the entries `spawns`. LoadBotTaskPrefabFromFile folds them into
  // `resetState` and clears this, so re-saves omit it. TODO(superdex): delete this field, its
  // MOCHI_FIELD_NAME line, and the fold.
  DynamicArray<SpawnEntry> _legacySpawns;

  MOCHI_STRUCT_BEGIN(superdex::robotics::BotTaskPrefab)
  MOCHI_FIELD(metadata) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_FIELD(resetState) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_FIELD(terminationCriteria) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_FIELD(taskRootOffset) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_FIELD(physicsHz) MOCHI_ATTRIBUTE(NoSerializeDefaults) MOCHI_ATTRIBUTE(Units("1/s"));
  MOCHI_FIELD(timeLimitS) MOCHI_ATTRIBUTE(NoSerializeDefaults) MOCHI_ATTRIBUTE(Units("s"));
  MOCHI_FIELD_NAME(_legacySpawns, "spawns") MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_STRUCT_END()
};

/* @brief Both outcomes of a task. */
struct TerminationOutcome {
  /* @brief Whether the task file declares this outcome at all. When false, the consumer keeps
   * whatever predicate it already implements; the matching flag below is then meaningless. */
  bool successAuthored = false;
  bool success = false;
  bool failureAuthored = false;
  bool failure = false;
};

/*
 * @brief Decides a task's termination criteria step by step, carrying the state they need across
 * steps: each criterion's evaluation schedule, held result and latch, and each group's dwell.
 *
 * @ref BotScene drives one of these from its own step callback; this class is the evaluator it
 * drives, kept free of the scene so its behaviour can be tested directly.
 *
 * The authored-vs-false distinction is reported rather than folded in, because getting it wrong
 * is silent: treat "no criteria declared" as "criteria said no" and a task that never opted in
 * starts reporting failure instead of deferring to the consumer's built-in predicate.
 */
class TerminationMonitor {
 public:
  TerminationMonitor() = default;
  explicit TerminationMonitor(TerminationCriteria criteria);

  /* @brief Evaluate at episode time @p timeS.
   *
   * Only criteria whose evaluation is due are evaluated; the rest hold their last result. Time
   * going backwards is taken as a new episode and clears all carried state. */
  TerminationOutcome Update(TaskObjectPoseLookup const& poseLookup, real timeS);

  /* @brief Clear held results, latches and dwell clocks; every criterion is evaluated again on the
   * next update. The evaluation schedule stays aligned to the episode's start. */
  void Reset();

 private:
  struct CriterionState {
    bool held = false;
    bool latched = false;
    real nextEvaluationS = 0.0_r;
  };
  struct GroupState {
    std::vector<CriterionState> criteria;
    std::optional<real> heldSince;
  };

  /* @brief Evaluate whichever of @p group's criteria are due, combine, and apply the dwell. */
  static bool UpdateGroup(
      CriteriaGroup const& group,
      GroupState& state,
      TaskObjectPoseLookup const& poseLookup,
      real timeS);

  TerminationCriteria _criteria;
  real _timeS = 0.0_r;
  GroupState _success;
  GroupState _failure;
};

/// Reserved @ref SpawnEntry::parent value referring to the task actor's own
/// frame. Also the implicit default when @ref SpawnEntry::parent is empty.
/// Spawn entries may not use this string as their @ref SpawnEntry::name.
constexpr std::string_view kTaskRootParentName = "task_root";

/**
 * @brief Load task parameters from a .superdex_task JSON file.
 *
 * Deserializes the file and validates that spawn names are non-empty, unique,
 * and not equal to @ref kTaskRootParentName; that each spawn references a
 * non-empty prefab name; and that @ref SpawnEntry::parent references either
 * @ref kTaskRootParentName or the name of another spawn in this task. Parent
 * graphs must be acyclic. Existence of @ref SpawnEntry::prefabName in any
 * scene is intentionally NOT validated here (loose binding) — that is checked
 * at apply time against the loaded scene.
 *
 * @param[in] path File path to the .superdex_task file.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 * @return Loaded @ref BotTaskPrefab, or default-constructed on failure.
 */
MOCHI_API BotTaskPrefab LoadBotTaskPrefabFromFile(std::string_view path, superdex::Error& error);

/**
 * @brief Validate a @ref BotTaskPrefab in isolation (no scene binding).
 *
 * @param[in] params The task parameters to validate.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 */
MOCHI_API void ValidateBotTaskPrefab(BotTaskPrefab const& params, superdex::Error& error);

/* @brief The DR episode key an entry's @ref PoseRandomization offset is drawn under,
 * `dr/spawn/<spawnName>/<domainRandomizationClass>/pose`. Its value is the offset pose
 * `[tx, ty, tz, qx, qy, qz, qw]` in the parent's frame. The DR system's samplers and manifests
 * already use this key, so it is defined once here for every producer and consumer. */
MOCHI_API DynamicString
SpawnPoseDRKey(std::string_view spawnName, std::string_view domainRandomizationClass);

/* @brief The DR episode key an entry's @ref PrefabSelection is drawn under,
 * `dr/spawn/<spawnName>/prefab`. Its value is a scalar index into
 * @ref PrefabSelection::prefabNames. */
MOCHI_API DynamicString SpawnPrefabDRKey(std::string_view spawnName);

/* @brief The DR class of @p entry's @ref PoseRandomization, or empty if it has none. For callers
 * that cannot see the variant, such as the Python binding. */
MOCHI_API DynamicString GetPoseRandomizationClass(SpawnEntry const& entry);

/* @brief The candidates of @p entry's @ref PrefabSelection, or empty if it has none. For callers
 * that cannot see the variant, such as the Python binding. */
MOCHI_API DynamicArray<DynamicString> GetPrefabSelection(SpawnEntry const& entry);

/* @brief One entry of a task's reset state, after its randomization was applied. */
struct ResolvedSpawn {
  DynamicString name;
  /* @brief The prefab to place: the @ref PrefabSelection draw, or the nominal
   * @ref SpawnEntry::prefabName without one. */
  DynamicString prefabName;
  /* @brief As declared; empty or @ref kTaskRootParentName for the task root. */
  DynamicString parent;
  /* @brief @ref SpawnEntry::parentFromSpawn with the @ref PoseRandomization offset applied. */
  TransformRT parentFromSpawn;
  /* @brief The entry's pose composed through its parent chain, with the task root at the
   * worldFromTask passed to @ref ResolveResetState. */
  TransformRT worldFromSpawn;
};

/*
 * @brief Where a task's root-level spawns go on a scene: scene.taskRoot * task.taskRootOffset, with
 * an absent offset taken as identity; identity when the scene declares no task root.
 *
 * A scene without a task root keeps every task in its own coordinates, which is where every task
 * went before scenes declared one. A task without an offset is authored in the task frame.
 *
 * @param[in] scene The bot scene's scene block.
 * @param[in] task The task.
 * @return The world-from-task transform to pass to @ref ResolveResetState.
 */
MOCHI_API TransformRT ComputeWorldFromTask(BaseSceneParams const& scene, BotTaskPrefab const& task);

/*
 * @brief Resolve a task's reset state for one episode from that episode's DR values.
 *
 * The single place reset-state randomization is applied, so every surface places objects the same
 * way. The DR system generates the values; this only reads them, under @ref SpawnPoseDRKey and
 * @ref SpawnPrefabDRKey. An entry whose key is absent is placed as declared, so empty @p values
 * resolve the nominal state.
 *
 * A pose offset is applied in the parent's frame before the parent's pose is composed in: its
 * translation is added to the entry's, and its rotation, renormalized, pre-multiplies the entry's.
 * A child without an offset therefore moves rigidly with a randomized parent. Root-level entries
 * are composed onto @p worldFromTask, so a root entry's offset follows the task frame.
 *
 * @param[in] task The task; need not have been loaded from a file.
 * @param[in] values The episode's DR values. Keys this does not read are ignored.
 * @param[in] worldFromTask Where the task's root-level spawns go; see @ref ComputeWorldFromTask.
 * @param[in,out] error Set for a parent that does not resolve, a pose value that is not seven
 *                      numbers or carries a degenerate rotation, or a prefab index that is not an
 *                      integer index into the entry's candidates.
 * @return One entry per reset-state entry, parents before their children; empty on error.
 */
MOCHI_API DynamicArray<ResolvedSpawn> ResolveResetState(
    BotTaskPrefab const& task,
    DynamicArray<DREpisodeEntry> const& values,
    TransformRT const& worldFromTask,
    superdex::Error& error);

/* @brief @ref ResolveResetState with the task in its own coordinates (identity worldFromTask). */
MOCHI_API DynamicArray<ResolvedSpawn> ResolveResetState(
    BotTaskPrefab const& task,
    DynamicArray<DREpisodeEntry> const& values,
    superdex::Error& error);

} // namespace superdex::robotics
