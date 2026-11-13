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

#include <mochi_core/utils/log.h>
#include <superdex_robotics/scene/bot_task.h>
#include <superdex_robotics/utils/file_utils.h>

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#if !MOCHI_USE_REFLECTION
#error Reflection is required for the mochi_bots library internals.
#endif

using namespace mochi;
using namespace superdex::robotics;

namespace superdex::robotics {

namespace {

/* @brief Resolve the spawn a criterion's task-object reference names.
 *
 * A reference is either a spawn name ("m8_knob") or a sub-actor address of one
 * ("m8_knob.knob" — an articulation link, or a companion actor of a multi-actor prefab). Only the
 * spawn part is checkable here: which sub-actors a prefab produces is not known until the task is
 * applied to a scene, matching how @ref SpawnEntry::prefabName is loosely bound. */
std::string_view SpawnOfTaskObjectRef(std::string_view ref) {
  auto const dot = ref.find('.');
  return dot == std::string_view::npos ? ref : ref.substr(0, dot);
}

void ValidateTaskObjectRef(
    std::unordered_set<std::string> const& spawnNames,
    std::string_view fieldName,
    std::string_view value,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  if (value.empty()) {
    MOCHI_LOG_ERROR(
        "BotTaskPrefab: termination criterion %s is empty", std::string(fieldName).c_str());
    MOCHI_ERROR_IF(true, error, "BotTaskPrefab: termination criterion has an empty task object");
    return;
  }
  auto const spawn = SpawnOfTaskObjectRef(value);
  if (spawnNames.find(std::string(spawn)) == spawnNames.end()) {
    MOCHI_LOG_ERROR(
        "BotTaskPrefab: termination criterion %s references unknown task object '%s'",
        std::string(fieldName).c_str(),
        std::string(value).c_str());
    MOCHI_ERROR_IF(
        true, error, "BotTaskPrefab: termination criterion references unknown task object");
  }
}

/* @brief Two references may not name the same BODY. Sub-actor addresses of the same spawn are
 * distinct bodies, so they are allowed — "m8_knob.knob" vs "m8_knob.screw" is a real comparison. */
void ValidateDistinct(std::string_view kind, std::string_view a, std::string_view b, Error& error) {
  MOCHI_ERROR_RETURN(error);
  if (a == b) {
    MOCHI_LOG_ERROR(
        "BotTaskPrefab: %s references the same task object '%s' on both sides",
        std::string(kind).c_str(),
        std::string(a).c_str());
    MOCHI_ERROR_IF(true, error, "BotTaskPrefab: criterion requires distinct task objects");
  }
}

real EvaluationRateS(TerminationCriterion const& criterion) {
  return std::visit([](auto const& c) { return c.evaluationRateS; }, criterion);
}

void ValidateEvaluationRate(std::string_view kind, real rateS, Error& error) {
  MOCHI_ERROR_RETURN(error);
  if (!(rateS >= 0.0_r)) {
    MOCHI_LOG_ERROR("BotTaskPrefab: %s evaluationRateS must be >= 0", std::string(kind).c_str());
    MOCHI_ERROR_IF(true, error, "BotTaskPrefab: evaluationRateS must be >= 0");
  }
}

void ValidateCriterion(
    SphericalProximity const& c,
    std::unordered_set<std::string> const& spawnNames,
    Error& error) {
  ValidateTaskObjectRef(spawnNames, "taskObjectA", c.taskObjectA, error);
  ValidateTaskObjectRef(spawnNames, "taskObjectB", c.taskObjectB, error);
  ValidateDistinct("SphericalProximity", c.taskObjectA, c.taskObjectB, error);
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(
      !(c.thresholdM >= 0.0_r), error, "BotTaskPrefab: SphericalProximity thresholdM must be >= 0");
  ValidateEvaluationRate("SphericalProximity", c.evaluationRateS, error);
}

void ValidateCriterion(
    RelativeBounds const& c,
    std::unordered_set<std::string> const& spawnNames,
    Error& error) {
  ValidateTaskObjectRef(spawnNames, "targetTaskObject", c.targetTaskObject, error);
  ValidateTaskObjectRef(spawnNames, "taskObject", c.taskObject, error);
  ValidateDistinct("RelativeBounds", c.targetTaskObject, c.taskObject, error);
  MOCHI_ERROR_RETURN(error);
  // Only absolute ranges are ordered. As offsets onto the target's box an inverted pair insets
  // that face, and whether it inverts the result depends on extents unknown until runtime.
  if (!c.useTargetBounds) {
    MOCHI_ERROR_IF(
        c.xRangeM[0] > c.xRangeM[1] || c.yRangeM[0] > c.yRangeM[1] || c.zRangeM[0] > c.zRangeM[1],
        error,
        "BotTaskPrefab: RelativeBounds range has min > max");
  }
  ValidateEvaluationRate("RelativeBounds", c.evaluationRateS, error);
}

void ValidateCriterion(
    AxisAlignment const& c,
    std::unordered_set<std::string> const& spawnNames,
    Error& error) {
  ValidateTaskObjectRef(spawnNames, "taskObjectA", c.taskObjectA, error);
  ValidateTaskObjectRef(spawnNames, "taskObjectB", c.taskObjectB, error);
  MOCHI_ERROR_RETURN(error);
  // Unlike the other two, the objects MAY coincide: aligning two axes of one body is meaningful.
  MOCHI_ERROR_IF(
      Dot(c.axisALocal, c.axisALocal) <= 0.0_r || Dot(c.axisBLocal, c.axisBLocal) <= 0.0_r,
      error,
      "BotTaskPrefab: AxisAlignment axes must be non-zero");
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(
      !(c.maxErrorThreshold >= 0.0_r),
      error,
      "BotTaskPrefab: AxisAlignment maxErrorThreshold must be >= 0");
  ValidateEvaluationRate("AxisAlignment", c.evaluationRateS, error);
}

void ValidateCriteriaGroup(
    CriteriaGroup const& group,
    std::string_view outcome,
    std::unordered_set<std::string> const& spawnNames,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  if (group.criteria.empty()) {
    return; // Outcome not authored.
  }
  if (!(group.validationTimeS >= 0.0_r)) {
    MOCHI_LOG_ERROR(
        "BotTaskPrefab: terminationCriteria.%s.validationTimeS must be >= 0",
        std::string(outcome).c_str());
    MOCHI_ERROR_IF(true, error, "BotTaskPrefab: validationTimeS must be >= 0");
    return;
  }
  real slowestRateS = 0.0_r;
  for (auto const& criterion : group.criteria) {
    std::visit([&](auto const& c) { ValidateCriterion(c, spawnNames, error); }, criterion);
    MOCHI_ERROR_RETURN(error);
    slowestRateS = std::max(slowestRateS, EvaluationRateS(criterion));
  }
  if (group.validationTimeS > 0.0_r && slowestRateS > group.validationTimeS) {
    MOCHI_LOG_WARNING(
        "BotTaskPrefab: terminationCriteria.%s.validationTimeS (%g s) is shorter than a "
        "criterion's evaluationRateS (%g s), so the dwell elapses before that criterion is "
        "evaluated again",
        std::string(outcome).c_str(),
        static_cast<double>(group.validationTimeS),
        static_cast<double>(slowestRateS));
  }
}

void ValidateRandomization(SpawnEntry const& spawn, Error& error) {
  MOCHI_ERROR_RETURN(error);
  if (spawn.randomization.empty()) {
    return;
  }
  // The DR keys embed the spawn name and DR class as path segments.
  MOCHI_ERROR_IF(
      std::string_view(spawn.name).find('/') != std::string_view::npos,
      error,
      "BotTaskPrefab: a randomized spawn's name may not contain '/'");
  MOCHI_ERROR_RETURN(error);
  int poseCount = 0;
  int selectionCount = 0;
  for (auto const& randomization : spawn.randomization) {
    if (auto const* pose = std::get_if<PoseRandomization>(&randomization)) {
      ++poseCount;
      std::string_view const drClass = pose->domainRandomizationClass;
      MOCHI_ERROR_IF(
          drClass.empty() || drClass.find('/') != std::string_view::npos,
          error,
          "BotTaskPrefab: PoseRandomization needs a domainRandomizationClass without '/'");
      MOCHI_ERROR_RETURN(error);
    } else if (auto const* selection = std::get_if<PrefabSelection>(&randomization)) {
      ++selectionCount;
      MOCHI_ERROR_IF(
          selection->prefabNames.empty(),
          error,
          "BotTaskPrefab: PrefabSelection needs at least one prefab name");
      MOCHI_ERROR_RETURN(error);
      std::unordered_set<std::string_view> seen;
      for (auto const& prefabName : selection->prefabNames) {
        MOCHI_ERROR_IF(
            prefabName.empty() || !seen.insert(std::string_view(prefabName)).second,
            error,
            "BotTaskPrefab: PrefabSelection prefab names must be non-empty and distinct");
        MOCHI_ERROR_RETURN(error);
      }
    }
  }
  if (poseCount > 1 || selectionCount > 1) {
    MOCHI_LOG_ERROR(
        "BotTaskPrefab: spawn '%s' declares more than one randomization of a type",
        spawn.name.c_str());
    MOCHI_ERROR_IF(true, error, "BotTaskPrefab: at most one randomization of each type per spawn");
  }
}

} // namespace

void ValidateBotTaskPrefab(BotTaskPrefab const& params, Error& error) {
  MOCHI_ERROR_RETURN(error);

  MOCHI_ERROR_IF(params.physicsHz < 0, error, "BotTaskPrefab: physicsHz is negative");
  MOCHI_ERROR_IF(
      !(params.timeLimitS >= 0.0_r) || !std::isfinite(params.timeLimitS),
      error,
      "BotTaskPrefab: timeLimitS is negative or not finite");
  MOCHI_ERROR_RETURN(error);

  std::unordered_set<std::string> spawnNames;
  std::unordered_map<std::string, std::string> nameToParent;
  for (auto const& spawn : params.resetState) {
    MOCHI_ERROR_IF(spawn.name.empty(), error, "BotTaskPrefab: spawn entry has empty name");
    MOCHI_ERROR_RETURN(error);
    if (std::string_view(spawn.name) == kTaskRootParentName) {
      MOCHI_LOG_ERROR(
          "BotTaskPrefab: spawn name '%s' is reserved for the task root frame", spawn.name.c_str());
      MOCHI_ERROR_IF(true, error, "BotTaskPrefab: spawn name conflicts with reserved task_root");
      return;
    }
    MOCHI_ERROR_IF(
        spawn.prefabName.empty(), error, "BotTaskPrefab: spawn entry has empty prefabName");
    MOCHI_ERROR_RETURN(error);
    MOCHI_ERROR_IF(
        !spawnNames.insert(std::string(spawn.name)).second,
        error,
        "BotTaskPrefab: duplicate spawn name");
    MOCHI_ERROR_RETURN(error);
    ValidateRandomization(spawn, error);
    MOCHI_ERROR_RETURN(error);
    nameToParent.emplace(std::string(spawn.name), std::string(spawn.parent));
  }

  // Validate parent references and reject cycles. Each spawn has at most one
  // parent, so cycles can only show up as a closed chain — walk the chain from
  // each spawn until we hit the root (empty / "task_root"), bail on revisit.
  for (auto const& spawn : params.resetState) {
    std::unordered_set<std::string> visited;
    visited.insert(std::string(spawn.name));
    std::string_view current = spawn.parent;
    while (!current.empty() && current != kTaskRootParentName) {
      if (spawnNames.find(std::string(current)) == spawnNames.end()) {
        MOCHI_LOG_ERROR(
            "BotTaskPrefab: spawn '%s' references unknown parent '%s'",
            spawn.name.c_str(),
            std::string(current).c_str());
        MOCHI_ERROR_IF(true, error, "BotTaskPrefab: spawn references unknown parent");
        return;
      }
      if (!visited.insert(std::string(current)).second) {
        MOCHI_LOG_ERROR("BotTaskPrefab: parent cycle involving spawn '%s'", spawn.name.c_str());
        MOCHI_ERROR_IF(true, error, "BotTaskPrefab: parent cycle");
        return;
      }
      current = nameToParent.at(std::string(current));
    }
  }

  ValidateCriteriaGroup(params.terminationCriteria.success, "success", spawnNames, error);
  MOCHI_ERROR_RETURN(error);
  ValidateCriteriaGroup(params.terminationCriteria.failure, "failure", spawnNames, error);
}

/* LEGACY ONLY: older tasks named the reset-state entries `spawns` (captured by
 * BotTaskPrefab::_legacySpawns). Move them into resetState and clear the legacy field so a re-save
 * emits only resetState. A file carrying both keys is ambiguous and rejected rather than merged.
 * Remove together with the legacy field. */
static void ApplyLegacySpawns(BotTaskPrefab& params, Error& error) {
  if (params._legacySpawns.empty()) {
    return;
  }
  MOCHI_ERROR_IF(
      !params.resetState.empty(),
      error,
      "BotTaskPrefab: declares both `resetState` and the legacy `spawns`; keep only `resetState`");
  MOCHI_ERROR_RETURN(error);
  params.resetState = std::move(params._legacySpawns);
  params._legacySpawns.clear();
}

/* LEGACY ONLY: older tasks named a pose DR class in a flat `domainRandomizationClass` key
 * (captured by SpawnEntry::_legacyDomainRandomizationClass). Fold it into a PoseRandomization
 * entry and clear it so a re-save emits only `randomization`. An entry carrying both is ambiguous
 * and rejected rather than merged. Remove together with the legacy field. */
static void ApplyLegacyDomainRandomizationClasses(BotTaskPrefab& params, Error& error) {
  for (auto& spawn : params.resetState) {
    if (spawn._legacyDomainRandomizationClass.empty()) {
      continue;
    }
    bool const hasPose = std::ranges::any_of(spawn.randomization, [](auto const& randomization) {
      return std::holds_alternative<PoseRandomization>(randomization);
    });
    MOCHI_ERROR_IF(
        hasPose,
        error,
        "BotTaskPrefab: a spawn declares both a PoseRandomization and the legacy "
        "`domainRandomizationClass`; keep only the PoseRandomization");
    MOCHI_ERROR_RETURN(error);
    spawn.randomization.push_back(
        PoseRandomization{std::move(spawn._legacyDomainRandomizationClass)});
    spawn._legacyDomainRandomizationClass = {};
  }
}

/* LEGACY ONLY: older tasks stored a spawn's pose as separate flat "translation"/"rotation" keys
 * (captured by SpawnEntry::_legacyTranslation/_legacyRotation during reflection). Fold those into
 * parentFromSpawn when it was not explicitly provided, then clear the legacy fields so a re-save
 * emits only parentFromSpawn. Remove together with the legacy fields on SpawnEntry. */
static void ApplyLegacySpawnTransforms(BotTaskPrefab& params) {
  for (auto& spawn : params.resetState) {
    TransformRT legacyTransform;
    legacyTransform.SetTranslation(spawn._legacyTranslation);
    legacyTransform.SetRotation(spawn._legacyRotation);
    // Only fold a real legacy pose, and never override an explicitly-provided parentFromSpawn.
    if (!(legacyTransform == TransformRT{}) && spawn.parentFromSpawn == TransformRT{}) {
      spawn.parentFromSpawn = legacyTransform;
    }
    spawn._legacyTranslation = {};
    spawn._legacyRotation = {};
  }
}

BotTaskPrefab LoadBotTaskPrefabFromFile(std::string_view requestedPath, Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  std::string const pathStr = RemapBotFilePath(requestedPath, error);
  MOCHI_ERROR_RETURN(error, {});
  std::string_view const path(pathStr);

  MOCHI_ERROR_IF(!IsBotTaskPath(path), error, "Unsupported file format; expected .superdex_task");
  MOCHI_ERROR_RETURN(error, {});

  // allowPartial: optional fields (e.g. parentFromSpawn) keep their defaults when omitted instead
  // of failing the load.
  auto params = LoadParamsFromFile<BotTaskPrefab>(path, error, /*allowPartial=*/true);
  MOCHI_ERROR_RETURN(error, {});

  // LEGACY: fold the old `spawns` key into resetState, then each entry's old flat
  // translation/rotation into parentFromSpawn and old flat DR class into a PoseRandomization.
  ApplyLegacySpawns(params, error);
  MOCHI_ERROR_RETURN(error, {});
  ApplyLegacySpawnTransforms(params);
  ApplyLegacyDomainRandomizationClasses(params, error);
  MOCHI_ERROR_RETURN(error, {});

  ValidateBotTaskPrefab(params, error);
  MOCHI_ERROR_RETURN(error, {});

  return params;
}

DynamicString SpawnPoseDRKey(
    std::string_view spawnName,
    std::string_view domainRandomizationClass) {
  std::string key = "dr/spawn/";
  key.append(spawnName).append("/").append(domainRandomizationClass).append("/pose");
  return DynamicString(std::string_view(key));
}

DynamicString SpawnPrefabDRKey(std::string_view spawnName) {
  std::string key = "dr/spawn/";
  key.append(spawnName).append("/prefab");
  return DynamicString(std::string_view(key));
}

DynamicString GetPoseRandomizationClass(SpawnEntry const& entry) {
  for (auto const& randomization : entry.randomization) {
    if (auto const* pose = std::get_if<PoseRandomization>(&randomization)) {
      return pose->domainRandomizationClass;
    }
  }
  return {};
}

DynamicArray<DynamicString> GetPrefabSelection(SpawnEntry const& entry) {
  for (auto const& randomization : entry.randomization) {
    if (auto const* selection = std::get_if<PrefabSelection>(&randomization)) {
      return selection->prefabNames;
    }
  }
  return {};
}

namespace {

/* Below this an offset quaternion has no direction to renormalize to. Matches the Python
 * DR pipeline's reference implementation. */
constexpr real kOffsetQuaternionNormEpsilon = 1e-12_r;

/* @brief Apply a `[tx, ty, tz, qx, qy, qz, qw]` offset in the parent's frame: translation added,
 * rotation renormalized and pre-multiplied. The norm is an explicit fixed-order sum of squares, as
 * in the Python DR pipeline's pose-offset reference, so both stay bit-for-bit on its goldens. */
TransformRT
ApplyPoseOffset(TransformRT const& parentFromSpawn, DREpisodeEntry const& offset, Error& error) {
  MOCHI_ERROR_IF(
      offset.value.size() != 7,
      error,
      "ResolveResetState: a spawn pose value must be [tx, ty, tz, qx, qy, qz, qw]");
  MOCHI_ERROR_RETURN(error, parentFromSpawn);
  real const qx = offset.value[3];
  real const qy = offset.value[4];
  real const qz = offset.value[5];
  real const qw = offset.value[6];
  real const norm = std::sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
  MOCHI_ERROR_IF(
      !(norm >= kOffsetQuaternionNormEpsilon),
      error,
      "ResolveResetState: a spawn pose value carries a degenerate rotation");
  MOCHI_ERROR_RETURN(error, parentFromSpawn);
  Quaternion const rotation(qx / norm, qy / norm, qz / norm, qw / norm);
  Real3 const translation{offset.value[0], offset.value[1], offset.value[2]};
  return TransformRT(
      rotation * parentFromSpawn.GetRotation(), parentFromSpawn.GetTranslation() + translation);
}

/* @brief The candidate a `[index]` prefab value selects. */
DynamicString
SelectPrefab(PrefabSelection const& selection, DREpisodeEntry const& index, Error& error) {
  bool const valid = index.value.size() == 1 && index.value[0] >= 0.0_r &&
      std::floor(index.value[0]) == index.value[0] &&
      index.value[0] < static_cast<real>(selection.prefabNames.size());
  MOCHI_ERROR_IF(
      !valid,
      error,
      "ResolveResetState: a spawn prefab value must be an index into its PrefabSelection");
  MOCHI_ERROR_RETURN(error, {});
  return selection.prefabNames[static_cast<size_t>(index.value[0])];
}

} // namespace

TransformRT ComputeWorldFromTask(BaseSceneParams const& scene, BotTaskPrefab const& task) {
  if (!scene.taskRoot.has_value()) {
    return TransformRT::Identity();
  }
  return *scene.taskRoot * task.taskRootOffset.value_or(TransformRT::Identity());
}

DynamicArray<ResolvedSpawn> ResolveResetState(
    BotTaskPrefab const& task,
    DynamicArray<DREpisodeEntry> const& values,
    Error& error) {
  return ResolveResetState(task, values, TransformRT::Identity(), error);
}

DynamicArray<ResolvedSpawn> ResolveResetState(
    BotTaskPrefab const& task,
    DynamicArray<DREpisodeEntry> const& values,
    TransformRT const& worldFromTask,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  std::unordered_map<std::string_view, DREpisodeEntry const*> valueByKey;
  for (auto const& value : values) {
    valueByKey.emplace(std::string_view(value.key), &value);
  }
  auto const valueOf = [&](DynamicString const& key) -> DREpisodeEntry const* {
    auto const it = valueByKey.find(std::string_view(key));
    return it == valueByKey.end() ? nullptr : it->second;
  };

  // Each entry's own randomized state, in declaration order.
  std::vector<ResolvedSpawn> entries;
  entries.reserve(task.resetState.size());
  for (auto const& spawn : task.resetState) {
    ResolvedSpawn& entry = entries.emplace_back();
    entry.name = spawn.name;
    entry.prefabName = spawn.prefabName;
    entry.parent = spawn.parent;
    entry.parentFromSpawn = spawn.parentFromSpawn;
    for (auto const& randomization : spawn.randomization) {
      if (auto const* pose = std::get_if<PoseRandomization>(&randomization)) {
        if (auto const* offset =
                valueOf(SpawnPoseDRKey(spawn.name, pose->domainRandomizationClass))) {
          entry.parentFromSpawn = ApplyPoseOffset(entry.parentFromSpawn, *offset, error);
        }
      } else if (auto const* selection = std::get_if<PrefabSelection>(&randomization)) {
        if (auto const* index = valueOf(SpawnPrefabDRKey(spawn.name))) {
          entry.prefabName = SelectPrefab(*selection, *index, error);
        }
      }
      MOCHI_ERROR_RETURN(error, {});
    }
  }

  // Compose through the parent chain in repeated passes; a child later in the list resolves in the
  // same pass as its parent, so the order matches the Unreal loader's.
  std::unordered_map<std::string, TransformRT> worldByName;
  std::vector<bool> placed(entries.size(), false);
  DynamicArray<ResolvedSpawn> resolved;
  resolved.reserve(entries.size());
  while (resolved.size() < entries.size()) {
    size_t const before = resolved.size();
    for (size_t i = 0; i < entries.size(); ++i) {
      if (placed[i]) {
        continue;
      }
      ResolvedSpawn& entry = entries[i];
      std::string_view const parent = entry.parent;
      TransformRT worldFromParent = worldFromTask;
      if (!parent.empty() && parent != kTaskRootParentName) {
        auto const it = worldByName.find(std::string(parent));
        if (it == worldByName.end()) {
          continue;
        }
        worldFromParent = it->second;
      }
      entry.worldFromSpawn = worldFromParent * entry.parentFromSpawn;
      worldByName.emplace(std::string(entry.name), entry.worldFromSpawn);
      placed[i] = true;
      resolved.push_back(entry);
    }
    MOCHI_ERROR_IF(
        resolved.size() == before,
        error,
        "ResolveResetState: a spawn's parent does not resolve (unknown name or cycle)");
    MOCHI_ERROR_RETURN(error, {});
  }
  return resolved;
}

} // namespace superdex::robotics

namespace superdex::robotics {

namespace {

/* Guards the divide when an authored axis is not unit length. The validator rejects a strictly
 * zero axis, so this only rescues near-zero ones. */
Real3 NormalizedOrZero(Real3 const& v) {
  real const norm = std::sqrt(Dot(v, v));
  return norm > 0.0_r ? Real3{v[0] / norm, v[1] / norm, v[2] / norm} : Real3{};
}

bool CompareProximity(ProximityComparator comparator, real distance, real threshold) {
  switch (comparator) {
    case ProximityComparator::LessThan:
      return distance < threshold;
    case ProximityComparator::LessEqual:
      return distance <= threshold;
    case ProximityComparator::GreaterThan:
      return distance > threshold;
    case ProximityComparator::GreaterEqual:
      return distance >= threshold;
  }
  return false;
}

bool EvaluateOne(SphericalProximity const& c, TaskObjectPoseLookup const& poseLookup) {
  auto const a = poseLookup(c.taskObjectA);
  auto const b = poseLookup(c.taskObjectB);
  if (!a.has_value() || !b.has_value()) {
    return false;
  }
  Real3 const delta = a->position - b->position;
  return CompareProximity(c.comparator, std::sqrt(Dot(delta, delta)), c.thresholdM);
}

bool EvaluateOne(RelativeBounds const& c, TaskObjectPoseLookup const& poseLookup) {
  auto const target = poseLookup(c.targetTaskObject);
  auto const object = poseLookup(c.taskObject);
  if (!target.has_value() || !object.has_value()) {
    return false;
  }
  Real3 lo{c.xRangeM[0], c.yRangeM[0], c.zRangeM[0]};
  Real3 hi{c.xRangeM[1], c.yRangeM[1], c.zRangeM[1]};
  if (c.useTargetBounds) {
    // Before the comparator, so a target carrying no box cannot report Outside.
    if (!target->localBounds.has_value()) {
      return false;
    }
    lo = lo + target->localBounds->GetMin();
    hi = hi + target->localBounds->GetMax();
  }
  // Into the target's local frame, so the box travels with it.
  Real3 const local = target->rotation.GetConjugate() * (object->position - target->position);
  bool const inside = local[0] >= lo[0] && local[0] <= hi[0] && local[1] >= lo[1] &&
      local[1] <= hi[1] && local[2] >= lo[2] && local[2] <= hi[2];
  switch (c.comparator) {
    case BoundsComparator::Inside:
      return inside;
    case BoundsComparator::Outside:
      return !inside;
  }
  return false;
}

bool EvaluateOne(AxisAlignment const& c, TaskObjectPoseLookup const& poseLookup) {
  auto const a = poseLookup(c.taskObjectA);
  auto const b = poseLookup(c.taskObjectB);
  if (!a.has_value() || !b.has_value()) {
    return false;
  }
  Real3 const axisAWorld = NormalizedOrZero(a->rotation * c.axisALocal);
  Real3 const axisBWorld = NormalizedOrZero(b->rotation * c.axisBLocal);
  // 1 - cos(angle): 0 when parallel, 2 when antiparallel. Deliberately NOT symmetric about
  // antiparallel — an axis pointing the wrong way is not aligned.
  return (1.0_r - Dot(axisAWorld, axisBWorld)) < c.maxErrorThreshold;
}

} // namespace

bool EvaluateTerminationCriterion(
    TerminationCriterion const& criterion,
    TaskObjectPoseLookup const& poseLookup) {
  return std::visit([&](auto const& c) { return EvaluateOne(c, poseLookup); }, criterion);
}

} // namespace superdex::robotics

namespace superdex::robotics {

namespace {

/* Absorbs the rounding of a clock summed from step sizes, so a step landing a hair before a
 * multiple of the rate takes that evaluation rather than slipping one step later. */
constexpr real kScheduleToleranceS = 1e-6_r;

bool IsLatched(TerminationCriterion const& criterion) {
  return std::visit([](auto const& c) { return c.latch; }, criterion);
}

/* @brief The next point after @p timeS on the episode's evaluation grid for @p rateS. */
real NextEvaluationS(real timeS, real rateS) {
  if (rateS <= 0.0_r) {
    return timeS;
  }
  return (std::floor((timeS + kScheduleToleranceS) / rateS) + 1.0_r) * rateS;
}

/* @brief Apply a group's dwell: report true only once @p satisfied has held continuously for
 * @ref CriteriaGroup::validationTimeS. One unsatisfied update restarts the clock. */
bool Held(CriteriaGroup const& group, bool satisfied, std::optional<real>& heldSince, real timeS) {
  if (!satisfied) {
    heldSince.reset();
    return false;
  }
  if (!heldSince.has_value()) {
    heldSince = timeS;
  }
  return timeS - *heldSince >= group.validationTimeS;
}

} // namespace

TerminationMonitor::TerminationMonitor(TerminationCriteria criteria)
    : _criteria(std::move(criteria)) {
  _success.criteria.assign(_criteria.success.criteria.size(), CriterionState{});
  _failure.criteria.assign(_criteria.failure.criteria.size(), CriterionState{});
}

bool TerminationMonitor::UpdateGroup(
    CriteriaGroup const& group,
    GroupState& state,
    TaskObjectPoseLookup const& poseLookup,
    real timeS) {
  // No short-circuit: under `All`, a stage listed after a criterion that does not hold yet must
  // still get its evaluation, or it could never latch.
  bool anyPassed = false;
  bool allPassed = true;
  for (size_t i = 0; i < group.criteria.size(); ++i) {
    TerminationCriterion const& criterion = group.criteria[i];
    CriterionState& criterionState = state.criteria[i];
    if (!criterionState.latched && timeS + kScheduleToleranceS >= criterionState.nextEvaluationS) {
      criterionState.held = EvaluateTerminationCriterion(criterion, poseLookup);
      criterionState.latched = criterionState.held && IsLatched(criterion);
      criterionState.nextEvaluationS = NextEvaluationS(timeS, EvaluationRateS(criterion));
    }
    bool const passes = criterionState.latched || criterionState.held;
    anyPassed = anyPassed || passes;
    allPassed = allPassed && passes;
  }
  bool const satisfied = group.combinator == CriteriaCombinator::Any ? anyPassed : allPassed;
  return Held(group, satisfied, state.heldSince, timeS);
}

TerminationOutcome TerminationMonitor::Update(TaskObjectPoseLookup const& poseLookup, real timeS) {
  if (timeS < _timeS) {
    Reset();
  }
  _timeS = timeS;

  TerminationOutcome outcome;
  outcome.successAuthored = !_criteria.success.criteria.empty();
  outcome.failureAuthored = !_criteria.failure.criteria.empty();
  if (outcome.successAuthored) {
    outcome.success = UpdateGroup(_criteria.success, _success, poseLookup, timeS);
  }
  if (outcome.failureAuthored) {
    outcome.failure = UpdateGroup(_criteria.failure, _failure, poseLookup, timeS);
  }
  return outcome;
}

void TerminationMonitor::Reset() {
  for (GroupState* state : {&_success, &_failure}) {
    std::ranges::fill(state->criteria, CriterionState{});
    state->heldSince.reset();
  }
}

} // namespace superdex::robotics
