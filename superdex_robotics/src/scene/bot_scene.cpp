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

#include <mochi_core/utils/container_utils.h>
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/json_utils.h>
#include <mochi_physics/utils/mochi_prefab.h>
#include <superdex_robotics/core/loader.h>
#include <superdex_robotics/scene/bot_scene.h>
#include <superdex_robotics/scene/bot_task.h>
#include <superdex_robotics/superdex_robotics.h>
#include <superdex_robotics/utils/archive_utils.h>
#include <superdex_robotics/utils/bot_utils.h>
#include <superdex_robotics/utils/file_utils.h>

#include <algorithm>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if !MOCHI_USE_REFLECTION
#error Reflection is required for the mochi_bots library internals.
#endif

using namespace mochi;
using namespace superdex::robotics;

// ---------------------------------------------------------------------------
// Validation (unchanged from Diff 2)
// ---------------------------------------------------------------------------

namespace {

void ValidateBotScenePrefab(BotScenePrefab const& params, Error& error) {
  MOCHI_ERROR_RETURN(error);

  std::unordered_set<std::string> botNames;
  for (auto const& bot : params.bots) {
    MOCHI_ERROR_IF(bot.name.empty(), error, "BotScenePrefab: bot entry has empty name");
    MOCHI_ERROR_RETURN(error);
    MOCHI_ERROR_IF(bot.path.empty(), error, "BotScenePrefab: bot entry has empty path");
    MOCHI_ERROR_RETURN(error);
    MOCHI_ERROR_IF(
        !botNames.insert(std::string(bot.name)).second,
        error,
        "BotScenePrefab: duplicate bot name");
    MOCHI_ERROR_RETURN(error);

    std::unordered_set<std::string> controllerNames;
    for (auto const& ctrl : bot.controllers) {
      MOCHI_ERROR_IF(ctrl.type.empty(), error, "BotScenePrefab: controller entry has empty type");
      MOCHI_ERROR_RETURN(error);
      MOCHI_ERROR_IF(ctrl.name.empty(), error, "BotScenePrefab: controller entry has empty name");
      MOCHI_ERROR_RETURN(error);
      MOCHI_ERROR_IF(
          !controllerNames.insert(std::string(ctrl.name)).second,
          error,
          "BotScenePrefab: duplicate controller name within bot");
      MOCHI_ERROR_RETURN(error);
    }
  }

  std::unordered_set<std::string> prefabNames;
  for (auto const& prefab : params.scene.spawnablePrefabs) {
    MOCHI_ERROR_IF(
        prefab.name.empty(), error, "BotScenePrefab: spawnable prefab entry has empty name");
    MOCHI_ERROR_RETURN(error);
    MOCHI_ERROR_IF(
        prefab.path.empty(), error, "BotScenePrefab: spawnable prefab entry has empty path");
    MOCHI_ERROR_RETURN(error);
    MOCHI_ERROR_IF(
        !prefabNames.insert(std::string(prefab.name)).second,
        error,
        "BotScenePrefab: duplicate spawnable prefab name");
    MOCHI_ERROR_RETURN(error);
  }

  std::unordered_set<std::string> sensorNames;
  for (auto const& sensor : params.sensors) {
    MOCHI_ERROR_IF(sensor.type.empty(), error, "BotScenePrefab: sensor entry has empty type");
    MOCHI_ERROR_RETURN(error);
    MOCHI_ERROR_IF(sensor.name.empty(), error, "BotScenePrefab: sensor entry has empty name");
    MOCHI_ERROR_RETURN(error);
    MOCHI_ERROR_IF(
        !sensorNames.insert(std::string(sensor.name)).second,
        error,
        "BotScenePrefab: duplicate sensor name");
    MOCHI_ERROR_RETURN(error);
  }

  // Structural validation only: superdex is a carrier for the slot map. It guarantees that slot
  // names are non-empty and unique, that a slot offers at least one bot, that no bot is listed
  // twice in one slot, and that every bot named actually exists. It knows nothing about what a
  // slot means, and does not require that every bot belong to one -- a scene may hold bots that
  // are always present rather than selectable.
  std::unordered_set<std::string> slotNames;
  for (auto const& slot : params.slots) {
    MOCHI_ERROR_IF(slot.name.empty(), error, "BotScenePrefab: slot entry has empty name");
    MOCHI_ERROR_RETURN(error);
    MOCHI_ERROR_IF(
        !slotNames.insert(std::string(slot.name)).second,
        error,
        "BotScenePrefab: duplicate slot name");
    MOCHI_ERROR_RETURN(error);
    // A slot nothing can fill can only ever fail at selection time, so reject it here instead.
    MOCHI_ERROR_IF(
        slot.controlledBots.empty(), error, "BotScenePrefab: slot entry names no controlled bots");
    MOCHI_ERROR_RETURN(error);

    std::unordered_set<std::string> slotBots;
    for (auto const& botName : slot.controlledBots) {
      MOCHI_ERROR_IF(
          !slotBots.insert(std::string(botName)).second,
          error,
          "BotScenePrefab: duplicate controlled bot within slot");
      MOCHI_ERROR_RETURN(error);
      MOCHI_ERROR_IF(
          botNames.find(std::string(botName)) == botNames.end(),
          error,
          "BotScenePrefab: slot names a controlled bot that is not in the scene");
      MOCHI_ERROR_RETURN(error);
    }
  }
}

void ResolvePaths(BotScenePrefab& params, std::filesystem::path const& basePath, Error& error) {
  MOCHI_ERROR_RETURN(error);
  MakePathAbsolute(params.scene.baseScene, basePath, kBotPathMaxParentDepth, error);
  for (auto& prefab : params.scene.spawnablePrefabs) {
    MakePathAbsolute(prefab.path, basePath, kBotPathMaxParentDepth, error);
  }
  for (auto& bot : params.bots) {
    MakePathAbsolute(bot.path, basePath, kBotPathMaxParentDepth, error);
    for (auto& ctrl : bot.controllers) {
      MakeParamsPathAbsolute(ctrl.params, basePath, error);
      // NOTE: ctrl.initArgs is intentionally NOT path-resolved here. Today its only fields are link
      // names (BASIC_OSC_PD baseLinkName/eeLinkName), which are not paths. If a future controller
      // puts a FILE PATH in its init args, resolve it here (and add the symmetric MakePathRelative
      // in SaveToFile(BotScenePrefab)) so it round-trips like ctrl.params.
    }
  }
  for (auto& sensor : params.sensors) {
    MakeParamsPathAbsolute(sensor.params, basePath, error);
  }
}

/* LEGACY ONLY: older scenes interleave the controller-specific fields as flat siblings of
 * type/name/params (e.g. baseLinkName/eeLinkName) instead of nesting them under "initArgs". For
 * backwards compatibility, fold those siblings into initArgs for any controller that lacks a nested
 * "initArgs" object -- new scenes carry that object, which the JsonString-tagged ControllerEntry
 * field captures directly during reflection (so there is nothing to do for them here). Saving
 * always emits the nested form, so a loaded legacy scene is rewritten on save. Do not rely on or
 * extend this. Matched to the deserialized structs by array position (SReflect preserves array
 * order). */
void ApplyLegacyControllerInitArgs(BotScenePrefab& params, picojson::value const& rawRoot) {
  if (!rawRoot.is<picojson::object>()) {
    return;
  }
  picojson::object const& rootObj = rawRoot.get<picojson::object>();
  auto const botsIt = rootObj.find("bots");
  if (botsIt == rootObj.end() || !botsIt->second.is<picojson::array>()) {
    return;
  }
  picojson::array const& rawBots = botsIt->second.get<picojson::array>();
  for (int b = 0; b < isize(params.bots); ++b) {
    if (b >= isize(rawBots) || !rawBots[b].is<picojson::object>()) {
      continue;
    }
    picojson::object const& rawBot = rawBots[b].get<picojson::object>();
    auto const ctrlsIt = rawBot.find("controllers");
    if (ctrlsIt == rawBot.end() || !ctrlsIt->second.is<picojson::array>()) {
      continue;
    }
    picojson::array const& rawCtrls = ctrlsIt->second.get<picojson::array>();
    for (int c = 0; c < isize(params.bots[b].controllers); ++c) {
      if (c >= isize(rawCtrls) || !rawCtrls[c].is<picojson::object>()) {
        continue;
      }
      picojson::object const& rawCtrl = rawCtrls[c].get<picojson::object>();
      if (rawCtrl.find("initArgs") != rawCtrl.end()) {
        // New format: the nested "initArgs" object was already captured by reflection.
        continue;
      }
      // Gather the interleaved controller-specific siblings (everything except the framework fields
      // type/name/params) into an initArgs object so old scenes still configure their controllers.
      picojson::object legacyInitArgs;
      for (auto const& [key, value] : rawCtrl) {
        if (key != "type" && key != "name" && key != "params") {
          legacyInitArgs.emplace(key, value);
        }
      }
      if (!legacyInitArgs.empty()) {
        params.bots[b].controllers[c].initArgs =
            DynamicString(picojson::value(legacyInitArgs).serialize());
      }
    }
  }
}

/* LEGACY ONLY: older scenes stored a scene sensor's pose as separate flat "translation"/"rotation"
 * keys (captured by SceneSensorEntry::_legacyTranslation/_legacyRotation during reflection) plus a
 * now-vestigial "attachTo" (absorbed and dropped via IgnoreExtraneousFields). Fold those into
 * parentFromSensor when it was not explicitly provided, then clear the legacy fields so a re-save
 * emits only parentFromSensor. Remove together with the legacy fields on SceneSensorEntry. */
void ApplyLegacySensorTransforms(BotScenePrefab& params) {
  for (auto& sensor : params.sensors) {
    TransformRT legacyTransform;
    legacyTransform.SetTranslation(sensor._legacyTranslation);
    legacyTransform.SetRotation(sensor._legacyRotation);
    // Only fold a real legacy pose, and never override an explicitly-provided parentFromSensor.
    if (!(legacyTransform == TransformRT{}) && sensor.parentFromSensor == TransformRT{}) {
      sensor.parentFromSensor = legacyTransform;
    }
    sensor._legacyTranslation = {};
    sensor._legacyRotation = {};
  }
}

/* LEGACY ONLY: older scenes stored a bot's placement as separate flat "translation"/"rotation" keys
 * (captured by BotEntry::_legacyTranslation/_legacyRotation during reflection). Fold those into
 * parentFromBot when it was not explicitly provided, then clear the legacy fields so a re-save
 * emits only parentFromBot. Remove together with the legacy fields on BotEntry. */
void ApplyLegacyBotEntryTransforms(BotScenePrefab& params) {
  for (auto& bot : params.bots) {
    TransformRT legacyTransform;
    legacyTransform.SetTranslation(bot._legacyTranslation);
    legacyTransform.SetRotation(bot._legacyRotation);
    // Only fold a real legacy pose, and never override an explicitly-provided parentFromBot.
    if (!(legacyTransform == TransformRT{}) && bot.parentFromBot == TransformRT{}) {
      bot.parentFromBot = legacyTransform;
    }
    bot._legacyTranslation = {};
    bot._legacyRotation = {};
  }
}

/* The file a bot scene actually loads from: an archive's extracted target, else `path` itself. Root
 * paths must be derived from this, not from the archive's own location, or references inside the
 * archive resolve against whatever tree the archive happens to sit in. */
std::string ResolveBotSceneFile(std::string_view path, Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  std::string pathStr(path);
  if (!IsSceneArchivePath(path)) {
    return pathStr;
  }
  // A missing archive is reported as "not found" rather than falling through to an extraction
  // error, which would misleadingly blame the archive format.
  if (!std::filesystem::is_regular_file(pathStr)) {
    MOCHI_LOG_ERROR("Bot scene archive not found (or not a regular file): '%s'", pathStr.c_str());
    MOCHI_ERROR_SET(error, "Bot scene archive not found (or not a regular file)");
    return {};
  }
  auto const extractedDir = ExtractBotSceneArchiveToCache(path, error);
  MOCHI_ERROR_RETURN(error, {});
  auto const targetPath = GetExtractedBotSceneArchiveTarget(extractedDir, error);
  MOCHI_ERROR_RETURN(error, {});
  return std::string(targetPath.c_str(), targetPath.size());
}

/* prefab::AddToScene is not transactional: on failure it can leave behind the actors and
 * constraints it reports in `result`. Destroy those, constraints first since they reference the
 * actors. Contact-filter changes it may have applied are not reported and are not undone. */
void DestroyAddedToScene(Scene* scene, prefab::AddToSceneResult const& result) {
  // Take handles up front: destroying an actor can invalidate pointers into the result.
  DynamicArray<ConstraintHandle> constraints;
  for (auto* constraint : result.constraints) {
    constraints.push_back(constraint->GetHandle());
  }
  DynamicArray<ActorHandle> actors;
  for (auto* actor : result.actors) {
    actors.push_back(actor->GetHandle());
  }
  for (auto handle : constraints) {
    if (scene->GetConstraint(handle) != nullptr) {
      scene->DestroyConstraint(handle);
    }
  }
  for (auto handle : actors) {
    if (scene->GetActor(handle) != nullptr) {
      scene->DestroyActor(handle);
    }
  }
}

using ActorRenderModelMap = std::unordered_map<uint64_t, ActorRenderModel>;
using ActorSkinRenderModelMap = std::unordered_map<uint64_t, ActorSkinRenderModel>;

/* The skin render model of a bot whose articulated actor is `actor`, placed as Studio places it:
 * through the rest pose of the link skin bone 0 binds to. Rest poses are the actor's own link
 * frames at zero pose, so they include joint-to-link offsets and the bot's scale; the model's
 * offset and scale carry the bot's scale too. Empty, with an error logged for the last case, if the
 * bot has no skin render model or no skin bones, or if the actor's links do not match the bot's. */
std::optional<ActorSkinRenderModel>
MakeSkinRenderModel(BotPrefab const& botPrefab, Actor const& actor, Span<ActorHandle const> links) {
  if (!botPrefab.skin.has_value() || botPrefab.skin->renderModelFile.empty() ||
      isize(links) != isize(botPrefab.links)) {
    return std::nullopt;
  }
  auto const& skin = *botPrefab.skin;
  Error error;
  ArticulatedShapeInfo const info = actor.GetArticulatedShapeInfo(error);
  if (error.IsOK() && isize(info.rootFromLinksAtRest) != isize(links)) {
    MOCHI_ERROR_SET(error, "the articulated actor's links do not match the bot's");
  }
  if (!error.IsOK()) {
    MOCHI_LOG_ERROR(
        "Cannot map the skin of bot '%s' onto its links (%s); drawing the links instead of the "
        "skin.",
        botPrefab.name.c_str(),
        error.ToString().c_str());
    return std::nullopt;
  }
  DynamicArray<DynamicString> linkNames;
  for (auto const& link : botPrefab.links) {
    linkNames.push_back(link.name);
  }
  DynamicArray<int> const boneLinks =
      ComputeSkinBoneLinkIndices(botPrefab, MakeConstSpan(linkNames));
  if (boneLinks.empty()) {
    return std::nullopt;
  }

  real const scale = botPrefab.scale;
  ActorSkinRenderModel model;
  model.file = skin.renderModelFile;
  int const rootLink = boneLinks[0] != kIndexNone ? boneLinks[0] : 0;
  model.rootFromModel = info.rootFromLinksAtRest[rootLink] *
      TransformRT(skin.renderModelRotation, skin.renderModelTranslation * scale);
  model.scale = skin.renderModelScale * scale;
  for (int const linkIndex : boneLinks) {
    bool const valid = linkIndex != kIndexNone;
    model.boneLinks.push_back(valid ? links[linkIndex] : ActorHandle{});
    model.rootFromBoneRest.push_back(
        valid ? info.rootFromLinksAtRest[linkIndex] : TransformRT::Identity());
  }
  DynamicArray<bool> const hiddenLinks =
      ComputeSkinHiddenLinks(botPrefab, MakeConstSpan(linkNames));
  for (int i = 0; i < isize(hiddenLinks); ++i) {
    if (hiddenLinks[i]) {
      model.hiddenLinks.push_back(links[i]);
    }
  }
  return model;
}

/* Pairs the actors prefab::AddToScene created from a loaded prefab with their render models. It
 * walks `prefab` -- the same object AddToScene was given -- in AddToScene's creation order: nested
 * prefabs depth-first, then articulated (one model per link), rigid, soft, rod, and soft-skinned
 * actors. Soft, rod, and soft-skinned actors get no model: their render geometry would have to
 * deform with them.
 * Scale follows the prefab importer's convention: a model's offset and scale are multiplied by its
 * actor's scale, including the scale of every enclosing nested prefab. */
class PrefabRenderModelRecorder {
 public:
  PrefabRenderModelRecorder(
      std::string_view rootPath,
      Span<Actor* const> actors,
      ActorRenderModelMap& models,
      DynamicArray<ActorHandle>& recorded)
      : _rootPath(rootPath), _actors(actors), _models(models), _recorded(recorded) {}

  /* Pairing is by position, so a walk that disagrees with AddToScene on the number of actors would
   * hand models to the wrong actors. Rather than mis-pair, such a prefab gets no render models. */
  void Record(prefab::ScenePrefab const& prefab) {
    Walk(prefab, 1_r);
    if (_overran || _next != _actors.size()) {
      MOCHI_LOG_ERROR(
          "Render models out of step with prefab::AddToScene's %d actors; dropping them.",
          static_cast<int>(_actors.size()));
      for (ActorHandle const actor : _recorded) {
        _models.erase(actor.value);
      }
      _recorded.clear();
    }
  }

 private:
  void Walk(prefab::ScenePrefab const& prefab, real prefabScale) {
    for (auto const& nested : prefab.prefabs) {
      if (nested.prefab) {
        Walk(*nested.prefab, prefabScale * nested.scale);
      }
    }
    for (auto const& actor : prefab.actors.articulated) {
      Actor* const created = Next();
      if (created == nullptr) {
        return;
      }
      real const scale = actor.scale * prefabScale;
      Span<ActorHandle const> const links = created->GetNestedLinkActors(ErrorAssert{});
      for (int i = 0; i < Min(isize(links), isize(actor.links)); ++i) {
        auto const& link = actor.links[i];
        Add(links[i],
            prefab,
            link.renderModelFile,
            link.renderModelRotation,
            link.renderModelTranslation,
            link.renderModelScale,
            Real3{scale, scale, scale});
      }
    }
    for (auto const& actor : prefab.actors.rigid) {
      Actor* const created = Next();
      if (created == nullptr) {
        return;
      }
      Add(created->GetHandle(),
          prefab,
          actor.renderModelFile,
          actor.renderModelRotation,
          actor.renderModelTranslation,
          actor.renderModelScale,
          actor.scale * prefabScale);
    }
    for (int i = 0; i <
         isize(prefab.actors.soft) + isize(prefab.actors.softSkinned) + isize(prefab.actors.rod);
         ++i) {
      Next();
    }
  }

  Actor* Next() {
    if (_next >= _actors.size()) {
      _overran = true;
      return nullptr;
    }
    return _actors[_next++];
  }

  void Add(
      ActorHandle actor,
      prefab::ScenePrefab const& prefab,
      DynamicString const& file,
      Quaternion const& rotation,
      Real3 const& translation,
      Real3 const& modelScale,
      Real3 const& actorScale) {
    if (file.empty()) {
      return;
    }
    std::string_view const prefabFile =
        prefab.sourceFilePath.has_value() ? std::string_view(*prefab.sourceFilePath) : "";
    ActorRenderModel& model = _models[actor.value];
    model.file = prefab::GetPrefabFullPath(file, _rootPath, prefabFile);
    model.actorFromModel = TransformRT(rotation, translation * actorScale);
    model.scale = modelScale * actorScale;
    _recorded.push_back(actor);
  }

  std::string_view _rootPath;
  Span<Actor* const> _actors;
  ActorRenderModelMap& _models;
  DynamicArray<ActorHandle>& _recorded;
  size_t _next = 0;
  bool _overran = false;
};

/* A bot's link actors are in BotPrefab::links order (see BuildArticulatedActorParams). */
void RecordBotRenderModels(
    Bot const& bot,
    ActorRenderModelMap& models,
    ActorSkinRenderModelMap& skinModels) {
  Actor* const actor = bot.GetArticulatedActor();
  if (actor == nullptr) {
    return;
  }
  Span<ActorHandle const> const links = actor->GetNestedLinkActors(ErrorAssert{});
  auto const& botPrefab = bot.GetBotPrefab();
  auto const& prefabLinks = botPrefab.links;
  for (int i = 0; i < Min(isize(links), isize(prefabLinks)); ++i) {
    auto const& link = prefabLinks[i];
    if (link.renderModelFile.empty()) {
      continue;
    }
    models[links[i].value] = ActorRenderModel{
        .file = link.renderModelFile,
        .actorFromModel = TransformRT(link.renderModelRotation, link.renderModelTranslation),
        .scale = link.renderModelScale};
  }
  if (auto skinModel = MakeSkinRenderModel(botPrefab, *actor, links)) {
    skinModels[actor->GetHandle().value] = std::move(*skinModel);
  }
}

void EraseBotRenderModels(
    Bot const& bot,
    ActorRenderModelMap& models,
    ActorSkinRenderModelMap& skinModels) {
  if (Actor* const actor = bot.GetArticulatedActor()) {
    for (ActorHandle const link : actor->GetNestedLinkActors(ErrorAssert{})) {
      models.erase(link.value);
    }
    skinModels.erase(actor->GetHandle().value);
  }
}

} // namespace

// ---------------------------------------------------------------------------
// LoadBotScenePrefabFromFile
// ---------------------------------------------------------------------------

BotScenePrefab superdex::robotics::LoadBotScenePrefabFromFile(
    std::string_view requestedPath,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});

  // Error descriptions must be string literals (Error stores a non-owning
  // char const*), so the offending path is surfaced via MOCHI_LOG_ERROR.
  std::string const pathStr = RemapBotFilePath(requestedPath, error);
  MOCHI_ERROR_RETURN(error, {});
  std::string_view const path(pathStr);

  // Transparent archive support: load from the extracted copy.
  if (IsSceneArchivePath(path)) {
    auto const targetPath = ResolveBotSceneFile(path, error);
    MOCHI_ERROR_RETURN(error, {});
    return LoadBotScenePrefabFromFile(targetPath, error);
  }

  if (!IsBotScenePath(path)) {
    MOCHI_LOG_ERROR(
        "Unsupported bot-scene file '%s': expected a '.superdex_scene' or "
        "'.superdex_scene_archive' file",
        pathStr.c_str());
    MOCHI_ERROR_SET(
        error,
        "Unsupported bot-scene file format; expected .superdex_scene or "
        ".superdex_scene_archive");
    return {};
  }

  if (!std::filesystem::is_regular_file(pathStr)) {
    MOCHI_LOG_ERROR("Bot scene file not found (or not a regular file): '%s'", pathStr.c_str());
    MOCHI_ERROR_SET(error, "Bot scene file not found (or not a regular file)");
    return {};
  }

  // Parse the scene JSON once; the same tree drives both the struct deserialize and the initArgs
  // extraction below (a controller's nested "initArgs" object is not a ControllerEntry field, so it
  // must be pulled out separately -- see AttachControllerInitArgs). Parsing once avoids reading and
  // parsing the file twice.
  picojson::value const rawRoot = ParseJsonFromFile(pathStr, error);
  MOCHI_ERROR_RETURN(error, {});

  // Deserialize with strict DeserializeFlags::Default (warn on extraneous keys). A controller's
  // nested "initArgs" object is captured directly by the JsonString-tagged
  // ControllerEntry::initArgs field, and ControllerEntry carries IgnoreExtraneousFields so legacy
  // scenes (which interleave flat controller keys) still load cleanly -- while typos elsewhere in
  // the scene are still flagged.
  BotScenePrefab params;
  int numIssues = 0;
  SReflect::FromJsonValue(params, rawRoot, SReflect::DeserializeFlags::Default, numIssues);
  if (numIssues > 0) {
    MOCHI_LOG_WARNING(
        "Detected %d issue(s) while loading bot scene: %s", numIssues, pathStr.c_str());
  }

  ValidateBotScenePrefab(params, error);
  MOCHI_ERROR_RETURN(error, {});

  ResolvePaths(params, path, error);
  MOCHI_ERROR_RETURN(error, {});

  // New scenes need nothing here (reflection already captured initArgs); this only folds legacy
  // flat controller fields into initArgs for backwards compatibility.
  ApplyLegacyControllerInitArgs(params, rawRoot);
  // Likewise fold legacy flat sensor translation/rotation into parentFromSensor.
  ApplyLegacySensorTransforms(params);
  // ...and legacy flat bot translation/rotation into parentFromBot.
  ApplyLegacyBotEntryTransforms(params);
  return params;
}

// ---------------------------------------------------------------------------
// BotScene implementation
// ---------------------------------------------------------------------------

namespace {

/* @brief Where an actor is now, in the frame @ref Actor::GetRootTransform reports.
 *
 * On an articulated actor @ref Actor::GetRootTransform is where it was placed, not where it is, so
 * this carries the root frame along with link 0 instead. Link 0's rest transform in the root frame
 * need not be the identity. */
TransformRT WorldFromActor(Actor const& actor) {
  if (actor.GetType() != ActorType::Articulated) {
    return actor.GetRootTransform();
  }
  Error error;
  ArticulatedShapeInfo const info = actor.GetArticulatedShapeInfo(error);
  DynamicArray<TransformRT> worldFromLinks;
  if (error.IsOK()) {
    worldFromLinks.resize(isize(info.rootFromLinksAtRest));
    actor.GetArticulatedLinkTransforms(MakeSpan(worldFromLinks), error);
  }
  if (!error.IsOK() || worldFromLinks.empty()) {
    MOCHI_LOG_ERROR(
        "Cannot read the links of articulated actor '%s' (%s); using its placement.",
        actor.GetName(),
        error.ToString().c_str());
    return actor.GetRootTransform();
  }
  return worldFromLinks[0] * Invert(info.rootFromLinksAtRest[0]);
}

std::optional<TaskObjectPose> PoseOf(Actor const* actor) {
  if (actor == nullptr) {
    return std::nullopt;
  }
  TransformRT const worldFromActor = WorldFromActor(*actor);
  TaskObjectPose pose;
  pose.position = worldFromActor.GetTranslation();
  pose.rotation = worldFromActor.GetRotation();
  mochi::Error boundsError;
  Aabb const localBounds = actor->GetAabbLocal(boundsError);
  if (boundsError.IsOK()) {
    pose.localBounds = localBounds;
  }
  return pose;
}

/* @brief Split a task-object reference into its spawn and optional sub-actor parts. */
std::pair<std::string_view, std::string_view> SplitTaskObjectRef(std::string_view ref) {
  auto const dot = ref.find('.');
  if (dot == std::string_view::npos) {
    return {ref, {}};
  }
  return {ref.substr(0, dot), ref.substr(dot + 1)};
}

/* @brief Resolve by actor name alone, for spawns that are not in the prefab-instance registry.
 *
 * A bare spawn falls back to the first `<spawn>/...` actor in scene order, which is creation order
 * and so the same root prefab::AddToScene returned first. */
Actor const* FindActorByName(Scene const& scene, std::string_view spawn, std::string_view sub) {
  std::string const exact =
      sub.empty() ? std::string(spawn) : std::string(spawn) + "/" + std::string(sub);
  std::string const prefix = std::string(spawn) + "/";
  Actor const* exactMatch = nullptr;
  Actor const* firstPrefixed = nullptr;
  scene.ForEachActor([&](Actor const* actor) {
    if (exactMatch != nullptr || actor == nullptr) {
      return;
    }
    std::string_view const name = actor->GetName();
    if (name == exact) {
      exactMatch = actor;
    } else if (sub.empty() && firstPrefixed == nullptr && name.starts_with(prefix)) {
      firstPrefixed = actor;
    }
  });
  return exactMatch != nullptr ? exactMatch : firstPrefixed;
}

} // namespace

struct BotScene::Impl {
  Impl() = default;
  Impl(Impl const&) = delete;
  Impl& operator=(Impl const&) = delete;
  Impl(Impl&&) = delete;
  Impl& operator=(Impl&&) = delete;

  /* Invoked (via the RegisterOnDestroy callback) when the owning RoboticsContext is destroyed
   * first. Drop our back-reference so ~Impl does not touch the freed context or the
   * bots/controllers it releases; ~Impl then only tears down the mochi Scene we own. */
  void OnRoboticsContextDestroyed() {
    botsCtx = nullptr;
  }

  BotScenePrefab params;
  Context* context = nullptr;
  RoboticsContext* botsCtx = nullptr;
  /* Token for our RoboticsContext teardown callback (see RegisterOnDestroy); cancelled in ~Impl
   * when this BotScene is destroyed before its context. */
  RoboticsContextCallbackHandle onDestroyHandle;
  /* Owning scene held as a stale-safe handle (resolved via @ref ResolveScene) rather than a raw
   * Scene*, so a BotScene that outlives its mochi Scene never dereferences a dangling pointer. */
  SceneHandle sceneHandle;
  bool ownsScene = true;

  /* Resolve the owning scene from its handle. Returns nullptr once the scene has been destroyed. */
  [[nodiscard]] Scene* ResolveScene() const {
    return context != nullptr ? context->GetScene(sceneHandle) : nullptr;
  }

  /* Base-scene actor handles (from the base ScenePrefab), in prefab::AddToScene order. */
  DynamicArray<ActorHandle> baseSceneActors;

  /* Render models of the actors and links this BotScene created, keyed by actor handle. Entries are
   * erased with their actors, since handles are recycled. */
  ActorRenderModelMap renderModels;

  /* Skin render models of the bots this BotScene created, keyed by articulated actor handle. Erased
   * with their bots. */
  ActorSkinRenderModelMap skinRenderModels;

  /* Bots keyed by their declared name. */
  std::unordered_map<std::string, Bot*> bots;

  /* Spawnable prefabs instantiated as articulated bots (via GetOrLoadSpawnableBot),
   * keyed by spawnable name. Owned here; destroyed in ~Impl. */
  std::unordered_map<std::string, Bot*> spawnableBots;

  /* Spawnable prefab absolute paths keyed by declared name. */
  std::unordered_map<std::string, std::string> spawnablePrefabPaths;

  struct PrefabInstance {
    std::string instanceName;
    DynamicArray<ActorHandle> actors;
    /* Actors and links of this instance that have an entry in renderModels. */
    DynamicArray<ActorHandle> renderModelActors;
    /* Set when the spawnable is a bot file: the instance is this bot, and actors holds its
     * articulated actor. Owned by botsCtx; destroyed with the instance. */
    Bot* bot = nullptr;
  };

  /* Every live spawnable-prefab instance, in creation order so GetPrefabInstanceNames is
   * deterministic. Instance counts are small, so lookup is linear. */
  std::vector<PrefabInstance> prefabInstances;

  [[nodiscard]] PrefabInstance* FindInstance(std::string_view instanceName) {
    for (auto& instance : prefabInstances) {
      if (instance.instanceName == instanceName) {
        return &instance;
      }
    }
    return nullptr;
  }

  [[nodiscard]] PrefabInstance const* FindInstance(std::string_view instanceName) const {
    for (auto const& instance : prefabInstances) {
      if (instance.instanceName == instanceName) {
        return &instance;
      }
    }
    return nullptr;
  }

  void DestroyInstanceActors(PrefabInstance const& instance) {
    if (instance.bot != nullptr) {
      /* Without botsCtx the RoboticsContext was destroyed first and already released the bot. */
      if (botsCtx != nullptr) {
        EraseBotRenderModels(*instance.bot, renderModels, skinRenderModels);
        DestroyBot(ResolveScene(), instance.bot);
      }
      return;
    }
    for (auto handle : instance.renderModelActors) {
      renderModels.erase(handle.value);
    }
    /* If the scene has already been destroyed, its actors went with it. */
    if (Scene* const scene = ResolveScene()) {
      for (auto handle : instance.actors) {
        scene->DestroyActor(handle);
      }
    }
  }

  /* Controller handles keyed by "botName/controllerName".
   * Lifetime is managed by the RoboticsContext — handles are destroyed in ~Impl. */
  std::unordered_map<std::string, ControllerHandle> controllers;

  /* Scene-level sensor handles keyed by their declared name. These belong to no bot (they are
   * created with no link actor), so DestroyBot never reaches them and ~Impl must. */
  std::unordered_map<std::string, SensorHandle> sceneSensors;

  /* The termination criteria being decided, written by any thread and evaluated on the stepping
   * thread; see StartTermination. */
  struct Termination {
    std::mutex mutex;
    std::optional<TerminationMonitor> monitor;
    double timeS = 0.0;
    TerminationOutcome outcome;
    /* Resolved actors by reference, each re-checked by name on use: a respawned or recycled
     * handle that no longer names the same actor is resolved again rather than trusted. */
    std::unordered_map<std::string, std::pair<ActorHandle, std::string>> resolved;
  };
  Termination termination;
  CallbackHandle terminationCallback;

  [[nodiscard]] Actor const* ResolveTaskObjectActor(std::string_view ref) const {
    Scene const* const scene = ResolveScene();
    if (scene == nullptr) {
      return nullptr;
    }
    auto const [spawn, sub] = SplitTaskObjectRef(ref);
    // A registered prefab instance already knows its actors and its root.
    if (PrefabInstance const* const instance = FindInstance(spawn)) {
      if (instance->actors.empty()) {
        return nullptr;
      }
      if (sub.empty()) {
        return scene->GetActor(instance->actors[0]);
      }
      std::string const wanted = std::string(spawn) + "/" + std::string(sub);
      for (auto const& handle : instance->actors) {
        Actor const* const actor = scene->GetActor(handle);
        if (actor != nullptr && std::string_view(actor->GetName()) == wanted) {
          return actor;
        }
      }
      // The links of an articulated actor are not instance actors themselves, but carry the
      // instance prefix; a free-root articulated object's moving body is only addressable this way.
      return FindActorByName(*scene, spawn, sub);
    }
    return FindActorByName(*scene, spawn, sub);
  }

  /* Requires termination.mutex. */
  [[nodiscard]] Actor const* ResolveTerminationActor(Scene const& scene, std::string_view ref) {
    std::string key(ref);
    auto const it = termination.resolved.find(key);
    if (it != termination.resolved.end()) {
      Actor const* const actor = scene.GetActor(it->second.first);
      if (actor != nullptr && std::string_view(actor->GetName()) == it->second.second) {
        return actor;
      }
    }
    Actor const* const actor = ResolveTaskObjectActor(ref);
    if (actor != nullptr) {
      termination.resolved[std::move(key)] = {actor->GetHandle(), std::string(actor->GetName())};
    } else if (it != termination.resolved.end()) {
      termination.resolved.erase(it);
    }
    return actor;
  }

  void EvaluateTermination(double timeStepS) {
    std::scoped_lock const lock(termination.mutex);
    if (!termination.monitor.has_value() || termination.outcome.success ||
        termination.outcome.failure) {
      return;
    }
    Scene const* const scene = ResolveScene();
    if (scene == nullptr) {
      return;
    }
    termination.timeS += timeStepS;
    TerminationOutcome const now = termination.monitor->Update(
        [&](std::string_view ref) { return PoseOf(ResolveTerminationActor(*scene, ref)); },
        static_cast<real>(termination.timeS));
    termination.outcome.success = now.success;
    termination.outcome.failure = now.failure;
  }

  /* Registered for the BotScene's whole life, on the thread that loads it — the scene's own
   * thread — and cancelled from ~Impl on the thread that destroys it, the same one; registering
   * per episode would put that on whichever thread starts the episode. */
  void RegisterTerminationCallback(Scene& scene) {
    terminationCallback = scene.RegisterPostStepCallback(
        "superdex::robotics::BotScene::EvaluateTermination",
        [this](StepInfo const& info) { EvaluateTermination(info.timeStepSec); });
  }

  ~Impl() {
    Scene* const resolvedScene = ResolveScene();
    if (resolvedScene != nullptr && terminationCallback.IsValid()) {
      resolvedScene->CancelCallback(terminationCallback);
    }
    /* Normal order (BotScene destroyed before its RoboticsContext): detach from the context, then
     * release our controllers and bots through it. DestroyBot resolves each bot's owning scene
     * through a handle and skips actor teardown once the scene is gone, so the Bot objects (owned
     * by botsCtx) are still released without touching a dead scene. */
    if (botsCtx != nullptr) {
      botsCtx->CancelOnDestroy(onDestroyHandle);
      for (auto& [key, handle] : controllers) {
        botsCtx->DestroyController(handle);
      }
      for (auto& [name, handle] : sceneSensors) {
        botsCtx->DestroySensor(handle);
      }
      for (auto& [name, bot] : bots) {
        DestroyBot(resolvedScene, bot);
      }
      for (auto& [name, bot] : spawnableBots) {
        DestroyBot(resolvedScene, bot);
      }
      for (auto& instance : prefabInstances) {
        if (instance.bot != nullptr) {
          DestroyBot(resolvedScene, instance.bot);
        }
      }
    }
    /* Destroy base-scene and prefab-instance actors regardless of scene ownership; only the
     * Scene itself is conditional on ownsScene. This ensures an externally-owned scene
     * (ownsScene==false) does not leak its actors when the BotScene is destroyed. */
    if (resolvedScene != nullptr) {
      for (auto handle : baseSceneActors) {
        resolvedScene->DestroyActor(handle);
      }
      for (auto& instance : prefabInstances) {
        if (instance.bot != nullptr) {
          continue;
        }
        for (auto handle : instance.actors) {
          resolvedScene->DestroyActor(handle);
        }
      }
    }
    /* If botsCtx is null the RoboticsContext was destroyed first and already released our bots and
     * controllers (see OnRoboticsContextDestroyed); skip that work. Either way, destroy the mochi
     * Scene we own if it is still alive. */
    if (resolvedScene != nullptr && context != nullptr && ownsScene) {
      context->DestroyScene(resolvedScene);
    }
  }
};

BotScene::BotScene() : impl_(std::make_unique<Impl>()) {}
BotScene::~BotScene() = default;
BotScene::BotScene(BotScene&&) noexcept = default;
BotScene& BotScene::operator=(BotScene&&) noexcept = default;

Scene* BotScene::GetScene() const {
  return impl_ ? impl_->ResolveScene() : nullptr;
}

Context* BotScene::GetMochiContext() const {
  return impl_ ? impl_->context : nullptr;
}

RoboticsContext* BotScene::GetRoboticsContext() const {
  return impl_ ? impl_->botsCtx : nullptr;
}

Bot* BotScene::GetBot(std::string_view name) const {
  if (!impl_) {
    return nullptr;
  }
  auto it = impl_->bots.find(std::string(name));
  return it != impl_->bots.end() ? it->second : nullptr;
}

Bot* BotScene::GetOrLoadSpawnableBot(std::string_view name, Error& error) {
  MOCHI_ERROR_RETURN(error, nullptr);
  MOCHI_ERROR_IF(!impl_, error, "BotScene is not initialized");
  MOCHI_ERROR_RETURN(error, nullptr);

  std::string nameStr(name);

  /* Return the cached bot if this spawnable was already instantiated. */
  auto cached = impl_->spawnableBots.find(nameStr);
  if (cached != impl_->spawnableBots.end()) {
    return cached->second;
  }

  auto pathIt = impl_->spawnablePrefabPaths.find(nameStr);
  MOCHI_ERROR_IF(
      pathIt == impl_->spawnablePrefabPaths.end(),
      error,
      "BotScene: unknown spawnable prefab name");
  MOCHI_ERROR_RETURN(error, nullptr);

  /* Only bot archives can be instantiated as articulated bots. Rigid
   * .mochi_scene spawnables must go through SpawnPrefabInstance instead. */
  std::string const& path = pathIt->second;
  MOCHI_ERROR_IF(
      !(IsBotArchivePath(path) || IsBotPath(path)),
      error,
      "BotScene: spawnable prefab is not a bot archive (.superdex_bot_archive/.superdex_bot)");
  MOCHI_ERROR_RETURN(error, nullptr);

  /* The stored spawnable path is already absolute (resolved at scene load), so
   * the bot archive loads without further path resolution. */
  BotPrefab botPrefab = LoadBotPrefabFromFile(path, error);
  MOCHI_ERROR_RETURN(error, nullptr);

  Scene* const scene = impl_->ResolveScene();
  MOCHI_ERROR_IF(scene == nullptr, error, "BotScene: scene is not available");
  MOCHI_ERROR_RETURN(error, nullptr);

  Bot* bot = CreateBot(scene, botPrefab, impl_->botsCtx, error);
  MOCHI_ERROR_RETURN(error, nullptr);
  RecordBotRenderModels(*bot, impl_->renderModels, impl_->skinRenderModels);

  impl_->spawnableBots[nameStr] = bot;
  return bot;
}

BotScenePrefab const& BotScene::GetBotScenePrefab() const {
  static BotScenePrefab const empty{};
  return impl_ ? impl_->params : empty;
}

ControllerHandle BotScene::GetControllerHandle(std::string_view key) const {
  if (!impl_) {
    return {};
  }
  auto it = impl_->controllers.find(std::string(key));
  return it != impl_->controllers.end() ? it->second : ControllerHandle{};
}

Span<ActorHandle const> BotScene::GetBaseSceneActorHandles() const {
  if (!impl_) {
    return {};
  }
  return MakeConstSpan(impl_->baseSceneActors);
}

ActorRenderModel const* BotScene::FindRenderModel(ActorHandle actor) const {
  if (!impl_) {
    return nullptr;
  }
  auto const it = impl_->renderModels.find(actor.value);
  return it != impl_->renderModels.end() ? &it->second : nullptr;
}

ActorSkinRenderModel const* BotScene::FindSkinRenderModel(ActorHandle actor) const {
  if (!impl_) {
    return nullptr;
  }
  auto const it = impl_->skinRenderModels.find(actor.value);
  return it != impl_->skinRenderModels.end() ? &it->second : nullptr;
}

Span<ActorHandle const> BotScene::SpawnPrefabInstance(
    std::string_view instanceName,
    std::string_view prefabName,
    TransformRT const& worldFromInstance,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  MOCHI_ERROR_IF(!impl_, error, "BotScene is not initialized");
  MOCHI_ERROR_RETURN(error, {});

  MOCHI_ERROR_IF(instanceName.empty(), error, "BotScene: prefab instance name is empty");
  MOCHI_ERROR_IF(
      impl_->FindInstance(instanceName) != nullptr,
      error,
      "BotScene: prefab instance name already live");
  auto pathIt = impl_->spawnablePrefabPaths.find(std::string(prefabName));
  MOCHI_ERROR_IF(
      pathIt == impl_->spawnablePrefabPaths.end(),
      error,
      "BotScene: unknown spawnable prefab name");
  Scene* const scene = impl_->ResolveScene();
  MOCHI_ERROR_IF(scene == nullptr, error, "BotScene: scene is not available");
  MOCHI_ERROR_RETURN(error, {});

  /* A bot file is not a ScenePrefab: it becomes an articulated Bot named after the instance, so its
   * link actors carry the instance prefix like an articulated prefab's. */
  std::string const& path = pathIt->second;
  if (IsBotArchivePath(path) || IsBotPath(path)) {
    MOCHI_ERROR_IF(!impl_->botsCtx, error, "BotScene: bots context is not initialized");
    BotPrefab botPrefab = LoadBotPrefabFromFile(path, error);
    MOCHI_ERROR_RETURN(error, {});
    botPrefab.name = DynamicString(instanceName);
    botPrefab.worldFromRoot = worldFromInstance;
    Bot* const bot = CreateBot(scene, botPrefab, impl_->botsCtx, error);
    MOCHI_ERROR_RETURN(error, {});
    RecordBotRenderModels(*bot, impl_->renderModels, impl_->skinRenderModels);

    Impl::PrefabInstance& instance = impl_->prefabInstances.emplace_back();
    instance.instanceName = std::string(instanceName);
    instance.bot = bot;
    instance.actors.push_back(bot->GetArticulatedActor()->GetHandle());
    return MakeConstSpan(instance.actors);
  }

  /* PrefabParams::name prefixes the created actor names, so two instances of one prefab stay
   * distinguishable in the scene. An instance must not replace the base scene's gravity or solver
   * configuration. */
  prefab::PrefabParams prefabParams;
  prefabParams.name = DynamicString(instanceName);
  prefabParams.rotation = worldFromInstance.GetRotation();
  prefabParams.translation = worldFromInstance.GetTranslation();
  prefabParams.applySceneSettings = false;

  std::string const rootPath = std::filesystem::path(pathIt->second).parent_path().string();
  /* Loaded here rather than by the path overload of AddToScene so the render models can be read
   * from the same prefab the actors were created from. */
  prefab::ScenePrefab const loaded =
      prefab::LoadFromFile(pathIt->second, rootPath, scene->GetContext(), error);
  auto const result = prefab::AddToScene(loaded, scene, prefabParams, error);
  if (!error.IsOK()) {
    DestroyAddedToScene(scene, result);
    return {};
  }

  Impl::PrefabInstance& instance = impl_->prefabInstances.emplace_back();
  instance.instanceName = std::string(instanceName);
  for (auto* actor : result.actors) {
    instance.actors.push_back(actor->GetHandle());
  }
  PrefabRenderModelRecorder(
      rootPath, MakeConstSpan(result.actors), impl_->renderModels, instance.renderModelActors)
      .Record(loaded);
  return MakeConstSpan(instance.actors);
}

void BotScene::DestroyPrefabInstance(std::string_view instanceName, Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(!impl_, error, "BotScene is not initialized");
  MOCHI_ERROR_RETURN(error);

  auto it = std::find_if(
      impl_->prefabInstances.begin(),
      impl_->prefabInstances.end(),
      [instanceName](Impl::PrefabInstance const& instance) {
        return instance.instanceName == instanceName;
      });
  MOCHI_ERROR_IF(
      it == impl_->prefabInstances.end(), error, "BotScene: prefab instance is not live");
  MOCHI_ERROR_RETURN(error);

  impl_->DestroyInstanceActors(*it);
  impl_->prefabInstances.erase(it);
}

void BotScene::DestroyAllPrefabInstances() {
  if (!impl_) {
    return;
  }
  for (auto const& instance : impl_->prefabInstances) {
    impl_->DestroyInstanceActors(instance);
  }
  impl_->prefabInstances.clear();
}

Span<ActorHandle const> BotScene::GetPrefabInstanceActors(std::string_view instanceName) const {
  if (!impl_) {
    return {};
  }
  Impl::PrefabInstance const* instance = impl_->FindInstance(instanceName);
  return instance != nullptr ? MakeConstSpan(instance->actors) : Span<ActorHandle const>{};
}

std::optional<TaskObjectPose> BotScene::ResolveTaskObjectPose(std::string_view ref) const {
  return impl_ ? PoseOf(impl_->ResolveTaskObjectActor(ref)) : std::nullopt;
}

void BotScene::StartTermination(BotTaskPrefab const& task) {
  if (!impl_) {
    return;
  }
  Impl::Termination& termination = impl_->termination;
  std::scoped_lock const lock(termination.mutex);
  TerminationCriteria const& criteria = task.terminationCriteria;
  termination.outcome = TerminationOutcome{};
  termination.outcome.successAuthored = !criteria.success.criteria.empty();
  termination.outcome.failureAuthored = !criteria.failure.criteria.empty();
  termination.timeS = 0.0;
  termination.resolved.clear();
  if (termination.outcome.successAuthored || termination.outcome.failureAuthored) {
    termination.monitor.emplace(criteria);
  } else {
    termination.monitor.reset();
  }
}

void BotScene::StopTermination() {
  if (!impl_) {
    return;
  }
  Impl::Termination& termination = impl_->termination;
  std::scoped_lock const lock(termination.mutex);
  termination.monitor.reset();
  termination.outcome = TerminationOutcome{};
  termination.resolved.clear();
}

void BotScene::RestartTerminationProgress() {
  if (!impl_) {
    return;
  }
  Impl::Termination& termination = impl_->termination;
  std::scoped_lock const lock(termination.mutex);
  if (termination.monitor.has_value()) {
    termination.monitor->Reset();
  }
  termination.resolved.clear();
}

TerminationOutcome BotScene::GetTerminationOutcome() const {
  if (!impl_) {
    return {};
  }
  std::scoped_lock const lock(impl_->termination.mutex);
  return impl_->termination.outcome;
}

DynamicArray<DynamicString> BotScene::GetPrefabInstanceNames() const {
  DynamicArray<DynamicString> names;
  if (!impl_) {
    return names;
  }
  names.reserve(impl_->prefabInstances.size());
  for (auto const& instance : impl_->prefabInstances) {
    names.push_back(DynamicString(instance.instanceName));
  }
  return names;
}

void BotScene::RemoveBot(std::string_view name, Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(!impl_, error, "BotScene is not initialized");
  MOCHI_ERROR_RETURN(error);

  std::string nameStr(name);
  auto it = impl_->bots.find(nameStr);
  MOCHI_ERROR_IF(it == impl_->bots.end(), error, "BotScene: bot not currently loaded");
  MOCHI_ERROR_RETURN(error);

  /* A registered bot implies botsCtx and scene are set; guard the invariant so a
   * partially-initialized BotScene raises rather than dereferencing null (as ~Impl
   * defensively guards its teardown). The scene is resolved from its handle and may be null if the
   * mochi Scene was already destroyed; DestroyBot below tolerates that (it skips actor teardown).
   */
  MOCHI_ERROR_IF(!impl_->botsCtx, error, "BotScene: bots context is not initialized");
  MOCHI_ERROR_RETURN(error);
  Scene* const scene = impl_->ResolveScene();

  /* Destroy this bot's controllers first, mirroring ~Impl's controllers-then-bot
   * teardown order. They are created via RoboticsContext::CreateController and tracked
   * here by "botName/controllerName" key, so DestroyBot does not own them. */
  std::string const keyPrefix = nameStr + "/";
  for (auto cit = impl_->controllers.begin(); cit != impl_->controllers.end();) {
    if (cit->first.starts_with(keyPrefix)) {
      impl_->botsCtx->DestroyController(cit->second);
      cit = impl_->controllers.erase(cit);
    } else {
      ++cit;
    }
  }

  /* DestroyBot destroys the bot's sensors before its articulated actor (see
   * RoboticsContext::DestroyBot), the ordering that makes retiring a seed-sensor
   * bot safe — the crash this method exists to prevent. */
  EraseBotRenderModels(*it->second, impl_->renderModels, impl_->skinRenderModels);
  DestroyBot(scene, it->second);
  impl_->bots.erase(it);
}

/* Scene-wide Find forwarders: delegate to the context-wide RoboticsContext finders (scope =
 * nullptr), which span every bot in this scene plus any scene-level components.
 * GetRoboticsContext() is null on an uninitialized/moved-from BotScene or once the owning
 * RoboticsContext has been destroyed; in that state there are no components to find, so each
 * forwarder logs a warning and returns an empty result rather than dereferencing null. These
 * forwarders take no Error& (matching every other no-Error lookup accessor here, e.g. GetBot); the
 * warning surfaces the likely misuse — querying a scene whose context is gone — without forcing an
 * error argument on callers. */
DynamicArray<ControllerHandle> BotScene::FindControllersByName(std::string_view name) const {
  RoboticsContext* const botsCtx = GetRoboticsContext();
  if (botsCtx == nullptr) {
    MOCHI_LOG_WARNING(
        "BotScene::FindControllersByName: scene has no context (uninitialized, moved-from, or context destroyed); returning empty.");
    return {};
  }
  return botsCtx->FindControllersByName(name, /*scope*/ nullptr);
}

DynamicArray<ControllerHandle> BotScene::FindControllersByType(std::string_view typeName) const {
  RoboticsContext* const botsCtx = GetRoboticsContext();
  if (botsCtx == nullptr) {
    MOCHI_LOG_WARNING(
        "BotScene::FindControllersByType: scene has no context (uninitialized, moved-from, or context destroyed); returning empty.");
    return {};
  }
  return botsCtx->FindControllersByType(typeName, /*scope*/ nullptr);
}

DynamicArray<SensorHandle> BotScene::FindSensorsByName(std::string_view name) const {
  RoboticsContext* const botsCtx = GetRoboticsContext();
  if (botsCtx == nullptr) {
    MOCHI_LOG_WARNING(
        "BotScene::FindSensorsByName: scene has no context (uninitialized, moved-from, or context destroyed); returning empty.");
    return {};
  }
  return botsCtx->FindSensorsByName(name, /*scope*/ nullptr);
}

DynamicArray<SensorHandle> BotScene::FindSensorsByType(std::string_view typeName) const {
  RoboticsContext* const botsCtx = GetRoboticsContext();
  if (botsCtx == nullptr) {
    MOCHI_LOG_WARNING(
        "BotScene::FindSensorsByType: scene has no context (uninitialized, moved-from, or context destroyed); returning empty.");
    return {};
  }
  return botsCtx->FindSensorsByType(typeName, /*scope*/ nullptr);
}

DynamicArray<ActuatorHandle> BotScene::FindActuatorsByName(std::string_view name) const {
  RoboticsContext* const botsCtx = GetRoboticsContext();
  if (botsCtx == nullptr) {
    MOCHI_LOG_WARNING(
        "BotScene::FindActuatorsByName: scene has no context (uninitialized, moved-from, or context destroyed); returning empty.");
    return {};
  }
  return botsCtx->FindActuatorsByName(name, /*scope*/ nullptr);
}

DynamicArray<ActuatorHandle> BotScene::FindActuatorsByType(std::string_view typeName) const {
  RoboticsContext* const botsCtx = GetRoboticsContext();
  if (botsCtx == nullptr) {
    MOCHI_LOG_WARNING(
        "BotScene::FindActuatorsByType: scene has no context (uninitialized, moved-from, or context destroyed); returning empty.");
    return {};
  }
  return botsCtx->FindActuatorsByType(typeName, /*scope*/ nullptr);
}

/* ---------------------------------------------------------------------------
 * Controller factory: create and configure any controller from a ControllerEntry.
 * --------------------------------------------------------------------------- */

/* Create and configure a controller from a scene entry. Fully generic: @c ctrlEntry.type is the
 * registered controller name (legacy scene spellings resolve via the backwards-compat aliases in
 * RegisterLegacyControllerTypes), and each controller configures itself from its params and its
 * verbatim JSON object (@c ctrlEntry.initArgs) through ControllerBase::ConfigureFromSceneEntry --
 * so there is no per-type code here. Unknown types (e.g. an internal-only type referenced by a
 * scene loaded in a public build) are skipped with a warning; the caller then records no handle, so
 * GetControllerHandle later returns an invalid handle for it. */
static ControllerHandle CreateControllerFromEntry(
    ControllerEntry const& ctrlEntry,
    BotPrefab const* prefab,
    Actor* actor,
    RoboticsContext* botsCtx,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});

  std::string_view const type(ctrlEntry.type.c_str(), ctrlEntry.type.size());

  if (!botsCtx->IsControllerTypeRegistered(type)) {
    MOCHI_LOG_WARNING(
        "BotScene: unknown controller type '%s'; skipping", std::string(ctrlEntry.type).c_str());
    return {};
  }

  /* Pass the bot's prefab so model-derived config (e.g. the per-joint effort limits consumed by
   * BASIC_OSC_PD's torque normalization) is harvested; the actor alone leaves them unpopulated.
   * The scene-authored instance name is applied here so the controller is findable via
   * FindControllersByName from the moment it exists. */
  ControllerHandle const handle = botsCtx->CreateController(
      type, prefab, actor, std::string_view(ctrlEntry.name.c_str(), ctrlEntry.name.size()), error);
  MOCHI_ERROR_RETURN(error, {});
  bool committed = false;
  auto guard = Defer([&]() {
    if (!committed) {
      botsCtx->DestroyController(handle);
    }
  });

  auto* const controller = botsCtx->GetController(handle);
  MOCHI_ERROR_IF(controller == nullptr, error, "Created controller handle is invalid");
  MOCHI_ERROR_RETURN(error, {});
  controller->ConfigureFromSceneEntry(
      std::string_view(ctrlEntry.params.c_str(), ctrlEntry.params.size()),
      std::string_view(ctrlEntry.initArgs.c_str(), ctrlEntry.initArgs.size()),
      error);
  MOCHI_ERROR_RETURN(error, {});

  committed = true;
  return handle;
}

static BotEntry const* FindBotEntry(BotScenePrefab const& params, std::string_view name) {
  for (auto const& botEntry : params.bots) {
    if (std::string_view(botEntry.name.c_str(), botEntry.name.size()) == name) {
      return &botEntry;
    }
  }
  return nullptr;
}

/* Instantiate one declared bot and its controllers, registering both under the bot's declared name.
 * A bot registered before a later step fails stays registered, so the caller decides whether to
 * unwind it. */
static Bot* InstantiateBot(
    BotScene::Impl& impl,
    Scene* scene,
    BotEntry const& botEntry,
    RoboticsContext* botsCtx,
    Error& error,
    BotPrefabLookup const& findBotPrefab = {},
    IBotLoader const* loader = nullptr) {
  MOCHI_ERROR_RETURN(error, nullptr);

  std::string_view const botPath(botEntry.path.c_str(), botEntry.path.size());
  BotPrefab const* const loadedPrefab = findBotPrefab ? findBotPrefab(botPath) : nullptr;
  BotPrefab botPrefab =
      loadedPrefab != nullptr ? *loadedPrefab : LoadBotPrefabFromFile(botPath, error);
  MOCHI_ERROR_RETURN(error, nullptr);

  // The bot is placed at parentFromBot relative to the scene root (identity scene root == world
  // for now, until scene spawn transforms are supported).
  botPrefab.worldFromRoot = botEntry.parentFromBot;
  if (!botEntry.initialPose.empty()) {
    botPrefab.defaultPose = botEntry.initialPose;
  }

  Bot* bot = loader != nullptr ? botsCtx->CreateBot(scene, botPrefab, *loader, error)
                               : CreateBot(scene, botPrefab, botsCtx, error);
  MOCHI_ERROR_RETURN(error, nullptr);
  RecordBotRenderModels(*bot, impl.renderModels, impl.skinRenderModels);

  impl.bots[std::string(botEntry.name)] = bot;
  Actor* actor = bot->GetArticulatedActor();

  for (auto const& ctrlEntry : botEntry.controllers) {
    std::string controllerKey = std::string(botEntry.name) + "/" + std::string(ctrlEntry.name);
    ControllerHandle handle =
        CreateControllerFromEntry(ctrlEntry, &bot->GetBotPrefab(), actor, botsCtx, error);
    MOCHI_ERROR_RETURN(error, nullptr);
    if (handle.IsValid()) {
      impl.controllers[controllerKey] = handle;
    }
  }
  return bot;
}

Bot* BotScene::InitializeBot(std::string_view name, Error& error) {
  MOCHI_ERROR_RETURN(error, nullptr);
  MOCHI_ERROR_IF(!impl_, error, "BotScene is not initialized");
  MOCHI_ERROR_RETURN(error, nullptr);

  BotEntry const* const botEntry = FindBotEntry(impl_->params, name);
  if (botEntry == nullptr) {
    MOCHI_LOG_ERROR("BotScene: no bot named '%s' is declared", std::string(name).c_str());
  }
  MOCHI_ERROR_IF(botEntry == nullptr, error, "BotScene: bot is not declared in the scene");
  MOCHI_ERROR_RETURN(error, nullptr);
  MOCHI_ERROR_IF(impl_->bots.contains(std::string(name)), error, "BotScene: bot is already loaded");
  MOCHI_ERROR_RETURN(error, nullptr);
  MOCHI_ERROR_IF(!impl_->botsCtx, error, "BotScene: bots context is not initialized");
  MOCHI_ERROR_RETURN(error, nullptr);
  Scene* const scene = impl_->ResolveScene();
  MOCHI_ERROR_IF(scene == nullptr, error, "BotScene: scene is not available");
  MOCHI_ERROR_RETURN(error, nullptr);

  Bot* const bot = InstantiateBot(*impl_, scene, *botEntry, impl_->botsCtx, error);
  if (!error.IsOK()) {
    // A bot left half-built -- its articulated actor without all its controllers -- would be
    // simulated but not drivable, so take back whatever was created.
    if (impl_->bots.contains(std::string(name))) {
      RemoveBot(name, ErrorLog{});
    }
    return nullptr;
  }
  return bot;
}

// ---------------------------------------------------------------------------
// Shared population: add the base scene + bots + controllers into `scene`.
// `impl.params` must already be set (with absolute paths). Does not create or
// destroy `scene`.
// ---------------------------------------------------------------------------

static void PopulateBotScene(
    BotScene::Impl& impl,
    Scene* scene,
    std::string const& baseRootPath,
    RoboticsContext* botsCtx,
    Error& error,
    BotPrefabLookup const& findBotPrefab = {},
    IBotLoader const* loader = nullptr,
    std::optional<Span<std::string_view const>> botsToInitialize = std::nullopt) {
  MOCHI_ERROR_RETURN(error);
  BotScenePrefab const& params = impl.params;

  // The scene may be caller-owned and outlive a failed load, so a failure at any step restores
  // what the base scene's settings overwrote.
  Real3 const gravityBefore = scene->GetGravity();
  SolverParams const solverBefore = scene->GetSolverParams();
  auto restoreSettings = Defer([&]() {
    if (!error.IsOK()) {
      scene->SetGravity(gravityBefore);
      scene->SetSolverParams(solverBefore, ErrorLog{});
    }
  });

  /* 1. Load the base_scene prefab into the scene and capture its actor handles. A scene under
        construction may not have a base scene yet, in which case nothing is loaded. */
  impl.baseSceneActors.clear();
  impl.renderModels.clear();
  impl.skinRenderModels.clear();
  if (!params.scene.baseScene.empty()) {
    prefab::PrefabParams prefabParams;
    prefabParams.name = DynamicString("scene");
    prefabParams.applySceneSettings = true;
    // Loaded here rather than by the path overload of AddToScene so the render models can be read
    // from the same prefab the actors were created from.
    prefab::ScenePrefab const basePrefab = prefab::LoadFromFile(
        std::string_view(params.scene.baseScene.c_str(), params.scene.baseScene.size()),
        baseRootPath,
        scene->GetContext(),
        error);
    auto const result = prefab::AddToScene(basePrefab, scene, prefabParams, error);
    if (!error.IsOK()) {
      DestroyAddedToScene(scene, result);
      return;
    }
    for (auto* actor : result.actors) {
      impl.baseSceneActors.push_back(actor->GetHandle());
    }
    DynamicArray<ActorHandle> recorded;
    PrefabRenderModelRecorder(
        baseRootPath, MakeConstSpan(result.actors), impl.renderModels, recorded)
        .Record(basePrefab);
  }

  /* 2. Register spawnable prefab paths for lazy loading. */
  for (auto const& prefabEntry : params.scene.spawnablePrefabs) {
    impl.spawnablePrefabPaths[std::string(prefabEntry.name)] =
        std::string(prefabEntry.path.c_str(), prefabEntry.path.size());
  }

  /* 3. Load and instantiate each bot the caller asked for, then its controllers. */
  for (auto const& botEntry : params.bots) {
    std::string_view const botName(botEntry.name.c_str(), botEntry.name.size());
    if (botsToInitialize && !Contains(*botsToInitialize, botName)) {
      continue;
    }
    InstantiateBot(impl, scene, botEntry, botsCtx, error, findBotPrefab, loader);
    MOCHI_ERROR_RETURN(error);
  }

  /* 4. Instantiate the scene-level sensors. These are declared on the scene rather than on a bot,
   * so they are created with no link actor: they belong to no bot, outlive any individual one, and
   * are posed relative to the scene root. An unregistered type is skipped with a warning, matching
   * how CreateControllerFromEntry treats an unknown controller type, so a scene authored against a
   * build that has extra sensor types still loads. */
  for (auto const& sensorEntry : params.sensors) {
    std::string_view const type(sensorEntry.type.c_str(), sensorEntry.type.size());
    if (!botsCtx->IsSensorTypeRegistered(type)) {
      MOCHI_LOG_WARNING(
          "BotScene: unknown sensor type '%s'; skipping", std::string(sensorEntry.type).c_str());
      continue;
    }
    SensorHandle const handle = botsCtx->CreateSensor(
        type,
        /*linkActor*/ nullptr,
        std::string_view(sensorEntry.name.c_str(), sensorEntry.name.size()),
        std::string_view(sensorEntry.params.c_str(), sensorEntry.params.size()),
        error);
    MOCHI_ERROR_RETURN(error);
    if (SensorBase* const sensor = botsCtx->GetSensor(handle); sensor != nullptr) {
      sensor->SetParentFromSensor(sensorEntry.parentFromSensor);
    }
    impl.sceneSensors[std::string(sensorEntry.name)] = handle;
  }
}

/* Attach `impl` to the caller-owned `scene` and populate it from the validated, path-resolved
 * `prefab`, instantiating every declared bot unless `botsToInitialize` names the ones to. */
static void PopulateCallerOwnedBotScene(
    BotScene::Impl& impl,
    Scene* scene,
    BotScenePrefab const& prefab,
    std::string const& rootPath,
    Context* context,
    RoboticsContext* botsCtx,
    Error& error,
    BotPrefabLookup const& findBotPrefab,
    IBotLoader const* loader,
    std::optional<Span<std::string_view const>> botsToInitialize) {
  impl.context = context;
  impl.botsCtx = botsCtx;
  impl.onDestroyHandle =
      botsCtx->RegisterOnDestroy([implPtr = &impl]() { implPtr->OnRoboticsContextDestroyed(); });
  impl.sceneHandle = scene->GetHandle();
  impl.ownsScene = false;
  // A file-loaded prefab already carries each controller's initArgs (captured from the raw scene
  // JSON in LoadBotScenePrefabFromFile). A prefab built directly in memory must set ControllerEntry
  // initArgs itself for controllers that need them (see ControllerBase::ConfigureFromSceneEntry).
  impl.params = prefab;

  PopulateBotScene(impl, scene, rootPath, botsCtx, error, findBotPrefab, loader, botsToInitialize);
  MOCHI_ERROR_RETURN(error);
  impl.RegisterTerminationCallback(*scene);
}

// ---------------------------------------------------------------------------
// LoadBotScene
// ---------------------------------------------------------------------------

BotScene superdex::robotics::LoadBotScene(
    std::string_view path,
    Context* context,
    RoboticsContext* botsCtx,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  MOCHI_ERROR_IF(context == nullptr, error, "LoadBotScene: context must not be null");
  MOCHI_ERROR_RETURN(error, {});
  MOCHI_ERROR_IF(botsCtx == nullptr, error, "LoadBotScene: botsCtx must not be null");
  MOCHI_ERROR_RETURN(error, {});

  /* 1. Load and validate the .superdex_scene file. */
  std::string const scenePath = ResolveBotSceneFile(path, error);
  MOCHI_ERROR_RETURN(error, {});
  BotScenePrefab params = LoadBotScenePrefabFromFile(scenePath, error);
  MOCHI_ERROR_RETURN(error, {});

  /* 2. Resolve the base scene root and create an owned scene. */
  std::string rootPath = GetBotSceneBaseRoot(
      scenePath, std::string_view(params.scene.baseScene.c_str(), params.scene.baseScene.size()));

  std::string sceneName = params.metadata.name.empty()
      ? "bot_scene"
      : std::string(params.metadata.name.c_str(), params.metadata.name.size());

  BotScene botScene;
  botScene.impl_->context = context;
  botScene.impl_->botsCtx = botsCtx;
  botScene.impl_->onDestroyHandle = botsCtx->RegisterOnDestroy(
      [impl = botScene.impl_.get()]() { impl->OnRoboticsContextDestroyed(); });
  Scene* const createdScene = context->CreateScene(sceneName);
  botScene.impl_->sceneHandle = createdScene->GetHandle();
  botScene.impl_->ownsScene = true;
  botScene.impl_->params = std::move(params);

  PopulateBotScene(*botScene.impl_, createdScene, rootPath, botsCtx, error);
  MOCHI_ERROR_RETURN(error, {});
  botScene.impl_->RegisterTerminationCallback(*createdScene);
  return botScene;
}

BotScene superdex::robotics::LoadBotScene(
    Scene* scene,
    BotScenePrefab const& prefab,
    std::string_view rootPath,
    Context* context,
    RoboticsContext* botsCtx,
    Error& error) {
  return LoadBotScene(scene, prefab, rootPath, context, botsCtx, error, {}, nullptr);
}

BotScene superdex::robotics::LoadBotScene(
    Scene* scene,
    BotScenePrefab const& prefab,
    std::string_view rootPath,
    Context* context,
    RoboticsContext* botsCtx,
    Error& error,
    BotPrefabLookup const& findBotPrefab,
    IBotLoader const* loader) {
  MOCHI_ERROR_RETURN(error, {});
  MOCHI_ERROR_IF(scene == nullptr, error, "LoadBotScene: scene must not be null");
  MOCHI_ERROR_RETURN(error, {});
  MOCHI_ERROR_IF(context == nullptr, error, "LoadBotScene: context must not be null");
  MOCHI_ERROR_RETURN(error, {});
  MOCHI_ERROR_IF(botsCtx == nullptr, error, "LoadBotScene: botsCtx must not be null");
  MOCHI_ERROR_RETURN(error, {});

  /* The prefab is already loaded and path-resolved (e.g. by an asset); validate it. */
  ValidateBotScenePrefab(prefab, error);
  MOCHI_ERROR_RETURN(error, {});

  BotScene botScene;
  PopulateCallerOwnedBotScene(
      *botScene.impl_,
      scene,
      prefab,
      std::string(rootPath),
      context,
      botsCtx,
      error,
      findBotPrefab,
      loader,
      std::nullopt);
  MOCHI_ERROR_RETURN(error, {});
  return botScene;
}

BotScene superdex::robotics::LoadBotScene(
    Scene* scene,
    std::string_view path,
    Span<std::string_view const> botsToInitialize,
    Context* context,
    RoboticsContext* botsCtx,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  MOCHI_ERROR_IF(scene == nullptr, error, "LoadBotScene: scene must not be null");
  MOCHI_ERROR_RETURN(error, {});
  MOCHI_ERROR_IF(context == nullptr, error, "LoadBotScene: context must not be null");
  MOCHI_ERROR_RETURN(error, {});
  MOCHI_ERROR_IF(botsCtx == nullptr, error, "LoadBotScene: botsCtx must not be null");
  MOCHI_ERROR_RETURN(error, {});

  std::string const scenePath = ResolveBotSceneFile(path, error);
  MOCHI_ERROR_RETURN(error, {});
  BotScenePrefab const prefab = LoadBotScenePrefabFromFile(scenePath, error);
  MOCHI_ERROR_RETURN(error, {});
  for (std::string_view const name : botsToInitialize) {
    if (FindBotEntry(prefab, name) == nullptr) {
      MOCHI_LOG_ERROR(
          "LoadBotScene: botsToInitialize names '%s', which the scene does not declare",
          std::string(name).c_str());
      MOCHI_ERROR_SET(
          error, "LoadBotScene: botsToInitialize names a bot the scene does not declare");
      return {};
    }
  }

  std::string const rootPath = GetBotSceneBaseRoot(
      scenePath, std::string_view(prefab.scene.baseScene.c_str(), prefab.scene.baseScene.size()));

  BotScene botScene;
  PopulateCallerOwnedBotScene(
      *botScene.impl_,
      scene,
      prefab,
      rootPath,
      context,
      botsCtx,
      error,
      {},
      nullptr,
      botsToInitialize);
  MOCHI_ERROR_RETURN(error, {});
  return botScene;
}

BotScene superdex::robotics::LoadBotScene(
    Scene* scene,
    std::string_view path,
    Context* context,
    RoboticsContext* botsCtx,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  MOCHI_ERROR_IF(scene == nullptr, error, "LoadBotScene: scene must not be null");
  MOCHI_ERROR_RETURN(error, {});
  MOCHI_ERROR_IF(context == nullptr, error, "LoadBotScene: context must not be null");
  MOCHI_ERROR_RETURN(error, {});
  MOCHI_ERROR_IF(botsCtx == nullptr, error, "LoadBotScene: botsCtx must not be null");
  MOCHI_ERROR_RETURN(error, {});

  // Load and validate through LoadBotScenePrefabFromFile, which also path-resolves the prefab
  // entries. An archive loads from, and takes its root from, its extracted copy.
  std::string const scenePath = ResolveBotSceneFile(path, error);
  MOCHI_ERROR_RETURN(error, {});
  BotScenePrefab prefab = LoadBotScenePrefabFromFile(scenePath, error);
  MOCHI_ERROR_RETURN(error, {});

  std::string rootPath = GetBotSceneBaseRoot(
      scenePath, std::string_view(prefab.scene.baseScene.c_str(), prefab.scene.baseScene.size()));

  return LoadBotScene(scene, prefab, rootPath, context, botsCtx, error);
}
