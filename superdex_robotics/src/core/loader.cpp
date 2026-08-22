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

#include <superdex_robotics/core/loader.h>
#include <superdex_robotics/utils/bot_utils.h>
#include <superdex_robotics/utils/file_utils.h>

#include <mochi_core/utils/json_utils.h>

using namespace mochi;
using namespace superdex::robotics;

#if !MOCHI_USE_REFLECTION
#error Reflection is required for the mochi_bots library internals. Please define `MOCHI_USE_REFLECTION=1` in your build system.
#endif

namespace {

// A mod bot is distinguished by the 'base' key, which BotPrefab does not have. Key presence --
// not a non-empty value -- is the discriminator: a newly created mod bot has no base yet.
bool HasModBotBaseKey(std::string_view path, Error& error) {
  MOCHI_ERROR_RETURN(error, false);

  picojson::value const root = ParseJsonFromFile(path, error);
  MOCHI_ERROR_RETURN(error, false);
  MOCHI_ERROR_IF(!root.is<picojson::object>(), error, "Failed to load .superdex_bot (JSON) file");
  MOCHI_ERROR_RETURN(error, false);

  auto const& rootObj = root.get<picojson::object>();
  return rootObj.find("base") != rootObj.end();
}

} // namespace

BotFileType FileBotLoader::GetBotFileType(std::string_view path, Error& error) const {
  MOCHI_ERROR_RETURN(error, {});
  MOCHI_ERROR_IF(!IsBotPath(path), error, "Unsupported file format; expected .superdex_bot");
  MOCHI_ERROR_RETURN(error, {});

  return HasModBotBaseKey(path, error) ? BotFileType::ModBotPrefab : BotFileType::BotPrefab;
}

BotPrefab FileBotLoader::LoadBotPrefab(std::string_view path, Error& error) const {
  MOCHI_ERROR_RETURN(error, {});
  MOCHI_ERROR_IF(!IsBotPath(path), error, "Unsupported file format; expected .superdex_bot");
  MOCHI_ERROR_RETURN(error, {});

  BotPrefab outBotPrefab;
  int numIssues = 0;
  std::string pathStr(path);
  bool success = SReflect::LoadFromJsonFile(
      outBotPrefab, pathStr.c_str(), SReflect::DeserializeFlags::Default, numIssues);
  if (numIssues > 0) {
    MOCHI_LOG_WARNING("Detected %d issues while loading: %s", numIssues, pathStr.c_str());
  }
  MOCHI_ERROR_IF(!success, error, "Failed to load BotPrefab .mochi_bot (JSON) file");
  MOCHI_ERROR_RETURN(error, {});

  // LEGACY: old files omitted BotJointPrefab::type (which used to default to Hard, now Invalid);
  // restore the old default. TODO: Delete this after bots have been resaved.
  ApplyLegacyBotJointTypes(outBotPrefab);

  // LEGACY: migrate old sensor typeName/paramsFile keys to type/params before path resolution.
  ApplyLegacyBotSensorFields(outBotPrefab);

  RebuildBotData(outBotPrefab, error);

  MakePathsAbsolute(outBotPrefab, path, error);
  MOCHI_ERROR_RETURN(error, {});
  return outBotPrefab;
}

ModBotPrefab FileBotLoader::LoadModBotPrefab(std::string_view path, Error& error) const {
  MOCHI_ERROR_RETURN(error, {});
  MOCHI_ERROR_IF(!IsBotPath(path), error, "Unsupported file format; expected .superdex_bot");
  MOCHI_ERROR_RETURN(error, {});

  // Reject plain BotPrefab files: deserializing one into a ModBotPrefab would succeed while
  // silently discarding every field ModBotPrefab does not declare (links, joints, colliders).
  bool const isModBot = HasModBotBaseKey(path, error);
  MOCHI_ERROR_RETURN(error, {});
  MOCHI_ERROR_IF(!isModBot, error, "File is not a ModBotPrefab; no 'base' key");
  MOCHI_ERROR_RETURN(error, {});

  ModBotPrefab outModBotPrefab;
  int numIssues = 0;
  std::string pathStr(path);
  bool const success = SReflect::LoadFromJsonFile(
      outModBotPrefab, pathStr.c_str(), SReflect::DeserializeFlags::Default, numIssues);
  if (numIssues > 0) {
    MOCHI_LOG_WARNING("Detected %d issues while loading: %s", numIssues, pathStr.c_str());
  }
  MOCHI_ERROR_IF(!success, error, "Failed to load ModBotPrefab from .mochi_bot (JSON) file");
  MOCHI_ERROR_RETURN(error, {});

  // LEGACY: apply the same joint-type and sensor-field fixups LoadBotPrefab applies to base bots.
  // Mod bots carry inline connecting joints (AttachLink/AttachBot) and inline sensors
  // (AttachLink/ReplaceLink links) that BuildBot merges into the assembled bot as-is.
  ApplyLegacyBotJointTypes(outModBotPrefab);
  ApplyLegacyBotSensorFields(outModBotPrefab);

  MakePathsAbsolute(outModBotPrefab, path, error);
  MOCHI_ERROR_RETURN(error, {});
  return outModBotPrefab;
}

ShapeHandle FileBotLoader::LoadShape(
    std::string_view path,
    Real3 const& bakeScale,
    TransformRT const& bakeTransform,
    Context* context,
    Error& error) const {
  MOCHI_ERROR_RETURN(error, {});
  return context->LoadShapeFromFile(path, bakeScale, bakeTransform, error);
}

BotPrefab superdex::robotics::LoadBotPrefab(
    std::string_view path,
    IBotLoader const& loader,
    bool validate,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  BotPrefab outParams;
  auto type = loader.GetBotFileType(path, error);
  if (type == BotFileType::ModBotPrefab) {
    ModBotPrefab modBotPrefab = loader.LoadModBotPrefab(path, error);
    outParams = BuildBot(modBotPrefab, loader, validate, error);
  } else if (type == BotFileType::BotPrefab) {
    outParams = loader.LoadBotPrefab(path, error);
    if (validate) {
      Validate(outParams, nullptr, error);
    }
  } else {
    MOCHI_ERROR_SET(error, "Loader failed to determine the bot file type.");
  }
  return outParams;
}
