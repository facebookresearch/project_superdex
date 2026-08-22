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

#include "meshing/processing_modifiers/processing_serialization.h"

#include "meshing/processing_modifiers/unknown_placeholder.h"

#include <picojson/picojson.h>

#include <superdex_robotics/utils/file_utils.h> // MakePathRelative / MakePathAbsolute

#include <mochi_core/utils/dynamic_string.h>
#include <mochi_core/utils/error.h>
#include <mochi_core/utils/log.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace superdex::studio {

namespace {
constexpr int kProcessingPipelineVersion = 1;

// A pipeline file and its referenced models form a local cluster: the file lives in the bot's
// intermediates/ folder, its models in sibling folders (render/, cad/, mochi/) under the same bot
// root. Allowing a file-relative path to ascend this many parent directories lets those sibling
// references (e.g. ../render/foo.glb) survive relocating the whole cluster as a unit, without
// depending on a .superdex_root marker. Two levels reaches intermediates/ -> bot root -> a sibling
// folder. Beyond this the path falls back to // / @tag / absolute (see UnresolveBotPath). Bots keep
// the strict descendant-only default (0). Raise this if the on-disk layout ever nests deeper.
constexpr int kPipelinePathMaxParentDepth = 2;

// The fixed per-entry fields (reflected); each entry's "properties" is stitched in separately.
struct ProcessingEntryHeader {
  mochi::DynamicString modifier;
  mochi::DynamicString method;
  bool enabled = true;
  bool collapsed = true;

  MOCHI_STRUCT_BEGIN(superdex::studio::ProcessingEntryHeader)
  MOCHI_FIELD(modifier)
  MOCHI_FIELD(method)
  MOCHI_FIELD(enabled)
  MOCHI_FIELD(collapsed)
  MOCHI_STRUCT_END()
};

// Rewrite @p path to its bot-relative form (relative to @p baseFile's directory) for on-disk
// storage. A path that cannot be relativized -- e.g. an export target outside the bot root -- is
// left absolute with a warning, so one stray path never aborts the whole save. On failure
// MakePathRelative leaves @p path untouched.
void RelativizePath(mochi::DynamicString& path, std::filesystem::path const& baseFile) {
  if (path.empty() || baseFile.empty()) {
    return;
  }
  mochi::Error error;
  superdex::robotics::MakePathRelative(path, baseFile, kPipelinePathMaxParentDepth, error);
  if (!error.IsOK()) {
    MOCHI_LOG_WARNING(
        "Could not store processing-pipeline path '%s' relative to '%s'; keeping it absolute.",
        path.c_str(),
        baseFile.generic_string().c_str());
  }
}

// Resolve @p path (relative, as stored on disk) back to an absolute path against @p baseFile's
// directory for in-memory use. A path that cannot be resolved is left as-is with a warning. On
// failure MakePathAbsolute leaves @p path untouched.
void AbsolutizePath(mochi::DynamicString& path, std::filesystem::path const& baseFile) {
  if (path.empty() || baseFile.empty()) {
    return;
  }
  mochi::Error error;
  superdex::robotics::MakePathAbsolute(path, baseFile, kPipelinePathMaxParentDepth, error);
  if (!error.IsOK()) {
    MOCHI_LOG_WARNING(
        "Could not resolve processing-pipeline path '%s' against '%s'; leaving it unchanged.",
        path.c_str(),
        baseFile.generic_string().c_str());
  }
}

// Applies @p transform (RelativizePath / AbsolutizePath) to the string stored at @p key in the
// reflected props object @p props, leaving non-string / missing values alone.
template <typename Transform>
void TransformJsonPathField(
    picojson::value& props,
    std::string_view key,
    std::filesystem::path const& baseFile,
    Transform&& transform) {
  if (!props.is<picojson::object>()) {
    return;
  }
  picojson::object& obj = props.get<picojson::object>();
  auto const it = obj.find(std::string(key));
  if (it == obj.end() || !it->second.is<std::string>()) {
    return;
  }
  std::string const& value = it->second.get<std::string>();
  mochi::DynamicString path(value.data(), value.size());
  transform(path, baseFile);
  it->second = picojson::value(std::string(path.data(), path.size()));
}
} // namespace

std::string SerializeProcessingPipeline(
    std::vector<std::unique_ptr<MeshProcessingModifier>> const& modifiers,
    ProcessingEditorState const& editorState,
    std::vector<ReferenceModelState> const& referenceModels,
    std::filesystem::path const& baseFile) {
  picojson::object root;
  root["version"] = picojson::value(static_cast<double>(kProcessingPipelineVersion));

  picojson::value editorVal;
  SReflect::ToJsonValue(editorState, editorVal);
  root["editorState"] = editorVal;

  // Viewer-only reference models (see ReferenceModelState). Serialized manually as an array of
  // reflected objects, mirroring the modifiers array below. Each model's path is stored relative to
  // the pipeline file so the JSON stays portable.
  picojson::array refs;
  refs.reserve(referenceModels.size());
  for (ReferenceModelState const& ref : referenceModels) {
    ReferenceModelState relative = ref;
    RelativizePath(relative.path, baseFile);
    picojson::value refVal;
    SReflect::ToJsonValue(relative, refVal);
    refs.push_back(std::move(refVal));
  }
  root["referenceModels"] = picojson::value(refs);

  picojson::array entries;
  entries.reserve(modifiers.size());
  for (auto const& modifier : modifiers) {
    ProcessingEntryHeader entry;
    entry.modifier = modifier->DisplayName();
    entry.method = modifier->ActiveMethodName();
    entry.enabled = modifier->enabled;
    entry.collapsed = modifier->collapsed;

    picojson::value entryVal;
    SReflect::ToJsonValue(entry, entryVal);
    picojson::value propsVal;
    modifier->ActiveMethod().SerializeProps(propsVal);
    // Store this method's file-path props relative to the pipeline file (no absolute paths on
    // disk).
    for (std::string_view const key : modifier->ActiveMethod().PathPropKeys()) {
      TransformJsonPathField(propsVal, key, baseFile, RelativizePath);
    }
    entryVal.get<picojson::object>()["properties"] = propsVal;
    entries.push_back(std::move(entryVal));
  }
  root["modifiers"] = picojson::value(entries);

  return picojson::value(root).serialize(/*prettify=*/true);
}

bool WriteProcessingPipelineFile(
    std::string const& path,
    std::string const& text,
    mochi::Error& error) {
  std::error_code ec;
  std::filesystem::path const fsPath(path);
  if (fsPath.has_parent_path()) {
    std::filesystem::create_directories(fsPath.parent_path(), ec);
    if (ec) {
      MOCHI_ERROR_SET(error, "Failed to create the processing-pipeline directory.");
      return false;
    }
  }
  std::filesystem::path tmpPath = fsPath;
  tmpPath += ".tmp";
  std::filesystem::remove(tmpPath, ec);
  if (ec) {
    MOCHI_ERROR_SET(error, "Failed to remove the temporary processing-pipeline file.");
    return false;
  }

  std::ofstream ofs(tmpPath, std::ios::binary | std::ios::trunc);
  if (!ofs) {
    MOCHI_ERROR_SET(error, "Failed to open the processing-pipeline file for writing.");
    return false;
  }
  ofs << text;
  if (!ofs.good()) {
    std::error_code cleanupEc;
    std::filesystem::remove(tmpPath, cleanupEc);
    MOCHI_ERROR_SET(error, "Failed to write the processing-pipeline file.");
    return false;
  }
  ofs.close();
  if (ofs.fail()) {
    std::error_code cleanupEc;
    std::filesystem::remove(tmpPath, cleanupEc);
    MOCHI_ERROR_SET(error, "Failed to write the processing-pipeline file.");
    return false;
  }
  std::filesystem::rename(tmpPath, fsPath, ec);
  if (ec) {
    std::error_code cleanupEc;
    std::filesystem::remove(tmpPath, cleanupEc);
    MOCHI_ERROR_SET(error, "Failed to replace the processing-pipeline file.");
    return false;
  }
  return true;
}

bool LoadProcessingPipeline(
    std::string const& path,
    std::filesystem::path const& baseFile,
    LoadedPipeline& out,
    mochi::Error& error) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) {
    MOCHI_ERROR_SET(error, "Processing-pipeline file could not be opened.");
    return false;
  }
  std::stringstream ss;
  ss << ifs.rdbuf();
  std::string const text = ss.str();

  picojson::value rootVal;
  std::string parseErr;
  picojson::parse(rootVal, text.begin(), text.end(), &parseErr);
  if (!parseErr.empty() || !rootVal.is<picojson::object>()) {
    MOCHI_ERROR_SET(error, "Processing-pipeline file is not valid JSON.");
    return false;
  }
  picojson::object const& root = rootVal.get<picojson::object>();

  // Optional editor state.
  auto const editorIt = root.find("editorState");
  if (editorIt != root.end() && editorIt->second.is<picojson::object>()) {
    int issues = 0;
    SReflect::FromJsonValue(
        out.editorState, editorIt->second, SReflect::DeserializeFlags::None, issues);
    out.hasEditorState = true;
  }

  // Viewer-only reference models (optional).
  auto const refsIt = root.find("referenceModels");
  if (refsIt != root.end() && refsIt->second.is<picojson::array>()) {
    for (picojson::value const& refVal : refsIt->second.get<picojson::array>()) {
      if (!refVal.is<picojson::object>()) {
        continue;
      }
      int issues = 0;
      ReferenceModelState ref;
      SReflect::FromJsonValue(ref, refVal, SReflect::DeserializeFlags::None, issues);
      AbsolutizePath(ref.path, baseFile);
      out.referenceModels.push_back(std::move(ref));
    }
  }

  // Modifiers.
  auto const modifiersIt = root.find("modifiers");
  if (modifiersIt != root.end() && modifiersIt->second.is<picojson::array>()) {
    for (picojson::value const& entryVal : modifiersIt->second.get<picojson::array>()) {
      if (!entryVal.is<picojson::object>()) {
        MOCHI_ERROR_SET(error, "Processing-pipeline modifier entry is not an object.");
        return false;
      }
      int issues = 0;
      ProcessingEntryHeader entry;
      SReflect::FromJsonValue(entry, entryVal, SReflect::DeserializeFlags::None, issues);

      picojson::object const& entryObj = entryVal.get<picojson::object>();
      auto const propsIt = entryObj.find("properties");
      bool const hasProps = propsIt != entryObj.end();

      // entry.modifier / entry.method are mochi::DynamicString (custom allocator); copy their bytes
      // into std::string explicitly (no implicit cross-allocator conversion exists).
      std::string const modName{entry.modifier.data(), entry.modifier.size()};
      std::string const methodName{entry.method.data(), entry.method.size()};

      std::unique_ptr<MeshProcessingModifier> modifier = MakeProcessingModifier(modName);
      if (modifier && modifier->SelectMethodByName(methodName)) {
        if (hasProps) {
          // Resolve this method's file-path props to absolute before applying them, on a copy so
          // the on-disk relative form is left intact.
          picojson::value props = propsIt->second;
          for (std::string_view const key : modifier->ActiveMethod().PathPropKeys()) {
            TransformJsonPathField(props, key, baseFile, AbsolutizePath);
          }
          modifier->ActiveMethod().DeserializeProps(props);
        }
      } else {
        // Unknown modifier or method: keep the data losslessly in a passthrough placeholder.
        std::string const propsJson =
            hasProps ? propsIt->second.serialize(/*prettify=*/false) : "{}";
        modifier = MakeUnknownPlaceholderModifier(modName, methodName, propsJson);
      }
      modifier->enabled = entry.enabled;
      modifier->collapsed = entry.collapsed;
      out.modifiers.push_back(std::move(modifier));
    }
  }
  return true;
}

} // namespace superdex::studio
