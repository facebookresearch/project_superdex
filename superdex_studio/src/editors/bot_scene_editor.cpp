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

#include "editors/bot_scene_editor.h"
#include "app/app.h"
#include "assets/asset_manager.h"
#include "assets/bot_asset.h"
#include "assets/bot_scene_asset.h"
#include "assets/controller_params.h"
#include "assets/mochi_prefab_asset.h"
#include "rendering/measure_tool.h"
#include "ui/imgui_widgets.h"

#include <superdex_robotics/utils/archive_utils.h>
#include <superdex_robotics/utils/bot_utils.h>
#include <superdex_robotics/utils/file_utils.h>

#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/path.h>
#include <mochi_core/utils/reflection.h>
#include <mochi_core/utils/span.h>
#include <mochi_physics/utils/mochi_prefab.h>

#include <imgui_internal.h> // dock node tab bar, to bring the Task Preview tab forward
#include <imguios/fonts/icons_font_awesome5.h>
#include <misc/cpp/imgui_stdlib.h>
#include <picojson/picojson.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <variant>

namespace superdex::studio {

//--------------------------------------------------------------------------------------------------
// AssetEditor
//--------------------------------------------------------------------------------------------------

namespace {

// Sets @p spawn's PoseRandomization class, adding the randomization when it has none; an empty
// class removes it. The entry's other randomization is left as it is.
void SetPoseRandomizationClass(superdex::robotics::SpawnEntry& spawn, std::string_view drClass) {
  auto& randomization = spawn.randomization;
  auto const pose = std::find_if(randomization.begin(), randomization.end(), [](auto const& r) {
    return std::holds_alternative<superdex::robotics::PoseRandomization>(r);
  });
  if (drClass.empty()) {
    if (pose != randomization.end()) {
      randomization.erase(pose);
    }
    return;
  }
  if (pose == randomization.end()) {
    superdex::robotics::PoseRandomization added;
    added.domainRandomizationClass = drClass;
    randomization.emplace_back(std::move(added));
    return;
  }
  std::get<superdex::robotics::PoseRandomization>(*pose).domainRandomizationClass = drClass;
}

// Returns a name of the form "<stem><N>" (N starting at 1) that is not in @p existing. Trailing
// digits of @p base are stripped to form the stem so numbering continues from it. Mirrors
// PrefabEditor::MakeUniquePrefabActorName so new entries get sensible non-conflicting names.
std::string MakeUniqueName(std::string_view base, std::set<std::string> const& existing) {
  std::string stem(base);
  while (!stem.empty() && std::isdigit(static_cast<unsigned char>(stem.back()))) {
    stem.pop_back();
  }
  if (stem.empty()) {
    stem = std::string(base);
  }
  for (int i = 1;; ++i) {
    std::string candidate = stem + std::to_string(i);
    if (existing.count(candidate) == 0) {
      return candidate;
    }
  }
}

std::string ValidateJson(std::string_view text, bool requireObject) {
  picojson::value value;
  std::string json{text};
  std::string error;
  picojson::parse(value, json.begin(), json.end(), &error);
  if (!error.empty()) {
    return error;
  }
  if (requireObject && !value.is<picojson::object>()) {
    return "Expected a JSON object";
  }
  return {};
}

bool IsOscControllerType(std::string_view type) {
  type = CanonicalControllerType(type);
  return type == superdex::robotics::ControllerBasicOscPd::TypeName()
#if MOCHI_INTERNAL
      || type == superdex::robotics::ControllerOscV1::TypeName() ||
      type == superdex::robotics::ControllerOscV2::TypeName()
#endif
      ;
}

bool ParseJsonObject(std::string_view text, picojson::object& object, std::string& error) {
  picojson::value value;
  std::string json{text};
  picojson::parse(value, json.begin(), json.end(), &error);
  if (!error.empty()) {
    return false;
  }
  if (!value.is<picojson::object>()) {
    error = "Expected a JSON object";
    return false;
  }
  object = value.get<picojson::object>();
  return true;
}

std::string GetJsonString(picojson::object const& object, char const* key) {
  auto const it = object.find(key);
  if (it == object.end() || !it->second.is<std::string>()) {
    return {};
  }
  return it->second.get<std::string>();
}

void SetJsonString(
    mochi::DynamicString& json,
    picojson::object& object,
    char const* key,
    std::string const& value) {
  object[key] = picojson::value(value);
  json = picojson::value(object).serialize();
}

bool LinkNameCombo(
    char const* label,
    superdex::robotics::BotPrefab const& prefab,
    std::string& selectedName) {
  bool changed = false;
  char const* preview = selectedName.empty() ? "Select a link" : selectedName.c_str();
  if (ImGui::BeginCombo(label, preview)) {
    for (auto const& link : prefab.links) {
      bool const selected = selectedName == link.name;
      if (ImGui::Selectable(link.name.c_str(), selected)) {
        selectedName = link.name;
        changed = true;
      }
      if (selected) {
        ImGui::SetItemDefaultFocus();
      }
    }
    ImGui::EndCombo();
  }
  return changed;
}

// Picks one of @p names into @p value, which is shown even when it is not one of them.
bool NameCombo(
    char const* label,
    std::vector<std::string> const& names,
    std::string_view preview,
    mochi::DynamicString& value) {
  bool changed = false;
  if (ImGui::BeginCombo(label, std::string(preview).c_str())) {
    for (auto const& name : names) {
      bool const selected = preview == name;
      if (ImGui::Selectable(name.c_str(), selected)) {
        value = mochi::DynamicString{name};
        changed = true;
      }
      if (selected) {
        ImGui::SetItemDefaultFocus();
      }
    }
    ImGui::EndCombo();
  }
  return changed;
}

// Why @p target cannot be stored in @p sceneFile, or empty if it can. Scenes store references only
// as descendants of their folder, `//` paths under their .superdex_root, or `@tag` paths, so a file
// under another root needs a tag in the scene's root.
std::string DescribeReferenceProblem(
    std::filesystem::path const& target,
    std::filesystem::path const& sceneFile) {
  mochi::Error error;
  (void)superdex::robotics::UnresolveBotPath(
      target, sceneFile, superdex::robotics::kBotPathMaxParentDepth, error);
  if (error.IsOK()) {
    return {};
  }
  auto const sceneRoot = superdex::robotics::FindBotsRoot(sceneFile);
  auto const targetRoot = superdex::robotics::FindBotsRoot(target);
  if (!sceneRoot.has_value() || !targetRoot.has_value()) {
    return "This scene cannot save a reference to this file: they are not under a .superdex_root";
  }
  return "This scene cannot save a reference to this file. Add \"@" +
      targetRoot->filename().string() + "\": \"" +
      std::filesystem::relative(*targetRoot, *sceneRoot).generic_string() + "\" to " +
      (*sceneRoot / superdex::robotics::kRootMarkerFile).generic_string();
}

void ShowWarning(std::string_view warning) {
  ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.65f, 0.15f, 1.0f));
  ImGui::TextWrapped(
      "%s %.*s", ICON_FA_EXCLAMATION_TRIANGLE, static_cast<int>(warning.size()), warning.data());
  ImGui::PopStyleColor();
}

// One label per DOF, in the order per-DOF controller params index them.
std::vector<std::string> DofLabels(superdex::robotics::BotPrefab const& botPrefab) {
  std::vector<std::string> labels;
  labels.reserve(botPrefab._dofIndices.size());
  int sphericalAxis = 0;
  for (size_t iDof = 0; iDof < botPrefab._dofIndices.size(); ++iDof) {
    int const iJoint = botPrefab._dofIndices[iDof];
    auto const& joint = botPrefab.joints[iJoint];
    if (joint.type != mochi::ArticulatedJointType::Spherical) {
      labels.emplace_back(joint.name);
      continue;
    }
    // Spherical joints have three consecutive DOFs.
    sphericalAxis = iDof > 0 && botPrefab._dofIndices[iDof - 1] == iJoint ? sphericalAxis + 1 : 0;
    labels.push_back(std::string(joint.name) + " (" + "XYZ"[sphericalAxis] + ")");
  }
  return labels;
}

// Edits normalized params JSON: typed fields for built-in controller types, raw JSON otherwise.
// With @p botPrefab, per-DOF arrays are labeled by joint and can be resized to the bot's DOFs.
bool EditParamsJson(
    std::string_view type,
    std::string& json,
    superdex::robotics::BotPrefab const* botPrefab) {
  bool changed = false;
  bool const builtin = ForEachBuiltinController([&]<typename T>() {
    if (T::TypeName() != CanonicalControllerType(type)) {
      return false;
    }
    typename T::Params params;
    SReflect::FromJsonString(params, json);
    std::vector<std::string> const dofLabels =
        botPrefab != nullptr ? DofLabels(*botPrefab) : std::vector<std::string>{};
    if (!dofLabels.empty()) {
      auto resized = params;
      if (ResizeArrayFields(
              static_cast<SReflect::StructTypeInfo const&>(
                  SReflect::GetTypeInfo<typename T::Params>()),
              &resized,
              dofLabels.size())) {
        if (ImGui::Button("Match Bot Joints")) {
          params = std::move(resized);
          changed = true;
        }
        ImGui::ItemTooltipWrapped(
            "Resize every per-joint array to the bot's joint count, filling new entries with the "
            "last value");
      }
    }
    ImGui::PushReflectionArrayLabels(dofLabels);
    changed |= ImGui::SimpleReflectionStruct(params);
    ImGui::PopReflectionArrayLabels();
    if (changed) {
      json = SReflect::ToJsonString(params, true);
    }
    return true;
  });
  if (!builtin) {
    changed =
        ImGui::InputTextMultiline("Params JSON", &json, ImVec2(-1, ImGui::GetTextLineHeight() * 6));
    std::string normalized;
    std::string error;
    if (!NormalizeParamsJson(type, json, normalized, error)) {
      ShowWarning(error);
    }
  }
  return changed;
}

mochi::Path
GetControllerParamsPath(char const* title, bool isSaveDialog, mochi::Path const& initial) {
  std::array<char const*, 1> const filters{{"*.superdex_controller"}};
#if MOCHI_PLATFORM_MACOS
  // Custom extensions without a registered UTI are disabled when the native filter is set.
  int constexpr numFilters = 0;
#else
  int constexpr numFilters = static_cast<int>(filters.size());
#endif
  return SuperDexStudio::GetFileDialogPath(
      title,
      filters.data(),
      numFilters,
      "SuperDex Controller (*.superdex_controller)",
      isSaveDialog,
      initial);
}

int CountControllersUsingParams(
    superdex::robotics::BotScenePrefab const& scene,
    std::string_view paramsPath) {
  int count = 0;
  for (auto const& bot : scene.bots) {
    for (auto const& controller : bot.controllers) {
      count += std::string_view(controller.params) == paramsPath ? 1 : 0;
    }
  }
  return count;
}

bool ResolveTaskSpawns(
    superdex::robotics::BotTaskPrefab const& task,
    superdex::robotics::BotScenePrefab const& scene,
    std::vector<ResolvedTaskSpawn>& out,
    std::string& error) {
  std::unordered_map<std::string, mochi::Path> prefabPaths;
  for (auto const& prefab : scene.scene.spawnablePrefabs) {
    if (!prefabPaths.emplace(std::string(prefab.name), mochi::Path{std::string(prefab.path)})
             .second) {
      error = "Bot scene declares more than one spawnable prefab named '" +
          std::string(prefab.name) + "'";
      return false;
    }
  }

  std::unordered_map<std::string, int> spawnIndices;
  for (int i = 0; i < mochi::isize(task.resetState); ++i) {
    if (!spawnIndices.emplace(std::string(task.resetState[i].name), i).second) {
      error = "Bot task declares more than one spawn named '" +
          std::string(task.resetState[i].name) + "'";
      return false;
    }
  }

  mochi::TransformRT const worldFromTask =
      superdex::robotics::ComputeWorldFromTask(scene.scene, task);
  out.clear();
  out.resize(task.resetState.size());
  std::vector<int> resolveState(task.resetState.size());
  std::function<bool(int)> resolve = [&](int index) {
    if (resolveState[index] == 2) {
      return true;
    }
    if (resolveState[index] == 1) {
      error = "Bot task contains a cyclic spawn parent chain";
      return false;
    }
    resolveState[index] = 1;

    auto const& spawn = task.resetState[index];
    auto const prefab = prefabPaths.find(std::string(spawn.prefabName));
    if (prefab == prefabPaths.end()) {
      error = "Task spawn '" + std::string(spawn.name) + "' references '" +
          std::string(spawn.prefabName) + "', which is not declared by this bot scene";
      return false;
    }

    mochi::TransformRT worldFromParent = worldFromTask;
    std::string const parent{spawn.parent};
    if (!parent.empty() && parent != superdex::robotics::kTaskRootParentName) {
      auto const parentIt = spawnIndices.find(parent);
      if (parentIt == spawnIndices.end()) {
        error = "Task spawn '" + std::string(spawn.name) + "' references unknown parent '" +
            parent + "'";
        return false;
      }
      if (!resolve(parentIt->second)) {
        return false;
      }
      worldFromParent = out[parentIt->second].worldFromSpawn;
    }

    out[index] = {
        .name = std::string(spawn.name),
        .prefabName = std::string(spawn.prefabName),
        .prefabPath = prefab->second,
        .worldFromSpawn = worldFromParent * spawn.parentFromSpawn,
    };
    resolveState[index] = 2;
    return true;
  };

  for (int i = 0; i < mochi::isize(task.resetState); ++i) {
    if (!resolve(i)) {
      out.clear();
      return false;
    }
  }
  return true;
}

struct ControllerEditResult {
  bool changed = false;
  bool structural = false;
  bool referencesChanged = false;
};

ControllerEditResult ShowControllers(
    superdex::robotics::BotEntry& bot,
    BotAsset const* botAsset,
    SuperDexStudio* studio,
    BotSceneAsset* sceneAsset,
    std::function<std::string const&(std::string_view)> const& referenceProblem) {
  ControllerEditResult result;
  superdex::robotics::BotPrefab const* const botPrefab =
      botAsset != nullptr ? &botAsset->GetBotPrefab() : nullptr;
  auto& controllers = bot.controllers;
  int controllerToDelete = -1;
  int controllerToMoveUp = -1;
  int controllerToMoveDown = -1;

  ImGui::PushID("Controllers");
  for (int i = 0; i < static_cast<int>(controllers.size()); ++i) {
    auto& controller = controllers[i];
    ImGui::PushID(i);
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(1, 0));
    if (ImGui::Button(ICON_FA_TRASH)) {
      controllerToDelete = i;
    }
    ImGui::SameLine();
    ImGui::BeginDisabled(i == 0);
    if (ImGui::Button(ICON_FA_CARET_UP)) {
      controllerToMoveUp = i;
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(i == static_cast<int>(controllers.size()) - 1);
    if (ImGui::Button(ICON_FA_CARET_DOWN)) {
      controllerToMoveDown = i;
    }
    ImGui::EndDisabled();
    ImGui::PopStyleVar();
    ImGui::SameLine();

    std::string const label =
        (controller.name.empty() ? std::string("(unnamed)") : std::string(controller.name)) + " (" +
        (controller.type.empty() ? std::string("no type") : std::string(controller.type)) +
        ")###controller";
    if (ImGui::CollapsingHeader(label.c_str())) {
      int matchingNames = 0;
      for (auto const& candidate : controllers) {
        matchingNames += !controller.name.empty() && candidate.name == controller.name ? 1 : 0;
      }
      bool const nameCollides = matchingNames > 1;
      if (nameCollides) {
        ImGui::PushStyleColor(ImGuiCol_FrameBg, kNameConflictColor);
      }
      result.changed |=
          ImGui::InputText("Name", &controller.name, ImGuiInputTextFlags_CharsNoBlank);
      if (nameCollides) {
        ImGui::PopStyleColor();
        if (ImGui::IsItemHovered()) {
          ImGui::SetTooltip("Another controller already uses this name.");
        }
      }
      if (controller.name.empty()) {
        ShowWarning("Controller name is required");
      }

      std::string const previousType{controller.type};
      char const* presetLabel = "Custom";
      for (auto const type : kBuiltinControllerTypes) {
        if (type == std::string_view(controller.type)) {
          presetLabel = type.data();
          break;
        }
      }
      if (ImGui::BeginCombo("Type Preset", presetLabel)) {
        for (auto const type : kBuiltinControllerTypes) {
          bool const selected = type == std::string_view(controller.type);
          if (ImGui::Selectable(type.data(), selected)) {
            controller.type = type;
            result.changed = true;
          }
          if (selected) {
            ImGui::SetItemDefaultFocus();
          }
        }
        ImGui::EndCombo();
      }
      result.changed |=
          ImGui::InputText("Type", &controller.type, ImGuiInputTextFlags_CharsNoBlank);
      if (controller.type.empty()) {
        ShowWarning("Controller type is required");
      } else if (!studio->GetRoboticsContext()->IsControllerTypeRegistered(
                     std::string_view(controller.type))) {
        ShowWarning("Controller type is not registered in this build");
      }
      // Params belong to a type, so another type starts from its own defaults.
      if (std::string_view(controller.type) != previousType && !controller.params.empty()) {
        controller.params.clear();
        result.referencesChanged = true;
      }

      std::string const type{controller.type};
      std::string const paramsValue{controller.params};
      bool const inlineParams = superdex::robotics::IsInlineJson(paramsValue);
      auto& paramsCache = sceneAsset->GetControllerParams();
      ControllerParamsCache::Entry* const entry =
          paramsValue.empty() || inlineParams ? nullptr : &paramsCache.GetOrLoad(paramsValue, type);
      mochi::Path const sceneDirectory = sceneAsset->GetPath().GetParentPath();

      ImGui::HoverableSeparatorText("Parameters");
      if (paramsValue.empty()) {
        ImGui::TextUnformatted("Defaults");
      } else if (inlineParams) {
        ImGui::TextUnformatted("Unsaved (a file is chosen when the scene is saved)");
      } else {
        std::string const shown = std::filesystem::path(paramsValue)
                                      .lexically_relative(sceneDirectory.AsFilesystemPath())
                                      .generic_string();
        ImGui::Text(
            "%s%s",
            shown.empty() ? paramsValue.c_str() : shown.c_str(),
            ControllerParamsCache::IsModified(*entry) ? " *" : "");
        if (ImGui::IsItemHovered()) {
          ImGui::SetTooltip("%s", paramsValue.c_str());
        }
      }
      if (ImGui::Button("Open...")) {
        auto const selected = GetControllerParamsPath(
            "Open Controller Parameters",
            false,
            entry != nullptr ? mochi::Path{paramsValue} : sceneDirectory);
        if (!selected.IsEmpty()) {
          controller.params = selected.ToString();
          result.changed = true;
          result.referencesChanged = true;
        }
      }
      ImGui::SameLine();
      if (ImGui::Button("Save As...")) {
        auto const selected = GetControllerParamsPath(
            "Save Controller Parameters As",
            true,
            entry != nullptr ? mochi::Path{paramsValue}
                             : sceneDirectory /
                    (std::string(bot.name) + "_" + std::string(controller.name) +
                     ".superdex_controller"));
        if (!selected.IsEmpty()) {
          if (entry != nullptr && entry->error.empty()) {
            paramsCache.SaveAs(paramsValue, selected.ToString());
          } else {
            std::string json;
            std::string error;
            if (!inlineParams || !NormalizeParamsJson(type, paramsValue, json, error)) {
              json = DefaultParamsJson(type);
            }
            paramsCache.Adopt(selected.ToString(), type, std::move(json));
          }
          controller.params = selected.ToString();
          result.changed = true;
          result.referencesChanged = true;
        }
      }
      ImGui::SameLine();
      ImGui::BeginDisabled(paramsValue.empty());
      if (ImGui::Button("Use Defaults")) {
        controller.params.clear();
        result.changed = true;
        result.referencesChanged = true;
      }
      ImGui::EndDisabled();
      if (entry != nullptr) {
        if (auto const& problem = referenceProblem(paramsValue); !problem.empty()) {
          ShowWarning(problem);
        }
      }

      if (entry != nullptr) {
        if (int const users = CountControllersUsingParams(sceneAsset->GetPrefab(), paramsValue);
            users > 1) {
          ImGui::TextDisabled("Shared with %d other controller(s)", users - 1);
        }
        if (!entry->error.empty()) {
          ShowWarning(entry->error);
          if (ImGui::Button("Reset to Defaults")) {
            paramsCache.Adopt(paramsValue, type, DefaultParamsJson(type));
            result.changed = true;
          }
        } else if (entry->type != type) {
          ShowWarning("This file is used by a '" + entry->type + "' controller in this scene");
        } else {
          result.changed |= EditParamsJson(type, entry->json, botPrefab);
        }
      } else {
        std::string json = paramsValue.empty() ? DefaultParamsJson(type) : paramsValue;
        if (inlineParams) {
          std::string error;
          if (!NormalizeParamsJson(type, paramsValue, json, error)) {
            ShowWarning(error);
            json = paramsValue;
          }
        }
        if (EditParamsJson(type, json, botPrefab)) {
          if (json == DefaultParamsJson(type)) {
            controller.params.clear();
          } else {
            controller.params = json;
          }
          result.changed = true;
        }
      }

      std::string const initArgsError = ValidateJson(controller.initArgs, true);
      if (IsOscControllerType(controller.type) && initArgsError.empty()) {
        if (botAsset == nullptr) {
          ImGui::TextDisabled("Load the referenced bot to select controller links");
        } else {
          picojson::object initArgs;
          std::string parseError;
          if (ParseJsonObject(controller.initArgs, initArgs, parseError)) {
            std::string baseLinkName = GetJsonString(initArgs, "baseLinkName");
            if (LinkNameCombo("Base Link", *botPrefab, baseLinkName)) {
              SetJsonString(controller.initArgs, initArgs, "baseLinkName", baseLinkName);
              result.changed = true;
            }
            if (baseLinkName.empty()) {
              ShowWarning("Base link is required");
            } else if (superdex::robotics::FindLinkIndexByName(*botPrefab, baseLinkName) < 0) {
              ShowWarning("Base link does not exist on the selected bot");
            }

            std::string eeLinkName = GetJsonString(initArgs, "eeLinkName");
            if (LinkNameCombo("End Effector Link", *botPrefab, eeLinkName)) {
              SetJsonString(controller.initArgs, initArgs, "eeLinkName", eeLinkName);
              result.changed = true;
            }
            if (eeLinkName.empty()) {
              ShowWarning("End effector link is required");
            } else if (superdex::robotics::FindLinkIndexByName(*botPrefab, eeLinkName) < 0) {
              ShowWarning("End effector link does not exist on the selected bot");
            }
          }
        }
      }

      result.changed |= ImGui::InputTextMultiline(
          "Init Args JSON", &controller.initArgs, ImVec2(-1, ImGui::GetTextLineHeight() * 4));
      if (!initArgsError.empty()) {
        ShowWarning(initArgsError);
      }
    }
    ImGui::PopID();
  }
  ImGui::PopID();

  if (controllerToMoveUp > 0) {
    std::swap(controllers[controllerToMoveUp], controllers[controllerToMoveUp - 1]);
    result.structural = true;
  }
  if (controllerToMoveDown >= 0 &&
      controllerToMoveDown < static_cast<int>(controllers.size()) - 1) {
    std::swap(controllers[controllerToMoveDown], controllers[controllerToMoveDown + 1]);
    result.structural = true;
  }
  if (controllerToDelete >= 0) {
    controllers.erase(controllers.begin() + controllerToDelete);
    result.structural = true;
    result.referencesChanged = true;
  }

  if (ImGui::Button(ICON_FA_PLUS "###AddController")) {
    ImGui::OpenPopup("AddControllerPopup");
  }
  ImGui::SameLine();
  ImGui::TextUnformatted("Add Controller");
  if (ImGui::BeginPopup("AddControllerPopup")) {
    auto addController = [&](std::string_view type) {
      std::set<std::string> existingNames;
      for (auto const& controller : controllers) {
        existingNames.insert(std::string(controller.name));
      }
      superdex::robotics::ControllerEntry entry;
      entry.name = MakeUniqueName("Controller", existingNames);
      entry.type = type;
      controllers.push_back(std::move(entry));
      result.structural = true;
      ImGui::CloseCurrentPopup();
    };
    for (auto const type : kBuiltinControllerTypes) {
      if (ImGui::MenuItem(type.data())) {
        addController(type);
      }
    }
    ImGui::Separator();
    if (ImGui::MenuItem("Custom...")) {
      addController({});
    }
    ImGui::EndPopup();
  }

  return result;
}

using ActiveSlotBots = std::unordered_map<std::string, std::string>;

// The bot @p slot includes: its bot in @p activeBots if it still lists that bot, else its first.
std::string ActiveSlotBot(
    superdex::robotics::SlotEntry const& slot,
    ActiveSlotBots const& activeBots) {
  if (auto const it = activeBots.find(std::string(slot.name)); it != activeBots.end() &&
      std::ranges::any_of(slot.controlledBots,
                          [&](auto const& name) { return std::string_view(name) == it->second; })) {
    return it->second;
  }
  return slot.controlledBots.empty() ? std::string() : std::string(slot.controlledBots.front());
}

// @p prefab without the bots its slots leave out: those listed in a slot but active in none. Like
// consumers that retire unselected slot alternatives, a bot active in any slot stays, as does a bot
// in no slot. The choice is made, so the slots are dropped, and a slot still being authored cannot
// fail the scene's validation.
superdex::robotics::BotScenePrefab WithActiveSlotBots(
    superdex::robotics::BotScenePrefab prefab,
    ActiveSlotBots const& activeBots) {
  std::set<std::string, std::less<>> slotted;
  std::set<std::string, std::less<>> active;
  for (auto const& slot : prefab.slots) {
    for (auto const& name : slot.controlledBots) {
      slotted.insert(std::string(name));
    }
    active.insert(ActiveSlotBot(slot, activeBots));
  }
  auto& bots = prefab.bots;
  bots.erase(
      std::remove_if(
          bots.begin(),
          bots.end(),
          [&](auto const& bot) {
            std::string_view const name{bot.name};
            return slotted.contains(name) && !active.contains(name);
          }),
      bots.end());
  prefab.slots.clear();
  return prefab;
}

} // namespace

BotSceneEditor::BotSceneEditor(SuperDexStudio* studio, BotSceneAsset* asset)
    : AssetEditor(studio, asset), _sceneAsset(asset), _stage(studio, "BotSceneEditorStage") {}

void BotSceneEditor::Initialize() {
  // Load referenced params files before the first undo snapshot so it records their values.
  auto& paramsCache = _sceneAsset->GetControllerParams();
  for (auto const& bot : _sceneAsset->GetPrefab().bots) {
    for (auto const& controller : bot.controllers) {
      if (!controller.params.empty() && !superdex::robotics::IsInlineJson(controller.params)) {
        paramsCache.GetOrLoad(std::string(controller.params), std::string(controller.type));
      }
    }
  }

  // Initialize _undoStack for every writable scene or archive.
  if (!_sceneAsset->IsReadOnly()) {
    _undoStack.Initialize(
        [this] { return TakeUndoSnapshot(); },
        [this](std::string const& json, int selIdx) { RestoreUndoSnapshot(json, selIdx); });
  }
  // Initialize Viewport (read-only: no gizmo).
  _viewport = Viewport::Create(_studio, _studio->GetViewSettings());
  _viewport->showTransformGizmoTarget = []() { return false; };
  // Register viewport "Show" toggle commands (keyboard shortcuts + top-left dropdown).
  _viewport->RegisterShowCommand(
      {.name = "Physics Debug Draw",
       .onToggle = [this] { _mochiScene.ToggleDebugDraw(); },
       .getState = [this] { return _mochiScene.IsDebugDrawEnabled(); },
       .shortcut = ImGuiKey_P});
  _viewport->RegisterShowCommand(
      {.name = "Render Only",
       .onToggle =
           [this] {
             _stageType = StageType::RenderModelOnly;
             RestageBotScene();
           },
       .getState = [this] { return _stageType == StageType::RenderModelOnly; },
       .shortcut = ImGuiKey_1});
  _viewport->RegisterShowCommand(
      {.name = "Collision Only",
       .onToggle =
           [this] {
             _stageType = StageType::MochiModelOnly;
             RestageBotScene();
           },
       .getState = [this] { return _stageType == StageType::MochiModelOnly; },
       .shortcut = ImGuiKey_2});
  _viewport->RegisterShowCommand(
      {.name = "Render (Collision Fallback)",
       .onToggle =
           [this] {
             _stageType = StageType::RenderModelFallbackToMochiModel;
             RestageBotScene();
           },
       .getState = [this] { return _stageType == StageType::RenderModelFallbackToMochiModel; },
       .shortcut = ImGuiKey_3});
  _viewport->RegisterShowCommand(
      {.name = "SDF (Collision Fallback)",
       .onToggle =
           [this] {
             _stageType = StageType::SdfFallbackToMochiModel;
             RestageBotScene();
           },
       .getState = [this] { return _stageType == StageType::SdfFallbackToMochiModel; },
       .shortcut = ImGuiKey_4});
  // Initialize Mochi scene and callbacks.
  _mochiScene.Initialize(_studio->GetMochiContext(), "BotSceneEditor");
  _mochiScene.SetSettings(_studio->GetAppSettings().physics);
  _mochiScene.createPhysicsActors = [this](mochi::Scene* s) { CreatePhysicsActors(s); };
  _mochiScene.destroyPhysicsActors = [this](mochi::Scene* s) { DestroyPhysicsActors(s); };
  _mochiScene.registerPostStepCallback = [this](mochi::AsyncScene* a) {
    return RegisterPostStepCallback(a);
  };
  // Create the per-session force-drag controller (this hook fires once per CreateScene); it is torn
  // down in OnStopPhysics so its lifetime matches the session.
  _mochiScene.registerPreStepCallback = [this](mochi::AsyncScene*) {
    _dragController = std::make_unique<PhysicsDragController>(
        _mochiScene,
        &_stage,
        _studio->GetAppSettings().physicsDrag,
        _studio->GetRendererToEditorSpaceConverter());
    return mochi::CallbackHandle{};
  };
  _mochiScene.onStartPhysics = [this]() { OnStartPhysics(); };
  _mochiScene.onStopPhysics = [this]() { OnStopPhysics(); };
  // Force-drag (left-drag) of simulated rigid bodies / articulated links. The controller only
  // exists while a session is running, so the hooks read the slot on each event.
  BindSceneObjectDragHooks(*_viewport, _dragController);
  // Bind the stage to the viewport's render scene and stage the bot scene.
  _stage.BindRenderScene(_viewport->GetRenderScene());
  // Measure tool (Ctrl+M): pick vertices/faces on the staged actors' render and collision meshes.
  BindSceneStageMeasureTargets(
      *_viewport,
      _stage,
      [this] { return _mochiScene.IsSimulating(); },
      [this] { return _mochiScene.IsPaused(); });
  RestageBotScene();
  _viewport->FocusCameraOnScene();
}

void BotSceneEditor::OnHandleInputs() {
  if (CanSimulate()) {
    _mochiScene.HandleHotkeys();
  }
}

void BotSceneEditor::OnRender(Renderer const* renderer) {
  // Sync data from physics simulation.
  if (_mochiScene.IsSimulating()) {
    _mochiScene.UpdateStats();
    SyncFromPhysics();
  }
  // Draw mochi scene debug.
  _mochiScene.DrawDebug(
      _viewport->GetRenderScene()->GetDebugDraw(), _studio->GetEditorToRendererSpaceConverter());
  // Draw force-drag visualization (grab point, target, connecting line).
  if (_dragController) {
    _dragController->DrawDebug(
        _viewport->GetRenderScene()->GetDebugDraw(),
        _viewport->GetDebugText(),
        _studio->GetEditorToRendererSpaceConverter());
  }
  // Render the scene.
  _viewport->RenderScene(renderer);
}

void BotSceneEditor::Shutdown() {
  if (_mochiScene.IsSimulating()) {
    _mochiScene.DestroyMochiScene();
  }
  _stage.Clear();
  _viewport.reset();
}

void BotSceneEditor::OnActivate() {
  _referenceProblems.clear();
  // A referenced base scene, spawnable prefab, or bot may have been edited (and saved) in another
  // tab while we were inactive. RestageBotScene reloads referenced assets and re-stages, so their
  // changes appear automatically when returning to this editor.
  if (!_mochiScene.IsSimulating()) {
    RestageBotScene();
  }
}

void BotSceneEditor::Refresh() {
  // A referenced base scene, spawnable prefab, or bot was replaced/renamed elsewhere in the app.
  // Stop any running simulation (restaging mid-sim is unsafe), then restage to reflect it.
  if (_mochiScene.IsSimulating()) {
    _mochiScene.DestroyMochiScene();
  }
  RestageBotScene();
}

void BotSceneEditor::OnDeactivate() {
  // Stop simulation when the user changes to another editor.
  if (_mochiScene.IsSimulating()) {
    _mochiScene.DestroyMochiScene();
  }
}

void BotSceneEditor::ShowTabContents() {
  ImGui::BeginChild("Viewport_Child", ImVec2(0, 0), 0, ImGuiWindowFlags_NoMove);
  // Force-drag only applies while the sim runs.
  _viewport->enableSceneObjectDrag = _mochiScene.IsSimulating();
  _viewport->ShowViewportContents(true);
  _viewport->ShowStatsOverlay(_mochiScene.GetStepsPerSecond());
  ImGui::BeginDisabled(!CanSimulate());
  _mochiScene.ShowPlayToolbarOverViewport();
  ImGui::EndDisabled();
  ImGui::EndChild(); // Viewport_Child
}

std::vector<AssetEditor::WindowDeclaration> BotSceneEditor::GetDefaultWindows() {
  using Dock = AssetEditor::DockRegion;
  return {
      {"Bot Scene Info", true, Dock::SidePanelTop},
      {"Task Preview", true, Dock::SidePanelBottom},
      {"Physics Settings", false, Dock::SidePanelBottom},
      MeasureWindowDeclaration(),
      {"Scene Stage Debug", false, Dock::SidePanelTop, true}};
}

AssetSceneOverrides BotSceneEditor::GetAssetSceneOverrides() const {
  // A base scene is the authoritative source for every simulation physics parameter, so it supplies
  // both gravity and the solver.
  bool const hasBaseScene = !_sceneAsset->GetPrefab().scene.baseScene.empty();
  return {hasBaseScene, hasBaseScene};
}

std::vector<AssetEditor::WindowDeclaration> BotSceneEditor::GetAuxiliaryWindows() const {
  return GetDefaultWindows();
}

void BotSceneEditor::ShowAuxiliaryWindows() {
  if (bool& open = _studio->GetWindowVisible("Bot Scene Info")) {
    ShowInfoWindow(&open);
  }
  if (bool& open = _studio->GetWindowVisible("Physics Settings")) {
    if (_mochiScene.ShowPhysicsSettingsWindow(
            "Physics Settings", &open, GetAssetSceneOverrides())) {
      _viewport->SetGroundPlaneHeight(_mochiScene.GetGroundPlaneHeight());
    }
  }
  ShowMeasureWindow();
  if (bool& open = _studio->GetWindowVisible("Task Preview")) {
    ShowTaskWindow(&open);
  }
  if (bool& open = _studio->GetWindowVisible("Scene Stage Debug")) {
    auto* simNames = _mochiScene.IsSimulating() ? &_simData.GetConsumerData().actorNames : nullptr;
    _stage.ShowSceneStageWindow("Scene Stage Debug", &open, simNames);
  }
}

void BotSceneEditor::ShowMainMenuItems() {
  if (!ImGui::BeginMenu("Bot Scene")) {
    return;
  }

  _mochiScene.ShowExportSimulationPrefabMenuItem(_sceneAsset->GetName());

  if (!_sceneAsset->IsArchive()) {
    if (ImGui::MenuItem("Create Scene Archive...", nullptr, false, !_sceneAsset->IsDirty())) {
      _studio->CreateSceneArchive(_sceneAsset->GetPath());
    }
    if (_sceneAsset->IsDirty() && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
      ImGui::SetTooltip("Save the scene before archiving it.");
    }
  } else {
    if (ImGui::MenuItem("Extract Archive to Loose Files...")) {
      auto const outputDirectory = SuperDexStudio::GetFolderDialogPath(
          "Select an Empty Folder for the Extracted Files", _sceneAsset->GetPath().GetParentPath());
      if (!outputDirectory.IsEmpty()) {
        mochi::Path scenePath;
        if (_sceneAsset->ExtractArchiveToLooseFiles(outputDirectory, scenePath)) {
          _studio->AddFolderToWorkspace(outputDirectory);
          _studio->OpenFile(scenePath);
        }
      }
    }
  }

  ImGui::EndMenu();
}

bool BotSceneEditor::CanUndoRedo() const {
  return !_mochiScene.IsSimulating();
}

void BotSceneEditor::ApplySceneViewSettings(mochi_renderer::SceneViewSettings const& viewSettings) {
  if (_viewport && _viewport->GetRenderScene()) {
    _viewport->GetRenderScene()->ApplyViewSettings(viewSettings);
  }
}

void BotSceneEditor::OnAppSettingsChanged(AppSettings const& settings) {
  // The drag controller snapshots its tuning per session, so a live one has to be told.
  if (_dragController) {
    _dragController->SetSettings(settings.physicsDrag);
  }
}
//--------------------------------------------------------------------------------------------------
// Undo/Redo
//--------------------------------------------------------------------------------------------------

std::string BotSceneEditor::TakeUndoSnapshot() const {
  picojson::object params;
  for (auto const& [path, json] : _sceneAsset->GetControllerParams().CaptureValues()) {
    params[path] = picojson::value(json);
  }
  picojson::object snapshot;
  snapshot["scene"] = picojson::value(SReflect::ToJsonString(_sceneAsset->GetPrefab(), false));
  snapshot["params"] = picojson::value(params);
  picojson::object tasks;
  for (auto const& [path, json] : _sceneAsset->GetTaskCache().CaptureValues()) {
    tasks[path] = picojson::value(json);
  }
  snapshot["tasks"] = picojson::value(tasks);
  snapshot["task"] = picojson::value(_taskPath.ToString());
  return picojson::value(snapshot).serialize();
}

void BotSceneEditor::RestoreUndoSnapshot(std::string const& json, int /*selectionIndex*/) {
  picojson::value snapshot;
  std::string parseError;
  picojson::parse(snapshot, json.begin(), json.end(), &parseError);
  picojson::object const empty;
  auto const& fields = snapshot.is<picojson::object>() ? snapshot.get<picojson::object>() : empty;
  auto const sceneField = fields.find("scene");
  auto const paramsField = fields.find("params");
  if (!parseError.empty() || sceneField == fields.end() || !sceneField->second.is<std::string>() ||
      paramsField == fields.end() || !paramsField->second.is<picojson::object>()) {
    MOCHI_LOG_ERROR("Failed to parse bot scene undo snapshot: %s", parseError.c_str());
    return;
  }
  std::map<std::string, std::string> params;
  for (auto const& [path, value] : paramsField->second.get<picojson::object>()) {
    if (value.is<std::string>()) {
      params.emplace(path, value.get<std::string>());
    }
  }
  _sceneAsset->GetControllerParams().RestoreValues(params);

  std::map<std::string, std::string> tasks;
  if (auto const tasksField = fields.find("tasks");
      tasksField != fields.end() && tasksField->second.is<picojson::object>()) {
    for (auto const& [path, value] : tasksField->second.get<picojson::object>()) {
      if (value.is<std::string>()) {
        tasks.emplace(path, value.get<std::string>());
      }
    }
  }
  _sceneAsset->GetTaskCache().RestoreValues(tasks);
  if (auto const taskField = fields.find("task");
      _taskPrefab.has_value() && taskField != fields.end() && taskField->second.is<std::string>()) {
    std::string const& path = taskField->second.get<std::string>();
    if (_sceneAsset->GetTaskCache().Find(path) != nullptr) {
      _taskPath = mochi::Path{path};
    }
  }
  SyncTaskFromCache();

  // Reset to defaults before deserializing — NoSerializeDefaults omits default-valued fields from
  // the JSON, so without a reset those fields would retain the current (edited) values instead of
  // reverting to defaults.
  _sceneAsset->GetPrefab() = superdex::robotics::BotScenePrefab{};
  SReflect::FromJsonString(
      _sceneAsset->GetPrefab(),
      sceneField->second.get<std::string>(),
      SReflect::DeserializeFlags::Default);

  // Reload referenced assets and refresh reference tracking, then re-stage the viewport.
  _studio->GetAssetManager().ResyncReferencer(_sceneAsset);
  RestageBotScene();

  // Save Task can leave the undo stack's saved state out of step with the task files.
  _sceneAsset->SetDirty(
      !GetUndoStack().IsAtSavedState() || _sceneAsset->GetTaskCache().AnyModified());
}

//--------------------------------------------------------------------------------------------------
// Staging
//--------------------------------------------------------------------------------------------------

void BotSceneEditor::RestageBotScene() {
  // Scene edits, undo/redo, and renames can change the prefab each task spawn binds to. The physics
  // thread reads _taskSpawns, so bindings stay fixed for the duration of a simulation.
  if (_taskPrefab.has_value() && !_mochiScene.IsSimulating()) {
    _taskBindError.clear();
    BindTaskSpawns(*_taskPrefab, _taskSpawns, _taskBindError);
  }
  auto const stagedPrefab = WithActiveSlotBots(_sceneAsset->GetPrefab(), _activeSlotBots);
  if (_taskPrefab.has_value()) {
    _stage.StageBotTask(stagedPrefab, mochi::MakeConstSpan(_taskSpawns), _stageType);
  } else {
    _stage.StageBotScene(stagedPrefab, _stageType);
  }
  // Position the drop-shadow ground plane where the studio physics ground plane goes: the scene's
  // lowest point (rest pose staged above; not called mid-sim) or the custom height.
  _mochiScene.SetLowestPointHeight(_viewport->UpdateGroundPlane());
  _viewport->SetGroundPlaneHeight(_mochiScene.GetGroundPlaneHeight());
}

bool BotSceneEditor::BindTaskSpawns(
    superdex::robotics::BotTaskPrefab const& task,
    std::vector<ResolvedTaskSpawn>& spawns,
    std::string& error) {
  if (!ResolveTaskSpawns(task, _sceneAsset->GetPrefab(), spawns, error)) {
    spawns.clear();
    return false;
  }
  // SceneStage only finds already-loaded assets.
  auto& assetManager = _studio->GetAssetManager();
  for (auto const& spawn : spawns) {
    bool loaded = false;
    if (superdex::robotics::IsBotArchivePath(spawn.prefabPath.ToString()) ||
        superdex::robotics::IsBotPath(spawn.prefabPath.ToString())) {
      loaded = assetManager.LoadBotAsset(spawn.prefabPath) != nullptr;
    } else {
      loaded = assetManager.LoadMochiPrefabAsset(spawn.prefabPath) != nullptr;
    }
    if (!loaded) {
      error = "Failed to load prefab '" + spawn.prefabPath.ToString() + "' for task spawn '" +
          spawn.name + "'";
      spawns.clear();
      return false;
    }
  }
  return true;
}

bool BotSceneEditor::OpenTask(mochi::Path const& path) {
  if (_mochiScene.IsSimulating()) {
    MOCHI_LOG_WARNING("Stop the simulation before loading bot task '%s'.", path.ToString().c_str());
    return false;
  }
  _studio->GetWindowVisible("Task Preview") = true;
  _focusTaskWindow = true;
  return LoadTask(path);
}

bool BotSceneEditor::LoadTask(mochi::Path const& path) {
  std::string error;
  auto const* entry = _sceneAsset->GetTaskCache().GetOrLoad(path.ToString(), error);
  if (entry == nullptr) {
    MOCHI_LOG_ERROR("Failed to load bot task '%s': %s", path.ToString().c_str(), error.c_str());
    _taskLoadError = "Failed to load '" + path.ToString() + "': " + error;
    return false;
  }
  _taskLoadError.clear();
  _taskSaveError.clear();
  _taskPath = path;
  _taskPrefab.emplace();
  SReflect::FromJsonString(*_taskPrefab, entry->json);
  _showCurrentTaskTransforms = false;
  // A task that does not bind to this scene still loads, so it can be edited until it does.
  RestageBotScene();
  _viewport->FocusCameraOnScene();
  return true;
}

void BotSceneEditor::ClearTask() {
  _taskPath = {};
  _taskPrefab.reset();
  _taskSpawns.clear();
  _taskBindError.clear();
  _taskLoadError.clear();
  _taskSaveError.clear();
  _showCurrentTaskTransforms = false;
  RestageBotScene();
}

void BotSceneEditor::SyncTaskFromCache() {
  if (!_taskPrefab.has_value()) {
    return;
  }
  if (auto const* entry = _sceneAsset->GetTaskCache().Find(_taskPath.ToString())) {
    _taskPrefab.emplace();
    SReflect::FromJsonString(*_taskPrefab, entry->json);
  }
}

void BotSceneEditor::SaveTaskAs() {
  std::string const extension{superdex::robotics::kBotTaskExtension};
  std::array<char const*, 1> const filters{{"*.superdex_task"}};
  int const numFilters = MOCHI_PLATFORM_MACOS ? 0 : static_cast<int>(filters.size());
  mochi::Path selected = SuperDexStudio::GetFileDialogPath(
      "Save Bot Task As",
      filters.data(),
      numFilters,
      "SuperDex Task (*.superdex_task)",
      true,
      _taskPath.GetParentPath() / (GetAssetNameFromPath(_taskPath) + "_copy" + extension));
  if (selected.IsEmpty()) {
    return;
  }
  if (!selected.GetFilename().ends_with(extension)) {
    selected = mochi::Path{selected.ToString() + extension};
  }
  _taskSaveError.clear();
  if (!_sceneAsset->GetTaskCache().SaveAs(
          _taskPath.ToString(), selected.ToString(), _taskSaveError)) {
    MOCHI_LOG_ERROR("%s", _taskSaveError.c_str());
    return;
  }
  _taskPath = selected;
  _studio->GetAssetBrowser().Refresh();
  // Undo snapshots record the loaded task, so undoing this switches back to the original.
  GetUndoStack().PushNow();
}

void BotSceneEditor::OnTaskEdited(bool structural) {
  if (!_taskPrefab.has_value()) {
    return;
  }
  if (auto* entry = _sceneAsset->GetTaskCache().Find(_taskPath.ToString())) {
    entry->json = SReflect::ToJsonString(*_taskPrefab, true);
  }
  RestageBotScene();
  _sceneAsset->SetDirty(true);
  if (structural) {
    GetUndoStack().PushNow();
  } else {
    GetUndoStack().MarkEdited();
  }
}

//--------------------------------------------------------------------------------------------------
// Mochi Scene
//--------------------------------------------------------------------------------------------------

bool BotSceneEditor::CanSimulate() const {
  return !_stage.IsEmpty();
}

void BotSceneEditor::CreatePhysicsActors(mochi::Scene* scene) {
  auto* mochiContext = _studio->GetMochiContext();
  auto* botsContext = _studio->GetRoboticsContext();

  auto const findBotPrefab = [this](std::string_view path) -> superdex::robotics::BotPrefab const* {
    auto const it = _physicsBotPrefabs.find(std::string(path));
    return it != _physicsBotPrefabs.end() ? &it->second : nullptr;
  };

  mochi::ErrorLog e;
  _botScene = superdex::robotics::LoadBotScene(
      scene,
      _physicsScenePrefab,
      _sceneAsset->GetBotsRootPath(),
      mochiContext,
      botsContext,
      e,
      findBotPrefab,
      &_studio->GetBotLoader());
  if (!e.IsOK()) {
    MOCHI_LOG_ERROR("Failed to load bot scene physics");
    _botScene.reset();
    return;
  }

  auto const& prefab = _physicsScenePrefab;

  // Each physics actor's skin bone remap (see ComputeSkinBoneLinkRemap), empty for all but bots
  // with a composed skin. Parallel to _physicsActors.
  std::vector<mochi::DynamicArray<int>> skinBoneLinkRemaps;
  auto const addActor = [&](mochi::ActorHandle handle, mochi::DynamicArray<int> skinBoneLinkRemap) {
    _physicsActors.push_back(handle);
    skinBoneLinkRemaps.push_back(std::move(skinBoneLinkRemap));
  };

  // Physics actor order must match SceneStage: base actors, spawnables/task instances, then bots.
  _physicsActors.clear();
  _skinSyncs.clear();
  _taskRuntimeSpawns.clear();
  _taskBots.clear();
  _taskPrefabActors.clear();
  _taskPrefabConstraints.clear();
  for (auto const& handle : _botScene->GetBaseSceneActorHandles()) {
    addActor(handle, {});
  }

  if (_taskPrefab.has_value()) {
    // A spawn that fails here is still staged, and SceneStage drops every simulated transform when
    // the actor counts disagree, so any failure disables physics as a whole.
    _taskRuntimeSpawns.resize(_taskSpawns.size());
    for (int i = 0; i < mochi::isize(_taskSpawns); ++i) {
      auto const& spawn = _taskSpawns[i];
      mochi::ErrorLog spawnError;
      if (superdex::robotics::IsBotArchivePath(spawn.prefabPath.ToString()) ||
          superdex::robotics::IsBotPath(spawn.prefabPath.ToString())) {
        auto const* loadedPrefab = findBotPrefab(spawn.prefabPath.ToString());
        auto botPrefab = loadedPrefab != nullptr
            ? *loadedPrefab
            : superdex::robotics::LoadBotPrefabFromFile(spawn.prefabPath.ToString(), spawnError);
        if (!spawnError.IsOK()) {
          MOCHI_LOG_ERROR("Failed to load task bot '%s'", spawn.name.c_str());
          DestroyPhysicsActors(scene);
          return;
        }
        botPrefab.name = spawn.name;
        botPrefab.worldFromRoot = spawn.worldFromSpawn;
        auto* bot = botsContext->CreateBot(scene, botPrefab, _studio->GetBotLoader(), spawnError);
        if (bot != nullptr) {
          _taskBots.push_back(bot);
        }
        if (!spawnError.IsOK() || bot == nullptr || bot->GetArticulatedActor() == nullptr) {
          MOCHI_LOG_ERROR("Failed to create task bot '%s'", spawn.name.c_str());
          DestroyPhysicsActors(scene);
          return;
        }
        auto* primary = bot->GetArticulatedActor();
        auto const handle = primary->GetHandle();
        addActor(handle, ComputeSkinBoneLinkRemap(botPrefab));
        _taskRuntimeSpawns[i].primaryActor = handle;
        _taskRuntimeSpawns[i].spawnFromPrimary =
            mochi::Invert(spawn.worldFromSpawn) * primary->GetRootTransform();
        continue;
      }

      mochi::prefab::PrefabParams params;
      params.name = spawn.name;
      params.rotation = spawn.worldFromSpawn.GetRotation();
      params.translation = spawn.worldFromSpawn.GetTranslation();
      params.applySceneSettings = false;
      auto const result = mochi::prefab::AddToScene(
          spawn.prefabPath.ToString(),
          spawn.prefabPath.GetParentPath().ToString(),
          scene,
          params,
          spawnError);
      for (auto* constraint : result.constraints) {
        _taskPrefabConstraints.push_back(constraint->GetHandle());
      }
      for (auto* actor : result.actors) {
        _taskPrefabActors.push_back(actor->GetHandle());
      }
      if (!spawnError.IsOK() || result.actors.empty()) {
        MOCHI_LOG_ERROR("Failed to create task object '%s'", spawn.name.c_str());
        DestroyPhysicsActors(scene);
        return;
      }
      for (auto* actor : result.actors) {
        addActor(actor->GetHandle(), {});
      }
      auto* primary = result.actors.front();
      _taskRuntimeSpawns[i].primaryActor = primary->GetHandle();
      _taskRuntimeSpawns[i].spawnFromPrimary =
          mochi::Invert(spawn.worldFromSpawn) * primary->GetRootTransform();
    }
  }
  for (auto const& botEntry : prefab.bots) {
    auto* bot = _botScene->GetBot(std::string(botEntry.name));
    auto* actor = bot != nullptr ? bot->GetArticulatedActor() : nullptr;
    if (actor != nullptr) {
      auto const* botPrefab = findBotPrefab(std::string_view(botEntry.path));
      addActor(
          actor->GetHandle(),
          botPrefab != nullptr ? ComputeSkinBoneLinkRemap(*botPrefab) : mochi::DynamicArray<int>{});
    }
  }

  // Track the skins SceneStage staged. Rows follow the per-step transform list: an articulated
  // actor's links, then its skin if one was staged; a rigid actor's root.
  int row = 0;
  for (int i = 0; i < mochi::isize(_physicsActors); ++i) {
    auto* actor = scene->GetActor(_physicsActors[i]);
    if (actor == nullptr) {
      continue;
    }
    if (actor->GetType() == mochi::ActorType::Articulated) {
      row += mochi::isize(actor->GetNestedLinkActors(e));
      if (row < mochi::isize(_stagedSkinRows) && _stagedSkinRows[row]) {
        _skinSyncs.emplace_back().Track(*actor, std::move(skinBoneLinkRemaps[i]), row);
        ++row;
      }
    } else if (actor->GetType() == mochi::ActorType::Rigid) {
      ++row;
    }
  }
}

void BotSceneEditor::DestroyPhysicsActors(mochi::Scene* scene) {
  for (auto& skinSync : _skinSyncs) {
    skinSync.Untrack(scene);
  }
  _skinSyncs.clear();
  // Destroy bots/controllers before the async scene destroys the scene. The BotScene is non-owning,
  // so the scene itself is left intact for MochiAsyncScene to destroy.
  for (auto* bot : _taskBots) {
    superdex::robotics::DestroyBot(scene, bot);
  }
  _taskBots.clear();
  for (auto const handle : _taskPrefabConstraints) {
    scene->DestroyConstraint(handle);
  }
  _taskPrefabConstraints.clear();
  for (auto const handle : _taskPrefabActors) {
    scene->DestroyActor(handle);
  }
  _taskPrefabActors.clear();
  _taskRuntimeSpawns.clear();
  _botScene.reset();
  _physicsActors.clear();
}

mochi::CallbackHandle BotSceneEditor::RegisterPostStepCallback(mochi::AsyncScene* scene) {
  return scene->RegisterPostStepCallback(
      "BotSceneEditor::ExtractActors", [this](mochi::StepInfo const& info) {
        mochi::ErrorLog e;
        auto& data = _simData.GetProducerData();
        data.actorTransforms.clear();
        data.actorNames.clear();
        data.taskTransforms.clear();
        auto const& converter = _studio->GetEditorToRendererSpaceConverter();
        // Skin updates are filled in place to reuse their buffers, then trimmed below.
        int numSkinPoses = 0;
        int numSkinSurfaces = 0;
        auto const nextUpdate = [](auto& updates, int& count) -> auto& {
          if (mochi::isize(updates) <= count) {
            updates.emplace_back();
          }
          return updates[count++];
        };
        // Build the ordered transform list
        // actors expand to their nested link transforms (then their staged skin's row, at the
        // actor's root), rigid actors push their root transform, and everything else (e.g. soft) is
        // skipped.
        for (auto const& handle : _physicsActors) {
          auto* actor = info.scene->GetActor(handle);
          if (!actor) {
            continue;
          }
          if (actor->GetType() == mochi::ActorType::Articulated) {
            auto const links = actor->GetNestedLinkActors(e);
            std::vector<mochi::TransformRT> linkTransforms(links.size());
            actor->GetArticulatedLinkTransforms(mochi::MakeSpan(linkTransforms), e);
            for (int i = 0; i < mochi::isize(linkTransforms); ++i) {
              data.actorTransforms.push_back(linkTransforms[i]);
              char const* const linkName = info.scene->GetActor(links[i])->GetName();
              data.actorNames.emplace_back(linkName ? linkName : "");
            }
            auto const skinSync =
                std::ranges::find(_skinSyncs, handle, &SkinSimSync::GetActorHandle);
            if (skinSync != _skinSyncs.end()) {
              data.actorTransforms.push_back(actor->GetRootTransform());
              data.actorNames.push_back(skinSync->GetName());
              SoftMeshUpdate* const surface = skinSync->HasSurface()
                  ? &nextUpdate(data.skinSurfaces, numSkinSurfaces)
                  : nullptr;
              skinSync->Produce(
                  *actor,
                  mochi::MakeConstSpan(linkTransforms),
                  converter,
                  nextUpdate(data.skinPoses, numSkinPoses),
                  surface);
            }
          } else if (actor->GetType() == mochi::ActorType::Rigid) {
            data.actorTransforms.push_back(actor->GetRootTransform());
            char const* const rigidName = actor->GetName();
            data.actorNames.emplace_back(rigidName ? rigidName : "");
          }
        }
        // Empty when CreatePhysicsActors failed, even with a task loaded.
        data.taskTransforms.reserve(_taskRuntimeSpawns.size());
        for (int i = 0; i < mochi::isize(_taskRuntimeSpawns); ++i) {
          auto const& runtime = _taskRuntimeSpawns[i];
          auto* primary = runtime.primaryActor.has_value()
              ? info.scene->GetActor(*runtime.primaryActor)
              : nullptr;
          data.taskTransforms.push_back(
              primary ? primary->GetRootTransform() * mochi::Invert(runtime.spawnFromPrimary)
                      : _taskSpawns[i].worldFromSpawn);
        }
        data.skinPoses.resize(numSkinPoses);
        data.skinSurfaces.resize(numSkinSurfaces);
        _simData.Produce();
      });
}

std::string const& BotSceneEditor::GetReferenceProblem(std::string_view path) {
  static std::string const kNone;
  // Archives import external files when saved, so any reference works.
  if (path.empty() || _sceneAsset->IsArchive() || superdex::robotics::IsInlineJson(path) ||
      !std::filesystem::path(path).is_absolute()) {
    return kNone;
  }
  auto const [it, inserted] = _referenceProblems.try_emplace(std::string(path));
  if (inserted) {
    it->second = DescribeReferenceProblem(
        std::filesystem::path(path), _sceneAsset->GetPath().AsFilesystemPath());
  }
  return it->second;
}

void BotSceneEditor::OnStartPhysics() {
  // Staged from the same prefab, so the staged and physics actors stay in step.
  _physicsScenePrefab = WithActiveSlotBots(_sceneAsset->GetPrefab(), _activeSlotBots);
  _stagedSkinRows.clear();
  for (auto const& staged : _stage.GetActors()) {
    _stagedSkinRows.push_back(IsSkinStagedName(staged.name));
  }
  auto& paramsCache = _sceneAsset->GetControllerParams();
  for (auto& bot : _physicsScenePrefab.bots) {
    for (auto& controller : bot.controllers) {
      auto const* entry = paramsCache.Find(std::string(controller.params));
      if (entry != nullptr && entry->error.empty() && ControllerParamsCache::IsModified(*entry)) {
        controller.params = entry->json;
      }
    }
  }

  _physicsBotPrefabs.clear();
  auto& assetManager = _studio->GetAssetManager();
  auto const copyBotPrefab = [&](std::string const& path) {
    if (path.empty() || _physicsBotPrefabs.contains(path)) {
      return;
    }
    if (auto* botAsset = assetManager.FindAssetByPath<BotAsset>(mochi::Path{path})) {
      botAsset->Rebuild(_studio->GetBotLoader());
      _physicsBotPrefabs.emplace(path, botAsset->GetBotPrefab());
    }
  };
  for (auto const& bot : _physicsScenePrefab.bots) {
    copyBotPrefab(std::string(bot.path));
  }
  for (auto const& spawn : _taskSpawns) {
    copyBotPrefab(spawn.prefabPath.ToString());
  }
}

void BotSceneEditor::OnStopPhysics() {
  // Tear down the per-session force-drag controller (its callback is already gone with the scene).
  _dragController.reset();
  _stage.ResetWorldTransforms(_studio->GetEditorToRendererSpaceConverter());
  _simData.Consume();
  _physicsActors.clear();
  _physicsBotPrefabs.clear();
  _showCurrentTaskTransforms = false;
}

void BotSceneEditor::SyncFromPhysics() {
  if (_simData.Consume()) {
    auto const& data = _simData.GetConsumerData();
    auto const& converter = _studio->GetEditorToRendererSpaceConverter();
    _stage.ApplyWorldTransforms(mochi::MakeConstSpan(data.actorTransforms), converter);
    _stage.ApplySoftMeshUpdates(mochi::MakeConstSpan(data.skinSurfaces), converter);
    _stage.ApplySkinnedPose(mochi::MakeConstSpan(data.skinPoses), converter);
  }
}

//--------------------------------------------------------------------------------------------------
// ImGui
//--------------------------------------------------------------------------------------------------

void BotSceneEditor::ShowTaskWindow(bool* open) {
  if (std::exchange(_focusTaskWindow, false)) {
    // Focusing a docked window does not select its tab, so select the tab directly.
    ImGuiWindow* const window = ImGui::FindWindowByName("Task Preview");
    if (window != nullptr && window->DockNode != nullptr && window->DockNode->TabBar != nullptr) {
      window->DockNode->TabBar->NextSelectedTabId = window->TabId;
    } else {
      ImGui::SetNextWindowFocus();
    }
  }
  ImGui::Begin("Task Preview", open);

  bool const simulating = _mochiScene.IsSimulating();
  ImGui::BeginDisabled(simulating);
  if (ImGui::Button(_taskPrefab.has_value() ? "Replace Task..." : "Load Task...")) {
    std::array<char const*, 2> const filters{{"*.superdex_task", "*.mochi_bot_task"}};
    int const numFilters = MOCHI_PLATFORM_MACOS ? 0 : static_cast<int>(filters.size());
    mochi::Path const initialPath =
        _taskPath.IsEmpty() ? _sceneAsset->GetPath().GetParentPath() : _taskPath;
    auto const selectedPath = SuperDexStudio::GetFileDialogPath(
        "Select Bot Task",
        filters.data(),
        numFilters,
        "SuperDex Task (*.superdex_task, *.mochi_bot_task)",
        false,
        initialPath);
    if (!selectedPath.IsEmpty()) {
      LoadTask(selectedPath);
    }
  }
  if (_taskPrefab.has_value()) {
    ImGui::SameLine();
    if (ImGui::Button("Clear Task")) {
      ClearTask();
    }
    auto const* entry = _sceneAsset->GetTaskCache().Find(_taskPath.ToString());
    ImGui::SameLine();
    ImGui::BeginDisabled(entry == nullptr || !BotTaskCache::IsModified(*entry));
    if (ImGui::Button("Save Task")) {
      _taskSaveError.clear();
      if (!_sceneAsset->GetTaskCache().Save(_taskPath.ToString(), _taskSaveError)) {
        MOCHI_LOG_ERROR("%s", _taskSaveError.c_str());
      }
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button("Save Task As...")) {
      SaveTaskAs();
    }
  }
  ImGui::EndDisabled();

  if (!_taskLoadError.empty()) {
    ShowWarning(_taskLoadError);
  }
  if (!_taskSaveError.empty()) {
    ShowWarning(_taskSaveError);
  }

  if (!_taskPrefab.has_value()) {
    ImGui::TextWrapped("Load a .superdex_task to place its spawnable objects into this scene.");
    ImGui::End();
    return;
  }

  auto const* entry = _sceneAsset->GetTaskCache().Find(_taskPath.ToString());
  bool const modified = entry != nullptr && BotTaskCache::IsModified(*entry);
  ImGui::TextDisabled("%s%s", _taskPath.ToString().c_str(), modified ? " *" : "");
  ImGui::Separator();

  if (!simulating) {
    ShowTaskEditor();
    ImGui::End();
    return;
  }

  if (!_taskBindError.empty()) {
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.65f, 0.15f, 1.0f));
    ImGui::TextWrapped(
        "%s Task objects are hidden: %s", ICON_FA_EXCLAMATION_TRIANGLE, _taskBindError.c_str());
    ImGui::PopStyleColor();
    ImGui::End();
    return;
  }

  bool const currentSelected = _showCurrentTaskTransforms;
  if (ImGui::RadioButton("Initial", !currentSelected)) {
    _showCurrentTaskTransforms = false;
  }
  ImGui::SameLine();
  if (ImGui::RadioButton("Current", currentSelected)) {
    _showCurrentTaskTransforms = true;
  }

  auto const& current = _simData.GetConsumerData().taskTransforms;
  bool const showCurrent = currentSelected && current.size() == _taskSpawns.size();
  if (currentSelected && !showCurrent) {
    ImGui::TextDisabled("Simulated transforms are unavailable; showing initial transforms.");
  }
  constexpr ImGuiTableFlags kTableFlags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
      ImGuiTableFlags_Resizable | ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingStretchProp;
  if (ImGui::BeginTable("##TaskObjects", 4, kTableFlags)) {
    ImGui::TableSetupScrollFreeze(0, 1);
    ImGui::TableSetupColumn("Object", ImGuiTableColumnFlags_WidthStretch, 1.5f);
    ImGui::TableSetupColumn("Prefab", ImGuiTableColumnFlags_WidthStretch, 1.5f);
    ImGui::TableSetupColumn("Position", ImGuiTableColumnFlags_WidthStretch, 2.0f);
    ImGui::TableSetupColumn("Rotation (xyzw)", ImGuiTableColumnFlags_WidthStretch, 2.5f);
    ImGui::TableHeadersRow();
    for (int i = 0; i < mochi::isize(_taskSpawns); ++i) {
      auto const& spawn = _taskSpawns[i];
      auto const& transform = showCurrent ? current[i] : spawn.worldFromSpawn;
      auto const translation = transform.GetTranslation();
      auto const rotation = transform.GetRotation();
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::TextUnformatted(spawn.name.c_str());
      ImGui::TableNextColumn();
      ImGui::TextUnformatted(spawn.prefabName.c_str());
      ImGui::TableNextColumn();
      ImGui::Text(
          "%.3f, %.3f, %.3f",
          static_cast<double>(translation[0]),
          static_cast<double>(translation[1]),
          static_cast<double>(translation[2]));
      ImGui::TableNextColumn();
      ImGui::Text(
          "%.3f, %.3f, %.3f, %.3f",
          static_cast<double>(rotation.data[0]),
          static_cast<double>(rotation.data[1]),
          static_cast<double>(rotation.data[2]),
          static_cast<double>(rotation.data[3]));
    }
    ImGui::EndTable();
  }

  ImGui::End();
}

void BotSceneEditor::ShowTaskEditor() {
  if (!_taskPrefab.has_value()) {
    return;
  }
  auto& task = *_taskPrefab;
  auto& spawns = task.resetState;
  bool changed = false;
  bool structural = false;
  ImGui::BeginDisabled(_sceneAsset->IsReadOnly());

  if (ImGui::CollapsingHeader("Metadata")) {
    changed |= ImGui::SimpleReflectionStruct(task.metadata);
  }

  mochi::Error validation;
  superdex::robotics::ValidateBotTaskPrefab(task, validation);
  if (!validation.IsOK()) {
    ShowWarning(validation.GetDescription());
  } else if (!_taskBindError.empty()) {
    ShowWarning("Task objects are hidden: " + _taskBindError);
  }

  std::vector<std::string> prefabNames;
  for (auto const& prefab : _sceneAsset->GetPrefab().scene.spawnablePrefabs) {
    prefabNames.emplace_back(prefab.name);
  }

  ImGui::HoverableSeparatorText("Spawns");
  int spawnToDelete = -1;
  int spawnToMoveUp = -1;
  int spawnToMoveDown = -1;
  ImGui::PushID("Spawns");
  for (int i = 0; i < mochi::isize(spawns); ++i) {
    auto& spawn = spawns[i];
    ImGui::PushID(i);
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(1, 0));
    if (ImGui::Button(ICON_FA_TRASH)) {
      spawnToDelete = i;
    }
    ImGui::SameLine();
    ImGui::BeginDisabled(i == 0);
    if (ImGui::Button(ICON_FA_CARET_UP)) {
      spawnToMoveUp = i;
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(i == mochi::isize(spawns) - 1);
    if (ImGui::Button(ICON_FA_CARET_DOWN)) {
      spawnToMoveDown = i;
    }
    ImGui::EndDisabled();
    ImGui::PopStyleVar();
    ImGui::SameLine();
    std::string const label =
        (spawn.name.empty() ? std::string("(unnamed)") : std::string(spawn.name)) + " (" +
        std::string(spawn.prefabName) + ")###spawn";
    if (ImGui::CollapsingHeader(label.c_str())) {
      std::string const oldName{spawn.name};
      bool renamed = ImGui::InputText("Name", &spawn.name);
      // Children are found when editing starts, so a name that is briefly empty or taken while
      // typing neither drops nor adopts any. An empty parent means task_root, so a spawn that had
      // no name had no children.
      if (ImGui::IsItemActivated()) {
        _spawnRename = {.name = oldName, .children = {}};
        for (int j = 0; j < mochi::isize(spawns); ++j) {
          if (!oldName.empty() && std::string_view(spawns[j].parent) == oldName) {
            _spawnRename.children.push_back(j);
          }
        }
      }
      // A name left empty or taken reverts, since spawns are parented by name.
      if (ImGui::IsItemDeactivatedAfterEdit() &&
          (spawn.name.empty() || std::ranges::any_of(spawns, [&](auto const& other) {
             return &other != &spawn &&
                 std::string_view(other.name) == std::string_view(spawn.name);
           }))) {
        spawn.name = mochi::DynamicString{_spawnRename.name};
        renamed = true;
      }
      if (renamed) {
        for (int const child : _spawnRename.children) {
          if (child < mochi::isize(spawns)) {
            spawns[child].parent = spawn.name;
          }
        }
        changed = true;
      }
      changed |= NameCombo("Prefab", prefabNames, spawn.prefabName, spawn.prefabName);
      std::vector<std::string> parentNames{std::string(superdex::robotics::kTaskRootParentName)};
      for (auto const& other : spawns) {
        if (&other != &spawn) {
          parentNames.emplace_back(other.name);
        }
      }
      std::string_view const parent = spawn.parent.empty() ? superdex::robotics::kTaskRootParentName
                                                           : std::string_view(spawn.parent);
      if (NameCombo("Parent", parentNames, parent, spawn.parent)) {
        if (std::string_view(spawn.parent) == superdex::robotics::kTaskRootParentName) {
          spawn.parent.clear();
        }
        changed = true;
      }
      mochi::DynamicString poseClass = superdex::robotics::GetPoseRandomizationClass(spawn);
      if (ImGui::InputText("Pose Randomization Class", &poseClass)) {
        SetPoseRandomizationClass(spawn, poseClass);
        changed = true;
      }
      changed |= ImGui::DragTransformRT("Spawn Transform", spawn.parentFromSpawn);
    }
    ImGui::PopID();
  }
  ImGui::PopID(); // Spawns
  if (spawnToMoveUp > 0) {
    std::swap(spawns[spawnToMoveUp], spawns[spawnToMoveUp - 1]);
    structural = true;
  }
  if (spawnToMoveDown >= 0 && spawnToMoveDown < mochi::isize(spawns) - 1) {
    std::swap(spawns[spawnToMoveDown], spawns[spawnToMoveDown + 1]);
    structural = true;
  }
  if (spawnToDelete >= 0) {
    // Reparent the deleted spawn's children to its parent, keeping where they are placed.
    auto const deleted = spawns[spawnToDelete];
    for (auto& other : spawns) {
      if (std::string_view(other.parent) == std::string_view(deleted.name)) {
        other.parent = deleted.parent;
        other.parentFromSpawn = deleted.parentFromSpawn * other.parentFromSpawn;
      }
    }
    spawns.erase(spawns.begin() + spawnToDelete);
    structural = true;
  }
  if (ImGui::Button(ICON_FA_PLUS "###AddSpawn")) {
    std::set<std::string> existingNames;
    for (auto const& spawn : spawns) {
      existingNames.insert(std::string(spawn.name));
    }
    superdex::robotics::SpawnEntry newSpawn;
    newSpawn.prefabName =
        prefabNames.empty() ? mochi::DynamicString{} : mochi::DynamicString{prefabNames.front()};
    newSpawn.name = mochi::DynamicString{MakeUniqueName("TaskObject", existingNames)};
    spawns.push_back(std::move(newSpawn));
    structural = true;
  }
  ImGui::SameLine();
  ImGui::TextUnformatted("Add Spawn");
  if (prefabNames.empty()) {
    ImGui::TextDisabled("Add spawnable prefabs to the scene to place them in tasks.");
  }

  ImGui::EndDisabled();
  if (changed || structural) {
    OnTaskEdited(structural);
  }
}

struct BotSceneEditor::SlotEditResult {
  bool changed = false;
  bool structural = false;
  // Only the active slot bots changed: the scene itself did not.
  bool activeChanged = false;
};

// Edits the scene's slots and picks each one's active bot. Warns about anything
// LoadBotScenePrefabFromFile would reject, since a scene saved that way cannot be reopened.
BotSceneEditor::SlotEditResult BotSceneEditor::ShowSlots(
    superdex::robotics::BotScenePrefab& prefab) {
  SlotEditResult result;
  auto& slots = prefab.slots;
  std::set<std::string, std::less<>> botNames;
  for (auto const& bot : prefab.bots) {
    botNames.insert(std::string(bot.name));
  }
  int slotToDelete = -1;
  int slotToMoveUp = -1;
  int slotToMoveDown = -1;

  ImGui::PushID("Slots");
  for (int i = 0; i < mochi::isize(slots); ++i) {
    auto& slot = slots[i];
    ImGui::PushID(i);
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(1, 0));
    if (ImGui::Button(ICON_FA_TRASH)) {
      slotToDelete = i;
    }
    ImGui::SameLine();
    ImGui::BeginDisabled(i == 0);
    if (ImGui::Button(ICON_FA_CARET_UP)) {
      slotToMoveUp = i;
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(i == mochi::isize(slots) - 1);
    if (ImGui::Button(ICON_FA_CARET_DOWN)) {
      slotToMoveDown = i;
    }
    ImGui::EndDisabled();
    ImGui::PopStyleVar();
    ImGui::SameLine();
    std::string const activeBot = ActiveSlotBot(slot, _activeSlotBots);
    std::string const label =
        (slot.name.empty() ? std::string("(unnamed)") : std::string(slot.name)) +
        (activeBot.empty() ? std::string() : " (" + activeBot + ")") + "###slot";
    if (ImGui::CollapsingHeader(label.c_str())) {
      ImGui::Indent();
      std::string const oldName{slot.name};
      bool const renamed = ImGui::InputText("Name", &slot.name);
      // The active bot moves with the name. Choices are restored from when editing started, so a
      // name that is briefly another slot's does not take over that slot's choice.
      if (ImGui::IsItemActivated()) {
        _slotRename = {.name = oldName, .activeBot = activeBot, .activeSlotBots = _activeSlotBots};
      }
      if (renamed) {
        _activeSlotBots = _slotRename.activeSlotBots;
        _activeSlotBots.erase(_slotRename.name);
        _activeSlotBots[std::string(slot.name)] = _slotRename.activeBot;
        result.changed = true;
      }
      if (slot.name.empty()) {
        ShowWarning("A slot needs a name");
      } else if (std::ranges::any_of(slots, [&](auto const& other) {
                   return &other != &slot &&
                       std::string_view(other.name) == std::string_view(slot.name);
                 })) {
        ShowWarning("Another slot has this name");
      }

      auto& controlledBots = slot.controlledBots;
      int botToRemove = -1;
      ImGui::PushID("ControlledBots");
      for (int j = 0; j < mochi::isize(controlledBots); ++j) {
        ImGui::PushID(j);
        if (ImGui::RadioButton("##active", std::string_view(controlledBots[j]) == activeBot)) {
          _activeSlotBots[std::string(slot.name)] = std::string(controlledBots[j]);
          result.activeChanged = true;
        }
        ImGui::SameLine();
        if (ImGui::Button(ICON_FA_TRASH)) {
          botToRemove = j;
        }
        ImGui::SameLine();
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(controlledBots[j].c_str());
        if (!botNames.contains(std::string_view(controlledBots[j]))) {
          ShowWarning("No bot in this scene has this name");
        }
        ImGui::PopID();
      }
      ImGui::PopID(); // ControlledBots
      if (botToRemove >= 0) {
        controlledBots.erase(controlledBots.begin() + botToRemove);
        result.structural = true;
      }
      if (controlledBots.empty()) {
        ShowWarning("A slot needs at least one bot");
      }

      std::vector<std::string> addableBots;
      for (auto const& name : botNames) {
        if (std::ranges::none_of(controlledBots, [&](auto const& controlled) {
              return std::string_view(controlled) == name;
            })) {
          addableBots.push_back(name);
        }
      }
      ImGui::BeginDisabled(addableBots.empty());
      if (ImGui::Button(ICON_FA_PLUS "###AddSlotBot")) {
        ImGui::OpenPopup("AddSlotBotPopup");
      }
      ImGui::EndDisabled();
      ImGui::SameLine();
      ImGui::TextUnformatted("Add Bot");
      if (ImGui::BeginPopup("AddSlotBotPopup")) {
        for (auto const& name : addableBots) {
          if (ImGui::MenuItem(name.c_str())) {
            controlledBots.push_back(mochi::DynamicString{name});
            result.structural = true;
          }
        }
        ImGui::EndPopup();
      }
      ImGui::Unindent();
    }
    ImGui::PopID();
  }
  ImGui::PopID(); // Slots
  if (slotToMoveUp > 0) {
    std::swap(slots[slotToMoveUp], slots[slotToMoveUp - 1]);
    result.structural = true;
  }
  if (slotToMoveDown >= 0 && slotToMoveDown < mochi::isize(slots) - 1) {
    std::swap(slots[slotToMoveDown], slots[slotToMoveDown + 1]);
    result.structural = true;
  }
  if (slotToDelete >= 0) {
    slots.erase(slots.begin() + slotToDelete);
    result.structural = true;
  }
  if (ImGui::Button(ICON_FA_PLUS "###AddSlot")) {
    std::set<std::string> existingNames;
    for (auto const& slot : slots) {
      existingNames.insert(std::string(slot.name));
    }
    superdex::robotics::SlotEntry newSlot;
    newSlot.name = mochi::DynamicString{MakeUniqueName("Slot", existingNames)};
    slots.push_back(std::move(newSlot));
    result.structural = true;
  }
  ImGui::SameLine();
  ImGui::TextUnformatted("Add Slot");

  return result;
}

void BotSceneEditor::ShowInfoWindow(bool* open) {
  auto const windowFlags =
      _sceneAsset->IsDirty() ? ImGuiWindowFlags_UnsavedDocument : ImGuiWindowFlags_None;
  ImGui::Begin("Bot Scene Info", open, windowFlags);

  auto& prefab = _sceneAsset->GetPrefab();
  auto& assetManager = _studio->GetAssetManager();

  // Big title (fixed, outside the scroll child): the asset/file name.
  ImGui::PushFont(_studio->GetFont("Roboto Bold Large"));
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted(_sceneAsset->GetName().c_str());
  ImGui::PopFont();
  if (_sceneAsset->IsArchive()) {
    ImGui::TextWrapped(
        "Editing an archive. Save overwrites the source archive; use the Bot Scene menu to save an "
        "archive copy or extract an editable scene tree.");
  }
  ImGui::Separator();

  // Continuous edits (debounced into a single undo entry) vs. discrete structural edits
  // (add/remove/reorder — pushed immediately). Applied together in the epilogue below.
  bool changed = false;
  bool structural = false;
  bool referencesChanged = false;
  bool activeSlotsChanged = false;

  ImGui::PushStyleColor(ImGuiCol_ChildBg, IM_COL32_BLACK_TRANS);
  ImGui::BeginChild("BotSceneInfoChild", ImVec2(0, 0));
  ImGui::BeginDisabled(_sceneAsset->IsReadOnly() || _mochiScene.IsSimulating());

  // ---- Metadata ----
  if (ImGui::CollapsingHeader("Metadata", ImGuiTreeNodeFlags_DefaultOpen)) {
    changed |= ImGui::SimpleReflectionStruct(prefab.metadata);
  }

  // ---- Scene ----
  if (ImGui::CollapsingHeader("Scene", ImGuiTreeNodeFlags_DefaultOpen)) {
    // Base scene.
    if (ImGui::AssetSlot(
            "Base Scene",
            prefab.scene.baseScene,
            assetManager,
            _studio,
            AssetType::MochiPrefab,
            true)) {
      if (!prefab.scene.baseScene.empty()) {
        assetManager.LoadMochiPrefabAsset(prefab.scene.baseScene);
      }
      assetManager.ResyncReferencer(_sceneAsset);
      changed = true;
    }
    if (auto const& problem = GetReferenceProblem(prefab.scene.baseScene); !problem.empty()) {
      ShowWarning(problem);
    }

    // Task root: where every task's root-level spawns are placed.
    bool hasTaskRoot = prefab.scene.taskRoot.has_value();
    if (ImGui::Checkbox("Task Root", &hasTaskRoot)) {
      if (hasTaskRoot) {
        prefab.scene.taskRoot.emplace();
      } else {
        prefab.scene.taskRoot.reset();
      }
      structural = true;
    }
    if (prefab.scene.taskRoot.has_value()) {
      changed |= ImGui::DragTransformRT("Task Root Transform", *prefab.scene.taskRoot);
    }

    // Spawnable prefabs (add / remove / reorder).
    ImGui::HoverableSeparatorText("Spawnable Prefabs");
    auto& spawnables = prefab.scene.spawnablePrefabs;
    int prefabToDelete = -1;
    int prefabToMoveUp = -1;
    int prefabToMoveDown = -1;
    // Section-level ID scope so per-item widget IDs don't collide with the Bots section (both
    // loops push the same integer index).
    ImGui::PushID("Spawnables");
    for (int i = 0; i < static_cast<int>(spawnables.size()); ++i) {
      auto& entry = spawnables[i];
      ImGui::PushID(i);
      ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(1, 0));
      if (ImGui::Button(ICON_FA_TRASH)) {
        prefabToDelete = i;
      }
      ImGui::SameLine();
      ImGui::BeginDisabled(i == 0);
      if (ImGui::Button(ICON_FA_CARET_UP)) {
        prefabToMoveUp = i;
      }
      ImGui::EndDisabled();
      ImGui::SameLine();
      ImGui::BeginDisabled(i == static_cast<int>(spawnables.size()) - 1);
      if (ImGui::Button(ICON_FA_CARET_DOWN)) {
        prefabToMoveDown = i;
      }
      ImGui::EndDisabled();
      ImGui::PopStyleVar();
      ImGui::SameLine();
      // "###" keeps the header ID stable (per PushID) as the visible name is edited.
      std::string const label =
          (entry.name.empty() ? std::string("(unnamed)") : std::string(entry.name)) + "###entry";
      if (ImGui::CollapsingHeader(label.c_str())) {
        changed |= ImGui::InputText("Name", &entry.name);
        if (ImGui::AssetSlot(
                "Prefab", entry.path, assetManager, _studio, AssetType::MochiPrefab, true)) {
          if (!entry.path.empty()) {
            assetManager.LoadMochiPrefabAsset(entry.path);
          }
          assetManager.ResyncReferencer(_sceneAsset);
          changed = true;
        }
        if (auto const& problem = GetReferenceProblem(entry.path); !problem.empty()) {
          ShowWarning(problem);
        }
      }
      ImGui::PopID();
    }
    ImGui::PopID(); // Spawnables
    if (prefabToMoveUp > 0) {
      std::swap(spawnables[prefabToMoveUp], spawnables[prefabToMoveUp - 1]);
      structural = true;
    }
    if (prefabToMoveDown >= 0 && prefabToMoveDown < static_cast<int>(spawnables.size()) - 1) {
      std::swap(spawnables[prefabToMoveDown], spawnables[prefabToMoveDown + 1]);
      structural = true;
    }
    if (prefabToDelete >= 0) {
      spawnables.erase(spawnables.begin() + prefabToDelete);
      assetManager.ResyncReferencer(_sceneAsset);
      structural = true;
    }
    if (ImGui::Button(ICON_FA_PLUS "###AddSpawnable")) {
      std::set<std::string> existingNames;
      for (auto const& e : spawnables) {
        existingNames.insert(std::string(e.name));
      }
      superdex::robotics::PrefabEntry newEntry;
      newEntry.name = mochi::DynamicString{MakeUniqueName("Prefab", existingNames)};
      spawnables.push_back(std::move(newEntry));
      structural = true;
    }
    ImGui::SameLine();
    ImGui::TextUnformatted("Add Spawnable Prefab");
  }

  // ---- Bots ----
  if (ImGui::CollapsingHeader("Bots", ImGuiTreeNodeFlags_DefaultOpen)) {
    auto& bots = prefab.bots;
    int botToDelete = -1;
    int botToMoveUp = -1;
    int botToMoveDown = -1;
    // Section-level ID scope so per-item widget IDs don't collide with the Spawnable Prefabs
    // section (both loops push the same integer index).
    ImGui::PushID("Bots");
    for (int i = 0; i < static_cast<int>(bots.size()); ++i) {
      auto& bot = bots[i];
      ImGui::PushID(i);
      ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(1, 0));
      if (ImGui::Button(ICON_FA_TRASH)) {
        botToDelete = i;
      }
      ImGui::SameLine();
      ImGui::BeginDisabled(i == 0);
      if (ImGui::Button(ICON_FA_CARET_UP)) {
        botToMoveUp = i;
      }
      ImGui::EndDisabled();
      ImGui::SameLine();
      ImGui::BeginDisabled(i == static_cast<int>(bots.size()) - 1);
      if (ImGui::Button(ICON_FA_CARET_DOWN)) {
        botToMoveDown = i;
      }
      ImGui::EndDisabled();
      ImGui::PopStyleVar();
      ImGui::SameLine();
      std::string const label =
          (bot.name.empty() ? std::string("(unnamed)") : std::string(bot.name)) + "###bot";
      if (ImGui::CollapsingHeader(label.c_str())) {
        std::string const oldName{bot.name};
        bool renamed = ImGui::InputText("Name", &bot.name);
        // Slot entries are found when editing starts, so a name that is briefly empty or taken
        // while typing neither drops nor adopts any.
        if (ImGui::IsItemActivated()) {
          _botRename = {.name = oldName, .slotEntries = {}, .activeSlots = {}};
          for (auto const& [slotName, activeBot] : _activeSlotBots) {
            if (activeBot == oldName) {
              _botRename.activeSlots.push_back(slotName);
            }
          }
          for (int s = 0; s < mochi::isize(prefab.slots); ++s) {
            auto const& controlledBots = prefab.slots[s].controlledBots;
            for (int j = 0; j < mochi::isize(controlledBots); ++j) {
              if (std::string_view(controlledBots[j]) == oldName) {
                _botRename.slotEntries.emplace_back(s, j);
              }
            }
          }
        }
        // A name left empty or taken reverts, since slots name bots.
        if (ImGui::IsItemDeactivatedAfterEdit() &&
            (bot.name.empty() || std::ranges::any_of(bots, [&](auto const& other) {
               return &other != &bot && std::string_view(other.name) == std::string_view(bot.name);
             }))) {
          bot.name = mochi::DynamicString{_botRename.name};
          renamed = true;
        }
        if (renamed) {
          for (auto const& [s, j] : _botRename.slotEntries) {
            if (s < mochi::isize(prefab.slots) &&
                j < mochi::isize(prefab.slots[s].controlledBots)) {
              prefab.slots[s].controlledBots[j] = bot.name;
            }
          }
          for (auto const& slotName : _botRename.activeSlots) {
            _activeSlotBots[slotName] = std::string(bot.name);
          }
          changed = true;
        }
        if (ImGui::AssetSlot("Bot", bot.path, assetManager, _studio, AssetType::Bot, true)) {
          // Reset the initial pose to the newly-assigned bot's default pose (the previous pose
          // belonged to a different bot with a different DOF layout).
          bot.initialPose.clear();
          if (!bot.path.empty()) {
            if (auto* newBotAsset = assetManager.LoadBotAsset(bot.path)) {
              bot.initialPose = newBotAsset->GetBotPrefab().defaultPose;
            }
          }
          assetManager.ResyncReferencer(_sceneAsset);
          changed = true;
        }
        if (auto const& problem = GetReferenceProblem(bot.path); !problem.empty()) {
          ShowWarning(problem);
        }
        changed |= ImGui::DragTransformRT("Spawn Transform", bot.parentFromBot);

        // Initial pose: limit-aware sliders identical to the bot editor's "Default Pose".
        auto* botAsset = assetManager.FindAssetByPath<BotAsset>(mochi::Path{bot.path.c_str()});
        if (botAsset != nullptr && !botAsset->GetBotPrefab()._dofIndices.empty()) {
          ImGui::HoverableSeparatorText("Initial Pose");
          changed |= ImGui::JointPoseEditor(botAsset->GetBotPrefab(), bot.initialPose);
        } else if (!bot.path.empty()) {
          ImGui::TextDisabled("Referenced bot not loaded");
        }

        ImGui::HoverableSeparatorText("Controllers");
        auto const controllerResult = ShowControllers(
            bot,
            botAsset,
            _studio,
            _sceneAsset,
            [this](std::string_view path) -> std::string const& {
              return GetReferenceProblem(path);
            });
        changed |= controllerResult.changed;
        structural |= controllerResult.structural;
        referencesChanged |= controllerResult.referencesChanged;
      }
      ImGui::PopID();
    }
    ImGui::PopID(); // Bots
    if (botToMoveUp > 0) {
      std::swap(bots[botToMoveUp], bots[botToMoveUp - 1]);
      structural = true;
    }
    if (botToMoveDown >= 0 && botToMoveDown < static_cast<int>(bots.size()) - 1) {
      std::swap(bots[botToMoveDown], bots[botToMoveDown + 1]);
      structural = true;
    }
    if (botToDelete >= 0) {
      std::string_view const deletedName{bots[botToDelete].name};
      for (auto& slot : prefab.slots) {
        auto& controlledBots = slot.controlledBots;
        controlledBots.erase(
            std::remove_if(
                controlledBots.begin(),
                controlledBots.end(),
                [&](auto const& name) { return std::string_view(name) == deletedName; }),
            controlledBots.end());
      }
      bots.erase(bots.begin() + botToDelete);
      assetManager.ResyncReferencer(_sceneAsset);
      structural = true;
    }
    if (ImGui::Button(ICON_FA_PLUS "###AddBot")) {
      std::set<std::string> existingNames;
      for (auto const& b : bots) {
        existingNames.insert(std::string(b.name));
      }
      superdex::robotics::BotEntry newBot;
      newBot.name = mochi::DynamicString{MakeUniqueName("Bot", existingNames)};
      bots.push_back(std::move(newBot));
      structural = true;
    }
    ImGui::SameLine();
    ImGui::TextUnformatted("Add Bot");

    ImGui::HoverableSeparatorText("Slots");
    auto const slotResult = ShowSlots(prefab);
    changed |= slotResult.changed;
    structural |= slotResult.structural;
    activeSlotsChanged = slotResult.activeChanged;
  }

  ImGui::EndDisabled();
  ImGui::EndChild();
  ImGui::PopStyleColor(); // ImGuiCol_ChildBg

  // Edit epilogue: reflect edits in the viewport, mark dirty, and record undo state. A bot scene
  // needs no derived "build" step — references resolve live during staging. Active slot bots are a
  // view setting: they restage, but neither dirty the scene nor record undo state.
  if (changed || structural || activeSlotsChanged) {
    if (referencesChanged) {
      assetManager.ResyncReferencer(_sceneAsset);
    }
    RestageBotScene();
  }
  if (changed || structural) {
    _sceneAsset->SetDirty(true);
    _sceneAsset->MarkThumbnailDirty();
    if (structural) {
      GetUndoStack().PushNow(); // discrete op → immediate undo entry
    } else {
      GetUndoStack().MarkEdited(); // continuous → debounced by the app loop
    }
  }

  ImGui::End();
}

} // namespace superdex::studio
