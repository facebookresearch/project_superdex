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

#include <superdex_robotics/ik/bot_ik_config.h>

#include <mochi_core/utils/file_utils.h>
#include <mochi_core/utils/math_utils.h>
#include <picojson/picojson.h>
#include <superdex_robotics/utils/bot_utils.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

using namespace mochi;
using namespace superdex::robotics;

namespace superdex::robotics {
static bool ValidateIKGoalBindingsImpl(BotIKConfig const&, std::string&);

namespace {

bool Fail(std::string& diagnostic, std::string const& message) {
  diagnostic = message;
  return false;
}

bool ValidFrame(TransformRT const& frame) {
  return IsFinite(frame) && NearEqual(Norm(frame.GetRotation().ToReal4()), 1_r, 1e-4_r);
}

bool ValidateCouplingValues(
    Span<BotIKJointCouplingConfig const> couplings,
    std::string& diagnostic) {
  std::set<std::pair<std::string, std::string>> pairs;
  for (int i = 0; i < isize(couplings); ++i) {
    auto const& coupling = couplings[i];
    std::string const prefix = "jointCouplings[" + std::to_string(i) + "]: ";
    if (coupling.jointA.empty() || coupling.jointB.empty() ||
        coupling.jointA == std::string_view(coupling.jointB)) {
      return Fail(diagnostic, prefix + "jointA and jointB must be nonempty and distinct");
    }
    if (!IsFinite(coupling.ratio) || coupling.ratio == 0_r) {
      return Fail(diagnostic, prefix + "ratio must be finite and nonzero");
    }
    if (!IsFinite(coupling.stiffness) || coupling.stiffness < 0_r) {
      return Fail(diagnostic, prefix + "stiffness must be finite and nonnegative");
    }
    std::string a(coupling.jointA), b(coupling.jointB);
    if (b < a) {
      std::swap(a, b);
    }
    if (!pairs.emplace(std::move(a), std::move(b)).second) {
      return Fail(diagnostic, prefix + "duplicate unordered joint pair");
    }
  }
  return true;
}

bool ValidateValues(BotIKConfig const& spec, std::string& diagnostic) {
  diagnostic.clear();
  if (spec.version != kIKConfigVersion) {
    return Fail(
        diagnostic, "Unsupported IK configuration version: " + std::to_string(spec.version));
  }
  if (spec.outputBot.empty()) {
    return Fail(diagnostic, "outputBot: an output bot reference is required");
  }
  for (real const position : spec.outputBotConfiguration) {
    if (!IsFinite(position)) {
      return Fail(diagnostic, "outputBotConfiguration must contain finite joint positions");
    }
  }
  if (!ValidFrame(spec.outputBaseFrameFromInputBaseFrame)) {
    return Fail(
        diagnostic, "outputBaseFrameFromInputBaseFrame must be finite with a unit rotation");
  }
  if (!IsFinite(spec.inputBotScale) || spec.inputBotScale <= 0_r) {
    return Fail(diagnostic, "inputBotScale must be positive and finite");
  }
  for (real const position : spec.inputBotConfiguration) {
    if (!IsFinite(position)) {
      return Fail(diagnostic, "inputBotConfiguration must contain finite joint positions");
    }
  }
  auto const& p = spec.createParams.solverParams;
  if (p.maxIter <= 0 || p.lineSearchMaxIter <= 0) {
    return Fail(diagnostic, "solverParams: maxIter and lineSearchMaxIter must be positive");
  }
  if (p.verbosity < VerbosityLevel::Silent || p.verbosity >= VerbosityLevel::Count) {
    return Fail(diagnostic, "solverParams.verbosity: invalid verbosity level");
  }
  for (auto const& entry :
       {std::pair{"absTol", p.absTol},
        std::pair{"relTol", p.relTol},
        std::pair{"positionErrorThres", p.positionErrorThres},
        std::pair{"rotationErrorThres", p.rotationErrorThres}}) {
    if (!IsFinite(entry.second) || entry.second < 0_r) {
      return Fail(
          diagnostic,
          std::string("solverParams.") + entry.first + ": must be finite and nonnegative");
    }
  }
  if (!std::isfinite(p.maxElapsedTimeSeconds) || p.maxElapsedTimeSeconds < 0) {
    return Fail(diagnostic, "solverParams.maxElapsedTimeSeconds: must be finite and nonnegative");
  }
  for (int i = 0; i < isize(spec.goals); ++i) {
    auto const& goal = spec.goals[i];
    std::string const prefix = "goals[" + std::to_string(i) + "]: ";
    if (goal.name.empty() || goal.outputBotLink.empty()) {
      return Fail(diagnostic, prefix + "name and outputBotLink must not be empty");
    }
    if (!ValidFrame(goal.baseOrInputLinkFrameFromConstraintFrame)) {
      return Fail(
          diagnostic,
          prefix + "baseOrInputLinkFrameFromConstraintFrame must be finite with a unit rotation");
    }
    if (!goal.inputBotLink.empty() && spec.inputBot.empty()) {
      return Fail(diagnostic, prefix + "inputBotLink requires a input bot");
    }
    if (!IsFinite(goal.linkFrameFromConstraintFrame) ||
        !NearEqual(Norm(goal.linkFrameFromConstraintFrame.GetRotation().ToReal4()), 1_r, 1e-4_r)) {
      return Fail(
          diagnostic, prefix + "linkFrameFromConstraintFrame must be finite with a unit rotation");
    }
    if (!IsFinite(goal.positionStiffness) || goal.positionStiffness < 0_r ||
        !IsFinite(goal.rotationStiffness) || goal.rotationStiffness < 0_r) {
      return Fail(diagnostic, prefix + "stiffness must be finite and nonnegative");
    }
  }
  return ValidateCouplingValues(spec.jointCouplings, diagnostic);
}

// Historical tokens belong only to explicit v1-v7 Target and v8 Reference input schemas.
// Mapping changes names, not TransformRT values or frame composition.
void RenameDocumentFields(picojson::object& object, int version, bool toCanonical) {
  if (version >= kIKConfigVersion) {
    return;
  }
  for (auto const& [oldName, newName] :
       {std::pair{"bot", "outputBot"},
        std::pair{"botConfiguration", "outputBotConfiguration"},
        std::pair{version < 8 ? "targetBot" : "referenceBot", "inputBot"},
        std::pair{
            version < 8 ? "baseFromTargetBot" : "baseFromReferenceBot",
            "outputBaseFrameFromInputBaseFrame"},
        std::pair{version < 8 ? "targetBotScale" : "referenceBotScale", "inputBotScale"},
        std::pair{
            version < 8 ? "targetBotConfiguration" : "referenceBotConfiguration",
            "inputBotConfiguration"}}) {
    auto const found = object.find(toCanonical ? oldName : newName);
    if (found != object.end()) {
      object.emplace(toCanonical ? newName : oldName, std::move(found->second));
      object.erase(found);
    }
  }
  if (version < 8) {
    auto const found = object.find(toCanonical ? "targets" : "goals");
    if (found != object.end()) {
      object.emplace(toCanonical ? "goals" : "targets", std::move(found->second));
      object.erase(found);
    }
  }
}

void RenameGoalFields(picojson::object& object, int version, bool toCanonical) {
  if (version >= kIKConfigVersion) {
    return;
  }
  for (auto const& [oldName, newName] :
       {std::pair{"link", "outputBotLink"},
        std::pair{"linkRelativeFrame", "linkFrameFromConstraintFrame"},
        std::pair{version < 8 ? "targetBotLink" : "referenceBotLink", "inputBotLink"},
        std::pair{
            version < 8 ? "targetFrame" : "goalFrame",
            "baseOrInputLinkFrameFromConstraintFrame"}}) {
    auto const found = object.find(toCanonical ? oldName : newName);
    if (found != object.end()) {
      object.emplace(toCanonical ? newName : oldName, std::move(found->second));
      object.erase(found);
    }
  }
}

// Check the JSON types before reflection can coerce strings, truncate integers, or overflow casts.
// The reflected defaults are the schema, including every nested solver field.
bool CheckTypes(
    picojson::value& value,
    picojson::value const& prototype,
    std::string const& path,
    std::string& diagnostic,
    int documentVersion = kIKConfigVersion) {
  if (prototype.is<picojson::object>()) {
    if (!value.is<picojson::object>()) {
      return Fail(diagnostic, path + ": expected an object");
    }
    auto const& schema = prototype.get<picojson::object>();
    for (auto& [key, child] : value.get<picojson::object>()) {
      auto const it = schema.find(key);
      std::string const childPath = path.empty() ? key : path + "." + key;
      if (it == schema.end()) {
        return Fail(diagnostic, childPath + ": unknown field");
      }
      if (!CheckTypes(child, it->second, childPath, diagnostic, documentVersion)) {
        return false;
      }
    }
  } else if (prototype.is<picojson::array>()) {
    if (!value.is<picojson::array>()) {
      return Fail(diagnostic, path + ": expected an array");
    }
    auto const& schema = prototype.get<picojson::array>();
    auto& elements = value.get<picojson::array>();
    bool const goalArray = path == "goals" || path == "targets";
    bool const configurationArray = path == "outputBotConfiguration" ||
        path == "inputBotConfiguration" || path == "botConfiguration" ||
        path == "referenceBotConfiguration" || path == "targetBotConfiguration";
    bool const variable = goalArray || path == "jointCouplings" || configurationArray;
    if (!variable && elements.size() != schema.size()) {
      return Fail(diagnostic, path + ": expected " + std::to_string(schema.size()) + " elements");
    }
    picojson::value elementSchema;
    if (goalArray) {
      SReflect::ToJsonValue(BotIKGoalConfig{}, elementSchema);
      RenameGoalFields(elementSchema.get<picojson::object>(), documentVersion, false);
      if (documentVersion == 1) {
        picojson::value vector;
        SReflect::ToJsonValue(Real3{}, vector);
        elementSchema.get<picojson::object>()["position"] = vector;
        elementSchema.get<picojson::object>()["rotation"] = vector;
      }
    } else if (path == "jointCouplings") {
      SReflect::ToJsonValue(BotIKJointCouplingConfig{}, elementSchema);
    } else if (configurationArray) {
      elementSchema = picojson::value(0.0);
    }
    for (size_t i = 0; i < elements.size(); ++i) {
      if (!CheckTypes(
              elements[i],
              variable ? elementSchema : schema[i],
              path + "[" + std::to_string(i) + "]",
              diagnostic,
              documentVersion)) {
        return false;
      }
    }
  } else if (prototype.is<double>()) {
    if (!value.is<double>()) {
      return Fail(diagnostic, path + ": expected a finite number");
    }
    // get<double>() mutates int64 nodes in Picojson, even through a const reference.
    double const number =
        value.is<int64_t>() ? static_cast<double>(value.get<int64_t>()) : value.get<double>();
    if (!std::isfinite(number)) {
      return Fail(diagnostic, path + ": expected a finite number");
    }
    if ((path == "inputBotScale" || path == "referenceBotScale" || path == "targetBotScale") &&
        (number <= 0 || number > std::numeric_limits<real>::max() || real(number) <= 0_r)) {
      return Fail(diagnostic, path + ": must be positive and in real precision range");
    }
    bool const couplingNumber = path.starts_with("jointCouplings[");
    if (couplingNumber && number != 0 &&
        (std::abs(number) > std::numeric_limits<real>::max() || real(number) == 0_r)) {
      return Fail(diagnostic, path + ": nonzero number must be in real precision range");
    }
    if (couplingNumber && path.find("].ratio") != std::string::npos && number == 0) {
      return Fail(diagnostic, path + ": must be nonzero");
    }
    bool const nonnegative = path.starts_with("createParams.solverParams.") ||
        path.find("Stiffness") != std::string::npos ||
        (couplingNumber && path.find("].stiffness") != std::string::npos);
    if (nonnegative && number < 0) {
      return Fail(diagnostic, path + ": must be nonnegative");
    }
    bool const integer = path == "version" || path == "createParams.solverParams.maxIter" ||
        path == "createParams.solverParams.lineSearchMaxIter";
    if (integer &&
        (std::trunc(number) != number || number < std::numeric_limits<int>::min() ||
         number > std::numeric_limits<int>::max())) {
      return Fail(diagnostic, path + ": expected an in-range integer");
    }
    if (integer) {
      value = picojson::value(static_cast<int64_t>(number));
    }
    if (!integer && path != "createParams.solverParams.maxElapsedTimeSeconds" &&
        std::abs(number) > std::numeric_limits<real>::max()) {
      return Fail(diagnostic, path + ": number exceeds real precision range");
    }
  } else if (prototype.is<bool>()) {
    if (!value.is<bool>()) {
      return Fail(diagnostic, path + ": expected a boolean");
    }
  } else if (prototype.is<std::string>()) {
    if (!value.is<std::string>()) {
      return Fail(diagnostic, path + ": expected a string");
    }
  } else {
    return Fail(diagnostic, path + ": unsupported schema type");
  }
  return true;
}

} // namespace

static bool ValidateIKGoalBindingsImpl(BotIKConfig const& spec, std::string& diagnostic) {
  diagnostic.clear();
  std::unordered_set<std::string> names;
  std::unordered_set<std::string> links;
  for (auto const& goal : spec.goals) {
    if (!names.insert(std::string(goal.name)).second) {
      if (!diagnostic.empty()) {
        diagnostic += "; ";
      }
      diagnostic += "Duplicate goal name: " + std::string(goal.name);
    }
    if (!links.insert(std::string(goal.outputBotLink)).second) {
      if (!diagnostic.empty()) {
        diagnostic += "; ";
      }
      diagnostic += "Duplicate goal link: " + std::string(goal.outputBotLink);
    }
  }
  return diagnostic.empty();
}

static bool ValidateIKConfigImpl(BotIKConfig const& spec, std::string& diagnostic) {
  return ValidateValues(spec, diagnostic) && ValidateIKGoalBindingsImpl(spec, diagnostic);
}

static std::string SerializeIKConfigImpl(BotIKConfig const& spec) {
  return SReflect::ToJsonString(spec, true);
}

static bool ParseIKConfigImpl(std::string const& text, BotIKConfig& out, std::string& diagnostic) {
  diagnostic.clear();
  picojson::value value;
  std::string::const_iterator end;
  try {
    end = picojson::parse(value, text.begin(), text.end(), &diagnostic);
  } catch (std::overflow_error const&) {
    return Fail(diagnostic, "IK configuration JSON contains a non-finite number");
  }
  if (!diagnostic.empty()) {
    return false;
  }
  if (!std::all_of(end, text.end(), [](unsigned char c) { return std::isspace(c) != 0; })) {
    return Fail(diagnostic, "Unexpected content after IK configuration JSON");
  }
  BotIKConfig candidate;
  if (!value.is<picojson::object>()) {
    return Fail(diagnostic, "expected an IK configuration object");
  }
  auto& object = value.get<picojson::object>();
  auto const version = object.find("version");
  bool const legacy = version != object.end() && version->second.is<double>() &&
      version->second.get<double>() == 1.0;
  bool const versionTwo = version != object.end() && version->second.is<double>() &&
      version->second.get<double>() == 2.0;
  bool const versionThree = version != object.end() && version->second.is<double>() &&
      version->second.get<double>() == 3.0;
  bool const versionFour = version != object.end() && version->second.is<double>() &&
      version->second.get<double>() == 4.0;
  bool const versionFive = version != object.end() && version->second.is<double>() &&
      version->second.get<double>() == 5.0;
  bool const versionSix = version != object.end() && version->second.is<double>() &&
      version->second.get<double>() == 6.0;
  bool const versionSeven = version != object.end() && version->second.is<double>() &&
      version->second.get<double>() == 7.0;
  bool const versionEight = version != object.end() && version->second.is<double>() &&
      version->second.get<double>() == 8.0;
  int const documentVersion = legacy ? 1
      : versionTwo                   ? 2
      : versionThree                 ? 3
      : versionFour                  ? 4
      : versionFive                  ? 5
      : versionSix                   ? 6
      : versionSeven                 ? 7
      : versionEight                 ? 8
                                     : kIKConfigVersion;
  bool const oldNames = documentVersion < kIKConfigVersion;
  if ((legacy || versionTwo || versionThree || versionFour || versionFive) &&
      object.count("jointCouplings")) {
    return Fail(diagnostic, "jointCouplings requires IK configuration version 6");
  }
  if ((legacy || versionTwo || versionThree || versionFour) && object.count("targetBotRecording")) {
    return Fail(diagnostic, "targetBotRecording requires IK configuration version 5");
  }
  if ((legacy || versionTwo || versionThree) && object.count("botConfiguration")) {
    return Fail(diagnostic, "botConfiguration requires IK configuration version 4");
  }
  if ((legacy || versionTwo) && object.count("targetBotScale")) {
    return Fail(diagnostic, "targetBotScale requires IK configuration version 3");
  }
  picojson::value schema;
  SReflect::ToJsonValue(candidate, schema);
  if (oldNames) {
    RenameDocumentFields(schema.get<picojson::object>(), documentVersion, false);
  }
  // Recording selection is no longer authored state. Only v5/v6 admitted this legacy key;
  // validate its original type before dropping it, without relaxing any other unknown fields.
  bool const droppedRecording = (versionFive || versionSix) && object.count("targetBotRecording");
  if (versionFive || versionSix) {
    schema.get<picojson::object>()["targetBotRecording"] = picojson::value("");
  }
  if (!CheckTypes(value, schema, "", diagnostic, documentVersion)) {
    return false;
  }
  if (droppedRecording) {
    object.erase("targetBotRecording");
  }
  char const* const outputBotField = oldNames ? "bot" : "outputBot";
  if (object.find("version") == object.end() || object.find(outputBotField) == object.end()) {
    return Fail(diagnostic, std::string("version and ") + outputBotField + " are required");
  }
  // The original schema has checked every object/array, numeric range, and unknown key.
  // Each schema admits only one vocabulary, so even equal-valued old/new mixtures reject.
  if (oldNames) {
    RenameDocumentFields(object, documentVersion, true);
    auto const goals = object.find("goals");
    if (goals != object.end()) {
      for (auto& item : goals->second.get<picojson::array>()) {
        RenameGoalFields(item.get<picojson::object>(), documentVersion, true);
      }
    }
    object["version"] = picojson::value(int64_t(kIKConfigVersion));
  }
  if (legacy) {
    auto const goals = object.find("goals");
    if (goals != object.end()) {
      for (auto& item : goals->second.get<picojson::array>()) {
        auto& fields = item.get<picojson::object>();
        bool const oldPose = fields.count("position") || fields.count("rotation");
        if (oldPose && fields.count("baseOrInputLinkFrameFromConstraintFrame")) {
          return Fail(
              diagnostic, "A legacy target cannot contain both targetFrame and legacy pose fields");
        }
        if (!fields.count("baseOrInputLinkFrameFromConstraintFrame")) {
          Real3 position{}, rotation{};
          for (auto const& [name, destination] :
               {std::pair{"position", &position}, std::pair{"rotation", &rotation}}) {
            auto const found = fields.find(name);
            if (found != fields.end()) {
              auto& numbers = found->second.get<picojson::array>();
              for (int i = 0; i < 3; ++i) {
                (*destination)[i] = real(numbers[i].get<double>());
              }
            }
          }
          TransformRT const frame{Quaternion::FromRotationVector(rotation), position};
          if (!ValidFrame(frame)) {
            return Fail(diagnostic, "Legacy position and rotation must define a finite pose");
          }
          SReflect::ToJsonValue(frame, fields["baseOrInputLinkFrameFromConstraintFrame"]);
        }
        fields.erase("position");
        fields.erase("rotation");
      }
    }
  }
  if (legacy || versionTwo) {
    object["inputBotScale"] = picojson::value(1.0);
  }
  int issues = 0;
  SReflect::FromJsonValue(candidate, value, SReflect::DeserializeFlags::Default, issues);
  if (issues != 0) {
    return Fail(
        diagnostic,
        "IK configuration contains malformed reflected fields (including solver verbosity)");
  }
  if (!ValidateValues(candidate, diagnostic)) {
    return false;
  }
  out = std::move(candidate);
  (void)ValidateIKGoalBindingsImpl(out, diagnostic);
  if (droppedRecording) {
    if (!diagnostic.empty()) {
      diagnostic += "; ";
    }
    diagnostic +=
        "Legacy targetBotRecording ignored; recording selection is no longer an IK document field";
  }
  return true;
}

static bool LoadIKConfigImpl(std::string_view path, BotIKConfig& out, std::string& diagnostic) {
  Error error;
  auto const text = ReadFileString(std::string(path), error);
  if (!error.IsOK()) {
    return Fail(diagnostic, error.GetDescription());
  }
  return ParseIKConfigImpl(text, out, diagnostic);
}

static bool
SaveIKConfigImpl(std::string_view path, BotIKConfig const& spec, std::string& diagnostic) {
  if (!ValidateIKConfigImpl(spec, diagnostic)) {
    return false;
  }
  auto const text = SerializeIKConfigImpl(spec);
  Error error;
  WriteFile(std::string(path), Span<char const>{text.data(), text.size()}, error);
  if (!error.IsOK()) {
    return Fail(diagnostic, error.GetDescription());
  }
  return true;
}

static std::string Utf8Path(std::filesystem::path const& path) {
  auto const text = path.generic_u8string();
  return {reinterpret_cast<char const*>(text.data()), text.size()};
}

static std::string ResolveIKBotPathImpl(std::string_view doc, std::string_view ref) {
  if (ref.empty()) {
    return {};
  }
  auto const path = std::filesystem::u8path(doc).parent_path() / std::filesystem::u8path(ref);
  return Utf8Path(std::filesystem::absolute(path).lexically_normal());
}

static std::string MakeIKBotReferenceImpl(std::string_view doc, std::string_view bot) {
  auto const full = std::filesystem::u8path(bot).lexically_normal();
  auto const relative = full.lexically_relative(std::filesystem::u8path(doc).parent_path());
  return Utf8Path(relative.empty() ? full : relative);
}

TransformRT IKGoalConstraintTransform(BotIKGoalConfig const& goal) {
  return goal.baseOrInputLinkFrameFromConstraintFrame;
}

void SetIKGoalConstraintTransform(BotIKGoalConfig& goal, TransformRT const& transform) {
  goal.baseOrInputLinkFrameFromConstraintFrame = transform;
}

TransformRT IKGoalLocalConstraintTransform(
    BotIKGoalConfig const& goal,
    TransformRT const& anchor,
    TransformRT const& worldFrameFromConstraintFrame,
    real inputBotScale) {
  TransformRT local = Invert(anchor) * worldFrameFromConstraintFrame;
  if (!goal.inputBotLink.empty()) {
    local.SetTranslation(local.GetTranslation() / inputBotScale);
  }
  return local;
}

void ComputeIKInputBotTransforms(
    BotPrefab const& inputBot,
    Span<real const> positions,
    TransformRT const& worldFrameFromInputBaseFrame,
    DynamicArray<TransformRT>& worldFromLinks,
    Error& error) {
  ComputeIKInputBotTransforms(
      inputBot, positions, worldFrameFromInputBaseFrame, 1_r, worldFromLinks, error);
}

void ComputeIKInputBotTransforms(
    BotPrefab const& inputBot,
    Span<real const> positions,
    TransformRT const& worldFrameFromInputBaseFrame,
    real inputBotScale,
    DynamicArray<TransformRT>& worldFromLinks,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF_NOT(
      IsFinite(inputBotScale) && inputBotScale > 0_r,
      error,
      "Input bot scale must be positive and finite.");
  MOCHI_ERROR_IF_NOT(
      ValidFrame(worldFrameFromInputBaseFrame), error, "Invalid input bot root frame.");
  if (positions.empty()) {
    positions = MakeConstSpan(inputBot.defaultPose);
  }
  MOCHI_ERROR_IF_NOT(
      isize(positions) == inputBot._numDofs, error, "Input bot DOF count does not match.");
  for (real const position : positions) {
    MOCHI_ERROR_IF_NOT(IsFinite(position), error, "Input bot joint positions must be finite.");
  }
  MOCHI_ERROR_RETURN(error);
  DynamicArray<TransformRT> links;
  ComputeLinkTransformsFromPose(
      inputBot, positions, links, LinkTransformSpace::RootFromParent, error);
  MOCHI_ERROR_RETURN(error);
  for (auto& link : links) {
    // Shared FK is unscaled; apply native size and retarget scale before root placement.
    link.SetTranslation((link.GetTranslation() * inputBot.scale) * inputBotScale);
    link = worldFrameFromInputBaseFrame * link;
    MOCHI_ERROR_IF_NOT(IsFinite(link), error, "Non-finite input bot link transform.");
  }
  MOCHI_ERROR_RETURN(error);
  worldFromLinks = std::move(links);
}

void ComputeIKGoalWorldConstraintTransform(
    BotIKGoalConfig const& goal,
    TransformRT const& worldFrameFromOutputBaseFrame,
    BotPrefab const* inputBot,
    Span<TransformRT const> worldFrameFromInputLinks,
    TransformRT& worldFrameFromConstraintFrame,
    Error& error) {
  ComputeIKGoalWorldConstraintTransform(
      goal,
      worldFrameFromOutputBaseFrame,
      inputBot,
      worldFrameFromInputLinks,
      1_r,
      worldFrameFromConstraintFrame,
      error);
}

void ComputeIKGoalWorldConstraintTransform(
    BotIKGoalConfig const& goal,
    TransformRT const& worldFrameFromOutputBaseFrame,
    BotPrefab const* inputBot,
    Span<TransformRT const> worldFrameFromInputLinks,
    real inputBotScale,
    TransformRT& worldFrameFromConstraintFrame,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF_NOT(
      IsFinite(inputBotScale) && inputBotScale > 0_r,
      error,
      "Input bot scale must be positive and finite.");
  MOCHI_ERROR_IF_NOT(
      ValidFrame(goal.baseOrInputLinkFrameFromConstraintFrame), error, "Invalid goal frame.");
  TransformRT anchor = worldFrameFromOutputBaseFrame;
  if (!goal.inputBotLink.empty()) {
    MOCHI_ERROR_IF(inputBot == nullptr, error, "A input bot is required for inputBotLink.");
    MOCHI_ERROR_RETURN(error);
    MOCHI_ERROR_IF_NOT(
        worldFrameFromInputLinks.size() == inputBot->links.size(),
        error,
        "Input bot link transform count does not match.");
    MOCHI_ERROR_RETURN(error);
    int index = kIndexNone;
    for (int i = 0; i < isize(inputBot->links); ++i) {
      if (inputBot->links[i].name == std::string_view(goal.inputBotLink)) {
        index = i;
        break;
      }
    }
    MOCHI_ERROR_IF_NOT(index != kIndexNone, error, "Unknown input bot link.");
    MOCHI_ERROR_RETURN(error);
    anchor = worldFrameFromInputLinks[index];
  }
  MOCHI_ERROR_IF_NOT(ValidFrame(anchor), error, "Invalid goal anchor frame.");
  MOCHI_ERROR_RETURN(error);
  TransformRT local = goal.baseOrInputLinkFrameFromConstraintFrame;
  if (!goal.inputBotLink.empty()) {
    local.SetTranslation(local.GetTranslation() * inputBotScale);
  }
  TransformRT const result = anchor * local;
  MOCHI_ERROR_IF_NOT(IsFinite(result), error, "Non-finite resolved goal pose.");
  MOCHI_ERROR_RETURN(error);
  worldFrameFromConstraintFrame = result;
}

static bool InitializeIKGoalsImpl(
    BotPrefab const& preparedOutputBot,
    BotIKConfig& spec,
    std::string& diagnostic) {
  diagnostic.clear();
  Error error;
  Validate(preparedOutputBot, nullptr, error);
  if (!error.IsOK()) {
    return Fail(diagnostic, error.GetDescription());
  }
  if (isize(preparedOutputBot.defaultPose) != preparedOutputBot._numDofs) {
    return Fail(diagnostic, "Prepared bot default pose has inconsistent DOF count");
  }
  // FK consumes the prepared rest transforms in parent-before-child order.
  for (int i = 0; i < isize(preparedOutputBot.links); ++i) {
    auto const& link = preparedOutputBot.links[i];
    if (link._index != i || (i > 0 && (link.parentLink < 0 || link.parentLink >= i)) ||
        link._parentFromLink !=
            (i == 0 ? TransformRT{} : preparedOutputBot.joints[i].parentLinkFromJoint)) {
      return Fail(diagnostic, "Leaf initialization requires prepared link topology and transforms");
    }
  }
  for (real const position : preparedOutputBot.defaultPose) {
    if (!IsFinite(position)) {
      return Fail(diagnostic, "Default joint positions must be finite");
    }
  }
  DynamicArray<TransformRT> rootFromLinks;
  ComputeLinkTransformsFromPose(
      preparedOutputBot,
      preparedOutputBot.defaultPose,
      rootFromLinks,
      LinkTransformSpace::RootFromParent,
      error);
  if (!error.IsOK()) {
    return Fail(diagnostic, error.GetDescription());
  }
  DynamicArray<BotIKGoalConfig> goals;
  for (int const index : FindLeafLinkIndices(preparedOutputBot)) {
    if (!IsFinite(rootFromLinks[index])) {
      return Fail(diagnostic, "Default pose produced a non-finite leaf transform");
    }
    BotIKGoalConfig goal;
    goal.name = std::string(std::string_view(preparedOutputBot.links[index].name));
    goal.outputBotLink = goal.name;
    SetIKGoalConstraintTransform(goal, rootFromLinks[index]);
    goals.push_back(std::move(goal));
  }
  spec.goals = std::move(goals);
  return true;
}

mochi::DynamicString SerializeIKConfig(BotIKConfig const& config) {
  return mochi::DynamicString(SerializeIKConfigImpl(config));
}
mochi::DynamicString ResolveIKBotPath(std::string_view doc, std::string_view ref) {
  return mochi::DynamicString(ResolveIKBotPathImpl(doc, ref));
}
mochi::DynamicString MakeIKBotReference(std::string_view doc, std::string_view bot) {
  return mochi::DynamicString(MakeIKBotReferenceImpl(doc, bot));
}
bool LoadIKConfig(std::string_view path, BotIKConfig& config, mochi::DynamicString& diagnostic) {
  std::string message;
  bool const result = LoadIKConfigImpl(path, config, message);
  diagnostic.assign(message.begin(), message.end());
  return result;
}
bool SaveIKConfig(
    std::string_view path,
    BotIKConfig const& config,
    mochi::DynamicString& diagnostic) {
  std::string message;
  bool const result = SaveIKConfigImpl(path, config, message);
  diagnostic.assign(message.begin(), message.end());
  return result;
}
bool ParseIKConfig(std::string_view text, BotIKConfig& config, mochi::DynamicString& diagnostic) {
  std::string message;
  bool const result = ParseIKConfigImpl(std::string(text), config, message);
  diagnostic.assign(message.begin(), message.end());
  return result;
}
bool ValidateIKConfig(BotIKConfig const& config, mochi::DynamicString& diagnostic) {
  std::string message;
  bool const result = ValidateIKConfigImpl(config, message);
  diagnostic.assign(message.begin(), message.end());
  return result;
}
bool ValidateIKGoalBindings(BotIKConfig const& config, mochi::DynamicString& diagnostic) {
  std::string message;
  bool const result = ValidateIKGoalBindingsImpl(config, message);
  diagnostic.assign(message.begin(), message.end());
  return result;
}
DynamicArray<BotIKJointCouplingParams> ToBotIKJointCouplingParams(
    Span<BotIKJointCouplingConfig const> couplings) {
  DynamicArray<BotIKJointCouplingParams> result;
  result.reserve(couplings.size());
  for (auto const& coupling : couplings) {
    BotIKJointCouplingParams params;
    params.jointA = coupling.jointA;
    params.jointB = coupling.jointB;
    params.ratio = coupling.ratio;
    params.stiffness = coupling.stiffness;
    params.enabled = coupling.enabled;
    result.push_back(std::move(params));
  }
  return result;
}

bool ValidateIKJointCouplings(
    BotPrefab const& bot,
    Span<BotIKJointCouplingConfig const> couplings,
    DynamicString& diagnostic) {
  std::string message;
  if (!ValidateCouplingValues(couplings, message)) {
    diagnostic = message;
    return false;
  }
  auto const params = ToBotIKJointCouplingParams(couplings);
  for (int i = 0; i < isize(couplings); ++i) {
    auto const& coupling = couplings[i];
    if (!coupling.enabled || coupling.stiffness == 0_r) {
      continue;
    }
    std::string const prefix = "jointCouplings[" + std::to_string(i) + "] (" +
        std::string(coupling.jointA) + ", " + std::string(coupling.jointB) + "): ";
    Error error;
    BotIK::ValidateJointCouplings(
        bot, Span<BotIKJointCouplingParams const>{params.data() + i, 1}, error);
    if (!error.IsOK()) {
      diagnostic = prefix + error.GetDescription();
      return false;
    }
    // Runtime validation already established that each active name resolves uniquely.
    auto const a = std::find_if(bot.joints.begin(), bot.joints.end(), [&](auto const& joint) {
      return joint.name == std::string_view(coupling.jointA);
    });
    auto const b = std::find_if(bot.joints.begin(), bot.joints.end(), [&](auto const& joint) {
      return joint.name == std::string_view(coupling.jointB);
    });
    if (a->type != b->type) {
      diagnostic = prefix + "active document joints must have the same kind";
      return false;
    }
  }
  diagnostic.clear();
  return true;
}

bool InitializeIKGoals(
    BotPrefab const& outputBot,
    BotIKConfig& config,
    mochi::DynamicString& diagnostic) {
  std::string message;
  bool const result = InitializeIKGoalsImpl(outputBot, config, message);
  diagnostic.assign(message.begin(), message.end());
  return result;
}

} // namespace superdex::robotics
