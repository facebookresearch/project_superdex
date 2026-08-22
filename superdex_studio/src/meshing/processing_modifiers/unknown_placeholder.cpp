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

#include "meshing/processing_modifiers/unknown_placeholder.h"

#include <imguios/fonts/icons_font_awesome5.h> // ICON_FA_EXCLAMATION_TRIANGLE
#include <imguios/imguios.h>

#include <picojson/picojson.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace superdex::studio {

namespace {

// The single method of an unknown-placeholder modifier. Holds the captured method name + the raw
// properties JSON (as text) and round-trips them verbatim.
class UnknownMethod : public MeshProcessingMethod {
 public:
  UnknownMethod(std::string methodName, std::string propertiesJson)
      : _methodName(std::move(methodName)), _propertiesJson(std::move(propertiesJson)) {}

  char const* Name() const override {
    return _methodName.c_str();
  }
  void ShowParams(ModifierGuiContext const& /*gui*/) override {
    ImGui::TextUnformatted("Unrecognized modifier/method (perhaps from a newer Studio).");
    ImGui::TextUnformatted("Passed through unchanged; its data is preserved on save.");
  }
  mochi::MeshData Run(
      mochi::MeshData const& input,
      ModifierRunContext const& /*ctx*/,
      mochi::Error& /*error*/) const override {
    return {input}; // passthrough
  }
  void SerializeProps(picojson::value& out) const override {
    std::string err;
    picojson::parse(out, _propertiesJson.begin(), _propertiesJson.end(), &err);
    if (!err.empty()) {
      out = picojson::value(picojson::object()); // captured text unparsable: emit an empty object
    }
  }
  void DeserializeProps(picojson::value const& in) override {
    _propertiesJson = in.serialize(/*prettify=*/false);
  }
  std::string PropsSignature(ModifierRunContext const& /*ctx*/) const override {
    return _propertiesJson;
  }

 private:
  std::string _methodName;
  std::string _propertiesJson;
};

// Modifier hosting a single UnknownMethod; only its header label differs (a warning + the original
// modifier name).
class UnknownModifier : public MeshProcessingModifier {
 public:
  UnknownModifier(
      std::string modifierName,
      std::vector<std::unique_ptr<MeshProcessingMethod>> methods)
      : MeshProcessingModifier(
            std::move(modifierName),
            ModifierKind::Transform,
            std::move(methods)) {}
  std::string HeaderLabel() const override {
    return std::string(ICON_FA_EXCLAMATION_TRIANGLE " ") + DisplayName() + " (unrecognized)";
  }
};

} // namespace

std::unique_ptr<MeshProcessingModifier> MakeUnknownPlaceholderModifier(
    std::string modifierName,
    std::string methodName,
    std::string propertiesJson) {
  std::vector<std::unique_ptr<MeshProcessingMethod>> methods;
  methods.push_back(
      std::make_unique<UnknownMethod>(std::move(methodName), std::move(propertiesJson)));
  return std::make_unique<UnknownModifier>(std::move(modifierName), std::move(methods));
}

} // namespace superdex::studio
