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

#include "meshing/processing_modifiers/processing_modifier.h"

#include <algorithm>
#include <string_view>
#include <utility>

namespace superdex::studio {

MeshProcessingModifier::MeshProcessingModifier(
    std::string displayName,
    ModifierKind kind,
    std::vector<std::unique_ptr<MeshProcessingMethod>> methods)
    : _displayName(std::move(displayName)), _kind(kind), _methods(std::move(methods)) {}

void MeshProcessingModifier::SelectMethod(int index) {
  if (_methods.empty()) {
    return;
  }
  _activeMethod = std::clamp(index, 0, static_cast<int>(_methods.size()) - 1);
}

bool MeshProcessingModifier::SelectMethodByName(std::string_view name) {
  for (std::size_t i = 0; i < _methods.size(); ++i) {
    if (name == _methods[i]->Name()) {
      _activeMethod = static_cast<int>(i);
      return true;
    }
  }
  return false;
}

std::string MeshProcessingModifier::HeaderLabel() const {
  // Multi-method modifiers lead with the selected method (the useful unique identifier when
  // collapsed); single-method modifiers just show their name.
  if (_methods.size() > 1) {
    return std::string(ActiveMethodName()) + " [" + _displayName + "]";
  }
  return _displayName;
}

void MeshProcessingModifier::RemapReferences(std::vector<int> const& oldToNew) {
  for (auto const& method : _methods) {
    method->RemapReferences(oldToNew);
  }
}

std::string MeshProcessingModifier::PropsSignature(ModifierRunContext const& ctx) const {
  // Prefix with the method name so switching method (not just editing params) invalidates the
  // stage.
  return std::string(ActiveMethodName()) + "|" + ActiveMethod().PropsSignature(ctx);
}

} // namespace superdex::studio
