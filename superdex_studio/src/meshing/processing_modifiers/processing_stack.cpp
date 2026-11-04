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

#include "meshing/processing_modifiers/processing_stack.h"

#include <mochi_core/geometry/model_data.h>
#include <mochi_core/geometry/model_utils.h> // model::Validate

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <string>
#include <string_view>
#include <utility>

namespace superdex::studio {

bool SameFilePath(std::string_view a, std::string_view b) {
  if (a.empty() || b.empty()) {
    return false; // an unset path collides with nothing
  }
  // The generic (forward slash) form, so a derived path built with native separators still matches
  // an equivalent one the user typed or that round-tripped through the pipeline JSON.
  std::string const normalA = std::filesystem::path(a).lexically_normal().generic_string();
  std::string const normalB = std::filesystem::path(b).lexically_normal().generic_string();
  return std::ranges::equal(normalA, normalB, [](unsigned char x, unsigned char y) {
    return std::tolower(x) == std::tolower(y);
  });
}

std::vector<std::size_t> BuildGenerationChain(ModifierStack const& stack, std::size_t index) {
  // Climb from @p index to the nearest preceding source, collecting the enabled providers (each
  // modifier's input is the nearest enabled modifier before it). Returns source-first; empty if no
  // source is reachable upstream.
  std::vector<std::size_t> chain;
  if (index >= stack.size()) {
    return chain;
  }
  std::size_t cur = index;
  while (true) {
    chain.push_back(cur);
    if (stack[cur]->Kind() == ModifierKind::Source) {
      break;
    }
    bool foundProvider = false;
    for (std::size_t j = cur; j-- > 0;) {
      if (stack[j]->enabled) {
        cur = j;
        foundProvider = true;
        break;
      }
    }
    if (!foundProvider) {
      return {}; // no upstream provider / source: cannot build
    }
  }
  if (stack[chain.back()]->Kind() != ModifierKind::Source) {
    return {};
  }
  std::ranges::reverse(chain);
  return chain;
}

std::vector<std::size_t> BuildFullGenerationChain(ModifierStack const& stack) {
  // Every enabled modifier from the first enabled source on, in stack order. A source ignores its
  // input, so a stack that holds SEVERAL source->...->export segments is already a valid execution
  // order exactly as it stands -- each later source just restarts the data flow from its own file
  // (typically the one the export above it writes). Climbing to the nearest source instead, the way
  // BuildGenerationChain does for a single modifier's Generate, would cover only the last segment.
  std::vector<std::size_t> chain;
  for (std::size_t i = 0; i < stack.size(); ++i) {
    if (!stack[i]->enabled) {
      continue;
    }
    if (chain.empty() && stack[i]->Kind() != ModifierKind::Source) {
      continue; // ahead of the first source: nothing to build from
    }
    chain.push_back(i);
  }
  return chain;
}

int ReferenceModifierIndex(ModifierStack const& stack, std::size_t index) {
  int const refIndex = stack[index]->ReferenceIndex();
  if (refIndex >= 0) {
    // An explicit reference is valid only when it points strictly upstream. If it does not (e.g.
    // the modifier was moved above its reference), return -1 so the edge flip gets no reference
    // mesh and fails generation -- rather than silently fitting toward a different source.
    return refIndex < static_cast<int>(index) ? refIndex : -1;
  }
  // Preceding-source sentinel: walk up to the nearest preceding source.
  for (std::size_t j = index; j-- > 0;) {
    if (stack[j]->Kind() == ModifierKind::Source) {
      return static_cast<int>(j);
    }
  }
  return -1;
}

std::string StackSourceFilePath(ModifierStack const& stack, ModifierRunContext const& ctx) {
  return stack.empty() ? std::string() : stack.front()->SourceFilePath(ctx);
}

void RefreshAutoExportPaths(ModifierStack& stack, std::string const& sourceFilePath) {
  for (auto const& modifier : stack) {
    modifier->RefreshAutoExportPath(sourceFilePath);
  }
}

void RedirectExports(ModifierStack& stack, std::filesystem::path const& folder) {
  for (auto const& modifier : stack) {
    std::string const path = modifier->ExportPath();
    if (modifier->enabled && modifier->ProvidesFileExport() && !path.empty()) {
      modifier->OverrideExportPath((folder / std::filesystem::path(path).filename()).string());
    }
  }
}

std::vector<bool> FindCollidingExportPaths(ModifierStack const& stack) {
  std::vector<bool> collides(stack.size(), false);
  // Quadratic, but over a handful of export modifiers in a stack drawn once per frame. Compares the
  // resolved paths rather than which ones are on Auto, so two hand-picked paths that happen to
  // match are caught too. Disabled modifiers write nothing, so they are left out.
  for (std::size_t i = 0; i < stack.size(); ++i) {
    if (!stack[i]->enabled || !stack[i]->ProvidesFileExport()) {
      continue;
    }
    std::string const pathI = stack[i]->ExportPath();
    for (std::size_t j = i + 1; j < stack.size(); ++j) {
      if (!stack[j]->enabled || !stack[j]->ProvidesFileExport()) {
        continue;
      }
      if (SameFilePath(pathI, stack[j]->ExportPath())) {
        collides.at(i) = true;
        collides.at(j) = true;
      }
    }
  }
  return collides;
}

StackRun::StackRun(
    ModifierStack const& stack,
    std::vector<std::size_t> chain,
    ModifierRunContext ctx,
    std::function<StageMesh(std::size_t modifierIndex)> const& currentOutput,
    std::function<mochi::MeshData(std::size_t modifierIndex)> const& currentReferenceMesh,
    std::function<mochi::MeshData(mochi::MeshData const& output)> referenceMeshFromOutput)
    : _referenceMeshFromOutput(std::move(referenceMeshFromOutput)), _ctx(std::move(ctx)) {
  std::size_t const n = chain.size();
  _modifiers.reserve(n);
  _outputs.reserve(n);
  _references.resize(n);
  for (std::size_t p = 0; p < n; ++p) {
    _modifiers.push_back(stack[chain[p]].get());
    _outputs.push_back(currentOutput(chain[p]));
    if (!_modifiers[p]->NeedsReferenceMesh()) {
      continue;
    }
    int const refModIdx = ReferenceModifierIndex(stack, chain[p]);
    Reference& ref = _references[p];
    for (std::size_t q = 0; q < n; ++q) {
      if (static_cast<int>(chain[q]) == refModIdx) {
        ref.chainPos = static_cast<int>(q);
        break;
      }
    }
    if (ref.chainPos < 0 && refModIdx >= 0) {
      ref.currentMesh = currentReferenceMesh(static_cast<std::size_t>(refModIdx));
    }
  }
}

bool StackRun::Generate(std::size_t p, mochi::Error& error) {
  MeshProcessingModifier const& modifier = *_modifiers[p];
  ModifierRunContext runCtx = _ctx;
  if (modifier.NeedsReferenceMesh()) {
    Reference const& ref = _references[p];
    runCtx.referenceMesh = ref.chainPos >= 0
        ? _referenceMeshFromOutput(_outputs[static_cast<std::size_t>(ref.chainPos)].mesh)
        : ref.currentMesh;
  }
  StageMesh out = modifier.Run(Input(p), runCtx, error);
  if (error.IsOK() && out.mesh.nodesPerElement == 4) {
    // Rejected here so that extracting the boundary surface later (DisplaySurface) cannot fail.
    mochi::ModelDataView model;
    model.mesh = out.mesh;
    mochi::model::Validate(model, error);
  }
  if (!error.IsOK() || out.mesh.GetNumElements() == 0) {
    // Leave no output, so the next stage receives an empty input and stops too (see
    // MeshProcessingModifier::Run) instead of running on this stage's stale mesh.
    _outputs[p] = StageMesh{};
    return false;
  }
  _outputs[p] = std::move(out);
  return true;
}

bool StackRun::Export(std::size_t p, mochi::Error& error) const {
  if (_outputs[p].mesh.GetNumElements() == 0) {
    MOCHI_ERROR_SET(error, "Cannot export a modifier that did not generate successfully.");
    return false;
  }
  _modifiers[p]->SaveToFile(Input(p).mesh, error);
  return error.IsOK();
}

StageMesh const& StackRun::Input(std::size_t p) const {
  static StageMesh const kNoInput;
  return p > 0 ? _outputs[p - 1] : kNoInput;
}

} // namespace superdex::studio
