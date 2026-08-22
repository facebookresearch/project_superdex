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

#include "io/importer.h"

#include <mochi_renderer/path.h>
#include <superdex_robotics/superdex_robotics.h>

#include <filesystem>
#include <string>
#include <vector>

namespace superdex::studio {

// Which resolved mesh a row draws its source from.
enum class UrdfMeshSource { Visual, Collision };

// Operation applied to a link's model in either tab.
enum class UrdfMeshOperation { Convert, Copy, Ignore };

// One mesh slot (visual or collision) of a link, snapshotted at import-begin.
//
// `path` is the resolved absolute mesh on disk, or empty when the URDF either has
// no mesh in this slot or references one that does not resolve. `ref` is the raw
// `<mesh filename="...">` reference verbatim (e.g. an unresolvable "package://" URI),
// empty only when the slot has no mesh. Keeping the raw reference separate lets the
// importer tell "referenced but missing" from "absent" and show the original text
// (unlike mochi::Path, which would normalize an unresolvable URI against the CWD).
struct UrdfMeshRef {
  mochi::Path path;
  std::string ref;

  // The URDF references a mesh for this slot (resolved or not).
  bool Referenced() const {
    return !ref.empty();
  }
  // The referenced mesh resolved to a file on disk.
  bool Exists() const {
    return !path.IsEmpty();
  }
  // The URDF references a mesh that cannot be found on disk (its task would fail).
  bool Missing() const {
    return Referenced() && !Exists();
  }

  // Filename for display: from the resolved path when it exists, else from the raw ref.
  std::string DisplayName() const {
    return Exists() ? path.GetFilename() : std::filesystem::path(ref).filename().string();
  }
  // Full reference for tooltips: the resolved absolute path when it exists, else the raw
  // ref shown verbatim (never normalized against the CWD).
  std::string DisplayRef() const {
    return Exists() ? path.ToString() : ref;
  }
};

// Per-link import choices, one entry per link in the loaded prefab. Both tabs
// share this entry: the Render tab drives renderSource/renderOp and the Mochi
// tab drives mochiSource/mochiOp.
struct UrdfLinkImportEntry {
  std::string linkName;
  // Resolved source meshes (and their raw references) snapshotted at import-begin.
  UrdfMeshRef visual;
  UrdfMeshRef collision;
  // Render Models tab selection.
  UrdfMeshSource renderSource = UrdfMeshSource::Visual;
  UrdfMeshOperation renderOp = UrdfMeshOperation::Ignore;
  // Mochi Models tab selection.
  UrdfMeshSource mochiSource = UrdfMeshSource::Collision;
  UrdfMeshOperation mochiOp = UrdfMeshOperation::Ignore;

  // The mesh slot a tab selection points at.
  UrdfMeshRef const& MeshRef(UrdfMeshSource source) const {
    return source == UrdfMeshSource::Visual ? visual : collision;
  }
};

struct UrdfImportOptions {
  std::string name;
  std::vector<UrdfLinkImportEntry> links;
  // Subdirectories (relative to the import destination) for the produced render
  // and mochi model files. Blank means write directly into the destination root.
  std::string renderDir;
  std::string mochiDir;
  // Remesh mochi (collision) models when converting them to .h5, producing a clean
  // watertight surface. Ignored when the mochi operation is Copy All (no conversion).
  bool mochiRemesh = true;
  // Bake a signed distance field into mochi (collision) models when converting them
  // to .h5. Ignored when the mochi operation is Copy All (no conversion).
  bool mochiBakeSdf = true;
};

class UrdfImporter : public Importer {
 public:
  explicit UrdfImporter(SuperDexStudio* studio);

  std::string const& GetDisplayName() const override;
  std::vector<std::string> const& GetExtensions() const override;
  ImU32 GetColor() const override;

 protected:
  void OnBeginImport(mochi::Path const& path, mochi::Path const& destDir) override;
  void OnShowImportOptions() override;
  bool OnFinishImport(std::vector<AsyncTask>& tasks, FinalizeFn& finalize) override;
  bool CanImport() override;
  ImVec2 GetModalSize() const override;
  // If the user left BOTH the render and mochi destinations at their defaults (_render / _mochi),
  // scaffolds all three standard sibling folders; otherwise the user has chosen a custom layout, so
  // scaffolds none and only the folders the import actually writes to are created (lazily).
  std::vector<std::string> SiblingFoldersToEnsure() const override;

 private:
  std::string _displayName;
  std::vector<std::string> _extensions;

  UrdfImportOptions _options;
  superdex::robotics::BotPrefab _botPrefab;

  // Transient state for the per-tab blanket radio controls (-1 = no choice yet).
  int _renderSourceChoice = -1;
  int _renderOpChoice = -1;
  int _mochiSourceChoice = -1;
  int _mochiOpChoice = -1;

  // False if the URDF failed to load or produced no link data; the dialog then
  // shows a warning and disables Import.
  bool _prefabValid = false;

  // True while the Name field would overwrite an existing bot in the destination.
  // Set each frame by OnShowImportOptions and gates Import in CanImport.
  bool _nameConflicts = false;
};

} // namespace superdex::studio
