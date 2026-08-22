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

#include "assets/asset.h"
#include "core/async_task.h"

#include <mochi_renderer/path.h>

#include <imgui.h>

#include <functional>
#include <string>
#include <vector>

namespace superdex::studio {

class SuperDexStudio;

// Abstract base for asset importers.
//
// An importer advertises the file extensions it can handle and a tile color, and
// owns a modal options dialog that is shown before the import runs. The base
// class implements the modal shell (deferred OpenPopup + centered
// BeginPopupModal + Import/Cancel buttons); subclasses fill in the options body
// (OnShowImportOptions) and the import action (OnFinishImport).
class Importer {
 public:
  explicit Importer(SuperDexStudio* studio) : _studio(studio) {}
  virtual ~Importer() = default;

  Importer(Importer const&) = delete;
  Importer& operator=(Importer const&) = delete;
  Importer(Importer&&) = delete;
  Importer& operator=(Importer&&) = delete;

  // Human-readable name, used for the menu label and modal title (e.g. "URDF").
  virtual std::string const& GetDisplayName() const = 0;

  // Handled extensions, lowercase and dot-prefixed (e.g. {".urdf"}).
  // Used for routing and file-dialog filters.
  virtual std::vector<std::string> const& GetExtensions() const = 0;

  // Color used to tint importable file tiles in the asset browser.
  virtual ImU32 GetColor() const = 0;

  // Begin importing `path` into `destDir`: store them, reset transient options,
  // and open the modal.
  void BeginImport(mochi::Path const& path, mochi::Path const& destDir);

  // Draw the options modal for one frame.
  // Returns true while the modal is still open; false once it has closed.
  bool ShowModalWindow();

 protected:
  // Optional hook to reset transient options when an import begins.
  virtual void OnBeginImport(mochi::Path const& /*path*/, mochi::Path const& /*destDir*/) {}
  // Optional hook to draw importer-specific option widgets inside the modal body.
  virtual void OnShowImportOptions() {};
  // Main-thread callback that performs the final, non-thread-safe import steps once
  // all background tasks have finished. On success it may set `outAssetPath` to the
  // resulting top-level asset to load and select. Returns false to report failure.
  using FinalizeFn = std::function<bool(mochi::Path& outAssetPath)>;

  // Build the import on the main thread. Push the heavy, independent units of work
  // into `tasks` (each runs off the main thread via the app's AsyncTaskRunner) and
  // set `finalize` to a main-thread callback that performs the final non-thread-safe
  // steps (e.g. SaveToFile) and reports the asset path to load. Return false to abort
  // before any work is launched.
  virtual bool OnFinishImport(std::vector<AsyncTask>& tasks, FinalizeFn& finalize) = 0;
  // Optional hook to perform shutdown actions when import is canceled.
  virtual void OnCancelImport() {}
  // Optional hook to gate import button being pressed.
  virtual bool CanImport() {
    return true;
  }

  // The sibling folders (relative to the import destination) to create on a successful or partial
  // import so the standard bot content layout exists. Default: the four standard folders
  // (_cad / _intermediates / _render / _mochi). Importers whose users can redirect outputs override
  // this to omit the standard folder for any destination the user pointed elsewhere.
  virtual std::vector<std::string> SiblingFoldersToEnsure() const;

  // Optional hook for the modal's fixed size in pixels. Return 0 for an axis to
  // let that axis auto-size to its content (default: fully auto on both axes).
  virtual ImVec2 GetModalSize() const {
    return {0.0f, 0.0f};
  }

  // Accessors for subclasses.
  mochi::Path const& GetImportedPath() const {
    return _importedPath;
  }
  mochi::Path const& GetDestDir() const {
    return _destDir;
  }

  SuperDexStudio* _studio = nullptr;

 private:
  mochi::Path _importedPath; // file being imported
  mochi::Path _destDir; // asset browser directory the import should write into
  bool _openModal = false;
};

} // namespace superdex::studio
