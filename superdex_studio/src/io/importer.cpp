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

#include "io/importer.h"
#include "app/app.h"
#include "assets/asset_manager.h"

#include <superdex_robotics/utils/file_utils.h>

#include <imgui.h>

#include <cfloat>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

namespace superdex::studio {

namespace {

// Creates each of @p subdirs inside @p destDir if it does not already exist. Run after an import
// produces an asset (full or partial success) so the Model Editor's sibling discovery and per-stage
// exports have a consistent place to look and write. The set of folders is chosen by the importer
// (see Importer::SiblingFoldersToEnsure), so an importer can omit a folder whose destination the
// user redirected elsewhere. The user is free to delete the folders should they wish and the
// discovery will handle that.
void EnsureSiblingFolders(mochi::Path const& destDir, std::vector<std::string> const& subdirs) {
  if (destDir.IsEmpty()) {
    return;
  }
  std::error_code ec;
  for (std::string const& subdir : subdirs) {
    std::filesystem::create_directories((destDir / subdir).AsFilesystemPath(), ec);
  }
}

} // namespace

std::vector<std::string> Importer::SiblingFoldersToEnsure() const {
  return {
      std::string(superdex::robotics::kCadSubdir),
      std::string(superdex::robotics::kIntermediatesSubdir),
      std::string(superdex::robotics::kRenderSubdir),
      std::string(superdex::robotics::kCollisionSubdir)};
}

void Importer::BeginImport(mochi::Path const& path, mochi::Path const& destDir) {
  _importedPath = path;
  _destDir = destDir;
  OnBeginImport(path, destDir);
  _openModal = true;
}

bool Importer::ShowModalWindow() {
  std::string const& title = GetDisplayName();
  if (_openModal) {
    ImGui::OpenPopup(title.c_str());
    _openModal = false;
  }
  ImVec2 const center = ImGui::GetMainViewport()->GetCenter();
  ImGui::SetNextWindowPos(center, ImGuiCond_Always, ImVec2(0.5f, 0.5f));
  // Importers may request a fixed width and/or height. A 0 axis auto-fits to
  // content. AlwaysAutoResize is only used when at least one axis is auto.
  ImVec2 const modalSize = GetModalSize();
  ImGuiWindowFlags windowFlags =
      ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings;
  if (modalSize.x > 0.0f && modalSize.y > 0.0f) {
    ImGui::SetNextWindowSize(modalSize, ImGuiCond_Always);
  } else {
    if (modalSize.x > 0.0f) {
      ImGui::SetNextWindowSizeConstraints(ImVec2(modalSize.x, 0.0f), ImVec2(modalSize.x, FLT_MAX));
    } else if (modalSize.y > 0.0f) {
      ImGui::SetNextWindowSizeConstraints(ImVec2(0.0f, modalSize.y), ImVec2(FLT_MAX, modalSize.y));
    }
    windowFlags |= ImGuiWindowFlags_AlwaysAutoResize;
  }
  bool open = false;
  // Make the modal use the regular window background instead of the (often
  // darker) popup background, so it matches the app's windows.
  ImGui::PushStyleColor(ImGuiCol_PopupBg, ImGui::GetStyleColorVec4(ImGuiCol_WindowBg));
  bool const modalVisible = ImGui::BeginPopupModal(title.c_str(), nullptr, windowFlags);
  ImGui::PopStyleColor();
  if (modalVisible) {
    open = true;
    ImGui::TextDisabled("%s", _importedPath.GetFilename().c_str());
    ImGui::Separator();
    ImGui::Spacing();

    // When the modal has a fixed height, host the options in a child that fills
    // the remaining space so the footer buttons always sit at the bottom
    // regardless of how much content the active tab draws.
    bool const pinFooter = modalSize.y > 0.0f;
    if (pinFooter) {
      float const footerHeight =
          ImGui::GetFrameHeightWithSpacing() + ImGui::GetStyle().ItemSpacing.y * 2.0f;
      ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
      ImGui::BeginChild("##ImporterBody", ImVec2(0.0f, -footerHeight));
    }
    OnShowImportOptions();
    if (pinFooter) {
      ImGui::EndChild();
      ImGui::PopStyleColor();
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::BeginDisabled(!CanImport());
    if (ImGui::Button("Import", ImVec2(120, 0))) {
      std::vector<AsyncTask> tasks;
      FinalizeFn finalize;
      if (OnFinishImport(tasks, finalize)) {
        // Close the options modal now and hand the background work off to the app's
        // runner, which shows its own progress modal. onComplete runs on the main
        // thread once the work finishes: it runs finalize, then performs the
        // post-import asset-browser/asset-manager steps (all main-thread only).
        SuperDexStudio* studio = _studio;
        mochi::Path const destDir = GetDestDir();
        // Snapshot the folders to create now (main thread): it reflects the importer's current
        // destination choices, which the modal below no longer draws once the import launches.
        std::vector<std::string> siblingFolders = SiblingFoldersToEnsure();
        studio->BeginAsyncTasks(
            GetDisplayName(),
            std::move(tasks),
            [studio,
             finalize = std::move(finalize),
             destDir,
             siblingFolders = std::move(siblingFolders)](bool /*allSucceeded*/) {
              mochi::Path resultPath;
              bool const ok = finalize ? finalize(resultPath) : false;
              if (ok) {
                // The import produced an asset (fully or partially -- some meshes may have failed):
                // ensure the standard sibling folders exist next to it.
                EnsureSiblingFolders(destDir, siblingFolders);
              }
              studio->GetAssetBrowser().Refresh();
              if (ok && !resultPath.IsEmpty()) {
                if (Asset* asset = studio->GetAssetManager().LoadAsset(resultPath)) {
                  studio->GetAssetBrowser().SelectAsset(asset);
                }
              }
            });
        ImGui::CloseCurrentPopup();
        open = false;
      }
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button("Cancel", ImVec2(120, 0))) {
      OnCancelImport();
      ImGui::CloseCurrentPopup();
      open = false;
    }
    ImGui::EndPopup();
  }
  return open;
}

} // namespace superdex::studio
