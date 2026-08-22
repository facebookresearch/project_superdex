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

#include "io/urdf_importer.h"
#include "app/app.h"
#include "meshing/mesh_conversion.h"
#include "ui/imgui_widgets.h"

#include <superdex_robotics/utils/file_utils.h>

#include <mochi_core/utils/coordinate_space_converter.h>
#include <mochi_core/utils/error.h>
#include <mochi_core/utils/log.h>
#include <mochi_renderer/render_space.h>

#include <imgui.h>
#include <imguios/fonts/icons_font_awesome5.h>
#include <misc/cpp/imgui_stdlib.h>

#include <algorithm>
#include <cfloat>
#include <filesystem>
#include <functional>
#include <set>
#include <string>
#include <string_view>

namespace superdex::studio {

// Blanket radio choices for the Source row.
static constexpr int kSourceVisual = 0;
static constexpr int kSourceCollision = 1;
static constexpr int kSourceCustom = 2;

// Blanket radio choices for the Operation row. "Convert All" uses the smart
// default (convert non-native sources, import already-native ones).
static constexpr int kOpConvertAll = 0;
static constexpr int kOpImportOnly = 1;
static constexpr int kOpCustom = 2;

// Aligned x offsets (px) for the blanket radio columns so both rows line up.
static constexpr float kRadioCol0 = 80.0f;
static constexpr float kRadioCol1 = 200.0f;
static constexpr float kRadioCol2 = 320.0f;

// Picks the source a default should use: the preferred slot when it exists, else the
// other slot when it exists, else the referenced-but-missing slot (preferred first) so
// its warning still surfaces, else preferred. This makes a missing preferred mesh fall
// back to an existing one instead of clearing the model.
static UrdfMeshSource
PreferAvailable(UrdfLinkImportEntry const& entry, UrdfMeshSource preferred, UrdfMeshSource other) {
  if (entry.MeshRef(preferred).Exists()) {
    return preferred;
  }
  if (entry.MeshRef(other).Exists()) {
    return other;
  }
  if (entry.MeshRef(preferred).Referenced()) {
    return preferred;
  }
  if (entry.MeshRef(other).Referenced()) {
    return other;
  }
  return preferred;
}

// Render models prefer the visual mesh; fall back to collision if visual is absent.
static UrdfMeshSource DefaultRenderSource(UrdfLinkImportEntry const& entry) {
  return PreferAvailable(entry, UrdfMeshSource::Visual, UrdfMeshSource::Collision);
}

// Mochi models prefer the collision mesh; fall back to visual if collision is absent.
static UrdfMeshSource DefaultMochiSource(UrdfLinkImportEntry const& entry) {
  return PreferAvailable(entry, UrdfMeshSource::Collision, UrdfMeshSource::Visual);
}

// Smart default for a source: already-native sources (matching `nativeExt`, e.g.
// ".glb" / ".mochi.h5") are copied as-is, others convert; empty → ignore.
static UrdfMeshOperation DefaultOp(mochi::Path const& source, std::string_view nativeExt) {
  if (source.IsEmpty()) {
    return UrdfMeshOperation::Ignore;
  }
  return source.AsLowercaseString().ends_with(nativeExt) ? UrdfMeshOperation::Copy
                                                         : UrdfMeshOperation::Convert;
}

// Recompute the Render tab default source + operation for a link.
static void AutoRender(UrdfLinkImportEntry& entry) {
  entry.renderSource = DefaultRenderSource(entry);
  entry.renderOp = DefaultOp(entry.MeshRef(entry.renderSource).path, kGlbExtension);
}

// Recompute the Mochi tab default source + operation for a link.
static void AutoMochi(UrdfLinkImportEntry& entry) {
  entry.mochiSource = DefaultMochiSource(entry);
  entry.mochiOp = DefaultOp(entry.MeshRef(entry.mochiSource).path, kMochiH5Extension);
}

// Copies `src` to `dst` (parent directory must already exist). Returns success.
static bool CopyMeshFile(mochi::Path const& src, mochi::Path const& dst) {
  std::error_code ec;
  std::filesystem::copy_file(
      src.AsFilesystemPath(),
      dst.AsFilesystemPath(),
      std::filesystem::copy_options::overwrite_existing,
      ec);
  if (ec) {
    MOCHI_LOG_ERROR("Failed to copy import asset: %s", src.ToString().c_str());
    return false;
  }
  return true;
}

// Output path for one link model slot, or empty when the slot produces no file
// (Ignore, or no source). For Copy the source filename is preserved; for Convert
// the stem gets `outExt`. The result is made unique against files already on disk
// and against `claimed` (outputs reserved earlier in this same import), then
// inserted into `claimed` so later meshes don't collide with it. This prevents one
// link's output from silently overwriting another's (URDFs commonly reference many
// links by the same mesh filename).
static mochi::Path ModelOutputPath(
    mochi::Path const& src,
    UrdfMeshOperation op,
    mochi::Path const& subDir,
    std::string_view outExt,
    std::set<mochi::Path>& claimed) {
  if (src.IsEmpty() || op == UrdfMeshOperation::Ignore) {
    return {};
  }
  // Split the desired name into stem + extension so the uniqueness suffix lands on
  // the stem ("mesh_2.glb", not "mesh.glb_2"). Copy keeps the source's own
  // extension; Convert uses the target extension (e.g. ".mochi.h5").
  std::string stem = src.GetStem();
  std::string extension = std::string(outExt);
  if (op == UrdfMeshOperation::Copy) {
    if (src.AsLowercaseString().ends_with(outExt)) {
      std::string const filename = src.GetFilename();
      stem = filename.substr(0, filename.size() - outExt.size());
    } else {
      extension = src.GetExtension();
    }
  }
  std::string const unique =
      MakeUniqueFileName(stem, subDir, extension, [&claimed](mochi::Path const& path) {
        return claimed.count(path) > 0;
      });
  mochi::Path result = subDir / (unique + extension);
  claimed.insert(result);
  return result;
}

// Outer size for the per-link tables: fill the remaining height of the modal
// body child (the footer buttons are reserved by the modal shell), so large
// URDFs scroll within the table.
static ImVec2 RemainingTableSize() {
  return {0.0f, ImGui::GetContentRegionAvail().y};
}

// Draws an inline warning icon (with a tooltip naming the file) when the row's
// selected source mesh is referenced by the URDF but missing on disk.
static void ShowMissingSourceWarning(UrdfLinkImportEntry const& entry, UrdfMeshSource source) {
  if (!entry.MeshRef(source).Missing()) {
    return;
  }
  ImGui::SameLine();
  ImGui::TextColored(ImVec4(0.95f, 0.85f, 0.45f, 1.0f), "%s", ICON_FA_EXCLAMATION_TRIANGLE);
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip(
        "Source mesh not found on disk:\n%s", entry.MeshRef(source).DisplayRef().c_str());
  }
}

static void ShowNonNativeMochiCopyWarning(UrdfLinkImportEntry const& entry) {
  UrdfMeshRef const& ref = entry.MeshRef(entry.mochiSource);
  if (entry.mochiOp != UrdfMeshOperation::Copy || ref.path.IsEmpty() ||
      ref.path.AsLowercaseString().ends_with(kMochiH5Extension)) {
    return;
  }
  ImGui::SameLine();
  ImGui::TextColored(ImVec4(0.95f, 0.85f, 0.45f, 1.0f), "%s", ICON_FA_EXCLAMATION_TRIANGLE);
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip(
        "Copy preserves source bytes. This source is not a .mochi.h5 model, so the imported "
        "collision reference may fail to load.");
  }
}

// Draws the source dropdown for a row, listing the referenced sources and showing the
// full reference as a tooltip on hover (on both the dropdown entries and the closed
// combo box itself, even when the combo is disabled). `source` is updated on selection.
static void ShowSourceCombo(UrdfLinkImportEntry const& entry, UrdfMeshSource& source) {
  UrdfMeshRef const& curRef = entry.MeshRef(source);
  std::string const curLabel = curRef.Referenced() ? curRef.DisplayName() : "<none>";
  if (ImGui::BeginCombo("##Source", curLabel.c_str())) {
    if (entry.visual.Referenced()) {
      bool const selected = source == UrdfMeshSource::Visual;
      std::string const label = "Visual: " + entry.visual.DisplayName();
      if (ImGui::Selectable(label.c_str(), selected)) {
        source = UrdfMeshSource::Visual;
      }
      if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("%s", entry.visual.DisplayRef().c_str());
      }
      if (selected) {
        ImGui::SetItemDefaultFocus();
      }
    }
    if (entry.collision.Referenced()) {
      bool const selected = source == UrdfMeshSource::Collision;
      std::string const label = "Collision: " + entry.collision.DisplayName();
      if (ImGui::Selectable(label.c_str(), selected)) {
        source = UrdfMeshSource::Collision;
      }
      if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("%s", entry.collision.DisplayRef().c_str());
      }
      if (selected) {
        ImGui::SetItemDefaultFocus();
      }
    }
    ImGui::EndCombo();
  }
  // Show the selected source's full reference when hovering the (possibly disabled)
  // combo box, so the path is discoverable without opening the dropdown.
  if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
    std::string const ref = curRef.DisplayRef();
    ImGui::SetTooltip("%s", ref.empty() ? "<none>" : ref.c_str());
  }
}

// Draws the two blanket radio rows (Source and Operation) with aligned columns.
// Non-custom choices drive every row; Custom unlocks the matching table column.
static void ShowBlanketRadios(int& sourceChoice, int& opChoice) {
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Source:");
  ImGui::SameLine(kRadioCol0);
  ImGui::RadioButton("Visual", &sourceChoice, kSourceVisual);
  ImGui::SameLine(kRadioCol1);
  ImGui::RadioButton("Collision", &sourceChoice, kSourceCollision);
  ImGui::SameLine(kRadioCol2);
  ImGui::RadioButton("Custom##source", &sourceChoice, kSourceCustom);

  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Operation:");
  ImGui::SameLine(kRadioCol0);
  ImGui::RadioButton("Convert All", &opChoice, kOpConvertAll);
  ImGui::SameLine(kRadioCol1);
  ImGui::RadioButton("Copy All", &opChoice, kOpImportOnly);
  ImGui::SameLine(kRadioCol2);
  ImGui::RadioButton("Custom##operation", &opChoice, kOpCustom);
}

// Draws the "Destination: [ input ]" row for a model tab. Blank → destination root.
static void ShowDestinationField(std::string& destSubdir) {
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Destination:");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(300.0f);
  ImGui::InputTextWithHint("##Destination", "(root)", &destSubdir);
}

// Draws the "Remesh" checkbox for the mochi (collision) models. Remeshing only
// runs during conversion, so it is disabled when the operation is Copy All.
static void ShowRemeshCheckbox(bool& remesh, int opChoice) {
  ImGui::BeginDisabled(opChoice == kOpImportOnly);
  ImGui::Checkbox("Remesh", &remesh);
  ImGui::EndDisabled();
  if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
    ImGui::SetTooltip(
        "Remesh the surface on conversion to produce a clean, watertight collision mesh.\n"
        "This uses default remeshing parameters; fine tuning can be done after import.");
  }
}

// Draws the "Bake SDF" checkbox for the mochi (collision) models. Baking only runs
// during conversion, so it is disabled when the operation is Copy All.
static void ShowBakeSdfCheckbox(bool& bakeSdf, int opChoice) {
  ImGui::BeginDisabled(opChoice == kOpImportOnly);
  ImGui::Checkbox("Bake SDF", &bakeSdf);
  ImGui::EndDisabled();
  if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
    ImGui::SetTooltip(
        "Bake a signed distance field on conversion and store it with the collision mesh.\n"
        "This uses default SDF parameters; fine tuning can be done after import.");
  }
}

// Applies a blanket Source choice to every link's render or mochi selection.
// Visual/Collision force the matching source where present; Custom leaves rows
// untouched so per-row edits survive.
static void
ApplyBlanketSource(std::vector<UrdfLinkImportEntry>& links, int sourceChoice, bool render) {
  for (auto& entry : links) {
    if (sourceChoice == kSourceVisual && entry.visual.Referenced()) {
      (render ? entry.renderSource : entry.mochiSource) = UrdfMeshSource::Visual;
    } else if (sourceChoice == kSourceCollision && entry.collision.Referenced()) {
      (render ? entry.renderSource : entry.mochiSource) = UrdfMeshSource::Collision;
    }
  }
}

// Draws an aggregate warning icon (SameLined) when any link's selected source for the
// given tab (render or mochi) is referenced by the URDF but missing on disk. The tooltip
// lists each offending link and its unresolved reference. Used on the General tab, whose
// blanket combos hide the per-link rows, to surface the same warnings the model tables show.
static void ShowAggregateMissingWarning(
    std::vector<UrdfLinkImportEntry> const& links,
    bool render) {
  std::string offenders;
  int numMissing = 0;
  for (auto const& entry : links) {
    UrdfMeshSource const source = render ? entry.renderSource : entry.mochiSource;
    if (entry.MeshRef(source).Missing()) {
      ++numMissing;
      offenders += "\n" + entry.linkName + ": " + entry.MeshRef(source).DisplayRef();
    }
  }
  if (numMissing == 0) {
    return;
  }
  ImGui::SameLine();
  ImGui::TextColored(ImVec4(0.95f, 0.85f, 0.45f, 1.0f), "%s", ICON_FA_EXCLAMATION_TRIANGLE);
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("%d source mesh(es) not found on disk:%s", numMissing, offenders.c_str());
  }
}

static void
ApplyBlanketOperation(std::vector<UrdfLinkImportEntry>& links, int opChoice, bool render) {
  std::string_view const nativeExt = render ? kGlbExtension : kMochiH5Extension;
  for (auto& entry : links) {
    UrdfMeshSource const source = render ? entry.renderSource : entry.mochiSource;
    UrdfMeshOperation& op = render ? entry.renderOp : entry.mochiOp;
    if (opChoice == kOpConvertAll) {
      op = DefaultOp(entry.MeshRef(source).path, nativeExt);
    } else if (opChoice == kOpImportOnly && !entry.MeshRef(source).path.IsEmpty()) {
      op = UrdfMeshOperation::Copy;
    }
  }
}

// Draws a "Visual / Collision / Custom" source combo for the General tab,
// mirroring the blanket Source radio on a model tab. On change it stores the
// choice (shared with that tab) and applies it to every link; Custom leaves
// rows untouched for per-row editing on the model tab. A warning icon follows the
// combo when any link's selected source is missing on disk.
static void ShowGeneralSourceCombo(
    char const* label,
    std::vector<UrdfLinkImportEntry>& links,
    int& sourceChoice,
    int opChoice,
    bool render) {
  char const* const items[] = {"Visual", "Collision", "Custom"};
  if (ImGui::Combo(label, &sourceChoice, items, IM_ARRAYSIZE(items))) {
    ApplyBlanketSource(links, sourceChoice, render);
    ApplyBlanketOperation(links, opChoice, render);
  }
  ShowAggregateMissingWarning(links, render);
}

static void ShowRenderModelsTab(
    std::vector<UrdfLinkImportEntry>& links,
    int& sourceChoice,
    int& opChoice,
    std::string& destSubdir) {
  ShowBlanketRadios(sourceChoice, opChoice);
  ShowDestinationField(destSubdir);

  // Enforce the active blanket choices every frame; Custom leaves rows untouched
  // (and unlocks the matching column for manual editing below).
  ApplyBlanketSource(links, sourceChoice, /*render=*/true);
  ApplyBlanketOperation(links, opChoice, /*render=*/true);

  bool const sourceCustom = sourceChoice == kSourceCustom;
  bool const opCustom = opChoice == kOpCustom;

  auto const tableFlags = ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable |
      ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_ScrollY;
  if (ImGui::BeginTable("##RenderModels", 3, tableFlags, RemainingTableSize())) {
    ImGui::TableSetupColumn("Link", ImGuiTableColumnFlags_WidthStretch, 2.0f);
    ImGui::TableSetupColumn("Source", ImGuiTableColumnFlags_WidthStretch, 3.0f);
    ImGui::TableSetupColumn("Operation", ImGuiTableColumnFlags_WidthStretch, 2.0f);
    ImGui::TableSetupScrollFreeze(0, 1);
    ImGui::TableHeadersRow();

    // Make the per-row combos blend into the table.
    ImGui::PushFramelessWidgetStyle();
    for (size_t iRow = 0; iRow < links.size(); ++iRow) {
      auto& entry = links[iRow];
      ImGui::PushID(static_cast<int>(iRow));
      ImGui::TableNextRow();

      // Link name (read-only).
      ImGui::TableNextColumn();
      ImGui::AlignTextToFramePadding();
      ImGui::TextUnformatted(entry.linkName.c_str());
      ShowMissingSourceWarning(entry, entry.renderSource);

      // Source dropdown (editable only when the Source radio is Custom).
      ImGui::TableNextColumn();
      ImGui::SetNextItemWidth(-FLT_MIN);
      ImGui::BeginDisabled(!sourceCustom);
      ShowSourceCombo(entry, entry.renderSource);
      ImGui::EndDisabled();

      // Operation combo (editable only when the Operation radio is Custom and a
      // source exists; otherwise forced to Ignore).
      ImGui::TableNextColumn();
      ImGui::SetNextItemWidth(-FLT_MIN);
      bool const hasSource = !entry.MeshRef(entry.renderSource).path.IsEmpty();
      if (!hasSource) {
        entry.renderOp = UrdfMeshOperation::Ignore;
      }
      ImGui::BeginDisabled(!opCustom || !hasSource);
      char const* const items[] = {"Convert to GLB", "Copy", "Ignore"};
      int current = static_cast<int>(entry.renderOp);
      if (ImGui::Combo("##Op", &current, items, IM_ARRAYSIZE(items))) {
        entry.renderOp = static_cast<UrdfMeshOperation>(current);
      }
      ImGui::EndDisabled();

      ImGui::PopID();
    }
    ImGui::PopFramelessWidgetStyle();
    ImGui::EndTable();
  }
}

static void ShowMochiModelsTab(
    std::vector<UrdfLinkImportEntry>& links,
    int& sourceChoice,
    int& opChoice,
    std::string& destSubdir,
    bool& remesh,
    bool& bakeSdf) {
  ShowBlanketRadios(sourceChoice, opChoice);
  ShowDestinationField(destSubdir);

  ImGui::TextUnformatted("Options:");
  ImGui::SameLine(kRadioCol0);
  ShowRemeshCheckbox(remesh, opChoice);
  ImGui::SameLine();
  ShowBakeSdfCheckbox(bakeSdf, opChoice);

  // Enforce the active blanket choices every frame; Custom leaves rows untouched
  // (and unlocks the matching column for manual editing below).
  ApplyBlanketSource(links, sourceChoice, /*render=*/false);
  ApplyBlanketOperation(links, opChoice, /*render=*/false);

  bool const sourceCustom = sourceChoice == kSourceCustom;
  bool const opCustom = opChoice == kOpCustom;

  auto const tableFlags = ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable |
      ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_ScrollY;
  if (ImGui::BeginTable("##MochiModels", 3, tableFlags, RemainingTableSize())) {
    ImGui::TableSetupColumn("Link", ImGuiTableColumnFlags_WidthStretch, 2.0f);
    ImGui::TableSetupColumn("Source", ImGuiTableColumnFlags_WidthStretch, 3.0f);
    ImGui::TableSetupColumn("Operation", ImGuiTableColumnFlags_WidthStretch, 2.0f);
    ImGui::TableSetupScrollFreeze(0, 1);
    ImGui::TableHeadersRow();

    // Make the per-row combos blend into the table.
    ImGui::PushFramelessWidgetStyle();
    for (size_t iRow = 0; iRow < links.size(); ++iRow) {
      auto& entry = links[iRow];
      ImGui::PushID(static_cast<int>(iRow));
      ImGui::TableNextRow();

      // Link name (read-only).
      ImGui::TableNextColumn();
      ImGui::AlignTextToFramePadding();
      ImGui::TextUnformatted(entry.linkName.c_str());
      ShowMissingSourceWarning(entry, entry.mochiSource);
      ShowNonNativeMochiCopyWarning(entry);

      // Source dropdown (editable only when the Source radio is Custom).
      ImGui::TableNextColumn();
      ImGui::SetNextItemWidth(-FLT_MIN);
      ImGui::BeginDisabled(!sourceCustom);
      ShowSourceCombo(entry, entry.mochiSource);
      ImGui::EndDisabled();

      // Operation combo (editable only when the Operation radio is Custom and a
      // source exists; otherwise forced to Ignore).
      ImGui::TableNextColumn();
      ImGui::SetNextItemWidth(-FLT_MIN);
      bool const hasSource = !entry.MeshRef(entry.mochiSource).path.IsEmpty();
      if (!hasSource) {
        entry.mochiOp = UrdfMeshOperation::Ignore;
      }
      ImGui::BeginDisabled(!opCustom || !hasSource);
      char const* const items[] = {"Convert to H5", "Copy", "Ignore"};
      int current = static_cast<int>(entry.mochiOp);
      if (ImGui::Combo("##Op", &current, items, IM_ARRAYSIZE(items))) {
        entry.mochiOp = static_cast<UrdfMeshOperation>(current);
      }
      ImGui::EndDisabled();

      ImGui::PopID();
    }
    ImGui::PopFramelessWidgetStyle();
    ImGui::EndTable();
  }
}

UrdfImporter::UrdfImporter(SuperDexStudio* studio)
    : Importer(studio),
      _displayName("URDF"),
      _extensions({std::string(superdex::robotics::kUrdfExtension)}) {}

std::string const& UrdfImporter::GetDisplayName() const {
  return _displayName;
}

std::vector<std::string> const& UrdfImporter::GetExtensions() const {
  return _extensions;
}

ImU32 UrdfImporter::GetColor() const {
  return GetAssetTypeColor(AssetType::Bot);
}

void UrdfImporter::OnBeginImport(mochi::Path const& path, mochi::Path const& destDir) {
  mochi::ErrorLog error;
  superdex::robotics::UrdfMeshReferences meshRefs;
  _botPrefab = superdex::robotics::LoadBotPrefabFromUrdfFile(path.ToString(), meshRefs, error);
  // A valid import requires a successful parse and at least one link.
  _prefabValid = error.IsOK() && !_botPrefab.links.empty();
  if (!_prefabValid) {
    MOCHI_LOG_ERROR("Failed to load URDF: %s", path.ToString().c_str());
  }
  _options = {};
  // Default to a name that won't overwrite an existing bot in the destination.
  _options.name = MakeUniqueFileName(
      std::string(_botPrefab.name), destDir, std::string(superdex::robotics::kBotExtension));
  _options.renderDir = std::string(superdex::robotics::kRenderSubdir);
  _options.mochiDir = std::string(superdex::robotics::kCollisionSubdir);

  // Default blanket choices: visual render models, collision mochi models.
  _renderSourceChoice = kSourceVisual;
  _renderOpChoice = kOpConvertAll;
  _mochiSourceChoice = kSourceCollision;
  _mochiOpChoice = kOpConvertAll;

  // Snapshot each link's source meshes. The loader stores the resolved visual mesh
  // path in renderModelFile and the resolved collision mesh path in shapeFile (empty
  // when unresolved), and the verbatim URDF references in meshRefs. These per-row
  // defaults are the starting point when a tab is switched to Custom.
  _options.links.reserve(_botPrefab.links.size());
  for (size_t i = 0; i < _botPrefab.links.size(); ++i) {
    auto const& link = _botPrefab.links[i];
    UrdfLinkImportEntry entry;
    entry.linkName = std::string(link.name);
    entry.visual.path = mochi::Path{link.renderModelFile};
    entry.collision.path = mochi::Path{link.shapeFile};
    if (i < meshRefs.links.size()) {
      entry.visual.ref = std::string(meshRefs.links[i].visual);
      entry.collision.ref = std::string(meshRefs.links[i].collision);
    }
    AutoRender(entry);
    AutoMochi(entry);
    _options.links.push_back(std::move(entry));
  }
}

void UrdfImporter::OnShowImportOptions() {
  if (!_prefabValid) {
    ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(230, 100, 100, 255));
    ImGui::TextWrapped(
        "Failed to load URDF. The file could not be parsed or contains no link data.");
    ImGui::PopStyleColor();
    return;
  }

  if (ImGui::BeginTabBar("##UrdfImportTabs")) {
    if (ImGui::BeginTabItem("General")) {
      // Flag (and tint red) a Name that would overwrite an existing bot in the
      // destination; this also gates Import via CanImport. The default name is
      // pre-deduplicated, so this only triggers once the user edits it to a clash.
      std::string const botExt(superdex::robotics::kBotExtension);
      std::error_code ec;
      _nameConflicts = !_options.name.empty() &&
          std::filesystem::exists((GetDestDir() / (_options.name + botExt)).AsFilesystemPath(), ec);
      if (_nameConflicts) {
        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.5f, 0.0f, 0.0f, 1.0f));
      }
      ImGui::InputText("Name", &_options.name);
      if (_nameConflicts) {
        ImGui::PopStyleColor();
        if (ImGui::IsItemHovered()) {
          ImGui::SetTooltip("A bot named \"%s\" already exists here.", _options.name.c_str());
        }
      }
      // The root joint (index 0) attaches the bot to the world: Free for a
      // floating base, Hard to weld it in place.
      if (!_botPrefab.joints.empty()) {
        ImGui::ComboArticulatedJointType(
            "World Joint",
            _botPrefab.joints[0].type,
            ImGui::ArticulatedJointTypeFilter::HardFreeOnly);
      }
      // Blanket Source pickers mirroring the model tabs' Source radio, applied to
      // every link, each with its destination subdirectory. The model tabs still
      // allow per-row Custom overrides. PushID keeps the duplicated Destination
      // input IDs unique on this shared tab.
      ImGui::SeparatorText("Render (.glb)");
      ImGui::PushID("Render");
      ShowGeneralSourceCombo(
          "Source", _options.links, _renderSourceChoice, _renderOpChoice, /*render=*/true);
      ImGui::InputTextWithHint("Destination", "(root)", &_options.renderDir);
      ImGui::PopID();
      ImGui::SeparatorText("Collision (.mochi.h5)");
      ImGui::PushID("Mochi");
      ShowGeneralSourceCombo(
          "Source", _options.links, _mochiSourceChoice, _mochiOpChoice, /*render=*/false);
      ImGui::InputTextWithHint("Destination", "(root)", &_options.mochiDir);
      ShowRemeshCheckbox(_options.mochiRemesh, _mochiOpChoice);
      ImGui::SameLine();
      ShowBakeSdfCheckbox(_options.mochiBakeSdf, _mochiOpChoice);
      ImGui::PopID();
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Render Models")) {
      ShowRenderModelsTab(_options.links, _renderSourceChoice, _renderOpChoice, _options.renderDir);
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Collision Models")) {
      ShowMochiModelsTab(
          _options.links,
          _mochiSourceChoice,
          _mochiOpChoice,
          _options.mochiDir,
          _options.mochiRemesh,
          _options.mochiBakeSdf);
      ImGui::EndTabItem();
    }
    ImGui::EndTabBar();
  }
}

// Appends a background task for one link model slot, or clears `*outFile` now when the
// slot produces nothing (Ignore / no source). The worker runs `convertFn` (Convert) or
// a file copy (Copy) and, on success, writes the produced path into `*outFile`; on
// failure it clears `*outFile` so the prefab never references a file that was not
// written. `outFile` points into the stable `_botPrefab.links` vector (no import can
// start while the batch runs, so the pointer stays valid). Returns true when a task was
// appended, so the caller can ensure the output directory exists before launching.
static bool BuildModelTask(
    std::vector<AsyncTask>& tasks,
    mochi::DynamicString* outFile,
    mochi::Path const& src,
    mochi::Path const& dst,
    UrdfMeshOperation op,
    std::string const& kindLabel,
    std::function<bool(mochi::Path const&, mochi::Path const&)> convertFn) {
  if (dst.IsEmpty()) {
    outFile->clear();
    return false;
  }
  std::string label;
  std::function<bool(AsyncCancelToken const&)> work;
  if (op == UrdfMeshOperation::Copy) {
    label = "Copy " + src.GetFilename();
    work = [outFile, src, dst](AsyncCancelToken const& cancel) {
      // Bail before doing work if the batch was already cancelled; clear the output so the prefab
      // never references a file this task did not write.
      if (cancel.IsCancelRequested()) {
        outFile->clear();
        return false;
      }
      if (CopyMeshFile(src, dst)) {
        *outFile = dst.ToString();
        return true;
      }
      outFile->clear();
      return false;
    };
  } else {
    label = "Convert " + src.GetFilename() + " -> " + kindLabel;
    work = [outFile, src, dst, convertFn = std::move(convertFn)](AsyncCancelToken const& cancel) {
      // Bail before doing work if the batch was already cancelled. (A conversion that itself
      // marshals to the superdex_mesh_cli helper -- remesh / SDF bake -- is also aborted mid-flight
      // by the subprocess kill that Config::onCancel triggers.)
      //
      // A pure-CPU conversion (e.g. OBJ/STL -> GLB geometry conversion, which never touches the
      // helper) is atomic from this task's point of view: convertFn is a single library call with
      // no cancellation checkpoints, so once it starts it runs to completion. Making it
      // interruptible mid-conversion would require threading the cancel token into convertFn
      // itself. These conversions are fast relative to a runaway remesh, so this is left as a
      // follow-up.
      if (cancel.IsCancelRequested()) {
        outFile->clear();
        return false;
      }
      if (convertFn(src, dst)) {
        *outFile = dst.ToString();
        return true;
      }
      MOCHI_LOG_ERROR(
          "Failed to convert import asset: %s -> %s",
          src.ToString().c_str(),
          dst.ToString().c_str());
      std::error_code ec;
      std::filesystem::remove(dst.AsFilesystemPath(), ec);
      outFile->clear();
      return false;
    };
  }
  tasks.push_back(AsyncTask{std::move(label), std::move(work)});
  return true;
}

bool UrdfImporter::OnFinishImport(std::vector<AsyncTask>& tasks, FinalizeFn& finalize) {
  mochi::Path const destDir = GetDestDir();

  // Converted/copied assets are written to the per-tab destination subdirectory,
  // or directly into the destination root when the subdirectory is left blank.
  mochi::Path const mochiDir = _options.mochiDir.empty() ? destDir : destDir / _options.mochiDir;
  mochi::Path const renderDir = _options.renderDir.empty() ? destDir : destDir / _options.renderDir;

  // URDF mesh assets are authored in Mochi's Z-up space; GLB render output must
  // be in the renderer's space (@ref mochi_renderer::RenderSpace). This converter is passed to
  // ConvertToGlb for OBJ/STL inputs below; COLLADA is excluded since its reader
  // already emits renderer-space geometry. Captured by value into each task.
  mochi::CoordinateSpaceConverter const renderConverter(
      mochi::CoordinateSpace::Default(), mochi_renderer::RenderSpace());

  // Inverse direction for collision output: render-space DAE/GLB sources must be
  // brought back into Mochi space. Passed to ConvertToH5 for .dae / .glb inputs;
  // native collision formats (.stl/.obj/.off/.ply) are loaded directly into Mochi
  // space and need no converter.
  mochi::CoordinateSpaceConverter const mochiConverter(
      mochi_renderer::RenderSpace(), mochi::CoordinateSpace::Default());

  bool needRenderDir = false;
  bool needMochiDir = false;

  // Output mesh names are de-duplicated against files already on disk and against
  // each other (tracked here) so no two links overwrite the same output file.
  std::set<mochi::Path> claimedOutputs;

  size_t const numLinks = std::min(_botPrefab.links.size(), _options.links.size());
  for (size_t i = 0; i < numLinks; ++i) {
    auto& link = _botPrefab.links[i];
    UrdfLinkImportEntry const& entry = _options.links[i];

    // Snapshot the per-source transforms resolved by the loader: the visual origin
    // lives in renderModel* and the collision origin in shape*. Snapshotting first
    // lets either tab draw from either source without corrupting the other.
    auto const visualRotation = link.renderModelRotation;
    auto const visualTranslation = link.renderModelTranslation;
    auto const visualScale = link.renderModelScale;
    auto const collisionRotation = link.shapeRotation;
    auto const collisionTranslation = link.shapeTranslation;
    auto const collisionScale = link.shapeScale;

    // Render model (visual slot). Use the chosen source's transform so the
    // produced model keeps the correct offset even when the source is swapped.
    UrdfMeshOperation const renderOp = entry.renderOp;
    if (renderOp != UrdfMeshOperation::Ignore) {
      if (entry.renderSource == UrdfMeshSource::Visual) {
        link.renderModelRotation = visualRotation;
        link.renderModelTranslation = visualTranslation;
        link.renderModelScale = visualScale;
      } else {
        link.renderModelRotation = collisionRotation;
        link.renderModelTranslation = collisionTranslation;
        link.renderModelScale = collisionScale;
      }
    }
    mochi::Path const renderSrc = entry.MeshRef(entry.renderSource).path;
    // ModelOutputPath returns empty for an empty source (a missing mesh has an empty
    // path), so a missing or absent source yields an empty dst and BuildModelTask
    // clears the field instead of producing a model.
    mochi::Path const renderDst =
        ModelOutputPath(renderSrc, renderOp, renderDir, kGlbExtension, claimedOutputs);
    needRenderDir |= BuildModelTask(
        tasks,
        &link.renderModelFile,
        renderSrc,
        renderDst,
        renderOp,
        ".glb",
        [renderConverter](mochi::Path const& in, mochi::Path const& out) {
          // OBJ/STL assets are authored in Mochi space and must be converted to
          // the renderer's space; COLLADA's reader already emits renderer-space
          // geometry, so it is converted as-is (no space converter).
          bool const isCollada = in.AsLowercaseString().ends_with(".dae");
          return ConvertToGlb(in, out, isCollada ? nullptr : &renderConverter);
        });

    // Mochi model (collision slot).
    UrdfMeshOperation const mochiOp = entry.mochiOp;
    if (mochiOp != UrdfMeshOperation::Ignore) {
      if (entry.mochiSource == UrdfMeshSource::Visual) {
        link.shapeRotation = visualRotation;
        link.shapeTranslation = visualTranslation;
        link.shapeScale = visualScale;
      } else {
        link.shapeRotation = collisionRotation;
        link.shapeTranslation = collisionTranslation;
        link.shapeScale = collisionScale;
      }
    }
    mochi::Path const mochiSrc = entry.MeshRef(entry.mochiSource).path;
    // Empty source (including a missing mesh) yields an empty dst; BuildModelTask then
    // clears the field instead of producing a model. See the render slot above.
    mochi::Path const mochiDst =
        ModelOutputPath(mochiSrc, mochiOp, mochiDir, kMochiH5Extension, claimedOutputs);
    needMochiDir |= BuildModelTask(
        tasks,
        &link.shapeFile,
        mochiSrc,
        mochiDst,
        mochiOp,
        ".mochi.h5",
        [mochiConverter, remesh = _options.mochiRemesh, bakeSdf = _options.mochiBakeSdf](
            mochi::Path const& in, mochi::Path const& out) {
          // DAE/GLB sources are render-space and must be converted to Mochi
          // space; native collision formats load directly into Mochi space.
          std::string const& lower = in.AsLowercaseString();
          bool const isRenderFormat = lower.ends_with(".dae") || lower.ends_with(".glb");
          return ConvertToH5(in, out, isRenderFormat ? &mochiConverter : nullptr, remesh, bakeSdf);
        });
  }

  // Create the output directories once here, before launching, so the workers never
  // race in create_directories. Only create a directory that will actually receive a
  // file, matching the prior lazy behavior (no stray empty directories).
  std::error_code ec;
  if (needRenderDir) {
    std::filesystem::create_directories(renderDir.AsFilesystemPath(), ec);
  }
  if (needMochiDir) {
    std::filesystem::create_directories(mochiDir.AsFilesystemPath(), ec);
  }

  // Finalize runs on the main thread once every task has finished: name the prefab
  // and save it. `this` outlives the batch (importers are app-owned and no new import
  // can start while one runs). The Name field is validated against existing files
  // (see OnShowImportOptions / CanImport), so `_options.name` is collision-free here.
  mochi::Path const outPath =
      destDir / (_options.name + std::string(superdex::robotics::kBotExtension));
  finalize = [this, outPath, destDir](mochi::Path& outAssetPath) -> bool {
    _botPrefab.name = _options.name;
    mochi::ErrorLog error;
    superdex::robotics::SaveToFile(_botPrefab, outPath.ToString(), error);
    if (!error.IsOK()) {
      MOCHI_LOG_ERROR("Failed to save bot: %s", outPath.ToString().c_str());
      return false;
    }
    // Anchor the imported bot under a bots root so its root-relative (`//`) and tag (`@`)
    // references resolve. If nothing up the ancestry is marked, mark the destination folder.
    if (!superdex::robotics::FindBotsRoot(outPath.AsFilesystemPath()).has_value()) {
      _studio->CreateRootMarker(destDir, mochi::ErrorLog{});
    }
    outAssetPath = outPath;
    return true;
  };
  return true;
}

bool UrdfImporter::CanImport() {
  return _prefabValid && !_options.name.empty() && !_nameConflicts;
}

std::vector<std::string> UrdfImporter::SiblingFoldersToEnsure() const {
  // Standard layout is all-or-nothing: users typically use both the _render and _mochi folders or
  // neither. If both destinations are still their defaults, scaffold all standard sibling folders.
  // If either was redirected (including to blank = the destination root), the user has chosen a
  // custom layout, so scaffold none here -- only the folders the import actually writes to are
  // created lazily by OnFinishImport.
  bool const standardLayout =
      std::string_view(_options.renderDir) == superdex::robotics::kRenderSubdir &&
      std::string_view(_options.mochiDir) == superdex::robotics::kCollisionSubdir;
  if (!standardLayout) {
    return {};
  }
  return {
      std::string(superdex::robotics::kCadSubdir),
      std::string(superdex::robotics::kIntermediatesSubdir),
      std::string(superdex::robotics::kRenderSubdir),
      std::string(superdex::robotics::kCollisionSubdir)};
}

ImVec2 UrdfImporter::GetModalSize() const {
  // Fixed size so the modal stays consistent across tabs and large URDFs scroll.
  return {600.0f, 700.0f};
}

} // namespace superdex::studio
