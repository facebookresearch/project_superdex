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

#include "editors/model_editor.h"
#include "app/app.h"
#include "assets/cad_model_asset.h"
#include "assets/mochi_model_asset.h"
#include "assets/render_model_asset.h"
#include "core/async_task.h"
#include "meshing/processing_modifiers/embed_mesh.h" // MeasureEmbeddedMesh
#include "meshing/processing_modifiers/preset_discovery.h"
#include "meshing/processing_modifiers/processing_mesh_utils.h"
#include "meshing/processing_modifiers/processing_serialization.h"
#include "meshing/processing_modifiers/processing_stack.h"
#include "rendering/measure_tool.h"
#include "ui/asset_browser.h"
#include "ui/imgui_widgets.h"
#include "ui/reorderable_list.h"

#include <imgui_internal.h> // ImGuiWindow / WorkRect, to bound the modifier header width
#include <tinyfiledialogs.h> // tinyfd_messageBox (preset-replace confirm)

#include <superdex_robotics/utils/file_utils.h> // AssetRoleFolderForWrite / kIntermediatesSubdir

#include <mochi_renderer/mesh.h>
#include <mochi_renderer/resource.h> // RenderModel, RenderModelFormat
#include <mochi_renderer/utils.h>

#include <mochi_mesh/mesh_statistics.h>
#include <mochi_mesh/step_mesh_body.h>
#include <mochi_mesh/step_mesh_stages.h>
#include <mochi_mesh/step_tessellation.h>
#include <mochi_mesh/surface_remeshing.h>

#include <mochi_core/geometry/aabb.h>
#include <mochi_core/geometry/geometry_utils.h>
#include <mochi_core/geometry/model_data.h>
#include <mochi_core/geometry/model_utils.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/container_utils.h>
#include <mochi_core/utils/coordinate_space_converter.h>
#include <mochi_core/utils/dynamic_string.h>
#include <mochi_core/utils/log.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/nd_array_utils.h>
#include <mochi_core/utils/span.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cfloat>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <initializer_list>
#include <iomanip>
#include <mutex>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace mochi_renderer;

namespace superdex::studio {

namespace {

// Asset content layout: an asset's base folder holds its bot / prefab file, and its models sit
// either in the standard role folders beside it (superdex::robotics::kCadSubdir = CAD models
// (STEP/STL), kRenderSubdir = render models, kCollisionSubdir = collision models) or flat in the
// base folder itself; kIntermediatesSubdir holds the saved .StudioProcessing.json pipelines under
// either layout. Both layouts, and partial or foreign ones, are resolved by FindAssetForSlot (see
// assets/asset.h for the search order) off the base folder that FindAssetBaseFolder anchors. These
// folder names are the single source of truth in file_utils.h.

// How often the editor polls slot files for on-disk changes (hot-reload after an export overwrite).
constexpr std::chrono::milliseconds kSlotPollInterval{100};

// The file's last-write time, or a default-constructed time when the path is empty or unreadable.
std::filesystem::file_time_type FileMtimeOrDefault(mochi::DynamicString const& path) {
  if (path.empty()) {
    return {};
  }
  std::error_code ec;
  auto const mtime = std::filesystem::last_write_time(path.c_str(), ec);
  return ec ? std::filesystem::file_time_type{} : mtime;
}

// Role suffixes appended to a mesh's color seed so an input surface and the SDF surface
// reconstructed from it get distinct but stable hashed colors. Shared by the Model Viewer's Mochi
// Model section and the Export Mochi Model modifier so the two reads look identical.
constexpr std::string_view kInputMeshColorRole = " input mesh";
constexpr std::string_view kOutputSdfColorRole = kSdfSurfaceColorRole;
// Role suffixes hashed with a collision model's path to color its contact skin and visual mesh.
constexpr std::string_view kContactSkinColorRole = " contact skin";
constexpr std::string_view kVisualMeshColorRole = " visual mesh";

// Deterministic surface color for a mesh identified by @p seed (a file path or stage label) in a
// given @p role: hashing seed+role keeps related meshes distinct yet stable across sessions. The
// wireframe is derived from the surface color via WireframeColorForSurface (imgui_widgets).
mochi::Real3 HashedMeshColor(std::string_view seed, std::string_view role) {
  std::string combined;
  combined.reserve(seed.size() + role.size());
  combined.append(seed.data(), seed.size());
  combined.append(role.data(), role.size());
  ImVec4 const c = HashStringToColor(combined);
  return {c.x, c.y, c.z};
}
filament::math::float3 ToFloat3(mochi::Real3 const& c) {
  return {static_cast<float>(c[0]), static_cast<float>(c[1]), static_cast<float>(c[2])};
}
// Wireframe edge color (float3, for CreateWireframeMaterial) derived from a Real3 surface color via
// the shared WireframeColorForSurface, so every wireframe here matches the app-wide scheme.
filament::math::float3 WireColor(mochi::Real3 const& surfaceColor) {
  return WireframeColorForSurface(ToFloat3(surfaceColor));
}
// Converts a neutral triangle MeshData into a single renderer MeshSection (positions + indices,
// normals computed later by the renderer).
MeshSection MeshDataToSection(mochi::MeshData const& mesh) {
  MeshSection section;
  section.positions.reserve(mesh.coordinates.size());
  for (mochi::real const coordinate : mesh.coordinates) {
    section.positions.push_back(static_cast<float>(coordinate));
  }
  section.indices.assign(mesh.connectivity.begin(), mesh.connectivity.end());
  section.hasNormals = false;
  return section;
}

// MeshDataToSection for a triangle mesh of a Mochi model, converted from Mochi to renderer space.
MeshSection ModelMeshToSection(
    mochi::MeshData const& mesh,
    mochi::CoordinateSpaceConverter const& converter) {
  MeshSection section = MeshDataToSection(mesh);
  converter.TranslationsToOutput(mochi::MakeSpan(section.positions), mochi::ErrorAssert{});
  return section;
}

// Node and tetrahedron counts for the stats block of a tetrahedral mesh; nullopt otherwise.
std::optional<mochi::Int2> TetCounts(mochi::MeshData const& mesh) {
  if (mesh.nodesPerElement != 4) {
    return std::nullopt;
  }
  return mochi::Int2{mesh.GetNumNodes(), mesh.GetNumElements()};
}

// Concatenates all sections into flat positions/indices (offsetting each section's indices) and
// computes angle-weighted per-vertex normals over the merged mesh.
void MergeSections(
    std::vector<MeshSection> const& sections,
    std::vector<float>& positions,
    std::vector<float>& normals,
    std::vector<int>& indices) {
  positions.clear();
  indices.clear();
  for (auto const& section : sections) {
    int const base = static_cast<int>(positions.size() / 3);
    positions.insert(positions.end(), section.positions.begin(), section.positions.end());
    for (int const index : section.indices) {
      indices.push_back(base + index);
    }
  }
  std::vector<float> faceNormals;
  ComputeFaceNormals(positions, indices, faceNormals);
  ComputeVertexNormalsAngleWeighted(positions, faceNormals, indices, normals);
}

// Flattens a stage buffer's sections into a single neutral triangle MeshData for a processing op.
mochi::MeshData SectionsToMeshData(std::vector<MeshSection> const& sections) {
  std::vector<float> positions;
  std::vector<float> normals;
  std::vector<int> indices;
  MergeSections(sections, positions, normals, indices);
  mochi::MeshData mesh;
  mesh.nodesPerElement = 3;
  mesh.coordinates.reserve(positions.size());
  for (float const value : positions) {
    mesh.coordinates.push_back(static_cast<mochi::real>(value));
  }
  mesh.connectivity.reserve(indices.size());
  for (int const index : indices) {
    mesh.connectivity.push_back(index);
  }
  return mesh;
}

// Builds the base mesh-statistics summary: vertex / edge / triangle counts, watertightness, AABB
// size, and edge-length / interior-angle distributions (coordinates are in meters). Uses the shared
// mochi::mesh::ComputeMeshStatistics (fully in-process; no CGAL/CLI). Edge count is not part of
// MeshStatistics, so it is counted here. Multi-section display meshes are concatenated, not welded
// (see SectionsToMeshData), so vertex count and watertightness reflect the display geometry.
// Annotations (SDF grid, Hausdorff) are composited around this by ComposeStats.
std::string FormatMeshStats(mochi::MeshData const& mesh) {
  if (mesh.GetNumElements() == 0) {
    return "No mesh.";
  }
  mochi::ErrorLog error;
  mochi::mesh::MeshStatistics const stats =
      mochi::mesh::ComputeMeshStatistics(mesh, nullptr, error);
  if (!error.IsOK()) {
    return "Statistics unavailable.";
  }

  // Unique undirected edges: each triangle contributes 3, deduped by (min,max) index pair.
  std::unordered_set<uint64_t> edges;
  edges.reserve(mesh.connectivity.size());
  auto const addEdge = [&edges](int a, int b) {
    auto const lo = static_cast<uint32_t>(std::min(a, b));
    auto const hi = static_cast<uint32_t>(std::max(a, b));
    edges.insert((static_cast<uint64_t>(lo) << 32) | hi);
  };
  for (std::size_t t = 0; t + 2 < mesh.connectivity.size(); t += 3) {
    addEdge(mesh.connectivity[t], mesh.connectivity[t + 1]);
    addEdge(mesh.connectivity[t + 1], mesh.connectivity[t + 2]);
    addEdge(mesh.connectivity[t + 2], mesh.connectivity[t]);
  }

  // AABB from the flat coordinate array (3 contiguous reals per vertex).
  mochi::Span<mochi::Real3 const> const coords(
      reinterpret_cast<mochi::Real3 const*>(mesh.coordinates.data()), mesh.coordinates.size() / 3);
  mochi::Real3 const size = mochi::CalcAabb(coords).GetSize();

  std::ostringstream out;
  out << "Vertices:     " << stats.numVertices << "\n"
      << "Edges:        " << edges.size() << "\n"
      << "Triangles:    " << stats.numFaces << "\n"
      << "Watertight:   " << (stats.isClosed ? "yes" : "no") << "\n";
  out << std::fixed << std::setprecision(4) << "AABB (m):     " << size[0] << " x " << size[1]
      << " x " << size[2] << "\n"
      << "Edge len (m): mean " << stats.edgeLengths.mean << "  std "
      << stats.edgeLengths.standardDeviation << "  min " << stats.edgeLengths.min << "  max "
      << stats.edgeLengths.max << "\n";
  out << std::setprecision(2) << "Angle (deg):  mean " << stats.angles.mean << "  std "
      << stats.angles.standardDeviation << "  min " << stats.angles.min << "  max "
      << stats.angles.max;
  return out.str();
}

std::string FormatMeshStats(std::vector<mochi_renderer::MeshSection> const& sections) {
  return FormatMeshStats(SectionsToMeshData(sections));
}

// On-disk byte size of the file at @p path, or nullopt when the path is empty / missing / errors.
// Used for the Model Viewer slots' "File Size" line (actual size, not an estimate).
std::optional<int64_t> OnDiskFileSize(mochi::DynamicString const& path) {
  if (path.empty()) {
    return std::nullopt;
  }
  std::error_code ec;
  auto const size = std::filesystem::file_size(std::filesystem::path(path.c_str()), ec);
  if (ec) {
    return std::nullopt;
  }
  return static_cast<int64_t>(size);
}

// Formats a byte count as a compact human-readable size (B / KB / MB / GB, binary units).
std::string FormatByteSize(int64_t bytes) {
  auto value = static_cast<double>(bytes);
  char const* const units[] = {"B", "KB", "MB", "GB", "TB"};
  int unit = 0;
  while (value >= 1024.0 && unit < 4) {
    value /= 1024.0;
    ++unit;
  }
  std::ostringstream out;
  if (unit == 0) {
    out << bytes << " B";
  } else {
    out << std::fixed << std::setprecision(1) << value << " " << units[unit];
  }
  return out.str();
}

// The stats heading of a mesh embedded in @p role, padded to line up with the other headings.
char const* EmbeddedMeshStatsHeading(EmbeddedMeshRole role) {
  switch (role) {
    case EmbeddedMeshRole::ContactSkin:
      return "Contact skin: ";
    case EmbeddedMeshRole::VisualMesh:
      return "Visual mesh:  ";
  }
  return "";
}

// Composites a MeshStats' display string from its base summary plus its annotations: the file-size
// line first (when set), then SDF grid, tet mesh and embedded mesh lines (when set), then the base
// summary (headed as the boundary surface for a tet mesh), and a Hausdorff line appended (when >=
// 0). Cheap; recompute whenever the base or an annotation changes (e.g. when the async Hausdorff
// result arrives). Extend here to add future annotations.
void ComposeStats(MeshStats& stats) {
  std::ostringstream out;
  if (stats.fileSizeBytes.has_value()) {
    std::string const label = stats.fileSizeLabel.empty() ? "File Size" : stats.fileSizeLabel;
    out << label << ": " << FormatByteSize(*stats.fileSizeBytes) << "\n";
  }
  if (stats.sdfGrid.has_value()) {
    mochi::Int3 const& d = *stats.sdfGrid;
    out << "SDF grid:     " << d[0] << " x " << d[1] << " x " << d[2] << "\n";
  }
  if (stats.tetMesh.has_value()) {
    out << "Tet mesh:     " << (*stats.tetMesh)[0] << " nodes, " << (*stats.tetMesh)[1] << " tets\n"
        << "Boundary surface:\n";
  }
  if (stats.embeddedMesh.has_value()) {
    EmbeddedMeshStats const& mesh = *stats.embeddedMesh;
    out << EmbeddedMeshStatsHeading(mesh.role) << mesh.nodes << " nodes, " << mesh.triangles
        << " tris, max extrapolation " << std::fixed << std::setprecision(2)
        << mesh.maxExtrapolation << "\n";
  }
  out << stats.base;
  if (stats.hausdorff >= 0.0) {
    out << "\nHausdorff (m): " << std::fixed << std::setprecision(6) << stats.hausdorff;
  }
  stats.display = out.str();
}

// Draws a read-only, selectable (copyable) multi-line text box for a statistics summary. Sized to
// the content up to a cap, after which it scrolls. @p font (when non-null) renders the block in a
// monospace face so the space-padded stat columns line up.
void DrawMeshStatsBlock(char const* id, std::string const& text, ImFont* font) {
  ImGui::ReadOnlyTextBlock(id, text, font);
}

// Shared visualization controls for one held mesh: Surface / Wireframe / Stats toggles on one row,
// an Opacity slider, and (when Stats is on) the read-only statistics block. Each on-change callback
// fires only when its control is edited; the caller performs the effect (visibility / material),
// which differs per mesh. @p statsText is precomputed by the caller when the geometry changes.
// @p idScope keeps ImGui ids distinct between co-located control groups (e.g. Mochi vs SDF).
void DrawMeshVizControls(
    char const* idScope,
    bool& showSurface,
    bool& showWireframe,
    bool& showStats,
    float& opacity,
    std::string const& statsText,
    ImFont* statsFont,
    std::function<void()> const& onShowSurface,
    std::function<void()> const& onShowWireframe,
    std::function<void()> const& onOpacity) {
  ImGui::PushID(idScope);
  if (ImGui::Checkbox("Surface", &showSurface)) {
    onShowSurface();
  }
  ImGui::SameLine();
  if (ImGui::Checkbox("Wireframe", &showWireframe)) {
    onShowWireframe();
  }
  ImGui::SameLine();
  ImGui::Checkbox("Stats", &showStats);
  if (ImGui::SliderFloat("Opacity", &opacity, 0.0f, 1.0f, "%.2f")) {
    onOpacity();
  }
  if (showStats) {
    DrawMeshStatsBlock("##stats", statsText, statsFont);
  }
  ImGui::PopID();
}

void ApplySurfaceVisibility(StageBuffer const& buffer) {
  if (buffer.surfaceMesh) {
    buffer.surfaceMesh->SetVisible(buffer.viz.showSurface);
  }
}

void ApplyWireframeVisibility(StageBuffer const& buffer) {
  if (buffer.wireframeMesh) {
    buffer.wireframeMesh->SetVisible(buffer.viz.showWireframe);
  }
}

// Icon button that flips @p value; the icon is dimmed while off. Returns true when clicked.
bool IconToggle(char const* label, bool& value) {
  ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
  ImGui::PushStyleColor(
      ImGuiCol_Text, ImGui::GetStyleColorVec4(value ? ImGuiCol_Text : ImGuiCol_TextDisabled));
  bool const clicked = ImGui::Button(label);
  ImGui::PopStyleColor(2);
  if (clicked) {
    value = !value;
  }
  return clicked;
}

// Hover tooltip for the most recently submitted widget (shown even when disabled); long text wraps.
void HoverTooltip(char const* text) {
  if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
    ImGui::PushStyleColor(ImGuiCol_PopupBg, IM_COL32(45, 55, 72, 230));
    ImGui::BeginTooltip();
    ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
    ImGui::TextUnformatted(text);
    ImGui::PopTextWrapPos();
    ImGui::EndTooltip();
    ImGui::PopStyleColor();
  }
}

// Surface / wireframe icon toggles for a header row, mirroring a mesh's Surface / Wireframe
// checkboxes. @p what names the mesh in the tooltips.
void DrawVizToggles(
    char const* idScope,
    char const* what,
    bool& showSurface,
    bool& showWireframe,
    std::function<void()> const& onShowSurface,
    std::function<void()> const& onShowWireframe) {
  ImGui::PushID(idScope);
  if (IconToggle(ICON_FA_CUBE, showSurface)) {
    onShowSurface();
  }
  HoverTooltip(("Show the " + std::string(what) + " surface.").c_str());
  ImGui::SameLine();
  if (IconToggle(ICON_FA_DRAW_POLYGON, showWireframe)) {
    onShowWireframe();
  }
  HoverTooltip(("Show the " + std::string(what) + " wireframe.").c_str());
  ImGui::PopID();
}

void DrawVizToggles(char const* idScope, char const* what, StageBuffer& buffer) {
  DrawVizToggles(
      idScope,
      what,
      buffer.viz.showSurface,
      buffer.viz.showWireframe,
      [&buffer] { ApplySurfaceVisibility(buffer); },
      [&buffer] { ApplyWireframeVisibility(buffer); });
}

void SameLineAfterTogglePair() {
  ImGui::SameLine(0.0f, ImGui::GetStyle().ItemSpacing.x);
  ImGui::Dummy(ImVec2(0.0f, 0.0f));
  ImGui::SameLine();
}

// Width of @p pairs toggle pairs laid out with SameLineAfterTogglePair between them.
float VizTogglesWidth(int pairs) {
  ImGuiStyle const& style = ImGui::GetStyle();
  float const pair = ImGui::CalcTextSize(ICON_FA_CUBE).x +
      ImGui::CalcTextSize(ICON_FA_DRAW_POLYGON).x + style.FramePadding.x * 4.0f +
      style.ItemSpacing.x;
  return static_cast<float>(pairs) * pair +
      static_cast<float>(pairs - 1) * style.ItemSpacing.x * 2.0f;
}

// A framed CollapsingHeader, taller than default, with @p togglePairs pairs of icons drawn by
// @p drawToggles in a gutter between the arrow and the label. The header is submitted with only
// @p id ("###..."), so its open state is keyed as usual, and @p label is drawn after the gutter.
bool CollapsingHeaderWithToggles(
    std::string const& label,
    char const* id,
    ImGuiTreeNodeFlags flags,
    bool dim,
    int togglePairs,
    std::function<void()> const& drawToggles) {
  ImGuiStyle const& style = ImGui::GetStyle();
  ImVec4 const textColor = ImGui::GetStyleColorVec4(dim ? ImGuiCol_TextDisabled : ImGuiCol_Text);
  ImGui::PushStyleColor(ImGuiCol_Text, textColor);
  PushHeaderRowFramePadding();
  bool const open = ImGui::CollapsingHeader(id, flags | ImGuiTreeNodeFlags_AllowOverlap);
  ImGui::PopStyleVar();
  ImGui::PopStyleColor();
  ImVec2 const min = ImGui::GetItemRectMin();
  ImVec2 const max = ImGui::GetItemRectMax();

  float const outerExtend = IM_TRUNC(ImGui::GetCurrentWindow()->WindowPadding.x * 0.5f);
  float const labelX = min.x + outerExtend + ImGui::GetFontSize() + style.FramePadding.x * 3.0f;
  float const iconsY = min.y + HeaderExtraPadY();
  float const gutterPad = IM_TRUNC(style.FramePadding.x * 0.5f);
  ImVec2 const gutterMin(labelX - style.FramePadding.x - gutterPad, iconsY);
  ImVec2 const gutterMax(
      gutterMin.x + VizTogglesWidth(togglePairs) + gutterPad * 2.0f,
      iconsY + ImGui::GetFrameHeight());
  ImDrawList* const drawList = ImGui::GetWindowDrawList();
  drawList->AddRectFilled(
      gutterMin, gutterMax, ImGui::GetColorU32(ImGuiCol_FrameBg), style.FrameRounding);
  // Headers can share a scope (the fixed slots sit at the window root), so key the gutter and its
  // icons by this header's own id.
  ImGui::PushID(id);
  ImGui::SameLine();
  ImGui::SetCursorScreenPos(gutterMin);
  ImGui::SetNextItemAllowOverlap();
  ImGui::InvisibleButton(
      "##toggleGutter", ImVec2(gutterMax.x - gutterMin.x, gutterMax.y - gutterMin.y));
  ImGui::SameLine();
  ImGui::SetCursorScreenPos(ImVec2(gutterMin.x + gutterPad, iconsY));
  ImGui::BeginGroup();
  ImGui::GetCurrentWindow()->DC.IsSameLine = false;
  drawToggles();
  ImGui::EndGroup();
  ImGui::PopID();

  ImVec2 const textPos(
      gutterMax.x + style.ItemSpacing.x, min.y + style.FramePadding.y + HeaderExtraPadY());
  drawList->PushClipRect(min, ImVec2(max.x - style.FramePadding.x, max.y), true);
  drawList->AddText(textPos, ImGui::GetColorU32(textColor), label.c_str());
  drawList->PopClipRect();
  return open;
}

// Reads a render-model file into renderer-space mesh sections, so the editor can build a wireframe
// overlay (and process the geometry). Shares processing_mesh_utils' per-format space convention, so
// an .obj/.stl slot lines up with the equivalent .glb rather than sitting a quarter turn off.
std::vector<MeshSection> ReadRenderModelSections(mochi::Path const& path) {
  return processing::ReadSectionsInRenderSpace(path.ToString());
}

std::span<AssetType const> AssetTypesForModelSlot(ModelSlotType type) {
  static constexpr std::array<AssetType, 3> kModelTypes = {
      AssetType::CadModel, AssetType::RenderModel, AssetType::MochiModel};
  switch (type) {
    case ModelSlotType::CadModel:
      return std::span(kModelTypes).subspan(0, 1);
    case ModelSlotType::RenderModel:
      return std::span(kModelTypes).subspan(1, 1);
    case ModelSlotType::MochiModel:
      return std::span(kModelTypes).subspan(2, 1);
    case ModelSlotType::AnyModel:
      return kModelTypes;
  }
  // Only a value outside the enum gets here: -Wswitch flags a new ModelSlotType above.
  MOCHI_LOG_ERROR("Unknown model slot type %d", static_cast<int>(type));
  return {};
}

class StudioModifierHost final : public ModifierHost {
 public:
  explicit StudioModifierHost(SuperDexStudio& studio) : _studio(studio) {}

  void ModelSlot(char const* id, mochi::DynamicString& path, ModelSlotType type) override {
    AssetManager& assets = _studio.GetAssetManager();
    if (!path.empty()) {
      std::string const key(path);
      mochi::Path const slotted{key};
      // A path set by a loaded pipeline (rather than dropped) names a file the asset manager may
      // not have loaded yet; load it once so the slot shows its tile.
      Asset* asset = assets.FindAssetByPath(slotted);
      if (asset == nullptr && _loadAttempted.insert(key).second) {
        std::error_code error;
        if (std::filesystem::is_regular_file(slotted.AsFilesystemPath(), error)) {
          asset = assets.LoadAsset(slotted);
        }
      }
      // A CAD model has no thumbnail until it is tessellated. An STL is read directly, as for the
      // editor's CAD references; a STEP would need the slow tessellator, so it keeps none.
      auto* const cad = dynamic_cast<CadModelAsset*>(asset);
      if (cad != nullptr && !cad->RendersThumbnail() &&
          slotted.AsLowercaseString().ends_with(".stl") && _tessellated.insert(key).second) {
        cad->SetTessellation(ReadRenderModelSections(slotted), _studio.GetResourceManager());
      }
    }
    ImGui::AssetSlot(
        id, path, assets, &_studio, AssetTypesForModelSlot(type), /*acceptDragDropPayload=*/true);
  }
  std::string SaveFileDialog(
      char const* title,
      std::span<char const* const> filters,
      char const* filterDescription,
      std::string const& defaultPath) override {
    return SuperDexStudio::GetFileDialogPath(
               title,
               filters.data(),
               mochi::isize(filters),
               filterDescription,
               /*isSaveDialog=*/true,
               mochi::Path{defaultPath})
        .ToString();
  }
  bool ReflectedStruct(SReflect::StructTypeInfo const& type, void* value) override {
    return ImGui::SimpleReflectionStruct(type, value);
  }
  bool DragXYZ(char const* label, mochi::Real3& value, float speed, char const* format) override {
    return ImGui::DragRealXYZ(label, value, speed, 0.0f, 0.0f, format, 0, 1.0f);
  }
  bool DragTransformRT(char const* label, mochi::Quaternion& rotation, mochi::Real3& translation)
      override {
    return ImGui::DragTransformRT(label, rotation, translation);
  }
  mochi::Real3 PathColor(std::string_view path) override {
    ImVec4 const c = HashStringToColor(path);
    return mochi::Real3{c.x, c.y, c.z};
  }
  bool RunBackgroundTask(
      char const* title,
      char const* label,
      std::function<bool()> work,
      std::function<void()> onDone) override {
    std::vector<AsyncTask> tasks;
    tasks.push_back(
        AsyncTask{label, [work = std::move(work)](AsyncCancelToken const&) { return work(); }});
    return _studio.BeginAsyncTasks(
        title, std::move(tasks), [onDone = std::move(onDone)](bool /*allSucceeded*/) { onDone(); });
  }
  bool IsBackgroundTaskRunning() const override {
    return _studio.IsAsyncTasksRunning();
  }

 private:
  SuperDexStudio& _studio;
  // Slotted paths already loaded or tessellated for their tiles, so neither is retried each frame
  // (a missing or unreadable file stays missing until the editor is reopened).
  std::unordered_set<std::string> _loadAttempted;
  std::unordered_set<std::string> _tessellated;
};

} // namespace

//--------------------------------------------------------------------------------------------------
// MOCHI MODEL EDITOR
//--------------------------------------------------------------------------------------------------

ModelEditor::ModelEditor(SuperDexStudio* studio, MochiModelAsset* asset)
    : AssetEditor(studio, asset), _mochiModelAsset(asset), _originType(AssetType::MochiModel) {}

ModelEditor::ModelEditor(SuperDexStudio* studio, RenderModelAsset* asset)
    : AssetEditor(studio, asset), _renderModelAsset(asset), _originType(AssetType::RenderModel) {}

ModelEditor::ModelEditor(SuperDexStudio* studio, CadModelAsset* asset)
    : AssetEditor(studio, asset), _cadModelAsset(asset), _originType(AssetType::CadModel) {}

void ModelEditor::Initialize() {
  _viewport = Viewport::Create(_studio, _studio->GetViewSettings());
  _viewport->enableViewportPicking = false;
  // Measure tool (Ctrl+M). It does its own CPU ray cast, so it works here despite object picking
  // being off; every slot, reference model and processing stage is measurable.
  _viewport->GetMeasureTool()->SetTargetProvider(
      [this](std::vector<MeasureTarget>& out) { CollectMeasureTargets(out); });
  if (_cadModelAsset) {
    _cadModelPath = _cadModelAsset->GetPath().ToString();
  }
  if (_renderModelAsset) {
    _renderModelPath = _renderModelAsset->GetPath().ToString();
  }
  if (_mochiModelAsset) {
    _mochiModelPath = _mochiModelAsset->GetPath().ToString();
  }
  DiscoverSiblingSlots();
  // Make sure every slotted path is loaded so the Update* lookups below resolve it. The originating
  // asset is already loaded; discovered siblings may not be.
  {
    auto& assetManager = _studio->GetAssetManager();
    if (!_cadModelPath.empty()) {
      assetManager.LoadAsset(_cadModelPath);
    }
    if (!_renderModelPath.empty()) {
      assetManager.LoadAsset(_renderModelPath);
    }
    if (!_mochiModelPath.empty()) {
      assetManager.LoadAsset(_mochiModelPath);
    }
  }

  // Load this model's saved processing pipeline (if any) before building the slots, so a saved CAD
  // transform is applied by the Update* calls below. Starts empty when there is no file.
  LoadProcessingPipelineOnOpen();

  // Only the opened (double-clicked) model starts visible; slots auto-populated by sibling
  // discovery stay hidden until the user toggles them on. Mochi restores its surface+wireframe
  // default; CAD and render restore surface only. Set before the Update* calls, which apply these
  // flags when building each slot's meshes.
  if (_originType == AssetType::CadModel) {
    _showCadModelSurface = true;
  } else if (_originType == AssetType::RenderModel) {
    _showRenderModelSurface = true;
  } else if (_originType == AssetType::MochiModel) {
    _showMochiModelSurface = true;
    _showMochiModelWireframe = true;
  }

  UpdateCadModel();
  UpdateRenderModel();
  UpdateMochiModel();
  // A STEP tessellates in the background, so a CAD-only open has nothing to frame yet; the
  // tessellation focuses once its geometry arrives.
  if (!_studio->IsAsyncTasksRunning() || _viewport->HasVisibleGeometry()) {
    _viewport->FocusCameraOnScene();
  }
}

void ModelEditor::OnBeginFrame() {
  PollSlotFileChanges();
}

void ModelEditor::OnRender(Renderer const* renderer) {
  PumpReferenceCadTessellations();
  PumpHausdorffResults();
  // Keep the drop-shadow ground plane at the lowest point of all meshes. The model editor runs no
  // physics sim, so a per-frame update is safe and robustly tracks async loads (CAD tessellation),
  // modifier cascades, reference-model edits, and hot-reloads without wiring every change site.
  _viewport->UpdateGroundPlane();
  _viewport->RenderScene(renderer);
}

bool ModelEditor::SlotFileChanged(
    mochi::DynamicString const& path,
    std::filesystem::file_time_type stored) {
  if (path.empty()) {
    return false;
  }
  std::error_code ec;
  auto const mtime = std::filesystem::last_write_time(path.c_str(), ec);
  if (ec) {
    return false;
  }
  return mtime != stored;
}

void ModelEditor::PollSlotFileChanges() {
  // Never hot-reload while an async task is running: an in-flight export may be mid-write to one of
  // these slot files, and reloading a partially-written (or writer-locked) file races the writer
  // and crashes (e.g. exporting the SDF to the currently-slotted .mochi.h5). The next poll after
  // the task finishes picks up the completed file.
  if (_studio->IsAsyncTasksRunning()) {
    return;
  }
  auto const now = std::chrono::steady_clock::now();
  if (now - _lastSlotPollTime < kSlotPollInterval) {
    return;
  }
  _lastSlotPollTime = now;

  bool const cadChanged = SlotFileChanged(_cadModelPath, _cadModelMtime);
  bool const renderChanged = SlotFileChanged(_renderModelPath, _renderModelMtime);
  bool const mochiChanged = SlotFileChanged(_mochiModelPath, _mochiModelMtime);
  bool anyRefChanged = false;
  for (ReferenceModel const& rm : _referenceModels) {
    if (SlotFileChanged(rm.path, rm.mtime)) {
      anyRefChanged = true;
      break;
    }
  }
  if (!cadChanged && !renderChanged && !mochiChanged && !anyRefChanged) {
    return;
  }

  // Drop this editor's asset references so the changed assets can be unloaded, then re-read them
  // from disk. The Update*Model calls below re-fetch the reloaded asset, rebuild its viz, and
  // re-record its modification time; re-register + resync restores the references afterwards.
  // An asset still referenced elsewhere (a loaded bot asset or an open bot tab) cannot be
  // unloaded + recreated, so it is re-read in place instead (Asset::ReloadFromDisk) when its type
  // supports it -- this refreshes every consumer at once. Types without in-place reload keep their
  // stale cached copy (matches the asset browser's Reload behavior).
  auto& assetManager = _studio->GetAssetManager();
  assetManager.UnregisterReferencer(this);
  auto const reload = [&assetManager](mochi::DynamicString const& path) {
    if (path.empty()) {
      return;
    }
    Asset* const existing = assetManager.FindAssetByPath(path);
    // Referenced elsewhere: UnloadAssetByPath would refuse to unload (ref count > 0) and the
    // subsequent LoadAsset would hand back the stale cached asset. Re-read into the existing object
    // in place instead (same address, so nothing dangles).
    if (existing != nullptr && assetManager.GetPathReferenceCount(path) != 0) {
      if (!existing->ReloadFromDisk()) {
        MOCHI_LOG_WARNING(
            "Hot-reload: '%s' is still referenced and could not be reloaded in place; keeping the "
            "cached copy.",
            path.c_str());
      }
      return;
    }
    if (existing != nullptr) {
      assetManager.UnloadAssetByPath(path);
    }
    assetManager.LoadAsset(path);
    // The freshly loaded asset has no cached shapes, but Mochi's context file cache and the
    // ShapeHandles pinned on loaded prefabs still refer to the pre-change file.
    assetManager.InvalidateShapeCachesForPath(path);
  };
  if (cadChanged) {
    reload(_cadModelPath);
    UpdateCadModel();
  }
  if (renderChanged) {
    reload(_renderModelPath);
    UpdateRenderModel();
  }
  if (mochiChanged) {
    reload(_mochiModelPath);
    UpdateMochiModel();
  }
  // Reference models hot-reload the same way: re-read a changed file (a CAD STEP reference re-arms
  // its async tessellation via UpdateReferenceModel, which the pump then runs).
  for (ReferenceModel& rm : _referenceModels) {
    if (SlotFileChanged(rm.path, rm.mtime)) {
      reload(rm.path);
      UpdateReferenceModel(rm);
    }
  }
  // A reloaded slot's Asset object was freed and recreated; the Update*Model calls above refreshed
  // the per-slot pointers, but the editor's base _asset (used for the tab -- GetAsset() /
  // GetTabColor() / GetTabDisplayName()) may alias a reloaded slot and would otherwise dangle, so
  // re-point it to the current origin asset. (Reproduced as a crash in ShowAssetEditorWindow when
  // the editor was opened for a mochi model that an export then overwrote.)
  Asset* origin = nullptr;
  if (_originType == AssetType::CadModel) {
    origin = _cadModelAsset;
  } else if (_originType == AssetType::RenderModel) {
    origin = _renderModelAsset;
  } else if (_originType == AssetType::MochiModel) {
    origin = _mochiModelAsset;
  }
  if (origin != nullptr) {
    _asset = origin;
  }
  assetManager.RegisterReferencer(this);
  assetManager.ResyncReferencer(this);
}

void ModelEditor::Shutdown() {
  // Persist the pipeline so reopening this model restores it (also saved on generate / build).
  SaveProcessingPipelineToDisk();
  // All scene objects are owned by the viewport's render scene, so they are destroyed with it --
  // except that the file-colored CAD surface's backing model lives in the resource manager, so its
  // instance goes first and then the model.
  if (_cadColorModel != nullptr) {
    _viewport->GetRenderScene()->DestroySceneObject(_cadModelSurfaceMesh);
    ReleaseCadColorModel();
  }
  _cadModelSurfaceMesh = nullptr;
  _cadModelWireframeMesh = nullptr;
  _renderModelSurfaceMesh = nullptr;
  _renderModelWireframeMesh = nullptr;
  _mochiModelSurfaceMesh = nullptr;
  _mochiModelWireframeMesh = nullptr;
  _sdfSurfaceMesh = nullptr;
  _sdfWireframeMesh = nullptr;
  for (StageBuffer* buffer : {&_contactSkinBuffer, &_visualMeshBuffer}) {
    buffer->surfaceMesh = nullptr;
    buffer->wireframeMesh = nullptr;
  }
  for (auto const& modifier : _modifiers) {
    modifier->output.surfaceMesh = nullptr;
    modifier->output.wireframeMesh = nullptr;
    modifier->inputView.surfaceMesh = nullptr;
    modifier->inputView.wireframeMesh = nullptr;
  }
  for (ReferenceModel& rm : _referenceModels) {
    rm.buffer.surfaceMesh = nullptr;
    rm.buffer.wireframeMesh = nullptr;
  }
  _viewport.reset();
}

bool ModelEditor::RepresentsAsset(Asset const* asset) const {
  // The editor represents its CAD / render / mochi slot assets (the primary origin asset is one of
  // these). Opening any of them should focus this editor rather than spawn a new instance.
  if (asset == nullptr) {
    return false;
  }
  return asset == _cadModelAsset || asset == _renderModelAsset || asset == _mochiModelAsset;
}

void ModelEditor::OnReopenedFor(Asset* asset) {
  // Focus visualization on the reopened slot, matching a fresh open: show that slot's default
  // visualization and hide the other two slots (and the collision model's SDF, contact skin and
  // visual mesh). The slot meshes already exist (built on open), so apply the new flags to them
  // directly.
  bool const isCad = asset != nullptr && asset == _cadModelAsset;
  bool const isRender = asset != nullptr && asset == _renderModelAsset;
  bool const isMochi = asset != nullptr && asset == _mochiModelAsset;
  if (!isCad && !isRender && !isMochi) {
    return; // not one of our slots (RepresentsAsset should have gated this)
  }
  _showCadModelSurface = isCad;
  _showCadModelWireframe = false;
  _showRenderModelSurface = isRender;
  _showRenderModelWireframe = false;
  _showMochiModelSurface = isMochi;
  _showMochiModelWireframe = isMochi; // a mochi origin shows its wireframe too (see Initialize)
  _showSdfSurface = false;
  _showSdfWireframe = false;

  if (_cadModelSurfaceMesh) {
    _cadModelSurfaceMesh->SetVisible(_showCadModelSurface);
  }
  if (_cadModelWireframeMesh) {
    _cadModelWireframeMesh->SetVisible(_showCadModelWireframe);
  }
  if (_renderModelSurfaceMesh) {
    _renderModelSurfaceMesh->SetVisible(_showRenderModelSurface);
  }
  if (_renderModelWireframeMesh) {
    _renderModelWireframeMesh->SetVisible(_showRenderModelWireframe);
  }
  if (_mochiModelSurfaceMesh) {
    _mochiModelSurfaceMesh->SetVisible(_showMochiModelSurface);
  }
  if (_mochiModelWireframeMesh) {
    _mochiModelWireframeMesh->SetVisible(_showMochiModelWireframe);
  }
  if (_sdfSurfaceMesh) {
    _sdfSurfaceMesh->SetVisible(_showSdfSurface);
  }
  if (_sdfWireframeMesh) {
    _sdfWireframeMesh->SetVisible(_showSdfWireframe);
  }
  for (StageBuffer* buffer : {&_contactSkinBuffer, &_visualMeshBuffer}) {
    buffer->viz.showSurface = false;
    buffer->viz.showWireframe = false;
    ApplySurfaceVisibility(*buffer);
    ApplyWireframeVisibility(*buffer);
  }
}

void ModelEditor::Refresh() {
  // A slotted model was replaced/renamed elsewhere in the app. Re-resolve each slot from its
  // current path and rebuild its visualization (same as a fresh open in Initialize).
  UpdateCadModel();
  UpdateRenderModel();
  UpdateMochiModel();
}

void ModelEditor::ForEachReferencedPath(
    std::function<void(mochi::Path const&)> const& callback) const {
  // Pin whichever assets are currently slotted so they cannot be deleted while shown.
  if (_cadModelAsset) {
    callback(_cadModelAsset->GetPath());
  }
  if (_renderModelAsset) {
    callback(_renderModelAsset->GetPath());
  }
  if (_mochiModelAsset) {
    callback(_mochiModelAsset->GetPath());
  }
  // Pin each reference model's asset too, so it stays loaded while shown.
  for (ReferenceModel const& rm : _referenceModels) {
    if (!rm.path.empty()) {
      callback(mochi::Path(rm.path));
    }
  }
}

void ModelEditor::DiscoverSiblingSlots() {
  // Exactly one slot is populated when a single asset is opened; that is the origin. Use its path
  // and base name to locate same-named files of the other two types.
  mochi::DynamicString const originPath = OriginModelPath();
  if (originPath.empty()) {
    return;
  }

  mochi::Path const origin = originPath;
  std::string const baseName = GetAssetNameFromPath(origin);
  mochi::Path const originDir = origin.GetParentPath();

  /* A file is associated with the other two only if discovery for its own type resolves back to
   * it. That one condition is what makes the association a partition rather than an arbitrary
   * graph: the set for a base name is {winner in cad, winner in render, winner in collision}, so
   * opening any member opens that same set, and a file some other candidate outranks -- a copy in
   * `render/internal/` shadowed by one in `render/` -- opens by itself rather than dragging in
   * models that belong to the winner. */
  std::string const canonical =
      FindAssetForSlot(baseName, ClassifyAssetTypeByPath(origin), originDir);
  _originIsShadowed = !canonical.empty() &&
      superdex::robotics::NormalizeBotPath(canonical) !=
          superdex::robotics::NormalizeBotPath(origin.AsFilesystemPath());
  if (_originIsShadowed) {
    MOCHI_LOG_WARNING(
        "Opened '%s' on its own: '%s' is the model the rest of '%s' associates with, so no other "
        "slots were filled. Its processing pipeline is kept separately, under this file's own name "
        "in the intermediates folder, so it will not disturb the other model's.",
        origin.ToString().c_str(),
        canonical.c_str(),
        baseName.c_str());
    return;
  }

  // (A JSON sidecar describing explicit slot paths will take precedence over name matching here in
  // the future.)
  auto discover = [&](mochi::DynamicString& slot, AssetType type) {
    if (!slot.empty()) {
      return;
    }
    std::string found = FindAssetForSlot(baseName, type, originDir);
    if (!found.empty()) {
      slot = found;
    }
  };

  discover(_cadModelPath, AssetType::CadModel);
  discover(_renderModelPath, AssetType::RenderModel);
  discover(_mochiModelPath, AssetType::MochiModel);
}

void ModelEditor::UpdateCadModel() {
  _cadModelAsset = _studio->GetAssetManager().FindAssetByPath<CadModelAsset>(_cadModelPath);
  if (_cadModelAsset) {
    // Default the slot color to the path-hash color (same scheme as mochi models).
    ImVec4 const c = HashStringToColor(_cadModelAsset->GetPath().ToString());
    _cadColor = {c.x, c.y, c.z};
  }
  // Drop any previous CAD geometry now (the slot changed); the new geometry is produced
  // asynchronously by RegenerateCadModel below.
  _cadSectionsOriginal.clear();
  _cadSections.clear();
  _cadHasFileColors = false;
  RebuildCadModelMeshes();
  // The processing-stage buffers derive from the CAD model, so they are stale once the slot
  // changes.
  ResetStageBuffers();
  if (_cadModelAsset) {
    RegenerateCadModel();
  }
  _cadModelMtime = FileMtimeOrDefault(_cadModelPath);
}

void ModelEditor::RebuildCadModelMeshes() {
  auto* scene = _viewport->GetRenderScene();
  if (_cadModelSurfaceMesh) {
    scene->DestroySceneObject(_cadModelSurfaceMesh);
    _cadModelSurfaceMesh = nullptr;
  }
  ReleaseCadColorModel();
  if (_cadModelWireframeMesh) {
    scene->DestroySceneObject(_cadModelWireframeMesh);
    _cadModelWireframeMesh = nullptr;
  }

  _cadStats.base = FormatMeshStats(_cadSections);
  _cadStats.fileSizeBytes = OnDiskFileSize(_cadModelPath);
  _cadStats.fileSizeLabel = "File Size";
  ComposeStats(_cadStats);

  // The CAD model is rendered from its mesh-section backing: in the file's own colors (one PBR
  // primitive per section) or flat-lit in the picker color, with a wireframe overlay built from the
  // same geometry.
  std::vector<float> positions;
  std::vector<float> normals;
  std::vector<int> indices;
  MergeSections(_cadSections, positions, normals, indices);
  if (indices.empty()) {
    return;
  }

  auto& resourceManager = _studio->GetResourceManager();
  if (ShowsCadFileColors()) {
    // Keyed per editor: two editors may show the same CAD file under different transforms.
    mochi::Path const key{
        "superdex_cad_viewport_" + std::to_string(reinterpret_cast<std::uintptr_t>(this))};
    _cadColorModel = resourceManager.LoadModelFromMeshSections(
        key, _cadSections, mochi_renderer::RenderModelFormat::Cad);
    if (_cadColorModel) {
      _cadModelSurfaceMesh = scene->AddSceneObjectToScene(_cadColorModel->GetInstance());
    }
  } else {
    auto surfaceMaterial = _cadModelOpacity >= 1.0f
        ? resourceManager.CreateFlatLitOpaqueMaterial(ToFloat3(_cadColor))
        : resourceManager.CreateFlatLitSeeThroughMaterial(ToFloat3(_cadColor), _cadModelOpacity);
    auto surface = mochi_renderer::Mesh::CreateMesh(
        _studio->GetEngine(),
        mochi::Span<float const>(positions.data(), positions.size()),
        mochi::Span<float const>(normals.data(), normals.size()),
        mochi::Span<int const>(indices.data(), indices.size()),
        surfaceMaterial,
        /*isDynamic=*/false,
        /*isClosed=*/true);
    if (surface) {
      _cadModelSurfaceMesh = scene->AddSceneObjectToScene(std::move(surface));
    }
  }
  if (_cadModelSurfaceMesh) {
    _cadModelSurfaceMesh->SetVisible(_showCadModelSurface);
  }

  filament::math::float3 const cadWire = WireColor(_cadColor);
  auto wireframe = mochi_renderer::WireframeMesh::CreateWireframeMesh(
      _studio->GetEngine(),
      mochi::Span<float const>(positions.data(), positions.size()),
      mochi::Span<float const>(normals.data(), normals.size()),
      mochi::Span<int const>(indices.data(), indices.size()),
      resourceManager.CreateWireframeMaterial({cadWire.x, cadWire.y, cadWire.z, 1.0f}),
      resourceManager.CreateWireframeDepthMaterial());
  if (wireframe) {
    _cadModelWireframeMesh = scene->AddSceneObjectToScene(std::move(wireframe));
    _cadModelWireframeMesh->SetVisible(_showCadModelWireframe);
  }
}

bool ModelEditor::ShowsCadFileColors() const {
  return _cadHasFileColors && !_cadOverrideColor && _cadModelOpacity >= 1.0f;
}

void ModelEditor::ReleaseCadColorModel() {
  if (_cadColorModel != nullptr) {
    _studio->GetResourceManager().UnloadResource(_cadColorModel->GetPath());
    _cadColorModel = nullptr;
  }
}

void ModelEditor::RegenerateCadModel() {
  if (!_cadModelAsset) {
    MOCHI_LOG_WARNING("CAD tessellation skipped: no CAD model is slotted.");
    return;
  }
  mochi::Path const cadPath = _cadModelAsset->GetPath();

  // STL CAD models are already triangle meshes, so read them directly (fast, synchronous) instead
  // of tessellating. Only STEP files need the OpenCascade tessellator (in the superdex_mesh_cli
  // helper), which is slow and therefore runs asynchronously below.
  if (cadPath.AsLowercaseString().ends_with(".stl")) {
    std::vector<MeshSection> sections = ReadRenderModelSections(cadPath);
    if (sections.empty()) {
      MOCHI_LOG_WARNING("CAD STL read produced no geometry; viewport left empty.");
      return;
    }
    _cadModelAsset->SetTessellation(sections, _studio->GetResourceManager());
    _cadSectionsOriginal = std::move(sections);
    _cadHasFileColors = false;
    bool const wasEmpty = !_viewport->HasVisibleGeometry();
    ApplyCadModelTransform();
    if (wasEmpty) {
      _viewport->FocusCameraOnScene();
    }
    return;
  }

  // The CAD Model section shows a standalone preview tessellation of the STEP file (independent of
  // the processing modifier stack, which has its own STEP-tessellation source). It uses fixed
  // default deflections; the stack's source is where tessellation is tuned.
  constexpr double kCadViewLinearDeflection = 0.05; // mm
  constexpr double kCadViewAngularDeflection = 0.25; // rad
  mochi::mesh::StepTessellationParams params;
  params.linearDeflection = kCadViewLinearDeflection;
  params.angularDeflection = kCadViewAngularDeflection;
  MOCHI_LOG(
      "Requesting CAD tessellation: %s (linear=%.4f, angular=%.4f)",
      cadPath.ToString().c_str(),
      params.linearDeflection,
      params.angularDeflection);

  // The worker thread tessellates into this buffer (no renderer access); onComplete consumes it on
  // the main thread. The AsyncTaskRunner modal blocks input while it runs, so `this` and the asset
  // cannot be torn down before onComplete fires.
  auto result = std::make_shared<std::vector<MeshSection>>();
  std::vector<AsyncTask> tasks;
  // CLI-backed: a cancel aborts it by killing the helper subprocess, so the token is unused here.
  tasks.push_back(
      AsyncTask{
          "Tessellate " + cadPath.GetFilename(),
          [cadPath, params, result](AsyncCancelToken const&) {
            mochi::ErrorLog error;
            *result = TessellateCadModelFile(cadPath, params, error);
            return error.IsOK() && !result->empty();
          }});

  bool const started = _studio->BeginAsyncTasks(
      "Tessellating CAD Model", std::move(tasks), [this, result](bool allSucceeded) {
        if (!allSucceeded || !_cadModelAsset) {
          MOCHI_LOG_WARNING("CAD tessellation produced no geometry; viewport left empty.");
          return;
        }
        _cadModelAsset->SetTessellation(*result, _studio->GetResourceManager());
        // Keep the raw tessellation so the CAD transform can be re-applied cheaply; ApplyCadModel
        // transform derives the displayed _cadSections from it and rebuilds the meshes.
        _cadSectionsOriginal = std::move(*result);
        _cadHasFileColors = true;
        // Frame the geometry when it is the first thing to show, as on a CAD-only open. Slotted
        // siblings start hidden, so their paths alone do not mean the scene had anything to frame.
        bool const wasEmpty = !_viewport->HasVisibleGeometry();
        ApplyCadModelTransform();
        if (wasEmpty) {
          _viewport->FocusCameraOnScene();
        }
      });
  if (!started) {
    MOCHI_LOG_WARNING("CAD tessellation not started: another async task is already running.");
  }
}

namespace {
// Applies scale, then rotation, then translation to each section's positions in place (the
// renderer's local-transform convention). Any stored normals are rotated to keep the buffer
// self-consistent for later processing; a unit quaternion preserves their length, and the rendered
// normals are recomputed from the transformed positions when the meshes are rebuilt. Shared by the
// CAD transform and the per-reference-model transform.
void ApplySectionsTransform(
    std::vector<MeshSection>& sections,
    mochi::Real3 const& scale,
    mochi::Quaternion const& rotation,
    mochi::Real3 const& translation) {
  using namespace mochi; // for Real3 / Quaternion arithmetic
  for (auto& section : sections) {
    for (std::size_t i = 0; i + 3 <= section.positions.size(); i += 3) {
      Real3 const local{section.positions[i], section.positions[i + 1], section.positions[i + 2]};
      Real3 const world = rotation * (local * scale) + translation;
      section.positions[i] = static_cast<float>(world[0]);
      section.positions[i + 1] = static_cast<float>(world[1]);
      section.positions[i + 2] = static_cast<float>(world[2]);
    }
    for (std::size_t i = 0; i + 3 <= section.normals.size(); i += 3) {
      Real3 const normal{section.normals[i], section.normals[i + 1], section.normals[i + 2]};
      Real3 const rotated = rotation * normal;
      section.normals[i] = static_cast<float>(rotated[0]);
      section.normals[i + 1] = static_cast<float>(rotated[1]);
      section.normals[i + 2] = static_cast<float>(rotated[2]);
    }
  }
}
} // namespace

void ModelEditor::ApplyCadTransform(std::vector<MeshSection>& sections) const {
  ApplySectionsTransform(sections, _cadScale, _cadRotation, _cadTranslation);
}

void ModelEditor::ApplyCadModelTransform() {
  // Re-derive the displayed viewing buffer from the untransformed tessellation. The processing
  // stages bake the transform into their own buffers at generation time (per-stage manual), so they
  // are not re-derived here.
  _cadSections = _cadSectionsOriginal;
  ApplyCadTransform(_cadSections);
  RebuildCadModelMeshes();
}

std::string ModelEditor::StageColorSeed(MeshProcessingModifier const& modifier) const {
  // Seed the hash with the opened model's file path (first populated slot) plus this stage's header
  // label, so each stage gets a stable, distinct color that also differs between models.
  // DynamicString carries a custom allocator, so it converts to std::string only element-wise.
  mochi::DynamicString const origin = OriginModelPath();
  std::string seed{origin.data(), origin.size()};
  seed += '|';
  seed += modifier.HeaderLabel();
  return seed;
}

mochi::Real3 ModelEditor::StageOutputColor(MeshProcessingModifier const& modifier) const {
  // A method may pin its own display color (e.g. Export Mesh File uses its configured material
  // color so the preview matches what it writes); otherwise derive a hashed stage color below.
  if (auto const preferred = modifier.PreferredDisplayColor()) {
    return *preferred;
  }
  ImVec4 const c = HashStringToColor(StageColorSeed(modifier));
  return {c.x, c.y, c.z};
}

mochi::Real3 ModelEditor::StageRoleColor(
    MeshProcessingModifier const& modifier,
    std::string_view role) const {
  if (auto const preferred = modifier.PreferredDisplayColor()) {
    return *preferred;
  }
  return HashedMeshColor(StageColorSeed(modifier), role);
}

void ModelEditor::RebuildStageMeshes(StageBuffer& stage, mochi::Real3 const& surfaceColor) {
  auto* scene = _viewport->GetRenderScene();
  if (stage.surfaceMesh) {
    scene->DestroySceneObject(stage.surfaceMesh);
    stage.surfaceMesh = nullptr;
  }
  if (stage.wireframeMesh) {
    scene->DestroySceneObject(stage.wireframeMesh);
    stage.wireframeMesh = nullptr;
  }

  stage.stats.base = FormatMeshStats(stage.sections);
  ComposeStats(stage.stats);

  std::vector<float> positions;
  std::vector<float> normals;
  std::vector<int> indices;
  MergeSections(stage.sections, positions, normals, indices);
  if (indices.empty()) {
    return;
  }

  auto& resourceManager = _studio->GetResourceManager();
  filament::math::float3 const surfColor{
      static_cast<float>(surfaceColor[0]),
      static_cast<float>(surfaceColor[1]),
      static_cast<float>(surfaceColor[2])};
  auto surfaceMaterial = stage.viz.opacity >= 1.0f
      ? resourceManager.CreateFlatLitOpaqueMaterial(surfColor)
      : resourceManager.CreateFlatLitSeeThroughMaterial(surfColor, stage.viz.opacity);
  auto surface = mochi_renderer::Mesh::CreateMesh(
      _studio->GetEngine(),
      mochi::Span<float const>(positions.data(), positions.size()),
      mochi::Span<float const>(normals.data(), normals.size()),
      mochi::Span<int const>(indices.data(), indices.size()),
      surfaceMaterial,
      /*isDynamic=*/false,
      /*isClosed=*/false);
  if (surface) {
    stage.surfaceMesh = scene->AddSceneObjectToScene(std::move(surface));
    stage.surfaceMesh->SetVisible(stage.viz.showSurface);
  }

  filament::math::float3 const wire = WireColor(surfaceColor);
  auto wireframe = mochi_renderer::WireframeMesh::CreateWireframeMesh(
      _studio->GetEngine(),
      mochi::Span<float const>(positions.data(), positions.size()),
      mochi::Span<float const>(normals.data(), normals.size()),
      mochi::Span<int const>(indices.data(), indices.size()),
      resourceManager.CreateWireframeMaterial({wire.x, wire.y, wire.z, 1.0f}),
      resourceManager.CreateWireframeDepthMaterial());
  if (wireframe) {
    stage.wireframeMesh = scene->AddSceneObjectToScene(std::move(wireframe));
    stage.wireframeMesh->SetVisible(stage.viz.showWireframe);
  }
}

void ModelEditor::DrawStageVizControls(
    char const* idScope,
    StageBuffer& buffer,
    std::function<mochi::Real3()> const& colorFn) {
  DrawMeshVizControls(
      idScope,
      buffer.viz.showSurface,
      buffer.viz.showWireframe,
      buffer.viz.showStats,
      buffer.viz.opacity,
      buffer.stats.display,
      _studio->GetFont("Roboto Mono"),
      [&buffer] { ApplySurfaceVisibility(buffer); },
      [&buffer] { ApplyWireframeVisibility(buffer); },
      [this, &buffer, &colorFn] {
        if (!buffer.surfaceMesh) {
          return;
        }
        auto& resourceManager = _studio->GetResourceManager();
        filament::math::float3 const color = ToFloat3(colorFn());
        buffer.surfaceMesh->SetMaterial(
            buffer.viz.opacity >= 1.0f
                ? resourceManager.CreateFlatLitOpaqueMaterial(color)
                : resourceManager.CreateFlatLitSeeThroughMaterial(color, buffer.viz.opacity));
      });
}

void ModelEditor::RecolorWireframe(
    mochi_renderer::WireframeMesh* wireframe,
    mochi::Real3 const& surfaceColor) {
  if (wireframe == nullptr) {
    return;
  }
  filament::math::float3 const c = WireColor(surfaceColor);
  wireframe->SetColor({c.x, c.y, c.z, 1.0f});
}

void ModelEditor::DestroyStageBufferMeshes(StageBuffer& stage) {
  auto* scene = _viewport->GetRenderScene();
  if (stage.surfaceMesh) {
    scene->DestroySceneObject(stage.surfaceMesh);
    stage.surfaceMesh = nullptr;
  }
  if (stage.wireframeMesh) {
    scene->DestroySceneObject(stage.wireframeMesh);
    stage.wireframeMesh = nullptr;
  }
}

void ModelEditor::ResetStageBuffers() {
  // Invalidate every modifier's output geometry (e.g. when the CAD slot changes). The modifiers
  // themselves (the recipe) are kept; only their buffers are dropped.
  // Every stage output is being invalidated, so cancel all in-flight Hausdorff jobs.
  _hausdorffQueue.Cancel(0);
  for (auto const& modifier : _modifiers) {
    DestroyStageBufferMeshes(modifier->output);
    modifier->output.sections.clear();
    modifier->output.tetMesh = {};
    modifier->output.embedded = {};
    modifier->output.stats = {};
    // The optional input-view buffer (Export Mochi Model) is invalidated with the output.
    DestroyStageBufferMeshes(modifier->inputView);
    modifier->inputView.sections.clear();
    modifier->inputView.stats = {};
    // Reset change-detection state so the next Generate treats the buffer as never-generated.
    modifier->outputGenId = 0;
    modifier->inputGenIdAtLastGen = 0;
    modifier->referenceGenIdAtLastGen = 0;
    modifier->lastGenSignature.clear();
  }
}

void ModelEditor::UpdateRenderModel() {
  auto* scene = _viewport->GetRenderScene();
  if (_renderModelWireframeMesh) {
    scene->DestroySceneObject(_renderModelWireframeMesh);
    _renderModelWireframeMesh = nullptr;
  }
  _renderSections.clear();
  _renderStats = {};
  _renderModelAsset =
      _studio->GetAssetManager().FindAssetByPath<RenderModelAsset>(_renderModelPath);
  _renderModelMtime = FileMtimeOrDefault(_renderModelPath);
  if (_renderModelAsset) {
    // Default the color-override color to the path-hash color (same scheme as mochi models).
    // Applies only when the override is enabled -- the render mesh shows its real textures by
    // default.
    ImVec4 const c = HashStringToColor(_renderModelAsset->GetPath().ToString());
    _renderColor = {c.x, c.y, c.z};
  }

  // Surface keeps the glTF/PBR look (RebuildRenderModelSurface handles opacity).
  RebuildRenderModelSurface();
  if (!_renderModelAsset) {
    return;
  }

  // Read the render file into sections to build a wireframe overlay (and back future processing).
  _renderSections = ReadRenderModelSections(_renderModelPath);
  _renderStats.base = FormatMeshStats(_renderSections);
  _renderStats.fileSizeBytes = OnDiskFileSize(_renderModelPath);
  _renderStats.fileSizeLabel = "File Size";
  ComposeStats(_renderStats);
  std::vector<float> positions;
  std::vector<float> normals;
  std::vector<int> indices;
  MergeSections(_renderSections, positions, normals, indices);
  if (indices.empty()) {
    return;
  }

  auto& resourceManager = _studio->GetResourceManager();
  filament::math::float3 const renderWire = WireColor(_renderColor);
  auto wireframe = mochi_renderer::WireframeMesh::CreateWireframeMesh(
      _studio->GetEngine(),
      mochi::Span<float const>(positions.data(), positions.size()),
      mochi::Span<float const>(normals.data(), normals.size()),
      mochi::Span<int const>(indices.data(), indices.size()),
      resourceManager.CreateWireframeMaterial({renderWire.x, renderWire.y, renderWire.z, 1.0f}),
      resourceManager.CreateWireframeDepthMaterial());
  if (wireframe) {
    _renderModelWireframeMesh = scene->AddSceneObjectToScene(std::move(wireframe));
    _renderModelWireframeMesh->SetVisible(_showRenderModelWireframe);
  }
}

void ModelEditor::RebuildRenderModelSurface() {
  auto* scene = _viewport->GetRenderScene();
  if (_renderModelSurfaceMesh) {
    scene->DestroySceneObject(_renderModelSurfaceMesh);
    _renderModelSurfaceMesh = nullptr;
  }
  if (!_renderModelAsset) {
    return;
  }
  _renderModelSurfaceMesh =
      scene->AddSceneObjectToScene(_renderModelAsset->GetRenderModelInstance());
  if (!_renderModelSurfaceMesh) {
    return;
  }
  // Keep the glTF PBR materials/textures at full opacity with no color override. Otherwise replace
  // them with a flat material in the picker color: the color override forces it opaque or not, and
  // a sub-1 opacity forces the translucent flat path regardless.
  if (_renderOverrideColor || _renderModelOpacity < 1.0f) {
    auto& rm = _studio->GetResourceManager();
    _renderModelSurfaceMesh->SetMaterial(
        _renderModelOpacity >= 1.0f
            ? rm.CreateFlatLitOpaqueMaterial(ToFloat3(_renderColor))
            : rm.CreateFlatLitSeeThroughMaterial(ToFloat3(_renderColor), _renderModelOpacity));
  }
  _renderModelSurfaceMesh->SetVisible(_showRenderModelSurface);
}

void ModelEditor::UpdateMochiModel() {
  auto* scene = _viewport->GetRenderScene();
  // The wireframe, SDF, contact skin and visual mesh derive from the mochi model, so tear them down
  // before swapping it.
  DestroyMochiModelWireframeMesh();
  DestroySdfMesh();
  DestroySdfWireframeMesh();
  for (StageBuffer* buffer : {&_contactSkinBuffer, &_visualMeshBuffer}) {
    DestroyStageBufferMeshes(*buffer);
    buffer->sections.clear();
    buffer->stats = {};
  }
  if (_mochiModelSurfaceMesh) {
    scene->DestroySceneObject(_mochiModelSurfaceMesh);
    _mochiModelSurfaceMesh = nullptr;
  }
  _mochiSections.clear();
  _mochiStats = {};
  _mochiModelAsset = _studio->GetAssetManager().FindAssetByPath<MochiModelAsset>(_mochiModelPath);
  _mochiModelMtime = FileMtimeOrDefault(_mochiModelPath);
  if (_mochiModelAsset) {
    // The Mochi model surface uses its canonical path-hash color (no role suffix), so the same
    // model shows the same color everywhere in the app (asset thumbnail, bot editor, etc.). The SDF
    // reconstruction has no path of its own, so give it a distinct hashed color via a role suffix.
    // No hard-coded colors.
    std::string const colorSeed = _mochiModelAsset->GetPath().ToString();
    ImVec4 const mochiHashColor = HashStringToColor(colorSeed);
    _mochiColor = {mochiHashColor.x, mochiHashColor.y, mochiHashColor.z};
    filament::math::float3 const sdfColor = _mochiModelAsset->GetSdfSurfaceColor();
    _sdfColor = {sdfColor.x, sdfColor.y, sdfColor.z};
    _contactSkinColor = HashedMeshColor(colorSeed, kContactSkinColorRole);
    _visualMeshColor = HashedMeshColor(colorSeed, kVisualMeshColorRole);
    // Working geometry backing (for future mesh processing). The surface/wireframe below are still
    // built from the model data directly.
    {
      auto const& converter = _studio->GetEditorToRendererSpaceConverter();
      std::vector<float> positions;
      std::vector<float> normals;
      std::vector<int> indices;
      if (BuildMochiModelGeometry(
              _mochiModelAsset->GetModelData(), &converter, positions, normals, indices)) {
        MeshSection section;
        section.positions = std::move(positions);
        section.normals = std::move(normals);
        section.indices = std::move(indices);
        section.hasNormals = !section.normals.empty();
        _mochiSections.push_back(std::move(section));
      }
    }
    // The editor shows the surface and wireframe as independent, separately toggleable meshes
    // (GetRenderModelInstance returns a combined flat-lit+wireframe mesh, which we do not want
    // here). Build all visualization geometry up front and keep it in the scene; the checkboxes
    // only toggle visibility.
    _mochiStats.base = FormatMeshStats(_mochiSections);
    _mochiStats.fileSizeBytes = OnDiskFileSize(_mochiModelPath);
    _mochiStats.fileSizeLabel = "File Size";
    if (_mochiModelAsset->GetModelData().mesh.has_value()) {
      _mochiStats.tetMesh = TetCounts(*_mochiModelAsset->GetModelData().mesh);
    }
    ComposeStats(_mochiStats);
    RebuildMochiModelSurfaceMesh();
    RebuildMochiModelWireframeMesh();
    RebuildSdfMeshes();
    mochi::ModelData const& modelData = _mochiModelAsset->GetModelData();
    auto const& converter = _studio->GetEditorToRendererSpaceConverter();
    if (modelData.contactSkinMesh.has_value()) {
      _contactSkinBuffer.sections = {ModelMeshToSection(*modelData.contactSkinMesh, converter)};
      RebuildStageMeshes(_contactSkinBuffer, _contactSkinColor);
    }
    if (modelData.visualMesh.has_value()) {
      _visualMeshBuffer.sections = {ModelMeshToSection(*modelData.visualMesh, converter)};
      RebuildStageMeshes(_visualMeshBuffer, _visualMeshColor);
    }
    ApplyModelTransform();
    if (_mochiModelSurfaceMesh) {
      _mochiModelSurfaceMesh->SetVisible(_showMochiModelSurface);
    }
    if (_mochiModelWireframeMesh) {
      _mochiModelWireframeMesh->SetVisible(_showMochiModelWireframe);
    }
    if (_sdfSurfaceMesh) {
      _sdfSurfaceMesh->SetVisible(_showSdfSurface);
    }
    if (_sdfWireframeMesh) {
      _sdfWireframeMesh->SetVisible(_showSdfWireframe);
    }
  }
}

void ModelEditor::ShowTabContents() {
  ImGui::BeginChild("Viewport_Child", ImVec2(0, 0), 0, ImGuiWindowFlags_NoMove);
  _viewport->ShowViewportContents(true);
  _viewport->ShowStatsOverlay();
  ImGui::EndChild();
}

void ModelEditor::ApplySceneViewSettings(mochi_renderer::SceneViewSettings const& viewSettings) {
  if (_viewport && _viewport->GetRenderScene()) {
    _viewport->GetRenderScene()->ApplyViewSettings(viewSettings);
  }
}

std::vector<AssetEditor::WindowDeclaration> ModelEditor::GetDefaultWindows() {
  using Dock = AssetEditor::DockRegion;
  return {
      // main windows
      {"Model Viewer", true, Dock::SidePanelTop},
      {"Model Processing", true, Dock::SidePanelBottom},
      MeasureWindowDeclaration(),
      // debug windows
      {"Render Scene Hierarchy", false, Dock::SidePanelTop, true},
      {"Render Scene Details", false, Dock::SidePanelBottom, true}};
}

std::vector<AssetEditor::WindowDeclaration> ModelEditor::GetAuxiliaryWindows() const {
  return GetDefaultWindows();
}

void ModelEditor::CollectMeasureTargets(std::vector<MeasureTarget>& out) const {
  MeasureGeometryCache& cache = _viewport->GetMeasureTool()->GetGeometryCache();
  auto add = [&](mochi_renderer::SceneObject* object,
                 std::vector<MeshSection> const& sections,
                 std::string label,
                 MeasureTargetKind kind) {
    if (object == nullptr || sections.empty()) {
      return;
    }
    out.push_back({std::move(label), kind, object, [&cache, &sections] {
                     return cache.GetForSections(&sections, sections);
                   }});
  };

  add(_cadModelSurfaceMesh, _cadSections, "CAD Model", MeasureTargetKind::Render);
  add(_renderModelSurfaceMesh, _renderSections, "Render Model", MeasureTargetKind::Render);
  add(_mochiModelSurfaceMesh, _mochiSections, "Mochi Model", MeasureTargetKind::Collision);
  for (ReferenceModel const& rm : _referenceModels) {
    MeasureTargetKind const kind =
        rm.type == AssetType::MochiModel ? MeasureTargetKind::Collision : MeasureTargetKind::Render;
    add(rm.buffer.surfaceMesh, rm.buffer.sections, mochi::Path(rm.path).GetFilename(), kind);
  }
  for (auto const& modifier : _modifiers) {
    // A stage's output is a working mesh with no collision/render identity of its own; treat it as
    // render geometry so the panel's default filter shows it.
    add(modifier->output.surfaceMesh,
        modifier->output.sections,
        modifier->HeaderLabel(),
        MeasureTargetKind::Render);
  }
}

void ModelEditor::ShowAuxiliaryWindows() {
  if (bool& open = _studio->GetWindowVisible("Model Viewer")) {
    ShowModelViewerWindow(&open);
  }
  if (bool& open = _studio->GetWindowVisible("Model Processing")) {
    ShowModelProcessingWindow(&open);
  }
  ShowMeasureWindow();
  // debug windows
  if (bool& open = _studio->GetWindowVisible("Render Scene Hierarchy")) {
    _viewport->ShowSceneHierarchyWindow("Render Scene Hierarchy", &open);
  }
  if (bool& open = _studio->GetWindowVisible("Render Scene Details")) {
    _viewport->ShowSelectedObjectDetailsWindow("Render Scene Details", &open);
  }
}

void ModelEditor::RebuildMochiModelSurfaceMesh() {
  // Reuse the geometry extracted once into _mochiSections (by UpdateMochiModel) instead of
  // re-running BuildMochiModelGeometry.
  if (!_mochiModelAsset || _mochiSections.empty()) {
    return;
  }
  auto const& section = _mochiSections.front();
  auto& resourceManager = _studio->GetResourceManager();
  filament::math::float3 const color = ToFloat3(_mochiColor);
  auto material = _mochiModelOpacity >= 1.0f
      ? resourceManager.CreateFlatLitOpaqueMaterial(color)
      : resourceManager.CreateFlatLitSeeThroughMaterial(color, _mochiModelOpacity);
  auto surface = mochi_renderer::Mesh::CreateMesh(
      _studio->GetEngine(),
      mochi::Span<float const>(section.positions.data(), section.positions.size()),
      mochi::Span<float const>(section.normals.data(), section.normals.size()),
      mochi::Span<int const>(section.indices.data(), section.indices.size()),
      material,
      /*isDynamic=*/false,
      /*isClosed=*/true);
  if (surface) {
    _mochiModelSurfaceMesh = _viewport->GetRenderScene()->AddSceneObjectToScene(std::move(surface));
    ApplyModelTransform();
  }
}

void ModelEditor::RebuildMochiModelWireframeMesh() {
  DestroyMochiModelWireframeMesh();
  // Reuse the geometry extracted once into _mochiSections (by UpdateMochiModel) instead of
  // re-running BuildMochiModelGeometry.
  if (!_mochiModelAsset || _mochiSections.empty()) {
    return;
  }
  auto const& section = _mochiSections.front();
  auto& resourceManager = _studio->GetResourceManager();
  // Derived from the current surface color so it tracks a color override (same app-wide scheme).
  filament::math::float3 const color = WireColor(_mochiColor);
  auto wireframe = mochi_renderer::WireframeMesh::CreateWireframeMesh(
      _studio->GetEngine(),
      mochi::Span<float const>(section.positions.data(), section.positions.size()),
      mochi::Span<float const>(section.normals.data(), section.normals.size()),
      mochi::Span<int const>(section.indices.data(), section.indices.size()),
      resourceManager.CreateWireframeMaterial({color.x, color.y, color.z, 1.0f}),
      resourceManager.CreateWireframeDepthMaterial());
  if (wireframe) {
    _mochiModelWireframeMesh =
        _viewport->GetRenderScene()->AddSceneObjectToScene(std::move(wireframe));
    ApplyModelTransform();
  }
}

void ModelEditor::DestroyMochiModelWireframeMesh() {
  if (_mochiModelWireframeMesh) {
    _viewport->GetRenderScene()->DestroySceneObject(_mochiModelWireframeMesh);
    _mochiModelWireframeMesh = nullptr;
  }
}

void ModelEditor::RebuildSdfMeshes() {
  DestroySdfMesh();
  DestroySdfWireframeMesh();
  // The asset reconstructs the SDF surface (a CLI subprocess -- the expensive part) once and caches
  // it for every viewport; build both the surface and wireframe meshes from that geometry.
  MochiModelAsset::SdfSurface const* const sdfSurface =
      _mochiModelAsset ? _mochiModelAsset->EnsureSdfSurface() : nullptr;
  if (sdfSurface == nullptr) {
    return; // DestroySdfMesh above already cleared _sdfStats
  }
  std::vector<float> const& positions = sdfSurface->positions;
  std::vector<float> const& normals = sdfSurface->normals;
  std::vector<int> const& indices = sdfSurface->indices;

  // The reconstructed SDF surface is a single welded mesh, so its statistics are exact. Reuse the
  // geometry (positions/indices are still needed below to build the meshes).
  {
    mochi_renderer::MeshSection statsSection;
    statsSection.positions = positions;
    statsSection.indices = indices;
    _sdfStats.base =
        FormatMeshStats(std::vector<mochi_renderer::MeshSection>{std::move(statsSection)});
    _sdfStats.sdfGrid = std::nullopt;
    if (_mochiModelAsset && _mochiModelAsset->GetModelData().sdf.has_value()) {
      _sdfStats.sdfGrid = _mochiModelAsset->GetModelData().sdf->dims;
    }
    ComposeStats(_sdfStats);
  }

  auto& resourceManager = _studio->GetResourceManager();
  auto* scene = _viewport->GetRenderScene();

  filament::math::float3 const surfColor = ToFloat3(_sdfColor);
  auto surfaceMaterial = _sdfOpacity >= 1.0f
      ? resourceManager.CreateFlatLitOpaqueMaterial(surfColor)
      : resourceManager.CreateFlatLitSeeThroughMaterial(surfColor, _sdfOpacity);
  auto surface = mochi_renderer::Mesh::CreateMesh(
      _studio->GetEngine(),
      mochi::Span<float const>(positions.data(), positions.size()),
      mochi::Span<float const>(normals.data(), normals.size()),
      mochi::Span<int const>(indices.data(), indices.size()),
      surfaceMaterial,
      /*isDynamic=*/false,
      /*isClosed=*/true);
  if (surface) {
    _sdfSurfaceMesh = scene->AddSceneObjectToScene(std::move(surface));
  }

  filament::math::float3 const wireColor = WireColor(_sdfColor);
  auto wireframe = mochi_renderer::WireframeMesh::CreateWireframeMesh(
      _studio->GetEngine(),
      mochi::Span<float const>(positions.data(), positions.size()),
      mochi::Span<float const>(normals.data(), normals.size()),
      mochi::Span<int const>(indices.data(), indices.size()),
      resourceManager.CreateWireframeMaterial({wireColor.x, wireColor.y, wireColor.z, 1.0f}),
      resourceManager.CreateWireframeDepthMaterial());
  if (wireframe) {
    _sdfWireframeMesh = scene->AddSceneObjectToScene(std::move(wireframe));
  }

  ApplyModelTransform();
}

void ModelEditor::DestroySdfMesh() {
  if (_sdfSurfaceMesh) {
    _viewport->GetRenderScene()->DestroySceneObject(_sdfSurfaceMesh);
    _sdfSurfaceMesh = nullptr;
  }
  // Clearing the SDF output clears its stats too (RebuildSdfMeshes recomputes them on success).
  _sdfStats = {};
}

void ModelEditor::DestroySdfWireframeMesh() {
  if (_sdfWireframeMesh) {
    _viewport->GetRenderScene()->DestroySceneObject(_sdfWireframeMesh);
    _sdfWireframeMesh = nullptr;
  }
}

void ModelEditor::ApplyModelTransform() {
  auto const* converter = &_studio->GetEditorToRendererSpaceConverter();
  if (_mochiModelSurfaceMesh) {
    _mochiModelSurfaceMesh->SetLocalTransform(
        _modelRotation, _modelTranslation, _modelScale, converter);
  }
  if (_mochiModelWireframeMesh) {
    _mochiModelWireframeMesh->SetLocalTransform(
        _modelRotation, _modelTranslation, _modelScale, converter);
  }
  if (_sdfSurfaceMesh) {
    _sdfSurfaceMesh->SetLocalTransform(_modelRotation, _modelTranslation, _modelScale, converter);
  }
  if (_sdfWireframeMesh) {
    _sdfWireframeMesh->SetLocalTransform(_modelRotation, _modelTranslation, _modelScale, converter);
  }
  for (StageBuffer const* buffer : {&_contactSkinBuffer, &_visualMeshBuffer}) {
    for (mochi_renderer::SceneObject* object : {buffer->surfaceMesh, buffer->wireframeMesh}) {
      if (object) {
        object->SetLocalTransform(_modelRotation, _modelTranslation, _modelScale, converter);
      }
    }
  }
}

void ModelEditor::ShowMeshModifierStack() {
  // The stack starts empty; the "Add Modifier" menu below offers source types so the user can add
  // the first source. Removing the lone source leaves the stack empty, and a source can be added
  // again from that menu.

  // Hover tooltip for the most recently submitted widget (shown even when disabled); long text
  // wraps.
  auto tooltip = [](char const* text) { HoverTooltip(text); };

  // Populate Default Processing: replace the stack with a shipped preset pipeline (from
  // processing_presets/ next to the studio exe). Selecting a preset loads it (confirmed first when
  // the stack is non-empty). Saving still targets this model's own _intermediates JSON.
  if (ImGui::BeginCombo("##populatepreset", "Select Preset", ImGuiComboFlags_HeightLarge)) {
    for (ProcessingPreset const& preset :
         DiscoverProcessingPresets(SuperDexStudio::GetExecutableDir())) {
      if (ImGui::Selectable(preset.name.c_str())) {
        PopulateFromPreset(preset.path);
      }
      if (!preset.description.empty()) {
        tooltip(preset.description.c_str());
      }
    }
    ImGui::EndCombo();
  }
  tooltip(
      "Replace the modifier stack with a shipped preset pipeline (from the processing_presets folder "
      "next to the studio exe).");

  ImGui::Separator();

  // Modifier bubble background, tinted by kind so sources/exports read at a glance: sources get a
  // slight blue tint, exports a slight orange tint, and transforms keep the neutral grey. Disabled
  // modifiers use a darker, muted variant of the same tint.
  auto bubbleColor = [](ModifierKind kind, bool enabled) -> ImU32 {
    switch (kind) {
      case ModifierKind::Source:
        return ImGui::GetColorU32(
            enabled ? ImVec4(0.21f, 0.24f, 0.34f, 1.0f) : ImVec4(0.11f, 0.13f, 0.20f, 1.0f));
      case ModifierKind::Transform:
        return ImGui::GetColorU32(
            enabled ? ImVec4(0.24f, 0.24f, 0.28f, 1.0f) : ImVec4(0.13f, 0.13f, 0.15f, 1.0f));
      case ModifierKind::Export:
        return ImGui::GetColorU32(
            enabled ? ImVec4(0.34f, 0.26f, 0.19f, 1.0f) : ImVec4(0.20f, 0.15f, 0.11f, 1.0f));
    }
    return ImGui::GetColorU32(ImVec4(0.24f, 0.24f, 0.28f, 1.0f)); // unreachable; all kinds handled
  };

  int removeIndex = -1;
  int saveOutputIndex = -1; // a modifier whose Save-output button was pressed this frame
  int saveIndex = -1; // an export modifier whose Export button was pressed this frame

  // GUI services shared by every modifier's ShowParams (the app widgets; the source file path is
  // what an export modifier's Browse offers as its suggestion, falling back to the opened model's
  // folder when there is no source).
  std::string const sourceFilePath = StackSourceFilePath(_modifiers, MakeRunContext());
  // Keep every Auto export path equal to what Browse would offer. Safe to do per frame: an Auto
  // path is never serialized, so however often it moves it cannot dirty the saved snapshot.
  RefreshAutoExportPaths(_modifiers, sourceFilePath);
  std::vector<bool> const exportPathCollides = FindCollidingExportPaths(_modifiers);
  // Display names by array index, so a modifier's reference dropdown (edge swap) can list the
  // elements above it. selfIndex is set per modifier just before its ShowParams call below.
  std::vector<std::string> modifierNames;
  modifierNames.reserve(_modifiers.size());
  for (auto const& modifier : _modifiers) {
    modifierNames.emplace_back(modifier->DisplayName());
  }
  StudioModifierHost host{*_studio};
  ModifierGuiContext gui{
      .tooltip = tooltip,
      .host = &host,
      .sourceFilePath = sourceFilePath,
      .modelFolder = OriginModelFolder(),
      .cadFilePath = _cadModelAsset ? _cadModelAsset->GetPath().ToString() : std::string()};
  gui.modifierNames = &modifierNames;

  DragReorderState& drag = _modifierDrag;
  ReorderableBubbleList list;
  list.count = _modifiers.size();
  list.id = [&](std::size_t i) { return _modifiers[i]->id; };
  list.bubbleColor = [&](std::size_t i, bool isPlaceholder) {
    return bubbleColor(_modifiers[i]->Kind(), isPlaceholder ? false : _modifiers[i]->enabled);
  };
  list.ghostLabel = [&](std::size_t i) { return _modifiers[i]->HeaderLabel(); };
  // A non-source cannot drop above the pinned source at index 0 (kept as a constraint we can lift
  // once multiple sources are supported).
  list.clampSlot = [&](int slot, int draggingIndex) {
    return _modifiers[static_cast<std::size_t>(draggingIndex)]->Kind() != ModifierKind::Source
        ? std::max(slot, 1)
        : slot;
  };
  list.move = [&](std::size_t from, std::size_t to) { MoveModifier(from, to); };
  list.drawContent = [&](std::size_t i) {
    MeshProcessingModifier& modifier = *_modifiers[i];

    // --- header row (after the drag grip): [index] [enable] [collapsible name] ... [trash remove]
    // --- Array index, right-justified in a fixed 2-digit-wide column so indices stay aligned once
    // they reach double digits. Drawing the text at (column right - text width) also keeps the
    // following widgets at a constant x regardless of digit count.
    ImGui::SameLine();
    float const indexColStart = ImGui::GetCursorPosX();
    float const indexColWidth = ImGui::CalcTextSize("00").x;
    std::string const indexText = std::to_string(i);
    float const indexTextWidth = ImGui::CalcTextSize(indexText.c_str()).x;
    // Right-justify within the column; clamp the pad to >= 0 so a 3+ digit index expands rightward
    // instead of overdrawing the grip.
    ImGui::SetCursorPosX(indexColStart + std::max(0.0f, indexColWidth - indexTextWidth));
    ImGui::TextUnformatted(indexText.c_str());

    ImGui::SameLine();
    float const rowY = ImGui::GetCursorScreenPos().y;
    ImGui::SetCursorScreenPos(ImVec2(ImGui::GetCursorScreenPos().x, rowY + HeaderExtraPadY()));
    ImGui::Checkbox("##enabled", &modifier.enabled);
    tooltip("When unchecked this modifier is skipped (passthrough).");
    ImGui::SameLine();
    // Reserve room on the right for the remove button so the framed header stops before it. A
    // framed CollapsingHeader always spans the full work rect, so temporarily shrink the work
    // rect's right edge around it (restored immediately after).
    // Match the AssetSlot clear-trashcan's natural button width (glyph + horizontal frame padding)
    // so the two trashcans render identically. GetFrameHeight() would force a square ~2px too
    // narrow (the trash glyph is slightly wider than the font cell is tall).
    float const removeWidth =
        ImGui::CalcTextSize(ICON_FA_TRASH).x + ImGui::GetStyle().FramePadding.x * 2.0f;
    float const removeGap = ImGui::GetStyle().ItemSpacing.x;
    ImGuiWindow* const window = ImGui::GetCurrentWindow();
    float const savedWorkRectMaxX = window->WorkRect.Max.x;
    window->WorkRect.Max.x -= removeWidth + removeGap;
    ImGui::SetNextItemOpen(!modifier.collapsed);
    // Mirrors of the Surface / Wireframe checkboxes in the bar. A disabled modifier's arrow and
    // label are greyed but stay clickable (visual only).
    bool const showsInput = modifier.ActiveMethod().ShowsInputVisualization();
    bool const bodyOpen = CollapsingHeaderWithToggles(
        modifier.HeaderLabel(),
        "###header",
        0,
        !modifier.enabled,
        showsInput ? 2 : 1,
        [&modifier, showsInput] {
          if (showsInput) {
            // Export Mochi Model's SDF method holds two meshes: the mesh it exports, then the SDF.
            DrawVizToggles("inputViz", "Mochi mesh", modifier.inputView);
            SameLineAfterTogglePair();
            DrawVizToggles("stageViz", "SDF", modifier.output);
          } else {
            DrawVizToggles("stageViz", "output", modifier.output);
          }
        });
    window->WorkRect.Max.x = savedWorkRectMaxX;
    modifier.collapsed = !bodyOpen;
    // Remove button in the reserved gap at the far right, as tall as the header bar.
    ImGui::SameLine();
    ImGui::SetCursorPosX(ImGui::GetContentRegionMax().x - removeWidth);
    PushHeaderRowFramePadding();
    bool const removeClicked = ImGui::Button(ICON_FA_TRASH);
    ImGui::PopStyleVar();
    if (removeClicked) {
      removeIndex = static_cast<int>(i);
    }
    tooltip("Remove this modifier.");

    if (bodyOpen) {
      // Parameters + Generate/Save grey out when the modifier is disabled.
      ImGui::BeginDisabled(!modifier.enabled);
      gui.selfIndex = i;
      gui.exportPathCollides = exportPathCollides[i];
      // Method selector, separate from the collapsing-header label: a modifier groups related
      // methods (e.g. Refine Mesh -> Make Manifold / Make Watertight / Edge Flip / ...). Only shown
      // when there is a choice; picking one persists it and the next Generate rebuilds this stage.
      if (modifier.MethodCount() > 1) {
        // Show each method's Description on hover -- of the option while the popup is open, and of
        // the collapsed selector for the active method -- skipping methods that supply none.
        auto methodTooltip = [&tooltip](MeshProcessingMethod const& method) {
          char const* const desc = method.Description();
          if (desc[0] != '\0') {
            tooltip(desc);
          }
        };
        if (ImGui::BeginCombo("Method", modifier.ActiveMethodName())) {
          for (int m = 0; m < static_cast<int>(modifier.MethodCount()); ++m) {
            MeshProcessingMethod const& method = *modifier.Methods()[static_cast<std::size_t>(m)];
            bool const selected = m == modifier.ActiveMethodIndex();
            if (ImGui::Selectable(method.Name(), selected)) {
              bool const wasShowingInput = modifier.ActiveMethod().ShowsInputVisualization();
              modifier.SelectMethod(m);
              bool const nowShowsInput = modifier.ActiveMethod().ShowsInputVisualization();
              if (wasShowingInput != nowShowsInput) {
                if (modifier.inputView.surfaceMesh) {
                  modifier.inputView.surfaceMesh->SetVisible(
                      nowShowsInput && modifier.inputView.viz.showSurface);
                }
                if (modifier.inputView.wireframeMesh) {
                  modifier.inputView.wireframeMesh->SetVisible(
                      nowShowsInput && modifier.inputView.viz.showWireframe);
                }
              }
              SaveProcessingPipelineToDisk();
            }
            methodTooltip(method);
            if (selected) {
              ImGui::SetItemDefaultFocus();
            }
          }
          ImGui::EndCombo();
        }
        methodTooltip(modifier.ActiveMethod());
      }
      modifier.ShowParams(gui);
      // Generate auto-cascades: it first rebuilds any stale modifiers above this one, then this
      // one. Deliberately NOT disabled when upstream buffers are stale/empty -- only when no usable
      // source is reachable.
      ImGui::BeginDisabled(!CanGenerateModifier(i));
      if (ImGui::Button("Generate")) {
        GenerateModifier(i);
      }
      ImGui::EndDisabled();
      tooltip(
          "Generate this modifier's output, first auto-rebuilding any stale modifiers above it. "
          "Stages already up to date are skipped.");
      // Save this modifier's current output as .vtk for a tetrahedral mesh, otherwise .glb/.obj.
      // The dialog defaults to the bot's _intermediates folder and is enabled once the modifier
      // has produced output.
      ImGui::SameLine();
      ImGui::BeginDisabled(modifier.output.sections.empty() || _studio->IsAsyncTasksRunning());
      if (ImGui::Button("Save")) {
        saveOutputIndex = static_cast<int>(i);
      }
      ImGui::EndDisabled();
      tooltip(
          modifier.output.tetMesh.GetNumElements() > 0
              ? "Save this modifier's current tetrahedral output to a .vtk file for inspection."
              : "Save this modifier's current surface output to a .glb or .obj file for "
                "inspection.");
      // Export modifiers additionally offer their configured-path export (writes their input mesh).
      if (modifier.ProvidesFileExport()) {
        ImGui::SameLine();
        ImGui::BeginDisabled(_studio->IsAsyncTasksRunning());
        if (ImGui::Button("Export")) {
          saveIndex = static_cast<int>(i);
        }
        ImGui::EndDisabled();
        tooltip(
            "Rebuild any stale stages above (like Generate), then write this modifier's input mesh "
            "to its configured export file.");
      }
      ImGui::EndDisabled(); // !enabled

      // Visualization controls stay active even when disabled, so a stale buffer can be inspected.
      MeshProcessingMethod const& method = modifier.ActiveMethod();
      if (method.ShowsInputVisualization()) {
        // Export Mochi Model: show the captured input surface then the reconstructed SDF, each
        // under its own heading -- mirroring the Model Viewer's Mochi Model / SDF Visualization
        // sections.
        ImGui::SeparatorText(method.InputVisualizationLabel());
        DrawStageVizControls("inputViz", modifier.inputView, [this, &modifier] {
          return StageRoleColor(modifier, kInputMeshColorRole);
        });
        ImGui::SeparatorText(method.OutputVisualizationLabel());
        DrawStageVizControls("stageViz", modifier.output, [this, &modifier] {
          return StageRoleColor(modifier, kOutputSdfColorRole);
        });
      } else {
        DrawStageVizControls(
            "stageViz", modifier.output, [this, &modifier] { return StageOutputColor(modifier); });
      }
    }
  };
  DrawReorderableBubbleList(drag, list);

  if (removeIndex >= 0) {
    RemoveModifier(static_cast<std::size_t>(removeIndex));
  }
  if (saveOutputIndex >= 0) {
    SaveModifierOutput(static_cast<std::size_t>(saveOutputIndex));
  }
  if (saveIndex >= 0) {
    GenerateAndExportModifier(static_cast<std::size_t>(saveIndex));
  }

  // Add a modifier. All groups are always available (a source can be added at any time and dragged
  // to the front); the groups are separated into sources, mesh modifiers, and exports.
  // HeightLarge raises the popup's max height (~20 rows) so all current entries are visible without
  // scrolling; the popup still shrinks to fit its content and grows a scrollbar past the cap.
  if (ImGui::BeginCombo(
          "##addmodifier",
          "+ Add Modifier",
          ImGuiComboFlags_NoArrowButton | ImGuiComboFlags_HeightLarge)) {
    auto const addFrom = [this](ModifierKind kind) {
      for (ModifierRegistryEntry const& entry : ProcessingModifierRegistry()) {
        if (entry.kind == kind && ImGui::Selectable(entry.name.c_str())) {
          AddModifier(entry.name);
        }
      }
    };
    // Thin light-grey separator lines group sources / mesh modifiers / exports without eating
    // selection space (the default separator color is near-black and invisible on the dark popup).
    ImGui::PushStyleColor(ImGuiCol_Separator, ImVec4(0.6f, 0.6f, 0.6f, 0.5f));
    addFrom(ModifierKind::Source);
    ImGui::Separator();
    addFrom(ModifierKind::Transform);
    ImGui::Separator();
    addFrom(ModifierKind::Export);
    ImGui::PopStyleColor();
    ImGui::EndCombo();
  }

  // Build (stale-aware, top-to-bottom) every modifier, then export any export modifiers that have a
  // configured path. Requires a source modifier so the chain has an input to build from.
  ImGui::Separator();
  bool const hasSource =
      std::any_of(_modifiers.begin(), _modifiers.end(), [](auto const& modifier) {
        return modifier->Kind() == ModifierKind::Source;
      });
  bool const canBuildAll = hasSource && !_studio->IsAsyncTasksRunning();
  ImGui::BeginDisabled(!canBuildAll);
  if (ImGui::Button("Build/Export All", ImVec2(-1.0f, 0.0f))) {
    BuildAndExportAll();
  }
  ImGui::EndDisabled();
  tooltip(
      "Generate every modifier top-to-bottom (skipping stages already up to date), then run each "
      "export modifier that has a configured output path.");
}

bool ModelEditor::CanGenerateModifier(std::size_t index) const {
  // Generate is available whenever the chain is buildable (a source is reachable) and that source
  // has its input file/slot -- NOT gated on intermediate upstream buffers being up to date, since
  // the cascade rebuilds those as needed.
  if (_studio->IsAsyncTasksRunning()) {
    return false;
  }
  std::vector<std::size_t> const chain = BuildGenerationChain(_modifiers, index);
  if (chain.empty()) {
    return false;
  }
  return _modifiers[chain.front()]->CanGenerate(MakeRunContext());
}

void ModelEditor::GenerateModifier(std::size_t index) {
  if (index >= _modifiers.size()) {
    return;
  }
  std::vector<std::size_t> const chain = BuildGenerationChain(_modifiers, index);
  if (chain.empty()) {
    MOCHI_LOG_WARNING(
        "Generate: '%s' has no source upstream to build from.", _modifiers[index]->DisplayName());
    return;
  }
  GenerateChainWithExports(chain, {});
}

void ModelEditor::GenerateAndExportModifier(std::size_t index) {
  if (index >= _modifiers.size()) {
    return;
  }
  std::vector<std::size_t> const chain = BuildGenerationChain(_modifiers, index);
  if (chain.empty()) {
    MOCHI_LOG_WARNING(
        "Export: '%s' has no source upstream to build from.", _modifiers[index]->DisplayName());
    return;
  }
  // Regenerate the chain (stale stages only), then export this modifier -- it is the chain's last
  // element.
  GenerateChainWithExports(chain, {chain.size() - 1});
}

void ModelEditor::BuildAndExportAll() {
  std::vector<std::size_t> const chain = BuildFullGenerationChain(_modifiers);
  if (chain.empty()) {
    MOCHI_LOG_WARNING("Build/Export All: no enabled source to build from.");
    return;
  }
  // Export every export modifier in the chain that has a configured output path.
  std::vector<std::size_t> exportChainPositions;
  for (std::size_t q = 0; q < chain.size(); ++q) {
    MeshProcessingModifier const& m = *_modifiers[chain[q]];
    if (m.ProvidesFileExport() && !m.ExportPath().empty()) {
      exportChainPositions.push_back(q);
    }
  }
  // Logged here rather than alongside the inline UI warning: this runs once per build instead of
  // every frame, so it records the overwrite in the log without flooding it.
  std::vector<bool> const collides = FindCollidingExportPaths(_modifiers);
  for (std::size_t const q : exportChainPositions) {
    if (collides[chain[q]]) {
      MOCHI_LOG_WARNING(
          "Build/Export All: '%s' shares its output path with another export; one will overwrite "
          "the other. Turn off Auto on one of them to give it its own path.",
          _modifiers[chain[q]]->DisplayName());
    }
  }
  GenerateChainWithExports(chain, exportChainPositions);
}

void ModelEditor::GenerateChainWithExports(
    std::vector<std::size_t> const& chain,
    std::vector<std::size_t> const& exportChainPositions) {
  // Persist the pipeline on every generate / build (also saved on close).
  SaveProcessingPipelineToDisk();

  ModifierRunContext const ctx = MakeRunContext();

  // Decide, on the main thread, which chain stages are stale. Staleness cascades downstream: once a
  // stage regenerates its output changes, so every later stage is stale too. A stage is stale if it
  // never generated, its own properties changed, or a mesh it consumes (its input, or an edge-swap
  // reference) changed since this stage last generated (globally-unique generation ids).
  //
  // A SOURCE is the exception on both counts: it builds from a file and ignores the stage above it,
  // so it neither inherits the cascade nor has an input generation id worth comparing. What can
  // change under it is the file's contents, and no generation id covers that -- nothing is handed
  // stage to stage across a file. So a source rebuilds whenever an export earlier in this same run
  // has already written; exports always run, which is what makes a source->export->source->export
  // stack rebuild end to end.
  //
  // Deliberately conservative: a later source is rebuilt whether or not the export above it wrote
  // the particular file that source reads. In the stack shape this exists for it did, and
  // re-reading a file is far cheaper than silently serving stale geometry. Note this covers only
  // rewrites made by this run -- a file changed outside the app between runs still needs a manual
  // Generate.
  std::size_t const n = chain.size();
  std::vector<bool> regen(n, false);
  bool exportedBefore = false; // an export earlier in this run has already written its file
  bool dirty = false;
  for (std::size_t p = 0; p < n; ++p) {
    MeshProcessingModifier const& m = *_modifiers[chain[p]];
    bool const isSource = m.Kind() == ModifierKind::Source;
    bool stale = m.outputGenId == 0 || m.PropsSignature(ctx) != m.lastGenSignature;
    if (isSource) {
      stale = stale || exportedBefore;
    } else {
      stale = stale || dirty ||
          (p > 0 && _modifiers[chain[p - 1]]->outputGenId != m.inputGenIdAtLastGen);
    }
    if (!stale && m.NeedsReferenceMesh()) {
      int const refIdx = ReferenceModifierIndex(_modifiers, chain[p]);
      int const refId = refIdx >= 0 ? _modifiers[static_cast<std::size_t>(refIdx)]->outputGenId : 0;
      stale = refId != m.referenceGenIdAtLastGen;
    }
    regen[p] = stale;
    // A source restarts the data flow, so downstream dirtiness begins again at it rather than
    // accumulating across the segment boundary.
    dirty = isSource ? stale : (dirty || stale);
    exportedBefore = exportedBefore || mochi::Contains(exportChainPositions, p);
  }

  if (!dirty && exportChainPositions.empty()) {
    // Nothing stale and nothing to export: confirm up the whole chain, then report the request was
    // fulfilled by the existing (non-stale) output rather than silently doing nothing.
    MOCHI_LOG(
        "Nothing to do: '%s' is already up to date; request fulfilled by existing output.",
        _modifiers[chain.back()]->DisplayName());
    return;
  }

  RunGenerationCascade(chain, regen, ctx, exportChainPositions);
}

void ModelEditor::RunGenerationCascade(
    std::vector<std::size_t> const& chain,
    std::vector<bool> const& regen,
    ModifierRunContext const& ctx,
    std::vector<std::size_t> const& exportChainPositions) {
  std::size_t const n = chain.size();

  // Shared with the serial tasks and onComplete. Stages that are not regenerated keep their full
  // current output; edge-flip references outside the chain use the displayed triangle surface.
  auto run = std::make_shared<StackRun>(
      _modifiers,
      chain,
      ctx,
      [this](std::size_t index) {
        StageBuffer const& output = _modifiers[index]->output;
        return StageMesh{
            .mesh = processing::BufferedStageOutput(output.tetMesh, output.sections),
            .embedded = output.embedded};
      },
      [this](std::size_t index) {
        MeshProcessingModifier const& modifier = *_modifiers[index];
        // An Embed stage displays the mesh it embedded, not the boundary of its output.
        return modifier.ActiveMethod().EmbeddedRole().has_value()
            ? processing::DisplaySurface(modifier.output.tetMesh)
            : SectionsToMeshData(modifier.output.sections);
      },
      processing::DisplaySurface);

  // One pass in chain order, each stage's export queued directly after the stage itself rather than
  // in a block at the end. Two things need that: a later source reads back the file an earlier
  // export writes, so the write has to have happened by the time it runs; and a failure part-way
  // down still leaves the exports above it written.
  // Each task depends on the tasks producing what it reads, so a stage whose input failed, timed
  // out, or was skipped is skipped instead of running on a missing or stale mesh.
  std::vector<AsyncTask> tasks;
  std::vector<int> generateTask(n, -1); // by chain position; -1 when the stage is not regenerated
  std::vector<std::pair<int, std::string>> exportTasks; // export task and the file it writes
  auto const inputTask = [&](std::size_t p) { return p > 0 ? generateTask[p - 1] : -1; };
  for (std::size_t p = 0; p < n; ++p) {
    MeshProcessingModifier const& mod = *_modifiers[chain[p]];
    char const* const name = mod.DisplayName();
    // regen[p] false: up to date, its current output is already in the run.
    if (regen[p]) {
      AsyncTask task{std::string("Generate ") + name, [run, p](AsyncCancelToken const& cancel) {
                       if (cancel.IsCancelRequested()) {
                         return false;
                       }
                       // A failed stage leaves no output; onComplete then keeps its prior buffer.
                       mochi::ErrorLog error;
                       return run->Generate(p, error);
                     }};
      if (mod.Kind() == ModifierKind::Source) {
        // A source reads a file, which an earlier export in this run may write.
        std::string const sourceFile = mod.SourceFilePath(ctx);
        for (auto const& [exportTask, exportFile] : exportTasks) {
          if (SameFilePath(sourceFile, exportFile)) {
            task.dependsOn.push_back(exportTask);
          }
        }
      } else if (inputTask(p) >= 0) {
        task.dependsOn.push_back(inputTask(p));
      }
      if (mod.NeedsReferenceMesh()) {
        int const refIdx = ReferenceModifierIndex(_modifiers, chain[p]);
        for (std::size_t q = 0; q < p; ++q) {
          if (static_cast<int>(chain[q]) == refIdx && generateTask[q] >= 0) {
            task.dependsOn.push_back(generateTask[q]);
          }
        }
      }
      generateTask[p] = static_cast<int>(tasks.size());
      tasks.push_back(std::move(task));
    }
    if (mochi::Contains(exportChainPositions, p)) {
      AsyncTask task{std::string("Export ") + name, [run, p](AsyncCancelToken const& cancel) {
                       if (cancel.IsCancelRequested()) {
                         return false;
                       }
                       mochi::ErrorLog error;
                       return run->Export(p, error);
                     }};
      int const prerequisite = generateTask[p] >= 0 ? generateTask[p] : inputTask(p);
      if (prerequisite >= 0) {
        task.dependsOn.push_back(prerequisite);
      }
      exportTasks.emplace_back(static_cast<int>(tasks.size()), mod.ExportPath());
      tasks.push_back(std::move(task));
    }
  }

  // Commit regenerated stages on the main thread, in chain order so each stage reads its input's
  // freshly-assigned generation id. This completion callback runs after RunGenerationCascade (and
  // its caller) have returned, so it must capture chain and regen BY VALUE -- they are references
  // into the caller's frame and would dangle otherwise.
  bool const hadExports = !exportChainPositions.empty();
  bool const started = _studio->BeginAsyncTasks(
      "Generating",
      std::move(tasks),
      [this, run, chain, regen, ctx, hadExports](bool allSucceeded) {
        std::size_t const m = chain.size();
        mochi::MeshData previousOutputSurface;
        std::size_t previousOutputSurfacePosition = m;
        for (std::size_t p = 0; p < m; ++p) {
          if (!regen[p]) {
            continue;
          }
          if (run->Output(p).GetNumElements() == 0) {
            continue; // failed/cancelled: keep the prior buffer + generation id
          }
          std::size_t const s = chain[p];
          MeshProcessingModifier& mod = *_modifiers[s];
          // A source built from a file, not from the stage above it (which a mid-chain source has),
          // so it has no input mesh to record or measure against.
          bool const isSource = mod.Kind() == ModifierKind::Source;
          // A tetrahedral output is displayed by its boundary while the full mesh is retained for
          // downstream processing, statistics, and Save.
          mochi::MeshData const& output = run->Output(p);
          mochi::MeshData outputSurface = processing::DisplaySurface(output);
          mochi::MeshData inputSurface;
          if (!isSource && p > 0 && run->Output(p - 1).GetNumElements() > 0) {
            inputSurface = previousOutputSurfacePosition == p - 1
                ? previousOutputSurface
                : processing::DisplaySurface(run->Output(p - 1));
          }
          // An Embed stage shows the mesh it embedded rather than the mesh it passes on unchanged.
          std::optional<EmbeddedMeshRole> const embeddedRole = mod.ActiveMethod().EmbeddedRole();
          std::optional<mochi::MeshData> const* const embeddedMesh =
              embeddedRole.has_value() ? &run->Embedded(p).Mesh(*embeddedRole) : nullptr;
          mochi::MeshData const* const shownEmbedded =
              embeddedMesh != nullptr && embeddedMesh->has_value() ? &**embeddedMesh : nullptr;
          mochi::MeshData const& shownSurface =
              shownEmbedded != nullptr ? *shownEmbedded : outputSurface;
          mod.output.sections = {MeshDataToSection(shownSurface)};
          mod.output.tetMesh = output.nodesPerElement == 4 ? output : mochi::MeshData{};
          mod.output.embedded = run->Embedded(p);
          mod.outputGenId = _nextGenerationId++;
          mod.inputGenIdAtLastGen =
              (!isSource && p > 0) ? _modifiers[chain[p - 1]]->outputGenId : 0;
          if (mod.NeedsReferenceMesh()) {
            int const refIdx = ReferenceModifierIndex(_modifiers, s);
            mod.referenceGenIdAtLastGen =
                refIdx >= 0 ? _modifiers[static_cast<std::size_t>(refIdx)]->outputGenId : 0;
          }
          mod.lastGenSignature = mod.PropsSignature(ctx);
          if (mod.ActiveMethod().ShowsInputVisualization()) {
            // Export Mochi Model shows its input surface next to the reconstructed SDF. Color both
            // by role and capture a COPY of the input mesh so the input preview survives later
            // reorders.
            mochi::Real3 const outColor = StageRoleColor(mod, kOutputSdfColorRole);
            RebuildStageMeshes(mod.output, outColor);
            mod.inputView.sections.clear();
            if (inputSurface.GetNumElements() > 0) {
              mod.inputView.sections = {MeshDataToSection(inputSurface)};
            }
            mochi::Real3 const inColor = StageRoleColor(mod, kInputMeshColorRole);
            RebuildStageMeshes(mod.inputView, inColor);
            mod.inputView.stats.tetMesh = p > 0 ? TetCounts(run->Output(p - 1)) : std::nullopt;
          } else {
            DestroyStageBufferMeshes(mod.inputView);
            mod.inputView.sections.clear();
            mod.inputView.tetMesh = {};
            mod.inputView.stats = {};
            mochi::Real3 const color = StageOutputColor(mod);
            RebuildStageMeshes(mod.output, color);
          }
          // Match the estimate to the format used by Save. Export modifiers can override it below.
          mod.output.stats.fileSizeBytes = output.nodesPerElement == 4
              ? processing::EstimateVtkSizeBytes(output)
              : processing::EstimateGlbSizeBytes(outputSurface);
          mod.output.stats.fileSizeLabel = "Est. File Size";
          mod.output.stats.tetMesh = shownEmbedded != nullptr ? std::nullopt : TetCounts(output);
          mod.output.stats.embeddedMesh = shownEmbedded != nullptr
              ? std::optional{MeasureEmbeddedMesh(*embeddedRole, *shownEmbedded)}
              : std::nullopt;
          mod.output.stats.sdfGrid = std::nullopt;
          // Let the modifier contribute output-specific stats (e.g. the baked SDF grid), then
          // recomposite the display string around the base summary RebuildStageMeshes set.
          mod.AnnotateStats(mod.output.stats);
          ComposeStats(mod.output.stats);
          // Mirror the estimated export size onto the input (Mochi Model) view too: the .mochi.h5
          // stores the input mesh plus the SDF, so its estimated size belongs on the mesh being
          // exported, not only the reconstructed-SDF preview.
          if (mod.ActiveMethod().ShowsInputVisualization()) {
            mod.inputView.stats.fileSizeBytes = mod.output.stats.fileSizeBytes;
            mod.inputView.stats.fileSizeLabel = mod.output.stats.fileSizeLabel;
            ComposeStats(mod.inputView.stats);
          }
          // Kick off the hidden input<->output Hausdorff for this stage (a source has no input);
          // it updates the stats and recomposites when it returns. For an Embed stage it measures
          // how far the embedded mesh strays from its input's surface.
          if (inputSurface.GetNumElements() > 0 && shownSurface.GetNumElements() > 0) {
            // The distance needs only the geometry. An embedded mesh's skinning refers to another
            // mesh's nodes, which the distance computation would otherwise validate it against.
            mochi::MeshData shownGeometry = shownSurface;
            shownGeometry.skinning.reset();
            KickoffModifierHausdorff(mod, std::move(inputSurface), std::move(shownGeometry));
          }
          // Cache the boundary for an adjacent regenerated stage's input.
          previousOutputSurface = std::move(outputSurface);
          previousOutputSurfacePosition = p;
        }
        if (!allSucceeded) {
          MOCHI_LOG_WARNING(
              "Generation cascade: one or more stages failed or were skipped (see log).");
        }
        if (hadExports) {
          // Newly written / overwritten export files only appear once the browser rescans.
          _studio->GetAssetBrowser().Refresh();
        }
      },
      /*serial=*/true);
  if (!started) {
    MOCHI_LOG_WARNING("Generate not started: another async task is already running.");
  }
}

void ModelEditor::KickoffModifierHausdorff(
    MeshProcessingModifier const& mod,
    mochi::MeshData input,
    mochi::MeshData output) {
  uint64_t const id = mod.id;
  int const genId = mod.outputGenId;
  auto inbox = _hausdorffInbox;
  _hausdorffQueue.Enqueue(
      id,
      [inbox, id, genId, input = std::move(input), output = std::move(output)](
          std::atomic<bool> const& cancelled) {
        mochi::MeshDataView const inputView(input);
        mochi::ErrorLog error;
        mochi::mesh::MeshStatistics const stats =
            mochi::mesh::ComputeMeshStatistics(output, &inputView, error);
        if (cancelled.load()) {
          return; // superseded / cleared while running: drop the result
        }
        HausdorffResult result{id, genId, error.IsOK() ? stats.hausdorffDistance : -1.0};
        std::lock_guard<std::mutex> const lock(inbox->mutex);
        inbox->results.push_back(result);
      });
}

void ModelEditor::PumpHausdorffResults() {
  std::vector<HausdorffResult> results;
  {
    std::lock_guard<std::mutex> const lock(_hausdorffInbox->mutex);
    results.swap(_hausdorffInbox->results);
  }
  for (HausdorffResult const& result : results) {
    if (result.distance < 0.0) {
      continue; // unavailable / failed: leave the stats without a Hausdorff line
    }
    for (auto const& modifier : _modifiers) {
      if (modifier->id == result.modifierId && modifier->outputGenId == result.genId) {
        modifier->output.stats.hausdorff = result.distance;
        ComposeStats(modifier->output.stats);
        break;
      }
    }
  }
}

ModifierRunContext ModelEditor::MakeRunContext() const {
  ModifierRunContext ctx;
  ctx.cadFilePath = _cadModelAsset ? _cadModelAsset->GetPath().ToString() : std::string();
  ctx.renderModelPath = _renderModelAsset ? _renderModelAsset->GetPath().ToString() : std::string();
  ctx.mochiModelPath = _mochiModelAsset ? _mochiModelAsset->GetPath().ToString() : std::string();
  ctx.cadScale = _cadScale;
  ctx.cadRotation = _cadRotation;
  ctx.cadTranslation = _cadTranslation;
  return ctx;
}

mochi::DynamicString ModelEditor::OriginModelPath() const {
  if (!_cadModelPath.empty()) {
    return _cadModelPath;
  }
  if (!_renderModelPath.empty()) {
    return _renderModelPath;
  }
  return _mochiModelPath;
}

std::string ModelEditor::OriginModelFolder() const {
  mochi::DynamicString const originPath = OriginModelPath();
  return originPath.empty() ? std::string() : mochi::Path(originPath).GetParentPath().ToString();
}

std::string ModelEditor::DefaultModifierSavePath(std::size_t index, char const* ext) const {
  // Named after the first populated slot (the opened model) in the asset's intermediates folder,
  // created on save if missing -- a modifier's saved output is generated content, so it goes there
  // under a flat layout too. A shadowed model is keyed apart there for the same reason its pipeline
  // is: so it does not write over the output of the model shadowing it.
  mochi::DynamicString const originPath = OriginModelPath();

  // Filename-safe token for the modifier type (strip non-alphanumerics from its display name). The
  // array index disambiguates multiple modifiers of the same type.
  std::string typeToken;
  for (char const c : std::string(_modifiers[index]->DisplayName())) {
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')) {
      typeToken.push_back(c);
    }
  }
  if (typeToken.empty()) {
    typeToken = "Modifier";
  }

  std::string const suffix = "_" + typeToken + "_" + std::to_string(index) + ext;
  if (originPath.empty()) {
    std::filesystem::path full = std::filesystem::current_path() / ("model" + suffix);
    full.make_preferred(); // native separators so the OS dialog honors the initial folder
    return full.string();
  }
  return AssetGeneratedFilePath(mochi::Path{originPath}, !_originIsShadowed, suffix);
}

std::string ModelEditor::ComputeProcessingJsonPath() const {
  // <intermediates>/<originBaseName>.StudioProcessing.json for the canonical model of a set; a
  // shadowed one keys on its own file name so it neither loads nor overwrites the canonical
  // model's pipeline (see AssetGeneratedFilePath). Created on save if missing.
  mochi::DynamicString const originPath = OriginModelPath();
  if (originPath.empty()) {
    return {};
  }
  return AssetGeneratedFilePath(
      mochi::Path{originPath}, !_originIsShadowed, kProcessingPipelineSuffix);
}

void ModelEditor::LoadProcessingPipelineOnOpen() {
  _processingJsonPath = ComputeProcessingJsonPath();
  if (_processingJsonPath.empty()) {
    return;
  }
  std::error_code ec;
  if (!std::filesystem::exists(_processingJsonPath.c_str(), ec)) {
    return; // no saved pipeline; start empty (the normal open path)
  }
  LoadedPipeline loaded;
  mochi::ErrorLog error;
  if (!LoadProcessingPipeline(_processingJsonPath, _processingJsonPath, loaded, error)) {
    _pipelineLoadFailed = true; // do not clobber a file we could not read on a later save
    MOCHI_LOG_WARNING(
        "Failed to load processing pipeline '%s'; starting empty.", _processingJsonPath.c_str());
    return;
  }
  for (auto& modifier : loaded.modifiers) {
    modifier->id = _nextModifierId++;
  }
  _modifiers = std::move(loaded.modifiers);
  if (loaded.hasEditorState) {
    _cadScale = loaded.editorState.cadScale;
    _cadRotation = loaded.editorState.cadRotation;
    _cadTranslation = loaded.editorState.cadTranslation;
  }
  // Restore the viewer-only reference models: rebuild the runtime list from the persisted state and
  // load each one's geometry. (STL/render/mochi render immediately; a CAD STEP reference restores
  // its slot but is tessellated on demand via its Generate button -- see UpdateReferenceModel.)
  for (ReferenceModelState const& s : loaded.referenceModels) {
    ReferenceModel rm;
    rm.uiId = _nextReferenceModelId++;
    rm.type = AssetTypeFromToken(s.type);
    if (rm.type == AssetType::Unknown) {
      rm.type = AssetType::RenderModel;
    }
    rm.path = s.path;
    rm.scale = s.scale;
    rm.rotation = s.rotation;
    rm.translation = s.translation;
    rm.buffer.viz.showSurface = s.showSurface;
    rm.buffer.viz.showWireframe = s.showWireframe;
    rm.buffer.viz.opacity = static_cast<float>(s.opacity);
    rm.color = s.color;
    rm.overrideColor = s.overrideColor;
    _referenceModels.push_back(std::move(rm));
  }
  for (ReferenceModel& rm : _referenceModels) {
    UpdateReferenceModel(rm);
  }
  // Baseline the change-detection snapshot to the just-loaded state, so opening and closing without
  // edits does not rewrite the file. Export paths are absent from the snapshot while they are on
  // Auto, so this baseline does not depend on the slots having resolved yet.
  _lastSavedPipelineJson = SerializeCurrentPipeline();
  MOCHI_LOG(
      "Loaded processing pipeline from '%s' (%d modifiers, %d reference models).",
      _processingJsonPath.c_str(),
      static_cast<int>(_modifiers.size()),
      static_cast<int>(_referenceModels.size()));
}

std::string ModelEditor::SerializeCurrentPipeline() const {
  ProcessingEditorState state;
  state.cadScale = _cadScale;
  state.cadRotation = _cadRotation;
  state.cadTranslation = _cadTranslation;

  std::vector<ReferenceModelState> refs;
  refs.reserve(_referenceModels.size());
  for (ReferenceModel const& rm : _referenceModels) {
    ReferenceModelState s;
    s.type = std::string(AssetTypeToToken(rm.type));
    s.path = rm.path;
    s.scale = rm.scale;
    s.rotation = rm.rotation;
    s.translation = rm.translation;
    s.showSurface = rm.buffer.viz.showSurface;
    s.showWireframe = rm.buffer.viz.showWireframe;
    s.opacity = rm.buffer.viz.opacity;
    s.color = rm.color;
    s.overrideColor = rm.overrideColor;
    refs.push_back(std::move(s));
  }
  return SerializeProcessingPipeline(_modifiers, state, refs, _processingJsonPath);
}

namespace {
// The serialized snapshot of a fresh Model Editor pipeline: no modifiers, no reference models, and
// the CAD transform at the ModelEditor _cad* member defaults (keep these in sync with those member
// initializers). When the live snapshot equals this, reopening the model with no JSON would
// reproduce the state exactly, so the JSON file is redundant.
std::string DefaultPipelineSnapshot() {
  ProcessingEditorState state;
  state.cadScale = {1.0f, 1.0f, 1.0f};
  state.cadRotation = {};
  state.cadTranslation = {0.0f, 0.0f, 0.0f};
  std::vector<std::unique_ptr<MeshProcessingModifier>> const noModifiers;
  std::vector<ReferenceModelState> const noReferenceModels;
  // No paths to relativize here (empty stack / no reference models), so the base file is
  // irrelevant.
  return SerializeProcessingPipeline(
      noModifiers, state, noReferenceModels, std::filesystem::path{});
}
} // namespace

void ModelEditor::SaveProcessingPipelineToDisk() {
  // No target, or the on-open load failed (don't clobber -- or delete -- a file we couldn't parse).
  if (_processingJsonPath.empty() || _pipelineLoadFailed) {
    return;
  }
  // Nothing changed since the last save / load / delete: repeated generates or an unedited
  // open+close never touch the file.
  std::string snapshot = SerializeCurrentPipeline();
  if (snapshot == _lastSavedPipelineJson) {
    return;
  }
  // The pipeline is back to the default state (no modifiers, no reference models, identity CAD
  // transform): reopening with no JSON would reproduce it, so the file is redundant. Delete any
  // existing file instead of persisting an empty pipeline.
  if (snapshot == DefaultPipelineSnapshot()) {
    std::error_code ec;
    if (std::filesystem::remove(_processingJsonPath, ec)) {
      MOCHI_LOG(
          "Removed redundant processing pipeline '%s' (state reset to default).",
          _processingJsonPath.c_str());
    } else if (ec) {
      MOCHI_LOG_WARNING(
          "Failed to delete redundant processing pipeline '%s'.", _processingJsonPath.c_str());
    }
    _lastSavedPipelineJson = std::move(snapshot);
    return;
  }
  mochi::ErrorLog error;
  if (!WriteProcessingPipelineFile(_processingJsonPath, snapshot, error)) {
    MOCHI_LOG_WARNING("Failed to save processing pipeline to '%s'.", _processingJsonPath.c_str());
    return;
  }
  _lastSavedPipelineJson = std::move(snapshot);
}

void ModelEditor::PopulateFromPreset(std::string const& path) {
  if (!_modifiers.empty()) {
    int const choice = tinyfd_messageBox(
        "Replace Pipeline",
        "Replace the current modifier stack with this preset?",
        "yesno",
        "question",
        /*aDefaultButton=*/0);
    if (choice != 1) {
      return; // user declined
    }
  }
  LoadedPipeline loaded;
  mochi::ErrorLog error;
  // Read the document from the preset, but resolve its paths against THIS model's pipeline file:
  // a preset's paths describe where things sit relative to the model it is applied to, so the
  // preset's own folder next to the executable must not enter into it.
  if (!LoadProcessingPipeline(path, _processingJsonPath, loaded, error)) {
    MOCHI_LOG_WARNING("Failed to load preset pipeline '%s'.", path.c_str());
    return;
  }
  // Destroy the current modifiers' scene objects before replacing the stack.
  auto* scene = _viewport->GetRenderScene();
  for (auto const& modifier : _modifiers) {
    if (modifier->output.surfaceMesh) {
      scene->DestroySceneObject(modifier->output.surfaceMesh);
    }
    if (modifier->output.wireframeMesh) {
      scene->DestroySceneObject(modifier->output.wireframeMesh);
    }
  }
  for (auto& modifier : loaded.modifiers) {
    modifier->id = _nextModifierId++;
  }
  _modifiers = std::move(loaded.modifiers);
  if (loaded.hasEditorState) {
    _cadScale = loaded.editorState.cadScale;
    _cadRotation = mochi::Quaternion(loaded.editorState.cadRotation);
    _cadTranslation = loaded.editorState.cadTranslation;
    ApplyCadModelTransform();
  }
  // The user explicitly chose a pipeline, so re-enable saving (a prior failed load blocked it).
  _pipelineLoadFailed = false;
  MOCHI_LOG(
      "Loaded processing pipeline from '%s' (%d modifiers).",
      path.c_str(),
      static_cast<int>(_modifiers.size()));
}

void ModelEditor::SaveModifierOutput(std::size_t index) {
  if (index >= _modifiers.size()) {
    return;
  }
  StageBuffer const& output = _modifiers[index]->output;
  bool const isTetMesh = output.tetMesh.GetNumElements() > 0;
  if (!isTetMesh && output.sections.empty()) {
    MOCHI_LOG_WARNING("Save: modifier has no output to save.");
    return;
  }

  // A tetrahedral output is saved as its tets (Legacy VTK, for third-party meshing tools), not as
  // the boundary surface it is displayed by.
  std::string const defaultPath = DefaultModifierSavePath(index, isTetMesh ? ".vtk" : ".glb");
  // The intermediates folder may not exist yet; create it up front, because the native dialog
  // silently falls back to its last-used directory when a default's parent is missing.
  superdex::robotics::EnsureDirectoriesCreated(
      defaultPath, mochi::ErrorLog{mochi::LogChannel::Warning});
  constexpr std::array<char const*, 2> surfaceFilters{"*.glb", "*.obj"};
  constexpr std::array<char const*, 1> tetFilters{"*.vtk"};
  std::span<char const* const> const filters =
      isTetMesh ? std::span<char const* const>{tetFilters} : surfaceFilters;
  mochi::Path const chosen = SuperDexStudio::GetFileDialogPath(
      "Save Modifier Output",
      filters.data(),
      static_cast<int>(filters.size()),
      isTetMesh ? "VTK mesh (*.vtk)" : "Mesh (*.glb, *.obj)",
      /*isSaveDialog=*/true,
      mochi::Path{defaultPath});
  if (chosen.IsEmpty()) {
    return;
  }

  std::string path = chosen.ToString();
  std::string const lower = mochi::Path{path}.AsLowercaseString();
  auto endsWith = [&lower](std::string_view ext) {
    return lower.size() >= ext.size() && lower.ends_with(ext);
  };
  bool const isObj = !isTetMesh && endsWith(".obj");
  if (isTetMesh && !endsWith(".vtk")) {
    path += ".vtk";
  } else if (!isTetMesh && !isObj && !endsWith(".glb") && !endsWith(".gltf")) {
    path += ".glb"; // default to GLB when no recognized suffix was chosen
  }

  std::shared_ptr<mochi::MeshData> tetMesh;
  std::shared_ptr<std::vector<MeshSection>> sections;
  if (isTetMesh) {
    tetMesh = std::make_shared<mochi::MeshData>(output.tetMesh);
  } else {
    sections = std::make_shared<std::vector<MeshSection>>(output.sections);
  }
  std::vector<AsyncTask> tasks;
  tasks.push_back(
      AsyncTask{
          "Save " + path, [path, sections, tetMesh, isTetMesh, isObj](AsyncCancelToken const&) {
            bool ok = false;
            if (isTetMesh) {
              mochi::ErrorLog error;
              processing::WriteVtkFile(path, *tetMesh, error);
              ok = error.IsOK();
            } else {
              ok = isObj ? WriteObjToFile(path.c_str(), *sections)
                         : WriteGlbToFile(path.c_str(), *sections);
            }
            if (!ok) {
              MOCHI_LOG_ERROR("Failed to write mesh file: %s", path.c_str());
            }
            return ok;
          }});
  bool const started =
      _studio->BeginAsyncTasks("Saving Mesh", std::move(tasks), [this](bool allSucceeded) {
        if (!allSucceeded) {
          MOCHI_LOG_WARNING("Mesh save failed (see log for details).");
          return;
        }
        // Surface the newly written file in the asset browser (it rescans on refresh).
        _studio->GetAssetBrowser().Refresh();
      });
  if (!started) {
    MOCHI_LOG_WARNING("Save not started: another async task is already running.");
  }
}

void ModelEditor::AddModifier(std::string_view name) {
  InsertModifier(name, _modifiers.size());
}

void ModelEditor::InsertModifier(std::string_view name, std::size_t at) {
  std::unique_ptr<MeshProcessingModifier> modifier = MakeProcessingModifier(name);
  if (!modifier) {
    MOCHI_LOG_WARNING(
        "Add modifier: unknown modifier type '%.*s'.", static_cast<int>(name.size()), name.data());
    return;
  }
  modifier->id = _nextModifierId++;
  at = std::min(at, _modifiers.size());
  // Elements at or after the insertion point shift up by one; every other index is unchanged. For
  // an append (at == size) this map is the identity, so the remap is a harmless no-op.
  std::size_t const n = _modifiers.size();
  std::vector<int> oldToNew(n);
  for (std::size_t k = 0; k < n; ++k) {
    oldToNew[k] = static_cast<int>(k < at ? k : k + 1);
  }
  _modifiers.insert(_modifiers.begin() + static_cast<long>(at), std::move(modifier));
  RemapModifierReferences(oldToNew);
}

void ModelEditor::RemapModifierReferences(std::vector<int> const& oldToNew) {
  for (auto const& modifier : _modifiers) {
    modifier->RemapReferences(oldToNew);
  }
}

void ModelEditor::AddDefaultStepPipeline() {
  // {modifier name, method name} (both must match the registry). The Alpha Wrap stage before the
  // isotropic remesh makes the pipeline more robust (a clean closed envelope to remesh).
  struct DefaultStage {
    char const* modifier;
    char const* method;
  };
  DefaultStage const kDefaultPipeline[] = {
      {"Source from CAD Model", "STEP->mesh"},
      {"Refine Mesh", "Make Manifold"},
      {"Refine Mesh", "Make Watertight"},
      {"Wrap Mesh", "Alpha Wrap"},
      {"Remesh", "Incremental Isotropic Remesh"},
      {"Refine Mesh", "Edge Flip Optimization"},
      {"Refine Mesh", "Edge Collapse (Simplification)"},
      {"Refine Mesh", "Tangential Relaxation (Smoothing)"},
      {"Export Mochi Model", "SDF"},
  };
  for (DefaultStage const& stage : kDefaultPipeline) {
    AddModifier(stage.modifier); // new modifiers default to collapsed
    if (!_modifiers.empty()) {
      _modifiers.back()->SelectMethodByName(stage.method);
    }
  }
}

void ModelEditor::RemoveModifier(std::size_t index) {
  if (index >= _modifiers.size()) {
    return;
  }
  // The output is going away; stop any in-flight Hausdorff for this modifier (others are
  // untouched).
  _hausdorffQueue.Cancel(_modifiers[index]->id);
  DestroyStageBufferMeshes(_modifiers[index]->output);
  DestroyStageBufferMeshes(_modifiers[index]->inputView);
  // The removed element maps to -1 (references to it fall back to the preceding source); elements
  // after it shift down by one.
  std::size_t const n = _modifiers.size();
  std::vector<int> oldToNew(n);
  for (std::size_t k = 0; k < n; ++k) {
    oldToNew[k] = (k == index) ? -1 : static_cast<int>(k > index ? k - 1 : k);
  }
  _modifiers.erase(_modifiers.begin() + static_cast<long>(index));
  RemapModifierReferences(oldToNew);
}

void ModelEditor::MoveModifier(std::size_t from, std::size_t to) {
  if (from >= _modifiers.size()) {
    return;
  }
  // Sources may move anywhere (including to index 0); other modifiers cannot become the first
  // element (index 0 should hold a source so the chain has an input).
  bool const isSource = _modifiers[from]->Kind() == ModifierKind::Source;
  std::size_t const minTo = isSource ? 0 : 1;
  to = std::clamp<std::size_t>(to, minTo, _modifiers.size() - 1);
  if (from == to) {
    return;
  }
  // Build the old->new index map for this move so reference-holding modifiers follow their targets.
  // The moved element goes to `to`; elements between from and to shift by one toward `from`.
  std::size_t const n = _modifiers.size();
  std::vector<int> oldToNew(n);
  for (std::size_t k = 0; k < n; ++k) {
    if (k == from) {
      oldToNew[k] = static_cast<int>(to);
    } else if (from < to && k > from && k <= to) {
      oldToNew[k] = static_cast<int>(k - 1);
    } else if (from > to && k >= to && k < from) {
      oldToNew[k] = static_cast<int>(k + 1);
    } else {
      oldToNew[k] = static_cast<int>(k);
    }
  }
  MoveElement(_modifiers, from, to);
  RemapModifierReferences(oldToNew);
}

void ModelEditor::ShowModelViewerWindow(bool* open) {
  ImGui::Begin("Model Viewer", open);
  auto const& assetManager = _studio->GetAssetManager();

  // Shared by each section's header icons and its Surface / Wireframe checkboxes.
  auto const showCadSurface = [this] {
    if (_cadModelSurfaceMesh) {
      _cadModelSurfaceMesh->SetVisible(_showCadModelSurface);
    }
  };
  auto const showCadWireframe = [this] {
    if (_cadModelWireframeMesh) {
      _cadModelWireframeMesh->SetVisible(_showCadModelWireframe);
    }
  };
  auto const showRenderSurface = [this] {
    if (_renderModelSurfaceMesh) {
      _renderModelSurfaceMesh->SetVisible(_showRenderModelSurface);
    }
  };
  auto const showRenderWireframe = [this] {
    if (_renderModelWireframeMesh) {
      _renderModelWireframeMesh->SetVisible(_showRenderModelWireframe);
    }
  };
  auto const showMochiSurface = [this] {
    if (_mochiModelSurfaceMesh) {
      _mochiModelSurfaceMesh->SetVisible(_showMochiModelSurface);
    }
  };
  auto const showMochiWireframe = [this] {
    if (_mochiModelWireframeMesh) {
      _mochiModelWireframeMesh->SetVisible(_showMochiModelWireframe);
    }
  };
  auto const showSdfSurface = [this] {
    if (_sdfSurfaceMesh) {
      _sdfSurfaceMesh->SetVisible(_showSdfSurface);
    }
  };
  auto const showSdfWireframe = [this] {
    if (_sdfWireframeMesh) {
      _sdfWireframeMesh->SetVisible(_showSdfWireframe);
    }
  };

  ImGuiTreeNodeFlags flags = _cadModelPath.empty() ? 0 : ImGuiTreeNodeFlags_DefaultOpen;
  bool const cadOpen =
      CollapsingHeaderWithToggles("CAD Model", "###CadModelHeader", flags, false, 1, [&] {
        ImGui::BeginDisabled(_cadModelPath.empty());
        DrawVizToggles(
            "cadHeaderViz",
            "CAD model",
            _showCadModelSurface,
            _showCadModelWireframe,
            showCadSurface,
            showCadWireframe);
        ImGui::EndDisabled();
      });
  if (cadOpen) {
    ImGui::PushID("CadModel");
    // Fixed slot: bound to the sibling-folder asset set, so drag-drop and clear are disabled (the
    // slot can still be viewed and browsed to). Reassigning models is done via the sibling folders
    // / the Additional Reference Models section, not by editing this slot.
    ImGui::AssetSlot(
        "##CadModel",
        _cadModelPath,
        assetManager,
        _studio,
        AssetType::CadModel,
        /*acceptDragDropPayload=*/false,
        /*showClearButton=*/false);
    if (_cadModelAsset) {
      // Swaps between the file-colored and flat surfaces as needed; only a flat surface is
      // recolored in place.
      auto applyCadMaterial = [this]() {
        if ((_cadColorModel != nullptr) != ShowsCadFileColors()) {
          RebuildCadModelMeshes();
          return;
        }
        if (!_cadModelSurfaceMesh || _cadColorModel != nullptr) {
          return;
        }
        auto& rm = _studio->GetResourceManager();
        _cadModelSurfaceMesh->SetMaterial(
            _cadModelOpacity >= 1.0f
                ? rm.CreateFlatLitOpaqueMaterial(ToFloat3(_cadColor))
                : rm.CreateFlatLitSeeThroughMaterial(ToFloat3(_cadColor), _cadModelOpacity));
      };

      // Color
      float cadCol[3] = {
          static_cast<float>(_cadColor[0]),
          static_cast<float>(_cadColor[1]),
          static_cast<float>(_cadColor[2])};
      if (ImGui::ColorEdit3("Color", cadCol, ImGuiColorEditFlags_NoInputs)) {
        _cadColor = {cadCol[0], cadCol[1], cadCol[2]};
        applyCadMaterial();
        RecolorWireframe(_cadModelWireframeMesh, _cadColor);
      }
      if (_cadHasFileColors) {
        ImGui::SameLine();
        if (ImGui::Checkbox("Override with color", &_cadOverrideColor)) {
          applyCadMaterial();
        }
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
          ImGui::SetTooltip(
              "Show the CAD model in this color instead of the STEP file's own face colors.");
        }
      }

      // Transform applied to the imported CAD geometry. Editing it re-applies the transform to the
      // cached untransformed tessellation and rebuilds the displayed mesh immediately (no
      // re-tessellation); the result also feeds the processing stages below.
      ImGui::SeparatorText("Transform");
      auto cadTooltip = [](char const* text) {
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
          ImGui::BeginTooltip();
          ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
          ImGui::TextUnformatted(text);
          ImGui::PopTextWrapPos();
          ImGui::EndTooltip();
        }
      };
      bool cadTransformChanged = false;
      cadTransformChanged |= ImGui::DragRealXYZ("Scale", _cadScale, 0.01f, 0, 0, "%.4f", 0, 1.0f);
      cadTooltip("Scale applied to the imported CAD geometry, before rotation and translation.");
      cadTransformChanged |= ImGui::DragTransformRT("Transform", _cadRotation, _cadTranslation);
      cadTooltip(
          "Rigid transform applied to the imported CAD geometry. Affects the displayed CAD mesh and "
          "every processed mesh.");
      if (cadTransformChanged) {
        ApplyCadModelTransform();
      }

      // MeshSection 0: the STEP file's own tessellation, shown independently of the processing
      // stages below.
      ImGui::SeparatorText("STEP->Mesh");
      DrawMeshVizControls(
          "cadViz",
          _showCadModelSurface,
          _showCadModelWireframe,
          _showCadModelStats,
          _cadModelOpacity,
          _cadStats.display,
          _studio->GetFont("Roboto Mono"),
          showCadSurface,
          showCadWireframe,
          [&applyCadMaterial] { applyCadMaterial(); });
    }
    if (_cadModelAsset) {
      ImGui::Separator();
      // Re-reads the STEP file and re-tessellates into the CAD mesh-section backing. Disabled while
      // any async stage is running (only one stage may run at a time).
      ImGui::BeginDisabled(_studio->IsAsyncTasksRunning());
      if (ImGui::Button("Generate")) {
        RegenerateCadModel();
      }
      ImGui::EndDisabled();
    }
    ImGui::PopID();
  }

  flags = _renderModelPath.empty() ? 0 : ImGuiTreeNodeFlags_DefaultOpen;
  bool const renderOpen =
      CollapsingHeaderWithToggles("Render Model", "###RenderModelHeader", flags, false, 1, [&] {
        ImGui::BeginDisabled(_renderModelPath.empty());
        DrawVizToggles(
            "renderHeaderViz",
            "render model",
            _showRenderModelSurface,
            _showRenderModelWireframe,
            showRenderSurface,
            showRenderWireframe);
        ImGui::EndDisabled();
      });
  if (renderOpen) {
    ImGui::PushID("RenderModel");
    // Fixed slot (see the CAD Model note): drag-drop and clear disabled; view + browse only.
    ImGui::AssetSlot(
        "##RenderModel",
        _renderModelPath,
        assetManager,
        _studio,
        AssetType::RenderModel,
        /*acceptDragDropPayload=*/false,
        /*showClearButton=*/false);
    if (_renderModelAsset) {
      // Color override: the render mesh keeps its real PBR materials/textures unless this is
      // checked.
      float renderCol[3] = {
          static_cast<float>(_renderColor[0]),
          static_cast<float>(_renderColor[1]),
          static_cast<float>(_renderColor[2])};
      if (ImGui::ColorEdit3("Color", renderCol, ImGuiColorEditFlags_NoInputs)) {
        _renderColor = {renderCol[0], renderCol[1], renderCol[2]};
        RebuildRenderModelSurface();
        RecolorWireframe(_renderModelWireframeMesh, _renderColor);
      }
      ImGui::SameLine();
      if (ImGui::Checkbox("Override with color", &_renderOverrideColor)) {
        RebuildRenderModelSurface();
      }

      ImGui::SeparatorText("Render Model Visualization");
      // Below full opacity the glTF surface is shown as a translucent flat material; at 1.0 it
      // keeps its PBR look unless the color override is on (see RebuildRenderModelSurface).
      DrawMeshVizControls(
          "renderViz",
          _showRenderModelSurface,
          _showRenderModelWireframe,
          _showRenderModelStats,
          _renderModelOpacity,
          _renderStats.display,
          _studio->GetFont("Roboto Mono"),
          showRenderSurface,
          showRenderWireframe,
          [this] { RebuildRenderModelSurface(); });
    }
    ImGui::PopID();
  }

  flags = _mochiModelPath.empty() ? 0 : ImGuiTreeNodeFlags_DefaultOpen;
  // A collision model holds a triangle mesh and an SDF, like Export Mochi Model, then optionally a
  // contact skin and a visual mesh.
  mochi::ModelData const* const mochiModelData =
      _mochiModelAsset ? &_mochiModelAsset->GetModelData() : nullptr;
  bool const mochiHasSdf = mochiModelData && mochiModelData->sdf.has_value();
  bool const mochiHasContactSkin = mochiModelData && mochiModelData->contactSkinMesh.has_value();
  bool const mochiHasVisualMesh = mochiModelData && mochiModelData->visualMesh.has_value();
  bool const mochiOpen = CollapsingHeaderWithToggles(
      "Collision Model", "###CollisionModelHeader", flags, false, 4, [&] {
        ImGui::BeginDisabled(_mochiModelPath.empty());
        DrawVizToggles(
            "mochiHeaderViz",
            "Mochi triangle mesh",
            _showMochiModelSurface,
            _showMochiModelWireframe,
            showMochiSurface,
            showMochiWireframe);
        ImGui::EndDisabled();
        SameLineAfterTogglePair();
        ImGui::BeginDisabled(!mochiHasSdf);
        DrawVizToggles(
            "sdfHeaderViz",
            "SDF",
            _showSdfSurface,
            _showSdfWireframe,
            showSdfSurface,
            showSdfWireframe);
        ImGui::EndDisabled();
        SameLineAfterTogglePair();
        ImGui::BeginDisabled(!mochiHasContactSkin);
        DrawVizToggles("contactSkinHeaderViz", "contact skin", _contactSkinBuffer);
        ImGui::EndDisabled();
        SameLineAfterTogglePair();
        ImGui::BeginDisabled(!mochiHasVisualMesh);
        DrawVizToggles("visualMeshHeaderViz", "visual mesh", _visualMeshBuffer);
        ImGui::EndDisabled();
      });
  if (mochiOpen) {
    ImGui::PushID("MochiModel");
    // Fixed slot (see the CAD Model note): drag-drop and clear disabled; view + browse only.
    ImGui::AssetSlot(
        "##MochiModel",
        _mochiModelPath,
        assetManager,
        _studio,
        AssetType::MochiModel,
        /*acceptDragDropPayload=*/false,
        /*showClearButton=*/false);
    if (_mochiModelAsset) {
      auto applyMochiMaterial = [this]() {
        if (!_mochiModelSurfaceMesh) {
          return;
        }
        auto& rm = _studio->GetResourceManager();
        _mochiModelSurfaceMesh->SetMaterial(
            _mochiModelOpacity >= 1.0f
                ? rm.CreateFlatLitOpaqueMaterial(ToFloat3(_mochiColor))
                : rm.CreateFlatLitSeeThroughMaterial(ToFloat3(_mochiColor), _mochiModelOpacity));
      };

      // Color
      float mochiCol[3] = {
          static_cast<float>(_mochiColor[0]),
          static_cast<float>(_mochiColor[1]),
          static_cast<float>(_mochiColor[2])};
      if (ImGui::ColorEdit3("Color", mochiCol, ImGuiColorEditFlags_NoInputs)) {
        _mochiColor = {mochiCol[0], mochiCol[1], mochiCol[2]};
        applyMochiMaterial();
        RecolorWireframe(_mochiModelWireframeMesh, _mochiColor);
      }

      ImGui::SeparatorText("Mochi Model Visualization");
      DrawMeshVizControls(
          "mochiViz",
          _showMochiModelSurface,
          _showMochiModelWireframe,
          _showMochiModelStats,
          _mochiModelOpacity,
          _mochiStats.display,
          _studio->GetFont("Roboto Mono"),
          showMochiSurface,
          showMochiWireframe,
          [&applyMochiMaterial] { applyMochiMaterial(); });

      // Mochi models carry their own surface mesh + SDF; this section only visualizes them. To
      // alter the model or its SDF, apply modifiers to produce the mesh, build the SDF, and export
      // back to the file. The SDF viz mesh is reconstructed on open (in UpdateMochiModel) when
      // present.
      ImGui::SeparatorText("SDF Visualization");
      ImGui::BeginDisabled(!mochiHasSdf);
      DrawMeshVizControls(
          "sdfViz",
          _showSdfSurface,
          _showSdfWireframe,
          _showSdfStats,
          _sdfOpacity,
          _sdfStats.display,
          _studio->GetFont("Roboto Mono"),
          showSdfSurface,
          showSdfWireframe,
          [this] {
            if (!_sdfSurfaceMesh) {
              return;
            }
            auto& rm = _studio->GetResourceManager();
            filament::math::float3 const color = ToFloat3(_sdfColor);
            _sdfSurfaceMesh->SetMaterial(
                _sdfOpacity >= 1.0f ? rm.CreateFlatLitOpaqueMaterial(color)
                                    : rm.CreateFlatLitSeeThroughMaterial(color, _sdfOpacity));
          });
      ImGui::EndDisabled();

      ImGui::SeparatorText("Contact Skin Visualization");
      ImGui::BeginDisabled(!mochiHasContactSkin);
      DrawStageVizControls(
          "contactSkinViz", _contactSkinBuffer, [this] { return _contactSkinColor; });
      ImGui::EndDisabled();

      ImGui::SeparatorText("Visual Mesh Visualization");
      ImGui::BeginDisabled(!mochiHasVisualMesh);
      DrawStageVizControls("visualMeshViz", _visualMeshBuffer, [this] { return _visualMeshColor; });
      ImGui::EndDisabled();
    }
    ImGui::PopID();
  }

  ShowReferenceModelsSection();

  ImGui::End();
}

void ModelEditor::ShowReferenceModelsSection() {
  auto const& assetManager = _studio->GetAssetManager();
  if (!ImGui::CollapsingHeader("Additional Reference Models", ImGuiTreeNodeFlags_DefaultOpen)) {
    return;
  }
  ImGui::PushID("ReferenceModels");
  ImGui::TextDisabled("Viewer-only context. Not used by the processing modifiers.");
  // Top button inserts at the front of the array; a matching button below appends to the back.
  if (ImGui::Button("Add Reference Model##top")) {
    AddReferenceModel(/*addToTop=*/true);
  }

  int removeIndex = -1;
  DragReorderState& drag = _referenceDrag;
  ReorderableBubbleList list;
  list.count = _referenceModels.size();
  list.id = [&](std::size_t i) { return _referenceModels[i].uiId; };
  // Neutral grey, matching the modifier stack's transform bubbles (darker for the drag
  // placeholder).
  list.bubbleColor = [](std::size_t /*i*/, bool isPlaceholder) {
    return ImGui::GetColorU32(
        isPlaceholder ? ImVec4(0.13f, 0.13f, 0.15f, 1.0f) : ImVec4(0.24f, 0.24f, 0.28f, 1.0f));
  };
  list.ghostLabel = [&](std::size_t i) {
    ReferenceModel const& rm = _referenceModels[i];
    return rm.path.empty() ? std::string("Reference Model")
                           : std::filesystem::path(rm.path.c_str()).filename().string();
  };
  // Reference models don't reference each other, so a reorder is a plain rotate with no upkeep.
  list.move = [&](std::size_t from, std::size_t to) {
    MoveElement(_referenceModels, from, to);
    SaveProcessingPipelineToDisk();
  };
  list.drawContent = [&](std::size_t i) {
    ReferenceModel& rm = _referenceModels[i];
    StageVisualization& viz = rm.buffer.viz;
    // Shared by the header icons and the Surface / Wireframe checkboxes. Visibility toggles are
    // discrete edits, so they persist immediately.
    auto const showSurface = [this, &rm] {
      ApplySurfaceVisibility(rm.buffer);
      SaveProcessingPipelineToDisk();
    };
    auto const showWireframe = [this, &rm] {
      ApplyWireframeVisibility(rm.buffer);
      SaveProcessingPipelineToDisk();
    };

    std::array<char const*, 3> const kTypeItems{"CAD Model", "Render Model", "Collision Model"};
    std::array<AssetType, 3> const kTypeValues{
        AssetType::CadModel, AssetType::RenderModel, AssetType::MochiModel};
    int typeIndex = 1; // default to Render Model (index into kTypeItems above)
    for (int t = 0; t < static_cast<int>(kTypeValues.size()); ++t) {
      if (kTypeValues[static_cast<std::size_t>(t)] == rm.type) {
        typeIndex = t;
      }
    }

    // --- header row (after the drag grip): [collapsible name  [surface][wireframe]] ... [trash]
    ImGui::SameLine();
    // Reserve room on the right for the remove button so the framed header stops before it (see
    // the modifier rows). Natural width matches the AssetSlot clear-trashcan.
    float const removeWidth =
        ImGui::CalcTextSize(ICON_FA_TRASH).x + ImGui::GetStyle().FramePadding.x * 2.0f;
    ImGuiWindow* const window = ImGui::GetCurrentWindow();
    float const savedWorkRectMaxX = window->WorkRect.Max.x;
    window->WorkRect.Max.x -= removeWidth + ImGui::GetStyle().ItemSpacing.x;
    std::string const header =
        list.ghostLabel(i) + " [" + kTypeItems[static_cast<std::size_t>(typeIndex)] + "]";
    bool const bodyOpen = CollapsingHeaderWithToggles(
        header, "###refHeader", ImGuiTreeNodeFlags_DefaultOpen, false, 1, [&] {
          ImGui::BeginDisabled(rm.path.empty());
          DrawVizToggles(
              "refHeaderViz",
              "reference model",
              viz.showSurface,
              viz.showWireframe,
              showSurface,
              showWireframe);
          ImGui::EndDisabled();
        });
    window->WorkRect.Max.x = savedWorkRectMaxX;
    ImGui::SameLine();
    ImGui::SetCursorPosX(ImGui::GetContentRegionMax().x - removeWidth);
    // Distinct ID (##suffix) from the AssetSlot's own clear-trashcan below, which shares the same
    // icon glyph and would otherwise collide (breaking the slot's clear button). As tall as the
    // header bar.
    PushHeaderRowFramePadding();
    bool const removeClicked = ImGui::Button(ICON_FA_TRASH "##removeReferenceModel");
    ImGui::PopStyleVar();
    if (removeClicked) {
      removeIndex = static_cast<int>(i);
    }
    if (ImGui::IsItemHovered()) {
      ImGui::SetTooltip("Remove this reference model.");
    }
    if (!bodyOpen) {
      return;
    }

    // Changing the type reinterprets the slotted path and reloads; an incompatible path is left in
    // place but will simply fail to load until repointed.
    ImGui::SetNextItemWidth(160.0f);
    if (ImGui::Combo("Type", &typeIndex, kTypeItems.data(), static_cast<int>(kTypeItems.size()))) {
      rm.type = kTypeValues[static_cast<std::size_t>(typeIndex)];
      UpdateReferenceModel(rm, /*assignDefaultColor=*/true);
      _studio->GetAssetManager().RegisterReferencer(this);
      SaveProcessingPipelineToDisk();
    }

    // --- model socket + path + browse + clear ---
    if (ImGui::AssetSlot("##refslot", rm.path, assetManager, _studio, rm.type, true)) {
      UpdateReferenceModel(rm, /*assignDefaultColor=*/true);
      _studio->GetAssetManager().RegisterReferencer(this);
      SaveProcessingPipelineToDisk();
    }

    if (!rm.path.empty()) {
      // --- color + optional override ---
      // A render reference keeps its real materials/textures unless "Override with color" is
      // checked; CAD/mochi references are always shown in this color.
      float col[3] = {
          static_cast<float>(rm.color[0]),
          static_cast<float>(rm.color[1]),
          static_cast<float>(rm.color[2])};
      if (ImGui::ColorEdit3("Color", col, ImGuiColorEditFlags_NoInputs)) {
        rm.color = {col[0], col[1], col[2]};
        RebuildReferenceMeshes(rm);
        _referenceModelsDirty = true;
      }
      if (rm.type == AssetType::RenderModel) {
        ImGui::SameLine();
        if (ImGui::Checkbox("Override with color", &rm.overrideColor)) {
          RebuildReferenceMeshes(rm);
          SaveProcessingPipelineToDisk();
        }
      }

      // --- transform ---
      // Transform is a live drag: apply immediately, mark dirty, and flush once the drag ends (see
      // the end-of-section flush below).
      ImGui::SeparatorText("Transform");
      bool transformChanged = false;
      transformChanged |= ImGui::DragRealXYZ("Scale", rm.scale, 0.01f, 0, 0, "%.4f", 0, 1.0f);
      transformChanged |= ImGui::DragTransformRT("Transform", rm.rotation, rm.translation);
      if (transformChanged) {
        ApplyReferenceModelTransform(rm);
        _referenceModelsDirty = true;
      }

      // --- show / hide: surface, wireframe, stats, opacity ---
      // Visibility toggles are discrete edits: persist immediately. Opacity is a live drag: apply
      // immediately, flush once the drag ends (see _referenceModelsDirty below).
      DrawMeshVizControls(
          "refViz",
          viz.showSurface,
          viz.showWireframe,
          viz.showStats,
          viz.opacity,
          rm.buffer.stats.display,
          _studio->GetFont("Roboto Mono"),
          showSurface,
          showWireframe,
          [this, &rm] {
            RebuildReferenceMeshes(rm);
            _referenceModelsDirty = true;
          });

      // A CAD STEP reference tessellates automatically (on drop and on disk change) via
      // PumpReferenceCadTessellations -- no manual Generate button. Show a hint while it is
      // pending.
      if (rm.needsCadTessellation) {
        ImGui::TextDisabled("Tessellating CAD reference...");
      }
    }
  };
  DrawReorderableBubbleList(drag, list);

  // Matching add button at the bottom (appends to the back), shown once the array is non-empty so
  // elements can be added at either end without scrolling.
  if (!_referenceModels.empty()) {
    if (ImGui::Button("Add Reference Model##bottom")) {
      AddReferenceModel(/*addToTop=*/false);
    }
  }

  if (removeIndex >= 0) {
    RemoveReferenceModel(static_cast<std::size_t>(removeIndex));
  }
  // Flush a finished drag (transform / opacity) once nothing is being actively edited, so the
  // change persists without writing the JSON every frame.
  if (_referenceModelsDirty && !ImGui::IsAnyItemActive()) {
    _referenceModelsDirty = false;
    SaveProcessingPipelineToDisk();
  }
  ImGui::PopID();
}

void ModelEditor::AddReferenceModel(bool addToTop) {
  ReferenceModel rm;
  rm.uiId = _nextReferenceModelId++;
  rm.buffer.viz.showSurface = true; // a freshly added reference starts visible once an asset is set
  // The color defaults to the path-hash color once an asset is assigned (see UpdateReferenceModel);
  // the empty slot keeps the struct default until then.
  if (addToTop) {
    _referenceModels.insert(_referenceModels.begin(), std::move(rm));
  } else {
    _referenceModels.push_back(std::move(rm));
  }
  SaveProcessingPipelineToDisk();
}

void ModelEditor::RemoveReferenceModel(std::size_t index) {
  if (index >= _referenceModels.size()) {
    return;
  }
  DestroyReferenceModelMeshes(_referenceModels[index]);
  _referenceModels.erase(_referenceModels.begin() + static_cast<std::ptrdiff_t>(index));
  _studio->GetAssetManager().RegisterReferencer(this);
  SaveProcessingPipelineToDisk();
}

void ModelEditor::DestroyReferenceModelMeshes(ReferenceModel& rm) {
  auto* scene = _viewport->GetRenderScene();
  if (rm.buffer.surfaceMesh) {
    scene->DestroySceneObject(rm.buffer.surfaceMesh);
    rm.buffer.surfaceMesh = nullptr;
  }
  if (rm.buffer.wireframeMesh) {
    scene->DestroySceneObject(rm.buffer.wireframeMesh);
    rm.buffer.wireframeMesh = nullptr;
  }
}

void ModelEditor::UpdateReferenceModel(ReferenceModel& rm, bool assignDefaultColor) {
  DestroyReferenceModelMeshes(rm);
  rm.sectionsOriginal.clear();
  rm.buffer.sections.clear();
  rm.buffer.stats = {};
  rm.needsCadTessellation = false;
  rm.mtime = FileMtimeOrDefault(rm.path);
  if (rm.path.empty()) {
    return;
  }
  mochi::Path const path = rm.path;
  // Load the asset so the slot renders a thumbnail (and so a mochi reference resolves via the asset
  // manager, matching the main Mochi slot's code path).
  auto& assetManager = _studio->GetAssetManager();
  assetManager.LoadAsset(path);

  // Default the reference color to the path-hash color (same scheme as the fixed slots / mochi
  // models), but only on a fresh assignment so a persisted / user-picked color survives reload.
  if (assignDefaultColor) {
    ImVec4 const c = HashStringToColor(path.ToString());
    rm.color = {c.x, c.y, c.z};
  }

  switch (rm.type) {
    case AssetType::RenderModel:
      rm.sectionsOriginal = ReadRenderModelSections(path);
      break;
    case AssetType::MochiModel: {
      if (auto* mochiAsset = assetManager.FindAssetByPath<MochiModelAsset>(path)) {
        auto const& converter = _studio->GetEditorToRendererSpaceConverter();
        std::vector<float> positions;
        std::vector<float> normals;
        std::vector<int> indices;
        if (BuildMochiModelGeometry(
                mochiAsset->GetModelData(), &converter, positions, normals, indices)) {
          MeshSection section;
          section.positions = std::move(positions);
          section.normals = std::move(normals);
          section.indices = std::move(indices);
          section.hasNormals = !section.normals.empty();
          rm.sectionsOriginal.push_back(std::move(section));
        }
      }
      break;
    }
    case AssetType::CadModel:
      if (path.AsLowercaseString().ends_with(".stl")) {
        rm.sectionsOriginal = ReadRenderModelSections(path);
        // Give the STL CAD reference a thumbnail too (parity with the STEP path below).
        if (auto* cadAsset = assetManager.FindAssetByPath<CadModelAsset>(path)) {
          cadAsset->SetTessellation(rm.sectionsOriginal, _studio->GetResourceManager());
          cadAsset->MarkThumbnailDirty();
        }
      } else {
        // STEP: tessellate asynchronously. The per-frame pump (PumpReferenceCadTessellations)
        // starts it as soon as the async runner is idle, so drops / disk changes always regenerate.
        rm.needsCadTessellation = true;
        return; // geometry (and its rebuild) is produced by the async completion
      }
      break;
    case AssetType::MochiPrefab:
    case AssetType::Bot:
    case AssetType::BotScene:
    case AssetType::BotTask:
    case AssetType::Tuner:
    case AssetType::Unknown:
    case AssetType::Count:
      break; // not a viewer-only reference model type; leave the slot empty
  }
  ApplyReferenceModelTransform(rm);
}

void ModelEditor::PumpReferenceCadTessellations() {
  if (_studio->IsAsyncTasksRunning()) {
    return; // one batch at a time; try again next frame
  }
  for (ReferenceModel& rm : _referenceModels) {
    if (rm.needsCadTessellation && !rm.path.empty()) {
      rm.needsCadTessellation = false;
      RegenerateReferenceCadModel(rm);
      return; // start one per frame; the rest follow as the runner frees up
    }
  }
}

void ModelEditor::ApplyReferenceModelTransform(ReferenceModel& rm) {
  rm.buffer.sections = rm.sectionsOriginal;
  ApplySectionsTransform(rm.buffer.sections, rm.scale, rm.rotation, rm.translation);
  // Recompute here (the geometry funnel) rather than in RebuildReferenceMeshes, which also runs on
  // every opacity-drag frame. A mochi reference surfaces its baked SDF's grid dimensions and its
  // tet counts.
  rm.buffer.stats.base = FormatMeshStats(rm.buffer.sections);
  rm.buffer.stats.fileSizeBytes = OnDiskFileSize(rm.path);
  rm.buffer.stats.fileSizeLabel = "File Size";
  rm.buffer.stats.sdfGrid = std::nullopt;
  rm.buffer.stats.tetMesh = std::nullopt;
  if (rm.type == AssetType::MochiModel) {
    if (auto* asset = _studio->GetAssetManager().FindAssetByPath<MochiModelAsset>(rm.path)) {
      if (asset->GetModelData().sdf.has_value()) {
        rm.buffer.stats.sdfGrid = asset->GetModelData().sdf->dims;
      }
      if (asset->GetModelData().mesh.has_value()) {
        rm.buffer.stats.tetMesh = TetCounts(*asset->GetModelData().mesh);
      }
    }
  }
  ComposeStats(rm.buffer.stats);
  RebuildReferenceMeshes(rm);
}

void ModelEditor::RebuildReferenceMeshes(ReferenceModel& rm) {
  auto* scene = _viewport->GetRenderScene();
  DestroyReferenceModelMeshes(rm);
  StageVisualization const& viz = rm.buffer.viz;
  auto& resourceManager = _studio->GetResourceManager();

  // Wireframe is always built from the transform-baked sections (a darker shade of the color).
  std::vector<float> positions;
  std::vector<float> normals;
  std::vector<int> indices;
  MergeSections(rm.buffer.sections, positions, normals, indices);
  if (!indices.empty()) {
    filament::math::float3 const wire = WireColor(rm.color);
    auto wireframe = mochi_renderer::WireframeMesh::CreateWireframeMesh(
        _studio->GetEngine(),
        mochi::Span<float const>(positions.data(), positions.size()),
        mochi::Span<float const>(normals.data(), normals.size()),
        mochi::Span<int const>(indices.data(), indices.size()),
        resourceManager.CreateWireframeMaterial({wire.x, wire.y, wire.z, 1.0f}),
        resourceManager.CreateWireframeDepthMaterial());
    if (wireframe) {
      rm.buffer.wireframeMesh = scene->AddSceneObjectToScene(std::move(wireframe));
      rm.buffer.wireframeMesh->SetVisible(viz.showWireframe);
    }
  }

  // A render reference keeps its real PBR materials/textures unless overridden or translucent; it
  // is posed with SetLocalTransform (the transform is NOT baked into the instance, so it matches
  // the baked wireframe). All other cases are flat-shaded from the baked sections in the reference
  // color.
  bool const isRender = rm.type == AssetType::RenderModel;
  bool const useFlat = !isRender || rm.overrideColor || viz.opacity < 1.0f;
  if (isRender && !useFlat) {
    if (auto* renderAsset = _studio->GetAssetManager().FindAssetByPath<RenderModelAsset>(rm.path)) {
      if (auto instance = renderAsset->GetRenderModelInstance()) {
        rm.buffer.surfaceMesh = scene->AddSceneObjectToScene(std::move(instance));
        if (rm.buffer.surfaceMesh) {
          rm.buffer.surfaceMesh->SetLocalTransform(rm.rotation, rm.translation, rm.scale, nullptr);
          rm.buffer.surfaceMesh->SetVisible(viz.showSurface);
        }
      }
    }
    return;
  }
  if (indices.empty()) {
    return;
  }
  auto material = viz.opacity >= 1.0f
      ? resourceManager.CreateFlatLitOpaqueMaterial(ToFloat3(rm.color))
      : resourceManager.CreateFlatLitSeeThroughMaterial(ToFloat3(rm.color), viz.opacity);
  auto surface = mochi_renderer::Mesh::CreateMesh(
      _studio->GetEngine(),
      mochi::Span<float const>(positions.data(), positions.size()),
      mochi::Span<float const>(normals.data(), normals.size()),
      mochi::Span<int const>(indices.data(), indices.size()),
      material,
      /*isDynamic=*/false,
      /*isClosed=*/false);
  if (surface) {
    rm.buffer.surfaceMesh = scene->AddSceneObjectToScene(std::move(surface));
    rm.buffer.surfaceMesh->SetVisible(viz.showSurface);
  }
}

void ModelEditor::RegenerateReferenceCadModel(ReferenceModel& rm) {
  mochi::Path const path = rm.path;
  if (path.IsEmpty()) {
    return;
  }
  constexpr double kCadViewLinearDeflection = 0.05; // mm
  constexpr double kCadViewAngularDeflection = 0.25; // rad
  mochi::mesh::StepTessellationParams params;
  params.linearDeflection = kCadViewLinearDeflection;
  params.angularDeflection = kCadViewAngularDeflection;

  auto result = std::make_shared<std::vector<MeshSection>>();
  std::vector<AsyncTask> tasks;
  tasks.push_back(
      AsyncTask{
          "Tessellate " + path.GetFilename(), [path, params, result](AsyncCancelToken const&) {
            mochi::ErrorLog error;
            *result = TessellateCadModelFile(path, params, error);
            return error.IsOK() && !result->empty();
          }});

  // Re-find the reference by its stable id on completion: the vector may have been mutated (add /
  // remove) while the async task ran, so a captured pointer/index could dangle.
  uint64_t const refId = rm.uiId;
  bool const started = _studio->BeginAsyncTasks(
      "Tessellating CAD Reference Model", std::move(tasks), [this, result, refId](bool ok) {
        if (!ok) {
          MOCHI_LOG_WARNING("CAD reference tessellation produced no geometry.");
          return;
        }
        for (ReferenceModel& ref : _referenceModels) {
          if (ref.uiId == refId) {
            // Build the reference CAD asset's render model so the slot gets a thumbnail (parity
            // with the main CAD slot), then display the tessellated geometry.
            if (auto* cadAsset =
                    _studio->GetAssetManager().FindAssetByPath<CadModelAsset>(ref.path)) {
              cadAsset->SetTessellation(*result, _studio->GetResourceManager());
              cadAsset->MarkThumbnailDirty();
            }
            ref.sectionsOriginal = std::move(*result);
            ApplyReferenceModelTransform(ref);
            return;
          }
        }
      });
  if (!started) {
    MOCHI_LOG_WARNING(
        "CAD reference tessellation not started: another async task is already running.");
  }
}

void ModelEditor::ShowModelProcessingWindow(bool* open) {
  ImGui::Begin("Model Processing", open);
  if (ImGui::CollapsingHeader("Mesh Processing Modifiers", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::PushID("MeshProcessingModifiers");
    ShowMeshModifierStack();
    ImGui::PopID();
  }
  ImGui::End();
}

} // namespace superdex::studio
