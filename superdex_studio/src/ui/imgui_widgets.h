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

#include <imgui.h>

#include <mochi_core/mochi_platform.h>
#include <mochi_core/utils/coordinate_space_converter.h>
#include <mochi_renderer/path.h>

#include <math/vec3.h>

#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <superdex_robotics/superdex_robotics.h>

namespace mochi::prefab {
struct ArticulatedLinkPrefab;
struct ArticulatedJointPrefab;
} // namespace mochi::prefab

namespace superdex::studio {

class RenderTarget;
class AssetManager;
class SuperDexStudio;
enum class AssetType;

// Deterministic RGBA color derived by hashing @p str (ImHashStr -> RGBA bytes, alpha forced to 1).
// Gives an asset/model a stable, distinct default color from its path (the same scheme mochi models
// use): the same string always maps to the same color, different strings to different colors.
ImVec4 HashStringToColor(std::string_view str);

// Derives a wireframe/overlay color from a surface color by shifting its HSV value (lightness) ~50%
// toward the opposite extreme (hue and saturation preserved), so the wireframe stays visible
// whether the surface is light or dark. Canonical helper: every wireframe in the app derives its
// color from its surface color through this function, so a surface color change (e.g. a color
// override) drives a matching wireframe color.
filament::math::float3 WireframeColorForSurface(filament::math::float3 surfaceColor);

// Shared axis colors (X/Y/Z) used by viewport gizmos and the XYZ drag widgets.
struct AxisColors {
  ImU32 normal;
  ImU32 hovered;
  ImU32 active;
};

inline constexpr AxisColors kAxisX = {
    IM_COL32(233, 55, 81, 255), // red, normal
    IM_COL32(255, 61, 90, 255), // red, hovered
    IM_COL32(207, 49, 72, 255)}; // red, active
inline constexpr AxisColors kAxisY = {
    IM_COL32(131, 204, 15, 255), // green, normal
    IM_COL32(148, 230, 17, 255), // green, hovered
    IM_COL32(114, 178, 13, 255)}; // green, active
inline constexpr AxisColors kAxisZ = {
    IM_COL32(46, 134, 233, 255), // blue, normal
    IM_COL32(50, 147, 255, 255), // blue, hovered
    IM_COL32(41, 119, 207, 255)}; // blue, active
inline constexpr AxisColors kAxes[3] = {kAxisX, kAxisY, kAxisZ};

// Frame-background highlight for a name input whose value collides with an existing name.
inline ImVec4 const kNameConflictColor(0.5f, 0.0f, 0.0f, 1.0f);

// Unit format enum for type-safe unit display in UI widgets.
enum class UnitFormat {
  // Joint-dependent units (vary by joint type)
  Stiffness, // N·m/rad (revolute) or N/m (prismatic)
  Damping, // N·m·s/rad (revolute) or N·s/m (prismatic)
  ViscousFriction, // N·m·s/rad (revolute) or N·s/m (prismatic)
  CoulombFriction, // N·m (revolute) or N (prismatic)
  Inertia, // kg·m² (revolute) or kg (prismatic)
  Position, // ° (revolute) or m (prismatic) - unit of joint position
  Effort, // N·m (revolute) or N (prismatic)

  // Standalone units (not joint-dependent)
  Degrees, // °
  Length, // m
  Mass, // kg
  Density, // kg/m³
  InertiaTensor, // kg·m² (mass moment of inertia)
};

// Get formatted unit string for joint-dependent units.
// Returns format string like "%.4f N·m/rad" or "%.4e N·m/rad" depending on value.
// The format uses fixed-point for medium values and scientific notation for very large
// or very small values. Precision is configurable. Value parameter enables adaptive
// formatting based on magnitude.
char const* GetUnitFormat(
    UnitFormat unit,
    mochi::ArticulatedJointType jointType,
    float value = 0.0f,
    int precision = 4);

// Get formatted unit string for standalone units.
// Returns format string like "%.4f kg" or "%.4e kg" depending on value.
// The format uses fixed-point for medium values and scientific notation for very large
// or very small values. Precision is configurable. Value parameter enables adaptive
// formatting based on magnitude.
// Asserts if a joint-dependent unit is passed.
char const* GetUnitFormat(UnitFormat unit, float value = 0.0f, int precision = 4);

// Returns a file-name stem (no extension) such that `dir/(stem + extension)` does
// not already exist on disk and, when `alsoTaken` is provided, does not satisfy
// `alsoTaken(candidatePath)`. Appends "_2", "_3", ... to `baseName` until both
// checks pass. Shared so creating and importing assets never silently overwrite a
// file; `alsoTaken` lets a caller additionally avoid paths it reserved earlier in
// the same batch (which are not yet written to disk).
std::string MakeUniqueFileName(
    std::string const& baseName,
    mochi::Path const& dir,
    std::string const& extension = "",
    std::function<bool(mochi::Path const&)> const& alsoTaken = {});

// Optional case transform applied at the end of the batch-rename pipeline.
enum class BatchRenameCase { None, Uppercase, Lowercase };

// Parameters for the shared batch-rename transform: find/replace, trim front/back characters,
// prefix/suffix, and an optional case change. Shared by the asset browser (file renames) and the
// bot editor (link/joint renames).
struct BatchRenameParams {
  std::string find;
  std::string replace;
  int trimFront = 0;
  int trimBack = 0;
  std::string prefix;
  std::string suffix;
  BatchRenameCase caseChange = BatchRenameCase::None;
};

// Applies the batch-rename transform to @p original: replace-all find->replace, then trim
// trimFront/trimBack characters, then wrap with prefix/suffix, then apply the case change. Returns
// the transformed name (possibly empty if trimming consumes the whole string).
std::string ComputeBatchRenamedName(std::string const& original, BatchRenameParams const& params);

} // namespace superdex::studio

namespace ImGui {

// Style colors to be consistent across all Superdex apps.
void StyleColorsSuperdex(ImVec4 const& accentColor = ImVec4(0.000f, 0.455f, 0.898f, 1.000f));

void Separator(ImVec4 const& color);

// Pushes transparent frame/border/button style colors so per-row widgets (combos, inputs,
// checkboxes, icon buttons) blend into a table cell instead of drawing their own frames. Always
// pair with PopFramelessWidgetStyle().
void PushFramelessWidgetStyle();
void PopFramelessWidgetStyle();

bool IconSelectable(char const* label, char const* icon, float text_offset_x = 35.0f);

bool InputText(
    char const* label,
    mochi::DynamicString* str,
    ImGuiInputTextFlags flags = 0,
    ImGuiInputTextCallback callback = nullptr,
    void* user_data = nullptr);

// Name InputText that highlights (kNameConflictColor) and shows a "name already used" tooltip when
// @p collides is true. Returns true if the text changed. Shared by the actor detail panels.
bool NameInputWithCollisionCheck(char const* label, mochi::DynamicString& name, bool collides);

void RenderTargetImage(
    superdex::studio::RenderTarget const* renderTarget,
    int width = -1,
    int height = -1);
bool RenderTargetButton(
    char const* str_id,
    superdex::studio::RenderTarget const* renderTarget,
    int width = -1,
    int height = -1);

bool ViewportOrientationGizmo(
    char const* str_id,
    double const* view_matrix,
    ImVec2 center,
    float radius,
    int* out_clicked_axis,
    bool* out_dragging,
    ImFont* font,
    mochi::CoordinateSpaceConverter const& converter);

bool DragFloatXYZ(
    char const* label,
    float v[3],
    float v_speed = 1.0f,
    float v_min = 0.0f,
    float v_max = 0.0f,
    char const* format = "%.4f",
    ImGuiSliderFlags flags = 0);

// Platform-specific UV defaults for render target display
#if MOCHI_PLATFORM_MACOS
// Metal: texture origin is top-left, matches ImGui - no flip needed
inline constexpr ImVec2 kDefaultRenderTargetUV0{0, 0};
inline constexpr ImVec2 kDefaultRenderTargetUV1{1, 1};
#else
// OpenGL: texture origin is bottom-left, ImGui expects top-left - flip Y
inline constexpr ImVec2 kDefaultRenderTargetUV0{0, 1};
inline constexpr ImVec2 kDefaultRenderTargetUV1{1, 0};
#endif

bool AssetThumbnail(
    char const* str_id,
    ImTextureID image,
    float thumb_size,
    ImU32 stripe_color,
    ImVec2 const& uv0 = kDefaultRenderTargetUV0,
    ImVec2 const& uv1 = kDefaultRenderTargetUV1,
    int num_checker = 10);

bool SimpleAssetThumbnail(
    char const* str_id,
    float thumb_size,
    char const* icon,
    ImFont const* icon_font,
    ImU32 stripe_color);

bool FolderThumbnail(
    char const* str_id,
    float thumb_size,
    ImFont const* icon_font,
    ImU32 icon_color);

struct TileRenameState {
  std::string buffer;
  bool valid = true;
  std::function<void(std::string const& newName)> onFinished;
  std::function<void()> onCanceled;
};

enum AssetTileState { AssetTileState_None, AssetTileState_Unsaved, AssetTileState_ReadOnly };

bool AssetTile(
    char const* str_id,
    ImTextureID image,
    float image_size,
    char const* name,
    char const* type,
    ImFont const* small_font,
    ImU32 stripe_color,
    AssetTileState state = AssetTileState_None,
    bool selected = false,
    char const* extra_icon = nullptr,
    ImU32 extra_icon_color = IM_COL32_BLACK_TRANS,
    ImVec2 const& uv0 = kDefaultRenderTargetUV0,
    ImVec2 const& uv1 = kDefaultRenderTargetUV1,
    int num_checker = 10,
    TileRenameState* rename = nullptr);

bool SimpleAssetTile(
    char const* str_id,
    float tile_size,
    char const* icon,
    ImFont const* icon_font,
    char const* name,
    char const* type,
    ImFont const* small_font,
    ImU32 stripe_color,
    bool selected = false,
    TileRenameState* rename = nullptr);

bool FolderTile(
    char const* str_id,
    float tile_size,
    ImFont const* icon_font,
    char const* name,
    ImFont const* small_font,
    ImU32 icon_color,
    bool selected = false,
    TileRenameState* rename = nullptr);

// Renders a 64px asset thumbnail slot for @p assetPath plus a path field and browse button.
// Returns true when the slotted path changes (drag-drop or clear). Double-clicking the thumbnail
// opens the asset's editor. Pass @p acceptDragDropPayload = false and @p showClearButton = false to
// render a read-only (locked) slot that can still be viewed and browsed to but not reassigned or
// cleared.
bool AssetSlot(
    char const* label,
    mochi::DynamicString& assetPath,
    superdex::studio::AssetManager const& assetManager,
    superdex::studio::SuperDexStudio* studio,
    superdex::studio::AssetType assetType,
    bool acceptDragDropPayload,
    bool showClearButton = true);

bool DragReal(
    char const* label,
    mochi::real* value,
    float v_speed = 0.01f,
    float v_min = 0.0f,
    float v_max = 0.0f,
    char const* format = "%.4f",
    float scale = 1.0f);

bool DragReal3(
    char const* label,
    mochi::Real3& value,
    float v_speed = 0.01f,
    float v_min = 0.0f,
    float v_max = 0.0f,
    char const* format = "%.4f",
    float scale = 1.0f);

bool DragRealXYZ(
    char const* label,
    mochi::Real3& value,
    float v_speed = 0.01f,
    float v_min = 0.0f,
    float v_max = 0.0f,
    char const* format = "%.4f",
    ImGuiSliderFlags flags = 0,
    float scale = 1.0f);

bool DragInertia(
    char const* label,
    mochi::Real6& value,
    float v_speed = 0.01f,
    float v_min = 0.0f,
    float v_max = 0.0f);

enum class QuaternionMode {
  XYZW, // Raw quaternion components (default)
  RPY // Roll-Pitch-Yaw angles in radians
};

bool DragQuaternion(
    char const* label,
    mochi::Quaternion& value,
    QuaternionMode mode = QuaternionMode::RPY);

bool DragQuaternion(
    char const* label,
    float* x,
    float* y,
    float* z,
    float* w,
    QuaternionMode mode = QuaternionMode::XYZW);

bool DragTransformRT(char const* label, mochi::Quaternion& rotation, mochi::Real3& translation);
bool DragTransformRT(char const* label, mochi::TransformRT& value);

// Filter controlling which joint types are offered by ComboArticulatedJointType.
enum class ArticulatedJointTypeFilter {
  All, // All joint types.
  NoFreeCycle, // All joint types except Free and Cycle.
  HardFreeOnly, // Only Free and Hard joint types.
};

bool ComboArticulatedJointType(
    char const* label,
    mochi::ArticulatedJointType& type,
    ArticulatedJointTypeFilter filter);

bool DragOptionalReal(
    char const* label,
    std::optional<mochi::real>& value,
    float v_speed = 0.01f,
    float v_min = 0.0f,
    float v_max = 0.0f,
    char const* format = "%.6f",
    float scale = 1.0f);

bool DragOptionalRealXYZ(
    char const* label,
    std::optional<mochi::Real3>& value,
    float v_speed = 0.01f,
    float v_min = 0.0f,
    float v_max = 0.0f,
    char const* format = "%.4f",
    ImGuiSliderFlags flags = 0,
    float scale = 1.0f);

// Lays out the "icon | value | label" row format: draws @p icon in a square, vertically-centered
// cell the width of a checkbox, then advances the cursor and sets the next item's width so that a
// single following labeled widget (e.g. InputFloat, DragReal, a combo) left-aligns with the value
// column of checkbox-fronted rows (see @ref DragOptionalReal) and shares their right edge. Shows
// @p tooltip on hover when non-null. Call exactly one item immediately afterwards.
void IconInputPrefix(char const* icon, char const* tooltip = nullptr);

bool TextButton(char const* label);

bool ButtonColored(char const* label, ImVec4 const& color, ImVec2 const& size = ImVec2(0, 0));

// Renders one segment of a split-button (e.g. a toggle joined to a dropdown caret). Only the
// corners in @p corners are rounded, so two adjacent segments butt together into a single pill with
// a flat seam between them. When @p active is true the segment is filled with the accent color used
// by active toolbar buttons; hover/press states are handled as usual. The glyph is centered within
// the segment regardless of its width (so a narrow caret stays centered), and it rounds to half the
// height to match fully-rounded pill buttons. Returns true when clicked.
bool SplitButtonSegment(char const* label, ImVec2 size, ImDrawFlags corners, bool active);

bool ColorSwatchButton(
    char const* label,
    ImU32 swatch_color,
    bool grayed_out = false,
    float width = 0.0f);

void HoverableSeparatorText(char const* label);

// Tooltip for the most recently submitted item, wrapping long text over several lines instead of
// running off the screen edge. Shown even when the item is disabled. Use instead of SetTooltip for
// prose (SetTooltip never wraps, so the caller would have to hard-code newlines).
void ItemTooltipWrapped(char const* text);

// True/False dropdown, the shape reflected bool fields get. Preferred over Checkbox in settings
// panels: it fills the value column like the drag widgets do, with the value centered to match
// them. Returns true when the value changed.
bool BoolCombo(char const* label, bool* value);

// Widget override callback type. Return true to skip the default widget for this field.
using SimpleReflectionWidgetOverride =
    bool (*)(char const* label, SReflect::FieldTypeInfo const& fieldInfo, void* fieldPtr);

// Per-field read-only predicate. Complements Attribute_ReadOnly for state the reflection metadata
// cannot know, such as a field the loaded asset supplies. Matching fields are drawn disabled.
using SimpleReflectionReadOnlyPredicate = bool (*)(SReflect::FieldTypeInfo const& fieldInfo);

// Render editable widgets for all reflected fields of a struct (type-erased entry point).
bool SimpleReflectionStruct(
    SReflect::StructTypeInfo const& typeInfo,
    void* structPtr,
    SimpleReflectionWidgetOverride overrideFn = nullptr,
    SimpleReflectionReadOnlyPredicate readOnlyFn = nullptr);

// Render editable widgets for all reflected fields of a struct (template convenience wrapper).
template <typename T>
bool SimpleReflectionStruct(
    T& value,
    SimpleReflectionWidgetOverride overrideFn = nullptr,
    SimpleReflectionReadOnlyPredicate readOnlyFn = nullptr) {
  auto const& typeInfo = SReflect::GetTypeInfo<T>();
  return SimpleReflectionStruct(
      static_cast<SReflect::StructTypeInfo const&>(typeInfo), &value, overrideFn, readOnlyFn);
}

// Render an editable combo for a reflected enum (type-erased entry point).
bool SimpleReflectionEnum(char const* label, SReflect::EnumTypeInfo const& enumInfo, void* enumPtr);

// Render an editable combo for a reflected enum (template convenience wrapper).
template <typename T>
bool SimpleReflectionEnum(char const* label, T& value) {
  auto const& enumInfo = static_cast<SReflect::EnumTypeInfo const&>(SReflect::GetTypeInfo<T>());
  return SimpleReflectionEnum(label, enumInfo, &value);
}

// Renders the "Inertial Properties" section: a setup combo (Default / Density Only / Mass Only /
// Mass,COM,MOI / Density,COM,MOI) plus the conditional density/mass/COM/inertia widgets. Operates
// directly on the optional inertial fields shared by rigid actors and articulated links. Returns
// true if any value changed.
bool InertialProperties(
    std::optional<mochi::real>& density,
    std::optional<mochi::real>& mass,
    std::optional<mochi::Real3>& centerOfMass,
    std::optional<mochi::Real6>& momentOfInertia);

// Renders a model editor section (asset slot + link/unlink toggle + transform + scale) under a
// SeparatorText(label). When the link toggle is enabled and the corresponding `other*` pointers
// are non-null, edits to rotation/translation/scale are mirrored onto them, keeping the Mochi-model
// and render-model transforms in sync. Sets `modelChanged` when the asset path changes. Returns
// true if any value changed.
bool ModelEditor(
    char const* label,
    superdex::studio::SuperDexStudio* studio,
    superdex::studio::AssetType assetType,
    mochi::DynamicString& path,
    mochi::Real3& scale,
    mochi::Quaternion& rotation,
    mochi::Real3& translation,
    mochi::Real3* otherScale,
    mochi::Quaternion* otherRotation,
    mochi::Real3* otherTranslation,
    superdex::studio::AssetManager const& assetManager,
    bool acceptDragDropPayload,
    bool& modelChanged);

// Renders a "Collision / Contact" section under a SeparatorText(label): collider-type combo,
// contact params, boundary element type, and (only when @p boundarySubsampling is non-null) the
// boundary-subsampling checkbox + density + strategy. Pass null for actor types with no
// boundary-subsampling field (e.g. soft actors). Returns true if any value changed.
bool CollisionContact(
    char const* label,
    mochi::ColliderType& colliderType,
    mochi::ContactParams& contact,
    mochi::ActorBoundaryElementType& boundaryElementType,
    std::optional<mochi::BoundarySubsamplingParams>* boundarySubsampling = nullptr);

// Renders the editable widgets for an articulated link's shared (base-class) fields: Info
// (name/layer/hasGravity), the "Parent Joint From Link" transform, inertial properties, the Mochi
// and render model editors (cross-wired so their transforms stay in sync), and collision/contact.
// Does not push an ID scope; callers should wrap in a PushID to disambiguate from sibling editors.
// Sets `modelChanged` when a model asset path changes. Returns true if any value changed.
bool ArticulatedLinkEditor(
    mochi::prefab::ArticulatedLinkPrefab& link,
    superdex::studio::SuperDexStudio* studio,
    superdex::studio::AssetManager const& assetManager,
    bool acceptDragDropPayload,
    bool& modelChanged);

// Renders the editable widgets for an articulated joint's shared (base-class) fields: Info
// (name + joint-type combo, filtered by `isRoot`), the "Parent Link From Joint" transform, the
// axis/limits section (including scalar<->axis limit reprojection), and the friction/inertia
// dynamics. Does not push an ID scope; callers should wrap in a PushID to disambiguate from sibling
// editors. Excludes bot-specific fields (e.g. effort limit). Returns true if any value changed.
bool ArticulatedJointEditor(mochi::prefab::ArticulatedJointPrefab& joint, bool isRoot);

// Renders the joint-pose editor for a built bot: preset buttons (Zero / Min / Max / Mid / Random,
// produced via superdex::robotics::MakeBotPose) followed by one limit-aware widget per DOF -- a
// SliderFloat when the joint has finite limits, otherwise a DragFloat. Revolute and spherical DOFs
// are shown in degrees, prismatic DOFs in meters. Resizes @p pose to the bot's DOF count. Does not
// render a section header (callers frame it, e.g. with HoverableSeparatorText). No-op returning
// false if
// @p builtPrefab has no DOFs. Returns true if @p pose changed.
bool JointPoseEditor(
    superdex::robotics::BotPrefab const& builtPrefab,
    mochi::DynamicArray<mochi::real>& pose);

// Renders the batch-rename input controls (Find, Replace, "Trim Front/Back" SliderInt2, Prefix,
// Suffix, "Case Change" combo) operating directly on @p params. @p trimMax caps the trim sliders
// (callers derive it from their entry set's longest name). Returns true if any input changed.
bool BatchRenameInputs(superdex::studio::BatchRenameParams& params, int trimMax);

// Renders the two-column "Old Name -> New Name" preview table (frozen header, vertical scroll).
// @p oldNames, @p newNames, and @p rowInvalid are parallel arrays. Rows flagged in @p rowInvalid
// render the new name in red, showing "<empty>" when the new name is empty. Callers handle their
// own empty-state (no entries) messaging.
void BatchRenamePreviewTable(
    std::vector<std::string> const& oldNames,
    std::vector<std::string> const& newNames,
    std::vector<bool> const& rowInvalid,
    float heightPx = 300.0f);

} // namespace ImGui
