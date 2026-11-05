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

// The asset types Studio knows, how a file's name classifies it, and where a model's siblings and
// generated files live. Only the file system is involved, so code that runs without the app, such
// as the processing command, can use it.

#include <mochi_renderer/path.h>

#include <array>
#include <string>
#include <string_view>

namespace superdex::studio {

enum class AssetType {
  RenderModel,
  MochiModel,
  MochiPrefab,
  Bot,
  BotScene,
  BotTask,
  CadModel,
  Tuner,
  Unknown,
  Count
};

constexpr int kAssetTypeCount = static_cast<int>(AssetType::Count);

constexpr std::array<char const*, kAssetTypeCount> kAssetTypeLabels = {
    "Render Model",
    "Collision Model",
    "Prefab",
    "Bot",
    "Bot Scene",
    "Task",
    "CAD Model",
    "Asset Tuner",
    "(Unknown)"};

inline constexpr char const* GetAssetTypeLabel(AssetType type) {
  return kAssetTypeLabels[static_cast<int>(type)];
}

struct AssetTypeExtension {
  std::string_view extension;
  AssetType type;
};

// Every extension Studio classifies, grouped by type and ordered within each type from most to
// least preferred. Both jobs read this one table: ClassifyAssetTypeByFilename takes the matching
// entry's type, and asset discovery uses the entry's position to break ties when one folder holds
// the same base name in two formats of a single type (`x.glb` beside `x.obj`). Keeping them
// together means a format cannot be taught to one and forgotten by the other.
//
// Order is significant twice over: the first matching entry wins, so an extension must precede any
// shorter one that is its own suffix; and within a type, earlier means preferred.
constexpr std::array<AssetTypeExtension, 22> kAssetTypeExtensions{
    {{".glb", AssetType::RenderModel},
     {".gltf", AssetType::RenderModel},
     {".dae", AssetType::RenderModel},
     {".obj", AssetType::RenderModel},
     // STL is a CAD model (a raw triangle soup, no PBR materials): it shares the CAD slot and is
     // fed to the mesh-processing stack the same way as a tessellated STEP.
     {".stp", AssetType::CadModel},
     {".step", AssetType::CadModel},
     {".stl", AssetType::CadModel},
     {".mochi.h5", AssetType::MochiModel},
     // Bot scenes and tasks are an internal-build concept; public builds do not classify these.
     {".superdex_scene", AssetType::BotScene},
     {".superdex_scene_archive", AssetType::BotScene},
     {".mochi_bot_scene", AssetType::BotScene},
     {".mochi_bot_scene_archive", AssetType::BotScene},
     {.extension = ".superdex_task", .type = AssetType::BotTask},
     {.extension = ".mochi_bot_task", .type = AssetType::BotTask},
     {".superdex_bot_archive", AssetType::Bot},
     {".superdex_bot", AssetType::Bot},
     {".mochi_bot_archive", AssetType::Bot},
     {".mochi_bot", AssetType::Bot},
     {".mochi_scene", AssetType::MochiPrefab},
     {".superdex_prefab", AssetType::MochiPrefab},
     {".mochi_prefab", AssetType::MochiPrefab},
     {".superdex_tuner", AssetType::Tuner}}};

AssetType ClassifyAssetTypeByFilename(std::string_view filename);
AssetType ClassifyAssetTypeByPath(mochi::Path const& path);
std::string GetAssetNameFromPath(mochi::Path const& path);

// Stable, compact, space-free token for an AssetType (e.g. "RenderModel"), and its inverse. Used
// for JSON persistence where the enum's integer value would be fragile across reordering. Unknown
// input maps to AssetType::Unknown.
std::string_view AssetTypeToToken(AssetType type);
AssetType AssetTypeFromToken(std::string_view token);

// The role folder a model of @p type belongs in (CAD -> `cad`, render -> `render`, mochi ->
// `collision`). Empty for every non-model type. This maps a type to a folder; it never works in
// reverse -- a file's type is decided by its extension alone, so an `.stl` in `collision/` is a CAD
// model that happens to live there, and the CAD slot stays loadable by the mesh-processing stack.
std::string_view AssetRoleFolderForType(AssetType type);

// Locate a file directly in @p dir whose base name equals @p baseName (case-insensitively) and
// whose classified asset type is @p type. Uses the shared classify/name helpers above, so
// multi-extension types (e.g. the render formats) and the `.mochi.h5` double extension are handled.
// Several extensions can classify as one type, so the winner is chosen by @ref kAssetTypeExtensions
// order and then by path, never by directory-iteration order. Returns the file path, or empty if
// none.
std::string
FindAssetFileByName(std::string const& baseName, AssetType type, mochi::Path const& dir);

// Discover the same-basenamed asset of @p type that belongs with a file in @p originDir -- the
// shared search editors use to auto-populate their model slots when a single asset is opened.
//
// Anchored on the asset's base folder (superdex::robotics::FindAssetBaseFolder), then searched in a
// fixed order, first hit wins, so exactly one file can ever fill a slot:
//   1. `<base>/<role folder for type>`  -- the role-subfolder layout
//   2. `<base>`                         -- the flat layout
//   3. @p originDir                     -- whatever folder the opened file actually sits in
//   4. the remaining role folders, plus the `visual` / `visuals` aliases a URDF source tree uses
//
// Role folders (steps 1 and 4) are searched as trees, breadth-first, so a role partitioned into
// subfolders -- `cad/internal/` for CAD sources that must not be open-sourced -- still resolves,
// with the shallowest match winning. The base folder and @p originDir are searched flat: a base
// folder can hold whole other assets (a two-handed bot's `left/` and `right/`), so descending from
// it would let one asset's models fill another's slots.
//
// Step 4 is the only step that can see two equally-plausible candidates, because the folders it
// sweeps have no precedence among themselves. It therefore requires a unique hit: two or more and
// it fills nothing and logs the collision, since an empty slot the user can browse to fill beats
// silently feeding the wrong mesh into a pipeline. `intermediates` is never searched -- it holds
// generated output, which must not shadow the real asset.
//
// Returns the path, or empty if none.
std::string
FindAssetForSlot(std::string const& baseName, AssetType type, mochi::Path const& originDir);

// Path of a Studio-generated file belonging to the model at @p originPath -- its saved
// `.StudioProcessing.json` pipeline, or a modifier's scratch export -- formed by appending
// @p suffix to a stem identifying the model, inside the asset's `intermediates` folder.
//
// @p isCanonical is whether discovery for the model's own type resolves back to it (see
// FindAssetForSlot). A canonical model keys on its base name, which is what every pipeline already
// on disk uses and what makes one pipeline serve the whole cad/render/collision set.
//
// A shadowed model is a set of one, so keying it the same way would point it at the pipeline of the
// model shadowing it -- silently loading that model's work and overwriting it on save. It instead
// keys on its own file name, under a mirror of its location within the asset, so
// `render/internal/part.glb` gets `intermediates/render/internal/part.glb.StudioProcessing.json`:
// distinct from the canonical file, distinct from any other shadowed copy, and stable, so the work
// saved there loads back on reopen.
std::string
AssetGeneratedFilePath(mochi::Path const& originPath, bool isCanonical, std::string_view suffix);

} // namespace superdex::studio
