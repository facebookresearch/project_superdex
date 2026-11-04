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

#include "io/mjcf_importer.h"

#include <superdex_robotics/utils/file_utils.h>

#include <mochi_core/utils/error.h>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

namespace superdex::studio {

using superdex::robotics::GeomReference;
using superdex::robotics::GeomType;
using superdex::robotics::MjcfBodyReference;

// How far into a .xml file to look for the <mujoco> root element.
static constexpr std::streamsize kSniffBytes = 64 * 1024;

MjcfImporter::MjcfImporter(SuperDexStudio* studio)
    : BotImporter(
          studio,
          Format{
              .displayName = "MJCF",
              .extensions = {std::string(superdex::robotics::kMjcfExtension), ".mjcf"},
              // MJCF geoms are always converted (merged into one model per link).
              .allowsCopy = false,
              // A body without collision geoms does not collide in MuJoCo.
              .collisionFromVisuals = false}) {}

bool MjcfImporter::CanImportFile(mochi::Path const& path) const {
  std::ifstream stream(path.AsFilesystemPath(), std::ios::binary);
  std::string head(static_cast<size_t>(kSniffBytes), '\0');
  stream.read(head.data(), kSniffBytes);
  head.resize(static_cast<size_t>(stream.gcount()));
  // A <mujoco> element, not <mujocoinclude> or other text that merely starts the same way.
  for (size_t at = head.find("<mujoco"); at != std::string::npos;
       at = head.find("<mujoco", at + 1)) {
    size_t const next = at + std::string_view("<mujoco").size();
    if (next < head.size() &&
        (std::isspace(static_cast<unsigned char>(head[next])) || head[next] == '>' ||
         head[next] == '/')) {
      return true;
    }
  }
  return false;
}

bool MjcfImporter::LoadSource(mochi::Path const& path, std::vector<std::string>& notes) {
  mochi::ErrorLog error;
  _info = {};
  _botPrefab = superdex::robotics::LoadBotPrefabFromMjcfFile(path.ToString(), _info, error);
  for (auto const& warning : _info.warnings) {
    notes.emplace_back(warning);
  }
  return error.IsOK();
}

int MjcfImporter::NumSourceLinks() const {
  return static_cast<int>(_info.bodies.size());
}

std::string MjcfImporter::SourceLinkName(int source) const {
  return std::string(_info.bodies[source].link);
}

mochi::DynamicArray<GeomReference>& MjcfImporter::SourceGeoms(int source) {
  return _info.bodies[source].geoms;
}

void MjcfImporter::PrepareGeoms(std::vector<std::string>& notes) {
  _objCopies.reset();
  mochi::ErrorLog tempDirError;
  _objCopies.emplace(mochi::CreateTempDirectory("mjcf_import", tempDirError));
  std::unordered_map<std::string, std::string> objFiles;
  for (MjcfBodyReference& body : _info.bodies) {
    for (GeomReference& geom : body.geoms) {
      std::string extension =
          std::filesystem::path(std::string(geom.meshFile)).extension().string();
      std::ranges::transform(extension, extension.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
      });
      if (extension == ".obj") {
        auto const [file, inserted] = objFiles.try_emplace(std::string(geom.meshFile));
        if (inserted) {
          file->second = FirstObjObjectFile(file->first, _objCopies->Path());
        }
        geom.meshFile = file->second;
      }
      if (geom.type != GeomType::Mesh && !geom.meshFile.empty()) {
        FitPrimitiveToMesh(geom, notes);
      }
    }
  }
}

void MjcfImporter::AddLinkTasks(
    int source,
    superdex::robotics::BotLinkPrefab* link,
    std::vector<AsyncTask>& tasks) {
  MjcfBodyReference const& body = _info.bodies[source];
  if (body.hasInertial || body.geoms.empty()) {
    return;
  }
  std::vector<std::string>* warnings = AddTaskWarnings();
  tasks.push_back(
      AsyncTask{
          "Compute the inertia of " + std::string(body.link),
          [link, geoms = GeomList(body.geoms.begin(), body.geoms.end()), warnings](
              AsyncCancelToken const& cancel) {
            if (cancel.IsCancelRequested()) {
              return false;
            }
            if (std::optional<GeomsInertia> const inertia = ComputeGeomsInertia(geoms, *warnings)) {
              link->mass = inertia->mass;
              link->centerOfMass = inertia->centerOfMass;
              link->momentOfInertia = inertia->momentOfInertia;
            }
            return true;
          }});
}

void MjcfImporter::FinishPrefab(bool anySeparate, mochi::Error& error) {
  if (anySeparate) {
    superdex::robotics::ExpandMjcfGeomLinks(_botPrefab, _info, error);
    MOCHI_ERROR_RETURN(error);
  }
  superdex::robotics::ApplyMjcfContactFilter(_botPrefab, _info);
}

} // namespace superdex::studio
