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

#include "meshing/collision_recipe.h"
#include "meshing/link_meshes.h"
#include "prepare_urdf_command.h"

#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/error.h>
#include <mochi_renderer/path.h>
#include <mochi_renderer/render_space.h>
#include <superdex_robotics/utils/file_utils.h>

#include <cstdio>
#include <filesystem>
#include <string>
#include <system_error>
#include <unordered_set>
#include <utility>
#include <vector>

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "Usage: %s <source.urdf> <empty-output-directory>\n", argv[0]);
    return 1;
  }
  namespace fs = std::filesystem;
  namespace robotics = superdex::robotics;
  namespace studio = superdex::studio;
  return studio::RunUrdfPreparation([&]() {
    fs::path const output = fs::absolute(argv[2]);
    if (fs::exists(output) && (!fs::is_directory(output) || !fs::is_empty(output))) {
      std::fprintf(stderr, "Output directory must be empty\n");
      return 1;
    }
    mochi::Error error;
    robotics::UrdfImportInfo info;
    std::vector<std::string> warnings;
    MOCHI_DEFER({
      for (auto const& warning : info.warnings) {
        std::fprintf(stderr, "Import warning: %s\n", warning.c_str());
      }
      for (auto const& warning : warnings) {
        std::fprintf(stderr, "Import warning: %s\n", warning.c_str());
      }
    });
    auto prefab = robotics::LoadBotPrefabFromUrdfFile(fs::absolute(argv[1]).string(), info, error);
    if (!error.IsOK()) {
      std::fprintf(stderr, "%s\n", error.GetDescription());
      return 1;
    }
    std::unordered_set<mochi::Path> outputPaths;
    for (auto const& link : prefab.links) {
      std::string const name(link.name);
      if (name.empty() || name.find_first_of("/\\") != std::string::npos || name == "." ||
          name == "..") {
        std::fprintf(stderr, "Unsafe link name\n");
        return 1;
      }
      if (!outputPaths.insert(mochi::Path{output / "render" / (name + ".glb")}).second) {
        std::fprintf(
            stderr, "Link output filenames must be distinct ignoring case: %s\n", name.c_str());
        return 1;
      }
    }
    for (auto const& link : info.links) {
      for (auto const& geom : link.geoms) {
        if (geom.type == robotics::GeomType::Mesh && geom.meshFile.empty()) {
          std::fprintf(stderr, "Missing mesh: %s\n", geom.meshRef.c_str());
          return 1;
        }
      }
    }
    bool complete = false;
    bool createdOutput = false;
    bool writingBot = false;
    std::vector<fs::path> ownedDirectories;
    ownedDirectories.reserve(3);
    auto removeOwned = [](fs::path const& path, bool recursive) {
      std::error_code cleanupError;
      if (recursive) {
        fs::remove_all(path, cleanupError);
      } else {
        fs::remove(path, cleanupError);
      }
      if (cleanupError) {
        std::fprintf(
            stderr,
            "Output cleanup failed for %s: %s\n",
            path.string().c_str(),
            cleanupError.message().c_str());
      }
    };
    MOCHI_DEFER({
      if (!complete) {
        if (writingBot) {
          removeOwned(output / "asset.superdex_bot", false);
        }
        for (auto it = ownedDirectories.rbegin(); it != ownedDirectories.rend(); ++it) {
          removeOwned(*it, true);
        }
        if (createdOutput) {
          removeOwned(output, false);
        }
      }
    });
    createdOutput = fs::create_directories(output);
    for (char const* name : {"collision", "render", "recipes"}) {
      fs::path directory = output / name;
      if (!fs::create_directory(directory)) {
        std::fprintf(stderr, "Output directory changed during preparation\n");
        return 1;
      }
      ownedDirectories.push_back(std::move(directory));
    }
    mochi::CoordinateSpaceConverter const toRender(
        mochi::CoordinateSpace::Default(), mochi_renderer::RenderSpace());
    for (size_t i = 0; i < info.links.size(); ++i) {
      auto& link = prefab.links[i];
      std::string const name(link.name);
      studio::GeomList visuals;
      std::vector<studio::SourcePart> collisions;
      for (auto const& geom : info.links[i].geoms) {
        if (geom.isVisual) {
          visuals.push_back(geom);
        }
        if (geom.isCollision) {
          collisions.push_back(studio::PartFromGeom(geom, std::string(geom.meshFile)));
        }
      }
      if (!visuals.empty()) {
        mochi::Path const file{(output / "render" / (name + ".glb")).string()};
        if (!studio::BuildLinkGlb(visuals, file, toRender, warnings)) {
          std::fprintf(stderr, "Render conversion failed for %s\n", name.c_str());
          return 1;
        }
        link.renderModelFile = file.ToString();
      }
      if (!collisions.empty()) {
        fs::path const shape = output / "collision" / (name + ".mochi.h5");
        studio::CollisionRecipeOptions options;
        options.remesh = false;
        options.bakeSdf = true;
        auto recipe = studio::MakeCollisionRecipe(std::move(collisions), options, shape.string());
        if (!studio::WriteCollisionRecipe(
                recipe, output / "recipes" / (name + ".StudioProcessing.json"), true, error)) {
          std::fprintf(stderr, "%s\n", error.GetDescription());
          return 1;
        }
        link.shapeFile = shape.string();
      }
    }
    fs::path const bot = output / "asset.superdex_bot";
    if (fs::exists(bot)) {
      std::fprintf(stderr, "Output directory changed during preparation\n");
      return 1;
    }
    writingBot = true;
    robotics::SaveToFile(prefab, bot.string(), error);
    if (!error.IsOK()) {
      std::fprintf(stderr, "%s\n", error.GetDescription());
      return 1;
    }
    complete = true;
    std::printf(
        "Prepared %zu links; run the emitted collision recipes before native loading\n",
        prefab.links.size());
    return 0;
  });
}
