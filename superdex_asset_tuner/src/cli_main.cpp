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

#include "superdex_asset_tuner/json_path.h"
#include "superdex_asset_tuner/scene_targets.h"
#include "superdex_asset_tuner/tuner_runner.h"
#include "superdex_asset_tuner/tuner_spec.h"

#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace superdex::tuner;

constexpr char kUsage[] =
    "Usage:\n"
    "  superdex_asset_tuner run <file.superdex_tuner> [--set path=value]... [--out DIR] [--tag TAG]\n"
    "                     [--no-files]\n"
    "  superdex_asset_tuner describe <file.superdex_tuner | file.mochi_scene>\n"
    "\n"
    "run       Simulates the tuner headlessly and prints summary.json to stdout. Writes a run\n"
    "          directory (tuner.resolved.json, signals.csv, summary.json, sim.rrd) under --out\n"
    "          (default: ./tuner_runs) unless --no-files is given.\n"
    "          --set applies a JSON override to the tuner spec, e.g. --set run.dt=0.005 or\n"
    "          --set loads[name=push].profile.magnitude=3\n"
    "describe  Prints the rigid actors, articulated links and joints (with DoF ranges) that\n"
    "          loads and probes can target.\n"
    "\n"
    "Exit codes: 0 ok (including runs ended by a stop condition; see summary.json status),\n"
    "1 error, 2 simulation diverged.\n";

constexpr int kExitOk = 0;
constexpr int kExitError = 1;
constexpr int kExitDiverged = 2;

int Fail(std::string const& message) {
  std::fprintf(stderr, "superdex_asset_tuner: %s\n", message.c_str());
  return kExitError;
}

bool LoadSpecOrPrefab(
    std::filesystem::path const& path,
    std::vector<SpecOverride> const& overrides,
    TunerSpec& outSpec,
    std::string& outError) {
  if (path.extension() == kTunerFileExtension) {
    return LoadTunerSpecFromFile(path, overrides, outSpec, outError);
  }
  outSpec = {};
  outSpec.prefab = DynamicString(std::filesystem::absolute(path).string());
  return true;
}

int Describe(std::filesystem::path const& path) {
  TunerSpec spec;
  SceneTargets targets;
  std::string error;
  if (!LoadSpecOrPrefab(path, {}, spec, error) || !DescribeTuner(spec, path, targets, error)) {
    return Fail(error);
  }
  std::printf("%s\n", SReflect::ToJsonString(targets, true).c_str());
  return kExitOk;
}

int Run(int argc, char** argv) {
  std::filesystem::path tunerPath;
  std::vector<SpecOverride> overrides;
  RunOptions options{.outDir = "tuner_runs"};
  for (int i = 2; i < argc; ++i) {
    std::string_view const arg = argv[i];
    bool const hasValue = i + 1 < argc;
    if (arg == "--set" && hasValue) {
      std::string_view const assignment = argv[++i];
      SpecOverride entry;
      if (!SplitAssignment(assignment, entry.path, entry.value)) {
        return Fail("--set expects path=value, got '" + std::string(assignment) + "'.");
      }
      overrides.push_back(std::move(entry));
    } else if (arg == "--out" && hasValue) {
      options.outDir = argv[++i];
    } else if (arg == "--tag" && hasValue) {
      options.tag = argv[++i];
    } else if (arg == "--no-files") {
      options.outDir.clear();
    } else if (!arg.starts_with("--") && tunerPath.empty()) {
      tunerPath = arg;
    } else {
      return Fail("Unexpected argument '" + std::string(arg) + "'.\n" + kUsage);
    }
  }
  if (tunerPath.empty()) {
    return Fail(std::string("Missing tuner file.\n") + kUsage);
  }

  TunerSpec spec;
  RunSummary summary;
  std::string error;
  if (!LoadTunerSpecFromFile(tunerPath, overrides, spec, error) ||
      !RunTuner(spec, tunerPath, options, summary, error)) {
    return Fail(error);
  }
  std::printf("%s\n", SReflect::ToJsonString(summary, true).c_str());
  return std::string_view(summary.status) == "diverged" ? kExitDiverged : kExitOk;
}

} // namespace

int main(int argc, char** argv) {
  std::string_view const command = argc > 1 ? argv[1] : "";
  if (command == "run") {
    return Run(argc, argv);
  }
  if (command == "describe" && argc == 3) {
    return Describe(argv[2]);
  }
  std::fprintf(stderr, "%s", kUsage);
  return command == "--help" || command == "-h" ? kExitOk : kExitError;
}
