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

#include <benchmark/benchmark.h>
#include <mochi_core/mochi_init.h>
#include <mochi_core/utils/log.h>
#include <mochi_core/utils/path.h>

#include "config.h"

#include <filesystem>
#include <functional>
#include <string>

// Declared in config.h
std::string mochi_benchmark::GetAssetPath(std::string const& relative) {
  mochi::Error error;
  std::string assetsDir = mochi::path::FindAssetsDirectory(error);
  std::string fullPath = assetsDir + relative;
  MOCHI_ASSERT(
      error.IsOK() && std::filesystem::exists(fullPath), "File \"%s\" not found", fullPath.c_str());
  return fullPath;
}

// Declared in config.h
MOCHI_NO_INLINE int mochi_benchmark::DoNotOptimizeRuntimeVar(int var) {
  return var;
}

// Declared in config.h
MOCHI_NO_INLINE void mochi_benchmark::CallNoInline(std::function<void()> const& fn) {
  fn();
}

int main(int argc, char** argv) {
  // Initialize mochi.
  mochi::Initialize();

  // Run benchmarks.
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
    return 1;
  }
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();

  // Done!
  return 0;
}
