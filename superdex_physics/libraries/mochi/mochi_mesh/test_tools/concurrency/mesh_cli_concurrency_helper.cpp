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

#include <mochi_mesh/mochi_mesh_cli_encoding.h>

#include <mochi_core/mochi_platform.h>

#include <array>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <span>
#include <thread>
#include <vector>

#if MOCHI_PLATFORM_WINDOWS
#include <fcntl.h>
#include <io.h>
#endif

namespace {

bool CreateMarker(std::filesystem::path const& path) {
  std::ofstream marker(path);
  return marker.good();
}

bool WaitForMarker(std::filesystem::path const& path) {
  auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  std::error_code ec;
  while (std::chrono::steady_clock::now() < deadline) {
    if (std::filesystem::exists(path, ec)) {
      return true;
    }
    ec.clear();
    // The marker is written by another process, so there is no in-process notifier to await.
    // NOLINTNEXTLINE(facebook-hte-BadCall-sleep_for)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return false;
}

bool ReadStdinToEof() {
  std::array<char, 4096> buffer{};
  while (std::fread(buffer.data(), 1, buffer.size(), stdin) > 0) {
  }
  return std::feof(stdin) != 0 && std::ferror(stdin) == 0;
}

bool WriteResponse() {
  std::vector<char> const response =
      mochi::mesh::cli::EncodeResponseFrame(0, std::span<char const>{});
  return std::fwrite(response.data(), 1, response.size(), stdout) == response.size() &&
      std::fflush(stdout) == 0;
}

} // namespace

int main(int argc, char** argv) {
#if MOCHI_PLATFORM_WINDOWS
  if (_setmode(_fileno(stdin), _O_BINARY) == -1 || _setmode(_fileno(stdout), _O_BINARY) == -1) {
    return 1;
  }
#endif

  if (argc != 2) {
    return 2;
  }

  std::filesystem::path const directory(argv[1]);
  std::filesystem::path const firstStarted = directory / "first_started";
  std::filesystem::path const firstClaim = directory / "first_claim";
  std::filesystem::path const secondStarted = directory / "second_started";
  std::filesystem::path const firstReachedEof = directory / "first_reached_eof";

  std::error_code ec;
  bool const isFirst = std::filesystem::create_directory(firstClaim, ec);
  if (ec) {
    return 6;
  }
  if (isFirst) {
    if (!CreateMarker(firstStarted) || !WaitForMarker(secondStarted) || !ReadStdinToEof() ||
        !CreateMarker(firstReachedEof)) {
      return 3;
    }
  } else {
    if (!CreateMarker(secondStarted) || !WaitForMarker(firstReachedEof) || !ReadStdinToEof()) {
      return 4;
    }
  }

  return WriteResponse() ? 0 : 5;
}
