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

#include "mesh_cli_client.h"

#include <mochi_mesh/mesh_cli_control.h>

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <future>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace mochi;
using namespace mochi::mesh;

namespace {

class ScopedTempDirectory {
 public:
  ScopedTempDirectory() {
    static std::atomic<unsigned int> nextSuffix{0};
    auto const timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
    for (int attempt = 0; attempt < 100; ++attempt) {
      std::filesystem::path const candidate = std::filesystem::temp_directory_path() /
          ("mochi_mesh_cli_concurrency_" + std::to_string(timestamp) + "_" +
           std::to_string(nextSuffix.fetch_add(1)));
      std::error_code ec;
      if (std::filesystem::create_directory(candidate, ec)) {
        _path = candidate;
        return;
      }
      if (ec) {
        return;
      }
    }
  }

  ~ScopedTempDirectory() {
    if (!_path.empty()) {
      std::error_code ec;
      std::filesystem::remove_all(_path, ec);
    }
  }

  std::filesystem::path const& Path() const {
    return _path;
  }

 private:
  std::filesystem::path _path;
};

bool WaitForMarker(
    std::filesystem::path const& path,
    std::chrono::steady_clock::time_point deadline) {
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
  return std::filesystem::exists(path, ec);
}

struct InvocationResult {
  bool isOk{};
  std::vector<char> response;
};

InvocationResult InvokePing(std::string const& coordinationPath, std::vector<char> payload) {
  Error error;
  std::vector<char> response =
      mesh::InvokeMeshCli(coordinationPath, cli::GeometryOp::Ping, payload, error);
  return {error.IsOK(), std::move(response)};
}

} // namespace

TEST(MeshCliClientTest, ConcurrentInvocationsReceiveStdinEof) {
  ScopedTempDirectory const coordinationDirectory;
  ASSERT_FALSE(coordinationDirectory.Path().empty());
  std::string const coordinationPath = coordinationDirectory.Path().string();
  std::future<InvocationResult> first = std::async(std::launch::async, [coordinationPath]() {
    return InvokePing(coordinationPath, std::vector<char>(1 << 20, 'a'));
  });

  auto const markerDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  if (!WaitForMarker(coordinationDirectory.Path() / "first_started", markerDeadline)) {
    CancelInFlightMeshCli();
    first.get();
    FAIL() << "The first concurrency helper did not start";
  }

  std::future<InvocationResult> second = std::async(
      std::launch::async, [coordinationPath]() { return InvokePing(coordinationPath, {}); });

  std::future_status const firstStatus = first.wait_for(std::chrono::seconds(10));
  std::future_status const secondStatus = second.wait_for(std::chrono::seconds(10));
  bool const completed =
      firstStatus == std::future_status::ready && secondStatus == std::future_status::ready;
  if (!completed) {
    CancelInFlightMeshCli();
  }

  InvocationResult const firstResult = first.get();
  InvocationResult const secondResult = second.get();
  EXPECT_TRUE(completed) << "Concurrent helpers did not receive stdin EOF";
  if (completed) {
    EXPECT_TRUE(firstResult.isOk);
    EXPECT_TRUE(secondResult.isOk);
    EXPECT_TRUE(firstResult.response.empty());
    EXPECT_TRUE(secondResult.response.empty());
  }
}
