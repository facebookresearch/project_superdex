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

#include <mochi_core/mochi_platform.h>
#include <mochi_core/test/mochi_test_helpers.h>

#include <cstdlib>
#include <span>
#include <string>
#include <vector>

// End-to-end transport tests: these spawn the real superdex_mesh_cli helper (its path is
// wired in via the SUPERDEX_MESH_CLI_PATH environment variable by the test target) and exercise the
// pipe round-trip, plus the clean-error behavior when the helper is missing.

using namespace mochi;
using namespace mochi::mesh;

namespace {

void SetEnvVar(char const* name, char const* value) {
#if MOCHI_PLATFORM_WINDOWS
  _putenv_s(name, value);
#else
  setenv(name, value, /*overwrite*/ 1);
#endif
}

void UnsetEnvVar(char const* name) {
#if MOCHI_PLATFORM_WINDOWS
  _putenv_s(name, "");
#else
  unsetenv(name);
#endif
}

} // namespace

TEST(MeshCliClientTest, PingRoundTrip) {
  std::string const text = "round trip through the helper process";
  std::vector<char> const payload(text.begin(), text.end());

  std::vector<char> const response =
      mesh::InvokeMeshCli(cli::GeometryOp::Ping, payload, test::ExpectOK{});
  EXPECT_EQ(response, payload);
}

TEST(MeshCliClientTest, PingEmptyPayload) {
  std::vector<char> const response =
      mesh::InvokeMeshCli(cli::GeometryOp::Ping, {}, test::ExpectOK{});
  EXPECT_TRUE(response.empty());
}

TEST(MeshCliClientTest, MissingHelperReturnsCleanError) {
  // Override the helper path to one that does not exist; the call must fail cleanly (no crash).
  char const* const saved = std::getenv("SUPERDEX_MESH_CLI_PATH");
  std::string const savedValue = saved != nullptr ? saved : "";
  bool const hadValue = saved != nullptr;

  SetEnvVar("SUPERDEX_MESH_CLI_PATH", "/this/path/does/not/exist/superdex_mesh_cli");
  {
    std::vector<char> const response =
        mesh::InvokeMeshCli(cli::GeometryOp::Ping, {}, test::ExpectNotOK{});
    EXPECT_TRUE(response.empty());
  }

  if (hadValue) {
    SetEnvVar("SUPERDEX_MESH_CLI_PATH", savedValue.c_str());
  } else {
    UnsetEnvVar("SUPERDEX_MESH_CLI_PATH");
  }
}
