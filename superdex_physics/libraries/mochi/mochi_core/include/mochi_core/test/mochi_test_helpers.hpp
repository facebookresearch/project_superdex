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

/**
  This file contains out-of-line definitions for functions in MochiTestHelpers.h.
  If you #include it in any ONE cpp file, it will generate the code necessary for linking.
*/
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/test/mochi_test_profile.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/nd_array_utils.h>
#include <mochi_core/utils/path.h>
#include <mochi_core/utils/span.h>

#ifdef _WIN32
#include <Windows.h> // For OutputDebugStringA
#endif

#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace mochi {
namespace test {

static void UnitTestPrintFn(
    [[maybe_unused]] char const* file,
    [[maybe_unused]] int line,
    char const* msg,
    [[maybe_unused]] bool debuggerOnly) {
#ifdef _WIN32
  // On Windows, always print to the debugger
  ::OutputDebugStringA(mochi::Format("%s(%d): %s", file, line, msg).c_str());

  // Optionally print to the console
  if (!debuggerOnly) {
    printf("%s", msg);
  }
#else
  // On other platforms, stdout is typically visible in the debugger automatically.
  printf("%s", msg);
#endif
}

static void OnMochiTestLog(mochi::LogChannel channel, char const* msg, char const* file, int line) {
  if ((channel == mochi::LogChannel::Warning) || (channel == mochi::LogChannel::Error)) {
    // Print using the default logging function (stdout and debugger)
    UnitTestPrintFn(file, line, msg, false);

    // If any test logs a warning or error, then presume that something is wrong and fail
    // the test. It is possible for an individual test case to change the logging function
    // if it needs to test code that is known to log warnings or errors.
    char const* chanName = (channel == mochi::LogChannel::Warning) ? "Warning" : "Error";
    ADD_FAILURE_AT(file, line) << "Unexpected logging to mochi::LogChannel::" << chanName;
  } else {
    // Redirect all other messages to the debugger, but not stdout (if they are different).
    // This keeps stdout clean when there are no problems.
    UnitTestPrintFn(file, line, msg, true);
  }
}

} // namespace test

test::TetMeshParams test::CreateMinimalTetMeshUnitGrid(Real3 scale, Int3 dims) {
  return UniformCubeTetMeshData(dims, scale);
}

test::TetMeshParams test::CreateMinimalTetMeshUnitCube(Real3 scale) {
  return UniformCubeTetMeshData({1, 1, 1}, scale);
}

test::TriMeshParams test::CreateMinimalTriMeshUnitCube(Real3 scale) {
  // A solid unit cube with one corner at (0,0,0)
  //
  //         2 ------- 3
  //       / |       / |
  //      /  |      /  |
  //     6 ------- 7   |
  //     |   0 ----|-- 1
  //     |  /      |  /
  //     | /       | /
  //     4 ------- 5
  //
  std::vector<Real3> const coordinates = {
      Real3{0.0_r, 0.0_r, 0.0_r}, // 0
      Real3{scale[0], 0.0_r, 0.0_r}, // 1
      Real3{0.0_r, scale[1], 0.0_r}, // 2
      Real3{scale[0], scale[1], 0.0_r}, // 3
      Real3{0.0_r, 0.0_r, scale[2]}, // 4
      Real3{scale[0], 0.0_r, scale[2]}, // 5
      Real3{0.0_r, scale[1], scale[2]}, // 6
      Real3{scale[0], scale[1], scale[2]}, // 7
  };
  std::vector<Int3> const connectivity = {
      Int3{0, 2, 1}, // back
      Int3{2, 3, 1}, // back
      Int3{1, 3, 5}, // right
      Int3{3, 7, 5}, // right
      Int3{5, 7, 4}, // front
      Int3{7, 6, 4}, // front
      Int3{4, 6, 0}, // left
      Int3{6, 2, 0}, // left
      Int3{2, 6, 3}, // top
      Int3{6, 7, 3}, // top
      Int3{0, 1, 4}, // bottom
      Int3{4, 1, 5}, // bottom
  };
  return {coordinates, connectivity};
}

test::TetMeshParams test::CreateMinimalTetMeshTwoShareFace(Real3 scale) {
  // Two tetrahedra sharing a face
  std::vector<Real3> const coordinates = {
      Real3{0.0_r, 0.0_r, 0.0_r}, // 0
      Real3{scale[0], 0.0_r, 0.0_r}, // 1
      Real3{0.0_r, scale[1], 0.0_r}, // 2
      Real3{0.0_r, 0.0_r, scale[2]}, // 3
      Real3{scale[0], 0.0_r, scale[2]}, // 4
  };
  std::vector<Int4> const connectivity = {Int4{0, 1, 2, 3}, Int4{1, 2, 3, 4}};

  return {coordinates, connectivity};
}

test::TetMeshParams test::CreateMinimalTetMeshTwoShareEdge(Real3 scale) {
  // Two tetrahedra sharing an edge
  std::vector<Real3> const coordinates = {
      Real3{0.0_r, 0.0_r, 0.0_r}, // 0
      Real3{scale[0], 0.0_r, 0.0_r}, // 1
      Real3{0.0_r, scale[1], 0.0_r}, // 2
      Real3{0.0_r, 0.0_r, scale[2]}, // 3
      Real3{scale[0], 0.0_r, scale[2]}, // 4
      Real3{scale[0], scale[1], scale[2]}, // 5
  };
  std::vector<Int4> const connectivity = {Int4{0, 1, 2, 3}, Int4{2, 1, 4, 5}};

  return {coordinates, connectivity};
}

test::TetMeshParams test::CreateMinimalTetMeshTwoShareNode(Real3 scale) {
  // Two tetrahedra sharing a node
  std::vector<Real3> const coordinates = {
      Real3{0.0_r, 0.0_r, 0.0_r}, // 0
      Real3{scale[0], 0.0_r, 0.0_r}, // 1
      Real3{0.0_r, scale[1], 0.0_r}, // 2
      Real3{0.0_r, 0.0_r, scale[2]}, // 3
      Real3{scale[0], 0.0_r, scale[2]}, // 4
      Real3{scale[0], scale[1], scale[2]}, // 5
      Real3{0.0_r, scale[1], scale[2]}, // 6
  };
  std::vector<Int4> const connectivity = {Int4{0, 1, 2, 3}, Int4{2, 6, 4, 5}};

  return {coordinates, connectivity};
}

test::TetMeshParams test::CreateMinimalTetMeshSingleTet(Real3 scale) {
  // Unit tet with one corner at (0,0,0)
  //
  //     3 _ _2
  //     | \ / \
  //     |  / \ \
  //     |/     \\
  //     0 -----1
  //
  std::vector<Real3> const coordinates = {
      Real3{0.0_r, 0.0_r, 0.0_r}, // 0
      Real3{scale[0], 0.0_r, 0.0_r}, // 1
      Real3{0.0_r, scale[1], 0.0_r}, // 2
      Real3{0.0_r, 0.0_r, scale[2]}, // 3
  };
  std::vector<Int4> const connectivity = {
      Int4{0, 1, 2, 3},
  };
  return {coordinates, connectivity};
}

test::TriMeshParams test::CreateMinimalTriMeshSingleTri(Real2 scale) {
  // Create a unit triangle in the x-y plane with one corner at (0,0,0)
  //
  //   2
  //   |\
  //   | \
  //   |  \
  //   0 - 1
  //
  std::vector<Real3> const coordinates = {
      Real3{0.0_r, 0.0_r, 0.0_r}, // 0
      Real3{scale[0], 0.0_r, 0.0_r}, // 1
      Real3{0.0_r, scale[1], 0.0_r}, // 2
  };
  std::vector<Int3> const connectivity = {
      Int3{0, 1, 2},
  };
  return {coordinates, connectivity};
}

std::string test::SerializeTetMesh(TetMeshParams const& mesh) {
  // Mochi currently uses JSON to load meshes. For unit tests, we generate one using simple
  // string formatting instead of taking a dependency on MochiCore. If this format stops
  // working, then it means there was a breaking change in Mochi's file format (intentional?).
  Span<real const> coordinates = Flatten(MakeSpan(mesh.first));
  Span<int const> connectivity = Flatten(MakeSpan(mesh.second));
  std::stringstream data;
  data << "{\n";
  data << "  \"coordinates\": [";
  for (int i = 0; i < (int)coordinates.size(); ++i) {
    data << coordinates[i];
    if (i != (int)coordinates.size() - 1) {
      data << ", ";
    }
  }
  data << "],\n";
  data << "  \"connectivity\": [";
  for (int i = 0; i < (int)connectivity.size(); ++i) {
    data << connectivity[i];
    if (i != (int)connectivity.size() - 1) {
      data << ", ";
    }
  }
  data << "]\n";
  data << "}";
  return data.str();
}

std::string test::GetAssetsDir() {
  // Find the common "assets" directory
  std::string fullPath = mochi::path::FindAssetsDirectory(ExpectOK{});
  EXPECT_TRUE(std::filesystem::is_directory(fullPath))
      << "Failed to find \"assests\" directory. Expected to find it at: " << fullPath;
  return fullPath;
}

std::string test::GetAssetPath(std::string const& relativePath) {
  std::string fullPath = test::GetAssetsDir() + relativePath;
  EXPECT_TRUE(std::filesystem::exists(fullPath)) << "File \"" << fullPath << "\" not found.";
  return fullPath;
}

std::string test::GetAssetPath(std::string_view relativePath) {
  return GetAssetPath(std::string(relativePath));
}

std::string test::GetAssetPath(char const* relativePath) {
  return GetAssetPath(std::string(relativePath));
}

void test::InitUnitTest(int& argc, char** argv) {
  // You can specify --gtest_break_on_failure on the command line to stop at a breakpoint any time a
  // test fails. This is such a useful feature, we enable it by default when a debugger is attached.
  // Opt out with argument --continue.
  if (IsDebuggerAttached() && !mochi::test::HasArgument(argc, argv, "--continue")) {
    testing::GTEST_FLAG(break_on_failure) = true;
  }

  // Fail the test if there is any unexpected warning or error logging
  SetLogCallback(&mochi::test::OnMochiTestLog);

  // Automatically profile every test case
  mochi::test::InitializeUnitTestProfiler(argc, argv);

  // Forward the command line arguments to Google Test.
  // They may modify argv and argc, which is why argc is passed by address.
  testing::InitGoogleTest(&argc, argv);
}

} // namespace mochi
