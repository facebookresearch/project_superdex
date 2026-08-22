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

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/file_utils.h>
#include <mochi_core/utils/hdf5_utils.h>

#if MOCHI_USE_HDF5

using namespace mochi;

static void H5TestWriteGroups(H5::Group& parent) {
  // Create a few groups
  [[maybe_unused]] auto g1 = mochi::hdf5::CreateGroup(parent, "one", test::ExpectOK{});
  [[maybe_unused]] auto g2 = mochi::hdf5::CreateGroup(parent, "two", test::ExpectOK{});
  [[maybe_unused]] auto g3 = mochi::hdf5::CreateGroup(parent, "three", test::ExpectOK{});
  auto child = mochi::hdf5::CreateGroup(parent, "child", test::ExpectOK{});
  [[maybe_unused]] auto grandchild =
      mochi::hdf5::CreateGroup(child, "grandchild", test::ExpectOK{});
}

static void H5TestReadGroups(H5::Group const& parent) {
  // Read the stuff written by H5TestWriteGroups
  parent.openGroup("one");
  parent.openGroup("two");
  parent.openGroup("three");
  auto child = parent.openGroup("child");
  child.openGroup("grandchild");
}

TEST(Hdf5Utils, WriteReadFile) {
  auto lock = std::lock_guard(hdf5::GetGlobalMutex());

  auto temp = CreateTempFile("hdf5_utils_test", ".h5", test::ExpectOK{});
  auto tempPath = temp.Path().string();

  // Write an H5 file
  {
    auto file = hdf5::OpenFileForWrite(tempPath.c_str(), test::ExpectOK{});
    H5TestWriteGroups(file);
  }

  // Read it back
  {
    auto file = hdf5::OpenFileForRead(tempPath.c_str(), test::ExpectOK{});
    H5TestReadGroups(file);
  }

  // Repeat with an in-memory buffer
  {
    auto fileBytes = ReadFileBytes(tempPath, test::ExpectOK{});
    auto file = hdf5::OpenFileBytesForRead(fileBytes, test::ExpectOK{});
    H5TestReadGroups(file);
  }

  // Test LooksLikeHdf5
  {
    auto fileBytes = ReadFileBytes(tempPath, test::ExpectOK{});
    EXPECT_TRUE(hdf5::LooksLikeHDF5(fileBytes));
    EXPECT_FALSE(hdf5::LooksLikeHDF5({}));
    EXPECT_FALSE(hdf5::LooksLikeHDF5("{}"));
  }
}

TEST(Hdf5Utils, OpenFileForRead) {
  auto lock = std::lock_guard(hdf5::GetGlobalMutex());

  // Open an existing file that was not necessary created via `hdf5::OpenFileForWrite`.
  auto file =
      hdf5::OpenFileForRead(test::GetAssetPath("/cube/cube_mesh.mochi.h5"), test::ExpectOK{});
  EXPECT_TRUE(file.nameExists("mesh"));
}

#endif // MOCHI_USE_HDF5
