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
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/file_utils.h>

#include <filesystem>
#include <thread>
#include <unordered_set>

using namespace mochi;

TEST(FileUtils, CreateTempDirectory) {
  std::filesystem::path path1;
  std::filesystem::path path2;

  {
    // Create an empty temp directory
    auto dir = CreateTempDirectory("file_utils_test", test::ExpectOK{});
    path1 = dir.Path();
    EXPECT_STRNE("", path1.string().c_str());
    EXPECT_STRNE("", path1.filename().string().c_str());
    EXPECT_NE(std::string::npos, path1.filename().string().find("file_utils_test"));
    EXPECT_TRUE(std::filesystem::exists(path1));
    EXPECT_TRUE(std::filesystem::is_directory(path1));
  }

  // It should have been deleted at end-of-scope
  EXPECT_FALSE(std::filesystem::exists(path1));

  {
    // Create two distinct directories
    auto dir1 = CreateTempDirectory("file_utils_test", test::ExpectOK{});
    auto dir2 = CreateTempDirectory("file_utils_test", test::ExpectOK{});
    path1 = dir1.Path();
    path2 = dir2.Path();
    EXPECT_STRNE("", path1.string().c_str());
    EXPECT_STRNE("", path2.string().c_str());
    EXPECT_STRNE(path1.string().c_str(), path2.string().c_str()); // not the same path
    EXPECT_NE(std::string::npos, path1.filename().string().find("file_utils_test"));
    EXPECT_TRUE(std::filesystem::exists(path1));
    EXPECT_TRUE(std::filesystem::is_directory(path1));
    EXPECT_NE(std::string::npos, path2.filename().string().find("file_utils_test"));
    EXPECT_TRUE(std::filesystem::exists(path2));
    EXPECT_TRUE(std::filesystem::is_directory(path2));

    // Prevent automatic cleanup of dir1
    dir1.DoNotDestroy();

    // Write a file into dir2
    WriteFile(path2 / "temp.txt", "Something cool", test::ExpectOK{});
  }

  // dir1 still exists, but not dir2 (nor dir2/temp.txt)
  EXPECT_TRUE(std::filesystem::exists(path1));
  EXPECT_FALSE(std::filesystem::exists(path2));
  EXPECT_FALSE(std::filesystem::exists(path2 / "temp.txt"));
  std::filesystem::remove(path1);
  EXPECT_FALSE(std::filesystem::exists(path1));

  {
    // Create another direcotry. This time, destroy before the end of scope.
    // This is not consider to be an error.
    auto dir1 = CreateTempDirectory("file_utils_test", test::ExpectOK{});
    path1 = dir1.Path();
    EXPECT_TRUE(std::filesystem::exists(path1));
    std::filesystem::remove(path1);
    EXPECT_FALSE(std::filesystem::exists(path1));
  }
}

TEST(FileUtils, CreateTempDirectoryConcurrent) {
  int constexpr kNumThreads = 4;
  int constexpr kNumPathsPerThread = 1000;
  DynamicArray<std::thread> threads;
  DynamicArray<DynamicArray<std::string>> pathsPerThread(kNumThreads);

  // Create temp directories simulatneously from multiple threads.
  // They should each get unique names that should not interfere with each other.
  for (int iThread = 0; iThread < kNumThreads; ++iThread) {
    threads.emplace_back([&, iThread]() {
      auto& pathsOut = pathsPerThread[iThread];
      pathsOut.reserve(kNumPathsPerThread);
      for (int i = 0; i < kNumPathsPerThread; ++i) {
        auto dir = CreateTempDirectory("file_utils_test", test::ExpectOK{});
        EXPECT_TRUE(std::filesystem::is_directory(dir.Path()));
        pathsOut.emplace_back(dir.Path().string());
        dir.DoNotDestroy();
      }
      // Cleanup
      for (auto const& path : pathsOut) {
        std::filesystem::remove(path);
        EXPECT_FALSE(std::filesystem::exists(path));
      }
    });
  }

  // Wait for completion
  for (auto& t : threads) {
    t.join();
  }

  // Each thread should have created directories with unique names.
  std::unordered_set<std::string> uniquePaths;
  for (auto const& paths : pathsPerThread) {
    for (auto const& path : paths) {
      EXPECT_EQ(uniquePaths.end(), uniquePaths.find(path));
      uniquePaths.insert(path);
    }
  }
}

TEST(FileUtils, CreateTempFile) {
  std::filesystem::path path1;
  std::filesystem::path path2;

  {
    // Create an empty temp file
    auto file = CreateTempFile("file_utils_test", ".txt", test::ExpectOK{});
    path1 = file.Path();
    EXPECT_STRNE("", path1.string().c_str());
    EXPECT_STRNE("", path1.filename().string().c_str());
    EXPECT_NE(std::string::npos, path1.filename().string().find("file_utils_test"));
    EXPECT_TRUE(path1.extension() == ".txt");
    EXPECT_TRUE(std::filesystem::exists(path1));
    EXPECT_TRUE(std::filesystem::is_regular_file(path1));
  }

  // It should have been deleted at end-of-scope
  EXPECT_FALSE(std::filesystem::exists(path1));

  {
    // Create two distinct files
    auto file1 = CreateTempFile("file_utils_test", ".dat", test::ExpectOK{});
    auto file2 = CreateTempFile("file_utils_test", ".dat", test::ExpectOK{});
    path1 = file1.Path();
    path2 = file2.Path();
    EXPECT_STRNE("", path1.string().c_str());
    EXPECT_STRNE("", path2.string().c_str());
    EXPECT_STRNE(path1.string().c_str(), path2.string().c_str()); // not the same path
    EXPECT_NE(std::string::npos, path1.filename().string().find("file_utils_test"));
    EXPECT_TRUE(path1.extension() == ".dat");
    EXPECT_TRUE(std::filesystem::exists(path1));
    EXPECT_TRUE(std::filesystem::is_regular_file(path1));
    EXPECT_NE(std::string::npos, path2.filename().string().find("file_utils_test"));
    EXPECT_TRUE(path2.extension() == ".dat");
    EXPECT_TRUE(std::filesystem::exists(path2));
    EXPECT_TRUE(std::filesystem::is_regular_file(path2));

    // Prevent automatic cleanup of file1
    file1.DoNotDestroy();

    // Write content to file2
    WriteFile(path2, "Something cool", test::ExpectOK{});
  }

  // file1 still exists, but not file2
  EXPECT_TRUE(std::filesystem::exists(path1));
  EXPECT_FALSE(std::filesystem::exists(path2));
  std::filesystem::remove(path1);
  EXPECT_FALSE(std::filesystem::exists(path1));

  {
    // Create another file. This time, destroy before the end of scope.
    // This is not considered to be an error.
    auto file1 = CreateTempFile("file_utils_test", ".tmp", test::ExpectOK{});
    path1 = file1.Path();
    EXPECT_TRUE(std::filesystem::exists(path1));
    std::filesystem::remove(path1);
    EXPECT_FALSE(std::filesystem::exists(path1));
  }

  {
    // Test with empty extension
    auto file = CreateTempFile("file_utils_test", "", test::ExpectOK{});
    path1 = file.Path();
    EXPECT_TRUE(std::filesystem::exists(path1));
    EXPECT_TRUE(path1.extension().empty());
    EXPECT_NE(std::string::npos, path1.filename().string().find("file_utils_test"));
  }
}

TEST(FileUtils, CreateTempFileConcurrent) {
  int constexpr kNumThreads = 4;
  int constexpr kNumPathsPerThread = 1000;
  DynamicArray<std::thread> threads;
  DynamicArray<DynamicArray<std::string>> pathsPerThread(kNumThreads);

  // Create temp files simultaneously from multiple threads.
  // They should each get unique names that should not interfere with each other.
  for (int iThread = 0; iThread < kNumThreads; ++iThread) {
    threads.emplace_back([&, iThread]() {
      auto& pathsOut = pathsPerThread[iThread];
      pathsOut.reserve(kNumPathsPerThread);
      for (int i = 0; i < kNumPathsPerThread; ++i) {
        auto file = CreateTempFile("file_utils_test", ".tmp", test::ExpectOK{});
        EXPECT_TRUE(std::filesystem::is_regular_file(file.Path()));
        pathsOut.emplace_back(file.Path().string());
        file.DoNotDestroy();
      }
      // Cleanup
      for (auto const& path : pathsOut) {
        std::filesystem::remove(path);
        EXPECT_FALSE(std::filesystem::exists(path));
      }
    });
  }

  // Wait for completion
  for (auto& t : threads) {
    t.join();
  }

  // Each thread should have created files with unique names.
  std::unordered_set<std::string> uniquePaths;
  for (auto const& paths : pathsPerThread) {
    for (auto const& path : paths) {
      EXPECT_EQ(uniquePaths.end(), uniquePaths.find(path));
      uniquePaths.insert(path);
    }
  }
}

TEST(FileUtils, WriteFile) {
  // Test writing to a new text file
  {
    auto tempFile = CreateTempFile("write_test", ".txt", test::ExpectOK{});
    std::filesystem::path filePath = tempFile.Path();

    // Write string data
    std::string testData = "Hello, World!";
    WriteFile(filePath, testData, test::ExpectOK{});

    // Verify file exists and has correct size
    EXPECT_TRUE(std::filesystem::exists(filePath));
    EXPECT_EQ(std::filesystem::file_size(filePath), testData.size());

    // Read back and verify contents
    std::string readBack = ReadFileString(filePath, test::ExpectOK{});
    EXPECT_STREQ(readBack.c_str(), testData.c_str());
  }

  // Test overwriting existing file
  {
    auto tempFile = CreateTempFile("overwrite_test", ".dat", test::ExpectOK{});
    std::filesystem::path filePath = tempFile.Path();

    // Write initial data
    std::string initialData = "Initial content";
    WriteFile(filePath, initialData, test::ExpectOK{});
    EXPECT_EQ(std::filesystem::file_size(filePath), initialData.size());

    // Overwrite with different data
    std::string newData = "New content that is longer than the initial content";
    WriteFile(filePath, newData, test::ExpectOK{});

    // Verify overwrite worked
    EXPECT_EQ(std::filesystem::file_size(filePath), newData.size());
    std::string readBack = ReadFileString(filePath, test::ExpectOK{});
    EXPECT_STREQ(readBack.c_str(), newData.c_str());
  }

  // Test writing binary data
  {
    auto tempFile = CreateTempFile("binary_test", ".bin", test::ExpectOK{});
    std::filesystem::path filePath = tempFile.Path();

    // Create binary data with null bytes
    DynamicArray<char> binaryData = {
        char(0x00),
        char(0x01),
        char(0x02),
        char(0x03),
        char(0xFF),
        char(0xFE),
        char(0x00),
        char(0x42)};
    WriteFile(filePath, binaryData, test::ExpectOK{});

    // Verify file size
    EXPECT_EQ(std::filesystem::file_size(filePath), binaryData.size());

    // Read back as binary and verify
    DynamicArray<char> readBack = ReadFileBytes(filePath, test::ExpectOK{});
    EXPECT_EQ(readBack.size(), binaryData.size());
    for (size_t i = 0; i < binaryData.size(); ++i) {
      EXPECT_EQ(readBack[i], binaryData[i]) << "Mismatch at index " << i;
    }
  }

  // Test creating directories if necessary
  {
    auto tempDir = CreateTempDirectory("write_test_dir", test::ExpectOK{});

    // Write to nested path that doesn't exist
    std::filesystem::path nestedPath = tempDir.Path() / "subdir1" / "subdir2" / "test.txt";
    EXPECT_FALSE(std::filesystem::exists(nestedPath));
    EXPECT_FALSE(std::filesystem::exists(nestedPath.parent_path()));
    std::string testData = "Creating nested directories";
    WriteFile(nestedPath, Span<char const>(testData.data(), testData.size()), test::ExpectOK{});

    // Verify directories were created
    EXPECT_TRUE(std::filesystem::exists(nestedPath));
    EXPECT_TRUE(std::filesystem::exists(nestedPath.parent_path()));
    EXPECT_TRUE(std::filesystem::is_directory(nestedPath.parent_path()));

    // Verify contents
    std::string readBack = ReadFileString(nestedPath, test::ExpectOK{});
    EXPECT_STREQ(readBack.c_str(), testData.c_str());
  }
}

TEST(FileUtils, ReadFile) {
  {
    // Write a text file
    auto tempFile = CreateTempFile("read_file_test", ".txt", test::ExpectOK{});
    std::string testData = "Hello, World!";
    WriteFile(tempFile.Path(), testData, test::ExpectOK{});

    // Test ReadFile with string output
    std::string result;
    ReadFile(tempFile.Path(), result, test::ExpectOK{});
    EXPECT_EQ(testData.size(), testData.size());
    EXPECT_STREQ(testData.c_str(), result.c_str());

    // Test ReadFileString
    EXPECT_EQ(result, ReadFileString(tempFile.Path(), test::ExpectOK{}));

    // Test ReadFileBytes
    DynamicArray<char> bytes = ReadFileBytes(tempFile.Path(), test::ExpectOK{});
    bytes.push_back('\0'); // null terminator
    EXPECT_STREQ(testData.c_str(), bytes.data());
  }

  {
    // Write a binary file with zero values in the middle
    auto tempFile = CreateTempFile("read_file_test", ".dat", test::ExpectOK{});
    DynamicArray<char> dataWithNulls = {
        'H', 'e', 'l', 'l', 'o', char(0x00), 'W', 'o', 'r', 'l', 'd', char(0x00), 'E', 'n', 'd'};
    WriteFile(tempFile.Path(), dataWithNulls, test::ExpectOK{});

    // Test ReadFile with DynamicArray output
    DynamicArray<char> result;
    ReadFile(tempFile.Path(), result, test::ExpectOK{});
    EXPECT_EQ(dataWithNulls.size(), result.size());
    for (size_t i = 0; i < dataWithNulls.size(); ++i) {
      EXPECT_EQ(dataWithNulls[i], result[i]) << "Mismatch at index " << i;
    }

    // Test ReadFileBytes
    EXPECT_EQ(result, ReadFileBytes(tempFile.Path(), test::ExpectOK{}));

    // Now, try reading it as a string. Expect an error because the whole file can't be represented
    // as a single null-terminated string.
    std::string strResult;
    ReadFile(tempFile.Path(), strResult, test::ExpectNotOK{});
    EXPECT_STREQ("", strResult.c_str());

    // Same with ReadFileString
    EXPECT_STREQ("", ReadFileString(tempFile.Path(), test::ExpectNotOK{}).c_str());
  }
}
