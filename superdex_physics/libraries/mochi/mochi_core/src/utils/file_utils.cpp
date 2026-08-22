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

#include <mochi_core/utils/file_utils.h>

#include <cinttypes>
#include <cstdio>
#include <filesystem>
#include <fstream>

#if MOCHI_PLATFORM_WINDOWS
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

using namespace mochi;

// Open a std::ifstream for reading. Return pair(ifstream, size).
static std::pair<std::ifstream, size_t> OpenFileForReading(
    std::filesystem::path const& filePath,
    Error& error) {
  MOCHI_ERROR_IF(filePath.empty(), error, "Empty file path");
  MOCHI_ERROR_RETURN(error, {});

  // Open file at end position
  std::ifstream stream = std::ifstream(filePath, std::ios::in | std::ios::binary | std::ios::ate);
  if (!stream.is_open()) {
    // Explain the error
    if (std::filesystem::exists(filePath)) {
      MOCHI_ERROR_SET(error, "Failed to open file for reading");
    } else {
      MOCHI_ERROR_SET(error, "File not found");
    }
    return {};
  }

  // Get the file size from the current (end) stream position,
  // then seek back to the start of the file.
  size_t fileSize = static_cast<size_t>(stream.tellg());
  stream.seekg(0, std::ios::beg);

  return std::pair{std::move(stream), fileSize};
}

// Create/overwrite a file and open an ofstream for writing.
// Create the parent directory(s) if necessary.
static std::ofstream OpenFileForWriting(std::filesystem::path const& filePath, Error& error) {
  MOCHI_ERROR_IF(filePath.empty(), error, "Empty file path");
  MOCHI_ERROR_RETURN(error, {});

  std::ofstream stream(filePath, std::ios::out | std::ios::binary | std::ios::trunc);
  if (stream.is_open()) {
    return stream;
  }

  // Maybe it failed because the directory didn't exist.
  auto parentDirPath = filePath.parent_path();
  if (!parentDirPath.empty()) {
    // Create all directories in the path, if they don't already exist.
    std::error_code ec;
    std::filesystem::create_directories(parentDirPath, ec);

    // The attempt could fail if another thread or process was trying to create
    // the same directory at the same time. That's OK as long as it exists in the end.
    if (ec && !std::filesystem::exists(parentDirPath)) {
      MOCHI_ERROR_SET(error, "Output directory does not exist and could not be created");
      return {};
    }

    // Try to open the file again
    stream = std::ofstream(filePath, std::ios::out | std::ios::binary | std::ios::trunc);
    if (stream.is_open()) {
      return stream;
    }
  }

  MOCHI_ERROR_SET(error, "Failed to open file for writing");
  return {};
}

void mochi::ReadFile(std::filesystem::path const& filePath, std::string& outString, Error& error) {
  auto&& [stream, fileSize] = OpenFileForReading(filePath, error);
  MOCHI_ERROR_RETURN(error);

  // Load text file into string
  outString.resize(fileSize);
  if (!stream.read(outString.data(), fileSize)) {
    MOCHI_ERROR_SET(error, "Failed to read file into memory");
    outString.clear();
    return;
  }

  // Make sure it was just text
  if (std::strlen(outString.c_str()) != fileSize) {
    MOCHI_ERROR_SET(error, "Attempting to read a file as if it were text. Early zero byte found.");
    outString.clear();
    return;
  }
}

void mochi::ReadFile(
    std::filesystem::path const& filePath,
    DynamicArray<char>& outBuffer,
    Error& error) {
  auto&& [stream, fileSize] = OpenFileForReading(filePath, error);
  MOCHI_ERROR_RETURN(error);

  // Load binary file into DynamicArray
  outBuffer.resize(fileSize);
  if (!stream.read(outBuffer.data(), fileSize)) {
    MOCHI_ERROR_SET(error, "Failed to read file into memory");
    outBuffer.clear();
  }
}

std::string mochi::ReadFileString(std::filesystem::path const& filePath, Error& error) {
  std::string out;
  ReadFile(filePath, out, error);
  return out;
}

DynamicArray<char>
mochi::ReadFileBytes(std::filesystem::path const& filePath, Allocator* allocator, Error& error) {
  DynamicArray<char> out(allocator);
  ReadFile(filePath, out, error);
  return out;
}

DynamicArray<char> mochi::ReadFileBytes(std::filesystem::path const& filePath, Error& error) {
  return ReadFileBytes(filePath, GetDefaultAllocator(), error);
}

void mochi::WriteFile(std::filesystem::path const& filePath, Span<char const> data, Error& error) {
  MOCHI_ERROR_IF(filePath.empty(), error, "Empty file path");
  MOCHI_ERROR_RETURN(error);
  auto stream = OpenFileForWriting(filePath, error);
  MOCHI_ERROR_RETURN(error);

  // Write the data
  MOCHI_ERROR_IF(!stream.write(data.data(), data.size()), error, "Failed to write data to file.");
  MOCHI_ERROR_RETURN(error);

  stream.flush();
  MOCHI_ERROR_IF(!stream, error, "Failed to flush data to file.");
  MOCHI_ERROR_RETURN(error);

  stream.close();
  MOCHI_ERROR_IF(!stream, error, "Failed to close file after writing.");
  MOCHI_ERROR_RETURN(error);
}

static std::filesystem::path GetPlatformTempDir() {
  auto path = std::filesystem::temp_directory_path();
  MOCHI_ASSERT_VERBOSE(!path.empty());
  MOCHI_ASSERT_VERBOSE(std::filesystem::is_directory(path));
  return path;
}

static std::string
GenerateNameWithTimestamp(std::string_view label, uint32_t retry, std::string_view suffix) {
  auto timeSinceEpoch = std::chrono::system_clock::now().time_since_epoch();
  uint64_t nanoseconds =
      std::chrono::duration_cast<std::chrono::nanoseconds>(timeSinceEpoch).count();
  return Format(
      "mochi_%s_%016" PRIx64 "%02x%s",
      std::string{label}.c_str(),
      nanoseconds,
      retry,
      std::string{suffix}.c_str());
}

// Create the specified file but DO NOT overwrite an existing file. Return true on success.
// If called concurrently from multiple threads or processes, only one of them can succeed.
static bool TryCreateFileExclusive(std::filesystem::path const& path) {
#if MOCHI_PLATFORM_WINDOWS
  HANDLE hFile = CreateFileW(
      path.c_str(), // This is a wchar_t string on Windows, unlike other platforms.
      GENERIC_WRITE,
      0, // No sharing
      NULL,
      CREATE_NEW, // Fail if exists
      FILE_ATTRIBUTE_NORMAL,
      NULL);
  if (hFile == INVALID_HANDLE_VALUE) {
    return false;
  }
  CloseHandle(hFile);
  return true;
#else
  int fd = open(path.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0644);
  if (fd == -1) {
    return false;
  }
  close(fd);
  return true;
#endif
}

// Create the specified directory but DO NOT create it if it already exists. Return true on success.
// If called concurrently from multiple threads or processes, only one of them can succeed.
static bool TryCreateDirectoryExclusive(std::filesystem::path const& path) {
#if MOCHI_PLATFORM_WINDOWS
  return CreateDirectoryW(
             path.c_str(), // This is a wchar_t string on Windows, unlike other platforms
             NULL) != 0;
#else
  return mkdir(path.c_str(), 0755) != -1;
#endif
}

TempDirCleanup::TempDirCleanup(std::filesystem::path path) : _path(std::move(path)) {
  _cleanup = !_path.empty() && std::filesystem::is_directory(_path);
}

TempDirCleanup::~TempDirCleanup() {
  if (_cleanup) {
    MOCHI_ASSERT(!_path.empty());
    try {
      std::error_code ec;
      std::filesystem::remove_all(_path, ec);
    } catch (...) {
    }
  }
}

TempDirCleanup mochi::CreateTempDirectory(std::string_view label, Error& error) {
  MOCHI_ERROR_IF(
      label.empty(),
      error,
      "Temp directory label is empty. Please provide a descriptive string to include in the directory name.");
  MOCHI_ERROR_RETURN(error, {});

  std::string name;
  std::filesystem::path path;
  std::filesystem::path root = GetPlatformTempDir();
  uint32_t retry = 0;
  constexpr uint32_t kMaxRetries = 100;
  for (;;) {
    name = GenerateNameWithTimestamp(label, retry, "");
    path = root / name;
    if (!std::filesystem::exists(path) && TryCreateDirectoryExclusive(path)) {
      // Success
      return TempDirCleanup{path};
    }
    ++retry;
    if (retry == kMaxRetries) {
      break;
    }
  }
  MOCHI_ERROR_SET(error, "Failed to create a unique temp directory");
  return {};
}

TempFileCleanup::TempFileCleanup(std::filesystem::path path) : _path(std::move(path)) {
  _cleanup = !_path.empty() && std::filesystem::is_regular_file(_path);
}

TempFileCleanup::~TempFileCleanup() {
  if (_cleanup) {
    MOCHI_ASSERT(!_path.empty());
    try {
      std::error_code ec;
      std::filesystem::remove(_path, ec);
    } catch (...) {
    }
  }
}

TempFileCleanup mochi::CreateTempFile(std::string_view label, std::string_view ext, Error& error) {
  MOCHI_ERROR_IF(
      label.empty(),
      error,
      "Temp file label is empty. Please provide a descriptive string to include in the file name.");
  MOCHI_ERROR_IF(
      !ext.empty() && !ext.starts_with('.'),
      error,
      "File extension should start with a period, or be empty.");
  MOCHI_ERROR_RETURN(error, {});

  std::string name;
  std::filesystem::path path;
  std::filesystem::path root = GetPlatformTempDir();
  uint32_t retry = 0;
  constexpr uint32_t kMaxRetries = 100;
  for (;;) {
    name = GenerateNameWithTimestamp(label, retry, ext);
    path = root / name;
    if (!std::filesystem::exists(path) && TryCreateFileExclusive(path)) {
      // Success
      return TempFileCleanup{path};
    }
    ++retry;
    if (retry == kMaxRetries) {
      break;
    }
  }
  MOCHI_ERROR_SET(error, "Failed to create a unique temp file");
  return {};
}
