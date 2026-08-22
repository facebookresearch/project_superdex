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

#pragma once

#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/error.h>
#include <mochi_core/utils/span.h>

#include <filesystem>
#include <string>
#include <utility>

namespace mochi {

/**
 * @brief Read the contents of a text file into a std::string (via out parameter)
 *
 * @param filePath Path of the file to read
 * @param outString Returns the file contents as a string
 * @param error Indicates success or failure
 */
void ReadFile(std::filesystem::path const& filePath, std::string& outString, Error& error);

/**
 * @brief Read the contents of a file into a DynamicArray (via out parameter)
 *
 * @param filePath Path of the file to read
 * @param outBuffer Return the file contents as a byte array.
 * @param error Indicates success or failure
 */
void ReadFile(std::filesystem::path const& filePath, DynamicArray<char>& outBuffer, Error& error);

/**
 * @brief Read the contents of a text file to return a string.
 *
 * @param filePath Path of file to read
 * @param error Indicates success or failure
 * @return File contents
 */
[[nodiscard]] std::string ReadFileString(std::filesystem::path const& filePath, Error& error);

/**
 * @brief Read the contents of a file to return a DynamicArray of bytes with custom allocator.
 *
 * @param filePath Path of file to read
 * @param allocator Allocator to use for new memory
 * @param error Indicates success or failure
 * @return File contents
 */
[[nodiscard]] DynamicArray<char>
ReadFileBytes(std::filesystem::path const& filePath, Allocator* allocator, Error& error);

/**
 * @brief Read the contents of a file to return a DynamicArray of bytes with default allocator.
 *
 * @param filePath Path of file to read
 * @param allocator Allocator to use for new memory
 * @param error Indicates success or failure
 * @return File contents
 */
[[nodiscard]] DynamicArray<char> ReadFileBytes(std::filesystem::path const& filePath, Error& error);

/**
 * @brief Write/overwrite a binary file using the provided binary data.
 * @remarks Creates the directory path if necessary
 *
 * @param filePath Path of file to write
 * @param data File contents to write
 * @param error Indicates success or failure
 */
void WriteFile(std::filesystem::path const& filePath, Span<char const> data, Error& error);

/**
 * @brief RAII object that manages a temporary directory. When the TempDirCleanup object is
 * destroyed, the directory will be deleted along with all file contents, unless you call
 * DoNotDestroy().
 *
 * @see CreateTempDirectory
 */
class TempDirCleanup final {
  MOCHI_DECLARE_NO_COPY(TempDirCleanup);

 public:
  TempDirCleanup() = default;
  TempDirCleanup(TempDirCleanup&& other) noexcept
      : _path(std::move(other._path)), _cleanup(other._cleanup) {
    other._cleanup = false;
  }
  TempDirCleanup& operator=(TempDirCleanup&&) = delete;
  explicit TempDirCleanup(std::filesystem::path path);
  ~TempDirCleanup();

  /// @brief Return the path of the temporary directory
  [[nodiscard]] std::filesystem::path const& Path() const {
    return _path;
  }

  /// @brief Prevents the temporary directory from being destroyed. The user will manage that.
  void DoNotDestroy() {
    _cleanup = false;
  }

 private:
  std::filesystem::path _path;
  bool _cleanup = false;
};

/**
 * @brief Create a new directory for temporary files. It will be automatically destroyed at the
 * end-of-scope unless you call TempDirCleanup::DoNotDestroy().
 *
 * @remarks The new directory name will include the given label. It will also include additional
 * characters required to make it unique. The location of the directory will be platform dependent.
 *
 * @param label String to include as part of the new directory name
 * @param error Indicates success or failure
 * @return RAII object for automatic cleanup. Call TempDirCleanup::Path() for the directory path.
 *
 * @see TempDirCleanup
 *
 * @code{.cpp}
 *    auto dir = CreateTempDirectory("my_stuff", error);
 *    WriteFile(dir.Path() / "file_a.txt", "File A contents", error);
 *    WriteFile(dir.Path() / "file_b.txt", "File B contents", error);
 *    // Directory + both files will be deleted at end-of-scope
 * @endcode
 */
TempDirCleanup CreateTempDirectory(std::string_view label, Error& error);

/**
 * @brief RAII object that manages a temporary file. When the TempFileCleanup object is
 * destroyed, the file will be deleted unless you call DoNotDestroy().
 *
 * @see CreateTempFile
 */
class TempFileCleanup final {
  MOCHI_DECLARE_NO_COPY(TempFileCleanup);

 public:
  TempFileCleanup() = default;
  TempFileCleanup(TempFileCleanup&& other) noexcept
      : _path(std::move(other._path)), _cleanup(other._cleanup) {
    other._cleanup = false;
  }
  TempFileCleanup& operator=(TempFileCleanup&&) = delete;
  explicit TempFileCleanup(std::filesystem::path path);
  ~TempFileCleanup();

  /// @brief Return the path of the temporary file
  [[nodiscard]] std::filesystem::path const& Path() const {
    return _path;
  }

  /// @brief Prevents the temporary file from being destroyed. The user will manage that.
  void DoNotDestroy() {
    _cleanup = false;
  }

 private:
  std::filesystem::path _path;
  bool _cleanup = false;
};

/**
 * @brief Create a new file with a unique name in a location that is suitable for temporary files.
 * It will be automatically destroyed at the end-of-scope unless you call
 * TempFileCleanup::DoNotDestroy().
 *
 * @remarks The new file name will include the given label. It will also include additional
 * characters required to make it unique. The location of the file will be platform dependent.
 *
 * @param label String to include as part of the new file name
 * @param ext File extension string, starting with a period (e.g. ".txt"), or empty string.
 * @param error Indicates success or failure
 * @return RAII object for automatic cleanup. Call TempFileCleanup::Path() for the file path.
 *
 * @see TempFileCleanup
 *
 * @code{.cpp}
 *   auto file = CreateTempFile("my_file", ".txt", error);
 *   WriteFile(file.Path(), "My file contents here", error);
 *   // File will be destroyed at end-of-scope
 * @endcode
 */
TempFileCleanup CreateTempFile(std::string_view label, std::string_view ext, Error& error);

} // namespace mochi
