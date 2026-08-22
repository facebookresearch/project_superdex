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

#include <mochi_core/memory/allocator.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/error.h>
#include <mochi_core/utils/span.h>

#include <array>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace mochi {

// Forward declarations
class GroupWriter;
class GroupReader;

/**********************************************************************************************
  Helper class to ensure that ExitGroup is always called after each successful EnterGroup.
  This template can be used by both GroupWriter and GroupReader.
*/
template <typename T>
struct GroupScopeGuard final {
  ~GroupScopeGuard() {
    if (_group) {
      _group->ExitGroup(ErrorAssert{});
    }
  }
  GroupScopeGuard() = default;
  explicit GroupScopeGuard(T* group) : _group(group) {}
  GroupScopeGuard(GroupScopeGuard&& rhs) noexcept : _group(rhs._group) {
    rhs._group = nullptr;
  }

  // Move-assignment is deleted because no implementation can be correct for stack-based group
  // backends (e.g. HDF5) when both guards are non-empty and the RHS comes from EnterGroup():
  //   - C++ evaluates the RHS (which pushes a new frame) BEFORE operator= runs.
  //   - ExitGroup() always pops the top of the stack, so calling it here pops the newly pushed
  //     frame, silently redirecting subsequent writes to the outgoing group.
  //   - Conversely, omitting ExitGroup() leaks the outgoing stack frame until the writer/reader is
  //     destroyed.
  // Callers must not replace a live guard with EnterGroup(...) in the same full-expression.
  // Destroy/reset the old guard in a separate statement, or use lexical scoping.
  GroupScopeGuard& operator=(GroupScopeGuard&&) = delete;

  operator bool() const {
    return _group != nullptr;
  }
  T* _group = nullptr;
};

/**********************************************************************************************
  GroupWriter is an abstract interface used to write data that is organized into a hierarchy of
  groups (like folders) with attributes (variables) and datasets (usually ND arrays of values).
  The design allows for data to be written to disk as it comes in (no need for large structures
  in memory). Can be used to write to HDF5 or other formats.
*/
class GroupWriter {
 public:
  // Type alias for the ScopeGuard
  using ScopeGuard = GroupScopeGuard<GroupWriter>;

  virtual ~GroupWriter() = default;

  // ZLib Compression Levels (0 - 9):
  // 0 = No compression
  // 1 = Minimum compression, fastest
  // 6 = Default compression, balance of size vs speed).
  // 9 = Maximum compression, slowest
  static constexpr int kNoCompression = 0;
  static constexpr int kMinCompression = 1;
  static constexpr int kDefaultCompression = 6;
  static constexpr int kMaxCompression = 9;

  // Set the compression level for datasets, if supported by the implementation.
  virtual void SetCompression(int level) = 0;

  // Create and enter a group (like a folder). ExitGroup() will be called automatically when the
  // ScopeGuard is destroyed.
  [[nodiscard]] virtual ScopeGuard EnterGroup(std::string_view name, Error& error) = 0;

  // Add an N-dimensional dataset within the current group.
  // Feel free to add overloads for additional data types.
  virtual void AddDataSet(
      std::string_view name,
      Span<double const> data,
      Span<size_t const> dims,
      Error& error) = 0;

  virtual void AddDataSet(
      std::string_view name,
      Span<float const> data,
      Span<size_t const> dims,
      Error& error) = 0;

  virtual void AddDataSet(
      std::string_view name,
      Span<int const> data,
      Span<size_t const> dims,
      Error& error) = 0;

  virtual void AddDataSet(
      std::string_view name,
      Span<uint8_t const> data,
      Span<size_t const> dims,
      Error& error) = 0;

  virtual void AddDataSet(
      std::string_view name,
      Span<std::string const> data,
      Span<size_t const> dims,
      Error& error) = 0;

  // Add a 1-dimensional dataset within the current group.
  template <typename T>
  void AddDataSet(std::string_view name, Span<T const> data, Error& error) {
    size_t dims[1] = {data.size()};
    AddDataSet(name, data, dims, error);
  }

  // An attribute always describes that which came BEFORE it.
  // Call AddAttribute AFTER EnterGroup to add attributes to the group.
  // Call AddAttribute AFTER AddDataSet to add attributes to the data set.
  // clang-format off
  virtual void AddAttribute(std::string_view name, char const* data, size_t count, Error& error) = 0;
  virtual void AddAttribute(std::string_view name, int32_t const* data, size_t count, Error& error) = 0;
  virtual void AddAttribute(std::string_view name, uint64_t const* data, size_t count, Error& error) = 0;
  virtual void AddAttribute(std::string_view name, float const* data, size_t count, Error& error) = 0;
  virtual void AddAttribute(std::string_view name, double const* data, size_t count, Error& error) = 0;
  // clang-format on

  // Syntax sugar for single values, strings, and containers
  template <typename T>
  void AddAttribute(std::string_view name, T const& data, Error& error) {
    if constexpr (std::is_arithmetic_v<T>) {
      // Single number becomes an array of 1
      AddAttribute(name, &data, 1, error);
    } else if constexpr (std::is_convertible_v<T, std::string_view>) {
      // std::string, std::string_view, or char const*
      std::string_view sv{data};
      AddAttribute(name, sv.data(), sv.size(), error);
    } else {
      // All other supported types have data() and size() members.
      // Examples include mochi::Span, mochi::NdArray, std::array, std::vector.
      AddAttribute(name, data.data(), data.size(), error);
    }
  }

 protected:
  // Don't call ExitGroup directly. The ScopeGuard will call it automatically at the end of the C++
  // scope.
  friend struct GroupScopeGuard<GroupWriter>;
  virtual void ExitGroup(Error& error) = 0;
};

/**********************************************************************************************
  GroupReader is an abstract interface used to read data that is organized into a hierarchy of
  groups (like folders) with attributes (variables) and datasets (usually ND arrays of values).
  The design allows for data to be read from disk efficiently. Can be used to read from HDF5
  or other formats.
*/
class GroupReader {
 public:
  // Type alias for the ScopeGuard
  using ScopeGuard = GroupScopeGuard<GroupReader>;

  virtual ~GroupReader() = default;

  // Return true if the child group name exists within the current group.
  [[nodiscard]] virtual bool HasGroup(std::string_view name) const = 0;

  // Return true if the DataSet exists in the current group.
  [[nodiscard]] virtual bool HasDataSet(std::string_view name) const = 0;

  // Return true if the current group or last read dataset has the specified attribute.
  [[nodiscard]] virtual bool HasAttribute(std::string_view name) const = 0;

  // Enter a group (like a folder). ExitGroup() will be called automatically when the
  // ScopeGuard is destroyed.
  [[nodiscard]] virtual ScopeGuard EnterGroup(std::string_view name, Error& error) = 0;

  // Query information about the current group. Items are returned in the order they were written.
  virtual DynamicArray<std::string> GetGroupNames(Error& error) const = 0;
  virtual DynamicArray<std::string> GetDataSetNames(Error& error) const = 0;
  virtual DynamicArray<std::string> GetAttributeNames(Error& error) const = 0;

  // Get dataset dimensions
  virtual DynamicArray<size_t> GetDataSetDimensions(std::string_view name, Error& error) const = 0;

  // Read N-dimensional datasets within the current group.
  // Call GetDataSetDimensions() first to determine the required data size.
  virtual void ReadDataSet(std::string_view name, Span<double> outData, Error& error) = 0;
  virtual void ReadDataSet(std::string_view name, Span<float> outData, Error& error) = 0;
  virtual void ReadDataSet(std::string_view name, Span<int> outData, Error& error) = 0;
  virtual void ReadDataSet(std::string_view name, Span<uint8_t> outData, Error& error) = 0;
  virtual void ReadDataSet(std::string_view name, Span<std::string> outData, Error& error) = 0;

  // Read an N-dimensional dataset from the current group, where the number of dimensions is known
  // up front, and the output is written to a flat DynamicArray. "outDims" determines the expected
  // number of dimensions and returns the size of each dimension.
  template <typename T>
  void
  ReadDataSet(std::string_view name, DynamicArray<T>& outData, Span<size_t> outDims, Error& error) {
    MOCHI_ERROR_RETURN(error);
    auto dims = GetDataSetDimensions(name, error);
    MOCHI_ERROR_IF(
        isize(dims) != isize(outDims), error, "Dataset has the wrong number of dimensions");
    MOCHI_ERROR_RETURN(error);
    size_t totalSize = 1;
    for (int i = 0; i < isize(dims); ++i) {
      totalSize *= dims[i];
      outDims[i] = dims[i];
    }
    if constexpr (std::is_trivially_copyable_v<T>) {
      outData.resize_noinit(totalSize);
    } else {
      outData.resize(totalSize);
    }
    ReadDataSet(name, MakeSpan(outData), error);
  }

  template <typename T>
  void ReadDataSet(std::string_view name, DynamicArray<T>& outData, Error& error) {
    size_t dims[1] = {};
    ReadDataSet(name, outData, dims, error);
  }

  // Read attributes from the current group or the last read dataset
  // Call GetAttributeSize() first to determine the required data size.
  // clang-format off
  virtual void ReadAttribute(std::string_view name, Span<char> outData, Error& error) const = 0;
  virtual void ReadAttribute(std::string_view name, Span<int32_t> outData, Error& error) const = 0;
  virtual void ReadAttribute(std::string_view name, Span<uint64_t> outData, Error& error) const = 0;
  virtual void ReadAttribute(std::string_view name, Span<float> outData, Error& error) const = 0;
  virtual void ReadAttribute(std::string_view name, Span<double> outData, Error& error) const = 0;
  // clang-format on

  // Syntax sugar for single values and strings
  template <typename T>
  void ReadAttribute(std::string_view name, T& outData, Error& error) const {
    if constexpr (std::is_arithmetic_v<T>) {
      // Single number - read as array of 1 and extract
      size_t size = GetAttributeSize(name, error);
      MOCHI_ERROR_RETURN(error);
      MOCHI_ERROR_IF(size != 1, error, "Attribute is not a single value");
      MOCHI_ERROR_RETURN(error);
      ReadAttribute(name, MakeSingletonSpan(outData), error);
      MOCHI_ERROR_RETURN(error);
    } else if constexpr (std::is_same_v<T, std::string>) {
      // String - read as char array and convert
      size_t size = GetStringAttributeSize(name, error);
      MOCHI_ERROR_RETURN(error);
      outData.resize(size);
      ReadAttribute(name, MakeSpan(outData), error);
      MOCHI_ERROR_RETURN(error);
      // Remove null terminator if present
      if (!outData.empty() && outData.back() == '\0') {
        outData.pop_back();
      }
    } else {
      // Container types - delegate to the array version
      ReadAttribute(name, MakeSpan(outData), error);
    }
  }

  // Get the allocator used by this GroupReader
  virtual Allocator* GetAllocator() const = 0;

  // Get attribute size (number of elements)
  virtual size_t GetAttributeSize(std::string_view name, Error& error) const = 0;

 protected:
  // Get string attribute size (number of characters)
  virtual size_t GetStringAttributeSize(std::string_view name, Error& error) const = 0;

  // Don't call ExitGroup directly. The ScopeGuard will call it automatically at the end of the C++
  // scope.
  friend struct GroupScopeGuard<GroupReader>;
  virtual void ExitGroup(Error& error) = 0;
};

/**********************************************************************************************
  HDF5 Implementation.
*/

// Open an H5 file for writing. Creates the directories if necessary. Clobbers existing.
std::unique_ptr<GroupWriter> CreateGroupWriterHDF5(std::string_view filePath, Error& error);

// Open an H5 file for reading.
std::unique_ptr<GroupReader> CreateGroupReaderHDF5(
    std::string_view filePath,
    Error& error,
    Allocator* allocator = GetDefaultAllocator());

// Open an H5 file for reading, where the file has already been loaded into memory.
std::unique_ptr<GroupReader> CreateGroupReaderFromBytesHDF5(
    Span<char const> fileData,
    Error& error,
    Allocator* allocator = GetDefaultAllocator());

} // namespace mochi
