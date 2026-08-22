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

#include <mochi_core/memory/filo_allocator.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/group_rw.h>
#include <mochi_core/utils/hdf5_utils.h>

#include <algorithm>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace mochi;

#if MOCHI_USE_HDF5

namespace {

/**********************************************************************************************
  Common HDF5 Base Class
*/

struct H5Frame {
  H5Frame() = default;
  H5Frame(H5::Group g) : group(g) {}

  H5::Group group;
  std::optional<H5::DataSet> attributeTarget;
};

// Common base class for HDF5 operations
class GroupH5Base {
 protected:
  GroupH5Base(H5::H5File&& file) : _file(file) {
    // Add a dummy frame for the root. It may hold a DataSet reference in the future.
    _stack.emplace_back();
  }

  ~GroupH5Base() {
    std::lock_guard lock{hdf5::GetGlobalMutex()};
    _file.close();
  }

  // Get the current group or the root
  H5::Group const& GetCurrentGroup() const {
    if (_stack.size() > 1) {
      return _stack.back().group;
    } else {
      // The first frame is actually the file's root. H5File derives from H5::Group,
      // but we can't store it in the first frame of the stack (which holds actual groups, not
      // pointers or references).
      return _file;
    }
  }

  // Get the object that should currently receive/provide attributes
  H5::H5Object const& GetAttributeTarget() const {
    if (_stack.back().attributeTarget.has_value()) {
      return *_stack.back().attributeTarget; // Use this dataset
    } else {
      return GetCurrentGroup(); // Use the current group
    }
  }

  void ExitGroupImpl(Error& error) {
    MOCHI_ERROR_IF(_stack.size() == 1, error, "ExitGroup called too many times.");
    MOCHI_ERROR_RETURN(error);
    std::lock_guard lock{hdf5::GetGlobalMutex()};

    try {
      _stack.pop_back();
    } catch (H5::Exception const& e) {
      hdf5::ReportException(e, error);
    }
  }

  // Protected Data:
  H5::H5File _file;
  std::vector<H5Frame> _stack;
};

/**********************************************************************************************
  GroupWriter HDF5 Implementation
*/

class GroupWriterH5 final : public GroupWriter, private GroupH5Base {
 public:
  explicit GroupWriterH5(H5::H5File&& file) : GroupH5Base(std::move(file)) {}

  // GroupWriter API:
  void SetCompression(int level) override {
    _compressionLevel = Clamp(level, kNoCompression, kMaxCompression);
  }
  ScopeGuard EnterGroup(std::string_view name, Error& error) override;
  void ExitGroup(Error& error) override {
    ExitGroupImpl(error);
  }
  void AddDataSet(
      std::string_view name,
      Span<double const> data,
      Span<size_t const> dims,
      Error& error) override {
    AddDataSetImpl(
        H5::PredType::NATIVE_DOUBLE, name, data.data(), data.size() * sizeof(data[0]), dims, error);
  }
  void AddDataSet(
      std::string_view name,
      Span<float const> data,
      Span<size_t const> dims,
      Error& error) override {
    AddDataSetImpl(
        H5::PredType::NATIVE_FLOAT, name, data.data(), data.size() * sizeof(data[0]), dims, error);
  }
  void AddDataSet(
      std::string_view name,
      Span<int const> data,
      Span<size_t const> dims,
      Error& error) override {
    AddDataSetImpl(
        H5::PredType::NATIVE_INT32, name, data.data(), data.size() * sizeof(data[0]), dims, error);
  }
  void AddDataSet(
      std::string_view name,
      Span<uint8_t const> data,
      Span<size_t const> dims,
      Error& error) override {
    AddDataSetImpl(
        H5::PredType::NATIVE_UINT8, name, data.data(), data.size() * sizeof(data[0]), dims, error);
  }

  void AddDataSet(
      std::string_view name,
      Span<std::string const> data,
      Span<size_t const> dims,
      Error& error) override {
    size_t maxStrLen = 0;
    for (auto const& d : data) {
      maxStrLen = std::max(maxStrLen, d.size());
    }
    maxStrLen += 1;
    std::vector<char> buffer(data.size() * maxStrLen, '\0');
    for (size_t i = 0; i < data.size(); ++i) {
      std::copy_n(data[i].begin(), data[i].size(), &buffer[i * maxStrLen]);
    }
    AddDataSetImpl(
        H5::StrType(H5::PredType::C_S1, maxStrLen),
        name,
        buffer.data(),
        buffer.size() * sizeof(char),
        dims,
        error);
  }

  void AddAttribute(std::string_view name, char const* data, size_t len, Error& error) override;

  void AddAttribute(std::string_view name, int32_t const* data, size_t count, Error& error)
      override {
    AddAttributeImpl(H5::PredType::NATIVE_INT32, name, data, count, error);
  }

  void AddAttribute(std::string_view name, uint64_t const* data, size_t count, Error& error)
      override {
    AddAttributeImpl(H5::PredType::NATIVE_UINT64, name, data, count, error);
  }

  void AddAttribute(std::string_view name, float const* data, size_t count, Error& error) override {
    AddAttributeImpl(H5::PredType::NATIVE_FLOAT, name, data, count, error);
  }

  void AddAttribute(std::string_view name, double const* data, size_t count, Error& error)
      override {
    AddAttributeImpl(H5::PredType::NATIVE_DOUBLE, name, data, count, error);
  }

 private:
  void AddAttributeImpl(
      H5::DataType const& htype,
      std::string_view name,
      void const* data,
      size_t count,
      Error& error);

  void AddDataSetImpl(
      H5::DataType const& htype,
      std::string_view name,
      void const* data,
      size_t dataSizeInBytes,
      Span<size_t const> dims,
      Error& error);

  int _compressionLevel = kDefaultCompression;
};

/**********************************************************************************************
  GroupReader HDF5 Implementation
*/

class GroupReaderH5 final : public GroupReader, private GroupH5Base {
 public:
  explicit GroupReaderH5(H5::H5File&& file, Allocator* allocator)
      : GroupH5Base(std::move(file)), _allocator(allocator ? allocator : GetDefaultAllocator()) {}

  // GroupReader API:

  bool HasGroup(std::string_view name) const override {
    std::lock_guard lock{hdf5::GetGlobalMutex()};
    auto const& group = GetCurrentGroup();
    std::string nameStr(name);
    return group.nameExists(nameStr) && group.childObjType(nameStr) == H5O_TYPE_GROUP;
  }

  bool HasDataSet(std::string_view name) const override {
    std::lock_guard lock{hdf5::GetGlobalMutex()};
    auto const& group = GetCurrentGroup();
    std::string nameStr(name);
    return group.nameExists(nameStr) && group.childObjType(nameStr) == H5O_TYPE_DATASET;
  }

  bool HasAttribute(std::string_view name) const override {
    std::lock_guard lock{hdf5::GetGlobalMutex()};
    auto const& obj = GetAttributeTarget();
    return obj.attrExists(std::string(name));
  }

  ScopeGuard EnterGroup(std::string_view name, Error& error) override;
  void ExitGroup(Error& error) override {
    ExitGroupImpl(error);
  }

  DynamicArray<std::string> GetGroupNames(Error& error) const override {
    return GetObjectNames(H5G_GROUP, error);
  }

  DynamicArray<std::string> GetDataSetNames(Error& error) const override {
    return GetObjectNames(H5G_DATASET, error);
  }

  DynamicArray<std::string> GetAttributeNames(Error& error) const override;

  DynamicArray<size_t> GetDataSetDimensions(std::string_view name, Error& error) const override;

  void ReadDataSet(std::string_view name, Span<double> outData, Error& error) override {
    ReadDataSetImpl(H5::PredType::NATIVE_DOUBLE, name, outData, error);
  }

  void ReadDataSet(std::string_view name, Span<float> outData, Error& error) override {
    ReadDataSetImpl(H5::PredType::NATIVE_FLOAT, name, outData, error);
  }

  void ReadDataSet(std::string_view name, Span<int> outData, Error& error) override {
    ReadDataSetImpl(H5::PredType::NATIVE_INT32, name, outData, error);
  }

  void ReadDataSet(std::string_view name, Span<uint8_t> outData, Error& error) override {
    ReadDataSetImpl(H5::PredType::NATIVE_UINT8, name, outData, error);
  }

  void ReadDataSet(std::string_view name, Span<std::string> outData, Error& error) override;

  void ReadAttribute(std::string_view name, Span<char> outData, Error& error) const override;

  void ReadAttribute(std::string_view name, Span<int32_t> outData, Error& error) const override {
    ReadAttributeImpl(H5::PredType::NATIVE_INT32, name, outData, error);
  }

  void ReadAttribute(std::string_view name, Span<uint64_t> outData, Error& error) const override {
    ReadAttributeImpl(H5::PredType::NATIVE_UINT64, name, outData, error);
  }

  void ReadAttribute(std::string_view name, Span<float> outData, Error& error) const override {
    ReadAttributeImpl(H5::PredType::NATIVE_FLOAT, name, outData, error);
  }

  void ReadAttribute(std::string_view name, Span<double> outData, Error& error) const override {
    ReadAttributeImpl(H5::PredType::NATIVE_DOUBLE, name, outData, error);
  }

  size_t GetAttributeSize(std::string_view name, Error& error) const override;

  size_t GetStringAttributeSize(std::string_view name, Error& error) const override;

  Allocator* GetAllocator() const override {
    return _allocator;
  }

 private:
  template <typename T>
  void ReadAttributeImpl(
      H5::DataType const& htype,
      std::string_view name,
      Span<T> outData,
      Error& error) const;

  template <typename T>
  void
  ReadDataSetImpl(H5::DataType const& htype, std::string_view name, Span<T> outData, Error& error);

  DynamicArray<std::string> GetObjectNames(H5G_obj_t objectType, Error& error) const;

  // Private Data:
  Allocator* _allocator;
};

} // namespace

// String attribute
void GroupWriterH5::AddAttribute(
    std::string_view name,
    char const* data,
    size_t len,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  std::lock_guard lock{hdf5::GetGlobalMutex()};

  try {
    H5::H5Object const& hobject = GetAttributeTarget();
    // HDF5 fixed-length string datatypes cannot have size 0. Encode a logical
    // empty string as one NUL byte; ReadAttribute(std::string&) trims it.
    auto const htypeLen = Max(len, size_t{1});
    H5::StrType htype(0, htypeLen); // The size is part of the DataType (unlike other arrays)
    H5::DataSpace hspace(H5S_SCALAR);
    H5::Attribute hattribute = hobject.createAttribute(std::string(name), htype, hspace);
    hattribute.write(htype, len == 0 ? "" : data);
  } catch (H5::Exception const& e) {
    hdf5::ReportException(e, error);
  }
}

// Used for all numbers and number arrays
void GroupWriterH5::AddAttributeImpl(
    H5::DataType const& htype,
    std::string_view name,
    void const* data,
    size_t count,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  std::lock_guard lock{hdf5::GetGlobalMutex()};

  try {
    // What are we adding attributes to?
    H5::H5Object const& hobject = GetAttributeTarget();
    auto const hdim = static_cast<hsize_t>(count);
    H5::DataSpace hspace = (count == 1) ? H5::DataSpace(H5S_SCALAR) : H5::DataSpace(1, &hdim);
    H5::Attribute hattribute = hobject.createAttribute(std::string(name), htype, hspace);
    hattribute.write(htype, data);
  } catch (H5::Exception const& e) {
    hdf5::ReportException(e, error);
  }
}

[[nodiscard]] GroupWriter::ScopeGuard GroupWriterH5::EnterGroup(
    std::string_view name,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  std::lock_guard lock{hdf5::GetGlobalMutex()};

  try {
    // Create a child of the current group.
    H5::Group child = hdf5::CreateGroup(GetCurrentGroup(), std::string(name).c_str(), error);

    // Push it on the stack of groups. The group will close when we pop it.
    _stack.emplace_back(std::move(child));

    // Only create a ScopeGuard if everything succeeds
    return ScopeGuard{this};
  } catch (H5::Exception const& e) {
    hdf5::ReportException(e, error);
    return {};
  }
}

void GroupWriterH5::AddDataSetImpl(
    H5::DataType const& htype,
    std::string_view name,
    void const* data,
    size_t dataSizeInBytes,
    Span<size_t const> dims,
    Error& error) {
  MOCHI_ERROR_IF(dims.size() == 0, error, "Invalid dims.");
  MOCHI_ERROR_RETURN(error);
  std::lock_guard lock{hdf5::GetGlobalMutex()};

  // Check the dimensions
  size_t expectedCount = dims[0];
  for (size_t i = 1; i < dims.size(); ++i) {
    expectedCount *= dims[i];
  }
  size_t htypeSize = htype.getSize();
  MOCHI_ERROR_IF(
      dataSizeInBytes != expectedCount * htypeSize,
      error,
      "Total data size does not match the given dimensions.");
  MOCHI_ERROR_RETURN(error);

  // Convert dims to hsize_t (not technically the same as size_t)
  std::vector<hsize_t> hdims(dims.size());
  std::copy(dims.begin(), dims.end(), hdims.begin());

  // Add a dataset to the current group
  try {
    H5::Group const& group = GetCurrentGroup();
    H5::DataSpace hspace((int)hdims.size(), hdims.data());

    H5::DSetCreatPropList hprops{};
    if (_compressionLevel > 0 && expectedCount > 0) {
      // Chunking is required for compression. We choose to use a single chunk for the entire
      // dataset, which yields the best compression. However, if we ever had a huge dataset and
      // wanted to support faster partial reads of that data, then a smaller chunk size could be
      // helpful.
      hprops.setChunk(isize(hdims), hdims.data());

      // Enable shuffle filter (improves compression for numeric data)
      hprops.setShuffle();

      // Enable deflate (gzip) compression
      hprops.setDeflate(_compressionLevel);
    }

    H5::DataSet hdataset = group.createDataSet(std::string(name), htype, hspace, hprops);
    hdataset.write(data, htype);

    // Attributes can now be assigned to the dataset (instead of the group)
    _stack.back().attributeTarget.emplace(std::move(hdataset));
  } catch (H5::Exception const& e) {
    hdf5::ReportException(e, error);
  }
}

[[nodiscard]] GroupReader::ScopeGuard GroupReaderH5::EnterGroup(
    std::string_view name,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  std::lock_guard lock{hdf5::GetGlobalMutex()};

  try {
    // Get the parent group (or file)
    H5::Group const& parent = GetCurrentGroup();

    // Open the child group
    H5::Group child = parent.openGroup(std::string(name));

    // Push it on the stack of groups. The group will close when we pop it.
    _stack.emplace_back(std::move(child));

    // Only create a ScopeGuard if everything succeeds
    return ScopeGuard{this};
  } catch (H5::Exception const& e) {
    hdf5::ReportException(e, error);
    return {};
  }
}

DynamicArray<std::string> GroupReaderH5::GetObjectNames(H5G_obj_t objectType, Error& error) const {
  MOCHI_ERROR_RETURN(error, {});
  std::lock_guard lock{hdf5::GetGlobalMutex()};

  try {
    DynamicArray<std::string> names(_allocator);
    H5::Group const& group = GetCurrentGroup();
    names.reserve(group.getNumObjs());

    struct IterData {
      DynamicArray<std::string>* names = nullptr;
      H5G_obj_t objectType = {};
    };
    IterData iterData{&names, objectType};

    auto callback =
        [](hid_t groupId, char const* name, H5L_info2_t const*, void* opData) -> herr_t {
      auto* data = static_cast<IterData*>(opData);
      H5G_stat_t statbuf{};
      // An H5::Gruop can have symbolic links to other groups. We are only interested in the objects
      // that are direct members of the specified group.
      H5Gget_objinfo(groupId, name, /*followSymbolicLinks*/ false, &statbuf);
      if (statbuf.type == data->objectType) {
        data->names->push_back(name);
      }
      return 0; // Continue iteration
    };

    // Iterate in the order the groups were created, if that information was tracked. Otherwise,
    // iterate in alphabetic order, which is always available. Unfortunately, we have to look for
    // this flag in a different place if it is the root group.
    unsigned crt_order_flags = 0;
    if (H5Iget_type(group.getId()) == H5I_FILE) {
      hid_t rootGroup = H5Gopen2(group.getId(), "/", H5P_DEFAULT);
      hid_t gcpl = H5Gget_create_plist(rootGroup);
      H5Pget_link_creation_order(gcpl, &crt_order_flags);
      H5Pclose(gcpl);
      H5Gclose(rootGroup);
    } else {
      hid_t gcpl = H5Gget_create_plist(group.getId());
      H5Pget_link_creation_order(gcpl, &crt_order_flags);
      H5Pclose(gcpl);
    }
    auto iterationOrder =
        (crt_order_flags & H5P_CRT_ORDER_TRACKED) ? H5_INDEX_CRT_ORDER : H5_INDEX_NAME;

    hsize_t idx = 0;
    H5Literate2(
        group.getId(),
        iterationOrder,
        H5_ITER_INC, // Ascending order
        &idx,
        callback,
        &iterData);

    return names;
  } catch (H5::Exception const& e) {
    hdf5::ReportException(e, error);
    return {};
  }
}

DynamicArray<std::string> GroupReaderH5::GetAttributeNames(Error& error) const {
  DynamicArray<std::string> names(_allocator);

  MOCHI_ERROR_RETURN(error, names);
  std::lock_guard lock{hdf5::GetGlobalMutex()};

  try {
    H5::H5Object const& obj = GetAttributeTarget();
    int numAttrs = obj.getNumAttrs();
    names.reserve(numAttrs);

    auto callback = [](hid_t, char const* name, H5A_info_t const*, void* opData) -> herr_t {
      auto* namesOut = static_cast<DynamicArray<std::string>*>(opData);
      namesOut->push_back(name);
      return 0; // Continue iteration
    };

    hsize_t idx = 0;
    herr_t result = H5Aiterate2(
        obj.getId(),
        H5_INDEX_CRT_ORDER, // Try creation order index first
        H5_ITER_INC, // Ascending order
        &idx,
        callback,
        &names);

    // If iteration by creation order failed (e.g., the file was not written with creation order
    // tracking enabled), fall back to iterating by name.
    if (result < 0) {
      names.clear();
      idx = 0;
      H5Aiterate2(
          obj.getId(),
          H5_INDEX_NAME, // Fall back to name index
          H5_ITER_INC, // Ascending order
          &idx,
          callback,
          &names);
    }
  } catch (H5::Exception const& e) {
    hdf5::ReportException(e, error);
    return {};
  }

  return names;
}

DynamicArray<size_t> GroupReaderH5::GetDataSetDimensions(std::string_view name, Error& error)
    const {
  DynamicArray<size_t> dims(_allocator);

  MOCHI_ERROR_RETURN(error, dims);
  std::lock_guard lock{hdf5::GetGlobalMutex()};

  try {
    H5::Group const& group = GetCurrentGroup();
    H5::DataSet dataset = group.openDataSet(std::string(name));
    H5::DataSpace dataspace = dataset.getSpace();

    int rank = dataspace.getSimpleExtentNdims();
    dims.resize_noinit(rank);
    {
      DynamicArray<hsize_t> hdims(rank, _allocator);
      dataspace.getSimpleExtentDims(hdims.data());
      std::copy(hdims.begin(), hdims.end(), dims.begin());
    }
  } catch (H5::Exception const& e) {
    hdf5::ReportException(e, error);
    return {};
  }

  return dims;
}

template <typename T>
void GroupReaderH5::ReadDataSetImpl(
    H5::DataType const& htype,
    std::string_view name,
    Span<T> outData,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  std::lock_guard lock{hdf5::GetGlobalMutex()};

  try {
    H5::Group const& group = GetCurrentGroup();
    H5::DataSet dataset = group.openDataSet(std::string(name));
    H5::DataSpace dataspace = dataset.getSpace();

    // Get dimensions and calculate total size
    int rank = dataspace.getSimpleExtentNdims();
    DynamicArray<hsize_t> hdims(rank, _allocator);
    dataspace.getSimpleExtentDims(hdims.data());

    size_t totalSize = 1;
    for (int i = 0; i < rank; ++i) {
      totalSize *= static_cast<size_t>(hdims[i]);
    }

    // Check that the Span is sized correctly
    MOCHI_ERROR_IF(outData.size() != totalSize, error, "Span size does not match dataset size.");
    MOCHI_ERROR_RETURN(error);

    // Read data into the provided span
    dataset.read(outData.data(), htype);

    // Store dataset reference for potential attribute reading
    _stack.back().attributeTarget.emplace(std::move(dataset));
  } catch (H5::Exception const& e) {
    hdf5::ReportException(e, error);
  }
}

void GroupReaderH5::ReadDataSet(std::string_view name, Span<std::string> outData, Error& error) {
  MOCHI_ERROR_RETURN(error);
  std::lock_guard lock{hdf5::GetGlobalMutex()};

  try {
    H5::Group const& group = GetCurrentGroup();
    H5::DataSet dataset = group.openDataSet(std::string(name));
    H5::DataSpace dataspace = dataset.getSpace();
    H5::DataType datatype = dataset.getDataType();

    // Get dimensions and calculate total size
    int rank = dataspace.getSimpleExtentNdims();
    DynamicArray<hsize_t> hdims(rank, _allocator);
    dataspace.getSimpleExtentDims(hdims.data());

    size_t totalSize = 1;
    for (int i = 0; i < rank; ++i) {
      totalSize *= static_cast<size_t>(hdims[i]);
    }

    // Check that the Span is sized correctly
    MOCHI_ERROR_IF(outData.size() != totalSize, error, "Span size does not match dataset size.");

    // Variable-length HDF5 strings read as char* slots that must be reclaimed by HDF5.
    // This flat byte-span API only supports fixed-length strings.
    MOCHI_ERROR_IF(
        datatype.isVariableStr(), error, "Variable-length string datasets are not supported.");
    MOCHI_ERROR_RETURN(error);

    // Get string length
    size_t strLen = datatype.getSize();

    // Read raw string data
    DynamicArray<char> buffer(totalSize * strLen, _allocator);
    dataset.read(buffer.data(), datatype);

    // Convert to strings
    for (size_t i = 0; i < totalSize; ++i) {
      char const* strStart = &buffer[i * strLen];
      outData[i] = std::string(strStart, strnlen(strStart, strLen));
    }

    // Store dataset reference for potential attribute reading
    _stack.back().attributeTarget.emplace(std::move(dataset));
  } catch (H5::Exception const& e) {
    hdf5::ReportException(e, error);
  }
}

template <typename T>
void GroupReaderH5::ReadAttributeImpl(
    H5::DataType const& htype,
    std::string_view name,
    Span<T> outData,
    Error& error) const {
  MOCHI_ERROR_RETURN(error);
  std::lock_guard lock{hdf5::GetGlobalMutex()};

  try {
    H5::H5Object const& obj = GetAttributeTarget();
    H5::Attribute attr = obj.openAttribute(std::string(name));
    H5::DataSpace dataspace = attr.getSpace();

    size_t expectedSize = 0;
    if (dataspace.isSimple()) {
      // Get the number of elements
      hssize_t numElements = dataspace.getSimpleExtentNpoints();
      expectedSize = static_cast<size_t>(numElements);
    } else {
      // Scalar attribute
      expectedSize = 1;
    }

    MOCHI_ERROR_IF(
        outData.size() != expectedSize, error, "Span size does not match attribute size.");
    MOCHI_ERROR_RETURN(error);

    attr.read(htype, outData.data());
  } catch (H5::Exception const& e) {
    hdf5::ReportException(e, error);
  }
}

void GroupReaderH5::ReadAttribute(std::string_view name, Span<char> outData, Error& error) const {
  MOCHI_ERROR_RETURN(error);
  std::lock_guard lock{hdf5::GetGlobalMutex()};

  try {
    H5::H5Object const& obj = GetAttributeTarget();
    H5::Attribute attr = obj.openAttribute(std::string(name));
    H5::DataType datatype = attr.getDataType();

    if (datatype.getClass() == H5T_STRING) {
      // Variable-length HDF5 strings read as char* slots that must be reclaimed by HDF5.
      // This flat byte-span API only supports fixed-length strings.
      MOCHI_ERROR_IF(
          datatype.isVariableStr(), error, "Variable-length string attributes are not supported.");
      MOCHI_ERROR_RETURN(error);

      // String attribute
      size_t strLen = datatype.getSize();
      MOCHI_ERROR_IF(
          outData.size() != strLen, error, "Span size does not match string attribute size.");
      MOCHI_ERROR_RETURN(error);
      attr.read(datatype, outData.data());
    } else {
      MOCHI_ERROR_SET(error, "Attribute is not a string type.");
    }
  } catch (H5::Exception const& e) {
    hdf5::ReportException(e, error);
  }
}

size_t GroupReaderH5::GetAttributeSize(std::string_view name, Error& error) const {
  MOCHI_ERROR_RETURN(error, 0);
  std::lock_guard lock{hdf5::GetGlobalMutex()};

  try {
    H5::H5Object const& obj = GetAttributeTarget();
    H5::Attribute attr = obj.openAttribute(std::string(name));
    H5::DataSpace dataspace = attr.getSpace();

    if (dataspace.isSimple()) {
      return static_cast<size_t>(dataspace.getSimpleExtentNpoints());
    } else {
      return 1; // Scalar
    }
  } catch (H5::Exception const& e) {
    hdf5::ReportException(e, error);
    return 0;
  }
}

size_t GroupReaderH5::GetStringAttributeSize(std::string_view name, Error& error) const {
  MOCHI_ERROR_RETURN(error, 0);
  std::lock_guard lock{hdf5::GetGlobalMutex()};

  try {
    H5::H5Object const& obj = GetAttributeTarget();
    H5::Attribute attr = obj.openAttribute(std::string(name));
    H5::DataType datatype = attr.getDataType();

    if (datatype.getClass() == H5T_STRING) {
      // For string attributes, return the string length (number of characters)
      return datatype.getSize();
    } else {
      MOCHI_ERROR_SET(error, "Attribute is not a string type.");
      return 0;
    }
  } catch (H5::Exception const& e) {
    hdf5::ReportException(e, error);
    return 0;
  }
}

#endif // MOCHI_USE_HDF5

std::unique_ptr<GroupWriter> mochi::CreateGroupWriterHDF5(
    [[maybe_unused]] std::string_view filePath,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});

#if MOCHI_USE_HDF5
  auto file = hdf5::OpenFileForWrite(filePath, error);
  MOCHI_ERROR_RETURN(error, {});
  return std::make_unique<GroupWriterH5>(std::move(file));
#else
  MOCHI_ERROR_SET(
      error, "Please build Mochi with MOCHI_USE_HDF5=1 if you want to use the HDF5 format.");
  return {};
#endif
}

std::unique_ptr<GroupReader> mochi::CreateGroupReaderHDF5(
    [[maybe_unused]] std::string_view filePath,
    Error& error,
    [[maybe_unused]] Allocator* allocator) {
  MOCHI_ERROR_RETURN(error, {});

#if MOCHI_USE_HDF5
  auto file = hdf5::OpenFileForRead(filePath, error);
  MOCHI_ERROR_RETURN(error, {});
  return std::make_unique<GroupReaderH5>(std::move(file), allocator);
#else
  MOCHI_ERROR_SET(
      error, "Please build Mochi with MOCHI_USE_HDF5=1 if you want to use the HDF5 format.");
  return {};
#endif
}

std::unique_ptr<GroupReader> mochi::CreateGroupReaderFromBytesHDF5(
    [[maybe_unused]] Span<char const> fileData,
    Error& error,
    [[maybe_unused]] Allocator* allocator) {
  MOCHI_ERROR_RETURN(error, {});
#if MOCHI_USE_HDF5
  auto file = hdf5::OpenFileBytesForRead(fileData, error);
  MOCHI_ERROR_RETURN(error, {});
  return std::make_unique<GroupReaderH5>(std::move(file), allocator);
#else
  MOCHI_ERROR_SET(
      error, "Please build Mochi with MOCHI_USE_HDF5=1 if you want to use the HDF5 format.");
  return {};
#endif
}
