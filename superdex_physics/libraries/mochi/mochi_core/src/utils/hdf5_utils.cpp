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

#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/hdf5_utils.h>
#include <mochi_core/utils/log.h>
#include <mochi_core/utils/span.h>

#include <cstring>
#include <filesystem>

using namespace mochi;

#if MOCHI_USE_HDF5

static std::recursive_mutex g_hdf5Mutex;

std::recursive_mutex& mochi::hdf5::GetGlobalMutex() {
  return g_hdf5Mutex;
}

void mochi::hdf5::ReportException(H5::Exception const& h5Exception, Error& error) {
  MOCHI_LOG_ERROR("HDF5 EXCEPTION: %s", h5Exception.getCDetailMsg());
  MOCHI_ERROR_SET(error, "HDF5 EXCEPTION (see log for details)");
}

H5::H5File mochi::hdf5::OpenFileForWrite(std::string_view filePath, Error& error) {
  MOCHI_ERROR_IF(filePath.empty(), error, "Invalid file path");
  MOCHI_ERROR_RETURN(error, {});

  // Create the output directory if necessary
  auto dirPath = std::filesystem::path{filePath}.parent_path();
  if (!dirPath.empty() && !std::filesystem::exists(dirPath)) {
    std::error_code ec{};
    std::filesystem::create_directories(dirPath, ec);
    MOCHI_ERROR_IF(ec, error, "Failed to create output directory");
    MOCHI_ERROR_RETURN(error, {});
  }

  std::lock_guard lock(GetGlobalMutex());

  try {
    // Use the C API because it allows us to set the ordering flags.
    hid_t fcpl = H5Pcreate(H5P_FILE_CREATE);
    H5Pset_link_creation_order(fcpl, H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED);
    hid_t fileId = H5Fcreate(std::string(filePath).c_str(), H5F_ACC_TRUNC, fcpl, H5P_DEFAULT);
    H5Pclose(fcpl);
    MOCHI_ERROR_IF(fileId < 0, error, "Failed to create HDF5 file");
    MOCHI_ERROR_RETURN(error, {});

    // Wrap in C++ object
    H5::H5File file;
    file.setId(fileId);
    H5Idec_ref(fileId); // The H5::H5File now holds the last reference

    return file;
  } catch (H5::Exception const& e) {
    ReportException(e, error);
    return {};
  }
}

H5::H5File mochi::hdf5::OpenFileForRead(std::string_view filePath, Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  MOCHI_ERROR_IF(filePath.empty(), error, "Invalid file path");
  MOCHI_ERROR_IF(!std::filesystem::exists(filePath), error, "File not found");
  MOCHI_ERROR_RETURN(error, {});

  std::lock_guard lock(GetGlobalMutex());

  try {
    return {std::string(filePath).c_str(), H5F_ACC_RDONLY};
  } catch (H5::Exception const& e) {
    ReportException(e, error);
    return {};
  }
}

H5::H5File mochi::hdf5::OpenFileBytesForRead(Span<char const> fileData, Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  std::lock_guard lock(GetGlobalMutex());

  try {
    // HDF5 calls a file in memory a "file image". The C++ wrapper for HDF5 does not expose this
    // functionality, so we go straight to the C API instead. It needs a non-const pointer, but it
    // shouldn't try to modify anything because the API defaults to read-only (and we do not set
    // flag H5LT_FILE_IMAGE_OPEN_RW).
    auto openFlags = H5LT_FILE_IMAGE_DONT_COPY | H5LT_FILE_IMAGE_DONT_RELEASE;
    hid_t fileId =
        H5LTopen_file_image(const_cast<char*>(fileData.data()), fileData.size(), openFlags);
    MOCHI_ERROR_IF(fileId < 0, error, "Not a valid HDF5 file");
    MOCHI_ERROR_RETURN(error, {});

    // Wrap in C++ object
    H5::H5File file;
    file.setId(fileId);
    H5Idec_ref(fileId); // The H5::H5File now holds the last reference

    return file;
  } catch (H5::Exception const& e) {
    ReportException(e, error);
    return {};
  }
}

H5::Group mochi::hdf5::CreateGroup(H5::Group const& parent, char const* name, Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  std::lock_guard lock(GetGlobalMutex());

  try {
    // Create a child group. This part uses the C API to set the creation order flags
    // because the HDF5 C++ API does not appear to expose this feature. Ordering is important
    // so that the reader can iterate the groups in the same order.
    hid_t plist = H5Pcreate(H5P_GROUP_CREATE);
    MOCHI_DEFER(H5Pclose(plist));
    herr_t err = H5Pset_link_creation_order(plist, H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED);
    MOCHI_ERROR_IF(err < 0, error, "Failed to set group properties");
    hid_t childId = H5Gcreate(parent.getId(), name, H5P_DEFAULT, plist, H5P_DEFAULT);
    MOCHI_ERROR_IF(childId < 0, error, "Failed to create group");
    MOCHI_ERROR_RETURN(error, {});

    // Transfer ownership of the group to an H5::Group object (C++ wrapper)
    H5::Group child;
    child.setId(childId); // Calls H5Iinc_ref internally
    H5Idec_ref(childId); // The H5::Group now holds the last reference

    return child;
  } catch (H5::Exception const& e) {
    hdf5::ReportException(e, error);
    return {};
  }
}

#endif // MOCHI_USE_HDF5

bool mochi::hdf5::LooksLikeHDF5(Span<char const> fileData) {
  constexpr char hdfHeader[] = {static_cast<char>(-119), 'H', 'D', 'F'};
  return (fileData.size() > std::size(hdfHeader)) &&
      (0 == memcmp(fileData.data(), hdfHeader, std::size(hdfHeader)));
}
