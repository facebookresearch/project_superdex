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

#include <mochi_core/mochi_config.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/error.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/reflection.h>
#include <mochi_core/utils/span.h>

#if MOCHI_USE_HDF5
MOCHI_WARNING_PUSH_IGNORE_ALL()
#include <H5Cpp.h>
#include <H5LTpublic.h>
MOCHI_WARNING_POP()
#endif // MOCHI_USE_HDF5

#include <mutex>
#include <string>
#include <string_view>
#include <type_traits>

namespace mochi::hdf5 {

#if MOCHI_USE_HDF5

// Hide HDF5 symbols when each shared library contains a private copy of mochi_core and HDF5.
#if defined(MOCHI_HDF5_LOCAL_VISIBILITY)
#define MOCHI_HDF5_LOCAL __attribute__((visibility("hidden")))
#else
#define MOCHI_HDF5_LOCAL
#endif

// Define H5T_NATIVE_REAL depending on type of mochi::real
#if MOCHI_USE_DOUBLE_PRECISION
#define MOCHI_H5T_NATIVE_REAL H5::PredType::NATIVE_DOUBLE
#else
#define MOCHI_H5T_NATIVE_REAL H5::PredType::NATIVE_FLOAT
#endif
#define MOCHI_H5T_NATIVE_INT H5::PredType::NATIVE_INT

/**
 * @brief Get the global mutex which is used to prevent concurrent use of the HDF5 library's API.
 *
 * @return std::recursive_mutex&
 *
 * @warning It is illegal to pass an HDF5 object across a dynamic library boundary because of the
 * fact that HDF5 library is statically link.
 *
 * @warning If HDF5 was built as a dynamic library, then this mutex would not provide sufficient
 * protection. In that case, it would be essential to define `H5_HAVE_THREADSAFE` in your build
 * system, so that the HDF5 library implements its own global mutex. We do not rely on that option
 * today because it is not supported for static linking on Windows.
 */
[[nodiscard]] MOCHI_HDF5_LOCAL std::recursive_mutex& GetGlobalMutex();

/**
 * @brief Handle an H5::Exception by logging an error and setting a mochi::Error result.
 *
 * @param[in] h5Exception The exception to handle
 * @param[in,out] error A description of the error.
 *
 * @note Use in your C++ catch block.
 */
MOCHI_HDF5_LOCAL void ReportException(H5::Exception const& h5Exception, Error& error);

/**
 * @brief Open an HDF5 file for writing with flags to preserve order of nested items.
 *
 * @param[in] filePath Path to the file.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 *
 * @note Creates the output directories if needed.
 * @note Replaces the file if it already exists.
 */
[[nodiscard]] MOCHI_HDF5_LOCAL H5::H5File OpenFileForWrite(std::string_view filePath, Error& error);

/**
 * @brief Open an HDF5 file for reading.
 *
 * @param[in] filePath Path to the file.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 */
[[nodiscard]] MOCHI_HDF5_LOCAL H5::H5File OpenFileForRead(std::string_view filePath, Error& error);

/**
 * @brief Open an HDF5 file for reading, where the file has already been loaded into memory.
 *
 * @param[in] fileData File data to read
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 */
[[nodiscard]] MOCHI_HDF5_LOCAL H5::H5File OpenFileBytesForRead(
    Span<char const> fileData,
    Error& error);

/**
 * @brief Create a group within a parent group.
 *
 * @param[in] parent The group within which a child group should be created.
 * @param[in] name Name of the group to create
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 */
[[nodiscard]] MOCHI_HDF5_LOCAL H5::Group
CreateGroup(H5::Group const& parent, char const* name, Error& error);

#endif // MOCHI_USE_HDF5

/**
 * @brief Return true if the given file data looks like an HDF5 file, by checking the header bytes.
 *
 * @param fileData[in] File data to check
 * @return True if the file data starts with an HDF5 header.
 */
[[nodiscard]] bool LooksLikeHDF5(Span<char const> fileData);

} // namespace mochi::hdf5
