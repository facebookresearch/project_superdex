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

// PLEASE DO NOT ADD OTHER INCLUDES HERE. This header is included in the mochi_physics public API.
#include <mochi_core/mochi_platform.h>
#include <mochi_core/utils/reflection.h>

namespace mochi {
// clang-format off
/**
 * @brief Verbosity levels for solvers and optimizers. Controls the amount of logging output during
 * solver execution.
 *
 * @note VerbosityLevel relates to the global @ref LogChannel system through the following pattern:
 *   if (params.verbosity >= VerbosityLevel::Error)   { MOCHI_LOG_ERROR(...);   } // Logs to @ref LogChannel::Error
 *   if (params.verbosity >= VerbosityLevel::Warning) { MOCHI_LOG_WARNING(...); } // Logs to @ref LogChannel::Warning
 *   if (params.verbosity >= VerbosityLevel::Verbose) { MOCHI_LOG(...);         } // Logs to @ref LogChannel::Info (NOT to @ref LogChannel::Verbose)
 */
// clang-format on
enum struct VerbosityLevel {
  Silent = 0, ///< No output.
  Error = 1, ///< Only errors.
  Warning = 2, ///< Errors and warnings.
  Verbose = 3, ///< Detailed output including iteration info.
  Count = 4 ///< Number of verbosity level enum values.
};
} // namespace mochi

MOCHI_ENUM_BEGIN(mochi::VerbosityLevel)
MOCHI_ENUM_ITEM(Silent)
MOCHI_ENUM_ITEM(Error)
MOCHI_ENUM_ITEM(Warning)
MOCHI_ENUM_ITEM(Verbose)
MOCHI_ENUM_COUNT(Count)
MOCHI_ENUM_END()
