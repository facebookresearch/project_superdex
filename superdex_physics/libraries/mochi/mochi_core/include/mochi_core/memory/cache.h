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

#include <mochi_core/mochi_platform.h>

#include <cstddef>

namespace mochi {

/**
 * @brief Data cache line size detection result.
 */
struct CacheLineInfo {
  /// Cache line size [bytes]
  size_t size = MOCHI_CONSERVATIVE_CACHE_LINE_SIZE;

  /// True if the actual size was detected. False if a conservative guess was returned instead.
  bool detected = false;
};

/**
 * @brief Return the current CPU's data cache line size, if possible.
 *
 * @details The result is detected once and cached for the lifetime of the process. On failure, @ref
 * MOCHI_CONSERVATIVE_CACHE_LINE_SIZE will be returned and @ref CacheLineInfo::detected will be
 * false.
 *
 * @return Detected cache line information, or the conservative fallback on failure.
 */
[[nodiscard]] CacheLineInfo GetCacheLineInfo();

} // namespace mochi
