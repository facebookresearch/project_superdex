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

#include <mochi_core/memory/cache.h>

#include <mochi_core/mochi_platform.h>
#include <mochi_core/utils/basic_utils.h>

#if MOCHI_PLATFORM_WINDOWS
#include <mochi_core/utils/dynamic_array.h>
#endif

#include <cstddef>

#if MOCHI_PLATFORM_MACOS
#include <sys/sysctl.h>
#elif MOCHI_PLATFORM_LINUX || MOCHI_PLATFORM_ANDROID
#include <unistd.h>
#elif MOCHI_PLATFORM_WINDOWS
#include <windows.h>
#endif

using namespace mochi;

static CacheLineInfo GetCacheLineInfoImpl() {
#if MOCHI_PLATFORM_MACOS
  size_t cacheLineSize = 0;
  size_t resultSize = sizeof(cacheLineSize);
  if (sysctlbyname("hw.cachelinesize", &cacheLineSize, &resultSize, nullptr, 0) == 0 &&
      resultSize == sizeof(cacheLineSize) && IsPowerOfTwo(cacheLineSize)) {
    return {cacheLineSize, true};
  }
#elif MOCHI_PLATFORM_LINUX || MOCHI_PLATFORM_ANDROID
#if defined(_SC_LEVEL1_DCACHE_LINESIZE)
  long const cacheLineSize = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
  if (IsPowerOfTwo(cacheLineSize)) {
    return {static_cast<size_t>(cacheLineSize), true};
  }
#endif
#elif MOCHI_PLATFORM_WINDOWS
  using ProcessorInfo = SYSTEM_LOGICAL_PROCESSOR_INFORMATION;

  DWORD bufferSize = 0;
  if (GetLogicalProcessorInformation(nullptr, &bufferSize) != FALSE ||
      GetLastError() != ERROR_INSUFFICIENT_BUFFER || bufferSize == 0 ||
      bufferSize % sizeof(ProcessorInfo) != 0) {
    return {};
  }

  DynamicArray<ProcessorInfo> buffer(bufferSize / sizeof(ProcessorInfo));
  int retryCount = 0;
  while (true) {
    DWORD const allocatedSize = bufferSize;
    if (GetLogicalProcessorInformation(buffer.data(), &bufferSize) != FALSE) {
      if (bufferSize > allocatedSize || bufferSize % sizeof(ProcessorInfo) != 0) {
        return {};
      }
      break;
    }

    DWORD const error = GetLastError();
    if (error != ERROR_INSUFFICIENT_BUFFER || bufferSize <= allocatedSize ||
        bufferSize % sizeof(ProcessorInfo) != 0) {
      return {};
    }

    ++retryCount;
    if (retryCount >= 3) {
      return {};
    }

    buffer.resize(bufferSize / sizeof(ProcessorInfo));
  }
  buffer.resize(bufferSize / sizeof(ProcessorInfo));

  for (auto const& info : buffer) {
    if (info.Relationship == RelationCache && info.Cache.Level == 1 &&
        (info.Cache.Type == CacheData || info.Cache.Type == CacheUnified) &&
        IsPowerOfTwo(info.Cache.LineSize)) {
      return {info.Cache.LineSize, true};
    }
  }
#endif

  return {};
}

CacheLineInfo mochi::GetCacheLineInfo() {
  static CacheLineInfo const s_info = GetCacheLineInfoImpl();
  return s_info;
}
