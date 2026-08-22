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

// Reverse include for Intellisense
#include "profile.h"

#if MOCHI_PROFILE_ENABLE && MOCHI_USE_TRACY
#include <tracy/Tracy.hpp>
#endif

#include <algorithm>
#include <string_view>

namespace mochi {

/********************************************************************************
  Tracy Profiler Implementation
*/

#if MOCHI_PROFILE_ENABLE && MOCHI_USE_TRACY

#define IMPL_MOCHI_PROFILE_SCOPE() ZoneScoped

#define IMPL_MOCHI_PROFILE_N(nameStringLiteral) ZoneScopedN(nameStringLiteral)

// Arguments can be (name) or (name, len)
#define IMPL_MOCHI_PROFILE_LABEL(...) \
  if (ProfilerIsConnected()) {        \
    std::string_view sv{__VA_ARGS__}; \
    ZoneName(sv.data(), sv.size());   \
  }

// Concatenate the two strings on the stack (no allocation)
#define IMPL_MOCHI_PROFILE_LABEL_2(prefix, suffix)                                    \
  if (ProfilerIsConnected()) {                                                        \
    std::string_view prefix_sv{prefix};                                               \
    std::string_view suffix_sv{suffix};                                               \
    char tempBuffer[64]; /*limited for performance*/                                  \
    size_t prefixLen = std::min(sizeof(tempBuffer) - 1, prefix_sv.size());            \
    size_t totalLen = std::min(sizeof(tempBuffer) - 1, prefixLen + suffix_sv.size()); \
    if (prefixLen) {                                                                  \
      memcpy(tempBuffer, prefix_sv.data(), prefixLen);                                \
    }                                                                                 \
    if (totalLen > prefixLen) {                                                       \
      memcpy(tempBuffer + prefixLen, suffix_sv.data(), totalLen - prefixLen);         \
    }                                                                                 \
    tempBuffer[totalLen] = '\0';                                                      \
    ZoneName(tempBuffer, totalLen);                                                   \
  }

// Arguments can be (str) or (str, len)
#define IMPL_MOCHI_PROFILE_DESCRIPTION(...)                 \
  if (ProfilerIsConnected()) {                              \
    std::string_view description_sv{__VA_ARGS__};           \
    ZoneText(description_sv.data(), description_sv.size()); \
  }

#define IMPL_MOCHI_PROFILE_DESCRIPTION_F(fmt, ...)                                   \
  if (ProfilerIsConnected()) {                                                       \
    char buf[64]; /* size limited for performance */                                 \
    int const written = snprintf(buf, sizeof(buf), fmt, __VA_ARGS__);                \
    size_t const len =                                                               \
        (written > 0) ? std::min(static_cast<size_t>(written), sizeof(buf) - 1) : 0; \
    IMPL_MOCHI_PROFILE_DESCRIPTION(buf, len);                                        \
  }

#define IMPL_MOCHI_PROFILE_END_FRAME() FrameMark

inline void ProfilerInitialize() {
#if defined(TRACY_DELAYED_INIT) && defined(TRACY_MANUAL_LIFETIME)
  if (!ProfilerIsInitialized()) {
    tracy::StartupProfiler();
  }
#endif
}

inline void ProfilerShutdown() {
#if defined(TRACY_DELAYED_INIT) && defined(TRACY_MANUAL_LIFETIME)
  if (ProfilerIsInitialized()) {
    tracy::ShutdownProfiler();
  }
#endif
}

inline bool ProfilerIsInitialized() {
  return tracy::ProfilerAvailable();
}

inline bool ProfilerIsConnected() {
  return TracyIsConnected;
}

#endif // MOCHI_PROFILE_ENABLE && MOCHI_USE_TRACY

/********************************************************************************
  Stub Implementation
*/

#if !MOCHI_PROFILE_ENABLE
#define IMPL_MOCHI_PROFILE_SCOPE()
#define IMPL_MOCHI_PROFILE_N(nameStringLiteral)
#define IMPL_MOCHI_PROFILE_LABEL(name)
#define IMPL_MOCHI_PROFILE_LABEL_2(prefix, suffix)
#define IMPL_MOCHI_PROFILE_DESCRIPTION(str)
#define IMPL_MOCHI_PROFILE_DESCRIPTION_F(fmt, ...)
#define IMPL_MOCHI_PROFILE_END_FRAME()
inline void ProfilerInitialize() {}
inline void ProfilerShutdown() {}
inline bool ProfilerIsConnected() {
  return false;
}
inline bool ProfilerIsInitialized() {
  return false;
}
#endif // !MOCHI_PROFILE_ENABLE

} // namespace mochi
